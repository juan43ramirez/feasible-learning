import logging
import operator
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import dotenv
import pydash
import submitit
import torch
import wandb

import shared
from src import cmps, datasets, optim, utils

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()

logger = shared.fetch_main_logger(apply_basic_config=True)


class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config

        logger.info("Initialized trainer with configuration:")
        logger.info(config)
        logger.info(f"Current working directory is {os.getcwd()}")

    def __call__(self):
        self._make_reproducible()

        self.dist = utils.distributed.init_distributed()
        self.device = self.dist.device
        self.dtype = self.config.task.dtype

        # Update the trainer logger to include rank information for multi-GPU training
        self._update_logger()

        self.wandb_run, self.run_checkpoint_dir = self._create_wandb_logger()

        logger.info("Trainer called with config:")
        logger.info(self.config)

        self.datasets, self.dataset_metadata = self._create_datasets()
        self.dataloaders = self._create_dataloaders()
        self.num_samples = utils.extract_to_namespace(self.datasets, extract_fn=lambda dataset: len(dataset))
        self.num_batches = utils.extract_to_namespace(self.dataloaders, extract_fn=lambda loader: len(loader))

        self.model, self.model_without_ddp = self._create_model()

        self.num_steps = self._init_stopping_condition()
        self.eval_period_steps = self._init_evaluation_period()

        self.metrics = self._create_metrics()

        self.cmp, self.multiplier = self._create_cmp_and_multiplier()
        self.cooper_optimizer, self.schedulers = self._create_optimizers_and_schedulers()

        self.main()

        self._clean_finish()

    def _update_logger(self):
        shared.configure_logger(
            logger=shared.fetch_main_logger(),
            custom_format=f"(Rank:{self.dist.rank}/WS:{self.dist.world_size}) %(module)s:%(funcName)s:%(lineno)d | %(message)s ",
            level=getattr(logging, self.config.logging.log_level),
            show_path=self.config.logging.wandb_mode == "disabled",  # Only show path hyperlinks if not using wandb
        )

    def _make_reproducible(self):
        utils.set_seed(self.config.train.seed)
        if self.config.train.use_deterministic_ops:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_datasets(self):
        return datasets.build_datasets(self.config)

    def _create_dataloaders(self):
        if self.config.data.dataloader_kwargs.use_distributed_sampler and not self.dist.multi_gpu:
            raise ValueError("Distributed sampler requires multi-gpu training.")

        best_num_workers = None
        dataloaders = {}
        for split in self.config.data.dataloader_kwargs.split_kwargs:
            split_kwargs = pydash.omit(self.config.data.dataloader_kwargs, ["split_kwargs", "num_workers"])
            split_kwargs.update(getattr(self.config.data.dataloader_kwargs.split_kwargs, split, {}))

            dataloader, last_num_workers = datasets.build_dataloader(
                dataset=getattr(self.datasets, split),
                split=split,
                device=self.device,
                num_workers=getattr(self.config.data.dataloader_kwargs, "num_workers", best_num_workers),
                **split_kwargs,
            )
            dataloaders[split] = dataloader

            # Use the number of workers from the first dataloader for the rest -- this
            # avoids having to find the best number of workers for each split
            if self.config.data.dataloader_kwargs.num_workers is None:
                best_num_workers = last_num_workers if best_num_workers is None else best_num_workers
                with self.config.unlocked():
                    # Update num_workers in config for inspection on wandb
                    self.config.data.dataloader_kwargs.num_workers = best_num_workers
                    wandb.config.update(self.config.to_dict(), allow_val_change=True)

            logger.info(
                f"Initialized {split} dataloader with batch size {dataloader.batch_size} and {dataloaders[split].num_workers} workers"
            )

        return SimpleNamespace(**dataloaders)

    def _create_model(self):
        logger.info("Starting model creation")
        init_kwargs = pydash.omit(self.config.model.init_kwargs, "from_trainer")
        with utils.RNGContext(self.config.model.init_seed):
            kwargs_from_trainer = {}
            for key, value in self.config.model.init_kwargs.from_trainer.items():
                kwargs_from_trainer[key] = operator.attrgetter(value)(self)

            model = self.config.model.model_class(
                input_shape=self.datasets.train.input_shape,
                output_size=self.datasets.train.output_size,
                **init_kwargs,
                **kwargs_from_trainer,
            )

        model.to(device=self.device, dtype=self.dtype)
        model_without_ddp = model
        param_count = sum([torch.prod(torch.tensor(p.shape)).item() for p in model.parameters()])
        logger.info(f"Created model {self.config.model.model_class} with " + f"{param_count} parameters")

        return model, model_without_ddp

    def _create_cmp_and_multiplier(self):
        logger.info("Starting CMP and multiplier creation")
        cmp_kwargs = pydash.omit(self.config.task.cmp_kwargs, "from_trainer")
        # Extract attributes from trainer needed by the CMP constructor
        kwargs_from_trainer = {
            key: operator.attrgetter(value)(self) for key, value in self.config.task.cmp_kwargs.from_trainer.items()
        }

        if self.config.task.cmp_class.has_dual_variables:
            multiplier_kwargs = self.config.task.multiplier_kwargs
            if "init" in multiplier_kwargs:
                init = torch.tensor(multiplier_kwargs["init"])
                if len(init.shape) == 0:
                    init.unsqueeze_(0)
                multiplier_kwargs = pydash.omit(multiplier_kwargs, "init")
                multiplier_kwargs = {"device": self.device, "init": init, **multiplier_kwargs}
            else:
                multiplier_kwargs = {"device": self.device, **multiplier_kwargs}

            cmp_kwargs = {**cmp_kwargs, "multiplier_kwargs": multiplier_kwargs}

        if self.config.task.cmp_class.prediction_type == cmps.PredictionType.CLASSIFICATION:
            target_pointwise_loss = cmps.probability_to_cross_entropy_loss(self.config.task.pointwise_probability)
            logger.info(
                f"Provided pointwise probability {self.config.task.pointwise_probability} is equivalent to a target \n"
                f"a target pointwise loss of {target_pointwise_loss}"
            )
        elif self.config.task.cmp_class.prediction_type == cmps.PredictionType.REGRESSION:
            target_pointwise_loss = self.config.task.pointwise_loss
            logger.info(f"Using provided target pointwise loss of {target_pointwise_loss}")
        else:
            raise ValueError(f"Unknown CMP class {self.config.task.cmp_class}")

        logger.info(f"Building {self.config.task.cmp_class.__name__} with kwargs: {cmp_kwargs}")
        cmp = self.config.task.cmp_class(
            target_pointwise_loss=target_pointwise_loss, **cmp_kwargs, **kwargs_from_trainer
        )

        cmp_log_dict = cmp.generate_logs_on_creation()
        if cmp_log_dict is not None:
            wandb.log(cmp_log_dict, step=0)

        if cmp.has_dual_variables:
            multiplier = cmp.feasibility_constraint.multiplier.to(dtype=self.dtype)
        else:
            multiplier = None

        return cmp, multiplier

    def _create_optimizers_and_schedulers(self):
        return optim.build_cooper_optimizer_and_schedulers(
            model=self.model_without_ddp, cmp=self.cmp, config=self.config
        )

    def _save_checkpoint(self):
        if self.config.checkpointing.enabled:
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)

            checkpoint = {
                "model": self.model_without_ddp.state_dict(),
                "cooper_optimizer": self.cooper_optimizer.state_dict(),
                "cmp": self.cmp.state_dict(),
                "steps_taken": self.steps_taken,
                "epoch": self.epoch,
                "elapsed_time": (time.time() - self.start_time) + self.elapsed_time,
            }

            for scheduler_name, scheduler in utils.scan_namespace(self.schedulers):
                if scheduler is not None:
                    checkpoint[f"{scheduler_name}_scheduler"] = scheduler.state_dict()

            filename = os.path.join(self.run_checkpoint_dir, "checkpoint.pt")
            torch.save(checkpoint, filename)
            # wandb.save(filename, base_path=self.run_checkpoint_dir)
            logger.info(f"Saved checkpoint to {filename} (step={self.steps_taken}; epoch={self.epoch})")

    def _load_checkpoint(self):
        logger.info("Attempting to resume from checkpoint...")
        if os.path.isfile(os.path.join(self.run_checkpoint_dir, "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(self.run_checkpoint_dir, "checkpoint.pt"))

            self.model_without_ddp.load_state_dict(checkpoint["model"])
            self.model.load_state_dict(checkpoint["model"])

            self.cooper_optimizer.load_state_dict(checkpoint["cooper_optimizer"])
            self.cmp.load_state_dict(checkpoint["cmp"])

            self.steps_taken = checkpoint["steps_taken"]
            self.epoch = checkpoint["epoch"]
            self.elapsed_time = checkpoint["elapsed_time"]
            self.start_time = time.time()

            for scheduler_name, scheduler in utils.scan_namespace(self.schedulers):
                if scheduler is not None:
                    scheduler.load_state_dict(checkpoint[f"{scheduler_name}_scheduler"])

        else:
            raise ValueError("WandB run requested resuming but no checkpoint found.")

        logger.info("Successfully loaded checkpoint")

    def _create_metrics(self):
        _metrics = SimpleNamespace(batch=self.config.metrics.batch, epoch=SimpleNamespace())

        logger.info(f"Metrics logged for every batch: {_metrics.batch}")

        logger.info(f"Epoch-level metrics:")
        for split in self.config.metrics.epoch:
            metrics_for_split = {}
            for metric_config in getattr(self.config.metrics.epoch, split):
                kwargs = metric_config.kwargs
                if getattr(metric_config.metric_class, "needs_num_classes", False):
                    kwargs = kwargs | {"num_classes": self.dataset_metadata.num_classes}

                _metric = metric_config.metric_class(log_name=metric_config.log_name, **kwargs)
                metrics_for_split[metric_config.log_name] = _metric
                logger.info(f"\t Split: {split} \t Class: {metric_config.metric_class} \t Kwargs: {kwargs}")

            _metrics.epoch.__setattr__(split, metrics_for_split)
            logger.info(f"Instantiated {len(metrics_for_split)} metrics for {split} split")

        return _metrics

    def _create_wandb_logger(self):
        is_local_job = not ("SLURM_JOB_ID" in os.environ.keys() and self.config.resources.cluster == "slurm")

        # This is compatible with preemption since the SLURM_JOB_ID value is
        # preserved after preemption.
        custom_run_id = None if is_local_job else os.environ["SLURM_JOB_ID"]

        run = wandb.init(
            entity=os.environ["WANDB_ENTITY"],
            project=os.environ["WANDB_PROJECT"],
            dir=os.environ["WANDB_DIR"],
            id=custom_run_id,
            mode=self.config.logging.wandb_mode,
            resume="allow",
            tags=self.config.logging.wandb_tags,
        )
        logger.info(f"Initialized WandB run with id {run.id}")

        wandb.config.update(self.config.to_dict(), allow_val_change=True)

        # Define metrics for custom x-axis
        wandb.define_metric("batch/*", step_metric="_step")
        wandb.define_metric("_epoch")
        wandb.define_metric("val/*", step_metric="_epoch")
        wandb.define_metric("train/*", step_metric="_epoch")
        wandb.define_metric("lambda/*", step_metric="_epoch")

        run_subdir = run.id if is_local_job else os.environ["SLURM_JOB_ID"]
        run_checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"]) / run_subdir
        logger.info(f"Checkpoints will be saved to {run_checkpoint_dir}")

        return run, run_checkpoint_dir

    def _format_logs_for_wandb(self, metrics: dict[str, float], prefix: str = "train/"):
        wandb_dict = {prefix + k: v for k, v in metrics.items()}
        wandb_dict["_epoch"] = self.epoch
        wandb_dict["wall_sec"] = self.elapsed_time + (time.time() - self.start_time)
        wandb_dict["training_steps"] = self.steps_taken

        return wandb_dict

    def _init_stopping_condition(self):
        train_config = self.config.train
        if train_config.total_epochs is not None and train_config.total_steps is not None:
            raise ValueError("Train config contains both 'total_epochs' and 'total_steps'. Please specify only one")
        elif train_config.total_steps is not None:
            num_steps = train_config.total_steps
        elif train_config.total_epochs is not None:
            num_steps = self.num_batches.train * train_config.total_epochs
        else:
            raise ValueError("No stopping condition was specified.")

        num_epochs = num_steps / self.num_batches.train
        logger.info(f"Training loop was configured to run for {num_steps} steps ({num_epochs:.2f} epochs)")

        return num_steps

    def _init_evaluation_period(self):
        eval_period_steps = self.config.logging.eval_period_steps
        eval_period_epochs = self.config.logging.eval_period_epochs

        if eval_period_steps is not None and eval_period_epochs is not None:
            raise ValueError("Train config should specify exactly one of 'eval_period_steps' and 'eval_period_epochs'.")
        if eval_period_steps:
            _eval_period_steps = eval_period_steps
        elif eval_period_epochs:
            _eval_period_steps = self.num_batches.train * eval_period_epochs
        else:
            raise ValueError("No evaluation period was specified.")

        _eval_period_epochs = _eval_period_steps / self.num_batches.train
        logger.info(f"Evaluation happening every {_eval_period_steps} steps ({_eval_period_epochs: .2f} epochs)")

        return _eval_period_steps

    def _gather_log_metrics(self, metrics: dict[str, float], prefix: str = "train/"):
        wandb_dict = {prefix + k: v for k, v in metrics.items()}
        wandb_dict["_epoch"] = self.epoch
        wandb_dict["wall_sec"] = time.time() - self.start_time
        wandb_dict["training_steps"] = self.steps_taken

        return wandb_dict

    def gather_and_log_losses_on_split(self, split, all_losses):
        all_losses = all_losses.half()

        if self.config.logging.save_losses_and_multipliers:
            losses_folder = f"{self.run_checkpoint_dir}/losses/{split}"
            file_path = os.path.join(losses_folder, f"epoch_{self.epoch}.pt")
            os.makedirs(losses_folder, exist_ok=True)
            torch.save(all_losses, file_path)
            wandb.save(file_path, base_path=self.run_checkpoint_dir)
            logger.info(f"Saved {split} losses to {file_path}")

        loss_histogram = utils.wandb_utils.make_wandb_histogram(all_losses)
        log_dict = self._format_logs_for_wandb({"loss/histogram": loss_histogram}, prefix=f"{split}/")
        wandb.log(log_dict, step=self.steps_taken)

    def log_multiplier_stats(self):
        multiplier_stats = self.cmp.extract_multiplier_stats()
        if multiplier_stats is not None:
            all_multiplier_values = multiplier_stats.pop("all_multiplier_values")
            all_multiplier_values = all_multiplier_values.half()

            if self.config.logging.save_losses_and_multipliers:
                multiplier_folder = f"{self.run_checkpoint_dir}/multipliers"
                file_path = os.path.join(multiplier_folder, f"epoch_{self.epoch}.pt")
                os.makedirs(multiplier_folder, exist_ok=True)
                torch.save(all_multiplier_values, file_path)
                wandb.save(file_path, base_path=self.run_checkpoint_dir)
                logger.info(f"Saved multiplier values to {file_path}")

            # Log multiplier stats and histogram to wandb
            multiplier_histogram = utils.wandb_utils.make_wandb_histogram(all_multiplier_values)
            multiplier_stats = {f"lambda/{stat_name}": v for stat_name, v in multiplier_stats.items()}
            multiplier_stats["lambda/histogram"] = multiplier_histogram

            wandb_multiplier_stats = self._format_logs_for_wandb(multiplier_stats, prefix="")
            wandb.log(wandb_multiplier_stats, step=self.steps_taken)

    def main(self):
        if self.wandb_run.resumed or self.config.checkpointing.resume_from_checkpoint:
            # Retrieves self.{steps_taken, epoch, elapsed_time} and loads checkpointed
            # state_dicts for the model, optimizers and schedulers.
            self._load_checkpoint()
        else:
            self.steps_taken = 0
            self.epoch = 0
            self.elapsed_time = 0
            self.start_time = time.time()
            logger.info("No checkpoint found, starting from scratch.")

            self._save_checkpoint()

        steps_since_last_epoch = self.steps_taken % len(self.dataloaders.train)
        if self.config.data.dataloader_kwargs.use_distributed_sampler:
            self.dataloaders.train.sampler.set_epoch(self.epoch)

        # Skip the training dataloader ahead to the current step
        train_data_iter = iter(self.dataloaders.train)
        for dummy_step in range(steps_since_last_epoch):
            batch_data = next(train_data_iter)

        # After loading a checkpoint, and forwarding the dataloader and schedulers,
        # we are ready to train.
        self._train_loop(train_data_iter)

        self.elapsed_time = self.elapsed_time + (time.time() - self.start_time)
        logger.info(f"Completed {self.steps_taken} steps of training" + f" ({self.elapsed_time:.2f} seconds)")

        # Final eval after training if we didn't just do one
        if not (self.steps_taken % self.eval_period_steps == 0):
            logger.info("Final model evaluation")
            self._eval_loop()
            self._save_checkpoint()

        logger.info("Training completed")

    def _train_loop(self, train_data_iter):
        """
        We take the train_data_iter as an argument to allow for the possibility of
        resuming from a checkpoint and skipping ahead in the training dataloader.
        """

        logger.info(f"Evaluating model performance at epoch {self.epoch} (step {self.steps_taken})")
        self._eval_loop()
        logger.info(f"Evaluation loop completed after step {self.steps_taken}")

        logger.info("Starting training loop")
        while True:
            logger.info(f"Starting epoch {self.epoch}")
            self.model.train()
            train_data_iter = self._train_one_epoch(train_data_iter)

            logger.info(f"Finished epoch {self.epoch} after step {self.steps_taken}")
            self.epoch += 1

            for scheduler in pydash.arrays.compact([self.schedulers.primal, self.schedulers.dual]):
                logger.info(f"Stepping {scheduler} -- current LR: {scheduler.get_last_lr()}")
                scheduler.step()
                wandb.log({"lr": scheduler.get_last_lr()[0]}, step=self.steps_taken)

            if self.config.data.dataloader_kwargs.use_distributed_sampler:
                self.dataloaders.train.sampler.set_epoch(self.epoch)

            if self.steps_taken % self.eval_period_steps == 0:
                logger.info(f"Evaluating val. performance at start of epoch {self.epoch} (step {self.steps_taken})")
                self.model.eval()
                is_all_train_feasible = self._eval_loop()
                logger.info(f"Validation loop completed at start of epoch {self.epoch} (step {self.steps_taken})")

                if is_all_train_feasible and getattr(self.config.task, "early_stop_on_feasible", False):
                    logger.info("All training datapoints are feasible. Terminating training early.")
                    break

            self._save_checkpoint()

            if self._should_terminate():
                break

    @abstractmethod
    def _train_one_epoch(self, train_data_iter) -> iter:
        raise NotImplementedError

    @torch.inference_mode()
    def process_batch_for_evaluation(self, batch_data, split_metrics):
        """
        Computes a forward and evaluates the loss and other metrics on the batch.
        We automatically populate the args needed by the metrics.
        """
        inputs = batch_data[0].to(device=self.device, non_blocking=True)
        targets = batch_data[1].to(device=self.device, non_blocking=True)
        predictions = self.model(inputs)
        per_sample_loss = self.cmp.loss_fn(predictions, targets, per_sample=True)

        known_args = {
            "targets": targets,
            "predictions": predictions,
            "per_sample_loss": per_sample_loss,
            "cmp": self.cmp,
        }

        for metric in split_metrics.values():
            kwargs_for_metric = {key: known_args[key] for key in metric.forward_args}
            kwargs_for_metric["return_value"] = False
            # This call computes the metric values and updates the meters internally
            metric(**kwargs_for_metric)

        return per_sample_loss, known_args

    @torch.inference_mode()
    def _eval_loop(self):

        logger.info(f"Initiating validation loop on rank {self.dist.rank}")
        self.model.eval()

        save_eval_logs_period = self.config.logging.save_eval_logs_every_n_epochs * self.eval_period_steps
        should_save_tensors = (self.steps_taken % save_eval_logs_period == 0) or (self.steps_taken == self.num_steps)
        if should_save_tensors and self.cmp.has_dual_variables:
            self.log_multiplier_stats()

        is_all_train_feasible = False

        for split, split_metrics in utils.scan_namespace(self.metrics.epoch):

            logger.info(f"Computing metrics for {split} split")

            split_meters = {}
            for metric in split_metrics.values():
                metric.reset_meters()
                split_meters[metric] = metric.meters

            if len(split_meters) == 0:
                continue

            # Evaluate batch-dependent metrics and update meters
            is_dataset_indexed = isinstance(getattr(self.datasets, split), datasets.utils.IndexedDataset)
            all_losses = torch.zeros(getattr(self.num_samples, split), device=self.device)

            for batch_data in getattr(self.dataloaders, split):
                per_sample_loss, known_args = self.process_batch_for_evaluation(batch_data, split_metrics)
                if is_dataset_indexed:
                    indices = batch_data[2].to(device=self.device, non_blocking=True)
                    all_losses[indices] = per_sample_loss

            logger.info(f"Aggregating {split} metrics on rank {self.dist.rank}")

            val_log_dict = {}
            for metric in split_meters.keys():
                logger.info(f"Aggregating values for metric {metric}")
                for metric_known_return, meter in metric.meters.items():
                    # This triggers the sync across processes at the meter level
                    meter_values = meter.get_result_dict()

                    if len(meter_values) == 1 and "avg" in meter_values:
                        create_key_fn = lambda metric_kr, meter_kr: f"{metric_kr}"
                    else:
                        create_key_fn = lambda metric_kr, meter_kr: f"{metric_kr}/{meter_kr}"

                    for meter_known_return in meter_values.keys():
                        key = create_key_fn(metric_known_return, meter_known_return)
                        if key in val_log_dict:
                            raise ValueError(f"Duplicate key {key} in {split} metric {metric}")
                        val_log_dict[key] = meter_values[meter_known_return]

            logger.info(f"{split} metrics at epoch {self.epoch} (step {self.steps_taken}):")
            for key in val_log_dict.keys():
                logger.info(f"\t~ {key}: {val_log_dict[ key] :.4f}")
            val_log_dict = self._format_logs_for_wandb(val_log_dict, prefix=f"{split}/")
            wandb.log(val_log_dict, step=self.steps_taken)

            # Log individual losses, class losses, and loss histograms
            if is_dataset_indexed and should_save_tensors:
                self.gather_and_log_losses_on_split(split, all_losses)

            # Check if all the training samples are feasible
            if split == "train" and "violation" in split_metrics:
                violation_meter = split_metrics["violation"].meters["violation"]
                if "max" in violation_meter.known_returns:
                    max_train_violation = violation_meter.get_result_dict()["max"]
                if max_train_violation <= 0:
                    logger.info(f"All training samples are feasible at step {self.steps_taken}!!!")
                    is_all_train_feasible = True

            logger.info(f"Finished measuring {split} metrics")

        logger.info(f"Finished measuring evaluation metrics")

        return is_all_train_feasible

    def _should_terminate(self):
        # NOTE: this method is called at the end of each epoch in _train_loop to stop
        # the training. By default, termination happens after a certain number of steps
        # has been reached.
        # To ensure correct stopping, this is an abstract method that should be
        # implemented by subclasses. If want to use the default behavior, simply
        # call `super()._should_terminate()` in the subclass implementation.
        if self.steps_taken >= self.num_steps:
            logger.info("Stopping training: reached maximum number of steps!")
            return True

        return False

    def __submitit_checkpoint__(self):
        """Function used by submitit when SLURM job is preempted"""

        if self.config.checkpointing.enabled:
            self._save_checkpoint()

            resume_config = self.config.copy()
            with resume_config.unlocked():
                resume_config.checkpointing.resume_from_checkpoint = True
            resume_trainer = self.__class__(resume_config)
            return submitit.helpers.DelayedSubmission(resume_trainer)
        else:
            logger.info("Checkpointing is disabled. NOT saving checkpoint and NOT resuming after preemption.")

    def _clean_finish(self):
        logger.info("Attempting to close WandB logger")
        wandb.finish()
        logger.info("Shutting down gracefully")
