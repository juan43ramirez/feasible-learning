import dotenv
import wandb

import shared
from src.datasets.utils import IndexedDataset

from .base_trainer import BaseTrainer

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()


logger = shared.fetch_main_logger(apply_basic_config=True)


class FeasibleLearningTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _train_one_epoch(self, train_data_iter) -> iter:
        while True:
            try:
                batch_data = next(train_data_iter)
            except StopIteration:
                return iter(self.dataloaders.train)

            inputs = batch_data[0].to(device=self.device, non_blocking=True)
            targets = batch_data[1].to(device=self.device, non_blocking=True)
            if isinstance(self.datasets.train, IndexedDataset):
                # If the training dataset is an IndexedDataset, the third element of the
                # batch_data tuple is the data_indices corresponding to the batch samples.
                data_indices = batch_data[2].to(device=self.device, non_blocking=True)
            else:
                data_indices = None

            compute_cmp_state_kwargs = dict(
                model=self.model, inputs=inputs, targets=targets, constraint_features=data_indices
            )
            roll_out = self.cooper_optimizer.roll(compute_cmp_state_kwargs=compute_cmp_state_kwargs)
            observed_multipliers = roll_out.primal_lagrangian_store.multiplier_values[self.cmp.feasibility_constraint]

            batch_metrics = {key: roll_out.cmp_state.misc[key] for key in self.config.metrics.batch}
            train_log_dict = self._format_logs_for_wandb(batch_metrics, prefix="batch/")
            wandb.log(train_log_dict, step=self.steps_taken)

            if self.cmp.has_dual_variables:
                max_batch_lambda = observed_multipliers.max().detach()
                wandb.log({"batch/max_lambda": max_batch_lambda}, step=self.steps_taken)

            if self.steps_taken % self.config.logging.print_train_stats_period_steps == 0:
                logger.info(
                    f"Step {self.steps_taken}/{self.num_steps} | Epoch {self.epoch} | Lagrangian: {roll_out.primal_lagrangian_store.lagrangian:.4f}"
                )

            self.steps_taken += 1
