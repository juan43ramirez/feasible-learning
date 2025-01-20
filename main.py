import logging
import os
import sys
from datetime import datetime

import cooper
import dotenv
import submitit
import torch
from absl import app
from absl.flags import FLAGS
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags as MLC_FLAGS

import shared

# Initialize "FLAGS.config" object with a placeholder config
MLC_FLAGS.DEFINE_config_file("config", default="configs/core.py")
MLC_FLAGS.DEFINE_config_file("model_config", default="configs/model.py")
MLC_FLAGS.DEFINE_config_file("data_config", default="configs/data.py")
MLC_FLAGS.DEFINE_config_file("task_config", default="configs/task.py")
MLC_FLAGS.DEFINE_config_file("metrics_config", default="configs/metrics.py")
MLC_FLAGS.DEFINE_config_file("resources_config", default="configs/resources.py")
logger = logging.getLogger()
# Configure root logger to use rich logging
shared.configure_logger(logger, level=logging.INFO)

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()


def inject_file_configs(config, injected_config_fields):
    for field_name in injected_config_fields:
        injected_config = getattr(FLAGS, f"{field_name}_config", {})

        if len(injected_config.keys()) > 0:

            logger.info(f"Using {field_name} config from `{field_name}_config` provided through CLI")
            for key_from_root, value in injected_config.items():
                with config.unlocked():
                    if config[key_from_root] is None:
                        # Need to split the key by "." and traverse the config to set the new value
                        split_key = key_from_root.split(".")
                        entry_in_config = config
                        for subkey in split_key[:-1]:
                            entry_in_config = entry_in_config[subkey]
                        entry_in_config[split_key[-1]] = value
                    elif isinstance(config[key_from_root], (dict, ConfigDict)):
                        config[key_from_root].update(value)
                    else:
                        raise RuntimeError(f"Cannot inject `{field_name}_config` into `{key_from_root}`")

    return config


def main(_):
    logger.info(f"Using Python version {sys.version}")
    logger.info(f"Using PyTorch version {torch.__version__}")
    logger.info(f"Using Cooper version {cooper.__version__}")

    injected_config_fields = ["model", "data", "task", "optim", "metrics", "resources"]
    config = inject_file_configs(FLAGS.config, injected_config_fields)

    logger.info("Instantiating Trainer from config")
    trainer = config.task.trainer_class(config)

    logger.info(f"Current working directory: {os.getcwd()}")

    # Get a directory name with current date and time
    job_submitit_dir = os.path.join(os.environ["SUBMITIT_DIR"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Queue a the job. If `cluster` is not SLURM, the slurm-exclusive parameters will be ignored.
    executor = submitit.AutoExecutor(cluster=config.resources.cluster, folder=job_submitit_dir)
    executor.update_parameters(
        name=config.exp_name,
        slurm_partition=config.resources.partition,
        timeout_min=config.resources.timeout_min,
        nodes=config.resources.nodes,
        slurm_ntasks_per_node=config.resources.tasks_per_node,
        cpus_per_task=config.resources.cpus_per_task,
        slurm_gpus_per_task=config.resources.gpus_per_task,
        slurm_mem=config.resources.mem,
    )

    job = executor.submit(trainer)
    logger.info(f"Submitted experiment with jobid {job.job_id}")

    if config.resources.cluster == "debug":
        if config.allow_silent_failure:
            job.exception()
        else:
            job.result()


if __name__ == "__main__":
    app.run(main)
