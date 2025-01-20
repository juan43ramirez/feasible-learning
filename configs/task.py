import cooper
import ml_collections as mlc
import torch

MLC_PH = mlc.config_dict.config_dict.placeholder

import shared
from configs.optim import _optimizer_module_config
from src.cmps import (
    CSERMClassificationProblem,
    CSERMRegressionProblem,
    ERMClassificationProblem,
    ERMRegressionProblem,
    FeasibleClassificationProblem,
    FeasibleRegressionProblem,
)
from src.trainers import ERMTrainer, FeasibleLearningTrainer


def _basic_config():
    _config = mlc.ConfigDict()

    _config.trainer_class = MLC_PH(type)
    _config.dtype = torch.float32

    _config.cmp_class = MLC_PH(type)
    _config.cmp_kwargs = mlc.ConfigDict()
    _config.cmp_kwargs.from_trainer = {}

    _config.multiplier_kwargs = mlc.ConfigDict()

    # Used to determine whether the dataset needs to be wrapped in an IndexedDataset for
    # extracting the right constraint and multiplier indices
    _config.use_indexed_dataset = True

    return _config


def feasible_classification_task_config():
    task_config = _basic_config()

    task_config.trainer_class = FeasibleLearningTrainer

    task_config.pointwise_probability = MLC_PH(float)
    task_config.pointwise_loss = MLC_PH(float)
    task_config.early_stop_on_feasible = False

    task_config.cmp_class = FeasibleClassificationProblem
    task_config.cmp_kwargs.use_strict_accuracy = False
    task_config.cmp_kwargs.from_trainer = {"num_constraints": "num_samples.train"}

    task_config.multiplier_kwargs.init = 0.0

    optim_config = mlc.ConfigDict()
    optim_config.cooper_optimizer_class = cooper.optim.AlternatingDualPrimalOptimizer

    optim_config.dual_optimizer = _optimizer_module_config()
    optim_config.dual_optimizer.optimizer_class = cooper.optim.nuPI
    optim_config.dual_optimizer.kwargs["lr"] = MLC_PH(float)

    return {"task": task_config, "optim": optim_config}


def feasible_regression_task_config():
    task_config = _basic_config()

    task_config.trainer_class = FeasibleLearningTrainer

    task_config.pointwise_loss = MLC_PH(float)
    task_config.early_stop_on_feasible = False

    task_config.cmp_class = FeasibleRegressionProblem
    task_config.cmp_kwargs.from_trainer = {"num_constraints": "num_samples.train"}

    task_config.multiplier_kwargs.init = 0.0

    optim_config = mlc.ConfigDict()
    optim_config.cooper_optimizer_class = cooper.optim.AlternatingDualPrimalOptimizer

    optim_config.dual_optimizer = _optimizer_module_config()
    optim_config.dual_optimizer.optimizer_class = cooper.optim.nuPI
    optim_config.dual_optimizer.kwargs["lr"] = 1.0

    return {"task": task_config, "optim": optim_config}


def erm_classification_task_config():
    task_config = _basic_config()

    task_config.trainer_class = ERMTrainer

    task_config.pointwise_probability = 1.0
    task_config.pointwise_loss = 0.0

    task_config.cmp_class = ERMClassificationProblem

    optim_config = mlc.ConfigDict()
    optim_config.cooper_optimizer_class = cooper.optim.UnconstrainedOptimizer

    return {"task": task_config, "optim": optim_config}


def erm_regression_task_config():
    task_config = _basic_config()

    task_config.trainer_class = ERMTrainer

    task_config.pointwise_loss = 0.0

    task_config.cmp_class = ERMRegressionProblem

    optim_config = mlc.ConfigDict()
    optim_config.cooper_optimizer_class = cooper.optim.UnconstrainedOptimizer

    return {"task": task_config, "optim": optim_config}


def cserm_classification_task_config():
    task_config = _basic_config()

    task_config.trainer_class = ERMTrainer

    task_config.pointwise_probability = MLC_PH(float)
    task_config.pointwise_loss = MLC_PH(float)

    task_config.cmp_class = CSERMClassificationProblem

    optim_config = mlc.ConfigDict()
    optim_config.cooper_optimizer_class = cooper.optim.UnconstrainedOptimizer

    return {"task": task_config, "optim": optim_config}


def cserm_regression_task_config():
    task_config = _basic_config()

    task_config.trainer_class = ERMTrainer

    task_config.pointwise_probability = MLC_PH(float)
    task_config.pointwise_loss = MLC_PH(float)

    task_config.cmp_class = CSERMRegressionProblem

    optim_config = mlc.ConfigDict()
    optim_config.cooper_optimizer_class = cooper.optim.UnconstrainedOptimizer

    return {"task": task_config, "optim": optim_config}


TASK_CONFIGS = {
    None: _basic_config,
    "feasible_classification": feasible_classification_task_config,
    "feasible_regression": feasible_regression_task_config,
    "erm_classification": erm_classification_task_config,
    "erm_regression": erm_regression_task_config,
    "cserm_classification": cserm_classification_task_config,
    "cserm_regression": cserm_regression_task_config,
}


def get_config(config_string=None):
    """Examples for config_string:
    - "task_type=feasible_classification task.pointwise_probability=0.9"
    - "task_type=erm_classification"
    - "task_type=cserm_classification pointwise_probability=0.9"
    """
    return shared.default_get_config(
        config_group_name=None, pop_key="task_type", preset_configs=TASK_CONFIGS, cli_cmd=config_string
    )
