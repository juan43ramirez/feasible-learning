import ml_collections as mlc
import torch

import shared
from configs.optim import _optimizer_module_config
from src import models

MLC_PH = mlc.config_dict.config_dict.placeholder


def _basic_config():
    _config = mlc.ConfigDict()

    _config.model_class = MLC_PH(type)
    _config.init_seed = MLC_PH(int)

    _config.init_kwargs = mlc.ConfigDict()
    _config.init_kwargs.from_trainer = {}

    return _config


def TwoDim_MLP_config():
    model_config = _basic_config()
    model_config.model_class = models.MLP
    model_config.init_seed = 135

    model_config.init_kwargs.activation_type = torch.nn.ReLU
    model_config.init_kwargs.hidden_sizes = (70, 70)

    optim_config = mlc.ConfigDict({"primal_optimizer": _optimizer_module_config()})
    optim_config.primal_optimizer.optimizer_class = torch.optim.AdamW
    optim_config.primal_optimizer.kwargs["lr"] = 5e-4

    return {"model": model_config, "optim": optim_config}


def CIFAR_ResNet18_SGD_config():
    model_config = _basic_config()
    model_config.model_class = models.ResNet18CIFAR
    model_config.init_kwargs.weights = None
    model_config.init_seed = 0

    # Optimizer recipe from https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/main.py
    optim_config = mlc.ConfigDict({"primal_optimizer": _optimizer_module_config()})
    optim_config.primal_optimizer.optimizer_class = torch.optim.SGD
    optim_config.primal_optimizer.kwargs["lr"] = 0.1
    optim_config.primal_optimizer.kwargs["momentum"] = 0.9
    optim_config.primal_optimizer.kwargs["weight_decay"] = 5e-4
    optim_config.primal_optimizer.scheduler.scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    optim_config.primal_optimizer.scheduler.kwargs = {"T_max": 200}

    return {"model": model_config, "optim": optim_config}


def UTKFace_ResNet18_config():
    model_config = _basic_config()
    model_config.model_class = models.ResNet18UTK
    model_config.init_kwargs.weights = None
    model_config.init_seed = 0

    optim_config = mlc.ConfigDict({"primal_optimizer": _optimizer_module_config()})
    optim_config.primal_optimizer.optimizer_class = torch.optim.AdamW
    optim_config.primal_optimizer.kwargs["lr"] = 1e-4

    return {"model": model_config, "optim": optim_config}


MODEL_CONFIGS = {
    None: _basic_config,
    "TwoDim_MLP": TwoDim_MLP_config,
    "CIFAR_ResNet18_SGD": CIFAR_ResNet18_SGD_config,
    "UTKFace_ResNet18": UTKFace_ResNet18_config,
}


def get_config(config_string=None):
    return shared.default_get_config(
        config_group_name=None, pop_key="model_type", preset_configs=MODEL_CONFIGS, cli_cmd=config_string
    )
