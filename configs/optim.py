import ml_collections as mlc

MLC_PH = mlc.config_dict.config_dict.placeholder


def _basic_config():
    _config = mlc.ConfigDict()

    _config.cooper_optimizer_class = MLC_PH(type)
    _config.primal_optimizer = mlc.ConfigDict()
    _config.dual_optimizer = mlc.ConfigDict()

    return _config


def _optimizer_module_config():
    _config = mlc.ConfigDict()

    _config.optimizer_class = MLC_PH(type)
    _config.kwargs = mlc.ConfigDict()
    _config.scheduler = mlc.ConfigDict({"scheduler_class": MLC_PH(type), "kwargs": MLC_PH(dict)})

    return _config
