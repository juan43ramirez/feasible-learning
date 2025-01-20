import copy
from functools import partial
from types import SimpleNamespace

import ml_collections as mlc

from src.utils.metrics import Accuracy, CrossEntropy, L2Loss, PerClassTop1Accuracy, Violation
from src.utils.metrics.meters import AverageMeter, StatsMeter

MLC_PH = mlc.config_dict.config_dict.placeholder

DEFAULT_AVG_KWARGS = {"meter_class": AverageMeter}
DEFAULT_STATS_KWARGS = {"meter_class": partial(StatsMeter, stats=["pos_avg", "nonpos_rate", "max"])}
AVG_STATS_KWARGS = {"meter_class": partial(StatsMeter, stats=["avg", "max"])}


def _basic_config():
    metrics_config = mlc.ConfigDict()
    metrics_config.batch = MLC_PH(list)
    metrics_config.epoch = mlc.ConfigDict()
    return metrics_config


def classification_config():
    metrics_config = _basic_config()
    metrics_config.batch = ["avg_acc", "avg_loss", "max_loss"]

    basic_classifcation_metrics = [
        SimpleNamespace(metric_class=Accuracy, log_name="avg_acc", kwargs=DEFAULT_AVG_KWARGS),
        SimpleNamespace(metric_class=PerClassTop1Accuracy, log_name="class_acc", kwargs=DEFAULT_AVG_KWARGS),
        SimpleNamespace(metric_class=CrossEntropy, log_name="loss", kwargs=AVG_STATS_KWARGS),
        SimpleNamespace(metric_class=Violation, log_name="violation", kwargs=DEFAULT_STATS_KWARGS),
    ]

    metrics_config.epoch.train = copy.deepcopy(basic_classifcation_metrics)
    metrics_config.epoch.val = copy.deepcopy(basic_classifcation_metrics)

    return metrics_config


def regression_config():
    metrics_config = _basic_config()
    metrics_config.batch = ["avg_loss", "max_loss"]

    metrics_config.epoch.train = [
        SimpleNamespace(metric_class=L2Loss, log_name="avg_loss", kwargs=AVG_STATS_KWARGS),
        SimpleNamespace(metric_class=Violation, log_name="violation", kwargs=DEFAULT_STATS_KWARGS),
    ]
    metrics_config.epoch.val = [
        SimpleNamespace(metric_class=L2Loss, log_name="avg_loss", kwargs=AVG_STATS_KWARGS),
        SimpleNamespace(metric_class=Violation, log_name="violation", kwargs=DEFAULT_STATS_KWARGS),
    ]
    return metrics_config


METRICS_CONFIGS = {
    "classification": classification_config,
    "regression": regression_config,
    None: _basic_config,
}


def get_config(config_string=None):
    return {"metrics": METRICS_CONFIGS[config_string]()}
