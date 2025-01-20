import abc
import inspect
from typing import Iterable, Type

import cooper
import torch

from . import meters


class Metric(abc.ABC):
    """Abstract class specifying the Metric interface"""

    def __init__(self, log_name: str, meter_class: Type[meters.Meter] = meters.AverageMeter):
        self.known_returns: list[str]
        self.log_name = log_name
        self.meters = {key: meter_class() for key in self.known_returns}

    def reset_meters(self):
        for meter in self.meters.values():
            meter.reset()

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> dict:
        pass

    @property
    def forward_args(cls):
        fwd_args = inspect.getfullargspec(cls.forward).args
        fwd_args.remove("self")
        return fwd_args

    def __call__(self, *args, return_value=True, **kwargs):
        metric_return = self.forward(*args, **kwargs)

        # Ensure that only valid keys are returned
        for key in metric_return.keys():
            if key not in self.known_returns:
                raise ValueError(f"Unknown return key {key} for metric {self.__class__.__name__}")

        if return_value:
            return metric_return

    def get_detached_items(self, result):
        extracted_result = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                extracted_result[k] = v.detach()
            else:
                extracted_result[k] = v
        return extracted_result


class L2Loss(Metric):
    known_returns = ["l2_loss"]

    def forward(self, per_sample_loss: torch.Tensor) -> dict:
        meter = self.meters["l2_loss"]
        if isinstance(meter, meters.AverageMeter):
            _val = per_sample_loss.mean()
            meter.update(val=_val, n=per_sample_loss.shape[0])
        elif isinstance(meter, meters.StatsMeter):
            _val = per_sample_loss
            meter.update(val=_val)
        else:
            raise NotImplementedError(f"Unknown meter of type {type(meter)}")

        return {"l2_loss": _val}


class CrossEntropy(Metric):
    known_returns = ["ce_loss"]

    def forward(self, per_sample_loss: torch.Tensor) -> dict:
        meter = self.meters["ce_loss"]
        if isinstance(meter, meters.AverageMeter):
            _val = per_sample_loss.mean()
            meter.update(val=_val, n=per_sample_loss.shape[0])
        elif isinstance(meter, meters.StatsMeter):
            _val = per_sample_loss
            meter.update(val=_val)
        else:
            raise NotImplementedError(f"Unknown meter of type {type(meter)}")

        return {"ce_loss": _val}


class Accuracy(Metric):
    def __init__(
        self, which_k: Iterable[int] = (1,), meter_class: Type[meters.Meter] = meters.AverageMeter, *args, **kwargs
    ):
        assert len(which_k) >= 1
        for val in which_k:
            assert val > 0
        self.which_k = which_k
        self.max_k: int = max(which_k)

        self.known_returns = [f"acc@{k}" for k in which_k]
        if meter_class != meters.AverageMeter:
            raise NotImplementedError("Only `AverageMeter` is supported for the Accuracy metric")

        super().__init__(meter_class=meter_class, *args, **kwargs)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        with torch.no_grad():
            batch_size = targets.size(0)
            _, pred = predictions.topk(self.max_k, 1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            result = {}

            for k in self.which_k:
                known_return = f"acc@{k}"
                correct_k = correct[:k].reshape(-1).float().sum(0)
                acc_at_k = correct_k.mul_(1 / batch_size)
                result[known_return] = acc_at_k
                self.meters[known_return].update(val=acc_at_k, n=batch_size)

            return result


class PerClassTop1Accuracy(Metric):
    needs_num_classes = True

    def __init__(self, num_classes=10, *args, **kwargs):
        self.num_classes = num_classes
        self.known_returns = [f"class_acc/{class_id}" for class_id in range(num_classes)]
        super().__init__(*args, **kwargs)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        with torch.no_grad():
            predicted_class = torch.argmax(predictions, dim=1)
            is_sample_correct = (predicted_class == targets).float()

            result = {}
            for class_id in range(self.num_classes):
                known_return = f"class_acc/{class_id}"
                class_filter = targets == class_id
                samples_in_class = class_filter.sum()
                if samples_in_class > 0:
                    class_acc = is_sample_correct[class_filter].sum() / samples_in_class
                    result[known_return] = class_acc
                    self.meters[known_return].update(val=class_acc, n=samples_in_class)

            return result


class Violation(Metric):
    known_returns = ["violation"]

    def forward(self, per_sample_loss: torch.Tensor, cmp: cooper.ConstrainedMinimizationProblem) -> dict:
        violation = cmp.compute_excess_loss(per_sample_loss)

        meter = self.meters["violation"]
        if isinstance(meter, meters.AverageMeter):
            _val = violation.mean()
            meter.update(val=_val, n=per_sample_loss.shape[0])
        elif isinstance(meter, meters.StatsMeter):
            _val = violation
            meter.update(val=_val)
        else:
            raise NotImplementedError(f"Unknown meter of type {type(meter)}")

        return {"violation": _val}
