import abc
from enum import Enum
from types import SimpleNamespace
from typing import Optional

import cooper
import numpy as np
import torch

from src.utils.metrics.functional import cross_entropy, l2_loss, top1_accuracy


class PredictionType(Enum):
    REGRESSION = 0
    CLASSIFICATION = 1


LOSS_CALLABLES = {PredictionType.REGRESSION: l2_loss, PredictionType.CLASSIFICATION: cross_entropy}


def probability_to_cross_entropy_loss(pointwise_probability: float) -> float:
    """When using a cross-entropy loss, the following statements are equivalent:
    - imposing an upper-bound constraint of epsilon on the loss
    - imposing a lower-bound constraint of exp(-epsilon) on the probability of the *correct* class

    It is more intuitive/interpretable to think in terms of the requested probability of
    the model predicting the correct class, so we specify the probability as the task
    parameter and convert it to a loss here.
    """
    if pointwise_probability is None:
        return 0.0
    else:
        if not (0 <= pointwise_probability <= 1):
            raise ValueError("Pointwise probability must be between 0 and 1")
        return np.log(1 / pointwise_probability)


class BaseProblem(cooper.ConstrainedMinimizationProblem, abc.ABC):
    has_dual_variables: bool
    prediction_type: PredictionType

    @abc.abstractmethod
    def compute_cmp_state(self) -> cooper.CMPState:
        pass

    @abc.abstractmethod
    def pointwise_loss_level(self):
        pass

    @abc.abstractmethod
    def extract_multiplier_stats(self):
        pass

    def generate_logs_on_creation(self) -> Optional[dict]:
        return {"pointwise_loss_level": self.pointwise_loss_level()}

    def compute_excess_loss(self, per_sample_loss):
        return per_sample_loss - self.pointwise_loss_level()

    def loss_fn(self, *args, **kwargs):
        return LOSS_CALLABLES[self.prediction_type](*args, **kwargs)

    def get_batch_log_metrics(self, fwd_struct: SimpleNamespace):

        if self.prediction_type == PredictionType.REGRESSION:
            return dict(avg_loss=fwd_struct.avg_loss.detach(), max_loss=fwd_struct.per_sample_loss.max().detach())
        else:
            return dict(
                avg_loss=fwd_struct.avg_loss.detach(),
                avg_acc=fwd_struct.average_acc,
                max_loss=fwd_struct.per_sample_loss.max().detach(),
            )

    def forward_and_loss_helper(
        self,
        model: torch.nn.Module,
        inputs: torch.tensor,
        targets: torch.tensor,
        pointwise_loss_level: Optional[float],
        apply_loss_clamp: bool,
        apply_loss_square: bool,
        prediction_type: PredictionType,
    ) -> SimpleNamespace:

        logits = model(inputs)
        per_sample_loss = self.loss_fn(logits, targets, per_sample=True)

        if apply_loss_clamp:
            per_sample_loss = torch.relu(per_sample_loss - pointwise_loss_level)
        if apply_loss_square:
            per_sample_loss = per_sample_loss**2

        avg_loss = per_sample_loss.sum() / logits.shape[0]

        fwd_struct = SimpleNamespace(per_sample_loss=per_sample_loss, avg_loss=avg_loss)

        if prediction_type == PredictionType.CLASSIFICATION:
            per_sample_acc = top1_accuracy(logits, targets, per_sample=True)
            average_acc = per_sample_acc.sum() / logits.shape[0]
            fwd_struct.per_sample_acc = per_sample_acc
            fwd_struct.average_acc = average_acc

        return fwd_struct
