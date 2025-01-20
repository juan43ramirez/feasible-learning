import math
from types import SimpleNamespace

import cooper
import torch

import shared

from .base_cmp import BaseProblem, PredictionType

logger = shared.fetch_main_logger()


class ERMProblem(BaseProblem):
    has_dual_variables = False
    apply_loss_clamp = False
    apply_loss_square = False

    def __init__(self, target_pointwise_loss: float):
        self.pointwise_loss_clamp = target_pointwise_loss
        logger.info(f"{self.__class__.__name__} will clamp pointwise losses below {self.pointwise_loss_clamp}")

        super().__init__()

    def pointwise_loss_level(self):
        return self.pointwise_loss_clamp

    def extract_multiplier_stats(self):
        return None

    def compute_cmp_state(self, model, inputs, targets, constraint_features=None) -> cooper.CMPState:
        fwd_struct = self.forward_and_loss_helper(
            model,
            inputs,
            targets,
            self.pointwise_loss_clamp,
            apply_loss_clamp=self.apply_loss_clamp,
            apply_loss_square=self.apply_loss_square,
            prediction_type=self.prediction_type,
        )

        return cooper.CMPState(
            loss=fwd_struct.avg_loss, observed_constraints={}, misc=self.get_batch_log_metrics(fwd_struct)
        )


class ERMClassificationProblem(ERMProblem):
    prediction_type = PredictionType.CLASSIFICATION


class ERMRegressionProblem(ERMProblem):
    prediction_type = PredictionType.REGRESSION


class CSERMClassificationProblem(ERMProblem):
    """Implements Clamped-Squared ERM Classification"""

    prediction_type = PredictionType.CLASSIFICATION
    apply_loss_clamp = True
    apply_loss_square = True


class CSERMRegressionProblem(ERMProblem):
    """Implements Clamped-Squared ERM Regression"""

    prediction_type = PredictionType.REGRESSION
    apply_loss_clamp = True
    apply_loss_square = True
