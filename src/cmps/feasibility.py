import abc

import cooper
import pydash
import torch

import shared

from .base_cmp import BaseProblem, PredictionType

logger = shared.fetch_main_logger()


class FeasibilityProblem(BaseProblem):
    has_dual_variables = True
    apply_loss_clamp = False
    apply_loss_square = False
    constraint_type = cooper.ConstraintType.INEQUALITY

    def __init__(self, target_pointwise_loss: float, num_constraints: int, multiplier_kwargs: dict = {}):
        super().__init__()

        self.target_pointwise_loss = target_pointwise_loss

        multiplier_init = multiplier_kwargs.get("init", 0.0) + torch.zeros((num_constraints,))
        multiplier = cooper.multipliers.IndexedMultiplier(
            constraint_type=self.constraint_type, init=multiplier_init, **pydash.omit(multiplier_kwargs, "init")
        )
        self.feasibility_constraint = cooper.Constraint(
            constraint_type=self.constraint_type, formulation_type=cooper.LagrangianFormulation, multiplier=multiplier
        )

    def pointwise_loss_level(self):
        return self.target_pointwise_loss

    def evaluate_multipliers(self, constraint_features=None):
        multiplier = self.feasibility_constraint.multiplier
        return multiplier() if constraint_features is None else multiplier(constraint_features)

    def extract_multiplier_stats(self):
        if not isinstance(self.feasibility_constraint.multiplier, cooper.multipliers.ExplicitMultiplier):
            raise NotImplementedError("This function is only intended to be used with `ExplicitMultiplier`s")

        all_multiplier_values = self.feasibility_constraint.multiplier.weight.data.detach()
        multiplier_stats = {
            "max": all_multiplier_values.max(),
            "avg": all_multiplier_values.mean(),
            "median": all_multiplier_values.median(),
            "rate_zeros": (all_multiplier_values == 0).float().mean(),
            "all_multiplier_values": all_multiplier_values,
        }

        return multiplier_stats

    def compute_cmp_state(self, model, inputs, targets, constraint_features) -> cooper.CMPState:
        fwd_struct = self.forward_and_loss_helper(
            model,
            inputs,
            targets,
            self.target_pointwise_loss,
            apply_loss_clamp=self.apply_loss_clamp,
            apply_loss_square=self.apply_loss_square,
            prediction_type=self.prediction_type,
        )
        constraint_state = cooper.ConstraintState(
            violation=self.compute_excess_loss(fwd_struct.per_sample_loss),
            strict_violation=self.compute_strict_violation(fwd_struct),
            constraint_features=constraint_features,
        )

        return cooper.CMPState(
            loss=None,
            observed_constraints={self.feasibility_constraint: constraint_state},
            misc=self.get_batch_log_metrics(fwd_struct),
        )

    @abc.abstractmethod
    def compute_strict_violation(self, fwd_struct):
        pass


class FeasibleClassificationProblem(FeasibilityProblem):
    prediction_type = PredictionType.CLASSIFICATION

    def __init__(self, use_strict_accuracy: bool, *args, **kwargs):
        self.use_strict_accuracy = use_strict_accuracy
        super().__init__(*args, **kwargs)

    def compute_strict_violation(self, fwd_struct):
        if self.use_strict_accuracy:
            # This is a "greater-than" constraint: accuracy >= 1.0
            # So in "less-than" convention, we have - accuracy + 1.0 <= 0
            # Use 0.5 as the threshold for strict accuracy
            return -fwd_struct.per_sample_acc + 0.5
        else:
            return None


class FeasibleRegressionProblem(FeasibilityProblem):
    prediction_type = PredictionType.REGRESSION

    def compute_strict_violation(self, fwd_struct):
        return None

    def get_batch_log_metrics(self, fwd_struct):
        return dict(avg_loss=fwd_struct.avg_loss.detach(), max_loss=fwd_struct.per_sample_loss.max().detach())
