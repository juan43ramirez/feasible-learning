import torch


def l2_loss(prediction: torch.Tensor, target: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
    per_sample_sq_loss = torch.linalg.vector_norm(prediction - target, ord=2, dim=1) ** 2
    return per_sample_sq_loss if per_sample else torch.mean(per_sample_sq_loss)


def cross_entropy(prediction: torch.Tensor, target: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
    reduction = "none" if per_sample else "mean"
    return torch.nn.functional.cross_entropy(prediction, target, reduction=reduction)


def top1_accuracy(prediction: torch.Tensor, target: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
    predicted_labels = torch.argmax(prediction, dim=1)
    per_sample_correct = (predicted_labels == target).float()
    return per_sample_correct if per_sample else per_sample_correct.sum() / prediction.shape[0]
