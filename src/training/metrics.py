from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EpochMetrics:
    loss: float
    top1_accuracy: float
    top5_accuracy: float


def compute_topk_accuracy(
    logits: Tensor,
    targets: Tensor,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[int, float]:
    """Compute top-k accuracy for each k value."""
    with torch.no_grad():
        batch_size = targets.size(0)
        max_k = max(top_k)
        _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
        predictions = predictions.t()
        correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

        return {
            k: correct[:k].reshape(-1).float().sum().item() / batch_size * 100.0
            for k in top_k
        }
