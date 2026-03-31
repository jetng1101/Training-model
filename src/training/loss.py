import torch.nn as nn
from omegaconf import DictConfig


def build_criterion(loss_config: DictConfig) -> nn.CrossEntropyLoss:
    """Build cross-entropy loss with optional label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=loss_config.label_smoothing)
