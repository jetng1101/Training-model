import torch.nn as nn


def build_criterion() -> nn.CrossEntropyLoss:
    """Build the standard cross-entropy loss for classification."""
    return nn.CrossEntropyLoss()
