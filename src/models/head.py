import torch.nn as nn
from omegaconf import DictConfig


def build_classification_head(
    in_features: int,
    head_config: DictConfig,
) -> nn.Sequential:
    """Build a classification head with a hidden layer and dropout."""
    return nn.Sequential(
        nn.Linear(in_features, head_config.hidden_dim),
        nn.BatchNorm1d(head_config.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=head_config.dropout_rate),
        nn.Linear(head_config.hidden_dim, head_config.num_classes),
    )
