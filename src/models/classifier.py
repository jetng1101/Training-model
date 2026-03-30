import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.backbone import build_backbone
from src.models.head import build_classification_head


class Cifar100Classifier(nn.Module):
    """
    Image classifier composed of a pretrained backbone and a classification head.

    Supports any backbone available in the timm library.
    """

    def __init__(self, model_config: DictConfig) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(model_config.backbone)
        self.head = build_classification_head(feature_dim, model_config.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)
