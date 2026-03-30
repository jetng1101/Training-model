import timm
import torch.nn as nn
from omegaconf import DictConfig


def build_backbone(backbone_config: DictConfig) -> tuple[nn.Module, int]:
    """
    Build a pretrained backbone from timm.

    Returns:
        Tuple of (backbone module, output feature dimension).
    """
    model = timm.create_model(
        backbone_config.name,
        pretrained=backbone_config.pretrained,
        num_classes=0,
        drop_rate=backbone_config.drop_rate,
    )
    feature_dim: int = model.num_features
    return model, feature_dim
