import torch.nn as nn
import timm
from omegaconf import DictConfig


def _adapt_for_small_images(model: nn.Module) -> None:
    """Replace aggressive downsampling layers for 32x32 inputs.

    Standard ResNet conv1 (7x7, stride 2) + maxpool (stride 2) reduces
    32x32 → 8x8 before residual blocks, destroying spatial information.
    This replaces them with a 3x3 stride-1 conv and identity pooling.
    """
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False,
    )
    model.maxpool = nn.Identity()


def build_backbone(backbone_config: DictConfig) -> tuple[nn.Module, int]:
    """Build a backbone from timm, optionally adapted for small images.

    Returns:
        Tuple of (backbone module, output feature dimension).
    """
    model = timm.create_model(
        backbone_config.name,
        pretrained=backbone_config.pretrained,
        num_classes=0,
        drop_rate=backbone_config.drop_rate,
    )
    if backbone_config.get("adapt_for_cifar", False):
        _adapt_for_small_images(model)
    feature_dim: int = model.num_features
    return model, feature_dim
