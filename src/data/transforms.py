import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def build_train_transforms(augmentation_config: DictConfig) -> A.Compose:
    """Build augmentation pipeline for training split."""
    cfg = augmentation_config
    train_cfg = cfg.train
    image_size = cfg.image_size

    return A.Compose([
        A.RandomCrop(height=image_size, width=image_size, pad_mode=4, pad_val=0),
        A.HorizontalFlip(p=train_cfg.random_horizontal_flip_probability),
        A.Normalize(mean=train_cfg.normalize_mean, std=train_cfg.normalize_std),
        ToTensorV2(),
    ])


def build_val_transforms(augmentation_config: DictConfig) -> A.Compose:
    """Build augmentation pipeline for validation/test split."""
    val_cfg = augmentation_config.val

    return A.Compose([
        A.Normalize(mean=val_cfg.normalize_mean, std=val_cfg.normalize_std),
        ToTensorV2(),
    ])
