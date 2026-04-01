import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def build_train_transforms(augmentation_config: DictConfig) -> A.Compose:
    """Build augmentation pipeline for training split."""
    cfg = augmentation_config
    train_cfg = cfg.train
    image_size = cfg.image_size
    cj = train_cfg.color_jitter
    cd = train_cfg.coarse_dropout

    return A.Compose([
        A.PadIfNeeded(
            min_height=image_size + train_cfg.random_crop_padding * 2,
            min_width=image_size + train_cfg.random_crop_padding * 2,
            border_mode=0,
            value=0,
        ),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=train_cfg.random_horizontal_flip_probability),
        A.ColorJitter(
            brightness=cj.brightness,
            contrast=cj.contrast,
            saturation=cj.saturation,
            hue=cj.hue,
            p=cj.probability,
        ),
        A.CoarseDropout(
            max_holes=cd.max_holes,
            max_height=cd.max_height,
            max_width=cd.max_width,
            min_holes=cd.min_holes,
            min_height=cd.min_height,
            min_width=cd.min_width,
            fill_value=cd.fill_value,
            p=cd.probability,
        ),
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
