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
    padded_size = image_size + train_cfg.random_crop_padding * 2

    return A.Compose([
        A.PadIfNeeded(
            min_height=padded_size,
            min_width=padded_size,
            border_mode=0,
            fill=0,
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
            num_holes_range=tuple(cd.num_holes_range),
            hole_height_range=tuple(cd.hole_height_range),
            hole_width_range=tuple(cd.hole_width_range),
            fill=cd.fill,
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
