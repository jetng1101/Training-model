from dataclasses import dataclass

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import Cifar100Dataset
from src.data.transforms import build_train_transforms, build_val_transforms


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader


def build_dataloaders(data_config: DictConfig) -> DataLoaders:
    """Build train and validation DataLoaders from config."""
    train_transform = build_train_transforms(data_config.augmentation)
    val_transform = build_val_transforms(data_config.augmentation)

    train_dataset = Cifar100Dataset(
        split="train",
        transform=train_transform,
        label_column=data_config.dataset.label_column,
    )
    val_dataset = Cifar100Dataset(
        split="test",
        transform=val_transform,
        label_column=data_config.dataset.label_column,
    )

    loader_cfg = data_config.loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=True,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
    )

    return DataLoaders(train=train_loader, val=val_loader)
