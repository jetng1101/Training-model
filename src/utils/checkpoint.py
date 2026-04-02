from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointManager:
    """Handles saving and loading model checkpoints."""

    def __init__(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_best(self, model: nn.Module, epoch: int, top1_accuracy: float) -> None:
        """Save the best model checkpoint (weights only, for inference)."""
        checkpoint_path = self._checkpoint_dir / "best.pth"
        torch.save(
            {
                "epoch": epoch,
                "top1_accuracy": top1_accuracy,
                "model_state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Saved best checkpoint → {checkpoint_path} (epoch={epoch}, top1={top1_accuracy:.2f}%)")

    def save_last(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: GradScaler,
        epoch: int,
        best_top1: float,
    ) -> None:
        """Save full training state every epoch for resume capability."""
        checkpoint_path = self._checkpoint_dir / "last.pth"
        torch.save(
            {
                "epoch": epoch,
                "best_top1": best_top1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Saved last checkpoint → {checkpoint_path} (epoch={epoch})")

    def load_best(self, model: nn.Module) -> tuple[nn.Module, int, float]:
        """Load the best saved checkpoint into the model (for inference)."""
        checkpoint_path = self._checkpoint_dir / "best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint["epoch"], checkpoint["top1_accuracy"]

    def load_last(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: GradScaler,
    ) -> tuple[int, float]:
        """Load the last checkpoint to resume training. Returns (epoch, best_top1)."""
        checkpoint_path = self._checkpoint_dir / "last.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Resumed from {checkpoint_path} (epoch={checkpoint['epoch']})")
        return checkpoint["epoch"], checkpoint["best_top1"]

    def has_last_checkpoint(self) -> bool:
        """Check whether a resumable checkpoint exists."""
        return (self._checkpoint_dir / "last.pth").exists()
