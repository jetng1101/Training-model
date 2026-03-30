import os
from pathlib import Path

import torch
import torch.nn as nn


class CheckpointManager:
    """Handles saving and loading model checkpoints."""

    def __init__(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_best(self, model: nn.Module, epoch: int, top1_accuracy: float) -> None:
        """Save the best model checkpoint, overwriting the previous best."""
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

    def load_best(self, model: nn.Module) -> tuple[nn.Module, int, float]:
        """Load the best saved checkpoint into the model."""
        checkpoint_path = self._checkpoint_dir / "best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint["epoch"], checkpoint["top1_accuracy"]
