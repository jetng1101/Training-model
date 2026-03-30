import wandb
from omegaconf import DictConfig, OmegaConf

from src.training.metrics import EpochMetrics


class Logger:
    """Unified logger that writes to both W&B and stdout."""

    def __init__(self, config: DictConfig) -> None:
        wandb_cfg = config.wandb
        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=config.experiment_name,
            tags=list(wandb_cfg.tags),
            config=OmegaConf.to_container(config, resolve=True),
        )

    def log_step(self, step: int, loss: float, accuracy: dict[int, float]) -> None:
        """Log a single training step."""
        wandb.log({"train/step_loss": loss, "train/step_top1": accuracy[1]})

    def log_epoch(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
    ) -> None:
        """Log aggregated metrics for a completed epoch."""
        wandb.log({
            "epoch": epoch,
            "train/loss": train_metrics.loss,
            "train/top1_accuracy": train_metrics.top1_accuracy,
            "train/top5_accuracy": train_metrics.top5_accuracy,
            "val/loss": val_metrics.loss,
            "val/top1_accuracy": val_metrics.top1_accuracy,
            "val/top5_accuracy": val_metrics.top5_accuracy,
        })
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_top1={train_metrics.top1_accuracy:.2f}% | "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_top1={val_metrics.top1_accuracy:.2f}%"
        )

    def finish(self) -> None:
        """Finalize the W&B run."""
        wandb.finish()
