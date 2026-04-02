import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.loss import build_criterion
from src.training.metrics import EpochMetrics, compute_topk_accuracy
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import Logger


class Trainer:
    """Manages the full training and validation loop with AMP support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
        logger: Logger,
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._config = config
        self._logger = logger
        self._device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._criterion = build_criterion(config.loss)
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()
        self._scaler = GradScaler(enabled=self._device.type == "cuda")
        self._checkpoint_manager = CheckpointManager(config.trainer.checkpoint_dir)
        self._model.to(self._device)

    def fit(self) -> None:
        """Run training for all epochs, with automatic resume support."""
        best_top1 = 0.0
        start_epoch = 1
        max_epochs = self._config.trainer.max_epochs
        if self._checkpoint_manager.has_last_checkpoint():
            start_epoch, best_top1 = self._checkpoint_manager.load_last(
                self._model, self._optimizer, self._scheduler, self._scaler,
            )
            start_epoch += 1
        for epoch in range(start_epoch, max_epochs + 1):
            train_metrics = self._run_train_epoch(epoch)
            val_metrics = self._run_val_epoch()
            self._scheduler.step()
            self._logger.log_epoch(epoch, train_metrics, val_metrics)
            if val_metrics.top1_accuracy > best_top1:
                best_top1 = val_metrics.top1_accuracy
                self._checkpoint_manager.save_best(self._model, epoch, best_top1)
            self._checkpoint_manager.save_last(
                self._model, self._optimizer, self._scheduler, self._scaler, epoch, best_top1,
            )
        self._logger.finish()

    def _run_train_epoch(self, epoch: int) -> EpochMetrics:
        """Run one full training epoch with mixed precision."""
        self._model.train()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        log_interval = self._config.trainer.log_interval
        is_cuda = self._device.type == "cuda"
        progress = tqdm(self._train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for step, (images, labels) in enumerate(progress):
            images = images.to(self._device)
            labels = labels.to(self._device)
            self._optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=is_cuda):
                logits = self._model(images)
                loss = self._criterion(logits, labels)
            self._scaler.scale(loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
            accuracy = compute_topk_accuracy(logits.float(), labels, top_k=(1, 5))
            total_loss += loss.item()
            total_top1 += accuracy[1]
            total_top5 += accuracy[5]
            if (step + 1) % log_interval == 0:
                self._logger.log_step(step + 1, loss.item(), accuracy)
        num_batches = len(self._train_loader)
        return EpochMetrics(
            loss=total_loss / num_batches,
            top1_accuracy=total_top1 / num_batches,
            top5_accuracy=total_top5 / num_batches,
        )

    @torch.no_grad()
    def _run_val_epoch(self) -> EpochMetrics:
        """Run one full validation epoch with mixed precision."""
        self._model.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        is_cuda = self._device.type == "cuda"
        progress = tqdm(self._val_loader, desc="Epoch [Val]", leave=False)
        for images, labels in progress:
            images = images.to(self._device)
            labels = labels.to(self._device)
            with autocast(device_type="cuda", enabled=is_cuda):
                logits = self._model(images)
                loss = self._criterion(logits, labels)
            accuracy = compute_topk_accuracy(logits.float(), labels, top_k=(1, 5))
            total_loss += loss.item()
            total_top1 += accuracy[1]
            total_top5 += accuracy[5]
        num_batches = len(self._val_loader)
        return EpochMetrics(
            loss=total_loss / num_batches,
            top1_accuracy=total_top1 / num_batches,
            top5_accuracy=total_top5 / num_batches,
        )

    def _build_optimizer(self) -> Optimizer:
        opt_cfg = self._config.optimizer
        return AdamW(
            self._model.parameters(),
            lr=opt_cfg.learning_rate,
            weight_decay=opt_cfg.weight_decay,
        )

    def _build_scheduler(self) -> LRScheduler:
        sched_cfg = self._config.scheduler
        return CosineAnnealingLR(
            self._optimizer,
            T_max=self._config.trainer.max_epochs - sched_cfg.warmup_epochs,
            eta_min=sched_cfg.min_lr,
        )
