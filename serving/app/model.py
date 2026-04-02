import io
import logging
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from PIL import Image

from serving.app.config import ServingConfig, settings
from src.models.classifier import Cifar100Classifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """Result of a single image inference."""

    predicted_label: str
    confidence: float
    all_scores: dict[str, float]


class GradingModel:
    """Loads a trained classifier and runs inference on images."""

    def __init__(self, config: ServingConfig = settings) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._labels = config.class_labels
        self._transform = self._build_transform()
        self._model = self._try_load_model()

    def predict(self, image_bytes: bytes) -> InferenceResult:
        """Run inference on raw image bytes and return the grading result."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Place checkpoint at: " + self._config.checkpoint_path)
        tensor = self._preprocess(image_bytes)
        tensor = tensor.to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = probabilities.argmax().item()
        return InferenceResult(
            predicted_label=self._labels[predicted_index],
            confidence=round(probabilities[predicted_index].item(), 4),
            all_scores={
                label: round(prob, 4)
                for label, prob in zip(self._labels, probabilities.tolist())
            },
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _try_load_model(self) -> torch.nn.Module | None:
        """Attempt to load model checkpoint. Returns None if checkpoint not found."""
        checkpoint_path = Path(self._config.checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning("Checkpoint not found at %s — service starts without model", checkpoint_path)
            return None
        model_config = OmegaConf.load(self._config.model_config_path)
        model = Cifar100Classifier(model_config)
        checkpoint = torch.load(
            self._config.checkpoint_path,
            map_location=self._device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self._device)
        model.eval()
        logger.info("Model loaded from %s", checkpoint_path)
        return model

    def _build_transform(self) -> A.Compose:
        """Build the same preprocessing pipeline used during validation."""
        return A.Compose([
            A.Resize(self._config.image_size, self._config.image_size),
            A.Normalize(
                mean=self._config.normalize_mean,
                std=self._config.normalize_std,
            ),
            ToTensorV2(),
        ])

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        numpy_image = np.array(image)
        transformed = self._transform(image=numpy_image)
        return transformed["image"].unsqueeze(0)
