import numpy as np
import albumentations as A
from torch import Tensor
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from PIL import Image


class Cifar100Dataset(Dataset):
    """PyTorch Dataset wrapper for the HuggingFace uoft-cs/cifar100 dataset."""

    def __init__(
        self,
        split: str,
        transform: A.Compose | None = None,
        label_column: str = "fine_label",
    ) -> None:
        self._transform = transform
        self._label_column = label_column
        self._data: HFDataset = load_dataset("uoft-cs/cifar100", split=split)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self._data[index]
        image = self._convert_to_numpy(sample["img"])
        label: int = sample[self._label_column]

        if self._transform is not None:
            augmented = self._transform(image=image)
            image = augmented["image"]

        return image, label

    def _convert_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to numpy array expected by albumentations."""
        return np.array(image)
