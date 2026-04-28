"""
CNN face dataset - loads face images by class for training/evaluation.

Dataset structure:
    data/cnn_faces/
        train/
            user1/
                face_001.jpg
                face_002.jpg
            user2/
                ...
        val/
            user1/
            user2/

Each subfolder name is the class (user_id). Images are face crops (any size;
resized to model input during preprocessing).
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

# ImageNet normalization (used by pretrained MobileNetV2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 224


class CNNFaceDataset(Dataset):
    """
    PyTorch Dataset for face images organized by class (user_id).

    Each class is a subfolder. Images are loaded as RGB, resized, normalized.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        input_size: int = DEFAULT_INPUT_SIZE,
        transform=None,
    ) -> None:
        """
        Args:
            root: Path to data/cnn_faces/
            split: "train" or "val"
            input_size: Target size for model input (224 for MobileNetV2)
            transform: Optional torchvision transform (overrides default resize+normalize)
        """
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.transform = transform

        self.samples: list[tuple[Path, int]] = []  # (path, class_idx)
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        self._load_samples()

    def _load_samples(self) -> None:
        """Scan split folder and build sample list."""
        split_dir = self.root / self.split
        if not split_dir.exists():
            return

        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        for class_name in classes:
            class_dir = split_dir / class_name
            idx = self.class_to_idx[class_name]
            for path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
                self.samples.append((path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        path, class_idx = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            # Expects PIL or tensor; convert if needed
            from PIL import Image
            img_pil = Image.fromarray(img)
            img = self.transform(img_pil)
        else:
            # Default: resize, to tensor, normalize
            img = cv2.resize(
                img,
                (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            img = img.astype(np.float32) / 255.0
            img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            import torch
            img = torch.from_numpy(img).float()

        return img, class_idx

    def get_num_classes(self) -> int:
        """Return number of classes (users)."""
        return len(self.class_to_idx)
