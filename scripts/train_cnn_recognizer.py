#!/usr/bin/env python3
"""
Train lightweight CNN face recognizer using MobileNetV2 + transfer learning.

Uses pretrained ImageNet weights, replaces classifier head with num_classes,
fine-tunes on face crops. Saves model and class mapping to data/cnn_models/.

Usage:
    python scripts/train_cnn_recognizer.py
    python scripts/train_cnn_recognizer.py --epochs 25 --aug full
    python scripts/train_cnn_recognizer.py --aug light   # minimal aug (legacy-style)
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vision.cnn_face_dataset import CNNFaceDataset, DEFAULT_INPUT_SIZE
from vision.cnn_face_model import create_model
from vision.model_status import mark_cnn_fresh

# Optional: torchvision transforms for augmentation
try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

CNN_FACES_DIR = project_root / "data" / "cnn_faces"
CNN_MODELS_DIR = project_root / "data" / "cnn_models"


def _collect_raw_user_counts(cnn_faces_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not cnn_faces_dir.exists():
        return counts
    for d in cnn_faces_dir.iterdir():
        if not d.is_dir() or d.name in ("train", "val") or d.name.startswith("."):
            continue
        n = 0
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            n += len(list(d.glob(pat)))
        counts[d.name] = n
    return counts


def get_train_transform(input_size: int, aug_level: str = "full") -> object:
    """
    Train-time augmentation (virtual "more data").

    full: scale/crop jitter, affine, flip, color, occasional grayscale + blur (live robustness).
    light: smaller flips/rotation/jitter only (closer to old behavior).
    """
    if not HAS_TORCHVISION:
        return None

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=mean, std=std)

    if aug_level == "light":
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            norm,
        ])

    # full: simulate distance / framing / motion / lighting seen in live mirror use
    scale = int(round(input_size * 1.15))
    return transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=12,
            translate=(0.1, 0.1),
            scale=(0.88, 1.12),
            shear=4,
        ),
        transforms.ColorJitter(
            brightness=0.28,
            contrast=0.28,
            saturation=0.28,
            hue=0.06,
        ),
        transforms.RandomGrayscale(p=0.07),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.15, 0.8))],
            p=0.18,
        ),
        transforms.ToTensor(),
        norm,
    ])


def get_val_transform(input_size: int):
    """No augmentation for validation."""
    if not HAS_TORCHVISION:
        return None
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _warm_start_from_existing(model: nn.Module, warm_start_path: Path) -> None:
    """
    Load matching weights from an existing model checkpoint.
    Classifier head shape may differ when new users are added, so mismatched keys are skipped.
    """
    if not warm_start_path.exists():
        print(f"Warm-start skipped: checkpoint not found at {warm_start_path}")
        return
    state = torch.load(warm_start_path, map_location="cpu")
    target = model.state_dict()
    compatible = {
        k: v
        for k, v in state.items()
        if k in target and tuple(v.shape) == tuple(target[k].shape)
    }
    if not compatible:
        print("Warm-start skipped: no compatible weights.")
        return
    target.update(compatible)
    model.load_state_dict(target)
    print(
        f"Warm-start loaded {len(compatible)}/{len(target)} tensors "
        f"from {warm_start_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN face recognizer")
    parser.add_argument("--data-dir", type=Path, default=CNN_FACES_DIR,
                        help="Path to data/cnn_faces/")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--output-dir", type=Path, default=CNN_MODELS_DIR)
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch (not recommended)")
    parser.add_argument(
        "--aug",
        choices=("full", "light"),
        default="full",
        help="Train augmentation: full (default) = strong; light = minimal",
    )
    parser.add_argument(
        "--warm-start-model",
        type=Path,
        default=None,
        help="Optional checkpoint path to initialize matching layers from existing model",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_transform = get_train_transform(args.input_size, aug_level=args.aug)
    val_transform = get_val_transform(args.input_size)

    train_ds = CNNFaceDataset(args.data_dir, split="train", input_size=args.input_size, transform=train_transform)
    val_ds = CNNFaceDataset(args.data_dir, split="val", input_size=args.input_size, transform=val_transform)

    if len(train_ds) == 0:
        print("Error: No training data. Run collect_cnn_faces.py and prepare_cnn_dataset.py first.")
        sys.exit(1)

    num_classes = train_ds.get_num_classes()
    class_to_idx = train_ds.class_to_idx
    idx_to_class = train_ds.idx_to_class

    print(f"Classes: {list(class_to_idx.keys())}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Augmentation: {args.aug}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = create_model(num_classes, pretrained=not args.no_pretrained)
    if args.warm_start_model is not None:
        _warm_start_from_existing(model, args.warm_start_model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        train_loss /= len(train_loader)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_dir / "cnn_face_model.pt")
            # Keep class_mapping.json in sync with every best checkpoint (same training run).
            mapping = {
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
            }
            with open(args.output_dir / "class_mapping.json", "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"  -> Saved best model + class_mapping (val_acc={val_acc:.1f}%)")

    # Final mapping write (same content) so file always exists after training
    mapping = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }
    with open(args.output_dir / "class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nTraining complete. Model saved to {args.output_dir}")
    print(f"  - cnn_face_model.pt")
    print(f"  - class_mapping.json")
    mark_cnn_fresh(
        args.output_dir,
        trained_users_count=num_classes,
        train_samples_count=len(train_ds),
        user_image_counts=_collect_raw_user_counts(args.data_dir),
    )
    print("  - model_status.json (cnn_outdated=false)")


if __name__ == "__main__":
    main()
