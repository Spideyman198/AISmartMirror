#!/usr/bin/env python3
"""
Prepare CNN dataset by splitting collected images into train/val.

Expects: data/cnn_faces/<user_id>/*.jpg (from collect_cnn_faces.py)
Creates: data/cnn_faces/train/<user_id>/, data/cnn_faces/val/<user_id>/
Uses 80/20 split by default.

Usage:
    python scripts/prepare_cnn_dataset.py
    python scripts/prepare_cnn_dataset.py --val-ratio 0.2
    python scripts/prepare_cnn_dataset.py --clean   # drop stale copies in train/val before split
"""

import argparse
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

CNN_FACES_DIR = project_root / "data" / "cnn_faces"


def clear_split_images(user_train: Path, user_val: Path) -> None:
    """Remove previous train/val copies for this user so stale files cannot linger."""
    for d in (user_train, user_val):
        if not d.is_dir():
            continue
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            for f in d.glob(pat):
                f.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split collected faces into train/val")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear train/<user> and val/<user> images before copying (recommended after holdout split)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Find class folders (user_id dirs with images, excluding train/val)
    class_dirs = [
        d for d in CNN_FACES_DIR.iterdir()
        if d.is_dir() and d.name not in ("train", "val")
    ]

    if not class_dirs:
        print("No class folders found. Run collect_cnn_faces.py first.")
        print(f"Expected: {CNN_FACES_DIR}/<user_id>/*.jpg")
        sys.exit(1)

    train_dir = CNN_FACES_DIR / "train"
    val_dir = CNN_FACES_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0

    for class_dir in sorted(class_dirs):
        user_id = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not images:
            print(f"  Skip {user_id}: no images")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_ratio))
        n_train = len(images) - n_val

        train_user = train_dir / user_id
        val_user = val_dir / user_id
        train_user.mkdir(exist_ok=True)
        val_user.mkdir(exist_ok=True)
        if args.clean:
            clear_split_images(train_user, val_user)

        for i, src in enumerate(images):
            dst = (val_user if i < n_val else train_user) / src.name
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                import shutil
                shutil.copy2(src, dst)

        total_train += n_train
        total_val += n_val
        print(f"  {user_id}: train={n_train}, val={n_val}")

    print(f"\nDone. train={total_train}, val={total_val}")
    print(f"  {train_dir}")
    print(f"  {val_dir}")


if __name__ == "__main__":
    main()
