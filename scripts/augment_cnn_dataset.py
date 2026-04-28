#!/usr/bin/env python3
"""
Offline augmentation: create extra JPGs in each data/cnn_faces/<user_id>/ folder.

Reads existing crops (from collect_cnn_faces.py), writes new files aug_XXXXX.jpg.
Then run prepare_cnn_dataset.py and train_cnn_recognizer.py as usual.

This adds diversity on disk; training-time augmentation in train_cnn_recognizer.py still applies.

Usage:
    python scripts/augment_cnn_dataset.py --per-image 4
    python scripts/augment_cnn_dataset.py --per-image 3 --dry-run
    python scripts/augment_cnn_dataset.py --only huda,ola,yahya   # new users only (comma-separated)
"""

import argparse
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from PIL import Image
    from torchvision import transforms
except ImportError as e:
    print("Requires Pillow and torchvision: pip install -r requirements-cnn.txt")
    raise SystemExit(1) from e

CNN_FACES_DIR = project_root / "data" / "cnn_faces"


def build_augment() -> transforms.Compose:
    """Random augment that keeps image size (same HxW as input)."""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12, fill=(0, 0, 0)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=4,
                fill=(0, 0, 0),
            ),
            transforms.ColorJitter(
                brightness=0.28,
                contrast=0.28,
                saturation=0.28,
                hue=0.06,
            ),
            transforms.RandomGrayscale(p=0.06),
        ]
    )


def next_aug_index(user_dir: Path) -> int:
    """First free index for aug_XXXXX.jpg in this folder."""
    existing = list(user_dir.glob("aug_*.jpg")) + list(user_dir.glob("aug_*.png"))
    best = 0
    for p in existing:
        try:
            n = int(p.stem.split("_")[1])
            best = max(best, n)
        except (IndexError, ValueError):
            continue
    return best + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate extra augmented images per user under data/cnn_faces/<user>/"
    )
    parser.add_argument(
        "--per-image",
        type=int,
        default=3,
        help="Number of augmented copies per source image (default 3)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=CNN_FACES_DIR,
        help="Root data/cnn_faces",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only, do not write files",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated user folder names to augment only (e.g. huda,ola). Default: all users.",
    )
    args = parser.parse_args()

    if args.per_image < 1:
        print("--per-image must be >= 1")
        sys.exit(1)

    random.seed(args.seed)

    class_dirs = [
        d
        for d in args.data_dir.iterdir()
        if d.is_dir() and d.name not in ("train", "val")
    ]

    if args.only:
        want = {s.strip().lower().replace(" ", "_") for s in args.only.split(",") if s.strip()}
        class_dirs = [d for d in class_dirs if d.name.lower() in want]
        missing = want - {d.name.lower() for d in class_dirs}
        if missing:
            print(f"Warning: --only folders not found: {sorted(missing)}")

    if not class_dirs:
        print(f"No user folders under {args.data_dir} (excluding train/val).")
        print("Expected: data/cnn_faces/<user_id>/*.jpg")
        sys.exit(1)

    aug = build_augment()
    total_write = 0
    total_src = 0

    for user_dir in sorted(class_dirs):
        images = sorted(user_dir.glob("*.jpg")) + sorted(user_dir.glob("*.png"))
        # Skip previous aug outputs as sources to avoid augmenting aug recursively
        images = [p for p in images if not p.name.startswith("aug_")]
        if not images:
            print(f"  Skip {user_dir.name}: no source images")
            continue

        idx = next_aug_index(user_dir)
        n_here = 0
        for src in images:
            total_src += 1
            try:
                pil = Image.open(src).convert("RGB")
            except OSError as e:
                print(f"  Skip unreadable {src}: {e}")
                continue
            for _ in range(args.per_image):
                out = aug(pil)
                out_path = user_dir / f"aug_{idx:05d}.jpg"
                idx += 1
                n_here += 1
                total_write += 1
                if not args.dry_run:
                    out.save(out_path, quality=92)

        print(f"  {user_dir.name}: {len(images)} sources * {args.per_image} = {n_here} new images")

    if args.dry_run:
        print(f"\nDry run: would write {total_write} files from {total_src} sources.")
    else:
        print(f"\nDone. Wrote {total_write} augmented images under {args.data_dir}/<user>/")
        print("Next: python scripts/prepare_cnn_dataset.py")
        print("      python scripts/train_cnn_recognizer.py --epochs 25 --aug full")


if __name__ == "__main__":
    main()
