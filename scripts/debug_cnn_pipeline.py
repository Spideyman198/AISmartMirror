#!/usr/bin/env python3
"""
Controlled CNN inference check: same model + mapping as live, compare val vs ad-hoc crop.

Run from project root with venv activated:

  python scripts/debug_cnn_pipeline.py
  python scripts/debug_cnn_pipeline.py --val-image data/cnn_faces/val/USER/face_0001.jpg
  python scripts/debug_cnn_pipeline.py --live-crop path/to/saved_live_crop.jpg

Uses CNNFaceRecognizer with debug logging (set LOG_LEVEL=INFO).
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2

from utils.logger import setup_logging
from vision.cnn_face_recognizer import CNNFaceRecognizer


def run_one(recognizer: CNNFaceRecognizer, label: str, path: Path) -> None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[{label}] Failed to read {path}")
        return
    print(f"\n=== {label}: {path.resolve()} ===")
    r = recognizer.recognize(img, debug=True)
    print(
        f"  is_known={r.is_known}  name={r.name!r}  conf={r.confidence:.4f}  "
        f"margin={r.margin:.4f}  reject={r.reject_reason!r}"
    )
    if r.debug_top3:
        print(f"  top3: {r.debug_top3}")


def main() -> None:
    setup_logging(level="INFO")

    parser = argparse.ArgumentParser(description="Debug CNN model + mapping + preprocessing")
    parser.add_argument("--model-dir", type=Path, default=project_root / "data" / "cnn_models")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--val-image", type=Path, default=None, help="A validation crop (jpg/png)")
    parser.add_argument("--live-crop", type=Path, default=None, help="A crop saved from live debug")
    args = parser.parse_args()

    model_pt = args.model_dir / "cnn_face_model.pt"
    if not model_pt.exists():
        print(f"Missing model: {model_pt}")
        sys.exit(1)

    rec = CNNFaceRecognizer(
        model_dir=args.model_dir,
        confidence_threshold=args.threshold,
        min_class_margin=args.margin,
        debug=True,
    )
    if not rec.is_loaded:
        print("Model failed to load.")
        sys.exit(1)

    faces = project_root / "data" / "cnn_faces"
    if args.val_image is None and args.live_crop is None:
        # Pick first val jpg if present
        val_dir = faces / "val"
        if val_dir.is_dir():
            for sub in sorted(val_dir.iterdir()):
                if sub.is_dir():
                    jpgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png"))
                    if jpgs:
                        args.val_image = jpgs[0]
                        print(f"Using default val image: {args.val_image}")
                        break

    if args.val_image:
        run_one(rec, "val (offline)", Path(args.val_image))
    if args.live_crop:
        run_one(rec, "live crop", Path(args.live_crop))

    if not args.val_image and not args.live_crop:
        print("No --val-image or --live-crop; pass at least one, or populate data/cnn_faces/val/")

    rec.close()


if __name__ == "__main__":
    main()
