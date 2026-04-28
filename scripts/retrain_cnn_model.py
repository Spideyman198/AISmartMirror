#!/usr/bin/env python3
"""
Manual retraining trigger for the CNN model.

Runs:
1) dataset preparation
2) CNN training

This trains on the full accumulated dataset (all users) to avoid forgetting.
"""

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain CNN model on full dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--aug", choices=("full", "light"), default="full")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset split prep step",
    )
    args = parser.parse_args()

    if not args.skip_prepare:
        print("Step 1/2: Preparing dataset split...")
        prep = subprocess.run(
            [sys.executable, "scripts/prepare_cnn_dataset.py", "--clean"],
            cwd=project_root,
            check=False,
        )
        if prep.returncode != 0:
            print("Dataset preparation failed.")
            sys.exit(prep.returncode)

    print("Step 2/2: Training CNN model...")
    train = subprocess.run(
        [
            sys.executable,
            "scripts/train_cnn_recognizer.py",
            "--epochs",
            str(args.epochs),
            "--aug",
            args.aug,
        ],
        cwd=project_root,
        check=False,
    )
    if train.returncode != 0:
        print("Training failed.")
        sys.exit(train.returncode)

    print("\nRetraining complete.")
    print("CNN model is refreshed and marked up-to-date.")


if __name__ == "__main__":
    main()
