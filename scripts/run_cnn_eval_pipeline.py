#!/usr/bin/env python3
"""
Run CNN data prep, training, val eval, and benchmark comparison in order.

Usage:
    python scripts/run_cnn_eval_pipeline.py
    python scripts/run_cnn_eval_pipeline.py --skip-train
    python scripts/run_cnn_eval_pipeline.py --epochs 20 --aug full
    python scripts/run_cnn_eval_pipeline.py --benchmark-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def run(cmd: list[str], label: str) -> None:
    print(f"\n>>> {label}\n    {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(project_root))
    if r.returncode != 0:
        print(f"Failed: {label} (exit {r.returncode})")
        sys.exit(r.returncode)


def main() -> None:
    py = sys.executable
    p = argparse.ArgumentParser(description="Prepare, train, and evaluate CNN recognition")
    p.add_argument("--skip-prepare", action="store_true", help="Skip prepare_cnn_dataset.py")
    p.add_argument(
        "--prepare-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --clean to prepare (clear train/val per user before copy; default: on)",
    )
    p.add_argument("--skip-train", action="store_true", help="Skip train_cnn_recognizer.py")
    p.add_argument("--skip-val-eval", action="store_true", help="Skip evaluate_cnn_recognizer.py")
    p.add_argument("--benchmark-only", action="store_true", help="Only run evaluate_recognition_comparison.py")
    p.add_argument("--val-ratio", type=float, default=None, help="Passed to prepare_cnn_dataset.py")
    p.add_argument("--seed", type=int, default=None, help="Passed to prepare_cnn_dataset.py")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--aug", choices=("full", "light"), default=None)
    p.add_argument("--benchmark-dir", type=Path, default=project_root / "data" / "eval_benchmark")
    p.add_argument("--cnn-model-dir", type=Path, default=project_root / "data" / "cnn_models")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--leak-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After prepare, run check_benchmark_leakage.py before training (default: on)",
    )
    args = p.parse_args()

    scripts = project_root / "scripts"

    if args.benchmark_only:
        cmd = [
            py,
            str(scripts / "evaluate_recognition_comparison.py"),
            "--benchmark-dir",
            str(args.benchmark_dir),
            "--cnn-model-dir",
            str(args.cnn_model_dir),
        ]
        if args.json_out:
            cmd.extend(["--json-out", str(args.json_out)])
        run(cmd, "Benchmark comparison (CNN vs embedding)")
        return

    if not args.skip_prepare:
        prep = [py, str(scripts / "prepare_cnn_dataset.py")]
        if args.prepare_clean:
            prep.append("--clean")
        if args.val_ratio is not None:
            prep.extend(["--val-ratio", str(args.val_ratio)])
        if args.seed is not None:
            prep.extend(["--seed", str(args.seed)])
        run(prep, "Split cnn_faces into train/val")

    if args.leak_check:
        run([py, str(scripts / "check_benchmark_leakage.py")], "Leak check (benchmark vs training)")

    if not args.skip_train:
        train = [py, str(scripts / "train_cnn_recognizer.py")]
        if args.epochs is not None:
            train.extend(["--epochs", str(args.epochs)])
        if args.batch_size is not None:
            train.extend(["--batch-size", str(args.batch_size)])
        if args.aug is not None:
            train.extend(["--aug", args.aug])
        run(train, "Train CNN")

    if not args.skip_val_eval:
        run([py, str(scripts / "evaluate_cnn_recognizer.py")], "CNN accuracy on val split")

    cmd = [
        py,
        str(scripts / "evaluate_recognition_comparison.py"),
        "--benchmark-dir",
        str(args.benchmark_dir),
        "--cnn-model-dir",
        str(args.cnn_model_dir),
    ]
    if args.json_out:
        cmd.extend(["--json-out", str(args.json_out)])
    run(cmd, "Benchmark comparison (CNN vs embedding)")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
