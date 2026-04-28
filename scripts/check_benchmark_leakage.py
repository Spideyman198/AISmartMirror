#!/usr/bin/env python3
"""
Detect content overlap between eval_benchmark and training-related folders.

Training-related = data/cnn_faces/train/, data/cnn_faces/val/, and optional
raw collection dirs data/cnn_faces/<user>/ (excluding train, val).

Exit code 1 if any file in eval_benchmark has the same SHA256 as a training file.

Usage:
    python scripts/check_benchmark_leakage.py
    python scripts/check_benchmark_leakage.py --include-raw-collection false
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

CNN_FACES_DIR = project_root / "data" / "cnn_faces"
EVAL_BENCHMARK_DIR = project_root / "data" / "eval_benchmark"

CHUNK = 1024 * 1024


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def collect_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("**/*.jpg", "**/*.jpeg", "**/*.png"):
        out.extend(root.glob(pat))
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Check benchmark vs training data leakage")
    p.add_argument("--cnn-root", type=Path, default=CNN_FACES_DIR)
    p.add_argument("--benchmark-root", type=Path, default=EVAL_BENCHMARK_DIR)
    p.add_argument(
        "--include-raw-collection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also hash images in cnn_faces/<user>/ (not train/val). Default: true",
    )
    args = p.parse_args()

    bench = args.benchmark_root
    if not bench.is_dir():
        print(f"No benchmark dir: {bench}")
        sys.exit(0)

    train_paths: list[Path] = []
    for sub in ("train", "val"):
        d = args.cnn_root / sub
        if d.is_dir():
            train_paths.extend(collect_images(d))

    if args.include_raw_collection:
        for d in args.cnn_root.iterdir():
            if not d.is_dir() or d.name in ("train", "val") or d.name.startswith("."):
                continue
            train_paths.extend(collect_images(d))

    bench_paths = collect_images(bench)
    if not bench_paths:
        print(f"No images under {bench}")
        sys.exit(0)

    print(f"Hashing {len(train_paths)} training-side files, {len(bench_paths)} benchmark files...")

    train_hashes: dict[str, list[Path]] = {}
    for path in train_paths:
        try:
            h = file_hash(path)
        except OSError as e:
            print(f"  Skip {path}: {e}")
            continue
        train_hashes.setdefault(h, []).append(path)

    leaks: list[tuple[Path, Path]] = []
    for bp in bench_paths:
        try:
            h = file_hash(bp)
        except OSError as e:
            print(f"  Skip {bp}: {e}")
            continue
        if h in train_hashes:
            for tp in train_hashes[h]:
                leaks.append((tp, bp))

    if not leaks:
        print("OK: no content overlap between benchmark and training-side folders.")
        sys.exit(0)

    print("LEAKAGE: same file content appears in training-side and benchmark:")
    for tp, bp in leaks[:50]:
        print(f"  train: {tp}")
        print(f"  bench: {bp}")
    if len(leaks) > 50:
        print(f"  ... and {len(leaks) - 50} more")
    sys.exit(1)


if __name__ == "__main__":
    main()
