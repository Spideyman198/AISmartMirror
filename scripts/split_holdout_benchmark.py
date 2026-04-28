#!/usr/bin/env python3
"""
Move or copy a fraction of each user's crops from data/cnn_faces/<user>/
into data/eval_benchmark/<user>/ so benchmark stays disjoint from training.

Does NOT touch train/ or val/ (those are outputs of prepare_cnn_dataset.py).
Run this BEFORE prepare if you currently have everything under cnn_faces/<user>/.

Usage:
    python scripts/split_holdout_benchmark.py --ratio 0.25 --move
    python scripts/split_holdout_benchmark.py --ratio 0.2 --copy --dry-run
    python scripts/split_holdout_benchmark.py --strangers-from data/incoming_strangers --move
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

CNN_FACES_DIR = project_root / "data" / "cnn_faces"
EVAL_BENCHMARK_DIR = project_root / "data" / "eval_benchmark"


def list_images(d: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        out.extend(d.glob(pat))
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Split holdout images from cnn_faces/<user>/ into eval_benchmark/<user>/"
    )
    p.add_argument(
        "--cnn-root",
        type=Path,
        default=CNN_FACES_DIR,
        help="Root (default: data/cnn_faces)",
    )
    p.add_argument(
        "--benchmark-root",
        type=Path,
        default=EVAL_BENCHMARK_DIR,
        help="Benchmark root (default: data/eval_benchmark)",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.25,
        help="Fraction of each user's images to send to benchmark (default 0.25)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for reproducible splits",
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files (recommended: avoids duplicate paths on disk)",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move (leaves originals; run check_benchmark_leakage.py before train)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only",
    )
    p.add_argument(
        "--strangers-from",
        type=Path,
        default=None,
        help="Folder of stranger crops -> eval_benchmark/_strangers/",
    )
    p.add_argument(
        "--min-keep",
        type=int,
        default=1,
        help="Minimum images to leave in cnn_faces/<user>/ when using --move (default 1)",
    )
    args = p.parse_args()

    if args.move and args.copy:
        print("Use either --move or --copy, not both.")
        sys.exit(1)
    if not args.move and not args.copy:
        print("Specify --move (recommended) or --copy.")
        sys.exit(1)

    if not (0.0 <= args.ratio <= 1.0):
        print("--ratio must be between 0 and 1")
        sys.exit(1)

    root = args.cnn_root
    bench = args.benchmark_root
    random.seed(args.seed)

    class_dirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and d.name not in ("train", "val") and not d.name.startswith(".")
    ]

    if not class_dirs:
        print(f"No user folders under {root} (expected data/cnn_faces/<user_id>/).")
        sys.exit(1)

    op = shutil.move if args.move else shutil.copy2
    total_out = 0
    total_stay = 0

    for class_dir in sorted(class_dirs, key=lambda x: x.name.lower()):
        user_id = class_dir.name
        images = list_images(class_dir)
        if not images:
            print(f"  Skip {user_id}: empty")
            continue

        n = len(images)
        n_holdout = int(round(n * args.ratio))
        if args.move and n - n_holdout < args.min_keep:
            n_holdout = max(0, n - args.min_keep)
        n_holdout = min(n_holdout, n)
        if n_holdout <= 0:
            print(f"  {user_id}: keep all {n} (nothing to hold out)")
            total_stay += n
            continue

        random.shuffle(images)
        holdout = images[:n_holdout]
        stay = images[n_holdout:]

        dest_user = bench / user_id
        if not args.dry_run:
            dest_user.mkdir(parents=True, exist_ok=True)

        for src in holdout:
            dst_name = src.name
            dst = dest_user / dst_name
            if dst.exists() and dst.resolve() != src.resolve():
                dst = dest_user / f"{src.stem}_holdout{src.suffix}"
            if args.dry_run:
                print(f"  [dry-run] {op.__name__} {src} -> {dst}")
            else:
                op(str(src), str(dst))
            total_out += 1
        total_stay += len(stay)
        print(f"  {user_id}: holdout {len(holdout)}, remain {len(stay)}")

    # Strangers: flat folder -> _strangers
    if args.strangers_from is not None:
        src_dir = args.strangers_from
        if not src_dir.is_dir():
            print(f"--strangers-from not a directory: {src_dir}")
            sys.exit(1)
        dest_s = bench / "_strangers"
        imgs = list_images(src_dir)
        if not imgs:
            print(f"No images in {src_dir}")
        else:
            if not args.dry_run:
                dest_s.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(imgs):
                dst = dest_s / f"stranger_{i:04d}{src.suffix.lower()}"
                if args.dry_run:
                    print(f"  [dry-run] {op.__name__} {src} -> {dst}")
                else:
                    op(str(src), str(dst))
            print(f"  _strangers: {len(imgs)} images -> {dest_s}")

    print(f"\nDone. Benchmark images added: {total_out}. Images left under cnn_faces users: {total_stay}")
    if args.copy:
        print("Warning: --copy leaves duplicates in cnn_faces; run check_benchmark_leakage.py before training.")


if __name__ == "__main__":
    main()
