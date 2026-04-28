#!/usr/bin/env python3
"""
Compare CNN vs embedding (dlib) recognizers on the same labeled face crops.

Dataset layout (you create this — e.g. holdout session not used in training):
    data/eval_benchmark/
        kinan/
            img01.jpg
        areej/
            ...
        _strangers/          # optional: faces of people NOT enrolled in either system
            x01.jpg

Metrics printed: per-method accuracy on known folders, wrong-ID rate, unknown/reject rate.
Use the same images for both recognizers to compare fairly.

Usage:
    python scripts/evaluate_recognition_comparison.py
    python scripts/evaluate_recognition_comparison.py --benchmark-dir data/eval_hard
    python scripts/evaluate_recognition_comparison.py --strangers-subdir _strangers
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2

from utils.logger import setup_logging
from vision.cnn_face_recognizer import CNNFaceRecognizer
from vision.face_recognizer import FaceRecognizer


def norm_id(s: str | None) -> str:
    return (s or "").strip().lower().replace(" ", "_")


def main() -> None:
    setup_logging(level="ERROR")
    parser = argparse.ArgumentParser(description="Compare CNN vs embedding on same images")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=project_root / "data" / "eval_benchmark",
        help="Folder with <user_id>/*.jpg subfolders (ground truth = folder name)",
    )
    parser.add_argument(
        "--cnn-model-dir",
        type=Path,
        default=project_root / "data" / "cnn_models",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=project_root / "data" / "known_faces",
        help="Embedding profiles (profiles.json + .npy)",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.6,
        help="FaceRecognizer max distance for match (same meaning as FACE_RECOGNITION_THRESHOLD)",
    )
    parser.add_argument(
        "--cnn-threshold",
        type=float,
        default=0.62,
    )
    parser.add_argument(
        "--cnn-margin",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--cnn-max-entropy",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--strangers-subdir",
        type=str,
        default="_strangers",
        help="Optional subfolder name for non-enrolled faces (should be unknown). Empty = skip.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write metrics JSON here",
    )
    args = parser.parse_args()

    root = args.benchmark_dir
    if not root.is_dir():
        print(f"Create benchmark set at {root.resolve()}")
        print("  Layout: <benchmark>/<user_id>/*.jpg  optional: <benchmark>/_strangers/*.jpg")
        sys.exit(1)

    cnn = CNNFaceRecognizer(
        model_dir=args.cnn_model_dir,
        confidence_threshold=args.cnn_threshold,
        min_class_margin=args.cnn_margin,
        max_softmax_entropy=args.cnn_max_entropy,
        debug=False,
    )
    if not cnn.is_loaded:
        print(f"CNN model not found under {args.cnn_model_dir}")
        sys.exit(1)

    emb = FaceRecognizer(
        profiles_dir=args.profiles_dir,
        threshold=args.embedding_threshold,
    )

    # Collect (path, ground_truth_id) — gt is folder name; strangers use special tag
    samples: list[tuple[Path, str]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name.startswith("."):
            continue
        is_strangers = args.strangers_subdir and name == args.strangers_subdir
        gt = "__stranger__" if is_strangers else norm_id(name)
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            for p in sub.glob(pat):
                samples.append((p, gt))

    if not samples:
        print(f"No images under {root}")
        sys.exit(1)

    # Metrics
    cnn_stats = defaultdict(int)  # correct, wrong_id, unknown
    emb_stats = defaultdict(int)

    per_class_cnn: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_class_emb: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path, gt in samples:
        img = cv2.imread(str(path))
        if img is None:
            continue

        cr = cnn.recognize(img)
        cnn_uid = norm_id(cr.user_id if cr.is_known else None)
        if gt == "__stranger__":
            if not cr.is_known:
                cnn_stats["tn_stranger"] += 1
            else:
                cnn_stats["fp_stranger"] += 1  # named a enrolled user
        else:
            if not cr.is_known:
                cnn_stats["fn_known"] += 1
                per_class_cnn[gt]["unknown"] += 1
            elif cnn_uid == gt:
                cnn_stats["correct"] += 1
                per_class_cnn[gt]["correct"] += 1
            else:
                cnn_stats["wrong_id"] += 1
                per_class_cnn[gt]["wrong"] += 1

        er = emb.recognize(img)
        emb_uid = norm_id(er.user_id if er.is_known else None)
        if gt == "__stranger__":
            if not er.is_known:
                emb_stats["tn_stranger"] += 1
            else:
                emb_stats["fp_stranger"] += 1
        else:
            if not er.is_known:
                emb_stats["fn_known"] += 1
                per_class_emb[gt]["unknown"] += 1
            elif emb_uid == gt:
                emb_stats["correct"] += 1
                per_class_emb[gt]["correct"] += 1
            else:
                emb_stats["wrong_id"] += 1
                per_class_emb[gt]["wrong"] += 1

    n = len(samples)
    n_known = sum(1 for _, g in samples if g != "__stranger__")
    n_str = sum(1 for _, g in samples if g == "__stranger__")

    def summarize(stats: dict, label: str) -> None:
        print(f"\n=== {label} ===")
        print(f"  Images: {n} (known folders: {n_known}, stranger: {n_str})")
        print(f"  Correct ID (known GT only): {stats.get('correct', 0)}")
        print(f"  Wrong ID (known GT):        {stats.get('wrong_id', 0)}")
        print(f"  Unknown / reject (known GT): {stats.get('fn_known', 0)}")
        if n_str:
            print(f"  Stranger → unknown (good):  {stats.get('tn_stranger', 0)}")
            print(f"  Stranger → named (bad):     {stats.get('fp_stranger', 0)}")

    summarize(dict(cnn_stats), "CNN")
    summarize(dict(emb_stats), "Embedding (dlib)")

    print("\n=== Per-class (known only) — CNN ===")
    for uid in sorted(per_class_cnn.keys()):
        d = dict(per_class_cnn[uid])
        print(f"  {uid}: {d}")

    print("\n=== Per-class (known only) — Embedding ===")
    for uid in sorted(per_class_emb.keys()):
        d = dict(per_class_emb[uid])
        print(f"  {uid}: {d}")

    out = {
        "benchmark_dir": str(root.resolve()),
        "cnn_model_dir": str(args.cnn_model_dir.resolve()),
        "profiles_dir": str(args.profiles_dir.resolve()),
        "counts": {"total": n, "known_gt": n_known, "stranger_gt": n_str},
        "cnn": dict(cnn_stats),
        "embedding": dict(emb_stats),
        "cnn_per_class": {k: dict(v) for k, v in per_class_cnn.items()},
        "embedding_per_class": {k: dict(v) for k, v in per_class_emb.items()},
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {args.json_out.resolve()}")

    cnn.close()
    emb.close()


if __name__ == "__main__":
    main()
