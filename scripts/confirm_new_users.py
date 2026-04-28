#!/usr/bin/env python3
"""
Confirm new/updated CNN users and run safe retraining pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from vision.model_status import load_model_status, mark_cnn_fresh

CNN_FACES_DIR = project_root / "data" / "cnn_faces"
CNN_MODELS_DIR = project_root / "data" / "cnn_models"
QUICK_UPDATE_EPOCHS = 5
QUICK_UPDATE_LR = 3e-4


def _count_images(directory: Path) -> int:
    total = 0
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        total += len(list(directory.glob(pat)))
    return total


def _collect_raw_user_counts(cnn_faces_dir: Path) -> dict[str, int]:
    users: dict[str, int] = {}
    if not cnn_faces_dir.exists():
        return users
    for d in sorted(cnn_faces_dir.iterdir()):
        if not d.is_dir() or d.name in ("train", "val") or d.name.startswith("."):
            continue
        users[d.name] = _count_images(d)
    return users


def _collect_split_user_counts(cnn_faces_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        split_dir = cnn_faces_dir / split
        if not split_dir.is_dir():
            continue
        for d in split_dir.iterdir():
            if not d.is_dir():
                continue
            counts[d.name] = counts.get(d.name, 0) + _count_images(d)
    return counts


def _load_model_users(model_dir: Path) -> set[str]:
    mapping_path = model_dir / "class_mapping.json"
    if not mapping_path.exists():
        return set()
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    idx_to_class = mapping.get("idx_to_class", {})
    return {str(v) for v in idx_to_class.values()}


def analyze_cnn_user_changes(
    cnn_faces_dir: Path,
    model_dir: Path,
) -> dict:
    raw_counts = _collect_raw_user_counts(cnn_faces_dir)
    model_users = _load_model_users(model_dir)
    status = load_model_status(model_dir)
    baseline_counts = status.get("cnn_user_image_counts", {}) or {}
    if not baseline_counts:
        baseline_counts = _collect_split_user_counts(cnn_faces_dir)

    new_users = sorted([u for u in raw_counts.keys() if u not in model_users and raw_counts[u] > 0])
    updated_users: list[tuple[str, int, int]] = []
    for user in sorted(raw_counts.keys()):
        if user not in model_users:
            continue
        previous = int(baseline_counts.get(user, 0))
        current = int(raw_counts[user])
        if current > previous:
            updated_users.append((user, previous, current))

    return {
        "model_users": sorted(model_users),
        "raw_counts": raw_counts,
        "new_users": new_users,
        "updated_users": updated_users,
    }


def _print_summary(summary: dict) -> None:
    print("\n=== Confirm New Users / Retrain CNN ===")
    print(f"Users in current CNN model: {len(summary['model_users'])}")
    if summary["model_users"]:
        print("  " + ", ".join(summary["model_users"]))
    print(f"New users found: {len(summary['new_users'])}")
    if summary["new_users"]:
        print("  " + ", ".join(summary["new_users"]))
    print(f"Users with new images: {len(summary['updated_users'])}")
    for user, prev, curr in summary["updated_users"]:
        print(f"  - {user}: {prev} -> {curr}")


def _backup_existing_model(model_dir: Path) -> Path | None:
    model_path = model_dir / "cnn_face_model.pt"
    mapping_path = model_dir / "class_mapping.json"
    status_path = model_dir / "model_status.json"
    existing = [p for p in (model_path, mapping_path, status_path) if p.exists()]
    if not existing:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = model_dir / "backups" / ts
    backup_dir.mkdir(parents=True, exist_ok=True)
    for src in existing:
        shutil.copy2(src, backup_dir / src.name)
    return backup_dir


def _restore_backup_if_needed(model_dir: Path, backup_dir: Path | None) -> None:
    if backup_dir is None or not backup_dir.exists():
        return
    for name in ("cnn_face_model.pt", "class_mapping.json", "model_status.json"):
        src = backup_dir / name
        if src.exists():
            shutil.copy2(src, model_dir / name)


def _run_step(command: list[str], label: str) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    proc = subprocess.run(command, cwd=project_root, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


def run_confirm_new_users_flow(auto_confirm: bool = False) -> int:
    model_dir = CNN_MODELS_DIR
    summary = analyze_cnn_user_changes(CNN_FACES_DIR, model_dir)
    _print_summary(summary)

    if not summary["new_users"] and not summary["updated_users"]:
        print("\nNo new/updated CNN user data detected. Retraining not required right now.")
        return 0

    if not auto_confirm:
        answer = input("\nProceed with CNN retraining now? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 0

    backup_dir = _backup_existing_model(model_dir)
    if backup_dir:
        print(f"Backed up current CNN model files to: {backup_dir}")
    else:
        print("No previous CNN model files found to back up (first training run).")

    try:
        _run_step([sys.executable, "scripts/prepare_cnn_dataset.py", "--clean"], "prepare dataset")
        leakage_script = project_root / "scripts" / "check_benchmark_leakage.py"
        if leakage_script.exists():
            _run_step([sys.executable, "scripts/check_benchmark_leakage.py"], "benchmark leakage check")
        warm_start_path = model_dir / "cnn_face_model.pt"
        train_cmd = [
            sys.executable,
            "scripts/train_cnn_recognizer.py",
            "--epochs",
            str(QUICK_UPDATE_EPOCHS),
            "--lr",
            str(QUICK_UPDATE_LR),
        ]
        if warm_start_path.exists():
            train_cmd.extend(["--warm-start-model", str(warm_start_path)])
            print(
                f"Quick update mode: {QUICK_UPDATE_EPOCHS} epochs, lr={QUICK_UPDATE_LR}, "
                f"warm-start from existing model."
            )
        else:
            print(
                f"Quick update mode: {QUICK_UPDATE_EPOCHS} epochs, lr={QUICK_UPDATE_LR}, "
                "no existing model to warm-start from."
            )
        _run_step(train_cmd, "train cnn")
        eval_script = project_root / "scripts" / "evaluate_cnn_recognizer.py"
        if eval_script.exists():
            _run_step([sys.executable, "scripts/evaluate_cnn_recognizer.py"], "evaluate cnn")
    except Exception as exc:
        print(f"\nRetraining failed: {exc}")
        _restore_backup_if_needed(model_dir, backup_dir)
        print("Previous CNN model has been restored/kept.")
        return 1

    raw_counts = _collect_raw_user_counts(CNN_FACES_DIR)
    mark_cnn_fresh(
        model_dir,
        trained_users_count=len(_load_model_users(model_dir)),
        train_samples_count=sum(raw_counts.values()),
        user_image_counts=raw_counts,
    )
    print("\nRetraining complete.")
    print(f"Updated model: {model_dir / 'cnn_face_model.pt'}")
    print(f"Updated class mapping: {model_dir / 'class_mapping.json'}")
    if backup_dir:
        print(f"Previous model backup kept at: {backup_dir}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm new users and retrain CNN safely")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    raise SystemExit(run_confirm_new_users_flow(auto_confirm=args.yes))


if __name__ == "__main__":
    main()
