"""
CNN model freshness/outdated tracking for hybrid recognition workflow.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(model_dir: Path) -> Path:
    return model_dir / "model_status.json"


def load_model_status(model_dir: Path) -> dict[str, Any]:
    path = _status_path(model_dir)
    if not path.exists():
        return {"cnn_outdated": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"cnn_outdated": False}


def save_model_status(model_dir: Path, payload: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = _status_path(model_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def mark_cnn_outdated(
    model_dir: Path,
    reason: str,
    *,
    user_id: str | None = None,
) -> None:
    data = load_model_status(model_dir)
    data["cnn_outdated"] = True
    data["outdated_reason"] = reason
    data["updated_at"] = _utc_now_iso()
    if user_id:
        data["last_enrolled_user_id"] = user_id
    save_model_status(model_dir, data)


def mark_cnn_fresh(
    model_dir: Path,
    *,
    trained_users_count: int | None = None,
    train_samples_count: int | None = None,
    user_image_counts: dict[str, int] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "cnn_outdated": False,
        "updated_at": _utc_now_iso(),
        "last_train_completed_at": _utc_now_iso(),
    }
    if trained_users_count is not None:
        payload["trained_users_count"] = trained_users_count
    if train_samples_count is not None:
        payload["train_samples_count"] = train_samples_count
    if user_image_counts is not None:
        payload["cnn_user_image_counts"] = user_image_counts
    save_model_status(model_dir, payload)
