"""
Enrollment tests - quality checks and pose estimation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.enroll_user import is_sample_acceptable, estimate_pose_bucket


def test_sample_too_small_rejected() -> None:
    """Very small crop is rejected."""
    small = np.zeros((40, 40, 3), dtype=np.uint8)
    ok, reason = is_sample_acceptable(small)
    assert ok is False
    assert "too small" in reason


def test_sample_large_enough_accepted_if_sharp() -> None:
    """Large enough, sharp crop is accepted."""
    # Use random pattern to get non-zero Laplacian variance (not uniform)
    sharp = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    ok, reason = is_sample_acceptable(sharp)
    assert ok is True
    assert reason == "ok"


def test_sample_blurry_rejected() -> None:
    """Very blurry (uniform) crop is rejected."""
    blurry = np.full((100, 100, 3), 128, dtype=np.uint8)  # Uniform = zero Laplacian var
    ok, reason = is_sample_acceptable(blurry)
    assert ok is False
    assert "blur" in reason.lower()


def _make_landmarks(eye_center_x: float, eye_center_y: float, nose_x: float, chin_y: float) -> dict:
    """Create minimal landmarks for pose estimation tests."""
    nose_y = eye_center_y + 15
    return {
        "left_eye": [(eye_center_x - 15, eye_center_y)] * 6,
        "right_eye": [(eye_center_x + 15, eye_center_y)] * 6,
        "nose_tip": [(nose_x, nose_y)],
        "nose_bridge": [(nose_x, eye_center_y + 5)],
        "chin": [(eye_center_x, chin_y)] * 17,
        "left_eyebrow": [(eye_center_x - 15, eye_center_y - 10)],
        "right_eyebrow": [(eye_center_x + 15, eye_center_y - 10)],
    }


def test_estimate_pose_center() -> None:
    """Frontal face returns center."""
    # nose centered (yaw~0), pitch_ratio ~1.5 (chin_to_nose / nose_to_eye)
    lm = _make_landmarks(50, 40, 50, 78)
    assert estimate_pose_bucket(lm) == "center"


def test_estimate_pose_left() -> None:
    """Nose right of center returns left (head turned left)."""
    lm = _make_landmarks(50, 40, 58, 78)  # nose_x > eye_center_x => yaw > 0.12
    assert estimate_pose_bucket(lm) == "left"


def test_estimate_pose_right() -> None:
    """Nose left of center returns right."""
    lm = _make_landmarks(50, 40, 42, 78)
    assert estimate_pose_bucket(lm) == "right"


def test_estimate_pose_invalid_returns_none() -> None:
    """Invalid landmarks return None."""
    assert estimate_pose_bucket({}) is None
    assert estimate_pose_bucket({"chin": []}) is None
