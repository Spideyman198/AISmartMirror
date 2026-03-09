"""
Pytest configuration and shared fixtures.

Ensures project root is in Python path. Provides fixtures for
mocked hardware and synthetic test data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path (for IDE and direct pytest runs)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Synthetic BGR frame (480x640x3) for testing without camera."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection() -> dict:
    """Sample face detection dict matching FaceDetector interface."""
    return {
        "bbox": (100, 100, 150, 150),
        "confidence": 0.92,
    }


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Configure logging for tests to reduce noise."""
    from utils.logger import setup_logging
    setup_logging(level="WARNING")
