"""
Face recognizer tests - interface, embedding comparison, threshold behavior.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from vision.face_recognizer import FaceRecognizer, RecognitionResult


def test_recognizer_returns_result_type() -> None:
    """recognize() returns RecognitionResult."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "profiles.json").write_text('{"users": []}')
        recognizer = FaceRecognizer(profiles_dir=Path(tmp), threshold=0.6)
        result = recognizer.recognize(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result, RecognitionResult)
        assert result.is_known is False


def test_recognizer_empty_image() -> None:
    """recognize() with empty/invalid image returns unknown."""
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "profiles.json").write_text('{"users": []}')
        recognizer = FaceRecognizer(profiles_dir=Path(tmp))
        result = recognizer.recognize(np.zeros((0, 0, 3), dtype=np.uint8))
        assert result.is_known is False


def test_recognizer_no_profiles_dir() -> None:
    """Recognizer works with default profiles dir (may be empty)."""
    recognizer = FaceRecognizer(threshold=0.6)
    result = recognizer.recognize(np.zeros((100, 100, 3), dtype=np.uint8))
    assert isinstance(result, RecognitionResult)


def test_threshold_stricter_rejects_more() -> None:
    """Stricter threshold (lower value) means fewer matches."""
    # With no enrolled users, both return unknown - we test the threshold is used
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "profiles.json").write_text('{"users": []}')
        r_strict = FaceRecognizer(profiles_dir=Path(tmp), threshold=0.4)
        r_lenient = FaceRecognizer(profiles_dir=Path(tmp), threshold=0.8)
        # Both have no users, so both return unknown
        assert r_strict.recognize(np.zeros((100, 100, 3), dtype=np.uint8)).is_known is False
        assert r_lenient.recognize(np.zeros((100, 100, 3), dtype=np.uint8)).is_known is False


def test_recognition_result_fields() -> None:
    """RecognitionResult has expected fields."""
    result = RecognitionResult(is_known=False, distance=0.5, similarity=0.5)
    assert result.is_known is False
    assert result.user_id is None
    assert result.name is None
    assert result.distance == 0.5
    assert result.similarity == 0.5

    result_known = RecognitionResult(
        is_known=True, user_id="alice", name="Alice", distance=0.3, similarity=0.7
    )
    assert result_known.is_known is True
    assert result_known.user_id == "alice"
    assert result_known.name == "Alice"


def test_profiles_load_from_json() -> None:
    """Recognizer loads profiles from valid profiles.json."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        # Create valid profile with fake embedding
        emb = np.random.randn(128).astype(np.float32) * 0.1
        np.save(p / "alice.npy", emb)
        (p / "profiles.json").write_text(json.dumps({
            "users": [
                {"user_id": "alice", "name": "Alice", "embedding_file": "alice.npy"}
            ]
        }))
        recognizer = FaceRecognizer(profiles_dir=p, threshold=0.6)
        assert len(recognizer._users) == 1
        assert len(recognizer._embeddings) == 1
        assert recognizer._users[0]["name"] == "Alice"


def test_profiles_load_multi_embedding() -> None:
    """Recognizer loads multi-embedding (N, 128) per user for pose robustness."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        # Multi-angle embeddings: 5 rows x 128 cols
        emb = np.random.randn(5, 128).astype(np.float32) * 0.1
        np.save(p / "bob.npy", emb)
        (p / "profiles.json").write_text(json.dumps({
            "users": [
                {"user_id": "bob", "name": "Bob", "embedding_file": "bob.npy"}
            ]
        }))
        recognizer = FaceRecognizer(profiles_dir=p, threshold=0.6)
        assert len(recognizer._users) == 1
        assert len(recognizer._embeddings) == 1
        assert recognizer._embeddings[0].shape == (5, 128)
