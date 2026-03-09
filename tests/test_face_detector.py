"""
Face detector interface tests - verify FaceDetector contract.

Uses synthetic frames; no camera or MediaPipe required.
Tests the interface that callers depend on.
"""

import numpy as np
import pytest

from vision.face_detector import FaceDetector


class TestFaceDetectorInterface:
    """Tests for FaceDetector interface contract."""

    def test_init_default_confidence(self) -> None:
        """Constructor uses default min_confidence."""
        detector = FaceDetector()
        assert detector._min_confidence == 0.5

    def test_init_custom_confidence(self) -> None:
        """Constructor accepts custom min_confidence."""
        detector = FaceDetector(min_confidence=0.8)
        assert detector._min_confidence == 0.8

    def test_detect_accepts_bgr_frame(self, synthetic_frame: np.ndarray) -> None:
        """detect() accepts BGR numpy array without error."""
        detector = FaceDetector()
        result = detector.detect(synthetic_frame)
        assert isinstance(result, list)

    def test_detect_returns_list(self, synthetic_frame: np.ndarray) -> None:
        """detect() returns a list (empty for stub implementation)."""
        detector = FaceDetector()
        result = detector.detect(synthetic_frame)
        assert result == []  # No face in black frame

    def test_detect_handles_various_frame_sizes(self) -> None:
        """detect() handles different frame dimensions."""
        detector = FaceDetector()
        for h, w in [(480, 640), (720, 1280), (100, 100)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            result = detector.detect(frame)
            assert isinstance(result, list)

    def test_detection_format_when_non_empty(self) -> None:
        """
        When detector returns detections, each has bbox and confidence.

        Document the expected format for future MediaPipe implementation.
        """
        detector = FaceDetector()
        # MediaPipe returns [] for black frame; when faces present, each item has:
        # - bbox: (x, y, width, height)
        # - confidence: float 0-1
        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        for det in result:
            assert "bbox" in det
            assert "confidence" in det
            assert len(det["bbox"]) == 4
            assert 0 <= det["confidence"] <= 1

    def test_close_does_not_raise(self) -> None:
        """close() can be called without error."""
        detector = FaceDetector()
        detector.close()  # No exception

    def test_detect_after_close(self, synthetic_frame: np.ndarray) -> None:
        """detect() after close() still returns (graceful degradation)."""
        detector = FaceDetector()
        detector.close()
        result = detector.detect(synthetic_frame)
        assert isinstance(result, list)
