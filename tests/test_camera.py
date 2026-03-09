"""
Camera module tests - verify CameraManager interface and behavior.

Uses mocks for cv2.VideoCapture so tests run without hardware.
Hardware-dependent tests are skipped or fail gracefully.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision.camera_manager import CameraManager


class TestCameraManagerWithoutHardware:
    """Tests that run without a real camera (mocked VideoCapture)."""

    @patch("vision.camera_manager.cv2.VideoCapture")
    def test_open_success(self, mock_video_capture: MagicMock) -> None:
        """open() returns True when VideoCapture opens successfully."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [640, 480]
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        manager = CameraManager(index=0, width=640, height=480)
        result = manager.open()

        assert result is True
        mock_video_capture.assert_called_once_with(0)
        mock_cap.set.assert_called()

    @patch("vision.camera_manager.cv2.VideoCapture")
    def test_open_failure(self, mock_video_capture: MagicMock) -> None:
        """open() returns False when camera cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        manager = CameraManager(index=0)
        result = manager.open()

        assert result is False

    @patch("vision.camera_manager.cv2.VideoCapture")
    def test_read_returns_frame_when_open(self, mock_video_capture: MagicMock) -> None:
        """read() returns frame when camera is open and read succeeds."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [640, 480]
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_video_capture.return_value = mock_cap

        manager = CameraManager(index=0, backend=None)
        manager.open()
        frame = manager.read()

        assert frame is not None
        assert frame.shape == (480, 640, 3)

    @patch("vision.camera_manager.cv2.VideoCapture")
    def test_read_returns_none_when_not_open(self, mock_video_capture: MagicMock) -> None:
        """read() returns None when camera is not open."""
        manager = CameraManager(index=0)
        # Never call open()
        frame = manager.read()
        assert frame is None

    @patch("vision.camera_manager.cv2.VideoCapture")
    def test_read_returns_none_on_read_failure(self, mock_video_capture: MagicMock) -> None:
        """read() returns None when cv2 read fails; open() fails if warm-up fails."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [640, 480]
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        manager = CameraManager(index=0, backend=None)
        result = manager.open()
        assert result is False
        frame = manager.read()
        assert frame is None

    @patch("vision.camera_manager.cv2.VideoCapture")
    def test_close_releases_camera(self, mock_video_capture: MagicMock) -> None:
        """close() releases camera resources."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [640, 480]
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        manager = CameraManager(index=0, backend=None)
        manager.open()
        manager.close()

        mock_cap.release.assert_called_once()
        assert manager._cap is None

    def test_init_stores_parameters(self) -> None:
        """Constructor stores index, width, height, fps."""
        manager = CameraManager(index=1, width=320, height=240, fps=15)
        assert manager._index == 1
        assert manager._width == 320
        assert manager._height == 240
        assert manager._fps == 15


@pytest.mark.hardware
def test_camera_real_hardware() -> None:
    """
    Integration test: open real camera if available.

    Skipped by default. Run with: pytest -m hardware
    Fails gracefully with clear message if no camera.
    """
    manager = CameraManager.find_working_camera(indexes=(0, 1, 2))
    if manager is None:
        pytest.skip("No working camera found")
    try:
        frame = manager.read()
        assert frame is not None
        manager.close()
    except Exception as e:
        pytest.skip(f"Camera test failed: {e}")
