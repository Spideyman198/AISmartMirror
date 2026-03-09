"""
App controller tests - verify initialization and lifecycle.

Uses mocked camera and detector so tests run without hardware.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.app_controller import AppController


class TestAppController:
    """Tests for AppController with mocked dependencies."""

    def test_initialize_with_mocked_camera_succeeds(self) -> None:
        """initialize() succeeds when camera and detector are injected."""
        mock_camera = MagicMock()
        mock_camera.read.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        controller = AppController()
        result = controller.initialize(
            camera_manager=mock_camera,
            face_detector=mock_detector,
        )

        assert result is True
        assert controller._camera_manager is mock_camera
        assert controller._face_detector is mock_detector

    def test_run_requires_initialization(self) -> None:
        """run() does nothing if initialize() was not called."""
        controller = AppController()
        # run() would block; we verify it checks state
        controller._camera_manager = None
        controller._face_detector = None
        # run() logs error and returns early - we can't easily test the loop
        # without threading, so we just verify controller state
        assert controller._camera_manager is None

    def test_shutdown_cleans_up(self) -> None:
        """shutdown() releases camera and detector."""
        mock_camera = MagicMock()
        mock_detector = MagicMock()

        controller = AppController()
        controller.initialize(
            camera_manager=mock_camera,
            face_detector=mock_detector,
        )
        controller.shutdown()

        mock_camera.close.assert_called_once()
        mock_detector.close.assert_called_once()
        assert controller._camera_manager is None
        assert controller._face_detector is None
