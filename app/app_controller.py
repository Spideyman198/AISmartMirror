"""
Application controller - orchestrates all modules and application lifecycle.

Initializes config, logger, camera, detectors, and manages the main loop.
Designed for clean startup/shutdown and dependency injection.
"""

from typing import Optional

import cv2

from config import get_settings
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

WINDOW_NAME = "AISmartMirror - Face Detection"


class AppController:
    """
    Central controller for the AI Smart Mirror application.

    Owns initialization and teardown of all subsystems.
    """

    def __init__(self) -> None:
        """Initialize controller (does not start subsystems yet)."""
        self._settings = get_settings()
        self._camera_manager = None
        self._face_detector = None
        self._face_recognizer = None
        self._gesture_recognizer = None
        self._running = False

    def initialize(
        self,
        camera_manager=None,
        face_detector=None,
    ) -> bool:
        """
        Initialize all subsystems: config, logging, camera, detectors.

        Args:
            camera_manager: Optional pre-initialized camera (for testing).
            face_detector: Optional pre-initialized detector (for testing).

        Returns:
            True if initialization succeeded, False otherwise.
        """
        setup_logging(level=self._settings.LOG_LEVEL)
        logger.info("Initializing %s", self._settings.APP_NAME)

        try:
            # Init face detector first (loads MediaPipe/TensorFlow) so camera opens last with fresh buffer
            if face_detector is not None:
                self._face_detector = face_detector
            else:
                from vision.face_detector import FaceDetector

                self._face_detector = FaceDetector(
                    min_confidence=self._settings.FACE_DETECTION_CONFIDENCE,
                )
            self._face_recognizer = None  # Placeholder for now
            self._gesture_recognizer = None  # Placeholder for now

            if camera_manager is not None:
                self._camera_manager = camera_manager
            else:
                from vision.camera_manager import (
                    CameraManager,
                    CAP_DSHOW,
                    CAP_MSMF,
                )

                backend = self._settings.CAMERA_BACKEND
                if backend:
                    # Use explicit backend from config
                    backend_map = {"DSHOW": CAP_DSHOW, "MSMF": CAP_MSMF}
                    backend_val = backend_map.get(backend.upper())
                    if backend_val is None:
                        logger.warning("Unknown CAMERA_BACKEND=%s, trying auto", backend)
                        self._camera_manager = CameraManager.find_working_camera(
                            indexes=(self._settings.CAMERA_INDEX, 0, 1, 2),
                            backends=(CAP_DSHOW, CAP_MSMF),
                            width=self._settings.CAMERA_WIDTH,
                            height=self._settings.CAMERA_HEIGHT,
                            fps=self._settings.CAMERA_FPS,
                        )
                    else:
                        mgr = CameraManager(
                            index=self._settings.CAMERA_INDEX,
                            width=self._settings.CAMERA_WIDTH,
                            height=self._settings.CAMERA_HEIGHT,
                            fps=self._settings.CAMERA_FPS,
                            backend=backend_val,
                        )
                        self._camera_manager = mgr if mgr.open() else None
                else:
                    # Auto: try DSHOW, MSMF across indexes
                    self._camera_manager = CameraManager.find_working_camera(
                        indexes=(self._settings.CAMERA_INDEX, 0, 1, 2),
                        backends=(CAP_DSHOW, CAP_MSMF),
                        width=self._settings.CAMERA_WIDTH,
                        height=self._settings.CAMERA_HEIGHT,
                        fps=self._settings.CAMERA_FPS,
                    )

                if not self._camera_manager:
                    logger.error("Failed to open camera (tried indexes and backends)")
                    return False

                logger.info(
                    "Camera ready: index=%d backend=%s",
                    self._camera_manager._index,
                    self._camera_manager.get_backend_name(),
                )

            logger.info("Initialization complete")
            return True

        except Exception as e:
            logger.exception("Initialization failed: %s", e)
            return False

    def run(self) -> None:
        """Run the main application loop with live webcam display."""
        if not self._camera_manager or not self._face_detector:
            logger.error("Cannot run: initialize() must succeed first")
            return

        self._running = True
        max_consecutive_failures = 30
        consecutive_failures = 0
        failure_log_interval = 10  # Log every N failures to avoid spam

        logger.info(
            "Starting main loop. Camera index=%d backend=%s. Press 'q' to quit.",
            self._camera_manager._index,
            self._camera_manager.get_backend_name(),
        )

        try:
            from vision.display import draw_face_boxes

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            # Flush buffer: grab without decode to discard stale frames
            self._camera_manager.flush_buffer(30)

            frame_count = 0
            debug_saved = False

            while self._running:
                frame = self._camera_manager.read()
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        logger.warning("Frame read failed (attempt 1)")
                    elif consecutive_failures % failure_log_interval == 0:
                        logger.warning(
                            "Frame read failed %d times in a row",
                            consecutive_failures,
                        )
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(
                            "Frame capture failed %d times. Camera may be disconnected. Stopping.",
                            max_consecutive_failures,
                        )
                        break
                    continue

                consecutive_failures = 0  # Reset on success
                frame_count += 1

                # Debug: log first frame, optionally save
                if frame_count == 1:
                    mean_val = float(frame.mean())
                    logger.info(
                        "Frame debug: shape=%s dtype=%s mean=%.1f (0=black, ~128=normal)",
                        frame.shape,
                        frame.dtype,
                        mean_val,
                    )
                    if self._settings.DEBUG_SAVE_FRAME and not debug_saved:
                        path = "debug_frame.jpg"
                        cv2.imwrite(path, frame)
                        logger.info("Debug frame saved to %s", path)
                        debug_saved = True

                # Detect faces (MediaPipe uses RGB copy internally; frame stays BGR)
                detections = self._face_detector.detect(frame)

                # Display: draw boxes on a copy so we never modify the raw frame
                display_frame = frame.copy()
                draw_face_boxes(display_frame, detections)
                cv2.imshow(WINDOW_NAME, display_frame)

                # Check for quit: 'q' key or window closed
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed")
                    break
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        logger.info("Window closed")
                        break
                except cv2.error:
                    pass  # getWindowProperty not supported on all platforms

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            cv2.destroyAllWindows()
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown of all subsystems."""
        self._running = False
        logger.info("Shutting down...")

        if self._camera_manager:
            self._camera_manager.close()
            self._camera_manager = None

        if self._face_detector:
            self._face_detector.close()
            self._face_detector = None

        logger.info("Shutdown complete")
