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
            # Open camera first (matches test_camera.py flow), then load face detector
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

            # Init face detector after camera (MediaPipe loads last)
            if face_detector is not None:
                self._face_detector = face_detector
            else:
                from vision.face_detector import FaceDetector

                self._face_detector = FaceDetector(
                    min_confidence=self._settings.FACE_DETECTION_CONFIDENCE,
                    model_selection=self._settings.FACE_DETECTION_MODEL,
                )
            # Init face recognizer (loads local embeddings)
            profiles_dir = self._settings.KNOWN_FACES_DIR
            if profiles_dir:
                from vision.face_recognizer import FaceRecognizer
                from pathlib import Path

                self._face_recognizer = FaceRecognizer(
                    profiles_dir=Path(profiles_dir),
                    threshold=self._settings.FACE_RECOGNITION_THRESHOLD,
                )
            else:
                from vision.face_recognizer import FaceRecognizer

                self._face_recognizer = FaceRecognizer(
                    threshold=self._settings.FACE_RECOGNITION_THRESHOLD,
                )

            self._gesture_recognizer = None  # Placeholder for now

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
            from vision.recognition_smoother import RecognitionSmoother

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

            frame_count = 0
            debug_saved = False
            last_labels: list[str] = []
            last_debug_infos: list[dict] | None = None
            recognition_interval = max(1, self._settings.FACE_RECOGNITION_INTERVAL_FRAMES)
            smoother = RecognitionSmoother(
                confirmation_count=max(1, self._settings.RECOGNITION_CONFIRMATION_COUNT),
            )
            show_debug = self._settings.DEBUG_RECOGNITION

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

                # Debug: log first frame, always save one for verification
                if frame_count == 1:
                    mean_val = float(frame.mean())
                    logger.info(
                        "Frame debug: shape=%s dtype=%s mean=%.1f (0=black, ~128=normal)",
                        frame.shape,
                        frame.dtype,
                        mean_val,
                    )
                    if not debug_saved:
                        path = "debug_frame.jpg"
                        cv2.imwrite(path, frame)
                        logger.info("Debug frame saved to %s (verify it is not black)", path)
                        debug_saved = True

                # Detect faces every frame (fast, MediaPipe)
                detections = self._face_detector.detect(frame)

                # Run recognition every N frames or when face count changes
                run_recognition = (
                    self._face_recognizer
                    and (
                        frame_count % recognition_interval == 0
                        or len(detections) != len(last_labels)
                    )
                )
                if run_recognition:
                    results = []
                    for det in detections:
                        x, y, w, h = det["bbox"]
                        crop = frame[y : y + h, x : x + w]
                        result = self._face_recognizer.recognize(crop)
                        results.append(result)
                    labels, debug_infos = smoother.update(
                        results,
                        threshold=self._settings.FACE_RECOGNITION_THRESHOLD,
                        show_debug=show_debug,
                    )
                    last_labels = labels
                    last_debug_infos = debug_infos if show_debug else None
                else:
                    labels = last_labels if len(detections) == len(last_labels) else []
                    debug_infos = last_debug_infos if labels else None

                # Display: draw boxes, labels, and optional debug info
                display_frame = frame.copy()
                draw_face_boxes(
                    display_frame,
                    detections,
                    labels=labels if labels else None,
                    debug_infos=debug_infos if show_debug else None,
                )
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

        if self._face_recognizer:
            self._face_recognizer.close()
            self._face_recognizer = None

        logger.info("Shutdown complete")
