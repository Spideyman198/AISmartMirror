"""
Face detector - real-time face detection using MediaPipe.

Provides a consistent detection format for downstream use (e.g. face recognition).
Designed for speed and Raspberry Pi compatibility.
"""

from typing import Any, List

import cv2
import mediapipe as mp
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# Consistent format for all face detections
# bbox: (x, y, width, height) in pixels
# confidence: float 0.0-1.0
FaceDetection = dict[str, Any]


class FaceDetector:
    """
    Real-time face detection using MediaPipe Face Detection.

    detect(frame) returns a list of detections. Each detection is a dict with:
        - bbox: (x, y, width, height) in pixel coordinates
        - confidence: float 0.0-1.0
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """
        Initialize the face detector.

        Args:
            min_confidence: Minimum detection confidence threshold (0.0-1.0).
        """
        self._min_confidence = min_confidence
        self._mp_face_detection = mp.solutions.face_detection
        self._face_detection = self._mp_face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=0,  # 0=short-range (2m), 1=full-range (5m)
        )
        logger.info("Face detector initialized (min_confidence=%.2f)", min_confidence)

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a BGR image frame.

        Args:
            frame: OpenCV BGR image (H, W, 3).

        Returns:
            List of detections. Each dict has:
                - bbox: (x, y, width, height) in pixels
                - confidence: float 0.0-1.0
        """
        if frame is None or frame.size == 0:
            return []
        if self._face_detection is None:
            return []  # Graceful degradation after close()

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_detection.process(rgb)

        detections: List[FaceDetection] = []
        if results.detections:
            for detection in results.detections:
                if detection.score[0] < self._min_confidence:
                    continue
                # MediaPipe uses normalized coords (0-1)
                bbox_rel = detection.location_data.relative_bounding_box
                x = int(bbox_rel.xmin * w)
                y = int(bbox_rel.ymin * h)
                bw = int(bbox_rel.width * w)
                bh = int(bbox_rel.height * h)
                # Clamp to frame bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                detections.append({
                    "bbox": (x, y, bw, bh),
                    "confidence": float(detection.score[0]),
                })
        return detections

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_detection.close()
        self._face_detection = None
        logger.debug("Face detector closed")
