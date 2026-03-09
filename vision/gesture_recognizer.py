"""
Gesture recognizer - detects hand gestures (swipe, wave, etc.).

Placeholder for future implementation. Will use MediaPipe Hands
for edge-first gesture recognition.
"""

from typing import Any, List

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class GestureRecognizer:
    """
    Recognizes hand gestures from camera frames.

    Placeholder: not implemented yet.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        """
        Initialize the gesture recognizer.

        Args:
            min_confidence: Minimum detection confidence.
        """
        self._min_confidence = min_confidence
        logger.info("GestureRecognizer placeholder (not yet implemented)")

    def recognize(self, frame: np.ndarray) -> List[dict[str, Any]]:
        """
        Detect gestures in the current frame.

        Args:
            frame: BGR image from camera.

        Returns:
            List of detected gestures (empty for now).
        """
        # TODO: Implement with MediaPipe Hands
        return []

    def close(self) -> None:
        """Release resources."""
        pass
