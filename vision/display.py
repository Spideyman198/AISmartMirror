"""
Display helpers - draw detection results on frames.

Keeps display logic separate from detection logic.
"""

from typing import Any, List

import cv2
import numpy as np


def draw_face_boxes(
    frame: np.ndarray,
    detections: List[dict[str, Any]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw face bounding boxes on a frame.

    Args:
        frame: BGR image to draw on (modified in place, also returned).
        detections: List of dicts with "bbox": (x, y, w, h).
        color: BGR color tuple (default green).
        thickness: Line thickness in pixels.

    Returns:
        The frame with boxes drawn.
    """
    for det in detections:
        bbox = det.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame
