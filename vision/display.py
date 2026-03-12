"""
Display helpers - draw detection results on frames.

Keeps display logic separate from detection logic.
"""

from typing import Any, List, Optional

import cv2
import numpy as np


def draw_face_boxes(
    frame: np.ndarray,
    detections: List[dict[str, Any]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    labels: Optional[List[str]] = None,
    debug_infos: Optional[List[dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Draw face bounding boxes, optional labels, and optional debug info.

    Args:
        frame: BGR image to draw on (modified in place, also returned).
        detections: List of dicts with "bbox": (x, y, w, h).
        color: BGR color tuple (default green).
        thickness: Line thickness in pixels.
        labels: Optional list of labels (one per detection). Known=name, unknown="Unknown".
        debug_infos: Optional list of dicts with matched_name, best_distance, threshold, state.

    Returns:
        The frame with boxes and labels drawn.
    """
    for i, det in enumerate(detections):
        bbox = det.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        line_y = y - 12  # Default for debug when no label
        if labels is not None and i < len(labels):
            label = labels[i]
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
                cv2.putText(
                    frame, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                )
                line_y = y - th - 10

        if debug_infos is not None and i < len(debug_infos):
            info = debug_infos[i]
            parts = [
                f"d={info.get('best_distance', 0):.2f}",
                f"t={info.get('threshold', 0):.2f}",
                info.get("state", ""),
            ]
            debug_str = " | ".join(parts)
            cv2.putText(
                frame, debug_str, (x, line_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1,
            )
    return frame
