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


def draw_guided_enrollment_overlay(
    frame: np.ndarray,
    guidance_text: str,
    current_target: str,
    total_collected: int,
    total_target: int,
    samples_per_pose: dict[str, int],
    quality_status: str,
    detected_pose: Optional[str] = None,
    current_pose_samples: Optional[tuple[int, int]] = None,
    mode_text: str = "Capturing if quality is good",
) -> np.ndarray:
    """Render enrollment guidance and progress HUD on top of the frame."""
    cv2.putText(
        frame,
        f"Target: {guidance_text}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 220, 60),
        2,
    )
    cv2.putText(
        frame,
        f"Progress: {total_collected}/{total_target}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
    )
    pose_text = " | ".join(f"{pose}:{count}" for pose, count in samples_per_pose.items())
    cv2.putText(
        frame,
        pose_text,
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1,
    )
    cv2.putText(
        frame,
        f"Pose detected: {detected_pose or '?'}",
        (12, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (255, 220, 120),
        1,
    )
    cv2.putText(
        frame,
        mode_text,
        (12, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (190, 255, 190),
        1,
    )
    if current_pose_samples is not None:
        cv2.putText(
            frame,
            f"Samples: {current_pose_samples[0]}/{current_pose_samples[1]} ({current_target})",
            (12, 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
        (180, 220, 255),
        1,
        )
    cv2.putText(
        frame,
        f"Quality: {quality_status}",
        (12, 178),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (180, 220, 255),
        1,
    )
    cv2.putText(
        frame,
        "Auto-capture active. Press N to skip pose, Q to cancel.",
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    return frame
