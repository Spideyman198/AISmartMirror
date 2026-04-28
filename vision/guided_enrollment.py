"""
Guided face enrollment using a standard webcam.

Collects high-quality samples automatically across pose targets and saves:
- CNN crops to data/cnn_faces/<user_id>/
- Embeddings to data/known_faces/<user_id>.npy (optional compatibility mode)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Optional

import cv2
import face_recognition
import numpy as np

from config import get_settings
from utils.logger import get_logger
from vision.camera_manager import CAP_DSHOW, CAP_MSMF, CameraManager
from vision.display import draw_guided_enrollment_overlay
from vision.face_detector import FaceDetector
from vision.model_status import mark_cnn_outdated

logger = get_logger(__name__)

WINDOW_NAME = "AISmartMirror - Guided Enrollment"
POSE_ORDER = ["center", "left", "right", "up", "down"]
GUIDANCE = {
    "center": "Look straight",
    "left": "Turn slightly left",
    "right": "Turn slightly right",
    "up": "Look up",
    "down": "Look down",
}


@dataclass
class EnrollmentConfig:
    total_samples: int
    min_face_size: int
    min_laplacian_var: float
    capture_interval_frames: int
    capture_cooldown_sec: float
    min_embedding_distance: float
    require_pose_match: bool
    pose_timeout_sec: float
    steady_frames_required: int
    save_embeddings: bool = True


@dataclass
class EnrollmentResult:
    completed: bool
    cancelled: bool
    user_id: str
    user_name: str
    total_samples: int
    saved_cnn_dir: Path
    saved_embedding_file: Optional[Path]
    pose_counts: dict[str, int]


def default_config() -> EnrollmentConfig:
    s = get_settings()
    return EnrollmentConfig(
        total_samples=max(10, min(80, s.ENROLLMENT_TOTAL_SAMPLES)),
        min_face_size=max(32, s.ENROLLMENT_MIN_FACE_SIZE),
        min_laplacian_var=max(0.0, s.ENROLLMENT_MIN_LAPLACIAN_VAR),
        capture_interval_frames=max(1, s.ENROLL_CAPTURE_INTERVAL_FRAMES),
        capture_cooldown_sec=max(0.1, s.ENROLL_CAPTURE_COOLDOWN_SECONDS),
        min_embedding_distance=max(0.0, s.ENROLLMENT_DUPLICATE_DISTANCE),
        require_pose_match=s.ENROLL_REQUIRE_POSE_MATCH,
        pose_timeout_sec=max(3.0, s.ENROLL_POSE_TIMEOUT_SEC),
        steady_frames_required=max(1, s.ENROLLMENT_STEADY_FRAMES_REQUIRED),
    )


def _laplacian_variance(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _slugify_user_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def _estimate_pose_bucket(landmarks: dict) -> Optional[str]:
    try:
        left_eye = np.array(landmarks["left_eye"])
        right_eye = np.array(landmarks["right_eye"])
        nose_tip = np.array(landmarks["nose_tip"]).mean(axis=0)
        chin = np.array(landmarks["chin"])
        eye_center = (left_eye.mean(axis=0) + right_eye.mean(axis=0)) / 2
        chin_bottom = chin[len(chin) // 2]
        top_y = min(
            p[1] for p in landmarks["left_eyebrow"] + landmarks["right_eyebrow"]
        )

        face_width = abs(right_eye[:, 0].max() - left_eye[:, 0].min())
        if face_width < 1:
            return None

        yaw = (nose_tip[0] - eye_center[0]) / face_width
        nose_to_eye = max(1.0, abs(nose_tip[1] - eye_center[1]))
        chin_to_nose = abs(chin_bottom[1] - nose_tip[1])
        pitch_ratio = chin_to_nose / nose_to_eye

        if abs(yaw) < 0.15 and 1.1 < pitch_ratio < 2.2:
            return "center"
        if yaw > 0.12:
            return "left"
        if yaw < -0.12:
            return "right"
        if pitch_ratio < 1.1:
            return "down"
        if pitch_ratio > 2.0:
            return "up"
        if top_y < 0:
            return None
        return "center"
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _build_targets(total_samples: int) -> tuple[list[str], dict[str, int]]:
    base_targets = list(POSE_ORDER)
    samples_per_target = max(2, total_samples // len(base_targets))
    return base_targets, {target: samples_per_target for target in base_targets}


def _required_blur_for_target(
    base_blur: float,
    target: str,
    face_w: int,
    face_h: int,
    min_face_size: int,
) -> float:
    """Adaptive blur threshold to reduce false rejects during guided turns."""
    required = base_blur
    if target != "center":
        required *= 0.6
    if face_w >= int(min_face_size * 1.4) and face_h >= int(min_face_size * 1.4):
        required *= 0.8
    return max(0.0, required)


def _pose_label_for_ui(bucket: Optional[str]) -> str:
    return bucket if bucket else "unknown"


def run_guided_enrollment(
    user_name: str,
    user_id: Optional[str] = None,
    config: Optional[EnrollmentConfig] = None,
    project_root: Optional[Path] = None,
) -> EnrollmentResult:
    cfg = config or default_config()
    root = project_root or Path(__file__).resolve().parent.parent
    name = user_name.strip()
    if not name:
        raise ValueError("User name cannot be empty.")
    uid = _slugify_user_id(user_id or name)

    cnn_dir = root / "data" / "cnn_faces" / uid
    known_faces_dir = root / "data" / "known_faces"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    known_faces_dir.mkdir(parents=True, exist_ok=True)

    target_order, target_goals = _build_targets(cfg.total_samples)
    target_total = sum(target_goals.values())
    pose_counts = {target: 0 for target in target_order}
    embeddings: list[np.ndarray] = []
    last_capture_ts = 0.0
    sample_idx = len(list(cnn_dir.glob("*.jpg")))

    mgr = CameraManager.find_working_camera(indexes=(0, 1, 2), backends=(CAP_DSHOW, CAP_MSMF))
    if mgr is None:
        raise RuntimeError("Could not open camera for enrollment.")
    detector = FaceDetector(min_confidence=0.5, model_selection=0)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cancelled = False
    steady_frames = 0
    pose_index = 0
    pose_started_ts = time.monotonic()
    frame_index = 0
    last_capture_attempt_ts = 0.0
    quality_status = "waiting for face"
    detected_pose_label = "unknown"
    attempt_gap_sec = max(0.05, cfg.capture_cooldown_sec * 0.25)
    try:
        while pose_index < len(target_order):
            frame = mgr.read()
            if frame is None:
                continue
            frame_index += 1

            detections = detector.detect(frame)
            display = frame.copy()
            current_target = target_order[pose_index]
            guidance_text = GUIDANCE[current_target]
            total_collected = sum(pose_counts.values())
            target_goal = target_goals[current_target]

            elapsed_on_pose = time.monotonic() - pose_started_ts
            if elapsed_on_pose >= cfg.pose_timeout_sec and pose_counts[current_target] < target_goal:
                logger.warning(
                    "Enrollment pose timeout: target=%s collected=%d/%d, moving on.",
                    current_target,
                    pose_counts[current_target],
                    target_goal,
                )
                pose_index += 1
                pose_started_ts = time.monotonic()
                steady_frames = 0
                quality_status = "pose timeout -> moving next"
                continue

            if len(detections) == 1:
                x, y, w, h = detections[0]["bbox"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                now = time.monotonic()
                on_capture_interval = (frame_index % cfg.capture_interval_frames) == 0
                cooldown_passed = (now - last_capture_ts) >= cfg.capture_cooldown_sec
                attempt_gap_passed = (now - last_capture_attempt_ts) >= attempt_gap_sec
                needs_more_for_pose = pose_counts[current_target] < target_goal
                should_attempt_capture = (
                    on_capture_interval
                    and cooldown_passed
                    and attempt_gap_passed
                    and needs_more_for_pose
                )

                if not should_attempt_capture:
                    if not cooldown_passed:
                        quality_status = "hold steady (cooldown)"
                    elif not on_capture_interval:
                        quality_status = "preview live (next capture tick)"
                    else:
                        quality_status = "capturing if quality is good"
                else:
                    last_capture_attempt_ts = now
                    crop = frame[y : y + h, x : x + w]

                    if w < cfg.min_face_size or h < cfg.min_face_size:
                        quality_status = f"reject: small crop ({w}x{h})"
                        steady_frames = 0
                    else:
                        sharpness = _laplacian_variance(crop)
                        required_blur = _required_blur_for_target(
                            cfg.min_laplacian_var,
                            target=current_target,
                            face_w=w,
                            face_h=h,
                            min_face_size=cfg.min_face_size,
                        )
                        if sharpness < required_blur:
                            quality_status = f"reject: blurry ({sharpness:.0f}/{required_blur:.0f})"
                            steady_frames = 0
                        else:
                            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            landmarks_list = face_recognition.face_landmarks(rgb)
                            pose_bucket = _estimate_pose_bucket(landmarks_list[0]) if landmarks_list else None
                            detected_pose_label = _pose_label_for_ui(pose_bucket)
                            if cfg.require_pose_match and pose_bucket != current_target:
                                quality_status = f"adjust pose ({detected_pose_label})"
                                steady_frames = 0
                            elif steady_frames < cfg.steady_frames_required:
                                steady_frames += 1
                                quality_status = f"hold steady ({steady_frames}/{cfg.steady_frames_required})"
                            else:
                                encodings = face_recognition.face_encodings(rgb)
                                if not encodings:
                                    quality_status = "reject: encoding failed"
                                else:
                                    new_embedding = encodings[0]
                                    is_duplicate = False
                                    if embeddings:
                                        distances = face_recognition.face_distance(embeddings, new_embedding)
                                        if float(np.min(distances)) < cfg.min_embedding_distance:
                                            is_duplicate = True
                                    if is_duplicate:
                                        quality_status = "reject: duplicate"
                                    else:
                                        embeddings.append(new_embedding)
                                        pose_counts[current_target] += 1
                                        last_capture_ts = time.monotonic()
                                        sample_idx += 1
                                        out_path = cnn_dir / f"face_{sample_idx:04d}.jpg"
                                        cv2.imwrite(str(out_path), crop)
                                        quality_status = "captured"
                                        steady_frames = 0
                                        if pose_counts[current_target] >= target_goal:
                                            pose_index += 1
                                            pose_started_ts = time.monotonic()

            elif len(detections) > 1:
                quality_status = "reject: multiple faces"
                steady_frames = 0
                detected_pose_label = "unknown"
            else:
                quality_status = "waiting for face"
                detected_pose_label = "unknown"

            draw_guided_enrollment_overlay(
                display,
                guidance_text=guidance_text,
                current_target=current_target,
                total_collected=total_collected,
                total_target=target_total,
                samples_per_pose=pose_counts,
                quality_status=quality_status,
                detected_pose=detected_pose_label,
                current_pose_samples=(pose_counts[current_target], target_goal),
                mode_text=(
                    "Capturing if quality is good"
                    if not cfg.require_pose_match
                    else "Capturing when quality + pose match"
                ),
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cancelled = True
                break
            if key == ord("n"):
                logger.warning(
                    "Enrollment pose skipped by user: target=%s collected=%d/%d",
                    current_target,
                    pose_counts[current_target],
                    target_goal,
                )
                pose_index += 1
                pose_started_ts = time.monotonic()
                steady_frames = 0
    finally:
        mgr.close()
        detector.close()
        cv2.destroyWindow(WINDOW_NAME)

    embedding_path: Optional[Path] = None
    if not cancelled and cfg.save_embeddings and embeddings:
        embedding_path = known_faces_dir / f"{uid}.npy"
        np.save(embedding_path, np.array(embeddings))
        profiles_path = known_faces_dir / "profiles.json"
        if profiles_path.exists():
            with open(profiles_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
        else:
            profile_data = {"users": []}

        users = profile_data.get("users", [])
        existing = next((u for u in users if u.get("user_id") == uid), None)
        payload = {"user_id": uid, "name": name, "embedding_file": f"{uid}.npy"}
        if existing:
            existing.update(payload)
        else:
            users.append(payload)
        profile_data["users"] = users
        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

    if not cancelled and sum(pose_counts.values()) > 0:
        settings = get_settings()
        model_dir = (
            Path(settings.CNN_MODEL_DIR)
            if settings.CNN_MODEL_DIR
            else root / "data" / "cnn_models"
        )
        mark_cnn_outdated(
            model_dir,
            reason="new_enrollment_pending_retrain",
            user_id=uid,
        )

    return EnrollmentResult(
        completed=not cancelled and pose_index >= len(target_order),
        cancelled=cancelled,
        user_id=uid,
        user_name=name,
        total_samples=sum(pose_counts.values()),
        saved_cnn_dir=cnn_dir,
        saved_embedding_file=embedding_path,
        pose_counts=pose_counts,
    )
