#!/usr/bin/env python3
"""
Guided auto-enrollment for face recognition.

Automatically collects ~20 high-quality samples across poses (center, left, right,
up, down) while the user follows on-screen guidance. Rejects poor quality and
duplicate samples. No biometric data is sent to the cloud.

Usage: python scripts/enroll_user.py --name Alice
       python scripts/enroll_user.py --name Alice --samples 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import face_recognition
import numpy as np

from utils.logger import get_logger, setup_logging
from vision.camera_manager import CameraManager, CAP_DSHOW, CAP_MSMF
from vision.face_detector import FaceDetector

logger = get_logger(__name__)

# Target: 20 high-quality embeddings, exactly 4 per pose in order
DEFAULT_SAMPLES = 20
SAMPLES_PER_BUCKET = 4  # Must get 4 center, then 4 left, then 4 right, then 4 up, then 4 down
MIN_FACE_SIZE = 64
MIN_LAPLACIAN_VAR = 80
MIN_EMBEDDING_DISTANCE = 0.03  # Reject if too similar to existing (duplicate)
CAPTURE_COOLDOWN_SEC = 0.6  # Min time between captures

# Pose buckets for balanced collection
BUCKETS = ["center", "left", "right", "up", "down"]
GUIDANCE = {
    "center": "Look straight at camera",
    "left": "Turn slightly left",
    "right": "Turn slightly right",
    "up": "Look slightly up",
    "down": "Look slightly down",
}


def _laplacian_variance(gray: np.ndarray) -> float:
    """Blur check: low variance = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sample_acceptable(crop: np.ndarray, min_size: int = MIN_FACE_SIZE) -> tuple[bool, str]:
    """Check if face crop is acceptable (size, sharpness). Returns (ok, reason)."""
    h, w = crop.shape[:2]
    if w < min_size or h < min_size:
        return False, f"face too small ({w}x{h})"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    if _laplacian_variance(gray) < MIN_LAPLACIAN_VAR:
        return False, "too blurry"
    return True, "ok"


def estimate_pose_bucket(landmarks: dict) -> str | None:
    """
    Estimate pose bucket from face landmarks (simple 2D heuristic).

    Yaw: nose horizontal offset from eye center (left/right).
    Pitch: chin-to-nose vs nose-to-eye ratio (up = chin closer to nose).
    Center: neutral yaw + neutral pitch. Fallback to center for ambiguous poses.
    """
    try:
        left_eye = np.array(landmarks["left_eye"])
        right_eye = np.array(landmarks["right_eye"])
        nose_tip = np.array(landmarks["nose_tip"]).mean(axis=0)
        nose_bridge = np.array(landmarks["nose_bridge"])
        chin = np.array(landmarks["chin"])

        eye_center = (left_eye.mean(axis=0) + right_eye.mean(axis=0)) / 2
        chin_bottom = chin[len(chin) // 2]
        top_y = min(
            p[1] for p in landmarks["left_eyebrow"] + landmarks["right_eyebrow"]
        )

        face_width = abs(right_eye[:, 0].max() - left_eye[:, 0].min())
        face_height = chin_bottom[1] - top_y
        if face_width < 1 or face_height < 1:
            return None

        # Yaw: nose offset from eye center. Positive = head turned left
        yaw = (nose_tip[0] - eye_center[0]) / face_width
        # Pitch: chin-nose distance vs nose-eye distance. Up = chin closer = smaller ratio
        nose_to_eye = abs(nose_tip[1] - eye_center[1])
        chin_to_nose = abs(chin_bottom[1] - nose_tip[1])
        if nose_to_eye < 1:
            nose_to_eye = 1
        pitch_ratio = chin_to_nose / nose_to_eye  # ~1.5 straight, <1.2 up, >2.0 down

        # Center: neutral yaw and pitch (prioritize center for ambiguous)
        if abs(yaw) < 0.15 and 1.1 < pitch_ratio < 2.2:
            return "center"
        if yaw > 0.12:
            return "left"
        if yaw < -0.12:
            return "right"
        if pitch_ratio < 1.1:
            return "down"  # chin close to nose = looking down (camera sees chin up)
        if pitch_ratio > 2.0:
            return "up"  # chin far from nose = looking up (camera sees chin down)
        return "center"  # Fallback: neutral pose
    except (KeyError, IndexError, TypeError):
        return None


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Guided auto-enrollment for face recognition")
    parser.add_argument("--name", "-n", required=True, help="Display name for the user")
    parser.add_argument("--samples", "-s", type=int, default=DEFAULT_SAMPLES,
                        help=f"Target samples (default {DEFAULT_SAMPLES})")
    parser.add_argument("--profiles-dir", "-p", default=None, help="Override profiles directory")
    args = parser.parse_args()

    name = args.name.strip()
    if not name:
        print("Error: Name cannot be empty")
        sys.exit(1)

    target = max(5, min(50, args.samples))
    user_id = name.lower().replace(" ", "_")
    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else project_root / "data" / "known_faces"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    print(f"Enrolling user: {name} (id={user_id})")
    print(f"Target: {target} samples across poses (center, left, right, up, down)")
    print("Follow the on-screen guidance. Samples capture automatically. Press Q to cancel.")
    print()

    mgr = CameraManager.find_working_camera(indexes=(0, 1, 2), backends=(CAP_DSHOW, CAP_MSMF))
    if mgr is None:
        print("Error: Could not open camera")
        sys.exit(1)

    detector = FaceDetector(min_confidence=0.5, model_selection=0)
    embeddings_list: list[np.ndarray] = []
    bucket_counts: dict[str, int] = {b: 0 for b in BUCKETS}
    last_capture_time = 0.0

    cv2.namedWindow("Enrollment", cv2.WINDOW_NORMAL)

    while sum(bucket_counts.values()) < target:
        frame = mgr.read()
        if frame is None:
            continue

        now = time.monotonic()
        detections = detector.detect(frame)
        display = frame.copy()

        # Strict order: center -> left -> right -> up -> down, exactly N each
        samples_per = target // len(BUCKETS)
        target_bucket = next(
            (b for b in BUCKETS if bucket_counts[b] < samples_per),
            BUCKETS[-1],
        )

        guidance_text = GUIDANCE[target_bucket]
        total = sum(bucket_counts.values())
        status = " | ".join(f"{b}:{bucket_counts[b]}" for b in BUCKETS)

        if len(detections) == 1:
            x, y, w, h = detections[0]["bbox"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            crop = frame[y : y + h, x : x + w]
            ok, reason = is_sample_acceptable(crop)
            if not ok:
                logger.debug("Enrollment rejected: %s", reason)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            landmarks_list = face_recognition.face_landmarks(rgb)
            pose_bucket = estimate_pose_bucket(landmarks_list[0]) if landmarks_list else None

            # Show current detected pose
            pose_hint = f"Pose: {pose_bucket or '?'}"
            if pose_bucket == target_bucket and ok:
                pose_hint += " [OK - hold steady]"

            cv2.putText(display, guidance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"{total}/{target}  {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, pose_hint, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Auto-capture when: quality OK, pose matches target, cooldown passed, bucket not full
            now = time.monotonic()
            if (
                ok
                and pose_bucket == target_bucket
                and (now - last_capture_time) >= CAPTURE_COOLDOWN_SEC
                and bucket_counts[target_bucket] < samples_per
            ):
                encodings = face_recognition.face_encodings(rgb)
                if not encodings:
                    logger.debug("Enrollment rejected: could not extract embedding (face landmarks?)")
                elif encodings:
                    new_enc = encodings[0]
                    # Reject duplicate
                    if embeddings_list:
                        dists = face_recognition.face_distance(embeddings_list, new_enc)
                        min_dist = float(np.min(dists))
                        if min_dist < MIN_EMBEDDING_DISTANCE:
                            logger.debug(
                                "Enrollment rejected: duplicate (min_distance=%.3f < %.2f)",
                                min_dist,
                                MIN_EMBEDDING_DISTANCE,
                            )
                        else:
                            embeddings_list.append(new_enc)
                            bucket_counts[target_bucket] += 1
                            last_capture_time = now
                            print(f"  Captured {sum(bucket_counts.values())}/{target}: {target_bucket}")
                    else:
                        embeddings_list.append(new_enc)
                        bucket_counts[target_bucket] += 1
                        last_capture_time = now
                        print(f"  Captured 1/{target}: {target_bucket}")
        elif len(detections) > 1:
            cv2.putText(display, "Only one face please", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, f"{total}/{target}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(display, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, guidance_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, f"{total}/{target}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Enrollment", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Cancelled.")
            mgr.close()
            detector.close()
            cv2.destroyAllWindows()
            sys.exit(0)

    mgr.close()
    detector.close()
    cv2.destroyAllWindows()

    if not embeddings_list:
        print("Error: No embeddings captured")
        sys.exit(1)

    embeddings_array = np.array(embeddings_list)
    embedding_file = f"{user_id}.npy"
    emb_path = profiles_dir / embedding_file
    np.save(emb_path, embeddings_array)

    profiles_path = profiles_dir / "profiles.json"
    if profiles_path.exists():
        with open(profiles_path) as f:
            data = json.load(f)
    else:
        data = {"users": []}

    users = data.get("users", [])
    existing = next((u for u in users if u.get("user_id") == user_id), None)
    if existing:
        existing["embedding_file"] = embedding_file
        existing["name"] = name
    else:
        users.append({"user_id": user_id, "name": name, "embedding_file": embedding_file})
    data["users"] = users

    with open(profiles_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nEnrollment complete. Saved to {profiles_dir}")
    print(f"  - {embedding_file} ({len(embeddings_list)} embeddings)")
    print(f"  - Buckets: {bucket_counts}")
    print(f"  - profiles.json")


if __name__ == "__main__":
    main()
