# Summary: Face Recognition Enrollment Improvements

## Overview

The enrollment flow was changed from manual capture to a guided auto-scanner that collects 20 samples across five poses in a fixed order.

---

## 1. Enrollment Flow: Manual → Guided Auto-Scanner

**Before:** User pressed SPACE to capture 5 samples (center, left, right, up, down).

**After:**

- Automatic capture when pose matches the current target and quality checks pass
- Fixed order: center (4) → left (4) → right (4) → up (4) → down (4)
- No time-based switching; prompt changes only when the current pose has 4 samples
- Cooldown of 0.6 s between captures to avoid duplicates

---

## 2. Pose Estimation

**Method:** 2D heuristics from `face_recognition.face_landmarks()`:

**Yaw (left/right):** Nose horizontal offset vs. eye center

- `|yaw| < 0.15` → center
- `yaw > 0.12` → left
- `yaw < -0.12` → right

**Pitch (up/down):** `chin_to_nose / nose_to_eye`

- Smaller ratio → chin closer to nose → looking down
- Larger ratio → chin farther from nose → looking up
- `pitch_ratio < 1.1` → down
- `pitch_ratio > 2.0` → up
- `1.1 < pitch_ratio < 2.2` → center

**Correction:** Up/down labels were swapped because the camera's view inverted them.

---

## 3. Quality Checks

- **Size:** Reject crops smaller than 64×64 px
- **Blur:** Reject if Laplacian variance < 80
- **Duplicates:** Reject if embedding distance < 0.03 to any existing sample

---

## 4. On-Screen Feedback

- Current prompt (e.g. "Look straight at camera")
- Progress (e.g. `12/20  center:4 left:4 right:2 up:0 down:0`)
- Detected pose (e.g. `Pose: center [OK - hold steady]`)

---

## 5. Storage and Recognition

- **Storage:** Same as before: `{user_id}.npy` with shape `(N, 128)`, `profiles.json`
- **Recognition:** Still compares against all stored embeddings and uses the best match
- **Threshold:** Configurable via `FACE_RECOGNITION_THRESHOLD` in `.env`
- **Lag:** Recognition runs every N frames (`FACE_RECOGNITION_INTERVAL_FRAMES`), reuses last result between runs

---

## 6. Files Touched

| File | Changes |
|------|---------|
| `scripts/enroll_user.py` | Guided auto-enrollment, pose estimation, quality checks, strict pose order |
| `vision/face_recognizer.py` | Multi-embedding support, threshold docstrings |
| `app/app_controller.py` | Recognition interval, reuse of last result |
| `config/settings.py` | `FACE_RECOGNITION_INTERVAL_FRAMES` |
| `tests/test_enroll_user.py` | Tests for quality checks and pose estimation |
| `README.md` | Auto-enrollment flow, limitations, manual testing steps |
| `.env.example` | New config options |

---

## 7. Current Behavior

1. Run `python scripts/enroll_user.py --name YourName`
2. Follow prompts in order: center → left → right → up → down
3. Hold each pose until 4 samples are captured
4. Samples are auto-captured when pose matches, quality is good, and cooldown has passed
5. Press Q to cancel
