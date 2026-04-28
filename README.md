# AISmartMirror

A professional capstone project: an AI-powered smart mirror with face detection, face recognition, gesture recognition, voice commands, and conversational AI. Built for Raspberry Pi 5 with development support on standard laptops.

## Overview

AISmartMirror uses an **edge-first** architecture:

- **Local (edge)**: Camera, face detection, face recognition, gesture recognition, system logic
- **Cloud**: Advanced speech-to-text, conversational AI, text-to-speech, notifications

## Features Roadmap

| Feature | Status |
|---------|--------|
| Face detection | Implemented (MediaPipe) |
| Face recognition | Implemented (local, known vs unknown) |
| Gesture recognition | Placeholder |
| Voice command pipeline | Placeholder |
| Conversational AI (OpenAI) | Placeholder |
| Smart mirror dashboard | Placeholder |
| Automation & notifications | Placeholder |

## Folder Structure

```
AISmartMirror/
├── app/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   └── app_controller.py     # Orchestrates modules
├── vision/
│   ├── __init__.py
│   ├── camera_manager.py    # Webcam init, frame capture
│   ├── face_detector.py     # MediaPipe face detection
│   ├── display.py           # Draw detection boxes
│   ├── face_recognizer.py   # Local embedding-based recognition
│   └── gesture_recognizer.py
├── audio/
│   ├── speech_to_text.py
│   ├── text_to_speech.py
│   └── voice_assistant.py
├── ui/
│   └── dashboard.py
├── integrations/
│   ├── openai_client.py
│   ├── elevenlabs_client.py
│   └── notifier.py
├── config/
│   └── settings.py
├── utils/
│   ├── logger.py
│   └── helpers.py
├── scripts/
│   └── run_local.py
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_smoke.py        # Smoke tests
│   ├── test_config.py       # Config tests
│   ├── test_camera.py       # Camera tests (mocked)
│   ├── test_face_detector.py
│   ├── test_app_controller.py
│   └── test_face_recognizer.py
├── docs/
│   └── architecture.md
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Enter project directory

```bash
cd AISmartMirror
```

### 2. Create and activate virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure (optional)

```bash
cp .env.example .env
# Edit .env if needed; app runs without it
```

## Run

From project root with venv activated:

```bash
python -m app.main
```

Startup menu options include:
- Start Smart Mirror
- Enroll New User
- Test Face Recognition
- Confirm New Users / Retrain CNN
- Exit

Or:

```bash
python scripts/run_local.py
```

A live webcam window opens with face bounding boxes and recognition labels (known user name or "Unknown"). Press **q** to quit cleanly.

If the camera fails to open, the app exits with an error message. Ensure no other app is using the webcam.

### Face recognition (known vs unknown)

1. **Enroll users** (one-time per user):
   ```bash
   python scripts/enroll_user.py --name Alice
   ```
   Guided auto-enrollment: follow the on-screen prompts (Look straight, Turn left, etc.). By default, capture is **prompt-based** and triggers when quality is good (face present, large enough, sharp enough, cooldown, non-duplicate). Pose detection is shown as feedback and is not required unless you enable strict mode. Profiles saved to `data/known_faces/`. Use `--samples 15` to capture fewer.

2. **Run live recognition**:
   ```bash
   python -m app.main
   ```
   Enrolled users see their name; others see "Unknown".

3. **Tune in `.env`**:
   - `FACE_RECOGNITION_THRESHOLD=0.6` — lower = stricter (fewer false matches), higher = lenient (better at angles)
   - `FACE_RECOGNITION_INTERVAL_FRAMES=5` — run recognition every N frames (higher = less lag)
   - `RECOGNITION_CONFIRMATION_COUNT=2` — require N consecutive matches before confirming (reduces flicker)
   - `DEBUG_RECOGNITION=true` — show distance, threshold, and state on screen

**Troubleshooting camera issues on Windows:** Run `python scripts/test_camera.py` to find a working index/backend combination. Set `CAMERA_BACKEND=DSHOW` or `CAMERA_BACKEND=MSMF` in `.env` if needed.

## Testing

### Run tests

From project root:

```bash
pytest tests/ -v
```

Run only fast tests (excludes hardware):

```bash
pytest tests/ -v -m "not hardware"
```

Run hardware tests (requires camera):

```bash
pytest tests/ -v -m hardware
```

### Health check

Verify the environment is ready:

```bash
python scripts/health_check.py
```

Skip camera check (e.g. in CI or without webcam):

```bash
python scripts/health_check.py --skip-camera
```

### Manual webcam face detection test

1. Run the app: `python -m app.main`
2. A window opens showing the webcam feed
3. Point the camera at a face — green boxes appear around detected faces
4. Press **q** to quit

### Face recognition limitations

- **Lightweight model**: Uses face_recognition (dlib) for 128-dim embeddings. Good for known vs unknown; not designed for large-scale identification.
- **Frontal faces work better**: The dlib model is trained primarily on frontal faces. When you look left, right, up, or down, the face shape and visible features change, so the embedding differs from enrollment. **Multi-angle enrollment** improves robustness.
- **Lighting/angle**: Recognition accuracy depends on similar lighting and pose to enrollment.
- **No cloud**: All processing is local. Biometric data never leaves the device.

### Guided auto-enrollment flow

The enrollment script uses a **guided auto-scanner** instead of manual capture:

1. **On-screen guidance** cycles through poses: Look straight, Turn left, Turn right, Look up, Look down.
2. **Automatic capture (default)** is prompt-based: for each target prompt, samples are captured when quality checks pass (face detected, minimum size, blur gate, cooldown, duplicate rejection).
3. **Pose detection feedback** is shown on screen (`Pose detected: center/left/right/up/down`) but does not block capture in default mode.
4. **Strict pose mode (optional)** can be enabled with `ENROLL_REQUIRE_POSE_MATCH=true` if you want target-pose matching to be required.
5. **Anti-stuck safeguards**: each pose has a timeout (`ENROLL_POSE_TIMEOUT_SEC`), and you can press `N` to skip to the next pose.
6. **Progress** shows overall samples and per-pose counts.

Why this default: strict 2D pose heuristics can be fragile in real webcam conditions (lighting, motion blur, occlusion), so strict matching may stall enrollment even when the user follows prompts.

### Hybrid recognition workflow (recommended)

AISmartMirror now uses a **hybrid runtime**:

- **Primary**: CNN recognizer (fast, stable, class-based)
- **Fallback**: embedding recognizer for CNN-Unknown faces

This gives the best of both:
- CNN remains the main recognizer during normal operation.
- Newly enrolled users can be recognized immediately (via embedding fallback) before CNN retraining.

When enrollment adds a user, the system marks `data/cnn_models/model_status.json` as:
- `cnn_outdated=true`

That flag is a reminder that CNN should be retrained on the latest dataset.

### Why multiple embeddings help

The recognizer compares the live face against **all** stored embeddings per user and uses the **best match** (lowest distance). One frontal embedding only matches frontal poses well. Storing embeddings from different angles (straight, slight left/right/up/down) and distances gives the recognizer more "reference points" in embedding space, so a live face at any of those poses is more likely to find a close match. ~20 high-quality samples is a good balance: enough variety for robustness without excessive storage or matching cost.

### Why poor-quality embeddings hurt

Blurry or very small face crops produce noisy embeddings that don't represent the person well. Including them in the stored set can:
- **Dilute good matches**: A live face may match a blurry embedding by chance, or fail to match because the blurry embedding is an outlier.
- **Increase false positives**: Noisy embeddings can accidentally match strangers.
- **Waste compute**: More embeddings mean more distance computations per frame.

The enrollment script rejects samples that are too small (<64px) or too blurry (low Laplacian variance) to keep the stored set high quality.

### How recognition interval improves responsiveness

Recognition (embedding + matching) is heavier than detection. Running it every frame can cause lag. The app runs **detection every frame** (fast) but **recognition every N frames** (configurable via `FACE_RECOGNITION_INTERVAL_FRAMES`). Between recognition updates, the last result is reused. This keeps the display responsive while still updating labels regularly. Typical values: 5 (default) for smooth updates, 10 for lower CPU use.

### How recognition smoothing works

The displayed identity can flicker when the raw recognition result oscillates near the threshold (e.g. frame 1: "Alice", frame 2: "Unknown", frame 3: "Alice"). **Recognition smoothing** requires a label to appear **consecutively** across multiple recognition cycles before confirming it. For example, with `RECOGNITION_CONFIRMATION_COUNT=2`, "Alice" is shown only after 2 consecutive recognition runs return "Alice". A single "Unknown" resets the streak. This reduces flicker and improves user experience. Both known and unknown classifications are stabilized the same way.

### Limitations for non-frontal poses

The dlib model is trained primarily on frontal faces. Recognition at up/down or side angles remains weaker than frontal, even with multi-angle enrollment. Extreme poses may still fail. Tuning the threshold (higher = more lenient) can help at the cost of more false positives.

### Testing enrollment and recognition manually

1. **Enrollment**: Run `python scripts/enroll_user.py --name YourName`. Follow the prompts (Look straight, Turn left, etc.). The scanner collects ~20 samples across poses. Press Q to cancel.
2. **Recognition**: Run `python -m app.main`. Your enrolled face should show your name. Try different poses (straight, slight angles) to verify robustness.
3. **Re-enroll** if recognition is weak: delete `data/known_faces/yourname.npy` and run enrollment again with better lighting and varied poses.

### Face detection tuning

Distant or small faces may be harder to detect with lightweight real-time models. Tune via `.env`:

- `FACE_DETECTION_CONFIDENCE` (default 0.4): Lower = more sensitive, more false positives
- `FACE_DETECTION_MODEL`: 0 = short-range (2m), 1 = full-range (5m, better for distant faces)

### CNN face recognizer (optional)

A second recognition module using a lightweight **MobileNetV2** classifier. Trained on your face crops; outputs class labels directly. Does not replace the baseline embedding-based recognizer.

**Install CNN dependencies:**
```bash
pip install -r requirements-cnn.txt
```

**1. Collect face data** (run per user):
```bash
python scripts/collect_cnn_faces.py --name alice --target 100
python scripts/collect_cnn_faces.py --name bob --target 100
```

**2. Prepare train/val split:**
```bash
python scripts/prepare_cnn_dataset.py
```

**3. Train:**
```bash
python scripts/train_cnn_recognizer.py --epochs 15
```

**4. Evaluate:**
```bash
python scripts/evaluate_cnn_recognizer.py
```

**5. Live test (webcam):**
```bash
python scripts/test_cnn_live.py
```

**6. Use in app:** Set `CNN_MODEL_DIR=data/cnn_models` in `.env` (or keep default path). The app uses CNN as primary when a trained model exists, with embedding fallback for immediate recognition of newly enrolled users.

### Retraining cadence and trigger

Use periodic retraining (e.g., after batching several new enrollments), not after every single user.

Best workflow from the app menu:
- Choose **Confirm New Users / Retrain CNN**
- The app scans `data/cnn_faces/<user_id>/` and compares to `data/cnn_models/class_mapping.json`
- It reports:
  - users currently in the CNN model
  - new users not yet in the model
  - existing users with increased image counts
- It asks confirmation before retraining
- It backs up the current model/mapping to `data/cnn_models/backups/<timestamp>/`
- Then runs:
  1. `prepare_cnn_dataset.py --clean`
  2. `check_benchmark_leakage.py` (if available)
  3. `train_cnn_recognizer.py` in quick-update mode:
     - 5 epochs
     - lower learning rate (`3e-4`)
     - warm-start from existing `cnn_face_model.pt` when available
  4. `evaluate_cnn_recognizer.py` (if available)
- If any step fails, previous model files are restored/kept from backup.

You can also run it directly:

```bash
python scripts/confirm_new_users.py
```

Legacy command still supported:

```bash
python scripts/retrain_cnn_model.py
```

Both flows retrain on the full dataset and mark CNN status fresh (`cnn_outdated=false`) after success.

### Avoiding catastrophic forgetting

Retraining uses the full accumulated dataset under `data/cnn_faces/` (all users), then rebuilds train/val from all classes.  
Because all previously enrolled users remain in training data, old identities are retained instead of being overwritten by only new-user data.

#### CNN live pipeline: smoothing and stability

- **`vision/cnn_live_pipeline.py`** wraps the CNN with:
  - **Throttled inference:** runs the network every **N** frames (`inference_interval`, default 2). Between runs, the **last stable labels** are reused so the UI stays responsive and CPU use drops.
  - **Confirmation (anti-flicker):** `RecognitionSmoother` requires the **same raw label** (name or Unknown) on **consecutive CNN outputs** before the **displayed** identity updates. So a single noisy frame does not flip the name. Switching to a **new** identity also requires **several consecutive** agreeing frames—this trades a little latency for stability.
- **Raw label** comes from `CNNFaceRecognizer`: known if softmax confidence and optional **class margin** pass; otherwise Unknown.

#### CNN unknown handling and confidence

Two layers reject weak predictions:

1. **Confidence threshold** (`confidence_threshold` on `CNNFaceRecognizer`, CLI `--threshold`): the softmax score of the predicted class must be **≥ this value** or the result is **Unknown**. **Higher** → fewer false IDs (more Unknown). **Lower** → more known hits (more risk of wrong person).
2. **Class margin** (`min_class_margin`, CLI `--margin`): optional **top1 − top2** softmax gap. If the top two classes are both plausible (small gap), the result is **Unknown** even when confidence is high—helps when two enrolled people look similar. **Typical:** `0.0` (off) or try **`0.05`–`0.15`**.

Environment (when wired in app): `CNN_CONFIDENCE_THRESHOLD` in `.env` maps to the same idea as `--threshold`.

#### Tuning the CNN confidence threshold

- Start at **`0.5`**. If you see **wrong names** for strangers or near-twin classmates, **raise** to `0.55`–`0.65` and/or enable **`--margin 0.08`**.
- If enrolled users are often **Unknown** despite being centered, **lower** slightly (e.g. `0.45`) or collect more varied training data.
- **Blur / distance:** enable quality gates in live test: `--min-blur 40` (tune to your camera) and `--min-det 0.5` so bad crops are not classified.

#### Better data collection for the next training round

- **Variety:** multiple distances, slight left/right/up/down, different lighting (day/evening), glasses on/off if relevant.
- **Quantity:** aim for **50+** crops per person; add more for anyone the model confuses.
- **Quality:** avoid extreme blur and tiny faces; the collector already expects a single face and minimum size.
- **Retrain** after adding folders, then run `prepare_cnn_dataset.py` and `train_cnn_recognizer.py` again.

#### Training augmentations (next run)

The trainer already uses flips, rotation, and color jitter. For harder robustness, you can extend `get_train_transform` in `scripts/train_cnn_recognizer.py` with small **affine** jitter, light **blur**, or occasional **grayscale** (see comment in that file).

**Dataset structure:**
```
data/cnn_faces/
  alice/          # From collect_cnn_faces.py
  bob/
  train/          # From prepare_cnn_dataset.py (80%)
    alice/
    bob/
  val/            # 20%
    alice/
    bob/
```

## Raspberry Pi Deployment

For deployment on Raspberry Pi 5:

1. Use `opencv-python-headless` instead of `opencv-python` if you have no display
2. Ensure camera is enabled (e.g. `raspi-config` → Interface → Camera)
3. Set `CAMERA_INDEX` for Pi camera module (often `0`)

## Tech Stack

- Python, OpenCV, NumPy, MediaPipe, face_recognition (dlib)
- python-dotenv, logging
- Optional: OpenAI, ElevenLabs, n8n

### MediaPipe version

This project pins **MediaPipe 0.10.13** because it uses the legacy Solutions API (`mp.solutions.face_detection`). Newer MediaPipe versions removed that interface in favor of the Tasks API. Do not upgrade MediaPipe without migrating the face detector code.

## License

MIT
