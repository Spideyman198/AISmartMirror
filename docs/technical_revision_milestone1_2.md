# AISmartMirror: Full Technical Revision (Milestone 1 & 2)

A structured revision for capstone discussion, viva, or technical presentation. Based on the actual codebase.

---

## Changelog (Updates Over Time)

*This section is updated whenever features are added, fixed, or changed. Read it to see everything that has been done since the initial revision.*

### Recognition smoothing (post–Milestone 2)

- **Added:** `vision/recognition_smoother.py` — module that requires N consecutive identical recognition results before confirming a displayed identity.
- **Config:** `RECOGNITION_CONFIRMATION_COUNT` (default 2) — identity shown only after 2 consecutive matches.
- **Effect:** Reduces flicker between "Alice" / "Unknown" when raw result oscillates near threshold.
- **Applied to:** Both known and unknown labels.

### Optional on-screen debug info

- **Added:** `DEBUG_RECOGNITION` in `.env` — when true, shows per face: `d=` best distance, `t=` threshold, `state=` "confirmed" or "pending 2/3".
- **Updated:** `vision/display.py` — accepts `debug_infos` parameter.
- **Updated:** `RecognitionResult` — added `best_match_name` for debug when match is rejected.

### Logging improvements

- **Recognition:** `logger.debug` for accepted/rejected matches, empty image, failed encoding, no enrolled users.
- **Enrollment:** `logger.debug` for quality rejections (size, blur), duplicates, failed embedding extraction.
- **Use:** Set `LOG_LEVEL=DEBUG` in `.env` to see these logs.

### New config options

| Option | Default | Purpose |
|--------|---------|---------|
| `RECOGNITION_CONFIRMATION_COUNT` | 2 | Consecutive matches required before confirming identity |
| `DEBUG_RECOGNITION` | false | Show distance, threshold, state on screen |

---

## 1. Milestone 1 Summary

### What Milestone 1 Does

Milestone 1 implements **real-time face detection** from a webcam. It finds faces in each frame, draws bounding boxes around them, and displays the result. It does **not** identify who the person is—only that a face is present and where it is.

### Files/Modules Implemented

| File | Purpose |
|------|---------|
| `app/main.py` | Entry point; creates AppController and runs it |
| `app/app_controller.py` | Orchestrates camera, detector, display; runs main loop |
| `vision/camera_manager.py` | Opens webcam, reads frames, handles Windows backends |
| `vision/face_detector.py` | Detects faces using MediaPipe |
| `vision/display.py` | Draws bounding boxes on frames |
| `config/settings.py` | Loads config from `.env` |
| `utils/logger.py` | Logging setup |

### Exact Pipeline: Webcam → Face Detection Output

```
1. main.py
   └── AppController.initialize()
       ├── CameraManager.find_working_camera() or CameraManager(index, backend).open()
       │   └── OpenCV VideoCapture, warm-up (discard ~20 frames), set resolution
       └── FaceDetector(min_confidence, model_selection)
           └── MediaPipe FaceDetection model loaded

2. AppController.run() — main loop (every frame):
   ├── camera_manager.read() → BGR frame (H×W×3) or None
   ├── face_detector.detect(frame) → list of {bbox, confidence}
   ├── display.draw_face_boxes(frame, detections) → frame with green boxes
   └── cv2.imshow() → display window
```

**Data flow:**
- **Input:** BGR image from OpenCV `VideoCapture.read()`
- **Detection:** MediaPipe processes RGB copy; returns normalized bounding boxes
- **Output:** List of dicts: `{"bbox": (x, y, w, h), "confidence": float}`

### Libraries/Models/Tools Used

| Component | Library/Model | Version |
|-----------|---------------|---------|
| Camera | OpenCV (`cv2.VideoCapture`) | opencv-python ≥4.8 |
| Face detection | MediaPipe Face Detection | 0.10.13 (pinned) |
| Config | python-dotenv | ≥1.0 |
| Numerics | NumPy | ≥1.24 |

### Why Those Choices Were Made

- **MediaPipe:** Lightweight, runs on CPU, suitable for Raspberry Pi; real-time performance.
- **MediaPipe 0.10.13:** Uses legacy Solutions API (`mp.solutions.face_detection`); newer versions use Tasks API and would require code changes.
- **OpenCV for camera:** Standard, cross-platform, supports Windows backends (DSHOW, MSMF).
- **Windows backends:** Default backend often returns black frames; DSHOW/MSMF are more reliable.

### Technical Problems Solved

1. **Black screen on Windows:** Open camera before loading MediaPipe; use explicit backend (DSHOW/MSMF); warm-up by discarding initial frames.
2. **Buffer size:** Set `CAP_PROP_BUFFERSIZE=1` to reduce stale frames.
3. **Camera discovery:** `find_working_camera()` tries multiple indexes and backends until one works.
4. **Normalized coordinates:** MediaPipe returns 0–1; converted to pixels with frame width/height and clamped to frame bounds.

### Current Limitations of Milestone 1

- No identity: only detects presence and location of faces.
- MediaPipe tuned for frontal/near-frontal faces; profile views may be missed.
- Distant/small faces need `model_selection=1` (full-range).
- Confidence threshold trades sensitivity vs. false positives.

---

## 2. Milestone 2 Summary

### What Milestone 2 Adds

Milestone 2 adds **face recognition**: known vs. unknown. For each detected face, it computes an embedding, compares it to enrolled users, and labels the face with a name or "Unknown".

### Pipeline: Detected Face → Known/Unknown

```
For each detection (when recognition runs):
1. Crop face from frame: frame[y:y+h, x:x+w]
2. FaceRecognizer.recognize(crop)
   ├── _preprocess: BGR → RGB
   ├── face_recognition.face_encodings(rgb) → 128-dim embedding
   ├── face_recognition.face_distance(all_stored_embeddings, live_embedding)
   ├── argmin → best match
   └── if best_distance <= threshold → known (name); else → unknown
3. Label shown above bbox: name or "Unknown"
```

### How Enrollment Works

1. Run `python scripts/enroll_user.py --name Alice`
2. Guided auto-scanner:
   - Prompts in order: center (4) → left (4) → right (4) → up (4) → down (4)
   - Detects face, estimates pose from landmarks
   - Auto-captures when pose matches target, quality OK, cooldown passed
   - Quality: min 64×64 px, Laplacian variance ≥ 80 (blur check)
   - Duplicate rejection: embedding distance ≥ 0.03 to existing
3. Saves `{user_id}.npy` (shape N×128) and updates `profiles.json`

### How Embeddings Are Stored

- **Location:** `data/known_faces/`
- **Files:**
  - `profiles.json`: `{"users": [{"user_id": "alice", "name": "Alice", "embedding_file": "alice.npy"}]}`
  - `alice.npy`: NumPy array shape `(20, 128)` — 20 embeddings, 128 dimensions each
- **Format:** No averaging; each embedding stored separately for best-match logic.

### How Matching Works

1. Load all user embeddings from `.npy` files.
2. Flatten to one list: `[e for user_embs in self._embeddings for e in user_embs]`
3. Track which embedding belongs to which user via `user_indices`.
4. `face_recognition.face_distance(all_embeddings, live_encoding)` → Euclidean distances
5. `argmin(distances)` → best match
6. Map best index back to user; if `distance <= threshold` → known.

### How Thresholding Works

- **Threshold:** Max Euclidean distance for a match (default 0.6).
- **Lower (e.g. 0.5):** Stricter — fewer false positives, more misses at angles.
- **Higher (e.g. 0.7):** Lenient — better at angles, more false positives.
- **Config:** `FACE_RECOGNITION_THRESHOLD` in `.env`.

### How Recognition Interval / Frame Skipping Works

- **Detection:** Every frame (MediaPipe is fast).
- **Recognition:** Every N frames (`FACE_RECOGNITION_INTERVAL_FRAMES`, default 5).
- **Logic:** `run_recognition = (frame_count % interval == 0) or (face_count != len(last_labels))`
- **Reuse:** When not running recognition, `labels = last_labels` (if face count unchanged).
- **Effect:** Less CPU use, smoother display; labels update every N frames.

### How Recognition Smoothing Works

- **Problem:** Raw result can flicker between "Alice" and "Unknown" when distance oscillates near threshold.
- **Solution:** `RecognitionSmoother` requires N consecutive identical results before confirming.
- **Per-face state:** `(last_seen, streak, confirmed)`. When raw label matches `last_seen`, `streak++`. Else reset to new label, `streak=1`. When `streak >= CONFIRMATION_COUNT`, `confirmed = last_seen`.
- **Display:** Always show `confirmed`; never show raw result directly.
- **Applied to:** Both known and unknown labels.

### Current Limitations of Milestone 2

- dlib trained mainly on frontal faces; up/down and profile poses are weaker.
- Pose estimation is heuristic (2D landmarks); no 3D pose.
- Threshold is a global tradeoff; no per-user tuning.
- Enrollment requires exactly one face; multi-face enrollment not supported.

---

## 3. Models / Algorithms Used

### Camera Capture

| Attribute | Detail |
|-----------|--------|
| **Name** | OpenCV `VideoCapture` |
| **What it does** | Opens camera device, reads BGR frames |
| **Input** | Device index, optional backend (DSHOW, MSMF) |
| **Output** | BGR numpy array (H×W×3), or None on failure |
| **Why selected** | Standard, cross-platform, backend control on Windows |
| **Tradeoffs** | Backend choice matters on Windows; default can fail |

### Face Detection

| Attribute | Detail |
|-----------|--------|
| **Name** | MediaPipe Face Detection |
| **What it does** | Detects faces, returns bounding boxes and confidence |
| **Input** | RGB image |
| **Output** | List of detections: normalized bbox (0–1), score |
| **Why selected** | Lightweight, CPU-friendly, real-time, Raspberry Pi compatible |
| **Tradeoffs** | Less accurate on profile/extreme angles; model 0 vs 1 (range) |

### Face Recognition (Embeddings)

| Attribute | Detail |
|-----------|--------|
| **Name** | face_recognition (dlib ResNet) |
| **What it does** | Produces 128-dim embedding from face image |
| **Input** | RGB face crop |
| **Output** | 128-dim float vector (embedding) |
| **Why selected** | Local, no API; good for known vs unknown |
| **Tradeoffs** | Needs dlib (build or dlib-bin); frontal bias |

### Embeddings / Feature Vectors

| Attribute | Detail |
|-----------|--------|
| **Name** | 128-dimensional face embedding |
| **What it does** | Encodes face identity into a vector; similar faces → similar vectors |
| **Input** | Face image (from face_recognition) |
| **Output** | NumPy array shape (128,) |
| **Why selected** | Compact, comparable via Euclidean distance |
| **Tradeoffs** | Sensitive to pose/lighting; not robust to large pose changes |

### Pose Estimation (Enrollment)

| Attribute | Detail |
|-----------|--------|
| **Name** | Heuristic 2D pose from landmarks |
| **What it does** | Buckets pose into center/left/right/up/down from face landmarks |
| **Input** | `face_recognition.face_landmarks()` dict |
| **Output** | String: "center", "left", "right", "up", "down" |
| **Why selected** | Simple, no extra model; good enough for guided enrollment |
| **Tradeoffs** | Approximate; camera-dependent; up/down labels swapped in our impl |

### Quality Checks

| Attribute | Detail |
|-----------|--------|
| **Name** | Rule-based checks |
| **What it does** | Rejects too-small or blurry crops |
| **Input** | Face crop (BGR) |
| **Output** | (ok: bool, reason: str) |
| **Checks** | Min 64×64 px; Laplacian variance ≥ 80 (blur) |
| **Tradeoffs** | Simple; no semantic quality model |

### Recognition Matching Logic

| Attribute | Detail |
|-----------|--------|
| **Name** | Euclidean distance + threshold |
| **What it does** | Compares live embedding to all stored; best match wins |
| **Input** | Live embedding, list of stored embeddings |
| **Output** | RecognitionResult (is_known, name, distance, similarity) |
| **Algorithm** | `face_distance()` → argmin → threshold check |
| **Tradeoffs** | O(n) in total embeddings; threshold is global |

---

## 4. Important Technical Concepts

### Face Detection vs Face Recognition

- **Detection:** "Is there a face? Where?" — outputs bounding boxes. No identity.
- **Recognition:** "Whose face is it?" — outputs identity (name or unknown). Needs detection first.

### Embedding

A **128-dimensional vector** that represents a face. Produced by a neural network (dlib). Similar faces → small Euclidean distance; different faces → larger distance. Used instead of raw pixels for comparison.

### Feature Vector

Same idea as embedding: a numeric vector that encodes meaningful information. In this project, "embedding" and "feature vector" are used interchangeably for the 128-dim face representation.

### Known vs Unknown

- **Known:** Live embedding matches a stored user (distance ≤ threshold) → show name.
- **Unknown:** No match above threshold → show "Unknown". Includes strangers and enrolled users at bad angles.

### Threshold

Max Euclidean distance for a match. Lower = stricter (fewer false matches, more misses). Higher = lenient (better at angles, more false matches). Typical range 0.5–0.7.

### Why Frontal Faces Work Better

dlib is trained mostly on frontal faces. Non-frontal poses change visible features (cheeks, nose, jaw), so the embedding moves in vector space and distance increases. Multi-angle enrollment helps by storing several poses per user.

### Why the System Can Be Laggy and How Interval Helps

Recognition (embedding + matching) is heavier than detection. Running it every frame can cause lag. Running it every N frames and reusing the last result keeps detection smooth while reducing CPU load.

### Why Multiple Embeddings Per User Are Better

One embedding is one point in 128-D space. Non-frontal poses land elsewhere. Multiple embeddings (e.g. center, left, right, up, down) give several reference points, so a live face at any of those poses can find a close match.

---

## 5. Capstone Defense Questions

### Q: What model are you using for face detection?

**Short:** MediaPipe Face Detection.

**Longer:** MediaPipe Face Detection, part of Google’s MediaPipe framework. It’s a lightweight CNN for real-time use. We use the legacy Solutions API (MediaPipe 0.10.13). It returns bounding boxes and confidence. We use `model_selection=1` (full-range, 5 m) for better detection of distant faces.

---

### Q: What model/approach are you using for face recognition?

**Short:** The face_recognition library, which uses dlib’s ResNet-based face encoder to produce 128-dim embeddings. We match by Euclidean distance against locally stored embeddings.

**Longer:** We use the `face_recognition` Python library, which wraps dlib’s deep metric learning model. It outputs 128-dim embeddings. We store embeddings per user (up to 20 from different poses), compare the live embedding to all stored ones with Euclidean distance, and label as known if the best distance is below a configurable threshold.

---

### Q: What are embeddings / feature vectors?

**Short:** 128-dimensional vectors that encode a face. Similar faces have similar vectors; we compare them with Euclidean distance.

**Longer:** Embeddings are 128-D float vectors from a neural network. They capture identity-related features. We don’t compare raw pixels; we compare these vectors. `face_recognition.face_distance()` returns Euclidean distance. Lower distance = more similar. We use a threshold (default 0.6) to decide known vs unknown.

---

### Q: How many embeddings do you store per user and why?

**Short:** 20 (4 per pose: center, left, right, up, down).

**Longer:** We store 20 embeddings per user, 4 per pose bucket. dlib is biased toward frontal faces. Storing multiple poses gives more reference points in embedding space, so recognition works better when the user isn’t looking straight. We use a guided enrollment that collects these poses in a fixed order.

---

### Q: Why does recognition fail sometimes for up/down poses?

**Short:** dlib is trained mainly on frontal faces, so embeddings for up/down poses are less reliable.

**Longer:** The dlib model is trained mostly on frontal/near-frontal data. For up/down, visible features change (chin, forehead, nose), so the embedding shifts. Even with multi-angle enrollment, the model itself is less robust to large pitch changes. We improve it by enrolling up/down poses, but the underlying model limits how well it works.

---

### Q: Why is edge processing important here?

**Short:** Privacy and latency: face data stays on the device; no network delay.

**Longer:** Edge processing means running on the device (e.g. Raspberry Pi) instead of the cloud. Benefits: (1) Privacy — biometric data never leaves the device. (2) Latency — no network round-trip. (3) Offline use — works without internet. (4) Cost — no cloud API fees. We use local models (MediaPipe, dlib) for detection and recognition.

---

### Q: Why did you choose this design for Raspberry Pi?

**Short:** MediaPipe and dlib run on CPU, are lightweight, and don’t need a GPU.

**Longer:** MediaPipe and dlib are CPU-oriented and suitable for Raspberry Pi 5. We avoid heavy models that need GPUs. The pipeline is modular (camera → detection → recognition → display) so we can swap components. Config via `.env` allows tuning without code changes. We also use `opencv-python-headless` on Pi when there’s no display.

---

### Q: Why do you not send face data to the cloud?

**Short:** Privacy and the project’s edge-first design.

**Longer:** Face images and embeddings are biometric data. Sending them to the cloud raises privacy and compliance issues. Our design is edge-first: detection and recognition run locally. Only optional features (e.g. voice, notifications) may use the cloud, and those don’t receive face data.

---

### Q: Why is recognition not run every frame?

**Short:** Recognition is heavier than detection; running it every N frames reduces lag while keeping detection smooth.

**Longer:** Detection (MediaPipe) is fast. Recognition (dlib embedding + distance computation) is slower. Running recognition every frame can cause noticeable lag. We run it every N frames (default 5) and reuse the last labels in between. Detection still runs every frame for responsive boxes; only the labels are throttled.

---

## 6. File-by-File Revision

### app/main.py

**What it does:** Entry point. Sets up logging, creates `AppController`, calls `initialize()` and `run()`.

**Why it matters:** Single entry point; delegates all logic to `AppController`.

---

### app/app_controller.py

**What it does:** Orchestrates the app. Initializes camera (with backend selection), face detector, face recognizer. Main loop: read frame → detect → recognize (every N frames) → draw → display. Handles shutdown.

**Why it matters:** Central controller; contains the main loop and recognition interval logic.

---

### vision/camera_manager.py

**What it does:** Manages webcam via OpenCV. Supports DSHOW and MSMF on Windows. `find_working_camera()` tries indexes and backends. Warm-up discards initial frames. `read()` returns BGR frame or None.

**Why it matters:** Abstracts camera setup and fixes Windows black-frame issues.

---

### vision/face_detector.py

**What it does:** Wraps MediaPipe Face Detection. `detect(frame)` returns list of `{bbox, confidence}`. Converts MediaPipe’s normalized coords to pixels and clamps to frame.

**Why it matters:** Single interface for detection; output format used by recognition and display.

---

### vision/face_recognizer.py

**What it does:** Loads profiles and embeddings from `data/known_faces/`. `recognize(face_crop)` returns `RecognitionResult`. Encodes crop to 128-dim, compares to all stored embeddings, returns best match if within threshold.

**Why it matters:** Implements known vs unknown logic; handles multi-embedding matching.

---

### vision/display.py

**What it does:** `draw_face_boxes(frame, detections, labels, debug_infos)` draws green boxes, optional labels, and optional debug info (distance, threshold, state) per face.

**Why it matters:** Keeps display logic separate from detection/recognition.

---

### vision/recognition_smoother.py

**What it does:** `RecognitionSmoother(confirmation_count)`. `update(results, threshold, show_debug)` returns smoothed labels and optional debug info. Requires N consecutive identical results before confirming identity.

**Why it matters:** Reduces flicker; improves user experience.

---

### scripts/enroll_user.py

**What it does:** Guided auto-enrollment. Uses camera and detector; estimates pose from landmarks; auto-captures when pose matches target and quality checks pass. Saves embeddings to `{user_id}.npy` and updates `profiles.json`.

**Why it matters:** Produces the data used by the recognizer; implements pose estimation and quality checks.

---

### config/settings.py

**What it does:** Loads config from `.env` via python-dotenv. Exposes camera, detection, recognition, and optional API settings. Uses `@lru_cache` for `get_settings()`.

**Why it matters:** Central config; allows tuning without code changes.

---

### Storage / Profile Files

- **profiles.json:** List of users with `user_id`, `name`, `embedding_file`.
- **{user_id}.npy:** NumPy array shape `(N, 128)` — N embeddings per user.
- **Location:** `data/known_faces/` (or `KNOWN_FACES_DIR`).

---

## 7. What to Memorize

### 10 Most Important Facts

1. **Milestone 1** = face detection only (where, not who).
2. **Milestone 2** = face recognition (known vs unknown).
3. **MediaPipe** for detection; **face_recognition (dlib)** for embeddings.
4. **Embeddings** are 128-dim vectors; compared with Euclidean distance.
5. **20 embeddings per user** (4 per pose) for better pose robustness.
6. **Threshold** (default 0.6): lower = stricter, higher = lenient.
7. **Recognition every N frames** (default 5) to reduce lag.
8. **Recognition smoothing** requires N consecutive matches before confirming (reduces flicker).
9. **All processing is local** — no face data sent to the cloud.
10. **Windows:** Use DSHOW or MSMF backend; warm-up frames to avoid black screen.

### 5 Most Important Technical Terms

1. **Embedding** — 128-dim vector representing a face; used for comparison.
2. **Euclidean distance** — Metric for comparing embeddings; lower = more similar.
3. **Threshold** — Max distance for a match; controls strictness.
4. **Face detection** — Finding face locations (bboxes); no identity.
5. **Face recognition** — Identifying who the face belongs to (known vs unknown).

### 5 Mistakes to Avoid in a Presentation

1. **Don’t say** "We use deep learning for everything" — Pose estimation is heuristic, not a learned model.
2. **Don’t say** "Recognition runs every frame" — It runs every N frames for performance.
3. **Don’t say** "We use one embedding per user" — We use up to 20 per user.
4. **Don’t say** "The system works perfectly at all angles" — dlib is biased toward frontal faces.
5. **Don’t say** "We use the cloud for face recognition" — All face processing is local/edge.
