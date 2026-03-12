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
| Face recognition | Placeholder |
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
│   ├── face_recognizer.py   # Placeholder
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
│   └── test_app_controller.py
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

Or:

```bash
python scripts/run_local.py
```

A live webcam window opens with face bounding boxes. Press **q** to quit cleanly.

If the camera fails to open, the app exits with an error message. Ensure no other app is using the webcam.

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

### Face detection tuning

Distant or small faces may be harder to detect with lightweight real-time models. Tune via `.env`:

- `FACE_DETECTION_CONFIDENCE` (default 0.4): Lower = more sensitive, more false positives
- `FACE_DETECTION_MODEL`: 0 = short-range (2m), 1 = full-range (5m, better for distant faces)

## Raspberry Pi Deployment

For deployment on Raspberry Pi 5:

1. Use `opencv-python-headless` instead of `opencv-python` if you have no display
2. Ensure camera is enabled (e.g. `raspi-config` → Interface → Camera)
3. Set `CAMERA_INDEX` for Pi camera module (often `0`)

## Tech Stack

- Python, OpenCV, NumPy, MediaPipe
- python-dotenv, logging
- Optional: OpenAI, ElevenLabs, n8n

### MediaPipe version

This project pins **MediaPipe 0.10.13** because it uses the legacy Solutions API (`mp.solutions.face_detection`). Newer MediaPipe versions removed that interface in favor of the Tasks API. Do not upgrade MediaPipe without migrating the face detector code.

## License

MIT
