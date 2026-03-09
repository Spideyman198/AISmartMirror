# AISmartMirror Architecture

## Overview

AISmartMirror is an edge-first AI smart mirror application. Local processing handles camera, face detection, face recognition, and gesture recognition. Cloud services are used only for advanced speech-to-text, conversational AI, text-to-speech, and notifications.

## Design Principles

- **Edge-first**: Maximize local processing for latency and privacy
- **Modular**: Clear separation of concerns, swappable backends
- **Simple**: No over-engineering, maintainable for capstone scope

## Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `app` | Entry point, app controller, main loop orchestration |
| `vision` | Camera, face detection, face recognition, gesture recognition |
| `audio` | Speech-to-text, text-to-speech, voice assistant |
| `ui` | Dashboard, display logic, mirror content |
| `integrations` | OpenAI, ElevenLabs, n8n, notifications |
| `config` | Settings from environment variables |
| `utils` | Logging, helpers |

## Data Flow

```
Camera → FaceDetector → FaceRecognizer
                    → GestureRecognizer
                    → (UI updates)

Microphone → SpeechToText → VoiceAssistant → OpenAI (optional)
                         → TextToSpeech
                         → Notifier
```

## Tech Stack

- **Python**: Core language
- **OpenCV**: Camera, image processing
- **MediaPipe**: Face detection, gesture recognition (planned)
- **NumPy**: Array operations
- **python-dotenv**: Environment configuration
- **logging**: Centralized logging

## Future Integrations

- OpenAI API (conversational AI)
- ElevenLabs (TTS)
- n8n (automation webhooks)
