"""
Smoke tests - quick sanity checks that the project is minimally functional.

These tests run without hardware and verify imports, basic structure,
and that the app can be loaded. Run first when debugging failures.
"""

import sys
from pathlib import Path

import pytest


def test_python_version() -> None:
    """Python 3.10+ required."""
    assert sys.version_info >= (3, 10)


def test_project_root_in_path() -> None:
    """Project root must be importable."""
    project_root = Path(__file__).resolve().parent.parent
    assert str(project_root) in sys.path or str(project_root) == sys.path[0]


def test_import_config() -> None:
    """Config module imports and returns valid settings."""
    from config import get_settings, Settings

    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.APP_NAME == "AISmartMirror"
    assert settings.CAMERA_INDEX >= 0


def test_import_utils() -> None:
    """Utils module imports."""
    from utils import get_logger, setup_logging

    setup_logging(level="WARNING")
    log = get_logger("test_smoke")
    assert log is not None


def test_import_vision() -> None:
    """Vision module imports."""
    from vision import CameraManager, FaceDetector
    from vision.face_recognizer import FaceRecognizer
    from vision.gesture_recognizer import GestureRecognizer

    assert CameraManager is not None
    assert FaceDetector is not None
    assert FaceRecognizer is not None
    assert GestureRecognizer is not None


def test_import_audio() -> None:
    """Audio module imports."""
    from audio.speech_to_text import SpeechToText
    from audio.text_to_speech import TextToSpeech
    from audio.voice_assistant import VoiceAssistant

    assert SpeechToText is not None
    assert TextToSpeech is not None
    assert VoiceAssistant is not None


def test_import_integrations() -> None:
    """Integrations module imports."""
    from integrations.openai_client import OpenAIClient
    from integrations.elevenlabs_client import ElevenLabsClient
    from integrations.notifier import Notifier

    assert OpenAIClient is not None
    assert ElevenLabsClient is not None
    assert Notifier is not None


def test_import_app() -> None:
    """App module imports and main is callable."""
    from app.app_controller import AppController
    from app.main import main

    assert AppController is not None
    assert callable(main)


def test_app_controller_instantiation() -> None:
    """AppController can be instantiated without hardware."""
    from app.app_controller import AppController

    controller = AppController()
    assert controller is not None
