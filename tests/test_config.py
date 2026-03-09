"""
Config tests - verify settings load correctly from environment.

Uses environment variable overrides; no .env file required for tests.
"""

import os
from unittest.mock import patch

import pytest


def test_settings_defaults() -> None:
    """Settings have expected default values."""
    from config.settings import get_settings

    # Clear cache to pick up env
    get_settings.cache_clear()
    with patch.dict(os.environ, {}, clear=False):
        settings = get_settings()
        assert settings.APP_NAME == "AISmartMirror"
        assert settings.LOG_LEVEL == "INFO"
        assert settings.CAMERA_INDEX == 0
        assert settings.CAMERA_WIDTH == 640
        assert settings.CAMERA_HEIGHT == 480
        assert settings.FACE_DETECTION_CONFIDENCE == 0.5
    get_settings.cache_clear()


def test_settings_from_env() -> None:
    """Settings respect environment variables."""
    from config.settings import get_settings

    get_settings.cache_clear()
    with patch.dict(
        os.environ,
        {
            "LOG_LEVEL": "DEBUG",
            "CAMERA_INDEX": "1",
            "CAMERA_WIDTH": "320",
            "FACE_DETECTION_CONFIDENCE": "0.7",
        },
        clear=False,
    ):
        settings = get_settings()
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.CAMERA_INDEX == 1
        assert settings.CAMERA_WIDTH == 320
        assert settings.FACE_DETECTION_CONFIDENCE == 0.7
    get_settings.cache_clear()


def test_settings_debug_flag() -> None:
    """DEBUG parses true/false correctly."""
    from config.settings import get_settings

    get_settings.cache_clear()
    with patch.dict(os.environ, {"DEBUG": "true"}, clear=False):
        settings = get_settings()
        assert settings.DEBUG is True
    with patch.dict(os.environ, {"DEBUG": "false"}, clear=False):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.DEBUG is False
    get_settings.cache_clear()


def test_settings_optional_api_keys() -> None:
    """Cloud API keys are optional (None when not set)."""
    from config.settings import get_settings

    get_settings.cache_clear()
    with patch.dict(os.environ, {}, clear=False):
        settings = get_settings()
        assert settings.OPENAI_API_KEY is None
        assert settings.ELEVENLABS_API_KEY is None
    get_settings.cache_clear()
