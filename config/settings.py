"""
Settings module - loads configuration from environment variables.

Uses python-dotenv for .env file support. All cloud API keys are optional
for the starter phase; the app runs without them.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        """Load settings from environment at instantiation time."""
        # App
        self.APP_NAME: str = "AISmartMirror"
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        self.DEBUG_SAVE_FRAME: bool = os.getenv("DEBUG_SAVE_FRAME", "false").lower() in ("true", "1", "yes")

        # Camera
        self.CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
        self.CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "640"))
        self.CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "480"))
        self.CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))
        # Backend: DSHOW, MSMF, or empty for auto (tries both)
        self.CAMERA_BACKEND: Optional[str] = os.getenv("CAMERA_BACKEND") or None

        # Vision
        self.FACE_DETECTION_CONFIDENCE: float = float(
            os.getenv("FACE_DETECTION_CONFIDENCE", "0.4")
        )
        # MediaPipe model: 0=short-range (2m), 1=full-range (5m) - use 1 for distant faces
        self.FACE_DETECTION_MODEL: int = int(os.getenv("FACE_DETECTION_MODEL", "1"))
        self.KNOWN_FACES_DIR: Optional[str] = os.getenv("KNOWN_FACES_DIR")

        # Cloud APIs (optional - app runs without these)
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
        self.N8N_WEBHOOK_URL: Optional[str] = os.getenv("N8N_WEBHOOK_URL")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
