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
        # Recognition: max distance for match. Lower=stricter (fewer false matches),
        # higher=lenient (better at angles, more false matches). Tune in .env.
        self.FACE_RECOGNITION_THRESHOLD: float = float(
            os.getenv("FACE_RECOGNITION_THRESHOLD", "0.6")
        )
        # Run recognition every N frames (1=every frame). Higher reduces lag.
        self.FACE_RECOGNITION_INTERVAL_FRAMES: int = int(
            os.getenv("FACE_RECOGNITION_INTERVAL_FRAMES", "5")
        )
        # Require N consecutive matching results before confirming identity (reduces flicker)
        self.RECOGNITION_CONFIRMATION_COUNT: int = int(
            os.getenv("RECOGNITION_CONFIRMATION_COUNT", "2")
        )
        # Show recognition debug info on screen (distance, threshold, state)
        self.DEBUG_RECOGNITION: bool = os.getenv("DEBUG_RECOGNITION", "false").lower() in (
            "true", "1", "yes"
        )

        # CNN face recognizer (optional - separate from embedding-based baseline)
        self.CNN_MODEL_DIR: Optional[str] = os.getenv("CNN_MODEL_DIR")
        self.CNN_CONFIDENCE_THRESHOLD: float = float(
            os.getenv("CNN_CONFIDENCE_THRESHOLD", "0.62")
        )
        _cnn_m = os.getenv("CNN_MIN_CLASS_MARGIN", "0.15")
        self.CNN_MIN_CLASS_MARGIN: float = float(_cnn_m) if _cnn_m else 0.0
        _cnn_e = os.getenv("CNN_MAX_SOFTMAX_ENTROPY", "").strip()
        self.CNN_MAX_SOFTMAX_ENTROPY: Optional[float] = (
            float(_cnn_e) if _cnn_e else None
        )
        self.USE_CNN_RECOGNIZER: bool = os.getenv("USE_CNN_RECOGNIZER", "false").lower() in (
            "true", "1", "yes"
        )
        # Guided enrollment (camera auto-scan)
        self.ENROLLMENT_TOTAL_SAMPLES: int = int(
            os.getenv("ENROLLMENT_TOTAL_SAMPLES", "24")
        )
        self.ENROLLMENT_MIN_FACE_SIZE: int = int(
            os.getenv("ENROLLMENT_MIN_FACE_SIZE", "72")
        )
        self.ENROLLMENT_MIN_LAPLACIAN_VAR: float = float(
            os.getenv("ENROLLMENT_MIN_LAPLACIAN_VAR", "45")
        )
        self.ENROLL_CAPTURE_INTERVAL_FRAMES: int = int(
            os.getenv("ENROLL_CAPTURE_INTERVAL_FRAMES", "2")
        )
        self.ENROLL_CAPTURE_COOLDOWN_SECONDS: float = float(
            os.getenv(
                "ENROLL_CAPTURE_COOLDOWN_SECONDS",
                os.getenv("ENROLLMENT_CAPTURE_COOLDOWN_SEC", "0.55"),
            )
        )
        # Backward-compatible alias used by older code paths.
        self.ENROLLMENT_CAPTURE_COOLDOWN_SEC: float = float(
            os.getenv("ENROLLMENT_CAPTURE_COOLDOWN_SEC", str(self.ENROLL_CAPTURE_COOLDOWN_SECONDS))
        )
        self.ENROLLMENT_DUPLICATE_DISTANCE: float = float(
            os.getenv("ENROLLMENT_DUPLICATE_DISTANCE", "0.03")
        )
        self.ENROLLMENT_ENABLE_DISTANCE_SWEEP: bool = os.getenv(
            "ENROLLMENT_ENABLE_DISTANCE_SWEEP", "true"
        ).lower() in ("true", "1", "yes")
        self.ENROLLMENT_STEADY_FRAMES_REQUIRED: int = int(
            os.getenv("ENROLLMENT_STEADY_FRAMES_REQUIRED", "3")
        )
        self.ENROLL_REQUIRE_POSE_MATCH: bool = os.getenv(
            "ENROLL_REQUIRE_POSE_MATCH", "false"
        ).lower() in ("true", "1", "yes")
        self.ENROLL_POSE_TIMEOUT_SEC: float = float(
            os.getenv("ENROLL_POSE_TIMEOUT_SEC", "12")
        )

        # Cloud APIs (optional - app runs without these)
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
        self.N8N_WEBHOOK_URL: Optional[str] = os.getenv("N8N_WEBHOOK_URL")


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
