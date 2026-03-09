"""
Speech-to-text - converts microphone audio to text.

Placeholder for cloud-based STT (e.g. OpenAI Whisper API).
Edge: optional local fallback with Vosk or Sphinx.
"""

from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class SpeechToText:
    """Placeholder: STT not yet implemented."""

    def listen(self, timeout: float = 5.0) -> Optional[str]:
        """Listen and transcribe. Returns None for now."""
        return None

    def close(self) -> None:
        """Release resources."""
        pass
