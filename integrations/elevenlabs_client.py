"""
ElevenLabs API client - high-quality TTS.

Placeholder. Requires ELEVENLABS_API_KEY when implemented.
"""

from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ElevenLabsClient:
    """Placeholder: ElevenLabs integration not yet implemented."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    def speak(self, text: str) -> bytes:
        """Generate speech from text. Returns empty bytes for now."""
        return b""
