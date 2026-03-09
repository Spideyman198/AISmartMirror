"""
Voice assistant - orchestrates STT, conversational AI, TTS.

Placeholder for voice command pipeline and AI integration.
"""

from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class VoiceAssistant:
    """Placeholder: voice assistant not yet implemented."""

    def process_command(self, text: str) -> Optional[str]:
        """Process voice command, return response. Returns None for now."""
        return None

    def listen_and_respond(self) -> Optional[str]:
        """Listen, process, speak. Returns None for now."""
        return None

    def close(self) -> None:
        """Release resources."""
        pass
