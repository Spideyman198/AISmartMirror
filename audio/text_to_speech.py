"""
Text-to-speech - converts text to spoken audio.

Placeholder for cloud TTS (e.g. ElevenLabs) or local (pyttsx3).
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class TextToSpeech:
    """Placeholder: TTS not yet implemented."""

    def speak(self, text: str) -> None:
        """Speak the given text."""
        pass

    def close(self) -> None:
        """Release resources."""
        pass
