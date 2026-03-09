"""
OpenAI API client - conversational AI integration.

Placeholder. Requires OPENAI_API_KEY when implemented.
"""

from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Placeholder: OpenAI integration not yet implemented."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """Send message, get AI response. Returns empty string for now."""
        return ""
