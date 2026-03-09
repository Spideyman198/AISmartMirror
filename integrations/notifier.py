"""
Notifier - automation and notification integrations.

Placeholder for n8n webhooks, push notifications, etc.
"""

from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class Notifier:
    """Placeholder: notifications not yet implemented."""

    def notify(self, event: str, payload: Optional[dict[str, Any]] = None) -> bool:
        """Send notification. Returns False for now."""
        return False
