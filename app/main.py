"""
AISmartMirror - Main entry point.

Run with: python -m app.main
From project root directory.
"""

import sys

from app.app_controller import AppController
from utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Entry point: initialize and run the application."""
    setup_logging()
    controller = AppController()
    if not controller.initialize():
        logger.error("Initialization failed. Check camera connection and try again.")
        sys.exit(1)
    controller.run()


if __name__ == "__main__":
    main()
