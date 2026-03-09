"""
Centralized logging configuration.

Provides consistent log format and level across the application.
"""

import logging
import sys
from typing import Optional

from config import get_settings


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure root logger for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to settings.
        format_string: Custom format. Defaults to standard format.
    """
    settings = get_settings()
    log_level = level or settings.LOG_LEVEL
    fmt = format_string or (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=date_fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
