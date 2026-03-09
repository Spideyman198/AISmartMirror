"""
Helper utilities for AISmartMirror.

Common functions used across modules.
"""

from pathlib import Path
from typing import Any, Optional


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if needed.

    Args:
        path: Directory path.

    Returns:
        The path for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested dict value.

    Args:
        d: Dictionary.
        *keys: Key path (e.g. "a", "b", "c" for d["a"]["b"]["c"]).
        default: Value if key path not found.

    Returns:
        Value or default.
    """
    for key in keys:
        try:
            d = d[key]
        except (KeyError, TypeError):
            return default
    return d
