"""
Recognition smoother - stabilizes displayed identity across recognition cycles.

Requires a label to appear consistently across multiple recognition cycles before
confirming it. Reduces flicker between "Alice" / "Unknown" when the raw result
oscillates near the threshold.
"""

from dataclasses import dataclass
from typing import Any, Optional

from vision.face_recognizer import RecognitionResult


@dataclass
class SmootherState:
    """Per-face state for smoothing."""

    last_seen: str
    streak: int
    confirmed: str


@dataclass
class DebugInfo:
    """Optional debug info for on-screen display."""

    matched_name: str
    best_distance: float
    threshold: float
    state: str  # e.g. "confirmed" or "pending 2/3"


class RecognitionSmoother:
    """
    Smooths recognition labels by requiring N consecutive identical results
    before confirming a displayed identity.
    """

    def __init__(self, confirmation_count: int = 2) -> None:
        """
        Args:
            confirmation_count: Number of consecutive matching results required
                before confirming. Default 2.
        """
        self._confirmation_count = max(1, confirmation_count)
        self._state: list[SmootherState] = []

    def update(
        self,
        results: list[RecognitionResult],
        threshold: float,
        show_debug: bool = False,
    ) -> tuple[list[str], Optional[list[dict[str, Any]]]]:
        """
        Update smoothing state with new recognition results.

        Args:
            results: Raw recognition results (one per face).
            threshold: Recognition threshold used.
            show_debug: If True, return debug info per face.

        Returns:
            (smoothed_labels, debug_infos or None)
        """
        n = len(results)
        if n == 0:
            self._state = []
            return [], None if not show_debug else []

        # Resize state when face count changes
        if len(self._state) != n:
            self._state = [
                SmootherState(last_seen="Unknown", streak=0, confirmed="Unknown")
                for _ in range(n)
            ]

        labels: list[str] = []
        debug_infos: Optional[list[dict[str, Any]]] = [] if show_debug else None

        for i, res in enumerate(results):
            raw_label = res.name if res.is_known else "Unknown"
            s = self._state[i]

            if raw_label == s.last_seen:
                s.streak += 1
            else:
                s.last_seen = raw_label
                s.streak = 1

            if s.streak >= self._confirmation_count:
                s.confirmed = raw_label

            labels.append(s.confirmed)

            if show_debug and debug_infos is not None:
                state_str = (
                    "confirmed"
                    if s.streak >= self._confirmation_count
                    else f"pending {s.streak}/{self._confirmation_count}"
                )
                display_name = res.name or res.best_match_name or "Unknown"
                debug_infos.append({
                    "matched_name": display_name,
                    "best_distance": res.distance,
                    "threshold": threshold,
                    "state": state_str,
                })

        return labels, debug_infos
