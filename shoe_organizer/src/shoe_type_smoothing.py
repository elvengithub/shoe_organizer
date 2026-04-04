"""Temporal smoothing for sports | leather | casual to reduce frame-to-frame flicker."""
from __future__ import annotations

from collections import Counter, deque


class ShoeTypeSmoother:
    """
    Keeps a short rolling window of fused shoe types and returns the mode (majority).
    Stabilizes labels without waiting many seconds — usually 2–3 frames after motion stops.
    """

    def __init__(self, window: int = 5) -> None:
        self._window = max(2, int(window))
        self._buf: deque[str] = deque(maxlen=self._window)

    def clear(self) -> None:
        self._buf.clear()

    def update(self, shoe_type: str) -> str:
        t = (shoe_type or "casual").lower()
        if t not in ("casual", "sports", "leather"):
            t = "casual"
        self._buf.append(t)
        if len(self._buf) == 1:
            return t
        return Counter(self._buf).most_common(1)[0][0]
