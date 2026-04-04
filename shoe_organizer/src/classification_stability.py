"""Phase 1: consecutive-frame confirmation before treating the object as a stable shoe."""
from __future__ import annotations


class ClassificationStability:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg.get("classification_stability", {})
        self._rs = 0
        self._rn = 0
        self._confirmed = False

    def tick(self, raw_is_shoe: bool) -> tuple[bool, bool]:
        """
        raw_is_shoe: single-frame pipeline result (after gate + binary + neg gallery).
        Returns (confirmed_is_shoe, stabilizing).
        """
        if not self._cfg.get("enabled", True):
            self._confirmed = raw_is_shoe
            return self._confirmed, False

        sc = max(1, int(self._cfg.get("shoe_confirm_frames", 3)))
        sn = max(1, int(self._cfg.get("not_shoe_confirm_frames", 2)))

        if raw_is_shoe:
            self._rn = 0
            self._rs += 1
            if self._rs >= sc:
                self._confirmed = True
        else:
            self._rs = 0
            self._rn += 1
            if self._rn >= sn:
                self._confirmed = False

        stabilizing = (raw_is_shoe and not self._confirmed and self._rs < sc) or (
            not raw_is_shoe and self._confirmed and self._rn < sn
        )
        return self._confirmed, stabilizing

    def confirmed(self) -> bool:
        return self._confirmed
