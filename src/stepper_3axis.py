from __future__ import annotations

import logging
import time

from .config_loader import load_config
from .hardware import GPIOBackend, pulse_step

log = logging.getLogger(__name__)


class ThreeAxisCartesian:
    """
    Minimal STEP/DIR control for X (GT2), Y (GT2), Z (T8 lead screw).
    No acceleration profile — add if you need smoother motion.
    """

    def __init__(self, cfg: dict | None = None, backend: GPIOBackend | None = None) -> None:
        self.cfg = cfg or load_config()
        self.backend = backend or GPIOBackend()
        m = self.cfg["motion"]
        s = self.cfg["steppers"]
        self._x = s["x"]
        self._y = s["y"]
        self._z = s["z"]
        self._steps_per_mm_xy = self._compute_steps_per_mm_gt2(m)
        self._steps_per_mm_z = m["z_steps_per_rev"] / m["z_mm_per_rev"]
        for axis in (self._x, self._y, self._z):
            self.backend.setup_output(axis["step"], 0)
            self.backend.setup_output(axis["dir"], 0)
            if "enable" in axis:
                self.backend.setup_output(axis["enable"], 0)
                self.backend.output(axis["enable"], 0)
        self._pos_mm = {"x": 0.0, "y": 0.0, "z": 0.0}

    @staticmethod
    def _compute_steps_per_mm_gt2(m: dict) -> float:
        teeth = m["pulley_teeth"]
        pitch = m["gt2_pitch_mm"]
        belt_mm_per_rev = 2.0 * pitch * teeth
        spr = m["motor_steps_per_rev"] * m["microsteps"]
        return spr / belt_mm_per_rev

    def _enable_axis(self, axis: dict, on: bool) -> None:
        en = axis.get("enable")
        if en is not None:
            self.backend.output(en, 0 if on else 1)

    def _move_axis(
        self,
        axis: dict,
        steps: int,
        delay_s: float,
    ) -> None:
        if steps == 0:
            return
        self._enable_axis(axis, True)
        self.backend.output(axis["dir"], 1 if steps > 0 else 0)
        n = abs(int(steps))
        for _ in range(n):
            pulse_step(axis["step"], self.backend, delay_s)

    def move_mm(self, dx: float, dy: float, dz: float) -> None:
        lim = self.cfg["limits"]
        sx = int(round(dx * self._steps_per_mm_xy))
        sy = int(round(dy * self._steps_per_mm_xy))
        sz = int(round(dz * self._steps_per_mm_z))
        m = self.cfg["motion"]
        dxy = max(m["max_feed_mm_s"], 0.001)
        dzz = max(m["z_max_feed_mm_s"], 0.001)
        delay_xy = 0.5 / (dxy * self._steps_per_mm_xy)
        delay_z = 0.5 / (dzz * self._steps_per_mm_z)
        self._move_axis(self._x, sx, min(delay_xy, 0.002))
        self._move_axis(self._y, sy, min(delay_xy, 0.002))
        self._move_axis(self._z, sz, min(delay_z, 0.003))
        self._pos_mm["x"] += dx
        self._pos_mm["y"] += dy
        self._pos_mm["z"] += dz
        log.info("moved steps x=%s y=%s z=%s pos_mm=%s", sx, sy, sz, self._pos_mm)

    def goto_compartment_y_index(self, storage_slot: int) -> None:
        """Move Y to the center line for a storage slot (ids from compartments.storage_ids)."""
        ys = self.cfg["limits"]["compartment_y_mm"]
        storage = list(self.cfg["compartments"]["storage_ids"])
        if storage_slot not in storage:
            raise ValueError(f"storage slot {storage_slot} not in compartments.storage_ids")
        idx = storage.index(int(storage_slot))
        if idx < 0 or idx >= len(ys):
            raise ValueError(
                "limits.compartment_y_mm must have one entry per storage_ids slot "
                f"(need {len(storage)} positions, have {len(ys)})"
            )
        target_y = ys[idx]
        dy = target_y - self._pos_mm["y"]
        self.move_mm(0, dy, 0)
