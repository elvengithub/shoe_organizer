"""
Timed wash-bay actuator schedule for ESP32 (or any client) polling GET /api/esp32/actuators.

When the camera reports a dirty shoe (needs wash), a sequence runs:
  • Initial delay (all outputs off)
  • Repeat ``repeat_cycles`` times:
      pump1 on → motors on → pump2 on → motors on
  Timings are configurable under ``wash_actuators`` in config.yaml.

Outputs are mutually exclusive per step (only the named actuators are on).
Thread-safe for Flask threaded server.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Step:
    name: str
    duration_s: float
    pump1: bool
    pump2: bool
    motors: bool


class WashBayActuatorSequence:
    def __init__(self, cfg: dict) -> None:
        block = cfg.get("wash_actuators") or {}
        self._initial_delay_s = float(block.get("initial_delay_s", 15.0))
        self._pump1_run_s = float(block.get("pump1_run_s", 7.0))
        self._motors_run_s = float(block.get("motors_run_s", 30.0))
        self._pump2_run_s = float(block.get("pump2_run_s", 10.0))
        self._motors_second_run_s = float(block.get("motors_second_run_s", 30.0))
        self._repeat_cycles = max(1, int(block.get("repeat_cycles", 3)))

        self._steps: list[_Step] = [
            _Step("pump1", self._pump1_run_s, True, False, False),
            _Step("motors_1", self._motors_run_s, False, False, True),
            _Step("pump2", self._pump2_run_s, False, True, False),
            _Step("motors_2", self._motors_second_run_s, False, False, True),
        ]

        self._lock = threading.Lock()
        self._state: str = "idle"  # idle | running | finished
        self._in_initial: bool = False
        self._cycle: int = 0
        self._step_i: int = 0
        self._phase_deadline: float = 0.0

    @property
    def repeat_total(self) -> int:
        return self._repeat_cycles

    def force_idle(self) -> None:
        """All off; ready for a new run after shoe clears and returns."""
        with self._lock:
            self._reset()

    def tick(self, *, raw_shoe: bool, shoe_clean: bool) -> dict[str, object]:
        """
        Call on each actuator poll. ``shoe_clean`` means shoe present but wash_mode none.
        """
        now = time.monotonic()

        with self._lock:
            if not raw_shoe or shoe_clean:
                self._reset()
                return self._snapshot(False, False, False, "idle", None, None, 0)

            if self._state == "finished":
                return self._snapshot(
                    False,
                    False,
                    False,
                    "finished_waiting_clear",
                    None,
                    None,
                    self._repeat_cycles,
                )

            if self._state == "idle":
                self._start(now)
                log.info(
                    "wash sequence: started (%s cycles after %.1fs delay)",
                    self._repeat_cycles,
                    self._initial_delay_s,
                )

            self._advance(now)

            if self._state == "finished":
                log.info("wash sequence: completed %s cycles — idle until shoe clears", self._repeat_cycles)
                return self._snapshot(
                    False,
                    False,
                    False,
                    "finished_waiting_clear",
                    None,
                    None,
                    self._repeat_cycles,
                )

            p1, p2, m, phase = self._outputs_for_current()
            cycle_ui = None if phase == "initial_delay" else self._cycle + 1
            return self._snapshot(p1, p2, m, "running", phase, cycle_ui, self._cycle)

    def _reset(self) -> None:
        self._state = "idle"
        self._in_initial = False
        self._cycle = 0
        self._step_i = 0
        self._phase_deadline = 0.0

    def _start(self, now: float) -> None:
        self._state = "running"
        self._in_initial = True
        self._cycle = 0
        self._step_i = 0
        self._phase_deadline = now + self._initial_delay_s

    def _advance(self, now: float) -> None:
        while self._state == "running" and now >= self._phase_deadline:
            if self._in_initial:
                self._in_initial = False
                self._step_i = 0
                self._phase_deadline = now + self._steps[0].duration_s
                continue

            self._step_i += 1
            if self._step_i >= len(self._steps):
                self._cycle += 1
                self._step_i = 0
                if self._cycle >= self._repeat_cycles:
                    self._state = "finished"
                    return

            self._phase_deadline = now + self._steps[self._step_i].duration_s

    def _outputs_for_current(self) -> tuple[bool, bool, bool, str]:
        if self._in_initial:
            return False, False, False, "initial_delay"
        st = self._steps[self._step_i]
        return st.pump1, st.pump2, st.motors, st.name

    def _snapshot(
        self,
        p1: bool,
        p2: bool,
        motors: bool,
        seq_state: str,
        phase: str | None,
        cycle_1based: int | None,
        cycle_0based: int,
    ) -> dict[str, object]:
        legacy = bool(p1 or p2 or motors)
        return {
            "pump1_on": p1,
            "pump2_on": p2,
            "motors_on": motors,
            "wash_sequence_state": seq_state,
            "wash_sequence_phase": phase,
            "wash_sequence_cycle": cycle_1based,
            "wash_sequence_cycle_index": cycle_0based,
            "wash_sequence_repeat_total": self._repeat_cycles,
            "pump_on": legacy,
            "fan_on": False,
        }
