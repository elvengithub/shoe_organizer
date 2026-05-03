from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

class WashBayActuatorSequence:
    def __init__(self, block: dict):
        self._lock = threading.Lock()
        self._initial_delay_s = float(block.get("initial_delay_s", 10))
        
        # Base timings from config (or defaults)
        self._p1_time = float(block.get("pump1_run_s", 8))
        self._p2_time = float(block.get("pump2_run_s", 5))
        self._ext_time = 20.0
        
        self._state = "idle"
        self._in_initial = False
        self._manual_mode_override = None # "soft" | "hard"
        self._step_i = 0
        self._cycle = 0
        self._phase_deadline = 0.0

    def tick(self, *, raw_shoe: bool, shoe_clean: bool, auto_start: bool = True) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            # 1. Auto-start logic (only if enabled and idle)
            if self._state == "idle" and auto_start and raw_shoe and not shoe_clean:
                self._manual_mode_override = None # Ensure it's treated as auto
                self._start(now)

            # 2. Safety check (Removed: Wash now runs to completion once started)
            # The user requested that it should not interrupt even if shoe is no longer detected.
            pass

            # 3. Progress sequence
            if self._state == "running":
                self._advance(now)

            # 4. Snapshot
            if self._state == "idle":
                return self._snapshot(False, False, False, "idle", None, None, 0)
            
            if self._state in ("finished", "finished_waiting_clear"):
                if raw_shoe:
                    self._state = "finished_waiting_clear"
                    return self._snapshot(False, False, False, "finished", None, None, 0)
                else:
                    self._reset()
                    return self._snapshot(False, False, False, "idle", None, None, 0)

            # 5. Calculate outputs for 'running'
            return self._get_running_snapshot(now)

    def _start(self, now: float) -> None:
        p1, p2, ext = self._get_config()
        print(f"\n>>> [SERVER] WASH SEQUENCE STARTING | Mode: {self._manual_mode_override or 'auto'} | Cycles: {p1}/{p2} | Ext: {ext}")
        self._state = "running"
        self._in_initial = True
        self._step_i = 0
        self._cycle = 0
        self._phase_deadline = now + self._initial_delay_s

    def _reset(self) -> None:
        if self._state != "idle":
            print(">>> [SERVER] WASH SEQUENCE RESETTING to idle.")
        self._state = "idle"
        self._in_initial = False
        self._manual_mode_override = None
        self._step_i = 0
        self._cycle = 0
        self._phase_deadline = 0.0

    def _get_config(self):
        # Returns (p1_max, p2_max, has_extension)
        if self._manual_mode_override == "hard":
            return 4, 3, True
        # Everything else (soft or auto) defaults to 3/2
        return 3, 2, False

    def _advance(self, now: float) -> None:
        p1_max, p2_max, has_ext = self._get_config()
        
        while self._state == "running" and now >= self._phase_deadline:
            if self._in_initial:
                self._in_initial = False
                self._phase_deadline = now + self._p1_time
                self._step_i = 1 # Start with P1
                continue
            
            # Step 1: Pump 1
            if self._step_i == 1:
                # Finished P1, check if we need P2
                if self._cycle < p2_max:
                    self._step_i = 2
                    self._phase_deadline = now + self._p2_time
                    return # Exit to wait for P2 time
                elif self._cycle + 1 < p1_max:
                    self._cycle += 1
                    self._step_i = 1
                    self._phase_deadline = now + self._p1_time
                    return # Exit to wait for next P1
                else:
                    # All pump cycles done
                    if has_ext:
                        self._step_i = 3
                        self._phase_deadline = now + self._ext_time
                        return # Exit to wait for EXTENSION
                    else:
                        self._state = "finished"
                        return
            
            # Step 2: Pump 2
            if self._step_i == 2:
                # Finished P2, always go back to P1 if more cycles left
                self._cycle += 1
                if self._cycle < p1_max:
                    self._step_i = 1
                    self._phase_deadline = now + self._p1_time
                    return
                else:
                    if has_ext:
                        self._step_i = 3
                        self._phase_deadline = now + self._ext_time
                        return
                    else:
                        self._state = "finished"
                        return
            
            # Step 3: Extension
            if self._step_i == 3:
                self._state = "finished"
                return

    def sync_countdown(self, seconds: float) -> None:
        """Sync the server's DELAY deadline to match the ESP32's timer."""
        if self._state == "running" and self._step_i == 0:
            import time
            new_deadline = time.monotonic() + seconds
            # Only update if the difference is significant (>0.5s) to avoid jitter
            if abs(self._phase_deadline - new_deadline) > 0.5:
                self._phase_deadline = new_deadline

    @property
    def manual_mode_override(self) -> str | None:
        return self._manual_mode_override

    @property
    def repeat_total(self) -> int:
        p1_max, _, _ = self._get_config()
        return p1_max

    def _get_running_snapshot(self, now: float) -> dict:
        phase = "initial_delay" if self._in_initial else (
            "pump1" if self._step_i == 1 else ("pump2" if self._step_i == 2 else "extension")
        )
        cd = round(max(0, self._phase_deadline - now))
        p1_max, _, _ = self._get_config()
        
        p1 = (phase == "pump1")
        p2 = (phase == "pump2")
        m = (not self._in_initial) # Motor ON during all pumps and extension
        
        return {
            "pumps_on": p1 or p2,
            "pump1_on": p1,
            "pump2_on": p2,
            "motors_on": m,
            "wash_sequence_state": "running",
            "wash_sequence_phase": phase,
            "wash_sequence_cycle": self._cycle + 1,
            "wash_sequence_total_cycles": p1_max,
            "wash_sequence_countdown": cd,
            "wash_trigger_id": getattr(self, "_trigger_id", 0),
        }

    def _snapshot(self, p1, p2, m, seq_state, phase, cycle, total_cycles, countdown=0) -> dict:
        return {
            "pumps_on": p1 or p2,
            "pump1_on": p1,
            "pump2_on": p2,
            "motors_on": m,
            "wash_sequence_state": seq_state,
            "wash_sequence_phase": phase,
            "wash_sequence_cycle": cycle,
            "wash_sequence_countdown": countdown,
            "wash_trigger_id": getattr(self, "_trigger_id", 0),
        }

    def trigger_manual(self, mode: str) -> None:
        with self._lock:
            # Forcefully interrupt and restart the sequence!
            self._manual_mode_override = mode
            self._trigger_id = getattr(self, "_trigger_id", 0) + 1
            self._start(time.monotonic())

    def force_idle(self) -> None:
        with self._lock:
            self._reset()

    @property
    def repeat_total(self) -> int:
        p1_max, _, _ = self._get_config()
        return p1_max
