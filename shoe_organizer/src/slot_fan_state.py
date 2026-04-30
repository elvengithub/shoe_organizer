"""
Desired ON/OFF state for storage compartment fans (slots 2–6), controlled from the web UI.

ESP32 firmware that drives `extra_relays` should poll GET /api/esp32/camera-relays and apply
the `extra_relay_on` list (length 6): index i maps to Thonny relay key str(i + 1), i.e. slot 2 → index 0.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
# compartment_id -> bool
_state: dict[int, bool] = {}

# Global forced states (e.g. from manual UI override)
_motors_forced_on = False
_pumps_forced_on = False


def set_slot_fan(compartment_id: int, on: bool) -> None:
    with _lock:
        _state[int(compartment_id)] = bool(on)


def apply_fan_report(storage_ids: list[int], fans: dict) -> int:
    """
    Merge JSON object ``fans`` (keys compartment id as str or int) into state.
    Only updates ids present in ``storage_ids``. Returns number of slots updated.
    """
    allowed = {int(x) for x in storage_ids}
    n = 0
    with _lock:
        for k, v in fans.items():
            try:
                cid = int(k)
            except (TypeError, ValueError):
                continue
            if cid not in allowed:
                continue
            _state[cid] = bool(v)
            n += 1
    return n


def get_slot_fan(compartment_id: int) -> bool:
    with _lock:
        return bool(_state.get(int(compartment_id), False))


def snapshot_slots(storage_ids: list[int]) -> dict[str, bool]:
    with _lock:
        return {str(cid): bool(_state.get(int(cid), False)) for cid in storage_ids}


def extra_relay_six_for_esp(storage_ids: list[int]) -> list[bool]:
    """Six booleans for six relays; slot 2..6 → indices 0..4; index 5 unused (False)."""
    try:
        ids = sorted(int(x) for x in storage_ids)
    except (TypeError, ValueError):
        ids = []
    with _lock:
        bits = [_state.get(cid, False) for cid in ids[:5]]
    while len(bits) < 6:
        bits.append(False)
    return bits[:6]


def set_global_motors(on: bool) -> None:
    global _motors_forced_on
    with _lock:
        _motors_forced_on = bool(on)


def get_global_motors() -> bool:
    with _lock:
        return _motors_forced_on


def set_global_pumps(on: bool) -> None:
    global _pumps_forced_on
    with _lock:
        _pumps_forced_on = bool(on)


def get_global_pumps() -> bool:
    with _lock:
        return _pumps_forced_on


def stop_all_actuators() -> None:
    """Force all fans, motors, and pumps OFF."""
    global _motors_forced_on, _pumps_forced_on
    with _lock:
        _state.clear()
        _motors_forced_on = False
        _pumps_forced_on = False
