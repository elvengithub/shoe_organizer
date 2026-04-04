"""
Last telemetry POST from an ESP32 (DHT + gas + ultrasonic). Thread-safe store.
Merged into /api/status for one configured storage compartment when data is fresh.
"""
from __future__ import annotations

import threading
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Request


def verify_esp32_secret(cfg: dict, request: "Request") -> bool:
    secret = str((cfg.get("esp32_telemetry") or {}).get("secret") or "").strip()
    if not secret:
        return True
    if request.headers.get("X-ESP32-Secret", "") == secret:
        return True
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:].strip() == secret:
        return True
    return False

_lock = threading.Lock()
_last: dict[str, Any] | None = None
_last_mono: float = 0.0


def _mono() -> float:
    return time.monotonic()


def update_from_body(body: dict[str, Any]) -> None:
    global _last, _last_mono
    with _lock:
        _last = dict(body)
        _last_mono = _mono()


def get_last() -> tuple[dict[str, Any] | None, float]:
    """Returns (payload, age_seconds) age is from monotonic clock delta."""
    with _lock:
        if _last is None:
            return None, -1.0
        return dict(_last), _mono() - _last_mono


def apply_to_climate_snapshot(out: dict[str, dict[str, Any]], cfg: dict) -> None:
    block = cfg.get("esp32_telemetry") or {}
    if not block.get("enabled"):
        return
    try:
        cid = int(block["compartment_id"])
    except (KeyError, TypeError, ValueError):
        return
    key = str(cid)
    if key not in out:
        return
    max_age = float(block.get("max_age_seconds", 10.0))
    payload, age = get_last()
    row = out[key]
    if payload is None or age < 0 or age > max_age:
        row["esp32_online"] = False
        return

    use_occ = bool(block.get("use_ultrasonic_occupancy", True))
    if "temperature_c" in payload and payload["temperature_c"] is not None:
        try:
            row["temperature_c"] = round(float(payload["temperature_c"]), 1)
        except (TypeError, ValueError):
            pass
    if "humidity_pct" in payload and payload["humidity_pct"] is not None:
        try:
            row["humidity_pct"] = round(float(payload["humidity_pct"]), 1)
        except (TypeError, ValueError):
            pass
    if use_occ and "occupied" in payload:
        row["occupied"] = bool(payload["occupied"])

    for k in (
        "distance_cm",
        "gas_raw",
        "odor_level_pct",
        "risk_level",
        "bio_message",
        "ultrasonic_max_cm",
        "ultrasonic_clear",
    ):
        if k in payload:
            row[k] = payload[k]
    row["esp32_online"] = True
    row["esp32_age_seconds"] = round(age, 2)
