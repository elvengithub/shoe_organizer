"""
Last telemetry POST from an ESP32 (DHT + gas + ultrasonic). Thread-safe store.
Merged into /api/status when data is fresh.

Supports:
  • Legacy: single JSON object (uses esp32_telemetry.compartment_id from config).
  • Per-slot: {"compartment_id": 3, "temperature_c": ..., ...} (no compartment_id key in payload fields).
  • Multi-slot: {"compartments": {"2": {...}, "3": {...}}} — updates several storage slots in one POST.
"""
from __future__ import annotations

import threading
import time
from typing import Any, TYPE_CHECKING

from .config_loader import load_config

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
# Legacy single broadcast (USB serial lines, old firmware)
_last: dict[str, Any] | None = None
_last_mono: float = 0.0
# Multi-compartment WiFi (or explicit compartment_id in JSON)
_last_by_cid: dict[int, dict[str, Any]] = {}
_mono_by_cid: dict[int, float] = {}


def _mono() -> float:
    return time.monotonic()


def update_from_body(body: dict[str, Any], cfg: dict | None = None) -> None:
    """Store telemetry. cfg used for legacy path when body has no compartment hints."""
    global _last, _last_mono
    with _lock:
        comps = body.get("compartments")
        if isinstance(comps, dict):
            for k, v in comps.items():
                try:
                    cid = int(k)
                except (TypeError, ValueError):
                    continue
                if isinstance(v, dict):
                    _last_by_cid[cid] = dict(v)
                    _mono_by_cid[cid] = _mono()
            return
        single_cid = body.get("compartment_id")
        if single_cid is not None:
            try:
                cid = int(single_cid)
            except (TypeError, ValueError):
                cid = None
            if cid is not None:
                pl = {k: v for k, v in body.items() if k != "compartment_id"}
                _last_by_cid[cid] = pl
                _mono_by_cid[cid] = _mono()
                return
        cfg = cfg or load_config()
        block = cfg.get("esp32_telemetry") or {}
        try:
            fallback_cid = int(block.get("compartment_id", 0))
        except (TypeError, ValueError):
            fallback_cid = 0
        if fallback_cid > 0 and not body.get("compartments"):
            _last_by_cid[fallback_cid] = dict(body)
            _mono_by_cid[fallback_cid] = _mono()
        _last = dict(body)
        _last_mono = _mono()


def get_last() -> tuple[dict[str, Any] | None, float]:
    """Legacy debug: last flat payload + age (may be stale if multi-compartment used)."""
    with _lock:
        if _last is None:
            return None, -1.0
        return dict(_last), _mono() - _last_mono


def get_last_by_compartments() -> dict[str, Any]:
    """Debug: snapshot of per-compartment payloads and ages."""
    now = _mono()
    with _lock:
        out: dict[str, Any] = {
            "by_compartment": {},
            "legacy_last": dict(_last) if _last is not None else None,
            "legacy_age_seconds": None if _last is None else round(now - _last_mono, 3),
        }
        for cid, pl in _last_by_cid.items():
            age = now - _mono_by_cid.get(cid, now)
            out["by_compartment"][str(cid)] = {
                "payload": dict(pl),
                "age_seconds": round(age, 3),
            }
        return out


def _merge_payload_into_row(row: dict[str, Any], payload: dict[str, Any], use_occ: bool) -> None:
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


def apply_to_climate_snapshot(out: dict[str, dict[str, Any]], cfg: dict) -> None:
    block = cfg.get("esp32_telemetry") or {}
    if not block.get("enabled"):
        return
    max_age = float(block.get("max_age_seconds", 10.0))
    use_occ = bool(block.get("use_ultrasonic_occupancy", True))
    try:
        legacy_cid = int(block.get("compartment_id", 0))
    except (TypeError, ValueError):
        legacy_cid = 0

    now = _mono()
    with _lock:
        snap_by = dict(_last_by_cid)
        snap_mono = dict(_mono_by_cid)
        legacy_pl = dict(_last) if _last is not None else None
        legacy_m = _last_mono

    for cid_str, row in out.items():
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        payload: dict[str, Any] | None = None
        age: float | None = None
        if cid in snap_by:
            payload = snap_by[cid]
            age = now - snap_mono.get(cid, now)
        elif legacy_pl is not None and cid == legacy_cid:
            payload = legacy_pl
            age = now - legacy_m
        if payload is None or age is None or age > max_age:
            row["esp32_online"] = False
            continue
        _merge_payload_into_row(row, payload, use_occ)
        row["esp32_online"] = True
        row["esp32_age_seconds"] = round(age, 2)
