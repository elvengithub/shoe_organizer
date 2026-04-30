# Merge into your relay script: call after serial toggle (or after applying extra_relay_on from Flask)
# so the dashboard matches Thonny. POST /api/esp32/storage-fans
#
# Mapping: Thonny keys 1..5 -> slots 2..6 (relay index i -> compartment i + 2).
# Key 6 (index 5) is not a storage slot in config; omit or map as you prefer.

import gc
import json

try:
    import urequests
except ImportError:
    urequests = None


def post_storage_fan_single(server_base, compartment_id, on, secret=""):
    """Report one slot: compartment_id 2..6, on True/False."""
    if urequests is None:
        return False
    gc.collect()
    url = str(server_base).rstrip("/") + "/api/esp32/storage-fans"
    body = json.dumps({"compartment": int(compartment_id), "on": bool(on)})
    headers = {"Content-Type": "application/json", "Connection": "close"}
    if secret:
        headers["X-ESP32-Secret"] = str(secret)
    try:
        r = urequests.post(url, data=body, headers=headers)
        try:
            ok = 200 <= getattr(r, "status_code", 200) < 300
        finally:
            r.close()
        return ok
    except OSError:
        return False


def post_storage_fans_map(server_base, fans_dict, secret=""):
    """
    Batch report: fans_dict like {"2": True, "3": False} (MicroPython: use str keys).
    """
    if urequests is None:
        return False
    gc.collect()
    url = str(server_base).rstrip("/") + "/api/esp32/storage-fans"
    body = json.dumps({"fans": fans_dict})
    headers = {"Content-Type": "application/json", "Connection": "close"}
    if secret:
        headers["X-ESP32-Secret"] = str(secret)
    try:
        r = urequests.post(url, data=body, headers=headers)
        try:
            ok = 200 <= getattr(r, "status_code", 200) < 300
        finally:
            r.close()
        return ok
    except OSError:
        return False


def relay_index_to_compartment(relay_index_zero_based):
    """relay 0 (key '1') -> slot 2."""
    return int(relay_index_zero_based) + 2


def sync_extra_relays_to_flask(extra_relays, server_base, drive_fn, secret=""):
    """
    Read actual GPIO state via drive_fn (same as _drive): returns True when relay is ON.
    Posts all slots 2..6 derived from first five relays.
    """
    fans = {}
    for i in range(min(5, len(extra_relays))):
        try:
            on = extra_relays[i].value() == drive_fn(True)
        except OSError:
            continue
        fans[str(relay_index_to_compartment(i))] = on
    return post_storage_fans_map(server_base, fans, secret=secret)
