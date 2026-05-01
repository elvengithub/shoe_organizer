from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

from flask import Flask, Response, jsonify, render_template, request

from .esp32_telemetry import get_last, get_last_by_compartments, update_from_body, verify_esp32_secret
from .slot_fan_state import (
    apply_fan_report,
    extra_relay_six_for_esp,
    get_slot_fan,
    set_slot_fan,
    snapshot_slots,
    set_global_motors,
    get_global_motors,
    set_global_pumps,
    get_global_pumps,
    stop_all_actuators,
    get_esp32_mode,
    set_esp32_mode,
)
from .orchestrator import ShoeOrganizerOrchestrator
from .text_presence import analyze_presented_text
from .shoe_taxonomy import SHOE_TYPE_LABELS, format_shoe_display_name
from .wash_decision import wash_ui_label
from .vision_service import blank_jpeg_bytes, encode_jpeg

log = logging.getLogger(__name__)


def _storage_id_set(compartments: dict) -> set[int]:
    """Compartment ids from config as ints (YAML may load them as strings)."""
    raw = compartments.get("storage_ids") or []
    out: set[int] = set()
    for x in raw:
        try:
            out.add(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _parse_json_on(body: dict) -> tuple[bool | None, str | None]:
    """Return (value, error) for a boolean ``on`` field; error is a short API error slug."""
    if "on" not in body:
        return None, "missing_on"
    raw = body["on"]
    if raw is None:
        return None, "missing_on"
    if isinstance(raw, bool):
        return raw, None
    if isinstance(raw, (int, float)):
        return bool(raw), None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True, None
        if s in ("0", "false", "no", "off", ""):
            return False, None
        return None, "invalid_on"
    return None, "invalid_on"


def create_app() -> Flask:
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    app = Flask(__name__, template_folder=os.path.join(root, "templates"), static_folder=os.path.join(root, "static"))

    @lru_cache(maxsize=1)
    def orch() -> ShoeOrganizerOrchestrator:
        return ShoeOrganizerOrchestrator()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.before_request
    def _api_cors_preflight():
        if request.method == "OPTIONS" and request.path.startswith("/api/"):
            return Response(
                "",
                204,
                {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-ESP32-Secret",
                },
            )
        return None

    @app.after_request
    def _api_cors_headers(resp):
        if request.path.startswith("/api/"):
            resp.headers.setdefault("Access-Control-Allow-Origin", "*")
        return resp

    @app.post("/api/analyze")
    def api_analyze():
        """Classify from free text only (no images). Same JSON shape as /api/camera/analyze when shoe path."""
        body = request.get_json(force=True, silent=True) or {}
        raw = body.get("description", "")
        desc = raw if isinstance(raw, str) else str(raw)
        o = orch()
        return jsonify(analyze_presented_text(desc, o.cfg))

    @app.get("/api/camera/analyze")
    def camera_analyze():
        """Live shoe-type + wash recommendation from the current frame (same logic as intake)."""
        return jsonify(orch().analyze_camera_live())

    @app.post("/api/motors")
    def api_motors():
        body = request.get_json(force=True, silent=True) or {}
        val, err = _parse_json_on(body)
        if err:
            return jsonify({"ok": False, "error": err}), 400
        if get_esp32_mode() == "AUTO":
            return jsonify({"ok": False, "error": "manual_control_disabled_in_auto_mode"}), 403
        set_global_motors(val)
        return jsonify({"ok": True, "motors_on": val})

    @app.post("/api/pumps")
    def api_pumps():
        body = request.get_json(force=True, silent=True) or {}
        val, err = _parse_json_on(body)
        if err:
            return jsonify({"ok": False, "error": err}), 400
        if get_esp32_mode() == "AUTO":
            return jsonify({"ok": False, "error": "manual_control_disabled_in_auto_mode"}), 403
        set_global_pumps(val)
        return jsonify({"ok": True, "pumps_on": val})

    @app.post("/api/stop-all")
    def api_stop_all():
        o = orch()
        stop_all_actuators()
        # Also stop server-side ventilation if any
        for cid in _storage_id_set(o.cfg["compartments"]):
            o.sensors.set_ventilation(cid, False)
        return jsonify({"ok": True, "message": "All actuators stopped."})

    @app.get("/api/camera/stream")
    def camera_stream():
        o = orch()
        if str(o.cfg.get("presenter", {}).get("mode", "camera")).lower() == "text":
            return Response(
                "presenter mode is text — no camera stream\n",
                status=404,
                mimetype="text/plain",
            )
        w = int(o.cfg["camera"]["width"])
        h = int(o.cfg["camera"]["height"])
        placeholder = blank_jpeg_bytes(w, h)

        def gen_mjpeg():
            boundary = b"--frame\r\n"
            while True:
                frame = o.cam.read()
                blob = encode_jpeg(frame) if frame is not None else None
                if blob is None:
                    blob = placeholder
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + blob + b"\r\n"
                time.sleep(1 / 12)

        return Response(
            gen_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
        )

    @app.post("/api/camera/frame")
    def camera_frame_upload():
        """ESP32 (MicroPython): POST raw JPEG body or multipart field `image` / `file`."""
        if str(orch().cfg.get("presenter", {}).get("mode", "camera")).lower() == "text":
            return jsonify({"ok": False, "error": "presenter_text_mode"}), 404
        data = b""
        ct = (request.content_type or "").lower()
        if "multipart/form-data" in ct:
            f = request.files.get("image") or request.files.get("file")
            if f:
                data = f.read()
        if not data:
            data = request.get_data()
        if not data:
            return jsonify({"ok": False, "error": "empty_body"}), 400
        o = orch()
        ok, err = o.cam.ingest_jpeg(data)
        if not ok:
            return jsonify({"ok": False, "error": err}), 400
        return jsonify({"ok": True})

    @app.post("/api/intake")
    def api_intake():
        body = request.get_json(force=True, silent=True) or {}
        has_desc = "description" in body
        raw = body.get("description", "")
        desc = raw if isinstance(raw, str) else str(raw)
        o = orch()
        r = o.run_intake_cycle(description=desc if has_desc else None)
        if r.outcome == "empty_description":
            return jsonify(
                {
                    "ok": False,
                    "error": "empty_description",
                    "is_shoe": None,
                    "wash_mode": None,
                    "wash_reason": None,
                    "compartment": None,
                    "message": r.message,
                    "dirt_score": None,
                    "shoe_category": None,
                }
            )
        if r.outcome == "no_camera":
            return jsonify(
                {
                    "ok": False,
                    "error": "camera_unavailable",
                    "is_shoe": None,
                    "wash_mode": None,
                    "wash_reason": None,
                    "compartment": None,
                    "message": r.message,
                    "dirt_score": None,
                    "shoe_category": None,
                }
            )
        if r.outcome == "analysis_failed":
            return jsonify(
                {
                    "ok": False,
                    "error": "analysis_failed",
                    "is_shoe": None,
                    "wash_mode": None,
                    "wash_reason": None,
                    "compartment": None,
                    "message": r.message,
                    "dirt_score": None,
                    "shoe_category": None,
                    "reject_stage": r.reject_stage,
                    "classification_error": r.classification_error,
                    "reject_detail": r.reject_detail,
                }
            )
        if r.outcome == "not_shoe":
            return jsonify(
                {
                    "ok": True,
                    "error": "not_shoe",
                    "is_shoe": False,
                    "object_classification": "not_shoe",
                    "wash_mode": None,
                    "wash_reason": None,
                    "compartment": None,
                    "message": r.message,
                    "dirt_score": None,
                    "shoe_category": None,
                    "reject_stage": r.reject_stage,
                    "classification_error": r.classification_error,
                    "reject_detail": r.reject_detail,
                }
            )
        if r.outcome == "stabilizing":
            return jsonify(
                {
                    "ok": True,
                    "error": "stabilizing",
                    "is_shoe": True,
                    "compartment": None,
                    "message": r.message,
                    "wash_mode": None,
                    "wash_reason": None,
                    "dirt_score": None,
                    "shoe_category": None,
                }
            )
        if r.outcome == "no_catalog_match":
            return jsonify(
                {
                    "ok": True,
                    "error": "no_catalog_match",
                    "is_shoe": True,
                    "catalog_match": False,
                    "object_classification": "unknown_catalog",
                    "wash_mode": None,
                    "wash_reason": None,
                    "compartment": None,
                    "message": r.message,
                    "dirt_score": None,
                    "shoe_category": None,
                    "classification_error": r.classification_error,
                    "reject_detail": r.reject_detail,
                }
            )
        st = r.shoe_category or "casual"
        sl = SHOE_TYPE_LABELS.get(st, st.title())
        extra = {
            "shoe_type_label": format_shoe_display_name(st, sl, r.catalog_category, r.catalog_style),
            "wash_label": wash_ui_label(r.wash.mode, st),
        }
        if r.edge_density is not None:
            extra["edge_density"] = round(float(r.edge_density), 4)
        if r.sports_edge_density_min is not None:
            extra["sports_edge_density_min"] = float(r.sports_edge_density_min)
        if r.sports_fusion_score is not None:
            extra["sports_fusion_score"] = round(float(r.sports_fusion_score), 4)
        if r.sports_fusion_threshold is not None:
            extra["sports_fusion_threshold"] = float(r.sports_fusion_threshold)
        return jsonify(
            {
                "ok": True,
                "error": None,
                "is_shoe": True,
                "catalog_match": True,
                "catalog_category": r.catalog_category,
                "catalog_style": r.catalog_style,
                "catalog_score": r.catalog_score,
                "object_classification": f"shoe_{r.shoe_category}" if r.shoe_category else None,
                "wash_mode": r.wash.mode,
                "wash_reason": r.wash.reason,
                "compartment": r.assigned_compartment,
                "message": r.message,
                "dirt_score": r.dirt_score,
                "shoe_category": r.shoe_category,
                **extra,
            }
        )

    @app.get("/api/esp32/ping")
    def esp32_ping():
        """
        ESP32 (MicroPython) connectivity check: GET from the same host:port as POST /api/esp32/telemetry.
        No auth — use only on a trusted LAN. Confirms Flask is listening and reachable from Wi-Fi.
        """
        return jsonify({"ok": True, "service": "shoe_organizer"})

    @app.get("/api/esp32/actuators")
    def esp32_actuators():
        """
        ESP32 poll: when the bay camera sees a shoe, pump_on and fan_on are true.
        Includes global overrides for manual control from the dashboard.
        """
        res = orch().esp32_actuator_snapshot()
        if res.get("ok"):
            res["motors_on"] = bool(res.get("motors_on") or get_global_motors())
            res["pumps_on"] = bool(res.get("pumps_on") or get_global_pumps())
            res["mode"] = get_esp32_mode()
        return jsonify(res)

    @app.get("/api/esp32/camera-relays")
    def esp32_camera_relays():
        """
        ESP32 poll: bay shoe presence plus per-slot fan commands from the dashboard.
        ``extra_relay_on`` is six booleans (slots 2–6 → indices 0–4; index 5 reserved).
        ``all_relays_on`` is kept false so slot fans do not force pump/motor activity; use ``shoe_detected`` only for that.
        """
        o = orch()
        snap = o.esp32_actuator_snapshot()
        shoe = bool(snap.get("shoe_detected")) if snap.get("ok") else False
        storage_ids = list(o.cfg["compartments"]["storage_ids"])
        bits = extra_relay_six_for_esp(storage_ids)
        return jsonify(
            {
                "ok": True,
                "shoe_detected": shoe,
                "all_relays_on": False,
                "motors_on": get_global_motors(),
                "pumps_on": get_global_pumps(),
                "extra_relay_on": bits,
                "mode": get_esp32_mode(),
            }
        )

    @app.get("/api/slot-fans")
    def api_slot_fans_get():
        o = orch()
        storage_ids = list(o.cfg["compartments"]["storage_ids"])
        return jsonify({"ok": True, "slots": snapshot_slots(storage_ids)})

    @app.get("/api/esp32/mode")
    def esp32_mode_get():
        return jsonify({"ok": True, "mode": get_esp32_mode()})

    @app.post("/api/esp32/mode")
    def esp32_mode_post():
        body = request.get_json(force=True, silent=True) or {}
        mode = str(body.get("mode") or "").upper()
        if mode not in ("MANUAL", "AUTO"):
            return jsonify({"ok": False, "error": "invalid_mode"}), 400
        set_esp32_mode(mode)
        log.info("esp32 mode changed UI: %s", mode)
        return jsonify({"ok": True, "mode": mode})

    @app.post("/api/slot-fans")
    def api_slot_fans_post():
        body = request.get_json(force=True, silent=True) or {}
        try:
            cid = int(body.get("compartment") or body.get("slot") or 0)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "invalid_compartment"}), 400
        o = orch()
        allowed = _storage_id_set(o.cfg["compartments"])
        if cid not in allowed:
            return jsonify({"ok": False, "error": "unknown_compartment"}), 400
        on, err = _parse_json_on(body)
        if err or on is None:
            return jsonify({"ok": False, "error": err or "invalid_on"}), 400
        
        # Sync both the polling state (for ESP32) and local GPIO
        set_slot_fan(cid, on)
        o.sensors.set_ventilation(cid, on)
        
        log.info("slot-fans UI: compartment %s -> %s", cid, on)
        return jsonify({"ok": True, "compartment": cid, "on": on})

    @app.post("/api/esp32/storage-fans")
    def esp32_storage_fans_post():
        """
        ESP32 (Thonny): after a serial fan toggle, POST state so the web UI matches the relays.
        Same auth as telemetry: if ``esp32_telemetry.secret`` is set, send ``X-ESP32-Secret`` or Bearer.
        Does not require telemetry to be enabled.

        Body (one of):
          • ``{"fans": {"2": true, "3": false, ...}}`` — keys are compartment ids (strings or ints in JSON)
          • ``{"compartment": 2, "on": true}`` or ``{"slot": 2, "on": true}`` — single slot
        """
        o = orch()
        cfg = o.cfg
        if not verify_esp32_secret(cfg, request):
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        body = request.get_json(force=True, silent=True)
        if not isinstance(body, dict):
            return jsonify({"ok": False, "error": "expected_json_object"}), 400
        storage_ids = list(o.cfg["compartments"]["storage_ids"])
        fans_obj = body.get("fans") or body.get("slots")
        if isinstance(fans_obj, dict) and fans_obj:
            n = apply_fan_report(storage_ids, fans_obj)
            log.info("esp32 storage-fans: updated %s slot(s) from batch", n)
            return jsonify({"ok": True, "updated": n, "slots": snapshot_slots(storage_ids)})
        if "compartment" in body or "slot" in body:
            try:
                cid = int(body.get("compartment") or body.get("slot") or 0)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": "invalid_compartment"}), 400
            allowed = _storage_id_set(o.cfg["compartments"])
            if cid not in allowed:
                return jsonify({"ok": False, "error": "unknown_compartment"}), 400
            on, err = _parse_json_on(body)
            if err or on is None:
                return jsonify({"ok": False, "error": err or "invalid_on"}), 400
            set_slot_fan(cid, on)
            log.info("esp32 storage-fans: slot %s -> %s", cid, on)
            return jsonify({"ok": True, "compartment": cid, "on": on})
        return jsonify({"ok": False, "error": "missing_fans_or_slot"}), 400

    @app.post("/api/esp32/telemetry")
    def esp32_telemetry_post():
        """
        MicroPython on ESP32: POST JSON from DHT22, gas ADC, HC-SR04-style distance.
        Merges into /api/status for esp32_telemetry.compartment_id when fresh.
        """
        try:
            o = orch()
            cfg = o.cfg
            block = cfg.get("esp32_telemetry") or {}
            if not block.get("enabled"):
                return jsonify({"ok": False, "error": "esp32_telemetry_disabled"}), 404
            if not verify_esp32_secret(cfg, request):
                return jsonify({"ok": False, "error": "unauthorized"}), 401
            body = request.get_json(force=True, silent=True)
            if not isinstance(body, dict):
                log.warning(
                    "esp32 telemetry: expected JSON object; content_type=%s",
                    request.content_type,
                )
                return jsonify({"ok": False, "error": "expected_json_object"}), 400
            update_from_body(body, cfg)
            log.info("esp32 telemetry ok: keys=%s", list(body.keys()))
            return jsonify({"ok": True})
        except Exception as e:
            log.exception("esp32 telemetry POST failed: %s", e)
            return jsonify({"ok": False, "error": "server_error"}), 500

    @app.get("/api/esp32/telemetry")
    def esp32_telemetry_get():
        """Debug: last payload and age (requires same secret rules when secret set)."""
        o = orch()
        cfg = o.cfg
        block = cfg.get("esp32_telemetry") or {}
        if not block.get("enabled"):
            return jsonify({"ok": False, "error": "esp32_telemetry_disabled"}), 404
        if not verify_esp32_secret(cfg, request):
            return jsonify({"ok": False, "error": "unauthorized"}), 401
        payload, age = get_last()
        snap = get_last_by_compartments()
        return jsonify(
            {
                "ok": True,
                "last": payload,
                "age_seconds": None if age < 0 else round(age, 3),
                **snap,
            }
        )

    @app.get("/api/status")
    def api_status():
        o = orch()
        snap = o.climate_snapshot()
        gm = get_global_motors()
        gp = get_global_pumps()
        for k, row in snap.items():
            try:
                cid = int(k)
            except ValueError:
                continue
            row["storage_fan_on"] = get_slot_fan(cid)
        
        # Return structured response so we don't lose global states if snap is empty
        return jsonify({
            "ok": True,
            "slots": snap,
            "global_motors_on": gm,
            "global_pumps_on": gp,
            "mode": get_esp32_mode(),
        })

    @app.post("/api/vent")
    def api_vent():
        body = request.get_json(force=True, silent=True) or {}
        cid = int(body.get("compartment", 0))
        on = bool(body.get("on", False))
        o = orch()
        if cid in _storage_id_set(o.cfg["compartments"]):
            o.sensors.set_ventilation(cid, on)
        return jsonify({"ok": True, "compartment": cid, "on": on})

    return app
