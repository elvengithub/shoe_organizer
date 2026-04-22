from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

from flask import Flask, Response, jsonify, render_template, request

from .esp32_telemetry import get_last, get_last_by_compartments, update_from_body, verify_esp32_secret
from .orchestrator import ShoeOrganizerOrchestrator
from .text_presence import analyze_presented_text
from .shoe_taxonomy import SHOE_TYPE_LABELS, format_shoe_display_name
from .wash_decision import wash_ui_label
from .vision_service import blank_jpeg_bytes, encode_jpeg

log = logging.getLogger(__name__)


def create_app() -> Flask:
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)))
    app = Flask(__name__, template_folder=os.path.join(root, "templates"), static_folder=os.path.join(root, "static"))

    @lru_cache(maxsize=1)
    def orch() -> ShoeOrganizerOrchestrator:
        return ShoeOrganizerOrchestrator()

    @app.route("/")
    def index():
        return render_template("index.html")

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
        ESP32 poll: when the bay camera sees a shoe, pump_on and fan_on are true (same Wi‑Fi / LAN as SERVER_BASE).
        Trusted LAN only — no secret (same policy as /api/esp32/ping).
        """
        return jsonify(orch().esp32_actuator_snapshot())

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
        return jsonify(orch().climate_snapshot())

    @app.post("/api/vent")
    def api_vent():
        body = request.get_json(force=True, silent=True) or {}
        cid = int(body.get("compartment", 0))
        on = bool(body.get("on", False))
        o = orch()
        if cid in o.cfg["compartments"]["storage_ids"]:
            o.sensors.set_ventilation(cid, on)
        return jsonify({"ok": True, "compartment": cid, "on": on})

    return app
