from __future__ import annotations

import logging
from dataclasses import dataclass

from .ai_camera import NO_CATALOG_MESSAGE, NOT_SHOE_MESSAGE, STABILIZING_MESSAGE, analyze_shoe_and_wash_from_bgr
from .classification_stability import ClassificationStability
from .camera_mux import CameraMux
from .config_loader import load_config
from .sensors import CompartmentSensors
from .stepper_3axis import ThreeAxisCartesian
from .esp32_telemetry import apply_to_climate_snapshot
from .serial_bridge import SerialBridge
from .text_presence import analyze_presented_text
from .vision_service import WebcamCapture
from .wash_decision import WashPlan

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    wash: WashPlan
    assigned_compartment: int | None
    message: str
    dirt_score: float | None = None
    shoe_category: str | None = None
    outcome: str = "ok"  # ok | no_camera | not_shoe | no_catalog_match | stabilizing | empty_description
    catalog_category: str | None = None
    catalog_style: str | None = None
    catalog_score: float | None = None


class ShoeOrganizerOrchestrator:
    def __init__(self) -> None:
        self.cfg = load_config()
        from .hardware import GPIOBackend

        self.gpio = GPIOBackend()
        self.motion = ThreeAxisCartesian(self.cfg, self.gpio)
        self.sensors = CompartmentSensors(self.cfg, self.gpio)
        self.cam = CameraMux(self.cfg, WebcamCapture(self.cfg))
        self._classification_stability = ClassificationStability(self.cfg)
        self._serial_bridge = SerialBridge(self.cfg)
        self._serial_bridge.start()

    def _text_mode(self) -> bool:
        return str(self.cfg.get("presenter", {}).get("mode", "camera")).lower() == "text"

    def pick_free_storage_slot(self) -> int | None:
        storage = self.cfg["compartments"]["storage_ids"]
        for cid in storage:
            if not self.sensors.occupancy_occupied(cid):
                return int(cid)
        return None

    def run_intake_cycle(self, description: str | None = None) -> CycleResult:
        use_text = self._text_mode() or description is not None
        if use_text:
            text = ((description or "").strip()) if description is not None else ""
            det = analyze_presented_text(text, self.cfg)
            if det.get("error") == "empty_description" or det.get("ok") is False:
                return CycleResult(
                    WashPlan("soft", "empty"),
                    None,
                    det.get("message", "Describe what is presented."),
                    outcome="empty_description",
                )
            if det.get("error") == "not_shoe" or not det.get("is_shoe", False):
                return CycleResult(
                    WashPlan("soft", "not a shoe"),
                    None,
                    NOT_SHOE_MESSAGE,
                    dirt_score=None,
                    shoe_category=None,
                    outcome="not_shoe",
                )
            wash = WashPlan(str(det["wash_mode"]), str(det["wash_reason"]))
            d = float(det.get("dirt_score") or 0.0)
            cc = det.get("catalog_category")
            cs = det.get("catalog_style")
            csc = float(det["catalog_score"]) if det.get("catalog_score") is not None else None
            shoe_cat = det.get("shoe_category")
            slot = self.pick_free_storage_slot()
            if slot is None:
                return CycleResult(
                    wash,
                    None,
                    "All storage compartments report occupied — clear a slot.",
                    dirt_score=d,
                    shoe_category=shoe_cat,
                    catalog_category=cc,
                    catalog_style=cs,
                    catalog_score=csc,
                )
            try:
                self.motion.goto_compartment_y_index(slot)
            except Exception as e:
                log.exception("motion failed")
                return CycleResult(
                    wash,
                    slot,
                    f"Wash={wash.mode}; motion error: {e}",
                    dirt_score=d,
                    shoe_category=shoe_cat,
                    catalog_category=cc,
                    catalog_style=cs,
                    catalog_score=csc,
                )
            for cid in self.cfg["compartments"]["storage_ids"]:
                self.sensors.set_ventilation(cid, False)
            self.sensors.set_ventilation(slot, True)
            return CycleResult(
                wash,
                slot,
                f"{wash.mode.upper()} wash; place into compartment {slot}; vent ON there only.",
                dirt_score=d,
                shoe_category=shoe_cat,
                catalog_category=cc,
                catalog_style=cs,
                catalog_score=csc,
            )

        frame = self.cam.read()
        if frame is None:
            src = str(self.cfg.get("camera", {}).get("source", "usb")).lower()
            hint = (
                "Waiting for ESP32 frame — POST JPEG to /api/camera/frame"
                if src == "esp32"
                else "Camera error — check USB index or ESP32 upload in config.yaml"
            )
            return CycleResult(
                WashPlan("soft", "no camera"),
                None,
                hint,
                outcome="no_camera",
            )
        vision, wash, det = analyze_shoe_and_wash_from_bgr(frame)
        if not det.get("raw_is_shoe", det.get("is_shoe", True)):
            return CycleResult(
                WashPlan("soft", "not a shoe"),
                None,
                NOT_SHOE_MESSAGE,
                dirt_score=None,
                shoe_category=None,
                outcome="not_shoe",
            )
        if det.get("raw_is_shoe", True) and not self._classification_stability.confirmed():
            return CycleResult(
                WashPlan("soft", "stabilizing"),
                None,
                STABILIZING_MESSAGE,
                dirt_score=None,
                shoe_category=None,
                outcome="stabilizing",
            )
        if det.get("catalog_match") is False:
            return CycleResult(
                WashPlan("soft", "catalog mismatch"),
                None,
                NO_CATALOG_MESSAGE,
                dirt_score=None,
                shoe_category=None,
                outcome="no_catalog_match",
            )
        assert vision is not None and wash is not None
        cc = det.get("catalog_category")
        cs = det.get("catalog_style")
        raw_sc = det.get("catalog_score")
        try:
            csc = float(raw_sc) if raw_sc is not None else None
        except (TypeError, ValueError):
            csc = None
        slot = self.pick_free_storage_slot()
        if slot is None:
            return CycleResult(
                wash,
                None,
                "All storage compartments report occupied — clear a slot.",
                dirt_score=vision.dirt_score,
                shoe_category=det.get("shoe_category"),
                catalog_category=cc,
                catalog_style=cs,
                catalog_score=csc,
            )
        try:
            self.motion.goto_compartment_y_index(slot)
        except Exception as e:
            log.exception("motion failed")
            return CycleResult(
                wash,
                slot,
                f"Wash={wash.mode}; motion error: {e}",
                dirt_score=vision.dirt_score,
                shoe_category=det.get("shoe_category"),
                catalog_category=cc,
                catalog_style=cs,
                catalog_score=csc,
            )
        for cid in self.cfg["compartments"]["storage_ids"]:
            self.sensors.set_ventilation(cid, False)
        self.sensors.set_ventilation(slot, True)
        return CycleResult(
            wash,
            slot,
            f"{wash.mode.upper()} wash; place into compartment {slot}; vent ON there only.",
            dirt_score=vision.dirt_score,
            shoe_category=det.get("shoe_category"),
            catalog_category=cc,
            catalog_style=cs,
            catalog_score=csc,
        )

    def climate_snapshot(self) -> dict:
        out = {}
        for cid in self.cfg["compartments"]["storage_ids"]:
            r = self.sensors.read_climate(cid)
            out[str(cid)] = {
                "temperature_c": r.temperature_c,
                "humidity_pct": r.humidity_pct,
                "occupied": self.sensors.occupancy_occupied(cid),
                "vent_on": self.sensors.ventilation_on(cid),
            }
        apply_to_climate_snapshot(out, self.cfg)
        return out

    def analyze_text_live(self, description: str) -> dict:
        """Same JSON shape as camera analyze, from free text only (no images)."""
        return analyze_presented_text(description, self.cfg)

    def analyze_camera_live(self) -> dict:
        """Automatic identification: shoe vs not shoe, type + wash when shoe (no motion)."""
        if self._text_mode():
            return {
                "ok": False,
                "error": "presenter_text_mode",
                "message": "Camera analyze is disabled — use text description (POST /api/analyze).",
                "input_mode": "text",
            }
        frame = self.cam.read()
        if frame is None:
            src = str(self.cfg.get("camera", {}).get("source", "usb")).lower()
            msg = (
                "Waiting for ESP32 camera frame…"
                if src == "esp32"
                else "No camera frame — check USB or POST /api/camera/frame"
            )
            return {
                "ok": False,
                "error": "camera_unavailable",
                "message": msg,
            }
        _v, _w, detail = analyze_shoe_and_wash_from_bgr(frame)
        raw = bool(detail.get("raw_is_shoe", detail.get("is_shoe", False)))
        confirmed, stabilizing = self._classification_stability.tick(raw)
        detail["confirmed_is_shoe"] = confirmed
        detail["stabilizing"] = stabilizing

        if not raw:
            return {"ok": True, "error": "not_shoe", **detail}
        if detail.get("catalog_match") is False:
            return {"ok": True, "error": "no_catalog_match", **detail}
        if stabilizing and raw:
            d2 = {**detail, "error": "stabilizing", "message": STABILIZING_MESSAGE}
            return {"ok": True, **d2}
        return {"ok": True, "error": None, **detail}
