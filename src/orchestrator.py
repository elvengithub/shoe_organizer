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
from .shoe_taxonomy import SHOE_TYPE_LABELS, format_shoe_display_name
from .shoe_type_smoothing import ShoeTypeSmoother
from .text_presence import analyze_presented_text
from .vision_service import ShoeCategory, VisionResult, WebcamCapture
from .wash_decision import WashPlan, decide_wash, wash_ui_label

log = logging.getLogger(__name__)


@dataclass
class CycleResult:
    wash: WashPlan
    assigned_compartment: int | None
    message: str
    dirt_score: float | None = None
    shoe_category: str | None = None
    outcome: str = "ok"  # ok | no_camera | not_shoe | no_catalog_match | stabilizing | empty_description | analysis_failed
    catalog_category: str | None = None
    catalog_style: str | None = None
    catalog_score: float | None = None
    reject_stage: str | None = None
    classification_error: str | None = None
    reject_detail: str | None = None


class ShoeOrganizerOrchestrator:
    def __init__(self) -> None:
        self.cfg = load_config()
        from .hardware import GPIOBackend

        self.gpio = GPIOBackend()
        self.motion = ThreeAxisCartesian(self.cfg, self.gpio)
        self.sensors = CompartmentSensors(self.cfg, self.gpio)
        self.cam = CameraMux(self.cfg, WebcamCapture(self.cfg))
        self._classification_stability = ClassificationStability(self.cfg)
        st_cfg = self.cfg.get("shoe_type_smoothing") or {}
        self._type_smoother = ShoeTypeSmoother(int(st_cfg.get("window", 5)))
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
        if det.get("reject_stage") == "pipeline_error":
            return CycleResult(
                WashPlan("soft", "pipeline error"),
                None,
                str(det.get("message") or "Analysis failed."),
                dirt_score=None,
                shoe_category=None,
                outcome="analysis_failed",
                reject_stage="pipeline_error",
                classification_error=det.get("classification_error"),
                reject_detail=det.get("reject_detail"),
            )
        if not det.get("raw_is_shoe", det.get("is_shoe", True)):
            return CycleResult(
                WashPlan("soft", "not a shoe"),
                None,
                str(det.get("message") or NOT_SHOE_MESSAGE),
                dirt_score=None,
                shoe_category=None,
                outcome="not_shoe",
                reject_stage=det.get("reject_stage"),
                classification_error=det.get("classification_error"),
                reject_detail=det.get("reject_detail"),
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
                classification_error=det.get("classification_error"),
                reject_detail=det.get("reject_detail"),
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
        if detail.get("reject_stage") == "pipeline_error":
            return {
                "ok": False,
                "error": "analysis_failed",
                "message": detail.get("message"),
                "classification_error": detail.get("classification_error"),
                "reject_stage": "pipeline_error",
                "is_shoe": None,
                "catalog_match": None,
            }
        raw = bool(detail.get("raw_is_shoe", detail.get("is_shoe", False)))
        confirmed, stabilizing = self._classification_stability.tick(raw)
        detail["confirmed_is_shoe"] = confirmed
        detail["stabilizing"] = stabilizing

        if not raw:
            self._type_smoother.clear()
            return {"ok": True, "error": "not_shoe", **detail}
        if detail.get("catalog_match") is False:
            self._type_smoother.clear()
            return {"ok": True, "error": "no_catalog_match", **detail}
        if stabilizing and raw:
            d2 = {**detail, "error": "stabilizing", "message": STABILIZING_MESSAGE}
            d2["shoe_category_raw"] = detail.get("shoe_category")
            d2["shoe_category"] = None
            d2["shoe_type_label"] = None
            d2["shoe_type_short"] = None
            d2["wash_mode"] = None
            d2["wash_label"] = None
            d2["wash_reason"] = None
            d2["object_classification"] = None
            return {"ok": True, **d2}
        raw_cat = str(detail.get("shoe_category") or "casual")
        smooth_cat = self._type_smoother.update(raw_cat)
        detail["shoe_category_raw"] = raw_cat
        detail["shoe_category"] = smooth_cat
        vis = VisionResult(dirt_score=float(detail["dirt_score"]), category=ShoeCategory.CASUAL)
        wplan = decide_wash(vis, smooth_cat)
        detail["wash_mode"] = wplan.mode
        detail["wash_label"] = wash_ui_label(wplan.mode, smooth_cat)
        detail["wash_reason"] = wplan.reason
        ts = SHOE_TYPE_LABELS[smooth_cat]
        detail["shoe_type_short"] = ts
        detail["shoe_type_label"] = format_shoe_display_name(
            smooth_cat, ts, detail.get("catalog_category"), detail.get("catalog_style")
        )
        detail["object_classification"] = f"shoe_{smooth_cat}"
        scores = detail.get("shoe_type_dataset_scores")
        if isinstance(scores, dict) and smooth_cat in scores:
            v = scores[smooth_cat]
            detail["shoe_type_dataset_score"] = round(float(v), 4) if float(v) >= 0.0 else None
        return {"ok": True, "error": None, **detail}
