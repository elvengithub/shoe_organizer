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
from .shoe_taxonomy import (
    SHOE_TYPE_LABELS,
    format_neural_shoe_label,
    format_shoe_display_name,
    pretty_dataset_class_name,
)
from .shoe_type_smoothing import ShoeTypeSmoother
from .text_presence import analyze_presented_text
from .vision_service import ShoeCategory, VisionResult, WebcamCapture
from .slot_fan_state import get_esp32_mode, set_slot_fan
from .wash_actuator_sequence import WashBayActuatorSequence
from .wash_decision import WashPlan, decide_wash, wash_ui_label

log = logging.getLogger(__name__)


def _vision_rule_debug(
    vision: VisionResult | None, cfg: dict,
) -> tuple[float | None, float | None, float | None, float | None]:
    """When rule_based_pipeline is on, expose edge + fusion scores for API / tuning."""
    vm = cfg.get("vision", {})
    if not vision or not vm.get("rule_based_pipeline"):
        return None, None, None, None
    thr_edge = float(vm.get("sports_edge_density_min", 0.09))
    ed = vision.edge_density
    fs = vision.sports_fusion_score
    ft = vision.sports_fusion_threshold
    return (
        float(ed) if ed is not None else None,
        thr_edge,
        float(fs) if fs is not None else None,
        float(ft) if ft is not None else None,
    )


@dataclass
class _ActiveRouting:
    """Last intake assignment to a storage slot (for kiosk order status on slot cards)."""

    slot: int
    line: str
    motion_error: bool = False
    saw_occupied: bool = False


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
    edge_density: float | None = None
    sports_edge_density_min: float | None = None
    sports_fusion_score: float | None = None
    sports_fusion_threshold: float | None = None


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
        self._active_routing: _ActiveRouting | None = None
        self._wash_seq = WashBayActuatorSequence(self.cfg.get("wash_actuators", {}))

    def _esp32_actuator_quiet(self) -> dict:
        """Return a snapshot representing all outputs OFF (no side effects)."""
        return {
            "pump1_on": False,
            "pump2_on": False,
            "motors_on": False,
            "wash_sequence_state": "idle",
            "wash_sequence_phase": None,
            "wash_sequence_cycle": None,
            "wash_sequence_cycle_index": 0,
            "wash_sequence_repeat_total": self._wash_seq.repeat_total,
            "pump_on": False,
            "fan_on": False,
        }

    def _set_active_routing(
        self,
        slot: int,
        wash: WashPlan,
        shoe_category: str | None,
        catalog_category: str | None,
        catalog_style: str | None,
        *,
        motion_error: bool,
    ) -> None:
        st = shoe_category or "casual"
        sl = SHOE_TYPE_LABELS.get(st, st.title())
        label = format_shoe_display_name(st, sl, catalog_category, catalog_style)
        wl = wash_ui_label(wash.mode, st)
        self._active_routing = _ActiveRouting(
            slot=int(slot),
            line=f"{label} · {wl}",
            motion_error=motion_error,
            saw_occupied=False,
        )

    def _tick_active_routing(self) -> None:
        if self._active_routing is None:
            return
        ar = self._active_routing
        if self.sensors.occupancy_occupied(ar.slot):
            ar.saw_occupied = True
        elif ar.saw_occupied:
            self._active_routing = None

    def _order_status_for_slot(self, cid: int) -> str:
        ar = self._active_routing
        if ar is None or ar.slot != cid:
            return "—"
        if ar.motion_error:
            return f"Motion error · {ar.line}"
        if self.sensors.occupancy_occupied(cid):
            return f"Stored · {ar.line}"
        return f"Awaiting placement · {ar.line}"

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
                self._set_active_routing(slot, wash, shoe_cat, cc, cs, motion_error=True)
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
            # Only turn on the fan for the assigned slot, don't force others OFF
            self.sensors.set_ventilation(slot, True)
            set_slot_fan(slot, True)
            self._set_active_routing(slot, wash, shoe_cat, cc, cs, motion_error=False)
            return CycleResult(
                wash,
                slot,
                f"{'No wash — shoe appears clean' if str(wash.mode).lower() == 'none' else wash.mode.upper() + ' wash'}; place into compartment {slot}; vent ON there only.",
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
        if not det.get("raw_is_shoe", det.get("is_shoe", False)):
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
        if det.get("raw_is_shoe", False) and not self._classification_stability.confirmed():
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
        ed_dbg, thr_dbg, fus_dbg, fus_thr_dbg = _vision_rule_debug(vision, self.cfg)
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
                edge_density=ed_dbg,
                sports_edge_density_min=thr_dbg,
                sports_fusion_score=fus_dbg,
                sports_fusion_threshold=fus_thr_dbg,
            )
        try:
            self.motion.goto_compartment_y_index(slot)
        except Exception as e:
            log.exception("motion failed")
            self._set_active_routing(slot, wash, det.get("shoe_category"), cc, cs, motion_error=True)
            return CycleResult(
                wash,
                slot,
                f"Wash={wash.mode}; motion error: {e}",
                dirt_score=vision.dirt_score,
                shoe_category=det.get("shoe_category"),
                catalog_category=cc,
                catalog_style=cs,
                catalog_score=csc,
                edge_density=ed_dbg,
                sports_edge_density_min=thr_dbg,
                sports_fusion_score=fus_dbg,
                sports_fusion_threshold=fus_thr_dbg,
            )
        from .slot_fan_state import set_slot_fan
        # Only turn on the fan for the assigned slot later, don't force others OFF here
        
        # Only turn on the fan immediately if no wash is needed
        if str(wash.mode).lower() == "none":
            self.sensors.set_ventilation(slot, True)
            set_slot_fan(slot, True)
        self._set_active_routing(slot, wash, det.get("shoe_category"), cc, cs, motion_error=False)
        return CycleResult(
            wash,
            slot,
            f"{'No wash — shoe appears clean' if str(wash.mode).lower() == 'none' else wash.mode.upper() + ' wash'}; place into compartment {slot}; vent ON there only.",
            dirt_score=vision.dirt_score,
            shoe_category=det.get("shoe_category"),
            catalog_category=cc,
            catalog_style=cs,
            catalog_score=csc,
            edge_density=ed_dbg,
            sports_edge_density_min=thr_dbg,
            sports_fusion_score=fus_dbg,
            sports_fusion_threshold=fus_thr_dbg,
        )

    def climate_snapshot(self) -> dict:
        self._tick_active_routing()
        out = {}
        for cid in self.cfg["compartments"]["storage_ids"]:
            r = self.sensors.read_climate(cid)
            out[str(cid)] = {
                "temperature_c": r.temperature_c,
                "humidity_pct": r.humidity_pct,
                "occupied": self.sensors.occupancy_occupied(cid),
                "vent_on": self.sensors.ventilation_on(cid),
                "order_status": self._order_status_for_slot(int(cid)),
            }
        apply_to_climate_snapshot(out, self.cfg)
        return out

    def esp32_actuator_snapshot(self) -> dict:
        """
        One-frame shoe presence for ESP32 plus timed wash sequence (pump1 / motors / pump2 / motors × N).
        Does not call classification stability (avoids interfering with /api/camera/analyze state).
        """
        quiet = self._esp32_actuator_quiet()
        if self._text_mode():
            return {
                "ok": True,
                "error": "presenter_text_mode",
                "shoe_detected": False,
                "shoe_clean": False,
                "status": "idle",
                **quiet,
            }
        frame = self.cam.read()
        if frame is None:
            src = str(self.cfg.get("camera", {}).get("source", "usb")).lower()
            return {
                "ok": False,
                "error": "camera_unavailable",
                "shoe_detected": False,
                "shoe_clean": False,
                "status": "idle",
                "message": (
                    "Waiting for ESP32 camera frame…"
                    if src == "esp32"
                    else "No camera frame — check USB or POST /api/camera/frame"
                ),
                **quiet,
            }
        _v, _w, detail = analyze_shoe_and_wash_from_bgr(frame)
        if detail.get("reject_stage") == "pipeline_error":
            return {
                "ok": False,
                "error": "analysis_failed",
                "shoe_detected": False,
                "shoe_clean": False,
                "status": "idle",
                **quiet,
            }
        raw = bool(detail.get("raw_is_shoe", detail.get("is_shoe", False)))
        wash_mode = str(detail.get("wash_mode") or "").strip().lower()
        shoe_clean = bool(raw and wash_mode == "none")
        st = "clean" if shoe_clean else ("wash" if raw else "idle")
        is_auto = (get_esp32_mode().upper() == "AUTO")
        seq = self._wash_seq.tick(raw_shoe=raw, shoe_clean=shoe_clean, auto_start=is_auto)
        
        # If a manual override is active (from the web UI), it wins
        if self._wash_seq.manual_mode_override:
            wash_mode = self._wash_seq.manual_mode_override
            # Force shoe_detected so the ESP32 internal engine doesn't skip the wash
            raw = True 

        return {
            "ok": True,
            "error": None,
            "shoe_detected": raw,
            "shoe_clean": shoe_clean,
            "wash_mode": wash_mode,
            "status": st,
            **seq,
        }

    def trigger_wash(self, mode: str) -> None:
        print(f">>> [ORCHESTRATOR] Manual wash trigger received: {mode}")
        self._wash_seq.trigger_manual(mode)

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
        dl = detail.get("dirt_level")
        ed = detail.get("edge_density")
        dpr = detail.get("dirty_pixel_ratio")
        sfs = detail.get("sports_fusion_score")
        sft = detail.get("sports_fusion_threshold")
        gm = detail.get("gradient_mean")
        tr = detail.get("texture_rms")
        sm = detail.get("saturation_mean")
        vis = VisionResult(
            dirt_score=float(detail["dirt_score"]),
            category=ShoeCategory.CASUAL,
            dirt_level=str(dl) if dl is not None else None,
            edge_density=float(ed) if ed is not None else None,
            dirty_pixel_ratio=float(dpr) if dpr is not None else None,
            sports_fusion_score=float(sfs) if sfs is not None else None,
            sports_fusion_threshold=float(sft) if sft is not None else None,
            gradient_mean=float(gm) if gm is not None else None,
            texture_rms=float(tr) if tr is not None else None,
            saturation_mean=float(sm) if sm is not None else None,
        )
        wplan = decide_wash(vis, smooth_cat)
        detail["wash_mode"] = wplan.mode
        detail["wash_label"] = wash_ui_label(wplan.mode, smooth_cat)
        detail["wash_reason"] = wplan.reason
        ts = SHOE_TYPE_LABELS.get(smooth_cat, (smooth_cat or "casual").title())
        detail["shoe_type_short"] = ts
        ap_pipe = self.cfg.get("ai_pipeline") or {}
        ds_ui = ap_pipe.get("dataset") or {}
        dataset_first = bool(ds_ui.get("show_dataset_class_in_ui", True))
        ft_raw = detail.get("fine_shoe_type")
        if (
            ft_raw is not None
            and str(detail.get("inference_backend", "")).startswith("yolov8")
        ):
            ft = str(ft_raw)
            detail["shoe_dataset_class"] = ft
            detail["shoe_dataset_style"] = pretty_dataset_class_name(ft)
            detail["shoe_type_label"] = format_neural_shoe_label(
                smooth_cat,
                ts,
                ft,
                catalog_category=detail.get("catalog_category"),
                catalog_style=detail.get("catalog_style"),
                dataset_class_first=dataset_first,
            )
        else:
            detail["shoe_type_label"] = format_shoe_display_name(
                smooth_cat, ts, detail.get("catalog_category"), detail.get("catalog_style")
            )
        detail["object_classification"] = f"shoe_{smooth_cat}"
        scores = detail.get("shoe_type_dataset_scores")
        if isinstance(scores, dict) and smooth_cat in scores:
            v = scores[smooth_cat]
            detail["shoe_type_dataset_score"] = round(float(v), 4) if float(v) >= 0.0 else None
        vm = self.cfg.get("vision", {})
        if vm.get("rule_based_pipeline") and detail.get("sports_edge_density_min") is None:
            detail["sports_edge_density_min"] = round(float(vm.get("sports_edge_density_min", 0.09)), 4)
        if vm.get("rule_based_pipeline") and detail.get("sports_fusion_threshold") is None:
            detail["sports_fusion_threshold"] = round(float(vm.get("sports_fusion_threshold", 0.48)), 4)
        
        # Include the wash sequence state so the UI stays in sync
        shoe_clean = bool(raw and detail.get("wash_mode") == "none")
        is_auto = (get_esp32_mode().upper() == "AUTO")
        seq = self._wash_seq.tick(raw_shoe=raw, shoe_clean=shoe_clean, auto_start=is_auto)
        
        # If manual override is active, it wins
        if self._wash_seq._manual_mode_override:
            detail["wash_mode"] = self._wash_seq._manual_mode_override
            detail["wash_label"] = "Gentle Wash" if detail["wash_mode"] == "soft" else "Deep Wash"
            detail["wash_reason"] = "Manual trigger"
            # Force shoe_detected so the UI doesn't hide the wash stats
            detail["is_shoe"] = True
            detail["confirmed_is_shoe"] = True

        return {"ok": True, "error": None, **detail, **seq}
