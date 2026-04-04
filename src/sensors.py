from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from .config_loader import is_mock_hardware, load_config
from .hardware import GPIOBackend

log = logging.getLogger(__name__)


@dataclass
class ClimateReading:
    compartment_id: int
    temperature_c: float | None
    humidity_pct: float | None
    ok: bool


def read_climate_mock(compartment_id: int) -> ClimateReading:
    return ClimateReading(
        compartment_id,
        round(26.0 + random.uniform(-1, 1), 1),
        round(55.0 + random.uniform(-5, 5), 1),
        True,
    )


def _make_dht(board_pin_name: str):
    import board  # type: ignore
    import adafruit_dht  # type: ignore

    pin = getattr(board, board_pin_name, None)
    if pin is None:
        return None
    return adafruit_dht.DHT22(pin)


class CompartmentSensors:
    def __init__(self, cfg: dict | None = None, backend: GPIOBackend | None = None) -> None:
        self.cfg = cfg or load_config()
        self.backend = backend or GPIOBackend()
        self.comp = self.cfg["compartments"]
        for _cid, pin in self.comp["occupancy_gpio"].items():
            self.backend.setup_input_pullup(pin)
        self._vent_on: dict[int, bool] = {}
        for cid, pin in self.comp["vent_gpio"].items():
            self.backend.setup_output(pin, 0)
            self.backend.output(pin, 0)
            self._vent_on[int(cid)] = False
        self._dht_board_names: dict[int, str] = self.comp.get("dht_board_pin", {})
        self._dht_cache: dict[int, object] = {}

    def occupancy_occupied(self, compartment_id: int) -> bool:
        pin = self.comp["occupancy_gpio"].get(compartment_id)
        if pin is None:
            return False
        v = self.backend.input(pin)
        return v == 0

    def set_ventilation(self, compartment_id: int, on: bool) -> None:
        pin = self.comp["vent_gpio"].get(compartment_id)
        if pin is None:
            return
        self.backend.output(pin, 1 if on else 0)
        self._vent_on[int(compartment_id)] = bool(on)

    def ventilation_on(self, compartment_id: int) -> bool:
        return self._vent_on.get(int(compartment_id), False)

    def read_climate(self, compartment_id: int) -> ClimateReading:
        if is_mock_hardware():
            return read_climate_mock(compartment_id)
        name = self._dht_board_names.get(compartment_id)
        if not name:
            return read_climate_mock(compartment_id)
        if compartment_id not in self._dht_cache:
            try:
                self._dht_cache[compartment_id] = _make_dht(name)
            except ImportError:
                log.warning("adafruit_dht not installed; mock climate")
                self._dht_cache[compartment_id] = "SKIP"
        cached = self._dht_cache.get(compartment_id)
        if cached == "SKIP":
            return read_climate_mock(compartment_id)
        sensor = cached
        if sensor is None:
            return ClimateReading(compartment_id, None, None, False)
        try:
            t = sensor.temperature
            h = sensor.humidity
            ok = t is not None and h is not None
            return ClimateReading(compartment_id, t, h, ok)
        except RuntimeError:
            return ClimateReading(compartment_id, None, None, False)
        finally:
            time.sleep(0.05)
