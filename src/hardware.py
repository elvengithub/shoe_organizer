from __future__ import annotations

import logging
import time
from typing import Callable

from .config_loader import is_mock_hardware

log = logging.getLogger(__name__)

try:
    import RPi.GPIO as GPIO  # type: ignore

    _HAS_GPIO = True
except ImportError:
    GPIO = None  # type: ignore
    _HAS_GPIO = False


class GPIOBackend:
    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    PUD_UP = "PUD_UP"
    LOW = 0
    HIGH = 1

    def __init__(self, mock: bool | None = None) -> None:
        self._mock = mock if mock is not None else (is_mock_hardware() or not _HAS_GPIO)
        self._out: dict[int, int] = {}
        self._in: dict[int, int] = {}
        if not self._mock:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

    def setup_output(self, pin: int, initial: int = 0) -> None:
        if self._mock:
            self._out[pin] = initial
            return
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH if initial else GPIO.LOW)

    def setup_input_pullup(self, pin: int) -> None:
        if self._mock:
            self._in[pin] = 1
            return
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def output(self, pin: int, value: int) -> None:
        if self._mock:
            self._out[pin] = value
            return
        GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)

    def input(self, pin: int) -> int:
        if self._mock:
            return self._in.get(pin, 1)
        return 1 if GPIO.input(pin) else 0

    def set_mock_input(self, pin: int, value: int) -> None:
        self._in[pin] = value

    def cleanup(self) -> None:
        if not self._mock and _HAS_GPIO:
            GPIO.cleanup()


def pulse_step(step_pin: int, backend: GPIOBackend, delay_s: float = 0.0005) -> None:
    backend.output(step_pin, 1)
    time.sleep(delay_s)
    backend.output(step_pin, 0)
    time.sleep(delay_s)
