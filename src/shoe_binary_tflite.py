"""
Phase 2: optional MobileNetV2-sized binary TFLite model (shoe vs not shoe).
Train with scripts/train_shoe_binary.py → models/shoe_binary.tflite
Runtime: pip install tflite-runtime (Pi) or tensorflow for Interpreter.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

_interpreter = None
_in_index = None
_out_index = None
_loaded_path: str | None = None


def _get_interpreter(model_path: str):
    global _interpreter, _in_index, _out_index, _loaded_path
    if _interpreter is not None and _loaded_path == model_path:
        return _interpreter, _in_index, _out_index
    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter

        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        ins = interp.get_input_details()
        outs = interp.get_output_details()
        _interpreter = interp
        _in_index = ins[0]["index"]
        _out_index = outs[0]["index"]
        _loaded_path = model_path
        return interp, _in_index, _out_index
    except Exception as e:
        log.warning("TFLite load failed %s: %s", model_path, e)
        return None, None, None


def predict_p_shoe(bgr: np.ndarray, model_path: str, sb: dict[str, Any]) -> float | None:
    path = Path(model_path)
    if not path.is_file():
        return None
    interp, ix, ox = _get_interpreter(str(path))
    if interp is None:
        return None

    inp = interp.get_input_details()[0]
    shape = inp["shape"]
    h, w = int(shape[1]), int(shape[2])
    if h <= 0 or w <= 0:
        h = w = int(sb.get("input_size", 224))

    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb / 127.5 - 1.0
    batch = np.expand_dims(rgb, axis=0)

    if inp["dtype"] == np.uint8:
        batch = ((batch + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    interp.set_tensor(ix, batch)
    interp.invoke()
    out = np.asarray(interp.get_tensor(ox)).flatten()
    if out.size == 1:
        return float(out[0])
    if out.size == 2:
        return float(out[1])
    return float(np.max(out))
