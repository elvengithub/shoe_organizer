"""
Shoe organizer — Raspberry Pi 4 control stack.
Development on PC: set MOCK_HARDWARE=1
"""
from __future__ import annotations

import logging
import os
import sys

# Add project root for `python run.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_loader import load_config  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    host = cfg["server"]["host"]
    port = int(cfg["server"]["port"])
    from src.app import create_app

    app = create_app()
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
