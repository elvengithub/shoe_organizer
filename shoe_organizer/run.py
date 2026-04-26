"""
Shoe organizer — Raspberry Pi 4 control stack.
Development on PC: set MOCK_HARDWARE=1
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading

# Add project root for `python run.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_loader import load_config  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    host = cfg["server"]["host"]
    port = int(cfg["server"]["port"])
    from src.app import create_app
    from werkzeug.serving import make_server

    app = create_app()
    # app.run() + threaded=True often ignores Ctrl+C on Windows (stuck in serve_forever with
    # open clients / camera stream). make_server + shutdown() fixes that.
    srv = make_server(host, port, app, threaded=True, processes=1)
    srv.log_startup()

    def _request_shutdown(*_a: object) -> None:
        threading.Thread(target=srv.shutdown, name="werkzeug-shutdown", daemon=True).start()

    signal.signal(signal.SIGINT, _request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_shutdown)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
