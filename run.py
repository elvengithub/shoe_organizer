"""
Run the app from the outer folder (the one that contains the inner `shoe_organizer` project).
"""
from __future__ import annotations

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.join(_ROOT, "shoe_organizer")
_INNER = os.path.join(_APP, "run.py")

if not os.path.isfile(_INNER):
    sys.stderr.write(
        "Missing shoe_organizer\\run.py. Open the folder that contains both "
        "this file and the inner 'shoe_organizer' project folder.\n"
    )
    sys.exit(1)

os.chdir(_APP)
sys.path.insert(0, _APP)
runpy.run_path(_INNER, run_name="__main__")
