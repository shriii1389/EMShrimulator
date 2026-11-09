"""Import before CSXCAD/openEMS: registers the openEMS DLL directory (Python 3.8+ ignores PATH for DLLs)."""
from __future__ import annotations

import os

from EMShrimulator.config import OPENEMS_DIR

_added = False


def ensure_dll_path() -> None:
    global _added
    if _added:
        return
    if hasattr(os, "add_dll_directory"):
        if OPENEMS_DIR.is_dir():
            os.add_dll_directory(str(OPENEMS_DIR))
    else:  # pragma: no cover - non-Windows fallback
        os.environ["PATH"] = str(OPENEMS_DIR) + os.pathsep + os.environ.get("PATH", "")
    _added = True


ensure_dll_path()
