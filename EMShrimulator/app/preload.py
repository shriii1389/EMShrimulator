"""Import the heavy libraries in the background while the start page is showing.

Call `wait()` before importing them on the UI thread.
"""
from __future__ import annotations

import importlib
import threading
from typing import Optional

MODULES = (
    "numpy",
    "pydantic",
    "EMShrimulator.core.services.modeler",
    "EMShrimulator.core.services.layout",
    "EMShrimulator.core.em.scene",
    "EMShrimulator.app.views.viewport3d",       # VTK and its Qt widget
    "EMShrimulator.app.views.modeler.modeler_view",
    "EMShrimulator.app.views.layout.layout_view",
    "EMShrimulator.core.em.preview",            # CSXCAD and openEMS, for the first mesh preview
)

_thread: Optional[threading.Thread] = None


def _run() -> None:
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - the real import reports it, with context, when it is needed
            pass


def start() -> None:
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_run, name="preload", daemon=True)
        _thread.start()


def wait() -> None:
    if _thread is not None:
        _thread.join()
