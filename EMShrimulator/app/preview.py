"""Debounced background previews (scene, mesh, validation) after modeler edits, on one worker thread."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal

# Merges edits made in the same instant; bursts already coalesce.
DEBOUNCE_MS = 10


@dataclass
class PreviewResult:
    generation: int
    scene: object = None                 # core.em.scene.Scene
    mesh: Optional[Dict] = None          # preview.mesh_preview output
    mesh_error: Optional[str] = None
    validation: Optional[Dict] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)


class _Signals(QObject):
    finished = Signal(object)


class _PreviewTask(QRunnable):
    def __init__(self, generation: int, resolution, want_mesh: bool, cache: dict, signals: _Signals):
        super().__init__()
        self.generation = generation
        self.resolution = resolution  # core.services.modeler.Resolution
        self.want_mesh = want_mesh
        self.cache = cache
        self.signals = signals

    def run(self) -> None:  # runs on the worker thread
        from EMShrimulator.core.em.scene import build_scene
        from EMShrimulator.core.services import modeler

        result = PreviewResult(self.generation)
        timings = result.timings_ms
        resolved = self.resolution.project

        started = time.perf_counter()
        try:
            result.validation = modeler.validate_resolution(self.resolution)
        except Exception as exc:  # noqa: BLE001
            result.validation = {"issues": [{"severity": "error", "code": "validator-crashed",
                                             "message": f"The validator failed: {exc}", "hint": "",
                                             "target": None}],
                                 "errors": 1, "warnings": 0, "infos": 0, "can_run": False}
        timings["validate"] = (time.perf_counter() - started) * 1000

        resolution_mm = 0.0
        if self.want_mesh:
            started = time.perf_counter()
            try:
                result.mesh = modeler.mesh_preview(resolved)
                resolution_mm = float(result.mesh.get("resolution_mm") or 0.0)
            except Exception as exc:  # noqa: BLE001 - a preview failure is a message, not a crash
                result.mesh_error = str(exc) or exc.__class__.__name__
                traceback.print_exc()
            timings["mesh"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        try:
            result.scene = build_scene(resolved, self.cache, resolution_mm)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        timings["scene"] = (time.perf_counter() - started) * 1000
        self.signals.finished.emit(result)


class PreviewScheduler(QObject):
    """Debounces preview requests and delivers only the newest result."""

    ready = Signal(object)      # PreviewResult
    busyChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._generation = 0
        self._pending: Optional[tuple] = None
        self._running = 0
        # The tessellation cache is only ever touched by the single worker.
        self._cache: dict = {}
        self._signals = _Signals()
        self._signals.finished.connect(self._deliver)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._submit)

    def request(self, resolution, want_mesh: bool = True) -> None:
        """Preview a session's resolution."""
        self._generation += 1
        self._pending = (self._generation, resolution, want_mesh)
        self._timer.start()

    def request_now(self, resolution, want_mesh: bool = True) -> None:
        self.request(resolution, want_mesh)
        self._timer.stop()
        self._submit()

    def _submit(self) -> None:
        if self._pending is None:
            return
        if self._running:
            # One preview at a time; only the newest request waits.
            return
        generation, resolution, want_mesh = self._pending
        self._pending = None
        self._running += 1
        self.busyChanged.emit(True)
        self._pool.start(_PreviewTask(generation, resolution, want_mesh, self._cache, self._signals))

    def _deliver(self, result: PreviewResult) -> None:
        self._running = max(self._running - 1, 0)
        if result.generation == self._generation:
            self.ready.emit(result)
        if self._pending is not None:
            self._submit()
        elif self._running == 0:
            self.busyChanged.emit(False)

    def shutdown(self) -> None:
        self._timer.stop()
        self._pending = None
        self._pool.clear()
        self._pool.waitForDone(5000)
