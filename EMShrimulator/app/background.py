"""Run slow calls on a single worker thread and deliver only the newest result."""
from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class _Signals(QObject):
    done = Signal(str, int, object, object)  # channel, generation, value, error


class _Task(QRunnable):
    def __init__(self, channel: str, generation: int, fn: Callable[[], Any], signals: _Signals):
        super().__init__()
        self.channel, self.generation, self.fn, self.signals = channel, generation, fn, signals

    def run(self) -> None:
        try:
            value = self.fn()
        except Exception as exc:  # noqa: BLE001 - delivered to the caller's error handler
            traceback.print_exc()
            self.signals.done.emit(self.channel, self.generation, None, exc)
            return
        self.signals.done.emit(self.channel, self.generation, value, None)


class BackgroundRunner(QObject):
    """Independent channels ("board", "raster:F_Cu"…), each keeping only its newest call."""

    def __init__(self, parent=None, threads: int = 1):
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(threads)
        self._generations: Dict[str, int] = {}
        self._callbacks: Dict[tuple, tuple] = {}
        self._signals = _Signals()
        self._signals.done.connect(self._deliver)

    def submit(self, channel: str, fn: Callable[[], Any], on_done: Callable[[Any], None],
               on_error: Optional[Callable[[Exception], None]] = None) -> None:
        generation = self._generations.get(channel, 0) + 1
        self._generations[channel] = generation
        self._callbacks[(channel, generation)] = (on_done, on_error)
        self._pool.start(_Task(channel, generation, fn, self._signals))

    def cancel(self, channel: str) -> None:
        self._generations[channel] = self._generations.get(channel, 0) + 1

    def _deliver(self, channel: str, generation: int, value: Any, error: Optional[Exception]) -> None:
        on_done, on_error = self._callbacks.pop((channel, generation), (None, None))
        if generation != self._generations.get(channel):
            return
        if error is not None:
            if on_error is not None:
                on_error(error)
        elif on_done is not None:
            on_done(value)

    def shutdown(self) -> None:
        for channel in list(self._generations):
            self.cancel(channel)
        self._pool.clear()
        self._pool.waitForDone(5000)
