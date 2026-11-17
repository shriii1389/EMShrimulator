"""A small pool of worker processes for GIL-bound work (resvg, gerbonara).

Submitted functions must be module-level with picklable arguments. If the pool breaks, calls run inline.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Optional

_lock = threading.Lock()
_pool: Optional[ProcessPoolExecutor] = None
_disabled = False


def workers() -> int:
    """Leave one core for the UI and the solver's own bookkeeping."""
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def _warm() -> int:
    """Imports every worker needs, so the first real task does not pay for them."""
    from EMShrimulator.core.layout import board, snap  # noqa: F401 - imported for the side effect

    return os.getpid()


def _executor() -> Optional[ProcessPoolExecutor]:
    global _pool, _disabled
    with _lock:
        if _pool is None and not _disabled:
            try:
                # spawn everywhere: forking the multi-threaded Qt process (Linux's default) can deadlock.
                _pool = ProcessPoolExecutor(max_workers=workers(), mp_context=multiprocessing.get_context("spawn"))
            except Exception:  # noqa: BLE001 - no processes available: run inline from now on
                _disabled = True
        return _pool


def _reset() -> None:
    global _pool
    with _lock:
        broken, _pool = _pool, None
    if broken is not None:
        broken.shutdown(wait=False, cancel_futures=True)


def warm() -> None:
    """Start the workers ahead of need. Returns immediately."""
    pool = _executor()
    if pool is None:
        return
    try:
        for _ in range(workers()):
            pool.submit(_warm)
    except (BrokenProcessPool, RuntimeError):
        _reset()


def call(fn: Callable[..., Any], *args: Any) -> Any:
    """Run `fn(*args)` in a worker process and return its result."""
    pool = _executor()
    if pool is not None:
        try:
            future: Optional[Future] = pool.submit(fn, *args)
        except RuntimeError:  # shut down or broken (BrokenProcessPool is a RuntimeError)
            _reset()
            future = None
        if future is not None:
            try:
                return future.result()
            except BrokenProcessPool:
                # A worker died; run it here instead.
                _reset()
    return fn(*args)


def shutdown() -> None:
    global _disabled
    with _lock:
        _disabled = True
    _reset()
