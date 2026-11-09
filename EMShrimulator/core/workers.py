"""Solver subprocess launch: `python -m` from source, gems-worker.exe when frozen."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List

from EMShrimulator.config import FROZEN, OPENEMS_DIR, REPO_ROOT

WORKERS: Dict[str, str] = {
    "run": "EMShrimulator.core.em.run_worker",
    "sweep": "EMShrimulator.core.em.sweep_worker",
    "gerber2ems": "gerber2ems.main",
    "doctor": "EMShrimulator.core.diagnostics",
}

WORKER_EXE_NAME = "gems-worker.exe" if sys.platform == "win32" else "gems-worker"


def _console_python() -> str:
    """This interpreter, swapping pythonw for python so children get a stdout."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        candidate = exe.with_name("python.exe")
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def command(worker: str, *args: str) -> List[str]:
    """The argv that runs `worker` with `args`."""
    if worker not in WORKERS:
        raise ValueError(f"unknown worker '{worker}'")
    if FROZEN:
        return [str(Path(sys.executable).resolve().parent / WORKER_EXE_NAME), worker, *args]
    return [_console_python(), "-m", "EMShrimulator.core.workers", worker, *args]


def environment() -> Dict[str, str]:
    """The environment every solver child runs with."""
    env = dict(os.environ)
    if not FROZEN:
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["OPENEMS_DIR"] = str(OPENEMS_DIR)
    env["OPENEMS_INSTALL_PATH"] = str(OPENEMS_DIR)  # read by CSXCAD to find its DLLs
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["MPLBACKEND"] = "Agg"  # the Qt backend crashes headless workers
    return env


def working_directory() -> Path:
    """Where a modeler worker starts. gerber2ems overrides this per project."""
    return REPO_ROOT


def dispatch(argv: List[str]) -> int:
    """Run the worker named by argv[0] with the rest as its arguments."""
    if not argv or argv[0] not in WORKERS:
        names = ", ".join(sorted(WORKERS))
        print(f"usage: gems-worker <{names}> [arguments...]", file=sys.stderr)
        return 2
    name, rest = argv[0], argv[1:]
    import EMShrimulator.bootstrap  # noqa: F401  (openEMS DLL directory)

    module = importlib.import_module(WORKERS[name])
    sys.argv = [WORKERS[name], *rest]
    result = module.main()
    return int(result or 0)


def main() -> int:
    import multiprocessing  # gerber2ems uses a Pool; required when frozen

    multiprocessing.freeze_support()
    return dispatch(sys.argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - make crashes readable in the job log
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
