"""Install diagnostics (`python -m EMShrimulator.core.diagnostics`). No GUI imports."""
from __future__ import annotations

import os
import platform
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List

from EMShrimulator.config import (
    APP_VERSION,
    FROZEN,
    GERBER2EMS_DIR,
    IS_WINDOWS,
    OPENEMS_BINARY,
    OPENEMS_DIR,
    OPENEMS_EXE,
    openems_library_dirs,
    projects_dir,
)

SETUP_HINT = r"Re-run .\gems.ps1 setup" if IS_WINDOWS else "Re-run ./gems.sh setup"
from EMShrimulator.core import jobs, workers


def check(name: str, ok: bool, detail: str, fix: str = "") -> Dict:
    return {"name": name, "ok": ok, "detail": detail, "fix": fix}


def module_check(module: str, label: str, fix: str) -> Dict:
    try:
        loaded = import_module(module)
        version = getattr(loaded, "__version__", "") or getattr(loaded, "version", "")
        location = getattr(loaded, "__file__", "") or ""
        return check(label, True, f"{version} {location}".strip())
    except Exception as exc:  # noqa: BLE001
        return check(label, False, str(exc), fix)


def diagnostics() -> Dict:
    checks: List[Dict] = []

    checks.append(check(
        "openEMS install directory", OPENEMS_DIR.is_dir(), str(OPENEMS_DIR),
        "Set the OPENEMS_DIR environment variable to your openEMS install, or reinstall openEMS. "
        + ("The compiled CSXCAD/openEMS wheels load their DLLs from this directory." if IS_WINDOWS else
           "./gems.sh setup builds openEMS into ~/opt/openEMS (override with OPENEMS_DIR)."),
    ))
    checks.append(check(
        OPENEMS_BINARY, OPENEMS_EXE.is_file(), str(OPENEMS_EXE),
        "The native solver binary is missing from the openEMS directory."
        + ("" if IS_WINDOWS else " Run ./gems.sh setup to build it."),
    ))

    launcher = Path(workers.command("run")[0])
    checks.append(check(
        "run launcher", launcher.is_file(), str(launcher),
        "Runs are started as separate processes through this executable. "
        + ("Reinstall the application." if FROZEN else f"{SETUP_HINT} to rebuild the venv."),
    ))

    # Needs EMShrimulator.bootstrap first.
    try:
        import EMShrimulator.bootstrap  # noqa: F401
        if IS_WINDOWS:
            dll_ok, dll_detail = True, f"os.add_dll_directory({OPENEMS_DIR})"
        else:
            libs = openems_library_dirs(OPENEMS_DIR)
            dll_ok = any(any(d.glob("libCSXCAD.so*")) for d in libs)
            dll_detail = ", ".join(str(d) for d in libs) or f"no lib/ or lib64/ under {OPENEMS_DIR}"
    except Exception as exc:  # noqa: BLE001
        dll_ok, dll_detail = False, str(exc)
    checks.append(check("shared library path" if not IS_WINDOWS else "DLL search path", dll_ok, dll_detail,
                        "Python 3.8+ on Windows ignores PATH for extension-module DLLs; "
                        "EMShrimulator/bootstrap.py registers the openEMS directory instead." if IS_WINDOWS else
                        "libCSXCAD.so was not found under the openEMS directory. Run ./gems.sh setup."))

    checks.append(module_check("CSXCAD", "CSXCAD bindings",
                               "pip install the csxcad wheel matching this Python's ABI (cp313)." if IS_WINDOWS
                               else f"{SETUP_HINT} (builds the CSXCAD bindings into the venv)."))
    checks.append(module_check("openEMS", "openEMS bindings",
                               "pip install the openems wheel matching this Python's ABI (cp313)." if IS_WINDOWS
                               else f"{SETUP_HINT} (builds the openEMS bindings into the venv)."))
    checks.append(module_check("manifold3d", "CSG kernel (manifold3d)",
                               "pip install manifold3d - needed for union/intersect/subtract objects."))
    checks.append(module_check("gerbonara", "Gerber parser (gerbonara)", "pip install gerbonara"))
    checks.append(module_check("resvg_py", "Rasterizer (resvg-py)",
                               "pip install resvg-py - replaces the gerbv dependency."))
    checks.append(module_check("skrf", "scikit-rf (Touchstone)", "pip install scikit-rf"))
    checks.append(module_check("nanomesh", "nanomesh (copper meshing)", "pip install nanomesh"))
    checks.append(module_check("h5py", "h5py (far-field data)", "pip install h5py"))

    try:
        import gerber2ems  # noqa: F401
        g2e_ok, g2e_detail = True, str(GERBER2EMS_DIR)
    except Exception as exc:  # noqa: BLE001
        g2e_ok, g2e_detail = False, str(exc)
    checks.append(check("gerber2ems (vendored)", g2e_ok, g2e_detail,
                        "pip install -e vendor/gerber2ems-main --no-deps "
                        "(with SETUPTOOLS_SCM_PRETEND_VERSION set)"))

    try:
        folder = projects_dir()
        checks.append(check("projects directory", folder.is_dir(), str(folder)))
    except OSError as exc:
        checks.append(check("projects directory", False, str(exc),
                            "Choose a writable projects folder in the application settings."))

    return {
        "version": APP_VERSION,
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "frozen": FROZEN,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "active_jobs": jobs.active_count(),
        },
    }


def print_report(report: Dict) -> None:
    for item in report["checks"]:
        print(("  OK  " if item["ok"] else "  XX  ") + item["name"] + ": " + item["detail"][:110])
        if not item["ok"] and item["fix"]:
            print("        fix: " + item["fix"])


def main() -> int:
    report = diagnostics()
    print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
