"""Paths and constants for both a source checkout and a frozen (PyInstaller) bundle."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

APP_NAME = "EMShrimulator Studio"
APP_VERSION = "1.0.0"
ORGANISATION = "EMShrimulator"

FROZEN = bool(getattr(sys, "frozen", False))

if FROZEN:
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    INSTALL_DIR = Path(sys.executable).resolve().parent
    REPO_ROOT = INSTALL_DIR
else:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    BUNDLE_DIR = REPO_ROOT
    INSTALL_DIR = REPO_ROOT

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = PACKAGE_DIR / "resources"
APP_ICON = RESOURCES_DIR / "icons" / "shrimp.svg"


# --- openEMS ---

IS_WINDOWS = sys.platform == "win32"
OPENEMS_BINARY = "openEMS.exe" if IS_WINDOWS else "openEMS"

# Bundled openEMS under _internal ("openEMS" would clash with the Python package).
BUNDLED_OPENEMS_DIRNAME = "solver"

if IS_WINDOWS:
    _OPENEMS_CANDIDATES = [
        Path(r"C:\OpenEMS\openEMS"),
        Path(r"D:\OpenEMS\openEMS"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "openEMS",
    ]
else:
    # ~/opt/openEMS is where `./gems.sh setup` (and upstream's update_openEMS.sh) install it.
    _OPENEMS_CANDIDATES = [
        Path.home() / "opt" / "openEMS",
        Path("/opt/openEMS"),
        Path("/usr/local"),
        Path("/usr"),
    ]


def openems_executable(root: Path) -> Path:
    """The solver binary under an install root: flat on Windows, root/bin on Linux."""
    if IS_WINDOWS:
        return root / OPENEMS_BINARY
    for candidate in (root / "bin" / OPENEMS_BINARY, root / OPENEMS_BINARY):
        if candidate.is_file():
            return candidate
    return root / "bin" / OPENEMS_BINARY


def openems_library_dirs(root: Path) -> List[Path]:
    """Where the openEMS/CSXCAD/fparser shared libraries live (DLLs sit next to the exe on Windows)."""
    if IS_WINDOWS:
        return [root]
    return [d for d in (root / "lib64", root / "lib") if d.is_dir()]


def _find_openems() -> Path:
    """OPENEMS_DIR, then the bundled copy, then common install locations, then PATH."""
    explicit = os.environ.get("OPENEMS_DIR")
    if explicit:
        return Path(explicit)
    bundled: List[Path] = [BUNDLE_DIR / BUNDLED_OPENEMS_DIRNAME, INSTALL_DIR / BUNDLED_OPENEMS_DIRNAME] if FROZEN else []
    for candidate in bundled + _OPENEMS_CANDIDATES:
        if openems_executable(candidate).is_file():
            return candidate
    on_path = shutil.which(OPENEMS_BINARY)
    if on_path:
        binary = Path(on_path).resolve()
        return binary.parent if IS_WINDOWS else binary.parent.parent
    return _OPENEMS_CANDIDATES[0]


OPENEMS_DIR = _find_openems()
OPENEMS_EXE = openems_executable(OPENEMS_DIR)


# --- bundled data ---

VENDOR_DIR = BUNDLE_DIR / "vendor"
GERBER2EMS_DIR = VENDOR_DIR / "gerber2ems-main"
GERBER2EMS_EXAMPLES = GERBER2EMS_DIR / "examples"


# --- user data ---

def _default_projects_dir() -> Path:
    if FROZEN:
        return Path.home() / "Documents" / "EMShrimulator Projects"
    return REPO_ROOT / "projects"


def _default_cache_dir() -> Path:
    if FROZEN:
        if IS_WINDOWS:
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
            return Path(base) / "EMShrimulator" / "cache"
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        return Path(base) / "EMShrimulator"
    return REPO_ROOT / ".cache"


_projects_dir: Optional[Path] = None


def projects_dir() -> Path:
    """Projects folder, created on first use (user-changeable, hence a function)."""
    global _projects_dir
    if _projects_dir is None:
        _projects_dir = Path(os.environ.get("GEMS_PROJECTS_DIR") or _default_projects_dir())
    _projects_dir.mkdir(parents=True, exist_ok=True)
    return _projects_dir


def set_projects_dir(path: Path | str) -> Path:
    global _projects_dir
    _projects_dir = Path(path)
    _projects_dir.mkdir(parents=True, exist_ok=True)
    return _projects_dir


TEMP_DIR = Path(os.environ.get("GEMS_TEMP_DIR") or _default_cache_dir())


LENGTH_UNIT_M = 1e-3  # 1 drawing unit = 1 mm
