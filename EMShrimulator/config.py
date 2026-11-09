"""Paths and constants for both a source checkout and a frozen (PyInstaller) bundle."""
from __future__ import annotations

import os
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

# Bundled openEMS under _internal ("openEMS" would clash with the Python package).
BUNDLED_OPENEMS_DIRNAME = "solver"

_OPENEMS_CANDIDATES = [
    Path(r"C:\OpenEMS\openEMS"),
    Path(r"D:\OpenEMS\openEMS"),
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "openEMS",
]


def _find_openems() -> Path:
    """OPENEMS_DIR, then the bundled copy, then common install locations."""
    explicit = os.environ.get("OPENEMS_DIR")
    if explicit:
        return Path(explicit)
    bundled: List[Path] = [BUNDLE_DIR / BUNDLED_OPENEMS_DIRNAME, INSTALL_DIR / BUNDLED_OPENEMS_DIRNAME] if FROZEN else []
    for candidate in bundled + _OPENEMS_CANDIDATES:
        if (candidate / "openEMS.exe").is_file():
            return candidate
    return _OPENEMS_CANDIDATES[0]


OPENEMS_DIR = _find_openems()
OPENEMS_EXE = OPENEMS_DIR / "openEMS.exe"


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
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "EMShrimulator" / "cache"
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
