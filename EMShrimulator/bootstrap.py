"""Import before CSXCAD/openEMS so their compiled bindings find the openEMS shared libraries.

Windows: Python 3.8+ ignores PATH for extension-module DLLs, so the openEMS directory is registered
with os.add_dll_directory.

Linux: the dynamic loader reads LD_LIBRARY_PATH only at process start, so setting it here would be
too late. Instead the openEMS libraries are preloaded by full path (RTLD_LOCAL, so they cannot clash
with the pip VTK the viewport uses); the bindings then resolve them by soname. gems.sh and the solver
workers also set LD_LIBRARY_PATH, and `gems.sh setup` builds openEMS with an RPATH, so this is only
the fallback for e.g. `python -m EMShrimulator` started by hand.
"""
from __future__ import annotations

import ctypes
import os
import sys

from EMShrimulator.config import OPENEMS_DIR, openems_library_dirs

_added = False

# Load order matters: each library needs the ones before it.
_LINUX_LIBRARIES = ("libfparser.so", "libCSXCAD.so", "libnf2ff.so", "libopenEMS.so")


def _preload_linux() -> None:
    for name in _LINUX_LIBRARIES:
        for folder in openems_library_dirs(OPENEMS_DIR):
            candidates = sorted(folder.glob(name + "*"), key=lambda p: len(p.name))
            if not candidates:
                continue
            try:
                ctypes.CDLL(str(candidates[0]))
            except OSError:
                pass  # the RPATH or LD_LIBRARY_PATH may still find it
            break


def ensure_dll_path() -> None:
    global _added
    if _added:
        return
    if sys.platform == "win32":
        if OPENEMS_DIR.is_dir():
            os.add_dll_directory(str(OPENEMS_DIR))
    else:
        _preload_linux()
    _added = True


ensure_dll_path()
