# Third-party notices

EMShrimulator Studio is licensed under GPL-3.0-or-later (`LICENSE`). It is built
on the projects below, which keep their own licenses. In the Windows build
(`EMShrimulator-<version>-win64.zip`) every text named here is in the
`licenses/` folder next to the program; in a source checkout they come with the
packages that `gems.ps1 setup` installs.

This file is a guide to the main components, not a replacement for their
license files.

## openEMS solver (`solver/` in the Windows build)

The Windows build bundles an unmodified openEMS Windows installation. Its
source is at <https://github.com/thliebig/openEMS-Project>.

| Component | License | Text |
| --- | --- | --- |
| openEMS, nf2ff | GPL-3.0-or-later | `licenses/GPL-3.0.txt` |
| CSXCAD, QCSXCAD | LGPL-3.0-or-later | `licenses/LGPL-3.0.txt` with `licenses/GPL-3.0.txt` |
| fparser | LGPL-3.0 | `licenses/LGPL-3.0.txt` with `licenses/GPL-3.0.txt` |
| Qt 5 | LGPL-3.0 | `licenses/LGPL-3.0.txt` with `licenses/GPL-3.0.txt` |
| VTK 8.2 and the libraries it bundles | BSD-3-Clause and others | `licenses/python/vtk/` (VTK's license) |
| Boost 1.72 | BSL-1.0 | — |
| Microsoft Visual C++ runtime | Microsoft redistributable terms | — |

The CSXCAD and openEMS Python bindings come from the same installation and are
under the same licenses.

## gerber2ems

`vendor/gerber2ems-main/` is a patched copy of Antmicro's gerber2ems, licensed
under Apache License 2.0 (`vendor/gerber2ems-main/LICENSE`, and
`licenses/Apache-2.0.txt` in the Windows build). Every changed or added part is
marked `PATCHED (EMShrimulator Studio)` or `ADDED (EMShrimulator Studio)` in the
source.

## Python and its packages

The Windows build includes CPython (`licenses/python/CPython/`) and the Python
packages listed in `requirements.txt` with their dependencies. Each package's
own license files are in `licenses/python/<package>/`. The notable licenses:

- Qt for Python (PySide6, shiboken6): LGPL-3.0, text in `licenses/LGPL-3.0.txt`
  and `licenses/GPL-3.0.txt`. Its source is at <https://code.qt.io/cgit/pyside/pyside-setup.git/>.
- gerbonara: Apache-2.0, `licenses/Apache-2.0.txt`.
- scikit-rf: BSD-3-Clause, `licenses/python/scikit-rf/`.
- NumPy, SciPy, Matplotlib, VTK, PyVista, h5py, Pillow, pydantic, manifold3d,
  nanomesh, PyInstaller's bootloader and the rest: in `licenses/python/`.

## Source code

The source of EMShrimulator Studio, including the patched gerber2ems, is at
<https://github.com/shriii1389/EMShrimulator>. Each release is tagged with its
version, so the source of any Windows build is the tag of the same version.
