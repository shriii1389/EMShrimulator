# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller: one folder with EMShrimulator (GUI) and gems-worker (console solver runs); .exe on Windows."""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected by PyInstaller
sys.path.insert(0, str(ROOT))

# Registers the openEMS DLL directory (Windows) / preloads its .so files (Linux) for analysis.
import EMShrimulator.bootstrap  # noqa: E402,F401

BINDINGS = [
    "CSXCAD.CSXCAD", "CSXCAD.CSPrimitives", "CSXCAD.CSProperties", "CSXCAD.CSRectGrid",
    "CSXCAD.CSTransform", "CSXCAD.ParameterObjects", "CSXCAD.Utilities", "CSXCAD.SmoothMeshLines",
    "openEMS.openEMS", "openEMS._nf2ff", "openEMS.nf2ff", "openEMS.ports", "openEMS.automesh",
    "openEMS.physical_constants", "openEMS.utilities",
]

ENGINE_IMPORTS = (
    BINDINGS
    + collect_submodules("EMShrimulator.core")
    + collect_submodules("gerber2ems")
    + collect_submodules("CSXCAD")
    + collect_submodules("openEMS")
    + collect_submodules("nanomesh")
    + collect_submodules("skrf")
    + ["EMShrimulator.config", "EMShrimulator.bootstrap", "manifold3d", "resvg_py", "gerbonara",
       "nanomesh", "h5py", "serde", "serde.json", "coloredlogs", "scipy.optimize", "PIL.Image",
       "matplotlib.backends.backend_agg"]
)

DATAS = [
    (str(ROOT / "EMShrimulator" / "resources"), "EMShrimulator/resources"),
    (str(ROOT / "vendor" / "gerber2ems-main" / "examples"), "vendor/gerber2ems-main/examples"),
] + collect_data_files("gerber2ems") + collect_data_files("skrf") + collect_data_files("gerbonara")

# Keep IPython and ipywidgets: nanomesh imports them.
EXCLUDES = [
    "tkinter", "ipympl", "itk", "itkwidgets", "notebook",
    "PyQt5", "PyQt6", "pytest", "pytest_qt",
]

app_analysis = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launch_app.py")],
    pathex=[str(ROOT)],
    # The viewport is plain VTK on its own Qt widget and the charts are QPainter: neither pyvistaqt
    # nor matplotlib's Qt backend is used by the app any more.
    hiddenimports=ENGINE_IMPORTS + collect_submodules("EMShrimulator.app")
    + ["vtkmodules.qt.QVTKRenderWindowInteractor", "vtkmodules.vtkRenderingOpenGL2",
       "vtkmodules.vtkRenderingFreeType", "vtkmodules.vtkInteractionStyle"],
    datas=DATAS,
    excludes=EXCLUDES + ["pyvistaqt", "matplotlib.backends.backend_qtagg"],
    noarchive=False,
)

worker_analysis = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "launch_worker.py")],
    pathex=[str(ROOT)],
    hiddenimports=ENGINE_IMPORTS,
    datas=DATAS,
    # pyvista/vtk stay: nanomesh needs them.
    excludes=EXCLUDES + ["PySide6", "shiboken6", "pyvistaqt", "EMShrimulator.app"],
    noarchive=False,
)

app_pyz = PYZ(app_analysis.pure)  # noqa: F821
worker_pyz = PYZ(worker_analysis.pure)  # noqa: F821

app_exe = EXE(  # noqa: F821
    app_pyz,
    app_analysis.scripts,
    [],
    exclude_binaries=True,
    name="EMShrimulator",
    # from tools/make_icon.py; Linux takes its icon from the .desktop file instead
    icon=str(ROOT / "packaging" / "EMShrimulator.ico") if sys.platform == "win32" else None,
    console=False,
    disable_windowed_traceback=False,
    upx=False,
)

worker_exe = EXE(  # noqa: F821
    worker_pyz,
    worker_analysis.scripts,
    [],
    exclude_binaries=True,
    name="gems-worker",
    console=True,
    upx=False,
)

COLLECT(  # noqa: F821
    app_exe,
    worker_exe,
    app_analysis.binaries,
    app_analysis.datas,
    worker_analysis.binaries,
    worker_analysis.datas,
    upx=False,
    name="EMShrimulator",
)
