"""A radiation pattern in 3D: spherical, rectangular, unrolled round a cylinder, or overlaid on the design.

Plain VTK, with the same render settings as the 3D view (FXAA, no MSAA).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import vtkmodules.vtkInteractionStyle  # noqa: F401 - registers the interactor styles
import vtkmodules.vtkRenderingFreeType  # noqa: F401 - text
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - the OpenGL render window behind the factory
from PySide6.QtWidgets import QHBoxLayout, QMenu, QToolButton, QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkLookupTable
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkActor, vtkLightKit, vtkPolyDataMapper, vtkRenderer, vtkTextActor

from EMShrimulator.app import theme
from EMShrimulator.app.views.results.chart import colormap_lut
from EMShrimulator.app.views.viewport3d import MULTI_SAMPLES, VIEWS, _actor, _lines, _polydata, _polygon, _rgb

DEFAULT_VIEW = "Trimetric"
OVERLAY_OPACITY = 0.6
# A system font with θ and φ; VTK's built-in one has no Greek.
FONT_FILES = (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf",
              Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
              # Fedora
              Path("/usr/share/fonts/google-noto/NotoSans-Regular.ttf"),
              Path("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"),
              Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"),
              Path("/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"),
              Path("/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"),
              # Debian/Ubuntu, macOS
              Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/Library/Fonts/Arial Unicode.ttf"))
VTK_FONT_FILE = 4  # vtkTextProperty's font family that reads a font file


@lru_cache(maxsize=1)
def _font_file() -> Optional[str]:
    found = next((str(p) for p in FONT_FILES if p.exists()), None)
    if found or not shutil.which("fc-match"):
        return found
    # Ask fontconfig for any sans font that has θ (U+03B8) and φ (U+03C6).
    try:
        out = subprocess.run(["fc-match", "-f", "%{file}", "sans-serif:charset=3b8 3c6"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.lower().endswith((".ttf", ".otf")) and Path(out).is_file() else None


def surface(theta_deg: np.ndarray, phi_deg: np.ndarray, rn: np.ndarray, shape: str) -> np.ndarray:
    """Points (θ, φ, 3) of a pattern whose normalised radius is `rn` (0…1)."""
    theta = np.radians(np.asarray(theta_deg, dtype=float))[:, None]
    phi = np.radians(np.asarray(phi_deg, dtype=float))[None, :]
    if shape == "rect3d":  # θ along x, φ along y, the pattern up z
        x = np.broadcast_to(theta / np.pi * 2 - 1, rn.shape)
        y = np.broadcast_to(phi / np.pi - 1, rn.shape)
        return np.stack([x, y, rn], axis=-1)
    if shape == "cylindrical":
        z = np.broadcast_to(1.0 - 2.0 * theta / np.pi, rn.shape)
        return np.stack([rn * np.cos(phi), rn * np.sin(phi), z], axis=-1)
    return np.stack([rn * np.sin(theta) * np.cos(phi), rn * np.sin(theta) * np.sin(phi),
                     rn * np.cos(theta) * np.ones_like(phi)], axis=-1)


def grid_triangles(rows: int, cols: int) -> np.ndarray:
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (i * cols + j).ravel()
    b, c, d = a + cols, a + cols + 1, a + 1
    return np.concatenate([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)])


class Pattern3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._closed = False
        self._vtk = QVTKRenderWindowInteractor(self)
        self._window = self._vtk.GetRenderWindow()
        self._window.SetMultiSamples(MULTI_SAMPLES)  # must precede the window's first render
        self._renderer = vtkRenderer()
        self._renderer.UseFXAAOn()
        self._window.AddRenderer(self._renderer)
        vtkLightKit().AddLightsToRenderer(self._renderer)
        self._renderer.LightFollowCameraOn()
        self._iren = self._window.GetInteractor()
        self._iren.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self._vtk.Initialize()
        self._iren.RemoveObservers("CharEvent")  # VTK's own keys quit the interactor or restyle every actor
        self._iren.AddObserver("KeyPressEvent", self._on_key)

        self._axes = vtkAxesActor()
        self._axes.SetNormalizedShaftLength(0.8, 0.8, 0.8)
        self._axes.SetNormalizedTipLength(0.2, 0.2, 0.2)
        self._marker = vtkOrientationMarkerWidget()
        self._marker.SetOrientationMarker(self._axes)
        self._marker.SetInteractor(self._iren)
        self._marker.SetViewport(0.0, 0.0, 0.18, 0.18)
        self._marker.SetCurrentRenderer(self._renderer)
        self._marker.SetEnabled(1)
        self._marker.InteractiveOff()

        self._lut = vtkLookupTable()
        lut = colormap_lut()
        self._lut.SetNumberOfTableValues(256)
        for i, word in enumerate(lut.tolist()):
            self._lut.SetTableValue(i, ((word >> 16) & 255) / 255, ((word >> 8) & 255) / 255, (word & 255) / 255, 1.0)
        self._lut.Build()
        self._bar = vtkScalarBarActor()
        self._bar.SetLookupTable(self._lut)
        self._bar.SetNumberOfLabels(6)
        self._bar.SetPosition(0.88, 0.1)
        self._bar.SetWidth(0.08)
        self._bar.SetHeight(0.8)
        self._bar.UnconstrainedFontSizeOn()
        self._renderer.AddViewProp(self._bar)
        self._title = vtkTextActor()
        self._title.SetPosition(12, 10)
        self._renderer.AddViewProp(self._title)
        self._actors = []
        self._framing = None

        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.setSpacing(2)
        bar.addStretch(1)
        menu = QMenu("View", self)
        for name in VIEWS:
            menu.addAction(name, lambda name=name: self.view(name))
        view = QToolButton()
        view.setText("View")
        view.setPopupMode(QToolButton.InstantPopup)
        view.setMenu(menu)
        bar.addWidget(view)
        fit = QToolButton()
        fit.setText("Fit")
        fit.clicked.connect(self.fit)
        bar.addWidget(fit)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(bar)
        layout.addWidget(self._vtk, 1)
        theme.manager().changed.connect(self._theme_changed)
        self.apply_theme(render=False)

    # ---------------------------------------------------------------- API

    def show_pattern(self, theta_deg, phi_deg, grid: np.ndarray, *, floor: float, peak: float, label: str,
                     shape: str = "spherical", scene=None, centre_mm=None) -> None:
        """Draw a (θ, φ) pattern from `floor` (centre) to `peak` (radius 1)."""
        c = theme.colors()
        for actor in self._actors:
            self._renderer.RemoveActor(actor)
        self._actors = []
        grid = np.asarray(grid, dtype=float)
        span = max(peak - floor, 1e-12)
        rn = np.clip((np.nan_to_num(grid, nan=floor) - floor) / span, 0.0, 1.0)
        centre = np.zeros(3)
        scale = 1.0
        overlay = shape == "overlay" and scene is not None and scene.bounds is not None
        if overlay:
            lo, hi = (np.asarray(b, dtype=float) for b in scene.bounds)
            centre = np.asarray(centre_mm, dtype=float) if centre_mm is not None else (lo + hi) / 2
            scale = max(float(np.linalg.norm(hi - lo)) / 2 * 1.1, 1e-3)
            for solid in scene.solids.values():
                if solid.model and len(solid.triangles):
                    self._add(_actor(_polydata(solid.vertices, solid.triangles), solid.color or c["accent"],
                                     opacity=solid.opacity, specular=True))
            for sheet in scene.sheets.values():
                if sheet.model and len(sheet.loops) == 1 and len(sheet.loops[0]) >= 3:
                    self._add(_actor(_polygon(sheet.loops[0]), sheet.color or c["accent"],
                                     opacity=max(sheet.opacity, 0.3)))
        points = surface(theta_deg, phi_deg, rn, shape if shape in ("cylindrical", "rect3d") else "spherical")
        points = points.reshape(-1, 3) * scale + centre
        poly = _polydata(points, grid_triangles(len(theta_deg), len(phi_deg)))
        poly.GetPointData().SetScalars(numpy_to_vtk(np.clip(np.nan_to_num(grid, nan=floor), floor, peak).ravel(),
                                                    deep=True))
        normals = vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.SplittingOff()
        normals.ConsistencyOn()
        normals.Update()
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.SetLookupTable(self._lut)
        mapper.SetScalarRange(floor, peak)
        mapper.ScalarVisibilityOn()
        pattern = vtkActor()
        pattern.SetMapper(mapper)
        prop = pattern.GetProperty()
        prop.SetInterpolationToGouraud()
        prop.SetSpecular(0.2)
        prop.SetSpecularPower(20)
        prop.SetOpacity(OVERLAY_OPACITY if overlay else 1.0)
        self._add(pattern)
        if shape == "rect3d":
            # The floor's frame: θ along x, φ along y.
            for key, a, b in (("axis_x", (-1, -1, 0), (1, -1, 0)), ("axis_y", (-1, -1, 0), (-1, 1, 0)),
                              ("axis_z", (-1, -1, 0), (-1, -1, 1)), ("line", (1, -1, 0), (1, 1, 0)),
                              ("line", (-1, 1, 0), (1, 1, 0))):
                frame = _lines(np.array([[a, b]], dtype=float))
                if frame is not None:
                    self._add(_actor(frame, c[key], line_width=2 if key != "line" else 1))
        else:
            # Axes through the pattern's centre.
            reach = 1.25 * scale
            for i, key in enumerate(("axis_x", "axis_y", "axis_z")):
                tip = np.zeros(3)
                tip[i] = reach
                # A cylinder's axis runs through the whole of it, θ = 0° to 180°.
                start = centre - tip if shape == "cylindrical" and i == 2 else centre
                poly_axis = _lines(np.array([[start, centre + tip]]))
                if poly_axis is not None:
                    self._add(_actor(poly_axis, c[key], line_width=2))
        self._lut.SetTableRange(floor, peak)
        self._bar.SetTitle(label)
        self._title.SetInput({"cylindrical": "φ round the axis · θ = 0° at the top, 180° at the bottom",
                              "rect3d": "θ 0°–180° along x (red) · φ 0°–360° along y (green) · the pattern up z",
                              "overlay": "Pattern over the design, centred on the phase centre"}.get(shape, ""))
        # Reframe for a new kind or size; a new frequency keeps the view.
        framing = (shape, round(scale, 6), tuple(np.round(centre, 6)))
        if framing != self._framing:
            self._framing = framing
            self.view(DEFAULT_VIEW, render=False)
        else:
            self._renderer.ResetCameraClippingRange()
        self.redraw()

    def view(self, name: str, render: bool = True) -> None:
        direction, up = VIEWS[name]
        bounds = np.array(self._renderer.ComputeVisiblePropBounds(), dtype=float)
        centre = (bounds[0::2] + bounds[1::2]) / 2 if np.isfinite(bounds).all() and bounds[1] >= bounds[0] \
            else np.zeros(3)
        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(*centre.tolist())
        camera.SetPosition(*(centre + np.array(direction, dtype=float) / np.linalg.norm(direction) * 10).tolist())
        camera.SetViewUp(*up)
        self._renderer.ResetCamera()
        if render:
            self.redraw()

    def fit(self) -> None:
        self._renderer.ResetCamera()
        self.redraw()

    def redraw(self) -> None:
        if not self._closed and self.isVisible():
            self._window.Render()

    def close_view(self) -> None:
        self._closed = True
        self._vtk.hide()
        try:
            self._marker.SetEnabled(0)
            self._vtk.Finalize()
        except Exception:  # noqa: BLE001 - shutting down
            pass

    def apply_theme(self, render: bool = True) -> None:
        c = theme.colors()
        self._renderer.SetBackground(*_rgb(c["vp_ground"]))
        self._renderer.SetBackground2(*_rgb(c["vp_sky"]))
        self._renderer.GradientBackgroundOn()
        for shaft, tip, key in ((self._axes.GetXAxisShaftProperty(), self._axes.GetXAxisTipProperty(), "axis_x"),
                                (self._axes.GetYAxisShaftProperty(), self._axes.GetYAxisTipProperty(), "axis_y"),
                                (self._axes.GetZAxisShaftProperty(), self._axes.GetZAxisTipProperty(), "axis_z")):
            shaft.SetColor(*_rgb(c[key]))
            tip.SetColor(*_rgb(c[key]))
        for caption in (self._axes.GetXAxisCaptionActor2D(), self._axes.GetYAxisCaptionActor2D(),
                        self._axes.GetZAxisCaptionActor2D()):
            caption.GetCaptionTextProperty().SetColor(*_rgb(c["text_dim"]))
            caption.GetCaptionTextProperty().ShadowOff()
            caption.GetCaptionTextProperty().ItalicOff()
        font = _font_file()
        for text in (self._bar.GetTitleTextProperty(), self._bar.GetLabelTextProperty(),
                     self._title.GetTextProperty()):
            text.SetColor(*_rgb(c["text"]))
            text.ShadowOff()
            text.ItalicOff()
            text.BoldOff()
            text.SetFontSize(12)
            if font:
                text.SetFontFamily(VTK_FONT_FILE)
                text.SetFontFile(font)
        if render:
            self.redraw()

    # ------------------------------------------------------------ internals

    def _add(self, actor) -> None:
        actor.PickableOff()
        self._renderer.AddActor(actor)
        self._actors.append(actor)

    def _theme_changed(self, _name: str) -> None:
        if not self._closed:
            self.apply_theme()

    def _on_key(self, *_args) -> None:
        key = self._iren.GetKeySym()
        if key in ("v", "V"):
            self.view(DEFAULT_VIEW)
        elif key in ("r", "R", "f", "F"):
            self.fit()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.redraw()
