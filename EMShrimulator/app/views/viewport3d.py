"""3D viewport on plain VTK. Actors are keyed by object id and only rebuilt when their geometry or colour changes."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtkmodules.vtkInteractionStyle  # noqa: F401 - registers the interactor styles
import vtkmodules.vtkRenderingFreeType  # noqa: F401 - text: port labels and axis captions
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401 - the OpenGL render window behind the factory
from PySide6.QtCore import QPoint, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCursor, QKeySequence
from PySide6.QtWidgets import QHBoxLayout, QMenu, QToolButton, QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import get_numpy_array_type, numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.util.vtkConstants import VTK_ID_TYPE, VTK_UNSIGNED_CHAR
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkFeatureEdges
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkActor2D,
    vtkCellPicker,
    vtkCoordinate,
    vtkLightKit,
    vtkPolyDataMapper,
    vtkPolyDataMapper2D,
    vtkRenderer,
    vtkTextActor,
)

from EMShrimulator.app import theme
from EMShrimulator.app.widgets import icons
from EMShrimulator.app.widgets.form import ChoiceBox
from EMShrimulator.core.em import snaps
from EMShrimulator.core.em.drawing import (
    PLANE_LABELS,
    PLANES,
    MeasureTool,
    Plane,
    Step,
    Tool,
    Units,
    closest_on_line,
    keep_to_axis,
)

SHADING = [("solid", "Solid"), ("xray", "X-ray"), ("wireframe", "Wire")]
# Standard views: (eye direction, view-up). Front looks along +y.
VIEWS = {
    "Top": ((0, 0, 1), (0, 1, 0)),
    "Bottom": ((0, 0, -1), (0, -1, 0)),
    "Front": ((0, -1, 0), (0, 0, 1)),
    "Back": ((0, 1, 0), (0, 0, 1)),
    "Left": ((-1, 0, 0), (0, 0, 1)),
    "Right": ((1, 0, 0), (0, 0, 1)),
    "Isometric": ((1, -1, 1), (0, 0, 1)),
    "Dimetric": ((1, -7 ** 0.5, 1), (0, 0, 1)),
    "Trimetric": ((1, -1.3, 0.9), (0, 0, 1)),
}
PICTORIAL = ("Isometric", "Dimetric", "Trimetric")
DEFAULT_VIEW = "Trimetric"  # the first view of a design, and V
CLICK_TOLERANCE_PX = 4
SNAP_RADIUS_PX = 22  # how near the cursor a snap target must be to take it
# Snap toggles: (kind, menu label, icon).
SNAP_TOGGLES = ((snaps.FACE, "Face centres", "snap_face"), (snaps.EDGE, "Edge centres", "snap_edge"),
                (snaps.CORNER, "Corners", "snap_corner"), (snaps.GRID, "Grid", "snap_grid"))
# Snap mark shapes: face centre ○, edge centre △, corner □, grid +, object centre ◇.
MARK_SHAPE = {snaps.FACE: "circle", snaps.ARC: "circle", snaps.EDGE: "triangle", snaps.QUADRANT: "triangle",
              snaps.CORNER: "square", snaps.GRID: "plus", snaps.CENTRE: "diamond"}
# Snap kinds in priority order; GRID is the fallback.
OBJECT_KINDS = (snaps.FACE, snaps.ARC, snaps.EDGE, snaps.QUADRANT, snaps.CORNER)
# Coordinate system triads on screen: (axis length, shaft width) in pixels.
TRIAD_SIZE = {"working": (30.0, 2.0), "selected": (30.0, 2.0), "global": (13.0, 1.5)}
FINE_STEP = 1e-3  # fraction of the grid step free points are rounded to
# No MSAA: it hangs AMD's integrated-graphics driver. FXAA instead.
MULTI_SAMPLES = 0


@dataclass
class _Entry:
    kind: str                 # "primitive" | "port"
    key: str
    surface: vtkActor
    edges: Optional[vtkActor]
    geometry: object  # identity of the source vertex array; None for sheets and ports
    signature: object         # sheets and ports: what their actors were built from
    color: str
    opacity: float
    label: Optional[vtkTextActor] = None
    loops: List[vtkActor] = field(default_factory=list)


# ------------------------------------------------------------------ polydata

# numpy dtype of vtkIdType (int64 on 64-bit builds). Looked up rather than imported as
# numpy_support.ID_TYPE_CODE, which VTK 9.7 removed.
ID_DTYPE = np.dtype(get_numpy_array_type(VTK_ID_TYPE))


def _ids(values: np.ndarray):
    return numpy_to_vtkIdTypeArray(np.ascontiguousarray(values, dtype=ID_DTYPE), deep=True)


def _points(vertices: np.ndarray) -> vtkPoints:
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64).reshape(-1, 3), deep=True))
    return points


def _cells(connectivity: np.ndarray, size: int) -> vtkCellArray:
    """Cells of one fixed size from a flat connectivity array."""
    cells = vtkCellArray()
    count = len(connectivity) // size
    cells.SetData(_ids(np.arange(0, count * size + 1, size)), _ids(connectivity))
    return cells


def _polydata(vertices: np.ndarray, triangles: np.ndarray) -> vtkPolyData:
    poly = vtkPolyData()
    poly.SetPoints(_points(vertices))
    poly.SetPolys(_cells(np.asarray(triangles).ravel(), 3))
    return poly


def _lines(segments: np.ndarray) -> Optional[vtkPolyData]:
    """Independent segments, (N, 2, 3)."""
    if len(segments) == 0:
        return None
    poly = vtkPolyData()
    poly.SetPoints(_points(segments))
    poly.SetLines(_cells(np.arange(2 * len(segments)), 2))
    return poly


def _loop(points: np.ndarray) -> vtkPolyData:
    n = len(points)
    poly = vtkPolyData()
    poly.SetPoints(_points(points))
    poly.SetLines(_cells(np.concatenate([np.arange(n), [0]]), n + 1))
    return poly


def _polygon(points: np.ndarray) -> vtkPolyData:
    n = len(points)
    poly = vtkPolyData()
    poly.SetPoints(_points(points))
    poly.SetPolys(_cells(np.arange(n), n))
    return poly


_BOX_EDGES = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]])
# Six quads, outward winding, over the corner numbering bit0=x bit1=y bit2=z.
_BOX_QUADS = np.array([[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]])


def _corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return np.array([[hi[0] if i & 1 else lo[0], hi[1] if i & 2 else lo[1], hi[2] if i & 4 else lo[2]]
                     for i in range(8)])


def _box_surface(lo, hi) -> vtkPolyData:
    poly = vtkPolyData()
    poly.SetPoints(_points(_corners(lo, hi)))
    poly.SetPolys(_cells(_BOX_QUADS.ravel(), 4))
    return poly


def _box_outline(lo, hi) -> vtkPolyData:
    return _lines(_corners(lo, hi)[_BOX_EDGES])


def _arrow_head(tip: np.ndarray, direction: np.ndarray, length: float, radius: float,
                segments: int = 16) -> vtkPolyData:
    """A closed cone with its point at `tip`, pointing along `direction`."""
    d = direction / np.linalg.norm(direction)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    base = tip - d * length
    angles = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    ring = base + radius * (np.outer(np.cos(angles), e1) + np.outer(np.sin(angles), e2))
    k = np.arange(segments)
    side = np.stack([np.zeros(segments, dtype=int), 2 + k, 2 + (k + 1) % segments], axis=1)
    cap = np.stack([np.ones(segments, dtype=int), 2 + (k + 1) % segments, 2 + k], axis=1)
    return _polydata(np.vstack([tip, base, ring]), np.vstack([side, cap]))


def _feature_edges(poly: vtkPolyData) -> Optional[vtkPolyData]:
    edges = vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.SetFeatureAngle(35)
    edges.BoundaryEdgesOn()
    edges.FeatureEdgesOn()
    edges.NonManifoldEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    out = vtkPolyData()
    out.ShallowCopy(edges.GetOutput())
    return out if out.GetNumberOfPoints() else None


def _actor(poly: vtkPolyData, color: str, *, opacity: float = 1.0, line_width: float = 1.0,
           pickable: bool = False, specular: bool = False) -> vtkActor:
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()
    actor = vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*_rgb(color))
    prop.SetOpacity(opacity)
    prop.SetLineWidth(line_width)
    prop.SetInterpolationToFlat()
    if specular:
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(18)
    actor.SetPickable(pickable)
    return actor


def _set_input(actor: vtkActor, poly: vtkPolyData) -> None:
    actor.GetMapper().SetInputData(poly)


def _paths(paths) -> Optional[vtkPolyData]:
    """Polylines, each (K, 3)."""
    paths = [np.asarray(p, dtype=float).reshape(-1, 3) for p in paths if len(p) >= 2]
    if not paths:
        return None
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in paths])])
    cells = vtkCellArray()
    cells.SetData(_ids(offsets), _ids(np.arange(offsets[-1])))
    poly = vtkPolyData()
    poly.SetPoints(_points(np.vstack(paths)))
    poly.SetLines(cells)
    return poly


# ------------------------------------------------------------------ snap marks

def _unit_shape(name: str) -> np.ndarray:
    """A closed outline whose sides are one unit from its centre."""
    if name == "circle":
        angles = np.linspace(0.0, 2 * np.pi, 28, endpoint=False)
    elif name == "triangle":
        angles = np.radians([90.0, 210.0, 330.0])
    elif name == "diamond":
        angles = np.radians([0.0, 90.0, 180.0, 270.0])
    else:
        angles = np.radians([45.0, 135.0, 225.0, 315.0])
    reach = 1.0 / np.cos(np.pi / len(angles))  # vertex distance for a unit apothem
    return reach * np.stack([np.cos(angles), np.sin(angles)], axis=1)


_SHAPES = {name: _unit_shape(name) for name in ("circle", "triangle", "square", "diamond")}
_SIZE_PX = {"circle": 7.0, "triangle": 5.5, "square": 6.5, "diamond": 6.0, "plus": 7.5}  # apothem; plus: half-length


def _band(centre: np.ndarray, shape: str, size: float, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of a shape's outline `width` wide."""
    unit = _SHAPES[shape]
    outer = centre + unit * (size + width / 2)
    inner = centre + unit * max(size - width / 2, 0.0)
    n = len(unit)
    k = np.arange(n)
    nxt = (k + 1) % n
    triangles = np.concatenate([np.stack([k, nxt, n + nxt], axis=1), np.stack([k, n + nxt, n + k], axis=1)])
    return np.vstack([outer, inner]), triangles


def _box2(centre: np.ndarray, half_w: float, half_h: float) -> Tuple[np.ndarray, np.ndarray]:
    x, y = centre
    points = np.array([[x - half_w, y - half_h], [x + half_w, y - half_h], [x + half_w, y + half_h],
                       [x - half_w, y + half_h]])
    return points, np.array([[0, 1, 2], [0, 2, 3]])


def _mark_parts(centre: np.ndarray, shape: str, style: str, scale: float, c: Dict[str, str]):
    """(points, triangles, rgba) pieces of one mark, halo first."""
    rgba = lambda key, alpha=255: (*(int(v * 255) for v in _rgb(c[key])), alpha)  # noqa: E731
    parts = []
    if style == "hint":  # a target on offer, not under the cursor: small and quiet
        if shape in _SHAPES:
            parts.append((*_band(centre, shape, _SIZE_PX[shape] * 0.7 * scale, 1.5 * scale), rgba("accent", 200)))
        return parts
    if style == "fixed" or shape == "dot":  # a point already clicked, or a free one under the cursor
        r = (3.5 if style == "fixed" else 2.5) * scale
        parts.append((*_box2(centre, r + 1.5 * scale, r + 1.5 * scale), rgba("snap_halo", 220)))
        parts.append((*_box2(centre, r, r), rgba("selection" if style == "fixed" else "snap")))
        return parts
    width, halo = 2.0 * scale, 3.5 * scale
    if shape == "plus":
        reach = _SIZE_PX["plus"] * scale
        for w, key, alpha in ((width + halo, "snap_halo", 220), (width, "snap", 255)):
            grow = (w - width) / 2
            parts.append((*_box2(centre, reach + grow, w / 2), rgba(key, alpha)))
            parts.append((*_box2(centre, w / 2, reach + grow), rgba(key, alpha)))
    else:
        size = _SIZE_PX[shape] * scale
        parts.append((*_band(centre, shape, size, width + halo), rgba("snap_halo", 220)))
        parts.append((*_band(centre, shape, size, width), rgba("snap")))
    parts.append((*_box2(centre, 1.5 * scale, 1.5 * scale), rgba("snap")))  # the exact point
    return parts


def _marks_poly(marks, scale: float, c: Dict[str, str]) -> vtkPolyData:
    """One polydata of per-cell coloured triangles for marks given as (screen xy, shape, style)."""
    return _poly2d([part for centre, shape, style in marks
                    for part in _mark_parts(np.asarray(centre, dtype=float), shape, style, scale, c)])


def _segment(a: np.ndarray, b: np.ndarray, width: float) -> Tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of a band `width` wide from a to b."""
    along = b - a
    side = np.array([-along[1], along[0]]) / (np.hypot(*along) or 1.0) * width / 2
    return np.array([a - side, b - side, b + side, a + side]), np.array([[0, 1, 2], [0, 2, 3]])


def _head(tip: np.ndarray, along: np.ndarray, length: float, half_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of an arrowhead at `tip` pointing along the unit 2D vector `along`."""
    base = tip - along * length
    side = np.array([-along[1], along[0]]) * half_width
    return np.array([tip, base + side, base - side]), np.array([[0, 1, 2]])


def _triad_parts(origin: np.ndarray, tips: np.ndarray, style: str, scale: float, c: Dict[str, str]):
    """(points, triangles, rgba) pieces of one coordinate system's axes on screen."""
    rgba = lambda key, alpha=255: (*(int(v * 255) for v in _rgb(c[key])), alpha)  # noqa: E731
    width = TRIAD_SIZE[style][1] * scale
    bare = style == "global"
    parts = []
    for tip, key in zip(tips, ("axis_x", "axis_y", "axis_z")):
        along = tip - origin
        length = float(np.hypot(*along))
        if length < 1.5 * scale:
            continue  # pointing straight at the eye
        along = along / length
        head = 0.0 if bare else min(6.0 * scale, length * 0.4)
        shaft_end = tip - along * head
        if not bare:
            parts.append((*_segment(origin, shaft_end, width + 2.5 * scale), rgba("snap_halo", 150)))
            parts.append((*_head(tip + along * 1.2 * scale, along, head + 2.4 * scale, head * 0.5 + 1.5 * scale),
                          rgba("snap_halo", 150)))
        parts.append((*_segment(origin, shaft_end, width), rgba(key, 200 if bare else 255)))
        if head:
            parts.append((*_head(tip, along, head, head * 0.45), rgba(key)))
    if not bare:
        r = 2.5 * scale
        parts.append((*_box2(origin, r + 1.5 * scale, r + 1.5 * scale), rgba("snap_halo", 200)))
        parts.append((*_box2(origin, r, r), rgba("accent" if style == "working" else "selection")))
    return parts


def _poly2d(parts) -> vtkPolyData:
    """One polydata of per-cell coloured triangles from (points, triangles, rgba) pieces."""
    points, triangles, colors, base = [], [], [], 0
    for pts, tris, rgba in parts:
        points.append(pts)
        triangles.append(tris + base)
        colors.append(np.tile(np.array(rgba, dtype=np.uint8), (len(tris), 1)))
        base += len(pts)
    poly = vtkPolyData()
    if not points:
        return poly
    flat = np.vstack(points)
    poly.SetPoints(_points(np.column_stack([flat, np.zeros(len(flat))])))
    poly.SetPolys(_cells(np.vstack(triangles).ravel(), 3))
    scalars = numpy_to_vtk(np.vstack(colors), deep=True, array_type=VTK_UNSIGNED_CHAR)
    scalars.SetName("colors")
    poly.GetCellData().SetScalars(scalars)
    return poly


@dataclass
class _Hover:
    """Where a click would land and what it snapped to."""

    point: Optional[np.ndarray]
    kind: Optional[str] = None
    owner: str = ""  # the object it snapped to


@dataclass
class _Pool:
    """The snap targets on offer while a tool runs."""

    points: np.ndarray
    kinds: np.ndarray
    owner_of: np.ndarray  # index into owners
    owners: List[str]
    scope: str  # the selected object's name, or "" for every shown object


class _PickRun:
    """A tool's clicks in progress, and what shows them."""

    def __init__(self, tool: Tool, units: Units, extras: Optional[snaps.SnapSet], extras_owner: str,
                 band: vtkActor, prompt: vtkTextActor):
        self.tool = tool
        self.units = units
        self.extras = extras
        self.extras_owner = extras_owner
        self.band = band
        self.prompt = prompt
        self.points: List[np.ndarray] = []
        self.kinds: List[Optional[str]] = []  # what each of `points` snapped to
        self.result: List[np.ndarray] = []  # a repeating tool's last finished clicks, still shown
        self.hover: Optional[_Hover] = None
        self.message = ""  # why the last click was not taken
        self.labels: List[vtkTextActor] = []  # text beside points, reused


# ------------------------------------------------------------------ widget

class Viewport3D(QWidget):
    picked = Signal(str, str)  # (kind, id); ("", "") for a click on empty space
    contextRequested = Signal(str, str, QPoint)  # a right click: what is under it, and where, on screen
    pickFinished = Signal(object, object)  # (tool, clicks) once a tool takes its last click
    pickEnded = Signal(object)  # a tool stopped, finished or cancelled
    drawingPlaneChanged = Signal(str)

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
        self._style = vtkInteractorStyleTrackballCamera()
        self._iren.SetInteractorStyle(self._style)
        self._vtk.Initialize()
        # Drop VTK's built-in keys ('e'/'q' quit, 'w'/'s' change representation); useful ones are re-added below.
        self._iren.RemoveObservers("CharEvent")
        self._iren.AddObserver("KeyPressEvent", self._on_key)
        self._axes_widget = self._axes_marker()
        # Drawn over the model so integration lines on sheets stay visible.
        self._front = vtkRenderer()
        layer = self._window.GetNumberOfLayers()
        self._window.SetNumberOfLayers(layer + 1)
        self._front.SetLayer(layer)
        self._front.InteractiveOff()
        self._front.SetActiveCamera(self._renderer.GetActiveCamera())
        self._window.AddRenderer(self._front)
        self._entries: Dict[str, _Entry] = {}      # "primitive:<id>" / "port:<id>" -> entry
        self._actor_keys: Dict[vtkActor, Tuple[str, str]] = {}
        self._selected: Optional[Tuple[str, str]] = None
        self._shading = "solid"
        self._visible: Dict[str, bool] = {"grid": True, "axes": True, "ports": True, "domain": True, "mesh": False}
        self._press: Optional[Tuple[int, int]] = None
        self._scene = None
        self._project = None
        self._object_visible: Dict[str, bool] = {}
        self._mesh = None
        self._camera_fitted = False
        # name -> (actor, what it was built from); rebuilt only when that changes
        self._overlays: Dict[str, Tuple[Optional[vtkActor], object]] = {}
        self._grid_key = None
        # Ray-cast picker: vtkPropPicker's GPU readback fails under the SSAA pass.
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.003)  # of the window diagonal: enough to hit a sheet outline
        self._right_press: Optional[Tuple[int, int]] = None
        self._pick: Optional[_PickRun] = None  # the tool begin_pick started, while it runs
        self._plane = "xy"  # the drawing plane, of the working coordinate system
        self._pool: Optional[_Pool] = None  # snap targets on offer
        self._object_snaps: Dict[str, Tuple[object, bytes, snaps.SnapSet]] = {}  # id -> (built from, axes, targets)
        self._last_xy: Optional[Tuple[int, int]] = None
        self._selected_cs: Optional[str] = None  # name of the coordinate system being edited, "" for global
        # Coordinate system triads and their names.
        self._triads = self._overlay_2d()
        self._triad_labels = {style: self._triad_label_actor() for style in ("working", "selected")}
        self._front.AddObserver("StartEvent", self._place_triads)
        # Snap marks, re-placed every frame.
        self._marks = self._overlay_2d()
        self._front.AddObserver("StartEvent", self._place_marks)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        toolbar.setSpacing(2)
        self.shading_box = ChoiceBox(SHADING)
        self.shading_box.set_value("solid")
        self.shading_box.committed.connect(self.set_shading)
        toolbar.addWidget(self.shading_box)
        self.toggles: Dict[str, QToolButton] = {}
        for key, label, tip in (("grid", "Grid", "Floor grid"),
                                ("axes", "Axes", "Orientation axes, and the working coordinate system's"),
                                ("ports", "Ports", "Port boxes"), ("domain", "Domain", "Simulation domain"),
                                ("mesh", "Mesh", "Overlay the real FDTD grid this design will run on (M)")):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked(self._visible[key])
            button.toggled.connect(lambda on, key=key: self.set_layer_visible(key, on))
            self.toggles[key] = button
            toolbar.addWidget(button)
        toolbar.addSpacing(10)
        self._build_snap_controls(toolbar)
        self._toolbar = toolbar
        toolbar.addStretch(1)
        self.view_menu = QMenu("View", self)
        self.view_menu.setToolTipsVisible(True)
        for name in VIEWS:
            if name == PICTORIAL[0]:
                self.view_menu.addSeparator()
            action = self.view_menu.addAction(name, lambda name=name: self.view(name))
            if name == DEFAULT_VIEW:
                action.setShortcut(QKeySequence("V"))  # shown only: the 3D view hears V itself
                action.setShortcutContext(Qt.WidgetShortcut)
        view_button = QToolButton()
        view_button.setText("View")
        view_button.setToolTip("Look at the design from the top, bottom, front, back, left or right, or in an "
                               "isometric, dimetric or trimetric view")
        view_button.setPopupMode(QToolButton.InstantPopup)
        view_button.setMenu(self.view_menu)
        toolbar.addWidget(view_button)
        fit = QToolButton()
        fit.setText("Fit")
        fit.setToolTip("Zoom to fit the design")
        fit.clicked.connect(self.fit)
        toolbar.addWidget(fit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(toolbar)
        layout.addWidget(self._vtk, 1)

        # Listen on the camera style, not the interactor: the trackball style grabs the mouse and VTK sends the
        # release only to it. Each handler also calls the style's own.
        style = self._style
        for event, ours, theirs, ours_first in (
                ("LeftButtonPressEvent", self._on_press, style.OnLeftButtonDown, True),
                ("LeftButtonReleaseEvent", self._on_release, style.OnLeftButtonUp, False),
                ("RightButtonPressEvent", self._on_right_press, style.OnRightButtonDown, True),
                ("RightButtonReleaseEvent", self._on_right_release, style.OnRightButtonUp, False)):
            style.AddObserver(event, partial(_both, ours, theirs, ours_first))
        self._iren.AddObserver("MouseMoveEvent", self._on_move)
        # A bound method, so Qt drops the connection when this widget is deleted.
        theme.manager().changed.connect(self._theme_changed)
        self.apply_theme(render=False)

    def _theme_changed(self, _name: str) -> None:
        if not self._closed:
            self.apply_theme()

    def _overlay_2d(self) -> vtkActor2D:
        """A 2D overlay actor, refilled before each frame."""
        actor = vtkActor2D()
        mapper = vtkPolyDataMapper2D()
        mapper.SetScalarModeToUseCellData()
        mapper.SetColorModeToDirectScalars()
        mapper.ScalarVisibilityOn()
        actor.SetMapper(mapper)
        actor.PickableOff()
        actor.SetVisibility(False)
        self._front.AddViewProp(actor)
        return actor

    def _axes_marker(self) -> vtkOrientationMarkerWidget:
        """The orientation triad in the bottom-left corner, coloured from the theme."""
        axes = vtkAxesActor()
        axes.SetConeRadius(0.4)
        axes.SetNormalizedShaftLength(0.8, 0.8, 0.8)
        axes.SetNormalizedTipLength(0.2, 0.2, 0.2)
        for caption in (axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()):
            caption.SetWidth(0.25)
            caption.SetHeight(0.1)
            text = caption.GetCaptionTextProperty()
            text.ItalicOff()
            text.ShadowOff()
        for prop in (axes.GetXAxisShaftProperty(), axes.GetYAxisShaftProperty(), axes.GetZAxisShaftProperty()):
            prop.SetLineWidth(2)
        self._axes = axes
        widget = vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(self._iren)
        widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        widget.SetCurrentRenderer(self._renderer)
        widget.SetEnabled(1)
        widget.InteractiveOff()
        return widget

    def _build_snap_controls(self, toolbar: QHBoxLayout) -> None:
        """Snap (S) with its menu, and the drawing plane. Both are remembered between sessions."""
        settings = QSettings()
        self.snap_action = QAction("Snap", self, checkable=True)
        self.snap_action.setToolTip("Snap clicked points to objects and the grid while drawing, measuring or "
                                    "placing an integration line (S)")
        self.snap_action.setShortcut(QKeySequence("S"))
        self.snap_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.snap_action.setChecked(settings.value("viewport/snap", True, type=bool))
        self.snap_action.toggled.connect(self._snap_changed)
        self.addAction(self.snap_action)
        self.snap_menu = QMenu("Snap", self)
        self.snap_menu.setToolTipsVisible(True)
        self.snap_menu.addAction(self.snap_action)
        self.snap_menu.addSeparator()
        self.snap_kinds: Dict[str, QAction] = {}
        tips = {snaps.FACE: "The centre of a flat face, or of a circle or arc - a hole's middle - marked ○",
                snaps.EDGE: "The middle of an edge, or a quarter "
                "point round a circle, marked △", snaps.CORNER: "Where edges meet, marked □",
                snaps.GRID: "The working grid's nodes, marked +, wherever no object is near"}
        for kind, label, _icon in SNAP_TOGGLES:
            action = QAction(label, self, checkable=True)
            action.setToolTip(tips[kind])
            action.setChecked(settings.value(f"viewport/snap_{kind}", True, type=bool))
            action.toggled.connect(self._snap_changed)
            self.snap_kinds[kind] = action
            self.snap_menu.addAction(action)
        snap_button = QToolButton()
        snap_button.setDefaultAction(self.snap_action)
        snap_button.setPopupMode(QToolButton.MenuButtonPopup)
        snap_button.setMenu(self.snap_menu)
        toolbar.addWidget(snap_button)

        self.plane_menu = QMenu("Drawing plane", self)
        self.plane_actions: Dict[str, QAction] = {}
        group = QActionGroup(self)
        for name in PLANES:
            action = QAction(f"{PLANE_LABELS[name]} plane", self, checkable=True)
            action.setChecked(name == self._plane)
            action.triggered.connect(lambda _=False, name=name: self.set_drawing_plane(name))
            group.addAction(action)
            self.plane_actions[name] = action
            self.plane_menu.addAction(action)
        self.plane_button = QToolButton()
        self.plane_button.setText(PLANE_LABELS[self._plane])
        self.plane_button.setToolTip("Drawing plane: new sheets, and the bases of new solids, lie in this plane of "
                                     "the working coordinate system. The grid follows it.")
        self.plane_button.setPopupMode(QToolButton.InstantPopup)
        self.plane_button.setMenu(self.plane_menu)
        toolbar.addWidget(self.plane_button)

    def add_tool_action(self, action: QAction) -> None:
        """A button for `action` beside the snap controls: the measure tool."""
        button = QToolButton()
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._toolbar.insertWidget(self._toolbar.indexOf(self.plane_button) + 1, button)

    def _paint_snap_icons(self) -> None:
        c = theme.colors()
        for kind, _label, icon_name in SNAP_TOGGLES:
            self.snap_kinds[kind].setIcon(icons.icon(icon_name, c["text_dim"], c["snap"]))

    def _snap_changed(self, _on: bool = False) -> None:
        settings = QSettings()
        settings.setValue("viewport/snap", self.snap_action.isChecked())
        for kind, action in self.snap_kinds.items():
            settings.setValue(f"viewport/snap_{kind}", action.isChecked())
        self._reaim()

    def snaps_to(self, toggle: str) -> bool:
        """Whether points snap to targets of this kind right now."""
        return self.snap_action.isChecked() and self.snap_kinds[toggle].isChecked()

    def set_drawing_plane(self, name: str) -> None:
        if name not in PLANES:
            return
        self._plane = name
        self.plane_actions[name].setChecked(True)
        self.plane_button.setText(PLANE_LABELS[name])
        self._draw_grid(theme.colors())
        self._follow_frame()
        self.drawingPlaneChanged.emit(name)
        self.redraw()

    @property
    def drawing_plane_name(self) -> str:
        return self._plane

    @property
    def scene(self):
        """The design as last drawn (core.em.scene.Scene), or None before the first preview."""
        return self._scene

    def _frame(self):
        """The working coordinate system as placed: origin and unit axes."""
        from EMShrimulator.core.em.scene import WorkingFrame

        return (self._scene.frame if self._scene is not None else None) or WorkingFrame.global_frame()

    def drawing_plane(self) -> Plane:
        """The drawing plane through the working origin."""
        frame = self._frame()
        return Plane.of_frame(frame.origin, frame.axes, self._plane)

    # --------------------------------------------------------------- API

    def close_plotter(self) -> None:
        # Inert from here on: rendering a finalized window would create a new GL context.
        self._closed = True
        self._vtk.hide()
        try:
            self._axes_widget.SetEnabled(0)
            self._vtk.Finalize()
        except Exception:  # noqa: BLE001 - shutting down; nothing useful to report
            pass

    def redraw(self) -> None:
        if not self._closed:
            self._window.Render()

    def apply_theme(self, render: bool = True) -> None:
        c = theme.colors()
        self._paint_snap_icons()
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
        if self._scene is not None:
            # Port, sheet and overlay colours follow the theme.
            self.set_scene(self._scene, self._project, self._object_visible, render=False)
        if render:
            self.redraw()

    def set_scene(self, scene, project, object_visible: Dict[str, bool], render: bool = True) -> None:
        """Draw a new scene, touching only what changed since the last one."""
        self._scene = scene
        self._project = project
        self._object_visible = object_visible
        c = theme.colors()
        live = set()

        for prim_id, solid in scene.solids.items():
            key = f"primitive:{prim_id}"
            live.add(key)
            color, opacity = solid.color or c["accent"], solid.opacity
            entry = self._entries.get(key)
            if entry is not None and entry.geometry is None:  # was a sheet
                self._remove(self._entries.pop(key))
                entry = None
            if entry is None or entry.geometry is not solid.vertices:
                poly = _polydata(solid.vertices, solid.triangles)
                edges_poly = _feature_edges(poly)
                if entry is None:
                    surface = _actor(poly, color, opacity=opacity, pickable=True, specular=True)
                    self._renderer.AddActor(surface)
                    entry = _Entry("primitive", prim_id, surface, None, solid.vertices, None, color, opacity)
                    self._entries[key] = entry
                    self._actor_keys[surface] = ("primitive", prim_id)
                else:
                    _set_input(entry.surface, poly)
                    entry.geometry = solid.vertices
                if edges_poly is None:
                    if entry.edges is not None:
                        self._renderer.RemoveActor(entry.edges)
                        entry.edges = None
                elif entry.edges is None:
                    entry.edges = _actor(edges_poly, c["line_strong"])
                    self._renderer.AddActor(entry.edges)
                else:
                    _set_input(entry.edges, edges_poly)
            entry.color, entry.opacity = color, opacity
            self._paint(entry)

        for prim_id, sheet in scene.sheets.items():
            key = f"primitive:{prim_id}"
            live.add(key)
            color, opacity = sheet.color or c["accent"], sheet.opacity
            signature = (color, opacity, tuple(loop.tobytes() for loop in sheet.loops))
            entry = self._entries.get(key)
            if entry is not None and entry.signature == signature:
                self._paint(entry)
                continue
            if entry is not None:
                self._remove(self._entries.pop(key))
            fill = None
            if len(sheet.loops) == 1 and len(sheet.loops[0]) >= 3:
                fill = _actor(_polygon(sheet.loops[0]), color, opacity=opacity, pickable=True)
                self._renderer.AddActor(fill)
            loops = []
            for loop in sheet.loops:
                if len(loop) < 2:
                    continue
                actor = _actor(_loop(loop), color, line_width=2, pickable=fill is None)
                self._renderer.AddActor(actor)
                loops.append(actor)
            surface = fill or (loops[0] if loops else None)
            if surface is None:
                continue
            entry = _Entry("primitive", prim_id, surface, None, None, signature, color, opacity, loops=loops)
            self._entries[key] = entry
            self._actor_keys[surface] = ("primitive", prim_id)
            self._paint(entry)

        live |= self._update_ports(scene.ports)

        for key in [k for k in self._entries if k not in live]:
            self._remove(self._entries.pop(key))
        for stale in [k for k in self._object_snaps if f"primitive:{k}" not in live]:
            del self._object_snaps[stale]
        self._pool = None

        self._draw_overlays()
        if scene.bounds is None:
            # Emptied: frame the next object afresh.
            self._camera_fitted = False
        elif not self._camera_fitted:
            self._camera_fitted = True
            self.view(DEFAULT_VIEW, render=False)
        self._follow_frame()
        if render:
            self.redraw()

    def _follow_frame(self) -> None:
        """Measurements and drawing-plane clicks follow the working coordinate system as it changes."""
        run = self._pick
        if run is None:
            return
        if isinstance(run.tool, MeasureTool):
            run.tool.plane = self.drawing_plane()
        self._show_pick(render=False)

    def _update_ports(self, ports) -> set:
        c = theme.colors()
        live = set()
        for port_id, port in ports.items():
            key = f"port:{port_id}"
            live.add(key)
            lo = np.minimum(port.p1, port.p2)
            hi = np.maximum(port.p1, port.p2)
            pad = np.where(hi - lo < 1e-9, 1e-3, 0.0)
            lo, hi = lo - pad, hi + pad
            color = c["danger"] if port.excite else c["warn"]
            awaiting = port.awaiting_line  # no box until the line is drawn
            text = f"#{port.number} {port.name}" + ("  · draw its integration line" if awaiting else "")
            line = None if port.line is None else np.asarray(port.line, dtype=float)
            signature = (lo.tobytes(), hi.tobytes(), color, text, None if line is None else line.tobytes())
            entry = self._entries.get(key)
            if entry is not None and entry.signature == signature:
                self._paint(entry)
                continue
            if entry is not None:  # ports are few: rebuilt whole rather than patched
                self._remove(self._entries.pop(key))
            surface = _actor(_box_surface(lo, hi), color, opacity=0.28, pickable=not awaiting)
            edges = None if awaiting else _actor(_box_outline(lo, hi), color, line_width=2)
            arrow = self._arrow(line, color) if line is not None else []
            label = vtkTextActor()
            label.GetPositionCoordinate().SetCoordinateSystemToWorld()
            label.GetTextProperty().SetFontSize(12)
            label.GetTextProperty().BoldOn()
            label.PickableOff()
            for actor in (surface, edges):
                if actor is not None:
                    self._renderer.AddActor(actor)
            for actor in arrow:
                self._front.AddActor(actor)
            self._renderer.AddViewProp(label)
            entry = _Entry("port", port_id, surface, edges, None, signature, color, 0.0 if awaiting else 0.28,
                           label=label, loops=arrow)
            self._entries[key] = entry
            self._actor_keys[surface] = ("port", port_id)
            entry.label.SetInput(text)
            anchor = line[1] if line is not None else (lo + hi) / 2  # beside the arrow's head, not over it
            entry.label.GetPositionCoordinate().SetValue(*anchor.tolist())
            entry.label.GetTextProperty().SetColor(*_rgb(color))
            self._paint(entry)
        return live

    @staticmethod
    def _arrow(line: np.ndarray, color: str) -> List[vtkActor]:
        """The integration line: a shaft from its negative end, a head at its positive one."""
        start, end = line[0], line[1]
        span = end - start
        length = float(np.linalg.norm(span))
        if length < 1e-9:
            return []
        head = length * 0.25
        shaft_end = end - span / length * head
        shaft = _actor(_lines(np.array([[start, shaft_end]])), color, line_width=3)
        cone = _actor(_arrow_head(end, span, head, head * 0.42), color)
        return [shaft, cone]

    def set_mesh(self, mesh: Optional[dict], render: bool = True) -> None:
        self._mesh = mesh
        if self._scene is not None:
            self._draw_overlays()
            if render:
                self.redraw()

    def focus(self, kind: str, key: str) -> None:
        """Turn the camera to look at one object or port, keeping the view direction."""
        entry = self._entries.get(f"{kind}:{key}")
        if entry is None:
            return
        bounds = entry.surface.GetBounds()
        if not all(np.isfinite(bounds)):
            return
        lo = np.array(bounds[0::2])
        hi = np.array(bounds[1::2])
        pad = np.maximum((hi - lo) * 0.6, 0.5)
        lo, hi = lo - pad, hi + pad
        self._renderer.ResetCamera(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        self.redraw()

    def set_selection(self, kind: Optional[str], key: Optional[str]) -> None:
        previous = self._selected
        selected = (kind, key) if kind in ("primitive", "port") else None
        cs = (key or "") if kind == "cs" else None
        cs_changed, self._selected_cs = cs != self._selected_cs, cs
        if selected == previous:
            if cs_changed:
                self.redraw()
            return
        self._selected = selected
        self._pool = None  # a selected object is the only one points snap to
        for selection in (previous, selected):
            if selection is not None:
                entry = self._entries.get(f"{selection[0]}:{selection[1]}")
                if entry is not None:
                    self._paint(entry)
        if self._pick is not None:
            self._reaim(render=False)
        self.redraw()

    def set_shading(self, mode: str) -> None:
        self._shading = mode
        for entry in self._entries.values():
            self._paint(entry)
        self.redraw()

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        self._visible[layer] = visible
        if self.toggles[layer].isChecked() != visible:
            self.toggles[layer].setChecked(visible)
            return
        if layer == "axes":  # the corner triad and the coordinate systems' triads
            self._axes_widget.SetEnabled(1 if visible else 0)
        elif layer == "ports":
            for entry in self._entries.values():
                if entry.kind == "port":
                    self._paint(entry)
        else:
            self._draw_overlays()
        self.redraw()

    def toggle_mesh(self) -> None:
        self.set_layer_visible("mesh", not self._visible["mesh"])

    def fit(self) -> None:
        self._renderer.ResetCamera(self._fit_bounds())
        self.redraw()

    def view(self, name: str, render: bool = True) -> None:
        direction, up = VIEWS[name]
        bounds = self._fit_bounds()
        center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1e-3)
        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(*center.tolist())
        camera.SetPosition(*(center + np.array(direction, dtype=float) / np.linalg.norm(direction) * span * 3).tolist())
        camera.SetViewUp(*up)
        self._renderer.ResetCamera(bounds)
        if render:
            self.redraw()

    def _on_key(self, *_args) -> None:
        key = self._iren.GetKeySym()
        if key == "Escape" and self._pick is not None:
            self.cancel_pick()
        elif key in ("v", "V"):
            self.view(DEFAULT_VIEW)
        elif key in ("r", "R", "f", "F"):
            self.fit()
        elif key in ("Up", "Down"):
            self._renderer.GetActiveCamera().Zoom(1.05 if key == "Up" else 0.95)
            self.redraw()

    # ---------------------------------------------------------- painting

    def _paint(self, entry: _Entry) -> None:
        c = theme.colors()
        selected = self._selected == (entry.kind, entry.key)
        if entry.kind == "port":
            visible = self._visible["ports"]
        else:
            visible = self._object_visible.get(entry.key, True)
        color = _rgb(c["selection"] if selected else entry.color)
        prop = entry.surface.GetProperty()
        prop.SetColor(*color)
        opacity = entry.opacity
        if entry.kind == "primitive" and entry.geometry is not None:
            if self._shading == "xray":
                opacity = min(opacity, 0.3)
            if self._shading == "wireframe":
                prop.SetRepresentationToWireframe()
            else:
                prop.SetRepresentationToSurface()
        boxless = entry.kind == "port" and entry.opacity == 0  # a port still waiting for its line
        if selected and not boxless:
            opacity = max(opacity, 0.55)
        prop.SetOpacity(opacity)
        entry.surface.SetVisibility(visible and not boxless)
        if entry.edges is not None:
            edge_prop = entry.edges.GetProperty()
            if entry.kind == "port" or entry.geometry is None:
                edge_prop.SetColor(*color)
            else:
                edge_prop.SetColor(*_rgb(c["selection"] if selected else c["line_strong"]))
            edge_prop.SetLineWidth(3 if selected else (2 if entry.kind == "port" else 1))
            entry.edges.SetVisibility(visible)
        if entry.label is not None:
            entry.label.SetVisibility(visible)
        for actor in entry.loops:
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(3 if selected else 2)
            actor.SetVisibility(visible)

    def _remove(self, entry: _Entry) -> None:
        self._actor_keys.pop(entry.surface, None)
        for actor in {entry.surface, entry.edges, *entry.loops} - {None}:
            self._renderer.RemoveActor(actor)
            self._front.RemoveActor(actor)
        if entry.label is not None:
            self._renderer.RemoveViewProp(entry.label)

    def _fit_bounds(self) -> Tuple[float, float, float, float, float, float]:
        if self._scene is not None and self._scene.bounds is not None:
            lo, hi = self._scene.bounds
        elif self._mesh:
            lo, hi = np.array(self._mesh["domain_min"]), np.array(self._mesh["domain_max"])
        else:
            lo, hi = np.full(3, -10.0), np.full(3, 10.0)
        pad = np.maximum((hi - lo) * 0.05, 0.5)
        lo, hi = lo - pad, hi + pad
        return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    def _overlay(self, name: str, spec, build) -> None:
        """Show overlay `name` built from `spec`; None hides it. Rebuilds only when `spec` changes."""
        actor, current = self._overlays.get(name, (None, None))
        if spec is None:
            if actor is not None:
                actor.SetVisibility(False)
            return
        if actor is None or spec != current:
            if actor is not None:
                self._renderer.RemoveActor(actor)
            actor = build()
            if actor is not None:
                self._renderer.AddActor(actor)
            self._overlays[name] = (actor, spec)
        if actor is not None:
            actor.SetVisibility(True)

    def _draw_overlays(self) -> None:
        from EMShrimulator.core.em.scene import mesh_planes

        c = theme.colors()
        mesh = self._mesh
        scene = self._scene

        domain = None
        if mesh and self._visible["domain"]:
            domain = (tuple(mesh["domain_min"]), tuple(mesh["domain_max"]), c["text_faint"])
        self._overlay("domain", domain, lambda: _actor(_box_outline(domain[0], domain[1]), domain[2]))

        lines = None
        if mesh and self._visible["mesh"]:
            lines = (tuple(tuple(mesh["lines"].get(a, ())) for a in "xyz"), tuple(mesh["domain_min"]),
                     tuple(mesh["domain_max"]), c["mesh_line"])

        def build_mesh():
            planes = mesh_planes(mesh["lines"], mesh["domain_min"], mesh["domain_max"])
            poly = _lines(np.concatenate(list(planes.values())))
            return None if poly is None else _actor(poly, c["mesh_line"], opacity=0.45)

        self._overlay("mesh", lines, build_mesh)
        self._draw_grid(c)

    def _draw_grid(self, c: Dict[str, str]) -> None:
        """The drawing plane: grid lines, in-plane axes and an origin marker, held just behind coplanar faces."""
        from EMShrimulator.core.em.scene import working_grid

        scene = self._scene
        shown = self._visible["grid"] and scene is not None
        frame, (iu, iv, iw) = self._grid_frame(), PLANES[self._plane]
        bounds = scene.bounds if scene is not None else None
        key = None
        if shown:
            key = (frame.origin.tobytes(), frame.axes.tobytes(),
                   None if bounds is None else (bounds[0].tobytes(), bounds[1].tobytes()),
                   c["vp_grid"], c["vp_grid_major"], c["axis_x"], c["axis_y"], c["axis_z"])
        grid = working_grid(frame, bounds) if key is not None and key != self._grid_key else None
        self._grid_key = key
        axis = ("axis_x", "axis_y", "axis_z")  # each line keeps its axis's colour on any plane
        layers = (("grid_minor", "minor", "vp_grid", 1.0), ("grid_major", "major", "vp_grid_major", 1.0),
                  ("grid_x", "x_axis", axis[iu], 2.0), ("grid_y", "y_axis", axis[iv], 2.0),
                  ("grid_z", "z_axis", axis[iw], 2.5))
        for name, part, color, width in layers:
            spec = None if key is None else (key, part)

            def build(part=part, color=color, width=width):
                poly = _lines(grid[part]) if grid is not None else None
                if poly is None:
                    return None
                actor = _actor(poly, c[color], line_width=width)
                # Push the lines back, behind a face in the same plane.
                actor.GetMapper().SetRelativeCoincidentTopologyLineOffsetParameters(2, 6)
                return actor

            self._overlay(name, spec, build)

        def build_origin():
            poly = vtkPolyData()
            poly.SetPoints(_points(grid["origin"][None, :]))
            poly.SetVerts(_cells(np.array([0]), 1))
            actor = _actor(poly, c["accent"])
            prop = actor.GetProperty()
            prop.SetPointSize(9)
            prop.RenderPointsAsSpheresOn()
            return actor

        self._overlay("grid_origin", None if key is None else (key, "origin"),
                      build_origin if grid is not None else (lambda: None))

    # ------------------------------------------------------- picking points

    def begin_pick(self, tool: Tool, units: Optional[Units] = None, extras: Optional[snaps.SnapSet] = None,
                   extras_owner: str = "") -> None:
        """Run a drawing tool (core.em.drawing): clicks snap to the nearest target or land where the tool's step says.

        Emits pickFinished with the clicks. Esc or a right click ends it.
        """
        self.cancel_pick(render=False)
        c = theme.colors()
        band = _actor(_paths([np.zeros((2, 3))]), c["selection"], line_width=2)
        band.SetVisibility(False)
        self._front.AddActor(band)
        prompt = vtkTextActor()
        prompt.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        prompt.GetPositionCoordinate().SetValue(0.015, 0.97)
        prompt.GetTextProperty().SetFontSize(15)
        prompt.GetTextProperty().BoldOn()
        prompt.GetTextProperty().SetVerticalJustificationToTop()  # the readout lines hang below
        prompt.GetTextProperty().SetColor(*_rgb(c["text"]))
        self._renderer.AddViewProp(prompt)
        self._pick = _PickRun(tool, units or Units(), extras, extras_owner, band, prompt)
        self._pool = None
        self._vtk.setCursor(Qt.CrossCursor)
        self._vtk.setFocus()  # for Esc and S
        self._show_pick()

    def cancel_pick(self, render: bool = True) -> None:
        run, self._pick = self._pick, None
        if run is None:
            return
        self._front.RemoveActor(run.band)
        self._renderer.RemoveViewProp(run.prompt)
        for label in run.labels:
            self._front.RemoveViewProp(label)
        self._vtk.unsetCursor()
        self._pool = None
        self.pickEnded.emit(run.tool)
        if render:
            self.redraw()

    @property
    def picking(self) -> bool:
        return self._pick is not None

    @property
    def pick_tool(self) -> Optional[Tool]:
        return self._pick.tool if self._pick is not None else None

    def _display_to_world(self, x: float, y: float, depth: float) -> np.ndarray:
        self._renderer.SetDisplayPoint(x, y, depth)
        self._renderer.DisplayToWorld()
        world = np.array(self._renderer.GetWorldPoint(), dtype=float)
        return world[:3] / (world[3] or 1.0)

    def _to_screen(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Screen positions (display pixels) and view depths of world points; inf behind the eye."""
        camera = self._renderer.GetActiveCamera()
        matrix = camera.GetCompositeProjectionTransformMatrix(self._renderer.GetTiledAspectRatio(), -1, 1)
        m = np.array([[matrix.GetElement(i, j) for j in range(4)] for i in range(4)])
        h = np.column_stack([points, np.ones(len(points))]) @ m.T
        w = np.where(np.abs(h[:, 3]) > 1e-12, h[:, 3], 1e-12)
        width, height = self._renderer.GetSize()
        ox, oy = self._renderer.GetOrigin()
        screen = np.column_stack([(h[:, 0] / w + 1) * width / 2 + ox, (h[:, 1] / w + 1) * height / 2 + oy])
        depth = (points - np.array(camera.GetPosition())) @ np.array(camera.GetDirectionOfProjection())
        return screen, np.where(h[:, 3] > 0, depth, np.inf)

    def _scale(self) -> float:
        """Display pixels per screen pixel."""
        return float(self._vtk.devicePixelRatioF() or 1.0)

    def _surface_point(self, x: int, y: int) -> Optional[np.ndarray]:
        """Where the ray under the cursor first meets an object."""
        self._picker.Pick(x, y, 0, self._renderer)
        actor = self._picker.GetActor()
        if actor is None or self._actor_keys.get(actor, ("", ""))[0] != "primitive":
            return None
        return np.array(self._picker.GetPickPosition(), dtype=float)

    def _grid_frame(self):
        """The working frame with its axes reordered so its XY plane is the drawing plane."""
        from EMShrimulator.core.em.scene import WorkingFrame

        frame = self._frame()
        return WorkingFrame(frame.origin, frame.axes[list(PLANES[self._plane])])

    def _grid_step(self) -> float:
        from EMShrimulator.core.em.scene import grid_step

        return grid_step(self._grid_frame(), self._scene.bounds if self._scene is not None else None)

    # --------------------------------------------------------- snapping

    def _object_name(self, prim_id: str) -> str:
        shape = self._scene.solids.get(prim_id) or self._scene.sheets.get(prim_id)
        return shape.name if shape is not None else ""

    def _snap_scope(self) -> Tuple[List[str], str]:
        """The objects points snap to (the selected one, else every shown one) and the selected one's name."""
        scene = self._scene
        if scene is None:
            return [], ""
        shown = [k for k in (*scene.solids, *scene.sheets) if self._object_visible.get(k, True)]
        if self._selected is not None and self._selected[0] == "primitive" and self._selected[1] in shown:
            return [self._selected[1]], self._object_name(self._selected[1])
        return shown, ""

    def _targets(self, prim_id: str) -> snaps.SnapSet:
        """One object's snap targets, cached per shape and working system."""
        axes = self._frame().axes
        solid = self._scene.solids.get(prim_id)
        if solid is not None:
            source = solid.vertices
        else:
            source = tuple(loop.tobytes() for loop in self._scene.sheets[prim_id].loops)
        cached = self._object_snaps.get(prim_id)
        if cached is not None and cached[1] == axes.tobytes() and (
                cached[0] is source or (isinstance(source, tuple) and cached[0] == source)):
            return cached[2]
        try:
            if solid is not None:
                targets = snaps.solid_snaps(solid.vertices, solid.triangles, axes)
            else:
                targets = snaps.sheet_snaps(self._scene.sheets[prim_id].loops, axes)
        except Exception:  # noqa: BLE001 - an odd mesh offers no targets; it must not stop the tool
            targets = snaps.SnapSet.empty()
        self._object_snaps[prim_id] = (source, axes.tobytes(), targets)
        return targets

    def _snap_pool(self) -> _Pool:
        if self._pool is not None:
            return self._pool
        run = self._pick
        ids, scope = self._snap_scope() if run is None or run.tool.objects else ([], "")
        sets, owners = [], []
        if run is not None and run.extras is not None and len(run.extras):
            sets.append(run.extras)
            owners.append(run.extras_owner)
        for prim_id in ids:
            sets.append(self._targets(prim_id))
            owners.append(self._object_name(prim_id))
        everything = snaps.SnapSet.concat(sets)
        owner_of = (np.concatenate([np.full(len(s), i) for i, s in enumerate(sets)]) if sets
                    else np.zeros(0, dtype=int))
        self._pool = _Pool(everything.points, everything.kinds, owner_of, owners, scope)
        return self._pool

    def _forced(self) -> Tuple[str, ...]:
        """Snap kinds the running tool always takes, or () when it follows the settings."""
        return self._pick.tool.snap_only if self._pick is not None else ()

    def _snap_target(self, x: int, y: int) -> Optional[_Hover]:
        allowed = list(self._forced()) or [kind for kind in OBJECT_KINDS if self.snaps_to(snaps.TOGGLE_OF[kind])]
        if not allowed:
            return None
        pool = self._snap_pool()
        if not len(pool.points):
            return None
        screen, depth = self._to_screen(pool.points)
        index = snaps.nearest(screen, depth, pool.kinds, (x, y), SNAP_RADIUS_PX * self._scale(), allowed)
        if index is None:
            return None
        return _Hover(pool.points[index].copy(), snaps.POINT_KINDS[int(pool.kinds[index])],
                      pool.owners[int(pool.owner_of[index])])

    def _free(self, x: int, y: int, step: Step) -> _Hover:
        """Where a click with no snap target lands: on the object, its plane or line, or the grid."""
        if step.surface:
            point = self._surface_point(x, y)
            if point is not None:
                return _Hover(point)
        near, far = self._display_to_world(x, y, 0.0), self._display_to_world(x, y, 1.0)
        grid = step.grid and self.snaps_to(snaps.GRID)
        spacing = self._grid_step() * (1.0 if grid else FINE_STEP)
        kind = snaps.GRID if grid else None
        if step.line is not None:
            base, direction = step.line
            point = closest_on_line(base, direction, near, far)
            if point is None:
                return _Hover(None)
            along = direction / np.linalg.norm(direction)
            return _Hover(base + along * snaps.round_to(float(along @ (point - base)), spacing), kind)
        if step.plane is not None:
            point = step.plane.hit(near, far)
            if point is None:
                return _Hover(None)  # looking along the plane
            return _Hover(snaps.on_grid(point, self._frame().origin, step.plane.u, step.plane.v, spacing), kind)
        return _Hover(None)

    def _aim(self, x: int, y: int) -> _Hover:
        """Where a click at (x, y) would land, and what it would take."""
        step = self._pick.tool.step(self._pick.points)
        hover = self._snap_target(x, y) if self.snap_action.isChecked() or self._forced() else None
        if hover is None:
            hover = self._free(x, y, step)
        elif step.project_snaps and step.plane is not None:
            hover.point = step.plane.project(hover.point)
        elif step.project_snaps and step.line is not None:
            base, direction = step.line
            along = direction / np.linalg.norm(direction)
            hover.point = base + along * float(along @ (hover.point - base))
        if hover.point is not None and step.axis_from is not None:
            hover.point = keep_to_axis(hover.point, step.axis_from)
        return hover

    def _reaim(self, render: bool = True) -> None:
        """Aim again where the cursor last was: what it snaps to changed under it."""
        self._pool = None
        if self._pick is None:
            return
        if self._last_xy is not None:
            self._pick.hover = self._aim(*self._last_xy)
        self._show_pick(render)

    # ------------------------------------------------------ showing a pick

    def _show_pick(self, render: bool = True) -> None:
        run = self._pick
        current = run.hover.point if run.hover is not None else None
        if run.points:
            paths = run.tool.preview(run.points, current)
            labels = run.tool.labels(run.points, current, run.units)
        elif run.result:
            paths = run.tool.preview(run.result[:-1], run.result[-1])
            labels = run.tool.labels(run.result[:-1], run.result[-1], run.units)
        else:
            paths, labels = [], []
        poly = _paths(paths)
        if poly is not None:
            _set_input(run.band, poly)
        run.band.SetVisibility(poly is not None)
        self._show_labels(run, labels)
        run.prompt.SetInput(self._pick_text())
        if render:
            self.redraw()

    def _show_labels(self, run: _PickRun, labels) -> None:
        """Text beside points, over everything: a few actors, made as needed and then reused."""
        c = theme.colors()
        while len(run.labels) < len(labels):
            actor = vtkTextActor()
            # A few pixels up and right of its point.
            anchor = vtkCoordinate()
            anchor.SetCoordinateSystemToWorld()
            position = actor.GetPositionCoordinate()
            position.SetCoordinateSystemToDisplay()
            position.SetReferenceCoordinate(anchor)
            position.SetValue(9 * self._scale(), 7 * self._scale())
            text = actor.GetTextProperty()
            text.SetFontSize(14)
            text.BoldOn()
            text.ShadowOn()
            self._front.AddViewProp(actor)
            run.labels.append(actor)
        for index, actor in enumerate(run.labels):
            shown = index < len(labels)
            actor.SetVisibility(shown)
            if shown:
                point, text = labels[index]
                actor.SetInput(text)
                actor.GetPositionCoordinate().GetReferenceCoordinate().SetValue(*(float(v) for v in point))
                actor.GetTextProperty().SetColor(*_rgb(c["text"]))

    def _pick_text(self) -> str:
        """Status text: the last result, the prompt, the snap target and the snapping state."""
        run = self._pick
        tool, units, hover = run.tool, run.units, run.hover
        frame = self._frame()
        lines = []
        finished = bool(run.result) and not run.points
        if finished:
            lines.append(tool.summary(run.result[:-1], run.result[-1], units, frame))
        lines.append(tool.again if finished and tool.again else tool.step(run.points).prompt)
        point = hover.point if hover is not None else None
        if point is not None:
            local = frame.axes @ (point - frame.origin)
            where = f"({', '.join(units.num(float(v)) for v in local)}) {units.label}"
            where += f" in {frame.name}" if frame.name else ""
            if hover.kind is not None:
                where += f" · {snaps.KIND_LABELS[hover.kind]}" + (f" of {hover.owner}" if hover.owner else "")
            lines.append(where)
            if run.points:
                lines.append(tool.summary(run.points, point, units, frame))
        if tool.snap_only:
            pass  # the tool takes its own points
        elif not self.snap_action.isChecked():
            lines.append("Snapping is off: S turns it on.")
        else:
            scope = self._snap_scope()[1]
            if scope:
                lines.append(f"Snapping to {scope} only: it is selected.")
        if run.message:
            lines.append(run.message)
        return "\n".join(line for line in lines if line)

    def _mark_list(self) -> List[Tuple[np.ndarray, str, str]]:
        """(world point, shape, style) of each mark: targets on offer, points clicked and the cursor."""
        run = self._pick
        if run is None:
            return []
        marks = []
        forced = run.tool.snap_only
        if run.extras is not None and (self.snap_action.isChecked() or forced):
            for point, kind in zip(run.extras.points, run.extras.kinds):
                name = snaps.POINT_KINDS[int(kind)]
                if (name in forced) if forced else self.snap_kinds[snaps.TOGGLE_OF[name]].isChecked():
                    marks.append((point, MARK_SHAPE[name], "hint"))
        for point in run.points or run.result:
            marks.append((point, "dot", "fixed"))
        hover = run.hover
        if hover is not None and hover.point is not None:
            marks.append((hover.point, MARK_SHAPE[hover.kind] if hover.kind else "dot", "hover"))
        return marks

    def _place_marks(self, *_args) -> None:
        """Before each frame: the marks, projected through the camera as it now stands."""
        marks = self._mark_list()
        if not marks:
            self._marks.SetVisibility(False)
            return
        screen, depth = self._to_screen(np.array([mark[0] for mark in marks], dtype=float))
        shown = [(xy, shape, style) for xy, d, (_, shape, style) in zip(screen, depth, marks) if np.isfinite(d)]
        self._marks.GetMapper().SetInputData(_marks_poly(shown, self._scale(), theme.colors()))
        self._marks.SetVisibility(bool(shown))

    # ------------------------------------------------- coordinate systems

    def _place_triads(self, *_args) -> None:
        """Place the coordinate system triads before each frame, at a fixed pixel size."""
        from EMShrimulator.core.em.scene import triads as triads_of

        shown = self._scene is not None and self._visible["axes"]
        triads = triads_of(self._scene, self._selected_cs) if shown else []
        labels_used = set()
        parts = []
        if triads:
            c, scale = theme.colors(), self._scale()
            camera = self._renderer.GetActiveCamera()
            right = np.cross(camera.GetDirectionOfProjection(), camera.GetViewUp())
            right /= np.linalg.norm(right) or 1.0
            for frame, style, label in triads:
                origin = np.asarray(frame.origin, dtype=float)
                screen, depth = self._to_screen(np.array([origin, origin + right]))
                pixels = float(np.hypot(*(screen[1] - screen[0])))
                if not np.isfinite(depth[0]) or pixels <= 1e-9:
                    continue  # behind the eye
                reach = TRIAD_SIZE[style][0] * scale / pixels  # world length of the axes
                tips, _ = self._to_screen(origin + np.asarray(frame.axes, dtype=float) * reach)
                parts += _triad_parts(screen[0], tips, style, scale, c)
                if label:
                    self._triad_label(style, label, origin, c)
                    labels_used.add(style)
        for style, actor in self._triad_labels.items():
            actor.SetVisibility(style in labels_used)
        if parts:
            self._triads.GetMapper().SetInputData(_poly2d(parts))
        self._triads.SetVisibility(bool(parts))

    def _triad_label_actor(self) -> vtkTextActor:
        """Text placed a few pixels below and right of a world point, hidden until used."""
        actor = vtkTextActor()
        anchor = vtkCoordinate()
        anchor.SetCoordinateSystemToWorld()
        position = actor.GetPositionCoordinate()
        position.SetCoordinateSystemToDisplay()
        position.SetReferenceCoordinate(anchor)
        position.SetValue(7 * self._scale(), -16 * self._scale())
        prop = actor.GetTextProperty()
        prop.SetFontSize(12)
        prop.BoldOn()
        prop.ShadowOn()
        actor.PickableOff()
        actor.SetVisibility(False)
        self._front.AddViewProp(actor)
        return actor

    def _triad_label(self, style: str, text: str, origin: np.ndarray, c: Dict[str, str]) -> None:
        """A coordinate system's name, just below and right of its origin."""
        actor = self._triad_labels[style]
        actor.SetInput(text)
        actor.GetPositionCoordinate().GetReferenceCoordinate().SetValue(*(float(v) for v in origin))
        actor.GetTextProperty().SetColor(*_rgb(c["accent" if style == "working" else "selection"]))

    def _pick_click(self, x: int, y: int) -> None:
        run = self._pick
        self._last_xy = (x, y)
        hover = self._aim(x, y)
        if hover.point is None:
            return  # nothing under the cursor to take: keep waiting
        if run.tool.repeat and not run.points:
            run.result = []
        run.message = run.tool.refuse(run.points, hover.point)
        if run.message:
            self._show_pick()
            return
        run.points.append(hover.point)
        run.kinds.append(hover.kind)
        run.hover = hover
        if len(run.points) < run.tool.clicks:
            self._show_pick()
            return
        tool, points = run.tool, list(run.points)
        tool.snapped = list(run.kinds)
        if tool.repeat:
            run.result, run.points, run.kinds = points, [], []
            self._show_pick()
        else:
            self.cancel_pick()
        self.pickFinished.emit(tool, points)

    def _on_move(self, *_args) -> None:
        run = self._pick
        if run is None or self._style.GetState() != 0:  # orbiting, panning or zooming: nothing to aim at
            return
        self._last_xy = tuple(self._iren.GetEventPosition())
        run.hover = self._aim(*self._last_xy)
        run.message = ""
        self._show_pick()

    # ------------------------------------------------------------ picking

    def _on_press(self, *_args) -> None:
        self._press = tuple(self._iren.GetEventPosition())

    def _on_release(self, *_args) -> None:
        if self._press is None:
            return
        x, y = self._iren.GetEventPosition()
        px, py = self._press
        self._press = None
        if abs(x - px) > CLICK_TOLERANCE_PX or abs(y - py) > CLICK_TOLERANCE_PX:
            return  # that was an orbit, not a click
        if self._pick is not None:
            self._pick_click(x, y)
            return
        self.picked.emit(*self._pick_at(x, y))

    def _pick_at(self, x: int, y: int) -> Tuple[str, str]:
        self._picker.Pick(x, y, 0, self._renderer)
        actor = self._picker.GetActor()
        return self._actor_keys.get(actor, ("", "")) if actor is not None else ("", "")

    def _on_right_press(self, *_args) -> None:
        self._right_press = tuple(self._iren.GetEventPosition())

    def _on_right_release(self, *_args) -> None:
        if self._right_press is None:
            return
        x, y = self._iren.GetEventPosition()
        px, py = self._right_press
        self._right_press = None
        if abs(x - px) > CLICK_TOLERANCE_PX or abs(y - py) > CLICK_TOLERANCE_PX:
            return  # a zoom drag
        if self._pick is not None:
            self.cancel_pick()
            return
        kind, key = self._pick_at(x, y)
        where = QCursor.pos()
        # Deferred: a menu's event loop must not run inside VTK's release handler.
        QTimer.singleShot(0, lambda: self.contextRequested.emit(kind, key, where))


def _both(ours, theirs, ours_first: bool, *_event) -> None:
    """A mouse button event for the viewport and for the camera style, in the order given."""
    for handler in ((ours, theirs) if ours_first else (theirs, ours)):
        handler()


def _rgb(hex_color: str) -> Tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
