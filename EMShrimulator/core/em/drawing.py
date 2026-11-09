"""Click-by-click drawing tools for the 3D view. Pure numpy; the viewport supplies snap targets and rays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from EMShrimulator.core.em.model import PrimitiveKind, SheetShape

# (u, v, normal) rows of the working frame for each drawing plane.
PLANES: Dict[str, Tuple[int, int, int]] = {"xy": (0, 1, 2), "yz": (1, 2, 0), "zx": (2, 0, 1)}
PLANE_LABELS = {"xy": "XY", "yz": "YZ", "zx": "ZX"}
NORMAL_AXIS = {"xy": "z", "yz": "x", "zx": "y"}
TINY = 1e-9  # mm: a size smaller than this is nothing


def _tidy(value: float) -> str:
    if abs(value) < 1e-9:  # what is left of a zero after a transform or two
        return "0"
    return f"{value:.5g}"


@dataclass
class Units:
    """How lengths read out: the label and how many mm one display unit is."""

    label: str = "mm"
    per_unit: float = 1.0

    def num(self, mm: float) -> str:
        return _tidy(mm / self.per_unit)

    def length(self, mm: float) -> str:
        return f"{self.num(mm)} {self.label}"


@dataclass
class Plane:
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    name: str = ""  # "xy", "yz" or "zx" for a drawing plane

    @classmethod
    def of_frame(cls, origin, axes, name: str = "xy") -> "Plane":
        """One of the frame's three planes through its origin. `axes` are the frame's unit x, y, z rows."""
        iu, iv, _ = PLANES[name]
        axes = np.asarray(axes, dtype=float)
        return cls(np.asarray(origin, dtype=float), axes[iu], axes[iv], name)

    @classmethod
    def normal_to(cls, origin, normal) -> "Plane":
        """The plane through `origin` facing `normal`."""
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        across = np.eye(3)[int(np.argmin(np.abs(n)))]
        u = across - n * float(across @ n)
        u /= np.linalg.norm(u)
        return cls(np.asarray(origin, dtype=float), u, np.cross(n, u))

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.u, self.v)
        return n / np.linalg.norm(n)

    def through(self, point) -> "Plane":
        return Plane(np.asarray(point, dtype=float), self.u, self.v, self.name)

    def project(self, point) -> np.ndarray:
        n = self.normal
        return point - n * float(n @ (point - self.origin))

    def hit(self, near, far) -> Optional[np.ndarray]:
        """Where the ray from `near` towards `far` meets the plane; None if parallel."""
        n = self.normal
        ray = np.asarray(far, dtype=float) - near
        facing = float(n @ ray)
        if abs(facing) < 1e-9 * (float(np.linalg.norm(ray)) or 1.0):
            return None
        return near + ray * float(n @ (self.origin - near)) / facing

    def ring(self, centre, radius: float, segments: int = 64) -> np.ndarray:
        """A closed circle in this plane, for previews."""
        angles = np.linspace(0.0, 2 * np.pi, segments + 1)
        return centre + radius * (np.outer(np.cos(angles), self.u) + np.outer(np.sin(angles), self.v))


def closest_on_line(point, direction, near, far) -> Optional[np.ndarray]:
    """The point on the line through `point` along `direction` nearest the ray from `near` to `far`."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    r = np.asarray(far, dtype=float) - near
    r = r / np.linalg.norm(r)
    w = np.asarray(point, dtype=float) - near
    b = float(d @ r)
    if 1 - b * b < 1e-9:
        return None  # looking straight down the line
    return point + d * (b * float(r @ w) - float(d @ w)) / (1 - b * b)


def keep_to_axis(point: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The point moved onto the nearest global axis through `start`."""
    along = np.array(start, dtype=float)
    index = int(np.argmax(np.abs(point - start)))
    along[index] = point[index]
    return along


@dataclass
class Step:
    """What one click asks for and where it may land."""

    prompt: str
    plane: Optional[Plane] = None                          # a free click lands here
    line: Optional[Tuple[np.ndarray, np.ndarray]] = None   # or on this line (point, direction): a height
    project_snaps: bool = False  # snapped points are projected onto the plane or line
    surface: bool = False  # free clicks land on the object under the cursor first
    axis_from: Optional[np.ndarray] = None  # keeps to the global axis it runs along most from here
    grid: bool = True             # a free click may snap to the grid


class Tool:
    """A sequence of clicks in the 3D view."""

    label = ""
    clicks = 2
    repeat = False  # start over after the last click (the measure tool)
    again = ""      # a repeating tool's prompt once it has a result to show
    # Snap kinds a tool always takes; () follows the snap settings.
    snap_only: Tuple[str, ...] = ()
    objects = True  # objects' own snap targets are offered too
    snapped: List[Optional[str]] = []  # what each click snapped to, or None

    def step(self, points: List[np.ndarray]) -> Step:
        raise NotImplementedError

    def refuse(self, points: List[np.ndarray], point: np.ndarray) -> str:
        """Why `point` cannot be the next click, or "" when it can."""
        return ""

    def preview(self, points: List[np.ndarray], current: Optional[np.ndarray]) -> List[np.ndarray]:
        """Polylines to show for the clicks so far and the point under the mouse."""
        return []

    def labels(self, points: List[np.ndarray], current: Optional[np.ndarray], units: Units
               ) -> List[Tuple[np.ndarray, str]]:
        """Text to show beside points in the view."""
        return []

    def summary(self, points: List[np.ndarray], current: Optional[np.ndarray], units: Units, frame=None) -> str:
        """Status lines for the clicks so far; `frame` is the working coordinate system."""
        return ""


class LineTool(Tool):
    """A port's integration line: start, then end, kept to an axis."""

    label = "integration line"

    def __init__(self, plane: Plane, prompts: Tuple[str, str]):
        self.plane = plane
        self.prompts = prompts

    def step(self, points):
        return Step(self.prompts[min(len(points), 1)], plane=self.plane, project_snaps=True, grid=False,
                    axis_from=points[0] if points else None)

    def refuse(self, points, point):
        if points and float(np.linalg.norm(point - points[0])) < TINY:
            return "The end must be somewhere else than the start."
        return ""

    def preview(self, points, current):
        return [np.array([points[0], current])] if points and current is not None else []

    def summary(self, points, current, units, frame=None):
        if not points or current is None:
            return ""
        return f"length {units.length(float(np.linalg.norm(current - points[0])))}"


def measure(a, b, axes=None) -> Dict[str, float]:
    """Distance, offsets, θ/φ direction and axis angles of the span a → b, in degrees along `axes`."""
    axes = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
    d = axes @ (np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
    length = float(np.linalg.norm(d))
    out = {"distance": length, "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2])}
    angles = ("theta", "phi", "to_x", "to_y", "to_z")
    if length < TINY:
        return {**out, **dict.fromkeys(angles, float("nan"))}
    cosines = np.clip(d / length, -1.0, 1.0)
    to_x, to_y, to_z = np.degrees(np.arccos(cosines))
    across = float(np.hypot(d[0], d[1]))
    phi = float(np.degrees(np.arctan2(d[1], d[0]))) if across > length * 1e-9 else float("nan")
    return {**out, "theta": float(to_z), "phi": phi, "to_x": float(to_x), "to_y": float(to_y), "to_z": float(to_z)}


def _angle(degrees: float) -> str:
    return "—" if np.isnan(degrees) else f"{_tidy(degrees)}°"


class MeasureTool(Tool):
    """Measure between two points in the working coordinate system."""

    label = "measure"
    repeat = True
    again = "Click a new first point to measure again.   Esc ends measuring."

    def __init__(self, plane: Plane):
        self.plane = plane  # where free clicks land

    def step(self, points):
        prompt = ("Measure: click the first point.   Esc ends measuring." if not points else
                  "Click the second point.   Esc ends measuring.")
        return Step(prompt, plane=self.plane, surface=True)

    def refuse(self, points, point):
        if points and float(np.linalg.norm(point - points[0])) < TINY:
            return "That is the first point again: click a second one somewhere else."
        return ""

    def preview(self, points, current):
        return [np.array([points[0], current])] if points and current is not None else []

    def labels(self, points, current, units):
        if not points:
            return []
        if current is None:
            return [(points[0], "1")]
        span = float(np.linalg.norm(current - points[0]))
        return [(points[0], "1"), (current, "2"), ((points[0] + current) / 2, units.length(span))]

    def summary(self, points, current, units, frame=None):
        if not points or current is None:
            return ""
        origin = np.zeros(3) if frame is None else np.asarray(frame.origin, dtype=float)
        axes = np.eye(3) if frame is None else np.asarray(frame.axes, dtype=float)
        name = getattr(frame, "name", "") or "the global system"
        m = measure(points[0], current, axes)

        def at(point) -> str:
            return "(" + ", ".join(units.num(float(v)) for v in axes @ (point - origin)) + ")"

        return "\n".join([
            f"1 {at(points[0])}   2 {at(current)} {units.label}, in {name}",
            f"distance {units.length(m['distance'])}   Δx {units.num(m['dx'])}   Δy {units.num(m['dy'])}   "
            f"Δz {units.num(m['dz'])} {units.label}",
            # Spelled out: the 3D view's font has Δ but no θ or φ.
            f"theta {_angle(m['theta'])}   phi {_angle(m['phi'])}   angle to x {_angle(m['to_x'])}, "
            f"y {_angle(m['to_y'])}, z {_angle(m['to_z'])}",
        ])


# ------------------------------------------------------------------ coordinate systems

class OriginTool(Tool):
    """One point: a new relative coordinate system's origin."""

    label = "origin"
    clicks = 1

    def __init__(self, plane: Plane, mode: str):
        self.plane = plane
        self.mode = mode  # the new system's: "offset" or "both"

    def step(self, points):
        return Step("Click the new coordinate system's origin.   Esc cancels.", plane=self.plane, surface=True)


class AnchorTool(Tool):
    """One snap point of an object for a coordinate system to sit on."""

    label = "anchor"
    clicks = 1
    objects = False
    snap_only = ("face", "edge", "corner", "centre")

    def __init__(self, object_id: str, object_name: str, mode: str = "offset", cs_name: str = ""):
        self.object_id = object_id
        self.object_name = object_name
        self.mode = mode
        self.cs_name = cs_name  # a system being moved to another point; "" for a new one

    def step(self, points):
        return Step(f"Click where on {self.object_name} the coordinate system sits: its centre, a face centre, "
                    "an edge centre or a corner.   Esc cancels.", grid=False)


# ------------------------------------------------------------------ objects

class DrawTool(Tool):
    """Draws one kind of object; `fields` turns its clicks into the object's values."""

    kind: PrimitiveKind = PrimitiveKind.BOX
    shape: Optional[SheetShape] = None

    def __init__(self, plane: Plane):
        self.plane = plane
        self.n = self.plane.normal

    @property
    def normal_index(self) -> int:
        return PLANES[self.plane.name][2]

    def in_plane(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """b - a along the plane's u and v."""
        return float((b - a) @ self.plane.u), float((b - a) @ self.plane.v)

    def rectangle(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        du, dv = self.in_plane(a, b)
        u, v = self.plane.u * du, self.plane.v * dv
        return np.array([a, a + u, a + u + v, a + v, a])

    def fields(self, local: np.ndarray) -> Dict[str, Any]:
        """The object's values from its clicks, given in the working coordinate system."""
        raise NotImplementedError


def _clean(values) -> Tuple[float, ...]:
    """Numbers as stored: no float noise from the transforms they came through."""
    return tuple(float(f"{float(v):.9g}") + 0.0 for v in np.atleast_1d(values))


class BoxTool(DrawTool):
    kind, label, clicks = PrimitiveKind.BOX, "box", 3

    def step(self, points):
        if not points:
            return Step("Click the first corner of the box's base.   Esc cancels.", plane=self.plane)
        if len(points) == 1:
            return Step("Click the opposite corner of its base.   Esc cancels.", plane=self.plane.through(points[0]),
                        project_snaps=True)
        return Step("Click its height.   Esc cancels.", line=(points[1], self.n), project_snaps=True)

    def refuse(self, points, point):
        if len(points) == 1 and min(abs(d) for d in self.in_plane(points[0], point)) < TINY:
            return "A box's base needs a width and a length: click off both lines through the first corner."
        if len(points) == 2 and abs(float((point - points[1]) @ self.n)) < TINY:
            return "A box needs a height."
        return ""

    def preview(self, points, current):
        if current is None or not points:
            return []
        if len(points) == 1:
            return [self.rectangle(points[0], current)]
        base = self.rectangle(points[0], points[1])
        lift = self.n * float((current - points[1]) @ self.n)
        return [base, base + lift] + [np.array([p, p + lift]) for p in base[:4]]

    def summary(self, points, current, units, frame=None):
        if current is None or not points:
            return ""
        du, dv = self.in_plane(points[0], current if len(points) == 1 else points[1])
        size = f"{units.num(abs(du))} × {units.num(abs(dv))}"
        if len(points) == 2:
            size += f" × {units.num(abs(float((current - points[1]) @ self.n)))}"
        return f"{size} {units.label}"

    def fields(self, local):
        # The third click sits straight above the second.
        return {"p1": _clean(local[0]), "p2": _clean(local[2])}


class RectangleTool(DrawTool):
    kind, shape, label = PrimitiveKind.SHEET, SheetShape.RECTANGLE, "rectangle"

    def step(self, points):
        if not points:
            return Step("Click the first corner.   Esc cancels.", plane=self.plane)
        return Step("Click the opposite corner.   Esc cancels.", plane=self.plane.through(points[0]),
                    project_snaps=True)

    def refuse(self, points, point):
        if points and min(abs(d) for d in self.in_plane(points[0], point)) < TINY:
            return "It needs a width and a length: click off both lines through the first corner."
        return ""

    def outline(self, a, b) -> np.ndarray:
        return self.rectangle(a, b)

    def preview(self, points, current):
        return [self.outline(points[0], current)] if points and current is not None else []

    def summary(self, points, current, units, frame=None):
        if not points or current is None:
            return ""
        du, dv = self.in_plane(points[0], current)
        return f"{units.num(abs(du))} × {units.num(abs(dv))} {units.label}"

    def fields(self, local):
        iu, iv, _ = PLANES[self.plane.name]
        span = np.abs(local[1] - local[0])
        return {"normal_axis": NORMAL_AXIS[self.plane.name], "origin": _clean((local[0] + local[1]) / 2),
                "sheet_size": _clean((span[iu], span[iv]))}


class EllipseTool(RectangleTool):
    shape, label = SheetShape.ELLIPSE, "ellipse"

    def outline(self, a, b):
        du, dv = self.in_plane(a, b)
        angles = np.linspace(0.0, 2 * np.pi, 65)
        centre = (a + b) / 2
        return centre + np.outer(np.cos(angles) * du / 2, self.plane.u) + np.outer(np.sin(angles) * dv / 2,
                                                                                   self.plane.v)


class CircleTool(DrawTool):
    kind, shape, label = PrimitiveKind.SHEET, SheetShape.CIRCLE, "circle"

    def step(self, points):
        if not points:
            return Step("Click the centre.   Esc cancels.", plane=self.plane)
        return Step("Click a point on the rim.   Esc cancels.", plane=self.plane.through(points[0]),
                    project_snaps=True)

    def radius(self, centre, point) -> float:
        return float(np.hypot(*self.in_plane(centre, point)))

    def refuse(self, points, point):
        return "The radius cannot be zero." if len(points) == 1 and self.radius(points[0], point) < TINY else ""

    def preview(self, points, current):
        if not points or current is None:
            return []
        return [self.plane.ring(points[0], self.radius(points[0], current)), np.array([points[0], current])]

    def summary(self, points, current, units, frame=None):
        return f"radius {units.length(self.radius(points[0], current))}" if points and current is not None else ""

    def fields(self, local):
        rim = local[1] - local[0]
        rim[self.normal_index] = 0.0
        return {"normal_axis": NORMAL_AXIS[self.plane.name], "origin": _clean(local[0]),
                "radius": _clean(np.linalg.norm(rim))[0]}


class CylinderTool(CircleTool):
    kind, shape, label, clicks = PrimitiveKind.CYLINDER, None, "cylinder", 3

    def step(self, points):
        if len(points) < 2:
            return CircleTool.step(self, points)
        return Step("Click its height.   Esc cancels.", line=(points[0], self.n), project_snaps=True)

    def refuse(self, points, point):
        if len(points) == 2 and abs(float((point - points[0]) @ self.n)) < TINY:
            return "A cylinder needs a height."
        return CircleTool.refuse(self, points, point)

    def preview(self, points, current):
        if len(points) < 2:
            return CircleTool.preview(self, points, current)
        if current is None:
            return []
        radius = self.radius(points[0], points[1])
        lift = self.n * float((current - points[0]) @ self.n)
        base = self.plane.ring(points[0], radius)
        sides = [np.array([points[0] + d * radius, points[0] + d * radius + lift])
                 for d in (self.plane.u, -self.plane.u, self.plane.v, -self.plane.v)]
        return [base, base + lift, np.array([points[0], points[0] + lift])] + sides

    def summary(self, points, current, units, frame=None):
        if len(points) < 2:
            return CircleTool.summary(self, points, current, units)
        if current is None:
            return ""
        height = abs(float((current - points[0]) @ self.n))
        return f"radius {units.num(self.radius(points[0], points[1]))} · height {units.length(height)}"

    def fields(self, local):
        rim = local[1] - local[0]
        rim[self.normal_index] = 0.0
        return {"p1": _clean(local[0]), "p2": _clean(local[2]), "radius": _clean(np.linalg.norm(rim))[0]}


class SphereTool(DrawTool):
    kind, label = PrimitiveKind.SPHERE, "sphere"

    def step(self, points):
        if not points:
            return Step("Click the centre.   Esc cancels.", plane=self.plane)
        # A snapped radius point is kept as is.
        return Step("Click a point on its surface.   Esc cancels.", plane=self.plane.through(points[0]))

    def refuse(self, points, point):
        return "The radius cannot be zero." if points and float(np.linalg.norm(point - points[0])) < TINY else ""

    def preview(self, points, current):
        if not points or current is None:
            return []
        radius = float(np.linalg.norm(current - points[0]))
        side = Plane(points[0], self.plane.u, self.n)
        upright = Plane(points[0], self.plane.v, self.n)
        return [self.plane.ring(points[0], radius), side.ring(points[0], radius), upright.ring(points[0], radius),
                np.array([points[0], current])]

    def summary(self, points, current, units, frame=None):
        if not points or current is None:
            return ""
        return f"radius {units.length(float(np.linalg.norm(current - points[0])))}"

    def fields(self, local):
        return {"center": _clean(local[0]), "radius": _clean(np.linalg.norm(local[1] - local[0]))[0]}


DRAW_TOOLS: Dict[Tuple[PrimitiveKind, Optional[SheetShape]], type] = {
    (PrimitiveKind.BOX, None): BoxTool,
    (PrimitiveKind.CYLINDER, None): CylinderTool,
    (PrimitiveKind.SPHERE, None): SphereTool,
    (PrimitiveKind.SHEET, SheetShape.RECTANGLE): RectangleTool,
    (PrimitiveKind.SHEET, SheetShape.CIRCLE): CircleTool,
    (PrimitiveKind.SHEET, SheetShape.ELLIPSE): EllipseTool,
}


def draw_tool(kind: PrimitiveKind, shape: Optional[SheetShape], plane: Plane) -> Optional[DrawTool]:
    """The drawing tool for this kind, or None if it is only typed in."""
    cls = DRAW_TOOLS.get((PrimitiveKind(kind), SheetShape(shape) if shape else None))
    return cls(plane) if cls is not None else None


def drawn_fields(tool: DrawTool, local: Sequence[np.ndarray]) -> Dict[str, Any]:
    return tool.fields(np.array(local, dtype=float))
