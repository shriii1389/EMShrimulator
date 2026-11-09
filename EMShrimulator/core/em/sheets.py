"""Zero-thickness sheets and the port geometry derived from them (shared by viewport, mesh and solver)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from EMShrimulator.core.em.model import Axis, Port, PortKind, Primitive, PrimitiveKind, SheetShape

# openEMS refuses zero-length guided-mode ports, so their sheets get extruded.
GUIDED_KINDS = frozenset({PortKind.COAXIAL, PortKind.RECT_WAVEGUIDE, PortKind.MICROSTRIP})

DEFAULT_EXTRUDE_CELLS = 10

# In-plane axes for each normal direction (CSXCAD convention).
UV_AXES: Dict[str, Tuple[int, int]] = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


def default_rot_axis(normal_axis: str) -> str:
    """A revolve's default sweep axis: the outline's v axis, like an (r, z) profile."""
    return "xyz"[UV_AXES[normal_axis][1]]


def normal_index(axis: Axis) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def is_sheet(prim: Primitive) -> bool:
    return prim.kind == PrimitiveKind.SHEET


def is_construction(prim: Primitive) -> bool:
    """A sheet with no material: marks a port, never built."""
    return is_sheet(prim) and not (prim.material or "").strip()


def is_built(prim: Primitive) -> bool:
    """Reaches the mesh and the solver: a model object that is not a construction sheet."""
    return prim.model and not is_construction(prim)


def sheet_by_name(prims: Sequence[Primitive], name: Optional[str]) -> Optional[Primitive]:
    """Find a sheet by name, searching boolean children too."""
    if not name:
        return None
    stack = list(prims)
    while stack:
        p = stack.pop()
        if p.name == name and is_sheet(p):
            return p
        stack.extend(p.children or [])
    return None


def _f(value, fallback: float = 0.0) -> float:
    """Float, or `fallback` for unresolved/non-finite values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def outline(prim: Primitive) -> List[List[Tuple[float, float]]]:
    """Closed in-plane loops relative to the origin. A ring gives two (outer, inner)."""
    shape = prim.sheet_shape or SheetShape.RECTANGLE
    segments = max(8, min(int(prim.sheet_segments or 48), 512))

    if shape == SheetShape.RECTANGLE:
        u = _f(prim.sheet_size[0] if prim.sheet_size else 0.0)
        v = _f(prim.sheet_size[1] if prim.sheet_size else 0.0)
        hu, hv = u / 2, v / 2
        return [[(-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv)]]

    if shape == SheetShape.POLYGON:
        pts = [(_f(a), _f(b)) for a, b in (prim.points or [])]
        return [pts] if len(pts) >= 3 else [[]]

    def ellipse(ru: float, rv: float) -> List[Tuple[float, float]]:
        return [
            (ru * math.cos(2 * math.pi * i / segments), rv * math.sin(2 * math.pi * i / segments))
            for i in range(segments)
        ]

    if shape == SheetShape.CIRCLE:
        r = _f(prim.radius)
        return [ellipse(r, r)]

    if shape == SheetShape.ELLIPSE:
        u = _f(prim.sheet_size[0] if prim.sheet_size else 0.0)
        v = _f(prim.sheet_size[1] if prim.sheet_size else 0.0)
        return [ellipse(u / 2, v / 2)]

    if shape == SheetShape.RING:
        r_out = _f(prim.radius)
        r_in = _f(prim.radius_inner)
        return [ellipse(r_out, r_out), ellipse(r_in, r_in)]

    return [[]]


def lift(prim: Primitive, uv: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Lift in-plane points onto the sheet's plane in local 3D coordinates."""
    axis = prim.normal_axis or "z"
    n = normal_index(axis)
    iu, iv = UV_AXES[axis]
    origin = [_f(c) for c in (prim.origin or (0.0, 0.0, 0.0))]
    out = np.zeros((len(uv), 3), dtype=float)
    for i, (u, v) in enumerate(uv):
        out[i, iu] = origin[iu] + u
        out[i, iv] = origin[iv] + v
        out[i, n] = origin[n]
    return out


def local_points(prim: Primitive) -> np.ndarray:
    """Every outline point of the sheet, in local 3D coordinates."""
    loops = [loop for loop in outline(prim) if loop]
    if not loops:
        return lift(prim, [(0.0, 0.0)])
    return np.vstack([lift(prim, loop) for loop in loops])


def extent_uv(prim: Primitive) -> Tuple[float, float]:
    """The sheet's full (u, v) bounding size in its own plane."""
    pts = np.array([p for loop in outline(prim) for p in loop], dtype=float)
    if pts.size == 0:
        return 0.0, 0.0
    return float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))


def coax_radii(prim: Primitive) -> Tuple[float, float]:
    """(inner, outer) coax radii. Only a ring has an inner radius."""
    inner = _f(prim.radius_inner) if (prim.sheet_shape == SheetShape.RING) else 0.0
    return inner, _f(prim.radius)


# --- integration lines ---

# Ports excited along a user-drawn integration line.
LINE_KINDS = frozenset({PortKind.LUMPED, PortKind.MICROSTRIP})


def sheet_ops(sheet: Primitive, cs_by_name) -> list:
    """What places the sheet: its coordinate system, then its own transforms."""
    from EMShrimulator.core.em.transforms import effective_ops

    return effective_ops(sheet.coordinate_system, sheet.transforms, cs_by_name)


def line_ends(port: Port) -> Optional[np.ndarray]:
    """The integration line's ends in the sheet's frame, (2, 3); None until both are set."""
    if port.line_start is None or port.line_end is None:
        return None
    return np.array([[_f(c) for c in port.line_start], [_f(c) for c in port.line_end]], dtype=float)


def snap_points(prim: Primitive) -> np.ndarray:
    """Points a line's ends snap to on a sheet: centre, corners, edge midpoints or quarter points."""
    shape = prim.sheet_shape or SheetShape.RECTANGLE
    uv: List[Tuple[float, float]] = [(0.0, 0.0)]
    if shape in (SheetShape.RECTANGLE, SheetShape.POLYGON):
        for loop in outline(prim):
            for i, (u, v) in enumerate(loop):
                nu, nv = loop[(i + 1) % len(loop)]
                uv += [(u, v), ((u + nu) / 2, (v + nv) / 2)]
    else:
        if shape == SheetShape.ELLIPSE:
            size = prim.sheet_size or (0.0, 0.0)
            radii = [(_f(size[0]) / 2, _f(size[1]) / 2)]
        else:
            radii = [(_f(prim.radius),) * 2]
            if shape == SheetShape.RING:
                radii.append((_f(prim.radius_inner),) * 2)
        for ru, rv in radii:
            uv += [(ru, 0.0), (-ru, 0.0), (0.0, rv), (0.0, -rv)]
    return lift(prim, uv)


def snap_kinds(prim: Primitive) -> List[str]:
    """The snap kind of each of snap_points."""
    shape = prim.sheet_shape or SheetShape.RECTANGLE
    kinds = ["face"]
    if shape in (SheetShape.RECTANGLE, SheetShape.POLYGON):
        for loop in outline(prim):
            kinds += ["corner", "edge"] * len(loop)
    else:
        kinds += ["quadrant"] * (8 if shape == SheetShape.RING else 4)
    return kinds


def _in_plane(prim: Primitive, point: np.ndarray) -> Tuple[float, float]:
    origin = [_f(c) for c in (prim.origin or (0.0, 0.0, 0.0))]
    iu, iv = UV_AXES[prim.normal_axis or "z"]
    return float(point[iu] - origin[iu]), float(point[iv] - origin[iv])


def _inside(u: float, v: float, loop) -> bool:
    inside = False
    for i, (u1, v1) in enumerate(loop):
        u2, v2 = loop[i - 1]
        if (v1 > v) != (v2 > v) and u < u1 + (v - v1) * (u2 - u1) / (v2 - v1):
            inside = not inside
    return inside


def _near(u: float, v: float, loop, tol: float) -> bool:
    for i, (u1, v1) in enumerate(loop):
        u2, v2 = loop[i - 1]
        du, dv = u2 - u1, v2 - v1
        t = max(0.0, min(1.0, ((u - u1) * du + (v - v1) * dv) / (du * du + dv * dv or 1.0)))
        if math.hypot(u - (u1 + t * du), v - (v1 + t * dv)) <= tol:
            return True
    return False


def _tolerance(prim: Primitive) -> float:
    # Loose enough for a point on the true circle.
    return max(5e-3 * max(extent_uv(prim)), 1e-9)


def in_plane(prim: Primitive, point: np.ndarray) -> bool:
    """Whether a local point lies in the sheet's plane."""
    n = normal_index(prim.normal_axis or "z")
    return abs(point[n] - _f((prim.origin or (0.0, 0.0, 0.0))[n])) <= _tolerance(prim)


def on_sheet(prim: Primitive, point: np.ndarray) -> bool:
    """Whether a local point lies on the sheet, edges included."""
    loops = [loop for loop in outline(prim) if len(loop) >= 3]
    if not loops or not in_plane(prim, point):
        return False
    tol = _tolerance(prim)
    u, v = _in_plane(prim, point)
    if any(_near(u, v, loop, tol) for loop in loops):
        return True
    hole = (prim.sheet_shape == SheetShape.RING and len(loops) > 1 and _inside(u, v, loops[1]))
    return _inside(u, v, loops[0]) and not hole


def runs_edge_to_edge(prim: Primitive, ends: np.ndarray) -> bool:
    """Whether the line crosses a rectangle edge to edge, making the whole rectangle the port."""
    if (prim.sheet_shape or SheetShape.RECTANGLE) != SheetShape.RECTANGLE:
        return False
    size = prim.sheet_size or (0.0, 0.0)
    half = (_f(size[0]) / 2, _f(size[1]) / 2)
    tol = 1e-6 * max(half[0], half[1], 1e-9)
    (u0, v0), (u1, v1) = _in_plane(prim, ends[0]), _in_plane(prim, ends[1])

    def across(a: float, b: float, h: float) -> bool:
        return abs(abs(a) - h) <= tol and abs(abs(b) - h) <= tol and a * b < 0

    return across(u0, u1, half[0]) or across(v0, v1, half[1])


# --- port geometry ---

def port_depth(port: Port, mesh_res: float) -> float:
    """Port depth along the sheet normal: explicit value, else a default for guided kinds, else 0."""
    explicit = _f(port.sheet_extrude_mm)
    if explicit:
        return explicit
    if port.kind in GUIDED_KINDS and mesh_res > 0:
        return DEFAULT_EXTRUDE_CELLS * mesh_res
    return 0.0


@dataclass
class PortPlacement:
    """A port's placed box. The order of `start` and `stop` carries polarity or direction of travel."""

    start: np.ndarray
    stop: np.ndarray
    ops: list
    sheet: Optional[Primitive] = None
    exc_axis: Optional[Axis] = None  # from the integration line
    prop_axis: Optional[Axis] = None     # guided kinds on a sheet: the sheet's normal, as placed
    line: Optional[np.ndarray] = None    # the integration line's ends, as placed, (2, 3)
    along_line: bool = False  # lumped source along the line itself, not the whole sheet


def _dominant(vector: np.ndarray) -> Tuple[int, float]:
    index = int(np.argmax(np.abs(vector)))
    return index, float(vector[index])


def _swap(start: np.ndarray, stop: np.ndarray, index: int) -> None:
    start[index], stop[index] = stop[index], start[index]


def port_geometry(project, port: Port, cs_by_name, mesh_res: float = 0.0) -> PortPlacement:
    """A port's box, taken from its sheet when it has one.

    A lumped port fills its sheet only when the line crosses a rectangle edge to edge; otherwise it is a source
    along the line. Lumped `stop` is the positive terminal; microstrip `start` is the strip.
    """
    from EMShrimulator.core.em.transforms import apply_ops, effective_ops

    sheet = sheet_by_name(project.primitives, port.sheet)
    if sheet is None:
        ops = effective_ops(port.coordinate_system, [], cs_by_name)
        p1 = np.array(port.p1, dtype=float)
        p2 = np.array(port.p2, dtype=float)
        if ops:
            moved = apply_ops(np.array([p1, p2]), ops)
            p1, p2 = moved[0], moved[1]
        return PortPlacement(p1, p2, ops)

    ops = effective_ops(sheet.coordinate_system, sheet.transforms, cs_by_name)

    def place(points: np.ndarray) -> np.ndarray:
        return apply_ops(points, ops) if ops else points

    n = normal_index(sheet.normal_axis or "z")
    pts = local_points(sheet)
    depth = port_depth(port, mesh_res)
    if depth:
        far = pts.copy()
        far[:, n] += depth
        pts = np.vstack([pts, far])
    pts = place(pts)
    placement = PortPlacement(pts.min(axis=0), pts.max(axis=0), ops, sheet)

    if port.kind in GUIDED_KINDS:
        centre = lift(sheet, [(0.0, 0.0)])[0]
        ahead = centre.copy()
        ahead[n] += 1.0 if depth >= 0 else -1.0
        index, toward = _dominant(np.diff(place(np.array([centre, ahead])), axis=0)[0])
        placement.prop_axis = "xyz"[index]  # type: ignore[assignment]
        if toward < 0:
            _swap(placement.start, placement.stop, index)

    local = line_ends(port) if port.kind in LINE_KINDS else None
    if local is not None:
        ends = place(local)
        placement.line = ends
        index, rise = _dominant(ends[1] - ends[0])
        if rise != 0:
            placement.exc_axis = "xyz"[index]  # type: ignore[assignment]
            if port.kind == PortKind.LUMPED and not runs_edge_to_edge(sheet, local):
                # Along a single axis: openEMS excites one axis.
                stop = ends[0].copy()
                stop[index] = ends[1][index]
                placement.start, placement.stop, placement.along_line = ends[0].copy(), stop, True
            elif (rise > 0) != (port.kind == PortKind.LUMPED):
                _swap(placement.start, placement.stop, index)
    return placement
