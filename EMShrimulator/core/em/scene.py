"""What the 3D viewport draws, keyed by object id. Tessellation is memoised per object; no openEMS imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Optional, Tuple

import numpy as np

from EMShrimulator.core.em import csg, sheets
from EMShrimulator.core.em.frames import geometry_key
from EMShrimulator.core.em.model import Material, ModelerProject, Primitive, PrimitiveKind
from EMShrimulator.core.em.transforms import apply_ops, effective_ops, frame_matrix

# Default look when neither the object nor its material sets one.
SOLID_LOOK = ("#8899aa", 0.85)  # Material's own defaults
SHEET_OPACITY = 0.6             # a sheet whose material is missing
CONSTRUCTION_OPACITY = 0.18


def appearance(prim: Primitive, materials: Dict[str, Material]) -> Tuple[Optional[str], float]:
    """An object's colour and opacity: its own, else its material's. None means the theme's accent."""
    material = materials.get(prim.material or "")
    if material is not None:
        color, opacity = material.color, float(material.opacity)
    elif prim.kind == PrimitiveKind.SHEET:
        color, opacity = None, CONSTRUCTION_OPACITY if sheets.is_construction(prim) else SHEET_OPACITY
    else:
        color, opacity = SOLID_LOOK
    return prim.color or color, float(prim.opacity) if prim.opacity is not None else opacity


@dataclass
class SolidMesh:
    id: str
    name: str
    material: str
    vertices: np.ndarray    # (N, 3) float, mm
    triangles: np.ndarray   # (M, 3) int
    color: Optional[str] = None  # see appearance()
    opacity: float = SOLID_LOOK[1]
    model: bool = True


@dataclass
class SheetOutline:
    id: str
    name: str
    material: str
    construction: bool
    loops: List[np.ndarray]  # each (K, 3), closed implicitly
    normal_axis: str
    center: np.ndarray
    color: Optional[str] = None  # see appearance()
    opacity: float = CONSTRUCTION_OPACITY
    model: bool = True


@dataclass
class PortBox:
    id: str
    name: str
    number: int
    kind: str
    excite: bool
    p1: np.ndarray
    p2: np.ndarray
    line: Optional[np.ndarray] = None  # the integration line, negative end first, (2, 3)
    on_sheet: bool = False
    awaiting_line: bool = False  # its line isn't drawn yet: show only its name


@dataclass
class Scene:
    solids: Dict[str, SolidMesh] = field(default_factory=dict)
    sheets: Dict[str, SheetOutline] = field(default_factory=dict)
    ports: Dict[str, PortBox] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    frame: Optional["WorkingFrame"] = None  # the working coordinate system; None = global
    frames: Dict[str, "WorkingFrame"] = field(default_factory=dict)  # every coordinate system, by name


def _cache_key(prim: Primitive, cs_by_name) -> str:
    """A key of everything a primitive's shape depends on."""
    return geometry_key(prim, cs_by_name)


def build_scene(resolved: ModelerProject, cache: Optional[MutableMapping[str, Tuple[np.ndarray, np.ndarray]]] = None,
                mesh_resolution_mm: float = 0.0) -> Scene:
    """Tessellate a resolved project, reusing `cache`. Per-object failures are recorded, not raised."""
    cache = {} if cache is None else cache
    cs_by_name = {cs.name: cs for cs in resolved.coordinate_systems}
    materials = {m.name: m for m in resolved.materials}
    scene = Scene()
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    live_keys = set()

    for prim in resolved.primitives:
        color, opacity = appearance(prim, materials)
        try:
            if prim.kind == PrimitiveKind.SHEET:
                outline = _sheet(prim, cs_by_name)
                outline.color, outline.opacity, outline.model = color, opacity, prim.model
                scene.sheets[prim.id] = outline
                points = np.vstack(outline.loops) if outline.loops else outline.center[None, :]
            else:
                key = _cache_key(prim, cs_by_name)
                live_keys.add(key)
                if key not in cache:
                    cache[key] = csg.mesh_of(prim, cs_by_name)
                vertices, triangles = cache[key]
                scene.solids[prim.id] = SolidMesh(prim.id, prim.name, prim.material, vertices, triangles,
                                                  color, opacity, prim.model)
                points = vertices
        except Exception as exc:  # noqa: BLE001 - one bad object must not blank the viewport
            scene.errors[prim.id] = str(exc)
            continue
        if points.size:
            lo = np.minimum(lo, points.min(axis=0))
            hi = np.maximum(hi, points.max(axis=0))

    for stale in [k for k in cache if k not in live_keys]:
        del cache[stale]

    for port in resolved.ports:
        try:
            placement = sheets.port_geometry(resolved, port, cs_by_name, mesh_resolution_mm)
        except Exception as exc:  # noqa: BLE001
            scene.errors[port.id] = str(exc)
            continue
        scene.ports[port.id] = PortBox(
            port.id, port.name, port.number, port.kind.value, port.excite,
            np.asarray(placement.start, dtype=float), np.asarray(placement.stop, dtype=float),
            line=placement.line, on_sheet=placement.sheet is not None,
            awaiting_line=placement.sheet is not None and port.kind in sheets.LINE_KINDS and placement.line is None,
        )

    if np.isfinite(lo).all():
        scene.bounds = (lo, hi)
    for cs in resolved.coordinate_systems:
        scene.frames[cs.name] = frame_of(cs.name, cs_by_name)
        if cs.placement_error:
            scene.errors[f"cs:{cs.name}"] = cs.placement_error
    try:
        scene.frame = working_frame(resolved)
    except Exception as exc:  # noqa: BLE001 - a bad working system falls back to global
        scene.errors["working_cs"] = str(exc)
    return scene


def _sheet(prim: Primitive, cs_by_name) -> SheetOutline:
    ops = effective_ops(prim.coordinate_system, prim.transforms, cs_by_name)
    loops = []
    for loop in sheets.outline(prim):
        if not loop:
            continue
        points = sheets.lift(prim, loop)
        loops.append(apply_ops(points, ops) if ops else points)
    center = sheets.lift(prim, [(0.0, 0.0)])
    if ops:
        center = apply_ops(center, ops)
    return SheetOutline(
        id=prim.id, name=prim.name, material=prim.material or "",
        construction=sheets.is_construction(prim), loops=loops,
        normal_axis=prim.normal_axis or "z", center=center[0],
    )


def mesh_planes(lines: Dict[str, List[float]], domain_min, domain_max) -> Dict[str, np.ndarray]:
    """Grid lines on the three centre planes of the domain, as segment pairs."""
    lo = np.asarray(domain_min, dtype=float)
    hi = np.asarray(domain_max, dtype=float)
    mid = (lo + hi) / 2
    axes = {"x": 0, "y": 1, "z": 2}
    out: Dict[str, np.ndarray] = {}
    for plane_axis, fixed in axes.items():
        parts = []
        others = [a for a in axes if a != plane_axis]
        for line_axis in others:
            across = axes[next(a for a in others if a != line_axis)]
            values = np.asarray(lines.get(line_axis, []), dtype=float)
            segments = np.empty((len(values), 2, 3))
            segments[:, :, fixed] = mid[fixed]
            segments[:, :, axes[line_axis]] = values[:, None]
            segments[:, 0, across] = lo[across]
            segments[:, 1, across] = hi[across]
            parts.append(segments)
        out[plane_axis] = np.concatenate(parts)
    return out


@dataclass
class WorkingFrame:
    """The working coordinate system's origin and unit axes (rows) in global space."""

    origin: np.ndarray
    axes: np.ndarray  # (3, 3): x, y, z
    name: str = ""    # the coordinate system's name; "" for global

    @classmethod
    def global_frame(cls) -> "WorkingFrame":
        return cls(np.zeros(3), np.eye(3))


def frame_of(name: Optional[str], cs_by_name) -> WorkingFrame:
    """A coordinate system as placed, the same way the objects in it are; global for None."""
    if not name:
        return WorkingFrame.global_frame()
    M = frame_matrix(name, cs_by_name)
    axes = M[:3, :3].T
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    if not np.all(norms > 1e-12):
        return WorkingFrame.global_frame()
    return WorkingFrame(M[:3, 3].copy(), axes / norms, name)


def triads(scene: Scene, selected: Optional[str]) -> List[Tuple[WorkingFrame, str, str]]:
    """(frame, style, label) of each coordinate system to draw: the working one, a small global one and the
    one being edited (`selected`). None while everything is global."""
    working = scene.frame.name if scene.frame is not None else ""
    out: List[Tuple[WorkingFrame, str, str]] = []
    if working:
        out.append((scene.frame, "working", working))
    if selected is not None and selected != working:
        frame = scene.frames.get(selected) if selected else WorkingFrame.global_frame()
        if frame is not None:
            out.append((frame, "selected", selected or "Global"))
    if working and selected != "":
        out.append((WorkingFrame.global_frame(), "global", ""))
    return out


def working_frame(resolved: ModelerProject) -> WorkingFrame:
    """The resolved and placed working coordinate system."""
    return frame_of(resolved.working_cs, {cs.name: cs for cs in resolved.coordinate_systems})


def _nice_step(span: float, target_cells: int = 12) -> float:
    """1, 2 or 5 times a power of ten, so the span holds about `target_cells` cells."""
    raw = max(span, 1e-6) / target_cells
    power = 10 ** np.floor(np.log10(raw))
    for mantissa in (1, 2, 5, 10):
        if mantissa * power >= raw:
            return float(mantissa * power)
    return float(10 * power)


def _grid_extent(frame: WorkingFrame, bounds: Optional[Tuple[np.ndarray, np.ndarray]]
                 ) -> Tuple[float, float, float, float, float]:
    """(u_lo, u_hi, v_lo, v_hi, span) the grid must cover."""
    if bounds is not None:
        lo, hi = bounds
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        local = (corners - frame.origin) @ frame.axes.T
        u_lo, v_lo = local[:, 0].min(), local[:, 1].min()
        u_hi, v_hi = local[:, 0].max(), local[:, 1].max()
    else:
        u_lo = v_lo = -10.0
        u_hi = v_hi = 10.0
    u_lo, v_lo = min(u_lo, 0.0), min(v_lo, 0.0)
    u_hi, v_hi = max(u_hi, 0.0), max(v_hi, 0.0)
    return u_lo, u_hi, v_lo, v_hi, max(u_hi - u_lo, v_hi - v_lo, 1e-3)


def grid_step(frame: WorkingFrame, bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> float:
    """The grid's minor line spacing."""
    return _nice_step(_grid_extent(frame, bounds)[4])


def working_grid(frame: WorkingFrame, bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, object]:
    """The grid on the frame's XY plane as segment pairs, covering the origin and the design's footprint.

    Every fifth line is major; the lines through the origin are returned as the X and Y axes.
    """
    u_lo, u_hi, v_lo, v_hi, span = _grid_extent(frame, bounds)
    step = _nice_step(span)
    major = step * 5
    pad = span * 0.25
    u0, u1 = np.floor((u_lo - pad) / major) * major, np.ceil((u_hi + pad) / major) * major
    v0, v1 = np.floor((v_lo - pad) / major) * major, np.ceil((v_hi + pad) / major) * major

    def lift(u, v) -> np.ndarray:
        return frame.origin + np.multiply.outer(u, frame.axes[0]) + np.multiply.outer(v, frame.axes[1])

    def lines(ticks: np.ndarray, along_u: bool) -> np.ndarray:
        if along_u:  # lines of constant v, running along u
            return np.stack([lift(np.full_like(ticks, u0), ticks), lift(np.full_like(ticks, u1), ticks)], axis=1)
        return np.stack([lift(ticks, np.full_like(ticks, v0)), lift(ticks, np.full_like(ticks, v1))], axis=1)

    out: Dict[str, object] = {"step": step, "major_step": major}
    for name, keep in (("minor", lambda t: (np.abs(np.round(t / major) * major - t) > step * 1e-3)),
                       ("major", lambda t: (np.abs(np.round(t / major) * major - t) <= step * 1e-3)
                        & (np.abs(t) > step * 1e-3))):
        u_ticks = np.arange(u0, u1 + step * 0.5, step)
        v_ticks = np.arange(v0, v1 + step * 0.5, step)
        parts = [lines(u_ticks[keep(u_ticks)], along_u=False), lines(v_ticks[keep(v_ticks)], along_u=True)]
        out[name] = np.concatenate([p for p in parts if len(p)]) if any(len(p) for p in parts) else np.empty((0, 2, 3))
    out["x_axis"] = np.array([[lift(u0, 0.0), lift(u1, 0.0)]])
    out["y_axis"] = np.array([[lift(0.0, v0), lift(0.0, v1)]])
    out["z_axis"] = np.array([[frame.origin, frame.origin + frame.axes[2] * major]])
    out["origin"] = frame.origin
    return out
