"""Boolean solids via manifold3d, emitted to CSXCAD as closed polyhedra (CSXCAD has no CSG)."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from EMShrimulator.core.em.model import BooleanOp, CoordinateSystem, Primitive, PrimitiveKind, TransformOp
from EMShrimulator.core.em.sheets import UV_AXES, default_rot_axis
from EMShrimulator.core.em.transforms import op_matrix, resolve_chain

# Optional: only booleans need manifold3d.
_MANIFOLD_ERROR: str | None = None
try:  # pragma: no cover - import guard
    import manifold3d as _m
except Exception as exc:  # pragma: no cover - import guard
    _m = None
    _MANIFOLD_ERROR = str(exc)


class CSGUnavailable(RuntimeError):
    pass


def available() -> bool:
    return _m is not None


def _require():
    if _m is None:
        raise CSGUnavailable(
            "Boolean (union/intersect/subtract) primitives need the 'manifold3d' package: "
            f"pip install manifold3d  (import failed: {_MANIFOLD_ERROR})"
        )
    return _m


# --- primitive -> manifold ---

CHORD_ERROR_RELATIVE = 0.005  # of the radius: 32 facets, whatever the size
CHORD_ERROR_MM = 0.01  # absolute, so larger curves get more facets


def _circular_segments(radius: float) -> int:
    """Facet count for a curved surface: chord error within 0.5% of the radius and 10 µm, clamped to 32..96."""
    if radius <= 0:
        return 32
    tolerance = min(CHORD_ERROR_RELATIVE * radius, CHORD_ERROR_MM)
    n = int(math.ceil(math.pi / math.acos(max(-1.0, min(1.0, 1.0 - tolerance / radius)))))
    return max(32, min(96, n))


def _box(p1: Sequence[float], p2: Sequence[float]):
    m = _require()
    lo = np.minimum(np.array(p1, dtype=float), np.array(p2, dtype=float))
    hi = np.maximum(np.array(p1, dtype=float), np.array(p2, dtype=float))
    size = np.maximum(hi - lo, 1e-9)
    return m.Manifold.cube(size).translate(lo)


def _align_z_to(a: np.ndarray, b: np.ndarray):
    """Rotation (as a manifold transform) taking +Z onto the a->b direction."""
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        return None, 0.0
    d = d / length
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(z, d))
    if s < 1e-12:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        axis = axis / s
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R, length


def _cylinder(p1, p2, radius: float, inner_radius: float = 0.0):
    m = _require()
    a, b = np.array(p1, dtype=float), np.array(p2, dtype=float)
    R, length = _align_z_to(a, b)
    if R is None:
        raise ValueError("cylinder end points coincide")
    segs = _circular_segments(radius)
    solid = m.Manifold.cylinder(length, radius, radius, segs)
    if inner_radius > 0:
        solid = solid - m.Manifold.cylinder(length * 3, inner_radius, inner_radius, segs).translate([0, 0, -length])
    mat = np.zeros((3, 4))
    mat[:, :3] = R
    mat[:, 3] = a
    return solid.transform(mat)


def _sphere(center, radius: float, inner_radius: float = 0.0):
    m = _require()
    segs = _circular_segments(radius)
    solid = m.Manifold.sphere(radius, segs)
    if inner_radius > 0:
        solid = solid - m.Manifold.sphere(inner_radius, segs)
    return solid.translate(np.array(center, dtype=float))


def _extruded_polygon(points: Sequence[Sequence[float]], normal_axis: str, elevation: float, length: float):
    """Extrude a 2D outline along `normal_axis`, using CSXCAD's AddLinPoly (u, v) convention."""
    m = _require()
    poly = np.array(points, dtype=float)
    depth = abs(length) if abs(length) > 1e-12 else 1e-6
    solid = m.Manifold.extrude(m.CrossSection([poly.tolist()]), depth)
    base = min(elevation, elevation + length)
    if normal_axis == "z":
        return solid.translate([0.0, 0.0, base])
    if normal_axis == "x":
        # (u, v) = (y, z), extrude along +x
        mat = np.array([[0.0, 0.0, 1.0, base], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        return solid.transform(mat)
    # normal_axis == "y": (u, v) = (z, x), extrude along +y
    mat = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, base], [1.0, 0.0, 0.0, 0.0]])
    return solid.transform(mat)


# --- revolve (ROT_POLY), in CSXCAD's CSPrimRotPoly frame ---
# The outline's coordinate along rot_axis is the height, the other in-plane one the radius.

def rot_poly_axes(normal_axis: str, rot_axis: str) -> Tuple[int, int, int]:
    """(normal, axial, radial) world axis indices of a revolve."""
    n, r = "xyz".index(normal_axis), "xyz".index(rot_axis)
    if n == r:
        raise ValueError("a revolve's sweep axis must lie in its outline's plane, not along the normal axis")
    return n, r, 3 - n - r


def rot_poly_profile(points, normal_axis: str, rot_axis: str) -> np.ndarray:
    """The outline as (radius, height) rows."""
    _, r, t = rot_poly_axes(normal_axis, rot_axis)
    iu, iv = UV_AXES[normal_axis]
    poly = np.array(points, dtype=float).reshape(-1, 2)
    world = {iu: poly[:, 0], iv: poly[:, 1]}
    return np.column_stack([world[t], world[r]])


def _counter_clockwise(poly: np.ndarray) -> np.ndarray:
    """Make an outline counter-clockwise, as CrossSection needs."""
    x, y = poly[:, 0], poly[:, 1]
    signed_area = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return poly[::-1] if signed_area < 0 else poly


def _rot_poly(points, normal_axis: str, elevation: float, rot_axis: str, a0: float, a1: float):
    m = _require()
    n, r, t = rot_poly_axes(normal_axis, rot_axis)
    profile = rot_poly_profile(points, normal_axis, rot_axis)
    start, stop = sorted((float(a0), float(a1)))
    sweep = min(stop - start, 360.0)
    segments = _circular_segments(float(np.abs(profile[:, 0]).max(initial=0.0)))
    # revolve() keeps only x >= 0, so the part across the axis is mirrored and swept from start + 180.
    halves = []
    for side, turn in ((1.0, start), (-1.0, start + 180.0)):
        part = profile * [side, 1.0]
        if part[:, 0].max() > 0:
            section = m.CrossSection([_counter_clockwise(part).tolist()])
            halves.append(m.Manifold.revolve(section, segments, sweep).rotate([0.0, 0.0, turn]))
    if not halves:
        raise ValueError("a revolve's outline has no extent away from its sweep axis")
    solid = halves[0] if len(halves) == 1 else m.Manifold.batch_boolean(halves, m.OpType.Add)
    # Then map revolve's x -> radial, z -> axial.
    e_t, e_r = np.eye(3)[t], np.eye(3)[r]
    mat = np.zeros((3, 4))
    mat[:, 0], mat[:, 1], mat[:, 2] = e_t, np.cross(e_r, e_t), e_r
    mat[n, 3] = elevation
    return solid.transform(mat)


def _wire(path: Sequence[Sequence[float]], radius: float):
    """A polyline swept with a round cross-section: capsules per segment, unioned."""
    m = _require()
    pts = [np.array(p, dtype=float) for p in path]
    if len(pts) < 2:
        raise ValueError("wire needs at least 2 points")
    parts = []
    segs = _circular_segments(radius)
    for a, b in zip(pts[:-1], pts[1:]):
        if np.linalg.norm(b - a) < 1e-12:
            continue
        parts.append(_cylinder(a, b, radius))
    for p in pts[1:-1]:  # round joints
        parts.append(m.Manifold.sphere(radius, segs).translate(p))
    if not parts:
        raise ValueError("wire has zero length")
    return m.Manifold.batch_boolean(parts, m.OpType.Add)


def _transform_matrix(ops: List[TransformOp]) -> np.ndarray:
    """4x4 affine for a transform chain, applied in order (first op first)."""
    M = np.eye(4)
    for op in ops:
        M = op_matrix(op) @ M
    return M


def _apply_ops(solid, ops: List[TransformOp]):
    if not ops:
        return solid
    M = _transform_matrix(ops)
    return solid.transform(M[:3, :])


def to_manifold(prim: Primitive, cs_by_name: Dict[str, CoordinateSystem], apply_cs: bool = True):
    """Evaluate one primitive (recursively, for booleans) into a manifold solid."""
    m = _require()
    k = prim.kind

    if k == PrimitiveKind.BOOLEAN:
        if not prim.children:
            raise ValueError(f"boolean '{prim.name}' has no operands")
        parts = [to_manifold(c, cs_by_name, apply_cs=True) for c in prim.children]
        op = prim.boolean_op or BooleanOp.UNION
        if op == BooleanOp.UNION:
            solid = m.Manifold.batch_boolean(parts, m.OpType.Add)
        elif op == BooleanOp.INTERSECT:
            solid = m.Manifold.batch_boolean(parts, m.OpType.Intersect)
        elif op == BooleanOp.SUBTRACT:
            solid = parts[0]
            for p in parts[1:]:
                solid = solid - p
        elif op == BooleanOp.XOR:
            union = m.Manifold.batch_boolean(parts, m.OpType.Add)
            inter = m.Manifold.batch_boolean(parts, m.OpType.Intersect)
            solid = union - inter
        elif op == BooleanOp.HULL:
            solid = m.Manifold.batch_hull(parts)
        else:
            raise ValueError(f"unknown boolean op {op}")
    elif k == PrimitiveKind.BOX:
        solid = _box(prim.p1, prim.p2)
    elif k == PrimitiveKind.CYLINDER:
        solid = _cylinder(prim.p1, prim.p2, prim.radius or 0.0)
    elif k == PrimitiveKind.CYLINDRICAL_SHELL:
        r, w = prim.radius or 0.0, prim.shell_width or 0.0
        solid = _cylinder(prim.p1, prim.p2, r + w / 2, max(r - w / 2, 0.0))
    elif k == PrimitiveKind.SPHERE:
        solid = _sphere(prim.center or (0, 0, 0), prim.radius or 0.0)
    elif k == PrimitiveKind.SPHERICAL_SHELL:
        r, w = prim.radius or 0.0, prim.shell_width or 0.0
        solid = _sphere(prim.center or (0, 0, 0), r + w / 2, max(r - w / 2, 0.0))
    elif k == PrimitiveKind.POLYGON:
        solid = _extruded_polygon(
            prim.points or [], prim.normal_axis or "z", prim.elevation or 0.0,
            prim.extrude_length if prim.extrude_length else 1e-6,
        )
    elif k == PrimitiveKind.ROT_POLY:
        normal = prim.normal_axis or "z"
        solid = _rot_poly(
            prim.points or [], normal, prim.elevation or 0.0,
            prim.rot_axis or default_rot_axis(normal), prim.angle_start_deg or 0.0,
            prim.angle_stop_deg if prim.angle_stop_deg is not None else 360.0,
        )
    elif k in (PrimitiveKind.WIRE, PrimitiveKind.CURVE):
        path = prim.path or ([prim.p1, prim.p2] if prim.p1 and prim.p2 else [])
        solid = _wire(path, prim.radius or 0.1)
    else:
        raise ValueError(f"primitive kind {k} cannot be used in a boolean operation")

    ops: List[TransformOp] = []
    if apply_cs and prim.coordinate_system:
        ops.extend(resolve_chain(prim.coordinate_system, cs_by_name))
    ops.extend(prim.transforms)
    return _apply_ops(solid, ops)


def mesh_of(prim: Primitive, cs_by_name: Dict[str, CoordinateSystem]) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices Nx3, triangles Mx3) of a fully-evaluated primitive."""
    solid = to_manifold(prim, cs_by_name)
    if solid.is_empty():
        raise ValueError(f"'{prim.name}' evaluates to an empty solid — check the operand order/overlap")
    # Doubles, to avoid manifold3d's float error (1.6 mm becoming 1.60000002).
    mesh = solid.to_mesh64() if hasattr(solid, "to_mesh64") else solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=float)[:, :3]
    tris = np.asarray(mesh.tri_verts, dtype=np.int64)
    return verts, tris


def add_to_property(prop, prim: Primitive, cs_by_name: Dict[str, CoordinateSystem], priority: int):
    """Evaluate `prim` and attach it to a CSXCAD property as a polyhedron."""
    verts, tris = mesh_of(prim, cs_by_name)
    poly = prop.AddPolyhedron(priority=priority)
    for v in verts:
        poly.AddVertex(float(v[0]), float(v[1]), float(v[2]))
    for t in tris:
        poly.AddFace([int(t[0]), int(t[1]), int(t[2])])
    ok, msg = poly.Update()
    if not ok:
        raise ValueError(f"CSXCAD rejected the polyhedron for '{prim.name}': {msg}")
    return poly, verts, tris
