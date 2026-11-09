"""Where each coordinate system sits in global space.

Relative systems are based on another system; object systems on a point of an object, which they follow.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from EMShrimulator.core.em import snaps
from EMShrimulator.core.em.model import CoordinateSystem, ModelerProject, Primitive, PrimitiveKind
from EMShrimulator.core.em.transforms import chain_matrix, effective_ops, frame_matrix, local_matrix

ANCHORS = ("centre", "face", "edge", "corner")
ANCHOR_LABELS = {"centre": "object centre", "face": "face centre", "edge": "edge centre", "corner": "corner"}
SNAP_KIND = {"centre": snaps.CENTRE, "face": snaps.FACE, "edge": snaps.EDGE, "corner": snaps.CORNER}
ANCHOR_OF = {kind: anchor for anchor, kind in SNAP_KIND.items()}
MODE_LABELS = {"offset": "Offset", "rotated": "Rotated", "both": "Offset and rotated"}

# A face with a shorter mean normal than this is curved and offers no point.
CURVED = 0.5
CACHE_SIZE = 64


class PlacementError(ValueError):
    pass


# ------------------------------------------------------------------ an object's points

@dataclass
class Features:
    """An object's anchor points in its own coordinates, and what places those in global space."""

    points: np.ndarray     # (N, 3)
    kinds: List[str]       # an anchor name for each
    normals: np.ndarray    # (N, 3): a face's outward normal; zero for the other kinds
    lo: np.ndarray         # the object's own bounding box
    hi: np.ndarray
    placement: np.ndarray  # 4x4, own coordinates to global ones

    def fractions(self, points: np.ndarray) -> np.ndarray:
        """Points' positions within the object's box, 0..1 per side (0.5 along a side with no length)."""
        span = self.hi - self.lo
        real = span > float(span.max()) * 1e-6
        return np.where(real, (np.asarray(points, dtype=float) - self.lo) / np.where(real, span, 1.0), 0.5)

    def to_global(self, points: np.ndarray) -> np.ndarray:
        homo = np.column_stack([points, np.ones(len(points))])
        return (homo @ self.placement.T)[:, :3]

    def to_local(self, points: np.ndarray) -> np.ndarray:
        homo = np.column_stack([points, np.ones(len(points))])
        return (homo @ np.linalg.inv(self.placement).T)[:, :3]


def systems_of(prim: Primitive) -> Set[str]:
    """Every coordinate system an object's shape depends on: its own and its operands'."""
    found = {prim.coordinate_system} if prim.coordinate_system else set()
    for child in prim.children:
        found |= systems_of(child)
    return found


def geometry_key(prim: Primitive, cs_by_name: Dict[str, CoordinateSystem]) -> str:
    """Everything an object's placed shape depends on, as one string."""
    payload = prim.model_dump(mode="json", exclude={"name", "visible", "priority", "material", "size_mode",
                                                    "model", "color", "opacity"})
    names = sorted(systems_of(prim))
    placed = [np.round(frame_matrix(name, cs_by_name), 12).tolist() for name in names]
    return json.dumps([payload, names, placed], sort_keys=True)


_cache: "OrderedDict[str, Features]" = OrderedDict()
_cache_lock = threading.Lock()


def features(prim: Primitive, cs_by_name: Dict[str, CoordinateSystem]) -> Features:
    """An object's centre, flat face centres, edge centres and corners, in its own coordinates."""
    key = geometry_key(prim, cs_by_name)
    with _cache_lock:
        found = _cache.get(key)
        if found is not None:
            _cache.move_to_end(key)
            return found
    found = _measure(prim, cs_by_name)
    with _cache_lock:
        _cache[key] = found
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return found


def _measure(prim: Primitive, cs_by_name: Dict[str, CoordinateSystem]) -> Features:
    from EMShrimulator.core.em import csg, sheets

    placement = chain_matrix(effective_ops(prim.coordinate_system, prim.transforms, cs_by_name))
    groups: List[Tuple[str, Sequence[np.ndarray], Optional[np.ndarray]]] = []
    if prim.kind == PrimitiveKind.SHEET:
        loops = [sheets.lift(prim, loop) for loop in sheets.outline(prim) if loop]
        if not loops:
            raise PlacementError(f"'{prim.name}' has no outline")
        points = np.vstack(loops)
        targets = snaps.sheet_snaps(loops)
        normal = np.zeros(3)
        normal[sheets.normal_index(prim.normal_axis or "z")] = 1.0
        faces = targets.of_kind(snaps.FACE)
        groups += [("face", list(faces), np.tile(normal, (len(faces), 1))),
                   ("edge", list(targets.of_kind(snaps.EDGE)), None),
                   ("corner", list(targets.of_kind(snaps.CORNER)), None)]
    else:
        vertices, triangles = csg.mesh_of(prim, cs_by_name)
        homo = np.column_stack([vertices, np.ones(len(vertices))])
        points = (homo @ np.linalg.inv(placement).T)[:, :3]
        centres, normals, middles, corners = snaps.solid_features(points, triangles)
        flat = np.linalg.norm(normals, axis=1) >= CURVED
        groups += [("face", list(centres[flat]), normals[flat]), ("edge", middles, None), ("corner", corners, None)]
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) / 2
    out_points, kinds, out_normals = [], [], []
    for kind, found, normals in groups:
        for i, point in enumerate(found):
            out_points.append(point)
            kinds.append(kind)
            out_normals.append(normals[i] / np.linalg.norm(normals[i]) if normals is not None else np.zeros(3))
    # The centre, unless a face centre is already there.
    span = float(np.max(hi - lo)) or 1.0
    taken = [p for p, k in zip(out_points, kinds) if k == "face"]
    if not any(np.linalg.norm(p - centre) <= span * 1e-9 for p in taken):
        out_points.insert(0, centre)
        kinds.insert(0, "centre")
        out_normals.insert(0, np.zeros(3))
    return Features(np.array(out_points, dtype=float).reshape(-1, 3), kinds,
                    np.array(out_normals, dtype=float).reshape(-1, 3), lo, hi, placement)


def nearest(found: Features, anchor: str, at: Optional[Sequence[float]]) -> int:
    """Index of the `anchor` point nearest the box fractions `at`."""
    candidates = [i for i, kind in enumerate(found.kinds) if kind == anchor]
    if not candidates:
        raise PlacementError(f"it has no {ANCHOR_LABELS[anchor]}")
    if at is None or len(candidates) == 1:
        return candidates[0]
    spots = found.fractions(found.points[candidates])
    return candidates[int(np.argmin(np.linalg.norm(spots - np.asarray(at, dtype=float), axis=1)))]


def _object_axes(placement: np.ndarray) -> np.ndarray:
    """The object's own axes in global space as unit columns, orthonormal even if scaled."""
    x = placement[:3, 0] / np.linalg.norm(placement[:3, 0])
    y = placement[:3, 1] - x * float(x @ placement[:3, 1])
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _face_axes(normal: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Z out of the face; X along the object axis closest to the face."""
    z = normal / np.linalg.norm(normal)
    lean = np.round(np.abs(axes.T @ z), 9)  # rounded, or float noise would pick between equal axes
    x = axes[:, int(np.argmin(lean))]
    x = x - z * float(z @ x)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def anchor_matrix(found: Features, index: int) -> np.ndarray:
    """4x4 frame at one of an object's points, before the system's own offset or rotation."""
    P = found.placement
    axes = _object_axes(P)
    if found.kinds[index] == "face" and np.any(found.normals[index]):
        normal = np.linalg.inv(P[:3, :3]).T @ found.normals[index]
        axes = _face_axes(normal, axes)
    M = np.eye(4)
    M[:3, :3] = axes
    M[:3, 3] = found.to_global(found.points[index:index + 1])[0]
    return M


# ------------------------------------------------------------------ placing systems

def place(project: ModelerProject) -> None:
    """Set `placement` or `placement_error` on every coordinate system of a resolved project."""
    by_name = {cs.name: cs for cs in project.coordinate_systems}
    prims = {p.id: p for p in project.primitives}
    for cs in project.coordinate_systems:
        cs.placement, cs.placement_error = None, ""
    # Placement stack for loop detection: ("cs", name) or ("object", name).
    stack: List[Tuple[str, str]] = []

    def fail(cs: CoordinateSystem, message: str) -> PlacementError:
        if not cs.placement_error:
            cs.placement_error = message
        return PlacementError(message)

    def frame(name: str) -> np.ndarray:
        cs = by_name[name]
        if cs.placement is not None:
            return np.array(cs.placement, dtype=float).reshape(4, 4)
        if cs.placement_error:
            raise PlacementError(cs.placement_error)
        if ("cs", name) in stack:
            loop = stack[stack.index(("cs", name)):]
            message = "it depends on itself: " + " → ".join(f"'{n}'" for _, n in loop) + f" → '{name}'"
            for kind, member in loop:
                if kind == "cs":
                    fail(by_name[member], message)
            raise PlacementError(message)
        stack.append(("cs", name))
        try:
            base = _base(cs)
        finally:
            stack.pop()
        if cs.placement_error:  # a member of a loop, told so while the loop unwound
            raise PlacementError(cs.placement_error)
        M = np.round(base @ local_matrix(cs), 12) + 0.0
        cs.placement = M.ravel().tolist()
        return M

    def _base(cs: CoordinateSystem) -> np.ndarray:
        if cs.kind != "object":
            if not cs.parent:
                return np.eye(4)
            if cs.parent not in by_name:
                raise fail(cs, f"the system it is based on, '{cs.parent}', does not exist")
            try:
                return frame(cs.parent)
            except PlacementError:
                raise fail(cs, f"the system it is based on, '{cs.parent}', cannot be placed") from None
        prim = prims.get(cs.object_id or "")
        if prim is None:
            raise fail(cs, "the object it sits on no longer exists")
        stack.append(("object", prim.name))
        try:
            for dependency in sorted(systems_of(prim)):
                if dependency not in by_name:
                    raise fail(cs, f"'{prim.name}' is placed in '{dependency}', which does not exist")
                try:
                    frame(dependency)
                except PlacementError:
                    raise fail(cs, f"'{prim.name}' is placed in '{dependency}', which cannot be placed") from None
        finally:
            stack.pop()
        try:
            found = features(prim, by_name)
            return anchor_matrix(found, nearest(found, cs.anchor, cs.anchor_at))
        except PlacementError as exc:
            raise fail(cs, f"'{prim.name}': {exc}") from None
        except Exception as exc:  # noqa: BLE001 - a shape that cannot be built: say why, place the rest
            raise fail(cs, f"'{prim.name}' could not be measured: {exc}") from None

    for cs in project.coordinate_systems:
        try:
            frame(cs.name)
        except PlacementError:
            pass


def ensure_placed(project: ModelerProject) -> None:
    """Place any systems params.resolve did not."""
    if any(cs.placement is None and not cs.placement_error for cs in project.coordinate_systems):
        place(project)


# ------------------------------------------------------------------ picking a point

def anchor_targets(project: ModelerProject, prim_id: str) -> Tuple[snaps.SnapSet, Features]:
    """An object's anchor points in global space, as snap targets for the 3D view."""
    prim = next((p for p in project.primitives if p.id == prim_id), None)
    if prim is None:
        raise PlacementError("That object no longer exists.")
    found = features(prim, {cs.name: cs for cs in project.coordinate_systems})
    targets = snaps.SnapSet.of(found.to_global(found.points), [SNAP_KIND[k] for k in found.kinds])
    return targets, found


def anchor_at(found: Features, point: np.ndarray) -> Tuple[float, float, float]:
    """A clicked point (global) as the fractions an object system keeps of it."""
    spot = found.fractions(found.to_local(np.asarray(point, dtype=float).reshape(1, 3)))[0]
    return tuple(round(float(v), 9) + 0.0 for v in spot)  # type: ignore[return-value]


def describe_anchor(cs: CoordinateSystem) -> str:
    """Which point an object system sits on, in words, e.g. "face centre (+Z)"."""
    label = ANCHOR_LABELS.get(cs.anchor, cs.anchor)
    if cs.anchor == "centre" or cs.anchor_at is None:
        return label
    sides = [("+" if v > 0.5 else "−") + axis for axis, v in zip("XYZ", cs.anchor_at) if abs(v - 0.5) > 0.25]
    return f"{label} ({' '.join(sides)})" if sides else label
