"""Transform math for coordinate-system frames, global bounding boxes and CSG.

CSXCAD's GetBoundBox ignores transforms, so boxes are computed here.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from EMShrimulator.core.em.model import CoordinateSystem, TransformOp

Point = np.ndarray


def _rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # z


def op_matrix(op: TransformOp) -> np.ndarray:
    """4x4 homogeneous matrix for one transform op."""
    M = np.eye(4)
    if op.kind == "translate" and op.translate:
        M[:3, 3] = np.array(op.translate, dtype=float)
    elif op.kind in ("rotate_axis", "rotate_origin") and op.axis and op.angle_deg is not None:
        M[:3, :3] = _rotation_matrix(op.axis, op.angle_deg)
    elif op.kind == "scale" and op.scale:
        M[:3, :3] = np.diag(np.array(op.scale, dtype=float))
    return M


def chain_matrix(ops: List[TransformOp]) -> np.ndarray:
    """4x4 matrix for a whole chain, ops applied in list order."""
    M = np.eye(4)
    for op in ops:
        M = op_matrix(op) @ M
    return M


def apply_ops(points: np.ndarray, ops: List[TransformOp]) -> np.ndarray:
    """points: (N,3) array. Applies ops in order, each acting on the result of the last."""
    if not ops:
        return points
    M = chain_matrix(ops)
    homo = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homo @ M.T)[:, :3]


def unapply_ops(points: np.ndarray, ops: List[TransformOp]) -> np.ndarray:
    """apply_ops undone: placed points back into the frame the ops start from."""
    if not ops:
        return points
    M = np.linalg.inv(chain_matrix(ops))
    homo = np.hstack([points, np.ones((points.shape[0], 1))])
    return (homo @ M.T)[:, :3]


# ------------------------------------------------------------------ frames

def _number(value) -> float:
    """A resolved value as a float; an expression not resolved yet reads as 0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def rotation(angles) -> np.ndarray:
    """3x3 rotation about X, then Y, then Z (degrees)."""
    ax, ay, az = (_number(a) for a in angles)
    return _rotation_matrix("z", az) @ _rotation_matrix("y", ay) @ _rotation_matrix("x", ax)


def local_matrix(cs: CoordinateSystem) -> np.ndarray:
    """4x4 from a system's coordinates to its base's."""
    M = np.eye(4)
    if cs.mode in ("rotated", "both"):
        M[:3, :3] = rotation(cs.angles)
    if cs.mode in ("offset", "both"):
        M[:3, 3] = [_number(v) for v in cs.origin]
    return M


def euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    """The angles (degrees) that `rotation` turns into R: about X, then Y, then Z."""
    if abs(R[2, 0]) < 1 - 1e-12:
        ay = np.arcsin(-R[2, 0])
        ax = np.arctan2(R[2, 1], R[2, 2])
        az = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: Z takes none of the turn
        ay = np.pi / 2 if R[2, 0] < 0 else -np.pi / 2
        ax = np.arctan2(R[0, 1], R[1, 1]) if R[2, 0] < 0 else np.arctan2(-R[0, 1], R[1, 1])
        az = 0.0
    return tuple(_tidy_angle(float(np.degrees(a))) for a in (ax, ay, az))  # type: ignore[return-value]


def _tidy_angle(degrees: float) -> float:
    degrees = round(degrees, 9) + 0.0
    return 180.0 if degrees == -180.0 else degrees


def offset_and_angles(M: np.ndarray) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """A frame as the offset and the angles that place it, numbers as they would be typed."""
    offset = tuple(round(float(v), 9) + 0.0 for v in M[:3, 3])
    return offset, euler_xyz(M[:3, :3])  # type: ignore[return-value]


def ops_of(M: np.ndarray) -> List[TransformOp]:
    """A rigid frame as rotations about X, Y and Z, then a translation."""
    offset, angles = offset_and_angles(M)
    ops = [TransformOp(kind="rotate_axis", axis=axis, angle_deg=angle)
           for axis, angle in zip("xyz", angles) if angle]
    if any(offset):
        ops.append(TransformOp(kind="translate", translate=offset))
    return ops


def frame_matrix(name: str | None, by_name: Dict[str, CoordinateSystem],
                 _seen: frozenset[str] = frozenset()) -> np.ndarray:
    """4x4 from a coordinate system to global; identity for global. Unplaceable systems sit at the origin."""
    cs = by_name.get(name) if name else None
    if cs is None or name in _seen:
        return np.eye(4)
    if cs.placement is not None:
        return np.array(cs.placement, dtype=float).reshape(4, 4)
    if cs.kind == "object":
        return np.eye(4)
    base = frame_matrix(cs.parent, by_name, _seen | {name}) if cs.parent else np.eye(4)
    return base @ local_matrix(cs)


def resolve_chain(name: str, by_name: Dict[str, CoordinateSystem]) -> List[TransformOp]:
    """A coordinate system's placement as a transform chain."""
    return ops_of(frame_matrix(name, by_name)) if name in by_name else []


def effective_ops(cs_name: str | None, own: List[TransformOp], by_name: Dict[str, CoordinateSystem]) -> List[TransformOp]:
    """A primitive's full transform chain: its coordinate system, then its own ops."""
    ops: List[TransformOp] = []
    if cs_name:
        ops.extend(resolve_chain(cs_name, by_name))
    ops.extend(own)
    return ops


def transform_points(points: np.ndarray, cs_name: str | None, by_name: Dict[str, CoordinateSystem]) -> np.ndarray:
    if not cs_name:
        return points
    return apply_ops(points, resolve_chain(cs_name, by_name))


def aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)


# ------------------------------------------------------------------ old saves

def _sum(values) -> object:
    """Terms added up: a number while they all are, else an expression."""
    terms = [v for v in values if not (isinstance(v, (int, float)) and v == 0)]
    if all(isinstance(v, (int, float)) for v in terms):
        return float(sum(terms))
    if len(terms) == 1:
        return terms[0]
    return " + ".join((t if t.isidentifier() else f"({t})") if isinstance(t, str) else f"{float(t):g}"
                      for t in terms)


def legacy_offset_rotation(ops: List[TransformOp]) -> dict:
    """mode, origin and angles for a system saved as an older global transform chain."""
    ops = [op for op in ops if op.kind != "scale"]
    rotations = [op for op in ops if op.kind in ("rotate_axis", "rotate_origin")]
    translations = [op for op in ops if op.kind == "translate" and op.translate]
    numeric = all(isinstance(v, (int, float)) for op in translations for v in op.translate) and all(
        isinstance(op.angle_deg, (int, float)) for op in rotations)
    order = [op.axis for op in rotations]
    symbolic = (ops[:len(rotations)] == rotations and order == sorted(order) and len(set(order)) == len(order))
    if numeric or not symbolic:
        offset, angles = offset_and_angles(chain_matrix(ops))
    else:
        offset = tuple(_sum(op.translate[i] for op in translations) for i in range(3))
        by_axis = {op.axis: op.angle_deg for op in rotations}
        angles = tuple(by_axis.get(axis, 0.0) for axis in "xyz")
    turned = any(not (isinstance(a, (int, float)) and a == 0) for a in angles)
    moved = any(not (isinstance(v, (int, float)) and v == 0) for v in offset)
    mode = "both" if turned and moved else "rotated" if turned else "offset"
    return {"mode": mode, "origin": offset, "angles": angles}
