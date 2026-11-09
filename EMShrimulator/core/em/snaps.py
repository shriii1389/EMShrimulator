"""Snap targets on the 3D model: face centres, edge centres, quadrant points, circle centres and corners.

Faces and edges are recovered from the triangle meshes the viewport draws. No Qt or VTK here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

FACE, EDGE, QUADRANT, CORNER, GRID, CENTRE, ARC = "face", "edge", "quadrant", "corner", "grid", "centre", "arc"
POINT_KINDS: Tuple[str, ...] = (FACE, EDGE, QUADRANT, CORNER, GRID, CENTRE, ARC)
KIND_LABELS = {FACE: "face centre", EDGE: "edge centre", QUADRANT: "edge quadrant", CORNER: "corner",
               GRID: "grid point", CENTRE: "object centre", ARC: "circle centre"}
# The snap toggle that offers each kind. Object centres are only offered when a tool asks for them.
TOGGLE_OF = {FACE: FACE, EDGE: EDGE, QUADRANT: EDGE, CORNER: CORNER, GRID: GRID, CENTRE: FACE, ARC: FACE}
# Weights on the cursor distance, so a corner beats an equally close face centre.
KIND_BIAS = np.array([1.0, 0.9, 1.0, 0.8, 1.0, 1.0, 1.0])

FEATURE_ANGLE_DEG = 35.0  # matches viewport3d's feature edges
CORNER_ANGLE_DEG = 25.0  # above a 32-facet circle's 11.25° turn
DEPTH_TIE_PX = 3.0  # within this of the best target on screen, the one nearest the eye wins
FLAT = 0.999  # mean-normal length above which a face is flat
ARC_FIT = 1e-3  # fit tolerance of a round edge, as a fraction of its radius
ARC_TURN_DEG = 20.0  # minimum turn for an arc to have a centre


@dataclass
class SnapSet:
    """Snap targets as flat arrays, for vectorised queries."""

    points: np.ndarray  # (N, 3)
    kinds: np.ndarray   # (N,) index into POINT_KINDS

    @classmethod
    def empty(cls) -> "SnapSet":
        return cls(np.zeros((0, 3)), np.zeros(0, dtype=int))

    @classmethod
    def of(cls, points, kinds: Sequence[str]) -> "SnapSet":
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return cls(points, np.array([POINT_KINDS.index(k) for k in kinds], dtype=int).reshape(-1))

    @classmethod
    def concat(cls, sets: Iterable["SnapSet"]) -> "SnapSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            return cls.empty()
        return cls(np.vstack([s.points for s in sets]), np.concatenate([s.kinds for s in sets]))

    def __len__(self) -> int:
        return len(self.points)

    def of_kind(self, kind: str) -> np.ndarray:
        return self.points[self.kinds == POINT_KINDS.index(kind)]


def _parts(*groups: Tuple[str, Sequence[np.ndarray]]) -> SnapSet:
    sets = [SnapSet.of(np.array(points), [kind] * len(points)) for kind, points in groups if len(points)]
    return SnapSet.concat(sets)


# ------------------------------------------------------------------ solids

def _weld(vertices: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge vertices at the same place, so triangles that touch share their edge."""
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(v) or not len(t):
        return v, t
    span = float(np.ptp(v, axis=0).max()) or 1.0
    _, first, inverse = np.unique(np.round(v / (span * 1e-9)), axis=0, return_index=True, return_inverse=True)
    return v[first], inverse.reshape(-1)[t]


def _components(count: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Connected components over the links a[i]-b[i], labelled by their lowest member."""
    labels = np.arange(count)
    while len(a):
        low = np.minimum(labels[a], labels[b])
        updated = labels.copy()
        for side in (a, b, labels[a], labels[b]):
            np.minimum.at(updated, side, low)
        updated = updated[updated]
        if np.array_equal(updated, labels):
            break
        labels = updated
    return labels


def solid_snaps(vertices: np.ndarray, triangles: np.ndarray, axes: Optional[np.ndarray] = None) -> SnapSet:
    """Snap targets of a triangle mesh; `axes` orient the quadrant points (global x, y, z when None)."""
    parts = _faces_and_edges(vertices, triangles)
    if parts is None:
        return SnapSet.empty()
    v, centres, normals, edges = parts
    centres = centres[np.linalg.norm(normals, axis=1) >= FLAT]
    corners, middles, quadrants, arcs = _edge_points(v, edges, axes)
    span = float(np.ptp(v, axis=0).max()) or 1.0
    return _parts((FACE, centres), (EDGE, middles), (QUADRANT, quadrants), (CORNER, corners),
                  (ARC, _apart(arcs, centres, span * 1e-6)))


def solid_features(vertices: np.ndarray, triangles: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """(face centres, mean face normals, edge centres, corners) of a closed triangle mesh."""
    parts = _faces_and_edges(vertices, triangles)
    if parts is None:
        return np.zeros((0, 3)), np.zeros((0, 3)), [], []
    v, centres, normals, edges = parts
    corners, middles, _quadrants, _arcs = _edge_points(v, edges, None)
    return centres, normals, middles, corners


def _apart(points: Sequence[np.ndarray], taken: np.ndarray, tol: float) -> List[np.ndarray]:
    """The points not already in `taken`."""
    taken = np.asarray(taken, dtype=float).reshape(-1, 3)
    out: List[np.ndarray] = []
    for point in points:
        if len(taken) and float(np.min(np.linalg.norm(taken - point, axis=1))) <= tol:
            continue
        if any(float(np.linalg.norm(kept - point)) <= tol for kept in out):
            continue
        out.append(point)
    return out


def _faces_and_edges(vertices: np.ndarray, triangles: np.ndarray):
    """(welded vertices, face centres, face mean normals, feature edges), or None if the mesh has no area."""
    v, t = _weld(vertices, triangles)
    if not len(t):
        return None
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    cross = np.cross(b - a, c - a)
    double_area = np.linalg.norm(cross, axis=1)
    keep = double_area > double_area.max() * 1e-12  # slivers have no direction to compare
    if not keep.any():
        return None
    t, a, b, c, cross, double_area = t[keep], a[keep], b[keep], c[keep], cross[keep], double_area[keep]
    normals = cross / double_area[:, None]

    # Every edge once, with the triangles on either side of it.
    edges = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(t)), 3)
    key = edges[:, 0] * len(v) + edges[:, 1]
    order = np.argsort(key, kind="stable")
    key, edges, owner = key[order], edges[order], owner[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    counts = np.diff(np.r_[starts, len(key)])
    paired = np.flatnonzero(counts == 2)
    left, right = owner[starts[paired]], owner[starts[paired] + 1]
    smooth = np.einsum("ij,ij->i", normals[left], normals[right]) > np.cos(np.radians(FEATURE_ANGLE_DEG))

    # Faces: triangles joined across smooth edges.
    _, faces = np.unique(_components(len(t), left[smooth], right[smooth]), return_inverse=True)
    area = double_area / 2
    centroid = (a + b + c) / 3
    face_area = np.bincount(faces, weights=area)
    centres = np.stack([np.bincount(faces, weights=area * centroid[:, i]) for i in range(3)], axis=1)
    mean_normals = np.stack([np.bincount(faces, weights=cross[:, i] / 2) for i in range(3)], axis=1)
    real = face_area > face_area.sum() * 1e-6  # not the slivers a boolean leaves behind
    centres = centres[real] / face_area[real, None]
    mean_normals = mean_normals[real] / face_area[real, None]

    # Edges: rims (one triangle), seams (more than two) and creases.
    feature = np.ones(len(starts), dtype=bool)
    feature[paired[smooth]] = False
    return v, centres, mean_normals, edges[starts[feature]]


# ------------------------------------------------------------------ edges

def _edge_points(v: np.ndarray, edges: np.ndarray, axes: Optional[np.ndarray]
                 ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """(corners, edge midpoints, quadrant points, circle centres) of a graph of edges between the points `v`."""
    if not len(edges):
        return [], [], [], []
    used, local = np.unique(edges, return_inverse=True)
    local = local.reshape(-1, 2)
    p = v[used]
    degree = np.bincount(local.ravel(), minlength=len(p))
    # A corner: where edges branch, end or turn sharply.
    corner = degree != 2
    ends = np.concatenate([local, local[:, ::-1]])
    ends = ends[np.argsort(ends[:, 0], kind="stable")]
    first = np.r_[0, np.cumsum(degree)[:-1]]
    two = np.flatnonzero(degree == 2)
    if len(two):
        incoming = p[two] - p[ends[first[two], 1]]
        outgoing = p[ends[first[two] + 1, 1]] - p[two]
        lengths = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
        cosine = np.einsum("ij,ij->i", incoming, outgoing) / np.where(lengths > 0, lengths, 1.0)
        corner[two] = cosine < np.cos(np.radians(CORNER_ANGLE_DEG))

    neighbours: List[List[Tuple[int, int]]] = [[] for _ in range(len(p))]
    for index, (x, y) in enumerate(local.tolist()):
        neighbours[x].append((y, index))
        neighbours[y].append((x, index))
    walked = np.zeros(len(local), dtype=bool)

    def walk(start: int, edge: int) -> List[int]:
        """From `start` along `edge`, on through plain points, to the next corner or back to the start."""
        chain, at = [start], start
        while True:
            walked[edge] = True
            x, y = local[edge]
            at = int(y if x == at else x)
            chain.append(at)
            if corner[at] or at == start:
                return chain
            (_, e0), (_, e1) = neighbours[at]
            edge = e1 if e0 == edge else e0
            if walked[edge]:
                return chain

    middles, arcs = [], []
    for start in np.flatnonzero(corner):
        for _, edge in neighbours[start]:
            if not walked[edge]:
                chain = p[walk(int(start), edge)]
                middles.append(_halfway(chain))
                centre = _arc_centre(chain)
                if centre is not None:
                    arcs.append(centre)
    quadrants: List[np.ndarray] = []
    for edge in range(len(local)):
        if not walked[edge]:  # a closed loop with no corner on it
            centre, points = _round_loop(p[walk(int(local[edge, 0]), edge)][:-1], axes)
            quadrants.extend(points)
            if centre is not None:
                arcs.append(centre)
    return list(p[corner]), middles, quadrants, arcs


def _halfway(chain: np.ndarray) -> np.ndarray:
    """The point halfway along a polyline, by length."""
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 0:
        return chain[0]
    reach = np.cumsum(lengths)
    i = min(int(np.searchsorted(reach, total / 2)), len(lengths) - 1)
    t = (total / 2 - (reach[i] - lengths[i])) / lengths[i] if lengths[i] > 0 else 0.0
    return chain[i] + (chain[i + 1] - chain[i]) * t


# Plain arithmetic: np.cross and np.roll are slow on small arrays.

def _next(points: np.ndarray) -> np.ndarray:
    """Each point's successor round a closed loop."""
    return np.concatenate([points[1:], points[:1]])


def _plane_of(loop: np.ndarray, axes: Optional[np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A closed loop's (normal, u, w), with u and w in its plane."""
    x, y, z = loop[:, 0], loop[:, 1], loop[:, 2]
    nxt = _next(loop)
    xn, yn, zn = nxt[:, 0], nxt[:, 1], nxt[:, 2]
    # Newell's method: twice the vector area.
    normal = np.array([(y * zn - z * yn).sum(), (z * xn - x * zn).sum(), (x * yn - y * xn).sum()])
    size = float(np.sqrt(normal @ normal))
    if size <= 1e-18:
        return None
    normal /= size
    axes = np.eye(3) if axes is None else np.asarray(axes, dtype=float)
    across = axes[int(np.argmin(np.abs(axes @ normal)))]
    u = across - normal * float(across @ normal)
    u /= np.sqrt(u @ u)
    w = np.array([normal[1] * u[2] - normal[2] * u[1], normal[2] * u[0] - normal[0] * u[2],
                  normal[0] * u[1] - normal[1] * u[0]])
    return normal, u, w


def _area_centroid(loop2: np.ndarray) -> Tuple[float, np.ndarray]:
    """Area (unsigned) and centroid of a closed 2D outline."""
    x, y = loop2[:, 0], loop2[:, 1]
    nxt = _next(loop2)
    xn, yn = nxt[:, 0], nxt[:, 1]
    cross = x * yn - xn * y
    area = cross.sum() / 2
    if abs(area) < 1e-18:
        return 0.0, loop2.mean(axis=0)
    return abs(area), np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * area)


_QUADRANT_RAYS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def _round_loop(loop: np.ndarray, axes: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """A round loop's centre and quadrant points, if it is a flat circle."""
    if len(loop) < 3:
        return None, []
    plane = _plane_of(loop, axes)
    if plane is None:
        return None, []
    normal, u, w = plane
    flat = np.stack([loop @ u, loop @ w], axis=1)
    middle = flat.mean(axis=0)  # shoelace sums are more accurate near the centre
    centre = _area_centroid(flat - middle)[1] + middle
    reach = _ray_exits(flat - centre, _QUADRANT_RAYS)
    found = np.isfinite(reach)
    at = np.vstack([centre, centre + _QUADRANT_RAYS[found] * reach[found, None]])
    # Back to 3D, keeping the loop's offset along its normal.
    lifted = loop[0] + np.outer(at[:, 0] - flat[0, 0], u) + np.outer(at[:, 1] - flat[0, 1], w)
    radii = np.hypot(*(flat - centre).T)
    round_ = _on_plane(loop, normal) and float(np.ptp(radii)) <= ARC_FIT * float(radii.mean())
    return (lifted[0] if round_ else None), list(lifted[1:])


def _on_plane(points: np.ndarray, normal: np.ndarray) -> bool:
    """Whether points lie in one plane facing `normal`, allowing for single-precision vertices."""
    heights = points @ normal
    spread = float(np.ptp(points, axis=0).max())
    return float(np.ptp(heights)) <= 1e-4 * spread + 1e-6 * float(np.abs(points).max())


def _arc_centre(chain: np.ndarray) -> Optional[np.ndarray]:
    """The centre of an arc edge between two corners, or None if it is not round."""
    if len(chain) < 4:  # three segments at least: two make any corner look round
        return None
    plane = _plane_of(chain, None)  # closed by its chord, which lies in the arc's plane
    if plane is None or not _on_plane(chain, plane[0]):
        return None
    normal, u, w = plane
    flat = np.stack([chain @ u, chain @ w], axis=1)
    middle = flat.mean(axis=0)
    rel = flat - middle
    # The circle through them, by least squares: x² + y² = 2ax + 2by + c.
    fit, *_ = np.linalg.lstsq(np.column_stack([2 * rel, np.ones(len(rel))]), (rel ** 2).sum(axis=1), rcond=None)
    centre = fit[:2]
    radius_sq = float(fit[2] + centre @ centre)
    if radius_sq <= 0:
        return None
    radius = np.sqrt(radius_sq)
    if float(np.max(np.abs(np.hypot(*(rel - centre).T) - radius))) > ARC_FIT * radius:
        return None
    steps = np.diff(rel, axis=0)
    turns = np.arctan2(steps[:-1, 0] * steps[1:, 1] - steps[:-1, 1] * steps[1:, 0],
                       np.einsum("ij,ij->i", steps[:-1], steps[1:]))
    if abs(float(np.degrees(turns.sum()))) < ARC_TURN_DEG:
        return None
    at = centre + middle
    return chain[0] + (at[0] - flat[0, 0]) * u + (at[1] - flat[0, 1]) * w


def _ray_exits(loop2: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Distance along each ray from the origin to its last crossing of a closed 2D outline; nan if none."""
    a = loop2
    edge = _next(loop2) - a
    dx, dy = directions[:, :1], directions[:, 1:]                 # (rays, 1) against (edges,)
    denom = dx * edge[:, 1] - dy * edge[:, 0]
    ok = np.abs(denom) > 1e-18
    safe = np.where(ok, denom, 1.0)
    s = (a[:, 0] * edge[:, 1] - a[:, 1] * edge[:, 0]) / safe     # along the ray
    t = (a[:, 0] * dy - a[:, 1] * dx) / safe                      # along the edge
    hit = ok & (s > 0) & (t >= -1e-9) & (t <= 1 + 1e-9)
    farthest = np.where(hit, s, -np.inf).max(axis=1)
    return np.where(np.isfinite(farthest), farthest, np.nan)


# ------------------------------------------------------------------ sheets

def sheet_snaps(loops: Sequence[np.ndarray], axes: Optional[np.ndarray] = None) -> SnapSet:
    """Snap targets of a sheet: its outline and, for a ring, its hole."""
    loops = [np.asarray(loop, dtype=float).reshape(-1, 3) for loop in loops if len(loop) >= 2]
    if not loops:
        return SnapSet.empty()
    centres = []
    plane = _plane_of(loops[0], axes) if len(loops[0]) >= 3 else None
    if plane is not None:
        normal, u, w = plane
        flat = [np.stack([loop @ u, loop @ w], axis=1) for loop in loops]
        middle = flat[0].mean(axis=0)  # shoelace sums are more accurate near the centre
        area, centre = _area_centroid(flat[0] - middle)
        moment = centre * area
        for hole in flat[1:]:
            hole_area, hole_centre = _area_centroid(hole - middle)
            area, moment = area - hole_area, moment - hole_centre * hole_area
        if area > 1e-18:
            centre = moment / area + middle
            offset = float(loops[0][0] @ normal)
            centres.append(centre[0] * u + centre[1] * w + offset * normal)
    points, edges, start = [], [], 0
    for loop in loops:
        n = len(loop)
        points.append(loop)
        ring = np.arange(n) + start
        edges.append(np.stack([ring, np.roll(ring, -1)], axis=1) if n > 2 else ring[None, :])
        start += n
    every = np.vstack(points)
    corners, middles, quadrants, arcs = _edge_points(every, np.vstack(edges), axes)
    span = float(np.ptp(every, axis=0).max()) or 1.0
    return _parts((FACE, centres), (EDGE, middles), (QUADRANT, quadrants), (CORNER, corners),
                  (ARC, _apart(arcs, np.array(centres), span * 1e-6)))


# ------------------------------------------------------------------ choosing

def nearest(screen: np.ndarray, depth: np.ndarray, kinds: np.ndarray, cursor: Sequence[float], radius: float,
            allowed: Iterable[str]) -> Optional[int]:
    """The target nearest the cursor within `radius`, weighed by kind; near-ties go to the one nearest the eye."""
    if not len(screen):
        return None
    allowed_kinds = [POINT_KINDS.index(k) for k in allowed]
    distance = np.hypot(screen[:, 0] - cursor[0], screen[:, 1] - cursor[1])
    candidates = np.flatnonzero(np.isin(kinds, allowed_kinds) & np.isfinite(depth) & (distance <= radius))
    if not len(candidates):
        return None
    best = candidates[np.argmin(distance[candidates] * KIND_BIAS[kinds[candidates]])]
    offset = screen[candidates] - screen[best]
    stacked = candidates[np.hypot(offset[:, 0], offset[:, 1]) <= DEPTH_TIE_PX]
    return int(stacked[np.argmin(depth[stacked])])


def on_grid(point: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray, step: float) -> np.ndarray:
    """The point moved to the nearest grid node, keeping its offset along the normal."""
    if step <= 0:
        return point
    rel = np.asarray(point, dtype=float) - origin
    a, b = float(rel @ u), float(rel @ v)
    return point + (round(a / step) * step - a) * u + (round(b / step) * step - b) * v


def round_to(value: float, step: float) -> float:
    return round(value / step) * step if step > 0 else value
