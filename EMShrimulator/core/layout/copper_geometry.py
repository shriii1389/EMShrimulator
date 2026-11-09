"""Copper pieces and their geometry, without Gerber parsing (see core.layout.copper).

Board-local mm, Y up. An edge's arc height is its signed sagitta, positive to the left of start -> end; 0 is straight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
EPS = 1e-9
JOIN_TOLERANCE_MM = 1e-3  # trace ends closer than this are one joint

TRACE, REGION, PAD = "trace", "region", "pad"
VIA = "via"  # a via as the copper tool selects it

# ------------------------------------------------------------------ geometry

def arc_center(a: Point, b: Point, height: float) -> Optional[Tuple[float, float, float]]:
    """(cx, cy, r) of the arc from a to b with this arc height, or None for a straight edge."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < EPS or abs(height) < EPS:
        return None
    nx, ny = -dy / length, dx / length  # left of a -> b
    t = (height * height - length * length / 4) / (2 * height)  # centre's offset from the chord's midpoint
    return (a[0] + b[0]) / 2 + nx * t, (a[1] + b[1]) / 2 + ny * t, abs(height - t)


def height_of(a: Point, b: Point, center: Point, clockwise: bool) -> float:
    """The arc height of a Gerber arc from a to b about `center`; 0 for a full circle."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < EPS:
        return 0.0
    nx, ny = -dy / length, dx / length
    t = (center[0] - (a[0] + b[0]) / 2) * nx + (center[1] - (a[1] + b[1]) / 2) * ny
    r = math.hypot(a[0] - center[0], a[1] - center[1])
    return t + r if clockwise else t - r


def _sweep(a: Point, b: Point, height: float, center: Tuple[float, float, float]) -> Tuple[float, float]:
    """(start angle, signed sweep) of an arc edge, clockwise negative."""
    cx, cy, _r = center
    a0 = math.atan2(a[1] - cy, a[0] - cx)
    sweep = math.atan2(b[1] - cy, b[0] - cx) - a0
    sweep = -((-sweep) % (2 * math.pi)) if height > 0 else sweep % (2 * math.pi)
    return a0, sweep


def edge_points(a: Point, b: Point, height: float, step_deg: float = 6.0) -> List[Point]:
    """The edge from a to b as a polyline: its points after a, ending on b."""
    center = arc_center(a, b, height)
    if center is None:
        return [b]
    cx, cy, r = center
    a0, sweep = _sweep(a, b, height, center)
    n = max(2, math.ceil(abs(sweep) / math.radians(step_deg)))
    return [(cx + r * math.cos(a0 + sweep * i / n), cy + r * math.sin(a0 + sweep * i / n)) for i in range(1, n)] + [b]


def edge_length(a: Point, b: Point, height: float) -> float:
    center = arc_center(a, b, height)
    if center is None:
        return math.hypot(b[0] - a[0], b[1] - a[1])
    return center[2] * abs(_sweep(a, b, height, center)[1])


def edge_point(a: Point, b: Point, height: float, t: float) -> Point:
    """The point a fraction `t` of the way along an edge."""
    center = arc_center(a, b, height)
    if center is None:
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    cx, cy, r = center
    a0, sweep = _sweep(a, b, height, center)
    return cx + r * math.cos(a0 + sweep * t), cy + r * math.sin(a0 + sweep * t)


def project_on_edge(a: Point, b: Point, height: float, p: Point) -> Tuple[float, float]:
    """(t, distance) of the edge's nearest point to p, t its fraction along the edge."""
    center = arc_center(a, b, height)
    if center is None:
        dx, dy = b[0] - a[0], b[1] - a[1]
        span = dx * dx + dy * dy
        t = 0.0 if span < EPS else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / span))
        return t, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)
    cx, cy, r = center
    a0, sweep = _sweep(a, b, height, center)
    angle = math.atan2(p[1] - cy, p[0] - cx) - a0
    angle = -((-angle) % (2 * math.pi)) if sweep < 0 else angle % (2 * math.pi)
    t = angle / sweep if abs(sweep) > EPS else 0.0
    if 0.0 <= t <= 1.0:
        return t, abs(math.hypot(p[0] - cx, p[1] - cy) - r)
    ends = ((0.0, math.hypot(p[0] - a[0], p[1] - a[1])), (1.0, math.hypot(p[0] - b[0], p[1] - b[1])))
    return min(ends, key=lambda end: end[1])


def split_height(height: float, a: Point, b: Point, t: float) -> Tuple[float, float]:
    """The arc heights of the two pieces an edge splits into at fraction t."""
    center = arc_center(a, b, height)
    if center is None:
        return 0.0, 0.0
    r = center[2]
    sweep = abs(_sweep(a, b, height, center)[1])
    sign = 1.0 if height > 0 else -1.0
    return tuple(sign * r * (1 - math.cos(s / 2)) for s in (sweep * t, sweep * (1 - t)))  # type: ignore[return-value]


def height_through(a: Point, b: Point, p: Point) -> float:
    """The arc height that bends edge a -> b through point p."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < EPS:
        return 0.0
    return ((p[0] - (a[0] + b[0]) / 2) * -dy + (p[1] - (a[1] + b[1]) / 2) * dx) / length


# ------------------------------------------------------------------ pieces

@dataclass
class Piece:
    """One copper object: a trace segment, a region (polygon) or a pad, in board-local mm."""

    kind: str
    points: List[Point]
    heights: List[float] = field(default_factory=list)
    width: float = 0.0
    net: Optional[str] = None
    dark: bool = True
    editable: bool = True
    outline: List[List[Point]] = field(default_factory=list)

    def edges(self) -> List[Tuple[Point, Point, float]]:
        pts = self.points
        if self.kind == TRACE:
            return [(pts[0], pts[1], self.heights[0] if self.heights else 0.0)]
        if self.kind == REGION:
            heights = list(self.heights) + [0.0] * (len(pts) - len(self.heights))
            return [(pts[i], pts[(i + 1) % len(pts)], heights[i]) for i in range(len(pts))]
        return []

    def polyline(self) -> List[Point]:
        """A trace's centreline or a region's closed outline, with arcs sampled."""
        edges = self.edges()
        if not edges:
            return list(self.points)
        out = [edges[0][0]]
        for a, b, h in edges:
            out.extend(edge_points(a, b, h))
        return out

    def pad_polygons(self) -> List[List[Point]]:
        cx, cy = self.points[0]
        return [[(cx + x, cy + y) for x, y in poly] for poly in self.outline]

    def bounds(self) -> Tuple[float, float, float, float]:
        if self.kind in (PAD, VIA):
            pts = [p for poly in self.pad_polygons() for p in poly] or list(self.points)
            pad = 0.0
        else:
            pts = self.polyline()
            pad = self.width / 2 if self.kind == TRACE else 0.0
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad

    def length(self) -> float:
        return sum(edge_length(a, b, h) for a, b, h in self.edges())

    def area(self) -> float:
        pts = self.polyline()
        return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2

    def moved(self, dx: float, dy: float) -> "Piece":
        return replace(self, points=[(x + dx, y + dy) for x, y in self.points])

    def as_edit(self) -> Dict:
        """What the project stores for it (core.layout.model.CopperObject's fields)."""
        return {"kind": self.kind, "points": [(round(x, 6), round(y, 6)) for x, y in self.points],
                "heights": [round(h, 6) for h in self.heights], "width_mm": round(self.width, 6)}

    def with_edit(self, edit) -> "Piece":
        """This piece reshaped as a stored edit has it; net and look stay."""
        data = edit if isinstance(edit, dict) else edit.model_dump()
        return replace(self, kind=data["kind"], points=[tuple(p) for p in data["points"]],
                       heights=list(data.get("heights") or []), width=float(data.get("width_mm") or 0.0))


def check_piece(piece: Dict) -> Optional[str]:
    """Why a piece cannot be written, or None."""
    kind, points = piece.get("kind"), piece.get("points") or []
    if kind == TRACE:
        if len(points) != 2:
            return "A trace segment has a start and an end."
        if float(piece.get("width_mm") or 0) <= 0:
            return "A trace needs a width above zero."
    elif kind == REGION:
        if len(points) < 3:
            return "A polygon needs at least three vertices."
    elif kind == PAD:
        if len(points) != 1:
            return "A pad has one centre."
    else:
        return f"'{kind}' is not a kind of copper."
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            return "A vertex is not a number."
    return None


# ------------------------------------------------------------------ a layer as edited

PieceId = Tuple[int, int]  # (index of the imported object, which of its pieces)


class EditedLayer:
    """One layer's pieces as edited: the imported ones, with edits standing in for theirs."""

    def __init__(self, name: str, pieces: List[Piece], editable: bool = True, reason: str = ""):
        self.name = name
        self.base = pieces
        self.editable = editable
        self.reason = reason
        self.edits: Dict[int, List[Piece]] = {}
        boxes = np.array([p.bounds() if p.points else (0, 0, 0, 0) for p in pieces], float).reshape(-1, 4)
        self._boxes = boxes
        self._kinds = np.array([{TRACE: 0, PAD: 1, REGION: 2}.get(p.kind, 1) for p in pieces], int)

    def set_edits(self, edits: Dict[int, List[Dict]]) -> None:
        self.edits = {index: [self.base[index].with_edit(e) for e in objs]
                      for index, objs in edits.items() if 0 <= index < len(self.base)}

    def get(self, pid: PieceId) -> Optional[Piece]:
        index, sub = pid
        if index in self.edits:
            objs = self.edits[index]
            return objs[sub] if 0 <= sub < len(objs) else None
        return self.base[index] if 0 <= index < len(self.base) and sub == 0 else None

    def pieces_of(self, index: int) -> List[Piece]:
        return list(self.edits[index]) if index in self.edits else [self.base[index]]

    def edited(self, index: int) -> bool:
        return index in self.edits

    def rows(self) -> Iterable[Tuple[PieceId, Piece]]:
        for index, piece in enumerate(self.base):
            if index in self.edits:
                for sub, obj in enumerate(self.edits[index]):
                    yield (index, sub), obj
            elif piece.points:
                yield (index, 0), piece

    def _near(self, x: float, y: float, tol: float) -> List[Tuple[PieceId, Piece]]:
        out = []
        if len(self._boxes):
            b = self._boxes
            hit = np.nonzero((b[:, 0] - tol <= x) & (x <= b[:, 2] + tol) & (b[:, 1] - tol <= y) & (y <= b[:, 3] + tol))[0]
            out = [((int(i), 0), self.base[int(i)]) for i in hit if int(i) not in self.edits]
        for index, objs in self.edits.items():
            for sub, obj in enumerate(objs):
                x0, y0, x1, y1 = obj.bounds()
                if x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol:
                    out.append(((index, sub), obj))
        return out

    def pick(self, x: float, y: float, tol: float) -> List[PieceId]:
        """What is under a point, best first: traces and pads by distance, then polygons, smallest first."""
        strokes, areas = [], []
        for pid, piece in self._near(x, y, tol):
            if not piece.dark or not piece.points:
                continue
            if piece.kind == TRACE:
                a, b, h = piece.edges()[0]
                d = project_on_edge(a, b, h, (x, y))[1] - piece.width / 2
                if d <= tol:
                    strokes.append((max(d, 0.0), pid))
            elif piece.kind == PAD:
                if any(_inside(poly, x, y) for poly in piece.pad_polygons()):
                    strokes.append((0.0, pid))
            elif _inside(piece.polyline(), x, y):
                areas.append((piece.area(), pid))
        return [pid for _d, pid in sorted(strokes)] + [pid for _a, pid in sorted(areas)]

    def trace_ends(self, point: Point, exclude: Iterable[PieceId] = (), tol: float = JOIN_TOLERANCE_MM
                   ) -> List[Tuple[PieceId, int]]:
        """Trace segments with an end on `point`: (piece, 0 for start or 1 for end)."""
        skip = set(exclude)
        out = []
        for pid, piece in self._near(point[0], point[1], tol):
            if piece.kind != TRACE or pid in skip:
                continue
            for end in (0, 1):
                p = piece.points[end]
                if abs(p[0] - point[0]) <= tol and abs(p[1] - point[1]) <= tol:
                    out.append((pid, end))
        return out

    def chain(self, start: PieceId) -> List[PieceId]:
        """Every trace segment joined end to end with `start`."""
        first = self.get(start)
        if first is None or first.kind != TRACE:
            return [start]
        seen, todo = {start}, [start]
        while todo:
            piece = self.get(todo.pop())
            for p in piece.points:
                for pid, _end in self.trace_ends(p):
                    other = self.get(pid)
                    if pid not in seen and other is not None and other.net == first.net:
                        seen.add(pid)
                        todo.append(pid)
        return sorted(seen)


def _inside(poly: Sequence[Point], x: float, y: float) -> bool:
    """Even-odd point in polygon test."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside

