"""Snap targets for the board canvas, read from the vector Gerbers.

Board-local mm, Y up. Layers are stackup copper indices; -1 means every layer.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gerbonara import GerberFile
from gerbonara import graphic_objects as go
from gerbonara import graphic_primitives as gp
from gerbonara.utils import MM

from EMShrimulator.core.layout import drill
from EMShrimulator.core.layout.board import copper_layers, edge_cuts_file
from EMShrimulator.core.layout.gerber_render import edge_cuts_bbox_mm

POINT_KINDS: Tuple[str, ...] = (
    "via", "pad", "trace_end", "edge_mid", "trace_mid", "vertex", "face", "outline",
)
KIND_LABELS: Dict[str, str] = {
    "via": "via centre", "pad": "pad centre", "trace_end": "trace end", "edge_mid": "edge centre",
    "trace_mid": "trace midpoint", "vertex": "vertex", "face": "face centre", "outline": "board edge",
}
# Weights on the cursor distance, so a pad centre beats an equally close vertex.
KIND_BIAS = np.array([0.7, 0.75, 0.8, 0.85, 1.0, 1.0, 1.0, 1.1])

# Edges shorter than this get no edge-centre target (zone fills have many tiny segments).
MIN_EDGE_MM = 0.05
_KEY_DIGITS = 4  # endpoints within 0.1 µm are the same joint


@dataclass
class SnapHit:
    kind: str
    x_mm: float
    y_mm: float
    layer: int
    distance_mm: float
    edge: Optional[int] = None  # index into SnapSet.edges when the point is an edge centre

    def label(self, layer_names: Sequence[str]) -> str:
        where = layer_names[self.layer] if 0 <= self.layer < len(layer_names) else "all layers"
        return f"{KIND_LABELS[self.kind]} · {where}"


@dataclass
class EdgeHit:
    index: int
    x1: float
    y1: float
    x2: float
    y2: float
    nx: float  # unit normal into the copper
    ny: float
    layer: int
    distance_mm: float

    @property
    def mid(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def length_mm(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def rotation_deg(self) -> float:
        """Port rotation: width along the edge, a microstrip launch pointing into the copper."""
        return math.degrees(math.atan2(-self.nx, self.ny)) % 360.0


class SnapSet:
    """All targets of one board, as flat arrays for vectorised nearest queries."""

    def __init__(self, points: np.ndarray, kinds: np.ndarray, layers: np.ndarray, edge_of: np.ndarray,
                 edges: np.ndarray, normals: np.ndarray, edge_layers: np.ndarray, layer_names: List[str]):
        self.points = points          # (N, 2)
        self.kinds = kinds            # (N,) index into POINT_KINDS
        self.layers = layers          # (N,)
        self.edge_of = edge_of        # (N,) edge index or -1
        self.edges = edges            # (M, 4) x1, y1, x2, y2
        self.normals = normals        # (M, 2)
        self.edge_layers = edge_layers  # (M,)
        self.layer_names = layer_names

    @classmethod
    def empty(cls) -> "SnapSet":
        return cls(np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int),
                   np.zeros((0, 4)), np.zeros((0, 2)), np.zeros(0, int), [])

    def __len__(self) -> int:
        return len(self.points)

    def counts(self) -> Dict[str, int]:
        c = Counter(int(k) for k in self.kinds)
        return {POINT_KINDS[k]: n for k, n in sorted(c.items())}

    def _layer_mask(self, layers: np.ndarray, allowed: Optional[Sequence[int]]) -> np.ndarray:
        if allowed is None:
            return np.ones(len(layers), bool)
        return (layers == -1) | np.isin(layers, np.asarray(list(allowed), int))

    def nearest_point(self, x: float, y: float, radius_mm: float,
                      layers: Optional[Sequence[int]] = None,
                      kinds: Optional[Sequence[str]] = None) -> Optional[SnapHit]:
        if not len(self.points):
            return None
        mask = self._layer_mask(self.layers, layers)
        if kinds is not None:
            mask &= np.isin(self.kinds, [POINT_KINDS.index(k) for k in kinds])
        # Cheap box cull before the distance maths.
        mask &= (np.abs(self.points[:, 0] - x) <= radius_mm) & (np.abs(self.points[:, 1] - y) <= radius_mm)
        idx = np.nonzero(mask)[0]
        if not len(idx):
            return None
        d = np.hypot(self.points[idx, 0] - x, self.points[idx, 1] - y)
        inside = d <= radius_mm
        if not inside.any():
            return None
        idx, d = idx[inside], d[inside]
        # Bias by kind; among equals prefer the upper layer.
        score = d * KIND_BIAS[self.kinds[idx]] + 1e-9 * np.maximum(self.layers[idx], 0)
        best = idx[int(np.argmin(score))]
        return SnapHit(POINT_KINDS[int(self.kinds[best])], float(self.points[best, 0]), float(self.points[best, 1]),
                       int(self.layers[best]), float(np.hypot(self.points[best, 0] - x, self.points[best, 1] - y)),
                       int(self.edge_of[best]) if self.edge_of[best] >= 0 else None)

    def nearest_edge(self, x: float, y: float, radius_mm: float,
                     layers: Optional[Sequence[int]] = None) -> Optional[EdgeHit]:
        if not len(self.edges):
            return None
        e = self.edges
        mask = self._layer_mask(self.edge_layers, layers)
        mask &= (np.minimum(e[:, 0], e[:, 2]) - radius_mm <= x) & (np.maximum(e[:, 0], e[:, 2]) + radius_mm >= x)
        mask &= (np.minimum(e[:, 1], e[:, 3]) - radius_mm <= y) & (np.maximum(e[:, 1], e[:, 3]) + radius_mm >= y)
        idx = np.nonzero(mask)[0]
        if not len(idx):
            return None
        d = _segment_distance(e[idx], x, y)
        best_local = int(np.argmin(d + 1e-9 * np.maximum(self.edge_layers[idx], 0)))
        if d[best_local] > radius_mm:
            return None
        return self.edge(int(idx[best_local]), float(d[best_local]))

    def edge(self, index: int, distance_mm: float = 0.0) -> EdgeHit:
        if index is None or not 0 <= index < len(self.edges):
            raise IndexError(f"no edge {index}")
        x1, y1, x2, y2 = (float(v) for v in self.edges[index])
        nx, ny = (float(v) for v in self.normals[index])
        return EdgeHit(index, x1, y1, x2, y2, nx, ny, int(self.edge_layers[index]), distance_mm)


def _segment_distance(segs: np.ndarray, x: float, y: float) -> np.ndarray:
    ax, ay, bx, by = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = np.where(length2 > 0, ((x - ax) * dx + (y - ay) * dy) / np.where(length2 > 0, length2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(ax + t * dx - x, ay + t * dy - y)


# ------------------------------------------------------------------ building


class _Builder:
    def __init__(self, ox: float, oy: float):
        self.ox, self.oy = ox, oy
        self.points: List[Tuple[float, float, int, int, int]] = []   # x, y, kind, layer, edge
        self.edges: List[Tuple[float, float, float, float]] = []
        self.normals: List[Tuple[float, float]] = []
        self.edge_layers: List[int] = []

    def point(self, x: float, y: float, kind: str, layer: int, local: bool = False, edge: int = -1) -> None:
        if not local:
            x, y = x - self.ox, y - self.oy
        self.points.append((x, y, POINT_KINDS.index(kind), layer, edge))

    def edge(self, x1: float, y1: float, x2: float, y2: float, nx: float, ny: float, layer: int,
             mid_kind: str = "edge_mid") -> None:
        """An edge in Gerber mm with its into-copper normal, plus its centre as a point target."""
        length = math.hypot(x2 - x1, y2 - y1)
        norm = math.hypot(nx, ny)
        if length < MIN_EDGE_MM or norm == 0:
            return
        x1, y1, x2, y2 = x1 - self.ox, y1 - self.oy, x2 - self.ox, y2 - self.oy
        self.edges.append((x1, y1, x2, y2))
        self.normals.append((nx / norm, ny / norm))
        self.edge_layers.append(layer)
        self.point((x1 + x2) / 2, (y1 + y2) / 2, mid_kind, layer, local=True, edge=len(self.edges) - 1)

    def polygon(self, outline: Sequence[Tuple[float, float]], layer: int, dark: bool = True,
                vertices: bool = True, face: bool = True) -> None:
        pts = [(float(x), float(y)) for x, y in outline]
        if len(pts) > 1 and math.isclose(pts[0][0], pts[-1][0]) and math.isclose(pts[0][1], pts[-1][1]):
            pts = pts[:-1]
        if len(pts) < 3:
            return
        area, cx, cy = _polygon_centroid(pts)
        # Inward is left of each edge on a counter-clockwise outline.
        side = 1.0 if (area > 0) == dark else -1.0
        for i, (x1, y1) in enumerate(pts):
            x2, y2 = pts[(i + 1) % len(pts)]
            self.edge(x1, y1, x2, y2, -(y2 - y1) * side, (x2 - x1) * side, layer)
            if vertices:
                self.point(x1, y1, "vertex", layer)
        if face and dark and abs(area) > 0:
            self.point(cx, cy, "face", layer)

    def build(self, layer_names: List[str]) -> SnapSet:
        if self.points:
            arr = np.asarray(self.points, float)
            points, kinds = arr[:, :2], arr[:, 2].astype(int)
            layers, edge_of = arr[:, 3].astype(int), arr[:, 4].astype(int)
        else:
            points, kinds, layers, edge_of = np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
        edges = np.asarray(self.edges, float).reshape(-1, 4)
        normals = np.asarray(self.normals, float).reshape(-1, 2)
        return SnapSet(points, kinds, layers, edge_of, edges, normals,
                       np.asarray(self.edge_layers, int), layer_names)


def _polygon_centroid(pts: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    """(signed area, cx, cy); positive area is counter-clockwise."""
    a = cx = cy = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    a *= 0.5
    if abs(a) < 1e-12:
        xs, ys = zip(*pts)
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    return a, cx / (6 * a), cy / (6 * a)


def _rect_corners(x: float, y: float, w: float, h: float, rotation: float) -> List[Tuple[float, float]]:
    c, s = math.cos(rotation), math.sin(rotation)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):  # counter-clockwise
        out.append((x + dx * c - dy * s, y + dx * s + dy * c))
    return out


def _arc_point(obj: go.Arc, t: float) -> Tuple[float, float]:
    """Point at fraction t along an arc."""
    cx, cy = obj.cx + obj.x1, obj.cy + obj.y1  # gerbonara stores the centre relative to p1
    a1 = math.atan2(obj.y1 - cy, obj.x1 - cx)
    sweep = obj.sweep_angle() if callable(getattr(obj, "sweep_angle", None)) else 0.0
    sweep = -abs(sweep) if obj.clockwise else abs(sweep)
    r = math.hypot(obj.x1 - cx, obj.y1 - cy)
    return cx + r * math.cos(a1 + sweep * t), cy + r * math.sin(a1 + sweep * t)


def _add_flash(b: _Builder, obj: go.Flash, layer: int) -> None:
    b.point(obj.x, obj.y, "pad", layer)
    try:
        primitives = list(obj.to_primitives(MM))
    except Exception:  # noqa: BLE001 - an exotic aperture macro still gets its centre
        return
    for prim in primitives:
        dark = getattr(prim, "polarity_dark", True)
        if isinstance(prim, gp.Rectangle):
            b.polygon(_rect_corners(prim.x, prim.y, prim.w, prim.h, prim.rotation), layer, dark, face=False)
        elif isinstance(prim, gp.ArcPoly):
            b.polygon(prim.outline, layer, dark, face=False)
        # Circles have a centre target already and no straight edge.


def _add_traces(b: _Builder, objects: Sequence, layer: int) -> None:
    """Trace ends (endpoints no other segment shares), end caps, sides and midpoints."""
    segs = [o for o in objects if isinstance(o, (go.Line, go.Arc))]
    ends: Counter = Counter()
    for o in segs:
        ends[(round(o.x1, _KEY_DIGITS), round(o.y1, _KEY_DIGITS))] += 1
        ends[(round(o.x2, _KEY_DIGITS), round(o.y2, _KEY_DIGITS))] += 1

    for o in segs:
        width = 0.0
        if o.aperture is not None:
            try:
                width = float(o.aperture.equivalent_width(MM))
            except Exception:  # noqa: BLE001
                width = 0.0
        if isinstance(o, go.Line):
            dx, dy = o.x2 - o.x1, o.y2 - o.y1
            length = math.hypot(dx, dy)
            b.point((o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2, "trace_mid", layer)
            if length == 0:
                continue
            ux, uy = dx / length, dy / length
            tangents = ((o.x1, o.y1, -ux, -uy), (o.x2, o.y2, ux, uy))
            if width > 0:
                nx, ny = -uy * width / 2, ux * width / 2
                # Left side's copper is to its right, and the other way round.
                b.edge(o.x1 + nx, o.y1 + ny, o.x2 + nx, o.y2 + ny, uy, -ux, layer)
                b.edge(o.x2 - nx, o.y2 - ny, o.x1 - nx, o.y1 - ny, -uy, ux, layer)
        else:
            mx, my = _arc_point(o, 0.5)
            b.point(mx, my, "trace_mid", layer)
            tangents = []
            for (px, py), t in (((o.x1, o.y1), 0.0), ((o.x2, o.y2), 1.0)):
                qx, qy = _arc_point(o, 0.02 if t == 0.0 else 0.98)
                d = math.hypot(px - qx, py - qy) or 1.0
                tangents.append((px, py, (px - qx) / d, (py - qy) / d))

        for px, py, tx, ty in tangents:
            if ends[(round(px, _KEY_DIGITS), round(py, _KEY_DIGITS))] != 1:
                continue
            before = len(b.points)
            if width > 0:
                # The end cap, perpendicular to the trace; its centre is the trace end.
                nx, ny = -ty * width / 2, tx * width / 2
                b.edge(px - nx, py - ny, px + nx, py + ny, -tx, -ty, layer)
            if len(b.points) > before:
                b.points[-1] = (px - b.ox, py - b.oy, POINT_KINDS.index("trace_end"), layer, len(b.edges) - 1)
            else:
                b.point(px, py, "trace_end", layer)


def _add_layer(b: _Builder, gerber: GerberFile, layer: int) -> None:
    objects = []
    for obj in gerber.objects:
        try:
            objects.append(obj.converted(MM))
        except Exception:  # noqa: BLE001 - skip what cannot be converted
            continue
    for obj in objects:
        if isinstance(obj, go.Flash):
            _add_flash(b, obj, layer)
        elif isinstance(obj, go.Region):
            outline = getattr(obj, "outline", None) or []
            b.polygon(outline, layer, getattr(obj, "polarity_dark", True))
    _add_traces(b, objects, layer)


def build(project_dir: Path, stackup: Optional[List[Dict]],
          vias: Optional[List[Tuple[float, float]]] = None) -> SnapSet:
    """Every snap target on the board, for stackup layers only. `vias`: via centres in mm; None reads the drill file."""
    fab_dir = project_dir / "fab"
    edge = edge_cuts_file(fab_dir) if fab_dir.is_dir() else None
    if edge is None:
        return SnapSet.empty()
    bbox = edge_cuts_bbox_mm(edge)
    (ox, oy), _ = bbox
    b = _Builder(ox, oy)

    metals = [str(layer.get("name", "")) for layer in (stackup or []) if layer.get("type") == "copper"]
    for layer in copper_layers(fab_dir, stackup):
        if layer.sim_index is None:
            continue
        try:
            _add_layer(b, GerberFile.open(str(fab_dir / layer.file)), layer.sim_index)
        except Exception:  # noqa: BLE001 - an unreadable layer just offers no targets
            continue

    centres = vias if vias is not None else [(h.x_mm, h.y_mm) for h in drill.holes_for_project(fab_dir, bbox)]
    for x, y in centres:
        b.point(x, y, "via", -1, local=True)

    try:
        outline_objects = [o.converted(MM) for o in GerberFile.open(str(edge)).objects]
    except Exception:  # noqa: BLE001
        outline_objects = []
    for o in outline_objects:
        if isinstance(o, (go.Line, go.Arc)):
            b.point(o.x1, o.y1, "outline", -1)
            b.point(o.x2, o.y2, "outline", -1)
            mx, my = ((o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2) if isinstance(o, go.Line) else _arc_point(o, 0.5)
            b.point(mx, my, "outline", -1)

    snaps = b.build([m.replace(".", "_") for m in metals])
    # Keep one target per (kind, layer, position).
    if len(snaps.points):
        key = np.column_stack([np.round(snaps.points, _KEY_DIGITS), snaps.kinds, snaps.layers])
        _, first = np.unique(key, axis=0, return_index=True)
        first.sort()
        snaps.points, snaps.kinds = snaps.points[first], snaps.kinds[first]
        snaps.layers, snaps.edge_of = snaps.layers[first], snaps.edge_of[first]
    return snaps
