"""Gerber -> vector shapes for the 2D canvas, so copper stays sharp at any zoom.

Shapes are in board-local mm, grouped into polarity runs, tiles and nets, and serialised in QDataStream's
QPainterPath format so the UI can read them without a Python loop. No Qt here; it runs in a worker process.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from gerbonara import GerberFile
from gerbonara import graphic_primitives as gp
from gerbonara.rs274x import GerberParser
from gerbonara.utils import MM

from EMShrimulator.core.layout.gerber_render import BBox

ELEMENT = np.dtype([("type", "<i4"), ("x", "<f8"), ("y", "<f8")])  # one QPainterPath::Element
MOVE, LINE, CURVE, CURVE_DATA = 0, 1, 2, 3
WINDING_FILL = 1
INDEX_COLUMNS = 8  # run, dark, stroke width (0 = filled), x0, y0, x1, y1 (mm), net (-1 = none)
TILES = 12  # along the board's long edge
MIN_STROKE_MM = 0.01  # gerbonara draws zero-width strokes this wide

Point = Tuple[float, float]
# A segment is (end,) for a line or (c1, c2, end) for a cubic.
Segment = Tuple[Point, ...]


def _arc(cx: float, cy: float, x1: float, y1: float, x2: float, y2: float, clockwise: bool) -> List[Segment]:
    """An arc from (x1, y1) to (x2, y2) about (cx, cy) as cubics of at most 90°; equal ends make a full circle."""
    r = math.hypot(x1 - cx, y1 - cy)
    a0 = math.atan2(y1 - cy, x1 - cx)
    full = x1 == x2 and y1 == y2
    sweep = math.atan2(y2 - cy, x2 - cx) - a0
    if clockwise:
        sweep = -2 * math.pi if full else -((-sweep) % (2 * math.pi))
    else:
        sweep = 2 * math.pi if full else sweep % (2 * math.pi)
    pieces = max(1, math.ceil(abs(sweep) / (math.pi / 2) - 1e-9))
    step = sweep / pieces
    k = 4.0 / 3.0 * math.tan(step / 4) * r
    out: List[Segment] = []
    a = a0
    for i in range(pieces):
        b = a0 + step * (i + 1)
        sa, ca, sb, cb = math.sin(a), math.cos(a), math.sin(b), math.cos(b)
        end = (x2, y2) if i == pieces - 1 else (cx + r * cb, cy + r * sb)
        out.append(((cx + r * ca - k * sa, cy + r * sa + k * ca), (cx + r * cb + k * sb, cy + r * sb - k * cb), end))
        a = b
    return out


def _signed_area(start: Point, segments: Sequence[Segment]) -> float:
    """Shoelace area over every point; only its sign is used."""
    pts = [start] + [p for seg in segments for p in seg]
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        total += x1 * y2 - x2 * y1
    return total / 2


def _reversed(start: Point, segments: Sequence[Segment]) -> Tuple[Point, List[Segment]]:
    nodes = [start] + [seg[-1] for seg in segments]
    out: List[Segment] = []
    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        out.append((nodes[i],) if len(seg) == 1 else (seg[1], seg[0], nodes[i]))
    return nodes[-1], out


def _rect(prim: gp.Rectangle) -> Tuple[Point, List[Segment]]:
    c, s = math.cos(prim.rotation), math.sin(prim.rotation)
    corners = [(prim.x + dx * c - dy * s, prim.y + dx * s + dy * c)
               for dx, dy in ((-prim.w / 2, -prim.h / 2), (prim.w / 2, -prim.h / 2),
                              (prim.w / 2, prim.h / 2), (-prim.w / 2, prim.h / 2))]
    return corners[0], [(p,) for p in corners[1:]]


def _arc_poly(prim: gp.ArcPoly) -> Tuple[Point, List[Segment]]:
    start = (float(prim.outline[0][0]), float(prim.outline[0][1]))
    segments: List[Segment] = []
    for (x1, y1), (x2, y2), (clockwise, (cx, cy)) in prim.segments:
        if clockwise is None:
            segments.append(((float(x2), float(y2)),))
        else:
            segments.extend(_arc(cx, cy, x1, y1, x2, y2, clockwise))
    return start, segments


class _Group:
    """The subpaths of one (run, tile, width) as QPainterPath elements, and their bounds."""

    def __init__(self):
        self.types: List[int] = []
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.last_move = 0
        self.bounds = [math.inf, math.inf, -math.inf, -math.inf]

    def add(self, start: Point, segments: Sequence[Segment], pad: float) -> None:
        self.last_move = len(self.types)
        self.types.append(MOVE)
        self.xs.append(start[0])
        self.ys.append(start[1])
        for seg in segments:
            if len(seg) == 1:
                self.types.append(LINE)
            else:
                self.types.extend((CURVE, CURVE_DATA, CURVE_DATA))
            for x, y in seg:
                self.xs.append(x)
                self.ys.append(y)
        b = self.bounds
        first = len(self.xs) - 1 - sum(len(seg) for seg in segments)
        xs, ys = self.xs[first:], self.ys[first:]  # control points bound their curves
        b[0], b[1] = min(b[0], min(xs) - pad), min(b[1], min(ys) - pad)
        b[2], b[3] = max(b[2], max(xs) + pad), max(b[3], max(ys) + pad)

    def extend(self, segments: Sequence[Segment], pad: float) -> None:
        """Carry on the last subpath (a trace continuing from where it ended)."""
        for seg in segments:
            if len(seg) == 1:
                self.types.append(LINE)
            else:
                self.types.extend((CURVE, CURVE_DATA, CURVE_DATA))
            for x, y in seg:
                self.xs.append(x)
                self.ys.append(y)
                b = self.bounds
                b[0], b[1] = min(b[0], x - pad), min(b[1], y - pad)
                b[2], b[3] = max(b[2], x + pad), max(b[3], y + pad)

    def to_bytes(self) -> bytes:
        body = np.empty(len(self.types), ELEMENT)
        body["type"], body["x"], body["y"] = self.types, self.xs, self.ys
        return (np.array([len(self.types)], "<i4").tobytes() + body.tobytes()
                + np.array([self.last_move, WINDING_FILL], "<i4").tobytes())


class _Builder:
    def __init__(self, bounds_mm: BBox):
        (x0, y0), (x1, y1) = bounds_mm
        self.ox, self.oy = x0, y0
        self.width, self.height = x1 - x0, y1 - y0
        self.tile = max(self.width, self.height, 1e-6) / TILES
        self.nx = max(1, math.ceil(self.width / self.tile))
        self.ny = max(1, math.ceil(self.height / self.tile))
        self.groups: Dict[Tuple[int, int, float, int], _Group] = {}  # (run, tile, width, net)
        self.dark: List[bool] = []
        self.nets: Dict[str, int] = {}
        # The open trace: (key, end point).
        self.trace = None

    def _local(self, start: Point, segments: Sequence[Segment]) -> Tuple[Point, List[Segment]]:
        ox, oy = self.ox, self.oy
        return (start[0] - ox, start[1] - oy), [tuple((x - ox, y - oy) for x, y in seg) for seg in segments]

    def _tile(self, x: float, y: float) -> int:
        tx = min(max(int(x / self.tile), 0), self.nx - 1)
        ty = min(max(int(y / self.tile), 0), self.ny - 1)
        return ty * self.nx + tx

    def _run(self, dark: bool) -> int:
        if not self.dark or self.dark[-1] != dark:
            self.dark.append(dark)
            self.trace = None
        return len(self.dark) - 1

    def net_id(self, net: Optional[str]) -> int:
        return -1 if net is None else self.nets.setdefault(net, len(self.nets))

    def fill(self, start: Point, segments: List[Segment], dark: bool, net: int = -1) -> None:
        if not segments:
            return
        start, segments = self._local(start, segments)
        if _signed_area(start, segments) < 0:
            start, segments = _reversed(start, segments)
        run = self._run(dark)
        xs = [start[0]] + [p[0] for seg in segments for p in seg]
        ys = [start[1]] + [p[1] for seg in segments for p in seg]
        tile = self._tile((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        self.groups.setdefault((run, tile, 0.0, net), _Group()).add(start, segments, 0.0)

    def stroke(self, start: Point, segments: List[Segment], width: float, dark: bool, net: int = -1) -> None:
        start, segments = self._local(start, segments)
        width = max(float(width), MIN_STROKE_MM)
        run = self._run(dark)
        pad = width / 2
        trace = self.trace
        key = (run, self._tile(*start), width, net)
        if trace is not None and trace[0][0] == run and trace[0][2:] == key[2:] and trace[1] == start:
            self.groups[trace[0]].extend(segments, pad)
        else:
            self.groups.setdefault(key, _Group()).add(start, segments, pad)
            trace = (key, None)
        self.trace = (trace[0], segments[-1][-1] if segments else start)

    def add(self, prim, net: int = -1) -> None:
        dark = bool(getattr(prim, "polarity_dark", True))
        if isinstance(prim, gp.Line):
            self.stroke((prim.x1, prim.y1), [((prim.x2, prim.y2),)], prim.width, dark, net)
        elif isinstance(prim, gp.Arc):
            self.stroke((prim.x1, prim.y1), _arc(prim.cx, prim.cy, prim.x1, prim.y1, prim.x2, prim.y2,
                                                 prim.clockwise), prim.width, dark, net)
        else:
            self.trace = None
            if isinstance(prim, gp.Circle):
                x, y, r = prim.x, prim.y, prim.r
                self.fill((x + r, y), _arc(x, y, x + r, y, x + r, y, False), dark, net)
            elif isinstance(prim, gp.Rectangle):
                self.fill(*_rect(prim), dark, net)
            elif isinstance(prim, gp.ArcPoly) and len(prim.outline):
                self.fill(*_arc_poly(prim), dark, net)

    def result(self) -> Dict[str, np.ndarray]:
        keys = sorted(self.groups)  # by run, so a clear run comes after what it erases
        index = np.empty((len(keys), INDEX_COLUMNS), float)
        chunks = []
        for i, key in enumerate(keys):
            group = self.groups[key]
            run, _tile, width, net = key
            index[i] = (run, float(self.dark[run]), width, *group.bounds, net)
            chunks.append(group.to_bytes())
        names = sorted(self.nets, key=self.nets.get)
        return {"index": index, "blob": np.frombuffer(b"".join(chunks), np.uint8),
                "nets": np.array(names, dtype=str) if names else np.empty(0, dtype="<U1")}


class _StampedObjects(list):
    """A GerberFile's objects, with regions stamped with their X2 attributes (gerbonara skips them)."""

    def __init__(self, parser: GerberParser):
        super().__init__()
        self.parser = parser

    def append(self, obj) -> None:
        if not isinstance(getattr(obj, "attrs", None), dict):
            obj.attrs = dict(self.parser.graphics_state.object_attrs)
        super().append(obj)


def open_with_nets(gerber_path: Path) -> GerberFile:
    """GerberFile.open with every object's X2 attributes, read as UTF-8."""
    gerber = GerberFile()
    parser = GerberParser(gerber)
    gerber.objects = _StampedObjects(parser)
    parser.parse(Path(gerber_path).read_text(encoding="utf-8", errors="replace"), filename=str(gerber_path))
    return gerber


def net_of(obj) -> Optional[str]:
    """An object's net from its `.N` attribute; None without one."""
    attrs = getattr(obj, "attrs", None)
    value = attrs.get(".N") if isinstance(attrs, dict) else None
    if not value:
        return None
    name = value[0] if isinstance(value, (tuple, list)) else str(value)
    return name or None


def layer_shapes(gerber_path: Path, bounds_mm: BBox) -> Dict[str, np.ndarray]:
    """One layer as {"index": (N, 8) float, "blob": serialised QPainterPaths, "nets": net names}."""
    builder = _Builder(bounds_mm)
    for obj in open_with_nets(gerber_path).objects:
        try:
            primitives = list(obj.to_primitives(unit=MM))
        except Exception:  # noqa: BLE001 - an exotic aperture macro is left out, not the whole layer
            continue
        net = builder.net_id(net_of(obj))
        for prim in primitives:
            builder.add(prim, net)
    return builder.result()
