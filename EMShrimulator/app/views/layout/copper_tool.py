"""Selecting and reshaping copper and vias on the 2D canvas.

Traces have end and bend handles, polygons have vertex and mid-edge handles, and dragging the body moves the
selection. Joined trace ends follow. Ctrl moves freely, Shift keeps 45° steps, Esc drops a drag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPainterPathStroker, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QApplication, QGraphicsObject

from EMShrimulator.app import theme
from EMShrimulator.app.views.layout import palette
from EMShrimulator.core.layout import padstacks as pstk
from EMShrimulator.core.layout.copper_geometry import (
    PAD,
    REGION,
    TRACE,
    VIA,
    EditedLayer,
    Piece,
    Point,
    edge_point,
    height_through,
    project_on_edge,
    split_height,
)

HANDLE_PX = 7.0  # a handle's size on screen
GRAB_PX = 6.0  # how near a handle the cursor must be to take it
PICK_PX = 4.0  # how near copper the cursor must be to pick it
DRAG_PX = 3.0  # how far a press must move to be a drag, not a click
STRAIGHT_PX = 6.0  # a bend dragged this near the chord straightens
EDGE_HANDLES_UP_TO = 200  # polygons with more vertices (pours) get no mid-edge handles
HOVER_MS = 40

Ref = Tuple[str, int, int]  # (layer, object index, piece index); a via's is (VIA_REF, id, 0)
VIA_REF = "#via"


def is_via(ref: Ref) -> bool:
    return ref[0] == VIA_REF


@dataclass
class Handle:
    kind: str  # "end", "bend", "vertex" or "edge"
    ref: Ref
    i: int  # which end, vertex or edge
    point: Point


@dataclass
class _Drag:
    kind: str  # a handle's kind, or "body"
    ref: Optional[Ref]
    i: int
    start: Point  # where the press was, mm
    anchor: Point  # the point that snaps
    pieces: Dict[Ref, Piece]  # what the drag reshapes, as it was
    joints: List[Tuple[Ref, int]] = field(default_factory=list)  # other trace ends that follow
    preview: Dict[Ref, Piece] = field(default_factory=dict)
    moved: bool = False


def _shift() -> bool:
    return bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)


def _octilinear(origin: Point, p: Point) -> Point:
    """p moved onto the nearest ray from origin at a multiple of 45°."""
    dx, dy = p[0] - origin[0], p[1] - origin[1]
    angle = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
    ux, uy = math.cos(angle), math.sin(angle)
    d = dx * ux + dy * uy
    return origin[0] + ux * d, origin[1] + uy * d


def describe(piece: Piece) -> str:
    if piece.kind == TRACE:
        return "Arc segment" if piece.heights and abs(piece.heights[0]) > 1e-9 else "Trace segment"
    return {REGION: "Polygon", VIA: "Via"}.get(piece.kind, "Pad")


class HoleOverlay(QGraphicsObject):
    """The holes of vias being dragged, at their new positions."""

    def __init__(self, tool: "CopperTool"):
        super().__init__()
        self.tool = tool
        self.setZValue(15.5)  # over the copper tool's previews (15), under ports (20)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.drawn = False  # holes are on screen: one more repaint takes them off

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        return self.tool.canvas.sceneRect()

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        tool = self.tool
        moving = [(tool.vias.get(ref[1]), piece) for ref, piece in tool.dragged().items() if is_via(ref)]
        self.drawn = bool(moving) and tool.show_holes
        if not self.drawn:
            return
        c = theme.colors()
        painter.setPen(QPen(QColor(c["text_faint"]), 0))
        painter.setBrush(QColor(c["sunken"]))
        for via, piece in moving:
            if via is not None:
                r = tool.hole_mm(via) / 2 * tool.canvas.base_ppm
                painter.drawEllipse(tool.canvas.to_scene(*piece.points[0]), r, r)


class CopperOverlay(QGraphicsObject):
    """Draws the selection, hover, handles and previews over the board."""

    def __init__(self, tool: "CopperTool"):
        super().__init__()
        self.tool = tool
        self.setZValue(15)  # over the board, under ports (20) and pairs (18)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt override
        return self.tool.canvas.sceneRect()

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        painter.save()
        try:
            self.tool.paint(painter)
        finally:
            painter.restore()


class CopperTool(QObject):
    selectionChanged = Signal(object)  # tuple of Refs
    # (label, {layer: {index: pieces | None}}, selection after, merge key, {via id: new (x, y) or None})
    editRequested = Signal(str, object, object, object, object)
    snapChanged = Signal(str)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.layers: Dict[str, EditedLayer] = {}
        self.edits: Dict[str, Dict[int, List[Dict]]] = {}
        self.selection: Tuple[Ref, ...] = ()
        self.hover: Optional[Ref] = None
        self.overlay: Optional[CopperOverlay] = None
        # The project's vias by id.
        self.vias: Dict[int, object] = {}
        self.padstacks: Dict[str, object] = {}
        self.metal_layers: List[str] = []
        self.show_holes = True
        self.holes: Optional[HoleOverlay] = None
        self._holes_shown: Optional[list] = None  # what the canvas was last given to draw
        self._owned: Dict[str, Dict[int, List]] = {}  # layer -> {imported pad a via took over: []}
        self._via_ids = np.zeros(0, int)
        self._via_xy = np.zeros((0, 2))
        self._via_r = np.zeros(0)
        self._via_span: Dict[int, FrozenSet[str]] = {}
        self._radius: Dict[int, float] = {}
        # Edited copper drawn over the board until its layer is re-rendered: layer -> (before, after).
        self.pending: Dict[str, List[Tuple[List[Piece], List[Piece]]]] = {}
        self._drag: Optional[_Drag] = None
        self._press: Optional[Tuple[QPointF, List[Ref], bool]] = None  # where, what was under it, was it selected
        self._last_click: Optional[QPointF] = None
        self._hover_at: Optional[QPointF] = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(HOVER_MS)
        self._hover_timer.timeout.connect(self._update_hover)

    # ---------------------------------------------------------------- content

    def reset(self) -> None:
        """A new board: the scene was cleared, and its pieces are loaded afresh."""
        self.layers.clear()
        self.pending.clear()
        self.overlay = None
        self.holes = None
        self._holes_shown = None  # the canvas lost its holes with its scene
        self._drag = self._press = None
        self.hover = None

    def _ensure_overlay(self) -> CopperOverlay:
        if self.overlay is None:
            self.overlay = CopperOverlay(self)
            self.canvas.scene().addItem(self.overlay)
        return self.overlay

    def _ensure_holes(self) -> HoleOverlay:
        if self.holes is None:
            self.holes = HoleOverlay(self)
            self.canvas.scene().addItem(self.holes)
        return self.holes

    def refresh(self) -> None:
        if self.overlay is not None or self.selection or self.pending:
            self._ensure_overlay().update()
        if self.holes is not None and (self._drag is not None or self.holes.drawn):
            self.holes.update()

    def _layer_edits(self, name: str) -> Dict[int, List[Dict]]:
        """A layer's copper edits, with the imported pads taken over by vias marked deleted."""
        owned = self._owned.get(name)
        if not owned:
            return self.edits.get(name, {})
        return {**self.edits.get(name, {}), **owned}

    def set_layer(self, name: str, data: Dict) -> None:
        layer = EditedLayer(name, data["pieces"], bool(data.get("editable", True)), str(data.get("reason") or ""))
        layer.set_edits(self._layer_edits(name))
        self.layers[name] = layer
        self._index_vias()  # an imported pad's size counts in its via's reach
        self.refresh()

    def set_edits(self, edits: Dict[str, Dict[int, List[Dict]]]) -> None:
        self.edits = edits
        for name, layer in self.layers.items():
            layer.set_edits(self._layer_edits(name))
        self.refresh()

    def set_vias(self, vias, padstacks, metal_layers: Sequence[str]) -> None:
        """The project's vias, padstacks and copper layers. Vias None: use the drill file's holes."""
        own = vias is not None
        self.vias = {v.id: v for v in vias or []}
        self.padstacks = {p.name: p for p in padstacks}
        self.metal_layers = list(metal_layers)
        owned: Dict[str, Dict[int, List]] = {}
        for via in self.vias.values():
            for link in pstk.owned_links(via, self.metal_layers):
                owned.setdefault(link.layer, {})[link.index] = []
        changed = owned != self._owned
        self._owned = owned
        if changed:
            for name, layer in self.layers.items():
                layer.set_edits(self._layer_edits(name))
        self._index_vias()
        if own:
            holes = [(v.x_mm, v.y_mm, self.hole_mm(v)) for v in self.vias.values()]
            if holes != self._holes_shown:
                self._holes_shown = holes
                self.canvas.set_holes(holes)
        if self.vias:
            self._ensure_holes()
        self.set_selection(self.selection)
        self.refresh()

    def _index_vias(self) -> None:
        """Each via's centre, reach and layers, for picking."""
        ids, xy, radii = [], [], []
        self._via_span.clear()
        self._radius.clear()
        for via in self.vias.values():
            ps = self.padstacks.get(via.padstack)
            imported = {}
            for link in via.links:
                layer = self.layers.get(link.layer)
                if layer is not None and 0 <= link.index < len(layer.base) and layer.base[link.index].points:
                    x0, y0, x1, y1 = layer.base[link.index].bounds()
                    imported[link.layer] = max(x1 - x0, y1 - y0) / 2 + math.hypot(link.dx, link.dy)
            r = pstk.pad_radius(via, ps, self.metal_layers, imported=imported)
            if ps is not None:
                start, stop = pstk.span(ps, self.metal_layers)
                self._via_span[via.id] = frozenset(self.metal_layers[start:stop + 1])
            ids.append(via.id)
            xy.append((via.x_mm, via.y_mm))
            radii.append(r)
            self._radius[via.id] = r
        self._via_ids = np.array(ids, int)
        self._via_xy = np.array(xy, float).reshape(-1, 2)
        self._via_r = np.array(radii, float)

    def hole_mm(self, via) -> float:
        ps = self.padstacks.get(via.padstack)
        return ps.hole_mm if ps is not None else 0.3

    def layer_redrawn(self, name: str) -> None:
        """The layer has been re-rendered: drop its stand-in."""
        if self.pending.pop(name, None) is not None:
            self.refresh()

    def _via_piece(self, via) -> Piece:
        d = 2 * self._radius.get(via.id, 0.15)
        return Piece(VIA, [(via.x_mm, via.y_mm)], width=d, net=via.net, outline=[pstk.circle(0.0, 0.0, d)])

    def piece(self, ref: Ref) -> Optional[Piece]:
        if is_via(ref):
            via = self.vias.get(ref[1])
            return self._via_piece(via) if via is not None else None
        layer = self.layers.get(ref[0])
        return layer.get(ref[1:]) if layer is not None else None

    def editable(self, ref: Ref) -> bool:
        if is_via(ref):
            return ref[1] in self.vias
        layer, piece = self.layers.get(ref[0]), self.piece(ref)
        return layer is not None and layer.editable and piece is not None and piece.editable

    def edited(self, ref: Ref) -> bool:
        if is_via(ref):
            return False
        layer = self.layers.get(ref[0])
        return layer is not None and layer.edited(ref[1])

    def dragged(self) -> Dict[Ref, Piece]:
        """What a drag in progress has moved, as it is now."""
        return self._drag.preview if self._drag is not None else {}

    def set_selection(self, refs: Sequence[Ref]) -> None:
        refs = tuple(r for r in refs if self.piece(r) is not None or (r[0] not in self.layers and not is_via(r)))
        if refs != self.selection:
            self.selection = refs
            self.refresh()

    def select(self, refs: Sequence[Ref]) -> None:
        self.set_selection(refs)
        self.selectionChanged.emit(self.selection)

    # ---------------------------------------------------------------- mapping

    def _ppm(self) -> float:
        """Screen pixels per mm."""
        return max(self.canvas.base_ppm * self.canvas.zoom(), 1e-9)

    def _mm(self, pos: QPointF) -> Point:
        return self.canvas.to_mm(self.canvas.mapToScene(pos.toPoint()))

    def _to_view(self) -> QTransform:
        return self.canvas._mm_to_scene() * self.canvas.viewportTransform()

    def _visible_layers(self) -> List[str]:
        """Loaded copper layers shown on the board, top first."""
        names = [n for n in self.canvas._layers if n != "edge_cuts"]
        return [n for n in reversed(names) if n in self.layers and self.canvas.visible_layers.get(n, True)]

    # ---------------------------------------------------------------- picking

    def pick(self, x: float, y: float) -> List[Ref]:
        """The pieces under a point, best first: traces and pads before polygons."""
        tol = PICK_PX / self._ppm()
        strokes, areas = [], []
        visible = self._visible_layers()
        for name in visible:
            for pid in self.layers[name].pick(x, y, tol):
                ref = (name, pid[0], pid[1])
                (areas if self.piece(ref).kind == REGION else strokes).append(ref)
        return self._pick_vias(x, y, tol, set(visible)) + strokes + areas

    def _pick_vias(self, x: float, y: float, tol: float, visible) -> List[Ref]:
        """Vias at the point on a shown layer, nearest first."""
        if not len(self._via_ids) or not self.canvas.board:
            return []
        d = np.hypot(self._via_xy[:, 0] - x, self._via_xy[:, 1] - y)
        hits = np.nonzero(d <= self._via_r + tol)[0]
        out = []
        for i in hits[np.argsort(d[hits])]:
            via_id = int(self._via_ids[i])
            span = self._via_span.get(via_id)
            if span is None or span & visible or not visible:
                out.append((VIA_REF, via_id, 0))
        return out

    def handles(self) -> List[Handle]:
        """The selected piece's handles; a selection of several has none (it moves as one)."""
        if len(self.selection) != 1 or not self.editable(self.selection[0]):
            return []
        ref = self.selection[0]
        piece = self.piece(ref)
        if piece.kind == TRACE:
            a, b, h = piece.edges()[0]
            return [Handle("end", ref, 0, a), Handle("end", ref, 1, b),
                    Handle("bend", ref, 0, edge_point(a, b, h, 0.5))]
        if piece.kind == REGION:
            out = [Handle("vertex", ref, i, p) for i, p in enumerate(piece.points)]
            if len(piece.points) <= EDGE_HANDLES_UP_TO:
                out += [Handle("edge", ref, i, edge_point(a, b, h, 0.5)) for i, (a, b, h) in enumerate(piece.edges())]
            return out
        return []

    def handle_at(self, pos: QPointF) -> Optional[Handle]:
        t = self._to_view()
        best, best_d = None, GRAB_PX
        for handle in self.handles():
            p = t.map(QPointF(*handle.point))
            d = max(abs(p.x() - pos.x()), abs(p.y() - pos.y()))
            # Ends and vertices win over the smaller handles beside them.
            if d <= best_d and (best is None or d < best_d or handle.kind in ("end", "vertex")):
                best, best_d = handle, d
        return best

    def item_at(self, pos: QPointF) -> Tuple[Optional[Ref], Optional[Handle], Point]:
        """What a right click is on: (piece, handle, point in mm)."""
        handle = self.handle_at(pos)
        point = self._mm(pos)
        if handle is not None:
            return handle.ref, handle, point
        under = self.pick(*point)
        selected = [r for r in under if r in self.selection]
        return (selected or under or [None])[0], None, point

    # ---------------------------------------------------------------- mouse

    def press(self, pos: QPointF, modifiers) -> bool:
        """A left press in the select tool, on no port. True when it was on copper."""
        self._hover_timer.stop()
        handle = self.handle_at(pos)
        if handle is not None:
            self._start_handle(handle, self._mm(pos))
            return True
        under = self.pick(*self._mm(pos))
        if not under:
            self._press = None
            return False
        if modifiers & Qt.ControlModifier:
            top = under[0]
            self.select(tuple(r for r in self.selection if r != top) if top in self.selection
                        else self.selection + (top,))
            return True
        was_selected = any(r in self.selection for r in under)
        if not was_selected:
            self.select((under[0],))
        self._press = (QPointF(pos), under, was_selected)
        return True

    def move(self, pos: QPointF) -> bool:
        """The mouse moved with the button down; True while the tool has the press."""
        if self._drag is not None:
            self._update_drag(self._mm(pos))
            return True
        if self._press is None:
            return False
        start = self._press[0]
        if max(abs(pos.x() - start.x()), abs(pos.y() - start.y())) >= DRAG_PX:
            self._start_body(self._mm(start))
            if self._drag is not None:
                self._update_drag(self._mm(pos))
        return True

    def release(self, pos: QPointF) -> bool:
        if self._drag is not None:
            drag, self._drag = self._drag, None
            self.canvas.hide_snap()
            if drag.moved:
                self._commit(drag)
            self.refresh()
            return True
        if self._press is None:
            return False
        start, under, was_selected = self._press
        self._press = None
        last, self._last_click = self._last_click, QPointF(pos)
        again = last is not None and max(abs(pos.x() - last.x()), abs(pos.y() - last.y())) < DRAG_PX
        if was_selected and again and len(under) > 1:
            # A second click on the same spot: the next thing under it.
            current = next((i for i, r in enumerate(under) if r in self.selection), -1)
            self.select((under[(current + 1) % len(under)],))
        elif was_selected and len(self.selection) > 1:
            self.select((next(r for r in under if r in self.selection),))
        return True

    def cancel(self) -> bool:
        """Esc: drop a drag in progress, else the selection."""
        if self._drag is not None:
            self._drag = None
            self.canvas.hide_snap()
            self.refresh()
            return True
        if self.selection:
            self.select(())
            return True
        return False

    def hover_moved(self, pos: QPointF) -> None:
        self._hover_at = QPointF(pos)
        self._hover_timer.start()
        handle = self.handle_at(pos)
        shape = (Qt.ArrowCursor if handle is None else
                 Qt.SizeAllCursor if handle.kind in ("end", "vertex") else Qt.CrossCursor)
        if self.canvas.cursor().shape() != shape:
            self.canvas.setCursor(shape)

    def hover_left(self) -> None:
        self._hover_timer.stop()
        self._hover_at = None
        if self.hover is not None:
            self.hover = None
            self.refresh()

    def _update_hover(self) -> None:
        if self._hover_at is None or self._drag is not None:
            return
        under = self.pick(*self._mm(self._hover_at)) if self.layers else []
        hover = under[0] if under else None
        if hover != self.hover:
            self.hover = hover
            self.refresh()

    # ---------------------------------------------------------------- dragging

    def _joints(self, ref: Ref, point: Point, moving: Sequence[Ref]) -> List[Tuple[Ref, int]]:
        layer = self.layers[ref[0]]
        skip = [(r[1], r[2]) for r in moving if r[0] == ref[0]]
        return [((ref[0], pid[0], pid[1]), end) for pid, end in layer.trace_ends(point, skip)
                if self.editable((ref[0], pid[0], pid[1]))]

    def pad_layers(self, via) -> List[str]:
        """The loaded layers the via has a pad on."""
        ps = self.padstacks.get(via.padstack)
        if ps is None:
            return []
        return [row.layer for row in pstk.via_layers(via, ps, self.metal_layers)
                if row.has_pad and row.layer in self.layers]

    def _joints_of(self, ref: Ref, piece: Piece, moving: Sequence[Ref]) -> List[Tuple[Ref, int]]:
        """The trace ends that follow a moved piece or via."""
        if is_via(ref):
            via = self.vias.get(ref[1])
            if via is None:
                return []
            out = []
            for name in self.pad_layers(via):
                out += self._joints((name, -1, 0), piece.points[0], moving)
            return out
        if piece.kind != TRACE:
            return []
        return [joint for p in piece.points for joint in self._joints(ref, p, moving)]

    def _start_handle(self, handle: Handle, at: Point) -> None:
        piece = self.piece(handle.ref)
        drag = _Drag(handle.kind, handle.ref, handle.i, at, handle.point, {handle.ref: piece})
        if handle.kind == "end":
            drag.joints = self._joints(handle.ref, handle.point, [handle.ref])
        elif handle.kind == "edge":
            # Insert a new vertex mid-edge, then drag it.
            a, b, h = piece.edges()[handle.i]
            h1, h2 = split_height(h, a, b, 0.5)
            points = list(piece.points)
            points.insert(handle.i + 1, handle.point)
            heights = list(piece.heights) + [0.0] * (len(piece.points) - len(piece.heights))
            heights[handle.i:handle.i + 1] = [h1, h2]
            drag.pieces = {handle.ref: replace(piece, points=points, heights=heights)}
            drag.kind, drag.i = "vertex", handle.i + 1
            drag.preview = dict(drag.pieces)
            drag.moved = True  # the vertex is new even if it is not moved
        for ref, end in drag.joints:
            drag.pieces[ref] = self.piece(ref)
        self._drag = drag
        self.hover = None

    def _start_body(self, at: Point) -> None:
        moving = [r for r in self.selection if self.editable(r)]
        if not moving:
            return
        pieces = {r: self.piece(r) for r in moving}
        points = [p for piece in pieces.values() for p in piece.points]
        anchor = min(points, key=lambda p: math.hypot(p[0] - at[0], p[1] - at[1]))
        drag = _Drag("body", None, 0, at, anchor, dict(pieces))
        for ref, piece in pieces.items():
            for joint in self._joints_of(ref, piece, moving):
                if joint not in drag.joints:
                    drag.joints.append(joint)
        for ref, _end in drag.joints:
            drag.pieces.setdefault(ref, self.piece(ref))
        self._drag = drag
        self.hover = None

    def _snapped(self, p: Point, own: Sequence[Point]) -> Point:
        """p snapped to the nearest target, unless that target is the dragged copper itself."""
        hit = self.canvas.snap_point(*p)
        tol = 1.5 / self._ppm()
        if hit is None or any(abs(hit.x_mm - q[0]) <= tol and abs(hit.y_mm - q[1]) <= tol for q in own):
            self.canvas.hide_snap()
            return p
        self.canvas.show_snap(hit)
        return hit.x_mm, hit.y_mm

    def _update_drag(self, at: Point) -> None:
        drag = self._drag
        piece = drag.pieces.get(drag.ref) if drag.ref is not None else None
        preview: Dict[Ref, Piece] = {}
        if drag.kind == "body":
            raw = (drag.anchor[0] + at[0] - drag.start[0], drag.anchor[1] + at[1] - drag.start[1])
            if _shift():
                raw = _octilinear(drag.anchor, raw)
            own = [p for r, q in drag.pieces.items() for p in q.points]
            target = self._snapped(raw, own)
            dx, dy = target[0] - drag.anchor[0], target[1] - drag.anchor[1]
            for ref in self.selection:
                if ref in drag.pieces:
                    preview[ref] = drag.pieces[ref].moved(dx, dy)
            for ref, end in drag.joints:
                old = drag.pieces[ref]
                points = list(preview.get(ref, old).points)
                points[end] = (old.points[end][0] + dx, old.points[end][1] + dy)
                preview[ref] = replace(preview.get(ref, old), points=points)
            moved = abs(dx) + abs(dy) > 1e-9
        elif drag.kind == "end":
            other = piece.points[1 - drag.i]
            raw = _octilinear(other, at) if _shift() else at
            target = self._snapped(raw, [drag.anchor])
            points = list(piece.points)
            points[drag.i] = target
            preview[drag.ref] = replace(piece, points=points)
            for ref, end in drag.joints:
                joined = list(drag.pieces[ref].points)
                joined[end] = target
                preview[ref] = replace(drag.pieces[ref], points=joined)
            moved = target != drag.anchor
        elif drag.kind == "bend":
            a, b = piece.points
            h = height_through(a, b, at)
            if abs(h) * self._ppm() < STRAIGHT_PX:
                h = 0.0
            preview[drag.ref] = replace(piece, heights=[h])
            moved = abs(h - (piece.heights[0] if piece.heights else 0.0)) > 1e-9
        else:  # a polygon's vertex
            n = len(piece.points)
            target = self._snapped(at, [piece.points[drag.i]])
            if _shift():
                target = _octilinear(piece.points[(drag.i - 1) % n], target)
            points = list(piece.points)
            points[drag.i] = target
            preview[drag.ref] = replace(piece, points=points)
            moved = drag.moved or target != piece.points[drag.i]
        drag.preview = preview
        drag.moved = moved
        self.snapChanged.emit(self._drag_text(drag))
        self.refresh()

    def _drag_text(self, drag: _Drag) -> str:
        if drag.kind == "bend":
            h = drag.preview[drag.ref].heights[0]
            return "straight" if not h else f"arc height {h * 1000:.1f} µm"
        if drag.kind == "body":
            p = next(iter(drag.preview.values())).points[0] if drag.preview else drag.anchor
            q = next(iter(drag.pieces.values())).points[0]
            return f"move Δ {(p[0] - q[0]):.4f}, {(p[1] - q[1]):.4f} mm"
        piece = drag.preview.get(drag.ref)
        if piece is None:
            return ""
        p = piece.points[drag.i]
        text = f"{p[0]:.4f}, {p[1]:.4f} mm"
        if drag.kind == "end":
            text += f" · length {piece.length() * 1000:.1f} µm"
        return text

    def _commit(self, drag: _Drag) -> None:
        labels = {"body": "Move copper", "end": "Move trace end", "bend": "Bend trace", "vertex": "Move vertex"}
        label = labels[drag.kind]
        if drag.kind == "vertex" and drag.ref is not None and \
                len(drag.preview[drag.ref].points) != len(self.piece(drag.ref).points):
            label = "Add vertex"
        self.apply(label, {ref: [piece] for ref, piece in drag.preview.items()}, self.selection)

    # ---------------------------------------------------------------- edits

    def apply(self, label: str, new: Dict[Ref, Optional[List[Piece]]], selection_after: Optional[Sequence[Ref]],
              merge_key=None) -> None:
        """Replace each ref's piece with a list of pieces ([] deletes, None restores the original)."""
        changes: Dict[str, Dict[int, Optional[List[Dict]]]] = {}
        shifts: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}  # (layer, index) -> [(sub, pieces it became)]
        before_after: Dict[str, List[Tuple[List[Piece], List[Piece]]]] = {}
        vias: Dict[int, Optional[Point]] = {}
        for ref in [r for r in new if is_via(r)]:
            pieces = new.pop(ref)
            via = self.vias.get(ref[1])
            if via is None or pieces is None:
                continue
            vias[via.id] = tuple(pieces[0].points[0]) if pieces else None
            for name, before, after in self.via_pads(via, vias[via.id]):
                before_after.setdefault(name, []).append((before, after))
        for (name, index), refs in _group(new).items():
            layer = self.layers[name]
            if any(new[r] is None for r in refs):
                changes.setdefault(name, {})[index] = None
                before_after.setdefault(name, []).append((layer.pieces_of(index), [layer.base[index]]))
                continue
            pieces = layer.pieces_of(index)
            out: List[Piece] = []
            for sub, piece in enumerate(pieces):
                ref = (name, index, sub)
                if ref in new:
                    out.extend(new[ref])
                    shifts.setdefault((name, index), []).append((sub, len(new[ref])))
                else:
                    out.append(piece)
            changes.setdefault(name, {})[index] = [p.as_edit() for p in out]
            before_after.setdefault(name, []).append((pieces, out))
        after = None
        if selection_after is not None:
            after = []
            for name, index, sub in selection_after:
                spans = shifts.get((name, index))
                if spans is None:
                    after.append((name, index, sub))
                    continue
                offset = 0
                for s, count in sorted(spans):
                    if s < sub:
                        offset += count - 1
                    elif s == sub:
                        after.extend((name, index, sub + offset + k) for k in range(count))
                        break
                else:
                    after.append((name, index, sub + offset))
            after = tuple(after)
        for name, pairs in before_after.items():
            self.pending.setdefault(name, []).extend(pairs)
        self.editRequested.emit(label, changes, after, merge_key, vias)

    def via_pads(self, via, to: Optional[Point]) -> List[Tuple[str, List[Piece], List[Piece]]]:
        """A via's pads per layer: (layer, current pads, pads at `to`)."""
        ps = self.padstacks.get(via.padstack)
        out = []
        if ps is None:
            return out
        for row in pstk.via_layers(via, ps, self.metal_layers):
            layer = self.layers.get(row.layer)
            if layer is None or not row.has_pad:
                continue
            if row.link is not None and 0 <= row.link < len(layer.base):
                outline = layer.base[row.link].outline
            elif row.pad_mm:
                outline = [pstk.circle(0.0, 0.0, row.pad_mm)]
            else:
                continue
            dx, dy = row.link_offset
            pad = Piece(PAD, [(via.x_mm + dx, via.y_mm + dy)], net=via.net, outline=outline)
            moved = [] if to is None else [replace(pad, points=[(to[0] + dx, to[1] + dy)])]
            out.append((row.layer, [pad], moved))
        return out

    def _joined_ends(self, ref: Ref, end: int, target: Point) -> Dict[Ref, List[Piece]]:
        """A trace end moved to `target`, with the ends joined to it."""
        piece = self.piece(ref)
        out: Dict[Ref, List[Piece]] = {}
        points = list(piece.points)
        old, points[end] = points[end], target
        out[ref] = [replace(piece, points=points)]
        for joint, joint_end in self._joints(ref, old, [ref]):
            other = self.piece(joint)
            moved = list(other.points)
            moved[joint_end] = target
            out[joint] = [replace(other, points=moved)]
        return out

    def move_end(self, ref: Ref, end: int, target: Point) -> None:
        self.apply("Move trace end", self._joined_ends(ref, end, target), self.selection,
                   merge_key=("end", ref, end))

    def set_vertex(self, ref: Ref, i: int, target: Point) -> None:
        piece = self.piece(ref)
        points = list(piece.points)
        points[i] = target
        self.apply("Move vertex", {ref: [replace(piece, points=points)]}, self.selection,
                   merge_key=("vertex", ref, i))

    def move_by(self, refs: Sequence[Ref], dx: float, dy: float, label: str = "Move copper") -> None:
        new: Dict[Ref, List[Piece]] = {}
        moving = [r for r in refs if self.editable(r)]
        for ref in moving:
            new[ref] = [self.piece(ref).moved(dx, dy)]
        for ref in moving:
            for joint, end in self._joints_of(ref, self.piece(ref), moving):
                other = new[joint][0] if joint in new else self.piece(joint)
                points = list(other.points)
                points[end] = (points[end][0] + dx, points[end][1] + dy)
                new[joint] = [replace(other, points=points)]
        self.apply(label, new, self.selection)

    def set_width(self, refs: Sequence[Ref], width_mm: float) -> None:
        new = {r: [replace(self.piece(r), width=width_mm)] for r in refs
               if self.editable(r) and self.piece(r).kind == TRACE}
        if new:
            self.apply("Trace width", new, self.selection, merge_key=("width",) + tuple(sorted(new)))

    def set_height(self, ref: Ref, height: float) -> None:
        self.apply("Bend trace" if height else "Straighten trace", {ref: [replace(self.piece(ref), heights=[height])]},
                   self.selection, merge_key=("height", ref))

    def split(self, ref: Ref, at: Optional[Point] = None) -> None:
        """Split a trace segment in two at `at` (its middle by default)."""
        piece = self.piece(ref)
        a, b, h = piece.edges()[0]
        t = 0.5 if at is None else project_on_edge(a, b, h, at)[0]
        t = min(max(t, 0.02), 0.98)
        m = edge_point(a, b, h, t)
        h1, h2 = split_height(h, a, b, t)
        first, second = replace(piece, points=[a, m], heights=[h1]), replace(piece, points=[m, b], heights=[h2])
        self.apply("Split trace", {ref: [first, second]}, (ref,))

    def insert_vertex(self, ref: Ref, at: Point) -> None:
        piece = self.piece(ref)
        edges = piece.edges()
        i = min(range(len(edges)), key=lambda k: project_on_edge(*edges[k], at)[1])
        a, b, h = edges[i]
        t = project_on_edge(a, b, h, at)[0]
        h1, h2 = split_height(h, a, b, t)
        points = list(piece.points)
        points.insert(i + 1, edge_point(a, b, h, t))
        heights = list(piece.heights) + [0.0] * (len(piece.points) - len(piece.heights))
        heights[i:i + 1] = [h1, h2]
        self.apply("Add vertex", {ref: [replace(piece, points=points, heights=heights)]}, self.selection)

    def delete_vertex(self, ref: Ref, i: int) -> None:
        piece = self.piece(ref)
        if len(piece.points) <= 3:
            return
        points = list(piece.points)
        del points[i]
        heights = list(piece.heights) + [0.0] * (len(piece.points) - len(piece.heights))
        # The edges either side of the vertex become one, straight.
        heights[(i - 1) % len(heights)] = 0.0
        del heights[i]
        self.apply("Delete vertex", {ref: [replace(piece, points=points, heights=heights)]}, self.selection)

    def delete(self, refs: Sequence[Ref]) -> None:
        new = {r: [] for r in refs if self.editable(r)}
        if new:
            self.apply("Delete copper", new, ())

    def restore(self, refs: Sequence[Ref]) -> None:
        """Put back the imported objects the pieces came from."""
        new = {r: None for r in refs if self.edited(r)}
        if new:
            self.apply("Restore imported copper", new, tuple(sorted({(r[0], r[1], 0) for r in new})))

    def select_chain(self, ref: Ref) -> None:
        layer = self.layers.get(ref[0])
        if layer is not None:
            self.select(tuple((ref[0], i, s) for i, s in layer.chain(ref[1:])))

    # ---------------------------------------------------------------- drawing

    def _path(self, piece: Piece, t: QTransform) -> QPainterPath:
        """A piece's area in view coordinates."""
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        if piece.kind == TRACE:
            line = QPainterPath()
            pts = [t.map(QPointF(*p)) for p in piece.polyline()]
            line.moveTo(pts[0])
            for p in pts[1:]:
                line.lineTo(p)
            if len(pts) == 2 and pts[0] == pts[1]:
                line.lineTo(pts[1] + QPointF(1e-3, 0))  # a dot still strokes round
            stroker = QPainterPathStroker()
            stroker.setCapStyle(Qt.RoundCap)
            stroker.setJoinStyle(Qt.RoundJoin)
            stroker.setWidth(max(piece.width * abs(t.m11()), 1.0))
            return stroker.createStroke(line).simplified()
        polys = piece.pad_polygons() if piece.kind in (PAD, VIA) else [piece.polyline()]
        for poly in polys:
            if len(poly) >= 3:
                path.addPolygon(QPolygonF([t.map(QPointF(*p)) for p in poly]))
                path.closeSubpath()
        return path

    def _visible(self, piece: Piece, view: QRectF, t: QTransform) -> bool:
        x0, y0, x1, y1 = piece.bounds()
        return t.mapRect(QRectF(x0, y0, x1 - x0, y1 - y0)).intersects(view)

    def paint(self, painter: QPainter) -> None:
        t = self.canvas._mm_to_scene() * painter.worldTransform()
        painter.setWorldTransform(QTransform())
        painter.setRenderHint(QPainter.Antialiasing)
        view = QRectF(self.canvas.viewport().rect()).adjusted(-50, -50, 50, 50)
        c = theme.colors()
        ink = palette.PortInk()
        select = QColor(ink.select)
        backing = QBrush(QColor(c["sunken"]))
        shown = set(self._visible_layers())

        # Erase the old shape and draw the new one over the board.
        previews: List[Tuple[str, List[Piece], List[Piece]]] = []
        for name, pairs in self.pending.items():
            for before, after in pairs:
                previews.append((name, before, after))
        drag = self._drag
        if drag is not None and drag.preview:
            for ref, piece in drag.preview.items():
                if is_via(ref):
                    via = self.vias.get(ref[1])
                    if via is not None:
                        previews.extend(self.via_pads(via, piece.points[0]))
                else:
                    previews.append((ref[0], [drag.pieces[ref]], [piece]))
        for name, before, after in previews:
            if name not in shown:
                continue
            fill = QColor(self.canvas._layer_color(name))
            fill.setAlphaF(0.92)
            for piece in before:
                if piece.points and self._visible(piece, view, t):
                    painter.fillPath(self._path(piece, t), backing)
            for piece in after:
                if piece.points and self._visible(piece, view, t):
                    painter.fillPath(self._path(piece, t), QBrush(fill))

        dragged = drag.preview if drag is not None else {}
        if self.hover is not None and self.hover not in self.selection and (self.hover[0] in shown
                                                                              or is_via(self.hover)):
            piece = self.piece(self.hover)
            if piece is not None:
                hover = QColor(select)
                hover.setAlphaF(0.75)
                painter.strokePath(self._path(piece, t), QPen(hover, 1.2, Qt.DashLine))

        tint = QColor(select)
        tint.setAlphaF(0.22)
        outline = QPen(select, 1.6)
        halo = QPen(ink.halo, 3.6)
        for ref in self.selection:
            piece = dragged.get(ref) or self.piece(ref)
            if piece is None or not piece.points or not (ref[0] in shown or is_via(ref)) \
                    or not self._visible(piece, view, t):
                continue
            path = self._path(piece, t)
            painter.fillPath(path, tint)
            painter.strokePath(path, halo)
            painter.strokePath(path, outline)
            if piece.kind == TRACE:  # the centreline: where the ends and bend are
                line = QPainterPath()
                pts = [t.map(QPointF(*p)) for p in piece.polyline()]
                line.moveTo(pts[0])
                for p in pts[1:]:
                    line.lineTo(p)
                painter.strokePath(line, QPen(select, 1.0, Qt.DotLine))
        for ref in dragged:  # the joined segments that follow a drag
            if ref not in self.selection and ref[0] in shown:
                painter.strokePath(self._path(dragged[ref], t), QPen(select, 1.0, Qt.DashLine))

        handles = self.handles()
        if drag is not None and drag.ref is not None and drag.ref in dragged:
            # Handles follow the shape being dragged.
            moved = dragged[drag.ref]
            handles = [replace(h, point=moved.points[h.i]) if h.kind in ("end", "vertex") and h.i < len(moved.points)
                       else h for h in handles]
            if drag.kind == "bend" or (drag.kind == "vertex" and len(moved.points) != len(self.piece(drag.ref).points)):
                handles = []
        self._paint_handles(painter, handles, t, view, ink, select)

    def _paint_handles(self, painter: QPainter, handles: List[Handle], t: QTransform, view: QRectF, ink,
                       select: QColor) -> None:
        s = HANDLE_PX / 2
        painter.setPen(QPen(select, 1.4))
        for handle in handles:
            p = t.map(QPointF(*handle.point))
            if not view.contains(p):
                continue
            if handle.kind in ("end", "vertex"):
                painter.setBrush(ink.paper)
                painter.drawRect(QRectF(p.x() - s, p.y() - s, 2 * s, 2 * s))
            elif handle.kind == "bend":
                painter.setBrush(select)
                painter.drawPolygon(QPolygonF([p + QPointF(0, -s - 1), p + QPointF(s + 1, 0),
                                               p + QPointF(0, s + 1), p + QPointF(-s - 1, 0)]))
            else:
                painter.setBrush(ink.paper)
                painter.drawEllipse(p, s - 1.5, s - 1.5)


def _group(new: Dict[Ref, object]) -> Dict[Tuple[str, int], List[Ref]]:
    out: Dict[Tuple[str, int], List[Ref]] = {}
    for ref in new:
        out.setdefault((ref[0], ref[1]), []).append(ref)
    return out
