"""The inspector page for selected copper: a trace's width, ends and arc height, a polygon's vertices, a pad's centre."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from EMShrimulator.app.views.layout.copper_tool import Ref, describe, is_via
from EMShrimulator.app.widgets.form import form_layout, hint
from EMShrimulator.app.widgets.unit_spinbox import UnitSpinBox
from EMShrimulator.core.layout.copper_geometry import PAD, REGION, TRACE, Piece, arc_center

VERTEX_TABLE_UP_TO = 64  # larger polygons (pours) are reshaped on the board only


def _row(*widgets: QWidget) -> QWidget:
    host = QWidget()
    layout = QHBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    return host


def _button(text: str, tip: str, slot, role: str = "") -> QPushButton:
    button = QPushButton(text)
    button.setToolTip(tip)
    if role:
        button.setProperty("role", role)
    button.clicked.connect(slot)
    return button


class CopperProperties(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.tool = None  # the canvas's CopperTool, set by the layout view
        self.refs: Tuple[Ref, ...] = ()

        self.title = QLabel("Copper")
        self.title.setProperty("role", "heading")
        self.where = hint()
        self.net = QLabel()
        self.net.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.width = UnitSpinBox("µm", scale=1e-3, decimals=2, minimum=0.1, step=5,
                                 tooltip="The trace's width. Its ends and shape stay as they are.")
        self.width.committed.connect(self._set_width)
        self.chain_width = _button("Width of whole trace", "Give every segment joined end to end with this one "
                                   "this width", self._chain_width)
        self.start_x = UnitSpinBox("mm", decimals=4, step=0.1)
        self.start_y = UnitSpinBox("mm", decimals=4, step=0.1)
        self.end_x = UnitSpinBox("mm", decimals=4, step=0.1)
        self.end_y = UnitSpinBox("mm", decimals=4, step=0.1)
        for box, end in ((self.start_x, 0), (self.start_y, 0), (self.end_x, 1), (self.end_y, 1)):
            box.committed.connect(lambda _v, end=end: self._move_end(end))
        self.height = UnitSpinBox("µm", scale=1e-3, decimals=2, step=10,
                                  tooltip="How far the segment's middle bows from the straight line between its "
                                          "ends: + to the left going start → end, − to the right, 0 straight")
        self.height.committed.connect(self._set_height)
        self.length = QLabel()
        self.arc = QLabel()
        self.centre_x = UnitSpinBox("mm", decimals=4, step=0.1)
        self.centre_y = UnitSpinBox("mm", decimals=4, step=0.1)
        self.centre_x.committed.connect(self._move_centre)
        self.centre_y.committed.connect(self._move_centre)
        self.shape = QLabel()
        self.vertices = QTableWidget(0, 2)
        self.vertices.setHorizontalHeaderLabels(["X (mm)", "Y (mm)"])
        self.vertices.verticalHeader().setDefaultSectionSize(22)
        self.vertices.horizontalHeader().setStretchLastSection(True)
        self.vertices.setMinimumHeight(140)
        self.vertices.itemChanged.connect(self._vertex_edited)
        self.vertex_hint = hint("Drag a vertex's handle on the board, or drag a mid-edge handle to add one. "
                                "Right-click a vertex to delete it.")
        self.dx = UnitSpinBox("mm", decimals=4, step=0.1)
        self.dy = UnitSpinBox("mm", decimals=4, step=0.1)
        move = _button("Move", "Move the selection by ΔX, ΔY; joined trace segments stretch to follow", self._move_by)
        self.move_row = _row(QLabel("Δ"), self.dx, self.dy, move)

        self.split = _button("Split", "Make this segment two, joined at its middle, each with ends of its own",
                             lambda: self.tool.split(self.refs[0]))
        self.straighten = _button("Straighten", "Make this arc a straight segment between the same ends",
                                  lambda: self.tool.set_height(self.refs[0], 0.0))
        self.whole = _button("Whole trace", "Select every segment joined end to end with this one",
                             lambda: self.tool.select_chain(self.refs[0]))
        self.restore = _button("Restore imported", "Put back the copper this came from as it was imported",
                               lambda: self.tool.restore(self.refs))
        self.delete = _button("Delete", "Remove this copper from the board", lambda: self.tool.delete(self.refs),
                              role="danger")
        self.trace_actions = _row(self.split, self.straighten, self.whole)
        self.edit_row = _row(self.restore, self.delete)
        self.note = hint(role="note")

        form = form_layout()
        self.form = form
        form.addRow(self.where)
        form.addRow("Net", self.net)
        form.addRow("Width", self.width)
        form.addRow("", self.chain_width)
        form.addRow("Start X", self.start_x)
        form.addRow("Start Y", self.start_y)
        form.addRow("End X", self.end_x)
        form.addRow("End Y", self.end_y)
        form.addRow("Arc height", self.height)
        form.addRow("Length", self.length)
        form.addRow("Arc", self.arc)
        form.addRow("Centre X", self.centre_x)
        form.addRow("Centre Y", self.centre_y)
        form.addRow("Shape", self.shape)
        form.addRow(self.vertices)
        form.addRow(self.vertex_hint)
        form.addRow("Move by", self.move_row)
        form.addRow("", self.trace_actions)
        form.addRow(self.note)
        form.addRow("", self.edit_row)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.addWidget(self.title)
        layout.addLayout(form)
        self._loading = False

    # ---------------------------------------------------------------- edits

    def _pieces(self) -> List[Piece]:
        return [p for p in (self.tool.piece(r) for r in self.refs) if p is not None] if self.tool else []

    def _set_width(self, width_mm: float) -> None:
        if self.tool is not None and width_mm > 0:
            self.tool.set_width(self.refs, width_mm)

    def _chain_width(self) -> None:
        if self.tool is None or not self.refs:
            return
        layer = self.tool.layers.get(self.refs[0][0])
        if layer is not None:
            chain = [(self.refs[0][0], i, s) for i, s in layer.chain(self.refs[0][1:])]
            self.tool.set_width(chain, self.width.stored())
            self.tool.select(chain)

    def _move_end(self, end: int) -> None:
        if self.tool is None or len(self.refs) != 1:
            return
        x, y = (self.start_x, self.start_y) if end == 0 else (self.end_x, self.end_y)
        self.tool.move_end(self.refs[0], end, (x.stored(), y.stored()))

    def _set_height(self, height_mm: float) -> None:
        if self.tool is not None and len(self.refs) == 1:
            self.tool.set_height(self.refs[0], height_mm)

    def _move_centre(self, _value: float) -> None:
        pieces = self._pieces()
        if len(pieces) == 1 and pieces[0].kind == PAD:
            cx, cy = pieces[0].points[0]
            self.tool.move_by(self.refs, self.centre_x.stored() - cx, self.centre_y.stored() - cy, "Move pad")

    def _move_by(self) -> None:
        if self.tool is not None and (self.dx.stored() or self.dy.stored()):
            self.tool.move_by(self.refs, self.dx.stored(), self.dy.stored())

    def _vertex_edited(self, item: QTableWidgetItem) -> None:
        if self._loading or self.tool is None or len(self.refs) != 1:
            return
        piece = self.tool.piece(self.refs[0])
        row = item.row()
        if piece is None or piece.kind != REGION or row >= len(piece.points):
            return
        try:
            value = float(item.text())
        except ValueError:
            self.load(self.refs)
            return
        x, y = piece.points[row]
        self.tool.set_vertex(self.refs[0], row, (value, y) if item.column() == 0 else (x, value))

    # ---------------------------------------------------------------- showing

    def _show(self, widget: QWidget, visible: bool) -> None:
        self.form.setRowVisible(widget, visible)

    def load(self, refs: Tuple[Ref, ...]) -> None:
        self.refs = tuple(refs)
        pieces = self._pieces()
        tool = self.tool
        if not pieces:
            self.title.setText("Copper")
            self.where.setText("Loading the board's copper…" if tool is not None else "")
            for widget in self._rows():
                self._show(widget, False)
            return
        kinds = {p.kind for p in pieces}
        single = pieces[0] if len(pieces) == 1 else None
        layers = sorted({"vias" if is_via(r) else r[0] for r in self.refs})
        editable = all(tool.editable(r) for r in self.refs)
        edited = any(tool.edited(r) for r in self.refs)
        if single is not None:
            self.title.setText(f"{describe(single)} · {layers[0]}")
        else:
            noun = {TRACE: "trace segments", REGION: "polygons", PAD: "pads"}.get(next(iter(kinds)), "")                 if len(kinds) == 1 else "copper pieces"
            self.title.setText(f"{len(pieces)} {noun} · {', '.join(layers)}")
        where = "Edited" if edited else "As imported"
        self.where.setText(where + ("" if editable else " · read-only"))
        nets = sorted({p.net or "no net" for p in pieces})
        self.net.setText(", ".join(nets[:4]) + (f" and {len(nets) - 4} more" if len(nets) > 4 else ""))

        traces = kinds == {TRACE}
        trace = single is not None and single.kind == TRACE
        region = single is not None and single.kind == REGION
        pad = single is not None and single.kind == PAD
        self._show(self.width, traces)
        self._show(self.chain_width, trace)
        for box in (self.start_x, self.start_y, self.end_x, self.end_y, self.height):
            self._show(box, trace)
        self._show(self.length, traces)
        self._show(self.arc, False)
        self._show(self.centre_x, pad)
        self._show(self.centre_y, pad)
        self._show(self.shape, region or pad)
        self._show(self.vertices, region and len(single.points) <= VERTEX_TABLE_UP_TO)
        self._show(self.vertex_hint, region)
        self._show(self.move_row, not trace)
        self._show(self.trace_actions, trace)
        self._show(self.net, True)
        self._show(self.note, True)
        self._show(self.edit_row, True)
        self.restore.setVisible(edited)

        if traces:
            widths = {round(p.width, 9) for p in pieces}
            self.width.set_stored(pieces[0].width if len(widths) == 1 else None)
            self.width.setSpecialValueText("" if len(widths) == 1 else "mixed")
            total = sum(p.length() for p in pieces)
            self.length.setText(f"{total * 1000:.1f} µm" + ("" if trace else f" in all, {len(pieces)} segments"))
        if trace:
            (ax, ay), (bx, by) = single.points
            h = single.heights[0] if single.heights else 0.0
            self.start_x.set_stored(ax)
            self.start_y.set_stored(ay)
            self.end_x.set_stored(bx)
            self.end_y.set_stored(by)
            self.height.set_stored(h)
            center = arc_center(single.points[0], single.points[1], h)
            if center is not None:
                sweep = single.length() / center[2]
                self.arc.setText(f"radius {center[2] * 1000:.1f} µm · {math.degrees(sweep):.1f}° "
                                 f"{'clockwise' if h > 0 else 'anticlockwise'}")
            self._show(self.arc, center is not None)
            self.straighten.setVisible(center is not None)
        if pad:
            cx, cy = single.points[0]
            self.centre_x.set_stored(cx)
            self.centre_y.set_stored(cy)
            x0, y0, x1, y1 = single.bounds()
            self.shape.setText(f"{(x1 - x0) * 1000:.0f} × {(y1 - y0) * 1000:.0f} µm, as imported")
        if region:
            arcs = sum(1 for h in single.heights if abs(h) > 1e-9)
            self.shape.setText(f"{len(single.points)} vertices{f', {arcs} arc edges' if arcs else ''} · "
                               f"area {single.area():.4f} mm²" + ("" if single.dark else " · clearance"))
            if len(single.points) <= VERTEX_TABLE_UP_TO:
                self._loading = True
                self.vertices.setRowCount(len(single.points))
                for i, (x, y) in enumerate(single.points):
                    for column, value in enumerate((x, y)):
                        self.vertices.setItem(i, column, QTableWidgetItem(f"{value:.4f}"))
                self._loading = False

        layer = tool.layers.get(layers[0]) if len(layers) == 1 else None
        if not editable:
            reason = layer.reason if layer is not None and layer.reason else "this copper has no statements of its " \
                                                                             "own in the Gerber (step and repeat)"
            self.note.setText(f"Read-only: {reason}.")
        elif trace:
            self.note.setText("Drag an end to move it - joined segments follow - or the diamond to bend the "
                              "segment. Right-click a point on it to split it there.")
        else:
            self.note.setText("Drag the selection to move it. Shift keeps a drag on 45°, Ctrl turns snapping off.")
        for widget in self._rows():
            widget.setEnabled(editable)

    def _rows(self) -> List[QWidget]:
        return [self.net, self.width, self.chain_width, self.start_x, self.start_y, self.end_x, self.end_y,
                self.height, self.length, self.arc, self.centre_x, self.centre_y, self.shape, self.vertices,
                self.vertex_hint, self.move_row, self.trace_actions, self.note, self.edit_row]

    def selected_piece(self) -> Optional[Piece]:
        pieces = self._pieces()
        return pieces[0] if len(pieces) == 1 else None
