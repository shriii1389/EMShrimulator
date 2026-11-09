"""Editors for outlines, paths and transforms. Rows are rebuilt only when the count changes."""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit, ExpressionVector
from EMShrimulator.app.widgets.form import ChoiceBox, form_layout
from EMShrimulator.core.em.model import TransformOp

PathProvider = Callable[[], Optional[str]]


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()


class PointListEditor(QWidget):
    """A list of 2D (u, v) or 3D points, each coordinate an expression."""

    committed = Signal(list)

    def __init__(self, session, dims: int, *, title: str, min_points: int,
                 path: PathProvider, parent=None):
        super().__init__(parent)
        self.session = session
        self.dims = dims
        self.min_points = min_points
        self.path_provider = path
        self._points: List[tuple] = []
        self._rows: List[ExpressionVector] = []
        self._removers: List[QPushButton] = []

        heading = QLabel(title)
        heading.setProperty("role", "hint")
        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        add = QPushButton("+ Point")
        add.clicked.connect(self._add)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(heading)
        layout.addLayout(self.rows_layout)
        layout.addWidget(add)

    def set_value(self, points) -> None:
        self._points = [tuple(p) for p in (points or [])]
        if len(self._points) != len(self._rows):
            self._rebuild()
        for row, point in zip(self._rows, self._points):
            row.set_value(point)
        for remover in self._removers:
            remover.setEnabled(len(self._points) > self.min_points)

    def _rebuild(self) -> None:
        _clear(self.rows_layout)
        self._rows, self._removers = [], []
        labels = ("u", "v") if self.dims == 2 else ("X", "Y", "Z")
        for i in range(len(self._points)):
            host = QWidget()
            h = QHBoxLayout(host)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(3)
            index = QLabel(str(i))
            index.setProperty("role", "note")
            index.setFixedWidth(14)
            vector = ExpressionVector(self.session, count=self.dims, axis_labels=labels,
                                      path=lambda i=i: self._point_path(i))
            vector.committed.connect(lambda value, i=i: self._set(i, value))
            remove = QPushButton("×")
            remove.setProperty("role", "flat")
            remove.setFixedWidth(24)
            remove.clicked.connect(lambda _=False, i=i: self._remove(i))
            h.addWidget(index)
            h.addWidget(vector, 1)
            h.addWidget(remove)
            self.rows_layout.addWidget(host)
            self._rows.append(vector)
            self._removers.append(remove)

    def _point_path(self, i: int) -> Optional[str]:
        base = self.path_provider()
        return f"{base}.{i}" if base else None

    def _set(self, i: int, value: tuple) -> None:
        points = list(self._points)
        points[i] = tuple(value)
        self.committed.emit(points)

    def _remove(self, i: int) -> None:
        if len(self._points) > self.min_points:
            self.committed.emit([p for j, p in enumerate(self._points) if j != i])

    def _add(self) -> None:
        last = self._points[-1] if self._points else tuple([0.0] * self.dims)
        nudged = list(last)
        try:
            nudged[0] = float(nudged[0]) + 1.0
        except (TypeError, ValueError):
            pass
        self.committed.emit(self._points + [tuple(nudged)])


class TransformListEditor(QWidget):
    """An object's own transform chain, applied in order."""

    committed = Signal(list)

    KINDS = [("translate", "Translate"), ("rotate_axis", "Rotate about axis"), ("scale", "Scale")]

    def __init__(self, session, *, path: PathProvider, parent=None):
        super().__init__(parent)
        self.session = session
        self.path_provider = path
        self._ops: List[TransformOp] = []
        self._signature: tuple = ()
        self._widgets: List[dict] = []

        heading = QLabel("Transforms (applied in order)")
        heading.setProperty("role", "hint")
        self.cards = QVBoxLayout()
        self.cards.setContentsMargins(0, 0, 0, 0)
        self.cards.setSpacing(4)
        self.empty = QLabel("None.")
        self.empty.setProperty("role", "note")
        add_translate = QPushButton("+ Translate")
        add_rotate = QPushButton("+ Rotate")
        add_translate.clicked.connect(lambda: self._append(TransformOp(kind="translate", translate=(0.0, 0.0, 0.0))))
        add_rotate.clicked.connect(lambda: self._append(TransformOp(kind="rotate_axis", axis="z", angle_deg=90.0)))
        buttons = QHBoxLayout()
        buttons.addWidget(add_translate)
        buttons.addWidget(add_rotate)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(heading)
        layout.addWidget(self.empty)
        layout.addLayout(self.cards)
        layout.addLayout(buttons)

    def set_value(self, ops: List[TransformOp]) -> None:
        self._ops = [op.model_copy(deep=True) for op in (ops or [])]
        signature = tuple(op.kind for op in self._ops)
        if signature != self._signature:
            self._signature = signature
            self._rebuild()
        self.empty.setVisible(not self._ops)
        for op, widgets in zip(self._ops, self._widgets):
            widgets["kind"].set_value(op.kind)
            if "vector" in widgets:
                widgets["vector"].set_value(op.translate if op.kind == "translate" else (op.scale or (1.0, 1.0, 1.0)))
            if "axis" in widgets:
                widgets["axis"].set_value(op.axis or "z")
                widgets["angle"].set_value(op.angle_deg if op.angle_deg is not None else 0.0)

    def _op_path(self, i: int, field: str) -> Optional[str]:
        base = self.path_provider()
        return f"{base}.{i}.{field}" if base else None

    def _rebuild(self) -> None:
        _clear(self.cards)
        self._widgets = []
        for i, op in enumerate(self._ops):
            card = QFrame()
            card.setProperty("role", "card")
            form = form_layout()
            form.setContentsMargins(6, 6, 6, 6)
            card.setLayout(form)
            widgets: dict = {}
            kind = ChoiceBox(self.KINDS)
            kind.committed.connect(lambda value, i=i: self._change(i, kind=value))
            widgets["kind"] = kind
            form.addRow("Type", kind)
            if op.kind in ("translate", "scale"):
                field = "translate" if op.kind == "translate" else "scale"
                vector = ExpressionVector(self.session, "length" if op.kind == "translate" else "plain",
                                          path=lambda i=i, field=field: self._op_path(i, field))
                vector.committed.connect(lambda value, i=i, field=field: self._change(i, **{field: value}))
                widgets["vector"] = vector
                form.addRow("Offset" if op.kind == "translate" else "Factor", vector)
            else:
                axis = ChoiceBox([("x", "x"), ("y", "y"), ("z", "z")])
                axis.committed.connect(lambda value, i=i: self._change(i, axis=value))
                angle = ExpressionEdit(self.session, "angle", step=15,
                                       path=lambda i=i: self._op_path(i, "angle_deg"))
                angle.committed.connect(lambda value, i=i: self._change(i, angle_deg=value))
                widgets["axis"], widgets["angle"] = axis, angle
                form.addRow("Axis", axis)
                form.addRow("Angle (°)", angle)
            remove = QPushButton("Remove")
            remove.setProperty("role", "danger")
            remove.clicked.connect(lambda _=False, i=i: self.committed.emit(
                [o for j, o in enumerate(self._ops) if j != i]))
            form.addRow("", remove)
            self.cards.addWidget(card)
            self._widgets.append(widgets)

    def _change(self, i: int, **changes) -> None:
        ops = [op.model_copy(deep=True) for op in self._ops]
        op = ops[i]
        for field, value in changes.items():
            setattr(op, field, value)
        if "kind" in changes:
            if op.kind == "translate" and op.translate is None:
                op.translate = (0.0, 0.0, 0.0)
            if op.kind == "scale" and op.scale is None:
                op.scale = (1.0, 1.0, 1.0)
            if op.kind == "rotate_axis":
                op.axis = op.axis or "z"
                op.angle_deg = op.angle_deg if op.angle_deg is not None else 0.0
        self.committed.emit(ops)

    def _append(self, op: TransformOp) -> None:
        self.committed.emit(self._ops + [op])
