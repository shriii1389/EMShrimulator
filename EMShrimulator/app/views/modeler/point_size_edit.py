"""Box corners, or a cylinder's axis ends, edited as start + size.

p2 is stored as `p1 + size`, but the size field shows only the size that was typed.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from EMShrimulator.app.views.modeler.expression_edit import BARE_NUMBER, ExpressionEdit, ExpressionVector
from EMShrimulator.app.views.modeler import units as unit_math
from EMShrimulator.app.widgets.form import tidy

Expr = object


def _start_text(a) -> str:
    """How corner_from_size writes a start value on the left of the `+`."""
    return tidy(float(a)) if isinstance(a, (int, float)) else f"({a})"


def _unwrap(text: str) -> str:
    """Remove one pair of brackets enclosing the whole text."""
    text = text.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return text
    depth = 0
    for i, char in enumerate(text):
        depth += char == "("
        depth -= char == ")"
        if depth == 0 and i < len(text) - 1:
            return text
    return text[1:-1].strip()


def size_of(a, b):
    """The typed size along one axis; None when p2 is not `p1 + size`."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(b) - float(a)
    if isinstance(a, (int, float)) and float(a) == 0:
        return b
    if not isinstance(b, str):
        return None
    prefix = f"{_start_text(a)} + "
    if not b.startswith(prefix):
        return None
    size = _unwrap(b[len(prefix):])
    if not size:
        return None
    # corner_from_size writes typed numbers in base units.
    return float(size) if BARE_NUMBER.match(size) else size


def corner_from_size(a, size):
    """The far corner for one axis given its start and its size."""
    if isinstance(a, (int, float)) and isinstance(size, (int, float)):
        return float(a) + float(size)
    if isinstance(a, (int, float)) and float(a) == 0:
        return size
    right = tidy(float(size)) if isinstance(size, (int, float)) else f"({size})"
    return f"{_start_text(a)} + {right}"


def axis_note(vector, factor: float, unit: str) -> str:
    """A cylinder's axis length and, if along a coordinate axis, which one."""
    length = math.sqrt(sum(float(v) ** 2 for v in vector))
    if length < 1e-12:
        return "zero length: the two end points coincide"
    text = f"axis {tidy(float(f'{length / factor:.6g}'))} {unit}"
    along = [i for i, v in enumerate(vector) if abs(float(v)) > length * 1e-9]
    if len(along) == 1:
        i = along[0]
        text += f" along {'+' if float(vector[i]) > 0 else '−'}{'xyz'[i]}"
    return text


class PointSizeEdit(QWidget):
    """Emits `committed(p1, p2)` and `modeChanged(size_mode)`."""

    committed = Signal(tuple, tuple)
    modeChanged = Signal(bool)

    def __init__(self, session, form: QFormLayout, *, path: Callable[[], Optional[str]],
                 first_field: str = "p1", second_field: str = "p2", parent=None):
        super().__init__(parent)
        self.session = session
        self.path_provider = path
        self._p1: Tuple = (0.0, 0.0, 0.0)
        self._p2: Tuple = (0.0, 0.0, 0.0)
        self._size_mode = False

        self.first = ExpressionVector(session, path=lambda: self._sub(first_field))
        self.second = ExpressionVector(session, path=lambda: self._sub(second_field))
        self.size_fields = [ExpressionEdit(session, "length", show_note=False, tooltip=axis)
                            for axis in ("X", "Y", "Z")]
        size_row = QWidget()
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(3)
        for i, field in enumerate(self.size_fields):
            field.edit.setPlaceholderText("XYZ"[i])
            field.committed.connect(lambda value, i=i: self._set_size(i, value))
            size_layout.addWidget(field, 1)
        self.size_row = size_row
        self.resolved_size = QLabel()
        self.resolved_size.setProperty("role", "note")
        # Wrapped, so a long note doesn't widen the inspector.
        self.resolved_size.setWordWrap(True)
        self._axis = False  # cylinders and tubes: the points are the axis ends

        self.toggle = QPushButton()
        self.toggle.setProperty("role", "flat")
        self.toggle.clicked.connect(lambda: self.modeChanged.emit(not self._size_mode))

        self.first_label = QLabel("Point 1")
        self.second_label = QLabel("Point 2")
        form.addRow(self.first_label, self.first)
        form.addRow(self.second_label, self._stack(self.second, size_row))
        form.addRow("", self._row(self.resolved_size, self.toggle))
        self._sizes: List = [None, None, None]  # the typed size per axis; None where p2 is its own formula

        self.first.committed.connect(self._set_start)
        self.second.committed.connect(lambda v: self.committed.emit(self._p1, v))

    @staticmethod
    def _stack(*widgets: QWidget) -> QWidget:
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            layout.addWidget(w, 1)
        return host

    @staticmethod
    def _row(label: QLabel, button: QPushButton) -> QWidget:
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label, 1)
        layout.addWidget(button)
        return host

    def _sub(self, field: str) -> Optional[str]:
        base = self.path_provider()
        return f"{base}.{field}" if base else None

    def set_value(self, p1, p2, size_mode: bool, axis: bool = False) -> None:
        self._p1 = tuple(p1 or (0.0, 0.0, 0.0))
        self._p2 = tuple(p2 or (0.0, 0.0, 0.0))
        self._size_mode = bool(size_mode)
        self._axis = axis
        units = self.session.project.display_units
        unit = unit_math.label("length", units)
        self.first_label.setText(f"Start ({unit})" if size_mode else f"Point 1 ({unit})")
        self.second_label.setText(f"Size ({unit})" if size_mode else f"Point 2 ({unit})")
        self.toggle.setText("↔ two corners" if size_mode else "↔ start + size")
        self.toggle.setToolTip("Edit the opposite corner instead" if size_mode
                               else "Edit a size from the start point instead")
        self.first.set_value(self._p1)
        self.second.set_value(self._p2)
        self.second.setVisible(not size_mode)
        self.size_row.setVisible(size_mode)
        resolved = self._resolved_sizes()
        for i, field in enumerate(self.size_fields):
            size = size_of(self._p1[i], self._p2[i])
            self._sizes[i] = size
            if size is None:
                # Point 2 is a formula: show its value; typing a size replaces it.
                field.set_value(resolved[i] if resolved else 0.0)
                field.edit.setToolTip(f"{'XYZ'[i]} size, worked out from the far corner {self._p2[i]}. "
                                      "Type a size to set it directly.")
            else:
                field.set_value(size)
        self._show_resolved_size(resolved)

    def _resolved_sizes(self) -> Optional[List[float]]:
        """p2 - p1 per axis, in base units, from the session's resolved project."""
        resolution = getattr(self.session, "resolution", None)
        base = self.path_provider()
        if not resolution or not base:
            return None
        from EMShrimulator.core.em.edits import get_path

        a = get_path(resolution.project, f"{base}.p1")
        b = get_path(resolution.project, f"{base}.p2")
        if not a or not b:
            return None
        return [float(b[i]) - float(a[i]) for i in range(3)]

    def _show_resolved_size(self, sizes: Optional[List[float]]) -> None:
        if not sizes:
            self.resolved_size.setText("")
            return
        factor = unit_math.factor("length", self.session.project.display_units)
        unit = unit_math.label("length", self.session.project.display_units)
        if self._axis:
            self.resolved_size.setText(axis_note(sizes, factor, unit))
            return
        size = [abs(s) / factor for s in sizes]
        flat = any(s < 1e-9 for s in size)
        text = " × ".join(tidy(float(f"{s:.6g}")) for s in size)
        self.resolved_size.setText(f"{text} {unit}" + ("  — flat: an infinitely thin sheet" if flat else ""))

    def _set_start(self, p1) -> None:
        """A new start point. In size mode the box moves and keeps its size."""
        p2 = list(self._p2)
        if self._size_mode:
            for i, size in enumerate(self._sizes):
                if size is not None:
                    p2[i] = corner_from_size(p1[i], size)
        self.committed.emit(tuple(p1), tuple(p2))

    def _set_size(self, i: int, size) -> None:
        p2 = list(self._p2)
        p2[i] = corner_from_size(self._p1[i], size)
        self.committed.emit(self._p1, tuple(p2))
