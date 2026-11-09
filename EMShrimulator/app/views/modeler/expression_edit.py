"""A dimension field that accepts a number (in display units) or an expression.

Commits on Enter or focus out; Escape cancels; Up/Down nudge plain numbers.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from EMShrimulator.app.views.modeler import units as unit_math
from EMShrimulator.app.widgets.form import tidy

BARE_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

PathProvider = Callable[[], Optional[str]]


def parse_entry(text: str, factor: float):
    """What a typed entry stores: a float in base units, a formula, or None."""
    trimmed = text.strip()
    if not trimmed:
        return None
    if BARE_NUMBER.match(trimmed):
        return float(trimmed) * factor
    return trimmed


def display_text(value, factor: float) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""  # not set: an empty field, not a 0 that looks like a choice
    try:
        return tidy(float(value if value is not None else 0.0) / factor)
    except (TypeError, ValueError):
        return "0"


class ExpressionEdit(QWidget):
    committed = Signal(object)  # float (base units) or str (formula)

    def __init__(self, session, quantity: str = "length", *, path: Optional[PathProvider] = None,
                 step: float = 0.0, tooltip: str = "", show_note: bool = True, parent=None):
        super().__init__(parent)
        self.session = session
        self.quantity = quantity
        self.path_provider = path
        self.step = step
        self._value = 0.0
        self._base_tooltip = tooltip
        self._defining = False

        self.edit = QLineEdit()
        self.edit.setMinimumWidth(48)
        self.edit.installEventFilter(self)
        self.edit.editingFinished.connect(self._commit)

        self.note = QLabel()
        self.note.setProperty("role", "note")
        self.note.setWordWrap(True)  # an error message wraps rather than widening the form
        self.note.setVisible(False)
        self._show_note = show_note

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.edit)
        layout.addWidget(self.note)

    # ------------------------------------------------------------ state

    def _factor(self) -> float:
        return unit_math.factor(self.quantity, self.session.project.display_units)

    def value(self):
        return self._value

    def set_value(self, value) -> None:
        """Show a stored value. Never overwrites text the user is typing."""
        self._value = value
        if not self.edit.hasFocus():
            # Avoid setText when unchanged: it resets undo and repaints.
            text = display_text(value, self._factor())
            if self.edit.text() != text:
                self.edit.setText(text)
                self.edit.setCursorPosition(0)
        self.refresh_note()

    def refresh_note(self) -> None:
        is_formula = isinstance(self._value, str)
        path = self.path_provider() if self.path_provider else None
        resolution = getattr(self.session, "resolution", None)
        error = resolution.error_for(path) if (resolution and path) else None

        self._set_flag("formula", is_formula)
        self._set_flag("invalid", error is not None)
        tip = self._base_tooltip
        if is_formula:
            tip = (tip + "\n" if tip else "") + f"Expression, in base units: {self._value}"
        self.edit.setToolTip(tip)

        if not self._show_note or not is_formula:
            self.note.setVisible(False)
            return
        if error is not None:
            text, role = error["message"], "note-error"
        else:
            resolved = self._resolved_value(path)
            if resolved is None:
                self.note.setVisible(False)
                return
            unit = unit_math.label(self.quantity, self.session.project.display_units)
            # Six significant digits.
            text, role = f"= {tidy(float(f'{resolved / self._factor():.6g}'))} {unit}".rstrip(), "note"
        self.note.setText(text)
        if self.note.property("role") != role:
            self.note.setProperty("role", role)
            self.note.style().unpolish(self.note)
            self.note.style().polish(self.note)
        self.note.setVisible(True)

    def _resolved_value(self, path: Optional[str]) -> Optional[float]:
        if not path:
            return None
        from EMShrimulator.core.em.edits import get_path

        resolution = getattr(self.session, "resolution", None)
        if resolution is None:
            return None
        value = get_path(resolution.project, path)
        return float(value) if isinstance(value, (int, float)) else None

    def _set_flag(self, name: str, on: bool) -> None:
        if self.edit.property(name) != on:
            self.edit.setProperty(name, on)
            self.edit.style().unpolish(self.edit)
            self.edit.style().polish(self.edit)

    # ----------------------------------------------------------- editing

    def _commit(self) -> None:
        if self._defining:
            return  # the define dialog took focus: same entry finishing again
        entry = parse_entry(self.edit.text(), self._factor())
        if entry is None:
            self.edit.setText(display_text(self._value, self._factor()))
            return
        if entry != self._value:
            definitions = self._define_new_names(entry)
            self._value = entry
            if definitions:
                from EMShrimulator.app.views.modeler.define_variable import push_with_definitions

                push_with_definitions(self.session, definitions, lambda: self.committed.emit(entry))
            else:
                self.committed.emit(entry)  # unknown names show as the usual error
        self.refresh_note()

    def _define_new_names(self, entry) -> list:
        """Ask for the variables a formula reads before they exist; empty if none or one is declined."""
        if not isinstance(entry, str):
            return []
        from EMShrimulator.app.views.modeler.define_variable import define_missing

        self._defining = True
        try:
            return define_missing(self.session, entry, quantity=self.quantity, parent=self.window()) or []
        finally:
            self._defining = False

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is self.edit and event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Escape:
                self.edit.setText(display_text(self._value, self._factor()))
                self.edit.clearFocus()
                return True
            if key in (Qt.Key_Up, Qt.Key_Down) and self.step and not isinstance(self._value, str) \
                    and not self.edit.isModified():
                delta = self.step * (10 if event.modifiers() & Qt.ShiftModifier else 1)
                delta = delta if key == Qt.Key_Up else -delta
                factor = self._factor()
                shown = float(self._value or 0.0) / factor + delta
                self._value = float(f"{shown:.10g}") * factor
                self.edit.setText(display_text(self._value, factor))
                self.committed.emit(self._value)
                return True
        return super().eventFilter(watched, event)


class ExpressionVector(QWidget):
    """Two or three dimensions side by side, each independently an expression."""

    committed = Signal(tuple)

    def __init__(self, session, quantity: str = "length", *, count: int = 3,
                 path: Optional[PathProvider] = None, step: float = 0.0,
                 axis_labels: Sequence[str] = ("X", "Y", "Z"), parent=None):
        super().__init__(parent)
        self.path_provider = path
        self._values: List = [0.0] * count
        self.fields: List[ExpressionEdit] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        for i in range(count):
            field = ExpressionEdit(session, quantity, path=self._element_path(i), step=step,
                                   tooltip=axis_labels[i] if i < len(axis_labels) else "")
            field.edit.setPlaceholderText(axis_labels[i] if i < len(axis_labels) else "")
            field.committed.connect(lambda value, i=i: self._commit(i, value))
            self.fields.append(field)
            layout.addWidget(field, 1)

    def _element_path(self, i: int) -> PathProvider:
        def provider() -> Optional[str]:
            base = self.path_provider() if self.path_provider else None
            return f"{base}.{i}" if base else None
        return provider

    def set_value(self, values) -> None:
        """Show a point; None shows empty fields, and typing into one fills the others with 0."""
        if values is None:
            self._values = [0.0] * len(self.fields)
            for field in self.fields:
                field.set_value(None)
            return
        values = list(values)
        values += [0.0] * (len(self.fields) - len(values))
        self._values = values[: len(self.fields)]
        for field, value in zip(self.fields, self._values):
            field.set_value(value)

    def value(self) -> tuple:
        return tuple(self._values)

    def _commit(self, i: int, value) -> None:
        self._values[i] = value
        self.committed.emit(tuple(self._values))
