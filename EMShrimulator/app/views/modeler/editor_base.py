"""What every property editor shares: the session, a form, commit helpers."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from EMShrimulator.app.notify import guarded
from EMShrimulator.app.widgets.form import form_layout


class NameEdit(QLineEdit):
    """A name field that commits on Enter or focus out."""

    def __init__(self, commit: Callable[[str], None], parent=None):
        super().__init__(parent)
        self._value = ""
        self._commit = commit
        self.editingFinished.connect(self._finish)

    def set_value(self, value: str) -> None:
        self._value = value
        if not self.hasFocus() and self.text() != value:
            self.setText(value)

    def _finish(self) -> None:
        text = self.text().strip()
        if not text:
            self.setText(self._value)
        elif text != self._value:
            self._value = text
            self._commit(text)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        from PySide6.QtCore import Qt

        if event.key() == Qt.Key_Escape:
            self.setText(self._value)
            self.clearFocus()
            return
        super().keyPressEvent(event)


class PropertyEditor(QWidget):
    """Base for the per-selection editors in the property panel."""

    def __init__(self, session, title: str = "", parent=None):
        super().__init__(parent)
        self.session = session
        self.title = QLabel(title)
        self.title.setProperty("role", "heading")
        self.form: QFormLayout = form_layout()
        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(6, 6, 6, 6)
        self.outer.setSpacing(6)
        self.outer.addWidget(self.title)
        self.outer.addLayout(self.form)

    def push(self, make_command: Callable[[], object]) -> Optional[object]:
        """Build and push an undo command, reporting a refused edit."""
        return guarded(lambda: self.session.push(make_command()))

    @staticmethod
    def show_row(form: QFormLayout, field: QWidget, visible: bool) -> None:
        form.setRowVisible(field, visible)
