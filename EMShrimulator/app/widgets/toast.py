"""Transient corner notifications. Errors stay up longer."""
from __future__ import annotations

from typing import List

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

DURATION_MS = {"ok": 3500, "info": 3500, "error": 8000}
ICONS = {"ok": "✓", "info": "•", "error": "⚠"}


class _Toast(QFrame):
    def __init__(self, message: str, kind: str, parent: QWidget):
        super().__init__(parent)
        self.setProperty("role", "toast-error" if kind == "error" else "toast")
        self.setMaximumWidth(460)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        icon = QLabel(ICONS.get(kind, "•"))
        text = QLabel(message)
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(icon, 0, Qt.AlignTop)
        layout.addWidget(text, 1)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.deleteLater()


class ToastHost(QObject):
    """Stacks toasts in the bottom-right corner of `window`."""

    def __init__(self, window: QWidget):
        super().__init__(window)
        self._window = window
        self._toasts: List[_Toast] = []
        self._column = QWidget(window)
        self._column.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._layout = QVBoxLayout(self._column)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        self._column.hide()
        window.installEventFilter(self)

    def show(self, message: str, kind: str = "info") -> None:
        toast = _Toast(message, kind, self._column)
        self._layout.addWidget(toast)
        self._toasts.append(toast)
        toast.destroyed.connect(lambda *_: self._forget(toast))
        QTimer.singleShot(DURATION_MS.get(kind, 3500), toast.deleteLater)
        self._column.show()
        self._column.raise_()
        self._place()

    def _forget(self, toast: _Toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        if not self._toasts:
            self._column.hide()
        else:
            QTimer.singleShot(0, self._place)

    def _place(self) -> None:
        self._column.adjustSize()
        margin = 18
        geometry = self._window.rect()
        x = geometry.right() - self._column.width() - margin
        y = geometry.bottom() - self._column.height() - margin - 24
        self._column.move(max(x, 0), max(y, 0))

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is self._window and event.type() == QEvent.Resize and self._toasts:
            self._place()
        return False
