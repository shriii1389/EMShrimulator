"""A titled section that folds away, remembering whether it was open."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    def __init__(self, title: str, *, key: str = "", open_by_default: bool = True, parent=None):
        super().__init__(parent)
        self._key = f"sections/{key}" if key else ""
        self._title = title

        self.header = QToolButton()
        self.header.setProperty("role", "section")
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setCheckable(True)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.count_label = QLabel()
        self.count_label.setProperty("role", "note")

        self.actions = QHBoxLayout()
        self.actions.setContentsMargins(0, 0, 0, 0)
        self.actions.setSpacing(2)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        head.addWidget(self.header, 1)
        head.addWidget(self.count_label)
        head.addLayout(self.actions)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 4, 4, 8)
        self.body_layout.setSpacing(4)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.body)

        is_open = open_by_default
        if self._key:
            stored = QSettings().value(self._key)
            if stored is not None:
                is_open = str(stored).lower() in ("true", "1")
        self.header.setChecked(is_open)
        self.header.toggled.connect(self._toggle)
        self._toggle(is_open, store=False)

    def add_action(self, widget: QWidget) -> None:
        self.actions.addWidget(widget)

    def add_widget(self, widget: QWidget, stretch: int = 0) -> None:
        self.body_layout.addWidget(widget, stretch)

    def set_count(self, count: Optional[int]) -> None:
        self.count_label.setText("" if count is None else str(count))

    def set_open(self, is_open: bool) -> None:
        self.header.setChecked(is_open)

    def _toggle(self, is_open: bool, store: bool = True) -> None:
        self.header.setText(("▾  " if is_open else "▸  ") + self._title)
        self.body.setVisible(is_open)
        if store and self._key:
            QSettings().setValue(self._key, is_open)
