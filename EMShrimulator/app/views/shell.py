"""Shared workspace pieces: a titled centre page, the run indicator and the inspector."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import theme
from EMShrimulator.app.views.job_console import FINISHED
from EMShrimulator.app.widgets.form import fmt_duration

SIDE_WIDTH = 270
INSPECTOR_WIDTH = 360


class Page(QWidget):
    """A centre page under a title bar: its name, a line of context, and its own buttons."""

    def __init__(self, title: str, content: QWidget, subtitle: str = "", actions: List[QWidget] = ()):
        super().__init__()
        self.content = content
        header = QFrame()
        header.setProperty("role", "page-header")
        self.title = QLabel(title)
        self.title.setProperty("role", "page-title")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setProperty("role", "note")
        self.subtitle.setWordWrap(True)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(self.title)
        text.addWidget(self.subtitle)
        head = QHBoxLayout(header)
        head.setContentsMargins(16, 10, 12, 10)
        head.addLayout(text, 1)
        for action in actions:
            head.addWidget(action)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(content, 1)


class RunIndicator(QWidget):
    """The run in progress in the status bar; a click opens the log."""

    def __init__(self, open_log, parent=None):
        super().__init__(parent)
        self._open_log = open_log
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Open the solver log")
        self.dot = QLabel("●")
        self.stage = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(110)
        self.progress.setFixedHeight(8)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.dot)
        layout.addWidget(self.stage)
        layout.addWidget(self.progress)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)  # hug its parts in the status bar
        # Hide the parts, not the indicator itself.
        self._parts = (self.dot, self.stage, self.progress)
        self._set_parts_visible(False)

    def _set_parts_visible(self, visible: bool) -> None:
        for part in self._parts:
            part.setVisible(visible)

    def show_status(self, status: Optional[dict]) -> None:
        if not status:
            self._set_parts_visible(False)
            return
        colors = theme.colors()
        running = status["status"] not in FINISHED
        color = colors["accent"] if running else colors["ok"] if status["status"] == "done" else colors["danger"]
        self.dot.setStyleSheet(f"color: {color}")
        stage = status.get("stage") or status["status"]
        self.stage.setText(f"{stage} · {fmt_duration(status.get('elapsed_s'))}")
        progress = status.get("progress")
        if running and progress is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int((progress or 0) * 1000))
        self.dot.setVisible(True)
        self.stage.setVisible(True)
        self.progress.setVisible(running)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._open_log()
        super().mousePressEvent(event)


class Inspector(QWidget):
    """The panel beside a design view that edits the selection; × closes it and clears the selection."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setObjectName("inspector")

        self.heading = QLabel("Properties")
        self.heading.setProperty("role", "heading")
        close = QPushButton("×")
        close.setProperty("role", "flat")
        close.setFixedWidth(26)
        close.setToolTip("Close (clears the selection)")
        close.clicked.connect(lambda: self.session.select(None))
        head = QHBoxLayout()
        head.setContentsMargins(10, 6, 4, 4)
        head.addWidget(self.heading, 1)
        head.addWidget(close)

        host = QWidget()
        self._layout = QVBoxLayout(host)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        # Fields shrink to fit rather than scrolling sideways.
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidget(host)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.scroll, 1)

    def add_editor(self, editor: QWidget) -> None:
        self._layout.insertWidget(self._layout.count() - 1, editor)  # above the stretch

    def show_title(self, editor: QWidget) -> None:
        """Show the editor's title in the panel heading."""
        editor.title.setVisible(False)
        self.heading.setText(editor.title.text())


def fit_inspector(split, shown: bool) -> None:
    """Give a just-shown inspector its width back if the splitter had squeezed it."""
    if not shown:
        return
    sizes = split.sizes()
    if len(sizes) == 2 and sizes[1] < 200:
        total = sum(sizes) or split.width()
        split.setSizes([max(total - INSPECTOR_WIDTH, 300), INSPECTOR_WIDTH])
