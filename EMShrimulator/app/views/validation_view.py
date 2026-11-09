"""The validator's report, one row per issue, click to select the culprit."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from EMShrimulator.app import theme

GLYPHS = {"error": "✖", "warning": "▲", "info": "ℹ"}
TINTS = {"error": "danger", "warning": "warn", "info": "text_dim"}


class ValidationView(QWidget):
    targetActivated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.summary = QLabel("Validation has not run yet.")
        self.summary.setProperty("role", "hint")
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["", "Issue", "Where"])
        self.tree.setRootIsDecorated(False)
        self.tree.setWordWrap(True)
        self.tree.setUniformRowHeights(False)
        self.tree.setColumnWidth(0, 28)
        self.tree.setColumnWidth(1, 560)
        self.tree.itemActivated.connect(self._activate)
        self.tree.itemClicked.connect(self._activate)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.summary)
        layout.addWidget(self.tree, 1)

    def show_report(self, report: Optional[Dict]) -> None:
        # Most edits leave the report unchanged.
        if report is not None and report == getattr(self, "_shown", None):
            return
        self._shown = report
        self.tree.clear()
        colors = theme.colors()
        if report is None:
            self.summary.setText("Validation has not run yet.")
            return
        issues = report.get("issues", [])
        if not issues:
            self.summary.setText("✓ No issues found — the design is ready to run.")
            self.summary.setStyleSheet(f"color: {colors['ok']}")
            return
        self.summary.setStyleSheet("")
        self.summary.setText(
            f"{report.get('errors', 0)} error(s), {report.get('warnings', 0)} warning(s), "
            f"{report.get('infos', 0)} note(s). Click an issue to select what it is about."
        )
        for issue in issues:
            text = issue["message"] + (f"\n{issue['hint']}" if issue.get("hint") else "")
            item = QTreeWidgetItem([GLYPHS.get(issue["severity"], "•"), text,
                                    f"{issue.get('code', '')}  {issue.get('target') or ''}".strip()])
            item.setForeground(0, QColor(colors[TINTS.get(issue["severity"], "text")]))
            item.setToolTip(1, text)
            item.setData(0, Qt.UserRole, issue.get("target"))
            self.tree.addTopLevelItem(item)

    def _activate(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        target = item.data(0, Qt.UserRole)
        if target:
            self.targetActivated.emit(str(target))
