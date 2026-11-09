"""The install report: every piece the solvers depend on, and how to fix it."""
from __future__ import annotations

from importlib import import_module
from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from EMShrimulator.app import theme
from EMShrimulator.core import diagnostics


def gui_checks() -> List[Dict]:
    """What only the desktop app needs, on top of the core's checks."""
    out = []
    for module, label, fix in (
        ("PySide6", "Qt (PySide6)", "pip install PySide6"),
        ("vtkmodules", "VTK (3D viewport)", "pip install vtk"),
        ("vtkmodules.qt.QVTKRenderWindowInteractor", "VTK Qt widget", "pip install vtk"),
    ):
        try:
            loaded = import_module(module)
            out.append(diagnostics.check(label, True, str(getattr(loaded, "__version__", ""))))
        except Exception as exc:  # noqa: BLE001
            out.append(diagnostics.check(label, False, str(exc), fix))
    return out


class DiagnosticsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Environment")
        self.resize(900, 520)
        report = diagnostics.diagnostics()
        report["checks"] = report["checks"] + gui_checks()
        env = report["environment"]

        intro = QLabel(
            f"Every piece the solvers depend on. Version {report['version']} · Python {env['python']} · "
            f"{env['cpu_count']} logical cores · {env['platform']}"
            + (" · installed build" if env.get("frozen") else " · source checkout")
        )
        intro.setProperty("role", "hint")
        intro.setWordWrap(True)

        tree = QTreeWidget()
        tree.setHeaderLabels(["", "Component", "Detail"])
        tree.setRootIsDecorated(False)
        tree.setWordWrap(True)
        tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        colors = theme.colors()
        for item in report["checks"]:
            detail = item["detail"] + (f"\nFix: {item['fix']}" if not item["ok"] and item["fix"] else "")
            row = QTreeWidgetItem(["●", item["name"], detail])
            row.setForeground(0, QColor(colors["ok" if item["ok"] else "danger"]))
            row.setToolTip(2, detail)
            tree.addTopLevelItem(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(tree, 1)
        layout.addWidget(buttons, 0, Qt.AlignRight)
