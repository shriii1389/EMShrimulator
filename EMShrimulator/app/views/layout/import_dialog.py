"""Report of how each imported file was identified."""
from __future__ import annotations

from typing import Dict

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from EMShrimulator.app.widgets.form import hint


class ImportReportDialog(QDialog):
    def __init__(self, report: Dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gerber import")
        self.resize(820, 520)
        files = report.get("files") or []
        used = [f for f in files if f.get("target")]
        ignored = [f for f in files if not f.get("target")]

        summary = QLabel(
            f"<b>{len(report.get('copper_layers') or [])}</b> copper layers · board outline "
            f"<b>{'found' if any('Edge_Cuts' in g for g in report.get('gerber_files') or []) else 'missing'}</b> · "
            f"drill file <b>{report.get('drill_file') or 'none'}</b>")
        layout = QVBoxLayout(self)
        layout.addWidget(hint("Each file was identified from its own Gerber X2 layer attribute where it had one, "
                              "and from its name otherwise, then renamed to the layer naming the solver expects."))
        layout.addWidget(summary)
        for warning in report.get("warnings") or []:
            label = QLabel(f"▲ {warning}")
            label.setWordWrap(True)
            label.setProperty("role", "error")
            layout.addWidget(label)

        table = QTableWidget(len(used), 4)
        table.setHorizontalHeaderLabels(["Imported as", "Read as", "Stored as", "How"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        for r, f in enumerate(used):
            for c, value in enumerate((f["source"], f.get("layer") or f.get("role"), f.get("target"), f.get("evidence"))):
                table.setItem(r, c, QTableWidgetItem(str(value or "")))
        layout.addWidget(table, 1)

        if ignored:
            layout.addWidget(hint("Not board data, so not imported. If one of these is a layer you need, rename it to "
                                  "say what it is (for example 'top copper' or 'Edge_Cuts') and import it again."))
            text = QPlainTextEdit(" · ".join(f["source"] for f in ignored))
            text.setReadOnly(True)
            text.setMaximumHeight(70)
            layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
