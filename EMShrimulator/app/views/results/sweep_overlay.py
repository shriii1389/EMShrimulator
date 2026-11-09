"""One S-parameter across every solved point of a sweep."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import theme
from EMShrimulator.app.views.results.chart import Chart, eng
from EMShrimulator.app.widgets.form import ChoiceBox, fmt_eng


class SweepOverlay(QWidget):
    sweepChosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: Optional[Dict] = None
        self._hidden: set = set()
        self.log_x = False

        self.sweep_box = ChoiceBox()
        self.sweep_box.committed.connect(lambda slug: self.sweepChosen.emit(slug))
        self.trace_box = ChoiceBox()
        self.trace_box.committed.connect(lambda _k: self.redraw())
        self.summary = QLabel()
        self.summary.setProperty("role", "note")
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Sweep"))
        bar.addWidget(self.sweep_box)
        bar.addWidget(QLabel("Trace"))
        bar.addWidget(self.trace_box)
        bar.addWidget(self.summary, 1)

        self.points = QListWidget()
        self.points.itemChanged.connect(self._toggle)
        self.canvas = Chart()
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Design point", "Best match", "At", "−10 dB bandwidth", "Solve time"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        right = QSplitter(Qt.Vertical)
        right.addWidget(self.canvas)
        right.addWidget(self.table)
        right.setSizes([400, 140])
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.points)
        split.addWidget(right)
        split.setSizes([220, 800])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(split, 1)

    def set_sweeps(self, sweeps: List[Dict], current: Optional[str]) -> None:
        self.sweep_box.set_options([(s["slug"], f"{s['name']} ({s['completed']}/{s['points']})") for s in sweeps])
        if current:
            self.sweep_box.set_value(current)

    def show_results(self, data: Dict) -> None:
        self.data = data
        done = [p for p in data["points"] if p["status"] == "done" and p["freq_hz"]]
        keys = sorted({k for p in done for k in p["traces"]})
        current = self.trace_box.value()
        self.trace_box.set_options([(k, k) for k in keys])
        self.trace_box.set_value(current if current in keys else ("S11" if "S11" in keys else (keys[0] if keys else None)))
        variables = ", ".join(v.get("parameter", "") for v in data.get("variables", []))
        self.summary.setText(f"{variables} · {len(done)} of {len(data['points'])} solved")

        colors = theme.colors()
        self.points.blockSignals(True)
        self.points.clear()
        for i, point in enumerate(data["points"]):
            status = point["status"]
            suffix = {"failed": "  ✕", "running": "  …", "pending": "  (pending)"}.get(status, "")
            item = QListWidgetItem(point["label"] + suffix)
            item.setData(Qt.UserRole, point["index"])
            item.setToolTip(point.get("error") or point["label"])
            if status == "done":
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked if point["index"] in self._hidden else Qt.Checked)
                item.setData(Qt.DecorationRole, QColor(theme.trace_color(i)))
            else:
                item.setFlags(Qt.ItemIsEnabled)
                item.setForeground(QColor(colors["danger"] if status == "failed" else colors["text_faint"]))
            self.points.addItem(item)
        self.points.blockSignals(False)

        with_marker = [p for p in done if p.get("marker")]
        self.table.setRowCount(len(with_marker))
        for row, point in enumerate(with_marker):
            marker = point["marker"]
            values = [point["label"], f"{marker['s11_db']:.1f} dB", fmt_eng(marker["frequency_hz"], "Hz"),
                      fmt_eng(marker.get("bandwidth_hz"), "Hz"),
                      f"{point['elapsed_s']:.1f} s" if point.get("elapsed_s") else "—"]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.setVisible(bool(with_marker))
        self.redraw()

    def _toggle(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked:
            self._hidden.discard(index)
        else:
            self._hidden.add(index)
        self.redraw()

    def redraw(self) -> None:
        chart = self.canvas
        chart.clear()
        c = theme.colors()
        key = self.trace_box.value()
        if not self.data or not key:
            chart.text(0.5, 0.5, "No design point has finished solving yet. The chart fills in as each one completes.",
                       color=c["text_dim"], axes=True, align=Qt.AlignCenter, size=9.0)
            return
        chart.set_x_unit("Hz")
        chart.set_log_x(self.log_x)
        for i, point in enumerate(self.data["points"]):
            if point["status"] != "done" or not point["freq_hz"] or point["index"] in self._hidden:
                continue
            values = point["traces"].get(key, {}).get("mag_db")
            if values:
                freqs = np.asarray(point["freq_hz"], dtype=float)
                values = np.asarray(values, dtype=float)
                chart.plot(freqs, values, theme.trace_color(i), width=1.4, label=point["label"],
                           readout=lambda j, label=point["label"], freqs=freqs, values=values:
                           f"{label}  {values[j]:.2f} dB @ {eng(float(freqs[j]), 'Hz')}")
        chart.set_labels(x="Frequency", y=f"{key} magnitude (dB)")
