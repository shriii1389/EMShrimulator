"""The Results overview's radiation pattern: polar cuts beside a θ–φ directivity map."""
from __future__ import annotations

from typing import Dict

from PySide6.QtWidgets import QHBoxLayout, QWidget

from EMShrimulator.app.views.results import report_draw
from EMShrimulator.app.views.results.chart import Chart
from EMShrimulator.core.em import reports
from EMShrimulator.core.em.model import ResultPlot


class FarfieldCharts(QWidget):
    """E/H cuts on the left, the θ-φ directivity map on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.polar = Chart()
        self.heat = Chart()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.polar, 1)
        layout.addWidget(self.heat, 1)


def plot_farfield(charts: FarfieldCharts, farfield: Dict, entry: Dict) -> None:
    """Draw one frequency's directivity: the φ = 0° and 90° cuts, and the whole sphere as a map."""
    plot = ResultPlot(name="", kind="farfield", display="polar", frequency_hz=float(entry["frequency_hz"]))
    report = reports.pattern_cuts(plot, farfield)
    grid, unit, message, _entry = reports.pattern_grid(plot, farfield)
    if grid is None:
        report_draw.message(charts.polar, report.message or message)
        report_draw.message(charts.heat, message)
        return
    floor, peak = reports.pattern_scale(plot, grid, unit)
    color_of = report_draw.color_map([s.key for s in report.series])
    report_draw.pattern_polar(charts.polar, report, floor, peak, color_of)
    report_draw.contour_pattern(charts.heat, entry["theta_deg"], entry["phi_deg"], grid, floor, peak,
                                reports.quantity_label(plot, unit), unit)
