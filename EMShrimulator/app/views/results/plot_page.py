"""One user report under Results, drawn from the newest run in any display its kind allows."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.background import BackgroundRunner
from EMShrimulator.app.notify import guarded, notify
from EMShrimulator.app.theme import trace_color
from EMShrimulator.app.views.results import report_draw
from EMShrimulator.app.views.results.chart import Chart
from EMShrimulator.app.views.results.data_table import DataTable
from EMShrimulator.app.widgets.form import ChoiceBox, fmt_eng, form_layout, hint, row
from EMShrimulator.core.em import farfield as ff
from EMShrimulator.core.em import reports
from EMShrimulator.core.em.edits import DISPLAY_LABELS, DISPLAYS, REPORT_LABELS, find_plot
from EMShrimulator.core.em.model import ReportTrace

PATTERN_3D = {"rect3d", "spherical", "cylindrical", "overlay"}
SWEEPS = [("theta", "θ, cuts at φ"), ("phi", "φ, cuts at θ")]
SCALES = [(True, "dB"), (False, "Linear")]
TDR_AXES = [("distance", "Distance"), ("time", "Time")]
COMPONENTS = [("total", "Magnitude"), ("x", "x component"), ("y", "y component"), ("z", "z component")]
AXIS_CHOICES = [("", "Automatic"), ("x", "x"), ("y", "y"), ("z", "z")]
Y_AXES = [("auto", "Automatic"), ("y1", "Y1 (left)"), ("y2", "Y2 (right)")]
DUMP_QUANTITY = {"e_freq": "E", "h_freq": "H"}
Draw = Tuple[List[str], List[List[str]]]


def _parse_list(text: str) -> List[float]:
    values = [float(part) for part in text.replace(";", ",").split(",") if part.strip()]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError
    return values


def _parse_optional(text: str) -> Optional[float]:
    text = text.strip()
    if not text:
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError
    return value


def _parse_angle(text: str) -> float:
    value = float(text.strip() or 0)
    if not math.isfinite(value):
        raise ValueError
    return value


class PlotPage(QWidget):
    def __init__(self, session, source, *, near_loader: Optional[Callable[[str, int], Optional[Dict]]] = None,
                 scene_provider: Optional[Callable[[], object]] = None, parent=None):
        """`source` is the results view; `near_loader(dump, port)` and `scene_provider()` are the 3D Modeler's."""
        super().__init__(parent)
        self.session = session
        self.source = source
        self.near_loader = near_loader
        self.scene_provider = scene_provider
        self.plot_id: Optional[str] = None
        self._syncing = False
        self._dirty = True
        self._curves: List[ReportTrace] = []
        self._export: Draw = ([], [])
        self._near: Dict[Tuple, Optional[Dict]] = {}
        self._near_pending: Optional[Tuple] = None
        self._background = BackgroundRunner(self)
        self._pattern3d = None

        self.kind_label = QLabel()
        self.kind_label.setProperty("role", "subheading")
        self.display = ChoiceBox(tooltip="How the report is drawn")
        self.display.committed.connect(lambda v: self._set("display", v))

        # ------------------------------------------------------------- traces
        self.traces_label = QLabel("Traces")
        self.traces_label.setProperty("role", "subheading")
        self.traces = QListWidget()
        self.traces.setObjectName("trace-list")
        self.traces.setMinimumHeight(96)
        self.traces.setToolTip("Untick a trace to hide it; Remove takes the selected one out of the report")
        self.traces.itemChanged.connect(self._trace_toggled)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_trace)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self._set_curves([]))
        self.trace_buttons = row(remove, clear)
        self.add_box = QWidget()
        add_form = form_layout()
        self.add_box.setLayout(add_form)
        self.add_category = ChoiceBox(tooltip="What kind of quantity: S, Y, Z, VSWR, gain…")
        self.add_category.committed.connect(lambda _v: self._fill_add_box())
        self.add_term = ChoiceBox(tooltip="Which: S11, Z21, a polarisation, an antenna parameter")
        self.add_function = ChoiceBox(tooltip="How it is taken: dB, magnitude, phase, real, imaginary")
        self.add_axis = ChoiceBox(Y_AXES, tooltip="Automatic puts a second unit on the right-hand axis")
        self.add_theta = QLineEdit("0")
        self.add_theta.setToolTip("θ of the direction this far-field trace is read at, in degrees")
        self.add_phi = QLineEdit("0")
        self.add_phi.setToolTip("φ of the direction this far-field trace is read at, in degrees")
        self.add_direction = row(QLabel("θ"), self.add_theta, QLabel("φ"), self.add_phi, stretch_last=True)
        add_button = QPushButton("Add trace")
        add_button.clicked.connect(self.add_trace)
        self.update_button = QPushButton("Update")
        self.update_button.setToolTip("Change the selected trace to these settings")
        self.update_button.clicked.connect(self.update_trace)
        add_form.addRow("Category", self.add_category)
        add_form.addRow("Quantity", self.add_term)
        add_form.addRow("Function", self.add_function)
        add_form.addRow("Axis", self.add_axis)
        add_form.addRow("At (°)", self.add_direction)
        add_form.addRow("", row(add_button, self.update_button))
        self.traces.currentRowChanged.connect(self._trace_selected)
        self.add_form = add_form
        self.log_x = QCheckBox("Log frequency axis")
        self.log_x.clicked.connect(lambda on: self._set("log_x", bool(on)))

        # ------------------------------------------------------------ options
        self.options = form_layout()
        self.rows: Dict[str, QWidget] = {}
        self.frequency = self._choice("frequency_hz", "Frequency", [], convert=float)
        self.all_frequencies = QCheckBox("Every far-field frequency")
        self.all_frequencies.setToolTip("Overlay the cuts at each frequency the far field was computed at")
        self.all_frequencies.clicked.connect(lambda on: self._set("all_frequencies", bool(on)))
        self.options.addRow("", self.all_frequencies)
        self.rows["all_frequencies"] = self.all_frequencies
        self.cross = QCheckBox("Show the cross-polarisation")
        self.cross.setToolTip("Beside each polarised trace, the polarisation orthogonal to it, dashed: φ for θ, "
                              "LHCP for RHCP, Ludwig-3 Y for X")
        self.cross.clicked.connect(lambda on: self._set("cross_pol", bool(on)))
        self.options.addRow("", self.cross)
        self.rows["cross"] = self.cross
        self.sweep = self._choice("sweep", "Cut along", SWEEPS)
        self.cuts = self._line("cuts", "Cuts at (°)", "0, 90", _parse_list,
                               "The angles the cuts are held at, comma separated: φ for cuts along θ (an elevation "
                               "pattern: 0, 90), θ for cuts along φ (an azimuth pattern: 90)")
        self.range_db = self._line("range_db", "Range (dB)", "40", lambda t: max(float(t), 3.0),
                                   "How far below the peak a polar, contour or 3D pattern reaches")
        self.tdr_axis = self._choice("tdr_axis", "Axis", TDR_AXES)
        self.rise = self._line("rise_time_ps", "Rise time (ps)", "fastest the sweep allows", _parse_optional,
                               "10-90 % rise time of the TDR step; leave empty for the sharpest edge the band allows")
        self.dump = self._choice("dump", "Field dump", [])
        self.port = self._choice("port", "Excitation", [], convert=int)
        self.field_component = self._choice("field_component", "Component", COMPONENTS)
        self.scale = self._choice("db", "Scale", SCALES)
        self.slice_axis = self._choice("slice_axis", "Plane across", AXIS_CHOICES, convert=lambda v: v or None)
        self.slice_mm = self._line("slice_mm", "Plane at (mm)", "middle", _parse_optional,
                                   "Where a box dump is cut; leave empty for its middle")
        self.line_axis = self._choice("line_axis", "Line along", AXIS_CHOICES, convert=lambda v: v or None)
        self.line_mm = self._line("line_mm", "Line at (mm)", "middle", _parse_optional,
                                  "Where the line crosses the plane's other axis; leave empty for the middle")
        self.export_button = QPushButton("Export data…")
        self.export_button.setToolTip("Save what this report shows as a CSV file")
        self.export_button.clicked.connect(self.export_data)
        self.summary = hint()
        self.summary.setWordWrap(True)

        controls = QWidget()
        controls.setObjectName("plot-controls")
        side = QVBoxLayout(controls)
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(8)
        side.addWidget(self.kind_label)
        display_form = form_layout()
        display_form.addRow("Display", self.display)
        side.addLayout(display_form)
        side.addWidget(self.traces_label)
        side.addWidget(self.traces, 1)
        side.addWidget(self.trace_buttons)
        side.addWidget(self.add_box)
        side.addWidget(self.log_x)
        side.addLayout(self.options)
        side.addWidget(self.export_button)
        side.addWidget(self.summary)
        self.side_stretch = QWidget()
        side.addWidget(self.side_stretch, 1)
        scroll = QScrollArea()
        scroll.setObjectName("plot-controls-scroll")
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(300)

        self.chart = Chart()
        self.table = DataTable()
        self.message = QLabel()
        self.message.setProperty("role", "hint")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)
        self.stack = QStackedWidget()
        for widget in (self.message, self.chart, self.table):
            self.stack.addWidget(widget)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(scroll)
        layout.addWidget(self.stack, 1)

        source.loaded.connect(self.refresh)
        session.projectChanged.connect(self.refresh)

    # ---------------------------------------------------------------- controls

    def _choice(self, field: str, label: str, options, convert=None) -> ChoiceBox:
        box = ChoiceBox(options)
        box.committed.connect(lambda v: self._set(field, convert(v) if convert else v))
        self.options.addRow(label, box)
        self.rows[field] = box
        return box

    def _line(self, field: str, label: str, placeholder: str, parse, tooltip: str) -> QLineEdit:
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(tooltip)

        def commit():
            if self._syncing:
                return
            try:
                value = parse(edit.text())
            except (TypeError, ValueError):
                notify(f"{label}: '{edit.text()}' is not a number this takes.", "error")
                self.refresh()
                return
            self._set(field, value)

        edit.editingFinished.connect(commit)
        self.options.addRow(label, edit)
        self.rows[field] = edit
        return edit

    def _show_rows(self, shown) -> None:
        for name, widget in self.rows.items():
            self.options.setRowVisible(widget, name in shown)

    @staticmethod
    def _set_text(edit: QLineEdit, value) -> None:
        if edit.hasFocus():
            return
        if value is None:
            edit.setText("")
        elif isinstance(value, list):
            edit.setText(", ".join(f"{v:g}" for v in value))
        else:
            edit.setText(f"{value:g}")

    # ---------------------------------------------------------------- state

    def show_plot(self, plot_id: str) -> None:
        if plot_id != self.plot_id:
            self.plot_id = plot_id
            self.chart.reset_view()
        self.refresh()

    def _plot(self):
        return find_plot(self.session.project, self.plot_id) if self.plot_id else None

    def _set(self, field: str, value) -> None:
        if self._syncing or self.plot_id is None:
            return
        plot = self._plot()
        if plot is not None and getattr(plot, field, object()) == value:
            return
        s, plot_id = self.session, self.plot_id
        guarded(lambda: s.push(cmd.SetPlotField(s, plot_id, field, value)))

    def _set_curves(self, curves: List[ReportTrace]) -> None:
        self._set("curves", [c.model_dump() for c in curves])

    # ---------------------------------------------------------------- traces

    def _frequency_report(self, plot) -> bool:
        """Whether the report takes traces against frequency."""
        return plot.kind in reports.FREQUENCY_KINDS and not (plot.kind == "antenna" and plot.display == "table"
                                                             and not plot.curves)

    def _trace_from_box(self) -> Optional[ReportTrace]:
        category, term = self.add_category.value(), self.add_term.value()
        if not category or term is None:
            return None
        try:
            theta, phi = _parse_angle(self.add_theta.text()), _parse_angle(self.add_phi.text())
        except ValueError:
            notify("The direction takes θ and φ in degrees.", "error")
            return None
        return ReportTrace(category=category, term=term, function=self.add_function.value() or "value",
                           axis=self.add_axis.value() or "auto", theta_deg=theta, phi_deg=phi)

    def add_trace(self) -> None:
        plot, trace = self._plot(), self._trace_from_box()
        if plot is None or trace is None:
            return
        if any(c.model_dump() == trace.model_dump() for c in self._curves):
            notify(f"{reports.trace_label(trace, self._frequency_report(plot))} is already in this report.", "info")
            return
        self._set_curves(self._curves + [trace])

    def update_trace(self) -> None:
        """Apply the add-trace box to the selected trace."""
        index, trace = self.traces.currentRow(), self._trace_from_box()
        if trace is None or not 0 <= index < len(self._curves):
            notify("Select the trace to change in the list first.", "info")
            return
        trace.visible = self._curves[index].visible
        self._set_curves(self._curves[:index] + [trace] + self._curves[index + 1:])

    def _trace_selected(self, index: int) -> None:
        """A trace chosen in the list shows its settings in the box, ready to change."""
        self.update_button.setEnabled(0 <= index < len(self._curves))
        if self._syncing or not 0 <= index < len(self._curves):
            return
        curve = self._curves[index]
        self.add_category.set_value(curve.category)
        self._fill_add_box()
        self.add_term.set_value(curve.term)
        self.add_function.set_value(curve.function)
        self.add_axis.set_value(curve.axis)
        self.add_theta.setText(f"{curve.theta_deg:g}")
        self.add_phi.setText(f"{curve.phi_deg:g}")

    def remove_trace(self) -> None:
        row_ = self.traces.currentRow()
        if 0 <= row_ < len(self._curves):
            self._set_curves(self._curves[:row_] + self._curves[row_ + 1:])

    def _trace_toggled(self, item: QListWidgetItem) -> None:
        if self._syncing:
            return
        index = self.traces.row(item)
        if 0 <= index < len(self._curves):
            curves = [c.model_copy() for c in self._curves]
            curves[index].visible = item.checkState() == Qt.Checked
            self._set_curves(curves)

    def _fill_traces(self, plot, frequency: bool) -> None:
        texts = []
        for i, curve in enumerate(self._curves):
            text = reports.trace_label(curve, at_direction=frequency)
            if curve.axis != "auto":
                text += f"   · {curve.axis.upper()}"
            texts.append(text)
        current = [self.traces.item(i).text() for i in range(self.traces.count())]
        if current != texts:
            selected = self.traces.currentRow()
            self.traces.clear()
            for i, text in enumerate(texts):
                item = QListWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                if frequency or plot.kind == "tdr":
                    item.setData(Qt.DecorationRole, QColor(trace_color(i)))
                self.traces.addItem(item)
            if 0 <= selected < self.traces.count():
                self.traces.setCurrentRow(selected)
        for i, curve in enumerate(self._curves):
            self.traces.item(i).setCheckState(Qt.Checked if curve.visible else Qt.Unchecked)
        self.update_button.setEnabled(0 <= self.traces.currentRow() < len(self._curves))

    def _fill_add_box(self) -> None:
        plot = self._plot()
        if plot is None:
            return
        data, farfield = self.source.data or {}, self.source.farfield or {}
        categories = reports.categories_for(plot.kind, data, farfield)
        self.add_category.set_options([(c, reports.CATEGORIES[c]) for c in categories])
        category = self.add_category.value()
        if category is None and categories:
            self.add_category.set_value(categories[0])
            category = categories[0]
        terms = reports.terms_for(category, data, farfield) if category else []
        self.add_term.set_options(terms)
        if self.add_term.value() is None and terms:
            self.add_term.set_value(terms[0][0])
        functions = reports.functions_for(category) if category else []
        self.add_function.set_options([(f, reports.TRACE_FUNCTIONS[f]) for f in functions])
        if self.add_function.value() is None and functions:
            self.add_function.set_value(functions[0])
        far_at = self._frequency_report(plot) and category in reports.FARFIELD
        self.add_form.setRowVisible(self.add_direction, far_at)
        self.add_form.setRowVisible(self.add_function, len(functions) > 1)
        self.add_form.setRowVisible(self.add_term, len(terms) > 1 or category != "axial_ratio")
        self.add_form.setRowVisible(self.add_axis, plot.display == "rect" or plot.kind not in ("farfield", "tdr"))
        self.add_box.setEnabled(bool(categories))

    # ---------------------------------------------------------------- draw

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def refresh(self) -> None:
        # A hidden plot only notes that it is out of date.
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        plot = self._plot()
        if plot is None:
            return
        data = self.source.data or {}
        farfield = self.source.farfield or {}
        kind, display = plot.kind, plot.display
        frequency = self._frequency_report(plot)
        self.kind_label.setText(REPORT_LABELS.get(kind, kind))
        self._curves = reports.curves_of(plot, data, farfield) if kind != "nearfield" else []
        near = self._near_data(plot, data) if kind == "nearfield" else None

        self._syncing = True
        try:
            self.display.set_options([(d, DISPLAY_LABELS[d]) for d in DISPLAYS.get(kind, (display,))])
            self.display.set_value(display)
            self._fill_traces(plot, frequency)
            self._fill_add_box()
            self.log_x.setChecked(plot.log_x)
            self._fill_options(plot, farfield, data, near)
        finally:
            self._syncing = False

        uses_traces = kind in ("farfield", "tdr") or frequency
        for widget in (self.traces_label, self.traces, self.trace_buttons, self.add_box):
            widget.setVisible(uses_traces)
        self.side_stretch.setVisible(not uses_traces)
        self.log_x.setVisible(frequency and display in ("rect", "table"))
        self._show_rows(self._shown_rows(plot))
        self.summary.setText("")
        self._export = ([], [])
        self.export_button.setEnabled(False)

        if frequency:
            self._draw_frequency(plot, data, farfield)
        elif kind == "tdr":
            self._draw_tdr(plot, data)
        elif kind == "farfield":
            self._draw_farfield(plot, farfield)
        elif kind == "antenna":
            self._draw_antenna_table(plot, farfield)
        elif kind == "nearfield":
            self._draw_nearfield(plot, data, near)
        self.export_button.setEnabled(bool(self._export[0]))

    def _fill_options(self, plot, farfield: Dict, data: Dict, near: Optional[Dict]) -> None:
        if plot.kind == "farfield":
            entries = farfield.get("frequencies") or []
            self.frequency.set_options([(float(e["frequency_hz"]), fmt_eng(e["frequency_hz"], "Hz")) for e in entries])
            entry = reports.pattern_entry(farfield, plot.frequency_hz)
            if entry is not None:
                self.frequency.set_value(float(entry["frequency_hz"]))
        elif plot.kind == "nearfield":
            freqs = [] if near is None else [float(f) for f in near["frequency_hz"]]
            self.frequency.set_options([(f, fmt_eng(f, "Hz")) for f in freqs])
            if freqs:
                self.frequency.set_value(freqs[reports.near_frequency(near, plot.frequency_hz)])
            dumps = self._dumps(data)
            names = list(dict.fromkeys(d["dump"] for d in dumps))
            self.dump.set_options([(n, n) for n in names])
            dump = self._dump_of(plot, dumps)
            if dump:
                self.dump.set_value(dump)
                ports = sorted(d["port"] for d in dumps if d["dump"] == dump)
                self.port.set_options([(p, f"Port {p} excited") for p in ports])
                self.port.set_value(self._port_of(plot, dumps))
        for box, value in ((self.sweep, plot.sweep), (self.scale, plot.db), (self.tdr_axis, plot.tdr_axis),
                           (self.field_component, plot.field_component), (self.slice_axis, plot.slice_axis or ""),
                           (self.line_axis, plot.line_axis or "")):
            box.set_value(value)
        self.cross.setChecked(plot.cross_pol)
        self.all_frequencies.setChecked(plot.all_frequencies)
        for edit, value in ((self.cuts, plot.cuts), (self.range_db, plot.range_db), (self.rise, plot.rise_time_ps),
                            (self.slice_mm, plot.slice_mm), (self.line_mm, plot.line_mm)):
            self._set_text(edit, value)

    @staticmethod
    def _shown_rows(plot) -> set:
        kind, display = plot.kind, plot.display
        shown = set()
        if kind == "tdr":
            shown |= {"tdr_axis", "rise_time_ps"}
        if kind == "farfield":
            cuts = display in ("rect", "polar", "table")
            shown.add("frequency_hz")
            if cuts:
                shown |= {"sweep", "cuts", "cross", "all_frequencies"}
            if display != "rect":
                shown.add("range_db")
        if kind == "nearfield":
            shown |= {"dump", "port", "frequency_hz", "field_component", "db", "slice_axis", "slice_mm"}
            if display in ("rect", "table"):
                shown |= {"line_axis", "line_mm"}
        return shown

    def _message(self, text: str) -> None:
        self.message.setText(text)
        self.stack.setCurrentWidget(self.message)

    def _show_report(self, report: reports.Report, display: str, draw) -> None:
        """Show a report on the chart or as a table, with its note, or why there is none."""
        if report.message:
            self._message(report.message)
            return
        self._export = reports.table(report)
        note = ""
        if display == "table":
            self.table.set_table(*self._export)
            self.stack.setCurrentWidget(self.table)
        else:
            note = draw() or ""
            self.stack.setCurrentWidget(self.chart)
        self.summary.setText(" ".join(part for part in (report.note, note) if part))

    @staticmethod
    def _color_of(report: reports.Report):
        return report_draw.color_map([s.key for s in report.series])

    # ------------------------------------------------------------ per kind

    def _draw_frequency(self, plot, data: Dict, farfield: Dict) -> None:
        has_far = bool(farfield.get("frequencies"))
        if not data.get("available") and not has_far:
            self._message("No results yet. Run the design (F5) and this report fills in when the run finishes.")
            return
        report = reports.frequency_report(plot, data, farfield, self._curves)
        color_of = self._color_of(report)
        freqs = data.get("freq_hz")
        if plot.display == "smith":
            self._show_report(report, "smith", lambda: report_draw.smith(self.chart, report, color_of, freqs))
        elif plot.display == "polar":
            self._show_report(report, "polar",
                              lambda: report_draw.complex_polar(self.chart, report, color_of, freqs))
        else:
            self._show_report(report, plot.display,
                              lambda: report_draw.rectangular(self.chart, report, color_of, plot.log_x))
        marker = (data.get("markers") or {}).get("resonance")
        if marker and not report.message and any(s.category in ("s", "z", "y", "vswr") for s in report.series):
            best = (f"Best match {marker['s11_db']:.1f} dB at {fmt_eng(marker['frequency_hz'], 'Hz')}"
                    + (f", −10 dB bandwidth {fmt_eng(marker['bandwidth_hz'], 'Hz')}"
                       if marker.get("bandwidth_hz") else "") + ".")
            self.summary.setText(" ".join(part for part in (best, self.summary.text()) if part))

    def _draw_tdr(self, plot, data: Dict) -> None:
        if not data.get("available"):
            self._message("No results yet. Run the design (F5) and this report fills in when the run finishes.")
            return
        keys = [c.term for c in self._curves if c.visible and c.category == "tdr"]
        report = reports.tdr(plot, data, keys)
        index = {c.term: i for i, c in enumerate(self._curves)}
        for s in report.series:
            s.key = f"c{index.get(s.key, 0)}"  # the colour its trace has in the list
        color_of = self._color_of(report)
        self._show_report(report, plot.display, lambda: report_draw.rectangular(self.chart, report, color_of))

    def _draw_farfield(self, plot, farfield: Dict) -> None:
        if plot.display in ("rect", "polar", "table"):
            report = reports.pattern_report(plot, farfield, self._curves)
            if report.message:
                self._message(report.message)
                return
            y1 = [s for s in report.series if s.axis != "y2"]
            floor, peak = reports.series_scale(plot, y1)
            color_of = self._color_of(report)
            if plot.display == "polar":
                draw = lambda: report_draw.pattern_polar(self.chart, report, floor, peak, color_of)  # noqa: E731
            else:
                draw = lambda: report_draw.rectangular(self.chart, report, color_of)  # noqa: E731
            self._show_report(report, plot.display, draw)
            return
        curve = reports.first_pattern_curve(plot, farfield)
        message = reports.cross_needed(plot, farfield)
        grid, unit, why, entry = reports.pattern_grid(plot, farfield, curve=curve)
        if grid is None:
            self._message(why)
            return
        floor, peak = reports.pattern_scale(plot, grid, unit, curve.category)
        label = reports.quantity_label(plot, unit, curve)
        theta, phi = np.asarray(entry["theta_deg"]), np.asarray(entry["phi_deg"])
        self._export = (["θ (°)", "φ (°)", label],
                        [[f"{t:g}", f"{p:g}", f"{grid[i, j]:.6g}"] for i, t in enumerate(theta)
                         for j, p in enumerate(phi)])
        shown = [c for c in self._curves if c.visible and c.category in reports.FARFIELD]
        extra = " Contour and 3D displays show the first trace." if len(shown) > 1 else ""
        self.summary.setText(reports.pattern_note(plot, entry, unit, curve) + extra + (f" {message}" if message else ""))
        if plot.display == "contour":
            report_draw.contour_pattern(self.chart, theta, phi, grid, floor, peak, label, unit)
            self.stack.setCurrentWidget(self.chart)
            return
        scene = self.scene_provider() if self.scene_provider else None
        if plot.display == "overlay" and (scene is None or scene.bounds is None):
            self._message("The pattern goes over the 3D model: open the design's model once so it is drawn, then "
                          "come back to this report.")
            return
        view = self._view3d()
        if view is None:
            return
        self.stack.setCurrentWidget(view)
        view.show_pattern(theta, phi, grid, floor=floor, peak=peak, label=label, shape=plot.display, scene=scene,
                          centre_mm=entry.get("center_mm"))

    def _draw_antenna_table(self, plot, farfield: Dict) -> None:
        report, table = reports.antenna(plot, farfield, [])
        if report.message:
            self._message(report.message)
            return
        entries = farfield.get("frequencies") or []
        columns = ["Parameter", "Unit"] + [f"{float(e['frequency_hz']) / 1e9:.6g} GHz" for e in entries]
        rows = [[row_["label"], row_["unit"]] + ["—" if v is None else f"{float(v):.6g}" for v in row_["values"]]
                for row_ in table.values()]
        self._export = (columns, rows)
        self.table.set_table(columns, rows)
        self.stack.setCurrentWidget(self.table)
        self.summary.setText(report.note)

    # ------------------------------------------------------------ near field

    @staticmethod
    def _dumps(data: Dict) -> List[Dict]:
        return list(data.get("nearfield") or [])

    @staticmethod
    def _dump_of(plot, dumps: List[Dict]) -> str:
        names = [d["dump"] for d in dumps]
        return plot.dump if plot.dump in names else (names[0] if names else "")

    def _port_of(self, plot, dumps: List[Dict]) -> Optional[int]:
        dump = self._dump_of(plot, dumps)
        ports = sorted(d["port"] for d in dumps if d["dump"] == dump)
        return plot.port if plot.port in ports else (ports[0] if ports else None)

    def _near_data(self, plot, data: Dict) -> Optional[Dict]:
        """The dump's data if it is loaded; otherwise None, and its loading is started."""
        dumps = self._dumps(data)
        dump, port = self._dump_of(plot, dumps), self._port_of(plot, dumps)
        if not dump or port is None or self.near_loader is None:
            return None
        key = (dump, port, id(self.source.data))
        if key in self._near:
            return self._near[key]
        if self._near_pending != key:
            self._near_pending = key
            loader = self.near_loader
            self._background.submit("near", lambda: loader(dump, port), lambda result: self._near_loaded(key, result),
                                    lambda exc: self._near_failed(key, exc))
        return None

    def _near_loaded(self, key: Tuple, result: Optional[Dict]) -> None:
        self._near = {key: result}  # one dump at a time: a box dump can be large
        self._near_pending = None
        self.refresh()

    def _near_failed(self, key: Tuple, exc: Exception) -> None:
        self._near_pending = None
        self._near = {key: None}
        notify(f"Could not read the field dump: {exc}", "error")
        self.refresh()

    def _draw_nearfield(self, plot, data: Dict, near: Optional[Dict]) -> None:
        dumps = self._dumps(data)
        if self.near_loader is None:
            self._message("Near-field reports read the 3D Modeler's field dumps.")
            return
        if not dumps:
            self._message("No field dumps in the newest run. Add one under Field dumps, as an E or H field in the "
                          "frequency domain at the frequencies you want, and run the design.")
            return
        dump, port = self._dump_of(plot, dumps), self._port_of(plot, dumps)
        key = (dump, port, id(self.source.data))
        if key not in self._near:
            self._message(f"Reading {dump}…")
            return
        if near is None:
            self._message(f"{dump} has no frequency-domain data: near-field reports read E or H field dumps of "
                          "the frequency kind.")
            return
        kinds = {d.name: d.kind.value for d in getattr(self.session.project, "dumps", [])}
        quantity = DUMP_QUANTITY.get(kinds.get(dump, ""), "E")
        frequency = float(near["frequency_hz"][reports.near_frequency(near, plot.frequency_hz)])
        incident = reports.incident_power(data.get("summary"), port, frequency)
        if plot.display == "contour":
            cut = reports.near_plane(plot, near, quantity, incident)
            label = reports.near_label(plot, quantity, cut["unit"])
            report_draw.contour_plane(self.chart, cut, label)
            self.stack.setCurrentWidget(self.chart)
            self._export = ([f"{cut['u']} (mm)", f"{cut['v']} (mm)", label],
                            [[f"{u:.6g}", f"{v:.6g}", f"{cut['values'][j, i]:.6g}"]
                             for j, v in enumerate(cut["v_mm"]) for i, u in enumerate(cut["u_mm"])])
            self.summary.setText(reports.near_note(cut, port))
            return
        report = reports.near_line(plot, near, quantity, incident, port)
        color_of = self._color_of(report)
        self._show_report(report, plot.display, lambda: report_draw.rectangular(self.chart, report, color_of))

    # ------------------------------------------------------------ 3D, export

    def _view3d(self):
        if self._pattern3d is None:
            try:
                from EMShrimulator.app.views.results.pattern3d import Pattern3D

                self._pattern3d = Pattern3D()
            except Exception as exc:  # noqa: BLE001 - no 3D graphics: say so, keep the 2D displays
                self._message(f"The 3D view could not start: {exc}")
                return None
            self.stack.addWidget(self._pattern3d)
        return self._pattern3d

    def export_data(self) -> None:
        plot = self._plot()
        columns, rows = self._export
        if plot is None or not columns:
            notify("This report has nothing to export yet.", "info")
            return
        target, _ = QFileDialog.getSaveFileName(self, "Export report data", str(Path.home() / f"{plot.name}.csv"),
                                                "CSV (*.csv)")
        if not target:
            return
        Path(target).write_text(reports.to_csv(columns, rows), encoding="utf-8")
        notify(f"Saved {Path(target).name}.", "ok")

    def shutdown(self) -> None:
        self._background.shutdown()
        if self._pattern3d is not None:
            self._pattern3d.close_view()
