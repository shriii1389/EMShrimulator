"""Results panel for both design types, loaded from the Touchstone on a worker thread."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app.background import BackgroundRunner
from EMShrimulator.app.models.results_model import TraceListModel, is_reflection
from EMShrimulator.app.notify import notify
from EMShrimulator.app.views.results.chart import Chart
from EMShrimulator.app.views.results.farfield_plot import FarfieldCharts, plot_farfield
from EMShrimulator.app.views.results.smith_chart import plot_smith
from EMShrimulator.app.views.results.sparam_plot import plot_frequency_view, plot_tdr
from EMShrimulator.app.views.results.sweep_overlay import SweepOverlay
from EMShrimulator.app.widgets.form import ChoiceBox, fmt_eng

VIEWS = [("magnitude", "Magnitude"), ("phase", "Phase"), ("smith", "Smith"), ("vswr", "VSWR"),
         ("impedance", "Impedance"), ("delay", "Group delay"), ("tdr", "TDR")]


def _impedances_text(data: Dict) -> str:
    """One reference impedance, or each port's in order when they differ."""
    values = data.get("z0_ports") or ([data["z0"]] if data.get("z0") is not None else [])
    if not values:
        return "—"
    return f"{values[0]:g}" if len(set(values)) == 1 else " / ".join(f"{z:g}" for z in values)


class ResultsView(QWidget):
    """`loader()` returns the results dict; `farfield_loader` and `sweep_loader(slug)` are modeler-only."""

    loaded = Signal()

    def __init__(self, loader: Callable[[], Dict], *, farfield_loader: Optional[Callable[[], Optional[Dict]]] = None,
                 sweep_loader: Optional[Callable[[str], Dict]] = None, parent=None):
        super().__init__(parent)
        self.loader = loader
        self.farfield_loader = farfield_loader
        self.sweep_loader = sweep_loader
        self.data: Optional[Dict] = None
        self.farfield: Optional[Dict] = None
        self._sweep_slug: Optional[str] = None
        self._background = BackgroundRunner(self)

        self.view_box = ChoiceBox()
        self.view_box.committed.connect(lambda _v: self.redraw())
        self.log_x = QCheckBox("log frequency")
        self.log_x.toggled.connect(lambda _on: self.redraw())
        self.file_label = QLabel()
        self.file_label.setProperty("role", "note")
        self.export_button = QPushButton("Save Touchstone…")
        self.export_button.clicked.connect(self.export_touchstone)
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload)
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 4)
        bar.addWidget(QLabel("View"))
        bar.addWidget(self.view_box)
        bar.addWidget(self.log_x)
        bar.addStretch(1)
        bar.addWidget(self.file_label)
        bar.addWidget(self.export_button)
        bar.addWidget(self.reload_button)

        self.readout = QWidget()
        grid = QGridLayout(self.readout)
        grid.setContentsMargins(8, 0, 8, 4)
        self.cells: Dict[str, QLabel] = {}
        for i, key in enumerate(("Ports", "Sweep", "Points", "Best match", "−10 dB bandwidth")):
            caption = QLabel(key)
            caption.setProperty("role", "note")
            value = QLabel("—")
            value.setProperty("role", "value")
            grid.addWidget(caption, 0, i)
            grid.addWidget(value, 1, i)
            self.cells[key] = value

        self.traces = TraceListModel(self)
        self.traces.selectionChanged.connect(self.redraw)
        trace_view = QListView()
        trace_view.setModel(self.traces)
        all_button = QPushButton("All")
        all_button.clicked.connect(lambda: self.traces.set_all(True))
        none_button = QPushButton("None")
        none_button.clicked.connect(lambda: self.traces.set_all(False))
        self.trace_panel = QWidget()
        tp = QVBoxLayout(self.trace_panel)
        tp.setContentsMargins(0, 0, 0, 0)
        tp.addWidget(trace_view, 1)
        buttons = QHBoxLayout()
        buttons.addWidget(all_button)
        buttons.addWidget(none_button)
        tp.addLayout(buttons)

        self.canvas = Chart()
        self.farfield_panel = QWidget()
        fp = QVBoxLayout(self.farfield_panel)
        fp.setContentsMargins(0, 0, 0, 0)
        ff_bar = QHBoxLayout()
        self.ff_frequency = ChoiceBox()
        self.ff_frequency.committed.connect(lambda _i: self.redraw())
        self.ff_summary = QLabel()
        self.ff_summary.setProperty("role", "hint")
        ff_bar.addWidget(QLabel("Frequency"))
        ff_bar.addWidget(self.ff_frequency)
        ff_bar.addWidget(self.ff_summary, 1)
        self.ff_canvas = FarfieldCharts()
        fp.addLayout(ff_bar)
        fp.addWidget(self.ff_canvas, 1)

        # Built when a sweep first appears.
        self._sweep: Optional[SweepOverlay] = None

        self.plots = QListWidget()
        self.plots.setViewMode(QListView.IconMode)
        self.plots.setIconSize(QSize(360, 260))
        self.plots.setResizeMode(QListView.Adjust)
        self.plots.setSpacing(8)
        self.plots.itemDoubleClicked.connect(
            lambda item: QDesktopServices.openUrl(QUrl.fromLocalFile(str(item.data(Qt.UserRole)))))

        self.message = QLabel("Loading results…")
        self.message.setProperty("role", "hint")
        self.message.setAlignment(Qt.AlignCenter)
        self.message.setWordWrap(True)

        self.chart_split = QSplitter(Qt.Horizontal)
        self.chart_split.addWidget(self.trace_panel)
        self.chart_split.addWidget(self.canvas)
        self.chart_split.setSizes([140, 900])

        self.stack = QStackedWidget()
        for widget in (self.message, self.chart_split, self.farfield_panel, self.plots):
            self.stack.addWidget(widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(bar)
        layout.addWidget(self.readout)
        layout.addWidget(self.stack, 1)

    @property
    def sweep(self) -> SweepOverlay:
        if self._sweep is None:
            self._sweep = SweepOverlay()
            self._sweep.sweepChosen.connect(self._load_sweep)
            self.stack.addWidget(self._sweep)
        return self._sweep

    # ----------------------------------------------------------- loading

    def reload(self) -> None:
        self.message.setText("Loading results…")
        if self.data is None:
            self.stack.setCurrentWidget(self.message)

        def load():
            data = self.loader()
            farfield = self.farfield_loader() if self.farfield_loader else None
            return data, farfield

        self._background.submit("results", load, self._loaded, self._failed)

    def _load_sweep(self, slug: str) -> None:
        if not self.sweep_loader or not slug:
            return
        self._sweep_slug = slug
        loader = self.sweep_loader
        self._background.submit("sweep", lambda: loader(slug), self.sweep.show_results,
                                lambda exc: notify(f"Could not load the sweep: {exc}", "error"))

    def _failed(self, exc: Exception) -> None:
        self.data = None
        self.message.setText(f"Could not load results: {exc}")
        self.stack.setCurrentWidget(self.message)
        self.loaded.emit()

    def _loaded(self, payload) -> None:
        self.data, self.farfield = payload
        self._rebuild_views()
        self.loaded.emit()

    # ------------------------------------------------------------ views

    def _rebuild_views(self) -> None:
        data = self.data or {}
        views = list(VIEWS) if data.get("available") else []
        ff_entries = (self.farfield or {}).get("frequencies") or []
        if ff_entries:
            views.append(("farfield", "Far field"))
        sweeps = data.get("sweeps") or []
        if sweeps:
            views.append(("sweep", f"Sweeps ({len(sweeps)})" if len(sweeps) > 1 else "Sweep"))
        if data.get("plots"):
            views.append(("plots", "gerber2ems plots"))
        current = self.view_box.value()
        self.view_box.set_options(views)
        if current in [v for v, _ in views]:
            self.view_box.set_value(current)

        self.traces.set_keys(list((data.get("traces") or {}).keys()))
        freqs = data.get("freq_hz") or []
        marker = (data.get("markers") or {}).get("resonance")
        self.cells["Ports"].setText(f"{data.get('ports', '—')} · Z₀ {_impedances_text(data)} Ω"
                                    if data.get("available") else "—")
        self.cells["Sweep"].setText(f"{fmt_eng(freqs[0], 'Hz')} – {fmt_eng(freqs[-1], 'Hz')}" if freqs else "—")
        self.cells["Points"].setText(str(len(freqs)) if freqs else "—")
        self.cells["Best match"].setText(f"{marker['s11_db']:.1f} dB @ {fmt_eng(marker['frequency_hz'], 'Hz')}"
                                         if marker else "—")
        self.cells["−10 dB bandwidth"].setText(fmt_eng(marker.get("bandwidth_hz"), "Hz") if marker else "—")
        self.file_label.setText(data.get("file") or "")
        notes, flags = [], []
        excess = data.get("passivity_excess") or {}
        if excess:
            at = fmt_eng(freqs[excess["index"]], "Hz") if freqs else "—"
            flags.append(f"⚠ not passive (+{excess['excess']:.0%} power)")
            notes.append(f"Port {excess['port']} driven, the power leaving the ports adds up to "
                         f"{1 + excess['excess']:.1%} of what went in at {at}: a passive board cannot gain energy. "
                         "Results from a run stopped too early, or from an older version of this app, do this; "
                         "run the design again.")
        if data.get("mixed_note"):
            flags.append("⚠ mixed-mode incomplete")
            notes.append(data["mixed_note"])
        self.file_label.setToolTip("\n\n".join(notes))
        if flags:
            self.file_label.setText("  ·  ".join([data.get("file") or ""] + flags))
        self.export_button.setEnabled(bool(data.get("path")))

        self.ff_frequency.set_options([(i, fmt_eng(e["frequency_hz"], "Hz")) for i, e in enumerate(ff_entries)])
        if ff_entries and self.ff_frequency.value() is None:
            self.ff_frequency.set_value(0)

        if sweeps:
            slug = self._sweep_slug if any(s["slug"] == self._sweep_slug for s in sweeps) else sweeps[0]["slug"]
            self.sweep.set_sweeps(sweeps, slug)
            self._load_sweep(slug)

        self.plots.clear()
        for path in data.get("plots") or []:
            item = QListWidgetItem(QIcon(QPixmap(path)), Path(path).name)
            item.setData(Qt.UserRole, path)
            self.plots.addItem(item)

        if not views:
            self.message.setText("No results yet. Run the design — the S-matrix, Smith chart, impedance and TDR "
                                 "views all appear here once it finishes.")
            self.stack.setCurrentWidget(self.message)
            self.readout.setVisible(False)
            return
        self.redraw()

    def redraw(self) -> None:
        view = self.view_box.value()
        data = self.data or {}
        if view is None:
            return
        frequency_view = view in dict(VIEWS)
        self.readout.setVisible(frequency_view)
        self.log_x.setVisible(frequency_view and view not in ("smith", "tdr") or view == "sweep")
        self.trace_panel.setVisible(frequency_view and view != "tdr")
        keys = self.traces.enabled_keys()

        if view == "farfield":
            entries = (self.farfield or {}).get("frequencies") or []
            entry = entries[int(self.ff_frequency.value() or 0)] if entries else None
            if entry:
                hpbw = lambda v: f"{v:.0f}°" if v else "—"  # noqa: E731
                gain = (f"gain {entry['gain_dbi']:.2f} dBi · realized {entry['realized_gain_dbi']:.2f} dBi · "
                        if entry.get("gain_dbi") is not None and entry.get("realized_gain_dbi") is not None else "")
                consistency = entry.get("consistency")
                warn = (f" · ⚠ far field carries {consistency:.0%} of the box's power: raise the air margin"
                        if consistency is not None and abs(consistency - 1) > 0.15 else "")
                self.ff_summary.setText(
                    f"Peak directivity {entry['dmax_dbi']:.2f} dBi · {gain}"
                    f"main beam θ {entry['peak_theta_deg']:.0f}° φ {entry['peak_phi_deg']:.0f}° · "
                    f"F/B {entry['front_to_back_db']:.1f} dB · HPBW φ0°/φ90° {hpbw(entry['hpbw_e_deg'])} / "
                    f"{hpbw(entry['hpbw_h_deg'])}{warn}")
                plot_farfield(self.ff_canvas, self.farfield or {}, entry)
            self.stack.setCurrentWidget(self.farfield_panel)
        elif view == "sweep":
            self.sweep.log_x = self.log_x.isChecked()
            self.sweep.redraw()
            self.stack.setCurrentWidget(self.sweep)
        elif view == "plots":
            self.stack.setCurrentWidget(self.plots)
        else:
            if view == "smith":
                plot_smith(self.canvas, data, [k for k in keys if is_reflection(k)], self.traces.color_of)
            elif view == "tdr":
                plot_tdr(self.canvas, data.get("tdr"))
            else:
                plot_frequency_view(self.canvas, view, data, keys, self.traces.color_of, self.log_x.isChecked())
            self.stack.setCurrentWidget(self.chart_split)

    def export_touchstone(self) -> None:
        source = (self.data or {}).get("path")
        if not source:
            return
        target, _ = QFileDialog.getSaveFileName(self, "Save Touchstone", str(Path.home() / Path(source).name),
                                                f"Touchstone (*{Path(source).suffix})")
        if target:
            shutil.copyfile(source, target)
            notify(f"Saved {Path(target).name}.", "ok")

    def shutdown(self) -> None:
        self._background.shutdown()
