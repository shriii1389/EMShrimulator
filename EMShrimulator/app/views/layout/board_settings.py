"""The board's solution setup page: one scrolling page of cards; `scroll_to` jumps to one."""
from __future__ import annotations

from typing import Dict

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.dielectric_loss import LOSS_MODELS, LOSS_TOOLTIP
from EMShrimulator.app.widgets.form import ChoiceBox, Segmented, card, centred_column, form_layout, row
from EMShrimulator.app.widgets.unit_spinbox import IntSpinBox, UnitSpinBox

EXPORT_FIELDS = ["outer", "cu-outer", "cu-inner", "substrate"]
DELTA_S_PRESETS = [(0.2, "0.2 fast"), (0.1, "0.1"), (0.02, "0.02"), (0.01, "0.01"), (0.005, "0.005 fine")]
SECTIONS = ("frequency", "convergence", "mesh", "export", "nets")


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()


class BoardSettings(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        self.setObjectName("setup-page")
        s = session

        def setting(path: str, label: str = ""):
            return lambda value: guarded(lambda: s.push(cmd.SetBoardSetting(s, path, value, label)))

        self.cards: Dict[str, QFrame] = {}

        # ----------------------------------------------------- frequencies
        frame, layout = card("Frequencies", "The band the S-parameters come out over. One broadband pulse covers it "
                                            "in a single run per excited port.")
        self.start = UnitSpinBox("GHz", scale=1e9, step=0.1, decimals=4, minimum=0)
        self.start.committed.connect(setting("frequency.start", "start frequency"))
        self.stop = UnitSpinBox("GHz", scale=1e9, step=0.1, decimals=4, minimum=0)
        self.stop.committed.connect(setting("frequency.stop", "stop frequency"))
        form = form_layout()
        form.addRow("Start", self.start)
        form.addRow("Stop", self.stop)
        layout.addLayout(form)
        self.cards["frequency"] = frame

        # ----------------------------------------------------- convergence
        frame, layout = card(
            "Convergence", "The run stops at whichever comes first. ΔS is the largest change in any S-parameter over "
                           "the band between checks one excitation pulse apart; it must hold for two checks in a row, "
                           "and is not tested until the pulse has crossed the board.")
        self.delta_s_on = QCheckBox("Stop when ΔS is below")
        self.delta_s_on.clicked.connect(
            lambda on: setting("convergence.delta_s")(self._last_delta_s if on else 0.0))
        self._last_delta_s = 0.02
        self.delta_s = UnitSpinBox(step=0.005, decimals=4, minimum=0.0001, maximum=1.0, tooltip=(
            "0.1–0.2: quick look. 0.02: good default. 0.01–0.005: accurate, runs longer."))
        self.delta_s.committed.connect(setting("convergence.delta_s"))
        self.delta_presets = Segmented([(v, label) for v, label in DELTA_S_PRESETS])
        self.delta_presets.committed.connect(lambda v: setting("convergence.delta_s")(float(v)))
        self.end_db = UnitSpinBox("dB", step=5, decimals=1, minimum=-200, maximum=-1, tooltip=(
            "openEMS energy decay, 10·log10(E / E_peak), as its log prints it. -50 dB is openEMS's default."))
        self.end_db.committed.connect(setting("convergence.end_criteria_db", "energy decay"))
        self.max_steps = IntSpinBox(minimum=100, maximum=100_000_000, step=10000)
        self.max_steps.committed.connect(setting("max_steps", "max timesteps"))
        form = form_layout()
        form.addRow(self.delta_s_on, self.delta_s)
        form.addRow("", self.delta_presets)
        form.addRow("Energy decay", self.end_db)
        form.addRow("Max timesteps", self.max_steps)
        layout.addLayout(form)
        self.cards["convergence"] = frame

        # ------------------------------------------------ mesh & materials
        frame, layout = card("Mesh & materials", "The copper is meshed from a raster of the gerbers; the dielectric's "
                                                 "loss is set here for the whole board. Each via's drill, plating and "
                                                 "fill are its padstack's: Padstacks, under Board in the project tree.")
        self.pixel = UnitSpinBox("µm", step=0.5, decimals=2, minimum=0.5, tooltip=(
            "Raster resolution the copper is meshed from — smaller resolves finer traces but costs memory and time"))
        self.pixel.committed.connect(setting("pixel_size_um", "pixel size"))
        self.dielectric_loss = ChoiceBox(LOSS_MODELS)
        self.dielectric_loss.setToolTip(LOSS_TOOLTIP)
        self.dielectric_loss.committed.connect(setting("dielectric_loss", "dielectric loss"))
        form = form_layout()
        form.addRow("Pixel size", self.pixel)
        form.addRow("Dielectric loss", self.dielectric_loss)
        layout.addLayout(form)
        self.cards["mesh"] = frame

        # ---------------------------------------------------- field export
        frame, layout = card("Field export", "Electric-field planes the run records across the whole board: just "
                                             "above and below it (outer), on the outer or inner copper layers, or "
                                             "through the middle of each dielectric (substrate). None ticked "
                                             "records none.")
        self.exports = {}
        boxes = []
        for key in EXPORT_FIELDS:
            box = QCheckBox(key)
            box.clicked.connect(lambda _on, key=key: self._toggle_export(key))
            self.exports[key] = box
            boxes.append(box)
        layout.addWidget(row(*boxes))
        self.cards["export"] = frame

        # -------------------------------------------------- nets under test
        frame, layout = card("Nets under test", "Single-ended traces, by port index: gerber2ems post-processes "
                                                "their delay. Differential pairs are under Ports in the project "
                                                "tree, with their own mode impedances.")
        self.nets_layout = QVBoxLayout()
        self.nets_layout.setSpacing(6)
        layout.addLayout(self.nets_layout)
        add_trace = QPushButton("+ Trace")
        add_trace.clicked.connect(lambda: guarded(lambda: s.push(cmd.AddTrace(s))))
        layout.addWidget(row(add_trace))
        self.cards["nets"] = frame
        self._net_signature = None

        self.setWidget(centred_column([self.cards[key] for key in SECTIONS]))
        session.projectChanged.connect(lambda: QTimer.singleShot(0, self.refresh))
        self.refresh()

    # --------------------------------------------------------------- public

    def scroll_to(self, section: str) -> None:
        frame = self.cards.get(section)
        if frame is None:
            self.verticalScrollBar().setValue(0)
            return
        # The layout may not have run yet after a page switch.
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(max(frame.y() - 8, 0)))

    def _toggle_export(self, key: str) -> None:
        current = list(self.session.project.export_field)
        updated = [k for k in current if k != key] if key in current else current + [key]
        guarded(lambda: self.session.push(cmd.SetBoardSetting(self.session, "export_field", updated, "field export")))

    def refresh(self) -> None:
        p = self.session.project
        self.start.set_stored(p.frequency.start)
        self.stop.set_stored(p.frequency.stop)
        self.max_steps.set_stored(p.max_steps)
        delta_s = p.convergence.delta_s
        if delta_s > 0:
            self._last_delta_s = delta_s
        self.delta_s_on.setChecked(delta_s > 0)
        self.delta_s.setEnabled(delta_s > 0)
        self.delta_presets.setEnabled(delta_s > 0)
        self.delta_s.set_stored(delta_s if delta_s > 0 else self._last_delta_s)
        self.delta_presets.set_value(delta_s)
        self.end_db.set_stored(p.convergence.end_criteria_db)
        self.pixel.set_stored(p.pixel_size_um)
        self.dielectric_loss.set_value(p.dielectric_loss)
        for key, box in self.exports.items():
            box.setChecked(key in p.export_field)
        signature = len(p.traces)
        if signature != self._net_signature:
            self._net_signature = signature
            self._build_nets()
        for i, trace in enumerate(p.traces):
            self._trace_fields[i]["start"].set_stored(trace.start)
            self._trace_fields[i]["stop"].set_stored(trace.stop)

    def _build_nets(self) -> None:
        _clear(self.nets_layout)
        p = self.session.project
        s = self.session
        self._trace_fields = []
        for i, _trace in enumerate(p.traces):
            fields = {}
            host = QWidget()
            h = QHBoxLayout(host)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(QLabel("Trace"))
            for field in ("start", "stop"):
                box = IntSpinBox(minimum=0, maximum=999)
                box.committed.connect(lambda v, i=i, field=field: guarded(
                    lambda: s.push(cmd.EditNet(s, "traces", i, field, int(v)))))
                fields[field] = box
                h.addWidget(box)
                if field == "start":
                    h.addWidget(QLabel("→"))
            h.addStretch(1)
            remove = QPushButton("×")
            remove.setProperty("role", "flat")
            remove.setToolTip("Remove this trace")
            remove.clicked.connect(lambda _=False, i=i: guarded(lambda: s.push(cmd.EditNet(s, "traces", i, None))))
            h.addWidget(remove)
            self.nets_layout.addWidget(host)
            self._trace_fields.append(fields)
