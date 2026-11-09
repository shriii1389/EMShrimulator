"""The solution setup page: one scrolling page of cards; `scroll_to` jumps to one."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.dielectric_loss import LOSS_MODELS, LOSS_TOOLTIP
from EMShrimulator.app.views.modeler import units as unit_math
from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit, ExpressionVector
from EMShrimulator.app.widgets.form import ChoiceBox, centred_column, fmt_count, fmt_eng, form_layout, hint, row
from EMShrimulator.app.widgets.form import card as _card
from EMShrimulator.app.widgets.unit_spinbox import IntSpinBox, UnitSpinBox
from EMShrimulator.core.em import region
from EMShrimulator.core.em.model import BOUNDARY_FACES, ExcitationKind
from EMShrimulator.core.em.params import FREQUENCY_UNITS, LENGTH_UNITS

BOUNDARY_OPTIONS = [("PML_8", "PML (8 cells) — absorbing"), ("MUR", "MUR — absorbing, cheap"),
                    ("PEC", "PEC — electric wall"), ("PMC", "PMC — magnetic wall")]
EXCITATIONS = [(ExcitationKind.GAUSSIAN.value, "Gaussian pulse — broadband"),
               (ExcitationKind.SINUSOIDAL.value, "Sinusoid — one frequency"),
               (ExcitationKind.STEP.value, "Step"), (ExcitationKind.DIRAC.value, "Dirac impulse")]
SOLUTION_TYPES = [("network", "Network — S-parameters"),
                  ("antenna", "Antenna — S-parameters and far-field pattern")]
# Accuracy presets: (cells per wavelength, energy decay in dB).
PRESETS = {"draft": ("Draft", 12.0, -30.0), "standard": ("Standard", 20.0, -50.0),
           "fine": ("Fine", 30.0, -60.0), "very_fine": ("Very fine", 40.0, -70.0)}
MAX_WIDTH = 860


def set_open_region(session, on: bool) -> None:
    """Open the region (every face absorbing, air sized automatically) or close it, keeping its box."""
    if on:
        values = {"boundary.open_region": True, **{f"boundary.{face}": region.ABSORBING for face in BOUNDARY_FACES}}
    else:
        margins = tuple(float(v) for v in session.resolution.project.mesh.air_margin)
        values = {"boundary.open_region": False, "mesh.air_margin": margins}
    guarded(lambda: session.push(cmd.SetDesignSettings(session, values, "Open region" if on else
                                                       "Close open region")))


def open_region_text(resolved, units) -> str:
    """What the open region made of the design, in the display units."""
    info = region.summary(resolved)
    if info["frequency"] <= 0:
        return "Set a frequency sweep: the air is sized at a quarter wavelength."
    l_unit, per_mm = unit_math.label("length", units), unit_math.factor("length", units)
    f_unit, per_hz = unit_math.label("frequency", units), unit_math.factor("frequency", units)
    text = (f"Every face absorbs (PML), with {info['quarter_mm'] / per_mm:.4g} {l_unit} of air on every side: λ/4 "
            f"at {info['frequency'] / per_hz:.4g} {f_unit}, {info['why']}. It follows the frequencies as they "
            "change.")
    kept = info["kept"]
    if kept:
        axes = ", ".join(f"{axis.upper()} keeps its {mm / per_mm:.4g} {l_unit}" for axis, mm in kept.items())
        text += (f" {axes} of the air margin: a guided port launches along "
                 f"{'it' if len(kept) == 1 else 'them'} and must run into the boundary.")
    return text


class FrequencyList(QWidget):
    """A short list of frequencies, each an expression; commits the whole list."""

    committed = Signal(list)

    def __init__(self, session, path_base: str, parent=None):
        super().__init__(parent)
        self.session = session
        self.path_base = path_base
        self._values: List = []
        self._rows: List[ExpressionEdit] = []
        self.rows = QVBoxLayout()
        self.rows.setContentsMargins(0, 0, 0, 0)
        self.rows.setSpacing(3)
        self.add = QPushButton("+ Frequency")
        self.add.clicked.connect(self._add)
        self.empty = QLabel("None yet: the centre of the sweep is used.")
        self.empty.setProperty("role", "note")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self.empty)
        layout.addLayout(self.rows)
        layout.addWidget(row(self.add))

    def set_value(self, values) -> None:
        self._values = list(values or [])
        if len(self._values) != len(self._rows):
            while self.rows.count():
                item = self.rows.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self._rows = []
            for i in range(len(self._values)):
                edit = ExpressionEdit(self.session, "frequency", step=0.1, path=lambda i=i: f"{self.path_base}.{i}")
                edit.committed.connect(lambda v, i=i: self._set(i, v))
                remove = QPushButton("×")
                remove.setProperty("role", "flat")
                remove.setFixedWidth(26)
                remove.setToolTip("Remove this frequency")
                remove.clicked.connect(lambda _=False, i=i: self.committed.emit(
                    [v for j, v in enumerate(self._values) if j != i]))
                host = QWidget()
                h = QHBoxLayout(host)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(3)
                h.addWidget(edit, 1)
                h.addWidget(remove)
                self.rows.addWidget(host)
                self._rows.append(edit)
        for edit, value in zip(self._rows, self._values):
            edit.set_value(value)
        self.empty.setVisible(not self._values)

    def _set(self, i: int, value) -> None:
        values = list(self._values)
        values[i] = value
        self.committed.emit(values)

    def _add(self) -> None:
        resolved = self.session.resolution.project.frequency
        start, stop = float(resolved.start), float(resolved.stop)
        if self._values:
            try:
                last = float(self.session.resolution.project.nf2ff.frequencies[-1])
            except (IndexError, TypeError, ValueError):
                last = (start + stop) / 2
            new = min(last + (stop - start) / 10, stop)
        else:
            new = (start + stop) / 2
        self.committed.emit(self._values + [float(f"{new:.6g}")])


class SolveSettings(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.mesh_preview: Optional[dict] = None
        self.mesh_error: Optional[str] = None
        self.setWidgetResizable(True)
        self.setObjectName("setup-page")
        s = session

        def setting(path: str, label: str = ""):
            return lambda value: guarded(lambda: s.push(cmd.SetDesignSetting(s, path, value, label)))

        self.cards: Dict[str, QFrame] = {}

        # -------------------------------------------------------- solution
        card, layout = _card("Solution", "What the solver computes. Every excited port is one run of the "
                                         "time-domain solver; a pulse covers the whole sweep at once.")
        self.solution_type = ChoiceBox(SOLUTION_TYPES)
        self.solution_type.committed.connect(lambda v: setting("nf2ff.enabled", "solution type")(v == "antenna"))
        self.excitation = ChoiceBox(EXCITATIONS)
        self.excitation.committed.connect(lambda v: setting("simulation.excitation.kind", "excitation")(ExcitationKind(v)))
        self.cw = ExpressionEdit(s, "frequency", path=lambda: "simulation.excitation.frequency")
        self.cw.committed.connect(setting("simulation.excitation.frequency", "CW frequency"))
        self.cw_label = QLabel()
        self.length_unit = ChoiceBox([(u, u) for u in LENGTH_UNITS])
        self.length_unit.committed.connect(setting("display_units.length", "length unit"))
        self.frequency_unit = ChoiceBox([(u, u) for u in FREQUENCY_UNITS])
        self.frequency_unit.committed.connect(setting("display_units.frequency", "frequency unit"))
        self.solution_form = form_layout()
        self.solution_form.addRow("Solution type", self.solution_type)
        self.solution_form.addRow("Excitation", self.excitation)
        self.solution_form.addRow(self.cw_label, self.cw)
        self.solution_form.addRow("Units", row(self.length_unit, self.frequency_unit))
        layout.addLayout(self.solution_form)
        layout.addWidget(hint("Units are for display only: stored values stay in millimetres and hertz."))
        self.cards["solution"] = card

        # ----------------------------------------------------- frequencies
        card, layout = _card("Frequencies", "The sweep is the band the S-parameters come out over. Operating "
                                            "frequencies are where the far-field pattern, gain and efficiency "
                                            "are worked out - add as many as you need.")
        self.f_start = ExpressionEdit(s, "frequency", step=0.1, path=lambda: "frequency.start")
        self.f_start.committed.connect(setting("frequency.start", "start frequency"))
        self.f_stop = ExpressionEdit(s, "frequency", step=0.1, path=lambda: "frequency.stop")
        self.f_stop.committed.connect(setting("frequency.stop", "stop frequency"))
        self.f_points = IntSpinBox(minimum=2, maximum=100000, step=100)
        self.f_points.committed.connect(setting("frequency.points", "frequency points"))
        self.step_note = QLabel()
        self.step_note.setProperty("role", "note")
        self.operating = FrequencyList(s, "nf2ff.frequencies")
        self.operating.committed.connect(setting("nf2ff.frequencies", "operating frequencies"))
        self.operating_note = hint()
        self.freq_labels = [QLabel(), QLabel(), QLabel()]
        form = form_layout()
        form.addRow(self.freq_labels[0], self.f_start)
        form.addRow(self.freq_labels[1], self.f_stop)
        form.addRow("Points", row(self.f_points, self.step_note))
        form.addRow(self.freq_labels[2], self.operating)
        layout.addLayout(form)
        layout.addWidget(self.operating_note)
        self.cards["frequency"] = card

        # -------------------------------------------------------- accuracy
        card, layout = _card(
            "Accuracy", "FDTD has no adaptive mesh of its own, so accuracy comes from three things: how fine the "
                        "mesh is, how long the fields are left to ring down, and - like ΔS convergence - solving "
                        "again on a finer mesh until the S-parameters stop moving.")
        self.preset = ChoiceBox()
        self.preset.committed.connect(self._apply_preset)
        self.cells_per_wl = UnitSpinBox(minimum=4, maximum=200, step=5, decimals=2,
                                        tooltip="Resolution at the stop frequency, in the highest-εr material")
        self.cells_per_wl.committed.connect(setting("mesh.cells_per_wavelength", "cells per wavelength"))
        # An energy ratio in dB, as openEMS logs it.
        self.end_criteria = UnitSpinBox("dB", minimum=-200, maximum=0, step=5, decimals=1,
                                        tooltip="Stop once the residual energy has decayed this far below its peak")
        self.end_criteria.committed.connect(lambda db: setting("simulation.end_criteria", "end criteria")(10 ** (db / 10)))
        self.passes = IntSpinBox(minimum=1, maximum=20, tooltip="1 solves once on the mesh as set")
        self.passes.committed.connect(setting("simulation.max_passes", "adaptive passes"))
        self.delta_s = UnitSpinBox(minimum=0.001, maximum=0.5, step=0.005, decimals=3,
                                   tooltip="Stop once no S-parameter changes by more than this between passes")
        self.delta_s.committed.connect(setting("simulation.delta_s", "ΔS target"))
        self.refine = UnitSpinBox("×", minimum=1.05, maximum=2.0, step=0.05, decimals=2,
                                  tooltip="How much finer each pass meshes than the one before")
        self.refine.committed.connect(setting("simulation.refine_ratio", "refinement per pass"))
        self.accuracy_form = form_layout()
        self.accuracy_form.addRow("Preset", self.preset)
        self.accuracy_form.addRow("Cells / λ", self.cells_per_wl)
        self.accuracy_form.addRow("Energy decay", self.end_criteria)
        self.accuracy_form.addRow("Adaptive passes", self.passes)
        self.accuracy_form.addRow("Max ΔS", self.delta_s)
        self.accuracy_form.addRow("Refine per pass", self.refine)
        layout.addLayout(self.accuracy_form)
        self.passes_note = hint()
        layout.addWidget(self.passes_note)

        mesh_heading = QLabel("Mesh")
        mesh_heading.setProperty("role", "subheading")
        layout.addWidget(mesh_heading)
        self.max_cell = ExpressionEdit(s, step=0.1, path=lambda: "mesh.max_cell_size",
                                       tooltip="0 = derive from cells per wavelength")
        self.max_cell.committed.connect(lambda v: setting("mesh.max_cell_size", "max cell")(
            None if (isinstance(v, float) and v == 0) else v))
        self.ratio = UnitSpinBox(minimum=1.05, maximum=3, step=0.05, decimals=3,
                                 tooltip="Largest allowed size ratio between neighbouring cells")
        self.ratio.committed.connect(setting("mesh.smooth_ratio", "grading ratio"))
        self.thin = IntSpinBox(minimum=1, maximum=50, tooltip=(
            "A substrate thinner than one cell gets at least this many — one cell cannot resolve a microstrip mode"))
        self.thin.committed.connect(setting("mesh.min_cells_per_layer", "cells per thin layer"))
        self.margin = ExpressionVector(s, step=1, path=lambda: "mesh.air_margin")
        self.margin.committed.connect(setting("mesh.air_margin", "air margin"))
        self.edges = QCheckBox("Metal edge refinement")
        self.edges.setToolTip("openEMS's 1/3–2/3 rule at conductor edges, where the current density is singular")
        self.edges.clicked.connect(lambda on: setting("mesh.metal_edge_refinement", "metal edge refinement")(bool(on)))
        self.mesh_labels = [QLabel(), QLabel()]
        mesh_form = form_layout()
        mesh_form.addRow(self.mesh_labels[0], self.max_cell)
        mesh_form.addRow("Grading ratio", self.ratio)
        mesh_form.addRow("Cells / thin layer", self.thin)
        mesh_form.addRow(self.mesh_labels[1], self.margin)
        mesh_form.addRow("", self.edges)
        layout.addLayout(mesh_form)
        readout = QWidget()
        grid = QGridLayout(readout)
        grid.setContentsMargins(0, 6, 0, 0)
        self.readout = {}
        for i, key in enumerate(("Cells", "Base cell", "Timestep", "Grid")):
            caption = QLabel(key)
            caption.setProperty("role", "note")
            value = QLabel("—")
            value.setProperty("role", "value")
            grid.addWidget(caption, 0, i)
            grid.addWidget(value, 1, i)
            self.readout[key] = value
        layout.addWidget(readout)
        self.mesh_error_label = QLabel()
        self.mesh_error_label.setProperty("role", "error")
        self.mesh_error_label.setWordWrap(True)
        layout.addWidget(self.mesh_error_label)
        self.estimate = hint()
        layout.addWidget(self.estimate)
        self.cards["accuracy"] = card

        # ------------------------------------------------------ boundaries
        card, layout = _card("Boundaries", "What each face of the simulation box does to a wave that reaches it.")
        self.open_region = QCheckBox("Open region: size the air box for me")
        self.open_region.setToolTip(
            "Like HFSS's Create Open Region: every face absorbs, and the air between the design and the boundary "
            "is the least that keeps the boundary out of the near field: a quarter wavelength at the lowest "
            "operating frequency of an antenna, else at the sweep's lowest frequency. It is kept up to date as the "
            "design and the frequencies change.")
        self.open_region.clicked.connect(self._open_region_clicked)
        self.open_note = hint()
        layout.addWidget(self.open_region)
        layout.addWidget(self.open_note)
        self.boundaries = {}
        bnd_grid = QGridLayout()
        bnd_grid.setHorizontalSpacing(8)
        bnd_grid.setVerticalSpacing(5)
        for i, face in enumerate(BOUNDARY_FACES):
            box = ChoiceBox(BOUNDARY_OPTIONS)
            box.committed.connect(setting(f"boundary.{face}", f"{face} boundary"))
            self.boundaries[face] = box
            label = QLabel(face)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            bnd_grid.addWidget(label, i // 2, (i % 2) * 2)
            bnd_grid.addWidget(box, i // 2, (i % 2) * 2 + 1)
        bnd_grid.setColumnStretch(1, 1)
        bnd_grid.setColumnStretch(3, 1)
        layout.addLayout(bnd_grid)
        self.boundary_buttons = []
        for value, text in (("PML_8", "All PML"), ("MUR", "All MUR"), ("PEC", "All PEC")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, value=value: guarded(lambda: s.push(cmd.SetDesignSettings(
                s, {f"boundary.{face}": value for face in BOUNDARY_FACES}, f"All boundaries {value}"))))
            self.boundary_buttons.append(b)
        layout.addWidget(row(*self.boundary_buttons))
        self.cards["boundaries"] = card

        # ---------------------------------------------------------- solver
        card, layout = _card("Solver limits")
        self.max_steps = IntSpinBox(minimum=100, maximum=100_000_000, step=5000)
        self.max_steps.committed.connect(setting("simulation.max_timesteps", "max timesteps"))
        self.dielectric_loss = ChoiceBox(LOSS_MODELS)
        self.dielectric_loss.setToolTip(LOSS_TOOLTIP)
        self.dielectric_loss.committed.connect(setting("simulation.dielectric_loss", "dielectric loss"))
        solver_form = form_layout()
        solver_form.addRow("Max timesteps", self.max_steps)
        solver_form.addRow("Dielectric loss", self.dielectric_loss)
        layout.addLayout(solver_form)
        self.cards["solver"] = card

        # ------------------------------------------------------- far field
        card, layout = _card("Far field", "The pattern is integrated over a Huygens box just inside the "
                                          "boundaries, at each operating frequency.")
        self.theta = UnitSpinBox("°", minimum=0.5, maximum=45, step=1, decimals=2)
        self.theta.committed.connect(setting("nf2ff.theta_step_deg", "θ step"))
        self.phi = UnitSpinBox("°", minimum=0.5, maximum=45, step=1, decimals=2)
        self.phi.committed.connect(setting("nf2ff.phi_step_deg", "φ step"))
        self.clearance = IntSpinBox(minimum=0, maximum=100, suffix="cells", tooltip=(
            "How far inside the boundary the Huygens surface sits; 0 lets openEMS size it"))
        self.clearance.committed.connect(setting("nf2ff.margin_cells", "far-field clearance"))
        ff_form = form_layout()
        ff_form.addRow("θ step", self.theta)
        ff_form.addRow("φ step", self.phi)
        ff_form.addRow("Clearance", self.clearance)
        layout.addLayout(ff_form)
        self.cards["farfield"] = card

        self.setWidget(centred_column([self.cards[key] for key in ("solution", "frequency", "accuracy", "boundaries",
                                                                    "solver", "farfield")], MAX_WIDTH))

        session.projectChanged.connect(lambda: QTimer.singleShot(0, self.refresh))
        self.refresh()

    # --------------------------------------------------------------- public

    def scroll_to(self, section: str) -> None:
        card = self.cards.get(section)
        if card is None:
            self.verticalScrollBar().setValue(0)
            return
        # The layout may not have run yet after a page switch.
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(max(card.y() - 8, 0)))

    def show_mesh_preview(self, mesh: Optional[dict], error: Optional[str]) -> None:
        self.mesh_preview = mesh
        self.mesh_error = error
        self._refresh_readout()

    # -------------------------------------------------------------- refresh

    def _open_region_clicked(self, on: bool) -> None:
        set_open_region(self.session, bool(on))

    def _apply_preset(self, key) -> None:
        if key not in PRESETS:
            return
        _label, cells, db = PRESETS[key]
        s = self.session
        guarded(lambda: s.push(cmd.SetDesignSettings(
            s, {"mesh.cells_per_wavelength": cells, "simulation.end_criteria": 10 ** (db / 10)},
            f"Accuracy preset {PRESETS[key][0]}")))

    def refresh(self) -> None:
        p = self.session.project
        units = p.display_units
        f_unit = unit_math.label("frequency", units)
        l_unit = unit_math.label("length", units)

        self.solution_type.set_value("antenna" if p.nf2ff.enabled else "network")
        self.excitation.set_value(p.simulation.excitation.kind.value)
        self.cw_label.setText(f"CW frequency ({f_unit})")
        self.cw.set_value(p.simulation.excitation.frequency if p.simulation.excitation.frequency is not None
                          else p.frequency.stop)
        self.solution_form.setRowVisible(self.cw, p.simulation.excitation.kind == ExcitationKind.SINUSOIDAL)
        self.length_unit.set_value(units.length)
        self.frequency_unit.set_value(units.frequency)

        self.freq_labels[0].setText(f"Start ({f_unit})")
        self.freq_labels[1].setText(f"Stop ({f_unit})")
        self.freq_labels[2].setText(f"Operating ({f_unit})")
        self.f_start.set_value(p.frequency.start)
        self.f_stop.set_value(p.frequency.stop)
        self.f_points.set_stored(p.frequency.points)
        resolved = self.session.resolution.project
        try:
            step = (float(resolved.frequency.stop) - float(resolved.frequency.start)) / max(p.frequency.points - 1, 1)
            self.step_note.setText(f"every {fmt_eng(step, 'Hz')}")
        except (TypeError, ValueError):
            self.step_note.setText("")
        self.operating.set_value(p.nf2ff.frequencies)
        self.operating_note.setText(
            "Patterns are computed at each of these after the run." if p.nf2ff.enabled else
            "Kept for the far field: switch the solution type to Antenna to compute patterns here.")

        cells = p.mesh.cells_per_wavelength
        decay = 10 * math.log10(max(p.simulation.end_criteria, 1e-30))
        preset = next((k for k, (_l, c, d) in PRESETS.items() if abs(c - cells) < 1e-6 and abs(d - decay) < 0.05),
                      None)
        options = [(k, f"{label} — {c:g} cells/λ, {d:g} dB") for k, (label, c, d) in PRESETS.items()]
        self.preset.set_options(options + ([] if preset else [("custom", "Custom")]))
        self.preset.set_value(preset or "custom")
        self.cells_per_wl.set_stored(cells)
        self.end_criteria.set_stored(decay)
        sim = p.simulation
        self.passes.set_stored(sim.max_passes)
        self.delta_s.set_stored(sim.delta_s)
        self.refine.set_stored(sim.refine_ratio)
        adaptive = sim.max_passes > 1
        for widget in (self.delta_s, self.refine):
            self.accuracy_form.setRowVisible(widget, adaptive)
        if adaptive:
            growth = sim.refine_ratio ** 4  # cells grow as r³, and the timestep shrinks as 1/r
            self.passes_note.setText(
                f"Up to {sim.max_passes} solves. Each pass meshes {sim.refine_ratio:g}× finer than the last and costs "
                f"roughly {growth:.1f}× as long; it stops early once the largest change in any S-parameter, at any "
                f"frequency, is below {sim.delta_s:g}.")
        else:
            self.passes_note.setText("One solve on the mesh as set. Raise the passes to keep refining until the "
                                     "S-parameters stop moving by more than ΔS.")

        self.mesh_labels[0].setText(f"Max cell ({l_unit})")
        self.mesh_labels[1].setText(f"Air margin ({l_unit})")
        self.max_cell.set_value(p.mesh.max_cell_size if p.mesh.max_cell_size is not None else 0.0)
        self.ratio.set_stored(p.mesh.smooth_ratio)
        self.thin.set_stored(p.mesh.min_cells_per_layer)
        opened = p.boundary.open_region
        # When open, the margin and faces are set by the region and read-only.
        self.margin.set_value(resolved.mesh.air_margin if opened else p.mesh.air_margin)
        self.margin.setEnabled(not opened)
        self.margin.setToolTip("Set by the open region (Boundaries)" if opened else
                               "Free space between the design and each pair of faces of the box")
        self.edges.setChecked(p.mesh.metal_edge_refinement)

        self.open_region.setChecked(opened)
        self.open_note.setText(open_region_text(resolved, units) if opened else
                               "Off: the faces and the air margin are as you set them.")
        source = resolved.boundary if opened else p.boundary
        for face, box in self.boundaries.items():
            box.set_value(getattr(source, face))
            box.setEnabled(not opened)
        for button in self.boundary_buttons:
            button.setEnabled(not opened)
        self.max_steps.set_stored(p.simulation.max_timesteps)
        self.dielectric_loss.set_value(p.simulation.dielectric_loss)

        self.theta.set_stored(p.nf2ff.theta_step_deg)
        self.phi.set_stored(p.nf2ff.phi_step_deg)
        self.clearance.set_stored(p.nf2ff.margin_cells)
        self.cards["farfield"].setVisible(p.nf2ff.enabled)
        self._refresh_readout()

    def _refresh_readout(self) -> None:
        mesh = self.mesh_preview
        if mesh:
            counts = mesh["line_counts"]
            self.readout["Cells"].setText(fmt_count(mesh["cells"]))
            self.readout["Base cell"].setText(f"{mesh['resolution_mm']:.3f} mm")
            self.readout["Timestep"].setText(fmt_eng(mesh.get("timestep_s"), "s"))
            self.readout["Grid"].setText(f"{counts['x']}×{counts['y']}×{counts['z']}")
            if mesh.get("estimated_seconds") is not None:
                self.estimate.setText(
                    f"At {fmt_count(mesh['cells'])} cells and Δt {fmt_eng(mesh.get('timestep_s'), 's')}, the full "
                    f"{fmt_count(self.session.project.simulation.max_timesteps)} timesteps simulate "
                    f"{fmt_eng(mesh['estimated_seconds'], 's')} of real time. Runs usually stop earlier, when the "
                    "energy criterion is met.")
        else:
            for value in self.readout.values():
                value.setText("—")
            self.estimate.setText("")
        self.mesh_error_label.setText(self.mesh_error or "")
        self.mesh_error_label.setVisible(bool(self.mesh_error))
