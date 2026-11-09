"""The port panel: one modeler port, in sections, with the actions a port needs at hand."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QLineEdit, QPushButton, QWidget

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.views.modeler import units as unit_math
from EMShrimulator.app.views.modeler.editor_base import NameEdit, PropertyEditor
from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit, ExpressionVector
from EMShrimulator.app.views.modeler.point_size_edit import PointSizeEdit
from EMShrimulator.app.views.modeler.primitive_editor import AXES, PLANE_AXES, coax_hint_text
from EMShrimulator.app.widgets.form import ChoiceBox, fmt_eng, form_layout, hint, row
from EMShrimulator.app.widgets.unit_spinbox import IntSpinBox
from EMShrimulator.core.em import edits, sheets
from EMShrimulator.core.em.model import Port, PortKind, PrimitiveKind, SheetShape


def _subheading(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "subheading")
    return label


LINE_HINTS = {
    PortKind.LUMPED: "From the − terminal to the + one: clearance edge to pin, ground to trace. Across a "
                     "rectangle from one edge to the opposite one, the whole sheet is the port; any other "
                     "line is a source along the line itself, as openEMS models a probe feed. openEMS "
                     "excites along X, Y or Z only, so draw it straight along one.",
    PortKind.MICROSTRIP: "From the ground plane up to the strip, across the substrate. It sets which way "
                         "the port is excited; the port's metal goes at its end.",
}


class PortEditor(PropertyEditor):
    plotRequested = Signal(int)          # this port's number
    focusRequested = Signal(str, str)    # ("port", id)
    drawLineRequested = Signal(str)      # port id: draw its integration line in the 3D view

    def __init__(self, session, parent=None):
        super().__init__(session, "Port", parent)
        self.port_id: Optional[str] = None
        self.index = -1
        s = session

        # ---------------------------------------------------------- actions
        self.summary = hint()
        self.excite_only = QPushButton("Excite only this")
        self.excite_only.setToolTip("Drive this port and switch every other one to measured only")
        self.excite_only.clicked.connect(lambda: self.push(lambda: cmd.ExciteOnly(self.session, self.port_id)))
        self.plot = QPushButton("Plot S")
        self.plot.clicked.connect(lambda: self.plotRequested.emit(self._number))
        self.zoom = QPushButton("Zoom to")
        self.zoom.setToolTip("Turn the camera to this port")
        self.zoom.clicked.connect(lambda: self.focusRequested.emit("port", self.port_id or ""))
        self.duplicate = QPushButton("Duplicate")
        self.duplicate.clicked.connect(self._duplicate)
        self.delete = QPushButton("Delete")
        self.delete.setProperty("role", "danger")
        self.delete.clicked.connect(self._delete)
        self.outer.addWidget(self.summary)
        self.outer.addWidget(row(self.excite_only, self.plot, self.zoom))
        self.outer.addWidget(row(self.duplicate, self.delete))
        self._number = 1

        # ---------------------------------------------------------- general
        general = form_layout()
        self.outer.addWidget(_subheading("General"))
        self.outer.addLayout(general)
        self.name = NameEdit(lambda v: self.set_field("name", v))
        self.number = IntSpinBox(minimum=1, maximum=999)
        self.number.committed.connect(lambda v: self.set_field("number", int(v)))
        self.kind = ChoiceBox([(k.value, label) for k, label in edits.PORT_LABELS.items()])
        self.kind.committed.connect(lambda v: self.set_field("kind", PortKind(v), "port type"))
        self.impedance = ExpressionEdit(s, "plain", step=1, path=lambda: self._sub("impedance"))
        self.impedance.committed.connect(lambda v: self.set_field("impedance", v, "reference impedance"))
        self.excite = QCheckBox("Excite this port")
        self.excite.setToolTip("Each excited port becomes its own FDTD run and fills one column of the S-matrix")
        self.excite.clicked.connect(lambda on: self.set_field("excite", bool(on), "excitation"))
        general.addRow("Name", self.name)
        general.addRow("Number", self.number)
        general.addRow("Type", self.kind)
        general.addRow("Z₀ (Ω)", self.impedance)
        general.addRow("", self.excite)

        # --------------------------------------------------------- geometry
        geometry = form_layout()
        self.outer.addWidget(_subheading("Geometry"))
        self.outer.addLayout(geometry)
        self.geometry = ChoiceBox([("box", "Start point and size"), ("sheet", "A sheet")])
        self.geometry.committed.connect(self._set_geometry_mode)
        self.sheet = ChoiceBox()
        self.sheet.committed.connect(lambda v: self.set_field("sheet", v, "sheet"))
        self.sheet_hint = hint()
        self.needs_hint = hint()
        self.extrude = ExpressionEdit(s, step=0.5, path=lambda: self._sub("sheet_extrude_mm"),
                                      tooltip="How far the port reaches along the sheet normal; 0 = a few cells")
        self.extrude.committed.connect(lambda v: self.set_field("sheet_extrude_mm", v, "extrusion"))
        self.extrude_hint = hint("This port launches a guided mode, and openEMS cannot excite one at zero length, "
                                 "so the sheet is extruded a few cells along its normal to give the mode a "
                                 "section of guide.")
        geometry.addRow("Defined by", self.geometry)
        geometry.addRow("Sheet", self.sheet)
        geometry.addRow(self.sheet_hint)
        geometry.addRow(self.needs_hint)
        self.extrude_label = QLabel()
        geometry.addRow(self.extrude_label, self.extrude)
        geometry.addRow(self.extrude_hint)
        self.box = PointSizeEdit(s, geometry, path=self.path)
        self.box.committed.connect(lambda p1, p2: self.set_fields({"p1": p1, "p2": p2}, "box"))
        self.box.modeChanged.connect(lambda mode: self.set_field("size_mode", mode, "dimension mode"))

        # ------------------------------------------------- integration line
        line = form_layout()
        self.line_heading = _subheading("Integration line")
        self.outer.addWidget(self.line_heading)
        self.outer.addLayout(line)
        self.line_state = hint()
        self.line_start = ExpressionVector(s, path=lambda: self._sub("line_start"))
        self.line_start.committed.connect(lambda v: self.set_field("line_start", v, "line start"))
        self.line_end = ExpressionVector(s, path=lambda: self._sub("line_end"))
        self.line_end.committed.connect(lambda v: self.set_field("line_end", v, "line end"))
        self.start_label, self.end_label = QLabel(), QLabel()
        self.draw_line = QPushButton("Draw in 3D view")
        self.draw_line.setToolTip("Click the line's start on the sheet, then its end. Ends snap to the sheet's "
                                  "centre, corners and edges; the end keeps to X, Y or Z from the start.")
        self.draw_line.clicked.connect(lambda: self.drawLineRequested.emit(self.port_id or ""))
        self.reverse_line = QPushButton("Reverse")
        self.reverse_line.setToolTip("Swap the line's ends")
        self.reverse_line.clicked.connect(self._reverse_line)
        self.clear_line = QPushButton("Clear")
        self.clear_line.clicked.connect(lambda: self._push_line(None, None))
        self.line_hint = hint()
        line.addRow(self.line_state)
        line.addRow(self.start_label, self.line_start)
        line.addRow(self.end_label, self.line_end)
        line.addRow("", row(self.draw_line, self.reverse_line, self.clear_line))
        line.addRow(self.line_hint)
        self._line_form = line

        # ------------------------------------------------------ line / mode
        mode = form_layout()
        self.mode_heading = _subheading("Line and mode")
        self.outer.addWidget(self.mode_heading)
        self.outer.addLayout(mode)
        self.direction = ChoiceBox(AXES)
        self.direction.committed.connect(lambda v: self.set_field("direction", v))
        self.prop_dir = ChoiceBox(AXES)
        self.prop_dir.committed.connect(lambda v: self.set_field("prop_dir", v, "propagation"))
        self.exc_dir = ChoiceBox(AXES)
        self.exc_dir.committed.connect(lambda v: self.set_field("exc_dir", v, "excitation direction"))
        self.feed = ExpressionEdit(s, step=0.5, path=lambda: self._sub("feed_shift_mm"),
                                   tooltip="Distance from the port start to the excitation plane; 0 = automatic")
        self.feed.committed.connect(lambda v: self.set_field("feed_shift_mm", v, "feed shift"))
        self.measure = ExpressionEdit(s, step=0.5, path=lambda: self._sub("measure_plane_shift_mm"),
                                      tooltip="De-embedding reference plane from the port start; 0 = automatic")
        self.measure.committed.connect(lambda v: self.set_field("measure_plane_shift_mm", v, "measurement plane"))
        self.msl_hint = hint("Both shifts must land inside the port box. Leave them at 0 and sensible values "
                             "are derived from the port's own length.")
        self.mode = QLineEdit()
        self.mode.setPlaceholderText("TE10")
        self.mode.editingFinished.connect(lambda: self.set_field("mode", self.mode.text().strip() or "TE10"))
        self.wg_a = ExpressionEdit(s, step=0.5, path=lambda: self._sub("wg_a_mm"))
        self.wg_a.committed.connect(lambda v: self.set_field("wg_a_mm", v, "a"))
        self.wg_b = ExpressionEdit(s, step=0.5, path=lambda: self._sub("wg_b_mm"))
        self.wg_b.committed.connect(lambda v: self.set_field("wg_b_mm", v, "b"))
        self.wg_hint = hint()
        self.r_inner = ExpressionEdit(s, step=0.1, path=lambda: self._sub("r_inner_mm"))
        self.r_inner.committed.connect(lambda v: self.set_field("r_inner_mm", v, "inner radius"))
        self.r_outer = ExpressionEdit(s, step=0.1, path=lambda: self._sub("r_outer_mm"))
        self.r_outer.committed.connect(lambda v: self.set_field("r_outer_mm", v, "outer radius"))
        self.coax_hint = hint()
        self.labels = {key: QLabel() for key in ("feed", "measure", "a", "b", "inner", "outer")}
        mode.addRow("Direction", self.direction)
        mode.addRow("Propagation", self.prop_dir)
        mode.addRow("Excitation", self.exc_dir)
        mode.addRow(self.labels["feed"], self.feed)
        mode.addRow(self.labels["measure"], self.measure)
        mode.addRow(self.msl_hint)
        mode.addRow("Mode", self.mode)
        mode.addRow(self.labels["a"], self.wg_a)
        mode.addRow(self.labels["b"], self.wg_b)
        mode.addRow(self.wg_hint)
        mode.addRow(self.labels["inner"], self.r_inner)
        mode.addRow(self.labels["outer"], self.r_outer)
        mode.addRow(self.coax_hint)
        self._forms: List[QFormLayout] = [general, geometry, line, mode]
        self._mode_form = mode

    # --------------------------------------------------------------- edits

    def path(self) -> Optional[str]:
        return f"ports.{self.index}" if self.index >= 0 else None

    def _sub(self, field: str) -> Optional[str]:
        base = self.path()
        return f"{base}.{field}" if base else None

    def set_field(self, field: str, value, label: str = "") -> None:
        port_id = self.port_id
        self.push(lambda: cmd.SetPortField(self.session, port_id, field, value, label))

    def set_fields(self, values: dict, label: str) -> None:
        port_id = self.port_id
        self.push(lambda: cmd.SetPortFields(self.session, port_id, values, label))

    def _sheets(self):
        return [p for p in self.session.project.primitives if p.kind == PrimitiveKind.SHEET]

    def _push_line(self, start, end) -> None:
        port_id = self.port_id
        self.push(lambda: cmd.SetPortLine(self.session, port_id, start, end))

    def _reverse_line(self) -> None:
        port = edits.find_port(self.session.project, self.port_id or "")
        if port is not None and port.line_start is not None and port.line_end is not None:
            self._push_line(port.line_end, port.line_start)

    def _set_geometry_mode(self, mode: str) -> None:
        sheets = self._sheets()
        self.set_field("sheet", sheets[0].name if mode == "sheet" and sheets else None, "port geometry")

    def _delete(self) -> None:
        port_id = self.port_id
        self.push(lambda: cmd.DeletePort(self.session, port_id))

    def _duplicate(self) -> None:
        port_id = self.port_id
        new_id = self.push(lambda: cmd.DuplicatePort(self.session, port_id))
        if new_id:
            self.session.select("port", new_id)

    def _show(self, widget: QWidget, visible: bool) -> None:
        """Show or hide a row in whichever section holds it."""
        for form in self._forms:
            if form.getWidgetPosition(widget)[0] >= 0:
                form.setRowVisible(widget, visible)
                return

    # ---------------------------------------------------------------- load

    def _load_line(self, port: Port, sheet, unit: str) -> None:
        shown = sheet is not None and port.kind in sheets.LINE_KINDS
        self.line_heading.setVisible(shown)
        for widget in (self.line_state, self.line_start, self.line_end, self.draw_line.parentWidget(),
                       self.line_hint):
            self._show(widget, shown)
        if not shown:
            return
        frame = sheet.coordinate_system or "Global"
        self.start_label.setText(f"Start ({unit})")
        self.end_label.setText(f"End ({unit})")
        for vector in (self.line_start, self.line_end):
            vector.setToolTip(f"In {frame} coordinates, like the sheet's centre. Numbers or expressions.")
        self.line_start.set_value(port.line_start)
        self.line_end.set_value(port.line_end)
        complete = port.line_start is not None and port.line_end is not None
        self.reverse_line.setEnabled(complete)
        self.clear_line.setEnabled(port.line_start is not None or port.line_end is not None)
        self.line_hint.setText(LINE_HINTS[port.kind])

        placed = None
        if complete:
            resolved = self.session.resolution.project
            resolved_port = edits.find_port(resolved, port.id)
            try:
                placed = sheets.port_geometry(resolved, resolved_port, {cs.name: cs for cs in resolved.coordinate_systems})
            except Exception:  # noqa: BLE001 - a broken coordinate system shows in validation
                placed = None
        if placed is None or placed.line is None or placed.exc_axis is None:
            text = ("Not drawn yet. Draw it in the 3D view, or type both ends." if not complete else
                    "Its ends do not resolve to two different points yet.")
            self._set_role(self.line_state, "error")
        else:
            span = placed.line[1] - placed.line[0]
            index = "xyz".index(placed.exc_axis)
            factor = unit_math.factor("length", self.session.project.display_units)
            length = float(np.linalg.norm(span)) / factor
            where = ("excites the source along this line" if placed.along_line else "excites the whole sheet")
            text = (f"Runs {'+' if span[index] > 0 else '−'}{placed.exc_axis.upper()}, "
                    f"{float(f'{length:.4g}'):g} {unit} · {where}.")
            self._set_role(self.line_state, "hint")
        self.line_state.setText(text)

    @staticmethod
    def _set_role(label: QLabel, role: str) -> None:
        if label.property("role") != role:
            label.setProperty("role", role)
            label.style().unpolish(label)
            label.style().polish(label)

    def load(self, port: Port, index: int) -> None:
        self.port_id = port.id
        self.index = index
        self._number = port.number
        units = self.session.project.display_units
        unit = unit_math.label("length", units)
        resolved = self.session.resolution.project
        show = self._show

        self.title.setText(f"Port {port.number} · {edits.PORT_LABELS[port.kind]}")
        self.name.set_value(port.name)
        self.number.set_stored(port.number)
        self.kind.set_value(port.kind.value)
        self.plot.setText(f"Plot S{port.number}{port.number}")
        self.plot.setEnabled(port.excite)
        self.plot.setToolTip("Open a plot of this port's reflection" if port.excite else
                             "Only an excited port has a reflection to plot")
        others_excited = any(p.excite for p in self.session.project.ports if p.id != port.id)
        self.excite_only.setEnabled(others_excited or not port.excite)

        sheet_list = self._sheets()
        self.geometry.set_options([("box", "Start point and size")] + ([("sheet", "A sheet")] if sheet_list else []))
        self.geometry.set_value("sheet" if port.sheet else "box")
        self.sheet.set_options([(sh.name, sh.name) for sh in sheet_list])
        self.sheet.set_value(port.sheet)
        bound = next((sh for sh in sheet_list if sh.name == port.sheet), None)
        on_sheet = bool(port.sheet)
        guided = port.kind != PortKind.LUMPED
        # Only ports from older projects are bare boxes.
        show(self.geometry, not on_sheet)
        self._load_line(port, bound, unit)

        z0 = edits.get_path(resolved, f"{self.path()}.impedance")
        where = f"on sheet {port.sheet}" if on_sheet else "a box"
        cs = f", in {port.coordinate_system}" if port.coordinate_system else ""
        self.summary.setText(f"{'Excited' if port.excite else 'Measured only'} · "
                             f"{float(z0):g} Ω reference · {where}{cs}" if isinstance(z0, (int, float)) else
                             f"{'Excited' if port.excite else 'Measured only'} · {where}{cs}")

        for widget in (self.sheet, self.sheet_hint, self.needs_hint):
            show(widget, on_sheet)
        if on_sheet:
            if bound is not None:
                shape = edits.SHEET_LABELS[bound.sheet_shape or SheetShape.RECTANGLE]
                normal = bound.normal_axis or "z"
                self.sheet_hint.setText(f"{shape} in the {''.join(PLANE_AXES[normal])} plane, normal "
                                        f"{normal.upper()}. The port follows it, so move the sheet to move the port.")
                self.sheet_hint.setProperty("role", "hint")
            else:
                self.sheet_hint.setText(f"No sheet named \"{port.sheet}\" exists any more.")
                self.sheet_hint.setProperty("role", "error")
            self.sheet_hint.style().polish(self.sheet_hint)
            self.needs_hint.setText(edits.SHEET_PORT_NEEDS[port.kind])
        self.extrude_label.setText(f"Extrude ({unit})")
        self.extrude.set_value(port.sheet_extrude_mm)
        show(self.extrude, on_sheet and guided)
        try:
            zero_extrude = float(port.sheet_extrude_mm) == 0.0
        except (TypeError, ValueError):
            zero_extrude = False
        show(self.extrude_hint, on_sheet and guided and zero_extrude)

        self.box.set_value(port.p1, port.p2, port.size_mode)
        for widget in (self.box.first, self.box.second.parentWidget(), self.box.toggle.parentWidget()):
            show(widget, not on_sheet)

        # Lumped excitation is in the sheet plane, so skip the normal.
        axes = [a for a in AXES if not bound or a[0] != (bound.normal_axis or "z")]
        self.direction.set_options(axes)
        self.direction.set_value(port.direction)
        show(self.direction, port.kind == PortKind.LUMPED and not on_sheet)  # on a sheet the line decides

        self.prop_dir.set_value(port.prop_dir or "x")
        show(self.prop_dir, guided and not on_sheet)

        msl = port.kind == PortKind.MICROSTRIP
        self.exc_dir.set_value(port.exc_dir or "z")
        self.labels["feed"].setText(f"Feed shift ({unit})")
        self.labels["measure"].setText(f"Meas. plane ({unit})")
        self.feed.set_value(port.feed_shift_mm)
        self.measure.set_value(port.measure_plane_shift_mm)
        for widget in (self.feed, self.measure, self.msl_hint):
            show(widget, msl)
        show(self.exc_dir, msl and not on_sheet)

        wg = port.kind == PortKind.RECT_WAVEGUIDE
        if not self.mode.hasFocus():
            self.mode.setText(port.mode)
        self.labels["a"].setText(f"a ({unit})")
        self.labels["b"].setText(f"b ({unit})")
        self.wg_a.set_value(port.wg_a_mm or 0.0)
        self.wg_b.set_value(port.wg_b_mm or 0.0)
        show(self.mode, wg)
        for widget in (self.wg_a, self.wg_b):
            show(widget, wg and not on_sheet)
        a = edits.get_path(resolved, f"{self.path()}.wg_a_mm")
        cutoff = 299_792_458 / (2 * float(a) / 1000) if isinstance(a, (int, float)) and a > 0 else None
        self.wg_hint.setText(f"TE10 cutoff {fmt_eng(cutoff, 'Hz')}." if cutoff else "")
        show(self.wg_hint, wg and not on_sheet and cutoff is not None)

        coax = port.kind == PortKind.COAXIAL
        self.labels["inner"].setText(f"Inner r ({unit})")
        self.labels["outer"].setText(f"Outer r ({unit})")
        self.r_inner.set_value(port.r_inner_mm or 0.0)
        self.r_outer.set_value(port.r_outer_mm or 0.0)
        for widget in (self.r_inner, self.r_outer):
            show(widget, coax and not on_sheet)
        r_in = edits.get_path(resolved, f"{self.path()}.r_inner_mm") or 0.0
        r_out = edits.get_path(resolved, f"{self.path()}.r_outer_mm") or 0.0
        coax_text = coax_hint_text(float(r_out), float(r_in), z0)
        self.coax_hint.setText(coax_text)
        show(self.coax_hint, coax and not on_sheet and bool(coax_text))

        # Hide a section heading when all its rows are hidden.
        form = self._mode_form
        self.mode_heading.setVisible(any(form.isRowVisible(i) for i in range(form.rowCount())))

        self.impedance.set_value(port.impedance)
        self.excite.setChecked(port.excite)
