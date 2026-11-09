"""The inspector beside the board: the editor for the selected port, pair, copper or vias."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.layout import palette
from EMShrimulator.app.views.layout.copper_inspector import CopperProperties
from EMShrimulator.app.views.layout.copper_tool import VIA_REF
from EMShrimulator.app.views.layout.via_inspector import ViaProperties
from EMShrimulator.app.views.shell import Inspector
from EMShrimulator.app.widgets.form import ChoiceBox, Segmented, form_layout, hint
from EMShrimulator.app.widgets.unit_spinbox import UnitSpinBox
from EMShrimulator.core.layout.edits import pair_of
from EMShrimulator.core.layout.model import LayoutPortKind
from EMShrimulator.core.layout.ports import metal_names

KIND_LABELS = [(LayoutPortKind.LUMPED.value, "Edge · gap port (lumped, to reference)"),
               (LayoutPortKind.MICROSTRIP.value, "Edge · microstrip launch (de-embedded)"),
               (LayoutPortKind.CIRCUIT.value, "Circuit port (+ / − terminals)")]
KIND_HINTS = {
    LayoutPortKind.LUMPED: "A resistive sheet standing on the edge, from the signal layer down (or up) to the "
                           "reference layer, as wide as the edge it sits on.",
    LayoutPortKind.MICROSTRIP: "A matched transmission-line launch that extends into the trace from the edge and "
                               "de-embeds to it. Rotation must be a multiple of 90°.",
    LayoutPortKind.CIRCUIT: "A resistive port from the + terminal to the − terminal. Both can sit on the same "
                            "layer (to a coplanar ground) or on different layers. Drag the − handle on the canvas "
                            "to move that terminal alone.",
}
AUTO = -1


class LayoutPortEditor(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.number: Optional[int] = None
        self.references = {}
        s = session

        def field(name: str):
            return lambda value: self._set(name, value)

        self.title = QLabel("Port")
        self.title.setProperty("role", "heading")
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set("name", self.name.text().strip())
                                          if self.name.text().strip() else None)
        self.kind = ChoiceBox(KIND_LABELS)
        self.kind.committed.connect(lambda v: self._set("kind", LayoutPortKind(v)))
        self.kind_hint = hint()
        self.snap_hint = hint()
        self.x = UnitSpinBox("mm", step=0.1, decimals=4)
        self.x.committed.connect(field("x_mm"))
        self.y = UnitSpinBox("mm", step=0.1, decimals=4)
        self.y.committed.connect(field("y_mm"))
        self.neg_x = UnitSpinBox("mm", step=0.1, decimals=4)
        self.neg_x.committed.connect(field("neg_x_mm"))
        self.neg_y = UnitSpinBox("mm", step=0.1, decimals=4)
        self.neg_y.committed.connect(field("neg_y_mm"))
        self.rotation = Segmented([(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")])
        self.rotation.committed.connect(lambda v: self._set("rotation_deg", float(v)))
        self.width = UnitSpinBox("µm", step=25, decimals=1, minimum=0)
        self.width.committed.connect(field("width_um"))
        self.length = UnitSpinBox("µm", step=100, decimals=1, minimum=0, tooltip=(
            "How far the launch runs into the trace; keep it on the straight part. The S-parameters are "
            "measured halfway along it, so a longer launch shifts their phase, not their size."))
        self.length.committed.connect(field("length_um"))
        self.layer = ChoiceBox()
        self.layer.committed.connect(lambda v: self._set("layer", int(v)))
        self.plane = ChoiceBox()
        self.plane.committed.connect(self._set_plane)
        self.reference_hint = hint()
        self.impedance = UnitSpinBox("Ω", step=1, decimals=2, minimum=0)
        self.impedance.committed.connect(field("impedance"))
        self.excite = QCheckBox("Excite this port")
        self.excite.clicked.connect(lambda on: self._set("excite", bool(on)))
        # Its differential pair, or a way to start one.
        self.pair_label = QLabel()
        self.pair_open = QPushButton("Open")
        self.pair_open.setProperty("role", "flat")
        self.pair_open.clicked.connect(self._open_pair)
        self.pair_new = QPushButton("New pair, this port +")
        self.pair_new.setToolTip("Start a differential pair with this as its + port, then choose its − port")
        self.pair_new.clicked.connect(self._new_pair)
        pair_row = QWidget()
        pair_layout = QHBoxLayout(pair_row)
        pair_layout.setContentsMargins(0, 0, 0, 0)
        pair_layout.setSpacing(6)
        pair_layout.addWidget(self.pair_label, 1)
        pair_layout.addWidget(self.pair_open)
        pair_layout.addWidget(self.pair_new)
        self._pair_index: Optional[int] = None
        delete = QPushButton("Delete port")
        delete.setProperty("role", "danger")
        delete.clicked.connect(lambda: guarded(lambda: s.push(cmd.DeleteLayoutPort(s, self.number))))

        form = form_layout()
        self.form = form
        form.addRow("Name", self.name)
        form.addRow("Type", self.kind)
        form.addRow(self.kind_hint)
        form.addRow("X", self.x)
        form.addRow("Y", self.y)
        form.addRow("Signal layer", self.layer)
        form.addRow("− X", self.neg_x)
        form.addRow("− Y", self.neg_y)
        form.addRow("Reference", self.plane)
        form.addRow(self.reference_hint)
        form.addRow("Rotation", self.rotation)
        form.addRow("Width", self.width)
        form.addRow("Length", self.length)
        form.addRow("Z₀", self.impedance)
        form.addRow(self.snap_hint)
        form.addRow("", self.excite)
        form.addRow("Diff. pair", pair_row)
        form.addRow("", delete)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.addWidget(self.title)
        layout.addLayout(form)

    def _set(self, field: str, value) -> None:
        number = self.number
        if number is None:
            return
        guarded(lambda: self.session.push(cmd.SetLayoutPortField(self.session, number, field, value)))

    def _set_plane(self, value) -> None:
        if int(value) == AUTO:
            self._set("reference", "auto")
        else:
            self._set("plane", int(value))

    def _show_row(self, widget: QWidget, visible: bool) -> None:
        self.form.setRowVisible(widget, visible)

    def _open_pair(self) -> None:
        if self._pair_index is not None:
            self.session.select("diff_pair", self._pair_index)

    def _new_pair(self) -> None:
        s, number = self.session, self.number
        if number is None:
            return
        index = guarded(lambda: s.push(cmd.AddDiffPair(s, number, None)))
        if index is not None:
            s.select("diff_pair", index)

    def load(self, port) -> None:
        self.number = port.number
        circuit = port.kind == LayoutPortKind.CIRCUIT
        self.title.setText(f"Port P{port.number + 1}")
        if not self.name.hasFocus():
            self.name.setText(port.name)
        self.kind.set_value(port.kind.value)
        self.kind_hint.setText(KIND_HINTS[port.kind])

        names = metal_names(self.session.project)
        layer_options = [(i, f"{i} · {name}") for i, name in enumerate(names)] or [(0, "0")]
        for extra in (port.layer, port.plane):
            if extra >= len(names):
                layer_options.append((extra, f"{extra} · (not in stackup)"))
        self.layer.set_options(layer_options)
        self.layer.set_value(port.layer)

        choice = self.references.get(port.number)
        if circuit:
            self.form.labelForField(self.layer).setText("+ layer")
            self.form.labelForField(self.plane).setText("− layer")
            self.form.labelForField(self.x).setText("+ X")
            self.form.labelForField(self.y).setText("+ Y")
            self.plane.set_options(layer_options)
            self.plane.set_value(port.plane)
            self.reference_hint.setText("")
        else:
            self.form.labelForField(self.layer).setText("Signal layer")
            self.form.labelForField(self.plane).setText("Reference")
            self.form.labelForField(self.x).setText("X")
            self.form.labelForField(self.y).setText("Y")
            auto_label = "Auto · nearest plane"
            if port.reference == "auto" and choice is not None and choice.plane is not None:
                auto_label = f"Auto · {names[choice.plane] if choice.plane < len(names) else choice.plane}"
            self.plane.set_options([(AUTO, auto_label)] + layer_options)
            self.plane.set_value(AUTO if port.reference == "auto" else port.plane)
            if port.reference == "auto":
                self.reference_hint.setText(
                    ("→ " + choice.note) if choice is not None else "→ resolving the nearest plane…")
            else:
                self.reference_hint.setText("Fixed reference layer. Choose Auto to use the nearest plane.")

        self.x.set_stored(port.x_mm)
        self.y.set_stored(port.y_mm)
        self.neg_x.set_stored(port.neg_x_mm)
        self.neg_y.set_stored(port.neg_y_mm)
        self.rotation.set_value(int(round(port.rotation_deg / 90) * 90) % 360)
        self.width.set_stored(port.width_um)
        self.length.set_stored(port.length_um)
        self.impedance.set_stored(port.impedance)
        self.excite.setChecked(port.excite)
        snaps = []
        if port.snap:
            snaps.append(f"+ {port.snap}" if circuit else port.snap)
        if circuit and port.neg_snap:
            snaps.append(f"− {port.neg_snap}")
        self.snap_hint.setText("Snapped to " + ", ".join(snaps) if snaps else "Free position (not snapped)")
        owner = pair_of(self.session.project, port.number)
        self._pair_index = owner[0] if owner is not None else None
        if owner is not None:
            side = "+" if owner[1].positive == port.number else "−"
            color = palette.pair_color(owner[0]).name()
            self.pair_label.setText(f"<span style='color:{color}'>●</span> {owner[1].name}, the {side} port")
        else:
            self.pair_label.setText("None")
        self.pair_open.setVisible(owner is not None)
        self.pair_new.setVisible(owner is None)

        # An edge port spans its edge, so its width is fixed.
        on_edge = port.kind == LayoutPortKind.LUMPED and (port.snap or "").startswith("edge")
        launch = port.kind == LayoutPortKind.MICROSTRIP
        self.width.setToolTip(
            "The launch strip's width. Match the trace: a different width is a step in impedance, seen as "
            "reflection in S11." if launch else "How wide the port is across the copper.")
        self._show_row(self.neg_x, circuit)
        self._show_row(self.neg_y, circuit)
        self._show_row(self.rotation, not circuit)
        self._show_row(self.width, not on_edge)
        self._show_row(self.length, launch)


UNSET = -1
PAIR_ABOUT = ("The two ports are driven together: the results add mixed-mode S-parameters, the "
              "differential (Sdd) and common (Scc) modes and the conversions between them (Sdc, Scd), "
              "referenced to the impedances below.")


class DiffPairEditor(QWidget):
    """One differential pair: its + and − ports and each mode's reference impedance."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.index: Optional[int] = None
        s = session

        self.title = QLabel("Differential pair")
        self.title.setProperty("role", "heading")
        self.name = QLineEdit()
        self.name.editingFinished.connect(lambda: self._set("name", self.name.text()))
        self.positive = ChoiceBox()
        self.positive.committed.connect(lambda v: self._set("positive", None if v == UNSET else int(v)))
        self.negative = ChoiceBox()
        self.negative.committed.connect(lambda v: self._set("negative", None if v == UNSET else int(v)))
        swap = QPushButton("Swap + and −")
        swap.setToolTip("Exchange the two ports: the pair's polarity")
        swap.clicked.connect(lambda: guarded(lambda: s.push(cmd.SwapDiffPair(s, self.index)))
                             if self.index is not None else None)
        self.z_diff = UnitSpinBox("Ω", step=1, decimals=2, minimum=0.01,
                                  tooltip="The differential mode's reference impedance (HFSS's default is 100 Ω)")
        self.z_diff.committed.connect(lambda v: self._set("z_diff", v))
        self.z_common = UnitSpinBox("Ω", step=1, decimals=2, minimum=0.01,
                                    tooltip="The common mode's reference impedance (HFSS's default is 25 Ω)")
        self.z_common.committed.connect(lambda v: self._set("z_common", v))
        self.drive_hint = hint()
        self.excite_both = QPushButton("Excite both ports")
        self.excite_both.clicked.connect(self._excite_both)
        self.results_hint = hint(role="note")
        delete = QPushButton("Delete pair")
        delete.setProperty("role", "danger")
        delete.setToolTip("The ports stay; only the pairing goes")
        delete.clicked.connect(lambda: guarded(lambda: s.push(cmd.DeleteDiffPair(s, self.index)))
                               if self.index is not None else None)

        form = form_layout()
        self.form = form
        form.addRow("Name", self.name)
        form.addRow(hint(PAIR_ABOUT))
        form.addRow("+ port", self.positive)
        form.addRow("− port", self.negative)
        form.addRow("", swap)
        form.addRow("Differential Z", self.z_diff)
        form.addRow("Common Z", self.z_common)
        form.addRow(self.drive_hint)
        form.addRow("", self.excite_both)
        form.addRow(self.results_hint)
        form.addRow("", delete)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.addWidget(self.title)
        layout.addLayout(form)

    def _set(self, field: str, value) -> None:
        index = self.index
        if index is None:
            return
        guarded(lambda: self.session.push(cmd.SetDiffPairField(self.session, index, field, value)))

    def _excite_both(self) -> None:
        if self.index is None:
            return
        pair = self.session.project.diff_pairs[self.index]
        for number in (pair.positive, pair.negative):
            port = next((p for p in self.session.project.ports if p.number == number), None)
            if port is not None and not port.excite:
                guarded(lambda n=number: self.session.push(cmd.SetLayoutPortField(self.session, n, "excite", True)))

    def load(self, index: int) -> None:
        p = self.session.project
        pair = p.diff_pairs[index]
        self.index = index
        color = palette.pair_color(index).name()
        self.title.setText(f"Differential pair {pair.name}")
        if not self.name.hasFocus():
            self.name.setText(pair.name)
        taken = {n for i, other in enumerate(p.diff_pairs) if i != index for n in (other.positive, other.negative)}

        def options(current: Optional[int]):
            out = [(UNSET, "— choose a port —")]
            for port in sorted(p.ports, key=lambda q: q.number):
                if port.number in taken and port.number != current:
                    continue
                label = f"P{port.number + 1}" + (f"  {port.name}" if port.name != f"P{port.number + 1}" else "")
                out.append((port.number, label + ("" if port.excite else "  · not excited")))
            return out

        self.positive.set_options(options(pair.positive))
        self.positive.set_value(UNSET if pair.positive is None else pair.positive)
        self.negative.set_options(options(pair.negative))
        self.negative.set_value(UNSET if pair.negative is None else pair.negative)
        self.z_diff.set_stored(pair.z_diff)
        self.z_common.set_stored(pair.z_common)

        ports = {q.number: q for q in p.ports}
        sides = [ports.get(n) for n in (pair.positive, pair.negative)]
        if any(side is None for side in sides):
            missing = " and ".join(sign for sign, side in zip("+−", sides) if side is None)
            self.drive_hint.setText(f"<span style='color:{color}'>●</span> Choose its {missing} port.")
            self.excite_both.setVisible(False)
        else:
            idle = [q for q in sides if not q.excite]
            if idle:
                names = " and ".join(f"P{q.number + 1}" for q in idle)
                self.drive_hint.setText(f"<span style='color:{color}'>●</span> {names} "
                                        f"{'are' if len(idle) > 1 else 'is'} not excited: {pair.name}'s own "
                                        "reflection (Sdd, Scc) cannot come out, only what reaches it.")
            else:
                self.drive_hint.setText(f"<span style='color:{color}'>●</span> Both ports are excited: "
                                        f"{pair.name} is driven in each mode.")
            self.excite_both.setVisible(bool(idle))
        k = index + 1
        self.results_hint.setText(f"In the results this is pair {k}: Sdd{k}{k} and Scc{k}{k} are its reflections, "
                                  f"and Zdiff/Zcomm {pair.name} its input impedances.")


class PortInspector(Inspector):
    """The inspector beside the board; `shownChanged` tells the workspace."""

    shownChanged = Signal(bool)

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self._shown = False
        self.editor = LayoutPortEditor(session)
        self.pair_editor = DiffPairEditor(session)
        self.copper = CopperProperties(session)
        self.vias = ViaProperties(session)
        self.add_editor(self.editor)
        self.add_editor(self.pair_editor)
        self.add_editor(self.copper)
        self.add_editor(self.vias)
        session.projectChanged.connect(self.refresh)
        session.selectionChanged.connect(lambda _s: self.refresh())
        self.refresh()

    @property
    def shown(self) -> bool:
        return self._shown

    def refresh(self) -> None:
        selection = self.session.selection
        port = pair = copper = vias = None
        if selection is not None and selection.kind == "copper":
            if selection.key and all(ref[0] == VIA_REF for ref in selection.key):
                vias = tuple(selection.key)
            else:
                copper = tuple(selection.key)
        elif selection is not None and selection.kind == "layout_port":
            port = next((p for p in self.session.project.ports if p.number == selection.key), None)
        elif selection is not None and selection.kind == "diff_pair" \
                and 0 <= int(selection.key) < len(self.session.project.diff_pairs):
            pair = int(selection.key)
        if port is not None:
            self.editor.load(port)
            self.show_title(self.editor)
        elif pair is not None:
            self.pair_editor.load(pair)
            self.show_title(self.pair_editor)
        elif copper is not None:
            self.copper.load(copper)
            self.show_title(self.copper)
        elif vias is not None:
            self.vias.load(vias)
            self.show_title(self.vias)
        self.editor.setVisible(port is not None)
        self.pair_editor.setVisible(pair is not None)
        self.copper.setVisible(copper is not None)
        self.vias.setVisible(vias is not None)
        shown = port is not None or pair is not None or copper is not None or vias is not None
        if shown != self._shown:
            self._shown = shown
            self.shownChanged.emit(self._shown)
