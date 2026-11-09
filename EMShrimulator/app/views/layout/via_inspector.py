"""The inspector page for selected vias: padstack, centre, net and backdrills. Edits apply to every selected via."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.layout.copper_tool import VIA_REF, Ref
from EMShrimulator.app.widgets.form import ChoiceBox, form_layout, hint, row
from EMShrimulator.app.widgets.unit_spinbox import UnitSpinBox
from EMShrimulator.core.layout import padstacks as pstk

SIDES = (("top", "From top"), ("bottom", "From bottom"))
MIXED = "\x00mixed"  # the selected vias differ


def _um(mm: Optional[float]) -> str:
    return "none" if mm is None else f"{mm * 1000:g} µm"


def default_backdrill(side: str, hole_mm: float, layers: List[str], span: Tuple[int, int]) -> Optional[Dict]:
    """A default backdrill (keeping the layer next to the far face, 200 µm over the hole, 50 µm stub); None
    for a two-layer via."""
    start, stop = span
    if stop - start < 2:
        return None
    keep = layers[stop - 1] if side == "top" else layers[start + 1]
    return {"keep_layer": keep, "diameter_mm": round(hole_mm + 0.2, 4), "stub_mm": 0.05}


class _BackdrillRow:
    """One face's backdrill: on/off, kept layer, diameter and stub."""

    def __init__(self, owner: "ViaProperties", side: str, label: str):
        self.owner, self.side = owner, side
        self.on = QCheckBox(label)
        self.on.clicked.connect(self._toggled)
        self.keep = ChoiceBox(tooltip="The deepest layer the via still connects: the drill stops short of it")
        self.keep.committed.connect(lambda _v: self._commit())
        self.diameter = UnitSpinBox("µm", scale=1e-3, decimals=1, minimum=1, step=25,
                                    tooltip="The backdrill's diameter: wider than the via's hole, as it cuts its "
                                            "plating out")
        self.diameter.committed.connect(lambda _v: self._commit())
        self.stub = UnitSpinBox("µm", scale=1e-3, decimals=1, minimum=0, step=10,
                                tooltip="How much barrel the drill leaves beyond the kept layer: the stub")
        self.stub.committed.connect(lambda _v: self._commit())
        self.detail = row(QLabel("keep"), self.keep, QLabel("⌀"), self.diameter, QLabel("stub"), self.stub)

    def _toggled(self, on: bool) -> None:
        o = self.owner
        if not on:
            o.set_backdrill(self.side, None)
            return
        via, ps = o.first()
        if via is None or ps is None:
            return
        layers = pstk.layer_names(o.session.project)
        value = default_backdrill(self.side, ps.hole_mm, layers, pstk.span(ps, layers))
        if value is None:
            self.on.setChecked(False)
            return
        o.set_backdrill(self.side, value)

    def _commit(self) -> None:
        if self.on.isChecked() and self.keep.value() is not None:
            self.owner.set_backdrill(self.side, {"keep_layer": self.keep.value(), "diameter_mm": self.diameter.stored(),
                                                 "stub_mm": self.stub.stored()})

    def load(self, drill, options: List[Tuple[str, str]]) -> None:
        self.on.setChecked(drill is not None)
        self.keep.set_options(options)
        self.detail.setVisible(drill is not None)
        if drill is not None:
            self.keep.set_value(drill.keep_layer)
            self.diameter.set_stored(drill.diameter_mm)
            self.stub.set_stored(drill.stub_mm)


class ViaProperties(QWidget):
    padstackRequested = Signal(str)  # open this padstack's page

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.tool = None  # the canvas's CopperTool, set by the layout view
        self.refs: Tuple[Ref, ...] = ()

        self.title = QLabel("Via")
        self.title.setProperty("role", "heading")
        self.padstack = ChoiceBox(tooltip="The padstack the selected vias are instances of")
        self.padstack.committed.connect(lambda name: self._set("padstack", name) if name != MIXED else None)
        edit = QPushButton("Edit…")
        edit.setToolTip("Open the padstack: its drill, plating, span and each layer's pad and antipad")
        edit.clicked.connect(self._edit_padstack)
        self.padstack_row = row(self.padstack, edit)
        self.summary = hint()
        self.net = QLineEdit()
        self.net.setPlaceholderText("no net")
        self.net.setToolTip("The net its pads carry in the Gerbers: what colour-by-net and highlighting show")
        self.net.editingFinished.connect(lambda: self._set("net", self.net.text().strip() or None))
        self.x = UnitSpinBox("mm", decimals=4, step=0.1)
        self.y = UnitSpinBox("mm", decimals=4, step=0.1)
        self.x.committed.connect(self._move_to)
        self.y.committed.connect(self._move_to)
        self.dx = UnitSpinBox("mm", decimals=4, step=0.1)
        self.dy = UnitSpinBox("mm", decimals=4, step=0.1)
        move = QPushButton("Move")
        move.setToolTip("Move the selected vias by ΔX, ΔY; the traces ending on them follow")
        move.clicked.connect(self._move_by)
        self.move_row = row(QLabel("Δ"), self.dx, self.dy, move)
        self.backdrills = {side: _BackdrillRow(self, side, label) for side, label in SIDES}
        self.layers_hint = hint()
        self.note = hint("Drag a via to move it: the traces ending on it follow on every layer it has a pad. "
                         "Ctrl+click adds to the selection.", role="note")
        select_all = QPushButton("Select all of this padstack")
        select_all.clicked.connect(self._select_padstack)
        delete = QPushButton("Delete")
        delete.setProperty("role", "danger")
        delete.setToolTip("Remove the selected vias, their pads with them")
        delete.clicked.connect(lambda: self.tool.delete(self.refs) if self.tool is not None else None)

        form = form_layout()
        self.form = form
        form.addRow("Padstack", self.padstack_row)
        form.addRow(self.summary)
        form.addRow("Net", self.net)
        form.addRow("Centre X", self.x)
        form.addRow("Centre Y", self.y)
        form.addRow("Move by", self.move_row)
        for side, _label in SIDES:
            drill = self.backdrills[side]
            form.addRow("Backdrill" if side == "top" else "", drill.on)
            form.addRow("", drill.detail)
        form.addRow(self.layers_hint)
        form.addRow(self.note)
        form.addRow("", row(select_all, delete))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.addWidget(self.title)
        layout.addLayout(form)

    # ---------------------------------------------------------------- edits

    def ids(self) -> List[int]:
        return [r[1] for r in self.refs if r[0] == VIA_REF]

    def vias(self) -> list:
        wanted = set(self.ids())
        return [v for v in self.session.project.vias or [] if v.id in wanted]

    def first(self):
        vias = self.vias()
        if not vias:
            return None, None
        return vias[0], pstk.padstack_map(self.session.project).get(vias[0].padstack)

    def _set(self, field: str, value) -> None:
        ids = self.ids()
        if ids:
            guarded(lambda: self.session.push(cmd.SetViasField(self.session, ids, field, value)))

    def set_backdrill(self, side: str, value: Optional[Dict]) -> None:
        self._set(f"backdrill_{side}", value)

    def _move_to(self, _value: float) -> None:
        via, _ps = self.first()
        if via is not None and len(self.refs) == 1 and self.tool is not None:
            self.tool.move_by(self.refs, self.x.stored() - via.x_mm, self.y.stored() - via.y_mm, "Move via")

    def _move_by(self) -> None:
        if self.tool is not None and (self.dx.stored() or self.dy.stored()):
            self.tool.move_by(self.refs, self.dx.stored(), self.dy.stored(), "Move vias")

    def _edit_padstack(self) -> None:
        via, _ps = self.first()
        name = self.padstack.value()
        self.padstackRequested.emit(name if name not in (None, MIXED) else via.padstack if via is not None else "")

    def _select_padstack(self) -> None:
        via, _ps = self.first()
        if via is None or self.tool is None:
            return
        self.tool.select(tuple((VIA_REF, v.id, 0) for v in self.session.project.vias or []
                               if v.padstack == via.padstack))

    # ---------------------------------------------------------------- showing

    def load(self, refs: Tuple[Ref, ...]) -> None:
        self.refs = tuple(refs)
        p = self.session.project
        vias = self.vias()
        via, ps = self.first()
        if via is None:
            self.title.setText("Via")
            self.summary.setText("That via no longer exists.")
            return
        layers = pstk.layer_names(p)
        single = len(vias) == 1
        names = sorted({v.padstack for v in vias})
        self.title.setText(f"Via V{via.id}" if single else f"{len(vias)} vias")
        self.padstack.set_options([(q.name, q.name) for q in p.padstacks] +
                                  ([] if len(names) == 1 else [(MIXED, "mixed")]))
        self.padstack.set_value(names[0] if len(names) == 1 else MIXED)
        if ps is not None:
            start, stop = pstk.span(ps, layers)
            through = (start, stop) == (0, len(layers) - 1)
            span = f"{layers[start]} → {layers[stop]}" + (" (through)" if through else "") if layers else "no layers"
            pad = "as imported" if ps.shape == "imported" else _um(ps.pad_mm)
            self.summary.setText(f"Hole {_um(ps.hole_mm)} · pad {pad} · {span}" +
                                 ("" if ps.plated else " · unplated: not simulated"))
        else:
            self.summary.setText(f"Its padstack {via.padstack} no longer exists.")
        nets = sorted({v.net or "" for v in vias})
        if not self.net.hasFocus():
            self.net.setText(nets[0] if len(nets) == 1 else "")
            self.net.setPlaceholderText("no net" if len(nets) == 1 else "mixed")
        self.x.set_stored(via.x_mm)
        self.y.set_stored(via.y_mm)
        self.form.setRowVisible(self.x, single)
        self.form.setRowVisible(self.y, single)

        if ps is not None and layers:
            start, stop = pstk.span(ps, layers)
            options = [(name, name) for name in layers[start:stop + 1]]
        else:
            options = []
        for side, _label in SIDES:
            drill = getattr(via, f"backdrill_{side}")
            self.backdrills[side].load(drill, options)
            self.backdrills[side].on.setEnabled(len(options) > 2 or drill is not None)
        if ps is not None:
            rows = pstk.via_layers(via, ps, layers)
            parts = []
            for r in rows:
                if r.drill_mm > 0:
                    parts.append(f"{r.layer}: backdrilled")
                elif r.link is not None:
                    parts.append(f"{r.layer}: imported pad")
                else:
                    parts.append(f"{r.layer}: {_um(r.pad_mm) if r.pad_mm else 'no pad'}"
                                 + (f", antipad {_um(r.antipad_mm)}" if r.antipad_mm else ""))
            self.layers_hint.setText(" · ".join(parts))
        else:
            self.layers_hint.setText("")
