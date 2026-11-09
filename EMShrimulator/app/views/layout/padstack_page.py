"""The padstacks page: one padstack's drill, plating, layer span, pads and antipads, with a cross-section."""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app import theme
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.widgets.form import ChoiceBox, Segmented, card, centred_column, form_layout, hint, row
from EMShrimulator.app.widgets.unit_spinbox import UnitSpinBox
from EMShrimulator.core.layout import padstacks as pstk

SHAPES = [("circle", "Disc pads"), ("imported", "Pads as imported")]
TOP, BOTTOM = "\x00top", "\x00bottom"  # the span boxes' values for the board's own faces


def _um_text(mm: Optional[float]) -> str:
    return "none" if mm is None else f"{mm * 1000:g}"


def _parse_um(text: str) -> Optional[float]:
    """A table cell in µm as mm; "none", "-" or empty gives None. Raises ValueError otherwise."""
    text = text.strip().lower().replace("µm", "").replace("um", "").strip()
    if text in ("", "none", "-", "—", "no"):
        return None
    return float(text) / 1000.0


class CrossSection(QWidget):
    """A cross-section of the via the padstack makes, through the stackup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.project = None
        self.padstack = None
        self.colors: Dict[str, str] = {}

    def show_padstack(self, project, padstack, colors: Dict[str, str]) -> None:
        self.project, self.padstack, self.colors = project, padstack, colors
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        p, ps = self.project, self.padstack
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        c = theme.colors()
        painter.fillRect(self.rect(), QColor(c["sunken"]))
        if p is None or ps is None or not p.stackup:
            painter.setPen(QColor(c["text_faint"]))
            painter.drawText(self.rect(), Qt.AlignCenter, "No stackup")
            return
        layers = pstk.layer_names(p)
        start, stop = pstk.span(ps, layers)
        label_w, margin = 92.0, 14.0
        area = QRectF(label_w, margin, self.width() - label_w - margin, self.height() - 2 * margin)
        stack = [layer for layer in p.stackup if layer.type in ("copper", "core", "prepreg")]  # no mask or silk
        # Each layer gets its share of the height, but at least a few pixels.
        thick = [max(layer.thickness_mm or 0.0, 1e-6) for layer in stack]
        floor = [5.0 if layer.type == "copper" else 12.0 for layer in stack]
        spare = max(area.height() - sum(floor), 0.0)
        total = sum(thick)
        heights = [f + spare * t / total for f, t in zip(floor, thick)]
        widest = max([ps.hole_mm, ps.pad_mm or 0.0, ps.antipad_mm] +
                     [max(r.pad_mm or 0.0, r.antipad_mm) for r in ps.layers]) * 1.5
        scale = min(area.width() * 0.8, 260.0) / max(widest, 1e-6)  # px per mm
        cx = area.center().x()
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF() - 1, 7))
        painter.setFont(font)
        y = area.top()
        metal = -1
        tops: Dict[int, tuple] = {}
        for layer, h in zip(stack, heights):
            band = QRectF(area.left(), y, area.width(), h)
            if layer.type == "copper":
                metal += 1
                name = layers[metal] if metal < len(layers) else layer.name
                color = QColor(self.colors.get(name.lower(), c["accent"]))
                tops[metal] = (y, h)
                pad, antipad = pstk.layer_pad(ps, name)
                spanned = start <= metal <= stop
                if ps.shape == "imported" and spanned and pad is not None:
                    pad = ps.pad_mm  # each via keeps its imported pad
                clear = (antipad or 0.0) if spanned else 0.0
                # The plane, cut back by the antipad, then the pad in the gap.
                cut = max(clear, pad or 0.0, ps.hole_mm) * scale / 2 if spanned else 0.0
                painter.fillRect(QRectF(band.left(), y, cx - cut - band.left(), h), color.darker(140))
                painter.fillRect(QRectF(cx + cut, y, band.right() - cx - cut, h), color.darker(140))
                if spanned and pad:
                    painter.fillRect(QRectF(cx - pad * scale / 2, y, pad * scale, h), color)
                painter.setPen(QColor(c["text"] if spanned else c["text_faint"]))
                painter.drawText(QRectF(0, y - 8, label_w - 8, h + 16), Qt.AlignRight | Qt.AlignVCenter, name)
            else:
                fill = QColor(c["line_strong"])
                fill.setAlphaF(0.18)
                painter.fillRect(band, fill)
            y += h
        # The barrel: plating each side of the hole, over the span.
        if ps.plated and start in tops and stop in tops:
            y0 = tops[start][0]
            y1 = tops[stop][0] + tops[stop][1]
            r = ps.hole_mm * scale / 2
            wall = max(ps.plating_um / 1000 * scale, 1.5)
            barrel = QColor(self.colors.get(layers[start].lower(), c["accent"]) if layers else c["accent"])
            painter.fillRect(QRectF(cx - r - wall, y0, wall, y1 - y0), barrel)
            painter.fillRect(QRectF(cx + r, y0, wall, y1 - y0), barrel)
            fill = QColor(c["text_faint"])
            fill.setAlphaF(0.12 if ps.fill_epsilon > 1.0 else 0.0)
            painter.fillRect(QRectF(cx - r, y0, 2 * r, y1 - y0), fill)
        elif not ps.plated:
            painter.setPen(QPen(QColor(c["text_faint"]), 1, Qt.DashLine))
            r = ps.hole_mm * scale / 2
            painter.drawLine(cx - r, area.top(), cx - r, area.bottom())
            painter.drawLine(cx + r, area.top(), cx + r, area.bottom())
        painter.setPen(QColor(c["text_faint"]))
        painter.drawText(QRectF(area.left(), self.height() - margin, area.width(), margin),
                         Qt.AlignHCenter | Qt.AlignVCenter, "widths to scale; thicknesses not")


class PadstackPage(QScrollArea):
    selectVias = Signal(str)  # select every via of this padstack on the board
    padstackShown = Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        self.setObjectName("setup-page")
        self.name: Optional[str] = None
        self.colors: Dict[str, str] = {}
        self._loading = False

        # ------------------------------------------------------------ which
        self.chooser = ChoiceBox(tooltip="The padstack shown")
        self.chooser.committed.connect(self.show_padstack)
        self.count = QLabel()
        select = QPushButton("Select its vias")
        select.setToolTip("Select every via of this padstack on the board")
        select.clicked.connect(lambda: self.selectVias.emit(self.name) if self.name else None)
        frame, layout = card("Padstack", "A via definition. Every via on the board is an instance of one; change it "
                                         "here and they all change. Vias read from the drill file share one "
                                         "per drill size and pad.")
        self.title_name = QLineEdit()
        self.title_name.editingFinished.connect(lambda: self._set("name", self.title_name.text()))
        form = form_layout()
        form.addRow("Shown", row(self.chooser, self.count, select))
        form.addRow("Name", self.title_name)
        layout.addLayout(form)
        self.head_card = frame

        # ------------------------------------------------------------ drill
        frame, layout = card("Drill", "The finished hole, its plating and what fills it. An unplated hole (a "
                                      "mounting hole) is drawn but carries no current, and is not simulated.")
        self.hole = UnitSpinBox("µm", scale=1e-3, decimals=1, minimum=1, step=25)
        self.hole.committed.connect(lambda v: self._set("hole_mm", v))
        self.plated = QCheckBox("Plated")
        self.plated.clicked.connect(lambda on: self._set("plated", bool(on)))
        self.plating = UnitSpinBox("µm", decimals=1, minimum=0, step=5,
                                   tooltip="The barrel's copper wall, inside the drilled hole")
        self.plating.committed.connect(lambda v: self._set("plating_um", v))
        self.fill = UnitSpinBox(decimals=2, minimum=1, step=0.1,
                                tooltip="What fills the barrel: 1 for an open via, ~3.5 for one plugged with resin")
        self.fill.committed.connect(lambda v: self._set("fill_epsilon", v))
        form = form_layout()
        form.addRow("Hole ⌀", self.hole)
        form.addRow("", self.plated)
        form.addRow("Plating", self.plating)
        form.addRow("Fill εr", self.fill)
        layout.addLayout(form)
        self.drill_card = frame

        # ------------------------------------------------------------- pads
        frame, layout = card("Pads and span", "The layers the via runs between - a blind or buried via spans "
                                              "fewer than all - and on each its pad and antipad. The antipad "
                                              "clears every copper around the pad within its diameter, a trace "
                                              "of the via's own net too: set it on planes. A row of the table "
                                              "differs from the padstack's own pad and antipad; type \"none\" for "
                                              "no pad on a layer.")
        self.shape = Segmented(SHAPES)
        self.shape.committed.connect(lambda v: self._set("shape", v))
        self.shape_hint = hint()
        self.pad = UnitSpinBox("µm", scale=1e-3, decimals=1, minimum=1, step=25)
        self.pad.committed.connect(lambda v: self._set("pad_mm", v))
        self.no_pad = QCheckBox("No pad")
        self.no_pad.clicked.connect(lambda on: self._set("pad_mm", None if on else max(self.pad.stored(), 0.001)))
        self.antipad = UnitSpinBox("µm", scale=1e-3, decimals=1, minimum=0, step=25,
                                   tooltip="0: no clearance; the copper stays as the board has it")
        self.antipad.committed.connect(lambda v: self._set("antipad_mm", v))
        self.start = ChoiceBox(tooltip="The top layer the via reaches")
        self.start.committed.connect(lambda v: self._set("start_layer", None if v == TOP else v))
        self.stop = ChoiceBox(tooltip="The bottom layer the via reaches")
        self.stop.committed.connect(lambda v: self._set("stop_layer", None if v == BOTTOM else v))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Layer", "Pad ⌀ (µm)", "Antipad ⌀ (µm)"])
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemChanged.connect(self._cell_edited)
        self.reset_row = QPushButton("Use the padstack's own")
        self.reset_row.setToolTip("The selected layer takes the padstack's pad and antipad again")
        self.reset_row.clicked.connect(self._reset_row)
        form = form_layout()
        form.addRow("Pads", self.shape)
        form.addRow(self.shape_hint)
        form.addRow("Pad ⌀", row(self.pad, self.no_pad))
        form.addRow("Antipad ⌀", self.antipad)
        form.addRow("Span", row(self.start, QLabel("to"), self.stop))
        layout.addLayout(form)
        layout.addWidget(self.table)
        layout.addWidget(row(self.reset_row))
        self.pads_card = frame

        # ----------------------------------------------------- cross-section
        frame, layout = card("Cross-section", "The via this padstack makes, through the stackup.")
        self.section = CrossSection()
        layout.addWidget(self.section)
        self.section_card = frame

        self.empty = hint("No padstacks yet. Import a board with a drill file, or make one with New padstack.")
        self.setWidget(centred_column([self.empty, self.head_card, self.drill_card, self.pads_card,
                                       self.section_card]))
        session.projectChanged.connect(lambda: QTimer.singleShot(0, self.refresh))
        self.refresh()

    # ---------------------------------------------------------------- edits

    def _set(self, field: str, value) -> None:
        name = self.name
        if name is None or self._loading:
            return
        if field == "name":
            value = str(value).strip()
            if value == name:
                return
        guarded(lambda: self.session.push(cmd.SetPadstackField(self.session, name, field, value)))
        if field == "name" and any(ps.name == value for ps in self.session.project.padstacks):
            self.name = value
            self.padstackShown.emit(value)

    def _cell_edited(self, item: QTableWidgetItem) -> None:
        if self._loading or self.name is None or item.column() == 0:
            return
        layer = self.table.item(item.row(), 0).data(Qt.UserRole)
        ps = pstk.padstack_map(self.session.project).get(self.name)
        if ps is None or layer is None:
            return
        pad, antipad = pstk.layer_pad(ps, layer)
        try:
            if item.column() == 1:
                pad = _parse_um(item.text())
            else:
                antipad = _parse_um(item.text()) or 0.0
        except ValueError:
            self.refresh()
            return
        name = self.name
        guarded(lambda: self.session.push(cmd.SetPadstackLayer(self.session, name, layer, pad, antipad)))

    def _reset_row(self) -> None:
        r = self.table.currentRow()
        if self.name is None or r < 0:
            return
        layer = self.table.item(r, 0).data(Qt.UserRole)
        name = self.name
        guarded(lambda: self.session.push(cmd.SetPadstackLayer(self.session, name, layer, None, 0.0,
                                                               same_as_default=True)))

    # ---------------------------------------------------------------- showing

    def set_layer_colors(self, layers: List[Dict]) -> None:
        self.colors = {layer["name"].lower(): layer["color"] for layer in layers}
        self.refresh()

    def show_padstack(self, name: Optional[str]) -> None:
        self.name = name
        self.refresh()
        if name:
            self.padstackShown.emit(name)

    def refresh(self) -> None:
        p = self.session.project
        names = [ps.name for ps in p.padstacks]
        if self.name not in names:
            self.name = names[0] if names else None
        for widget in (self.head_card, self.drill_card, self.pads_card, self.section_card):
            widget.setVisible(self.name is not None)
        self.empty.setVisible(self.name is None)
        if self.name is None:
            return
        ps = pstk.padstack_map(p)[self.name]
        layers = pstk.layer_names(p)
        used = sum(1 for v in p.vias or [] if v.padstack == ps.name)
        self._loading = True
        try:
            self.chooser.set_options([(n, n) for n in names])
            self.chooser.set_value(ps.name)
            self.count.setText(f"{used} via{'s' if used != 1 else ''}")
            if not self.title_name.hasFocus():
                self.title_name.setText(ps.name)
            self.hole.set_stored(ps.hole_mm)
            self.plated.setChecked(ps.plated)
            self.plating.set_stored(ps.plating_um)
            self.fill.set_stored(ps.fill_epsilon)
            self.plating.setEnabled(ps.plated)
            self.fill.setEnabled(ps.plated)
            self.shape.set_value(ps.shape)
            self.shape_hint.setText(
                "Each via keeps the pad it was imported with (a square or oval pin pad) and moves it with it. A "
                "via placed new, having none, gets a disc of the pad diameter below." if ps.shape == "imported"
                else "Every pad a disc of its layer's diameter.")
            self.pad.set_stored(ps.pad_mm if ps.pad_mm is not None else 0.0)
            self.pad.setEnabled(ps.pad_mm is not None)
            self.no_pad.setChecked(ps.pad_mm is None)
            self.antipad.set_stored(ps.antipad_mm)
            self.start.set_options([(TOP, f"{layers[0]} (top)" if layers else "top")] +
                                   [(n, n) for n in layers[1:]])
            self.start.set_value(ps.start_layer or TOP)
            self.stop.set_options([(n, n) for n in layers[:-1]] +
                                  [(BOTTOM, f"{layers[-1]} (bottom)" if layers else "bottom")])
            self.stop.set_value(ps.stop_layer or BOTTOM)
            start, stop = pstk.span(ps, layers)
            own = {r.layer for r in ps.layers}
            self.table.setRowCount(len(layers))
            for i, name in enumerate(layers):
                pad, antipad = pstk.layer_pad(ps, name)
                spanned = start <= i <= stop
                if ps.shape == "imported" and pad is not None:
                    pad_text = "imported"
                else:
                    pad_text = _um_text(pad)
                cells = [name, pad_text if spanned else "—", (f"{antipad * 1000:g}" if spanned else "—")]
                for column, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if column == 0:
                        item.setData(Qt.UserRole, name)
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        if name in own:
                            font = item.font()
                            font.setBold(True)
                            item.setFont(font)
                            item.setToolTip("This layer has its own pad and antipad")
                    elif not spanned:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                        item.setToolTip("Outside the padstack's span")
                    self.table.setItem(i, column, item)
            self.table.setFixedHeight(28 + 22 * max(len(layers), 1))
            self.section.show_padstack(p, ps, self.colors)
        finally:
            self._loading = False
