"""The stackup page: the cross-section to scale and the editable table with presets."""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QStyledItemDelegate,
    QTableView,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.models.stackup_model import EDITABLE_TYPES, StackupTableModel
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.layout.stackup_widget import StackupWidget
from EMShrimulator.app.widgets.form import ChoiceBox, card, centred_column, hint, row
from EMShrimulator.core.services import layout as layout_service


class _TypeDelegate(QStyledItemDelegate):
    """The Type column edits through a combo box of the valid layer types."""

    def createEditor(self, parent, _option, _index):  # noqa: N802 - Qt override
        box = QComboBox(parent)
        box.addItems(EDITABLE_TYPES)
        return box

    def setEditorData(self, editor, index):  # noqa: N802 - Qt override
        value = index.data()
        if value in EDITABLE_TYPES:
            editor.setCurrentText(value)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt override
        model.setData(index, editor.currentText())


class StackupPanel(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        self.setObjectName("setup-page")
        self.board_layers: List[Dict] = []

        self.section = StackupWidget()
        self.section.layerClicked.connect(lambda i: session.select("stackup", i))
        self.summary = hint()
        self.presets = ChoiceBox()
        self.presets.set_options([("", "Apply a preset…")] +
                                 [(p["id"], p["title"]) for p in layout_service.stackup_presets()])
        self.presets.committed.connect(self.apply_preset)

        self.model = StackupTableModel(session, self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setItemDelegateForColumn(1, _TypeDelegate(self.table))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        # Sized once per refresh: ResizeToContents is slow.
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        # The page scrolls, not the table: it is as tall as its rows.
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.clicked.connect(lambda index: session.select("stackup", index.row()))

        add = QPushButton("+ Layer")
        add.clicked.connect(self.add_layer)
        up = QPushButton("↑")
        up.clicked.connect(lambda: self.move_layer(-1))
        down = QPushButton("↓")
        down.clicked.connect(lambda: self.move_layer(1))
        delete = QPushButton("Delete")
        delete.setProperty("role", "danger")
        delete.clicked.connect(self.delete_layer)

        section, layout = card("Cross-section", "To scale, top to bottom. Each copper band takes the colour of the "
                                                "gerber of the same name; a dashed band has none. Click a band to "
                                                "select its row.")
        layout.addWidget(self.section)
        layout.addWidget(self.summary)
        table, layout = card("Layers", "Names must match the copper gerbers: that is how the solver pairs them.")
        layout.addWidget(row(self.presets))
        layout.addWidget(self.table)
        layout.addWidget(row(add, up, down, delete))
        self.setWidget(centred_column([section, table]))

        session.projectChanged.connect(lambda: QTimer.singleShot(0, self.refresh))
        session.selectionChanged.connect(lambda _s: self.refresh())
        self.refresh()
        self.table.resizeColumnsToContents()

    def set_board_layers(self, layers: List[Dict]) -> None:
        self.board_layers = layers
        self.refresh()

    def _selected_index(self) -> Optional[int]:
        selection = self.session.selection
        if selection is not None and selection.kind == "stackup":
            return int(selection.key)
        return None

    def refresh(self) -> None:
        if self.model.refresh():
            self.table.resizeColumnsToContents()
        header = self.table.horizontalHeader().sizeHint().height()
        rows = self.table.verticalHeader().length() if self.model.rowCount() else 38
        self.table.setFixedHeight(header + rows + 2 * self.table.frameWidth() + 2)
        selected = self._selected_index()
        self.section.set_data(self.session.project.stackup, self.board_layers, selected)
        self.summary.setText(self.section.summary())
        if selected is not None:
            self.table.selectRow(selected)
        else:
            self.table.clearSelection()

    def apply_preset(self, preset_id: str) -> None:
        if not preset_id:
            return
        layers = guarded(layout_service.preset_stackup, preset_id)
        if layers:
            title = next(p["title"] for p in layout_service.stackup_presets() if p["id"] == preset_id)
            guarded(lambda: self.session.push(cmd.ApplyStackupPreset(self.session, layers, title)))
        self.presets.set_value("")

    def add_layer(self) -> None:
        """A layer below the selected one, or at the bottom."""
        selected = self._selected_index()
        index = guarded(lambda: self.session.push(cmd.AddStackupLayer(
            self.session, None if selected is None else selected + 1)))
        if index is not None:
            self.session.select("stackup", index)

    def move_layer(self, delta: int) -> None:
        selected = self._selected_index()
        if selected is not None:
            index = guarded(lambda: self.session.push(cmd.MoveStackupLayer(self.session, selected, delta)))
            if index is not None:
                self.session.select("stackup", index)

    def delete_layer(self) -> None:
        selected = self._selected_index()
        if selected is not None:
            self.session.select(None)
            guarded(lambda: self.session.push(cmd.DeleteStackupLayer(self.session, selected)))
