"""The layout stackup as an editable table, top layer first."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from EMShrimulator.app import commands as cmd
from EMShrimulator.app import theme
from EMShrimulator.app.notify import guarded

COLUMNS = ["Layer", "Type", "Thickness (mm)", "εr", "tan δ"]
FIELDS = ["name", "type", "thickness_mm", "epsilon", "loss_tangent"]
EDITABLE_TYPES = ["copper", "core", "prepreg", "other"]


def _cell(layer, field: str, copper: bool):
    """(display text, edit value) of one cell."""
    value = getattr(layer, field)
    if copper and field in ("epsilon", "loss_tangent"):
        return "—", None
    if value is None:
        return "", None
    return (f"{value:g}" if isinstance(value, float) else str(value)), value


class StackupTableModel(QAbstractTableModel):
    """Cells are computed once per refresh."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._layers, self._rows, self._copper_color = self._build()
        theme.manager().changed.connect(self._theme_changed)

    def _theme_changed(self, _name: str) -> None:
        self.refresh()  # the cached colour belongs to the old theme

    def _build(self):
        layers = list(self.session.project.stackup)
        rows = []
        for layer in layers:
            copper = layer.type == "copper"
            rows.append((copper, [_cell(layer, field, copper) for field in FIELDS]))
        return layers, rows, QColor(theme.colors()["warn"])

    def refresh(self) -> bool:
        """Reload from the session; False, with no reset, when no cell would change."""
        built = self._build()
        if built[1:] == (self._rows, self._copper_color):
            self._layers = built[0]
            return False
        self.beginResetModel()
        self._layers, self._rows, self._copper_color = built
        self.endResetModel()
        return True

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._layers)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        copper, cells = self._rows[index.row()]
        if role == Qt.DisplayRole:
            return cells[index.column()][0]
        if role == Qt.EditRole:
            return cells[index.column()][1]
        if role == Qt.ForegroundRole and copper and index.column() == 0:
            return self._copper_color
        return None

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        copper, _cells = self._rows[index.row()]
        if not (copper and FIELDS[index.column()] in ("epsilon", "loss_tangent")):
            flags |= Qt.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:  # noqa: N802
        if role != Qt.EditRole or not index.isValid():
            return False
        field = FIELDS[index.column()]
        row = index.row()
        if field in ("thickness_mm", "epsilon", "loss_tangent"):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return False
        else:
            parsed = str(value).strip()
            if not parsed:
                return False
        if getattr(self._layers[row], field) == parsed:
            return False
        s = self.session
        guarded(lambda: s.push(cmd.SetStackupField(s, row, field, parsed)))
        return True
