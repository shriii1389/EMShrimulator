"""Which S-parameter traces are drawn: a checkable list with a colour per trace."""
from __future__ import annotations

from typing import Any, Dict, List

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor

from EMShrimulator.app.theme import trace_color
from EMShrimulator.core.em.reports import _SIJ, default_enabled, is_reflection

__all__ = ["TraceListModel", "default_enabled", "is_reflection"]


class TraceListModel(QAbstractListModel):
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._keys: List[str] = []
        self._enabled: Dict[str, bool] = {}

    def set_keys(self, keys: List[str]) -> None:
        keys = sorted(keys)
        # With differential pairs, start with the single-ended terms off.
        mixed = any((m := _SIJ.match(k)) and m.group(1) for k in keys)

        def default(key: str) -> bool:
            match = _SIJ.match(key)
            return default_enabled(key) and not (mixed and match and not match.group(1))

        self.beginResetModel()
        # Keep choices across a reload of the same results.
        self._enabled = {k: self._enabled.get(k, default(k)) for k in keys}
        self._keys = keys
        self.endResetModel()
        self.selectionChanged.emit()

    def keys(self) -> List[str]:
        return list(self._keys)

    def enabled_keys(self) -> List[str]:
        return [k for k in self._keys if self._enabled.get(k)]

    def color_of(self, key: str) -> str:
        return trace_color(self._keys.index(key)) if key in self._keys else "#888888"

    def set_all(self, enabled: bool) -> None:
        for key in self._keys:
            self._enabled[key] = enabled
        if self._keys:
            self.dataChanged.emit(self.index(0), self.index(len(self._keys) - 1))
        self.selectionChanged.emit()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._keys)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        key = self._keys[index.row()]
        if role == Qt.DisplayRole:
            return key
        if role == Qt.CheckStateRole:
            return Qt.Checked if self._enabled.get(key) else Qt.Unchecked
        if role == Qt.DecorationRole:
            return QColor(trace_color(index.row()))
        return None

    def flags(self, index: QModelIndex):
        return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable if index.isValid() else Qt.NoItemFlags

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:  # noqa: N802
        if role != Qt.CheckStateRole or not index.isValid():
            return False
        self._enabled[self._keys[index.row()]] = Qt.CheckState(value) == Qt.Checked
        self.dataChanged.emit(index, index)
        self.selectionChanged.emit()
        return True
