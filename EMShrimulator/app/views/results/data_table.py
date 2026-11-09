"""A report as a data table: one row per point, one column per quantity."""
from __future__ import annotations

from typing import Any, List

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication, QHeaderView, QMenu, QTableView


class _Model(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.columns: List[str] = []
        self.rows: List[List[str]] = []

    def set_table(self, columns: List[str], rows: List[List[str]]) -> None:
        if columns == self.columns and rows == self.rows:
            return
        self.beginResetModel()
        self.columns, self.rows = list(columns), rows
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt override
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt override
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row = self.rows[index.row()]
            return row[index.column()] if index.column() < len(row) else ""
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section: int, orientation, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.columns[section] if section < len(self.columns) else ""
        return section + 1


class DataTable(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_ = _Model(self)
        self.setModel(self.model_)
        self.setObjectName("report-table")
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(False)
        self.verticalHeader().setDefaultSectionSize(22)

    def set_table(self, columns: List[str], rows: List[List[str]]) -> None:
        self.model_.set_table(columns, rows)
        self.resizeColumnsToContents()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override
        menu = QMenu(self)
        menu.addAction("Copy selection", self.copy_selection)
        menu.addAction("Copy table", self.copy_all)
        menu.exec(event.globalPos())

    def copy_selection(self) -> None:
        indexes = sorted(self.selectedIndexes(), key=lambda i: (i.row(), i.column()))
        if not indexes:
            return
        rows = {}
        for index in indexes:
            rows.setdefault(index.row(), []).append(str(self.model_.data(index) or ""))
        QApplication.clipboard().setText("\n".join("\t".join(cells) for cells in rows.values()))

    def copy_all(self) -> None:
        lines = ["\t".join(self.model_.columns)] + ["\t".join(row) for row in self.model_.rows]
        QApplication.clipboard().setText("\n".join(lines))
