"""Design-variable table model. Renames go through RenameParameter."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase

from EMShrimulator.app import commands as cmd
from EMShrimulator.app import theme
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.widgets.form import fmt_eng, tidy

COLUMNS = ["Name", "Expression", "Value", "Unit", "Description"]
NAME, EXPRESSION, VALUE, UNIT, DESCRIPTION = range(5)


def format_value(value: float, unit: str) -> str:
    if abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-3):
        return fmt_eng(value, unit)
    text = tidy(float(f"{value:.6g}"))
    return f"{text} {unit}".strip()


class ParameterTableModel(QAbstractTableModel):
    """Rows are computed once per refresh."""

    # (name, expression), committed after defining any new variables it reads.
    expressionEntered = Signal(str, str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self._mono.setStyleHint(QFont.Monospace)
        self._params, self._rows, self._danger, self._dim = self._build()
        theme.manager().changed.connect(self._theme_changed)

    def _build(self):
        params = list(self.session.project.parameters)
        resolution = self.session.resolution
        errors = {}
        for e in resolution.errors:
            errors.setdefault(e["target"], e)  # the first, as Resolution.error_for answers
        colors = theme.colors()
        rows = []
        for param in params:
            error = errors.get(param.name)
            value = resolution.values.get(param.name)
            shown = "!" if error else ("—" if value is None else format_value(value, param.unit))
            rows.append((
                (param.name, param.expression, shown, param.unit, param.description),
                error,
                error["message"] if error else (param.description or None),
            ))
        return params, rows, QColor(colors["danger"]), QColor(colors["text_dim"])

    def refresh(self) -> bool:
        """Reload from the session; False, with no reset, when no cell would change."""
        built = self._build()
        if built[1:] == (self._rows, self._danger, self._dim):
            self._params = built[0]
            return False
        self.beginResetModel()
        self._params, self._rows, self._danger, self._dim = built
        self.endResetModel()
        return True

    def _theme_changed(self, _name: str) -> None:
        self.refresh()  # the cached colours belong to the old theme

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._params)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLUMNS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        texts, error, tooltip = self._rows[index.row()]
        column = index.column()
        if role == Qt.DisplayRole:
            return texts[column]
        if role == Qt.EditRole:
            return None if (column == VALUE and error) else texts[column]
        if role == Qt.FontRole and column in (NAME, EXPRESSION, VALUE):
            return self._mono
        if role == Qt.ForegroundRole:
            if error and column in (EXPRESSION, VALUE):
                return self._danger
            if column == VALUE:
                return self._dim
        if role == Qt.ToolTipRole:
            return tooltip
        return None

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.NoItemFlags
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() != VALUE:
            flags |= Qt.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:  # noqa: N802
        if role != Qt.EditRole or not index.isValid():
            return False
        param = self._params[index.row()]
        text = str(value).strip()
        column = index.column()
        s = self.session
        if column == NAME and text and text != param.name:
            guarded(lambda: s.push(cmd.RenameParameter(s, param.name, text)))
        elif column == EXPRESSION and text and text != param.expression:
            self.expressionEntered.emit(param.name, text)
        elif column == UNIT and text != param.unit:
            guarded(lambda: s.push(cmd.SetParameterField(s, param.name, "unit", text)))
        elif column == DESCRIPTION and text != param.description:
            guarded(lambda: s.push(cmd.SetParameterField(s, param.name, "description", text)))
        else:
            return False
        return True

    def name_at(self, row: int) -> str:
        return self._params[row].name if 0 <= row < len(self._params) else ""
