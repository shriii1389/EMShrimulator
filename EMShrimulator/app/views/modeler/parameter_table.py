"""The design-variable table, and the reference of what an expression may use."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableView,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.models.parameter_model import ParameterTableModel
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.modeler.define_variable import (
    define_missing,
    missing_names,
    push_with_definitions,
    quantity_of_unit,
)
from EMShrimulator.app.widgets.form import hint, row
from EMShrimulator.core.services import modeler as modeler_service

WORKED_EXAMPLE = """Parameters
  f0     = 2.45GHz
  er     = 3.66
  h      = 62mil                      -> 1.5748 mm
  Wp     = patch_width(f0, er)        -> 40.08 mm
  Lp     = patch_length(f0, er, h)    -> 31.53 mm
  inset  = patch_inset(f0, er, h, 50) -> 11.41 mm

Patch object, start point and size
  start  = -Wp/2, -Lp/2, h
  size   = Wp, Lp, 0

Feed port, following the inset
  start  = 0, -Lp/2 + inset, 0
  size   = 0, 0, h

Wilkinson divider on the same board
  zb     = wilkinson_z(50)            -> 70.71 ohm
  wb     = microstrip_width(zb, er, h)
  lb     = quarter_wave(f0, microstrip_eeff(wb, er, h))"""


class FunctionHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("What you can type into a dimension")
        self.resize(700, 640)
        ref = modeler_service.function_reference()
        design = [f for f in ref["functions"] if f["group"] == "design"]
        maths = [f["name"] for f in ref["functions"] if f["group"] == "math"]
        html = [
            "<p>Stored values are millimetres, hertz and degrees. Any literal may carry a unit "
            "(10mil, 2.45GHz, 0.5in, 90deg) and converts to that base.</p>",
            "<h3>Units</h3><p>A bare number is read in the project's display unit. Give a literal its own suffix "
            "and that always wins, so <code>2mm + 40mil</code> is a good length whatever the project is set to.</p>",
            f"<p><code>{' · '.join(ref['suffixes'])}</code></p>",
            "<h3>RF design formulas</h3><p>The syntheses you would otherwise run in a side calculator. They "
            "return millimetres, so the result drops straight into a dimension.</p><ul>",
            *[f"<li><code>{f['signature']}</code></li>" for f in design],
            "</ul><h3>Maths</h3><p>Trigonometry takes and returns degrees, matching every angle field.</p>",
            f"<p><code>{' · '.join(maths)}</code></p><h3>Constants</h3><table>",
            *[f"<tr><td><code>{c['name']}</code></td><td>&nbsp;{c['value']:.8g}</td></tr>" for c in ref["constants"]],
            "</table><h3>A worked example</h3>",
            f"<pre>{WORKED_EXAMPLE}</pre>",
        ]
        browser = QTextBrowser()
        browser.setHtml("".join(html))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(browser, 1)
        layout.addWidget(buttons)


class ParameterTable(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.model = ParameterTableModel(session, self)
        self.model.expressionEntered.connect(self._expression_entered)

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                   | QAbstractItemView.AnyKeyPressed)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        # Sized once per refresh (ResizeToContents is slow); the description takes the spare width.
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setMinimumHeight(150)

        add = QPushButton("+ Variable")
        add.clicked.connect(self._add)
        remove = QPushButton("Delete")
        remove.setProperty("role", "danger")
        remove.clicked.connect(self._delete)
        functions = QPushButton("ƒ Functions")
        functions.setToolTip("Functions, constants and units you can use in any dimension")
        functions.clicked.connect(lambda: FunctionHelpDialog(self).exec())

        self.empty = hint("No variables yet. Add one, or just type a new name into any dimension — "
                          "Wp/2, patch_l + 2mm — and you will be asked for its value.")
        self.field_errors = QLabel()
        self.field_errors.setProperty("role", "error")
        self.field_errors.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(row(add, remove, functions))
        layout.addWidget(self.empty)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.field_errors)

        session.projectChanged.connect(lambda: QTimer.singleShot(0, self.refresh))
        self.refresh()
        self.table.resizeColumnsToContents()

    def refresh(self) -> None:
        if self.model.refresh():
            self.table.resizeColumnsToContents()
        project = self.session.project
        self.empty.setVisible(not project.parameters)
        names = {p.name for p in project.parameters}
        field_errors = [e for e in self.session.resolution.errors if e["target"] not in names]
        if field_errors:
            lines = [f"<b>{e.get('label') or e['target']}</b> — {e['message']}" for e in field_errors[:8]]
            more = f"<br>…and {len(field_errors) - 8} more" if len(field_errors) > 8 else ""
            self.field_errors.setText("Fields that will not evaluate:<br>" + "<br>".join(lines) + more)
        self.field_errors.setVisible(bool(field_errors))

    def select_name(self, name: str) -> None:
        """Highlight one variable's row, as the project tree asks."""
        for row_index in range(self.model.rowCount()):
            if self.model.name_at(row_index) == name:
                self.table.selectRow(row_index)
                self.table.scrollTo(self.model.index(row_index, 0))
                return

    def add(self) -> None:
        self._add()

    def _add(self) -> None:
        name = guarded(lambda: self.session.push(cmd.AddParameter(self.session)))
        if name:
            QTimer.singleShot(0, lambda: self.table.edit(self.model.index(self.model.rowCount() - 1, 0)))

    def _expression_entered(self, name: str, text: str) -> None:
        if not missing_names(self.session, text, owner=name):
            self._set_expression(name, text)
            return
        # After the cell editor has closed, so the dialog doesn't close it twice.
        QTimer.singleShot(0, lambda: self._define_then_set(name, text))

    def _define_then_set(self, name: str, text: str) -> None:
        param = next((p for p in self.session.project.parameters if p.name == name), None)
        if param is None:
            return
        definitions = define_missing(self.session, text, quantity=quantity_of_unit(param.unit), owner=name,
                                     parent=self.window())
        push_with_definitions(self.session, definitions or [], lambda: self._set_expression(name, text))

    def _set_expression(self, name: str, text: str) -> None:
        s = self.session
        guarded(lambda: s.push(cmd.SetParameterField(s, name, "expression", text)))

    def _delete(self) -> None:
        index = self.table.currentIndex()
        name = self.model.name_at(index.row()) if index.isValid() else ""
        if name:
            guarded(lambda: self.session.push(cmd.DeleteParameter(self.session, name)))
