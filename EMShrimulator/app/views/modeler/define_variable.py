"""Define a variable the moment a field first reads it, asking for its value and unit.

Declining any new name defines none of them.
"""
from __future__ import annotations

import html
from typing import Callable, Dict, Iterable, List, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.models.parameter_model import format_value
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.modeler.expression_edit import BARE_NUMBER
from EMShrimulator.app.widgets.form import form_layout, hint
from EMShrimulator.core.em.edits import IDENTIFIER
from EMShrimulator.core.em.model import Parameter
from EMShrimulator.core.em.params import (
    CONSTANTS,
    FREQUENCY_UNITS,
    LENGTH_UNITS,
    ExpressionError,
    check_syntax,
    evaluate,
    referenced_names,
    solve_parameters,
    undefined_names,
)

# Unit menu by quantity: (suffix, label). The first is the base unit.
UNIT_GROUPS = {
    "length": [("mm", "mm"), ("um", "µm"), ("cm", "cm"), ("m", "m"), ("mil", "mil"), ("in", "in")],
    "frequency": [("Hz", "Hz"), ("kHz", "kHz"), ("MHz", "MHz"), ("GHz", "GHz")],
    "angle": [("deg", "deg"), ("rad", "rad")],
    "impedance": [("ohm", "Ω")],
    "plain": [("", "no unit")],
}
QUANTITY_OF = {suffix: quantity for quantity, units in UNIT_GROUPS.items() for suffix, _ in units}
BASE_UNIT = {quantity: units[0][0] for quantity, units in UNIT_GROUPS.items()}


def quantity_of_unit(unit: str) -> str:
    """The quantity a variable's Unit column names: `mm` is a length, `GHz` a frequency."""
    unit = (unit or "").strip()
    if unit in LENGTH_UNITS:
        return "length"
    if unit.lower() in {u.lower() for u in FREQUENCY_UNITS}:
        return "frequency"
    if unit in ("deg", "rad", "°"):
        return "angle"
    if unit.lower() in ("ohm", "Ω"):
        return "impedance"
    return "plain"


def default_unit(quantity: str, display_units) -> str:
    """The unit offered first for a new variable read by a field of this quantity."""
    if quantity == "length":
        return display_units.length
    if quantity == "frequency":
        return display_units.frequency
    return BASE_UNIT.get(quantity, "")


def stored_expression(text: str, suffix: str) -> str:
    """The stored value: a plain number gets the chosen unit, anything else is kept as typed."""
    text = text.strip()
    if not BARE_NUMBER.match(text):
        return text
    number = repr(float(text))
    number = number[:-2] if number.endswith(".0") else number
    return number if suffix == BASE_UNIT[QUANTITY_OF.get(suffix, "plain")] else f"{number}{suffix}"


def _definable(names: Iterable[str]) -> List[str]:
    return [n for n in names if IDENTIFIER.match(n)]


def _reads(start: Iterable[str], expressions: Dict[str, str], target: str) -> bool:
    """Whether any of `start` reads `target`, directly or through other variables."""
    seen: Set[str] = set()
    stack = list(start)
    while stack:
        name = stack.pop()
        if name == target:
            return True
        if name in seen or name not in expressions:
            continue
        seen.add(name)
        stack.extend(referenced_names(expressions[name]))
    return False


class DefineVariableDialog(QDialog):
    """One new variable's value and unit."""

    def __init__(self, session, name: str, pending: Dict[str, Parameter], *, used_in: str, unit: str,
                 ancestors: Iterable[str] = (), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define variable")
        self.setMinimumWidth(400)
        self.session = session
        self.name = name
        self.pending = pending                       # shared by the whole chain of dialogs
        self.ancestors = set(ancestors)              # variables waiting on this one: reading them is a loop

        heading = QLabel("New variable")  # headings are capitalised, which would mangle the name
        heading.setProperty("role", "heading")
        where = "" if used_in == name else f" is used in <code>{html.escape(used_in)}</code> but"
        context = hint(f"<b>{html.escape(name)}</b>{where} is not a variable yet. Give it a value to create it.")
        context.setTextFormat(Qt.RichText)

        self.value = QLineEdit()
        self.value.setPlaceholderText("40, 62mil or 2*Wp")
        self.value.textChanged.connect(self._changed)
        self.unit = QComboBox()
        for i, units in enumerate(UNIT_GROUPS.values()):
            if i:
                self.unit.insertSeparator(self.unit.count())
            for suffix, label in units:
                self.unit.addItem(label, suffix)
        index = self.unit.findData(unit)
        self.unit.setCurrentIndex(index if index >= 0 else self.unit.findData(""))
        self.unit.currentIndexChanged.connect(self._changed)
        self.unit.setToolTip("A plain number is read in this unit. An expression is worked in mm, Hz and "
                             "degrees, so give its literals their own unit: 2*Wp + 10mil.")
        value_row = QHBoxLayout()
        value_row.setSpacing(4)
        value_row.addWidget(self.value, 1)
        value_row.addWidget(self.unit)
        form = form_layout()
        form.addRow("Value", value_row)

        self.note = QLabel()
        self.note.setProperty("role", "note")
        self.note.setWordWrap(True)
        self.error = QLabel()
        self.error.setProperty("role", "error")
        self.error.setWordWrap(True)
        self.error.setVisible(False)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Define")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(context)
        layout.addLayout(form)
        layout.addWidget(self.note)
        layout.addWidget(self.error)
        layout.addWidget(buttons)

        self._values = self._solve()
        self._changed()

    # ------------------------------------------------------------ state

    def _parameters(self) -> List[Parameter]:
        return [*self.session.project.parameters, *self.pending.values()]

    def _solve(self) -> Dict[str, float]:
        return solve_parameters(self._parameters())[0]

    def _known(self) -> Set[str]:
        return {p.name for p in self._parameters()} | self.ancestors | {self.name}

    def suffix(self) -> str:
        return self.unit.currentData() or ""

    def expression(self) -> str:
        return stored_expression(self.value.text(), self.suffix())

    def _changed(self, *_args) -> None:
        self.error.setVisible(False)
        text = self.value.text().strip()
        if not text:
            self.note.setText("A number in the unit beside it, or an expression such as 2*Wp + 10mil.")
            return
        expression = self.expression()
        new = _definable(undefined_names(expression, self._known()))
        if new:
            them = "it" if len(new) == 1 else "them"
            self.note.setText(f"{', '.join(new)} {'is' if len(new) == 1 else 'are'} new as well; "
                              f"you will define {them} next.")
            return
        base = BASE_UNIT[QUANTITY_OF.get(self.suffix(), "plain")]
        if BARE_NUMBER.match(text) and self.suffix() == base:
            self.note.setText("")  # a number in the base unit: nothing to confirm
            return
        try:
            value = evaluate(expression, self._values)
        except ExpressionError:
            self.note.setText("")  # reported on Define, not while it is still being typed
            return
        self.note.setText(f"= {format_value(value, base)}")

    def _fail(self, message: str) -> None:
        self.error.setText(message)
        self.error.setVisible(True)
        self.value.setFocus()
        self.value.selectAll()

    # ------------------------------------------------------------ accept

    def accept(self) -> None:  # noqa: D401 - Qt override
        if not self.value.text().strip():
            self._fail(f"Give {self.name} a value, or Cancel to leave it undefined.")
            return
        expression = self.expression()
        try:
            check_syntax(expression)
        except ExpressionError as exc:
            self._fail(str(exc))
            return
        reads = referenced_names(expression)
        if self.name in reads:
            self._fail(f"{self.name} cannot be worked out from itself.")
            return
        loop = next((n for n in reads if n in self.ancestors), None)
        if loop:
            self._fail(f"{self.name} cannot read {loop}: {loop} is being defined from {self.name}, "
                       "so each would wait for the other.")
            return
        bad = [n for n in undefined_names(expression, self._known()) if not IDENTIFIER.match(n)]
        if bad:
            self._fail(f"'{bad[0]}' cannot be a variable: a name uses letters, digits and underscore, "
                       "starting with a letter.")
            return

        for name in _definable(undefined_names(expression, self._known())):
            if name in self.pending:
                continue  # defined while answering an earlier one
            child = DefineVariableDialog(self.session, name, self.pending, used_in=f"{self.name} = {expression}",
                                         unit=self.suffix(), ancestors=self.ancestors | {self.name}, parent=self)
            if child.exec() != QDialog.Accepted:
                self.reject()  # declining one declines the whole chain
                return
        self._values = self._solve()

        expressions = {p.name: p.expression for p in self._parameters()}
        if _reads(reads, expressions, self.name):
            self._fail(f"{self.name} would read itself through another variable.")
            return
        if all(n in self._values or n in CONSTANTS for n in reads):
            try:
                evaluate(expression, self._values)
            except ExpressionError as exc:
                self._fail(str(exc))
                return
        # Otherwise an existing variable it reads is broken and shows its own error.

        self.pending[self.name] = Parameter(name=self.name, expression=expression,
                                            unit=BASE_UNIT[QUANTITY_OF.get(self.suffix(), "plain")])
        super().accept()


# ---------------------------------------------------------------- the flow

def missing_names(session, expression, owner: Optional[str] = None) -> List[str]:
    """The new variables an entry reads that could be defined on the spot."""
    parameters = getattr(session.project, "parameters", None)
    if parameters is None or not isinstance(expression, str):
        return []
    known = {p.name for p in parameters} | ({owner} if owner else set())
    return _definable(undefined_names(expression, known))


def _needed(expression: str, pending: Dict[str, Parameter]) -> List[Parameter]:
    """The definitions the entry still reads, in creation order."""
    reached: Set[str] = set()
    stack = referenced_names(expression)
    while stack:
        name = stack.pop()
        if name in pending and name not in reached:
            reached.add(name)
            stack.extend(referenced_names(pending[name].expression))
    return [p for name, p in pending.items() if name in reached]


def define_missing(session, expression, *, quantity: str = "length", owner: Optional[str] = None,
                   parent=None) -> Optional[List[Parameter]]:
    """Ask for each new variable `expression` reads; `owner` is the variable it belongs to, if any.

    Returns the variables to add, or None if one was declined.
    """
    missing = missing_names(session, expression, owner)
    if not missing:
        return []
    pending: Dict[str, Parameter] = {}
    unit = default_unit(quantity, session.project.display_units)
    used_in = f"{owner} = {expression}" if owner else expression
    parent = parent or QApplication.activeWindow()
    for name in missing:
        if name in pending:
            continue
        dialog = DefineVariableDialog(session, name, pending, used_in=used_in, unit=unit,
                                      ancestors={owner} if owner else (), parent=parent)
        if dialog.exec() != QDialog.Accepted:
            return None
    return _needed(expression, pending)


def push_with_definitions(session, definitions: List[Parameter], edit: Callable[[], object]) -> None:
    """Add the new variables and make the edit that reads them, as one undo step."""
    if not definitions:
        edit()
        return
    stack = session.undo_stack
    stack.beginMacro(f"Define {', '.join(d.name for d in definitions)}")
    try:
        guarded(lambda: session.push(cmd.DefineParameters(session, definitions)))
        edit()
    finally:
        stack.endMacro()
