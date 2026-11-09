"""Parameter sweep editor, previewing points with the same expand() the worker uses."""
from __future__ import annotations

from typing import Callable, List

from PySide6.QtCore import Signal, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.widgets.form import ChoiceBox, form_layout, hint, row
from EMShrimulator.app.widgets.unit_spinbox import IntSpinBox, UnitSpinBox
from EMShrimulator.core.em import sweep as sweep_math
from EMShrimulator.core.em.model import SweepCombine, SweepMode


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()


class SweepCard(QFrame):
    runRequested = Signal(str)

    def __init__(self, session, index: int, parent=None):
        super().__init__(parent)
        self.session = session
        self.index = index
        self.setProperty("role", "card")
        self._variable_count = -1

        self.name = QLineEdit()
        self.name.editingFinished.connect(self._rename)
        self.combine = ChoiceBox([(SweepCombine.PRODUCT.value, "Every combination"),
                                  (SweepCombine.ZIP.value, "Move together")])
        self.combine.committed.connect(lambda v: self.edit("Set sweep combination",
                                                           lambda s: setattr(s, "combine", SweepCombine(v))))
        form = form_layout()
        form.addRow("Name", self.name)
        form.addRow("Combine", self.combine)

        self.variables = QVBoxLayout()
        self.variables.setSpacing(3)
        add_variable = QPushButton("+ Variable")
        add_variable.clicked.connect(lambda: guarded(lambda: session.push(cmd.AddSweepVariable(session, index))))
        delete = QPushButton("Delete sweep")
        delete.setProperty("role", "danger")
        delete.clicked.connect(lambda: guarded(lambda: session.push(cmd.DeleteSweep(session, index))))
        self.points = QLabel()
        self.points.setWordWrap(True)
        self.points.setProperty("role", "note")
        self.error = QLabel()
        self.error.setProperty("role", "error")
        self.error.setWordWrap(True)
        self.run = QPushButton("▶ Run")
        self.run.setProperty("role", "primary")
        self.run.clicked.connect(lambda: self.runRequested.emit(self.session.project.sweeps[self.index].name))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(form)
        layout.addLayout(self.variables)
        layout.addWidget(row(add_variable, delete))
        layout.addWidget(self.error)
        layout.addWidget(self.points)
        layout.addWidget(self.run)

    def edit(self, text: str, mutate: Callable, merge_field=None) -> None:
        guarded(lambda: self.session.push(cmd.EditSweep(self.session, self.index, text, mutate, merge_field)))

    def _rename(self) -> None:
        text = self.name.text().strip()
        current = self.session.project.sweeps[self.index].name
        if not text:
            self.name.setText(current)
        elif text != current:
            self.edit("Rename sweep", lambda s: setattr(s, "name", text))

    def load(self, running: bool) -> None:
        sweep = self.session.project.sweeps[self.index]
        if not self.name.hasFocus():
            self.name.setText(sweep.name)
        self.combine.set_value(sweep.combine.value)
        if len(sweep.variables) != self._variable_count:
            self._variable_count = len(sweep.variables)
            _clear(self.variables)
            self._rows: List[dict] = [self._variable_row(i) for i in range(len(sweep.variables))]
        names = [(p.name, p.name) for p in self.session.project.parameters]
        for i, variable in enumerate(sweep.variables):
            widgets = self._rows[i]
            widgets["parameter"].set_options(names)
            widgets["parameter"].set_value(variable.parameter)
            widgets["mode"].set_value(variable.mode.value)
            listed = variable.mode == SweepMode.LIST
            widgets["range"].setVisible(not listed)
            widgets["list"].setVisible(listed)
            widgets["start"].set_stored(variable.start)
            widgets["stop"].set_stored(variable.stop)
            widgets["count"].set_stored(variable.points)
            if not widgets["list"].hasFocus():
                widgets["list"].setText(", ".join(f"{v:g}" for v in variable.values))

        description = sweep_math.describe(sweep)
        if description["ok"]:
            labels = [p["label"] for p in description["points"][:12]]
            more = f"  …and {description['count'] - 12} more" if description["count"] > 12 else ""
            self.points.setText("  ·  ".join(labels) + more)
            self.error.setVisible(False)
            self.run.setText(f"▶ Run {description['count']} point{'s' if description['count'] != 1 else ''}")
            self.run.setEnabled(not running and description["count"] > 0)
        else:
            self.error.setText(description["error"])
            self.error.setVisible(True)
            self.points.setText("")
            self.run.setText("▶ Run")
            self.run.setEnabled(False)

    def _variable_row(self, i: int) -> dict:
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)
        parameter = ChoiceBox()
        parameter.committed.connect(lambda v: self.edit("Set swept variable",
                                                        lambda s: setattr(s.variables[i], "parameter", v)))
        mode = ChoiceBox([(SweepMode.LINEAR.value, "Linear"), (SweepMode.LOG.value, "Log"),
                          (SweepMode.LIST.value, "List")])
        mode.committed.connect(lambda v: self.edit("Set sweep spacing",
                                                   lambda s: setattr(s.variables[i], "mode", SweepMode(v))))
        remove = QPushButton("×")
        remove.setProperty("role", "flat")
        remove.setFixedWidth(24)
        remove.clicked.connect(lambda: self.edit("Remove swept variable", lambda s: s.variables.pop(i)))
        top = QHBoxLayout()
        top.addWidget(parameter, 1)
        top.addWidget(mode)
        top.addWidget(remove)

        start = UnitSpinBox(decimals=6)
        stop = UnitSpinBox(decimals=6)
        count = IntSpinBox(minimum=1, maximum=sweep_math.MAX_POINTS)
        start.committed.connect(lambda v: self.edit("Set sweep start", lambda s: setattr(s.variables[i], "start", v), "start"))
        stop.committed.connect(lambda v: self.edit("Set sweep stop", lambda s: setattr(s.variables[i], "stop", v), "stop"))
        count.committed.connect(lambda v: self.edit("Set sweep points", lambda s: setattr(s.variables[i], "points", int(v)), "points"))
        range_row = QWidget()
        r = QHBoxLayout(range_row)
        r.setContentsMargins(0, 0, 0, 0)
        for text, widget in (("from", start), ("to", stop), ("in", count)):
            label = QLabel(text)
            label.setProperty("role", "note")
            r.addWidget(label)
            r.addWidget(widget, 1)

        values = QLineEdit()
        values.setPlaceholderText("1.2, 1.6, 2.0")
        values.setToolTip("Comma-separated values, in the variable's own units")

        def commit_list() -> None:
            parsed = []
            for token in values.text().replace(";", ",").replace(" ", ",").split(","):
                try:
                    parsed.append(float(token))
                except ValueError:
                    continue
            self.edit("Set sweep values", lambda s: setattr(s.variables[i], "values", parsed))

        values.editingFinished.connect(commit_list)
        layout.addLayout(top)
        layout.addWidget(range_row)
        layout.addWidget(values)
        self.variables.addWidget(host)
        return {"parameter": parameter, "mode": mode, "start": start, "stop": stop, "count": count,
                "range": range_row, "list": values}


class SweepEditor(QWidget):
    runRequested = Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.running = False
        self._cards: List[SweepCard] = []

        self.add = QPushButton("+ Sweep")
        self.add.clicked.connect(self._add)
        self.intro = hint()
        self.cards = QVBoxLayout()
        self.cards.setSpacing(6)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(row(self.add))
        layout.addWidget(self.intro)
        layout.addLayout(self.cards)
        layout.addStretch(1)

        session.projectChanged.connect(lambda: QTimer.singleShot(0, self.refresh))
        self.refresh()

    def set_running(self, running: bool) -> None:
        self.running = running
        self.refresh()

    def refresh(self) -> None:
        project = self.session.project
        self.add.setEnabled(bool(project.parameters))
        self.intro.setText(
            "Define a variable first — a sweep varies a parameter, not a raw dimension." if not project.parameters else
            "Each point overrides its variables, re-derives the whole design from them, and runs a full solve. "
            "Results appear on the Results tab's Sweep view as each point finishes.")
        if len(self._cards) != len(project.sweeps):
            _clear(self.cards)
            self._cards = []
            for i in range(len(project.sweeps)):
                card = SweepCard(self.session, i)
                card.runRequested.connect(self.runRequested)
                self.cards.addWidget(card)
                self._cards.append(card)
        for card in self._cards:
            card.load(self.running)

    def _add(self) -> None:
        guarded(lambda: self.session.push(cmd.AddSweep(self.session)))
