"""The property editor for one field-dump box."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QLabel, QPushButton

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.views.modeler import units as unit_math
from EMShrimulator.app.views.modeler.editor_base import NameEdit, PropertyEditor
from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit, ExpressionVector
from EMShrimulator.app.widgets.form import ChoiceBox, hint
from EMShrimulator.core.em.model import DumpKind, FieldDump

DUMP_LABELS = [
    (DumpKind.E_TIME.value, "E field, time"),
    (DumpKind.H_TIME.value, "H field, time"),
    (DumpKind.E_FREQ.value, "E field, frequency"),
    (DumpKind.H_FREQ.value, "H field, frequency"),
    (DumpKind.CURRENT_TIME.value, "Current, time"),
    (DumpKind.SAR_LOCAL.value, "Local SAR"),
]


class DumpEditor(PropertyEditor):
    def __init__(self, session, parent=None):
        super().__init__(session, "Field dump", parent)
        self.dump_id: Optional[str] = None
        self.index = -1
        self.outer.insertWidget(1, hint("A dump box records the field inside it during the run — time-domain "
                                        "dumps land as VTK for ParaView, frequency-domain ones as HDF5."))
        self.name = NameEdit(lambda v: self.set_field("name", v))
        self.kind = ChoiceBox(DUMP_LABELS)
        self.kind.committed.connect(lambda v: self.set_field("kind", DumpKind(v)))
        self.p1 = ExpressionVector(session, path=lambda: self._sub("p1"))
        self.p1.committed.connect(lambda v: self.set_field("p1", v))
        self.p2 = ExpressionVector(session, path=lambda: self._sub("p2"))
        self.p2.committed.connect(lambda v: self.set_field("p2", v))
        self.frequency = ExpressionEdit(session, "frequency", step=0.1, path=lambda: self._sub("frequencies.0"))
        self.frequency.committed.connect(lambda v: self.set_field("frequencies", [v]))
        self.delete = QPushButton("Delete dump")
        self.delete.setProperty("role", "danger")
        self.delete.clicked.connect(self._delete)
        self.labels = [QLabel(), QLabel(), QLabel()]
        self.form.addRow("Name", self.name)
        self.form.addRow("Type", self.kind)
        self.form.addRow(self.labels[0], self.p1)
        self.form.addRow(self.labels[1], self.p2)
        self.form.addRow(self.labels[2], self.frequency)
        self.form.addRow("", self.delete)

    def _sub(self, field: str) -> Optional[str]:
        return f"dumps.{self.index}.{field}" if self.index >= 0 else None

    def set_field(self, field: str, value) -> None:
        dump_id = self.dump_id
        self.push(lambda: cmd.SetDumpField(self.session, dump_id, field, value))

    def _delete(self) -> None:
        dump_id = self.dump_id
        self.push(lambda: cmd.DeleteDump(self.session, dump_id))

    def load(self, dump: FieldDump, index: int) -> None:
        self.dump_id = dump.id
        self.index = index
        units = self.session.project.display_units
        self.title.setText(f"Field dump · {dump.name}")
        self.name.set_value(dump.name)
        self.kind.set_value(dump.kind.value)
        self.labels[0].setText(f"From ({unit_math.label('length', units)})")
        self.labels[1].setText(f"To ({unit_math.label('length', units)})")
        self.labels[2].setText(f"Frequency ({unit_math.label('frequency', units)})")
        self.p1.set_value(dump.p1)
        self.p2.set_value(dump.p2)
        self.frequency.set_value(dump.frequencies[0] if dump.frequencies else 0.0)
        self.form.setRowVisible(self.frequency, dump.kind.value.endswith("freq"))
