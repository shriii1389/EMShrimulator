"""Combine objects into one boolean solid. Operands are held by id."""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from EMShrimulator.app.widgets.form import ChoiceBox, form_layout, hint
from EMShrimulator.core.em.edits import BOOLEAN_LABELS, MIN_OPERANDS, PRIMITIVE_LABELS
from EMShrimulator.core.em.model import BooleanOp, ModelerProject, PrimitiveKind


class BooleanDialog(QDialog):
    def __init__(self, project: ModelerProject, op: BooleanOp = BooleanOp.UNION,
                 preselect: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine objects")
        self.resize(640, 460)
        self.project = project
        self.picked: List[str] = [preselect] if preselect else []

        self.operation = ChoiceBox([(o.value, label) for o, label in BOOLEAN_LABELS.items()])
        self.operation.set_value(BooleanOp(op).value)
        self.operation.committed.connect(lambda _v: self._refresh())
        form = form_layout()
        form.addRow("Operation", self.operation)
        self.explanation = hint()

        self.objects = QListWidget()
        for prim in project.primitives:
            if prim.kind == PrimitiveKind.SHEET:
                continue  # a sheet has no volume to combine
            material = next((m for m in project.materials if m.name == prim.material), None)
            item = QListWidgetItem(f"{prim.name}   ·  {PRIMITIVE_LABELS[prim.kind]}")
            item.setData(Qt.UserRole, prim.id)
            item.setData(Qt.DecorationRole, QColor(material.color if material else "#777777"))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if prim.id in self.picked else Qt.Unchecked)
            self.objects.addItem(item)
        self.objects.itemChanged.connect(self._toggled)

        self.order = QListWidget()
        up = QPushButton("↑")
        down = QPushButton("↓")
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        order_buttons = QVBoxLayout()
        order_buttons.addWidget(up)
        order_buttons.addWidget(down)
        order_buttons.addStretch(1)

        lists = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Objects"))
        left.addWidget(self.objects, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("Operand order"))
        right.addWidget(self.order, 1)
        lists.addLayout(left, 1)
        lists.addLayout(right, 1)
        lists.addLayout(order_buttons)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.explanation)
        layout.addLayout(lists, 1)
        layout.addWidget(self.buttons)
        self._refresh()

    def op(self) -> BooleanOp:
        return BooleanOp(self.operation.value())

    def operand_ids(self) -> List[str]:
        return list(self.picked)

    def _name(self, prim_id: str) -> str:
        return next((p.name for p in self.project.primitives if p.id == prim_id), prim_id)

    def _toggled(self, item: QListWidgetItem) -> None:
        prim_id = item.data(Qt.UserRole)
        checked = item.checkState() == Qt.Checked
        if checked and prim_id not in self.picked:
            self.picked.append(prim_id)
        elif not checked and prim_id in self.picked:
            self.picked.remove(prim_id)
        self._refresh()

    def _move(self, delta: int) -> None:
        row = self.order.currentRow()
        target = row + delta
        if 0 <= row < len(self.picked) and 0 <= target < len(self.picked):
            self.picked[row], self.picked[target] = self.picked[target], self.picked[row]
            self._refresh()
            self.order.setCurrentRow(target)

    def _refresh(self) -> None:
        op = self.op()
        self.order.clear()
        for i, prim_id in enumerate(self.picked):
            base = "   (base — the one being cut)" if i == 0 and op == BooleanOp.SUBTRACT else ""
            self.order.addItem(f"{i + 1}. {self._name(prim_id)}{base}")
        if op == BooleanOp.SUBTRACT:
            text = "The first object in the order is the one being cut; every other object is removed from it."
        elif op == BooleanOp.HULL:
            text = "The convex hull wraps every selected object in the smallest convex solid that contains them."
        else:
            text = "Order does not matter for this operation."
        self.explanation.setText(text + " The result becomes a single solid — CSXCAD has no CSG of its own, so it is "
                                 "evaluated here and handed to the solver as a polyhedron.")
        needed = MIN_OPERANDS[op]
        ok = self.buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled(len(self.picked) >= needed)
        ok.setText(f"Create {op.value}" if len(self.picked) >= needed else f"Pick at least {needed}")
