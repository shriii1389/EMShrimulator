"""The property editor for one material."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QPushButton

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.views.modeler.appearance_edit import AppearanceEdit
from EMShrimulator.app.views.modeler.editor_base import NameEdit, PropertyEditor
from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit
from EMShrimulator.app.widgets.form import ChoiceBox, hint
from EMShrimulator.core.em import edits
from EMShrimulator.core.em.model import Material, MaterialKind

KIND_LABELS = [
    (MaterialKind.PEC.value, "Perfect conductor (PEC)"),
    (MaterialKind.CONDUCTOR.value, "Lossy metal (σ)"),
    (MaterialKind.DIELECTRIC.value, "Dielectric (εr, tanδ)"),
    (MaterialKind.SHEET.value, "Conducting sheet (thin)"),
]


class MaterialEditor(PropertyEditor):
    def __init__(self, session, parent=None):
        super().__init__(session, "Material", parent)
        self.material_name: Optional[str] = None
        self.index = -1
        s = session
        f = self.form

        self.name = NameEdit(self._rename)
        self.kind = ChoiceBox(KIND_LABELS)
        self.kind.committed.connect(lambda v: self.set_field("kind", MaterialKind(v)))
        self.epsilon = ExpressionEdit(s, "plain", step=0.1, path=lambda: self._sub("epsilon"))
        self.epsilon.committed.connect(lambda v: self.set_field("epsilon", v))
        self.tan_delta = ExpressionEdit(s, "plain", step=0.001, path=lambda: self._sub("tan_delta"))
        self.tan_delta.committed.connect(lambda v: self.set_field("tan_delta", v))
        self.tan_delta.setToolTip("εr and tan δ are exact at the excitation centre. With Setup → Solver → "
                                  "Dielectric loss on Wideband, tan δ holds across the whole sweep.")
        self.mue = ExpressionEdit(s, "plain", step=0.1, path=lambda: self._sub("mue"))
        self.mue.committed.connect(lambda v: self.set_field("mue", v))
        self.kappa = ExpressionEdit(s, "plain", step=1e6, path=lambda: self._sub("kappa"))
        self.kappa.committed.connect(lambda v: self.set_field("kappa", v))
        self.thickness = ExpressionEdit(s, "plain", step=1, path=lambda: self._sub("thickness_um"))
        self.thickness.committed.connect(lambda v: self.set_field("thickness_um", v))
        self.appearance = AppearanceEdit("Material colour")
        self.appearance.setToolTip("How objects of this material are drawn, unless one sets its own")
        self.appearance.colorCommitted.connect(lambda v: self.set_field("color", v))
        self.appearance.opacityCommitted.connect(lambda v: self.set_field("opacity", v))
        self.used_hint = hint()
        self.delete = QPushButton("Delete material")
        self.delete.setProperty("role", "danger")
        self.delete.clicked.connect(self._delete)

        f.addRow("Name", self.name)
        f.addRow("Kind", self.kind)
        f.addRow("εr", self.epsilon)
        f.addRow("tan δ", self.tan_delta)
        f.addRow("µr", self.mue)
        f.addRow("σ (S/m)", self.kappa)
        f.addRow("Thickness (µm)", self.thickness)
        f.addRow("Colour", self.appearance)
        f.addRow(self.used_hint)
        f.addRow("", self.delete)

    def _sub(self, field: str) -> Optional[str]:
        return f"materials.{self.index}.{field}" if self.index >= 0 else None

    def set_field(self, field: str, value) -> None:
        name = self.material_name
        self.push(lambda: cmd.SetMaterialField(self.session, name, field, value))

    def _rename(self, new: str) -> None:
        old = self.material_name
        if self.push(lambda: cmd.RenameMaterial(self.session, old, new)) is not None or \
                edits.find_material(self.session.project, new) is not None:
            self.session.select("material", new)

    def _delete(self) -> None:
        name = self.material_name
        self.push(lambda: cmd.DeleteMaterial(self.session, name))

    def load(self, material: Material, index: int) -> None:
        self.material_name = material.name
        self.index = index
        self.title.setText(f"Material · {material.name}")
        self.name.set_value(material.name)
        self.kind.set_value(material.kind.value)
        for field in ("epsilon", "tan_delta", "mue", "kappa", "thickness_um"):
            getattr(self, "thickness" if field == "thickness_um" else field).set_value(getattr(material, field))
        dielectric = material.kind == MaterialKind.DIELECTRIC
        for widget in (self.epsilon, self.tan_delta, self.mue):
            self.form.setRowVisible(widget, dielectric)
        self.form.setRowVisible(self.kappa, material.kind in (MaterialKind.CONDUCTOR, MaterialKind.SHEET))
        self.form.setRowVisible(self.thickness, material.kind == MaterialKind.SHEET)
        self.appearance.set_value(material.color, material.opacity)
        users = edits.material_users(self.session.project, material.name)
        shown = ", ".join(users[:4]) + (f" and {len(users) - 4} more" if len(users) > 4 else "")
        self.used_hint.setText(f"Used by {shown}." if users else "Not used by any object.")
        self.delete.setEnabled(not users)
        self.delete.setToolTip("Give those objects another material first" if users else "Delete this material")
