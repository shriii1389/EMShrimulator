"""The property editor for one coordinate system, or the global one."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.views.modeler.editor_base import NameEdit, PropertyEditor
from EMShrimulator.app.views.modeler.expression_edit import ExpressionVector
from EMShrimulator.app.widgets.form import ChoiceBox, Segmented, hint, row
from EMShrimulator.core.em import edits, frames
from EMShrimulator.core.em.model import CoordinateSystem

MODES = [("offset", "Offset"), ("rotated", "Rotated"), ("both", "Both")]


class CoordinateSystemEditor(PropertyEditor):
    anchorPickRequested = Signal(str, str)  # (this system, its object's id)

    def __init__(self, session, parent=None):
        super().__init__(session, "Coordinate system", parent)
        self.cs_name: Optional[str] = None  # None = the global system
        self.index = -1
        s = session

        self.name = NameEdit(self._rename)
        self.kind = QLabel()
        self.parent_box = ChoiceBox(tooltip="The system this one is offset from and turned from")
        self.parent_box.committed.connect(lambda v: self.set_field("parent", v or None))
        self.object_box = ChoiceBox(tooltip="The object it sits on. Another object keeps the same kind of point, "
                                            "in the same spot")
        self.object_box.committed.connect(lambda v: self.set_field("object_id", v))
        self.anchor = QLabel()
        self.anchor.setWordWrap(True)
        self.pick = QPushButton("Pick…")
        self.pick.setToolTip("Click another point of the object in the 3D view: its centre, a face centre, an "
                             "edge centre or a corner")
        self.pick.clicked.connect(self._pick)
        self.mode = Segmented(MODES)
        self.mode.setToolTip("What moves it from its base: an offset, a rotation, or both")
        self.mode.committed.connect(lambda v: self.set_field("mode", v))
        self.origin = ExpressionVector(s, path=lambda: self._sub("origin"))
        self.origin.setToolTip("How far its origin is from its base's, along the base's axes")
        self.origin.committed.connect(lambda v: self.set_field("origin", v))
        self.angles = ExpressionVector(s, "angle", step=15, path=lambda: self._sub("angles"))
        self.angles.setToolTip("Degrees it is turned about its base's X axis, then its Y, then its Z")
        self.angles.committed.connect(lambda v: self.set_field("angles", v))
        self.form.addRow("Name", self.name)
        self.form.addRow("Type", self.kind)
        self.form.addRow("Based on", self.parent_box)
        self.form.addRow("Object", self.object_box)
        self.anchor_row = row(self.anchor, self.pick)
        self.form.addRow("At", self.anchor_row)
        self.form.addRow("Moved by", self.mode)
        self.form.addRow("Offset", self.origin)
        self.form.addRow("Rotation (°)", self.angles)
        self.problem = hint(role="note-error")
        self.about = hint()
        self.outer.addWidget(self.problem)
        self.outer.addWidget(self.about)

        self.working = QPushButton("Make this the working system")
        self.working.setToolTip("New objects and ports are drawn in the working system, and the grid lies on its "
                                "XY plane with its origin marked")
        self.working.clicked.connect(self._make_working)
        self.working_note = QLabel("This is the working system.")
        self.working_note.setProperty("role", "hint")
        self.used = hint()
        self.delete = QPushButton("Delete")
        self.delete.setProperty("role", "danger")
        self.delete.clicked.connect(self._delete)
        self.outer.addWidget(self.working_note)
        self.outer.addWidget(row(self.working, self.delete))
        self.outer.addWidget(self.used)
        self._object_id = ""

    def _sub(self, field: str) -> Optional[str]:
        return f"coordinate_systems.{self.index}.{field}" if self.index >= 0 else None

    def set_field(self, field: str, value) -> None:
        name = self.cs_name
        if name:
            self.push(lambda: cmd.SetCoordinateSystemField(self.session, name, field, value))

    def _rename(self, new: str) -> None:
        old = self.cs_name
        if not old:
            return
        self.push(lambda: cmd.RenameCoordinateSystem(self.session, old, new))
        if edits.find_coordinate_system(self.session.project, new.strip()) is not None:
            self.session.select("cs", new.strip())

    def _pick(self) -> None:
        if self.cs_name and self._object_id:
            self.anchorPickRequested.emit(self.cs_name, self._object_id)

    def _make_working(self) -> None:
        name = self.cs_name
        self.push(lambda: cmd.SetWorkingCS(self.session, name))

    def _delete(self) -> None:
        name = self.cs_name
        if name:
            self.push(lambda: cmd.DeleteCoordinateSystem(self.session, name))

    def _show(self, widget, visible: bool) -> None:
        self.form.setRowVisible(widget, visible)

    def load(self, cs: Optional[CoordinateSystem], index: int) -> None:
        project = self.session.project
        self.cs_name = cs.name if cs is not None else None
        self.index = index
        is_global = cs is None
        on_object = not is_global and cs.kind == "object"
        self.title.setText("Coordinate system · Global" if is_global else f"Coordinate system · {cs.name}")
        for widget in (self.name, self.kind, self.mode):
            self._show(widget, not is_global)
        self._show(self.parent_box, not is_global and not on_object)
        self._show(self.object_box, on_object)
        self._show(self.anchor_row, on_object)
        self._show(self.origin, not is_global and cs.mode in ("offset", "both"))
        self._show(self.angles, not is_global and cs.mode in ("rotated", "both"))
        self.delete.setVisible(not is_global)
        working = project.working_cs == self.cs_name or (is_global and not project.working_cs)
        self.working.setVisible(not working)
        self.working_note.setVisible(working)
        if is_global:
            self.problem.setVisible(False)
            self.about.setText("The fixed global system. Its origin is where the red X and green Y axes of the "
                               "grid cross while it is the working system. Create a relative or an object "
                               "coordinate system from Modeling › Coordinate system.")
            self.used.setText("")
            return

        self.name.set_value(cs.name)
        self.mode.set_value(cs.mode)
        self.origin.set_value(cs.origin)
        self.angles.set_value(cs.angles)
        if on_object:
            prim = edits.find_primitive(project, cs.object_id or "")
            self._object_id = cs.object_id or ""
            self.kind.setText("Object - follows a point of an object")
            self.object_box.set_options([(p.id, p.name) for p in project.primitives])
            self.object_box.set_value(cs.object_id)
            self.anchor.setText(frames.describe_anchor(cs))
            self.pick.setEnabled(prim is not None)
            where = f"{frames.describe_anchor(cs)} of {prim.name}" if prim is not None else "a missing object"
            self.about.setText(f"Sits on the {where} and moves, turns and grows with it. On a face its Z axis "
                               "points out of the face; elsewhere its axes are the object's. The offset runs "
                               "along those axes and the rotation turns about them.")
        else:
            self._object_id = ""
            base = cs.parent or "Global"
            self.kind.setText("Relative - offset and turned from another system")
            others = [(c.name, c.name) for c in project.coordinate_systems if c.name != cs.name]
            self.parent_box.set_options([("", "Global")] + others)
            self.parent_box.set_value(cs.parent or "")
            self.about.setText(f"The offset runs along {base}'s axes and the rotation turns about them, X first, "
                               "then Y, then Z. Everything placed in it follows.")
        resolved = edits.find_coordinate_system(self.session.resolution.project, cs.name)
        error = resolved.placement_error if resolved is not None else ""
        self.problem.setText(f"Cannot be placed: {error}." if error else "")
        self.problem.setVisible(bool(error))
        users = edits.placed_in(project, cs.name)
        self.used.setText(f"Places {len(users)}: {', '.join(users[:6])}{'…' if len(users) > 6 else ''}." if users
                          else "Nothing is placed in it yet.")
        self.delete.setEnabled(not users)
        self.delete.setToolTip("Move what it places first" if users else "Delete this coordinate system")
