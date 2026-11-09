"""The property editor for one modeled object, one page per kind of primitive."""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QMenu, QPushButton, QWidget

from EMShrimulator.app import commands as cmd
from EMShrimulator.app import theme
from EMShrimulator.app.views.modeler import units as unit_math
from EMShrimulator.app.views.modeler.appearance_edit import AppearanceEdit
from EMShrimulator.app.views.modeler.editor_base import NameEdit, PropertyEditor
from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit, ExpressionVector
from EMShrimulator.app.views.modeler.list_editors import PointListEditor, TransformListEditor
from EMShrimulator.app.views.modeler.point_size_edit import PointSizeEdit
from EMShrimulator.app.widgets.form import ChoiceBox, Segmented, form_layout, hint, row
from EMShrimulator.app.widgets.unit_spinbox import IntSpinBox
from EMShrimulator.core.em import edits, sheets
from EMShrimulator.core.em.model import BooleanOp, PortKind, Primitive, PrimitiveKind, SheetShape
from EMShrimulator.core.em.scene import appearance

AXES = [("x", "x"), ("y", "y"), ("z", "z")]
PLANE_AXES = {"x": ("Y", "Z"), "y": ("Z", "X"), "z": ("X", "Y")}

VOLUME_KINDS = {PrimitiveKind.BOX, PrimitiveKind.CYLINDER, PrimitiveKind.CYLINDRICAL_SHELL}
SPHERE_KINDS = {PrimitiveKind.SPHERE, PrimitiveKind.SPHERICAL_SHELL}
POLY_KINDS = {PrimitiveKind.POLYGON, PrimitiveKind.ROT_POLY}
WIRE_KINDS = {PrimitiveKind.WIRE, PrimitiveKind.CURVE}


class _Page(QWidget):
    """One kind-specific page; `load` shows an object, `set` commits a field."""

    def __init__(self, editor: "PrimitiveEditor"):
        super().__init__()
        self.editor = editor
        self.session = editor.session
        self.form: QFormLayout = form_layout()
        self.setLayout(self.form)

    def path(self, field: str) -> Optional[str]:
        base = self.editor.path()
        return f"{base}.{field}" if base else None

    def set(self, field: str, value, label: str = "") -> None:
        self.editor.set_field(field, value, label)

    def length_label(self, text: str) -> str:
        return f"{text} ({unit_math.label('length', self.session.project.display_units)})"

    def load(self, prim: Primitive) -> None:
        raise NotImplementedError


class VolumePage(_Page):
    """Box, cylinder and tube: two end points, and a radius for the round ones."""

    def __init__(self, editor):
        super().__init__(editor)
        self.points = PointSizeEdit(self.session, self.form, path=self.editor.path)
        self.points.committed.connect(lambda p1, p2: self.editor.set_fields({"p1": p1, "p2": p2}, "dimensions"))
        self.points.modeChanged.connect(lambda mode: self.set("size_mode", mode, "dimension mode"))
        self.radius = ExpressionEdit(self.session, step=0.1, path=lambda: self.path("radius"))
        self.radius.committed.connect(lambda v: self.set("radius", v))
        self.wall = ExpressionEdit(self.session, step=0.05, path=lambda: self.path("shell_width"))
        self.wall.committed.connect(lambda v: self.set("shell_width", v, "wall"))
        self.radius_label = QLabel()
        self.wall_label = QLabel()
        self.form.addRow(self.radius_label, self.radius)
        self.form.addRow(self.wall_label, self.wall)

    def load(self, prim: Primitive) -> None:
        round_kind = prim.kind != PrimitiveKind.BOX
        self.points.set_value(prim.p1, prim.p2, prim.size_mode, axis=round_kind)
        self.radius_label.setText(self.length_label("Radius"))
        self.wall_label.setText(self.length_label("Wall"))
        self.form.setRowVisible(self.radius, round_kind)
        self.form.setRowVisible(self.wall, prim.kind == PrimitiveKind.CYLINDRICAL_SHELL)
        self.radius.set_value(prim.radius or 0.0)
        self.wall.set_value(prim.shell_width or 0.0)


class SpherePage(_Page):
    def __init__(self, editor):
        super().__init__(editor)
        self.center = ExpressionVector(self.session, path=lambda: self.path("center"))
        self.center.committed.connect(lambda v: self.set("center", v, "centre"))
        self.radius = ExpressionEdit(self.session, step=0.1, path=lambda: self.path("radius"))
        self.radius.committed.connect(lambda v: self.set("radius", v))
        self.wall = ExpressionEdit(self.session, step=0.05, path=lambda: self.path("shell_width"))
        self.wall.committed.connect(lambda v: self.set("shell_width", v, "wall"))
        self.labels = [QLabel(), QLabel(), QLabel()]
        self.form.addRow(self.labels[0], self.center)
        self.form.addRow(self.labels[1], self.radius)
        self.form.addRow(self.labels[2], self.wall)

    def load(self, prim: Primitive) -> None:
        for label, text in zip(self.labels, ("Centre", "Radius", "Wall")):
            label.setText(self.length_label(text))
        self.center.set_value(prim.center or (0.0, 0.0, 0.0))
        self.radius.set_value(prim.radius or 0.0)
        self.wall.set_value(prim.shell_width or 0.0)
        self.form.setRowVisible(self.wall, prim.kind == PrimitiveKind.SPHERICAL_SHELL)


class PolygonPage(_Page):
    """An extruded or revolved outline in the plane normal to an axis."""

    def __init__(self, editor):
        super().__init__(editor)
        self.normal = ChoiceBox(AXES)
        self.normal.committed.connect(lambda v: self.set("normal_axis", v, "normal axis"))
        self.elevation = ExpressionEdit(self.session, step=0.1, path=lambda: self.path("elevation"))
        self.elevation.committed.connect(lambda v: self.set("elevation", v))
        self.extrude = ExpressionEdit(self.session, step=0.1, path=lambda: self.path("extrude_length"))
        self.extrude.committed.connect(lambda v: self.set("extrude_length", v, "extrusion"))
        self.rot_axis = ChoiceBox(AXES)
        self.rot_axis.committed.connect(lambda v: self.set("rot_axis", v, "sweep axis"))
        self.start = ExpressionEdit(self.session, "angle", step=15, path=lambda: self.path("angle_start_deg"))
        self.start.committed.connect(lambda v: self.set("angle_start_deg", v, "sweep start"))
        self.stop = ExpressionEdit(self.session, "angle", step=15, path=lambda: self.path("angle_stop_deg"))
        self.stop.committed.connect(lambda v: self.set("angle_stop_deg", v, "sweep stop"))
        self.points = PointListEditor(self.session, 2, title="Outline points (u, v)", min_points=3,
                                      path=lambda: self.path("points"))
        self.points.committed.connect(lambda v: self.set("points", v, "outline"))
        self.elevation_label, self.extrude_label = QLabel(), QLabel()
        self.form.addRow("Normal axis", self.normal)
        self.form.addRow(self.elevation_label, self.elevation)
        self.form.addRow(self.extrude_label, self.extrude)
        self.form.addRow("Sweep axis", self.rot_axis)
        self.form.addRow("Start (°)", self.start)
        self.form.addRow("Stop (°)", self.stop)
        self.form.addRow(self.points)

    def load(self, prim: Primitive) -> None:
        revolve = prim.kind == PrimitiveKind.ROT_POLY
        self.elevation_label.setText(self.length_label("Elevation"))
        self.extrude_label.setText(self.length_label("Extrude"))
        self.normal.set_value(prim.normal_axis or "z")
        self.elevation.set_value(prim.elevation or 0.0)
        self.extrude.set_value(prim.extrude_length or 0.0)
        self.rot_axis.set_value(prim.rot_axis or sheets.default_rot_axis(prim.normal_axis or "z"))
        self.start.set_value(prim.angle_start_deg if prim.angle_start_deg is not None else 0.0)
        self.stop.set_value(prim.angle_stop_deg if prim.angle_stop_deg is not None else 360.0)
        self.points.set_value(prim.points or [])
        self.form.setRowVisible(self.extrude, not revolve)
        for widget in (self.rot_axis, self.start, self.stop):
            self.form.setRowVisible(widget, revolve)


class SheetPage(_Page):
    """A zero-thickness outline - usually the cross-section a port is assigned to."""

    def __init__(self, editor):
        super().__init__(editor)
        self.shape = ChoiceBox([(s.value, label) for s, label in edits.SHEET_LABELS.items()])
        self.shape.committed.connect(lambda v: self.set("sheet_shape", SheetShape(v), "sheet shape"))
        self.normal = Segmented([("x", "X"), ("y", "Y"), ("z", "Z")])
        self.normal.committed.connect(lambda v: self.set("normal_axis", v, "sheet normal"))
        self.origin = ExpressionVector(self.session, path=lambda: self.path("origin"))
        self.origin.committed.connect(lambda v: self.set("origin", v, "centre"))
        self.size_u = ExpressionEdit(self.session, step=0.5, path=lambda: self.path("sheet_size.0"))
        self.size_u.committed.connect(lambda v: self._set_size(0, v))
        self.size_v = ExpressionEdit(self.session, step=0.5, path=lambda: self.path("sheet_size.1"))
        self.size_v.committed.connect(lambda v: self._set_size(1, v))
        self.radius = ExpressionEdit(self.session, step=0.1, path=lambda: self.path("radius"))
        self.radius.committed.connect(lambda v: self.set("radius", v))
        self.inner = ExpressionEdit(self.session, step=0.1, path=lambda: self.path("radius_inner"),
                                    tooltip="The centre conductor's outside radius")
        self.inner.committed.connect(lambda v: self.set("radius_inner", v, "inner radius"))
        self.segments = IntSpinBox(minimum=8, maximum=512, step=8,
                                   tooltip="How finely the curve is broken up, for meshing and drawing")
        self.segments.committed.connect(lambda v: self.set("sheet_segments", int(v), "segments"))
        self.points = PointListEditor(self.session, 2, title="Outline points (u, v), relative to the centre",
                                      min_points=3, path=lambda: self.path("points"))
        self.points.committed.connect(lambda v: self.set("points", v, "outline"))
        self.coax_hint = hint()
        self.material_hint = hint()
        self.ports_hint = hint()
        # A port is assigned on the sheet itself.
        self.assign = QPushButton("Assign lumped port")
        self.assign.setProperty("role", "primary")
        self.assign.setToolTip("Make this sheet a lumped port, with an integration line across it")
        self.assign.clicked.connect(lambda: self._assign(PortKind.LUMPED))
        self.assign_other = QPushButton("Other port")
        other = QMenu(self.assign_other)
        for kind in (PortKind.MICROSTRIP, PortKind.RECT_WAVEGUIDE, PortKind.COAXIAL):
            action = other.addAction(edits.PORT_LABELS[kind], lambda kind=kind: self._assign(kind))
            action.setToolTip(edits.SHEET_PORT_NEEDS[kind])
        other.setToolTipsVisible(True)
        self.assign_other.setMenu(other)
        self.edit_port = QPushButton()
        self.edit_port.clicked.connect(lambda: self.session.select("port", self._port_ids[0]))
        self._port_ids = []
        self.labels = {name: QLabel() for name in ("origin", "u", "v", "radius", "inner")}

        self.form.addRow("Shape", self.shape)
        self.form.addRow("Normal", self.normal)
        self.form.addRow(self.labels["origin"], self.origin)
        self.form.addRow(self.labels["u"], self.size_u)
        self.form.addRow(self.labels["v"], self.size_v)
        self.form.addRow(self.labels["radius"], self.radius)
        self.form.addRow(self.labels["inner"], self.inner)
        self.form.addRow("Segments", self.segments)
        self.form.addRow(self.points)
        self.form.addRow(self.coax_hint)
        self.form.addRow(self.material_hint)
        self.form.addRow(self.ports_hint)
        self.form.addRow(row(self.assign, self.assign_other, self.edit_port))
        self._size = (0.0, 0.0)

    def _assign(self, kind: PortKind) -> None:
        prim_id = self.editor.prim_id
        port_id = self.editor.push(lambda: cmd.AssignPort(self.session, prim_id, kind))
        if port_id:
            self.session.select("port", port_id)

    def _set_size(self, i: int, value) -> None:
        size = list(self._size)
        size[i] = value
        self.set("sheet_size", tuple(size), "sheet size")

    def load(self, prim: Primitive) -> None:
        shape = prim.sheet_shape or SheetShape.RECTANGLE
        normal = prim.normal_axis or "z"
        u_axis, v_axis = PLANE_AXES[normal]
        self._size = tuple(prim.sheet_size or (0.0, 0.0))
        self.shape.set_value(shape.value)
        self.normal.set_value(normal)
        self.labels["origin"].setText(self.length_label("Centre"))
        self.labels["u"].setText(self.length_label(f"Size {u_axis}"))
        self.labels["v"].setText(self.length_label(f"Size {v_axis}"))
        self.labels["radius"].setText(self.length_label("Outer r" if shape == SheetShape.RING else "Radius"))
        self.labels["inner"].setText(self.length_label("Inner r"))
        self.origin.set_value(prim.origin or (0.0, 0.0, 0.0))
        self.size_u.set_value(self._size[0])
        self.size_v.set_value(self._size[1])
        self.radius.set_value(prim.radius or 0.0)
        self.inner.set_value(prim.radius_inner or 0.0)
        self.segments.set_stored(prim.sheet_segments)
        self.points.set_value(prim.points or [])

        boxy = shape in (SheetShape.RECTANGLE, SheetShape.ELLIPSE)
        round_ = shape in (SheetShape.CIRCLE, SheetShape.RING)
        self.form.setRowVisible(self.size_u, boxy)
        self.form.setRowVisible(self.size_v, boxy)
        self.form.setRowVisible(self.radius, round_)
        self.form.setRowVisible(self.inner, shape == SheetShape.RING)
        self.form.setRowVisible(self.segments, shape in (SheetShape.CIRCLE, SheetShape.ELLIPSE, SheetShape.RING))
        self.form.setRowVisible(self.points, shape == SheetShape.POLYGON)

        ports = [p for p in self.session.project.ports if p.sheet == prim.name]
        self.coax_hint.setVisible(False)
        if shape == SheetShape.RING:
            resolved = self.session.resolution.project
            r_out = float(edits.get_path(resolved, f"{self.editor.path()}.radius", 0.0) or 0.0)
            r_in = float(edits.get_path(resolved, f"{self.editor.path()}.radius_inner", 0.0) or 0.0)
            text = coax_hint_text(r_out, r_in, ports[0].impedance if ports else None)
            if text:
                self.coax_hint.setText(text)
                self.coax_hint.setVisible(True)

        if not prim.material:
            self.material_hint.setText("Construction sheet: nothing is built here. Assign a port to it, or give "
                                       "it a material to make it a real zero-thickness object.")
            self.material_hint.setProperty("role", "hint")
        elif shape == SheetShape.RING:
            self.material_hint.setText("A ring has a hole, which a flat CSXCAD polygon cannot have. Clear the "
                                       "material to use this as a coaxial port, or model the metal with a Tube.")
            self.material_hint.setProperty("role", "error")
        else:
            self.material_hint.setText("")
        self.material_hint.style().polish(self.material_hint)
        self.material_hint.setVisible(bool(self.material_hint.text()))
        self._port_ids = [p.id for p in ports]
        self.ports_hint.setText(
            "Not a port yet. Assign one to excite the design here." if not ports else
            f"This sheet is {', '.join(f'{p.name} ({edits.PORT_LABELS[p.kind].lower()})' for p in ports)}.")
        self.assign.setVisible(not ports)
        self.assign_other.setVisible(not ports)
        self.edit_port.setVisible(bool(ports))
        if ports:
            self.edit_port.setText(f"Edit {ports[0].name}")


class WirePage(_Page):
    def __init__(self, editor):
        super().__init__(editor)
        self.radius = ExpressionEdit(self.session, step=0.05, path=lambda: self.path("radius"))
        self.radius.committed.connect(lambda v: self.set("radius", v))
        self.radius_label = QLabel()
        self.path_points = PointListEditor(self.session, 3, title="Path points", min_points=2,
                                           path=lambda: self.path("path"))
        self.path_points.committed.connect(lambda v: self.set("path", v, "path"))
        self.form.addRow(self.radius_label, self.radius)
        self.form.addRow(self.path_points)

    def load(self, prim: Primitive) -> None:
        self.radius_label.setText(self.length_label("Radius"))
        self.radius.set_value(prim.radius or 0.0)
        self.path_points.set_value(prim.path or [])


class BooleanPage(_Page):
    def __init__(self, editor):
        super().__init__(editor)
        self.op = ChoiceBox([(op.value, label) for op, label in edits.BOOLEAN_LABELS.items()])
        self.op.committed.connect(lambda v: self.set("boolean_op", BooleanOp(v), "operation"))
        self.operands = hint()
        self.explode = QPushButton("Explode into operands")
        self.explode.clicked.connect(self._explode)
        self.form.addRow("Operation", self.op)
        self.form.addRow(self.operands)
        self.form.addRow("", self.explode)

    def load(self, prim: Primitive) -> None:
        self.op.set_value((prim.boolean_op or BooleanOp.UNION).value)
        names = ", ".join(c.name for c in prim.children)
        self.operands.setText(
            f"{len(prim.children)} operand(s): {names}. Evaluated into a single solid and handed to CSXCAD "
            "as a polyhedron. For subtraction the first operand is the one being cut.")

    def _explode(self) -> None:
        prim_id = self.editor.prim_id
        ids = self.editor.push(lambda: cmd.ExplodeBoolean(self.session, prim_id))
        if ids:
            self.session.select("primitive", ids[0])


def coax_hint_text(r_outer: float, r_inner: float, impedance) -> str:
    """Coax Z0 for the sheet radii, and the fill εr that would match the port impedance."""
    if not (r_inner > 0 and r_outer > r_inner):
        return ""
    air = 60 * math.log(r_outer / r_inner)
    text = f"Z₀ ≈ {air:.1f} Ω in air, or that divided by √εr once filled."
    try:
        reference = float(impedance) if impedance is not None else 0.0
    except (TypeError, ValueError):
        reference = 0.0
    if reference > 0:
        text += f" A fill of εr {(air / reference) ** 2:.2f} gives the {reference:.0f} Ω this port references."
    return text


class PrimitiveEditor(PropertyEditor):
    focusRequested = Signal(str, str)  # ("primitive", id)

    def __init__(self, session, parent=None):
        super().__init__(session, "Object", parent)
        self.prim_id: Optional[str] = None
        self.index = -1

        zoom = QPushButton("Zoom to")
        zoom.setToolTip("Turn the camera to this object")
        zoom.clicked.connect(lambda: self.focusRequested.emit("primitive", self.prim_id or ""))
        duplicate = QPushButton("Duplicate")
        duplicate.clicked.connect(self._duplicate)
        delete = QPushButton("Delete")
        delete.setProperty("role", "danger")
        delete.clicked.connect(self._delete)
        self.outer.insertWidget(1, row(zoom, duplicate, delete))

        self.name = NameEdit(lambda v: self.set_field("name", v))
        self.material = ChoiceBox()
        self.material.committed.connect(lambda v: self.set_field("material", v))
        self.priority = IntSpinBox(minimum=0, maximum=100000, tooltip=(
            "Higher priority wins where objects overlap - CSXCAD resolves overlap by priority, not by order"))
        self.priority.committed.connect(lambda v: self.set_field("priority", int(v)))
        self.model = QCheckBox("Model")
        self.model.setToolTip("Untick to make this a non-model object: it stays in the drawing and can be snapped "
                              "to, but the mesh and the solver leave it out")
        self.model.clicked.connect(lambda on: self.set_field("model", bool(on), "model"))
        self.appearance = AppearanceEdit("Object colour")
        self.appearance.colorCommitted.connect(lambda v: self.set_field("color", v, "colour"))
        self.appearance.opacityCommitted.connect(lambda v: self.set_field("opacity", v))
        self.use_material = QPushButton("Use material")
        self.use_material.clicked.connect(lambda: self.set_fields({"color": None, "opacity": None}, "appearance"))
        self.appearance.layout().addWidget(self.use_material)
        self.form.addRow("Name", self.name)
        self.form.addRow("Material", self.material)
        self.form.addRow("Priority", self.priority)
        self.form.addRow("", self.model)
        self.form.addRow("Colour", self.appearance)

        # One page per group of kinds; only the current one is visible.
        self.pages = []
        self.page_for = {}
        for page_class, kinds in ((VolumePage, VOLUME_KINDS), (SpherePage, SPHERE_KINDS),
                                  (PolygonPage, POLY_KINDS), (SheetPage, {PrimitiveKind.SHEET}),
                                  (WirePage, WIRE_KINDS), (BooleanPage, {PrimitiveKind.BOOLEAN})):
            page = page_class(self)
            page.setVisible(False)
            self.pages.append(page)
            self.outer.addWidget(page)
            for kind in kinds:
                self.page_for[kind] = page

        tail = form_layout()
        self.coordinate_system = ChoiceBox()
        self.coordinate_system.committed.connect(lambda v: self.set_field("coordinate_system", v or None,
                                                                           "coordinate system"))
        tail.addRow("Coord. system", self.coordinate_system)
        self.outer.addLayout(tail)
        self.transforms = TransformListEditor(session, path=lambda: self._sub("transforms"))
        self.transforms.committed.connect(lambda v: self.set_field("transforms", v, "transforms"))
        self.outer.addWidget(self.transforms)

    def path(self) -> Optional[str]:
        return f"primitives.{self.index}" if self.index >= 0 else None

    def _sub(self, field: str) -> Optional[str]:
        base = self.path()
        return f"{base}.{field}" if base else None

    def set_field(self, field: str, value, label: str = "") -> None:
        prim_id = self.prim_id
        self.push(lambda: cmd.SetPrimitiveField(self.session, prim_id, field, value, label))

    def set_fields(self, values: dict, label: str) -> None:
        prim_id = self.prim_id
        self.push(lambda: cmd.SetPrimitiveFields(self.session, prim_id, values, label))

    def _duplicate(self) -> None:
        prim_id = self.prim_id
        new_id = self.push(lambda: cmd.DuplicatePrimitive(self.session, prim_id))
        if new_id:
            self.session.select("primitive", new_id)

    def _delete(self) -> None:
        prim_id = self.prim_id
        self.push(lambda: cmd.DeletePrimitive(self.session, prim_id))

    def load(self, prim: Primitive, index: int) -> None:
        self.prim_id = prim.id
        self.index = index
        project = self.session.project
        self.title.setText(f"Object · {edits.PRIMITIVE_LABELS[prim.kind]}")
        self.name.set_value(prim.name)
        options = [(m.name, m.name) for m in project.materials]
        if prim.kind == PrimitiveKind.SHEET:
            options.insert(0, ("", "None — construction only"))
        self.material.set_options(options)
        self.material.set_value(prim.material)
        self.priority.set_stored(prim.priority)
        self.model.setChecked(prim.model)
        color, opacity = appearance(prim, {m.name: m for m in project.materials})
        self.appearance.set_value(color or theme.colors()["accent"], opacity)
        own = prim.color is not None or prim.opacity is not None
        self.use_material.setVisible(own)
        self.use_material.setText("Use material" if prim.material else "Use default")
        self.use_material.setToolTip("Go back to the material's colour and opacity" if prim.material else
                                     "Go back to the default colour and opacity")
        page = self.page_for[prim.kind]
        page.load(prim)
        for candidate in self.pages:
            candidate.setVisible(candidate is page)
        self.coordinate_system.set_options([("", "Global")] + [(cs.name, cs.name) for cs in project.coordinate_systems])
        self.coordinate_system.set_value(prim.coordinate_system or "")
        self.transforms.set_value(prim.transforms)
