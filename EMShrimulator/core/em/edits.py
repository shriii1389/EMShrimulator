"""Modeler project edits. Each mutates the project in place; objects by id, materials/parameters by name."""
from __future__ import annotations

import colorsys
import random
import re
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from EMShrimulator.core.em.model import (
    BooleanOp,
    CoordinateSystem,
    FieldDump,
    Material,
    MaterialKind,
    ModelerProject,
    Parameter,
    ParameterSweep,
    Port,
    PortKind,
    Primitive,
    PrimitiveKind,
    ResultPlot,
    SheetShape,
    SweepVariable,
)
from EMShrimulator.core.em.params import FUNCTIONS
from EMShrimulator.core.errors import EditError

PRIMITIVE_LABELS: Dict[PrimitiveKind, str] = {
    PrimitiveKind.BOX: "Box",
    PrimitiveKind.SHEET: "Sheet",
    PrimitiveKind.CYLINDER: "Cylinder",
    PrimitiveKind.CYLINDRICAL_SHELL: "Tube",
    PrimitiveKind.SPHERE: "Sphere",
    PrimitiveKind.SPHERICAL_SHELL: "Spherical shell",
    PrimitiveKind.POLYGON: "Polygon",
    PrimitiveKind.ROT_POLY: "Revolve",
    PrimitiveKind.WIRE: "Wire",
    PrimitiveKind.CURVE: "Curve",
    PrimitiveKind.BOOLEAN: "Boolean",
}

BOOLEAN_LABELS: Dict[BooleanOp, str] = {
    BooleanOp.UNION: "Union (A ∪ B)",
    BooleanOp.INTERSECT: "Intersect (A ∩ B)",
    BooleanOp.SUBTRACT: "Subtract (A − B)",
    BooleanOp.XOR: "Symmetric difference (A ⊖ B)",
    BooleanOp.HULL: "Convex hull",
}

PORT_LABELS: Dict[PortKind, str] = {
    PortKind.LUMPED: "Lumped (circuit)",
    PortKind.MICROSTRIP: "Microstrip (de-embedded)",
    PortKind.RECT_WAVEGUIDE: "Rectangular waveguide",
    PortKind.COAXIAL: "Coaxial (TEM)",
}

SHEET_LABELS: Dict[SheetShape, str] = {
    SheetShape.RECTANGLE: "Rectangle",
    SheetShape.CIRCLE: "Circle",
    SheetShape.ELLIPSE: "Ellipse",
    SheetShape.RING: "Ring (coax cross-section)",
    SheetShape.POLYGON: "Polygon",
}

# Report kinds against frequency; any of their traces can share one report.
REPORT_LABELS: Dict[str, str] = {
    "s": "Solution data", "y": "Solution data", "z": "Solution data", "vswr": "Solution data",
    "group_delay": "Solution data", "tdr": "TDR impedance", "farfield": "Far field",
    "antenna": "Antenna parameters", "nearfield": "Near field",
}
DISPLAY_LABELS: Dict[str, str] = {
    "rect": "Rectangular", "polar": "Polar", "smith": "Smith chart", "table": "Data table",
    "contour": "Contour", "rect3d": "3D rectangular", "spherical": "3D spherical", "cylindrical": "3D cylindrical",
    "overlay": "On the model",
}
FREQUENCY_DISPLAYS = ("rect", "polar", "smith", "table")
# The displays each report kind allows; the first is the default.
DISPLAYS: Dict[str, Tuple[str, ...]] = {
    "s": FREQUENCY_DISPLAYS, "y": FREQUENCY_DISPLAYS, "z": FREQUENCY_DISPLAYS, "vswr": FREQUENCY_DISPLAYS,
    "group_delay": FREQUENCY_DISPLAYS, "tdr": ("rect", "table"),
    "farfield": ("polar", "rect", "contour", "rect3d", "spherical", "cylindrical", "overlay", "table"),
    "antenna": ("table",) + FREQUENCY_DISPLAYS, "nearfield": ("contour", "rect", "table"),
}
FREQUENCY_KINDS = ("s", "y", "z", "vswr", "group_delay", "antenna")


def _trace(category: str, term: str, function: str, axis: str = "auto") -> Dict[str, Any]:
    return {"category": category, "term": term, "function": function, "axis": axis}
# New report menu: preset -> (name, fields), grouped by category.
PLOT_PRESETS: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "s_db": ("S-parameters (dB)", {"kind": "s", "display": "rect", "function": "db"}),
    "s_phase": ("S-parameter phase", {"kind": "s", "display": "rect", "function": "phase"}),
    "s_polar": ("S-parameters (polar)", {"kind": "s", "display": "polar"}),
    "smith": ("Smith chart", {"kind": "s", "display": "smith"}),
    "s_table": ("S-parameter table", {"kind": "s", "display": "table", "function": "db"}),
    "z": ("Z-parameters", {"kind": "z", "display": "rect", "function": "re_im"}),
    "y": ("Y-parameters", {"kind": "y", "display": "rect", "function": "re_im"}),
    "vswr": ("VSWR", {"kind": "vswr", "display": "rect"}),
    "group_delay": ("Group delay", {"kind": "group_delay", "display": "rect"}),
    "tdr": ("TDR impedance", {"kind": "tdr", "display": "rect"}),
    "match": ("Return loss and input impedance", {"kind": "s", "display": "rect", "curves": [
        _trace("s", "S11", "db"), _trace("z", "Z11", "re"), _trace("z", "Z11", "im")]}),
    "match_smith": ("S11 and Z11 on the Smith chart", {"kind": "s", "display": "smith", "curves": [
        _trace("s", "S11", "db"), _trace("z", "Z11", "re")]}),
    "elevation": ("Elevation pattern, φ = 0° and 90°", {
        "kind": "farfield", "display": "polar", "sweep": "theta", "cuts": [0.0, 90.0],
        "curves": [_trace("gain", "total", "db")]}),
    "elevation_rect": ("Elevation pattern, φ = 0° and 90° (rectangular)", {
        "kind": "farfield", "display": "rect", "sweep": "theta", "cuts": [0.0, 90.0],
        "curves": [_trace("gain", "total", "db")]}),
    "azimuth": ("Azimuth pattern, θ = 90°", {
        "kind": "farfield", "display": "polar", "sweep": "phi", "cuts": [90.0],
        "curves": [_trace("gain", "total", "db")]}),
    "pattern_polar": ("Radiation pattern (polar)", {"kind": "farfield", "display": "polar"}),
    "pattern_rect": ("Radiation pattern (rectangular)", {"kind": "farfield", "display": "rect"}),
    "pattern_3d": ("3D radiation pattern", {"kind": "farfield", "display": "spherical"}),
    "pattern_rect3d": ("3D rectangular pattern", {"kind": "farfield", "display": "rect3d"}),
    "pattern_cylindrical": ("3D cylindrical pattern", {"kind": "farfield", "display": "cylindrical"}),
    "pattern_overlay": ("Pattern on the model", {"kind": "farfield", "display": "overlay"}),
    "pattern_map": ("Radiation pattern (θ–φ contour)", {"kind": "farfield", "display": "contour"}),
    "axial_ratio": ("Axial ratio", {"kind": "farfield", "display": "rect", "quantity": "axial_ratio"}),
    # The plot asks for the co-polarisation rather than guess it.
    "cross_pol": ("Co- and cross-polarisation", {"kind": "farfield", "display": "polar", "cross_pol": True}),
    "gain_freq": ("Gain and axial ratio vs frequency", {"kind": "antenna", "display": "rect", "curves": [
        _trace("realized_gain", "total", "db"), _trace("axial_ratio", "total", "value")]}),
    "antenna": ("Antenna parameters", {"kind": "antenna", "display": "table"}),
    "antenna_freq": ("Antenna parameters vs frequency", {"kind": "antenna", "display": "rect"}),
    "nearfield": ("Near field", {"kind": "nearfield", "display": "contour"}),
    "nearfield_line": ("Near field along a line", {"kind": "nearfield", "display": "rect"}),
}
PLOT_LABELS: Dict[str, str] = {preset: label for preset, (label, _fields) in PLOT_PRESETS.items()}
PLOT_MENU: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Solution data", ("s_db", "s_phase", "s_polar", "smith", "s_table", "match", "match_smith", "z", "y", "vswr",
                       "group_delay", "tdr")),
    ("Far field", ("elevation", "elevation_rect", "azimuth", "pattern_polar", "pattern_rect", "pattern_3d",
                   "pattern_rect3d", "pattern_cylindrical", "pattern_overlay", "pattern_map", "axial_ratio",
                   "cross_pol", "gain_freq", "antenna", "antenna_freq")),
    ("Near field", ("nearfield", "nearfield_line")),
)


def plot_label(plot: ResultPlot) -> str:
    """What a report shows and how: 'Far field · 3D spherical'."""
    return f"{REPORT_LABELS.get(plot.kind, plot.kind)} · {DISPLAY_LABELS.get(plot.display, plot.display)}"

SHEET_PORT_NEEDS: Dict[PortKind, str] = {
    PortKind.LUMPED: "Any shape, with an integration line drawn on it.",
    PortKind.MICROSTRIP: "A rectangle across the substrate, with an extrude depth to de-embed along and an "
                         "integration line from the ground up to the strip.",
    PortKind.RECT_WAVEGUIDE: "A rectangle. The TE/TM modes are the closed-form rectangular ones.",
    PortKind.COAXIAL: "A ring. The outer radius is the shield, the inner one the centre conductor.",
}

MIN_OPERANDS = {op: (1 if op == BooleanOp.HULL else 2) for op in BooleanOp}


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def unique_name(base: str, existing: Iterable[str]) -> str:
    """`base1`, `base2`, … - the first not already taken."""
    taken = set(existing)
    n = 1
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


# --- lookups ---

def find_primitive(project: ModelerProject, primitive_id: str) -> Optional[Primitive]:
    return next((p for p in project.primitives if p.id == primitive_id), None)


def find_port(project: ModelerProject, port_id: str) -> Optional[Port]:
    return next((p for p in project.ports if p.id == port_id), None)


def find_dump(project: ModelerProject, dump_id: str) -> Optional[FieldDump]:
    return next((d for d in project.dumps if d.id == dump_id), None)


def find_material(project: ModelerProject, name: str) -> Optional[Material]:
    return next((m for m in project.materials if m.name == name), None)


def index_of(items: List[Any], item_id: str) -> int:
    return next((i for i, x in enumerate(items) if getattr(x, "id", None) == item_id), -1)


def _number(value: Any, fallback: float = 0.0) -> float:
    """Best-effort number for a field that may still be holding an expression."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def suggest_scale(project: ModelerProject) -> float:
    """A default size for new objects, scaled to the existing design."""
    extent = 0.0
    for prim in project.primitives:
        for point in (prim.p1, prim.p2, prim.center):
            if point:
                extent = max(extent, *(abs(_number(v)) for v in point))
        extent = max(extent, abs(_number(prim.radius)))
    return max(extent * 0.4, 0.1) if extent > 0 else 10.0


def material_users(project: ModelerProject, name: str) -> List[str]:
    """Names of the objects made of this material (boolean operands excluded)."""
    return [p.name for p in project.primitives if p.material == name]


def uses_material(project: ModelerProject, name: str) -> bool:
    return bool(material_users(project, name))


# --- primitives ---

def add_primitive(project: ModelerProject, kind: PrimitiveKind, sheet_shape: Optional[SheetShape] = None) -> str:
    """Add an object of `kind` sized to the design and return its id; sheets use `sheet_shape`."""
    kind = PrimitiveKind(kind)
    s = suggest_scale(project)
    base_name = PRIMITIVE_LABELS[kind].lower().replace(" ", "_") + "_"
    if kind == PrimitiveKind.SHEET and sheet_shape is not None:
        base_name = f"{SheetShape(sheet_shape).value}_"
    prim = Primitive(
        id=new_id(),
        name=unique_name(base_name, (p.name for p in project.primitives)),
        kind=kind,
        material=project.materials[0].name if project.materials else "PEC",
        coordinate_system=project.working_cs,
        priority=10,
        size_mode=kind == PrimitiveKind.BOX,
        p1=(0.0, 0.0, 0.0),
        p2=(s, s, s * 0.1),
    )
    if kind in (PrimitiveKind.CYLINDER, PrimitiveKind.CYLINDRICAL_SHELL):
        prim.p2 = (0.0, 0.0, s)
        prim.radius = s * 0.2
        prim.shell_width = s * 0.05
    elif kind in (PrimitiveKind.SPHERE, PrimitiveKind.SPHERICAL_SHELL):
        prim.p1 = prim.p2 = None
        prim.center = (0.0, 0.0, 0.0)
        prim.radius = s * 0.3
        prim.shell_width = s * 0.05
    elif kind in (PrimitiveKind.POLYGON, PrimitiveKind.ROT_POLY):
        prim.p1 = prim.p2 = None
        prim.points = [(0.0, 0.0), (s, 0.0), (s, s * 0.6), (0.0, s * 0.6)]
        prim.normal_axis = "z"
        prim.elevation = 0.0
        prim.extrude_length = s * 0.1
        prim.rot_axis = "y"  # in the outline's plane: u is the radius, v the height
        prim.angle_start_deg = 0.0
        prim.angle_stop_deg = 360.0
    elif kind == PrimitiveKind.SHEET:
        prim.p1 = prim.p2 = None
        prim.material = ""
        prim.sheet_shape = SheetShape(sheet_shape or SheetShape.RECTANGLE)
        prim.normal_axis = "z"
        prim.origin = (0.0, 0.0, 0.0)
        prim.sheet_size = (s * 0.4, s * 0.4)
        prim.radius = s * 0.2
        prim.radius_inner = s * 0.07
        prim.points = [(0.0, 0.0), (s * 0.3, 0.0), (s * 0.3, s * 0.3)]
        prim.priority = 100
    elif kind in (PrimitiveKind.WIRE, PrimitiveKind.CURVE):
        prim.p1 = prim.p2 = None
        prim.path = [(0.0, 0.0, 0.0), (s, 0.0, 0.0), (s, s, 0.0)]
        prim.radius = s * 0.03
    elif kind == PrimitiveKind.BOOLEAN:
        raise EditError("A boolean is made by combining existing objects, not drawn directly.")
    project.primitives.append(prim)
    return prim.id


# Fields set by drawing in the 3D view.
DRAWN_FIELDS = frozenset({"p1", "p2", "center", "radius", "origin", "normal_axis", "sheet_size"})


def draw_primitive(project: ModelerProject, kind: PrimitiveKind, sheet_shape: Optional[SheetShape],
                   fields: Dict[str, Any]) -> str:
    """Add an object as drawn in the 3D view."""
    unknown = set(fields) - DRAWN_FIELDS
    if unknown:
        raise EditError(f"Drawing cannot set {', '.join(sorted(unknown))}.")
    prim_id = add_primitive(project, kind, sheet_shape)
    prim = find_primitive(project, prim_id)
    for field, value in fields.items():
        setattr(prim, field, value)
    return prim_id


def delete_primitive(project: ModelerProject, primitive_id: str) -> None:
    """Delete an object and any ports on it. Coordinate systems on it stay, based on global."""
    prim = find_primitive(project, primitive_id)
    if prim is None:
        raise EditError("That object no longer exists.")
    _release_anchors(project, [primitive_id])
    project.primitives = [p for p in project.primitives if p.id != primitive_id]
    if prim.kind == PrimitiveKind.SHEET and not any(p.name == prim.name for p in project.primitives):
        project.ports = [port for port in project.ports if port.sheet != prim.name]


def duplicate_primitive(project: ModelerProject, primitive_id: str) -> str:
    source = find_primitive(project, primitive_id)
    if source is None:
        raise EditError("That object no longer exists.")
    copy = source.model_copy(deep=True)
    copy.id = new_id()
    copy.name = unique_name(f"{source.name}_copy", (p.name for p in project.primitives))
    project.primitives.append(copy)
    return copy.id


def set_primitive_field(project: ModelerProject, primitive_id: str, field: str, value: Any) -> None:
    prim = find_primitive(project, primitive_id)
    if prim is None:
        raise EditError("That object no longer exists.")
    if field == "name":
        value = str(value).strip()
        if not value:
            raise EditError("An object needs a name.")
        old = prim.name
        for port in project.ports:
            if port.sheet == old:
                port.sheet = value
    was = getattr(prim, field)
    setattr(prim, field, value)
    if field == "coordinate_system":
        _refuse_loops(project, lambda: setattr(prim, field, was))


def make_boolean(project: ModelerProject, op: BooleanOp, operand_ids: List[str]) -> str:
    """Replace the operands with one boolean solid, in the given order."""
    op = BooleanOp(op)
    operands = [find_primitive(project, i) for i in operand_ids]
    if any(o is None for o in operands):
        raise EditError("Some of the chosen objects no longer exist - nothing was combined.")
    if len(operands) < MIN_OPERANDS[op]:
        raise EditError(f"{BOOLEAN_LABELS[op]} needs at least {MIN_OPERANDS[op]} objects.")
    if any(o.kind == PrimitiveKind.SHEET for o in operands):  # type: ignore[union-attr]
        raise EditError("A sheet has no volume and cannot be an operand of a boolean.")
    _release_anchors(project, operand_ids)
    combined = Primitive(
        id=new_id(),
        name=unique_name(f"{op.value}_", (p.name for p in project.primitives)),
        kind=PrimitiveKind.BOOLEAN,
        boolean_op=op,
        material=operands[0].material,  # type: ignore[union-attr]
        priority=max(o.priority for o in operands),  # type: ignore[union-attr]
        # The first operand gives the result its look and solve state.
        model=operands[0].model, color=operands[0].color, opacity=operands[0].opacity,  # type: ignore[union-attr]
        children=[o.model_copy(deep=True) for o in operands],  # type: ignore[union-attr]
    )
    project.primitives = [p for p in project.primitives if p.id not in operand_ids]
    project.primitives.append(combined)
    return combined.id


def explode_boolean(project: ModelerProject, primitive_id: str) -> List[str]:
    """Split a boolean back into its operands, in place in the list."""
    idx = index_of(project.primitives, primitive_id)
    if idx < 0 or project.primitives[idx].kind != PrimitiveKind.BOOLEAN:
        raise EditError("Only a boolean can be exploded.")
    _release_anchors(project, [primitive_id])
    target = project.primitives[idx]
    taken = [p.name for p in project.primitives]
    kids = []
    for child in target.children:
        kid = child.model_copy(deep=True)
        kid.id = new_id()
        kid.name = unique_name(f"{child.name}_", taken)
        kid.material = child.material or target.material
        taken.append(kid.name)
        kids.append(kid)
    project.primitives[idx:idx + 1] = kids
    return [k.id for k in kids]


# --- ports ---

def add_port(project: ModelerProject, kind: PortKind) -> str:
    kind = PortKind(kind)
    s = suggest_scale(project)
    port = Port(
        id=new_id(),
        name=unique_name("port", (p.name for p in project.ports)),
        kind=kind,
        number=max((p.number for p in project.ports), default=0) + 1,
        p1=(0.0, 0.0, 0.0),
        p2=(s * 0.3, s * 0.05, s * 0.05) if kind == PortKind.MICROSTRIP else (0.0, 0.0, s * 0.1),
        direction="z",
        excite=not project.ports,
        prop_dir=None if kind == PortKind.LUMPED else "x",
        exc_dir="z" if kind == PortKind.MICROSTRIP else None,
        wg_a_mm=22.86 if kind == PortKind.RECT_WAVEGUIDE else None,
        wg_b_mm=10.16 if kind == PortKind.RECT_WAVEGUIDE else None,
        r_inner_mm=0.5 if kind == PortKind.COAXIAL else None,
        r_outer_mm=1.5 if kind == PortKind.COAXIAL else None,
        coordinate_system=project.working_cs,
    )
    project.ports.append(port)
    return port.id


def assign_port(project: ModelerProject, sheet_id: str, kind: PortKind) -> str:
    """Assign a port to a sheet and return its id. Lumped and microstrip ports then need an integration line."""
    kind = PortKind(kind)
    sheet = find_primitive(project, sheet_id)
    if sheet is None or sheet.kind != PrimitiveKind.SHEET:
        raise EditError("A port is assigned to a sheet. Draw a rectangle or a circle where the port goes, "
                        "select it, then assign the port.")
    port = Port(
        id=new_id(),
        name=unique_name("port", (p.name for p in project.ports)),
        kind=kind,
        number=max((p.number for p in project.ports), default=0) + 1,
        direction="z",
        excite=not project.ports,
        sheet=sheet.name,
        prop_dir=None if kind == PortKind.LUMPED else (sheet.normal_axis or "z"),
    )
    project.ports.append(port)
    return port.id


def set_port_line(project: ModelerProject, port_id: str, start, end) -> None:
    """Set or clear (None) a sheet port's integration line, in the sheet's coordinate system."""
    from EMShrimulator.core.em import sheets

    port = find_port(project, port_id)
    if port is None:
        raise EditError("That port no longer exists.")
    if sheets.sheet_by_name(project.primitives, port.sheet) is None:
        raise EditError("Only a port on a sheet has an integration line.")

    def tidy(point):
        if point is None:
            return None
        return tuple(float(f"{v:.10g}") if isinstance(v, (int, float)) else v for v in point)

    start, end = tidy(start), tidy(end)
    if start is not None and start == end:
        raise EditError("An integration line needs two different ends.")
    port.line_start, port.line_end = start, end


def delete_port(project: ModelerProject, port_id: str) -> None:
    before = len(project.ports)
    project.ports = [p for p in project.ports if p.id != port_id]
    if len(project.ports) == before:
        raise EditError("That port no longer exists.")


def duplicate_port(project: ModelerProject, port_id: str) -> str:
    """Duplicate a port under the next free number, not excited."""
    source = find_port(project, port_id)
    if source is None:
        raise EditError("That port no longer exists.")
    copy = source.model_copy(deep=True)
    copy.id = new_id()
    copy.name = unique_name(f"{source.name}_copy", (p.name for p in project.ports))
    copy.number = max((p.number for p in project.ports), default=0) + 1
    copy.excite = False
    project.ports.append(copy)
    return copy.id


def excite_only(project: ModelerProject, port_id: str) -> None:
    """Drive this port and no other."""
    if find_port(project, port_id) is None:
        raise EditError("That port no longer exists.")
    for port in project.ports:
        port.excite = port.id == port_id


def set_port_field(project: ModelerProject, port_id: str, field: str, value: Any) -> None:
    port = find_port(project, port_id)
    if port is None:
        raise EditError("That port no longer exists.")
    if field == "name" and not str(value).strip():
        raise EditError("A port needs a name.")
    setattr(port, field, value)


# --- materials ---

def random_colour() -> str:
    r, g, b = colorsys.hls_to_rgb(random.random(), 0.55, 0.45)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def add_material(project: ModelerProject) -> str:
    material = Material(
        name=unique_name("Material", (m.name for m in project.materials)),
        kind=MaterialKind.DIELECTRIC, color=random_colour(),
    )
    project.materials.append(material)
    return material.name


def set_material_field(project: ModelerProject, name: str, field: str, value: Any) -> None:
    material = find_material(project, name)
    if material is None:
        raise EditError(f"There is no material named '{name}'.")
    if field == "name":
        raise EditError("Use rename_material to rename a material.")
    setattr(material, field, value)


def rename_material(project: ModelerProject, old: str, new: str) -> None:
    """Rename a material and every reference, including boolean operands."""
    new = new.strip()
    material = find_material(project, old)
    if material is None:
        raise EditError(f"There is no material named '{old}'.")
    if not new or new == old:
        return
    if find_material(project, new) is not None:
        raise EditError(f"A material named '{new}' already exists.")
    material.name = new

    def retarget(items: List[Primitive]) -> None:
        for prim in items:
            if prim.material == old:
                prim.material = new
            retarget(prim.children)

    retarget(project.primitives)


def delete_material(project: ModelerProject, name: str) -> None:
    """Delete an unused material; boolean operands naming it take their boolean's material."""
    users = material_users(project, name)
    if users:
        shown = ", ".join(users[:3]) + (f" and {len(users) - 3} more" if len(users) > 3 else "")
        raise EditError(f"{shown} {'is' if len(users) == 1 else 'are'} made of '{name}'; give "
                        f"{'it' if len(users) == 1 else 'them'} another material first.")

    def retarget(items: List[Primitive], material: str) -> None:
        for prim in items:
            if prim.material == name:
                prim.material = material
            retarget(prim.children, material)

    for prim in project.primitives:
        retarget(prim.children, prim.material)
    project.materials = [m for m in project.materials if m.name != name]


# --- parameters ---

IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")


def add_parameter(project: ModelerProject) -> str:
    name = unique_name("var", (p.name for p in project.parameters))
    project.parameters.append(Parameter(name=name, expression="1"))
    return name


def check_parameter_name(project: ModelerProject, name: str) -> None:
    """Refuse a name a new or renamed variable cannot take."""
    if not IDENTIFIER.match(name):
        raise EditError("A variable name uses letters, digits and underscore, starting with a letter.")
    if name in FUNCTIONS:
        raise EditError(f"'{name}' is a built-in function; pick another name for the variable.")
    if any(p.name == name for p in project.parameters):
        raise EditError(f"A variable named '{name}' already exists.")


def define_parameters(project: ModelerProject, definitions: Iterable[Parameter]) -> List[str]:
    """Add several variables at once; none are added if one is refused."""
    definitions = list(definitions)
    names = [d.name for d in definitions]
    for i, name in enumerate(names):
        check_parameter_name(project, name)
        if name in names[:i]:
            raise EditError(f"'{name}' is defined twice.")
    project.parameters.extend(d.model_copy() for d in definitions)
    return names


def set_parameter_field(project: ModelerProject, name: str, field: str, value: Any) -> None:
    param = next((p for p in project.parameters if p.name == name), None)
    if param is None:
        raise EditError(f"There is no parameter named '{name}'.")
    if field == "name":
        raise EditError("Use rename_parameter to rename a parameter.")
    setattr(param, field, value)


def delete_parameter(project: ModelerProject, name: str) -> None:
    project.parameters = [p for p in project.parameters if p.name != name]


def _swap_names(value: Any, pattern: re.Pattern, new: str) -> Any:
    if isinstance(value, str):
        return pattern.sub(new, value)
    if isinstance(value, tuple):
        return tuple(_swap_names(v, pattern, new) for v in value)
    if isinstance(value, list):
        return [_swap_names(v, pattern, new) for v in value]
    return value


def _rewrite_expressions(node: Any, pattern: re.Pattern, new: str) -> None:
    """Rewrite a name in every expression field of a model tree, in place."""
    from pydantic import BaseModel

    from EMShrimulator.core.em.model import EXPR_FIELDS

    if isinstance(node, BaseModel):
        fields = EXPR_FIELDS.get(type(node), set())
        for name in type(node).model_fields:
            value = getattr(node, name, None)
            if value is None:
                continue
            if name in fields:
                setattr(node, name, _swap_names(value, pattern, new))
            else:
                _rewrite_expressions(value, pattern, new)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _rewrite_expressions(item, pattern, new)


def rename_parameter(project: ModelerProject, old: str, new: str) -> None:
    """Rename a variable and rewrite every expression that reads it."""
    new = new.strip()
    if new == old:
        return
    check_parameter_name(project, new)
    param = next((p for p in project.parameters if p.name == old), None)
    if param is None:
        raise EditError(f"There is no parameter named '{old}'.")

    # Skip function calls: a variable named like a function must leave `db(x)` alone.
    pattern = re.compile(rf"\b{re.escape(old)}\b(?!\s*\()")
    param.name = new
    for p in project.parameters:
        p.expression = pattern.sub(new, p.expression)
    _rewrite_expressions(project.primitives, pattern, new)
    _rewrite_expressions(project.ports, pattern, new)
    _rewrite_expressions(project.materials, pattern, new)
    _rewrite_expressions(project.dumps, pattern, new)
    _rewrite_expressions([project.mesh, project.frequency, project.simulation, project.nf2ff,
                          *project.coordinate_systems], pattern, new)
    for sweep in project.sweeps:
        for variable in sweep.variables:
            if variable.parameter == old:
                variable.parameter = new


# --- sweeps ---

def add_sweep(project: ModelerProject) -> int:
    if not project.parameters:
        raise EditError("Define a variable first - a sweep varies a parameter, not a raw dimension.")
    name = f"Sweep {len(project.sweeps) + 1}"
    while any(s.name == name for s in project.sweeps):
        name = name + "'"
    project.sweeps.append(ParameterSweep(
        name=name,
        variables=[SweepVariable(parameter=project.parameters[0].name, start=1.0, stop=2.0, points=3)],
    ))
    return len(project.sweeps) - 1


def add_sweep_variable(project: ModelerProject, sweep_index: int) -> None:
    if not project.parameters:
        raise EditError("There are no variables to sweep.")
    project.sweeps[sweep_index].variables.append(
        SweepVariable(parameter=project.parameters[0].name, start=1.0, stop=2.0, points=3))


# --- field dumps ---

def add_dump(project: ModelerProject) -> str:
    s = suggest_scale(project)
    dump = FieldDump(
        id=new_id(), name=unique_name("dump", (d.name for d in project.dumps)),
        p1=(-s, 0.0, -s), p2=(s, 0.0, s),
    )
    project.dumps.append(dump)
    return dump.id


def delete_dump(project: ModelerProject, dump_id: str) -> None:
    project.dumps = [d for d in project.dumps if d.id != dump_id]


def set_dump_field(project: ModelerProject, dump_id: str, field: str, value: Any) -> None:
    dump = find_dump(project, dump_id)
    if dump is None:
        raise EditError("That field dump no longer exists.")
    setattr(dump, field, value)


# --- coordinate systems ---

def find_coordinate_system(project: ModelerProject, name: str) -> Optional[CoordinateSystem]:
    return next((cs for cs in project.coordinate_systems if cs.name == name), None)


def placed_in(project: ModelerProject, name: str) -> List[str]:
    """Everything a coordinate system places: objects, ports and child systems."""
    found: List[str] = []

    def walk(items: Iterable[Primitive]) -> None:
        for prim in items:
            if prim.coordinate_system == name:
                found.append(prim.name)
            walk(prim.children)

    walk(project.primitives)
    found += [p.name for p in project.ports if p.coordinate_system == name]
    found += [cs.name for cs in project.coordinate_systems if cs.parent == name]
    return found


CS_MODES = ("offset", "rotated", "both")
CS_FIELDS = frozenset({"parent", "mode", "origin", "angles", "object_id", "anchor", "anchor_at"})
ANCHORS = ("centre", "face", "edge", "corner")


def _check_mode(mode: str) -> None:
    if mode not in CS_MODES:
        raise EditError(f"'{mode}' is not a way to move a coordinate system: offset, rotated or both.")


def add_coordinate_system(project: ModelerProject, parent: Optional[str] = None, mode: str = "offset",
                          origin=(0.0, 0.0, 0.0), angles=(0.0, 0.0, 0.0), make_working: bool = False) -> str:
    """Add a relative coordinate system based on `parent` (None = global) and return its name."""
    if parent is not None and find_coordinate_system(project, parent) is None:
        raise EditError(f"There is no coordinate system named '{parent}'.")
    _check_mode(mode)
    name = unique_name("CS", (cs.name for cs in project.coordinate_systems))
    project.coordinate_systems.append(CoordinateSystem(name=name, parent=parent, mode=mode, origin=tuple(origin),
                                                       angles=tuple(angles)))
    if make_working:
        project.working_cs = name
    return name


def _anchor_object(project: ModelerProject, object_id: Optional[str], anchor: str) -> Primitive:
    prim = find_primitive(project, object_id) if object_id else None
    if prim is None:
        raise EditError("Choose the object first: an object coordinate system sits on one of its points - its "
                        "centre, a face centre, an edge centre or a corner.")
    if anchor not in ANCHORS:
        raise EditError(f"'{anchor}' is not a point a coordinate system can sit on.")
    return prim


def add_object_coordinate_system(project: ModelerProject, object_id: str, anchor: str, at=None,
                                 mode: str = "offset", make_working: bool = False) -> str:
    """Add a coordinate system on a point of an object that follows it; `at` picks the point. Returns its name."""
    _anchor_object(project, object_id, anchor)
    _check_mode(mode)
    name = unique_name("CS", (cs.name for cs in project.coordinate_systems))
    project.coordinate_systems.append(CoordinateSystem(
        name=name, kind="object", object_id=object_id, anchor=anchor,
        anchor_at=tuple(float(v) for v in at) if at is not None else None, mode=mode))
    _refuse_loops(project, project.coordinate_systems.pop)
    if make_working:
        project.working_cs = name
    return name


def place_coordinate_system(project: ModelerProject, name: str, object_id: str, anchor: str, at=None) -> None:
    """Move a system onto another object point, keeping its offset and rotation."""
    cs = find_coordinate_system(project, name)
    if cs is None:
        raise EditError(f"There is no coordinate system named '{name}'.")
    _anchor_object(project, object_id, anchor)
    was = cs.model_copy()
    cs.kind, cs.parent, cs.object_id, cs.anchor = "object", None, object_id, anchor
    cs.anchor_at = tuple(float(v) for v in at) if at is not None else None
    _refuse_loops(project, lambda: [setattr(cs, f, getattr(was, f)) for f in ("kind", "parent", "object_id",
                                                                              "anchor", "anchor_at")])


def dependency_loop(project: ModelerProject) -> Optional[str]:
    """A dependency loop of coordinate systems and objects, described in words; None if there is none."""
    from EMShrimulator.core.em.frames import systems_of

    by_name = {cs.name: cs for cs in project.coordinate_systems}
    prims = {p.id: p for p in project.primitives}

    def after(node):
        kind, key = node
        if kind == "object":
            return [("cs", n) for n in sorted(systems_of(prims[key])) if n in by_name]
        cs = by_name[key]
        if cs.kind == "object":
            return [("object", cs.object_id)] if cs.object_id in prims else []
        return [("cs", cs.parent)] if cs.parent in by_name else []

    def label(node) -> str:
        return f"'{prims[node[1]].name}'" if node[0] == "object" else f"'{node[1]}'"

    def say(a, b) -> str:
        if a[0] == "object":
            return f"{label(a)} is placed in {label(b)}"
        if b[0] == "object":
            return f"{label(a)} sits on {label(b)}"
        return f"{label(a)} is based on {label(b)}"

    done, path = set(), []

    def walk(node) -> Optional[str]:
        if node in path:
            loop = path[path.index(node):] + [node]
            return ", ".join(say(a, b) for a, b in zip(loop, loop[1:]))
        if node in done:
            return None
        path.append(node)
        for nxt in after(node):
            found = walk(nxt)
            if found:
                return found
        path.pop()
        done.add(node)
        return None

    for name in by_name:
        found = walk(("cs", name))
        if found:
            return found
    return None


def _refuse_loops(project: ModelerProject, undo: Callable[[], None]) -> None:
    """Take the edit just made back with `undo`, and refuse it, if it made a loop."""
    loop = dependency_loop(project)
    if loop:
        undo()
        raise EditError(f"That would make a loop - {loop} - and nothing in it could be placed.")


def _release_anchors(project: ModelerProject, object_ids: Iterable[str]) -> List[str]:
    """Re-base systems on objects about to be removed onto global, in place. Returns their names."""
    gone = set(object_ids)
    held = [cs for cs in project.coordinate_systems if cs.kind == "object" and cs.object_id in gone]
    if not held:
        return []
    from EMShrimulator.core.em import params, transforms

    resolved = params.resolve(project)[0]
    by_name = {cs.name: cs for cs in resolved.coordinate_systems}
    for cs in held:
        origin, angles = transforms.offset_and_angles(transforms.frame_matrix(cs.name, by_name))
        turned, moved = any(angles), any(origin)
        cs.kind, cs.parent, cs.object_id, cs.anchor_at = "relative", None, None, None
        cs.mode = "both" if turned and moved else "rotated" if turned else "offset"
        cs.origin, cs.angles = origin, angles
    return [cs.name for cs in held]


def delete_coordinate_system(project: ModelerProject, name: str) -> None:
    if find_coordinate_system(project, name) is None:
        raise EditError(f"There is no coordinate system named '{name}'.")
    users = placed_in(project, name)
    if users:
        listed = ", ".join(users[:3]) + ("…" if len(users) > 3 else "")
        raise EditError(f"'{name}' still places {listed}. Move them to another coordinate system first.")
    project.coordinate_systems = [cs for cs in project.coordinate_systems if cs.name != name]
    if project.working_cs == name:
        project.working_cs = None


def rename_coordinate_system(project: ModelerProject, old: str, new: str) -> None:
    """Rename a system and every reference to it."""
    new = new.strip()
    cs = find_coordinate_system(project, old)
    if cs is None:
        raise EditError(f"There is no coordinate system named '{old}'.")
    if not new or new == old:
        return
    if find_coordinate_system(project, new) is not None:
        raise EditError(f"A coordinate system named '{new}' already exists.")
    cs.name = new

    def retarget(items: List[Primitive]) -> None:
        for prim in items:
            if prim.coordinate_system == old:
                prim.coordinate_system = new
            retarget(prim.children)

    retarget(project.primitives)
    for port in project.ports:
        if port.coordinate_system == old:
            port.coordinate_system = new
    for other in project.coordinate_systems:
        if other.parent == old:
            other.parent = new
    if project.working_cs == old:
        project.working_cs = new


def set_coordinate_system_field(project: ModelerProject, name: str, field: str, value: Any) -> None:
    cs = find_coordinate_system(project, name)
    if cs is None:
        raise EditError(f"There is no coordinate system named '{name}'.")
    if field == "name":
        raise EditError("Use rename_coordinate_system to rename a coordinate system.")
    if field not in CS_FIELDS:
        raise EditError(f"'{field}' is not a setting of a coordinate system.")
    if field == "parent":
        value = value or None
        if value == name:
            raise EditError("A coordinate system cannot be based on itself.")
        if value is not None and find_coordinate_system(project, value) is None:
            raise EditError(f"There is no coordinate system named '{value}'.")
    elif field == "mode":
        _check_mode(value)
    elif field in ("object_id", "anchor"):
        _anchor_object(project, value if field == "object_id" else cs.object_id,
                       value if field == "anchor" else cs.anchor)
    was = getattr(cs, field)
    setattr(cs, field, value)
    _refuse_loops(project, lambda: setattr(cs, field, was))


def set_working_cs(project: ModelerProject, name: Optional[str]) -> None:
    """The system new objects and ports are drawn in, and where the viewport grid lies."""
    if name and find_coordinate_system(project, name) is None:
        raise EditError(f"There is no coordinate system named '{name}'.")
    project.working_cs = name or None


# --- result plots ---

# These work on any project with a `plots` list.

def find_plot(project: ModelerProject, plot_id: str) -> Optional[ResultPlot]:
    return next((p for p in project.plots if p.id == plot_id), None)


def add_plot(project: ModelerProject, kind: str, traces: Optional[List[str]] = None) -> str:
    """A new report from a preset, or of a report kind on its first display."""
    if kind in PLOT_PRESETS:
        label, fields = PLOT_PRESETS[kind]
    elif kind in REPORT_LABELS:
        label, fields = REPORT_LABELS[kind], {"kind": kind, "display": DISPLAYS[kind][0]}
    else:
        raise EditError(f"'{kind}' is not a kind of plot.")
    if fields["kind"] in ("y", "z") and "function" not in fields:
        fields = {**fields, "function": "re_im"}
    plot = ResultPlot(id=new_id(), name=unique_name(f"{label} ", (p.name for p in project.plots)),
                      traces=list(traces or []), **fields)
    project.plots.append(plot)
    return plot.id


def delete_plot(project: ModelerProject, plot_id: str) -> None:
    before = len(project.plots)
    project.plots = [p for p in project.plots if p.id != plot_id]
    if len(project.plots) == before:
        raise EditError("That plot no longer exists.")


def set_plot_field(project: ModelerProject, plot_id: str, field: str, value: Any) -> None:
    plot = find_plot(project, plot_id)
    if plot is None:
        raise EditError("That plot no longer exists.")
    if field == "name":
        value = str(value).strip()
        if not value:
            raise EditError("A plot needs a name.")
    elif field == "display" and value not in DISPLAYS.get(plot.kind, ()):
        raise EditError(f"A {REPORT_LABELS.get(plot.kind, plot.kind)} report cannot be drawn as "
                        f"{DISPLAY_LABELS.get(value, value)}.")
    elif field not in ResultPlot.model_fields or field in ("id", "kind"):
        raise EditError(f"'{field}' is not a plot setting.")
    # Through the schema, so wrong values are refused.
    try:
        checked = ResultPlot.model_validate({**plot.model_dump(), field: value})
    except ValueError as exc:
        raise EditError(f"That is not a valid {field.replace('_', ' ')}: {exc}") from exc
    setattr(plot, field, getattr(checked, field))


# --- whole-design settings ---

def set_path(project: ModelerProject, path: str, value: Any) -> None:
    """Assign a singleton design setting by dotted path, e.g. `mesh.smooth_ratio`."""
    parts = path.split(".")
    if parts[0] not in {"frequency", "mesh", "boundary", "simulation", "nf2ff", "display_units", "description"}:
        raise EditError(f"'{path}' is not a design setting.")
    node: Any = project
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def get_path(project: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path (attributes and list indices) out of a project."""
    node: Any = project
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, (list, tuple)):
            try:
                node = node[int(part)]
            except (ValueError, IndexError):
                return default
        else:
            node = getattr(node, part, default)
    return node


Mutation = Callable[[ModelerProject], Any]
