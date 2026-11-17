"""Pure checks behind the modeler workspace: no openEMS, no window.

Run with `.venv/bin/python -m pytest -q tests/test_modeler_workspace.py`.
"""
import sys

import numpy as np
import pytest

from EMShrimulator.app.views.modeler.point_size_edit import axis_note, corner_from_size, size_of
from EMShrimulator.core.em import edits, params, sheets
from EMShrimulator.core.em.model import (
    CoordinateSystem, Material, MaterialKind, ModelerProject, Parameter, Port, PortKind, Primitive, PrimitiveKind,
    SheetShape, TransformOp,
)
from EMShrimulator.core.em.results import impedance_traces
from EMShrimulator.core.em.run_worker import add_gain, max_delta_s, refined
from EMShrimulator.core.em.scene import WorkingFrame, working_frame, working_grid
from EMShrimulator.core.errors import EditError


# --- the size field shows what was typed ------------------------------------

@pytest.mark.parametrize("start, size", [
    ("-Wp/2", "Wp"), ("-Wp", "Wp*2"), ("h", "t"), ("x0 + 1", "L - 2*g"), (1.5, "Wp"), ("Wp/2", 10.0),
    (0.0, "Wp"), (2.0, 3.0), ("a", "(b + c)"),
])
def test_size_round_trips(start, size):
    shown = size_of(start, corner_from_size(start, size))
    assert shown == (pytest.approx(size) if isinstance(size, float) else size)


def test_size_of_projects_written_before():
    # The patch in the bundled example project, exactly as stored.
    assert size_of("-Wp/2", "(-Wp/2) + (Wp)") == "Wp"
    assert size_of("h", "(h) + (t)") == "t"
    assert size_of("-Wp", "(-Wp) + (Wp*2)") == "Wp*2"


def test_size_of_an_independent_corner_is_not_guessed():
    # A far corner with its own formula shows the resolved number.
    assert size_of("-pad", "x_end + pad") is None
    # Only one pair of brackets enclosing the whole size comes off.
    assert size_of("a", "(a) + (b) + (c)") == "(b) + (c)"


def test_size_corner_evaluates_to_start_plus_size():
    values = {"Wp": 5.0, "t": 0.035}
    corner = corner_from_size("-Wp/2", "Wp")
    assert params.evaluate(corner, values) == pytest.approx(2.5)
    corner = corner_from_size("-Wp/2", 10.0)
    assert params.evaluate(corner, values) == pytest.approx(7.5)


def test_cylinder_note_describes_its_axis():
    # A cylinder along z is a solid; its note is the axis.
    assert axis_note((0.0, 0.0, 0.01), 1e-3, "mm") == "axis 10 mm along +z"
    assert axis_note((-0.002, 0.0, 0.0), 1e-3, "mm") == "axis 2 mm along −x"
    assert axis_note((0.003, 0.004, 0.0), 1e-3, "mm") == "axis 5 mm"
    assert axis_note((0.0, 0.0, 0.0), 1e-3, "mm").startswith("zero length")


# --- variables defined on first use ---------------------------------------------

def test_undefined_names_in_reading_order():
    # Constants, functions and known variables are never asked for.
    assert params.undefined_names("2*Wp + Lp - pi + sin(a) + Wp", {"Lp"}) == ["Wp", "a"]
    assert params.undefined_names("40mil + 2.45GHz", ()) == []


def test_check_syntax_ignores_unknown_names():
    params.check_syntax("2*Wp + undefined_yet")
    with pytest.raises(params.ExpressionError):
        params.check_syntax("2*")
    with pytest.raises(params.ExpressionError):
        params.check_syntax("   ")


def test_define_parameters_adds_all_or_none():
    p = _project()
    p.parameters.append(Parameter(name="h", expression="1.5", unit="mm"))
    assert edits.define_parameters(p, [Parameter(name="Wp", expression="40"),
                                       Parameter(name="Wsub", expression="2*Wp")]) == ["Wp", "Wsub"]
    assert [x.name for x in p.parameters] == ["h", "Wp", "Wsub"]
    for refused in ([Parameter(name="a"), Parameter(name="h")],      # already exists
                    [Parameter(name="b"), Parameter(name="b")],      # twice in one go
                    [Parameter(name="c"), Parameter(name="sqrt")],   # a function
                    [Parameter(name="d"), Parameter(name="2x")]):    # not a name
        with pytest.raises(EditError):
            edits.define_parameters(p, refused)
    assert [x.name for x in p.parameters] == ["h", "Wp", "Wsub"]


# --- ports made from sheets, HFSS style ---------------------------------------------

def _sheet_project(normal="x", size=(3.0, 1.5), origin=(0.0, 0.0, 0.75), cs=None):
    """An upright rectangle between ground at z = 0 and a strip at z = 1.5."""
    p = _project()
    sheet_id = edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RECTANGLE)
    sheet = edits.find_primitive(p, sheet_id)
    sheet.normal_axis, sheet.sheet_size, sheet.origin, sheet.coordinate_system = normal, size, origin, cs
    return p, sheet


def test_drawn_sheets_are_named_by_shape():
    p = _project()
    a = edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RECTANGLE)
    b = edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.CIRCLE)
    assert [edits.find_primitive(p, i).name for i in (a, b)] == ["rectangle_1", "circle_1"]
    assert edits.find_primitive(p, b).sheet_shape == SheetShape.CIRCLE
    assert not edits.find_primitive(p, a).material  # construction: nothing is built until it is a port


def test_assigning_a_port_draws_no_line():
    p, sheet = _sheet_project()
    lumped = edits.find_port(p, edits.assign_port(p, sheet.id, PortKind.LUMPED))
    assert lumped.sheet == sheet.name and lumped.excite
    assert lumped.line_start is None and lumped.line_end is None  # the user draws it
    msl = edits.find_port(p, edits.assign_port(p, sheet.id, PortKind.MICROSTRIP))
    assert msl.number == 2 and not msl.excite
    box = edits.add_primitive(p, PrimitiveKind.BOX)
    with pytest.raises(EditError):
        edits.assign_port(p, box, PortKind.LUMPED)


def test_across_a_rectangle_the_whole_sheet_is_the_port():
    p, sheet = _sheet_project()
    port = edits.find_port(p, edits.assign_port(p, sheet.id, PortKind.LUMPED))
    edits.set_port_line(p, port.id, (0.0, 0.0, 0.0), (0.0, 0.0, 1.5))  # ground edge up to the strip edge
    placed = sheets.port_geometry(p, port, {})
    # stop is the + terminal, at the strip.
    assert placed.exc_axis == "z" and not placed.along_line
    assert np.allclose(placed.start, [0, -1.5, 0]) and np.allclose(placed.stop, [0, 1.5, 1.5])
    edits.set_port_line(p, port.id, (0.0, 0.0, 1.5), (0.0, 0.0, 0.0))
    placed = sheets.port_geometry(p, port, {})
    assert placed.start[2] == 1.5 and placed.stop[2] == 0.0


def test_any_other_line_is_a_source_along_itself():
    """A probe feed: a clearance in the ground, the line from its edge to the pin."""
    p = _project()
    clearance = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.CIRCLE))
    clearance.radius, clearance.origin = 1.2, (5.0, 3.0, 0.0)
    port = edits.find_port(p, edits.assign_port(p, clearance.id, PortKind.LUMPED))
    edits.set_port_line(p, port.id, (6.2, 3.0, 0.0), (5.0, 3.0, 0.0))
    placed = sheets.port_geometry(p, port, {})
    assert placed.along_line and placed.exc_axis == "x"
    assert np.allclose(placed.start, [6.2, 3, 0]) and np.allclose(placed.stop, [5, 3, 0])  # + at the pin
    # Halfway across a rectangle is not edge to edge either.
    p2, sheet = _sheet_project()
    half = edits.find_port(p2, edits.assign_port(p2, sheet.id, PortKind.LUMPED))
    edits.set_port_line(p2, half.id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.75))
    assert sheets.port_geometry(p2, half, {}).along_line


def test_a_port_without_a_line_still_previews():
    p, sheet = _sheet_project()
    p.ports.append(Port(name="old", kind=PortKind.LUMPED, number=1, direction="y", sheet=sheet.name))
    placed = sheets.port_geometry(p, p.ports[0], {})
    assert placed.exc_axis is None and placed.line is None  # the builder falls back to `direction`


def test_microstrip_metal_goes_at_the_strip_and_launches_along_the_extrusion():
    p, sheet = _sheet_project()
    port = edits.find_port(p, edits.assign_port(p, sheet.id, PortKind.MICROSTRIP))
    edits.set_port_line(p, port.id, (0.0, 0.0, 0.0), (0.0, 0.0, 1.5))
    port.sheet_extrude_mm = -2.0  # a port at the far end of a line looks back along -x
    placed = sheets.port_geometry(p, port, {}, 0.1)
    assert placed.prop_axis == "x" and placed.exc_axis == "z" and not placed.along_line
    assert placed.start[0] == 0.0 and placed.stop[0] == -2.0  # start on the sheet, stop down the line
    assert placed.start[2] == 1.5 and placed.stop[2] == 0.0  # openEMS puts the strip at start


def test_line_turns_with_its_coordinate_system():
    p, sheet = _sheet_project(normal="z", size=(0.5, 3.0), origin=(0.0, 0.0, 0.0), cs="CS1")
    p.coordinate_systems.append(CoordinateSystem(name="CS1", transforms=[
        TransformOp(kind="rotate_axis", axis="z", angle_deg=90.0)]))
    port = edits.find_port(p, edits.assign_port(p, sheet.id, PortKind.LUMPED))
    edits.set_port_line(p, port.id, (-0.25, 0.0, 0.0), (0.25, 0.0, 0.0))  # across the gap, in CS1
    cs = {c.name: c for c in p.coordinate_systems}
    placed = sheets.port_geometry(p, port, cs)
    assert placed.exc_axis == "y" and not placed.along_line  # local x, turned a quarter about z
    assert np.allclose(placed.line, [[0, -0.25, 0], [0, 0.25, 0]], atol=1e-9)


def test_snap_points_follow_the_shape():
    p, rect = _sheet_project()
    assert len(sheets.snap_points(rect)) == 9  # centre, four corners, four edge midpoints
    ring = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RING))
    ring.radius, ring.radius_inner = 2.0, 0.5
    points = sheets.snap_points(ring)
    assert len(points) == 9 and np.allclose(points[0], [0, 0, 0])
    assert {round(float(np.hypot(x, y)), 6) for x, y, _ in points[1:]} == {2.0, 0.5}


def test_deleting_a_sheet_takes_its_port():
    p, sheet = _sheet_project()
    edits.assign_port(p, sheet.id, PortKind.LUMPED)
    other = edits.add_primitive(p, PrimitiveKind.BOX)
    edits.delete_primitive(p, other)
    assert len(p.ports) == 1
    edits.delete_primitive(p, sheet.id)
    assert p.ports == []


def test_setting_a_line():
    p, sheet = _sheet_project()
    port_id = edits.assign_port(p, sheet.id, PortKind.LUMPED)
    with pytest.raises(EditError):
        edits.set_port_line(p, port_id, (0.0, 0.0, 0.5), (0.0, 0.0, 0.5))
    edits.set_port_line(p, port_id, (0.1 + 0.2, 0.0, 0.0), ("feed_x", 0.0, 1.5))  # drawn noise, a typed name
    port = edits.find_port(p, port_id)
    assert port.line_start == (0.3, 0.0, 0.0) and port.line_end == ("feed_x", 0.0, 1.5)
    edits.set_port_line(p, port_id, None, None)
    assert port.line_start is None and port.line_end is None
    box_port = edits.add_port(p, PortKind.LUMPED)
    with pytest.raises(EditError):
        edits.set_port_line(p, box_port, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def test_a_typed_line_follows_its_variables():
    p, sheet = _sheet_project()
    p.parameters.append(Parameter(name="h", expression="1.5"))
    port_id = edits.assign_port(p, sheet.id, PortKind.LUMPED)
    edits.set_port_line(p, port_id, (0.0, 0.0, 0.0), (0.0, 0.0, "h"))
    resolved = params.resolve(p)[0]
    assert sheets.line_ends(edits.find_port(resolved, port_id))[1][2] == 1.5
    edits.rename_parameter(p, "h", "sub_h")
    assert edits.find_port(p, port_id).line_end == (0.0, 0.0, "sub_h")


def test_validation_reads_the_line():
    from EMShrimulator.core.em.validate import validate

    p, sheet = _sheet_project()
    port_id = edits.assign_port(p, sheet.id, PortKind.LUMPED)

    def codes():
        return {issue["code"] for issue in validate(p)["issues"] if issue["code"].startswith("port")}

    assert codes() == {"port-no-line"} and not validate(p)["can_run"]
    edits.set_port_line(p, port_id, (0.0, 0.0, 0.0), (0.0, 0.0, 1.5))
    assert codes() == set()
    edits.set_port_line(p, port_id, (0.0, -1.0, 0.0), (0.0, 1.0, 1.5))
    assert codes() == {"port-line-diagonal"}
    edits.set_port_line(p, port_id, (0.0, 0.0, 0.0), (0.0, 0.0, 4.0))
    assert codes() == {"port-line-off-sheet"}
    edits.set_port_line(p, port_id, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert codes() == {"port-line-off-plane"}


# --- the session -------------------------------------------------------------

def test_closing_an_unsaved_session_raises_nothing(monkeypatch):
    # The undo stack emits cleanChanged as it is destroyed.
    import shiboken6
    from PySide6.QtCore import QCoreApplication

    from EMShrimulator.app import commands as cmd
    from EMShrimulator.app.session import ModelerSession

    _app = QCoreApplication.instance() or QCoreApplication([])
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(exc))
    session = ModelerSession("t", _project())
    session.push(cmd.AddPrimitive(session, PrimitiveKind.BOX))
    assert session.dirty
    shiboken6.delete(session)
    assert errors == []


# --- coordinate systems ------------------------------------------------------

def _project():
    return ModelerProject(name="t", materials=[Material(name="PEC", kind=MaterialKind.PEC)])


def test_coordinate_system_lifecycle():
    p = _project()
    name = edits.add_coordinate_system(p)
    cs = p.coordinate_systems[0]
    assert name == "CS1" and cs.kind == "relative" and cs.mode == "offset" and cs.origin == (0.0, 0.0, 0.0)
    assert p.working_cs is None  # only the UI's command makes a new system the working one
    edits.set_working_cs(p, name)
    prim_id = edits.add_primitive(p, PrimitiveKind.BOX)
    port_id = edits.add_port(p, "lumped")
    assert edits.find_primitive(p, prim_id).coordinate_system == "CS1"
    assert edits.find_port(p, port_id).coordinate_system == "CS1"
    with pytest.raises(EditError):
        edits.delete_coordinate_system(p, name)  # it still places the box and the port
    edits.rename_coordinate_system(p, "CS1", "Feed")
    assert p.working_cs == "Feed"
    assert edits.find_primitive(p, prim_id).coordinate_system == "Feed"
    assert edits.find_port(p, port_id).coordinate_system == "Feed"


def test_coordinate_system_cannot_nest_in_itself():
    p = _project()
    a = edits.add_coordinate_system(p)
    b = edits.add_coordinate_system(p, parent=a)
    with pytest.raises(EditError):
        edits.set_coordinate_system_field(p, a, "parent", b)
    edits.set_coordinate_system_field(p, b, "parent", "")
    assert edits.find_coordinate_system(p, b).parent is None


def test_working_frame_follows_the_system():
    p = _project()
    p.coordinate_systems.append(CoordinateSystem(name="CS1", transforms=[
        TransformOp(kind="rotate_axis", axis="z", angle_deg=90.0),
        TransformOp(kind="translate", translate=(10.0, 0.0, 2.0))]))
    p.working_cs = "CS1"
    frame = working_frame(p)
    assert np.allclose(frame.origin, [10.0, 0.0, 2.0])
    assert np.allclose(frame.axes[0], [0.0, 1.0, 0.0], atol=1e-9)  # x turned onto y
    assert np.allclose(frame.axes[2], [0.0, 0.0, 1.0])


def test_working_grid_contains_the_origin_and_the_design():
    frame = WorkingFrame(np.array([3.0, -2.0, 1.0]), np.eye(3))
    grid = working_grid(frame, (np.array([10.0, 10.0, 0.0]), np.array([30.0, 20.0, 5.0])))
    x_axis, y_axis = grid["x_axis"][0], grid["y_axis"][0]
    assert np.allclose(x_axis[:, 1], -2.0) and np.allclose(x_axis[:, 2], 1.0)  # through the origin
    assert np.allclose(y_axis[:, 0], 3.0)
    assert x_axis[:, 0].min() <= 3.0 <= x_axis[:, 0].max() and x_axis[:, 0].max() >= 30.0
    assert y_axis[:, 1].max() >= 20.0
    major = grid["major"]
    assert len(major) and np.allclose(major[:, :, 2], 1.0)  # every line on the working plane
    # Major lines fall on whole multiples of the major step from the origin.
    step = grid["major_step"]
    offsets = np.concatenate([major[:, :, 0].ravel() - 3.0, major[:, :, 1].ravel() + 2.0]) / step
    assert np.allclose(offsets, np.round(offsets), atol=1e-6)


def test_grid_without_a_design_is_still_drawn():
    grid = working_grid(WorkingFrame.global_frame(), None)
    assert len(grid["minor"]) > 0 and len(grid["x_axis"]) == 1


# --- plots, ports --------------------------------------------------------------

def test_plots_are_named_and_edited():
    p = _project()
    first = edits.add_plot(p, "s_db")
    second = edits.add_plot(p, "s_db", ["S21"])
    assert [x.name for x in p.plots] == ["S-parameters (dB) 1", "S-parameters (dB) 2"]
    assert edits.find_plot(p, second).traces == ["S21"]
    edits.set_plot_field(p, first, "quantity", "gain")
    with pytest.raises(EditError):
        edits.set_plot_field(p, first, "name", "  ")
    with pytest.raises(EditError):
        edits.add_plot(p, "nonsense")
    edits.delete_plot(p, first)
    assert [x.id for x in p.plots] == [second]


def test_duplicate_port_is_numbered_and_passive():
    p = _project()
    a = edits.add_port(p, "lumped")
    b = edits.duplicate_port(p, a)
    assert edits.find_port(p, a).excite and not edits.find_port(p, b).excite
    assert edits.find_port(p, b).number == 2
    edits.excite_only(p, b)
    assert [x.excite for x in p.ports] == [False, True]


def test_old_projects_load_with_the_new_fields():
    raw = ('{"schema_version": 1, "name": "old", "simulation": {"max_timesteps": 5000}, '
           '"primitives": [{"name": "b", "kind": "box", "material": "PEC"}]}')
    p = ModelerProject.model_validate_json(raw)
    assert p.plots == [] and p.working_cs is None
    assert p.simulation.max_passes == 1 and p.simulation.max_timesteps == 5000
    b = p.primitives[0]
    assert b.model and b.color is None and b.opacity is None


# --- per-object look, non-model objects ----------------------------------------

def _two_boxes():
    p = ModelerProject(name="t", materials=[Material(name="PEC", kind=MaterialKind.PEC, color="#c9ced6"),
                                            Material(name="FR4", color="#2f7a3f", opacity=0.55)])
    p.primitives = [Primitive(name="a", kind=PrimitiveKind.BOX, material="PEC", p1=(0, 0, 0), p2=(1, 1, 1)),
                    Primitive(name="b", kind=PrimitiveKind.BOX, material="PEC", p1=(2, 0, 0), p2=(3, 1, 1))]
    return p


def test_an_object_can_override_its_materials_look():
    from EMShrimulator.core.em.frames import geometry_key
    from EMShrimulator.core.em.scene import appearance, build_scene

    p = _two_boxes()
    a, b = p.primitives
    materials = {m.name: m for m in p.materials}
    assert appearance(a, materials) == ("#c9ced6", 0.85)
    key = geometry_key(a, {})
    edits.set_primitive_field(p, a.id, "opacity", 0.3)
    assert appearance(a, materials) == ("#c9ced6", 0.3)  # its own opacity, the material's colour
    edits.set_primitive_field(p, a.id, "color", "#ff0000")
    edits.set_primitive_field(p, a.id, "model", False)
    assert geometry_key(a, {}) == key  # look and Model flag never re-tessellate
    scene = build_scene(p)
    assert (scene.solids[a.id].color, scene.solids[a.id].opacity, scene.solids[a.id].model) == ("#ff0000", 0.3, False)
    assert (scene.solids[b.id].color, scene.solids[b.id].opacity) == ("#c9ced6", 0.85)
    sheet = Primitive(name="s", kind=PrimitiveKind.SHEET, material="", sheet_shape=SheetShape.RECTANGLE)
    assert appearance(sheet, materials) == (None, 0.18)  # construction: the theme's accent, faint
    sheet.material = "FR4"
    assert appearance(sheet, materials) == ("#2f7a3f", 0.55)


def test_a_boolean_keeps_the_first_operands_look_and_flag():
    p = _two_boxes()
    a, b = p.primitives
    a.model, a.color, a.opacity = False, "#123456", 0.4
    combined = edits.find_primitive(p, edits.make_boolean(p, "union", [a.id, b.id]))
    assert (combined.model, combined.color, combined.opacity) == (False, "#123456", 0.4)


def test_validation_leaves_non_model_objects_out():
    from EMShrimulator.core.em import validate

    def codes(project):
        return {i["code"] for i in validate.validate(project)["issues"]}

    p = _two_boxes()
    p.primitives[1].material = "gone"
    assert "unknown-material" in codes(p)
    p.primitives[1].model = False
    found = codes(p)
    assert "unknown-material" not in found
    assert "no-geometry" not in found
    p.primitives[0].model = False
    assert "no-geometry" in codes(p)  # every object is non-model


def test_the_tree_groups_non_model_objects_apart():
    from EMShrimulator.app.views.modeler.project_tree import object_groups

    p = _two_boxes()
    assert object_groups(p)["nonmodel"] == {}
    p.primitives[1].model = False
    groups = object_groups(p)
    assert [x.name for x in groups["solids"]["PEC"]] == ["a"]
    assert [x.name for x in groups["nonmodel"]["PEC"]] == ["b"]


# --- solver arithmetic ---------------------------------------------------------

def test_max_delta_s_is_the_worst_change():
    a = {1: {1: np.array([0.5, 0.4]), 2: np.array([0.1, 0.2])}}
    b = {1: {1: np.array([0.5, 0.43]), 2: np.array([0.1, 0.25 + 0.0j])}}
    assert max_delta_s(a, b) == pytest.approx(0.05)
    assert max_delta_s(a, a) == 0.0


def test_refined_meshes_finer():
    p = _project()
    p.mesh.max_cell_size = 1.0
    finer = refined(p, 1.5)
    assert finer.mesh.cells_per_wavelength == pytest.approx(p.mesh.cells_per_wavelength * 1.5)
    assert finer.mesh.max_cell_size == pytest.approx(1 / 1.5)
    assert p.mesh.max_cell_size == 1.0  # the original is untouched
    assert refined(p, 1.0) is p


def test_gain_from_accepted_power():
    payload = {"frequencies": [{"frequency_hz": 2e9, "dmax_dbi": 6.0, "radiated_power_w": 0.5}]}
    freqs = np.array([1e9, 3e9])
    add_gain(payload, freqs, {"accepted": np.array([1.0, 1.0]), "incident": np.array([2.0, 2.0])})
    entry = payload["frequencies"][0]
    assert entry["efficiency"] == pytest.approx(0.5)
    assert entry["gain_dbi"] == pytest.approx(6.0 + 10 * np.log10(0.5))
    assert entry["realized_gain_dbi"] == pytest.approx(6.0 + 10 * np.log10(0.25))


def test_no_z_matrix_for_a_series_element():
    # A series resistor between 50 ohm ports has no Z-matrix (I - S is singular).
    s = np.tile(np.array([[1 / 3, 2 / 3], [2 / 3, 1 / 3]], dtype=complex), (3, 1, 1))
    assert impedance_traces(s, [50.0, 50.0], {}) == {}


def test_z_matrix_of_a_t_network():
    # A T of 10 ohm series arms and a 100 ohm shunt: Z = [[110, 100], [100, 110]].
    zmat = np.array([[110.0, 100.0], [100.0, 110.0]])
    z0 = 50.0
    eye = np.eye(2)
    smat = (zmat - z0 * eye) @ np.linalg.inv(zmat + z0 * eye)
    z = impedance_traces(np.tile(smat.astype(complex), (2, 1, 1)), [z0, z0], {})
    assert z["Z11"]["z_real"][0] == pytest.approx(110.0)
    assert z["Z21"]["z_real"][0] == pytest.approx(100.0)
    assert z["Z12"]["z_imag"][0] == pytest.approx(0.0, abs=1e-9)


def test_impedance_with_one_port_driven_is_input_impedance():
    s = np.zeros((2, 2, 2), dtype=complex)
    s[:, 0, 0] = 0.2
    s[:, 1, 0] = 0.5  # only port 1 driven...
    s[:, 0, 1] = 0.5  # column copied by reciprocity; S22 stays 0
    traces = {"S11": {"z_real": [75.0, 75.0], "z_imag": [0.0, 0.0]}, "S21": {}}
    z = impedance_traces(s, [50.0, 50.0], traces)
    assert list(z) == ["Zin1"] and z["Zin1"]["z_real"][0] == 75.0


# --- materials: many conductors share one ----------------------------------------------

def test_conductors_may_share_a_material():
    from EMShrimulator.core.em import validate

    p = _two_boxes()  # two PEC boxes: two conductors of one material
    found = {i["code"]: i for i in validate.validate(p)["issues"]}
    assert "shared-conductor-material" not in found
    assert found["unused-material"]["target"] == "FR4"


def test_a_material_only_operands_name_is_unused_and_deletes():
    p = _two_boxes()
    a, b = p.primitives
    b.material = "FR4"  # the cut operand keeps FR4
    edits.make_boolean(p, "subtract", [a.id, b.id])
    assert edits.material_users(p, "FR4") == [] and edits.material_users(p, "PEC") == ["subtract_1"]
    edits.delete_material(p, "FR4")
    assert [m.name for m in p.materials] == ["PEC"]
    assert [c.material for c in p.primitives[0].children] == ["PEC", "PEC"]  # an explode stays valid
    with pytest.raises(EditError, match="subtract_1 is made of 'PEC'"):
        edits.delete_material(p, "PEC")


# --- the open region sizes the air ----------------------------------------------------------

def _open(start, stop, operating=(), antenna=False):
    p = _two_boxes()
    p.frequency.start, p.frequency.stop = start, stop
    p.nf2ff.enabled, p.nf2ff.frequencies = antenna, list(operating)
    p.boundary.open_region = True
    p.boundary.zmin = "PEC"
    return p


def test_the_open_region_is_a_quarter_wave_at_the_frequency_that_matters():
    from EMShrimulator.core.em import region

    # A band: λ/4 at its lowest frequency, 7 GHz, is 10.71 mm.
    resolved = params.resolve(_open(7e9, 9e9))[0]
    assert resolved.mesh.air_margin == (10.8, 10.8, 10.8)
    assert {getattr(resolved.boundary, f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")} == {"PML_8"}
    # An antenna: its lowest operating frequency.
    assert region.reference_frequency(params.resolve(_open(1e9, 4e9, [2.45e9, 3e9], antenna=True))[0]) == (
        2.45e9, "the lowest operating frequency")
    # A sweep from near DC would make the box absurd: its centre instead.
    frequency, why = region.reference_frequency(params.resolve(_open(1e6, 6e9))[0])
    assert frequency == pytest.approx(3.0005e9) and "centre" in why


def test_the_open_region_leaves_a_guided_axis_its_own_margin():
    p = _open(8e9, 12e9)
    p.mesh.air_margin = (0.0, 0.0, 2.0)
    sheet_id = edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RECTANGLE)
    p.ports.append(Port(name="P1", kind=PortKind.RECT_WAVEGUIDE, number=1, direction="z", excite=True,
                        sheet=edits.find_primitive(p, sheet_id).name))
    assert params.resolve(p)[0].mesh.air_margin == (9.37, 9.37, 2.0)  # the guide runs into the boundary on z
    assert p.mesh.air_margin == (0.0, 0.0, 2.0) and p.boundary.zmin == "PEC"  # the design itself is untouched


def test_an_open_region_never_warns_of_a_tight_margin():
    from EMShrimulator.core.em import validate

    p = _open(7e9, 9e9)
    p.mesh.air_margin = (1.0, 1.0, 1.0)
    codes = {i["code"] for i in validate.validate(params.resolve(p)[0])["issues"]}
    assert "tight-air-margin" not in codes
    p.boundary.open_region = False
    codes = {i["code"] for i in validate.validate(params.resolve(p)[0])["issues"]}
    assert "tight-air-margin" in codes


# --- templates are built the way the modeler builds a design --------------------------------

def test_every_template_port_is_on_a_sheet_with_what_it_needs():
    from EMShrimulator.core.em import templates, validate

    for key in templates.TEMPLATES:
        p = templates.build(key, key)
        for port in p.ports:
            assert sheets.sheet_by_name(p.primitives, port.sheet) is not None, (key, port.name)
            if port.kind in sheets.LINE_KINDS:
                assert sheets.line_ends(params.resolve(p)[0].ports[p.ports.index(port)]) is not None, (key, port.name)
        metals = {m.name for m in p.materials if m.kind != MaterialKind.DIELECTRIC}
        used = {prim.material for prim in p.primitives if prim.material in metals}
        assert len(used) <= 1, (key, used)  # every conductor is of one material
        report = validate.validate(params.resolve(p)[0])
        assert key == "blank" or report["can_run"], (key, [i for i in report["issues"] if i["severity"] == "error"])
