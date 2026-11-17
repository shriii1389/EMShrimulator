"""Relative and object coordinate systems: placing them, keeping them on their point, and refusing loops.

Run with `.venv/bin/python -m pytest -q tests/test_coordinate_systems.py`.
"""
import numpy as np
import pytest

from EMShrimulator.core.em import edits, frames, params, sheets
from EMShrimulator.core.em.csx_builder import global_aabb
from EMShrimulator.core.em.model import (
    CoordinateSystem, Material, MaterialKind, ModelerProject, Parameter, PortKind, PrimitiveKind, SheetShape,
)
from EMShrimulator.core.em.scene import WorkingFrame, build_scene, triads
from EMShrimulator.core.em.transforms import chain_matrix, frame_matrix, ops_of, rotation
from EMShrimulator.core.errors import EditError


def _project():
    return ModelerProject(name="t", materials=[Material(name="PEC", kind=MaterialKind.PEC)])


def _box(p, p1=(0.0, 0.0, 0.0), p2=(10.0, 4.0, 2.0), cs=None):
    box_id = edits.add_primitive(p, PrimitiveKind.BOX)
    box = edits.find_primitive(p, box_id)
    box.p1, box.p2, box.coordinate_system = p1, p2, cs
    return box


def _placed(p):
    resolved = params.resolve(p)[0]
    return resolved, {cs.name: cs for cs in resolved.coordinate_systems}


def _frame(p, name):
    resolved, by_name = _placed(p)
    assert not by_name[name].placement_error, by_name[name].placement_error
    return frame_matrix(name, by_name)


# --- relative systems ---------------------------------------------------------------------------------------

def test_offset_rotated_or_both():
    p = _project()
    p.coordinate_systems += [
        CoordinateSystem(name="A", mode="offset", origin=(1.0, 2.0, 3.0), angles=(0.0, 0.0, 90.0)),
        CoordinateSystem(name="B", mode="rotated", origin=(1.0, 2.0, 3.0), angles=(0.0, 0.0, 90.0)),
        CoordinateSystem(name="C", mode="both", origin=(1.0, 2.0, 3.0), angles=(0.0, 0.0, 90.0)),
    ]
    a, b, c = (_frame(p, n) for n in "ABC")
    assert np.allclose(a[:3, 3], [1, 2, 3]) and np.allclose(a[:3, :3], np.eye(3))  # the angles wait unused
    assert np.allclose(b[:3, 3], 0) and np.allclose(b[:3, 0], [0, 1, 0])           # and here the offset
    assert np.allclose(c[:3, 3], [1, 2, 3]) and np.allclose(c[:3, 0], [0, 1, 0])


def test_a_nested_system_moves_along_its_base_axes():
    p = _project()
    p.coordinate_systems += [CoordinateSystem(name="Base", mode="both", origin=(10.0, 0.0, 0.0),
                                              angles=(0.0, 0.0, 90.0)),
                             CoordinateSystem(name="Child", parent="Base", origin=(5.0, 0.0, 0.0))]
    child = _frame(p, "Child")
    assert np.allclose(child[:3, 3], [10, 5, 0])  # 5 along Base's x, which is global y


def test_rotation_turns_about_x_then_y_then_z():
    R = rotation((90.0, 0.0, 90.0))
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-12)  # z, turned onto -y about x, then onto x about z
    for angles in [(30.0, -45.0, 120.0), (0.0, 90.0, 10.0), (12.0, -90.0, 0.0)]:
        M = np.eye(4)
        M[:3, :3] = rotation(angles)
        M[:3, 3] = (1.0, -2.0, 0.5)
        assert np.allclose(chain_matrix(ops_of(M)), M, atol=1e-9)  # the chain CSXCAD replays is the same frame


def test_expressions_drive_a_system():
    p = _project()
    p.parameters.append(Parameter(name="h", expression="1.6"))
    p.coordinate_systems.append(CoordinateSystem(name="Top", origin=(0.0, 0.0, "h")))
    assert np.allclose(_frame(p, "Top")[:3, 3], [0, 0, 1.6])
    edits.rename_parameter(p, "h", "sub_h")
    assert p.coordinate_systems[0].origin == (0.0, 0.0, "sub_h")


def test_an_old_transform_chain_still_loads():
    old = {"name": "CS1", "parent": None, "transforms": [
        {"kind": "rotate_axis", "axis": "z", "angle_deg": 90.0},
        {"kind": "translate", "translate": [10.0, 0.0, 2.0]}]}
    p = ModelerProject.model_validate({"name": "t", "coordinate_systems": [old]})
    cs = p.coordinate_systems[0]
    assert (cs.mode, cs.origin, cs.angles) == ("both", (10.0, 0.0, 2.0), (0.0, 0.0, 90.0))
    typed = CoordinateSystem.model_validate({"name": "T", "transforms": [
        {"kind": "translate", "translate": ["w", 0.0, 0.0]}, {"kind": "translate", "translate": [1.0, 0.0, 0.0]}]})
    assert typed.mode == "offset" and typed.origin == ("w + 1", 0.0, 0.0)
    assert "placement" not in p.model_dump()  # worked out on every resolve, never saved


# --- object systems -----------------------------------------------------------------------------------------

def _on(p, box, anchor, at=None, name="O"):
    p.coordinate_systems.append(CoordinateSystem(name=name, kind="object", object_id=box.id, anchor=anchor,
                                                 anchor_at=at))


def test_each_kind_of_point_on_a_box():
    p = _project()
    box = _box(p)
    _on(p, box, "centre", name="Mid")
    _on(p, box, "face", (0.5, 0.5, 1.0), "Top")
    _on(p, box, "edge", (0.5, 0.0, 1.0), "Edge")
    _on(p, box, "corner", (1.0, 1.0, 0.0), "Corner")
    expected = {"Mid": [5, 2, 1], "Top": [5, 2, 2], "Edge": [5, 0, 2], "Corner": [10, 4, 0]}
    for name, origin in expected.items():
        assert np.allclose(_frame(p, name)[:3, 3], origin), name


def test_an_object_system_stays_on_its_point_as_the_object_changes():
    p = _project()
    p.coordinate_systems.append(CoordinateSystem(name="Move", origin=(0.0, 0.0, 0.0)))
    box = _box(p, cs="Move")
    _on(p, box, "face", (0.5, 0.5, 1.0), "Top")
    assert np.allclose(_frame(p, "Top")[:3, 3], [5, 2, 2])
    box.p2 = (20.0, 8.0, 5.0)  # grown
    assert np.allclose(_frame(p, "Top")[:3, 3], [10, 4, 5])
    p.coordinate_systems[0].origin = (100.0, 0.0, 0.0)  # and moved with the system it is drawn in
    assert np.allclose(_frame(p, "Top")[:3, 3], [110, 4, 5])
    p.coordinate_systems[0].mode, p.coordinate_systems[0].angles = "both", (0.0, 0.0, 90.0)  # and turned
    top = _frame(p, "Top")
    assert np.allclose(top[:3, 3], [96, 10, 5]) and np.allclose(top[:3, 0], [0, 1, 0], atol=1e-9)


def test_on_a_face_z_points_out_of_it():
    p = _project()
    box = _box(p)
    _on(p, box, "face", (1.0, 0.5, 0.5), "Side")
    _on(p, box, "face", (0.5, 0.5, 0.0), "Bottom")
    side = _frame(p, "Side")
    assert np.allclose(side[:3, 3], [10, 2, 1])
    assert np.allclose(side[:3, 2], [1, 0, 0]) and np.allclose(side[:3, 0], [0, 1, 0])  # X along the box's y
    assert np.isclose(np.linalg.det(side[:3, :3]), 1.0)
    bottom = _frame(p, "Bottom")
    assert np.allclose(bottom[:3, 2], [0, 0, -1]) and np.allclose(bottom[:3, 0], [1, 0, 0])


def test_a_sheet_drawn_on_a_face_system_lands_on_the_face():
    """A port on a substrate's end: a system on the face, a rectangle in its XY plane."""
    p = _project()
    box = _box(p)
    _on(p, box, "face", (1.0, 0.5, 0.5), "End")
    p.working_cs = "End"
    sheet = edits.find_primitive(p, edits.draw_primitive(p, PrimitiveKind.SHEET, SheetShape.RECTANGLE, {
        "normal_axis": "z", "origin": (0.0, 0.0, 0.0), "sheet_size": (4.0, 2.0)}))
    assert sheet.coordinate_system == "End"
    port_id = edits.assign_port(p, sheet.id, PortKind.RECT_WAVEGUIDE)
    resolved, by_name = _placed(p)
    lo, hi = global_aabb(edits.find_primitive(resolved, sheet.id), by_name)
    assert np.allclose(lo, [10, 0, 0]) and np.allclose(hi, [10, 4, 2])  # covering the end face exactly
    placed = sheets.port_geometry(resolved, edits.find_port(resolved, port_id), by_name, 0.1)
    assert placed.prop_axis == "x"


def test_points_of_round_shapes_and_sheets():
    p = _project()
    cylinder = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.CYLINDER))
    cylinder.p1, cylinder.p2, cylinder.radius = (0.0, 0.0, 0.0), (0.0, 0.0, 4.0), 1.0
    sheet = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RECTANGLE))
    sheet.normal_axis, sheet.origin, sheet.sheet_size = "x", (3.0, 0.0, 0.0), (2.0, 2.0)
    resolved = params.resolve(p)[0]
    by_name = {}
    round_kinds = frames.features(edits.find_primitive(resolved, cylinder.id), by_name).kinds
    assert round_kinds.count("face") == 2 and "centre" in round_kinds  # the caps; the curved side offers none
    assert "corner" not in round_kinds and "edge" not in round_kinds  # round rims have neither
    flat = frames.features(edits.find_primitive(resolved, sheet.id), by_name)
    assert "centre" not in flat.kinds and flat.kinds.count("corner") == 4  # a sheet's centre is its face's
    _on(p, sheet, "face", None, "OnSheet")
    on_sheet = _frame(p, "OnSheet")
    assert np.allclose(on_sheet[:3, 3], [3, 0, 0]) and np.allclose(on_sheet[:3, 2], [1, 0, 0])


def test_clicking_a_point_keeps_that_point():
    p = _project()
    box = _box(p)
    resolved = params.resolve(p)[0]
    targets, found = frames.anchor_targets(resolved, box.id)
    kinds = {k for k in targets.kinds}
    assert len(kinds) == 4  # centre, faces, edges and corners only
    for point, kind in zip(targets.points, targets.kinds):
        anchor = frames.ANCHOR_OF[frames.snaps.POINT_KINDS[int(kind)]]
        name = edits.add_object_coordinate_system(p, box.id, anchor, frames.anchor_at(found, point))
        assert np.allclose(_frame(p, name)[:3, 3], point)


def test_describing_the_point():
    cs = CoordinateSystem(name="O", kind="object", anchor="edge", anchor_at=(0.5, 0.0, 1.0))
    assert frames.describe_anchor(cs) == "edge centre (−Y +Z)"
    cs.anchor, cs.anchor_at = "face", (0.5, 0.5, 1.0)
    assert frames.describe_anchor(cs) == "face centre (+Z)"
    assert frames.describe_anchor(CoordinateSystem(name="C", kind="object")) == "object centre"


# --- edits ----------------------------------------------------------------------------------------------------

def test_loops_are_refused():
    p = _project()
    box = _box(p)
    name = edits.add_object_coordinate_system(p, box.id, "face", (0.5, 0.5, 1.0))
    with pytest.raises(EditError, match="loop"):
        edits.set_primitive_field(p, box.id, "coordinate_system", name)  # the box would sit on itself
    a = edits.add_coordinate_system(p, parent=name)
    with pytest.raises(EditError):
        edits.set_coordinate_system_field(p, a, "parent", a)
    other = _box(p, cs=a)
    with pytest.raises(EditError, match="loop"):
        edits.set_coordinate_system_field(p, name, "object_id", other.id)  # name -> other -> a -> name
    edits.set_coordinate_system_field(p, a, "mode", "both")
    with pytest.raises(EditError):
        edits.set_coordinate_system_field(p, a, "mode", "sideways")
    with pytest.raises(EditError):
        edits.add_object_coordinate_system(p, "nothing", "face")


def test_a_new_system_can_become_the_working_one():
    p = _project()
    box = _box(p)
    name = edits.add_object_coordinate_system(p, box.id, "centre", make_working=True)
    assert p.working_cs == name
    rel = edits.add_coordinate_system(p, parent=name, mode="both", origin=(1.0, 0.0, 0.0), make_working=True)
    assert p.working_cs == rel and edits.find_coordinate_system(p, rel).parent == name


def test_a_system_stays_put_when_its_object_goes():
    p = _project()
    box = _box(p)
    name = edits.add_object_coordinate_system(p, box.id, "face", (1.0, 0.5, 0.5), mode="both")
    p.coordinate_systems[0].origin = (0.0, 0.0, 1.0)
    before = _frame(p, name)
    sheet = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RECTANGLE))
    sheet.coordinate_system = name
    edits.delete_primitive(p, box.id)
    cs = edits.find_coordinate_system(p, name)
    assert cs.kind == "relative" and cs.parent is None and cs.object_id is None
    assert np.allclose(_frame(p, name), before, atol=1e-9)  # the sheet placed in it has not moved


def test_combining_the_object_keeps_its_systems_in_place():
    p = _project()
    box = _box(p)
    hole = _box(p, (4.0, 1.0, -1.0), (6.0, 3.0, 3.0))
    name = edits.add_object_coordinate_system(p, box.id, "face", (0.5, 0.5, 1.0))
    before = _frame(p, name)
    edits.make_boolean(p, "subtract", [box.id, hole.id])
    assert edits.find_coordinate_system(p, name).kind == "relative"
    assert np.allclose(_frame(p, name), before, atol=1e-9)


def test_moving_a_system_to_another_point():
    p = _project()
    box = _box(p)
    name = edits.add_object_coordinate_system(p, box.id, "face", (0.5, 0.5, 1.0))
    edits.place_coordinate_system(p, name, box.id, "corner", (0.0, 0.0, 0.0))
    assert np.allclose(_frame(p, name)[:3, 3], [0, 0, 0])
    other = _box(p, (20.0, 0.0, 0.0), (22.0, 2.0, 2.0))
    edits.set_coordinate_system_field(p, name, "object_id", other.id)  # the same corner of another object
    assert np.allclose(_frame(p, name)[:3, 3], [20, 0, 0])


def test_validation_reports_a_system_that_cannot_be_placed():
    from EMShrimulator.core.em.validate import validate

    p = _project()
    _box(p)
    p.coordinate_systems.append(CoordinateSystem(name="Lost", kind="object", object_id="gone", anchor="centre"))
    p.coordinate_systems.append(CoordinateSystem(name="Child", parent="Lost"))
    report = validate(params.resolve(p)[0])
    messages = {issue["target"]: issue["message"] for issue in report["issues"] if issue["code"] == "cs-unplaced"}
    assert "no longer exists" in messages["Lost"] and "cannot be placed" in messages["Child"]
    assert not report["can_run"]


# --- the 3D view ------------------------------------------------------------------------------------------------

def test_axes_shown_for_the_working_and_edited_systems():
    p = _project()
    box = _box(p)
    name = edits.add_object_coordinate_system(p, box.id, "face", (0.5, 0.5, 1.0))
    scene = build_scene(params.resolve(p)[0])
    assert triads(scene, None) == []  # global is the working system: the grid says it all
    assert [(style, label) for _f, style, label in triads(scene, name)] == [("selected", name)]
    p.working_cs = name
    scene = build_scene(params.resolve(p)[0])
    shown = triads(scene, None)
    assert [(style, label) for _f, style, label in shown] == [("working", name), ("global", "")]
    assert np.allclose(shown[0][0].origin, [5, 2, 2]) and np.allclose(shown[1][0].origin, 0)
    assert [style for _f, style, _l in triads(scene, "")] == ["working", "selected"]  # global, edited
    assert isinstance(shown[1][0], WorkingFrame)
