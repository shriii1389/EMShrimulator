"""Snapping and click-by-click drawing in the 3D view: the pure parts, no window and no openEMS.

Run with `.venv/bin/python -m pytest -q tests/test_snapping.py`.
"""
import numpy as np
import pytest

from EMShrimulator.core.em import csg, drawing, edits, sheets, snaps
from EMShrimulator.core.em.model import (
    CoordinateSystem, Material, MaterialKind, ModelerProject, Primitive, PrimitiveKind, SheetShape, TransformOp,
)
from EMShrimulator.core.em.scene import WorkingFrame, grid_step, working_grid
from EMShrimulator.core.em.transforms import apply_ops, effective_ops, unapply_ops
from EMShrimulator.core.errors import EditError


def _project():
    return ModelerProject(name="t", materials=[Material(name="PEC", kind=MaterialKind.PEC)])


def _targets(prim):
    return snaps.solid_snaps(*csg.mesh_of(prim, {}))


def _has(points, point):
    return bool(np.any(np.all(np.isclose(points, point, atol=1e-6), axis=1)))


# --- what an object offers ---------------------------------------------------

def test_a_box_offers_six_face_centres_twelve_edge_centres_eight_corners():
    t = _targets(Primitive(name="b", kind="box", material="PEC", p1=(0, 0, 0), p2=(10, 6, 1.6)))
    assert [len(t.of_kind(k)) for k in (snaps.FACE, snaps.EDGE, snaps.QUADRANT, snaps.CORNER)] == [6, 12, 0, 8]
    assert _has(t.of_kind(snaps.FACE), [5, 3, 1.6])     # the top face
    assert _has(t.of_kind(snaps.EDGE), [5, 0, 1.6])     # the middle of a top edge
    assert _has(t.of_kind(snaps.CORNER), [10, 6, 1.6])


def test_a_cylinder_offers_its_cap_centres_and_rim_quarter_points():
    # Radius 3 has 39 facets: quarter points still land on the axis.
    t = _targets(Primitive(name="c", kind="cylinder", material="PEC", p1=(1, 2, 0), p2=(1, 2, 5), radius=3.0))
    faces = t.of_kind(snaps.FACE)
    # Only the caps; the side's centre is inside the cylinder.
    assert len(faces) == 2 and _has(faces, [1, 2, 0]) and _has(faces, [1, 2, 5])
    assert len(t.of_kind(snaps.ARC)) == 0  # each rim's centre is its cap's centre, offered once
    assert len(t.of_kind(snaps.CORNER)) == 0  # a round rim has no corners
    quads = t.of_kind(snaps.QUADRANT)
    assert len(quads) == 8
    rel = quads - [1, 2, 0]
    # Within 1 nm: manifold3d hands vertices back in single precision.
    assert np.all(np.isclose(rel[:, 0], 0, atol=1e-6) | np.isclose(rel[:, 1], 0, atol=1e-6))
    radii = np.hypot(rel[:, 0], rel[:, 1])
    assert np.all((radii <= 3.0 + 1e-6) & (radii >= 3.0 * np.cos(np.pi / 39) - 1e-6))


def test_a_hole_through_a_board_offers_its_rims():
    board = Primitive(name="b", kind="box", material="PEC", p1=(0, 0, 0), p2=(10, 6, 1.6))
    pin = Primitive(name="p", kind="cylinder", material="PEC", p1=(5, 3, -1), p2=(5, 3, 3), radius=0.5)
    t = _targets(Primitive(name="h", kind="boolean", material="PEC", boolean_op="subtract", children=[board, pin]))
    assert len(t.of_kind(snaps.CORNER)) == 8 and len(t.of_kind(snaps.EDGE)) == 12
    assert _has(t.of_kind(snaps.QUADRANT), [5.5, 3, 1.6]) and _has(t.of_kind(snaps.QUADRANT), [5, 2.5, 0])
    # The middle of the hole on each face, said once.
    centres = np.vstack([t.of_kind(snaps.FACE), t.of_kind(snaps.ARC)])
    assert _has(centres, [5, 3, 0]) and _has(centres, [5, 3, 1.6]) and len(t.of_kind(snaps.ARC)) == 0
    assert not _has(t.of_kind(snaps.FACE), [5, 3, 0.8])  # the round wall offers no centre inside the board


def test_a_half_round_notch_offers_its_centre():
    board = Primitive(name="b", kind="box", material="PEC", p1=(0, 0, 0), p2=(10, 6, 1.6))
    notch = Primitive(name="n", kind="cylinder", material="PEC", p1=(10, 3, -1), p2=(10, 3, 3), radius=1.5)
    t = _targets(Primitive(name="h", kind="boolean", material="PEC", boolean_op="subtract", children=[board, notch]))
    arcs = t.of_kind(snaps.ARC)
    assert len(arcs) == 2 and np.allclose(arcs[:, :2], [10, 3], atol=1e-5)  # on the board's edge, both faces


def test_a_clearance_circle_on_a_thin_ground_offers_its_centre():
    # A probe feed through a hole in 35 µm ground.
    ground = Primitive(name="g", kind="box", material="PEC", p1=(-10, -10, -0.035), p2=(10, 10, 0))
    hole = Primitive(name="h", kind="cylinder", material="PEC", p1=(0, -3, -0.035), p2=(0, -3, 1.6), radius=1.7)
    plate = _targets(Primitive(name="p", kind="boolean", material="PEC", boolean_op="subtract",
                               children=[ground, hole]))
    assert _has(plate.of_kind(snaps.ARC), [0, -3, -0.035]) and _has(plate.of_kind(snaps.ARC), [0, -3, 0])
    on_axis = [p for p in plate.of_kind(snaps.FACE) if np.hypot(p[0], p[1] + 3) < 1e-3]
    assert not on_axis  # nothing halfway up the hole's wall
    pin = _targets(Primitive(name="pin", kind="cylinder", material="PEC", p1=(0, -3, -0.035), p2=(0, -3, 1.6),
                             radius=0.35))
    assert len(pin.of_kind(snaps.FACE)) == 2  # its two ends; its side has none


def test_sheets_offer_centre_edges_and_corners():
    p = _project()
    rect = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RECTANGLE))
    rect.normal_axis, rect.sheet_size, rect.origin = "x", (3.0, 1.5), (2.0, 0.0, 0.75)
    t = snaps.sheet_snaps([sheets.lift(rect, loop) for loop in sheets.outline(rect)])
    assert np.allclose(t.of_kind(snaps.FACE), [[2, 0, 0.75]])
    assert len(t.of_kind(snaps.CORNER)) == 4 and _has(t.of_kind(snaps.EDGE), [2, 0, 0])
    ring = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.SHEET, SheetShape.RING))
    ring.radius, ring.radius_inner, ring.origin = 2.0, 0.5, (3.0, 4.0, 0.0)
    t = snaps.sheet_snaps([sheets.lift(ring, loop) for loop in sheets.outline(ring)])
    assert np.allclose(t.of_kind(snaps.FACE), [[3, 4, 0]])  # the hole does not pull the centre off
    assert _has(t.of_kind(snaps.QUADRANT), [5, 4, 0]) and _has(t.of_kind(snaps.QUADRANT), [3, 3.5, 0])


def test_sheet_snap_kinds_line_up_with_its_points():
    p = _project()
    for shape in SheetShape:
        prim = edits.find_primitive(p, edits.add_primitive(p, PrimitiveKind.SHEET, shape))
        assert len(sheets.snap_kinds(prim)) == len(sheets.snap_points(prim))


# --- choosing one ----------------------------------------------------------------

def test_the_nearest_target_wins_and_a_corner_beats_a_face_centre_as_close():
    kinds = np.array([snaps.POINT_KINDS.index(k) for k in (snaps.FACE, snaps.CORNER, snaps.EDGE)])
    screen = np.array([[100.0, 100.0], [110.0, 100.0], [140.0, 100.0]])
    depth = np.ones(3)
    every = (snaps.FACE, snaps.EDGE, snaps.QUADRANT, snaps.CORNER)
    assert snaps.nearest(screen, depth, kinds, (105, 100), 22, every) == 1      # equally near: the corner
    assert snaps.nearest(screen, depth, kinds, (101, 100), 22, every) == 0
    assert snaps.nearest(screen, depth, kinds, (105, 100), 22, (snaps.FACE,)) == 0  # corners turned off
    assert snaps.nearest(screen, depth, kinds, (300, 100), 22, every) is None    # nothing near enough


def test_of_two_targets_on_top_of_each_other_the_one_nearer_the_eye_wins():
    # A thin board from above: top and bottom corners on the same pixel.
    kinds = np.full(2, snaps.POINT_KINDS.index(snaps.CORNER))
    screen = np.array([[50.0, 50.0], [50.5, 50.0]])
    assert snaps.nearest(screen, np.array([9.0, 8.0]), kinds, (50.4, 50), 22, (snaps.CORNER,)) == 1
    behind = np.array([np.inf, 8.0])
    assert snaps.nearest(screen, behind, kinds, (50, 50), 22, (snaps.CORNER,)) == 1


def test_grid_snap_keeps_the_height_of_a_lifted_plane():
    u, v = np.eye(3)[0], np.eye(3)[1]
    point = snaps.on_grid(np.array([1.26, -0.74, 1.6]), np.zeros(3), u, v, 0.5)
    assert np.allclose(point, [1.5, -0.5, 1.6])
    assert snaps.round_to(0.74, 0.5) == pytest.approx(0.5)


def test_grid_step_is_the_drawn_spacing():
    frame = WorkingFrame(np.array([3.0, -2.0, 1.0]), np.eye(3))
    bounds = (np.array([10.0, 10.0, 0.0]), np.array([30.0, 20.0, 5.0]))
    assert grid_step(frame, bounds) == working_grid(frame, bounds)["step"]


# --- clicks into objects -------------------------------------------------------------

XY = drawing.Plane.of_frame(np.zeros(3), np.eye(3), "xy")


def test_a_box_from_two_corners_and_a_height():
    tool = drawing.BoxTool(XY)
    a, b, c = np.array([0.0, 0, 0]), np.array([10.0, 6, 0]), np.array([10.0, 6, 1.6])
    assert tool.step([]).plane is XY and not tool.step([]).project_snaps
    assert tool.step([a]).project_snaps  # the far corner keeps to the base's plane
    assert tool.refuse([a], np.array([10.0, 0, 0]))  # no length
    assert tool.refuse([a, b], b)                     # no height
    assert tool.summary([a, b], c, drawing.Units()) == "10 × 6 × 1.6 mm"
    assert drawing.drawn_fields(tool, [a, b, c]) == {"p1": (0.0, 0.0, 0.0), "p2": (10.0, 6.0, 1.6)}
    assert len(tool.preview([a, b], c)) == 6  # base, top and four uprights


def test_a_rectangle_on_the_yz_plane_is_an_x_normal_sheet():
    tool = drawing.RectangleTool(drawing.Plane.of_frame(np.zeros(3), np.eye(3), "yz"))
    fields = drawing.drawn_fields(tool, [np.array([0.0, -1.5, 0]), np.array([0.0, 1.5, 1.6])])
    assert fields == {"normal_axis": "x", "origin": (0.0, 0.0, 0.8), "sheet_size": (3.0, 1.6)}


def test_round_objects():
    circle = drawing.CircleTool(XY)
    fields = drawing.drawn_fields(circle, [np.array([5.0, 3, 0]), np.array([6.2, 3, 0])])
    assert fields == {"normal_axis": "z", "origin": (5.0, 3.0, 0.0), "radius": 1.2}
    cylinder = drawing.CylinderTool(XY)
    clicks = [np.array([5.0, 3, 0]), np.array([5.0, 3.4, 0]), np.array([5.0, 3, 2])]
    assert cylinder.clicks == 3 and cylinder.step(clicks[:2]).line is not None
    assert drawing.drawn_fields(cylinder, clicks) == {"p1": (5.0, 3.0, 0.0), "p2": (5.0, 3.0, 2.0), "radius": 0.4}
    sphere = drawing.SphereTool(XY)
    assert drawing.drawn_fields(sphere, [np.zeros(3), np.array([0.0, 3, 4])]) == {"center": (0.0, 0.0, 0.0),
                                                                                    "radius": 5.0}
    assert drawing.draw_tool(PrimitiveKind.POLYGON, None, XY) is None  # typed in, not clicked


def test_a_box_drawn_in_a_turned_working_system_lands_where_it_was_clicked():
    p = _project()
    p.coordinate_systems.append(CoordinateSystem(name="CS1", transforms=[
        TransformOp(kind="rotate_axis", axis="z", angle_deg=90.0), TransformOp(kind="translate", translate=(10, 0, 2))]))
    p.working_cs = "CS1"
    ops = effective_ops("CS1", [], {cs.name: cs for cs in p.coordinate_systems})
    # The base on CS1's XY plane, clicked in global coordinates, then 1.5 up CS1's z.
    local_clicks = np.array([[0.0, 0, 0], [4.0, 2, 0], [4.0, 2, 1.5]])
    world = apply_ops(local_clicks, ops)
    frame_axes = apply_ops(np.eye(3), ops) - apply_ops(np.zeros((1, 3)), ops)
    tool = drawing.BoxTool(drawing.Plane.of_frame(world[0], frame_axes, "xy"))
    fields = drawing.drawn_fields(tool, unapply_ops(world, ops))
    box = edits.find_primitive(p, edits.draw_primitive(p, PrimitiveKind.BOX, None, fields))
    assert box.coordinate_system == "CS1" and box.p1 == (0.0, 0.0, 0.0) and box.p2 == (4.0, 2.0, 1.5)
    vertices, _ = csg.mesh_of(box, {cs.name: cs for cs in p.coordinate_systems})
    assert np.allclose(vertices.min(axis=0), world.min(axis=0)) and np.allclose(vertices.max(axis=0), world.max(axis=0))


def test_drawing_sets_only_placement_and_size():
    p = _project()
    sheet_id = edits.draw_primitive(p, PrimitiveKind.SHEET, SheetShape.CIRCLE,
                                    {"normal_axis": "z", "origin": (1.0, 2.0, 0.0), "radius": 0.5})
    sheet = edits.find_primitive(p, sheet_id)
    assert sheet.name == "circle_1" and sheet.radius == 0.5 and not sheet.material  # a construction sheet
    with pytest.raises(EditError):
        edits.draw_primitive(p, PrimitiveKind.BOX, None, {"material": "PEC"})


# --- the integration line and the measure -------------------------------------------------

def test_a_line_end_keeps_to_one_axis():
    plane = drawing.Plane.normal_to(np.zeros(3), [1.0, 0, 0])
    assert abs(float(plane.normal @ [1.0, 0, 0])) == pytest.approx(1.0)
    tool = drawing.LineTool(plane, ("start", "end"))
    start = np.array([0.0, 0, 0])
    step = tool.step([start])
    assert step.prompt == "end" and step.project_snaps and not step.grid
    assert np.allclose(drawing.keep_to_axis(np.array([0.0, 0.2, 1.5]), step.axis_from), [0, 0, 1.5])
    assert tool.refuse([start], start)


def test_rays_meet_planes_and_lines():
    near, far = np.array([2.0, 3.0, 10.0]), np.array([2.0, 3.0, -10.0])
    assert np.allclose(XY.hit(near, far), [2, 3, 0])
    assert XY.hit(np.array([0.0, 0, 1]), np.array([5.0, 0, 1])) is None  # along the plane
    # Looking along -y at the line through the origin.
    point = drawing.closest_on_line(np.zeros(3), np.array([0.0, 0, 1]), np.array([0.0, 10, 2.5]),
                                    np.array([0.0, -10, 2.5]))
    assert np.allclose(point, [0, 0, 2.5])


def test_a_measurement_is_two_points():
    tool = drawing.MeasureTool(XY)
    first, second = np.array([0.0, 0, 1.6]), np.array([3.0, 4.0, 1.6])
    assert tool.repeat and tool.clicks == 2 and tool.step([]).surface
    assert "first point" in tool.step([]).prompt and "second point" in tool.step([first]).prompt
    assert tool.refuse([first], first)  # the same point twice measures nothing
    assert [text for _, text in tool.labels([first], second, drawing.Units())] == ["1", "2", "5 mm"]
    text = tool.summary([first], second, drawing.Units("mil", 0.0254))
    assert "distance 196.85 mil" in text and "Δx 118.11" in text and "Δz 0 mil" in text
    assert "in the global system" in text and "theta 90°" in text and "phi 53.13°" in text
    assert drawing.Units().num(-2.2204e-16) == "0"  # a transform's leftover, not a length


def test_measured_angles():
    m = drawing.measure([0, 0, 0], [3.0, 4.0, 0.0])
    assert m["distance"] == pytest.approx(5.0) and (m["dx"], m["dy"], m["dz"]) == (3.0, 4.0, 0.0)
    assert m["theta"] == pytest.approx(90.0) and m["phi"] == pytest.approx(53.130102)
    assert (m["to_x"], m["to_y"], m["to_z"]) == pytest.approx((53.130102, 36.869898, 90.0))
    up = drawing.measure([1, 1, 1], [1, 1, 3])
    assert up["theta"] == pytest.approx(0.0) and np.isnan(up["phi"])  # straight up z: no φ
    down_left = drawing.measure([0, 0, 0], [-1.0, 0.0, -1.0])
    assert down_left["theta"] == pytest.approx(135.0) and down_left["phi"] == pytest.approx(180.0)
    assert np.isnan(drawing.measure([1, 2, 3], [1, 2, 3])["theta"])


def test_measuring_reads_in_the_chosen_coordinate_system():
    # CS1 is turned a quarter about z: its x is global y, its y is global -x.
    axes = np.array([[0.0, 1, 0], [-1.0, 0, 0], [0.0, 0, 1]])
    m = drawing.measure([0, 0, 0], [3.0, 4.0, 0.0], axes)
    assert (m["dx"], m["dy"]) == pytest.approx((4.0, -3.0)) and m["phi"] == pytest.approx(-36.869898)
    assert m["distance"] == pytest.approx(5.0)  # a length does not depend on the system

    class Frame:
        origin, name = np.array([10.0, 0, 2]), "CS1"

    Frame.axes = axes
    text = drawing.MeasureTool(XY).summary([np.array([10.0, 0, 2])], np.array([13.0, 4.0, 2.0]), drawing.Units(),
                                           Frame)
    assert "1 (0, 0, 0)" in text and "2 (4, -3, 0)" in text and "in CS1" in text and "phi -36.87°" in text
