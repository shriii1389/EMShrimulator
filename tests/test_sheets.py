"""End-to-end check: sheets build, sheet-bound ports build, validation fires."""
import numpy as np

from EMShrimulator.core.em.model import (
    Material, MaterialKind, ModelerProject, Port, PortKind, Primitive,
    PrimitiveKind, SheetShape,
)
from EMShrimulator.core.em import preview, validate
from EMShrimulator.core.em.csx_builder import build_simulation


def base(**kw):
    return ModelerProject(
        name="sheet-test",
        materials=[
            Material(name="PEC", kind=MaterialKind.PEC),
            Material(name="Sub", kind=MaterialKind.DIELECTRIC, epsilon=3.0),
        ],
        **kw,
    )


results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:
        results.append(("FAIL", label, f"{type(e).__name__}: {e}"))


# --- 1. a ring sheet + coaxial wave port actually builds -------------------
ring = Primitive(
    name="CoaxFeed", kind=PrimitiveKind.SHEET, material="",
    sheet_shape=SheetShape.RING, normal_axis="z",
    origin=(0.0, 0.0, 0.0), radius=2.3, radius_inner=0.7,
)
gnd = Primitive(name="Gnd", kind=PrimitiveKind.BOX, material="PEC",
                p1=(-20, -20, -1), p2=(20, 20, -0.9))
coax_prj = base(
    primitives=[gnd, ring],
    ports=[Port(name="P1", kind=PortKind.COAXIAL, number=1, direction="z",
                excite=True, sheet="CoaxFeed")],
)
check("ring sheet + coax port builds", lambda: build_simulation(coax_prj, 1))


def coax_geometry():
    b = build_simulation(coax_prj, 1)
    # The port must sit on the sheet's plane and span its outer diameter.
    g = b.CSX.GetGrid()
    xs = np.array(g.GetLines("x"))
    assert abs(xs.min() + 20) < 25, xs.min()
    # 4.6 mm of outer diameter must appear as fixed lines
    for want in (-2.3, 2.3):
        assert np.min(np.abs(xs - want)) < 1e-6, f"no grid line at x={want}"


check("coax port fixes grid lines at the ring diameter", coax_geometry)

# --- 2. a rectangle sheet + lumped port builds -----------------------------
rect = Primitive(
    name="FeedGap", kind=PrimitiveKind.SHEET, material="",
    sheet_shape=SheetShape.RECTANGLE, normal_axis="y",
    origin=(0.0, 0.0, 0.5), sheet_size=(2.0, 1.0),
)
# Up the gap, edge to edge: the whole rectangle is the port.
FEED_LINE = dict(line_start=(0.0, 0.0, -0.5), line_end=(0.0, 0.0, 1.5))
lumped_prj = base(
    primitives=[gnd, rect],
    ports=[Port(name="P1", kind=PortKind.LUMPED, number=1, direction="z",
                excite=True, sheet="FeedGap", **FEED_LINE)],
)
check("rectangle sheet + lumped port builds", lambda: build_simulation(lumped_prj, 1))

# --- 3. a sheet carrying a material becomes real geometry -----------------
patch = Primitive(
    name="Patch", kind=PrimitiveKind.SHEET, material="PEC",
    sheet_shape=SheetShape.CIRCLE, normal_axis="z",
    origin=(0.0, 0.0, 1.0), radius=8.0,
)
patch_prj = base(
    primitives=[gnd, patch, rect],
    ports=[Port(name="P1", kind=PortKind.LUMPED, number=1, direction="z",
                excite=True, sheet="FeedGap", **FEED_LINE)],
)
check("circle sheet with a material builds as a polygon",
      lambda: build_simulation(patch_prj, 1))

# --- 4. preview emits sheet outlines --------------------------------------


def preview_sheets():
    out = preview.geometry_preview(coax_prj)
    s = out["sheets"]
    assert len(s) == 1, s
    assert s[0]["shape"] == "ring"
    assert s[0]["construction"] is True
    assert len(s[0]["loops"]) == 2, "a ring needs an outer and an inner loop"
    pts = np.array(s[0]["loops"][0]).reshape(-1, 3)
    assert abs(np.abs(pts[:, :2]).max() - 2.3) < 1e-6


check("geometry preview emits the ring outline", preview_sheets)

# --- 5. validation catches the cases openEMS cannot excite ----------------


def codes(prj):
    return {i["code"] for i in validate.validate(prj)["issues"]}


def hollow_circle_rejected():
    prj = base(
        primitives=[gnd, Primitive(
            name="Hole", kind=PrimitiveKind.SHEET, material="",
            sheet_shape=SheetShape.CIRCLE, normal_axis="z",
            origin=(0, 0, 0), radius=3.0)],
        ports=[Port(name="P1", kind=PortKind.COAXIAL, number=1, direction="z",
                    excite=True, sheet="Hole")],
    )
    assert "port-sheet-coax-hollow" in codes(prj), codes(prj)


def lumped_along_normal_rejected():
    prj = base(
        primitives=[gnd, rect],
        ports=[Port(name="P1", kind=PortKind.LUMPED, number=1, direction="y", excite=True, sheet="FeedGap",
                    line_start=(0.0, -1.0, 0.5), line_end=(0.0, 1.0, 0.5))],
    )
    assert "port-line-off-plane" in codes(prj), codes(prj)


def lumped_without_line_rejected():
    prj = base(primitives=[gnd, rect],
               ports=[Port(name="P1", kind=PortKind.LUMPED, number=1, direction="z", excite=True, sheet="FeedGap")])
    assert "port-no-line" in codes(prj), codes(prj)
    assert validate.validate(prj)["can_run"] is False


# A clearance in the ground with the line from its edge in to the centre: a probe feed.
radial_prj = base(
    primitives=[gnd, Primitive(
        name="Round", kind=PrimitiveKind.SHEET, material="",
        sheet_shape=SheetShape.CIRCLE, normal_axis="y",
        origin=(0, 0, 0), radius=3.0)],
    ports=[Port(name="P1", kind=PortKind.LUMPED, number=1, direction="z", excite=True, sheet="Round",
                line_start=(3.0, 0.0, 0.0), line_end=(0.0, 0.0, 0.0))],
)


def circle_radial_line_is_a_probe_source():
    from EMShrimulator.core.em.csx_builder import _port_geometry
    prj = radial_prj
    assert not {c for c in codes(prj) if c.startswith("port")}, codes(prj)
    placed = _port_geometry(prj, prj.ports[0], {})
    assert placed.along_line and placed.exc_axis == "x", placed
    built = build_simulation(prj, 1).ports[1]
    assert (built.exc_ny, built.direction) == (0, -1), (built.exc_ny, built.direction)


def missing_sheet_rejected():
    prj = base(primitives=[gnd],
               ports=[Port(name="P1", kind=PortKind.LUMPED, number=1, direction="z",
                           excite=True, sheet="Nope")])
    assert "port-sheet-missing" in codes(prj), codes(prj)


def ring_with_material_rejected():
    prj = base(
        primitives=[gnd, Primitive(
            name="R", kind=PrimitiveKind.SHEET, material="PEC",
            sheet_shape=SheetShape.RING, normal_axis="z",
            origin=(0, 0, 0), radius=3.0, radius_inner=1.0)],
        ports=[Port(name="P1", kind=PortKind.COAXIAL, number=1, direction="z",
                    excite=True, sheet="R")],
    )
    assert "sheet-ring-material" in codes(prj), codes(prj)


def clean_coax_has_no_sheet_errors():
    # Check all errors, not just sheet codes.
    bad = [i["code"] for i in validate.validate(coax_prj)["issues"] if i["severity"] == "error"]
    assert not bad, bad
    assert validate.validate(coax_prj)["can_run"] is True


check("hollow circle coax port is rejected", hollow_circle_rejected)
check("lumped port with its line along the sheet normal is rejected", lumped_along_normal_rejected)
check("lumped port without an integration line cannot run", lumped_without_line_rejected)
check("a radial line on a circle builds a source along the line", circle_radial_line_is_a_probe_source)
check("port naming a missing sheet is rejected", missing_sheet_rejected)
check("ring sheet with a material is rejected", ring_with_material_rejected)
def clean_lumped_is_runnable():
    bad = [i["code"] for i in validate.validate(lumped_prj)["issues"] if i["severity"] == "error"]
    assert not bad, bad


def circle_sheet_is_never_a_coax():
    """A circle has no inner conductor, even with a stale radius_inner."""
    from EMShrimulator.core.em import sheets as sh
    c = Primitive(name="C", kind=PrimitiveKind.SHEET, material="",
                  sheet_shape=SheetShape.CIRCLE, normal_axis="z",
                  origin=(0, 0, 0), radius=3.0, radius_inner=1.0)
    assert sh.coax_radii(c)[0] == 0.0


check("a correct coax sheet port raises no errors at all", clean_coax_has_no_sheet_errors)
check("a correct lumped sheet port raises no errors at all", clean_lumped_is_runnable)
check("a circle never reports an inner conductor", circle_sheet_is_never_a_coax)

# --- 6. moving the sheet moves the port -----------------------------------


def sheet_drives_port():
    from EMShrimulator.core.em.csx_builder import _port_geometry
    cs = {}
    a = _port_geometry(coax_prj, coax_prj.ports[0], cs)
    moved = coax_prj.model_copy(deep=True)
    for p in moved.primitives:
        if p.name == "CoaxFeed":
            p.origin = (5.0, 0.0, 0.0)
    b = _port_geometry(moved, moved.ports[0], cs)
    assert a.sheet is not None
    assert abs((b.start[0] - a.start[0]) - 5.0) < 1e-9, (a.start, b.start)


check("moving the sheet moves the port with it", sheet_drives_port)


def line_sets_polarity():
    """The integration line reaches openEMS: it picks the lumped port's axis, and its sign."""
    prj = lumped_prj.model_copy(deep=True)  # FeedGap stands in the ZX plane, z -0.5..1.5, x -0.5..0.5
    port = prj.ports[0]
    built = build_simulation(prj, 1).ports[1]
    assert (built.exc_ny, built.direction) == (2, 1), (built.exc_ny, built.direction)
    port.line_start, port.line_end = port.line_end, port.line_start  # down z
    built = build_simulation(prj, 1).ports[1]
    assert (built.exc_ny, built.direction) == (2, -1), (built.exc_ny, built.direction)
    port.line_start, port.line_end = (-0.5, 0.0, 0.5), (0.5, 0.0, 0.5)  # across x
    built = build_simulation(prj, 1).ports[1]
    assert (built.exc_ny, built.direction) == (0, 1), (built.exc_ny, built.direction)


check("the integration line sets the lumped port's axis and polarity", line_sets_polarity)

# --- 7. the coax mode actually excites, rather than matching zero ---------


def run_briefly(prj, timesteps):
    """Run a few timesteps; (sim dir, openEMS's output captured at fd level)."""
    import os, tempfile
    prj = prj.model_copy(deep=True)
    prj.simulation.max_timesteps = timesteps
    built = build_simulation(prj, 1)
    d = tempfile.mkdtemp()
    log = os.path.join(tempfile.mkdtemp(), "openems.log")
    fd = os.open(log, os.O_RDWR | os.O_CREAT)
    saved = os.dup(1), os.dup(2)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        built.FDTD.Run(d, cleanup=True, verbose=0)
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1]); os.close(fd)
    return d, built, open(log, encoding="utf-8", errors="replace").read()


def coax_mode_really_runs():
    """Run a few timesteps and fail on openEMS parser errors (they only appear at run time)."""
    _d, _built, text = run_briefly(coax_prj, 40)
    assert text.strip(), "captured nothing from openEMS - the fd capture is broken"
    assert "error occurred parsing the mode matching function" not in text, text[-800:]
    assert "Unknown identifier" not in text, text[-800:]


def line_source_lies_on_the_grid():
    """Grid lines go through a lumped line source so openEMS excites it."""
    built = build_simulation(radial_prj, 1)  # held: the grid belongs to it, and is freed with it
    grid = built.CSX.GetGrid()
    lines = {axis: np.array(grid.GetLines(axis)) for axis in "xyz"}
    for axis, value in (("x", 3.0), ("x", 0.0), ("y", 0.0), ("z", 0.0)):
        assert np.min(np.abs(lines[axis] - value)) < 1e-9, f"no {axis} grid line at {value}"


check("coax mode expression parses in a real run", coax_mode_really_runs)
check("a source along a line lies on grid lines, so it excites", line_source_lies_on_the_grid)

print()
for status, label, detail in results:
    print(f"  {status}  {label}")
    if detail:
        print(f"        {detail}")
fails = sum(1 for s, _, _ in results if s == "FAIL")
print(f"\n{len(results) - fails}/{len(results)} passed")
raise SystemExit(1 if fails else 0)
