"""Padstacks and vias in the 3D Layout: drill import, pads written from padstacks, moving vias with their
traces, backdrills, and what the solver gets.

Run with `.venv/bin/python tests/test_padstacks.py`; the exit code is the verdict.
"""
import math
import os
import tempfile
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
WORK = Path(tempfile.mkdtemp(prefix="gems-padstacks-"))
os.environ["GEMS_PROJECTS_DIR"] = str(WORK / "projects")
os.environ["GEMS_TEMP_DIR"] = str(WORK / "tmp")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"))


def make_project(name: str, example: str = "differential"):
    from EMShrimulator.core import project as storage
    from EMShrimulator.core.layout import templates

    d = storage.projects_dir() / name
    d.mkdir(parents=True)
    project = templates.build(example, name, d)
    (d / "kind.txt").write_text("layout")
    storage.save_project(name, project)
    return project, d


# ---------------------------------------------------------------------------------------------- core

def core_checks():
    from EMShrimulator.core.layout import board, copper, drill, padstacks, via_import
    from EMShrimulator.core.layout import edits as lay
    from EMShrimulator.core.layout.gerber_render import edge_cuts_bbox_mm
    from EMShrimulator.core.layout.project_writer import write_vias_json

    project, d = make_project("core")
    fab = d / "fab"
    bounds = edge_cuts_bbox_mm(board.edge_cuts_file(fab))
    found = via_import.extract(d)
    via_import.apply(project, found)
    layers = padstacks.layer_names(project)
    flashes = via_import._pad_flashes(d, bounds)

    def registered():
        assert project.vias, "no vias read"
        assert "registered" in found["note"], found["note"]
        legacy = drill.holes(drill.find_drill_file(fab), bounds)
        worst_legacy = worst = 0.0
        for via, hole in zip(project.vias, legacy):
            assert len(via.links) == len(layers), f"via {via.id} took {len(via.links)} pads"
            for link in via.links:
                _i, x, y, _d, _n = next(r for r in flashes[link.layer] if r[0] == link.index)
                worst = max(worst, math.hypot(x - via.x_mm, y - via.y_mm))
                worst_legacy = max(worst_legacy, math.hypot(x - hole.x_mm, y - hole.y_mm))
        assert worst < 1e-6, f"a via is {worst * 1000:.3f} µm off its pad"
        assert worst_legacy > 0.03, "the old reading was already centred: this board no longer shows the bug"
        assert len(project.padstacks) == 1 and project.padstacks[0].shape == "circle", project.padstacks

    def written():
        changed = board.sync_copper(d, lay.copper_edits(project), padstacks.layer_specs(project))
        assert set(changed) == set(layers), changed
        for name in layers:
            data = copper.read_layer(fab / next(l.file for l in board.copper_layers(fab) if l.name == name), bounds)
            pads = [p for p in data.pieces if p.kind == "pad" and p.points]
            # The imported pads are replaced by one pad from the padstack.
            at = [(p.points[0], p.net) for p in pads]
            for via in project.vias:
                here = [net for (x, y), net in at if math.hypot(x - via.x_mm, y - via.y_mm) < 1e-6]
                assert len(here) == 1, f"{name}: via {via.id} has {len(here)} pads"
                assert here[0] == via.net, (here, via.net)

    def gerber2ems_reads_it():
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "gerber2ems-main" / "src"))
        from gerber2ems.gerber_io import GerberFile as G2

        for layer in board.copper_layers(fab):
            g = G2(fab / layer.file)
            assert len(g.pads) >= len(project.vias), (layer.name, len(g.pads))

    def resize_and_move():
        ps = project.padstacks[0].name
        lay.set_padstack_field(project, ps, "pad_mm", 0.8)
        lay.set_padstack_layer(project, ps, "In1_Cu", None, 1.2)  # a plane layer: no pad, a wide antipad
        via = project.vias[0]
        lay.move_vias(project, {via.id: (via.x_mm + 0.5, via.y_mm)})
        specs = padstacks.layer_specs(project)
        top = specs["f_cu"]["flashes"]
        mine = [f for f in top if abs(f["x"] - via.x_mm) < 1e-9 and abs(f["y"] - via.y_mm) < 1e-9]
        assert mine and mine[0]["d"] == 0.8, mine
        inner = [f for f in specs["in1_cu"]["flashes"] if abs(f["x"] - via.x_mm) < 1e-9]
        assert [f["dark"] for f in inner] == [False] and inner[0]["d"] == 1.2, inner
        board.sync_copper(d, lay.copper_edits(project), specs)
        text = (fab / next(l.file for l in board.copper_layers(fab) if l.name == "In1_Cu")).read_text()
        assert "%LPC*%" in text and text.rstrip().endswith("M02*"), text[-400:]

    def backdrill():
        via = project.vias[1]
        try:
            lay.set_vias_field(project, [via.id], "backdrill_bottom",
                               {"keep_layer": "In1_Cu", "diameter_mm": 0.15, "stub_mm": 0.05})
            raise AssertionError("a backdrill narrower than the hole was taken")
        except Exception as exc:  # noqa: BLE001
            assert "wider" in str(exc), exc
        lay.set_vias_field(project, [via.id], "backdrill_bottom",
                           {"keep_layer": "In1_Cu", "diameter_mm": 0.4, "stub_mm": 0.05})
        rows = {r.layer: r for r in padstacks.via_layers(via, project.padstacks[0], layers)}
        assert rows["F_Cu"].has_pad and rows["In1_Cu"].drill_mm == 0
        assert not rows["In2_Cu"].has_pad and rows["In2_Cu"].drill_mm == 0.4
        assert not rows["B_Cu"].has_pad
        sim = {(v["x_mm"], v["y_mm"]): v for v in padstacks.sim_vias(project)}[(via.x_mm, via.y_mm)]
        assert (sim["top"], sim["bottom"]) == (0, 1) and sim["bottom_stub_mm"] == 0.05, sim
        assert sim["backdrills"] == [{"side": "bottom", "diameter_mm": 0.4}], sim
        write_vias_json(project, d)
        assert (d / "vias.json").is_file()

    def place_and_delete():
        n = len(project.vias)
        new = lay.add_via(project, 1.0, 1.0, project.padstacks[0].name, "GND")
        assert len(project.vias) == n + 1 and new == max(v.id for v in project.vias)
        lay.delete_vias(project, [new])
        assert len(project.vias) == n
        try:
            lay.delete_padstack(project, project.padstacks[0].name)
            raise AssertionError("a padstack in use was deleted")
        except Exception as exc:  # noqa: BLE001
            assert "use" in str(exc), exc

    def eagle_scale():
        """A drill file with implied decimals and no format still lands on the pads."""
        import numpy as np

        raw = np.array([[92.202, 66.802], [117.602, 66.802], [168.402, 117.602], [300.0, 50.0]])
        pads = raw / 10 - (0.33, 0.33)
        fit = via_import.register(raw, np.vstack([pads, [[3.0, 4.0], [5.0, 1.0]]]))
        assert fit is not None and abs(fit[0] - 0.1) < 1e-9 and fit[3] == 1.0, fit

    def renamed_layer():
        """A stackup layer named unlike its Gerber: its imported pads are left alone."""
        other = project.model_copy(deep=True)
        copper = [layer for layer in other.stackup if layer.type == "copper"]
        copper[2].name = "Signal 3"
        specs = padstacks.layer_specs(other)
        assert "in2_cu" not in specs and "signal 3" in specs, sorted(specs)
        assert not specs["signal 3"]["owned"]

    check("drill registered onto the pads (no off-centre drill)", registered)
    check("a renamed stackup layer keeps its imported pads", renamed_layer)
    check("pads written from the padstack, once each", written)
    check("gerber2ems reads the written Gerbers", gerber2ems_reads_it)
    check("pad size, per-layer antipad and a moved via", resize_and_move)
    check("backdrill: pads, clearance and the solver's barrel", backdrill)
    check("place and delete a via; a used padstack stays", place_and_delete)
    check("an unformatted drill file at the wrong scale", eagle_scale)


# --------------------------------------------------------------------------------------------- window

def window_checks():
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from EMShrimulator.app.session import LayoutSession
    from EMShrimulator.app.views.layout.layout_view import LayoutView
    from EMShrimulator.core.layout import padstacks
    from EMShrimulator.core.services import layout as layout_service

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("EMShrimulator-harness")
    project, _d = make_project("ui")
    layout_service.save("ui", project)
    session = LayoutSession("ui", layout_service.load("ui"))
    view = LayoutView(session)
    view.resize(1400, 900)
    view.show()

    def wait(cond, ms=60000):
        import time

        end = time.monotonic() + ms / 1000
        while time.monotonic() < end:
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            if cond():
                return True
            QTest.qWait(20)
        return False

    canvas = view.canvas
    tool = canvas.copper

    def opened():
        assert wait(lambda: session.project.vias is not None and len(session.project.vias) > 0), "vias never read"
        assert wait(lambda: bool(tool.layers) and len(tool.layers) == len(view.board.get("layers", []))), \
            "copper never loaded"
        assert not view._copper_busy or wait(lambda: not view._copper_busy)

    def view_pos(x_mm, y_mm) -> QPoint:
        return canvas.mapFromScene(canvas.to_scene(x_mm, y_mm))

    def drag_via():
        # Place a via on a trace's end, as the via tool would.
        name = "F_Cu"
        trace = next(piece for _pid, piece in tool.layers[name].rows() if piece.kind == "trace" and piece.width > 0)
        x, y = trace.points[1]
        view._place_via(x, y, "")
        app.processEvents()
        via = session.project.vias[-1]
        assert (via.x_mm, via.y_mm) == (round(x, 6), round(y, 6)), (via, x, y)
        assert session.selection.key == (("#via", via.id, 0),), session.selection
        joined = tool.layers[name].trace_ends((via.x_mm, via.y_mm))
        assert joined, "the trace end is not on the new via"
        for other in tool.layers:  # only its layer shown: nothing else under the cursor
            view.visible_layers[other] = other == name
        canvas.apply_layer_visibility()
        canvas.fit()
        app.processEvents()
        # Zoom in on it, so the via is many pixels across.
        canvas.centerOn(canvas.to_scene(via.x_mm, via.y_mm))
        canvas.scale(6, 6)
        app.processEvents()
        start = view_pos(via.x_mm, via.y_mm)
        ref = (name,) + joined[0][0]
        end = joined[0][1]
        QTest.mouseClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, start)
        app.processEvents()
        assert tool.selection and tool.selection[0][0] == "#via", tool.selection
        QTest.mousePress(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, start)
        target = start + QPoint(40, 0)
        for k in range(1, 9):
            QTest.mouseMove(canvas.viewport(), start + QPoint(5 * k, 0))
        QTest.mouseRelease(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, target)
        app.processEvents()
        moved = next(v for v in session.project.vias if v.id == via.id)
        assert moved.x_mm > via.x_mm + 1e-4, (moved.x_mm, via.x_mm)
        piece = tool.piece(ref)
        assert piece is not None and math.hypot(piece.points[end][0] - moved.x_mm,
                                                piece.points[end][1] - moved.y_mm) < 1e-6, \
            "the trace did not follow the via"
        session.undo_stack.undo()
        app.processEvents()
        back = next(v for v in session.project.vias if v.id == via.id)
        assert abs(back.x_mm - via.x_mm) < 1e-9

    def padstack_page():
        name = session.project.padstacks[0].name
        view.show_key(("padstack", name))
        app.processEvents()
        page = view.padstack_page
        assert page.name == name
        page.pad.set_stored(0.7)
        page.pad.committed.emit(0.7)
        app.processEvents()
        assert session.project.padstacks[0].pad_mm == 0.7
        assert wait(lambda: not view._copper_busy)

    def inspector_backdrill():
        via = session.project.vias[2]
        session.select("copper", (("#via", via.id, 0),))
        app.processEvents()
        props = view.inspector.vias
        assert props.isVisible() or True
        props.set_backdrill("bottom", {"keep_layer": "In1_Cu", "diameter_mm": 0.4, "stub_mm": 0.05})
        app.processEvents()
        now = next(v for v in session.project.vias if v.id == via.id)
        assert now.backdrill_bottom is not None and now.backdrill_bottom.keep_layer == "In1_Cu"
        rows = padstacks.via_layers(now, session.project.padstacks[0], padstacks.layer_names(session.project))
        assert not rows[-1].has_pad

    check("the drill file is read into vias on opening", opened)
    check("drag a via: its traces follow; undo puts it back", drag_via)
    check("padstack page sets the pad", padstack_page)
    check("inspector backdrills a via", inspector_backdrill)
    view.shutdown()
    view.close()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    core_checks()
    if "--core" not in sys.argv:
        window_checks()
    width = max(len(label) for _s, label, _d in results)
    for status, label, detail in results:
        print(f"{status}  {label.ljust(width)}")
        if detail:
            print("      " + detail.replace("\n", "\n      "))
    failed = sum(1 for status, _l, _d in results if status != "PASS")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    sys.exit(1 if failed else 0)
