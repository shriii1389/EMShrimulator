"""Copper editing in the 3D Layout: selecting, reshaping, splitting and deleting traces, polygons and pads.

Run with `.venv/bin/python tests/test_copper_edit.py`; the exit code is the verdict.
"""
import math
import os
import tempfile
import time
import traceback
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="gems-copper-"))
os.environ["GEMS_PROJECTS_DIR"] = str(WORK / "projects")
os.environ["GEMS_TEMP_DIR"] = str(WORK / "tmp")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"))


def close(a, b, tol=1e-6):
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


# ---------------------------------------------------------------------------------------------- core

def core_checks():
    from EMShrimulator.core import project as storage
    from EMShrimulator.core.layout import board, copper, templates
    from EMShrimulator.core.layout import copper_geometry as geo
    from EMShrimulator.core.layout import edits as lay
    from EMShrimulator.core.layout.gerber_render import edge_cuts_bbox_mm

    name = "core"
    d = storage.projects_dir() / name
    d.mkdir(parents=True)
    project = templates.build("meander_tight", name, d)
    (d / "kind.txt").write_text("layout")
    fab = d / "fab"
    bounds = edge_cuts_bbox_mm(board.edge_cuts_file(fab))
    layers = {layer.name: layer.file for layer in board.copper_layers(fab)}
    top = fab / layers["F_Cu"]
    imported = top.read_bytes()
    pieces = copper.read_layer(top, bounds).pieces

    def geometry():
        a, b = (0.0, 0.0), (2.0, 0.0)
        for h in (0.4, -0.4, 1.0, 1.7, -2.5):  # minor, semicircle, major arcs, both ways
            cx, cy, r = geo.arc_center(a, b, h)
            assert abs(math.hypot(cx, cy) - r) < 1e-9 and abs(math.hypot(cx - 2, cy) - r) < 1e-9
            mid = geo.edge_point(a, b, h, 0.5)
            assert abs(mid[1] - h) < 1e-9, (h, mid)  # + bulges left of a -> b, here +y
            assert abs(geo.height_of(a, b, (cx, cy), h > 0) - h) < 1e-9
            h1, h2 = geo.split_height(h, a, b, 0.3)
            m = geo.edge_point(a, b, h, 0.3)
            assert abs(geo.edge_length(a, m, h1) + geo.edge_length(m, b, h2) - geo.edge_length(a, b, h)) < 1e-9
            t, dist = geo.project_on_edge(a, b, h, mid)
            assert abs(t - 0.5) < 1e-9 and dist < 1e-9
        assert abs(geo.edge_length(a, b, 1.0) - math.pi) < 1e-9  # a semicircle of radius 1

    def imported_shapes():
        kinds = {p.kind for p in pieces}
        assert kinds == {geo.TRACE, geo.PAD}, kinds
        arcs = [p for p in pieces if p.kind == geo.TRACE and abs(p.heights[0]) > 1e-9]
        assert arcs, "the meander's bends should read as arc segments"
        inner = copper.read_layer(fab / layers["In1_Cu"], bounds).pieces
        assert any(p.kind == geo.REGION for p in inner), "the inner layer's plane should be a polygon"

    def roundtrip():
        layer = copper.read_layer(top, bounds)
        same = {i: [p.as_edit()] for i, p in enumerate(layer.pieces) if p.editable and p.points}
        out = WORK / "same.gbr"
        out.write_text(copper.patch(layer, same), encoding="utf-8")
        again = copper.read_layer(out, bounds).pieces
        assert len(again) == len(layer.pieces)
        for p, q in zip(layer.pieces, again):
            assert p.kind == q.kind and p.net == q.net and abs(p.width - q.width) < 1e-7
            assert all(close(a, b, 2e-6) for a, b in zip(p.points, q.points))

    trace = next(i for i, p in enumerate(pieces) if p.kind == geo.TRACE and abs(p.heights[0]) < 1e-9
                 and p.length() > 0.5)
    piece = pieces[trace]
    a, b = piece.points

    def edit_and_sync():
        moved = geo.Piece(**{**piece.__dict__, "points": [a, (b[0] + 0.25, b[1] + 0.1)], "width": 0.3})
        lay.set_copper(project, "F_Cu", {trace: [moved.as_edit()]})
        changed = board.sync_copper(d, lay.copper_edits(project))
        assert changed == ["F_Cu"], changed
        assert (fab / ".original" / layers["F_Cu"]).read_bytes() == imported, "the import must be kept aside"
        text = top.read_text(encoding="utf-8")
        assert "%TO.N," in text, "net attributes must survive"
        after = copper.read_layer(top, bounds).pieces
        assert len(after) == len(pieces)
        assert close(after[trace].points[1], (b[0] + 0.25, b[1] + 0.1), 2e-6)
        assert abs(after[trace].width - 0.3) < 1e-6 and after[trace].net == piece.net
        others = [i for i in range(len(pieces)) if i != trace and pieces[i].points]
        assert all(after[i].points == pieces[i].points or all(close(p, q, 2e-6) for p, q in
                   zip(after[i].points, pieces[i].points)) for i in others), "nothing else may move"
        assert board.sync_copper(d, lay.copper_edits(project)) == [], "an unchanged edit is not written again"

    def split_bend_delete():
        m = geo.edge_point(a, b, 0.0, 0.5)
        first = geo.Piece(**{**piece.__dict__, "points": [a, m], "heights": [0.2]})
        second = geo.Piece(**{**piece.__dict__, "points": [m, b], "heights": [0.0]})
        pad = next(i for i, p in enumerate(pieces) if p.kind == geo.PAD)
        moved_pad = pieces[pad].moved(0.5, 0.0)
        lay.set_copper(project, "F_Cu", {trace: [first.as_edit(), second.as_edit()], pad: [moved_pad.as_edit()]})
        board.sync_copper(d, lay.copper_edits(project))
        after = copper.read_layer(top, bounds).pieces
        assert len(after) == len(pieces) + 1, "a split adds one object"
        bent = after[trace]
        assert abs(bent.heights[0] - 0.2) < 1e-6 and close(bent.points[1], m, 2e-6)
        assert close(after[trace + 1].points[0], m, 2e-6)
        pads = [p for p in after if p.kind == geo.PAD]
        assert any(close(p.points[0], moved_pad.points[0], 2e-6) for p in pads)
        lay.set_copper(project, "F_Cu", {pad: []})
        board.sync_copper(d, lay.copper_edits(project))
        assert len(copper.read_layer(top, bounds).pieces) == len(pieces), "split +1, pad -1"

    def plane_vertex():
        inner = fab / layers["In1_Cu"]
        planes = copper.read_layer(inner, bounds).pieces
        index = next(i for i, p in enumerate(planes) if p.kind == geo.REGION)
        plane = planes[index]
        points = list(plane.points)
        points[0] = (points[0][0] + 0.3, points[0][1] + 0.3)
        lay.set_copper(project, "In1_Cu", {index: [geo.Piece(**{**plane.__dict__, "points": points}).as_edit()]})
        assert set(board.sync_copper(d, lay.copper_edits(project))) == {"In1_Cu"}
        again = copper.read_layer(inner, bounds).pieces[index]
        assert again.kind == geo.REGION and close(again.points[0], points[0], 2e-6)
        assert len(again.points) == len(points)

    def restore():
        lay.revert_copper(project)
        assert set(board.sync_copper(d, {})) == {"F_Cu", "In1_Cu"}
        assert top.read_bytes() == imported, "restoring gives back the imported file byte for byte"
        assert not (fab / ".original" / layers["F_Cu"]).exists()

    def simulation_sees_edits():
        lay.set_copper(project, "F_Cu", {trace: []})
        board.sync_copper(d, lay.copper_edits(project))
        written = board.ensure_simulation_pngs(d, 20.0)
        png = d / "ems" / "geometry" / next(w for w in written if "F_Cu" in w)
        edited = png.read_bytes()
        lay.revert_copper(project)
        board.sync_copper(d, {})
        board.ensure_simulation_pngs(d, 20.0)
        assert png.read_bytes() != edited, "the solver's copper must follow the edit"

    def bad_pieces():
        from EMShrimulator.core.errors import EditError
        for bad in ({"kind": "trace", "points": [(0, 0), (1, 0)], "width_mm": 0},
                    {"kind": "region", "points": [(0, 0), (1, 0)]},
                    {"kind": "blob", "points": [(0, 0)]}):
            try:
                lay.set_copper(project, "F_Cu", {0: [bad]})
            except EditError:
                continue
            raise AssertionError(f"{bad} should be refused")

    def chain():
        layer = geo.EditedLayer("F_Cu", pieces)
        whole = layer.chain((trace, 0))
        assert len(whole) > 3, "a meander is many segments joined end to end"
        assert all(layer.get(pid).net == piece.net for pid in whole)
        ends = layer.trace_ends(b, [(trace, 0)])
        assert ends, "the segment's end should be joined to the next"

    check("arc geometry: centre, height, split and length agree", geometry)
    check("a KiCad meander reads as straight and arc segments; planes as polygons", imported_shapes)
    check("every object rewritten as itself reads back the same", roundtrip)
    check("an edited trace is written in place, nets kept, the import kept aside", edit_and_sync)
    check("split, bend, moved and deleted copper land in the Gerber", split_bend_delete)
    check("a plane's vertex moves", plane_vertex)
    check("restoring gives back the imported bytes", restore)
    check("the solver's geometry follows copper edits", simulation_sees_edits)
    check("impossible copper is refused", bad_pieces)
    check("a trace's segments chain end to end", chain)


# ---------------------------------------------------------------------------------------------- window

def window_checks():
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    QApplication.setOrganizationName("EMShrimulator-tests")
    app = QApplication.instance() or QApplication([])
    QSettings().setValue("layout/color_by", "layer")
    from EMShrimulator.app.session import LayoutSession
    from EMShrimulator.app.views.layout.layout_view import LayoutView
    from EMShrimulator.core import project as storage
    from EMShrimulator.core.layout import board, copper, templates
    from EMShrimulator.core.layout import copper_geometry as geo
    from EMShrimulator.core.layout.gerber_render import edge_cuts_bbox_mm

    def pump(seconds=0.0, until=None, timeout=120.0):
        start = time.monotonic()
        while True:
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            done = until() if until else True
            if done and time.monotonic() - start >= seconds:
                return
            if time.monotonic() - start > timeout:
                raise TimeoutError("the window did not settle")
            time.sleep(0.01)

    name = "ui"
    d = storage.projects_dir() / name
    d.mkdir(parents=True)
    project = templates.build("meander_tight", name, d)
    (d / "kind.txt").write_text("layout")
    storage.save_project(name, project)
    session = LayoutSession(name, storage.load_project(name))
    view = LayoutView(session)
    view.resize(1400, 860)
    view.show()
    canvas = view.canvas
    tool = canvas.copper
    pump(0.5, until=lambda: view.board.get("ready") and not canvas._inflight and len(tool.layers) >= 4)
    # The drill file is read into vias on opening; undo returns to the Gerber as it is then.
    pump(0.5, until=lambda: session.project.vias is not None and not view._copper_busy and not canvas._inflight)
    fab = d / "fab"
    top = fab / next(layer.file for layer in board.copper_layers(fab) if layer.name == "F_Cu")
    imported = top.read_bytes()
    bounds = edge_cuts_bbox_mm(board.edge_cuts_file(fab))

    def settled():
        return not view._copper_busy and not canvas._inflight and not tool.pending

    def view_pos(x, y) -> QPoint:
        return canvas.mapFromScene(canvas.to_scene(x, y))

    def press_drag(start: QPoint, end: QPoint, modifiers=Qt.NoModifier):
        viewport = canvas.viewport()
        QTest.mouseMove(viewport, start)
        pump(0.05)
        QTest.mousePress(viewport, Qt.LeftButton, modifiers, start)
        steps = 6
        for k in range(1, steps + 1):
            p = QPoint(start.x() + (end.x() - start.x()) * k // steps, start.y() + (end.y() - start.y()) * k // steps)
            QTest.mouseMove(viewport, p)
            pump(0.02)
        QTest.mouseRelease(viewport, Qt.LeftButton, modifiers, end)
        pump(0.1)

    def selected():
        s = session.selection
        return tuple(s.key) if s is not None and s.kind == "copper" else ()

    layer = tool.layers["F_Cu"]
    target = next(pid for pid, p in layer.rows() if p.kind == geo.TRACE and abs(p.heights[0]) < 1e-9
                  and p.length() > 1.0)
    ref = ("F_Cu", target[0], target[1])
    piece = layer.get(target)

    def zoom_to(piece):
        x0, y0, x1, y1 = piece.bounds()
        view._show_layers("F_Cu")  # only the top layer: what is clicked is unambiguous
        pump(0.1)
        from PySide6.QtCore import QRectF
        rect = QRectF(canvas.to_scene(x0, y1), canvas.to_scene(x1, y0)).normalized()
        m = max(rect.width(), rect.height()) * 1.2
        canvas.fitInView(rect.adjusted(-m, -m, m, m), Qt.KeepAspectRatio)
        canvas._geometry_changed()
        pump(0.2)

    def click_selects():
        zoom_to(piece)
        mid = geo.edge_point(*piece.edges()[0], 0.5)
        QTest.mouseClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, view_pos(*mid))
        pump(0.1)
        assert selected() == (ref,), selected()
        assert view.inspector.isVisible() and view.inspector.copper.isVisible()
        assert "Trace segment" in view.inspector.heading.text(), view.inspector.heading.text()
        assert abs(view.inspector.copper.width.stored() - piece.width) < 1e-6

    def drag_end():
        a, b = piece.points
        joined = tool.layers["F_Cu"].trace_ends(b, [target])
        ppm = canvas.base_ppm * canvas.zoom()
        new_b = (b[0] + 40 / ppm, b[1] - 25 / ppm)  # far from any snap target
        press_drag(view_pos(*b), view_pos(*new_b), Qt.ControlModifier)  # Ctrl: no snapping
        edit = next(e for e in session.project.copper_edits if e.index == target[0])
        end = edit.objects[0].points[1]
        assert math.hypot(end[0] - new_b[0], end[1] - new_b[1]) < 3 / ppm, (end, new_b)
        for pid, which in joined:  # the joined segments followed
            other = tool.piece(("F_Cu", pid[0], pid[1]))
            assert close(other.points[which], end, 1e-6), "a joined segment must follow the dragged end"
        assert abs(edit.objects[0].width_mm - piece.width) < 1e-9, "a drag keeps the width"
        pump(0.2, until=settled)
        # By position: via pads are appended, so object indices differ from the imported file's.
        again = copper.read_layer(top, bounds).pieces
        assert any(p.kind == geo.TRACE and close(p.points[1], end, 2e-6) for p in again), "the Gerber must follow"

    def bend():
        current = tool.piece(ref)
        a, b = current.points
        mid = geo.edge_point(a, b, 0.0, 0.5)
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ppm = canvas.base_ppm * canvas.zoom()
        off = 30 / ppm
        to = (mid[0] - dy / length * off, mid[1] + dx / length * off)  # to the left: + arc height
        press_drag(view_pos(*mid), view_pos(*to), Qt.ControlModifier)
        h = tool.piece(ref).heights[0]
        assert h > 0 and abs(h - off) < 3 / ppm, (h, off)
        assert "Arc segment" in view.inspector.heading.text()

    def width_in_inspector():
        box = view.inspector.copper.width
        box.setFocus()
        box.lineEdit().selectAll()
        QTest.keyClicks(box.lineEdit(), "321")
        QTest.keyClick(box.lineEdit(), Qt.Key_Return)
        pump(0.1)
        assert abs(tool.piece(ref).width - 0.321) < 1e-9, tool.piece(ref).width
        canvas.setFocus()

    def split():
        before = tool.piece(ref)
        tool.split(ref)
        pump(0.1)
        refs = selected()
        assert len(refs) == 2 and refs[0] == ref, refs
        first, second = tool.piece(refs[0]), tool.piece(refs[1])
        assert close(first.points[1], second.points[0]) and close(first.points[0], before.points[0])
        assert close(second.points[1], before.points[1])
        assert abs(first.length() + second.length() - before.length()) < 1e-6
        pump(0.2, until=settled)

    def undo_all():
        while session.undo_stack.canUndo():
            session.undo_stack.undo()
        pump(0.2, until=settled)
        pump(0.3, until=lambda: settled() and top.read_bytes() == imported, timeout=60)
        assert not session.project.copper_edits

    def delete_key():
        mid = geo.edge_point(*piece.edges()[0], 0.5)
        QTest.mouseClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, view_pos(*mid))
        pump(0.1)
        assert selected() == (ref,)
        canvas.setFocus()
        QTest.keyClick(canvas, Qt.Key_Delete)
        pump(0.1)
        edit = next(e for e in session.project.copper_edits if e.index == target[0])
        assert edit.objects == [] and selected() == ()
        session.undo_stack.undo()
        pump(0.2, until=settled)

    def plane_vertex():
        view._show_layers("In1_Cu")
        pump(0.1)
        plane_layer = tool.layers["In1_Cu"]
        pid, plane = next((pid, p) for pid, p in plane_layer.rows() if p.kind == geo.REGION)
        canvas.fit()
        pump(0.2)
        # Click inside the plane near a vertex, away from pad clearances.
        x0, y0, x1, y1 = plane.bounds()
        inside = next(p for p in ((x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy) for fx in (0.03, 0.1, 0.5, 0.9)
                                  for fy in (0.03, 0.1, 0.5, 0.9))
                      if tool.pick(*p) and tool.pick(*p)[0][1:] == pid)
        QTest.mouseClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, view_pos(*inside))
        pump(0.1)
        plane_ref = ("In1_Cu",) + pid
        assert selected() == (plane_ref,), selected()
        assert "Polygon" in view.inspector.heading.text()
        handles = [h for h in tool.handles() if h.kind == "vertex"]
        assert len(handles) == len(plane.points)
        corner = handles[0].point
        ppm = canvas.base_ppm * canvas.zoom()
        to = (corner[0] + 20 / ppm, corner[1] + 20 / ppm)
        press_drag(view_pos(*corner), view_pos(*to), Qt.ControlModifier)
        moved = tool.piece(plane_ref).points[0]
        assert math.hypot(moved[0] - to[0], moved[1] - to[1]) < 3 / ppm, (moved, to)
        pump(0.2, until=settled)

    def escape_cancels():
        current = tool.piece(("In1_Cu",) + next(pid for pid, p in tool.layers["In1_Cu"].rows()
                                                 if p.kind == geo.REGION))
        corner = current.points[1]
        viewport = canvas.viewport()
        start = view_pos(*corner)
        QTest.mouseMove(viewport, start)
        QTest.mousePress(viewport, Qt.LeftButton, Qt.ControlModifier, start)
        QTest.mouseMove(viewport, QPoint(start.x() + 30, start.y() + 30))
        pump(0.05)
        count = session.undo_stack.count()
        QTest.keyClick(canvas, Qt.Key_Escape)
        QTest.mouseRelease(viewport, Qt.LeftButton, Qt.ControlModifier, QPoint(start.x() + 30, start.y() + 30))
        pump(0.1)
        assert session.undo_stack.count() == count, "Esc must drop the drag"

    def saved_and_reopened():
        session.save()
        view.shutdown()
        view.close()
        view.deleteLater()
        pump(0.2)
        reopened = storage.load_project(name)
        assert reopened.copper_edits, "the edits are kept with the project"

    check("clicking a trace segment selects it and shows it in the inspector", click_selects)
    check("dragging a segment's end moves it; joined segments follow; width kept", drag_end)
    check("dragging the bend handle makes an arc", bend)
    check("the inspector sets the width", width_in_inspector)
    check("a split segment is two, joined, the same length", split)
    check("undo puts the Gerber back as it was before the edits", undo_all)
    check("the Delete key deletes the selected copper", delete_key)
    check("a plane's vertex drags", plane_vertex)
    check("Esc drops a drag", escape_cancels)
    check("copper edits are saved with the project", saved_and_reopened)


def main():
    core_checks()
    window_checks()
    for status, label, detail in results:
        print(f"{status}  {label}")
        if detail:
            print("      " + detail.replace("\n", "\n      "))
    failed = sum(1 for r in results if r[0] == "FAIL")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    from EMShrimulator.core import procpool
    procpool.shutdown()
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
