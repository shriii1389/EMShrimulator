"""The 3D Layout's differential pairs and net colours, driven with QTest on a shown window.

Run with `.venv/bin/python tests/test_layout_pairs_ui.py`; the exit code is the verdict.
"""
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

WORK = Path(tempfile.mkdtemp(prefix="gems-pairs-ui-"))
os.environ["GEMS_PROJECTS_DIR"] = str(WORK / "projects")
os.environ["GEMS_TEMP_DIR"] = str(WORK / "tmp")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"))


def main():
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    QApplication.setOrganizationName("EMShrimulator-tests")  # keep QSettings writes out of the user's
    app = QApplication.instance() or QApplication([])
    QSettings().setValue("layout/color_by", "layer")
    from EMShrimulator.app import commands as cmd
    from EMShrimulator.app.session import LayoutSession
    from EMShrimulator.app.views.layout.layout_view import LayoutView
    from EMShrimulator.core import project as storage
    from EMShrimulator.core.layout import templates

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

    name = "pairs"
    d = storage.projects_dir() / name
    d.mkdir(parents=True)
    project = templates.build("differential", name, d)
    project.diff_pairs = []  # start with none: the test makes them
    (d / "kind.txt").write_text("layout")
    storage.save_project(name, project)
    session = LayoutSession(name, storage.load_project(name))
    view = LayoutView(session)
    view.resize(1400, 860)
    view.show()
    canvas = view.canvas
    pump(0.5, until=lambda: view.board.get("ready") and not canvas._inflight)
    canvas.fit()
    pump(0.3)

    def click_port(number, button=Qt.LeftButton):
        item = canvas._port_items[number]
        point = canvas.mapFromScene(item.pos())
        QTest.mouseMove(canvas.viewport(), point)
        pump(0.05)
        QTest.mouseClick(canvas.viewport(), button, Qt.NoModifier, point)
        pump(0.1)

    def pair_tool_makes_a_pair_by_two_clicks():
        view.tool_actions["pair"].trigger()
        assert canvas.tool == "pair"
        click_port(2)  # + first
        assert canvas._pair_first == 2
        click_port(0)  # then −
        pump(0.2)
        pairs = session.project.diff_pairs
        assert [(p.positive, p.negative) for p in pairs] == [(2, 0)], pairs
        assert canvas.tool == "select" and not view.tool_actions["pair"].isChecked()
        assert session.selection is not None and session.selection.kind == "diff_pair"
        assert view.inspector.isVisible() and view.inspector.pair_editor.isVisible()
        assert ("diff_pair", 0) in set(view.tree.keys())
        item = canvas._pair_items[0]
        assert item.isVisible() and item.selected_pair and not item.boundingRect().isEmpty()

    def a_port_in_a_pair_is_refused_for_another():
        before = list(session.project.diff_pairs)
        view._make_pair(0, 1)  # P1 is already the − of D1
        pump(0.1)
        assert session.project.diff_pairs == before

    def the_inspector_edits_the_pair():
        editor = view.inspector.pair_editor
        editor.z_diff.committed.emit(90.0)
        pump(0.1)
        assert session.project.diff_pairs[0].z_diff == 90.0
        editor.positive.committed.emit(1)  # P2 for +
        pump(0.1)
        assert session.project.diff_pairs[0].positive == 1
        session.push(cmd.SwapDiffPair(session, 0))
        pump(0.1)
        assert (session.project.diff_pairs[0].positive, session.project.diff_pairs[0].negative) == (0, 1)
        session.select("layout_port", 0)
        pump(0.1)
        assert canvas._port_items[0]._label_text()[1].endswith("+ of D1"), "a selected port names its pair"

    def clicking_the_bracket_selects_the_pair():
        session.select(None)
        pump(0.1)
        item = canvas._pair_items[0]
        t = item.deviceTransform(canvas.viewportTransform())
        geometry = item._geometry(t)
        centre = geometry["centre"].toPoint()
        QTest.mouseClick(canvas.viewport(), Qt.LeftButton, Qt.NoModifier, centre)
        pump(0.1)
        assert session.selection is not None and session.selection.kind == "diff_pair", session.selection

    def delete_key_removes_the_pair_not_its_ports():
        session.select("diff_pair", 0)
        view.show_page("board")
        canvas.setFocus()
        pump(0.1)
        QTest.keyClick(canvas, Qt.Key_Delete)
        pump(0.2)
        assert session.project.diff_pairs == [] and len(session.project.ports) == 4
        assert not canvas._pair_items

    def colour_by_net_and_a_highlight():
        net = next(action for action in view.color_group.actions() if action.data() == "net")
        net.trigger()
        pump(0.2, until=lambda: not canvas._inflight)
        assert canvas.color_by == "net" and canvas._styled, "no layer was coloured by net"
        view.show_key(("net", "/DIFF_P"))
        pump(0.2, until=lambda: not canvas._inflight)
        assert canvas.highlight == "/DIFF_P" and view.tree.current_key() == ("net", "/DIFF_P")
        assert canvas.focus_net("/DIFF_P")
        view.show_key(("nets",))
        assert canvas.highlight is None
        next(action for action in view.color_group.actions() if action.data() == "layer").trigger()
        pump(0.1)
        assert canvas.color_by == "layer"

    check("pair tool: click the + port, then the − port", pair_tool_makes_a_pair_by_two_clicks)
    check("pairs: a port already paired is refused for another pair", a_port_in_a_pair_is_refused_for_another)
    check("inspector: mode impedance, + port and polarity edit the pair", the_inspector_edits_the_pair)
    check("canvas: clicking a pair's label selects it", clicking_the_bracket_selects_the_pair)
    check("Delete removes the selected pair and keeps its ports", delete_key_removes_the_pair_not_its_ports)
    check("colour by net, highlight and zoom to a net from the tree", colour_by_net_and_a_highlight)
    view.shutdown()
    view.deleteLater()
    pump(0.2)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    failed = 0
    for status, label, detail in results:
        print(f"{status}  {label}")
        if detail:
            print("      " + detail.replace("\n", "\n      "))
            failed += 1
    raise SystemExit(1 if failed or not results else 0)
