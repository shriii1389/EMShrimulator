"""Coordinate systems through the inspector: what each kind shows, and edits made there.

Off screen. Run with `.venv/bin/python -m pytest -q tests/test_coordinate_system_ui.py`.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from EMShrimulator.app import commands as cmd  # noqa: E402
from EMShrimulator.app.session import ModelerSession  # noqa: E402
from EMShrimulator.core.em.model import Material, MaterialKind, ModelerProject, PrimitiveKind  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication([])
    if not isinstance(app, QApplication):
        pytest.skip("a QCoreApplication from another test is already running")
    return app


@pytest.fixture
def session(qapp):
    s = ModelerSession("t", ModelerProject(name="t", materials=[Material(name="PEC", kind=MaterialKind.PEC)]))
    box_id = s.push(cmd.AddPrimitive(s, PrimitiveKind.BOX))
    s.push(cmd.SetPrimitiveFields(s, box_id, {"p1": (0.0, 0.0, 0.0), "p2": (10.0, 4.0, 2.0)}, "box"))
    s.box_id = box_id
    return s


def _editor(session, name):
    from EMShrimulator.app.views.modeler.coordinate_editor import CoordinateSystemEditor

    editor = CoordinateSystemEditor(session)
    project = session.project
    cs = next((c for c in project.coordinate_systems if c.name == name), None)
    editor.load(cs, project.coordinate_systems.index(cs) if cs is not None else -1)
    return editor


def test_a_new_system_is_the_working_one_and_one_undo_step(session):
    name = session.push(cmd.AddObjectCoordinateSystem(session, session.box_id, "face", (0.5, 0.5, 1.0)))
    assert session.project.working_cs == name
    resolved = session.resolution.project.coordinate_systems[0]
    assert resolved.placement is not None and resolved.placement[3] == 5.0  # the top face's centre, x
    session.undo_stack.undo()
    assert session.project.coordinate_systems == [] and session.project.working_cs is None


def test_an_object_system_shows_its_object_and_point(session):
    name = session.push(cmd.AddObjectCoordinateSystem(session, session.box_id, "face", (0.5, 0.5, 1.0)))
    editor = _editor(session, name)
    form = editor.form
    assert form.isRowVisible(editor.object_box) and form.isRowVisible(editor.anchor_row)
    assert not form.isRowVisible(editor.parent_box)
    assert editor.anchor.text() == "face centre (+Z)" and editor.working_note.isVisibleTo(editor)
    assert form.isRowVisible(editor.origin) and not form.isRowVisible(editor.angles)  # offset only, at first
    asked = []
    editor.anchorPickRequested.connect(lambda *args: asked.append(args))
    editor.pick.click()
    assert asked == [(name, session.box_id)]


def test_choosing_rotated_shows_the_angles(session):
    name = session.push(cmd.AddCoordinateSystem(session, None, "offset", (1.0, 0.0, 0.0)))
    editor = _editor(session, name)
    assert editor.form.isRowVisible(editor.parent_box) and not editor.form.isRowVisible(editor.object_box)
    editor.mode.committed.emit("both")
    editor = _editor(session, name)
    assert editor.form.isRowVisible(editor.origin) and editor.form.isRowVisible(editor.angles)
    field = editor.angles.fields[2]
    field.edit.setText("90")
    field._commit()
    cs = session.project.coordinate_systems[0]
    assert cs.mode == "both" and cs.angles == (0.0, 0.0, 90.0)


def test_a_loop_is_refused_and_nothing_changes(session):
    name = session.push(cmd.AddObjectCoordinateSystem(session, session.box_id, "centre", None))
    before = session.project
    with pytest.raises(Exception, match="loop"):
        session.push(cmd.SetPrimitiveField(session, session.box_id, "coordinate_system", name))
    assert session.project is before


def test_the_global_system(session):
    editor = _editor(session, None)
    assert not editor.form.isRowVisible(editor.mode) and editor.delete.isHidden()
    assert "fixed global system" in editor.about.text()
