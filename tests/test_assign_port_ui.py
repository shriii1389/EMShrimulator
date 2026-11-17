"""Assigning a port through the inspectors: draw a sheet, assign a port, set its integration line.

Off screen. Run with `.venv/bin/python -m pytest -q tests/test_assign_port_ui.py`.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from EMShrimulator.app import commands as cmd  # noqa: E402
from EMShrimulator.app.session import ModelerSession  # noqa: E402
from EMShrimulator.core.em.model import (  # noqa: E402
    Material, MaterialKind, ModelerProject, PortKind, PrimitiveKind, SheetShape,
)


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
    sheet_id = s.push(cmd.AddPrimitive(s, PrimitiveKind.SHEET, SheetShape.RECTANGLE))
    # Upright in the YZ plane from z = 0 to 1.5: a feed from a ground plane up to a strip.
    s.push(cmd.SetPrimitiveFields(s, sheet_id, {"normal_axis": "x", "sheet_size": (3.0, 1.5),
                                                "origin": (0.0, 0.0, 0.75)}, "sheet"))
    s.sheet_id = sheet_id
    return s


def _load_sheet(session):
    from EMShrimulator.app.views.modeler.primitive_editor import PrimitiveEditor

    editor = PrimitiveEditor(session)
    index = next(i for i, p in enumerate(session.project.primitives) if p.id == session.sheet_id)
    editor.load(session.project.primitives[index], index)
    return editor, editor.page_for[PrimitiveKind.SHEET]


def _load_port(session, port_id):
    from EMShrimulator.app.views.modeler.port_editor import PortEditor

    editor = PortEditor(session)
    index = next(i for i, p in enumerate(session.project.ports) if p.id == port_id)
    editor.load(session.project.ports[index], index)
    return editor


def test_sheet_inspector_assigns_a_port_and_selects_it(session):
    editor, page = _load_sheet(session)
    assert not page.assign.isHidden() and page.edit_port.isHidden()
    page.assign.click()
    (port,) = session.project.ports
    assert port.kind == PortKind.LUMPED and port.sheet == "rectangle_1"
    assert session.selection is not None and (session.selection.kind, session.selection.key) == ("port", port.id)
    _editor, page = _load_sheet(session)
    assert page.assign.isHidden() and page.edit_port.text() == "Edit port1"


def _type(vector, index, text):
    field = vector.fields[index]
    field.edit.setText(text)
    field._commit()


def test_a_new_port_waits_for_its_line(session):
    port_id = session.push(cmd.AssignPort(session, session.sheet_id, PortKind.LUMPED))
    editor = _load_port(session, port_id)
    assert not editor.line_heading.isHidden()
    assert editor._line_form.isRowVisible(editor.line_start)
    assert not editor._mode_form.isRowVisible(editor.direction)  # the line decides the direction now
    assert editor.line_state.text().startswith("Not drawn yet") and editor.line_state.property("role") == "error"
    assert [f.edit.text() for f in editor.line_start.fields] == ["", "", ""]  # empty, not a made-up 0
    assert not editor.reverse_line.isEnabled()


def test_typing_the_line_ends(session):
    port_id = session.push(cmd.AssignPort(session, session.sheet_id, PortKind.LUMPED))
    editor = _load_port(session, port_id)
    _type(editor.line_start, 2, "0")
    editor = _load_port(session, port_id)
    _type(editor.line_end, 2, "1.5")
    port = session.project.ports[0]
    assert port.line_start == (0.0, 0.0, 0.0) and port.line_end == (0.0, 0.0, 1.5)
    editor = _load_port(session, port_id)
    assert editor.line_state.text() == "Runs +Z, 1.5 mm · excites the whole sheet."

    _type(editor.line_end, 2, "0.75")  # halfway up: a source along the line itself
    editor = _load_port(session, port_id)
    assert "along this line" in editor.line_state.text()
    editor.reverse_line.click()
    port = session.project.ports[0]
    assert port.line_start == (0.0, 0.0, 0.75) and port.line_end == (0.0, 0.0, 0.0)
    editor = _load_port(session, port_id)
    assert editor.line_state.text().startswith("Runs −Z")
    editor.clear_line.click()
    assert session.project.ports[0].line_start is None


def test_draw_in_3d_asks_for_this_port(session):
    port_id = session.push(cmd.AssignPort(session, session.sheet_id, PortKind.LUMPED))
    editor = _load_port(session, port_id)
    asked = []
    editor.drawLineRequested.connect(asked.append)
    editor.draw_line.click()
    assert asked == [port_id]


def test_guided_ports_have_no_line(session):
    port_id = session.push(cmd.AssignPort(session, session.sheet_id, PortKind.RECT_WAVEGUIDE))
    editor = _load_port(session, port_id)
    assert editor.line_heading.isHidden()


def test_deleting_the_sheet_is_one_undo_step_with_its_port(session):
    session.push(cmd.AssignPort(session, session.sheet_id, PortKind.LUMPED))
    session.push(cmd.DeletePrimitive(session, session.sheet_id))
    assert session.project.ports == []
    session.undo_stack.undo()
    assert len(session.project.ports) == 1 and any(p.id == session.sheet_id for p in session.project.primitives)
