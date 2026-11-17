"""Typing a name that is not a variable yet asks for it instead of failing.

Off screen, with scripted answers. Run with `.venv/bin/python -m pytest -q tests/test_define_variable.py`.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

from EMShrimulator.app import commands as cmd  # noqa: E402
from EMShrimulator.app.session import ModelerSession  # noqa: E402
from EMShrimulator.app.views.modeler import define_variable as dv  # noqa: E402
from EMShrimulator.core.em.model import Material, MaterialKind, ModelerProject, Parameter  # noqa: E402


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
    project = ModelerProject(name="t", materials=[Material(name="PEC", kind=MaterialKind.PEC)],
                             parameters=[Parameter(name="h", expression="1.5", unit="mm")])
    return ModelerSession("t", project)


@pytest.fixture
def answers(monkeypatch):
    """Answer each dialog by name with (value, unit); a name with no answer is declined."""
    script = {}
    log = []

    def exec_(dialog):
        log.append(dialog.name)
        answer = script.get(dialog.name)
        if answer is None:
            dialog.reject()
            return dialog.result()
        value, unit = answer
        dialog.value.setText(value)
        dialog.unit.setCurrentIndex(dialog.unit.findData(unit))
        dialog.accept()
        if dialog.result() != QDialog.Accepted:
            if not dialog.error.isHidden():  # refused: the user reads the error and gives up
                log.append(("refused", dialog.name, dialog.error.text()))
            dialog.reject()
        return dialog.result()

    monkeypatch.setattr(dv.DefineVariableDialog, "exec", exec_)
    return script, log


def test_number_takes_the_unit_and_formula_is_kept():
    assert dv.stored_expression("40", "mm") == "40"
    assert dv.stored_expression("40", "mil") == "40mil"
    assert dv.stored_expression(".5", "mm") == "0.5"
    assert dv.stored_expression("2.45", "GHz") == "2.45GHz"
    assert dv.stored_expression("2*Wp", "mil") == "2*Wp"
    assert dv.stored_expression("62mil", "mm") == "62mil"
    assert dv.quantity_of_unit("mm") == "length" and dv.quantity_of_unit("GHz") == "frequency"
    assert dv.quantity_of_unit("ohm") == "impedance" and dv.quantity_of_unit("") == "plain"


def test_chain_is_asked_for_in_turn(session, answers):
    script, log = answers
    script.update({"Wsub": ("2*Wp", "mm"), "Wp": ("40", "mil")})
    definitions = dv.define_missing(session, "Wsub + h", quantity="length")
    assert log == ["Wsub", "Wp"]
    # Wp first: Wsub is worked out from it.
    assert [(d.name, d.expression, d.unit) for d in definitions] == [("Wp", "40mil", "mm"), ("Wsub", "2*Wp", "mm")]


def test_declining_any_defines_none(session, answers):
    script, log = answers
    script.update({"Wsub": ("2*Wp", "mm")})
    assert dv.define_missing(session, "Wsub", quantity="length") is None
    assert log == ["Wsub", "Wp"]
    assert [p.name for p in session.project.parameters] == ["h"]


def test_a_loop_is_refused(session, answers):
    script, log = answers
    script.update({"Wsub": ("2*Wp", "mm"), "Wp": ("Wsub/2", "mm")})
    assert dv.define_missing(session, "Wsub", quantity="length") is None
    refused = [entry for entry in log if isinstance(entry, tuple)]
    assert refused and refused[0][1] == "Wp" and "Wsub" in refused[0][2]


def test_a_bad_value_is_refused(session, answers):
    script, log = answers
    script.update({"a": ("sqrt(-h)", "mm")})
    assert dv.define_missing(session, "a", quantity="length") is None
    assert log[-1][0] == "refused" and "sqrt" in log[-1][2]


def test_field_and_its_new_variables_are_one_undo_step(session, answers):
    from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit

    script, _log = answers
    script.update({"cell": ("lam/20", "mm"), "lam": ("120", "mm")})
    field = ExpressionEdit(session, "length", path=lambda: "mesh.max_cell_size")
    field.committed.connect(lambda v: session.push(cmd.SetDesignSetting(session, "mesh.max_cell_size", v)))
    field.edit.setText("cell")
    field._commit()

    project = session.project
    assert [p.name for p in project.parameters] == ["h", "lam", "cell"]
    assert project.mesh.max_cell_size == "cell"
    assert session.resolution.ok and session.resolution.project.mesh.max_cell_size == pytest.approx(6.0)
    assert session.undo_stack.count() == 1
    session.undo_stack.undo()
    assert [p.name for p in session.project.parameters] == ["h"]
    assert session.project.mesh.max_cell_size is None


def test_declined_field_keeps_the_entry_and_its_error(session, answers):
    from EMShrimulator.app.views.modeler.expression_edit import ExpressionEdit

    field = ExpressionEdit(session, "length", path=lambda: "mesh.max_cell_size")
    field.committed.connect(lambda v: session.push(cmd.SetDesignSetting(session, "mesh.max_cell_size", v)))
    field.edit.setText("nope/2")
    field._commit()
    assert session.project.mesh.max_cell_size == "nope/2"
    assert [p.name for p in session.project.parameters] == ["h"]
    assert "nope" in session.resolution.error_for("mesh.max_cell_size")["message"]


def test_parameter_table_asks_too(session, answers, qapp):
    from EMShrimulator.app.models.parameter_model import EXPRESSION
    from EMShrimulator.app.views.modeler.parameter_table import ParameterTable

    script, log = answers
    script.update({"t": ("35", "um")})
    table = ParameterTable(session)
    table.model.setData(table.model.index(0, EXPRESSION), "t*2")
    assert log == []  # the dialog waits for the cell editor to close
    for _ in range(3):
        qapp.processEvents()
    assert log == ["t"]
    assert [(p.name, p.expression, p.unit) for p in session.project.parameters] == [
        ("h", "t*2", "mm"), ("t", "35um", "mm")]
    assert session.resolution.values["h"] == pytest.approx(0.07)
    assert session.undo_stack.count() == 1
