"""Differential pairs: model and migration, edits, validation, and mixed-mode S.

Run with `.venv/bin/python tests/test_diff_pairs.py`; the exit code is the verdict.
"""
import json
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

from EMShrimulator.app.models.results_model import default_enabled, is_reflection
from EMShrimulator.core.em import touchstone
from EMShrimulator.core.em.results import mixed_mode_traces, read_results, renormalize
from EMShrimulator.core.errors import EditError
from EMShrimulator.core.layout import edits, templates
from EMShrimulator.core.layout.model import LayoutPort, LayoutProject
from EMShrimulator.core.layout.project_writer import write_simulation_json
from EMShrimulator.core.layout.validate import _pair_issues

results = []
WORK = Path(tempfile.mkdtemp(prefix="gems-pairs-"))


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


def board(n=4, z=50.0, excite=(0, 1)):
    return LayoutProject(name="p", ports=[
        LayoutPort(name=f"P{i + 1}", number=i, x_mm=i, y_mm=0, impedance=z, excite=i in excite) for i in range(n)])


def raises(fn, text):
    try:
        fn()
    except EditError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected an error about {text!r}")


# ------------------------------------------------------------------ model

def old_projects_migrate():
    old = {"name": "x", "diff_pairs": [{"start_p": 0, "stop_p": 1, "start_n": 2, "stop_n": 3, "name": "DIFF",
                                        "nets": ["/A"]}]}
    project = LayoutProject.model_validate_json(json.dumps(old))
    got = [(p.name, p.positive, p.negative, p.z_diff, p.z_common) for p in project.diff_pairs]
    assert got == [("DIFF start", 0, 2, 100.0, 25.0), ("DIFF stop", 1, 3, 100.0, 25.0)], got
    unnamed = LayoutProject.model_validate({"name": "x", "diff_pairs": [
        {"start_p": 0, "stop_p": 1, "start_n": 2, "stop_n": 3}]})
    assert [p.name for p in unnamed.diff_pairs] == ["D1", "D2"]
    again = LayoutProject.model_validate_json(project.model_dump_json())  # the new form loads as itself
    assert again.diff_pairs == project.diff_pairs
    example = templates.build("differential", "d", WORK / "d")
    assert [(p.positive, p.negative) for p in example.diff_pairs] == [(0, 2), (1, 3)]


# ------------------------------------------------------------------ edits

def edits_keep_pairs_consistent():
    p = board()
    i = edits.add_diff_pair(p, 0, 2)
    pair = p.diff_pairs[i]
    assert (pair.name, pair.positive, pair.negative, pair.z_diff, pair.z_common) == ("D1", 0, 2, 100.0, 25.0)
    raises(lambda: edits.add_diff_pair(p, 2, 3), "already in differential pair D1")
    raises(lambda: edits.add_diff_pair(p, 1, 1), "two different ports")
    raises(lambda: edits.add_diff_pair(p, 1, 9), "no port P10")
    j = edits.add_diff_pair(p, 1)  # the − port left for later
    assert p.diff_pairs[j].name == "D2" and p.diff_pairs[j].negative is None
    raises(lambda: edits.set_diff_pair_field(p, j, "negative", 2), "already in differential pair D1")
    edits.set_diff_pair_field(p, j, "negative", 3)
    raises(lambda: edits.set_diff_pair_field(p, j, "z_diff", 0), "positive")
    raises(lambda: edits.set_diff_pair_field(p, j, "name", "D1"), "already a differential pair named D1")
    edits.swap_diff_pair(p, i)
    assert (p.diff_pairs[i].positive, p.diff_pairs[i].negative) == (2, 0)
    edits.delete_port(p, 0)
    assert p.diff_pairs[i].negative is None, "a deleted port leaves the pair waiting for another"
    mapping = edits.renumber_ports(p)
    assert mapping == {1: 0, 2: 1, 3: 2}
    assert (p.diff_pairs[i].positive, p.diff_pairs[j].positive, p.diff_pairs[j].negative) == (1, 0, 2)
    edits.delete_diff_pair(p, i)
    assert [d.name for d in p.diff_pairs] == ["D2"]
    # Unequal ports: the modes start where two uncoupled ports put them.
    q = board()
    q.ports[0].impedance, q.ports[1].impedance = 40.0, 60.0
    k = edits.add_diff_pair(q, 0, 1)
    assert (q.diff_pairs[k].z_diff, q.diff_pairs[k].z_common) == (100.0, 24.0)


def validation_names_what_is_missing():
    p = board(excite=(0,))
    edits.add_diff_pair(p, 0, 1)
    edits.add_diff_pair(p, 2)
    codes = sorted(i["code"] for i in _pair_issues(p))
    assert codes == ["pair-not-driven", "pair-port-missing"], codes
    p.diff_pairs[1].negative = 0  # shared, bypassing the edit's check
    assert "pair-port-shared" in {i["code"] for i in _pair_issues(p)}
    p.diff_pairs[1].negative = 7
    assert "pair-port-gone" in {i["code"] for i in _pair_issues(p)}
    message = next(i["message"] for i in _pair_issues(board(excite=())) or [{"message": ""}])
    assert message == "", "a board without pairs has no pair issues"
    idle = board(excite=())
    edits.add_diff_pair(idle, 0, 1)
    text = _pair_issues(idle)[0]["message"]
    assert "P1 and P2 are not excited" in text, text


def writer_leaves_pairs_to_the_app():
    p = board()
    edits.add_diff_pair(p, 0, 1)
    write_simulation_json(p, WORK)
    data = json.loads((WORK / "simulation.json").read_text(encoding="utf-8"))
    assert data["differential_pairs"] == [], data["differential_pairs"]


# -------------------------------------------------------------- mixed mode

F = np.linspace(1e8, 5e9, 7)
T = np.exp(-1j * 2 * np.pi * F / 5e9 * 0.3)


def two_lines(reflection=1e-9):
    """Two uncoupled 50 Ω lines: P1/P2 the near end's + and −, P3/P4 the far end's."""
    s = np.zeros((len(F), 4, 4), complex)
    for a, b in ((0, 2), (1, 3)):
        s[:, a, b] = s[:, b, a] = T
    for k in range(4):
        s[:, k, k] = reflection  # driven: a simulated term is never exactly zero
    return s


PAIRS = [dict(name="D1", p=0, n=1, z_diff=100, z_common=25), dict(name="D2", p=2, n=3, z_diff=100, z_common=25)]


def mixed_mode_matches_closed_forms():
    z = 73 + 20j
    one = ((z - 50) / (z + 50)) * np.ones((len(F), 1, 1), complex)
    assert np.allclose(renormalize(one, [50], [75])[:, 0, 0], (z - 75) / (z + 75))
    traces, note = mixed_mode_traces(F, two_lines(), [50] * 4, PAIRS)
    assert not note and len(traces) == 16
    sdd21 = np.array(traces["Sdd21"]["real"]) + 1j * np.array(traces["Sdd21"]["imag"])
    assert np.allclose(sdd21, T)
    assert np.allclose(traces["Scd21"]["mag_lin"], 0) and np.allclose(traces["Sdd11"]["mag_lin"], 0, atol=1e-8)
    assert abs(traces["Sdd11"]["z_real"][0] - 100) < 1e-3 and abs(traces["Scc11"]["z_real"][0] - 25) < 1e-3
    assert traces["Sdd11"]["z_label"] == "Zdiff D1" and traces["Scc22"]["z_label"] == "Zcomm D2"
    # A 100 Ω differential line seen from 90 Ω ports.
    traces, _ = mixed_mode_traces(F, two_lines(), [50] * 4, [dict(p, z_diff=90) for p in PAIRS])
    g = (100 - 90) / (100 + 90)
    s11 = np.array(traces["Sdd11"]["real"]) + 1j * np.array(traces["Sdd11"]["imag"])
    assert np.allclose(s11, g * (1 - T ** 2) / (1 - g ** 2 * T ** 2), atol=1e-6)
    # An imbalance converts modes: a longer − line.
    s = two_lines()
    s[:, 1, 3] = s[:, 3, 1] = T * np.exp(-1j * 0.2)
    traces, _ = mixed_mode_traces(F, s, [50] * 4, PAIRS)
    assert min(traces["Scd21"]["mag_lin"]) > 0.05, "skew should show as mode conversion"


def undriven_ports_leave_terms_out():
    s = two_lines()
    s[:, 2, 2] = s[:, 3, 3] = 0  # the far pair never excited
    traces, note = mixed_mode_traces(F, s, [50] * 4, PAIRS)
    # Sdd12 comes out too, by reciprocity.
    assert {"Sdd11", "Sdd21", "Sdd12", "Scc21"} <= set(traces) and "Sdd22" not in traces and "Scc22" not in traces
    assert note
    traces, note = mixed_mode_traces(F, s, [50] * 4, [dict(p, z_diff=90) for p in PAIRS])
    assert not traces and "P3" in note and "P4" in note, note


def results_carry_the_pairs():
    path = touchstone.write(WORK / "lines", F, two_lines(), [50.0] * 4)
    data = read_results(path, pairs=PAIRS)
    assert data["pairs"] == ["D1", "D2"] and "Sdd21" in data["traces"] and "S21" not in data["traces"]
    assert "S31" in data["traces"]
    assert is_reflection("Sdd11") and is_reflection("Scc22") and not is_reflection("Sdc11")
    assert not is_reflection("Sdd21") and is_reflection("S11")
    assert default_enabled("Sdd11") and default_enabled("Sdd21") and not default_enabled("Scc11")
    assert not default_enabled("Sdc21") and default_enabled("S11") and not default_enabled("S32")


def gain_is_flagged():
    from EMShrimulator.core.em.results import passivity_excess
    assert passivity_excess(two_lines()) == {}, "lossless matched lines conserve power exactly"
    gained = two_lines()
    gained[:, 2, 0] *= 1.04  # a 4 % gain on one line
    excess = passivity_excess(gained)
    assert excess and excess["port"] == 1 and abs(excess["excess"] - 0.0816) < 1e-3, excess
    path = touchstone.write(WORK / "gain", F, gained, [50.0] * 4)
    assert read_results(path)["passivity_excess"]["port"] == 1


if __name__ == "__main__":
    try:
        check("model: four-port lines from older projects become a pair at each end", old_projects_migrate)
        check("edits: one pair per port, polarity swaps, deletes and renumbers follow", edits_keep_pairs_consistent)
        check("validation: missing, shared and undriven ports are named", validation_names_what_is_missing)
        check("writer: gerber2ems gets no pairs (the app computes mixed mode)", writer_leaves_pairs_to_the_app)
        check("mixed mode: uncoupled lines, re-referencing and skew match closed forms",
              mixed_mode_matches_closed_forms)
        check("mixed mode: terms needing an undriven port are left out", undriven_ports_leave_terms_out)
        check("results: Touchstone plus pairs gives Sdd/Scc traces the views understand", results_carry_the_pairs)
        check("results: a board that gains energy is flagged", gain_is_flagged)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    failed = 0
    for status, label, detail in results:
        print(f"{status}  {label}")
        if detail:
            print("      " + detail.replace("\n", "\n      "))
            failed += 1
    raise SystemExit(1 if failed else 0)
