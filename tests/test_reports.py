"""Report numbers against closed forms. No openEMS, no window.

Run with `.venv/bin/python -m pytest -q tests/test_reports.py`.
"""
import math

import numpy as np
import pytest

from EMShrimulator.core.em import edits, farfield as ff, nearfield as nf, reports
from EMShrimulator.core.em.model import ModelerProject, ResultPlot
from EMShrimulator.core.em.results import admittance_traces, fastest_rise_s, impedance_traces, read_results
from EMShrimulator.core.em.touchstone import write_touchstone
from EMShrimulator.core.errors import EditError

THETA = np.arange(0.0, 180.0 + 1, 2.0)
PHI = np.arange(0.0, 360.0 + 1, 2.0)


def _fields(e_theta, e_phi, freq=3e9):
    return {"frequency_hz": np.array([freq]), "theta_deg": THETA, "phi_deg": PHI, "radius_m": np.array([1.0]),
            "e_theta": e_theta[None].astype(np.complex64), "e_phi": e_phi[None].astype(np.complex64)}


def _entry(e_theta, e_phi, freq=3e9, **powers):
    """farfield.json's entry for a pattern, as the solver reduces one."""
    intensity = (np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2) / (2 * ff.ETA0)
    prad = float((intensity * ff.sphere_weights(THETA, PHI)).sum())
    entry = ff.summarise(freq, THETA, PHI, 4 * np.pi * intensity / prad)
    entry.update(radiated_power_w=prad, **powers)
    return entry


def _dipole():
    """A short z-dipole: E_θ ∝ sin θ, no E_φ. D = 1.5 (1.76 dBi), linearly polarised."""
    t = np.radians(THETA)[:, None] * np.ones((1, PHI.size))
    return np.sin(t) + 0j, np.zeros_like(t, dtype=complex)


def _rhcp():
    """A right-hand circular pattern towards +z: E_φ = -jE_θ everywhere, so E_LHCP = 0."""
    t = np.radians(THETA)[:, None] * np.ones((1, PHI.size))
    e_theta = (1 + np.cos(t)) / 2 + 0j
    return e_theta, -1j * e_theta


# --- plots: presets, displays, old projects ---------------------------------------------------------------

def test_presets_make_reports_of_the_right_kind():
    p = ModelerProject(name="t")
    for group, presets in edits.PLOT_MENU:
        for preset in presets:
            plot = edits.find_plot(p, edits.add_plot(p, preset))
            assert plot.display in edits.DISPLAYS[plot.kind], (preset, plot.kind, plot.display)
    z = edits.find_plot(p, edits.add_plot(p, "z"))
    assert z.function == "re_im"
    cross = edits.find_plot(p, edits.add_plot(p, "cross_pol"))
    assert cross.cross_pol and cross.component == "total"  # the co-polarisation is left for the user to choose
    assert "Choose the co-polarisation" in reports.cross_needed(cross)


def test_a_display_must_suit_its_kind():
    p = ModelerProject(name="t")
    plot_id = edits.add_plot(p, "s_db")
    edits.set_plot_field(p, plot_id, "display", "smith")
    with pytest.raises(EditError):
        edits.set_plot_field(p, plot_id, "display", "spherical")
    with pytest.raises(EditError):
        edits.set_plot_field(p, plot_id, "nonsense", 1)
    with pytest.raises(EditError):
        edits.set_plot_field(p, plot_id, "kind", "farfield")


def test_plots_saved_before_reports_load_as_reports():
    raw = ('{"schema_version": 1, "name": "old", "plots": ['
           '{"id": "a", "name": "p", "kind": "pattern_polar", "cut": "e", "quantity": "gain"},'
           '{"id": "b", "name": "m", "kind": "pattern_map"},'
           '{"id": "c", "name": "z", "kind": "z", "traces": ["Z11"]},'
           '{"id": "d", "name": "s", "kind": "s_phase", "log_x": true}]}')
    a, b, c, d = ModelerProject.model_validate_json(raw).plots
    assert (a.kind, a.display, a.cuts, a.quantity) == ("farfield", "polar", [0.0], "gain")
    assert (b.kind, b.display) == ("farfield", "contour")
    assert (c.kind, c.display, c.function, c.traces) == ("z", "rect", "re_im", ["Z11"])
    assert (d.kind, d.function, d.log_x) == ("s", "phase", True)
    again = ModelerProject.model_validate_json(ModelerProject.model_validate_json(raw).model_dump_json())
    assert [p.model_dump() for p in again.plots] == [p.model_dump() for p in ModelerProject.model_validate_json(raw).plots]


def test_the_layout_keeps_reports_too():
    from EMShrimulator.core.layout.model import LayoutProject

    p = LayoutProject(name="board")
    plot_id = edits.add_plot(p, "tdr")
    assert edits.find_plot(p, plot_id).kind == "tdr"
    assert LayoutProject.model_validate_json(p.model_dump_json()).plots[0].id == plot_id


# --- far field -------------------------------------------------------------------------------------------

def test_the_sphere_weights_sum_to_four_pi():
    assert ff.sphere_weights(THETA, PHI).sum() == pytest.approx(4 * np.pi, rel=1e-3)


def test_a_short_dipoles_directivity_and_polarisation():
    e_theta, e_phi = _dipole()
    fields, entry = _fields(e_theta, e_phi), _entry(e_theta, e_phi)
    assert entry["dmax_dbi"] == pytest.approx(10 * math.log10(1.5), abs=0.01)
    total, unit, why = ff.pattern(entry, fields, "directivity", "total")
    assert unit == "dBi" and not why and total.max() == pytest.approx(1.761, abs=0.01)
    theta, _u, _w = ff.pattern(entry, fields, "directivity", "theta")
    assert np.allclose(theta, total, atol=1e-3)
    phi, _u, _w = ff.pattern(entry, fields, "directivity", "phi")
    assert phi.max() < -100
    for circular in ("rhcp", "lhcp"):  # a linear wave is half of each
        grid, _u, _w = ff.pattern(entry, fields, "directivity", circular)
        assert grid.max() == pytest.approx(total.max() - 10 * math.log10(2), abs=0.01)
    ratio, unit, _w = ff.pattern(entry, fields, "axial_ratio", "total")
    assert unit == "dB" and ratio.min() == pytest.approx(ff.AXIAL_RATIO_CAP_DB)


def test_a_circular_patterns_axial_ratio_and_hand():
    e_theta, e_phi = _rhcp()
    fields, entry = _fields(e_theta, e_phi), _entry(e_theta, e_phi)
    ratio, _u, _w = ff.pattern(entry, fields, "axial_ratio", "total")
    assert np.nanmax(ratio[:45]) == pytest.approx(0.0, abs=1e-3)  # 0 dB: circular wherever it radiates
    total, _u, _w = ff.pattern(entry, fields, "directivity", "total")
    right, _u, _w = ff.pattern(entry, fields, "directivity", "rhcp")
    left, _u, _w = ff.pattern(entry, fields, "directivity", "lhcp")
    assert right.max() == pytest.approx(total.max(), abs=1e-3) and left.max() < -100


def test_gain_and_realized_gain_follow_the_ports_power():
    e_theta, e_phi = _dipole()
    entry = _entry(e_theta, e_phi)
    prad = entry["radiated_power_w"]
    entry.update(accepted_power_w=prad / 0.8, incident_power_w=prad / 0.5)
    fields = _fields(e_theta, e_phi)
    d, _u, _w = ff.pattern(entry, fields, "directivity", "total")
    g, _u, _w = ff.pattern(entry, fields, "gain", "total")
    r, _u, _w = ff.pattern(entry, fields, "realized_gain", "total")
    assert g.max() - d.max() == pytest.approx(10 * math.log10(0.8), abs=1e-3)
    assert r.max() - d.max() == pytest.approx(10 * math.log10(0.5), abs=1e-3)
    # rE is for 1 W incident: |E| scales by 1/√P_incident.
    re, unit, _w = ff.pattern(entry, fields, "re", "total", db=False)
    assert unit == "V" and re.max() == pytest.approx(1.0 / math.sqrt(prad / 0.5), rel=1e-3)


def test_old_runs_without_fields_still_give_the_total():
    e_theta, e_phi = _dipole()
    entry = _entry(e_theta, e_phi)
    grid, unit, why = ff.pattern(entry, None, "directivity", "total")
    assert unit == "dBi" and not why and grid.max() == pytest.approx(1.761, abs=0.01)
    grid, _unit, why = ff.pattern(entry, None, "directivity", "rhcp")
    assert grid is None and "Run the design again" in why


def test_a_theta_cut_joins_both_halves_of_the_plane():
    # Strong at φ = 0° only: negative cut angles are the φ = 180° half.
    t = np.radians(THETA)[:, None]
    p = np.radians(PHI)[None, :]
    grid = np.sin(t) * (1 + np.cos(p))
    angles, values, held = ff.cut(grid, THETA, PHI, "theta", 0.0)
    assert held == 0.0 and angles[0] == -180.0 and angles[-1] == 180.0 and np.all(np.diff(angles) > 0)
    assert values[np.argmin(np.abs(angles - 90))] == pytest.approx(2.0, abs=1e-6)   # φ = 0° side
    assert values[np.argmin(np.abs(angles + 90))] == pytest.approx(0.0, abs=1e-6)   # φ = 180° side


def test_cuts_and_cross_polarisation_reports():
    e_theta, e_phi = _rhcp()
    farfield = {"frequencies": [_entry(e_theta, e_phi)], "fields": _fields(e_theta, e_phi)}
    plot = ResultPlot(name="p", kind="farfield", display="polar", component="rhcp", cross_pol=True, cuts=[0, 90])
    report = reports.pattern_cuts(plot, farfield)
    assert not report.message
    assert [s.dashed for s in report.series] == [False, True, False, True]
    assert report.series[0].label.startswith("dB(Directivity RHCP) co")
    assert report.series[1].label.startswith("dB(Directivity LHCP) cross")
    columns, rows = reports.table(report)
    assert columns[0] == "θ (°)" and len(rows) == report.series[0].x.size


def test_antenna_parameters_over_frequency():
    e_theta, e_phi = _rhcp()
    entries = []
    for f in (2e9, 2.5e9, 3e9):
        entry = _entry(e_theta, e_phi, f)
        entry.update(accepted_power_w=2 * entry["radiated_power_w"], incident_power_w=4 * entry["radiated_power_w"],
                     efficiency=0.5)
        entries.append(entry)
    fields = {**_fields(e_theta, e_phi), "frequency_hz": np.array([2e9, 2.5e9, 3e9]),
              "e_theta": np.repeat(e_theta[None], 3, 0).astype(np.complex64),
              "e_phi": np.repeat(e_phi[None], 3, 0).astype(np.complex64)}
    report, table = reports.antenna(ResultPlot(name="a", kind="antenna", display="rect"),
                                    {"frequencies": entries, "fields": fields}, ["peak_directivity", "efficiency"])
    assert [s.label for s in report.series] == ["Peak directivity", "Radiation efficiency"]
    assert table["radiated_power"]["values"] == pytest.approx([0.25] * 3)  # per 1 W incident
    assert table["axial_ratio"]["values"] == pytest.approx([0.0] * 3, abs=1e-3)
    assert "1 W incident" in report.note


# --- network parameters --------------------------------------------------------------------------------

def test_y_is_the_inverse_of_z():
    zmat = np.array([[110.0, 100.0], [100.0, 110.0]])
    z0 = 50.0
    smat = (zmat - z0 * np.eye(2)) @ np.linalg.inv(zmat + z0 * np.eye(2))
    s = np.tile(smat.astype(complex), (3, 1, 1))
    z = impedance_traces(s, [z0, z0], {})
    y = admittance_traces(s, [z0, z0], z)
    ymat = np.array([[y[f"Y{i}{j}"]["real"][0] + 1j * y[f"Y{i}{j}"]["imag"][0] for j in (1, 2)] for i in (1, 2)])
    assert np.allclose(ymat, np.linalg.inv(zmat))


def test_input_admittance_when_only_one_port_was_driven():
    z = {"Zin1": {"z_real": [75.0], "z_imag": [25.0]}}
    s = np.zeros((1, 2, 2), dtype=complex)
    s[:, 0, 0] = 0.2
    y = admittance_traces(s, [50.0, 50.0], z)
    assert list(y) == ["Yin1"]
    assert complex(y["Yin1"]["real"][0], y["Yin1"]["imag"][0]) == pytest.approx(1 / (75 + 25j))


def test_solution_reports_from_a_touchstone(tmp_path):
    freqs = np.linspace(0.01e9, 5e9, 200)
    # A matched 50 Ω line 30 mm long in air: S11 = 0, S21 = e^{-jβl}.
    s21 = np.exp(-1j * 2 * np.pi * freqs * 0.03 / 299_792_458.0)
    s11 = np.full_like(s21, 1e-4)
    path = write_touchstone(freqs, {1: {1: s11, 2: s21}, 2: {1: s21, 2: s11}}, [1, 2], 50.0, tmp_path / "line")
    data = read_results(path)
    assert set(data["y_traces"]) == {"Y11", "Y12", "Y21", "Y22"}
    keys = reports.trace_keys("s", data)
    phase = reports.solution(ResultPlot(name="p", kind="s", function="phase"), data, ["S21"])
    delay = -np.gradient(np.radians(phase.series[0].y), 2 * np.pi * freqs)
    assert np.median(delay) == pytest.approx(0.03 / 299_792_458.0, rel=1e-3)
    both = reports.solution(ResultPlot(name="z", kind="z", function="re_im"), data, ["Z11"])
    assert [s.label for s in both.series] == ["Re(Z11)", "Im(Z11)"] and both.series[1].dashed
    assert "S11" in keys and reports.default_traces("s", keys) == ["S11", "S21", "S22"]


def test_tdr_rise_time_and_its_limit():
    assert fastest_rise_s(10e9) == pytest.approx(1.238e-10, rel=0.01)
    freqs = np.linspace(0.0, 10e9, 401)
    data = {"freq_hz": freqs.tolist(), "epsilon_eff": 1.0,
            "traces": {"S11": {"real": [0.2] * 401, "imag": [0.0] * 401, "ref_ohm": 50.0}}}
    sharp = reports.tdr(ResultPlot(name="t", kind="tdr"), data, ["S11"])
    slow = reports.tdr(ResultPlot(name="t", kind="tdr", rise_time_ps=500), data, ["S11"])
    too_fast = reports.tdr(ResultPlot(name="t", kind="tdr", rise_time_ps=20), data, ["S11"])
    z = 50 * 1.2 / 0.8  # a constant reflection of 0.2: 75 Ω
    assert sharp.series[0].y[len(sharp.series[0].y) // 2] == pytest.approx(z, rel=0.01)
    assert slow.series[0].y[len(slow.series[0].y) // 2] == pytest.approx(z, rel=0.01)
    assert "shorter than the sweep resolves" in too_fast.note
    # 25 MHz spacing resolves 40 ns, so the trace stops there.
    timed = reports.tdr(ResultPlot(name="t", kind="tdr", tdr_axis="time"), data, ["S11"])
    assert timed.series[0].x[-1] <= 40.0 + 0.1


def test_tdr_finds_a_load_down_a_line():
    # 30 mm of 50 Ω air line into 100 Ω: 50 Ω out to 30 mm, then 100 Ω.
    freqs = np.linspace(0.0, 20e9, 801)
    gamma = (100 - 50) / (100 + 50) * np.exp(-2j * 2 * np.pi * freqs * 0.03 / 299_792_458.0)
    data = {"freq_hz": freqs.tolist(), "epsilon_eff": 1.0,
            "traces": {"S11": {"real": gamma.real.tolist(), "imag": gamma.imag.tolist(), "ref_ohm": 50.0}}}
    report = reports.tdr(ResultPlot(name="t", kind="tdr"), data, ["S11"])
    x, z = report.series[0].x, report.series[0].y
    assert z[np.argmin(np.abs(x - 15))] == pytest.approx(50.0, abs=1.0)
    assert z[np.argmin(np.abs(x - 60))] == pytest.approx(100.0, abs=1.5)
    edge = x[np.argmax(z > 75.0)]
    assert edge == pytest.approx(30.0, abs=1.0)


# --- near field ----------------------------------------------------------------------------------------

def _box():
    x = np.linspace(-5, 5, 11)
    y = np.linspace(-2, 2, 5)
    z = np.linspace(0, 1, 3)
    field = np.zeros((1, 3, z.size, y.size, x.size), dtype=np.complex64)
    field[0, 2] = (x[None, None, :] + 10) * (1 + 0 * y[None, :, None]) * (1 + z[:, None, None])  # Ez = (x+10)(1+z)
    return {"x_mm": x, "y_mm": y, "z_mm": z, "frequency_hz": np.array([1e9]), "field": field}


def test_a_plane_and_a_line_of_a_box_dump(tmp_path):
    box = _box()
    cut = nf.plane(box, 0, "total", None, None)
    assert cut["held"] == "z" and cut["held_mm"] == 0.5  # the thinnest axis, through the middle
    assert (cut["u"], cut["v"]) == ("x", "y") and cut["values"].shape == (5, 11)
    assert cut["values"][0, 0] == pytest.approx(5 * 1.5)
    across = nf.plane(box, 0, "z", "x", 5.0)
    assert across["held_mm"] == 5.0 and (across["u"], across["v"]) == ("y", "z")
    assert across["values"][2, 0] == pytest.approx(15 * 2.0)
    plot = ResultPlot(name="n", kind="nearfield", display="rect", db=False)
    line = reports.near_line(plot, box, "E", incident_w=4.0)
    assert line.x_label == "x (mm)" and line.series[0].y[-1] == pytest.approx(15 * 1.5 / 2)  # 1 W incident
    np.savez_compressed(tmp_path / "nearfield.npz", **box)
    (tmp_path / "nearfield").mkdir()
    np.savez_compressed(tmp_path / "nearfield" / "probe__port2.npz", **box)
    assert nf.available(tmp_path) == [{"dump": "probe", "port": 2}]
    assert nf.load(tmp_path, "probe", 2)["field"].shape == box["field"].shape


def test_incident_power_comes_from_the_runs_summary():
    summary = {"port_powers": {"freq_hz": [1e9, 2e9], "1": {"incident": [2.0, 4.0], "accepted": [1.0, 1.0]}}}
    assert reports.incident_power(summary, 1, 1.5e9) == pytest.approx(3.0)
    assert reports.incident_power(summary, 2, 1.5e9) is None
    assert reports.incident_power(None, 1, 1e9) is None


def test_csv_export():
    report = reports.Report(series=[reports.Series("S11", "S11", np.array([1e9, 2e9]), np.array([-10.0, -20.0]),
                                                   unit="dB")],
                            x_label="Frequency", x_unit="Hz", y_unit="dB")
    columns, rows = reports.table(report)
    assert columns == ["Frequency (GHz)", "S11 (dB)"] and rows == [["1", "-10"], ["2", "-20"]]
    assert reports.to_csv(columns, rows) == "Frequency (GHz),S11 (dB)\n1,-10\n2,-20\n"


# --- traces: any quantity beside any other ----------------------------------------------------------------

def _one_port(tmp_path):
    freqs = np.linspace(1e9, 3e9, 21)
    s11 = 0.2 * np.exp(-1j * freqs / 1e9)
    return read_results(write_touchstone(freqs, {1: {1: s11}}, [1], 50.0, tmp_path / "one")), freqs, s11


def test_s11_and_z11_share_a_graph_on_two_axes(tmp_path):
    data, freqs, s11 = _one_port(tmp_path)
    p = ModelerProject(name="t")
    plot = edits.find_plot(p, edits.add_plot(p, "match"))
    report = reports.frequency_report(plot, data, None)
    assert [s.label for s in report.series] == ["dB(S11)", "Re(Z11)", "Im(Z11)"]
    assert [s.axis for s in report.series] == ["y1", "y2", "y2"]
    assert report.y_unit == "dB" and report.y2_unit == "Ω" and "Impedance" in report.y2_label
    z = 50 * (1 + s11) / (1 - s11)
    assert np.allclose(report.series[1].y, z.real) and np.allclose(report.series[2].y, z.imag)
    # A trace put on Y1 by hand stays there.
    plot.curves[2].axis = "y1"
    report = reports.frequency_report(plot, data, None)
    assert report.series[2].axis == "y1" and report.y_unit == "" and "S-parameter" in report.y_label


def test_traces_are_checked_as_they_are_set():
    p = ModelerProject(name="t")
    plot_id = edits.add_plot(p, "s_db")
    edits.set_plot_field(p, plot_id, "curves", [{"category": "z", "term": "Z11", "function": "re"}])
    (curve,) = edits.find_plot(p, plot_id).curves
    assert curve.category == "z" and curve.axis == "auto" and curve.visible
    with pytest.raises(EditError):
        edits.set_plot_field(p, plot_id, "curves", [{"category": "nonsense"}])
    assert edits.find_plot(p, plot_id).curves[0].term == "Z11"


def test_a_one_port_z11_is_its_input_impedance_when_not_every_port_was_driven(tmp_path):
    freqs = np.linspace(1e9, 3e9, 5)
    s11 = np.full(5, 0.2 + 0j)
    s21 = np.full(5, 0.5 + 0j)
    data = read_results(write_touchstone(freqs, {1: {1: s11, 2: s21}}, [1, 2], 50.0, tmp_path / "two"))
    assert "Z11" not in data["z_traces"] and "Zin1" in data["z_traces"]
    plot = ResultPlot(name="m", kind="s", curves=[{"category": "z", "term": "Z11", "function": "re"}])
    (series,) = reports.frequency_report(plot, data, None).series
    assert series.label == "Re(Zin1)" and series.y[0] == pytest.approx(75.0)


def test_far_field_traces_against_frequency_at_a_direction(tmp_path):
    data, _freqs, _s11 = _one_port(tmp_path)
    e_theta, e_phi = _dipole()
    entries = [_entry(e_theta, e_phi, f) for f in (1.5e9, 2e9, 2.5e9)]
    fields = {**_fields(e_theta, e_phi), "frequency_hz": np.array([1.5e9, 2e9, 2.5e9]),
              "e_theta": np.repeat(e_theta[None], 3, 0).astype(np.complex64),
              "e_phi": np.repeat(e_phi[None], 3, 0).astype(np.complex64)}
    farfield = {"frequencies": entries, "fields": fields}
    plot = ResultPlot(name="g", kind="antenna", display="rect", curves=[
        {"category": "s", "term": "S11", "function": "db"},
        {"category": "directivity", "term": "total", "function": "db", "theta_deg": 90, "phi_deg": 0},
        {"category": "directivity", "term": "total", "function": "db", "theta_deg": 0, "phi_deg": 0}])
    report = reports.frequency_report(plot, data, farfield)
    s11, side, top = report.series
    assert side.markers and side.x.tolist() == [1.5e9, 2e9, 2.5e9]
    assert side.y == pytest.approx([10 * math.log10(1.5)] * 3, abs=0.01)  # broadside of a short dipole
    assert top.y.max() < -50  # its axis: a null
    assert side.label == "dB(Directivity Total) @ θ 90° φ 0°" and side.axis == "y2" and s11.axis == "y1"
    columns, rows = reports.table(report)
    assert len(rows) == 21 + 0 and rows[5][2] != "" and rows[0][2] == ""  # far-field points where they fall


def test_an_elevation_pattern_at_every_frequency():
    e_theta, e_phi = _dipole()
    entries = [_entry(e_theta, e_phi, f) for f in (2e9, 3e9)]
    farfield = {"frequencies": entries,
                "fields": {**_fields(e_theta, e_phi), "frequency_hz": np.array([2e9, 3e9]),
                           "e_theta": np.repeat(e_theta[None], 2, 0).astype(np.complex64),
                           "e_phi": np.repeat(e_phi[None], 2, 0).astype(np.complex64)}}
    p = ModelerProject(name="t")
    plot = edits.find_plot(p, edits.add_plot(p, "elevation"))
    plot.curves[0].category = "directivity"  # the synthetic run records no port powers for gain
    plot.all_frequencies = True
    report = reports.pattern_report(plot, farfield)
    assert len(report.series) == 2 * 2  # φ = 0° and 90°, at both frequencies
    assert report.series[0].label == "dB(Directivity Total) · φ = 0° · 2 GHz"
    azimuth = edits.find_plot(p, edits.add_plot(p, "azimuth"))
    azimuth.curves[0].category = "directivity"
    (cut,) = reports.pattern_report(azimuth, farfield).series
    assert cut.x[0] == 0 and cut.x[-1] == 360 and np.ptp(cut.y) < 1e-6  # a dipole is round in azimuth


def test_axial_ratio_beside_gain_goes_to_the_second_axis():
    e_theta, e_phi = _rhcp()
    farfield = {"frequencies": [_entry(e_theta, e_phi)], "fields": _fields(e_theta, e_phi)}
    plot = ResultPlot(name="p", kind="farfield", display="rect", cuts=[0.0], curves=[
        {"category": "directivity", "term": "rhcp", "function": "db"},
        {"category": "axial_ratio", "term": "total", "function": "value"}])
    report = reports.pattern_report(plot, farfield)
    assert [s.axis for s in report.series] == ["y1", "y2"]
    assert report.y_unit == "dBi" and report.y2_unit == "dB"
