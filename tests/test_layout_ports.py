"""3D Layout ports: snap targets, automatic references, circuit-port geometry, the ΔS stop."""
import json
import math
import shutil
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import EMShrimulator.bootstrap  # noqa: F401  (openEMS DLLs, for the gerber2ems imports)
from EMShrimulator.config import GERBER2EMS_EXAMPLES
from EMShrimulator.core.layout import edits, ports, snap, templates, validate
from EMShrimulator.core.layout.model import LayoutPortKind
from EMShrimulator.core.layout.project_writer import write_simulation_json

results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


WORK = Path(tempfile.mkdtemp(prefix="gems-ports-"))


def example(example_id):
    d = WORK / example_id
    if not d.exists():
        project = templates.build(example_id, example_id, d)
        (d / "project.json").write_text(project.model_dump_json(), encoding="utf-8")
    from EMShrimulator.core.layout.model import LayoutProject
    return LayoutProject.model_validate_json((d / "project.json").read_text(encoding="utf-8")), d


# ------------------------------------------------------------------ snapping

def snaps_find_trace_ends_and_vias():
    project, d = example("differential")
    s = snap.build(d, [layer.model_dump() for layer in project.stackup])
    counts = s.counts()
    assert counts.get("via", 0) == 31, counts
    assert counts.get("trace_end", 0) >= 4, counts
    assert s.layer_names[:2] == ["F_Cu", "In1_Cu"], s.layer_names


def snap_prefers_centre_and_respects_layers():
    project, d = example("differential")
    s = snap.build(d, [layer.model_dump() for layer in project.stackup])
    via = s.points[s.kinds == snap.POINT_KINDS.index("via")][0]
    hit = s.nearest_point(via[0] + 0.05, via[1], radius_mm=0.3)
    assert hit is not None and hit.kind in ("via", "pad"), hit
    assert math.hypot(hit.x_mm - via[0], hit.y_mm - via[1]) < 1e-6, hit
    # A layer filter excluding everything leaves only all-layer targets.
    hit = s.nearest_point(via[0], via[1], radius_mm=0.3, layers=[99])
    assert hit is not None and hit.layer == -1, hit


def edge_normal_points_into_copper():
    # SP1 sits at the east end of a westward trace.
    project, d = example("differential")
    s = snap.build(d, [layer.model_dump() for layer in project.stackup])
    end = s.nearest_point(47.9, 13.33, 0.6, kinds=["trace_end"])
    assert end is not None, "no trace end near SP1"
    edge = s.edge(end.edge)
    assert edge.nx < -0.99 and abs(edge.ny) < 1e-6, (edge.nx, edge.ny)
    assert abs(edge.rotation_deg() - 90.0) < 1e-6, edge.rotation_deg()
    assert abs(edge.length_mm - 0.185) < 1e-3, edge.length_mm


check("snap targets: vias, trace ends, layer names", snaps_find_trace_ends_and_vias)
check("snap: nearest centre wins, layer filter keeps all-layer targets", snap_prefers_centre_and_respects_layers)
check("edge snap: normal into the copper, rotation matches the example launch", edge_normal_points_into_copper)


# ---------------------------------------------------------------- references

def auto_reference_finds_inner_plane():
    project, d = example("differential")
    s = snap.build(d, [layer.model_dump() for layer in project.stackup])
    end = s.nearest_point(47.9, 13.33, 0.6, kinds=["trace_end"])
    edge = s.edge(end.edge)
    project.ports = []
    n = edits.place_edge_port(project, edge.x1, edge.y1, edge.x2, edge.y2, edge.rotation_deg(), edge.layer)
    port = edits.find_port(project, n)
    assert port.reference == "auto" and port.kind == LayoutPortKind.LUMPED
    choice = ports.resolve_references(project, d)[n]
    assert choice.found and choice.plane == 1, choice
    resolved, _ = ports.resolved_project(project, d)
    assert resolved.ports[0].plane == 1


def auto_reference_skips_crossing_trace():
    # A narrow strip under the port is not a plane.
    project, d = example("differential")
    masks = ports.CopperMasks(d, project)
    masks.layer_names = {1: masks.layer_names[1], 2: masks.layer_names[1]}
    real = masks.mask

    def fake_mask(index):
        m = real(1)
        if index == 1:
            strip = np.zeros_like(m)
            strip[:, : m.shape[1] // 200] = True
            return strip
        return m

    masks.mask = fake_mask
    project.ports = []
    n = edits.place_port(project, 25.0, 7.0, layer=0)
    choice = ports.nearest_reference(project, edits.find_port(project, n), masks)
    assert choice.plane == 2 and choice.found, choice


def no_plane_is_a_warning():
    project, d = example("differential")
    project.ports = []
    n = edits.place_port(project, 0.05, 0.05, layer=3)  # bottom layer, a board corner
    project.ports[0].excite = True
    masks = ports.CopperMasks(d, project)
    masks.mask = lambda index: np.zeros((10, 10), bool)
    choice = ports.nearest_reference(project, edits.find_port(project, n), masks)
    assert not choice.found and choice.plane == 2, choice


check("auto reference: edge port on F_Cu trace finds the In1_Cu plane", auto_reference_finds_inner_plane)
check("auto reference: a crossing strip is not a plane", auto_reference_skips_crossing_trace)
check("auto reference: no plane falls back to the nearest layer, unfound", no_plane_is_a_warning)


# ------------------------------------------------------------- circuit ports

def circuit_port_edits_and_writer():
    project, d = example("differential")
    project.ports = []
    n = edits.place_circuit_port(project, (10.0, 5.0, 0), (10.3, 5.0, 0), "edge centre · F_Cu", "edge centre · F_Cu")
    edits.move_port(project, n, 11.0, 6.0)
    port = edits.find_port(project, n)
    assert (port.neg_x_mm, port.neg_y_mm) == (11.3, 6.0), (port.neg_x_mm, port.neg_y_mm)
    edits.move_port_terminal(project, n, "neg", 11.0, 6.0, layer=1)
    assert port.plane == 1 and port.reference == "layer"
    write_simulation_json(project, d)
    cfg = json.loads((d / "simulation.json").read_text(encoding="utf-8"))
    entry = cfg["ports"][0]
    assert entry["type"] == "circuit" and entry["neg_position"] == [11.0, 6.0], entry
    assert abs(cfg["convergence"]["end_criteria"] - 1e-5) < 1e-12, cfg["convergence"]
    assert cfg["convergence"]["delta_s"] == 0.02


def circuit_validation():
    project, d = example("differential")
    project.ports = []
    edits.place_circuit_port(project, (10.0, 5.0, 0), (10.0, 5.0, 0))
    codes = {i["code"] for i in validate.validate(project, d)["issues"]}
    assert "circuit-terminals-coincide" in codes, codes
    assert "port-layer-equals-plane" not in codes, codes
    project.ports[0].neg_x_mm, project.ports[0].neg_y_mm = 10.4, 5.3
    codes = {i["code"] for i in validate.validate(project, d)["issues"]}
    assert "circuit-off-axis" in codes and "circuit-terminals-coincide" not in codes, codes


def gerber2ems_reads_circuit_config():
    from serde import from_dict
    from gerber2ems.config import _Config
    project, d = example("differential")
    project.ports = []
    edits.place_circuit_port(project, (10.0, 5.0, 0), (10.3, 5.0, 1))
    write_simulation_json(project, d)
    data = json.loads((d / "simulation.json").read_text(encoding="utf-8"))
    cfg = from_dict(_Config, data)
    assert cfg.ports[0].type == "circuit" and cfg.ports[0].neg_position == [10.3, 5.0], cfg.ports[0]
    assert cfg.convergence.delta_s == 0.02 and abs(cfg.convergence.end_criteria - 1e-5) < 1e-12


def circuit_geometry():
    from gerber2ems.simulation import Simulation
    z = {0: 0.0, 1: -2000.0}
    stub = SimpleNamespace(get_metal_layer_offset=lambda i: z[i])

    def pc(pos, neg, layer, plane, width=100.0):
        return SimpleNamespace(name="P", position=pos, neg_position=neg, layer=layer, plane=plane, width=width)

    # Same layer, along x: the box spans the gap, no leads.
    (start, stop), leads = Simulation.circuit_port_geometry(stub, pc((0, 0), (3000, 0), 0, 0))
    assert start == [0, -50, 0.0] and stop == [3000, 50, 0.0] and not leads, (start, stop, leads)
    # Same layer, diagonal: along the longer axis, then one lead.
    (start, stop), leads = Simulation.circuit_port_geometry(stub, pc((0, 0), (3000, 1000), 0, 0))
    assert stop[0] == 3000 and start[1] == -50 and len(leads) == 1, (start, stop, leads)
    assert leads[0] == (2950, -50, 3050, 1050), leads
    # Across layers at one point: a vertical box, no leads.
    (start, stop), leads = Simulation.circuit_port_geometry(stub, pc((500, 500), (500, 500), 0, 1))
    assert start[2] == 0.0 and stop[2] == -2000.0 and not leads, (start, stop, leads)
    # Across layers and offset: vertical at +, a lead on the - layer.
    (_, _), leads = Simulation.circuit_port_geometry(stub, pc((0, 0), (0, 4000), 0, 1))
    assert leads == [(-50, -50, 50, 4050)], leads
    # Coincident on one layer: refused.
    box, _ = Simulation.circuit_port_geometry(stub, pc((0, 0), (0, 0), 0, 0))
    assert box is None


check("circuit port: edits move terminals, writer emits neg_position and convergence", circuit_port_edits_and_writer)
check("circuit port: validation (coincident, off-axis) replaces layer==plane", circuit_validation)
check("gerber2ems parses circuit ports and convergence", gerber2ems_reads_circuit_config)
check("gerber2ems circuit port geometry: gap, diagonal lead, vertical, vertical lead", circuit_geometry)


# ---------------------------------------------------------------- ΔS monitor

def _write_probe(path, t, v, upto, partial=False):
    """Append samples with t <= upto, as a buffered writer would: optionally stop mid-line."""
    done = getattr(_write_probe, "done", {})
    start, held = done.get(path, (0, ""))
    stop = int(np.searchsorted(t, upto, side="right"))
    text = "% time voltage\n" if start == 0 and not held else ""
    text += "".join(f"{t[k]:.12e}\t{v[k]:.12e}\n" for k in range(start, stop))
    text = held + text
    held = ""
    if partial and len(text) > 20:
        text, held = text[:-7], text[-7:]
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)
    done[path] = (stop, held)
    _write_probe.done = done


def delta_s_monitor_converges_and_aborts():
    from gerber2ems.convergence import DeltaSMonitor, PortProbes, pulse_length_s
    sim = WORK / "dsim"
    shutil.rmtree(sim, ignore_errors=True)
    sim.mkdir()
    f1, f2 = 1e9, 5e9
    pulse = pulse_length_s(f1, f2)
    t = np.arange(0, 40 * pulse, pulse / 200)
    f0, fc = (f1 + f2) / 2, (f2 - f1) / 2
    gauss = np.cos(2 * np.pi * f0 * (t - pulse / 2)) * np.exp(-((t - pulse / 2) / (pulse / 6)) ** 2)
    # Port 0: incident pulse plus a reflection ringing down; port 1: a delayed decaying tone.
    ring = 0.3 * np.sin(2 * np.pi * 3e9 * t) * np.exp(-t / (3 * pulse)) * (t > pulse)
    u0, i0 = gauss + ring, (gauss - ring) / 50.0
    tone = 0.5 * np.sin(2 * np.pi * 2e9 * t) * np.exp(-t / (2 * pulse)) * (t > 2 * pulse)
    u1, i1 = tone, -tone / 50.0
    files = {"port_ut_0": u0, "port_it_0": i0, "port_ut_1": u1, "port_it_1": i1}
    _write_probe.done = {}

    monitor = DeltaSMonitor(str(sim), [PortProbes(["port_ut_0"], ["port_it_0"], 50.0),
                                       PortProbes(["port_ut_1"], ["port_it_1"], 50.0)],
                            excited=0, f_start=f1, f_stop=f2, delta_s=0.01, t_min=4 * pulse, window=pulse)
    stopped_at = None
    for k in range(1, 41):
        for name, v in files.items():
            _write_probe(str(sim / name), t, v, k * pulse, partial=(k % 3 == 0))
        monitor.step()
        if monitor.converged:
            stopped_at = k
            break
    assert stopped_at is not None, f"never converged: {monitor.history[-5:]}"
    assert stopped_at * pulse >= 4 * pulse, stopped_at
    assert stopped_at < 40, stopped_at
    # Early checks were far from converged.
    assert monitor.history[1][1] > 0.01 and monitor.history[-1][1] < 0.01, monitor.history
    # A looser target stops sooner.
    _write_probe.done = {}
    shutil.rmtree(sim)
    sim.mkdir()
    loose = DeltaSMonitor(str(sim), [PortProbes(["port_ut_0"], ["port_it_0"], 50.0),
                                     PortProbes(["port_ut_1"], ["port_it_1"], 50.0)],
                          excited=0, f_start=f1, f_stop=f2, delta_s=0.2, t_min=4 * pulse, window=pulse)
    for k in range(1, 41):
        for name, v in files.items():
            _write_probe(str(sim / name), t, v, k * pulse)
        loose.step()
        if loose.converged:
            break
    assert k < stopped_at, (k, stopped_at)

    # The thread writes ABORT while the solver is still writing; cleanup removes it.
    import time
    import gerber2ems.convergence as conv
    _write_probe.done = {}
    shutil.rmtree(sim)
    sim.mkdir()
    conv.POLL_SECONDS = 0.02
    thread = DeltaSMonitor(str(sim), [PortProbes(["port_ut_0"], ["port_it_0"], 50.0)], excited=0,
                           f_start=f1, f_stop=f2, delta_s=10.0, t_min=0.0, window=pulse)
    thread.start()
    for k in range(1, 41):
        for name in ("port_ut_0", "port_it_0"):
            _write_probe(str(sim / name), t, files[name], k * pulse)
        time.sleep(0.1)
        if (sim / "ABORT").exists():
            break
    thread.join(timeout=5)
    assert (sim / "ABORT").exists(), "no ABORT written"
    thread.cleanup()
    assert not (sim / "ABORT").exists()


check("ΔS monitor: converges on a decaying response, looser stops sooner, writes ABORT",
      delta_s_monitor_converges_and_aborts)


# ---------------------------------------------------------- Gerber macros, S-parameters

def macro_expressions_are_arithmetic_only():
    from gerber2ems.gerber_io import _parse_expr
    assert _parse_expr("$1x2+0.5")([0.0, 3.0]) == 6.5
    assert _parse_expr("-$2/$1")([0.0, 4.0, 2.0]) == -0.5
    count = _parse_expr("4")([0.0])
    assert count == 4 and isinstance(count, int), count  # an outline's vertex count feeds range()
    # The AST parser refuses this before anything runs.
    marker = WORK / "macro-ran"
    payload = ("().__class__.__bases__[0].__subclasses__()[0].__init__.__globals__"
               f"['__builtins__']['open']({str(marker)!r}, 'w')")
    for text in (payload, "__import__('os')", "$1 if $1 else 0", "[1, 2]", "$1 ** 2"):
        call = _parse_expr(text)  # parses lazily: only a used macro fails
        try:
            call([0.0, 1.0])
        except ValueError:
            continue
        raise AssertionError(f"{text!r} was evaluated")
    assert not marker.exists()


def example_macros_still_parse():
    from gerber2ems.gerber_io import ApertureMacro, GerberFile
    gerber = GerberFile(GERBER2EMS_EXAMPLES / "stub_long" / "fab" / "si-simulation-test-board-F_Cu.gbr")
    macros = [a.data for a in gerber.apertures.values() if isinstance(a.data, ApertureMacro)]
    assert macros, "the stub_long board's RoundRect pads are macro apertures"
    for macro in macros:
        assert macro.contours(), macro.name


def touchstone_gets_each_ports_impedance():
    from EMShrimulator.core.em import touchstone
    from EMShrimulator.core.layout.results import build_touchstone
    d = WORK / "z0s"
    sim = d / "ems" / "simulation"
    sim.mkdir(parents=True)
    (d / "simulation.json").write_text(json.dumps({"ports": [{"impedance": 50}, {"impedance": 75}]}))
    # gerber2ems's Sx0.csv: port 0 excited, columns S<to>-<from>.
    rows = "\n".join(f"{f}, 0.1, 0.8, 0.0, -0.2" for f in (1000, 2000, 3000))
    (sim / "Sx0.csv").write_text("Frequency [MHz], re(S0-0), re(S1-0), im(S0-0), im(S1-0), \n" + rows + "\n")
    net = touchstone.read(build_touchstone(d, "z0s"))
    assert list(net.z0_ports) == [50.0, 75.0], net.z0_ports
    assert np.allclose(net.s[:, 1, 0], 0.8 - 0.2j) and np.allclose(net.s[:, 0, 1], 0.8 - 0.2j)  # reciprocity


def gerber2ems_s_is_power_waves():
    from gerber2ems.config import Config, _Config
    from gerber2ems.postprocess import Postprocesor
    Config.set_config(_Config())  # no config until simulation.json is loaded
    freqs = np.array([1e9, 2e9])
    post = Postprocesor(freqs, 2)
    post.reference_zs = np.array([50.0, 200.0])
    one = np.ones(2, complex)
    post.add_port_data(0, 0, 2 * one, 0.2 * one)  # port 0 excited: incident 2 V, reflected 0.2 V
    post.add_port_data(1, 0, 0 * one, 1.0 * one)  # 1 V leaves port 1
    post.calculate_sparams()
    assert np.allclose(post.s_params[0][0], 0.1)
    # b1 / a0 = 0.25, not the voltage ratio 0.5.
    assert np.allclose(post.s_params[1][0], 0.25), post.s_params[1][0]


def substrate_loss_reaches_the_solver():
    from CSXCAD import ContinuousStructure
    from gerber2ems.config import Config, LayerConfig, _Config
    from gerber2ems.simulation import Simulation
    cfg = _Config()
    cfg.frequency.start, cfg.frequency.stop = 1e9, 10e9
    cfg.layers = [LayerConfig({"type": "core", "name": "core", "thickness": 0.2, "epsilon": 4.3, "lossTangent": 0.02}),
                  LayerConfig({"type": "prepreg", "name": "pp", "thickness": 0.1, "epsilon": 3.9, "lossTangent": None})]
    from gerber2ems.config import PortConfig
    port = PortConfig()
    port.layer, port.plane = 0, 1
    cfg.ports = [port]  # poles go along the port's line: both substrates here
    Config.set_config(cfg)
    stub = SimpleNamespace(csx=ContinuousStructure(), gerber_materials=[], substrate_materials=[])
    Simulation.create_materials(stub)
    lossy, plain = stub.substrate_materials
    assert type(lossy).__name__ == "CSPropDebyeMaterial" and lossy.GetDispersionOrder() >= 2, type(lossy).__name__
    assert type(plain).__name__ == "CSPropMaterial", type(plain).__name__
    # Upstream read only epsilon: the loss tangent never reached openEMS.
    delta = [lossy.GetDispersiveMaterialProperty("eps_delta", k) for k in range(lossy.GetDispersionOrder())]
    assert all(d > 0 for d in delta) and lossy.GetMaterialProperty("epsilon") < 4.3
    # Faces bordering air or a pole-less dielectric halve the poles; edges quarter them.
    cfg.pcb_width, cfg.pcb_height = 20000, 10000
    Simulation.add_substrates(stub)
    expression = lossy.GetDispersiveMaterialWeight("eps_delta", 0)
    weight = lambda x, y, z: float(eval(expression, {"abs": abs}, {"x": x, "y": y, "z": z}))  # noqa: E731
    core = cfg.layers[0].thickness
    assert weight(10000, 5000, -core / 2) == 1.0 and weight(10000, 5000, 0) == 0.5
    assert weight(0, 5000, -core / 2) == 0.5 and weight(0, 5000, 0) == 0.25

    # "centre": a plain conductivity exact at the band centre.
    from EMShrimulator.core.em.dielectric import centre_conductivity
    cfg.dielectric_loss = "centre"
    stub = SimpleNamespace(csx=ContinuousStructure(), gerber_materials=[], substrate_materials=[])
    Simulation.create_materials(stub)
    centre = stub.substrate_materials[0]
    assert type(centre).__name__ == "CSPropMaterial", type(centre).__name__
    assert abs(centre.GetMaterialProperty("kappa") / centre_conductivity(4.3, 0.02, 5.5e9) - 1) < 1e-6


def poles_only_where_the_lines_run():
    """Only the substrates beside a port's line get Debye poles; the rest use the band-centre model."""
    from CSXCAD import ContinuousStructure
    from gerber2ems.config import Config, LayerConfig, PortConfig, _Config
    from gerber2ems.simulation import Simulation

    def stack():
        c = _Config()
        c.frequency.start, c.frequency.stop = 2e8, 6e9

        def metal(name):
            return LayerConfig({"type": "copper", "name": name, "thickness": 0.035})

        def dielectric(name, t):
            return LayerConfig({"type": "core", "name": name, "thickness": t, "epsilon": 4.18, "lossTangent": 0.02})

        c.layers = [metal("F.Cu"), dielectric("d1", 0.12), metal("In1.Cu"), dielectric("d2", 1.2),
                    metal("In2.Cu"), dielectric("d3", 0.12), metal("B.Cu")]
        return c

    def ports(*pairs):
        out = []
        for layer, plane in pairs:
            p = PortConfig()
            p.layer, p.plane = layer, plane
            out.append(p)
        return out

    for case, port_list, expected in (
            ("microstrip on F.Cu over In1", ports((0, 1)), {0}),
            ("stripline on In1, either side", ports((1, 2)), {0, 1}),
            ("gap port from F.Cu down to In2", ports((0, 2)), {0, 1}),
            ("both outer layers", ports((0, 1), (3, 2)), {0, 2})):
        cfg = stack()
        cfg.ports = port_list
        Config.set_config(cfg)
        assert Simulation.field_substrates() == expected, (case, Simulation.field_substrates())
        stub = SimpleNamespace(csx=ContinuousStructure(), gerber_materials=[], substrate_materials=[])
        Simulation.create_materials(stub)
        kinds = [type(m).__name__ for m in stub.substrate_materials]
        assert [i for i, k in enumerate(kinds) if k == "CSPropDebyeMaterial"] == sorted(expected), (case, kinds)
        assert all(m.GetMaterialProperty("kappa") > 0 for m in stub.substrate_materials), "every substrate stays lossy"
    # Two poles for 0.2-6 GHz, plus the conductivity.
    debye = next(m for m in stub.substrate_materials if type(m).__name__ == "CSPropDebyeMaterial")
    assert debye.GetDispersionOrder() == 2, debye.GetDispersionOrder()


def writer_passes_the_loss_model():
    from serde import from_dict
    from gerber2ems.config import _Config
    project, d = example("differential")
    for model in ("wideband", "centre"):
        project.dielectric_loss = model
        write_simulation_json(project, d)
        data = json.loads((d / "simulation.json").read_text(encoding="utf-8"))
        assert data["dielectric_loss"] == model and from_dict(_Config, data).dielectric_loss == model


check("simulation.json carries the dielectric-loss model, and gerber2ems reads it", writer_passes_the_loss_model)


check("layout substrates keep their loss tangent (Debye poles), lossless ones stay plain", substrate_loss_reaches_the_solver)
check("Debye poles only in the substrates the ports' lines run in; the rest stay lossy at no cost",
      poles_only_where_the_lines_run)
check("aperture macros: arithmetic evaluates, code in a macro is refused", macro_expressions_are_arithmetic_only)
check("the example boards' RoundRect macros still parse", example_macros_still_parse)
check("layout Touchstone: each port's own reference impedance, reciprocity fill", touchstone_gets_each_ports_impedance)
check("gerber2ems S-parameters are power waves when port impedances differ", gerber2ems_s_is_power_waves)

shutil.rmtree(WORK, ignore_errors=True)
print()
for status, label, detail in results:
    print(f"  {status}  {label}")
    if detail:
        print(f"        {detail}")
fails = sum(1 for s, _, _ in results if s == "FAIL")
print(f"\n{len(results) - fails}/{len(results)} passed")
raise SystemExit(1 if fails else 0)
