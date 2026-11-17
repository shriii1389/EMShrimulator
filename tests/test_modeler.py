"""3D Modeler engine: revolves match CSXCAD, facet counts, parameter renames, guided axes, power-wave S."""
import os
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np

import EMShrimulator.bootstrap  # noqa: F401  (openEMS DLLs)
from CSXCAD import ContinuousStructure
from EMShrimulator.core.em import csg, edits, region, templates, validate
from EMShrimulator.core.em.csx_builder import global_aabb
from EMShrimulator.core.em.model import (
    BoundaryConditions, FrequencySweep, Material, MaterialKind, MeshSettings, ModelerProject, Parameter, Port,
    PortKind, Primitive, PrimitiveKind, SimulationSettings,
)
from EMShrimulator.core.errors import EditError

results = []


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


# --- 1. a revolve is the solid CSXCAD builds ----------------------------------

def _csxcad_revolve(profile, normal, rot, elevation, a0, a1):
    prop = ContinuousStructure().AddMetal("m")
    prim = prop.AddRotPoly([[p[0] for p in profile], [p[1] for p in profile]], norm_dir=normal, elevation=0.0,
                           rot_axis=rot, angle=[np.radians(a0), np.radians(a1)])
    if elevation:  # as csx_builder applies it
        shift = [0.0, 0.0, 0.0]
        shift["xyz".index(normal)] = elevation
        prim.AddTransform("Translate", shift)
    assert prim.Update()[0]
    return prim


def revolve_matches_csxcad():
    m = csg._require()
    points = np.random.default_rng(7).uniform(-3.5, 3.5, size=(600, 3))
    eps = 1e-3
    cases = [
        ([(1.0, 0.5), (2.5, 0.5), (2.5, 2.0), (1.0, 2.0)], "z", "y", 0.0, 0.0, 360.0),
        ([(1.0, 0.5), (2.5, 0.5), (2.5, 2.0), (1.0, 2.0)], "z", "x", 1.5, 30.0, 200.0),
        ([(1.0, 0.5), (2.5, 0.5), (2.5, 2.0), (1.0, 2.0)], "x", "y", 0.0, -45.0, 60.0),
        ([(-1.0, -0.5), (2.0, -0.5), (2.0, 1.5), (-1.0, 1.0)], "y", "z", 1.5, 30.0, 200.0),
    ]
    for profile, normal, rot, elevation, a0, a1 in cases:
        truth = _csxcad_revolve(profile, normal, rot, elevation, a0, a1)
        solid = csg._rot_poly(profile, normal, elevation, rot, a0, a1)
        inside_csx = np.array([truth.IsInside(list(p)) for p in points])
        inside_mf = np.array([(solid ^ m.Manifold.cube([eps] * 3).translate(list(p - eps / 2))).volume() > 0
                              for p in points])
        assert inside_csx.sum() > 5, "the sample misses the solid"
        agree = (inside_csx == inside_mf).mean()
        assert agree > 0.99, f"n={normal} r={rot} e={elevation} {a0}..{a1}: {agree:.3f} agree"


def revolve_axis_rules():
    prj = ModelerProject(name="rev", materials=[Material(name="PEC", kind=MaterialKind.PEC)])
    prim = Primitive(name="R", kind=PrimitiveKind.ROT_POLY, material="PEC")
    prj.primitives.append(prim)
    prim.points = [(0.5, 0.0), (2.0, 0.0), (2.0, 1.0), (0.5, 1.0)]
    prim.normal_axis, prim.rot_axis, prim.elevation = "z", "z", 0.0
    codes = {i["code"] for i in validate.validate(prj)["issues"]}
    assert "revolve-axis-along-normal" in codes, codes
    prim.rot_axis = "y"
    codes = {i["code"] for i in validate.validate(prj)["issues"]}
    assert "revolve-axis-along-normal" not in codes, codes
    # The tight box: radius 2 about y, height 0..1 along y, shifted by elevation along z.
    prim.elevation = 3.0
    lo, hi = global_aabb(prim, {})
    assert np.allclose(lo, [-2, 0, 1]) and np.allclose(hi, [2, 1, 5]), (lo, hi)
    # No sweep axis set: the outline's v axis, never the normal.
    for normal in "xyz":
        assert csg.default_rot_axis(normal) != normal


def new_revolve_builds_something():
    prj = templates.build("blank", "rev")
    prim = edits.find_primitive(prj, edits.add_primitive(prj, PrimitiveKind.ROT_POLY))
    assert prim.rot_axis != (prim.normal_axis or "z"), (prim.normal_axis, prim.rot_axis)
    assert not any(i["code"] == "revolve-axis-along-normal" for i in validate.validate(prj)["issues"])
    solid = _csxcad_revolve(prim.points, prim.normal_axis, prim.rot_axis, prim.elevation or 0.0,
                            prim.angle_start_deg or 0.0, prim.angle_stop_deg or 360.0)
    lo, hi = global_aabb(prim, {})
    centre = (np.asarray(lo) + np.asarray(hi)) / 2
    probe = [float(v) for v in centre]
    radius_axis = "xyz".index(prim.rot_axis)
    off_axis = [c for c in range(3) if c != radius_axis][0]
    probe[off_axis] = hi[off_axis] * 0.75  # inside the swept outline, off the axis
    assert solid.IsInside(probe), f"CSXCAD builds nothing at {probe} (box {lo}..{hi})"


check("a revolve is the solid CSXCAD builds: beside and across the axis, partial, elevated", revolve_matches_csxcad)
check("revolve: sweep axis along the normal is an error; tight bounding box; valid default axis", revolve_axis_rules)
check("a new revolve from the toolbar builds a non-empty CSXCAD solid", new_revolve_builds_something)


# --- 2. facets follow the size of the curve ------------------------------------
def facets_follow_radius():
    counts = [csg._circular_segments(r) for r in (0.1, 1.0, 2.0, 5.0, 10.0, 50.0, 500.0)]
    assert counts[:3] == [32, 32, 32], counts       # small curves: 0.5 % of the radius already < 10 µm
    assert counts == sorted(counts) and counts[3] < counts[4] < counts[5], counts
    assert counts[-1] == 96, counts                  # capped
    r = 10.0
    n = csg._circular_segments(r)
    assert r * (1 - np.cos(np.pi / n)) <= csg.CHORD_ERROR_MM + 1e-12, n


check("curved surfaces get more facets as they grow, within 10 µm, 32..96", facets_follow_radius)


# --- 3. renaming a variable leaves function calls alone --------------------------
def rename_keeps_calls():
    prj = ModelerProject(name="p", parameters=[Parameter(name="db", expression="3"),
                                               Parameter(name="w", expression="db(2) + db * 2")])
    edits.rename_parameter(prj, "db", "gain")
    assert prj.parameters[1].expression == "db(2) + gain * 2", prj.parameters[1].expression
    try:
        edits.rename_parameter(prj, "w", "sqrt")
    except EditError:
        pass
    else:
        raise AssertionError("a variable may not take a function's name")


check("renaming a variable rewrites reads, not calls; function names are refused", rename_keeps_calls)


# --- 4. a sheet-assigned guided port launches along the sheet normal -------------
def guided_axis_from_sheet():
    prj = templates.build("coax_fed_patch", "cfp")
    sheet_ports = [p for p in prj.ports if p.sheet]
    assert sheet_ports, "the coax-fed patch template feeds through a sheet port"
    for port in sheet_ports:
        sheet = next(p for p in prj.primitives if p.name == port.sheet)
        assert region.guided_axis(prj, port) == (sheet.normal_axis or "z"), (port.name, port.prop_dir)


check("validation: a sheet-assigned guided port launches along the sheet normal", guided_axis_from_sheet)


# --- 5. a loss tangent holds across the sweep --------------------------------------
def loss_tangent_across_band():
    from EMShrimulator.core.em import dielectric
    from EMShrimulator.core.em.csx_builder import build_simulation

    for eps_r, tan_delta, f1, f2 in ((4.3, 0.02, 1e6, 6e9), (3.66, 0.0037, 1e8, 10e9), (2.1, 2e-4, 2e9, 2.5e9)):
        f0 = (f1 + f2) / 2
        fit = dielectric.constant_loss(eps_r, tan_delta, f1, f2, f0)
        f = np.logspace(np.log10(max(f1, f2 / 100)), np.log10(f2), 200)
        eps = fit.epsilon(f)
        tan = -eps.imag / eps.real
        assert np.all(np.abs(tan / tan_delta - 1) < 0.06), (eps_r, tan_delta, tan.min(), tan.max())
        assert abs(fit.epsilon(f0).real - eps_r) < 1e-9 * eps_r
        assert len(fit.delta) <= 8, len(fit.delta)  # each pole costs solver time in every material cell
        # The old model's tan δ ∝ 1/f: a factor of 100 over two decades.
    prj = templates.build("microstrip_line", "loss")
    built = build_simulation(prj, 1)
    props = {p.GetName(): p for p in built.CSX.GetAllProperties()}
    kinds = {name: type(p).__name__ for name, p in props.items()}
    assert kinds["RO4350B"] == "CSPropDebyeMaterial" and kinds["PEC__strip"] == "CSPropMetal", kinds

    # The builder halves Debye poles on dielectric/air faces.
    sub = next(p for p in prj.primitives if p.name == "substrate")
    (x0, y0, z0), (x1, y1, z1) = sub.p1, sub.p2
    expression = props["RO4350B"].GetDispersiveMaterialWeight("eps_delta", 0)
    weight = lambda x, y, z: float(eval(expression, {"abs": abs}, {"x": x, "y": y, "z": z}))  # noqa: E731
    mid = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    assert weight(*mid) == 1.0
    assert weight(mid[0], mid[1], z1) == 0.5            # the top face, under the strip's fringing field
    assert weight(x0, mid[1], z1) == 0.25               # where the top face meets an end face
    # Every pole stays above openEMS's three-timestep cutoff.
    from EMShrimulator.core.em.csx_builder import _courant_timestep
    dt = _courant_timestep(built.CSX.GetGrid())
    taus = [props["RO4350B"].GetDispersiveMaterialProperty("eps_relax", k)
            for k in range(props["RO4350B"].GetDispersionOrder())]
    assert min(taus) > 3 * dt, (min(taus), dt)
    capped = dielectric.constant_loss(3.66, 0.02, 1e9, 10e9, 5.5e9, max_pole_hz=12e9)
    assert max(1 / (2 * np.pi * t) for t in capped.tau) <= 12e9 * (1 + 1e-9)

    # "centre": a conductivity exact at the excitation centre.
    prj.simulation.dielectric_loss = "centre"
    sub_mat = next(m for m in prj.materials if m.name == "RO4350B")
    f0 = (prj.frequency.start + prj.frequency.stop) / 2
    centre_built = build_simulation(prj, 1)  # keep it alive: it owns the properties read below
    centre = {p.GetName(): p for p in centre_built.CSX.GetAllProperties()}["RO4350B"]
    assert type(centre).__name__ == "CSPropMaterial", type(centre).__name__
    kappa = centre.GetMaterialProperty("kappa")
    assert abs(kappa / dielectric.centre_conductivity(sub_mat.epsilon, sub_mat.tan_delta, f0) - 1) < 1e-6, kappa


check("a loss tangent holds within 6 % across the sweep, exact εr at f0; lossy materials are Debye", loss_tangent_across_band)


# --- 5b. a non-model object is drawn but never built or meshed -------------------
def non_model_is_not_built():
    from EMShrimulator.core.em.csx_builder import build_simulation

    prj = ModelerProject(
        name="nm", materials=[Material(name="PEC", kind=MaterialKind.PEC), Material(name="FR4", epsilon=4.3)],
        primitives=[Primitive(name="strip", kind=PrimitiveKind.BOX, material="PEC", p1=(0, 0, 0), p2=(5, 1, 0)),
                    Primitive(name="far", kind=PrimitiveKind.BOX, material="FR4", p1=(80, 80, 80), p2=(90, 90, 90),
                              model=False)],
        mesh=MeshSettings(air_margin=(2.0, 2.0, 2.0)),
    )
    built = build_simulation(prj, None)
    names = {p.GetName() for p in built.CSX.GetAllProperties()}
    assert "FR4" not in names and "PEC__strip" in names, names
    assert np.all(np.asarray(built.domain_max) < 20), built.domain_max  # the far box sets no mesh or domain
    prj.primitives[1].model = True
    assert np.all(np.asarray(build_simulation(prj, None).domain_max) >= 90)


check("a non-model object never reaches CSXCAD, the mesh or the domain", non_model_is_not_built)


# --- 6. S-parameters are power waves: reciprocal across different port impedances

def reciprocal_with_unequal_ports():
    from EMShrimulator.core.em import touchstone
    from EMShrimulator.core.em.run_worker import solve_project

    h, length = 1.0, 12.0
    prj = ModelerProject(
        name="recip",
        materials=[Material(name="PEC", kind=MaterialKind.PEC)],
        primitives=[
            Primitive(name="gnd", kind=PrimitiveKind.BOX, material="PEC", p1=(0, -3, 0), p2=(length, 3, 0)),
            Primitive(name="strip", kind=PrimitiveKind.BOX, material="PEC", p1=(0, -0.6, h), p2=(length, 0.6, h)),
        ],
        ports=[
            Port(name="P1", kind=PortKind.LUMPED, number=1, p1=(0, -0.6, 0), p2=(0, 0.6, h), direction="z",
                 impedance=50.0, excite=True),
            Port(name="P2", kind=PortKind.LUMPED, number=2, p1=(length, -0.6, 0), p2=(length, 0.6, h),
                 direction="z", impedance=100.0, excite=True),
        ],
        boundary=BoundaryConditions(**{f: "MUR" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}),
        mesh=MeshSettings(cells_per_wavelength=15, air_margin=(3.0, 3.0, 3.0), max_cell_size=0.5,
                          metal_edge_refinement=False),
        frequency=FrequencySweep(start=1e9, stop=4e9, points=31),
        simulation=SimulationSettings(max_timesteps=20000, end_criteria=1e-4),
    )
    out = Path(tempfile.mkdtemp(prefix="gems-recip-"))
    log = out.parent / f"{out.name}.log"
    fd = os.open(log, os.O_RDWR | os.O_CREAT)
    saved, cwd = (os.dup(1), os.dup(2)), os.getcwd()
    try:  # keep openEMS's fd 1 output out of the test log
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        summary = solve_project(prj, out, run_validation=False)
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1]); os.close(fd)
        os.chdir(cwd)  # openEMS's Run() leaves the process in the last sim folder
    net = touchstone.read(out / summary["touchstone"])
    shutil.rmtree(out, ignore_errors=True)
    log.unlink(missing_ok=True)
    assert list(net.z0_ports) == [50.0, 100.0], net.z0_ports
    s21, s12 = net.s[:, 1, 0], net.s[:, 0, 1]
    ratio = np.abs(s21) / np.abs(s12)
    assert np.all(np.abs(ratio - 1) < 0.05), f"|S21|/|S12| = {ratio.min():.3f}..{ratio.max():.3f}, not 1"
    # Lossless: no more power out than in.
    assert np.all(np.abs(net.s[:, 0, 0]) ** 2 + np.abs(s21) ** 2 < 1.05), "|S11|² + |S21|² exceeds 1"


check("S-parameters are power waves: S21 = S12 with 50 and 100 ohm ports (real solve)",
      reciprocal_with_unequal_ports)


def _solve_quietly(prj, prefix):
    """solve_project with openEMS's console output kept out of the test report."""
    from EMShrimulator.core.em.run_worker import solve_project

    out = Path(tempfile.mkdtemp(prefix=prefix))
    log = out.parent / f"{out.name}.log"
    fd = os.open(log, os.O_RDWR | os.O_CREAT)
    saved, cwd = (os.dup(1), os.dup(2)), os.getcwd()
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        summary = solve_project(prj, out, run_validation=False)
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0]); os.close(saved[1]); os.close(fd)
        os.chdir(cwd)
    return out, log, summary


# --- 7. adaptive passes refine the mesh and report ΔS; the Z-matrix is reciprocal

def adaptive_passes_and_z_matrix():
    from EMShrimulator.core.em import results as em_results
    from EMShrimulator.core.em import touchstone

    h, length = 1.0, 12.0
    prj = ModelerProject(
        name="passes",
        materials=[Material(name="PEC", kind=MaterialKind.PEC)],
        primitives=[
            Primitive(name="gnd", kind=PrimitiveKind.BOX, material="PEC", p1=(0, -3, 0), p2=(length, 3, 0)),
            Primitive(name="strip", kind=PrimitiveKind.BOX, material="PEC", p1=(0, -0.6, h), p2=(length, 0.6, h)),
        ],
        ports=[
            Port(name="P1", kind=PortKind.LUMPED, number=1, p1=(0, -0.6, 0), p2=(0, 0.6, h), direction="z",
                 impedance=50.0, excite=True),
            Port(name="P2", kind=PortKind.LUMPED, number=2, p1=(length, -0.6, 0), p2=(length, 0.6, h),
                 direction="z", impedance=50.0, excite=True),
        ],
        boundary=BoundaryConditions(**{f: "MUR" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}),
        mesh=MeshSettings(cells_per_wavelength=15, air_margin=(3.0, 3.0, 3.0), max_cell_size=0.6,
                          metal_edge_refinement=False),
        frequency=FrequencySweep(start=1e9, stop=4e9, points=31),
        simulation=SimulationSettings(max_timesteps=20000, end_criteria=1e-4, max_passes=2, delta_s=1e-9,
                                      refine_ratio=1.3),
    )
    out, log, summary = _solve_quietly(prj, "gems-passes-")
    net = touchstone.read(out / summary["touchstone"])
    shutil.rmtree(out, ignore_errors=True)
    log.unlink(missing_ok=True)
    passes = summary["passes"]
    assert len(passes) == 2 and not summary["converged"], summary
    assert passes[1]["cells"] > passes[0]["cells"], passes
    assert 0 < passes[1]["delta_s"] < 0.5, passes
    z = em_results.impedance_traces(net.s, net.z0_ports, {})
    z12 = np.asarray(z["Z12"]["z_real"]) + 1j * np.asarray(z["Z12"]["z_imag"])
    z21 = np.asarray(z["Z21"]["z_real"]) + 1j * np.asarray(z["Z21"]["z_imag"])
    finite = np.isfinite(z12) & np.isfinite(z21)
    assert finite.sum() > 20, "the Z-matrix is missing at most frequencies"
    spread = np.abs(z12[finite] - z21[finite]) / np.maximum(np.abs(z12[finite]), 1.0)
    assert np.median(spread) < 0.05, f"Z12 and Z21 differ by {np.median(spread):.3f} (median)"


check("Adaptive passes refine the mesh and report ΔS; Z12 = Z21 (real solve, two passes)",
      adaptive_passes_and_z_matrix)


# --- 8. gain: a lossless dipole radiates what it accepts, so gain = directivity

def lossless_dipole_gain():
    from EMShrimulator.core.em.model import NF2FFSettings

    arm, w, gap = 23.0, 1.0, 1.0
    prj = ModelerProject(
        name="dipole",
        materials=[Material(name="PEC", kind=MaterialKind.PEC)],
        primitives=[
            Primitive(name="top", kind=PrimitiveKind.BOX, material="PEC", p1=(-w / 2, -w / 2, gap / 2),
                      p2=(w / 2, w / 2, gap / 2 + arm)),
            Primitive(name="bottom", kind=PrimitiveKind.BOX, material="PEC", p1=(-w / 2, -w / 2, -gap / 2 - arm),
                      p2=(w / 2, w / 2, -gap / 2)),
        ],
        ports=[Port(name="feed", kind=PortKind.LUMPED, number=1, p1=(-w / 2, -w / 2, -gap / 2),
                    p2=(w / 2, w / 2, gap / 2), direction="z", impedance=73.0, excite=True)],
        boundary=BoundaryConditions(),
        mesh=MeshSettings(cells_per_wavelength=15, air_margin=(25.0, 25.0, 25.0), metal_edge_refinement=False),
        frequency=FrequencySweep(start=2e9, stop=4e9, points=41),
        simulation=SimulationSettings(max_timesteps=30000, end_criteria=1e-4),
        nf2ff=NF2FFSettings(enabled=True, frequencies=[3e9], theta_step_deg=10.0, phi_step_deg=15.0),
    )
    out, log, summary = _solve_quietly(prj, "gems-dipole-")
    import json

    payload = json.loads((out / "farfield.json").read_text(encoding="utf-8"))
    shutil.rmtree(out, ignore_errors=True)
    log.unlink(missing_ok=True)
    entry = payload["frequencies"][0]
    assert "gain_dbi" in entry and "realized_gain_dbi" in entry, sorted(entry)
    assert 0.8 < entry["efficiency"] < 1.2, f"radiation efficiency {entry['efficiency']:.3f} for PEC in air"
    assert abs(entry["gain_dbi"] - entry["dmax_dbi"]) < 1.0, (entry["gain_dbi"], entry["dmax_dbi"])
    assert entry["realized_gain_dbi"] <= entry["gain_dbi"] + 0.05, "realized gain above gain"
    # A slightly short dipole sits between 1.76 dBi and 2.15 dBi.
    assert 1.5 < entry["dmax_dbi"] < 2.6, f"directivity {entry['dmax_dbi']:.2f} dBi for a near half-wave dipole"
    assert abs(entry["consistency"] - 1) < 0.1, f"far field / box power {entry['consistency']:.3f}"


check("Gain: a lossless dipole's gain is its directivity, efficiency near 100 % (real solve)",
      lossless_dipole_gain)


# --- 9. absorbers sit outside the air, and the NF2FF box encloses the design -------------------------------

def pml_outside_the_air():
    from EMShrimulator.core.em.csx_builder import build_simulation
    from EMShrimulator.core.em.model import NF2FFSettings

    prj = ModelerProject(
        name="box", materials=[Material(name="PEC", kind=MaterialKind.PEC)],
        primitives=[Primitive(name="arm", kind=PrimitiveKind.BOX, material="PEC", p1=(-0.5, -0.5, -24),
                              p2=(0.5, 0.5, 24))],
        ports=[Port(name="feed", kind=PortKind.LUMPED, number=1, p1=(-0.5, -0.5, -0.5), p2=(0.5, 0.5, 0.5),
                    direction="z", impedance=73.0, excite=True)],
        boundary=BoundaryConditions(zmin="PEC"),
        mesh=MeshSettings(cells_per_wavelength=15, air_margin=(25.0, 25.0, 25.0), metal_edge_refinement=False),
        frequency=FrequencySweep(start=2e9, stop=4e9, points=11),
        nf2ff=NF2FFSettings(enabled=True, frequencies=[3e9]),
    )
    built = build_simulation(prj, excite_port_number=1)
    grid = built.CSX.GetGrid()
    for n, axis in enumerate("xyz"):
        lines = np.array(grid.GetLines(axis))
        assert np.all(np.diff(lines) > 0), f"{axis} lines are not sorted"
        low = -0.5 if axis != "z" else -24.0
        high = 0.5 if axis != "z" else 24.0
        # 25 mm of air, then 8 PML cells beyond it, except below where the face is PEC.
        assert lines[-1] > high + 25.0 + 1e-6 and np.count_nonzero(lines > high + 25.0 + 1e-6) == 8, axis
        assert np.count_nonzero(lines < low - 25.0 - 1e-6) == (0 if axis == "z" else 8), axis
        start, stop = built.nf2ff.start[n], built.nf2ff.stop[n]
        assert start < low and stop > high, f"the NF2FF box ({start:.2f}, {stop:.2f}) cuts the design on {axis}"


check("PML cells go outside the air margin; the NF2FF box encloses the design", pml_outside_the_air)


# --- 10. mode ports are referenced to their own mode impedance ------------------------------------------------
def waveguide_passes_its_band():
    from EMShrimulator.core.em import templates, touchstone

    out, log, summary = _solve_quietly(templates.build("waveguide_wr90", "wr90"), "gems-wr90-")
    net = touchstone.read(out / summary["touchstone"])
    shutil.rmtree(out, ignore_errors=True)
    log.unlink(missing_ok=True)
    band = (net.freq_hz >= 8.2e9) & (net.freq_hz <= 12.4e9)
    s11 = 20 * np.log10(np.abs(net.s[band, 0, 0]))
    s21 = 20 * np.log10(np.abs(net.s[band, 1, 0]))
    assert s21.min() > -0.5 and s21.max() < 0.1, f"|S21| {s21.min():.2f}..{s21.max():.2f} dB, not 0 dB"
    assert s11.max() < -15, f"|S11| up to {s11.max():.1f} dB on a straight guide"


check("A straight WR-90 guide passes its band: TE10 ports use the mode's impedance (real solve)",
      waveguide_passes_its_band)


def coax_lines_pass():
    import math

    from EMShrimulator.core.em import templates, touchstone

    def solve(er):
        prj = templates.build("coaxial_line", "coax")
        a = 1.5
        b = a * math.exp(50.0 * math.sqrt(er) / 60.0)
        for m in prj.materials:
            if m.name == "Air":
                m.epsilon = er
        for p in prj.primitives:
            if p.name == "dielectric" or p.kind == PrimitiveKind.SHEET:
                p.radius = b
            elif p.name == "shield":
                p.radius = b + 0.25
        out, log, summary = _solve_quietly(prj, "gems-coax-")
        net = touchstone.read(out / summary["touchstone"])
        shutil.rmtree(out, ignore_errors=True)
        log.unlink(missing_ok=True)
        band = net.freq_hz >= 1e9
        return 20 * np.log10(np.abs(net.s[band, 0, 0])), 20 * np.log10(np.abs(net.s[band, 1, 0]))

    for er in (1.0, 2.1):  # PTFE: the port takes its wave impedance from the filling, Z0/sqrt(er)
        s11, s21 = solve(er)
        assert np.abs(s21).max() < 0.1, f"er {er}: |S21| {s21.min():.2f}..{s21.max():.2f} dB, not 0 dB"
        assert s11.max() < -25, f"er {er}: |S11| up to {s11.max():.1f} dB on a matched line"


check("50 ohm coax lines, air and PTFE, pass end to end: coax ports (real solves)", coax_lines_pass)


def cylinder_box_follows_its_axis():
    from EMShrimulator.core.em.csx_builder import local_corners

    along_z = Primitive(name="c", kind=PrimitiveKind.CYLINDER, material="PEC", p1=(0, 0, 0), p2=(0, 0, 10), radius=2)
    corners = local_corners(along_z)
    assert np.allclose(corners.min(axis=0), [-2, -2, 0]) and np.allclose(corners.max(axis=0), [2, 2, 10]), corners
    slanted = Primitive(name="s", kind=PrimitiveKind.CYLINDER, material="PEC", p1=(0, 0, 0), p2=(3, 0, 4), radius=1)
    lo, hi = local_corners(slanted).min(axis=0), local_corners(slanted).max(axis=0)
    assert np.allclose([lo[0], hi[0], lo[2], hi[2]], [-0.8, 3.8, -0.6, 4.6]) and np.allclose([lo[1], hi[1]], [-1, 1])


check("A cylinder's box reaches only its end discs along its axis, so a line can end on the boundary",
      cylinder_box_follows_its_axis)

print()
for status, label, detail in results:
    print(f"  {status}  {label}")
    if detail:
        print(f"        {detail}")
fails = sum(1 for s, _, _ in results if s == "FAIL")
print(f"\n{len(results) - fails}/{len(results)} passed")
raise SystemExit(1 if fails else 0)
