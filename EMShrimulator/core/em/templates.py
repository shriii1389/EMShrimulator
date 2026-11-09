"""Ready-to-run example designs for the 3D Modeler. Ports are sheets with an excitation assigned."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from EMShrimulator.core.em.model import (
    BooleanOp,
    BoundaryConditions,
    DumpKind,
    Excitation,
    FieldDump,
    FrequencySweep,
    Material,
    MaterialKind,
    MeshSettings,
    ModelerProject,
    NF2FFSettings,
    Parameter,
    ParameterSweep,
    Port,
    PortKind,
    Primitive,
    PrimitiveKind,
    SheetShape,
    SimulationSettings,
    SweepMode,
    SweepVariable,
)
from EMShrimulator.core.em.params import microstrip_width

C0 = 299_792_458.0
METAL = "#9aa4b2"   # PEC as drawn
GOLD = "#e0b64a"  # the main conductor
PIN_W = 1.27  # mm, an SMA pin's width


def _mm(m: float) -> float:
    return m * 1000.0


def _pec() -> Material:
    return Material(name="PEC", kind=MaterialKind.PEC, color=METAL)


def _sheet(name: str, normal: str, origin, size=None, shape: SheetShape = SheetShape.RECTANGLE,
           radius=None, radius_inner=None, segments: int = 48) -> Primitive:
    """A port's sheet; `size` runs along the in-plane axes."""
    return Primitive(name=name, kind=PrimitiveKind.SHEET, material="", priority=100, sheet_shape=shape,
                     normal_axis=normal, origin=origin, sheet_size=size, radius=radius, radius_inner=radius_inner,
                     sheet_segments=segments)


def _line_port(name: str, number: int, sheet: str, start, end, axis: str, impedance=50.0,
               excite: bool = False, kind: PortKind = PortKind.LUMPED, depth=0.0,
               prop_dir: Optional[str] = None) -> Port:
    """A lumped or microstrip port on `sheet`, with its integration line from `start` (−) to `end` (+)."""
    return Port(name=name, kind=kind, number=number, sheet=sheet, line_start=start, line_end=end, direction=axis,
                exc_dir=axis if kind == PortKind.MICROSTRIP else None, prop_dir=prop_dir, impedance=impedance,
                excite=excite, sheet_extrude_mm=depth)


def _mode_port(name: str, number: int, sheet: str, kind: PortKind, axis: str, depth, impedance=50.0,
               excite: bool = False) -> Port:
    """A waveguide or coax port on `sheet`, launching over `depth` along its normal."""
    return Port(name=name, kind=kind, number=number, sheet=sheet, direction=axis, prop_dir=axis,
                sheet_extrude_mm=depth, impedance=impedance, excite=excite)


# --- 1. Rectangular patch antenna, 2.35 GHz, probe-fed ---

def patch_antenna(name: str) -> ModelerProject:
    """Probe-fed patch on 1.524 mm RO4350B (transmission-line model), fed L/3 in for ~50 ohm."""
    f0 = 2.45e9  # transmission-line model target
    f_res = 2.35e9  # where the full-wave solve resonates
    er = 3.66
    h = 1.524e-3

    w = C0 / (2 * f0) * math.sqrt(2 / (er + 1))
    e_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / w) ** -0.5
    dl = h * 0.412 * (e_eff + 0.3) * (w / h + 0.264) / ((e_eff - 0.258) * (w / h + 0.8))
    length = C0 / (2 * f0 * math.sqrt(e_eff)) - 2 * dl

    W, L, H = _mm(w), _mm(length), _mm(h)
    sub_w, sub_l = W + 20, L + 20
    feed_y = -L / 2 + L / 3  # L/3 in from the radiating edge

    return ModelerProject(
        name=name,
        description=(
            f"Rectangular microstrip patch on {H} mm RO4350B (εr {er}) resonating at {f_res / 1e9:.2f} GHz. "
            f"Sized by the transmission-line model for {f0 / 1e9:.2f} GHz: W = {W:.2f} mm, L = {L:.2f} mm "
            f"(ε_eff = {e_eff:.3f}); the full-wave resonance lands about 4 % lower, since the model ignores "
            "the finite ground plane and the probe's inductance. "
            f"Probe-fed {L / 3:.1f} mm in from the radiating edge, where cos²(πy₀/L) transforms the "
            "~200 Ω edge impedance down to about 50 Ω. The feed is a lumped port on a sheet the width of an "
            f"SMA pin ({PIN_W} mm) standing from the ground up to the patch, with its integration line drawn "
            "up the middle. The design sits in an open region and NF2FF is on, so the run also produces a "
            f"directivity pattern at {f_res / 1e9:.2f} GHz — expect roughly 7 dBi broadside."
        ),
        materials=[
            _pec(),
            Material(name="RO4350B", kind=MaterialKind.DIELECTRIC, epsilon=er, tan_delta=0.0037,
                     color="#7a6234", opacity=0.5),
        ],
        primitives=[
            Primitive(name="substrate", kind=PrimitiveKind.BOX, material="RO4350B", priority=0,
                      p1=(-sub_w / 2, -sub_l / 2, 0.0), p2=(sub_w / 2, sub_l / 2, H)),
            Primitive(name="ground", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=(-sub_w / 2, -sub_l / 2, 0.0), p2=(sub_w / 2, sub_l / 2, 0.0)),
            Primitive(name="patch", kind=PrimitiveKind.BOX, material="PEC", priority=10, color=GOLD,
                      p1=(-W / 2, -L / 2, H), p2=(W / 2, L / 2, H)),
            _sheet("probe", "x", (0.0, feed_y, H / 2), (PIN_W, H)),
        ],
        ports=[_line_port("feed", 1, "probe", (0.0, feed_y, 0.0), (0.0, feed_y, H), "z", excite=True)],
        boundary=BoundaryConditions(open_region=True),
        frequency=FrequencySweep(start=1.5e9, stop=3.5e9, points=801),
        mesh=MeshSettings(cells_per_wavelength=20),
        simulation=SimulationSettings(max_timesteps=30000, end_criteria=1e-4),
        nf2ff=NF2FFSettings(enabled=True, frequencies=[f_res], theta_step_deg=3.0, phi_step_deg=6.0),
    )


# --- 2. 50-ohm microstrip through-line with de-embedded MSL ports ---

def microstrip_line(name: str) -> ModelerProject:
    """Straight 50 Ω microstrip on 0.508 mm RO4350B with de-embedded MSL ports."""
    er = 3.66
    h = 0.508
    z0 = 50.0
    b = 377 * math.pi / (2 * z0 * math.sqrt(er))
    w_over_h = (2 / math.pi) * (b - 1 - math.log(2 * b - 1)
                                + (er - 1) / (2 * er) * (math.log(b - 1) + 0.39 - 0.61 / er))
    w = w_over_h * h
    line_len = 40.0
    gnd_w = w * 12

    return ModelerProject(
        name=name,
        description=(
            f"Straight 50 Ω microstrip on {h} mm RO4350B: w = {w:.3f} mm from Wheeler's synthesis "
            f"formula (w/h = {w_over_h:.2f}). Both ports are de-embedded MSL launches (openEMS "
            "AddMSLPort), each a sheet across the end of the line, under the strip, with its integration "
            "line from the ground up to the strip and a third of the line to de-embed along. Expect |S21| ≈ 0 dB "
            "and |S11| below −19 dB up to 10 GHz."
        ),
        materials=[
            _pec(),
            Material(name="RO4350B", kind=MaterialKind.DIELECTRIC, epsilon=er, tan_delta=0.0037,
                     color="#7a6234", opacity=0.5),
        ],
        primitives=[
            Primitive(name="substrate", kind=PrimitiveKind.BOX, material="RO4350B", priority=0,
                      p1=(0.0, -gnd_w / 2, 0.0), p2=(line_len, gnd_w / 2, h)),
            Primitive(name="ground", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=(0.0, -gnd_w / 2, 0.0), p2=(line_len, gnd_w / 2, 0.0)),
            Primitive(name="strip", kind=PrimitiveKind.BOX, material="PEC", priority=10, color=GOLD,
                      p1=(0.0, -w / 2, h), p2=(line_len, w / 2, h)),
            _sheet("P1 face", "x", (0.0, 0.0, h / 2), (w, h)),
            _sheet("P2 face", "x", (line_len, 0.0, h / 2), (w, h)),
        ],
        ports=[
            _line_port("P1", 1, "P1 face", (0.0, 0.0, 0.0), (0.0, 0.0, h), "z", excite=True,
                       kind=PortKind.MICROSTRIP, depth=line_len / 3, prop_dir="x"),
            _line_port("P2", 2, "P2 face", (line_len, 0.0, 0.0), (line_len, 0.0, h), "z",
                       kind=PortKind.MICROSTRIP, depth=-line_len / 3, prop_dir="x"),
        ],
        # MUR on the propagation axis makes this run diverge.
        boundary=BoundaryConditions(xmin="PML_8", xmax="PML_8", ymin="MUR", ymax="MUR", zmin="PEC", zmax="PML_8"),
        frequency=FrequencySweep(start=1e8, stop=10e9, points=1001),
        mesh=MeshSettings(cells_per_wavelength=25, air_margin=(8.0, 5.0, 8.0), max_cell_size=0.5),
        simulation=SimulationSettings(max_timesteps=25000, end_criteria=1e-5),
    )


# --- 3. Half-wave dipole in free space, 0.92 GHz ---

def dipole(name: str) -> ModelerProject:
    """Half-wave dipole cut for 1 GHz with a lumped gap port; it resonates at 0.92 GHz (~68 Ω, 2.2 dBi)."""
    f0 = 1e9  # the arms are cut for this
    f_res = 0.92e9  # where the full-wave solve resonates
    lam_mm = _mm(C0 / f0)
    arm = lam_mm * 0.238  # slightly under λ/4
    gap = 1.0
    radius = 0.5

    return ModelerProject(
        name=name,
        description=(
            f"Half-wave dipole resonating at {f_res / 1e9:.2f} GHz: two {arm:.1f} mm cylindrical arms (0.238 λ "
            f"at {f0 / 1e9:.0f} GHz) with a {gap} mm feed gap, excited by a lumped port on a sheet across the gap, "
            "the arms' width. In an open region. At resonance Zin ≈ 68 Ω, and the pattern is the 2.2 dBi "
            "doughnut."
        ),
        materials=[_pec()],
        primitives=[
            Primitive(name="arm_upper", kind=PrimitiveKind.CYLINDER, material="PEC", priority=10,
                      p1=(0.0, 0.0, gap / 2), p2=(0.0, 0.0, gap / 2 + arm), radius=radius),
            Primitive(name="arm_lower", kind=PrimitiveKind.CYLINDER, material="PEC", priority=10,
                      p1=(0.0, 0.0, -gap / 2), p2=(0.0, 0.0, -gap / 2 - arm), radius=radius),
            _sheet("gap", "x", (0.0, 0.0, 0.0), (2 * radius, gap)),
        ],
        ports=[_line_port("feed", 1, "gap", (0.0, 0.0, -gap / 2), (0.0, 0.0, gap / 2), "z", impedance=73.0,
                          excite=True)],
        boundary=BoundaryConditions(open_region=True),
        frequency=FrequencySweep(start=200e6, stop=2e9, points=901),
        mesh=MeshSettings(cells_per_wavelength=25, metal_edge_dirs="xyz"),
        simulation=SimulationSettings(max_timesteps=40000, end_criteria=1e-5),
        nf2ff=NF2FFSettings(enabled=True, frequencies=[f_res], theta_step_deg=3.0, phi_step_deg=6.0),
    )


# --- 4. WR-90 rectangular waveguide section (waveguide ports) ---

def waveguide_wr90(name: str) -> ModelerProject:
    """Straight WR-90 section with TE10 ports at both ends."""
    a, b = 22.86, 10.16
    length = 60.0
    fc = C0 / (2 * a / 1000) / 1e9

    return ModelerProject(
        name=name,
        description=(
            f"WR-90 X-band waveguide section, a = {a} mm, b = {b} mm, {length} mm long. "
            f"TE10 cutoff is {fc:.3f} GHz; the sweep covers the 8.2–12.4 GHz band where a clean "
            "run gives |S21| ≈ 0 dB. Both ends are modal TE10 ports, each a rectangle the size of the guide's "
            "cross-section launching its mode over 10 mm; the PEC boundaries are the guide's walls."
        ),
        materials=[
            Material(name="Air", kind=MaterialKind.DIELECTRIC, epsilon=1.0, color="#7fb8ff", opacity=0.1),
        ],
        primitives=[
            Primitive(name="cavity", kind=PrimitiveKind.BOX, material="Air", priority=0,
                      p1=(0.0, 0.0, 0.0), p2=(a, b, length)),
            _sheet("P1 face", "z", (a / 2, b / 2, 5.0), (a, b)),
            _sheet("P2 face", "z", (a / 2, b / 2, length - 5.0), (a, b)),
        ],
        ports=[
            _mode_port("P1", 1, "P1 face", PortKind.RECT_WAVEGUIDE, "z", 10.0, impedance=1.0, excite=True),
            _mode_port("P2", 2, "P2 face", PortKind.RECT_WAVEGUIDE, "z", -10.0, impedance=1.0),
        ],
        boundary=BoundaryConditions(xmin="PEC", xmax="PEC", ymin="PEC", ymax="PEC", zmin="PML_8", zmax="PML_8"),
        frequency=FrequencySweep(start=7e9, stop=13e9, points=601),
        mesh=MeshSettings(cells_per_wavelength=25, air_margin=(0.0, 0.0, 2.0), max_cell_size=1.0),
        simulation=SimulationSettings(max_timesteps=20000, end_criteria=1e-5),
    )


# --- 5. 50-ohm coaxial line (coaxial TEM port) ---

def coaxial_line(name: str) -> ModelerProject:
    """50 Ω air coax driven by an analytic TEM port."""
    a = 1.5
    b = a * math.exp(50.0 / 60.0)
    length = 50.0
    port_z = 12.0  # clear of the 8 absorbing cells at each end

    return ModelerProject(
        name=name,
        description=(
            f"Air-dielectric coaxial line, inner radius {a} mm, outer radius {b:.3f} mm "
            f"→ Z0 = 60·ln(b/a) = {60 * math.log(b / a):.1f} Ω, {length} mm long. Both ends run into the "
            f"absorbing boundary. The ports are ring sheets across the gap between the conductors, {port_z} mm in "
            "from each end, driven with an analytic TEM mode through openEMS's AddWaveGuidePort. Expect "
            "|S21| ≈ 0 dB and a low |S11|."
        ),
        materials=[
            _pec(),
            Material(name="Air", kind=MaterialKind.DIELECTRIC, epsilon=1.0, color="#7fb8ff", opacity=0.1),
        ],
        primitives=[
            Primitive(name="dielectric", kind=PrimitiveKind.CYLINDER, material="Air", priority=0,
                      p1=(0.0, 0.0, 0.0), p2=(0.0, 0.0, length), radius=b),
            Primitive(name="shield", kind=PrimitiveKind.CYLINDRICAL_SHELL, material="PEC", priority=5,
                      opacity=0.2, p1=(0.0, 0.0, 0.0), p2=(0.0, 0.0, length), radius=b + 0.25, shell_width=0.5),
            Primitive(name="inner", kind=PrimitiveKind.CYLINDER, material="PEC", priority=10, color=GOLD,
                      p1=(0.0, 0.0, 0.0), p2=(0.0, 0.0, length), radius=a),
            _sheet("P1 face", "z", (0.0, 0.0, port_z), shape=SheetShape.RING, radius=b, radius_inner=a,
                   segments=64),
            _sheet("P2 face", "z", (0.0, 0.0, length - port_z), shape=SheetShape.RING, radius=b, radius_inner=a,
                   segments=64),
        ],
        ports=[
            _mode_port("P1", 1, "P1 face", PortKind.COAXIAL, "z", 2.0, excite=True),
            _mode_port("P2", 2, "P2 face", PortKind.COAXIAL, "z", -2.0),
        ],
        boundary=BoundaryConditions(xmin="PEC", xmax="PEC", ymin="PEC", ymax="PEC", zmin="PML_8", zmax="PML_8"),
        frequency=FrequencySweep(start=1e8, stop=10e9, points=601),
        # No margin along z: the line has to run into the absorber, not end open.
        mesh=MeshSettings(cells_per_wavelength=25, air_margin=(1.0, 1.0, 0.0), max_cell_size=0.5),
        simulation=SimulationSettings(max_timesteps=20000, end_criteria=1e-5),
    )


# --- 6. Boolean showcase: an iris-coupled cavity ---

def cavity_with_iris(name: str) -> ModelerProject:
    """Rectangular cavity with a rounded coupling iris, built from nested booleans."""
    a, b, L = 22.86, 10.16, 40.0
    t = 1.0
    slot_w, slot_h = 10.0, 2.0

    return ModelerProject(
        name=name,
        description=(
            "Boolean showcase: a closed PEC cavity (outer box − inner box) with a rounded-end "
            "coupling iris (slot ∪ two circles) subtracted from one wall. Three CSG operations "
            "evaluated into CSXCAD polyhedra — the same shape is impossible with priority stacking alone. "
            "A probe from the floor halfway up excites it: a lumped port on a 1 mm sheet, its integration "
            "line up from the wall."
        ),
        materials=[
            Material(name="PEC", kind=MaterialKind.PEC, color="#9fb0c6", opacity=0.45),
            Material(name="Air", kind=MaterialKind.DIELECTRIC, epsilon=1.0, color="#7fb8ff", opacity=0.08),
        ],
        primitives=[
            Primitive(
                name="cavity_walls", kind=PrimitiveKind.BOOLEAN, material="PEC", priority=10,
                boolean_op=BooleanOp.SUBTRACT,
                children=[
                    Primitive(name="_shell", kind=PrimitiveKind.BOOLEAN, material="PEC",
                              boolean_op=BooleanOp.SUBTRACT, children=[
                                  Primitive(name="_outer", kind=PrimitiveKind.BOX, material="PEC",
                                            p1=(-t, -t, -t), p2=(a + t, b + t, L + t)),
                                  Primitive(name="_inner", kind=PrimitiveKind.BOX, material="PEC",
                                            p1=(0.0, 0.0, 0.0), p2=(a, b, L)),
                              ]),
                    Primitive(name="_iris", kind=PrimitiveKind.BOOLEAN, material="PEC",
                              boolean_op=BooleanOp.UNION, children=[
                                  Primitive(name="_slot", kind=PrimitiveKind.BOX, material="PEC",
                                            p1=(a / 2 - slot_w / 2, b / 2 - slot_h / 2, -2 * t),
                                            p2=(a / 2 + slot_w / 2, b / 2 + slot_h / 2, 2 * t)),
                                  Primitive(name="_end_a", kind=PrimitiveKind.CYLINDER, material="PEC",
                                            p1=(a / 2 - slot_w / 2, b / 2, -2 * t),
                                            p2=(a / 2 - slot_w / 2, b / 2, 2 * t), radius=slot_h / 2),
                                  Primitive(name="_end_b", kind=PrimitiveKind.CYLINDER, material="PEC",
                                            p1=(a / 2 + slot_w / 2, b / 2, -2 * t),
                                            p2=(a / 2 + slot_w / 2, b / 2, 2 * t), radius=slot_h / 2),
                              ]),
                ],
            ),
            Primitive(name="fill", kind=PrimitiveKind.BOX, material="Air", priority=0,
                      p1=(0.0, 0.0, 0.0), p2=(a, b, L)),
            _sheet("probe", "x", (a / 2, b / 4, L / 2), (b / 2, 1.0)),
        ],
        ports=[_line_port("probe", 1, "probe", (a / 2, 0.0, L / 2), (a / 2, b / 2, L / 2), "y", excite=True)],
        boundary=BoundaryConditions(xmin="MUR", xmax="MUR", ymin="MUR", ymax="MUR", zmin="MUR", zmax="MUR"),
        frequency=FrequencySweep(start=4e9, stop=16e9, points=801),
        mesh=MeshSettings(cells_per_wavelength=20, air_margin=(8.0, 8.0, 8.0), max_cell_size=1.5),
        simulation=SimulationSettings(max_timesteps=25000, end_criteria=1e-5),
        dumps=[
            FieldDump(name="Et_midplane", kind=DumpKind.E_TIME,
                      p1=(-t, b / 2, -t), p2=(a + t, b / 2, L + t)),
        ],
    )


# --- 7. Parametric patch antenna - every dimension derived from four variables ---

def parametric_patch(name: str) -> ModelerProject:
    """Patch with every dimension an expression over f0, er, h and margin, plus a height sweep."""
    return ModelerProject(
        name=name,
        description=(
            "Fully parametric patch on RO4350B, sized for f0 = 2.45 GHz by the transmission-line model. Every "
            "dimension is an expression over f0, er, h and margin, so changing f0 retunes the whole antenna, and "
            "the open region around it follows. The full-wave resonance lands about 4 % below f0, at 2.34 GHz; "
            "f_res holds that, and the far field is taken there. A substrate-height sweep is set up and ready "
            "to run."
        ),
        parameters=[
            Parameter(name="f0", expression="2.45GHz", unit="Hz", description="Design frequency"),
            Parameter(name="f_res", expression="0.957 * f0", unit="Hz",
                      description="Where it resonates: the transmission-line model lands about 4 % high"),
            Parameter(name="er", expression="3.66", description="Substrate relative permittivity"),
            Parameter(name="h", expression="1.524", unit="mm", description="Substrate thickness"),
            Parameter(name="margin", expression="12", unit="mm", description="Ground plane beyond the patch"),
            Parameter(name="patch_w", expression="patch_width(f0, er)", unit="mm",
                      description="Radiating edge width"),
            Parameter(name="patch_l", expression="patch_length(f0, er, h)", unit="mm",
                      description="Resonant length, fringing-corrected"),
            Parameter(name="feed_inset", expression="patch_inset(f0, er, h, 50)", unit="mm",
                      description="Inset depth that matches the edge to 50 ohm"),
            Parameter(name="feed_y", expression="-patch_l/2 + feed_inset", unit="mm",
                      description="Where the probe stands"),
            Parameter(name="feed_w", expression=str(PIN_W), unit="mm",
                      description="Width of the probe's port sheet: an SMA pin's"),
        ],
        sweeps=[
            ParameterSweep(
                name="Substrate height",
                description="Thinner laminate, narrower bandwidth - the classic patch trade.",
                variables=[SweepVariable(parameter="h", mode=SweepMode.LIST, values=[0.762, 1.524, 3.048])],
            ),
        ],
        materials=[
            _pec(),
            Material(name="RO4350B", kind=MaterialKind.DIELECTRIC, epsilon="er", tan_delta=0.0037,
                     color="#8a6f3a", opacity=0.5),
        ],
        primitives=[
            Primitive(name="substrate", kind=PrimitiveKind.BOX, material="RO4350B", priority=1,
                      p1=("-(patch_w/2 + margin)", "-(patch_l/2 + margin)", 0.0),
                      p2=("patch_w/2 + margin", "patch_l/2 + margin", "h")),
            Primitive(name="ground", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=("-(patch_w/2 + margin)", "-(patch_l/2 + margin)", 0.0),
                      p2=("patch_w/2 + margin", "patch_l/2 + margin", 0.0)),
            Primitive(name="patch", kind=PrimitiveKind.BOX, material="PEC", priority=10, color="#d9c37a",
                      p1=("-patch_w/2", "-patch_l/2", "h"),
                      p2=("patch_w/2", "patch_l/2", "h")),
            _sheet("probe", "x", (0.0, "feed_y", "h/2"), ("feed_w", "h")),
        ],
        ports=[_line_port("feed", 1, "probe", (0.0, "feed_y", 0.0), (0.0, "feed_y", "h"), "z", excite=True)],
        frequency=FrequencySweep(start="f0 * 0.75", stop="f0 * 1.25", points=601),
        mesh=MeshSettings(cells_per_wavelength=22, min_cells_per_layer=3),
        boundary=BoundaryConditions(open_region=True),
        nf2ff=NF2FFSettings(enabled=True, frequencies=["f_res"], theta_step_deg=2.0, phi_step_deg=5.0),
        simulation=SimulationSettings(max_timesteps=60000, end_criteria=1e-4, excitation=Excitation()),
    )


# --- 8. Wilkinson power divider - the canonical parametric layout ---

def wilkinson_divider(name: str) -> ModelerProject:
    """Equal-split Wilkinson divider. No isolation resistor, so S23 is optimistic."""
    w_main_mm = microstrip_width(50.0, 3.66, 0.508)  # sweep start/stop are plain numbers

    return ModelerProject(
        name=name,
        description=(
            "Equal-split Wilkinson divider at 3 GHz on 0.508 mm RO4350B. Every trace width comes from "
            "microstrip_width() and the branch length from quarter_wave(), so changing f0, er or h "
            "re-synthesises the layout. The isolation resistor is not modelled, so read S21/S31 and the "
            "input match rather than S23."
        ),
        parameters=[
            Parameter(name="f0", expression="3GHz", unit="Hz", description="Design frequency"),
            Parameter(name="er", expression="3.66", description="Substrate relative permittivity"),
            Parameter(name="h", expression="0.508", unit="mm", description="Substrate thickness"),
            Parameter(name="z0", expression="50", unit="ohm", description="System impedance"),
            Parameter(name="w_main", expression="microstrip_width(z0, er, h)", unit="mm",
                      description="Width of the feed and output lines"),
            Parameter(name="z_branch", expression="wilkinson_z(z0)", unit="ohm",
                      description="Branch impedance, z0 * sqrt(2)"),
            Parameter(name="w_branch", expression="microstrip_width(z_branch, er, h)", unit="mm",
                      description="Width of each quarter-wave branch"),
            Parameter(name="l_branch", expression="quarter_wave(f0, microstrip_eeff(w_branch, er, h))",
                      unit="mm", description="Quarter wavelength in the branch line"),
            Parameter(name="sep", expression="6 * w_main", unit="mm",
                      description="Centre-to-centre spacing of the output arms"),
            Parameter(name="feed_len", expression="8 * w_main", unit="mm",
                      description="Straight line length at each port"),
            Parameter(name="pad", expression="6", unit="mm", description="Substrate beyond the copper"),
            Parameter(name="x_split", expression="feed_len", unit="mm", description="Where the input fans out"),
            Parameter(name="l_run", expression="l_branch - sep/2 - w_main - w_branch/2", unit="mm",
                      description="Straight part of each branch: the quarter wave less the junction and bends"),
            Parameter(name="x_out", expression="x_split + l_run", unit="mm",
                      description="Where the branches meet the output lines"),
            Parameter(name="x_end", expression="x_out + feed_len", unit="mm", description="Output port plane"),
        ],
        sweeps=[
            ParameterSweep(
                name="Arm spacing",
                description="How far apart the branches sit before coupling between them spoils the match.",
                variables=[SweepVariable(parameter="sep", mode=SweepMode.LINEAR,
                                         start=4 * w_main_mm, stop=10 * w_main_mm, points=4)],
            ),
        ],
        materials=[
            Material(name="PEC", kind=MaterialKind.PEC, color="#c08840"),
            Material(name="RO4350B", kind=MaterialKind.DIELECTRIC, epsilon="er", tan_delta=0.0037,
                     color="#8a6f3a", opacity=0.45),
        ],
        primitives=[
            Primitive(name="substrate", kind=PrimitiveKind.BOX, material="RO4350B", priority=1,
                      p1=("-pad", "-(sep/2 + pad)", 0.0),
                      p2=("x_end + pad", "sep/2 + pad", "h")),
            Primitive(name="ground", kind=PrimitiveKind.BOX, material="PEC", priority=10, color=METAL,
                      p1=("-pad", "-(sep/2 + pad)", 0.0),
                      p2=("x_end + pad", "sep/2 + pad", 0.0)),
            Primitive(name="input line", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=(0.0, "-w_main/2", "h"), p2=("x_split", "w_main/2", "h")),
            Primitive(name="branch +", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=("x_split", "sep/2 - w_branch/2", "h"),
                      p2=("x_out", "sep/2 + w_branch/2", "h")),
            Primitive(name="branch -", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=("x_split", "-(sep/2 + w_branch/2)", "h"),
                      p2=("x_out", "-(sep/2 - w_branch/2)", "h")),
            Primitive(name="junction", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=("x_split", "-(sep/2 + w_branch/2)", "h"),
                      p2=("x_split + w_branch", "sep/2 + w_branch/2", "h")),
            Primitive(name="output +", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=("x_out", "sep/2 - w_main/2", "h"), p2=("x_end", "sep/2 + w_main/2", "h")),
            Primitive(name="output -", kind=PrimitiveKind.BOX, material="PEC", priority=10,
                      p1=("x_out", "-(sep/2 + w_main/2)", "h"), p2=("x_end", "-(sep/2 - w_main/2)", "h")),
            # Each port stands under its line's end, from ground up to the strip.
            _sheet("input port", "x", (0.0, 0.0, "h/2"), ("w_main", "h")),
            _sheet("out+ port", "x", ("x_end", "sep/2", "h/2"), ("w_main", "h")),
            _sheet("out- port", "x", ("x_end", "-sep/2", "h/2"), ("w_main", "h")),
        ],
        ports=[
            _line_port("input", 1, "input port", (0.0, 0.0, 0.0), (0.0, 0.0, "h"), "z", impedance="z0",
                       excite=True),
            _line_port("out+", 2, "out+ port", ("x_end", "sep/2", 0.0), ("x_end", "sep/2", "h"), "z",
                       impedance="z0"),
            _line_port("out-", 3, "out- port", ("x_end", "-sep/2", 0.0), ("x_end", "-sep/2", "h"), "z",
                       impedance="z0"),
        ],
        frequency=FrequencySweep(start="f0 * 0.5", stop="f0 * 1.5", points=601),
        # ~λ/4 clearance at the low band edge
        mesh=MeshSettings(cells_per_wavelength=20, min_cells_per_layer=3, air_margin=(27.0, 27.0, 27.0)),
        boundary=BoundaryConditions(),
        simulation=SimulationSettings(max_timesteps=80000, end_criteria=1e-4, excitation=Excitation()),
    )


def blank(name: str) -> ModelerProject:
    return ModelerProject(
        name=name,
        description="Empty design with a 1 – 10 GHz sweep, in an open region: PML on every face and the air "
                    "around the design sized for you.",
        boundary=BoundaryConditions(open_region=True),
        simulation=SimulationSettings(excitation=Excitation()),
    )


# --- 10. Sheet-defined ports: a coax-fed patch antenna ---

def coax_fed_patch(name: str) -> ModelerProject:
    """The patch fed by a 50 Ω coax through the ground, its port on a ring sheet. S11 only."""
    f0 = 2.45e9
    er = 3.66
    h = 1.524e-3

    w = C0 / (2 * f0) * math.sqrt(2 / (er + 1))
    e_eff = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 * h / w) ** -0.5
    dl = h * 0.412 * (e_eff + 0.3) * (w / h + 0.264) / ((e_eff - 0.258) * (w / h + 0.8))
    length = C0 / (2 * f0 * math.sqrt(e_eff)) - 2 * dl

    W, L, H = _mm(w), _mm(length), _mm(h)
    sub_w, sub_l = W + 20, L + 20
    feed_y = -L / 2 + L / 3

    # 50 ohm air coax: openEMS's coax port takes its wave impedance for an air filling.
    r_pin = 0.635
    r_shield = r_pin * math.exp(50.0 / 60.0)
    f_low, f_high = 1.8e9, 3.0e9  # wide enough for a short pulse, which the patch then rings down after
    air = _mm(C0 / f_low) / 4   # λ/4 above and below the board at the bottom of the sweep
    pml = 16.0                  # 8 absorbing cells of at most 2 mm, inside the domain along z
    coax_bottom = -(air + pml)
    port_z = coax_bottom + pml + 4.0

    return ModelerProject(
        name=name,
        description=(
            f"Coax-fed rectangular patch on {H} mm RO4350B: W = {W:.2f} mm, L = {L:.2f} mm, fed {L / 3:.1f} mm "
            f"in from the radiating edge by a 50 Ω air coax (pin r = {r_pin} mm, shield r = {r_shield:.3f} mm) "
            "coming up through a clearance hole in the ground. The port is assigned to a ring sheet inside the "
            "coax, so the probe's inductance and the clearance fringing are in the answer. The coax runs into "
            "the absorbing boundary below, and an air box gives the patch a quarter wavelength of air above. "
            "S11 only; the far field is in the Patch antenna template."
        ),
        materials=[
            _pec(),
            Material(name="RO4350B", kind=MaterialKind.DIELECTRIC, epsilon=er, tan_delta=0.0037,
                     color="#7a6234", opacity=0.5),
            Material(name="Air", kind=MaterialKind.DIELECTRIC, epsilon=1.0, color="#7fb8ff", opacity=0.05),
        ],
        primitives=[
            Primitive(name="substrate", kind=PrimitiveKind.BOX, material="RO4350B", priority=1,
                      p1=(-sub_w / 2, -sub_l / 2, 0.0), p2=(sub_w / 2, sub_l / 2, H)),
            Primitive(name="air above", kind=PrimitiveKind.BOX, material="Air", priority=0,
                      p1=(-sub_w / 2, -sub_l / 2, H), p2=(sub_w / 2, sub_l / 2, H + air + pml)),
            Primitive(
                name="ground", kind=PrimitiveKind.BOOLEAN, material="PEC", priority=10,
                boolean_op=BooleanOp.SUBTRACT,
                children=[
                    # 0.3 mm thick: a thinner plate forces a far smaller timestep for no visible difference.
                    Primitive(name="ground_plate", kind=PrimitiveKind.BOX, material="PEC",
                              p1=(-sub_w / 2, -sub_l / 2, -0.3), p2=(sub_w / 2, sub_l / 2, 0.0)),
                    Primitive(name="clearance", kind=PrimitiveKind.CYLINDER, material="PEC",
                              p1=(0.0, feed_y, -0.6), p2=(0.0, feed_y, 0.3), radius=r_shield),
                ],
            ),
            Primitive(name="patch", kind=PrimitiveKind.BOX, material="PEC", priority=10, color=GOLD,
                      p1=(-W / 2, -L / 2, H), p2=(W / 2, L / 2, H)),
            Primitive(name="coax_shield", kind=PrimitiveKind.CYLINDRICAL_SHELL, material="PEC", opacity=0.25,
                      priority=5, p1=(0.0, feed_y, coax_bottom), p2=(0.0, feed_y, 0.0),
                      radius=r_shield + 0.15, shell_width=0.3),
            Primitive(name="feed_pin", kind=PrimitiveKind.CYLINDER, material="PEC", priority=20, color="#d8d2c0",
                      p1=(0.0, feed_y, coax_bottom), p2=(0.0, feed_y, H), radius=r_pin),
            _sheet("coax_face", "z", (0.0, feed_y, port_z), shape=SheetShape.RING, radius=r_shield,
                   radius_inner=r_pin, segments=64),
        ],
        ports=[_mode_port("coax", 1, "coax_face", PortKind.COAXIAL, "z", 2.0, excite=True)],
        boundary=BoundaryConditions(open_region=True),
        frequency=FrequencySweep(start=f_low, stop=f_high, points=481),
        # No margin along z: the coax has to run into the absorber; the air box gives the patch its air.
        mesh=MeshSettings(cells_per_wavelength=20, air_margin=(30.0, 30.0, 0.0), max_cell_size=2.0),
        simulation=SimulationSettings(max_timesteps=80000, end_criteria=1e-4),
    )


TEMPLATES: Dict[str, Dict] = {
    "blank": {
        "title": "Blank design",
        "summary": "An empty 3D Modeler project in an open region, with sensible defaults.",
        "tags": ["start", "open region"],
        "builder": blank,
    },
    "patch_antenna": {
        "title": "Patch antenna, 2.35 GHz",
        "summary": "Probe-fed rectangular patch on RO4350B in an open region, with far-field pattern.",
        "tags": ["antenna", "nf2ff", "lumped port", "open region"],
        "builder": patch_antenna,
    },
    "microstrip_line": {
        "title": "50 Ω microstrip line",
        "summary": "Straight through-line with de-embedded MSL ports — the sanity check.",
        "tags": ["signal integrity", "msl port"],
        "builder": microstrip_line,
    },
    "dipole": {
        "title": "Half-wave dipole, 0.92 GHz",
        "summary": "Free-space wire dipole in an open region; resonates at 0.92 GHz, about 68 Ω and 2.2 dBi.",
        "tags": ["antenna", "nf2ff", "cylinder", "open region"],
        "builder": dipole,
    },
    "waveguide_wr90": {
        "title": "WR-90 waveguide section",
        "summary": "X-band straight section driven with modal TE10 ports.",
        "tags": ["waveguide", "modal port"],
        "builder": waveguide_wr90,
    },
    "coaxial_line": {
        "title": "50 Ω coaxial line",
        "summary": "Air coax with analytic TEM mode ports.",
        "tags": ["transmission line", "coax port"],
        "builder": coaxial_line,
    },
    "cavity_iris": {
        "title": "Iris-coupled cavity (CSG)",
        "summary": "Union, subtract and nested booleans evaluated into CSXCAD polyhedra.",
        "tags": ["boolean", "cavity", "field dump"],
        "builder": cavity_with_iris,
    },
    "parametric_patch": {
        "title": "Parametric patch antenna",
        "summary": "Every dimension derived from f0, er and h - retune by editing one number. Sweep included.",
        "tags": ["parametric", "antenna", "sweep", "nf2ff", "open region"],
        "builder": parametric_patch,
    },
    "coax_fed_patch": {
        "title": "Coax-fed patch (coax port)",
        "summary": "The patch fed by a 50 Ω coax through the ground, its port on a ring sheet. S11 only.",
        "tags": ["antenna", "coax port", "nf2ff", "open region"],
        "builder": coax_fed_patch,
    },
    "wilkinson_divider": {
        "title": "Wilkinson power divider",
        "summary": "Equal-split 3-port; every width synthesised from its impedance. Sweep included.",
        "tags": ["parametric", "power divider", "sweep", "3 ports"],
        "builder": wilkinson_divider,
    },
}


def list_templates() -> List[Dict]:
    return [
        {"id": key, "title": t["title"], "summary": t["summary"], "tags": t["tags"], "kind": "modeler"}
        for key, t in TEMPLATES.items()
    ]


def build(template_id: str, name: str) -> ModelerProject:
    if template_id not in TEMPLATES:
        raise KeyError(template_id)
    builder: Callable[[str], ModelerProject] = TEMPLATES[template_id]["builder"]
    return builder(name)
