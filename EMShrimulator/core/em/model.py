"""Modeler project schema. Lengths in mm, angles in degrees, frequencies in Hz.

`Expr` fields hold a number or an expression; params.resolve() evaluates the fields in EXPR_FIELDS.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import Dict, List, Literal, Optional, Set, Tuple, Type, Union

from pydantic import BaseModel, Field, model_validator

Expr = Union[float, str]

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]
Vec3E = Tuple[Expr, Expr, Expr]
Vec2E = Tuple[Expr, Expr]
Axis = Literal["x", "y", "z"]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


LengthUnit = Literal["mm", "um", "cm", "m", "mil", "in"]
FrequencyUnit = Literal["Hz", "kHz", "MHz", "GHz"]


class DisplayUnits(BaseModel):
    """Display units only; stored values are always in base units."""

    length: LengthUnit = "mm"
    frequency: FrequencyUnit = "GHz"


class TransformOp(BaseModel):
    """One step of a transform chain; maps onto CSTransform.AddTransform."""

    kind: Literal["translate", "rotate_axis", "rotate_origin", "scale"]
    translate: Optional[Vec3E] = None          # kind == translate (mm)
    axis: Optional[Axis] = None                # kind == rotate_axis
    angle_deg: Optional[Expr] = None           # kind == rotate_axis / rotate_origin
    scale: Optional[Vec3E] = None              # kind == scale


CSKind = Literal["relative", "object"]
CSMode = Literal["offset", "rotated", "both"]
Anchor = Literal["centre", "face", "edge", "corner"]


class CoordinateSystem(BaseModel):
    """A named coordinate system, offset and/or rotated from its base (another system, or a point of an object).

    Rotation is about the base's X, then Y, then Z axis.
    """

    name: str
    kind: CSKind = "relative"
    parent: Optional[str] = None  # base system name; None = global
    object_id: Optional[str] = None          # object: the Primitive.id it sits on
    anchor: Anchor = "centre"                # object: the kind of point
    anchor_at: Optional[Vec3] = None  # position within the object's bounding box, 0..1
    mode: CSMode = "offset"
    origin: Vec3E = (0.0, 0.0, 0.0)          # offset, mm (mode offset or both)
    angles: Vec3E = (0.0, 0.0, 0.0)          # rotation about X, Y, Z, degrees (mode rotated or both)
    # Set by params.resolve, never saved: the global 4x4 placement (row-major), or why it can't be placed.
    placement: Optional[List[float]] = Field(default=None, exclude=True)
    placement_error: str = Field(default="", exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _from_transforms(cls, data):
        """Systems were once saved as a transform chain; read one as its offset and rotation."""
        if isinstance(data, dict) and "transforms" in data:
            from EMShrimulator.core.em.transforms import legacy_offset_rotation

            data = dict(data)
            ops = [op if isinstance(op, TransformOp) else TransformOp.model_validate(op)
                   for op in data.pop("transforms") or []]
            if not any(key in data for key in ("mode", "origin", "angles")):
                data.update(legacy_offset_rotation(ops))
        return data


class MaterialKind(str, Enum):
    PEC = "pec"                 # perfect electric conductor (CSX Metal)
    DIELECTRIC = "dielectric"   # CSX Material with epsilon/mue/kappa
    CONDUCTOR = "conductor"     # CSX Material with finite kappa (lossy metal, e.g. copper)
    SHEET = "sheet"             # CSX ConductingSheet: thin lossy metal (kappa + thickness)


class Material(BaseModel):
    name: str
    kind: MaterialKind = MaterialKind.DIELECTRIC
    epsilon: Expr = 1.0         # relative permittivity
    mue: Expr = 1.0             # relative permeability
    kappa: Expr = 0.0           # electric conductivity S/m
    tan_delta: Expr = 0.0  # loss tangent, wideband via Debye poles (dielectric.py)
    thickness_um: Expr = 35.0   # SHEET only: conducting-sheet thickness
    color: str = "#8899aa"      # UI only
    opacity: float = 0.85       # UI only


class SheetShape(str, Enum):
    """Outline of a zero-thickness sheet. RING is the coax cross-section."""

    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    RING = "ring"
    POLYGON = "polygon"


class PrimitiveKind(str, Enum):
    BOX = "box"
    SHEET = "sheet"
    CYLINDER = "cylinder"
    CYLINDRICAL_SHELL = "cylindrical_shell"
    SPHERE = "sphere"
    SPHERICAL_SHELL = "spherical_shell"
    POLYGON = "polygon"
    ROT_POLY = "rot_poly"
    WIRE = "wire"
    CURVE = "curve"
    BOOLEAN = "boolean"


class BooleanOp(str, Enum):
    UNION = "union"
    INTERSECT = "intersect"
    SUBTRACT = "subtract"
    XOR = "xor"
    HULL = "hull"


class Primitive(BaseModel):
    """One modeled solid. BOOLEAN is evaluated by csg.py into a polyhedron."""

    id: str = Field(default_factory=_new_id)
    name: str
    kind: PrimitiveKind
    material: str                # Material.name
    coordinate_system: Optional[str] = None  # CoordinateSystem.name, None = global
    priority: int = 1            # CSXCAD priority: higher wins on overlap
    visible: bool = True         # UI only
    model: bool = True  # False: drawn and snapped to, but never meshed or solved
    color: Optional[str] = None      # UI only; None = the material's
    opacity: Optional[float] = None  # UI only; None = the material's
    transforms: List[TransformOp] = Field(default_factory=list)  # per-object, applied after its CS

    # BOX / CYLINDER / CYLINDRICAL_SHELL / WIRE end points
    p1: Optional[Vec3E] = None
    p2: Optional[Vec3E] = None
    size_mode: bool = False  # UI only: edit p2 as a size from p1

    # CYLINDER / CYLINDRICAL_SHELL / SPHERE / SPHERICAL_SHELL / WIRE
    radius: Optional[Expr] = None
    shell_width: Optional[Expr] = None    # *_SHELL only
    center: Optional[Vec3E] = None        # SPHERE / SPHERICAL_SHELL

    # POLYGON / ROT_POLY: 2D points extruded along `normal_axis` from `elevation`
    points: Optional[List[Vec2E]] = None
    normal_axis: Optional[Axis] = None
    elevation: Optional[Expr] = None
    extrude_length: Optional[Expr] = None
    rot_axis: Optional[Axis] = None        # ROT_POLY: axis to sweep around
    angle_start_deg: Optional[Expr] = None
    angle_stop_deg: Optional[Expr] = None

    # SHEET: outline in the plane normal to `normal_axis`, centred on `origin`.
    # RECTANGLE/ELLIPSE use sheet_size, CIRCLE radius, RING radius and radius_inner, POLYGON points.
    sheet_shape: Optional[SheetShape] = None
    origin: Optional[Vec3E] = None
    sheet_size: Optional[Vec2E] = None
    radius_inner: Optional[Expr] = None
    sheet_segments: int = 48

    # CURVE / WIRE polyline
    path: Optional[List[Vec3E]] = None

    # BOOLEAN
    boolean_op: Optional[BooleanOp] = None
    children: List["Primitive"] = Field(default_factory=list)


Primitive.model_rebuild()


class PortKind(str, Enum):
    LUMPED = "lumped"  # two-terminal port: R + excitation
    MICROSTRIP = "microstrip"    # openEMS AddMSLPort: de-embedded microstrip launch
    RECT_WAVEGUIDE = "rect_waveguide"  # openEMS AddRectWaveGuidePort, TE/TM modes
    COAXIAL = "coaxial"          # analytic TEM coax mode via AddWaveGuidePort


class Port(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    kind: PortKind = PortKind.LUMPED
    number: int                  # 1-based port number, drives Sxx numbering
    # Ignored when `sheet` is set.
    p1: Vec3E = (0.0, 0.0, 0.0)
    p2: Vec3E = (0.0, 0.0, 0.0)
    size_mode: bool = False       # UI only, as on Primitive
    direction: Axis               # excitation/current direction across the gap (lumped)
    impedance: Expr = 50.0
    excite: bool = False          # every True port gets its own FDTD run
    coordinate_system: Optional[str] = None
    priority: int = 100
    active: bool = True           # False = measured only, never excited

    # MICROSTRIP
    prop_dir: Optional[Axis] = None
    exc_dir: Optional[Axis] = None
    feed_shift_mm: Expr = 0.0
    measure_plane_shift_mm: Expr = 0.0

    # RECT_WAVEGUIDE: cross-section a/b (mm) and mode, e.g. "TE10"
    wg_a_mm: Optional[Expr] = None
    wg_b_mm: Optional[Expr] = None
    mode: str = "TE10"

    # COAXIAL: inner/outer radius (mm), axis is derived from p1->p2
    r_inner_mm: Optional[Expr] = None
    r_outer_mm: Optional[Expr] = None

    # The SHEET primitive this port is assigned to.
    sheet: Optional[str] = None
    # Extrusion along the sheet normal (mm). Microstrip ports need a non-zero value.
    sheet_extrude_mm: Expr = 0.0
    # Integration line from the negative to the positive terminal, in the sheet's coordinate system.
    # Lumped and microstrip ports need one to run.
    line_start: Optional[Vec3E] = None
    line_end: Optional[Vec3E] = None


BoundaryType = Literal["PEC", "PMC", "MUR", "PML_8"]
BOUNDARY_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


class BoundaryConditions(BaseModel):
    xmin: BoundaryType = "PML_8"
    xmax: BoundaryType = "PML_8"
    ymin: BoundaryType = "PML_8"
    ymax: BoundaryType = "PML_8"
    zmin: BoundaryType = "PML_8"
    zmax: BoundaryType = "PML_8"
    # Open region: every face absorbing, air margin of λ/4 at the lowest operating frequency (region.py).
    open_region: bool = False


class MeshSettings(BaseModel):
    cells_per_wavelength: float = 20.0   # resolution at frequency.stop
    smooth_ratio: float = 1.4            # max ratio between neighboring cell sizes
    max_cell_size: Optional[Expr] = None  # mm; if set, caps cell size regardless of wavelength
    min_cell_size: Optional[Expr] = None  # mm; floor for auto-refinement, guards runaway grids
    # Minimum cells across layers thinner than one cell (1 = off).
    min_cells_per_layer: int = 3
    metal_edge_refinement: bool = True   # openEMS AddEdges2Grid on metal properties
    metal_edge_dirs: str = "xy"          # which directions get the 1/3-2/3 metal edge rule
    extra_lines_x: List[Expr] = Field(default_factory=list)
    extra_lines_y: List[Expr] = Field(default_factory=list)
    extra_lines_z: List[Expr] = Field(default_factory=list)
    air_margin: Vec3E = (10.0, 10.0, 10.0)  # mm of free space around the geometry bbox


class FrequencySweep(BaseModel):
    start: Expr = 1e9
    stop: Expr = 10e9
    points: int = 601


class ExcitationKind(str, Enum):
    GAUSSIAN = "gaussian"
    SINUSOIDAL = "sinusoidal"
    STEP = "step"
    DIRAC = "dirac"


class Excitation(BaseModel):
    kind: ExcitationKind = ExcitationKind.GAUSSIAN
    # GAUSSIAN: f0/fc default to the sweep centre/half-width.
    f0: Optional[Expr] = None
    fc: Optional[Expr] = None
    # SINUSOIDAL
    frequency: Optional[Expr] = None


# Dielectric loss: "wideband" uses Debye poles (slower), "centre" a conductivity exact at the band centre.
DielectricLoss = Literal["wideband", "centre"]


class SimulationSettings(BaseModel):
    max_timesteps: int = 100_000
    end_criteria: float = 1e-5  # energy decay stop criterion
    oversampling: int = 0       # 0 = engine default
    excitation: Excitation = Field(default_factory=Excitation)
    dielectric_loss: DielectricLoss = "wideband"
    # Adaptive passes: refine the mesh by `refine_ratio` until no S-parameter moves more than `delta_s`.
    max_passes: int = 1         # 1 = one solve on the mesh as set
    delta_s: float = 0.02  # largest |ΔS| between passes
    refine_ratio: float = 1.3   # cells-per-wavelength growth from one pass to the next


class DumpKind(str, Enum):
    E_TIME = "e_time"
    H_TIME = "h_time"
    E_FREQ = "e_freq"
    H_FREQ = "h_freq"
    CURRENT_TIME = "current_time"
    SAR_LOCAL = "sar_local"


class FieldDump(BaseModel):
    """A field-probe box openEMS writes into <results>/dumps/."""

    id: str = Field(default_factory=_new_id)
    name: str
    kind: DumpKind = DumpKind.E_TIME
    p1: Vec3E
    p2: Vec3E
    frequencies: List[Expr] = Field(default_factory=list)  # *_FREQ kinds only
    sub_sampling: Optional[int] = None


class NF2FFSettings(BaseModel):
    """Near-field-to-far-field box for antenna patterns."""

    enabled: bool = False
    # 0: openEMS sizes the box; > 0 insets it that many cells.
    margin_cells: int = 0
    frequencies: List[Expr] = Field(default_factory=list)  # empty = sweep centre
    theta_step_deg: float = 2.0
    phi_step_deg: float = 2.0


class Parameter(BaseModel):
    """A named design variable; may reference other parameters in any order."""

    name: str
    expression: str = "0"
    description: str = ""
    unit: str = ""   # UI label only, e.g. "mm", "GHz"; never converts a value


class SweepMode(str, Enum):
    LINEAR = "linear"   # `points` values evenly spaced from start to stop
    LOG = "log"         # `points` values on a log scale (start and stop > 0)
    LIST = "list"       # exactly the values in `values`


class SweepVariable(BaseModel):
    parameter: str                 # Parameter.name this varies
    mode: SweepMode = SweepMode.LINEAR
    start: float = 0.0
    stop: float = 1.0
    points: int = 3
    values: List[float] = Field(default_factory=list)


class SweepCombine(str, Enum):
    PRODUCT = "product"  # every combination: 3 x 3 = 9 runs
    ZIP = "zip"          # walk the variables together: 3 and 3 = 3 runs


class ParameterSweep(BaseModel):
    """Design points solved one after another, each a full re-resolve and run."""

    name: str = "Sweep 1"
    description: str = ""
    enabled: bool = True
    combine: SweepCombine = SweepCombine.PRODUCT
    variables: List[SweepVariable] = Field(default_factory=list)


# A report: what is plotted (kind, fixed) and how (display).
ReportKind = Literal["s", "y", "z", "vswr", "group_delay", "tdr", "farfield", "antenna", "nearfield"]
DisplayKind = Literal["rect", "polar", "smith", "table", "contour", "spherical", "cylindrical", "overlay", "rect3d"]
ComplexFunction = Literal["db", "mag", "phase", "re", "im", "re_im"]
PatternQuantity = Literal["re", "directivity", "gain", "realized_gain", "axial_ratio"]
Polarisation = Literal["total", "theta", "phi", "rhcp", "lhcp", "l3x", "l3y"]
TraceCategory = Literal["s", "y", "z", "vswr", "group_delay", "tdr", "antenna",
                        "re", "directivity", "gain", "realized_gain", "axial_ratio"]
TraceFunction = Literal["db", "mag", "phase", "re", "im", "lin", "value"]


class ReportTrace(BaseModel):
    """One trace of a report: a category (S, Z, gain...), a quantity (S11, Z21...) and a function (dB, phase...)."""

    category: TraceCategory = "s"
    term: str = ""                                   # S11 / Z21 / Zin1; a polarisation; an antenna parameter
    function: TraceFunction = "db"
    axis: Literal["auto", "y1", "y2"] = "auto"       # auto: Y2 when its unit is not the first trace's
    theta_deg: float = 0.0                           # a far-field trace against frequency: its direction
    phi_deg: float = 0.0
    visible: bool = True

# Plots saved before reports had a display: their kind named both.
LEGACY_PLOT_KINDS: Dict[str, Dict[str, object]] = {
    "s_db": {"kind": "s", "display": "rect", "function": "db"},
    "s_phase": {"kind": "s", "display": "rect", "function": "phase"},
    "smith": {"kind": "s", "display": "smith"},
    "z": {"kind": "z", "display": "rect", "function": "re_im"},
    "vswr": {"kind": "vswr", "display": "rect"},
    "group_delay": {"kind": "group_delay", "display": "rect"},
    "tdr": {"kind": "tdr", "display": "rect"},
    "pattern_polar": {"kind": "farfield", "display": "polar"},
    "pattern_map": {"kind": "farfield", "display": "contour"},
}
_LEGACY_CUTS = {"both": [0.0, 90.0], "e": [0.0], "h": [90.0]}


class ResultPlot(BaseModel):
    """A user report, drawn from the newest run. UI only."""

    id: str = Field(default_factory=_new_id)
    name: str
    kind: ReportKind = "s"
    display: DisplayKind = "rect"
    # The traces drawn; empty means the kind's defaults.
    curves: List[ReportTrace] = Field(default_factory=list)
    # S/Y/Z terms, or antenna parameters; empty = the usual defaults for the kind
    traces: List[str] = Field(default_factory=list)
    function: ComplexFunction = "db"                 # how a complex quantity is drawn on rectangular axes
    log_x: bool = False
    # far field
    frequency_hz: Optional[float] = None             # None = the first computed frequency
    all_frequencies: bool = False                    # cuts at every far-field frequency, overlaid
    quantity: PatternQuantity = "directivity"
    component: Polarisation = "total"
    cross_pol: bool = False                          # also draw the polarisation orthogonal to `component`
    sweep: Literal["theta", "phi"] = "theta"         # the angle a cut runs along; `cuts` hold the other fixed
    cuts: List[float] = Field(default_factory=lambda: [0.0, 90.0])
    db: bool = True                                  # pattern in dB (dBi, dBV); False = linear
    range_db: float = 40.0                           # dynamic range of polar and 3D patterns
    # TDR
    rise_time_ps: Optional[float] = None             # None = the fastest edge the sweep allows
    tdr_axis: Literal["distance", "time"] = "distance"
    tdr_port: Optional[int] = None                   # None = the first port whose reflection was measured
    # near field: a frequency-domain field dump
    dump: str = ""
    port: Optional[int] = None                       # the run (excited port) it is read from; None = the first
    field_component: Literal["total", "x", "y", "z"] = "total"
    slice_axis: Optional[Axis] = None                # a box dump is cut across this axis; None = its thinnest
    slice_mm: Optional[float] = None                 # None = through the middle
    line_axis: Optional[Axis] = None  # in-plane axis of a line cut; None = the longer one
    line_mm: Optional[float] = None                  # where it crosses the other; None = the middle

    @model_validator(mode="before")
    @classmethod
    def _legacy_kind(cls, data):
        """Upgrade plot kinds saved before displays existed, like `s_db`."""
        if not isinstance(data, dict) or "display" in data or data.get("kind") not in LEGACY_PLOT_KINDS:
            return data
        data = {**data, **LEGACY_PLOT_KINDS[data["kind"]]}
        cut = data.pop("cut", None)
        if data["kind"] == "farfield" and cut in _LEGACY_CUTS:
            data["cuts"] = list(_LEGACY_CUTS[cut])
        return data


class ModelerProject(BaseModel):
    schema_version: Literal[1] = 1
    name: str
    description: str = ""
    parameters: List[Parameter] = Field(default_factory=list)
    sweeps: List[ParameterSweep] = Field(default_factory=list)
    display_units: DisplayUnits = Field(default_factory=DisplayUnits)
    coordinate_systems: List[CoordinateSystem] = Field(default_factory=list)
    working_cs: Optional[str] = None  # coordinate system new objects are drawn in; None = global
    materials: List[Material] = Field(
        default_factory=lambda: [
            Material(name="PEC", kind=MaterialKind.PEC, color="#c9ced6"),
            Material(name="Copper", kind=MaterialKind.CONDUCTOR, kappa=5.8e7, color="#b87333"),
            Material(name="FR4", kind=MaterialKind.DIELECTRIC, epsilon=4.3, tan_delta=0.02, color="#2f7a3f", opacity=0.55),
            Material(name="Rogers RO4350B", kind=MaterialKind.DIELECTRIC, epsilon=3.66, tan_delta=0.0037, color="#8a6f3a", opacity=0.55),
            Material(name="Air", kind=MaterialKind.DIELECTRIC, epsilon=1.0, color="#7fb8ff", opacity=0.15),
        ]
    )
    primitives: List[Primitive] = Field(default_factory=list)
    ports: List[Port] = Field(default_factory=list)
    boundary: BoundaryConditions = Field(default_factory=BoundaryConditions)
    mesh: MeshSettings = Field(default_factory=MeshSettings)
    frequency: FrequencySweep = Field(default_factory=FrequencySweep)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    dumps: List[FieldDump] = Field(default_factory=list)
    nf2ff: NF2FFSettings = Field(default_factory=NF2FFSettings)
    plots: List[ResultPlot] = Field(default_factory=list)
    ui_state: Dict[str, object] = Field(default_factory=dict)


# Fields params.resolve() evaluates; everything else is left as-is.
EXPR_FIELDS: Dict[Type[BaseModel], Set[str]] = {
    TransformOp: {"translate", "angle_deg", "scale"},
    CoordinateSystem: {"origin", "angles"},
    Material: {"epsilon", "mue", "kappa", "tan_delta", "thickness_um"},
    Primitive: {
        "p1", "p2", "radius", "shell_width", "center", "points",
        "elevation", "extrude_length", "angle_start_deg", "angle_stop_deg", "path",
        "origin", "sheet_size", "radius_inner",
    },
    Port: {
        "p1", "p2", "impedance", "feed_shift_mm", "measure_plane_shift_mm",
        "wg_a_mm", "wg_b_mm", "r_inner_mm", "r_outer_mm", "sheet_extrude_mm", "line_start", "line_end",
    },
    MeshSettings: {
        "max_cell_size", "min_cell_size",
        "extra_lines_x", "extra_lines_y", "extra_lines_z", "air_margin",
    },
    FrequencySweep: {"start", "stop"},
    Excitation: {"f0", "fc", "frequency"},
    FieldDump: {"p1", "p2", "frequencies"},
    NF2FFSettings: {"frequencies"},
}
