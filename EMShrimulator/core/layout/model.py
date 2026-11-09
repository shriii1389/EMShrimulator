"""Layout (PCB / gerber2ems) project schema. Fields mirror gerber2ems's simulation.json and stackup.json."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from EMShrimulator.core.em.model import ResultPlot

StackupLayerType = Literal[
    "copper", "core", "prepreg", "Top Silk Screen", "Bottom Silk Screen",
    "Top Solder Mask", "Bottom Solder Mask", "Top Solder Paste", "Bottom Solder Paste", "other",
]


class StackupLayer(BaseModel):
    name: str
    type: StackupLayerType
    thickness_mm: Optional[float] = None
    material: Optional[str] = None
    epsilon: Optional[float] = None
    loss_tangent: Optional[float] = None


class LayoutPortKind(str, Enum):
    """How a port excites the board: MICROSTRIP launch, LUMPED gap port, or CIRCUIT port between two terminals."""

    MICROSTRIP = "microstrip"
    LUMPED = "lumped"
    CIRCUIT = "circuit"


class LayoutPort(BaseModel):
    name: str
    number: int  # 0-based, matches gerber2ems port index / refdes-1
    kind: LayoutPortKind = LayoutPortKind.MICROSTRIP
    # The port position; for a circuit port, the + terminal.
    x_mm: float
    y_mm: float
    # Width runs along (cos r, sin r); a microstrip launch extends along (-sin r, cos r).
    rotation_deg: float = 0.0
    width_um: float = 200
    length_um: float = 1000
    impedance: float = 50
    layer: int = 0   # index into the metal (copper) layers, 0 = first/top
    # Reference layer index; for a circuit port, the - terminal's layer.
    plane: int = 1
    # "auto": the nearest layer with copper under the port, resolved at run time.
    reference: Literal["auto", "layer"] = "layer"
    # Circuit ports only: the - terminal, board-local mm.
    neg_x_mm: Optional[float] = None
    neg_y_mm: Optional[float] = None
    excite: bool = False
    db_margin: float = -15
    net: Optional[str] = None  # informational; shown on the canvas
    snap: Optional[str] = None      # what the position was snapped to, e.g. "trace end · F_Cu"
    neg_snap: Optional[str] = None


class SingleEndedTrace(BaseModel):
    name: Optional[str] = None
    start: int  # port number
    stop: int
    nets: List[str] = Field(default_factory=list)


class DifferentialPair(BaseModel):
    """A differential pair of + and − ports, with the mode impedances for mixed-mode S-parameters."""

    name: str = "D1"
    positive: Optional[int] = None  # port number of the + side
    negative: Optional[int] = None  # port number of the − side
    z_diff: float = 100.0
    z_common: float = 25.0


def pairs_from_lines(entries: List[Dict]) -> List[Dict]:
    """gerber2ems differential "lines" as pairs, one at each end."""
    out: List[Dict] = []
    for entry in entries:
        if not isinstance(entry, dict) or "start_p" not in entry:
            out.append(entry)
            continue
        base = entry.get("name") or ""
        for end, (p, n) in (("start", (entry.get("start_p"), entry.get("start_n"))),
                            ("stop", (entry.get("stop_p"), entry.get("stop_n")))):
            out.append({"name": f"{base} {end}".strip() if base else "", "positive": p, "negative": n})
    for i, entry in enumerate(out):
        if isinstance(entry, dict) and not entry.get("name"):
            entry["name"] = f"D{i + 1}"
    return out


class CopperObject(BaseModel):
    """One piece of copper in board-local mm: a trace, a region or a pad (see core.layout.copper)."""

    kind: Literal["trace", "region", "pad"]
    points: List[Tuple[float, float]]
    heights: List[float] = Field(default_factory=list)
    width_mm: float = 0.0


class CopperEdit(BaseModel):
    """Replacement pieces for the `index`-th object of `layer`'s Gerber; none deletes it."""

    layer: str
    index: int
    objects: List[CopperObject] = Field(default_factory=list)


class PadstackLayer(BaseModel):
    """One copper layer's pad and antipad where they differ from the padstack's own."""

    layer: str  # e.g. "In1_Cu"
    pad_mm: Optional[float] = None  # None: no pad on this layer
    antipad_mm: float = 0.0


class Padstack(BaseModel):
    """A via or hole definition: drill, plating, layer span, and a pad and antipad per layer.

    shape "circle": round pads of `pad_mm`. "imported": each via keeps the pads it was imported with.
    """

    name: str
    hole_mm: float = 0.3  # finished drill diameter
    plated: bool = True  # unplated holes are drawn but not simulated
    plating_um: float = 25.0
    fill_epsilon: float = 1.0  # what fills the barrel: 1 open, ~3.5 resin plugged
    shape: Literal["circle", "imported"] = "circle"
    pad_mm: Optional[float] = 0.6  # pad on every spanned layer unless `layers` overrides; None: no pads
    antipad_mm: float = 0.0
    start_layer: Optional[str] = None  # span, top to bottom; None: the board's outer layer
    stop_layer: Optional[str] = None
    layers: List[PadstackLayer] = Field(default_factory=list)


class Backdrill(BaseModel):
    """A backdrill from one face, stopping `stub_mm` short of `keep_layer`."""

    keep_layer: str
    diameter_mm: float = 0.5
    stub_mm: float = 0.05


class ViaLink(BaseModel):
    """An imported pad taken over by a via: the `index`-th object of `layer`'s Gerber, `dx, dy` off centre."""

    layer: str
    index: int
    dx: float = 0.0
    dy: float = 0.0


class Via(BaseModel):
    """A padstack instance: one via (or hole) on the board, board-local mm."""

    id: int
    padstack: str
    x_mm: float
    y_mm: float
    net: Optional[str] = None
    backdrill_top: Optional[Backdrill] = None
    backdrill_bottom: Optional[Backdrill] = None
    links: List[ViaLink] = Field(default_factory=list)


class LayoutFrequency(BaseModel):
    start: float = 1e6
    stop: float = 6e9


class ViaSettings(BaseModel):
    """What padstacks taken from a drill file start with; each padstack then has its own."""

    plating_thickness_um: float = 25.0
    filling_epsilon: float = 1.0


class LayoutConvergence(BaseModel):
    """When the FDTD run stops: ΔS, energy decay or max timesteps, whichever comes first."""

    delta_s: float = 0.02
    end_criteria_db: float = -50.0


class LayoutProject(BaseModel):
    schema_version: Literal[1] = 1
    name: str
    description: str = ""
    stackup: List[StackupLayer] = Field(default_factory=list)
    ports: List[LayoutPort] = Field(default_factory=list)
    traces: List[SingleEndedTrace] = Field(default_factory=list)
    diff_pairs: List[DifferentialPair] = Field(default_factory=list)
    frequency: LayoutFrequency = Field(default_factory=LayoutFrequency)
    max_steps: int = 100_000
    convergence: LayoutConvergence = Field(default_factory=LayoutConvergence)
    pixel_size_um: float = 5.0
    # "wideband": Debye poles (slower); "centre": exact at the centre frequency only.
    dielectric_loss: Literal["wideband", "centre"] = "wideband"
    via: ViaSettings = Field(default_factory=ViaSettings)
    # gerber2ems --export-field choices; empty = none.
    export_field: List[str] = Field(default_factory=list)
    use_pml: bool = False
    # Files in this project's fab/ directory.
    gerber_files: List[str] = Field(default_factory=list)
    drill_file: Optional[str] = None
    source_format: Literal["gerber", "odb++"] = "gerber"
    # Copper edits by imported object; the Gerbers in fab/ are written from these.
    copper_edits: List[CopperEdit] = Field(default_factory=list)
    # Padstacks and vias. None: the drill file hasn't been read into vias yet.
    padstacks: List[Padstack] = Field(default_factory=list)
    vias: Optional[List[Via]] = None
    # Reports under Results, edited by core.em.edits' plot functions.
    plots: List[ResultPlot] = Field(default_factory=list)
    ui_state: Dict[str, object] = Field(default_factory=dict)

    @field_validator("diff_pairs", mode="before")
    @classmethod
    def _old_pairs(cls, value):
        """Projects saved before pairs were HFSS-style held gerber2ems's four-port lines."""
        return pairs_from_lines(list(value)) if isinstance(value, list) else value


# --- stackup presets ---

def _copper(name: str, thickness_mm: float = 0.035) -> StackupLayer:
    return StackupLayer(name=name, type="copper", thickness_mm=thickness_mm)


def _dielectric(name: str, kind: str, thickness_mm: float, material: str, eps: float, tand: float) -> StackupLayer:
    return StackupLayer(name=name, type=kind, thickness_mm=thickness_mm, material=material,
                        epsilon=eps, loss_tangent=tand)


STACKUP_PRESETS: Dict[str, Dict] = {
    "2layer_fr4_1mm6": {
        "title": "2-layer FR4, 1.6 mm",
        "summary": "The default JLCPCB/PCBWay two-layer build: 1.6 mm FR4 core, 1 oz copper.",
        "layers": [
            _copper("F_Cu"),
            _dielectric("Dielectric 1", "core", 1.51, "FR4", 4.3, 0.02),
            _copper("B_Cu"),
        ],
    },
    "4layer_fr4_1mm6": {
        "title": "4-layer FR4, 1.6 mm",
        "summary": "Signal / GND / PWR / Signal, the standard 1.6 mm four-layer stack.",
        "layers": [
            _copper("F_Cu"),
            _dielectric("Prepreg 1", "prepreg", 0.2104, "FR4", 4.3, 0.02),
            _copper("In1_Cu", 0.0175),
            _dielectric("Core", "core", 1.065, "FR4", 4.3, 0.02),
            _copper("In2_Cu", 0.0175),
            _dielectric("Prepreg 2", "prepreg", 0.2104, "FR4", 4.3, 0.02),
            _copper("B_Cu"),
        ],
    },
    "4layer_ro4350b": {
        "title": "4-layer Rogers RO4350B",
        "summary": "Low-loss RF build: RO4350B cores, εr 3.66, tanδ 0.0037.",
        "layers": [
            _copper("F_Cu"),
            _dielectric("RO4350B 1", "core", 0.508, "RO4350B", 3.66, 0.0037),
            _copper("In1_Cu", 0.0175),
            _dielectric("RO4450F", "prepreg", 0.202, "RO4450F", 3.52, 0.004),
            _copper("In2_Cu", 0.0175),
            _dielectric("RO4350B 2", "core", 0.508, "RO4350B", 3.66, 0.0037),
            _copper("B_Cu"),
        ],
    },
    "2layer_ro4003c": {
        "title": "2-layer Rogers RO4003C, 0.508 mm",
        "summary": "Thin low-loss microwave laminate, εr 3.38 — matches the bundled test_patch example.",
        "layers": [
            _copper("F_Cu"),
            _dielectric("RO4003C", "core", 0.508, "RO4003C", 3.38, 0.0027),
            _copper("B_Cu"),
        ],
    },
}


def preset_list() -> List[Dict]:
    return [
        {"id": key, "title": v["title"], "summary": v["summary"], "layers": len(v["layers"])}
        for key, v in STACKUP_PRESETS.items()
    ]


def apply_preset(preset_id: str) -> List[StackupLayer]:
    if preset_id not in STACKUP_PRESETS:
        raise KeyError(preset_id)
    return [layer.model_copy(deep=True) for layer in STACKUP_PRESETS[preset_id]["layers"]]
