"""Layout examples built from gerber2ems's bundled boards, with their ports already placed."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from EMShrimulator.config import GERBER2EMS_EXAMPLES
from EMShrimulator.core.layout.model import (
    DifferentialPair,
    LayoutFrequency,
    LayoutPort,
    LayoutPortKind,
    LayoutProject,
    SingleEndedTrace,
    StackupLayer,
    pairs_from_lines,
)

DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "differential": {
        "title": "Differential pair",
        "summary": "A 4-port differential pair on a 4-layer stack — the board the layout engine was verified on.",
    },
    "filter": {
        "title": "Microstrip filter",
        "summary": "A filter section with measured VNA reference data alongside it.",
    },
    "meander_loose": {
        "title": "Meander, loose coupling",
        "summary": "Serpentine delay line with wide spacing; compare against the tight variant.",
    },
    "meander_tight": {
        "title": "Meander, tight coupling",
        "summary": "The same delay line meandered tightly — shows coupling-induced crosstalk.",
    },
    "stub_short": {
        "title": "Short stub",
        "summary": "Open-stub discontinuity on a through-line; a clean TDR/​S11 teaching case.",
    },
    "stub_long": {
        "title": "Long stub",
        "summary": "The same board with a longer stub — the resonant notch moves down in frequency.",
    },
}


def _example_dir(example_id: str) -> Path:
    d = GERBER2EMS_EXAMPLES / example_id
    if not d.is_dir() or not (d / "fab").is_dir():
        raise KeyError(example_id)
    return d


def list_examples() -> List[Dict]:
    if not GERBER2EMS_EXAMPLES.is_dir():
        return []
    out = []
    for d in sorted(GERBER2EMS_EXAMPLES.iterdir()):
        if not (d / "fab").is_dir():
            continue
        meta = DESCRIPTIONS.get(d.name, {"title": d.name, "summary": "Bundled gerber2ems example board."})
        gerbers = sorted(p.name for p in (d / "fab").glob("*_Cu.gbr"))
        out.append({
            "id": d.name,
            "title": meta["title"],
            "summary": meta["summary"],
            "tags": ["gerber2ems example", f"{len(gerbers)} copper layers"],
            "kind": "layout",
        })
    return out


def _stackup_from_json(path: Path) -> List[StackupLayer]:
    data = json.loads(path.read_text(encoding="utf-8"))
    layers: List[StackupLayer] = []
    for entry in data.get("layers", []):
        kind = entry.get("type") or "other"
        lowered = str(kind).lower()
        if "copper" in lowered:
            kind = "copper"
        elif "core" in lowered:
            kind = "core"
        elif "prepreg" in lowered or "dielectric" in lowered:
            kind = "prepreg"
        elif kind not in (
            "Top Silk Screen", "Bottom Silk Screen", "Top Solder Mask", "Bottom Solder Mask",
            "Top Solder Paste", "Bottom Solder Paste",
        ):
            kind = "other"
        layers.append(StackupLayer(
            name=entry.get("name") or entry.get("user-name") or "layer",
            type=kind,
            thickness_mm=entry.get("thickness"),
            material=entry.get("material"),
            epsilon=entry.get("epsilon"),
            loss_tangent=entry.get("lossTangent"),
        ))
    return layers


def _ports_from_example(example: Path) -> List[LayoutPort]:
    """Merge the example's `simulation.json` port list with its pos.csv positions."""
    sim = json.loads((example / "simulation.json").read_text(encoding="utf-8"))
    definitions = sim.get("ports", [])

    positions: Dict[int, tuple[float, float, float]] = {}
    for pos_file in (example / "fab").glob("*pos.csv"):
        with pos_file.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 6:
                    continue
                if "Simulation_Port" in row[2] or "Simulation-Port" in row[2]:
                    try:
                        index = int(row[0][2:]) - 1
                        positions[index] = (float(row[3]), float(row[4]), float(row[5]))
                    except ValueError:
                        continue

    ports: List[LayoutPort] = []
    for i, definition in enumerate(definitions):
        x, y, rot = positions.get(i, (0.0, 0.0, 0.0))
        ports.append(LayoutPort(
            name=definition.get("name") or f"P{i + 1}",
            number=i,
            kind=LayoutPortKind.MICROSTRIP,
            x_mm=x,
            y_mm=y,
            rotation_deg=rot,
            width_um=float(definition.get("width", 200)),
            length_um=float(definition.get("length", 1000)),
            impedance=float(definition.get("impedance", 50)),
            layer=int(definition.get("layer", 0)),
            plane=int(definition.get("plane", 1)),
            excite=bool(definition.get("excite", i == 0)),
            db_margin=float(definition.get("dB_margin", -15)),
        ))
    return ports


def build(example_id: str, name: str, project_dir: Path) -> LayoutProject:
    """Copy an example's fab files into `project_dir` and return its project."""
    example = _example_dir(example_id)
    fab_dst = project_dir / "fab"
    fab_dst.mkdir(parents=True, exist_ok=True)

    gerber_files: List[str] = []
    drill_file: Optional[str] = None
    for src in sorted((example / "fab").iterdir()):
        if not src.is_file():
            continue
        # Skip pos.csv: project_writer regenerates it.
        if src.name.lower().endswith("pos.csv"):
            continue
        shutil.copy2(src, fab_dst / src.name)
        if src.suffix.lower() == ".gbr":
            gerber_files.append(src.name)
        elif src.suffix.lower() == ".drl" and "PTH" in src.name.upper():
            drill_file = src.name

    # Positions are already board-local, so copy them unchanged.
    sim = json.loads((example / "simulation.json").read_text(encoding="utf-8"))
    stackup_path = example / "fab" / "stackup.json"
    meta = DESCRIPTIONS.get(example_id, {})

    project = LayoutProject(
        name=name,
        description=(
            f"{meta.get('summary', '')} Imported from gerber2ems's bundled `{example_id}` example "
            "(Antmicro SI test board); ports are at the coordinates its authors used."
        ).strip(),
        stackup=_stackup_from_json(stackup_path) if stackup_path.exists() else [],
        ports=_ports_from_example(example),
        traces=[SingleEndedTrace(**t) for t in sim.get("traces", [])],
        diff_pairs=[DifferentialPair(**d) for d in pairs_from_lines(sim.get("differential_pairs", []))],
        frequency=LayoutFrequency(
            start=float(sim.get("frequency", {}).get("start", 1e6)),
            stop=float(sim.get("frequency", {}).get("stop", 6e9)),
        ),
        max_steps=int(sim.get("max_steps", 100_000)),
        pixel_size_um=float(sim.get("pixel_size", 5.0)),
        gerber_files=gerber_files,
        drill_file=drill_file,
    )
    return project
