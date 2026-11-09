"""Write a LayoutProject as the files gerber2ems reads: fab/stackup.json, a pos.csv, simulation.json and
vias.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from EMShrimulator.core.layout.model import LayoutPortKind, LayoutProject
from EMShrimulator.core.layout.padstacks import sim_vias


def write_stackup_json(project: LayoutProject, fab_dir: Path) -> None:
    layers = []
    for layer in project.stackup:
        layers.append({
            "name": layer.name,
            "type": layer.type,
            "color": None,
            "thickness": layer.thickness_mm,
            "material": layer.material,
            "epsilon": layer.epsilon,
            "lossTangent": layer.loss_tangent,
            "user-name": layer.name,
        })
    (fab_dir / "stackup.json").write_text(
        json.dumps({"layers": layers, "format_version": "1.0"}, indent=2), encoding="utf-8")


def write_port_positions_csv(project: LayoutProject, fab_dir: Path) -> None:
    """Write the KiCad pick-and-place CSV gerber2ems reads ports from, removing other *pos.csv files."""
    for stale in fab_dir.glob("*pos.csv"):
        stale.unlink()

    path = fab_dir / f"{project.name}-top-pos.csv"
    ports_sorted = sorted(project.ports, key=lambda p: p.number)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Ref", "Val", "Package", "PosX", "PosY", "Rot", "Side"])
        for port in ports_sorted:
            w.writerow([
                f"SP{port.number + 1}", "Simulation_Port", "Simulation_Port",
                f"{port.x_mm:.6f}", f"{port.y_mm:.6f}", f"{port.rotation_deg:.3f}", "top",
            ])


def write_simulation_json(project: LayoutProject, project_dir: Path) -> None:
    """`project` must have its "auto" references resolved (layout.ports.resolved_project)."""
    ports = []
    for p in sorted(project.ports, key=lambda p: p.number):
        entry = {
            "name": p.name,
            "number": p.number + 1,
            "type": p.kind.value if isinstance(p.kind, LayoutPortKind) else str(p.kind),
            "width": p.width_um,
            "length": p.length_um,
            "impedance": p.impedance,
            "layer": p.layer,
            "plane": p.plane,
            "dB_margin": p.db_margin,
            "excite": p.excite,
        }
        if p.kind == LayoutPortKind.CIRCUIT and p.neg_x_mm is not None and p.neg_y_mm is not None:
            # The + terminal goes in pos.csv; the - terminal here.
            entry["neg_position"] = [round(p.neg_x_mm, 6), round(p.neg_y_mm, 6)]
        ports.append(entry)
    cfg = {
        "format_version": "1.2",
        "frequency": {"start": project.frequency.start, "stop": project.frequency.stop},
        "max_steps": project.max_steps,
        "convergence": {
            "delta_s": project.convergence.delta_s,
            "end_criteria": 10 ** (project.convergence.end_criteria_db / 10),
        },
        "pixel_size": project.pixel_size_um,
        "dielectric_loss": project.dielectric_loss,
        "via": {
            "plating_thickness": project.via.plating_thickness_um,
            "filling_epsilon": project.via.filling_epsilon,
        },
        "ports": ports,
        "traces": [
            {"start": t.start, "stop": t.stop, "nets": t.nets, **({"name": t.name} if t.name else {})}
            for t in project.traces
        ],
        # Differential pairs are handled by the app (core.em.results), not gerber2ems.
        "differential_pairs": [],
    }
    (project_dir / "simulation.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def write_vias_json(project: LayoutProject, project_dir: Path) -> None:
    """The vias for gerber2ems; none if the drill file hasn't been read into vias."""
    path = project_dir / "vias.json"
    if project.vias is None:
        path.unlink(missing_ok=True)
        return
    path.write_text(json.dumps({"format_version": "1.0", "vias": sim_vias(project)}, indent=1), encoding="utf-8")


def write_all(project: LayoutProject, project_dir: Path) -> None:
    fab_dir = project_dir / "fab"
    fab_dir.mkdir(exist_ok=True)
    write_stackup_json(project, fab_dir)
    write_port_positions_csv(project, fab_dir)
    write_simulation_json(project, project_dir)
    write_vias_json(project, project_dir)
