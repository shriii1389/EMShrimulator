"""Subprocess: `python -m EMShrimulator.core.em.run_worker <project.json> <out_dir>`.

One FDTD run per excited port, then Touchstone, optional far field and summary.json.
`@@GEMS` lines on stdout carry structured progress for the UI.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from openEMS.ports import WaveguidePort

from EMShrimulator.core.em import farfield, params
from EMShrimulator.core.em.csx_builder import build_simulation
from EMShrimulator.core.em.model import ModelerProject
from EMShrimulator.core.em.touchstone import write_touchstone
from EMShrimulator.core.em.validate import validate


def emit(event: str, **fields) -> None:
    """One structured progress line. Plain text stays readable in the log."""
    print(f"@@GEMS {json.dumps({'event': event, **fields})}", flush=True)


def force_utf8_stdout() -> None:
    """Log messages contain non-ASCII (λ, ε, Ω) that cp1252 can't encode."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
            pass


def _geometry_center(project: ModelerProject, built) -> list[float]:
    """Centre of the modelled geometry in mm, falling back to the domain centre."""
    from EMShrimulator.core.em.csx_builder import global_aabb

    cs_by_name = {cs.name: cs for cs in project.coordinate_systems}
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for prim in project.primitives:
        if not prim.model:
            continue
        try:
            pmin, pmax = global_aabb(prim, cs_by_name)
        except Exception:  # noqa: BLE001 - a shape we cannot bound is skipped
            continue
        lo = np.minimum(lo, pmin)
        hi = np.maximum(hi, pmax)
    if not np.isfinite(lo).all():
        return ((built.domain_min + built.domain_max) / 2).tolist()
    return ((lo + hi) / 2).tolist()


def load_resolved(project_path: Path, overrides: Optional[Dict[str, float]] = None) -> ModelerProject:
    """Read a project and resolve its expressions to numbers."""
    raw = ModelerProject.model_validate_json(project_path.read_text(encoding="utf-8"))
    resolved, values, errors = params.resolve(raw, overrides)
    if values:
        print("Parameters: " + ", ".join(f"{k} = {v:.6g}" for k, v in sorted(values.items())), flush=True)
    for error in errors:
        print(f"[ERROR] parameter '{error['target']}': {error['message']}", flush=True)
    if errors:
        raise ValueError(f"{len(errors)} parameter expression(s) could not be evaluated")
    return resolved


SMatrix = Dict[int, Dict[int, np.ndarray]]  # [excited][measured]


def refined(project: ModelerProject, factor: float) -> ModelerProject:
    """The design meshed `factor` times finer."""
    if factor == 1.0:
        return project
    finer = project.model_copy(deep=True)
    finer.mesh.cells_per_wavelength = project.mesh.cells_per_wavelength * factor
    if finer.mesh.max_cell_size:
        finer.mesh.max_cell_size = float(project.mesh.max_cell_size) / factor
    return finer


def _cells(project: ModelerProject) -> Optional[int]:
    """The cell count a design would mesh to, without solving it."""
    from EMShrimulator.core.em.preview import mesh_preview

    try:
        return int(mesh_preview(project)["cells"])
    except Exception:  # noqa: BLE001 - a count that cannot be had only skips the check
        return None


def next_factor(project: ModelerProject, factor: float, ratio: float, previous_cells: Optional[int],
                tries: int = 4) -> float:
    """The refinement for the next pass: `ratio` finer, and finer still until the cell count really grows."""
    factor *= ratio
    for _ in range(tries):
        cells = _cells(refined(project, factor))
        if previous_cells is None or cells is None or cells > previous_cells:
            break
        factor *= ratio
    return factor


def max_delta_s(previous: SMatrix, current: SMatrix) -> float:
    """The largest |S_new - S_old| over every shared term and frequency."""
    worst = 0.0
    for excited, column in current.items():
        for measured, values in column.items():
            old = previous.get(excited, {}).get(measured)
            if old is not None and len(old) == len(values):
                worst = max(worst, float(np.max(np.abs(np.asarray(values) - np.asarray(old)))))
    return worst


def port_powers(port_obj, freqs: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Incident and accepted power at a driven port over `freqs`."""
    try:
        incident = 0.5 * np.real(np.asarray(port_obj.uf_inc) * np.conj(np.asarray(port_obj.if_inc)))
        accepted = 0.5 * np.real(np.asarray(port_obj.uf_tot) * np.conj(np.asarray(port_obj.if_tot)))
    except AttributeError:
        return None
    if incident.shape != freqs.shape or accepted.shape != freqs.shape:
        return None
    return {"incident": incident, "accepted": accepted}


def add_gain(payload: Dict, freqs: np.ndarray, powers: Optional[Dict[str, np.ndarray]]) -> None:
    """Gain from accepted power; realized gain from incident power."""
    if not powers:
        return
    for entry in payload.get("frequencies", []):
        f = entry["frequency_hz"]
        accepted = float(np.interp(f, freqs, powers["accepted"]))
        incident = float(np.interp(f, freqs, powers["incident"]))
        prad = float(entry.get("radiated_power_w") or 0.0)
        entry["accepted_power_w"] = accepted
        entry["incident_power_w"] = incident
        if accepted > 0 and prad > 0:
            entry["efficiency"] = prad / accepted
            entry["gain_dbi"] = float(entry["dmax_dbi"] + 10 * np.log10(prad / accepted))
        if incident > 0 and prad > 0:
            entry["realized_gain_dbi"] = float(entry["dmax_dbi"] + 10 * np.log10(prad / incident))


def clear_run_dir(path: Path, attempts: int = 6) -> None:
    """Remove a previous run's folder, retrying while Windows still holds a file in it."""
    os.chdir(path.parent)  # Run() leaves the process inside the last sim folder
    for attempt in range(attempts):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            path.mkdir()
            return
        gc.collect()  # release solver objects still holding files
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Could not clear {path}: something still has a file in it open.")


C0 = 299_792_458.0


def mode_impedance(port_obj, freqs: np.ndarray) -> np.ndarray:
    """Wave impedance k0·Z0/β of a mode port's mode in its filling: Z0/√εr for TEM. openEMS's own ZL scales k0 by
    the filling's index too, which gives Z0 for any TEM line."""
    eps = getattr(port_obj, "fill_epsilon", 1.0)
    k0 = 2 * np.pi * np.asarray(freqs, dtype=float) / C0
    beta = np.sqrt((k0 ** 2 * eps - port_obj.kc ** 2).astype(complex))
    return k0 * farfield.ETA0 / beta


def _solve_ports(project: ModelerProject, out_dir: Path, freqs: np.ndarray, z_ref: Dict[int, float],
                 label: str, run_offset: int, run_total: int, pass_info: Optional[Dict] = None):
    """One FDTD run per excited port. Returns the S columns and the first run (for the far field)."""
    excited = [p for p in project.ports if p.excite]
    s_matrix: SMatrix = {}
    first = None
    powers: Dict[int, Dict[str, np.ndarray]] = {}
    for k, src_port in enumerate(excited):
        index = run_offset + k + 1
        emit("run_begin", index=index, total=run_total, port=src_port.number, name=src_port.name,
             **(pass_info or {}))
        prefix = f"{label} " if label else ""
        print(f"=== {prefix}[{index}/{run_total}] Exciting port {src_port.number} ({src_port.name}) ===", flush=True)

        sim_path = (out_dir / f"sim_port_{src_port.number}").resolve()
        built = build_simulation(project, excite_port_number=src_port.number)

        grid = built.CSX.GetGrid()
        cells = 1
        for axis in "xyz":
            cells *= max(grid.GetQtyLines(axis) - 1, 1)
        emit("mesh", cells=int(cells), resolution_mm=built.resolution_mm)
        print(f"Mesh: {cells:,} cells at {built.resolution_mm:.4f} mm base resolution", flush=True)

        clear_run_dir(sim_path)
        built.FDTD.Run(str(sim_path), verbose=0, cleanup=False)

        for number, port_obj in built.ports.items():
            # A mode port's voltage and current are mode integrals: it is referenced to the mode's wave impedance.
            reference = mode_impedance(port_obj, freqs) if isinstance(port_obj, WaveguidePort) else z_ref[number]
            port_obj.CalcPort(str(sim_path), freqs, ref_impedance=reference)

        src_obj = built.ports[src_port.number]
        incident = src_obj.uf_inc
        column = s_matrix.setdefault(src_port.number, {})
        for number, port_obj in built.ports.items():
            # uf_* are voltage waves; S is a ratio of power waves.
            column[number] = port_obj.uf_ref / incident * np.sqrt(src_obj.Z_ref / port_obj.Z_ref)
        measured = port_powers(src_obj, freqs)
        if measured is not None:
            powers[src_port.number] = measured
        if first is None:
            first = {"built": built, "sim_path": sim_path, "cells": int(cells), "powers": measured}

        emit("run_end", index=index, total=run_total, port=src_port.number)
        print(f"=== Port {src_port.number} run complete ===", flush=True)
    if first is not None:
        first["all_powers"] = powers
    return s_matrix, first


def solve_project(
    project: ModelerProject,
    out_dir: Path,
    *,
    run_validation: bool = True,
    label: str = "",
) -> Dict[str, Any]:
    """Solve one resolved design into `out_dir` and return the summary. Raises if it can't run."""
    # openEMS Run() needs an absolute path matching os.getcwd().
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_validation:
        report = validate(project)
        for issue in report["issues"]:
            if issue["severity"] in ("error", "warning"):
                print(f"[{issue['severity'].upper()}] {issue['message']}", flush=True)
                if issue["hint"]:
                    print(f"          {issue['hint']}", flush=True)
        if not report["can_run"]:
            raise ValueError("design has validation errors")

    # Remove an earlier run's pattern.
    for stale in ("farfield.json", farfield.FIELDS_FILE):
        (out_dir / stale).unlink(missing_ok=True)

    excited = [p for p in project.ports if p.excite]
    all_numbers = sorted(p.number for p in project.ports)
    z_ref = {p.number: float(p.impedance) for p in project.ports}
    freqs = np.linspace(project.frequency.start, project.frequency.stop, project.frequency.points)
    passes = max(1, int(project.simulation.max_passes))
    target = float(project.simulation.delta_s)
    ratio = max(float(project.simulation.refine_ratio), 1.01)

    emit("start", runs=len(excited) * passes, ports=all_numbers, label=label, passes=passes,
         freq_start=project.frequency.start, freq_stop=project.frequency.stop)

    started = time.time()
    s_matrix: Optional[SMatrix] = None
    first = None
    final = project
    history = []
    converged = passes == 1
    factor = 1.0
    for n in range(1, passes + 1):
        if n > 1:
            factor = next_factor(project, factor, ratio, first["cells"] if first else None)
        current = refined(project, factor)
        pass_info = {"pass_index": n, "pass_total": passes} if passes > 1 else None
        if passes > 1:
            emit("pass_begin", index=n, total=passes, cells_per_wavelength=current.mesh.cells_per_wavelength)
            print(f"--- Adaptive pass {n}/{passes}: {current.mesh.cells_per_wavelength:.3g} cells per wavelength ---",
                  flush=True)
        first = None  # release the last pass's solver objects before clearing folders
        gc.collect()
        columns, first = _solve_ports(current, out_dir, freqs, z_ref, label, (n - 1) * len(excited),
                                      len(excited) * passes, pass_info)
        entry: Dict[str, Any] = {"pass": n, "cells_per_wavelength": current.mesh.cells_per_wavelength,
                                 "cells": first["cells"] if first else None}
        if s_matrix is not None:
            entry["delta_s"] = max_delta_s(s_matrix, columns)
            print(f"Pass {n}: max |ΔS| = {entry['delta_s']:.4g} (target {target:g})", flush=True)
            emit("pass_end", index=n, total=passes, delta_s=entry["delta_s"], target=target)
        history.append(entry)
        s_matrix, final = columns, current
        if entry.get("delta_s") is not None and entry["delta_s"] <= target:
            converged = True
            print(f"Converged after {n} passes.", flush=True)
            break
    if passes > 1 and not converged:
        print(f"WARNING: not converged after {passes} passes (max |ΔS| {history[-1].get('delta_s', 0):.4g}, "
              f"target {target:g}). Raise the pass limit or loosen ΔS.", flush=True)

    farfield_payload = None
    if final.nf2ff.enabled and first is not None and first["built"].nf2ff is not None:
        nf_freqs = final.nf2ff.frequencies or [float(np.mean([final.frequency.start, final.frequency.stop]))]
        print(f"Computing far field at {', '.join(f'{f / 1e9:.3f} GHz' for f in nf_freqs)} ...", flush=True)
        emit("nf2ff_begin", frequencies=nf_freqs)
        try:
            # Phase centre at the geometry centre, not the padded domain.
            center_mm = _geometry_center(final, first["built"])
            farfield_payload, fields = farfield.compute(
                first["built"].nf2ff, str(first["sim_path"]), nf_freqs,
                final.nf2ff.theta_step_deg, final.nf2ff.phi_step_deg, center_mm=center_mm,
            )
            add_gain(farfield_payload, freqs, first["powers"])
            (out_dir / farfield.FIELDS_FILE).unlink(missing_ok=True)
            farfield.write(farfield_payload, out_dir / "farfield.json", fields)
            for entry in farfield_payload["frequencies"]:
                gain = f", G {entry['gain_dbi']:.2f} dBi" if entry.get("gain_dbi") is not None else ""
                print(
                    f"  {entry['frequency_hz'] / 1e9:.3f} GHz: Dmax {entry['dmax_dbi']:.2f} dBi{gain}, "
                    f"Prad {entry['radiated_power_w']:.4g} W, F/B {entry['front_to_back_db']:.1f} dB",
                    flush=True,
                )
                consistency = entry.get("consistency")
                if consistency is not None and abs(consistency - 1) > farfield.CONSISTENCY_TOLERANCE:
                    print(f"WARNING: at {entry['frequency_hz'] / 1e9:.3f} GHz the far field carries "
                          f"{consistency:.0%} of the power through the NF2FF box. The box must enclose the whole "
                          "design, clear of the absorbing boundary: raise the air margin.", flush=True)
            emit("nf2ff_done")
        except Exception as exc:  # noqa: BLE001 - a pattern failure must not lose the S-params
            print(f"WARNING: far-field transform failed: {exc}", flush=True)
            traceback.print_exc()
            emit("nf2ff_failed", error=str(exc))

    if final.dumps:
        from EMShrimulator.core.em import nearfield

        try:
            sims = {p.number: out_dir / f"sim_port_{p.number}" for p in excited}
            written = nearfield.export(out_dir, sims, [d.name for d in final.dumps])
            if written:
                print(f"Near-field data: {len(written)} frequency-domain dump(s) ready for reports.", flush=True)
        except Exception as exc:  # noqa: BLE001 - the S-parameters stand without them
            print(f"WARNING: could not read the field dumps for reports: {exc}", flush=True)

    ts_path = write_touchstone(freqs, s_matrix or {}, all_numbers, [z_ref[n] for n in all_numbers],
                               out_dir / project.name)

    summary = {
        "project": project.name,
        "touchstone": ts_path.name,
        "ports": all_numbers,
        "excited": [p.number for p in excited],
        "elapsed_s": time.time() - started,
        "farfield": (out_dir / "farfield.json").name if farfield_payload else None,
        "dumps": sorted(p.name for p in project.dumps),
        "passes": history,
        "converged": converged,
        "finished_at": time.time(),
        # Each driven port's incident and accepted power, used to scale reports to 1 W incident.
        "port_powers": {
            "freq_hz": freqs.tolist(),
            **{str(number): {key: values.tolist() for key, values in measured.items()}
               for number, measured in ((first or {}).get("all_powers") or {}).items()},
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    force_utf8_stdout()
    if len(sys.argv) != 3:
        print("usage: run_worker.py <project.json> <out_dir>", file=sys.stderr)
        return 2

    project_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    try:
        project = load_resolved(project_path)
    except ValueError as exc:
        emit("failed", reason="parameters")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        summary = solve_project(project, out_dir)
    except ValueError as exc:
        emit("failed", reason="validation")
        print(f"ERROR: {exc}; fix the reported issues and run again.", file=sys.stderr)
        return 1

    emit("complete", touchstone=summary["touchstone"], elapsed_s=summary["elapsed_s"])
    print(f"RESULT_TOUCHSTONE={out_dir / summary['touchstone']}", flush=True)
    print(f"RUN_COMPLETE in {summary['elapsed_s']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - make crashes readable in the job log
        traceback.print_exc()
        emit("failed", reason="exception")
        raise SystemExit(1)
