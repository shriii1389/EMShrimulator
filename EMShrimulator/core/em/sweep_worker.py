"""Subprocess: `python -m EMShrimulator.core.em.sweep_worker <project.json> <out_dir> <sweep name>`.

Solves each point in turn and rewrites the index after each. A failed point doesn't stop the sweep.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from EMShrimulator.core.em import sweep as sweep_mod
from EMShrimulator.core.em.model import ModelerProject
from EMShrimulator.core.em.params import resolve
from EMShrimulator.core.em.run_worker import emit, force_utf8_stdout, solve_project


def main() -> int:
    force_utf8_stdout()
    if len(sys.argv) != 4:
        print("usage: sweep_worker.py <project.json> <out_dir> <sweep name>", file=sys.stderr)
        return 2

    project_path = Path(sys.argv[1]).resolve()
    results_dir = Path(sys.argv[2]).resolve()
    sweep_name = sys.argv[3]

    raw = ModelerProject.model_validate_json(project_path.read_text(encoding="utf-8"))
    spec = next((s for s in raw.sweeps if s.name == sweep_name), None)
    if spec is None:
        print(f"ERROR: no sweep named '{sweep_name}' in this project.", file=sys.stderr)
        emit("failed", reason="no-such-sweep")
        return 1

    try:
        points = sweep_mod.expand(spec)
    except sweep_mod.SweepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        emit("failed", reason="bad-sweep")
        return 1

    print(f"=== Sweep '{spec.name}': {len(points)} design point(s) ===", flush=True)
    for i, point in enumerate(points):
        print(f"  {i:3d}  {sweep_mod.point_label(point)}", flush=True)
    emit("sweep_start", sweep=spec.name, points=len(points))

    entries: List[Dict[str, Any]] = [
        {
            "index": i,
            "values": point,
            "label": sweep_mod.point_label(point),
            "status": "pending",
            "touchstone": None,
            "farfield": None,
            "elapsed_s": None,
            "error": None,
        }
        for i, point in enumerate(points)
    ]
    sweep_mod.write_index(results_dir, spec, entries)

    started = time.time()
    succeeded = 0

    for i, point in enumerate(points):
        entry = entries[i]
        label = f"[point {i + 1}/{len(points)}]"
        print(f"\n=== {label} {entry['label']} ===", flush=True)
        emit("sweep_point_begin", index=i, total=len(points), values=point, label=entry["label"])
        entry["status"] = "running"
        sweep_mod.write_index(results_dir, spec, entries)

        point_dir = sweep_mod.point_dir(results_dir, spec.name, i)
        try:
            resolved, values, errors = resolve(raw, point)
            if errors:
                raise ValueError("; ".join(f"{e['target']}: {e['message']}" for e in errors))
            entry["parameters"] = values
            summary = solve_project(resolved, point_dir, label=label)
        except Exception as exc:  # noqa: BLE001 - one bad point must not end the sweep
            entry["status"] = "failed"
            entry["error"] = str(exc)
            print(f"[ERROR] {label} failed: {exc}", flush=True)
            traceback.print_exc()
            emit("sweep_point_failed", index=i, error=str(exc))
        else:
            entry["status"] = "done"
            entry["touchstone"] = summary["touchstone"]
            entry["farfield"] = summary["farfield"]
            entry["elapsed_s"] = summary["elapsed_s"]
            entry["ports"] = summary["ports"]
            succeeded += 1
            emit("sweep_point_end", index=i, total=len(points), elapsed_s=summary["elapsed_s"])
            print(f"=== {label} complete in {summary['elapsed_s']:.1f}s ===", flush=True)

        sweep_mod.write_index(results_dir, spec, entries)

    elapsed = time.time() - started
    emit("complete", sweep=spec.name, points=len(points), succeeded=succeeded, elapsed_s=elapsed)
    print(f"\nRUN_COMPLETE sweep '{spec.name}': {succeeded}/{len(points)} points in {elapsed:.1f}s", flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - make crashes readable in the job log
        traceback.print_exc()
        emit("failed", reason="exception")
        raise SystemExit(1)
