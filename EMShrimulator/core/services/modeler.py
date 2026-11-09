"""3D Modeler operations. openEMS-dependent imports are lazy (they take seconds)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from EMShrimulator.core import jobs, workers
from EMShrimulator.core import project as storage
from EMShrimulator.core.em import params as em_params
from EMShrimulator.core.em import sweep as em_sweep
from EMShrimulator.core.em.model import MaterialKind, ModelerProject
from EMShrimulator.core.errors import InvalidInput, NotFound, ValidationFailed


# --- persistence ---

def load(name: str) -> ModelerProject:
    try:
        kind = storage.get_kind(name)
    except FileNotFoundError as exc:
        raise NotFound(f"There is no project named '{name}'.") from exc
    except storage.InvalidName as exc:
        raise InvalidInput(str(exc)) from exc
    if kind != "modeler":
        raise InvalidInput(f"'{name}' is not a 3D Modeler project.")
    return storage.load_project(name)  # type: ignore[return-value]


def save(name: str, project: ModelerProject) -> None:
    storage.save_project(name, project)


def results_dir(name: str) -> Path:
    return storage.project_dir(name) / "results"


# --- parameters ---

@dataclass
class Resolution:
    """A project with its expressions folded into numbers, and what failed."""

    project: ModelerProject
    values: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error_for(self, target: str) -> Optional[Dict[str, str]]:
        return next((e for e in self.errors if e["target"] == target), None)


def resolve(project: ModelerProject, overrides: Optional[Dict[str, float]] = None) -> Resolution:
    resolved, values, errors = em_params.resolve(project, overrides)
    return Resolution(resolved, values, errors)


def expression_issues(errors: List[Dict[str, str]]) -> List[Dict]:
    """Expression failures, shaped as validation issues."""
    return [
        {"severity": "error", "code": "bad-expression", "message": e["message"], "hint": "",
         "target": e.get("label") or e["target"]}
        for e in errors
    ]


def resolved_or_raise(project: ModelerProject) -> ModelerProject:
    """The numeric project everything that touches the engine works from."""
    result = resolve(project)
    if result.errors:
        raise ValidationFailed(
            f"{len(result.errors)} parameter expression(s) could not be evaluated.",
            expression_issues(result.errors),
        )
    return result.project


def function_reference() -> Dict[str, Any]:
    """The functions, constants and units an expression may use."""
    return {
        "functions": [
            {"name": key, "signature": em_params.FUNCTION_HELP.get(key, f"{key}(...)"),
             "group": "design" if key in em_params.DESIGN_FUNCTIONS else "math"}
            for key in sorted(em_params.FUNCTIONS)
        ],
        "constants": [{"name": key, "value": value} for key, value in sorted(em_params.CONSTANTS.items())],
        "length_units": dict(em_params.LENGTH_UNITS),
        "frequency_units": dict(em_params.FREQUENCY_UNITS),
        "suffixes": sorted(em_params.UNIT_FACTORS),
    }


# --- previews and validation ---

def validate(project: ModelerProject) -> Dict:
    """Design-rule report, with expression failures first."""
    return validate_resolution(resolve(project))


def validate_resolution(result: Resolution) -> Dict:
    """`validate` for a project that has already been resolved."""
    from EMShrimulator.core.em import validate as em_validate

    report = em_validate.validate(result.project)
    if result.errors:
        issues = expression_issues(result.errors) + report["issues"]
        report = {**report, "issues": issues, "errors": report["errors"] + len(result.errors), "can_run": False}
    return report


def mesh_preview(resolved: ModelerProject) -> Dict:
    """The real CSXCAD grid for an already-resolved project."""
    from EMShrimulator.core.em import preview as em_preview

    return em_preview.mesh_preview(resolved)


def geometry_preview(resolved: ModelerProject) -> Dict:
    from EMShrimulator.core.em import preview as em_preview

    return em_preview.geometry_preview(resolved)


def sweep_preview(project: ModelerProject) -> Dict[str, Dict]:
    """Every sweep expanded into its design points, or why it cannot be."""
    return {sweep.name: em_sweep.describe(sweep) for sweep in project.sweeps}


# --- runs ---

def _raise_unless_runnable(project: ModelerProject, message: str) -> None:
    report = validate(project)
    if not report["can_run"]:
        raise ValidationFailed(message, [i for i in report["issues"] if i["severity"] == "error"])


def start_run(name: str, project: ModelerProject) -> jobs.Job:
    """Validate and launch one FDTD run. The worker reads project.json, so save first."""
    _raise_unless_runnable(project, "The design has validation errors and cannot be simulated.")
    pdir = storage.project_dir(name)
    excited = sum(1 for p in project.ports if p.excite)
    cmd = workers.command("run", str(pdir / "project.json"), str(pdir / "results"))
    return jobs.start_job(
        "modeler", name, cmd, cwd=workers.working_directory(), log_dir=pdir / "jobs",
        env=workers.environment(),
        label=f"FDTD · {excited} excitation{'s' if excited != 1 else ''}",
    )


def start_sweep(name: str, project: ModelerProject, sweep_name: str) -> jobs.Job:
    """Validate the first design point of a sweep and launch every point."""
    spec = next((s for s in project.sweeps if s.name == sweep_name), None)
    if spec is None:
        raise NotFound(f"There is no sweep named '{sweep_name}' in this project.")
    try:
        points = em_sweep.expand(spec)
    except em_sweep.SweepError as exc:
        raise InvalidInput(str(exc)) from exc

    first = resolve(project, points[0])
    if first.errors:
        raise ValidationFailed("The first design point could not be resolved.",
                               expression_issues(first.errors))
    from EMShrimulator.core.em import validate as em_validate

    report = em_validate.validate(first.project)
    if not report["can_run"]:
        raise ValidationFailed(
            "The first design point has validation errors, so the sweep would fail at every point.",
            [i for i in report["issues"] if i["severity"] == "error"],
        )

    pdir = storage.project_dir(name)
    cmd = workers.command("sweep", str(pdir / "project.json"), str(pdir / "results"), spec.name)
    return jobs.start_job(
        "modeler", name, cmd, cwd=workers.working_directory(), log_dir=pdir / "jobs",
        env=workers.environment(),
        label=f"Sweep '{spec.name}' · {len(points)} point{'s' if len(points) != 1 else ''}",
    )


# --- results ---

def epsilon_eff(project: ModelerProject) -> float:
    """The medium TDR distances are computed in: the highest dielectric εr."""
    values = []
    for m in project.materials:
        if m.kind == MaterialKind.DIELECTRIC:
            try:
                values.append(float(m.epsilon))
            except (TypeError, ValueError):
                continue
    return max(values, default=1.0)


def results(name: str, project: ModelerProject) -> Dict:
    """Every single-run view derived from the newest Touchstone, plus sweeps."""
    from EMShrimulator.core.em import results as em_results

    rdir = results_dir(name)
    touchstone = em_results.find_touchstone(rdir)
    summary_path = rdir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    sweeps = em_sweep.list_sweeps(rdir)

    dumps = nearfield_dumps(name)
    if touchstone is None:
        return {"available": False, "summary": summary, "sweeps": sweeps,
                "files": em_results.list_touchstones(rdir), "nearfield": dumps}

    data = em_results.read_results(touchstone, epsilon_eff=epsilon_eff(resolve(project).project))
    data.update(available=True, summary=summary, sweeps=sweeps, path=str(touchstone),
                files=em_results.list_touchstones(rdir), nearfield=dumps)
    return data


def sweep_results(name: str, project: ModelerProject, slug: str) -> Dict:
    """Every finished point of one sweep, for overlaying on one chart."""
    from EMShrimulator.core.em import results as em_results

    root = results_dir(name) / "sweeps" / em_sweep.slug(slug)
    index_path = root / "index.json"
    if not index_path.exists():
        raise NotFound(f"There are no sweep results named '{slug}'.")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    epsilon = epsilon_eff(resolve(project).project)
    points: List[Dict[str, Any]] = []

    for entry in index.get("points", []):
        point = {
            "index": entry.get("index"),
            "label": entry.get("label", ""),
            "values": entry.get("values", {}),
            "status": entry.get("status", "pending"),
            "error": entry.get("error"),
            "elapsed_s": entry.get("elapsed_s"),
            "freq_hz": [],
            "traces": {},
            "marker": None,
        }
        touchstone_name = entry.get("touchstone")
        if touchstone_name:
            path = root / f"point_{int(entry['index']):03d}" / touchstone_name
            if path.exists():
                try:
                    data = em_results.read_results(path, epsilon_eff=epsilon)
                except Exception as exc:  # noqa: BLE001 - one unreadable point is not fatal
                    point["error"] = f"could not read {touchstone_name}: {exc}"
                else:
                    point["freq_hz"] = data.get("freq_hz", [])
                    point["traces"] = {
                        key: {k: trace.get(k) for k in ("mag_db", "phase_deg", "real", "imag", "vswr")}
                        for key, trace in (data.get("traces") or {}).items()
                    }
                    point["marker"] = (data.get("markers") or {}).get("resonance")
        points.append(point)

    return {
        "sweep": index.get("sweep", slug),
        "slug": em_sweep.slug(slug),
        "variables": index.get("variables", []),
        "combine": index.get("combine", "product"),
        "points": points,
    }


def farfield(name: str) -> Optional[Dict]:
    """farfield.json, with E_θ/E_φ arrays under "fields" (None for older runs)."""
    from EMShrimulator.core.em import farfield as em_farfield
    from EMShrimulator.core.em import results as em_results

    payload = em_results.read_farfield(results_dir(name) / "farfield.json")
    if payload is not None:
        payload["fields"] = em_farfield.read_fields(results_dir(name))
    return payload


def nearfield_dumps(name: str) -> List[Dict]:
    """The frequency-domain dumps the newest run left: [{dump, port}]."""
    from EMShrimulator.core.em import nearfield as em_nearfield

    return em_nearfield.available(results_dir(name))


def nearfield(name: str, dump: str, port: int) -> Optional[Dict]:
    """One field dump of one run."""
    from EMShrimulator.core.em import nearfield as em_nearfield

    return em_nearfield.load(results_dir(name), dump, port)
