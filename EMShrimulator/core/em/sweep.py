"""Expand a ParameterSweep into parameter overrides and index results under results/sweeps/<slug>/."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from EMShrimulator.core.em.model import ParameterSweep, SweepCombine, SweepMode, SweepVariable

MAX_POINTS = 256


class SweepError(ValueError):
    """A sweep definition that cannot be expanded into design points."""


def variable_values(variable: SweepVariable) -> List[float]:
    """The values one swept variable takes, in order."""
    mode = variable.mode
    if mode == SweepMode.LIST:
        if not variable.values:
            raise SweepError(f"'{variable.parameter}' is set to a value list but the list is empty")
        return [float(v) for v in variable.values]

    if variable.points < 1:
        raise SweepError(f"'{variable.parameter}' needs at least one point")
    if variable.points == 1:
        return [float(variable.start)]

    if mode == SweepMode.LOG:
        if variable.start <= 0 or variable.stop <= 0:
            raise SweepError(
                f"'{variable.parameter}' is swept logarithmically, so start and stop must both be positive"
            )
        lo = math.log10(variable.start)
        hi = math.log10(variable.stop)
        step = (hi - lo) / (variable.points - 1)
        return [10 ** (lo + step * i) for i in range(variable.points)]

    step = (variable.stop - variable.start) / (variable.points - 1)
    return [variable.start + step * i for i in range(variable.points)]


def expand(sweep: ParameterSweep) -> List[Dict[str, float]]:
    """Every design point in the sweep, as parameter overrides."""
    if not sweep.variables:
        raise SweepError("the sweep has no variables")

    columns = [variable_values(v) for v in sweep.variables]
    names = [v.parameter for v in sweep.variables]

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SweepError(f"parameter(s) swept more than once in the same sweep: {', '.join(duplicates)}")

    if sweep.combine == SweepCombine.ZIP:
        lengths = {len(c) for c in columns}
        if len(lengths) > 1:
            raise SweepError(
                "a zipped sweep walks its variables together, so they must all have the same "
                f"number of points; got {', '.join(f'{n}={len(c)}' for n, c in zip(names, columns))}"
            )
        points = [dict(zip(names, row)) for row in zip(*columns)]
    else:
        points = [{}]
        for name, column in zip(names, columns):
            points = [{**base, name: value} for base in points for value in column]

    if len(points) > MAX_POINTS:
        raise SweepError(
            f"this sweep expands to {len(points)} design points, and each one is a full FDTD solve. "
            f"The limit is {MAX_POINTS} - reduce the point counts, or switch the combination to "
            "'together' so the variables move as one."
        )
    return points


def slug(name: str) -> str:
    """A filesystem-safe directory name for a sweep."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return cleaned or "sweep"


def point_label(point: Dict[str, float]) -> str:
    """A short human label for one design point, used on the chart legend."""
    return ", ".join(f"{name}={_fmt(value)}" for name, value in point.items())


def _fmt(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.6g}"


def sweep_dir(results_dir: Path, sweep_name: str) -> Path:
    return results_dir / "sweeps" / slug(sweep_name)


def point_dir(results_dir: Path, sweep_name: str, index: int) -> Path:
    return sweep_dir(results_dir, sweep_name) / f"point_{index:03d}"


def write_index(results_dir: Path, sweep: ParameterSweep, entries: List[Dict[str, Any]]) -> Path:
    """The manifest the Results tab reads to overlay a sweep's points."""
    target = sweep_dir(results_dir, sweep.name)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "index.json"
    path.write_text(json.dumps({
        "sweep": sweep.name,
        "description": sweep.description,
        "combine": sweep.combine.value if isinstance(sweep.combine, SweepCombine) else str(sweep.combine),
        "variables": [v.model_dump(mode="json") for v in sweep.variables],
        "points": entries,
    }, indent=2), encoding="utf-8")
    return path


def read_index(results_dir: Path, sweep_name: str) -> Dict[str, Any] | None:
    path = sweep_dir(results_dir, sweep_name) / "index.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_sweeps(results_dir: Path) -> List[Dict[str, Any]]:
    """Every sweep that has results on disk, newest-looking first."""
    root = results_dir / "sweeps"
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        index_path = child / "index.json"
        if not index_path.exists():
            continue
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        done = sum(1 for p in data.get("points", []) if p.get("touchstone"))
        out.append({
            "slug": child.name,
            "name": data.get("sweep", child.name),
            "points": len(data.get("points", [])),
            "completed": done,
            "variables": [v.get("parameter") for v in data.get("variables", [])],
        })
    return out


def describe(sweep: ParameterSweep) -> Dict[str, Any]:
    """A dry-run summary: the points this sweep would solve, or why it cannot."""
    try:
        points = expand(sweep)
    except SweepError as exc:
        return {"ok": False, "error": str(exc), "points": []}
    return {
        "ok": True,
        "error": None,
        "count": len(points),
        "points": [{"index": i, "values": p, "label": point_label(p)} for i, p in enumerate(points)],
    }


def iter_points(sweep: ParameterSweep) -> Iterable[tuple[int, Dict[str, float]]]:
    return enumerate(expand(sweep))
