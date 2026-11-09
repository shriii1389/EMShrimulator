"""The open region: every face absorbing, λ/4 of air at the lowest operating frequency (or sweep start).

Axes a guided port launches along keep their manual margin. No openEMS imports.
"""
from __future__ import annotations

import math
from typing import Optional, Set, Tuple

from EMShrimulator.core.em import sheets
from EMShrimulator.core.em.model import BOUNDARY_FACES, ModelerProject, Port, PortKind

C0 = 299_792_458.0
ABSORBING = "PML_8"
NEAR_DC = 0.1  # fraction of the stop frequency below which a sweep starts near DC


def _f(value, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def guided_axis(project: ModelerProject, port: Port) -> Optional[str]:
    """The axis a guided port launches along, as csx_builder._add_port decides it."""
    sheet = sheets.sheet_by_name(project.primitives, port.sheet)
    if sheet is not None:
        return sheet.normal_axis or "z"
    if port.kind == PortKind.MICROSTRIP:
        return port.prop_dir or "x"
    return port.prop_dir or port.direction


def guided_axes(project: ModelerProject) -> Set[str]:
    return {axis for axis in (guided_axis(project, p) for p in project.ports if p.kind in sheets.GUIDED_KINDS)
            if axis}


def reference_frequency(project: ModelerProject) -> Tuple[float, str]:
    """The frequency the air is sized at, and why; (0, "") without a sweep."""
    start, stop = _f(project.frequency.start), _f(project.frequency.stop)
    if stop <= 0:
        return 0.0, ""
    operating = [f for f in (_f(v) for v in project.nf2ff.frequencies) if f > 0] if project.nf2ff.enabled else []
    if operating:
        return min(operating), "the lowest operating frequency"
    if start >= stop * NEAR_DC:
        return start, "the sweep's lowest frequency"
    return (max(start, 0.0) + stop) / 2, "the sweep's centre, as the sweep starts near DC"


def _round_up(mm: float) -> float:
    """Up to three significant figures: 10.71 -> 10.8."""
    if mm <= 0:
        return 0.0
    step = 10 ** (math.floor(math.log10(mm)) - 2)
    return float(f"{math.ceil(mm / step - 1e-9) * step:.6g}")


def quarter_wave_mm(project: ModelerProject) -> float:
    frequency, _why = reference_frequency(project)
    return _round_up(C0 / frequency * 1000 / 4) if frequency > 0 else 0.0


def margins(project: ModelerProject) -> Tuple[float, float, float]:
    """The air on each axis around the design."""
    quarter = quarter_wave_mm(project)
    kept = guided_axes(project)
    own = [_f(v) for v in project.mesh.air_margin]
    return tuple(own[i] if axis in kept or quarter <= 0 else quarter  # type: ignore[return-value]
                 for i, axis in enumerate("xyz"))


def apply(resolved: ModelerProject) -> None:
    """Open the region of a resolved project in place."""
    if not resolved.boundary.open_region:
        return
    resolved.mesh.air_margin = margins(resolved)
    for face in BOUNDARY_FACES:
        setattr(resolved.boundary, face, ABSORBING)


def summary(resolved: ModelerProject) -> dict:
    """The open region's sizing for the setup page: frequency, reason, λ/4 (mm) and guided axes' margins."""
    frequency, why = reference_frequency(resolved)
    own = dict(zip("xyz", (_f(v) for v in resolved.mesh.air_margin)))
    return {"frequency": frequency, "why": why, "quarter_mm": quarter_wave_mm(resolved),
            "kept": {axis: own[axis] for axis in sorted(guided_axes(resolved))}}
