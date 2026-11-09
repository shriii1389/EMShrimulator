"""Smooth rectilinear mesh-line generation between fixed lines."""
from __future__ import annotations

import math
from typing import Iterable, List


def _grow_half(span: float, start_res: float, ratio: float, max_res: float) -> List[float]:
    """Distances from 0, growing geometrically, stopping at span/2."""
    pts = [0.0]
    d = 0.0
    cell = min(start_res, max_res)
    half = span / 2
    while d + cell < half - 1e-9:
        d += cell
        pts.append(d)
        cell = min(cell * ratio, max_res)
    return pts


def fill_gap(a: float, b: float, max_res: float, ratio: float, start_res: float) -> List[float]:
    """Interior mesh points strictly between a and b (exclusive)."""
    span = b - a
    if span <= max_res * (1 + 1e-6):
        return []
    left = _grow_half(span, start_res, ratio, max_res)[1:]
    right = [span - x for x in _grow_half(span, start_res, ratio, max_res)[1:]]
    pts = sorted(set(round(p, 9) for p in left + right if 0 < p < span))
    return [a + p for p in pts]


def smooth_mesh_lines(
    fixed: Iterable[float],
    max_res: float,
    ratio: float = 1.4,
    start_res: float | None = None,
) -> List[float]:
    """Sorted lines through the fixed points, graded by `ratio` up to `max_res`."""
    fx = sorted(set(round(v, 9) for v in fixed))
    if len(fx) < 2:
        return fx
    start_res = start_res if start_res is not None else max_res
    lines = [fx[0]]
    for a, b in zip(fx[:-1], fx[1:]):
        if b - a < 1e-9:
            continue
        lines.extend(fill_gap(a, b, max_res, ratio, start_res))
        lines.append(b)
    return sorted(set(round(v, 9) for v in lines))


def wavelength_resolution(freq_hz: float, cells_per_wavelength: float, epsilon_r: float = 1.0) -> float:
    """Max cell size (mm) for `cells_per_wavelength` cells per wavelength at freq_hz in a medium of epsilon_r."""
    c0 = 299_792_458.0  # m/s
    wavelength_m = c0 / (freq_hz * math.sqrt(epsilon_r))
    return (wavelength_m * 1000.0) / cells_per_wavelength  # mm
