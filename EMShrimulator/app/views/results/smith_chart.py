"""The Smith chart: reflection coefficients on the normalised impedance grid."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import numpy as np
from PySide6.QtCore import Qt

from EMShrimulator.app import theme
from EMShrimulator.app.views.results.chart import Chart, eng

RESISTANCE_CIRCLES = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0)
REACTANCE_ARCS = (0.2, 0.5, 1.0, 2.0, 5.0)
LIMIT = 1.08


@lru_cache(maxsize=1)
def _grid() -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """Every grid curve as (x, y): the same for every chart, so built once."""
    curves = []
    t = np.linspace(0, 2 * np.pi, 361)
    for r in RESISTANCE_CIRCLES:
        # Constant-resistance circle: centre (r/(1+r), 0), radius 1/(1+r).
        centre, radius = r / (1 + r), 1 / (1 + r)
        curves.append((centre + radius * np.cos(t), radius * np.sin(t)))
    for x in REACTANCE_ARCS:
        # Constant-reactance arc, clipped to the unit circle.
        for sign in (1, -1):
            z = 1j * sign * x + np.concatenate([np.linspace(0, 50, 400), [1e6]])
            gamma = (z - 1) / (z + 1)
            curves.append((gamma.real, gamma.imag))
    curves.append((np.array([-LIMIT, LIMIT]), np.zeros(2)))
    return tuple(curves)


def _reflection_port(key: str) -> int:
    """The k of an "Skk" key (S11, S1010), or 0."""
    digits = key[1:]
    half = len(digits) // 2
    if key[:1] == "S" and digits.isdigit() and len(digits) % 2 == 0 and digits[:half] == digits[half:]:
        return int(digits[:half])
    return 0


def _readout(key: str, freqs: List[float], real: np.ndarray, imag: np.ndarray, z0: float) -> Callable[[int], str]:
    def text(i: int) -> str:
        gamma = complex(real[i], imag[i])
        z = z0 * (1 + gamma) / (1 - gamma) if abs(1 - gamma) > 1e-12 else complex(float("inf"))
        at = f" @ {eng(freqs[i], 'Hz')}" if i < len(freqs) else ""
        return f"{key}{at}  Γ {gamma.real:+.3f}{gamma.imag:+.3f}j  Z {z.real:.1f}{z.imag:+.1f}j Ω"
    return text


def plot_smith(chart: Chart, data: Dict, keys: List[str], color_of: Callable[[str], str]) -> None:
    chart.clear()
    chart.set_equal_aspect(True)
    chart.set_axes_visible(False)
    chart.set_limits(x=(-LIMIT, LIMIT), y=(-LIMIT, LIMIT))
    chart.set_legend("upper right")
    c = theme.colors()
    for x, y in _grid():
        chart.plot(x, y, c["grid_major"], width=0.6, guide=True)
    for r in (0.5, 1.0, 2.0):
        chart.text(r / (1 + r) - 1 / (1 + r), 0.02, f"{r:g}", color=c["text_faint"], size=7.0)
    traces = data.get("traces") or {}
    freqs = data.get("freq_hz") or []
    z0 = float(data.get("z0") or 50.0)
    z0_ports = data.get("z0_ports") or []
    for key in keys:
        trace = traces.get(key)
        if not trace or trace.get("real") is None:
            continue
        real = np.asarray(trace["real"], dtype=float)
        imag = np.asarray(trace["imag"], dtype=float)
        color = color_of(key)
        port = _reflection_port(key)
        z_ref = float(z0_ports[port - 1]) if port and port <= len(z0_ports) else z0
        chart.plot(real, imag, color, width=1.6, label=key, readout=_readout(key, freqs, real, imag, z_ref))
        chart.plot(real[:1], imag[:1], color, marker="o", size=4)
        chart.plot(real[-1:], imag[-1:], color, marker="s", size=4)
    if freqs:
        chart.text(-1.05, -1.05, f"● {freqs[0] / 1e9:.3g} GHz   ■ {freqs[-1] / 1e9:.3g} GHz",
                   color=c["text_faint"], align=Qt.AlignLeft | Qt.AlignBottom)
