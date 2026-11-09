"""The rectangular views of one run: magnitude, phase, VSWR, impedance, delay, TDR."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt

from EMShrimulator.app import theme
from EMShrimulator.app.views.results.chart import Chart, eng

# view -> (trace field, y label, readout unit, only reflection terms)
XY_VIEWS: Dict[str, tuple] = {
    "magnitude": ("mag_db", "Magnitude (dB)", "dB", False),
    "phase": ("phase_deg", "Phase, unwrapped (°)", "°", False),
    "vswr": ("vswr", "VSWR", "", True),
    "delay": ("group_delay_ns", "Group delay (ns)", "ns", False),
}
MAGNITUDE_SPAN_DB = 120.0  # never-excited terms (~-300 dB) must not flatten the plot
_TOP_LEFT = Qt.AlignLeft | Qt.AlignTop
_CENTER = Qt.AlignCenter


def _readout(key: str, freqs: np.ndarray, values: np.ndarray, unit: str) -> Callable[[int], str]:
    return lambda i: f"{key}  {values[i]:.4g}{' ' + unit if unit else ''} @ {eng(float(freqs[i]), 'Hz')}"


def plot_frequency_view(chart: Chart, view: str, data: Dict, keys: List[str],
                        color_of: Callable[[str], str], log_x: bool = False) -> None:
    """Draw one frequency-domain view of the enabled traces."""
    chart.clear()
    chart.set_x_unit("Hz")
    chart.set_log_x(log_x)
    freqs = np.asarray(data.get("freq_hz", []), dtype=float)
    traces = data.get("traces") or {}

    if view == "impedance":
        palette = theme.TRACE_COLORS
        n = 0
        for key in keys:
            trace = traces.get(key, {})
            if trace.get("z_real") is None:
                continue
            name = trace.get("z_label", key)  # a pair's reflection is its differential or common impedance
            for part, values, dashed in (("Re", trace["z_real"], False), ("Im", trace["z_imag"], True)):
                values = np.asarray(values, dtype=float)
                chart.plot(freqs, values, palette[(2 * n + dashed) % len(palette)], width=1.4, dashed=dashed,
                           label=f"{part} {name}", readout=_readout(f"{part} {name}", freqs, values, "Ω"))
            n += 1
        chart.set_labels(x="Frequency", y="Input impedance (Ω)")
        return

    field, label, unit, _reflection_only = XY_VIEWS[view]
    plotted = []
    for key in keys:
        values = traces.get(key, {}).get(field)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        plotted.append(values)
        chart.plot(freqs, values, color_of(key), label=key, readout=_readout(key, freqs, values, unit))
    chart.set_labels(x="Frequency", y=label)
    if view == "vswr":
        chart.set_limits(y=(1.0, 10.0))
    elif view == "magnitude" and plotted:
        values = np.concatenate(plotted)
        values = values[np.isfinite(values)]
        if values.size:
            low, high = float(values.min()), float(values.max())
            pad = max((high - low) * 0.05, 0.5)
            chart.set_limits(y=(max(low - pad, high + pad - MAGNITUDE_SPAN_DB), high + pad))
    marker = (data.get("markers") or {}).get("resonance")
    if view == "magnitude" and marker and "S11" in keys:
        accent = theme.colors()["accent"]
        chart.vline(marker["frequency_hz"], accent, 0.8)
        chart.text(marker["frequency_hz"], marker["s11_db"], f"{marker['s11_db']:.1f} dB", color=accent,
                   offset=(6, 4), align=_TOP_LEFT)


def plot_tdr(chart: Chart, tdr: Optional[Dict]) -> None:
    chart.clear()
    c = theme.colors()
    if not tdr:
        chart.text(0.5, 0.5, "TDR needs at least a one-port sweep starting near DC.", color=c["text_dim"],
                   axes=True, align=_CENTER, size=9.0)
        return
    distance = np.asarray(tdr["distance_mm"], dtype=float)
    impedance = np.asarray(tdr["impedance_ohm"], dtype=float)
    chart.plot(distance, impedance, theme.trace_color(0), label="Z(x)",
               readout=lambda i: f"Z  {impedance[i]:.1f} Ω @ {distance[i]:.2f} mm")
    chart.set_labels(x="Distance (mm, one way)", y="Impedance (Ω)")
