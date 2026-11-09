"""Drawing reports (core.em.reports) on the QPainter chart: rectangular, polar, Smith and contour."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np
from PySide6.QtCore import Qt

from EMShrimulator.app import theme
from EMShrimulator.app.views.results.chart import Chart, eng, nice_ticks
from EMShrimulator.core.em.reports import Report, Series

_CIRCLE = np.linspace(0, 2 * np.pi, 241)
ColorOf = Callable[[str], str]


def _value(value: float, unit: str) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.4g}{' ' + unit if unit else ''}"


def _x_text(report: Report, x: float) -> str:
    if report.x_unit == "Hz":
        return eng(x, "Hz")
    label = report.x_label.split(" (")[0]
    unit = "°" if "°" in report.x_label else (" mm" if "mm" in report.x_label else (" ns" if "ns" in report.x_label
                                                                                    else ""))
    return f"{label} {x:.4g}{unit}"


def _readout(report: Report, s: Series) -> Callable[[int], str]:
    unit = s.unit if s.unit or s.category else report.y_unit
    return lambda i: f"{s.label}  {_value(float(s.y[i]), unit)} @ {_x_text(report, float(s.x[i]))}"


def message(chart: Chart, text: str) -> None:
    chart.clear()
    chart.set_axes_visible(False)
    chart.set_legend(None)
    chart.text(0.5, 0.5, text, color=theme.colors()["text_dim"], axes=True, align=Qt.AlignCenter, size=9.0)


def rectangular(chart: Chart, report: Report, color_of: ColorOf, log_x: bool = False) -> None:
    """Traces against x, a second unit on the right axis; few-point traces get markers."""
    chart.clear()
    chart.set_x_unit(report.x_unit)
    chart.set_y_unit(report.y_unit)
    chart.set_log_x(log_x and report.x_unit == "Hz")
    for s in report.series:
        axis = "y2" if s.axis == "y2" else "y1"
        color = color_of(s.key)
        chart.plot(s.x, s.y, color, width=1.5, dashed=s.dashed, label=s.label, readout=_readout(report, s), axis=axis)
        if s.markers:
            chart.plot(s.x, s.y, color, marker="o", size=6, axis=axis)
    chart.set_labels(x=report.x_label, y=report.y_label)
    chart.set_y2(report.y2_label, report.y2_unit)
    if report.y_limits is not None:
        chart.set_limits(y=report.y_limits)


def smith(chart: Chart, report: Report, color_of: ColorOf, freqs) -> str:
    """Every complex trace on one Smith chart (Z and Y as their reflection). Returns what could not be drawn."""
    from EMShrimulator.app.views.results.smith_chart import LIMIT, _grid

    c = theme.colors()
    chart.clear()
    chart.set_equal_aspect(True)
    chart.set_axes_visible(False)
    chart.set_limits(x=(-LIMIT, LIMIT), y=(-LIMIT, LIMIT))
    chart.set_legend("upper right")
    for x, y in _grid():
        chart.plot(x, y, c["grid_major"], width=0.6, guide=True)
    for r in (0.5, 1.0, 2.0):
        chart.text(r / (1 + r) - 1 / (1 + r), 0.02, f"{r:g}", color=c["text_faint"], size=7.0)
    freqs = list(freqs or [])
    seen, left_out = set(), []
    for s in report.series:
        if s.z is None:
            left_out.append(s.label)
            continue
        if s.source in seen:
            continue  # Re and Im of one term are one curve here
        seen.add(s.source)
        category, key = (s.source.split(":", 1) + [""])[:2]
        z0 = float(s.ref_ohm or 50.0)
        if category == "s":
            gamma, name = s.z, key
        else:
            impedance = 1 / s.z if category == "y" else s.z
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma = (impedance - z0) / (impedance + z0)
            name = f"{key} (Γ against {z0:g} Ω)"

        def readout(i, gamma=gamma, name=name, z0=z0):
            g = complex(gamma[i])
            at = f" @ {eng(freqs[i], 'Hz')}" if i < len(freqs) else ""
            z = z0 * (1 + g) / (1 - g) if abs(1 - g) > 1e-12 else complex(float("inf"))
            return f"{name}{at}  Γ {g.real:+.3f}{g.imag:+.3f}j  Z {z.real:.1f}{z.imag:+.1f}j Ω"

        color = color_of(s.key)
        chart.plot(gamma.real, gamma.imag, color, width=1.6, label=name, readout=readout)
        chart.plot(gamma.real[:1], gamma.imag[:1], color, marker="o", size=4)
        chart.plot(gamma.real[-1:], gamma.imag[-1:], color, marker="s", size=4)
    if freqs:
        chart.text(-1.05, -1.05, f"● {freqs[0] / 1e9:.3g} GHz   ■ {freqs[-1] / 1e9:.3g} GHz",
                   color=c["text_faint"], align=Qt.AlignLeft | Qt.AlignBottom)
    return f"Not on the Smith chart (not a complex quantity): {', '.join(left_out)}." if left_out else ""


def _polar_frame(chart: Chart, title: str = "") -> None:
    chart.clear()
    chart.set_equal_aspect(True)
    chart.set_axes_visible(False)
    chart.set_limits(x=(-1.2, 1.2), y=(-1.2, 1.2))
    chart.set_labels(title=title)
    chart.set_legend("lower left")


def complex_polar(chart: Chart, report: Report, color_of: ColorOf, freqs) -> str:
    """S, Y or Z terms in the complex plane, one kind per chart. Returns what was left out."""
    c = theme.colors()
    series, seen, left_out = [], set(), []
    family = next((s.source.split(":")[0] for s in report.series if s.z is not None), "")
    for s in report.series:
        if s.z is None or s.source.split(":")[0] != family:
            left_out.append(s.label)
        elif s.source not in seen:
            seen.add(s.source)
            series.append(s)
    peak = max((float(np.nanmax(np.abs(s.z))) for s in series if np.isfinite(s.z).any()), default=1.0)
    _ticks, step = nice_ticks(0.0, peak if peak > 0 else 1.0, 4)
    outer = math.ceil(peak / step - 1e-9) * step if step > 0 and peak > 0 else 1.0
    rings = np.arange(step, outer + step / 2, step)
    _polar_frame(chart)
    grid = c["grid_minor"]
    for r in rings:
        if r <= 0:
            continue
        rn = r / outer
        chart.plot(rn * np.cos(_CIRCLE), rn * np.sin(_CIRCLE), grid, guide=True)
        chart.text(rn * math.cos(math.radians(80)), rn * math.sin(math.radians(80)), f"{r:g}",
                   color=c["text_faint"], size=7.0)
    chart.plot(np.cos(_CIRCLE), np.sin(_CIRCLE), c["line"], guide=True)
    for degrees in range(0, 360, 30):
        a = math.radians(degrees)
        chart.plot([0, math.cos(a)], [0, math.sin(a)], grid, guide=True)
        label = degrees if degrees <= 180 else degrees - 360
        chart.text(1.1 * math.cos(a), 1.1 * math.sin(a), f"{label}°", color=c["text_dim"], size=7.0,
                   align=Qt.AlignCenter)
    unit = {"z": " Ω", "y": " S"}.get(family, "")
    freqs = list(freqs or [])
    for s in series:
        z = s.z / outer
        color = color_of(s.key)
        name = s.source.split(":", 1)[-1] or s.label

        def readout(i, s=s, name=name):
            at = f" @ {eng(freqs[i], 'Hz')}" if i < len(freqs) else ""
            v = s.z[i]
            return f"{name}{at}  |{abs(v):.4g}|{unit} ∠ {math.degrees(np.angle(v)):.1f}°  ({v.real:.4g}{v.imag:+.4g}j)"

        chart.plot(z.real, z.imag, color, width=1.6, label=name, readout=readout)
        chart.plot(z.real[:1], z.imag[:1], color, marker="o", size=4)
        chart.plot(z.real[-1:], z.imag[-1:], color, marker="s", size=4)
    if freqs:
        chart.text(-1.15, -1.15, f"● {freqs[0] / 1e9:.3g} GHz   ■ {freqs[-1] / 1e9:.3g} GHz   outer ring {outer:g}{unit}",
                   color=c["text_faint"], align=Qt.AlignLeft | Qt.AlignBottom)
    return (f"Polar shows one kind of term at a time; left out: {', '.join(left_out)}. The Smith chart shows S "
            "and Z together." if left_out else "")


def pattern_polar(chart: Chart, report: Report, floor: float, peak: float, color_of: ColorOf) -> str:
    """Pattern cuts on polar axes, 0° at the top, clockwise, one unit per chart. Returns what was left out."""
    c = theme.colors()
    shown = [s for s in report.series if s.axis != "y2"]
    left_out = list(dict.fromkeys(s.label for s in report.series if s.axis == "y2"))
    _polar_frame(chart, report.y_label)
    db = report.y_unit.startswith("dB")
    top = peak + (1.0 if db else 0.0)
    span = max(top - floor, 1e-12)

    def radius(values):
        return (np.clip(np.asarray(values, dtype=float), floor, None) - floor) / span

    grid = c["grid_minor"]
    rings, _step = nice_ticks(floor, top, 5)
    for r in rings:
        rn = (r - floor) / span
        if rn <= 0:
            continue
        chart.plot(rn * np.sin(_CIRCLE), rn * np.cos(_CIRCLE), grid, guide=True)
        angle = math.radians(22.5)
        chart.text(rn * math.sin(angle), rn * math.cos(angle), f"{r:g}", color=c["text_faint"], size=7.0)
    chart.plot(np.sin(_CIRCLE), np.cos(_CIRCLE), c["line"], guide=True)
    sweep_phi = report.x_label.startswith("φ")
    for degrees in range(0, 360, 30):
        a = math.radians(degrees)
        chart.plot([0, math.sin(a)], [0, math.cos(a)], grid, guide=True)
        label = degrees if sweep_phi or degrees <= 180 else degrees - 360
        chart.text(1.1 * math.sin(a), 1.1 * math.cos(a), f"{label}°", color=c["text_dim"], size=7.0,
                   align=Qt.AlignCenter)
    for s in shown:
        a = np.radians(s.x)
        rn = radius(s.y)
        chart.plot(rn * np.sin(a), rn * np.cos(a), color_of(s.key), width=1.6, dashed=s.dashed, label=s.label,
                   readout=_readout(report, s))
    return (f"A polar pattern has one radial scale: {', '.join(left_out)} (another unit) is on the rectangular "
            "display." if left_out else "")


def _resample(u: np.ndarray, v: np.ndarray, values: np.ndarray, size: int = 320):
    """Resample a graded mesh's plane onto an even grid."""
    uu = np.linspace(float(u[0]), float(u[-1]), size) if u.size > 1 else u
    vv = np.linspace(float(v[0]), float(v[-1]), size) if v.size > 1 else v
    rows = np.array([np.interp(uu, u, row) for row in values]) if u.size > 1 else values
    out = np.array([np.interp(vv, v, col) for col in rows.T]).T if v.size > 1 else rows
    return uu, vv, out


def contour_pattern(chart: Chart, theta, phi, grid: np.ndarray, floor: float, peak: float, label: str,
                    unit: str) -> None:
    """θ–φ map: φ across, θ up."""
    chart.clear()
    theta, phi = np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
    values = np.clip(np.asarray(grid, dtype=float), floor, None)
    chart.image(values, (float(phi[0]), float(phi[-1]), float(theta[0]), float(theta[-1])), vmin=floor, vmax=peak,
                label=label, readout=lambda x, y, v: f"φ {x:.0f}°  θ {y:.0f}°  {_value(v, unit)}")
    chart.set_labels(x="φ (°)", y="θ (°)", title=label)
    chart.set_legend(None)


def contour_plane(chart: Chart, cut: Dict, label: str) -> None:
    """A near-field plane, in mm, to scale."""
    chart.clear()
    u, v, values = _resample(np.asarray(cut["u_mm"]), np.asarray(cut["v_mm"]), np.asarray(cut["values"]))
    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if finite.size else 1.0
    vmin = float(finite.min()) if finite.size else 0.0
    if cut["unit"].startswith("dB"):
        vmin = max(vmin, vmax - 60.0)
    unit = cut["unit"]
    chart.set_equal_aspect(True)
    chart.image(np.clip(values, vmin, None), (float(u[0]), float(u[-1]), float(v[0]), float(v[-1])), vmin=vmin,
                vmax=vmax, label=label,
                readout=lambda x, y, value: f"{cut['u']} {x:.3g} mm  {cut['v']} {y:.3g} mm  {_value(value, unit)}")
    chart.set_labels(x=f"{cut['u']} (mm)", y=f"{cut['v']} (mm)",
                     title=f"{label} · {cut['held']} = {cut['held_mm']:.4g} mm")
    chart.set_legend(None)


def color_map(report_keys, stable: Optional[list] = None) -> ColorOf:
    """Colours by key, stable per trace."""
    order: Dict[str, int] = {}
    for key in report_keys:
        order.setdefault(key, len(order))
    stable = stable or []

    def color_of(key: str) -> str:
        if key in stable:
            return theme.trace_color(stable.index(key))
        if key[:1] == "c" and key[1:].isdigit():  # a report's n-th trace: the colour its list entry shows
            return theme.trace_color(int(key[1:]))
        return theme.trace_color(order.get(key, 0))

    return color_of
