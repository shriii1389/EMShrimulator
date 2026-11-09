"""Every report's numbers from one run's results, for the views in app/views/results to draw. numpy only.

A report is a list of traces (model.ReportTrace): category, quantity and function.
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from EMShrimulator.core.em import farfield as ff
from EMShrimulator.core.em import nearfield as nf
from EMShrimulator.core.em.model import ReportTrace
from EMShrimulator.core.em.results import fastest_rise_s, tdr_of

# S21, or a mixed-mode term like Sdd21 or Scd11.
_SIJ = re.compile(r"^S(?:(dd|cc|dc|cd))?(\d+)(\d+)$")
FUNCTIONS: Dict[str, str] = {"db": "dB", "mag": "Magnitude", "phase": "Phase", "re": "Real", "im": "Imaginary",
                             "re_im": "Real and imaginary"}
TRACE_FUNCTIONS: Dict[str, str] = {"db": "dB", "mag": "Magnitude", "phase": "Phase (°)", "re": "Real",
                                   "im": "Imaginary", "lin": "Linear", "value": "Value"}
CATEGORIES: Dict[str, str] = {
    "s": "S-parameter", "y": "Y-parameter", "z": "Z-parameter", "vswr": "VSWR", "group_delay": "Group delay",
    "gain": "Gain", "realized_gain": "Realized gain", "directivity": "Directivity", "re": "rE",
    "axial_ratio": "Axial ratio", "antenna": "Antenna parameter", "tdr": "TDR impedance",
}
NETWORK = ("s", "y", "z", "vswr", "group_delay")
COMPLEX = ("s", "y", "z")
FARFIELD = ("gain", "realized_gain", "directivity", "re", "axial_ratio")
FREQUENCY_KINDS = ("s", "y", "z", "vswr", "group_delay", "antenna")
MAGNITUDE_SPAN_DB = 120.0  # never-excited terms (~-300 dB) must not flatten a plot
ANTENNA_KEYS = ("peak_directivity", "peak_gain", "peak_realized_gain", "efficiency", "axial_ratio",
                "front_to_back", "radiated_power", "accepted_power", "incident_power", "hpbw_e", "hpbw_h",
                "peak_theta", "peak_phi")


def natural(key: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", key)]


def default_enabled(key: str) -> bool:
    """Default S terms: reflections, first-column transmissions and differential mixed-mode terms."""
    match = _SIJ.match(key)
    if not match:
        return True
    modes, out, into = match.groups()
    if modes not in (None, "dd"):
        return False
    return out == into or into == "1"


def is_reflection(key: str) -> bool:
    """S11, Sdd11, Scc11: a port's or mode's own reflection."""
    match = _SIJ.match(key)
    return bool(match) and match.group(2) == match.group(3) and match.group(1) in (None, "dd", "cc")


def diagonal(key: str) -> bool:
    """Z11, Y1010, Zin1: one port's own term."""
    if key[1:3] == "in":
        return True
    digits = key[1:]
    half = len(digits) // 2
    return digits.isdigit() and len(digits) % 2 == 0 and digits[:half] == digits[half:]


@dataclass
class Series:
    key: str                              # what it is drawn for; its colour follows it
    label: str
    x: np.ndarray
    y: np.ndarray
    dashed: bool = False
    z: Optional[np.ndarray] = None        # complex values, for polar and Smith displays
    ref_ohm: Optional[float] = None       # a reflection's reference impedance
    unit: str = ""
    axis: str = "y1"                      # "auto" until the report places it
    quantity: str = ""                    # what the axis label calls it
    category: str = ""
    markers: bool = False                 # a few points (the far-field frequencies), not a sweep
    source: str = ""                      # the complex term behind it ("z:Z11"): Re and Im share one


@dataclass
class Report:
    series: List[Series] = field(default_factory=list)
    x_label: str = ""
    x_unit: str = ""                      # "Hz" gives engineering ticks
    y_label: str = ""
    y_unit: str = ""
    message: str = ""                     # why there is nothing to draw
    note: str = ""                        # said beside what is drawn
    y_limits: Optional[Tuple[float, float]] = None
    y2_label: str = ""
    y2_unit: str = ""


# ------------------------------------------------------------------ what can be added

def functions_for(category: str) -> List[str]:
    if category in COMPLEX:
        return ["db", "mag", "phase", "re", "im"]
    if category in FARFIELD and category != "axial_ratio":
        return ["db", "lin"]
    return ["value"]


def categories_for(kind: str, data: Dict, farfield: Optional[Dict]) -> List[str]:
    """The categories Add trace offers a report of `kind`, given what the run computed."""
    has_far = bool((farfield or {}).get("frequencies"))
    if kind == "tdr":
        return ["tdr"]
    if kind == "farfield":
        return list(FARFIELD)
    if kind == "nearfield":
        return []
    out = list(NETWORK) if data.get("available") else []
    if has_far:
        out += list(FARFIELD) + ["antenna"]
    return out


def terms_for(category: str, data: Dict, farfield: Optional[Dict] = None) -> List[Tuple[str, str]]:
    """(term, label) a trace of `category` can take from this run."""
    if category == "antenna":
        labels = {key: row["label"] for key, row in ff.antenna_parameters([], None).items()}
        return [(key, labels[key]) for key in ANTENNA_KEYS]
    if category == "axial_ratio":
        return [("total", "—")]
    if category in FARFIELD:
        return list(ff.POLARISATIONS.items())
    return [(key, key) for key in trace_keys(category, data, farfield)]


def trace_keys(kind: str, data: Dict, farfield: Optional[Dict] = None) -> List[str]:
    """The S/Y/Z terms (or reflections, antenna parameters) a report of `kind` can show."""
    if kind == "antenna":
        return list(ANTENNA_KEYS)
    source = {"y": "y_traces", "z": "z_traces"}.get(kind, "traces")
    keys = sorted((data.get(source) or {}).keys(), key=natural)
    if kind in ("vswr", "tdr"):
        traces = data.get("traces") or {}
        return [k for k in keys if is_reflection(k) and traces[k].get("ref_ohm" if kind == "tdr" else "vswr")
                is not None]
    return keys if kind in ("s", "y", "z", "group_delay") else []


def default_traces(kind: str, keys: List[str]) -> List[str]:
    """The terms a new report shows before anyone picks."""
    if kind in ("y", "z"):
        return [k for k in keys if diagonal(k)] or keys[:1]
    if kind == "vswr":
        return keys
    if kind == "tdr":
        return keys[:1]
    if kind == "antenna":
        return ["peak_directivity", "peak_gain", "peak_realized_gain"]
    mixed = any((m := _SIJ.match(k)) and m.group(1) for k in keys)
    # With differential pairs, start with the single-ended terms off.
    return [k for k in keys if default_enabled(k) and not (mixed and (m := _SIJ.match(k)) and not m.group(1))]


def default_curves(plot, data: Dict, farfield: Optional[Dict] = None) -> List[ReportTrace]:
    """The traces of a report nobody has added any to, from its kind and older settings."""
    kind = plot.kind
    if kind == "farfield":
        function = "value" if plot.quantity == "axial_ratio" else ("db" if plot.db else "lin")
        return [ReportTrace(category=plot.quantity, term=plot.component, function=function)]
    keys = trace_keys(kind, data, farfield)
    chosen = [k for k in plot.traces if k in keys] if plot.traces else default_traces(kind, keys)
    if kind in COMPLEX:
        functions = ["re", "im"] if plot.function == "re_im" else [plot.function]
        return [ReportTrace(category=kind, term=k, function=f) for k in chosen for f in functions]
    if kind in ("vswr", "group_delay", "tdr", "antenna"):
        return [ReportTrace(category=kind, term=k, function="value") for k in chosen]
    return []


def curves_of(plot, data: Dict, farfield: Optional[Dict] = None) -> List[ReportTrace]:
    return list(plot.curves) if plot.curves else default_curves(plot, data, farfield)


# ------------------------------------------------------------------ names and units

def _short(category: str, term: str) -> str:
    if category in FARFIELD:
        if category == "axial_ratio":
            return "Axial ratio"
        pol = ff.POLARISATIONS.get(term, term)
        return f"{ff.QUANTITIES[category]} {pol}"
    if category == "antenna":
        return {key: row["label"] for key, row in ff.antenna_parameters([], None).items()}.get(term, term)
    if category == "vswr":
        return f"VSWR({term})"
    if category == "group_delay":
        return f"Group delay({term})"
    if category == "tdr":
        return f"TDR({term})"
    return term


def trace_label(trace: ReportTrace, at_direction: bool = False) -> str:
    """dB(S11), Re(Z11), ∠S21, dB(Gain RHCP) @ θ 0° φ 0°."""
    name = _short(trace.category, trace.term)
    function = trace.function
    if trace.category in COMPLEX:
        name = {"db": f"dB({name})", "mag": f"|{name}|", "phase": f"∠{name}", "re": f"Re({name})",
                "im": f"Im({name})"}[function]
    elif trace.category in FARFIELD and trace.category != "axial_ratio":
        name = f"dB({name})" if function == "db" else name
    if at_direction and trace.category in FARFIELD:
        name += f" @ θ {trace.theta_deg:g}° φ {trace.phi_deg:g}°"
    return name


def trace_unit(trace: ReportTrace) -> str:
    c, f = trace.category, trace.function
    if c in COMPLEX:
        if f == "phase":
            return "°"
        if f == "db":
            return {"s": "dB", "y": "dBS", "z": "dBΩ"}[c]
        return {"s": "", "y": "S", "z": "Ω"}[c]
    if c == "group_delay":
        return "ns"
    if c == "tdr":
        return "Ω"
    if c == "axial_ratio":
        return "dB"
    if c == "re":
        return "dBV" if f == "db" else "V"
    if c in FARFIELD:
        return "dBi" if f == "db" else ""
    if c == "antenna":
        return {key: row["unit"] for key, row in ff.antenna_parameters([], None).items()}.get(trace.term, "")
    return ""


def _quantity(trace: ReportTrace) -> str:
    """What an axis of this trace's unit is called."""
    c = trace.category
    if c in COMPLEX:
        if trace.function == "phase":
            return "Phase"
        return {"s": "S-parameter", "y": "Admittance", "z": "Impedance"}[c]
    if c in FARFIELD:
        return ff.QUANTITIES[c]
    return CATEGORIES.get(c, c)


def _place_axes(report: Report) -> None:
    """Assign auto traces to Y1 or Y2 by unit, then name both axes."""
    y1_unit = next((s.unit for s in report.series if s.axis != "y2"), "")
    for s in report.series:
        if s.axis == "auto":
            s.axis = "y1" if s.unit == y1_unit else "y2"
    for axis in ("y1", "y2"):
        on = [s for s in report.series if s.axis == axis]
        if not on:
            continue
        units = list(dict.fromkeys(s.unit for s in on))
        names = list(dict.fromkeys(s.quantity for s in on))
        unit = units[0] if len(units) == 1 else ""
        label = ", ".join(names[:3]) + (f" ({unit})" if unit else (f" ({', '.join(u for u in units if u)})"
                                                                     if any(units) else ""))
        if axis == "y1":
            report.y_label, report.y_unit = label, unit
        else:
            report.y2_label, report.y2_unit = label, unit


# ------------------------------------------------------------------ against frequency

def _complex_term(category: str, data: Dict, term: str) -> Tuple[Optional[np.ndarray], str, Optional[float]]:
    """(values, key used, reference impedance). Z11 falls back to Zin1."""
    source = {"y": "y_traces", "z": "z_traces"}.get(category, "traces")
    traces = data.get(source) or {}
    key = term
    if key not in traces and category in ("y", "z") and diagonal(key) and not key.startswith(f"{category.upper()}in"):
        digits = key[1:]
        key = f"{key[0]}in{digits[:len(digits) // 2]}"
    trace = traces.get(key)
    if not trace:
        return None, term, None
    real, imag = ("z_real", "z_imag") if category == "z" else ("real", "imag")
    if trace.get(real) is None:
        return None, term, None
    values = np.asarray(trace[real], dtype=float) + 1j * np.asarray(trace[imag], dtype=float)
    ref = trace.get("ref_ohm")
    if ref is None:
        ref = port_reference(data, key)
    return values, key, ref


def port_reference(data: Dict, key: str) -> Optional[float]:
    """The reference impedance of a diagonal term's port, for a Smith chart."""
    z0s = data.get("z0_ports") or []
    digits = key[3:] if key[1:3] == "in" else key[1:][: len(key[1:]) // 2]
    if digits.isdigit() and 0 < int(digits) <= len(z0s):
        return float(z0s[int(digits) - 1])
    return float(data["z0"]) if data.get("z0") is not None else None


def _apply(function: str, values: np.ndarray, unwrap: bool = True) -> np.ndarray:
    if function == "db":
        return 20 * np.log10(np.maximum(np.abs(values), 1e-15))
    if function == "mag":
        return np.abs(values)
    if function == "phase":
        angle = np.angle(values)
        if unwrap:
            finite = np.isfinite(angle)
            angle = angle.copy()
            angle[finite] = np.unwrap(angle[finite])
        return np.degrees(angle)
    if function == "im":
        return values.imag
    return values.real


class _FarCache:
    """Pattern grids, worked out once per frequency, quantity, polarisation and scale."""

    def __init__(self, farfield: Optional[Dict]):
        self.entries = (farfield or {}).get("frequencies") or []
        self.fields = (farfield or {}).get("fields")
        self._grids: Dict[Tuple, Tuple] = {}

    def grid(self, index: int, category: str, term: str, db: bool):
        key = (index, category, term, db)
        if key not in self._grids:
            self._grids[key] = ff.pattern(self.entries[index], self.fields, category, term, db)
        return self._grids[key]


def frequency_report(plot, data: Dict, farfield: Optional[Dict], curves: Optional[List[ReportTrace]] = None
                     ) -> Report:
    """Traces against frequency: network terms, far field at a direction and antenna parameters."""
    every = curves if curves is not None else curves_of(plot, data, farfield)
    curves = [c for c in every if c.visible]
    if every and not curves:
        return Report(message="Every trace is hidden: tick one in the list.")
    freqs = np.asarray(data.get("freq_hz") or [], dtype=float)
    traces = data.get("traces") or {}
    far = _FarCache(farfield)
    far_freqs = np.asarray([float(e["frequency_hz"]) for e in far.entries], dtype=float)
    table = ff.antenna_parameters(far.entries, far.fields) if far.entries else {}
    report = Report(x_label="Frequency", x_unit="Hz")
    missing, notes = [], []
    for i, curve in enumerate(curves):
        label, unit = trace_label(curve, at_direction=True), trace_unit(curve)
        series = None
        if curve.category in COMPLEX:
            values, key, ref = _complex_term(curve.category, data, curve.term)
            if values is not None:
                name = label if key == curve.term else label.replace(curve.term, key)
                series = Series(curve.term, name, freqs, _apply(curve.function, values), z=values, ref_ohm=ref,
                                source=f"{curve.category}:{key}")
        elif curve.category in ("vswr", "group_delay"):
            values = (traces.get(curve.term) or {}).get("vswr" if curve.category == "vswr" else "group_delay_ns")
            if values is not None:
                series = Series(curve.term, label, freqs, np.asarray(values, dtype=float))
        elif curve.category == "antenna" and curve.term in table:
            values = [np.nan if v is None else float(v) for v in table[curve.term]["values"]]
            series = Series(f"antenna:{curve.term}", label, far_freqs, np.asarray(values), markers=True)
        elif curve.category in FARFIELD and far.entries:
            values, why = [], ""
            for n, entry in enumerate(far.entries):
                grid, _u, why = far.grid(n, curve.category, curve.term, curve.function != "lin")
                if grid is None:
                    break
                ti = int(np.argmin(np.abs(np.asarray(entry["theta_deg"]) - curve.theta_deg)))
                pj = int(np.argmin(np.abs((np.asarray(entry["phi_deg"]) - curve.phi_deg + 180) % 360 - 180)))
                values.append(float(grid[ti, pj]))
            if len(values) == len(far.entries):
                series = Series(f"far:{i}", label, far_freqs, np.asarray(values), markers=True)
            elif why:
                notes.append(why)
        if series is None:
            missing.append(label)
            continue
        # One colour per trace, the same on every display of the report.
        series.key = f"c{i}"
        series.unit, series.axis, series.quantity, series.category = unit, curve.axis, _quantity(curve), \
            curve.category
        report.series.append(series)
    _place_axes(report)
    y1 = [s for s in report.series if s.axis == "y1"]
    if y1 and all(s.category == "s" and s.unit == "dB" for s in y1):
        values = np.concatenate([s.y[np.isfinite(s.y)] for s in y1])
        if values.size:
            low, high = float(values.min()), float(values.max())
            pad = max((high - low) * 0.05, 0.5)
            report.y_limits = (max(low - pad, high + pad - MAGNITUDE_SPAN_DB), high + pad)
    elif y1 and all(s.category == "vswr" for s in y1):
        report.y_limits = (1.0, 10.0)
    if missing:
        notes.append(f"Not in this run: {', '.join(missing)}.")
    if any(s.markers for s in report.series) and len(far_freqs) == 1:
        notes.append("One far-field frequency: add more under Solution setup › Far field to see these vary.")
    report.note = " ".join(notes)
    if not report.series:
        report.message = ("None of these traces is in this run." if curves else
                          "No traces: add one with Add trace.")
    return report


def solution(plot, data: Dict, keys: List[str]) -> Report:
    """Network terms of `keys`, drawn by the plot's function (reports without traces)."""
    kind = plot.kind
    if kind in COMPLEX:
        functions = ["re", "im"] if plot.function == "re_im" else [plot.function]
        curves = [ReportTrace(category=kind, term=k, function=f) for k in keys for f in functions]
    else:
        curves = [ReportTrace(category=kind, term=k, function="value") for k in keys]
    report = frequency_report(plot, data, None, curves)
    for s in report.series:
        s.dashed = s.label.startswith("Im(")
    return report


def antenna(plot, farfield: Optional[Dict], keys: List[str]) -> Tuple[Report, Dict[str, Dict]]:
    """Antenna parameters against frequency, and every parameter for a table."""
    entries = (farfield or {}).get("frequencies") or []
    if not entries:
        return Report(message="No far-field results. In Solution setup choose the Antenna solution type, add "
                              "operating frequencies, and run the design."), {}
    table = ff.antenna_parameters(entries, (farfield or {}).get("fields"))
    curves = [ReportTrace(category="antenna", term=k, function="value") for k in keys]
    report = frequency_report(plot, {}, farfield, curves) if curves else Report(x_label="Frequency", x_unit="Hz")
    notes = ["Powers are for 1 W incident at the port, as HFSS reports them."] if any(
        e.get("incident_power_w") for e in entries) else []
    report.note = " ".join(notes + ([report.note] if report.note else []))
    return report, table


# ------------------------------------------------------------------ TDR

def tdr(plot, data: Dict, keys: List[str]) -> Report:
    report = Report()
    rise = plot.rise_time_ps * 1e-12 if plot.rise_time_ps else None
    distance = plot.tdr_axis == "distance"
    notes = []
    for key in keys:
        result = tdr_of(data, key, rise)
        if result is None:
            continue
        x = np.asarray(result["distance_mm" if distance else "time_ns"], dtype=float)
        report.series.append(Series(key, f"Z(t) {key}" if not distance else f"Z {key}", x,
                                    np.asarray(result["impedance_ohm"], dtype=float), unit="Ω",
                                    quantity="Impedance", category="tdr"))
        if result.get("band_limited"):
            notes.append(key)
    freqs = data.get("freq_hz") or []
    if distance:
        report.x_label = f"Distance (mm, one way, εr {float(data.get('epsilon_eff') or 1.0):g})"
    else:
        report.x_label = "Time (ns, round trip)"
    report.y_label, report.y_unit = "Impedance (Ω)", "Ω"
    fastest = fastest_rise_s(float(freqs[-1])) if freqs else 0.0
    if notes:
        report.note = (f"The rise time is shorter than the sweep resolves (fastest ≈ {fastest * 1e12:.0f} ps up to "
                       f"{float(freqs[-1]) / 1e9:.3g} GHz), so the edge is tapered to the band.")
    elif fastest:
        report.note = f"Fastest edge this sweep resolves ≈ {fastest * 1e12:.0f} ps (10–90 %)."
    if freqs and float(freqs[0]) > 0.1 * float(freqs[-1]):
        report.note = (f"⚠ The sweep starts at {float(freqs[0]) / 1e9:.3g} GHz: a TDR is built from the response "
                       "down to DC, so without the low end this is not a true impedance profile. Sweep from near "
                       "0 Hz for TDR. " + report.note)
    if not report.series:
        report.message = "TDR needs a reflection term of a port the run excited, with the sweep starting near DC."
    return report


# ------------------------------------------------------------------ far field

def pattern_entry(farfield: Optional[Dict], frequency_hz: Optional[float]) -> Optional[Dict]:
    entries = (farfield or {}).get("frequencies") or []
    if not entries:
        return None
    if frequency_hz is None:
        return entries[0]
    return min(entries, key=lambda e: abs(float(e["frequency_hz"]) - frequency_hz))


NO_FARFIELD = ("No far-field results. In Solution setup choose the Antenna solution type, add an operating "
               "frequency, and run the design.")


def quantity_label(plot, unit: str, curve: Optional[ReportTrace] = None) -> str:
    """'Gain RHCP (dBi)', 'rE θ (V)', 'Axial ratio (dB)'."""
    category = curve.category if curve is not None else plot.quantity
    component = curve.term if curve is not None else plot.component
    name = ff.QUANTITIES[category]
    if category != "axial_ratio" and component != "total":
        name += f" {ff.POLARISATIONS.get(component, component)}"
    return f"{name} ({unit})" if unit else name


def first_pattern_curve(plot, farfield: Optional[Dict] = None) -> ReportTrace:
    """The trace a contour or 3D pattern shows: the first far-field one shown."""
    curves = [c for c in curves_of(plot, {}, farfield) if c.visible and c.category in FARFIELD]
    return curves[0] if curves else default_curves(plot, {}, farfield)[0] if plot.kind == "farfield" else \
        ReportTrace(category="directivity", term="total", function="db")


def pattern_grid(plot, farfield: Optional[Dict], component: Optional[str] = None,
                 curve: Optional[ReportTrace] = None, entry: Optional[Dict] = None
                 ) -> Tuple[Optional[np.ndarray], str, str, Optional[Dict]]:
    """(θ×φ grid, unit, message, frequency entry) of one trace, the plot's first by default."""
    entry = entry or pattern_entry(farfield, plot.frequency_hz)
    if entry is None:
        return None, "", NO_FARFIELD, None
    curve = curve or first_pattern_curve(plot, farfield)
    grid, unit, message = ff.pattern(entry, (farfield or {}).get("fields"), curve.category,
                                     component or curve.term, curve.function != "lin")
    return grid, unit, message, entry


def cross_needed(plot, farfield: Optional[Dict] = None) -> str:
    """Why a co-/cross-polar plot cannot be drawn yet, or ''."""
    if not plot.cross_pol:
        return ""
    curves = [c for c in curves_of(plot, {}, farfield) if c.visible and c.category in FARFIELD
              and c.category != "axial_ratio"]
    if curves and not any(c.term in ff.CROSS for c in curves):
        return ("Choose the co-polarisation under Polarisation (θ, φ, RHCP, LHCP, Ludwig-3 X or Y): the cross-"
                "polarisation is the one orthogonal to it.")
    return ""


def _frequencies(plot, farfield: Optional[Dict]) -> List[Dict]:
    entries = (farfield or {}).get("frequencies") or []
    if plot.all_frequencies:
        return list(entries)
    entry = pattern_entry(farfield, plot.frequency_hz)
    return [entry] if entry is not None else []


def pattern_report(plot, farfield: Optional[Dict], curves: Optional[List[ReportTrace]] = None) -> Report:
    """Far-field cuts along θ or φ, at one frequency or all, with optional cross-polarisation."""
    report = Report(x_label="θ (°)" if plot.sweep == "theta" else "φ (°)")
    report.message = cross_needed(plot, farfield)
    if report.message:
        return report
    entries = _frequencies(plot, farfield)
    if not entries:
        report.message = NO_FARFIELD
        return report
    curves = [c for c in (curves if curves is not None else curves_of(plot, {}, farfield))
              if c.visible and c.category in FARFIELD]
    if not curves:
        report.message = "No traces: add one with Add trace."
        return report
    far = _FarCache(farfield)
    index_of = {id(e): n for n, e in enumerate(far.entries)}
    held = "φ" if plot.sweep == "theta" else "θ"
    several = len(entries) > 1
    for ci, curve in enumerate(curves):
        unit = trace_unit(curve)
        db = curve.function != "lin"
        cross = ff.CROSS.get(curve.term) if plot.cross_pol and curve.category != "axial_ratio" else None
        for entry in entries:
            n = index_of.get(id(entry), 0)
            grid, _u, why = far.grid(n, curve.category, curve.term, db)
            if grid is None:
                report.message = why
                return report
            cross_grid = far.grid(n, curve.category, cross, db)[0] if cross else None
            theta, phi = np.asarray(entry["theta_deg"], dtype=float), np.asarray(entry["phi_deg"], dtype=float)
            at_freq = f" · {float(entry['frequency_hz']) / 1e9:.4g} GHz" if several else ""
            for k, angle in enumerate(plot.cuts or [0.0]):
                x, y, at = ff.cut(grid, theta, phi, plot.sweep, float(angle))
                key = f"{ci}:{k}:{n}"
                co = "" if cross_grid is None else " co"
                report.series.append(Series(key, f"{trace_label(curve)}{co} · {held} = {at:g}°{at_freq}", x, y,
                                            unit=unit, axis=curve.axis, quantity=_quantity(curve),
                                            category=curve.category))
                if cross_grid is not None:
                    xc, yc, _at = ff.cut(cross_grid, theta, phi, plot.sweep, float(angle))
                    cross_curve = ReportTrace(category=curve.category, term=cross, function=curve.function)
                    report.series.append(Series(key, f"{trace_label(cross_curve)} cross · {held} = {at:g}°{at_freq}",
                                                xc, yc, dashed=True, unit=unit, axis=curve.axis,
                                                quantity=_quantity(curve), category=curve.category))
    _place_axes(report)
    first = _frequencies(plot, farfield)[0]
    report.note = pattern_note(plot, first, report.y_unit, curves[0])
    return report


def pattern_cuts(plot, farfield: Optional[Dict]) -> Report:
    return pattern_report(plot, farfield)


def pattern_note(plot, entry: Dict, unit: str, curve: Optional[ReportTrace] = None) -> str:
    curve = curve or first_pattern_curve(plot)
    parts = [f"{entry['frequency_hz'] / 1e9:.4g} GHz" if not plot.all_frequencies else "every far-field frequency"]
    offset = ff.power_offset(entry, "gain")
    level = float(entry["dmax_dbi"])
    if curve.category in ("gain", "realized_gain"):
        extra = ff.power_offset(entry, curve.category)
        level += 10 * math.log10(extra) if extra else 0.0
    name = ff.QUANTITIES.get(curve.category, "Directivity").lower() \
        if curve.category in ("directivity", "gain", "realized_gain") else "directivity"
    parts.append(f"peak {name} {level:.2f} dBi (total)")
    parts.append(f"main beam θ {entry['peak_theta_deg']:.0f}° φ {float(entry['peak_phi_deg']) % 360:.0f}°")
    if offset:
        parts.append(f"radiation efficiency {100 * offset:.1f} %")
    if curve.category == "re":
        parts.append("rE at 1 m for 1 W incident" if ff.per_watt(entry) != 1.0 else
                     "rE at openEMS's own excitation level: run again to scale it to 1 W incident")
    consistency = entry.get("consistency")
    if consistency is not None and abs(consistency - 1) > ff.CONSISTENCY_TOLERANCE:
        parts.append(f"⚠ the far field carries {consistency:.0%} of the power through the NF2FF box: raise the air "
                     "margin")
    return " · ".join(parts) + "."


def pattern_scale(plot, grid: np.ndarray, unit: str, category: Optional[str] = None) -> Tuple[float, float]:
    """(floor, peak) of a pattern: `range_db` below the peak in dB, zero up when linear."""
    category = category or plot.quantity
    finite = grid[np.isfinite(grid)]
    peak = float(finite.max()) if finite.size else 0.0
    if category == "axial_ratio":
        return 0.0, min(max(peak, 3.0), ff.AXIAL_RATIO_CAP_DB)
    if unit.startswith("dB"):
        return peak - max(float(plot.range_db), 3.0), peak
    return 0.0, peak if peak > 0 else 1.0


def series_scale(plot, series: List[Series]) -> Tuple[float, float]:
    """(floor, peak) for a polar pattern of several traces of one unit."""
    values = np.concatenate([s.y[np.isfinite(s.y)] for s in series]) if series else np.array([])
    peak = float(values.max()) if values.size else 0.0
    if series and series[0].category == "axial_ratio":
        return 0.0, min(max(peak, 3.0), ff.AXIAL_RATIO_CAP_DB)
    if series and series[0].unit.startswith("dB"):
        return peak - max(float(plot.range_db), 3.0), peak
    return 0.0, peak if peak > 0 else 1.0


# ------------------------------------------------------------------ near field

def near_frequency(near: Dict, frequency_hz: Optional[float]) -> int:
    freqs = np.asarray(near["frequency_hz"], dtype=float)
    if frequency_hz is None or not freqs.size:
        return 0
    return int(np.argmin(np.abs(freqs - frequency_hz)))


def near_values(plot, values: np.ndarray, quantity: str) -> Tuple[np.ndarray, str]:
    """Magnitudes as the plot shows them: dB(V/m) or V/m (A/m for an H dump)."""
    unit = "A/m" if quantity == "H" else "V/m"
    if plot.db:
        return 20 * np.log10(np.maximum(values, 1e-30)), f"dB{unit}"
    return values, unit


def near_label(plot, quantity: str, unit: str) -> str:
    part = f"|{quantity}|" if plot.field_component == "total" else f"|{quantity}{plot.field_component}|"
    return f"{part} ({unit})"


def incident_power(summary: Optional[Dict], port: int, frequency_hz: float) -> Optional[float]:
    """A driven port's incident power at a frequency, from the run's summary."""
    powers = (summary or {}).get("port_powers") or {}
    measured = powers.get(str(port))
    if not measured or not powers.get("freq_hz"):
        return None
    value = float(np.interp(frequency_hz, powers["freq_hz"], measured["incident"]))
    return value if value > 0 else None


def near_plane(plot, near: Dict, quantity: str, incident_w: Optional[float] = None) -> Dict:
    """One plane of a field dump, scaled to 1 W incident when `incident_w` is given."""
    index = near_frequency(near, plot.frequency_hz)
    cut = nf.plane(near, index, plot.field_component, plot.slice_axis, plot.slice_mm)
    if incident_w:
        cut["values"] = cut["values"] / math.sqrt(incident_w)
    cut["values"], cut["unit"] = near_values(plot, cut["values"], quantity)
    cut["frequency_hz"] = float(near["frequency_hz"][index])
    cut["per_watt"] = bool(incident_w)
    return cut


def near_note(cut: Dict, port: int) -> str:
    level = ("for 1 W incident" if cut["per_watt"] else
             "at openEMS's own excitation level (run again to scale it to 1 W incident)")
    return (f"{cut['frequency_hz'] / 1e9:.4g} GHz · port {port} excited, {level} · plane "
            f"{cut['held']} = {cut['held_mm']:.4g} mm.")


def near_line(plot, near: Dict, quantity: str, incident_w: Optional[float] = None, port: int = 1) -> Report:
    cut = near_plane(plot, near, quantity, incident_w)
    along, x, y, other, at = nf.line_cut(cut, plot.line_axis, plot.line_mm)
    report = Report(x_label=f"{along} (mm)", y_unit=cut["unit"], y_label=near_label(plot, quantity, cut["unit"]))
    report.series.append(Series("line", f"{other} = {at:.4g} mm, {cut['held']} = {cut['held_mm']:.4g} mm", x, y,
                                unit=cut["unit"]))
    report.note = near_note(cut, port)
    return report


# ------------------------------------------------------------------ tables

def table(report: Report) -> Tuple[List[str], List[List[str]]]:
    """Columns and rows of a report's data table: x, then one column per series."""
    if not report.series:
        return [], []
    scale, x_label = 1.0, report.x_label or "x"
    if report.x_unit == "Hz":
        scale, x_label = 1e-9, f"{x_label} (GHz)"
    elif report.x_unit:
        x_label = f"{x_label} ({report.x_unit})"
    columns = [x_label] + [f"{s.label}" + (f" ({s.unit})" if s.unit else "") for s in report.series]
    x = report.series[0].x
    if all(s.x.shape == x.shape and np.allclose(s.x, x, equal_nan=True) for s in report.series):
        return columns, [[_number(x[i] * scale)] + [_number(s.y[i]) for s in report.series] for i in range(x.size)]
    every = np.unique(np.concatenate([s.x[np.isfinite(s.x)] for s in report.series]))
    rows = []
    for xi in every:
        row = [_number(xi * scale)]
        for s in report.series:
            hit = np.flatnonzero(np.isclose(s.x, xi, rtol=1e-9, atol=1e-12))
            row.append(_number(s.y[hit[0]]) if hit.size else "")
        rows.append(row)
    return columns, rows


def _number(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "" if not math.isfinite(value) else f"{value:.6g}"


def to_csv(columns: List[str], rows: List[List[str]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return out.getvalue()
