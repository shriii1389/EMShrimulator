"""Read gerber2ems output and assemble a Touchstone from ems/simulation/Sx<n>.csv."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _results_dir(project_dir: Path) -> Path:
    return project_dir / "ems" / "results"


def _simulation_dir(project_dir: Path) -> Path:
    return project_dir / "ems" / "simulation"


def list_result_ports(project_dir: Path) -> List[int]:
    results_dir = _results_dir(project_dir)
    if not results_dir.exists():
        return []
    ports = []
    for f in results_dir.glob("Port_*_data.csv"):
        m = re.match(r"Port_(\d+)_data\.csv", f.name)
        if m:
            ports.append(int(m.group(1)))
    return sorted(ports)


def read_port_data(project_dir: Path, port_number: int) -> Dict:
    path = _results_dir(project_dir) / f"Port_{port_number}_data.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [[float(v) for v in row] for row in reader if row]

    mag_cols = {i: h for i, h in enumerate(header) if h.startswith("|S")}
    # Exact (to, from) match: a substring test confuses S1-1 with S1-11.
    arg_cols = {(m.group(1), m.group(2)): i for i, h in enumerate(header) if (m := re.search(r"Arg\(S(\d+)-(\d+)\)", h))}
    z_mag_col = next((i for i, h in enumerate(header) if h.startswith("|Z")), None)
    z_arg_col = next((i for i, h in enumerate(header) if h.startswith("Arg(Z")), None)

    freqs_mhz = [row[0] for row in rows]
    series: Dict[str, Dict] = {}
    for i, h in mag_cols.items():
        m = re.search(r"\|S(\d+)-(\d+)\|", h)
        if not m:
            continue
        to_port = m.group(1)
        arg_i = arg_cols.get((to_port, m.group(2)))
        mags = [row[i] for row in rows]
        phases = [row[arg_i] for row in rows] if arg_i is not None else [0.0] * len(rows)
        series[f"S{to_port}{port_number}"] = {
            "mag_db": [20.0 * math.log10(max(v, 1e-12)) for v in mags],
            "mag_lin": mags,
            "phase_rad": phases,
        }

    impedance = None
    if z_mag_col is not None:
        z_mag = [row[z_mag_col] for row in rows]
        z_arg = [row[z_arg_col] for row in rows] if z_arg_col is not None else [0.0] * len(rows)
        impedance = {
            "real": [m * math.cos(a) for m, a in zip(z_mag, z_arg)],
            "imag": [m * math.sin(a) for m, a in zip(z_mag, z_arg)],
        }

    return {"port": port_number, "freq_mhz": freqs_mhz, "series": series, "impedance": impedance}


# --- Touchstone assembly ---

_SPARAM_RE = re.compile(r"(re|im)\(S(\d+)-(\d+)\)", re.IGNORECASE)


def _read_sx(path: Path) -> Optional[Dict]:
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader, [])]
        rows = [[float(v) for v in row] for row in reader if row and not row[0].strip().startswith("#")]
    if not rows:
        return None
    freqs_hz = np.array([r[0] for r in rows]) * 1e6
    columns: Dict[tuple, Dict[str, np.ndarray]] = {}
    for i, h in enumerate(header):
        m = _SPARAM_RE.search(h)
        if not m:
            continue
        part, to_port, from_port = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        columns.setdefault((to_port, from_port), {})[part] = np.array([r[i] for r in rows])
    return {"freqs_hz": freqs_hz, "columns": columns}


def _port_impedances(project_dir: Path) -> Dict[int, float]:
    """Reference impedance by gerber2ems port index, from the simulation.json it ran with."""
    try:
        config = json.loads((project_dir / "simulation.json").read_text(encoding="utf-8"))
        return {i: float(port.get("impedance", 50.0)) for i, port in enumerate(config.get("ports") or [])}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def build_touchstone(project_dir: Path, name: str) -> Optional[Path]:
    """Assemble every `Sx<n>.csv` into one Touchstone file. Returns its path."""
    from EMShrimulator.core.em import touchstone

    sim_dir = _simulation_dir(project_dir)
    files = sorted(sim_dir.glob("Sx*.csv"))
    if not files:
        return None

    parsed = [(int(re.search(r"Sx(\d+)", f.name).group(1)), _read_sx(f)) for f in files
              if re.search(r"Sx(\d+)", f.name)]
    parsed = [(n, d) for n, d in parsed if d]
    if not parsed:
        return None

    freqs = parsed[0][1]["freqs_hz"]
    port_indices = sorted({idx for _, d in parsed for pair in d["columns"] for idx in pair})
    n = len(port_indices)
    pos = {p: i for i, p in enumerate(port_indices)}

    s = np.zeros((len(freqs), n, n), dtype=complex)
    filled = np.zeros((n, n), dtype=bool)
    for _, data in parsed:
        for (to_port, from_port), parts in data["columns"].items():
            if "re" not in parts or "im" not in parts:
                continue
            col = parts["re"] + 1j * parts["im"]
            if len(col) != len(freqs):
                col = np.interp(freqs, data["freqs_hz"], col.real) + 1j * np.interp(freqs, data["freqs_hz"], col.imag)
            i, j = pos[to_port], pos[from_port]
            s[:, i, j] = col
            filled[i, j] = True

    # Fill unexcited columns by reciprocity.
    for i in range(n):
        for j in range(n):
            if not filled[i, j] and filled[j, i]:
                s[:, i, j] = s[:, j, i]

    impedances = _port_impedances(project_dir)
    z0 = [impedances.get(p, 50.0) for p in port_indices]
    return touchstone.write(_results_dir(project_dir) / name, freqs, s, z0)


def find_touchstone(project_dir: Path) -> Optional[Path]:
    results = _results_dir(project_dir)
    if not results.exists():
        return None
    candidates = sorted(results.glob("*.s*p"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def list_plots(project_dir: Path) -> List[str]:
    """PNG plots gerber2ems's own postprocess step renders (S-params, Smith, …)."""
    results = _results_dir(project_dir)
    if not results.exists():
        return []
    return sorted(p.name for p in results.glob("*.png"))
