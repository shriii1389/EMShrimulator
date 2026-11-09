"""Near-field reports from a run's frequency-domain field dumps.

The solver copies each HDF5 dump to results/nearfield/<dump>__port<n>.npz (mm, complex64) so the app can read it
with numpy alone; h5py is only needed for older runs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FOLDER = "nearfield"
_NAME = re.compile(r"^(?P<dump>.+)__port(?P<port>\d+)\.npz$")
AXES = "xyz"


def read_h5(path: Path) -> Dict[str, np.ndarray]:
    """One openEMS frequency-domain dump: mesh in mm, frequencies, field (F, 3, nz, ny, nx)."""
    import h5py

    with h5py.File(path, "r") as f:
        mesh = {axis: np.asarray(f[f"Mesh/{axis}"], dtype=float) * 1e3 for axis in AXES}
        group = f["FieldData/FD"]
        listed = np.atleast_1d(np.asarray(group.attrs.get("frequency", []), dtype=float))
        count = len([k for k in group.keys() if k.endswith("_real")])
        freqs, fields = [], []
        for n in range(count):
            real = group[f"f{n}_real"]
            own = np.atleast_1d(np.asarray(real.attrs.get("frequency", []), dtype=float))
            freqs.append(float(own[0]) if own.size else float(listed[n]) if n < listed.size else float("nan"))
            fields.append(np.asarray(real) + 1j * np.asarray(group[f"f{n}_imag"]))
    return {"x_mm": mesh["x"], "y_mm": mesh["y"], "z_mm": mesh["z"], "frequency_hz": np.asarray(freqs),
            "field": np.asarray(fields, dtype=np.complex64)}


def export(results_dir: Path, sim_dirs: Dict[int, Path], dump_names: List[str]) -> List[Path]:
    """Copy every frequency-domain dump of every run into results/nearfield as .npz."""
    target = Path(results_dir) / FOLDER
    for old in target.glob("*.npz"):
        old.unlink(missing_ok=True)
    written = []
    for port, folder in sim_dirs.items():
        for name in dump_names:
            source = Path(folder) / f"{name}.h5"
            if not source.exists():
                continue  # a time-domain dump (VTK), or one openEMS left empty
            data = read_h5(source)
            target.mkdir(parents=True, exist_ok=True)
            path = target / f"{name}__port{port}.npz"
            np.savez_compressed(path, **data)
            written.append(path)
    return written


def available(results_dir: Path) -> List[Dict]:
    """The dumps a run left: [{dump, port}], from the copies, else the HDF5 files."""
    results_dir = Path(results_dir)
    found = []
    for path in sorted((results_dir / FOLDER).glob("*.npz")):
        match = _NAME.match(path.name)
        if match:
            found.append({"dump": match["dump"], "port": int(match["port"])})
    if found:
        return found
    for folder in sorted(results_dir.glob("sim_port_*")):
        port = folder.name.rsplit("_", 1)[-1]
        if not port.isdigit():
            continue
        for path in sorted(folder.glob("*.h5")):
            if not path.stem.startswith("nf2ff"):
                found.append({"dump": path.stem, "port": int(port)})
    return found


def load(results_dir: Path, dump: str, port: int) -> Optional[Dict[str, np.ndarray]]:
    results_dir = Path(results_dir)
    path = results_dir / FOLDER / f"{dump}__port{port}.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    source = results_dir / f"sim_port_{port}" / f"{dump}.h5"
    return read_h5(source) if source.exists() else None


def magnitude(field: np.ndarray, component: str) -> np.ndarray:
    """|E| (or |H|) of one frequency's field, or one Cartesian component: (nz, ny, nx)."""
    if component in AXES:
        return np.abs(field[AXES.index(component)])
    return np.sqrt((np.abs(field) ** 2).sum(axis=0))


def plane(data: Dict[str, np.ndarray], index: int, component: str, slice_axis: Optional[str] = None,
          slice_mm: Optional[float] = None) -> Dict:
    """A plane of the dump: held axis, in-plane axes, their coordinates and values (v rows, u columns)."""
    coords = {axis: np.asarray(data[f"{axis}_mm"], dtype=float) for axis in AXES}
    values = magnitude(np.asarray(data["field"][index]), component)  # (nz, ny, nx)
    flat = [axis for axis in AXES if coords[axis].size == 1]
    if flat:
        held = flat[0]
    elif slice_axis and slice_axis in AXES:
        held = slice_axis
    else:
        held = min(AXES, key=lambda a: float(np.ptp(coords[a])))
    line = coords[held]
    at = int(np.argmin(np.abs(line - slice_mm))) if slice_mm is not None else line.size // 2
    # Values are (z, y, x); take the plane and order it (v, u).
    u_axis, v_axis = [axis for axis in AXES if axis != held]
    taken = np.take(values, at, axis=2 - AXES.index(held))  # remaining axes in reversed order: (v, u)
    return {"held": held, "held_mm": float(line[at]), "u": u_axis, "v": v_axis, "u_mm": coords[u_axis],
            "v_mm": coords[v_axis], "values": taken}


def line_cut(cut: Dict, along: Optional[str] = None, at_mm: Optional[float] = None) -> Tuple[str, np.ndarray,
                                                                                               np.ndarray, str, float]:
    """A line through `plane` along one in-plane axis: (axis, coordinates, values, held axis, held at)."""
    u, v = cut["u"], cut["v"]
    if along not in (u, v):
        along = u if np.ptp(cut["u_mm"]) >= np.ptp(cut["v_mm"]) else v
    other = v if along == u else u
    across = cut[f"{'v' if along == u else 'u'}_mm"]
    at = int(np.argmin(np.abs(across - at_mm))) if at_mm is not None else across.size // 2
    values = cut["values"][at, :] if along == u else cut["values"][:, at]
    return along, cut[f"{'u' if along == u else 'v'}_mm"], values, other, float(across[at])
