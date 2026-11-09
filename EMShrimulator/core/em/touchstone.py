"""Touchstone S-parameter files, read and written with numpy alone.

The reader is strict about the format this app writes and hands anything else to scikit-rf.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np

_FREQUENCY_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
_FORMATS = ("ri", "ma", "db")
_PORTS_RE = re.compile(r"\.s(\d+)p$", re.IGNORECASE)
_PORT_IMPEDANCE = "port impedance"
# Touchstone 2 keywords that don't affect a full S-matrix; others go to scikit-rf.
_V2_BOOKKEEPING = ("version", "number of ports", "number of frequencies", "network data", "end")

Impedance = Union[float, Sequence[float], np.ndarray]


class NotPlainTouchstone(ValueError):
    """A file outside the strict S-parameter subset the native reader handles."""


@dataclass
class Touchstone:
    freq_hz: np.ndarray   # (F,)
    s: np.ndarray         # (F, N, N) complex, s[:, i, j] = S(i+1)(j+1)
    z0_ports: np.ndarray  # (N,) reference impedance of each port, ohms

    @property
    def ports(self) -> int:
        return self.s.shape[1]

    @property
    def z0(self) -> float:
        """Port 1's reference impedance, ohms."""
        return float(self.z0_ports[0])


def port_impedances(z0: Impedance, n: int) -> np.ndarray:
    """One reference impedance per port, from a single value or one value per port."""
    values = np.atleast_1d(np.asarray(z0, dtype=float))
    if values.size == 1:
        return np.full(n, float(values[0]))
    if values.size != n:
        raise ValueError(f"{values.size} reference impedances for {n} ports")
    return values


def port_count(path: Path) -> int:
    match = _PORTS_RE.search(path.name)
    if not match or int(match.group(1)) < 1:
        raise NotPlainTouchstone(f"'{path.name}' does not name its port count as .s<N>p")
    return int(match.group(1))


def _parse(text: str, n: int) -> Touchstone:
    unit, parameter, fmt, reference = "ghz", "s", "ma", 50.0
    port_z0 = None          # scikit-rf's "! Port Impedance" comments
    references = None       # Touchstone 2's [Reference]
    in_references = False   # [Reference] values may run over several lines
    column_major = n == 2  # two-port order is S11 S21 S12 S22 by default
    values: List[str] = []
    seen_options = False
    for raw in text.splitlines():
        line, _, comment = raw.partition("!")
        if port_z0 is None and comment.strip().lower().startswith(_PORT_IMPEDANCE):
            # scikit-rf writes each port's reference as real and imaginary parts.
            numbers = comment.strip()[len(_PORT_IMPEDANCE):].split()
            if len(numbers) >= 2 * n:
                port_z0 = [float(v) for v in numbers[0:2 * n:2]]
            elif numbers:
                port_z0 = [float(numbers[0])] * n
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            keyword, _, rest = line[1:].partition("]")
            keyword, rest = keyword.strip().lower(), rest.strip()
            in_references = False
            if keyword == "reference":
                references, in_references = [], True
                if not rest:
                    continue
                line = rest
            elif keyword == "two-port data order" and rest in ("12_21", "21_12"):
                column_major = n == 2 and rest == "21_12"
                continue
            elif keyword == "number of ports" and rest.isdigit() and int(rest) != n:
                raise NotPlainTouchstone(f"[Number of Ports] {rest} in a {n}-port file")
            elif keyword in _V2_BOOKKEEPING or (keyword == "matrix format" and rest.lower() == "full"):
                continue
            else:
                raise NotPlainTouchstone(f"[{keyword}] {rest}".strip() + " is left to scikit-rf")
        if in_references:
            try:
                references.extend(float(v) for v in line.split())
            except ValueError as exc:
                raise NotPlainTouchstone(f"bad [Reference] value: {exc}") from exc
            in_references = len(references) < n
            continue
        if line.startswith("#"):
            if seen_options:
                continue  # the spec says to ignore every option line after the first
            seen_options = True
            tokens = line[1:].lower().split()
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token in _FREQUENCY_UNITS:
                    unit = token
                elif token in ("s", "y", "z", "h", "g"):
                    parameter = token
                elif token in _FORMATS:
                    fmt = token
                elif token == "r":
                    if i + 1 < len(tokens):
                        try:
                            reference = float(tokens[i + 1])
                            i += 1
                        except ValueError:
                            pass
                else:
                    raise NotPlainTouchstone(f"unknown option '{token}'")
                i += 1
            continue
        values.extend(line.split())

    if parameter != "s":
        raise NotPlainTouchstone(f"{parameter.upper()}-parameters are left to scikit-rf")
    try:
        data = np.array(values, dtype=float)
    except ValueError as exc:
        raise NotPlainTouchstone(f"non-numeric data: {exc}") from exc
    width = 1 + 2 * n * n
    if data.size == 0 or data.size % width:
        raise NotPlainTouchstone(f"{data.size} numbers do not divide into {n}-port records of {width}")
    rows = data.reshape(-1, width)
    freq_hz = rows[:, 0] * _FREQUENCY_UNITS[unit]
    a, b = rows[:, 1::2], rows[:, 2::2]
    if fmt == "ri":
        flat = a + 1j * b
    else:
        magnitude = a if fmt == "ma" else 10.0 ** (a / 20.0)
        flat = magnitude * np.exp(1j * np.radians(b))
    s = flat.reshape(-1, n, n)
    if column_major:
        s = s.transpose(0, 2, 1)
    if references is not None and len(references) != n:
        raise NotPlainTouchstone(f"[Reference] gives {len(references)} impedances for {n} ports")
    z0 = references if references is not None else port_z0 if port_z0 is not None else reference
    return Touchstone(freq_hz, np.ascontiguousarray(s), port_impedances(z0, n))


def read(path: Path) -> Touchstone:
    """Frequencies, S-matrix and per-port reference impedances of a Touchstone file."""
    path = Path(path)
    try:
        return _parse(path.read_text(encoding="utf-8", errors="replace"), port_count(path))
    except NotPlainTouchstone:
        import skrf  # slow, and only for files this application did not write

        net = skrf.Network(str(path))
        n = net.s.shape[1]
        z0 = np.real(net.z0[0, :]) if net.z0 is not None and net.z0.size else 50.0
        return Touchstone(np.asarray(net.f, dtype=float), np.asarray(net.s), port_impedances(z0, n))


def _ohms(z: float) -> str:
    """Shortest text that reads back as exactly `z`."""
    short = f"{z:g}"
    return short if float(short) == z else repr(float(z))


def write(base: Path, freq_hz: np.ndarray, s: np.ndarray, z0: Impedance) -> Path:
    """Write `<base>.s<N>p` in real/imaginary form and return its path.

    `z0` is one reference impedance or one per port; different ones make a Touchstone 2 file.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    s = np.asarray(s, dtype=complex)
    n = s.shape[1]
    z0_ports = port_impedances(z0, n)
    uniform = bool(np.all(z0_ports == z0_ports[0]))
    path = Path(f"{base}.s{n}p")
    ordered = s.transpose(0, 2, 1) if n == 2 else s
    names = [[f"S{j + 1}{i + 1}" if n == 2 else f"S{i + 1}{j + 1}" for j in range(n)] for i in range(n)]
    lines = ["! Written by EMShrimulator Studio", "! S-parameter uses the power definition"]
    if not uniform:
        lines.append("[Version] 2.0")
    lines.append(f"# Hz S RI R {_ohms(z0_ports[0])}")
    if not uniform:
        lines.append(f"[Number of Ports] {n}")
        if n == 2:
            lines.append("[Two-Port Data Order] 21_12")
        lines += [f"[Number of Frequencies] {len(freq_hz)}",
                  "[Reference] " + " ".join(_ohms(z) for z in z0_ports),
                  "[Network Data]"]
    lines.append("!freq " + "\n! ".join(" ".join(f"Re{k} Im{k}" for k in row) for row in names))
    for f, matrix in zip(freq_hz, ordered):
        rows = [" ".join(f"{repr(float(v.real))} {repr(float(v.imag))}" for v in row) for row in matrix]
        lines.append(f"{repr(float(f))} " + "\n ".join(rows))
    if not uniform:
        lines.append("[End]")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_touchstone(freqs_hz: np.ndarray, s_matrix: Dict[int, Dict[int, np.ndarray]], port_numbers: List[int],
                     z0: Impedance, path: Path) -> Path:
    """s_matrix[i][j] is S_(j,i) over freqs_hz, for excited ports i (1-based).

    Unexcited ports get their column by reciprocity and a zero reflection. `z0` is one impedance or one per port.
    """
    n = len(port_numbers)
    s = np.zeros((len(freqs_hz), n, n), dtype=complex)
    for a, i in enumerate(port_numbers):
        for b, j in enumerate(port_numbers):
            column = s_matrix.get(i, {}).get(j)
            if column is None:
                column = s_matrix.get(j, {}).get(i)
            if column is not None:
                s[:, b, a] = column
    return write(path.with_suffix(""), freqs_hz, s, z0)
