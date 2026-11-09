"""Results views derived from the Touchstone file: magnitude/phase, VSWR, Z, group delay, TDR, Smith."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

C0 = 299_792_458.0


def find_touchstone(results_dir: Path) -> Optional[Path]:
    if not results_dir.exists():
        return None
    candidates = sorted(results_dir.glob("*.s*p"), key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.suffix.lower().startswith(".s") and c.suffix.lower().endswith("p"):
            return c
    return None


def list_touchstones(results_dir: Path) -> List[Dict]:
    if not results_dir.exists():
        return []
    out = []
    for p in sorted(results_dir.glob("*.s*p")):
        st = p.stat()
        out.append({"file": p.name, "size": st.st_size, "modified": st.st_mtime})
    return out


def _group_delay(freqs: np.ndarray, s: np.ndarray) -> np.ndarray:
    """-d(unwrapped phase)/d(omega), in seconds."""
    if len(freqs) < 2:
        return np.zeros_like(freqs)
    phase = np.unwrap(np.angle(s))
    omega = 2 * np.pi * freqs
    return -np.gradient(phase, omega)


RISE_SPAN = 2.563  # 10-90 % rise of a Gaussian step, in standard deviations


def fastest_rise_s(f_max: float) -> float:
    """The shortest 10-90 % rise time a sweep up to `f_max` can resolve."""
    return RISE_SPAN * math.sqrt(2 * math.log(100)) / (2 * math.pi * f_max) if f_max > 0 else 0.0


def _tdr(freqs: np.ndarray, s11: np.ndarray, z0: float, epsilon_eff: float = 1.0,
         rise_time_s: Optional[float] = None) -> Dict:
    """Band-limited step-response TDR from S11, with distance in a medium of `epsilon_eff`."""
    n_uniform = 1024
    f_uniform = np.linspace(0.0, float(freqs[-1]), n_uniform)
    real = np.interp(f_uniform, freqs, s11.real, left=float(s11.real[0]))
    imag = np.interp(f_uniform, freqs, s11.imag, left=0.0)
    h_f = real + 1j * imag
    # Hann window over the populated band keeps the step edge clean.
    window = 0.5 * (1 - np.cos(np.pi * np.arange(n_uniform) / (n_uniform - 1)))[::-1]
    limited = False
    if rise_time_s:
        sigma = float(rise_time_s) / RISE_SPAN
        gaussian = np.exp(-0.5 * (2 * np.pi * f_uniform * sigma) ** 2)
        limited = gaussian[-1] > 0.01
        window = gaussian * window if limited else gaussian
    h_f = h_f * window

    full = np.concatenate([h_f, np.conj(h_f[-2:0:-1])])
    # The impulse is centred on t = 0, so shift it before summing into a step.
    impulse = np.fft.fftshift(np.real(np.fft.ifft(full)))
    step_all = np.cumsum(impulse)

    df = f_uniform[1] - f_uniform[0]
    dt = 1.0 / (len(full) * df)
    t_all = (np.arange(len(full)) - len(full) // 2) * dt
    start = len(full) // 2  # t = 0
    stop = len(full)
    # Past 1/Δf the step folds back on itself.
    spacing = float(np.min(np.diff(freqs))) if len(freqs) > 1 else 0.0
    if spacing > 0:
        stop = max(start + 2, min(stop, int(np.searchsorted(t_all, 1.0 / spacing, side="right"))))
    t, step = t_all[start:stop], step_all[start:stop]

    with np.errstate(divide="ignore", invalid="ignore"):
        z = z0 * (1 + step) / (1 - step)
    z = np.nan_to_num(z, nan=z0, posinf=z0 * 20, neginf=0.0)
    z = np.clip(z, 0.0, z0 * 20)

    velocity = C0 / math.sqrt(max(epsilon_eff, 1.0))
    distance_mm = (t * velocity / 2) * 1000.0

    return {
        "time_ns": (t * 1e9).tolist(),
        "distance_mm": distance_mm.tolist(),
        "impedance_ohm": z.tolist(),
        "rho": step.tolist(),
        "band_limited": limited,
    }


def tdr_of(data: Dict, key: str, rise_time_s: Optional[float] = None) -> Optional[Dict]:
    """TDR of one reflection term (S11, Sdd11) against its own reference impedance."""
    trace = (data.get("traces") or {}).get(key)
    freqs = np.asarray(data.get("freq_hz") or [], dtype=float)
    if not trace or trace.get("ref_ohm") is None or len(freqs) < 2:
        return None
    s = np.asarray(trace["real"], dtype=float) + 1j * np.asarray(trace["imag"], dtype=float)
    return _tdr(freqs, s, float(trace["ref_ohm"]), float(data.get("epsilon_eff") or 1.0), rise_time_s)


def _trace(freqs: np.ndarray, sij: np.ndarray, z_ref: Optional[float] = None) -> Dict:
    """One S term in every form the views draw; with `z_ref`, also VSWR and impedance."""
    mag = np.abs(sij)
    entry = {
        "mag_db": (20 * np.log10(np.maximum(mag, 1e-15))).tolist(),
        "mag_lin": mag.tolist(),
        "phase_deg": np.degrees(np.unwrap(np.angle(sij))).tolist(),
        "real": sij.real.tolist(),
        "imag": sij.imag.tolist(),
        "group_delay_ns": (_group_delay(freqs, sij) * 1e9).tolist(),
    }
    if z_ref is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            vswr = (1 + mag) / (1 - mag)
        entry["vswr"] = np.nan_to_num(vswr, nan=99.0, posinf=99.0).clip(1.0, 99.0).tolist()
        zin = z_ref * (1 + sij) / np.where(np.abs(1 - sij) < 1e-12, 1e-12, (1 - sij))
        entry["z_real"] = np.real(zin).tolist()
        entry["z_imag"] = np.imag(zin).tolist()
        entry["ref_ohm"] = float(z_ref)
    return entry


def renormalize(s: np.ndarray, z_from, z_to) -> np.ndarray:
    """Re-reference S (F, N, N) from real impedances `z_from` to `z_to` (power waves).

    S' = A (S - G)(I - G S)^-1 A^-1,  G = diag((Z' - Z)/(Z' + Z)),  A = diag((Z' + Z)/(2 sqrt(Z Z')))
    """
    z_from, z_to = np.asarray(z_from, dtype=float), np.asarray(z_to, dtype=float)
    gamma = np.diag((z_to - z_from) / (z_to + z_from))
    alpha = (z_to + z_from) / (2 * np.sqrt(z_from * z_to))
    eye = np.eye(len(z_from))
    return np.diag(alpha) @ (s - gamma) @ np.linalg.inv(eye - gamma @ s) @ np.diag(1 / alpha)


def mixed_mode_traces(freqs: np.ndarray, s: np.ndarray, z0_ports, pairs: List[Dict]) -> Tuple[Dict[str, Dict], str]:
    """Mixed-mode S of differential pairs, and a note on what could not be computed.

    `pairs`: [{"name", "p", "n", "z_diff", "z_common"}] with p and n Touchstone port indices. Keys are
    S<out mode><in mode><out pair><in pair>, e.g. Sdd21, Scd11. Terms needing an unexcited port are left out.
    """
    n_ports = s.shape[1]
    pairs = [p for p in pairs if 0 <= p["p"] < n_ports and 0 <= p["n"] < n_ports and p["p"] != p["n"]]
    if not pairs:
        return {}, ""
    z0_ports = np.asarray(z0_ports, dtype=float)
    driven = [bool(np.any(s[:, j, j])) for j in range(n_ports)]
    index = [i for pair in pairs for i in (pair["p"], pair["n"])]
    sub = s[:, index][:, :, index]
    known = np.array([[driven[r] if r == c else driven[r] or driven[c] for c in index] for r in index])
    complete = bool(known.all())

    # One reference per pair: the mean of its ports'.
    z_se = z0_ports[index]
    z_pair = np.repeat([(z_se[2 * k] + z_se[2 * k + 1]) / 2 for k in range(len(pairs))], 2)
    if not np.allclose(z_se, z_pair):
        if not complete:
            return {}, ("Mixed-mode terms need every port of every pair excited when a pair's two ports "
                        "have different reference impedances.")
        sub = renormalize(sub, z_se, z_pair)

    m = np.zeros((len(index), len(index)))
    for k in range(len(pairs)):
        m[2 * k, 2 * k], m[2 * k, 2 * k + 1] = 1, -1    # differential
        m[2 * k + 1, 2 * k], m[2 * k + 1, 2 * k + 1] = 1, 1  # common
    m /= np.sqrt(2)
    mm = m @ sub @ m.T
    natural = np.array([z for k in range(len(pairs)) for z in (2 * z_pair[2 * k], z_pair[2 * k] / 2)])
    chosen = np.array([z for pair in pairs for z in (float(pair["z_diff"]), float(pair["z_common"]))])
    note = ""
    if not np.allclose(natural, chosen):
        if not complete:
            idle = sorted({i + 1 for i in index if not driven[i]})
            return {}, ("Mixed-mode terms are re-referenced to each pair's own impedances, which needs every "
                        f"port of every pair excited; port{'s' if len(idle) > 1 else ''} "
                        f"{', '.join(f'P{i}' for i in idle)} {'were' if len(idle) > 1 else 'was'} not.")
        mm = renormalize(mm, natural, chosen)
    elif not complete:
        note = "Some mixed-mode terms are missing: they need ports that were not excited."

    traces: Dict[str, Dict] = {}
    modes = "dc"
    for k, out_pair in enumerate(pairs):
        for u in range(2):
            for l_, in_pair in enumerate(pairs):
                for v in range(2):
                    if not known[2 * k:2 * k + 2, 2 * l_:2 * l_ + 2].all():
                        continue
                    term = mm[:, 2 * k + u, 2 * l_ + v]
                    reflection = k == l_ and u == v
                    entry = _trace(freqs, term, chosen[2 * k + u] if reflection else None)
                    if reflection:
                        entry["z_label"] = f"{'Zdiff' if u == 0 else 'Zcomm'} {out_pair['name']}"
                    traces[f"S{modes[u]}{modes[v]}{k + 1}{l_ + 1}"] = entry
    return traces, note


def read_results(touchstone: Path, epsilon_eff: float = 1.0, pairs: Optional[List[Dict]] = None) -> Dict:
    """Parse a Touchstone file into every derived view; `pairs` adds mixed-mode terms."""
    from EMShrimulator.core.em import touchstone as ts

    network = ts.read(touchstone)
    freqs = network.freq_hz
    s = network.s
    n = network.ports
    z0_ports = network.z0_ports

    traces: Dict[str, Dict] = {}
    for i in range(n):
        for j in range(n):
            sij = s[:, i, j]
            if not np.any(sij):
                # Exactly zero everywhere: an unexcited port's reflection, not a result.
                continue
            traces[f"S{i + 1}{j + 1}"] = _trace(freqs, sij, float(z0_ports[i]) if i == j else None)

    z_traces = impedance_traces(s, z0_ports, traces)
    y_traces = admittance_traces(s, z0_ports, z_traces)
    excess = passivity_excess(s)
    mixed_note = ""
    if pairs:
        mixed, mixed_note = mixed_mode_traces(freqs, s, z0_ports, pairs)
        traces.update(mixed)

    tdr = None
    # From the first port whose reflection was measured.
    tdr_port = next((k for k in range(n) if f"S{k + 1}{k + 1}" in traces), None)
    if tdr_port is not None:
        try:
            tdr = _tdr(freqs, s[:, tdr_port, tdr_port], float(z0_ports[tdr_port]), epsilon_eff)
            tdr["port"] = tdr_port + 1
        except Exception:  # noqa: BLE001 - TDR is a bonus view, never fatal
            tdr = None

    markers = {}
    s11_db = np.array(traces["S11"]["mag_db"]) if "S11" in traces else None
    if s11_db is not None and len(s11_db):
        idx = int(np.argmin(s11_db))
        markers["resonance"] = {
            "frequency_hz": float(freqs[idx]),
            "s11_db": float(s11_db[idx]),
            "bandwidth_hz": _bandwidth_at(freqs, s11_db, -10.0),
        }

    return {
        "file": touchstone.name,
        "ports": n,
        "z0": float(z0_ports[0]),
        "z0_ports": z0_ports.tolist(),
        "freq_hz": freqs.tolist(),
        "traces": traces,
        "z_traces": z_traces,
        "y_traces": y_traces,
        "tdr": tdr,
        "epsilon_eff": float(epsilon_eff),
        "markers": markers,
        "pairs": [pair["name"] for pair in pairs or []],
        "mixed_note": mixed_note,
        "passivity_excess": excess,
    }


PASSIVITY_TOLERANCE = 0.02  # FDTD leaves about a percent; more is not physics


def passivity_excess(s: np.ndarray) -> Dict:
    """The worst excess over 1 of sum_i |S_ij|^2 for driven ports, and where; {} within PASSIVITY_TOLERANCE."""
    if s.ndim != 3 or not s.shape[1]:
        return {}
    worst = {}
    for j in range(s.shape[1]):
        if not np.any(s[:, j, j]):
            continue  # not driven: filled by reciprocity
        power = (np.abs(s[:, :, j]) ** 2).sum(axis=1)
        k = int(np.argmax(power))
        if power[k] - 1.0 > max(PASSIVITY_TOLERANCE, worst.get("excess", 0.0)):
            worst = {"excess": float(power[k] - 1.0), "port": j + 1, "index": k}
    return worst


def impedance_traces(s: np.ndarray, z0_ports, traces: Dict[str, Dict]) -> Dict[str, Dict]:
    """Z-matrix terms when every port was driven, Z = G (I + S)(I - S)^-1 G; else each driven port's Zin<k>."""
    n = s.shape[1] if s.ndim == 3 else 0
    out: Dict[str, Dict] = {}
    # A port was driven if its reflection isn't exactly zero.
    if n and all(np.any(s[:, j, j]) for j in range(n)):
        eye = np.eye(n)
        g = np.diag(np.sqrt(np.asarray(z0_ports, dtype=float)))
        m = eye - s
        # Leave frequencies where I - S is (nearly) singular blank.
        good = np.linalg.cond(m) < 1e9
        z = None
        if good.any():
            inverse = np.full_like(m, np.nan)
            inverse[good] = np.linalg.inv(m[good])
            z = g @ (eye + s) @ inverse @ g
        if z is not None:
            for i in range(n):
                for j in range(n):
                    zij = z[:, i, j]
                    out[f"Z{i + 1}{j + 1}"] = {"z_real": zij.real.tolist(), "z_imag": zij.imag.tolist(),
                                               "z_mag": np.abs(zij).tolist()}
            return out
    for key, trace in traces.items():
        digits = key[1:]
        half = len(digits) // 2
        if "z_real" in trace and digits[:half] == digits[half:]:
            real, imag = np.asarray(trace["z_real"]), np.asarray(trace["z_imag"])
            out[f"Zin{digits[:half]}"] = {"z_real": trace["z_real"], "z_imag": trace["z_imag"],
                                          "z_mag": np.hypot(real, imag).tolist()}
    return out


def admittance_traces(s: np.ndarray, z0_ports, z_traces: Dict[str, Dict]) -> Dict[str, Dict]:
    """Y-matrix terms when every port was driven, Y = G^-1 (I - S)(I + S)^-1 G^-1; else each driven port's Yin<k>."""
    n = s.shape[1] if s.ndim == 3 else 0
    out: Dict[str, Dict] = {}
    if n and all(np.any(s[:, j, j]) for j in range(n)):
        eye = np.eye(n)
        g_inv = np.diag(1 / np.sqrt(np.asarray(z0_ports, dtype=float)))
        m = eye + s
        # Leave frequencies where I + S is (nearly) singular blank.
        good = np.linalg.cond(m) < 1e9
        if good.any():
            inverse = np.full_like(m, np.nan)
            inverse[good] = np.linalg.inv(m[good])
            y = g_inv @ (eye - s) @ inverse @ g_inv
            for i in range(n):
                for j in range(n):
                    yij = y[:, i, j]
                    out[f"Y{i + 1}{j + 1}"] = {"real": yij.real.tolist(), "imag": yij.imag.tolist()}
        return out
    for key, trace in z_traces.items():
        if key.startswith("Zin"):
            z = np.asarray(trace["z_real"], dtype=float) + 1j * np.asarray(trace["z_imag"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.where(np.abs(z) > 1e-12, 1 / z, np.nan)
            out[f"Yin{key[3:]}"] = {"real": y.real.tolist(), "imag": y.imag.tolist()}
    return out


def _bandwidth_at(freqs: np.ndarray, s_db: np.ndarray, threshold_db: float) -> Optional[float]:
    """Contiguous bandwidth around the best match where |S11| stays below threshold."""
    below = s_db < threshold_db
    if not below.any():
        return None
    best = int(np.argmin(s_db))
    if not below[best]:
        return None
    lo = best
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = best
    while hi < len(below) - 1 and below[hi + 1]:
        hi += 1
    return float(freqs[hi] - freqs[lo])


def read_farfield(path: Path) -> Optional[Dict]:
    """Far-field pattern written by run_worker (see nf2ff.py)."""
    import json

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
