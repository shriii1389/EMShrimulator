"""NF2FF post-processing: the solver reduces openEMS's transform; the app derives every far-field report from it."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ETA0 = 376.730313668  # free-space wave impedance, Ω
FIELDS_FILE = "farfield_fields.npz"
CONSISTENCY_TOLERANCE = 0.15  # far- vs near-field power mismatch above which the box is suspect
AXIAL_RATIO_CAP_DB = 60.0     # a linear wave's axial ratio is infinite; plots stop here

POLARISATIONS: Dict[str, str] = {
    "total": "Total", "theta": "θ", "phi": "φ", "rhcp": "RHCP", "lhcp": "LHCP", "l3x": "Ludwig-3 X",
    "l3y": "Ludwig-3 Y",
}
# The cross-polarisation of each polarisation.
CROSS: Dict[str, str] = {"theta": "phi", "phi": "theta", "rhcp": "lhcp", "lhcp": "rhcp", "l3x": "l3y", "l3y": "l3x"}
QUANTITIES: Dict[str, str] = {"re": "rE", "directivity": "Directivity", "gain": "Gain",
                              "realized_gain": "Realized gain", "axial_ratio": "Axial ratio"}


# ------------------------------------------------------------------ solver side

def sphere_weights(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Solid angle (sr) of each (θ, φ) grid point; sums to 4π over a full sphere."""
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    wt = np.zeros_like(theta)
    if theta.size > 1:
        d = np.diff(theta)
        wt[:-1] += d / 2
        wt[1:] += d / 2
    wp = np.zeros_like(phi)
    if phi.size > 1:
        d = np.diff(phi)
        wp[:-1] += d / 2
        wp[1:] += d / 2
        if abs(phi[-1] - phi[0] - 2 * np.pi) < 1e-6:
            # Closed: the φ = 360° repeat column counts nothing.
            wp[0] += wp[-1]
            wp[-1] = 0.0
    return np.outer(np.sin(theta) * wt, wp)


def compute(nf2ff_box, sim_path: str, freqs: Sequence[float], theta_step_deg: float,
            phi_step_deg: float, center_mm=(0, 0, 0)) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Run the NF2FF transform and reduce it to the summary (farfield.json) and field arrays (farfield_fields.npz).

    Directivity is 4πU / ∮U dΩ; `consistency` compares the far-field power with the box's Poynting flux.
    """
    theta = np.arange(0.0, 180.0 + theta_step_deg / 2, theta_step_deg)
    phi = np.arange(0.0, 360.0 + phi_step_deg / 2, phi_step_deg)
    center_m = [float(v) * 1e-3 for v in center_mm]
    res = nf2ff_box.CalcNF2FF(sim_path, np.array(freqs, dtype=float), theta, phi,
                              center=center_m, read_cached=False, verbose=0)
    radius = float(np.atleast_1d(res.r)[0]) if getattr(res, "r", None) is not None else 1.0
    theta_arr = np.degrees(res.theta) if float(np.max(res.theta)) <= 2 * np.pi + 1e-6 else np.asarray(res.theta)
    phi_arr = np.degrees(res.phi) if float(np.max(res.phi)) <= 2 * np.pi + 1e-6 else np.asarray(res.phi)
    weights = sphere_weights(theta_arr, phi_arr)

    out_freqs: List[Dict] = []
    kept: List[int] = []
    for n, f in enumerate(np.atleast_1d(res.freq)):
        e_theta, e_phi = np.asarray(res.E_theta[n]), np.asarray(res.E_phi[n])
        intensity = (np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2) * radius ** 2 / (2 * ETA0)  # U, W/sr
        prad_ff = float((intensity * weights).sum())
        prad_box = float(np.atleast_1d(res.Prad)[n])
        if prad_ff <= 0:
            continue
        d_lin = 4 * np.pi * intensity / prad_ff
        entry = summarise(float(f), theta_arr, phi_arr, d_lin)
        entry.update(radiated_power_w=prad_box, radiated_power_ff_w=prad_ff,
                     consistency=prad_ff / prad_box if prad_box > 0 else None,
                     center_mm=[float(v) for v in center_mm])
        out_freqs.append(entry)
        kept.append(n)

    fields = {
        "frequency_hz": np.asarray([float(np.atleast_1d(res.freq)[n]) for n in kept]),
        "theta_deg": theta_arr.astype(float), "phi_deg": phi_arr.astype(float), "radius_m": np.array([radius]),
        "e_theta": np.asarray([res.E_theta[n] for n in kept], dtype=np.complex64).reshape(
            len(kept), theta_arr.size, phi_arr.size),
        "e_phi": np.asarray([res.E_phi[n] for n in kept], dtype=np.complex64).reshape(
            len(kept), theta_arr.size, phi_arr.size),
    }
    return {"frequencies": out_freqs}, fields


def summarise(frequency_hz: float, theta_arr: np.ndarray, phi_arr: np.ndarray, d_lin: np.ndarray) -> Dict:
    """One frequency's directivity grid (linear) reduced to what farfield.json holds."""
    d_dbi = 10 * np.log10(np.maximum(d_lin, 1e-12))

    def cut_at_phi(target_deg: float) -> List[float]:
        idx = int(np.argmin(np.abs(phi_arr - target_deg)))
        return d_dbi[:, idx].tolist()

    i_max = np.unravel_index(int(np.argmax(d_dbi)), d_dbi.shape)
    peak_theta = float(theta_arr[i_max[0]])
    peak_phi = float(phi_arr[i_max[1]]) % 360.0  # the seam's 360° is 0°
    # front-to-back: the peak against the direction opposite it
    back_theta_idx = int(np.argmin(np.abs(theta_arr - (180.0 - peak_theta))))
    back_phi_idx = int(np.argmin(np.abs(phi_arr - ((peak_phi + 180.0) % 360.0))))
    return {
        "frequency_hz": frequency_hz,
        "theta_deg": theta_arr.tolist(),
        "phi_deg": phi_arr.tolist(),
        "directivity_dbi": d_dbi.tolist(),
        "e_plane_dbi": cut_at_phi(0.0),
        "h_plane_dbi": cut_at_phi(90.0),
        "dmax_dbi": float(d_dbi[i_max]),
        "peak_theta_deg": peak_theta,
        "peak_phi_deg": peak_phi,
        "front_to_back_db": float(d_dbi[i_max] - d_dbi[back_theta_idx, back_phi_idx]),
        "hpbw_e_deg": _hpbw(theta_arr, np.array(cut_at_phi(0.0))),
        "hpbw_h_deg": _hpbw(theta_arr, np.array(cut_at_phi(90.0))),
    }


def _hpbw(angles_deg: np.ndarray, pattern_dbi: np.ndarray) -> float | None:
    """3 dB beamwidth around the pattern's peak, in degrees."""
    if len(pattern_dbi) < 3:
        return None
    peak_idx = int(np.argmax(pattern_dbi))
    threshold = pattern_dbi[peak_idx] - 3.0
    lo = peak_idx
    while lo > 0 and pattern_dbi[lo] >= threshold:
        lo -= 1
    hi = peak_idx
    while hi < len(pattern_dbi) - 1 and pattern_dbi[hi] >= threshold:
        hi += 1
    if lo == 0 and hi == len(pattern_dbi) - 1:
        return None
    return float(abs(angles_deg[hi] - angles_deg[lo]))


def write(payload: Dict, path: Path, fields: Optional[Dict[str, np.ndarray]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if fields is not None:
        np.savez_compressed(path.parent / FIELDS_FILE, **fields)
    return path


def read_fields(results_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """The E_θ/E_φ a run saved, or None for runs before they were saved."""
    path = Path(results_dir) / FIELDS_FILE
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


# ------------------------------------------------------------------ app side

def field_index(fields: Optional[Dict[str, np.ndarray]], frequency_hz: float) -> Optional[int]:
    if fields is None or not len(fields["frequency_hz"]):
        return None
    index = int(np.argmin(np.abs(fields["frequency_hz"] - frequency_hz)))
    return index if abs(fields["frequency_hz"][index] - frequency_hz) <= 1e-6 * max(frequency_hz, 1.0) else None


def polarised(e_theta: np.ndarray, e_phi: np.ndarray, phi_deg: np.ndarray, component: str) -> np.ndarray:
    """The complex far-field component of a polarisation, (θ, φ); 'total' is a magnitude."""
    phi = np.radians(np.asarray(phi_deg, dtype=float))[None, :]
    if component == "theta":
        return e_theta
    if component == "phi":
        return e_phi
    if component == "rhcp":
        return (e_theta + 1j * e_phi) / math.sqrt(2)
    if component == "lhcp":
        return (e_theta - 1j * e_phi) / math.sqrt(2)
    if component == "l3x":
        return e_theta * np.cos(phi) - e_phi * np.sin(phi)
    if component == "l3y":
        return e_theta * np.sin(phi) + e_phi * np.cos(phi)
    return np.sqrt(np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)


def axial_ratio_db(e_theta: np.ndarray, e_phi: np.ndarray) -> np.ndarray:
    """Axial ratio in dB: 0 for circular, large for linear."""
    right = np.abs(e_theta + 1j * e_phi)
    left = np.abs(e_theta - 1j * e_phi)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (right + left) / np.abs(right - left)
    return np.clip(np.nan_to_num(20 * np.log10(ratio), nan=AXIAL_RATIO_CAP_DB, posinf=AXIAL_RATIO_CAP_DB),
                   0.0, AXIAL_RATIO_CAP_DB)


def per_watt(entry: Dict) -> float:
    """Scale from openEMS's excitation to 1 W incident: 1/√P_incident, or 1 if not recorded."""
    incident = entry.get("incident_power_w")
    return 1.0 / math.sqrt(float(incident)) if incident and float(incident) > 0 else 1.0


def power_offset(entry: Dict, quantity: str) -> Optional[float]:
    """Gain over directivity (linear) for the quantity; None if not recorded."""
    if quantity in ("directivity", "re", "axial_ratio"):
        return 1.0
    prad = entry.get("radiated_power_w")
    key = "accepted_power_w" if quantity == "gain" else "incident_power_w"
    power = entry.get(key)
    if not prad or not power or power <= 0:
        return None
    return float(prad) / float(power)


def pattern(entry: Dict, fields: Optional[Dict[str, np.ndarray]], quantity: str, component: str,
            db: bool = True) -> Tuple[Optional[np.ndarray], str, str]:
    """One frequency's pattern as a (θ, φ) grid, its unit, and why it is unavailable ('' when it is)."""
    offset = power_offset(entry, quantity)
    if offset is None:
        return None, "", (f"{QUANTITIES[quantity]} needs the power at the port, which this run did not record. "
                          "Run the design again.")
    index = field_index(fields, float(entry["frequency_hz"]))
    if index is None:
        if quantity in ("re", "axial_ratio") or component != "total":
            return None, "", (f"{QUANTITIES[quantity]}{'' if component == 'total' else ' ' + POLARISATIONS[component]}"
                              " needs the far field's θ and φ components, which runs before this version did not "
                              "save. Run the design again.")
        d_db = np.asarray(entry["directivity_dbi"], dtype=float) + 10 * math.log10(offset)
        return (d_db, "dBi", "") if db else (10 ** (d_db / 10), "", "")
    e_theta, e_phi = fields["e_theta"][index].astype(complex), fields["e_phi"][index].astype(complex)
    if quantity == "axial_ratio":
        return axial_ratio_db(e_theta, e_phi), "dB", ""
    field = np.abs(polarised(e_theta, e_phi, fields["phi_deg"], component))
    if quantity == "re":
        # rE for 1 W incident at the port, when recorded.
        values = float(fields["radius_m"][0]) * field * per_watt(entry)
        return (20 * np.log10(np.maximum(values, 1e-30)), "dBV", "") if db else (values, "V", "")
    # A polarisation's share of the intensity over the total's mean.
    total = np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2
    mean = float((total * sphere_weights(fields["theta_deg"], fields["phi_deg"])).sum()) / (4 * np.pi)
    if mean <= 0:
        return None, "", "The far field is zero at this frequency."
    values = field ** 2 / mean * offset
    return (10 * np.log10(np.maximum(values, 1e-12)), "dBi", "") if db else (values, "", "")


def cut(grid: np.ndarray, theta_deg: np.ndarray, phi_deg: np.ndarray, sweep: str, at_deg: float
        ) -> Tuple[np.ndarray, np.ndarray, float]:
    """A θ cut at the nearest φ (spanning −180°…180°) or a φ cut at the nearest θ: (angles, values, angle held)."""
    theta_deg, phi_deg = np.asarray(theta_deg, dtype=float), np.asarray(phi_deg, dtype=float)
    if sweep == "phi":
        i = int(np.argmin(np.abs(theta_deg - at_deg)))
        return phi_deg, grid[i, :], float(theta_deg[i])

    def nearest_phi(target: float) -> int:
        return int(np.argmin(np.abs((phi_deg - target + 180.0) % 360.0 - 180.0)))

    j = nearest_phi(at_deg)
    held = float(phi_deg[j])
    k = nearest_phi(held + 180.0)
    back = grid[::-1, k][:-1] if theta_deg[0] == 0 else grid[::-1, k]
    angles = np.concatenate([-theta_deg[::-1][:back.size], theta_deg])
    return angles, np.concatenate([back, grid[:, j]]), held


def antenna_parameters(entries: List[Dict], fields: Optional[Dict[str, np.ndarray]]) -> Dict[str, Dict]:
    """Antenna parameters per far-field frequency: key -> {label, unit, values}, for 1 W incident."""

    def watts(key: str):
        def get(e):
            value, incident = e.get(key), e.get("incident_power_w")
            if value is None:
                return None
            return float(value) / float(incident) if incident and float(incident) > 0 else float(value)
        return get

    rows = {
        "peak_directivity": ("Peak directivity", "dBi", lambda e: e.get("dmax_dbi")),
        "peak_gain": ("Peak gain", "dBi", lambda e: e.get("gain_dbi")),
        "peak_realized_gain": ("Peak realized gain", "dBi", lambda e: e.get("realized_gain_dbi")),
        "efficiency": ("Radiation efficiency", "%", lambda e: None if e.get("efficiency") is None
                       else 100 * float(e["efficiency"])),
        "radiated_power": ("Radiated power", "W", watts("radiated_power_w")),
        "accepted_power": ("Accepted power", "W", watts("accepted_power_w")),
        "incident_power": ("Incident power", "W", watts("incident_power_w")),
        "front_to_back": ("Front-to-back ratio", "dB", lambda e: e.get("front_to_back_db")),
        "hpbw_e": ("Beamwidth, φ = 0° (−3 dB)", "°", lambda e: e.get("hpbw_e_deg")),
        "hpbw_h": ("Beamwidth, φ = 90° (−3 dB)", "°", lambda e: e.get("hpbw_h_deg")),
        "peak_theta": ("Peak direction θ", "°", lambda e: e.get("peak_theta_deg")),
        "peak_phi": ("Peak direction φ", "°", lambda e: None if e.get("peak_phi_deg") is None
                     else float(e["peak_phi_deg"]) % 360.0),
    }
    out = {key: {"label": label, "unit": unit, "values": [get(e) for e in entries]}
           for key, (label, unit, get) in rows.items()}
    axial = []
    for e in entries:
        index = field_index(fields, float(e["frequency_hz"]))
        if index is None:
            axial.append(None)
            continue
        theta, phi = np.asarray(e["theta_deg"]), np.asarray(e["phi_deg"])
        i = int(np.argmin(np.abs(theta - float(e["peak_theta_deg"]))))
        j = int(np.argmin(np.abs(phi - float(e["peak_phi_deg"]))))
        ratio = axial_ratio_db(fields["e_theta"][index][i:i + 1, j:j + 1].astype(complex),
                               fields["e_phi"][index][i:i + 1, j:j + 1].astype(complex))
        axial.append(float(ratio[0, 0]))
    out["axial_ratio"] = {"label": "Axial ratio at the peak", "unit": "dB", "values": axial}
    return out
