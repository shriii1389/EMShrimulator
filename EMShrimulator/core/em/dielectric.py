"""Wideband dielectric loss: a conductivity plus the fewest Debye poles that hold tan δ across the band.

Each pole costs solver time in every cell of the material, so the fit uses as few as it can.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

EDGE_DECADES = 0.5  # poles may sit this far beyond the band
LOWEST_DECADES = 2.0  # fit at most this far below the top
FIT_TOLERANCE = 0.04  # allowed tan δ error over the band
MAX_POLES = 3
CANDIDATES = 24          # pole frequencies tried, evenly in log frequency
EPS0 = 8.85418781762e-12


def centre_conductivity(eps_r: float, tan_delta: float, f0: float) -> float:
    """The "centre" model: tan δ as a conductivity, exact at f0 only."""
    return 2 * math.pi * f0 * EPS0 * eps_r * tan_delta


@dataclass
class DebyeFit:
    eps_inf: float
    delta: List[float]  # Δε of each pole
    tau: List[float]    # relaxation time of each pole, s
    kappa: float = 0.0  # conductivity, S/m

    def epsilon(self, freq_hz) -> np.ndarray:
        """Complex relative permittivity ε' − jε'' (openEMS's e^{jωt} convention)."""
        w = 2 * np.pi * np.asarray(freq_hz, dtype=float)
        poles = np.sum(np.asarray(self.delta) / (1 + 1j * w[..., None] * np.asarray(self.tau)), axis=-1)
        return self.eps_inf + poles - 1j * self.kappa / (w * EPS0)


def constant_loss(eps_r: float, tan_delta: float, f_start: float, f_stop: float, f_ref: float,
                  max_pole_hz: float = math.inf) -> DebyeFit:
    """A conductivity and the fewest Debye poles holding tan δ over [f_start, f_stop], with ε'(f_ref) = εr.

    `max_pole_hz` caps the fastest pole, since openEMS skips poles faster than three timesteps.
    """
    f_hi = max(float(f_stop), 1.0)
    f_lo = min(max(float(f_start), f_hi / 10 ** LOWEST_DECADES), f_hi / 1.25)
    kappa, delta, tau = _fit(round(eps_r * tan_delta, 15), round(f_lo), round(f_hi),
                             max_pole_hz if math.isfinite(max_pole_hz) else 0.0)
    wt_ref = 2 * np.pi * f_ref * np.asarray(tau)
    eps_inf = max(1.0, eps_r - float(np.sum(np.asarray(delta) / (1 + wt_ref ** 2))))
    return DebyeFit(eps_inf, list(delta), list(tau), kappa)


@lru_cache(maxsize=64)
def _fit(target: float, f_lo: float, f_hi: float, max_pole_hz: float) -> Tuple[float, Tuple[float, ...], Tuple[float, ...]]:
    """(κ, Δε per pole, τ per pole) fitting ε'' to `target` over [f_lo, f_hi] with 1, 2, then 3 poles."""
    freqs = np.logspace(math.log10(f_lo), math.log10(f_hi), 60)
    w = 2 * np.pi * freqs
    lo = math.log10(f_lo) - EDGE_DECADES
    hi = math.log10(f_hi) + EDGE_DECADES
    if max_pole_hz:
        hi = max(min(hi, math.log10(max_pole_hz)), lo + 0.25)
    candidates = 1 / (2 * np.pi * np.logspace(lo, hi, CANDIDATES))
    columns = (w[:, None] * candidates[None, :]) / (1 + (w[:, None] * candidates[None, :]) ** 2) / target
    conductivity = (1 / (w * EPS0)) / target
    best = None  # (error, κ, Δε, τ)
    for count in range(1, MAX_POLES + 1):
        # Every candidate set at once, by batched least squares; sets needing a negative weight are skipped.
        combos = np.array(list(itertools.combinations(range(CANDIDATES), count)))
        a = np.concatenate([np.broadcast_to(conductivity[None, :, None], (len(combos), len(freqs), 1)),
                            columns[:, combos].transpose(1, 0, 2)], axis=2)
        gram = a.transpose(0, 2, 1) @ a
        solvable = np.abs(np.linalg.det(gram)) > 1e-300
        x = np.full((len(combos), count + 1), -1.0)
        x[solvable] = np.linalg.solve(gram[solvable], a[solvable].sum(axis=1)[..., None])[..., 0]
        error = np.abs(a @ x[..., None] - 1)[..., 0].max(axis=1)
        error[(x < 0).any(axis=1)] = np.inf
        k = int(np.argmin(error))
        if np.isfinite(error[k]) and (best is None or error[k] < best[0] - 1e-12):
            best = (float(error[k]), float(x[k, 0]), tuple(float(v) for v in x[k, 1:]),
                    tuple(float(candidates[c]) for c in combos[k]))
        if best is not None and best[0] <= FIT_TOLERANCE:
            break
    if best is None:  # no passive fit: conductivity alone
        middle = 2 * np.pi * math.sqrt(f_lo * f_hi)
        return target * middle * EPS0, (), ()
    _error, kappa, delta, tau = best
    keep = [i for i, d in enumerate(delta) if d > 0]
    return kappa, tuple(delta[i] for i in keep), tuple(tau[i] for i in keep)


def add_lossy_dielectric(csx, name: str, eps_r: float, tan_delta: float, f_start: float, f_stop: float,
                         f_ref: float, mue: float = 1.0, kappa: float = 0.0):
    """A CSXCAD material with εr and a band-wide tan δ; returns the property."""
    from CSXCAD.CSProperties import CSPropDebyeMaterial

    prop = CSPropDebyeMaterial(csx.GetParameterSet(), epsilon=eps_r, mue=mue, kappa=kappa)
    prop.SetName(name)
    set_poles(prop, constant_loss(eps_r, tan_delta, f_start, f_stop, f_ref), base_kappa=kappa)
    csx.AddProperty(prop)  # returns None, unlike AddMaterial
    return prop


def set_poles(prop, fit: DebyeFit, base_kappa: float = 0.0) -> None:
    """Set a fit's ε∞, conductivity and poles on a Debye material."""
    prop.SetMaterialProperty(epsilon=fit.eps_inf, kappa=base_kappa + fit.kappa)
    prop.SetDispersionOrder(len(fit.delta))
    for k, (delta, tau) in enumerate(zip(fit.delta, fit.tau)):
        prop.SetDispersiveMaterialProperty(k, eps_delta=delta, eps_relax=tau)


# --- faces of the material ---
# openEMS averages ε and σ at dielectric/air faces but not its Debye poles, so pole weights are halved on
# faces and quartered on edges.

Box = Tuple[Sequence[float], Sequence[float]]


def _num(v: float) -> str:
    return repr(float(v))


def face_weight(boxes: Sequence[Box], touching: Sequence[Box], tol: float) -> str:
    """Weight expression in x, y, z (drawing units) for the poles of a material made of `boxes`."""
    names = "xyz"
    terms = []
    for lo, hi in boxes:
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        for a in range(3):
            if hi[a] - lo[a] <= tol:
                continue  # a flat box has no inside to average against
            b, c = (a + 1) % 3, (a + 2) % 3
            for side, plane in ((-1, lo[a]), (1, hi[a])):
                rect = (f"(abs({names[a]}-{_num(plane)})<={_num(tol)})"
                        f"*({names[b]}>={_num(lo[b] - tol)})*({names[b]}<={_num(hi[b] + tol)})"
                        f"*({names[c]}>={_num(lo[c] - tol)})*({names[c]}<={_num(hi[c] + tol)})")
                covered = []
                for olo, ohi in touching:
                    olo, ohi = np.minimum(olo, ohi), np.maximum(olo, ohi)
                    if abs((olo[a] if side > 0 else ohi[a]) - plane) > tol:
                        continue  # not across this face
                    blo, bhi = max(lo[b], olo[b]), min(hi[b], ohi[b])
                    clo, chi = max(lo[c], olo[c]), min(hi[c], ohi[c])
                    if bhi - blo > tol and chi - clo > tol:
                        covered.append(f"(1-({names[b]}>={_num(blo - tol)})*({names[b]}<={_num(bhi + tol)})"
                                       f"*({names[c]}>={_num(clo - tol)})*({names[c]}<={_num(chi + tol)}))")
                terms.append(f"(1-0.5*{rect}" + "".join(f"*{c}" for c in covered) + ")")
    return "*".join(terms) if terms else "1"


def weight_poles(prop, expression: str) -> None:
    """Apply a weight expression to every pole's Δε."""
    for k in range(prop.GetDispersionOrder()):
        prop.SetDispersiveMaterialWeight(k, eps_delta=expression)
