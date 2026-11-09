"""S-parameter (ΔS) convergence stop for a running openEMS simulation.

ADDED (EMShrimulator Studio). Tails the port probe files, and once max |S(t) - S(t - window)| stays below
`delta_s` on consecutive checks, writes the `ABORT` file openEMS polls for.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONSECUTIVE = 2
FREQUENCY_POINTS = 41
POLL_SECONDS = 1.0
ABORT_FILE = "ABORT"


@dataclass
class PortProbes:
    """The probe files that give one port's total voltage and current."""

    voltage: List[str]
    current: List[str]
    z_ref: float
    current_weight: float = 1.0  # 0.5 when two current probes are averaged (microstrip)


class ProbeReader:
    """Incrementally reads one openEMS probe file and accumulates its DFT up to a time horizon."""

    def __init__(self, path: str, freqs: np.ndarray):
        self.path = path
        self.freqs = freqs
        self.offset = 0
        self.tail = b""
        self.pending_t = np.zeros(0)
        self.pending_v = np.zeros(0)
        self.sum = np.zeros(len(freqs), complex)
        self.horizon = -math.inf  # latest time accumulated into `sum`

    def poll(self) -> None:
        """Read whatever has been flushed since the last poll."""
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except OSError:
            return
        if not chunk:
            return
        self.offset += len(chunk)
        data = self.tail + chunk
        cut = data.rfind(b"\n")
        if cut < 0:
            self.tail = data
            return
        self.tail = data[cut + 1:]
        times, values = [], []
        for line in data[:cut].splitlines():
            line = line.strip()
            if not line or line.startswith(b"%"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                t, v = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if t <= (times[-1] if times else self.latest):
                continue  # keeps the samples strictly increasing for advance()'s search
            times.append(t)
            values.append(v)
        if times:
            self.pending_t = np.concatenate([self.pending_t, times])
            self.pending_v = np.concatenate([self.pending_v, values])

    @property
    def latest(self) -> float:
        if len(self.pending_t):
            return float(self.pending_t[-1])
        return self.horizon

    def advance(self, horizon: float) -> None:
        """Fold every pending sample up to `horizon` into the running transform."""
        n = int(np.searchsorted(self.pending_t, horizon, side="right"))
        if n:
            t, v = self.pending_t[:n], self.pending_v[:n]
            # sum over samples of v * exp(-j 2 pi f t); the dt factor cancels in S.
            self.sum += np.exp(-2j * np.pi * np.outer(self.freqs, t)) @ v
            self.pending_t, self.pending_v = self.pending_t[n:], self.pending_v[n:]
        self.horizon = max(self.horizon, horizon)


@dataclass
class _Port:
    probes: PortProbes
    voltage: List[ProbeReader] = field(default_factory=list)
    current: List[ProbeReader] = field(default_factory=list)

    def readers(self) -> List[ProbeReader]:
        return self.voltage + self.current

    def waves(self) -> Tuple[np.ndarray, np.ndarray]:
        """(incident, reflected) voltage waves, as openEMS's Port.CalcPort defines them."""
        u = sum(r.sum for r in self.voltage)
        i = sum(r.sum for r in self.current) * self.probes.current_weight
        z = self.probes.z_ref
        return 0.5 * (u + i * z), 0.5 * (u - i * z)


def pulse_length_s(f_start: float, f_stop: float) -> float:
    """Length of openEMS's Gaussian excitation for SetGaussExcite((f1+f2)/2, (f2-f1)/2)."""
    fc = max((f_stop - f_start) / 2, 1.0)
    return 9.0 / (math.pi * fc)


class DeltaSMonitor(threading.Thread):
    """Watches a run in `sim_dir` and stops it once S has converged to `delta_s`."""

    def __init__(self, sim_dir: str, ports: Sequence[Optional[PortProbes]], excited: int,
                 f_start: float, f_stop: float, delta_s: float, t_min: float, window: float):
        super().__init__(name="delta-s-monitor", daemon=True)
        self.sim_dir = sim_dir
        self.excited = excited
        self.delta_s = delta_s
        self.t_min = t_min
        self.window = window
        self.freqs = np.linspace(f_start, f_stop, FREQUENCY_POINTS)
        self._halt = threading.Event()
        self.ports: Dict[int, _Port] = {}
        for index, probes in enumerate(ports):
            if probes is None:
                continue
            port = _Port(probes)
            port.voltage = [ProbeReader(os.path.join(sim_dir, n), self.freqs) for n in probes.voltage]
            port.current = [ProbeReader(os.path.join(sim_dir, n), self.freqs) for n in probes.current]
            self.ports[index] = port
        self.previous: Optional[Dict[int, np.ndarray]] = None
        self.last_check = -math.inf
        self.streak = 0
        self.history: List[Tuple[float, float]] = []  # (simulated time, max |ΔS|)
        self.converged = False

    # ---------------------------------------------------------------- core

    def s_params(self) -> Dict[int, np.ndarray]:
        incident, _ = self.ports[self.excited].waves()
        scale = np.abs(incident)
        valid = scale > 1e-3 * max(float(scale.max(initial=0.0)), 1e-300)
        out = {}
        for index, port in self.ports.items():
            _, reflected = port.waves()
            s = np.full(len(self.freqs), np.nan, complex)
            s[valid] = reflected[valid] / incident[valid]
            out[index] = s
        return out

    def step(self) -> Optional[float]:
        """Poll the files and, if a window has passed, run one check. Returns max |ΔS| when it checked."""
        readers = [r for port in self.ports.values() for r in port.readers()]
        for reader in readers:
            reader.poll()
        if not readers or self.excited not in self.ports:
            return None
        horizon = min(r.latest for r in readers)
        if not math.isfinite(horizon) or horizon - self.last_check < self.window:
            return None
        for reader in readers:
            reader.advance(horizon)
        self.last_check = horizon

        current = self.s_params()
        previous, self.previous = self.previous, current
        if previous is None:
            return None
        comparable = [i for i in current
                      if (np.isfinite(current[i]) & np.isfinite(previous[i])).any()]
        if not comparable:
            # No incident wave yet at one of the two checks: nothing to compare.
            return None
        worst = float(max(np.nanmax(np.abs(current[i] - previous[i])) for i in comparable))
        self.history.append((horizon, worst))
        if horizon < self.t_min:
            logger.info("ΔS check at %.3f ns: max |ΔS| = %.4g (waiting for the pulse to cross the board, %.3f ns)",
                        horizon * 1e9, worst, self.t_min * 1e9)
            self.streak = 0
            return worst
        self.streak = self.streak + 1 if worst < self.delta_s else 0
        logger.info("ΔS check at %.3f ns: max |ΔS| = %.4g (target %.4g, %d/%d)",
                    horizon * 1e9, worst, self.delta_s, self.streak, CONSECUTIVE)
        if self.streak >= CONSECUTIVE:
            self.converged = True
        return worst

    # -------------------------------------------------------------- thread

    def run(self) -> None:
        try:
            while not self._halt.wait(POLL_SECONDS):
                self.step()
                if self.converged:
                    logger.info("S-parameters converged (ΔS < %.4g); stopping the solver", self.delta_s)
                    with open(os.path.join(self.sim_dir, ABORT_FILE), "w", encoding="utf-8") as f:
                        f.write("converged: delta S\n")
                    return
        except Exception:  # noqa: BLE001 - the monitor must never take the run down
            logger.exception("ΔS monitor failed; the run continues to its energy or timestep limit")

    def stop(self) -> None:
        self._halt.set()

    def cleanup(self) -> None:
        """Stop and remove the ABORT file, so nothing later mistakes it for a user abort."""
        self.stop()
        self.join(timeout=5)
        try:
            os.remove(os.path.join(self.sim_dir, ABORT_FILE))
        except OSError:
            pass
