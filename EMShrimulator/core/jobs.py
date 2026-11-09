"""Background subprocess jobs: log tailing, progress parsing and whole-tree cancellation."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_jobs_lock = threading.Lock()
_jobs: Dict[str, "Job"] = {}
MAX_JOBS_KEPT = 200

# openEMS prints e.g. "[@   12s] Timestep   3400 of 100000 (3.4%) ..."
_OPENEMS_PROGRESS = re.compile(r"Timestep\s+(\d+)\s+of\s+(\d+)")
_OPENEMS_PERCENT = re.compile(r"\(\s*([\d.]+)%\)")
_GEMS_LINE = re.compile(r"^@@GEMS (\{.*\})\s*$", re.MULTILINE)
# gerber2ems logs its pipeline stages through coloredlogs
_G2E_STAGE = re.compile(r"\[(?:INFO|DEBU)\]\s+(.*)$")


@dataclass
class Job:
    id: str
    kind: str  # "modeler" | "layout"
    project_name: str
    cmd: List[str]
    cwd: Path
    log_path: Path
    label: str = ""
    env: Optional[Dict[str, str]] = None
    status: str = "pending"  # pending -> running -> done | failed | cancelled
    exit_code: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: Optional[float] = None       # 0..1
    stage: str = ""
    detail: Dict = field(default_factory=dict)
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _cancelled: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "project_name": self.project_name,
            "label": self.label,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (self.finished_at or time.time()) - self.started_at if self.started_at else None,
            "progress": self.progress,
            "stage": self.stage,
            "detail": self.detail,
            "command": " ".join(self.cmd),
        }


def _popen_kwargs() -> dict:
    """New process group (for tree kill) and no console window."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _scan_progress(job: Job, text: str) -> None:
    """Update job.progress/stage from a freshly-read chunk of log output."""
    for match in _GEMS_LINE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        event = payload.get("event")
        if event == "start":
            job.detail["runs"] = payload.get("runs")
            job.stage = "Building geometry"
        elif event == "run_begin":
            job.detail["run_index"] = payload.get("index")
            job.detail["run_total"] = payload.get("total")
            stage = f"Exciting port {payload.get('port')} ({payload.get('index')}/{payload.get('total')})"
            if payload.get("pass_total"):
                stage = f"Pass {payload.get('pass_index')}/{payload.get('pass_total')} · {stage}"
            job.stage = stage
        elif event == "pass_end":
            job.detail["delta_s"] = payload.get("delta_s")
        elif event == "mesh":
            job.detail["cells"] = payload.get("cells")
            job.stage = f"Meshed {payload.get('cells', 0):,} cells"
        elif event == "nf2ff_begin":
            job.stage = "Far-field transform"
        elif event == "complete":
            job.progress = 1.0
            job.stage = "Complete"
            job.detail["touchstone"] = payload.get("touchstone")

    ts = None
    for ts in _OPENEMS_PROGRESS.finditer(text):
        pass
    if ts:
        done, total = int(ts.group(1)), max(int(ts.group(2)), 1)
        # Scale a single solver run into its slice of the whole job.
        index = job.detail.get("run_index") or 1
        runs = job.detail.get("run_total") or job.detail.get("runs") or 1
        job.progress = min(((index - 1) + done / total) / runs, 0.999)
        job.detail["timestep"] = done
        job.detail["timesteps_total"] = total
    else:
        pct = None
        for pct in _OPENEMS_PERCENT.finditer(text):
            pass
        if pct:
            job.progress = min(float(pct.group(1)) / 100.0, 0.999)

    if job.kind == "layout":
        stage = None
        for stage in _G2E_STAGE.finditer(text):
            pass
        if stage:
            message = stage.group(1).strip()
            if message and len(message) < 120:
                job.stage = message


def _run(job: Job) -> None:
    job.status = "running"
    job.started_at = time.time()
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    with job.log_path.open("w", encoding="utf-8", errors="replace") as logf:
        logf.write(f"$ {' '.join(job.cmd)}\n  (in {job.cwd})\n\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                job.cmd, cwd=str(job.cwd), stdout=logf, stderr=subprocess.STDOUT, text=True,
                env=job.env or dict(os.environ), **_popen_kwargs(),
            )
            job._proc = proc
            proc.wait()
            job.exit_code = proc.returncode
            if job._cancelled:
                job.status = "cancelled"
            else:
                job.status = "done" if proc.returncode == 0 else "failed"
        except Exception as e:  # noqa: BLE001
            logf.write(f"\n[job runner] failed to launch subprocess: {e}\n")
            job.status = "failed"
            job.exit_code = -1
    job.finished_at = time.time()
    if job.status == "done":
        job.progress = 1.0
        job.stage = job.stage or "Complete"
    elif job.status == "cancelled":
        job.stage = "Cancelled"


def start_job(
    kind: str, project_name: str, cmd: List[str], cwd: Path, log_dir: Path,
    env: Optional[Dict[str, str]] = None, label: str = "",
) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id, kind=kind, project_name=project_name, cmd=cmd, cwd=cwd,
        log_path=log_dir / f"{job_id}.log", env=env, label=label,
    )
    with _jobs_lock:
        _jobs[job_id] = job
        if len(_jobs) > MAX_JOBS_KEPT:
            finished = sorted(
                (j for j in _jobs.values() if j.status in ("done", "failed", "cancelled")),
                key=lambda j: j.finished_at or 0,
            )
            for old in finished[: len(_jobs) - MAX_JOBS_KEPT]:
                _jobs.pop(old.id, None)
    threading.Thread(target=_run, args=(job,), daemon=True).start()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _jobs_lock:
        return _jobs.get(job_id)


def cancel_job(job_id: str) -> bool:
    job = get_job(job_id)
    if job is None or job.status not in ("pending", "running"):
        return False
    job._cancelled = True
    if job._proc is not None:
        _kill_tree(job._proc)
    return True


def read_log(job_id: str, offset: int = 0) -> tuple[str, int]:
    job = get_job(job_id)
    if job is None or not job.log_path.exists():
        return "", offset
    data = job.log_path.read_bytes()
    chunk = data[offset:]
    text = chunk.decode("utf-8", errors="replace")
    if text:
        _scan_progress(job, text)
    return text, len(data)


def list_jobs(project_name: Optional[str] = None) -> List[dict]:
    with _jobs_lock:
        jobs = list(_jobs.values())
    if project_name:
        jobs = [j for j in jobs if j.project_name == project_name]
    return [j.to_dict() for j in sorted(jobs, key=lambda j: j.started_at or 0, reverse=True)]


def active_count() -> int:
    with _jobs_lock:
        return sum(1 for j in _jobs.values() if j.status in ("pending", "running"))
