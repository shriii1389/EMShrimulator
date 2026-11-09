"""On-disk project storage: projects/<name>/{kind.txt, project.json, fab/, jobs/, results/}."""
from __future__ import annotations

import io
import json
import ntpath
import re
import shutil
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, List, Literal, Optional, Union

from EMShrimulator.config import projects_dir

if TYPE_CHECKING:
    from EMShrimulator.core.em.model import ModelerProject
    from EMShrimulator.core.layout.model import LayoutProject

ProjectKind = Literal["modeler", "layout"]
AnyProject = Union["ModelerProject", "LayoutProject"]

# Parentheses allowed: imports de-duplicate as "Name (2)".
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()\-]{0,63}$")


class InvalidName(ValueError):
    pass


def check_name(name: str) -> str:
    """Reject names that could escape the projects folder."""
    name = name.strip()
    if not _NAME_RE.match(name) or ".." in name:
        raise InvalidName(
            "Project names may use letters, digits, spaces, dot, dash and underscore, "
            "must start with a letter or digit, and can be at most 64 characters."
        )
    return name


def project_dir(name: str) -> Path:
    resolved = (projects_dir() / check_name(name)).resolve()
    if projects_dir().resolve() not in resolved.parents:
        raise InvalidName("project path escapes the projects directory")
    return resolved


def _describe(d: Path) -> dict:
    kind = (d / "kind.txt").read_text().strip()
    info = {"name": d.name, "kind": kind, "modified": (d / "project.json").stat().st_mtime
            if (d / "project.json").exists() else d.stat().st_mtime}
    try:
        data = json.loads((d / "project.json").read_text(encoding="utf-8"))
        info["description"] = data.get("description", "")
        if kind == "modeler":
            info["objects"] = len(data.get("primitives", []))
            info["ports"] = len(data.get("ports", []))
        else:
            info["objects"] = len(data.get("stackup", []))
            info["ports"] = len(data.get("ports", []))
    except Exception:  # noqa: BLE001 - a listing must never fail on one bad project
        info.setdefault("description", "")
    results = d / "results"
    ems_results = d / "ems" / "results"
    info["has_results"] = (results.exists() and any(results.glob("*.s*p"))) or \
                          (ems_results.exists() and any(ems_results.glob("*.csv")))
    return info


def list_projects() -> List[dict]:
    out = []
    if not projects_dir().exists():
        return out
    for d in sorted(projects_dir().iterdir()):
        if d.is_dir() and (d / "kind.txt").exists():
            out.append(_describe(d))
    return sorted(out, key=lambda p: p.get("modified", 0), reverse=True)


def exists(name: str) -> bool:
    try:
        return (project_dir(name) / "kind.txt").exists()
    except InvalidName:
        return False


def create_project(name: str, kind: ProjectKind, project: Optional[AnyProject] = None) -> AnyProject:
    d = project_dir(name)
    if d.exists():
        raise FileExistsError(f"project '{name}' already exists")
    d.mkdir(parents=True)
    (d / "jobs").mkdir()
    (d / "results").mkdir()
    (d / "kind.txt").write_text(kind)
    if project is None:
        project = _schema(kind)(name=name)
    if kind == "layout":
        (d / "fab").mkdir(exist_ok=True)
    save_project(name, project)
    return project


def get_kind(name: str) -> ProjectKind:
    kind_file = project_dir(name) / "kind.txt"
    if not kind_file.exists():
        raise FileNotFoundError(f"no such project '{name}'")
    return kind_file.read_text().strip()  # type: ignore[return-value]


def _schema(kind: str):
    if kind == "modeler":
        from EMShrimulator.core.em.model import ModelerProject

        return ModelerProject
    from EMShrimulator.core.layout.model import LayoutProject

    return LayoutProject


def load_project(name: str) -> AnyProject:
    kind = get_kind(name)
    data = (project_dir(name) / "project.json").read_text(encoding="utf-8")
    return _schema(kind).model_validate_json(data)


def save_project(name: str, project: AnyProject) -> None:
    d = project_dir(name)
    path = d / "project.json"
    # Write-then-replace so a crash never leaves a half-written file.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def delete_project(name: str) -> None:
    d = project_dir(name)
    if d.exists():
        shutil.rmtree(d)


def duplicate_project(name: str, new_name: str) -> dict:
    src = project_dir(name)
    dst = project_dir(new_name)
    if dst.exists():
        raise FileExistsError(f"project '{new_name}' already exists")
    if not src.exists():
        raise FileNotFoundError(f"no such project '{name}'")
    # Results and job logs belong to the original run, not the copy.
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("results", "jobs", "ems", "geometry"))
    (dst / "jobs").mkdir(exist_ok=True)
    (dst / "results").mkdir(exist_ok=True)
    project = load_project(new_name)
    project.name = new_name
    save_project(new_name, project)
    return _describe(dst)


def rename_project(name: str, new_name: str) -> dict:
    src = project_dir(name)
    dst = project_dir(new_name)
    if dst.exists():
        raise FileExistsError(f"project '{new_name}' already exists")
    src.rename(dst)
    project = load_project(new_name)
    project.name = new_name
    save_project(new_name, project)
    return _describe(dst)


def _is_solver_scratch(relative: Path) -> bool:
    """Large regenerated solver data and preview caches, excluded from exports."""
    parts = relative.parts
    if parts and parts[0] == "cache":
        return True
    if any(p.startswith("sim_port_") for p in parts):
        return True
    if len(parts) >= 2 and parts[0] == "ems" and parts[1] in ("simulation", "geometry"):
        return True
    return parts and parts[0] == "jobs" and relative.suffix == ".log"


def export_archive(name: str, include_results: bool = True) -> bytes:
    """Zip the project (forward-slash member names): inputs always, results by default."""
    d = project_dir(name)
    if not d.exists():
        raise FileNotFoundError(f"no such project '{name}'")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(d.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(d)
            if _is_solver_scratch(relative):
                continue
            if not include_results and relative.parts and relative.parts[0] in ("results", "ems"):
                continue
            zf.write(path, arcname="/".join((name, *relative.parts)))
    return buf.getvalue()


def import_archive(data: bytes, name_hint: Optional[str] = None) -> dict:
    """Restore a project from an `export_archive` zip."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n.replace("\\", "/") for n in zf.namelist()]
        original = dict(zip(names, zf.namelist()))
        kind_entries = [n for n in names if n.endswith("kind.txt")]
        if not kind_entries:
            raise ValueError("this zip has no kind.txt — it is not a EMShrimulator project export")
        root = kind_entries[0].rsplit("kind.txt", 1)[0].rstrip("/")
        name = check_name(name_hint or (root or "imported"))
        base = name
        n = 2
        while exists(name):
            name = f"{base} ({n})"
            n += 1
        prefix = f"{root}/" if root else ""
        files = []
        for entry in names:
            if entry.endswith("/") or not entry.startswith(prefix):
                continue
            relative = entry[len(prefix):]
            posix = PurePosixPath(relative)
            drive, _ = ntpath.splitdrive(relative)
            if not relative or posix.is_absolute() or drive or ".." in posix.parts:
                raise ValueError("project archive contains an unsafe path")
            files.append((entry, relative))
        dest = project_dir(name)
        dest.mkdir(parents=True)
        for entry, relative in files:
            target = (dest / relative).resolve()
            try:
                target.relative_to(dest.resolve())
            except ValueError as exc:
                raise ValueError("project archive contains a path outside the project") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(original[entry]))
    for sub in ("jobs", "results"):
        (dest / sub).mkdir(exist_ok=True)
    project = load_project(name)
    project.name = name
    save_project(name, project)
    return _describe(dest)


def unique_name(base: str) -> str:
    """`base`, or `base (2)`, `base (3)`… — the first that is free."""
    base = check_name(base)
    if not exists(base):
        return base
    n = 2
    while exists(f"{base} ({n})"):
        n += 1
    return f"{base} ({n})"
