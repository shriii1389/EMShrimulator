"""3D Layout operations: board import, the 2D canvas data, run, results."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from EMShrimulator.config import TEMP_DIR
from EMShrimulator.core import jobs, procpool
from EMShrimulator.core import project as storage
from EMShrimulator.core.errors import InvalidInput, NotFound, ValidationFailed
from EMShrimulator.core.layout import board as layout_board
from EMShrimulator.core.layout import gerber_import
from EMShrimulator.core.layout import ports as layout_ports
from EMShrimulator.core.layout import results as layout_results
from EMShrimulator.core.layout import snap as layout_snap
from EMShrimulator.core.layout import validate as layout_validate
from EMShrimulator.core.layout import via_import
from EMShrimulator.core.layout.padstacks import layer_specs
from EMShrimulator.core.layout.copper import drop_originals
from EMShrimulator.core.layout.edits import copper_edits
from EMShrimulator.core.layout.model import LayoutProject, StackupLayer, apply_preset, preset_list
from EMShrimulator.core.layout.project_writer import write_all
from EMShrimulator.core.layout.runner import build_command, build_env

# What the Run step picker offers, in the order it offers it.
STEP_LABELS: Dict[str, str] = {
    "all": "Geometry, simulate and post-process",
    "geometry": "Geometry only (mesh the copper, place the ports)",
    "simulate": "Simulate only (reuse the saved geometry)",
    "postprocess": "Post-process only (re-plot existing results)",
}


# --- persistence ---

def load(name: str) -> LayoutProject:
    try:
        kind = storage.get_kind(name)
    except FileNotFoundError as exc:
        raise NotFound(f"There is no project named '{name}'.") from exc
    except storage.InvalidName as exc:
        raise InvalidInput(str(exc)) from exc
    if kind != "layout":
        raise InvalidInput(f"'{name}' is not a 3D Layout project.")
    return storage.load_project(name)  # type: ignore[return-value]


def save(name: str, project: LayoutProject) -> None:
    storage.save_project(name, project)


def project_dir(name: str) -> Path:
    return storage.project_dir(name)


# --- board files ---

def _is_stackup_json(path: Path) -> bool:
    return path.suffix.lower() == ".json" and "stackup" in path.name.lower()


def _import_stackup_json(project: LayoutProject, source: Path, fab_dir: Path) -> Dict:
    """A KiCad "Export Stackup" JSON is exactly the shape the stackup holds."""
    from EMShrimulator.core.layout.templates import _stackup_from_json

    dest = fab_dir / source.name
    if source.resolve() != dest.resolve():
        shutil.copyfile(source, dest)
    try:
        project.stackup = _stackup_from_json(dest)
    except Exception as exc:  # noqa: BLE001 - keep the file, report the miss
        return {"warnings": [f"'{source.name}' was stored but not understood as a stackup: {exc}"]}
    return {"warnings": [], "stackup_layers": len(project.stackup)}


def _stage(source: Path, staged: Path, name: str) -> None:
    """Stage one input. An archive replaces the staging area; a loose file adds to it."""
    with source.open("rb") as handle:
        payload_head = handle.read(512)
    if gerber_import.is_archive_bytes(payload_head, source.name):
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        work_dir = TEMP_DIR / f"{name}-gerber-work"
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            members = gerber_import.extract_archive(source, work_dir)
            shutil.rmtree(staged, ignore_errors=True)
            staged.mkdir(exist_ok=True)
            for member in members:
                shutil.copyfile(member, staged / member.name)
        except gerber_import.GerberImportError as exc:
            raise InvalidInput(str(exc)) from exc
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    else:
        shutil.copyfile(source, staged / source.name)


def import_files(name: str, project: LayoutProject, sources: Iterable[Path]) -> Tuple[LayoutProject, Dict]:
    """Import Gerber archives, layer files and stackup JSON. Returns (unsaved project, report)."""
    pdir = storage.project_dir(name)
    fab_dir = pdir / "fab"
    fab_dir.mkdir(exist_ok=True)
    staged = pdir / "uploads"
    staged.mkdir(exist_ok=True)

    sources = [Path(s) for s in sources]
    warnings: List[str] = []
    board_sources = []
    for source in sources:
        if not source.is_file():
            raise NotFound(f"'{source}' does not exist.")
        if _is_stackup_json(source):
            warnings += _import_stackup_json(project, source, fab_dir)["warnings"]
        else:
            board_sources.append(source)

    if not board_sources:
        return project, {"warnings": warnings, "stackup_layers": len(project.stackup), "files": []}

    for source in board_sources:
        _stage(source, staged, name)

    try:
        report = gerber_import.normalise_into_fab(
            sorted(p for p in staged.iterdir() if p.is_file()), fab_dir, name)
    except gerber_import.GerberImportError as exc:
        raise InvalidInput(str(exc)) from exc

    project.gerber_files = report.gerber_files
    project.drill_file = report.drill_file
    project.source_format = "gerber"
    _forget_copper_edits(project, fab_dir)
    _read_vias(project, pdir)
    if not project.stackup:
        project.stackup = [StackupLayer(**layer)
                           for layer in gerber_import.infer_stackup(len(report.copper_layers))]
        report.warnings.append(
            f"A standard {len(report.copper_layers)}-layer 1.6 mm FR4 stackup was assumed. "
            "Check the thicknesses and εr against your fab's real build in the Stackup panel — "
            "they set the impedance of every trace on the board."
        )
    result = report.to_dict()
    result["warnings"] = warnings + result["warnings"]
    return project, result


def import_odbpp(name: str, project: LayoutProject, source: Path, step: Optional[str] = None) -> Tuple[LayoutProject, Dict]:
    """Import an ODB++ product model, converted to the same Gerber set as above."""
    from EMShrimulator.core.layout import odb_convert

    pdir = storage.project_dir(name)
    fab_dir = pdir / "fab"
    fab_dir.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = TEMP_DIR / f"{name}-odb-work"
    shutil.rmtree(work_dir, ignore_errors=True)
    try:
        # In a worker process, so parsing doesn't hold the GIL and stall the window.
        report = procpool.call(odb_convert.import_archive, Path(source), work_dir, fab_dir, name, step)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
        raise InvalidInput(f"ODB++ import failed: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    project.gerber_files = list(report["gerber_files"])
    project.drill_file = report["drill_file"]
    project.source_format = "odb++"
    _forget_copper_edits(project, fab_dir)
    _read_vias(project, pdir)
    if not project.stackup:
        project.stackup = [StackupLayer(**layer) for layer in report["stackup"]]
    return project, report


def _read_vias(project: LayoutProject, pdir: Path) -> str:
    """Read the drill file into padstacks and vias in a worker process."""
    found = procpool.call(via_import.extract, pdir, project.via.plating_thickness_um, project.via.filling_epsilon)
    via_import.apply(project, found)
    return found.get("note", "")


def read_vias(name: str, project: LayoutProject) -> Dict:
    """Padstacks and vias from an older project's drill file: {"padstacks", "vias", "note"}."""
    return procpool.call(via_import.extract, storage.project_dir(name), project.via.plating_thickness_um,
                         project.via.filling_epsilon)


def _forget_copper_edits(project: LayoutProject, fab_dir: Path) -> None:
    """New board files: the copper edits and vias were of objects in the old ones."""
    drop_originals(fab_dir)
    project.copper_edits = []
    project.padstacks, project.vias = [], None


def clear_files(name: str, project: LayoutProject) -> LayoutProject:
    """Drop every imported board file, so a fresh Gerber set starts clean."""
    pdir = storage.project_dir(name)
    shutil.rmtree(pdir / "uploads", ignore_errors=True)
    shutil.rmtree(pdir / "cache", ignore_errors=True)
    for stale in (pdir / "fab").glob("*"):
        if stale.is_file() and stale.name.lower() != "stackup.json":
            stale.unlink(missing_ok=True)
    _forget_copper_edits(project, pdir / "fab")
    project.gerber_files = []
    project.drill_file = None
    project.vias = []
    return project


def delete_file(name: str, project: LayoutProject, filename: str) -> LayoutProject:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise InvalidInput("invalid filename")
    pdir = storage.project_dir(name)
    (pdir / "fab" / filename).unlink(missing_ok=True)
    drop_originals(pdir / "fab", filename)
    copper_layer = layout_board.copper_layer_name(filename)
    if copper_layer is not None:
        project.copper_edits = [e for e in project.copper_edits if e.layer != copper_layer]
        for via in project.vias or []:  # pads in the removed file
            via.links = [link for link in via.links if link.layer != copper_layer]

    # Remove the staged original too, or the next import brings it back.
    staged = pdir / "uploads"
    if staged.is_dir():
        for candidate in sorted(staged.iterdir()):
            if not candidate.is_file():
                continue
            try:
                identified = gerber_import.identify(candidate)
            except Exception:  # noqa: BLE001 - an unreadable stray file is left alone
                continue
            layer = identified.layer or identified.role
            if (candidate.name == filename or filename.endswith(f"-{layer}.gbr")
                    or filename.endswith(f"-{layer}.drl")):
                candidate.unlink(missing_ok=True)

    project.gerber_files = [f for f in project.gerber_files if f != filename]
    if project.drill_file == filename:
        project.drill_file = None
    return project


# --- 2D canvas ---

def board_info(name: str, project: LayoutProject, px_per_mm: Optional[float] = None) -> Dict:
    """Board extent, copper layers and the mm<->pixel mapping for the canvas."""
    return layout_board.board_info(
        storage.project_dir(name),
        stackup=[layer.model_dump() for layer in project.stackup],
        preview_px_per_mm=px_per_mm,
    )


def layer_png(name: str, layer_name: str, px_per_mm: float) -> bytes:
    """One canvas layer, white copper on transparent, cached on disk."""
    try:
        return layout_board.render_preview(storage.project_dir(name), layer_name, px_per_mm)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc


def drill_holes(name: str) -> List[Dict]:
    return layout_board.drill_holes(storage.project_dir(name))


# These run in worker processes (core.procpool): Gerber parsing and resvg hold the GIL.

def canvas_data(name: str, project: LayoutProject) -> Tuple[Dict, List[Dict]]:
    """(board_info, drill holes) for the canvas, after writing the edited copper."""
    return procpool.call(layout_board.canvas_data, storage.project_dir(name),
                         [layer.model_dump() for layer in project.stackup], copper_edits(project),
                         layer_specs(project))


def sync_copper(name: str, project: LayoutProject) -> List[str]:
    """Write the copper Gerbers with the project's edits and vias; returns the layers whose file changed."""
    return procpool.call(layout_board.sync_copper, storage.project_dir(name), copper_edits(project),
                         layer_specs(project))


def copper_pieces(name: str, layer_name: str) -> Dict:
    """One layer's copper as imported, as pieces to select and reshape."""
    try:
        return procpool.call(layout_board.copper_pieces, storage.project_dir(name), layer_name)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc


def layer_alpha(name: str, layer_name: str, px_per_mm: float) -> Tuple[int, int, bytes]:
    """(width, height, 8-bit coverage) of one canvas layer."""
    try:
        return procpool.call(layout_board.render_preview_alpha, storage.project_dir(name), layer_name, px_per_mm)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc


def layer_shapes(name: str, layer_name: str) -> Dict:
    """One canvas layer as vector shapes, for drawing it zoomed in past its raster."""
    try:
        return procpool.call(layout_board.preview_shapes, storage.project_dir(name), layer_name)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc


def stackup_presets() -> List[Dict]:
    return preset_list()


def preset_stackup(preset_id: str) -> List[StackupLayer]:
    try:
        return apply_preset(preset_id)
    except KeyError as exc:
        raise NotFound(f"There is no stackup preset '{preset_id}'.") from exc


# --- validate + run ---

def snap_targets(name: str, project: LayoutProject) -> layout_snap.SnapSet:
    """Every snap target on the board, built in a worker process."""
    vias = None if project.vias is None else [(v.x_mm, v.y_mm) for v in project.vias]
    return procpool.call(layout_snap.build, storage.project_dir(name),
                         [layer.model_dump() for layer in project.stackup], vias)


def resolve_references(name: str, project: LayoutProject) -> Dict[int, layout_ports.ReferenceChoice]:
    """Where each "auto" port's reference lands, by port number."""
    return layout_ports.resolve_references(project, storage.project_dir(name))


def _write_resolved(project: LayoutProject, pdir: Path) -> None:
    # The copper gerber2ems meshes, the vias' pads in it.
    procpool.call(layout_board.sync_copper, pdir, copper_edits(project), layer_specs(project))
    resolved, _choices = layout_ports.resolved_project(project, pdir)
    write_all(resolved, pdir)


def validate(name: str, project: LayoutProject) -> Dict:
    return layout_validate.validate(project, storage.project_dir(name))


def start_run(name: str, project: LayoutProject, steps: str = "all") -> jobs.Job:
    """steps: comma-separated subset of geometry,simulate,postprocess, or 'all'."""
    pdir = storage.project_dir(name)
    step_list: List[str] = ["all"] if steps == "all" else [s.strip() for s in steps.split(",") if s.strip()]
    if "all" in step_list or "simulate" in step_list:
        report = validate(name, project)
        if not report["can_run"]:
            raise ValidationFailed("The board has validation errors and cannot be simulated.",
                                   [i for i in report["issues"] if i["severity"] == "error"])
    _write_resolved(project, pdir)
    try:
        cmd = build_command(step_list, project)
    except ValueError as exc:
        raise InvalidInput(str(exc)) from exc
    return jobs.start_job("layout", name, cmd, cwd=pdir, log_dir=pdir / "jobs", env=build_env(),
                          label="gerber2ems · " + ", ".join(step_list))


def simulation_json(name: str, project: LayoutProject) -> Dict:
    """The exact `simulation.json` gerber2ems will be handed."""
    pdir = storage.project_dir(name)
    _write_resolved(project, pdir)
    return json.loads((pdir / "simulation.json").read_text(encoding="utf-8"))


# --- results ---

def results(name: str, project: LayoutProject) -> Dict:
    """Assemble (or reuse) a Touchstone so layout gets the full Results views."""
    from EMShrimulator.core.em import results as em_results

    pdir = storage.project_dir(name)
    touchstone = layout_results.find_touchstone(pdir)
    if touchstone is None:
        try:
            touchstone = layout_results.build_touchstone(pdir, name)
        except Exception:  # noqa: BLE001 - fall back to the per-port CSVs
            touchstone = None
    plots = [str(pdir / "ems" / "results" / p) for p in layout_results.list_plots(pdir)]
    if touchstone is None:
        return {"available": False, "ports": layout_results.list_result_ports(pdir), "plots": plots}

    eps = max((layer.epsilon or 1.0 for layer in project.stackup), default=1.0)
    # gerber2ems numbers ports in their order, so a port's index is its Touchstone index.
    order = sorted(p.number for p in project.ports)
    pairs = [{"name": pair.name, "p": order.index(pair.positive), "n": order.index(pair.negative),
              "z_diff": pair.z_diff, "z_common": pair.z_common}
             for pair in project.diff_pairs if pair.positive in order and pair.negative in order]
    data = em_results.read_results(touchstone, epsilon_eff=eps, pairs=pairs)
    data.update(available=True, plots=plots, path=str(touchstone),
                result_ports=layout_results.list_result_ports(pdir))
    return data
