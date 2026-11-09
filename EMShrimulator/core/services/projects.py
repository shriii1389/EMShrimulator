"""Project lifecycle: listing, creating from templates, importing and exporting archives."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from EMShrimulator.core import project as storage
from EMShrimulator.core.errors import AlreadyExists, InvalidInput, NotFound

KINDS = ("modeler", "layout")


def list_projects() -> List[Dict]:
    return storage.list_projects()


def list_templates() -> Dict[str, List[Dict]]:
    """Everything the New Project dialog can start from, both design types."""
    from EMShrimulator.core.em import templates as modeler_templates
    from EMShrimulator.core.layout import templates as layout_templates

    return {
        "modeler": modeler_templates.list_templates(),
        "layout": layout_templates.list_examples(),
    }


def check_new_name(name: str) -> str:
    """The cleaned name, or an InvalidInput/AlreadyExists saying why not."""
    try:
        clean = storage.check_name(name)
    except storage.InvalidName as exc:
        raise InvalidInput(str(exc)) from exc
    if storage.exists(clean):
        raise AlreadyExists(f"A project named '{clean}' already exists.")
    return clean


def create_project(name: str, kind: str, template: Optional[str] = None):
    """Create a project, optionally from a template, and return it loaded."""
    if kind not in KINDS:
        raise InvalidInput("kind must be 'modeler' or 'layout'")
    name = check_new_name(name)
    from EMShrimulator.core.em import templates as modeler_templates
    from EMShrimulator.core.layout import templates as layout_templates

    try:
        if kind == "modeler":
            project = modeler_templates.build(template or "blank", name)
            storage.create_project(name, "modeler", project)
        else:
            storage.create_project(name, "layout")
            if template:
                project = layout_templates.build(template, name, storage.project_dir(name))
                storage.save_project(name, project)
    except KeyError as exc:
        storage.delete_project(name)
        raise NotFound(f"There is no template '{template}'.") from exc
    except FileExistsError as exc:
        raise AlreadyExists(str(exc)) from exc
    return storage.load_project(name)


def kind_of(name: str) -> str:
    try:
        return storage.get_kind(name)
    except FileNotFoundError as exc:
        raise NotFound(f"There is no project named '{name}'.") from exc
    except storage.InvalidName as exc:
        raise InvalidInput(str(exc)) from exc


def rename_project(name: str, new_name: str) -> Dict:
    new_name = check_new_name(new_name)
    try:
        return storage.rename_project(name, new_name)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc


def duplicate_project(name: str, new_name: str) -> Dict:
    new_name = check_new_name(new_name)
    try:
        return storage.duplicate_project(name, new_name)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc


def delete_project(name: str) -> None:
    try:
        storage.delete_project(name)
    except storage.InvalidName as exc:
        raise InvalidInput(str(exc)) from exc


def suggest_copy_name(name: str) -> str:
    return storage.unique_name(f"{name} copy")


def export_to_file(name: str, destination: Path, include_results: bool = True) -> Path:
    """Write the project archive to `destination`, returning the path written."""
    try:
        data = storage.export_archive(name, include_results=include_results)
    except FileNotFoundError as exc:
        raise NotFound(str(exc)) from exc
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def import_from_file(source: Path) -> Dict:
    """Restore a project from an exported `.gems.zip`."""
    try:
        return storage.import_archive(Path(source).read_bytes(), name_hint=None)
    except (ValueError, storage.InvalidName) as exc:
        raise InvalidInput(str(exc)) from exc
