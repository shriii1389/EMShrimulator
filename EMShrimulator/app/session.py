"""The open project: state, dirty flag, selection and undo stack.

All changes go through `push()` as undo commands, which swap in a new project object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from EMShrimulator.core.em.model import ModelerProject
from EMShrimulator.core.layout.model import LayoutProject
from EMShrimulator.core.services import layout as layout_service
from EMShrimulator.core.services import modeler as modeler_service


@dataclass(frozen=True)
class Selection:
    """kind: primitive/port/dump (key = id), material/parameter/cs (name), layout_port (number), stackup and
    diff_pair (index), copper (tuple of (layer, object index, piece))."""

    kind: str
    key: Any


class ProjectSession(QObject):
    projectChanged = Signal()
    selectionChanged = Signal(object)
    dirtyChanged = Signal(bool)
    saved = Signal()

    def __init__(self, name: str, kind: str, project, parent=None):
        super().__init__(parent)
        self.name = name
        self.kind = kind
        self._project = project
        self._selection: Optional[Selection] = None
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(500)
        # A bound method, so Qt drops the connection when the session is destroyed.
        self.undo_stack.cleanChanged.connect(self._clean_changed)

    def _clean_changed(self, clean: bool) -> None:
        self.dirtyChanged.emit(not clean)

    # ------------------------------------------------------------ project

    @property
    def project(self):
        return self._project

    def _install(self, project) -> None:
        """Swap in a project state. Called by commands, never by views."""
        self._project = project
        self._on_project_installed()
        self._drop_stale_selection()
        self.projectChanged.emit()

    def _on_project_installed(self) -> None:
        """Hook for subclasses that derive state from the project."""

    def push(self, command: QUndoCommand) -> Any:
        """Apply an edit as one undo step and return what the edit returned."""
        self.undo_stack.push(command)
        return getattr(command, "result", None)

    @property
    def dirty(self) -> bool:
        return not self.undo_stack.isClean()

    def save(self) -> None:
        self._save(self._project)
        self.undo_stack.setClean()
        self.saved.emit()

    def _save(self, project) -> None:
        raise NotImplementedError

    def replace_external(self, project) -> None:
        """Adopt a project changed along with files on disk: save it and reset undo."""
        self._save(project)
        self.undo_stack.clear()
        self._install(project)

    # ---------------------------------------------------------- selection

    @property
    def selection(self) -> Optional[Selection]:
        return self._selection

    def select(self, kind: Optional[str], key: Any = None) -> None:
        selection = Selection(kind, key) if kind else None
        if selection == self._selection:
            return
        self._selection = selection
        self.selectionChanged.emit(selection)

    def _selection_exists(self, selection: Selection) -> bool:
        return True

    def _drop_stale_selection(self) -> None:
        if self._selection is not None and not self._selection_exists(self._selection):
            self._selection = None
            self.selectionChanged.emit(None)


class ModelerSession(ProjectSession):
    """A modeler project plus its resolved form, re-resolved on every change."""

    def __init__(self, name: str, project: ModelerProject, parent=None):
        super().__init__(name, "modeler", project, parent)
        self.resolution = modeler_service.resolve(project)

    @property
    def project(self) -> ModelerProject:
        return self._project

    def _on_project_installed(self) -> None:
        self.resolution = modeler_service.resolve(self._project)

    def _save(self, project) -> None:
        modeler_service.save(self.name, project)

    def _selection_exists(self, selection: Selection) -> bool:
        p = self._project
        if selection.kind == "primitive":
            return any(x.id == selection.key for x in p.primitives)
        if selection.kind == "port":
            return any(x.id == selection.key for x in p.ports)
        if selection.kind == "dump":
            return any(x.id == selection.key for x in p.dumps)
        if selection.kind == "material":
            return any(x.name == selection.key for x in p.materials)
        if selection.kind == "cs":  # "" is the global system
            return not selection.key or any(x.name == selection.key for x in p.coordinate_systems)
        return True


class LayoutSession(ProjectSession):
    def __init__(self, name: str, project: LayoutProject, parent=None):
        super().__init__(name, "layout", project, parent)

    @property
    def project(self) -> LayoutProject:
        return self._project

    def _save(self, project) -> None:
        layout_service.save(self.name, project)

    def _selection_exists(self, selection: Selection) -> bool:
        if selection.kind == "layout_port":
            return any(x.number == selection.key for x in self._project.ports)
        if selection.kind == "stackup":
            return 0 <= int(selection.key) < len(self._project.stackup)
        if selection.kind == "diff_pair":
            return 0 <= int(selection.key) < len(self._project.diff_pairs)
        if selection.kind == "copper":
            return bool(selection.key) and all(self._copper_exists(*key) for key in selection.key)
        return True

    def _copper_exists(self, layer: str, index: int, sub: int) -> bool:
        """Whether a copper piece (layer, object index, piece) still exists; a via is ("#via", id, 0)."""
        if layer == "#via":
            return any(v.id == index for v in self._project.vias or [])
        edit = next((e for e in self._project.copper_edits if e.layer == layer and e.index == index), None)
        return sub == 0 if edit is None else sub < len(edit.objects)
