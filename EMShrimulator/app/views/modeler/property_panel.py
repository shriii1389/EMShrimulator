"""The inspector beside the 3D view: one editor per selection kind, built on first use."""
from __future__ import annotations

from typing import Dict

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QWidget

from EMShrimulator.app.views.modeler.coordinate_editor import CoordinateSystemEditor
from EMShrimulator.app.views.modeler.dump_editor import DumpEditor
from EMShrimulator.app.views.modeler.material_editor import MaterialEditor
from EMShrimulator.app.views.modeler.port_editor import PortEditor
from EMShrimulator.app.views.modeler.primitive_editor import PrimitiveEditor
from EMShrimulator.app.views.shell import Inspector
from EMShrimulator.core.em.edits import find_coordinate_system, index_of

EDITORS = {"primitive": PrimitiveEditor, "port": PortEditor, "material": MaterialEditor, "dump": DumpEditor,
           "cs": CoordinateSystemEditor}
PREBUILD_AFTER_MS = 1200


class PropertyPanel(Inspector):
    shownChanged = Signal(bool)
    plotRequested = Signal(int)          # port number: plot its reflection
    focusRequested = Signal(str, str)    # (kind, id): turn the camera to it
    lineDrawRequested = Signal(str)      # port id: draw its integration line in the 3D view
    anchorPickRequested = Signal(str, str)  # (coordinate system, object id)

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        self._editors: Dict[str, QWidget] = {}
        self._shown = False

        session.projectChanged.connect(self.refresh)
        session.selectionChanged.connect(lambda _s: self.refresh())
        self.refresh()

        # Owned by the panel, so it dies with it if the project closes first.
        self._prebuild = QTimer(self)
        self._prebuild.setSingleShot(True)
        self._prebuild.timeout.connect(lambda: self.editor("primitive"))
        self._prebuild.start(PREBUILD_AFTER_MS)

    @property
    def shown(self) -> bool:
        return self._shown

    def editor(self, kind: str) -> QWidget:
        """The editor for `kind`, built on first use."""
        editor = self._editors.get(kind)
        if editor is None:
            editor = EDITORS[kind](self.session)
            editor.setVisible(False)
            if kind == "port":
                editor.plotRequested.connect(self.plotRequested)
                editor.drawLineRequested.connect(self.lineDrawRequested)
            if hasattr(editor, "focusRequested"):
                editor.focusRequested.connect(self.focusRequested)
            if hasattr(editor, "anchorPickRequested"):
                editor.anchorPickRequested.connect(self.anchorPickRequested)
            self.add_editor(editor)
            self._editors[kind] = editor
        return editor

    def refresh(self) -> None:
        selection = self.session.selection
        project = self.session.project
        page = None
        if selection is not None:
            if selection.kind == "primitive":
                i = index_of(project.primitives, selection.key)
                if i >= 0:
                    page = self.editor("primitive")
                    page.load(project.primitives[i], i)
            elif selection.kind == "port":
                i = index_of(project.ports, selection.key)
                if i >= 0:
                    page = self.editor("port")
                    page.load(project.ports[i], i)
            elif selection.kind == "dump":
                i = index_of(project.dumps, selection.key)
                if i >= 0:
                    page = self.editor("dump")
                    page.load(project.dumps[i], i)
            elif selection.kind == "material":
                i = next((j for j, m in enumerate(project.materials) if m.name == selection.key), -1)
                if i >= 0:
                    page = self.editor("material")
                    page.load(project.materials[i], i)
            elif selection.kind == "cs":
                cs = find_coordinate_system(project, selection.key) if selection.key else None
                if cs is not None or not selection.key:
                    page = self.editor("cs")
                    page.load(cs, project.coordinate_systems.index(cs) if cs is not None else -1)
        for candidate in self._editors.values():
            candidate.setVisible(candidate is page)
        if page is not None:
            self.show_title(page)
        shown = page is not None
        if shown != self._shown:
            self._shown = shown
            self.shownChanged.emit(shown)
