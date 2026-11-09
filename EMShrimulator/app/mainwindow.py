"""Main window: menus, status bar, and a stack of the project gallery and the open workspace."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
)

from EMShrimulator.app import notify, theme
from EMShrimulator.app.views.project_gallery import ProjectGallery
from EMShrimulator.app.widgets.toast import ToastHost
from EMShrimulator.config import APP_NAME, APP_VERSION, projects_dir, set_projects_dir
from EMShrimulator.core import jobs
from EMShrimulator.core.services import projects


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1480, 920)
        self.workspace = None
        self.session = None

        self.toasts = ToastHost(self)
        notify.install(self.toasts.show)

        self.gallery = ProjectGallery()
        self.gallery.openRequested.connect(self.open_project)
        self.pages = QStackedWidget()
        self.pages.addWidget(self.gallery)
        self.setCentralWidget(self.pages)

        self._build_menus()
        self.statusBar().showMessage("Ready")
        geometry = QSettings().value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    # ---------------------------------------------------------------- menus

    def _action(self, menu, text: str, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        if shortcut is not None:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False: slot())
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self._action(file_menu, "&New project…", self.new_project, QKeySequence.New)
        self.open_project_action = self._action(file_menu, "&Open project", self.gallery.open_selected)
        self.close_project_action = self._action(file_menu, "&Close project", self.close_project, "Ctrl+Shift+O")
        file_menu.addSeparator()
        self.save_action = self._action(file_menu, "&Save", self._save)
        file_menu.addSeparator()
        # These act on the start page's selected project.
        self.rename_project_action = self._action(file_menu, "&Rename project…", self.gallery.rename_selected, "F2")
        self.duplicate_project_action = self._action(file_menu, "&Duplicate project",
                                                     self.gallery.duplicate_selected)
        self.delete_project_action = self._action(file_menu, "De&lete project…", self.gallery.delete_selected,
                                                  QKeySequence.Delete)
        for action in (self.rename_project_action, self.delete_project_action):
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        file_menu.addSeparator()
        self._action(file_menu, "&Import project…", self.gallery.import_project)
        self.export_action = self._action(file_menu, "&Export project…", self._export)
        file_menu.addSeparator()
        self._action(file_menu, "Projects &folder…", self._choose_projects_folder)
        file_menu.addSeparator()
        self._action(file_menu, "E&xit", self.close, QKeySequence.Quit)
        self.gallery.use_actions([self.open_project_action, None, self.rename_project_action,
                                  self.duplicate_project_action, self.delete_project_action, None, self.export_action])
        self.gallery.selectionChanged.connect(self._update_selection_actions)

        self.edit_menu = bar.addMenu("&Edit")

        self.view_menu = bar.addMenu("&View")
        theme_menu = self.view_menu.addMenu("Theme")
        group = QActionGroup(self)
        for key, tokens in theme.THEMES.items():
            action = QAction(tokens["label"], self, checkable=True)
            action.setChecked(key == theme.manager().name)
            action.triggered.connect(lambda _=False, key=key: self._set_theme(key))
            group.addAction(action)
            theme_menu.addAction(action)
        self.panels_menu = self.view_menu.addMenu("Panels")

        self.simulation_menu = bar.addMenu("&Simulation")
        # Shortcuts live on the workspace toolbar, not here.
        self.run_menu_action = self._action(self.simulation_menu, "Run    F5", self._run)
        self.stop_menu_action = self._action(self.simulation_menu, "Stop", self._stop)

        self.help_menu = bar.addMenu("&Help")
        self._action(self.help_menu, "&Environment…", self._diagnostics)
        self._action(self.help_menu, "&About", self._about)
        self._workspace_menus = []
        self._update_project_actions()

    def _update_project_actions(self) -> None:
        open_ = self.workspace is not None
        for action in (self.close_project_action, self.save_action, self.run_menu_action, self.stop_menu_action):
            action.setEnabled(open_)
        self._update_selection_actions()
        self.panels_menu.clear()
        if open_:
            for action in self.workspace.dock_actions():
                self.panels_menu.addAction(action)
        self._install_workspace_menus()

    def _install_workspace_menus(self) -> None:
        """Insert the workspace's menus before Help; its Simulation menu replaces the plain one."""
        bar = self.menuBar()
        for action in self._workspace_menus:
            bar.removeAction(action)
        self._workspace_menus = []
        menus = self.workspace.menus() if self.workspace is not None else []
        for menu in menus:
            self._workspace_menus.append(bar.insertMenu(self.help_menu.menuAction(), menu))
        self.simulation_menu.menuAction().setVisible(not menus)

    def _update_selection_actions(self) -> None:
        # The start page's selection only counts while it is shown.
        picked = self.workspace is None and self.gallery.selected() is not None
        for action in (self.open_project_action, self.rename_project_action, self.duplicate_project_action,
                       self.delete_project_action):
            action.setEnabled(picked)
        self.export_action.setEnabled(picked or self.workspace is not None)

    def _bind_undo(self) -> None:
        # clear() deletes the actions, so placeholders are made afresh.
        self.edit_menu.clear()
        if self.session is None:
            for text in ("Undo", "Redo"):
                self.edit_menu.addAction(text).setEnabled(False)
            return
        undo = self.session.undo_stack.createUndoAction(self, "&Undo")
        undo.setShortcut(QKeySequence.Undo)
        redo = self.session.undo_stack.createRedoAction(self, "&Redo")
        redo.setShortcut(QKeySequence.Redo)
        self.edit_menu.addAction(undo)
        self.edit_menu.addAction(redo)
        edits = [getattr(self.workspace, name, None) for name in ("duplicate_action", "delete_action")]
        edits = [action for action in edits if action is not None]
        if edits:
            self.edit_menu.addSeparator()
            self.edit_menu.addActions(edits)

    # ------------------------------------------------------------- projects

    def new_project(self) -> None:
        # Imported on first use.
        from EMShrimulator.app.views.new_project_dialog import NewProjectDialog

        dialog = NewProjectDialog(self)
        if not dialog.exec():
            return
        name, kind, template = dialog.chosen_name(), dialog.chosen_kind(), dialog.chosen_template()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            created = notify.guarded(projects.create_project, name, kind, template)
        finally:
            QApplication.restoreOverrideCursor()
        if created is not None:
            notify.notify(f"Created \"{name}\".", "ok")
            self.gallery.refresh()
            self.open_project(name)

    def open_project(self, name: str) -> None:
        if self.workspace is not None and not self.close_project():
            return
        kind = notify.guarded(projects.kind_of, name)
        if kind is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(f"Opening {name}…")
        try:
            workspace = notify.guarded(self._build_workspace, name, kind)
        finally:
            QApplication.restoreOverrideCursor()
        if workspace is None:
            self.statusBar().showMessage("Ready")
            return
        self.workspace = workspace
        self.session = workspace.session
        self.pages.addWidget(workspace)
        self.pages.setCurrentWidget(workspace)
        for widget in workspace.status_widgets:
            self.statusBar().addPermanentWidget(widget)
            widget.show()
        self.statusBar().clearMessage()
        workspace.titleChanged.connect(self._retitle)
        self._retitle()
        self._bind_undo()
        self._update_project_actions()
        QSettings().setValue("recent/project", name)

    def _build_workspace(self, name: str, kind: str):
        from EMShrimulator.app import preload

        preload.wait()
        if kind == "modeler":
            from EMShrimulator.app.session import ModelerSession
            from EMShrimulator.app.views.modeler.modeler_view import ModelerView
            from EMShrimulator.core.services import modeler as modeler_service

            return ModelerView(ModelerSession(name, modeler_service.load(name)))
        from EMShrimulator.app.session import LayoutSession
        from EMShrimulator.app.views.layout.layout_view import LayoutView
        from EMShrimulator.core.services import layout as layout_service

        return LayoutView(LayoutSession(name, layout_service.load(name)))

    def close_project(self) -> bool:
        """Close the open project, asking first about unsaved edits. True if closed."""
        if self.workspace is None:
            self.pages.setCurrentWidget(self.gallery)
            return True
        if self.session.dirty:
            answer = QMessageBox.question(
                self, "Unsaved changes", f"Save the changes to \"{self.session.name}\" before closing it?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel:
                return False
            if answer == QMessageBox.Save and not self.workspace.save():
                return False
        for widget in self.workspace.status_widgets:
            self.statusBar().removeWidget(widget)
        self.workspace.shutdown()
        self.pages.removeWidget(self.workspace)
        self.workspace.deleteLater()
        self.workspace = None
        self.session = None
        self._bind_undo()
        self._update_project_actions()
        self.pages.setCurrentWidget(self.gallery)
        self.gallery.refresh()
        self._retitle()
        return True

    def _retitle(self, *_args) -> None:
        if self.workspace is None:
            self.setWindowTitle(APP_NAME)
        else:
            label = "3D Modeler" if self.session.kind == "modeler" else "3D Layout"
            self.setWindowTitle(f"{self.workspace.window_title()} — {label} — {APP_NAME}")

    # -------------------------------------------------------------- actions

    def _save(self) -> None:
        if self.workspace is not None and self.workspace.save():
            notify.notify("Saved.", "ok")

    def _run(self) -> None:
        if self.workspace is not None:
            self.workspace.run()

    def _stop(self) -> None:
        if self.workspace is not None:
            self.workspace.stop()

    def _export(self) -> None:
        """The open project, or on the start page the selected one."""
        if self.session is None:
            self.gallery.export_selected()
            return
        if self.session.dirty:
            self.workspace.save()
        target, _ = QFileDialog.getSaveFileName(self, "Export project",
                                                str(Path.home() / f"{self.session.name}.gems.zip"),
                                                "EMShrimulator project (*.gems.zip *.zip)")
        if target and notify.guarded(projects.export_to_file, self.session.name, Path(target)):
            notify.notify(f"Exported to {target}.", "ok")

    def _choose_projects_folder(self) -> None:
        if self.workspace is not None and not self.close_project():
            return
        folder = QFileDialog.getExistingDirectory(self, "Projects folder", str(projects_dir()))
        if folder:
            set_projects_dir(folder)
            QSettings().setValue("projects/folder", folder)
            self.gallery.refresh()

    def _set_theme(self, name: str) -> None:
        theme.manager().apply(name)
        QSettings().setValue("appearance/theme", name)

    def _diagnostics(self) -> None:
        from EMShrimulator.app.views.diagnostics_dialog import DiagnosticsDialog

        DiagnosticsDialog(self).exec()

    def _about(self) -> None:
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> {APP_VERSION}<br><br>An EM simulation environment over openEMS, with a parametric "
            "3D modeler and a Gerber/PCB workflow.<br><br>Licensed GPL-3.0-or-later. openEMS and CSXCAD are "
            "© their authors; gerber2ems is Apache-2.0 (vendored, with patches).")

    # ---------------------------------------------------------------- close

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        active = [j for j in jobs.list_jobs() if j["status"] in ("pending", "running")]
        if active:
            answer = QMessageBox.warning(
                self, "Runs in progress",
                f"{len(active)} simulation(s) are still running. Quitting stops them. Quit anyway?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        if not self.close_project():
            event.ignore()
            return
        for job in active:
            jobs.cancel_job(job["id"])
        QSettings().setValue("window/geometry", self.saveGeometry())
        event.accept()
