"""Shared base for the modeler and layout workspaces: project tree, centre pages, runs and saving."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QByteArray, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.notify import guarded, notify
from EMShrimulator.app.views.job_console import FINISHED, JobConsole, RunsView
from EMShrimulator.app.views.shell import SIDE_WIDTH, Page, RunIndicator
from EMShrimulator.app.views.validation_view import ValidationView
from EMShrimulator.core import jobs
from EMShrimulator.core.em.edits import PLOT_LABELS, PLOT_MENU, find_plot, plot_label
from EMShrimulator.core.errors import ValidationFailed

LAYOUT_VERSION = 2


class Workspace(QMainWindow):
    titleChanged = Signal(str)
    runningChanged = Signal(bool)

    kind = "workspace"
    layout_version = LAYOUT_VERSION

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.status_widgets: List[QWidget] = []
        self.tree = None
        self.pages: Dict[str, QWidget] = {}
        self._page_key = ""
        self._splitters: Dict[str, QSplitter] = {}

        # ------------------------------------------------------ output views
        self.console = JobConsole()
        self.console.finished.connect(self._job_finished)
        self.validation = ValidationView()
        self.validation.targetActivated.connect(self.select_by_name)
        self.runs = RunsView(session.name)
        self.runs.openLog.connect(self.follow_job)

        # -------------------------------------------------------- actions
        self.save_action = QAction("Save", self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.save_action.triggered.connect(self.save)
        self.run_action = QAction("▶ Run", self)
        self.run_action.setShortcut(QKeySequence("F5"))
        self.run_action.setToolTip("Validate, save and run the simulation (F5)")
        self.run_action.triggered.connect(self.run)
        self.stop_action = QAction("■ Stop", self)
        self.stop_action.setToolTip("Stop the running simulation")
        self.stop_action.triggered.connect(self.stop)
        self.stop_action.setEnabled(False)

        # Workspaces put their own labels between these two.
        self.run_indicator = RunIndicator(lambda: self.show_key(("log",)))
        self.console.monitor.statusChanged.connect(self._run_status)
        self.dirty_label = QLabel("saved")
        self.status_widgets.extend([self.run_indicator, self.dirty_label])
        session.dirtyChanged.connect(self._dirty_changed)

    # ---------------------------------------------------------------- shell

    def output_pages(self) -> Dict[str, QWidget]:
        """The solver log, validation and past runs, as pages."""
        return {
            "log": Page("Solver log", self.console, "Live output of the run in progress, or the last one."),
            "validation": Page("Validation", self.validation,
                               "Checks run on every edit. Double-click an issue to select what it names."),
            "runs": Page("Runs", self.runs, "Every run of this project in this session. Double-click to read its log."),
        }

    def build_shell(self, tree, pages: Dict[str, QWidget], first: str) -> None:
        """Lay out the project tree and the centre `pages`, starting on `first`."""
        self.tree = tree
        tree.navigate.connect(self.show_key)
        self.pages = pages
        self.stack = QStackedWidget()
        for page in pages.values():
            self.stack.addWidget(page)

        side = QWidget()
        side.setObjectName("project-side")
        heading = QLabel("Project manager")
        heading.setProperty("role", "heading")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_head = QHBoxLayout()
        side_head.setContentsMargins(12, 8, 8, 6)
        side_head.addWidget(heading)
        side_layout.addLayout(side_head)
        side_layout.addWidget(tree, 1)
        self.side = side

        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(side)
        self.split.addWidget(self.stack)
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 1)
        self._splitters["split"] = self.split
        self.setCentralWidget(self.split)
        self.side_action = QAction("Project manager", self, checkable=True)
        self.side_action.setChecked(True)
        self.side_action.toggled.connect(side.setVisible)
        self.show_page(first)

    def show_page(self, key: str) -> None:
        self._page_key = key
        self.stack.setCurrentWidget(self.pages[key])

    def show_key(self, key) -> None:
        """Show what a project-tree item stands for. Overridden per design type."""

    def dock_actions(self) -> List[QAction]:
        return [self.side_action] if self.tree is not None else []

    def menus(self) -> List[QMenu]:
        """Menus this workspace adds to the main window's menu bar while it is open."""
        return []

    # --------------------------------------------------------------- layout

    def restore_layout(self) -> None:
        settings = QSettings()
        state = settings.value(f"layout/{self.kind}/state")
        if isinstance(state, QByteArray):
            self.restoreState(state, self.layout_version)  # the toolbar
        for key, splitter in self._splitters.items():
            saved = settings.value(f"layout/{self.kind}/{key}")
            if isinstance(saved, QByteArray):
                splitter.restoreState(saved)
        self._split_restored = isinstance(settings.value(f"layout/{self.kind}/split"), QByteArray)

    def store_layout(self) -> None:
        settings = QSettings()
        settings.setValue(f"layout/{self.kind}/state", self.saveState(self.layout_version))
        for key, splitter in self._splitters.items():
            settings.setValue(f"layout/{self.kind}/{key}", splitter.saveState())

    def default_sizes(self) -> None:
        """Size the panels for a first launch."""
        if "split" in self._splitters and not getattr(self, "_split_restored", False):
            self.split.setSizes([SIDE_WIDTH, max(self.width() - SIDE_WIDTH, 600)])

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        # Splitters are sized once shown, when they know their width.
        if not getattr(self, "_sized", False):
            self._sized = True
            QTimer.singleShot(0, self.default_sizes)

    def show_output(self, tab: QWidget) -> None:
        # A run shows in the status bar and on its own page, never replacing the design view.
        if tab is self.validation:
            self.show_key(("validation",))
        elif tab is getattr(self, "results", None) and self.tree is not None:
            self.tree.set_badge(("results",), "updated")

    # ----------------------------------------------------------------- save

    def save(self) -> bool:
        return guarded(self._save) is True

    def _save(self) -> bool:
        self.session.save()
        return True

    def _dirty_changed(self, dirty: bool) -> None:
        self.dirty_label.setText("unsaved changes" if dirty else "saved")
        self.titleChanged.emit(self.window_title())

    def window_title(self) -> str:
        return f"{self.session.name}{' •' if self.session.dirty else ''}"

    # ------------------------------------------------------------------ run

    @property
    def running(self) -> bool:
        return self.console.running

    def run(self) -> None:
        if self.running:
            notify("A run is already in progress for this project.", "info")
            return
        if not self.save():
            return
        self.launch(self.start_job)

    def launch(self, start) -> None:
        """Start a job through `start()`, surfacing a refusal where it helps."""
        try:
            job = start()
        except ValidationFailed as exc:
            self.show_validation_failure(exc)
            notify(exc.message, "error")
            return
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            notify(str(exc), "error")
            return
        if job is not None:
            self.follow_job(job.id)
            notify(f"Started: {job.label}")

    def start_job(self):
        raise NotImplementedError

    def follow_job(self, job_id: str) -> None:
        self.console.follow(job_id)
        self.show_output(self.console)
        self._set_running(self.console.running)

    def stop(self) -> None:
        self.console.cancel()

    def _set_running(self, running: bool) -> None:
        self.run_action.setEnabled(not running)
        self.stop_action.setEnabled(running)
        self.runningChanged.emit(running)

    def _run_status(self, status: dict) -> None:
        self.run_indicator.show_status(status)
        if self.tree is not None:
            self.tree.set_badge(("log",), "running" if status["status"] not in FINISHED else status["status"])

    def _job_finished(self, status: dict) -> None:
        self._set_running(False)
        self.runs.refresh()
        if status["status"] == "done":
            notify(f"Finished: {status['label']} in {int(status.get('elapsed_s') or 0)} s.", "ok")
            self.on_results_ready()
        elif status["status"] == "failed":
            notify("The run failed — see the log.", "error")
        else:
            notify("The run was stopped.", "info")

    def on_results_ready(self) -> None:
        """Show the results of a run that just finished."""

    def show_validation_failure(self, exc: ValidationFailed) -> None:
        issues = exc.issues
        self.validation.show_report({
            "issues": issues, "errors": sum(1 for i in issues if i["severity"] == "error"),
            "warnings": sum(1 for i in issues if i["severity"] == "warning"), "infos": 0, "can_run": False,
        })
        self.show_output(self.validation)

    def validation_badge(self, report: Optional[dict]) -> None:
        """The validation item's suffix in the tree: the errors, else the warnings, else OK."""
        if not report or self.tree is None:
            return
        errors, warnings = report.get("errors", 0), report.get("warnings", 0)
        self.tree.set_badge(("validation",), f"{errors} error{'s' if errors != 1 else ''}" if errors else
                            (f"{warnings} warning{'s' if warnings != 1 else ''}" if warnings else "OK"))

    def select_by_name(self, target: str) -> None:
        """Select what a validation issue names. Overridden per design type."""

    # ------------------------------------------------------------ reports

    # New report menu groups (core.em.edits.PLOT_MENU).
    report_menu: Sequence[Tuple[str, Tuple[str, ...]]] = PLOT_MENU

    def report_pages(self, results, about: str, **plot_options) -> Dict[str, QWidget]:
        """The results overview and the single-report page, drawing from `results`."""
        from EMShrimulator.app.views.results.plot_page import PlotPage

        self.plot_page = PlotPage(self.session, results, **plot_options)
        results.loaded.connect(self._results_loaded)
        self.new_plot_button = QToolButton()
        self.new_plot_button.setText("New report")
        self.new_plot_button.setToolTip("A report of your own: S, Y or Z parameters, VSWR, delay, TDR"
                                        + (", far and near fields" if len(self.report_menu) > 1 else "")
                                        + ", in the display you choose")
        self.new_plot_button.setPopupMode(QToolButton.InstantPopup)
        self.new_plot_button.setMenu(self.fill_report_menu(QMenu(self)))
        rename = QToolButton()
        rename.setText("Rename…")
        rename.clicked.connect(lambda: self.rename_plot(self.plot_page.plot_id))
        delete = QToolButton()
        delete.setText("Delete")
        delete.clicked.connect(lambda: self.delete_plot(self.plot_page.plot_id))
        return {"results": Page("Results", results, about, [self.new_plot_button]),
                "plot": Page("Plot", self.plot_page, "", [rename, delete])}

    def fill_report_menu(self, menu: QMenu) -> QMenu:
        """The New report menu: solution data, then far and near fields where available."""
        groups = list(self.report_menu)
        for group, presets in groups:
            target = menu.addMenu(group) if len(groups) > 1 else menu
            for preset in presets:
                target.addAction(PLOT_LABELS[preset], lambda preset=preset: self.new_plot(preset))
        return menu

    def new_plot(self, kind: str, traces: Optional[List[str]] = None) -> None:
        plot_id = guarded(lambda: self.session.push(cmd.AddPlot(self.session, kind, traces)))
        if plot_id:
            self.show_key(("plot", plot_id))

    def show_plot(self, plot_id: str) -> None:
        plot = find_plot(self.session.project, plot_id)
        if plot is None:
            self.show_key(("results",))
            return
        page = self.pages["plot"]
        page.title.setText(plot.name)
        page.subtitle.setText(f"{plot_label(plot)} · from the newest run")
        self.show_page("plot")
        self.plot_page.show_plot(plot_id)

    def rename_plot(self, plot_id: Optional[str]) -> None:
        plot = find_plot(self.session.project, plot_id) if plot_id else None
        if plot is None:
            return
        text, ok = QInputDialog.getText(self, "Rename report", "Name", text=plot.name)
        if ok and text.strip():
            guarded(lambda: self.session.push(cmd.SetPlotField(self.session, plot_id, "name", text.strip())))
            self.show_plot(plot_id)

    def delete_plot(self, plot_id: Optional[str]) -> None:
        if not plot_id:
            return
        guarded(lambda: self.session.push(cmd.DeletePlot(self.session, plot_id)))
        if find_plot(self.session.project, plot_id) is None:
            self.show_key(("results",))

    def report_tree_menu(self, menu: QMenu, key) -> None:
        """The tree's menu on Results and on a report."""
        if key[0] == "results":
            self.fill_report_menu(menu.addMenu("New report"))
        elif key[0] == "plot":
            menu.addAction("Rename…", lambda: self.rename_plot(key[1]))
            menu.addAction("Delete", lambda: self.delete_plot(key[1]))

    def follow_plot_edits(self) -> None:
        """After an edit: leave a deleted report's page, and keep the title current."""
        if self._page_key != "plot":
            return
        plot = find_plot(self.session.project, self.plot_page.plot_id or "")
        if plot is None:
            self.show_key(("results",))
        else:
            self.pages["plot"].title.setText(plot.name)
            self.pages["plot"].subtitle.setText(f"{plot_label(plot)} · from the newest run")

    def _results_loaded(self) -> None:
        if self._page_key == "plot":
            self.plot_page.refresh()

    # ------------------------------------------------------------ shutdown

    def active_job_ids(self) -> List[str]:
        return [j["id"] for j in jobs.list_jobs(self.session.name) if j["status"] in ("pending", "running")]

    def shutdown(self) -> None:
        self.store_layout()
        self.console.monitor.stop_following()
        plot_page = getattr(self, "plot_page", None)
        if plot_page is not None:
            plot_page.shutdown()
