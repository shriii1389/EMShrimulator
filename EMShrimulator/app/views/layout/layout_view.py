"""The 3D Layout (PCB) workspace: project tree, board canvas and inspector, on the shared `Workspace` shell."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QFontDatabase, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.background import BackgroundRunner
from EMShrimulator.app.notify import guarded, notify
from EMShrimulator.app.views.layout.board_canvas import BoardCanvas
from EMShrimulator.app.views.layout.board_files import IMPORT_HINT, BoardFiles
from EMShrimulator.app.views.layout.board_settings import BoardSettings
from EMShrimulator.app.views.layout.board_tree import OUTLINE, BoardTree
from EMShrimulator.app.views.layout.copper_tool import VIA_REF, is_via
from EMShrimulator.app.views.layout.import_dialog import ImportReportDialog
from EMShrimulator.app.views.layout.padstack_page import PadstackPage
from EMShrimulator.app.views.layout.port_editor import PortInspector
from EMShrimulator.app.views.layout.stackup_panel import StackupPanel
from EMShrimulator.app.views.results.results_view import ResultsView
from EMShrimulator.app.views.shell import Page, fit_inspector
from EMShrimulator.app.views.tree_base import KEY_ROLE
from EMShrimulator.app.views.workspace import Workspace
from EMShrimulator.app.widgets.form import fmt_eng, hint
from EMShrimulator.app.widgets.icons import swatch
from EMShrimulator.core import procpool
from EMShrimulator.core.em.edits import PLOT_MENU
from EMShrimulator.core.layout.copper_geometry import REGION, TRACE
from EMShrimulator.core.layout.edits import copper_edits, pair_of
from EMShrimulator.core.layout import via_import
from EMShrimulator.core.layout.padstacks import layer_names
from EMShrimulator.core.services import layout as layout_service

GERBER_FILTER = ("Gerber exports (*.zip *.tgz *.tar *.gz *.gbr *.ger *.drl *.xln *.txt *.nc *.gtl *.gbl *.gko "
                 "*.gts *.gbs *.gto *.gbo *.gm1 *.g1 *.g2 *.g3 *.g4 *.json);;All files (*)")
ODB_FILTER = "ODB++ product model (*.tgz *.tar.gz *.zip *.tar);;All files (*)"
# Port tools: (key, label, tooltip).
TOOLS = [
    ("edge", "Edge port",
     "Hover a copper edge - a trace end, pad side or pour edge - and click. The port spans the edge "
     "and references the nearest plane below or above it."),
    ("circuit", "Circuit port",
     "Click the + terminal, then the − terminal. Both snap to centres, ends and edges, and can be on "
     "the same layer (a coplanar ground) or different ones."),
    ("point", "Point port", "Click to drop a port at a snapped point (Ctrl for a free point)."),
    ("pair", "Differential pair",
     "Click the + port, then the − port. The two are then one differential port: the results add its "
     "mixed-mode S (Sdd, Scc, Sdc, Scd) and its differential and common impedances."),
]
PORT_TOOLS = tuple(tool for tool, _label, _tip in TOOLS)
VIA_TOOL = ("via", "Place via",
            "Click to place a via of the current padstack - the selected via's, or the last placed - snapped to "
            "centres and ends (Ctrl for a free point). Keeps placing until Esc.")
STEPS_KEY = "layout/steps"
COLOR_KEY = "layout/color_by"
COLOR_MODES = (("layer", "Colour by layer"), ("net", "Colour by net"))


class LayoutView(Workspace):
    kind = "layout"
    layout_version = 3  # the old dock layout's saved state is not restored
    report_menu = PLOT_MENU[:1]  # gerber2ems solves S-parameters only

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        s = session
        self.board: Dict = {"ready": False, "layers": []}
        self.drills: List[Dict] = []
        self.visible_layers: Dict[str, bool] = {}  # shared with the canvas and the tree
        self.last_report: Optional[Dict] = None
        self._board_signature = None
        self._reference_signature = None
        self._copper_signature = None
        self._copper_busy = False  # a copper sync is writing the Gerbers; one at a time
        self._copper_again = False  # edits changed during a sync: sync again after
        self._copper_changed: set = set()  # layers rewritten since the canvas last reloaded
        self._reading_vias = False  # an old project's drill file is being read into vias
        self.via_padstack: Optional[str] = None  # what the via tool places
        # Independent background channels; each keeps only its newest call.
        self._background = BackgroundRunner(self, threads=3)
        procpool.warm()

        self.validation_label = QLabel("not validated")
        self.board_label = QLabel("")
        self.cursor_label = QLabel("")
        self.net_label = QLabel("")
        self.snap_label = QLabel("")
        self.render_label = QLabel("")
        self.status_widgets[1:1] = [self.validation_label, self.board_label, self.cursor_label, self.net_label,
                                    self.snap_label, self.render_label]

        # ---------------------------------------------------------- board
        self.canvas = BoardCanvas()
        self.canvas.raster_loader = lambda layer, ppm: layout_service.layer_alpha(s.name, layer, ppm)
        self.canvas.shape_loader = lambda layer: layout_service.layer_shapes(s.name, layer)
        self.canvas.visible_layers = self.visible_layers
        self.canvas.placeRequested.connect(self._place)
        self.canvas.edgePortRequested.connect(self._place_edge)
        self.canvas.circuitPortRequested.connect(self._place_circuit)
        self.canvas.pairRequested.connect(self._make_pair)
        self.canvas.viaRequested.connect(self._place_via)
        self.canvas.portSelected.connect(lambda n: s.select("layout_port", n))
        self.canvas.pairSelected.connect(lambda i: s.select("diff_pair", i))
        self.canvas.netHovered.connect(lambda text: self.net_label.setText(f"net {text}" if text else ""))
        self.canvas.netActivated.connect(lambda net: self.show_key(("net", net)))
        self.canvas.pairMenuRequested.connect(lambda i, pos: self._tree_menu(("diff_pair", i), pos))
        self.canvas.portMoved.connect(
            lambda n, x, y, snap: guarded(lambda: s.push(cmd.MoveLayoutPort(s, n, x, y, snap or None))))
        self.canvas.terminalMoved.connect(
            lambda n, which, x, y, layer, snap: guarded(lambda: s.push(cmd.MoveLayoutPortTerminal(
                s, n, which, x, y, layer if layer >= 0 else None, snap or None))))
        self.canvas.toolFinished.connect(self._tool_finished)
        self.canvas.snapChanged.connect(self.snap_label.setText)
        self.canvas.emptyClicked.connect(lambda: s.select(None))
        self.canvas.cursorMoved.connect(lambda x, y: self.cursor_label.setText(f"cursor {x:.4f}, {y:.4f} mm"))
        self.canvas.renderingChanged.connect(lambda busy: self.render_label.setText("rendering copper…" if busy else ""))
        self.canvas.contextRequested.connect(self._canvas_menu)
        self.canvas.copperMenuRequested.connect(self._copper_menu)
        self.canvas.copper.selectionChanged.connect(self._copper_selected)
        self.canvas.copper.editRequested.connect(self._edit_copper)
        self.board_stack = QStackedWidget()
        self.board_stack.addWidget(self._empty_board())
        self.board_stack.addWidget(self.canvas)
        self.inspector = PortInspector(s)
        self.inspector.shownChanged.connect(self._inspector_shown)
        self.inspector.setVisible(False)
        self.inspector.editor.references = {}
        self.inspector.copper.tool = self.canvas.copper
        self.inspector.vias.tool = self.canvas.copper
        self.inspector.vias.padstackRequested.connect(lambda name: self.show_key(("padstack", name)))
        self.board_split = QSplitter(Qt.Horizontal)
        self.board_split.setChildrenCollapsible(False)
        self.board_split.addWidget(self.board_stack)
        self.board_split.addWidget(self.inspector)
        self.board_split.setStretchFactor(0, 1)
        self.board_split.setStretchFactor(1, 0)
        self._splitters["board_split"] = self.board_split

        # ---------------------------------------------------------- pages
        self.files = BoardFiles()
        self.files.deleteFile.connect(self.delete_file)
        self.files.clearFiles.connect(self.clear_files)
        self.files.showReport.connect(self.show_report)
        self.stackup = StackupPanel(s)
        self.settings = BoardSettings(s)
        self.results = ResultsView(lambda: layout_service.results(s.name, s.project))
        import_gerbers = QToolButton()
        import_gerbers.setText("Import Gerbers…")
        import_gerbers.setToolTip("A .zip of the whole Gerber export, or individual layer files")
        import_gerbers.clicked.connect(self.import_gerbers)
        import_odb = QToolButton()
        import_odb.setText("ODB++…")
        import_odb.clicked.connect(self.import_odb)
        self.padstack_page = PadstackPage(s)
        self.padstack_page.selectVias.connect(self._select_padstack_vias)
        self.padstack_page.padstackShown.connect(lambda name: self.tree.set_current(("padstack", name)))
        padstack_buttons = []
        for text, tip, slot in (
                ("New padstack", "A plain through via, to set up and place", lambda: self._new_padstack(None)),
                ("Duplicate", "A copy of the padstack shown", lambda: self._new_padstack(self.padstack_page.name)),
                ("Delete", "Delete the padstack shown; no via may use it", self._delete_padstack)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            padstack_buttons.append(b)
        pages: Dict[str, QWidget] = {
            "board": self.board_split,
            "files": Page("Board files", self.files, "What the board is drawn and simulated from.",
                          [import_gerbers, import_odb]),
            "padstacks": Page("Padstacks", self.padstack_page,
                              "Via definitions. Every via on the board is an instance of one.", padstack_buttons),
            "stackup": Page("Stackup", self.stackup,
                            "The board's layers, top to bottom: the copper, and the dielectric between."),
            "setup": Page("Solution setup", self.settings,
                          "Over which frequencies, for how long and how finely the board is solved."),
            **self.report_pages(self.results, "S-parameters of the newest run. Make a report of your own with New "
                                              "report - S, Y, Z, VSWR, delay or TDR, in any display; it is listed "
                                              "under Results in the project tree."),
            **self.output_pages(),
        }

        # ----------------------------------------------------- project tree
        self.tree = BoardTree(s)
        self.tree.menuRequested.connect(self._tree_menu)
        self.tree.layerToggled.connect(self._toggle_layer)
        self.tree.holesToggled.connect(self._set_holes)
        self.tree.tree.itemDoubleClicked.connect(self._tree_double_clicked)
        self.build_shell(self.tree, pages, "board")

        # -------------------------------------------- actions, menus, toolbar
        self._build_actions()
        self._menus = [self._layout_menu(), self._simulation_menu(), self._results_menu()]
        self._build_toolbar()

        self._validate_timer = QTimer(self)
        self._validate_timer.setSingleShot(True)
        self._validate_timer.setInterval(250)
        self._validate_timer.timeout.connect(self._validate)
        s.projectChanged.connect(self._project_changed)
        s.selectionChanged.connect(self._selection_changed)

        self.restore_layout()
        self.reload_board()
        self._project_changed()
        self.tree.set_current(("project",))
        QTimer.singleShot(0, self.results.reload)

    def _empty_board(self) -> QWidget:
        """In the board's place until there is one: what is missing, and the imports."""
        self.empty_title = QLabel("No board yet")
        self.empty_title.setProperty("role", "page-title")
        self.empty_title.setAlignment(Qt.AlignHCenter)
        self.empty_text = hint(IMPORT_HINT)
        self.empty_text.setAlignment(Qt.AlignHCenter)
        gerbers = QPushButton("Import Gerbers…")
        gerbers.setProperty("role", "primary")
        gerbers.clicked.connect(self.import_gerbers)
        odb = QPushButton("ODB++…")
        odb.clicked.connect(self.import_odb)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(gerbers)
        buttons.addWidget(odb)
        buttons.addStretch(1)
        column = QWidget()
        column.setFixedWidth(480)
        inner = QVBoxLayout(column)
        inner.setSpacing(10)
        inner.addWidget(self.empty_title)
        inner.addWidget(self.empty_text)
        inner.addLayout(buttons)
        middle = QHBoxLayout()
        middle.addStretch(1)
        middle.addWidget(column)
        middle.addStretch(1)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.addStretch(1)
        layout.addLayout(middle)
        layout.addStretch(2)
        return host

    # ----------------------------------------------------------- layout

    def restore_layout(self) -> None:
        super().restore_layout()
        # The inspector follows the selection, not the saved state.
        self.inspector.setVisible(self.inspector.shown)

    def menus(self) -> List[QMenu]:
        return list(self._menus)

    # ------------------------------------------------------- navigation

    def show_key(self, key) -> None:
        """Show what a project-tree item stands for."""
        kind = key[0]
        s = self.session
        if kind in ("project", "board"):
            s.select(None)  # the whole board: nothing in particular
            self.canvas.set_highlight(None)
            self.show_page("board")
        elif kind in ("layers", "layer", "holes", "ports", "pairs"):
            self.show_page("board")
        elif kind == "nets":
            self.canvas.set_highlight(None)
            self.show_page("board")
        elif kind == "net":
            self.tree.open_nets()
            self.canvas.set_highlight(key[1])
            self.show_page("board")
        elif kind == "layout_port":
            s.select("layout_port", key[1])
            self.show_page("board")
        elif kind == "diff_pair":
            s.select("diff_pair", key[1])
            self.show_page("board")
        elif kind in ("files", "file"):
            self.show_page("files")
        elif kind in ("padstacks", "padstack"):
            if len(key) > 1 and key[1]:
                self.padstack_page.show_padstack(key[1])
            self.show_page("padstacks")
        elif kind == "stackup":
            if len(key) > 1:
                s.select("stackup", key[1])
            self.show_page("stackup")
        elif kind == "setup":
            self.show_page("setup")
            self.settings.scroll_to(key[1] if len(key) > 1 else "")
        elif kind == "results":
            self.show_page("results")
            self.tree.set_badge(("results",), "")
        elif kind == "plot":
            self.show_plot(key[1])
        elif kind in ("simulation", "log"):
            self.show_page("log")
        elif kind in ("validation", "runs"):
            self.show_page(kind)
        if self.tree.current_key() != key:
            self.tree.set_current(key)

    # ---------------------------------------------------------- actions

    def _build_actions(self) -> None:
        self.tool_actions: Dict[str, QAction] = {}
        for tool, label, tip in TOOLS + [VIA_TOOL]:
            action = QAction(label, self, checkable=True)
            action.setToolTip(tip + " Esc cancels.")
            action.toggled.connect(lambda on, tool=tool: self._select_tool(tool, on))
            self.tool_actions[tool] = action
        self.snap_action = QAction("Snap to copper", self, checkable=True)
        self.snap_action.setChecked(True)
        self.snap_action.setToolTip("Snap to via and pad centres, trace ends, edge centres, vertices and face "
                                    "centres. Hold Ctrl to place freely.")
        self.snap_action.toggled.connect(lambda on: setattr(self.canvas, "snapping", on))
        self.holes_action = QAction("Show holes", self, checkable=True)
        self.holes_action.setChecked(True)
        self.holes_action.toggled.connect(self._set_holes)
        self.fit_action = QAction("Fit board", self)
        self.fit_action.triggered.connect(self.canvas.fit)
        self.color_group = QActionGroup(self)
        stored = str(QSettings().value(COLOR_KEY) or "layer")
        for mode, label in COLOR_MODES:
            action = QAction(label, self.color_group, checkable=True)
            action.setData(mode)
            action.setChecked(mode == stored)
        if self.color_group.checkedAction() is None:
            self.color_group.actions()[0].setChecked(True)
        self.color_group.triggered.connect(self._color_chosen)
        self.canvas.set_color_by(str(self.color_group.checkedAction().data()))
        self.clear_highlight_action = QAction("Clear net highlight", self)
        self.clear_highlight_action.triggered.connect(lambda: self.show_key(("nets",)))
        self.renumber_action = QAction("Renumber ports", self)
        self.renumber_action.setToolTip("Number the ports P1…PN in their current order, closing any gaps; the "
                                        "nets under test follow")
        self.renumber_action.triggered.connect(
            lambda: guarded(lambda: self.session.push(cmd.RenumberLayoutPorts(self.session))))
        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut(QKeySequence.Delete)
        self.delete_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.delete_action.triggered.connect(self.delete_selected)
        self.canvas.addAction(self.delete_action)
        self.tree.tree.addAction(self.delete_action)
        self.validate_action = QAction("Validation", self)
        self.validate_action.triggered.connect(lambda: self.show_key(("validation",)))
        self.steps_group = QActionGroup(self)
        current = str(QSettings().value(STEPS_KEY) or "all")
        for step, label in layout_service.STEP_LABELS.items():
            action = QAction(label, self.steps_group, checkable=True)
            action.setData(step)
            action.setChecked(step == current)
        if self.steps_group.checkedAction() is None:
            self.steps_group.actions()[0].setChecked(True)
        self.steps_group.triggered.connect(self._steps_chosen)
        self._steps_chosen(self.steps_group.checkedAction(), store=False)

    def _layout_menu(self) -> QMenu:
        menu = QMenu("&Layout", self)
        menu.setToolTipsVisible(True)
        menu.addAction("Import Gerbers…", self.import_gerbers)
        menu.addAction("Import ODB++…", self.import_odb)
        menu.addAction("Board files", lambda: self.show_key(("files",)))
        menu.addSeparator()
        menu.addAction("Stackup", lambda: self.show_key(("stackup",)))
        self._preset_menu(menu.addMenu("Apply stackup preset"))
        menu.addSeparator()
        self.layers_menu = menu.addMenu("Copper layers")
        self.layers_menu.aboutToShow.connect(lambda: self._fill_layers_menu(self.layers_menu))
        self._fill_layers_menu(self.layers_menu)
        menu.addActions(self.color_group.actions())
        menu.addAction(self.clear_highlight_action)
        menu.addAction(self.holes_action)
        menu.addAction(self.snap_action)
        menu.addSeparator()
        menu.addAction(self.fit_action)
        menu.addSeparator()
        menu.addAction(self.tool_actions["via"])
        menu.addAction("Padstacks", lambda: self.show_key(("padstacks",)))
        return menu

    def _port_tool_actions(self) -> List[QAction]:
        return [self.tool_actions[tool] for tool in PORT_TOOLS]

    def _simulation_menu(self) -> QMenu:
        menu = QMenu("&Simulation", self)
        menu.setToolTipsVisible(True)
        excite = menu.addMenu("Assign excitation")
        excite.setToolTipsVisible(True)
        excite.addActions([a for tool, a in self.tool_actions.items() if tool in PORT_TOOLS and tool != "pair"])
        menu.addAction(self.tool_actions["pair"])
        menu.addAction("Differential pairs", lambda: self.show_key(("pairs",)))
        menu.addAction(self.renumber_action)
        menu.addSeparator()
        menu.addAction("Solution setup", lambda: self.show_key(("setup",)))
        menu.addAction("Nets under test", lambda: self.show_key(("setup", "nets")))
        menu.addSeparator()
        menu.addAction(self.validate_action)
        steps = menu.addMenu("Run steps")
        steps.addActions(self.steps_group.actions())
        menu.addAction(self.run_action)
        menu.addAction(self.stop_action)
        menu.addSeparator()
        menu.addAction("Solver log", lambda: self.show_key(("log",)))
        menu.addAction("Runs", lambda: self.show_key(("runs",)))
        menu.addAction("Show simulation.json…", self.show_simulation_json)
        return menu

    def _results_menu(self) -> QMenu:
        menu = QMenu("&Results", self)
        self.fill_report_menu(menu.addMenu("New report"))
        menu.addSeparator()
        menu.addAction("Results overview", lambda: self.show_key(("results",)))
        menu.addAction("Reload results", self.results.reload)
        menu.addAction("Save Touchstone…", self.results.export_touchstone)
        return menu

    def _preset_menu(self, menu: QMenu) -> QMenu:
        for preset in layout_service.stackup_presets():
            menu.addAction(preset["title"], lambda pid=preset["id"]: self.stackup.apply_preset(pid))
        return menu

    def _fill_layers_menu(self, menu: QMenu) -> None:
        menu.clear()
        layers = self.board.get("layers", [])
        for layer in layers:
            action = menu.addAction(swatch(layer["color"]), layer["name"])
            action.setCheckable(True)
            action.setChecked(self.visible_layers.get(layer["name"], True))
            action.toggled.connect(lambda on, name=layer["name"]: self._toggle_layer(name, on))
        outline = menu.addAction("Board outline")
        outline.setCheckable(True)
        outline.setChecked(self.visible_layers.get(OUTLINE, True))
        outline.toggled.connect(lambda on: self._toggle_layer(OUTLINE, on))
        menu.addSeparator()
        menu.addAction("Show all", lambda: self._show_layers(None))
        menu.setEnabled(bool(layers))

    def _build_toolbar(self) -> None:
        bar = QToolBar("Layout")
        bar.setObjectName("layout-toolbar")
        bar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, bar)
        for action in self._port_tool_actions():
            bar.addAction(action)
        bar.addSeparator()
        bar.addAction(self.tool_actions["via"])
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        bar.addAction(self.save_action)
        bar.addAction(self.run_action)
        bar.widgetForAction(self.run_action).setProperty("role", "run")
        bar.addAction(self.stop_action)

    def _color_chosen(self, action: QAction) -> None:
        QSettings().setValue(COLOR_KEY, action.data())
        self.canvas.set_color_by(str(action.data()))
        if action.data() == "net" and self.board.get("ready") and not self.canvas.has_nets():
            notify("These Gerbers name no nets (X2 %TO.N% attributes), so all copper shows as one colour. "
                   "KiCad writes them with \"Include netlist attributes\" ticked.", "warn")

    def _steps_chosen(self, action: QAction, store: bool = True) -> None:
        if store:
            QSettings().setValue(STEPS_KEY, action.data())
        self.run_action.setToolTip(f"Validate, save and run: {action.text().lower()} (F5). "
                                   "Choose under Simulation › Run steps.")

    def _steps(self) -> str:
        action = self.steps_group.checkedAction()
        return str(action.data()) if action is not None else "all"

    # --------------------------------------------------------------- board

    def _signature(self):
        p = self.session.project
        return (tuple((layer.name, layer.type) for layer in p.stackup), tuple(p.gerber_files), p.drill_file)

    def _copper_sig(self):
        p = self.session.project
        return (tuple((e.layer, e.index, tuple((o.kind, tuple(o.points), tuple(o.heights), o.width_mm)
                                               for o in e.objects))
                      for e in p.copper_edits),
                None if p.vias is None else tuple(
                    (v.id, v.padstack, v.x_mm, v.y_mm, v.net, repr(v.backdrill_top), repr(v.backdrill_bottom),
                     len(v.links)) for v in p.vias),
                tuple(repr(ps) for ps in p.padstacks))

    def _set_vias(self) -> None:
        """The copper tool, and through it the canvas, take the project's vias."""
        p = self.session.project
        self.canvas.copper.set_vias(p.vias, p.padstacks, layer_names(p))

    def _hole_count(self) -> int:
        vias = self.session.project.vias
        return len(self.drills) if vias is None else len(vias)

    def _read_vias(self) -> None:
        """Read an older project's drill file into vias once, on opening."""
        s = self.session
        if self._reading_vias or s.project.vias is not None or not self.board.get("ready"):
            return
        self._reading_vias = True
        self.render_label.setText("reading the drill file into vias…")
        project = s.project.model_copy(deep=True)

        def done(found) -> None:
            self._reading_vias = False
            self.render_label.setText("")
            if s.project.vias is not None:
                return
            # Applied to the project as it is now.
            new_project = s.project.model_copy(deep=True)
            via_import.apply(new_project, found)
            s.replace_external(new_project)
            count = len(new_project.vias or [])
            if count:
                notify(f"{count} vias read from the drill file into {len(new_project.padstacks)} padstacks, "
                       f"{found.get('note', '')}.", "ok")

        def failed(exc: Exception) -> None:
            self._reading_vias = False
            self.render_label.setText("")
            notify(f"Could not read the drill file into vias: {exc}", "warn")

        self._background.submit("vias", lambda: layout_service.read_vias(s.name, project), done, failed)

    def reload_board(self) -> None:
        s = self.session
        project = s.project
        self._board_signature = self._signature()
        self._copper_signature = self._copper_sig()
        # canvas_data writes the edited copper first.
        self._background.submit("board", lambda: layout_service.canvas_data(s.name, project), self._board_loaded,
                                lambda exc: notify(f"Could not read the board: {exc}", "error"))

    def _load_snaps(self) -> None:
        s = self.session
        project = s.project
        self._background.submit("snaps", lambda: layout_service.snap_targets(s.name, project),
                                self.canvas.set_snaps,
                                lambda exc: notify(f"Snap targets unavailable: {exc}", "warn"))

    def _load_copper(self, layers: List[Dict]) -> None:
        """Each layer's imported copper as selectable pieces."""
        name = self.session.name
        for layer in layers:
            layer_name = layer["name"]
            self._background.submit(
                f"pieces:{layer_name}", lambda n=layer_name: layout_service.copper_pieces(name, n),
                lambda data, n=layer_name: self.canvas.copper.set_layer(n, data),
                lambda exc, n=layer_name: notify(f"The copper of {n} cannot be edited: {exc}", "warn"))

    def _board_loaded(self, payload) -> None:
        board, drills = payload
        self.board, self.drills = board, drills
        for layer in board.get("layers", []):
            self.visible_layers.setdefault(layer["name"], True)
        ready = bool(board.get("ready"))
        # Vias draw their own holes; the drill file's are used until it is read into vias.
        self.canvas.set_board(board, drills if self.session.project.vias is None else [])
        self.canvas.copper.set_edits(copper_edits(self.session.project))
        self._set_vias()
        self.padstack_page.set_layer_colors(board.get("layers", []))
        if ready:
            self._load_copper(board.get("layers", []))
            self._load_snaps()
            self._read_vias()
        self.board_stack.setCurrentIndex(1 if ready else 0)
        p = self.session.project
        has_files = bool(p.gerber_files or p.drill_file)
        self.empty_title.setText("The board cannot be drawn yet" if has_files else "No board yet")
        self.empty_text.setText((board.get("reason") or IMPORT_HINT) if has_files else IMPORT_HINT)
        self._sync_ports()
        self.stackup.set_board_layers(board.get("layers", []))
        self.files.show_files(p, len(drills))
        self.tree.set_board(board, self._hole_count(), self.visible_layers, self.holes_action.isChecked())
        self._fill_layers_menu(self.layers_menu)
        for action in self.tool_actions.values():
            action.setEnabled(ready)
        self._reference_signature = None
        self._refresh_references()
        self._update_status()

    def _refresh_references(self) -> None:
        """Resolve "auto" port references off the UI thread."""
        s = self.session
        p = s.project
        signature = (self._board_signature,
                     tuple((q.number, q.kind.value, q.reference, q.layer, q.x_mm, q.y_mm, q.width_um) for q in p.ports))
        if signature == self._reference_signature or not self.board.get("ready"):
            return
        self._reference_signature = signature
        project = p.model_copy(deep=True)

        def done(choices) -> None:
            self.canvas.set_references(choices)
            self.inspector.editor.references = choices
            self.inspector.refresh()
            self.tree.set_references(choices)

        self._background.submit("references", lambda: layout_service.resolve_references(s.name, project), done,
                                lambda exc: notify(f"Could not resolve port references: {exc}", "warn"))

    def _select_tool(self, tool: str, on: bool) -> None:
        if on:
            for other, action in self.tool_actions.items():
                if other != tool and action.isChecked():
                    action.blockSignals(True)
                    action.setChecked(False)
                    action.blockSignals(False)
            self.show_page("board")
            self.canvas.set_tool(tool)
            self.canvas.setFocus()
        elif self.canvas.tool == tool:
            self.canvas.set_tool("select")

    def _tool_finished(self) -> None:
        for action in self.tool_actions.values():
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)

    def _toggle_layer(self, name: str, visible: bool) -> None:
        if self.visible_layers.get(name, True) == visible:
            return
        self.visible_layers[name] = visible
        self.canvas.apply_layer_visibility()
        self.tree.refresh()

    def _show_layers(self, only: Optional[str]) -> None:
        """Every copper layer, or `only` that one; the outline stays as it is."""
        for layer in self.board.get("layers", []):
            self.visible_layers[layer["name"]] = only is None or layer["name"] == only
        self.canvas.apply_layer_visibility()
        self.tree.refresh()

    def _set_holes(self, on: bool) -> None:
        if self.holes_action.isChecked() != on:
            self.holes_action.blockSignals(True)
            self.holes_action.setChecked(on)
            self.holes_action.blockSignals(False)
        self.canvas.set_drills_visible(on)
        self.tree.set_board(self.board, self._hole_count(), self.visible_layers, on)

    # ------------------------------------------------------------ changes

    def _project_changed(self) -> None:
        if self._signature() != self._board_signature:
            self.reload_board()
        else:
            self.files.show_files(self.session.project, len(self.drills))
            if self._copper_sig() != self._copper_signature:
                self._copper_edits_changed()
        self._sync_ports()
        self._refresh_references()
        self._validate_timer.start()
        self._update_status()
        self.follow_plot_edits()

    def _selection_changed(self, selection) -> None:
        self._sync_ports()
        self.canvas.copper.set_selection(selection.key if selection is not None and selection.kind == "copper"
                                         else ())
        if selection is None:
            return
        if selection.kind in ("layout_port", "diff_pair", "copper") and self._page_key != "board":
            self.show_page("board")
        elif selection.kind == "stackup" and self._page_key != "stackup":
            self.show_page("stackup")

    def _sync_ports(self) -> None:
        selection = self.session.selection
        selected = selection.key if selection is not None and selection.kind == "layout_port" else None
        pair = int(selection.key) if selection is not None and selection.kind == "diff_pair" else None
        if self.board.get("ready"):
            p = self.session.project
            self.canvas.set_ports(p.ports, selected, p.diff_pairs, pair)

    def _inspector_shown(self, shown: bool) -> None:
        self.inspector.setVisible(shown)
        fit_inspector(self.board_split, shown)

    def _validate(self) -> None:
        """Check the board off the UI thread: the reference checks read the copper masks."""
        s = self.session
        project = s.project.model_copy(deep=True)
        self._background.submit("validate", lambda: layout_service.validate(s.name, project), self._validated,
                                lambda exc: notify(f"Could not validate the board: {exc}", "warn"))

    def _validated(self, report) -> None:
        self.validation.show_report(report)
        self.validation_label.setText(f"{report['errors']} error(s), {report['warnings']} warning(s)")
        self.validation_badge(report)

    def _update_status(self) -> None:
        p = self.session.project
        board = self.board.get("board")
        size = f"board {board['width_mm']:.1f} × {board['height_mm']:.1f} mm · " if board else ""
        excited = sum(1 for port in p.ports if port.excite)
        edits = f" ({len(p.copper_edits)} edited)" if p.copper_edits else ""
        self.board_label.setText(
            f"{size}copper {len(self.board.get('layers', []))}{edits} · vias {self._hole_count()} · ports "
            f"{len(p.ports)} ({excited} excited) · {fmt_eng(p.frequency.start, 'Hz')}–{fmt_eng(p.frequency.stop, 'Hz')}")

    # ------------------------------------------------------------- copper

    def _copper_selected(self, refs) -> None:
        s = self.session
        if refs:
            s.select("copper", tuple(refs))
        elif s.selection is not None and s.selection.kind == "copper":
            s.select(None)

    def _edit_copper(self, label: str, changes, after, merge_key, vias=None) -> None:
        s = self.session
        done = guarded(lambda: s.push(cmd.EditCopper(s, label, changes, merge_key, vias)))
        if not done:
            self.canvas.copper.pending.clear()
            self.canvas.copper.refresh()
            return
        if after is not None:
            if after:
                s.select("copper", tuple(after))
            elif s.selection is not None and s.selection.kind == "copper":
                s.select(None)

    def _copper_edits_changed(self) -> None:
        """The copper changed: show it now, then rewrite the Gerbers."""
        old = self._copper_signature
        self._copper_signature = self._copper_sig()
        self.canvas.copper.set_edits(copper_edits(self.session.project))
        self._set_vias()
        if old is not None and (old[1] is None) != (self._copper_signature[1] is None):
            # The vias changed, so the snaps' via centres did too.
            if self.session.project.vias is None:
                self.canvas.set_drill_holes(self.drills)
            self._load_snaps()
        self.tree.set_board(self.board, self._hole_count(), self.visible_layers, self.holes_action.isChecked())
        self.inspector.refresh()
        self._sync_copper()

    def _sync_copper(self) -> None:
        if self._copper_busy:
            self._copper_again = True
            return
        s = self.session
        project = s.project  # edits replace the project object
        self._copper_busy = True
        self.render_label.setText("writing copper…")
        self._background.submit("copper", lambda: layout_service.sync_copper(s.name, project),
                                self._copper_synced, self._copper_failed)

    def _copper_synced(self, changed: List[str]) -> None:
        self._copper_busy = False
        self._copper_changed.update(changed)
        if self._copper_again:
            self._copper_again = False
            self._sync_copper()
            return
        self.render_label.setText("")
        layers, self._copper_changed = sorted(self._copper_changed), set()
        self.canvas.reload_layers(layers)
        for name in list(self.canvas.copper.pending):
            if name not in layers:
                self.canvas.copper.layer_redrawn(name)
        if layers:
            self._load_snaps()
            self._reference_signature = None
            self._refresh_references()
            self._validate_timer.start()

    def _copper_failed(self, exc: Exception) -> None:
        self._copper_busy = False
        self.render_label.setText("")
        notify(f"Could not write the edited copper: {exc}", "error")
        if self._copper_again:
            self._copper_again = False
            self._sync_copper()

    def _copper_menu(self, ref, handle, point, where) -> None:
        """Right-click on copper: what can be done to it, there."""
        tool = self.canvas.copper
        piece = tool.piece(ref)
        if piece is None:
            return
        if ref not in tool.selection:
            tool.select((ref,))
        refs = tool.selection
        editable = all(tool.editable(r) for r in refs)
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        if all(is_via(r) for r in refs):
            self._via_menu(menu, refs)
            menu.exec(where)
            return
        actions = []
        if piece.kind == TRACE and len(refs) == 1:
            split = menu.addAction("Split here", lambda: tool.split(ref, point))
            split.setToolTip("Make the segment two, joined at this point, each with ends of its own")
            actions.append(split)
            if piece.heights and abs(piece.heights[0]) > 1e-9:
                actions.append(menu.addAction("Straighten", lambda: tool.set_height(ref, 0.0)))
        if piece.kind == TRACE:
            menu.addAction("Select whole trace", lambda: tool.select_chain(ref))
        if piece.kind == REGION and len(refs) == 1:
            if handle is not None and handle.kind == "vertex":
                remove = menu.addAction("Delete vertex", lambda: tool.delete_vertex(ref, handle.i))
                remove.setEnabled(len(piece.points) > 3)
                actions.append(remove)
            else:
                actions.append(menu.addAction("Add vertex here", lambda: tool.insert_vertex(ref, point)))
        if piece.net:
            menu.addSeparator()
            menu.addAction(f"Highlight net {piece.net}", lambda: self.show_key(("net", piece.net)))
        menu.addSeparator()
        if any(tool.edited(r) for r in refs):
            actions.append(menu.addAction("Restore imported", lambda: tool.restore(refs)))
        actions.append(menu.addAction("Delete", lambda: tool.delete(refs)))
        for action in actions:
            action.setEnabled(action.isEnabled() and editable)
        menu.exec(where)

    def _via_menu(self, menu: QMenu, refs) -> None:
        """Right-click on vias: their padstack, its page, the rest of its vias, their net."""
        s = self.session
        tool = self.canvas.copper
        ids = [r[1] for r in refs]
        vias = [tool.vias[i] for i in ids if i in tool.vias]
        if not vias:
            return
        names = sorted({v.padstack for v in vias})
        change = menu.addMenu("Padstack")
        for ps in s.project.padstacks:
            action = change.addAction(ps.name, lambda n=ps.name: guarded(
                lambda: s.push(cmd.SetViasField(s, ids, "padstack", n))))
            action.setCheckable(True)
            action.setChecked(names == [ps.name])
        if len(names) == 1:
            menu.addAction(f"Edit padstack {names[0]}…", lambda: self.show_key(("padstack", names[0])))
            menu.addAction(f"Select all {names[0]} vias", lambda: self._select_padstack_vias(names[0]))
        nets = sorted({v.net for v in vias if v.net})
        if len(nets) == 1:
            menu.addSeparator()
            menu.addAction(f"Highlight net {nets[0]}", lambda: self.show_key(("net", nets[0])))
        menu.addSeparator()
        menu.addAction("Delete", lambda: tool.delete(refs))

    # ------------------------------------------------------------- edits

    def _place(self, x_mm: float, y_mm: float, layer: int, snap: str) -> None:
        layer = layer if layer >= 0 else self.canvas.default_layer()
        number = guarded(lambda: self.session.push(
            cmd.PlaceLayoutPort(self.session, x_mm, y_mm, layer, snap or None)))
        self._placed(number)

    def _place_edge(self, edge, snap: str) -> None:
        number = guarded(lambda: self.session.push(cmd.PlaceEdgePort(self.session, edge, snap or None)))
        self._placed(number)

    def _place_circuit(self, pos, neg, snap: str, neg_snap: str) -> None:
        number = guarded(lambda: self.session.push(
            cmd.PlaceCircuitPort(self.session, tuple(pos), tuple(neg), snap or None, neg_snap or None)))
        self._placed(number)

    def _placed(self, number: Optional[int]) -> None:
        self._tool_finished()
        if number is not None:
            self.session.select("layout_port", number)

    # -------------------------------------------------------------- vias

    def _current_padstack(self) -> Optional[str]:
        """The padstack the via tool places: the selected via's, else the last placed, else the first plated."""
        p = self.session.project
        names = {ps.name for ps in p.padstacks}
        selection = self.session.selection
        if selection is not None and selection.kind == "copper":
            vias = {v.id: v for v in p.vias or []}
            for ref in selection.key:
                if is_via(ref) and ref[1] in vias:
                    return vias[ref[1]].padstack
        if self.via_padstack in names:
            return self.via_padstack
        plated = [ps.name for ps in p.padstacks if ps.plated and ps.shape == "circle"]
        return plated[0] if plated else next(iter(sorted(names)), None)

    def _place_via(self, x_mm: float, y_mm: float, _snap: str) -> None:
        s = self.session
        name = self._current_padstack()
        if name is None:
            name = guarded(lambda: s.push(cmd.AddPadstack(s)))
            if name is None:
                return
            notify(f"No padstack yet: made {name}, a plain through via. Set it up under Padstacks.", "info")
        hit = self.canvas.net_at(x_mm, y_mm)
        net = hit[0] if hit is not None and hit[0] else None
        via_id = guarded(lambda: s.push(cmd.PlaceVia(s, x_mm, y_mm, name, net)))
        if via_id is not None:
            self.via_padstack = name
            s.select("copper", ((VIA_REF, via_id, 0),))

    def _select_padstack_vias(self, name: str) -> None:
        refs = tuple((VIA_REF, v.id, 0) for v in self.session.project.vias or [] if v.padstack == name)
        self.show_page("board")
        self.tree.set_current(("board",))
        if refs:
            self.session.select("copper", refs)
        else:
            notify(f"No via uses {name}.", "info")

    def _new_padstack(self, like: Optional[str]) -> None:
        s = self.session
        name = guarded(lambda: s.push(cmd.AddPadstack(s, like)))
        if name is not None:
            self.show_key(("padstack", name))

    def _delete_padstack(self, name: Optional[str] = None) -> None:
        s = self.session
        name = name or self.padstack_page.name
        if not name:
            return
        guarded(lambda: s.push(cmd.DeletePadstack(s, name)))
        if not any(ps.name == name for ps in s.project.padstacks):
            self.show_key(("padstacks",))

    def _make_pair(self, positive: int, negative: int) -> None:
        """The pair tool's two clicks: the first port is +, the second −."""
        index = guarded(lambda: self.session.push(cmd.AddDiffPair(self.session, positive, negative)))
        self._tool_finished()
        if index is not None:
            self.session.select("diff_pair", index)

    def delete_selected(self) -> None:
        selection = self.session.selection
        # Only with it in view: never delete what the user cannot see.
        if selection is None:
            return
        if selection.kind == "layout_port" and self._page_key == "board":
            guarded(lambda: self.session.push(cmd.DeleteLayoutPort(self.session, selection.key)))
        elif selection.kind == "diff_pair" and self._page_key == "board":
            guarded(lambda: self.session.push(cmd.DeleteDiffPair(self.session, int(selection.key))))
        elif selection.kind == "copper" and self._page_key == "board":
            self.canvas.copper.delete(selection.key)
        elif selection.kind == "stackup" and self._page_key == "stackup":
            self.stackup.delete_layer()

    def _set_excited(self, number: int, on: bool) -> None:
        guarded(lambda: self.session.push(cmd.SetLayoutPortField(self.session, number, "excite", on)))

    def _focus_port(self, number: int) -> None:
        self.show_page("board")
        self.canvas.focus_port(number)

    # ------------------------------------------------------------ imports

    def _import(self, title: str, work, report_dialog: bool) -> None:
        s = self.session
        project = s.project.model_copy(deep=True)
        self.render_label.setText(title)

        def done(result) -> None:
            new_project, report = result
            s.replace_external(new_project)
            self.render_label.setText("")
            self.reload_board()
            if report_dialog and report.get("files"):
                self.last_report = report
                self.files.report_button.setVisible(True)
                ImportReportDialog(report, self).exec()
            copper = report.get("copper_layers")
            nets = f", {report['nets']} nets" if report.get("nets") else ""
            notify(f"Imported {len(copper)} copper layers{nets}." if copper else "Imported.", "ok")
            if not report_dialog:
                for warning in report.get("warnings") or []:
                    notify(warning, "warn")

        def failed(exc: Exception) -> None:
            self.render_label.setText("")
            notify(str(exc), "error")

        self._background.submit("import", lambda: work(project), done, failed)

    def import_gerbers(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Import Gerbers", str(Path.home()), GERBER_FILTER)
        if files:
            name = self.session.name
            paths = [Path(f) for f in files]
            label = f"Unpacking {Path(files[0]).name}…" if len(files) == 1 else f"Importing {len(files)} files…"
            self._import(label, lambda project: layout_service.import_files(name, project, paths), True)

    def import_odb(self) -> None:
        source, _ = QFileDialog.getOpenFileName(self, "Import ODB++", str(Path.home()), ODB_FILTER)
        if source:
            name = self.session.name
            self._import("Converting the ODB++ product model…",
                         lambda project: layout_service.import_odbpp(name, project, Path(source)), False)

    def show_report(self) -> None:
        if self.last_report:
            ImportReportDialog(self.last_report, self).exec()

    def clear_files(self) -> None:
        answer = QMessageBox.question(self, "Clear board files",
                                      "Remove every imported board file from this project? Ports and settings are kept.")
        if answer == QMessageBox.Yes:
            project = guarded(layout_service.clear_files, self.session.name, self.session.project.model_copy(deep=True))
            if project is not None:
                self.session.replace_external(project)
                notify("Board files cleared.", "ok")

    def delete_file(self, filename: str) -> None:
        answer = QMessageBox.question(self, "Remove file", f"Remove {filename} from this board?")
        if answer == QMessageBox.Yes:
            project = guarded(layout_service.delete_file, self.session.name,
                              self.session.project.model_copy(deep=True), filename)
            if project is not None:
                self.session.replace_external(project)

    def show_simulation_json(self) -> None:
        data = guarded(layout_service.simulation_json, self.session.name, self.session.project)
        if data is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("simulation.json handed to gerber2ems")
        dialog.resize(640, 560)
        text = QPlainTextEdit(json.dumps(data, indent=2))
        text.setReadOnly(True)
        text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        info = QLabel("Written into the project directory on every run. This is gerber2ems's own config format — "
                      "nothing is hidden between this file and the solver.")
        info.setWordWrap(True)
        info.setProperty("role", "hint")
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(info)
        layout.addWidget(text, 1)
        layout.addWidget(buttons)
        dialog.exec()

    # ---------------------------------------------------- context menus

    def _tree_menu(self, key, pos) -> None:
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        kind = key[0]
        s = self.session
        if kind in ("project", "board"):
            menu.addActions(list(self.tool_actions.values()))
            menu.addSeparator()
            menu.addActions(self.color_group.actions())
            if self.canvas.highlight is not None:
                menu.addAction(self.clear_highlight_action)
            menu.addSeparator()
            menu.addAction("Import Gerbers…", self.import_gerbers)
            menu.addAction("Import ODB++…", self.import_odb)
            if s.project.copper_edits:
                menu.addAction("Restore all imported copper",
                               lambda: guarded(lambda: s.push(cmd.RevertCopper(s)))).setToolTip(
                    "Undo every copper edit, on every layer")
            menu.addSeparator()
            menu.addAction(self.fit_action)
        elif kind in ("nets", "net"):
            if kind == "net":
                menu.addAction("Highlight on the board", lambda: self.show_key(key))
                menu.addAction("Zoom to net", lambda: (self.show_key(key), self.canvas.focus_net(key[1])))
            if self.canvas.highlight is not None:
                menu.addAction(self.clear_highlight_action)
            menu.addSeparator()
            menu.addActions(self.color_group.actions())
        elif kind == "pairs":
            menu.addAction(self.tool_actions["pair"])
            menu.addAction("New empty pair", lambda: self._new_empty_pair())
        elif kind == "diff_pair":
            if not 0 <= key[1] < len(s.project.diff_pairs):
                return
            index = key[1]
            s.select("diff_pair", index)
            menu.addAction("Swap + and −", lambda: guarded(lambda: s.push(cmd.SwapDiffPair(s, index))))
            menu.addAction("Excite both ports", self.inspector.pair_editor._excite_both)
            menu.addAction("Zoom to", lambda: (self.show_page("board"), self.canvas.focus_pair(index)))
            menu.addSeparator()
            menu.addAction("Delete pair", lambda: guarded(lambda: s.push(cmd.DeleteDiffPair(s, index))))
        elif kind in ("layers", "holes"):
            menu.addAction("Show all layers", lambda: self._show_layers(None))
            menu.addAction(self.holes_action)
        elif kind == "layer":
            if key[1] != OUTLINE:
                menu.addAction("Show only this layer", lambda: self._show_layers(key[1]))
            menu.addAction("Show all layers", lambda: self._show_layers(None))
            if any(e.layer == key[1] for e in s.project.copper_edits):
                menu.addSeparator()
                menu.addAction("Restore imported copper", lambda: guarded(lambda: s.push(cmd.RevertCopper(s, key[1]))))
        elif kind == "files":
            menu.addAction("Import Gerbers…", self.import_gerbers)
            menu.addAction("Import ODB++…", self.import_odb)
            if self.last_report:
                menu.addAction("Import report", self.show_report)
            if s.project.gerber_files or s.project.drill_file:
                menu.addSeparator()
                menu.addAction("Clear all…", self.clear_files)
        elif kind == "file":
            menu.addAction("Remove…", lambda: self.delete_file(key[1]))
        elif kind == "padstacks":
            menu.addAction("New padstack", lambda: self._new_padstack(None))
            menu.addAction(self.tool_actions["via"])
        elif kind == "padstack":
            menu.addAction("Select its vias", lambda: self._select_padstack_vias(key[1]))
            menu.addAction("Place vias of it", lambda: (setattr(self, "via_padstack", key[1]),
                                                        self.tool_actions["via"].setChecked(True)))
            menu.addAction("Duplicate", lambda: self._new_padstack(key[1]))
            menu.addSeparator()
            menu.addAction("Delete", lambda: self._delete_padstack(key[1]))
        elif kind == "stackup" and len(key) == 1:
            self._preset_menu(menu.addMenu("Apply preset"))
            menu.addAction("Add layer", self.stackup.add_layer)
        elif kind == "stackup":
            s.select("stackup", key[1])
            menu.addAction("Add layer below", self.stackup.add_layer)
            menu.addAction("Move up", lambda: self.stackup.move_layer(-1)).setEnabled(key[1] > 0)
            menu.addAction("Move down", lambda: self.stackup.move_layer(1)).setEnabled(
                key[1] < len(s.project.stackup) - 1)
            menu.addSeparator()
            menu.addAction("Delete", self.stackup.delete_layer)
        elif kind == "ports":
            menu.addActions(self._port_tool_actions())
            menu.addSeparator()
            menu.addAction(self.renumber_action)
        elif kind == "layout_port":
            port = next((p for p in s.project.ports if p.number == key[1]), None)
            if port is None:
                return
            s.select("layout_port", port.number)
            excite = menu.addAction("Excite this port")
            excite.setCheckable(True)
            excite.setChecked(port.excite)
            excite.triggered.connect(lambda on: self._set_excited(port.number, on))
            excite.setToolTip("An excited port is driven in a run of its own and gives one column of S")
            menu.addAction("Zoom to", lambda: self._focus_port(port.number))
            owner = pair_of(s.project, port.number)
            if owner is not None:
                menu.addAction(f"Differential pair {owner[1].name}", lambda: self.show_key(("diff_pair", owner[0])))
            else:
                paired = {n for d in s.project.diff_pairs for n in (d.positive, d.negative)}
                others = [q for q in sorted(s.project.ports, key=lambda q: q.number)
                          if q.number != port.number and q.number not in paired]
                pairs_menu = menu.addMenu("Differential pair with")
                pairs_menu.setEnabled(bool(others))
                for other in others:
                    pairs_menu.addAction(f"P{other.number + 1}   (P{port.number + 1} +, P{other.number + 1} −)",
                                         lambda o=other.number: self._make_pair(port.number, o))
            menu.addSeparator()
            menu.addAction("Delete", lambda: guarded(lambda: s.push(cmd.DeleteLayoutPort(s, port.number))))
        elif kind == "setup" and key[1:] == ("nets",):
            menu.addAction("Add trace", lambda: guarded(lambda: s.push(cmd.AddTrace(s))))
            menu.addAction("Differential pairs", lambda: self.show_key(("pairs",)))
        elif kind == "results":
            self.report_tree_menu(menu, key)
            menu.addAction("Reload results", self.results.reload)
            menu.addAction("Save Touchstone…", self.results.export_touchstone)
        elif kind == "plot":
            self.report_tree_menu(menu, key)
        if not menu.isEmpty():
            menu.exec(pos)

    def _canvas_menu(self, number: int, pos) -> None:
        """Right-click on the board: the port's tree menu, or the board's on empty copper."""
        self._tree_menu(("layout_port", number) if number >= 0 else ("board",), pos)

    def _new_empty_pair(self) -> None:
        index = guarded(lambda: self.session.push(cmd.AddDiffPair(self.session)))
        if index is not None:
            self.show_key(("diff_pair", index))

    def _tree_double_clicked(self, item, _column: int) -> None:
        key = item.data(0, KEY_ROLE)
        if key and key[0] == "layout_port":
            self._focus_port(key[1])
        elif key and key[0] == "diff_pair":
            self.canvas.focus_pair(key[1])
        elif key and key[0] == "net":
            self.canvas.focus_net(key[1])

    # -------------------------------------------------------------- runs

    def start_job(self):
        return layout_service.start_run(self.session.name, self.session.project, self._steps())

    def on_results_ready(self) -> None:
        self.results.reload()
        self.show_output(self.results)

    def select_by_name(self, target: str) -> None:
        p = self.session.project
        port = next((x for x in p.ports if x.name == target), None)
        if port is not None:
            self.show_key(("layout_port", port.number))
            return
        pair = next((i for i, d in enumerate(p.diff_pairs) if d.name == target), None)
        if pair is not None:
            self.show_key(("diff_pair", pair))
            return
        index = next((i for i, layer in enumerate(p.stackup) if layer.name == target), None)
        if index is not None:
            self.show_key(("stackup", index))

    def shutdown(self) -> None:
        super().shutdown()
        self._validate_timer.stop()
        self._background.shutdown()
        self.canvas.shutdown()
        self.results.shutdown()
