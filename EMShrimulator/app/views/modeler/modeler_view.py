"""The 3D Modeler workspace: project tree, centre page and inspector, on the shared `Workspace` shell."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QInputDialog,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolBar,
    QToolButton,
    QWidget,
)

from EMShrimulator.app import commands as cmd
from EMShrimulator.app.notify import guarded, notify
from EMShrimulator.app.preview import PreviewScheduler
from EMShrimulator.app.views.modeler.boolean_dialog import BooleanDialog
from EMShrimulator.app.views.modeler.parameter_table import ParameterTable
from EMShrimulator.app.views.modeler.project_tree import KEY_ROLE, ProjectTree
from EMShrimulator.app.views.modeler.property_panel import PropertyPanel
from EMShrimulator.app.views.modeler.solve_settings import SolveSettings, set_open_region
from EMShrimulator.app.views.modeler.sweep_editor import SweepEditor
from EMShrimulator.app.views.results.results_view import ResultsView
from EMShrimulator.app.views.shell import Page, fit_inspector
from EMShrimulator.app.views.viewport3d import Viewport3D
from EMShrimulator.app.views.workspace import Workspace
from EMShrimulator.app.widgets.form import fmt_count, fmt_eng
from EMShrimulator.app.views.modeler import units as unit_math
from EMShrimulator.core.em import drawing, frames, sheets, snaps
from EMShrimulator.core.em.edits import (
    BOOLEAN_LABELS,
    PRIMITIVE_LABELS,
    SHEET_PORT_NEEDS,
    find_coordinate_system,
    find_port,
)
from EMShrimulator.core.em.model import BooleanOp, PortKind, Primitive, PrimitiveKind, SheetShape
from EMShrimulator.core.em.transforms import apply_ops, effective_ops, unapply_ops
from EMShrimulator.core.services import modeler as modeler_service

DrawItem = Tuple[str, PrimitiveKind, Optional[SheetShape]]  # (label, kind, sheet shape)


def _solid(kind: PrimitiveKind) -> DrawItem:
    return PRIMITIVE_LABELS[kind], kind, None


def _sheet(shape: SheetShape, label: str) -> DrawItem:
    return label, PrimitiveKind.SHEET, shape


SOLID_DRAW = [_solid(k) for k in (PrimitiveKind.BOX, PrimitiveKind.CYLINDER, PrimitiveKind.SPHERE,
                                  PrimitiveKind.POLYGON)]
# Sheet shapes a port is made from.
SHEET_DRAW = [_sheet(SheetShape.RECTANGLE, "Rectangle"), _sheet(SheetShape.CIRCLE, "Circle")]
MORE_SHEETS = [_sheet(SheetShape.ELLIPSE, "Ellipse"), _sheet(SheetShape.RING, "Ring")]
MORE_SOLIDS = [_solid(k) for k in (PrimitiveKind.CYLINDRICAL_SHELL, PrimitiveKind.SPHERICAL_SHELL,
                                   PrimitiveKind.ROT_POLY)]
LINE_DRAW = [_solid(PrimitiveKind.WIRE), _solid(PrimitiveKind.CURVE)]
MORE_DRAW = MORE_SOLIDS + MORE_SHEETS + LINE_DRAW
GROUP_DRAW = {"solids": SOLID_DRAW + MORE_SOLIDS, "sheets": SHEET_DRAW + MORE_SHEETS, "lines": LINE_DRAW}
# Objects drawn by clicking in the 3D view; the rest are added, then typed in.
DRAW_HOW = {
    (PrimitiveKind.BOX, None): "click two opposite corners of its base, then its height",
    (PrimitiveKind.CYLINDER, None): "click its centre, a point on its rim, then its height",
    (PrimitiveKind.SPHERE, None): "click its centre, then a point on its surface",
    (PrimitiveKind.SHEET, SheetShape.RECTANGLE): "click two opposite corners",
    (PrimitiveKind.SHEET, SheetShape.CIRCLE): "click its centre, then a point on its rim",
    (PrimitiveKind.SHEET, SheetShape.ELLIPSE): "click two opposite corners of the box round it",
}
LINE_PROMPTS = {
    PortKind.LUMPED: ("Click where the integration line starts: the − terminal.   Esc cancels.",
                      "Click where it ends: the + terminal.   Esc cancels."),
    PortKind.MICROSTRIP: ("Click where the integration line starts, on the ground plane.   Esc cancels.",
                          "Click where it ends, at the strip.   Esc cancels."),
}
BOOLEAN_MENU = [(BooleanOp.UNION, "Unite…"), (BooleanOp.SUBTRACT, "Subtract…"), (BooleanOp.INTERSECT, "Intersect…"),
                (BooleanOp.XOR, "Symmetric difference…"), (BooleanOp.HULL, "Convex hull…")]
# Relative CS and Object CS modes.
CS_MODES = [("offset", "Offset", "Moved, not turned"), ("rotated", "Rotated", "Turned, not moved"),
            ("both", "Both", "Moved and turned")]
RELATIVE_HOW = {"offset": "click its origin; its axes stay parallel to the working system's",
                "rotated": "at the working system's origin; type its rotation",
                "both": "click its origin, then type its rotation"}
SHORT_PORT = {PortKind.LUMPED: "Lumped port", PortKind.MICROSTRIP: "Microstrip port",
              PortKind.RECT_WAVEGUIDE: "Waveguide port", PortKind.COAXIAL: "Coaxial port"}
EDITABLE = ("primitive", "port", "material", "dump", "cs")


class ModelerView(Workspace):
    kind = "modeler"
    layout_version = 3

    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        s = session

        # ------------------------------------------------------- 3D model
        self.viewport = Viewport3D()
        self.viewport.picked.connect(self._picked)
        self.viewport.contextRequested.connect(self._viewport_menu)
        self.viewport.pickFinished.connect(self._pick_finished)
        self.viewport.pickEnded.connect(self._pick_ended)
        self.viewport.drawingPlaneChanged.connect(lambda _name: self._update_status())
        self.properties = PropertyPanel(s)
        self.properties.shownChanged.connect(self._inspector_shown)
        self.properties.plotRequested.connect(self.plot_port)
        self.properties.focusRequested.connect(self.viewport.focus)
        self.properties.lineDrawRequested.connect(self.draw_port_line)
        self.properties.anchorPickRequested.connect(self.repick_anchor)
        self.properties.setVisible(False)
        self.model_split = QSplitter(Qt.Horizontal)
        self.model_split.setChildrenCollapsible(False)
        self.model_split.addWidget(self.viewport)
        self.model_split.addWidget(self.properties)
        self.model_split.setStretchFactor(0, 1)
        self.model_split.setStretchFactor(1, 0)
        self._splitters["model_split"] = self.model_split

        # ---------------------------------------------------------- pages
        self.parameters = ParameterTable(s)
        self.setup = SolveSettings(s)
        self.sweeps = SweepEditor(s)
        self.sweeps.runRequested.connect(self.run_sweep)
        sweeps_scroll = QScrollArea()
        sweeps_scroll.setWidgetResizable(True)
        sweeps_scroll.setWidget(self.sweeps)
        self.results = ResultsView(
            lambda: modeler_service.results(s.name, s.project),
            farfield_loader=lambda: modeler_service.farfield(s.name),
            sweep_loader=lambda slug: modeler_service.sweep_results(s.name, s.project, slug),
        )
        reports = self.report_pages(
            self.results, "The newest run. Make a report of your own with New report - S, Y, Z, TDR, far and near "
                          "fields, in any display; it is listed under Results in the project tree.",
            near_loader=lambda dump, port: modeler_service.nearfield(s.name, dump, port),
            scene_provider=lambda: self.viewport.scene)

        pages: Dict[str, QWidget] = {
            "model": self.model_split,
            "parameters": Page("Design parameters", self.parameters,
                               "Named values any dimension can use. Type a name - Wp, h, f0 - into a field and the "
                               "design follows it; a sweep varies them."),
            "setup": Page("Solution setup", self.setup,
                          "What is solved, over which frequencies, and how accurately."),
            "sweeps": Page("Parametric sweeps", sweeps_scroll,
                           "Solve the design again for each combination of variable values."),
            **reports,
            **self.output_pages(),
        }

        # ----------------------------------------------------- project tree
        self.tree = ProjectTree(s)
        self.tree.menuRequested.connect(self._tree_menu)
        self.tree.tree.itemDoubleClicked.connect(self._tree_double_clicked)
        self.build_shell(self.tree, pages, "model")

        # -------------------------------------------- actions, menus, toolbar
        self._build_actions()
        self._menus = [self._modeling_menu(), self._simulation_menu(), self._results_menu()]
        self._build_toolbar()

        # ----------------------------------------------------- status bar
        self.validation_label = QLabel("not validated")
        self.cells_label = QLabel("")
        self.params_label = QLabel("")
        self.preview_label = QLabel("")
        self.status_widgets[1:1] = [self.validation_label, self.cells_label, self.params_label, self.preview_label]

        # --------------------------------------------------------- preview
        self.previews = PreviewScheduler(self)
        self.previews.ready.connect(self._preview_ready)
        self.previews.busyChanged.connect(lambda busy: self.preview_label.setText("updating preview…" if busy else ""))
        s.projectChanged.connect(self._project_changed)
        s.selectionChanged.connect(self._selection_changed)
        self.runningChanged.connect(self.sweeps.set_running)

        self.restore_layout()
        self._update_status()
        self.tree.set_current(("project",))
        QTimer.singleShot(0, lambda: self.previews.request_now(s.resolution))
        QTimer.singleShot(0, self.results.reload)

    # ----------------------------------------------------------- layout

    def restore_layout(self) -> None:
        super().restore_layout()
        # The inspector follows the selection, not the saved state.
        self.properties.setVisible(self.properties.shown)

    def menus(self) -> List[QMenu]:
        return list(self._menus)

    # ------------------------------------------------------- navigation

    def show_key(self, key) -> None:
        """Show what a project-tree item stands for."""
        kind = key[0]
        s = self.session
        if kind in EDITABLE:
            target = key[1] if kind != "cs" else (key[1] or "")
            s.select(kind, target)
            self.show_page("model")
        elif kind in ("project", "model"):
            s.select(None)  # the whole design: nothing in particular
            self.show_page("model")
        elif kind in ("group", "matgroup", "materials", "ports", "dumps"):
            self.show_page("model")
        elif kind in ("parameters", "parameter"):
            self.show_page("parameters")
            if kind == "parameter":
                self.parameters.select_name(key[1])
        elif kind == "setup":
            self.show_page("setup")
            self.setup.scroll_to(key[1] if len(key) > 1 else "")
        elif kind in ("sweeps", "sweep"):
            self.show_page("sweeps")
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
        self.duplicate_action = QAction("Duplicate", self)
        self.duplicate_action.setShortcut(QKeySequence("Ctrl+D"))
        self.duplicate_action.triggered.connect(self.duplicate_selected)
        self.delete_action = QAction("Delete", self)
        self.delete_action.setShortcut(QKeySequence.Delete)
        self.delete_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.delete_action.triggered.connect(self.delete_selected)
        self.mesh_action = QAction("Show mesh", self)
        self.mesh_action.setShortcut(QKeySequence("M"))
        self.mesh_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.mesh_action.triggered.connect(self.viewport.toggle_mesh)
        self.viewport.addAction(self.mesh_action)
        self.viewport.addAction(self.delete_action)
        self.tree.tree.addAction(self.delete_action)
        self.explode_action = QAction("Explode boolean", self)
        self.explode_action.setToolTip("Split the selected boolean back into the objects it was made from")
        self.explode_action.triggered.connect(self.explode_selected)
        self.validate_action = QAction("Validation", self)
        self.validate_action.triggered.connect(lambda: self.show_key(("validation",)))
        self.measure_action = QAction("Measure", self, checkable=True)
        self.measure_action.setToolTip("Measure between two points in the 3D view. Points snap as they do when "
                                       "drawing; Esc ends measuring")
        self.measure_action.toggled.connect(self._measure_toggled)
        self.viewport.add_tool_action(self.measure_action)

    def _draw_menu(self, menu: QMenu, items: List[DrawItem]) -> QMenu:
        for label, kind, shape in items:
            menu.addAction(label, lambda kind=kind, shape=shape: self.draw(kind, shape))
        return menu

    def _port_menu(self, menu: QMenu, sheet: Optional[Primitive] = None) -> QMenu:
        """Port kinds to assign to `sheet`, or to the selected sheet when the menu opens."""
        menu.setToolTipsVisible(True)
        if sheet is not None:
            self._fill_port_menu(menu, sheet)
        else:
            menu.aboutToShow.connect(lambda: self._fill_port_menu(menu, self._selected_sheet()))
        return menu

    def _fill_port_menu(self, menu: QMenu, sheet: Optional[Primitive]) -> None:
        menu.clear()
        header = menu.addAction(f"On {sheet.name}" if sheet is not None else
                                "Select a rectangle or circle first: a port is made from a sheet")
        header.setEnabled(False)
        menu.addSeparator()
        for kind in PortKind:
            action = menu.addAction(SHORT_PORT[kind], lambda kind=kind, sheet=sheet: self.assign_port(kind, sheet))
            action.setToolTip(SHEET_PORT_NEEDS[kind])
            action.setEnabled(sheet is not None)
        if sheet is None:
            menu.addSeparator()
            self._draw_menu(menu, SHEET_DRAW)

    def _modeling_menu(self) -> QMenu:
        menu = QMenu("&Modeling", self)
        self._draw_menu(menu, SOLID_DRAW)
        menu.addSeparator()
        self._draw_menu(menu, SHEET_DRAW)
        self._draw_menu(menu.addMenu("More shapes"), MORE_DRAW)
        menu.addSeparator()
        for op, label in BOOLEAN_MENU:
            action = menu.addAction(label, lambda op=op: self.combine(op))
            action.setToolTip(BOOLEAN_LABELS[op])
        menu.addAction(self.explode_action)
        menu.addSeparator()
        menu.addAction(self.duplicate_action)
        menu.addAction(self.delete_action)
        menu.addAction("Show all objects", self.show_all)
        menu.addSeparator()
        self._cs_menu(menu.addMenu("Coordinate system"))
        self.working_menu = menu.addMenu("Working coordinate system")
        self.working_menu.aboutToShow.connect(lambda: self._fill_working_menu(self.working_menu))
        menu.addAction("New material", self.add_material)
        menu.addSeparator()
        menu.addMenu(self.viewport.plane_menu)
        menu.addMenu(self.viewport.snap_menu)
        menu.addAction(self.measure_action)
        menu.addSeparator()
        menu.addMenu(self.viewport.view_menu)
        menu.addAction("Fit all", self.viewport.fit)
        menu.addAction(self.mesh_action)
        menu.aboutToShow.connect(self._update_modeling_menu)
        return menu

    def _simulation_menu(self) -> QMenu:
        menu = QMenu("&Simulation", self)
        menu.setToolTipsVisible(True)
        self._port_menu(menu.addMenu("Assign excitation"))
        menu.addAction("Add field dump", self.add_dump)
        self.open_region_action = QAction("Open region", self, checkable=True)
        self.open_region_action.setToolTip("Every face absorbing, with the air between the design and the boundary "
                                           "sized for you: λ/4 at the lowest operating frequency")
        self.open_region_action.triggered.connect(lambda on: set_open_region(self.session, bool(on)))
        menu.addAction(self.open_region_action)
        menu.aboutToShow.connect(
            lambda: self.open_region_action.setChecked(self.session.project.boundary.open_region))
        menu.addSeparator()
        menu.addAction("Design parameters", lambda: self.show_key(("parameters",)))
        menu.addAction("Solution setup", lambda: self.show_key(("setup",)))
        menu.addAction("Parametric sweeps", lambda: self.show_key(("sweeps",)))
        menu.addSeparator()
        menu.addAction(self.validate_action)
        menu.addAction(self.run_action)
        menu.addAction(self.stop_action)
        menu.addSeparator()
        menu.addAction("Solver log", lambda: self.show_key(("log",)))
        menu.addAction("Runs", lambda: self.show_key(("runs",)))
        return menu

    def _results_menu(self) -> QMenu:
        menu = QMenu("&Results", self)
        self.fill_report_menu(menu.addMenu("New report"))
        menu.addSeparator()
        menu.addAction("Results overview", lambda: self.show_key(("results",)))
        menu.addAction("Reload results", self.results.reload)
        menu.addAction("Save Touchstone…", self.results.export_touchstone)
        return menu

    def _update_modeling_menu(self) -> None:
        selection = self.session.selection
        prim = None
        if selection is not None and selection.kind == "primitive":
            prim = next((p for p in self.session.project.primitives if p.id == selection.key), None)
        self.explode_action.setEnabled(prim is not None and prim.kind == PrimitiveKind.BOOLEAN)
        self.duplicate_action.setEnabled(prim is not None)

    def _fill_working_menu(self, menu: QMenu) -> None:
        menu.clear()
        group = QActionGroup(menu)
        current = self.session.project.working_cs
        for name, label in [(None, "Global")] + [(cs.name, cs.name) for cs in self.session.project.coordinate_systems]:
            action = menu.addAction(label, lambda name=name: self.set_working_cs(name))
            action.setCheckable(True)
            action.setChecked(name == current)
            group.addAction(action)

    def _cs_menu(self, menu: QMenu) -> QMenu:
        """Relative CS and Object CS submenus; a new system becomes the working one."""
        menu.setToolTipsVisible(True)
        self._relative_cs_menu(menu.addMenu("Relative CS"))
        self._object_cs_menu(menu.addMenu("Object CS"))
        return menu

    def _relative_cs_menu(self, menu: QMenu) -> QMenu:
        menu.setToolTipsVisible(True)
        for mode, label, _tip in CS_MODES:
            action = menu.addAction(label + ("…" if mode != "rotated" else ""),
                                    lambda mode=mode: self.new_relative_cs(mode))
            action.setToolTip(f"Based on the working coordinate system: {RELATIVE_HOW[mode]}")
        return menu

    def _object_cs_menu(self, menu: QMenu, prim: Optional[Primitive] = None) -> QMenu:
        """On `prim`, or on the object selected when the menu opens."""
        menu.setToolTipsVisible(True)

        def fill() -> None:
            target = prim or self._selected_primitive()
            menu.clear()
            header = menu.addAction(f"On {target.name}" if target is not None else
                                    "Select an object first: the system sits on one of its points")
            header.setEnabled(False)
            menu.addSeparator()
            for mode, label, tip in CS_MODES:
                action = menu.addAction(f"{label}…", lambda mode=mode: self.new_object_cs(mode, target))
                action.setToolTip(f"{tip} from a point you click on the object - its centre, a face centre, an "
                                  "edge centre or a corner - and following that point as the object changes")
                action.setEnabled(target is not None)

        fill()  # filled now too, for shortcuts
        if prim is None:
            menu.aboutToShow.connect(fill)
        return menu

    def _build_toolbar(self) -> None:
        bar = QToolBar("Modeler")
        bar.setObjectName("modeler-toolbar")
        bar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, bar)
        for label, kind, shape in SOLID_DRAW:
            action = bar.addAction(label, lambda kind=kind: self.draw(kind))
            how = DRAW_HOW.get((kind, shape))
            action.setToolTip(f"Draw a {label.lower()}: {how}" if how else
                              f"Add a {label.lower()} in the working coordinate system, then type its size")
        bar.addSeparator()
        for label, kind, shape in SHEET_DRAW:
            action = bar.addAction(label, lambda kind=kind, shape=shape: self.draw(kind, shape))
            action.setToolTip(f"Draw a {label.lower()} sheet on the drawing plane: {DRAW_HOW[(kind, shape)]}. "
                              "Right-click it to assign a port, or give it a material to make it metal")
        more = QToolButton()
        more.setText("More")
        more.setPopupMode(QToolButton.InstantPopup)
        more.setMenu(self._draw_menu(QMenu(more), MORE_DRAW))
        bar.addWidget(more)
        bar.addSeparator()
        ports = QToolButton()
        ports.setText("Port")
        ports.setToolTip("Assign a port to the selected sheet")
        ports.setPopupMode(QToolButton.InstantPopup)
        ports.setMenu(self._port_menu(QMenu(ports)))
        bar.addWidget(ports)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        bar.addAction(self.save_action)
        bar.addAction(self.run_action)
        bar.widgetForAction(self.run_action).setProperty("role", "run")
        bar.addAction(self.stop_action)

    # ----------------------------------------------------------- edits

    def add_primitive(self, kind: PrimitiveKind, sheet_shape: Optional[SheetShape] = None) -> None:
        prim_id = guarded(lambda: self.session.push(cmd.AddPrimitive(self.session, kind, sheet_shape)))
        if prim_id:
            self.show_key(("primitive", prim_id))

    def draw(self, kind: PrimitiveKind, sheet_shape: Optional[SheetShape] = None) -> None:
        """Draw an object by clicking in the 3D view, or add it at a default size if it isn't drawn."""
        tool = drawing.draw_tool(kind, sheet_shape, self.viewport.drawing_plane())
        if tool is None:
            self.add_primitive(kind, sheet_shape)
            return
        self.show_page("model")
        self.viewport.begin_pick(tool, self._units())

    def _units(self) -> drawing.Units:
        units = self.session.project.display_units
        return drawing.Units(unit_math.label("length", units), unit_math.factor("length", units))

    def _measure_toggled(self, on: bool) -> None:
        """Measure between two clicked points in the working coordinate system."""
        if on:
            self.show_page("model")
            self.viewport.begin_pick(drawing.MeasureTool(self.viewport.drawing_plane()), self._units())
        elif isinstance(self.viewport.pick_tool, drawing.MeasureTool):
            self.viewport.cancel_pick()

    def _pick_ended(self, tool) -> None:
        if isinstance(tool, drawing.MeasureTool) and self.measure_action.isChecked():
            self.measure_action.blockSignals(True)
            self.measure_action.setChecked(False)
            self.measure_action.blockSignals(False)

    def _pick_finished(self, tool, points) -> None:
        """A tool's last click: set a port's line, create the drawn object or place a coordinate system."""
        if isinstance(tool, drawing.OriginTool):
            self._finish_relative_cs(tool, points[0])
        elif isinstance(tool, drawing.AnchorTool):
            self._finish_object_cs(tool, points[0])
        elif isinstance(tool, drawing.LineTool):
            # Stored in the sheet's own frame, as typed coordinates are.
            local = unapply_ops(np.array(points, dtype=float), tool.ops)
            guarded(lambda: self.session.push(cmd.SetPortLine(self.session, tool.port_id, tuple(local[0]),
                                                              tuple(local[1]))))
        elif isinstance(tool, drawing.DrawTool):
            # In the working coordinate system, like add_primitive.
            resolved = self.session.resolution.project
            ops = effective_ops(resolved.working_cs, [], {cs.name: cs for cs in resolved.coordinate_systems})
            fields = drawing.drawn_fields(tool, unapply_ops(np.array(points, dtype=float), ops))
            prim_id = guarded(lambda: self.session.push(cmd.DrawPrimitive(self.session, tool.kind, tool.shape,
                                                                          fields)))
            if prim_id:
                self.show_key(("primitive", prim_id))

    def _selected_primitive(self) -> Optional[Primitive]:
        selection = self.session.selection
        if selection is None or selection.kind != "primitive":
            return None
        return next((p for p in self.session.project.primitives if p.id == selection.key), None)

    def _selected_sheet(self) -> Optional[Primitive]:
        prim = self._selected_primitive()
        return prim if prim is not None and prim.kind == PrimitiveKind.SHEET else None

    # ------------------------------------------------- coordinate systems

    def new_relative_cs(self, mode: str) -> None:
        """Create a Relative CS from the working system; it becomes the working one."""
        if mode == "rotated":
            self._push_cs(cmd.AddCoordinateSystem(self.session, self.session.project.working_cs, mode))
            return
        self.show_page("model")
        self.viewport.begin_pick(drawing.OriginTool(self.viewport.drawing_plane(), mode), self._units())

    def _finish_relative_cs(self, tool: drawing.OriginTool, point: np.ndarray) -> None:
        # The offset is along the working system's axes.
        resolved = self.session.resolution.project
        ops = effective_ops(resolved.working_cs, [], {cs.name: cs for cs in resolved.coordinate_systems})
        local = unapply_ops(np.array([point], dtype=float), ops)[0]
        origin = tuple(float(f"{v:.9g}") + 0.0 for v in local)
        self._push_cs(cmd.AddCoordinateSystem(self.session, resolved.working_cs, tool.mode, origin))

    def new_object_cs(self, mode: str, prim: Optional[Primitive] = None) -> None:
        """Create an Object CS on a clicked point of the chosen object."""
        prim = prim or self._selected_primitive()
        if prim is None:
            notify("Select an object first: an object coordinate system sits on one of its points - its centre, "
                   "a face centre, an edge centre or a corner - and follows it as the object changes.", "info")
            return
        self._pick_anchor(drawing.AnchorTool(prim.id, prim.name, mode))

    def repick_anchor(self, cs_name: str, object_id: str) -> None:
        """Move an object system to another point of its object, clicked in the 3D view."""
        prim = next((p for p in self.session.project.primitives if p.id == object_id), None)
        if prim is not None:
            self._pick_anchor(drawing.AnchorTool(prim.id, prim.name, cs_name=cs_name))

    def _pick_anchor(self, tool: drawing.AnchorTool) -> None:
        try:
            targets, _found = frames.anchor_targets(self.session.resolution.project, tool.object_id)
        except Exception as exc:  # noqa: BLE001 - a shape that cannot be built offers no points
            notify(f"{tool.object_name} has no points to put a coordinate system on: {exc}", "warning")
            return
        self.show_page("model")
        self.viewport.begin_pick(tool, self._units(), targets, tool.object_name)

    def _finish_object_cs(self, tool: drawing.AnchorTool, point: np.ndarray) -> None:
        anchor = frames.ANCHOR_OF.get(tool.snapped[0] if tool.snapped else None)
        if anchor is None:
            return  # only snapped points arrive here
        try:
            _targets, found = frames.anchor_targets(self.session.resolution.project, tool.object_id)
        except Exception as exc:  # noqa: BLE001 - the object changed under the click
            notify(str(exc), "warning")
            return
        at = frames.anchor_at(found, point)
        if tool.cs_name:
            guarded(lambda: self.session.push(cmd.PlaceCoordinateSystem(self.session, tool.cs_name, tool.object_id,
                                                                        anchor, at)))
            return
        self._push_cs(cmd.AddObjectCoordinateSystem(self.session, tool.object_id, anchor, at, tool.mode))

    def _push_cs(self, command) -> None:
        name = guarded(lambda: self.session.push(command))
        if name:
            self.show_key(("cs", name))

    def _released(self, before) -> None:
        """Report object systems left in place because their object was removed."""
        was = {cs.name for cs in before.coordinate_systems if cs.kind == "object"}
        now = {cs.name for cs in self.session.project.coordinate_systems if cs.kind == "object"}
        freed = sorted(name for name in was - now if name in
                       {cs.name for cs in self.session.project.coordinate_systems})
        if freed:
            names = ", ".join(freed)
            notify(f"{names} sat on an object that is gone, so {'it stays' if len(freed) == 1 else 'they stay'} "
                   "where it was, based on global.", "info")

    def assign_port(self, kind: PortKind, sheet: Optional[Primitive] = None) -> None:
        """Assign a port to a sheet."""
        sheet = sheet or self._selected_sheet()
        if sheet is None:
            notify("Select a sheet first: draw a Rectangle or Circle where the port goes, then assign the "
                   "port to it.", "info")
            return
        port_id = guarded(lambda: self.session.push(cmd.AssignPort(self.session, sheet.id, kind)))
        if port_id:
            self.show_key(("port", port_id))

    def draw_port_line(self, port_id: str) -> None:
        """Draw a port's integration line in the 3D view: start, then end."""
        resolved = self.session.resolution.project
        port = find_port(resolved, port_id)
        sheet = sheets.sheet_by_name(resolved.primitives, port.sheet) if port is not None else None
        if sheet is None or port.kind not in sheets.LINE_KINDS:
            return
        ops = sheets.sheet_ops(sheet, {cs.name: cs for cs in resolved.coordinate_systems})
        local = np.vstack([sheets.snap_points(sheet), sheets.lift(sheet, [(0.0, 0.0)])])
        local[-1, sheets.normal_index(sheet.normal_axis or "z")] += 1.0  # a point off the plane, for its normal
        placed = apply_ops(local, ops)
        targets, centre, ahead = placed[:-1], placed[0], placed[-1]
        tool = drawing.LineTool(drawing.Plane.normal_to(centre, ahead - centre), LINE_PROMPTS[port.kind])
        tool.port_id, tool.ops = port_id, ops  # where _pick_finished stores the line
        self.show_page("model")
        # The sheet's centre, corners, edge midpoints and quarter points are always offered.
        self.viewport.begin_pick(tool, self._units(), snaps.SnapSet.of(targets, sheets.snap_kinds(sheet)),
                                 sheet.name)

    def add_dump(self) -> None:
        dump_id = guarded(lambda: self.session.push(cmd.AddDump(self.session)))
        if dump_id:
            self.show_key(("dump", dump_id))

    def add_material(self) -> None:
        name = guarded(lambda: self.session.push(cmd.AddMaterial(self.session)))
        if name:
            self.show_key(("material", name))

    def set_working_cs(self, name: Optional[str]) -> None:
        guarded(lambda: self.session.push(cmd.SetWorkingCS(self.session, name)))

    def combine(self, op: BooleanOp) -> None:
        solids = [p for p in self.session.project.primitives if p.kind != PrimitiveKind.SHEET]
        if len(solids) < 2 and op != BooleanOp.HULL:
            notify("A boolean needs at least two solid objects.", "info")
            return
        selection = self.session.selection
        preselect = selection.key if selection is not None and selection.kind == "primitive" else None
        dialog = BooleanDialog(self.session.project, op, preselect, self)
        if dialog.exec():
            before = self.session.project
            new_id = guarded(lambda: self.session.push(cmd.MakeBoolean(self.session, dialog.op(), dialog.operand_ids())))
            self._released(before)
            if new_id:
                self.show_key(("primitive", new_id))

    def explode_selected(self) -> None:
        selection = self.session.selection
        if selection is None or selection.kind != "primitive":
            return
        before = self.session.project
        ids = guarded(lambda: self.session.push(cmd.ExplodeBoolean(self.session, selection.key)))
        self._released(before)
        if ids:
            self.show_key(("primitive", ids[0]))

    def delete_primitive(self, prim_id: str) -> None:
        before = self.session.project
        guarded(lambda: self.session.push(cmd.DeletePrimitive(self.session, prim_id)))
        self._released(before)

    def duplicate_selected(self) -> None:
        selection = self.session.selection
        # Only with the model in view.
        if selection is None or self._page_key != "model":
            return
        if selection.kind == "primitive":
            new_id = guarded(lambda: self.session.push(cmd.DuplicatePrimitive(self.session, selection.key)))
            if new_id:
                self.show_key(("primitive", new_id))
        elif selection.kind == "port":
            new_id = guarded(lambda: self.session.push(cmd.DuplicatePort(self.session, selection.key)))
            if new_id:
                self.show_key(("port", new_id))

    def delete_selected(self) -> None:
        selection = self.session.selection
        # Don't delete an object when Delete is pressed on another page.
        if selection is None or self._page_key != "model":
            return
        s = self.session
        if selection.kind == "primitive":
            self.delete_primitive(selection.key)
            return
        factories = {
            "port": lambda: cmd.DeletePort(s, selection.key),
            "dump": lambda: cmd.DeleteDump(s, selection.key),
            "material": lambda: cmd.DeleteMaterial(s, selection.key),
            "cs": lambda: cmd.DeleteCoordinateSystem(s, selection.key),
        }
        if selection.kind in factories and (selection.kind != "cs" or selection.key):
            guarded(lambda: s.push(factories[selection.kind]()))

    def show_all(self) -> None:
        s = self.session
        hidden = [p.id for p in s.project.primitives if not p.visible]
        for prim_id in hidden:
            guarded(lambda prim_id=prim_id: s.push(cmd.SetPrimitiveField(s, prim_id, "visible", True, "visibility")))

    def rename_primitive(self, prim_id: str) -> None:
        prim = next((p for p in self.session.project.primitives if p.id == prim_id), None)
        if prim is None:
            return
        text, ok = QInputDialog.getText(self, "Rename object", "Name", text=prim.name)
        if ok and text.strip() and text.strip() != prim.name:
            guarded(lambda: self.session.push(cmd.SetPrimitiveField(self.session, prim_id, "name", text.strip())))

    # ----------------------------------------------------------- plots

    def plot_port(self, number: int) -> None:
        self.new_plot("s_db", [f"S{number}{number}"])

    # ---------------------------------------------------- context menus

    def _tree_menu(self, key, pos) -> None:
        menu = QMenu(self)
        kind = key[0]
        s = self.session
        if kind == "primitive":
            prim = next((p for p in s.project.primitives if p.id == key[1]), None)
            if prim is None:
                return
            s.select("primitive", prim.id)
            menu.addAction("Rename…", lambda: self.rename_primitive(prim.id))
            menu.addAction("Duplicate", self.duplicate_selected)
            menu.addAction("Zoom to", lambda: self._focus("primitive", prim.id))
            menu.addAction("Hide" if prim.visible else "Show", lambda: guarded(lambda: s.push(cmd.SetPrimitiveField(
                s, prim.id, "visible", not prim.visible, "visibility"))))
            model = menu.addAction("Model", lambda: guarded(lambda: s.push(cmd.SetPrimitiveField(
                s, prim.id, "model", not prim.model, "model"))))
            model.setCheckable(True)
            model.setChecked(prim.model)
            model.setToolTip("Untick to keep the object in the drawing but out of the mesh and the solve")
            menu.setToolTipsVisible(True)
            if prim.kind == PrimitiveKind.BOOLEAN:
                menu.addAction(self.explode_action)
                self.explode_action.setEnabled(True)
            if prim.kind == PrimitiveKind.SHEET:
                menu.addSeparator()
                ports = [p for p in s.project.ports if p.sheet == prim.name]
                for port in ports:
                    menu.addAction(f"Edit {port.name}", lambda port=port: self.show_key(("port", port.id)))
                if not ports:
                    self._port_menu(menu.addMenu("Assign excitation"), prim)
            menu.addSeparator()
            self._object_cs_menu(menu.addMenu("Object CS"), prim)
            menu.addSeparator()
            menu.addAction("Delete", lambda: self.delete_primitive(prim.id))
        elif kind in ("model", "project"):
            draw = menu.addMenu("Draw")
            self._draw_menu(draw, SOLID_DRAW)
            draw.addSeparator()
            self._draw_menu(draw, SHEET_DRAW + MORE_DRAW)
            menu.addAction("Show all objects", self.show_all)
            menu.addSeparator()
            menu.addMenu(self.viewport.view_menu)
            menu.addAction("Fit all", self.viewport.fit)
        elif kind == "group":
            if key[1] == "cs":
                self._cs_menu(menu)
                self._fill_working_menu(menu.addMenu("Working coordinate system"))
            elif key[1] in GROUP_DRAW:
                self._draw_menu(menu, GROUP_DRAW[key[1]])
        elif kind == "cs":
            name = key[1] or None
            working = menu.addAction("Make working system", lambda: self.set_working_cs(name))
            working.setEnabled(s.project.working_cs != name)
            cs = find_coordinate_system(s.project, name) if name else None
            if cs is not None and cs.kind == "object":
                menu.addAction("Move to another point…", lambda: self.repick_anchor(cs.name, cs.object_id or ""))
            if cs is not None:
                menu.addSeparator()
                menu.addAction("Delete", lambda: guarded(lambda: s.push(cmd.DeleteCoordinateSystem(s, cs.name))))
        elif kind in ("materials", "matgroup", "material"):
            menu.addAction("New material", self.add_material)
            if kind == "material":
                menu.addAction("Delete material", lambda: guarded(lambda: s.push(cmd.DeleteMaterial(s, key[1]))))
        elif kind == "ports":
            self._fill_port_menu(menu, self._selected_sheet())
        elif kind == "port":
            port = next((p for p in s.project.ports if p.id == key[1]), None)
            if port is None:
                return
            s.select("port", port.id)
            menu.addAction("Excite only this", lambda: guarded(lambda: s.push(cmd.ExciteOnly(s, port.id))))
            if port.sheet and port.kind in sheets.LINE_KINDS:
                menu.addAction("Draw integration line", lambda: self.draw_port_line(port.id))
            menu.addAction(f"Plot S{port.number}{port.number}", lambda: self.plot_port(port.number))
            menu.addAction("Zoom to", lambda: self._focus("port", port.id))
            menu.addAction("Duplicate", lambda: self.show_key(("port", guarded(
                lambda: s.push(cmd.DuplicatePort(s, port.id))) or port.id)))
            menu.addSeparator()
            menu.addAction("Delete", lambda: guarded(lambda: s.push(cmd.DeletePort(s, port.id))))
        elif kind == "dumps":
            menu.addAction("Add field dump", self.add_dump)
        elif kind == "dump":
            menu.addAction("Delete", lambda: guarded(lambda: s.push(cmd.DeleteDump(s, key[1]))))
        elif kind in ("parameters", "parameter"):
            menu.addAction("Add variable", lambda: (self.show_key(("parameters",)), self.parameters.add()))
        elif kind in ("sweeps", "sweep"):
            menu.addAction("Add sweep", lambda: (self.show_key(("sweeps",)), self.sweeps._add()))
        elif kind == "results":
            self.report_tree_menu(menu, key)
            menu.addAction("Reload results", self.results.reload)
        elif kind == "plot":
            self.report_tree_menu(menu, key)
        if not menu.isEmpty():
            menu.exec(pos)

    def _viewport_menu(self, kind: str, key: str, pos) -> None:
        """Right-click in the 3D view: the item's tree menu, or the model's on empty space."""
        if kind:
            self.session.select(kind, key)
            self._tree_menu((kind, key), pos)
        else:
            self._tree_menu(("model",), pos)

    def _focus(self, kind: str, key: str) -> None:
        self.show_page("model")
        self.viewport.focus(kind, key)

    def _tree_double_clicked(self, item, _column: int) -> None:
        key = item.data(0, KEY_ROLE)
        if key and key[0] in ("primitive", "port"):
            self._focus(key[0], key[1])

    def select_by_name(self, target: str) -> None:
        """Issues name objects the way their messages do; selection is by id."""
        project = self.session.project
        name = target.split(" · ")[0]
        prim = next((p for p in project.primitives if p.name == name), None)
        if prim is not None:
            self.show_key(("primitive", prim.id))
            return
        port = next((p for p in project.ports if p.name == name), None)
        if port is not None:
            self.show_key(("port", port.id))
            return
        if any(m.name == name for m in project.materials):
            self.show_key(("material", name))
            return
        if any(p.name == name for p in project.parameters):
            self.show_key(("parameter", name))
            return
        if find_coordinate_system(project, name) is not None:
            self.show_key(("cs", name))

    # -------------------------------------------------------------- runs

    def start_job(self):
        return modeler_service.start_run(self.session.name, self.session.project)

    def run_sweep(self, sweep_name: str) -> None:
        if self.running:
            notify("A run is already in progress for this project.", "info")
            return
        if not self.save():
            return
        self.launch(lambda: modeler_service.start_sweep(self.session.name, self.session.project, sweep_name))

    def on_results_ready(self) -> None:
        self.results.reload()
        self.tree.set_badge(("results",), "updated")

    # ----------------------------------------------------------- updates

    def _inspector_shown(self, shown: bool) -> None:
        self.properties.setVisible(shown)
        fit_inspector(self.model_split, shown)

    def _project_changed(self) -> None:
        s = self.session
        # An undo can remove what is being drawn on; a measurement stays.
        if self.viewport.picking and not isinstance(self.viewport.pick_tool, drawing.MeasureTool):
            self.viewport.cancel_pick()
        self.previews.request(s.resolution)
        self._update_status()
        self.follow_plot_edits()

    def _selection_changed(self, selection) -> None:
        if selection is None:
            self.viewport.set_selection(None, None)
        else:
            self.viewport.set_selection(selection.kind, selection.key)
            if selection.kind in EDITABLE and self._page_key != "model":
                self.show_page("model")

    def _picked(self, kind: str, key: str) -> None:
        self.session.select(kind or None, key or None)

    def _preview_ready(self, result) -> None:
        s = self.session
        visible = {p.id: p.visible for p in s.project.primitives}
        # One render for the whole update.
        self.viewport.set_mesh(result.mesh, render=result.scene is None)
        if result.scene is not None:
            self.viewport.set_scene(result.scene, s.resolution.project, visible)
            selection = s.selection
            if selection is not None:
                self.viewport.set_selection(selection.kind, selection.key)
        self.setup.show_mesh_preview(result.mesh, result.mesh_error)
        self.validation.show_report(result.validation)
        report = result.validation or {}
        errors, warnings = report.get("errors", 0), report.get("warnings", 0)
        self.validation_label.setText(f"{errors} error(s), {warnings} warning(s)" if report else "not validated")
        self.validation_badge(report)
        mesh = result.mesh
        self.cells_label.setText(
            f"cells {fmt_count(mesh['cells'])} · Δt {fmt_eng(mesh.get('timestep_s'), 's')}" if mesh
            else (f"mesh: {result.mesh_error[:80]}" if result.mesh_error else ""))

    def _update_status(self) -> None:
        s = self.session
        p = s.project
        errors = s.resolution.errors
        if errors:
            self.params_label.setText(f"{len(errors)} expression error(s): {errors[0].get('label') or errors[0]['target']}")
        else:
            excited = sum(1 for port in p.ports if port.excite)
            plane = drawing.PLANE_LABELS[self.viewport.drawing_plane_name]
            cs = f" · drawing on {plane}" + (f" of {p.working_cs}" if p.working_cs else "")
            self.params_label.setText(
                f"objects {len(p.primitives)} · ports {len(p.ports)} ({excited} excited) · "
                f"{len(p.parameters)} variable(s) · {p.display_units.length} · {p.display_units.frequency}{cs}")

    def shutdown(self) -> None:
        super().shutdown()
        self.previews.shutdown()
        self.results.shutdown()
        self.plot_page.shutdown()
        self.viewport.close_plotter()
