"""The 3D Layout's project tree: board, layers, nets, padstacks, files, stackup, ports, setup and results."""
from __future__ import annotations

from collections import Counter
from typing import Dict

from PySide6.QtCore import QTimer, Signal

from EMShrimulator.app import theme
from EMShrimulator.app.views.layout import palette
from EMShrimulator.app.views.tree_base import KEY_ROLE, Key, ProjectTreeBase
from EMShrimulator.app.widgets.icons import icon
from EMShrimulator.core.layout.model import LayoutPortKind
from EMShrimulator.core.layout.ports import metal_names

SETUP_SECTIONS = (("frequency", "Frequencies"), ("convergence", "Convergence"), ("mesh", "Mesh & materials"),
                  ("export", "Field export"), ("nets", "Nets under test"))
KIND_SHORT = {LayoutPortKind.LUMPED: "Edge gap port", LayoutPortKind.MICROSTRIP: "Microstrip launch",
              LayoutPortKind.CIRCUIT: "Circuit port"}
DIELECTRIC = "#3f7f5a"
OUTLINE = "edge_cuts"


def copper_key(name: str) -> str:
    """Normalise a layer name the way gerber2ems matches them: case-insensitive, '.' as '_'."""
    return name.replace(".", "_").lower()


class BoardTree(ProjectTreeBase):
    layerToggled = Signal(str, bool)  # layer name (or "edge_cuts"), shown
    holesToggled = Signal(bool)

    selection_kinds = ("layout_port", "stackup", "diff_pair")
    folded_kinds = ("files", "stackup", "setup", "simulation", "nets", "padstacks")

    def __init__(self, session, parent=None):
        self.board: Dict = {"ready": False, "layers": []}
        self.drill_count = 0
        self.visible: Dict[str, bool] = {}
        self.show_holes = True
        self.references: Dict = {}
        # Nets are listed only once the group is opened; dense boards have hundreds.
        self._nets_open = False
        super().__init__(session, parent)
        self.tree.itemExpanded.connect(self._expanded)
        self.tree.itemCollapsed.connect(self._collapsed)

    # -------------------------------------------------------------- public

    def set_board(self, board: Dict, drill_count: int, visible: Dict[str, bool], show_holes: bool) -> None:
        self.board, self.drill_count, self.visible, self.show_holes = board, drill_count, visible, show_holes
        self.refresh()

    def set_references(self, references: Dict) -> None:
        """Which plane each "auto" port resolved to, for its label."""
        self.references = references
        self.refresh()

    def open_nets(self) -> None:
        """List the nets, so one can be shown as current."""
        if not self._nets_open:
            self._nets_open = True
            self.refresh()

    def _expanded(self, item) -> None:
        if item.data(0, KEY_ROLE) == ("nets",) and not self._nets_open:
            self._nets_open = True
            QTimer.singleShot(0, self, self.refresh)

    def _collapsed(self, item) -> None:
        if item.data(0, KEY_ROLE) == ("nets",):
            self._nets_open = False  # dropped at the next rebuild

    # ------------------------------------------------------- the base's hooks

    def signature(self) -> tuple:
        p = self.session.project
        layers = tuple((layer["name"], layer["color"], layer.get("sim_index"), self.visible.get(layer["name"], True))
                       for layer in self.board.get("layers", []))
        references = tuple(sorted((n, c.plane, c.found) for n, c in self.references.items()))
        return (
            p.name, bool(self.board.get("ready")), layers, self.visible.get(OUTLINE, True), self.drill_count,
            self.show_holes, tuple(p.gerber_files), p.drill_file, p.source_format,
            tuple((s.name, s.type, s.thickness_mm, s.material, s.epsilon) for s in p.stackup),
            tuple((q.number, q.name, q.kind, q.excite, q.layer, q.plane, q.reference, q.snap, q.impedance)
                  for q in p.ports),
            references, len(p.traces),
            tuple((d.name, d.positive, d.negative, d.z_diff, d.z_common) for d in p.diff_pairs),
            tuple(net["name"] for net in self.board.get("nets", [])), self._nets_open,
            tuple((q.id, q.name, q.kind) for q in p.plots),
            tuple((ps.name, ps.hole_mm, ps.pad_mm, ps.plated, ps.shape) for ps in p.padstacks),
            tuple(sorted(Counter(v.padstack for v in p.vias or []).items())),
        )

    def selection_key(self, selection) -> Key:
        return selection.kind, int(selection.key) if selection.kind in ("stackup", "diff_pair") else selection.key

    def item_checked(self, key: Key, checked: bool) -> None:
        if key[0] == "layer":
            self.layerToggled.emit(key[1], checked)
        elif key[0] == "holes":
            self.holesToggled.emit(checked)

    # --------------------------------------------------------------- build

    def build(self) -> None:
        p = self.session.project
        c = theme.colors()
        root = self._item(self.tree, ("project",), p.name, "project", bold=True,
                          tooltip="The whole board. Choose it to return to the board.")

        # ------------------------------------------------------------ board
        board = self._item(root, ("board",), "Board", "board", tooltip="The board as it is drawn and simulated")
        layers = self.board.get("layers", [])
        if self.board.get("ready"):
            group = self._item(board, ("layers",), f"Copper layers ({len(layers)})", "layers",
                               tooltip="Tick to show a layer on the board. A layer simulates under the index shown; "
                                       "one the stackup does not name is drawn but never simulated.")
            for layer in layers:
                sim = layer.get("sim_index")
                self._item(group, ("layer", layer["name"]),
                           layer["name"] + (f"   ·  {sim}" if sim is not None else "   ·  not simulated"),
                           swatch_color=layer["color"], color=None if sim is not None else c["text_faint"],
                           checked=self.visible.get(layer["name"], True),
                           tooltip=(f"Index {sim} in the simulation: a port's signal layer and reference plane "
                                    "refer to it by this number" if sim is not None else
                                    f"No stackup layer is named {layer['name']}: drawn, never simulated"))
            self._item(group, ("layer", OUTLINE), "Board outline", "outline", checked=self.visible.get(OUTLINE, True))
            self._item(group, ("holes",), f"Holes ({self.drill_count})" if self.drill_count else "Holes", "hole",
                       checked=self.show_holes, tooltip="The vias' and other holes' drills, drawn over the copper")
        else:
            self._item(board, ("layers",), "No board yet", "layers", color=c["text_faint"],
                       tooltip=self.board.get("reason") or "Import the board's Gerber export to see it here")
        if self.board.get("ready"):
            nets = self.board.get("nets", [])
            if nets:
                group = self._item(board, ("nets",), f"Nets ({len(nets)})", "layers",
                                   tooltip="The nets the board's files name. Choose one to highlight it on the "
                                           "board; Layout › Colour by net paints each in its colour.")
                if not self._nets_open:
                    self._item(group, ("nets", "…"), "…", selectable=False)  # keeps the group's expander
                for net in palette.legend(nets) if self._nets_open else []:
                    on = ", ".join(net["layers"])
                    self._item(group, ("net", net["name"]), net["name"], swatch_color=net["color"],
                               tooltip=f"{net['name']} · on {on}" if on else net["name"])
            else:
                self._item(board, ("nets",), "Nets: none in the files", "layers", color=c["text_faint"],
                           tooltip="These Gerbers carry no net names (Gerber X2 %TO.N% attributes). KiCad "
                                   "writes them with \"Include netlist attributes\" ticked; an ODB++ import "
                                   "brings its own.")
        if p.padstacks or p.vias:
            uses = Counter(v.padstack for v in p.vias or [])
            group = self._item(board, ("padstacks",), f"Padstacks ({len(p.padstacks)})", "hole",
                               tooltip="Via definitions: drill, plating, span, and each layer's pad and antipad. "
                                       "Every via is an instance of one.")
            for ps in p.padstacks:
                n = uses.get(ps.name, 0)
                self._item(group, ("padstack", ps.name), f"{ps.name}   ·  {n} via{'s' if n != 1 else ''}", "hole",
                           color=c["text_faint"] if not n else None,
                           tooltip=f"Hole {ps.hole_mm * 1000:g} µm" +
                                   ("" if ps.plated else ", unplated") +
                                   (f", pad {ps.pad_mm * 1000:g} µm" if ps.pad_mm and ps.shape == "circle" else
                                    ", pads as imported" if ps.shape == "imported" else ", no pad"))
        files = list(p.gerber_files) + ([p.drill_file] if p.drill_file else [])
        group = self._item(board, ("files",), f"Files ({len(files)})" if files else "Files", "file",
                           tooltip="The imported board files" + (" (converted from ODB++)"
                                                                 if p.source_format == "odb++" else ""))
        for name in p.gerber_files:
            self._item(group, ("file", name), name, "file")
        if p.drill_file:
            holes = f"   ·  {self.drill_count} holes" if self.drill_count else ""
            self._item(group, ("file", p.drill_file), f"{p.drill_file}{holes}", "hole")

        # ---------------------------------------------------------- stackup
        colors = {copper_key(layer["name"]): layer["color"] for layer in layers}
        stack = self._item(root, ("stackup",), f"Stackup ({len(p.stackup)})" if p.stackup else "Stackup", "stackup",
                           tooltip="The board's cross-section, top to bottom")
        for i, layer in enumerate(p.stackup):
            copper = layer.type == "copper"
            thickness = f"   ·  {layer.thickness_mm:g} mm" if layer.thickness_mm else ""
            gerber = colors.get(copper_key(layer.name)) if copper else None
            if copper and gerber is None:
                tip = f"{layer.name} · copper with no gerber: simulates as bare dielectric"
            else:
                tip = f"{layer.name} · {layer.type}" + (f" · {layer.material}" if layer.material else "") + \
                      (f" · εr {layer.epsilon:g}" if layer.epsilon else "")
            self._item(stack, ("stackup", i), f"{layer.name}{thickness}",
                       swatch_color=gerber or (c["text_faint"] if copper else DIELECTRIC),
                       color=c["danger"] if copper and gerber is None and layers else None, tooltip=tip)

        # ------------------------------------------------------------ ports
        names = metal_names(p)

        def layer_name(i: int) -> str:
            return names[i] if 0 <= i < len(names) else str(i)

        ports = self._item(root, ("ports",), f"Ports ({len(p.ports)})" if p.ports else "Ports", "port",
                           tooltip="Excitations. Each excited port is one solver run and one column of S.")
        roles = {}
        for i, pair in enumerate(p.diff_pairs):
            for number, sign in ((pair.positive, "+"), (pair.negative, "−")):
                if number is not None:
                    roles.setdefault(number, f"{pair.name}{sign}")
        for port in sorted(p.ports, key=lambda q: q.number):
            if port.kind == LayoutPortKind.CIRCUIT:
                layers_text = f"{layer_name(port.layer)}+ / {layer_name(port.plane)}−"
            else:
                choice = self.references.get(port.number) if port.reference == "auto" else None
                plane = choice.plane if choice is not None and choice.plane is not None else port.plane
                layers_text = f"{layer_name(port.layer)} → {layer_name(plane)}" + (" (auto)" if choice else "")
            role = roles.get(port.number)
            short = f"P{port.number + 1}"
            label = short + (f"  {port.name}" if port.name != short else "") + (f"   ·  {role}" if role else "")
            item = self._item(ports, ("layout_port", port.number), label,
                              tooltip=f"{KIND_SHORT[port.kind]} · {layers_text} · {port.impedance:g} Ω · "
                                      f"{'excited' if port.excite else 'measured only'}" +
                                      (f"\nIn differential pair {role[:-1]} as its {role[-1]} port" if role else "") +
                                      (f"\nSnapped to {port.snap}" if port.snap else ""))
            # Excited ports are solid, measure-only ones hollow.
            item.setIcon(0, icon("port_driven" if port.excite else "port_sense", c["text"], c["text"]))
        if p.diff_pairs or len(p.ports) >= 2:
            group = self._item(ports, ("pairs",), f"Differential pairs ({len(p.diff_pairs)})" if p.diff_pairs
                               else "Differential pairs", "pair",
                               tooltip="Two ports taken as one differential port each: + and − ports, and "
                                       "the reference impedance of the differential and common modes. The "
                                       "results add their mixed-mode S (Sdd, Scc, Sdc, Scd).")
            for i, pair in enumerate(p.diff_pairs):
                plus = f"P{pair.positive + 1}" if pair.positive is not None else "?"
                minus = f"P{pair.negative + 1}" if pair.negative is not None else "?"
                missing = pair.positive is None or pair.negative is None
                self._item(group, ("diff_pair", i), f"{pair.name}   {plus}+ / {minus}−",
                           swatch_color=palette.pair_color(i).name(),
                           color=c["danger"] if missing else None,
                           tooltip=f"Differential pair {pair.name}: {plus} is +, {minus} is −. "
                                   f"Differential {pair.z_diff:g} Ω, common {pair.z_common:g} Ω." +
                                   ("\nA port is still to be chosen." if missing else ""))

        # ------------------------------------------------------------ setup
        setup = self._item(root, ("setup",), "Solution setup", "setup",
                           tooltip="Frequencies, convergence, mesh and the nets under test")
        nets = len(p.traces) + len(p.diff_pairs)
        for section, label in SETUP_SECTIONS:
            if section == "nets" and nets:
                label = f"{label} ({nets})"
            self._item(setup, ("setup", section), label, "section")

        results = self._item(root, ("results",), "Results", "results",
                             tooltip="S-parameters of the newest run. Right-click to add a report.")
        for plot in p.plots:
            self._item(results, ("plot", plot.id), plot.name, "plot")
        sim = self._item(root, ("simulation",), "Simulation", "log", tooltip="Checks, the solver log and past runs")
        self._item(sim, ("validation",), "Validation", "check")
        self._item(sim, ("log",), "Solver log", "log")
        self._item(sim, ("runs",), "Runs", "runs")
