"""The modeler's project tree: objects grouped by kind and material, then setup, parameters and results."""
from __future__ import annotations

import re
from typing import Dict, List

from EMShrimulator.app import commands as cmd
from EMShrimulator.app import theme
from EMShrimulator.app.notify import guarded
from EMShrimulator.app.views.tree_base import KEY_ROLE, Key, ProjectTreeBase
from EMShrimulator.app.widgets.form import tidy
from EMShrimulator.app.widgets.icons import icon
from EMShrimulator.core.em import frames
from EMShrimulator.core.em.edits import PORT_LABELS, PRIMITIVE_LABELS
from EMShrimulator.core.em.model import ModelerProject, Primitive, PrimitiveKind

LINE_KINDS = {PrimitiveKind.WIRE, PrimitiveKind.CURVE}
GROUPS = (("solids", "Solids", "solid"), ("sheets", "Sheets", "sheet"), ("lines", "Lines", "line"),
          ("nonmodel", "Non model", "nonmodel"))
SETUP_SECTIONS = (("solution", "Solution type"), ("frequency", "Frequencies & sweep"),
                  ("accuracy", "Accuracy & mesh"), ("boundaries", "Boundaries"), ("farfield", "Far field"))
PATTERN_PLOTS = {"farfield", "antenna"}
__all__ = ["KEY_ROLE", "ProjectTree", "group_of", "natural_key", "object_groups"]


def group_of(prim: Primitive) -> str:
    if prim.kind == PrimitiveKind.SHEET:
        return "sheets"
    if prim.kind in LINE_KINDS:
        return "lines"
    return "solids"


def natural_key(text: str):
    """`box_2` before `box_10`."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def object_groups(project: ModelerProject) -> Dict[str, Dict[str, List[Primitive]]]:
    """group -> material -> objects sorted by name; non-model objects are grouped apart."""
    out: Dict[str, Dict[str, List[Primitive]]] = {g: {} for g, _label, _icon in GROUPS}
    for prim in project.primitives:
        out[group_of(prim) if prim.model else "nonmodel"].setdefault(prim.material or "", []).append(prim)
    for materials in out.values():
        for objects in materials.values():
            objects.sort(key=lambda p: natural_key(p.name))
    return {g: dict(sorted(m.items(), key=lambda kv: (kv[0] == "", natural_key(kv[0])))) for g, m in out.items()}


class ProjectTree(ProjectTreeBase):
    selection_kinds = ("primitive", "port", "material", "dump", "cs")
    folded_kinds = ("primitive", "materials", "parameters", "sweeps", "simulation")

    def signature(self) -> tuple:
        project = self.session.project
        values = self.session.resolution.values
        return (
            project.name,
            tuple((p.id, p.name, p.kind, p.material, p.visible, p.model, p.coordinate_system,
                   tuple(c.name for c in p.children)) for p in project.primitives),
            tuple((m.name, m.color) for m in project.materials),
            tuple((cs.name, cs.kind, cs.parent, cs.object_id, cs.anchor, cs.anchor_at, cs.mode)
                  for cs in project.coordinate_systems), project.working_cs,
            tuple((p.id, p.name, p.number, p.kind, p.excite, p.sheet) for p in project.ports),
            tuple((d.id, d.name) for d in project.dumps),
            tuple((p.name, p.expression, values.get(p.name), p.unit) for p in project.parameters),
            tuple(s.name for s in project.sweeps),
            tuple((p.id, p.name, p.kind) for p in project.plots),
            project.nf2ff.enabled, project.display_units.length, project.boundary.open_region,
        )

    def selection_key(self, selection) -> Key:
        return selection.kind, selection.key if selection.kind != "cs" else (selection.key or "")

    # --------------------------------------------------------------- build

    def build(self) -> None:
        self._build(self.session.project, self.session.resolution)

    def _build(self, project: ModelerProject, resolution) -> None:
        c = theme.colors()
        root = self._item(self.tree, ("project",), project.name, "project", bold=True,
                          tooltip="The whole design. Choose it to return to the 3D model.")

        # ------------------------------------------------------------ model
        model = self._item(root, ("model",), "Model", "model", tooltip="Every object, by type and material")
        swatches = {m.name: m.color for m in project.materials}
        groups = object_groups(project)
        for group, label, icon_name in GROUPS:
            materials = groups[group]
            count = sum(len(v) for v in materials.values())
            if group == "nonmodel" and not count:
                continue
            node = self._item(model, ("group", group), f"{label} ({count})" if count else label, icon_name,
                              tooltip="Drawn and snapped to, but never meshed or solved. Tick Model on one to "
                                      "put it back in the design." if group == "nonmodel" else "")
            for material, objects in materials.items():
                name = material or "Construction"
                tip = (f"Objects made of {material}" if material else
                       "Sheets without a material: they build nothing and mark where ports go")
                mat_node = self._item(node, ("matgroup", group, material), name, "section",
                                      swatch_color=swatches.get(material) if material else None, tooltip=tip)
                for prim in objects:
                    self._primitive_item(mat_node, prim)
        cs_node = self._item(model, ("group", "cs"), "Coordinate systems", "cs",
                             tooltip="Right-click to create a relative or an object coordinate system")
        working = project.working_cs
        self._item(cs_node, ("cs", ""), "Global" + ("   (working)" if not working else ""), "cs", bold=not working,
                   tooltip="The global system, fixed")
        names = {p.id: p.name for p in project.primitives}
        for cs in sorted(project.coordinate_systems, key=lambda x: natural_key(x.name)):
            if cs.kind == "object":
                where = f"On {names.get(cs.object_id or '', 'a missing object')}, at its {frames.describe_anchor(cs)}"
            else:
                where = f"Based on {cs.parent or 'Global'}"
            self._item(cs_node, ("cs", cs.name), f"{cs.name}{'   (working)' if cs.name == working else ''}", "cs",
                       bold=cs.name == working,
                       tooltip=f"{where} · {frames.MODE_LABELS[cs.mode].lower()}" +
                               ("\nThe working system: new objects are drawn in it" if cs.name == working else ""))

        # -------------------------------------------------- materials, ports
        mats = self._item(root, ("materials",), f"Materials ({len(project.materials)})", "material")
        for m in sorted(project.materials, key=lambda x: natural_key(x.name)):
            self._item(mats, ("material", m.name), m.name, swatch_color=m.color,
                       tooltip=f"{m.name} · {m.kind.value}")

        ports = self._item(root, ("ports",), f"Ports ({len(project.ports)})" if project.ports else "Ports", "port",
                           tooltip="Excitations. Each excited port is one solver run and one column of S.")
        for port in sorted(project.ports, key=lambda p: p.number):
            where = f" on {port.sheet}" if port.sheet else ""
            item = self._item(ports, ("port", port.id), f"{port.number}  {port.name}", "port",
                              tooltip=f"{PORT_LABELS[port.kind]}{where} · "
                                      f"{'excited' if port.excite else 'measured only'}")
            item.setIcon(0, icon("port", c["text_faint"], c["warn"] if not port.excite else c["danger"]))

        dumps = self._item(root, ("dumps",), f"Field dumps ({len(project.dumps)})" if project.dumps else "Field dumps",
                           "dump", tooltip="Boxes the solver records E, H or current fields in")
        for dump in project.dumps:
            self._item(dumps, ("dump", dump.id), dump.name, "dump")

        # ------------------------------------------------------- parameters
        params = self._item(root, ("parameters",),
                            f"Design parameters ({len(project.parameters)})" if project.parameters else
                            "Design parameters", "variable",
                            tooltip="Named variables any dimension can use")
        for p in project.parameters:
            value = resolution.values.get(p.name)
            shown = tidy(float(f"{value:.6g}")) if value is not None else "?"
            unit = f" {p.unit}" if p.unit else ""
            self._item(params, ("parameter", p.name), f"{p.name} = {shown}{unit}", "variable",
                       tooltip=f"{p.name} = {p.expression}" + (f"\n{p.description}" if p.description else ""),
                       color=c["danger"] if value is None else None)

        # ----------------------------------------------------- setup, sweeps
        setup = self._item(root, ("setup",), "Solution setup", "setup",
                           tooltip="Solution type, frequencies, accuracy, mesh and boundaries")
        for section, label in SETUP_SECTIONS:
            if section == "farfield" and not project.nf2ff.enabled:
                continue
            if section == "boundaries" and project.boundary.open_region:
                label = "Boundaries: open region"
            self._item(setup, ("setup", section), label, "section")
        sweeps = self._item(root, ("sweeps",), f"Parametric sweeps ({len(project.sweeps)})" if project.sweeps else
                            "Parametric sweeps", "sweep", tooltip="Run the design over ranges of its variables")
        for i, sweep in enumerate(project.sweeps):
            self._item(sweeps, ("sweep", i), sweep.name, "sweep")

        # ----------------------------------------------------------- results
        results = self._item(root, ("results",), "Results", "results",
                             tooltip="The newest run. Right-click to add a report.")
        for plot in project.plots:
            self._item(results, ("plot", plot.id), plot.name, "pattern" if plot.kind in PATTERN_PLOTS else "plot")

        sim = self._item(root, ("simulation",), "Simulation", "log", tooltip="Checks, the solver log and past runs")
        self._item(sim, ("validation",), "Validation", "check")
        self._item(sim, ("log",), "Solver log", "log")
        self._item(sim, ("runs",), "Runs", "runs")

    def _primitive_item(self, parent, prim: Primitive) -> None:
        c = theme.colors()
        kind = PRIMITIVE_LABELS[prim.kind]
        cs = f" · in {prim.coordinate_system}" if prim.coordinate_system else ""
        non_model = "" if prim.model else " · non-model"
        item = self._item(parent, ("primitive", prim.id), prim.name,
                          "solid" if group_of(prim) == "solids" else group_of(prim)[:-1],
                          tooltip=f"{prim.name} · {kind} · {prim.material or 'construction'}{cs}{non_model}",
                          color=None if prim.visible else c["text_faint"], checked=prim.visible)
        for i, child in enumerate(prim.children):
            self._item(item, ("operand", prim.id, i), f"{child.name}  ({PRIMITIVE_LABELS[child.kind]})", "section",
                       color=c["text_faint"], selectable=False,
                       tooltip="An operand of this boolean. Explode the boolean to edit it.")

    # -------------------------------------------------------------- events

    def item_checked(self, key: Key, visible: bool) -> None:
        if key[0] != "primitive":
            return
        prim = next((p for p in self.session.project.primitives if p.id == key[1]), None)
        if prim is not None and prim.visible != visible:
            s = self.session
            guarded(lambda: s.push(cmd.SetPrimitiveField(s, key[1], "visible", visible, "visibility")))
