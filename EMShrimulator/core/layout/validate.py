"""Design-rule checks for a layout project. Only "error" issues block a run."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from EMShrimulator.core.layout.board import copper_layers, edge_cuts_file
from EMShrimulator.core.layout.model import LayoutPort, LayoutPortKind, LayoutProject
from EMShrimulator.core.layout.padstacks import problems as padstack_problems
from EMShrimulator.core.layout.ports import resolve_references


def _issue(severity: str, code: str, message: str, hint: str = "", target: str | None = None) -> Dict:
    return {"severity": severity, "code": code, "message": message, "hint": hint, "target": target}


def _circuit_issues(project: LayoutProject, port: LayoutPort) -> List[Dict]:
    if port.neg_x_mm is None or port.neg_y_mm is None:
        return [_issue("error", "circuit-no-negative",
                       f"Circuit port '{port.name}' has no − terminal.",
                       "Draw it again with the Circuit port tool, or set − X/Y in the port editor.", port.name)]
    out: List[Dict] = []
    dx, dy = abs(port.neg_x_mm - port.x_mm) * 1000, abs(port.neg_y_mm - port.y_mm) * 1000  # µm
    cell = max(project.pixel_size_um * 2, 1.0)
    if port.layer == port.plane:
        if max(dx, dy) < cell:
            out.append(_issue("error", "circuit-terminals-coincide",
                              f"Circuit port '{port.name}' has both terminals on the same layer at the same point.",
                              "Move the − terminal onto the other conductor, or put it on another layer.", port.name))
        elif min(dx, dy) > port.width_um / 2:
            out.append(_issue("warning", "circuit-off-axis",
                              f"Circuit port '{port.name}' runs diagonally ({dx:.0f} × {dy:.0f} µm).",
                              "The FDTD grid only has x/y/z edges: the port is laid along the longer axis and a "
                              "copper lead closes the short one, which adds a little inductance. Snap the terminals "
                              "to facing edge centres to keep it straight.", port.name))
    elif max(dx, dy) > port.width_um / 2:
        out.append(_issue("info", "circuit-vertical-lead",
                          f"Circuit port '{port.name}' spans layers and its terminals are {max(dx, dy):.0f} µm apart "
                          "in plane.",
                          "The port drops vertically at the + terminal and a copper lead on the − layer runs to the "
                          "− terminal.", port.name))
    return out


def _pair_issues(project: LayoutProject) -> List[Dict]:
    """Check differential pairs: two distinct existing ports, no port in two pairs, positive references."""
    out: List[Dict] = []
    ports = {p.number: p for p in project.ports}
    owner: Dict[int, str] = {}
    for pair in project.diff_pairs:
        name = pair.name
        sides = (("+", pair.positive), ("−", pair.negative))
        for sign, number in sides:
            if number is None:
                out.append(_issue("error", "pair-port-missing", f"Differential pair {name} has no {sign} port.",
                                  "Choose it in the pair's properties, or delete the pair.", name))
            elif number not in ports:
                out.append(_issue("error", "pair-port-gone", f"Differential pair {name}'s {sign} port "
                                  f"(P{number + 1}) does not exist.", "Choose another port for it.", name))
            elif number in owner:
                out.append(_issue("error", "pair-port-shared", f"P{number + 1} is in both {owner[number]} and "
                                  f"{name}.", "A port can be in one differential pair.", name))
            else:
                owner[number] = name
        if pair.positive is not None and pair.positive == pair.negative:
            out.append(_issue("error", "pair-same-port", f"Differential pair {name} uses P{pair.positive + 1} "
                              "as both its + and − port.", "", name))
        if pair.z_diff <= 0 or pair.z_common <= 0:
            out.append(_issue("error", "pair-impedance", f"Differential pair {name} has a non-positive mode "
                              "reference impedance.", "", name))
        both = [ports[n] for _sign, n in sides if n in ports]
        if len(both) == 2 and not all(p.excite for p in both):
            idle = [f"P{p.number + 1}" for p in both if not p.excite]
            out.append(_issue("warning", "pair-not-driven",
                              f"Differential pair {name}: {' and '.join(idle)} "
                              f"{'are' if len(idle) > 1 else 'is'} not excited, so {name}'s own reflection "
                              "(Sdd, Scc) cannot be computed.",
                              "Excite both ports of a pair you want to see the input of. Transmission from an "
                              "excited pair into this one still comes out.", name))
    return out


def validate(project: LayoutProject, project_dir: Path) -> Dict:
    issues: List[Dict] = []
    fab_dir = project_dir / "fab"

    stackup_dicts = [layer.model_dump() for layer in project.stackup]
    layers = copper_layers(fab_dir, stackup_dicts) if fab_dir.exists() else []
    edge = edge_cuts_file(fab_dir) if fab_dir.exists() else None

    if not layers:
        issues.append(_issue(
            "error", "no-copper", "No copper gerbers found in this project.",
            "Upload the KiCad Gerber X2 export — at minimum …-F_Cu.gbr and …-Edge_Cuts.gbr — "
            "or import an ODB++ archive.",
        ))
    if edge is None:
        issues.append(_issue(
            "error", "no-edge-cuts", "No board outline (…-Edge_Cuts.gbr) found.",
            "Every other coordinate in the project is measured from the Edge_Cuts bounding box, "
            "so the run cannot start without it.",
        ))
    if not project.drill_file and not project.vias:
        issues.append(_issue(
            "warning", "no-drill", "No plated-through drill file (…-PTH.drl) uploaded, and no vias placed.",
            "Vias will not be modelled. If the board has none, ignore this.",
        ))
    for problem in padstack_problems(project):
        issues.append(_issue("error", "padstack", problem,
                             "Fix it under Padstacks in the project tree, or in the via's inspector."))

    stack_copper = [layer for layer in project.stackup if layer.type == "copper"]
    if not project.stackup:
        issues.append(_issue(
            "error", "no-stackup", "The stackup is empty.",
            "Pick a preset in the Stackup panel or import your fab's KiCad stackup.json — "
            "layer thickness and εr set the impedance of every trace on the board.",
        ))
    elif layers:
        # Every stackup copper layer needs a gerber; extra gerbers are ignored.
        gerber_names = {layer.name.lower() for layer in layers}
        for layer in stack_copper:
            expected = layer.name.replace(".", "_").lower()
            if expected not in gerber_names:
                issues.append(_issue(
                    "error", "stackup-layer-no-gerber",
                    f"Stackup copper layer '{layer.name}' has no matching gerber "
                    f"(expected one named …-{layer.name.replace('.', '_')}.gbr).",
                    "gerber2ems finds each metal layer's copper by name. Rename the stackup layer to match "
                    "the gerber, or upload the missing layer.",
                    layer.name,
                ))
        unused = [layer.name for layer in layers
                  if layer.name.lower() not in {s.name.replace(".", "_").lower() for s in stack_copper}]
        if unused:
            issues.append(_issue(
                "info", "gerber-not-in-stackup",
                f"{len(unused)} copper gerber(s) are not in the stackup and will not be simulated: {', '.join(unused)}.",
                "Add them to the stackup if they should be modelled — otherwise this is fine and intentional.",
            ))

    for layer in project.stackup:
        if layer.type in ("core", "prepreg"):
            if not layer.thickness_mm:
                issues.append(_issue("error", "layer-no-thickness",
                                     f"Dielectric layer '{layer.name}' has no thickness.", "", layer.name))
            if not layer.epsilon:
                issues.append(_issue("error", "layer-no-epsilon",
                                     f"Dielectric layer '{layer.name}' has no εr.",
                                     "FR4 is about 4.3; check your fab's datasheet for the real number.", layer.name))
        if layer.type == "copper" and not layer.thickness_mm:
            issues.append(_issue("warning", "copper-no-thickness",
                                 f"Copper layer '{layer.name}' has no thickness.",
                                 "1 oz copper is 0.035 mm.", layer.name))

    if not project.ports:
        issues.append(_issue("error", "no-ports", "The design has no ports.",
                             "Click on the 2D canvas to place one, or add one numerically."))
    if project.ports and not any(p.excite for p in project.ports):
        issues.append(_issue("error", "no-excited-port", "No port is set to Excite.",
                             "Tick Excite on at least one port — each becomes its own FDTD run."))

    numbers = [p.number for p in project.ports]
    for n in sorted({n for n in numbers if numbers.count(n) > 1}):
        issues.append(_issue("error", "duplicate-port-number", f"Port index {n} is used more than once.",
                             "gerber2ems indexes ports positionally; duplicates collide."))
    if numbers and sorted(numbers) != list(range(len(numbers))):
        issues.append(_issue("error", "port-numbering",
                             f"Port indices must be a contiguous 0..{len(numbers) - 1} run; got {sorted(numbers)}.",
                             "Use Renumber ports to fix this automatically."))

    n_copper = len(stack_copper) or len(layers)
    choices = {}
    if edge is not None and layers and any(p.reference == "auto" for p in project.ports):
        try:
            choices = resolve_references(project, project_dir)
        except Exception as exc:  # noqa: BLE001 - a raster problem should not hide every other check
            issues.append(_issue("warning", "reference-unresolved",
                                 f"Automatic port references could not be resolved: {exc}", ""))
    for port in project.ports:
        circuit = port.kind == LayoutPortKind.CIRCUIT
        choice = choices.get(port.number)
        plane = choice.plane if choice is not None and choice.plane is not None else port.plane
        if choice is not None:
            if choice.found:
                issues.append(_issue("info", "port-reference-auto",
                                     f"Port '{port.name}' references {choice.note}.", "", port.name))
            else:
                issues.append(_issue("warning", "port-reference-no-plane",
                                     f"Port '{port.name}': {choice.note}.",
                                     "Pick the reference layer by hand if the return path is elsewhere, or use a "
                                     "circuit port to a coplanar ground.", port.name))
        if n_copper and port.layer >= n_copper:
            issues.append(_issue("error", "port-layer-range",
                                 f"Port '{port.name}' targets copper layer {port.layer}, but the board has "
                                 f"{n_copper} (0..{n_copper - 1}).", "", port.name))
        if n_copper and plane >= n_copper:
            issues.append(_issue("error", "port-plane-range",
                                 f"Port '{port.name}' references reference plane {plane}, out of range.",
                                 "", port.name))
        if circuit:
            issues.extend(_circuit_issues(project, port))
        elif port.layer == plane:
            issues.append(_issue("error", "port-layer-equals-plane",
                                 f"Port '{port.name}' has its signal layer and reference plane on the same copper layer.",
                                 "A port needs two different conductors to develop a voltage between. For a "
                                 "coplanar ground, use a circuit port.", port.name))
        if port.kind == LayoutPortKind.MICROSTRIP and round(port.rotation_deg) % 90 != 0:
            issues.append(_issue("error", "port-rotation",
                                 f"Microstrip port '{port.name}' is rotated {port.rotation_deg}°.",
                                 "gerber2ems's microstrip launch only supports multiples of 90°. "
                                 "Rotate to the nearest axis, or switch this port to the lumped type, "
                                 "which has no direction constraint.", port.name))
        if port.impedance <= 0:
            issues.append(_issue("error", "port-impedance",
                                 f"Port '{port.name}' has a non-positive reference impedance.", "", port.name))
        if port.width_um <= 0 or port.length_um <= 0:
            issues.append(_issue("error", "port-size", f"Port '{port.name}' has a zero width or length.", "", port.name))
        if port.width_um < project.pixel_size_um * 2:
            issues.append(_issue("warning", "port-below-pixel",
                                 f"Port '{port.name}' is {port.width_um:g} µm wide but the raster pixel size "
                                 f"is {project.pixel_size_um:g} µm.",
                                 "The copper mesh cannot resolve a feature narrower than a couple of pixels. "
                                 "Reduce Pixel size (slower, more memory) or widen the port.", port.name))

    port_count = len(project.ports)
    for trace in project.traces:
        for value, label in ((trace.start, "start"), (trace.stop, "stop")):
            if value >= port_count:
                issues.append(_issue("error", "trace-port-range",
                                     f"Trace '{trace.name or ''}' references port {value} as {label}, "
                                     f"but only {port_count} ports exist.", "", trace.name))
    issues.extend(_pair_issues(project))

    if project.frequency.stop <= project.frequency.start:
        issues.append(_issue("error", "bad-sweep", "Stop frequency must be greater than start frequency.", ""))
    conv = project.convergence
    if conv.delta_s < 0:
        issues.append(_issue("error", "bad-delta-s", "ΔS must be positive, or 0 to turn the check off.", ""))
    elif conv.delta_s > 0.5:
        issues.append(_issue("warning", "loose-delta-s",
                             f"ΔS {conv.delta_s:g} is very loose; S-parameters are bounded by 1, so the run may stop "
                             "before the pulse has crossed the board.", "0.1–0.2 is fast, 0.01–0.005 is accurate."))
    if conv.end_criteria_db >= 0:
        issues.append(_issue("error", "bad-end-criteria", "The energy decay criterion must be below 0 dB.", ""))
    if project.pixel_size_um <= 0:
        issues.append(_issue("error", "bad-pixel-size", "Pixel size must be positive.", ""))
    elif project.pixel_size_um > 25:
        issues.append(_issue("warning", "coarse-raster",
                             f"Pixel size {project.pixel_size_um:g} µm is coarse.",
                             "Fine traces and gaps will be lost when the copper is rasterized. "
                             "2.5–5 µm is the range gerber2ems's own examples use."))

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: order[i["severity"]])
    return {
        "issues": issues,
        "errors": sum(1 for i in issues if i["severity"] == "error"),
        "warnings": sum(1 for i in issues if i["severity"] == "warning"),
        "infos": sum(1 for i in issues if i["severity"] == "info"),
        "can_run": not any(i["severity"] == "error" for i in issues),
    }
