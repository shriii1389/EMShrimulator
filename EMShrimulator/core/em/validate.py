"""Design-rule checks for a resolved modeler project. Only "error" issues block a run."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from EMShrimulator.core.em import frames, region, sheets
from EMShrimulator.core.em.mesh import wavelength_resolution
from EMShrimulator.core.em.model import (
    BOUNDARY_FACES,
    MaterialKind,
    ModelerProject,
    Port,
    PortKind,
    PrimitiveKind,
    SheetShape,
)


def _f(value, fallback: float = 0.0) -> float:
    """A resolved dimension as a float; an unresolved expression reads as 0."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if np.isfinite(out) else fallback


def _issue(severity: str, code: str, message: str, hint: str = "", target: str | None = None) -> Dict:
    return {"severity": severity, "code": code, "message": message, "hint": hint, "target": target}


def _check_ports(project: ModelerProject) -> List[Dict]:
    out: List[Dict] = []
    if not project.ports:
        out.append(_issue(
            "error", "no-ports", "The design has no ports.",
            "Draw a rectangle where the port goes, then right-click it and assign a port — without one "
            "there is nothing to excite and no S-parameter to compute.",
        ))
        return out

    excited = [p for p in project.ports if p.excite]
    if not excited:
        out.append(_issue(
            "error", "no-excited-port", "No port is set to Excite.",
            "Tick Excite on at least one port. Each excited port becomes its own FDTD run "
            "and fills one column of the S-matrix.",
        ))

    numbers = [p.number for p in project.ports]
    dupes = {n for n in numbers if numbers.count(n) > 1}
    for n in sorted(dupes):
        out.append(_issue(
            "error", "duplicate-port-number", f"Port number {n} is used more than once.",
            "Port numbers drive the Sij naming and must be unique.",
        ))
    if numbers and sorted(set(numbers)) != list(range(1, len(set(numbers)) + 1)):
        out.append(_issue(
            "warning", "port-numbering-gap", f"Port numbers {sorted(set(numbers))} are not a contiguous 1..N run.",
            "Touchstone files index ports positionally; gaps make the .sNp harder to read in other tools.",
        ))

    for p in project.ports:
        if p.sheet:
            # _check_sheets covers sheet ports.
            if p.impedance <= 0:
                out.append(_issue("error", "port-impedance",
                                  f"Port '{p.name}' has a non-positive reference impedance.", "", p.name))
            continue
        span = np.array(p.p2, dtype=float) - np.array(p.p1, dtype=float)
        if float(np.linalg.norm(span)) < 1e-9:
            out.append(_issue(
                "error", "degenerate-port", f"Port '{p.name}' has zero length (P1 == P2).",
                "A lumped port needs a gap to develop a voltage across.", p.name,
            ))
        if p.kind == PortKind.LUMPED:
            axis_idx = {"x": 0, "y": 1, "z": 2}[p.direction]
            if abs(span[axis_idx]) < 1e-9:
                out.append(_issue(
                    "error", "port-direction-flat",
                    f"Port '{p.name}' has direction '{p.direction}' but no extent along that axis.",
                    "The excitation direction must be the axis the port gap spans.", p.name,
                ))
        if p.kind == PortKind.MICROSTRIP and (not p.prop_dir or not p.exc_dir):
            out.append(_issue(
                "error", "msl-missing-dirs", f"Microstrip port '{p.name}' needs both a propagation and an excitation direction.",
                "Propagation runs along the line; excitation points from the strip down to the reference plane.", p.name,
            ))
        if p.kind == PortKind.MICROSTRIP and p.prop_dir and p.prop_dir == p.exc_dir:
            out.append(_issue(
                "error", "msl-parallel-dirs", f"Microstrip port '{p.name}' has propagation and excitation on the same axis.",
                "They must be perpendicular.", p.name,
            ))
        if p.kind == PortKind.RECT_WAVEGUIDE and not (p.wg_a_mm and p.wg_b_mm):
            out.append(_issue(
                "warning", "wg-missing-dims", f"Waveguide port '{p.name}' has no a/b dimensions set.",
                "They will be inferred from the port box, which is only right if the box is exactly the guide cross-section.", p.name,
            ))
        if p.kind == PortKind.COAXIAL:
            if not (p.r_inner_mm and p.r_outer_mm):
                out.append(_issue(
                    "error", "coax-missing-radii", f"Coaxial port '{p.name}' needs both an inner and an outer radius.", "", p.name,
                ))
            elif p.r_inner_mm >= p.r_outer_mm:
                out.append(_issue(
                    "error", "coax-radii-order", f"Coaxial port '{p.name}' has inner radius >= outer radius.", "", p.name,
                ))
        if p.impedance <= 0:
            out.append(_issue("error", "port-impedance", f"Port '{p.name}' has a non-positive reference impedance.", "", p.name))
    return out


def _check_sheets(project: ModelerProject) -> List[Dict]:
    """Sheet shape checks, and port kinds assigned to shapes openEMS can't excite."""
    out: List[Dict] = []
    by_name = {p.name: p for p in project.primitives if sheets.is_sheet(p)}

    for sheet in by_name.values():
        shape = sheet.sheet_shape or SheetShape.RECTANGLE
        u, v = sheets.extent_uv(sheet)
        if u <= 0 or v <= 0:
            out.append(_issue(
                "error", "sheet-empty", f"Sheet {sheet.name!r} has no area.",
                "Give it a non-zero size - a rectangle needs both in-plane dimensions, "
                "a circle needs a radius.", sheet.name,
            ))
        if shape == SheetShape.RING:
            r_in, r_out = sheets.coax_radii(sheet)
            if not 0 < r_in < r_out:
                out.append(_issue(
                    "error", "sheet-ring-radii",
                    f"Sheet {sheet.name!r} is a ring but its radii are not 0 < inner < outer.",
                    "The inner radius is the centre conductor's outside radius; the outer "
                    "radius is the shield's inside radius.", sheet.name,
                ))
            if sheets.is_built(sheet):
                out.append(_issue(
                    "error", "sheet-ring-material",
                    f"Sheet {sheet.name!r} is a ring with a material assigned.",
                    "A ring has a hole, and a flat CSXCAD polygon cannot have one. Clear its "
                    "material to use it as a coaxial port cross-section, or model the metal "
                    "with a Tube primitive.", sheet.name,
                ))
        if shape == SheetShape.POLYGON and len(sheet.points or []) < 3:
            out.append(_issue(
                "error", "sheet-polygon-points",
                f"Sheet {sheet.name!r} is a polygon with fewer than three points.",
                "Add points until the outline closes.", sheet.name,
            ))

    for port in project.ports:
        if not port.sheet:
            continue
        sheet = by_name.get(port.sheet)
        if sheet is None:
            out.append(_issue(
                "error", "port-sheet-missing",
                f"Port {port.name!r} is assigned to sheet {port.sheet!r}, which does not exist.",
                "Pick an existing sheet, or switch the port back to explicit start/end points.",
                port.name,
            ))
            continue

        shape = sheet.sheet_shape or SheetShape.RECTANGLE

        if port.kind in sheets.LINE_KINDS:
            out.extend(_check_line(project, port, sheet))

        if port.kind == PortKind.RECT_WAVEGUIDE and shape != SheetShape.RECTANGLE:
            out.append(_issue(
                "error", "port-sheet-wg-shape",
                f"Port {port.name!r} is a rectangular waveguide port on a {shape.value} sheet.",
                "The TE/TM modes openEMS evaluates are the closed-form rectangular ones, so "
                "this port needs a rectangle sheet.", port.name,
            ))

        elif port.kind == PortKind.COAXIAL:
            if shape not in (SheetShape.RING, SheetShape.CIRCLE):
                out.append(_issue(
                    "error", "port-sheet-coax-shape",
                    f"Port {port.name!r} is a coaxial port on a {shape.value} sheet.",
                    "A coaxial port needs a ring sheet - outer radius at the shield, inner "
                    "radius at the centre conductor.", port.name,
                ))
            else:
                r_in, r_out = sheets.coax_radii(sheet)
                if r_in <= 0:
                    out.append(_issue(
                        "error", "port-sheet-coax-hollow",
                        f"Port {port.name!r} is a coaxial port on a sheet with no centre "
                        "conductor.",
                        "A hollow round port is a circular waveguide, and its modes need Bessel "
                        "functions that openEMS's expression parser does not have, so the run "
                        "would match against nothing. Make the sheet a ring and set its inner "
                        "radius to the centre conductor.", port.name,
                    ))
                elif r_in >= r_out:
                    out.append(_issue(
                        "error", "port-sheet-coax-radii",
                        f"Port {port.name!r} has a centre conductor at least as large as its "
                        "shield.",
                        "The sheet's inner radius must be smaller than its outer radius.",
                        port.name,
                    ))

    for port in project.ports:
        if not port.sheet or port.kind not in sheets.GUIDED_KINDS:
            continue
        if port.sheet not in by_name or _f(port.sheet_extrude_mm):
            continue
        out.append(_issue(
            "info", "port-sheet-depth",
            f"Port {port.name!r} launches a guided mode, so its sheet is extruded "
            f"{sheets.DEFAULT_EXTRUDE_CELLS} cells along its normal to give the mode a "
            "section of guide.",
            "openEMS cannot excite a guided-mode port of zero length, so it needs depth "
            "rather than a bare surface. Set the port's extrude depth to choose that "
            "length yourself - a microstrip port in particular needs enough of it to "
            "de-embed along.", port.name,
        ))

    return out


def _check_line(project: ModelerProject, port: Port, sheet) -> List[Dict]:
    """The integration line the user drew: present, on the sheet, along a grid axis."""
    ends = sheets.line_ends(port)
    if ends is None:
        what = "from the ground plane up to the strip" if port.kind == PortKind.MICROSTRIP else \
            "from the − terminal to the + one"
        return [_issue(
            "error", "port-no-line", f"Port {port.name!r} has no integration line yet.",
            f"Draw it on sheet {sheet.name!r} in the 3D view, or type its two ends in the port's panel. It "
            f"runs {what}, and sets which way the port is excited.", port.name,
        )]
    out: List[Dict] = []
    if float(np.linalg.norm(ends[1] - ends[0])) < 1e-9:
        return [_issue("error", "port-line-zero", f"Port {port.name!r} has an integration line of zero length.",
                       "Give its two ends different places.", port.name)]
    lifted = [label for label, end in zip(("start", "end"), ends) if not sheets.in_plane(sheet, end)]
    if lifted:
        return [_issue(
            "error", "port-line-off-plane",
            f"The integration line of port {port.name!r} has its {' and '.join(lifted)} out of the plane of "
            f"sheet {sheet.name!r}.",
            "The line runs across the sheet, in its plane: draw it on the sheet, or check the ends typed in "
            "the port's panel against where the sheet is now.", port.name,
        )]
    off = [label for label, end in zip(("start", "end"), ends) if not sheets.on_sheet(sheet, end)]
    if off:
        out.append(_issue(
            "warning", "port-line-off-sheet",
            f"The integration line of port {port.name!r} has its {' and '.join(off)} outside sheet "
            f"{sheet.name!r}.",
            "The port is solved along the line wherever it goes, but it belongs on the port's sheet: check "
            "the ends against where the sheet is now.", port.name,
        ))
    try:
        placement = sheets.port_geometry(project, port, {cs.name: cs for cs in project.coordinate_systems})
    except Exception:  # noqa: BLE001 - a broken coordinate system is reported on its own
        return out
    if placement.line is not None:
        span = placement.line[1] - placement.line[0]
        length = float(np.linalg.norm(span))
        if length > 1e-12 and float(np.max(np.abs(span))) < 0.999 * length:
            out.append(_issue(
                "warning", "port-line-diagonal",
                f"The integration line of port {port.name!r} is not along X, Y or Z.",
                f"openEMS excites a port along one grid axis, so this one is excited along "
                f"{(placement.exc_axis or '?').upper()}, the way the line mostly runs. Draw it straight "
                "along an axis.", port.name,
            ))
    return out


def _check_geometry(project: ModelerProject) -> List[Dict]:
    out: List[Dict] = []
    names = [p.name for p in project.primitives]
    for name in {n for n in names if names.count(n) > 1}:
        out.append(_issue("warning", "duplicate-primitive-name", f"More than one object is named '{name}'.",
                          "Names are how the tree, the validator and the log refer to objects.", name))

    mat_names = {m.name for m in project.materials}
    for prim in project.primitives:
        if not sheets.is_built(prim):  # the solver never sees it
            continue
        if prim.material not in mat_names:
            out.append(_issue("error", "unknown-material",
                              f"'{prim.name}' uses material '{prim.material}', which no longer exists.",
                              "Pick an existing material or re-create that one.", prim.name))
        if prim.kind == PrimitiveKind.BOX and prim.p1 and prim.p2:
            if any(abs(a - b) < 1e-12 for a, b in zip(prim.p1, prim.p2)):
                flat = [ax for ax, (a, b) in zip("xyz", zip(prim.p1, prim.p2)) if abs(a - b) < 1e-12]
                out.append(_issue("info", "flat-box", f"'{prim.name}' is flat in {','.join(flat)}.",
                                  "That is fine and normal for thin metal — openEMS treats a zero-thickness "
                                  "box as an infinitely thin sheet of that material.", prim.name))
        if prim.kind in (PrimitiveKind.CYLINDER, PrimitiveKind.CYLINDRICAL_SHELL) and not prim.radius:
            out.append(_issue("error", "no-radius", f"'{prim.name}' has no radius.", "", prim.name))
        if prim.kind == PrimitiveKind.ROT_POLY and prim.rot_axis and prim.rot_axis == (prim.normal_axis or "z"):
            out.append(_issue("error", "revolve-axis-along-normal",
                              f"Revolve '{prim.name}' sweeps about its own normal axis ({prim.rot_axis}).",
                              "openEMS revolves the outline about an axis in the outline's plane and builds nothing "
                              "otherwise. Pick a sweep axis different from the normal axis.", prim.name))
        if prim.kind == PrimitiveKind.BOOLEAN:
            if not prim.children:
                out.append(_issue("error", "empty-boolean", f"Boolean '{prim.name}' has no operands.",
                                  "Add at least two objects to combine.", prim.name))
            elif len(prim.children) < 2 and prim.boolean_op and prim.boolean_op.value != "union":
                out.append(_issue("warning", "single-operand-boolean",
                                  f"Boolean '{prim.name}' has only one operand.",
                                  "Subtract/intersect need at least two.", prim.name))

    frames.ensure_placed(project)
    for cs in project.coordinate_systems:
        if cs.placement_error:
            out.append(_issue("error", "cs-unplaced", f"Coordinate system '{cs.name}' cannot be placed: "
                              f"{cs.placement_error}.",
                              "Everything placed in it would land in the wrong place. Base it on another system "
                              "or put it on another object, or delete it.", cs.name))

    if not project.primitives:
        out.append(_issue("warning", "no-geometry", "The design has no objects.",
                          "Add at least one solid — a port alone radiates into empty space."))
    elif not any(p.model for p in project.primitives):
        out.append(_issue("warning", "no-geometry", "Every object is non-model, so the solver sees none of them.",
                          "Tick Model on the objects that make up the design — a port alone radiates into "
                          "empty space."))
    return out


def _check_domain(project: ModelerProject) -> List[Dict]:
    """Boundary/mesh sanity, in wavelengths — the units that actually matter."""
    out: List[Dict] = []
    f_stop = project.frequency.stop
    f_start = project.frequency.start
    if f_stop <= f_start:
        out.append(_issue("error", "bad-sweep", "Stop frequency must be greater than start frequency.", ""))
        return out
    if f_start <= 0:
        out.append(_issue("warning", "zero-start-frequency", "Start frequency is 0 Hz.",
                          "FDTD has no DC point; start the sweep at a small non-zero frequency (e.g. 1 MHz)."))
    if project.frequency.points < 2:
        out.append(_issue("error", "too-few-points", "The sweep needs at least 2 frequency points.", ""))

    absorbing = [f for f in BOUNDARY_FACES if getattr(project.boundary, f) in ("PML_8", "MUR")]
    # The λ/4 clearance rule doesn't apply along guided-port axes, or to non-radiating designs.
    guided_axes = region.guided_axes(project)
    radiating = project.nf2ff.enabled or not guided_axes
    if absorbing and radiating and not project.boundary.open_region:
        needed = region.quarter_wave_mm(project)
        f_ref, why = region.reference_frequency(project)
        margins = dict(zip("xyz", project.mesh.air_margin))
        for face in absorbing:
            axis = face[0]
            if axis in guided_axes:
                continue
            if margins[axis] < needed * 0.5:
                out.append(_issue(
                    "warning", "tight-air-margin",
                    f"Air margin on {axis} is {margins[axis]:.1f} mm; an absorbing {face} boundary "
                    f"usually wants about {needed:.1f} mm (λ/4 at {f_ref / 1e9:.3f} GHz, {why}).",
                    "Too little space and the PML absorbs reactive near field, which shows up as a "
                    "wrong-looking S11 at the low end. Turn on Open region in Solution setup › Boundaries to "
                    "have the air sized for you, or raise the air margin.",
                ))
                break

    max_eps = max((m.epsilon for m in project.materials if m.kind == MaterialKind.DIELECTRIC), default=1.0)
    res = project.mesh.max_cell_size or wavelength_resolution(f_stop, project.mesh.cells_per_wavelength, max_eps)
    if project.mesh.cells_per_wavelength < 10:
        out.append(_issue("warning", "coarse-mesh",
                          f"{project.mesh.cells_per_wavelength:g} cells per wavelength is coarse.",
                          "Below ~10 the numerical dispersion error becomes visible; 15-20 is the usual working range."))
    # MUR next to a graded mesh can diverge.
    mur_faces = [f for f in BOUNDARY_FACES if getattr(project.boundary, f) == "MUR"]
    if mur_faces and project.mesh.metal_edge_refinement:
        margins = dict(zip("xyz", project.mesh.air_margin))
        tight = [f for f in mur_faces if margins[f[0]] < 4 * res]
        if tight:
            out.append(_issue(
                "warning", "mur-near-graded-mesh",
                f"MUR boundary on {', '.join(tight)} sits within a few cells of the geometry.",
                "MUR assumes a locally uniform grid; next to metal-edge refinement it does not "
                "just reflect, it can make the run diverge (you get |S11| > 0 dB and |S21| near "
                "−120 dB). Use PML_8 on these faces, or increase Mesh > Air margin so the mesh "
                "is uniform where the boundary sits.",
            ))

    if project.mesh.smooth_ratio > 2.0:
        out.append(_issue("warning", "harsh-mesh-grading",
                          f"Mesh smoothing ratio {project.mesh.smooth_ratio:g} is aggressive.",
                          "Neighbouring cells differing by more than ~2x reflect energy at the size step. 1.3-1.5 is typical."))

    if project.nf2ff.enabled:
        if not any(getattr(project.boundary, f) in ("PML_8", "MUR") for f in BOUNDARY_FACES):
            out.append(_issue("error", "nf2ff-no-absorbing",
                              "Far-field (NF2FF) needs at least one absorbing boundary.",
                              "Set the outer faces to PML_8 — a PEC box has no radiation to integrate."))
        if min(project.mesh.air_margin) < project.nf2ff.margin_cells * res:
            out.append(_issue("warning", "nf2ff-tight",
                              "The NF2FF box clearance is larger than the air margin.",
                              "Increase Mesh > Air margin, or lower the NF2FF margin, or the Huygens "
                              "surface ends up inside the PML."))

        for f in project.nf2ff.frequencies:
            if not f_start <= float(f) <= f_stop:
                out.append(_issue("warning", "nf2ff-outside-sweep",
                                  f"Operating frequency {float(f) / 1e9:.4g} GHz is outside the "
                                  f"{f_start / 1e9:.4g}–{f_stop / 1e9:.4g} GHz sweep.",
                                  "The excitation carries little energy there, so the pattern is mostly "
                                  "numerical noise. Widen the sweep or move the frequency inside it."))

    sim = project.simulation
    if sim.max_passes < 1 or sim.max_passes > 20:
        out.append(_issue("error", "bad-pass-count", f"{sim.max_passes} adaptive passes is out of range.",
                          "Use 1 (a single solve) up to 20."))
    if sim.max_passes > 1:
        if sim.delta_s <= 0:
            out.append(_issue("error", "bad-delta-s", "The ΔS target must be above zero.",
                              "0.02 is a usual target; 0.01 is strict."))
        if not 1.05 <= sim.refine_ratio <= 2.0:
            out.append(_issue("warning", "refine-ratio", f"A refinement ratio of {sim.refine_ratio:g} per pass is "
                              "unusual.", "1.2-1.5 refines steadily without a huge jump in cells."))

    for d in project.dumps:
        if d.kind.value.endswith("freq") and not d.frequencies:
            out.append(_issue("warning", "dump-no-frequency",
                              f"Frequency-domain dump '{d.name}' lists no frequencies.",
                              "Add at least one, otherwise openEMS records nothing.", d.name))
    return out


def _check_materials(project: ModelerProject) -> List[Dict]:
    out: List[Dict] = []
    names = [m.name for m in project.materials]
    for name in {n for n in names if names.count(n) > 1}:
        out.append(_issue("error", "duplicate-material", f"More than one material is named '{name}'.",
                          "Material names are the key primitives reference.", name))
    used = {p.material for p in project.primitives}
    for m in project.materials:
        if m.kind == MaterialKind.DIELECTRIC and m.epsilon < 1.0:
            out.append(_issue("warning", "epsilon-below-one", f"'{m.name}' has εr = {m.epsilon:g}, below vacuum.",
                              "FDTD is unstable for εr < 1 at the standard Courant limit.", m.name))
        if m.kind == MaterialKind.CONDUCTOR and m.kappa <= 0:
            out.append(_issue("warning", "conductor-no-kappa", f"Conductor '{m.name}' has zero conductivity.",
                              "It will behave as a lossless dielectric. Copper is 5.8e7 S/m; "
                              "for a perfect conductor use the PEC kind instead.", m.name))
        if m.name not in used and project.primitives:
            out.append(_issue("info", "unused-material", f"Material '{m.name}' is not used by any object.",
                              "Nothing is made of it, so it can be deleted.", m.name))
    return out


def validate(project: ModelerProject) -> Dict:
    issues: List[Dict] = []
    issues += _check_geometry(project)
    issues += _check_materials(project)
    issues += _check_ports(project)
    issues += _check_sheets(project)
    issues += _check_domain(project)

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda i: order[i["severity"]])
    return {
        "issues": issues,
        "errors": sum(1 for i in issues if i["severity"] == "error"),
        "warnings": sum(1 for i in issues if i["severity"] == "warning"),
        "infos": sum(1 for i in issues if i["severity"] == "info"),
        "can_run": not any(i["severity"] == "error" for i in issues),
    }
