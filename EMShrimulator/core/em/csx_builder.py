"""Builds a CSXCAD/openEMS simulation from a resolved ModelerProject (CSXCAD 0.6.3 / openEMS 0.0.36)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import EMShrimulator.bootstrap  # noqa: F401 - must run before importing CSXCAD/openEMS
from CSXCAD import ContinuousStructure
from openEMS import openEMS

from EMShrimulator.core.em import csg, dielectric, sheets
from EMShrimulator.core.em.mesh import wavelength_resolution
from EMShrimulator.core.em.model import (
    DumpKind,
    ExcitationKind,
    Material,
    MaterialKind,
    ModelerProject,
    Port,
    PortKind,
    Primitive,
    PrimitiveKind,
    SheetShape,
)
from EMShrimulator.core.em.transforms import aabb, apply_ops, effective_ops

C0 = 299_792_458.0

_DUMP_TYPE = {
    DumpKind.E_TIME: 0,
    DumpKind.H_TIME: 1,
    DumpKind.CURRENT_TIME: 2,
    DumpKind.E_FREQ: 10,
    DumpKind.H_FREQ: 11,
    DumpKind.SAR_LOCAL: 20,
}


@dataclass
class BuiltSimulation:
    FDTD: object
    CSX: object
    ports: Dict[int, object]                       # port number -> openEMS port object
    nf2ff: object | None = None
    domain_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    domain_max: np.ndarray = field(default_factory=lambda: np.zeros(3))
    resolution_mm: float = 0.0
    dump_names: List[str] = field(default_factory=list)


# --- materials ---

METALS = (MaterialKind.PEC, MaterialKind.CONDUCTOR, MaterialKind.SHEET)


def _add_material(CSX, m: Material, f0: float, f_start: float, f_stop: float, wideband: bool = True,
                  name: Optional[str] = None):
    """A CSXCAD property with the material's values, named `name` (default: the material's)."""
    name = name or m.name
    if m.kind == MaterialKind.PEC:
        return CSX.AddMetal(name)
    if m.kind == MaterialKind.SHEET:
        return CSX.AddConductingSheet(name, conductivity=m.kappa or 5.8e7, thickness=m.thickness_um * 1e-6)
    if m.kind == MaterialKind.DIELECTRIC and m.tan_delta:
        if wideband:
            # tan δ across the sweep via Debye poles, exact at f0.
            return dielectric.add_lossy_dielectric(CSX, name, m.epsilon, m.tan_delta, f_start, f_stop, f0,
                                                   mue=m.mue, kappa=m.kappa)
        kappa = m.kappa + dielectric.centre_conductivity(m.epsilon, m.tan_delta, f0)
        return CSX.AddMaterial(name, epsilon=m.epsilon, mue=m.mue, kappa=kappa)
    return CSX.AddMaterial(name, epsilon=m.epsilon, mue=m.mue, kappa=m.kappa)


def _conductor_name(material: str, prim: str, taken: set) -> str:
    """A free CSXCAD property name for one conductor, e.g. `Copper__patch`; added to `taken`."""
    base = re.sub(r"[^\w.+-]+", "_", f"{material}__{prim}")
    name, n = base, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    taken.add(name)
    return name


# --- geometry ---

def _apply_transform_calls(prim_obj, ops):
    """Replay a resolved transform chain onto a real CSXCAD primitive."""
    for op in ops:
        if op.kind == "translate" and op.translate:
            prim_obj.AddTransform("Translate", list(op.translate))
        elif op.kind in ("rotate_axis", "rotate_origin") and op.axis and op.angle_deg is not None:
            prim_obj.AddTransform("RotateAxis", op.axis, op.angle_deg)
        elif op.kind == "scale" and op.scale:
            prim_obj.AddTransform("Scale", list(op.scale))


def _axis_aligned(prim: Primitive, cs_by_name) -> bool:
    """Whether the box's transforms keep its faces axis-aligned."""
    for op in effective_ops(prim.coordinate_system, prim.transforms, cs_by_name):
        if op.kind in ("rotate_axis", "rotate_origin") and float(op.angle_deg or 0.0) % 90:
            return False
    return True


def _courant_timestep(grid) -> float:
    """The Courant time-step limit for the smallest cells, in seconds."""
    inverse = 0.0
    for axis in "xyz":
        steps = np.diff(np.sort(np.asarray(grid.GetLines(axis), dtype=float)))
        steps = steps[steps > 0]
        if len(steps):
            inverse += 1 / (steps.min() * grid.GetDeltaUnit()) ** 2
    return 1 / (C0 * np.sqrt(inverse)) if inverse else 1.0


def _polygon_uv(points, normal_axis: str):
    """CSXCAD wants two parallel coordinate arrays, not (x, y) pairs."""
    return [[p[0] for p in points], [p[1] for p in points]]


def _thin_layer_lines(fixed: List[float], max_res: float, min_cells: int) -> List[float]:
    """Extra lines inside gaps thinner than one cell; slivers below max_res/20 are skipped."""
    ordered = sorted(set(round(v, 12) for v in fixed))
    out: List[float] = []
    for a, b in zip(ordered[:-1], ordered[1:]):
        gap = b - a
        if gap >= max_res or gap < max_res / 20:
            continue
        for i in range(1, min_cells):
            out.append(a + gap * i / min_cells)
    return out


def _pml_cells(boundary: str) -> int:
    """8 for 'PML_8'; 0 for any other boundary."""
    kind, _, cells = str(boundary).partition("_")
    return int(cells) if kind == "PML" and cells.isdigit() else 0


def _pad_for_pml(grid, boundary, margin: np.ndarray) -> None:
    """Add PML cells beyond the air margin, so the margin is all air.

    Axes with no margin keep the PML inside, so guides can run into it.
    """
    faces = [(boundary.xmin, boundary.xmax), (boundary.ymin, boundary.ymax), (boundary.zmin, boundary.zmax)]
    for n, axis in enumerate("xyz"):
        if margin[n] <= 0:
            continue
        lines = np.unique(np.array([float(v) for v in grid.GetLines(axis)], dtype=float))
        if lines.size < 2:
            continue
        low, high = _pml_cells(faces[n][0]), _pml_cells(faces[n][1])
        below = lines[0] - (lines[1] - lines[0]) * np.arange(low, 0, -1)
        above = lines[-1] + (lines[-1] - lines[-2]) * np.arange(1, high + 1)
        grid.SetLines(axis, np.concatenate([below, lines, above]).tolist())


def _sort_lines(grid) -> None:
    """Sort grid lines: CreateNF2FFBox counts PML cells by list index."""
    for axis in "xyz":
        grid.SetLines(axis, np.unique(np.array([float(v) for v in grid.GetLines(axis)], dtype=float)).tolist())


def _snap_to_fixed(grid, axis: str, fixed: List[float], tol: float = 1e-6) -> None:
    """Snap lines back onto exact geometry coordinates after smoothing.

    openEMS silently drops zero-thickness metal that no grid line exactly touches.
    """
    if not fixed:
        return
    unique_fixed = np.unique(np.array(sorted(set(fixed)), dtype=float))
    lines = np.array([float(v) for v in grid.GetLines(axis)], dtype=float)
    if lines.size == 0:
        return
    lines = np.unique(lines)
    idx = np.searchsorted(unique_fixed, lines).clip(1, len(unique_fixed) - 1)
    left = unique_fixed[idx - 1]
    right = unique_fixed[idx]
    nearest = np.where(np.abs(lines - left) <= np.abs(lines - right), left, right)
    snapped = np.where(np.abs(lines - nearest) <= tol, nearest, lines)
    grid.SetLines(axis, np.unique(snapped).tolist())


def local_corners(p: Primitive) -> np.ndarray:
    """Conservative local-space corners for AABB estimation."""
    k = p.kind
    if k == PrimitiveKind.BOX:
        return np.array([p.p1, p.p2], dtype=float)
    if k == PrimitiveKind.SHEET:
        return sheets.local_points(p)
    if k in (PrimitiveKind.CYLINDER, PrimitiveKind.CYLINDRICAL_SHELL):
        a, b = np.array(p.p1, dtype=float), np.array(p.p2, dtype=float)
        r = (p.radius or 0.0) + (p.shell_width or 0.0) / 2
        length = np.linalg.norm(b - a)
        # The end discs reach r·sqrt(1 - d_i²) along each axis: nothing along the cylinder's own axis, so a line
        # can end exactly on the boundary.
        pad = r * np.sqrt(np.clip(1.0 - ((b - a) / length) ** 2, 0.0, 1.0)) if length > 0 else np.full(3, r)
        return np.array([a - pad, a + pad, b - pad, b + pad])
    if k in (PrimitiveKind.SPHERE, PrimitiveKind.SPHERICAL_SHELL):
        c = np.array(p.center or (0, 0, 0), dtype=float)
        r = (p.radius or 0.0) + (p.shell_width or 0.0) / 2
        pad = np.array([r, r, r])
        return np.array([c - pad, c + pad])
    if k == PrimitiveKind.ROT_POLY:
        normal = p.normal_axis or "z"
        rot = p.rot_axis or csg.default_rot_axis(normal)
        try:
            n, r, _t = csg.rot_poly_axes(normal, rot)
            profile = csg.rot_poly_profile(p.points or [[0, 0]], normal, rot)
        except ValueError:  # sweep axis along the normal: the validator reports it
            return np.zeros((1, 3))
        radius = float(np.abs(profile[:, 0]).max())
        lo, hi = np.full(3, -radius), np.full(3, radius)
        lo[r], hi[r] = profile[:, 1].min(), profile[:, 1].max()
        lo[n] += p.elevation or 0.0
        hi[n] += p.elevation or 0.0
        return np.array([lo, hi])
    if k == PrimitiveKind.POLYGON:
        pts2d = np.array(p.points or [[0, 0]], dtype=float)
        axis = p.normal_axis or "z"
        elev = p.elevation or 0.0
        length = p.extrude_length or 0.0
        corners = []
        for u, v in pts2d:
            for e in (elev, elev + length):
                if axis == "z":
                    corners.append([u, v, e])
                elif axis == "x":
                    corners.append([e, u, v])
                else:  # y
                    corners.append([v, e, u])
        return np.array(corners, dtype=float)
    if k in (PrimitiveKind.WIRE, PrimitiveKind.CURVE):
        path = np.array(p.path or [p.p1, p.p2], dtype=float)
        r = p.radius or 0.0
        return np.vstack([path - r, path + r])
    if k == PrimitiveKind.BOOLEAN:
        return np.vstack([local_corners(c) for c in p.children]) if p.children else np.zeros((1, 3))
    raise ValueError(f"unknown primitive kind {k}")


def global_aabb(p: Primitive, cs_by_name) -> Tuple[np.ndarray, np.ndarray]:
    """Global-space AABB, transforms included. Booleans use the CSG mesh."""
    if p.kind == PrimitiveKind.BOOLEAN and csg.available():
        try:
            verts, _ = csg.mesh_of(p, cs_by_name)
            return aabb(verts)
        except Exception:  # fall through to the conservative estimate
            pass
    pts = local_corners(p)
    if p.kind == PrimitiveKind.BOOLEAN:
        # Children carry their own CS; only the boolean's own ops apply.
        pts = apply_ops(pts, p.transforms)
        return aabb(pts)
    pts = apply_ops(pts, effective_ops(p.coordinate_system, p.transforms, cs_by_name))
    return aabb(pts)


def _add_sheet(prop, prim: Primitive, pr: int):
    """Rectangles become flat boxes, other outlines flat polygons. Rings can't carry material."""
    shape = prim.sheet_shape or SheetShape.RECTANGLE
    axis = prim.normal_axis or "z"
    n = sheets.normal_index(axis)
    elevation = float((prim.origin or (0.0, 0.0, 0.0))[n])

    if shape == SheetShape.RECTANGLE:
        pts = sheets.local_points(prim)
        return prop.AddBox(list(pts.min(axis=0)), list(pts.max(axis=0)), priority=pr)

    if shape == SheetShape.RING:
        raise ValueError(
            f"sheet '{prim.name}' is a ring, which has a hole and cannot be built as a "
            f"filled surface; give it no material and use it for a coaxial port, or model "
            f"the metal with a Tube primitive"
        )

    loops = [loop for loop in sheets.outline(prim) if loop]
    if not loops:
        raise ValueError(f"sheet '{prim.name}' has no outline")
    iu, iv = sheets.UV_AXES[axis]
    origin = [float(c) for c in (prim.origin or (0.0, 0.0, 0.0))]
    uv = np.array([[origin[iu] + u, origin[iv] + v] for u, v in loops[0]], dtype=float).T
    return prop.AddPolygon(uv, norm_dir=axis, elevation=elevation, priority=pr)


def _add_primitive(prop, prim: Primitive, cs_by_name):
    """Attach one primitive to a CSXCAD property, returning the CSX object."""
    k = prim.kind
    pr = prim.priority

    if k == PrimitiveKind.BOOLEAN:
        obj, _, _ = csg.add_to_property(prop, prim, cs_by_name, pr)
        return obj, True  # transforms already baked into the evaluated mesh

    if k == PrimitiveKind.BOX:
        obj = prop.AddBox(list(prim.p1), list(prim.p2), priority=pr)
    elif k == PrimitiveKind.SHEET:
        obj = _add_sheet(prop, prim, pr)
    elif k == PrimitiveKind.CYLINDER:
        obj = prop.AddCylinder(list(prim.p1), list(prim.p2), prim.radius, priority=pr)
    elif k == PrimitiveKind.CYLINDRICAL_SHELL:
        obj = prop.AddCylindricalShell(list(prim.p1), list(prim.p2), prim.radius, prim.shell_width, priority=pr)
    elif k == PrimitiveKind.SPHERE:
        obj = prop.AddSphere(list(prim.center or (0, 0, 0)), prim.radius, priority=pr)
    elif k == PrimitiveKind.SPHERICAL_SHELL:
        obj = prop.AddSphericalShell(list(prim.center or (0, 0, 0)), prim.radius, prim.shell_width, priority=pr)
    elif k == PrimitiveKind.POLYGON:
        uv = _polygon_uv(prim.points, prim.normal_axis or "z")
        if prim.extrude_length:
            obj = prop.AddLinPoly(
                uv, norm_dir=prim.normal_axis or "z", elevation=prim.elevation or 0.0,
                length=prim.extrude_length, priority=pr,
            )
        else:
            obj = prop.AddPolygon(uv, norm_dir=prim.normal_axis or "z", elevation=prim.elevation or 0.0, priority=pr)
    elif k == PrimitiveKind.ROT_POLY:
        normal = prim.normal_axis or "z"
        uv = _polygon_uv(prim.points, normal)
        # CSXCAD revolves only with the outline plane through the origin; elevation shifts the whole solid.
        obj = prop.AddRotPoly(
            uv, norm_dir=normal, elevation=0.0,
            rot_axis=prim.rot_axis or csg.default_rot_axis(normal),
            angle=[np.radians(prim.angle_start_deg or 0.0), np.radians(prim.angle_stop_deg if prim.angle_stop_deg is not None else 360.0)],
            priority=pr,
        )
        if prim.elevation:
            shift = [0.0, 0.0, 0.0]
            shift["xyz".index(normal)] = float(prim.elevation)
            obj.AddTransform("Translate", shift)
    elif k == PrimitiveKind.WIRE:
        path = prim.path or [prim.p1, prim.p2]
        obj = prop.AddWire(np.array(path, dtype=float).T, prim.radius or 0.1, priority=pr)
    elif k == PrimitiveKind.CURVE:
        path = prim.path or [prim.p1, prim.p2]
        obj = prop.AddCurve(np.array(path, dtype=float).T, priority=pr)
    else:
        raise ValueError(f"unsupported primitive kind {k}")
    return obj, False


# --- ports ---

def _coax_mode_functions(r_in: float, r_out: float, axis: str, center: Tuple[float, float]):
    """Analytic TEM coax mode (E, H ~ 1/r, kc = 0) for AddWaveGuidePort, zero outside the gap: the port's box is
    square, and its corners reach past the shield."""
    u, v = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}[axis]
    cu, cv = center
    r = f"sqrt(({u}-{cu})*({u}-{cu}) + ({v}-{cv})*({v}-{cv}))"

    def gap(expr: str) -> str:
        return f"if(({r}>={r_in})*({r}<={r_out}), {expr}, 0)"

    E, H = ["0"] * 3, ["0"] * 3
    iu, iv = "xyz".index(u), "xyz".index(v)
    E[iu], E[iv] = gap(f"(({u}-{cu})/{r})/{r}"), gap(f"(({v}-{cv})/{r})/{r}")
    H[iu], H[iv] = gap(f"(-({v}-{cv})/{r})/{r}"), gap(f"(({u}-{cu})/{r})/{r}")
    return E, H, 0.0


def _filling_epsilon(CSX, point) -> float:
    """εr of the material filling a mode port at `point`: 1 for air or nothing."""
    try:
        for prop in CSX.GetAllProperties():  # the lookup only sees primitives that have been updated
            for prim in prop.GetAllPrimitives():
                prim.Update()
        prop = CSX.GetPropertyByCoordPriority(list(point))
        eps = prop.GetMaterialProperty("epsilon") if prop is not None and hasattr(prop, "GetMaterialProperty") else 1
        return float(np.mean(eps)) or 1.0
    except Exception:  # noqa: BLE001 - a lookup that fails leaves the port as air
        return 1.0


def _port_geometry(project: ModelerProject, port: Port, cs_by_name, mesh_res: float = 0.0):
    """See sheets.port_geometry."""
    return sheets.port_geometry(project, port, cs_by_name, mesh_res)


def _add_port(FDTD, CSX, port: Port, excite_weight: float, mesh_res: float, placement: sheets.PortPlacement):
    """Create one openEMS port from its placed box."""
    p1, p2, sheet = placement.start, placement.stop, placement.sheet
    start, stop = list(p1), list(p2)
    sheet_axis = placement.prop_axis

    if port.kind == PortKind.LUMPED:
        obj = FDTD.AddLumpedPort(
            port.number, port.impedance, start, stop, placement.exc_axis or port.direction,
            excite_weight, priority=port.priority, edges2grid="xy",
        )
    elif port.kind == PortKind.MICROSTRIP:
        prop_dir = sheet_axis or port.prop_dir or "x"
        exc_dir = placement.exc_axis or port.exc_dir or "z"
        axis = {"x": 0, "y": 1, "z": 2}[prop_dir]
        port_length = abs(p2[axis] - p1[axis])
        # Both shifts must land inside the port.
        feed_shift = port.feed_shift_mm or min(10 * mesh_res, 0.3 * port_length)
        meas_shift = port.measure_plane_shift_mm or max(0.6 * port_length, feed_shift + 2 * mesh_res)
        meas_shift = min(meas_shift, 0.9 * port_length)
        obj = FDTD.AddMSLPort(
            port.number, CSX.AddMetal(f"__port{port.number}_msl"), start, stop,
            prop_dir, exc_dir, excite=excite_weight, priority=port.priority,
            Feed_R=port.impedance,
            FeedShift=feed_shift,
            MeasPlaneShift=meas_shift,
        )
    elif port.kind == PortKind.RECT_WAVEGUIDE:
        p_dir = sheet_axis or port.prop_dir or port.direction
        if sheet is not None:
            a, b = sheets.extent_uv(sheet)
        else:
            a = port.wg_a_mm or abs(p2[0] - p1[0])
            b = port.wg_b_mm or abs(p2[1] - p1[1])
        unit = CSX.GetGrid().GetDeltaUnit()  # openEMS takes a and b in metres
        obj = FDTD.AddRectWaveGuidePort(
            port.number, start, stop, p_dir, a * unit, b * unit, port.mode or "TE10",
            excite=excite_weight, priority=port.priority,
        )
        obj.fill_epsilon = _filling_epsilon(CSX, [(p1[i] + p2[i]) / 2 for i in range(3)])
    elif port.kind == PortKind.COAXIAL:
        p_dir = sheet_axis or port.prop_dir or port.direction
        idx = {"x": 0, "y": 1, "z": 2}[p_dir]
        transverse = [i for i in range(3) if i != idx]
        if sheet is not None:
            r_in, r_out = sheets.coax_radii(sheet)
        else:
            r_in = port.r_inner_mm or 0.5
            r_out = port.r_outer_mm or 1.5
        center = ((p1[transverse[0]] + p2[transverse[0]]) / 2,
                  (p1[transverse[1]] + p2[transverse[1]]) / 2)
        E, H, kc = _coax_mode_functions(r_in, r_out, p_dir, center)
        obj = FDTD.AddWaveGuidePort(
            port.number, start, stop, p_dir, E, H, kc,
            excite=excite_weight, priority=port.priority,
        )
        point = [(p1[i] + p2[i]) / 2 for i in range(3)]
        point[transverse[0]] = center[0] + (r_in + r_out) / 2
        point[transverse[1]] = center[1]
        obj.fill_epsilon = _filling_epsilon(CSX, point)
    else:
        raise ValueError(f"unsupported port kind {port.kind}")

    return obj


# --- build ---

def build_simulation(project: ModelerProject, excite_port_number: Optional[int]) -> BuiltSimulation:
    """Build FDTD+CSX exciting only `excite_port_number` (None = no excitation, for previews)."""
    cs_by_name = {cs.name: cs for cs in project.coordinate_systems}

    f_start, f_stop = project.frequency.start, project.frequency.stop
    exc = project.simulation.excitation
    f0 = exc.f0 if exc.f0 else (f_start + f_stop) / 2
    fc = exc.fc if exc.fc else max((f_stop - f_start) / 2, f_stop * 0.1)

    FDTD = openEMS(NrTS=project.simulation.max_timesteps, EndCriteria=project.simulation.end_criteria)
    if exc.kind == ExcitationKind.GAUSSIAN:
        FDTD.SetGaussExcite(f0, fc)
    elif exc.kind == ExcitationKind.SINUSOIDAL:
        FDTD.SetSinusExcite(exc.frequency or f0)
    elif exc.kind == ExcitationKind.STEP:
        FDTD.SetStepExcite(f_stop)
    elif exc.kind == ExcitationKind.DIRAC:
        FDTD.SetDiracExcite(f_stop)
    if project.simulation.oversampling:
        FDTD.SetOverSampling(project.simulation.oversampling)

    b = project.boundary
    FDTD.SetBoundaryCond([b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    grid = CSX.GetGrid()
    grid.SetDeltaUnit(1e-3)  # project units are millimeters

    materials_by_name = {m.name: m for m in project.materials}
    csx_props: Dict[str, object] = {}
    wideband = project.simulation.dielectric_loss == "wideband"

    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    fixed_lines: Dict[str, List[float]] = {"x": [], "y": [], "z": []}
    metal_props_used: List[object] = []

    max_epsilon = max((m.epsilon for m in project.materials if m.kind == MaterialKind.DIELECTRIC), default=1.0)
    auto_res = wavelength_resolution(f_stop, project.mesh.cells_per_wavelength, max_epsilon)
    max_res = project.mesh.max_cell_size if project.mesh.max_cell_size else auto_res

    taken_names = {m.name for m in project.materials}
    for prim in project.primitives:
        if not sheets.is_built(prim):
            continue
        if prim.material not in materials_by_name:
            raise ValueError(f"'{prim.name}' references unknown material '{prim.material}'")
        mat = materials_by_name[prim.material]
        if mat.kind in METALS:
            # Every conductor gets its own property, as in the openEMS examples.
            prop = _add_material(CSX, mat, f0, f_start, f_stop, wideband=wideband,
                                 name=_conductor_name(mat.name, prim.name, taken_names))
        else:
            if prim.material not in csx_props:
                csx_props[prim.material] = _add_material(CSX, mat, f0, f_start, f_stop, wideband=wideband)
            prop = csx_props[prim.material]

        obj, transforms_baked = _add_primitive(prop, prim, cs_by_name)
        if not transforms_baked:
            _apply_transform_calls(obj, effective_ops(prim.coordinate_system, prim.transforms, cs_by_name))

        pmin, pmax = global_aabb(prim, cs_by_name)
        global_min = np.minimum(global_min, pmin)
        global_max = np.maximum(global_max, pmax)
        for axis_idx, axis_name in enumerate("xyz"):
            fixed_lines[axis_name].extend([pmin[axis_idx], pmax[axis_idx]])

        if mat.kind in METALS:
            metal_props_used.append(prop)

    # Debye poles at half strength on faces bordering air, matching openEMS's ε/σ averaging.
    lossy = {name: prop for name, prop in csx_props.items() if wideband
             and materials_by_name[name].kind == MaterialKind.DIELECTRIC and materials_by_name[name].tan_delta}
    lossy_boxes: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {name: [] for name in lossy}
    for prim in project.primitives:
        if prim.model and prim.material in lossy and prim.kind == PrimitiveKind.BOX and _axis_aligned(prim, cs_by_name):
            lossy_boxes[prim.material].append(global_aabb(prim, cs_by_name))
    every_box = [box for boxes in lossy_boxes.values() for box in boxes]

    # Port boxes feed the mesh; port objects are created after it.
    resolved_ports = []
    for port in sorted(project.ports, key=lambda p: p.number):
        placement = _port_geometry(project, port, cs_by_name, max_res)
        resolved_ports.append((port, placement))
        pmin, pmax = np.minimum(placement.start, placement.stop), np.maximum(placement.start, placement.stop)
        for axis_idx, axis_name in enumerate("xyz"):
            fixed_lines[axis_name].extend([pmin[axis_idx], pmax[axis_idx]])
        global_min = np.minimum(global_min, pmin)
        global_max = np.maximum(global_max, pmax)

    if not np.isfinite(global_min).all():
        raise ValueError("Project has no geometry or ports to determine a simulation domain")

    margin = np.array(project.mesh.air_margin, dtype=float)
    domain_min = global_min - margin
    domain_max = global_max + margin
    for axis_idx, axis_name in enumerate("xyz"):
        fixed_lines[axis_name].extend([domain_min[axis_idx], domain_max[axis_idx]])

    dump_names: List[str] = []
    for dump in project.dumps:
        dump_type = _DUMP_TYPE[dump.kind]
        props = {
            "dump_type": dump_type,
            # HDF5 for frequency/SAR, VTK for time domain
            "file_type": 1 if dump_type >= 10 else 0,
        }
        if dump.frequencies:
            props["frequency"] = list(dump.frequencies)
        if dump.sub_sampling:
            props["sub_sampling"] = f"{dump.sub_sampling},{dump.sub_sampling},{dump.sub_sampling}"
        dbox = CSX.AddDump(dump.name, **props)
        dbox.AddBox(list(dump.p1), list(dump.p2))
        dump_names.append(dump.name)

    for axis_name in "xyz":
        lines = fixed_lines[axis_name] + list(getattr(project.mesh, f"extra_lines_{axis_name}"))
        if project.mesh.min_cells_per_layer > 1:
            lines += _thin_layer_lines(lines, max_res, project.mesh.min_cells_per_layer)
        grid.AddLine(axis_name, lines)
    grid.SmoothMeshLines("all", max_res, project.mesh.smooth_ratio)
    for axis_name in "xyz":
        _snap_to_fixed(grid, axis_name, fixed_lines[axis_name])

    if project.mesh.metal_edge_refinement:
        for prop in metal_props_used:
            FDTD.AddEdges2Grid(dirs=project.mesh.metal_edge_dirs, properties=prop, metal_edge_res=max_res / 2)
        for axis_name in "xyz":
            _snap_to_fixed(grid, axis_name, fixed_lines[axis_name])

    # The air box ends at the margin; absorbers go beyond it.
    _pad_for_pml(grid, project.boundary, margin)
    air_min, air_max = domain_min.copy(), domain_max.copy()
    domain_min = np.array([min(float(v) for v in grid.GetLines(a)) for a in "xyz"])
    domain_max = np.array([max(float(v) for v in grid.GetLines(a)) for a in "xyz"])

    if lossy:
        # openEMS drops poles faster than 3 timesteps; keep every pole well above that.
        max_pole_hz = 1 / (2 * np.pi * 8 * _courant_timestep(grid))
        for name, prop in lossy.items():
            m = materials_by_name[name]
            fit = dielectric.constant_loss(m.epsilon, m.tan_delta, f_start, f_stop, f0, max_pole_hz=max_pole_hz)
            dielectric.set_poles(prop, fit, base_kappa=m.kappa)
            if lossy_boxes[name]:
                dielectric.weight_poles(prop, dielectric.face_weight(lossy_boxes[name], every_box, 1e-6))

    ports: Dict[int, object] = {}
    for port, placement in resolved_ports:
        weight = 1.0 if (excite_port_number is not None and port.number == excite_port_number) else 0.0
        obj = _add_port(FDTD, CSX, port, weight, max_res, placement)
        ports[port.number] = obj

    nf2ff = None
    if project.nf2ff.enabled:
        _sort_lines(grid)
        if project.nf2ff.margin_cells > 0:
            inset = project.nf2ff.margin_cells * max_res
            nf_min = air_min + inset
            nf_max = air_max - inset
            if np.any(nf_max <= nf_min):
                raise ValueError(
                    "NF2FF box collapsed: the air margin is too small for "
                    f"{project.nf2ff.margin_cells} cells of clearance — increase Mesh > Air margin"
                )
            nf2ff = FDTD.CreateNF2FFBox("nf2ff", list(nf_min), list(nf_max))
        else:
            # Must come after the grid and boundaries.
            nf2ff = FDTD.CreateNF2FFBox("nf2ff")

    return BuiltSimulation(
        FDTD=FDTD, CSX=CSX, ports=ports, nf2ff=nf2ff,
        domain_min=domain_min, domain_max=domain_max,
        resolution_mm=max_res, dump_names=dump_names,
    )
