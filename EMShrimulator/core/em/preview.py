"""Mesh and geometry previews from the real engine (CSXCAD grid, manifold3d meshes), without an FDTD run."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

import EMShrimulator.bootstrap  # noqa: F401
from EMShrimulator.core.em import csg, sheets
from EMShrimulator.core.em.csx_builder import C0, build_simulation, global_aabb
from EMShrimulator.core.em.transforms import apply_ops, effective_ops
from EMShrimulator.core.em.model import MaterialKind, ModelerProject, PrimitiveKind


def _decimate(lines: List[float], limit: int = 400) -> List[float]:
    """Thin a grid axis for display, keeping both ends."""
    if len(lines) <= limit:
        return lines
    stride = int(np.ceil(len(lines) / limit))
    kept = lines[::stride]
    if kept[-1] != lines[-1]:
        kept.append(lines[-1])
    return kept


def mesh_preview(project: ModelerProject) -> Dict:
    """Real CSXCAD grid lines + the run-size numbers they imply."""
    built = build_simulation(project, excite_port_number=None)
    grid = built.CSX.GetGrid()
    axes = {}
    counts = {}
    for axis in "xyz":
        lines = [float(v) for v in grid.GetLines(axis)]
        counts[axis] = len(lines)
        axes[axis] = _decimate(lines)

    nx, ny, nz = counts["x"], counts["y"], counts["z"]
    cells = max(nx - 1, 1) * max(ny - 1, 1) * max(nz - 1, 1)

    # Smallest cell sets the Courant timestep.
    min_delta_mm = np.inf
    for axis in "xyz":
        lines = np.array([float(v) for v in grid.GetLines(axis)])
        if len(lines) > 1:
            # Ignore zero gaps from coincident refinement lines.
            deltas = np.diff(lines)
            deltas = deltas[deltas > 1e-9]
            if deltas.size:
                min_delta_mm = min(min_delta_mm, float(np.min(deltas)))
    dt_s = None
    if np.isfinite(min_delta_mm) and min_delta_mm > 0:
        # 3D Courant limit for a uniform-ish cell: dt <= dx / (c * sqrt(3))
        dt_s = (min_delta_mm * 1e-3) / (C0 * np.sqrt(3))

    max_eps = max((m.epsilon for m in project.materials if m.kind == MaterialKind.DIELECTRIC), default=1.0)

    return {
        "lines": axes,
        "line_counts": counts,
        "cells": cells,
        "resolution_mm": built.resolution_mm,
        "min_cell_mm": None if not np.isfinite(min_delta_mm) else min_delta_mm,
        "timestep_s": dt_s,
        "estimated_seconds": None if dt_s is None else project.simulation.max_timesteps * dt_s,
        "domain_min": [float(v) for v in built.domain_min],
        "domain_max": [float(v) for v in built.domain_max],
        "max_epsilon": max_eps,
        "dumps": built.dump_names,
        "nf2ff": project.nf2ff.enabled,
    }


def geometry_preview(project: ModelerProject) -> Dict:
    """Triangle meshes for booleans and curved kinds, plus sheet outlines."""
    cs_by_name = {cs.name: cs for cs in project.coordinate_systems}
    tess_kinds = {
        PrimitiveKind.BOOLEAN, PrimitiveKind.ROT_POLY,
        PrimitiveKind.WIRE, PrimitiveKind.CURVE,
        PrimitiveKind.CYLINDRICAL_SHELL, PrimitiveKind.SPHERICAL_SHELL,
    }
    sheet_outlines = []
    for prim in project.primitives:
        if not sheets.is_sheet(prim):
            continue
        ops = effective_ops(prim.coordinate_system, prim.transforms, cs_by_name)
        loops = []
        for loop in sheets.outline(prim):
            if not loop:
                continue
            pts = sheets.lift(prim, loop)
            if ops:
                pts = apply_ops(pts, ops)
            loops.append([float(v) for v in pts.ravel()])
        normal = sheets.lift(prim, [(0.0, 0.0)])
        if ops:
            normal = apply_ops(normal, ops)
        sheet_outlines.append({
            "name": prim.name,
            "shape": (prim.sheet_shape.value if prim.sheet_shape else "rectangle"),
            "normal_axis": prim.normal_axis or "z",
            "material": prim.material or None,
            "construction": sheets.is_construction(prim),
            "center": [float(v) for v in normal[0]],
            "loops": loops,
        })

    meshes = []
    errors = []
    for prim in project.primitives:
        if prim.kind not in tess_kinds:
            continue
        try:
            verts, tris = csg.mesh_of(prim, cs_by_name)
            meshes.append({
                "name": prim.name,
                "material": prim.material,
                "positions": verts.astype(float).ravel().tolist(),
                "indices": tris.astype(int).ravel().tolist(),
            })
        except Exception as exc:  # noqa: BLE001 - reported to the UI, not fatal
            errors.append({"name": prim.name, "error": str(exc)})

    bounds_min = np.array([np.inf] * 3)
    bounds_max = np.array([-np.inf] * 3)
    for prim in project.primitives:
        try:
            pmin, pmax = global_aabb(prim, cs_by_name)
            bounds_min = np.minimum(bounds_min, pmin)
            bounds_max = np.maximum(bounds_max, pmax)
        except Exception:  # noqa: BLE001
            continue

    return {
        "meshes": meshes,
        "sheets": sheet_outlines,
        "errors": errors,
        "csg_available": csg.available(),
        "bounds": None if not np.isfinite(bounds_min).all() else {
            "min": [float(v) for v in bounds_min],
            "max": [float(v) for v in bounds_max],
        },
    }
