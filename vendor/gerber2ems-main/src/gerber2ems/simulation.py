"""Module containing Simulation class used for interacting with openEMS."""

import logging
import os
import re
import sys
import math
from typing import Tuple, List, Any, Optional
from multiprocessing import Pool

import CSXCAD
import openEMS
import numpy as np

from gerber2ems.config import Config, PortConfig, LayerKind
from gerber2ems.constants import (
    UNIT_MULTIPLIER,
    BASE_UNIT,
    SIMULATION_DIR,
    GEOMETRY_DIR,
    VIA_POLYGON,
)
import gerber2ems.importer as importer
from gerber2ems.convergence import DeltaSMonitor, PortProbes, pulse_length_s
from gerber2ems.grid_gen import GridGenerator
from gerber2ems.gerber_io import Pad, ApertureRect, Aperture, Position

logger = logging.getLogger(__name__)
cfg = Config()


def port_kind(port_config: PortConfig) -> str:
    """"microstrip", "lumped" or "circuit"; "resistive" is an old name for lumped."""
    kind = (getattr(port_config, "type", "microstrip") or "microstrip").lower()
    return "lumped" if kind == "resistive" else kind


class Simulation:
    """Class used for interacting with openEMS."""

    def __init__(self) -> None:
        """Initialize simulation object."""
        self.csx = CSXCAD.ContinuousStructure()
        # PATCHED (EMShrimulator Studio): the energy end criterion is configurable (was openEMS's 1e-5).
        self.fdtd = openEMS.openEMS(NrTS=cfg.max_steps, EndCriteria=cfg.convergence.end_criteria)
        self.fdtd.SetCSX(self.csx)
        self.grid = self.csx.GetGrid()
        self.grid.SetDeltaUnit(BASE_UNIT / UNIT_MULTIPLIER)

        self.ports: List[openEMS.ports.MSLPort] = []

        # Separate metal materials for easier switching of layers
        self.gerber_materials: List[Any] = []
        self.substrate_materials: List[Any] = []
        self.plane_material = self.csx.AddMetal("Plane")
        self.via_material = self.csx.AddMetal("Via")
        self.via_filling_material = self.csx.AddMaterial("ViaFilling", epsilon=cfg.via.filling_epsilon)
        self._via_fills: dict = {}  # padstack vias' fill materials, by εr
        self._backdrill: Any = None

    @staticmethod
    def field_substrates() -> set:
        """Indices (into cfg.get_substrates()) of the substrates the ports' lines run in.

        PATCHED (EMShrimulator Studio): only these get Debye poles, which are slow.
        """
        above_each = []  # per substrate, the index of the metal layer above it (-1: none)
        metal = -1
        for layer in cfg.layers:
            if layer.kind == LayerKind.METAL:
                metal += 1
            elif layer.kind == LayerKind.SUBSTRATE:
                above_each.append(metal)
        near = set()
        for port in cfg.ports:
            low, high = sorted((port.layer, port.plane))
            for i, above in enumerate(above_each):
                below = above + 1  # substrates between the same two metals share the gap
                if port.layer in (above, below) or (above >= low and below <= high):
                    near.add(i)
        return near

    def create_materials(self) -> None:
        """Create materials required for simulation."""
        for i, _ in enumerate(cfg.get_metals()):
            self.gerber_materials.append(self.csx.AddMetal(f"Gerber_{i}"))
        near = Simulation.field_substrates()
        self.dispersive_substrates = set()
        for i, layer in enumerate(cfg.get_substrates()):
            # PATCHED (EMShrimulator Studio): substrate tan δ as Debye poles ("wideband") or a conductivity ("centre").
            tan_delta = getattr(layer, "loss_tangent", 0.0)
            start, stop = cfg.frequency.start, cfg.frequency.stop
            if tan_delta > 0 and cfg.dielectric_loss == "wideband" and i in near:
                self.dispersive_substrates.add(i)
                from EMShrimulator.core.em.dielectric import add_lossy_dielectric

                material = add_lossy_dielectric(self.csx, f"Substrate_{i}", layer.epsilon, tan_delta,
                                                start, stop, (start + stop) / 2)
            elif tan_delta > 0:
                from EMShrimulator.core.em.dielectric import centre_conductivity

                material = self.csx.AddMaterial(f"Substrate_{i}", epsilon=layer.epsilon,
                                                kappa=centre_conductivity(layer.epsilon, tan_delta, (start + stop) / 2))
            else:
                material = self.csx.AddMaterial(f"Substrate_{i}", epsilon=layer.epsilon)
            self.substrate_materials.append(material)
        lossy = [i for i, layer in enumerate(cfg.get_substrates()) if getattr(layer, "loss_tangent", 0.0) > 0]
        if lossy and cfg.dielectric_loss == "wideband":
            names = [layer.name for layer in cfg.get_substrates()]
            logger.info("Dielectric loss: across the band (Debye poles) in %s; at the band centre in %s",
                        ", ".join(names[i] for i in sorted(self.dispersive_substrates)) or "none",
                        ", ".join(names[i] for i in lossy if i not in self.dispersive_substrates) or "none")

    def print_grid_stats(self) -> None:
        """Print stats about grid: line count, minimal spacing, positions."""
        logger.debug("Grid x lines: %s", self.grid.GetLines("x"))
        logger.debug("Grid y lines: %s", self.grid.GetLines("y"))
        logger.debug("Grid z lines: %s", self.grid.GetLines("z"))

        def get_stat(lines: List[float]) -> List[float]:
            msize = math.inf
            mscale = -math.inf
            prev = lines[1]
            prev_size = abs((lines[0] - lines[1]))
            for line in lines:
                size = abs(prev - line)
                msize = min(size, msize)
                scale = prev_size / size
                scale = scale if scale > 1 else 1 / scale
                mscale = max(scale, mscale)
                prev = line
                prev_size = size
            return [msize / UNIT_MULTIPLIER, mscale]

        sx = get_stat(self.grid.GetLines("x"))
        sy = get_stat(self.grid.GetLines("y"))
        sz = get_stat(self.grid.GetLines("z"))
        xyz = [
            self.grid.GetQtyLines("x"),
            self.grid.GetQtyLines("y"),
            self.grid.GetQtyLines("z"),
        ]
        logger.info(
            "Grid line count, x: %d, y: %d z: %d. Total number of cells: ~%.2fM",
            xyz[0],
            xyz[1],
            xyz[2],
            xyz[0] * xyz[1] * xyz[2] / 1.0e6,
        )
        logger.info("Minimal cell size, x: %f, y: %f z: %f [um]", sx[0], sy[0], sz[0])
        logger.info("Max cell size ratio, x: %f, y: %f z: %f ", sx[1], sy[1], sz[1])

    def add_grid(self) -> None:
        """Add grid to simulation."""
        self.grid_gen = GridGenerator()
        self.add_port_grid()
        logger.info("Compiling grid")
        self.grid = self.grid_gen.generate(self.grid)

        self.print_grid_stats()

    def add_gerbers(self) -> None:
        """Add metal from all gerber files."""
        logger.info("Adding copper from gerber files")

        offset = 0
        index = 0
        contours = []
        with Pool(initargs=(cfg._config,), initializer=Config.set_config) as p:
            contours = p.map(
                importer.get_triangles,
                [lc.file + ".png" for lc in cfg.layers if lc.kind == LayerKind.METAL],
            )

        icontours = iter(contours)
        for layer in cfg.layers:
            if layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
            elif layer.kind == LayerKind.METAL:
                logger.info("Adding metal mesh for %s", layer.file)
                self.add_contours(next(icontours), offset, index)
                index += 1

    def add_contours(self, contours: np.ndarray, z_height: float, layer_index: int) -> None:
        """Add contours as flat polygons on specified z-height."""
        logger.debug("Adding contours on z=%f", z_height)
        for contour in contours:
            points: List[List[float]] = [[], []]
            for point in contour:
                # Half of the border thickness is subtracted as image is shifted by it
                points[0].append((point[1]))
                points[1].append(cfg.pcb_height - point[0])

            self.gerber_materials[layer_index].AddPolygon(points, "z", z_height, priority=1)

    def get_metal_layer_offset(self, index: int) -> float:
        """Get z offset of nth metal layer."""
        current_metal_index = -1
        offset = 0
        for layer in cfg.layers:
            if layer.kind == LayerKind.METAL:
                current_metal_index += 1
                if current_metal_index == index:
                    return offset
            elif layer.kind == LayerKind.SUBSTRATE:
                offset -= layer.thickness
        logger.error("Hadn't found %dth metal layer", index)
        sys.exit(1)

    def _add_grid_rect(self, cx: float, cy: float, wx: float, wy: float) -> None:
        ap = f"PORT{len(self.grid_gen.add_apertures) + 1}"
        self.grid_gen.add_pads.append(Pad(ap, "PORT", Position(cx + self.grid_gen.xmin, cy + self.grid_gen.ymin)))
        self.grid_gen.add_apertures[ap] = Aperture("", ApertureRect(abs(wx), abs(wy)))

    def add_port_grid(self) -> None:
        """Add grid around ports for simulation.

        PATCHED (EMShrimulator Studio): skip ports with no position; refine lumped and circuit ports' own footprint.
        """
        logger.info("Adding ports grid")

        for port_config in cfg.ports:
            if port_config.position is None or port_config.direction is None:
                logger.error("Port has no defined position or rotation, skipping")
                continue
            kind = port_kind(port_config)
            if kind == "circuit":
                box, leads = self.circuit_port_geometry(port_config)
                if box is None:
                    continue
                for (x1, y1, x2, y2) in [(box[0][0], box[0][1], box[1][0], box[1][1])] + leads:
                    self._add_grid_rect((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                continue
            if kind == "lumped":
                start, stop = self.lumped_port_footprint(port_config)
                self._add_grid_rect((start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2,
                                    stop[0] - start[0], stop[1] - start[1])
                continue
            angle = port_config.direction / 360 * 2 * math.pi
            (w, h) = ((port_config.width), (port_config.length))
            (width, height) = (
                w * round(math.cos(angle)) - h * round(math.sin(angle)),
                w * round(math.sin(angle)) + h * round(math.cos(angle)),
            )
            self._add_grid_rect(port_config.position[0] + width / 2, port_config.position[1], width, height)

    def add_msl_port(self, port_config: PortConfig, port_number: int, excite: bool = False) -> None:
        """Add microstripline port based on config."""
        logger.debug("Adding port number %d", len(self.ports))

        if port_config.position is None or port_config.direction is None:
            logger.error("Port has no defined position or rotation, skipping")
            return

        while port_config.direction < 0:
            port_config.direction += 360

        dir_map = {0: "y", 90: "x", 180: "y", 270: "x"}
        if int(port_config.direction) not in dir_map:
            logger.error("Ports rotation is not a multiple of 90 degrees which is not supported, skipping")
            return

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane)

        angle = port_config.direction / 360 * 2 * math.pi

        start = [
            round(port_config.position[0] - (port_config.width / 2) * round(math.cos(angle))),
            round(port_config.position[1] - (port_config.width / 2) * round(math.sin(angle))),
            round(start_z),
        ]
        stop = [
            round(
                port_config.position[0]
                + (port_config.width / 2) * round(math.cos(angle))
                - port_config.length * round(math.sin(angle))
            ),
            round(
                port_config.position[1]
                + (port_config.width / 2) * round(math.sin(angle))
                + port_config.length * round(math.cos(angle))
            ),
            round(stop_z),
        ]
        logger.debug("Adding port at start: %s end: %s", start, stop)

        port = self.fdtd.AddMSLPort(
            port_number,
            self.csx.AddMetal(f"Port_{port_number}"),
            start,
            stop,
            dir_map[int(port_config.direction)],
            "z",
            Feed_R=port_config.impedance,
            priority=100,
            excite=1 if excite else 0,
        )
        self.ports.append(port)

    def snap_to_mesh(self, start: List[float], stop: List[float], exc_axis: Optional[int] = None,
                     ) -> Tuple[List[float], List[float]]:
        """Move a box onto existing mesh lines.

        PATCHED (EMShrimulator Studio): an off-grid lumped port excites nothing, so its box is snapped to the grid.
        """
        start, stop = list(start), list(stop)
        for i, axis in enumerate("xyz"):
            lines = np.asarray(self.grid.GetLines(axis), float)
            if not len(lines):
                continue
            a = int(np.argmin(np.abs(lines - start[i])))
            b = int(np.argmin(np.abs(lines - stop[i])))
            if i == exc_axis and a == b and len(lines) > 1:
                if stop[i] >= start[i]:
                    b = a + 1 if a + 1 < len(lines) else a - 1
                else:
                    b = a - 1 if a > 0 else a + 1
            start[i], stop[i] = float(lines[a]), float(lines[b])
        return start, stop

    @staticmethod
    def lumped_port_footprint(port_config: PortConfig) -> Tuple[List[float], List[float]]:
        """(start, stop) xy of a lumped port's sheet: `width` along the axis nearest its rotation."""
        angle = port_config.direction / 360 * 2 * math.pi
        half = port_config.width / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        if abs(cos_a) >= abs(sin_a):
            dx, dy = (1 if cos_a >= 0 else -1), 0
        else:
            dx, dy = 0, (1 if sin_a >= 0 else -1)
        x, y = port_config.position
        return [x - half * dx, y - half * dy], [x + half * dx, y + half * dy]

    def circuit_port_geometry(self, port_config: PortConfig):
        """The port box and copper leads of a circuit port, in internal units.

        Returns ((start, stop), leads) with start/stop as [x, y, z] and each lead an axis-aligned
        (x1, y1, x2, y2) strip on the lead's layer, or (None, []) if the port cannot be built.

        openEMS lumped ports are boxes with one excitation axis, so the + to - line is laid out
        on grid axes:
        - Terminals on different layers: the port drops vertically at the + terminal to the -
          layer; if the - terminal is elsewhere in plane, a copper lead on the - layer runs from
          under the + terminal to it.
        - Same layer: the port runs along the longer of the x/y offsets, level with the + terminal;
          if the terminals are also offset on the other axis, a lead closes that leg to the - terminal.
        """
        if port_config.position is None or port_config.neg_position is None:
            logger.error("Circuit port %s has no - terminal, skipping", port_config.name)
            return None, []
        xp, yp = port_config.position
        xn, yn = port_config.neg_position
        zp = self.get_metal_layer_offset(port_config.layer)
        zn = self.get_metal_layer_offset(port_config.plane)
        half = port_config.width / 2
        dx, dy = xn - xp, yn - yp
        leads: List[Tuple[float, float, float, float]] = []

        def lead(x1: float, y1: float, x2: float, y2: float) -> None:
            if abs(x2 - x1) > half:
                leads.append((min(x1, x2) - half, y1 - half, max(x1, x2) + half, y1 + half))
            if abs(y2 - y1) > half:
                leads.append((x2 - half, min(y1, y2) - half, x2 + half, max(y1, y2) + half))

        if zp != zn:
            if abs(dx) >= abs(dy):
                start, stop = [xp - half, yp, zp], [xp + half, yp, zn]
            else:
                start, stop = [xp, yp - half, zp], [xp, yp + half, zn]
            if math.hypot(dx, dy) > half:
                lead(xp, yp, xn, yn)
            return (start, stop), [(x1, y1, x2, y2) for x1, y1, x2, y2 in leads]
        if max(abs(dx), abs(dy)) < 1:
            logger.error("Circuit port %s has both terminals at the same point on one layer, skipping",
                         port_config.name)
            return None, []
        if abs(dx) >= abs(dy):
            start, stop = [xp, yp - half, zp], [xn, yp + half, zp]
            corner = (xn, yp)
        else:
            start, stop = [xp - half, yp, zp], [xp + half, yn, zp]
            corner = (xp, yn)
        if math.hypot(xn - corner[0], yn - corner[1]) > half:
            lead(corner[0], corner[1], xn, yn)
        return (start, stop), leads

    def add_circuit_port(self, port_config: PortConfig, number: int, excite: bool = False) -> None:
        """ADDED (EMShrimulator Studio): a lumped port between a + and a - terminal. See circuit_port_geometry."""
        box, leads = self.circuit_port_geometry(port_config)
        if box is None:
            return
        start, stop = box
        spans = [abs(stop[i] - start[i]) for i in range(3)]
        axis = "z" if start[2] != stop[2] else ("x" if spans[0] >= spans[1] else "y")
        lead_z = self.get_metal_layer_offset(port_config.plane)
        start, stop = self.snap_to_mesh(start, stop, "xyz".index(axis))
        if leads:
            metal = self.csx.AddMetal(f"PortLead_{number}")
            for x1, y1, x2, y2 in leads:
                p1, p2 = self.snap_to_mesh([x1, y1, lead_z], [x2, y2, lead_z])
                metal.AddBox(p1, p2, priority=90)
        logger.debug("Adding circuit port %d along %s at start: %s end: %s, %d lead(s)",
                     number, axis, start, stop, len(leads))
        port = self.fdtd.AddLumpedPort(
            number,
            port_config.impedance,
            start,
            stop,
            axis,
            excite=1 if excite else 0,
            priority=100,
        )
        self.ports.append(port)

    def add_resistive_port(self, port_config: PortConfig, excite: bool = False, number: Optional[int] = None) -> None:
        """Add a two-terminal lumped port between a signal layer and a plane.

        PATCHED (EMShrimulator Studio): reachable now; any rotation allowed and the box given a real width.
        """
        logger.debug("Adding port number %d", len(self.ports))

        if port_config.position is None or port_config.direction is None:
            logger.error("Port has no defined position or rotation, skipping")
            return

        start_z = self.get_metal_layer_offset(port_config.layer)
        stop_z = self.get_metal_layer_offset(port_config.plane)
        if start_z == stop_z:
            logger.error(
                "Port %s has its signal layer and reference plane on the same copper layer, skipping",
                port_config.name,
            )
            return

        # Lay the width along whichever transverse axis the rotation points
        # most nearly at, and keep the other flat. Taking the *dominant* axis
        # rather than rounding both components keeps the footprint exactly
        # `width` long for every angle -- rounding both would turn anything
        # near 45 degrees into a square that is width*sqrt(2) across.
        (x1, y1), (x2, y2) = self.lumped_port_footprint(port_config)
        start, stop = self.snap_to_mesh([x1, y1, start_z], [x2, y2, stop_z], exc_axis=2)
        logger.debug("Adding resistive port at start: %s end: %s", start, stop)

        # The probe files are named by this number, which must be the port's config index for the
        # ΔS monitor and the post-processing stand-ins to find them.
        port = self.fdtd.AddLumpedPort(
            len(self.ports) if number is None else number,
            port_config.impedance,
            start,
            stop,
            "z",
            excite=1 if excite else 0,
            priority=100,
        )
        self.ports.append(port)

    def add_virtual_port(self, port_config: PortConfig) -> None:
        """Add virtual port for extracting sim data from files. Needed due to OpenEMS api desing.

        PATCHED (EMShrimulator Studio): dispatch on the port type, so stand-ins match the probes written.
        """
        logger.debug("Adding virtual port number %d", len(self.ports))
        for i in range(11):
            self.grid.AddLine("x", i)
            self.grid.AddLine("y", i)
        self.grid.AddLine("z", 0)
        self.grid.AddLine("z", 10)

        if port_kind(port_config) in ("lumped", "circuit"):
            port = self.fdtd.AddLumpedPort(
                len(self.ports),
                port_config.impedance,
                [0, 0, 0],
                [10, 10, 10],
                "z",
                excite=0,
                priority=100,
            )
        else:
            port = self.fdtd.AddMSLPort(
                len(self.ports),
                self.csx.AddMetal(f"VirtualPort_{len(self.ports)}"),
                [0, 0, 0],
                [10, 10, 10],
                "x",
                "z",
                Feed_R=port_config.impedance,
                priority=100,
                excite=0,
            )
        self.ports.append(port)

    def add_plane(self, z_height: float) -> None:
        """Add metal plane in whole bounding box of the PCB."""
        self.plane_material.AddBox(
            [0, 0, z_height],
            [cfg.pcb_width, cfg.pcb_height, z_height],
            priority=1,
        )

    def add_substrates(self) -> None:
        """Add substrate in whole bounding box of the PCB."""
        logger.info("Adding substrates")

        offset = 0
        boxes = []
        for i, layer in enumerate(cfg.get_substrates()):
            self.substrate_materials[i].AddBox(
                [0, 0, offset],
                [
                    cfg.pcb_width,
                    cfg.pcb_height,
                    offset - layer.thickness,
                ],
                priority=-i - 1,
            )
            boxes.append(([0, 0, offset], [cfg.pcb_width, cfg.pcb_height, offset - layer.thickness]))
            logger.debug("Added substrate from %f to %f", offset, offset - layer.thickness)
            offset -= layer.thickness

        # PATCHED (EMShrimulator Studio): Debye poles at half strength on the stack's outer faces. Internal units: 0.1 µm.
        dispersive = sorted(getattr(self, "dispersive_substrates", set()))
        if dispersive:
            from EMShrimulator.core.em.dielectric import face_weight, weight_poles

            touching = [boxes[i] for i in dispersive]
            for i in dispersive:
                weight_poles(self.substrate_materials[i], face_weight([boxes[i]], touching, 0.5))

    def add_vias(self) -> None:
        """Add all vias from excellon file.

        PATCHED (EMShrimulator Studio): the app's padstacks from vias.json when present (blind, buried, backdrilled).
        """
        padstack_vias = importer.get_padstack_vias()
        if padstack_vias is not None:
            logger.info("Adding %d vias from the padstacks", len(padstack_vias))
            for via in padstack_vias:
                self.add_padstack_via(via)
            return
        logger.info("Adding vias from excellon file")
        vias = importer.get_vias()
        for via in vias:
            self.add_via(via[0], via[1], via[2])

    @staticmethod
    def _prism(x_pos: float, y_pos: float, radius: float, reverse: bool = False) -> List[List[float]]:
        order = range(VIA_POLYGON)[::-1] if reverse else range(VIA_POLYGON)
        return [[x_pos + np.sin(i / VIA_POLYGON * 2 * np.pi) * radius for i in order],
                [y_pos + np.cos(i / VIA_POLYGON * 2 * np.pi) * radius for i in order]]

    def add_padstack_via(self, via: dict) -> None:
        """One via of vias.json (mm, µm), in internal units: a plated barrel from metal `top` to metal
        `bottom`, each end moved out by a backdrill's stub; a backdrilled hole is air from the board's face
        to the barrel, over the copper sheets it cut."""
        mm = 1 / 1000 / BASE_UNIT * UNIT_MULTIPLIER
        x_pos, y_pos = via["x_mm"] * mm, via["y_mm"] * mm
        face_top = 0.0
        face_bottom = -sum(layer.thickness for layer in cfg.get_substrates())
        z_top = self.get_metal_layer_offset(int(via["top"]))
        z_bottom = self.get_metal_layer_offset(int(via["bottom"]))
        if via.get("top_stub_mm") is not None:
            z_top = min(z_top + via["top_stub_mm"] * mm, face_top)
        if via.get("bottom_stub_mm") is not None:
            z_bottom = max(z_bottom - via["bottom_stub_mm"] * mm, face_bottom)
        radius = via["hole_mm"] * mm / 2
        plating = via["plating_um"] * mm / 1000
        if z_top - z_bottom > 0:
            fill = self._via_fill(float(via.get("fill_epsilon", 1.0)))
            fill.AddLinPoly(self._prism(x_pos, y_pos, radius), "z", z_bottom, z_top - z_bottom, priority=51)
            self.via_material.AddLinPoly(self._prism(x_pos, y_pos, radius + plating, reverse=True), "z",
                                         z_bottom, z_top - z_bottom, priority=50)
        for drill in via.get("backdrills", []):
            r = drill["diameter_mm"] * mm / 2
            start, stop = (z_top, face_top) if drill["side"] == "top" else (face_bottom, z_bottom)
            if stop - start > 0:
                self._backdrill_material().AddLinPoly(self._prism(x_pos, y_pos, r), "z", start, stop - start,
                                                      priority=49)

    def _via_fill(self, epsilon: float) -> Any:
        if epsilon not in self._via_fills:
            self._via_fills[epsilon] = self.csx.AddMaterial(f"ViaFilling_{len(self._via_fills)}", epsilon=epsilon)
        return self._via_fills[epsilon]

    def _backdrill_material(self) -> Any:
        if self._backdrill is None:
            self._backdrill = self.csx.AddMaterial("Backdrill", epsilon=1.0)
        return self._backdrill

    def add_via(self, x_pos: float, y_pos: float, diameter: float) -> None:
        """Add via at specified position with specified diameter."""
        thickness = sum(layer.thickness for layer in cfg.get_substrates())

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON):
            x_coords.append(x_pos + np.sin(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
            y_coords.append(y_pos + np.cos(i / VIA_POLYGON * 2 * np.pi) * diameter / 2)
        self.via_filling_material.AddLinPoly([x_coords, y_coords], "z", -thickness, thickness, priority=51)

        x_coords = []
        y_coords = []
        for i in range(VIA_POLYGON)[::-1]:
            x_coords.append(x_pos + np.sin(i / VIA_POLYGON * 2 * np.pi) * (diameter / 2 + cfg.via.plating_thickness))
            y_coords.append(y_pos + np.cos(i / VIA_POLYGON * 2 * np.pi) * (diameter / 2 + cfg.via.plating_thickness))
        self.via_material.AddLinPoly([x_coords, y_coords], "z", -thickness, thickness, priority=50)

    def add_single_dump_box(self, name: str, z: float) -> None:
        """Add electric field dump box in whole bounding box of the PCB at specified Z-position."""
        logger.debug("Adding dump box at %f", z)
        dump = self.csx.AddDump(name, sub_sampling=[1, 1, 1])
        start = [
            -cfg.grid.margin.xy,
            -cfg.grid.margin.xy,
            z,
        ]
        stop = [
            cfg.pcb_width + cfg.grid.margin.xy,
            cfg.pcb_height + cfg.grid.margin.xy,
            z,
        ]
        dump.AddBox(start, stop)

    def add_dump_boxes(self) -> None:
        """Add electric field dump box in whole bounding box of the PCB at half the thickness of each substrate."""
        if cfg.arguments.export_field is None:
            return
        logger.info("Adding field dump boxes")

        if len(cfg.arguments.export_field) == 0:
            cfg.arguments.export_field = ["outer", "cu-outer", "cu-inner", "substrate"]

        offset = 0
        metal_idx = 0
        metal_count = len(cfg.get_metals())
        for layer in cfg.layers:
            norm_name = (
                layer.name.replace(".", "_").replace("(", "_").replace(")", "_").replace(" ", "_").replace("/", "_")
            )
            if layer.kind == LayerKind.SUBSTRATE:
                if "substrate" in cfg.arguments.export_field:
                    height = offset - layer.thickness / 2
                    self.add_single_dump_box(f"e_field_{norm_name}", height)
                offset -= layer.thickness
            elif layer.kind == LayerKind.METAL:
                export_inner = "cu-inner" in cfg.arguments.export_field and metal_idx not in [0, metal_count]
                export_outer = "cu-outer" in cfg.arguments.export_field and metal_idx in [0, metal_count]
                if export_inner or export_outer:
                    self.add_single_dump_box(f"e_field_{norm_name}", offset)
                metal_idx += 1

        if "outer" in cfg.arguments.export_field:
            self.add_single_dump_box("e_field_top_over", 100)
            self.add_single_dump_box("e_field_bottom_over", offset - 100)

    def set_boundary_conditions(self, pml: bool = False) -> None:
        """Add boundary conditions. MUR for fast simulation, PML for more accurate."""
        if pml:
            logger.info("Adding perfectly matched layer boundary condition")
            self.fdtd.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"])
        else:
            logger.info("Adding MUR boundary condition")
            self.fdtd.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

    def set_excitation(self) -> None:
        """Set gauss excitation according to config."""
        logger.debug(
            "Setting excitation to gaussian pulse from %f to %f",
            cfg.frequency.start,
            cfg.frequency.stop,
        )
        self.fdtd.SetGaussExcite(
            (cfg.frequency.start + cfg.frequency.stop) / 2,
            (cfg.frequency.stop - cfg.frequency.start) / 2,
        )

    def set_sinus_excitation(self, freq: float) -> None:
        """Set sinus excitation at specified frequency."""
        logger.debug("Setting excitation to sine at %f", freq)
        self.fdtd.SetSinusExcite(freq)

    def run(self, excited_port_number: int) -> None:
        """Execute simulation.

        PATCHED (EMShrimulator Studio): stops early once S-parameters settle when `convergence.delta_s` is set.
        """
        logger.info("Starting simulation")
        cwd = os.getcwd()
        sim_dir = os.path.join(os.getcwd(), SIMULATION_DIR, str(excited_port_number))
        self.fdtd.SetOverSampling(cfg.arguments.oversampling)
        logger.info(
            "Stop criteria: energy %.1f dB, %d timesteps max%s",
            10 * math.log10(max(cfg.convergence.end_criteria, 1e-300)), int(cfg.max_steps),
            f", ΔS < {cfg.convergence.delta_s:g}" if cfg.convergence.delta_s > 0 else " (ΔS check off)",
        )
        monitor = self.delta_s_monitor(sim_dir, excited_port_number) if cfg.convergence.delta_s > 0 else None
        if monitor is not None:
            os.makedirs(sim_dir, exist_ok=True)
            if os.path.exists(os.path.join(sim_dir, "ABORT")):
                os.remove(os.path.join(sim_dir, "ABORT"))
            monitor.start()
        try:
            self.fdtd.Run(sim_dir)
        finally:
            if monitor is not None:
                monitor.cleanup()
                if monitor.converged:
                    logger.info("Run for port #%d stopped on ΔS convergence", excited_port_number)
                elif monitor.history:
                    logger.info("Run for port #%d ended with last max |ΔS| = %.4g (target %g)", excited_port_number,
                                monitor.history[-1][1], cfg.convergence.delta_s)

        os.chdir(cwd)

    def delta_s_monitor(self, sim_dir: str, excited: int) -> DeltaSMonitor:
        """A monitor over every port's probe files, named as add_ports creates them."""
        probes: List[Optional[PortProbes]] = []
        for index, port_config in enumerate(cfg.ports):
            if port_config.position is None:
                probes.append(None)
            elif port_kind(port_config) in ("lumped", "circuit"):
                probes.append(PortProbes([f"port_ut_{index}"], [f"port_it_{index}"], port_config.impedance))
            else:
                probes.append(PortProbes([f"port_ut_{index}B"], [f"port_it_{index}A", f"port_it_{index}B"],
                                         port_config.impedance, current_weight=0.5))
        pulse = pulse_length_s(cfg.frequency.start, cfg.frequency.stop)
        eps = max((layer.epsilon or 1.0 for layer in cfg.get_substrates()), default=1.0)
        # The mesh spans the board plus margins; it is loaded in every step, unlike cfg.pcb_width,
        # which only the geometry step sets.
        to_m = BASE_UNIT / UNIT_MULTIPLIER
        spans = []
        for axis in ("x", "y"):
            lines = self.grid.GetLines(axis)
            spans.append((max(lines) - min(lines)) if len(lines) > 1 else 0.0)
        diagonal_m = math.hypot(*spans) * to_m
        transit = diagonal_m * math.sqrt(eps) / 299_792_458.0
        return DeltaSMonitor(sim_dir, probes, excited, cfg.frequency.start, cfg.frequency.stop,
                             cfg.convergence.delta_s, t_min=3 * pulse + 2 * transit, window=pulse)

    def save_geometry(self) -> None:
        """Save geometry to file."""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Saving geometry to %s", filename)
        self.csx.Write2XML(filename)

        # Replacing , with . for numerals in the file
        # (openEMS bug mitigation for locale that uses , as decimal separator)
        with open(filename, "r") as f:
            content = f.read()
        new_content = re.sub(r"([0-9]+),([0-9]+e)", r"\g<1>.\g<2>", content)
        with open(filename, "w") as f:
            f.write(new_content)

    def load_geometry(self) -> None:
        """Load geometry from file."""
        filename = os.path.join(os.getcwd(), GEOMETRY_DIR, "geometry.xml")
        logger.info("Loading geometry from %s", filename)
        if not os.path.exists(filename):
            logger.error("Geometry file does not exist. Did you run geometry step?")
            sys.exit(1)
        self.csx.ReadFromXML(filename)
        self.grid = self.csx.GetGrid()

    def get_port_parameters(self, exindex: int, frequencies: np.ndarray) -> Tuple[List, List]:
        """Return reflected and incident power vs frequency for each port.

        PATCHED (EMShrimulator Studio): every port is referenced to its configured impedance.
        """
        result_path = os.path.join(os.getcwd(), SIMULATION_DIR, str(exindex))

        incident: List[np.ndarray] = []
        reflected: List[np.ndarray] = []
        for index, port in enumerate(self.ports):
            try:
                port.CalcPort(result_path, frequencies, ref_impedance=cfg.ports[index].impedance)
                logger.debug("Found data for port %d", index)
            except IOError:
                logger.error("Port data files do not exist. Did you run simulation step?")
                sys.exit(1)
            incident.append(port.uf_inc)
            reflected.append(port.uf_ref)

        return (reflected, incident)

    def setup_ports(self, enabled_idx: int) -> None:
        """Set up ports excitation."""
        logger.info("Setting up ports")

        for prop in self.csx.GetAllProperties():
            if not isinstance(prop, CSXCAD.CSProperties.CSPropExcitation):
                continue
            pname = prop.GetName()
            idx = int(pname.removeprefix(pname.rstrip("0123456789")))
            if idx != enabled_idx:
                prop.SetExcitation([0, 0, 0])
            else:
                prop.SetExcitation([0, 0, 1])

    def add_ports(self) -> None:
        """Add ports for simulation.

        PATCHED (EMShrimulator Studio): dispatch on `port_config.type` (microstrip or lumped).
        """
        logger.info("Adding ports")

        self.ports = []

        for index, port_config in enumerate(cfg.ports):
            kind = port_kind(port_config)
            if kind == "circuit":
                logger.info("Port #%d '%s': circuit port, layer %d (+) to layer %d (-)", index, port_config.name,
                            port_config.layer, port_config.plane)
                self.add_circuit_port(port_config, index, excite=True)
            elif kind == "lumped":
                logger.info("Port #%d '%s': gap (lumped) port, layer %d to reference layer %d", index,
                            port_config.name, port_config.layer, port_config.plane)
                self.add_resistive_port(port_config, excite=True, number=index)
            else:
                logger.info("Port #%d '%s': microstrip launch", index, port_config.name)
                self.add_msl_port(port_config, index, True)

    def add_virtual_ports(self) -> None:
        """Add virtual ports needed for data postprocessing due to openEMS api design."""
        logger.info("Adding virtual ports")
        for port_config in cfg.ports:
            self.add_virtual_port(port_config)
