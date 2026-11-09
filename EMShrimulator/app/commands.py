"""Undo commands, one per edit function in core/*/edits.py. Edits to one field within a second merge."""
from __future__ import annotations

import time
import zlib
from typing import Any, Callable, Hashable, List, Optional

from PySide6.QtGui import QUndoCommand

from EMShrimulator.core.em import edits as em
from EMShrimulator.core.em.model import BooleanOp, PortKind, PrimitiveKind, SheetShape
from EMShrimulator.core.layout import edits as lay
from EMShrimulator.core.layout.model import StackupLayer

MERGE_WINDOW_S = 1.0


class EditCommand(QUndoCommand):
    """Snapshot undo around one edit function."""

    def __init__(self, session, text: str, mutate: Callable[[Any], Any],
                 merge_key: Optional[Hashable] = None):
        super().__init__(text)
        self.session = session
        self.before = session.project
        self.after = session.project.model_copy(deep=True)
        self.result = mutate(self.after)
        self.merge_key = merge_key
        self.stamp = time.monotonic()

    def redo(self) -> None:
        self.session._install(self.after)

    def undo(self) -> None:
        self.session._install(self.before)

    def id(self) -> int:  # noqa: A003 - Qt override
        if self.merge_key is None:
            return -1
        return zlib.crc32(repr(self.merge_key).encode()) & 0x7FFFFFFF

    def mergeWith(self, other: QUndoCommand) -> bool:  # noqa: N802 - Qt override
        if not isinstance(other, EditCommand) or other.merge_key != self.merge_key:
            return False
        if other.stamp - self.stamp > MERGE_WINDOW_S:
            return False
        self.after = other.after
        self.result = other.result
        self.stamp = other.stamp
        return True


# --- 3D Modeler ---

class AddPrimitive(EditCommand):
    def __init__(self, session, kind: PrimitiveKind, sheet_shape: Optional[SheetShape] = None):
        label = SheetShape(sheet_shape).value if sheet_shape else em.PRIMITIVE_LABELS[PrimitiveKind(kind)].lower()
        super().__init__(session, f"Add {label}", lambda p: em.add_primitive(p, kind, sheet_shape))


class DrawPrimitive(EditCommand):
    """An object drawn click by click in the 3D view."""

    def __init__(self, session, kind: PrimitiveKind, sheet_shape: Optional[SheetShape], fields: dict):
        label = SheetShape(sheet_shape).value if sheet_shape else em.PRIMITIVE_LABELS[PrimitiveKind(kind)].lower()
        super().__init__(session, f"Draw {label}", lambda p: em.draw_primitive(p, kind, sheet_shape, fields))


class DeletePrimitive(EditCommand):
    def __init__(self, session, primitive_id: str):
        super().__init__(session, "Delete object", lambda p: em.delete_primitive(p, primitive_id))


class DuplicatePrimitive(EditCommand):
    def __init__(self, session, primitive_id: str):
        super().__init__(session, "Duplicate object", lambda p: em.duplicate_primitive(p, primitive_id))


class SetPrimitiveField(EditCommand):
    def __init__(self, session, primitive_id: str, field: str, value: Any, label: str = ""):
        super().__init__(session, f"Edit {label or field.replace('_', ' ')}",
                         lambda p: em.set_primitive_field(p, primitive_id, field, value),
                         merge_key=("primitive", primitive_id, field))


class SetPrimitiveFields(EditCommand):
    """Several fields of one object as a single step - a box's two corners."""

    def __init__(self, session, primitive_id: str, values: dict, label: str):
        def apply(p):
            for field, value in values.items():
                em.set_primitive_field(p, primitive_id, field, value)
        super().__init__(session, f"Edit {label}", apply,
                         merge_key=("primitive", primitive_id, tuple(sorted(values))))


class MakeBoolean(EditCommand):
    def __init__(self, session, op: BooleanOp, operand_ids: List[str]):
        super().__init__(session, f"Create {BooleanOp(op).value}",
                         lambda p: em.make_boolean(p, op, operand_ids))


class ExplodeBoolean(EditCommand):
    def __init__(self, session, primitive_id: str):
        super().__init__(session, "Explode boolean", lambda p: em.explode_boolean(p, primitive_id))


class AddPort(EditCommand):
    def __init__(self, session, kind: PortKind):
        super().__init__(session, f"Add {em.PORT_LABELS[PortKind(kind)].lower()} port",
                         lambda p: em.add_port(p, kind))


class AssignPort(EditCommand):
    def __init__(self, session, sheet_id: str, kind: PortKind):
        super().__init__(session, f"Assign {em.PORT_LABELS[PortKind(kind)].lower()} port",
                         lambda p: em.assign_port(p, sheet_id, kind))


class SetPortLine(EditCommand):
    def __init__(self, session, port_id: str, start, end):
        super().__init__(session, "Set integration line" if start is not None else "Clear integration line",
                         lambda p: em.set_port_line(p, port_id, start, end))


class DeletePort(EditCommand):
    def __init__(self, session, port_id: str):
        super().__init__(session, "Delete port", lambda p: em.delete_port(p, port_id))


class SetPortField(EditCommand):
    def __init__(self, session, port_id: str, field: str, value: Any, label: str = ""):
        super().__init__(session, f"Edit port {label or field.replace('_', ' ')}",
                         lambda p: em.set_port_field(p, port_id, field, value),
                         merge_key=("port", port_id, field))


class SetPortFields(EditCommand):
    def __init__(self, session, port_id: str, values: dict, label: str):
        def apply(p):
            for field, value in values.items():
                em.set_port_field(p, port_id, field, value)
        super().__init__(session, f"Edit port {label}", apply,
                         merge_key=("port", port_id, tuple(sorted(values))))


class DuplicatePort(EditCommand):
    def __init__(self, session, port_id: str):
        super().__init__(session, "Duplicate port", lambda p: em.duplicate_port(p, port_id))


class ExciteOnly(EditCommand):
    def __init__(self, session, port_id: str):
        super().__init__(session, "Excite one port", lambda p: em.excite_only(p, port_id))


class AddCoordinateSystem(EditCommand):
    """A relative system, made the working one."""

    def __init__(self, session, parent: Optional[str] = None, mode: str = "offset", origin=(0.0, 0.0, 0.0)):
        super().__init__(session, "Create relative coordinate system", lambda p: em.add_coordinate_system(
            p, parent, mode, origin, make_working=True))


class AddObjectCoordinateSystem(EditCommand):
    """A system on a point of an object, made the working one."""

    def __init__(self, session, object_id: str, anchor: str, at, mode: str = "offset"):
        super().__init__(session, "Create object coordinate system", lambda p: em.add_object_coordinate_system(
            p, object_id, anchor, at, mode, make_working=True))


class PlaceCoordinateSystem(EditCommand):
    def __init__(self, session, name: str, object_id: str, anchor: str, at):
        super().__init__(session, f"Move {name} to another point",
                         lambda p: em.place_coordinate_system(p, name, object_id, anchor, at))


class DeleteCoordinateSystem(EditCommand):
    def __init__(self, session, name: str):
        super().__init__(session, "Delete coordinate system", lambda p: em.delete_coordinate_system(p, name))


class RenameCoordinateSystem(EditCommand):
    def __init__(self, session, old: str, new: str):
        super().__init__(session, "Rename coordinate system", lambda p: em.rename_coordinate_system(p, old, new))


class SetCoordinateSystemField(EditCommand):
    def __init__(self, session, name: str, field: str, value: Any):
        super().__init__(session, f"Edit coordinate system {field}",
                         lambda p: em.set_coordinate_system_field(p, name, field, value),
                         merge_key=("cs", name, field))


class SetWorkingCS(EditCommand):
    def __init__(self, session, name: Optional[str]):
        super().__init__(session, "Set working coordinate system", lambda p: em.set_working_cs(p, name))


class AddPlot(EditCommand):
    def __init__(self, session, kind: str, traces: Optional[List[str]] = None):
        super().__init__(session, f"New {em.PLOT_LABELS.get(kind, 'plot')}", lambda p: em.add_plot(p, kind, traces))


class DeletePlot(EditCommand):
    def __init__(self, session, plot_id: str):
        super().__init__(session, "Delete plot", lambda p: em.delete_plot(p, plot_id))


class SetPlotField(EditCommand):
    def __init__(self, session, plot_id: str, field: str, value: Any):
        super().__init__(session, f"Edit plot {field.replace('_', ' ')}",
                         lambda p: em.set_plot_field(p, plot_id, field, value),
                         merge_key=("plot", plot_id, field))


class AddMaterial(EditCommand):
    def __init__(self, session):
        super().__init__(session, "Add material", em.add_material)


class SetMaterialField(EditCommand):
    def __init__(self, session, name: str, field: str, value: Any):
        super().__init__(session, f"Edit material {field.replace('_', ' ')}",
                         lambda p: em.set_material_field(p, name, field, value),
                         merge_key=("material", name, field))


class RenameMaterial(EditCommand):
    def __init__(self, session, old: str, new: str):
        super().__init__(session, "Rename material", lambda p: em.rename_material(p, old, new))


class DeleteMaterial(EditCommand):
    def __init__(self, session, name: str):
        super().__init__(session, "Delete material", lambda p: em.delete_material(p, name))


class AddParameter(EditCommand):
    def __init__(self, session):
        super().__init__(session, "Add variable", em.add_parameter)


class DefineParameters(EditCommand):
    def __init__(self, session, definitions: list):
        names = ", ".join(d.name for d in definitions)
        super().__init__(session, f"Define variable{'s' if len(definitions) > 1 else ''} {names}",
                         lambda p: em.define_parameters(p, definitions))


class SetParameterField(EditCommand):
    def __init__(self, session, name: str, field: str, value: Any):
        super().__init__(session, f"Edit variable {field}",
                         lambda p: em.set_parameter_field(p, name, field, value),
                         merge_key=("parameter", name, field))


class RenameParameter(EditCommand):
    def __init__(self, session, old: str, new: str):
        super().__init__(session, f"Rename variable {old}", lambda p: em.rename_parameter(p, old, new))


class DeleteParameter(EditCommand):
    def __init__(self, session, name: str):
        super().__init__(session, f"Delete variable {name}", lambda p: em.delete_parameter(p, name))


class AddSweep(EditCommand):
    def __init__(self, session):
        super().__init__(session, "Add sweep", em.add_sweep)


class EditSweep(EditCommand):
    """Any change inside one sweep, addressed by index."""

    def __init__(self, session, index: int, text: str, mutate: Callable[[Any], None],
                 merge_field: Optional[str] = None):
        super().__init__(session, text, lambda p: mutate(p.sweeps[index]),
                         merge_key=("sweep", index, merge_field) if merge_field else None)


class DeleteSweep(EditCommand):
    def __init__(self, session, index: int):
        super().__init__(session, "Delete sweep", lambda p: p.sweeps.pop(index))


class AddSweepVariable(EditCommand):
    def __init__(self, session, index: int):
        super().__init__(session, "Add swept variable", lambda p: em.add_sweep_variable(p, index))


class AddDump(EditCommand):
    def __init__(self, session):
        super().__init__(session, "Add field dump", em.add_dump)


class DeleteDump(EditCommand):
    def __init__(self, session, dump_id: str):
        super().__init__(session, "Delete field dump", lambda p: em.delete_dump(p, dump_id))


class SetDumpField(EditCommand):
    def __init__(self, session, dump_id: str, field: str, value: Any):
        super().__init__(session, f"Edit dump {field}",
                         lambda p: em.set_dump_field(p, dump_id, field, value),
                         merge_key=("dump", dump_id, field))


class SetDesignSetting(EditCommand):
    """A whole-design setting by dotted path: `mesh.smooth_ratio`, `boundary.xmin`."""

    def __init__(self, session, path: str, value: Any, label: str = ""):
        super().__init__(session, f"Set {label or path}", lambda p: em.set_path(p, path, value),
                         merge_key=("setting", path))


class SetDesignSettings(EditCommand):
    def __init__(self, session, values: dict, label: str):
        def apply(p):
            for path, value in values.items():
                em.set_path(p, path, value)
        super().__init__(session, label, apply)


# --- 3D Layout ---

class PlaceLayoutPort(EditCommand):
    def __init__(self, session, x_mm: float, y_mm: float, layer: int = 0, snap: Optional[str] = None):
        super().__init__(session, "Place port", lambda p: lay.place_port(p, x_mm, y_mm, layer, snap))


class PlaceEdgePort(EditCommand):
    def __init__(self, session, edge, snap: Optional[str] = None):
        super().__init__(session, "Place edge port", lambda p: lay.place_edge_port(
            p, edge.x1, edge.y1, edge.x2, edge.y2, edge.rotation_deg(), edge.layer, snap=snap))


class PlaceCircuitPort(EditCommand):
    def __init__(self, session, pos, neg, snap: Optional[str] = None, neg_snap: Optional[str] = None):
        super().__init__(session, "Place circuit port",
                         lambda p: lay.place_circuit_port(p, pos, neg, snap, neg_snap))


class MoveLayoutPort(EditCommand):
    def __init__(self, session, number: int, x_mm: float, y_mm: float, snap: Optional[str] = None):
        super().__init__(session, "Move port", lambda p: lay.move_port(p, number, x_mm, y_mm, snap),
                         merge_key=("layout_port_move", number))


class MoveLayoutPortTerminal(EditCommand):
    def __init__(self, session, number: int, terminal: str, x_mm: float, y_mm: float,
                 layer: Optional[int] = None, snap: Optional[str] = None):
        super().__init__(session, "Move port terminal",
                         lambda p: lay.move_port_terminal(p, number, terminal, x_mm, y_mm, layer, snap),
                         merge_key=("layout_port_terminal", number, terminal))


class SetLayoutPortField(EditCommand):
    def __init__(self, session, number: int, field: str, value: Any):
        super().__init__(session, f"Edit port {field.replace('_', ' ')}",
                         lambda p: lay.set_port_field(p, number, field, value),
                         merge_key=("layout_port", number, field))


class DeleteLayoutPort(EditCommand):
    def __init__(self, session, number: int):
        super().__init__(session, "Delete port", lambda p: lay.delete_port(p, number))


class RenumberLayoutPorts(EditCommand):
    def __init__(self, session):
        super().__init__(session, "Renumber ports", lay.renumber_ports)


class AddStackupLayer(EditCommand):
    def __init__(self, session, index: Optional[int] = None):
        super().__init__(session, "Add stackup layer", lambda p: lay.add_stackup_layer(p, index))


class SetStackupField(EditCommand):
    def __init__(self, session, index: int, field: str, value: Any):
        super().__init__(session, f"Edit stackup {field}",
                         lambda p: lay.set_stackup_field(p, index, field, value),
                         merge_key=("stackup", index, field))


class DeleteStackupLayer(EditCommand):
    def __init__(self, session, index: int):
        super().__init__(session, "Delete stackup layer", lambda p: lay.delete_stackup_layer(p, index))


class MoveStackupLayer(EditCommand):
    def __init__(self, session, index: int, delta: int):
        super().__init__(session, "Reorder stackup", lambda p: lay.move_stackup_layer(p, index, delta))


class ApplyStackupPreset(EditCommand):
    def __init__(self, session, layers: List[StackupLayer], title: str):
        def apply(p):
            p.stackup = [layer.model_copy(deep=True) for layer in layers]
        super().__init__(session, f"Apply stackup preset '{title}'", apply)


class AddTrace(EditCommand):
    def __init__(self, session):
        super().__init__(session, "Add trace under test", lay.add_trace)


class AddDiffPair(EditCommand):
    def __init__(self, session, positive: Optional[int] = None, negative: Optional[int] = None):
        super().__init__(session, "Add differential pair", lambda p: lay.add_diff_pair(p, positive, negative))


class SetDiffPairField(EditCommand):
    def __init__(self, session, index: int, field: str, value: Any):
        label = {"positive": "+ port", "negative": "− port", "z_diff": "differential impedance",
                 "z_common": "common-mode impedance"}.get(field, field)
        super().__init__(session, f"Edit pair {label}", lambda p: lay.set_diff_pair_field(p, index, field, value),
                         merge_key=("diff_pair", index, field))


class SwapDiffPair(EditCommand):
    def __init__(self, session, index: int):
        super().__init__(session, "Swap pair polarity", lambda p: lay.swap_diff_pair(p, index))


class DeleteDiffPair(EditCommand):
    def __init__(self, session, index: int):
        super().__init__(session, "Delete differential pair", lambda p: lay.delete_diff_pair(p, index))


class EditNet(EditCommand):
    """One field of a trace or differential pair, addressed by list and index."""

    def __init__(self, session, collection: str, index: int, field: Optional[str], value: Any = None):
        def apply(p):
            items = getattr(p, collection)
            if field is None:
                del items[index]
            else:
                setattr(items[index], field, value)
        super().__init__(session, "Delete net" if field is None else f"Edit {field}", apply,
                         merge_key=None if field is None else (collection, index, field))


class EditCopper(EditCommand):
    """Reshape, split or delete copper and move or delete vias as one undo step."""

    def __init__(self, session, label: str, changes, merge_key=None, vias=None):
        def apply(p):
            for layer, replacements in changes.items():
                lay.set_copper(p, layer, replacements)
            if vias:
                lay.move_vias(p, {i: xy for i, xy in vias.items() if xy is not None})
                lay.delete_vias(p, [i for i, xy in vias.items() if xy is None])
            return True
        super().__init__(session, label, apply, merge_key=("copper",) + tuple(merge_key) if merge_key else None)


class PlaceVia(EditCommand):
    def __init__(self, session, x_mm: float, y_mm: float, padstack: str, net: Optional[str] = None):
        super().__init__(session, "Place via", lambda p: lay.add_via(p, x_mm, y_mm, padstack, net))


class SetViasField(EditCommand):
    def __init__(self, session, ids, field: str, value: Any):
        ids = tuple(ids)
        label = {"padstack": "via padstack", "net": "via net", "backdrill_top": "backdrill from top",
                 "backdrill_bottom": "backdrill from bottom"}.get(field, field)
        super().__init__(session, f"Set {label}", lambda p: lay.set_vias_field(p, ids, field, value),
                         merge_key=("vias", ids, field))


class AddPadstack(EditCommand):
    def __init__(self, session, like: Optional[str] = None):
        super().__init__(session, "Duplicate padstack" if like else "Add padstack", lambda p: lay.add_padstack(p, like))


class SetPadstackField(EditCommand):
    def __init__(self, session, name: str, field: str, value: Any):
        label = {"hole_mm": "hole", "pad_mm": "pad", "antipad_mm": "antipad", "plating_um": "plating",
                 "fill_epsilon": "fill εr", "start_layer": "span", "stop_layer": "span"}.get(field, field)
        super().__init__(session, f"Set padstack {label}", lambda p: lay.set_padstack_field(p, name, field, value),
                         merge_key=("padstack", name, field))


class SetPadstackLayer(EditCommand):
    def __init__(self, session, name: str, layer: str, pad_mm: Optional[float], antipad_mm: float,
                 same_as_default: bool = False):
        super().__init__(session, f"Set {layer} pad",
                         lambda p: lay.set_padstack_layer(p, name, layer, pad_mm, antipad_mm, same_as_default),
                         merge_key=("padstack_layer", name, layer))


class DeletePadstack(EditCommand):
    def __init__(self, session, name: str):
        super().__init__(session, "Delete padstack", lambda p: lay.delete_padstack(p, name))


class RevertCopper(EditCommand):
    def __init__(self, session, layer: Optional[str] = None):
        super().__init__(session, "Restore imported copper" + (f" on {layer}" if layer else ""),
                         lambda p: lay.revert_copper(p, layer))


class SetBoardSetting(EditCommand):
    def __init__(self, session, path: str, value: Any, label: str = ""):
        super().__init__(session, f"Set {label or path}", lambda p: lay.set_path(p, path, value),
                         merge_key=("board_setting", path))
