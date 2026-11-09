"""Parameters, safe expression evaluation with unit suffixes, and RF design helpers (mm, Hz, deg)."""
from __future__ import annotations

import ast
import math
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

C0 = 299_792_458.0            # m/s
MU0 = 4e-7 * math.pi          # H/m
EPS0 = 1.0 / (MU0 * C0 * C0)  # F/m


class ExpressionError(ValueError):
    """A parameter or field expression that could not be evaluated."""


UNIT_FACTORS: Dict[str, float] = {
    # length -> mm
    "m": 1000.0, "cm": 10.0, "mm": 1.0, "um": 1e-3, "\u00b5m": 1e-3, "nm": 1e-6,
    "mil": 0.0254, "thou": 0.0254, "in": 25.4, "inch": 25.4, "ft": 304.8,
    # frequency -> Hz
    "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12,
    # time -> s
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "\u00b5s": 1e-6, "ns": 1e-9, "ps": 1e-12,
    # angle -> degrees
    "deg": 1.0, "rad": 180.0 / math.pi,
    # bare markers
    "ohm": 1.0, "\u03a9": 1.0,
}

# Longest first, so `mm` wins over `m` and `mhz` over `m`.
_UNIT_ALTERNATION = "|".join(
    re.escape(u) for u in sorted(UNIT_FACTORS, key=len, reverse=True)
)
_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*(" + _UNIT_ALTERNATION + r")(?![\w])",
    re.IGNORECASE,
)


def expand_units(text: str) -> str:
    """Rewrite `40mil` as `(40*0.0254)` before parsing."""

    def swap(match: re.Match) -> str:
        number, unit = match.group(1), match.group(2)
        factor = UNIT_FACTORS[unit.lower()]
        return f"({number}*{factor!r})"

    return _UNIT_RE.sub(swap, text)


LENGTH_UNITS: Dict[str, float] = {
    "mm": 1.0, "um": 1e-3, "cm": 10.0, "m": 1000.0, "mil": 0.0254, "in": 25.4,
}
FREQUENCY_UNITS: Dict[str, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


# RF design helpers (all lengths in mm)

def wavelength(frequency_hz: float, epsilon_r: float = 1.0) -> float:
    """Wavelength in mm at `frequency_hz` inside a medium of relative epsilon_r."""
    if frequency_hz <= 0:
        raise ExpressionError("wavelength() needs a positive frequency")
    if epsilon_r <= 0:
        raise ExpressionError("wavelength() needs a positive epsilon_r")
    return 1000.0 * C0 / (frequency_hz * math.sqrt(epsilon_r))


def quarter_wave(frequency_hz: float, epsilon_r: float = 1.0) -> float:
    """Quarter wavelength in mm."""
    return wavelength(frequency_hz, epsilon_r) / 4.0


def microstrip_eeff(width_mm: float, epsilon_r: float, height_mm: float) -> float:
    """Effective permittivity of a microstrip (Hammerstad, thin-conductor form)."""
    if height_mm <= 0 or width_mm <= 0:
        raise ExpressionError("microstrip_eeff() needs a positive width and height")
    u = width_mm / height_mm
    a = (epsilon_r + 1) / 2
    b = (epsilon_r - 1) / 2
    if u >= 1:
        return a + b * (1 + 12 / u) ** -0.5
    return a + b * ((1 + 12 / u) ** -0.5 + 0.04 * (1 - u) ** 2)


def microstrip_z0(width_mm: float, epsilon_r: float, height_mm: float) -> float:
    """Characteristic impedance in ohms of a microstrip (Hammerstad)."""
    if height_mm <= 0 or width_mm <= 0:
        raise ExpressionError("microstrip_z0() needs a positive width and height")
    u = width_mm / height_mm
    e_eff = microstrip_eeff(width_mm, epsilon_r, height_mm)
    if u <= 1:
        return (60 / math.sqrt(e_eff)) * math.log(8 / u + u / 4)
    return (120 * math.pi / math.sqrt(e_eff)) / (u + 1.393 + 0.667 * math.log(u + 1.444))


def microstrip_width(z0_ohm: float, epsilon_r: float, height_mm: float) -> float:
    """Microstrip width in mm for `z0_ohm` (Wheeler synthesis)."""
    if height_mm <= 0:
        raise ExpressionError("microstrip_width() needs a positive height")
    if z0_ohm <= 0:
        raise ExpressionError("microstrip_width() needs a positive impedance")
    a = z0_ohm / 60 * math.sqrt((epsilon_r + 1) / 2) + (epsilon_r - 1) / (epsilon_r + 1) * (
        0.23 + 0.11 / epsilon_r
    )
    b = 377 * math.pi / (2 * z0_ohm * math.sqrt(epsilon_r))
    # Wheeler gives two branches; the wide one is valid for W/h > 2.
    u_narrow = 8 * math.exp(a) / (math.exp(2 * a) - 2)
    u_wide = (2 / math.pi) * (
        b - 1 - math.log(2 * b - 1)
        + (epsilon_r - 1) / (2 * epsilon_r) * (math.log(b - 1) + 0.39 - 0.61 / epsilon_r)
    )
    u = u_narrow if u_narrow < 2 else u_wide
    return max(u, 1e-6) * height_mm


def guide_wavelength(frequency_hz: float, width_mm: float, epsilon_r: float, height_mm: float) -> float:
    """Wavelength in mm along a microstrip of the given cross-section."""
    return wavelength(frequency_hz, microstrip_eeff(width_mm, epsilon_r, height_mm))


def patch_width(frequency_hz: float, epsilon_r: float) -> float:
    """Rectangular patch width in mm for best radiation efficiency (Balanis 14-6)."""
    if frequency_hz <= 0:
        raise ExpressionError("patch_width() needs a positive frequency")
    return 1000.0 * (C0 / (2 * frequency_hz)) * math.sqrt(2 / (epsilon_r + 1))


def patch_eeff(frequency_hz: float, epsilon_r: float, height_mm: float) -> float:
    """Effective permittivity seen by a patch of the width `patch_width` gives."""
    w = patch_width(frequency_hz, epsilon_r)
    return microstrip_eeff(w, epsilon_r, height_mm)


def patch_length(frequency_hz: float, epsilon_r: float, height_mm: float) -> float:
    """Resonant patch length in mm, fringing-shortened (Balanis 14-7 with dL)."""
    w = patch_width(frequency_hz, epsilon_r)
    e_eff = microstrip_eeff(w, epsilon_r, height_mm)
    d_l = 0.412 * height_mm * ((e_eff + 0.3) * (w / height_mm + 0.264)) / (
        (e_eff - 0.258) * (w / height_mm + 0.8)
    )
    return 1000.0 * C0 / (2 * frequency_hz * math.sqrt(e_eff)) - 2 * d_l


def patch_inset(frequency_hz: float, epsilon_r: float, height_mm: float, z0_ohm: float = 50.0) -> float:
    """Inset-feed depth in mm matching the patch edge to `z0_ohm` (cos^2 model)."""
    length = patch_length(frequency_hz, epsilon_r, height_mm)
    width = patch_width(frequency_hz, epsilon_r)
    denom = epsilon_r - 1 if epsilon_r > 1.05 else 0.05
    r_edge = 90.0 * (epsilon_r ** 2) / denom * (length / width) ** 2
    r_edge = max(r_edge, z0_ohm * 1.0001)
    return length / math.pi * math.acos(math.sqrt(z0_ohm / r_edge))


def wilkinson_z(z0_ohm: float = 50.0) -> float:
    """Branch impedance of an equal-split Wilkinson divider: Z0 * sqrt(2)."""
    return z0_ohm * math.sqrt(2.0)


def coax_z0(r_inner_mm: float, r_outer_mm: float, epsilon_r: float = 1.0) -> float:
    """Characteristic impedance in ohms of a coaxial line."""
    if r_inner_mm <= 0 or r_outer_mm <= r_inner_mm:
        raise ExpressionError("coax_z0() needs 0 < r_inner < r_outer")
    return 59.9585 / math.sqrt(epsilon_r) * math.log(r_outer_mm / r_inner_mm)


def wg_cutoff(width_mm: float, epsilon_r: float = 1.0) -> float:
    """TE10 cutoff frequency in Hz of a rectangular waveguide of that broad wall."""
    if width_mm <= 0:
        raise ExpressionError("wg_cutoff() needs a positive width")
    return C0 / (2 * (width_mm / 1000.0) * math.sqrt(epsilon_r))


def db(linear: float) -> float:
    """20*log10."""
    if linear <= 0:
        raise ExpressionError("db() needs a positive value")
    return 20 * math.log10(linear)


def undb(decibels: float) -> float:
    """The inverse of db()."""
    return 10 ** (decibels / 20)


DESIGN_FUNCTIONS = {
    "wavelength": wavelength,
    "quarter_wave": quarter_wave,
    "microstrip_width": microstrip_width,
    "microstrip_z0": microstrip_z0,
    "microstrip_eeff": microstrip_eeff,
    "guide_wavelength": guide_wavelength,
    "patch_width": patch_width,
    "patch_length": patch_length,
    "patch_eeff": patch_eeff,
    "patch_inset": patch_inset,
    "wilkinson_z": wilkinson_z,
    "coax_z0": coax_z0,
    "wg_cutoff": wg_cutoff,
    "db": db,
    "undb": undb,
}

# Trig works in degrees, like every angle field.
MATH_FUNCTIONS = {
    "sin": lambda d: math.sin(math.radians(d)),
    "cos": lambda d: math.cos(math.radians(d)),
    "tan": lambda d: math.tan(math.radians(d)),
    "asin": lambda v: math.degrees(math.asin(v)),
    "acos": lambda v: math.degrees(math.acos(v)),
    "atan": lambda v: math.degrees(math.atan(v)),
    "atan2": lambda y, x: math.degrees(math.atan2(y, x)),
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "min": min,
    "max": max,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "hypot": math.hypot,
    "sign": lambda v: (v > 0) - (v < 0),
    "clamp": lambda v, lo, hi: max(lo, min(hi, v)),
}

FUNCTIONS: Dict[str, Any] = {**MATH_FUNCTIONS, **DESIGN_FUNCTIONS}

CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "c0": C0,
    "eps0": EPS0,
    "mu0": MU0,
    "eta0": MU0 * C0,
}

FUNCTION_HELP: Dict[str, str] = {
    "wavelength": "wavelength(f_Hz, er=1) -> mm",
    "quarter_wave": "quarter_wave(f_Hz, er=1) -> mm",
    "microstrip_width": "microstrip_width(Z0, er, h_mm) -> strip width in mm",
    "microstrip_z0": "microstrip_z0(w_mm, er, h_mm) -> ohms",
    "microstrip_eeff": "microstrip_eeff(w_mm, er, h_mm) -> effective er",
    "guide_wavelength": "guide_wavelength(f_Hz, w_mm, er, h_mm) -> mm along the line",
    "patch_width": "patch_width(f_Hz, er) -> mm",
    "patch_length": "patch_length(f_Hz, er, h_mm) -> resonant length in mm",
    "patch_eeff": "patch_eeff(f_Hz, er, h_mm) -> effective er of that patch",
    "patch_inset": "patch_inset(f_Hz, er, h_mm, Z0=50) -> inset feed depth in mm",
    "wilkinson_z": "wilkinson_z(Z0=50) -> branch impedance, Z0*sqrt(2)",
    "coax_z0": "coax_z0(r_in_mm, r_out_mm, er=1) -> ohms",
    "wg_cutoff": "wg_cutoff(a_mm, er=1) -> TE10 cutoff in Hz",
    "db": "db(ratio) -> dB",
    "undb": "undb(dB) -> ratio",
}


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.IfExp, ast.Compare, ast.BoolOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or,
)


@lru_cache(maxsize=8192)
def _checked_tree(original: str) -> ast.Expression:
    """The parsed, whitelisted tree of an expression (cached and shared)."""
    text = expand_units(original)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"{exc.msg} in '{original}'") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"'{type(node).__name__}' is not allowed in an expression - "
                "use numbers, variables, arithmetic and the built-in functions"
            )
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ExpressionError("only plain function calls are allowed")
    return tree


def evaluate(expression: str, variables: Mapping[str, float]) -> float:
    """Evaluate an expression; only arithmetic, comparisons and whitelisted functions are allowed."""
    original = str(expression).strip()
    if not original:
        raise ExpressionError("empty expression")
    tree = _checked_tree(original)

    value = _eval_node(tree.body, variables)
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ExpressionError(f"'{original}' did not evaluate to a finite number")
    return float(value)


def _eval_node(node: ast.AST, variables: Mapping[str, float]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ExpressionError(f"{node.value!r} is not a number")

    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        known = sorted(set(variables) | set(CONSTANTS))
        hint = f" Known names: {', '.join(known)}." if known else ""
        raise ExpressionError(f"unknown name '{node.id}'.{hint}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        return not operand

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                if abs(right) > 64:
                    raise ExpressionError("exponent is too large")
                return left ** right
        except ZeroDivisionError as exc:
            raise ExpressionError("division by zero") from exc
        except OverflowError as exc:
            raise ExpressionError("arithmetic overflow") from exc

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, variables)
            ok = (
                left == right if isinstance(op, ast.Eq)
                else left != right if isinstance(op, ast.NotEq)
                else left < right if isinstance(op, ast.Lt)
                else left <= right if isinstance(op, ast.LtE)
                else left > right if isinstance(op, ast.Gt)
                else left >= right
            )
            if not ok:
                return False
            left = right
        return True

    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, variables) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    if isinstance(node, ast.IfExp):
        return _eval_node(node.body if _eval_node(node.test, variables) else node.orelse, variables)

    if isinstance(node, ast.Call):
        name = node.func.id  # type: ignore[union-attr] - guarded in evaluate()
        func = FUNCTIONS.get(name)
        if func is None:
            raise ExpressionError(
                f"unknown function '{name}'. Available: {', '.join(sorted(FUNCTIONS))}"
            )
        if node.keywords:
            raise ExpressionError(f"'{name}' takes positional arguments only")
        args = [_eval_node(a, variables) for a in node.args]
        try:
            return func(*args)
        except ExpressionError:
            raise
        except TypeError as exc:
            help_text = FUNCTION_HELP.get(name, "")
            raise ExpressionError(f"'{name}' called wrongly: {exc}. {help_text}".strip()) from exc
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ExpressionError(f"'{name}' failed: {exc}") from exc

    raise ExpressionError(f"'{type(node).__name__}' is not allowed in an expression")


def check_syntax(expression: str) -> None:
    """Raise ExpressionError when an expression would not parse, whatever names it reads."""
    original = str(expression).strip()
    if not original:
        raise ExpressionError("empty expression")
    _checked_tree(original)


def referenced_names(expression: str) -> List[str]:
    """Every bare name an expression reads, in reading order, for dependency ordering."""
    return list(_referenced_names(str(expression)))


@lru_cache(maxsize=8192)
def _referenced_names(expression: str) -> Tuple[str, ...]:
    try:
        tree = ast.parse(expand_units(expression), mode="eval")
    except SyntaxError:
        return ()
    names = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id not in FUNCTIONS),
                   key=lambda node: node.col_offset)
    return tuple(dict.fromkeys(node.id for node in names))


def undefined_names(expression: str, known: Iterable[str]) -> List[str]:
    """The names an expression reads that are neither in `known` nor a constant, in reading order."""
    known = set(known)
    return [n for n in referenced_names(expression) if n not in known and n not in CONSTANTS]


def solve_parameters(
    parameters: Iterable[Any],
    overrides: Mapping[str, float] | None = None,
) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
    """Evaluate parameters in dependency order. Returns (values, errors); cycles become errors."""
    params = list(parameters)
    overrides = dict(overrides or {})
    by_name = {p.name: p for p in params}

    values: Dict[str, float] = {}
    errors: List[Dict[str, str]] = []

    for name, value in overrides.items():
        if name in by_name:
            values[name] = float(value)

    pending = [p for p in params if p.name not in values]
    known_names = set(by_name)

    while pending:
        progressed = False
        blocked: List[Any] = []
        for param in pending:
            deps = [d for d in referenced_names(param.expression) if d in known_names]
            if any(d not in values for d in deps):
                blocked.append(param)
                continue
            try:
                values[param.name] = evaluate(param.expression, values)
            except ExpressionError as exc:
                errors.append({"target": param.name, "message": str(exc)})
            progressed = True
        if not progressed:
            for param in blocked:
                errors.append({
                    "target": param.name,
                    "message": "circular reference - this parameter depends on itself, "
                               "directly or through another parameter",
                })
            break
        pending = blocked

    return values, errors


def resolve(project: Any, overrides: Mapping[str, float] | None = None) -> Tuple[Any, Dict[str, float], List[Dict[str, str]]]:
    """Copy of `project` with expressions evaluated and coordinate systems placed. Failed fields become 0."""
    from EMShrimulator.core.em.model import EXPR_FIELDS  # local: model imports nothing from here

    resolved = project.model_copy(deep=True)
    values, errors = solve_parameters(getattr(resolved, "parameters", []), overrides)
    _resolve_node(resolved, EXPR_FIELDS, values, errors, "", "")
    if getattr(resolved, "coordinate_systems", None):
        from EMShrimulator.core.em import frames

        frames.place(resolved)
    if getattr(getattr(resolved, "boundary", None), "open_region", False):
        from EMShrimulator.core.em import region

        region.apply(resolved)
    return resolved, values, errors


def _resolve_node(
    node: Any,
    expr_fields: Mapping[type, Any],
    values: Mapping[str, float],
    errors: List[Dict[str, str]],
    path: str,
    label: str,
) -> None:
    """Walk a pydantic model in place, evaluating its expression fields."""
    from pydantic import BaseModel

    if not isinstance(node, BaseModel):
        return

    fields = expr_fields.get(type(node), frozenset())
    own_label = getattr(node, "name", None) or label

    for name in type(node).model_fields:
        current = getattr(node, name, None)
        if current is None:
            continue
        child_path = f"{path}{name}" if not path else f"{path}.{name}"

        if name in fields:
            setattr(node, name, _resolve_value(
                current, values, errors, child_path, f"{own_label} · {name}" if own_label else name))
            continue

        if isinstance(current, BaseModel):
            _resolve_node(current, expr_fields, values, errors, child_path, own_label)
        elif isinstance(current, (list, tuple)):
            for i, item in enumerate(current):
                _resolve_node(item, expr_fields, values, errors, f"{child_path}.{i}", own_label)
        elif isinstance(current, dict):
            for key, item in current.items():
                _resolve_node(item, expr_fields, values, errors, f"{child_path}.{key}", own_label)


_AXIS_LABELS = ("X", "Y", "Z")


def _resolve_value(
    value: Any,
    values: Mapping[str, float],
    errors: List[Dict[str, str]],
    path: str,
    label: str,
) -> Any:
    """Evaluate one expression field, which may be a scalar, tuple or list."""
    if isinstance(value, str):
        try:
            return evaluate(value, values)
        except ExpressionError as exc:
            errors.append({"target": path, "label": label, "message": str(exc)})
            return 0.0
    if isinstance(value, tuple):
        return tuple(
            _resolve_value(v, values, errors, f"{path}.{i}",
                           f"{label}.{_AXIS_LABELS[i]}" if i < 3 and len(value) <= 3 else f"{label}[{i}]")
            for i, v in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _resolve_value(v, values, errors, f"{path}.{i}", f"{label}[{i}]")
            for i, v in enumerate(value)
        ]
    return value


def is_expression(value: Any) -> bool:
    """True when a field is carrying a formula rather than a plain number."""
    return isinstance(value, str)


def collect_expressions(node: Any, expr_fields: Mapping[type, Any] | None = None) -> Dict[str, str]:
    """Every expression string in a project, keyed by field path."""
    from EMShrimulator.core.em.model import EXPR_FIELDS
    from pydantic import BaseModel

    expr_fields = expr_fields or EXPR_FIELDS
    out: Dict[str, str] = {}

    def walk(current: Any, path: str) -> None:
        if isinstance(current, BaseModel):
            fields = expr_fields.get(type(current), frozenset())
            for name in type(current).model_fields:
                value = getattr(current, name, None)
                child = f"{path}.{name}" if path else name
                if name in fields:
                    _collect_leaf(value, child, out)
                else:
                    walk(value, child)
        elif isinstance(current, (list, tuple)):
            for i, item in enumerate(current):
                walk(item, f"{path}.{i}")

    walk(node, "")
    return out


def _collect_leaf(value: Any, path: str, out: Dict[str, str]) -> None:
    if isinstance(value, str):
        out[path] = value
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _collect_leaf(item, f"{path}.{i}", out)
