"""Identify exported Gerber/drill files and copy them into fab/ as `<board>-<KiCad layer>.gbr`.

Layer identity comes from the X2 `%TF.FileFunction` attribute when present, else the filename.
"""
from __future__ import annotations

import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Files that are unmistakably not board data, however they are named.
_IGNORED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".pdf", ".doc", ".docx",
    ".xls", ".xlsx", ".md", ".html", ".htm", ".json", ".ipc", ".rep", ".log",
    ".ods", ".odt", ".step", ".stp", ".dxf", ".zip", ".rar", ".7z",
}
_DRILL_SUFFIXES = {".drl", ".xln", ".txt", ".nc", ".tap", ".exc", ".drd", ".ncd"}
_GERBER_SUFFIXES = {
    ".gbr", ".ger", ".gbx", ".art", ".pho", ".gdo",
    ".gtl", ".gbl", ".gts", ".gbs", ".gto", ".gbo", ".gtp", ".gbp",
    ".gko", ".gm1", ".gm2", ".gml", ".gpt", ".gpb", ".gd1", ".gg1",
}
_INNER_EXT_RE = re.compile(r"^\.g(?:p?)(\d{1,2})$", re.IGNORECASE)

_FILE_FUNCTION_RE = re.compile(r"%TF\.FileFunction,([^*%]+)", re.IGNORECASE)


@dataclass
class ImportedFile:
    """One input file and what we concluded it was."""

    source: str                  # original filename as uploaded
    role: str                    # 'copper' | 'outline' | 'drill' | 'mask' | 'silk' | 'paste' | 'unknown'
    layer: Optional[str] = None  # canonical KiCad layer name for copper/outline
    order: Optional[int] = None  # 1-based stack position for copper
    target: Optional[str] = None # filename written into fab/
    evidence: str = ""           # how we decided, shown in the import report


@dataclass
class ImportReport:
    board_name: str
    files: List[ImportedFile] = field(default_factory=list)
    gerber_files: List[str] = field(default_factory=list)
    drill_file: Optional[str] = None
    copper_layers: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "board_name": self.board_name,
            "gerber_files": self.gerber_files,
            "drill_file": self.drill_file,
            "copper_layers": self.copper_layers,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "files": [
                {"source": f.source, "role": f.role, "layer": f.layer,
                 "target": f.target, "evidence": f.evidence}
                for f in self.files
            ],
        }


class GerberImportError(ValueError):
    """An archive or file set that could not be interpreted as a board."""


# --- archive handling ---

def is_archive(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        return zipfile.is_zipfile(path)
    if suffix in (".tgz", ".tar", ".gz", ".bz2", ".xz"):
        return tarfile.is_tarfile(path)
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def is_archive_bytes(payload: bytes, filename: str = "") -> bool:
    """Archive detection by magic number, falling back to the extension."""
    if payload[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return True
    if payload[:2] == b"\x1f\x8b":          # gzip, so .tar.gz / .tgz
        return True
    if payload[:3] == b"BZh" or payload[:6] == b"\xfd7zXZ\x00":
        return True
    if len(payload) > 262 and payload[257:262] == b"ustar":
        return True
    return Path(filename).suffix.lower() in (".zip", ".tgz", ".tar")


def safe_board_name(name: str) -> str:
    """Filename-safe board prefix; never empty and never ending in '-' (gerber2ems splits on it)."""
    cleaned = re.sub(r"[^A-Za-z0-9_.]+", "_", name).strip("_-.")
    return cleaned or "board"


def extract_archive(archive: Path, work_dir: Path) -> List[Path]:
    """Unpack an archive flat into `work_dir` and return its files."""
    work_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                _extract_member(info.filename, work_dir, lambda: zf.read(info))
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                stream = tf.extractfile(member)
                if stream is None:
                    continue
                _extract_member(member.name, work_dir, stream.read)
    else:
        raise GerberImportError(
            f"'{archive.name}' is not a zip or tar archive. Upload a .zip of the Gerber "
            "export, or the individual .gbr and .drl files."
        )

    files = sorted(p for p in work_dir.rglob("*") if p.is_file())
    if not files:
        raise GerberImportError(f"'{archive.name}' is empty.")
    return files


def _extract_member(name: str, work_dir: Path, read) -> None:
    leaf = Path(name.replace("\\", "/")).name
    if not leaf or leaf.startswith("."):
        return
    dest = work_dir / leaf
    counter = 1
    while dest.exists():
        dest = work_dir / f"{Path(leaf).stem}_{counter}{Path(leaf).suffix}"
        counter += 1
    dest.write_bytes(read())


# --- identification ---

def read_file_function(path: Path) -> Optional[str]:
    """The Gerber X2 `.FileFunction` attribute, if the file carries one."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(16384)
    except OSError:
        return None
    match = _FILE_FUNCTION_RE.search(head)
    return match.group(1).strip() if match else None


def _from_file_function(value: str) -> Tuple[str, Optional[str], Optional[int], str]:
    """Map an X2 FileFunction to (role, layer, stack order, evidence)."""
    parts = [p.strip() for p in value.split(",")]
    kind = parts[0].lower() if parts else ""
    evidence = f"Gerber X2 attribute: {value}"

    if kind == "copper":
        order = None
        position = ""
        for part in parts[1:]:
            m = re.fullmatch(r"L(\d+)", part, re.IGNORECASE)
            if m:
                order = int(m.group(1))
            elif part.lower() in ("top", "bot", "bottom", "inr", "inner"):
                position = part.lower()
        return "copper", position or None, order, evidence
    if kind in ("profile", "outline", "edge"):
        return "outline", "Edge_Cuts", None, evidence
    if kind in ("plated", "nonplated", "mixedplating"):
        return "drill", None, None, evidence
    if kind == "soldermask":
        return "mask", None, None, evidence
    if kind == "legend":
        return "silk", None, None, evidence
    if kind == "paste":
        return "paste", None, None, evidence
    if kind in ("drillmap", "other", "component", "assemblydrawing", "fabricationdrawing"):
        return "unknown", None, None, evidence
    return "unknown", None, None, evidence


_NAME_COPPER_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:^|[-_. ])f[._]cu(?:$|[-_. ])", re.I), "F_Cu"),
    (re.compile(r"(?:^|[-_. ])b[._]cu(?:$|[-_. ])", re.I), "B_Cu"),
    (re.compile(r"(?:^|[-_. ])top[._ ]?(?:copper|layer)?(?:$|[-_. ])", re.I), "F_Cu"),
    (re.compile(r"(?:^|[-_. ])bot(?:tom)?[._ ]?(?:copper|layer)?(?:$|[-_. ])", re.I), "B_Cu"),
    (re.compile(r"(?:^|[-_. ])copper[._ ]?top(?:$|[-_. ])", re.I), "F_Cu"),
    (re.compile(r"(?:^|[-_. ])copper[._ ]?bot(?:tom)?(?:$|[-_. ])", re.I), "B_Cu"),
]
_NAME_INNER_RE = re.compile(r"(?:^|[-_. ])(?:in|inner|internal|signal|layer)[._ ]?(\d{1,2})(?:[._]cu)?(?:$|[-_. ])", re.I)
_NAME_OUTLINE_RE = re.compile(
    r"(?:^|[-_. ])(?:edge[._ ]?cuts?|outline|board[._ ]?outline|profile|contour|"
    r"mechanical[._ ]?1?|boardoutline|dimension)(?:$|[-_. ])", re.I)
_NAME_MASK_RE = re.compile(r"(?:^|[-_. ])(?:[fb][._]mask|solder[._ ]?mask|mask)(?:$|[-_. ])", re.I)
_NAME_SILK_RE = re.compile(r"(?:^|[-_. ])(?:[fb][._]silk\w*|silk\w*|legend|overlay)(?:$|[-_. ])", re.I)
_NAME_PASTE_RE = re.compile(r"(?:^|[-_. ])(?:[fb][._]paste|paste)(?:$|[-_. ])", re.I)
_NAME_DRILL_RE = re.compile(r"(?:^|[-_. ])(?:pth|npth|drill|drl|plated|excellon|holes?)(?:$|[-_. ])", re.I)

_EXTENSION_ROLES: Dict[str, Tuple[str, Optional[str]]] = {
    ".gtl": ("copper", "F_Cu"),
    ".gbl": ("copper", "B_Cu"),
    ".gko": ("outline", "Edge_Cuts"),
    ".gm1": ("outline", "Edge_Cuts"),
    ".gm2": ("outline", "Edge_Cuts"),
    ".gml": ("outline", "Edge_Cuts"),
    ".gts": ("mask", None),
    ".gbs": ("mask", None),
    ".gto": ("silk", None),
    ".gbo": ("silk", None),
    ".gtp": ("paste", None),
    ".gbp": ("paste", None),
}


def identify(path: Path) -> ImportedFile:
    """Work out what one file is, by content first and name second."""
    name = path.name
    stem = path.stem
    suffix = path.suffix.lower()

    if suffix in _IGNORED_SUFFIXES:
        return ImportedFile(name, "unknown", evidence=f"'{suffix}' is not board data")

    # Drill files are Excellon, not Gerber, and carry no FileFunction.
    looks_like_drill = suffix in _DRILL_SUFFIXES and _looks_like_excellon(path)

    function = None if looks_like_drill else read_file_function(path)
    if function:
        role, layer, order, evidence = _from_file_function(function)
        if role == "copper":
            return ImportedFile(name, "copper", layer=layer, order=order, evidence=evidence)
        return ImportedFile(name, role, layer=layer, evidence=evidence)

    if looks_like_drill or (suffix in _DRILL_SUFFIXES and _NAME_DRILL_RE.search(stem)):
        return ImportedFile(name, "drill", evidence="Excellon drill data")

    for pattern, layer in _NAME_COPPER_PATTERNS:
        if pattern.search(stem) or pattern.search(f"-{stem}-"):
            return ImportedFile(name, "copper", layer=layer, evidence=f"filename matches '{layer}'")

    inner = _NAME_INNER_RE.search(stem)
    if inner:
        index = int(inner.group(1))
        return ImportedFile(name, "copper", layer=f"In{index}_Cu", order=index + 1,
                            evidence=f"filename names inner layer {index}")

    if _NAME_OUTLINE_RE.search(stem):
        return ImportedFile(name, "outline", layer="Edge_Cuts", evidence="filename names the board outline")
    if _NAME_MASK_RE.search(stem):
        return ImportedFile(name, "mask", evidence="filename names a solder mask")
    if _NAME_SILK_RE.search(stem):
        return ImportedFile(name, "silk", evidence="filename names silkscreen")
    if _NAME_PASTE_RE.search(stem):
        return ImportedFile(name, "paste", evidence="filename names solder paste")

    if suffix in _EXTENSION_ROLES:
        role, layer = _EXTENSION_ROLES[suffix]
        return ImportedFile(name, role, layer=layer, evidence=f"'{suffix}' is the standard extension for this layer")

    inner_ext = _INNER_EXT_RE.fullmatch(suffix)
    if inner_ext:
        index = int(inner_ext.group(1))
        return ImportedFile(name, "copper", layer=f"In{index}_Cu", order=index + 1,
                            evidence=f"'{suffix}' is inner layer {index}")

    if suffix in _GERBER_SUFFIXES:
        return ImportedFile(name, "unknown", evidence="Gerber data, but its layer could not be identified")

    return ImportedFile(name, "unknown", evidence="not recognised as Gerber or drill data")


def _looks_like_excellon(path: Path) -> bool:
    """Excellon starts with an M48 header, or at least uses tool selects."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    if "%TF." in head or head.lstrip().startswith("G04"):
        return False
    return bool(re.search(r"^\s*M48\b", head, re.MULTILINE)
                or re.search(r"^\s*T\d+C[\d.]+", head, re.MULTILINE)
                or "FMAT" in head or "EXCELLON" in head.upper())


# --- normalisation ---

def _assign_copper_names(copper: List[ImportedFile]) -> List[ImportedFile]:
    """Assign F_Cu / In<n>_Cu / B_Cu top to bottom, using X2 stack order when complete."""
    if not copper:
        return []

    with_order = [f for f in copper if f.order is not None]
    if len(with_order) == len(copper) and len({f.order for f in copper}) == len(copper):
        ordered = sorted(copper, key=lambda f: f.order or 0)
    else:
        def key(f: ImportedFile) -> Tuple[int, int]:
            layer = (f.layer or "").lower()
            if layer in ("f_cu", "top"):
                return (0, 0)
            if layer in ("b_cu", "bot", "bottom"):
                return (2, 0)
            m = re.fullmatch(r"in(\d+)_cu", layer)
            if m:
                return (1, int(m.group(1)))
            return (1, f.order or 99)
        ordered = sorted(copper, key=key)

    total = len(ordered)
    for position, entry in enumerate(ordered):
        if position == 0:
            entry.layer = "F_Cu"
        elif position == total - 1 and total > 1:
            entry.layer = "B_Cu"
        else:
            entry.layer = f"In{position}_Cu"
        entry.order = position + 1
    return ordered


def normalise_into_fab(
    sources: List[Path],
    fab_dir: Path,
    board_name: str,
    *,
    replace: bool = True,
) -> ImportReport:
    """Identify every file and write the board data into `fab_dir` under canonical names."""
    fab_dir.mkdir(parents=True, exist_ok=True)
    board_name = safe_board_name(board_name)
    report = ImportReport(board_name=board_name)

    identified = [identify(path) for path in sources]
    by_source = {entry.source: path for entry, path in zip(identified, sources)}

    copper = _assign_copper_names([f for f in identified if f.role == "copper"])
    outline = [f for f in identified if f.role == "outline"]
    drills = [f for f in identified if f.role == "drill"]

    if not copper:
        found = ", ".join(sorted({p.name for p in sources})[:12]) or "nothing"
        raise GerberImportError(
            "No copper layers could be identified in this upload. Files seen: "
            f"{found}. A Gerber export needs at least a top copper layer and a board "
            "outline; check that the archive is the Gerber export rather than the "
            "design files."
        )
    if not outline:
        raise GerberImportError(
            "No board outline could be identified. Every coordinate in the project is "
            "measured from the outline's bounding box, so the import cannot continue "
            "without it. Re-export with the Edge.Cuts / board outline layer included."
        )

    if replace:
        for stale in list(fab_dir.glob("*.gbr")) + list(fab_dir.glob("*.drl")):
            stale.unlink(missing_ok=True)

    def write(entry: ImportedFile, layer_name: str, suffix: str = ".gbr") -> None:
        target = f"{board_name}-{layer_name}{suffix}"
        shutil.copyfile(by_source[entry.source], fab_dir / target)
        entry.target = target

    for entry in copper:
        write(entry, entry.layer or "F_Cu")
        report.gerber_files.append(entry.target or "")
        report.copper_layers.append(entry.layer or "")

    write(outline[0], "Edge_Cuts")
    report.gerber_files.append(outline[0].target or "")
    for extra in outline[1:]:
        report.warnings.append(
            f"'{extra.source}' also looks like a board outline; '{outline[0].source}' was used."
        )

    if drills:
        plated = next((d for d in drills if "npth" not in d.source.lower()), drills[0])
        write(plated, "PTH", ".drl")
        report.drill_file = plated.target
        for extra in drills:
            if extra is not plated:
                report.warnings.append(f"'{extra.source}' was not used; only one drill file is simulated.")
    else:
        report.warnings.append(
            "No drill file was found, so vias will not be modelled. If the board has "
            "no plated holes this is fine."
        )

    for entry in identified:
        if entry.role in ("mask", "silk", "paste") and entry.target is None:
            suffix_map = {"mask": "Mask", "silk": "Silkscreen", "paste": "Paste"}
            side = "F" if re.search(r"(?:^|[-_. ])(?:f[._ ]|top)|\.gt", entry.source, re.I) else "B"
            candidate = f"{side}_{suffix_map[entry.role]}"
            # Keep both sides distinct even when the side guess collides.
            if (fab_dir / f"{board_name}-{candidate}.gbr").exists():
                candidate = f"{'B' if side == 'F' else 'F'}_{suffix_map[entry.role]}"
            if not (fab_dir / f"{board_name}-{candidate}.gbr").exists():
                write(entry, candidate)
                report.gerber_files.append(entry.target or "")

    report.files = identified
    report.skipped = [f.source for f in identified if f.target is None]
    report.gerber_files = [f for f in report.gerber_files if f]
    return report


def infer_stackup(copper_layer_count: int, total_thickness_mm: float = 1.6) -> List[Dict]:
    """A symmetric FR4 stackup guess for this copper count, for the user to correct."""
    copper_mm = 0.035
    inner_copper_mm = 0.0175
    count = max(copper_layer_count, 1)
    if count == 1:
        return [{"name": "F_Cu", "type": "copper", "thickness_mm": copper_mm,
                 "material": None, "epsilon": None, "loss_tangent": None}]

    copper_total = 2 * copper_mm + (count - 2) * inner_copper_mm
    dielectric_total = max(total_thickness_mm - copper_total, 0.1)
    gap = dielectric_total / (count - 1)

    layers: List[Dict] = []
    for index in range(count):
        name = "F_Cu" if index == 0 else "B_Cu" if index == count - 1 else f"In{index}_Cu"
        layers.append({
            "name": name, "type": "copper",
            "thickness_mm": copper_mm if index in (0, count - 1) else inner_copper_mm,
            "material": None, "epsilon": None, "loss_tangent": None,
        })
        if index < count - 1:
            is_core = count > 2 and index == (count - 1) // 2
            layers.append({
                "name": "Core" if is_core else f"Prepreg {index + 1}",
                "type": "core" if is_core else "prepreg",
                "thickness_mm": round(gap, 4),
                "material": "FR4", "epsilon": 4.3, "loss_tangent": 0.02,
            })
    return layers
