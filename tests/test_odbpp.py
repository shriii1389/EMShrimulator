"""ODB++ import: a small synthetic product converted and rendered. Set GEMS_ODB_SAMPLE to a real .tgz to
import it too.

Run with `.venv/bin/python tests/test_odbpp.py`; the exit code is the verdict.
"""
import io
import os
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from EMShrimulator.core.layout import gerber_shapes, odb_convert, odbpp
from EMShrimulator.core.layout.gerber_render import edge_cuts_bbox_mm, render_layer_preview_png

results = []
WORK = Path(tempfile.mkdtemp(prefix="gems-odb-"))
HELLO_Z = bytes.fromhex("1f9d9048cac488014301")  # `compress` of b"He110\n", from unlzw3's test data


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:  # noqa: BLE001
        results.append(("FAIL", label, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="ascii")


def product() -> Path:
    """Two copper layers, a drill layer and nets; lower-case directories, upper-case matrix."""
    root = WORK / "job"
    if root.exists():
        return root
    write(root / "matrix" / "matrix", """
STEP {
    COL=1
    NAME=PCB
}

LAYER {
    ROW=1
    CONTEXT=BOARD
    TYPE=SIGNAL
    NAME=TOP
    POLARITY=POSITIVE
}

LAYER {
    ROW=2
    CONTEXT=BOARD
    TYPE=DIELECTRIC
    NAME=CORE
}

LAYER {
    ROW=3
    CONTEXT=BOARD
    TYPE=POWER_GROUND
    NAME=BOT
    POLARITY=POSITIVE
}

LAYER {
    ROW=4
    CONTEXT=BOARD
    TYPE=DRILL
    NAME=DRILL
    START_NAME=TOP
    END_NAME=BOT
}
""")
    step = root / "steps" / "pcb"
    write(step / "profile", """
UNITS=INCH
S P 0
OB 0 0 I
OS 1 0
OS 1 0.5
OS 0 0.5
OE
SE
""")
    # Inches, symbols in mils: rotated rects, a trace, a long clockwise arc and a user symbol.
    write(step / "layers" / "top" / "features", """
UNITS=INCH
F 5
$0 r10
$1 rect40x20
$2 myshape
P 0.2 0.25 1 P 0 1 ;0=1
P 0.4 0.25 1 P 0 8 45
L 0.1 0.1 0.6 0.1 0 P 0 ;;ID=7
A 0.6 0.1 0.8 0.3 0.6 0.3 0 P 0 Y
P 0.75 0.42 2 P 0 0
""")
    # Millimetres, symbols in microns: a plane with a hole and another net's pad inside it.
    write(step / "layers" / "bot" / "features", """
UNITS=MM
$0 r200
S P 0
OB 1 1 I
OS 20 1
OS 20 11
OS 1 11
OE
OB 5 5 H
OS 5 7
OS 7 7
OS 7 5
OE
SE
P 6 6 0 P 0 0
""")
    write(step / "layers" / "drill" / "features", """
UNITS=INCH
$0 r12
P 0.5 0.25 0 P 0 0
""")
    write(step / "eda" / "data", """
HDR test
UNITS=INCH
LYR top core bot drill
NET SIG;;ID=1
SNT TRC
FID C 0 2
FID C 0 3
NET GND
FID C 2 0
NET VIA_NET
FID C 2 1
NET $NONE$
FID C 0 0
""")
    write(root / "symbols" / "myshape" / "features", """
UNITS=INCH
S P 0
OB -0.01 -0.01 I
OS 0.01 -0.01
OS 0.01 0.01
OS -0.01 0.01
OE
SE
""")
    return root


def converted():
    fab = WORK / "fab"
    if not fab.exists():
        report = odb_convert.import_archive(product(), WORK / "work", fab, "t")
        (WORK / "report.txt").write_text(repr(report), encoding="utf-8")
    return fab, eval((WORK / "report.txt").read_text(encoding="utf-8"))  # noqa: S307 - our own repr


def alpha_at(fab: Path, layer: str, points_mm):
    """Coverage (0-255) of a converted layer at board-local mm points."""
    edge = fab / "t-Edge_Cuts.gbr"
    (x0, y0), (x1, y1) = edge_cuts_bbox_mm(edge)
    ppm = 40.0
    png = render_layer_preview_png(fab / f"t-{layer}.gbr", ((x0, y0), (x1, y1)), 1000.0 / ppm)
    alpha = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[:, :, 3]
    return [int(alpha[int((y1 - y) * ppm), int((x - x0) * ppm)]) for x, y in points_mm]


OFFSET = odb_convert.OUTLINE_WIDTH_MM / 2  # the stroked outline's half width


def mm(x_in, y_in):
    return x_in * 25.4 + OFFSET, y_in * 25.4 + OFFSET


# ---------------------------------------------------------------- checks

def matrix_blocks_both_ways():
    layers, steps = odbpp.read_matrix(product())
    assert steps == ["PCB"] and [l.name for l in layers] == ["TOP", "CORE", "BOT", "DRILL"]
    assert layers[2].type == "POWER_GROUND" and layers[3].start_name == "TOP"
    old = odbpp._parse_blocks("{\nNAME=A\nCOL=1\n}\n{\nNAME=B\nTYPE=SIGNAL\nROW=1\n}\n")
    assert [kind for kind, _ in old] == ["", ""] and old[1][1]["TYPE"] == "SIGNAL"


def symbols_are_mils_and_microns():
    inch = odbpp.standard_symbol("r10", 25.4 / 1000)
    assert abs(inch.width - 0.254) < 1e-9, inch.width
    rect = odbpp.standard_symbol("rect40x20xr5", 25.4 / 1000)
    assert rect.kind == "outline" and abs(rect.width - 1.016) < 1e-9
    assert any(s[0] == "A" for s in rect.contours[0].segments), "rounded corners stay arcs"
    donut = odbpp.standard_symbol("donut_r500x250", 1 / 1000)
    assert (donut.kind, donut.width, donut.inner) == ("donut", 0.5, 0.25)
    assert odbpp.standard_symbol("thr500x300x45x4x50", 1 / 1000).approximate
    assert odbpp.standard_symbol("special_thing", 1 / 1000).kind == "unknown"


def conversion_report():
    fab, report = converted()
    assert report["copper_layers"] == ["F_Cu", "B_Cu"], report["copper_layers"]
    assert report["nets"] == 3 and report["vias"] == 1, report
    assert not [w for w in report["warnings"] if "not understood" in w], report["warnings"]
    drill = (fab / "t-PTH.drl").read_text()
    assert "T1C0.305" in drill, drill  # 12 mil


def pads_rotate_and_traces_keep_their_width():
    fab, _ = converted()
    text = (fab / "t-F_Cu.gbr").read_text()
    assert "R,0.508X1.016" in text, "a rect turned 90° flashes with its sides swapped"
    assert "C,0.254" in text, "a 10 mil trace is 0.254 mm wide"
    assert text.count("G36*") == 2, "the 45° rect and the user symbol are regions"
    gerber = gerber_shapes.open_with_nets(fab / "t-F_Cu.gbr")
    arcs = [o for o in gerber.objects if type(o).__name__ == "Arc"]
    assert len(arcs) == 1 and arcs[0].clockwise, "the arc's direction is the field after its D-code"
    # Clockwise, three quarters of a turn through the left side.
    left, right_below = alpha_at(fab, "F_Cu", [mm(0.4, 0.3), mm(0.6 + 0.2 * 0.7071, 0.3 - 0.2 * 0.7071)])
    assert left > 200 and right_below < 50, (left, right_below)
    user = alpha_at(fab, "F_Cu", [mm(0.75, 0.42), mm(0.75 + 0.02, 0.42)])
    assert user[0] > 200 and user[1] < 50, f"the user symbol is a 20 mil square: {user}"


def holes_cut_their_island_only():
    fab, _ = converted()
    hole, pad, plane = alpha_at(fab, "B_Cu", [(5.4 + OFFSET, 5.4 + OFFSET), (6 + OFFSET, 6 + OFFSET),
                                             (12 + OFFSET, 9 + OFFSET)])
    assert hole < 30, f"the plane's hole is drawn over ({hole})"
    assert pad > 200, f"the other net's pad in the hole was erased ({pad})"
    assert plane > 200, plane
    assert "%LPC" not in (fab / "t-B_Cu.gbr").read_text(), "holes are cut in, not drawn clear"


def nets_reach_every_object():
    fab, _ = converted()
    nets = {}
    for obj in gerber_shapes.open_with_nets(fab / "t-B_Cu.gbr").objects:
        nets.setdefault(type(obj).__name__, set()).add(gerber_shapes.net_of(obj))
    assert nets == {"Region": {"GND"}, "Flash": {"VIA_NET"}}, nets
    top = gerber_shapes.layer_shapes(fab / "t-F_Cu.gbr", edge_cuts_bbox_mm(fab / "t-Edge_Cuts.gbr"))
    assert "SIG" in set(top["nets"]) and (top["index"][:, 7] == -1).any(), "unnamed copper stays net -1"


def compressed_files_decode():
    assert odbpp.unlzw(HELLO_Z) == b"He110\n"
    folder = WORK / "z"
    folder.mkdir(exist_ok=True)
    (folder / "features.Z").write_bytes(HELLO_Z)
    assert odbpp._read_text(folder / "features") == "He110\n"
    try:
        odbpp.unlzw(b"\x1f\x9d\x05xx")
    except odbpp.ODBError:
        pass
    else:
        raise AssertionError("5-bit codes are not compress output")


def the_official_sample():
    sample = os.environ.get("GEMS_ODB_SAMPLE")
    if not sample or not Path(sample).is_file():
        print("      (GEMS_ODB_SAMPLE not set: the Siemens sample is skipped)")
        return
    report = odb_convert.import_archive(Path(sample), WORK / "sample-work", WORK / "sample-fab", "s")
    assert len(report["copper_layers"]) == 10, report["copper_layers"]
    assert report["nets"] > 600 and report["vias"] > 9000, (report["nets"], report["vias"])
    assert not [w for w in report["warnings"] if "not understood" in w], report["warnings"]


if __name__ == "__main__":
    try:
        check("matrix: `LAYER {` blocks and bare `{` ones", matrix_blocks_both_ways)
        check("symbols: sizes in mils or microns, rounded corners as arcs", symbols_are_mils_and_microns)
        check("convert: two copper layers, three nets, the hole's diameter", conversion_report)
        check("convert: rotated pads, trace widths, the arc's direction, user symbols", pads_rotate_and_traces_keep_their_width)
        check("convert: a plane's hole cuts the plane, not the pad sitting in it", holes_cut_their_island_only)
        check("nets: every object carries its net, regions included", nets_reach_every_object)
        check("files: Unix-compressed (.Z) features decode", compressed_files_decode)
        check("the official ODB++ 8.1 sample imports (with GEMS_ODB_SAMPLE)", the_official_sample)
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    failed = 0
    for status, label, detail in results:
        print(f"{status}  {label}")
        if detail:
            print("      " + detail.replace("\n", "\n      "))
            failed += 1
    raise SystemExit(1 if failed else 0)
