"""Download a real third-party Gerber export to test the importer: python tools/fetch_test_board.py

Writes testdata/usbvil-gerbers.zip (Gerber X1, Eagle extensions, inches, an implied-decimal Excellon .txt).
The board (tracespace's usbvil fixture) is GPL-3.0, so it is downloaded rather than committed:
https://github.com/tracespace/tracespace/tree/main/packages/fixtures/boards/usbvil
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/tracespace/tracespace/main"
        "/packages/fixtures/boards/usbvil")
FILES = [
    "pic18f14k50.gtl",   # top copper
    "pic18f14k50.gbl",   # bottom copper
    "pic18f14k50.gko",   # board outline
    "pic18f14k50.gts",   # top soldermask
    "pic18f14k50.gbs",   # bottom soldermask
    "pic18f14k50.gto",   # top silkscreen
    "pic18f14k50.gbo",   # bottom silkscreen
    "pic18f14k50.txt",   # Excellon drill, despite the extension
    "LICENSE",
]

NOTE = """This board is the PIC18F14K50 USB VIL adapter from the tracespace
project's test fixtures, licensed GPL-3.0. It is here to exercise the Gerber
importer against an export that is not KiCad's: Gerber X1 with no layer
attributes, Eagle extensions, inch units, and an Excellon drill named .txt.

Source: https://github.com/tracespace/tracespace/tree/main/packages/fixtures/boards/usbvil
"""


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "testdata"
    out_dir.mkdir(exist_ok=True)
    archive = out_dir / "usbvil-gerbers.zip"

    payloads = {}
    for name in FILES:
        url = f"{BASE}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payloads[name] = response.read()
        except Exception as exc:  # noqa: BLE001 - a download failure is the message
            print(f"could not fetch {url}: {exc}", file=sys.stderr)
            return 1
        print(f"  {len(payloads[name]):7d} bytes  {name}")

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in payloads.items():
            zf.writestr(f"usbvil/{name}", data)
        zf.writestr("usbvil/README.txt", NOTE)

    print(f"\nWrote {archive} ({archive.stat().st_size} bytes).")
    print("Import it with the 3D Layout workspace's 'Import Gerbers' button.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
