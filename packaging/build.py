"""Build the standalone folder and archive into dist/ (zip on Windows, tar.gz on Linux).

    python packaging/build.py               PyInstaller + bundled openEMS + zip
    python packaging/build.py --no-openems  without openEMS
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from EMShrimulator.config import (  # noqa: E402
    APP_VERSION,
    BUNDLED_OPENEMS_DIRNAME,
    IS_WINDOWS,
    OPENEMS_DIR,
    openems_executable,
)

EXE_SUFFIX = ".exe" if IS_WINDOWS else ""

SPEC = ROOT / "packaging" / "EMShrimulator.spec"
WORK = ROOT / "build" / "pyinstaller"
DIST = ROOT / "dist"
BUNDLE = WORK / "dist" / "EMShrimulator"

# Including backups and stray copies like "openEMS.dll.bak".
OPENEMS_SKIP = ("matlab", "Tutorials", "my_ems_project", "include", "python", "*.lib", "*.pdb",
                "*.bak", "*.orig", "*.tmp", "*.dll.*",
                # Linux install prefix extras: headers, CMake files, docs, static libs
                "share", "cmake", "pkgconfig", "*.a", "venv")

LICENSE_TEXTS = ROOT / "packaging" / "licenses"
# In the build venv for building and testing, not in the bundle.
NOT_BUNDLED = {"pip", "setuptools", "wheel", "pytest", "pytest-qt", "pluggy", "iniconfig"}
_LICENSE_NAME = re.compile(r"(?i)^(licen[cs]e|copying|notice|copyright|authors)")


def step(message: str) -> None:
    print(f"==> {message}", flush=True)


def pyinstaller(clean: bool) -> None:
    step("PyInstaller (this takes several minutes)")
    shutil.rmtree(BUNDLE, ignore_errors=True)
    subprocess.check_call([
        sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm", *(["--clean"] if clean else []),
        "--distpath", str(WORK / "dist"), "--workpath", str(WORK / "work"),
    ], cwd=str(ROOT))


def bundle_openems(source: Path) -> None:
    if not openems_executable(source).is_file():
        raise SystemExit(f"openEMS was not found at {source}. Set OPENEMS_DIR, or pass --no-openems.")
    # Not "_internal/openEMS", which clashes with the Python package on a case-insensitive filesystem.
    target = BUNDLE / "_internal" / BUNDLED_OPENEMS_DIRNAME
    step(f"Bundling openEMS from {source}")
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*OPENEMS_SKIP))


def _is_license_file(path) -> bool:
    parts = path.parts
    return (len(parts) >= 2 and parts[0].endswith(".dist-info")
            and (parts[1] == "licenses" or (len(parts) == 2 and bool(_LICENSE_NAME.match(path.name)))))


def bundle_licenses() -> None:
    """Collect the licence texts the bundled components require (see THIRD_PARTY_NOTICES.md)."""
    step("Collecting licence texts")
    target = BUNDLE / "licenses"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir()
    shutil.copyfile(ROOT / "LICENSE", BUNDLE / "LICENSE.txt")
    shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", BUNDLE / "THIRD_PARTY_NOTICES.md")
    shutil.copyfile(ROOT / "LICENSE", target / "GPL-3.0.txt")
    shutil.copyfile(LICENSE_TEXTS / "LGPL-3.0.txt", target / "LGPL-3.0.txt")
    # Apache-2.0 for gerber2ems and gerbonara (whose wheel ships none).
    shutil.copyfile(ROOT / "vendor" / "gerber2ems-main" / "LICENSE", target / "Apache-2.0.txt")

    python = target / "python"
    cpython = Path(sys.base_prefix) / "LICENSE.txt"
    if cpython.is_file():
        (python / "CPython").mkdir(parents=True)
        shutil.copyfile(cpython, python / "CPython" / "LICENSE.txt")
    without = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name or name.lower().replace("_", "-") in NOT_BUNDLED:
            continue
        files = [f for f in dist.files or [] if _is_license_file(f)]
        if not files:
            without.append(name)
        for f in files:
            relative = f.parts[2:] if f.parts[1] == "licenses" else f.parts[1:]
            dest = python.joinpath(name, *relative)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f.locate(), dest)
    # scikit-rf's wheel leaves its BSD licence out.
    (python / "scikit-rf").mkdir(exist_ok=True)
    shutil.copyfile(LICENSE_TEXTS / "scikit-rf-BSD-3-Clause.txt", python / "scikit-rf" / "LICENSE.txt")
    if without:
        # Covered by the GPL/LGPL/Apache texts above.
        print("  no licence file in the metadata of: " + ", ".join(sorted(without, key=str.lower)), flush=True)


def smoke_test(expect_openems: bool) -> None:
    step("Checking the frozen engine (gems-worker doctor)")
    env = dict(os.environ)
    env.pop("OPENEMS_DIR", None)
    env["GEMS_PROJECTS_DIR"] = str(WORK / "smoke-projects")
    result = subprocess.run([str(BUNDLE / f"gems-worker{EXE_SUFFIX}"), "doctor"], capture_output=True, text=True,
                            env=env, timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        if expect_openems:
            raise SystemExit("The frozen engine failed its own diagnostics - see above.")
        print("(expected without a bundled openEMS)")


def make_zip() -> Path:
    DIST.mkdir(exist_ok=True)
    base = DIST / f"EMShrimulator-{APP_VERSION}-{'win64' if IS_WINDOWS else 'linux-x86_64'}"
    kind = "zip" if IS_WINDOWS else "gztar"  # tar keeps the executable bits
    step(f"Writing {base.name}.{'zip' if IS_WINDOWS else 'tar.gz'}")
    return Path(shutil.make_archive(str(base), kind, root_dir=BUNDLE.parent, base_dir=BUNDLE.name))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-openems", action="store_true", help="do not bundle openEMS")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="reuse the existing PyInstaller output")
    parser.add_argument("--clean", action="store_true", help="discard PyInstaller's analysis cache first")
    args = parser.parse_args()

    if not args.skip_pyinstaller:
        pyinstaller(args.clean)
    if not args.no_openems:
        bundle_openems(OPENEMS_DIR)
    bundle_licenses()
    smoke_test(expect_openems=not args.no_openems)
    archive = make_zip()
    step(f"Done. {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
