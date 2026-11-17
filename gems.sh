#!/usr/bin/env bash
# EMShrimulator Studio launcher for Linux (Fedora first): setup, run, build, diagnose.
#
#   ./gems.sh                    start the desktop application
#   ./gems.sh run --open DIR     start it and open a project
#   ./gems.sh setup              install system packages, build openEMS, create the venv
#   ./gems.sh setup --force      ... recreating the venv from scratch
#   ./gems.sh openems            (re)build only openEMS and its Python bindings
#   ./gems.sh doctor             check the install and print what is missing
#   ./gems.sh test               run the test suite
#   ./gems.sh build              build the standalone folder and tar.gz in dist/
#   ./gems.sh desktop            add EMShrimulator Studio to the GNOME/KDE app menu
#
# Environment overrides:
#   OPENEMS_DIR        openEMS install prefix            (default: ~/opt/openEMS)
#   GEMS_PYTHON        Python used to create the venv    (default: python3.13)
#   GEMS_OPENEMS_REF   openEMS-Project git tag to build  (default: v0.37.0)
#   GEMS_JOBS          parallel compile jobs             (default: number of CPUs)
#   GEMS_SKIP_DNF=1    do not install system packages with dnf
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"
OPENEMS_DIR="${OPENEMS_DIR:-$HOME/opt/openEMS}"
OPENEMS_REF="${GEMS_OPENEMS_REF:-v0.37.0}"
OPENEMS_REPO="https://github.com/thliebig/openEMS-Project.git"
OPENEMS_SRC="$ROOT/.cache/openEMS-Project-$OPENEMS_REF"
JOBS="${GEMS_JOBS:-$(nproc 2>/dev/null || echo 4)}"
DESKTOP_ID="emshrimulator-studio"

# Build and runtime packages. openEMS is not packaged for Fedora, so it is compiled from source;
# the xcb/xkb packages are what Qt's X11 plugin needs (the 3D viewport runs through XWayland).
DNF_PACKAGES=(
    gcc gcc-c++ make cmake git
    boost-devel tinyxml-devel hdf5-devel CGAL-devel vtk-devel
    python3.13 python3.13-devel
    xcb-util-cursor xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil
    libxkbcommon-x11 mesa-libGL mesa-libEGL fontconfig
)

if [[ -t 1 ]]; then
    C_STEP=$'\e[36m'; C_OK=$'\e[32m'; C_WARN=$'\e[33m'; C_BAD=$'\e[31m'; C_OFF=$'\e[0m'
else
    C_STEP=""; C_OK=""; C_WARN=""; C_BAD=""; C_OFF=""
fi
step() { printf '%s==> %s%s\n' "$C_STEP" "$*" "$C_OFF"; }
ok()   { printf '%s  OK  %s%s\n' "$C_OK" "$*" "$C_OFF"; }
warn() { printf '%s  !!  %s%s\n' "$C_WARN" "$*" "$C_OFF"; }
bad()  { printf '%s  XX  %s%s\n' "$C_BAD" "$*" "$C_OFF" >&2; }
die()  { bad "$*"; exit 1; }

# ------------------------------------------------------------------ helpers

openems_lib_dirs() {
    local d out=""
    for d in "$OPENEMS_DIR/lib64" "$OPENEMS_DIR/lib"; do
        [[ -d "$d" ]] && out="${out:+$out:}$d"
    done
    printf '%s' "$out"
}

openems_installed() {
    [[ -x "$OPENEMS_DIR/bin/openEMS" ]] && compgen -G "$OPENEMS_DIR/lib*/libCSXCAD.so*" >/dev/null
}

use_environment() {
    export OPENEMS_DIR
    export OPENEMS_INSTALL_PATH="$OPENEMS_DIR"
    local libs; libs="$(openems_lib_dirs)"
    if [[ -n "$libs" ]]; then
        export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    [[ -d "$OPENEMS_DIR/bin" ]] && export PATH="$OPENEMS_DIR/bin:$PATH"
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONIOENCODING=utf-8
    # VTK's Qt widget needs an X11 window; under Wayland use XWayland unless told otherwise.
    if [[ -n "${WAYLAND_DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
        export QT_QPA_PLATFORM=xcb
    fi
}

assert_venv() {
    if [[ ! -x "$VENV_PY" ]]; then
        bad "The virtual environment is missing."
        echo "     Run:  ./gems.sh setup"
        exit 1
    fi
}

find_python() {
    local candidate
    for candidate in "${GEMS_PYTHON:-}" python3.13 python3; do
        [[ -z "$candidate" ]] && continue
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c 'import sys; raise SystemExit(sys.version_info[:2] < (3, 13))' 2>/dev/null; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

# ------------------------------------------------------------------- setup

install_system_packages() {
    if [[ "${GEMS_SKIP_DNF:-0}" == 1 ]]; then
        warn "GEMS_SKIP_DNF=1 - not installing system packages."
        return
    fi
    if ! command -v dnf >/dev/null 2>&1; then
        warn "dnf not found - this is not Fedora. Install the equivalents of these yourself:"
        echo "     ${DNF_PACKAGES[*]}"
        return
    fi
    local missing=() pkg
    for pkg in "${DNF_PACKAGES[@]}"; do
        rpm -q --whatprovides "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        ok "System packages already installed"
        return
    fi
    step "Installing system packages with dnf (asks for your password): ${missing[*]}"
    sudo dnf install -y "${missing[@]}" || die "dnf install failed."
    ok "System packages installed"
}

fetch_openems_source() {
    if [[ -d "$OPENEMS_SRC/.git" ]]; then
        ok "openEMS source at $OPENEMS_SRC"
        return
    fi
    step "Downloading openEMS-Project $OPENEMS_REF"
    mkdir -p "$(dirname "$OPENEMS_SRC")"
    rm -rf "$OPENEMS_SRC.partial"
    git clone --depth 1 --branch "$OPENEMS_REF" --recurse-submodules --shallow-submodules \
        "$OPENEMS_REPO" "$OPENEMS_SRC.partial" || die "Cloning $OPENEMS_REPO failed."
    mv "$OPENEMS_SRC.partial" "$OPENEMS_SRC"
}

build_openems() {
    fetch_openems_source
    local build="$OPENEMS_SRC/build"
    local toolchain="$OPENEMS_SRC/gems-toolchain.cmake"
    # The superbuild forwards only CMAKE_TOOLCHAIN_FILE to fparser/CSXCAD/openEMS, so the RPATH
    # (libraries found without LD_LIBRARY_PATH) and Release mode travel in a toolchain file.
    cat > "$toolchain" <<EOF
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_INSTALL_RPATH "$OPENEMS_DIR/lib64;$OPENEMS_DIR/lib" CACHE STRING "" FORCE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON CACHE BOOL "" FORCE)
EOF
    step "Building openEMS $OPENEMS_REF into $OPENEMS_DIR with $JOBS jobs (10-30 minutes)"
    mkdir -p "$OPENEMS_DIR" "$build"
    local log="$build/gems-build.log"
    echo "     full log: $log"
    {
        cmake -S "$OPENEMS_SRC" -B "$build" \
            -DBUILD_APPCSXCAD=NO \
            -DCMAKE_INSTALL_PREFIX="$OPENEMS_DIR" \
            -DCMAKE_TOOLCHAIN_FILE="$toolchain" &&
        CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" cmake --build "$build" -j "$JOBS"
    } >"$log" 2>&1 || { tail -n 40 "$log" >&2; die "Building openEMS failed - see $log"; }
    openems_installed || die "The build finished but $OPENEMS_DIR/bin/openEMS or libCSXCAD.so is missing."
    ok "openEMS installed at $OPENEMS_DIR"
}

install_openems_bindings() {
    fetch_openems_source
    step "Building the CSXCAD and openEMS Python bindings into the venv"
    local libs; libs="$(openems_lib_dirs)"
    CSXCAD_INSTALL_PATH="$OPENEMS_DIR" OPENEMS_INSTALL_PATH="$OPENEMS_DIR" \
    LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$VENV_PY" -m pip install --quiet "$OPENEMS_SRC/CSXCAD/python" \
        || die "Building the CSXCAD Python bindings failed."
    CSXCAD_INSTALL_PATH="$OPENEMS_DIR" OPENEMS_INSTALL_PATH="$OPENEMS_DIR" \
    LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$VENV_PY" -m pip install --quiet "$OPENEMS_SRC/openEMS/python" \
        || die "Building the openEMS Python bindings failed."
    ok "CSXCAD and openEMS bindings installed"
}

create_venv() {
    if [[ -d "$VENV" && ! -x "$VENV_PY" ]]; then
        # A venv made on Windows (Scripts\python.exe, paths into C:\) cannot run here.
        warn "$VENV is not a Linux venv (probably created on Windows) - recreating it."
        rm -rf "$VENV"
    fi
    if [[ "$FORCE" == 1 && -d "$VENV" ]]; then
        step "Removing the existing venv (--force)"
        rm -rf "$VENV"
    fi
    if [[ ! -x "$VENV_PY" ]]; then
        local py
        py="$(find_python)" || die "Python 3.13 or newer was not found. Install it with:  sudo dnf install python3.13 python3.13-devel"
        case "$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" in
            3.13) ;;
            *) warn "Using $py; the app is tested on Python 3.13 (sudo dnf install python3.13 python3.13-devel)." ;;
        esac
        step "Creating the virtual environment with $py"
        "$py" -m venv "$VENV" || die "Creating the venv failed."
    fi
    ok "venv at $VENV"
}

cmd_setup() {
    step "Setting up EMShrimulator Studio"
    install_system_packages
    create_venv

    step "Installing Python dependencies (this takes a few minutes the first time)"
    "$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel || die "Upgrading pip failed."
    "$VENV_PY" -m pip install --quiet -r "$ROOT/requirements.txt" || die "Installing the Python dependencies failed."
    ok "Python dependencies installed"

    if openems_installed && [[ "$FORCE" != 1 ]]; then
        ok "openEMS already built at $OPENEMS_DIR (./gems.sh openems rebuilds it)"
    else
        build_openems
    fi
    if [[ "$FORCE" == 1 ]] || ! "$VENV_PY" -c 'import CSXCAD, openEMS' >/dev/null 2>&1; then
        install_openems_bindings
    else
        ok "CSXCAD and openEMS bindings already in the venv"
    fi

    step "Installing the vendored gerber2ems"
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
        "$VENV_PY" -m pip install --quiet -e "$ROOT/vendor/gerber2ems-main" --no-deps \
        || die "Installing gerber2ems failed."
    ok "gerber2ems installed (patched: gerbv replaced by gerbonara + resvg)"

    echo
    cmd_doctor || true
    echo
    ok "Setup complete. Start the app with:  ./gems.sh"
    echo "     Add it to your app menu with:     ./gems.sh desktop"
}

cmd_openems() {
    assert_venv
    build_openems
    install_openems_bindings
}

# ------------------------------------------------------------------ doctor

cmd_doctor() {
    assert_venv
    use_environment
    step "Checking the install"
    local code=0
    "$VENV_PY" -m EMShrimulator.core.diagnostics || code=1
    if ! "$VENV_PY" -c "import PySide6, vtkmodules, vtkmodules.qt.QVTKRenderWindowInteractor; print('  OK  Qt', PySide6.__version__, '| VTK', vtkmodules.__version__)"; then
        code=1
        bad "The desktop GUI libraries are missing - run: ./gems.sh setup"
    fi
    echo
    if [[ $code == 0 ]]; then ok "Everything checks out."; else bad "Problems found - see above."; fi
    return $code
}

# --------------------------------------------------------------------- run

cmd_run() {
    assert_venv
    use_environment
    local args=(-m EMShrimulator)
    [[ -n "$OPEN" ]] && args+=(--open "$OPEN")
    cd "$ROOT"
    step "Starting EMShrimulator Studio (the first start loads the solver and 3D libraries, which can take a while)"
    exec "$VENV_PY" "${args[@]}"
}

# ------------------------------------------------------------------- build / test

cmd_build() {
    assert_venv
    use_environment
    "$VENV_PY" "$ROOT/packaging/build.py" "${PASSTHROUGH[@]}"
}

cmd_test() {
    assert_venv
    use_environment
    cd "$ROOT"
    local pytest_files=() script_files=() f
    for f in tests/test_*.py; do
        # Files made of test_ functions need pytest; run as scripts they would only define them.
        if grep -q '^def test_' "$f"; then pytest_files+=("$f"); else script_files+=("$f"); fi
    done
    if [[ ${#pytest_files[@]} -gt 0 ]]; then
        "$VENV_PY" -m pytest -q "${pytest_files[@]}" || die "pytest tests failed"
    fi
    for f in "${script_files[@]}"; do
        "$VENV_PY" "$f" || die "$(basename "$f") failed"
    done
    ok "All tests passed"
}

# ---------------------------------------------------------------- desktop entry

cmd_desktop() {
    local apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    local icons="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
    mkdir -p "$apps" "$icons"
    cp "$ROOT/EMShrimulator/resources/icons/shrimp.svg" "$icons/$DESKTOP_ID.svg"
    cat > "$apps/$DESKTOP_ID.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=EMShrimulator Studio
Comment=EM simulation over openEMS
Exec="$ROOT/gems.sh" run
Path=$ROOT
Icon=$DESKTOP_ID
Terminal=false
Categories=Science;Engineering;Electronics;
EOF
    chmod +x "$apps/$DESKTOP_ID.desktop"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$apps" >/dev/null 2>&1 || true
    ok "Added to the app menu: $apps/$DESKTOP_ID.desktop"
}

# ------------------------------------------------------------------ dispatch

COMMAND="run"
OPEN=""
FORCE=0
PASSTHROUGH=()
if [[ $# -gt 0 && "$1" != -* ]]; then COMMAND="$1"; shift; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --open) OPEN="${2:?--open needs a project folder}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

case "$COMMAND" in
    run)     cmd_run ;;
    setup)   cmd_setup ;;
    openems) cmd_openems ;;
    doctor)  cmd_doctor ;;
    test)    cmd_test ;;
    build)   cmd_build ;;
    desktop) cmd_desktop ;;
    *) bad "Unknown command '$COMMAND'"; sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
