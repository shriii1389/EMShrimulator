"""The numpy Touchstone reader and writer: exact round trips, the 2-port order, skrf's own files."""
import tempfile
from pathlib import Path

import numpy as np

from EMShrimulator.core.em import touchstone as ts
from EMShrimulator.core.em.results import read_results

results = []
TMP = Path(tempfile.mkdtemp())


def check(label, fn):
    try:
        fn()
        results.append(("PASS", label, ""))
    except Exception as e:
        results.append(("FAIL", label, f"{type(e).__name__}: {e}"))


def random_network(n, points=41, seed=0):
    rng = np.random.default_rng(seed + n)
    f = np.linspace(1e9, 6e9, points)
    s = rng.normal(size=(points, n, n)) + 1j * rng.normal(size=(points, n, n))
    return f, s


# --- 1. what the writer writes, the reader reads back bit for bit -------------
def round_trip():
    for n in (1, 2, 3, 4):
        f, s = random_network(n)
        path = ts.write(TMP / f"rt{n}", f, s, 50.0)
        assert path.name == f"rt{n}.s{n}p", path.name
        back = ts.read(path)
        assert np.array_equal(back.freq_hz, f), f"{n}-port frequencies differ"
        assert np.array_equal(back.s, s), f"{n}-port S-matrix differs"
        assert back.z0 == 50.0 and back.ports == n


check("1- to 4-port files round-trip exactly", round_trip)


# --- 2. a two-port record is S11 S21 S12 S22 ----------------------------------
def two_port_order():
    path = TMP / "order.s2p"
    path.write_text("# Hz S RI R 50\n1e9  1 0  2 0  3 0  4 0\n")
    s = ts.read(path).s[0]
    assert s[0, 0] == 1 and s[1, 0] == 2 and s[0, 1] == 3 and s[1, 1] == 4, s


check("2-port records are column-major (S11 S21 S12 S22)", two_port_order)


def three_port_order():
    path = TMP / "order.s3p"
    path.write_text("# Hz S RI R 50\n1e9 11 0 12 0 13 0\n 21 0 22 0 23 0\n 31 0 32 0 33 0\n")
    s = ts.read(path).s[0].real
    assert np.array_equal(s, [[11, 12, 13], [21, 22, 23], [31, 32, 33]]), s


check("3-port records are row-major", three_port_order)


# --- 3. the other formats and units --------------------------------------------
def formats_and_units():
    ma = TMP / "ma.s1p"
    ma.write_text("! comment\n# GHz S MA R 75\n2.5 0.5 90\n")
    net = ts.read(ma)
    assert net.freq_hz[0] == 2.5e9 and net.z0 == 75.0
    assert np.allclose(net.s[0, 0, 0], 0.5j), net.s
    db = TMP / "db.s1p"
    db.write_text("# MHz S DB R 50\n100 -20 180\n")
    net = ts.read(db)
    assert net.freq_hz[0] == 1e8 and np.allclose(net.s[0, 0, 0], -0.1), net.s
    default = TMP / "default.s1p"
    default.write_text("1 1 0\n")  # no option line: GHz, MA, 50 ohms
    net = ts.read(default)
    assert net.freq_hz[0] == 1e9 and net.z0 == 50.0


check("MA and DB data, kHz-GHz units and the option-line defaults", formats_and_units)


# --- 4. scikit-rf's own layout, with its per-port impedance comments ----------
def skrf_layout():
    path = TMP / "skrf.s1p"
    path.write_text(
        "! Created with skrf 1.9.0 (http://scikit-rf.org).\n"
        "# Hz S RI R\n!freq ReS11 ImS11\n"
        "1000000000.0 0.25 -0.5\n! Port Impedance 42.00000000000000 0.00000000000000 \n"
        "2000000000.0 0.125 0.0\n! Port Impedance 42.00000000000000 0.00000000000000 \n")
    net = ts.read(path)
    assert net.z0 == 42.0, net.z0
    assert np.array_equal(net.s[:, 0, 0], [0.25 - 0.5j, 0.125]), net.s


check("scikit-rf files: an R with no value, z0 from its Port Impedance comments", skrf_layout)


# --- 5. what the strict reader refuses ------------------------------------------
def refuses_the_unusual():
    v2 = "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n"
    for name, text in (("noise.s2p", v2 + "[Network Data]\n1e9 1 0 2 0 3 0 4 0\n[Noise Data]\n1e9 1 0 0 50\n"),
                       ("lower.s2p", v2 + "[Matrix Format] Lower\n[Network Data]\n1e9 1 0 2 0 4 0\n"),
                       ("ports.s2p", "[Version] 2.0\n# Hz S RI R 50\n[Number of Ports] 3\n1e9 1 0 2 0 3 0 4 0\n"),
                       ("z.s1p", "# Hz Z RI R 50\n1e9 1 0\n"),
                       ("short.s2p", "# Hz S RI R 50\n1e9 1 0 2 0 3 0\n")):
        try:
            ts._parse(text, ts.port_count(Path(name)))
        except ts.NotPlainTouchstone:
            continue
        raise AssertionError(f"{name} was accepted")


check("Touchstone 2 noise/lower-matrix data, Z-parameters and truncated records go to scikit-rf",
      refuses_the_unusual)


# --- 5b. different port impedances: Touchstone 2 [Reference] --------------------
def per_port_references():
    for n, z0 in ((2, [50.0, 75.0]), (3, [50.0, 75.0, 12.5])):
        f, s = random_network(n, seed=3)
        path = ts.write(TMP / f"ref{n}", f, s, z0)
        text = path.read_text()
        assert "[Reference] " + " ".join(f"{z:g}" for z in z0) in text and "[End]" in text, text[:300]
        back = ts._parse(text, n)  # natively, without scikit-rf
        assert np.array_equal(back.s, s) and list(back.z0_ports) == z0, f"{n}-port differs"
        try:
            import skrf
        except ImportError:
            continue
        net = skrf.Network(str(path))
        assert np.array_equal(net.s, s) and np.allclose(net.z0[0], z0), f"{n}-port differs under skrf"
    # Equal impedances stay Touchstone 1, and a 12_21 file is row-major.
    assert "[" not in ts.write(TMP / "same", *random_network(2), [50.0, 50.0]).read_text()
    row_major = "[Version] 2.0\n# Hz S RI R 50\n[Two-Port Data Order] 12_21\n[Reference] 50\n 60\n1e9 1 0 2 0 3 0 4 0\n"
    net = ts._parse(row_major, 2)
    assert net.s[0, 0, 1] == 2 and net.s[0, 1, 0] == 3 and list(net.z0_ports) == [50, 60], net


check("different port impedances: a Touchstone 2 [Reference] file, exact natively and in scikit-rf",
      per_port_references)


# --- 6. scikit-rf reads what the writer writes ----------------------------------
def skrf_reads_ours():
    try:
        import skrf
    except ImportError:
        return  # optional for the app
    for n in (1, 2, 4):
        f, s = random_network(n, seed=7)
        net = skrf.Network(str(ts.write(TMP / f"x{n}", f, s, 50.0)))
        assert np.array_equal(net.f, f) and np.array_equal(net.s, s), f"{n}-port differs under skrf"


check("scikit-rf reads the written files identically", skrf_reads_ours)


# --- 7. the results views come out of a written file ---------------------------
def results_views():
    f = np.linspace(1e9, 3e9, 201)
    s11 = 0.9 * (1 - 0.99 * np.exp(-((f - 2e9) / 1e8) ** 2))  # a dip at 2 GHz
    s = np.zeros((len(f), 2, 2), dtype=complex)
    s[:, 0, 0] = s11
    s[:, 1, 0] = s[:, 0, 1] = np.sqrt(1 - s11 ** 2)
    data = read_results(ts.write(TMP / "views", f, s, 50.0))
    # S22 is zero everywhere, so it isn't offered.
    assert data["ports"] == 2 and set(data["traces"]) == {"S11", "S12", "S21"}, set(data["traces"])
    marker = data["markers"]["resonance"]
    assert abs(marker["frequency_hz"] - 2e9) < 2e7, marker
    assert data["tdr"] is not None and len(data["traces"]["S11"]["vswr"]) == len(f)


check("read_results derives every view from a written file, leaving out what no run measured", results_views)


# --- 8. unexcited ports: reciprocity, per-port impedance, TDR -------------------
def unexcited_ports():
    f = np.linspace(1e9, 2e9, 11)
    s11, s21 = np.full(len(f), 0.2 + 0j), np.full(len(f), 0.7 - 0.1j)
    # Only port 1 was excited: S12 comes from S21, S22 stays unmeasured.
    path = ts.write_touchstone(f, {1: {1: s11, 2: s21}}, [1, 2], [50.0, 75.0], TMP / "only1.s2p")
    net = ts.read(path)
    assert np.array_equal(net.s[:, 0, 1], s21) and not net.s[:, 1, 1].any(), net.s[0]
    data = read_results(path)
    assert set(data["traces"]) == {"S11", "S12", "S21"} and data["tdr"]["port"] == 1, data["traces"].keys()
    assert data["z0_ports"] == [50.0, 75.0]
    # Only port 2 excited: Zin uses its 75 ohms, TDR uses S22.
    path = ts.write_touchstone(f, {2: {1: s21, 2: s11}}, [1, 2], [50.0, 75.0], TMP / "only2.s2p")
    data = read_results(path)
    assert set(data["traces"]) == {"S12", "S21", "S22"} and data["tdr"]["port"] == 2, data["traces"].keys()
    assert np.allclose(data["traces"]["S22"]["z_real"], 75.0 * 1.2 / 0.8), data["traces"]["S22"]["z_real"][:2]
    assert "resonance" not in data["markers"]  # it is an S11 marker


check("unexcited ports: reciprocity fill, their own reference impedance, TDR from a measured port",
      unexcited_ports)

print()
for status, label, detail in results:
    print(f"  {status}  {label}")
    if detail:
        print(f"        {detail}")
fails = sum(1 for s, _, _ in results if s == "FAIL")
print(f"\n{len(results) - fails}/{len(results)} passed")
raise SystemExit(1 if fails else 0)
