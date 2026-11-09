"""The dielectric-loss choice, shared by the modeler's setup and the layout's board settings."""

LOSS_MODELS = [("wideband", "Wideband — tan δ right across the sweep, slower"),
               ("centre", "Band centre — no extra time, tan δ right at one frequency")]

LOSS_TOOLTIP = (
    "Wideband holds each dielectric's tan δ within about 5 % across the sweep: a conductivity plus as few "
    "Debye poles as that takes (one for a narrow band, two for 0.2–6 GHz). openEMS updates every pole in every "
    "cell of the material at several times the cost of the cell itself, so in the 3D Layout only the "
    "dielectrics the ports' lines run in get poles; the others keep tan δ at the band centre, which costs "
    "nothing and carries little of the field (the core between two ground planes, say).\n\n"
    "Band centre turns tan δ into a conductivity everywhere: no extra time, but a conductivity's loss tangent "
    "falls as 1/f, so it is exact at the excitation centre only. On a 0.2–6 GHz sweep its loss is about 6× "
    "too high at 0.5 GHz and half what it should be at 6 GHz."
)
