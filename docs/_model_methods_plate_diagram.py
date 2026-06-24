#!/usr/bin/env python3
"""
Plate diagram for the joint tissue-phenotype population-dynamics model
(model_methods.tex).

This diagram corresponds to the Gamma-Poisson generative model:
    M_{z.}     ~ Gamma(a_z, b_z)               (one row of the growth matrix M)
    xtilde_irc = x_irc / d^src_irs             (depth-rescaled source; deterministic)
    mu_irc     = xtilde_irc M                  (destination intensity vector)
    y_irc(z')  ~ Poisson(d_irs * mu_irc(z'))   over non-missing tissue sub-blocks

M is a non-negative growth matrix with free row sums (M_{zz'} = expected number
of destination-z' cells per source-z cell over one 8-week step); it is not
row-stochastic.

The observation side is drawn with explicit nesting so the data hierarchy is
visible:

    patient i
      └── forward step r in R_i          (r = source timepoint; dest = r+1)
            └── clonotype c in C_ir
                  └── tissue s in S          (per-tissue destination draw)

x_irc and mu_irc live in the clone plate (the source counts and the full
intensity vector are shared across tissues); the per-tissue destination draw
y_irc, its sequencing-depth exposure d_irs, and its missingness flag m_irs live
in the innermost tissue sub-plate. The deterministic depth-rescaling
xtilde = x / d^src is folded into the x -> mu edge, so x_irc stays the observed
raw-count node and xtilde has no node of its own.

The shared prior block ((a_z, b_z) -> M_{z.}) sits OUTSIDE the patient plate: a
single growth matrix M is shared across all patients and steps.

The variational quantities used for inference (shape/rate, responsibilities r,
allocations xi) are intentionally not shown here; they are introduced separately
in the Inference section of model_methods.tex.

Outputs: docs/figures/plate_joint_transition.{pdf,png}

Dependency note: this needs the plate-notation library `daft` (daft-pgm.org),
whose import name collides with the unrelated `daft` dataframe engine of the
same name. Install the PGM one in an isolated env, e.g.:

    python -m venv /tmp/daftpgm
    /tmp/daftpgm/bin/pip install daft-pgm matplotlib
    /tmp/daftpgm/bin/python docs/_model_methods_plate_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc

try:
    import daft

    if not hasattr(daft, "PGM"):
        raise ImportError
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script needs the plate-diagram library 'daft-pgm' "
        "(imports as 'daft'). Note that the PyPI name 'daft' now refers to an "
        "unrelated dataframe library. Install with:\n\n"
        "    pip install daft-pgm\n"
    ) from exc

rc("font", family="serif", size=11)
rc("text", usetex=False)  # mathtext -> no LaTeX install required

# Figures live alongside the rest of the docs figures, regardless of cwd.
OUT = Path(__file__).resolve().parent / "figures"


def build_plate(out_dir: Path = OUT) -> None:
    # Gamma-Poisson population model.  Nesting: patient > step > clone > tissue.
    #   * prior block (a_z,b_z) -> M_{z.} sits in the top z-plate, outside the
    #     patient plate; (a,b) -> M and M -> mu are clean verticals;
    #   * x -> mu is horizontal (the deterministic depth-rescaling
    #     xtilde = x / d^src is folded into this edge, so xtilde has no node);
    #   * mu -> y is horizontal and crosses into the innermost tissue sub-plate;
    #   * within the tissue sub-plate the destination draw y is fed by its
    #     sequencing-depth exposure d_irs and missingness flag m_irs.
    # x_irc and mu_irc stay in the clone plate (shared across tissues); only the
    # destination draw {y, d, m} is per-tissue.
    pgm = daft.PGM(
        shape=(8.6, 7.2),
        origin=(0.0, 0.0),
        observed_style="inner",
        grid_unit=1.6,
        node_unit=1.15,
        directed=True,
    )

    # ---------------------------------------------------------------------
    # Gamma prior over growth-matrix rows  (top plate, indexed by z).
    # Shared across all patients and steps -> drawn OUTSIDE the patient plate.
    # (a_z,b_z), M_{z.} and mu share the x of mu, so (a,b) -> M and M -> mu are
    # both clean vertical edges.
    # ---------------------------------------------------------------------
    pgm.add_node("ab", r"$(a_z, b_z)$", 4.00, 6.40, fixed=True)
    pgm.add_node("Mrow", r"$M_{z\cdot}$", 4.00, 5.40)

    pgm.add_edge("ab", "Mrow")       # vertical

    pgm.add_plate(
        [2.90, 4.85, 2.20, 2.10],
        label=r"$z \in \mathcal{Z}$",
        shift=-0.12,
        rect_params={"ec": "k", "fc": "none"},
    )

    # ---------------------------------------------------------------------
    # Observation model.  mu_irc = xtilde_irc M is the deterministic intensity
    # (open circle); the depth-rescaling xtilde = x / d^src is folded into the
    # x -> mu edge, so x_irc stays the observed raw-count node.
    # ---------------------------------------------------------------------
    pgm.add_node("x", r"$x_{irc}$", 2.10, 2.25, observed=True)
    pgm.add_node("mu", r"$\mu_{irc}$", 4.00, 2.25)
    pgm.add_node("y", r"$y_{irc}$", 5.90, 2.25, observed=True)
    # Per-tissue destination Poisson exposure (depth) and missingness gate.
    # offset lifts the labels clear of the dots.
    pgm.add_node("d", r"$d_{irs}$", 5.50, 3.10, fixed=True, offset=(0, 11))
    pgm.add_node("m", r"$m_{irs}$", 6.30, 3.10, fixed=True, offset=(0, 11))

    pgm.add_edge("x", "mu")          # horizontal (depth-rescaling folded in)
    pgm.add_edge("Mrow", "mu")       # vertical
    pgm.add_edge("mu", "y")          # horizontal, crosses into the tissue plate
    pgm.add_edge("d", "y")
    pgm.add_edge("m", "y")

    # Tissue sub-plate (innermost): the per-tissue destination draw {y, d, m}.
    pgm.add_plate(
        [4.95, 1.65, 2.05, 2.05],
        label=r"$s \in \mathcal{S}$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Clone plate: wraps x, mu and the tissue sub-plate.
    pgm.add_plate(
        [1.55, 1.30, 5.65, 2.65],
        label=r"$c \in \mathcal{C}_{ir}$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Step plate, nested inside patient.
    pgm.add_plate(
        [1.20, 0.70, 6.15, 3.45],
        label=r"$r \in \mathcal{R}_i$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Patient plate (outermost on the observation side; M lives outside it).
    pgm.add_plate(
        [0.85, 0.10, 6.85, 4.45],
        label=r"$i = 1,\ldots,I$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # ---------------------------------------------------------------------
    # Render and save
    # ---------------------------------------------------------------------
    pgm.render()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "plate_joint_transition.pdf"
    out_png = out_dir / "plate_joint_transition.png"

    pgm.savefig(str(out_pdf))
    pgm.figure.savefig(str(out_png), dpi=300, bbox_inches="tight")
    plt.close(pgm.figure)
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    build_plate()
