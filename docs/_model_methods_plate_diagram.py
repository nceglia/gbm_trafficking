#!/usr/bin/env python3
"""
Plate diagram for the joint tissue-phenotype transition model (model_methods.tex).

This diagram corresponds to the generative model:
    T_z ~ Dirichlet(alpha_z)
    theta_iqc = Normalize(x_iqc)
    pi_iqc    = theta_iqc T
    y_iqc     ~ Multinomial(N_dst_iqc, pi_iqc)

(Biologically implausible transitions may be masked out in the implementation;
that is a fixed preprocessing choice, not part of the model form, so it is not
shown here.)

The observation side is drawn with explicit nesting so the data hierarchy is
visible rather than hidden in a flattened index (Option B: theta has no node of
its own; the normalization theta = Normalize(x) is folded into the x -> pi edge):

    patient i
      └── forward step r in R_i        (r = source timepoint; dest = r+1)
            └── clonotype c in C_ir
                  └── x_irc -> pi_irc -> y_irc

The shared transition block (alpha_z -> T_z) sits OUTSIDE the patient plate: a
single matrix T is shared across all patients and steps. The patient plate is
present because observations are grouped by patient, not because T varies by
patient. For algebra/VI the triple can be flattened to a single index
j = (i, r, c).

The variational quantities used for inference (lambda, r, xi) are intentionally
not shown here; they are introduced separately in the Computational Inference
section of model_methods.tex.

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
    # Option B layout.  Every edge is horizontal, vertical, or 45 degrees:
    #   * the observation row x -> pi -> y is horizontal (the deterministic
    #     normalization theta = Normalize(x) is folded into the x -> pi edge, so
    #     theta has no node of its own);
    #   * the shared T sits directly above pi, so T -> pi is a clean vertical;
    #   * alpha sits directly above T, so alpha -> T is vertical;
    #   * N^dst sits directly above y, so N^dst -> y is vertical.
    # The three nested observation plates (patient > step > clone) get generous,
    # even margins so each plate's enumeration label sits in a clear band.
    pgm = daft.PGM(
        shape=(7.8, 7.2),
        origin=(0.0, 0.0),
        observed_style="inner",
        grid_unit=1.6,
        node_unit=1.15,
        directed=True,
    )

    # ---------------------------------------------------------------------
    # Global / row-level transition prior  (top plate, indexed by z).
    # Shared across all patients and steps -> drawn OUTSIDE the patient plate.
    # alpha_z, T_z and pi share the x of pi, so alpha -> T and T -> pi are both
    # clean vertical edges.
    # ---------------------------------------------------------------------
    pgm.add_node("alpha", r"$\alpha_z$", 4.00, 6.40, fixed=True)
    pgm.add_node("Trow", r"$T_{z\cdot}$", 4.00, 5.40)

    pgm.add_edge("alpha", "Trow")    # vertical

    pgm.add_plate(
        [2.90, 4.85, 2.20, 2.10],
        label=r"$z \in \mathcal{Z}$",
        shift=-0.12,
        rect_params={"ec": "k", "fc": "none"},
    )

    # ---------------------------------------------------------------------
    # Observation model with explicit patient / step / clone nesting.
    # pi_irc = theta_irc T is the deterministic pushforward; it keeps a node
    # (where the data x and the latent T meet) while theta does not.
    # ---------------------------------------------------------------------
    pgm.add_node("x", r"$x_{irc}$", 2.10, 2.25, observed=True)
    pgm.add_node("pi", r"$\pi_{irc}$", 4.00, 2.25)
    # offset lifts the label clear of the dot (daft only nudges fixed-node
    # labels up by 6 pt by default, which lets the irc subscript touch the dot).
    pgm.add_node("Ndst", r"$N^{\mathrm{dst}}_{irc}$", 5.90, 3.05, fixed=True, offset=(0, 11))
    pgm.add_node("y", r"$y_{irc}$", 5.90, 2.25, observed=True)

    pgm.add_edge("x", "pi")          # horizontal (normalization folded in)
    pgm.add_edge("Trow", "pi")       # vertical
    pgm.add_edge("pi", "y")          # horizontal
    pgm.add_edge("Ndst", "y")        # vertical

    # Clone plate (innermost): wraps x, pi, y and the per-clone total N^dst_irc.
    pgm.add_plate(
        [1.55, 1.40, 4.90, 2.50],
        label=r"$c \in \mathcal{C}_{ir}$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Step plate, nested inside patient.
    pgm.add_plate(
        [1.20, 0.80, 5.60, 3.40],
        label=r"$r \in \mathcal{R}_i$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Patient plate (outermost on the observation side; T lives outside it).
    pgm.add_plate(
        [0.85, 0.20, 6.30, 4.30],
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
