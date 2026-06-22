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
visible rather than hidden in a flattened index:

    patient i
      └── forward interval q in Q_i
            └── clonotype c in C_iq
                  └── x_iqc -> theta_iqc -> pi_iqc -> y_iqc

The shared transition block (alpha_z -> T_z) sits OUTSIDE the patient plate: a
single matrix T is shared across all patients and intervals. The patient plate
is present because observations are grouped by patient, not because T varies by
patient. For algebra/VI the triple can be flattened to a single index
j = (i, q, c).

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
    # Geometry is laid out on a regular grid so that every edge is purely
    # horizontal, vertical, or 45 degrees:
    #   * the observation chain x -> theta -> pi -> y is one horizontal row;
    #   * the shared T sits directly above pi, so T -> pi is a clean vertical;
    #   * N^dst sits directly above y, so N^dst -> y is vertical;
    #   * the priors M, alpha sit one unit up and one unit out from T, so
    #     M -> T and alpha -> T are symmetric 45-degree diagonals.
    # The three nested observation plates (patient > interval > clone) get
    # generous, even margins (~0.6 grid units between successive borders) so
    # each plate's enumeration label sits in a clear band and is never crossed
    # by an inner border.
    pgm = daft.PGM(
        shape=(8.7, 7.2),
        origin=(0.0, 0.0),
        observed_style="inner",
        grid_unit=1.6,
        node_unit=1.15,
        directed=True,
    )

    # ---------------------------------------------------------------------
    # Global / row-level transition prior  (top plate, indexed by z).
    # Shared across all patients and intervals -> drawn OUTSIDE the patient
    # plate below. alpha_z, T_z and pi share the x of pi, so alpha -> T and
    # T -> pi are both clean vertical edges.
    # ---------------------------------------------------------------------
    pgm.add_node("alpha", r"$\alpha_z$", 5.15, 6.40, fixed=True)
    pgm.add_node("Trow", r"$T_{z}$", 5.15, 5.40)

    pgm.add_edge("alpha", "Trow")    # vertical

    pgm.add_plate(
        [4.05, 4.85, 2.20, 2.10],
        label=r"$z \in \mathcal{Z}$",
        shift=-0.12,
        rect_params={"ec": "k", "fc": "none"},
    )

    # ---------------------------------------------------------------------
    # Observation model with explicit patient / interval / clone nesting.
    # The pi node is kept compact ($\pi_{iqc}$); the relation
    # pi_iqc = theta_iqc T is carried by the theta->pi and T->pi edges (and
    # stated in the caption), which avoids a wide label overrunning its arrows.
    # ---------------------------------------------------------------------
    pgm.add_node("x", r"$x_{iqc}$", 1.95, 2.25, observed=True)
    pgm.add_node("theta", r"$\theta_{iqc}$", 3.55, 2.25)
    pgm.add_node("pi", r"$\pi_{iqc}$", 5.15, 2.25)
    # offset lifts the label clear of the dot (daft only nudges fixed-node
    # labels up by 6 pt by default, which lets the iqc subscript touch the dot).
    pgm.add_node("Ndst", r"$N^{\mathrm{dst}}_{iqc}$", 6.75, 3.05, fixed=True, offset=(0, 11))
    pgm.add_node("y", r"$y_{iqc}$", 6.75, 2.25, observed=True)

    pgm.add_edge("x", "theta")       # horizontal
    pgm.add_edge("theta", "pi")      # horizontal
    pgm.add_edge("Trow", "pi")       # vertical
    pgm.add_edge("pi", "y")          # horizontal
    pgm.add_edge("Ndst", "y")        # vertical

    # Clone plate (innermost): wraps x, theta, pi, y and the per-clone total
    # N^dst_iqc.
    pgm.add_plate(
        [1.40, 1.40, 5.90, 2.50],
        label=r"$c \in \mathcal{C}_{iq}$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Interval plate, nested inside patient.
    pgm.add_plate(
        [1.05, 0.80, 6.60, 3.40],
        label=r"$q \in \mathcal{Q}_i$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Patient plate (outermost on the observation side; T lives outside it).
    pgm.add_plate(
        [0.70, 0.20, 7.30, 4.30],
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
