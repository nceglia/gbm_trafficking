"""Minimal plate diagram for the basic (Bayesian-led) trafficking problem.

Generative model (hierarchical phenotype-transition, per directed tissue
pair t1 -> t2 and per lineage):

    kappa            ~ Gamma(5, 1)                          (global pooling)
    T^g_k            ~ Dirichlet(alpha)                     (global row, source state k)
    T^p_k            ~ Dirichlet(kappa * T^g_k)             (patient p, source state k)
    pi_c = theta_c . T^{p(c)}                               (deterministic mean)
    n^dst_c          ~ Multinomial(N_c, pi_c)               (observed dest counts)

theta_c (source phenotype distribution) and N_c (dest cell total) are fixed
data; alpha is a fixed hyperparameter centred on the destination composition.

Dependency note: this needs the plate-notation library `daft` (Foreman-Mackey,
daft-pgm.org), whose import name `daft` collides with the unrelated dataframe
engine of the same import name. Install the PGM one and run in an isolated env:

    python -m venv /tmp/daftpgm
    /tmp/daftpgm/bin/pip install daft-pgm matplotlib
    /tmp/daftpgm/bin/python docs/figures/plate_basic.py

Outputs: docs/figures/plate_basic.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import rc

import daft

if not hasattr(daft, "PGM"):
    raise SystemExit(
        "The imported `daft` has no PGM (it is the dataframe engine, not the "
        "plate-notation library). Install `daft-pgm` in an isolated venv — see "
        "the module docstring."
    )

rc("font", family="serif", size=11)
rc("text", usetex=False)  # mathtext -> no LaTeX install required

OUT = Path(__file__).resolve().parent

pgm = daft.PGM()

# ---- nodes ---------------------------------------------------------------
# fixed hyperparameters / data: small dots ; observed data: shaded
pgm.add_node("alpha", r"$\alpha$", 0.0, 3.0, fixed=True)
pgm.add_node("kappa", r"$\kappa$", 2.5, 4.55)
pgm.add_node("Tg", r"$T^{g}_{k}$", 1.1, 3.0)
pgm.add_node("Tp", r"$T^{p}_{k}$", 2.5, 3.0)
pgm.add_node("theta", r"$\theta_c$", 4.3, 3.55, observed=True)
pgm.add_node("y", r"$n^{\mathrm{dst}}_{c}$", 4.3, 2.55, observed=True, scale=1.25)
pgm.add_node("N", r"$N_c$", 5.5, 2.55, fixed=True)

# ---- edges ---------------------------------------------------------------
pgm.add_edge("alpha", "Tg")
pgm.add_edge("kappa", "Tp")
pgm.add_edge("Tg", "Tp")
pgm.add_edge("Tp", "y")
pgm.add_edge("theta", "y")
pgm.add_edge("N", "y")

# ---- plates --------------------------------------------------------------
pgm.add_plate([0.55, 2.35, 1.1, 1.3], label=r"$k = 1 \ldots K$",
              fontsize=9, position="bottom left")
pgm.add_plate([1.95, 1.8, 4.15, 2.25], label=r"patient $p = 1 \ldots P$",
              fontsize=9, position="bottom left")
pgm.add_plate([2.05, 2.35, 0.9, 1.3], label=r"$k = 1 \ldots K$",
              fontsize=9, position="bottom left")
pgm.add_plate([3.75, 1.95, 2.2, 1.95], label=r"clone $c \in p$",
              fontsize=9, position="bottom left")

pgm.render()
pgm.figure.savefig(OUT / "plate_basic.png", dpi=200, bbox_inches="tight")
pgm.figure.savefig(OUT / "plate_basic.pdf", bbox_inches="tight")
print(f"wrote {OUT/'plate_basic.png'} and .pdf")
