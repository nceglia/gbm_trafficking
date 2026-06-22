"""Collapsed 4-node plate diagram for the trafficking transition model (Daft).

This is the *skeleton* of `trafficking.model.transition_model`: collapse the
temporal chain to a single x_t -> x_{t+1} transition and drop the hierarchy,
leaving just the global transition matrix and the two endpoints.

    alpha            Dirichlet prior on the rows of T
    T      ~ Dir(alpha)        global transition matrix (K x K), the inference target
    x_t                        observed source state (tissue-phenotype count vector)
    x_{t+1} ~ T(x_t)           observed destination state

Everything is observed *per transition*, indexed by
(patient p, clone c, timepoint-pair t->t', tissue-pair a->b) — the plate.

For the full implemented model (kappa, T_patient, pi_c, y_c) see
`trafficking.plate.render_transition_plate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional

import matplotlib

matplotlib.use("Agg")

import daft  # noqa: E402


def render_simple_plate(
    *,
    out: str | Path = "results/plate_transition_collapsed",
    formats: Iterable[Literal["png", "pdf", "svg"]] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    pgm = daft.PGM(dpi=dpi, alternate_style="outer")

    # --- global parameters (outside the plate) -------------------------------
    # alpha: fixed hyperparameter (small dot), feeds the Dirichlet prior on T.
    pgm.add_node("alpha", r"$\alpha$", 2.3, 3.7, fixed=True, offset=(0, 8))
    # T: the global transition matrix — the latent we infer.
    pgm.add_node("T", r"$T$", 2.3, 2.8, scale=1.5)

    # --- per-transition states (inside the plate) ----------------------------
    pgm.add_node("x_now", r"$x_t$", 1.5, 1.4, observed=True, scale=1.5)
    pgm.add_node("x_next", r"$x_{t+1}$", 3.1, 1.4, observed=True, scale=1.5)

    # --- edges ---------------------------------------------------------------
    pgm.add_edge("alpha", "T")          # alpha -> T   (prior)
    pgm.add_edge("x_now", "x_next")     # x_t   -> x_{t+1}  (carried-forward cells)
    pgm.add_edge("T", "x_next")         # T     -> x_{t+1}  (applied transition)

    # --- plate over every observed transition --------------------------------
    pgm.add_plate(
        [0.7, 0.6, 3.2, 1.55],
        label=r"patient $p$, clone $c$, tp-pair $t\!\to\!t'$, tissue-pair $a\!\to\!b$",
        position="bottom left",
        shift=-0.1,
    )

    pgm.render()

    base = Path(out)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.with_suffix("")
    outputs: list[Path] = []
    for fmt in formats:
        p = stem.with_suffix(f".{fmt}")
        pgm.savefig(p)
        outputs.append(p)
    return outputs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/plate_transition_collapsed",
                    help="Output path stem.")
    ap.add_argument("--pdf", action="store_true", help="Also write PDF.")
    ap.add_argument("--svg", action="store_true", help="Also write SVG.")
    args = ap.parse_args()

    fmts: list[str] = ["png"]
    if args.pdf:
        fmts.append("pdf")
    if args.svg:
        fmts.append("svg")

    paths = render_simple_plate(out=args.out, formats=fmts)
    print("Wrote:", *[str(p) for p in paths], sep="\n  ")
