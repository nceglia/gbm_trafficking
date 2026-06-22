"""Plate diagram for `trafficking.model.transition_model`.

Designed for main-figure use: clean layout, math typesetting, and a
restrained palette consistent with `trafficking/style.py`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


# --- Palette --------------------------------------------------------------
_INK = "#1f2933"          # primary stroke / text
_INK_SOFT = "#52606d"     # secondary stroke
_MUTED = "#8895a3"        # captions, plate-index labels
_PLATE_EDGE = "#000000"
_LATENT_FC = "#ffffff"
_OBSERVED_FC = "#b0b5bb"  # shaded gray (formerly CSF blue)
_GIVEN_FC = "#f1f2f4"


# (facecolor, edgecolor, linewidth, linestyle)
_NODE_STYLES: dict[str, tuple[str, str, float, object]] = {
    "latent":   (_LATENT_FC,   _INK,      1.5, "-"),
    "observed": (_OBSERVED_FC, _INK,      1.5, "-"),
    "given":    (_GIVEN_FC,    _INK_SOFT, 1.1, "-"),
    "det":      (_LATENT_FC,   _INK_SOFT, 1.2, (0, (3, 2))),
}


def _node(
    ax,
    x: float,
    y: float,
    symbol: str,
    *,
    kind: str = "latent",
    r: float = 0.42,
    caption: Optional[str] = None,
    symbol_size: float = 15,
    caption_size: float = 9.0,
) -> tuple[float, float, float]:
    fc, ec, lw, ls = _NODE_STYLES[kind]
    ax.add_patch(
        Circle(
            (x, y), r,
            facecolor=fc, edgecolor=ec,
            linewidth=lw, linestyle=ls,
            zorder=3,
        )
    )
    ax.text(
        x, y, symbol,
        ha="center", va="center",
        fontsize=symbol_size, color=_INK, zorder=4,
    )
    if caption:
        ax.text(
            x, y - r - 0.20, caption,
            ha="center", va="top",
            fontsize=caption_size, color=_MUTED, style="italic", zorder=4,
        )
    return (x, y, r)


def _arrow(ax, src: tuple[float, float, float], dst: tuple[float, float, float]) -> None:
    """Arrow from circle edge to circle edge."""
    sx, sy, sr = src
    dx, dy, dr = dst
    vx, vy = dx - sx, dy - sy
    dist = math.hypot(vx, vy)
    if dist < 1e-6:
        return
    ux, uy = vx / dist, vy / dist
    a = (sx + ux * sr, sy + uy * sr)
    b = (dx - ux * dr, dy - uy * dr)
    ax.add_patch(
        FancyArrowPatch(
            a, b,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=_INK,
            shrinkA=0, shrinkB=0,
            connectionstyle="arc3,rad=0.0",
            zorder=2,
        )
    )


def _plate(
    ax,
    x: float, y: float, w: float, h: float,
    label: str,
    *,
    label_pos: Literal["tl", "tr", "bl", "br"] = "tl",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="none",
            edgecolor=_PLATE_EDGE,
            linewidth=1.0,
            alpha=1.0,
            zorder=1,
        )
    )
    pad = 0.16
    if label_pos == "tl":
        tx, ty, ha, va = x + pad, y + h - pad, "left", "top"
    elif label_pos == "tr":
        tx, ty, ha, va = x + w - pad, y + h - pad, "right", "top"
    elif label_pos == "bl":
        tx, ty, ha, va = x + pad, y + pad, "left", "bottom"
    else:
        tx, ty, ha, va = x + w - pad, y + pad, "right", "bottom"
    ax.text(
        tx, ty, label,
        ha=ha, va=va,
        fontsize=10, color=_MUTED, style="italic", zorder=4,
    )


def _legend(ax, *, x0: float, y: float, gap: float = 1.85) -> None:
    entries = [
        ("latent",        "latent"),
        ("observed",      "observed"),
        ("fixed / input", "given"),
        ("deterministic", "det"),
    ]
    r = 0.16
    x = x0
    for label, kind in entries:
        fc, ec, lw, ls = _NODE_STYLES[kind]
        ax.add_patch(
            Circle(
                (x, y), r,
                facecolor=fc, edgecolor=ec,
                linewidth=lw, linestyle=ls,
                zorder=3,
            )
        )
        ax.text(
            x + r + 0.12, y, label,
            ha="left", va="center",
            fontsize=10, color=_INK_SOFT, zorder=4,
        )
        x += gap


def render_transition_plate(
    *,
    hierarchical: bool = True,
    include_dest_fracs: bool = True,
    title: Optional[str] = "Hierarchical Dirichlet transition model",
    out: Optional[str | Path] = None,
    formats: Iterable[Literal["png", "pdf", "svg"]] = ("png",),
    dpi: int = 300,
    show_legend: bool = True,
) -> list[Path]:
    """Render a publication-quality plate diagram for `transition_model`.

    Generative process (mirrors `trafficking.model.transition_model`):

        d                                              empirical dest. composition
        α  ∝  d · K   (or symmetric)                   Dirichlet prior
        T*_k     ~ Dirichlet(α)                        global rows
        κ        ~ Gamma(5, 1)                         row concentration
        T_{p,k}  ~ Dirichlet(κ · T*_k)                 patient rows
        π_c      =  θ_c · T_{patient(c), :}            (deterministic)
        y_c      ~  Multinomial(n_c, π_c)              observed dest. counts
    """
    fig, ax = plt.subplots(figsize=(11, 7.1))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    if title:
        ax.text(
            5.5, 6.85, title,
            ha="center", va="top",
            fontsize=14, color=_INK, fontweight="medium",
        )

    # ── Plates (drawn behind nodes) ──────────────────────────────────────
    _plate(ax, 2.05, 2.40, 1.65, 2.40, r"$K$", label_pos="tl")
    if hierarchical:
        _plate(ax, 4.90, 2.15, 2.45, 2.85, r"$P$", label_pos="tl")
        _plate(ax, 5.25, 2.50, 1.85, 2.20, r"$K$", label_pos="tl")
    _plate(ax, 7.80, 1.00, 1.90, 5.15, r"$N$", label_pos="tl")

    # ── Nodes ────────────────────────────────────────────────────────────
    nodes: dict[str, tuple[float, float, float]] = {}
    y_top = 6.00

    if include_dest_fracs:
        nodes["d"] = _node(
            ax, 0.70, y_top, r"$d$",
            kind="given", r=0.30,
            caption="empirical baseline",
            symbol_size=12.5,
            caption_size=8.5,
        )
    nodes["alpha"] = _node(
        ax, 2.875, y_top, r"$\alpha$",
        kind="given", r=0.30,
        caption="Dirichlet prior",
        symbol_size=13,
        caption_size=8.5,
    )
    if hierarchical:
        nodes["kappa"] = _node(
            ax, 6.125, y_top, r"$\kappa$",
            kind="latent", r=0.36,
            caption="row concentration",
            symbol_size=14,
            caption_size=8.5,
        )

    nodes["T_global"] = _node(
        ax, 2.875, 3.55, r"$T^{*}_{\,k}$",
        kind="latent", r=0.48,
        caption="global transitions",
        symbol_size=15,
    )
    if hierarchical:
        nodes["T_patient"] = _node(
            ax, 6.125, 3.55, r"$T_{p,k}$",
            kind="latent", r=0.48,
            caption="patient transitions",
            symbol_size=15,
        )

    nodes["theta"] = _node(
        ax, 8.75, 5.30, r"$\theta_{c}$",
        kind="given", r=0.36,
        caption="source composition",
        symbol_size=14,
    )
    nodes["pi"] = _node(
        ax, 8.75, 3.55, r"$\pi_{c}$",
        kind="det", r=0.36,
        caption="expected destinations",
        symbol_size=14,
    )
    nodes["y"] = _node(
        ax, 8.75, 1.90, r"$y_{c}$",
        kind="observed", r=0.40,
        caption="observed counts",
        symbol_size=14,
    )

    # ── Edges ────────────────────────────────────────────────────────────
    if include_dest_fracs:
        _arrow(ax, nodes["d"], nodes["alpha"])
    _arrow(ax, nodes["alpha"], nodes["T_global"])
    if hierarchical:
        _arrow(ax, nodes["T_global"], nodes["T_patient"])
        _arrow(ax, nodes["kappa"], nodes["T_patient"])
        _arrow(ax, nodes["T_patient"], nodes["pi"])
    else:
        _arrow(ax, nodes["T_global"], nodes["pi"])
    _arrow(ax, nodes["theta"], nodes["pi"])
    _arrow(ax, nodes["pi"], nodes["y"])

    # ── Legend ───────────────────────────────────────────────────────────
    if show_legend:
        _legend(ax, x0=1.50, y=0.30)

    # ── Output ───────────────────────────────────────────────────────────
    outputs: list[Path] = []
    if out is not None:
        base = Path(out)
        if base.parent and str(base.parent) != ".":
            base.parent.mkdir(parents=True, exist_ok=True)
        stem = base.with_suffix("")
        for fmt in formats:
            p = stem.with_suffix(f".{fmt}")
            fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.15,
                        facecolor="white")
            outputs.append(p)
    plt.close(fig)
    return outputs


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Render plate diagram for trafficking transition model."
    )
    p.add_argument("--out", required=True,
                   help="Output path stem (e.g. results/plate_transition_model).")
    p.add_argument("--pdf", action="store_true", help="Also write PDF.")
    p.add_argument("--svg", action="store_true", help="Also write SVG.")
    p.add_argument("--non-hierarchical", action="store_true",
                   help="Render the non-hierarchical model branch.")
    p.add_argument("--no-dest-fracs", action="store_true",
                   help="Omit dest_fracs -> alpha prior-centre annotation.")
    p.add_argument("--no-legend", action="store_true",
                   help="Suppress the bottom legend.")
    p.add_argument("--no-title", action="store_true",
                   help="Render without the top title.")
    args = p.parse_args(argv)

    fmts: list[str] = ["png"]
    if args.pdf:
        fmts.append("pdf")
    if args.svg:
        fmts.append("svg")

    hierarchical = not args.non_hierarchical
    include_dest_fracs = not args.no_dest_fracs
    title = (
        None if args.no_title
        else ("Hierarchical Dirichlet transition model"
              if hierarchical else "Dirichlet transition model")
    )

    render_transition_plate(
        hierarchical=hierarchical,
        include_dest_fracs=include_dest_fracs,
        title=title,
        out=args.out,
        formats=fmts,
        show_legend=not args.no_legend,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
