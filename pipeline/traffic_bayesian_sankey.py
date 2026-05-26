# %%
"""Bayesian transition matrices for all 9 directed tissue edges + Sankey plots.

For every (lineage ∈ {CD8, CD4}, directed edge (t1, t2)) we fit the
hierarchical Dirichlet transition model in ``trafficking.model`` /
``inference`` and render the posterior ``T_global`` matrix two ways:

  1. As a plotly Sankey (one per edge) — per-edge interactive HTML plus
     a composed 3×3 PNG/PDF (rendered through kaleido).
  2. As a matplotlib K×K heatmap grid (3×3), one per lineage.

Same-tissue edges (PBMC→PBMC, CSF→CSF, TP→TP) capture within-tissue
phenotype switches at consecutive timepoints; cross-tissue edges capture
the inter-compartment trajectory.

Outputs go to ``results/06g_bayesian_sankey/``.
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_ORDER,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TISSUE_COLORS,
)

from trafficking.inference import run_inference  # noqa: E402
from trafficking.data import _clean_tcr  # noqa: E402


# %%
# ---- Config ----
from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
OUT_DIR = REPO_ROOT / "results" / "06g_bayesian_sankey"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SANKEY_DIR = OUT_DIR / "sankey_individual"
SANKEY_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = ("PBMC", "CSF", "TP")
EDGES = [
    ("PBMC", "PBMC"), ("CSF", "CSF"), ("TP", "TP"),
    ("PBMC", "CSF"), ("PBMC", "TP"), ("CSF", "PBMC"),
    ("CSF", "TP"),  ("TP", "PBMC"), ("TP", "CSF"),
]
LINEAGES = ["CD8", "CD4"]
N_STEPS = 2000
LR = 0.01
N_SAMPLES = 1000
HIERARCHICAL = True

LINK_THRESHOLD = 0.02  # Sankey link cutoff
HEATMAP_VMAX = 0.5
HEATMAP_ANNOT_THRESH = 0.05
DPI = 200


# %%
# ---- Try plotly / kaleido availability ----
HAS_PLOTLY = True
HAS_KALEIDO = True
try:
    import plotly.graph_objects as go  # noqa: F401
except ImportError:
    HAS_PLOTLY = False
    print("WARNING: plotly is not installed; Sankey HTML/PNGs will be skipped.")
    print("         install with: pip install plotly")

try:
    import kaleido  # noqa: F401
except ImportError:
    HAS_KALEIDO = False
    print("WARNING: kaleido is not installed; Sankey PNG composition will be "
          "skipped (HTMLs and matplotlib heatmap grid will still be produced).")
    print("         install with: pip install kaleido --break-system-packages")


# %%
# ---- Load adata ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")


# %%
# ---- Phenotype short-name → full-name mapping (used for palette lookup) ----
SHORT_TO_FULL = {}
for full in TCELL_PHENOTYPE_ORDER:
    short = (full
             .replace("CD8_Activated_", "")
             .replace("CD8_Quiescent_", "")
             .replace("CD4_", ""))
    SHORT_TO_FULL[short] = full


def _short_to_full(short):
    return SHORT_TO_FULL.get(short, short)


def _pheno_color(short):
    full = _short_to_full(short)
    return TCELL_PHENOTYPE_COLORS.get(full, "#9e9e9e")


def _pheno_label(short):
    full = _short_to_full(short)
    return TCELL_PHENOTYPE_LABELS.get(full, short)


def _compute_src_weights(adata, t1, lineage, short_phenotypes):
    """Empirical source phenotype fractions at tissue ``t1`` for the
    given ``lineage``, aligned to ``short_phenotypes``. Returns a (K,)
    numpy array summing to 1.0, or None if the source pool is empty."""
    obs = adata.obs
    level1 = obs["phenotype"].astype(str).map(
        lambda x: "CD8" if "CD8" in x else "CD4")
    mask = (obs["tissue"] == t1) & (level1 == lineage)
    counts = obs.loc[mask, "phenotype"].astype(str).value_counts()
    full_names = [_short_to_full(p) for p in short_phenotypes]
    weights = np.array([float(counts.get(fn, 0)) for fn in full_names],
                       dtype=float)
    s = float(weights.sum())
    if s <= 0:
        return None
    return weights / s


def _add_src_marginal_bar(ax, src_weights, phenos, size="5%", pad=0.05):
    """Append a thin colored bar to the LEFT of ``ax`` showing source
    phenotype prevalence (one row per phenotype, length proportional to
    ``src_weights[i]`` divided by the largest weight, colored by
    phenotype). Returns the new axis, or None if ``src_weights`` is
    None / all-zero."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if src_weights is None:
        return None
    weights = np.asarray(src_weights, dtype=float)
    wmax = float(np.nanmax(weights)) if weights.size else 0.0
    if not np.isfinite(wmax) or wmax <= 0:
        return None
    div = make_axes_locatable(ax)
    bar_ax = div.append_axes("left", size=size, pad=pad, sharey=ax)
    for i, w in enumerate(weights):
        bar_ax.barh(i, float(w) / wmax, height=0.85,
                    color=_pheno_color(phenos[i]),
                    edgecolor="none", align="center")
    bar_ax.set_xlim(0, 1.0)
    bar_ax.invert_xaxis()
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        bar_ax.spines[s].set_visible(False)
    return bar_ax


# %%
# ---- Pre-flight: shared-clone counts for each (lineage, edge) ----
def _preflight_clone_count(adata, t1, t2, lineage):
    """Number of clones eligible for extract_temporal_transitions(t1, t2)."""
    tcr = _clean_tcr(adata, lineage, clone_key="trb",
                     phenotype_key="phenotype", tissue_key="tissue")
    tcr = tcr[tcr["tissue"].isin([t1, t2])]
    if tcr.empty:
        return 0, 0
    if t1 == t2:
        # need ≥ 2 distinct timepoints for the same tissue per clone
        per_clone = tcr.groupby("trb", observed=True)["timepoint"].nunique()
        eligible = per_clone[per_clone >= 2].index
    else:
        groups = tcr.groupby("trb", observed=True)["tissue"].apply(set)
        eligible = groups[groups.apply(lambda s: t1 in s and t2 in s)].index
    sub = tcr[tcr["trb"].isin(eligible)]
    return int(len(eligible)), int(len(sub))


print("\n=== Pre-flight: shared-clone counts per (lineage, edge) ===")
pre_rows = []
for lin in LINEAGES:
    for t1, t2 in EDGES:
        n_clones, n_cells = _preflight_clone_count(adata, t1, t2, lin)
        edge_type = "same" if t1 == t2 else "cross"
        pre_rows.append({
            "lineage": lin, "t1": t1, "t2": t2,
            "edge_type": edge_type,
            "n_shared_clones": n_clones, "total_cells": n_cells,
        })
        print(f"  {lin:>3s} {t1:>4s} -> {t2:<4s} [{edge_type:<5s}]  "
              f"clones={n_clones:>5d}, cells={n_cells:>6d}")
pre_df = pd.DataFrame(pre_rows)
pre_df.to_csv(OUT_DIR / "preflight_counts.csv", index=False)
n_pairs = len(LINEAGES) * len(EDGES)
est_min = max(1, int(round(30 * n_pairs / 60)))
print(f"\nEstimated wall time: ~30s × {n_pairs} pairs ≈ {est_min} min.")


# %%
# ---- Step 1: hierarchical Bayesian inference for every (lineage, edge) ----
print("\n=== Step 1: hierarchical Dirichlet inference ===")
bayes_results = {}
summary_rows = []

for lin in LINEAGES:
    for t1, t2 in EDGES:
        tag = f"{lin}_{t1}_to_{t2}"
        edge_type = "same" if t1 == t2 else "cross"
        t0 = time.time()
        print(f"\n[{tag}] starting ({edge_type})...")
        try:
            result, data, phenotypes = run_inference(
                adata, t1, t2, lineage=lin, temporal=True,
                n_steps=N_STEPS, lr=LR, hierarchical=HIERARCHICAL,
                n_samples=N_SAMPLES, verbose=True,
            )
        except Exception as e:
            print(f"[{tag}] ERROR: {e}")
            summary_rows.append({
                "lineage": lin, "t1": t1, "t2": t2,
                "edge_type": edge_type,
                "n_clones": 0, "n_observations": 0,
                "kappa_mean": np.nan, "final_elbo": np.nan,
                "status": f"error: {e}",
                "elapsed_seconds": time.time() - t0,
            })
            continue

        elapsed = time.time() - t0
        if result is None or len(data) == 0:
            print(f"[{tag}] SKIP (no shared clones / empty data). "
                  f"Elapsed {elapsed:.1f}s")
            summary_rows.append({
                "lineage": lin, "t1": t1, "t2": t2,
                "edge_type": edge_type,
                "n_clones": int(data["clone"].nunique()) if len(data) else 0,
                "n_observations": int(len(data)),
                "kappa_mean": np.nan, "final_elbo": np.nan,
                "status": "skipped: no shared clones",
                "elapsed_seconds": elapsed,
            })
            continue

        T_mean = result["T_mean"]
        T_std = result["T_std"]
        kappa_mean = float(result.get("kappa_mean", np.nan))
        final_elbo = float(result["losses"][-1]) if result.get("losses") else float("nan")

        dest_fracs = result.get("dest_fracs")
        if dest_fracs is not None:
            dest_fracs_arr = (dest_fracs.detach().cpu().numpy()
                              if hasattr(dest_fracs, "detach")
                              else np.asarray(dest_fracs))
        else:
            dest_fracs_arr = None

        src_weights_arr = _compute_src_weights(adata, t1, lin, phenotypes)

        bayes_results[(lin, t1, t2)] = {
            "T_mean": T_mean, "T_std": T_std,
            "phenotypes": phenotypes,
            "kappa_mean": kappa_mean,
            "n_clones": int(data["clone"].nunique()),
            "n_observations": int(len(data)),
            "dest_fracs": dest_fracs_arr,
            "src_weights": src_weights_arr,
        }

        # Per-fit verification: dest_fracs vs T_diag vs enrichment.
        if dest_fracs_arr is not None:
            diag = np.diag(T_mean)
            enrich = diag / np.clip(dest_fracs_arr, 1e-9, None)
            print(f"  prior centre (dest_fracs @ {t2}, {lin}) | "
                  f"T_diag | enrichment (T_diag / dest_frac):")
            for p, f, d, e in zip(phenotypes, dest_fracs_arr, diag, enrich):
                print(f"    {p:>14s}  prior={f:.3f}  diag={d:.3f}  "
                      f"enrich={e:.2f}{'  *' if e > 2 else ''}")

        # persist per-edge T matrices
        T_df = pd.DataFrame(T_mean, index=phenotypes, columns=phenotypes)
        T_df.index.name = "source"
        T_df.columns.name = "destination"
        T_df.to_csv(OUT_DIR / f"T_{tag}.csv")
        Tstd_df = pd.DataFrame(T_std, index=phenotypes, columns=phenotypes)
        Tstd_df.to_csv(OUT_DIR / f"Tstd_{tag}.csv")

        if dest_fracs_arr is not None:
            diag = np.diag(T_mean)
            pd.DataFrame({
                "phenotype": phenotypes,
                "dest_frac": dest_fracs_arr,
                "T_diag": diag,
                "enrichment": diag / np.clip(dest_fracs_arr, 1e-9, None),
            }).to_csv(OUT_DIR / f"dest_prior_{tag}.csv", index=False)

        summary_rows.append({
            "lineage": lin, "t1": t1, "t2": t2,
            "edge_type": edge_type,
            "n_clones": int(data["clone"].nunique()),
            "n_observations": int(len(data)),
            "kappa_mean": kappa_mean, "final_elbo": final_elbo,
            "status": "ok",
            "elapsed_seconds": elapsed,
        })
        print(f"[{tag}] done. n_clones={data['clone'].nunique()}, "
              f"n_obs={len(data)}, kappa≈{kappa_mean:.2f}, "
              f"elapsed {elapsed:.1f}s")

fit_summary = pd.DataFrame(summary_rows)
fit_summary.to_csv(OUT_DIR / "fit_summary.csv", index=False)
print(f"\nWrote fit_summary.csv "
      f"({(fit_summary['status'] == 'ok').sum()}/{len(fit_summary)} ok)")


# %%
# ---- Step 2: plotly Sankey per (lineage, edge) ----
def _hex_to_rgba_str(hex_color, alpha):
    r, g, b, _ = to_rgba(hex_color, alpha=alpha)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha:.2f})"


def build_sankey_fig(T, phenotypes, t1, t2, src_weights=None):
    """Plotly Sankey for one (t1 -> t2) panel.

    ``src_weights`` is an optional (K,) array of source-phenotype
    fractions summing to 1. When provided, link values become
    ``src_weights[k] * T[k, j]`` (absolute flow), so plotly's automatic
    node sizing yields:
      - left node k height  = src_weights[k]
      - right node j height = sum_k src_weights[k] * T[k, j]
        (the marginal destination distribution implied by routing the
         source composition through T).
    When ``src_weights`` is None we fall back to uniform weights
    (1/K per source), which preserves the prior visual behavior of
    equal-height left bars.
    """
    K = len(phenotypes)
    if src_weights is None:
        weights = np.full(K, 1.0 / K)
    else:
        weights = np.asarray(src_weights, dtype=float)
        s = float(weights.sum())
        weights = weights / s if s > 0 else np.full(K, 1.0 / K)

    node_labels = [_pheno_label(p) for p in phenotypes] * 2
    node_colors = [_pheno_color(p) for p in phenotypes] * 2

    sources, targets, values, link_colors = [], [], [], []
    for i in range(K):
        for j in range(K):
            tij = float(T[i, j])
            if tij < LINK_THRESHOLD:
                continue
            v = float(weights[i]) * tij
            if v <= 0:
                continue
            sources.append(i)
            targets.append(K + j)
            values.append(v)
            link_colors.append(_hex_to_rgba_str(_pheno_color(phenotypes[i]),
                                                alpha=0.35))

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12, thickness=18,
            line=dict(color="black", width=0.5),
            label=node_labels, color=node_colors,
        ),
        link=dict(source=sources, target=targets,
                  value=values, color=link_colors),
    ))
    title_color = TISSUE_COLORS.get(t2, "black")
    fig.update_layout(
        title=dict(text=f"{t1} → {t2}",
                   font=dict(size=14, color=title_color)),
        font=dict(size=10),
        width=520, height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


sankey_png_paths = {}  # (lin, t1, t2) -> Path
if HAS_PLOTLY:
    print("\n=== Step 2: plotly Sankey diagrams ===")
    for (lin, t1, t2), bres in bayes_results.items():
        tag = f"{lin}_{t1}_to_{t2}"
        fig = build_sankey_fig(bres["T_mean"], bres["phenotypes"], t1, t2,
                               src_weights=bres.get("src_weights"))
        html_path = SANKEY_DIR / f"sankey_{tag}.html"
        fig.write_html(str(html_path),
                       include_plotlyjs="cdn", full_html=True)
        if HAS_KALEIDO:
            png_path = SANKEY_DIR / f"sankey_{tag}.png"
            try:
                fig.write_image(str(png_path), width=520, height=420, scale=2)
                sankey_png_paths[(lin, t1, t2)] = png_path
            except Exception as e:
                print(f"  [{tag}] PNG export failed: {e}")
    print(f"  wrote {len(bayes_results)} HTMLs and "
          f"{len(sankey_png_paths)} PNGs to {SANKEY_DIR}")


# %%
# ---- Sankey composition: 3×3 grid per lineage ----
def _set_row_col_labels(fig, axes, panel_size, title):
    """Annotate the 3×3 grid with Source/Dest tissue labels."""
    for r, src_tis in enumerate(TISSUES):
        axes[r, 0].set_ylabel(f"Source: {src_tis}", fontsize=11,
                               fontweight="bold",
                               color=TISSUE_COLORS.get(src_tis, "black"),
                               labelpad=12)
    for c, dst_tis in enumerate(TISSUES):
        axes[0, c].set_title(f"Dest: {dst_tis}", fontsize=11,
                              fontweight="bold",
                              color=TISSUE_COLORS.get(dst_tis, "black"),
                              pad=10)
    fig.suptitle(title, fontsize=14, fontweight="bold")


if HAS_PLOTLY and HAS_KALEIDO and sankey_png_paths:
    from PIL import Image

    print("\nComposing per-lineage Sankey grids (PNG via kaleido)...")
    for lin in LINEAGES:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        for idx, (t1, t2) in enumerate(EDGES):
            r, c = TISSUES.index(t1), TISSUES.index(t2)
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(False)
            png = sankey_png_paths.get((lin, t1, t2))
            if png is None or not png.exists():
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                         fontsize=14, color="dimgray",
                         transform=ax.transAxes)
                if t1 == t2:
                    ax.add_patch(Rectangle(
                        (0, 0), 1, 1, transform=ax.transAxes,
                        facecolor=TISSUE_COLORS.get(t1, "gray"),
                        alpha=0.05, zorder=0))
                continue
            img = Image.open(png)
            ax.imshow(img)
            if t1 == t2:
                ax.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    facecolor=TISSUE_COLORS.get(t1, "gray"),
                    alpha=0.05, zorder=-1))
        _set_row_col_labels(fig, axes, panel_size=None,
                             title=f"{lin} phenotype transitions — "
                                   f"Bayesian T_global")
        fig.tight_layout(rect=(0.02, 0, 1, 0.96))
        fig.savefig(OUT_DIR / f"sankey_{lin}.png",
                     dpi=DPI, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"sankey_{lin}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote sankey_{lin}.png/.pdf")
else:
    print("\nSankey grid composition: SKIPPED "
          "(plotly or kaleido missing, or no fits produced PNGs).")


# %%
# ---- Step 3: matplotlib heatmap grid per lineage ----
print("\n=== Step 3: matplotlib heatmap grids ===")
for lin in LINEAGES:
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    valid_im = None
    for (t1, t2) in EDGES:
        r, c = TISSUES.index(t1), TISSUES.index(t2)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        if t1 == t2:
            ax.add_patch(Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                facecolor=TISSUE_COLORS.get(t1, "gray"),
                alpha=0.05, zorder=-1))
        if (lin, t1, t2) not in bayes_results:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                     fontsize=12, color="dimgray",
                     transform=ax.transAxes)
            ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                          color=TISSUE_COLORS.get(t2, "black"), pad=4)
            continue
        bres = bayes_results[(lin, t1, t2)]
        T = bres["T_mean"]
        phenos = bres["phenotypes"]
        K = len(phenos)

        im = ax.imshow(T, cmap="YlOrRd", vmin=0.0, vmax=HEATMAP_VMAX,
                        aspect="equal")
        valid_im = im
        ax.set_xticks(range(K))
        ax.set_xticklabels([_pheno_label(p) for p in phenos],
                            rotation=60, ha="right", fontsize=6)
        for tick, p in zip(ax.get_xticklabels(), phenos):
            tick.set_color(_pheno_color(p))
        ax.set_yticks(range(K))
        ax.set_yticklabels([_pheno_label(p) for p in phenos], fontsize=6)
        for tick, p in zip(ax.get_yticklabels(), phenos):
            tick.set_color(_pheno_color(p))
            tick.set_fontweight("bold")
        ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                      color=TISSUE_COLORS.get(t2, "black"), pad=4)

        for i in range(K):
            for j in range(K):
                v = float(T[i, j])
                if v < HEATMAP_ANNOT_THRESH:
                    continue
                rgba = im.cmap(im.norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc = "white" if lum < 0.5 else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=5, color=tc, fontweight=weight)
        ax.tick_params(length=0)
        _add_src_marginal_bar(ax, bres.get("src_weights"), phenos)

    # row labels / column labels colored by tissue
    for r_idx, src_tis in enumerate(TISSUES):
        axes[r_idx, 0].set_ylabel(f"Source: {src_tis}", fontsize=11,
                                    fontweight="bold",
                                    color=TISSUE_COLORS.get(src_tis, "black"),
                                    labelpad=14)
    for c_idx, dst_tis in enumerate(TISSUES):
        # Place a header text above the column (above the existing title)
        axes[0, c_idx].text(0.5, 1.18, f"Dest: {dst_tis}",
                              transform=axes[0, c_idx].transAxes,
                              ha="center", va="bottom",
                              fontsize=11, fontweight="bold",
                              color=TISSUE_COLORS.get(dst_tis, "black"))
    fig.suptitle(f"{lin} phenotype transitions — Bayesian T_global heatmaps",
                  fontsize=14, fontweight="bold")
    if valid_im is not None:
        cbar = fig.colorbar(valid_im, ax=axes.ravel().tolist(),
                             fraction=0.025, pad=0.02)
        cbar.set_label("P(source → destination)", fontsize=10)
    fig.savefig(OUT_DIR / f"heatmap_{lin}.png",
                 dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"heatmap_{lin}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote heatmap_{lin}.png/.pdf")


# %%
# ---- Step 4: transition_highlights.csv + summary printout ----
print("\n=== Step 4: transition highlights ===")
highlight_rows = []
for (lin, t1, t2), bres in bayes_results.items():
    T = bres["T_mean"]
    phenos = bres["phenotypes"]
    K = len(phenos)
    diag_dominance = float(np.diag(T).mean())
    kappa = bres["kappa_mean"]

    off = T.copy()
    np.fill_diagonal(off, -np.inf)
    top_idx = np.dstack(
        np.unravel_index(np.argsort(-off.ravel())[:3], off.shape))[0]
    top_entries = []
    for ri, ci in top_idx:
        ri = int(ri); ci = int(ci)
        if not np.isfinite(off[ri, ci]):
            continue
        top_entries.append((phenos[ri], phenos[ci], float(T[ri, ci])))

    for rank, (s, d, p) in enumerate(top_entries, 1):
        highlight_rows.append({
            "lineage": lin, "t1": t1, "t2": t2,
            "edge_type": "same" if t1 == t2 else "cross",
            "rank": rank,
            "source_phenotype": s,
            "dest_phenotype": d,
            "probability": p,
            "diag_dominance": diag_dominance,
            "kappa_mean": kappa,
        })
    print(f"\n[{lin} {t1}->{t2}] diag dominance = {diag_dominance:.3f}, "
          f"kappa = {kappa:.2f}")
    for rank, (s, d, p) in enumerate(top_entries, 1):
        print(f"  #{rank} off-diag: {s:>14s} -> {d:<14s}  P = {p:.3f}")

highlights_df = pd.DataFrame(highlight_rows)
highlights_df.to_csv(OUT_DIR / "transition_highlights.csv", index=False)
print(f"\nWrote {OUT_DIR / 'transition_highlights.csv'} "
      f"({len(highlights_df)} rows)")


# %%
# ---- Step 5: enrichment heatmaps log2(T / dest_fracs) per lineage ----
print("\n=== Step 5: enrichment heatmaps (log2 of routing / marginal) ===")

ENRICH_VMIN, ENRICH_VMAX = -3.0, 3.0   # cap at 8x enrich / 0.125x depletion
ANNOT_THRESH = 0.5                     # |log2| > 0.5 ≈ >1.4x or <0.7x
BOLD_THRESH = 1.5                      # |log2| > 1.5 ≈ >2.8x or <0.35x
RARE_DEST_FLOOR = 0.01                 # dest_fracs[j] below this -> grayed column

global_enriched = []   # (lin, t1, t2, src_pheno, dst_pheno, log2_e)
global_depleted = []

for lin in LINEAGES:
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    valid_im = None

    for (t1, t2) in EDGES:
        r, c = TISSUES.index(t1), TISSUES.index(t2)
        ax = axes[r, c]
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        if t1 == t2:
            ax.add_patch(Rectangle(
                (0, 0), 1, 1, transform=ax.transAxes,
                facecolor=TISSUE_COLORS.get(t1, "gray"),
                alpha=0.05, zorder=-1))

        if (lin, t1, t2) not in bayes_results:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                     fontsize=12, color="dimgray",
                     transform=ax.transAxes)
            ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                          color=TISSUE_COLORS.get(t2, "black"), pad=4)
            continue

        bres = bayes_results[(lin, t1, t2)]
        T = bres["T_mean"]
        phenos = bres["phenotypes"]
        dest_fracs = bres.get("dest_fracs")
        if dest_fracs is None:
            ax.text(0.5, 0.5, "no dest_fracs", ha="center", va="center",
                     fontsize=11, color="dimgray", transform=ax.transAxes)
            ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                          color=TISSUE_COLORS.get(t2, "black"), pad=4)
            continue
        K = len(phenos)

        safe_dest = np.clip(np.asarray(dest_fracs, dtype=float), 1e-9, None)
        E = T / safe_dest[np.newaxis, :]
        log2E = np.log2(np.clip(E, 1e-9, None))

        # Persist per-edge log2(E) matrix.
        tag = f"{lin}_{t1}_to_{t2}"
        pd.DataFrame(log2E, index=phenos, columns=phenos) \
          .rename_axis(index="source", columns="destination") \
          .to_csv(OUT_DIR / f"enrichment_{tag}.csv")

        # Collect top-enriched / top-depleted entries for the summary.
        for i in range(K):
            for j in range(K):
                if safe_dest[j] < RARE_DEST_FLOOR:
                    continue
                v = float(log2E[i, j])
                rec = (lin, t1, t2, phenos[i], phenos[j], v)
                if v > BOLD_THRESH:
                    global_enriched.append(rec)
                if v < -BOLD_THRESH:
                    global_depleted.append(rec)

        im = ax.imshow(log2E, cmap="RdBu_r",
                        vmin=ENRICH_VMIN, vmax=ENRICH_VMAX,
                        aspect="equal")
        valid_im = im

        # Gray out columns where dest is essentially absent.
        rare_cols = np.where(np.asarray(dest_fracs) < RARE_DEST_FLOOR)[0]
        for j in rare_cols:
            ax.add_patch(Rectangle(
                (j - 0.5, -0.5), 1, K,
                facecolor="lightgray", edgecolor="none",
                alpha=0.55, zorder=2))

        # Annotate informative cells.
        for i in range(K):
            for j in range(K):
                if j in rare_cols:
                    continue
                v = float(log2E[i, j])
                if abs(v) < ANNOT_THRESH:
                    continue
                tc = "white" if abs(v) > 2.2 else "black"
                fw = "bold" if abs(v) > BOLD_THRESH else "normal"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                         fontsize=5, color=tc, fontweight=fw, zorder=3)

        # Diagonal boxes (self-retention).
        for i in range(K):
            ax.add_patch(Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=0.9, zorder=4))

        ax.set_xticks(range(K))
        ax.set_xticklabels([_pheno_label(p) for p in phenos],
                            rotation=60, ha="right", fontsize=6)
        for tick, p in zip(ax.get_xticklabels(), phenos):
            tick.set_color(_pheno_color(p))
        ax.set_yticks(range(K))
        ax.set_yticklabels([_pheno_label(p) for p in phenos], fontsize=6)
        for tick, p in zip(ax.get_yticklabels(), phenos):
            tick.set_color(_pheno_color(p))
            tick.set_fontweight("bold")
        ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                      color=TISSUE_COLORS.get(t2, "black"), pad=4)

        # Second x-axis row: dest_fracs as percentages under each column.
        for j in range(K):
            pct = 100.0 * float(dest_fracs[j])
            color = "lightgray" if dest_fracs[j] < RARE_DEST_FLOOR else "dimgray"
            ax.text(j, K - 0.35, f"{pct:.0f}%",
                     ha="center", va="top", fontsize=5,
                     color=color, style="italic",
                     transform=ax.transData)
        ax.tick_params(length=0)
        _add_src_marginal_bar(ax, bres.get("src_weights"), phenos)

    # Row labels (Source: ...) and column headers (Dest: ...)
    for r_idx, src_tis in enumerate(TISSUES):
        axes[r_idx, 0].set_ylabel(f"Source: {src_tis}", fontsize=11,
                                    fontweight="bold",
                                    color=TISSUE_COLORS.get(src_tis, "black"),
                                    labelpad=14)
    for c_idx, dst_tis in enumerate(TISSUES):
        axes[0, c_idx].text(0.5, 1.18, f"Dest: {dst_tis}",
                              transform=axes[0, c_idx].transAxes,
                              ha="center", va="bottom",
                              fontsize=11, fontweight="bold",
                              color=TISSUE_COLORS.get(dst_tis, "black"))

    fig.suptitle(
        f"{lin} phenotype routing enrichment — log₂(T / marginal)",
        fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.965,
              "Red = preferential routing above baseline · "
              "Blue = depletion · "
              "White = expected from destination composition · "
              "gray columns = destination phenotype < 1% (unreliable)",
              ha="center", va="top", fontsize=9, color="dimgray",
              style="italic")
    if valid_im is not None:
        cbar = fig.colorbar(valid_im, ax=axes.ravel().tolist(),
                             fraction=0.025, pad=0.02)
        cbar.set_label("log₂(T / dest_fracs)", fontsize=10)
    fig.savefig(OUT_DIR / f"enrichment_{lin}.png",
                 dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"enrichment_{lin}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote enrichment_{lin}.png/.pdf")


# ---- Top-10 enriched / depleted summary ----
def _format_row(rec):
    lin, t1, t2, src, dst, v = rec
    fold = 2.0 ** v
    return (f"  {lin} {t1:>4s}→{t2:<4s}  "
            f"{src:>14s} → {dst:<14s}  "
            f"log2={v:+.2f}  ({fold:.2f}x above baseline)" if v > 0 else
            f"  {lin} {t1:>4s}→{t2:<4s}  "
            f"{src:>14s} → {dst:<14s}  "
            f"log2={v:+.2f}  ({fold:.2f}x baseline)")


global_enriched.sort(key=lambda r: -r[5])
global_depleted.sort(key=lambda r: r[5])

print("\nTop enriched routings (log2 > 1.5):")
if not global_enriched:
    print("  (none)")
for rec in global_enriched[:10]:
    print(_format_row(rec))

print("\nTop depleted routings (log2 < -1.5):")
if not global_depleted:
    print("  (none)")
for rec in global_depleted[:10]:
    print(_format_row(rec))

# Persist the full lists too.
enrich_cols = ["lineage", "t1", "t2", "source", "destination", "log2_E"]
pd.DataFrame(global_enriched, columns=enrich_cols).to_csv(
    OUT_DIR / "top_enriched.csv", index=False)
pd.DataFrame(global_depleted, columns=enrich_cols).to_csv(
    OUT_DIR / "top_depleted.csv", index=False)
print(f"\nWrote top_enriched.csv ({len(global_enriched)} rows), "
      f"top_depleted.csv ({len(global_depleted)} rows)")


# %%
# ==================================================================
# Step 6A: extract empirical P sub-blocks (row-normalised) per (lineage, edge)
# ==================================================================
# Ground truth for comparison: the empirical 3K×3K transition matrix from
# pipeline/06c_empirical_Q.py. For each directed edge (t1, t2) we lift the
# K×K block whose rows are {t1}__{phenotype} and columns are
# {t2}__{phenotype}, then row-normalise it so it's directly comparable to
# the Bayesian T_global (which is row-stochastic).
from scipy.stats import pearsonr, spearmanr  # noqa: E402

print("\n=== Step 6A: empirical P sub-block extraction ===")
P_EMP_PATH = REPO_ROOT / "results" / "06c_empirical_Q" / "P_empirical.csv"
if P_EMP_PATH.exists():
    HAS_EMPIRICAL = True
    P_emp_df = pd.read_csv(P_EMP_PATH, index_col=0)
    print(f"  loaded {P_EMP_PATH.name}: shape {P_emp_df.shape}")
else:
    HAS_EMPIRICAL = False
    print(f"  WARNING: {P_EMP_PATH} not found — empirical comparison skipped.")


def _lineage_phenotypes(lineage):
    """Sorted short-phenotype list for a lineage (matches model output)."""
    shorts = []
    for full in TCELL_PHENOTYPE_ORDER:
        if not full.startswith(lineage):
            continue
        shorts.append(full
                      .replace("CD8_Activated_", "")
                      .replace("CD8_Quiescent_", "")
                      .replace("CD4_", ""))
    return sorted(shorts)


def _extract_empirical_subblock(P_emp_df, t1, t2, short_phenos):
    """Row-normalised K×K sub-block of P_emp for (t1 → t2). Returns matrix
    and a (K,) boolean array marking which source rows had any mass."""
    full_phenos = [_short_to_full(p) for p in short_phenos]
    row_labels = [f"{t1}__{p}" for p in full_phenos]
    col_labels = [f"{t2}__{p}" for p in full_phenos]
    K = len(short_phenos)
    T_emp = np.zeros((K, K))
    valid = np.zeros(K, dtype=bool)
    for i, rl in enumerate(row_labels):
        if rl not in P_emp_df.index:
            continue
        row = np.zeros(K)
        for j, cl in enumerate(col_labels):
            if cl in P_emp_df.columns:
                row[j] = float(P_emp_df.at[rl, cl])
        s = row.sum()
        if s > 0:
            T_emp[i] = row / s
            valid[i] = True
    return T_emp, valid


empirical_results = {}
if HAS_EMPIRICAL:
    for lin in LINEAGES:
        for (t1, t2) in EDGES:
            if (lin, t1, t2) in bayes_results:
                short_phenos = bayes_results[(lin, t1, t2)]["phenotypes"]
            else:
                short_phenos = _lineage_phenotypes(lin)
            T_emp, valid = _extract_empirical_subblock(
                P_emp_df, t1, t2, short_phenos)
            src_weights_emp = _compute_src_weights(
                adata, t1, lin, short_phenos)
            empirical_results[(lin, t1, t2)] = {
                "T_empirical": T_emp,
                "phenotypes": short_phenos,
                "valid_rows": valid,
                "src_weights": src_weights_emp,
            }
            tag = f"{lin}_{t1}_to_{t2}"
            pd.DataFrame(T_emp, index=short_phenos, columns=short_phenos) \
              .rename_axis(index="source", columns="destination") \
              .to_csv(OUT_DIR / f"T_empirical_{tag}.csv")


# %%
# ---- Step 6 (compare): Bayesian T vs empirical sub-block ----
emp_summary_rows = []
if HAS_EMPIRICAL:
    print("\n  Bayesian vs empirical — flattened-matrix correlation:")
    print(f"  {'edge':<22s} {'pearson':>8s} {'spearman':>9s} "
          f"{'frob':>8s} {'max|d|':>8s}  top-diff")
    for (lin, t1, t2), bres in bayes_results.items():
        eres = empirical_results.get((lin, t1, t2))
        if eres is None:
            continue
        T_b = bres["T_mean"]
        T_e = eres["T_empirical"]
        if T_b.shape != T_e.shape:
            print(f"  shape mismatch {lin} {t1}->{t2}: "
                  f"{T_b.shape} vs {T_e.shape} — skip")
            continue

        b_flat = T_b.flatten()
        e_flat = T_e.flatten()
        if np.std(b_flat) == 0 or np.std(e_flat) == 0:
            r_p = p_p = r_s = p_s = float("nan")
        else:
            r_p, p_p = pearsonr(b_flat, e_flat)
            r_s, p_s = spearmanr(b_flat, e_flat)
        frob = float(np.linalg.norm(T_b - T_e))
        diff = T_b - T_e
        max_abs_diff = float(np.abs(diff).max())
        mi, mj = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        phenos = eres["phenotypes"]
        max_entry = f"{phenos[int(mi)]}->{phenos[int(mj)]}"
        sign = "+" if diff[mi, mj] >= 0 else "-"

        tag = f"{lin}_{t1}_to_{t2}"
        pd.DataFrame(diff, index=phenos, columns=phenos) \
          .rename_axis(index="source", columns="destination") \
          .to_csv(OUT_DIR / f"diff_{tag}.csv")

        emp_summary_rows.append({
            "lineage": lin, "t1": t1, "t2": t2,
            "pearson_r": float(r_p), "pearson_p": float(p_p),
            "spearman_r": float(r_s), "spearman_p": float(p_s),
            "frobenius_distance": frob,
            "max_abs_diff": max_abs_diff,
            "max_diff_entry": max_entry,
            "max_diff_sign": sign,
            "max_diff_value": float(diff[mi, mj]),
        })
        print(f"  {lin+' '+t1+'->'+t2:<22s} {r_p:>8.3f} {r_s:>9.3f} "
              f"{frob:>8.3f} {max_abs_diff:>8.3f}  "
              f"{sign}{max_abs_diff:.3f} @ {max_entry}")

    pd.DataFrame(emp_summary_rows).to_csv(
        OUT_DIR / "empirical_vs_bayesian_summary.csv", index=False)
    print(f"\n  wrote empirical_vs_bayesian_summary.csv "
          f"({len(emp_summary_rows)} rows)")


# %%
# ---- Step 6B: empirical Sankey figures (one composed grid per lineage) ----
sankey_emp_png_paths = {}
if HAS_EMPIRICAL and HAS_PLOTLY:
    print("\nBuilding empirical Sankey diagrams...")
    for (lin, t1, t2), eres in empirical_results.items():
        T_emp = eres["T_empirical"]
        if T_emp.sum() == 0:
            continue
        tag = f"{lin}_{t1}_to_{t2}"
        fig = build_sankey_fig(T_emp, eres["phenotypes"], t1, t2,
                               src_weights=eres.get("src_weights"))
        html_path = SANKEY_DIR / f"sankey_empirical_{tag}.html"
        fig.write_html(str(html_path),
                       include_plotlyjs="cdn", full_html=True)
        if HAS_KALEIDO:
            png_path = SANKEY_DIR / f"sankey_empirical_{tag}.png"
            try:
                fig.write_image(str(png_path), width=520, height=420, scale=2)
                sankey_emp_png_paths[(lin, t1, t2)] = png_path
            except Exception as e:
                print(f"  [{tag}] PNG export failed: {e}")
    print(f"  wrote {len(sankey_emp_png_paths)} empirical Sankey PNGs to "
          f"{SANKEY_DIR}")

if HAS_EMPIRICAL and HAS_PLOTLY and HAS_KALEIDO and sankey_emp_png_paths:
    from PIL import Image
    print("\nComposing empirical Sankey grids...")
    for lin in LINEAGES:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        for (t1, t2) in EDGES:
            r, c = TISSUES.index(t1), TISSUES.index(t2)
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(False)
            png = sankey_emp_png_paths.get((lin, t1, t2))
            if png is None or not png.exists():
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                         fontsize=14, color="dimgray",
                         transform=ax.transAxes)
                if t1 == t2:
                    ax.add_patch(Rectangle(
                        (0, 0), 1, 1, transform=ax.transAxes,
                        facecolor=TISSUE_COLORS.get(t1, "gray"),
                        alpha=0.05, zorder=0))
                continue
            img = Image.open(png)
            ax.imshow(img)
            if t1 == t2:
                ax.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    facecolor=TISSUE_COLORS.get(t1, "gray"),
                    alpha=0.05, zorder=-1))
        _set_row_col_labels(fig, axes, panel_size=None,
                             title=f"{lin} phenotype transitions — "
                                   f"Empirical P sub-blocks")
        fig.tight_layout(rect=(0.02, 0, 1, 0.96))
        fig.savefig(OUT_DIR / f"sankey_empirical_{lin}.png",
                     dpi=DPI, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"sankey_empirical_{lin}.pdf",
                     bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote sankey_empirical_{lin}.png/.pdf")


# %%
# ---- Step 6C: empirical heatmap grid per lineage ----
if HAS_EMPIRICAL:
    print("\nGenerating empirical heatmap grids...")
    for lin in LINEAGES:
        fig, axes = plt.subplots(3, 3, figsize=(15, 14))
        valid_im = None
        for (t1, t2) in EDGES:
            r, c = TISSUES.index(t1), TISSUES.index(t2)
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(False)
            if t1 == t2:
                ax.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    facecolor=TISSUE_COLORS.get(t1, "gray"),
                    alpha=0.05, zorder=-1))
            eres = empirical_results.get((lin, t1, t2))
            if eres is None or eres["T_empirical"].sum() == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                         fontsize=12, color="dimgray",
                         transform=ax.transAxes)
                ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                              color=TISSUE_COLORS.get(t2, "black"), pad=4)
                continue
            T_e = eres["T_empirical"]
            phenos = eres["phenotypes"]
            K = len(phenos)
            im = ax.imshow(T_e, cmap="YlOrRd", vmin=0.0,
                            vmax=HEATMAP_VMAX, aspect="equal")
            valid_im = im
            ax.set_xticks(range(K))
            ax.set_xticklabels([_pheno_label(p) for p in phenos],
                                rotation=60, ha="right", fontsize=6)
            for tick, p in zip(ax.get_xticklabels(), phenos):
                tick.set_color(_pheno_color(p))
            ax.set_yticks(range(K))
            ax.set_yticklabels([_pheno_label(p) for p in phenos], fontsize=6)
            for tick, p in zip(ax.get_yticklabels(), phenos):
                tick.set_color(_pheno_color(p))
                tick.set_fontweight("bold")
            ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                          color=TISSUE_COLORS.get(t2, "black"), pad=4)
            for i in range(K):
                for j in range(K):
                    v = float(T_e[i, j])
                    if v < HEATMAP_ANNOT_THRESH:
                        continue
                    rgba = im.cmap(im.norm(v))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    tc = "white" if lum < 0.5 else "black"
                    weight = "bold" if i == j else "normal"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=5, color=tc, fontweight=weight)
            ax.tick_params(length=0)
            _add_src_marginal_bar(ax, eres.get("src_weights"), phenos)

        for r_idx, src_tis in enumerate(TISSUES):
            axes[r_idx, 0].set_ylabel(f"Source: {src_tis}", fontsize=11,
                                        fontweight="bold",
                                        color=TISSUE_COLORS.get(src_tis,
                                                                "black"),
                                        labelpad=14)
        for c_idx, dst_tis in enumerate(TISSUES):
            axes[0, c_idx].text(0.5, 1.18, f"Dest: {dst_tis}",
                                  transform=axes[0, c_idx].transAxes,
                                  ha="center", va="bottom",
                                  fontsize=11, fontweight="bold",
                                  color=TISSUE_COLORS.get(dst_tis, "black"))
        fig.suptitle(f"{lin} phenotype transitions — Empirical P sub-blocks",
                      fontsize=14, fontweight="bold")
        if valid_im is not None:
            cbar = fig.colorbar(valid_im, ax=axes.ravel().tolist(),
                                 fraction=0.025, pad=0.02)
            cbar.set_label("P(source → destination), row-normalised",
                            fontsize=10)
        fig.savefig(OUT_DIR / f"heatmap_empirical_{lin}.png",
                     dpi=DPI, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"heatmap_empirical_{lin}.pdf",
                     bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote heatmap_empirical_{lin}.png/.pdf")


# %%
# ---- Step 6D: difference heatmap (Bayesian − Empirical) per lineage ----
if HAS_EMPIRICAL:
    print("\nGenerating Bayesian − Empirical difference heatmaps...")
    DIFF_VMIN, DIFF_VMAX = -0.3, 0.3
    DIFF_ANNOT_THRESH = 0.05
    for lin in LINEAGES:
        fig, axes = plt.subplots(3, 3, figsize=(15, 14))
        valid_im = None
        for (t1, t2) in EDGES:
            r, c = TISSUES.index(t1), TISSUES.index(t2)
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "bottom", "left"):
                ax.spines[s].set_visible(False)
            if t1 == t2:
                ax.add_patch(Rectangle(
                    (0, 0), 1, 1, transform=ax.transAxes,
                    facecolor=TISSUE_COLORS.get(t1, "gray"),
                    alpha=0.05, zorder=-1))
            bres = bayes_results.get((lin, t1, t2))
            eres = empirical_results.get((lin, t1, t2))
            if (bres is None or eres is None
                    or eres["T_empirical"].sum() == 0):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                         fontsize=12, color="dimgray",
                         transform=ax.transAxes)
                ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                              color=TISSUE_COLORS.get(t2, "black"), pad=4)
                continue
            T_b = bres["T_mean"]
            T_e = eres["T_empirical"]
            if T_b.shape != T_e.shape:
                ax.text(0.5, 0.5, "shape mismatch", ha="center", va="center",
                         fontsize=10, color="dimgray",
                         transform=ax.transAxes)
                continue
            diff = T_b - T_e
            phenos = eres["phenotypes"]
            K = len(phenos)
            im = ax.imshow(diff, cmap="RdBu_r", vmin=DIFF_VMIN, vmax=DIFF_VMAX,
                            aspect="equal")
            valid_im = im
            ax.set_xticks(range(K))
            ax.set_xticklabels([_pheno_label(p) for p in phenos],
                                rotation=60, ha="right", fontsize=6)
            for tick, p in zip(ax.get_xticklabels(), phenos):
                tick.set_color(_pheno_color(p))
            ax.set_yticks(range(K))
            ax.set_yticklabels([_pheno_label(p) for p in phenos], fontsize=6)
            for tick, p in zip(ax.get_yticklabels(), phenos):
                tick.set_color(_pheno_color(p))
                tick.set_fontweight("bold")
            ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                          color=TISSUE_COLORS.get(t2, "black"), pad=4)
            for i in range(K):
                for j in range(K):
                    v = float(diff[i, j])
                    if abs(v) < DIFF_ANNOT_THRESH:
                        continue
                    tc = "white" if abs(v) > 0.2 else "black"
                    fw = "bold" if abs(v) > 0.15 else "normal"
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                             fontsize=5, color=tc, fontweight=fw)
            ax.tick_params(length=0)
            _add_src_marginal_bar(ax, bres.get("src_weights"), phenos)

        for r_idx, src_tis in enumerate(TISSUES):
            axes[r_idx, 0].set_ylabel(f"Source: {src_tis}", fontsize=11,
                                        fontweight="bold",
                                        color=TISSUE_COLORS.get(src_tis,
                                                                "black"),
                                        labelpad=14)
        for c_idx, dst_tis in enumerate(TISSUES):
            axes[0, c_idx].text(0.5, 1.18, f"Dest: {dst_tis}",
                                  transform=axes[0, c_idx].transAxes,
                                  ha="center", va="bottom",
                                  fontsize=11, fontweight="bold",
                                  color=TISSUE_COLORS.get(dst_tis, "black"))
        fig.suptitle(
            f"{lin} Bayesian minus Empirical (red = Bayesian higher)",
            fontsize=14, fontweight="bold")
        if valid_im is not None:
            cbar = fig.colorbar(valid_im, ax=axes.ravel().tolist(),
                                 fraction=0.025, pad=0.02)
            cbar.set_label("T_Bayesian − T_empirical", fontsize=10)
        fig.savefig(OUT_DIR / f"diff_bayesian_empirical_{lin}.png",
                     dpi=DPI, bbox_inches="tight")
        fig.savefig(OUT_DIR / f"diff_bayesian_empirical_{lin}.pdf",
                     bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote diff_bayesian_empirical_{lin}.png/.pdf")


print(f"\nDone. All outputs in: {OUT_DIR}")
