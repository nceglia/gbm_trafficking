# %%
"""Empirical sankey diagrams + heatmap grids for all 9 directed tissue edges.

For every (lineage ∈ {CD8, CD4}, directed edge (t1, t2)) we extract the
K×K sub-block of the empirical 3K×3K transition matrix
``results/traffic_migration_rates/P_empirical.csv`` (rows = source phenotype @ t1,
columns = destination phenotype @ t2), row-normalise it, and render:

  1. Per-edge interactive Sankey HTML + PNG (plotly + kaleido) and a
     composed 3×3 grid per lineage.
  2. A matplotlib K×K heatmap grid per lineage.

If posterior ``T_*`` matrices from ``traffic_bayesian_sankey`` are
available on disk, we also generate per-lineage (Bayesian − Empirical)
difference heatmap grids.

This was split out of ``traffic_bayesian_sankey.py`` so empirical and
model-based plots are independent pipeline steps. Inputs come from
``traffic_migration_rates.py`` (P_empirical) and the singlet T-cell
AnnData (for source-phenotype prevalence bars).

Outputs go to ``results/traffic_empirical_sankey/``.
"""
import sys
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
from modules import paths  # noqa: E402


# %%
# ---- Config ----
DATA_PATH = paths.H5AD_TCELLS
OUT_DIR = paths.EMPIRICAL_SANKEY_DIR
paths.ensure(OUT_DIR)
SANKEY_DIR = OUT_DIR / "sankey_individual"
paths.ensure(SANKEY_DIR)

P_EMP_PATH = paths.EMPIRICAL_Q_DIR / "P_empirical.csv"
BAYES_DIR = paths.BAYESIAN_SANKEY_DIR  # for optional diff plots

TISSUES = ("PBMC", "CSF", "TP")
EDGES = [
    ("PBMC", "PBMC"), ("CSF", "CSF"), ("TP", "TP"),
    ("PBMC", "CSF"), ("PBMC", "TP"), ("CSF", "PBMC"),
    ("CSF", "TP"),  ("TP", "PBMC"), ("TP", "CSF"),
]
LINEAGES = ["CD8", "CD4"]

LINK_THRESHOLD = 0.02
HEATMAP_VMAX = 0.5
HEATMAP_ANNOT_THRESH = 0.05
DIFF_VMIN, DIFF_VMAX = -0.3, 0.3
DIFF_ANNOT_THRESH = 0.05
DPI = 200


# %%
# ---- Optional plotting deps ----
HAS_PLOTLY = True
HAS_KALEIDO = True
try:
    import plotly.graph_objects as go  # noqa: F401
except ImportError:
    HAS_PLOTLY = False
    print("WARNING: plotly is not installed; Sankey HTML/PNGs will be skipped.")

try:
    import kaleido  # noqa: F401
except ImportError:
    HAS_KALEIDO = False
    print("WARNING: kaleido is not installed; Sankey PNG composition will be "
          "skipped (HTMLs and matplotlib heatmaps still produced).")


# %%
# ---- Empirical P load ----
if not P_EMP_PATH.exists():
    raise FileNotFoundError(
        f"{P_EMP_PATH} not found. Run pipeline/traffic_migration_rates.py "
        f"first to produce the empirical 3K×3K transition matrix.")

print(f"Loading {P_EMP_PATH.name}...")
P_emp_df = pd.read_csv(P_EMP_PATH, index_col=0)
print(f"  shape {P_emp_df.shape}")


# %%
# ---- AnnData for source-phenotype prevalence bars ----
print(f"Loading {DATA_PATH.name}...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")


# %%
# ---- Phenotype short↔full mapping ----
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
    return TCELL_PHENOTYPE_COLORS.get(_short_to_full(short), "#9e9e9e")


def _pheno_label(short):
    return TCELL_PHENOTYPE_LABELS.get(_short_to_full(short), short)


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


def _compute_src_weights(adata, t1, lineage, short_phenotypes):
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


def _extract_empirical_subblock(P_emp_df, t1, t2, short_phenos):
    """Row-normalised K×K sub-block of P_emp for (t1 → t2)."""
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


# %%
# ---- Extract empirical sub-blocks per (lineage, edge) ----
print("\n=== Extracting empirical sub-blocks ===")
empirical_results = {}
for lin in LINEAGES:
    short_phenos = _lineage_phenotypes(lin)
    for (t1, t2) in EDGES:
        T_emp, valid = _extract_empirical_subblock(P_emp_df, t1, t2, short_phenos)
        src_weights = _compute_src_weights(adata, t1, lin, short_phenos)
        empirical_results[(lin, t1, t2)] = {
            "T_empirical": T_emp,
            "phenotypes": short_phenos,
            "valid_rows": valid,
            "src_weights": src_weights,
        }
        tag = f"{lin}_{t1}_to_{t2}"
        pd.DataFrame(T_emp, index=short_phenos, columns=short_phenos) \
          .rename_axis(index="source", columns="destination") \
          .to_csv(OUT_DIR / f"T_empirical_{tag}.csv")
        print(f"  {lin} {t1:>4s}->{t2:<4s}  K={len(short_phenos)}  "
              f"valid rows={int(valid.sum())}/{len(short_phenos)}  "
              f"sum={T_emp.sum():.2f}")


# %%
# ---- Sankey builder (plotly) ----
def _hex_to_rgba_str(hex_color, alpha):
    r, g, b, _ = to_rgba(hex_color, alpha=alpha)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha:.2f})"


def build_sankey_fig(T, phenotypes, t1, t2, src_weights=None):
    """Plotly Sankey for one (t1 → t2) panel.

    Link values are ``src_weights[k] * T[k, j]`` (absolute flow) so node
    heights reflect both the source composition and the routing.
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


sankey_png_paths = {}
if HAS_PLOTLY:
    print("\n=== Empirical Sankey diagrams ===")
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
                sankey_png_paths[(lin, t1, t2)] = png_path
            except Exception as e:
                print(f"  [{tag}] PNG export failed: {e}")
    print(f"  wrote {len(sankey_png_paths)} PNGs / "
          f"{len(empirical_results)} HTMLs to {SANKEY_DIR}")


# %%
# ---- Composed 3×3 Sankey grid per lineage ----
def _set_row_col_labels(fig, axes, title):
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

    print("\nComposing empirical Sankey grids...")
    for lin in LINEAGES:
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        for (t1, t2) in EDGES:
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
        _set_row_col_labels(fig, axes,
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
# ---- Heatmap grid per lineage ----
print("\n=== Empirical heatmap grids ===")
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
                                    color=TISSUE_COLORS.get(src_tis, "black"),
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
# ---- Optional: difference heatmap (Bayesian − Empirical) ----
# Only runs if traffic_bayesian_sankey.py has been executed and its
# T_*.csv files exist; otherwise this block is skipped.
def _load_bayesian_T(lin, t1, t2):
    """Return the Bayesian T matrix and its phenotype list, or (None, None)
    if the per-edge CSV is missing. The phenotype list may be a strict
    subset of the empirical one — the Bayesian fit only sees phenotypes
    that appear in both (t1, t2) for at least one shared clone."""
    path = BAYES_DIR / f"T_{lin}_{t1}_to_{t2}.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, index_col=0)
    return df.values, list(df.index)


def _align_empirical_to_bayes(T_emp, emp_phenos, bayes_phenos):
    """Sub-select rows/cols of the empirical T to match the Bayesian
    phenotype set, then row-renormalise."""
    idx = [emp_phenos.index(p) for p in bayes_phenos if p in emp_phenos]
    if len(idx) != len(bayes_phenos):
        return None
    T = T_emp[np.ix_(idx, idx)]
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


bayes_available = any(
    (BAYES_DIR / f"T_{lin}_{t1}_to_{t2}.csv").exists()
    for lin in LINEAGES for (t1, t2) in EDGES)

if bayes_available:
    print("\n=== Diff heatmaps (Bayesian − Empirical) ===")
    diff_summary_rows = []
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
            emp_phenos = eres["phenotypes"]
            T_b, bayes_phenos = _load_bayesian_T(lin, t1, t2)
            if T_b is None:
                ax.text(0.5, 0.5, "no Bayesian fit", ha="center", va="center",
                         fontsize=11, color="dimgray", transform=ax.transAxes)
                ax.set_title(f"{t1} → {t2}", fontsize=10, fontweight="bold",
                              color=TISSUE_COLORS.get(t2, "black"), pad=4)
                continue
            T_e_aligned = _align_empirical_to_bayes(
                eres["T_empirical"], emp_phenos, bayes_phenos)
            if T_e_aligned is None:
                ax.text(0.5, 0.5, "pheno mismatch", ha="center", va="center",
                         fontsize=11, color="dimgray", transform=ax.transAxes)
                continue
            phenos = bayes_phenos
            diff = T_b - T_e_aligned
            K = len(phenos)
            tag = f"{lin}_{t1}_to_{t2}"
            pd.DataFrame(diff, index=phenos, columns=phenos) \
              .rename_axis(index="source", columns="destination") \
              .to_csv(OUT_DIR / f"diff_{tag}.csv")
            frob = float(np.linalg.norm(diff))
            max_abs = float(np.abs(diff).max())
            mi, mj = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
            diff_summary_rows.append({
                "lineage": lin, "t1": t1, "t2": t2,
                "frobenius": frob,
                "max_abs_diff": max_abs,
                "max_diff_entry": f"{phenos[int(mi)]}->{phenos[int(mj)]}",
                "max_diff_value": float(diff[mi, mj]),
            })
            im = ax.imshow(diff, cmap="RdBu_r",
                            vmin=DIFF_VMIN, vmax=DIFF_VMAX, aspect="equal")
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
            # Source-marginal bar — recompute for the aligned (possibly
            # smaller) phenotype set so the bar lines up with the heatmap.
            aligned_src = _compute_src_weights(adata, t1, lin, phenos)
            _add_src_marginal_bar(ax, aligned_src, phenos)

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

    if diff_summary_rows:
        pd.DataFrame(diff_summary_rows).to_csv(
            OUT_DIR / "diff_summary.csv", index=False)
        print(f"  wrote diff_summary.csv ({len(diff_summary_rows)} rows)")
else:
    print("\n[diff heatmaps] skipped — no T_*.csv found in "
          f"{BAYES_DIR}. Run traffic_bayesian_sankey.py first if you "
          f"want Bayesian-vs-empirical difference plots.")


print(f"\nDone. All outputs in: {OUT_DIR}")
