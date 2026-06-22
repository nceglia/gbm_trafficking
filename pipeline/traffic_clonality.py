# %%
"""Cohort clonality summary + per-phenotype temporal trends.

Three figures, all built on a shared aesthetic:
  results/traffic_clonality/clonality_summary.{png,pdf}
  results/traffic_clonality/clonality_phenotype_trends.{png,pdf}
  results/traffic_clonality/clonality_phenotype_correlations.{png,pdf}

Plus the four CSVs:
  clonality_table.csv     long form: G1..G5, G_lineage, G_phenotype
  dropped.csv             strata with 1 <= n_cells < MIN_CELLS
  temporal_trends.csv     per (phenotype, tissue) MixedLM fits
  trend_correlations.csv  pairwise residualized Pearson over trends
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.clonality import compute_clonality, fit_temporal_trends
from modules.style import (
    LINEAGE_COLORS, LINEAGE_ORDER,
    TCELL_PHENOTYPE_COLORS, TCELL_PHENOTYPE_LABELS, TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS, TISSUE_ORDER,
)

# %%
# ---- Global config (single source of truth) ----
MIN_CELLS = 10
LW_TREND = 2.5
MARKER_SIZE = 6
CAPSIZE = 3
TITLE_FS = 13
LABEL_FS = 11
TICK_FS = 9
ANNOT_FS = 8
DPI = 200

from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
OUT_DIR = REPO_ROOT / "results" / "clonality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = list(TISSUE_ORDER)
LINEAGES = list(LINEAGE_ORDER)
TIMEPOINTS = ["1", "2", "3", "4", "5", "6"]
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)
ALL_AXES = ["patient", "tissue", "timepoint", "lineage", "phenotype"]

GROUPINGS = {
    "G1":          ["patient", "tissue"],
    "G2":          ["patient", "tissue", "timepoint"],
    "G3":          ["patient", "tissue", "lineage"],
    "G4":          ["tissue", "phenotype"],
    "G5":          ["patient", "tissue", "lineage", "timepoint"],
    "G_lineage":   ["patient", "tissue", "timepoint", "lineage"],
    "G_phenotype": ["patient", "tissue", "timepoint", "phenotype"],
}


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.tick_params(labelsize=TICK_FS)


# %%
# ---- Load ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
obs = adata.obs[["trb", "tissue", "timepoint", "phenotype",
                 "lineage", "patient"]].copy()


def _remap_pheno(p):
    if p in TCELL_PHENOTYPE_COLORS:
        return p
    if p in ("CD4_Th1_polarized", "CD4_Th2_polarized"):
        return "CD4_Th"
    return p


obs["phenotype"] = obs["phenotype"].astype(str).map(_remap_pheno)
print(f"  {len(obs)} cells, {obs['phenotype'].nunique()} phenotypes")

# %%
# ---- Compute clonality across all groupings ----
frames = []
for name, cols in GROUPINGS.items():
    res = compute_clonality(obs, cols)
    res["grouping"] = name
    for ax in ALL_AXES:
        if ax not in res.columns:
            res[ax] = np.nan
    frames.append(res)
    print(f"  {name}: {len(res)} strata")

ct = pd.concat(frames, ignore_index=True)
ct = ct[["grouping"] + ALL_AXES + [
    "n_cells", "n_clones", "shannon", "pielou", "clonality", "inv_simpson",
    "clonality_lo", "clonality_hi", "inv_simpson_lo", "inv_simpson_hi",
]]
ct.to_csv(OUT_DIR / "clonality_table.csv", index=False)

dropped = (ct[(ct["n_cells"] >= 1) & (ct["n_cells"] < MIN_CELLS)]
           .rename(columns={"grouping": "grouping_type"}))
dropped[["grouping_type"] + ALL_AXES + ["n_cells", "n_clones"]].to_csv(
    OUT_DIR / "dropped.csv", index=False)
print(f"  dropped (n<{MIN_CELLS}): {len(dropped)}")


def _g(name):
    return ct[(ct["grouping"] == name) & (ct["n_cells"] >= MIN_CELLS)].copy()


# %%
# ---- Temporal trends ----
print("Fitting temporal trends...")
g_phen = _g("G_phenotype")
trends = fit_temporal_trends(g_phen, n_min=10)
trends.to_csv(OUT_DIR / "temporal_trends.csv", index=False)
print(f"  {len(trends)} (phenotype, tissue) trends fit")

g_lin = _g("G_lineage")
rng_strip = np.random.default_rng(0)
tp_cmap = plt.get_cmap("viridis")
tp_norm = Normalize(vmin=1, vmax=6)


def _trend_points(sub_t):
    """Per-tissue mean / SEM / x where n_patients >= 2."""
    means, sems, xs_m = [], [], []
    for tp in TIMEPOINTS:
        s = sub_t[sub_t["timepoint"] == tp]
        if s["patient"].nunique() >= 2:
            means.append(s["clonality"].mean())
            sems.append(s["clonality"].std(ddof=1) / np.sqrt(len(s)))
            xs_m.append(int(tp))
    return np.array(xs_m), np.array(means), np.array(sems)


# %%
# ============================================================
# FIGURE 1: clonality_summary
# ============================================================
print("Plotting clonality_summary...")

fig1 = plt.figure(figsize=(11, 12))
gs1 = GridSpec(
    3, 2, height_ratios=[1, 1, 1], hspace=0.55, wspace=0.18,
    left=0.10, right=0.82, top=0.88, bottom=0.10,
)
ax_a8 = fig1.add_subplot(gs1[0, 0])
ax_a4 = fig1.add_subplot(gs1[0, 1], sharey=ax_a8)
ax_b8 = fig1.add_subplot(gs1[1, :])
ax_b4 = fig1.add_subplot(gs1[2, :], sharex=ax_b8, sharey=ax_b8)

# ---- Row 1: boxplots ----
for ax, lineage in [(ax_a8, "CD8"), (ax_a4, "CD4")]:
    sub_l = g_lin[g_lin["lineage"] == lineage]
    box_data = [sub_l[sub_l["tissue"] == t]["clonality"].dropna().to_numpy()
                for t in TISSUES]
    bp = ax.boxplot(
        box_data, positions=range(len(TISSUES)), widths=0.6,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, t in zip(bp["boxes"], TISSUES):
        patch.set_facecolor(TISSUE_COLORS[t])
        patch.set_alpha(0.85)

    for ti, t in enumerate(TISSUES):
        s = sub_l[sub_l["tissue"] == t].dropna(subset=["clonality"])
        if s.empty:
            continue
        tps = s["timepoint"].astype(int).to_numpy()
        cols = tp_cmap(tp_norm(tps))
        jit = rng_strip.uniform(-0.18, 0.18, len(s))
        ax.scatter(np.full(len(s), ti) + jit, s["clonality"],
                   s=28, c=cols, alpha=0.75,
                   edgecolors="none", zorder=3)

    ax.set_xticks(range(len(TISSUES)))
    ax.set_xticklabels(TISSUES, fontsize=TICK_FS)
    for tick, t in zip(ax.get_xticklabels(), TISSUES):
        tick.set_color(TISSUE_COLORS[t])
        tick.set_fontweight("bold")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(lineage, fontsize=TITLE_FS, fontweight="bold")
    _style_axis(ax)

ax_a8.set_ylabel("Clonality", fontsize=LABEL_FS)
plt.setp(ax_a4.get_yticklabels(), visible=False)

# Timepoint colorbar to the right of the CD4 axis (explicit add_axes)
bbox_a4 = gs1[0, 1].get_position(fig1)
cax_tp = fig1.add_axes(
    [bbox_a4.x1 + 0.04, bbox_a4.y0, 0.012, bbox_a4.height])
sm = ScalarMappable(norm=tp_norm, cmap=tp_cmap)
sm.set_array([])
cb_tp = fig1.colorbar(sm, cax=cax_tp)
cb_tp.set_ticks([1, 2, 3, 4, 5, 6])
cb_tp.set_ticklabels([str(i) for i in range(1, 7)])
cb_tp.ax.tick_params(labelsize=TICK_FS)
cb_tp.set_label("timepoint", fontsize=LABEL_FS, rotation=90)
cb_tp.ax.yaxis.set_label_position("right")

# ---- Rows 2 & 3: trend lines, no patient dots, no fill ----
for ax, lineage in [(ax_b8, "CD8"), (ax_b4, "CD4")]:
    sub_l = g_lin[g_lin["lineage"] == lineage]
    for tissue in TISSUES:
        sub_t = sub_l[sub_l["tissue"] == tissue].dropna(subset=["clonality"])
        xs_m, means, sems = _trend_points(sub_t)
        if len(xs_m) < 2:
            continue
        ax.errorbar(xs_m, means, yerr=sems,
                    color=TISSUE_COLORS[tissue],
                    linewidth=LW_TREND, marker="o",
                    markersize=MARKER_SIZE, capsize=CAPSIZE,
                    label=tissue, zorder=3)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Clonality", fontsize=LABEL_FS)
    ax.set_title(f"{lineage} trajectories",
                 fontsize=TITLE_FS, fontweight="bold")
    _style_axis(ax)

ax_b4.set_xlabel("Timepoint", fontsize=LABEL_FS)
plt.setp(ax_b8.get_xticklabels(), visible=False)

# Tissue legend at top
tissue_handles = [Line2D([0], [0], color=TISSUE_COLORS[t],
                          linewidth=LW_TREND, marker="o",
                          markersize=MARKER_SIZE, label=t)
                  for t in TISSUES]
fig1.legend(handles=tissue_handles, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, 0.94),
            frameon=False, fontsize=LABEL_FS)

fig1.suptitle("Clonality summary",
              fontsize=TITLE_FS + 1, fontweight="bold", y=0.97)
fig1.text(
    0.5, 0.04,
    "Clonality = 1 − Shannon/log(n_clones). n_cells ≥ 10 per stratum. "
    "Boxplot dots: (patient, timepoint) observations. "
    "Trend lines: mean ± SEM across patients.",
    ha="center", va="bottom", fontsize=ANNOT_FS,
    color="dimgray", style="italic",
)
fig1.savefig(OUT_DIR / "clonality_summary.png", dpi=DPI, bbox_inches="tight")
fig1.savefig(OUT_DIR / "clonality_summary.pdf", bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'clonality_summary.png'}")

# %%
# ============================================================
# FIGURE 2: clonality_phenotype_trends
# ============================================================
print("Plotting clonality_phenotype_trends...")

ncols, nrows = 4, 3
fig2 = plt.figure(figsize=(15, 11))
gs2 = GridSpec(nrows, ncols,
               hspace=0.45, wspace=0.15,
               left=0.07, right=0.97, top=0.86, bottom=0.10)

ax0 = None
axes2 = []
for idx, p in enumerate(PHENOTYPES):
    r, c = idx // ncols, idx % ncols
    if ax0 is None:
        ax = fig2.add_subplot(gs2[r, c])
        ax0 = ax
    else:
        ax = fig2.add_subplot(gs2[r, c], sharex=ax0, sharey=ax0)
    axes2.append(ax)

    sub_p = g_phen[g_phen["phenotype"] == p]
    for tissue in TISSUES:
        sub_t = sub_p[sub_p["tissue"] == tissue].dropna(subset=["clonality"])
        xs_m, means, sems = _trend_points(sub_t)
        if len(xs_m) < 1:
            continue
        ax.errorbar(xs_m, means, yerr=sems,
                    color=TISSUE_COLORS[tissue],
                    linewidth=LW_TREND, marker="o",
                    markersize=MARKER_SIZE, capsize=CAPSIZE,
                    label=tissue, zorder=3)

    # Significance annotations (text only, top-left, per tissue)
    line_y = 0.97
    for tissue in TISSUES:
        tr = trends[(trends["phenotype"] == p) & (trends["tissue"] == tissue)]
        if not len(tr):
            continue
        q = tr.iloc[0]["slope_q"]
        if pd.isna(q):
            continue
        star = " **" if q < 0.10 else ""
        ax.text(0.03, line_y, f"{tissue} q={q:.2f}{star}",
                transform=ax.transAxes, fontsize=ANNOT_FS - 1,
                color=TISSUE_COLORS[tissue],
                va="top", ha="left", fontweight="bold" if star else "normal")
        line_y -= 0.07

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 1)
    ax.set_title(TCELL_PHENOTYPE_LABELS.get(p, p),
                 fontsize=LABEL_FS, fontweight="bold")
    if c == 0:
        ax.set_ylabel("Clonality", fontsize=LABEL_FS)
    if r == nrows - 1:
        ax.set_xlabel("Timepoint", fontsize=LABEL_FS)
    _style_axis(ax)

# Hide unused panels
for idx in range(len(PHENOTYPES), ncols * nrows):
    r, c = idx // ncols, idx % ncols
    ax_empty = fig2.add_subplot(gs2[r, c])
    ax_empty.axis("off")

# Single shared tissue legend at top
fig2.legend(handles=tissue_handles, ncol=3,
            loc="upper center", bbox_to_anchor=(0.5, 0.94),
            frameon=False, fontsize=LABEL_FS)

fig2.suptitle("Phenotype × tissue clonality trends",
              fontsize=TITLE_FS + 1, fontweight="bold", y=0.97)
fig2.text(
    0.5, 0.04,
    "** indicates linear mixed-effects slope q < 0.10 (BH). "
    "All lines solid; ** is the only marker of significance.",
    ha="center", va="bottom", fontsize=ANNOT_FS,
    color="dimgray", style="italic",
)
fig2.savefig(OUT_DIR / "clonality_phenotype_trends.png",
             dpi=DPI, bbox_inches="tight")
fig2.savefig(OUT_DIR / "clonality_phenotype_trends.pdf",
             bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'clonality_phenotype_trends.png'}")

# %%
# ============================================================
# FIGURE 3: clonality_phenotype_correlations
# ============================================================
print("Plotting clonality_phenotype_correlations...")

g_phen2 = _g("G_phenotype").copy()
g_phen2["unit"] = (g_phen2["patient"].astype(str)
                   + "|T" + g_phen2["timepoint"].astype(str))
g_phen2["trend"] = (g_phen2["phenotype"].astype(str)
                    + " | " + g_phen2["tissue"].astype(str))
M = g_phen2.pivot_table(index="unit", columns="trend",
                         values="clonality", aggfunc="first")
M = M.loc[:, M.count() >= 10]
print(f"  matrix: {M.shape[0]} units × {M.shape[1]} trends")

# Residualize per patient
patient_of_unit = pd.Series(M.index, index=M.index).str.split("|").str[0]
M_resid = M - M.groupby(patient_of_unit).transform("mean")

corr = M_resid.corr(method="pearson", min_periods=8)
trends_list = list(corr.index)
nT = len(trends_list)

# Pairwise p-values + BH-FDR
pvals = np.full((nT, nT), np.nan)
for i in range(nT):
    for j in range(i + 1, nT):
        s = M_resid[[trends_list[i], trends_list[j]]].dropna()
        if len(s) >= 8:
            r_ij, p_ij = pearsonr(s.iloc[:, 0], s.iloc[:, 1])
            pvals[i, j] = pvals[j, i] = p_ij
upper = np.triu_indices(nT, k=1)
flat_p = pvals[upper]
qvals = np.full((nT, nT), np.nan)
valid = ~np.isnan(flat_p)
if valid.sum() > 0:
    _, q_flat, _, _ = multipletests(flat_p[valid], method="fdr_bh")
    q_full = np.full(len(flat_p), np.nan)
    q_full[valid] = q_flat
    for k, (i, j) in enumerate(zip(*upper)):
        qvals[i, j] = qvals[j, i] = q_full[k]

rows_corr = []
for k, (i, j) in enumerate(zip(*upper)):
    s = M_resid[[trends_list[i], trends_list[j]]].dropna()
    rows_corr.append(dict(
        trend1=trends_list[i], trend2=trends_list[j],
        n_obs=len(s),
        pearson_r=(float(corr.iat[i, j])
                   if pd.notna(corr.iat[i, j]) else np.nan),
        p=float(pvals[i, j]) if not np.isnan(pvals[i, j]) else np.nan,
        q=float(qvals[i, j]) if not np.isnan(qvals[i, j]) else np.nan,
    ))
pd.DataFrame(rows_corr).to_csv(OUT_DIR / "trend_correlations.csv", index=False)

# Hierarchical clustering on (1 - corr) distance
dist_mat = 1.0 - corr.fillna(0).to_numpy()
np.fill_diagonal(dist_mat, 0.0)
dist_mat = np.clip((dist_mat + dist_mat.T) / 2.0, 0.0, 2.0)
Z = linkage(squareform(dist_mat, checks=False), method="average")
order = leaves_list(Z)
trends_ord = [trends_list[i] for i in order]
corr_o = corr.loc[trends_ord, trends_ord].to_numpy()
qvals_o = qvals[np.ix_(order, order)]


def _parse_trend(t):
    p, _, ti = t.partition(" | ")
    return p, ti


phenos_ord = [_parse_trend(t)[0] for t in trends_ord]
tissues_ord = [_parse_trend(t)[1] for t in trends_ord]
lineages_ord = ["CD8" if "CD8" in p else "CD4" for p in phenos_ord]

fig3 = plt.figure(figsize=(14, 13))
# Geometry keeps y-tick labels left of the lineage track and the
# footer text below the bottom x-tick labels.
ax_main = fig3.add_axes([0.24, 0.16, 0.56, 0.62])
ax_top  = fig3.add_axes([0.24, 0.79, 0.56, 0.018])
ax_left = fig3.add_axes([0.22, 0.16, 0.018, 0.62])
ax_cbar = fig3.add_axes([0.83, 0.16, 0.015, 0.62])

# Tissue annotation track (top)
tissue_idx = np.array([TISSUES.index(t) for t in tissues_ord]).reshape(1, -1)
tissue_cmap = ListedColormap([TISSUE_COLORS[t] for t in TISSUES])
ax_top.imshow(tissue_idx, cmap=tissue_cmap, aspect="auto",
              vmin=-0.5, vmax=len(TISSUES) - 0.5,
              extent=(-0.5, nT - 0.5, 0, 1))
ax_top.set_xlim(-0.5, nT - 0.5)
ax_top.set_xticks([])
ax_top.set_yticks([])
for s in ax_top.spines.values():
    s.set_visible(False)

# Lineage annotation track (left)
lineage_idx = np.array([LINEAGES.index(l)
                         for l in lineages_ord]).reshape(-1, 1)
lineage_cmap = ListedColormap([LINEAGE_COLORS[l] for l in LINEAGES])
ax_left.imshow(lineage_idx, cmap=lineage_cmap, aspect="auto",
               vmin=-0.5, vmax=len(LINEAGES) - 0.5,
               extent=(0, 1, nT - 0.5, -0.5))
ax_left.set_ylim(nT - 0.5, -0.5)
ax_left.set_xticks([])
ax_left.set_yticks([])
for s in ax_left.spines.values():
    s.set_visible(False)

# Main heatmap
im = ax_main.imshow(corr_o, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
for i in range(nT):
    for j in range(nT):
        if i == j:
            continue
        if not np.isnan(qvals_o[i, j]) and qvals_o[i, j] < 0.10:
            ax_main.plot(j, i, marker="o", color="black",
                         markersize=4, zorder=3)

short_labels = [
    f"{TCELL_PHENOTYPE_LABELS.get(_parse_trend(t)[0], _parse_trend(t)[0])}"
    f" | {_parse_trend(t)[1]}"
    for t in trends_ord
]
ax_main.set_xticks(range(nT))
ax_main.set_xticklabels(short_labels, rotation=45,
                        ha="right", fontsize=ANNOT_FS)
ax_main.set_yticks(range(nT))
ax_main.set_yticklabels(short_labels, fontsize=ANNOT_FS)
ax_main.tick_params(axis="x", length=0, pad=2)
ax_main.tick_params(axis="y", length=0, pad=22)

# Tick label colors: x by tissue, y by lineage
for tick, t in zip(ax_main.get_xticklabels(), tissues_ord):
    tick.set_color(TISSUE_COLORS[t])
for tick, l in zip(ax_main.get_yticklabels(), lineages_ord):
    tick.set_color(LINEAGE_COLORS[l])

# Bold labels for trends with slope_q < 0.10
sig_trends = set()
if "slope_q" in trends.columns:
    for _, r in trends.iterrows():
        if pd.notna(r["slope_q"]) and r["slope_q"] < 0.10:
            sig_trends.add(f"{r['phenotype']} | {r['tissue']}")
for tick, t in zip(ax_main.get_xticklabels(), trends_ord):
    if t in sig_trends:
        tick.set_fontweight("bold")
for tick, t in zip(ax_main.get_yticklabels(), trends_ord):
    if t in sig_trends:
        tick.set_fontweight("bold")

cb = fig3.colorbar(im, cax=ax_cbar)
cb.set_label("Pearson r (within-patient residuals)", fontsize=LABEL_FS)
cb.ax.tick_params(labelsize=TICK_FS)

fig3.suptitle("Co-variation of (phenotype × tissue) clonality trends",
              fontsize=TITLE_FS + 1, fontweight="bold", y=0.93)
fig3.text(
    0.5, 0.03,
    "Black dot: corr_q < 0.10. Bold labels: slope_q < 0.10 in plot 2. "
    "Row color = lineage, column color = tissue.",
    ha="center", va="bottom", fontsize=ANNOT_FS,
    color="dimgray", style="italic",
)
fig3.savefig(OUT_DIR / "clonality_phenotype_correlations.png",
             dpi=DPI, bbox_inches="tight")
fig3.savefig(OUT_DIR / "clonality_phenotype_correlations.pdf",
             bbox_inches="tight")
print(f"Saved: {OUT_DIR / 'clonality_phenotype_correlations.png'}")

# %%
print("Done.")
