# %%
"""Phenotype-prior pseudotime for one lineage (CD8 or CD4).

Question: does the current phenotype labeling support a coherent
differentiation order? We pin palantir's start cell inside the lineage's
naive phenotype and one terminal cell inside each labeled terminal
phenotype, rather than letting palantir freely discover endpoints.

    python pipeline/traffic_pseudotime_phenotypes.py --lineage CD8
    python pipeline/traffic_pseudotime_phenotypes.py --lineage CD4
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import harmonypy as hp
from matplotlib.gridspec import GridSpec

import palantir

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths
from modules.clone_helpers import infer_lineage_from_phenotype
from modules.style import (
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_LABELS,
)

# %% Config
_parser = argparse.ArgumentParser()
_parser.add_argument("--lineage", choices=["CD8", "CD4"], default="CD8")
_args, _ = _parser.parse_known_args()
LINEAGE = _args.lineage

DATA_PATH = paths.H5AD_TCELLS
UMAP_PKL = REPO_ROOT / "data" / "embeddings" / "X_umap.pkl"
OUT_DIR = REPO_ROOT / "results" / "traffic_pseudotime_phenotypes" / LINEAGE
OUT_DIR.mkdir(parents=True, exist_ok=True)

NAIVE_PHENOTYPE = {
    "CD8": "CD8_Quiescent_Naive",
    "CD4": "CD4_Naive_Memory",
}
NAIVE_MARKERS = {
    "CD8": ["TCF7", "LEF1", "CCR7", "SELL", "IL7R", "MAL", "BACH2"],
    "CD4": ["IL7R", "TCF7", "CCR7", "SELL", "LEF1", "MAL"],
}
TERMINAL_PHENOTYPES = {
    "CD8": {
        "CD8_Activated_TEXterm": ["TOX", "LAG3", "PDCD1", "HAVCR2", "ENTPD1", "CXCL13"],
        "CD8_Activated_TEMRA":   ["CX3CR1", "S1PR5", "KLF2", "FGFBP2", "FCGR3A"],
        "CD8_Activated_TRM":     ["CD69", "ZNF683", "ITGAE", "IFNG"],
    },
    "CD4": {
        "CD4_Treg":      ["FOXP3", "IKZF2", "IL2RA", "ICOS", "CTLA4"],
        "CD4_Exhausted": ["PDCD1", "LAG3", "CXCL13", "TOX", "CTLA4"],
    },
}
NAIVE_PHENOTYPE_LIN = NAIVE_PHENOTYPE[LINEAGE]
NAIVE_MARKERS_LIN = NAIVE_MARKERS[LINEAGE]
TERMINALS_LIN = TERMINAL_PHENOTYPES[LINEAGE]

N_HVG = 3000
N_PCA = 50
N_NEIGHBORS = 30
N_DIFFUSION = 10
N_WAYPOINTS = 500
SEED = 42

sc.settings.verbosity = 1

# %% Load + subset to lineage
adata_full = sc.read_h5ad(str(DATA_PATH))
print(f"Full adata: {adata_full.n_obs:,} cells x {adata_full.n_vars:,} genes")

lineage_mask = adata_full.obs["phenotype"].astype(str).map(
    infer_lineage_from_phenotype) == LINEAGE
full_positions = np.where(lineage_mask.values)[0]
adata = adata_full[lineage_mask].copy()
print(f"{LINEAGE} subset: {adata.n_obs:,} cells")
print(adata.obs["phenotype"].value_counts().to_string())

# %% Canonical UMAP (positional align to full adata)
umap_arr = pd.read_pickle(UMAP_PKL)
if not isinstance(umap_arr, np.ndarray):
    raise TypeError(f"X_umap.pkl: expected ndarray, got {type(umap_arr).__name__}")
if umap_arr.shape[0] != adata_full.n_obs:
    raise ValueError(
        f"X_umap.pkl rows ({umap_arr.shape[0]}) != full adata.n_obs "
        f"({adata_full.n_obs}); cannot positionally align.")
adata.obsm["X_umap"] = umap_arr[full_positions].astype(np.float32)
print(f"UMAP loaded: {adata.obsm['X_umap'].shape}")
del adata_full

# %% Rebuild from raw counts
adata.X = adata.layers["counts"].copy()
adata.uns.pop("log1p", None)
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat_v3")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["log1p"] = adata.X.copy()
print(f"HVGs: {int(adata.var['highly_variable'].sum())}")

# %% PCA + Harmony
# scanpy.external.pp.harmony_integrate calls .T on Z_corr, which assumed
# harmonypy 1.x layout (n_pcs, n_cells). harmonypy 2.0+ already returns
# (n_cells, n_pcs), so we call run_harmony directly and assign as-is.
sc.tl.pca(adata, n_comps=N_PCA, use_highly_variable=True)
var_ratio = adata.uns["pca"]["variance_ratio"]
print(f"PCA: top-10 var explained = {var_ratio[:10].sum():.2%}, "
      f"all-{N_PCA} = {var_ratio.sum():.2%}")
ho = hp.run_harmony(adata.obsm["X_pca"], adata.obs, "patient",
                    random_state=SEED)
adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr)

# %% Neighbors (Harmony-corrected)
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=N_NEIGHBORS)

# %% Palantir diffusion + multiscale
palantir.utils.run_diffusion_maps(
    adata, pca_key="X_pca_harmony",
    n_components=N_DIFFUSION, knn=N_NEIGHBORS)
palantir.utils.determine_multiscale_space(adata)

# %% Phenotype-prior start + terminal selection
def _filter_markers(adata, markers, label):
    present = [g for g in markers if g in adata.var_names]
    dropped = [g for g in markers if g not in adata.var_names]
    if dropped:
        print(f"  [{label}] dropped (not in var_names): {dropped}")
    if not present:
        raise RuntimeError(f"[{label}] no markers present after filtering.")
    return present


def _pick_in_phenotype(adata, phenotype, markers, score_name):
    sc.tl.score_genes(adata, gene_list=markers, score_name=score_name,
                      use_raw=False, random_state=SEED)
    mask = (adata.obs["phenotype"].astype(str) == phenotype).values
    if mask.sum() == 0:
        raise RuntimeError(f"phenotype {phenotype!r} has 0 cells in subset")
    scores = adata.obs[score_name].to_numpy().copy()
    scores[~mask] = -np.inf
    pos = int(np.argmax(scores))
    cell = adata.obs_names[pos]
    actual_pheno = str(adata.obs["phenotype"].iloc[pos])
    print(f"  picked {score_name}={scores[pos]:.3f}  cell={cell}  "
          f"phenotype={actual_pheno}")
    return cell


print("\nNaive (start) cell:")
naive_present = _filter_markers(adata, NAIVE_MARKERS_LIN, "naive")
start = _pick_in_phenotype(adata, NAIVE_PHENOTYPE_LIN, naive_present,
                           "score_naive")

print("\nTerminal cells:")
terminal_states = {}
for full_pheno, markers in TERMINALS_LIN.items():
    short = full_pheno.rsplit("_", 1)[-1]
    present = _filter_markers(adata, markers, short)
    cell = _pick_in_phenotype(adata, full_pheno, present, f"score_{short}")
    terminal_states[cell] = short

ts_series = pd.Series(terminal_states)
print(f"\nstart cell: {start}")
print(f"terminal states: {dict(ts_series)}")

# %% Run palantir
pr_res = palantir.core.run_palantir(
    adata, early_cell=start, terminal_states=ts_series,
    num_waypoints=N_WAYPOINTS, knn=N_NEIGHBORS, seed=SEED,
)

# %% Branch cells
palantir.presults.select_branch_cells(adata, q=0.01, eps=0.01)

# %% PAGA on phenotype (lineage-ordered)
lin_order = [p for p in TCELL_PHENOTYPE_ORDER
             if p in adata.obs["phenotype"].unique()]
adata.obs["phenotype"] = pd.Categorical(
    adata.obs["phenotype"].astype(str), categories=lin_order, ordered=True)
sc.tl.paga(adata, groups="phenotype")

# %% Save h5ad
out_h5ad = OUT_DIR / f"{LINEAGE}_pseudotime.h5ad"
adata.write_h5ad(out_h5ad)
print(f"\nWrote {out_h5ad}")

# %% Sanity report
pt = adata.obs["palantir_pseudotime"].to_numpy()
ent = adata.obs["palantir_entropy"].to_numpy()
print(f"\npseudotime: min={pt.min():.3f}  max={pt.max():.3f}  "
      f"median={np.median(pt):.3f}")
print(f"entropy:    min={ent.min():.3f}  max={ent.max():.3f}  "
      f"median={np.median(ent):.3f}")

branch_masks = adata.obsm["branch_masks"]
if isinstance(branch_masks, pd.DataFrame):
    branch_cols = list(branch_masks.columns)
    branch_arr = branch_masks.to_numpy()
else:
    branch_cols = list(ts_series.values)
    branch_arr = np.asarray(branch_masks)

print("\nBranch mass per terminal (fraction of cells assigned):")
total = adata.n_obs
flags = []
for j, name in enumerate(branch_cols):
    frac = float(branch_arr[:, j].sum()) / total
    flag = ""
    if frac > 0.80:
        flag = "  <-- FLAG: >80% of cells"
    elif frac < 0.05:
        flag = "  <-- FLAG: <5% of cells"
    if flag:
        flags.append((name, frac, flag.strip()))
    print(f"  {name:>10s}: {frac:.3f}{flag}")

# %% Figure helpers
def _bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=220,
                bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


X = adata.obsm["X_umap"]


# %% Fig 1: UMAP overview
fig = plt.figure(figsize=(15, 9), facecolor="white")
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.18)

ax_p = fig.add_subplot(gs[:, :2])
phenos = list(adata.obs["phenotype"].cat.categories)
order_idx = np.argsort([list(phenos).index(p)
                        for p in adata.obs["phenotype"]])
for p in phenos:
    sel = (adata.obs["phenotype"] == p).values
    ax_p.scatter(X[sel, 0], X[sel, 1],
                 s=2.0, alpha=0.65, linewidths=0,
                 color=TCELL_PHENOTYPE_COLORS[p],
                 label=TCELL_PHENOTYPE_LABELS[p],
                 rasterized=True)
ax_p.set_title("Phenotype", loc="left", fontsize=14, fontweight="bold")
_bare(ax_p)
ax_p.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
            ncol=4, frameon=False, fontsize=9, markerscale=3,
            handletextpad=0.3, columnspacing=0.8)

ax_t = fig.add_subplot(gs[0, 2:])
for tis in adata.obs["tissue"].astype(str).unique():
    sel = (adata.obs["tissue"].astype(str) == tis).values
    ax_t.scatter(X[sel, 0], X[sel, 1], s=1.5, alpha=0.6, linewidths=0,
                 color=TISSUE_COLORS.get(tis, "#999"),
                 label=TISSUE_LABELS.get(tis, tis), rasterized=True)
ax_t.set_title("Tissue", loc="left", fontsize=12, fontweight="bold")
_bare(ax_t)
ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05),
            ncol=3, frameon=False, fontsize=8, markerscale=3,
            handletextpad=0.3, columnspacing=0.8)

ax_pt = fig.add_subplot(gs[1, 2:])
sc_pt = ax_pt.scatter(X[:, 0], X[:, 1], c=pt, s=1.5, alpha=0.7,
                      linewidths=0, cmap="magma", rasterized=True)
ax_pt.set_title("Pseudotime", loc="left", fontsize=12, fontweight="bold")
_bare(ax_pt)
fig.colorbar(sc_pt, ax=ax_pt, fraction=0.04, pad=0.01,
             orientation="vertical")

ax_en = fig.add_subplot(gs[2, 2:])
sc_en = ax_en.scatter(X[:, 0], X[:, 1], c=ent, s=1.5, alpha=0.7,
                      linewidths=0, cmap="viridis", rasterized=True)
ax_en.set_title("Differentiation potential (entropy)", loc="left",
                fontsize=12, fontweight="bold")
_bare(ax_en)
fig.colorbar(sc_en, ax=ax_en, fraction=0.04, pad=0.01,
             orientation="vertical")

_save(fig, f"fig1_umap_overview_{LINEAGE}")


# %% Fig 2: PAGA differentiation order
paga_conn = adata.uns["paga"]["connectivities"].toarray()
paga_phenos = list(adata.obs["phenotype"].cat.categories)
centroids = np.array([
    X[(adata.obs["phenotype"] == p).values].mean(axis=0)
    for p in paga_phenos])
counts = np.array([(adata.obs["phenotype"] == p).sum()
                   for p in paga_phenos], dtype=float)

EDGE_THRESH = 0.05
fig = plt.figure(figsize=(14, 11), facecolor="white")
gs2 = GridSpec(2, 1, figure=fig, hspace=0.28, height_ratios=[1.1, 1.0])

# top: UMAP + PAGA overlay
ax_top = fig.add_subplot(gs2[0])
ax_top.scatter(X[:, 0], X[:, 1], s=1.0, alpha=0.18, linewidths=0,
               color="#cccccc", rasterized=True)
for i in range(len(paga_phenos)):
    for j in range(i + 1, len(paga_phenos)):
        w = paga_conn[i, j]
        if w < EDGE_THRESH:
            continue
        ax_top.plot([centroids[i, 0], centroids[j, 0]],
                    [centroids[i, 1], centroids[j, 1]],
                    color="#444", linewidth=0.8 + 6.0 * w,
                    alpha=min(0.85, 0.25 + w), solid_capstyle="round",
                    zorder=2)
for i, p in enumerate(paga_phenos):
    ax_top.scatter(centroids[i, 0], centroids[i, 1],
                   s=60 + 600 * (counts[i] / counts.max()),
                   color=TCELL_PHENOTYPE_COLORS[p],
                   edgecolors="black", linewidths=1.0, zorder=3)
    ax_top.annotate(TCELL_PHENOTYPE_LABELS[p],
                    (centroids[i, 0], centroids[i, 1]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=10, fontweight="bold")
ax_top.set_title("PAGA on UMAP (centroid graph, edge ≥ 0.05)",
                  loc="left", fontsize=13, fontweight="bold")
_bare(ax_top)

# bottom: median-pseudotime ordering. CD8 splits y into Activated (top)
# vs Quiescent (bottom); CD4 has no Activated phenotypes so it collapses
# to a single row. Within each row, walk nodes in pseudotime order and
# rotate label offsets through 4 levels so labels never overlap even
# when several nodes share nearly-identical median pseudotime.
ax_bot = fig.add_subplot(gs2[1])
median_pt = np.array([
    float(np.median(pt[(adata.obs["phenotype"] == p).values]))
    for p in paga_phenos])

if LINEAGE == "CD8":
    is_activated = np.array(["Activated" in p for p in paga_phenos])
    y_row = np.where(is_activated, 1.0, -1.0)
    row_values = (1.0, -1.0)
else:
    y_row = np.zeros(len(paga_phenos))
    row_values = (0.0,)

label_dy = np.zeros(len(paga_phenos))
ABOVE = [0.55, 1.40]
BELOW = [-0.55, -1.40]
for row_val in row_values:
    members = np.where(y_row == row_val)[0]
    members = members[np.argsort(median_pt[members])]
    if row_val > 0:
        pattern = ABOVE + BELOW
    elif row_val < 0:
        pattern = BELOW + ABOVE
    else:
        pattern = [ABOVE[0], BELOW[0], ABOVE[1], BELOW[1]]
    for k, idx in enumerate(members):
        label_dy[idx] = pattern[k % 4]

for i in range(len(paga_phenos)):
    for j in range(i + 1, len(paga_phenos)):
        w = paga_conn[i, j]
        if w < EDGE_THRESH:
            continue
        ax_bot.plot([median_pt[i], median_pt[j]],
                    [y_row[i], y_row[j]],
                    color="#666", linewidth=0.6 + 5.0 * w,
                    alpha=min(0.85, 0.2 + w), solid_capstyle="round",
                    zorder=1)

for i, p in enumerate(paga_phenos):
    ax_bot.scatter(median_pt[i], y_row[i],
                   s=80 + 1400 * (counts[i] / counts.max()),
                   color=TCELL_PHENOTYPE_COLORS[p],
                   edgecolors="black", linewidths=1.0, zorder=3)
    ly = y_row[i] + label_dy[i]
    ax_bot.plot([median_pt[i], median_pt[i]],
                [y_row[i], ly], color="#999",
                linewidth=0.5, alpha=0.6, zorder=2)
    label = f"{TCELL_PHENOTYPE_LABELS[p]}\nn={int(counts[i]):,}"
    va = "bottom" if label_dy[i] > 0 else "top"
    ax_bot.annotate(label, (median_pt[i], ly),
                    ha="center", va=va, fontsize=9, fontweight="bold")

if LINEAGE == "CD8":
    ax_bot.set_yticks([1.0, -1.0])
    ax_bot.set_yticklabels(["Activated", "Quiescent"],
                            fontsize=11, fontweight="bold")
    ax_bot.set_ylim(-2.6, 2.6)
    ax_bot.axhline(0, color="#ddd", linewidth=0.5, zorder=0)
else:
    ax_bot.set_yticks([])
    ax_bot.set_ylim(-2.0, 2.0)

ax_bot.set_xlabel("Median palantir pseudotime", fontsize=11)
ax_bot.set_xlim(median_pt.min() - 0.05, median_pt.max() + 0.05)
ax_bot.spines["top"].set_visible(False)
ax_bot.spines["right"].set_visible(False)
ax_bot.spines["left"].set_color("#aaa")
ax_bot.spines["bottom"].set_color("#aaa")
ax_bot.set_title(f"{LINEAGE}: phenotypes ordered by median pseudotime "
                  "(edges from PAGA connectivity)",
                  loc="left", fontsize=13, fontweight="bold")

_save(fig, f"fig2_paga_diff_order_{LINEAGE}")


# %% Fig 3: pseudotime profiles
fig = plt.figure(figsize=(14, 8), facecolor="white")
gs3 = GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[1.0, 1.0])

# Top: pseudotime distribution per phenotype, ordered by median
ax_v = fig.add_subplot(gs3[0])
order_pt = [p for _, p in sorted(zip(median_pt, paga_phenos))]
df_v = pd.DataFrame({
    "phenotype": adata.obs["phenotype"].astype(str).values,
    "pseudotime": pt,
})
df_v["phenotype"] = pd.Categorical(df_v["phenotype"], categories=order_pt,
                                    ordered=True)
sns.violinplot(data=df_v, x="phenotype", y="pseudotime", inner="quartile",
               linewidth=0.8, cut=0,
               palette=[TCELL_PHENOTYPE_COLORS[p] for p in order_pt],
               ax=ax_v)
ax_v.set_xticklabels([TCELL_PHENOTYPE_LABELS[p] for p in order_pt],
                      rotation=30, ha="right", fontsize=9)
ax_v.set_title("Pseudotime per phenotype (ordered by median)",
                loc="left", fontsize=13, fontweight="bold")
ax_v.set_xlabel("")
ax_v.set_ylabel("Pseudotime")
ax_v.spines["top"].set_visible(False)
ax_v.spines["right"].set_visible(False)

# Bottom: stacked area of branch probabilities along pseudotime
ax_a = fig.add_subplot(gs3[1])
branch_probs = adata.obsm["palantir_fate_probabilities"]
if isinstance(branch_probs, pd.DataFrame):
    bp_cols = list(branch_probs.columns)
    bp_arr = branch_probs.to_numpy()
else:
    bp_cols = list(ts_series.values)
    bp_arr = np.asarray(branch_probs)

nbin = 50
bins = np.linspace(pt.min(), pt.max(), nbin + 1)
mids = 0.5 * (bins[:-1] + bins[1:])
which = np.clip(np.digitize(pt, bins) - 1, 0, nbin - 1)

binned = np.zeros((nbin, bp_arr.shape[1]), dtype=float)
counts_per_bin = np.zeros(nbin, dtype=float)
for b in range(nbin):
    sel = which == b
    if not sel.any():
        continue
    binned[b] = bp_arr[sel].mean(axis=0)
    counts_per_bin[b] = sel.sum()

# row-normalize so each bin sums to 1 (smooths out empty bins)
row_sums = binned.sum(axis=1, keepdims=True)
binned = np.divide(binned, row_sums, where=row_sums > 0,
                    out=np.zeros_like(binned))

short_to_full = {p.rsplit("_", 1)[-1]: p for p in TERMINALS_LIN}
colors = [TCELL_PHENOTYPE_COLORS.get(short_to_full.get(c, c), "#888")
          for c in bp_cols]
ax_a.stackplot(mids, binned.T, labels=bp_cols, colors=colors, alpha=0.85)
ax_a.set_xlim(mids[0], mids[-1])
ax_a.set_ylim(0, 1)
ax_a.set_xlabel("Pseudotime")
ax_a.set_ylabel("Branch probability")
ax_a.set_title("Branch probabilities along pseudotime "
                f"(binned, n_bin={nbin})",
                loc="left", fontsize=13, fontweight="bold")
ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=len(bp_cols), frameon=False, fontsize=9)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

_save(fig, f"fig3_pseudotime_profiles_{LINEAGE}")

print("\nDone.")
