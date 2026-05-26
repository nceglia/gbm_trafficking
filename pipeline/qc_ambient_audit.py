# %%
"""Ambient-RNA audit on the T-cell singlets AnnData.

Diagnostic only. Answers four questions before any decontamination tool
is run:

  (a) Is off-lineage contamination uniformly distributed across batches,
      or concentrated in specific (patient × tissue) samples?
  (b) Is it concentrated in specific T-cell phenotypes?
  (c) Does it correlate with cell UMI total (low-UMI ambient signal vs
      high-UMI doublet signal)?
  (d) Are unrelated lineage programs positively correlated within
      batches (ambient soup fingerprint)?

Reads:
  paths.H5AD_TCELLS                           (Scrublet-filtered singlets)
  paths.cellranger_h5(sample) for every sample (per-sample soup)

Writes to results/qc_ambient_audit/:
  per_cell_scores.csv, per_sample_soup.csv, contamination_fraction.csv,
  batch_summary.csv, D1..D6 plots, audit_summary.txt
"""
import sys
import warnings
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.style import (  # noqa: E402
    LINEAGE_COLORS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_ORDER,
)

paths.ensure(paths.QC_DOUBLETS_DIR.parent)
OUT_DIR = paths.RESULTS_DIR / "qc_ambient_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200

# Marker panels — small, conservative, GBM-appropriate.
MYELOID_MARKERS = [
    "CD14", "CD68", "CSF1R", "LYZ", "S100A8", "S100A9", "TYROBP",
    "FCER1G", "AIF1", "C1QA", "C1QB", "APOE", "APOC1", "IFI30", "SPI1",
]
PLASMA_B_MARKERS = [
    "CD19", "MS4A1", "CD79A", "CD79B", "IGHM", "IGHG1", "IGHA1",
    "JCHAIN", "MZB1", "XBP1",
]
T_POSITIVE_MARKERS = [
    "CD3D", "CD3E", "CD3G", "TRAC", "TRBC1", "TRBC2",
    "CD4", "CD8A", "CD8B", "IL7R", "LEF1", "TCF7",
]
PANELS = {
    "myeloid": MYELOID_MARKERS,
    "plasma":  PLASMA_B_MARKERS,
    "tcell":   T_POSITIVE_MARKERS,
}

# Bottom-decile of UMI per sample is a soup proxy (mirrors SoupX's
# "background cluster" trick when raw matrices aren't available).
LOW_UMI_PCT = 0.10


# %%
# =========================================================
# Load singlets AnnData + score every cell
# =========================================================
print(f"Loading {paths.H5AD_TCELLS.name}...")
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

if "log1p" in adata.layers:
    adata.X = adata.layers["log1p"]
    print("  scoring on layers['log1p']")
else:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("  log-normalized for scoring")

print("Scoring marker panels with sc.tl.score_genes...")
for panel_name, genes in PANELS.items():
    present = [g for g in genes if g in adata.var_names]
    missing = sorted(set(genes) - set(present))
    if missing:
        print(f"  {panel_name}: using {len(present)}/{len(genes)} "
              f"(missing: {','.join(missing)})")
    sc.tl.score_genes(adata, gene_list=present,
                       score_name=f"{panel_name}_score",
                       use_raw=False)

adata.obs["sample"] = adata.obs["sample"].astype(str)
adata.obs["patient"] = adata.obs["patient"].astype(str)
adata.obs["tissue"] = adata.obs["tissue"].astype(str)
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
adata.obs["phenotype"] = adata.obs["phenotype"].astype(str)
adata.obs["umi_total"] = np.asarray(
    adata.layers["counts"].sum(axis=1)).ravel()

cell_cols = ["sample", "patient", "tissue", "timepoint", "phenotype",
             "umi_total", "myeloid_score", "plasma_score", "tcell_score"]
per_cell = adata.obs[cell_cols].copy()
per_cell.to_csv(OUT_DIR / "per_cell_scores.csv", index=True)
print(f"  wrote per_cell_scores.csv ({len(per_cell):,} rows)")


# %%
# =========================================================
# Per-sample soup estimate from cellranger filtered h5
# =========================================================
print("\nReading 78 cellranger filtered h5s for per-sample soup...")


def _read_filtered_h5(path):
    """Load a CellRanger filtered_feature_bc_matrix.h5 as (X_csr, gene_symbols)."""
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(m["shape"][:])
        # features sub-group has 'name' (gene symbol) and 'id' (Ensembl).
        feat = m["features"]
        names = feat["name"][:].astype(str)
    # CellRanger writes (n_features, n_barcodes) CSC; convert to csr (barcodes × features).
    X_csc = sp.csc_matrix((data, indices, indptr), shape=shape)
    X = X_csc.T.tocsr()
    return X, names


# Build lookup of indices into the singlets-h5ad gene name space so we
# can compute scores in cellranger features and join back.
gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

samples = sorted(adata.obs["sample"].unique())
print(f"  {len(samples)} samples present in obs")

soup_rows = []
low_umi_soup_rows = []
sample_to_soup = {}  # sample -> dict(gene -> proportion)
sample_to_low_soup = {}

for s in tqdm(samples, desc="  soup"):
    h5_path = paths.cellranger_h5(s)
    if not h5_path.exists():
        print(f"  WARNING: missing {h5_path.name}, skipping"); continue
    X, gene_names = _read_filtered_h5(h5_path)
    n_droplets = X.shape[0]
    umi_per_droplet = np.asarray(X.sum(axis=1)).ravel()
    # Global soup = sum of all counts, normalized to fractions.
    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    if total.sum() > 0:
        global_soup = total / total.sum()
    else:
        global_soup = total
    # Low-UMI soup = soup estimated from bottom-decile-UMI droplets.
    if n_droplets >= 100:
        threshold = np.quantile(umi_per_droplet, LOW_UMI_PCT)
        low_mask = umi_per_droplet <= threshold
    else:
        low_mask = np.ones(n_droplets, dtype=bool)
    if low_mask.any():
        low_total = np.asarray(X[low_mask].sum(axis=0)).ravel().astype(float)
        low_soup = low_total / max(low_total.sum(), 1.0)
    else:
        low_soup = global_soup

    # Store as gene→proportion dict (using cellranger gene symbols).
    sample_to_soup[s] = dict(zip(gene_names, global_soup))
    sample_to_low_soup[s] = dict(zip(gene_names, low_soup))

    # Compact row form (long-form CSV with all panel genes).
    panel_genes = set(MYELOID_MARKERS + PLASMA_B_MARKERS + T_POSITIVE_MARKERS)
    for gname, gprop in zip(gene_names, global_soup):
        if gname not in panel_genes:
            continue
        soup_rows.append({
            "sample": s, "gene": gname, "soup_prop": float(gprop),
            "soup_kind": "global",
        })
    for gname, gprop in zip(gene_names, low_soup):
        if gname not in panel_genes:
            continue
        low_umi_soup_rows.append({
            "sample": s, "gene": gname, "soup_prop": float(gprop),
            "soup_kind": "low_umi",
        })

soup_df = pd.DataFrame(soup_rows + low_umi_soup_rows)
soup_df.to_csv(OUT_DIR / "per_sample_soup.csv", index=False)
print(f"  wrote per_sample_soup.csv ({len(soup_df):,} rows)")


# %%
# =========================================================
# Per-cell contamination math
#  For each off-lineage panel marker, compute fraction of the cell's
#  expression explainable by sample-level soup × cell UMI.
# =========================================================
print("\nComputing per-cell contamination-explained fractions...")

# Counts matrix for the off-lineage genes that exist in the AnnData.
contam_genes = sorted(set(MYELOID_MARKERS + PLASMA_B_MARKERS)
                       & set(adata.var_names))
contam_idx = np.array([gene_to_idx[g] for g in contam_genes])
counts = adata.layers["counts"][:, contam_idx]
counts_arr = np.asarray(counts.todense() if sp.issparse(counts)
                         else counts)

sample_arr = adata.obs["sample"].values
umi_total_arr = adata.obs["umi_total"].values

# soup_matrix[sample, gene] in panel-gene space.
soup_matrix = np.zeros((len(samples), len(contam_genes)))
sample_idx = {s: i for i, s in enumerate(samples)}
for s, soup_dict in sample_to_low_soup.items():
    if s not in sample_idx:
        continue
    si = sample_idx[s]
    for gi, gname in enumerate(contam_genes):
        soup_matrix[si, gi] = soup_dict.get(gname, 0.0)

# For each cell, expected ambient counts per gene = soup_prop * umi_total
# Per-cell sample index:
cell_sample_idx = np.array([sample_idx.get(s, -1) for s in sample_arr])
valid = cell_sample_idx >= 0
expected = np.zeros_like(counts_arr, dtype=float)
expected[valid] = (soup_matrix[cell_sample_idx[valid]]
                   * umi_total_arr[valid, None])

# Contamination-explained fraction per (cell, gene) = min(expected/observed, 1).
with np.errstate(divide="ignore", invalid="ignore"):
    frac = np.where(counts_arr > 0,
                    np.clip(expected / np.maximum(counts_arr, 1e-9), 0, 1),
                    0.0)

# Per-cell mean across off-lineage genes — a single "this cell looks
# ambient-contaminated" number.
contam_per_cell = frac.mean(axis=1)
adata.obs["contam_fraction"] = contam_per_cell

# Per-sample mean.
contam_per_sample = (pd.DataFrame({
    "sample": sample_arr,
    "contam_fraction": contam_per_cell,
}).groupby("sample")["contam_fraction"]
   .agg(["mean", "median", "std", "count"])
   .reset_index())
contam_per_sample.to_csv(OUT_DIR / "contamination_fraction.csv", index=False)
print(f"  wrote contamination_fraction.csv (per-sample summary)")


# %%
# =========================================================
# Batch summary table
# =========================================================
batch = (adata.obs.groupby(["patient", "tissue"], observed=True)
         .agg(n_cells=("sample", "size"),
              myeloid_median=("myeloid_score", "median"),
              myeloid_q95=("myeloid_score", lambda s: s.quantile(0.95)),
              plasma_median=("plasma_score", "median"),
              plasma_q95=("plasma_score", lambda s: s.quantile(0.95)),
              tcell_median=("tcell_score", "median"),
              contam_mean=("contam_fraction", "mean"),
              contam_q95=("contam_fraction",
                          lambda s: s.quantile(0.95)),
              )
         .reset_index())
batch.to_csv(OUT_DIR / "batch_summary.csv", index=False)
print(f"  wrote batch_summary.csv")


# %%
# =========================================================
# D1 — off-lineage scores per (patient × tissue), boxplots
# =========================================================
print("\nPlotting D1...")
batches = sorted(adata.obs.groupby(["patient", "tissue"], observed=True).groups.keys())
batch_labels = [f"{p}|{t}" for p, t in batches]
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
for ax, panel, color_anchor in [
    (axes[0], "myeloid", "#a87521"),
    (axes[1], "plasma",  "#a3387c"),
]:
    data = []
    colors = []
    for p, t in batches:
        sub = adata.obs[(adata.obs["patient"] == p)
                        & (adata.obs["tissue"] == t)]
        data.append(sub[f"{panel}_score"].values)
        colors.append(TISSUE_COLORS.get(t, "#888"))
    bp = ax.boxplot(data, positions=range(len(data)),
                    showfliers=False, widths=0.65, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)
        patch.set_edgecolor("black"); patch.set_linewidth(0.4)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(0.8)
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batch_labels, rotation=60, ha="right", fontsize=7)
    ax.axhline(0, color="#888", lw=0.5, ls="--")
    ax.set_title(f"{panel} score per (patient × tissue)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{panel}_score", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "D1_offlineage_by_batch.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# D2 — off-lineage scores per T-cell phenotype
# =========================================================
print("Plotting D2...")
phenotypes = [p for p in TCELL_PHENOTYPE_ORDER
              if p in adata.obs["phenotype"].unique()]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
for ax, panel in [(axes[0], "myeloid"), (axes[1], "plasma")]:
    data = []
    for p in phenotypes:
        sub = adata.obs[adata.obs["phenotype"] == p]
        data.append(sub[f"{panel}_score"].values)
    bp = ax.boxplot(data, positions=range(len(data)),
                    showfliers=False, widths=0.65, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cccccc"); patch.set_alpha(0.6)
        patch.set_edgecolor("black"); patch.set_linewidth(0.4)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(0.8)
    ax.set_xticks(range(len(phenotypes)))
    ax.set_xticklabels(phenotypes, rotation=55, ha="right", fontsize=7)
    ax.axhline(0, color="#888", lw=0.5, ls="--")
    ax.set_title(f"{panel} score by T-cell phenotype",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel(f"{panel}_score", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "D2_offlineage_by_phenotype.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# D3 — off-lineage score vs cell UMI, faceted by tissue
# =========================================================
print("Plotting D3...")
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey="row")
for col, t in enumerate(TISSUE_ORDER):
    sub = adata.obs[adata.obs["tissue"] == t]
    if sub.empty: continue
    for row, panel in enumerate(("myeloid", "plasma")):
        ax = axes[row, col]
        ax.scatter(sub["umi_total"], sub[f"{panel}_score"],
                   s=2, c=TISSUE_COLORS.get(t, "#888"),
                   alpha=0.25, edgecolors="none", rasterized=True)
        ax.set_xscale("log")
        ax.axhline(0, color="#888", lw=0.5, ls="--")
        if row == 0:
            ax.set_title(f"{t}", fontsize=10, fontweight="bold",
                         color=TISSUE_COLORS.get(t, "#888"))
        if row == 1:
            ax.set_xlabel("UMI total (log)", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"{panel}_score", fontsize=9)
        # Pearson r per panel as small text top-left
        x = np.log10(np.maximum(sub["umi_total"].values, 1))
        y = sub[f"{panel}_score"].values
        if len(x) > 5:
            r = float(np.corrcoef(x, y)[0, 1])
            ax.text(0.02, 0.95, f"r = {r:+.2f}",
                    transform=ax.transAxes, fontsize=8,
                    va="top", ha="left",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.8, pad=1))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
fig.suptitle("Off-lineage score vs cell UMI total",
             fontsize=11, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "D3_score_vs_umi.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# D4 — within-batch cross-program correlation (T × Myeloid, T × Plasma)
# =========================================================
print("Plotting D4...")
corr_rows = []
for (p, t), sub in adata.obs.groupby(["patient", "tissue"], observed=True):
    if len(sub) < 10: continue
    rm = float(np.corrcoef(sub["tcell_score"], sub["myeloid_score"])[0, 1])
    rp = float(np.corrcoef(sub["tcell_score"], sub["plasma_score"])[0, 1])
    rmp = float(np.corrcoef(sub["myeloid_score"], sub["plasma_score"])[0, 1])
    corr_rows.append({"patient": p, "tissue": t,
                      "t_vs_myeloid": rm, "t_vs_plasma": rp,
                      "myeloid_vs_plasma": rmp, "n_cells": len(sub)})
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT_DIR / "cross_program_correlations.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4.5))
mat = corr_df[["t_vs_myeloid", "t_vs_plasma",
               "myeloid_vs_plasma"]].values
batch_lab = [f"{r['patient']}|{r['tissue']}" for _, r in corr_df.iterrows()]
im = ax.imshow(mat.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_yticks(range(3))
ax.set_yticklabels(["T × Myeloid", "T × Plasma", "Myeloid × Plasma"],
                   fontsize=9)
ax.set_xticks(range(len(batch_lab)))
ax.set_xticklabels(batch_lab, rotation=60, ha="right", fontsize=7)
for i, lab in enumerate(batch_lab):
    tis = lab.split("|")[-1]
    ax.get_xticklabels()[i].set_color(TISSUE_COLORS.get(tis, "#444"))
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(i, j, f"{mat[i, j]:+.2f}",
                ha="center", va="center", fontsize=6,
                color="white" if abs(mat[i, j]) > 0.5 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("Pearson r", fontsize=9)
cb.ax.tick_params(labelsize=7)
ax.set_title("Within-batch cross-program correlation "
             "(ambient soup → positive cross-correlation)",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "D4_corr_heatmap.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# D5 — per-sample top-25 soup genes, annotated by lineage
# =========================================================
print("Plotting D5...")
top_n = 25
# Use global soup.
top_rows = []
for s in samples:
    soup = sample_to_soup[s]
    items = sorted(soup.items(), key=lambda kv: -kv[1])[:top_n]
    for rank, (gene, prop) in enumerate(items, start=1):
        if gene in MYELOID_MARKERS:    tag = "myeloid"
        elif gene in PLASMA_B_MARKERS: tag = "plasma"
        elif gene in T_POSITIVE_MARKERS: tag = "tcell"
        else: tag = "other"
        top_rows.append({"sample": s, "rank": rank, "gene": gene,
                         "soup_prop": float(prop), "tag": tag})
top_df = pd.DataFrame(top_rows)
top_df.to_csv(OUT_DIR / "top_genes_per_soup.csv", index=False)

# Bar count of off-lineage genes in each sample's top-25.
tag_counts = (top_df[top_df["tag"].isin(["myeloid", "plasma"])]
              .groupby(["sample", "tag"]).size().unstack(fill_value=0))
tag_counts = tag_counts.reindex(samples).fillna(0).astype(int)
# Add tissue for color.
sample_tissue = (adata.obs.groupby("sample", observed=True)["tissue"]
                 .first().to_dict())
fig, ax = plt.subplots(figsize=(14, 5))
xs = np.arange(len(samples))
w = 0.4
for i, panel in enumerate(["myeloid", "plasma"]):
    ys = tag_counts.get(panel, pd.Series(0, index=samples)).values
    ax.bar(xs + (i - 0.5) * w, ys, width=w,
           label=panel,
           color=("#a87521" if panel == "myeloid" else "#a3387c"),
           edgecolor="black", linewidth=0.3)
ax.set_xticks(xs)
ax.set_xticklabels(samples, rotation=80, ha="right", fontsize=6)
for tick_lab, s in zip(ax.get_xticklabels(), samples):
    tick_lab.set_color(TISSUE_COLORS.get(sample_tissue.get(s, ""), "#444"))
ax.set_ylabel(f"# off-lineage genes in top-{top_n} soup", fontsize=9)
ax.set_title(f"Per-sample soup: count of myeloid / plasma genes "
             f"in top-{top_n} expressed",
             fontsize=10, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "D5_soup_top_genes.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# D6 — per-sample mean contamination-explained fraction
# =========================================================
print("Plotting D6...")
contam_per_sample_sorted = contam_per_sample.sort_values(
    "mean", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14, 4.5))
xs = np.arange(len(contam_per_sample_sorted))
cols = [TISSUE_COLORS.get(sample_tissue.get(s, ""), "#888")
        for s in contam_per_sample_sorted["sample"]]
ax.bar(xs, contam_per_sample_sorted["mean"], color=cols,
       edgecolor="black", linewidth=0.3)
ax.errorbar(xs, contam_per_sample_sorted["mean"],
            yerr=contam_per_sample_sorted["std"]
                  / np.sqrt(contam_per_sample_sorted["count"].clip(lower=1)),
            fmt="none", ecolor="black", elinewidth=0.5, capsize=1.5)
ax.set_xticks(xs)
ax.set_xticklabels(contam_per_sample_sorted["sample"],
                   rotation=80, ha="right", fontsize=6)
for lab, s in zip(ax.get_xticklabels(),
                  contam_per_sample_sorted["sample"]):
    lab.set_color(TISSUE_COLORS.get(sample_tissue.get(s, ""), "#444"))
ax.set_ylabel("Mean contamination-explained fraction\n(low-UMI soup)",
              fontsize=9)
ax.set_title("Per-sample contamination of off-lineage marker counts "
             "(ranked desc; high = soup explains most of the signal)",
             fontsize=10, fontweight="bold")
ax.axhline(0.05, color="#aa3333", lw=0.6, ls="--",
           label="0.05 (low)")
ax.axhline(0.20, color="#660000", lw=0.6, ls="--",
           label="0.20 (high)")
ax.legend(loc="upper right", fontsize=8, frameon=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "D6_contam_fraction_per_sample.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Verdict — pattern characterization, no tool recommendation
# =========================================================
print("\nGenerating verdict...")
mean_contam_per_sample = (
    contam_per_sample.set_index("sample")["mean"].to_dict())
overall_mean = float(np.nanmean(list(mean_contam_per_sample.values())))
n_low  = int(sum(v < 0.05 for v in mean_contam_per_sample.values()))
n_mid  = int(sum(0.05 <= v < 0.20 for v in mean_contam_per_sample.values()))
n_high = int(sum(v >= 0.20 for v in mean_contam_per_sample.values()))
top5 = sorted(mean_contam_per_sample.items(),
              key=lambda kv: -kv[1])[:5]

mean_corr_tm = float(corr_df["t_vs_myeloid"].mean())
mean_corr_tp = float(corr_df["t_vs_plasma"].mean())
mean_corr_mp = float(corr_df["myeloid_vs_plasma"].mean())
n_batches_pos_mp = int((corr_df["myeloid_vs_plasma"] > 0.1).sum())

# Phenotype concentration
phen_means = (adata.obs.groupby("phenotype", observed=True)
              [["myeloid_score", "plasma_score"]].mean().sort_values(
                  "myeloid_score", ascending=False))
top_phen = phen_means.head(3).index.tolist()

# UMI–score correlation (per tissue), reported in D3
umi_corrs = {}
for t in TISSUE_ORDER:
    sub = adata.obs[adata.obs["tissue"] == t]
    if len(sub) < 5: continue
    x = np.log10(np.maximum(sub["umi_total"].values, 1))
    for panel in ("myeloid", "plasma"):
        y = sub[f"{panel}_score"].values
        umi_corrs[(t, panel)] = float(np.corrcoef(x, y)[0, 1])

summary = []
summary.append("======== Ambient-RNA audit verdict ========")
summary.append("")
summary.append(f"Input: {paths.H5AD_TCELLS.name}")
summary.append(f"Cells scored: {adata.n_obs:,}")
summary.append(f"Samples: {len(samples)}")
summary.append("")
summary.append("(1) Distribution across batches:")
summary.append(f"    overall mean contamination-explained fraction = "
               f"{overall_mean:.3f}")
summary.append(f"    samples by tier: {n_low} <0.05  |  "
               f"{n_mid} 0.05–0.20  |  {n_high} ≥0.20")
summary.append("    top 5 most-contaminated samples:")
for s, v in top5:
    summary.append(f"      {s:<35s}  {v:.3f}  "
                   f"(tissue={sample_tissue.get(s, '?')})")
summary.append("")
summary.append("(2) Cross-program correlation (D4):")
summary.append(f"    mean Pearson T × Myeloid    = {mean_corr_tm:+.3f}")
summary.append(f"    mean Pearson T × Plasma     = {mean_corr_tp:+.3f}")
summary.append(f"    mean Pearson Myeloid × Plasma = {mean_corr_mp:+.3f}")
summary.append(f"    batches with Myeloid×Plasma > 0.10: "
               f"{n_batches_pos_mp}/{len(corr_df)}")
summary.append("")
summary.append("(3) Phenotype concentration (D2):")
summary.append("    top 3 phenotypes by mean myeloid_score:")
for p in top_phen:
    summary.append(f"      {p:<35s}  "
                   f"myeloid={phen_means.loc[p, 'myeloid_score']:+.3f}  "
                   f"plasma={phen_means.loc[p, 'plasma_score']:+.3f}")
summary.append("")
summary.append("(4) UMI dependence (D3):")
for (t, panel), r in sorted(umi_corrs.items()):
    summary.append(f"    {t:<5s} {panel:<8s} Pearson(log10(UMI), score) = {r:+.3f}")
summary.append("")
summary.append("Pattern characterization:")
verdict = []
if overall_mean < 0.05:
    verdict.append(
        "Contamination is low overall (<5% of off-lineage counts "
        "explainable by soup). No batch crosses the 20% high tier.")
elif n_high > 0:
    verdict.append(
        f"Contamination is heterogeneous: {n_high} sample(s) in the "
        f"high tier (≥0.20). See top-5 list above.")
else:
    verdict.append(
        f"Contamination is moderate (mean {overall_mean:.3f}); no "
        "single batch is extreme.")
if mean_corr_mp > 0.20:
    verdict.append(
        "Myeloid × Plasma correlation is positive on average — "
        "consistent with a shared ambient soup leaking both programs "
        "into T cells.")
elif mean_corr_mp < 0.05:
    verdict.append(
        "Cross-lineage program correlation is near zero — argues "
        "against systemic ambient leakage.")
if mean_corr_tm > 0.30:
    verdict.append(
        "T × Myeloid correlation is positive — programs are not "
        "orthogonal, may indicate residual doublets or biology "
        "(e.g., TRM-myeloid axis).")
max_umi_r = max(abs(r) for r in umi_corrs.values()) if umi_corrs else 0.0
if max_umi_r > 0.30:
    verdict.append(
        f"Strong UMI–score relationship in at least one tissue "
        f"(|r|={max_umi_r:.2f}) — direction matters: positive r = "
        "doublet-like, negative r = low-UMI ambient.")
for line in verdict:
    summary.append(f"  • {line}")
summary.append("")
summary.append("(Tool recommendations intentionally omitted — diagnostic only.)")

verdict_text = "\n".join(summary)
print()
print(verdict_text)
(OUT_DIR / "audit_summary.txt").write_text(verdict_text + "\n")
print(f"\n  wrote audit_summary.txt")

print(f"\nDone. All outputs in {OUT_DIR}")
