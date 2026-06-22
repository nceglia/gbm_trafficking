# %%
"""Transcriptome similarity between tissues per phenotype.

Pseudobulk per (phenotype, tissue, patient), then pairwise cosine
distances per phenotype × tissue-pair, patient-matched and aggregate.

Reads:
  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad  (paths.H5AD_TCELLS)

Writes to results/traffic_transcriptome_cosine/:
  cosine_distance_summary.csv        (consumed by Figure 2 panels B/C)
  tissue_cosine_heatmap.png
  tissue_cosine_boxplots.png
  tissue_cosine_per_patient.png
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

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.constants import MIN_CELLS, TISSUE_PAIRS, TISSUES  # noqa: E402
from modules.pseudobulk import pseudobulk_mean_expression  # noqa: E402
from modules.similarity import tissue_distances_per_phenotype  # noqa: E402
from modules.style import (  # noqa: E402
    MYELOID_PHENOTYPE_COLORS,
    MYELOID_PHENOTYPE_LABELS,
    MYELOID_PHENOTYPE_ORDER,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TISSUE_COLORS,
    TISSUE_LABELS,
)

_parser = argparse.ArgumentParser(description="Tissue cosine similarity per phenotype.")
_parser.add_argument(
    "--lineage",
    choices=["tcell", "myeloid"],
    default="tcell",
)
_args = _parser.parse_args()

if _args.lineage == "myeloid":
    DATA_PATH = paths.H5AD_MYELOID
    OUTPUT_DIR = paths.TRANSCRIPTOME_SIMILARITY_MYELOID_DIR
    PHENOTYPE_COLORS = MYELOID_PHENOTYPE_COLORS
    PHENOTYPE_LABELS = MYELOID_PHENOTYPE_LABELS
    PHENOTYPE_ORDER = list(MYELOID_PHENOTYPE_ORDER)
else:
    DATA_PATH = paths.H5AD_TCELLS
    OUTPUT_DIR = paths.TRANSCRIPTOME_SIMILARITY_DIR
    PHENOTYPE_COLORS = TCELL_PHENOTYPE_COLORS
    PHENOTYPE_LABELS = TCELL_PHENOTYPE_LABELS
    from modules.style import TCELL_PHENOTYPE_ORDER  # noqa: E402

    PHENOTYPE_ORDER = list(TCELL_PHENOTYPE_ORDER)

paths.ensure(OUTPUT_DIR)

adata = sc.read(str(DATA_PATH))

# %%
# =========================================================
# 1. Pseudobulk per phenotype × tissue × patient
# =========================================================
pb_df, expr_mat = pseudobulk_mean_expression(
    adata,
    ["phenotype", "tissue", "patient"],
    min_cells=MIN_CELLS,
)

print(f"Pseudobulk samples: {len(pb_df)}")
print(pb_df.groupby(["phenotype", "tissue"]).size().unstack(fill_value=0))

# %%
# =========================================================
# 2. Pairwise cosine distance between tissues per phenotype
#    (using patient-matched pairs where possible)
# =========================================================
dist_df = tissue_distances_per_phenotype(pb_df, expr_mat, TISSUE_PAIRS)
dist_df.to_csv(OUTPUT_DIR / "cosine_distance_summary.csv", index=False)

# %%
# =========================================================
# 3. Summary: which tissue pair × phenotype is most divergent
# =========================================================
print("\n" + "=" * 70)
print("AGGREGATE COSINE DISTANCE: PHENOTYPE × TISSUE PAIR")
print("=" * 70)
agg = dist_df[dist_df["type"] == "aggregate"].pivot_table(
    index="phenotype", columns="tissue_pair", values="cosine_dist")
print(agg.round(3).to_string())

print("\n" + "=" * 70)
print("PATIENT-MATCHED COSINE DISTANCES (mean ± sem)")
print("=" * 70)
matched = dist_df[dist_df["type"] == "matched"]
summary = matched.groupby(["phenotype", "tissue_pair"])["cosine_dist"].agg(["mean", "sem", "count"])
print(summary.round(3).to_string())

# %%
# =========================================================
# 4. Per-patient pseudobulk distances
# =========================================================
print("\n" + "=" * 70)
print("PER-PATIENT COSINE DISTANCES")
print("=" * 70)
for pat in sorted(pb_df["patient"].unique()):
    pat_data = matched[matched["patient"] == pat]
    if len(pat_data) == 0:
        continue
    piv = pat_data.pivot_table(index="phenotype", columns="tissue_pair", values="cosine_dist")
    print(f"\n--- {pat} ---")
    print(piv.round(3).to_string())

# %%
# =========================================================
# 5. Rank tissues by transcriptional uniqueness
# =========================================================
print("\n" + "=" * 70)
print("TISSUE UNIQUENESS RANKING (mean cosine distance across phenotypes)")
print("=" * 70)
tissue_scores = {}
for pair in agg.columns:
    t1, t2 = pair.split("_vs_")
    for t in [t1, t2]:
        tissue_scores.setdefault(t, []).append(agg[pair].mean())
for t, scores in sorted(tissue_scores.items(), key=lambda x: -np.mean(x[1])):
    print(f"  {t}: mean dist = {np.mean(scores):.4f}")

# Phenotype ranking by max divergence
print("\nPHENOTYPE RANKING (max cosine distance across any tissue pair):")
max_div = agg.max(axis=1).sort_values(ascending=False)
for pheno, d in max_div.items():
    most_diff_pair = agg.loc[pheno].idxmax()
    print(f"  {pheno}: {d:.4f} ({most_diff_pair})")

# %%
# =========================================================
# 6. DEGs per phenotype across tissues (Wilcoxon)
# =========================================================
print("\n" + "=" * 70)
print("TOP DEGs PER PHENOTYPE × TISSUE (Wilcoxon, single-cell level)")
print("=" * 70)

for pheno in sorted(adata.obs["phenotype"].unique()):
    sub = adata[adata.obs["phenotype"] == pheno].copy()
    if sub.obs["tissue"].nunique() < 2:
        continue
    tissue_counts = sub.obs["tissue"].value_counts()
    if (tissue_counts < MIN_CELLS).any():
        sub = sub[sub.obs["tissue"].isin(tissue_counts[tissue_counts >= MIN_CELLS].index)]
    if sub.obs["tissue"].nunique() < 2:
        continue
    sc.tl.rank_genes_groups(sub, "tissue", method="wilcoxon", use_raw=False)
    print(f"\n--- {pheno} ---")
    for tissue in sub.obs["tissue"].unique():
        df = sc.get.rank_genes_groups_df(sub, tissue)
        top = df[df["pvals_adj"] < 0.05].head(5)
        if len(top) > 0:
            genes = ", ".join(top["names"].tolist())
            print(f"  {tissue} up: {genes}")

# %%
# =========================================================
# 7. Plots
# =========================================================

def _pair_title(pair):
    t1, t2 = pair.split("_vs_")
    return f"{TISSUE_LABELS.get(t1, t1)} vs {TISSUE_LABELS.get(t2, t2)}"


# Heatmap: aggregate cosine distances
fig, ax = plt.subplots(figsize=(10, 8))
plot_data = agg.copy()
plot_data.index = [PHENOTYPE_LABELS.get(x, x) for x in plot_data.index]
plot_data.columns = [_pair_title(c) for c in plot_data.columns]
sns.heatmap(plot_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
            linewidths=0.5)
ax.set_title("Cosine Distance Between Tissues per Phenotype\n"
             "(pseudobulk, patients as replicates)", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tissue_cosine_heatmap.png", dpi=200,
            bbox_inches="tight")

# Patient-matched boxplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, pair in zip(axes, [f"{t1}_vs_{t2}" for t1, t2 in TISSUE_PAIRS]):
    sub = matched[matched["tissue_pair"] == pair]
    if len(sub) == 0:
        ax.set_title(_pair_title(pair))
        continue
    order = (sub.groupby("phenotype")["cosine_dist"].mean()
             .sort_values(ascending=False).index)
    short_labels = [PHENOTYPE_LABELS.get(x, x) for x in order]
    palette = [PHENOTYPE_COLORS.get(p, "#888") for p in order]
    sns.boxplot(data=sub, x="phenotype", y="cosine_dist", order=order,
                ax=ax, palette=palette)
    sns.stripplot(data=sub, x="phenotype", y="cosine_dist", order=order,
                  ax=ax, color="black", size=4, alpha=0.6)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    t1, t2 = pair.split("_vs_")
    ax.set_title(_pair_title(pair), fontsize=12, fontweight="bold",
                 color=TISSUE_COLORS.get(t2, "black"))
    ax.set_xlabel("")
    ax.set_ylabel("Cosine distance" if ax == axes[0] else "")
plt.suptitle("Tissue Divergence per Phenotype (patient-matched)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tissue_cosine_boxplots.png", dpi=200,
            bbox_inches="tight")

# Per-patient heatmaps
patients = sorted(pb_df["patient"].unique())
fig, axes = plt.subplots(1, len(patients), figsize=(6 * len(patients), 7),
                          sharey=True)
if len(patients) == 1:
    axes = [axes]
for ax, pat in zip(axes, patients):
    pat_data = matched[matched["patient"] == pat]
    if len(pat_data) == 0:
        ax.set_title(pat)
        continue
    piv = pat_data.pivot_table(index="phenotype", columns="tissue_pair",
                                values="cosine_dist")
    piv.index = [PHENOTYPE_LABELS.get(x, x) for x in piv.index]
    piv.columns = [_pair_title(c) for c in piv.columns]
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=0.5,
                cbar=ax == axes[-1])
    ax.set_title(pat, fontsize=12, fontweight="bold")
    ax.set_ylabel("" if ax != axes[0] else "Phenotype")
plt.suptitle("Per-Patient Tissue Cosine Distances",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tissue_cosine_per_patient.png", dpi=200,
            bbox_inches="tight")

print("\nDone.")
