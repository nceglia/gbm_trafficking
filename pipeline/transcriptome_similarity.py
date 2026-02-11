import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from modules.constants import MIN_CELLS, TISSUE_PAIRS, TISSUES
from modules.pseudobulk import pseudobulk_mean_expression
from modules.similarity import tissue_distances_per_phenotype

warnings.filterwarnings('ignore')

adata = sc.read("GBM_TCR_POS_TCELLS.h5ad")

# ============================================================
# 1. Pseudobulk per phenotype × tissue × patient
# ============================================================

pb_df, expr_mat = pseudobulk_mean_expression(
    adata,
    ["phenotype", "tissue", "patient"],
    min_cells=MIN_CELLS,
)

print(f"Pseudobulk samples: {len(pb_df)}")
print(pb_df.groupby(["phenotype", "tissue"]).size().unstack(fill_value=0))

# ============================================================
# 2. Pairwise cosine distance between tissues per phenotype
#    (using patient-matched pairs where possible)
# ============================================================

dist_df = tissue_distances_per_phenotype(pb_df, expr_mat, TISSUE_PAIRS)

# ============================================================
# 3. Summary: which tissue pair × phenotype is most divergent
# ============================================================

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

# ============================================================
# 4. Per-patient pseudobulk distances
# ============================================================

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

# ============================================================
# 5. Rank tissues by transcriptional uniqueness
# ============================================================

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

# ============================================================
# 6. DEGs per phenotype across tissues (Wilcoxon)
# ============================================================

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

# ============================================================
# 7. Plots
# ============================================================

# Heatmap: aggregate cosine distances
fig, ax = plt.subplots(figsize=(10, 8))
plot_data = agg.copy()
plot_data.index = [x.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_").replace("CD4_", "CD4_") for x in plot_data.index]
sns.heatmap(plot_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Cosine Distance Between Tissues per Phenotype\n(pseudobulk, patients as replicates)", fontsize=12)
plt.tight_layout()
plt.savefig("tissue_cosine_heatmap.png", dpi=200, bbox_inches="tight")
plt.show()

# Patient-matched boxplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for ax, pair in zip(axes, [f"{t1}_vs_{t2}" for t1, t2 in TISSUE_PAIRS]):
    sub = matched[matched["tissue_pair"] == pair]
    if len(sub) == 0:
        ax.set_title(pair)
        continue
    order = sub.groupby("phenotype")["cosine_dist"].mean().sort_values(ascending=False).index
    short = [x.replace("CD8_Activated_", "").replace("CD8_Quiescent_", "").replace("CD4_", "") for x in order]
    sns.boxplot(data=sub, x="phenotype", y="cosine_dist", order=order, ax=ax, palette="Set2")
    sns.stripplot(data=sub, x="phenotype", y="cosine_dist", order=order, ax=ax,
                  color="black", size=4, alpha=0.6)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_title(pair.replace("_", " "), fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Cosine distance" if ax == axes[0] else "")
plt.suptitle("Tissue Divergence per Phenotype (patient-matched)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("tissue_cosine_boxplots.png", dpi=200, bbox_inches="tight")
plt.show()

# Per-patient heatmaps
patients = sorted(pb_df["patient"].unique())
fig, axes = plt.subplots(1, len(patients), figsize=(6 * len(patients), 7), sharey=True)
if len(patients) == 1:
    axes = [axes]
for ax, pat in zip(axes, patients):
    pat_data = matched[matched["patient"] == pat]
    if len(pat_data) == 0:
        ax.set_title(pat)
        continue
    piv = pat_data.pivot_table(index="phenotype", columns="tissue_pair", values="cosine_dist")
    piv.index = [x.replace("CD8_Activated_", "").replace("CD8_Quiescent_", "").replace("CD4_", "") for x in piv.index]
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, linewidths=0.5,
                vmin=0, vmax=0.5, cbar=ax == axes[-1])
    ax.set_title(pat, fontsize=12, fontweight="bold")
    ax.set_ylabel("" if ax != axes[0] else "Phenotype")
plt.suptitle("Per-Patient Tissue Cosine Distances", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("tissue_cosine_per_patient.png", dpi=200, bbox_inches="tight")
plt.show()

print("\nDone.")
