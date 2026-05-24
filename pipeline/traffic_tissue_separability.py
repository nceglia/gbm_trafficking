# %%
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.constants import DIRECTED_TISSUE_PAIRS
from modules.style import LINEAGE_COLORS, TISSUE_COLORS

# %%
# ---- Config ----
DATA_PATH = REPO_ROOT / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"
OUTPUT_DIR = REPO_ROOT / "results" / "03_tissue_separability"
CACHE_FILE = OUTPUT_DIR / "augur_results_cache.pkl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = ("PBMC", "CSF", "TP")
TISSUE_PAIRS = list(DIRECTED_TISSUE_PAIRS)
PAIR_LABELS = {"PBMC_TP": "PBMC vs Tumor", "PBMC_CSF": "PBMC vs CSF", "CSF_TP": "CSF vs Tumor"}
MIN_CELLS = 30

# %%
adata = sc.read(str(DATA_PATH))
adata.X = adata.layers["counts"]
print(adata)

# %%
# ============================================================
# 1. Augur per tissue pair (cached)
# ============================================================

if CACHE_FILE.exists():
    print(f"Loading cached Augur results from {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        augur_results = pickle.load(f)
else:
    augur_results = {}
    for t1, t2 in TISSUE_PAIRS:
        pair_key = f"{t1}_{t2}"
        print(f"\n{'='*60}\nAUGUR: {t1} vs {t2}\n{'='*60}")

        sub = adata[adata.obs["tissue"].isin([t1, t2])].copy()
        keep = [
            p for p in sub.obs["phenotype"].unique()
            if (sub[sub.obs["phenotype"] == p].obs["tissue"].value_counts().get(t1, 0) >= MIN_CELLS
                and sub[sub.obs["phenotype"] == p].obs["tissue"].value_counts().get(t2, 0) >= MIN_CELLS)
        ]
        sub = sub[sub.obs["phenotype"].isin(keep)].copy()
        print(f"  Phenotypes with >={MIN_CELLS} cells in both: {len(keep)}")

        ag = pt.tl.Augur("random_forest_classifier")
        loaded, results = ag.predict(
            sub, label_col="tissue", cell_type_col="phenotype",
            n_folds=3, subsample_size=20, n_subsamples=50,
            random_state=42, select_variance_features=True, span=0.75,
        )
        auc_df = results["summary_metrics"].sort_values("mean_augur_score", ascending=False)
        augur_results[pair_key] = {"auc": auc_df, "full": results, "adata": loaded}

        print(f"\n  AUC scores:")
        for _, row in auc_df.iterrows():
            print(f"    {row.name}: AUC = {row['mean_augur_score']:.3f}")

        with open(CACHE_FILE, "wb") as f:
            pickle.dump(augur_results, f)
        print(f"  Cached to {CACHE_FILE}")

# %%
# ============================================================
# 2. Feature importances
# ============================================================

print("\n" + "=" * 60)
print("TOP DISCRIMINATING GENES PER PHENOTYPE × TISSUE PAIR")
print("=" * 60)

feat_importance = {}
for pair_key, res in augur_results.items():
    fi = res["full"]["feature_importances"]
    feat_importance[pair_key] = fi
    gene_col = [c for c in fi.columns if c in ["genes", "gene", "feature"]][0]
    imp_col = [c for c in fi.columns if "import" in c.lower()][0]
    ct_col = [c for c in fi.columns if "cell" in c.lower()][0]
    print(f"\n--- {PAIR_LABELS[pair_key]} ---")
    for pheno in fi[ct_col].unique():
        top = fi[fi[ct_col] == pheno].nlargest(10, imp_col)
        genes = ", ".join([f"{r[gene_col]}({r[imp_col]:.3f})" for _, r in top.iterrows()])
        print(f"  {pheno}: {genes}")

# %%
# ============================================================
# 3. Per-patient Augur
# ============================================================

print("\n" + "=" * 60)
print("PER-PATIENT AUGUR")
print("=" * 60)

patient_records = []
for t1, t2 in TISSUE_PAIRS:
    pair_key = f"{t1}_{t2}"
    sub = adata[adata.obs["tissue"].isin([t1, t2])].copy()
    for pat in sorted(adata.obs["patient"].unique()):
        pat_sub = sub[sub.obs["patient"] == pat].copy()
        keep = [
            p for p in pat_sub.obs["phenotype"].unique()
            if (pat_sub[pat_sub.obs["phenotype"] == p].obs["tissue"].value_counts().get(t1, 0) >= 10
                and pat_sub[pat_sub.obs["phenotype"] == p].obs["tissue"].value_counts().get(t2, 0) >= 10)
        ]
        if not keep:
            continue
        pat_sub = pat_sub[pat_sub.obs["phenotype"].isin(keep)].copy()
        try:
            ag = pt.tl.Augur("random_forest_classifier")
            _, results = ag.predict(
                pat_sub, label_col="tissue", cell_type_col="phenotype",
                n_folds=3, subsample_size=15, n_subsamples=25,
                random_state=42, select_variance_features=True, span=0.75,
            )
            for _, row in results["summary_metrics"].iterrows():
                patient_records.append({
                    "pair": pair_key, "patient": pat,
                    "phenotype": row.name, "AUC": row["mean_augur_score"],
                })
        except Exception as e:
            print(f"  {pair_key} {pat}: failed ({e})")

patient_df = pd.DataFrame(patient_records)
if len(patient_df) > 0:
    print("\nPer-patient AUC summary (mean ± sem):")
    print(patient_df.groupby(["pair", "phenotype"])["AUC"]
          .agg(["mean", "sem", "count"]).round(3).to_string())

# %%
# ============================================================
# 4. Cosine distances per phenotype × tissue pair
# ============================================================

print("\n" + "=" * 60)
print("COSINE DISTANCES")
print("=" * 60)

cosine_records = []
for t1, t2 in TISSUE_PAIRS:
    pair_key = f"{t1}_{t2}"
    for pheno in adata.obs["phenotype"].unique():
        s1 = adata[(adata.obs["phenotype"] == pheno) & (adata.obs["tissue"] == t1)]
        s2 = adata[(adata.obs["phenotype"] == pheno) & (adata.obs["tissue"] == t2)]
        if s1.n_obs < MIN_CELLS or s2.n_obs < MIN_CELLS:
            continue
        X1 = s1.X.toarray().mean(0) if sparse.issparse(s1.X) else s1.X.mean(0)
        X2 = s2.X.toarray().mean(0) if sparse.issparse(s2.X) else s2.X.mean(0)
        cosine_records.append({
            "pair": pair_key, "phenotype": pheno,
            "cosine_dist": cosine(np.asarray(X1).flatten(), np.asarray(X2).flatten()),
        })

cosine_df = pd.DataFrame(cosine_records)
print(cosine_df.pivot_table(index="phenotype", columns="pair", values="cosine_dist").round(4).to_string())

# %%
# ============================================================
# 5. AUC vs cosine distance correlation
# ============================================================

print("\n" + "=" * 60)
print("AUGUR AUC vs COSINE DISTANCE")
print("=" * 60)

for pair_key, res in augur_results.items():
    auc = res["auc"].reset_index()
    auc.columns = ["phenotype", "AUC"]
    merged = auc.merge(cosine_df[cosine_df["pair"] == pair_key], on="phenotype")
    if len(merged) > 2:
        rho, pval = spearmanr(merged["AUC"], merged["cosine_dist"])
        print(f"  {PAIR_LABELS[pair_key]}: rho={rho:.3f}, p={pval:.3f} (n={len(merged)})")

# %%
# ============================================================
# 6. Figure: AUC barplot
# ============================================================

def shorten(label):
    return (label.replace("CD8_Activated_", "CD8a_")
                 .replace("CD8_Quiescent_", "CD8q_")
                 .replace("CD4_", "CD4_"))

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
for ax, (t1, t2) in zip(axes, TISSUE_PAIRS):
    pair_key = f"{t1}_{t2}"
    if pair_key not in augur_results:
        continue
    auc = augur_results[pair_key]["auc"]["mean_augur_score"].sort_values()
    colors = [LINEAGE_COLORS["CD8"] if "CD8" in p else LINEAGE_COLORS["CD4"] for p in auc.index]
    ax.barh([shorten(p) for p in auc.index], auc.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("AUC")
    ax.set_title(PAIR_LABELS[pair_key], fontsize=13, fontweight="bold")
    ax.set_xlim(0.4, 1.0)
plt.suptitle("Augur: Tissue Separability per Phenotype (AUC)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "augur_auc_barplot.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# ============================================================
# 7. Figure: per-patient AUC boxplots
# ============================================================

if len(patient_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for ax, (t1, t2) in zip(axes, TISSUE_PAIRS):
        pair_key = f"{t1}_{t2}"
        sub = patient_df[patient_df["pair"] == pair_key]
        if len(sub) == 0:
            continue
        order = sub.groupby("phenotype")["AUC"].mean().sort_values(ascending=False).index
        short = [shorten(p).replace("CD8a_", "").replace("CD8q_", "").replace("CD4_", "")
                 for p in order]
        sns.boxplot(data=sub, x="phenotype", y="AUC", order=order, ax=ax, palette="Set2")
        sns.stripplot(data=sub, x="phenotype", y="AUC", order=order, ax=ax,
                      color="black", size=4, alpha=0.6)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(PAIR_LABELS[pair_key], fontsize=12, fontweight="bold")
        ax.set_xlabel("")
    plt.suptitle("Augur AUC per Patient", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "augur_per_patient.png", dpi=200, bbox_inches="tight")
    plt.show()

# %%
# ============================================================
# 8. Figure: AUC vs cosine distance scatter
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (t1, t2) in zip(axes, TISSUE_PAIRS):
    pair_key = f"{t1}_{t2}"
    if pair_key not in augur_results:
        continue
    auc = augur_results[pair_key]["auc"].reset_index()
    auc.columns = ["phenotype", "AUC"]
    merged = auc.merge(cosine_df[cosine_df["pair"] == pair_key], on="phenotype")
    if len(merged) == 0:
        continue
    colors = [LINEAGE_COLORS["CD8"] if "CD8" in p else LINEAGE_COLORS["CD4"] for p in merged["phenotype"]]
    ax.scatter(merged["cosine_dist"], merged["AUC"], c=colors, s=60, edgecolors="black", linewidth=0.5)
    for _, row in merged.iterrows():
        short = shorten(row["phenotype"]).replace("CD8a_", "").replace("CD8q_", "").replace("CD4_", "")
        ax.annotate(short, (row["cosine_dist"], row["AUC"]), fontsize=6, ha="left",
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Augur AUC")
    ax.set_title(PAIR_LABELS[pair_key], fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
plt.suptitle("Augur AUC vs Cosine Distance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "augur_vs_cosine.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# ============================================================
# 9. Figure: feature importance heatmaps
# ============================================================

for pair_key, fi in feat_importance.items():
    gene_col = [c for c in fi.columns if c in ["genes", "gene", "feature"]][0]
    imp_col = [c for c in fi.columns if "import" in c.lower()][0]
    ct_col = [c for c in fi.columns if "cell" in c.lower()][0]

    top_genes = fi.groupby(gene_col)[imp_col].mean().nlargest(25).index.tolist()
    piv = fi[fi[gene_col].isin(top_genes)].pivot_table(
        index=gene_col, columns=ct_col, values=imp_col, aggfunc="mean")
    piv.columns = [shorten(c) for c in piv.columns]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 1.2), max(5, len(piv) * 0.3)))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(f"Top Discriminating Genes: {PAIR_LABELS[pair_key]}", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"augur_features_{pair_key}.png", dpi=200, bbox_inches="tight")
    plt.show()

# %%
# ============================================================
# 10. Save summary tables
# ============================================================

auc_records = []
for pair_key, res in augur_results.items():
    auc = res["auc"]
    for pheno in auc.index:
        auc_records.append({"pair": pair_key, "phenotype": pheno, "AUC": auc.loc[pheno, "mean_augur_score"]})

auc_df = pd.DataFrame(auc_records)
auc_pivot = auc_df.pivot_table(index="phenotype", columns="pair", values="AUC")
cosine_pivot = cosine_df.pivot_table(index="phenotype", columns="pair", values="cosine_dist")

auc_pivot.to_csv(OUTPUT_DIR / "augur_auc_summary.csv")
cosine_pivot.to_csv(OUTPUT_DIR / "cosine_distance_summary.csv")
if len(patient_df) > 0:
    patient_df.to_csv(OUTPUT_DIR / "augur_per_patient.csv", index=False)

print(f"\nAll outputs saved to {OUTPUT_DIR}")
print("Done.")