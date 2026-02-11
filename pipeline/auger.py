import scanpy as sc
import pandas as pd
import numpy as np
import pertpy as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import pickle, os
import warnings
warnings.filterwarnings('ignore')

adata = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")
adata.X = adata.layers["counts"]
TISSUE_PAIRS = [("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP")]
PAIR_LABELS = {"PBMC_TP": "PBMC vs Tumor", "PBMC_CSF": "PBMC vs CSF", "CSF_TP": "CSF vs Tumor"}
MIN_CELLS = 30
CACHE_FILE = "augur_results_cache.pkl"

# ============================================================
# 1. Run Augur per tissue pair (with caching)
# ============================================================

if os.path.exists(CACHE_FILE):
    print(f"Loading cached Augur results from {CACHE_FILE}")
    with open(CACHE_FILE, "rb") as f:
        augur_results = pickle.load(f)
    for pair_key, res in augur_results.items():
        print(f"\n  {PAIR_LABELS[pair_key]} AUC scores:")
        for pheno, auc in res["auc"].items():
            print(f"    {pheno}: AUC = {auc:.3f}")
else:
    augur_results = {}
    for t1, t2 in TISSUE_PAIRS:
        pair_key = f"{t1}_{t2}"
        print(f"\n{'=' * 60}")
        print(f"AUGUR: {t1} vs {t2}")
        print(f"{'=' * 60}")

        sub = adata[adata.obs["tissue"].isin([t1, t2])].copy()
        keep = []
        for pheno in sub.obs["phenotype"].unique():
            counts = sub[sub.obs["phenotype"] == pheno].obs["tissue"].value_counts()
            if counts.get(t1, 0) >= MIN_CELLS and counts.get(t2, 0) >= MIN_CELLS:
                keep.append(pheno)
        sub = sub[sub.obs["phenotype"].isin(keep)].copy()
        print(f"  Phenotypes with >={MIN_CELLS} cells in both: {len(keep)}")

        ag = pt.tl.Augur("random_forest_classifier")
        loaded = ag.load(sub, label_col="tissue", cell_type_col="phenotype")
        loaded, results = ag.predict(
            loaded, subsample_size=20, n_subsamples=50,
            folds=3, random_state=42, select_variance_features=True, span=0.75,
        )
        auc_series = results["summary_metrics"].loc["mean_augur_score"].sort_values(ascending=False)
        augur_results[pair_key] = {"auc": auc_series, "full": results}

        print(f"\n  AUC scores:")
        for pheno, auc in auc_series.items():
            print(f"    {pheno}: AUC = {auc:.3f}")

        # Save after each pair so partial results survive crashes
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(augur_results, f)
        print(f"  Cached to {CACHE_FILE}")

# ============================================================
# 2. Feature importances
# ============================================================

print("\n" + "=" * 60)
print("TOP DISCRIMINATING GENES PER PHENOTYPE x TISSUE PAIR")
print("=" * 60)

feat_importance = {}
for pair_key, res in augur_results.items():
    fi = res["full"]["feature_importances"]
    feat_importance[pair_key] = fi
    print(f"\n--- {PAIR_LABELS[pair_key]} ---")
    if pair_key == list(augur_results.keys())[0]:
        print(f"  Feature importance columns: {fi.columns.tolist()}")
    gene_col = [c for c in fi.columns if c in ["genes", "gene", "feature"]][0]
    imp_col = [c for c in fi.columns if "import" in c.lower()][0]
    ct_col = [c for c in fi.columns if "cell" in c.lower()][0]
    for pheno in fi[ct_col].unique():
        top = fi[fi[ct_col] == pheno].nlargest(10, imp_col)
        genes = ", ".join([f"{r[gene_col]}({r[imp_col]:.3f})" for _, r in top.iterrows()])
        print(f"  {pheno}: {genes}")

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
        keep = []
        for pheno in pat_sub.obs["phenotype"].unique():
            counts = pat_sub[pat_sub.obs["phenotype"] == pheno].obs["tissue"].value_counts()
            if counts.get(t1, 0) >= 10 and counts.get(t2, 0) >= 10:
                keep.append(pheno)
        if not keep:
            continue
        pat_sub = pat_sub[pat_sub.obs["phenotype"].isin(keep)].copy()
        try:
            ag = pt.tl.Augur("random_forest_classifier")
            loaded_pat = ag.load(pat_sub, label_col="tissue", cell_type_col="phenotype")
            _, res = ag.predict(
                loaded_pat, subsample_size=15, n_subsamples=25,
                folds=3, random_state=42, select_variance_features=True, span=0.75,
            )
            for pheno, auc_val in res["summary_metrics"].loc["mean_augur_score"].items():
                patient_records.append({"pair": pair_key, "patient": pat,
                                        "phenotype": pheno, "AUC": auc_val})
        except Exception as e:
            print(f"  {pair_key} {pat}: failed ({e})")

patient_df = pd.DataFrame(patient_records)
if len(patient_df) > 0:
    print(patient_df.groupby(["pair", "phenotype"])["AUC"].agg(["mean", "sem", "count"])
          .round(3).to_string())

# ============================================================
# 4. Augur AUC vs cosine distance
# ============================================================

print("\n" + "=" * 60)
print("AUGUR AUC vs COSINE DISTANCE")
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
        cosine_records.append({"pair": pair_key, "phenotype": pheno,
                               "cosine_dist": cosine(np.asarray(X1).flatten(),
                                                     np.asarray(X2).flatten())})
cosine_df = pd.DataFrame(cosine_records)

for pair_key, res in augur_results.items():
    auc = res["auc"].reset_index()
    auc.columns = ["phenotype", "AUC"]
    merged = auc.merge(cosine_df[cosine_df["pair"] == pair_key], on="phenotype")
    if len(merged) > 2:
        rho, pval = spearmanr(merged["AUC"], merged["cosine_dist"])
        print(f"  {PAIR_LABELS[pair_key]}: rho={rho:.3f}, p={pval:.3f} (n={len(merged)})")

# ============================================================
# 5. Plots
# ============================================================

# --- AUC barplot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
for ax, (t1, t2) in zip(axes, TISSUE_PAIRS):
    pair_key = f"{t1}_{t2}"
    if pair_key not in augur_results:
        continue
    auc = augur_results[pair_key]["auc"].sort_values()
    short = [p.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
             .replace("CD4_", "CD4_") for p in auc.index]
    colors = ["#2166ac" if "CD8" in p else "#b2182b" for p in auc.index]
    ax.barh(short, auc.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("AUC")
    ax.set_title(PAIR_LABELS[pair_key], fontsize=13, fontweight="bold")
    ax.set_xlim(0.4, 1.0)
plt.suptitle("Tissue Separability per Phenotype (Augur AUC)", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("augur_auc_barplot.png", dpi=200, bbox_inches="tight")
plt.show()

# --- Per-patient boxplots ---
if len(patient_df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for ax, (t1, t2) in zip(axes, TISSUE_PAIRS):
        pair_key = f"{t1}_{t2}"
        sub = patient_df[patient_df["pair"] == pair_key]
        if len(sub) == 0:
            continue
        order = sub.groupby("phenotype")["AUC"].mean().sort_values(ascending=False).index
        short = [p.replace("CD8_Activated_", "").replace("CD8_Quiescent_", "")
                 .replace("CD4_", "") for p in order]
        sns.boxplot(data=sub, x="phenotype", y="AUC", order=order, ax=ax, palette="Set2")
        sns.stripplot(data=sub, x="phenotype", y="AUC", order=order, ax=ax,
                      color="black", size=4, alpha=0.6)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(PAIR_LABELS[pair_key], fontsize=12, fontweight="bold")
        ax.set_xlabel("")
    plt.suptitle("Augur AUC per Patient", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("augur_per_patient.png", dpi=200, bbox_inches="tight")
    plt.show()

# --- AUC vs cosine scatter ---
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
    colors = ["#2166ac" if "CD8" in p else "#b2182b" for p in merged["phenotype"]]
    ax.scatter(merged["cosine_dist"], merged["AUC"], c=colors, s=60,
               edgecolors="black", linewidth=0.5)
    for _, row in merged.iterrows():
        short = (row["phenotype"].replace("CD8_Activated_", "").replace("CD8_Quiescent_", "")
                 .replace("CD4_", ""))
        ax.annotate(short, (row["cosine_dist"], row["AUC"]), fontsize=6,
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Augur AUC")
    ax.set_title(PAIR_LABELS[pair_key], fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
plt.suptitle("Augur AUC vs Cosine Distance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("augur_vs_cosine.png", dpi=200, bbox_inches="tight")
plt.show()

# --- Feature importance heatmap ---
for pair_key, fi in feat_importance.items():
    gene_col = [c for c in fi.columns if c in ["genes", "gene", "feature"]][0]
    imp_col = [c for c in fi.columns if "import" in c.lower()][0]
    ct_col = [c for c in fi.columns if "cell" in c.lower()][0]
    top_genes = fi.groupby(gene_col)[imp_col].mean().nlargest(25).index.tolist()
    piv = fi[fi[gene_col].isin(top_genes)].pivot_table(
        index=gene_col, columns=ct_col, values=imp_col, aggfunc="mean")
    piv.columns = [c.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
                   .replace("CD4_", "CD4_") for c in piv.columns]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 1.2), max(5, len(piv) * 0.35)))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(f"Top Discriminating Genes: {PAIR_LABELS[pair_key]}", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"augur_features_{pair_key}.png", dpi=200, bbox_inches="tight")
    plt.show()

print("\nDone.")