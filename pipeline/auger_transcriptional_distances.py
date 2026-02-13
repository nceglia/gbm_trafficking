import scanpy as sc
import pandas as pd
import numpy as np
import pertpy as pt
import matplotlib.pyplot as plt
import seaborn as sns
<<<<<<< HEAD
from scipy import sparse
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
=======
>>>>>>> origin/main
import warnings
warnings.filterwarnings('ignore')

adata = sc.read("GBM_TCR_POS_TCELLS.h5ad")

<<<<<<< HEAD
TISSUE_PAIRS = [("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP")]
PAIR_LABELS = {"PBMC_TP": "PBMC vs Tumor", "PBMC_CSF": "PBMC vs CSF", "CSF_TP": "CSF vs Tumor"}
=======
TISSUES = ["PBMC", "CSF", "TP"]
TISSUE_PAIRS = [("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP")]
PAIR_LABELS = {"PBMC_TP": "PBMC vs Tumor", "PBMC_CSF": "PBMC vs CSF", "CSF_TP": "CSF vs Tumor"}
TISSUE_COLORS = {'CSF': '#cd442a', 'PBMC': '#f0bd00', 'TP': '#7e9437'}
>>>>>>> origin/main
MIN_CELLS = 30

# ============================================================
# 1. Run Augur per tissue pair
# ============================================================

augur_results = {}

for t1, t2 in TISSUE_PAIRS:
    pair_key = f"{t1}_{t2}"
    print(f"\n{'=' * 60}")
    print(f"AUGUR: {t1} vs {t2}")
    print(f"{'=' * 60}")

    sub = adata[adata.obs["tissue"].isin([t1, t2])].copy()
<<<<<<< HEAD
=======

    # Filter phenotypes with enough cells in both tissues
>>>>>>> origin/main
    keep = []
    for pheno in sub.obs["phenotype"].unique():
        counts = sub[sub.obs["phenotype"] == pheno].obs["tissue"].value_counts()
        if counts.get(t1, 0) >= MIN_CELLS and counts.get(t2, 0) >= MIN_CELLS:
            keep.append(pheno)
    sub = sub[sub.obs["phenotype"].isin(keep)].copy()
    print(f"  Phenotypes with >={MIN_CELLS} cells in both: {len(keep)}")
<<<<<<< HEAD

    ag = pt.tl.Augur("random_forest_classifier")
    loaded, results = ag.predict(
        sub, label_col="tissue", cell_type_col="phenotype",
        n_folds=3, subsample_size=20, n_subsamples=50,
        random_state=42, select_variance_features=True, span=0.75,
    )
    auc_df = results["summary_metrics"].sort_values("mean_augur_score", ascending=False)
=======
    print(f"  {keep}")

    ag = pt.tl.Augur("random_forest_classifier")
    loaded, results = ag.predict(
        sub,
        label_col="tissue",
        cell_type_col="phenotype",
        n_folds=3,
        subsample_size=20,
        n_subsamples=50,
        random_state=42,
        select_variance_features=True,
        span=0.75,
    )

    auc_df = results["summary_metrics"]
    auc_df = auc_df.sort_values("mean_augur_score", ascending=False)
>>>>>>> origin/main
    augur_results[pair_key] = {"auc": auc_df, "full": results, "adata": loaded}

    print(f"\n  AUC scores:")
    for _, row in auc_df.iterrows():
        print(f"    {row.name}: AUC = {row['mean_augur_score']:.3f}")

# ============================================================
<<<<<<< HEAD
# 2. Feature importances
# ============================================================

print("\n" + "=" * 60)
print("TOP DISCRIMINATING GENES PER PHENOTYPE x TISSUE PAIR")
=======
# 2. Feature importances per phenotype × tissue pair
# ============================================================

print("\n" + "=" * 60)
print("TOP DISCRIMINATING GENES PER PHENOTYPE × TISSUE PAIR")
>>>>>>> origin/main
print("=" * 60)

feat_importance = {}
for pair_key, res in augur_results.items():
    fi = res["full"]["feature_importances"]
    feat_importance[pair_key] = fi
    print(f"\n--- {PAIR_LABELS[pair_key]} ---")
    for pheno in fi["cell_type"].unique():
        top = fi[fi["cell_type"] == pheno].nlargest(10, "importance")
        genes = ", ".join([f"{r['genes']}({r['importance']:.3f})" for _, r in top.iterrows()])
        print(f"  {pheno}: {genes}")

# ============================================================
<<<<<<< HEAD
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
=======
# 3. Per-patient Augur (leave-one-patient-out)
# ============================================================

print("\n" + "=" * 60)
print("PER-PATIENT AUGUR (leave-one-out)")
print("=" * 60)

patients = sorted(adata.obs["patient"].unique())
patient_aucs = []

for t1, t2 in TISSUE_PAIRS:
    pair_key = f"{t1}_{t2}"
    sub = adata[adata.obs["tissue"].isin([t1, t2])].copy()

    for pat in patients:
>>>>>>> origin/main
        pat_sub = sub[sub.obs["patient"] == pat].copy()
        keep = []
        for pheno in pat_sub.obs["phenotype"].unique():
            counts = pat_sub[pat_sub.obs["phenotype"] == pheno].obs["tissue"].value_counts()
            if counts.get(t1, 0) >= 10 and counts.get(t2, 0) >= 10:
                keep.append(pheno)
<<<<<<< HEAD
        if not keep:
            continue
        pat_sub = pat_sub[pat_sub.obs["phenotype"].isin(keep)].copy()
        try:
            ag = pt.tl.Augur("random_forest_classifier")
            _, res = ag.predict(
=======
        if len(keep) == 0:
            continue
        pat_sub = pat_sub[pat_sub.obs["phenotype"].isin(keep)].copy()

        try:
            ag = pt.tl.Augur("random_forest_classifier")
            _, results = ag.predict(
>>>>>>> origin/main
                pat_sub, label_col="tissue", cell_type_col="phenotype",
                n_folds=3, subsample_size=15, n_subsamples=25,
                random_state=42, select_variance_features=True, span=0.75,
            )
<<<<<<< HEAD
            for _, row in res["summary_metrics"].iterrows():
                patient_records.append({"pair": pair_key, "patient": pat,
                                        "phenotype": row.name, "AUC": row["mean_augur_score"]})
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
=======
            for _, row in results["summary_metrics"].iterrows():
                patient_aucs.append({
                    "pair": pair_key, "patient": pat, "phenotype": row.name,
                    "AUC": row["mean_augur_score"],
                })
        except Exception as e:
            print(f"  {pair_key} {pat}: failed ({e})")

patient_auc_df = pd.DataFrame(patient_aucs)
if len(patient_auc_df) > 0:
    print("\nPer-patient AUC summary (mean ± sem):")
    summary = patient_auc_df.groupby(["pair", "phenotype"])["AUC"].agg(["mean", "sem", "count"])
    print(summary.round(3).to_string())

# ============================================================
# 4. Comparison: Augur AUC vs cosine distance
# ============================================================

from scipy.spatial.distance import cosine

print("\n" + "=" * 60)
print("AUGUR AUC vs COSINE DISTANCE CORRELATION")
>>>>>>> origin/main
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
<<<<<<< HEAD
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
=======
        cosine_records.append({"pair": pair_key, "phenotype": pheno, "cosine_dist": cosine(X1, X2)})

from scipy import sparse
cosine_df = pd.DataFrame(cosine_records)

for pair_key in augur_results:
    auc = augur_results[pair_key]["auc"].reset_index()
    auc.columns = ["phenotype", "AUC"]
    merged = auc.merge(cosine_df[cosine_df["pair"] == pair_key], on="phenotype")
    if len(merged) > 2:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(merged["AUC"], merged["cosine_dist"])
        print(f"  {PAIR_LABELS[pair_key]}: Spearman rho={rho:.3f}, p={pval:.3f} (n={len(merged)})")
>>>>>>> origin/main

# ============================================================
# 5. Plots
# ============================================================

<<<<<<< HEAD
# --- AUC barplot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
=======
# --- Combined AUC barplot ---
fig, axes = plt.subplots(1, len(TISSUE_PAIRS), figsize=(6 * len(TISSUE_PAIRS), 7), sharey=True)
>>>>>>> origin/main
for ax, (t1, t2) in zip(axes, TISSUE_PAIRS):
    pair_key = f"{t1}_{t2}"
    if pair_key not in augur_results:
        continue
<<<<<<< HEAD
    auc = augur_results[pair_key]["auc"]["mean_augur_score"].sort_values()
    short = [p.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
             .replace("CD4_", "CD4_") for p in auc.index]
    colors = ["#2166ac" if "CD8" in p else "#b2182b" for p in auc.index]
    ax.barh(short, auc.values, color=colors, edgecolor="black", linewidth=0.5)
=======
    auc = augur_results[pair_key]["auc"].reset_index()
    auc.columns = ["phenotype", "AUC"]
    auc = auc.sort_values("AUC", ascending=True)
    auc["short"] = (auc["phenotype"]
        .str.replace("CD8_Activated_", "CD8a_")
        .str.replace("CD8_Quiescent_", "CD8q_")
        .str.replace("CD4_", "CD4_"))
    colors = ["#2166ac" if "CD8" in p else "#b2182b" for p in auc["phenotype"]]
    ax.barh(auc["short"], auc["AUC"], color=colors, edgecolor="black", linewidth=0.5)
>>>>>>> origin/main
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("AUC")
    ax.set_title(PAIR_LABELS[pair_key], fontsize=13, fontweight="bold")
    ax.set_xlim(0.4, 1.0)
<<<<<<< HEAD
plt.suptitle("Tissue Separability per Phenotype (Augur AUC)", fontsize=15, fontweight="bold")
=======
plt.suptitle("Augur: Tissue Separability per Phenotype (AUC)", fontsize=15, fontweight="bold")
>>>>>>> origin/main
plt.tight_layout()
plt.savefig("augur_auc_barplot.png", dpi=200, bbox_inches="tight")
plt.show()

<<<<<<< HEAD
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
=======
# --- Per-patient AUC boxplots ---
if len(patient_auc_df) > 0:
    fig, axes = plt.subplots(1, len(TISSUE_PAIRS), figsize=(6 * len(TISSUE_PAIRS), 7), sharey=True)
    for ax, (t1, t2) in zip(axes, TISSUE_PAIRS):
        pair_key = f"{t1}_{t2}"
        sub = patient_auc_df[patient_auc_df["pair"] == pair_key]
        if len(sub) == 0:
            continue
        order = sub.groupby("phenotype")["AUC"].mean().sort_values(ascending=False).index
        short = [p.replace("CD8_Activated_", "").replace("CD8_Quiescent_", "").replace("CD4_", "")
                 for p in order]
>>>>>>> origin/main
        sns.boxplot(data=sub, x="phenotype", y="AUC", order=order, ax=ax, palette="Set2")
        sns.stripplot(data=sub, x="phenotype", y="AUC", order=order, ax=ax,
                      color="black", size=4, alpha=0.6)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(PAIR_LABELS[pair_key], fontsize=12, fontweight="bold")
        ax.set_xlabel("")
<<<<<<< HEAD
    plt.suptitle("Augur AUC per Patient", fontsize=14, fontweight="bold")
=======
    plt.suptitle("Augur AUC per Patient (leave-one-out)", fontsize=14, fontweight="bold")
>>>>>>> origin/main
    plt.tight_layout()
    plt.savefig("augur_per_patient.png", dpi=200, bbox_inches="tight")
    plt.show()

<<<<<<< HEAD
# --- AUC vs cosine scatter ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
=======
# --- Augur AUC vs cosine distance scatterplot ---
fig, axes = plt.subplots(1, len(TISSUE_PAIRS), figsize=(5 * len(TISSUE_PAIRS), 5))
>>>>>>> origin/main
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
<<<<<<< HEAD
    ax.scatter(merged["cosine_dist"], merged["AUC"], c=colors, s=60,
               edgecolors="black", linewidth=0.5)
    for _, row in merged.iterrows():
        short = (row["phenotype"].replace("CD8_Activated_", "").replace("CD8_Quiescent_", "")
                 .replace("CD4_", ""))
        ax.annotate(short, (row["cosine_dist"], row["AUC"]), fontsize=6,
=======
    ax.scatter(merged["cosine_dist"], merged["AUC"], c=colors, s=60, edgecolors="black", linewidth=0.5)
    for _, row in merged.iterrows():
        short = (row["phenotype"].replace("CD8_Activated_", "").replace("CD8_Quiescent_", "")
                 .replace("CD4_", ""))
        ax.annotate(short, (row["cosine_dist"], row["AUC"]), fontsize=6, ha="left",
>>>>>>> origin/main
                    xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Cosine distance")
    ax.set_ylabel("Augur AUC")
    ax.set_title(PAIR_LABELS[pair_key], fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
plt.suptitle("Augur AUC vs Cosine Distance", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("augur_vs_cosine.png", dpi=200, bbox_inches="tight")
plt.show()

<<<<<<< HEAD
# --- Feature importance heatmap ---
=======
# --- Top features heatmap ---
>>>>>>> origin/main
for pair_key, fi in feat_importance.items():
    top_genes = fi.groupby("genes")["importance"].mean().nlargest(25).index.tolist()
    piv = fi[fi["genes"].isin(top_genes)].pivot_table(
        index="genes", columns="cell_type", values="importance", aggfunc="mean")
    piv.columns = [c.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
                   .replace("CD4_", "CD4_") for c in piv.columns]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

<<<<<<< HEAD
    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 1.2), max(5, len(piv) * 0.35)))
=======
    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 1.2), max(5, len(piv) * 0.3)))
>>>>>>> origin/main
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(f"Top Discriminating Genes: {PAIR_LABELS[pair_key]}", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"augur_features_{pair_key}.png", dpi=200, bbox_inches="tight")
    plt.show()

print("\nDone.")