<<<<<<< HEAD
# %%
=======
>>>>>>> origin/main
import pickle
import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

TISSUE_PAIRS = [("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP")]
PAIR_LABELS = {"PBMC_TP": "PBMC vs Tumor", "PBMC_CSF": "PBMC vs CSF", "CSF_TP": "CSF vs Tumor"}
PAIR_KEYS = ["PBMC_TP", "PBMC_CSF", "CSF_TP"]
<<<<<<< HEAD
# %%
=======

>>>>>>> origin/main
# ============================================================
# Load cached Augur results
# ============================================================

with open("augur_results_cache.pkl", "rb") as f:
    augur_results = pickle.load(f)

# Extract AUC per pair
augur_auc = {}
for pk in PAIR_KEYS:
    auc = augur_results[pk]["auc"]
    if isinstance(auc, pd.DataFrame):
        augur_auc[pk] = auc["mean_augur_score"] if "mean_augur_score" in auc.columns else auc.iloc[0]
    else:
        augur_auc[pk] = auc

# Build AUC matrix: phenotype × pair
all_phenos = set()
for pk in PAIR_KEYS:
    all_phenos |= set(augur_auc[pk].index)
all_phenos = sorted(all_phenos)

auc_mat = pd.DataFrame(index=all_phenos, columns=PAIR_KEYS, dtype=float)
for pk in PAIR_KEYS:
    for pheno in augur_auc[pk].index:
        auc_mat.loc[pheno, pk] = augur_auc[pk][pheno]
<<<<<<< HEAD
# %%
=======

>>>>>>> origin/main
# ============================================================
# Load GSEA results (recompute Wilcoxon-based per-phenotype)
# ============================================================

import scanpy as sc
<<<<<<< HEAD
adata = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")
=======
adata = sc.read("GBM_TCR_POS_TCELLS.h5ad")
>>>>>>> origin/main

TISSUES = ["PBMC", "CSF", "TP"]
GENE_SETS = ['MSigDB_Hallmark_2020']
MIN_CELLS = 30

def run_wilcoxon_gsea(adata_sub, groupby="tissue", tissues=TISSUES):
    sc.tl.rank_genes_groups(adata_sub, groupby, method="wilcoxon", use_raw=False)
    results = {}
    for t in tissues:
        if t not in adata_sub.obs[groupby].unique():
            continue
        df = sc.get.rank_genes_groups_df(adata_sub, group=t)
        ranking = df.set_index("names")["scores"].dropna()
        ranking = ranking[~ranking.index.duplicated()]
        pre = gp.prerank(rnk=ranking, gene_sets=GENE_SETS, outdir=None,
                         seed=42, min_size=10, max_size=500,
                         permutation_num=1000, no_plot=True, verbose=False)
        results[t] = pre.res2d.copy()
    return results

print("Computing per-phenotype GSEA...")
phenotypes = sorted(adata.obs["phenotype"].unique())
pheno_gsea = {}
for pheno in phenotypes:
    sub = adata[adata.obs["phenotype"] == pheno].copy()
    tissue_counts = sub.obs["tissue"].value_counts()
    valid = tissue_counts[tissue_counts >= MIN_CELLS].index.tolist()
    if len(valid) < 2:
        continue
    sub = sub[sub.obs["tissue"].isin(valid)]
    print(f"  {pheno}: {len(sub)} cells")
    pheno_gsea[pheno] = run_wilcoxon_gsea(sub, tissues=valid)
<<<<<<< HEAD
# %%
=======

>>>>>>> origin/main
# Build delta NES: for each tissue pair, compute |NES_t1 - NES_t2| averaged across pathways
delta_nes_records = []
for pheno, tissue_res in pheno_gsea.items():
    for (t1, t2), pk in zip(TISSUE_PAIRS, PAIR_KEYS):
        if t1 in tissue_res and t2 in tissue_res:
            nes1 = tissue_res[t1].set_index("Term")["NES"]
            nes2 = tissue_res[t2].set_index("Term")["NES"]
            common = nes1.index.intersection(nes2.index)
            if len(common) > 0:
                abs_delta = (nes1[common] - nes2[common]).abs().mean()
                max_delta = (nes1[common] - nes2[common]).abs().max()
                n_sig = ((tissue_res[t1].set_index("Term").loc[common, "FDR q-val"] < 0.25) |
                         (tissue_res[t2].set_index("Term").loc[common, "FDR q-val"] < 0.25)).sum()
                delta_nes_records.append({
                    "phenotype": pheno, "pair": pk,
                    "mean_abs_delta_NES": abs_delta,
                    "max_abs_delta_NES": max_delta,
                    "n_sig_pathways": n_sig,
                })

delta_df = pd.DataFrame(delta_nes_records)
delta_mat = delta_df.pivot_table(index="phenotype", columns="pair", values="mean_abs_delta_NES")
<<<<<<< HEAD
# %%
=======

>>>>>>> origin/main
# ============================================================
# FIGURE 1: Combined Augur AUC + GSEA delta NES
# ============================================================

print("\nGenerating combined figure...")

# Merge into one comparison table
compare = auc_mat.copy()
compare.columns = [f"AUC_{pk}" for pk in PAIR_KEYS]
for pk in PAIR_KEYS:
    if pk in delta_mat.columns:
        compare[f"dNES_{pk}"] = delta_mat[pk]

compare = compare.dropna(how="all")

# Order phenotypes: CD8 first, then CD4
cd8 = sorted([p for p in compare.index if "CD8" in p])
cd4 = sorted([p for p in compare.index if "CD4" in p])
order = cd8 + cd4
compare = compare.loc[[p for p in order if p in compare.index]]

fig = plt.figure(figsize=(22, 10))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.2, 1.5], wspace=0.35)

# Panel A: Augur AUC heatmap
ax1 = fig.add_subplot(gs[0])
auc_plot = compare[[f"AUC_{pk}" for pk in PAIR_KEYS]].copy()
auc_plot.columns = [PAIR_LABELS[pk] for pk in PAIR_KEYS]
short_idx = [p.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
             .replace("CD4_", "CD4_") for p in auc_plot.index]
auc_plot.index = short_idx

im1 = ax1.imshow(auc_plot.values, aspect="auto", cmap="YlOrRd", vmin=0.5, vmax=0.9)
for i in range(auc_plot.shape[0]):
    for j in range(auc_plot.shape[1]):
        v = auc_plot.iloc[i, j]
        if pd.notna(v):
            color = "white" if v > 0.8 else "black"
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
ax1.set_xticks(range(len(auc_plot.columns)))
ax1.set_xticklabels(auc_plot.columns, rotation=30, ha="right", fontsize=9)
ax1.set_yticks(range(len(auc_plot.index)))
ax1.set_yticklabels(auc_plot.index, fontsize=9)
cd8_count = sum(1 for p in compare.index if "CD8" in p)
if cd8_count < len(compare):
    ax1.axhline(cd8_count - 0.5, color="black", linewidth=2, linestyle="--")
plt.colorbar(im1, ax=ax1, label="Augur AUC", shrink=0.6, pad=0.08)
ax1.set_title("A. Tissue Separability\n(Augur AUC)", fontsize=12, fontweight="bold")

# Panel B: GSEA delta NES heatmap
ax2 = fig.add_subplot(gs[1])
dnes_plot = compare[[f"dNES_{pk}" for pk in PAIR_KEYS if f"dNES_{pk}" in compare.columns]].copy()
dnes_plot.columns = [PAIR_LABELS[pk] for pk in PAIR_KEYS if f"dNES_{pk}" in compare.columns]
dnes_plot.index = short_idx[:len(dnes_plot)]

im2 = ax2.imshow(dnes_plot.values, aspect="auto", cmap="PuRd", vmin=0, vmax=2.5)
for i in range(dnes_plot.shape[0]):
    for j in range(dnes_plot.shape[1]):
        v = dnes_plot.iloc[i, j]
        if pd.notna(v):
            color = "white" if v > 1.8 else "black"
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
ax2.set_xticks(range(len(dnes_plot.columns)))
ax2.set_xticklabels(dnes_plot.columns, rotation=30, ha="right", fontsize=9)
ax2.set_yticks(range(len(dnes_plot.index)))
ax2.set_yticklabels(dnes_plot.index, fontsize=9)
if cd8_count < len(compare):
    ax2.axhline(cd8_count - 0.5, color="black", linewidth=2, linestyle="--")
plt.colorbar(im2, ax=ax2, label="Mean |ΔNES|", shrink=0.6, pad=0.08)
ax2.set_title("B. Pathway Divergence\n(Mean |ΔNES| across Hallmark)", fontsize=12, fontweight="bold")

# Panel C: AUC vs delta NES scatter
ax3 = fig.add_subplot(gs[2])
from scipy.stats import spearmanr
colors_scatter = {"PBMC_TP": "#E67E22", "PBMC_CSF": "#3498DB", "CSF_TP": "#8E44AD"}
markers_scatter = {"PBMC_TP": "o", "PBMC_CSF": "s", "CSF_TP": "^"}

for pk in PAIR_KEYS:
    auc_col = f"AUC_{pk}"
    dnes_col = f"dNES_{pk}"
    if dnes_col not in compare.columns:
        continue
    sub = compare[[auc_col, dnes_col]].dropna()
    ax3.scatter(sub[dnes_col], sub[auc_col], c=colors_scatter[pk], s=60,
                marker=markers_scatter[pk], edgecolors="black", linewidth=0.5,
                label=PAIR_LABELS[pk], alpha=0.8)
    for pheno in sub.index:
        short = (pheno.replace("CD8_Activated_", "").replace("CD8_Quiescent_", "")
                 .replace("CD4_", ""))
        ax3.annotate(short, (sub.loc[pheno, dnes_col], sub.loc[pheno, auc_col]),
                     fontsize=5.5, ha="left", xytext=(3, 2), textcoords="offset points")

# Overall correlation
all_auc = []
all_dnes = []
for pk in PAIR_KEYS:
    auc_col, dnes_col = f"AUC_{pk}", f"dNES_{pk}"
    if dnes_col in compare.columns:
        sub = compare[[auc_col, dnes_col]].dropna()
        all_auc.extend(sub[auc_col].tolist())
        all_dnes.extend(sub[dnes_col].tolist())

if len(all_auc) > 3:
    rho, pval = spearmanr(all_dnes, all_auc)
    ax3.text(0.05, 0.95, f"ρ={rho:.2f}, p={pval:.3f}\n(all pairs pooled)",
             transform=ax3.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax3.set_xlabel("Mean |ΔNES| (pathway divergence)", fontsize=10)
ax3.set_ylabel("Augur AUC (tissue separability)", fontsize=10)
ax3.legend(fontsize=8, loc="lower right")
ax3.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
ax3.set_title("C. Separability vs Pathway Divergence", fontsize=12, fontweight="bold")
ax3.grid(alpha=0.15)

fig.suptitle("Tissue Compartment Analysis: Augur Separability + GSEA Pathway Divergence",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("augur_gsea_combined.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved: augur_gsea_combined.png")
<<<<<<< HEAD
# %%
=======

>>>>>>> origin/main
# ============================================================
# AUGUR GENE PATHWAY ENRICHMENT
# ============================================================

print("\n" + "=" * 60)
print("AUGUR GENE PATHWAY ENRICHMENT (top features → enrichr)")
print("=" * 60)

ENRICH_SETS = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2023']
TOP_N = 50  # top genes per phenotype per pair

augur_pathway_results = {}

for pk in PAIR_KEYS:
    fi = augur_results[pk]["full"]["feature_importances"]
    # Dynamic column detection
    gene_col = [c for c in fi.columns if c in ["genes", "gene", "feature"]][0]
    imp_col = [c for c in fi.columns if "import" in c.lower()][0]
    ct_col = [c for c in fi.columns if "cell" in c.lower()][0]

    print(f"\n--- {PAIR_LABELS[pk]} ---")
    pair_results = {}

    for pheno in fi[ct_col].unique():
        pheno_fi = fi[fi[ct_col] == pheno]
        # Aggregate importance across folds/subsamples
        gene_imp = pheno_fi.groupby(gene_col)[imp_col].mean().sort_values(ascending=False)
        top_genes = gene_imp.head(TOP_N).index.tolist()

        if len(top_genes) < 5:
            continue

        try:
            enr = gp.enrichr(gene_list=top_genes, gene_sets=ENRICH_SETS,
                             organism='Human', outdir=None, no_plot=True, verbose=False)
            sig = enr.results[enr.results["Adjusted P-value"] < 0.1].copy()
            if len(sig) > 0:
                sig = sig.sort_values("Adjusted P-value")
                pair_results[pheno] = sig
                short = pheno.replace("CD8_Activated_", "").replace("CD8_Quiescent_", "").replace("CD4_", "")
                top3 = sig.head(3)
                terms = [f"{r['Term'].split('__')[-1][:50]} (p={r['Adjusted P-value']:.3f})"
                         for _, r in top3.iterrows()]
                print(f"  {short}: {' | '.join(terms)}")
                if len(sig) > 3:
                    print(f"    + {len(sig)-3} more significant terms")
            else:
                print(f"  {pheno}: no significant enrichments")
        except Exception as e:
            print(f"  {pheno}: enrichr failed ({e})")

    augur_pathway_results[pk] = pair_results
<<<<<<< HEAD
# %%
=======

>>>>>>> origin/main
# ============================================================
# FIGURE 2: Augur gene pathway heatmap
# ============================================================

<<<<<<< HEAD
=======
print("\nGenerating Augur pathway heatmaps...")

>>>>>>> origin/main
for pk in PAIR_KEYS:
    pair_res = augur_pathway_results.get(pk, {})
    if not pair_res:
        continue

<<<<<<< HEAD
    records = []
    for pheno, sig in pair_res.items():
        for _, row in sig.head(15).iterrows():
            records.append({
                "phenotype": pheno,
                "term": row["Term"].split("__")[-1][:50],
                "neg_log_p": -np.log10(row["Adjusted P-value"] + 1e-10),
            })

    if not records:
        continue

    hdf = pd.DataFrame(records)
    piv = hdf.pivot_table(index="term", columns="phenotype", values="neg_log_p", aggfunc="max")
    piv.columns = [c.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
                   .replace("CD4_", "CD4_") for c in piv.columns]

    # Keep terms enriched in >=2 phenotypes
    keep = piv.index[piv.notna().sum(axis=1) >= 2]
    if len(keep) < 3:
        keep = piv.index[piv.max(axis=1).nlargest(15).index]
    piv = piv.loc[keep].fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 1.3), max(6, len(piv) * 0.35)))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax,
                cbar_kws={"label": "-log10(adj p)"}, vmin=0)
    ax.set_title(f"Augur Top Gene Enrichment: {PAIR_LABELS[pk]}\n"
                 f"(top {TOP_N} discriminating genes per phenotype)",
=======
    # Collect all significant Hallmark terms
    hallmark_records = []
    for pheno, sig in pair_res.items():
        hallmark = sig[sig["Term"].str.contains("Hallmark", case=False)]
        for _, row in hallmark.iterrows():
            hallmark_records.append({
                "phenotype": pheno,
                "term": row["Term"].replace("MSigDB_Hallmark_2020__", ""),
                "neg_log_p": -np.log10(row["Adjusted P-value"] + 1e-10),
                "odds_ratio": row.get("Odds Ratio", row.get("odds_ratio", np.nan)),
            })

    if not hallmark_records:
        continue

    hdf = pd.DataFrame(hallmark_records)
    piv = hdf.pivot_table(index="term", columns="phenotype", values="neg_log_p")
    piv.columns = [c.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
                   .replace("CD4_", "CD4_") for c in piv.columns]

    # Keep terms enriched in >=2 phenotypes or very significant
    keep = piv.index[(piv.notna().sum(axis=1) >= 2) | (piv.max(axis=1) > 3)]
    if len(keep) == 0:
        keep = piv.index[piv.max(axis=1) > 1]
    piv = piv.loc[keep].fillna(0)

    if len(piv) == 0:
        continue

    fig, ax = plt.subplots(figsize=(max(8, len(piv.columns) * 1.3), max(5, len(piv) * 0.35)))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5, ax=ax,
                cbar_kws={"label": "-log10(adj p)"}, vmin=0)
    ax.set_title(f"Augur Top Gene Enrichment: {PAIR_LABELS[pk]}\n"
                 f"(Hallmark pathways from top {TOP_N} discriminating genes per phenotype)",
>>>>>>> origin/main
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"augur_gene_pathways_{pk}.png", dpi=200, bbox_inches="tight")
    plt.show()
<<<<<<< HEAD
# %%
=======
    print(f"Saved: augur_gene_pathways_{pk}.png")

>>>>>>> origin/main
# ============================================================
# Summary: top Augur-derived pathways per phenotype × pair
# ============================================================

print("\n" + "=" * 60)
print("FULL SUMMARY: AUGUR GENE PATHWAY ENRICHMENT")
print("=" * 60)

for pk in PAIR_KEYS:
    pair_res = augur_pathway_results.get(pk, {})
    if not pair_res:
        continue
    print(f"\n{'=' * 40}")
    print(f"{PAIR_LABELS[pk]}")
    print(f"{'=' * 40}")
    for pheno in sorted(pair_res.keys()):
        sig = pair_res[pheno]
        short = pheno.replace("CD8_Activated_", "").replace("CD8_Quiescent_", "").replace("CD4_", "")
        print(f"\n  {short} ({len(sig)} significant terms):")
        for _, row in sig.head(8).iterrows():
            term = row["Term"].split("__")[-1][:55]
            print(f"    {term}: p={row['Adjusted P-value']:.4f}, "
                  f"genes={row.get('Genes', row.get('genes', 'N/A'))[:60]}")

<<<<<<< HEAD
print("\nDone.")
=======
print("\nDone.")
>>>>>>> origin/main
