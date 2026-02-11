# %%            
import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from modules.constants import TISSUES
from modules.differential_expression import pseudobulk_deseq2

warnings.filterwarnings('ignore')

adata = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")

GENE_SETS = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 'GO_Biological_Process_2023']
TOP_N = 150

# %%        
# ============================================================
# 1. Single-cell Wilcoxon DEG (quick, for enrichr)
# ============================================================

sc.tl.rank_genes_groups(adata, "tissue", method="wilcoxon", use_raw=False)

tissue_genes = {}
tissue_ranked = {}
for t in TISSUES:
    df = sc.get.rank_genes_groups_df(adata, group=t)
    tissue_genes[t] = df["names"].tolist()[:TOP_N]
    tissue_ranked[t] = df.set_index("names")["scores"]

# ============================================================
# 2. Pseudobulk DEG (patients as replicates, pyDESeq2)
#    — proper statistical framework
# ============================================================
print("=" * 70)
print("PSEUDOBULK DESeq2 (patients as replicates)")
print("=" * 70)
deseq_results = pseudobulk_deseq2(adata)
for contrast, res in deseq_results.items():
    tissue = contrast.split("_vs_")[0]
    print(f"\n{contrast}: {(res['padj'] < 0.05).sum()} DEGs (padj<0.05)")
    print(f"  Up in {tissue}: {((res['padj'] < 0.05) & (res['log2FoldChange'] > 0)).sum()}")
    print(f"  Up in PBMC: {((res['padj'] < 0.05) & (res['log2FoldChange'] < 0)).sum()}")
    print(res.head(10)[['log2FoldChange', 'pvalue', 'padj']].to_string())
# %%
# ============================================================
# 3. Enrichr (overrepresentation, top N genes)
# ============================================================

print("\n" + "=" * 70)
print("ENRICHR: TOP GENES PER TISSUE (single-cell Wilcoxon)")
print("=" * 70)

enrichr_results = {}
for t in TISSUES:
    enr = gp.enrichr(gene_list=tissue_genes[t], gene_sets=GENE_SETS,
                      organism='human', outdir=None)
    enrichr_results[t] = enr.results[enr.results["Adjusted P-value"] < 0.05].head(20)
    print(f"\n--- {t} ---")
    for _, row in enrichr_results[t].iterrows():
        print(f"  {row['Term']}: p={row['Adjusted P-value']:.2e}, genes={row['Overlap']}")
# %%
# ============================================================
# 4. GSEA preranked (full ranked list, better than enrichr)
# ============================================================

print("\n" + "=" * 70)
print("GSEA PRERANKED: PSEUDOBULK DESeq2 RANKINGS")
print("=" * 70)

gsea_results = {}
for contrast, res in deseq_results.items():
    ranking = res["stat"].dropna().sort_values(ascending=False)
    ranking = ranking[~ranking.index.duplicated()]
    
    pre = gp.prerank(rnk=ranking, gene_sets=GENE_SETS,
                     outdir=None, seed=42, min_size=10, max_size=500,
                     permutation_num=1000, no_plot=True)
    sig = pre.res2d[pre.res2d["FDR q-val"] < 0.25].sort_values("NES", ascending=False)
    gsea_results[contrast] = sig
    
    print(f"\n--- {contrast} ---")
    print(f"  Significant terms (FDR<0.25): {len(sig)}")
    if len(sig) > 0:
        print("  Positive (enriched in first tissue):")
        pos = sig[sig["NES"] > 0].head(10)
        for _, row in pos.iterrows():
            print(f"    {row['Term']}: NES={row['NES']:.2f}, FDR={row['FDR q-val']:.3f}")
        print("  Negative (enriched in reference):")
        neg = sig[sig["NES"] < 0].tail(10)
        for _, row in neg.iterrows():
            print(f"    {row['Term']}: NES={row['NES']:.2f}, FDR={row['FDR q-val']:.3f}")
# %%
# Also run preranked on Wilcoxon scores (for the 3-way tissue comparison)
print("\n" + "=" * 70)
print("GSEA PRERANKED: WILCOXON SCORES (per tissue vs rest)")
print("=" * 70)

gsea_wilcox = {}
for t in TISSUES:
    ranking = tissue_ranked[t].dropna().sort_values(ascending=False)
    ranking = ranking[~ranking.index.duplicated()]
    
    pre = gp.prerank(rnk=ranking, gene_sets=GENE_SETS,
                     outdir=None, seed=42, min_size=10, max_size=500,
                     permutation_num=1000, no_plot=True)
    sig = pre.res2d[pre.res2d["FDR q-val"] < 0.25].sort_values("NES", ascending=False)
    gsea_wilcox[t] = sig
    
    print(f"\n--- {t} (vs rest) ---")
    top_pos = sig[sig["NES"] > 0].head(10)
    for _, row in top_pos.iterrows():
        print(f"  {row['Term']}: NES={row['NES']:.2f}, FDR={row['FDR q-val']:.3f}")
# %%
# ============================================================
# 5. Plots
# ============================================================

# --- Enrichr dotplot per tissue ---
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=False)
for ax, t in zip(axes, TISSUES):
    res = enrichr_results[t]
    if len(res) == 0:
        ax.set_title(t); continue
    res = res.head(15).copy()
    res["-log10(p)"] = -np.log10(res["Adjusted P-value"].clip(1e-30))
    res["Term_short"] = res["Term"].str[:50]
    res = res.sort_values("-log10(p)")
    ax.barh(res["Term_short"], res["-log10(p)"], color={"PBMC": "#f0bd00", "CSF": "#cd442a", "TP": "#7e9437"}[t])
    ax.set_xlabel("-log10(adj p)")
    ax.set_title(f"{t}", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=7)
plt.suptitle("Enrichr: Top Pathways per Tissue (top 150 Wilcoxon DEGs)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("enrichr_tissue_pathways.png", dpi=200, bbox_inches="tight")
plt.show()

# %%
# --- GSEA NES heatmap (pseudobulk DESeq2) ---
all_terms = set()
for contrast, sig in gsea_results.items():
    all_terms |= set(sig.head(15)["Term"]) | set(sig.tail(10)["Term"])

if all_terms:
    nes_mat = pd.DataFrame(index=sorted(all_terms), columns=list(gsea_results.keys()))
    fdr_mat = pd.DataFrame(index=sorted(all_terms), columns=list(gsea_results.keys()))
    for contrast, sig in gsea_results.items():
        full = gsea_results[contrast] if contrast in gsea_results else pd.DataFrame()
        # Get from full prerank results
        for term in all_terms:
            match = sig[sig["Term"] == term]
            if len(match) > 0:
                nes_mat.loc[term, contrast] = match.iloc[0]["NES"]
                fdr_mat.loc[term, contrast] = match.iloc[0]["FDR q-val"]
    
    nes_mat = nes_mat.dropna(how="all").astype(float)
    if len(nes_mat) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(nes_mat) * 0.3)))
        sns.heatmap(nes_mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "NES"})
        ax.set_title("GSEA Preranked NES (pseudobulk DESeq2)", fontsize=12, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        plt.tight_layout()
        plt.savefig("gsea_nes_heatmap.png", dpi=200, bbox_inches="tight")
        plt.show()
# %%
# --- GSEA NES heatmap (Wilcoxon, tissue vs rest) ---
all_terms_w = set()
for t, sig in gsea_wilcox.items():
    all_terms_w |= set(sig[sig["NES"] > 0].head(10)["Term"])

if all_terms_w:
    nes_w = pd.DataFrame(index=sorted(all_terms_w), columns=TISSUES)
    for t, sig in gsea_wilcox.items():
        for term in all_terms_w:
            match = sig[sig["Term"] == term]
            if len(match) > 0:
                nes_w.loc[term, t] = match.iloc[0]["NES"]
    
    nes_w = nes_w.dropna(how="all").astype(float)
    if len(nes_w) > 0:
        fig, ax = plt.subplots(figsize=(8, max(6, len(nes_w) * 0.3)))
        sns.heatmap(nes_w, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "NES"})
        ax.set_title("GSEA Preranked NES per Tissue (Wilcoxon vs rest)", fontsize=12, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        plt.tight_layout()
        plt.savefig("gsea_nes_tissue_heatmap.png", dpi=200, bbox_inches="tight")
        plt.show()

print("\nDone.")
# %%
