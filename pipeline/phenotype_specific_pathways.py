# %%
import scanpy as sc
import pandas as pd
import numpy as np
import gseapy as gp
from scipy import sparse
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

adata = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")

TISSUES = ["PBMC", "CSF", "TP"]
GENE_SETS = ['MSigDB_Hallmark_2020', 'KEGG_2021_Human']
MIN_CELLS = 30
FLAG_TERMS = ["Hypoxia", "TGF", "Interferon", "IFN"]

# %%
# ============================================================
# 1. Global tissue GSEA (reference baseline)
# ============================================================

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

print("Computing global tissue GSEA...")
global_gsea = run_wilcoxon_gsea(adata)

global_nes = {}
for t, res in global_gsea.items():
    for _, row in res.iterrows():
        global_nes[(t, row["Term"])] = row["NES"]
# %%
# ============================================================
# 2. Per-phenotype tissue GSEA
# ============================================================

phenotypes = sorted(adata.obs["phenotype"].unique())
pheno_gsea = {}

for pheno in phenotypes:
    sub = adata[adata.obs["phenotype"] == pheno].copy()
    tissue_counts = sub.obs["tissue"].value_counts()
    valid = tissue_counts[tissue_counts >= MIN_CELLS].index.tolist()
    if len(valid) < 2:
        print(f"  {pheno}: skipped (<2 tissues with {MIN_CELLS}+ cells)")
        continue
    sub = sub[sub.obs["tissue"].isin(valid)]
    print(f"  {pheno}: {len(sub)} cells, tissues={valid}")
    pheno_gsea[pheno] = run_wilcoxon_gsea(sub, tissues=valid)
# %%
# ============================================================
# 3. Build NES matrix: phenotype × tissue × term
# ============================================================

all_terms = set()
for pheno, tissue_res in pheno_gsea.items():
    for t, res in tissue_res.items():
        sig = res[res["FDR q-val"] < 0.25]
        all_terms |= set(sig["Term"])

records = []
for pheno, tissue_res in pheno_gsea.items():
    for t, res in tissue_res.items():
        for _, row in res.iterrows():
            if row["Term"] in all_terms:
                g_nes = global_nes.get((t, row["Term"]), np.nan)
                records.append({
                    "phenotype": pheno, "tissue": t, "term": row["Term"],
                    "NES": row["NES"], "FDR": row["FDR q-val"],
                    "global_NES": g_nes,
                    "delta_NES": row["NES"] - g_nes if not np.isnan(g_nes) else np.nan,
                })

df = pd.DataFrame(records)
# %%
# ============================================================
# 4. Flag CSF hits for hypoxia/TGF-beta/interferon
# ============================================================

print("\n" + "=" * 70)
print("FLAGGED: HYPOXIA / TGF-BETA / INTERFERON IN CSF")
print("=" * 70)

csf = df[(df["tissue"] == "CSF") & (df["FDR"] < 0.25)]
flagged = csf[csf["term"].str.contains("|".join(FLAG_TERMS), case=False)]
if len(flagged) > 0:
    for _, row in flagged.sort_values("NES", ascending=False).iterrows():
        direction = "UP" if row["NES"] > 0 else "DOWN"
        delta = f"delta={row['delta_NES']:+.2f}" if not np.isnan(row["delta_NES"]) else ""
        print(f"  [{direction}] {row['phenotype']}: {row['term']} "
              f"NES={row['NES']:.2f}, FDR={row['FDR']:.3f} {delta}")
else:
    print("  No significant hits for flagged terms in CSF")

# Also check sub-threshold (FDR < 0.5) for these terms
csf_sub = df[(df["tissue"] == "CSF") & (df["FDR"] < 0.5)]
flagged_sub = csf_sub[csf_sub["term"].str.contains("|".join(FLAG_TERMS), case=False)]
if len(flagged_sub) > len(flagged):
    print("\n  Sub-threshold (FDR<0.5):")
    extra = flagged_sub[~flagged_sub.index.isin(flagged.index)]
    for _, row in extra.sort_values("NES", ascending=False).iterrows():
        direction = "UP" if row["NES"] > 0 else "DOWN"
        print(f"  [{direction}] {row['phenotype']}: {row['term']} "
              f"NES={row['NES']:.2f}, FDR={row['FDR']:.3f}")
# %%
# ============================================================
# 5. Phenotype-specific deviations from global tissue signature
# ============================================================

print("\n" + "=" * 70)
print("LARGEST DEVIATIONS FROM GLOBAL TISSUE SIGNATURE (|delta_NES| > 0.5)")
print("=" * 70)

deviations = df[df["delta_NES"].notna() & (df["delta_NES"].abs() > 0.5) & (df["FDR"] < 0.25)]
deviations = deviations.sort_values("delta_NES", key=abs, ascending=False)
for _, row in deviations.head(40).iterrows():
    print(f"  {row['phenotype']} | {row['tissue']}: {row['term']}")
    print(f"    phenotype NES={row['NES']:.2f}, global NES={row['global_NES']:.2f}, "
          f"delta={row['delta_NES']:+.2f}, FDR={row['FDR']:.3f}")
# %%
# ============================================================
# 6. Summary table: top enriched per phenotype × tissue
# ============================================================

print("\n" + "=" * 70)
print("TOP 3 ENRICHED PATHWAYS PER PHENOTYPE × TISSUE")
print("=" * 70)

for pheno in phenotypes:
    if pheno not in pheno_gsea:
        continue
    print(f"\n--- {pheno} ---")
    for t in TISSUES:
        if t not in pheno_gsea[pheno]:
            continue
        res = pheno_gsea[pheno][t]
        sig = res[res["FDR q-val"] < 0.25].sort_values("NES", ascending=False)
        top = sig.head(3)
        if len(top) > 0:
            terms = [f"{r['Term'].split('__')[-1][:45]} (NES={r['NES']:.2f})"
                     for _, r in top.iterrows()]
            print(f"  {t}: " + " | ".join(terms))
# %%
# ============================================================
# 7. Plots
# ============================================================

# --- Heatmap: NES per phenotype × tissue for Hallmark terms ---
hallmark = df[df["term"].str.contains("Hallmark")].copy()
hallmark["term_short"] = hallmark["term"].str.replace("MSigDB_Hallmark_2020__", "")

for tissue in TISSUES:
    t_data = hallmark[hallmark["tissue"] == tissue]
    if len(t_data) == 0:
        continue
    piv = t_data.pivot_table(index="term_short", columns="phenotype", values="NES")
    if len(piv) == 0:
        continue
    piv.columns = [c.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
                   .replace("CD4_", "CD4_") for c in piv.columns]
    # Only keep terms with at least one significant hit
    sig_mask = hallmark[(hallmark["tissue"] == tissue) & (hallmark["FDR"] < 0.25)]
    keep_terms = sig_mask["term_short"].unique()
    piv = piv.loc[piv.index.isin(keep_terms)]
    if len(piv) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(max(10, len(piv.columns) * 1.2), max(6, len(piv) * 0.4)))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "NES"}, vmin=-3, vmax=3)
    ax.set_title(f"Hallmark Pathway NES in {tissue} (per phenotype, vs rest)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"gsea_hallmark_{tissue}_per_phenotype.png", dpi=200, bbox_inches="tight")
    plt.show()
# %%
# --- Delta NES heatmap (phenotype deviation from global) ---
hallmark_delta = hallmark[hallmark["delta_NES"].notna()].copy()
for tissue in TISSUES:
    t_data = hallmark_delta[hallmark_delta["tissue"] == tissue]
    if len(t_data) == 0:
        continue
    piv = t_data.pivot_table(index="term_short", columns="phenotype", values="delta_NES")
    if len(piv) == 0:
        continue
    piv.columns = [c.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
                   .replace("CD4_", "CD4_") for c in piv.columns]
    # Keep terms where any phenotype deviates > 0.3
    keep = piv.index[piv.abs().max(axis=1) > 0.3]
    piv = piv.loc[keep]
    if len(piv) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(max(10, len(piv.columns) * 1.2), max(5, len(piv) * 0.4)))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="PiYG", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "ΔNES (pheno − global)"})
    ax.set_title(f"Deviation from Global Tissue Signature in {tissue}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"gsea_delta_{tissue}_per_phenotype.png", dpi=200, bbox_inches="tight")
    plt.show()
# %%
# --- CSF flagged pathways detail plot ---
csf_flag_terms = [t for t in df["term"].unique()
                  if any(f.lower() in t.lower() for f in FLAG_TERMS)]
if csf_flag_terms:
    csf_all = df[(df["tissue"] == "CSF") & (df["term"].isin(csf_flag_terms))]
    piv = csf_all.pivot_table(index="term", columns="phenotype", values="NES")
    if len(piv) > 0:
        piv.columns = [c.replace("CD8_Activated_", "CD8a_").replace("CD8_Quiescent_", "CD8q_")
                       .replace("CD4_", "CD4_") for c in piv.columns]
        piv.index = [t.split("__")[-1][:60] for t in piv.index]
        
        # Add FDR annotation
        fdr_piv = csf_all.pivot_table(index="term", columns="phenotype", values="FDR")
        fdr_piv.index = piv.index
        fdr_piv.columns = piv.columns
        
        annot = piv.copy().astype(str)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iloc[i, j]
                f = fdr_piv.iloc[i, j]
                if pd.notna(v):
                    star = "***" if f < 0.01 else "**" if f < 0.05 else "*" if f < 0.25 else ""
                    annot.iloc[i, j] = f"{v:.2f}{star}"
                else:
                    annot.iloc[i, j] = ""
        
        fig, ax = plt.subplots(figsize=(max(10, len(piv.columns) * 1.2), max(4, len(piv) * 0.5)))
        sns.heatmap(piv, annot=annot, fmt="", cmap="RdBu_r", center=0,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "NES"})
        ax.set_title("CSF: Hypoxia / TGF-β / Interferon Pathways per Phenotype\n(* FDR<0.25, ** <0.05, *** <0.01)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig("csf_flagged_pathways.png", dpi=200, bbox_inches="tight")
        plt.show()

print("\nDone.")
# %%
