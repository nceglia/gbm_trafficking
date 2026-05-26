# %%
"""Clone-level pseudobulk DESeq2 + GSEA prerank.

Unit of analysis: clone (not cell). For each (lineage, tissue_pair), pseudobulk
raw counts per (clone x tissue), run DESeq2 with ~patient + tissue, then GSEA
prerank on the Wald statistic.
"""
import sys
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.clone_helpers import infer_lineage_from_phenotype
from modules.constants import DIRECTED_TISSUE_PAIRS
from modules.differential_expression import run_deseq2
from modules.pseudobulk import pseudobulk_counts_by_clone_tissue

# %%
# ---- Config ----
DATA_PATH = REPO_ROOT / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"
OUTPUT_DIR = REPO_ROOT / "results" / "04_pseudobulk_de_gsea"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LINEAGES = ("CD8", "CD4")
MIN_CELLS = 3
MIN_PSEUDOBULKS = 10

GENE_SETS = "GO_Biological_Process_2023"
FDR_THRESHOLD = 0.25

LIBRARY_SLUGS = {
    "MSigDB_Hallmark_2020": "hallmark",
    "KEGG_2021_Human": "kegg",
    "GO_Biological_Process_2023": "gobp",
}
LIBRARY_TAG = LIBRARY_SLUGS[GENE_SETS]

# %%
# ---- Load data + sanity check ----
adata = sc.read(str(DATA_PATH))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
print(f"Gene-set library: {GENE_SETS} (tag: {LIBRARY_TAG})")

tcr = adata.obs[adata.obs["trb"].notna()].copy()
tcr["_lineage"] = tcr["phenotype"].map(infer_lineage_from_phenotype)
cl = tcr.groupby("trb")["_lineage"].nunique()
n_switching = (cl > 1).sum()
print(f"Lineage-switching clones remaining: {n_switching} (should be 0)")

# %%
# ---- Step 2: Pseudobulk + DESeq2 per (lineage, pair) ----
de_results = {}

for lineage in LINEAGES:
    for t1, t2 in DIRECTED_TISSUE_PAIRS:
        label = f"{lineage}_{t1}_vs_{t2}"
        print(f"\n{'=' * 60}")
        print(f"DE: {label}")
        print(f"{'=' * 60}")

        counts, meta = pseudobulk_counts_by_clone_tissue(
            adata, clone_key="trb", tissues=(t1, t2),
            lineage=lineage, min_cells=MIN_CELLS, layer="counts",
        )

        n_clones = meta["clone"].nunique()
        print(f"  Clones in both tissues (>={MIN_CELLS} cells each): {n_clones}")
        print(f"  Pseudobulk samples: {len(meta)}  ({meta['tissue'].value_counts().to_dict()})")
        print(f"  Patients: {sorted(meta['patient'].unique().tolist())}")
        print(f"  Cells per pseudobulk: median={meta['n_cells'].median():.0f}, "
              f"range=[{meta['n_cells'].min()}, {meta['n_cells'].max()}]")

        if len(meta) < MIN_PSEUDOBULKS:
            print(f"  SKIPPED: only {len(meta)} pseudobulks (need >={MIN_PSEUDOBULKS})")
            continue

        n_patients = meta["patient"].nunique()
        if n_patients < 2:
            print(f"  SKIPPED: only {n_patients} patient(s); need >=2 for patient covariate")
            continue

        design = "~ patient + tissue"
        contrast = ["tissue", t2, t1]
        print(f"  Design: {design}, contrast: {contrast}")

        res = run_deseq2(counts, meta, design, contrast)

        n_up = ((res["padj"] < 0.05) & (res["log2FoldChange"] > 0)).sum()
        n_down = ((res["padj"] < 0.05) & (res["log2FoldChange"] < 0)).sum()
        print(f"  DEGs (padj<0.05): {n_up + n_down}  (up in {t2}: {n_up}, up in {t1}: {n_down})")
        print(f"  Top 5 by |stat|:")
        print(res.head(5)[["log2FoldChange", "stat", "pvalue", "padj"]].to_string())

        out_path = OUTPUT_DIR / f"de_{label}.csv"
        res.to_csv(out_path, index_label="gene")
        de_results[label] = res
        print(f"  Saved: {out_path.name}")

# %%
# ---- Step 3: GSEA prerank on saved DE tables ----
gsea_results = {}

for label, res in de_results.items():
    print(f"\n{'=' * 60}")
    print(f"GSEA ({LIBRARY_TAG}): {label}")
    print(f"{'=' * 60}")

    ranking = res["stat"].dropna().sort_values(ascending=False)
    ranking = ranking[~ranking.index.duplicated()]
    print(f"  Genes ranked: {len(ranking)}")

    pre = gp.prerank(
        rnk=ranking, gene_sets=GENE_SETS, outdir=None,
        seed=42, min_size=10, max_size=500,
        permutation_num=1000, no_plot=True,
    )
    gsea = pre.res2d.copy()
    gsea["FDR q-val"] = gsea["FDR q-val"].astype(float)
    gsea["NES"] = gsea["NES"].astype(float)
    gsea = gsea.sort_values("NES", ascending=False)

    sig = gsea[gsea["FDR q-val"] < FDR_THRESHOLD]
    print(f"  Significant pathways (FDR<{FDR_THRESHOLD}): {len(sig)}")
    if len(sig) > 0:
        pos = sig[sig["NES"] > 0]
        neg = sig[sig["NES"] < 0]
        print(f"    Positive NES: {len(pos)}")
        for _, row in pos.head(5).iterrows():
            print(f"      {row['Term']}: NES={row['NES']:.2f}, FDR={row['FDR q-val']:.3f}")
        print(f"    Negative NES: {len(neg)}")
        for _, row in neg.tail(5).iterrows():
            print(f"      {row['Term']}: NES={row['NES']:.2f}, FDR={row['FDR q-val']:.3f}")

    out_path = OUTPUT_DIR / f"gsea_{LIBRARY_TAG}_{label}.csv"
    gsea.drop(columns=["Name"]).to_csv(out_path, index=False)
    gsea_results[label] = gsea
    print(f"  Saved: {out_path.name}")

# %%
# ---- Step 4: Combined dot-heatmap ----
if not gsea_results:
    print("No GSEA results to plot.")
else:
    # Collect pathways significant in at least one context
    sig_terms = set()
    for label, gsea in gsea_results.items():
        sig_terms |= set(gsea[gsea["FDR q-val"] < FDR_THRESHOLD]["Term"])

    if not sig_terms:
        print(f"No pathways reached FDR<{FDR_THRESHOLD} in any context.")
    else:
        contexts = list(gsea_results.keys())
        terms = sorted(sig_terms)

        nes_mat = pd.DataFrame(np.nan, index=terms, columns=contexts)
        fdr_mat = pd.DataFrame(np.nan, index=terms, columns=contexts)

        for label, gsea in gsea_results.items():
            for _, row in gsea.iterrows():
                if row["Term"] in sig_terms:
                    nes_mat.loc[row["Term"], label] = row["NES"]
                    fdr_mat.loc[row["Term"], label] = row["FDR q-val"]

        # Strip library prefix from term names for display
        prefix = GENE_SETS + "__"
        nes_mat.index = [t.replace(prefix, "") for t in nes_mat.index]
        fdr_mat.index = nes_mat.index

        neg_log_fdr = -np.log10(fdr_mat.clip(lower=1e-10))

        # Dot-heatmap: color = NES, size = -log10(FDR)
        n_terms = len(terms)
        n_contexts = len(contexts)

        fig, ax = plt.subplots(figsize=(max(8, n_contexts * 1.8), max(6, n_terms * 0.35)))

        # Normalise dot sizes
        max_neg_log = neg_log_fdr.max().max()
        size_scale = 200 / max(max_neg_log, 1)

        # NES color normalisation
        vmax = max(abs(nes_mat.min().min()), abs(nes_mat.max().max()), 1)
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)
        cmap = plt.cm.RdBu_r

        for i, term in enumerate(nes_mat.index):
            for j, ctx in enumerate(contexts):
                nes_val = nes_mat.loc[term, ctx]
                fdr_val = neg_log_fdr.loc[term, ctx]
                if pd.isna(nes_val):
                    continue
                ax.scatter(
                    j, i,
                    s=fdr_val * size_scale,
                    c=[cmap(norm(nes_val))],
                    edgecolors="k", linewidths=0.3,
                )

        ax.set_xticks(range(n_contexts))
        ax.set_xticklabels(contexts, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_terms))
        ax.set_yticklabels(nes_mat.index, fontsize=8)
        ax.set_xlim(-0.5, n_contexts - 0.5)
        ax.set_ylim(-0.5, n_terms - 0.5)
        ax.invert_yaxis()
        ax.grid(axis="both", alpha=0.15)

        # Colorbar for NES
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("NES", fontsize=10)

        # Size legend
        legend_sizes = [0.5, 1.0, 2.0]
        legend_sizes = [s for s in legend_sizes if s <= max_neg_log]
        if not legend_sizes:
            legend_sizes = [max_neg_log / 2]
        handles = [
            plt.scatter([], [], s=s * size_scale, c="gray", edgecolors="k", linewidths=0.3)
            for s in legend_sizes
        ]
        labels = [f"{s:.1f}" for s in legend_sizes]
        ax.legend(handles, labels, title="-log10(FDR)", loc="upper left",
                  bbox_to_anchor=(1.15, 1), frameon=False, fontsize=8, title_fontsize=9)

        ax.set_title(f"Clone Pseudobulk Pathway Enrichment ({LIBRARY_TAG})\n"
                     "(DESeq2 ~patient + tissue, GSEA prerank)",
                      fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"clone_pseudobulk_gsea_dotheatmap_{LIBRARY_TAG}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved dot-heatmap: {out_path}")

print("\nDone.")

# %%
