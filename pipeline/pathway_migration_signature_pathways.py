# %%
"""Pathway enrichment on the OT-distilled clean migration signature.

Picks up where pathway_migration_signature_ot.py leaves off: that script
ranks every gene by `migration_act` (the symmetric, tissue-context-
controlled OT delta across opposite migration directions). Here we ask
which pathways are systematically enriched at the top (or bottom) of
that ranking — i.e. which gene-sets contain the genes whose expression
goes up (or down) on the act of migrating, not on the tissue identity
of the destination.

This uses preranked GSEA (gseapy.prerank) rather than over-
representation against a thresholded list, so we keep the quantitative
signal from all 18,929 genes.

Inputs:
    results/pathway_migration_signature_ot/signature_distillation_table.csv

Outputs (per lineage):
    results/pathway_migration_signature_ot/gsea_prerank_<lineage>.csv
    results/pathway_migration_signature_ot/gsea_top_terms_<lineage>.png

Usage:
    python pipeline/pathway_migration_signature_pathways.py
"""
import sys
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

OUT_DIR = REPO_ROOT / "results" / "pathway_migration_signature_ot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_SETS = [
    "MSigDB_Hallmark_2020",
    "KEGG_2021_Human",
    "GO_Biological_Process_2023",
]
LINEAGES = ["CD8", "CD4"]
TOP_N_DISPLAY = 12          # top up + down per panel
FDR_THRESHOLD = 0.25
RANDOM_STATE = 42


# %% Load signature distillation table
sig_table = pd.read_csv(OUT_DIR / "signature_distillation_table.csv")
print(f"Loaded signature_distillation_table.csv: {len(sig_table)} rows "
      f"({sig_table['lineage'].nunique()} lineages)")


# %% Run preranked GSEA per lineage
all_gsea = []
for lineage in LINEAGES:
    sub = sig_table[sig_table["lineage"] == lineage].copy()
    if sub.empty:
        print(f"  {lineage}: no genes; skipping")
        continue

    # Rank genes by migration_act (symmetric component, averaged across pairs).
    rnk = sub[["gene", "mean_act"]].copy()
    rnk = rnk.dropna()
    # gseapy.prerank requires unique gene names
    rnk = rnk.groupby("gene", as_index=False)["mean_act"].mean()
    rnk = rnk.sort_values("mean_act", ascending=False)
    print(f"\n[{lineage}] running preranked GSEA on {len(rnk)} genes...")

    per_lib = []
    for lib in GENE_SETS:
        print(f"  - {lib}")
        try:
            pre = gp.prerank(
                rnk=rnk, gene_sets=lib,
                seed=RANDOM_STATE, threads=4,
                outdir=None, min_size=10, max_size=500,
                permutation_num=1000, verbose=False,
            )
        except Exception as e:
            print(f"    failed: {e}")
            continue
        df = pre.res2d.copy()
        df["library"] = lib
        df["lineage"] = lineage
        per_lib.append(df)
    if not per_lib:
        continue
    lin_gsea = pd.concat(per_lib, ignore_index=True)
    # Tidy column names
    rename = {c: c.replace(" ", "_").replace("-", "_") for c in lin_gsea.columns}
    lin_gsea = lin_gsea.rename(columns=rename)
    if "Term" in lin_gsea.columns:
        lin_gsea["Term_full"] = lin_gsea["library"] + " :: " + lin_gsea["Term"]
    lin_gsea.to_csv(OUT_DIR / f"gsea_prerank_{lineage}.csv", index=False)
    all_gsea.append(lin_gsea)
    print(f"  saved gsea_prerank_{lineage}.csv ({len(lin_gsea)} rows)")


# %% Plot top terms per lineage
def _short(label, maxlen=55):
    s = str(label)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


for lin_gsea in all_gsea:
    lineage = lin_gsea["lineage"].iloc[0]
    # NES + FDR columns vary by gseapy version
    nes_col = "NES" if "NES" in lin_gsea.columns else "nes"
    fdr_col = ("FDR_q_val" if "FDR_q_val" in lin_gsea.columns
               else "FDR q-val" if "FDR q-val" in lin_gsea.columns
               else "fdr")
    if nes_col not in lin_gsea.columns:
        print(f"  no NES col for {lineage}; skipping plot")
        continue
    df = lin_gsea.copy()
    df[nes_col] = pd.to_numeric(df[nes_col], errors="coerce")
    if fdr_col in df.columns:
        df[fdr_col] = pd.to_numeric(df[fdr_col], errors="coerce")
    df = df.dropna(subset=[nes_col])
    # Top N up and N down by NES, filtered by FDR
    df_sig = df[df[fdr_col] < FDR_THRESHOLD] if fdr_col in df.columns else df
    top_up = df_sig.nlargest(TOP_N_DISPLAY, nes_col)
    top_dn = df_sig.nsmallest(TOP_N_DISPLAY, nes_col)
    if top_up.empty and top_dn.empty:
        # fall back to unfiltered
        top_up = df.nlargest(TOP_N_DISPLAY, nes_col)
        top_dn = df.nsmallest(TOP_N_DISPLAY, nes_col)

    panel = pd.concat([top_up.assign(direction="migration_up"),
                       top_dn.assign(direction="migration_down")],
                      ignore_index=True)
    panel = panel.sort_values(nes_col)

    fig, ax = plt.subplots(figsize=(11, 0.34 * len(panel) + 1.5))
    colors = ["#2166ac" if v < 0 else "#b2182b" for v in panel[nes_col].values]
    y = np.arange(len(panel))
    ax.barh(y, panel[nes_col].values, color=colors, alpha=0.85,
             edgecolor="black", linewidth=0.5)
    # Add FDR significance stars
    for i, (nes, fdr) in enumerate(zip(panel[nes_col].values, panel[fdr_col].values)):
        try:
            f = float(fdr)
        except Exception:
            f = 1.0
        star = "***" if f < 0.001 else "**" if f < 0.01 else "*" if f < 0.05 else ""
        tx = nes + (0.05 if nes >= 0 else -0.05)
        ax.text(tx, i, star, va="center",
                ha="left" if nes >= 0 else "right", fontsize=9)
    labels = [f"[{lib.split('_')[0]}] {_short(t)}"
              for lib, t in zip(panel["library"], panel["Term"])]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("NES (preranked GSEA on migration_act ranking)")
    ax.set_title(
        f"{lineage}: pathways enriched in OT-distilled clean migration signature\n"
        f"(positive NES = up on act of migrating; negative = down; "
        f"* FDR<0.05, ** FDR<0.01, *** FDR<0.001)"
    )
    fig.tight_layout()
    out_png = OUT_DIR / f"gsea_top_terms_{lineage}.png"
    out_pdf = OUT_DIR / f"gsea_top_terms_{lineage}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png.name}")


# %% Final
print("\n" + "=" * 60)
print("DONE — pathway enrichment outputs in",
      OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.glob("gsea_*")):
    print(f"  {f.name}")
