# %%
"""Sample-level (patient × tissue) pseudobulk DESeq2 + GSEA prerank for myeloid.

Mirrors the T-cell clone-based analysis in ``pathway_de_gsea_prerank.py`` but
aggregates all cells per (patient, tissue) — no TCR clonotypes.

Reads:
  data/objects/MYELOID_GBM.h5ad

Writes:
  results/04_pseudobulk_de_gsea_myeloid/
"""
import sys
import warnings
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.constants import DIRECTED_TISSUE_PAIRS  # noqa: E402
from modules.differential_expression import run_deseq2  # noqa: E402
from modules.pseudobulk import pseudobulk_counts_by_group  # noqa: E402

OUTPUT_DIR = paths.PSEUDOBULK_DE_GSEA_MYELOID_DIR
paths.ensure(OUTPUT_DIR)

MIN_CELLS = 10
MIN_PSEUDOBULKS = 6
GENE_SETS = "GO_Biological_Process_2023"
FDR_THRESHOLD = 0.25

# %%
print(f"Loading {paths.H5AD_MYELOID.name}...")
adata = sc.read(str(paths.H5AD_MYELOID))
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

de_results = {}

for t1, t2 in DIRECTED_TISSUE_PAIRS:
    label = f"myeloid_{t1}_vs_{t2}"
    print(f"\n{'=' * 60}\nDE: {label}\n{'=' * 60}")

    sub = adata[adata.obs["tissue"].isin([t1, t2])].copy()
    counts, meta = pseudobulk_counts_by_group(
        sub, group_key="tissue", patient_key="patient", layer="counts",
    )
    meta = meta[meta["tissue"].isin([t1, t2])].copy()
    counts = counts.loc[meta.index]

    print(f"  Pseudobulks: {len(meta)}  tissues={meta['tissue'].value_counts().to_dict()}")

    if len(meta) < MIN_PSEUDOBULKS:
        print(f"  SKIPPED: {len(meta)} pseudobulks (need >={MIN_PSEUDOBULKS})")
        continue
    if meta["patient"].nunique() < 2:
        print("  SKIPPED: need >=2 patients")
        continue

    res = run_deseq2(counts, meta, "~ patient + tissue", ["tissue", t2, t1])
    out_path = OUTPUT_DIR / f"de_{label}.csv"
    res.to_csv(out_path, index_label="gene")
    de_results[label] = res
    print(f"  Saved {out_path.name}")

# %%
gsea_rows = []
for label, res in de_results.items():
    ranking = res["stat"].dropna().sort_values(ascending=False)
    ranking = ranking[~ranking.index.duplicated()]
    if ranking.empty:
        continue
    pre_res = gp.prerank(
        rnk=ranking,
        gene_sets=GENE_SETS,
        processes=4,
        permutation_num=100,
        seed=7,
        verbose=False,
    )
    if pre_res.res2d is None or pre_res.res2d.empty:
        continue
    tab = pre_res.res2d.copy()
    tab["comparison"] = label
    tab.to_csv(OUTPUT_DIR / f"gsea_{label}.csv")
    gsea_rows.append(tab.assign(comparison=label))

summary_path = OUTPUT_DIR / "sample_pseudobulk_gsea_summary.csv"
if gsea_rows:
    pd.concat(gsea_rows, ignore_index=True).to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
else:
    pd.DataFrame(columns=["Term", "comparison"]).to_csv(summary_path, index=False)
    print(f"\nNo GSEA results; wrote empty {summary_path}")

print(f"\nDone. Outputs in {OUTPUT_DIR}")
