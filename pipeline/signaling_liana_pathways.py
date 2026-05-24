# %%
"""Ligand-receptor signaling with LIANA, intersected with cross-lineage
pathway-correlation candidates from script 11.

For each (tissue, timepoint) bin we run LIANA's CellPhoneDB method on the
combined T-cell + myeloid object using `phenotype` as the cell-type label.
The resulting L-R rows are then annotated with pathway membership (from
script 10's pathway_definitions.csv) and filtered to the candidate
T-pathway / M-pathway pairs that came out of script 11.
"""
import subprocess
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

# %%
# ---- Dependency check ----
try:
    import liana as li
except ImportError:
    print("liana is required for this script.")
    print("Install with: pip install liana decoupler-py")
    sys.exit(0)

# %%
# ---- Config ----
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.modules.myeloid_groups import MYELOID_GROUPS, regroup_obs

COMBINED_PATH = "/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS_MYELOID_combined.h5ad"
PATHWAY_DEFS_PATH = REPO_ROOT / "results" / "10_temporal_scores" / "pathway_definitions.csv"
PATHWAY_CORRS_PATH = REPO_ROOT / "results" / "11_cross_lineage_correlations" / "pathway_correlations.csv"

OUTPUT_DIR = REPO_ROOT / "results" / "12_liana_signaling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TOTAL_CELLS = 100
MIN_PHENOTYPE_CELLS = 20
MIN_PHENOTYPES = 2
EXPR_PROP = 0.10
Q_THRESHOLD = 0.25

t_start = time.time()

# %%
# ---- Load combined AnnData + sanity checks ----
adata = sc.read(COMBINED_PATH)
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes from combined object")

# X must be log-normalized for LIANA's expression-fraction logic.
nz_max = adata.X[:5000].max()
if nz_max > 50:
    raise ValueError(
        "X looks like raw counts; ensure the build_combined_object script "
        "writes log-normalized X"
    )

required_obs = ["phenotype", "tissue", "timepoint", "patient"]
for col in required_obs:
    if col not in adata.obs.columns:
        raise KeyError(f"Required obs column missing: {col!r}")

pheno = adata.obs["phenotype"]
mask = pheno.notna() & (pheno.astype(str).str.len() > 0)
n_dropped = (~mask).sum()
adata = adata[mask].copy()
print(f"Dropped {n_dropped} cells with NaN/empty phenotype; {adata.n_obs} remain")

# %%
# ---- Apply coarse myeloid regrouping (T cells pass through untouched) ----
# Cast to object so we can introduce new label values (Microglia, Mac_Immunosupp,
# etc.) that aren't in the original categorical levels.
adata.obs["phenotype"] = adata.obs["phenotype"].astype(object)
is_myeloid = adata.obs["phenotype"].isin(MYELOID_GROUPS.keys())
remapped = regroup_obs(adata.obs.loc[is_myeloid])
adata.obs.loc[is_myeloid, "phenotype"] = remapped.values
# Drop cells whose myeloid phenotype didn't map (Mast cell, immature, REMOVE)
n_before = adata.n_obs
adata = adata[~adata.obs["phenotype"].isna()].copy()
n_remap = int(is_myeloid.sum())
print(f"Regrouped {n_remap} myeloid cells onto coarse groups "
      f"({sorted(set(MYELOID_GROUPS.values()))}); "
      f"{n_before - adata.n_obs} unmapped cells dropped, {adata.n_obs} remain")

# %%
# ---- LIANA per (tissue, timepoint) bin ----
results = []
skipped = []
tissues = sorted(adata.obs["tissue"].dropna().unique())
timepoints = sorted(adata.obs["timepoint"].dropna().unique())
print(f"\nTissues: {tissues}")
print(f"Timepoints: {timepoints}")

for tissue, tp in product(tissues, timepoints):
    sub = adata[(adata.obs["tissue"] == tissue) & (adata.obs["timepoint"] == tp)].copy()
    if sub.n_obs < MIN_TOTAL_CELLS:
        skipped.append({
            "tissue": tissue, "timepoint": tp,
            "reason": "fewer_than_100_total_cells", "n_cells": sub.n_obs,
        })
        continue

    counts = sub.obs["phenotype"].value_counts()
    keep_phenos = counts[counts >= MIN_PHENOTYPE_CELLS].index.tolist()
    if len(keep_phenos) < MIN_PHENOTYPES:
        skipped.append({
            "tissue": tissue, "timepoint": tp,
            "reason": "fewer_than_2_phenotypes_with_20_cells", "n_cells": sub.n_obs,
        })
        continue

    sub = sub[sub.obs["phenotype"].isin(keep_phenos)].copy()
    sub.obs["phenotype"] = sub.obs["phenotype"].astype(str)

    try:
        li.mt.cellphonedb(
            sub,
            groupby="phenotype",
            resource_name="consensus",
            expr_prop=EXPR_PROP,
            verbose=False,
            use_raw=False,
        )
    except Exception as e:
        skipped.append({
            "tissue": tissue, "timepoint": tp,
            "reason": f"liana_failed: {type(e).__name__}: {e}",
            "n_cells": sub.n_obs,
        })
        continue

    df = sub.uns["liana_res"].copy()
    df["tissue"] = tissue
    df["timepoint"] = tp
    df["status"] = "scored"
    results.append(df)
    print(f"  {tissue} t{tp}: {len(df)} L-R rows over {len(keep_phenos)} phenotypes ({sub.n_obs} cells)")

if results:
    liana_full = pd.concat(results, ignore_index=True)
else:
    liana_full = pd.DataFrame()

skipped_df = pd.DataFrame(skipped)
if len(skipped_df):
    skipped_df.to_csv(OUTPUT_DIR / "skipped_bins.csv", index=False)

# %%
# ---- Pathway membership lookup ----
pathway_defs = pd.read_csv(PATHWAY_DEFS_PATH)
gene_to_pathways = {}
for _, row in pathway_defs.iterrows():
    genes = str(row["gene_list"]).split(";")
    for g in genes:
        g = g.strip()
        if g:
            gene_to_pathways.setdefault(g, set()).add(row["pathway"])
print(f"\nLoaded {len(pathway_defs)} pathways covering {len(gene_to_pathways)} genes")

# %%
# ---- Annotate L-R rows with pathway membership ----
def _complex_pathways(s):
    parts = str(s).split("_")
    pset = set().union(*(gene_to_pathways.get(g.strip(), set()) for g in parts))
    return ";".join(sorted(pset))

if len(liana_full):
    liana_full["ligand_pathways"] = liana_full["ligand_complex"].apply(_complex_pathways)
    liana_full["receptor_pathways"] = liana_full["receptor_complex"].apply(_complex_pathways)

liana_full.to_csv(OUTPUT_DIR / "liana_per_timepoint.csv", index=False)
print(f"Wrote {len(liana_full)} LIANA rows to liana_per_timepoint.csv")

# %%
# ---- Filter to candidate pathway pairs from script 11 ----
edges = []
if PATHWAY_CORRS_PATH.exists() and len(liana_full):
    pathway_corrs = pd.read_csv(PATHWAY_CORRS_PATH)
    candidates = pathway_corrs[pathway_corrs["q_bh"] < Q_THRESHOLD].copy()
    print(f"Loaded {len(pathway_corrs)} pathway correlations; "
          f"{len(candidates)} pass q_bh < {Q_THRESHOLD}")

    cand_by_tissue = {t: g for t, g in candidates.groupby("tissue")}

    for row in liana_full.itertuples(index=False):
        tissue = row.tissue
        cand = cand_by_tissue.get(tissue)
        if cand is None or not len(cand):
            continue
        lp = set(row.ligand_pathways.split(";")) if row.ligand_pathways else set()
        rp = set(row.receptor_pathways.split(";")) if row.receptor_pathways else set()
        if not lp and not rp:
            continue

        for c in cand.itertuples(index=False):
            t_in_l = c.t_pathway in lp
            m_in_r = c.m_pathway in rp
            t_in_r = c.t_pathway in rp
            m_in_l = c.m_pathway in lp

            direction = None
            if t_in_l and m_in_r:
                direction = "ligand_in_t_pathway"
            elif t_in_r and m_in_l:
                direction = "ligand_in_m_pathway"
            else:
                continue

            edges.append({
                "tissue": tissue,
                "timepoint": row.timepoint,
                "source_phenotype": row.source,
                "target_phenotype": row.target,
                "ligand_complex": row.ligand_complex,
                "receptor_complex": row.receptor_complex,
                "lr_means": getattr(row, "lr_means", np.nan),
                "cellphone_pvals": getattr(row, "cellphone_pvals", np.nan),
                "candidate_t_phenotype": getattr(c, "t_phenotype", None),
                "candidate_m_phenotype": getattr(c, "m_phenotype", None),
                "candidate_t_pathway": c.t_pathway,
                "candidate_m_pathway": c.m_pathway,
                "candidate_q_bh": c.q_bh,
                "direction": direction,
            })
elif not PATHWAY_CORRS_PATH.exists():
    print(f"WARNING: {PATHWAY_CORRS_PATH} not found — skipping edge construction")

edges_df = pd.DataFrame(edges)
edges_df.to_csv(OUTPUT_DIR / "signaling_edges.csv", index=False)

# %%
# ---- Summary ----
elapsed = time.time() - t_start

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

if len(liana_full):
    by_bin = liana_full.groupby(["tissue", "timepoint"]).size().rename("n_lr_rows")
    print("\nLIANA rows scored by (tissue, timepoint):")
    print(by_bin.to_string())
    n_unique = liana_full[["ligand_complex", "receptor_complex"]].drop_duplicates().shape[0]
    print(f"\nUnique L-R pairs: {n_unique}")
else:
    print("\nNo LIANA rows scored.")

print(f"\nBins skipped: {len(skipped_df)}")
if len(skipped_df):
    print(skipped_df.groupby("reason").size().to_string())

print(f"\nsignaling_edges.csv rows: {len(edges_df)}")

try:
    du_out = subprocess.run(
        ["du", "-sh", str(OUTPUT_DIR)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"\nOutput dir: {du_out}")
except Exception as e:
    print(f"\nOutput dir size unavailable: {e}")

print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
