# %%
"""Temporal pathway and gene-expression scoring (myeloid lineage).

Mirrors 10_temporal_scores.py but for the myeloid AnnData. Reuses the
T-cell-derived pathway_definitions.csv and unions the T-cell gene panel
into the myeloid panel so cross-lineage gene comparisons are possible.
All output rows carry lineage="Myeloid".
"""
import subprocess
import sys
import warnings
from pathlib import Path

import gseapy as gp  # noqa: F401  (kept for parity; unused now that defs are reused)
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402

# %%
# ---- Config ----
DATA_PATH = paths.H5AD_MYELOID
OUTPUT_DIR = paths.TEMPORAL_SCORES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATHWAY_DEF_PATH = OUTPUT_DIR / "pathway_definitions.csv"
TCELL_PANEL_PATH = OUTPUT_DIR / "gene_panel_tcell.txt"

GROUP_KEYS = ["patient", "tissue", "timepoint", "phenotype"]
SAMPLE_KEYS = ["patient", "tissue", "timepoint"]
# Composition: keep singleton phenotypes; drop samples with <10 total cells.
MIN_COMP_CELLS   = 1
MIN_SAMPLE_CELLS = 10
# Pathway / gene aggregation: mean is too noisy below 3 cells.
MIN_SCORE_CELLS  = 3
MIN_PATHWAY_GENES = 10
TOP_MARKERS_PER_PHENO = 50
GENE_CHUNK = 100
LINEAGE_LABEL = "Myeloid"

# %%
# ---- Load AnnData ----
# The myeloid h5ad's obs schema mostly matches the T-cell object
# (phenotype, patient, timepoint). Tissue is exposed as 'sample_type'
# in this export; we alias it to 'tissue' for downstream consistency.
adata = sc.read(str(DATA_PATH))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes from {DATA_PATH.name}")

if "tissue" not in adata.obs.columns and "sample_type" in adata.obs.columns:
    adata.obs = adata.obs.rename(columns={"sample_type": "tissue"})
    print("aliased sample_type -> tissue")

required = ["phenotype", "patient", "timepoint", "tissue"]
missing = [c for c in required if c not in adata.obs.columns]
if missing:
    raise KeyError(f"Expected obs columns missing: {missing}; got {list(adata.obs.columns)}")

# Coerce types. timepoint goes through int first to drop "1.0" -> "1";
# pd.to_numeric is idempotent if values are already integer-formatted strings.
adata.obs["timepoint"] = pd.to_numeric(adata.obs["timepoint"]).astype(int).astype(str)
for c in ["patient", "tissue", "phenotype"]:
    adata.obs[c] = adata.obs[c].astype(str)

n_phenos = adata.obs["phenotype"].nunique()
print(f"Unique myeloid phenotypes: {n_phenos}")
print(f"  {sorted(adata.obs['phenotype'].unique().tolist())}")
print(f"Unique timepoints: {sorted(adata.obs['timepoint'].unique().tolist())}")
print(f"Unique patients:   {sorted(adata.obs['patient'].unique().tolist())}")
print(f"Unique tissues:    {sorted(adata.obs['tissue'].unique().tolist())}")

# %%
# ---- Confirm X is normalized log-counts; normalize in-memory if not ----
sample_n = min(2000, adata.n_obs)
sample = adata.X[:sample_n]
if sparse.issparse(sample):
    sample = sample.toarray()
sample = np.asarray(sample)
nz = sample[sample > 0]
if nz.size == 0:
    raise RuntimeError("adata.X has no non-zero entries in the first sample window")

max_val = float(nz.max())
all_int = bool(np.allclose(nz, np.round(nz)))
print(f"X check: nz max={max_val:.3f}, all_integer={all_int}")
if all_int or max_val > 50:
    print("Looks like raw counts; running normalize_total + log1p in-memory")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
else:
    print("Treating X as already normalized log-counts; not modifying")

# %%
# ---- Composition table ----
print("\nBuilding composition table...")
comp = (
    adata.obs.groupby(GROUP_KEYS, observed=True)
    .size().rename("n_cells_phenotype").reset_index()
)
comp["n_cells_total"] = comp.groupby(SAMPLE_KEYS, observed=True)["n_cells_phenotype"].transform("sum")
comp = comp[(comp["n_cells_phenotype"] >= MIN_COMP_CELLS) &
            (comp["n_cells_total"] >= MIN_SAMPLE_CELLS)].copy()
comp["frac"] = comp["n_cells_phenotype"] / comp["n_cells_total"]
comp["lineage"] = LINEAGE_LABEL
comp = comp[SAMPLE_KEYS + ["phenotype", "lineage", "n_cells_phenotype", "n_cells_total", "frac"]]
composition_path = OUTPUT_DIR / "temporal_composition_myeloid.csv"
comp.to_csv(composition_path, index=False)
print(f"Composition rows: {len(comp)} (samples kept: {comp[SAMPLE_KEYS].drop_duplicates().shape[0]})")

# %%
# ---- Reuse pathway definitions from the T cell run ----
if not PATHWAY_DEF_PATH.exists():
    raise FileNotFoundError(
        f"{PATHWAY_DEF_PATH} not found; run pipeline/10_temporal_scores.py first"
    )
pathway_def = pd.read_csv(PATHWAY_DEF_PATH)
print(f"\nLoaded {len(pathway_def)} pathway definitions from {PATHWAY_DEF_PATH.name}")

# %%
# ---- Build myeloid gene panel (markers ∪ T cell panel, intersected with var_names) ----
print("\nComputing top markers per myeloid phenotype...")
sc.tl.rank_genes_groups(adata, "phenotype", method="wilcoxon", use_raw=False)
marker_genes = set()
phenos = [str(p) for p in adata.obs["phenotype"].dropna().unique()]
for pheno in phenos:
    df = sc.get.rank_genes_groups_df(adata, group=pheno)
    marker_genes |= set(df["names"].head(TOP_MARKERS_PER_PHENO))
print(f"Marker genes (union of top {TOP_MARKERS_PER_PHENO} per phenotype): {len(marker_genes)}")

if not TCELL_PANEL_PATH.exists():
    raise FileNotFoundError(
        f"{TCELL_PANEL_PATH} not found; run pipeline/10_temporal_scores.py first"
    )
tcell_panel = {g.strip() for g in TCELL_PANEL_PATH.read_text().splitlines() if g.strip()}
print(f"T cell panel loaded: {len(tcell_panel)} genes")

panel = sorted((marker_genes | tcell_panel) & set(adata.var_names))
print(f"Final myeloid gene panel (intersected with adata.var_names): {len(panel)}")

(OUTPUT_DIR / "gene_panel_myeloid.txt").write_text("\n".join(panel) + "\n")

# %%
# ---- Pathway scoring ----
if "_score" in adata.obs.columns:
    del adata.obs["_score"]

print(f"\nScoring pathways with >= {MIN_PATHWAY_GENES} genes present in adata...")
gene_set = set(adata.var_names)
score_records = []
n_scored = 0
n_skipped = 0

for _, row in pathway_def.iterrows():
    genes = [g for g in str(row["gene_list"]).split(";") if g in gene_set]
    if len(genes) < MIN_PATHWAY_GENES:
        n_skipped += 1
        continue
    sc.tl.score_genes(
        adata, gene_list=genes, score_name="_score",
        use_raw=False, random_state=42,
    )
    grouped = (
        adata.obs.groupby(GROUP_KEYS, observed=True)["_score"]
        .agg(["mean", "size"]).reset_index()
        .rename(columns={"mean": "mean_score", "size": "n_cells"})
    )
    grouped = grouped[grouped["n_cells"] >= MIN_SCORE_CELLS]
    grouped["pathway"] = row["pathway"]
    grouped["source"] = row["source"]
    grouped["lineage"] = LINEAGE_LABEL
    score_records.append(grouped)
    n_scored += 1
    if n_scored % 25 == 0:
        print(f"  scored {n_scored} pathways...")

if "_score" in adata.obs.columns:
    del adata.obs["_score"]

if score_records:
    pathway_scores = pd.concat(score_records, ignore_index=True)
else:
    pathway_scores = pd.DataFrame(
        columns=GROUP_KEYS + ["mean_score", "n_cells", "pathway", "source", "lineage"]
    )

pathway_scores_path = OUTPUT_DIR / "temporal_pathway_scores_myeloid.csv"
pathway_scores.to_csv(pathway_scores_path, index=False)
print(f"Pathways scored: {n_scored}, skipped (<{MIN_PATHWAY_GENES} genes present): {n_skipped}")
print(f"Pathway score rows: {len(pathway_scores)}")

# %%
# ---- Gene expression aggregation (chunked, incremental write) ----
print(f"\nAggregating myeloid gene expression for {len(panel)} genes in chunks of {GENE_CHUNK}...")
gene_expr_path = OUTPUT_DIR / "temporal_gene_expression_myeloid.csv"
if gene_expr_path.exists():
    gene_expr_path.unlink()

group_idx = adata.obs[GROUP_KEYS].groupby(GROUP_KEYS, observed=True).indices
group_keys_keep = [k for k, idx in group_idx.items() if len(idx) >= MIN_SCORE_CELLS]
print(f"Groups passing min-cells filter: {len(group_keys_keep)} (of {len(group_idx)})")

if not group_keys_keep:
    raise RuntimeError(f"No groups have >= {MIN_SCORE_CELLS} cells")

n_cells_per_group = np.array([len(group_idx[k]) for k in group_keys_keep])
patients = np.array([k[0] for k in group_keys_keep])
tissues = np.array([k[1] for k in group_keys_keep])
timepoints = np.array([k[2] for k in group_keys_keep])
phenotypes = np.array([str(k[3]) for k in group_keys_keep])
lineages = np.array([LINEAGE_LABEL] * len(group_keys_keep))

n_rows_written = 0
header_written = False
for chunk_start in range(0, len(panel), GENE_CHUNK):
    chunk_genes = panel[chunk_start:chunk_start + GENE_CHUNK]
    var_loc = adata.var_names.get_indexer(chunk_genes)
    X_chunk = adata.X[:, var_loc]
    if sparse.issparse(X_chunk):
        X_chunk = X_chunk.toarray()
    X_chunk = np.asarray(X_chunk, dtype=np.float32)

    n_groups = len(group_keys_keep)
    n_genes = len(chunk_genes)
    means = np.empty((n_groups, n_genes), dtype=np.float32)
    fracs = np.empty((n_groups, n_genes), dtype=np.float32)
    for gi, key in enumerate(group_keys_keep):
        sub = X_chunk[group_idx[key], :]
        means[gi] = sub.mean(axis=0)
        fracs[gi] = (sub > 0).mean(axis=0)

    chunk_df = pd.DataFrame({
        "patient": np.repeat(patients, n_genes),
        "tissue": np.repeat(tissues, n_genes),
        "timepoint": np.repeat(timepoints, n_genes),
        "phenotype": np.repeat(phenotypes, n_genes),
        "lineage": np.repeat(lineages, n_genes),
        "gene": np.tile(chunk_genes, n_groups),
        "mean_expr": means.ravel(),
        "frac_expressing": fracs.ravel(),
        "n_cells": np.repeat(n_cells_per_group, n_genes),
    })
    chunk_df.to_csv(gene_expr_path, mode="a", index=False, header=not header_written)
    header_written = True
    n_rows_written += len(chunk_df)
    end_idx = chunk_start + len(chunk_genes)
    print(f"  genes {chunk_start}..{end_idx}: wrote {len(chunk_df)} rows (cum: {n_rows_written})")

# %%
# ---- Final summary ----
print("\n" + "=" * 60)
print("DONE (myeloid)")
print("=" * 60)
print(f"  unique myeloid phenotypes              : {n_phenos}")
print(f"  gene_panel_myeloid.txt                 : {len(panel)} genes")
print(f"  temporal_composition_myeloid.csv       : {len(comp)} rows")
print(f"  temporal_pathway_scores_myeloid.csv    : {len(pathway_scores)} rows ({n_scored} pathways scored)")
print(f"  temporal_gene_expression_myeloid.csv   : {n_rows_written} rows")
try:
    du = subprocess.check_output(["du", "-sh", str(OUTPUT_DIR)]).decode().strip()
    print(f"  output dir: {du}")
except Exception as e:
    print(f"  (du failed: {e})")
