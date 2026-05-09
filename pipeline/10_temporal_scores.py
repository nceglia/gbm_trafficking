# %%
"""Temporal pathway and gene-expression scoring.

Aggregates per-cell pathway scores and gene expression up to
(patient, tissue, timepoint, phenotype) groups so downstream temporal
models can consume one row per group rather than per cell.
"""
import subprocess
import sys
import warnings
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.clone_helpers import infer_lineage_from_phenotype

# %%
# ---- Config ----
DATA_PATH = REPO_ROOT / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"
GSEA_DIR = REPO_ROOT / "results" / "04_pseudobulk_de_gsea"
OUTPUT_DIR = REPO_ROOT / "results" / "10_temporal_scores"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODULES_DIR = REPO_ROOT / "pipeline" / "modules"
HALLMARK_FAMILY_TSV = MODULES_DIR / "pathway_families_hallmark.tsv"
KEGG_FAMILY_TSV = MODULES_DIR / "pathway_families_kegg.tsv"
KEGG_EXCLUDED_TSV = MODULES_DIR / "pathway_families_kegg_excluded.tsv"

GROUP_KEYS = ["patient", "tissue", "timepoint", "phenotype"]
# Composition: keep singleton phenotypes (cells>=1), drop whole samples
# with fewer than 10 cells across all phenotypes.
MIN_COMP_CELLS   = 1
MIN_SAMPLE_CELLS = 10
# Pathway / gene aggregation: mean is too noisy below 3 cells.
MIN_SCORE_CELLS  = 3
MIN_PATHWAY_GENES = 10
TOP_MARKERS_PER_PHENO = 50
GSEA_FDR_THRESHOLD = 0.25
GENE_CHUNK = 100

# %%
# ---- Load AnnData ----
adata = sc.read(str(DATA_PATH))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

# %%
# ---- Composition table (cells per phenotype within each sample) ----
print("\nBuilding composition table...")
comp = (
    adata.obs.groupby(GROUP_KEYS, observed=True)
    .size().rename("n_cells_phenotype").reset_index()
)
sample_keys = ["patient", "tissue", "timepoint"]
comp["n_cells_total"] = comp.groupby(sample_keys, observed=True)["n_cells_phenotype"].transform("sum")
comp = comp[(comp["n_cells_phenotype"] >= MIN_COMP_CELLS) &
            (comp["n_cells_total"] >= MIN_SAMPLE_CELLS)].copy()
comp["frac"] = comp["n_cells_phenotype"] / comp["n_cells_total"]
comp["lineage"] = comp["phenotype"].astype(str).map(infer_lineage_from_phenotype)
comp = comp[sample_keys + ["phenotype", "lineage", "n_cells_phenotype", "n_cells_total", "frac"]]
composition_path = OUTPUT_DIR / "temporal_composition_tcell.csv"
comp.to_csv(composition_path, index=False)
print(f"Composition rows: {len(comp)} (samples kept: {comp[sample_keys].drop_duplicates().shape[0]})")

# %%
# ---- Build pathway definitions ----
def _read_term_family(tsv_path: Path) -> dict:
    """Return {term -> family} from a pathway-families tsv (cols: term, family)."""
    if not tsv_path.exists():
        return {}
    df = pd.read_csv(tsv_path, sep="\t")
    if "term" not in df.columns or "family" not in df.columns:
        return {}
    return dict(zip(df["term"].astype(str), df["family"].astype(str)))


def _read_excluded(tsv_path: Path) -> set:
    """Return {term} from the kegg_excluded tsv (col: term)."""
    if not tsv_path.exists():
        return set()
    df = pd.read_csv(tsv_path, sep="\t")
    if "term" not in df.columns:
        return set()
    return set(df["term"].astype(str))


hallmark_lib = gp.get_library(name="MSigDB_Hallmark_2020")
kegg_lib = gp.get_library(name="KEGG_2021_Human")
print(f"Hallmark library: {len(hallmark_lib)} pathways")
print(f"KEGG library:     {len(kegg_lib)} pathways")

hallmark_families = _read_term_family(HALLMARK_FAMILY_TSV)
kegg_families = _read_term_family(KEGG_FAMILY_TSV)
kegg_excluded = _read_excluded(KEGG_EXCLUDED_TSV)
print(f"KEGG excluded:    {len(kegg_excluded)} pathways")

pathway_records = []
for name, genes in hallmark_lib.items():
    pathway_records.append({
        "pathway": name,
        "source": "hallmark",
        "family": hallmark_families.get(name, "Unmapped"),
        "gene_list": ";".join(sorted(set(genes))),
    })
for name, genes in kegg_lib.items():
    if name in kegg_excluded:
        continue
    pathway_records.append({
        "pathway": name,
        "source": "kegg",
        "family": kegg_families.get(name, "Unmapped"),
        "gene_list": ";".join(sorted(set(genes))),
    })

pathway_def = pd.DataFrame(pathway_records)
pathway_def.to_csv(OUTPUT_DIR / "pathway_definitions.csv", index=False)
n_h = int((pathway_def["source"] == "hallmark").sum())
n_k = int((pathway_def["source"] == "kegg").sum())
print(f"Pathway definitions: {len(pathway_def)} (hallmark={n_h}, kegg={n_k})")

# %%
# ---- Build gene panel ----
print("\nComputing top markers per phenotype...")
sc.tl.rank_genes_groups(adata, "phenotype", method="wilcoxon", use_raw=False)
marker_genes = set()
phenos = [str(p) for p in adata.obs["phenotype"].dropna().unique()]
for pheno in phenos:
    df = sc.get.rank_genes_groups_df(adata, group=pheno)
    marker_genes |= set(df["names"].head(TOP_MARKERS_PER_PHENO))
print(f"Marker genes (union of top {TOP_MARKERS_PER_PHENO} per phenotype): {len(marker_genes)}")

gsea_genes = set()
gsea_files = sorted(GSEA_DIR.glob("gsea_hallmark_*.csv")) + sorted(GSEA_DIR.glob("gsea_kegg_*.csv"))
for f in gsea_files:
    df = pd.read_csv(f)
    if "FDR q-val" not in df.columns or "Lead_genes" not in df.columns:
        continue
    sig = df[df["FDR q-val"] < GSEA_FDR_THRESHOLD]
    for lead in sig["Lead_genes"].dropna():
        for g in str(lead).split(";"):
            g = g.strip()
            if g:
                gsea_genes.add(g)
print(f"Leading-edge genes (FDR<{GSEA_FDR_THRESHOLD}, {len(gsea_files)} files): {len(gsea_genes)}")

panel = sorted((marker_genes | gsea_genes) & set(adata.var_names))
print(f"Final gene panel (intersected with adata.var_names): {len(panel)}")

(OUTPUT_DIR / "gene_panel_tcell.txt").write_text("\n".join(panel) + "\n")

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
    genes = [g for g in row["gene_list"].split(";") if g in gene_set]
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
    grouped["lineage"] = grouped["phenotype"].astype(str).map(infer_lineage_from_phenotype)
    score_records.append(grouped)
    n_scored += 1
    if n_scored % 25 == 0:
        print(f"  scored {n_scored} pathways...")

if "_score" in adata.obs.columns:
    del adata.obs["_score"]

if score_records:
    pathway_scores = pd.concat(score_records, ignore_index=True)
else:
    pathway_scores = pd.DataFrame(columns=GROUP_KEYS + ["mean_score", "n_cells", "pathway", "source", "lineage"])

pathway_scores_path = OUTPUT_DIR / "temporal_pathway_scores_tcell.csv"
pathway_scores.to_csv(pathway_scores_path, index=False)
print(f"Pathways scored: {n_scored}, skipped (<{MIN_PATHWAY_GENES} genes present): {n_skipped}")
print(f"Pathway score rows: {len(pathway_scores)}")

# %%
# ---- Gene expression aggregation (chunked, incremental write) ----
print(f"\nAggregating gene expression for {len(panel)} genes in chunks of {GENE_CHUNK}...")
gene_expr_path = OUTPUT_DIR / "temporal_gene_expression_tcell.csv"
if gene_expr_path.exists():
    gene_expr_path.unlink()

# Pre-compute per-group cell indices once (reused across chunks)
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
lineages = np.array([infer_lineage_from_phenotype(p) for p in phenotypes])

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
print("DONE")
print("=" * 60)
print(f"  pathway_definitions.csv            : {len(pathway_def)} pathways")
print(f"  gene_panel_tcell.txt               : {len(panel)} genes")
print(f"  temporal_composition_tcell.csv     : {len(comp)} rows")
print(f"  temporal_pathway_scores_tcell.csv  : {len(pathway_scores)} rows ({n_scored} pathways scored)")
print(f"  temporal_gene_expression_tcell.csv : {n_rows_written} rows")
try:
    du = subprocess.check_output(["du", "-sh", str(OUTPUT_DIR)]).decode().strip()
    print(f"  output dir: {du}")
except Exception as e:
    print(f"  (du failed: {e})")
