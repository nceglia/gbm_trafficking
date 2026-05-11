# %%
"""Memory-bounded intersection of LIANA L-R rows with cross-lineage
pathway candidates.

Processes one tissue at a time, filters LIANA on cellphone_pvals < 0.05
BEFORE any join, restricts ligand/receptor pathway memberships to the
pathways that appear in the per-tissue candidate set, and streams the
raw output CSV to disk in append mode rather than concatenating in
memory.

Reads:
  results/12_liana_signaling/liana_per_timepoint.csv
  results/11_cross_lineage_correlations/pathway_correlations.csv
  results/10_temporal_scores/pathway_definitions.csv

Writes:
  results/12_liana_signaling/signaling_edges_raw.csv
  results/12_liana_signaling/signaling_edges_summary.csv
"""
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "results"

LIANA_CSV = RES / "12_liana_signaling" / "liana_per_timepoint.csv"
PATHWAY_CORRS_CSV = RES / "11_cross_lineage_correlations" / "pathway_correlations.csv"
PATHWAY_DEFS_CSV = RES / "10_temporal_scores" / "pathway_definitions.csv"

OUTPUT_DIR = RES / "12_liana_signaling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUTPUT_DIR / "signaling_edges_raw.csv"
SUMMARY_PATH = OUTPUT_DIR / "signaling_edges_summary.csv"

LIANA_PVAL_CUTOFF = 0.05
PATHWAY_Q_CUTOFF = 0.10
PATHWAY_RHO_CUTOFF = 0.4

t_start = time.time()

RAW_COLS = [
    "tissue", "timepoint", "source", "target",
    "ligand_complex", "receptor_complex",
    "lr_means", "cellphone_pvals",
    "t_phenotype", "m_phenotype", "t_pathway", "m_pathway",
    "rho", "q_bh", "direction",
]
KEEP_LIANA = [
    "lr_id", "timepoint", "source", "target",
    "ligand_complex", "receptor_complex",
    "lr_means", "cellphone_pvals",
]

# %%
# ---- Step 1: Load + filter pathway candidates ----
cands = pd.read_csv(PATHWAY_CORRS_CSV)
n_cands_in = len(cands)
cands = cands[
    (cands["q_bh"] < PATHWAY_Q_CUTOFF)
    & (cands["rho"].abs() > PATHWAY_RHO_CUTOFF)
].reset_index(drop=True)
print(f"Candidates: {n_cands_in} -> {len(cands)} after q<{PATHWAY_Q_CUTOFF} & |rho|>{PATHWAY_RHO_CUTOFF}")
if len(cands) > 200_000:
    print(f"WARNING: {len(cands)} candidates is large; merges may be heavy")

# %%
# ---- Step 2: gene -> pathway long table ----
pdefs = pd.read_csv(PATHWAY_DEFS_CSV)
gene_pw = (
    pdefs.assign(gene=pdefs["gene_list"].str.split(";"))
         .explode("gene")
         .assign(gene=lambda d: d["gene"].str.strip())
         .query("gene != ''")[["gene", "pathway"]]
         .drop_duplicates()
         .reset_index(drop=True)
)
print(f"gene_pw: {len(gene_pw)} (gene, pathway) edges over "
      f"{gene_pw['gene'].nunique()} genes / {gene_pw['pathway'].nunique()} pathways")

# %%
# ---- Step 3: Stream LIANA in tissue chunks ----
liana_full = pd.read_csv(LIANA_CSV)
n_in = len(liana_full)
liana_full = liana_full[liana_full["cellphone_pvals"] < LIANA_PVAL_CUTOFF]
print(f"LIANA: {n_in} -> {len(liana_full)} after p<{LIANA_PVAL_CUTOFF}")

if not len(liana_full) or not len(cands):
    print("Nothing to intersect; writing empty outputs and exiting.")
    pd.DataFrame(columns=RAW_COLS).to_csv(RAW_PATH, index=False)
    pd.DataFrame().to_csv(SUMMARY_PATH, index=False)
    sys.exit(0)

header_written = False
per_tissue_summaries = []
n_raw_total = 0

for tissue in sorted(liana_full["tissue"].unique()):
    liana = liana_full[liana_full["tissue"] == tissue].copy()
    liana["lr_id"] = np.arange(len(liana))
    cands_t = cands[cands["tissue"] == tissue]
    if not len(liana) or not len(cands_t):
        continue

    # Pathways that can possibly participate for this tissue.
    cand_t_paths = set(cands_t["t_pathway"]).union(set(cands_t["m_pathway"]))

    lig = (
        liana[["lr_id", "ligand_complex"]]
        .assign(gene=liana["ligand_complex"].astype(str).str.split("_"))
        .explode("gene")
        .merge(gene_pw, on="gene", how="inner")
        .rename(columns={"pathway": "ligand_pathway"})
        [["lr_id", "ligand_pathway"]]
        .drop_duplicates()
    )
    rec = (
        liana[["lr_id", "receptor_complex"]]
        .assign(gene=liana["receptor_complex"].astype(str).str.split("_"))
        .explode("gene")
        .merge(gene_pw, on="gene", how="inner")
        .rename(columns={"pathway": "receptor_pathway"})
        [["lr_id", "receptor_pathway"]]
        .drop_duplicates()
    )

    lig = lig[lig["ligand_pathway"].isin(cand_t_paths)]
    rec = rec[rec["receptor_pathway"].isin(cand_t_paths)]
    print(f"  {tissue}: {len(liana)} LIANA, {len(cands_t)} cands, "
          f"{len(lig)} lig-pw, {len(rec)} rec-pw")

    if not len(lig) or not len(rec):
        del lig, rec
        continue

    edges_A = (
        lig.merge(cands_t, left_on="ligand_pathway", right_on="t_pathway", how="inner")
           .merge(rec, on="lr_id", how="inner")
           .query("receptor_pathway == m_pathway")
           .assign(direction="ligand_in_t_pathway")
    )
    edges_B = (
        lig.merge(cands_t, left_on="ligand_pathway", right_on="m_pathway", how="inner")
           .merge(rec, on="lr_id", how="inner")
           .query("receptor_pathway == t_pathway")
           .assign(direction="ligand_in_m_pathway")
    )
    edges = pd.concat([edges_A, edges_B], ignore_index=True)
    del edges_A, edges_B, lig, rec

    if not len(edges):
        del edges
        continue

    edges = edges.merge(liana[KEEP_LIANA], on="lr_id", how="inner")
    edges["tissue"] = tissue

    edges[RAW_COLS].to_csv(
        RAW_PATH,
        mode="w" if not header_written else "a",
        header=not header_written,
        index=False,
    )
    header_written = True
    n_raw_total += len(edges)
    print(f"  {tissue}: wrote {len(edges)} raw edges")

    summ = (
        edges.groupby(
            ["source", "target", "ligand_complex", "receptor_complex",
             "t_phenotype", "m_phenotype", "t_pathway", "m_pathway",
             "direction"],
            as_index=False,
        ).agg(
            n_bins_active=("timepoint", "nunique"),
            timepoints=("timepoint", lambda s: ",".join(sorted(map(str, set(s))))),
            mean_lr_means=("lr_means", "mean"),
            min_p=("cellphone_pvals", "min"),
            rho=("rho", "first"),
            q_bh=("q_bh", "first"),
        )
    )
    summ["tissue"] = tissue
    per_tissue_summaries.append(summ)
    del edges, summ, liana, cands_t

# %%
# ---- Step 4: Combine summaries ----
if per_tissue_summaries:
    summary = pd.concat(per_tissue_summaries, ignore_index=True)
else:
    summary = pd.DataFrame()
summary.to_csv(SUMMARY_PATH, index=False)

# Ensure raw file exists even if no tissue produced output.
if not header_written:
    pd.DataFrame(columns=RAW_COLS).to_csv(RAW_PATH, index=False)

# %%
# ---- Step 5: Final stats ----
elapsed = time.time() - t_start
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

# Count raw rows from disk to keep memory low.
try:
    wc = subprocess.run(
        ["wc", "-l", str(RAW_PATH)],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split()
    n_raw_disk = max(0, int(wc[0]) - 1)  # subtract header
except Exception:
    n_raw_disk = n_raw_total

print(f"Raw edges written:       {n_raw_disk}")
print(f"Summary edges:           {len(summary)}")
if len(summary):
    n_unique_lr = summary[["ligand_complex", "receptor_complex"]].drop_duplicates().shape[0]
    n_unique_pwctx = summary[["t_pathway", "m_pathway"]].drop_duplicates().shape[0]
    print(f"Unique L-R pairs:        {n_unique_lr}")
    print(f"Unique pathway contexts: {n_unique_pwctx}")
    top = summary.sort_values("n_bins_active", ascending=False).head(10)
    print("\nTop 10 summary rows by n_bins_active:")
    print(top.to_string(index=False))

try:
    du_out = subprocess.run(
        ["du", "-sh", str(OUTPUT_DIR)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"\nOutput dir: {du_out}")
except Exception as e:
    print(f"\nOutput dir size unavailable: {e}")

print(f"Elapsed: {elapsed:.2f}s")
