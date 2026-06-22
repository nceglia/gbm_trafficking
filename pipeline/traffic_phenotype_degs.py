# %%
"""Per-phenotype one-vs-rest DEGs, globally and stratified by tissue /
timepoint. Patients are the unit of replication: pseudobulk by
(patient, phenotype) within each stratum, then DESeq2 with
``~ patient + is_target`` and contrast ``["is_target","1","0"]`` so
positive log2FC means up in the target phenotype.

    python pipeline/traffic_phenotype_degs.py --dry-run   # CD4 global Treg only
    python pipeline/traffic_phenotype_degs.py             # full sweep
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.differential_expression import run_deseq2
from modules.pseudobulk import pseudobulk_counts_by_group

# %% Config
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true",
                    help="Only run CD4 global Treg contrast, then exit.")
args, _ = parser.parse_known_args()

INPUT_DIR = REPO_ROOT / "results" / "traffic_pseudotime_phenotypes"
OUT_DIR = REPO_ROOT / "results" / "traffic_phenotype_degs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINEAGES = ["CD8", "CD4"]
TISSUES = ["PBMC", "CSF", "TP"]
TIMEPOINTS = ["1", "2", "3", "4", "5", "6"]

MIN_CELLS_PER_PB = 10
MIN_PATIENTS_PER_ARM = 2
SIG_PADJ = 0.05
SIG_LFC = 1.0

LINEAGE_PATHS = {
    "CD8": INPUT_DIR / "CD8" / "CD8_pseudotime.h5ad",
    "CD4": INPUT_DIR / "CD4" / "CD4_pseudotime.h5ad",
}


# %% Helpers
def _build_strata(stratifier):
    if stratifier == "global":
        return [("all", None)]
    if stratifier == "tissue":
        return [(t, ("tissue", t)) for t in TISSUES]
    if stratifier == "timepoint":
        return [(tp, ("timepoint", tp)) for tp in TIMEPOINTS]
    raise ValueError(stratifier)


def _subset(adata, slicer):
    if slicer is None:
        return adata
    key, val = slicer
    mask = adata.obs[key].astype(str) == str(val)
    return adata[mask]


def _pseudobulk_stratum(adata_stratum):
    counts, meta = pseudobulk_counts_by_group(
        adata_stratum, group_key="phenotype",
        patient_key="patient", layer="counts")
    keep = meta["n_cells"] >= MIN_CELLS_PER_PB
    dropped = int((~keep).sum())
    meta = meta.loc[keep].copy()
    counts = counts.loc[meta.index].copy()
    return counts, meta, dropped


def _contrast_one(counts, meta, dropped, target,
                  lineage, stratifier, stratum):
    target_pb = meta["phenotype"].astype(str) == target
    base = {
        "lineage": lineage, "stratifier": stratifier, "stratum": stratum,
        "phenotype": target,
        "n_patients_in": 0, "n_patients_out": 0,
        "n_pseudobulks_dropped": dropped, "n_sig_genes": 0,
    }
    if not target_pb.any():
        return None, {**base, "status": "skipped:no_cells"}
    meta = meta.copy()
    meta["is_target"] = np.where(target_pb, "1", "0")
    n_in = int(meta.loc[meta["is_target"] == "1", "patient"].nunique())
    n_out = int(meta.loc[meta["is_target"] == "0", "patient"].nunique())
    base.update(n_patients_in=n_in, n_patients_out=n_out)
    if n_in < MIN_PATIENTS_PER_ARM or n_out < MIN_PATIENTS_PER_ARM:
        return None, {**base, "status": "skipped:too_few_patients"}
    try:
        res = run_deseq2(counts, meta, "~ patient + is_target",
                         ["is_target", "1", "0"])
    except Exception as e:
        return None, {**base, "status": f"error:{type(e).__name__}"}
    res = (res.reset_index()
              .rename(columns={"index": "gene",
                               "log2FoldChange": "log2FC"}))
    res["lineage"] = lineage
    res["stratifier"] = stratifier
    res["stratum"] = stratum
    res["phenotype"] = target
    n_pb_in = int((meta["is_target"] == "1").sum())
    n_pb_out = int((meta["is_target"] == "0").sum())
    res["n_pseudobulks_in"] = n_pb_in
    res["n_pseudobulks_out"] = n_pb_out
    res = res[["lineage", "stratifier", "stratum", "phenotype", "gene",
               "log2FC", "stat", "pvalue", "padj",
               "n_pseudobulks_in", "n_pseudobulks_out"]]
    n_sig = int(((res["padj"] < SIG_PADJ)
                 & (res["log2FC"].abs() > SIG_LFC)).sum())
    return res, {**base, "status": "ok", "n_sig_genes": n_sig}


def _process_stratum(adata_lin, all_phenos, lineage, stratifier,
                     stratum, slicer):
    sub = _subset(adata_lin, slicer)
    if sub.n_obs == 0:
        return [], [{
            "lineage": lineage, "stratifier": stratifier,
            "stratum": stratum, "phenotype": p,
            "n_patients_in": 0, "n_patients_out": 0,
            "n_pseudobulks_dropped": 0, "n_sig_genes": 0,
            "status": "skipped:no_cells",
        } for p in all_phenos]
    counts, meta, dropped = _pseudobulk_stratum(sub)
    rows, summs = [], []
    for target in all_phenos:
        rec, summ = _contrast_one(counts, meta, dropped, target,
                                   lineage, stratifier, stratum)
        if rec is not None:
            rows.append(rec)
        summs.append(summ)
    return rows, summs


# %% Dry-run gate: CD4 global, target = CD4_Treg
print("=== Dry-run: CD4 global, target=CD4_Treg ===")
cd4 = ad.read_h5ad(str(LINEAGE_PATHS["CD4"]))
print(f"  loaded CD4 h5ad: {cd4.n_obs:,} cells x {cd4.n_vars:,} genes")
_dry_counts, _dry_meta, _dry_drop = _pseudobulk_stratum(cd4)
_dry_res, _dry_summ = _contrast_one(_dry_counts, _dry_meta, _dry_drop,
                                     "CD4_Treg", "CD4", "global", "all")
if _dry_res is None:
    raise RuntimeError(f"Dry-run failed: {_dry_summ['status']}")
print(f"  patients  in={_dry_summ['n_patients_in']}  "
      f"out={_dry_summ['n_patients_out']}")
print(f"  pseudobulks dropped (n_cells<{MIN_CELLS_PER_PB})="
      f"{_dry_summ['n_pseudobulks_dropped']}  "
      f"in={_dry_res['n_pseudobulks_in'].iat[0]}  "
      f"out={_dry_res['n_pseudobulks_out'].iat[0]}")
print(f"  n_sig_genes (padj<{SIG_PADJ}, |log2FC|>{SIG_LFC}): "
      f"{_dry_summ['n_sig_genes']}")
print("\nTop 20 by padj (positive log2FC = up in Treg):")
top20 = (_dry_res.nsmallest(20, "padj")
         [["gene", "log2FC", "stat", "padj"]])
print(top20.to_string(index=False))
EXPECTED = ["FOXP3", "IL2RA", "IKZF2", "CTLA4"]
top50 = set(_dry_res.nsmallest(50, "padj")["gene"])
matched = [g for g in EXPECTED if g in top50]
print(f"\nExpected Treg markers in top-50 by padj: "
      f"{matched} ({len(matched)}/{len(EXPECTED)})")
if len(matched) < 3:
    raise RuntimeError(
        f"Dry-run gate: only {len(matched)}/{len(EXPECTED)} expected Treg "
        "markers in top 50. Contrast direction or design likely wrong.")
print("Dry-run gate OK.\n")
del _dry_counts, _dry_meta, _dry_res

if args.dry_run:
    print("--dry-run set: stopping before full sweep.")
    sys.exit(0)


# %% Full sweep
all_rows, summary_rows = [], []
t_start = time.time()
n_done = 0

for lineage in LINEAGES:
    print(f"\n=== {lineage} ===")
    adata = cd4 if lineage == "CD4" else ad.read_h5ad(
        str(LINEAGE_PATHS[lineage]))
    all_phenos = sorted(set(adata.obs["phenotype"].astype(str).unique()))
    print(f"  {adata.n_obs:,} cells, {len(all_phenos)} phenotypes: "
          f"{all_phenos}")

    for stratifier in ["global", "tissue", "timepoint"]:
        for stratum, slicer in _build_strata(stratifier):
            rows, summs = _process_stratum(
                adata, all_phenos, lineage, stratifier, stratum, slicer)
            all_rows.extend(rows)
            summary_rows.extend(summs)
            n_done += len(summs)
            n_ok = sum(1 for s in summs if s["status"] == "ok")
            print(f"  {stratifier:>9s} / {stratum:>4s} : "
                  f"{n_ok}/{len(summs)} ok  "
                  f"(elapsed {time.time() - t_start:.0f}s)")

    if lineage == "CD4":
        del cd4
    del adata


# %% Write outputs
degs = (pd.concat(all_rows, ignore_index=True) if all_rows
        else pd.DataFrame(columns=[
            "lineage", "stratifier", "stratum", "phenotype", "gene",
            "log2FC", "stat", "pvalue", "padj",
            "n_pseudobulks_in", "n_pseudobulks_out"]))
summary = pd.DataFrame(summary_rows)
degs.to_csv(OUT_DIR / "degs.csv", index=False)
summary.to_csv(OUT_DIR / "summary.csv", index=False)
print(f"\nWrote degs.csv ({len(degs):,} rows) and "
      f"summary.csv ({len(summary)} rows)")


# %% Sanity report
print("\n=== Sanity ===")
print(f"Contrasts attempted: {len(summary)}")
print("Status breakdown:")
print(summary["status"].value_counts().to_string())
n_unique = degs[["lineage", "stratifier", "stratum",
                 "phenotype"]].drop_duplicates().shape[0]
print(f"\nUnique contrasts in degs.csv: {n_unique}")

print("\nSig genes per (lineage, stratifier):")
sig_table = (summary[summary["status"] == "ok"]
             .groupby(["lineage", "stratifier"])["n_sig_genes"]
             .agg(["sum", "count"])
             .rename(columns={"sum": "total_sig",
                              "count": "n_ok_contrasts"})
             .reset_index())
print(sig_table.to_string(index=False))

print(f"\nTotal wall time: {time.time() - t_start:.0f}s")
