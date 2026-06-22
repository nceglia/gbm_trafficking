"""Streaming, memory-bounded replacement for signaling_intersect_pathways.py.

Processes one tissue at a time across three passes over liana_per_timepoint.csv:
for each tissue, builds a summary dict in memory, writes that tissue's rows to
signaling_edges_summary.csv (append mode), then frees the dict before the next
tissue. Bounds memory to a single tissue's group set rather than the union.

A header-only signaling_edges_raw.csv is also written for downstream probes;
the full per-edge raw table would be tens of GB on this data and is not in the
manifest contract for this script.
"""
import csv
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402

LIANA_CSV = paths.LIANA_SIGNALING_DIR / "liana_per_timepoint.csv"
PATHWAY_CORRS_PATH = paths.CROSS_LINEAGE_CORR_DIR / "pathway_correlations.csv"
RAW_PATH = paths.LIANA_SIGNALING_DIR / "signaling_edges_raw.csv"
SUMMARY_PATH = paths.LIANA_SIGNALING_DIR / "signaling_edges_summary.csv"

LIANA_PVAL_CUTOFF = 0.01
Q_THRESHOLD = 0.05
RHO_MAGNITUDE = 0.9

SUMMARY_COLS = [
    "source", "target", "ligand_complex", "receptor_complex",
    "t_phenotype", "m_phenotype", "t_pathway", "m_pathway", "direction",
    "n_bins_active", "timepoints", "mean_lr_means", "min_p", "rho", "q_bh",
    "tissue",
]
RAW_COLS = [
    "tissue", "timepoint", "source", "target",
    "ligand_complex", "receptor_complex",
    "lr_means", "cellphone_pvals",
    "t_phenotype", "m_phenotype", "t_pathway", "m_pathway",
    "rho", "q_bh", "direction",
]

print(f"Reading {PATHWAY_CORRS_PATH} ...", flush=True)
candidates = pd.read_csv(PATHWAY_CORRS_PATH)
n_cands_total = len(candidates)
candidates = candidates[
    (candidates["q_bh"] < Q_THRESHOLD)
    & (candidates["rho"].abs() > RHO_MAGNITUDE)
]
print(f"  {n_cands_total:,} -> {len(candidates):,} after q<{Q_THRESHOLD} "
      f"& |rho|>{RHO_MAGNITUDE}", flush=True)

# Per-tissue indices.
print("Building per-tissue candidate indices ...", flush=True)
idx_t_all = defaultdict(lambda: defaultdict(list))
for tissue, tp, mp, tph, mph, rho, qbh in zip(
    candidates["tissue"], candidates["t_pathway"], candidates["m_pathway"],
    candidates["t_phenotype"], candidates["m_phenotype"],
    candidates["rho"], candidates["q_bh"],
):
    idx_t_all[tissue][tp].append((mp, tph, mph, rho, qbh))
tissues = sorted(idx_t_all.keys())
print(f"  Per-tissue unique t_pathways: "
      f"{ {t: len(v) for t, v in idx_t_all.items()} }", flush=True)
del candidates

# Initialize summary CSV with header; we'll append per-tissue blocks.
with open(SUMMARY_PATH, "w", newline="") as fout:
    csv.writer(fout).writerow(SUMMARY_COLS)
# Header-only raw placeholder for downstream code that probes for the file.
with open(RAW_PATH, "w", newline="") as fout:
    csv.writer(fout).writerow(RAW_COLS)

total_rows_in_all = 0
total_edges_all = 0
total_groups_all = 0
t_start = time.time()

for tissue in tissues:
    t_index = idx_t_all[tissue]
    summary = {}
    print(f"\n[{tissue}] streaming ...", flush=True)
    t_tissue = time.time()
    last_report = time.time()
    rows_in = 0
    rows_passing = 0
    edges_for_tissue = 0

    with open(LIANA_CSV) as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            rows_in += 1
            if row["tissue"] != tissue:
                continue
            try:
                pv = float(row["cellphone_pvals"])
            except (TypeError, ValueError):
                continue
            if pv >= LIANA_PVAL_CUTOFF:
                continue

            lp_str = row.get("ligand_pathways", "") or ""
            rp_str = row.get("receptor_pathways", "") or ""
            if not lp_str and not rp_str:
                continue
            lp_set = set(lp_str.split(";")) if lp_str else set()
            rp_set = set(rp_str.split(";")) if rp_str else set()
            lp_set.discard("")
            rp_set.discard("")

            try:
                lr_mean = float(row.get("lr_means", "") or 0.0)
            except (TypeError, ValueError):
                lr_mean = 0.0

            rows_passing += 1
            timepoint = row["timepoint"]
            source = row["source"]
            target = row["target"]
            ligand_complex = row["ligand_complex"]
            receptor_complex = row["receptor_complex"]

            # Direction 1: t_pathway in ligand, m_pathway in receptor.
            for lp in lp_set:
                for mp, tph, mph, rho, qbh in t_index.get(lp, ()):
                    if mp in rp_set:
                        edges_for_tissue += 1
                        key = (source, target, ligand_complex,
                               receptor_complex, tph, mph, lp, mp,
                               "ligand_in_t_pathway")
                        agg = summary.get(key)
                        if agg is None:
                            summary[key] = [{timepoint}, lr_mean, 1, pv, rho, qbh]
                        else:
                            agg[0].add(timepoint)
                            agg[1] += lr_mean
                            agg[2] += 1
                            if pv < agg[3]:
                                agg[3] = pv

            # Direction 2: t_pathway in receptor, m_pathway in ligand.
            for rp in rp_set:
                for mp, tph, mph, rho, qbh in t_index.get(rp, ()):
                    if mp in lp_set:
                        edges_for_tissue += 1
                        key = (source, target, ligand_complex,
                               receptor_complex, tph, mph, rp, mp,
                               "ligand_in_m_pathway")
                        agg = summary.get(key)
                        if agg is None:
                            summary[key] = [{timepoint}, lr_mean, 1, pv, rho, qbh]
                        else:
                            agg[0].add(timepoint)
                            agg[1] += lr_mean
                            agg[2] += 1
                            if pv < agg[3]:
                                agg[3] = pv

            now = time.time()
            if now - last_report >= 15:
                el = now - t_tissue
                rate = rows_in / el if el else 0
                print(f"  [{tissue}] {rows_in:,} read, "
                      f"{rows_passing:,} kept, {edges_for_tissue:,} edges, "
                      f"{len(summary):,} groups, {rate:.0f} rows/s",
                      flush=True)
                last_report = now

    print(f"\n[{tissue}] done streaming: {rows_in:,} read, "
          f"{rows_passing:,} kept, {edges_for_tissue:,} edges, "
          f"{len(summary):,} groups", flush=True)

    # Append this tissue's summary to disk.
    print(f"[{tissue}] writing summary rows ...", flush=True)
    with open(SUMMARY_PATH, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=SUMMARY_COLS)
        for key, agg in summary.items():
            (source, target, ligand_complex, receptor_complex,
             tph, mph, tp, mp, direction) = key
            timepoints, sum_lr_means, count, min_p, rho, qbh = agg
            writer.writerow({
                "source": source,
                "target": target,
                "ligand_complex": ligand_complex,
                "receptor_complex": receptor_complex,
                "t_phenotype": tph,
                "m_phenotype": mph,
                "t_pathway": tp,
                "m_pathway": mp,
                "direction": direction,
                "n_bins_active": len(timepoints),
                "timepoints": ",".join(sorted(map(str, timepoints))),
                "mean_lr_means": sum_lr_means / count,
                "min_p": min_p,
                "rho": rho,
                "q_bh": qbh,
                "tissue": tissue,
            })
    print(f"[{tissue}] wrote {len(summary):,} summary rows", flush=True)
    total_rows_in_all += rows_in
    total_edges_all += edges_for_tissue
    total_groups_all += len(summary)
    del summary
    gc.collect()

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.0f}s. "
      f"{total_rows_in_all:,} L-R rows read (across {len(tissues)} passes), "
      f"{total_edges_all:,} raw edges, {total_groups_all:,} summary rows.",
      flush=True)
