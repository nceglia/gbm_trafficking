"""Memory-bounded, streaming replacement for the signaling_edges.csv filter
in signaling_liana_pathways.py.

Strategy: build a per-(tissue, t_pathway) -> list of (m_pathway, t_phenotype,
m_phenotype, q_bh) dict once, then stream over L-R rows. For each row, walk its
ligand_pathways and look up candidates with that t_pathway in this tissue; for
each candidate, test m_pathway in the receptor_pathways set. Emit edges to
CSV in append mode.
"""
import csv
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
OUT_PATH = paths.LIANA_SIGNALING_DIR / "signaling_edges.csv"

Q_THRESHOLD = 0.25
# Per signaling_intersect_pathways.py: only keep L-R rows that pass the
# cellphonedb permutation test. Without this filter the unfiltered cartesian
# pathway × L-R join blows past 60 GB and isn't a useful artifact anyway.
LIANA_PVAL_CUTOFF = 0.05

OUT_COLS = [
    "tissue", "timepoint", "source_phenotype", "target_phenotype",
    "ligand_complex", "receptor_complex", "lr_means", "cellphone_pvals",
    "candidate_t_phenotype", "candidate_m_phenotype",
    "candidate_t_pathway", "candidate_m_pathway", "candidate_q_bh", "direction",
]

print(f"Reading {PATHWAY_CORRS_PATH} ...", flush=True)
candidates = pd.read_csv(PATHWAY_CORRS_PATH)
candidates = candidates[candidates["q_bh"] < Q_THRESHOLD]
print(f"  {len(candidates):,} pathway correlations pass q_bh < {Q_THRESHOLD}",
      flush=True)

# Per-tissue indices.
# idx_t[tissue][t_pathway] = list of (m_pathway, t_phenotype, m_phenotype, q_bh)
# idx_m[tissue][m_pathway] = list of (t_pathway, t_phenotype, m_phenotype, q_bh)
print("Building per-tissue candidate indices ...", flush=True)
idx_t = defaultdict(lambda: defaultdict(list))
idx_m = defaultdict(lambda: defaultdict(list))
for tissue, tp, mp, tph, mph, qbh in zip(
    candidates["tissue"], candidates["t_pathway"], candidates["m_pathway"],
    candidates["t_phenotype"], candidates["m_phenotype"], candidates["q_bh"],
):
    idx_t[tissue][tp].append((mp, tph, mph, qbh))
    idx_m[tissue][mp].append((tp, tph, mph, qbh))
print(f"  Built indices: "
      f"{ {t: len(v) for t, v in idx_t.items()} } unique t_pathways per tissue",
      flush=True)

# Free the candidates frame.
del candidates

print(f"Streaming {LIANA_CSV} and writing {OUT_PATH} ...", flush=True)
t_start = time.time()

with open(LIANA_CSV) as fin, open(OUT_PATH, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=OUT_COLS)
    writer.writeheader()

    total_rows_in = 0
    total_edges = 0
    last_report = time.time()
    for row in reader:
        total_rows_in += 1
        tissue = row["tissue"]
        if tissue not in idx_t:
            continue
        pv = row.get("cellphone_pvals", "")
        try:
            if pv == "" or float(pv) >= LIANA_PVAL_CUTOFF:
                continue
        except (TypeError, ValueError):
            continue
        t_index = idx_t[tissue]
        m_index = idx_m[tissue]

        lp_str = row.get("ligand_pathways", "") or ""
        rp_str = row.get("receptor_pathways", "") or ""
        if not lp_str and not rp_str:
            continue
        lp_set = set(lp_str.split(";")) if lp_str else set()
        rp_set = set(rp_str.split(";")) if rp_str else set()
        lp_set.discard("")
        rp_set.discard("")

        base = {
            "tissue": tissue,
            "timepoint": row["timepoint"],
            "source_phenotype": row["source"],
            "target_phenotype": row["target"],
            "ligand_complex": row["ligand_complex"],
            "receptor_complex": row["receptor_complex"],
            "lr_means": row.get("lr_means", ""),
            "cellphone_pvals": row.get("cellphone_pvals", ""),
        }

        # Direction 1: t_pathway in ligand_pathways, m_pathway in receptor_pathways.
        for lp in lp_set:
            for mp, tph, mph, qbh in t_index.get(lp, ()):
                if mp in rp_set:
                    writer.writerow({
                        **base,
                        "candidate_t_phenotype": tph,
                        "candidate_m_phenotype": mph,
                        "candidate_t_pathway": lp,
                        "candidate_m_pathway": mp,
                        "candidate_q_bh": qbh,
                        "direction": "ligand_in_t_pathway",
                    })
                    total_edges += 1

        # Direction 2: t_pathway in receptor_pathways, m_pathway in ligand_pathways.
        for rp in rp_set:
            for tp_cand, tph, mph, qbh in m_index.get(rp, ()):
                # rp is m_pathway side here — we want t_pathway in receptor.
                pass  # placeholder; direction 2 logic below
        for rp in rp_set:
            for mp_cand_t, tph, mph, qbh in t_index.get(rp, ()):
                # Here rp matches a t_pathway. We need m_pathway in ligand_pathways.
                if mp_cand_t in lp_set:
                    writer.writerow({
                        **base,
                        "candidate_t_phenotype": tph,
                        "candidate_m_phenotype": mph,
                        "candidate_t_pathway": rp,
                        "candidate_m_pathway": mp_cand_t,
                        "candidate_q_bh": qbh,
                        "direction": "ligand_in_m_pathway",
                    })
                    total_edges += 1

        now = time.time()
        if now - last_report >= 15:
            elapsed = now - t_start
            rate = total_rows_in / elapsed if elapsed else 0
            print(f"  {total_rows_in:,} L-R rows scanned, "
                  f"{total_edges:,} edges written, "
                  f"{rate:.0f} rows/s", flush=True)
            last_report = now

print(f"\nDone. {total_rows_in:,} L-R rows scanned, "
      f"{total_edges:,} signaling edges written to {OUT_PATH}", flush=True)
