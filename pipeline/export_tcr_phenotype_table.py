#!/usr/bin/env python
"""Export a per-T-cell TCR table matched to RNA phenotype.

Joins the phenotyped, doublet-filtered T-cell object
(`GBM_TCR_POS_TCELLS_singlets.h5ad`) back to the original CellRanger VDJ
per-sample contig CSVs, producing one row per T cell with:

  - RNA subset classification (phenotype) + lineage + CD8A/CD8B/CD4 counts
  - patient / timepoint / tissue / sample
  - the alpha (TRA) and beta (TRB) chains, primary + secondary, as parsed
    by scirpy (V/D/J gene calls, CDR3 aa, CDR3 nt)
  - every CellRanger contig for the cell (reconstructed V(D)J nucleotide
    sequence per contig)
  - the CellRanger clonotype id (per-sample raw + a cohort-unique global id)
    and the cohort CDR3b clone key used by the trafficking pipeline

Outputs:
  results/tcr_phenotype_export/tcr_phenotype_matched.csv
  results/tcr_phenotype_export/README.md

Run:  python pipeline/export_tcr_phenotype_table.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))
from modules import paths  # noqa: E402

H5AD = paths.H5AD_TCELLS                      # GBM_TCR_POS_TCELLS_singlets.h5ad
CELLRANGER_DIR = paths.CELLRANGER_DIR
OUT_DIR = REPO_ROOT / "results" / "tcr_phenotype_export"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "tcr_phenotype_matched.csv"

MARKERS = ["CD8A", "CD8B", "CD4"]
# CellRanger annotation segments, 5'->3', that reconstruct the V(D)J region.
SEG_COLS = ["fwr1_nt", "cdr1_nt", "fwr2_nt", "cdr2_nt",
            "fwr3_nt", "cdr3_nt", "fwr4_nt"]
_MISSING = {"", "nan", "None", "none", "NaN", "<NA>"}


def _clean(x) -> str:
    """Normalize a possibly-missing obs cell to a plain string ('' if missing)."""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s in _MISSING else s


def _chain_str(v: str, j: str, cdr3: str, d: str = "") -> str:
    """Compact 'V,(D,)J,CDR3aa' string, e.g. TRAV12-2,TRAJ53,CAVQPTDSGGSNYKLTF."""
    parts = [p for p in (v, d, j, cdr3) if p]
    return ",".join(parts)


# ── 1. Load phenotype obs (backed; no full matrix in memory) ──────────
print(f"Loading {H5AD.name} (backed) ...")
A = ad.read_h5ad(H5AD, backed="r")
obs = A.obs
sample = obs["sample"].astype(str).to_numpy()
cell_id = obs.index.to_numpy()
# obs index is "<barcode>-<sample>"; recover the original 10x barcode.
barcode = np.array([cid[: -(len(s) + 1)] for cid, s in zip(cell_id, sample)])
print(f"  {len(cell_id):,} cells across {len(set(sample))} samples")

df = pd.DataFrame({
    "cell_barcode": barcode,
    "cell_id": cell_id,
    "sample": sample,
    "patient": obs["patient"].astype(str).to_numpy(),
    "timepoint": obs["timepoint"].astype(str).to_numpy(),
    "tissue": obs["tissue"].astype(str).to_numpy(),
    "subset": obs["phenotype"].astype(str).to_numpy(),
    "lineage": obs["lineage"].astype(str).to_numpy(),
    "clone_cdr3b": [_clean(x) for x in obs["trb"]],
    "clone_size": obs["clone_size"].to_numpy(),
})

# ── 2. CD8A / CD8B / CD4 raw counts (chunked column read) ─────────────
print("Reading CD8A/CD8B/CD4 counts ...")
gidx = [A.var_names.get_loc(g) for g in MARKERS]
n = A.n_obs
counts = np.zeros((n, len(MARKERS)), dtype=np.int32)
STEP = 20000
for start in range(0, n, STEP):
    stop = min(start + STEP, n)
    block = A[start:stop].to_memory()
    M = block.layers["counts"]
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    counts[start:stop] = M[:, gidx]
    print(f"  {stop:,}/{n:,}", end="\r")
print()
for i, g in enumerate(MARKERS):
    df[f"{g}_counts"] = counts[:, i]

# ── 3. TCR chains from scirpy IR_* obs columns ────────────────────────
#   IR_VJ_*  = alpha (TRA);  IR_VDJ_* = beta (TRB);  _1 primary, _2 secondary.
print("Assembling chain fields from scirpy IR columns ...")


def _col(name):
    return [_clean(x) for x in obs[name]] if name in obs.columns else [""] * len(obs)


chain_spec = {
    "alpha_1": ("IR_VJ_1",  False),
    "alpha_2": ("IR_VJ_2",  False),
    "beta_1":  ("IR_VDJ_1", True),
    "beta_2":  ("IR_VDJ_2", True),
}
for label, (pre, is_beta) in chain_spec.items():
    v = _col(f"{pre}_v_call")
    j = _col(f"{pre}_j_call")
    d = _col(f"{pre}_d_call") if is_beta else [""] * len(obs)
    aa = _col(f"{pre}_junction_aa")
    nt = _col(f"{pre}_junction")
    df[f"{label}_v_gene"] = v
    if is_beta:
        df[f"{label}_d_gene"] = d
    df[f"{label}_j_gene"] = j
    df[f"{label}_cdr3_aa"] = aa
    df[f"{label}_cdr3_nt"] = nt
    # Compact "TRAV..,TRAJ..,CDR3aa" string matching the requested example.
    df[f"tcr_{label}"] = [_chain_str(vv, jj, cc, dd)
                          for vv, jj, cc, dd in zip(v, j, aa, d)]

# ── 4. CellRanger clonotype id + all contigs, per (sample, barcode) ───
print("Indexing CellRanger contig CSVs ...")
# (sample, barcode) -> dict(clonotype, contig_ids[], chains[], nts[])
clono_by_key: dict[tuple[str, str], str] = {}
contigs_by_key: dict[tuple[str, str], list[tuple[str, str, float, str]]] = {}

need = pd.DataFrame({"sample": sample, "barcode": barcode}).drop_duplicates()
need_by_sample = {s: set(g["barcode"]) for s, g in need.groupby("sample")}

for s, wanted in need_by_sample.items():
    csv = CELLRANGER_DIR / f"{s}.csv"
    cols = ["barcode", "contig_id", "chain", "umis", "raw_clonotype_id"] + SEG_COLS
    c = pd.read_csv(csv, usecols=lambda x: x in cols)
    c = c[c["barcode"].isin(wanted)]
    for bc, grp in c.groupby("barcode"):
        key = (s, bc)
        ct = grp["raw_clonotype_id"].dropna()
        # CSF libraries were run through `cellranger multi`, which prefixes
        # clonotype ids with its internal sample label ("sample0_"); strip it
        # so the id is uniform cohort-wide. NB: these ids are per-sample, so
        # `clonotype_id_global` (sample-prefixed) is the only unique key.
        raw = ct.iloc[0] if len(ct) else ""
        clono_by_key[key] = re.sub(r"^sample\d+_", "", str(raw)) if raw else ""
        rows = []
        for _, r in grp.iterrows():
            seq = "".join(str(r[col]) for col in SEG_COLS
                          if col in r and pd.notna(r[col]))
            rows.append((str(r["contig_id"]), str(r.get("chain", "")),
                         float(r.get("umis", 0) or 0), seq))
        # order: TRA before TRB, then by UMI support desc
        rows.sort(key=lambda t: (0 if t[1] == "TRA" else 1 if t[1] == "TRB" else 2,
                                 -t[2]))
        contigs_by_key[key] = rows
    print(f"  {s}", end="\r")
print()

keys = list(zip(df["sample"], df["cell_barcode"]))
df["cellranger_clonotype_id"] = [clono_by_key.get(k, "") for k in keys]
df["clonotype_id_global"] = [f"{s}::{c}" if c else "" for s, c in
                             zip(df["sample"], df["cellranger_clonotype_id"])]

max_contigs = max((len(contigs_by_key.get(k, [])) for k in keys), default=0)
print(f"  max contigs per cell: {max_contigs}")
df["n_contigs"] = [len(contigs_by_key.get(k, [])) for k in keys]
df["contig_ids"] = [";".join(t[0] for t in contigs_by_key.get(k, [])) for k in keys]
df["contig_chains"] = [";".join(t[1] for t in contigs_by_key.get(k, [])) for k in keys]
for i in range(max_contigs):
    df[f"contig_{i+1}_nt"] = [
        (contigs_by_key.get(k, [])[i][3] if len(contigs_by_key.get(k, [])) > i else "")
        for k in keys
    ]

# ── 5. Write ──────────────────────────────────────────────────────────
# Column order: identity -> classification -> clonotype -> chains -> contigs
front = ["cell_barcode", "cell_id", "sample", "patient", "timepoint", "tissue",
         "subset", "lineage", "CD8A_counts", "CD8B_counts", "CD4_counts",
         "cellranger_clonotype_id", "clonotype_id_global", "clone_cdr3b",
         "clone_size",
         "tcr_alpha_1", "tcr_alpha_2", "tcr_beta_1", "tcr_beta_2"]
rest = [c for c in df.columns if c not in front]
df = df[front + rest]
df.to_csv(OUT_CSV, index=False)
print(f"\nWrote {OUT_CSV}  ({df.shape[0]:,} cells x {df.shape[1]} cols)")
print(df[["subset", "patient", "timepoint", "tcr_alpha_1", "tcr_beta_1",
          "cellranger_clonotype_id", "n_contigs"]].head(6).to_string(index=False))
