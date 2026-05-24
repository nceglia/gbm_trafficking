# %%
"""Time-aware, branch-membership-restricted L-R differential analysis.

Edge unit = (src tissue i, dst tissue j); 9 edges total including
same-tissue. For each edge, pool branch-participating cells across all
(patient, transition) and run LIANA on src side and dst side
separately. Compare per L-R pair.

Output: results/06b_branch_signaling/
"""
import subprocess
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import leaves_list, linkage
from tqdm import tqdm

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
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.myeloid_groups import MYELOID_GROUPS, regroup_obs
from modules.style import (
    MYELOID_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
)

DATA_PATH = (REPO_ROOT / "data" / "objects"
             / "GBM_TCR_POS_TCELLS_MYELOID_combined.h5ad")
OUT_DIR = REPO_ROOT / "results" / "06b_branch_signaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS_PER_BRANCH_END = 2
LIANA_PVAL_CUTOFF = 0.05
EXPR_PROP = 0.10
TOP_N_PER_EDGE = 8

MIN_TOTAL_CELLS = 100
MIN_PHENOTYPE_CELLS = 20
MIN_PHENOTYPES = 2

LIANA_N_PERMS = 100
LIANA_N_JOBS = -1

# Heatmap aesthetics
HEATMAP_CLIP = 5.0            # symmetric log2fc clip for color only
DOT_P_CUTOFF = 0.01           # side-specific p threshold for dot marker
DOT_LFC_CUTOFF = 1.0          # |log2fc| threshold for dot marker
DOT_SIZE = 8
DEFAULT_PSEUDO = 0.01         # fallback when no positive lr_means in group

TISSUES = ("PBMC", "CSF", "TP")
SAME_TISSUE_EDGES = [("PBMC", "PBMC"), ("CSF", "CSF"), ("TP", "TP")]
CROSS_TISSUE_EDGES = [
    ("PBMC", "CSF"), ("PBMC", "TP"),
    ("CSF", "PBMC"), ("CSF", "TP"),
    ("TP", "PBMC"), ("TP", "CSF"),
]
EDGES = SAME_TISSUE_EDGES + CROSS_TISSUE_EDGES
EDGE_LABELS = [f"{a}→{b}" for a, b in EDGES]

T_PHENOTYPES = set(TCELL_PHENOTYPE_ORDER)
M_PHENOTYPES = set(MYELOID_GROUPS.values())

DPI = 200

t_start = time.time()

# %%
# ---- Step 1: Load + regroup ----
print("Loading combined adata...")
adata = sc.read(str(DATA_PATH))
print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

for col in ("patient", "tissue", "timepoint", "phenotype",
            "trb", "major_lineage"):
    if col not in adata.obs.columns:
        raise KeyError(f"Required obs column missing: {col!r}")

adata.obs["phenotype"] = adata.obs["phenotype"].astype(object)
is_myeloid = adata.obs["phenotype"].isin(MYELOID_GROUPS.keys())
remapped = regroup_obs(adata.obs.loc[is_myeloid])
adata.obs.loc[is_myeloid, "phenotype"] = remapped.values

n_before = adata.n_obs
adata = adata[~adata.obs["phenotype"].isna()].copy()
print(f"Regrouped {int(is_myeloid.sum())} myeloid cells; "
      f"{n_before - adata.n_obs} unmapped dropped; {adata.n_obs} remain")

adata.obs["timepoint"] = adata.obs["timepoint"].astype(int)
adata.obs["phenotype"] = adata.obs["phenotype"].astype(str)
adata.obs["is_t"] = adata.obs["major_lineage"].astype(str) == "T"
adata.obs["is_myeloid"] = adata.obs["major_lineage"].astype(str) == "Myeloid"

# %%
# ---- Step 2: Enumerate branches (T-cell side) ----
print("\nEnumerating branches...")
t_obs = adata.obs[adata.obs["is_t"]
                  & adata.obs["trb"].notna()].copy()
t_obs["timepoint"] = t_obs["timepoint"].astype(int)

bin_counts = (t_obs.groupby(
    ["trb", "patient", "tissue", "timepoint"], observed=True)
    .size().rename("n").reset_index())
bin_counts = bin_counts[bin_counts["n"] >= MIN_CELLS_PER_BRANCH_END].copy()

branch_records = []
for (trb, patient), grp in bin_counts.groupby(["trb", "patient"], observed=True):
    rows = grp[["tissue", "timepoint", "n"]].to_dict("records")
    for r_src in rows:
        for r_dst in rows:
            if r_dst["timepoint"] != r_src["timepoint"] + 1:
                continue
            branch_records.append({
                "trb": trb,
                "patient": patient,
                "src_tissue": r_src["tissue"],
                "src_t": int(r_src["timepoint"]),
                "dst_tissue": r_dst["tissue"],
                "dst_t": int(r_dst["timepoint"]),
                "n_src": int(r_src["n"]),
                "n_dst": int(r_dst["n"]),
            })
branches = pd.DataFrame(branch_records)
branches.to_csv(OUT_DIR / "branches.csv", index=False)
print(f"  n_branches: {len(branches)}")

n_branches_per_edge = {edge: 0 for edge in EDGES}
for edge in EDGES:
    i, j = edge
    n_branches_per_edge[edge] = int(
        ((branches["src_tissue"] == i)
         & (branches["dst_tissue"] == j)).sum()
    ) if len(branches) else 0

# %%
# ---- Step 3: Build per-edge src and dst AnnData subsets ----
print("\nBuilding per-edge src/dst pools...")
edge_subsets = {}  # edge -> {"src": adata_sub, "dst": adata_sub}

myeloid_cells = adata[adata.obs["is_myeloid"]]
m_keys = pd.MultiIndex.from_arrays([
    myeloid_cells.obs["patient"].values,
    myeloid_cells.obs["tissue"].values,
    myeloid_cells.obs["timepoint"].astype(int).values,
])
myeloid_by_key = {}
for k, idx in zip(m_keys, np.arange(myeloid_cells.n_obs)):
    myeloid_by_key.setdefault(k, []).append(idx)

t_cells_all = adata[adata.obs["is_t"]]
t_obs_all = t_cells_all.obs.copy()
t_obs_all["timepoint"] = t_obs_all["timepoint"].astype(int)
t_obs_all["_idx"] = np.arange(t_cells_all.n_obs)
t_groups = t_obs_all.groupby(
    ["patient", "tissue", "timepoint"], observed=True)

n_t_cells_edge = {edge: {"src": 0, "dst": 0} for edge in EDGES}
n_m_cells_edge = {edge: {"src": 0, "dst": 0} for edge in EDGES}

for edge in EDGES:
    i, j = edge
    bsub = branches[(branches["src_tissue"] == i)
                    & (branches["dst_tissue"] == j)] if len(branches) else \
           pd.DataFrame(columns=branches.columns if len(branches.columns) else [])
    if not len(bsub):
        edge_subsets[edge] = {"src": None, "dst": None}
        continue

    src_t_idx = []
    src_m_idx = []
    seen_src_samples = set()
    for (patient, t), pgrp in bsub.groupby(["patient", "src_t"], observed=True):
        clone_set = set(pgrp["trb"].unique())
        key = (patient, i, int(t))
        if key in t_groups.indices:
            t_sub = t_obs_all.iloc[t_groups.indices[key]]
            mask = t_sub["trb"].isin(clone_set)
            src_t_idx.extend(t_sub.loc[mask, "_idx"].tolist())
        if key not in seen_src_samples:
            seen_src_samples.add(key)
            if key in myeloid_by_key:
                src_m_idx.extend(myeloid_by_key[key])

    dst_t_idx = []
    dst_m_idx = []
    seen_dst_samples = set()
    for (patient, t), pgrp in bsub.groupby(["patient", "dst_t"], observed=True):
        clone_set = set(pgrp["trb"].unique())
        key = (patient, j, int(t))
        if key in t_groups.indices:
            t_sub = t_obs_all.iloc[t_groups.indices[key]]
            mask = t_sub["trb"].isin(clone_set)
            dst_t_idx.extend(t_sub.loc[mask, "_idx"].tolist())
        if key not in seen_dst_samples:
            seen_dst_samples.add(key)
            if key in myeloid_by_key:
                dst_m_idx.extend(myeloid_by_key[key])

    src_t_idx = np.unique(src_t_idx).astype(int)
    src_m_idx = np.unique(src_m_idx).astype(int)
    dst_t_idx = np.unique(dst_t_idx).astype(int)
    dst_m_idx = np.unique(dst_m_idx).astype(int)

    n_t_cells_edge[edge]["src"] = int(len(src_t_idx))
    n_t_cells_edge[edge]["dst"] = int(len(dst_t_idx))
    n_m_cells_edge[edge]["src"] = int(len(src_m_idx))
    n_m_cells_edge[edge]["dst"] = int(len(dst_m_idx))

    if len(src_t_idx) + len(src_m_idx) > 0:
        src_t_ad = t_cells_all[src_t_idx] if len(src_t_idx) else None
        src_m_ad = myeloid_cells[src_m_idx] if len(src_m_idx) else None
        if src_t_ad is not None and src_m_ad is not None:
            src_ad = ad.concat([src_t_ad, src_m_ad], join="outer")
        elif src_t_ad is not None:
            src_ad = src_t_ad.copy()
        else:
            src_ad = src_m_ad.copy()
        src_ad.obs["phenotype"] = src_ad.obs["phenotype"].astype(str)
    else:
        src_ad = None

    if len(dst_t_idx) + len(dst_m_idx) > 0:
        dst_t_ad = t_cells_all[dst_t_idx] if len(dst_t_idx) else None
        dst_m_ad = myeloid_cells[dst_m_idx] if len(dst_m_idx) else None
        if dst_t_ad is not None and dst_m_ad is not None:
            dst_ad = ad.concat([dst_t_ad, dst_m_ad], join="outer")
        elif dst_t_ad is not None:
            dst_ad = dst_t_ad.copy()
        else:
            dst_ad = dst_m_ad.copy()
        dst_ad.obs["phenotype"] = dst_ad.obs["phenotype"].astype(str)
    else:
        dst_ad = None

    edge_subsets[edge] = {"src": src_ad, "dst": dst_ad}
    print(f"  {i}->{j}: src T={len(src_t_idx)}, M={len(src_m_idx)}; "
          f"dst T={len(dst_t_idx)}, M={len(dst_m_idx)}")

# %%
# ---- Step 4a: Pre-flight (gate filtering, no LIANA yet) ----
print("\nPre-flight: filtering (edge, side) pools against LIANA gates...")
liana_rows = []
skip_log = []
n_liana_rows_edge = {edge: {"src": 0, "dst": 0} for edge in EDGES}

prepared = {}        # (edge, side) -> filtered AnnData or None
preflight_rows = []


def _prepare_side(sub, edge, side):
    if sub is None or sub.n_obs < MIN_TOTAL_CELLS:
        skip_log.append({"edge": f"{edge[0]}->{edge[1]}", "side": side,
                         "reason": "fewer_than_100_total_cells",
                         "n_cells": 0 if sub is None else int(sub.n_obs)})
        return None, 0
    counts = sub.obs["phenotype"].value_counts()
    n_phen = int((counts >= MIN_PHENOTYPE_CELLS).sum())
    keep_phenos = counts[counts >= MIN_PHENOTYPE_CELLS].index.tolist()
    if len(keep_phenos) < MIN_PHENOTYPES:
        skip_log.append({"edge": f"{edge[0]}->{edge[1]}", "side": side,
                         "reason": "fewer_than_2_phenotypes_with_20_cells",
                         "n_cells": int(sub.n_obs)})
        return None, n_phen
    filtered = sub[sub.obs["phenotype"].isin(keep_phenos)].copy()
    filtered.obs["phenotype"] = filtered.obs["phenotype"].astype(str)
    return filtered, n_phen


for edge in EDGES:
    i, j = edge
    for side in ("src", "dst"):
        sub = edge_subsets[edge][side]
        filtered, n_phen = _prepare_side(sub, edge, side)
        prepared[(edge, side)] = filtered
        preflight_rows.append({
            "edge": f"{i}->{j}",
            "side": side,
            "n_t_cells": n_t_cells_edge[edge][side],
            "n_myeloid_cells": n_m_cells_edge[edge][side],
            "n_phenotypes_ge20": n_phen,
            "n_cells_after_gate": int(filtered.n_obs) if filtered is not None else 0,
            "will_run": filtered is not None,
        })

preflight_df = pd.DataFrame(preflight_rows)
preflight_df.to_csv(OUT_DIR / "preflight.csv", index=False)

total_cells_all_sides = int(
    (preflight_df["n_t_cells"] + preflight_df["n_myeloid_cells"]).sum()
)
total_cells_run = int(preflight_df.loc[preflight_df["will_run"],
                                        "n_cells_after_gate"].sum())
n_sides_run = int(preflight_df["will_run"].sum())
n_sides_skip = int((~preflight_df["will_run"]).sum())

print(f"\n  {'edge':<14}{'side':>5}{'n_T':>8}{'n_M':>8}{'n_phen20':>10}"
      f"{'gated':>10}{'run':>6}")
for r in preflight_rows:
    print(f"  {r['edge']:<14}{r['side']:>5}{r['n_t_cells']:>8}"
          f"{r['n_myeloid_cells']:>8}{r['n_phenotypes_ge20']:>10}"
          f"{r['n_cells_after_gate']:>10}{('Y' if r['will_run'] else '.'):>6}")
print(f"\nSides to run: {n_sides_run}; skip: {n_sides_skip}; "
      f"total cells (all sides): {total_cells_all_sides}; "
      f"total cells (gated runs): {total_cells_run}")

try:
    _res = li.rs.select_resource("consensus")
    n_lr_consensus = int(_res.drop_duplicates(
        subset=["ligand_complex", "receptor_complex"]).shape[0]) \
        if hasattr(_res, "shape") else int(len(_res))
except Exception:
    n_lr_consensus = None
if n_lr_consensus is not None:
    print(f"Consensus resource: ~{n_lr_consensus} unique L-R pairs")
    avg_phen = float(preflight_df.loc[preflight_df["will_run"],
                                       "n_phenotypes_ge20"].mean() or 0.0)
    est_pair_rows = n_lr_consensus * (avg_phen * avg_phen)
    print(f"Estimated upper-bound output rows: ~{int(est_pair_rows * n_sides_run):,} "
          f"(n_lr × phen² × n_sides; avg phen={avg_phen:.1f})")

# %%
# ---- Step 4b: Calibration + tqdm loop ----
sides_to_run = [(edge, side, prepared[(edge, side)])
                 for edge in EDGES for side in ("src", "dst")
                 if prepared[(edge, side)] is not None]


def _liana_call(sub):
    li.mt.cellphonedb(
        sub, groupby="phenotype", resource_name="consensus",
        expr_prop=EXPR_PROP,
        n_perms=LIANA_N_PERMS, n_jobs=LIANA_N_JOBS,
        verbose=False, use_raw=False,
    )
    return sub.uns["liana_res"].copy()


def _tag_and_collect(df, edge, side):
    i, j = edge
    df = df.copy()
    df["side"] = side
    df["edge"] = f"{i}->{j}"
    df["src_tissue"] = i
    df["dst_tissue"] = j
    df["n_t_cells"] = n_t_cells_edge[edge][side]
    df["n_myeloid_cells"] = n_m_cells_edge[edge][side]
    liana_rows.append(df)
    n_liana_rows_edge[edge][side] = int(len(df))


if not sides_to_run:
    print("\nNo (edge, side) pools pass the gates; skipping LIANA.")
else:
    sides_to_run.sort(key=lambda x: x[2].n_obs)
    cal_edge, cal_side, cal_sub = sides_to_run[0]
    cal_n = int(cal_sub.n_obs)
    print(f"\nCalibration: {cal_edge[0]}->{cal_edge[1]} {cal_side} "
          f"(n={cal_n} cells)")
    t_cal = time.time()
    try:
        cal_df = _liana_call(cal_sub)
        cal_secs = time.time() - t_cal
        _tag_and_collect(cal_df, cal_edge, cal_side)
    except Exception as e:
        cal_secs = None
        skip_log.append({"edge": f"{cal_edge[0]}->{cal_edge[1]}",
                         "side": cal_side,
                         "reason": f"liana_failed: {type(e).__name__}: {e}",
                         "n_cells": cal_n})
        print(f"  Calibration FAILED: {e}")

    others = sides_to_run[1:]
    others_n = sum(int(s.n_obs) for _, _, s in others)
    if cal_secs and cal_n > 0 and len(others):
        est_total_secs = cal_secs * others_n / cal_n
        print(f"Calibration: {cal_secs:.1f}s on {cal_n} cells. "
              f"Estimated total: {est_total_secs / 60:.1f} min over "
              f"{len(others)} remaining runs.")
    elif cal_secs:
        print(f"Calibration: {cal_secs:.1f}s on {cal_n} cells. "
              f"No remaining runs.")

    for edge, side, sub in tqdm(others, desc="LIANA (edge, side)", unit="run"):
        i, j = edge
        n = int(sub.n_obs)
        t0 = time.time()
        try:
            df = _liana_call(sub)
        except Exception as e:
            skip_log.append({"edge": f"{i}->{j}", "side": side,
                             "reason": f"liana_failed: {type(e).__name__}: {e}",
                             "n_cells": n})
            tqdm.write(f"  {i}->{j} {side}: FAIL n={n} ({e})")
            continue
        secs = time.time() - t0
        _tag_and_collect(df, edge, side)
        tqdm.write(f"  {i}->{j} {side}: n={n} cells, {secs:.1f}s, "
                   f"{len(df)} L-R rows")

if liana_rows:
    liana_full = pd.concat(liana_rows, ignore_index=True)
else:
    liana_full = pd.DataFrame()

if len(skip_log):
    pd.DataFrame(skip_log).to_csv(OUT_DIR / "skipped_sides.csv", index=False)

# %%
# ---- Step 5: Classify direction ----
def _direction(source, target):
    s_t = source in T_PHENOTYPES
    s_m = source in M_PHENOTYPES
    t_t = target in T_PHENOTYPES
    t_m = target in M_PHENOTYPES
    if s_m and t_t:
        return "M_to_T"
    if s_t and t_m:
        return "T_to_M"
    return "other"


if len(liana_full):
    liana_full["direction"] = [
        _direction(s, t) for s, t in
        zip(liana_full["source"], liana_full["target"])
    ]
    liana_full = liana_full[liana_full["direction"] != "other"].copy()
else:
    liana_full["direction"] = pd.Series(dtype=str)

print(f"\nLIANA rows after direction filter: {len(liana_full)}")

# %%
# ---- Step 6: Pivot to one row per (edge, direction, source, target, L-R) ----
KEY_COLS = ["edge", "src_tissue", "dst_tissue", "direction",
            "source", "target", "ligand_complex", "receptor_complex"]

if len(liana_full):
    pvt_mean = (liana_full.pivot_table(
        index=KEY_COLS, columns="side", values="lr_means",
        aggfunc="first"))
    pvt_p = (liana_full.pivot_table(
        index=KEY_COLS, columns="side", values="cellphone_pvals",
        aggfunc="first"))
    for c in ("src", "dst"):
        if c not in pvt_mean.columns:
            pvt_mean[c] = np.nan
        if c not in pvt_p.columns:
            pvt_p[c] = np.nan
    diff = pd.DataFrame({
        "src_lr_means": pvt_mean["src"],
        "dst_lr_means": pvt_mean["dst"],
        "src_p": pvt_p["src"],
        "dst_p": pvt_p["dst"],
    }).reset_index()

    # Half-min pseudocount per (edge, direction). Avoids the 1e-6 blow-up
    # when one side is exactly zero. Falls back to DEFAULT_PSEUDO if no
    # positive lr_means exist in the group.
    pos_long = pd.concat([
        diff[["edge", "direction", "src_lr_means"]]
            .rename(columns={"src_lr_means": "v"}),
        diff[["edge", "direction", "dst_lr_means"]]
            .rename(columns={"dst_lr_means": "v"}),
    ], ignore_index=True)
    pos_long = pos_long[pos_long["v"].notna() & (pos_long["v"] > 0)]
    pseudo_by_group = (pos_long.groupby(["edge", "direction"])["v"].min()
                                .mul(0.5).to_dict())
    diff["_pseudo"] = [
        pseudo_by_group.get((e, d), DEFAULT_PSEUDO)
        for e, d in zip(diff["edge"], diff["direction"])
    ]
    diff["log2fc"] = np.log2(
        (diff["dst_lr_means"].fillna(0) + diff["_pseudo"])
        / (diff["src_lr_means"].fillna(0) + diff["_pseudo"])
    )
    diff["extreme"] = (
        diff["src_lr_means"].fillna(0).eq(0)
        | diff["dst_lr_means"].fillna(0).eq(0)
    )
    diff = diff.drop(columns=["_pseudo"])
    diff["min_p"] = diff[["src_p", "dst_p"]].min(axis=1, skipna=True)
    sig_mask = (
        (diff["src_p"].fillna(1.0) < LIANA_PVAL_CUTOFF)
        | (diff["dst_p"].fillna(1.0) < LIANA_PVAL_CUTOFF)
    )
    diff = diff[sig_mask].copy()
else:
    diff = pd.DataFrame(columns=KEY_COLS + [
        "src_lr_means", "dst_lr_means", "src_p", "dst_p",
        "log2fc", "extreme", "min_p"])

diff.to_csv(OUT_DIR / "signaling_edge_differential.csv", index=False)
print(f"signaling_edge_differential.csv: {len(diff)} rows")

# %%
# ---- Step 7: Per-edge summary ----
summary_rows = []
for edge in EDGES:
    i, j = edge
    elabel = f"{i}->{j}"
    for direction in ("M_to_T", "T_to_M"):
        sub = diff[(diff["edge"] == elabel)
                   & (diff["direction"] == direction)]
        if not len(sub):
            summary_rows.append({
                "edge": elabel, "direction": direction,
                "n_lr_significant_either": 0,
                "n_dst_up": 0, "n_src_up": 0,
                "rewiring_volume": 0.0,
            })
            continue
        n_dst_up = int(((sub["log2fc"] > 1)
                         & (sub["dst_p"].fillna(1.0) < 0.05)).sum())
        n_src_up = int(((sub["log2fc"] < -1)
                         & (sub["src_p"].fillna(1.0) < 0.05)).sum())
        # Rewiring volume: only non-extreme rows with min(src_p, dst_p) < 0.05.
        # Extreme rows are dominated by the pseudocount and inflate the sum.
        rv_mask = (sub["min_p"].fillna(1.0) < 0.05) & (~sub["extreme"])
        rewiring_volume = float(sub.loc[rv_mask, "log2fc"].abs().sum())
        summary_rows.append({
            "edge": elabel, "direction": direction,
            "n_lr_significant_either": int(len(sub)),
            "n_dst_up": n_dst_up,
            "n_src_up": n_src_up,
            "rewiring_volume": rewiring_volume,
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "signaling_edge_summary.csv", index=False)

# %%
# ---- Step 8: Headline heatmap ----
print("\nBuilding heatmap...")


def _build_dir_matrix(direction):
    sub = diff[diff["direction"] == direction].copy()
    if not len(sub):
        return None, None, None
    # Exclude extreme one-sided rows from the heatmap row pool. They live
    # in the CSV but distort row ranking and color scale.
    sub = sub[(~sub["extreme"])
               & (sub["src_lr_means"].fillna(0) > 0)
               & (sub["dst_lr_means"].fillna(0) > 0)].copy()
    if not len(sub):
        return None, None, None
    sub["lr_pair"] = (sub["ligand_complex"].astype(str)
                       + " → " + sub["receptor_complex"].astype(str))

    # Per-edge signed Z-score of log2fc. Outlier rows can no longer
    # dominate ranking — they're normalized against the edge's own
    # log2fc distribution.
    def _zscore(g):
        v = g.values.astype(float)
        mu = np.nanmean(v)
        sd = np.nanstd(v)
        if not np.isfinite(sd) or sd <= 1e-12:
            return pd.Series(np.zeros_like(v), index=g.index)
        return pd.Series((v - mu) / sd, index=g.index)

    sub["lfc_z"] = sub.groupby("edge")["log2fc"].transform(_zscore)

    top_pairs = set()
    for edge in EDGES:
        i, j = edge
        elabel = f"{i}->{j}"
        es = sub[sub["edge"] == elabel]
        if not len(es):
            continue
        es_pp = (es.assign(absz=es["lfc_z"].abs())
                    .groupby("lr_pair")["absz"].max()
                    .sort_values(ascending=False))
        top_pairs.update(es_pp.head(TOP_N_PER_EDGE).index.tolist())

    if not top_pairs:
        return None, None, None

    sub = sub[sub["lr_pair"].isin(top_pairs)].copy()
    edge_cols = [f"{i}->{j}" for i, j in EDGES]
    mat = (sub.groupby(["lr_pair", "edge"])["log2fc"].mean()
              .unstack("edge").reindex(columns=edge_cols))
    dst_p_mat = (sub.groupby(["lr_pair", "edge"])["dst_p"].min()
                    .unstack("edge").reindex(columns=edge_cols))
    src_p_mat = (sub.groupby(["lr_pair", "edge"])["src_p"].min()
                    .unstack("edge").reindex(columns=edge_cols))

    # Hierarchical clustering of rows (per direction panel) on log2fc.
    if mat.shape[0] >= 2:
        link_mat = np.nan_to_num(mat.values, nan=0.0)
        Z = linkage(link_mat, method="average", metric="euclidean")
        order = leaves_list(Z)
        mat = mat.iloc[order]
        dst_p_mat = dst_p_mat.iloc[order]
        src_p_mat = src_p_mat.iloc[order]

    sig_mat = np.zeros_like(mat.values, dtype=bool)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat.iat[ri, ci]
            if pd.isna(v):
                continue
            src_p_v = src_p_mat.iat[ri, ci]
            dst_p_v = dst_p_mat.iat[ri, ci]
            if pd.isna(src_p_v) and pd.isna(dst_p_v):
                # L-R pair not tested on this edge → no dot.
                continue
            p = dst_p_v if v >= 0 else src_p_v
            if (pd.notna(p) and p < DOT_P_CUTOFF
                    and abs(v) > DOT_LFC_CUTOFF):
                sig_mat[ri, ci] = True
    return mat, sig_mat, (dst_p_mat, src_p_mat)


m_mat, m_sig, _ = _build_dir_matrix("M_to_T")
t_mat, t_sig, _ = _build_dir_matrix("T_to_M")

if m_mat is None and t_mat is None:
    print("No significant L-R rows in either direction; skipping heatmap.")
else:
    edge_cols = [f"{i}->{j}" for i, j in EDGES]
    n_rows_m = m_mat.shape[0] if m_mat is not None else 0
    n_rows_t = t_mat.shape[0] if t_mat is not None else 0
    height_total = 1.6 + 0.30 * max(n_rows_m, n_rows_t)

    # Color scale clipped to ±HEATMAP_CLIP. Raw values stay in CSV.
    norm = TwoSlopeNorm(vmin=-HEATMAP_CLIP, vcenter=0.0, vmax=HEATMAP_CLIP)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(15, max(height_total, 4.5)))
    gs = GridSpec(
        2, 3,
        height_ratios=[0.6, max(n_rows_m, n_rows_t, 1)],
        width_ratios=[1.0, 1.0, 0.035],
        hspace=0.05, wspace=0.18,
        left=0.10, right=0.94, top=0.93, bottom=0.20,
    )
    ax_bar_m = fig.add_subplot(gs[0, 0])
    ax_bar_t = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    def _vol(direction):
        s = summary_df[summary_df["direction"] == direction] \
            .set_index("edge").reindex(edge_cols)
        return s["rewiring_volume"].fillna(0.0).values

    vol_m = _vol("M_to_T")
    vol_t = _vol("T_to_M")
    vol_max = float(max(vol_m.max(), vol_t.max(), 1e-9))

    for ax_bar, vols, title in [
        (ax_bar_m, vol_m, "M → T"),
        (ax_bar_t, vol_t, "T → M"),
    ]:
        ax_bar.bar(range(len(edge_cols)), vols,
                   color=["#888"] * len(edge_cols), edgecolor="black",
                   linewidth=0.4)
        ax_bar.set_xlim(-0.5, len(edge_cols) - 0.5)
        ax_bar.set_xticks([])
        ax_bar.set_ylim(0, vol_max * 1.1)
        ax_bar.set_ylabel("rewire\n(sum |log2fc|,\np<0.05, raw)", fontsize=7)
        ax_bar.tick_params(axis="y", labelsize=7)
        for s in ("top", "right"):
            ax_bar.spines[s].set_visible(False)
        ax_bar.set_title(title, fontsize=12, fontweight="bold")

    def _draw_heatmap(ax, mat, sig_mat, ylabel_show):
        if mat is None or not len(mat):
            ax.text(0.5, 0.5, "no significant L-R rows",
                    ha="center", va="center", fontsize=11,
                    color="dimgray", transform=ax.transAxes)
            ax.set_xticks(range(len(edge_cols)))
            ax.set_xticklabels(edge_cols, rotation=45, ha="right",
                                fontsize=8)
            ax.set_yticks([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            return None
        # Clip values for color only; raw values remain in `mat` for the
        # underlying CSV.
        disp = np.clip(mat.values, -HEATMAP_CLIP, HEATMAP_CLIP)
        im = ax.imshow(disp, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(edge_cols)))
        ax.set_xticklabels(edge_cols, rotation=45, ha="right", fontsize=8)
        for tick, (i, j) in zip(ax.get_xticklabels(), EDGES):
            tick.set_color(TISSUE_COLORS[i])
            if (i, j) in SAME_TISSUE_EDGES:
                tick.set_fontweight("bold")
        if ylabel_show:
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index, fontsize=7)
        else:
            ax.set_yticks([])
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                if sig_mat[ri, ci]:
                    ax.scatter(ci, ri, s=DOT_SIZE, marker="o",
                                facecolor="black", edgecolor="none",
                                zorder=3)
        ax.axvline(2.5, color="black", lw=0.8, alpha=0.5)
        return im

    im_m = _draw_heatmap(ax_m, m_mat, m_sig, ylabel_show=True)
    im_t = _draw_heatmap(ax_t, t_mat, t_sig, ylabel_show=True)

    im_for_cbar = im_m if im_m is not None else im_t
    if im_for_cbar is not None:
        cb = fig.colorbar(im_for_cbar, cax=cax)
        cb.set_label("log₂(dst / src) L-R means", fontsize=9)
        cb.ax.tick_params(labelsize=7)
    else:
        cax.axis("off")

    fig.suptitle(
        "Branch-restricted L-R differential (src vs dst pool)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.04,
        "Branch-restricted L-R differential. Columns: 9 (src->dst) edges, "
        "same-tissue first then cross-tissue. Color = log2(dst/src) of "
        "lr_means, clipped to [-5,+5] for display. Dots: side-specific "
        "p<0.01 and |log2fc|>1. Rows clustered within direction panel.",
        ha="center", va="bottom", fontsize=8, color="dimgray", style="italic",
        wrap=True,
    )

    png_path = OUT_DIR / "signaling_edge_heatmap.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

# %%
# ---- Step 9: Print summary ----
elapsed = time.time() - t_start

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print("\nBranches per edge:")
for edge in EDGES:
    i, j = edge
    print(f"  {i}->{j}: {n_branches_per_edge[edge]}")

print("\nCells per (edge, side):")
print(f"  {'edge':<14}{'src_T':>8}{'src_M':>8}{'dst_T':>8}{'dst_M':>8}")
for edge in EDGES:
    i, j = edge
    print(f"  {i+'->'+j:<14}"
          f"{n_t_cells_edge[edge]['src']:>8}"
          f"{n_m_cells_edge[edge]['src']:>8}"
          f"{n_t_cells_edge[edge]['dst']:>8}"
          f"{n_m_cells_edge[edge]['dst']:>8}")

print("\nLIANA rows per (edge, side) [pre-direction filter]:")
print(f"  {'edge':<14}{'src':>8}{'dst':>8}")
empty_edges = []
for edge in EDGES:
    i, j = edge
    s = n_liana_rows_edge[edge]["src"]
    d = n_liana_rows_edge[edge]["dst"]
    if s == 0 and d == 0:
        empty_edges.append(f"{i}->{j}")
    print(f"  {i+'->'+j:<14}{s:>8}{d:>8}")
if empty_edges:
    print(f"  Empty edges (no LIANA on either side): {empty_edges}")

print("\nSignificant rows per (edge, direction) [post-filter, pivoted]:")
print(f"  {'edge':<14}{'direction':<10}{'n_sig':>8}"
      f"{'n_dst_up':>10}{'n_src_up':>10}{'vol':>10}")
for _, r in summary_df.iterrows():
    print(f"  {r['edge']:<14}{r['direction']:<10}"
          f"{r['n_lr_significant_either']:>8}"
          f"{r['n_dst_up']:>10}{r['n_src_up']:>10}"
          f"{r['rewiring_volume']:>10.2f}")

try:
    du_out = subprocess.run(
        ["du", "-sh", str(OUT_DIR)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"\nOutput dir: {du_out}")
except Exception as e:
    print(f"\nOutput dir size unavailable: {e}")

print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
print("Done.")
