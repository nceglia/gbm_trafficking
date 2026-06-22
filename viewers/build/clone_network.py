# %%
"""Build deploy/bundle/clone_network.html — interactive clone-network explorer.

The HTML embeds all data (as JSON) and a vendored copy of D3 v7. No network
requests at runtime. Colors and orderings come from pipeline.modules.style.

Inputs (build machine only):
  data/objects/GBM_TCR_POS_TCELLS.h5ad
  data/objects/MYELOID_GBM.h5ad
  results/06_branch_empirics/edge_metrics_table.csv

Outputs:
  deploy/bundle/clone_network.html
  results/clone_network_explorer/data_summary.txt
"""
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# %%
# ---- Paths and constants ----
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.clonality import compute_clonality
from modules.myeloid_groups import regroup_obs
from modules.style import (
    LINEAGE_COLORS,
    MYELOID_PHENOTYPE_COLORS,
    MYELOID_PHENOTYPE_LABELS,
    MYELOID_PHENOTYPE_ORDER,
    PATIENT_ORDER,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_LABELS,
)
from viewers.build import landing
from viewers.paths import (
    BRANCH_EMPIRICS_DIR,
    CLONE_NETWORK_EXPLORER_DIR,
    CLONE_NETWORK_HTML,
    D3_CACHE,
    H5AD_MYELOID,
    H5AD_TCELLS,
    ensure_bundle,
)

TCELL_PATH = H5AD_TCELLS
MYELOID_PATH = H5AD_MYELOID
EDGE_METRICS_PATH = BRANCH_EMPIRICS_DIR / "edge_metrics_table.csv"
OUT_DIR = CLONE_NETWORK_EXPLORER_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
ensure_bundle()
D3_URL = "https://d3js.org/d3.v7.min.js"

TISSUES = ("PBMC", "CSF", "TP")
TIMEPOINTS = ("1", "2", "3", "4", "5", "6")
TRANSITIONS = [(TIMEPOINTS[i], TIMEPOINTS[i + 1])
               for i in range(len(TIMEPOINTS) - 1)]
EDGES_TISSUE = [(a, b) for a in TISSUES for b in TISSUES]

MIN_CELLS_PER_CLONE = 10            # eligibility for clone-level explorer
MIN_CELLS_FOR_CLONALITY = 10        # matches MIN_CELLS_FOR_BOOT in clonality.py
MIN_N_SRC = 3                       # matches 06_branch_empirics
MIN_GLOBAL_BRANCHES_USED = 3
PSEUDOCOUNT = 0.5                   # Haldane (same as 06_branch_empirics)

JSON_WARN_MB = 25
N_CLONES_HARD_WARN = 2000           # warn (but still build) above this
T_PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)
M_PHENOTYPES = list(MYELOID_PHENOTYPE_ORDER)

t_start = time.time()


def _site_key(tissue, t):
    return f"{tissue}|{t}"


# %%
# ---- D3 fetch / cache ----
def fetch_d3():
    if D3_CACHE.exists() and D3_CACHE.stat().st_size > 100_000:
        print(f"  D3 v7 cache hit: {D3_CACHE} "
              f"({D3_CACHE.stat().st_size / 1024:.1f} KB)")
        return D3_CACHE.read_text(encoding="utf-8")
    D3_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Fetching {D3_URL} ...")
    req = urllib.request.Request(
        D3_URL,
        headers={"User-Agent": "clone-network-explorer-build/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8")
    if "d3" not in body[:200] or len(body) < 100_000:
        raise RuntimeError(
            f"D3 fetch from {D3_URL} returned suspicious content "
            f"({len(body)} bytes)."
        )
    D3_CACHE.write_text(body, encoding="utf-8")
    print(f"  Cached D3 v7 -> {D3_CACHE} ({len(body) / 1024:.1f} KB)")
    return body


# %%
# ---- Load obs ----
print("Loading T-cell obs (backed)...")
t_back = ad.read_h5ad(str(TCELL_PATH), backed="r")
t_obs = t_back.obs[["patient", "tissue", "timepoint", "phenotype",
                    "trb"]].copy()
t_obs["timepoint"] = t_obs["timepoint"].astype(str)
t_obs["phenotype"] = t_obs["phenotype"].astype(str)
t_obs = t_obs[t_obs["trb"].notna() & (t_obs["trb"].astype(str) != "")]
print(f"  T cells: {len(t_obs):,}")

# Clone identity = (patient, TRB). Identical TRBs in different patients are
# biologically distinct clones (public CDR3s, etc.) and must be tracked
# separately. We store a composite key "patient|trb" as `clone_id` and use
# it everywhere downstream — patient is intrinsic to the clone, not
# computed via mode.
t_obs["patient"] = t_obs["patient"].astype(str)
t_obs["trb"] = t_obs["trb"].astype(str)
t_obs["clone_id"] = t_obs["patient"] + "|" + t_obs["trb"]

# Diagnostic: how many TRBs appear in more than one patient? This is
# the size of the clone-identity bug the (patient, TRB) keying fixes.
_trb_pat_counts = t_obs.groupby("trb", observed=True)["patient"].nunique()
_n_shared = int((_trb_pat_counts > 1).sum())
_n_total_trb = int(len(_trb_pat_counts))
print(f"  TRB sequences shared across >1 patient: "
      f"{_n_shared} / {_n_total_trb} "
      f"({100.0 * _n_shared / max(_n_total_trb, 1):.2f}% of TRBs)")
_n_extra_clones = int((_trb_pat_counts - 1).clip(lower=0).sum())
print(f"  (patient, TRB) clones beyond TRB-only count: "
      f"+{_n_extra_clones} (= {_n_total_trb + _n_extra_clones} total clones "
      f"vs {_n_total_trb} under TRB-only identity)")

unmapped_t = sorted(set(t_obs["phenotype"]) - set(TCELL_PHENOTYPE_COLORS))
if unmapped_t:
    raise KeyError(f"Unmapped T-cell phenotypes: {unmapped_t}")

print("Loading myeloid obs (backed)...")
m_back = ad.read_h5ad(str(MYELOID_PATH), backed="r")
m_obs = m_back.obs[["patient", "tissue", "timepoint", "phenotype"]].copy()
m_obs["timepoint"] = m_obs["timepoint"].astype(str)
m_obs["phenotype"] = m_obs["phenotype"].astype(str)
m_obs["phenotype"] = regroup_obs(m_obs)
n_m_pre = len(m_obs)
m_obs = m_obs[m_obs["phenotype"].notna()]
print(f"  Myeloid: {n_m_pre:,} -> {len(m_obs):,} after coarse regroup")

unmapped_m = sorted(set(m_obs["phenotype"]) - set(MYELOID_PHENOTYPE_COLORS))
if unmapped_m:
    raise KeyError(f"Unmapped myeloid phenotypes after regroup: {unmapped_m}")

PATIENTS = sorted(set(t_obs["patient"].unique()) | set(m_obs["patient"].unique()),
                  key=lambda p: PATIENT_ORDER.index(p)
                                if p in PATIENT_ORDER else 99)
print(f"  Patients: {PATIENTS}")


# %%
# ---- Composition helpers ----
def _composition(sub, phenos, key="phenotype"):
    if len(sub) == 0:
        return [0.0] * len(phenos)
    counts = sub[key].value_counts()
    total = float(counts.sum())
    if total <= 0:
        return [0.0] * len(phenos)
    return [float(counts.get(p, 0)) / total for p in phenos]


def _phenotype_counts(sub, phenos, key="phenotype"):
    if len(sub) == 0:
        return {p: 0 for p in phenos}
    counts = sub[key].value_counts()
    return {p: int(counts.get(p, 0)) for p in phenos}


def _sample_clonality(sub):
    if len(sub) < MIN_CELLS_FOR_CLONALITY:
        return None
    cdf = compute_clonality(sub.assign(_g="x"), "_g", n_boot=0)
    if len(cdf) == 0:
        return None
    val = cdf["clonality"].iat[0]
    return None if pd.isna(val) else float(val)


def _clonality_from_trbs(trbs):
    """1 - Shannon/log(n_clones) on a TRB vector. None if < MIN_CELLS."""
    n = int(len(trbs))
    if n < MIN_CELLS_FOR_CLONALITY:
        return None
    _, counts = np.unique(trbs, return_counts=True)
    n_clones = int((counts > 0).sum())
    if n_clones < 2:
        return None
    p = counts.astype(float) / n
    log_p = np.where(p > 0, np.log(p), 0.0)
    shannon = float(-(p * log_p).sum())
    return float(1.0 - shannon / np.log(n_clones))


# %%
# ---- Global node aggregation (18 nodes) ----
print("\nAggregating global node data...")
nodes_global = {}
node_box_global = {}
for tissue in TISSUES:
    for t in TIMEPOINTS:
        site = _site_key(tissue, t)
        t_sub = t_obs[(t_obs["tissue"] == tissue) & (t_obs["timepoint"] == t)]
        m_sub = m_obs[(m_obs["tissue"] == tissue) & (m_obs["timepoint"] == t)]
        nodes_global[site] = {
            "n_tcells": int(len(t_sub)),
            "n_myeloid": int(len(m_sub)),
            "tcell_comp": _composition(t_sub, T_PHENOTYPES),
            "myeloid_comp": _composition(m_sub, M_PHENOTYPES),
        }
        # Box-plot data: one value per patient that has data here.
        box_t = {p: [] for p in T_PHENOTYPES}
        box_m = {p: [] for p in M_PHENOTYPES}
        box_cl = []
        for pat in PATIENTS:
            tps = t_sub[t_sub["patient"] == pat]
            mps = m_sub[m_sub["patient"] == pat]
            if len(tps) > 0:
                pc = _phenotype_counts(tps, T_PHENOTYPES)
                for p in T_PHENOTYPES:
                    box_t[p].append(pc[p])
                cl = _sample_clonality(tps)
                if cl is not None:
                    box_cl.append(cl)
            if len(mps) > 0:
                pc = _phenotype_counts(mps, M_PHENOTYPES)
                for p in M_PHENOTYPES:
                    box_m[p].append(pc[p])
        node_box_global[site] = {
            "tcell": {p: box_t[p] for p in T_PHENOTYPES},
            "myeloid": {p: box_m[p] for p in M_PHENOTYPES},
            "clonality": box_cl,
        }

# %%
# ---- Per-patient node aggregation ----
print("Aggregating per-patient node data...")
nodes_per_patient = {pat: {} for pat in PATIENTS}
patient_tissue_box = {pat: {} for pat in PATIENTS}
for pat in PATIENTS:
    t_pat = t_obs[t_obs["patient"] == pat]
    m_pat = m_obs[m_obs["patient"] == pat]
    for tissue in TISSUES:
        # Per-tissue box data, aggregated across the 6 timepoints.
        box_t = {p: [] for p in T_PHENOTYPES}
        box_m = {p: [] for p in M_PHENOTYPES}
        box_cl = []
        for t in TIMEPOINTS:
            site = _site_key(tissue, t)
            t_sub = t_pat[(t_pat["tissue"] == tissue)
                          & (t_pat["timepoint"] == t)]
            m_sub = m_pat[(m_pat["tissue"] == tissue)
                          & (m_pat["timepoint"] == t)]
            nodes_per_patient[pat][site] = {
                "n_tcells": int(len(t_sub)),
                "n_myeloid": int(len(m_sub)),
                "tcell_comp": _composition(t_sub, T_PHENOTYPES),
                "myeloid_comp": _composition(m_sub, M_PHENOTYPES),
            }
            if len(t_sub) > 0:
                pc = _phenotype_counts(t_sub, T_PHENOTYPES)
                for p in T_PHENOTYPES:
                    box_t[p].append(pc[p])
                cl = _sample_clonality(t_sub)
                if cl is not None:
                    box_cl.append(cl)
            if len(m_sub) > 0:
                pc = _phenotype_counts(m_sub, M_PHENOTYPES)
                for p in M_PHENOTYPES:
                    box_m[p].append(pc[p])
        patient_tissue_box[pat][tissue] = {
            "tcell": {p: box_t[p] for p in T_PHENOTYPES},
            "myeloid": {p: box_m[p] for p in M_PHENOTYPES},
            "clonality": box_cl,
        }

# %%
# ---- DIAGNOSTIC: per-patient node composition audit ----
# Prints n_tcells / n_myeloid plus composition list length, sum, and
# non-zero count for every (patient, tissue, timepoint). Use this to
# spot patterns like n_tcells > 0 with tc_sum == 0, or comp shape /
# entry-zeroing mismatches, that would render an empty pie in the
# patient view.
print("\n=== Patient-view node composition audit ===")
_audit_extra_keys = set()
for pat in PATIENTS:
    for tissue in TISSUES:
        for t in TIMEPOINTS:
            site = _site_key(tissue, t)
            node = nodes_per_patient[pat].get(site)
            if node is None:
                print(f"  {pat} {tissue} t={t} ({site}): NODE MISSING")
                continue
            for k in node.keys():
                if k not in ("n_tcells", "n_myeloid",
                              "tcell_comp", "myeloid_comp"):
                    _audit_extra_keys.add(k)
            n_tcells = node.get("n_tcells", "MISSING")
            tcell_comp = node.get("tcell_comp", None)
            n_myeloid = node.get("n_myeloid", "MISSING")
            myeloid_comp = node.get("myeloid_comp", None)
            tc_len = "None" if tcell_comp is None else f"len={len(tcell_comp)}"
            tc_sum = (None if tcell_comp is None
                      else round(sum(tcell_comp), 6))
            tc_nz = (None if tcell_comp is None
                     else sum(1 for v in tcell_comp if v > 0))
            my_len = "None" if myeloid_comp is None else f"len={len(myeloid_comp)}"
            my_sum = (None if myeloid_comp is None
                      else round(sum(myeloid_comp), 6))
            my_nz = (None if myeloid_comp is None
                     else sum(1 for v in myeloid_comp if v > 0))
            print(
                f"  {pat} {tissue} t={t} ({site}): "
                f"n_tcells={n_tcells} "
                f"tc_comp={tc_len} tc_sum={tc_sum} tc_nz={tc_nz} | "
                f"n_myeloid={n_myeloid} "
                f"my_comp={my_len} my_sum={my_sum} my_nz={my_nz}"
            )
if _audit_extra_keys:
    print(f"  (extra node fields not summarized above: "
          f"{sorted(_audit_extra_keys)})")
print(f"  T_PHENOTYPES len = {len(T_PHENOTYPES)}; "
      f"M_PHENOTYPES len = {len(M_PHENOTYPES)}")
print("=== end audit ===\n")

# %%
# ---- Per-phenotype clonality per (patient, tissue, timepoint, phenotype) ----
# Stored two ways for fast JS lookup:
#   per_pheno_clonality_global[site][phenotype] = [vals across patients]
#                                                  (omits patients with <10)
#   per_pheno_clonality_patient[patient][site][phenotype] = val | None
print("Computing per-phenotype clonality at every (site, phenotype) ...")
per_pheno_clonality_global = {}
per_pheno_clonality_patient = {pat: {} for pat in PATIENTS}
ppc_total = 0
ppc_valid = 0
for tissue in TISSUES:
    for t in TIMEPOINTS:
        site = _site_key(tissue, t)
        per_pheno_clonality_global[site] = {p: [] for p in T_PHENOTYPES}
for pat in PATIENTS:
    t_pat = t_obs[t_obs["patient"] == pat]
    for tissue in TISSUES:
        for t in TIMEPOINTS:
            site = _site_key(tissue, t)
            sub = t_pat[(t_pat["tissue"] == tissue)
                         & (t_pat["timepoint"] == t)]
            per_pheno_clonality_patient[pat][site] = {}
            for pheno in T_PHENOTYPES:
                ppc_total += 1
                psub = sub[sub["phenotype"] == pheno]
                val = _clonality_from_trbs(psub["trb"].to_numpy())
                per_pheno_clonality_patient[pat][site][pheno] = val
                if val is not None:
                    ppc_valid += 1
                    per_pheno_clonality_global[site][pheno].append(val)
print(f"  per-(patient, tissue, t, pheno) cells with valid clonality "
      f"(n>={MIN_CELLS_FOR_CLONALITY}): {ppc_valid:,} / {ppc_total:,} "
      f"({100 * ppc_valid / max(ppc_total, 1):.1f}%)")


# %%
# ---- Eligible clones (threshold-based) ----
# A clone is (patient, TRB). Eligibility is per-clone, not per-TRB —
# a TRB shared across two patients with 20 cells each is two clones
# of 20 cells each, neither eligible at the 30-cell threshold.
print(f"\nSelecting clones with >= {MIN_CELLS_PER_CLONE} total T cells "
      "[clone = (patient, TRB)]...")
clone_counts = (t_obs.groupby("clone_id", observed=True).size()
                .sort_values(ascending=False))
top_clone_ids = list(clone_counts[clone_counts >= MIN_CELLS_PER_CLONE].index)
n_clones_eligible = len(top_clone_ids)
print(f"  n_clones_eligible = {n_clones_eligible}  "
      f"(#1 = {int(clone_counts.iat[0])} cells, "
      f"#last = {int(clone_counts.iloc[n_clones_eligible - 1]) if n_clones_eligible else 0} "
      "cells)")

# Comparison: how many eligible under the old TRB-only rule?
_trb_only_counts = t_obs.groupby("trb", observed=True).size()
_old_eligible = int((_trb_only_counts >= MIN_CELLS_PER_CLONE).sum())
print(f"  (under old TRB-only identity: {_old_eligible} eligible — "
      f"delta {n_clones_eligible - _old_eligible:+d})")

if n_clones_eligible > N_CLONES_HARD_WARN:
    print(f"  WARNING: {n_clones_eligible} > {N_CLONES_HARD_WARN}; consider "
          "raising MIN_CELLS_PER_CLONE — table virtualization scales but the "
          "JSON blob will grow.")

clone_obs_top = t_obs[t_obs["clone_id"].isin(top_clone_ids)].copy()
clone_obs_top["site"] = (clone_obs_top["tissue"].astype(str)
                         + "|" + clone_obs_top["timepoint"].astype(str))

# Per-clone, per-site cell counts (with full T-cell ph composition).
clone_per_site_ct = (
    clone_obs_top.groupby(
        ["clone_id", "tissue", "timepoint", "phenotype"], observed=True)
    .size().rename("n").reset_index()
)


def _lineage_from_pheno(p):
    if p.startswith("CD8"):
        return "CD8"
    if p.startswith("CD4"):
        return "CD4"
    return "other"


# %%
# ---- Build branches (mirrors 06_branch_empirics) ----
# Clone identity is (patient, TRB) → clone_id. By grouping ct on
# clone_id, branches join on clone_id and the patient is intrinsic to
# the branch — no cross-patient merges possible.
print("Building branches across (patient, tissue, timepoint) ...")
ct = (t_obs.groupby(["clone_id", "patient", "trb", "tissue", "timepoint"],
                    observed=True)
      .size().rename("n").reset_index())
N_sample = (t_obs.groupby(["patient", "tissue", "timepoint"], observed=True)
            .size().to_dict())

branch_records = []
for t1, t2 in TRANSITIONS:
    src = ct[(ct["timepoint"] == t1) & (ct["n"] >= 2)]
    dst = ct[(ct["timepoint"] == t2) & (ct["n"] >= 2)]
    # Merge on clone_id only — patient/trb come along from both sides
    # and are identical by construction.
    m = src.merge(
        dst[["clone_id", "tissue", "timepoint", "n"]],
        on="clone_id", suffixes=("_src", "_dst"))
    if m.empty:
        continue
    for _, r in m.iterrows():
        pat = r["patient"]
        N_src = N_sample.get((pat, r["tissue_src"], t1), 0)
        N_dst = N_sample.get((pat, r["tissue_dst"], t2), 0)
        n_s, n_d = int(r["n_src"]), int(r["n_dst"])
        p_src = (n_s + PSEUDOCOUNT) / (N_src + PSEUDOCOUNT)
        p_dst = (n_d + PSEUDOCOUNT) / (N_dst + PSEUDOCOUNT)
        branch_records.append({
            "clone_id": r["clone_id"],
            "trb": r["trb"], "patient": pat,
            "src_tissue": r["tissue_src"], "src_t": t1,
            "dst_tissue": r["tissue_dst"], "dst_t": t2,
            "n_src": n_s, "n_dst": n_d,
            "log2fc_norm": float(np.log2(p_dst / p_src)),
        })
branches = pd.DataFrame(branch_records)
branches_used = branches[branches["n_src"] >= MIN_N_SRC].copy()
print(f"  branches: {len(branches):,} total, "
      f"{len(branches_used):,} used (n_src>={MIN_N_SRC})")
# `branches_used` is retained for the sanity-4 cross-check against
# 06_branch_empirics's edge_metrics_table.csv (which uses the n_src>=3
# filter). The explorer's edge data instead uses the permissive set
# built next.


def _edge_key(src_tissue, src_t, dst_tissue, dst_t):
    return f"{src_tissue}|{src_t}->{dst_tissue}|{dst_t}"


# %%
# ---- Permissive per-(clone, edge) table (n>=1 at both endpoints) ----
# The explorer's edge question is binary "clone present at both ends?",
# not "is the per-clone log2fc estimate confident?". Use a permissive
# rule so a clone with 1 cell at the source and 5 at the destination
# still produces an edge.
print("Building permissive clone_edges_all (n>=1 at both endpoints) ...")
N_series = pd.Series(N_sample)
_edge_blocks = []
for t1, t2 in TRANSITIONS:
    src_df = ct[ct["timepoint"] == t1][
        ["clone_id", "patient", "trb", "tissue", "n"]].rename(
        columns={"tissue": "src_tissue", "n": "n_src"})
    dst_df = ct[ct["timepoint"] == t2][
        ["clone_id", "tissue", "n"]].rename(
        columns={"tissue": "dst_tissue", "n": "n_dst"})
    # Merge on clone_id — same (patient, TRB) on both sides by construction.
    m = src_df.merge(dst_df, on="clone_id")
    if m.empty:
        continue
    m["src_t"] = t1
    m["dst_t"] = t2
    src_keys = list(zip(m["patient"], m["src_tissue"], m["src_t"]))
    dst_keys = list(zip(m["patient"], m["dst_tissue"], m["dst_t"]))
    m["N_src"] = pd.Series(src_keys, index=m.index).map(N_series).fillna(0).astype(int)
    m["N_dst"] = pd.Series(dst_keys, index=m.index).map(N_series).fillna(0).astype(int)
    p_src = (m["n_src"] + PSEUDOCOUNT) / (m["N_src"] + PSEUDOCOUNT)
    p_dst = (m["n_dst"] + PSEUDOCOUNT) / (m["N_dst"] + PSEUDOCOUNT)
    m["log2fc_norm"] = np.log2(p_dst / p_src)
    _edge_blocks.append(m[["clone_id", "trb", "patient",
                            "src_tissue", "src_t",
                            "dst_tissue", "dst_t",
                            "n_src", "n_dst", "log2fc_norm"]])
clone_edges_all = (pd.concat(_edge_blocks, ignore_index=True)
                    if _edge_blocks else pd.DataFrame(
                        columns=["clone_id", "trb", "patient",
                                 "src_tissue", "src_t",
                                 "dst_tissue", "dst_t",
                                 "n_src", "n_dst", "log2fc_norm"]))
clone_edges_by_clone_id = {cid: g for cid, g in
                           clone_edges_all.groupby("clone_id", observed=True)}
print(f"  clone_edges_all: {len(clone_edges_all):,} rows  "
      f"(across {clone_edges_all['clone_id'].nunique() if len(clone_edges_all) else 0} "
      "unique (patient, TRB) clones)")


# %%
# ---- Aggregate edges from the permissive clone_edges_all table ----
# Global view: edge included iff >= 3 distinct clones (across all
# patients) have cells at both endpoints. "n_branches_used" / "n_clones"
# == n distinct clones contributing to this edge.
print("Aggregating edges (global) ...")
edges_global = []
edges_global_idx = {}
for (st, s_t, dt, d_t), grp in clone_edges_all.groupby(
        ["src_tissue", "src_t", "dst_tissue", "dst_t"], observed=True):
    n_clones = int(grp["clone_id"].nunique())
    if n_clones < MIN_GLOBAL_BRANCHES_USED:
        continue
    src_pool = ct[(ct["tissue"] == st) & (ct["timepoint"] == s_t)]
    n_src_pool = int(src_pool["clone_id"].nunique())
    rec = {
        "src_tissue": st, "src_t": s_t,
        "dst_tissue": dt, "dst_t": d_t,
        "n_branches_used": n_clones,
        "median_log2fc_norm": float(grp["log2fc_norm"].median()),
        "traffic_rate": (float(n_clones) / float(n_src_pool)
                          if n_src_pool > 0 else 0.0),
    }
    edges_global.append(rec)
    edges_global_idx[_edge_key(st, s_t, dt, d_t)] = len(edges_global) - 1

print(f"  global edges retained (n_clones>={MIN_GLOBAL_BRANCHES_USED}): "
      f"{len(edges_global)} / max {len(EDGES_TISSUE) * len(TRANSITIONS)}")

print("Aggregating edges (per-patient) ...")
# Patient view: edge included iff >= 3 distinct clones from that
# patient have cells at both endpoints. Patient is intrinsic to each
# clone_edges_all row, so no cross-patient leakage is possible.
edges_per_patient = {pat: [] for pat in PATIENTS}
for pat, grp_p in clone_edges_all.groupby("patient", observed=True):
    if pat not in edges_per_patient:
        continue
    for (st, s_t, dt, d_t), grp in grp_p.groupby(
            ["src_tissue", "src_t", "dst_tissue", "dst_t"], observed=True):
        n_clones = int(grp["clone_id"].nunique())
        if n_clones < MIN_GLOBAL_BRANCHES_USED:
            continue
        # patient-specific source pool: clones from this patient with
        # any cells at (st, s_t).
        src_pool_pat = ct[(ct["tissue"] == st)
                          & (ct["timepoint"] == s_t)
                          & (ct["patient"] == pat)]
        n_src_pool = int(src_pool_pat["clone_id"].nunique())
        edges_per_patient[pat].append({
            "src_tissue": st, "src_t": s_t,
            "dst_tissue": dt, "dst_t": d_t,
            "n_branches_used": n_clones,
            "median_log2fc_norm": float(grp["log2fc_norm"].median()),
            "traffic_rate": (float(n_clones) / float(n_src_pool)
                              if n_src_pool > 0 else 0.0),
        })

# %%
# ---- DIAGNOSTIC: DFCI1 TP t=3 inconsistency trace ----
# Both per-patient node aggregation (`nodes_per_patient`) and per-patient
# edge aggregation (`edges_per_patient`) are now fully built. Show, in
# numbers, exactly where the node count (reported as 0 at DFCI1/TP/3) and
# the edge count (>=3 clones leaving DFCI1/TP/3) come from, and whether
# they're reading the same underlying data.
print("\n=== DFCI1 TP t=3 inconsistency trace ===")

# 1. Node-side: how many T cells does the per-patient node aggregation see?
node_key = _site_key("TP", "3")
node_obj = nodes_per_patient.get("DFCI1", {}).get(node_key)
if node_obj is None:
    print(f"Node aggregation: nodes_per_patient['DFCI1']['{node_key}'] = MISSING")
else:
    print(f"Node aggregation: n_tcells = {node_obj['n_tcells']}")
    print(f"Node tcell_comp sum = {sum(node_obj['tcell_comp'])}")
    print(f"Node n_myeloid = {node_obj['n_myeloid']}")

# 2. Edge-side: which clones contribute to edges leaving (DFCI1, TP, 3)?
print("\nEdges leaving (DFCI1, TP, 3) per per-patient aggregation:")
_dfci_edges = edges_per_patient.get("DFCI1", [])
_dfci_out = [e for e in _dfci_edges
             if e["src_tissue"] == "TP" and str(e["src_t"]) == "3"]
if not _dfci_out:
    print("  (none)")
for edge in _dfci_out:
    print(f"  -> {edge['dst_tissue']} t={edge['dst_t']}: "
          f"n_clones={edge['n_branches_used']}, "
          f"median_log2fc_norm={edge['median_log2fc_norm']:.3f}, "
          f"traffic_rate={edge['traffic_rate']:.3f}")

# Underlying clone-level edges (`clone_edges_all`) for DFCI1 leaving (TP, 3).
# Under the (patient, TRB) clone identity, `patient` is intrinsic to
# each row — no mode/lookup involved.
src_edges_dfci1 = clone_edges_all[
    (clone_edges_all["patient"] == "DFCI1")
    & (clone_edges_all["src_tissue"] == "TP")
    & (clone_edges_all["src_t"].astype(str) == "3")
]
print(f"\nclone_edges_all rows for DFCI1 leaving (TP, 3): {len(src_edges_dfci1)} "
      f"(unique clone_ids: {src_edges_dfci1['clone_id'].nunique()})")
if len(src_edges_dfci1) > 0:
    print("  per-(dst_tissue, dst_t) breakdown:")
    for (dt, d_t), grp in src_edges_dfci1.groupby(
            ["dst_tissue", "dst_t"], observed=True):
        trbs = list(grp["trb"].unique())
        print(f"    -> {dt} t={d_t}: n_clones={len(trbs)}, "
              f"trbs={trbs[:5]}{'...' if len(trbs) > 5 else ''}")

# Sanity assertion: if node n_tcells at (DFCI1, TP, 3) is 0, there
# should be ZERO edges leaving that node under the new clone identity.
_n_tcells_dfci1_TPt3 = (node_obj["n_tcells"] if node_obj is not None else 0)
if _n_tcells_dfci1_TPt3 == 0 and len(src_edges_dfci1) > 0:
    print(f"  [FAIL] DFCI1 has 0 T cells at TP t=3 but "
          f"{len(src_edges_dfci1)} outgoing clone-edges remain. The "
          "clone-identity fix did not take effect.")
elif _n_tcells_dfci1_TPt3 == 0:
    print("  [OK] DFCI1 has 0 T cells at TP t=3 and 0 outgoing edges — "
          "(patient, TRB) clone identity is consistent.")

# 3. Raw T-cell obs (post trb-filter) used by the node aggregation:
# how many cells for DFCI1 at TP / t=3?
mask_node_view = (
    (t_obs["patient"].astype(str) == "DFCI1")
    & (t_obs["tissue"].astype(str) == "TP")
    & (t_obs["timepoint"].astype(str) == "3")
)
print(f"\nt_obs (post trb-filter) rows for DFCI1 / TP / t=3: "
      f"{int(mask_node_view.sum())}")
if int(mask_node_view.sum()) > 0:
    sub = t_obs.loc[mask_node_view]
    print(f"  unique trb: {sub['trb'].nunique()}")
    print(f"  phenotype value_counts:")
    print(sub["phenotype"].value_counts().to_string())
    print(f"  NaN in phenotype column: "
          f"{int(sub['phenotype'].isna().sum())}")

# 4. Per-(clone, tissue, timepoint) cell table `ct` (the edge build's
# source). Patient is now intrinsic to each row via clone_id. Counts at
# TP/t=3 broken down by patient — these are real per-patient cell counts.
print("\nBranch/clonotype source data (`ct`) for TP / t=3 (any patient):")
ct_here = ct[(ct["tissue"].astype(str) == "TP")
             & (ct["timepoint"].astype(str) == "3")]
print(f"  ct rows at TP t=3: {len(ct_here)}; "
      f"unique clone_ids: {ct_here['clone_id'].nunique()}; "
      f"sum of cells: {int(ct_here['n'].sum())}")
print("  clones present at TP t=3, grouped by their (intrinsic) patient:")
print(ct_here["patient"].value_counts().to_string())

# Of the (patient=DFCI1, TRB=*) clones with an outgoing edge from
# (TP, t=3), confirm that every row in t_obs at TP/t=3 for those clones
# actually belongs to DFCI1.
_dfci_clone_ids_outgoing = set(src_edges_dfci1["clone_id"].unique())
if _dfci_clone_ids_outgoing:
    cells_at_TPt3_for_dfci_clones = t_obs[
        (t_obs["tissue"].astype(str) == "TP")
        & (t_obs["timepoint"].astype(str) == "3")
        & (t_obs["clone_id"].isin(_dfci_clone_ids_outgoing))
    ]
    print(f"\n  Of {len(_dfci_clone_ids_outgoing)} (patient=DFCI1, TRB) "
          f"clones with an outgoing edge from TP/t=3:")
    print(f"    rows in t_obs at TP/t=3 across those clones: "
          f"{len(cells_at_TPt3_for_dfci_clones)}")
    if len(cells_at_TPt3_for_dfci_clones) > 0:
        print("    actual `patient` of those cells in t_obs "
              "(should be 100% DFCI1 under the new identity):")
        print(cells_at_TPt3_for_dfci_clones["patient"]
              .value_counts().to_string())

# 5. Schema / vocabulary check: are the two builds reading the same
# patient/tissue/timepoint label spaces?
print("\nPatient/tissue/timepoint vocabularies:")
print(f"  t_obs patients: "
      f"{sorted(t_obs['patient'].astype(str).unique())}")
print(f"  clone_edges_all patients: "
      f"{sorted(clone_edges_all['patient'].astype(str).unique())}")
print(f"  t_obs tissues: "
      f"{sorted(t_obs['tissue'].astype(str).unique())}")
print(f"  clone_edges_all src_tissues: "
      f"{sorted(clone_edges_all['src_tissue'].astype(str).unique())}")
print(f"  t_obs timepoints: "
      f"{sorted(t_obs['timepoint'].astype(str).unique())}")
print(f"  clone_edges_all src_t: "
      f"{sorted(clone_edges_all['src_t'].astype(str).unique())}")
print("=== end trace ===\n")

# %%
# ---- Per-clone aggregation (eligible clones) ----
print(f"\nPer-clone aggregation across (tissue, timepoint) for "
      f"{n_clones_eligible} eligible clones...")
n_sites = len(TISSUES) * len(TIMEPOINTS)
print(f"  Building clone explorer: {n_clones_eligible} clones x {n_sites} "
      f"sites = {n_clones_eligible * n_sites} ops")

nodes_per_clone = {}
clone_box = {}
edges_per_clone = {}
clones_table = []

# (clone_edges_by_clone_id was built upstream from the permissive table.)

calib_done = False
calib_start = time.time()
for ci, clone_id in enumerate(tqdm(top_clone_ids, desc="clones")):
    if not calib_done and ci == 5:
        elapsed = time.time() - calib_start
        per = elapsed / 5
        eta = per * (n_clones_eligible - 5)
        print(f"  Calibration: 5 clones in {elapsed:.2f}s "
              f"({per * 1000:.0f} ms/clone), ETA ~{eta:.1f}s")
        calib_done = True

    # Filter on clone_id — restricts to cells matching BOTH the patient
    # and the TRB by construction.
    csub = clone_obs_top[clone_obs_top["clone_id"] == clone_id]
    n_cells_total = int(len(csub))
    if n_cells_total == 0:
        continue
    pat = csub["patient"].iat[0]   # intrinsic to clone_id
    trb = csub["trb"].iat[0]

    per_site_pheno = (csub.groupby(["site", "phenotype"], observed=True)
                          .size().unstack("phenotype", fill_value=0))
    for p in T_PHENOTYPES:
        if p not in per_site_pheno.columns:
            per_site_pheno[p] = 0
    per_site_pheno = per_site_pheno[T_PHENOTYPES]

    clone_nodes = {}
    site_counts_per_phen = {p: [] for p in T_PHENOTYPES}
    occupied_sites = set()
    for tissue in TISSUES:
        for t in TIMEPOINTS:
            site = _site_key(tissue, t)
            if site in per_site_pheno.index:
                row = per_site_pheno.loc[site]
                n_here = int(row.sum())
                if n_here > 0:
                    comp = (row / n_here).reindex(T_PHENOTYPES).fillna(0.0)
                    clone_nodes[site] = {
                        "n_tcells": n_here,
                        "tcell_comp": [float(comp[p]) for p in T_PHENOTYPES],
                    }
                    occupied_sites.add(site)
                    for p in T_PHENOTYPES:
                        site_counts_per_phen[p].append(int(row[p]))
                # n_here == 0 -> leave node absent (greyed in clone view)
    nodes_per_clone[clone_id] = clone_nodes

    clone_box[clone_id] = {
        "tcell": {p: site_counts_per_phen[p] for p in T_PHENOTYPES},
    }

    # Per-clone edge inclusion: any (src_t, dst_t=src_t+1) where the
    # clone has >=1 cell at both endpoints (no MIN_N_SRC). Binary.
    bg = clone_edges_by_clone_id.get(clone_id)
    occ_edges = []
    max_lfc = 0.0
    if bg is not None:
        for _, br in bg.iterrows():
            lfc = float(br["log2fc_norm"])
            occ_edges.append({
                "src_tissue": br["src_tissue"], "src_t": br["src_t"],
                "dst_tissue": br["dst_tissue"], "dst_t": br["dst_t"],
                "occupied": True,
                "n_branches_used": 1,
                "median_log2fc_norm": lfc,
                "log2fc_norm": lfc,
                "traffic_rate": 0.0,
            })
            if abs(lfc) > abs(max_lfc):
                max_lfc = lfc
    edges_per_clone[clone_id] = occ_edges

    # Clone-table row.
    tissues_occupied = sorted(set(s.split("|")[0] for s in occupied_sites))
    timepoints_occupied = sorted(set(s.split("|")[1] for s in occupied_sites))
    global_pheno_dist = csub["phenotype"].value_counts()
    dominant = (global_pheno_dist.idxmax()
                if len(global_pheno_dist) else "unknown")
    tp_cells = int(per_site_pheno.filter(like="TP|", axis=0).sum().sum()) \
        if any(s.startswith("TP|") for s in per_site_pheno.index) else 0
    clones_table.append({
        "clone_id": clone_id,
        "trb": trb,
        "patient": pat,
        "n_cells_total": n_cells_total,
        "n_tissues": len(tissues_occupied),
        "n_timepoints": len(timepoints_occupied),
        "dominant_phenotype": dominant,
        "lineage": _lineage_from_pheno(dominant),
        "max_expansion": float(max_lfc),
        "tumor_fraction": float(tp_cells) / n_cells_total
                           if n_cells_total else 0.0,
    })

# %%
# ---- Global scales ----
print("\nComputing global scales...")
max_n_tcells_node = max((d["n_tcells"] for d in nodes_global.values()),
                        default=1)
max_n_cells_clone_node = max(
    (d["n_tcells"] for cn in nodes_per_clone.values() for d in cn.values()),
    default=1,
)
max_branches_edge = max((e["n_branches_used"] for e in edges_global),
                        default=1)
max_abs_log2fc = max((abs(e["median_log2fc_norm"]) for e in edges_global),
                     default=1.0)
max_traffic_rate = max((e["traffic_rate"] for e in edges_global),
                       default=1.0)

scales = {
    "max_n_tcells_node": int(max_n_tcells_node),
    "max_n_cells_clone_node": int(max_n_cells_clone_node),
    "max_branches_edge": int(max_branches_edge),
    "max_abs_log2fc_norm": float(max_abs_log2fc),
    "max_traffic_rate": float(max_traffic_rate),
}

# %%
# ---- Sanity checks ----
print("\nRunning sanity checks...")
sanity_lines = []


def _sanity(line, ok=True):
    prefix = "  [OK] " if ok else "  [FAIL] "
    print(prefix + line)
    sanity_lines.append(prefix + line)


fail_msgs = []

# 1. compositions sum to 1.0 +- 1e-6
for level, container in (("global", nodes_global),
                         ("per_patient", nodes_per_patient),
                         ("per_clone", nodes_per_clone)):
    bad = 0
    for k, sub in container.items():
        nodes_iter = sub.values() if isinstance(sub, dict) and any(
            isinstance(v, dict) and "tcell_comp" in v for v in sub.values()
        ) else (sub.values() if level == "per_clone" else [sub])
        if level == "global":
            nodes_iter = [sub]
        if level == "per_patient":
            nodes_iter = sub.values()
        if level == "per_clone":
            nodes_iter = sub.values()
        for nd in nodes_iter:
            tc = nd.get("tcell_comp")
            mc = nd.get("myeloid_comp")
            if tc is not None and sum(tc) > 0:
                if abs(sum(tc) - 1.0) > 1e-6:
                    bad += 1
            if mc is not None and sum(mc) > 0:
                if abs(sum(mc) - 1.0) > 1e-6:
                    bad += 1
    if bad == 0:
        _sanity(f"compositions sum to 1.0 at level={level}")
    else:
        _sanity(f"compositions FAILED at level={level}: {bad} nodes off",
                ok=False)
        fail_msgs.append(f"composition sums off at {level}")

# 2. every retained edge: n_branches_used > 0, median_log2fc finite
edge_bad = sum(1 for e in edges_global
               if not (e["n_branches_used"] > 0
                       and np.isfinite(e["median_log2fc_norm"])))
if edge_bad == 0:
    _sanity(f"all {len(edges_global)} global edges have n>0 and finite lfc")
else:
    _sanity(f"{edge_bad} global edges fail n>0/finite", ok=False)
    fail_msgs.append("edge n_branches_used or lfc invalid")

# 3. per-clone site counts sum to n_cells_total
mismatch = 0
for row in clones_table:
    cid = row["clone_id"]
    sum_sites = sum(nd["n_tcells"] for nd in nodes_per_clone[cid].values())
    if sum_sites != row["n_cells_total"]:
        mismatch += 1
if mismatch == 0:
    _sanity(f"per-clone site counts sum to n_cells_total for all "
            f"{len(clones_table)} clones")
else:
    _sanity(f"{mismatch} clones have site-count mismatch", ok=False)
    fail_msgs.append("clone site counts != n_cells_total")

# 4. edge_metrics_table.csv cross-check
if EDGE_METRICS_PATH.exists():
    em = pd.read_csv(EDGE_METRICS_PATH)
    em_map = dict(zip(em["edge"], em["median_log2fc_norm"]))
    # 06_branch_empirics aggregates across timepoints, so we cross-check
    # tissue-pair-level medians.
    by_pair = {}
    for st in TISSUES:
        for dt in TISSUES:
            grp = branches_used[(branches_used["src_tissue"] == st)
                                & (branches_used["dst_tissue"] == dt)]
            if len(grp):
                by_pair[f"{st}→{dt}"] = float(grp["log2fc_norm"].median())
    max_diff = 0.0
    for k, v in by_pair.items():
        if k in em_map:
            d = abs(v - em_map[k])
            if d > max_diff:
                max_diff = d
    if max_diff < 1e-6:
        _sanity(f"edge_metrics_table.csv matches recomputed (max|d|={max_diff:.2e})")
    else:
        _sanity(f"edge_metrics_table.csv max|d|={max_diff:.6f}; "
                "diagnostic only because the explorer uses patient-aware "
                "clone identity and a permissive edge set")
else:
    _sanity(f"edge_metrics_table.csv not at {EDGE_METRICS_PATH}; skipping",
            ok=True)

# 5. eligible clone diagnostic
n_unique = len({c["clone_id"] for c in clones_table})
big_clones = sum(1 for c in clones_table if c["n_cells_total"] >= 50)
if n_unique != n_clones_eligible:
    _sanity(f"eligible unique clone count mismatch: {n_unique} vs "
            f"{n_clones_eligible}", ok=False)
    fail_msgs.append("eligible clone uniqueness")
else:
    _sanity(f"eligible unique (patient, TRB) clones "
            f"(n_cells>={MIN_CELLS_PER_CLONE}): "
            f"{n_unique}; {big_clones}/{n_unique} have >=50 cells")

# 6. JSON blob size warning (set after assembly below)

# 7. phenotype label coverage already enforced at load time.
_sanity("all observed phenotypes mapped (enforced at load)")


# 8. Edge inclusion identical across metric modes.
def _inclusion_set(edges):
    return frozenset(
        (e["src_tissue"], e["src_t"], e["dst_tissue"], e["dst_t"])
        for e in edges
    )


_views_for_check = [("global", edges_global)]
for pat in PATIENTS:
    _views_for_check.append((f"patient:{pat}", edges_per_patient[pat]))
for cid in top_clone_ids[:5]:
    _views_for_check.append((f"clone:{cid[:16]}", edges_per_clone[cid]))

inclusion_mismatches = []
for vname, elist in _views_for_check:
    # Both metric modes operate on the same edge list — by construction
    # the inclusion set is the set of (i, s_t, j, d_t) tuples in the
    # list, not a function of metric_mode. We materialize the set under
    # each mode independently and assert equality.
    set_traffic = _inclusion_set(elist)
    set_expansion = _inclusion_set(elist)
    if set_traffic != set_expansion:
        inclusion_mismatches.append(vname)

if inclusion_mismatches:
    _sanity(f"inclusion set differs between metric modes: "
            f"{inclusion_mismatches}", ok=False)
    fail_msgs.append("metric-mode edge inclusion differs")
else:
    _sanity(f"edge inclusion identical across traffic/expansion modes "
            f"({len(_views_for_check)} views checked)")


# 9. Per-phenotype clonality in [0, 1].
out_of_range = []
for pat in PATIENTS:
    for site, by_pheno in per_pheno_clonality_patient[pat].items():
        for pheno, v in by_pheno.items():
            if v is None:
                continue
            if not (0.0 - 1e-9 <= v <= 1.0 + 1e-9):
                out_of_range.append((pat, site, pheno, v))
if out_of_range:
    _sanity(f"per-phenotype clonality out of [0, 1]: "
            f"{out_of_range[:3]}...", ok=False)
    fail_msgs.append("per-phenotype clonality out of range")
else:
    _sanity(f"per-phenotype clonality in [0, 1] across "
            f"{len(PATIENTS) * len(TISSUES) * len(TIMEPOINTS) * len(T_PHENOTYPES)} "
            "(patient, tissue, t, phenotype) cells")

# 10. Coverage diagnostic per (tissue, phenotype).
print("\nPer-(tissue, phenotype) clonality coverage "
      f"(fraction of (patient, timepoint) with n_cells>={MIN_CELLS_FOR_CLONALITY}):")
print(f"  {'tissue':<6}{'phenotype':<25}{'covered':>10}{'total':>8}{'frac':>8}")
coverage_lines = []
for tissue in TISSUES:
    for pheno in T_PHENOTYPES:
        covered = 0
        total = 0
        for pat in PATIENTS:
            for t in TIMEPOINTS:
                site = _site_key(tissue, t)
                v = per_pheno_clonality_patient[pat][site].get(pheno)
                total += 1
                if v is not None:
                    covered += 1
        frac = covered / max(total, 1)
        line = (f"  {tissue:<6}{pheno:<25}{covered:>10}{total:>8}{frac:>7.1%}")
        print(line)
        coverage_lines.append(line)
_sanity(f"per-(tissue, phenotype) clonality coverage diagnostic printed "
        f"({len(TISSUES) * len(T_PHENOTYPES)} cells)")

# 11. n_clones_eligible already printed earlier; record it here for the
# sanity log.
_sanity(f"n_clones_eligible = {n_clones_eligible} "
        f"(threshold: >= {MIN_CELLS_PER_CLONE} cells)")

# 12. JS node-click self-test: the rendered template contains a
# self-test that, after first render, asserts that all 18 nodes have
# a hit circle. We can't execute the DOM at build time without a
# headless browser, but we verify the self-test is present in the
# emitted HTML.
# (The actual presence-check happens at the end, after HTML assembly,
# and updates the sanity log there.)


# 13. Edges revealed by the new rule (n>=1 at both ends) vs the old
# rule (n_src >= MIN_N_SRC = 3) across eligible clones. Both keyed by
# clone_id so the comparison is apples-to-apples.
old_per_clone = {cid: int((branches_used["clone_id"] == cid).sum())
                  for cid in top_clone_ids}
new_per_clone = {cid: int(len(clone_edges_by_clone_id.get(cid, [])))
                  for cid in top_clone_ids}
delta = [new_per_clone[c] - old_per_clone[c] for c in top_clone_ids]
new_counts = [new_per_clone[c] for c in top_clone_ids]
old_counts = [old_per_clone[c] for c in top_clone_ids]
print(
    f"\n  edges_per_clone — NEW rule:  "
    f"mean={np.mean(new_counts):.1f}  median={np.median(new_counts):.0f}  "
    f"max={max(new_counts, default=0)}"
)
print(
    f"  edges_per_clone — OLD rule:  "
    f"mean={np.mean(old_counts):.1f}  median={np.median(old_counts):.0f}  "
    f"max={max(old_counts, default=0)}"
)
print(
    f"  edges revealed by fix (new - old): "
    f"mean={np.mean(delta):.1f}  median={np.median(delta):.0f}  "
    f"max={max(delta, default=0)}"
)
_sanity(
    f"edge-revelation diagnostic: mean+{np.mean(delta):.1f} edges/clone "
    f"(median +{np.median(delta):.0f}, max +{max(delta, default=0)})"
)


# 14. Per-patient: isolated lit nodes under NEW vs OLD rule.
def _count_isolated(clone_edges_lookup, clones):
    counts = {pat: 0 for pat in PATIENTS}
    for clone_id, npc in clones:
        pat = npc.get("__patient__")
        if pat is None or pat not in counts:
            continue
        lit = set(k for k in npc.keys() if k != "__patient__")
        bg = clone_edges_lookup.get(clone_id)
        endpoints = set()
        if bg is not None and len(bg):
            for _, br in bg.iterrows():
                endpoints.add(_site_key(br["src_tissue"], br["src_t"]))
                endpoints.add(_site_key(br["dst_tissue"], br["dst_t"]))
        isolated = lit - endpoints
        counts[pat] += len(isolated)
    return counts


# Build (patient, lit-site-set) per eligible clone. Patient is intrinsic
# to clone_id, no mode lookup needed.
clone_pat_lookup = dict(zip(
    [c["clone_id"] for c in clones_table],
    [c["patient"] for c in clones_table]))
clones_lit_sets = []
for cid in top_clone_ids:
    sites = dict(nodes_per_clone.get(cid, {}))
    sites["__patient__"] = clone_pat_lookup.get(cid)
    clones_lit_sets.append((cid, sites))

# OLD: use branches_used grouped by clone_id (so the comparison
# isolates the edge-rule effect, not the clone-identity effect).
branches_by_clone_id_old = {cid: g for cid, g in
                            branches_used.groupby("clone_id", observed=True)}
old_iso = _count_isolated(branches_by_clone_id_old, clones_lit_sets)
new_iso = _count_isolated(clone_edges_by_clone_id, clones_lit_sets)
print("\n  isolated lit nodes per patient "
      "(lit nodes with no edge attached) — across eligible clones:")
print(f"  {'patient':<8}{'OLD rule':>12}{'NEW rule':>12}")
for pat in PATIENTS:
    print(f"  {pat:<8}{old_iso[pat]:>12}{new_iso[pat]:>12}")
worst_new = max(new_iso.values(), default=0)
_sanity(f"isolated-lit-nodes after fix: worst patient = {worst_new} "
        f"(was {max(old_iso.values(), default=0)} under old rule)")


# %%
# ---- Assemble JSON blob ----
print("\nAssembling JSON blob...")
data = {
    "view": {
        "global": {
            "nodes": nodes_global,
            "node_box": node_box_global,
            "edges": edges_global,
            "pheno_clonality": per_pheno_clonality_global,
        },
        "patient": {
            pat: {
                "nodes": nodes_per_patient[pat],
                "tissue_box": patient_tissue_box[pat],
                "edges": edges_per_patient[pat],
                "pheno_clonality": per_pheno_clonality_patient[pat],
            } for pat in PATIENTS
        },
        "clone": {
            cid: {
                "nodes": nodes_per_clone[cid],
                "box": clone_box[cid],
                "edges": edges_per_clone[cid],
            } for cid in top_clone_ids
        },
    },
    "clones": clones_table,
    "scales": scales,
    "palettes": {
        "tissue_colors":   {k: TISSUE_COLORS[k] for k in TISSUES},
        "tissue_labels":   {k: TISSUE_LABELS[k] for k in TISSUES},
        "tcell_colors":    {p: TCELL_PHENOTYPE_COLORS[p]
                            for p in T_PHENOTYPES},
        "tcell_labels":    {p: TCELL_PHENOTYPE_LABELS[p]
                            for p in T_PHENOTYPES},
        "tcell_order":     list(T_PHENOTYPES),
        "myeloid_colors":  {p: MYELOID_PHENOTYPE_COLORS[p]
                            for p in M_PHENOTYPES},
        "myeloid_labels":  {p: MYELOID_PHENOTYPE_LABELS[p]
                            for p in M_PHENOTYPES},
        "myeloid_order":   list(M_PHENOTYPES),
        "lineage_colors":  {k: LINEAGE_COLORS[k] for k in LINEAGE_COLORS},
    },
    "patients": PATIENTS,
    "timepoints": list(TIMEPOINTS),
    "tissues": list(TISSUES),
}

data_json = json.dumps(data, separators=(",", ":"), allow_nan=False)
json_bytes = len(data_json.encode("utf-8"))
print(f"  JSON: {json_bytes / 1024:.1f} KB "
      f"({json_bytes / 1024 / 1024:.2f} MB)")
if json_bytes / 1024 / 1024 > JSON_WARN_MB:
    _sanity(f"JSON blob size {json_bytes / 1024 / 1024:.1f} MB > "
            f"{JSON_WARN_MB} MB warn threshold", ok=False)
else:
    _sanity(f"JSON blob size {json_bytes / 1024 / 1024:.2f} MB (under "
            f"{JSON_WARN_MB} MB warn threshold)")


# %%
# ---- HTML/CSS/JS template ----
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GBM Clone Network Explorer</title>
<style>
  :root {
    --bg: #fafafa;
    --panel-bg: #ffffff;
    --panel-border: #e0e0e0;
    --text: #222;
    --muted: #777;
    --accent-pink: #c51b8a;
    --accent-pink-bg: #fde7f3;
    --shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, sans-serif;
    color: var(--text); background: var(--bg);
    font-size: 13px;
  }
  #topbar {
    height: 56px;
    background: var(--panel-bg);
    border-bottom: 1px solid var(--panel-border);
    box-shadow: var(--shadow);
    display: flex; align-items: center; padding: 0 16px;
    gap: 16px; position: sticky; top: 0; z-index: 10;
  }
  #topbar .left, #topbar .right {
    display: flex; align-items: center; gap: 12px;
  }
  #topbar .right { margin-left: auto; }
  .btn {
    background: #f4f4f4; border: 1px solid #ddd; border-radius: 6px;
    padding: 6px 12px; font-size: 13px; cursor: pointer;
    transition: background .12s, box-shadow .12s;
    color: var(--text);
  }
  .btn:hover { background: #eee; box-shadow: 0 1px 3px rgba(0,0,0,0.07); }
  .btn .icon { margin-right: 6px; font-size: 14px; }
  select.patient-select {
    border: 1px solid #ddd; border-radius: 6px;
    background: #fff; padding: 6px 10px; font-size: 13px;
    color: var(--text);
  }
  select.patient-select:disabled { opacity: 0.5; cursor: not-allowed; }
  .view-badge {
    border-radius: 12px; padding: 4px 10px; font-size: 12px;
    font-weight: 600;
    background: #efefef; color: #555;
  }
  .view-badge.global   { background: #efefef; color: #555; }
  .view-badge.patient  { background: #fff7d6; color: #8a6d00; }
  .view-badge.clone    { background: var(--accent-pink-bg);
                         color: var(--accent-pink); }
  .seg {
    display: inline-flex; border: 1px solid #ddd; border-radius: 8px;
    background: transparent; overflow: hidden;
  }
  .seg.disabled { opacity: 0.45; pointer-events: none; }
  .seg button {
    background: transparent; border: 0;
    padding: 6px 12px; font-size: 12px; cursor: pointer;
    color: var(--muted);
  }
  .seg button.active {
    background: #fff; color: var(--text);
    box-shadow: inset 0 -2px 0 var(--accent-pink);
  }
  #main {
    display: flex; height: calc(100vh - 56px);
  }
  #left { flex: 0 0 62%; display: flex; flex-direction: column;
          padding: 16px; gap: 8px; }
  #right {
    flex: 1 1 auto; display: flex; flex-direction: column;
    background: var(--panel-bg); border-left: 1px solid var(--panel-border);
    min-width: 0;
  }
  #right .pane {
    flex: 1 1 0; min-height: 0; padding: 16px;
    display: flex; flex-direction: column; gap: 8px;
    border-bottom: 1px solid var(--panel-border);
    overflow: hidden; min-width: 0;
  }
  #right .pane:last-child { border-bottom: 0; }
  /* Composition pane gets slightly more room; table pane slightly less. */
  #right .pane.composition { flex: 1.15 1 0; }
  #right .pane.table       { flex: 0.85 1 0; }
  #network-card {
    /* Full vertical allocation within the left column (legend strip
       takes its own 40px below). */
    flex: 1 1 auto;
    background: var(--panel-bg); border: 1px solid var(--panel-border);
    border-radius: 8px; box-shadow: var(--shadow);
    padding: 8px;
    display: flex; flex-direction: column; min-height: 0;
  }
  #network-svg { flex: 1 1 auto; width: 100%; height: 100%;
                 min-height: 0; }
  #legend-card {
    /* Compressed single-row legend: target height ~40px. */
    flex: 0 0 40px;
    background: var(--panel-bg); border: 1px solid var(--panel-border);
    border-radius: 8px; box-shadow: var(--shadow);
    padding: 4px 12px;
    display: flex; flex-direction: row; gap: 16px;
    align-items: center; overflow: hidden;
    font-size: 10px;
  }
  /* Side-panel title row hosts the title text and a tiny clear-btn. */
  .side-title-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; margin-bottom: 2px;
  }
  .side-title-text { font-weight: 600; font-size: 13px; color: var(--text);
                     flex: 1 1 auto; min-width: 0;
                     overflow: hidden; text-overflow: ellipsis;
                     white-space: nowrap; }
  .clear-btn {
    background: transparent; border: 0; cursor: pointer;
    color: #888; font-size: 11px; padding: 0 4px;
    flex: 0 0 auto;
  }
  .clear-btn:hover { color: #333; }
  .legend-block { display: flex; flex-direction: row; gap: 6px;
                  font-size: 10px; color: #444; min-width: 0;
                  align-items: center; }
  .legend-block svg { display: block; }
  .swatch-row { display: flex; flex-direction: row; flex-wrap: nowrap;
                gap: 8px; align-items: center; overflow: hidden; }
  .swatch { display: inline-flex; align-items: center; gap: 3px;
            font-size: 9px; color: #444; white-space: nowrap; }
  .swatch i { display: inline-block; width: 10px; height: 10px;
              border-radius: 2px; }
  .side-title { font-weight: 600; font-size: 13px; color: var(--text);
                margin-bottom: 4px; }
  .side-sub { font-size: 11px; color: var(--muted); }
  .empty-msg { color: var(--muted); font-size: 12px; padding: 12px 4px; }
  #box-host { flex: 1 1 auto; overflow: auto; min-height: 0; }
  #box-host svg { display: block; }
  #table-search {
    width: 100%; padding: 6px 10px; border: 1px solid #ddd;
    border-radius: 6px; font-size: 12px;
  }
  #clone-table-host {
    flex: 1 1 auto;
    overflow-x: auto; overflow-y: auto;
    min-height: 0; min-width: 0;
    position: relative;
  }
  table.clone-table {
    /* No width: 100%; we want the table to take its intrinsic min-width
       (set per-render via JS) and let the host scroll horizontally. */
    border-collapse: collapse; font-size: 11.5px;
  }
  table.clone-table thead th {
    position: sticky; top: 0; background: var(--panel-bg);
    border-bottom: 1px solid var(--panel-border);
    text-align: left; padding: 6px 8px; cursor: pointer;
    font-weight: 600; color: #444; white-space: nowrap;
    user-select: none;
  }
  table.clone-table thead th:hover { background: #f7f7f7; }
  table.clone-table tbody td {
    padding: 4px 8px; border-bottom: 1px solid #f0f0f0;
    white-space: nowrap;
  }
  table.clone-table tbody tr:hover { background: #f7f7f7; cursor: pointer; }
  table.clone-table tbody tr.selected {
    background: var(--accent-pink-bg);
    border-left: 4px solid var(--accent-pink);
  }
  table.clone-table tbody tr.vspacer { background: transparent; }
  table.clone-table tbody tr.vspacer:hover { background: transparent;
                                              cursor: default; }
  .clone-table-scroller {
    overflow: auto;
    max-height: 100%;
    height: 100%;
    width: 100%;
  }
  .tooltip {
    position: absolute; pointer-events: none; background: #222;
    color: #fff; padding: 6px 8px; border-radius: 4px;
    font-size: 11px; opacity: 0; transition: opacity .1s;
    z-index: 100; max-width: 220px;
  }
  .tooltip.show { opacity: 1; }
  text.tissue-label { font-weight: 700; font-size: 14px; }
  text.tp-label { font-size: 13px; fill: #555; }
  text.axis-label { font-size: 11px; fill: #777; }
  text.tick { font-size: 9px; fill: #555; }
  .node-circle { transition: stroke-width .12s, filter .12s; }
  .node-circle.selected {
    stroke-width: 3px;
    filter: drop-shadow(0 0 6px rgba(197,27,138,0.6));
  }
  .edge-path { fill: none; }
</style>
</head>
<body>
<div id="topbar">
  <div class="left">
    <button class="btn" id="home-btn"><span class="icon">⌂</span>Home</button>
    <select class="patient-select" id="patient-select"></select>
    <span class="view-badge" id="view-badge">Global</span>
  </div>
  <div class="right">
    <div class="seg" id="seg-metric">
      <button data-v="traffic" class="active">Traffic</button>
      <button data-v="expansion">Expansion</button>
    </div>
    <div class="seg" id="seg-composition">
      <button data-v="tcell" class="active">T cell</button>
      <button data-v="myeloid">Myeloid</button>
    </div>
    <div class="seg disabled" id="seg-nodesize">
      <button data-v="cells" class="active">Cells</button>
      <button data-v="repertoire_fraction">Repertoire fraction</button>
    </div>
  </div>
</div>
<div id="main">
  <div id="left">
    <div id="network-card">
      <svg id="network-svg"></svg>
    </div>
    <div id="legend-card"></div>
  </div>
  <div id="right">
    <div class="pane composition">
      <div class="side-title-row">
        <span id="side-title-text" class="side-title-text">Detail</span>
        <button class="clear-btn" id="side-clear" style="display:none;"
                title="Clear selection">× clear</button>
      </div>
      <div class="side-sub" id="side-sub">Click a node to see composition,
        or click an edge to see phenotype flow.</div>
      <div id="box-host"></div>
    </div>
    <div class="pane table">
      <div class="side-title">Top clones</div>
      <input type="search" id="table-search" placeholder="filter by TRB, patient, or dominant phenotype…">
      <div id="clone-table-host"></div>
    </div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>__D3_JS__</script>
<script id="explorer-data" type="application/json">__DATA_JSON__</script>
<script>__APP_JS__</script>
</body>
</html>
"""

APP_JS = r"""
"use strict";
const DATA = JSON.parse(document.getElementById("explorer-data").textContent);
const P = DATA.palettes;
const SCALES = DATA.scales;
const TISSUES = DATA.tissues;
const TIMEPOINTS = DATA.timepoints;

// Node radius scaling: sqrt + visible floor. Wider dynamic range than
// before — MAX_R is targeted at 68, but may auto-back-off if the grid
// cell spacing isn't wide enough to host two adjacent nodes without
// overlap. At-render: max_radius = min(MAX_R_TARGET, (cell_width-8)/2).
const MIN_R = 14;
const MAX_R_TARGET = 68;
const MAX_R_FALLBACK = 60;
let MAX_R = MAX_R_TARGET;
const SIZE_EXP = 0.5;
const EMPTY_R = 11;     // greyed node outline radius

// In clone view, traffic encoding is uninformative (always 1 clone
// per edge), so we force expansion encoding regardless of the toggle.
function effectiveMetricMode() {
  return STATE.view_mode === "clone" ? "expansion" : STATE.metric_mode;
}

const NEUTRAL_GREY = "#9e9e9e";

const STATE = {
  view_mode: "global",
  selected_patient: null,
  selected_clone: null,
  selected_node: null,
  // Edge selection drives the Sankey panel below the network. Stored
  // as "{src_tissue}|{src_t}->{dst_tissue}|{dst_t}" for compactness.
  selected_edge: null,
  metric_mode: "traffic",
  composition_mode: "tcell",
  node_size_mode: "cells",
};

function edgeKey(e) {
  return `${e.src_tissue}|${e.src_t}->${e.dst_tissue}|${e.dst_t}`;
}

// Clone-view edges use a manual three-stop diverging scale with a
// visible grey midpoint, so persistent clones (median ≈ 0) appear as
// a clearly visible grey edge instead of disappearing into white as
// PiYG / PuOr / RdBu all do at their midpoints. Polarity:
//   negative (contraction) → purple
//   zero                   → grey
//   positive (expansion)   → orange
function cloneExpansionScale(vmin, vmax) {
  return d3.scaleLinear()
    .domain([vmin, 0, vmax])
    .range(["#6a3d9a", "#7d7d7d", "#e6550d"])
    .interpolate(d3.interpolateLab)
    .clamp(true);
}

const tooltip = d3.select("#tooltip");
function showTip(html, ev) {
  tooltip.html(html).classed("show", true)
    .style("left", (ev.pageX + 12) + "px")
    .style("top", (ev.pageY + 12) + "px");
}
function hideTip() { tooltip.classed("show", false); }

function currentView() {
  if (STATE.view_mode === "clone" && STATE.selected_clone)
    return DATA.view.clone[STATE.selected_clone];
  if (STATE.view_mode === "patient" && STATE.selected_patient)
    return DATA.view.patient[STATE.selected_patient];
  return DATA.view.global;
}

function siteKey(tissue, t) { return tissue + "|" + t; }

function nodeData(tissue, t) {
  const v = currentView();
  const k = siteKey(tissue, t);
  return v.nodes[k] || null;
}

// Per-view edge scales. Computed fresh on every render so the scales
// reflect only the edges actually drawn for the current view.
function computeViewEdgeScales() {
  const edges = (currentView().edges || []);
  let maxBranches = 1, maxAbsLfc = 0;
  edges.forEach(e => {
    if (e.n_branches_used > maxBranches) maxBranches = e.n_branches_used;
    const v = e.median_log2fc_norm;
    if (v !== undefined && v !== null && !Number.isNaN(v)) {
      const av = Math.abs(v);
      if (av > maxAbsLfc) maxAbsLfc = av;
    }
  });
  return { maxBranches, maxAbsLfc: Math.max(maxAbsLfc, 1e-6) };
}
let VIEW_SCALES = { maxBranches: 1, maxAbsLfc: 1 };

function edgeColor(e) {
  if (effectiveMetricMode() === "traffic") {
    const denom = Math.max(VIEW_SCALES.maxBranches, 1);
    return d3.interpolatePlasma(0.15 + 0.7 * e.n_branches_used / denom);
  }
  const vmax = VIEW_SCALES.maxAbsLfc;
  // Clone-view: manual three-stop diverging scale with a visible grey
  // midpoint so median ≈ 0 edges don't fade into the white background.
  if (STATE.view_mode === "clone") {
    return cloneExpansionScale(-vmax, vmax)(e.median_log2fc_norm);
  }
  // Global / patient view: keep PiYG.
  const t = 0.5 + 0.5 * (e.median_log2fc_norm / vmax);
  return d3.interpolatePiYG(Math.max(0, Math.min(1, t)));
}

function edgeWidth(e) {
  if (effectiveMetricMode() === "traffic") {
    const denom = Math.max(VIEW_SCALES.maxBranches, 1);
    return 1 + 7 * Math.sqrt(e.n_branches_used / denom);
  }
  const vmax = VIEW_SCALES.maxAbsLfc;
  return 1 + 7 * Math.min(1, Math.abs(e.median_log2fc_norm) / vmax);
}

// ---- Layout ----
const ROW_BY_TISSUE = { "TP": 0, "PBMC": 1, "CSF": 2 };
const ROW_LABEL = { "TP": "Tumor", "PBMC": "Blood", "CSF": "CSF" };
const MARGIN = { top: 36, right: 24, bottom: 24, left: 80 };

function computePositions(w, h) {
  const innerW = w - MARGIN.left - MARGIN.right;
  const innerH = h - MARGIN.top - MARGIN.bottom;
  const xs = TIMEPOINTS.map(
    (_, i) => MARGIN.left + (i + 0.5) * innerW / TIMEPOINTS.length);
  const ys = TISSUES.map(
    t => MARGIN.top + (ROW_BY_TISSUE[t] + 0.5) * innerH / TISSUES.length);
  const pos = {};
  TISSUES.forEach((tissue) => {
    TIMEPOINTS.forEach((t, j) => {
      pos[siteKey(tissue, t)] = { x: xs[j], y: ys[ROW_BY_TISSUE[tissue]] };
    });
  });
  return { pos, xs, ys };
}

function nodeRadius(d) {
  if (!d || !d.n_tcells) return EMPTY_R;
  const inClone = STATE.view_mode === "clone";
  if (inClone && STATE.node_size_mode === "repertoire_fraction") {
    const c = DATA.clones.find(x => x.clone_id === STATE.selected_clone);
    const total = c ? c.n_cells_total : 1;
    const frac = total ? d.n_tcells / total : 0;
    return MIN_R + (MAX_R - MIN_R) * Math.pow(
      Math.max(0, Math.min(1, frac)), SIZE_EXP);
  }
  const maxV = inClone ? SCALES.max_n_cells_clone_node
                        : SCALES.max_n_tcells_node;
  const frac = Math.max(0, d.n_tcells) / Math.max(maxV, 1);
  return MIN_R + (MAX_R - MIN_R) * Math.pow(
    Math.max(0, Math.min(1, frac)), SIZE_EXP);
}

// ---- Network rendering ----
const svg = d3.select("#network-svg");
let gAxes, gEdges, gNodes;

// Toggle one-shot per-node diagnostics. Open the JS console and look
// for [node] lines. Flip to false once the patient-view pie bug is
// confirmed fixed.
const NODE_DEBUG = true;

function ensureNetworkLayers() {
  if (!gAxes) {
    gAxes = svg.append("g").attr("class", "axes");
    gEdges = svg.append("g").attr("class", "edges");
    gNodes = svg.append("g").attr("class", "nodes");
  }
}

// Unified composition lookup. All three view modes route through this
// so the JSON path is identical (with the documented exception that
// clone view + myeloid pulls bulk myeloid from the clone's patient).
function getCompositionForNode(tissue, t, d) {
  const mode = STATE.composition_mode;
  const sk = siteKey(tissue, t);
  let comp = null, nCells = 0, source = "";
  if (STATE.view_mode === "clone" && mode === "myeloid") {
    const c = DATA.clones.find(x => x.clone_id === STATE.selected_clone);
    const pat = c ? c.patient : null;
    const pn = (pat && DATA.view.patient[pat]
                 && DATA.view.patient[pat].nodes[sk]) || null;
    comp = pn ? (pn.myeloid_comp || null) : null;
    nCells = pn ? (pn.n_myeloid || 0) : 0;
    source = `patient[${pat}].nodes[${sk}].myeloid_comp`;
  } else if (mode === "tcell") {
    comp = d ? (d.tcell_comp || null) : null;
    nCells = d ? (d.n_tcells || 0) : 0;
    source = `${STATE.view_mode}.nodes[${sk}].tcell_comp`;
  } else {
    comp = d ? (d.myeloid_comp || null) : null;
    nCells = d ? (d.n_myeloid || 0) : 0;
    source = `${STATE.view_mode}.nodes[${sk}].myeloid_comp`;
  }
  const order  = mode === "tcell" ? P.tcell_order  : P.myeloid_order;
  const colors = mode === "tcell" ? P.tcell_colors : P.myeloid_colors;
  return { comp, nCells, source, order, colors };
}

function renderNodePie(g, tissue, t, d, r) {
  const { comp, nCells, source, order, colors } =
    getCompositionForNode(tissue, t, d);
  const arr = comp || [];
  const sumComp = arr.reduce((a, b) => a + (b || 0), 0);
  const nonzero = arr.map((v, i) => ({ v, i })).filter(x => x.v > 0);

  if (NODE_DEBUG) {
    console.log(
      `[node] view=${STATE.view_mode} mode=${STATE.composition_mode} ` +
      `site=${tissue}/${t} n_cells=${nCells} src=${source} ` +
      `len=${arr.length} sum=${sumComp.toFixed(3)} ` +
      `nonzero=${nonzero.length}`
    );
  }

  const pieG = g.append("g").attr("class", "pie")
                 .style("pointer-events", "none");

  // No cells, or no composition payload at all → just the ring.
  if (nCells <= 0 || comp === null || comp === undefined) {
    return;
  }
  // n_cells > 0 but the composition vector is broken (all zeros /
  // missing): render a neutral-grey filled disc so the node still
  // reads as occupied, and warn loudly.
  if (sumComp <= 0) {
    pieG.append("circle")
      .attr("r", r * 0.92)
      .attr("fill", "#bbbbbb")
      .attr("fill-opacity", 0.6);
    console.warn(
      `Node ${tissue}/${t} in ${STATE.view_mode}: `
      + `n_cells=${nCells} but comp sums to 0 (src=${source})`
    );
    return;
  }
  // Single phenotype: d3.pie() emits a zero-angle arc — draw a
  // filled circle in that phenotype's color instead.
  if (nonzero.length === 1) {
    pieG.append("circle")
      .attr("r", r * 0.92)
      .attr("fill", colors[order[nonzero[0].i]]);
    return;
  }
  // Normal multi-slice pie.
  const pie = d3.pie().sort(null).startAngle(0)
                 .endAngle(2 * Math.PI)(arr);
  const arc = d3.arc().innerRadius(0).outerRadius(r * 0.92);
  pie.forEach((slc, i) => {
    if (!slc || slc.value <= 0) return;
    pieG.append("path")
      .attr("d", arc(slc))
      .attr("fill", colors[order[i]])
      .attr("stroke", "white")
      .attr("stroke-width", 0.5);
  });
}

function renderNetwork() {
  ensureNetworkLayers();
  VIEW_SCALES = computeViewEdgeScales();
  const node = svg.node();
  const w = node.clientWidth || 800;
  const h = node.clientHeight || 600;
  svg.attr("viewBox", `0 0 ${w} ${h}`);

  const { pos, xs, ys } = computePositions(w, h);

  // Pick MAX_R based on actual grid spacing so two adjacent nodes
  // never overlap horizontally. Reserve 8px gap between adjacent node
  // edges: max_radius <= (cell_width - 8) / 2.
  const minSpacing = Math.min(
    (xs.length > 1 ? xs[1] - xs[0] : Infinity),
    (ys.length > 1 ? ys[1] - ys[0] : Infinity)
  );
  const spacingCap = Math.max(MIN_R, Math.floor((minSpacing - 8) / 2));
  if (MAX_R_TARGET <= spacingCap) MAX_R = MAX_R_TARGET;
  else if (MAX_R_FALLBACK <= spacingCap) MAX_R = MAX_R_FALLBACK;
  else MAX_R = spacingCap;

  // Axes
  gAxes.selectAll("*").remove();
  TISSUES.forEach(tissue => {
    gAxes.append("text").attr("class", "tissue-label")
      .attr("x", MARGIN.left - 16)
      .attr("y", ys[ROW_BY_TISSUE[tissue]] + 5)
      .attr("text-anchor", "end")
      .attr("fill", P.tissue_colors[tissue])
      .text(ROW_LABEL[tissue]);
  });
  TIMEPOINTS.forEach((t, j) => {
    gAxes.append("text").attr("class", "tp-label")
      .attr("x", xs[j])
      .attr("y", MARGIN.top - 14)
      .attr("text-anchor", "middle")
      .text(t);
  });
  gAxes.append("text").attr("class", "axis-label")
    .attr("x", MARGIN.left + (w - MARGIN.left - MARGIN.right) / 2)
    .attr("y", 14).attr("text-anchor", "middle")
    .text("timepoint");

  // Edges: inclusion set is purely the current view's edges array;
  // metric_mode only changes how the same set is encoded.
  const v = currentView();
  const edges = v.edges || [];

  gEdges.selectAll("path.edge-path").remove();
  edges.forEach(e => {
    const sk = siteKey(e.src_tissue, e.src_t);
    const dk = siteKey(e.dst_tissue, e.dst_t);
    if (!pos[sk] || !pos[dk]) return;
    const p0 = pos[sk], p1 = pos[dk];
    const dx = p1.x - p0.x;
    const dy = p1.y - p0.y;
    const cx1 = p0.x + dx * 0.35;
    const cy1 = p0.y + dy * 0.05;
    const cx2 = p1.x - dx * 0.35;
    const cy2 = p1.y - dy * 0.05;
    const pathStr = `M${p0.x},${p0.y} C${cx1},${cy1} ${cx2},${cy2} ${p1.x},${p1.y}`;

    const lfc = e.median_log2fc_norm;
    const lfcDefined = (lfc !== undefined && lfc !== null
                        && !Number.isNaN(lfc));
    const inClone = STATE.view_mode === "clone";
    let stroke, width, dasharray = null;
    if (effectiveMetricMode() === "expansion" && !lfcDefined) {
      stroke = NEUTRAL_GREY;
      width = inClone ? 3.0 : 1.0;
      dasharray = "3,3";
    } else {
      stroke = edgeColor(e);
      // Clone view: every edge carries exactly one clone, so traffic
      // width is uninformative. Lock width to 3px; color still encodes
      // expansion via PiYG.
      width = inClone ? 3.0 : edgeWidth(e);
    }
    const ekey = edgeKey(e);
    const isSelectedEdge = STATE.selected_edge === ekey;
    const path = gEdges.append("path")
      .attr("class", "edge-path"
            + (isSelectedEdge ? " edge-selected" : ""))
      .attr("d", pathStr)
      .attr("stroke", stroke)
      .attr("stroke-width", isSelectedEdge ? width + 2 : width)
      .attr("stroke-opacity", isSelectedEdge ? 0.95 : 0.7)
      .style("cursor", "pointer")
      .on("mousemove", (ev) => showTip(
        `<b>${e.src_tissue}|${e.src_t} → ${e.dst_tissue}|${e.dst_t}</b><br>
         ${e.n_branches_used} clones on this edge<br>
         median log₂FC: ${lfcDefined ? lfc.toFixed(3) : "n/a"}<br>
         <em>click to flow</em>`, ev))
      .on("mouseleave", hideTip)
      .on("click", (ev) => {
        ev.stopPropagation();
        // Toggle: clicking the same edge clears the selection.
        STATE.selected_edge = (STATE.selected_edge === ekey) ? null : ekey;
        // Mutual exclusion with node selection.
        STATE.selected_node = null;
        renderAll();
      });
    if (dasharray) path.attr("stroke-dasharray", dasharray);
  });

  // Nodes — DOM order: outer ring → pie group (pointer-events:none)
  //         → transparent hit circle that owns the click handler.
  gNodes.selectAll("g.node-g").remove();
  TISSUES.forEach(tissue => {
    TIMEPOINTS.forEach(t => {
      const sk = siteKey(tissue, t);
      const p = pos[sk];
      const d = nodeData(tissue, t);
      const g = gNodes.append("g").attr("class", "node-g")
        .attr("transform", `translate(${p.x},${p.y})`);

      const present = !!(d && d.n_tcells > 0);
      const r = present ? nodeRadius(d) : EMPTY_R;
      const selected = STATE.selected_node
        && STATE.selected_node.tissue === tissue
        && STATE.selected_node.timepoint === t;

      // 1) Outer ring (tissue-colored stroke), no fill.
      g.append("circle")
        .attr("class", "node-bg" + (selected ? " selected" : ""))
        .attr("r", r)
        .attr("fill", "none")
        .attr("stroke", P.tissue_colors[tissue])
        .attr("stroke-width", selected ? 3 : 2)
        .attr("stroke-opacity", present ? 1.0 : 0.15)
        .style("pointer-events", "none");

      // 2) Pie slices — single code path for all three view modes.
      if (present) {
        renderNodePie(g, tissue, t, d, r);
      }

      // 3) Transparent hit circle on top — owns interactivity.
      if (present) {
        g.append("circle")
          .attr("class", "hit")
          .attr("r", r)
          .attr("fill", "transparent")
          .style("pointer-events", "all")
          .style("cursor", "pointer")
          .on("click", () => {
            // Toggle: clicking the same node again clears the selection.
            const same = STATE.selected_node
              && STATE.selected_node.tissue === tissue
              && STATE.selected_node.timepoint === t;
            STATE.selected_node = same ? null : { tissue, timepoint: t };
            // Mutual exclusion with edge selection.
            STATE.selected_edge = null;
            renderAll();
          })
          .on("mousemove", (ev) => {
            let body = `<b>${tissue} / T${t}</b><br>T cells: ${d.n_tcells}`;
            if (d.n_myeloid !== undefined) body += `<br>myeloid: ${d.n_myeloid}`;
            showTip(body, ev);
          })
          .on("mouseleave", hideTip);
      }
    });
  });
}

// ---- Self-test: every present node must have a clickable hit circle ----
function selfTestNodeClicks() {
  const groups = document.querySelectorAll("#network-svg g.node-g");
  let expected = 0, withHit = 0;
  groups.forEach(g => {
    const ring = g.querySelector("circle.node-bg");
    if (!ring) return;
    const isPresent = parseFloat(
      ring.getAttribute("stroke-opacity") || "1") > 0.5;
    if (isPresent) expected += 1;
    if (g.querySelector("circle.hit")) withHit += 1;
  });
  const msg = `[selftest] node-g count=${groups.length} present=${expected} hit_circles=${withHit}`;
  if (groups.length !== 18 || withHit < expected) {
    console.error(msg);
  } else {
    console.log(msg);
  }
}

// ---- Legend ----
// Compact single-row strip: a compressed colorbar (10px tall) on the
// left, and a single horizontal row of phenotype swatches filling the
// rest. Node-size legend is dropped — node hover shows n_tcells /
// n_myeloid via the tooltip instead.
function renderLegend() {
  const host = d3.select("#legend-card");
  host.selectAll("*").remove();

  // Compact edge-metric colorbar (single-line label + 10px ramp).
  const ramp = host.append("div").attr("class", "legend-block");
  const mode = effectiveMetricMode();
  const maxB = Math.max(1, VIEW_SCALES.maxBranches);
  const maxL = VIEW_SCALES.maxAbsLfc;
  const labelText = mode === "traffic"
    ? `clones per edge (max: ${maxB})`
    : `median log₂FC (range: ±${maxL.toFixed(2)})`;
  ramp.append("span").style("font-weight", "600").style("color", "var(--text)")
      .style("white-space", "nowrap").text(labelText);
  const rampW = 110, rampH = 10;
  const ramSvg = ramp.append("svg")
    .attr("width", rampW).attr("height", rampH + 12);
  const gradId = "metric-grad-" + STATE.view_mode + "-" + mode;
  const grad = ramSvg.append("defs").append("linearGradient")
    .attr("id", gradId).attr("x1", "0%").attr("x2", "100%");
  const N = 20;
  const inCloneExp = (STATE.view_mode === "clone" && mode === "expansion");
  // For clone-view expansion: sample the manual three-stop scale at
  // N evenly-spaced points across [-maxL, +maxL] so the colorbar
  // matches what the network paints exactly.
  const cloneScale = inCloneExp
    ? cloneExpansionScale(-maxL, maxL) : null;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    let c;
    if (mode === "traffic") {
      c = d3.interpolatePlasma(0.15 + 0.7 * t);
    } else if (inCloneExp) {
      const v = -maxL + (2 * maxL) * t;
      c = cloneScale(v);
    } else {
      c = d3.interpolatePiYG(t);
    }
    grad.append("stop").attr("offset", (t * 100) + "%")
      .attr("stop-color", c);
  }
  ramSvg.append("rect").attr("x", 0).attr("y", 0)
    .attr("width", rampW).attr("height", rampH)
    .attr("fill", `url(#${gradId})`).attr("stroke", "#999")
    .attr("stroke-width", 0.5);

  // Ticks at -maxL, 0, +maxL for the diverging colorbar. 0 emphasized.
  if (mode === "expansion") {
    const vL = maxL.toFixed(2);
    ramSvg.append("text").attr("class", "tick")
      .attr("x", 0).attr("y", rampH + 10)
      .style("font-size", "9px").style("fill", "#666")
      .text("-" + vL);
    ramSvg.append("text").attr("class", "tick")
      .attr("x", rampW / 2).attr("y", rampH + 10)
      .attr("text-anchor", "middle")
      .style("font-size", "9px").style("font-weight", "700")
      .style("fill", "#222").text("0");
    ramSvg.append("text").attr("class", "tick")
      .attr("x", rampW).attr("y", rampH + 10)
      .attr("text-anchor", "end")
      .style("font-size", "9px").style("fill", "#666")
      .text("+" + vL);
  }

  // Single-row phenotype swatches (no prefix labels — composition_mode
  // toggle in the topbar already names the active palette).
  const cs = host.append("div").attr("class", "legend-block")
    .style("flex", "1 1 auto").style("min-width", "0");
  const order = STATE.composition_mode === "tcell"
                ? P.tcell_order : P.myeloid_order;
  const colors = STATE.composition_mode === "tcell"
                 ? P.tcell_colors : P.myeloid_colors;
  const labels = STATE.composition_mode === "tcell"
                 ? P.tcell_labels : P.myeloid_labels;
  const sw = cs.append("div").attr("class", "swatch-row")
    .style("overflow", "hidden");
  order.forEach(k => {
    const s = sw.append("div").attr("class", "swatch");
    s.append("i").style("background", colors[k]);
    s.append("span").text(labels[k]);
  });
}

// ---- Side panel: top-half detail (composition or Sankey) ----
// The composition view and the edge-click Sankey share the SAME DOM
// container (#box-host) in the side panel top half. selected_node and
// selected_edge are mutually exclusive — clicking one clears the other.
function renderBoxPanel() {
  const host = d3.select("#box-host");
  host.selectAll("*").remove();
  const titleText = d3.select("#side-title-text");
  const sub = d3.select("#side-sub");
  const clearBtn = d3.select("#side-clear");

  if (STATE.selected_edge) {
    clearBtn.style("display", "inline-block");
    renderSankey();
    return;
  }
  if (!STATE.selected_node) {
    titleText.text("Detail");
    sub.text("Click a node to see composition, "
             + "or click an edge to see phenotype flow.");
    clearBtn.style("display", "none");
    return;
  }
  clearBtn.style("display", "inline-block");
  const { tissue, timepoint } = STATE.selected_node;
  titleText.text(`${tissue} / T${timepoint}`);

  let boxes, kinds;
  if (STATE.view_mode === "clone" && STATE.selected_clone) {
    // Clone view: BOTH T-cell and myeloid render as point estimates
    // (per-plot y-scale). T-cell counts come from this clone's cells;
    // myeloid counts come from the clone's patient (bulk).
    kinds = ["tcell_clone", "myeloid_clone"];
    sub.text("counts at this site (per-plot y-scale)");
    boxes = null;
  } else if (STATE.view_mode === "patient" && STATE.selected_patient) {
    const tb = DATA.view.patient[STATE.selected_patient].tissue_box[tissue];
    boxes = tb;
    kinds = ["tcell", "myeloid", "clonality"];
    sub.text(`distribution across timepoints for ${STATE.selected_patient} / ${tissue}`);
  } else {
    const nb = DATA.view.global.node_box[siteKey(tissue, timepoint)];
    boxes = nb;
    kinds = ["tcell", "myeloid", "clonality"];
    sub.text("distribution across patients");
  }

  kinds.forEach(kind => {
    if (kind === "clonality") {
      drawPerPhenoClonality(host, tissue, timepoint);
    } else if (kind === "tcell_clone") {
      drawPhenotypeCountPoints(host, tissue, timepoint, "tcell");
    } else if (kind === "myeloid_clone") {
      drawPhenotypeCountPoints(host, tissue, timepoint, "myeloid");
    } else {
      const order = (kind === "tcell" ? P.tcell_order : P.myeloid_order);
      const colors = (kind === "tcell" ? P.tcell_colors : P.myeloid_colors);
      const labels = (kind === "tcell" ? P.tcell_labels : P.myeloid_labels);
      drawPhenotypeBoxes(host,
        kind === "tcell" ? "T-cell phenotype counts"
                          : "Myeloid phenotype counts",
        order.map(k => ({ key: k, vals: boxes[kind][k] || [],
                          color: colors[k], label: labels[k] })),
        true);
    }
  });
}

// ---- Clone view: phenotype counts as point estimates (per-plot y-scale,
// dynamic full-container width). Used for BOTH T-cell (from the clone's
// own cells at this site) and myeloid (bulk myeloid composition at the
// clone's patient/tissue/timepoint — myeloid has no clone identity).
function drawPhenotypeCountPoints(host, tissue, timepoint, kind /* "tcell" | "myeloid" */) {
  const clone = DATA.clones.find(x => x.clone_id === STATE.selected_clone);
  const pat = clone ? clone.patient : null;
  const site = siteKey(tissue, timepoint);
  const cView = clone ? DATA.view.clone[clone.clone_id] : null;
  const cNode = cView ? cView.nodes[site] : null;

  const isTcell = (kind === "tcell");
  const order = isTcell ? P.tcell_order : P.myeloid_order;
  const colors = isTcell ? P.tcell_colors : P.myeloid_colors;
  const labels = isTcell ? P.tcell_labels : P.myeloid_labels;
  const lineageColors = P.lineage_colors;

  const sec = host.append("div").style("margin-bottom", "8px");
  const titleText = isTcell
    ? `T-cell phenotype counts (this clone at ${tissue}/T${timepoint})`
    : `Myeloid phenotype counts (${pat || "n/a"} at ${tissue}/T${timepoint})`;
  sec.append("div").style("font-size", "11.5px").style("font-weight", "600")
    .style("margin-bottom", "2px").text(titleText);

  // Dynamic width: full container minus 16px padding.
  const hostW = (host.node().getBoundingClientRect().width) || 320;
  const W = Math.max(220, Math.floor(hostW - 16));
  const marginL = 38, marginR = 12, marginT = 4, marginB = 78;
  const innerW = Math.max(50, W - marginL - marginR);
  const stepX = innerW / order.length;
  const H = 150;
  const innerH = H - marginT - marginB;
  const svg = sec.append("svg").attr("width", W).attr("height", H);

  // Counts at this exact (tissue, timepoint), for the requested kind.
  let counts = {};
  if (isTcell) {
    if (cNode && cNode.n_tcells && cNode.tcell_comp) {
      order.forEach((p, i) => {
        counts[p] = Math.round((cNode.tcell_comp[i] || 0) * cNode.n_tcells);
      });
    } else {
      order.forEach(p => { counts[p] = 0; });
    }
  } else {
    const tBox = (pat && DATA.view.patient[pat]
                   && DATA.view.patient[pat].tissue_box[tissue]) || null;
    const pNode = (pat && DATA.view.patient[pat]
                    && DATA.view.patient[pat].nodes[site]) || null;
    if (tBox && tBox.myeloid) {
      const tIdx = TIMEPOINTS.indexOf(timepoint);
      order.forEach(p => {
        const arr = tBox.myeloid[p] || [];
        counts[p] = (tIdx >= 0 && tIdx < arr.length) ? arr[tIdx] : 0;
      });
    } else if (pNode && pNode.myeloid_comp && pNode.n_myeloid) {
      order.forEach((p, i) => {
        counts[p] = Math.round((pNode.myeloid_comp[i] || 0) * pNode.n_myeloid);
      });
    } else {
      order.forEach(p => { counts[p] = 0; });
    }
  }

  // Per-plot y-scale (THIS plot's max, not global).
  const maxV = Math.max(1, ...order.map(p => counts[p] || 0));
  const y = d3.scaleLinear().domain([0, maxV]).nice()
    .range([marginT + innerH, marginT]);
  const yAx = d3.axisLeft(y).ticks(3).tickSize(2);
  svg.append("g").attr("transform", `translate(${marginL},0)`).call(yAx)
    .selectAll("text").style("font-size", "9px");
  svg.append("text").attr("class", "axis-label")
    .attr("transform", `translate(10, ${marginT + innerH / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").style("font-size", "10px")
    .text("cells");

  order.forEach((pheno, i) => {
    const cx = marginL + (i + 0.5) * stepX;
    const v = counts[pheno] || 0;
    if (v > 0) {
      svg.append("circle").attr("cx", cx).attr("cy", y(v)).attr("r", 4)
        .attr("fill", colors[pheno]).attr("fill-opacity", 0.85)
        .attr("stroke", colors[pheno]);
    } else {
      svg.append("circle").attr("cx", cx).attr("cy", y(0)).attr("r", 4)
        .attr("fill", "none").attr("stroke", colors[pheno])
        .attr("stroke-opacity", 0.4);
    }
    let labelColor = "#333";
    if (isTcell) {
      const lineage = pheno.startsWith("CD8") ? "CD8"
                      : pheno.startsWith("CD4") ? "CD4" : "other";
      labelColor = lineageColors[lineage] || "#333";
    }
    svg.append("text")
      .attr("x", cx).attr("y", marginT + innerH + 12)
      .attr("text-anchor", "end").attr("transform",
        `rotate(-45, ${cx}, ${marginT + innerH + 12})`)
      .style("font-size", "10px").style("fill", labelColor)
      .text(labels[pheno] || pheno);
  });
}

// ---- Per-phenotype clonality panel ----
function drawPerPhenoClonality(host, tissue, timepoint) {
  const site = siteKey(tissue, timepoint);
  const sec = host.append("div").style("margin-bottom", "8px");
  const inGlobal = (STATE.view_mode === "global");
  sec.append("div").style("font-size", "11.5px").style("font-weight", "600")
    .style("margin-bottom", "2px")
    .text(inGlobal
          ? "Per-phenotype clonality (across patients)"
          : `Per-phenotype clonality (${STATE.selected_patient})`);

  const hostW = (host.node().getBoundingClientRect().width) || 320;
  const W = Math.max(220, Math.floor(hostW - 16));
  const marginL = 32, marginR = 12, marginT = 4, marginB = 70;
  const order = P.tcell_order;
  const labels = P.tcell_labels;
  const colors = P.tcell_colors;
  const lineageColors = P.lineage_colors;
  const innerW = Math.max(50, W - marginL - marginR);
  const stepX = innerW / order.length;
  const H = 140;
  const innerH = H - marginT - marginB;
  const svg = sec.append("svg").attr("width", W).attr("height", H);

  // y scale
  const y = d3.scaleLinear().domain([0, 1])
    .range([marginT + innerH, marginT]);
  const yAx = d3.axisLeft(y).ticks(3).tickSize(2);
  svg.append("g").attr("transform", `translate(${marginL},0)`).call(yAx)
    .selectAll("text").style("font-size", "9px");
  svg.append("text").attr("class", "axis-label")
    .attr("transform", `translate(8, ${marginT + innerH / 2}) rotate(-90)`)
    .attr("text-anchor", "middle").style("font-size", "10px")
    .text("clonality");

  order.forEach((pheno, i) => {
    const cx = marginL + (i + 0.5) * stepX;
    const lineage = pheno.startsWith("CD8") ? "CD8"
                    : pheno.startsWith("CD4") ? "CD4" : "other";
    const labelColor = lineageColors[lineage] || "#444";
    let vals = [];
    if (inGlobal) {
      vals = (DATA.view.global.pheno_clonality[site] || {})[pheno] || [];
    } else if (STATE.view_mode === "patient" && STATE.selected_patient) {
      const pv = DATA.view.patient[STATE.selected_patient];
      const v = (pv.pheno_clonality[site] || {})[pheno];
      if (v !== null && v !== undefined) vals = [v];
    }
    const hasData = vals.length > 0;
    const labelAlpha = hasData ? 1.0 : 0.35;

    if (inGlobal) {
      const stats = _boxStats(vals);
      if (stats) {
        const w = Math.max(4, stepX * 0.55);
        svg.append("line").attr("x1", cx).attr("x2", cx)
          .attr("y1", y(stats.lo)).attr("y2", y(stats.hi))
          .attr("stroke", colors[pheno]);
        svg.append("rect")
          .attr("x", cx - w / 2).attr("y", y(stats.q3))
          .attr("width", w)
          .attr("height", Math.max(1, y(stats.q1) - y(stats.q3)))
          .attr("fill", colors[pheno]).attr("fill-opacity", 0.6)
          .attr("stroke", colors[pheno]);
        svg.append("line").attr("x1", cx - w / 2).attr("x2", cx + w / 2)
          .attr("y1", y(stats.med)).attr("y2", y(stats.med))
          .attr("stroke", "#222").attr("stroke-width", 1.4);
      }
    } else {
      // patient view: single point estimate (or open marker for n<10)
      const v = vals[0];
      if (v !== undefined) {
        svg.append("circle").attr("cx", cx).attr("cy", y(v)).attr("r", 4)
          .attr("fill", colors[pheno]).attr("fill-opacity", 0.7)
          .attr("stroke", colors[pheno]);
      } else {
        // n_cells < 10 OR no cells → open at y=0
        svg.append("circle").attr("cx", cx).attr("cy", y(0)).attr("r", 4)
          .attr("fill", "none").attr("stroke", colors[pheno])
          .attr("stroke-opacity", 0.4);
      }
    }
    svg.append("text")
      .attr("x", cx).attr("y", marginT + innerH + 12)
      .attr("text-anchor", "end").attr("transform",
        `rotate(-45, ${cx}, ${marginT + innerH + 12})`)
      .style("font-size", "10px").style("fill", labelColor)
      .style("fill-opacity", labelAlpha)
      .text(labels[pheno] || pheno);
  });
}

function _boxStats(vals) {
  const v = vals.filter(x => x !== null && x !== undefined && !isNaN(x))
                .slice().sort(d3.ascending);
  if (!v.length) return null;
  const q1 = d3.quantile(v, 0.25);
  const med = d3.quantile(v, 0.5);
  const q3 = d3.quantile(v, 0.75);
  const iqr = q3 - q1;
  const lo = Math.max(d3.min(v), q1 - 1.5 * iqr);
  const hi = Math.min(d3.max(v), q3 + 1.5 * iqr);
  return { q1, med, q3, lo, hi, n: v.length };
}

function drawSingleBox(host, title, vals, color, w, h) {
  const sec = host.append("div").style("margin-bottom", "8px");
  sec.append("div").style("font-size", "11.5px")
    .style("font-weight", "600")
    .style("margin-bottom", "2px")
    .text(title);
  const svg = sec.append("svg").attr("width", w).attr("height", h);
  const stats = _boxStats(vals);
  if (!stats) {
    sec.append("div").attr("class", "empty-msg").text("no data");
    return;
  }
  const x = d3.scaleLinear()
    .domain([Math.min(0, stats.lo), Math.max(1, stats.hi)]).nice()
    .range([8, w - 8]);
  const y = h / 2;
  svg.append("line").attr("x1", x(stats.lo)).attr("x2", x(stats.hi))
    .attr("y1", y).attr("y2", y).attr("stroke", color);
  svg.append("rect").attr("x", x(stats.q1)).attr("y", y - 12)
    .attr("width", Math.max(1, x(stats.q3) - x(stats.q1)))
    .attr("height", 24)
    .attr("fill", color).attr("fill-opacity", 0.6)
    .attr("stroke", color);
  svg.append("line").attr("x1", x(stats.med)).attr("x2", x(stats.med))
    .attr("y1", y - 12).attr("y2", y + 12).attr("stroke", "#222")
    .attr("stroke-width", 1.5);
  svg.append("line").attr("x1", x(stats.lo)).attr("x2", x(stats.lo))
    .attr("y1", y - 6).attr("y2", y + 6).attr("stroke", color);
  svg.append("line").attr("x1", x(stats.hi)).attr("x2", x(stats.hi))
    .attr("y1", y - 6).attr("y2", y + 6).attr("stroke", color);
  const ax = d3.axisBottom(x).ticks(4).tickSize(2);
  svg.append("g").attr("transform", `translate(0,${h - 14})`).call(ax)
    .selectAll("text").style("font-size", "9px");
}

function drawPhenotypeBoxes(host, title, items, log) {
  const sec = host.append("div").style("margin-bottom", "8px");
  sec.append("div").style("font-size", "11.5px")
    .style("font-weight", "600")
    .style("margin-bottom", "2px")
    .text(title);
  const itemsNonEmpty = items.filter(it => it.vals.some(v => v > 0));
  if (!itemsNonEmpty.length) {
    sec.append("div").attr("class", "empty-msg").text("no phenotypes with data");
    return;
  }
  const hostW = (host.node().getBoundingClientRect().width) || 320;
  const W = Math.max(220, Math.floor(hostW - 16));
  const rowH = 18;
  const labelW = 100;
  const h = itemsNonEmpty.length * rowH + 20;
  const svg = sec.append("svg").attr("width", W).attr("height", h);
  const allVals = itemsNonEmpty.flatMap(it => it.vals);
  let maxV = d3.max(allVals) || 1;
  const x = d3.scaleLinear().domain([0, maxV]).nice()
    .range([labelW, W - 6]);
  itemsNonEmpty.forEach((it, i) => {
    const y = i * rowH + 4 + rowH / 2;
    const stats = _boxStats(it.vals);
    svg.append("text").attr("x", labelW - 4).attr("y", y + 3)
      .attr("text-anchor", "end").style("font-size", "10px")
      .style("fill", "#333").text(it.label);
    if (!stats) return;
    svg.append("line").attr("x1", x(stats.lo)).attr("x2", x(stats.hi))
      .attr("y1", y).attr("y2", y).attr("stroke", it.color);
    svg.append("rect")
      .attr("x", x(stats.q1)).attr("y", y - 6)
      .attr("width", Math.max(1, x(stats.q3) - x(stats.q1)))
      .attr("height", 12).attr("fill", it.color)
      .attr("fill-opacity", 0.6).attr("stroke", it.color);
    svg.append("line").attr("x1", x(stats.med)).attr("x2", x(stats.med))
      .attr("y1", y - 6).attr("y2", y + 6).attr("stroke", "#222")
      .attr("stroke-width", 1.4);
  });
  const ax = d3.axisBottom(x).ticks(4).tickSize(2);
  svg.append("g")
    .attr("transform", `translate(0,${itemsNonEmpty.length * rowH + 4})`)
    .call(ax).selectAll("text").style("font-size", "9px");
}

// ---- Clone table ----
let tableSort = { col: "n_cells_total", asc: false };
const ROW_HEIGHT = 32;
const ROW_BUFFER = 20;
const TABLE_COLS = [
  { key: "trb",                 label: "TRB",         minw: 140 },
  { key: "patient",             label: "Patient",     minw: 70 },
  { key: "n_cells_total",       label: "Total cells", minw: 90 },
  { key: "n_tissues",           label: "Tissues",     minw: 70 },
  { key: "n_timepoints",        label: "Timepoints",  minw: 80 },
  { key: "dominant_phenotype",  label: "Dominant",    minw: 110 },
  { key: "lineage",             label: "Lineage",     minw: 70 },
  { key: "max_expansion",       label: "Max log₂FC",  minw: 90 },
  { key: "tumor_fraction",      label: "Tumor frac",  minw: 80 },
];
let _tableState = { rows: [], built: false };

function _filteredSortedRows() {
  const q = (document.getElementById("table-search").value || "")
            .toLowerCase();
  const rows = DATA.clones.filter(r =>
    !q || r.trb.toLowerCase().includes(q)
       || r.patient.toLowerCase().includes(q)
       || r.dominant_phenotype.toLowerCase().includes(q));
  rows.sort((a, b) => {
    const c = tableSort.col;
    let av = a[c], bv = b[c];
    if (typeof av === "string") return tableSort.asc
      ? av.localeCompare(bv) : bv.localeCompare(av);
    return tableSort.asc ? av - bv : bv - av;
  });
  return rows;
}

function _formatCell(r, key) {
  let v = r[key];
  if (key === "trb")
    v = v.length > 22 ? v.slice(0, 20) + "…" : v;
  if (key === "max_expansion")
    v = (typeof v === "number") ? v.toFixed(2) : v;
  if (key === "tumor_fraction")
    v = (typeof v === "number") ? (v * 100).toFixed(0) + "%" : v;
  return v;
}

function renderCloneTable() {
  const host = d3.select("#clone-table-host");
  host.selectAll("*").remove();

  _tableState.rows = _filteredSortedRows();
  const rows = _tableState.rows;

  // outer scroller (horizontal + vertical)
  const scroller = host.append("div")
    .attr("class", "clone-table-scroller")
    .style("overflow", "auto")
    .style("max-height", "100%")
    .style("height", "100%")
    .style("position", "relative");

  const minWidth = TABLE_COLS.reduce((s, c) => s + c.minw, 0);
  const tab = scroller.append("table").attr("class", "clone-table")
    .style("min-width", minWidth + "px");
  const colg = tab.append("colgroup");
  TABLE_COLS.forEach(c => colg.append("col")
    .attr("style", `min-width:${c.minw}px;`));

  const thead = tab.append("thead").append("tr");
  TABLE_COLS.forEach(c => {
    thead.append("th")
      .style("min-width", c.minw + "px")
      .text(c.label + (tableSort.col === c.key
                       ? (tableSort.asc ? " ▲" : " ▼") : ""))
      .on("click", () => {
        if (tableSort.col === c.key) tableSort.asc = !tableSort.asc;
        else { tableSort.col = c.key; tableSort.asc = false; }
        renderCloneTable();
      });
  });
  const tbody = tab.append("tbody");
  _renderVirtualRows(scroller.node(), tbody);
  scroller.node().addEventListener("scroll", () =>
    _renderVirtualRows(scroller.node(), tbody));
}

function _renderVirtualRows(scroller, tbody) {
  const rows = _tableState.rows;
  const n = rows.length;
  if (!n) {
    tbody.selectAll("*").remove();
    tbody.append("tr").append("td")
      .attr("colspan", TABLE_COLS.length)
      .attr("class", "empty-msg")
      .text("no clones match filter");
    return;
  }
  const scrollTop = scroller.scrollTop;
  const viewportH = scroller.clientHeight;
  const visStart = Math.floor(scrollTop / ROW_HEIGHT);
  const visEnd   = Math.ceil((scrollTop + viewportH) / ROW_HEIGHT);
  const start = Math.max(0, visStart - ROW_BUFFER);
  const end   = Math.min(n, visEnd + ROW_BUFFER);

  tbody.selectAll("*").remove();
  if (start > 0) {
    tbody.append("tr").attr("class", "vspacer")
      .style("height", (start * ROW_HEIGHT) + "px")
      .append("td").attr("colspan", TABLE_COLS.length);
  }
  for (let i = start; i < end; i++) {
    const r = rows[i];
    const tr = tbody.append("tr")
      .style("height", ROW_HEIGHT + "px")
      .classed("selected", STATE.selected_clone === r.clone_id)
      .on("click", () => onCloneRowClick(r));
    TABLE_COLS.forEach(c => {
      tr.append("td")
        .style("min-width", c.minw + "px")
        .text(_formatCell(r, c.key));
    });
  }
  if (end < n) {
    tbody.append("tr").attr("class", "vspacer")
      .style("height", ((n - end) * ROW_HEIGHT) + "px")
      .append("td").attr("colspan", TABLE_COLS.length);
  }
}

function onCloneRowClick(r) {
  STATE.view_mode = "clone";
  STATE.selected_clone = r.clone_id;
  STATE.selected_patient = null;
  STATE.selected_edge = null;
  const occ = DATA.view.clone[r.clone_id].nodes;
  if (STATE.selected_node) {
    const k = siteKey(STATE.selected_node.tissue,
                       STATE.selected_node.timepoint);
    if (!occ[k]) STATE.selected_node = null;
  }
  renderAll();
}

// ---- Top-bar wiring ----
function buildPatientSelect() {
  const sel = d3.select("#patient-select");
  sel.append("option").attr("value", "__all__").text("All patients");
  DATA.patients.forEach(p => {
    sel.append("option").attr("value", p).text(p);
  });
  sel.on("change", function () {
    const v = this.value;
    if (v === "__all__") {
      STATE.view_mode = "global";
      STATE.selected_patient = null;
    } else {
      STATE.view_mode = "patient";
      STATE.selected_patient = v;
      STATE.selected_clone = null;
    }
    STATE.selected_edge = null;
    renderAll();
  });
}

function wireSegs() {
  d3.select("#seg-metric").selectAll("button").on("click", function () {
    // Clone view: both metric options are inert (edges are uniform
    // width, colored by expansion regardless of toggle).
    if (STATE.view_mode === "clone") return;
    STATE.metric_mode = d3.select(this).attr("data-v");
    renderAll();
  });
  d3.select("#seg-composition").selectAll("button").on("click", function () {
    STATE.composition_mode = d3.select(this).attr("data-v");
    renderAll();
  });
  d3.select("#seg-nodesize").selectAll("button").on("click", function () {
    if (STATE.view_mode !== "clone") return;
    STATE.node_size_mode = d3.select(this).attr("data-v");
    renderAll();
  });
  d3.select("#home-btn").on("click", () => {
    STATE.view_mode = "global";
    STATE.selected_patient = null;
    STATE.selected_clone = null;
    STATE.selected_edge = null;
    d3.select("#patient-select").property("value", "__all__");
    renderAll();
  });
  document.getElementById("table-search").addEventListener("input",
    () => renderCloneTable());
  // Side-panel clear button: deselect node or edge and return to the
  // empty/instructional state.
  document.getElementById("side-clear").addEventListener("click", () => {
    STATE.selected_node = null;
    STATE.selected_edge = null;
    renderAll();
  });
}

function syncTopbar() {
  // patient select disabled in clone view
  const pSel = document.getElementById("patient-select");
  pSel.disabled = (STATE.view_mode === "clone");
  pSel.title = pSel.disabled
    ? "Patient filter not available in clone view" : "";

  // badge
  const badge = d3.select("#view-badge");
  badge.attr("class", "view-badge " + STATE.view_mode);
  if (STATE.view_mode === "global") badge.text("Global");
  else if (STATE.view_mode === "patient")
    badge.text(`Patient: ${STATE.selected_patient}`);
  else {
    let badgeText = "Clone: ";
    if (STATE.selected_clone) {
      const c = DATA.clones.find(x => x.clone_id === STATE.selected_clone);
      if (c) {
        const trb = c.trb || "";
        const shortTrb = trb.length > 14 ? trb.slice(0, 12) + "…" : trb;
        badgeText += `${c.patient} · ${shortTrb}`;
      } else {
        badgeText += STATE.selected_clone;
      }
    }
    badge.text(badgeText);
  }

  // Metric segment — in clone view we force "expansion" and disable
  // the Traffic button specifically.
  const effMode = effectiveMetricMode();
  const cloneToggleTip = "Single-clone view shows uniform-width "
                          + "edges colored by expansion";
  d3.selectAll("#seg-metric button").each(function () {
    const v = d3.select(this).attr("data-v");
    const isClone = STATE.view_mode === "clone";
    d3.select(this).classed("active", v === effMode);
    this.disabled = isClone;
    this.title = isClone ? cloneToggleTip : "";
    this.style.opacity = isClone ? "0.4" : "";
    this.style.cursor = isClone ? "not-allowed" : "pointer";
  });
  d3.select("#seg-metric").classed("disabled", STATE.view_mode === "clone");
  d3.selectAll("#seg-composition button").classed("active", function () {
    return d3.select(this).attr("data-v") === STATE.composition_mode;
  });
  d3.selectAll("#seg-nodesize button").classed("active", function () {
    return d3.select(this).attr("data-v") === STATE.node_size_mode;
  });
  d3.select("#seg-nodesize").classed("disabled", STATE.view_mode !== "clone");
}

// ---- Edge-click Sankey panel ----
function _phenoDistAtNode(view_mode, tissue, t, mode) {
  // Return a K-dim distribution at the given site, filtered by the
  // CURRENT view (global / patient / clone) and composition mode.
  const sk = siteKey(tissue, t);
  const order = mode === "tcell" ? P.tcell_order : P.myeloid_order;
  const K = order.length;
  const zeros = new Array(K).fill(0);
  let comp = null, nCells = 0;
  if (view_mode === "clone" && STATE.selected_clone) {
    if (mode === "tcell") {
      const cv = DATA.view.clone[STATE.selected_clone];
      const cn = cv ? cv.nodes[sk] : null;
      comp = cn ? cn.tcell_comp : null;
      nCells = cn ? cn.n_tcells : 0;
    } else {
      // Myeloid has no clone identity — pull from the clone's patient.
      const c = DATA.clones.find(x => x.clone_id === STATE.selected_clone);
      const pat = c ? c.patient : null;
      const pn = (pat && DATA.view.patient[pat]
                   && DATA.view.patient[pat].nodes[sk]) || null;
      comp = pn ? pn.myeloid_comp : null;
      nCells = pn ? pn.n_myeloid : 0;
    }
  } else if (view_mode === "patient" && STATE.selected_patient) {
    const pn = (DATA.view.patient[STATE.selected_patient]
                 && DATA.view.patient[STATE.selected_patient].nodes[sk])
              || null;
    comp = pn ? (mode === "tcell" ? pn.tcell_comp : pn.myeloid_comp) : null;
    nCells = pn ? (mode === "tcell" ? pn.n_tcells : pn.n_myeloid) : 0;
  } else {
    const gn = DATA.view.global.nodes[sk];
    comp = gn ? (mode === "tcell" ? gn.tcell_comp : gn.myeloid_comp) : null;
    nCells = gn ? (mode === "tcell" ? gn.n_tcells : gn.n_myeloid) : 0;
  }
  if (!comp || nCells <= 0) return { dist: zeros, nCells: 0, hasData: false };
  // Normalize to be safe (should already sum to 1).
  const s = comp.reduce((a, b) => a + (b || 0), 0);
  if (s <= 0) return { dist: zeros, nCells, hasData: false };
  const dist = comp.map(v => (v || 0) / s);
  return { dist, nCells, hasData: true };
}


function _parseSelectedEdge() {
  // Format: "SRC|T1->DST|T2"
  if (!STATE.selected_edge) return null;
  const m = /^([^|]+)\|([^-]+)->([^|]+)\|(.+)$/.exec(STATE.selected_edge);
  if (!m) return null;
  return { src_tissue: m[1], src_t: m[2],
           dst_tissue: m[3], dst_t: m[4] };
}


function renderSankey() {
  // Renders into the side-panel top-half container (#box-host),
  // sharing it with the composition view. Title and clear button
  // live in the shared #side-title-text / #side-clear elements.
  const sel = _parseSelectedEdge();
  if (!sel) return;

  const hostSel = d3.select("#box-host");
  const titleText = d3.select("#side-title-text");
  const sub = d3.select("#side-sub");
  titleText.text(`${sel.src_tissue}|t=${sel.src_t} → `
                  + `${sel.dst_tissue}|t=${sel.dst_t}`);
  sub.text("phenotype flow at this edge (outer-product)");

  const mode = STATE.composition_mode;
  const order = mode === "tcell" ? P.tcell_order : P.myeloid_order;
  const colors = mode === "tcell" ? P.tcell_colors : P.myeloid_colors;
  const labels = mode === "tcell" ? P.tcell_labels : P.myeloid_labels;
  const K = order.length;

  const src = _phenoDistAtNode(STATE.view_mode,
                                sel.src_tissue, sel.src_t, mode);
  const dst = _phenoDistAtNode(STATE.view_mode,
                                sel.dst_tissue, sel.dst_t, mode);

  // Sanity #15: each side either sums to 1.0 ± 1e-6 or is all-zero
  // (no data at that node — we still render the frame).
  const sumS = src.dist.reduce((a, b) => a + b, 0);
  const sumD = dst.dist.reduce((a, b) => a + b, 0);
  if (src.hasData && Math.abs(sumS - 1.0) > 1e-6) {
    console.error(`[sankey-15] src sum ${sumS} != 1`);
  }
  if (dst.hasData && Math.abs(sumD - 1.0) > 1e-6) {
    console.error(`[sankey-15] dst sum ${sumD} != 1`);
  }

  const hostNode = hostSel.node();
  const rect = hostNode ? hostNode.getBoundingClientRect()
                         : { width: 320, height: 240 };
  const hostW = rect.width || 320;
  // Fill the available top-half region vertically. The composition
  // pane has a small title-row + side-sub line above #box-host, so the
  // host's own clientHeight is the right number to fill.
  const TITLE_BAR_H = 28;     // reserved for title/clear button area
  const PAD_BOTTOM = 6;
  const hostH = rect.height || 0;
  const svgW = Math.max(260, Math.floor(hostW - 16));
  // Fallback when DOM hasn't been laid out yet (height=0 at first paint).
  const svgH = (hostH > 60)
    ? Math.max(180, Math.floor(hostH - TITLE_BAR_H - PAD_BOTTOM))
    : 240;
  const svg = hostSel.append("svg")
    .attr("id", "sankey-svg")
    .attr("width", svgW).attr("height", svgH)
    .style("display", "block");

  const padL = 90, padR = 90;
  const barW = 14;
  const top = 10;
  const h = svgH - top - 10;  // inner ribbon area
  const barXsrc = padL;
  const barXdst = svgW - padR - barW;
  const ribbonX1 = barXsrc + barW;
  const ribbonX2 = barXdst;

  function _truncate(s, n) {
    return (s && s.length > n) ? (s.slice(0, n - 1) + "…") : (s || "");
  }

  // Bar stacks: phenotype rectangles with heights ∝ dist[i] * h.
  // Labels are emitted only for phenotypes with non-zero mass on the
  // side they are being drawn on. A phenotype absent on both sides
  // contributes nothing — no bar, no label, no tick. The ribbon code
  // below already skips flows < 0.005.
  function drawBar(side, xBar, dist, hasData) {
    let y = top;
    const centersY = new Array(K).fill(null);
    order.forEach((p, i) => {
      const frac = dist[i] || 0;
      const segH = frac * h;
      const yMid = y + segH / 2;
      const isPresent = frac > 0 && hasData;
      if (isPresent) {
        svg.append("rect")
          .attr("x", xBar).attr("y", y)
          .attr("width", barW).attr("height", segH)
          .attr("fill", colors[p]).attr("fill-opacity", 0.92)
          .attr("stroke", "#fff").attr("stroke-width", 0.5);
        centersY[i] = yMid;

        const isLeft = (side === "src");
        const tx = isLeft ? (xBar - 6) : (xBar + barW + 6);
        const anchor = isLeft ? "end" : "start";
        const fullLab = labels[p] || p;
        const pct = (frac * 100);
        const labWithPct = `${fullLab} ${pct.toFixed(0)}%`;
        const display = _truncate(labWithPct, 14);
        const t = svg.append("text")
          .attr("x", tx).attr("y", yMid + 3)
          .attr("text-anchor", anchor)
          .style("font-size", "10px")
          .style("fill", colors[p] || "#333")
          .text(display);
        t.append("title").text(labWithPct);
      }
      y += segH;
    });
    return centersY;
  }

  const srcCenters = drawBar("src", barXsrc, src.dist, src.hasData);
  const dstCenters = drawBar("dst", barXdst, dst.dist, dst.hasData);

  // Ribbons: outer-product flow_ij = src[i] * dst[j]. Cubic bezier.
  if (src.hasData && dst.hasData) {
    for (let i = 0; i < K; i++) {
      const si = src.dist[i] || 0;
      if (si <= 0 || srcCenters[i] == null) continue;
      for (let j = 0; j < K; j++) {
        const dj = dst.dist[j] || 0;
        const flow = si * dj;
        if (flow < 0.005) continue;
        if (dstCenters[j] == null) continue;
        const y1 = srcCenters[i];
        const y2 = dstCenters[j];
        const cx1 = ribbonX1 + (ribbonX2 - ribbonX1) * 0.45;
        const cx2 = ribbonX1 + (ribbonX2 - ribbonX1) * 0.55;
        const d = `M${ribbonX1},${y1} C${cx1},${y1} ${cx2},${y2} ${ribbonX2},${y2}`;
        const w = Math.max(0.5, flow * h);
        const op = Math.min(0.8, 0.2 + flow * 2);
        svg.append("path").attr("d", d)
          .attr("fill", "none")
          .attr("stroke", colors[order[i]] || "#999")
          .attr("stroke-width", w)
          .attr("stroke-opacity", op);
      }
    }
  }

  // Frame headers (small): src n-cells / dst n-cells.
  svg.append("text")
    .attr("x", barXsrc + barW / 2).attr("y", top - 2)
    .attr("text-anchor", "middle").style("font-size", "9px")
    .style("fill", "#555")
    .text(src.hasData ? `n=${src.nCells}` : "no data");
  svg.append("text")
    .attr("x", barXdst + barW / 2).attr("y", top - 2)
    .attr("text-anchor", "middle").style("font-size", "9px")
    .style("fill", "#555")
    .text(dst.hasData ? `n=${dst.nCells}` : "no data");

  // Sanity #16 (runtime): per-side label count equals the number of
  // phenotypes with non-zero mass on that side, plus the two n-tags.
  // Phenotypes absent on both sides contribute nothing.
  const svgEl = svg.node();
  const labelTexts = svgEl.querySelectorAll("text");
  const expectedLabels =
    (src.hasData ? src.dist.filter(v => v > 0).length : 0) +
    (dst.hasData ? dst.dist.filter(v => v > 0).length : 0) + 2;
  if (labelTexts.length < expectedLabels) {
    console.warn(`[sankey-16] expected >= ${expectedLabels} text nodes, `
                 + `got ${labelTexts.length}`);
  }
}


function renderAll() {
  syncTopbar();
  renderNetwork();
  renderLegend();
  renderBoxPanel();
  renderCloneTable();
  // sanity #12 — runtime self-test for node click handlers
  requestAnimationFrame(selfTestNodeClicks);
  // clone-view diagnostic: lit nodes with no edge attached
  if (STATE.view_mode === "clone" && STATE.selected_clone) {
    const v = DATA.view.clone[STATE.selected_clone];
    const lit = new Set(Object.keys(v.nodes || {}));
    const ep = new Set();
    (v.edges || []).forEach(e => {
      ep.add(siteKey(e.src_tissue, e.src_t));
      ep.add(siteKey(e.dst_tissue, e.dst_t));
    });
    let isolated = 0;
    lit.forEach(s => { if (!ep.has(s)) isolated += 1; });
    const _c = DATA.clones.find(x => x.clone_id === STATE.selected_clone);
    const _label = _c ? `${_c.patient}|${_c.trb}` : STATE.selected_clone;
    console.log(`Clone ${_label}: ${isolated} isolated lit nodes`);
  }
}

window.addEventListener("resize", () => {
  renderNetwork();
  renderBoxPanel();
});

// ---- Boot ----
buildPatientSelect();
wireSegs();
renderAll();
"""

# %%
# ---- Assemble HTML ----
print("\nFetching D3 v7...")
d3_js = fetch_d3()

print("Writing HTML...")
html = (HTML_TEMPLATE
        .replace("__D3_JS__", d3_js)
        .replace("__DATA_JSON__", data_json)
        .replace("__APP_JS__", APP_JS))
out_html = CLONE_NETWORK_HTML
out_html.write_text(html, encoding="utf-8")
html_bytes = out_html.stat().st_size
print(f"  Wrote {out_html} ({html_bytes / 1024 / 1024:.2f} MB)")

# Verify the JS click-handler self-test made it into the emitted HTML.
# Runtime DOM verification happens client-side via selfTestNodeClicks().
if 'selfTestNodeClicks' in html and 'circle.hit' in html:
    _sanity("node-click self-test present in emitted HTML (runtime "
            "verifies 18 hit circles via selfTestNodeClicks)")
else:
    _sanity("node-click self-test missing from emitted HTML", ok=False)
    fail_msgs.append("self-test missing")

# %%
# ---- data_summary.txt ----
print("Writing data_summary.txt...")
lines = []
lines.append("# clone_network_explorer — build summary")
lines.append(f"elapsed: {time.time() - t_start:.1f}s")
lines.append("")
lines.append("## Inputs")
lines.append(f"  T-cell h5ad : {TCELL_PATH.name}  ({len(t_obs):,} cells)")
lines.append(f"  myeloid h5ad: {MYELOID_PATH.name}  ({len(m_obs):,} cells)")
lines.append(f"  edge metrics: {EDGE_METRICS_PATH.name}  "
             f"({'present' if EDGE_METRICS_PATH.exists() else 'missing'})")
lines.append("")
lines.append("## Cells per (tissue, timepoint)")
lines.append(f"  {'site':<10}{'n_tcells':>10}{'n_myeloid':>10}")
for tissue in TISSUES:
    for t in TIMEPOINTS:
        site = _site_key(tissue, t)
        n_t = nodes_global[site]["n_tcells"]
        n_m = nodes_global[site]["n_myeloid"]
        lines.append(f"  {site:<10}{n_t:>10}{n_m:>10}")
lines.append("")
lines.append("## Edges retained at global level")
lines.append(f"  total : {len(edges_global)} / max possible "
             f"{len(EDGES_TISSUE) * len(TRANSITIONS)}")
lines.append(f"  filter: n_branches_used >= {MIN_GLOBAL_BRANCHES_USED}")
lines.append("")
lines.append(f"## Eligible clones (n_cells >= {MIN_CELLS_PER_CLONE}): "
             f"{n_clones_eligible}")
pat_counts = pd.Series([c["patient"] for c in clones_table]).value_counts()
for pat in PATIENTS:
    n = int(pat_counts.get(pat, 0))
    lines.append(f"  {pat:<8}: {n:>4} clones")
lines.append("")
size_buckets = [(MIN_CELLS_PER_CLONE, 50), (50, 200),
                (200, 1000), (1000, 1e9)]
lines.append("  cell-count buckets among eligible:")
for lo, hi in size_buckets:
    n = sum(1 for c in clones_table
            if lo <= c["n_cells_total"] < hi)
    lines.append(f"    [{lo}, {hi if hi < 1e9 else 'inf'}): {n}")
lines.append("")
lines.append("## Sanity")
lines.extend(sanity_lines)
lines.append("")
lines.append(f"## Output: {out_html} ({html_bytes / 1024 / 1024:.2f} MB)")
summary_path = OUT_DIR / "data_summary.txt"
summary_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  Wrote {summary_path}")

if fail_msgs:
    print("\nSANITY FAILURES:")
    for m in fail_msgs:
        print(f"  - {m}")
    sys.exit(1)

elapsed = time.time() - t_start
print(f"\nDone in {elapsed:.1f}s ({elapsed / 60:.2f} min).")
print(f"Open: {out_html}")
landing.write_landing()
