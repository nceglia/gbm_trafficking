# %%
"""Consolidated gene-level OT rewiring across every cross-tissue transition.

For each (lineage, src_tissue, dst_tissue) with sufficient draining
events, this driver:
  1. parses src→dst edges out of retained_graph_edges,
  2. runs clone-constrained Sinkhorn OT per event in two geometries
     (PCA on HVGs and full log1p transcriptome),
  3. computes a mass-weighted per-gene delta with a permutation null,
  4. ranks pathways with GSEA prerank (Hallmark + KEGG),
  5. anchors each significantly-rewired gene to the phenotype with the
     highest z-scored expression in the destination tissue,
  6. pulls CollecTRI TF→target links among the displayed genes, and
  7. renders a panel.png/pdf (and a `qc/` set of diagnostic plots).

Subsumes:
  pipeline/traffic_draining_ot.py
  pipeline/figure_cd8_drainage_amplitude.py
  pipeline/figure_cd8_rewiring_amplitude.py

Reads:
  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad
  results/traffic_archetype_graphs/clone_archetypes.csv

Writes (per transition × lineage) to
  results/traffic_drainage_rewiring/<SRC>_to_<DST>/<lineage>/:
    rewiring_genes.csv             — PCA-OT mean_delta + perm padj
    rewiring_genes_fullexpr.csv    — full-transcriptome OT delta
    draining_multiplicity.csv      — per-clone multiplicity class
    gsea.csv                       — Hallmark + KEGG NES
    selected_genes.csv             — figure-displayed genes
    tf_target_links.csv            — CollecTRI links inside display
    phenotype_expression_target.csv
    phenotype_expression_source.csv
    panel.png / panel.pdf
    qc/multiplicity_breakdown.png
    qc/event_mass_distribution.png
    qc/rewiring_volcano.png
    qc/gsea_top.png
    qc/geometry_comparison.png
And:
  results/traffic_drainage_rewiring/run_summary.csv
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import decoupler as dc
import gseapy as gp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.style import (  # noqa: E402
    LINEAGE_COLORS,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_LABELS,
)

# =====================================================================
# Configuration
# =====================================================================
OUT_DIR = paths.RESULTS_DIR / "traffic_drainage_rewiring"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLONE_ARCH_CSV = paths.TRAFFIC_ARCHETYPE_GRAPHS_DIR / "clone_archetypes.csv"

# Every directed pair over the three tissues. Pairs with no observed
# events are skipped gracefully — the cost is one pass over edges.
DEFAULT_TRANSITIONS = (
    ("PBMC", "CSF"),
    ("PBMC", "TP"),
    ("CSF",  "TP"),
    ("CSF",  "PBMC"),
    ("TP",   "CSF"),
    ("TP",   "PBMC"),
)
DEFAULT_LINEAGES = ("CD8", "CD4")

# OT
N_HVG = 2000
N_PCS = 50
SINKHORN_EPS = 0.1
SINKHORN_REG_M = 1.0          # marginal relaxation (unbalanced OT)
MIN_CELLS_PER_SIDE = 2
N_PERM = 1000
N_BOOTSTRAP = 1000             # event bootstrap for CI
GENE_SETS = ["MSigDB_Hallmark_2020", "KEGG_2021_Human"]
RNG_SEED = 42

# Figure — these are the DEFAULTS, overridable per-run via CLI
# (--padj-thr / --delta-thr / --max-genes). DELTA_THR is tuned for
# unbalanced OT, which produces smaller per-gene Δ than balanced
# Sinkhorn — drop it lower if the panel still looks too sparse.
PADJ_THR  = 0.05
DELTA_THR = 1.2
MAX_GENES = 80
DPI = 200
ARC_COLOR = "#101010"
ARC_LW = 1.05
ARC_ALPHA = 0.55
TF_ACTIVITY_PADJ = 0.05        # TFs with ulm padj <= this count as
                                # "empirically active drivers"

# Multiplicity
MULT_ORDER  = ("single", "sequential", "overlapping", "pulse")
MULT_COLORS = {
    "single":      "#bdbdbd",
    "sequential":  "#2166ac",
    "overlapping": "#7e9437",
    "pulse":       "#c51b8a",
}

EDGE_RE = re.compile(r"^([A-Z]+)@T(\d+)->([A-Z]+)@T(\d+)$")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lineage", choices=["CD8", "CD4", "both"],
                   default="both")
    p.add_argument("--transitions", type=str, default="",
                   help="Comma-separated SRC_to_DST list "
                        "(e.g. 'CSF_to_TP,PBMC_to_CSF'); "
                        "default = all 6 directed pairs.")
    p.add_argument("--skip-perm", action="store_true",
                   help="Skip the permutation null (fast pass for testing).")
    p.add_argument("--padj-thr", type=float, default=PADJ_THR,
                   help=f"Display padj cutoff (default {PADJ_THR}).")
    p.add_argument("--delta-thr", type=float, default=DELTA_THR,
                   help=f"Display |Δ| cutoff (default {DELTA_THR}).")
    p.add_argument("--max-genes", type=int, default=MAX_GENES,
                   help=f"Cap on displayed genes (default {MAX_GENES}).")
    p.add_argument("--bar-h", type=float, default=4.6,
                   help="Bar-axis height in inches (default 4.6). "
                        "Lower this to ~2 to render a wider/shorter "
                        "panel suitable for composite figures.")
    p.add_argument("--no-title", action="store_true",
                   help="Omit the title + subtitle on the rendered "
                        "panel (intended for composite figures that "
                        "supply their own title).")
    p.add_argument("--legend-size", type=float, default=7.5,
                   help="Legend font size in points (default 7.5).")
    return p.parse_args()


# =====================================================================
# Edge parsing + multiplicity
# =====================================================================
def parse_edges(edge_str):
    if not isinstance(edge_str, str) or not edge_str.strip():
        return []
    out = []
    for tok in edge_str.strip().split():
        m = EDGE_RE.match(tok)
        if m:
            s_t, s_tp, d_t, d_tp = m.groups()
            out.append((s_t, int(s_tp), d_t, int(d_tp)))
    return out


def classify_multiplicity(src_tps, dst_tps):
    s = sorted(set(src_tps))
    d = sorted(set(dst_tps))
    if len(s) == 1 and len(d) == 1:
        return "single"
    if set(s) & set(d):
        return "overlapping"
    if s and d and max(s) < min(d):
        return "sequential"
    if s and d:
        first_d = min(d)
        if [t for t in s if t < first_d] and [t for t in s if t >= first_d]:
            return "pulse"
    return "single"


# =====================================================================
# Per-lineage setup (HVG + PCA — done ONCE; reused across transitions)
# =====================================================================
def setup_lineage(adata, lineage):
    print(f"\n=== building PCA geometry for {lineage} ===")
    obs_full = adata.obs[["trb", "tissue", "timepoint",
                           "phenotype", "patient"]].copy()
    obs_full["cell_index"] = np.arange(adata.n_obs)
    obs = obs_full.copy()
    obs["trb"] = obs["trb"].astype(str)
    obs = obs[(obs["trb"].notna()) & (obs["trb"] != "")]
    for c in ("tissue", "timepoint", "phenotype", "patient"):
        obs[c] = obs[c].astype(str)
    obs["clone_id"] = obs["patient"] + "|" + obs["trb"]
    obs["lineage"] = np.where(obs["phenotype"].str.contains("CD8"),
                              "CD8", "CD4")
    obs_lin = obs[obs["lineage"] == lineage].copy()
    obs_lin["tp_int"] = obs_lin["timepoint"].astype(int)

    adata_lin = adata[obs_lin["cell_index"].values].copy()
    try:
        sc.pp.highly_variable_genes(
            adata_lin, n_top_genes=N_HVG, flavor="seurat_v3",
            layer="counts", subset=False)
    except Exception:
        adata_lin.X = adata_lin.layers["log1p"].copy()
        sc.pp.highly_variable_genes(
            adata_lin, n_top_genes=N_HVG, flavor="seurat", subset=False)
    hvg_mask = adata_lin.var["highly_variable"].values
    adata_hvg = adata_lin[:, hvg_mask].copy()
    adata_hvg.X = (adata_hvg.layers["log1p"].copy()
                   if "log1p" in adata_hvg.layers else adata_hvg.X)
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=N_PCS, random_state=RNG_SEED)
    PC = adata_hvg.obsm["X_pca"].astype(np.float32)
    PC = (PC - PC.mean(0)) / (PC.std(0) + 1e-8)
    print(f"  {lineage}: {adata_lin.n_obs:,} cells, PC {PC.shape}, "
          f"HVGs {int(hvg_mask.sum())}")

    cells_map = (obs_lin.groupby(["clone_id", "tissue", "tp_int"])
                          ["cell_index"]
                          .apply(lambda s:
                                 np.asarray(s.values, dtype=np.int64)))
    global_to_local = {int(g): i for i, g in
                       enumerate(obs_lin["cell_index"].values)}
    return {
        "obs_lin": obs_lin,
        "cells_map": cells_map,
        "PC": PC,
        "global_to_local": global_to_local,
    }


# =====================================================================
# OT primitives
# =====================================================================
def _cosine_cost(A, B):
    an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (1.0 - an @ bn.T).astype(np.float32)


def _densify(log1p_X, idx_global):
    rows = log1p_X[idx_global]
    if sp.issparse(rows):
        rows = rows.toarray()
    return rows.astype(np.float32)


def run_event_ot(src_g, tgt_g, *, geometry, PC, log1p_X, global_to_local):
    """Per-event unbalanced Sinkhorn OT in `pca` or `full` geometry.

    Unbalanced (with marginal-relaxation `SINKHORN_REG_M`) so cells that
    have no good match in the other tissue can be partially unmatched
    instead of forced onto a far neighbour — which matters when a CSF
    subset has drifted phenotypically with no analogue in the timepoint
    we land in.
    """
    if geometry == "pca":
        A = PC[[global_to_local[int(g)] for g in src_g]]
        B = PC[[global_to_local[int(g)] for g in tgt_g]]
    else:
        A = _densify(log1p_X, src_g)
        B = _densify(log1p_X, tgt_g)
    n_s, n_t = A.shape[0], B.shape[0]
    m = float(min(n_s, n_t))
    a = np.full(n_s, m / n_s, dtype=np.float64)
    b = np.full(n_t, m / n_t, dtype=np.float64)
    C = _cosine_cost(A, B).astype(np.float64)
    pi = ot.unbalanced.sinkhorn_unbalanced(
        a, b, C, reg=SINKHORN_EPS, reg_m=SINKHORN_REG_M,
        numItermax=500, stopThr=1e-7)
    pi = np.asarray(pi, dtype=np.float32)
    mass = float(pi.sum())
    mean_cost = float((pi * C).sum() / max(mass, 1e-12))
    return pi, mass, mean_cost


# =====================================================================
# Per-transition pipeline
# =====================================================================
def run_transition(*, adata, clones_lin, setup, lineage, src, dst,
                   outdir, rng, skip_perm=False):
    obs_lin = setup["obs_lin"]
    cells_map = setup["cells_map"]
    PC = setup["PC"]
    global_to_local = setup["global_to_local"]
    log1p_X = adata.layers["log1p"]
    gene_names = np.asarray(adata.var_names)
    n_genes = adata.n_vars

    print(f"\n{'='*72}\n[{lineage}]  {src} → {dst}\n{'='*72}")

    # --- draining clones for this transition ---
    drain_per = {}
    for _, row in clones_lin.iterrows():
        edges = parse_edges(row.get("retained_graph_edges", ""))
        drains = [(tp1, tp2) for (t1, tp1, t2, tp2) in edges
                  if t1 == src and t2 == dst and tp1 < tp2]
        if drains:
            drain_per[row["clone_id"]] = drains
    if not drain_per:
        print(f"  No clones with ≥1 {src}→{dst} edge; skipping.")
        return None
    print(f"  Clones with ≥1 {src}→{dst} edge: {len(drain_per)}")

    # --- multiplicity classification ---
    obs_d = obs_lin[obs_lin["clone_id"].isin(drain_per.keys())].copy()
    mult_rows = []
    for cid in drain_per.keys():
        sub = obs_d[obs_d["clone_id"] == cid]
        s_tps = sub.loc[sub["tissue"] == src, "tp_int"].tolist()
        d_tps = sub.loc[sub["tissue"] == dst, "tp_int"].tolist()
        mult_rows.append({
            "clone_id": cid,
            "multiplicity_class": classify_multiplicity(s_tps, d_tps),
            "n_source_obs": len(set(s_tps)),
            "n_target_obs": len(set(d_tps)),
            "n_draining_edges": len(drain_per[cid]),
        })
    mult_df = pd.DataFrame(mult_rows)
    mult_df.to_csv(outdir / "draining_multiplicity.csv", index=False)
    print("  Multiplicity:")
    print(mult_df["multiplicity_class"].value_counts()
          .reindex(MULT_ORDER, fill_value=0).to_string())

    # --- candidate events with prior-dst filter ---
    raw_events = [(cid, tp1, tp2)
                  for cid, drains in drain_per.items()
                  for (tp1, tp2) in drains]
    filtered = []
    for (cid, t_s, t_d) in raw_events:
        sub = obs_d[obs_d["clone_id"] == cid]
        if not len(sub[(sub["tissue"] == dst) & (sub["tp_int"] < t_s)]):
            filtered.append((cid, t_s, t_d))
    print(f"  Candidate events: raw={len(raw_events)}, "
          f"post-prior-{dst}-filter={len(filtered)}")

    # --- soft-filter on cell counts ---
    events_ok = []
    for (cid, t_s, t_d) in filtered:
        s_g = cells_map.get((cid, src, t_s), np.array([], dtype=np.int64))
        t_g = cells_map.get((cid, dst, t_d), np.array([], dtype=np.int64))
        if len(s_g) >= MIN_CELLS_PER_SIDE and len(t_g) >= MIN_CELLS_PER_SIDE:
            events_ok.append((cid, t_s, t_d, s_g, t_g))
    print(f"  Events OT-eligible (≥{MIN_CELLS_PER_SIDE} cells/side): "
          f"{len(events_ok)}")
    if not events_ok:
        return None

    # --- PCA-OT for every event ---
    print(f"  Running PCA-OT on {len(events_ok)} events...")
    pca_results = []
    for (cid, t_s, t_d, s_g, t_g) in tqdm(events_ok, leave=False,
                                          desc=f"PCA-OT {src}→{dst}"):
        pi, mass, mc = run_event_ot(
            s_g, t_g, geometry="pca",
            PC=PC, log1p_X=log1p_X, global_to_local=global_to_local)
        pca_results.append({
            "clone_id": cid, "t_src": t_s, "t_dst": t_d,
            "src": s_g, "tgt": t_g, "pi": pi, "mass": mass,
            "mean_cost": mc,
        })

    # --- per-event gene delta on full transcriptome ---
    print(f"  Per-event gene deltas (log1p full transcriptome)...")
    per_event_delta = np.zeros((len(pca_results), n_genes), dtype=np.float32)
    src_blocks, tgt_blocks, masses = [], [], []
    for e_i, ev in enumerate(tqdm(pca_results, leave=False,
                                  desc=f"delta {src}→{dst}")):
        expr_s = _densify(log1p_X, ev["src"])
        expr_t = _densify(log1p_X, ev["tgt"])
        pi = ev["pi"]
        per_event_delta[e_i] = (pi.sum(axis=0) @ expr_t
                                - pi.sum(axis=1) @ expr_s)
        src_blocks.append(expr_s)
        tgt_blocks.append(expr_t)
        masses.append(ev["mass"])
    masses = np.asarray(masses, dtype=np.float32)
    total_mass = float(masses.sum())
    weighted_delta = (masses[:, None] * per_event_delta).sum(0) / total_mass
    delta_std = per_event_delta.std(axis=0)

    # --- matching balance (unbalanced-OT diagnostics) ---
    print("  Computing matching balance...")
    obs_phenotype_all = np.asarray(adata.obs["phenotype"].astype(str))
    qc_dir = outdir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    matching = compute_matching_balance(
        pca_results, obs_phenotype_all,
        src=src, dst=dst, outdir=outdir)
    plot_matching_balance(matching["match_per_phenotype"],
                            src=src, dst=dst, lineage=lineage,
                            outpath=qc_dir / "matching_balance.png")
    print(f"    matched fraction  source={matching['matched_fraction_source']:.3f}  "
          f"target={matching['matched_fraction_target']:.3f}")

    # --- cache per_event_delta + event_meta (lets us re-run perm / boot
    # / TF-activity without re-solving Sinkhorn)
    event_meta = pd.DataFrame([{
        "event_id": (f"{ev['clone_id']}|{ev['t_src']}->{ev['t_dst']}"),
        "clone_id": ev["clone_id"],
        "t_src": ev["t_src"], "t_dst": ev["t_dst"],
        "n_src": int(len(ev["src"])),
        "n_tgt": int(len(ev["tgt"])),
        "mass": ev["mass"],
        "mean_cost": ev["mean_cost"],
        "src_cell_indices": ";".join(map(str, ev["src"].tolist())),
        "tgt_cell_indices": ";".join(map(str, ev["tgt"].tolist())),
    } for ev in pca_results])
    event_meta.to_csv(outdir / "event_meta.csv", index=False)
    ped_df = pd.DataFrame(per_event_delta, columns=gene_names)
    ped_df.insert(0, "event_id", event_meta["event_id"].values)
    try:
        ped_df.to_parquet(outdir / "per_event_delta.parquet")
    except Exception:
        ped_df.to_csv(outdir / "per_event_delta.csv.gz",
                       index=False, compression="gzip")

    # --- bootstrap CI over events (mass-weighted) ---
    print(f"  Bootstrap CI (n={N_BOOTSTRAP})...")
    n_ev = len(pca_results)
    boot_means = np.zeros((N_BOOTSTRAP, n_genes), dtype=np.float32)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_ev, n_ev)
        m_b = masses[idx]
        boot_means[b] = (m_b[:, None] * per_event_delta[idx]).sum(0) \
                          / max(m_b.sum(), 1e-12)
    ci_lo = np.percentile(boot_means, 2.5, axis=0)
    ci_hi = np.percentile(boot_means, 97.5, axis=0)
    se_boot = boot_means.std(axis=0)

    # --- per-event consistency (fraction of events whose sign matches
    # the mass-weighted mean — independent of magnitude / null)
    consistency = (np.sign(per_event_delta)
                   == np.sign(weighted_delta)[None, :]).mean(axis=0).astype(
                       np.float32)

    # --- permutation null ---
    if skip_perm:
        print("  (skipping permutation null on request)")
        pvals = np.full(n_genes, np.nan, dtype=np.float64)
        padj = np.full(n_genes, np.nan, dtype=np.float64)
    else:
        print(f"  Permutation null (n={N_PERM})...")
        pooled = [np.vstack([src_blocks[i], tgt_blocks[i]])
                  for i in range(len(pca_results))]
        n_s_arr = np.array([s.shape[0] for s in src_blocks])
        obs_abs = np.abs(weighted_delta)
        perm_counts = np.zeros(n_genes, dtype=np.int64)
        for _ in tqdm(range(N_PERM), leave=False, desc=f"perm {src}→{dst}"):
            agg = np.zeros(n_genes, dtype=np.float32)
            for e_i in range(len(pca_results)):
                pool = pooled[e_i]
                idx = rng.permutation(pool.shape[0])
                ns = n_s_arr[e_i]
                m_s = pool[idx[:ns]].mean(axis=0)
                m_t = pool[idx[ns:]].mean(axis=0)
                agg += (masses[e_i] ** 2) * (m_t - m_s)
            perm = agg / total_mass
            perm_counts += (np.abs(perm) >= obs_abs).astype(np.int64)
        pvals = (perm_counts + 1.0) / (N_PERM + 1.0)
        order = np.argsort(pvals)
        ranked = pvals[order]
        n = len(pvals)
        padj_sorted = np.minimum.accumulate(
            (ranked * n / (np.arange(n) + 1))[::-1])[::-1]
        padj_sorted = np.minimum(padj_sorted, 1.0)
        padj = np.empty_like(padj_sorted)
        padj[order] = padj_sorted

    rewire_df = pd.DataFrame({
        "gene": gene_names,
        "mean_delta": weighted_delta,
        "std_delta": delta_std,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "se_bootstrap": se_boot,
        "consistency_score": consistency,
        "pval": pvals,
        "padj": padj,
    }).sort_values("mean_delta", ascending=False)
    rewire_df.to_csv(outdir / "rewiring_genes.csv", index=False)

    # --- full-expr-geometry OT (dual) ---
    print("  Full-expr OT...")
    fe_delta = np.zeros((len(events_ok), n_genes), dtype=np.float32)
    fe_masses = []
    for e_i, (cid, t_s, t_d, s_g, t_g) in enumerate(tqdm(
            events_ok, leave=False, desc=f"OT-fullexpr {src}→{dst}")):
        pi, mass, _ = run_event_ot(
            s_g, t_g, geometry="full",
            PC=PC, log1p_X=log1p_X, global_to_local=global_to_local)
        expr_s = _densify(log1p_X, s_g)
        expr_t = _densify(log1p_X, t_g)
        fe_delta[e_i] = pi.sum(0) @ expr_t - pi.sum(1) @ expr_s
        fe_masses.append(mass)
    fe_masses = np.asarray(fe_masses, dtype=np.float32)
    fe_total = float(fe_masses.sum())
    fe_mean_delta = (fe_masses[:, None] * fe_delta).sum(0) / fe_total
    fe_std = fe_delta.std(axis=0)
    fe_df = pd.DataFrame({
        "gene": gene_names,
        "mean_delta": fe_mean_delta,
        "std_delta": fe_std,
    }).sort_values("mean_delta", ascending=False)
    fe_df.to_csv(outdir / "rewiring_genes_fullexpr.csv", index=False)
    rho, _ = spearmanr(weighted_delta, fe_mean_delta)
    print(f"  Spearman(PCA mean_delta, fullexpr mean_delta) = {rho:.3f}")

    # --- GSEA prerank on PCA mean_delta ---
    print("  GSEA prerank...")
    rnk = (rewire_df.set_index("gene")["mean_delta"]
           .sort_values(ascending=False))
    rnk = rnk[~rnk.index.duplicated()]
    try:
        pre = gp.prerank(rnk=rnk, gene_sets=GENE_SETS, outdir=None,
                          seed=RNG_SEED, min_size=10, max_size=500,
                          permutation_num=1000, no_plot=True)
        gsea = pre.res2d.copy()
        gsea["NES"] = gsea["NES"].astype(float)
        gsea["FDR q-val"] = gsea["FDR q-val"].astype(float)
        gsea = gsea.sort_values("NES", ascending=False)
        gsea.to_csv(outdir / "gsea.csv", index=False)
        print(f"  Saved gsea.csv ({len(gsea)} terms)")
    except Exception as e:
        print(f"  GSEA failed: {e}")
        gsea = pd.DataFrame()

    return {
        "lineage": lineage, "src": src, "dst": dst,
        "n_clones": len(drain_per),
        "n_events_filtered": len(filtered),
        "n_events_ot": len(events_ok),
        "rewire_df": rewire_df,
        "fe_df": fe_df,
        "gsea": gsea,
        "mult_df": mult_df,
        "masses": masses,
        "weighted_delta": weighted_delta,
        "fe_mean_delta": fe_mean_delta,
        "spearman": rho,
        "gene_names": gene_names,
        "consistency": consistency,
        "matching": matching,
    }


# =====================================================================
# Matching balance (unbalanced-OT diagnostics)
# =====================================================================
def compute_matching_balance(pca_results, obs_phenotype, *, src, dst, outdir):
    """How much of each cell's "max possible" mass actually got matched?

    With unbalanced Sinkhorn, π row/col sums are NOT pinned to the input
    marginals. Per-cell matched_fraction = (row/col sum of π) / (uniform
    max). Aggregated per phenotype this answers:

      • source side, low matched_fraction → CSF subset has no good TP
        analogue (filtered out at the BBB, died in transit, or
        differentiated beyond recognition).
      • target side, low matched_fraction → TP subset wasn't seen in
        the matched CSF clones (recruited from elsewhere or arose by
        terminal differentiation in the destination).

    Writes:
      matching_per_event.csv     (per cell, per event)
      matching_per_phenotype.csv (aggregated)
    Returns the per-phenotype aggregate plus transition-level efficiencies.
    """
    rows = []
    for ev in pca_results:
        eid = f"{ev['clone_id']}|{ev['t_src']}->{ev['t_dst']}"
        pi = ev["pi"]
        src_g, tgt_g = ev["src"], ev["tgt"]
        n_s, n_t = len(src_g), len(tgt_g)
        m = float(min(n_s, n_t))
        a_max = m / n_s if n_s else 0.0
        b_max = m / n_t if n_t else 0.0
        src_assigned = pi.sum(axis=1)
        tgt_assigned = pi.sum(axis=0)
        for i, g in enumerate(src_g):
            rows.append({
                "event_id": eid, "side": "source", "tissue": src,
                "cell_index": int(g),
                "phenotype": obs_phenotype[int(g)],
                "mass_max": a_max,
                "mass_assigned": float(src_assigned[i]),
            })
        for j, g in enumerate(tgt_g):
            rows.append({
                "event_id": eid, "side": "target", "tissue": dst,
                "cell_index": int(g),
                "phenotype": obs_phenotype[int(g)],
                "mass_max": b_max,
                "mass_assigned": float(tgt_assigned[j]),
            })
    df = pd.DataFrame(rows)
    df["matched_fraction"] = (df["mass_assigned"]
                                / df["mass_max"].clip(lower=1e-12)).clip(0, 1)
    df.to_csv(outdir / "matching_per_event.csv", index=False)

    agg = (df.groupby(["side", "tissue", "phenotype"])
              .agg(n_cells=("cell_index", "count"),
                    total_mass=("mass_max", "sum"),
                    matched_mass=("mass_assigned", "sum"))
              .reset_index())
    agg["matched_fraction"] = (agg["matched_mass"]
                                / agg["total_mass"].clip(lower=1e-12))
    agg["unmatched_fraction"] = 1.0 - agg["matched_fraction"]
    agg.to_csv(outdir / "matching_per_phenotype.csv", index=False)

    eff_src = float(df.loc[df["side"] == "source", "mass_assigned"].sum()
                     / max(df.loc[df["side"] == "source",
                                     "mass_max"].sum(), 1e-12))
    eff_tgt = float(df.loc[df["side"] == "target", "mass_assigned"].sum()
                     / max(df.loc[df["side"] == "target",
                                     "mass_max"].sum(), 1e-12))
    return {
        "matched_fraction_source": eff_src,
        "matched_fraction_target": eff_tgt,
        "match_per_phenotype": agg,
    }


def plot_matching_balance(agg, *, src, dst, lineage, outpath):
    """Paired stacked bars per phenotype: matched (filled) vs unmatched
    (hatched) for the source and target tissues."""
    if agg is None or agg.empty:
        return
    lin_order = [p for p in TCELL_PHENOTYPE_ORDER if p.startswith(lineage)]
    src_lbl = TISSUE_LABELS.get(src, src)
    dst_lbl = TISSUE_LABELS.get(dst, dst)

    src_df = (agg[agg["side"] == "source"]
               .set_index("phenotype").reindex(lin_order))
    tgt_df = (agg[agg["side"] == "target"]
               .set_index("phenotype").reindex(lin_order))

    phenos = [p for p in lin_order
              if (p in src_df.index and pd.notna(src_df.loc[p, "n_cells"]))
              or (p in tgt_df.index and pd.notna(tgt_df.loc[p, "n_cells"]))]
    if not phenos:
        return
    pos = np.arange(len(phenos))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(phenos)), 4.2))
    for k, ph in enumerate(phenos):
        col = TCELL_PHENOTYPE_COLORS.get(ph, "#9e9e9e")
        # source bar (left)
        s = src_df.loc[ph] if ph in src_df.index else None
        if s is not None and pd.notna(s["n_cells"]):
            mf = s["matched_fraction"]
            ax.bar(pos[k] - w/2, mf, w, color=col, edgecolor="black",
                    linewidth=0.4, alpha=0.9)
            ax.bar(pos[k] - w/2, 1 - mf, w, bottom=mf, color=col,
                    edgecolor="black", linewidth=0.4, alpha=0.25,
                    hatch="///")
        # target bar (right)
        t = tgt_df.loc[ph] if ph in tgt_df.index else None
        if t is not None and pd.notna(t["n_cells"]):
            mf = t["matched_fraction"]
            ax.bar(pos[k] + w/2, mf, w, color=col, edgecolor="black",
                    linewidth=0.4, alpha=0.9)
            ax.bar(pos[k] + w/2, 1 - mf, w, bottom=mf, color=col,
                    edgecolor="black", linewidth=0.4, alpha=0.25,
                    hatch="///")
    ax.set_xticks(pos)
    ax.set_xticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in phenos],
                        rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("matched fraction of OT-feasible mass")
    ax.axhline(1.0, color="#888", lw=0.5, ls=":")
    legend = [
        Patch(facecolor="#666", alpha=0.9, label="matched"),
        Patch(facecolor="#666", alpha=0.25, hatch="///", label="unmatched"),
    ]
    ax.legend(handles=legend, frameon=False, fontsize=8, loc="lower right")
    ax.set_title(
        f"{lineage}  {src_lbl} → {dst_lbl}  matching balance "
        "(left bar = source, right bar = target)",
        fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# TF activity inference (ulm on the OT-derived delta)
# =====================================================================
def infer_tf_activity(weighted_delta, gene_names, collectri_net):
    """Univariate linear-model TF activity from a single-vector delta.

    Returns a DataFrame with columns: TF, activity, pval, padj.
    Returns empty frame on any failure / empty network.
    """
    if collectri_net is None or collectri_net.empty:
        return pd.DataFrame(columns=["TF", "activity", "pval", "padj"])
    mat = pd.DataFrame(weighted_delta[None, :].astype(np.float32),
                        columns=list(gene_names), index=["delta"])
    try:
        acts, pvals = dc.mt.ulm(data=mat, net=collectri_net,
                                  tmin=5, verbose=False)
    except Exception as e:
        print(f"  TF activity inference failed: {e}")
        return pd.DataFrame(columns=["TF", "activity", "pval", "padj"])
    tfs = list(acts.columns)
    activity = acts.iloc[0].to_numpy()
    pv = pvals.iloc[0].to_numpy()
    # BH adjust
    order = np.argsort(pv)
    ranked = pv[order]
    n_tf = len(ranked)
    if n_tf:
        padj_sorted = np.minimum.accumulate(
            (ranked * n_tf / (np.arange(n_tf) + 1))[::-1])[::-1]
        padj_sorted = np.minimum(padj_sorted, 1.0)
        padj = np.empty_like(padj_sorted)
        padj[order] = padj_sorted
    else:
        padj = pv
    return pd.DataFrame({
        "TF": tfs, "activity": activity, "pval": pv, "padj": padj,
    }).sort_values("padj")


# =====================================================================
# Phenotype anchoring + CollecTRI links
# =====================================================================
def compute_phenotype_anchors(adata, sel, *, src, dst, lineage, outdir):
    """For every selected gene: anchor to the phenotype with the highest
    z-scored mean log1p in the destination tissue (positive deltas) or
    the source tissue (negative deltas)."""
    obs_tissue = adata.obs["tissue"].astype(str)
    obs_pheno = adata.obs["phenotype"].astype(str)
    lin_mask = obs_pheno.str.startswith(lineage).values

    disp_genes = [g for g in sel["gene"] if g in adata.var_names]
    gene_idx = [adata.var_names.get_loc(g) for g in disp_genes]

    lin_phenos = [p for p in TCELL_PHENOTYPE_ORDER if p.startswith(lineage)]

    def mean_per_phenotype(tissue):
        m = lin_mask & (obs_tissue.values == tissue)
        if m.sum() == 0:
            return pd.DataFrame(columns=disp_genes)
        X = adata.layers["log1p"][m][:, gene_idx]
        if sp.issparse(X):
            X = X.toarray()
        df = pd.DataFrame(X, columns=disp_genes)
        df["phenotype"] = obs_pheno.values[m]
        agg = df.groupby("phenotype").mean()
        return agg.reindex([p for p in lin_phenos if p in agg.index])

    exp_dst = mean_per_phenotype(dst)
    exp_src = mean_per_phenotype(src)
    exp_dst.to_csv(outdir / "phenotype_expression_target.csv")
    exp_src.to_csv(outdir / "phenotype_expression_source.csv")

    def _z_argmax(mat, gene):
        if gene not in mat.columns or mat.shape[0] == 0:
            return None, np.nan
        col = mat[gene].to_numpy()
        if np.nanstd(col) < 1e-8:
            return mat.index[int(np.nanargmax(col))], 0.0
        z = (col - np.nanmean(col)) / np.nanstd(col)
        i = int(np.nanargmax(z))
        return mat.index[i], float(z[i])

    fallback = lin_phenos[0] if lin_phenos else "Unknown"
    sel = sel.copy()
    sel["anchor_phenotype"] = ""
    sel["anchor_tissue"] = ""
    sel["anchor_zscore"] = np.nan
    for i, row in sel.iterrows():
        if row["mean_delta"] > 0:
            ph, z = _z_argmax(exp_dst, row["gene"])
            tissue = dst
        else:
            ph, z = _z_argmax(exp_src, row["gene"])
            tissue = src
        sel.at[i, "anchor_phenotype"] = ph if ph is not None else fallback
        sel.at[i, "anchor_tissue"] = tissue
        sel.at[i, "anchor_zscore"] = z
    return sel


def compute_tf_links(sel, collectri_net, outdir, *, active_tfs=None):
    """Return TF→target edges where both endpoints are in sel.

    Only TFs in `active_tfs` (empirically inferred by ulm) are considered
    drivers — both for the `is_TF` flag (used by the star marker) and
    for arc origins. If `active_tfs` is None we fall back to the legacy
    'any CollecTRI TF in display' behaviour.
    """
    if collectri_net is None or collectri_net.empty:
        sel = sel.copy()
        sel["is_TF"] = False
        empty = pd.DataFrame(columns=["TF", "target", "mode",
                                       "TF_mean_delta", "target_mean_delta",
                                       "TF_anchor_phenotype",
                                       "target_anchor_phenotype"])
        empty.to_csv(outdir / "tf_target_links.csv", index=False)
        return sel, empty

    all_tfs = set(collectri_net["source"])
    display = set(sel["gene"].tolist())
    sel = sel.copy()
    if active_tfs is None:
        sel["is_TF"] = sel["gene"].isin(all_tfs)
    else:
        # active_tfs is a set of empirically-active TF symbols
        sel["is_TF"] = sel["gene"].isin(set(active_tfs))
    tfs_in_disp = sel.loc[sel["is_TF"], "gene"].tolist()

    rows = []
    for tf in tfs_in_disp:
        tf_md = float(sel.loc[sel["gene"] == tf, "mean_delta"].iloc[0])
        tf_anchor = sel.loc[sel["gene"] == tf, "anchor_phenotype"].iloc[0]
        sub = collectri_net[(collectri_net["source"] == tf) &
                             (collectri_net["target"].isin(display))]
        for _, r in sub.iterrows():
            tgt = r["target"]
            if tgt == tf:
                continue
            tgt_md = float(sel.loc[sel["gene"] == tgt, "mean_delta"].iloc[0])
            tgt_anchor = sel.loc[sel["gene"] == tgt,
                                  "anchor_phenotype"].iloc[0]
            rows.append({
                "TF": tf, "target": tgt,
                "mode": "act" if r["weight"] > 0 else "rep",
                "TF_mean_delta": tf_md, "target_mean_delta": tgt_md,
                "TF_anchor_phenotype": tf_anchor,
                "target_anchor_phenotype": tgt_anchor,
            })
    tf_links = pd.DataFrame(rows)
    tf_links.to_csv(outdir / "tf_target_links.csv", index=False)
    return sel, tf_links


# =====================================================================
# Main panel rendering — cleaned up
# =====================================================================
def select_display_genes(rewire_df, fe_df, *,
                           padj_thr=PADJ_THR, delta_thr=DELTA_THR,
                           max_genes=MAX_GENES):
    """Filter to figure-display genes: padj<thr, |delta|>thr, top max_genes.

    Uses rewire_df (PCA-OT) for mean_delta + CI + padj so the bars and
    their error bars are statistically consistent. fe_df contributes a
    cross-geometry sanity column (fe_mean_delta) for the CSV only.
    Falls back to delta-only filter when padj is all NaN (--skip-perm).
    """
    df = rewire_df.merge(
        fe_df[["gene", "mean_delta"]].rename(
            columns={"mean_delta": "fe_mean_delta"}),
        on="gene", how="left")
    if df["padj"].notna().any():
        mask = (df["padj"] < padj_thr) & (df["mean_delta"].abs() > delta_thr)
    else:
        mask = df["mean_delta"].abs() > delta_thr
    sel = df[mask].copy()
    if len(sel) > max_genes:
        sel = sel.reindex(
            sel["mean_delta"].abs().sort_values(ascending=False).index
        ).head(max_genes)
    sel = sel.sort_values("mean_delta", ascending=False).reset_index(drop=True)
    return sel


def _axis_aspect(ax):
    """Pixels-per-y-data-unit divided by pixels-per-x-data-unit.
    Needed because matplotlib's arc3 rad operates in display coords, so
    converting a desired apex height in data y units to a rad requires
    the visual aspect ratio of the axis."""
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (bbox.height / (y1 - y0)) / (bbox.width / (x1 - x0))


def _draw_arcs(ax, sel, tf_links, *, y_base=0.04):
    """All TF→target arcs in one shared lane.

    - TF dot at (x_tf, y_base).
    - Arcs bow strictly UP with apex height ∝ sqrt(distance).
    - Per-link stable hash jitter spreads arcs from the same TF.
    - Black solid for activation, black dashed for repression.
    - NO arrowheads — with hub TFs that have many targets, arrowheads
      pile up and become noise; the curve geometry already encodes
      source (TF dot) → target (curve termination).
    """
    if len(sel) == 0:
        return
    g_to_x = {g: i for i, g in enumerate(sel["gene"].tolist())}
    tf_genes = sel.loc[sel["is_TF"], "gene"].tolist()

    for tf in tf_genes:
        ax.scatter([g_to_x[tf]], [y_base], s=22,
                   color="#101010", edgecolor="white",
                   linewidth=0.5, zorder=6)

    if tf_links is None or len(tf_links) == 0:
        return

    dists = []
    for _, row in tf_links.iterrows():
        if row["TF"] in g_to_x and row["target"] in g_to_x:
            dists.append(abs(g_to_x[row["target"]] - g_to_x[row["TF"]]))
    if not dists:
        return
    max_dist = max(dists)
    # Tightened apex range: arcs read as a slim ideogram, not a dome.
    apex_floor = 0.20
    apex_ceil  = 0.70
    aspect = _axis_aspect(ax)

    ordered = tf_links.copy()
    ordered["__o"] = (ordered["mode"] == "rep").astype(int)
    ordered = ordered.sort_values(["__o", "TF", "target"], kind="stable")

    for _, row in ordered.iterrows():
        tf, tgt, mode = row["TF"], row["target"], row["mode"]
        if tf == tgt or tf not in g_to_x or tgt not in g_to_x:
            continue
        x1, x2 = g_to_x[tf], g_to_x[tgt]
        dist = abs(x2 - x1)
        if dist < 1:
            continue
        apex_norm = (dist / max_dist) ** 0.5
        h_jit = ((hash((tf, tgt)) & 0xFFFF) / 0xFFFF - 0.5) * 0.10
        apex_h = apex_floor + (apex_ceil - apex_floor) * apex_norm + h_jit
        apex_h = min(max(apex_h, 0.18), 0.94)
        magnitude = 2 * apex_h / dist * aspect
        rad = -magnitude * np.sign(x2 - x1)
        style = "-" if mode == "act" else (0, (3.8, 2.2))
        # Plain line, no arrowhead.
        arc = FancyArrowPatch(
            (x1, y_base + 0.005), (x2, y_base + 0.005),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            color=ARC_COLOR, lw=ARC_LW, alpha=ARC_ALPHA,
            linestyle=style,
            shrinkA=1.5, shrinkB=1.5,
            zorder=3 if mode == "act" else 4,
            capstyle="round", joinstyle="round",
        )
        ax.add_patch(arc)


def render_panel(*, sel, tf_links, src, dst, lineage, n_total_genes,
                 outdir, padj_thr=PADJ_THR, delta_thr=DELTA_THR,
                 bar_h=4.6, show_title=True, legend_size=7.5):
    n = len(sel)
    if n == 0:
        print("  [no display genes — skipping panel]")
        return

    n_tfs = int(sel["is_TF"].sum())
    n_links = len(tf_links) if tf_links is not None else 0

    # Slim arc lane — should look like a header band, not a dome.
    arc_h = max(0.8, 0.06 * max(n_links, n_tfs))
    arc_h = min(arc_h, 2.0)
    # bar_h is parameter-controlled so callers (e.g. figure3 composer)
    # can ask for a wider/shorter panel.
    lab_h = 1.2
    # Fixed-inch headroom for title/legend so they don't collide when
    # the figure is short. When the title is suppressed, leave just
    # enough vertical space for the legend.
    HEADER_IN = 1.4 if show_title else 0.55
    FOOTER_IN = 0.4

    fig_w = max(11.0, 0.42 * n)
    fig_h = arc_h + bar_h + lab_h + HEADER_IN + FOOTER_IN

    top_frac    = 1.0 - HEADER_IN / fig_h
    bottom_frac = FOOTER_IN / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[arc_h, bar_h, lab_h],
        hspace=0.04,
        left=0.05, right=0.99, top=top_frac, bottom=bottom_frac,
    )
    ax_arc = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharex=ax_arc)
    ax_lab = fig.add_subplot(gs[2], sharex=ax_arc)

    xs = np.arange(n)
    heights = sel["mean_delta"].to_numpy()
    # Asymmetric 95% bootstrap CI: lower / upper distance from bar top
    if {"ci_lo", "ci_hi"}.issubset(sel.columns) \
            and sel["ci_lo"].notna().any():
        lo = (heights - sel["ci_lo"].to_numpy()).clip(min=0)
        hi = (sel["ci_hi"].to_numpy() - heights).clip(min=0)
        yerr = np.vstack([lo, hi])
        err_top = np.maximum(heights + hi, heights)
        err_bot = np.minimum(heights - lo, heights)
    else:
        std = sel["std_delta"].to_numpy()
        yerr = std
        err_top = heights + std
        err_bot = heights - std
    colors = [TCELL_PHENOTYPE_COLORS.get(p, "#9e9e9e")
              for p in sel["anchor_phenotype"]]

    # ----- bars -----
    ax_bar.bar(xs, heights, yerr=yerr, color=colors, alpha=0.92,
                edgecolor="black", linewidth=0.35, width=0.78,
                error_kw=dict(ecolor="#444", lw=0.6, capsize=2))
    ax_bar.axhline(0, color="black", lw=0.6)
    top_y = np.where(heights >= 0, err_top, err_bot)
    for i, row in sel.iterrows():
        if row["is_TF"]:
            offset = 0.35 if heights[i] >= 0 else -0.35
            ax_bar.text(i, top_y[i] + offset, "★", ha="center",
                         va="bottom" if heights[i] >= 0 else "top",
                         fontsize=10, color="#222")
    ymin, ymax = ax_bar.get_ylim()
    ax_bar.set_ylim(ymin * 1.10, ymax * 1.18)
    src_lbl = TISSUE_LABELS.get(src, src)
    dst_lbl = TISSUE_LABELS.get(dst, dst)
    ax_bar.set_ylabel(
        f"mean Δ expression  ({src_lbl} → {dst_lbl}, PCA-OT)  ·  95% CI",
        fontsize=10)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.set_xlim(-0.5, n - 0.5)
    for s in ("top", "right"):
        ax_bar.spines[s].set_visible(False)
    ax_bar.spines["bottom"].set_position(("data", 0))
    ax_bar.yaxis.grid(True, color="#e8e8e8", lw=0.5, zorder=0)
    ax_bar.set_axisbelow(True)

    # ----- arc lane -----
    ax_arc.set_xlim(-0.5, n - 0.5)
    ax_arc.set_ylim(0, 1.0)
    ax_arc.set_axis_off()
    # Force layout so _axis_aspect() reads the final on-screen bbox.
    fig.canvas.draw()
    _draw_arcs(ax_arc, sel, tf_links, y_base=0.04)

    # ----- gene labels (visually centered on each bar) -----
    # rotation_mode='anchor' + ha='center' anchors the rotated label so
    # the bar's x-center bisects the text diagonal; this lines the label
    # up under its bar instead of its right-end sitting on the tick.
    ax_lab.set_xlim(-0.5, n - 0.5)
    ax_lab.set_ylim(0, 1)
    ax_lab.set_xticks(xs)
    text_labels = ax_lab.set_xticklabels(
        sel["gene"].tolist(),
        rotation=45, ha="center", va="top",
        rotation_mode="anchor", fontsize=9)
    for lbl, is_tf in zip(text_labels, sel["is_TF"]):
        if is_tf:
            lbl.set_fontweight("bold")
            lbl.set_fontsize(10)
    ax_lab.set_yticks([])
    for s in ("top", "left", "right", "bottom"):
        ax_lab.spines[s].set_visible(False)
    ax_lab.tick_params(axis="x", length=0, pad=4)

    # ----- legend -----
    lin_order = [p for p in TCELL_PHENOTYPE_ORDER if p.startswith(lineage)]
    present = [p for p in lin_order if p in set(sel["anchor_phenotype"])]
    handles = [Patch(facecolor=TCELL_PHENOTYPE_COLORS[p],
                       label=TCELL_PHENOTYPE_LABELS.get(p, p))
               for p in present]
    handles += [
        Line2D([0], [0], marker="*", color="w",
                markerfacecolor="#222", markersize=10, label="TF"),
        Line2D([0], [0], color=ARC_COLOR, lw=1.5, label="activation"),
        Line2D([0], [0], color=ARC_COLOR, lw=1.5,
                ls=(0, (4, 2)), label="repression"),
    ]
    # Anchor legend in the fixed inch-based header so it doesn't
    # collide with the (also inch-anchored) title. When title is
    # suppressed, legend sits closer to the top of the figure.
    if show_title:
        legend_y = 1.0 - 0.28 * (HEADER_IN / fig_h)
    else:
        legend_y = 1.0 - 0.10 * (HEADER_IN / fig_h)
    fig.legend(handles=handles, loc="upper right",
                ncol=min(len(handles), 5), fontsize=legend_size,
                frameon=False,
                bbox_to_anchor=(0.99, legend_y),
                handlelength=1.6, columnspacing=1.2)

    if show_title:
        title_y    = 1.0 - 0.18 * (HEADER_IN / fig_h)
        subtitle_y = 1.0 - 0.62 * (HEADER_IN / fig_h)
        fig.suptitle(
            f"{lineage} {src_lbl} → {dst_lbl} gene-level rewiring "
            "(OT-derived, anchored to phenotype identity)",
            fontsize=12, fontweight="bold",
            y=title_y, x=0.05, ha="left")
        fig.text(0.05, subtitle_y,
                 f"n={n} displayed (padj<{padj_thr}, |Δ|>{delta_thr}; "
                 f"of {n_total_genes:,} tested) ·  bar = highest-z "
                 f"phenotype in target tissue  ·  TF★ + arcs: CollecTRI",
                 ha="left", fontsize=8.5, color="#555")

    fig.savefig(outdir / "panel.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(outdir / "panel.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved panel.png + panel.pdf")


# =====================================================================
# QC plots
# =====================================================================
def render_qc_plots(*, result, outdir, src, dst, lineage):
    qc = outdir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    mult_df = result["mult_df"]
    rewire_df = result["rewire_df"]
    gsea = result["gsea"]
    masses = result["masses"]
    weighted_delta = result["weighted_delta"]
    fe_mean_delta = result["fe_mean_delta"]
    rho = result["spearman"]

    # multiplicity breakdown
    counts = (mult_df["multiplicity_class"].value_counts()
              .reindex(MULT_ORDER, fill_value=0))
    fig, ax = plt.subplots(figsize=(4, 4))
    bottom = 0
    for cls in MULT_ORDER:
        ax.bar([f"{src}→{dst}"], [counts[cls]], bottom=bottom,
                color=MULT_COLORS[cls], edgecolor="white", label=cls)
        bottom += counts[cls]
    ax.set_ylabel("Draining clones")
    ax.set_title(f"Multiplicity  ({lineage}  {src}→{dst})", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(qc / "multiplicity_breakdown.png", dpi=DPI)
    plt.close(fig)

    # event mass
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(masses, bins=30,
             color=LINEAGE_COLORS.get(lineage, "#666"), edgecolor="white")
    ax.set_xlabel("Total transport mass per event")
    ax.set_ylabel("Events")
    ax.set_title(f"Event mass  ({lineage}  {src}→{dst})", fontsize=10)
    fig.tight_layout()
    fig.savefig(qc / "event_mass_distribution.png", dpi=DPI)
    plt.close(fig)

    # volcano
    fig, ax = plt.subplots(figsize=(6, 5))
    eps = 1e-300
    if rewire_df["padj"].notna().any():
        neglogp = -np.log10(np.maximum(rewire_df["padj"].values, eps))
        ax.scatter(rewire_df["mean_delta"], neglogp, s=4, alpha=0.4,
                    color="#666", linewidths=0)
        top = pd.concat([rewire_df.nlargest(10, "mean_delta"),
                         rewire_df.nsmallest(10, "mean_delta")])
        texts = []
        for _, row in top.iterrows():
            y = -np.log10(max(row["padj"], eps))
            ax.scatter(row["mean_delta"], y, s=14,
                        color=LINEAGE_COLORS.get(lineage, "#666"))
            texts.append(ax.text(row["mean_delta"], y, row["gene"],
                                  fontsize=7))
        try:
            adjust_text(texts, ax=ax,
                         arrowprops=dict(arrowstyle="-",
                                          color="grey", lw=0.4))
        except Exception:
            pass
    ax.axvline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel(f"mean delta ({src}→{dst}, PCA OT)")
    ax.set_ylabel("-log10 padj")
    ax.set_title(f"Rewiring volcano ({lineage}  {src}→{dst})", fontsize=10)
    fig.tight_layout()
    fig.savefig(qc / "rewiring_volcano.png", dpi=DPI)
    plt.close(fig)

    # gsea top
    if len(gsea):
        top_up = gsea.nlargest(10, "NES")
        top_dn = gsea.nsmallest(10, "NES")
        top = pd.concat([top_up, top_dn[::-1]])
        fig, ax = plt.subplots(figsize=(7, 6))
        bar_colors = ["#c51b8a" if v > 0 else "#2166ac" for v in top["NES"]]
        ax.barh(np.arange(len(top))[::-1], top["NES"], color=bar_colors,
                 edgecolor="white")
        ax.set_yticks(np.arange(len(top))[::-1])
        ax.set_yticklabels(top["Term"].str.slice(0, 60), fontsize=7)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("NES")
        ax.set_title(f"GSEA ({lineage}  {src}→{dst})", fontsize=10)
        fig.tight_layout()
        fig.savefig(qc / "gsea_top.png", dpi=DPI)
        plt.close(fig)

    # geometry comparison
    gene_names = result["gene_names"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(weighted_delta, fe_mean_delta, s=3, alpha=0.3, color="#888")
    top_idx = np.argsort(-np.abs(weighted_delta))[:20]
    texts = []
    for i in top_idx:
        ax.scatter(weighted_delta[i], fe_mean_delta[i], s=14,
                    color=LINEAGE_COLORS.get(lineage, "#666"))
        texts.append(ax.text(weighted_delta[i], fe_mean_delta[i],
                              gene_names[i], fontsize=7))
    try:
        adjust_text(texts, ax=ax,
                     arrowprops=dict(arrowstyle="-",
                                      color="grey", lw=0.4))
    except Exception:
        pass
    lim = max(np.abs(weighted_delta).max(), np.abs(fe_mean_delta).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
    ax.set_xlabel("mean delta (PCA OT)")
    ax.set_ylabel("mean delta (full-expr OT)")
    ax.set_title(f"Geometry comparison  Spearman={rho:.2f} "
                  f"({lineage}  {src}→{dst})", fontsize=10)
    fig.tight_layout()
    fig.savefig(qc / "geometry_comparison.png", dpi=DPI)
    plt.close(fig)


# =====================================================================
# Cross-transition aggregation
# =====================================================================
def _collect_per_transition(rows):
    """rows = list of (lineage, src, dst, rewire_df, tf_activity_df).
    Returns one tidy long DataFrame across all transitions."""
    out = []
    for (lineage, src, dst, rewire_df, tf_act) in rows:
        if rewire_df is None or rewire_df.empty:
            continue
        df = rewire_df.copy()
        df["lineage"] = lineage
        df["src"] = src
        df["dst"] = dst
        df["transition"] = f"{src}_to_{dst}"
        # mark TF activity if available
        if tf_act is not None and not tf_act.empty:
            act_map = dict(zip(tf_act["TF"], tf_act["activity"]))
            padj_map = dict(zip(tf_act["TF"], tf_act["padj"]))
            df["tf_activity"] = df["gene"].map(act_map)
            df["tf_activity_padj"] = df["gene"].map(padj_map)
            df["is_active_TF"] = (df["tf_activity_padj"]
                                   .fillna(1.0) <= TF_ACTIVITY_PADJ)
        else:
            df["tf_activity"] = np.nan
            df["tf_activity_padj"] = np.nan
            df["is_active_TF"] = False
        out.append(df)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def build_cross_transition_outputs(per_transition_rows, root_outdir):
    """Write: all_rewiring_long.csv (tidy), cross_transition_delta.csv
    (wide pivot), tissue_pair_specificity.csv (per-gene rank)."""
    long_df = _collect_per_transition(per_transition_rows)
    if long_df.empty:
        print("  [no per-transition results — skipping cross-transition outputs]")
        return long_df
    cols = ["lineage", "transition", "src", "dst", "gene",
            "mean_delta", "se_bootstrap", "ci_lo", "ci_hi",
            "consistency_score", "pval", "padj",
            "tf_activity", "tf_activity_padj", "is_active_TF"]
    cols = [c for c in cols if c in long_df.columns]
    long_df[cols].to_csv(root_outdir / "all_rewiring_long.csv", index=False)
    print(f"  Saved all_rewiring_long.csv  ({len(long_df):,} rows)")

    # Wide pivot: mean_delta per (lineage, gene) × transition
    wide = (long_df.pivot_table(index=["lineage", "gene"],
                                  columns="transition",
                                  values="mean_delta",
                                  aggfunc="first")
            .reset_index())
    wide.to_csv(root_outdir / "cross_transition_delta.csv", index=False)
    print(f"  Saved cross_transition_delta.csv  ({len(wide):,} rows)")

    # Specificity table: for each (lineage, gene), count significant
    # transitions and identify the strongest one.
    sig = (long_df["padj"] < PADJ_THR) & \
          (long_df["mean_delta"].abs() > DELTA_THR)
    long_df["__sig"] = sig
    grp = long_df.groupby(["lineage", "gene"])
    rows = []
    for (lin, g), sub in grp:
        sub_sig = sub[sub["__sig"]]
        n_sig = int(sub_sig.shape[0])
        if not len(sub):
            continue
        i_max = sub["mean_delta"].abs().idxmax()
        top_tr = sub.loc[i_max, "transition"]
        top_dl = float(sub.loc[i_max, "mean_delta"])
        rows.append({
            "lineage": lin, "gene": g,
            "n_significant_transitions": n_sig,
            "specificity_score": (1.0 / n_sig) if n_sig else 0.0,
            "top_transition": top_tr,
            "top_mean_delta": top_dl,
            "max_abs_delta": float(sub["mean_delta"].abs().max()),
            "significant_in":
                ";".join(sorted(sub_sig["transition"].unique())),
        })
    spec_df = (pd.DataFrame(rows)
               .sort_values(["lineage", "n_significant_transitions",
                              "max_abs_delta"],
                             ascending=[True, True, False]))
    spec_df.to_csv(root_outdir / "tissue_pair_specificity.csv",
                    index=False)
    print(f"  Saved tissue_pair_specificity.csv  ({len(spec_df):,} rows)")
    return long_df


def build_pathway_heatmap(per_transition_gsea, root_outdir,
                            top_n_pathways=30):
    """Cross-transition × pathway NES heatmap from per-transition gsea
    DataFrames. per_transition_gsea = list of (lineage, src, dst, gsea_df)."""
    rows = []
    for (lin, src, dst, g) in per_transition_gsea:
        if g is None or g.empty:
            continue
        sub = g[["Term", "NES", "FDR q-val"]].copy()
        sub["lineage"] = lin
        sub["transition"] = f"{src}_to_{dst}"
        rows.append(sub)
    if not rows:
        print("  [no GSEA results — skipping pathway heatmap]")
        return
    g_long = pd.concat(rows, ignore_index=True)
    g_long.to_csv(root_outdir / "all_gsea_long.csv", index=False)
    g_long["col"] = g_long["lineage"] + ":" + g_long["transition"]

    nes_mat = (g_long.pivot_table(index="Term", columns="col",
                                    values="NES", aggfunc="first")
                .fillna(0))
    fdr_mat = (g_long.pivot_table(index="Term", columns="col",
                                    values="FDR q-val", aggfunc="first")
                .fillna(1))
    # Pick top N pathways by max |NES|
    top = nes_mat.abs().max(axis=1).sort_values(ascending=False).head(
        top_n_pathways).index
    nes_top = nes_mat.loc[top]
    fdr_top = fdr_mat.loc[top]

    # Order columns: CD8 first, then CD4; within each, transition
    col_order = sorted(nes_mat.columns.tolist(),
                        key=lambda c: (0 if c.startswith("CD8") else 1, c))
    nes_top = nes_top[col_order]
    fdr_top = fdr_top[col_order]

    fig_h = max(6, 0.35 * len(top))
    fig_w = max(8, 1.3 * len(col_order))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(abs(nes_top.values.min()), abs(nes_top.values.max()), 1.0)
    im = ax.imshow(nes_top.values, aspect="auto",
                    cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(nes_top.shape[1]))
    ax.set_xticklabels(nes_top.columns, rotation=45, ha="right",
                        fontsize=8)
    ax.set_yticks(np.arange(nes_top.shape[0]))
    ax.set_yticklabels([t[:60] for t in nes_top.index], fontsize=7)
    # FDR stars
    for i in range(nes_top.shape[0]):
        for j in range(nes_top.shape[1]):
            fdr = fdr_top.values[i, j]
            mark = ("***" if fdr < 0.001
                    else "**" if fdr < 0.01
                    else "*" if fdr < 0.05
                    else "")
            if mark:
                ax.text(j, i, mark, ha="center", va="center",
                         fontsize=6, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("NES", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(
        f"Cross-transition pathway NES  ·  top {len(top)} by |NES|  "
        "(* FDR<0.05, ** <0.01, *** <0.001)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(root_outdir / "pathway_heatmap.png", dpi=DPI,
                 bbox_inches="tight")
    fig.savefig(root_outdir / "pathway_heatmap.pdf",
                 bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pathway_heatmap.png + pathway_heatmap.pdf  "
          f"({len(top)} pathways × {len(col_order)} transitions)")


# =====================================================================
# Main
# =====================================================================
def _parse_transition_arg(s):
    if not s:
        return list(DEFAULT_TRANSITIONS)
    parsed = []
    for tok in s.split(","):
        tok = tok.strip()
        if "_to_" not in tok:
            raise SystemExit(
                f"--transitions must use SRC_to_DST syntax (got '{tok}')")
        a, b = tok.split("_to_", 1)
        parsed.append((a.strip().upper(), b.strip().upper()))
    return parsed


def main():
    args = _parse_args()
    lineages = (("CD8", "CD4") if args.lineage == "both"
                else (args.lineage,))
    transitions = _parse_transition_arg(args.transitions)
    rng = np.random.default_rng(RNG_SEED)

    print(f"Loading {paths.H5AD_TCELLS.name}...")
    adata = sc.read(str(paths.H5AD_TCELLS))
    if "log1p" not in adata.layers:
        raise RuntimeError("Expected 'log1p' layer on adata.")
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    print(f"Loading {CLONE_ARCH_CSV.name}...")
    clones = pd.read_csv(CLONE_ARCH_CSV)
    clones["timepoint"] = clones.get("timepoint", "")

    print("Loading CollecTRI (human)...")
    try:
        collectri_net = dc.op.collectri(organism="human")
        print(f"  {len(collectri_net):,} edges")
    except Exception as e:
        print(f"  CollecTRI failed to load: {e}")
        collectri_net = pd.DataFrame()

    summary_rows = []
    per_transition_rewire = []   # for tidy long + specificity
    per_transition_gsea = []     # for pathway heatmap
    per_transition_matching = [] # for cross-transition matching summary
    for lineage in lineages:
        clones_lin = clones[clones["lineage"] == lineage].copy()
        if clones_lin.empty:
            print(f"\n[{lineage}] no clones in archetype table; skipping.")
            continue
        setup = setup_lineage(adata, lineage)
        for (src, dst) in transitions:
            if src not in ("PBMC", "CSF", "TP") or \
               dst not in ("PBMC", "CSF", "TP") or src == dst:
                print(f"  Skipping invalid transition {src}→{dst}")
                continue
            outdir = OUT_DIR / f"{src}_to_{dst}" / lineage
            outdir.mkdir(parents=True, exist_ok=True)
            result = run_transition(
                adata=adata, clones_lin=clones_lin, setup=setup,
                lineage=lineage, src=src, dst=dst, outdir=outdir,
                rng=rng, skip_perm=args.skip_perm)
            if result is None:
                summary_rows.append({
                    "lineage": lineage, "src": src, "dst": dst,
                    "n_clones": 0, "n_events_filtered": 0,
                    "n_events_ot": 0, "n_genes_displayed": 0,
                    "spearman_pca_fullexpr": np.nan,
                    "n_active_tfs": 0,
                    "top5_up": "", "top5_down": "",
                })
                continue

            # Empirical TF activity (ulm on the weighted delta)
            print("  Inferring TF activity (ulm)...")
            tf_act = infer_tf_activity(
                result["weighted_delta"], result["gene_names"],
                collectri_net)
            if not tf_act.empty:
                tf_act.to_csv(outdir / "tf_activity.csv", index=False)
                active_tfs = set(tf_act.loc[
                    tf_act["padj"] <= TF_ACTIVITY_PADJ, "TF"].tolist())
                print(f"    {len(active_tfs)}/{len(tf_act)} TFs active "
                      f"at padj<={TF_ACTIVITY_PADJ}")
            else:
                active_tfs = set()

            # Phenotype anchoring + (active-TF-filtered) link extraction
            sel = select_display_genes(
                result["rewire_df"], result["fe_df"],
                padj_thr=args.padj_thr, delta_thr=args.delta_thr,
                max_genes=args.max_genes)
            sel = compute_phenotype_anchors(
                adata, sel, src=src, dst=dst,
                lineage=lineage, outdir=outdir)
            sel, tf_links = compute_tf_links(
                sel, collectri_net, outdir, active_tfs=active_tfs)

            # selected_genes.csv
            n_tgt = (tf_links.groupby("TF").size().to_dict()
                     if len(tf_links) else {})
            sel["n_targets_in_display"] = (
                sel["gene"].map(n_tgt).fillna(0).astype(int))
            keep_cols = [c for c in [
                "gene", "mean_delta", "std_delta", "ci_lo", "ci_hi",
                "se_bootstrap", "consistency_score", "padj",
                "fe_mean_delta", "anchor_phenotype", "anchor_tissue",
                "anchor_zscore", "is_TF", "n_targets_in_display",
            ] if c in sel.columns]
            sel[keep_cols].to_csv(outdir / "selected_genes.csv", index=False)

            render_panel(
                sel=sel, tf_links=tf_links, src=src, dst=dst,
                lineage=lineage,
                n_total_genes=len(result["rewire_df"]),
                outdir=outdir,
                padj_thr=args.padj_thr, delta_thr=args.delta_thr,
                bar_h=args.bar_h, show_title=not args.no_title,
                legend_size=args.legend_size)
            render_qc_plots(result=result, outdir=outdir,
                             src=src, dst=dst, lineage=lineage)

            per_transition_rewire.append(
                (lineage, src, dst, result["rewire_df"], tf_act))
            per_transition_gsea.append(
                (lineage, src, dst, result["gsea"]))
            mp = result["matching"]["match_per_phenotype"].copy()
            mp["lineage"] = lineage
            mp["src"] = src
            mp["dst"] = dst
            mp["transition"] = f"{src}_to_{dst}"
            per_transition_matching.append(mp)

            top5_up = result["rewire_df"].nlargest(5, "mean_delta")["gene"].tolist()
            top5_dn = result["rewire_df"].nsmallest(5, "mean_delta")["gene"].tolist()
            summary_rows.append({
                "lineage": lineage, "src": src, "dst": dst,
                "n_clones": result["n_clones"],
                "n_events_filtered": result["n_events_filtered"],
                "n_events_ot": result["n_events_ot"],
                "n_genes_displayed": len(sel),
                "spearman_pca_fullexpr": result["spearman"],
                "n_active_tfs": len(active_tfs),
                "matched_fraction_source":
                    result["matching"]["matched_fraction_source"],
                "matched_fraction_target":
                    result["matching"]["matched_fraction_target"],
                "top5_up": ";".join(top5_up),
                "top5_down": ";".join(top5_dn),
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUT_DIR / "run_summary.csv", index=False)
        print("\n========================================")
        print("Run summary:")
        print(summary_df.to_string(index=False))

    if per_transition_rewire:
        print("\nBuilding cross-transition outputs...")
        build_cross_transition_outputs(per_transition_rewire, OUT_DIR)
        build_pathway_heatmap(per_transition_gsea, OUT_DIR)
    if per_transition_matching:
        long_match = pd.concat(per_transition_matching, ignore_index=True)
        cols = ["lineage", "transition", "src", "dst", "side", "tissue",
                "phenotype", "n_cells", "total_mass", "matched_mass",
                "matched_fraction", "unmatched_fraction"]
        long_match[[c for c in cols if c in long_match.columns]].to_csv(
            OUT_DIR / "matching_balance_long.csv", index=False)
        print(f"  Saved matching_balance_long.csv  "
              f"({len(long_match):,} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
