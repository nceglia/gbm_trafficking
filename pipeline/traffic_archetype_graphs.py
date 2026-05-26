# %%
"""Clonal trafficking archetypes as retained-path graphs.

For each clone, identify observed (tissue, timepoint) nodes on the
3 × T grid, then connect consecutive observed nodes by the cheapest
forward path under within-tissue=1 / cross-tissue=CROSS_WEIGHT edge
weights. The union of all path edges is the clone's retained graph.
Hash by canonical edge list; clones with identical edge sets form
one archetype. No clustering, no merging — strict equality.

Reads:
  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad   (paths.H5AD_TCELLS)

Writes to results/traffic_archetype_graphs/:
  clone_archetypes.csv, archetype_graphs.csv, archetype_summary.csv,
  subset_matrix.csv,
  archetype_top_graphs.png, archetype_size_distribution.png,
  subset_diagnostic.png, archetype_phenotypes.png
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.patches import FancyArrowPatch
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.style import (  # noqa: E402
    LINEAGE_COLORS,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_ORDER,
)

OUT_DIR = paths.TRAFFIC_ARCHETYPE_GRAPHS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = list(TISSUE_ORDER)
TISSUE_IDX = {t: i for i, t in enumerate(TISSUES)}
MIN_OBSERVED_NODES = 3
CROSS_WEIGHT = 2  # within-tissue edge weight is 1
TOP_N = 20
DPI = 200


# %%
# =========================================================
# Load adata + eligible clones
# =========================================================
print(f"Loading {paths.H5AD_TCELLS.name}...")
adata = sc.read(str(paths.H5AD_TCELLS))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

obs = adata.obs[["trb", "tissue", "timepoint", "phenotype",
                  "patient", "sample"]].copy()
obs = obs[obs["trb"].notna() & (obs["trb"].astype(str) != "")]
for c in ("trb", "tissue", "timepoint", "phenotype", "patient", "sample"):
    obs[c] = obs[c].astype(str)
obs = obs[obs["tissue"].isin(TISSUES)]
obs["clone_id"] = obs["patient"] + "|" + obs["trb"]
obs["lineage"] = np.where(obs["phenotype"].str.contains("CD8"), "CD8", "CD4")

TPS = sorted(obs["timepoint"].unique(), key=lambda s: int(s))
T = len(TPS)
TP_IDX = {t: i for i, t in enumerate(TPS)}
print(f"  timepoints: {TPS}  (T={T})")

bin_obs = (obs.groupby(["clone_id", "tissue", "timepoint"], observed=True)
              .size().reset_index(name="n_cells"))
n_nodes_per_clone = (bin_obs.groupby("clone_id").size())
eligible_clones = n_nodes_per_clone[
    n_nodes_per_clone >= MIN_OBSERVED_NODES].index.tolist()
obs_e = obs[obs["clone_id"].isin(eligible_clones)].copy()
bin_obs_e = bin_obs[bin_obs["clone_id"].isin(eligible_clones)].copy()

clone_meta = (obs_e.groupby("clone_id", observed=True)
              .agg(patient=("patient", "first"),
                   trb=("trb", "first"),
                   lineage=("lineage", lambda s: s.mode().iloc[0]),
                   clone_size=("clone_id", "size"))
              .reset_index())
clone_meta["n_observed_nodes"] = (
    bin_obs_e.groupby("clone_id").size().reindex(clone_meta["clone_id"])
    .values)

print(f"\nEligible clones (≥{MIN_OBSERVED_NODES} observed nodes): "
      f"{len(eligible_clones):,}")
print("\nDistribution of n_observed_nodes (eligible clones):")
print(clone_meta["n_observed_nodes"]
      .describe(percentiles=[.5, .75, .9, .99]).to_string())
print("\nLineage breakdown:")
print(clone_meta["lineage"].value_counts().to_string())


# %%
# =========================================================
# Graph construction (with calibration)
# =========================================================
print("\nBuilding per-clone retained-path graphs...")
bin_obs_e_idx = bin_obs_e.set_index("clone_id").sort_index()
bin_obs_groups = bin_obs_e_idx.groupby(level=0)
INF = float("inf")


def _retained_edges(observed_set):
    """observed_set: set of (tissue_idx, tp_idx). Return frozenset of
    canonical directed edges ((tis_i, tp_i), (tis_j, tp_i+1)) covering
    cheapest forward paths between every observed node and its next
    observed node in each tissue's timeline.

    Tiebreaker hierarchy when multiple paths share the minimum cost:
      (a) earliest first cross-tissue edge (lower step index wins)
      (b) fewer total cross-tissue edges
      (c) lexicographic on the tissue sequence
    """
    if len(observed_set) < 2:
        return frozenset()
    obs_by_tissue = [sorted(t for (ti, t) in observed_set if ti == k)
                     for k in range(3)]
    edges = set()
    for u_ti, u_tp in observed_set:
        for v_ti in range(3):
            v_candidates = [t for t in obs_by_tissue[v_ti] if t > u_tp]
            if not v_candidates:
                continue
            v_tp = v_candidates[0]
            n = v_tp - u_tp
            # Enumerate all min-cost paths: paths_at[step][tissue] is
            # a list of tissue-tuples reaching (tissue, step) at min cost.
            costs = [[INF] * 3 for _ in range(n + 1)]
            paths_at = [[[] for _ in range(3)] for _ in range(n + 1)]
            costs[0][u_ti] = 0.0
            paths_at[0][u_ti] = [(u_ti,)]
            for step in range(n):
                for src in range(3):
                    if costs[step][src] == INF:
                        continue
                    for dst in range(3):
                        w = 1.0 if src == dst else float(CROSS_WEIGHT)
                        c = costs[step][src] + w
                        if c < costs[step + 1][dst]:
                            costs[step + 1][dst] = c
                            paths_at[step + 1][dst] = [
                                p + (dst,) for p in paths_at[step][src]
                            ]
                        elif c == costs[step + 1][dst]:
                            paths_at[step + 1][dst].extend(
                                p + (dst,) for p in paths_at[step][src]
                            )
            cands = paths_at[n][v_ti]
            if not cands:
                continue
            # Hierarchical tiebreaker.
            def _key(seq):
                first_cross = next(
                    (k for k in range(1, len(seq))
                     if seq[k] != seq[k - 1]), len(seq))
                n_crosses = sum(1 for k in range(1, len(seq))
                                if seq[k] != seq[k - 1])
                return (first_cross, n_crosses, seq)
            best_seq = min(cands, key=_key)
            for k in range(n):
                edges.add(((best_seq[k], u_tp + k),
                            (best_seq[k + 1], u_tp + k + 1)))
    return frozenset(edges)


# Calibration on 500 clones.
calib_n = min(500, len(eligible_clones))
calib_ids = eligible_clones[:calib_n]
_t0 = time.time()
for cid in calib_ids:
    g = bin_obs_groups.get_group(cid)
    nodes = {(TISSUE_IDX[r["tissue"]], TP_IDX[r["timepoint"]])
             for _, r in g.iterrows()}
    _retained_edges(nodes)
calib_dt = time.time() - _t0
eta_full = calib_dt * (len(eligible_clones) / calib_n)
print(f"  calibration: {calib_n} clones in {calib_dt:.1f}s "
      f"→ ETA full ({len(eligible_clones)} clones): {eta_full:.0f}s")

clone_graphs = {}
for cid in tqdm(eligible_clones, desc="  graphs"):
    g = bin_obs_groups.get_group(cid)
    nodes = {(TISSUE_IDX[r["tissue"]], TP_IDX[r["timepoint"]])
             for _, r in g.iterrows()}
    clone_graphs[cid] = (_retained_edges(nodes), nodes)


# %%
# =========================================================
# Hash by canonical edge list + count
# =========================================================
print("\nHashing graphs by canonical edge list...")
edges_to_clones = {}
for cid, (edges, _nodes) in clone_graphs.items():
    edges_to_clones.setdefault(edges, []).append(cid)

archetypes_sorted = sorted(edges_to_clones.items(),
                            key=lambda kv: -len(kv[1]))
n_unique = len(archetypes_sorted)
print(f"  unique archetypes (strict edge-set equality): {n_unique:,}")
print(f"  top 10 archetype clone counts: "
      f"{[len(v) for _, v in archetypes_sorted[:10]]}")

archetype_id_map = {}
for idx, (edges, cids) in enumerate(archetypes_sorted):
    for cid in cids:
        archetype_id_map[cid] = idx

clone_meta["archetype_id"] = clone_meta["clone_id"].map(archetype_id_map)


# %%
# =========================================================
# Subset diagnostic on top 20 (no merging)
# =========================================================
print("\nSubset diagnostic (top 20)...")
top_archetypes = archetypes_sorted[:TOP_N]
n_top = len(top_archetypes)
subset_mat = np.zeros((n_top, n_top), dtype=int)
for i, (edges_i, _) in enumerate(top_archetypes):
    for j, (edges_j, _) in enumerate(top_archetypes):
        if i == j:
            continue
        if edges_i <= edges_j:
            subset_mat[i, j] = 1
n_subset_rel = int(subset_mat.sum())
print(f"  {n_subset_rel} subset relationships among top {n_top}")
subset_df = pd.DataFrame(
    subset_mat,
    index=[f"A{i}" for i in range(n_top)],
    columns=[f"A{j}" for j in range(n_top)])
subset_df.to_csv(OUT_DIR / "subset_matrix.csv")


# %%
# =========================================================
# Per-archetype summary
# =========================================================
print("\nPer-archetype summaries (top 20)...")
obs_e["archetype_id"] = obs_e["clone_id"].map(archetype_id_map)

phenotypes_present = [p for p in TCELL_PHENOTYPE_ORDER
                       if p in obs_e["phenotype"].unique()]


def _edge_str(e):
    (a_ti, a_tp), (b_ti, b_tp) = e
    return (f"{TISSUES[a_ti]}@T{TPS[a_tp]}"
            f"->{TISSUES[b_ti]}@T{TPS[b_tp]}")


def _describe_archetype(edges):
    if not edges:
        return "no edges (degenerate)"
    edge_list = sorted(edges)
    tissues_seen = set()
    tps_seen = set()
    for (a_ti, a_tp), (b_ti, b_tp) in edge_list:
        tissues_seen.add(a_ti); tissues_seen.add(b_ti)
        tps_seen.add(a_tp); tps_seen.add(b_tp)
    src_only = [e for e in edge_list
                if e[0][0] != e[1][0]]
    if not src_only:
        # All within-tissue: pure residence.
        tis = TISSUES[next(iter(tissues_seen))]
        return (f"{tis}-resident T{TPS[min(tps_seen)]}-T{TPS[max(tps_seen)]}")
    # Cross-tissue edges present — describe migration as src→dst summary.
    crosses = [(TISSUES[a_ti], TISSUES[b_ti])
               for (a_ti, _), (b_ti, _) in src_only]
    uniq = []
    for c in crosses:
        if c not in uniq:
            uniq.append(c)
    transitions = ", ".join(f"{a}→{b}" for a, b in uniq)
    return f"{transitions}; spans T{TPS[min(tps_seen)]}-T{TPS[max(tps_seen)]}"


summary_rows = []
graph_rows = []
for idx, (edges, cids) in enumerate(archetypes_sorted):
    sub_c = clone_meta[clone_meta["archetype_id"] == idx]
    sub_o = obs_e[obs_e["archetype_id"] == idx]
    edge_list = sorted(edges)
    edge_serialized = [
        [[int(a_ti), int(a_tp)], [int(b_ti), int(b_tp)]]
        for (a_ti, a_tp), (b_ti, b_tp) in edge_list
    ]
    tissues_seen = set()
    tps_seen = set()
    for (a_ti, a_tp), (b_ti, b_tp) in edge_list:
        tissues_seen.add(a_ti); tissues_seen.add(b_ti)
        tps_seen.add(a_tp); tps_seen.add(b_tp)
    graph_rows.append({
        "archetype_id": idx,
        "n_clones": len(cids),
        "edge_list_json": json.dumps(edge_serialized),
        "n_unique_tissues": int(len(tissues_seen)),
        "n_timepoints_spanned": int(len(tps_seen)),
        "n_edges": int(len(edge_list)),
    })
    if idx >= TOP_N:
        continue
    top_phen = (sub_o["phenotype"].value_counts().head(1).index[0]
                if not sub_o.empty else "")
    summary_rows.append({
        "archetype_id": idx,
        "n_clones": int(len(sub_c)),
        "clone_size_mean": float(sub_c["clone_size"].mean()),
        "clone_size_std": float(sub_c["clone_size"].std()),
        "fraction_CD8": float((sub_c["lineage"] == "CD8").mean()),
        "fraction_CD4": float((sub_c["lineage"] == "CD4").mean()),
        "top_phenotype": top_phen,
        "n_unique_patients": int(sub_c["patient"].nunique()),
        "patients_top": ",".join(
            sub_c["patient"].value_counts().head(3).index.tolist()),
        "n_edges": int(len(edge_list)),
        "n_unique_tissues": int(len(tissues_seen)),
        "n_timepoints_spanned": int(len(tps_seen)),
        "description": _describe_archetype(edges),
    })

pd.DataFrame(graph_rows).to_csv(
    OUT_DIR / "archetype_graphs.csv", index=False)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "archetype_summary.csv", index=False)
print(summary_df.to_string(index=False))

# Per-clone output table.
clone_meta["retained_graph_edges"] = clone_meta["clone_id"].map(
    lambda c: " ".join(_edge_str(e)
                       for e in sorted(clone_graphs[c][0])))


# %%
# =========================================================
# Path-subset merging
# Each archetype whose root-to-leaf path-set is a strict subset of
# another archetype's path-set is absorbed into its maximal superset
# (chosen by clone count on ties). Stricter than edge-subset: a graph
# whose paths extend a shorter graph's leaves does NOT contain it.
# =========================================================
print("\nPath-subset merging...")
edge_lists = [edges for edges, _ in archetypes_sorted]
clone_lists = [cids for _, cids in archetypes_sorted]
n_arch = len(edge_lists)


def _root_to_leaf_paths(edges):
    """Return frozenset of root-to-leaf paths (tuples of nodes)
    enumerated by DFS over the DAG defined by edges."""
    if not edges:
        return frozenset()
    out_adj = {}
    nodes = set()
    has_in = set()
    for src, dst in edges:
        out_adj.setdefault(src, []).append(dst)
        nodes.add(src); nodes.add(dst)
        has_in.add(dst)
    roots = sorted(nodes - has_in)
    paths = set()
    stack = [(r, (r,)) for r in roots]
    while stack:
        node, path = stack.pop()
        succs = out_adj.get(node, ())
        if not succs:
            paths.add(path)
            continue
        for nxt in succs:
            stack.append((nxt, path + (nxt,)))
    return frozenset(paths)


path_sets = [_root_to_leaf_paths(e) for e in edge_lists]


def _tissues_in(edges):
    s = set()
    for (a_ti, _), (b_ti, _) in edges:
        s.add(a_ti); s.add(b_ti)
    return frozenset(s)


tissues_per = [_tissues_in(e) for e in edge_lists]

# Strict-subset relation under two constraints:
#   (1) paths(i) ⊊ paths(j)             (existing path-subset rule)
#   (2) tissues(i) == tissues(j)        (NEW: identical tissue scope)
# (2) prevents a PBMC↔TP archetype from being absorbed into an
# all-tissue archetype even if its paths happen to be sub-paths there.
subset_of = [set() for _ in range(n_arch)]
for i in range(n_arch):
    p_i = path_sets[i]
    if not p_i:
        continue
    t_i = tissues_per[i]
    for j in range(n_arch):
        if i == j:
            continue
        if tissues_per[j] != t_i:
            continue
        if p_i < path_sets[j]:
            subset_of[i].add(j)

# Each archetype's maximal supersets = its supersets not contained in
# any other of its supersets.
merged_assignment = [None] * n_arch
for i in range(n_arch):
    sups = subset_of[i]
    if not sups:
        merged_assignment[i] = i
        continue
    maximals = {j for j in sups if subset_of[j].isdisjoint(sups)}
    # Fallback (shouldn't happen for finite poset): everything in sups
    # is also subset of something else in sups → pick highest-clone.
    if not maximals:
        maximals = sups
    merged_assignment[i] = min(
        maximals, key=lambda j: (-len(clone_lists[j]), j))

# Aggregate clone counts per merged class, then renumber by total clones.
merged_counts = {}
merged_sources = {}
for i, m in enumerate(merged_assignment):
    merged_counts[m] = merged_counts.get(m, 0) + len(clone_lists[i])
    merged_sources.setdefault(m, []).append(i)
order_old = sorted(merged_counts.keys(),
                    key=lambda m: (-merged_counts[m], m))
merged_renumber = {old: new for new, old in enumerate(order_old)}

# Per-clone mapping: clone → strict archetype → merged archetype.
clone_meta["merged_archetype_id"] = clone_meta["archetype_id"].map(
    lambda a: merged_renumber[merged_assignment[int(a)]])

# Rewrite clone_archetypes.csv with merged column.
clone_meta[["trb", "patient", "clone_id", "archetype_id",
            "merged_archetype_id", "clone_size", "lineage",
            "n_observed_nodes", "retained_graph_edges"]].to_csv(
    OUT_DIR / "clone_archetypes.csv", index=False)

# merge_map.csv: per strict archetype, its merged destination + count.
merge_map_rows = []
for i in range(n_arch):
    merge_map_rows.append({
        "archetype_id": i,
        "merged_archetype_id": merged_renumber[merged_assignment[i]],
        "n_clones_in_this_strict_archetype": len(clone_lists[i]),
    })
pd.DataFrame(merge_map_rows).to_csv(
    OUT_DIR / "merge_map.csv", index=False)

# merged_archetypes.csv.
merged_arch_rows = []
for new_id, old_id in enumerate(order_old):
    rep_edges = edge_lists[old_id]
    rep_edge_list = sorted(rep_edges)
    tissues_seen = set()
    tps_seen = set()
    for (a_ti, a_tp), (b_ti, b_tp) in rep_edge_list:
        tissues_seen.add(a_ti); tissues_seen.add(b_ti)
        tps_seen.add(a_tp); tps_seen.add(b_tp)
    merged_arch_rows.append({
        "merged_archetype_id": new_id,
        "n_clones": merged_counts[old_id],
        "edge_list_json": json.dumps([
            [[int(a_ti), int(a_tp)], [int(b_ti), int(b_tp)]]
            for (a_ti, a_tp), (b_ti, b_tp) in rep_edge_list]),
        "n_unique_tissues": int(len(tissues_seen)),
        "n_timepoints_spanned": int(len(tps_seen)),
        "n_edges": int(len(rep_edge_list)),
        "source_archetype_ids": ",".join(
            str(s) for s in sorted(merged_sources[old_id])),
        "n_source_archetypes": len(merged_sources[old_id]),
    })
merged_arch_df = pd.DataFrame(merged_arch_rows)
merged_arch_df.to_csv(
    OUT_DIR / "merged_archetypes.csv", index=False)

# merged_archetype_summary.csv: per merged archetype, top-level stats.
obs_e["merged_archetype_id"] = obs_e["clone_id"].map(
    dict(zip(clone_meta["clone_id"],
              clone_meta["merged_archetype_id"])))
merged_summary_rows = []
for new_id, old_id in enumerate(order_old):
    if new_id >= TOP_N:
        break
    sub_c = clone_meta[clone_meta["merged_archetype_id"] == new_id]
    sub_o = obs_e[obs_e["merged_archetype_id"] == new_id]
    top_phen = (sub_o["phenotype"].value_counts().head(1).index[0]
                if not sub_o.empty else "")
    desc = _describe_archetype(edge_lists[old_id])
    merged_summary_rows.append({
        "merged_archetype_id": new_id,
        "n_clones": int(len(sub_c)),
        "clone_size_mean": float(sub_c["clone_size"].mean()),
        "clone_size_std": float(sub_c["clone_size"].std()),
        "fraction_CD8": float((sub_c["lineage"] == "CD8").mean()),
        "fraction_CD4": float((sub_c["lineage"] == "CD4").mean()),
        "top_phenotype": top_phen,
        "n_unique_patients": int(sub_c["patient"].nunique()),
        "patients_top": ",".join(
            sub_c["patient"].value_counts().head(3).index.tolist()),
        "n_source_archetypes": len(merged_sources[old_id]),
        "description": desc,
    })
merged_summary_df = pd.DataFrame(merged_summary_rows)
merged_summary_df.to_csv(
    OUT_DIR / "merged_archetype_summary.csv", index=False)

n_merged = len(order_old)
print(f"  strict archetypes: {n_arch:,} → merged archetypes: "
      f"{n_merged:,}")
big_tents = [(new_id, old_id, len(merged_sources[old_id]),
              merged_counts[old_id])
             for new_id, old_id in enumerate(order_old)
             if len(merged_sources[old_id]) > 5]
print(f"  merged archetypes absorbing >5 strict subsets: "
      f"{len(big_tents)}")
for new_id, old_id, n_src, n_cl in big_tents[:10]:
    print(f"    M{new_id}: absorbs {n_src} strict, "
          f"{n_cl} clones — {_describe_archetype(edge_lists[old_id])}")

# Pre-build lookups for downstream plots.
merged_rep_edges = {new_id: edge_lists[old_id]
                     for new_id, old_id in enumerate(order_old)}
merged_rep_old_id = {new_id: old_id
                      for new_id, old_id in enumerate(order_old)}


# %%
# =========================================================
# Plot 1: top-N archetype retained graphs (TOP_N = 20)
# =========================================================
print(f"\nPlotting top-{TOP_N} archetype graphs...")
top_archs = archetypes_sorted[:TOP_N]
n_cols = 5
n_rows = int(np.ceil(len(top_archs) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 4.0, n_rows * 2.8),
                         squeeze=False)
for idx, (edges, cids) in enumerate(top_archs):
    ax = axes[idx // n_cols, idx % n_cols]
    n_clones = len(cids)
    desc = _describe_archetype(edges)
    nodes_in_graph = set()
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        nodes_in_graph.add((a_ti, a_tp))
        nodes_in_graph.add((b_ti, b_tp))
    obs_freq = {}
    for cid in cids:
        for n_ in clone_graphs[cid][1]:
            obs_freq[n_] = obs_freq.get(n_, 0) + 1
    observed_majority = {n_ for n_, c in obs_freq.items()
                         if c >= 0.5 * n_clones}
    for j in range(T):
        for k in range(3):
            ax.plot(j, 2 - k, "o", color="#dddddd",
                    markersize=5, zorder=1)
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        arr = FancyArrowPatch(
            posA=(a_tp, 2 - a_ti),
            posB=(b_tp, 2 - b_ti),
            arrowstyle="-|>", mutation_scale=12,
            color="#555", lw=1.1,
            shrinkA=9, shrinkB=9,
            connectionstyle="arc3,rad=0.0", zorder=2,
        )
        ax.add_patch(arr)
    for (ti, tp) in nodes_in_graph:
        x, y = tp, 2 - ti
        tname = TISSUES[ti]
        color = TISSUE_COLORS[tname]
        if (ti, tp) in observed_majority:
            ax.plot(x, y, "o", markerfacecolor=color,
                    markeredgecolor="black", markersize=12, zorder=3)
        else:
            ax.plot(x, y, "o", markerfacecolor="white",
                    markeredgecolor=color, markersize=12, mew=1.8,
                    zorder=3)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"T{t}" for t in TPS], fontsize=7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([TISSUES[2], TISSUES[1], TISSUES[0]],
                        fontsize=7)
    for tick, tname in zip(ax.get_yticklabels(),
                            [TISSUES[2], TISSUES[1], TISSUES[0]]):
        tick.set_color(TISSUE_COLORS.get(tname, "#444"))
        tick.set_fontweight("bold")
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title(f"A{idx}: n={n_clones}\n{desc}",
                  fontsize=8, linespacing=1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_aspect("auto")
# Hide unused panels if any.
for k in range(len(top_archs), n_rows * n_cols):
    axes[k // n_cols, k % n_cols].axis("off")
fig.suptitle(f"Top-{TOP_N} retained-path archetypes "
             "(filled = majority-observed; hollow = on-path inferred)",
             fontsize=11, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_top_graphs.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 1b: top-N MULTI-TISSUE archetype retained graphs
# (filter to archetypes touching ≥2 tissues; preserve original IDs)
# =========================================================
print(f"\nPlotting top-{TOP_N} multi-tissue archetype graphs...")


def _n_unique_tissues(edges):
    s = set()
    for (a_ti, _), (b_ti, _) in edges:
        s.add(a_ti); s.add(b_ti)
    return len(s)


multi_archs = [(i, edges, cids)
               for i, (edges, cids) in enumerate(archetypes_sorted)
               if _n_unique_tissues(edges) >= 2]
top_multi = multi_archs[:TOP_N]
print(f"  multi-tissue archetypes total: {len(multi_archs):,}; "
      f"plotting top {len(top_multi)}")

n_cols_m = 5
n_rows_m = int(np.ceil(len(top_multi) / n_cols_m))
fig, axes = plt.subplots(n_rows_m, n_cols_m,
                         figsize=(n_cols_m * 4.0, n_rows_m * 2.8),
                         squeeze=False)
for slot, (orig_id, edges, cids) in enumerate(top_multi):
    ax = axes[slot // n_cols_m, slot % n_cols_m]
    n_clones = len(cids)
    desc = _describe_archetype(edges)
    nodes_in_graph = set()
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        nodes_in_graph.add((a_ti, a_tp))
        nodes_in_graph.add((b_ti, b_tp))
    obs_freq = {}
    for cid in cids:
        for n_ in clone_graphs[cid][1]:
            obs_freq[n_] = obs_freq.get(n_, 0) + 1
    observed_majority = {n_ for n_, c in obs_freq.items()
                         if c >= 0.5 * n_clones}
    for j in range(T):
        for k in range(3):
            ax.plot(j, 2 - k, "o", color="#dddddd",
                    markersize=5, zorder=1)
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        arr = FancyArrowPatch(
            posA=(a_tp, 2 - a_ti),
            posB=(b_tp, 2 - b_ti),
            arrowstyle="-|>", mutation_scale=12,
            color="#555", lw=1.1,
            shrinkA=9, shrinkB=9,
            connectionstyle="arc3,rad=0.0", zorder=2,
        )
        ax.add_patch(arr)
    for (ti, tp) in nodes_in_graph:
        x, y = tp, 2 - ti
        tname = TISSUES[ti]
        color = TISSUE_COLORS[tname]
        if (ti, tp) in observed_majority:
            ax.plot(x, y, "o", markerfacecolor=color,
                    markeredgecolor="black", markersize=12, zorder=3)
        else:
            ax.plot(x, y, "o", markerfacecolor="white",
                    markeredgecolor=color, markersize=12, mew=1.8,
                    zorder=3)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"T{t}" for t in TPS], fontsize=7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([TISSUES[2], TISSUES[1], TISSUES[0]],
                        fontsize=7)
    for tick, tname in zip(ax.get_yticklabels(),
                            [TISSUES[2], TISSUES[1], TISSUES[0]]):
        tick.set_color(TISSUE_COLORS.get(tname, "#444"))
        tick.set_fontweight("bold")
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title(f"A{orig_id}: n={n_clones}\n{desc}",
                  fontsize=8, linespacing=1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_aspect("auto")
for k in range(len(top_multi), n_rows_m * n_cols_m):
    axes[k // n_cols_m, k % n_cols_m].axis("off")
fig.suptitle(f"Top-{TOP_N} multi-tissue archetypes "
             f"(≥2 tissues; original archetype IDs preserved)",
             fontsize=11, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_top_graphs_multi_tissue.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 2: top-20 size distribution
# =========================================================
print("Plotting top-20 size distribution...")
top20 = archetypes_sorted[:TOP_N]
fig, ax = plt.subplots(figsize=(11, 4))
xs = np.arange(len(top20))
counts = [len(cids) for _, cids in top20]
ax.bar(xs, counts, color="#5e8ec1",
       edgecolor="black", linewidth=0.4)
ax.set_xticks(xs)
ax.set_xticklabels([f"A{i}" for i in range(len(top20))], fontsize=8)
ax.set_ylabel("# clones", fontsize=9)
ax.set_xlabel("Archetype rank", fontsize=9)
ax.set_title(f"Top-{TOP_N} archetypes by clone count "
             f"(of {n_unique:,} unique total)",
             fontsize=10, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
# Annotate top-5 with short description.
for i in range(min(5, len(top20))):
    ax.text(i, counts[i] * 1.02,
            _describe_archetype(top20[i][0])[:30],
            ha="center", va="bottom", fontsize=6,
            rotation=20)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_size_distribution.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 3: subset diagnostic heatmap
# =========================================================
print("Plotting subset diagnostic...")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(subset_mat, cmap="Blues", aspect="equal",
                vmin=0, vmax=1)
ax.set_xticks(range(n_top))
ax.set_xticklabels([f"A{j}" for j in range(n_top)], fontsize=7,
                    rotation=45, ha="right")
ax.set_yticks(range(n_top))
ax.set_yticklabels([f"A{i}" for i in range(n_top)], fontsize=7)
ax.set_xlabel("contains (superset)", fontsize=9)
ax.set_ylabel("contained in (subset)", fontsize=9)
ax.set_title(f"Edge-set subset relationships, top {n_top}\n"
             f"M[i,j]=1 iff edges(A_i) ⊆ edges(A_j); "
             f"{n_subset_rel} relationships",
             fontsize=10, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "subset_diagnostic.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 4: per top-5 archetype, phenotype heatmap per (tissue × timepoint)
# =========================================================
print("Plotting archetype phenotype heatmaps...")
fig, axes = plt.subplots(5, 3, figsize=(11, 12),
                         squeeze=False, sharey=True)
for a_idx in range(5):
    sub_a = obs_e[obs_e["archetype_id"] == a_idx]
    for ti, tis in enumerate(TISSUES):
        ax = axes[a_idx, ti]
        mat = np.zeros((len(phenotypes_present), T), dtype=float)
        for tj, tp in enumerate(TPS):
            sub = sub_a[(sub_a["tissue"] == tis)
                          & (sub_a["timepoint"] == tp)]
            if sub.empty:
                continue
            cnt = sub["phenotype"].value_counts()
            tot = cnt.sum()
            for pi, p in enumerate(phenotypes_present):
                mat[pi, tj] = cnt.get(p, 0) / max(tot, 1)
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"T{t}" for t in TPS], fontsize=6)
        if ti == 0:
            ax.set_yticks(range(len(phenotypes_present)))
            ax.set_yticklabels(phenotypes_present, fontsize=6)
            for tick, p in zip(ax.get_yticklabels(), phenotypes_present):
                tick.set_color(TCELL_PHENOTYPE_COLORS.get(p, "#444"))
        else:
            ax.set_yticks([])
        if a_idx == 0:
            ax.set_title(tis, fontsize=9, fontweight="bold",
                          color=TISSUE_COLORS.get(tis, "#444"))
        if ti == 0:
            ax.set_ylabel(
                f"A{a_idx}\n(n={len(archetypes_sorted[a_idx][1])})",
                          fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=22)
        ax.tick_params(length=0)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
fig.suptitle("Phenotype composition per (archetype × tissue × timepoint)",
             fontsize=11, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_phenotypes.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 5: top-5 MERGED archetype retained graphs
# =========================================================
print("\nPlotting top-5 merged archetype graphs...")
top5_merged_new = list(range(min(5, n_merged)))
fig, axes = plt.subplots(1, len(top5_merged_new),
                         figsize=(len(top5_merged_new) * 4.0, 4.0),
                         squeeze=False)
for col, new_id in enumerate(top5_merged_new):
    ax = axes[0, col]
    edges = merged_rep_edges[new_id]
    old_id = merged_rep_old_id[new_id]
    n_clones = merged_counts[old_id]
    n_src = len(merged_sources[old_id])
    desc = _describe_archetype(edges)
    nodes_in_graph = set()
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        nodes_in_graph.add((a_ti, a_tp))
        nodes_in_graph.add((b_ti, b_tp))
    # For merged: "observed in majority" = union of observed nodes
    # across all clones in any of the absorbed strict archetypes.
    obs_freq = {}
    for strict_id in merged_sources[old_id]:
        for cid in clone_lists[strict_id]:
            for n_ in clone_graphs[cid][1]:
                obs_freq[n_] = obs_freq.get(n_, 0) + 1
    observed_majority = {n_ for n_, c in obs_freq.items()
                         if c >= 0.5 * n_clones}
    for j in range(T):
        for k in range(3):
            ax.plot(j, 2 - k, "o", color="#dddddd",
                    markersize=6, zorder=1)
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        arr = FancyArrowPatch(
            posA=(a_tp, 2 - a_ti), posB=(b_tp, 2 - b_ti),
            arrowstyle="-|>", mutation_scale=14,
            color="#555", lw=1.2,
            shrinkA=10, shrinkB=10,
            connectionstyle="arc3,rad=0.0", zorder=2,
        )
        ax.add_patch(arr)
    for (ti, tp) in nodes_in_graph:
        x, y = tp, 2 - ti
        color = TISSUE_COLORS[TISSUES[ti]]
        if (ti, tp) in observed_majority:
            ax.plot(x, y, "o", markerfacecolor=color,
                    markeredgecolor="black", markersize=14, zorder=3)
        else:
            ax.plot(x, y, "o", markerfacecolor="white",
                    markeredgecolor=color, markersize=14, mew=2,
                    zorder=3)
    ax.set_xticks(range(T))
    ax.set_xticklabels([f"T{t}" for t in TPS], fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([TISSUES[2], TISSUES[1], TISSUES[0]],
                        fontsize=8)
    for tick, tname in zip(ax.get_yticklabels(),
                            [TISSUES[2], TISSUES[1], TISSUES[0]]):
        tick.set_color(TISSUE_COLORS.get(tname, "#444"))
        tick.set_fontweight("bold")
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_title(f"M{new_id}: n={n_clones} (absorbed {n_src})\n{desc}",
                  fontsize=9, linespacing=1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("Top-5 merged-archetype retained graphs",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "merged_top_graphs.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 6: merged archetype size distribution
# =========================================================
print("Plotting merged size distribution...")
top_merged_for_bar = list(range(min(TOP_N, n_merged)))
fig, ax = plt.subplots(figsize=(11, 4))
xs = np.arange(len(top_merged_for_bar))
counts = [merged_counts[merged_rep_old_id[m]]
          for m in top_merged_for_bar]
n_src_list = [len(merged_sources[merged_rep_old_id[m]])
              for m in top_merged_for_bar]
ax.bar(xs, counts, color="#7d3c98",
       edgecolor="black", linewidth=0.4)
ax.set_xticks(xs)
ax.set_xticklabels([f"M{m}" for m in top_merged_for_bar], fontsize=8)
ax.set_ylabel("# clones (incl. absorbed strict)", fontsize=9)
ax.set_xlabel("Merged archetype rank", fontsize=9)
ax.set_title(f"Top-{TOP_N} merged archetypes "
             f"({n_merged:,} merged, from {n_arch:,} strict)",
             fontsize=10, fontweight="bold")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for i in range(min(5, len(top_merged_for_bar))):
    new_id = top_merged_for_bar[i]
    ax.text(i, counts[i] * 1.02,
             f"+{n_src_list[i]} src\n"
             f"{_describe_archetype(merged_rep_edges[new_id])[:24]}",
             ha="center", va="bottom", fontsize=6, rotation=15)
fig.tight_layout()
fig.savefig(OUT_DIR / "merged_size_distribution.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 7: merge flow (top-20 strict → top-N merged, line widths
# proportional to clone count). Bipartite layout.
# =========================================================
print("Plotting merge flow...")
top_strict_for_flow = list(range(min(TOP_N, n_arch)))
merged_targets = sorted({
    merged_renumber[merged_assignment[i]]
    for i in top_strict_for_flow
})
fig, ax = plt.subplots(figsize=(10, 7))
y_left = {sid: -i for i, sid in enumerate(top_strict_for_flow)}
y_right = {mid: -i for i, mid in enumerate(merged_targets)}
left_counts = {sid: len(clone_lists[sid]) for sid in top_strict_for_flow}
right_counts = {mid: merged_counts[merged_rep_old_id[mid]]
                for mid in merged_targets}
max_cnt = max(max(left_counts.values()),
              max(right_counts.values())) or 1
# Draw left bars.
for sid in top_strict_for_flow:
    w = left_counts[sid] / max_cnt * 0.8
    ax.add_patch(plt.Rectangle((0, y_left[sid] - 0.4),
                                 w, 0.8, color="#5e8ec1",
                                 edgecolor="black", linewidth=0.3))
    ax.text(-0.05, y_left[sid], f"A{sid} (n={left_counts[sid]})",
            ha="right", va="center", fontsize=7)
# Right bars.
for mid in merged_targets:
    w = right_counts[mid] / max_cnt * 0.8
    ax.add_patch(plt.Rectangle((3 - w, y_right[mid] - 0.4),
                                 w, 0.8, color="#7d3c98",
                                 edgecolor="black", linewidth=0.3))
    ax.text(3.05, y_right[mid], f"M{mid} (n={right_counts[mid]})",
            ha="left", va="center", fontsize=7)
# Flow ribbons.
for sid in top_strict_for_flow:
    mid = merged_renumber[merged_assignment[sid]]
    self_merge = (mid == merged_renumber[sid]
                  if sid in merged_renumber else False)
    color = "#aaaaaa" if self_merge else "#cc6666"
    lw = max(0.5, left_counts[sid] / max_cnt * 6.0)
    x0 = left_counts[sid] / max_cnt * 0.8
    x1 = 3 - right_counts[mid] / max_cnt * 0.8
    ax.plot([x0, x1], [y_left[sid], y_right[mid]],
             color=color, lw=lw, alpha=0.55, zorder=1)
ax.set_xlim(-1.3, 4.0)
ax.set_ylim(min(min(y_left.values(), default=0),
                min(y_right.values(), default=0)) - 1, 1)
ax.set_xticks([0.4, 2.6])
ax.set_xticklabels(["strict (top-20)", "merged"],
                    fontsize=10, fontweight="bold")
ax.set_yticks([])
for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)
ax.set_title("Merge flow: top-20 strict archetypes → "
             "merged destinations\n"
             "(gray = stays its own merged class; red = absorbed)",
             fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "merge_flow.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Final summary
# =========================================================
print("\n========== SUMMARY ==========")
print(f"Eligible clones: {len(eligible_clones):,}")
print(f"Unique archetypes (strict edge-set): {n_unique:,}")
n_top5 = sum(len(cids) for _, cids in archetypes_sorted[:5])
print(f"Top-5 cover {n_top5:,} clones "
      f"({n_top5/len(eligible_clones)*100:.1f}% of eligible)")
print(f"Subset relationships among top {TOP_N}: {n_subset_rel}")
print()
for _, r in summary_df.head(5).iterrows():
    print(f"A{int(r['archetype_id'])}: n={int(r['n_clones']):5d}  "
          f"sizē={r['clone_size_mean']:6.1f}  "
          f"{r['description']}")

print(f"\n----- merging -----")
print(f"Strict → merged: {n_arch:,} → {n_merged:,}")
print("Top 5 merged archetypes:")
for _, r in merged_summary_df.head(5).iterrows():
    print(f"M{int(r['merged_archetype_id'])}: "
          f"n={int(r['n_clones']):5d}  "
          f"absorbed {int(r['n_source_archetypes'])}  "
          f"{r['description']}")
if big_tents:
    print("\nBig-tent merged archetypes (absorbed >5 strict):")
    for new_id, old_id, n_src, n_cl in big_tents[:10]:
        print(f"  M{new_id}: {n_src} strict → {n_cl} clones — "
              f"{_describe_archetype(edge_lists[old_id])}")

print(f"\nDone. All outputs in {OUT_DIR}")
