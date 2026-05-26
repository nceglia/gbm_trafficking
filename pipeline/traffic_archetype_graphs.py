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
    observed node in each tissue's timeline."""
    if len(observed_set) < 2:
        return frozenset()
    # Index next observed timepoint per tissue.
    next_in_tissue = [{} for _ in range(3)]
    obs_by_tissue = [sorted(t for (ti, t) in observed_set if ti == k)
                     for k in range(3)]
    edges = set()
    for u_ti, u_tp in observed_set:
        for v_ti in range(3):
            v_candidates = [t for t in obs_by_tissue[v_ti] if t > u_tp]
            if not v_candidates:
                continue
            v_tp = v_candidates[0]
            # Shortest path on the (3, [u_tp..v_tp]) grid.
            n = v_tp - u_tp  # number of edges in the path
            dp = [[INF] * 3 for _ in range(n + 1)]
            parent = [[None] * 3 for _ in range(n + 1)]
            dp[0][u_ti] = 0.0
            for step in range(n):
                for src in range(3):
                    if dp[step][src] == INF:
                        continue
                    for dst in range(3):
                        w = 1.0 if src == dst else float(CROSS_WEIGHT)
                        cost = dp[step][src] + w
                        # Tie-break: prefer staying in same tissue, then
                        # lower tissue index (deterministic canonical).
                        existing = dp[step + 1][dst]
                        if cost < existing or (
                                cost == existing
                                and parent[step + 1][dst] is not None
                                and src < parent[step + 1][dst]):
                            dp[step + 1][dst] = cost
                            parent[step + 1][dst] = src
            # Backtrack from (v_ti, v_tp).
            cur = v_ti
            for step in range(n, 0, -1):
                src = parent[step][cur]
                if src is None:
                    break
                edges.add(((src, u_tp + step - 1),
                            (cur, u_tp + step)))
                cur = src
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
clone_meta[["trb", "patient", "clone_id", "archetype_id",
            "clone_size", "lineage", "n_observed_nodes",
            "retained_graph_edges"]].to_csv(
    OUT_DIR / "clone_archetypes.csv", index=False)


# %%
# =========================================================
# Plot 1: top-5 archetype retained graphs
# =========================================================
print("\nPlotting top-5 archetype graphs...")
top5 = archetypes_sorted[:5]
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), squeeze=False)
for col, (edges, cids) in enumerate(top5):
    ax = axes[0, col]
    n_clones = len(cids)
    desc = _describe_archetype(edges)
    # Layout: x = timepoint index, y = tissue index (PBMC top → TP bottom).
    nodes_in_graph = set()
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        nodes_in_graph.add((a_ti, a_tp))
        nodes_in_graph.add((b_ti, b_tp))
    # Mark which nodes are observed in at least 60% of this archetype's
    # clones (since strict equality already requires identical edge
    # sets, observed nodes per clone are essentially the same).
    obs_freq = {}
    for cid in cids:
        for n_ in clone_graphs[cid][1]:
            obs_freq[n_] = obs_freq.get(n_, 0) + 1
    observed_majority = {n_ for n_, c in obs_freq.items()
                         if c >= 0.5 * n_clones}
    # Draw grid lattice (faint).
    for j in range(T):
        for k in range(3):
            ax.plot(j, 2 - k, "o", color="#dddddd",
                    markersize=6, zorder=1)
    # Draw edges.
    for (a_ti, a_tp), (b_ti, b_tp) in edges:
        arr = FancyArrowPatch(
            posA=(a_tp, 2 - a_ti),
            posB=(b_tp, 2 - b_ti),
            arrowstyle="-|>", mutation_scale=14,
            color="#555", lw=1.2,
            shrinkA=10, shrinkB=10,
            connectionstyle="arc3,rad=0.0", zorder=2,
        )
        ax.add_patch(arr)
    # Draw nodes — observed filled, on-path unobserved hollow.
    for (ti, tp) in nodes_in_graph:
        x, y = tp, 2 - ti
        tname = TISSUES[ti]
        color = TISSUE_COLORS[tname]
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
    ax.set_title(f"A{col}: n={n_clones}\n{desc}",
                  fontsize=9, linespacing=1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_aspect("auto")
fig.suptitle("Top-5 retained-path archetypes (filled = majority-observed; "
             "hollow = on-path inferred)",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_top_graphs.png",
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
            ax.set_ylabel(f"A{a_idx}\n(n={len(top5[a_idx][1])})",
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

print(f"\nDone. All outputs in {OUT_DIR}")
