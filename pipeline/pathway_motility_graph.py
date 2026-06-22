# %%
"""Per-clone T-cell-migration delta across the clonal-traffic graph.

The traffic graph has one node per (tissue, timepoint) — 3 tissues × 6
timepoints = 18 nodes max. We draw all *forward* edges (timepoint t →
t+1), 9 candidate edges per timepoint step (3 src tissues × 3 dst
tissues), and color each edge by the **per-clone change** in a T-cell-
specific migration score between the source and the target node.

For every edge:
  - Find clones with ≥ MIN_CELLS_PER_NODE cells at the source node
    AND ≥ MIN_CELLS_PER_NODE cells at the target node (paired observation).
  - Compute Δ = (mean score in target) − (mean score in source) per clone.
  - Mean Δ across clones is the edge's color (red = motility up, blue = down).
  - Wilcoxon signed-rank on Δ across clones gives a p-value → asterisks.

Migration score: union of three GO BP gene sets that are *specifically*
T-cell motility (not general leukocyte adhesion):
    T Cell Migration (GO:0072678)
    T Cell Chemotaxis (GO:0010818)
    Positive Regulation Of T Cell Migration (GO:2000406)

Usage:
    python pipeline/pathway_motility_graph.py
"""
import sys
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.clone_helpers import infer_lineage_from_phenotype  # noqa: E402
from modules.style import TISSUE_COLORS, TISSUE_LABELS  # noqa: E402

# %% Config
OUT_DIR = REPO_ROOT / "results" / "pathway_motility_traffic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS_PER_NODE = 3        # minimum cells per (clone, tissue, timepoint)
MIN_CLONES_PER_EDGE = 3       # minimum paired clones to consider an edge
TISSUES = ["PBMC", "CSF", "TP"]
TIMEPOINTS = [1, 2, 3, 4, 5, 6]
TISSUE_Y = {"PBMC": 2, "CSF": 1, "TP": 0}   # y positions
RANDOM_STATE = 42

GO_TERMS = [
    "T Cell Migration (GO:0072678)",
    "T Cell Chemotaxis (GO:0010818)",
    "Positive Regulation Of T Cell Migration (GO:2000406)",
]


# %% Load adata, build migration gene panel
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

gobp = gp.get_library(name="GO_Biological_Process_2023")
mig_genes = set()
for term in GO_TERMS:
    if term in gobp:
        mig_genes.update(gobp[term])
mig_genes = sorted(mig_genes & set(adata.var_names))
print(f"T-cell migration gene panel: {len(mig_genes)} genes")
print(f"  ({sorted(mig_genes)})")

# Score with log1p
orig_X = adata.X
adata.X = adata.layers["log1p"]
sc.tl.score_genes(adata, gene_list=mig_genes,
                  score_name="tcell_mig_score", random_state=RANDOM_STATE)
adata.X = orig_X


# %% Per (patient, clone, tissue, timepoint) mean score
obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb",
                  "tcell_mig_score"]].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)

clone_node = (
    obs.groupby(["patient", "trb", "lineage", "tissue", "timepoint"], observed=True)
       .agg(mean_score=("tcell_mig_score", "mean"),
            n_cells=("tcell_mig_score", "size"))
       .reset_index()
)
clone_node = clone_node[clone_node["n_cells"] >= MIN_CELLS_PER_NODE]
clone_node.to_csv(OUT_DIR / "clone_node_tcell_mig_score.csv", index=False)
print(f"Per (clone, node) pseudobulks: {len(clone_node)}")


# %% For each forward edge, compute per-clone Δ
edges_rows = []
for lineage in ["CD8", "CD4"]:
    cn = clone_node[clone_node["lineage"] == lineage]
    for t in TIMEPOINTS[:-1]:
        tn = t + 1
        src_pool = cn[cn["timepoint"] == t]
        dst_pool = cn[cn["timepoint"] == tn]
        for src_tissue in TISSUES:
            src = src_pool[src_pool["tissue"] == src_tissue][
                ["patient", "trb", "mean_score", "n_cells"]
            ].rename(columns={"mean_score": "score_src", "n_cells": "n_src"})
            for dst_tissue in TISSUES:
                dst = dst_pool[dst_pool["tissue"] == dst_tissue][
                    ["patient", "trb", "mean_score", "n_cells"]
                ].rename(columns={"mean_score": "score_dst", "n_cells": "n_dst"})
                paired = src.merge(dst, on=["patient", "trb"], how="inner")
                if len(paired) < MIN_CLONES_PER_EDGE:
                    continue
                delta = (paired["score_dst"] - paired["score_src"]).values
                try:
                    wstat, wp = stats.wilcoxon(delta)
                except ValueError:
                    wstat, wp = np.nan, np.nan
                edges_rows.append({
                    "lineage": lineage,
                    "src_tissue": src_tissue, "t_src": t,
                    "dst_tissue": dst_tissue, "t_dst": tn,
                    "n_clones": int(len(paired)),
                    "mean_delta": float(np.mean(delta)),
                    "median_delta": float(np.median(delta)),
                    "wilcoxon_stat": float(wstat) if wstat is not None else np.nan,
                    "wilcoxon_p": float(wp) if wp is not None else np.nan,
                })

edges = pd.DataFrame(edges_rows)
edges["edge_kind"] = np.where(edges["src_tissue"] == edges["dst_tissue"],
                              "within_tissue", "migration")
edges.to_csv(OUT_DIR / "tcell_mig_graph_edges.csv", index=False)
print(f"Edges with ≥{MIN_CLONES_PER_EDGE} paired clones: {len(edges)}")


# %% Draw the graph
def stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def draw_graph(lineage: str, out_path: Path):
    sub = edges[edges["lineage"] == lineage].copy()
    if sub.empty:
        print(f"  no edges for {lineage}; skipping")
        return
    vmax = max(0.05, float(np.nanmax(np.abs(sub["mean_delta"].values))))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(13, 5))
    # Draw nodes
    node_size_by_n = {}
    for tissue in TISSUES:
        for t in TIMEPOINTS:
            # node size = log10(total cells in this lineage at this node) - just for visual
            n_cells = clone_node[(clone_node["lineage"] == lineage)
                                 & (clone_node["tissue"] == tissue)
                                 & (clone_node["timepoint"] == t)]["n_cells"].sum()
            node_size_by_n[(tissue, t)] = n_cells
    max_n = max(node_size_by_n.values()) if node_size_by_n else 1
    for tissue in TISSUES:
        for t in TIMEPOINTS:
            n_cells = node_size_by_n.get((tissue, t), 0)
            r = 0.07 + 0.18 * np.sqrt(n_cells / max_n) if max_n > 0 else 0.07
            circ = plt.Circle((t, TISSUE_Y[tissue]), r,
                              color=TISSUE_COLORS[tissue], alpha=0.85,
                              ec="black", lw=0.7, zorder=3)
            ax.add_patch(circ)
            ax.text(t, TISSUE_Y[tissue], f"{TISSUE_LABELS[tissue][0]}{t}",
                    ha="center", va="center", color="white", fontsize=8,
                    fontweight="bold", zorder=4)

    # Draw edges: line color by mean_delta, alpha/width by n_clones
    max_clones = sub["n_clones"].max() if len(sub) else 1
    for _, r in sub.iterrows():
        x0, y0 = r["t_src"], TISSUE_Y[r["src_tissue"]]
        x1, y1 = r["t_dst"], TISSUE_Y[r["dst_tissue"]]
        color = cmap(norm(r["mean_delta"]))
        lw = 0.8 + 3.0 * (r["n_clones"] / max_clones)
        # Curve cross-tissue edges so they don't all overlap
        if r["src_tissue"] == r["dst_tissue"]:
            connectionstyle = "arc3,rad=0.0"
        else:
            # bend by direction (going up or down)
            dy = y1 - y0
            connectionstyle = f"arc3,rad={0.15 * np.sign(dy):.2f}"
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle=connectionstyle,
            arrowstyle="->,head_length=4,head_width=3",
            color=color, lw=lw, alpha=0.85, zorder=2,
        )
        ax.add_patch(arrow)
        # Place stars near mid-edge
        s = stars(r["wilcoxon_p"])
        if s:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            # Push label off the edge a bit, perpendicular to direction
            dx, dy = x1 - x0, y1 - y0
            length = np.hypot(dx, dy)
            ox, oy = (-dy / length * 0.05, dx / length * 0.05) if length > 0 else (0, 0)
            ax.text(mx + ox, my + oy, s, fontsize=9, ha="center", va="center",
                    color="black", zorder=5)

    ax.set_xlim(0.5, max(TIMEPOINTS) + 0.5)
    ax.set_ylim(-0.6, max(TISSUE_Y.values()) + 0.6)
    ax.set_yticks(list(TISSUE_Y.values()))
    ax.set_yticklabels([TISSUE_LABELS[t] for t in
                        sorted(TISSUE_Y, key=lambda x: -TISSUE_Y[x])])
    ax.set_xticks(TIMEPOINTS)
    ax.set_xticklabels([f"T{t}" for t in TIMEPOINTS])
    ax.set_xlabel("Timepoint")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"{lineage} — per-clone Δ T-cell-migration score on forward traffic edges\n"
        f"(red = score rises src→dst; blue = falls; "
        f"line width ~ n paired clones; *p<0.05 **p<0.01 ***p<0.001 Wilcoxon)"
    )

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean Δ score (target − source)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.name}")


for lineage in ["CD8", "CD4"]:
    draw_graph(lineage, OUT_DIR / f"tcell_mig_graph_{lineage}.png")

print("\nDONE")
