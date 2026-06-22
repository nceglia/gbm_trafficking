# %%
"""Collapsed 3-tissue migration graph with per-clone Δ scores.

This is the simpler, more powerful version of pathway_motility_graph.py.
Instead of one node per (tissue × timepoint), we collapse the temporal
axis: clones moving T1→T2, T2→T3, …, T5→T6 are pooled. Each (clone,
timepoint-step) is a paired observation. That gives many more
replicates per edge (e.g. ~150–400 paired clone-steps for CSF↔TP).

Layout follows the migration-rates figure aesthetic:
    CSF top, TP middle, PBMC bottom (triangle).

Two output figures:

  1. tcell_mig_graph_collapsed_<lineage>.png
     One score = a leading-edge migration set distilled from the
     up-enriched motility pathways in the first heatmap. The leading
     edges are those genes within the candidate pathways whose mean
     per-clone Δ on PBMC→TP and CSF→TP transitions is positive.

  2. tcell_mig_graph_navigate_vs_adhered_<lineage>.png
     Two panels side-by-side:
       LEFT  — chemokine-receptor "navigate" score
               (CCR2, CCR5, CXCR3/4, S1PR1, SELL, CCR7, GPR183, …).
       RIGHT — adhesion / Rho / actin "arrived" score
               (cell adhesion molecules, transendothelial migration,
                integrin-mediated, Rho, actin reorganization).
     Tells the "leaving" → "arrived" story in one figure.

Usage:
    python pipeline/pathway_motility_graph_collapsed.py
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
from scipy import sparse, stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.clone_helpers import infer_lineage_from_phenotype  # noqa: E402
from modules.style import TISSUE_COLORS, TISSUE_LABELS  # noqa: E402

# %% Config
OUT_DIR = REPO_ROOT / "results" / "pathway_motility_traffic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS_PER_NODE = 3
MIN_CLONE_STEPS_PER_EDGE = 5
RANDOM_STATE = 42

# Triangle layout — CSF top, TP middle-right, PBMC bottom
TISSUE_XY = {
    "CSF":  (0.0, 1.0),
    "TP":   (0.87, -0.3),
    "PBMC": (0.0, -1.0),
}
TISSUES = list(TISSUE_XY.keys())
TIMEPOINT_STEPS = [(t, t + 1) for t in [1, 2, 3, 4, 5]]

# --- Pathway candidates for the leading-edge migration set ---
# Pathways that came up enriched (mean_diff > 0, FDR < 0.05) in CSF→TP
# arrivers vs TP residents in our earlier motility_directional_tests:
LEADING_EDGE_PATHWAYS = [
    ("KEGG_2021_Human", "Cell adhesion molecules"),
    ("KEGG_2021_Human", "Leukocyte transendothelial migration"),
    ("KEGG_2021_Human", "Chemokine signaling pathway"),
    ("MSigDB_Hallmark_2020", "Apical Junction"),
    ("GO_Biological_Process_2023", "Rho Protein Signal Transduction (GO:0007266)"),
    ("GO_Biological_Process_2023", "Integrin-Mediated Signaling Pathway (GO:0007229)"),
    ("GO_Biological_Process_2023", "Lymphocyte Migration (GO:0072676)"),
    ("GO_Biological_Process_2023", "Lymphocyte Chemotaxis (GO:0048247)"),
    ("GO_Biological_Process_2023",
     "Negative Regulation Of Actin Filament Polymerization (GO:0030837)"),
    ("GO_Biological_Process_2023",
     "Positive Regulation Of Lymphocyte Chemotaxis (GO:0140131)"),
    ("GO_Biological_Process_2023",
     "Positive Regulation Of T Cell Migration (GO:2000406)"),
    ("GO_Biological_Process_2023",
     "Leukocyte Chemotaxis Involved In Inflammatory Response (GO:0002232)"),
    ("GO_Biological_Process_2023",
     "Positive Regulation Of Leukocyte Migration (GO:0002687)"),
]

# Curated "navigate" gene panel — chemokine receptors + ligands +
# trafficking guidance. Anchors trafficking through circulation.
NAVIGATE_GENES = [
    "CCR1", "CCR2", "CCR3", "CCR4", "CCR5", "CCR6", "CCR7", "CCR9", "CCR10",
    "CXCR3", "CXCR4", "CXCR5", "CXCR6", "CX3CR1",
    "S1PR1", "S1PR4", "S1PR5",
    "SELL", "SELPLG",
    "GPR183",   # EBI2 — secondary-lymphoid migration
    "CCL3", "CCL4", "CCL5", "CCL20",
    "CXCL10", "CXCL11", "CXCL12", "CXCL13", "CXCL16",
    "XCL1", "XCL2",
]


# %% Load adata
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")


# %% Build candidate union of genes from enriched pathways
print("\nFetching pathway libraries and building candidate gene set...")
lib_cache: dict[str, dict] = {}
candidate_genes: set[str] = set()
for lib, term in LEADING_EDGE_PATHWAYS:
    if lib not in lib_cache:
        lib_cache[lib] = gp.get_library(name=lib)
    if term in lib_cache[lib]:
        candidate_genes.update(lib_cache[lib][term])
    else:
        print(f"  WARN: term not found: {lib}::{term}")
candidate_genes = sorted(candidate_genes & set(adata.var_names))
print(f"  candidate genes (union, in adata): {len(candidate_genes)}")


# %% Pick the leading-edge subset: genes with positive mean Δ on PBMC→TP & CSF→TP transitions
# Strategy: compute per (patient, clone, tissue) mean log1p for every
# candidate gene, then per-clone Δ for migration edges, average across
# clones × edges. Keep genes whose averaged Δ is positive.
print("\nIdentifying leading-edge genes (up-on-migration-into-TP)...")
gene_idx = adata.var_names.get_indexer(candidate_genes)
X = adata.layers["log1p"][:, gene_idx]
if sparse.issparse(X):
    X = X.toarray()
X = np.asarray(X, dtype=np.float32)
print(f"  candidate expression matrix: {X.shape}")

obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb"]].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)
obs["row_idx"] = np.arange(adata.n_obs)[obs.index.map(lambda x: True).values]  # placeholder

# Re-derive row idx properly (adata.obs is index-aligned with X rows)
row_pos = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
obs["row_idx"] = obs.index.map(row_pos)

# Per (patient, clone, tissue) mean gene matrix (collapse timepoints for leading-edge calc)
print("  computing per (patient, clone, tissue) gene means for leading-edge selection...")
key = list(zip(obs["patient"].values, obs["trb"].values, obs["tissue"].values))
df_key = pd.DataFrame({"key": key, "row": obs["row_idx"].values,
                        "lineage": obs["lineage"].values})
# Group rows by key
groups = df_key.groupby("key", sort=False)
group_keys = list(groups.groups.keys())
gene_means = np.zeros((len(group_keys), len(candidate_genes)), dtype=np.float32)
group_meta = []
for gi, (k, idx) in enumerate(groups):
    rows = idx["row"].values
    gene_means[gi] = X[rows].mean(axis=0)
    group_meta.append({"patient": k[0], "trb": k[1], "tissue": k[2],
                       "n_cells": len(rows),
                       "lineage": idx["lineage"].iloc[0]})
gm = pd.DataFrame(group_meta)
gm["gi"] = np.arange(len(gm))
print(f"  per (patient, clone, tissue) groups: {len(gm)}")

# For each clone, compute Δ for CSF→TP and PBMC→TP if both endpoints exist
def per_clone_delta(src_tissue, dst_tissue):
    src = gm[gm["tissue"] == src_tissue].set_index(["patient", "trb"])["gi"]
    dst = gm[gm["tissue"] == dst_tissue].set_index(["patient", "trb"])["gi"]
    common = src.index.intersection(dst.index)
    if len(common) == 0:
        return np.zeros(len(candidate_genes)), 0
    src_idx = src.loc[common].values
    dst_idx = dst.loc[common].values
    delta = gene_means[dst_idx] - gene_means[src_idx]
    return delta.mean(axis=0), len(common)


d_csf_tp, n_csf_tp = per_clone_delta("CSF", "TP")
d_pbmc_tp, n_pbmc_tp = per_clone_delta("PBMC", "TP")
print(f"  paired clones CSF→TP: {n_csf_tp}, PBMC→TP: {n_pbmc_tp}")

mean_delta = 0.5 * d_csf_tp + 0.5 * d_pbmc_tp
gene_rank = pd.DataFrame({"gene": candidate_genes,
                          "delta_csf_to_tp": d_csf_tp,
                          "delta_pbmc_to_tp": d_pbmc_tp,
                          "delta_mean": mean_delta})
gene_rank = gene_rank.sort_values("delta_mean", ascending=False)
gene_rank.to_csv(OUT_DIR / "leading_edge_gene_rank.csv", index=False)

# Take genes with mean Δ > 0.05 (modest but consistent), capped at 80
leading_edge = gene_rank[gene_rank["delta_mean"] > 0.05].head(80)
print(f"  leading-edge genes (mean Δ > 0.05, top 80): {len(leading_edge)}")
print(f"  top 15: {leading_edge['gene'].head(15).tolist()}")

LEADING_EDGE_GENES = leading_edge["gene"].tolist()
(OUT_DIR / "leading_edge_genes.txt").write_text("\n".join(LEADING_EDGE_GENES) + "\n")


# %% Score the three panels on every cell
print("\nScoring three panels on every cell...")
orig_X = adata.X
adata.X = adata.layers["log1p"]

sc.tl.score_genes(adata, gene_list=LEADING_EDGE_GENES,
                  score_name="mig_leading_edge", random_state=RANDOM_STATE)

navigate_present = [g for g in NAVIGATE_GENES if g in adata.var_names]
print(f"  navigate panel: {len(navigate_present)}/{len(NAVIGATE_GENES)} genes present")
sc.tl.score_genes(adata, gene_list=navigate_present,
                  score_name="mig_navigate", random_state=RANDOM_STATE)

# Adhered panel = KEGG cell adhesion molecules + KEGG transendothelial
# + GO Rho + GO Integrin + GO Actin reorganization + Apical Junction
adhered_genes: set[str] = set()
for lib, term in [
    ("KEGG_2021_Human", "Cell adhesion molecules"),
    ("KEGG_2021_Human", "Leukocyte transendothelial migration"),
    ("GO_Biological_Process_2023", "Rho Protein Signal Transduction (GO:0007266)"),
    ("GO_Biological_Process_2023", "Integrin-Mediated Signaling Pathway (GO:0007229)"),
    ("GO_Biological_Process_2023", "Actin Cytoskeleton Reorganization (GO:0031532)"),
    ("MSigDB_Hallmark_2020", "Apical Junction"),
]:
    if lib not in lib_cache:
        lib_cache[lib] = gp.get_library(name=lib)
    if term in lib_cache[lib]:
        adhered_genes.update(lib_cache[lib][term])
adhered_genes = sorted(adhered_genes & set(adata.var_names))
print(f"  adhered panel: {len(adhered_genes)} genes")
sc.tl.score_genes(adata, gene_list=adhered_genes,
                  score_name="mig_adhered", random_state=RANDOM_STATE)

adata.X = orig_X


# %% Per (patient, clone, tissue, timepoint) means for the three scores
print("\nAggregating per (patient, clone, tissue, timepoint)...")
obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb",
                  "mig_leading_edge", "mig_navigate", "mig_adhered"]].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)

clone_node = (
    obs.groupby(["patient", "trb", "lineage", "tissue", "timepoint"], observed=True)
       .agg(n_cells=("mig_leading_edge", "size"),
            score_leading=("mig_leading_edge", "mean"),
            score_navigate=("mig_navigate", "mean"),
            score_adhered=("mig_adhered", "mean"))
       .reset_index()
)
clone_node = clone_node[clone_node["n_cells"] >= MIN_CELLS_PER_NODE]
clone_node.to_csv(OUT_DIR / "clone_node_three_scores.csv", index=False)
print(f"  per (clone, node) pseudobulks: {len(clone_node)}")


# %% Collapsed edges: for each (src_tissue, dst_tissue), collect per (clone, t→t+1) Δ
print("\nComputing collapsed-edge deltas per score...")
edge_rows = []
for lineage in ["CD8", "CD4"]:
    cn = clone_node[clone_node["lineage"] == lineage]
    for src_tissue in TISSUES:
        for dst_tissue in TISSUES:
            paired_deltas = {"leading": [], "navigate": [], "adhered": []}
            n_patients_set = set()
            for t_src, t_dst in TIMEPOINT_STEPS:
                src = cn[(cn["tissue"] == src_tissue) & (cn["timepoint"] == t_src)]
                dst = cn[(cn["tissue"] == dst_tissue) & (cn["timepoint"] == t_dst)]
                merged = src.merge(dst, on=["patient", "trb"],
                                   suffixes=("_s", "_d"))
                if merged.empty:
                    continue
                paired_deltas["leading"].extend(
                    (merged["score_leading_d"] - merged["score_leading_s"]).tolist())
                paired_deltas["navigate"].extend(
                    (merged["score_navigate_d"] - merged["score_navigate_s"]).tolist())
                paired_deltas["adhered"].extend(
                    (merged["score_adhered_d"] - merged["score_adhered_s"]).tolist())
                n_patients_set.update(merged["patient"].tolist())
            n_steps = len(paired_deltas["leading"])
            if n_steps < MIN_CLONE_STEPS_PER_EDGE:
                continue
            row = {
                "lineage": lineage,
                "src_tissue": src_tissue, "dst_tissue": dst_tissue,
                "n_clone_steps": int(n_steps),
                "n_patients": int(len(n_patients_set)),
            }
            for score_name, arr in paired_deltas.items():
                arr = np.asarray(arr)
                try:
                    wstat, wp = stats.wilcoxon(arr)
                except ValueError:
                    wstat, wp = np.nan, np.nan
                row[f"mean_delta_{score_name}"] = float(arr.mean())
                row[f"median_delta_{score_name}"] = float(np.median(arr))
                row[f"wilcoxon_p_{score_name}"] = float(wp) if wp is not None else np.nan
            edge_rows.append(row)

edges = pd.DataFrame(edge_rows)
edges["edge_kind"] = np.where(edges["src_tissue"] == edges["dst_tissue"],
                              "within_tissue", "migration")
edges.to_csv(OUT_DIR / "tcell_mig_graph_collapsed_edges.csv", index=False)
print(f"  edges: {len(edges)}")


# %% Plotting helpers
def stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def offset_along_edge(p0, p1, fraction=0.18):
    """Inset endpoint by `fraction` of edge length so arrows don't overlap node circle."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = np.hypot(dx, dy)
    if length == 0:
        return p0, p1
    ux, uy = dx / length, dy / length
    return (p0[0] + fraction * ux, p0[1] + fraction * uy), \
           (p1[0] - fraction * ux, p1[1] - fraction * uy)


def draw_collapsed_graph(ax, edges_sub: pd.DataFrame, score_col: str, p_col: str,
                          title: str, node_size_by_tissue: dict, vmax: float):
    """Draw the 3-tissue collapsed graph onto ax."""
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    # Nodes
    max_n = max(node_size_by_tissue.values()) if node_size_by_tissue else 1
    for tissue, (x, y) in TISSUE_XY.items():
        n = node_size_by_tissue.get(tissue, 0)
        r = 0.12 + 0.10 * np.sqrt(n / max_n) if max_n > 0 else 0.12
        circ = plt.Circle((x, y), r, color=TISSUE_COLORS[tissue], alpha=0.9,
                           ec="black", lw=1.2, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, TISSUE_LABELS[tissue],
                ha="center", va="center", color="white", fontsize=11,
                fontweight="bold", zorder=4)

    max_steps = edges_sub["n_clone_steps"].max() if len(edges_sub) else 1
    for _, r in edges_sub.iterrows():
        src_tissue, dst_tissue = r["src_tissue"], r["dst_tissue"]
        delta = r[score_col]; pval = r[p_col]; n = r["n_clone_steps"]
        color = cmap(norm(delta))
        lw = 1.5 + 4.5 * (n / max_steps)
        p0 = TISSUE_XY[src_tissue]; p1 = TISSUE_XY[dst_tissue]

        if src_tissue == dst_tissue:
            # Self-loop: draw a small arc outside the node
            x, y = p0
            # Push the loop outward depending on tissue position
            cx = x + (0.45 if x >= 0 else -0.45)
            cy = y + (0.05 if y >= 0 else -0.05)
            arc = FancyArrowPatch(
                (x + 0.05, y + 0.08), (x + 0.05, y - 0.08),
                connectionstyle=f"arc3,rad={(0.9 if x >= 0 else -0.9)}",
                arrowstyle="->,head_length=5,head_width=4",
                color=color, lw=lw, alpha=0.92, zorder=2,
            )
            ax.add_patch(arc)
            tx, ty = cx + (0.18 if x >= 0 else -0.18), cy
            ax.text(tx, ty, f"{delta:+.3f}{stars(pval)}",
                    fontsize=8, ha="center", va="center",
                    color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.7))
        else:
            (sx, sy), (tx_, ty_) = offset_along_edge(p0, p1, fraction=0.18)
            # Curve so PBMC↔CSF, PBMC↔TP, CSF↔TP don't overlap (each pair has two directions)
            # Curve to the LEFT relative to the direction of travel
            rad = 0.18
            arrow = FancyArrowPatch(
                (sx, sy), (tx_, ty_),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="->,head_length=6,head_width=4.5",
                color=color, lw=lw, alpha=0.92, zorder=2,
            )
            ax.add_patch(arrow)
            # Label at midpoint, offset perpendicular to edge
            mx, my = (sx + tx_) / 2, (sy + ty_) / 2
            dx, dy = tx_ - sx, ty_ - sy
            length = np.hypot(dx, dy)
            ox, oy = (-dy / length * 0.14, dx / length * 0.14) if length > 0 else (0, 0)
            ax.text(mx + ox, my + oy, f"{delta:+.3f}{stars(pval)}",
                    fontsize=8, ha="center", va="center",
                    color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.7))

    ax.set_xlim(-1.4, 1.7)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    return sm


# %% Draw single-score plot per lineage (leading-edge migration)
print("\nDrawing collapsed graphs...")
for lineage in ["CD8", "CD4"]:
    sub = edges[edges["lineage"] == lineage]
    if sub.empty:
        continue
    node_n = (clone_node[clone_node["lineage"] == lineage]
                .groupby("tissue")["n_cells"].sum().to_dict())
    vmax = max(0.02, float(np.nanmax(np.abs(sub["mean_delta_leading"].values))))

    fig, ax = plt.subplots(figsize=(7, 7))
    sm = draw_collapsed_graph(
        ax, sub, "mean_delta_leading", "wilcoxon_p_leading",
        title=f"{lineage}: leading-edge migration score\n"
              f"per-clone Δ along forward traffic edges (T_n → T_n+1, collapsed)\n"
              "(red = score rises src→dst; blue = falls; line width ~ #paired clone-steps)",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean Δ leading-edge score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"tcell_mig_graph_collapsed_{lineage}.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"tcell_mig_graph_collapsed_{lineage}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote tcell_mig_graph_collapsed_{lineage}.png")


# %% Draw paired navigate-vs-adhered plot
for lineage in ["CD8", "CD4"]:
    sub = edges[edges["lineage"] == lineage]
    if sub.empty:
        continue
    node_n = (clone_node[clone_node["lineage"] == lineage]
                .groupby("tissue")["n_cells"].sum().to_dict())
    vmax_nav = max(0.02, float(np.nanmax(np.abs(sub["mean_delta_navigate"].values))))
    vmax_adh = max(0.02, float(np.nanmax(np.abs(sub["mean_delta_adhered"].values))))
    vmax = max(vmax_nav, vmax_adh)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    sm = draw_collapsed_graph(
        axes[0], sub, "mean_delta_navigate", "wilcoxon_p_navigate",
        title=f"{lineage} — Navigate score\n"
              "(chemokine receptors + ligands: CCR/CXCR/S1PR/SELL/etc.)",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    draw_collapsed_graph(
        axes[1], sub, "mean_delta_adhered", "wilcoxon_p_adhered",
        title=f"{lineage} — Arrived / adhered score\n"
              "(cell adhesion, transendothelial, integrin, Rho, actin reorg)",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02, location="right",
                        shrink=0.8)
    cbar.set_label("mean Δ score (target − source)")
    fig.suptitle(
        f"{lineage}: 'leaving' (navigate) vs 'arrived' (adhered) — paired per clone × t→t+1",
        y=1.02,
    )
    fig.savefig(OUT_DIR / f"tcell_mig_graph_navigate_vs_adhered_{lineage}.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"tcell_mig_graph_navigate_vs_adhered_{lineage}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote tcell_mig_graph_navigate_vs_adhered_{lineage}.png")


# %% Final
print("\n" + "=" * 60)
print("DONE — outputs in", OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.glob("tcell_mig_graph_collapsed_*")):
    print(f"  {f.name}")
for f in sorted(OUT_DIR.glob("tcell_mig_graph_navigate*")):
    print(f"  {f.name}")
for f in sorted(OUT_DIR.glob("leading_edge*")):
    print(f"  {f.name}")
