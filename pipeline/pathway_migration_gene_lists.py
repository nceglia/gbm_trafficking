# %%
"""Consolidate the three migration-associated gene lists.

This script ends the analysis arc by producing three clean,
clinically-deliverable gene lists and scoring each on the collapsed
traffic graph. No up/down interpretive narrative — the deliverable is
simply: "these are the genes that change on clonal transitions."

The three lists:

  1. OT clean migration signature (from pathway_migration_signature_ot)
     — genes that "rewire" when clones transit between tissues, using
     clone-constrained Sinkhorn OT and a symmetric-component filter
     (|migration_act| ≥ |tissue_context|).  Per-lineage: CD8 has 77,
     CD4 has 100.

  2. Pathway leading-edge (from pathway_motility_graph_collapsed)
     — start from the union of 13 enriched motility/migration pathways
     (KEGG cell adhesion molecules, KEGG transendothelial migration,
     KEGG chemokine signaling, Hallmark Apical Junction, GO BP Rho
     signaling, integrin-mediated signaling, lymphocyte/T-cell
     migration, etc.). Rank every gene by its mean per-clone Δ on
     PBMC→TP and CSF→TP transitions (collapsed across timepoint pairs).
     Keep genes with delta_mean > 0.05; cap at 80 → 45 genes.

  3. Intersection — genes that appear in (1) AND (2). These are the
     most defensible "migration-rewiring" genes: they survive both the
     pathway-prior approach and the data-driven OT decomposition.

For each list we then re-build the collapsed traffic graph (per clone
× t→t+1 step) and color edges by the mean Δ of a single net score
across all genes in the list.

Outputs:
  results/migration_gene_lists/
    ot_migration_genes.csv             — union of CD8 + CD4 OT signature
    pathway_leading_edge_genes.csv     — 45 genes
    intersection_genes.csv             — overlap
    venn.png                            — Venn diagram of the three
    network_<list>_<lineage>.png        — collapsed traffic graphs (6)

Usage:
    python pipeline/pathway_migration_gene_lists.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
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
OT_DIR  = REPO_ROOT / "results" / "pathway_migration_signature_ot"
PWY_DIR = REPO_ROOT / "results" / "pathway_motility_traffic"
OUT_DIR = REPO_ROOT / "results" / "migration_gene_lists"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUE_XY = {"CSF": (0.0, 1.0), "TP": (0.87, -0.3), "PBMC": (0.0, -1.0)}
TISSUES = list(TISSUE_XY.keys())
TIMEPOINT_STEPS = [(t, t + 1) for t in [1, 2, 3, 4, 5]]
MIN_CELLS_PER_NODE = 3
MIN_CLONE_STEPS_PER_EDGE = 5
RANDOM_STATE = 42


# %% 1. Build the three definitive gene tables
print("Building OT migration gene table...")
cd8_ot = pd.read_csv(OT_DIR / "clean_migration_genes_CD8.csv")
cd4_ot = pd.read_csv(OT_DIR / "clean_migration_genes_CD4.csv")
cd8_ot["lineage"] = "CD8"; cd4_ot["lineage"] = "CD4"

# Long form: gene × lineage with direction + stats
ot_long = pd.concat([cd8_ot, cd4_ot], ignore_index=True)
ot_long = ot_long[["gene", "lineage", "direction", "mean_act", "median_act",
                    "sign_consistency", "n_pairs", "mean_tissue_context",
                    "mean_abs_tissue", "specificity_ratio"]]

# Pivot to per-gene wide form for the deliverable
ot_wide_pivot = ot_long.pivot_table(
    index="gene",
    columns="lineage",
    values=["direction", "mean_act", "mean_tissue_context", "specificity_ratio"],
    aggfunc="first",
)
ot_wide_pivot.columns = [f"{val}_{lin}" for val, lin in ot_wide_pivot.columns]
ot_wide_pivot = ot_wide_pivot.reset_index()
ot_wide_pivot["in_CD8"] = ot_wide_pivot["direction_CD8"].notna()
ot_wide_pivot["in_CD4"] = ot_wide_pivot["direction_CD4"].notna()
ot_wide_pivot["lineage_provenance"] = np.where(
    ot_wide_pivot["in_CD8"] & ot_wide_pivot["in_CD4"], "both",
    np.where(ot_wide_pivot["in_CD8"], "CD8_only", "CD4_only"),
)
ot_genes = set(ot_wide_pivot["gene"].astype(str).tolist())
ot_wide_pivot = ot_wide_pivot.sort_values(
    ["in_CD8", "in_CD4", "gene"], ascending=[False, False, True])
ot_wide_pivot.to_csv(OUT_DIR / "ot_migration_genes.csv", index=False)
print(f"  OT migration gene list: {len(ot_wide_pivot)} unique genes "
      f"({(ot_wide_pivot['lineage_provenance']=='both').sum()} in both lineages, "
      f"{(ot_wide_pivot['lineage_provenance']=='CD8_only').sum()} CD8-only, "
      f"{(ot_wide_pivot['lineage_provenance']=='CD4_only').sum()} CD4-only)")

print("\nBuilding pathway leading-edge gene table...")
le_rank = pd.read_csv(PWY_DIR / "leading_edge_gene_rank.csv")
le_genes_list = (REPO_ROOT / "results/pathway_motility_traffic/leading_edge_genes.txt").read_text().split()
le_table = le_rank[le_rank["gene"].isin(le_genes_list)].copy()
le_table = le_table.sort_values("delta_mean", ascending=False).reset_index(drop=True)
le_table["rank"] = np.arange(1, len(le_table) + 1)
le_table.to_csv(OUT_DIR / "pathway_leading_edge_genes.csv", index=False)
pathway_genes = set(le_table["gene"].astype(str).tolist())
print(f"  Pathway leading-edge gene list: {len(le_table)} genes")

print("\nBuilding intersection...")
inter_genes = ot_genes & pathway_genes
inter_rows = []
for g in sorted(inter_genes):
    row = {"gene": g}
    # OT provenance
    cd8_row = cd8_ot[cd8_ot["gene"] == g]
    cd4_row = cd4_ot[cd4_ot["gene"] == g]
    row["ot_in_CD8"] = len(cd8_row) > 0
    row["ot_in_CD4"] = len(cd4_row) > 0
    row["ot_direction_CD8"] = (cd8_row.iloc[0]["direction"]
                                if len(cd8_row) else "")
    row["ot_direction_CD4"] = (cd4_row.iloc[0]["direction"]
                                if len(cd4_row) else "")
    row["ot_mean_act_CD8"] = (float(cd8_row.iloc[0]["mean_act"])
                               if len(cd8_row) else np.nan)
    row["ot_mean_act_CD4"] = (float(cd4_row.iloc[0]["mean_act"])
                               if len(cd4_row) else np.nan)
    # Pathway provenance
    le_row = le_table[le_table["gene"] == g]
    row["pathway_delta_csf_to_tp"] = (float(le_row.iloc[0]["delta_csf_to_tp"])
                                       if len(le_row) else np.nan)
    row["pathway_delta_pbmc_to_tp"] = (float(le_row.iloc[0]["delta_pbmc_to_tp"])
                                        if len(le_row) else np.nan)
    row["pathway_delta_mean"] = (float(le_row.iloc[0]["delta_mean"])
                                  if len(le_row) else np.nan)
    inter_rows.append(row)
inter_table = pd.DataFrame(inter_rows)
inter_table.to_csv(OUT_DIR / "intersection_genes.csv", index=False)
print(f"  Intersection: {len(inter_table)} genes")
print(f"  Genes: {sorted(inter_genes)}")


# %% 2. Venn diagram
print("\nDrawing Venn diagram...")
fig, ax = plt.subplots(figsize=(7, 6))
n_ot   = len(ot_genes)
n_pwy  = len(pathway_genes)
n_both = len(inter_genes)
n_ot_only  = n_ot - n_both
n_pwy_only = n_pwy - n_both

# Simple Venn (matplotlib only — avoid matplotlib_venn import)
c1 = plt.Circle((-0.4, 0), 0.7, color="#377eb8", alpha=0.45)
c2 = plt.Circle(( 0.4, 0), 0.7, color="#e41a1c", alpha=0.45)
ax.add_patch(c1); ax.add_patch(c2)
ax.text(-0.85, 0, f"OT signature\n{n_ot_only} only\n({n_ot} total)",
        ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(0.85, 0, f"Pathway LE\n{n_pwy_only} only\n({n_pwy} total)",
        ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(0, 0, f"both\n{n_both}", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")
ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Overlap between OT and pathway-leading-edge migration gene lists",
              fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "venn.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "venn.pdf", bbox_inches="tight")
plt.close(fig)


# %% 3. Score each list on the collapsed traffic graph
print("\nScoring three lists on the collapsed traffic graph...")
adata = sc.read(str(paths.H5AD_TCELLS))
orig_X = adata.X
adata.X = adata.layers["log1p"]

LISTS = {
    "ot_migration":      sorted(ot_genes),
    "pathway_leading":   sorted(pathway_genes),
    "intersection":      sorted(inter_genes),
}
for name, glist in LISTS.items():
    glist_p = [g for g in glist if g in adata.var_names]
    if not glist_p:
        print(f"  {name}: no genes present; skipping")
        continue
    sc.tl.score_genes(adata, gene_list=glist_p,
                       score_name=f"score_{name}",
                       random_state=RANDOM_STATE)
    print(f"  scored {name}: {len(glist_p)}/{len(glist)} genes present")
adata.X = orig_X


# %% 4. Per (patient, clone, tissue, timepoint) means, then collapsed Δ
print("\nAggregating + collapsed edges...")
obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb"]
                 + [f"score_{n}" for n in LISTS]].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)

clone_node_all = (
    obs.groupby(["patient", "trb", "lineage", "tissue", "timepoint"], observed=True)
        .agg(n_cells=("score_ot_migration", "size"),
             **{f"m_{n}": (f"score_{n}", "mean") for n in LISTS})
        .reset_index()
)
clone_node_all = clone_node_all[clone_node_all["n_cells"] >= MIN_CELLS_PER_NODE]

edge_rows = []
for lineage in ["CD8", "CD4"]:
    cn = clone_node_all[clone_node_all["lineage"] == lineage]
    for src in TISSUES:
        for dst in TISSUES:
            paired = {n: [] for n in LISTS}
            n_patients = set()
            for t_s, t_d in TIMEPOINT_STEPS:
                src_n = cn[(cn["tissue"] == src) & (cn["timepoint"] == t_s)]
                dst_n = cn[(cn["tissue"] == dst) & (cn["timepoint"] == t_d)]
                m = src_n.merge(dst_n, on=["patient", "trb"],
                                suffixes=("_s", "_d"))
                if m.empty:
                    continue
                for n in LISTS:
                    paired[n].extend(
                        (m[f"m_{n}_d"] - m[f"m_{n}_s"]).tolist())
                n_patients.update(m["patient"].tolist())
            n_steps = len(paired["ot_migration"])
            if n_steps < MIN_CLONE_STEPS_PER_EDGE:
                continue
            row = {"lineage": lineage, "src_tissue": src, "dst_tissue": dst,
                   "n_clone_steps": int(n_steps),
                   "n_patients": int(len(n_patients))}
            for n, arr in paired.items():
                a = np.asarray(arr)
                try:
                    _, pv = stats.wilcoxon(a)
                except ValueError:
                    pv = np.nan
                row[f"mean_delta_{n}"] = float(a.mean())
                row[f"wilcoxon_p_{n}"] = float(pv) if pv is not None else np.nan
            edge_rows.append(row)
edges = pd.DataFrame(edge_rows)
edges.to_csv(OUT_DIR / "collapsed_edges_all_lists.csv", index=False)


# %% 5. Draw 6 collapsed graphs (3 lists × 2 lineages)
def stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def offset_along_edge(p0, p1, fraction=0.18):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = np.hypot(dx, dy)
    if length == 0: return p0, p1
    ux, uy = dx / length, dy / length
    return (p0[0] + fraction * ux, p0[1] + fraction * uy), \
           (p1[0] - fraction * ux, p1[1] - fraction * uy)


def draw_collapsed_graph(ax, edges_sub, score_col, p_col, title,
                          node_size_by_tissue, vmax):
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    max_n = max(node_size_by_tissue.values()) if node_size_by_tissue else 1
    for tissue, (x, y) in TISSUE_XY.items():
        n = node_size_by_tissue.get(tissue, 0)
        r = 0.12 + 0.10 * np.sqrt(n / max_n) if max_n > 0 else 0.12
        circ = plt.Circle((x, y), r, color=TISSUE_COLORS[tissue], alpha=0.9,
                           ec="black", lw=1.2, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, TISSUE_LABELS[tissue], ha="center", va="center",
                color="white", fontsize=11, fontweight="bold", zorder=4)
    max_steps = edges_sub["n_clone_steps"].max() if len(edges_sub) else 1
    for _, r in edges_sub.iterrows():
        s_t, d_t = r["src_tissue"], r["dst_tissue"]
        delta = r[score_col]; pval = r[p_col]; n = r["n_clone_steps"]
        color = cmap(norm(delta))
        lw = 1.5 + 4.5 * (n / max_steps)
        p0 = TISSUE_XY[s_t]; p1 = TISSUE_XY[d_t]
        if s_t == d_t:
            x, y = p0
            cx = x + (0.45 if x >= 0 else -0.45)
            cy = y + (0.05 if y >= 0 else -0.05)
            arc = FancyArrowPatch(
                (x + 0.05, y + 0.08), (x + 0.05, y - 0.08),
                connectionstyle=f"arc3,rad={(0.9 if x >= 0 else -0.9)}",
                arrowstyle="->,head_length=5,head_width=4",
                color=color, lw=lw, alpha=0.92, zorder=2,
            )
            ax.add_patch(arc)
            tx_, ty_ = cx + (0.18 if x >= 0 else -0.18), cy
            ax.text(tx_, ty_, f"{delta:+.3f}{stars(pval)}", fontsize=8,
                    ha="center", va="center", color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.7))
        else:
            (sx, sy), (tx_, ty_) = offset_along_edge(p0, p1, fraction=0.18)
            arrow = FancyArrowPatch(
                (sx, sy), (tx_, ty_),
                connectionstyle="arc3,rad=0.18",
                arrowstyle="->,head_length=6,head_width=4.5",
                color=color, lw=lw, alpha=0.92, zorder=2,
            )
            ax.add_patch(arrow)
            mx, my = (sx + tx_) / 2, (sy + ty_) / 2
            dx, dy = tx_ - sx, ty_ - sy
            length = np.hypot(dx, dy)
            ox, oy = (-dy / length * 0.14, dx / length * 0.14) if length > 0 else (0, 0)
            ax.text(mx + ox, my + oy, f"{delta:+.3f}{stars(pval)}", fontsize=8,
                    ha="center", va="center", color="black", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="none", alpha=0.7))
    ax.set_xlim(-1.4, 1.7); ax.set_ylim(-1.55, 1.55); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    return sm


LIST_LABELS = {
    "ot_migration":    "OT-derived migration signature",
    "pathway_leading": "Pathway leading-edge",
    "intersection":    "Intersection (OT ∩ Pathway)",
}
print("\nDrawing collapsed graphs...")
for list_name in LISTS:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, lineage in zip(axes, ["CD8", "CD4"]):
        sub = edges[edges["lineage"] == lineage]
        if sub.empty:
            ax.axis("off")
            continue
        node_n = (clone_node_all[clone_node_all["lineage"] == lineage]
                    .groupby("tissue")["n_cells"].sum().to_dict())
        score_col = f"mean_delta_{list_name}"
        p_col = f"wilcoxon_p_{list_name}"
        vmax = max(0.02, float(np.nanmax(np.abs(sub[score_col].values))))
        sm = draw_collapsed_graph(
            ax, sub, score_col, p_col,
            title=f"{lineage} — {LIST_LABELS[list_name]} "
                  f"(n={len([g for g in LISTS[list_name] if g in adata.var_names])} genes)",
            node_size_by_tissue=node_n, vmax=vmax,
        )
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02,
                        location="right", shrink=0.8)
    cbar.set_label("mean Δ score (target − source)")
    fig.suptitle(f"Collapsed traffic graph: {LIST_LABELS[list_name]}",
                  y=1.02, fontsize=12)
    fig.savefig(OUT_DIR / f"network_{list_name}.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"network_{list_name}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote network_{list_name}.png")


# %% 6. Summary table for the cover sheet
print("\nWriting README summary table...")
summary_rows = []
for list_name, glist in LISTS.items():
    n_total = len(glist)
    n_present = sum(1 for g in glist if g in adata.var_names)
    summary_rows.append({
        "list": list_name,
        "label": LIST_LABELS[list_name],
        "n_genes_total": n_total,
        "n_genes_scored": n_present,
        "csv": ("ot_migration_genes.csv" if list_name == "ot_migration"
                else "pathway_leading_edge_genes.csv" if list_name == "pathway_leading"
                else "intersection_genes.csv"),
    })
pd.DataFrame(summary_rows).to_csv(OUT_DIR / "gene_list_summary.csv", index=False)


# %% Final
print("\n" + "=" * 60)
print("DONE — clinical deliverable in",
      OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
