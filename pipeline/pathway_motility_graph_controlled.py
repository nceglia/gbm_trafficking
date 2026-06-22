# %%
"""Inflammation-controlled migration signature on the collapsed traffic graph.

Concern from the previous leading-edge analysis: the top genes were
dominated by NF-κB / TNF / HLA / checkpoint — i.e. a generic tumor
inflammation signature, not migration per se. So we couldn't tell
whether the migration program is independent or just inflammation.

This script splits the signal cleanly:

  - mig_machinery   : strict migration-and-motility genes only —
                      chemokine receptors (CCR/CXCR/S1PR/SELL/SELPLG/
                      CX3CR1), TRM markers (CXCR6, ITGAE, ITGA1, CD69,
                      ZNF683, RUNX3, PRDM1), KLF2/S1PR (lymphoid-egress),
                      cytoskeletal motility machinery (RHO/RAC/CDC42,
                      ROCK, ARP2/3, WASF, MSN/EZR/RDX, ACTB/ACTG1),
                      DOCK2/8, integrins used in extravasation (ITGB1/2/
                      7, ITGAL/A4/M, ICAM/VCAM).
                      Explicitly NOT NF-κB / TNF / HLA / checkpoint.

  - inflammation    : NF-κB family (NFKBIA/B, NFKB1/2, RELA, RELB,
                      TNFAIP3), TNF / TRAF / LT, IRF1/4, STAT1/3,
                      GBP1/5, HLA class I/II, B2M, TAP1/2, checkpoint
                      (CTLA4, PDCD1, LAG3, TIGIT, HAVCR2), ICOS.

  - mig_residual    : per-cell, regress mig_machinery on inflammation
                      with a simple OLS (across all cells); residual
                      score is the migration variation NOT explained
                      by inflammation. If migration is just inflammation,
                      the residual graph will be flat. If it survives,
                      motility is its own program.

We then re-draw the collapsed 3-tissue graph (CSF top, TP middle, PBMC
bottom) for each of the three signatures so you can read directly
whether the migration story holds after controlling for inflammation.

Usage:
    python pipeline/pathway_motility_graph_controlled.py
"""
import sys
import warnings
from pathlib import Path

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

MIN_CELLS_PER_NODE = 3
MIN_CLONE_STEPS_PER_EDGE = 5
RANDOM_STATE = 42

# CSF top, TP middle-right, PBMC bottom
TISSUE_XY = {
    "CSF":  (0.0, 1.0),
    "TP":   (0.87, -0.3),
    "PBMC": (0.0, -1.0),
}
TISSUES = list(TISSUE_XY.keys())
TIMEPOINT_STEPS = [(t, t + 1) for t in [1, 2, 3, 4, 5]]


# --- Curated gene panels ---
# Migration machinery: chemokine receptors, tissue residency markers,
# motility cytoskeleton, integrins / endothelial-adhesion molecules.
# DELIBERATELY EXCLUDES NF-κB, TNF, HLA, checkpoint, IFN-γ targets.
MIGRATION_MACHINERY = [
    # Chemokine receptors
    "CCR1", "CCR2", "CCR3", "CCR4", "CCR5", "CCR6", "CCR7", "CCR9",
    "CXCR3", "CXCR4", "CXCR5", "CX3CR1",
    "S1PR1", "S1PR4", "S1PR5",
    # Lymphoid egress / blood trafficking
    "SELL", "SELPLG", "KLF2",
    # Tissue residency markers (TRM commitment)
    "CXCR6", "ITGAE", "ITGA1", "CD69", "ZNF683", "RUNX3", "PRDM1",
    # Motility cytoskeleton: Rho/Rac/CDC42 GTPases
    "RHOA", "RHOB", "RHOH", "RAC1", "RAC2", "CDC42",
    "ROCK1", "ROCK2",
    # Actin nucleation / dynamics
    "WASF1", "WASF2", "ARPC1B", "ARPC2", "ARPC3", "ARPC4", "ARPC5",
    "ACTB", "ACTG1", "ACTR2", "ACTR3",
    # Ezrin / membrane-actin scaffolding
    "MSN", "EZR", "RDX",
    # Pan-T-cell chemotaxis GEFs
    "DOCK2", "DOCK8",
    # Integrins used in extravasation / endothelial transmigration
    "ITGB1", "ITGB2", "ITGB7", "ITGAL", "ITGA4", "ITGAM",
    # Adhesion-molecule receptors / counter-receptors
    "ICAM1", "ICAM2", "ICAM3", "VCAM1", "CD99", "JAM3", "PECAM1",
    # Sphingosine kinase / lipid sensing
    "SPHK1", "SPHK2",
]

# Inflammation panel: things to control for.
INFLAMMATION = [
    # NF-κB family
    "NFKBIA", "NFKBIB", "NFKBIE", "NFKBIZ", "NFKB1", "NFKB2",
    "RELA", "RELB", "REL", "TNFAIP3", "NFKB2",
    # TNF / LT
    "TNF", "LTA", "LTB", "TRAF1", "TRAF2", "TRAF3",
    "TNFRSF1A", "TNFRSF1B",
    # IFN-γ / IRF / STAT
    "IRF1", "IRF4", "STAT1", "STAT3", "STAT4",
    # IFN-γ targets
    "GBP1", "GBP5", "WARS1",
    # HLA class I / antigen processing (IFN-γ-induced)
    "HLA-A", "HLA-B", "HLA-C", "HLA-E", "B2M", "TAP1", "TAP2", "TAPBP",
    # HLA class II (activation-induced on T cells)
    "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1",
    # T-cell activation / checkpoint
    "CTLA4", "PDCD1", "LAG3", "TIGIT", "HAVCR2", "ICOS",
    "CD27", "TNFRSF9",  # 4-1BB
    # Activation-induced cytokine production (autocrine)
    "IFNG", "IL2", "IL2RA",
]


# %% Load
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

mig_present = [g for g in MIGRATION_MACHINERY if g in adata.var_names]
inf_present = [g for g in INFLAMMATION if g in adata.var_names]
print(f"migration_machinery: {len(mig_present)}/{len(MIGRATION_MACHINERY)} present")
print(f"inflammation       : {len(inf_present)}/{len(INFLAMMATION)} present")

orig_X = adata.X
adata.X = adata.layers["log1p"]
sc.tl.score_genes(adata, gene_list=mig_present,
                  score_name="mig_machinery", random_state=RANDOM_STATE)
sc.tl.score_genes(adata, gene_list=inf_present,
                  score_name="inflammation", random_state=RANDOM_STATE)
adata.X = orig_X


# %% Per-cell residual: mig_machinery after regressing out inflammation
print("\nComputing per-cell residual migration score (OLS regression on inflammation)...")
x = adata.obs["inflammation"].values
y = adata.obs["mig_machinery"].values
# Fit a single global linear model: y = a + b * x
mask = np.isfinite(x) & np.isfinite(y)
slope, intercept = np.polyfit(x[mask], y[mask], 1)
y_hat = slope * x + intercept
adata.obs["mig_residual"] = y - y_hat
print(f"  OLS: mig_machinery = {slope:.3f} * inflammation + {intercept:.3f}")
print(f"  Pearson r (mig_machinery, inflammation) = "
      f"{np.corrcoef(x[mask], y[mask])[0,1]:.3f}")


# %% Per (patient, clone, tissue, timepoint) means
print("\nAggregating per (patient, clone, tissue, timepoint)...")
obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb",
                  "mig_machinery", "inflammation", "mig_residual"]].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)

clone_node = (
    obs.groupby(["patient", "trb", "lineage", "tissue", "timepoint"], observed=True)
       .agg(n_cells=("mig_machinery", "size"),
            score_mig=("mig_machinery", "mean"),
            score_inf=("inflammation", "mean"),
            score_resid=("mig_residual", "mean"))
       .reset_index()
)
clone_node = clone_node[clone_node["n_cells"] >= MIN_CELLS_PER_NODE]
clone_node.to_csv(OUT_DIR / "clone_node_controlled_scores.csv", index=False)
print(f"  per (clone, node) pseudobulks: {len(clone_node)}")


# %% Collapsed per-edge Δ
print("\nComputing collapsed-edge deltas per score...")
edge_rows = []
for lineage in ["CD8", "CD4"]:
    cn = clone_node[clone_node["lineage"] == lineage]
    for src_tissue in TISSUES:
        for dst_tissue in TISSUES:
            paired_deltas = {"mig": [], "inf": [], "resid": []}
            n_patients_set = set()
            for t_src, t_dst in TIMEPOINT_STEPS:
                src = cn[(cn["tissue"] == src_tissue) & (cn["timepoint"] == t_src)]
                dst = cn[(cn["tissue"] == dst_tissue) & (cn["timepoint"] == t_dst)]
                merged = src.merge(dst, on=["patient", "trb"],
                                   suffixes=("_s", "_d"))
                if merged.empty:
                    continue
                paired_deltas["mig"].extend(
                    (merged["score_mig_d"] - merged["score_mig_s"]).tolist())
                paired_deltas["inf"].extend(
                    (merged["score_inf_d"] - merged["score_inf_s"]).tolist())
                paired_deltas["resid"].extend(
                    (merged["score_resid_d"] - merged["score_resid_s"]).tolist())
                n_patients_set.update(merged["patient"].tolist())
            n_steps = len(paired_deltas["mig"])
            if n_steps < MIN_CLONE_STEPS_PER_EDGE:
                continue
            row = {"lineage": lineage,
                   "src_tissue": src_tissue, "dst_tissue": dst_tissue,
                   "n_clone_steps": int(n_steps),
                   "n_patients": int(len(n_patients_set))}
            for s_name, arr in paired_deltas.items():
                a = np.asarray(arr)
                try:
                    wstat, wp = stats.wilcoxon(a)
                except ValueError:
                    wstat, wp = np.nan, np.nan
                row[f"mean_delta_{s_name}"] = float(a.mean())
                row[f"wilcoxon_p_{s_name}"] = float(wp) if wp is not None else np.nan
            edge_rows.append(row)

edges = pd.DataFrame(edge_rows)
edges["edge_kind"] = np.where(edges["src_tissue"] == edges["dst_tissue"],
                              "within_tissue", "migration")
edges.to_csv(OUT_DIR / "tcell_mig_graph_controlled_edges.csv", index=False)
print(f"  edges: {len(edges)}")


# %% Plot helpers (same as before)
def stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def offset_along_edge(p0, p1, fraction=0.18):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = np.hypot(dx, dy)
    if length == 0: return p0, p1
    ux, uy = dx / length, dy / length
    return (p0[0] + fraction * ux, p0[1] + fraction * uy), \
           (p1[0] - fraction * ux, p1[1] - fraction * uy)


def draw_collapsed_graph(ax, edges_sub: pd.DataFrame, score_col: str, p_col: str,
                          title: str, node_size_by_tissue: dict, vmax: float):
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
        src_tissue, dst_tissue = r["src_tissue"], r["dst_tissue"]
        delta = r[score_col]; pval = r[p_col]; n = r["n_clone_steps"]
        color = cmap(norm(delta))
        lw = 1.5 + 4.5 * (n / max_steps)
        p0 = TISSUE_XY[src_tissue]; p1 = TISSUE_XY[dst_tissue]
        if src_tissue == dst_tissue:
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
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7))
        else:
            (sx, sy), (tx_, ty_) = offset_along_edge(p0, p1, fraction=0.18)
            rad = 0.18
            arrow = FancyArrowPatch(
                (sx, sy), (tx_, ty_),
                connectionstyle=f"arc3,rad={rad}",
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
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.7))
    ax.set_xlim(-1.4, 1.7); ax.set_ylim(-1.55, 1.55); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    return sm


# %% Draw 3-panel figure per lineage
print("\nDrawing 3-panel controlled graphs...")
for lineage in ["CD8", "CD4"]:
    sub = edges[edges["lineage"] == lineage]
    if sub.empty:
        continue
    node_n = (clone_node[clone_node["lineage"] == lineage]
                .groupby("tissue")["n_cells"].sum().to_dict())
    vmax = max(
        0.02,
        float(np.nanmax(np.abs(sub["mean_delta_mig"].values))),
        float(np.nanmax(np.abs(sub["mean_delta_inf"].values))),
        float(np.nanmax(np.abs(sub["mean_delta_resid"].values))),
    )
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    sm = draw_collapsed_graph(
        axes[0], sub, "mean_delta_mig", "wilcoxon_p_mig",
        title=f"{lineage} — Migration machinery\n"
              "(chemokine receptors, TRM markers, Rho/Rac/CDC42, ARP2/3, ezrin, integrins)",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    draw_collapsed_graph(
        axes[1], sub, "mean_delta_inf", "wilcoxon_p_inf",
        title=f"{lineage} — Inflammation\n"
              "(NF-κB / TNF / IFN-γ / HLA / checkpoint)",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    draw_collapsed_graph(
        axes[2], sub, "mean_delta_resid", "wilcoxon_p_resid",
        title=f"{lineage} — Migration residual\n"
              "(migration after regressing out inflammation per cell)",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02,
                        location="right", shrink=0.8)
    cbar.set_label("mean Δ score (target − source)")
    fig.suptitle(
        f"{lineage}: Migration vs inflammation on collapsed traffic edges — "
        "is motility independent of inflammation?",
        y=1.02, fontsize=12,
    )
    fig.savefig(OUT_DIR / f"tcell_mig_graph_controlled_{lineage}.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"tcell_mig_graph_controlled_{lineage}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote tcell_mig_graph_controlled_{lineage}.png")


# %% Save panel definitions
panels_df = pd.DataFrame({
    "panel": (["migration_machinery"] * len(mig_present)
               + ["inflammation"] * len(inf_present)),
    "gene": mig_present + inf_present,
})
panels_df.to_csv(OUT_DIR / "controlled_panels_genes.csv", index=False)


# %% Final summary
print("\n" + "=" * 60)
print("DONE — controlled-migration outputs in",
      OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.glob("*controlled*")):
    print(f"  {f.name}")
