# %%
"""Motility / migration pathway enrichment on clonal-traffic edges.

For every clonal-traffic edge (clone present at (src tissue, t_src) and
(dst tissue, t_dst); from traffic_branch_empirics/branches.csv) we ask
whether *cells participating in that edge* carry elevated motility-
related pathway activity, and whether migration edges (cross-tissue)
differ from within-tissue edges.

Motility panel: a hand-curated mix of KEGG (Leukocyte transendothelial
migration, Focal adhesion, Regulation of actin cytoskeleton, Cell
adhesion molecules, Chemokine signaling, Adherens / Tight junctions),
Hallmark (Apical Junction, EMT, TNF/NF-kB as comparator), and small,
T-cell-relevant GO BP sets (T Cell Migration, T Cell Chemotaxis, Rho
Protein Signal Transduction, Actin Cytoskeleton Reorganization,
Leukocyte Chemotaxis, Lymphocyte Migration, Cell Migration).

Comparison frame:
  - For each edge in branches.csv, "source cells" are cells in
    (patient, trb) at (src tissue, t_src); "target cells" similarly at
    (dst tissue, t_dst).
  - We label each cell by the *kind* of edge it participates in:
    'migration_<src>_<dst>' for cross-tissue edges,
    'within_<tissue>' for same-tissue edges.
  - For each motility pathway, score every cell (sc.tl.score_genes on
    log1p), then compare migration-edge source/target cells against
    within-tissue source cells.

We report:
  - Heatmap of mean pathway score per (edge_label × role) cell pool
  - Per-pathway directional test: migration source vs within-tissue
    source (does motility ramp up while leaving) and migration target
    vs within-tissue persister (does motility stay elevated on arrival)
  - Patient-level pseudobulk version for sanity (per-cell tests are
    cell-count-powered).

Usage:
    python pipeline/pathway_motility_traffic.py
"""
import sys
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.clone_helpers import infer_lineage_from_phenotype  # noqa: E402

# %% Config
BRANCHES_CSV = REPO_ROOT / "results" / "traffic_branch_empirics" / "branches.csv"
OUT_DIR = REPO_ROOT / "results" / "pathway_motility_traffic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS_PER_GROUP = 20
RANDOM_STATE = 42

# Motility / migration pathway library (curated).
KEGG_TERMS = [
    "Leukocyte transendothelial migration",
    "Focal adhesion",
    "Regulation of actin cytoskeleton",
    "Cell adhesion molecules",
    "Chemokine signaling pathway",
    "Adherens junction",
    "Tight junction",
    "Rap1 signaling pathway",
    "ECM-receptor interaction",
]
HALLMARK_TERMS = [
    "Apical Junction",
    "Epithelial Mesenchymal Transition",
    "TNF-alpha Signaling via NF-kB",   # comparator (already known up)
    "IL2/STAT5 Signaling",              # comparator
]
GOBP_TERMS_PATTERNS = [
    r"\bT Cell Migration\b",
    r"\bT Cell Chemotaxis\b",
    r"\bLymphocyte Migration\b",
    r"\bLymphocyte Chemotaxis\b",
    r"\bLeukocyte Chemotaxis\b",
    r"\bLeukocyte Migration\b",
    r"\bCell Migration\b",
    r"\bRho Protein Signal Transduction\b",
    r"\bActin Cytoskeleton Reorganization\b",
    r"\bActin Filament Polymerization\b",
    r"\bAmeboidal-Type Cell Migration\b",
    r"\bIntegrin-Mediated Signaling Pathway\b",
]


def build_pathway_panel() -> dict[str, list[str]]:
    """Return {pathway_label: [genes]} for the curated motility panel."""
    panel = {}
    kegg = gp.get_library(name="KEGG_2021_Human")
    for t in KEGG_TERMS:
        if t in kegg:
            panel[f"KEGG: {t}"] = list(set(kegg[t]))
        else:
            print(f"  WARN: KEGG term not found: {t}")
    hall = gp.get_library(name="MSigDB_Hallmark_2020")
    for t in HALLMARK_TERMS:
        if t in hall:
            panel[f"Hallmark: {t}"] = list(set(hall[t]))
        else:
            print(f"  WARN: Hallmark term not found: {t}")
    gobp = gp.get_library(name="GO_Biological_Process_2023")
    import re
    for patt in GOBP_TERMS_PATTERNS:
        hits = [k for k in gobp if re.search(patt, k, re.IGNORECASE)]
        # Just take the first exact GO term match; sets are typically singletons
        for h in hits[:2]:
            panel[f"GOBP: {h}"] = list(set(gobp[h]))
    return panel


# %% Load AnnData and branches
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

branches = pd.read_csv(BRANCHES_CSV)
print(f"branches.csv: {len(branches)} edges across {branches['patient'].nunique()} patients")

# Migration vs within-tissue tags
branches["edge_kind"] = np.where(branches["src"] == branches["dst"],
                                 "within_tissue", "migration")
branches["edge_label"] = branches["src"].astype(str) + "_to_" + branches["dst"].astype(str)
print("\nEdges per kind:")
print(branches["edge_kind"].value_counts())
print("\nMigration edges per label × lineage:")
print(branches[branches["edge_kind"] == "migration"]
      .groupby(["edge_label", "lineage"]).size().unstack(fill_value=0))


# %% Map cells to edges (source / target)
print("\nMapping cells onto edges (this is the slow step)...")
obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb"]].copy()
obs["cell_idx"] = np.arange(len(obs))
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)

# Build cell-edge participation tables.
# Cell participates as SOURCE in edge if (patient, trb, src tissue, t_src) matches.
src_keys = branches[["patient", "trb", "src", "t_src", "edge_kind", "edge_label", "lineage"]] \
    .rename(columns={"src": "tissue", "t_src": "timepoint"}).drop_duplicates()
dst_keys = branches[["patient", "trb", "dst", "t_dst", "edge_kind", "edge_label", "lineage"]] \
    .rename(columns={"dst": "tissue", "t_dst": "timepoint"}).drop_duplicates()

src_keys["role"] = "source"
dst_keys["role"] = "target"
edge_keys = pd.concat([src_keys, dst_keys], ignore_index=True)

cell_edges = obs.merge(edge_keys, on=["patient", "trb", "tissue", "timepoint"], how="inner",
                       suffixes=("", "_edge"))
# A cell can participate in multiple edges (clone visits multiple
# timepoint pairs); we'll aggregate downstream by edge_label/role.
print(f"  cell-edge memberships: {len(cell_edges)} (unique cells {cell_edges['cell_idx'].nunique()})")
print("\nMemberships per edge_label × role × lineage_edge:")
print(cell_edges.groupby(["edge_label", "role", "lineage"]).size()
      .unstack(fill_value=0).head(30))


# %% Score pathways
print("\nBuilding pathway panel...")
panel = build_pathway_panel()
print(f"  pathways in panel: {len(panel)}")

# Intersect each pathway with adata.var_names
adata_genes = set(adata.var_names)
panel_use = {name: sorted(set(genes) & adata_genes) for name, genes in panel.items()}
panel_use = {n: g for n, g in panel_use.items() if len(g) >= 5}
print(f"  pathways usable (>=5 genes present): {len(panel_use)}")
for n, g in panel_use.items():
    print(f"    {n}  ({len(g)} genes)")

# Use log1p layer
orig_X = adata.X
adata.X = adata.layers["log1p"]

score_cols = []
for name, genes in panel_use.items():
    score_col = f"score__{name}"
    sc.tl.score_genes(adata, gene_list=genes, score_name=score_col,
                      random_state=RANDOM_STATE)
    score_cols.append(score_col)
adata.X = orig_X

scores_df = adata.obs[score_cols].copy()
scores_df["cell_idx"] = np.arange(len(adata.obs))
# Attach scores onto cell_edges via cell_idx
cell_edges = cell_edges.merge(scores_df, on="cell_idx", how="left")


# %% Per (edge_label × role × lineage) mean score
print("\nAggregating per edge_label × role × lineage...")
agg_rows = []
for (edge_label, role, lineage), sub in cell_edges.groupby(
        ["edge_label", "role", "lineage"]):
    if len(sub) < MIN_CELLS_PER_GROUP:
        continue
    for name in panel_use:
        col = f"score__{name}"
        agg_rows.append({
            "edge_label": edge_label, "role": role, "lineage": lineage,
            "pathway": name,
            "n_cells": int(len(sub)), "n_patients": int(sub["patient"].nunique()),
            "mean_score": float(sub[col].mean()),
            "median_score": float(sub[col].median()),
            "std_score": float(sub[col].std()),
            "edge_kind": "migration" if edge_label.split("_to_")[0] != edge_label.split("_to_")[1]
                                       else "within_tissue",
        })
agg = pd.DataFrame(agg_rows)
agg.to_csv(OUT_DIR / "motility_scores_per_edge.csv", index=False)
print(f"  rows: {len(agg)}")


# %% Heatmap: pathway × (edge_label × role) for each lineage
print("\nBuilding per-lineage heatmaps...")
edge_order = ["PBMC_to_PBMC", "PBMC_to_CSF", "PBMC_to_TP",
              "CSF_to_PBMC", "CSF_to_CSF", "CSF_to_TP",
              "TP_to_PBMC", "TP_to_CSF", "TP_to_TP"]
role_order = ["source", "target"]

for lineage in ["CD8", "CD4"]:
    sub = agg[agg["lineage"] == lineage].copy()
    if sub.empty:
        print(f"  skipping {lineage} — no rows")
        continue
    sub["group"] = sub["edge_label"] + " | " + sub["role"]
    group_order = [f"{e} | {r}" for e in edge_order for r in role_order]
    group_order = [g for g in group_order if g in sub["group"].unique()]
    mat = sub.pivot_table(index="pathway", columns="group", values="mean_score")
    mat = mat[group_order]
    # z-score per pathway across groups
    mat_z = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(group_order)),
                                    max(5, 0.32 * len(mat))))
    sns.heatmap(mat_z, cmap="RdBu_r", center=0, vmin=-2.5, vmax=2.5,
                cbar_kws={"label": "z-score (mean log1p score)"},
                linewidths=0.3, linecolor="white", ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_xticklabels(group_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("")
    ax.set_title(f"Motility pathway scores across clonal-traffic edges — {lineage}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"motility_heatmap_{lineage}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"motility_heatmap_{lineage}.pdf", bbox_inches="tight")
    plt.close(fig)


# %% Directional tests: migration source/target vs within-tissue source/persister
# For each pathway and each migration edge label (e.g. CSF_to_TP), compare:
#   (a) migration source cells (CSF half of CSF→TP) vs within-tissue source cells (CSF→CSF)
#       — "are leaving cells primed?"
#   (b) migration target cells (TP half of CSF→TP) vs within-tissue persisters (TP→TP target)
#       — "do arriving cells stay elevated, or look like residents?"
print("\nDirectional tests...")
test_rows = []
mig_edges = [e for e in cell_edges["edge_label"].unique()
             if e.split("_to_")[0] != e.split("_to_")[1]]
within_edges = {t: f"{t}_to_{t}" for t in ["PBMC", "CSF", "TP"]}

for lineage in ["CD8", "CD4"]:
    for me in mig_edges:
        src, dst = me.split("_to_")
        within_src = within_edges.get(src)
        within_dst = within_edges.get(dst)
        if within_src is None or within_dst is None:
            continue
        for pathway in panel_use:
            col = f"score__{pathway}"
            # (a) migration source vs within-tissue source
            a = cell_edges[(cell_edges["edge_label"] == me)
                            & (cell_edges["role"] == "source")
                            & (cell_edges["lineage"] == lineage)][col].values
            b = cell_edges[(cell_edges["edge_label"] == within_src)
                            & (cell_edges["role"] == "source")
                            & (cell_edges["lineage"] == lineage)][col].values
            if len(a) >= MIN_CELLS_PER_GROUP and len(b) >= MIN_CELLS_PER_GROUP:
                try:
                    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    u, p = np.nan, np.nan
                test_rows.append({
                    "lineage": lineage, "contrast": "migration_src_vs_within_src",
                    "migration_edge": me, "within_edge": within_src,
                    "pathway": pathway,
                    "n_a": int(len(a)), "n_b": int(len(b)),
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "mean_diff": float(a.mean() - b.mean()),
                    "mw_p": float(p) if p is not None else np.nan,
                })
            # (b) migration target vs within-tissue (persister) target
            a = cell_edges[(cell_edges["edge_label"] == me)
                            & (cell_edges["role"] == "target")
                            & (cell_edges["lineage"] == lineage)][col].values
            b = cell_edges[(cell_edges["edge_label"] == within_dst)
                            & (cell_edges["role"] == "target")
                            & (cell_edges["lineage"] == lineage)][col].values
            if len(a) >= MIN_CELLS_PER_GROUP and len(b) >= MIN_CELLS_PER_GROUP:
                try:
                    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    u, p = np.nan, np.nan
                test_rows.append({
                    "lineage": lineage, "contrast": "migration_tgt_vs_within_tgt",
                    "migration_edge": me, "within_edge": within_dst,
                    "pathway": pathway,
                    "n_a": int(len(a)), "n_b": int(len(b)),
                    "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                    "mean_diff": float(a.mean() - b.mean()),
                    "mw_p": float(p) if p is not None else np.nan,
                })

tests = pd.DataFrame(test_rows)
# BH FDR across all rows
def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    n = len(pv)
    ranks = np.empty(n); ranks[order] = np.arange(1, n + 1)
    qv = pv * n / ranks
    # monotone correction
    qsorted = qv[order]
    qsorted = np.minimum.accumulate(qsorted[::-1])[::-1]
    out = np.empty(n)
    out[order] = qsorted
    q[valid] = np.clip(out, 0, 1)
    return q


tests["fdr"] = bh_fdr(tests["mw_p"].values)
tests.to_csv(OUT_DIR / "motility_directional_tests.csv", index=False)
print(f"  tests: {len(tests)}")
print(f"  significant (FDR<0.05): {(tests['fdr'] < 0.05).sum()}")


# %% Summary plot: per migration edge, which motility pathways are elevated
print("\nBuilding summary deltas plot...")
mig_summary = (
    tests[tests["contrast"] == "migration_tgt_vs_within_tgt"]
    .groupby(["lineage", "migration_edge", "pathway"], as_index=False)
    .agg(mean_diff=("mean_diff", "mean"),
         min_fdr=("fdr", "min"))
)
for lineage in ["CD8", "CD4"]:
    sub = mig_summary[mig_summary["lineage"] == lineage].copy()
    if sub.empty:
        continue
    sub["edge_order_key"] = sub["migration_edge"]
    mat = sub.pivot_table(index="pathway", columns="migration_edge",
                         values="mean_diff")
    fdr = sub.pivot_table(index="pathway", columns="migration_edge",
                          values="min_fdr")
    edge_present = [e for e in ["PBMC_to_CSF", "PBMC_to_TP",
                                "CSF_to_PBMC", "CSF_to_TP",
                                "TP_to_PBMC", "TP_to_CSF"] if e in mat.columns]
    mat = mat[edge_present]; fdr = fdr[edge_present]

    annot = mat.copy().astype(object)
    for i in annot.index:
        for c in annot.columns:
            d = mat.loc[i, c]; f = fdr.loc[i, c]
            if pd.isna(d):
                annot.loc[i, c] = ""
            else:
                star = ""
                if f < 0.001: star = "***"
                elif f < 0.01: star = "**"
                elif f < 0.05: star = "*"
                annot.loc[i, c] = f"{d:+.2f}{star}"
    vmax = float(np.nanmax(np.abs(mat.values))) if np.isfinite(np.nanmax(np.abs(mat.values))) else 0.01
    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(edge_present) + 4),
                                    max(5, 0.32 * len(mat))))
    sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                annot=annot.values, fmt="",
                cbar_kws={"label": "Δ score (migration target − within-tissue target)"},
                linewidths=0.3, linecolor="white", ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_title(f"Motility pathway: migration arrivers vs within-tissue residents — {lineage}\n"
                 "(* FDR<0.05, ** FDR<0.01, *** FDR<0.001)")
    ax.set_xticklabels(edge_present, rotation=30, ha="right")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"motility_target_vs_resident_{lineage}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# %% Per-pathway PBMC/CSF/TP × edge_kind summary (compact)
print("\nBuilding edge_kind × tissue summary...")
agg2 = (cell_edges.melt(id_vars=["patient", "lineage", "edge_label", "role", "tissue"],
                         value_vars=[f"score__{p}" for p in panel_use],
                         var_name="pathway", value_name="score")
        .assign(pathway=lambda d: d["pathway"].str.replace("score__", "", regex=False),
                edge_kind=lambda d: np.where(d["edge_label"].str.split("_to_").str[0]
                                              == d["edge_label"].str.split("_to_").str[1],
                                              "within_tissue", "migration")))
sum2 = (agg2.groupby(["lineage", "tissue", "edge_kind", "pathway"], as_index=False)
            .agg(n_cells=("score", "size"), mean_score=("score", "mean")))
sum2.to_csv(OUT_DIR / "motility_by_tissue_edgekind.csv", index=False)

# %% Final
print("\n" + "=" * 60)
print("DONE — outputs in", OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
