# %%
"""Distill a clean migration signature from OT outputs across all transitions.

Premise. The previous "leading-edge" migration signature was dominated
by inflammation (NF-κB / TNF / HLA / checkpoint), because the same
clones in tumor are simply more activated than the same clones in CSF
or PBMC. To separate the *act of migrating* from *being in the
destination tissue*, we use a symmetry trick on the OT outputs.

For every gene g and every directed (src, dst) transition, OT gives us
a mass-weighted per-gene Δ:
    Δ_OT(src → dst, g) = (target log1p mean − source log1p mean)
                          on cells matched by clone-constrained Sinkhorn.

Define for each unordered tissue pair {A, B}:
    Δ_AB     = Δ_OT(A → B)
    Δ_BA     = Δ_OT(B → A)

Decompose per-gene:
    tissue_AB(g)   = (Δ_AB − Δ_BA) / 2     # anti-symmetric: tissue context
                                            #   (gene higher in B than A)
    migration_AB(g) = (Δ_AB + Δ_BA) / 2    # symmetric: induced by the act
                                            #   of migrating, same sign both
                                            #   ways

Genes with consistent positive `migration_AB` across multiple unordered
pairs are the migration-act signature — they're induced regardless of
which way the cell is moving. Anti-symmetric components capture pure
tissue-residence biology and are explicitly excluded.

We also report the within-clone, within-tissue pseudobulk Δ for the
same clones (does the gene also change in clones that stay put?). A
true migration gene should change only on cross-tissue events.

Outputs:
  results/pathway_migration_signature_ot/
    pairwise_decomposition.csv      — per-pair (tissue_AB, migration_AB)
    clean_migration_genes.csv       — top symmetric genes
    panel_scores_collapsed_<lin>.png — re-plot the collapsed traffic
                                       graph using the new clean signature
    inflammation_vs_clean_<lin>.png  — direct comparison panel

Usage:
    python pipeline/pathway_migration_signature_ot.py
"""
import sys
import warnings
from itertools import combinations
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
OT_DIR = REPO_ROOT / "results" / "traffic_drainage_rewiring"
OUT_DIR = REPO_ROOT / "results" / "pathway_migration_signature_ot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINEAGES = ["CD8", "CD4"]
TISSUES = ["PBMC", "CSF", "TP"]
TISSUE_XY = {  # CSF top, TP middle-right, PBMC bottom
    "CSF":  (0.0, 1.0),
    "TP":   (0.87, -0.3),
    "PBMC": (0.0, -1.0),
}
TIMEPOINT_STEPS = [(t, t + 1) for t in [1, 2, 3, 4, 5]]
TOP_K_SIGNATURE = 50
MIN_CELLS_PER_NODE = 3
MIN_CLONE_STEPS_PER_EDGE = 5
RANDOM_STATE = 42


def read_rewire(src: str, dst: str, lineage: str) -> pd.DataFrame | None:
    """Read rewiring_genes_fullexpr.csv for one transition × lineage."""
    f = OT_DIR / f"{src}_to_{dst}" / lineage / "rewiring_genes_fullexpr.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df.rename(columns={"gene": "gene", "mean_delta": "mean_delta_fe"})
    df["src"] = src; df["dst"] = dst; df["lineage"] = lineage
    return df


# %% 1. Read all available OT outputs
print("Reading OT outputs across transitions × lineages...")
rewire_long = []
for lineage in LINEAGES:
    for src in TISSUES:
        for dst in TISSUES:
            if src == dst:
                continue
            r = read_rewire(src, dst, lineage)
            if r is None:
                print(f"  MISSING: {src}_to_{dst} / {lineage}")
                continue
            rewire_long.append(r)
            print(f"  loaded  {src}_to_{dst} / {lineage}: {len(r)} genes")
if not rewire_long:
    raise SystemExit(
        "No OT outputs found. Run pipeline/traffic_drainage_rewiring.py first.")
rewire_long = pd.concat(rewire_long, ignore_index=True)
rewire_long.to_csv(OUT_DIR / "rewire_long_all_transitions.csv", index=False)


# %% 2. Pivot to gene × transition table, decompose into symmetric / anti-symmetric
print("\nDecomposing per-pair Δs into tissue_context vs migration_act...")
pair_rows = []
for lineage in LINEAGES:
    sub = rewire_long[rewire_long["lineage"] == lineage]
    if sub.empty:
        continue
    wide = sub.pivot_table(index="gene", columns=["src", "dst"],
                            values="mean_delta_fe")
    for a, b in combinations(TISSUES, 2):
        if (a, b) not in wide.columns or (b, a) not in wide.columns:
            print(f"  {lineage}: pair {a}↔{b} incomplete; skipping")
            continue
        d_ab = wide[(a, b)]
        d_ba = wide[(b, a)]
        for g in wide.index:
            v_ab = d_ab.loc[g]
            v_ba = d_ba.loc[g]
            if pd.isna(v_ab) or pd.isna(v_ba):
                continue
            pair_rows.append({
                "lineage": lineage,
                "tissue_a": a, "tissue_b": b,
                "gene": g,
                "delta_a_to_b": float(v_ab),
                "delta_b_to_a": float(v_ba),
                "tissue_context": float(v_ab - v_ba) / 2,    # gene higher in B
                "migration_act": float(v_ab + v_ba) / 2,      # symmetric
            })
pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(OUT_DIR / "pairwise_decomposition.csv", index=False)
print(f"  decomposition rows: {len(pair_df)}")


# %% 3. Build the clean migration signature
# Strategy: a gene is in the clean migration signature iff:
#   (a) its `migration_act` is positive (>= threshold) in MOST pairs
#       where data is available
#   (b) the per-pair effect is consistent in sign across pairs
# We rank by mean migration_act across pairs, weighted by pair coverage.
print("\nDistilling clean migration signature...")
sig_rows = []
for lineage in LINEAGES:
    sub = pair_df[pair_df["lineage"] == lineage]
    if sub.empty:
        continue
    g = (sub.groupby("gene")
            .agg(mean_act=("migration_act", "mean"),
                 median_act=("migration_act", "median"),
                 n_pairs=("migration_act", "size"),
                 mean_tissue_context=("tissue_context", "mean"),
                 mean_abs_tissue=("tissue_context", lambda s: np.mean(np.abs(s))),
                 sign_consistency=("migration_act",
                                   lambda s: max((s > 0).mean(), (s < 0).mean())))
            .reset_index())
    g["lineage"] = lineage
    sig_rows.append(g)
sig_table = pd.concat(sig_rows, ignore_index=True) if sig_rows else pd.DataFrame()


def _is_tcr_seg(g):
    return str(g).startswith(("TRAV", "TRBV", "TRAJ", "TRBJ", "TRAC", "TRBC",
                              "TRDV", "TRGV"))


# Drop TCR segments + ribosomal + mito (noise)
def _is_noise(g):
    s = str(g)
    return (_is_tcr_seg(s) or s.startswith(("RPS", "RPL", "MT-", "MRPS", "MRPL")))


sig_table = sig_table[~sig_table["gene"].apply(_is_noise)]
sig_table.to_csv(OUT_DIR / "signature_distillation_table.csv", index=False)


# Pick clean migration signature with STRICT filtering:
#  1. n_pairs >= 2 (gene observed across multiple unordered tissue pairs)
#  2. sign_consistency >= 0.66 (mostly same direction across pairs)
#  3. |migration_act| > |tissue_context|       — symmetric component dominates
#  4. specificity_ratio = |mean_act| / (|mean_abs_tissue| + epsilon) > 1
#     — gene is migration-event-driven, not tissue-context-driven
print("\nSelecting top clean migration genes per lineage (strict)...")
EPSILON = 0.1
for lineage in LINEAGES:
    s = sig_table[sig_table["lineage"] == lineage].copy()
    if s.empty:
        continue
    s = s[s["n_pairs"] >= 2]
    s = s[s["sign_consistency"] >= 0.66]
    # Specificity: symmetric magnitude divided by mean *absolute* tissue
    # context (so a gene that swings +5/-5 across tissues is penalized
    # even though its mean tissue context is ~0).
    s["specificity_ratio"] = (
        s["mean_act"].abs() / (s["mean_abs_tissue"] + EPSILON)
    )
    s_strict = s[s["specificity_ratio"] >= 1.0]
    print(f"  {lineage}: {len(s)} candidates → {len(s_strict)} after "
          f"|migration_act| / |tissue_context| >= 1 filter")
    up = s_strict[s_strict["mean_act"] > 0] \
        .sort_values("mean_act", ascending=False).head(TOP_K_SIGNATURE)
    dn = s_strict[s_strict["mean_act"] < 0] \
        .sort_values("mean_act", ascending=True).head(TOP_K_SIGNATURE)
    pd.concat([up.assign(direction="migration_up"),
               dn.assign(direction="migration_down")], ignore_index=True).to_csv(
        OUT_DIR / f"clean_migration_genes_{lineage}.csv", index=False)
    print(f"  {lineage}: {len(up)} migration-up, {len(dn)} migration-down")


# %% 4. Re-score on cells using the new clean signature, plot collapsed graph
print("\nLoading AnnData for signature scoring...")
adata = sc.read(str(paths.H5AD_TCELLS))
orig_X = adata.X
adata.X = adata.layers["log1p"]
inf_panel = []
for lineage in LINEAGES:
    f = OUT_DIR / f"clean_migration_genes_{lineage}.csv"
    if not f.exists():
        continue
    g = pd.read_csv(f)
    up = g[g["direction"] == "migration_up"]["gene"].tolist()
    dn = g[g["direction"] == "migration_down"]["gene"].tolist()
    up_p = [x for x in up if x in adata.var_names]
    dn_p = [x for x in dn if x in adata.var_names]
    print(f"  {lineage}: up_present={len(up_p)} down_present={len(dn_p)}")
    if up_p:
        sc.tl.score_genes(adata, gene_list=up_p,
                          score_name=f"clean_mig_up_{lineage}",
                          random_state=RANDOM_STATE)
    else:
        adata.obs[f"clean_mig_up_{lineage}"] = 0.0
    if dn_p:
        sc.tl.score_genes(adata, gene_list=dn_p,
                          score_name=f"clean_mig_dn_{lineage}",
                          random_state=RANDOM_STATE)
    else:
        adata.obs[f"clean_mig_dn_{lineage}"] = 0.0
    adata.obs[f"clean_mig_net_{lineage}"] = (
        adata.obs[f"clean_mig_up_{lineage}"]
        - adata.obs[f"clean_mig_dn_{lineage}"]
    )

# Inflammation control panel (same as before)
inflammation_panel = [
    "NFKBIA", "NFKBIB", "NFKBIE", "NFKBIZ", "NFKB1", "NFKB2",
    "RELA", "RELB", "TNFAIP3", "TNF", "TRAF1", "TRAF2", "TNFRSF1A",
    "IRF1", "STAT1", "GBP1", "GBP5",
    "HLA-A", "HLA-B", "HLA-C", "HLA-E", "B2M", "TAP1", "TAP2",
    "HLA-DRA", "HLA-DRB1",
    "CTLA4", "PDCD1", "LAG3", "TIGIT", "HAVCR2", "ICOS",
]
inflammation_present = [g for g in inflammation_panel if g in adata.var_names]
sc.tl.score_genes(adata, gene_list=inflammation_present,
                  score_name="inflammation", random_state=RANDOM_STATE)
adata.X = orig_X


# %% 5. Per (clone × node) means, then collapsed Δ per edge
print("\nAggregating per (patient, clone, tissue, timepoint)...")
score_cols = (
    [f"clean_mig_net_{lin}" for lin in LINEAGES]
    + ["inflammation"]
)
obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb"] + score_cols].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)

clone_node = (
    obs.groupby(["patient", "trb", "lineage", "tissue", "timepoint"], observed=True)
       .agg(n_cells=("inflammation", "size"),
            score_clean_cd8=("clean_mig_net_CD8", "mean"),
            score_clean_cd4=("clean_mig_net_CD4", "mean"),
            score_inflammation=("inflammation", "mean"))
       .reset_index()
)
clone_node = clone_node[clone_node["n_cells"] >= MIN_CELLS_PER_NODE]
clone_node.to_csv(OUT_DIR / "clone_node_scores.csv", index=False)


# %% 6. Collapsed-edge deltas
print("Computing collapsed-edge deltas...")
edge_rows = []
for lineage in LINEAGES:
    cn = clone_node[clone_node["lineage"] == lineage]
    score_clean = f"score_clean_cd8" if lineage == "CD8" else "score_clean_cd4"
    for src in TISSUES:
        for dst in TISSUES:
            paired_clean, paired_inf = [], []
            n_patients_set = set()
            for t_s, t_d in TIMEPOINT_STEPS:
                src_n = cn[(cn["tissue"] == src) & (cn["timepoint"] == t_s)]
                dst_n = cn[(cn["tissue"] == dst) & (cn["timepoint"] == t_d)]
                merged = src_n.merge(dst_n, on=["patient", "trb"],
                                     suffixes=("_s", "_d"))
                if merged.empty:
                    continue
                paired_clean.extend(
                    (merged[f"{score_clean}_d"] - merged[f"{score_clean}_s"]).tolist())
                paired_inf.extend(
                    (merged["score_inflammation_d"] - merged["score_inflammation_s"]).tolist())
                n_patients_set.update(merged["patient"].tolist())
            if len(paired_clean) < MIN_CLONE_STEPS_PER_EDGE:
                continue
            row = {"lineage": lineage,
                   "src_tissue": src, "dst_tissue": dst,
                   "n_clone_steps": int(len(paired_clean)),
                   "n_patients": int(len(n_patients_set))}
            for name, arr in [("clean", paired_clean), ("inf", paired_inf)]:
                a = np.asarray(arr)
                try:
                    wstat, wp = stats.wilcoxon(a)
                except ValueError:
                    wstat, wp = np.nan, np.nan
                row[f"mean_delta_{name}"] = float(a.mean())
                row[f"wilcoxon_p_{name}"] = float(wp) if wp is not None else np.nan
            edge_rows.append(row)
edges = pd.DataFrame(edge_rows)
edges.to_csv(OUT_DIR / "collapsed_edges.csv", index=False)


# %% 7. Plot
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


for lineage in LINEAGES:
    sub = edges[edges["lineage"] == lineage]
    if sub.empty:
        continue
    node_n = (clone_node[clone_node["lineage"] == lineage]
                .groupby("tissue")["n_cells"].sum().to_dict())
    # Single-panel "clean signature only" plot
    vmax_clean = max(0.02,
                      float(np.nanmax(np.abs(sub["mean_delta_clean"].values))))
    fig, ax = plt.subplots(figsize=(8, 7))
    sm = draw_collapsed_graph(
        ax, sub, "mean_delta_clean", "wilcoxon_p_clean",
        title=f"{lineage} — OT-distilled clean migration signature\n"
              "per-clone Δ along forward traffic edges (T_n → T_n+1, collapsed)",
        node_size_by_tissue=node_n, vmax=vmax_clean,
    )
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("mean Δ clean migration score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"clean_migration_{lineage}.png",
                dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"clean_migration_{lineage}.pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote clean_migration_{lineage}.png")

    # Side-by-side with inflammation (kept for reference)
    vmax = max(
        0.02,
        float(np.nanmax(np.abs(sub["mean_delta_clean"].values))),
        float(np.nanmax(np.abs(sub["mean_delta_inf"].values))),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    sm = draw_collapsed_graph(
        axes[0], sub, "mean_delta_clean", "wilcoxon_p_clean",
        title=f"{lineage} — OT-distilled migration signature",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    draw_collapsed_graph(
        axes[1], sub, "mean_delta_inf", "wilcoxon_p_inf",
        title=f"{lineage} — Inflammation panel (for reference)",
        node_size_by_tissue=node_n, vmax=vmax,
    )
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02,
                        location="right", shrink=0.8)
    cbar.set_label("mean Δ score (target − source)")
    fig.savefig(OUT_DIR / f"clean_migration_vs_inflammation_{lineage}.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# %% Final
print("\n" + "=" * 60)
print("DONE — outputs in", OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
