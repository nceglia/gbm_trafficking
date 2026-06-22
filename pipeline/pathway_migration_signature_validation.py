# %%
"""Validate the two-component story of the OT-distilled migration signature.

Hypothesis from pathway enrichment of the symmetric (migration_act)
component:

  A. "Effector / broadcaster" sub-signature (UP on migration_act):
        secreted chemokines + MHC class I + immunoproteasome + granule
        markers. T cells crossing tissue boundaries upregulate the
        cytotoxic / chemokine-output program, symmetrically with
        respect to direction.

  B. "Cytoskeletal scaffold turnover" sub-signature (DOWN on
     migration_act): actin-membrane crosslinkers (MSN, LCP1, ACTN4),
     Rho/Rac GEFs (DOCK10, ARAP2), GTPase regulators (IQGAP1,
     CDC42SE2), microtubule/membrane (TUBB, DNM2, NCKAP1L). The cell
     remodels steady-state scaffolds during active migration.

If this story is right, we should see four things:

  (1) Per-edge collapsed graph: A goes UP and B goes DOWN on migration
      edges (both into and out of tumor); within-tissue self-loops are
      flat or weakly opposite.

  (2) Joint per-cell (A, B) plot: cells in different tissues separate
      along both axes; cells in transit (clones with multi-tissue
      presence) sit between resident-cohort centroids.

  (3) Longitudinal decay in TP: recently-entered TP cells
      (entry_w_prior_circ from pathway_entry_signature) should have
      LOWER scaffold-B score than long persisters — i.e. the scaffold
      machinery recovers over time inside tumor.

  (4) Per-gene per-pair consistency heatmap: each individual gene we
      put in A or B should show migration_act of the predicted sign in
      all three unordered tissue pairs (PBMC↔CSF, PBMC↔TP, CSF↔TP).

Usage:
    python pipeline/pathway_migration_signature_validation.py
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
OT_SIG_DIR = REPO_ROOT / "results" / "pathway_migration_signature_ot"
ENTRY_DIR = REPO_ROOT / "results" / "pathway_entry_signature"
OUT_DIR = OT_SIG_DIR  # write into same dir
TISSUE_XY = {"CSF": (0.0, 1.0), "TP": (0.87, -0.3), "PBMC": (0.0, -1.0)}
TISSUES = list(TISSUE_XY.keys())
TIMEPOINT_STEPS = [(t, t + 1) for t in [1, 2, 3, 4, 5]]
MIN_CELLS_PER_NODE = 3
MIN_CLONE_STEPS_PER_EDGE = 5
RANDOM_STATE = 42


# --- Manual categorization of the CD8 clean signature genes ---
# (derived from clean_migration_genes_CD8.csv — top 50 each direction,
# bucketed by biological function from the gene-level inspection)
EFFECTOR_BROADCAST = [
    # secreted/effector chemokines and granule
    "CCL5", "CST7",
    # MHC class I / antigen presentation
    "HLA-A", "HLA-C", "HLA-E", "PSMB9",
    # TCR / costimulatory
    "CD3E", "TNFRSF14",
    # cytotoxic / metabolism support
    "GPX4", "FTL", "ITM2A",
]
SCAFFOLD_TURNOVER = [
    # actin-membrane crosslinkers and bundlers
    "MSN", "LCP1", "ACTN4", "TPM4",
    # Rho/Rac/CDC42 GEFs and regulators
    "DOCK10", "ARAP2", "IQGAP1", "CDC42SE2", "NCKAP1L",
    # ARP2/3 actin nucleation
    "ACTR2",
    # microtubule / membrane dynamics
    "TUBB", "DNM2",
    # other cytoskeletal/regulatory
    "PKN2", "ITSN2",
]


# %% Load adata + score the two sub-signatures
print("Loading AnnData and scoring sub-signatures...")
adata = sc.read(str(paths.H5AD_TCELLS))
orig_X = adata.X
adata.X = adata.layers["log1p"]

eff_present = [g for g in EFFECTOR_BROADCAST if g in adata.var_names]
sca_present = [g for g in SCAFFOLD_TURNOVER if g in adata.var_names]
print(f"  effector/broadcast: {len(eff_present)}/{len(EFFECTOR_BROADCAST)} present")
print(f"  scaffold/turnover:  {len(sca_present)}/{len(SCAFFOLD_TURNOVER)} present")
sc.tl.score_genes(adata, gene_list=eff_present,
                  score_name="effector_score", random_state=RANDOM_STATE)
sc.tl.score_genes(adata, gene_list=sca_present,
                  score_name="scaffold_score", random_state=RANDOM_STATE)
adata.X = orig_X


# %% Per (patient, clone, tissue, timepoint) means + lineage filter
print("\nAggregating per (patient, clone, tissue, timepoint)...")
obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb",
                  "effector_score", "scaffold_score"]].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)
# Restrict everything to CD8 for the headline validation; CD4 is parallel
obs_cd8 = obs[obs["lineage"] == "CD8"].copy()

clone_node = (
    obs_cd8.groupby(["patient", "trb", "tissue", "timepoint"], observed=True)
       .agg(n_cells=("effector_score", "size"),
            eff=("effector_score", "mean"),
            sca=("scaffold_score", "mean"))
       .reset_index()
)
clone_node = clone_node[clone_node["n_cells"] >= MIN_CELLS_PER_NODE]


# %% (1) Collapsed-edge Δ for both sub-signatures
print("\n[1/4] Collapsed-edge deltas for both sub-signatures...")
edge_rows = []
for src in TISSUES:
    for dst in TISSUES:
        eff_d, sca_d = [], []
        for t_s, t_d in TIMEPOINT_STEPS:
            src_n = clone_node[(clone_node["tissue"] == src)
                                & (clone_node["timepoint"] == t_s)]
            dst_n = clone_node[(clone_node["tissue"] == dst)
                                & (clone_node["timepoint"] == t_d)]
            m = src_n.merge(dst_n, on=["patient", "trb"],
                            suffixes=("_s", "_d"))
            if m.empty:
                continue
            eff_d.extend((m["eff_d"] - m["eff_s"]).tolist())
            sca_d.extend((m["sca_d"] - m["sca_s"]).tolist())
        if len(eff_d) < MIN_CLONE_STEPS_PER_EDGE:
            continue
        try:
            _, pe = stats.wilcoxon(eff_d)
        except ValueError:
            pe = np.nan
        try:
            _, ps = stats.wilcoxon(sca_d)
        except ValueError:
            ps = np.nan
        edge_rows.append({"src_tissue": src, "dst_tissue": dst,
                          "n_clone_steps": int(len(eff_d)),
                          "mean_delta_eff": float(np.mean(eff_d)),
                          "wilcoxon_p_eff": float(pe),
                          "mean_delta_sca": float(np.mean(sca_d)),
                          "wilcoxon_p_sca": float(ps)})
edges = pd.DataFrame(edge_rows)
edges.to_csv(OUT_DIR / "validation_edges_CD8.csv", index=False)


# --- Helpers (re-using draw_collapsed_graph from the earlier scripts) ---
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


node_n = clone_node.groupby("tissue")["n_cells"].sum().to_dict()
vmax = max(0.02,
            float(np.nanmax(np.abs(edges["mean_delta_eff"].values))),
            float(np.nanmax(np.abs(edges["mean_delta_sca"].values))))
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
sm = draw_collapsed_graph(
    axes[0], edges, "mean_delta_eff", "wilcoxon_p_eff",
    title=f"CD8 — Effector/broadcast sub-signature\n"
          f"({len(eff_present)} genes: CCL5, CST7, HLA-A/C/E, PSMB9, …)",
    node_size_by_tissue=node_n, vmax=vmax,
)
draw_collapsed_graph(
    axes[1], edges, "mean_delta_sca", "wilcoxon_p_sca",
    title=f"CD8 — Scaffold turnover sub-signature\n"
          f"({len(sca_present)} genes: MSN, LCP1, IQGAP1, DOCK10, TPM4, …)",
    node_size_by_tissue=node_n, vmax=vmax,
)
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02,
                    location="right", shrink=0.8)
cbar.set_label("mean Δ score (target − source)")
fig.suptitle("CD8 sub-signatures on collapsed traffic edges — "
              "validation: effector goes UP, scaffold goes DOWN",
              y=1.02, fontsize=12)
fig.savefig(OUT_DIR / "validation_panel1_edges.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "validation_panel1_edges.pdf", bbox_inches="tight")
plt.close(fig)
print("  wrote validation_panel1_edges.png")


# %% (2) Joint per-cell (effector, scaffold) density, colored by status
print("\n[2/4] Joint per-cell (effector, scaffold) plot...")
# Use the entry labels from pathway_entry_signature if available
entry_labels_path = ENTRY_DIR / "tp_class_labels.csv"
status_lookup = pd.DataFrame()
if entry_labels_path.exists():
    status_lookup = pd.read_csv(entry_labels_path)
    status_lookup["timepoint"] = status_lookup["timepoint"].astype(int)


def assign_status(row):
    if row["tissue"] != "TP":
        return row["tissue"]   # PBMC or CSF as is
    # TP cell — look up tp_class
    return row.get("tp_class", "TP_unlabeled") or "TP_unlabeled"


obs_cd8 = obs_cd8.merge(status_lookup, on=["patient", "trb", "timepoint"], how="left")
obs_cd8["status"] = obs_cd8.apply(assign_status, axis=1)
obs_cd8["status"] = obs_cd8["status"].fillna("unlabeled")

STATUS_ORDER = ["PBMC", "CSF", "entry_w_prior_circ",
                "persister_short", "persister_long"]
STATUS_COLORS = {
    "PBMC": "#b2182b",
    "CSF":  "#2166ac",
    "entry_w_prior_circ": "#d62728",
    "persister_short": "#9ecae1",
    "persister_long":  "#08519c",
}

plot_cells = obs_cd8[obs_cd8["status"].isin(STATUS_ORDER)].copy()
# Subsample for visualization
samp_n = min(15000, len(plot_cells))
plot_cells_samp = plot_cells.sample(samp_n, random_state=42)

fig, ax = plt.subplots(figsize=(8, 7))
# Per-status KDE contours on top of background scatter
ax.scatter(plot_cells_samp["scaffold_score"],
           plot_cells_samp["effector_score"],
           s=1.5, c="lightgray", alpha=0.25, zorder=1)
for st in STATUS_ORDER:
    if st not in plot_cells["status"].unique():
        continue
    sub = plot_cells[plot_cells["status"] == st]
    if len(sub) < 50:
        continue
    # KDE level set
    sns.kdeplot(data=sub.sample(min(5000, len(sub)), random_state=42),
                x="scaffold_score", y="effector_score",
                color=STATUS_COLORS[st], levels=[0.3, 0.6],
                linewidths=2, ax=ax, alpha=0.95, zorder=3)
    cx = sub["scaffold_score"].median()
    cy = sub["effector_score"].median()
    ax.scatter([cx], [cy], s=120, marker="o", color=STATUS_COLORS[st],
               ec="black", lw=1.4, zorder=5,
               label=f"{st} (n={len(sub):,})")
ax.set_xlabel("Scaffold turnover score (MSN, LCP1, IQGAP1, DOCK10, …)")
ax.set_ylabel("Effector / broadcast score (CCL5, CST7, HLA class I, …)")
ax.set_title("Per-cell joint distribution: where does each status sit?\n"
              "Prediction: cells in transit (entry) should have LOW scaffold + "
              "moderate-to-high effector")
ax.legend(loc="lower right", frameon=True, fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "validation_panel2_joint.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "validation_panel2_joint.pdf", bbox_inches="tight")
plt.close(fig)
print("  wrote validation_panel2_joint.png")


# %% (3) Score across TP entry → persister stratification
print("\n[3/4] Longitudinal decay across entry classes within TP...")
tp_cells = obs_cd8[(obs_cd8["tissue"] == "TP")
                   & obs_cd8["status"].isin(["entry_w_prior_circ",
                                              "persister_short",
                                              "persister_long"])].copy()
status_order_tp = ["entry_w_prior_circ", "persister_short", "persister_long"]
tp_cells["status"] = pd.Categorical(tp_cells["status"],
                                     categories=status_order_tp, ordered=True)

# Per-patient means → patient-level paired test
patient_means = (tp_cells.groupby(["patient", "status"], observed=True)
                  .agg(eff=("effector_score", "mean"),
                       sca=("scaffold_score", "mean"),
                       n_cells=("effector_score", "size"))
                  .reset_index())
patient_means = patient_means[patient_means["n_cells"] >= 20]
patient_means.to_csv(OUT_DIR / "validation_panel3_patient_means.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
for ax, score_col, score_label, expected in [
    (axes[0], "effector_score", "Effector / broadcast score",
     "expected: high in entry → decays as cell persists"),
    (axes[1], "scaffold_score", "Scaffold turnover score",
     "expected: low in entry → recovers as cell persists"),
]:
    sns.boxplot(data=tp_cells, x="status", y=score_col,
                order=status_order_tp,
                palette={k: STATUS_COLORS[k] for k in status_order_tp},
                showfliers=False, ax=ax, width=0.55)
    sns.stripplot(data=patient_means.assign(
        sc=lambda d: d[score_col.split("_")[0][:3] if "effector" in score_col else "sca"]
                       if False else d.get("eff" if "effector" in score_col else "sca", np.nan)),
                  x="status", y="eff" if "effector" in score_col else "sca",
                  order=status_order_tp, hue="patient",
                  palette="Dark2", size=8, ax=ax,
                  edgecolor="black", linewidth=0.5)
    ax.set_xlabel(""); ax.set_ylabel(score_label)
    ax.set_title(score_label + "\n" + expected, fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False, fontsize=7)
    ax.tick_params(axis="x", labelrotation=15)
fig.suptitle("CD8 TP cells stratified by entry status (longitudinal)",
              y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "validation_panel3_entry_decay.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "validation_panel3_entry_decay.pdf", bbox_inches="tight")
plt.close(fig)
print("  wrote validation_panel3_entry_decay.png")

# Stats: per-patient paired Wilcoxon entry vs persister_long
rows = []
wide = patient_means.pivot(index="patient", columns="status",
                            values="eff")
if "entry_w_prior_circ" in wide.columns and "persister_long" in wide.columns:
    pairs = wide[["entry_w_prior_circ", "persister_long"]].dropna()
    if len(pairs) >= 3:
        s, p = stats.wilcoxon(pairs["entry_w_prior_circ"], pairs["persister_long"])
        rows.append({"score": "effector", "n_patients": len(pairs),
                     "mean_entry": float(pairs["entry_w_prior_circ"].mean()),
                     "mean_pers_long": float(pairs["persister_long"].mean()),
                     "mean_diff": float((pairs["entry_w_prior_circ"]
                                          - pairs["persister_long"]).mean()),
                     "wilcoxon_p": float(p)})
wide = patient_means.pivot(index="patient", columns="status", values="sca")
if "entry_w_prior_circ" in wide.columns and "persister_long" in wide.columns:
    pairs = wide[["entry_w_prior_circ", "persister_long"]].dropna()
    if len(pairs) >= 3:
        s, p = stats.wilcoxon(pairs["entry_w_prior_circ"], pairs["persister_long"])
        rows.append({"score": "scaffold", "n_patients": len(pairs),
                     "mean_entry": float(pairs["entry_w_prior_circ"].mean()),
                     "mean_pers_long": float(pairs["persister_long"].mean()),
                     "mean_diff": float((pairs["entry_w_prior_circ"]
                                          - pairs["persister_long"]).mean()),
                     "wilcoxon_p": float(p)})
pd.DataFrame(rows).to_csv(OUT_DIR / "validation_panel3_paired_tests.csv", index=False)


# %% (4) Per-gene per-pair consistency heatmap
print("\n[4/4] Per-gene per-pair migration_act heatmap...")
pair_df = pd.read_csv(OT_SIG_DIR / "pairwise_decomposition.csv")
pair_cd8 = pair_df[pair_df["lineage"] == "CD8"]
# Pivot: rows = gene, cols = (tissue_a, tissue_b), values = migration_act
piv = pair_cd8.pivot_table(index="gene",
                            columns=["tissue_a", "tissue_b"],
                            values="migration_act")
gene_order = (
    [g for g in eff_present if g in piv.index]
    + [g for g in sca_present if g in piv.index]
)
gene_labels = (
    [f"[E] {g}" for g in eff_present if g in piv.index]
    + [f"[S] {g}" for g in sca_present if g in piv.index]
)
mat = piv.loc[gene_order]

# Also annotate tissue_context magnitude as a marker on the right
ctx = pair_cd8.pivot_table(index="gene", columns=["tissue_a", "tissue_b"],
                            values="tissue_context").loc[gene_order]

vmax = max(0.02, float(np.nanmax(np.abs(mat.values))))
fig, axes = plt.subplots(1, 2, figsize=(8, 0.28 * len(mat) + 1.5),
                          gridspec_kw={"width_ratios": [3, 3]})
sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
             cbar_kws={"label": "migration_act (symmetric Δ)"},
             linewidths=0.3, linecolor="white", ax=axes[0],
             xticklabels=[f"{a}↔{b}" for a, b in mat.columns],
             yticklabels=gene_labels)
axes[0].set_title("Symmetric migration_act per gene × tissue pair",
                   fontsize=10)
axes[0].set_xlabel(""); axes[0].set_ylabel("")

vmax2 = max(0.02, float(np.nanmax(np.abs(ctx.values))))
sns.heatmap(ctx, cmap="PuOr_r", center=0, vmin=-vmax2, vmax=vmax2,
             cbar_kws={"label": "tissue_context (anti-symmetric Δ)"},
             linewidths=0.3, linecolor="white", ax=axes[1],
             xticklabels=[f"{a}↔{b}" for a, b in ctx.columns],
             yticklabels=False)
axes[1].set_title("Anti-symmetric tissue_context per gene × pair",
                   fontsize=10)
axes[1].set_xlabel(""); axes[1].set_ylabel("")
fig.suptitle(
    "CD8 sub-signature genes: symmetric vs anti-symmetric components\n"
    "[E] = effector/broadcast (predicted symmetric UP);   "
    "[S] = scaffold turnover (predicted symmetric DOWN)",
    y=1.005, fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT_DIR / "validation_panel4_gene_consistency.png",
            dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "validation_panel4_gene_consistency.pdf",
            bbox_inches="tight")
plt.close(fig)
print("  wrote validation_panel4_gene_consistency.png")


# %% Final
print("\n" + "=" * 60)
print("DONE — validation outputs in", OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.glob("validation_panel*")):
    print(f"  {f.name}")
