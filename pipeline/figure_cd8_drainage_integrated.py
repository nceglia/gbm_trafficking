# %%
"""CD8 drainage CSF -> TP: phenotype sankey + per-ribbon gene delta heatmap.

Re-runs per-event OT (PCA on HVGs) to attribute transport mass to
(P_src, P_dst) ribbons, then plots a phenotype sankey above and the
per-ribbon gene-level delta heatmap below.

Reads:
  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad
  results/traffic_archetype_graphs/clone_archetypes.csv
  results/traffic_drainage_rewiring/CSF_to_TP/CD8/rewiring_genes_fullexpr.csv
  results/traffic_bayesian_sankey/T_CD8_CSF_to_TP.csv

Writes to results/figure_cd8_drainage_integrated/:
  events_meta.csv
  ribbon_summary.csv
  ribbon_gene_deltas.csv
  top_genes.csv
  bayesian_sankey_csf_tp.csv
  main.png / main.pdf
"""
import re
import sys
import time
import warnings
from pathlib import Path

import decoupler as dc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
)

CLONE_ARCH_CSV = paths.TRAFFIC_ARCHETYPE_GRAPHS_DIR / "clone_archetypes.csv"
DRAINING_DELTA_CSV = (paths.RESULTS_DIR / "traffic_drainage_rewiring"
                       / "CSF_to_TP" / "CD8"
                       / "rewiring_genes_fullexpr.csv")
BAYES_T_CSV = paths.BAYESIAN_SANKEY_DIR / "T_CD8_CSF_to_TP.csv"
OUT_DIR = paths.RESULTS_DIR / "figure_cd8_drainage_integrated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_HVG = 2000
N_PCS = 50
SINKHORN_EPS = 0.1
MIN_CELLS_PER_SIDE = 2
N_TOP_GENES = 20      # per direction; 20 + 20 = 40
N_TOP_RIBBONS = 6
RIBBON_MIN_PAIRS = 10
DPI = 200
RNG_SEED = 42

EDGE_RE = re.compile(r"^([A-Z]+)@T(\d+)->([A-Z]+)@T(\d+)$")

# Short <-> full phenotype name mapping (CD8 only here)
SHORT_TO_FULL = {}
for full in TCELL_PHENOTYPE_ORDER:
    short = (full.replace("CD8_Activated_", "")
                 .replace("CD8_Quiescent_", "")
                 .replace("CD4_", ""))
    SHORT_TO_FULL[short] = full
FULL_TO_SHORT = {v: k for k, v in SHORT_TO_FULL.items()}

CD8_SHORTS = ["Memory", "Naive", "TEMRA", "TEXeff", "TEXprog", "TEXterm", "TRM"]


def pheno_color(short):
    return TCELL_PHENOTYPE_COLORS.get(SHORT_TO_FULL.get(short, short),
                                       "#9e9e9e")


def pheno_label(short):
    return TCELL_PHENOTYPE_LABELS.get(SHORT_TO_FULL.get(short, short),
                                       short)


# %%
# =========================================================
# Load adata, clones, CD8 cells
# =========================================================
print(f"Loading {paths.H5AD_TCELLS.name}...")
adata = sc.read(str(paths.H5AD_TCELLS))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

obs = adata.obs[["trb", "tissue", "timepoint", "phenotype", "patient"]].copy()
obs = obs[obs["trb"].notna() & (obs["trb"].astype(str) != "")]
for c in ("trb", "tissue", "timepoint", "phenotype", "patient"):
    obs[c] = obs[c].astype(str)
obs["clone_id"] = obs["patient"] + "|" + obs["trb"]
obs["lineage"] = np.where(obs["phenotype"].str.contains("CD8"), "CD8", "CD4")
obs_full = adata.obs.copy()
obs_full["cell_index"] = np.arange(adata.n_obs)
obs = obs.join(obs_full["cell_index"], how="left")
obs_cd8 = obs[obs["lineage"] == "CD8"].copy()
obs_cd8["tp_int"] = obs_cd8["timepoint"].astype(int)
print(f"CD8 cells with TRB: {len(obs_cd8):,}")

clones = pd.read_csv(CLONE_ARCH_CSV)
clones = clones[clones["lineage"] == "CD8"]
print(f"CD8 clones (archetype table): {len(clones)}")


# %%
# =========================================================
# Draining clones + candidate events (CD8) — mirrors traffic_draining_ot
# =========================================================
def parse_edges(s):
    if not isinstance(s, str) or not s.strip():
        return []
    out = []
    for tok in s.strip().split():
        m = EDGE_RE.match(tok)
        if m:
            t1, tp1, t2, tp2 = m.groups()
            out.append((t1, int(tp1), t2, int(tp2)))
    return out


draining_per_clone = {}
for _, row in clones.iterrows():
    cid = row["clone_id"]
    drains = [(tp1, tp2) for (t1, tp1, t2, tp2) in
              parse_edges(row.get("retained_graph_edges", ""))
              if t1 == "CSF" and t2 == "TP"]
    if drains:
        draining_per_clone[cid] = drains

n_drain = len(draining_per_clone)
print(f"\nDraining clones: {n_drain}")

obs_drain = obs_cd8[obs_cd8["clone_id"].isin(draining_per_clone)]
raw_events = []
for cid, drains in draining_per_clone.items():
    for (tp1, tp2) in drains:
        if tp1 < tp2:
            raw_events.append((cid, tp1, tp2))
filtered_events = []
for (cid, t_csf, t_tp) in raw_events:
    sub = obs_drain[obs_drain["clone_id"] == cid]
    prior_tp = sub[(sub["tissue"] == "TP") & (sub["tp_int"] < t_csf)]
    if len(prior_tp) == 0:
        filtered_events.append((cid, t_csf, t_tp))
print(f"Candidate events: raw={len(raw_events)}, filtered={len(filtered_events)}")

cells_map = (obs_cd8.groupby(["clone_id", "tissue", "tp_int"])["cell_index"]
             .apply(lambda s: np.asarray(s.values, dtype=np.int64)))
events_ok = []
for (cid, t_csf, t_tp) in filtered_events:
    s = cells_map.get((cid, "CSF", t_csf), np.array([], dtype=np.int64))
    t = cells_map.get((cid, "TP",  t_tp),  np.array([], dtype=np.int64))
    if len(s) >= MIN_CELLS_PER_SIDE and len(t) >= MIN_CELLS_PER_SIDE:
        events_ok.append((cid, t_csf, t_tp, s, t))
print(f"Events OT-eligible: {len(events_ok)}")


# %%
# =========================================================
# PCA-HVG geometry on CD8
# =========================================================
print("\nBuilding PCA geometry...")
adata_cd8 = adata[obs_cd8["cell_index"].values].copy()
try:
    sc.pp.highly_variable_genes(adata_cd8, n_top_genes=N_HVG,
                                 flavor="seurat_v3", layer="counts",
                                 subset=False)
except Exception:
    adata_cd8.X = adata_cd8.layers["log1p"].copy()
    sc.pp.highly_variable_genes(adata_cd8, n_top_genes=N_HVG,
                                 flavor="seurat", subset=False)
hvg_mask = adata_cd8.var["highly_variable"].values
adata_hvg = adata_cd8[:, hvg_mask].copy()
adata_hvg.X = adata_hvg.layers["log1p"].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=N_PCS, random_state=RNG_SEED)
PC = adata_hvg.obsm["X_pca"].astype(np.float32)
PC = (PC - PC.mean(0)) / (PC.std(0) + 1e-8)
global_to_local = {int(g): i for i, g in
                   enumerate(obs_cd8["cell_index"].values)}
print(f"  PC shape: {PC.shape}")


# %%
# =========================================================
# Per-event OT + ribbon-resolved aggregation
# =========================================================
log1p_X = adata.layers["log1p"]
n_genes = adata.n_vars
gene_names = np.asarray(adata.var_names)
pheno_arr = adata.obs["phenotype"].astype(str).to_numpy()


def cosine_cost(A, B):
    an = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (1.0 - an @ bn.T).astype(np.float32)


def densify(idx):
    rows = log1p_X[idx]
    if sp.issparse(rows):
        rows = rows.toarray()
    return rows.astype(np.float32)


def short_pheno(full):
    return FULL_TO_SHORT.get(full, full)


# Accumulators
ribbon_mass = {}
ribbon_npairs = {}
ribbon_cost = {}
ribbon_gene = {}   # (s,d) -> ndarray (n_genes,)
events_meta_rows = []

print(f"\nRunning OT on {len(events_ok)} events...")
for e_idx, (cid, t_csf, t_tp, s_g, t_g) in enumerate(tqdm(events_ok,
                                                            desc="OT")):
    A = PC[[global_to_local[int(g)] for g in s_g]]
    B = PC[[global_to_local[int(g)] for g in t_g]]
    n_s, n_t = A.shape[0], B.shape[0]
    m = float(min(n_s, n_t))
    a = np.full(n_s, m / n_s, dtype=np.float32)
    b = np.full(n_t, m / n_t, dtype=np.float32)
    C = cosine_cost(A, B)
    pi = np.asarray(ot.sinkhorn(a, b, C, reg=SINKHORN_EPS, numItermax=500,
                                 stopThr=1e-7), dtype=np.float32)
    mass = float(pi.sum())
    mean_cost = float((pi * C).sum() / max(mass, 1e-12))

    src_ph = np.array([short_pheno(pheno_arr[int(g)]) for g in s_g])
    tgt_ph = np.array([short_pheno(pheno_arr[int(g)]) for g in t_g])
    expr_s = densify(s_g)
    expr_t = densify(t_g)

    events_meta_rows.append({
        "event_id": e_idx, "clone_id": cid, "t_csf": t_csf, "t_tp": t_tp,
        "n_src": n_s, "n_tgt": n_t, "mass": mass, "mean_cost": mean_cost,
    })

    # Accumulate per-ribbon contributions
    # Group source rows by phenotype and target rows by phenotype.
    uniq_s = pd.unique(src_ph)
    uniq_t = pd.unique(tgt_ph)
    for ps in uniq_s:
        i_mask = (src_ph == ps)
        for pd_ in uniq_t:
            j_mask = (tgt_ph == pd_)
            sub_pi = pi[np.ix_(i_mask, j_mask)]
            sub_C  = C [np.ix_(i_mask, j_mask)]
            sub_mass = float(sub_pi.sum())
            if sub_mass <= 0:
                continue
            n_pairs = int(sub_pi.size)
            cost_sum = float((sub_pi * sub_C).sum())
            # gene delta: row_sum @ expr_s (over masked rows),
            # col_sum @ expr_t (over masked cols)
            row_sum = sub_pi.sum(axis=1)
            col_sum = sub_pi.sum(axis=0)
            tgt_contrib = col_sum @ expr_t[j_mask]
            src_contrib = row_sum @ expr_s[i_mask]
            delta = tgt_contrib - src_contrib

            key = (ps, pd_)
            ribbon_mass[key] = ribbon_mass.get(key, 0.0) + sub_mass
            ribbon_npairs[key] = ribbon_npairs.get(key, 0) + n_pairs
            ribbon_cost[key] = ribbon_cost.get(key, 0.0) + cost_sum
            if key not in ribbon_gene:
                ribbon_gene[key] = np.zeros(n_genes, dtype=np.float64)
            ribbon_gene[key] += delta

print(f"\nRibbons total: {len(ribbon_mass)}")
ribbons_kept_mask = {k: v >= RIBBON_MIN_PAIRS
                     for k, v in ribbon_npairs.items()}
n_kept = sum(ribbons_kept_mask.values())
print(f"Ribbons with n_pairs >= {RIBBON_MIN_PAIRS}: {n_kept}")


# %%
# =========================================================
# Save intermediate CSVs
# =========================================================
events_meta_df = pd.DataFrame(events_meta_rows)
events_meta_df.to_csv(OUT_DIR / "events_meta.csv", index=False)
print(f"Saved: events_meta.csv ({len(events_meta_df)} rows)")

ribbon_rows = []
for k in ribbon_mass:
    mass = ribbon_mass[k]
    ribbon_rows.append({
        "P_src": k[0], "P_dst": k[1], "mass": mass,
        "n_pairs": ribbon_npairs[k],
        "mean_cost": (ribbon_cost[k] / mass) if mass > 0 else np.nan,
    })
ribbon_summary = pd.DataFrame(ribbon_rows).sort_values(
    "mass", ascending=False)
ribbon_summary.to_csv(OUT_DIR / "ribbon_summary.csv", index=False)
print(f"Saved: ribbon_summary.csv ({len(ribbon_summary)} rows)")

long_rows = []
for k, vec in ribbon_gene.items():
    m = ribbon_mass[k]
    if m <= 0:
        continue
    delta_norm = vec / m
    for g_i, g in enumerate(gene_names):
        long_rows.append({
            "P_src": k[0], "P_dst": k[1], "gene": g,
            "delta_ribbon": float(delta_norm[g_i]),
            "n_pairs_ribbon": ribbon_npairs[k],
        })
ribbon_gene_df = pd.DataFrame(long_rows)
ribbon_gene_df.to_csv(OUT_DIR / "ribbon_gene_deltas.csv", index=False)
print(f"Saved: ribbon_gene_deltas.csv ({len(ribbon_gene_df)} rows)")

# top_genes: 20 up + 20 down from the existing rewiring CSV
rw = pd.read_csv(DRAINING_DELTA_CSV).rename(
    columns={"mean_delta": "global_mean_delta",
             "std_delta":  "global_std_delta"})
rw_s = rw.sort_values("global_mean_delta", ascending=False).reset_index(drop=True)
top_up = rw_s.head(N_TOP_GENES).copy()
top_up["rank"] = np.arange(1, N_TOP_GENES + 1)
top_dn = rw_s.tail(N_TOP_GENES).iloc[::-1].copy()
top_dn["rank"] = np.arange(N_TOP_GENES + 1, 2 * N_TOP_GENES + 1)
top40 = pd.concat([top_up, top_dn])[["gene", "global_mean_delta",
                                       "global_std_delta", "rank"]]
top40.to_csv(OUT_DIR / "top_genes.csv", index=False)
print(f"Saved: top_genes.csv ({len(top40)} genes)")

# bayesian sankey csf->tp long form
T = pd.read_csv(BAYES_T_CSV, index_col=0)
bayes_phenos = list(T.index)
csf_obs = obs[(obs["lineage"] == "CD8") & (obs["tissue"] == "CSF")]
src_full_counts = csf_obs["phenotype"].value_counts()
src_w = pd.Series({p: src_full_counts.get(SHORT_TO_FULL.get(p, p), 0)
                    for p in bayes_phenos}, dtype=float)
src_w = src_w / src_w.sum() if src_w.sum() > 0 else src_w
bayes_long_rows = []
for i, ps in enumerate(bayes_phenos):
    for j, pd_ in enumerate(bayes_phenos):
        bayes_long_rows.append({
            "P_src": ps, "P_dst": pd_,
            "weight": float(src_w[ps] * T.iloc[i, j]),
        })
bayes_long = pd.DataFrame(bayes_long_rows)
bayes_long.to_csv(OUT_DIR / "bayesian_sankey_csf_tp.csv", index=False)
print(f"Saved: bayesian_sankey_csf_tp.csv ({len(bayes_long)} rows)")


# %%
# =========================================================
# Pick top ribbons to display
# =========================================================
disp_ribbons = (ribbon_summary[ribbon_summary["n_pairs"] >= RIBBON_MIN_PAIRS]
                .sort_values("mass", ascending=False)
                .head(N_TOP_RIBBONS).copy())
disp_ribbons = disp_ribbons.reset_index(drop=True)
print(f"\nDisplay ribbons (top {N_TOP_RIBBONS} by mass, "
      f"n_pairs>={RIBBON_MIN_PAIRS}):")
print(disp_ribbons.to_string(index=False))


# %%
# =========================================================
# CollecTRI TFs (for label bolding)
# =========================================================
print("\nLoading CollecTRI for TF marking...")
collectri_tfs = set(dc.op.collectri(organism="human")["source"])


# %%
# =========================================================
# Figure
# =========================================================
def ribbon_path(x0, y0t, y0b, x1, y1t, y1b):
    cx = (x0 + x1) / 2.0
    verts = [
        (x0, y0t),
        (cx, y0t), (cx, y1t), (x1, y1t),
        (x1, y1b),
        (cx, y1b), (cx, y0b), (x0, y0b),
        (x0, y0t),
    ]
    codes = [MPath.MOVETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.CLOSEPOLY]
    return MPath(verts, codes)


def draw_sankey(ax, sankey_df, phenos):
    sankey_df = sankey_df[sankey_df["weight"] > 1e-6].copy()
    src_tot = sankey_df.groupby("P_src")["weight"].sum() \
                       .reindex(phenos, fill_value=0)
    dst_tot = sankey_df.groupby("P_dst")["weight"].sum() \
                       .reindex(phenos, fill_value=0)
    gap = 0.015 * max(src_tot.sum(), dst_tot.sum())

    src_y0, cur = {}, 0.0
    for p in phenos:
        src_y0[p] = cur
        cur += src_tot[p] + gap
    src_height = cur - gap
    dst_y0, cur = {}, 0.0
    for p in phenos:
        dst_y0[p] = cur
        cur += dst_tot[p] + gap
    dst_height = cur - gap
    total_h = max(src_height, dst_height)

    bar_w = 0.025
    left_x = 0.10
    right_x = 0.90

    # Bars + labels
    for p in phenos:
        if src_tot[p] > 0:
            ax.add_patch(Rectangle((left_x - bar_w, src_y0[p]),
                                    bar_w, src_tot[p],
                                    color=pheno_color(p),
                                    ec="black", lw=0.4, zorder=4))
            ax.text(left_x - bar_w - 0.005,
                    src_y0[p] + src_tot[p] / 2,
                    pheno_label(p), ha="right", va="center",
                    fontsize=9, color=pheno_color(p),
                    fontweight="bold")
        if dst_tot[p] > 0:
            ax.add_patch(Rectangle((right_x, dst_y0[p]),
                                    bar_w, dst_tot[p],
                                    color=pheno_color(p),
                                    ec="black", lw=0.4, zorder=4))
            ax.text(right_x + bar_w + 0.005,
                    dst_y0[p] + dst_tot[p] / 2,
                    pheno_label(p), ha="left", va="center",
                    fontsize=9, color=pheno_color(p),
                    fontweight="bold")

    # Ribbons: source-major then dest order so they don't twist
    consumed_l = {p: 0.0 for p in phenos}
    consumed_r = {p: 0.0 for p in phenos}
    # Sort ribbons by source position then dest position
    for ps in phenos:
        sub = sankey_df[sankey_df["P_src"] == ps].copy()
        sub["dst_rank"] = sub["P_dst"].map({p: i for i, p in enumerate(phenos)})
        sub = sub.sort_values("dst_rank")
        for _, row in sub.iterrows():
            v = float(row["weight"])
            pd_ = row["P_dst"]
            y0t = src_y0[ps] + consumed_l[ps]
            y0b = y0t + v
            y1t = dst_y0[pd_] + consumed_r[pd_]
            y1b = y1t + v
            consumed_l[ps] += v
            consumed_r[pd_] += v
            path = ribbon_path(left_x, y0t, y0b, right_x, y1t, y1b)
            ax.add_patch(PathPatch(path, facecolor=pheno_color(ps),
                                    edgecolor="none", alpha=0.55,
                                    zorder=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(total_h + gap, -gap)
    ax.set_axis_off()


# ---- Build figure ----
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    width_ratios=[1, 0.025], height_ratios=[1.1, 1.0],
    hspace=0.18, wspace=0.04,
    left=0.10, right=0.96, top=0.92, bottom=0.06,
)
ax_sankey = fig.add_subplot(gs[0, 0])
ax_heat = fig.add_subplot(gs[1, 0])
cax = fig.add_subplot(gs[1, 1])

# Sankey
phenos_present = [p for p in CD8_SHORTS
                  if p in set(bayes_long["P_src"]) | set(bayes_long["P_dst"])]
draw_sankey(ax_sankey, bayes_long, phenos_present)
ax_sankey.set_title("CSF → TP phenotype transitions (CD8, Bayesian T_global)",
                     fontsize=12, pad=6)

# Heatmap
genes_order = top40["gene"].tolist()
gene_to_row = {g: i for i, g in enumerate(genes_order)}
n_rows = len(genes_order)
n_cols = len(disp_ribbons)

# Build matrix from ribbon_gene_df
sub_long = ribbon_gene_df.merge(
    disp_ribbons[["P_src", "P_dst"]], on=["P_src", "P_dst"], how="inner")
sub_long = sub_long[sub_long["gene"].isin(genes_order)]
pivot = sub_long.pivot_table(index="gene", columns=["P_src", "P_dst"],
                              values="delta_ribbon", aggfunc="first")
col_order = [(r["P_src"], r["P_dst"]) for _, r in disp_ribbons.iterrows()]
pivot = pivot.reindex(index=genes_order, columns=col_order)
mat = pivot.to_numpy()
vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax_heat.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto",
                    interpolation="nearest")
# White grid
ax_heat.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
ax_heat.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
ax_heat.grid(which="minor", color="white", lw=0.5)
ax_heat.tick_params(which="minor", bottom=False, left=False)

# Y ticks: gene names (TFs bold)
ax_heat.set_yticks(np.arange(n_rows))
ax_heat.set_yticklabels(genes_order, fontsize=7)
for tick, g in zip(ax_heat.get_yticklabels(), genes_order):
    if g in collectri_tfs:
        tick.set_fontweight("bold")

# Top/bottom direction separator
ax_heat.axhline(N_TOP_GENES - 0.5, color="black", lw=1.0)

# X ticks: ribbon labels
ax_heat.set_xticks(np.arange(n_cols))
xticklabels = [f"{r['P_src']} → {r['P_dst']}"
               for _, r in disp_ribbons.iterrows()]
ax_heat.set_xticklabels(xticklabels, rotation=45, ha="right",
                         fontsize=9)
# Column header bars + mass annotation
for ci, (_, r) in enumerate(disp_ribbons.iterrows()):
    cs = pheno_color(r["P_src"])
    cd = pheno_color(r["P_dst"])
    # half-half color bar above the column
    y_top = -0.6
    y_bot = -1.1
    ax_heat.add_patch(Rectangle((ci - 0.5, y_bot), 0.5,
                                 y_top - y_bot, color=cs,
                                 clip_on=False, lw=0))
    ax_heat.add_patch(Rectangle((ci,       y_bot), 0.5,
                                 y_top - y_bot, color=cd,
                                 clip_on=False, lw=0))
    ax_heat.text(ci, -1.4, f"mass={r['mass']:.1f}", ha="center",
                  va="bottom", fontsize=7, color="#444",
                  clip_on=False)

ax_heat.set_xlabel("")
ax_heat.set_ylabel("")
for s in ("top", "right"):
    ax_heat.spines[s].set_visible(False)

cb = fig.colorbar(im, cax=cax)
cb.set_label("delta_ribbon (log1p, CSF→TP)", fontsize=8)
cb.ax.tick_params(labelsize=7)

# Suptitle / subtitle
total_mass = float(ribbon_summary["mass"].sum())
fig.suptitle("CD8 drainage CSF → TP: phenotype transitions "
              "and per-ribbon gene rewiring",
              fontsize=14, fontweight="bold", y=0.98)
fig.text(0.5, 0.945,
         f"{len(events_ok)} events · {n_cols} ribbons shown "
         f"(n_pairs ≥ {RIBBON_MIN_PAIRS}) · "
         f"total transport mass {total_mass:.1f}",
         ha="center", fontsize=9, color="#555")

fig.savefig(OUT_DIR / "main.png", dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_DIR / "main.pdf", bbox_inches="tight")
plt.close(fig)
print("\nSaved: main.png + main.pdf")


# %%
# =========================================================
# Summary
# =========================================================
print("\n========================================")
print(f"Top {N_TOP_RIBBONS} ribbons shown:")
for _, r in disp_ribbons.iterrows():
    key = (r["P_src"], r["P_dst"])
    deltas = ribbon_gene[key] / ribbon_mass[key]
    g_idx_up = np.argsort(-deltas)[:3]
    g_idx_dn = np.argsort(deltas)[:3]
    up = [f"{gene_names[i]}({deltas[i]:+.2f})" for i in g_idx_up]
    dn = [f"{gene_names[i]}({deltas[i]:+.2f})" for i in g_idx_dn]
    print(f"  {r['P_src']} → {r['P_dst']}  mass={r['mass']:.2f} "
          f"n_pairs={r['n_pairs']}")
    print(f"     up:   {up}")
    print(f"     down: {dn}")
print("\nDone.")
