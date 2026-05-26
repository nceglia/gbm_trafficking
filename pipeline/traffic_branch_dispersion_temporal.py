# %%
"""Temporal extension of stayer-vs-mover (pipeline/branch_dispersion.py).

Three analyses:

  1. Per-timepoint stayer/mover phenotype enrichment heatmap.
     Δ_phenotype = mover (cell-weighted) − stayer (cell-weighted)
     evaluated separately for each transition pair (1→2 … 5→6) and
     each source tissue. Underpowered cells (n_mover < 10 OR
     n_stayer < 10) are greyed out.

  2. Overlay of per-tissue block retention (from
     results/06d_empirical_Q_per_timepoint/) and JSD(mover, stayer)
     per transition pair. Drop in retention concurrent with drop in
     JSD = "everyone leaves indiscriminately at this transition".

  3. Per-tissue scatter of phenotype mover-Δ (time-averaged) vs
     mean optimizer migration rate (sum across destinations,
     from results/06c_empirical_Q/migration_rates.csv). Spearman r.

Outputs to results/branch_dispersion_temporal/.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.patches import Rectangle
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_ORDER,
)


# %%
# ---- Config ----
MIN_N_SRC = 3
TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
NEXT_TP = {t1: t2 for t1, t2 in TRANSITIONS}
TRANSITION_LABELS = [f"{a}→{b}" for a, b in TRANSITIONS]
MIN_PER_CATEGORY = 10
DPI = 200

from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
RETENTION_PATH = (REPO_ROOT / "results" / "06d_empirical_Q_per_timepoint"
                  / "block_retention_per_timepoint.csv")
MIGRATION_PATH = (REPO_ROOT / "results" / "06c_empirical_Q"
                  / "migration_rates.csv")
OUT_DIR = REPO_ROOT / "results" / "branch_dispersion_temporal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = list(TISSUE_ORDER)
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)
N_PHEN = len(PHENOTYPES)


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)


# %%
# ---- Load adata + reuse the stayer/mover categorization logic ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
obs = adata.obs[["trb", "tissue", "timepoint", "phenotype", "patient"]].copy()
obs = obs[obs["trb"].notna() & (obs["trb"].astype(str) != "")]
print(f"  {len(obs):,} TCR+ cells")

ph = (obs.groupby(["patient", "trb", "tissue", "timepoint", "phenotype"],
                   observed=True)
        .size().unstack("phenotype", fill_value=0))
for p in PHENOTYPES:
    if p not in ph.columns:
        ph[p] = 0
ph = ph[PHENOTYPES]
ph["_n"] = ph.sum(axis=1)
ph = ph.reset_index()

present_set = set(
    obs.groupby(["patient", "trb", "tissue", "timepoint"], observed=True)
       .indices.keys()
)

print("Categorizing (clone, src_tissue, t_src) entries...")
records = []
for _, row in ph.iterrows():
    n_src = int(row["_n"])
    if n_src < MIN_N_SRC:
        continue
    t_src = str(row["timepoint"])
    next_tp = NEXT_TP.get(t_src)
    if next_tp is None:
        continue
    patient = row["patient"]
    trb = row["trb"]
    src_tissue = row["tissue"]

    in_src_next = (patient, trb, src_tissue, next_tp) in present_set
    in_other_next = any(
        (patient, trb, j, next_tp) in present_set
        for j in TISSUES if j != src_tissue
    )
    if not (in_src_next or in_other_next):
        continue
    if in_src_next and not in_other_next:
        cat = "STAYER_ONLY"
    elif in_other_next and not in_src_next:
        cat = "MOVER_ONLY"
    else:
        cat = "BOTH"

    frac = row[PHENOTYPES].to_numpy(dtype=float) / n_src
    rec = {
        "patient": patient, "trb": trb,
        "src_tissue": src_tissue, "t_src": t_src,
        "n_src": n_src, "category": cat,
    }
    for k, p in enumerate(PHENOTYPES):
        rec[p] = float(frac[k])
    records.append(rec)
entries = pd.DataFrame(records)
print(f"  {len(entries)} entries; categories: "
      f"{entries['category'].value_counts().to_dict()}")


# %%
# ---- Analysis 1 + 2: per-(tissue, t_src) cell-weighted mover & stayer dists ----
def _cell_weighted_mean(sub):
    if sub.empty:
        return np.zeros(N_PHEN), 0
    w = sub["n_src"].to_numpy(dtype=float)
    frac_mat = sub[PHENOTYPES].to_numpy(dtype=float)
    if w.sum() <= 0:
        return np.zeros(N_PHEN), 0
    return (frac_mat * w[:, None]).sum(axis=0) / w.sum(), int(w.sum())


per_tp_rows = []
delta_arr = {}    # (tissue, t_src) -> (N_PHEN,) mover - stayer
n_mover = {}
n_stayer = {}
jsd = {}          # (tissue, t_src) -> float

for tissue in TISSUES:
    for t_src, _ in TRANSITIONS:
        sub = entries[(entries["src_tissue"] == tissue)
                       & (entries["t_src"] == t_src)]
        mov = sub[sub["category"] == "MOVER_ONLY"]
        sta = sub[sub["category"] == "STAYER_ONLY"]
        mv_mean, mv_n = _cell_weighted_mean(mov)
        st_mean, st_n = _cell_weighted_mean(sta)
        d = mv_mean - st_mean
        delta_arr[(tissue, t_src)] = d
        n_mover[(tissue, t_src)] = mv_n
        n_stayer[(tissue, t_src)] = st_n

        # JSD on the two compositions (skip if either side is empty).
        if mv_n > 0 and st_n > 0:
            mp = mv_mean / max(mv_mean.sum(), 1e-12)
            sp = st_mean / max(st_mean.sum(), 1e-12)
            d_js = float(jensenshannon(mp, sp, base=2))
            if not np.isfinite(d_js):
                d_js = float("nan")
            jsd[(tissue, t_src)] = d_js
        else:
            jsd[(tissue, t_src)] = float("nan")

        for k, p in enumerate(PHENOTYPES):
            per_tp_rows.append({
                "tissue": tissue, "t_src": t_src, "t_dst": NEXT_TP[t_src],
                "transition": f"{t_src}→{NEXT_TP[t_src]}",
                "phenotype": p,
                "mover_frac": float(mv_mean[k]),
                "stayer_frac": float(st_mean[k]),
                "delta_mover_minus_stayer": float(d[k]),
                "n_mover_clones": int(len(mov)),
                "n_stayer_clones": int(len(sta)),
                "n_mover_cells": mv_n,
                "n_stayer_cells": st_n,
                "underpowered": bool((len(mov) < MIN_PER_CATEGORY)
                                      or (len(sta) < MIN_PER_CATEGORY)),
            })

per_tp_df = pd.DataFrame(per_tp_rows)
per_tp_df.to_csv(OUT_DIR / "mover_stayer_delta_per_timepoint.csv",
                  index=False)
print(f"  wrote mover_stayer_delta_per_timepoint.csv ({len(per_tp_df)} rows)")

jsd_rows = []
for tissue in TISSUES:
    for t_src, t_dst in TRANSITIONS:
        jsd_rows.append({
            "tissue": tissue, "t_src": t_src, "t_dst": t_dst,
            "transition": f"{t_src}→{t_dst}",
            "jsd_mover_vs_stayer": jsd[(tissue, t_src)],
            "n_mover_clones": int(
                ((entries["src_tissue"] == tissue)
                 & (entries["t_src"] == t_src)
                 & (entries["category"] == "MOVER_ONLY")).sum()),
            "n_stayer_clones": int(
                ((entries["src_tissue"] == tissue)
                 & (entries["t_src"] == t_src)
                 & (entries["category"] == "STAYER_ONLY")).sum()),
        })
jsd_df = pd.DataFrame(jsd_rows)
jsd_df.to_csv(OUT_DIR / "jsd_mover_vs_stayer_per_timepoint.csv", index=False)
print(f"  wrote jsd_mover_vs_stayer_per_timepoint.csv ({len(jsd_df)} rows)")


# %%
# ---- Figure 1: per-timepoint mover-Δ heatmap, 3 tissues × 5 transitions ----
print("\nFigure 1: mover-Δ heatmap per (tissue, transition)...")
VMIN, VMAX = -0.3, 0.3
ANNOT_THRESH = 0.05

fig, axes = plt.subplots(3, 1, figsize=(8.2, 11.0))
last_im = None
for r, tissue in enumerate(TISSUES):
    ax = axes[r]
    mat = np.zeros((N_PHEN, len(TRANSITIONS)))
    mask = np.zeros((N_PHEN, len(TRANSITIONS)), dtype=bool)
    for j, (t_src, _) in enumerate(TRANSITIONS):
        d = delta_arr[(tissue, t_src)]
        mat[:, j] = d
        clones_m = int(((entries["src_tissue"] == tissue)
                        & (entries["t_src"] == t_src)
                        & (entries["category"] == "MOVER_ONLY")).sum())
        clones_s = int(((entries["src_tissue"] == tissue)
                        & (entries["t_src"] == t_src)
                        & (entries["category"] == "STAYER_ONLY")).sum())
        if clones_m < MIN_PER_CATEGORY or clones_s < MIN_PER_CATEGORY:
            mask[:, j] = True

    im = ax.imshow(mat, cmap="RdBu_r", vmin=VMIN, vmax=VMAX,
                    aspect="auto")
    last_im = im

    # Grey overlay for underpowered columns.
    for j in range(len(TRANSITIONS)):
        if mask[:, j].any():
            ax.add_patch(Rectangle(
                (j - 0.5, -0.5), 1, N_PHEN,
                facecolor="#dddddd", edgecolor="none",
                alpha=0.65, zorder=2))

    # Annotate cells with |Δ| > threshold (skip masked).
    for i in range(N_PHEN):
        for j in range(len(TRANSITIONS)):
            if mask[i, j]:
                continue
            v = float(mat[i, j])
            if abs(v) < ANNOT_THRESH:
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.5 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                     fontsize=6, color=tc, zorder=3)

    ax.set_xticks(range(len(TRANSITIONS)))
    ax.set_xticklabels(TRANSITION_LABELS, fontsize=9)
    ax.set_yticks(range(N_PHEN))
    ax.set_yticklabels(
        [TCELL_PHENOTYPE_LABELS.get(p, p) for p in PHENOTYPES],
        fontsize=8)
    for tick, p in zip(ax.get_yticklabels(), PHENOTYPES):
        tick.set_color(TCELL_PHENOTYPE_COLORS.get(p, "black"))
        tick.set_fontweight("bold")
    ax.set_title(tissue, fontsize=12, fontweight="bold",
                  color=TISSUE_COLORS.get(tissue, "black"), pad=4)
    ax.tick_params(length=0)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)

if last_im is not None:
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                         fraction=0.025, pad=0.02)
    cbar.set_label("Δ phenotype frac (mover − stayer)", fontsize=10)
fig.suptitle("Mover − Stayer phenotype enrichment by transition pair\n"
              "(grey = underpowered, fewer than 10 mover or stayer clones)",
              fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_DIR / "mover_minus_stayer_per_timepoint.png",
             dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_DIR / "mover_minus_stayer_per_timepoint.pdf",
             bbox_inches="tight")
plt.close(fig)
print("  wrote mover_minus_stayer_per_timepoint.{png,pdf}")


# %%
# ---- Figure 2: block retention + mover/stayer JSD overlay ----
print("\nFigure 2: block retention + mover/stayer JSD overlay...")
if not RETENTION_PATH.exists():
    print(f"  WARNING: {RETENTION_PATH} not found — Figure 2 will skip "
          "the retention curve. (Run 06d_empirical_Q_per_timepoint.py.)")
    retention_df = None
else:
    retention_df = pd.read_csv(RETENTION_PATH)
    retention_df["transition"] = (retention_df["t_src"].astype(str)
                                   + "→" + retention_df["t_dst"].astype(str))

fig, ax_ret = plt.subplots(figsize=(8, 4.6))
ax_jsd = ax_ret.twinx()

# Retention (solid, primary y)
if retention_df is not None:
    for tissue in TISSUES:
        sub = (retention_df[retention_df["tissue"] == tissue]
               .copy()
               .assign(_ord=lambda d: d["transition"].map(
                   {lab: i for i, lab in enumerate(TRANSITION_LABELS)}))
               .sort_values("_ord"))
        if sub.empty:
            continue
        ax_ret.plot(sub["transition"], sub["fraction_retained"],
                     marker="o", lw=2.0,
                     color=TISSUE_COLORS.get(tissue, "gray"),
                     label=f"{tissue} retention")
ax_ret.set_xlabel("transition pair (t → t+1)", fontsize=11)
ax_ret.set_ylabel("P[same tissue] (block sum / row total)", fontsize=11)
ax_ret.set_ylim(0, 1.05)

# JSD (dashed, secondary y)
for tissue in TISSUES:
    ys = [jsd[(tissue, t_src)] for t_src, _ in TRANSITIONS]
    ax_jsd.plot(TRANSITION_LABELS, ys,
                 marker="s", linestyle="--", lw=1.6,
                 color=TISSUE_COLORS.get(tissue, "gray"),
                 alpha=0.85, label=f"{tissue} JSD")
ax_jsd.set_ylabel("JSD(mover, stayer) — selectivity",
                  fontsize=11)
ax_jsd.set_ylim(0, 1.0)

_style_axis(ax_ret)
ax_ret.tick_params(axis="x", rotation=15)
for s in ("top",):
    ax_jsd.spines[s].set_visible(False)
ax_jsd.spines["right"].set_visible(True)

# Combined legend
lines1, labels1 = ax_ret.get_legend_handles_labels()
lines2, labels2 = ax_jsd.get_legend_handles_labels()
ax_ret.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=8, ncol=2, frameon=False,
               bbox_to_anchor=(1.0, 1.18))

fig.suptitle("Retention vs mover phenotype selectivity\n"
              "solid = P[same tissue]; dashed = JSD(mover, stayer)",
              fontsize=12, fontweight="bold", y=1.04)
fig.tight_layout()
fig.savefig(OUT_DIR / "retention_vs_jsd.png", dpi=DPI, bbox_inches="tight")
fig.savefig(OUT_DIR / "retention_vs_jsd.pdf", bbox_inches="tight")
plt.close(fig)
print("  wrote retention_vs_jsd.{png,pdf}")


# %%
# ---- Analysis 3: scatter of mean migration rate vs time-averaged mover Δ ----
print("\nFigure 3: mean optimizer migration rate vs mover Δ (time-avg)...")
if not MIGRATION_PATH.exists():
    print(f"  WARNING: {MIGRATION_PATH} not found — Figure 3 will skip.")
    mig_df = None
else:
    mig_raw = pd.read_csv(MIGRATION_PATH)
    mig_df = mig_raw[mig_raw["method"] == "optimizer"]
    print(f"  migration_rates.csv loaded: optimizer rows = {len(mig_df)}")

# Time-averaged mover Δ: cell-weighted average across t_src for each
# (tissue, phenotype) — sum (delta_t * n_mover_t) / sum(n_mover_t).
delta_avg = {tissue: np.zeros(N_PHEN) for tissue in TISSUES}
delta_avg_n = {tissue: 0 for tissue in TISSUES}
for tissue in TISSUES:
    weights = np.array([n_mover[(tissue, t)] for t, _ in TRANSITIONS],
                        dtype=float)
    if weights.sum() <= 0:
        delta_avg[tissue] = np.zeros(N_PHEN)
        continue
    stack = np.stack([delta_arr[(tissue, t)] for t, _ in TRANSITIONS], axis=0)
    delta_avg[tissue] = (stack * weights[:, None]).sum(axis=0) / weights.sum()
    delta_avg_n[tissue] = int(weights.sum())

# Per-(tissue, phenotype) outflow migration rate = sum over destinations.
mig_outflow = {tissue: {p: float("nan") for p in PHENOTYPES}
                for tissue in TISSUES}
if mig_df is not None:
    for tissue in TISSUES:
        sub = mig_df[mig_df["src"] == tissue]
        if sub.empty:
            continue
        agg = sub.groupby("phenotype", observed=True)["rate"].sum()
        for p in PHENOTYPES:
            mig_outflow[tissue][p] = float(agg.get(p, 0.0))

scatter_rows = []
for tissue in TISSUES:
    for k, p in enumerate(PHENOTYPES):
        scatter_rows.append({
            "tissue": tissue, "phenotype": p,
            "mean_migration_rate_outflow": mig_outflow[tissue][p],
            "delta_mover_minus_stayer_timeavg": float(delta_avg[tissue][k]),
        })
scatter_df = pd.DataFrame(scatter_rows)
scatter_df.to_csv(OUT_DIR / "migration_vs_mover_delta.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
for ax, tissue in zip(axes, TISSUES):
    sub = scatter_df[scatter_df["tissue"] == tissue].copy()
    sub = sub.dropna(subset=["mean_migration_rate_outflow",
                              "delta_mover_minus_stayer_timeavg"])
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                 transform=ax.transAxes, color="dimgray")
        ax.set_title(tissue, fontsize=12, fontweight="bold",
                      color=TISSUE_COLORS.get(tissue, "black"))
        continue
    x = sub["mean_migration_rate_outflow"].to_numpy()
    y = sub["delta_mover_minus_stayer_timeavg"].to_numpy()

    for _, row in sub.iterrows():
        c = TCELL_PHENOTYPE_COLORS.get(row["phenotype"], "#999")
        ax.scatter(row["mean_migration_rate_outflow"],
                    row["delta_mover_minus_stayer_timeavg"],
                    s=66, color=c, edgecolor="black", linewidth=0.4,
                    zorder=3)
        lab = TCELL_PHENOTYPE_LABELS.get(row["phenotype"],
                                          row["phenotype"])
        ax.annotate(
            lab,
            xy=(row["mean_migration_rate_outflow"],
                row["delta_mover_minus_stayer_timeavg"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color=c, zorder=4)

    ax.axhline(0, color="#aaa", lw=0.7, alpha=0.6)
    if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
        r, p = spearmanr(x, y)
        txt = f"Spearman r = {r:.2f}\np = {p:.2g}, n = {len(x)}"
    else:
        txt = f"Spearman r = n/a (n = {len(x)})"
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
             ha="left", va="top", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",
                        facecolor="white", edgecolor="lightgray",
                        alpha=0.85))
    ax.set_title(tissue, fontsize=12, fontweight="bold",
                  color=TISSUE_COLORS.get(tissue, "black"))
    ax.set_xlabel("Σ outflow migration rate (optimizer)", fontsize=10)
    ax.set_ylabel("Δ mover − stayer  (time-averaged)", fontsize=10)
    _style_axis(ax)

fig.suptitle("Optimizer migration rate vs mover phenotype enrichment\n"
              "(internal consistency: high outflow → high mover Δ)",
              fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT_DIR / "migration_vs_mover_delta.png", dpi=DPI,
             bbox_inches="tight")
fig.savefig(OUT_DIR / "migration_vs_mover_delta.pdf",
             bbox_inches="tight")
plt.close(fig)
print("  wrote migration_vs_mover_delta.{png,pdf}")


# %%
# ---- Brief printed summary ----
print("\n=== Summary: JSD(mover, stayer) per (tissue, transition) ===")
print(jsd_df.pivot(index="tissue", columns="transition",
                    values="jsd_mover_vs_stayer").round(3))

print("\n=== Summary: per-tissue Spearman(mig_rate, mover Δ) ===")
for tissue in TISSUES:
    sub = scatter_df[scatter_df["tissue"] == tissue].dropna()
    if len(sub) >= 2 and sub["mean_migration_rate_outflow"].std() > 0:
        r, p = spearmanr(sub["mean_migration_rate_outflow"],
                         sub["delta_mover_minus_stayer_timeavg"])
        print(f"  {tissue:>4s}: r = {r:+.3f}, p = {p:.3g}, n = {len(sub)}")
    else:
        print(f"  {tissue:>4s}: insufficient data")

print(f"\nDone. Outputs in {OUT_DIR}")
