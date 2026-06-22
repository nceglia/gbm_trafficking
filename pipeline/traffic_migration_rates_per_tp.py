# %%
"""Per-timepoint empirical Q decomposition + stability analysis.

For each consecutive timepoint pair (t -> t+1), build P_t from clone state
vectors observed at exactly those two timepoints, decompose it into
migration / transition / expansion rates via both matrix logarithm and the
structured optimizer, then summarise consistency of the rates across time.

Outputs go to results/traffic_migration_rates_per_tp/.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_ORDER,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TISSUE_COLORS,
)

from trafficking.ctmc import StateSpace  # noqa: E402
from trafficking.empirical import (  # noqa: E402
    build_clone_state_vectors,
    build_empirical_P,
    decompose_P_logm,
    decompose_P_optimizer,
)

# %%
# ---- Config ----
from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
OUT_DIR = REPO_ROOT / "results" / "traffic_migration_rates_per_tp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = ("PBMC", "CSF", "TP")
PHENOTYPES = tuple(TCELL_PHENOTYPE_ORDER)
TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
MIN_CELLS = 2
DT = 1.0
PSEUDOCOUNT = 0.01

OPT_L2_REG = 1e-3
OPT_MAXITER = 2000
OPT_PRINT_EVERY = 400

METHOD_FOR_PLOTS = "optimizer"  # non-negative; "logm" can have negatives

EDGES = [(a, b) for a in TISSUES for b in TISSUES if a != b]


# %%
# ---- Load ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

ss = StateSpace(TISSUES, PHENOTYPES)
print(f"  {ss}")
state_labels = [f"{site}__{ph}" for site, ph in ss.labels]

# Build full observation list once, then partition by transition pair.
print("\nBuilding clone state vectors (all transitions)...")
all_obs = build_clone_state_vectors(
    adata, ss, TRANSITIONS, min_cells=MIN_CELLS,
)
obs_by_pair = {pair: [] for pair in TRANSITIONS}
for o in all_obs:
    obs_by_pair[(o["t_src"], o["t_dst"])].append(o)

# %%
# ---- Per-timepoint fits ----
per_time_dir = OUT_DIR / "per_timepoint"
per_time_dir.mkdir(exist_ok=True)

rate_rows = []
diag_rows = []
block_rows = []

for (t1, t2) in TRANSITIONS:
    obs = obs_by_pair[(t1, t2)]
    pair_tag = f"{t1}_to_{t2}"
    sub_dir = per_time_dir / pair_tag
    sub_dir.mkdir(exist_ok=True)

    if len(obs) < 2:
        print(f"\n--- {pair_tag}: SKIP (only {len(obs)} obs) ---")
        continue
    print(f"\n--- {pair_tag}: {len(obs)} observations ---")

    P, P_diag = build_empirical_P(obs, ss, pseudocount=PSEUDOCOUNT)
    np.save(sub_dir / "P_empirical.npy", P)
    (pd.DataFrame(P, index=state_labels, columns=state_labels)
       .rename_axis("from")
       .to_csv(sub_dir / "P_empirical.csv"))

    # diagonal-block fraction (probability mass retained in each tissue)
    for site in ss.sites:
        idx = ss.site_indices(site)
        block_total = float(P[np.ix_(idx, idx)].sum())
        row_total = float(P[idx, :].sum())
        block_rows.append({
            "transition": pair_tag, "t_src": t1, "t_dst": t2,
            "tissue": site,
            "block_sum": block_total,
            "row_total": row_total,
            "fraction_retained": (block_total / row_total
                                   if row_total > 0 else float("nan")),
        })

    print("  logm decomposition...")
    res_logm = decompose_P_logm(P, ss, dt=DT)
    np.save(sub_dir / "Q_logm.npy", res_logm["Q"])

    print("  optimizer decomposition...")
    res_opt = decompose_P_optimizer(
        P, ss, dt=DT, l2_reg=OPT_L2_REG, maxiter=OPT_MAXITER,
        verbose=True, print_every=OPT_PRINT_EVERY,
    )
    np.save(sub_dir / "Q_optimizer.npy", res_opt["Q"])
    np.save(sub_dir / "P_fitted_optimizer.npy", res_opt["P_fitted"])

    diag_rows.append({
        "transition": pair_tag, "t_src": t1, "t_dst": t2,
        "n_obs": len(obs),
        "condition_number": float(P_diag["condition_number"]),
        "n_states_lt5": int((P_diag["row_coverage"] < 5).sum()),
        "logm_valid": bool(res_logm["valid"]),
        "logm_violations": int(res_logm["n_violations"]),
        "opt_loss": float(res_opt["loss"]),
        "opt_frob_err": float(res_opt["frobenius_error"]),
        "opt_converged": bool(res_opt["scipy_result"].success),
    })

    for method_name, result in [("logm", res_logm), ("optimizer", res_opt)]:
        mig = result["migration"]
        for s_from in range(ss.S):
            for s_to in range(ss.S):
                if s_from == s_to:
                    continue
                for k in range(ss.K):
                    rate_rows.append({
                        "transition": pair_tag, "t_src": t1, "t_dst": t2,
                        "component": "migration", "method": method_name,
                        "src": ss.sites[s_from], "dst": ss.sites[s_to],
                        "site": "",
                        "phenotype": ss.phenotypes[k],
                        "from_pheno": "", "to_pheno": "",
                        "rate": float(mig[s_from, s_to, k]),
                    })

        trans = result["transition"]
        for s in range(ss.S):
            for k_from in range(ss.K):
                for k_to in range(ss.K):
                    if k_from == k_to:
                        continue
                    rate_rows.append({
                        "transition": pair_tag, "t_src": t1, "t_dst": t2,
                        "component": "transition", "method": method_name,
                        "src": "", "dst": "",
                        "site": ss.sites[s],
                        "phenotype": "",
                        "from_pheno": ss.phenotypes[k_from],
                        "to_pheno": ss.phenotypes[k_to],
                        "rate": float(trans[s, k_from, k_to]),
                    })

        exp_arr = result["expansion"]
        for s in range(ss.S):
            for k in range(ss.K):
                rate_rows.append({
                    "transition": pair_tag, "t_src": t1, "t_dst": t2,
                    "component": "expansion", "method": method_name,
                    "src": "", "dst": "",
                    "site": ss.sites[s],
                    "phenotype": ss.phenotypes[k],
                    "from_pheno": "", "to_pheno": "",
                    "rate": float(exp_arr[s, k]),
                })

rates_df = pd.DataFrame(rate_rows)
rates_df.to_csv(OUT_DIR / "rates_per_timepoint.csv", index=False)
diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(OUT_DIR / "diagnostics_per_timepoint.csv", index=False)
block_df = pd.DataFrame(block_rows)
block_df.to_csv(OUT_DIR / "block_retention_per_timepoint.csv", index=False)
print(f"\nWrote rates_per_timepoint.csv ({len(rates_df)} rows)")
print(f"Wrote diagnostics_per_timepoint.csv ({len(diag_df)} rows)")
print(f"Wrote block_retention_per_timepoint.csv ({len(block_df)} rows)")


# %%
# ---- Stability summary (mean / std / CV across time) ----
def _summarize(df, group_cols):
    g = df.groupby(group_cols + ["method"], observed=True)["rate"]
    summary = g.agg(["mean", "std", "median", "min", "max", "count"]).reset_index()
    abs_mean = summary["mean"].abs().replace(0, np.nan)
    summary["cv"] = summary["std"] / abs_mean
    return summary


mig_long = rates_df[rates_df["component"] == "migration"]
trans_long = rates_df[rates_df["component"] == "transition"]

mig_summary = _summarize(mig_long, ["src", "dst", "phenotype"])
trans_summary = _summarize(trans_long, ["site", "from_pheno", "to_pheno"])

mig_summary.to_csv(OUT_DIR / "migration_stability_summary.csv", index=False)
trans_summary.to_csv(OUT_DIR / "transition_stability_summary.csv", index=False)


# %%
# ---- Plots ----
def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)


mig_plot = mig_long[mig_long["method"] == METHOD_FOR_PLOTS].copy()
mig_plot["edge"] = mig_plot["src"] + "→" + mig_plot["dst"]
edge_labels = [f"{a}→{b}" for a, b in EDGES]
edge_colors = {f"{a}→{b}": TISSUE_COLORS.get(b, "gray") for a, b in EDGES}

# ---- 1. Per-phenotype panel grid: migration rate across time ----
n_ph = len(ss.phenotypes)
ncols = 4
nrows = int(np.ceil(n_ph / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 2.4 * nrows),
                          sharex=True, sharey=False)
handles_seen = {}
for i, ph in enumerate(ss.phenotypes):
    ax = axes.flat[i]
    sub = mig_plot[mig_plot["phenotype"] == ph]
    for edge in edge_labels:
        esub = (sub[sub["edge"] == edge]
                .sort_values("transition"))
        if esub.empty:
            continue
        h, = ax.plot(esub["transition"], esub["rate"],
                     marker="o", lw=1.4,
                     color=edge_colors[edge], alpha=0.9, label=edge)
        handles_seen.setdefault(edge, h)
    ax.set_title(TCELL_PHENOTYPE_LABELS.get(ph, ph),
                 fontsize=9, fontweight="bold",
                 color=TCELL_PHENOTYPE_COLORS.get(ph, "black"))
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    _style_axis(ax)
for j in range(n_ph, nrows * ncols):
    axes.flat[j].axis("off")
if handles_seen:
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(list(handles_seen.values()), list(handles_seen.keys()),
                      loc="center", fontsize=8, frameon=False,
                      title="edge", title_fontsize=9)
fig.supylabel("migration rate", fontsize=10)
fig.supxlabel("transition pair", fontsize=10)
fig.suptitle("Migration rate consistency across timepoints — per phenotype",
              fontsize=12, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "migration_consistency_per_phenotype.pdf",
             bbox_inches="tight")
fig.savefig(OUT_DIR / "migration_consistency_per_phenotype.png",
             dpi=180, bbox_inches="tight")
plt.close(fig)

# ---- 2. Per-edge panel grid: migration rate across time, colored by phenotype ----
fig, axes = plt.subplots(2, 3, figsize=(4 * 3, 3.2 * 2), sharex=True)
for i, edge in enumerate(edge_labels):
    ax = axes.flat[i]
    sub = mig_plot[mig_plot["edge"] == edge]
    for ph in ss.phenotypes:
        psub = (sub[sub["phenotype"] == ph]
                .sort_values("transition"))
        if psub.empty:
            continue
        ax.plot(psub["transition"], psub["rate"], marker="o", lw=1.3,
                color=TCELL_PHENOTYPE_COLORS.get(ph, "gray"),
                label=TCELL_PHENOTYPE_LABELS.get(ph, ph), alpha=0.9)
    ax.set_title(edge, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    _style_axis(ax)
handles, labels = axes.flat[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(1.04, 0.5), fontsize=8,
               frameon=False, title="phenotype")
fig.supylabel("migration rate", fontsize=10)
fig.suptitle("Migration rate consistency across timepoints — per edge",
              fontsize=12, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 0.94, 1.0])
fig.savefig(OUT_DIR / "migration_consistency_per_edge.pdf",
             bbox_inches="tight")
fig.savefig(OUT_DIR / "migration_consistency_per_edge.png",
             dpi=180, bbox_inches="tight")
plt.close(fig)

# ---- 3. Stability heatmaps: mean rate + CV across time ----
mig_sum_opt = mig_summary[mig_summary["method"] == METHOD_FOR_PLOTS].copy()
mig_sum_opt["edge"] = mig_sum_opt["src"] + "→" + mig_sum_opt["dst"]
piv_mean = (mig_sum_opt
            .pivot(index="phenotype", columns="edge", values="mean")
            .reindex(index=list(ss.phenotypes), columns=edge_labels))
piv_cv = (mig_sum_opt
          .pivot(index="phenotype", columns="edge", values="cv")
          .reindex(index=list(ss.phenotypes), columns=edge_labels))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
im0 = axes[0].imshow(piv_mean.values, aspect="auto", cmap="viridis")
axes[0].set_xticks(range(len(piv_mean.columns)))
axes[0].set_xticklabels(piv_mean.columns, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(len(piv_mean.index)))
axes[0].set_yticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in piv_mean.index],
                         fontsize=8)
axes[0].set_title("Mean migration rate across time", fontsize=11)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

cv_vmax = float(np.nanpercentile(piv_cv.values, 95)) if piv_cv.values.size else 1.0
im1 = axes[1].imshow(piv_cv.values, aspect="auto", cmap="magma",
                      vmin=0, vmax=max(cv_vmax, 1e-6))
axes[1].set_xticks(range(len(piv_cv.columns)))
axes[1].set_xticklabels(piv_cv.columns, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(len(piv_cv.index)))
axes[1].set_yticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in piv_cv.index],
                         fontsize=8)
axes[1].set_title("Coefficient of variation σ/|μ| (lower = more stable)",
                  fontsize=11)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle(
    f"Migration rate stability across {len(TRANSITIONS)} timepoint pairs "
    f"(method = {METHOD_FOR_PLOTS})",
    fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "migration_stability_heatmap.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "migration_stability_heatmap.png", dpi=180,
             bbox_inches="tight")
plt.close(fig)

# ---- 4. Per-tissue diagonal-block retention across time ----
fig, ax = plt.subplots(figsize=(6, 3.6))
for site in ss.sites:
    sub = (block_df[block_df["tissue"] == site]
           .sort_values("transition"))
    ax.plot(sub["transition"], sub["fraction_retained"],
            marker="o", lw=1.8,
            color=TISSUE_COLORS.get(site, "gray"),
            label=site)
ax.set_xlabel("transition pair")
ax.set_ylabel("P[same tissue] fraction (block sum / row total)")
ax.set_title("Tissue retention stability across time",
             fontsize=11, fontweight="bold")
ax.legend(loc="best", frameon=False)
_style_axis(ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "block_retention_stability.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "block_retention_stability.png", dpi=180,
             bbox_inches="tight")
plt.close(fig)


# %%
# ---- Brief printed summary of the most/least stable phenotype × edge pairs ----
print("\n=== Migration stability (optimizer rates) ===")
mig_sum_opt_sorted = mig_sum_opt[mig_sum_opt["count"] > 1].copy()
mig_sum_opt_sorted["edge"] = (mig_sum_opt_sorted["src"]
                              + "->" + mig_sum_opt_sorted["dst"])

print("\nTop 5 LEAST variable (low CV, well-resolved across time):")
top_stable = (mig_sum_opt_sorted[mig_sum_opt_sorted["mean"] > 0.05]
              .nsmallest(5, "cv"))
for _, r in top_stable.iterrows():
    print(f"  {r['edge']:>12} [{TCELL_PHENOTYPE_LABELS.get(r['phenotype'], r['phenotype']):>10}]: "
          f"mean={r['mean']:+.3f}, std={r['std']:.3f}, CV={r['cv']:.2f}")

print("\nTop 5 MOST variable (high CV, unstable):")
top_var = (mig_sum_opt_sorted[mig_sum_opt_sorted["mean"] > 0.05]
           .nlargest(5, "cv"))
for _, r in top_var.iterrows():
    print(f"  {r['edge']:>12} [{TCELL_PHENOTYPE_LABELS.get(r['phenotype'], r['phenotype']):>10}]: "
          f"mean={r['mean']:+.3f}, std={r['std']:.3f}, CV={r['cv']:.2f}")

print(f"\nDone. Outputs in {OUT_DIR}")
