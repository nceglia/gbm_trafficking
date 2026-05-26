# %%
"""Bayesian (hierarchical Dirichlet) vs empirical-P comparison.

For each (lineage ∈ {CD8, CD4}, directed tissue pair) we fit the
hierarchical transition model in ``trafficking.model`` / ``inference``
and pull the posterior ``T_global`` matrix. We then extract the analogous
cross-tissue block from the empirical P estimated in
``results/06c_empirical_Q/P_empirical.csv``, row-normalise it, and
compare entry-by-entry.

Outputs go to ``results/06f_bayesian_comparison/``.
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr

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
from modules.constants import DIRECTED_TISSUE_PAIRS  # noqa: E402

from trafficking.inference import run_inference  # noqa: E402
from trafficking.data import _clean_tcr  # noqa: E402

# %%
# ---- Config ----
from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
P_EMP_PATH = REPO_ROOT / "results" / "06c_empirical_Q" / "P_empirical.csv"
OUT_DIR = REPO_ROOT / "results" / "06f_bayesian_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 6 directed pairs from the 3 undirected pairs (both directions).
ALL_DIRECTED_PAIRS = []
for a, b in DIRECTED_TISSUE_PAIRS:
    ALL_DIRECTED_PAIRS.append((a, b))
    ALL_DIRECTED_PAIRS.append((b, a))

LINEAGES = ["CD8", "CD4"]
N_STEPS = 2000
LR = 0.01
N_SAMPLES = 1000
HIERARCHICAL = True

DPI = 200

# %%
# ---- Load adata ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")


# %%
# ---- Pre-flight: per (lineage, tissue pair) shared-clone counts ----
def _shared_clone_stats(adata, t1, t2, lineage):
    tcr = _clean_tcr(adata, lineage, clone_key="trb",
                     phenotype_key="phenotype", tissue_key="tissue")
    tcr = tcr[tcr["tissue"].isin([t1, t2])]
    if tcr.empty:
        return 0, 0
    clones = tcr.groupby("trb", observed=True)["tissue"].apply(set)
    shared = clones[clones.apply(lambda s: t1 in s and t2 in s)].index
    sub = tcr[tcr["trb"].isin(shared)]
    return int(len(shared)), int(len(sub))


print("\n=== Pre-flight: shared-clone counts per (lineage, directed pair) ===")
preflight_rows = []
for lin in LINEAGES:
    for t1, t2 in ALL_DIRECTED_PAIRS:
        n_clones, n_cells = _shared_clone_stats(adata, t1, t2, lin)
        preflight_rows.append({
            "lineage": lin, "t1": t1, "t2": t2,
            "shared_clones": n_clones, "total_cells": n_cells,
        })
        print(f"  {lin:>3s} {t1:>4s} -> {t2:<4s} : "
              f"shared clones = {n_clones:>5d}, total cells = {n_cells:>6d}")
pd.DataFrame(preflight_rows).to_csv(OUT_DIR / "preflight_counts.csv",
                                    index=False)
print(f"\nEstimated time: ~30s × {len(LINEAGES) * len(ALL_DIRECTED_PAIRS)} "
      f"= ~{30 * len(LINEAGES) * len(ALL_DIRECTED_PAIRS) // 60:.0f} min "
      "(plus posterior sampling).")


# %%
# ---- Phenotype short-name → full-name mapping (for indexing into P_empirical) ----
SHORT_TO_FULL = {}
for full in TCELL_PHENOTYPE_ORDER:
    short = (full
             .replace("CD8_Activated_", "")
             .replace("CD8_Quiescent_", "")
             .replace("CD4_", ""))
    SHORT_TO_FULL[short] = full


def _short_to_full(short, lineage):
    if short not in SHORT_TO_FULL:
        raise KeyError(f"Unknown short phenotype name: {short}")
    return SHORT_TO_FULL[short]


# %%
# ---- Step 1: run Bayesian inference for every (lineage, t1, t2) ----
print("\n=== Step 1: hierarchical Bayesian inference ===")
bayes_results = {}
summary_rows = []

for lin in LINEAGES:
    for t1, t2 in ALL_DIRECTED_PAIRS:
        tag = f"{lin}_{t1}_to_{t2}"
        t0 = time.time()
        print(f"\n[{tag}] starting...")
        try:
            result, data, phenotypes = run_inference(
                adata, t1, t2, lineage=lin, temporal=True,
                n_steps=N_STEPS, lr=LR, hierarchical=HIERARCHICAL,
                n_samples=N_SAMPLES, verbose=True,
            )
        except Exception as e:
            print(f"[{tag}] ERROR: {e}")
            summary_rows.append({
                "lineage": lin, "t1": t1, "t2": t2,
                "n_clones": 0, "n_observations": 0,
                "kappa_mean": np.nan,
                "status": f"error: {e}",
            })
            continue

        elapsed = time.time() - t0
        if result is None or len(data) == 0:
            print(f"[{tag}] SKIP (no shared clones). Elapsed {elapsed:.1f}s")
            summary_rows.append({
                "lineage": lin, "t1": t1, "t2": t2,
                "n_clones": int(data["clone"].nunique()) if len(data) else 0,
                "n_observations": int(len(data)),
                "kappa_mean": np.nan,
                "status": "skipped: no shared clones",
            })
            continue

        T_mean = result["T_mean"]
        T_std = result["T_std"]
        kappa_mean = float(result.get("kappa_mean", np.nan))

        bayes_results[(lin, t1, t2)] = {
            "T_mean": T_mean, "T_std": T_std,
            "phenotypes": phenotypes,
            "kappa_mean": kappa_mean,
            "n_clones": int(data["clone"].nunique()),
            "n_observations": int(len(data)),
        }

        Tdf = pd.DataFrame(T_mean, index=phenotypes, columns=phenotypes)
        Tdf.index.name = "source"
        Tdf.columns.name = "destination"
        Tdf.to_csv(OUT_DIR / f"bayesian_T_{tag}.csv")
        np.save(OUT_DIR / f"bayesian_T_{tag}.npy", T_mean)
        np.save(OUT_DIR / f"bayesian_Tstd_{tag}.npy", T_std)

        summary_rows.append({
            "lineage": lin, "t1": t1, "t2": t2,
            "n_clones": int(data["clone"].nunique()),
            "n_observations": int(len(data)),
            "kappa_mean": kappa_mean,
            "status": "ok",
        })
        print(f"[{tag}] done. n_clones={data['clone'].nunique()}, "
              f"n_obs={len(data)}, kappa≈{kappa_mean:.2f}, "
              f"elapsed {elapsed:.1f}s")

bayesian_summary = pd.DataFrame(summary_rows)
bayesian_summary.to_csv(OUT_DIR / "bayesian_summary.csv", index=False)
print(f"\nWrote bayesian_summary.csv "
      f"({(bayesian_summary['status'] == 'ok').sum()} successful runs)")


# %%
# ---- Step 2: load P_empirical and extract per-(lineage, pair) sub-blocks ----
print("\n=== Step 2: extracting matched empirical-P sub-blocks ===")
P_emp = pd.read_csv(P_EMP_PATH, index_col=0)
print(f"  P_empirical shape: {P_emp.shape}")

block_records = {}  # (lin, t1, t2) -> {"P_block": DataFrame, "phenotypes": list}
for (lin, t1, t2), bres in bayes_results.items():
    short_phenos = bres["phenotypes"]  # ordered as Bayesian model expects
    full_phenos = [_short_to_full(p, lin) for p in short_phenos]
    row_labels = [f"{t1}__{p}" for p in full_phenos]
    col_labels = [f"{t2}__{p}" for p in full_phenos]

    # Some labels might be missing if a phenotype is absent at a tissue; skip gracefully.
    rows_present = [r for r in row_labels if r in P_emp.index]
    cols_present = [c for c in col_labels if c in P_emp.columns]
    if len(rows_present) != len(row_labels) or len(cols_present) != len(col_labels):
        missing = set(row_labels + col_labels) - (set(rows_present) | set(cols_present))
        print(f"  [{lin} {t1}->{t2}] WARNING: missing labels in P_emp: {missing}")

    block = P_emp.loc[rows_present, cols_present].copy()
    # Re-index back to the (possibly larger) full set so shapes line up;
    # missing rows/cols become NaN, then we drop them.
    block = block.reindex(index=row_labels, columns=col_labels).fillna(0.0)
    # Row-normalize so it is comparable to a Dirichlet row-stochastic T.
    row_sums = block.values.sum(axis=1, keepdims=True)
    block_norm = block.values / np.where(row_sums > 0, row_sums, 1.0)
    block_df = pd.DataFrame(block_norm,
                             index=short_phenos, columns=short_phenos)
    block_df.index.name = "source"
    block_df.columns.name = "destination"
    block_records[(lin, t1, t2)] = {
        "P_block": block_df,
        "phenotypes": short_phenos,
        "full_phenotypes": full_phenos,
    }
    tag = f"{lin}_{t1}_to_{t2}"
    block_df.to_csv(OUT_DIR / f"P_block_{tag}.csv")


# %%
# ---- Step 3: compare Bayesian T vs empirical P-block ----
print("\n=== Step 3: per-pair comparison ===")
cmp_rows = []
for (lin, t1, t2), bres in bayes_results.items():
    T = bres["T_mean"]
    block = block_records[(lin, t1, t2)]["P_block"].values
    short = bres["phenotypes"]
    if T.shape != block.shape:
        print(f"  [{lin} {t1}->{t2}] shape mismatch "
              f"{T.shape} vs {block.shape} — skipping.")
        continue
    t_flat = T.flatten()
    b_flat = block.flatten()
    if np.std(t_flat) == 0 or np.std(b_flat) == 0:
        r_p, p_p = (np.nan, np.nan)
        r_s, p_s = (np.nan, np.nan)
    else:
        r_p, p_p = pearsonr(t_flat, b_flat)
        r_s, p_s = spearmanr(t_flat, b_flat)
    frob = float(np.linalg.norm(T - block))

    diff = np.abs(T - block)
    top_idx = np.dstack(np.unravel_index(np.argsort(-diff.ravel())[:3], diff.shape))[0]
    top_entries = []
    for ri, ci in top_idx:
        top_entries.append({
            "from": short[int(ri)], "to": short[int(ci)],
            "bayes": float(T[ri, ci]), "empirical": float(block[ri, ci]),
            "abs_diff": float(diff[ri, ci]),
        })

    cmp_rows.append({
        "lineage": lin, "t1": t1, "t2": t2,
        "n_phenotypes": len(short),
        "pearson_r": float(r_p), "pearson_p": float(p_p),
        "spearman_r": float(r_s), "spearman_p": float(p_s),
        "frobenius": frob,
        "n_clones": bres["n_clones"],
        "kappa_mean": bres["kappa_mean"],
        "top_disagreement_entry":
            f"{top_entries[0]['from']}->{top_entries[0]['to']}",
        "top_disagreement_value": top_entries[0]["abs_diff"],
        "top1_bayes": top_entries[0]["bayes"],
        "top1_empirical": top_entries[0]["empirical"],
        "top2_entry":
            f"{top_entries[1]['from']}->{top_entries[1]['to']}",
        "top2_value": top_entries[1]["abs_diff"],
        "top3_entry":
            f"{top_entries[2]['from']}->{top_entries[2]['to']}",
        "top3_value": top_entries[2]["abs_diff"],
    })

    print(f"\n[{lin} {t1}->{t2}] ({len(short)} phenotypes, "
          f"n_clones={bres['n_clones']})")
    print(f"  pearson r  = {r_p:.3f} (p={p_p:.2g})")
    print(f"  spearman ρ = {r_s:.3f} (p={p_s:.2g})")
    print(f"  ||T_bayes - P_block||_F = {frob:.3f}")
    print("  top-3 entry disagreements:")
    for e in top_entries:
        print(f"    {e['from']:>14s} -> {e['to']:<14s}  "
              f"bayes={e['bayes']:.3f}, empirical={e['empirical']:.3f}, "
              f"|diff|={e['abs_diff']:.3f}")

cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(OUT_DIR / "comparison_table.csv", index=False)
print(f"\nWrote {OUT_DIR / 'comparison_table.csv'}")


# %%
# ---- Step 4: scatter figure ----
print("\n=== Step 4: bayesian_vs_empirical figure ===")
plotted = [(lin, t1, t2) for (lin, t1, t2) in bayes_results
           if (lin, t1, t2) in block_records]
n_panels = len(plotted)

# 2 rows × 6 cols (lineages × 6 directed pairs) when full coverage,
# otherwise pack into a tight grid.
ncols = 6 if n_panels >= 6 else max(1, n_panels)
nrows = int(np.ceil(n_panels / ncols))
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(2.6 * ncols, 2.7 * nrows),
                          squeeze=False)

for ax in axes.flat:
    ax.axis("off")

for ax_idx, (lin, t1, t2) in enumerate(plotted):
    ax = axes.flat[ax_idx]
    ax.axis("on")
    T = bayes_results[(lin, t1, t2)]["T_mean"]
    block = block_records[(lin, t1, t2)]["P_block"].values
    short = bayes_results[(lin, t1, t2)]["phenotypes"]

    xs, ys, colors = [], [], []
    for i in range(T.shape[0]):
        full_src = _short_to_full(short[i], lin)
        c = TCELL_PHENOTYPE_COLORS.get(full_src, "gray")
        for j in range(T.shape[1]):
            xs.append(T[i, j])
            ys.append(block[i, j])
            colors.append(c)

    ax.scatter(xs, ys, c=colors, s=20, edgecolors="black", linewidths=0.3,
                alpha=0.85)
    upper = max(max(xs), max(ys), 0.01)
    ax.plot([0, upper], [0, upper], ls="--", color="gray", lw=0.8)
    ax.set_xlim(-0.02, upper + 0.05)
    ax.set_ylim(-0.02, upper + 0.05)
    ax.set_xlabel("Bayesian T", fontsize=8)
    ax.set_ylabel("Empirical P-block", fontsize=8)
    r_p = next((row["pearson_r"] for row in cmp_rows
                if row["lineage"] == lin
                and row["t1"] == t1 and row["t2"] == t2), np.nan)
    ax.text(0.04, 0.95, f"r = {r_p:.2f}", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                       edgecolor="lightgray", alpha=0.85))
    ax.set_title(f"{lin}  {t1}→{t2}", fontsize=9, fontweight="bold",
                  color=TISSUE_COLORS.get(t2, "black"))
    ax.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("Bayesian T_global vs empirical P sub-block",
              fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_DIR / "bayesian_vs_empirical.png", dpi=DPI,
             bbox_inches="tight")
fig.savefig(OUT_DIR / "bayesian_vs_empirical.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  wrote {OUT_DIR / 'bayesian_vs_empirical.png'}")

print("\nDone.")
print(f"All outputs in: {OUT_DIR}")
