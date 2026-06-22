# %%
"""Per-batch Scrublet doublet detection on the T cell AnnData.

Runs Scrublet separately per (patient × tissue) batch — heterogeneous
capture efficiency across batches makes a global run flag whole batches
spuriously. Writes:

  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad   (filtered AnnData)
  results/qc_scrublet_doublets/scrublet_summary.csv        (per-cell scores)
  results/qc_scrublet_doublets/doublet_score_hist.png      (per-batch histogram)

Reads paths.H5AD_TCELLS_RAW (celltyping output). Downstream pipeline steps
use paths.H5AD_TCELLS, which points at the singlets file written here.

Implementation note: scrublet 0.2.3 + annoy 1.17.x collapses all cells
to identical scores when use_approx_neighbors=True (the default). We
force exact sklearn kNN. Slower but correct.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.style import TISSUE_COLORS  # noqa: E402

paths.ensure(paths.QC_DOUBLETS_DIR)

EXPECTED_DOUBLET_RATE = 0.06
MIN_CELLS_PER_BATCH = 30
# Fallback threshold when Scrublet's auto-detection cannot find a clean
# bimodal cut (returns predicted=None). 0.25 is the value the Scrublet
# tutorial recommends for "I-just-want-a-number" usage.
FALLBACK_THRESHOLD = 0.25


# %%
# =========================================================
# Load + per-batch Scrublet
# =========================================================
print(f"Loading {paths.H5AD_TCELLS_RAW.name}...")
adata = sc.read(str(paths.H5AD_TCELLS_RAW))
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

adata.obs["doublet_score"] = np.nan
adata.obs["predicted_doublet"] = False
adata.obs["scrublet_threshold"] = np.nan
adata.obs["scrublet_batch"] = (
    adata.obs["patient"].astype(str) + "|"
    + adata.obs["tissue"].astype(str)
)

per_batch_rows = []
counts_layer = adata.layers["counts"]
batches = sorted(adata.obs["scrublet_batch"].unique().tolist())
print(f"  scrublet batches: {len(batches)}")

for batch in tqdm(batches, desc="  scrublet"):
    idx = np.where(adata.obs["scrublet_batch"].values == batch)[0]
    if len(idx) < MIN_CELLS_PER_BATCH:
        per_batch_rows.append({
            "batch": batch, "n_cells": int(len(idx)),
            "threshold": np.nan, "n_doublet": 0,
            "frac_doublet": 0.0, "status": "skipped_low_n",
        })
        continue
    X_batch = counts_layer[idx]
    try:
        scrub = scr.Scrublet(
            X_batch, expected_doublet_rate=EXPECTED_DOUBLET_RATE,
            random_state=0,
        )
        scores, predicted = scrub.scrub_doublets(
            verbose=False, use_approx_neighbors=False)
        if predicted is None:
            threshold = FALLBACK_THRESHOLD
            predicted = (np.asarray(scores) >= threshold)
            status = "ok_fallback_thresh"
        else:
            threshold = float(np.nanmin(np.asarray(scores)[predicted])) \
                if np.any(predicted) else FALLBACK_THRESHOLD
            status = "ok_auto_thresh"
    except Exception as e:
        scores = np.full(len(idx), np.nan)
        predicted = np.zeros(len(idx), dtype=bool)
        threshold = float("nan")
        status = f"failed: {type(e).__name__}"
    adata.obs.iloc[idx, adata.obs.columns.get_loc("doublet_score")] = scores
    adata.obs.iloc[idx, adata.obs.columns.get_loc("predicted_doublet")] = (
        predicted)
    adata.obs.iloc[idx, adata.obs.columns.get_loc("scrublet_threshold")] = (
        threshold)
    per_batch_rows.append({
        "batch": batch, "n_cells": int(len(idx)),
        "threshold": threshold, "n_doublet": int(np.nansum(predicted)),
        "frac_doublet": float(np.nansum(predicted) / max(len(idx), 1)),
        "status": status,
    })

per_batch_df = pd.DataFrame(per_batch_rows)
print("\nPer-batch scrublet results:")
print(per_batch_df.to_string(index=False))
n_doublet = int(adata.obs["predicted_doublet"].sum())
print(f"\nTotal cells flagged as doublets: {n_doublet:,} "
      f"({n_doublet / adata.n_obs * 100:.2f}%)")


# %%
# =========================================================
# Save filtered AnnData + summary CSV
# =========================================================
summary_csv = paths.QC_DOUBLETS_DIR / "scrublet_summary.csv"
per_batch_df.to_csv(summary_csv, index=False)

per_cell_csv = paths.QC_DOUBLETS_DIR / "scrublet_per_cell.csv"
(adata.obs[["patient", "tissue", "timepoint",
            "scrublet_batch", "doublet_score",
            "predicted_doublet", "scrublet_threshold"]]
 .to_csv(per_cell_csv, index=True))
print(f"  wrote {summary_csv.name}, {per_cell_csv.name}")

keep_mask = ~adata.obs["predicted_doublet"].astype(bool).values
adata_singlets = adata[keep_mask].copy()
print(f"\nSinglets: {adata_singlets.n_obs:,} / {adata.n_obs:,} cells")
adata_singlets.write_h5ad(str(paths.H5AD_TCELLS_SINGLETS))
print(f"  wrote {paths.H5AD_TCELLS_SINGLETS}")


# %%
# =========================================================
# Diagnostic plot — per-batch doublet score histogram
# =========================================================
ok_batches = [r["batch"] for r in per_batch_rows
               if r["status"].startswith("ok")]
if ok_batches:
    n_b = len(ok_batches)
    n_cols = min(4, n_b)
    n_rows = int(np.ceil(n_b / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 2.2 * n_rows),
                             squeeze=False)
    for ax, batch in zip(axes.ravel(), ok_batches):
        mask = (adata.obs["scrublet_batch"].values == batch)
        scores = adata.obs.loc[mask, "doublet_score"].dropna().values
        thr = per_batch_df.loc[per_batch_df["batch"] == batch,
                                "threshold"].iloc[0]
        tis = batch.split("|")[-1]
        ax.hist(scores, bins=40, color=TISSUE_COLORS.get(tis, "#888"),
                alpha=0.85, edgecolor="black", linewidth=0.3)
        if np.isfinite(thr):
            ax.axvline(thr, color="#aa3333", lw=1.0, ls="--")
        n_doub = int((scores >= thr).sum()) if np.isfinite(thr) else 0
        ax.set_title(f"{batch}\nn={len(scores):,}  d={n_doub}",
                     fontsize=7, linespacing=1.1)
        ax.set_xlabel("doublet score", fontsize=7)
        ax.set_ylabel("cells", fontsize=7)
        ax.tick_params(labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(ok_batches):]:
        ax.axis("off")
    fig.suptitle("Scrublet doublet scores per (patient × tissue)",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout()
    hist_png = paths.QC_DOUBLETS_DIR / "doublet_score_hist.png"
    fig.savefig(hist_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {hist_png.name}")

print(f"\nDone. Outputs in {paths.QC_DOUBLETS_DIR}")
