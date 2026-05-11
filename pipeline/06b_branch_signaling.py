# %%
"""Time-aware, branch-membership-restricted L-R differential analysis.

Edge unit = (src tissue i, dst tissue j); 9 edges total including
same-tissue. For each edge, pool branch-participating cells across all
(patient, transition) and run LIANA on src side and dst side
separately. Compare per L-R pair.

Output: results/06b_branch_signaling/
"""
import subprocess
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# %%
# ---- Dependency check ----
try:
    import liana as li
except ImportError:
    print("liana is required for this script.")
    print("Install with: pip install liana decoupler-py")
    sys.exit(0)

# %%
# ---- Config ----
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.myeloid_groups import MYELOID_GROUPS, regroup_obs
from modules.style import (
    MYELOID_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
)

DATA_PATH = (REPO_ROOT / "data" / "objects"
             / "GBM_TCR_POS_TCELLS_MYELOID_combined.h5ad")
OUT_DIR = REPO_ROOT / "results" / "06b_branch_signaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS_PER_BRANCH_END = 2
LIANA_PVAL_CUTOFF = 0.05
EXPR_PROP = 0.10
TOP_N_PER_EDGE = 8

MIN_TOTAL_CELLS = 100
MIN_PHENOTYPE_CELLS = 20
MIN_PHENOTYPES = 2

TISSUES = ("PBMC", "CSF", "TP")
SAME_TISSUE_EDGES = [("PBMC", "PBMC"), ("CSF", "CSF"), ("TP", "TP")]
CROSS_TISSUE_EDGES = [
    ("PBMC", "CSF"), ("PBMC", "TP"),
    ("CSF", "PBMC"), ("CSF", "TP"),
    ("TP", "PBMC"), ("TP", "CSF"),
]
EDGES = SAME_TISSUE_EDGES + CROSS_TISSUE_EDGES
EDGE_LABELS = [f"{a}→{b}" for a, b in EDGES]

T_PHENOTYPES = set(TCELL_PHENOTYPE_ORDER)
M_PHENOTYPES = set(MYELOID_GROUPS.values())

DPI = 200

t_start = time.time()

# %%
# ---- Step 1: Load + regroup ----
print("Loading combined adata...")
adata = sc.read(str(DATA_PATH))
print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

for col in ("patient", "tissue", "timepoint", "phenotype",
            "trb", "major_lineage"):
    if col not in adata.obs.columns:
        raise KeyError(f"Required obs column missing: {col!r}")

adata.obs["phenotype"] = adata.obs["phenotype"].astype(object)
is_myeloid = adata.obs["phenotype"].isin(MYELOID_GROUPS.keys())
remapped = regroup_obs(adata.obs.loc[is_myeloid])
adata.obs.loc[is_myeloid, "phenotype"] = remapped.values

n_before = adata.n_obs
adata = adata[~adata.obs["phenotype"].isna()].copy()
print(f"Regrouped {int(is_myeloid.sum())} myeloid cells; "
      f"{n_before - adata.n_obs} unmapped dropped; {adata.n_obs} remain")

adata.obs["timepoint"] = adata.obs["timepoint"].astype(int)
adata.obs["phenotype"] = adata.obs["phenotype"].astype(str)
adata.obs["is_t"] = adata.obs["major_lineage"].astype(str) == "T"
adata.obs["is_myeloid"] = adata.obs["major_lineage"].astype(str) == "Myeloid"

# %%
# ---- Step 2: Enumerate branches (T-cell side) ----
print("\nEnumerating branches...")
t_obs = adata.obs[adata.obs["is_t"]
                  & adata.obs["trb"].notna()].copy()
t_obs["timepoint"] = t_obs["timepoint"].astype(int)

bin_counts = (t_obs.groupby(
    ["trb", "patient", "tissue", "timepoint"], observed=True)
    .size().rename("n").reset_index())
bin_counts = bin_counts[bin_counts["n"] >= MIN_CELLS_PER_BRANCH_END].copy()

branch_records = []
for (trb, patient), grp in bin_counts.groupby(["trb", "patient"], observed=True):
    rows = grp[["tissue", "timepoint", "n"]].to_dict("records")
    for r_src in rows:
        for r_dst in rows:
            if r_dst["timepoint"] != r_src["timepoint"] + 1:
                continue
            branch_records.append({
                "trb": trb,
                "patient": patient,
                "src_tissue": r_src["tissue"],
                "src_t": int(r_src["timepoint"]),
                "dst_tissue": r_dst["tissue"],
                "dst_t": int(r_dst["timepoint"]),
                "n_src": int(r_src["n"]),
                "n_dst": int(r_dst["n"]),
            })
branches = pd.DataFrame(branch_records)
branches.to_csv(OUT_DIR / "branches.csv", index=False)
print(f"  n_branches: {len(branches)}")

n_branches_per_edge = {edge: 0 for edge in EDGES}
for edge in EDGES:
    i, j = edge
    n_branches_per_edge[edge] = int(
        ((branches["src_tissue"] == i)
         & (branches["dst_tissue"] == j)).sum()
    ) if len(branches) else 0

# %%
# ---- Step 3: Build per-edge src and dst AnnData subsets ----
print("\nBuilding per-edge src/dst pools...")
edge_subsets = {}  # edge -> {"src": adata_sub, "dst": adata_sub}

myeloid_cells = adata[adata.obs["is_myeloid"]]
m_keys = pd.MultiIndex.from_arrays([
    myeloid_cells.obs["patient"].values,
    myeloid_cells.obs["tissue"].values,
    myeloid_cells.obs["timepoint"].astype(int).values,
])
myeloid_by_key = {}
for k, idx in zip(m_keys, np.arange(myeloid_cells.n_obs)):
    myeloid_by_key.setdefault(k, []).append(idx)

t_cells_all = adata[adata.obs["is_t"]]
t_obs_all = t_cells_all.obs.copy()
t_obs_all["timepoint"] = t_obs_all["timepoint"].astype(int)
t_obs_all["_idx"] = np.arange(t_cells_all.n_obs)
t_groups = t_obs_all.groupby(
    ["patient", "tissue", "timepoint"], observed=True)

n_t_cells_edge = {edge: {"src": 0, "dst": 0} for edge in EDGES}
n_m_cells_edge = {edge: {"src": 0, "dst": 0} for edge in EDGES}

for edge in EDGES:
    i, j = edge
    bsub = branches[(branches["src_tissue"] == i)
                    & (branches["dst_tissue"] == j)] if len(branches) else \
           pd.DataFrame(columns=branches.columns if len(branches.columns) else [])
    if not len(bsub):
        edge_subsets[edge] = {"src": None, "dst": None}
        continue

    src_t_idx = []
    src_m_idx = []
    seen_src_samples = set()
    for (patient, t), pgrp in bsub.groupby(["patient", "src_t"], observed=True):
        clone_set = set(pgrp["trb"].unique())
        key = (patient, i, int(t))
        if key in t_groups.indices:
            t_sub = t_obs_all.iloc[t_groups.indices[key]]
            mask = t_sub["trb"].isin(clone_set)
            src_t_idx.extend(t_sub.loc[mask, "_idx"].tolist())
        if key not in seen_src_samples:
            seen_src_samples.add(key)
            if key in myeloid_by_key:
                src_m_idx.extend(myeloid_by_key[key])

    dst_t_idx = []
    dst_m_idx = []
    seen_dst_samples = set()
    for (patient, t), pgrp in bsub.groupby(["patient", "dst_t"], observed=True):
        clone_set = set(pgrp["trb"].unique())
        key = (patient, j, int(t))
        if key in t_groups.indices:
            t_sub = t_obs_all.iloc[t_groups.indices[key]]
            mask = t_sub["trb"].isin(clone_set)
            dst_t_idx.extend(t_sub.loc[mask, "_idx"].tolist())
        if key not in seen_dst_samples:
            seen_dst_samples.add(key)
            if key in myeloid_by_key:
                dst_m_idx.extend(myeloid_by_key[key])

    src_t_idx = np.unique(src_t_idx).astype(int)
    src_m_idx = np.unique(src_m_idx).astype(int)
    dst_t_idx = np.unique(dst_t_idx).astype(int)
    dst_m_idx = np.unique(dst_m_idx).astype(int)

    n_t_cells_edge[edge]["src"] = int(len(src_t_idx))
    n_t_cells_edge[edge]["dst"] = int(len(dst_t_idx))
    n_m_cells_edge[edge]["src"] = int(len(src_m_idx))
    n_m_cells_edge[edge]["dst"] = int(len(dst_m_idx))

    if len(src_t_idx) + len(src_m_idx) > 0:
        src_t_ad = t_cells_all[src_t_idx] if len(src_t_idx) else None
        src_m_ad = myeloid_cells[src_m_idx] if len(src_m_idx) else None
        if src_t_ad is not None and src_m_ad is not None:
            src_ad = ad.concat([src_t_ad, src_m_ad], join="outer")
        elif src_t_ad is not None:
            src_ad = src_t_ad.copy()
        else:
            src_ad = src_m_ad.copy()
        src_ad.obs["phenotype"] = src_ad.obs["phenotype"].astype(str)
    else:
        src_ad = None

    if len(dst_t_idx) + len(dst_m_idx) > 0:
        dst_t_ad = t_cells_all[dst_t_idx] if len(dst_t_idx) else None
        dst_m_ad = myeloid_cells[dst_m_idx] if len(dst_m_idx) else None
        if dst_t_ad is not None and dst_m_ad is not None:
            dst_ad = ad.concat([dst_t_ad, dst_m_ad], join="outer")
        elif dst_t_ad is not None:
            dst_ad = dst_t_ad.copy()
        else:
            dst_ad = dst_m_ad.copy()
        dst_ad.obs["phenotype"] = dst_ad.obs["phenotype"].astype(str)
    else:
        dst_ad = None

    edge_subsets[edge] = {"src": src_ad, "dst": dst_ad}
    print(f"  {i}->{j}: src T={len(src_t_idx)}, M={len(src_m_idx)}; "
          f"dst T={len(dst_t_idx)}, M={len(dst_m_idx)}")

# %%
# ---- Step 4: Run LIANA per side ----
print("\nRunning LIANA per (edge, side)...")
liana_rows = []
skip_log = []
n_liana_rows_edge = {edge: {"src": 0, "dst": 0} for edge in EDGES}


def _run_liana(sub, edge, side):
    if sub is None or sub.n_obs < MIN_TOTAL_CELLS:
        skip_log.append({"edge": f"{edge[0]}->{edge[1]}", "side": side,
                         "reason": "fewer_than_100_total_cells",
                         "n_cells": 0 if sub is None else int(sub.n_obs)})
        return None
    counts = sub.obs["phenotype"].value_counts()
    keep_phenos = counts[counts >= MIN_PHENOTYPE_CELLS].index.tolist()
    if len(keep_phenos) < MIN_PHENOTYPES:
        skip_log.append({"edge": f"{edge[0]}->{edge[1]}", "side": side,
                         "reason": "fewer_than_2_phenotypes_with_20_cells",
                         "n_cells": int(sub.n_obs)})
        return None
    sub = sub[sub.obs["phenotype"].isin(keep_phenos)].copy()
    sub.obs["phenotype"] = sub.obs["phenotype"].astype(str)
    try:
        li.mt.cellphonedb(
            sub, groupby="phenotype", resource_name="consensus",
            expr_prop=EXPR_PROP, verbose=False, use_raw=False,
        )
    except Exception as e:
        skip_log.append({"edge": f"{edge[0]}->{edge[1]}", "side": side,
                         "reason": f"liana_failed: {type(e).__name__}: {e}",
                         "n_cells": int(sub.n_obs)})
        return None
    return sub.uns["liana_res"].copy()


for edge in EDGES:
    i, j = edge
    for side in ("src", "dst"):
        sub = edge_subsets[edge][side]
        df = _run_liana(sub, edge, side)
        if df is None:
            print(f"  {i}->{j} {side}: SKIP")
            continue
        df = df.copy()
        df["side"] = side
        df["edge"] = f"{i}->{j}"
        df["src_tissue"] = i
        df["dst_tissue"] = j
        df["n_t_cells"] = n_t_cells_edge[edge][side]
        df["n_myeloid_cells"] = n_m_cells_edge[edge][side]
        liana_rows.append(df)
        n_liana_rows_edge[edge][side] = int(len(df))
        print(f"  {i}->{j} {side}: {len(df)} L-R rows")

if liana_rows:
    liana_full = pd.concat(liana_rows, ignore_index=True)
else:
    liana_full = pd.DataFrame()

if len(skip_log):
    pd.DataFrame(skip_log).to_csv(OUT_DIR / "skipped_sides.csv", index=False)

# %%
# ---- Step 5: Classify direction ----
def _direction(source, target):
    s_t = source in T_PHENOTYPES
    s_m = source in M_PHENOTYPES
    t_t = target in T_PHENOTYPES
    t_m = target in M_PHENOTYPES
    if s_m and t_t:
        return "M_to_T"
    if s_t and t_m:
        return "T_to_M"
    return "other"


if len(liana_full):
    liana_full["direction"] = [
        _direction(s, t) for s, t in
        zip(liana_full["source"], liana_full["target"])
    ]
    liana_full = liana_full[liana_full["direction"] != "other"].copy()
else:
    liana_full["direction"] = pd.Series(dtype=str)

print(f"\nLIANA rows after direction filter: {len(liana_full)}")

# %%
# ---- Step 6: Pivot to one row per (edge, direction, source, target, L-R) ----
KEY_COLS = ["edge", "src_tissue", "dst_tissue", "direction",
            "source", "target", "ligand_complex", "receptor_complex"]

if len(liana_full):
    pvt_mean = (liana_full.pivot_table(
        index=KEY_COLS, columns="side", values="lr_means",
        aggfunc="first"))
    pvt_p = (liana_full.pivot_table(
        index=KEY_COLS, columns="side", values="cellphone_pvals",
        aggfunc="first"))
    for c in ("src", "dst"):
        if c not in pvt_mean.columns:
            pvt_mean[c] = np.nan
        if c not in pvt_p.columns:
            pvt_p[c] = np.nan
    diff = pd.DataFrame({
        "src_lr_means": pvt_mean["src"],
        "dst_lr_means": pvt_mean["dst"],
        "src_p": pvt_p["src"],
        "dst_p": pvt_p["dst"],
    }).reset_index()
    diff["log2fc"] = np.log2(
        (diff["dst_lr_means"].fillna(0) + 1e-6)
        / (diff["src_lr_means"].fillna(0) + 1e-6)
    )
    diff["min_p"] = diff[["src_p", "dst_p"]].min(axis=1, skipna=True)
    sig_mask = (
        (diff["src_p"].fillna(1.0) < LIANA_PVAL_CUTOFF)
        | (diff["dst_p"].fillna(1.0) < LIANA_PVAL_CUTOFF)
    )
    diff = diff[sig_mask].copy()
else:
    diff = pd.DataFrame(columns=KEY_COLS + [
        "src_lr_means", "dst_lr_means", "src_p", "dst_p",
        "log2fc", "min_p"])

diff.to_csv(OUT_DIR / "signaling_edge_differential.csv", index=False)
print(f"signaling_edge_differential.csv: {len(diff)} rows")

# %%
# ---- Step 7: Per-edge summary ----
summary_rows = []
for edge in EDGES:
    i, j = edge
    elabel = f"{i}->{j}"
    for direction in ("M_to_T", "T_to_M"):
        sub = diff[(diff["edge"] == elabel)
                   & (diff["direction"] == direction)]
        if not len(sub):
            summary_rows.append({
                "edge": elabel, "direction": direction,
                "n_lr_significant_either": 0,
                "n_dst_up": 0, "n_src_up": 0,
                "rewiring_volume": 0.0,
            })
            continue
        n_dst_up = int(((sub["log2fc"] > 1)
                         & (sub["dst_p"].fillna(1.0) < 0.05)).sum())
        n_src_up = int(((sub["log2fc"] < -1)
                         & (sub["src_p"].fillna(1.0) < 0.05)).sum())
        summary_rows.append({
            "edge": elabel, "direction": direction,
            "n_lr_significant_either": int(len(sub)),
            "n_dst_up": n_dst_up,
            "n_src_up": n_src_up,
            "rewiring_volume": float(sub["log2fc"].abs().sum()),
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "signaling_edge_summary.csv", index=False)

# %%
# ---- Step 8: Headline heatmap ----
print("\nBuilding heatmap...")


def _build_dir_matrix(direction):
    sub = diff[diff["direction"] == direction].copy()
    if not len(sub):
        return None, None, None
    sub["lr_pair"] = (sub["ligand_complex"].astype(str)
                       + " → " + sub["receptor_complex"].astype(str))

    top_pairs = set()
    for edge in EDGES:
        i, j = edge
        elabel = f"{i}->{j}"
        es = sub[sub["edge"] == elabel].copy()
        if not len(es):
            continue
        es["abs_lfc"] = es["log2fc"].abs()
        per_pair = (es.groupby("lr_pair")["abs_lfc"].max()
                      .sort_values(ascending=False))
        top_pairs.update(per_pair.head(TOP_N_PER_EDGE).index.tolist())

    if not top_pairs:
        return None, None, None

    sub = sub[sub["lr_pair"].isin(top_pairs)].copy()
    edge_cols = [f"{i}->{j}" for i, j in EDGES]
    mat = (sub.groupby(["lr_pair", "edge"])["log2fc"].mean()
              .unstack("edge").reindex(columns=edge_cols))
    dst_p_mat = (sub.groupby(["lr_pair", "edge"])["dst_p"].min()
                    .unstack("edge").reindex(columns=edge_cols))
    src_p_mat = (sub.groupby(["lr_pair", "edge"])["src_p"].min()
                    .unstack("edge").reindex(columns=edge_cols))

    row_order = (mat.abs().max(axis=1).sort_values(ascending=False).index)
    mat = mat.loc[row_order]
    dst_p_mat = dst_p_mat.loc[row_order]
    src_p_mat = src_p_mat.loc[row_order]

    sig_mat = np.zeros_like(mat.values, dtype=bool)
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat.iat[ri, ci]
            if pd.isna(v):
                continue
            if v >= 0:
                p = dst_p_mat.iat[ri, ci]
            else:
                p = src_p_mat.iat[ri, ci]
            if pd.notna(p) and p < 0.05:
                sig_mat[ri, ci] = True
    return mat, sig_mat, (dst_p_mat, src_p_mat)


m_mat, m_sig, _ = _build_dir_matrix("M_to_T")
t_mat, t_sig, _ = _build_dir_matrix("T_to_M")

if m_mat is None and t_mat is None:
    print("No significant L-R rows in either direction; skipping heatmap.")
else:
    edge_cols = [f"{i}->{j}" for i, j in EDGES]
    n_rows_m = m_mat.shape[0] if m_mat is not None else 0
    n_rows_t = t_mat.shape[0] if t_mat is not None else 0
    height_total = 1.6 + 0.30 * max(n_rows_m, n_rows_t)

    max_abs = 0.0
    for mat in (m_mat, t_mat):
        if mat is not None:
            v = np.nanmax(np.abs(mat.values))
            if np.isfinite(v):
                max_abs = max(max_abs, float(v))
    if max_abs <= 0:
        max_abs = 1.0
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(15, max(height_total, 4.5)))
    gs = GridSpec(
        2, 3,
        height_ratios=[0.6, max(n_rows_m, n_rows_t, 1)],
        width_ratios=[1.0, 1.0, 0.035],
        hspace=0.05, wspace=0.18,
        left=0.10, right=0.94, top=0.93, bottom=0.20,
    )
    ax_bar_m = fig.add_subplot(gs[0, 0])
    ax_bar_t = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    def _vol(direction):
        s = summary_df[summary_df["direction"] == direction] \
            .set_index("edge").reindex(edge_cols)
        return s["rewiring_volume"].fillna(0.0).values

    vol_m = _vol("M_to_T")
    vol_t = _vol("T_to_M")
    vol_max = float(max(vol_m.max(), vol_t.max(), 1e-9))

    for ax_bar, vols, title in [
        (ax_bar_m, vol_m, "M → T"),
        (ax_bar_t, vol_t, "T → M"),
    ]:
        ax_bar.bar(range(len(edge_cols)), vols,
                   color=["#888"] * len(edge_cols), edgecolor="black",
                   linewidth=0.4)
        ax_bar.set_xlim(-0.5, len(edge_cols) - 0.5)
        ax_bar.set_xticks([])
        ax_bar.set_ylim(0, vol_max * 1.1)
        ax_bar.set_ylabel("rewire vol", fontsize=8)
        ax_bar.tick_params(axis="y", labelsize=7)
        for s in ("top", "right"):
            ax_bar.spines[s].set_visible(False)
        ax_bar.set_title(title, fontsize=12, fontweight="bold")

    def _draw_heatmap(ax, mat, sig_mat, ylabel_show):
        if mat is None or not len(mat):
            ax.text(0.5, 0.5, "no significant L-R rows",
                    ha="center", va="center", fontsize=11,
                    color="dimgray", transform=ax.transAxes)
            ax.set_xticks(range(len(edge_cols)))
            ax.set_xticklabels(edge_cols, rotation=45, ha="right",
                                fontsize=8)
            ax.set_yticks([])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            return None
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(range(len(edge_cols)))
        ax.set_xticklabels(edge_cols, rotation=45, ha="right", fontsize=8)
        for tick, (i, j) in zip(ax.get_xticklabels(), EDGES):
            tick.set_color(TISSUE_COLORS[i])
            if (i, j) in SAME_TISSUE_EDGES:
                tick.set_fontweight("bold")
        if ylabel_show:
            ax.set_yticks(range(len(mat.index)))
            ax.set_yticklabels(mat.index, fontsize=7)
        else:
            ax.set_yticks([])
        for ri in range(mat.shape[0]):
            for ci in range(mat.shape[1]):
                if sig_mat[ri, ci]:
                    ax.scatter(ci, ri, s=14, marker="o",
                                facecolor="black", edgecolor="white",
                                linewidth=0.5, zorder=3)
        ax.axvline(2.5, color="black", lw=0.8, alpha=0.5)
        return im

    im_m = _draw_heatmap(ax_m, m_mat, m_sig, ylabel_show=True)
    im_t = _draw_heatmap(ax_t, t_mat, t_sig, ylabel_show=True)

    im_for_cbar = im_m if im_m is not None else im_t
    if im_for_cbar is not None:
        cb = fig.colorbar(im_for_cbar, cax=cax)
        cb.set_label("log₂(dst / src) L-R means", fontsize=9)
        cb.ax.tick_params(labelsize=7)
    else:
        cax.axis("off")

    fig.suptitle(
        "Branch-restricted L-R differential (src vs dst pool)",
        fontsize=13, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.06,
        "columns: 9 (src→dst) edges, same-tissue first then cross-tissue. "
        "Color = mean log₂fc across (source, target) pairs within an L-R "
        "pair. Dot = side-specific p<0.05 (dst_p for log₂fc≥0, src_p else).",
        ha="center", va="bottom", fontsize=8, color="dimgray", style="italic",
    )

    png_path = OUT_DIR / "signaling_edge_heatmap.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

# %%
# ---- Step 9: Print summary ----
elapsed = time.time() - t_start

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

print("\nBranches per edge:")
for edge in EDGES:
    i, j = edge
    print(f"  {i}->{j}: {n_branches_per_edge[edge]}")

print("\nCells per (edge, side):")
print(f"  {'edge':<14}{'src_T':>8}{'src_M':>8}{'dst_T':>8}{'dst_M':>8}")
for edge in EDGES:
    i, j = edge
    print(f"  {i+'->'+j:<14}"
          f"{n_t_cells_edge[edge]['src']:>8}"
          f"{n_m_cells_edge[edge]['src']:>8}"
          f"{n_t_cells_edge[edge]['dst']:>8}"
          f"{n_m_cells_edge[edge]['dst']:>8}")

print("\nLIANA rows per (edge, side) [pre-direction filter]:")
print(f"  {'edge':<14}{'src':>8}{'dst':>8}")
empty_edges = []
for edge in EDGES:
    i, j = edge
    s = n_liana_rows_edge[edge]["src"]
    d = n_liana_rows_edge[edge]["dst"]
    if s == 0 and d == 0:
        empty_edges.append(f"{i}->{j}")
    print(f"  {i+'->'+j:<14}{s:>8}{d:>8}")
if empty_edges:
    print(f"  Empty edges (no LIANA on either side): {empty_edges}")

print("\nSignificant rows per (edge, direction) [post-filter, pivoted]:")
print(f"  {'edge':<14}{'direction':<10}{'n_sig':>8}"
      f"{'n_dst_up':>10}{'n_src_up':>10}{'vol':>10}")
for _, r in summary_df.iterrows():
    print(f"  {r['edge']:<14}{r['direction']:<10}"
          f"{r['n_lr_significant_either']:>8}"
          f"{r['n_dst_up']:>10}{r['n_src_up']:>10}"
          f"{r['rewiring_volume']:>10.2f}")

try:
    du_out = subprocess.run(
        ["du", "-sh", str(OUT_DIR)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"\nOutput dir: {du_out}")
except Exception as e:
    print(f"\nOutput dir size unavailable: {e}")

print(f"Elapsed: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
print("Done.")
