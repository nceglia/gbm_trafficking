# %%
"""Cross-lineage (T vs myeloid) correlations of phenotype frequency
and pathway scores across patients and timepoints, per tissue.

Default behavior: myeloid phenotypes are collapsed to coarse groups via
pipeline.modules.myeloid_groups before correlation. The fine-grained
version is preserved as 11a_cross_lineage_correlations_fine.py.

Inputs come from script 10's temporal aggregations. For each tissue we
correlate:
  (1) phenotype-frequency T vs M (regrouped), paired on (patient, timepoint)
  (2) pathway-score (T-pheno, T-pathway) vs (M-pheno, M-pathway),
      paired on (patient, timepoint), vectorized

Correlations use weighted Spearman (ranks weighted by min cell count
across the two lineages in each pair) and BH-FDR per tissue. Pathway
correlations are filtered to q<0.10 & |rho|>0.4 before write; phenotype
correlations are written in full so the heatmap can dim non-significant
cells.
"""
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import rankdata, t as tdist
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.modules.myeloid_groups import (
    regroup_composition,
    regroup_pathway_scores,
)

# %%
# ---- Config ----
INPUT_DIR = REPO_ROOT / "results" / "pathway_temporal_scores"
OUTPUT_DIR = REPO_ROOT / "results" / "pathway_cross_lineage_corr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS = 6
MIN_MEAN_FREQ = 0.01
MIN_CELLS_PATHWAY = 3
PHENOTYPE_Q_CUTOFF = 0.10
PHENOTYPE_RHO_CUTOFF = 0.4
PATHWAY_Q_CUTOFF = 0.10
PATHWAY_RHO_CUTOFF = 0.4
HEATMAP_TISSUES = ["PBMC", "Tumor"]

# %%
# ---- Helpers ----
def weighted_spearman(x, y, w):
    """Spearman rho computed as weighted Pearson on ranks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(x) < 3:
        return np.nan, np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    if w.sum() <= 0:
        return np.nan, np.nan
    mx = np.average(rx, weights=w)
    my = np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    vx = np.average((rx - mx) ** 2, weights=w)
    vy = np.average((ry - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return np.nan, np.nan
    rho = float(np.clip(cov / np.sqrt(vx * vy), -1.0, 1.0))
    n = len(x)
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t_stat = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
        p = 2.0 * (1.0 - tdist.cdf(abs(t_stat), n - 2))
    return rho, float(p)


def vectorized_weighted_spearman(X, Y, w):
    """All pairwise weighted Spearman correlations between columns of X and Y."""
    w = np.asarray(w, dtype=float)
    w = w * (len(w) / w.sum())
    Xr = np.apply_along_axis(rankdata, 0, X)
    Yr = np.apply_along_axis(rankdata, 0, Y)
    W = w.sum()
    muX = (Xr * w[:, None]).sum(0) / W
    muY = (Yr * w[:, None]).sum(0) / W
    Xc = Xr - muX
    Yc = Yr - muY
    cov = (Xc * w[:, None]).T @ Yc / W
    vX = ((Xc ** 2) * w[:, None]).sum(0) / W
    vY = ((Yc ** 2) * w[:, None]).sum(0) / W
    rho = cov / np.sqrt(np.outer(vX, vY))
    n = len(w)
    tstat = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
    p = 2 * (1 - tdist.cdf(np.abs(tstat), n - 2))
    return rho, p


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. NaNs preserved."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    if mask.sum() == 0:
        return out
    pm = p[mask]
    n = len(pm)
    order = np.argsort(pm)
    ranked = pm[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    qfull = np.empty(n)
    qfull[order] = q
    out[mask] = qfull
    return out

# %%
# ---- Load inputs ----
comp_t = pd.read_csv(INPUT_DIR / "temporal_composition_tcell.csv")
comp_m = pd.read_csv(INPUT_DIR / "temporal_composition_myeloid.csv")
path_t = pd.read_csv(INPUT_DIR / "temporal_pathway_scores_tcell.csv")
path_m = pd.read_csv(INPUT_DIR / "temporal_pathway_scores_myeloid.csv")

print(f"T composition: {len(comp_t)} rows; M composition: {len(comp_m)} rows (fine)")
print(f"T pathway:     {len(path_t)} rows; M pathway:     {len(path_m)} rows (fine)")

# Apply coarse myeloid grouping.
comp_m = regroup_composition(comp_m)
path_m = regroup_pathway_scores(path_m)
print(f"Regrouped M composition: {len(comp_m)} rows over "
      f"{comp_m['phenotype'].nunique()} groups")
print(f"Regrouped M pathway:     {len(path_m)} rows over "
      f"{path_m['phenotype'].nunique()} groups")

# Tag any missing lineage column for safety.
if "lineage" not in comp_t.columns:
    comp_t["lineage"] = "T"
if "lineage" not in comp_m.columns:
    comp_m["lineage"] = "Myeloid"

# %%
# ---- patient_observations.csv ----
patient_obs = pd.concat(
    [
        comp_t.assign(metric="frac")[
            ["patient", "tissue", "timepoint", "lineage", "phenotype",
             "n_cells_phenotype", "n_cells_total", "frac"]
        ].rename(columns={"frac": "value"}),
        comp_m.assign(metric="frac")[
            ["patient", "tissue", "timepoint", "lineage", "phenotype",
             "n_cells_phenotype", "n_cells_total", "frac"]
        ].rename(columns={"frac": "value"}),
    ],
    ignore_index=True,
)

_obs_path = OUTPUT_DIR / "patient_observations.csv"
_chunk = max(1, len(patient_obs) // 50)
with tqdm(total=len(patient_obs), desc="patient observations",
          leave=False, unit="row") as pb:
    for i, start in enumerate(range(0, len(patient_obs), _chunk)):
        patient_obs.iloc[start:start + _chunk].to_csv(
            _obs_path, mode="w" if i == 0 else "a",
            header=(i == 0), index=False,
        )
        pb.update(min(_chunk, len(patient_obs) - start))
print(f"Wrote {len(patient_obs)} rows to patient_observations.csv")

# %%
# ---- (1) Phenotype-frequency correlations ----
print("\n" + "=" * 60)
print("Phenotype-frequency correlations")
print("=" * 60)

tissues = sorted(set(comp_t["tissue"]).union(comp_m["tissue"]))

pheno_rows = []
for tissue in tqdm(tissues, desc="phenotype corrs (tissues)", unit="tissue"):
    ct = comp_t[comp_t["tissue"] == tissue]
    cm = comp_m[comp_m["tissue"] == tissue]
    if not len(ct) or not len(cm):
        continue

    t_phenos = sorted(ct["phenotype"].unique())
    m_phenos = sorted(cm["phenotype"].unique())

    combos = list(product(t_phenos, m_phenos))
    for tp, mp in tqdm(combos, desc=f"phenotype corrs {tissue}",
                       leave=False, unit="pair"):
        ct_p = ct[ct["phenotype"] == tp][
            ["patient", "timepoint", "frac", "n_cells_total"]
        ].rename(columns={"frac": "t_frac", "n_cells_total": "n_t_cells"})
        cm_p = cm[cm["phenotype"] == mp][
            ["patient", "timepoint", "frac", "n_cells_total"]
        ].rename(columns={"frac": "m_frac", "n_cells_total": "n_m_cells"})

        joined = ct_p.merge(cm_p, on=["patient", "timepoint"], how="inner")
        n_obs = len(joined)
        if n_obs < MIN_OBS:
            continue

        mt = joined["t_frac"].mean()
        mm = joined["m_frac"].mean()
        if mt < MIN_MEAN_FREQ or mm < MIN_MEAN_FREQ:
            continue

        w = np.minimum(joined["n_t_cells"].values, joined["n_m_cells"].values)
        rho, p = weighted_spearman(joined["t_frac"], joined["m_frac"], w)
        pheno_rows.append({
            "tissue": tissue, "t_phenotype": tp, "m_phenotype": mp,
            "n_obs": n_obs, "rho": rho, "p": p,
            "mean_t_freq": mt, "mean_m_freq": mm,
        })

pheno_df = pd.DataFrame(pheno_rows)
if len(pheno_df):
    pheno_df["q_bh"] = pheno_df.groupby("tissue")["p"].transform(
        lambda s: bh_fdr(s.values)
    )
    pheno_df = pheno_df.sort_values(
        ["tissue", "q_bh", "p"], kind="mergesort"
    ).reset_index(drop=True)
else:
    pheno_df["q_bh"] = []

# Write FULL phenotype correlations (the heatmap dims non-significant cells).
pheno_df.to_csv(OUTPUT_DIR / "phenotype_correlations.csv", index=False)
print(f"Wrote {len(pheno_df)} phenotype-correlation rows")
if len(pheno_df):
    sig_mask = (pheno_df["q_bh"] < PHENOTYPE_Q_CUTOFF) & (
        pheno_df["rho"].abs() > PHENOTYPE_RHO_CUTOFF
    )
    print(pheno_df.groupby("tissue").size().rename("n_tested").to_string())
    print(f"  Total q<{PHENOTYPE_Q_CUTOFF} & |rho|>{PHENOTYPE_RHO_CUTOFF}: "
          f"{int(sig_mask.sum())}")

# %%
# ---- (2) Pathway-score correlations (vectorized per candidate) ----
print("\n" + "=" * 60)
print("Pathway-score correlations")
print("=" * 60)

path_t = path_t[path_t["n_cells"] >= MIN_CELLS_PATHWAY].copy()
path_m = path_m[path_m["n_cells"] >= MIN_CELLS_PATHWAY].copy()

if len(pheno_df):
    candidates = pheno_df[["tissue", "t_phenotype", "m_phenotype"]].drop_duplicates().reset_index(drop=True)
else:
    candidates = pd.DataFrame(columns=["tissue", "t_phenotype", "m_phenotype"])

n_t_pathways = path_t["pathway"].nunique()
n_m_pathways = path_m["pathway"].nunique()
n_candidates = len(candidates)
print(f"n_candidates = {n_candidates}")
print(f"n_pathway_pairs_per_candidate = {n_t_pathways} * {n_m_pathways} = "
      f"{n_t_pathways * n_m_pathways}")
print(f"est_total_correlations = {n_candidates * n_t_pathways * n_m_pathways}")

path_frames = []
for cand in tqdm(candidates.itertuples(index=False), total=n_candidates,
                 desc="pathway corrs", unit="cand"):
    tissue = cand.tissue
    tp_pheno = cand.t_phenotype
    mp_pheno = cand.m_phenotype
    t0 = time.time()

    pt_sub = path_t[(path_t["tissue"] == tissue) & (path_t["phenotype"] == tp_pheno)]
    pm_sub = path_m[(path_m["tissue"] == tissue) & (path_m["phenotype"] == mp_pheno)]
    if not len(pt_sub) or not len(pm_sub):
        continue

    t_score = pt_sub.pivot_table(
        index=["patient", "timepoint"], columns="pathway",
        values="mean_score", aggfunc="mean",
    )
    m_score = pm_sub.pivot_table(
        index=["patient", "timepoint"], columns="pathway",
        values="mean_score", aggfunc="mean",
    )
    t_ncells = pt_sub.groupby(["patient", "timepoint"], observed=True)["n_cells"].max()
    m_ncells = pm_sub.groupby(["patient", "timepoint"], observed=True)["n_cells"].max()

    common_idx = t_score.index.intersection(m_score.index)
    n_obs = len(common_idx)
    if n_obs < MIN_OBS:
        continue

    X = t_score.loc[common_idx].to_numpy(dtype=float)
    Y = m_score.loc[common_idx].to_numpy(dtype=float)
    t_keep = ~(np.all(np.isnan(X), axis=0) | (np.nanstd(X, axis=0) == 0))
    m_keep = ~(np.all(np.isnan(Y), axis=0) | (np.nanstd(Y, axis=0) == 0))
    if not t_keep.any() or not m_keep.any():
        continue
    X = np.nan_to_num(X[:, t_keep], nan=0.0)
    Y = np.nan_to_num(Y[:, m_keep], nan=0.0)
    t_pw_names = t_score.columns[t_keep].to_numpy()
    m_pw_names = m_score.columns[m_keep].to_numpy()

    w = np.minimum(
        t_ncells.loc[common_idx].to_numpy(dtype=float),
        m_ncells.loc[common_idx].to_numpy(dtype=float),
    )
    if w.sum() <= 0:
        continue

    rho, p = vectorized_weighted_spearman(X, Y, w)
    n_t, n_m = rho.shape
    df_cand = pd.DataFrame({
        "tissue": tissue,
        "t_phenotype": tp_pheno,
        "m_phenotype": mp_pheno,
        "t_pathway": np.repeat(t_pw_names, n_m),
        "m_pathway": np.tile(m_pw_names, n_t),
        "n_obs": n_obs,
        "rho": rho.ravel(),
        "p": p.ravel(),
    })
    path_frames.append(df_cand)
    elapsed = time.time() - t0
    tqdm.write(f"  {tp_pheno} | {mp_pheno} | {tissue}: {n_t}x{n_m} pathways, "
               f"n_obs={n_obs}, {elapsed:.2f}s")

if path_frames:
    path_corr_df = pd.concat(path_frames, ignore_index=True)
else:
    path_corr_df = pd.DataFrame(columns=[
        "tissue", "t_phenotype", "m_phenotype", "t_pathway", "m_pathway",
        "n_obs", "rho", "p",
    ])

# Global BH within tissue.
if len(path_corr_df):
    path_corr_df["q_bh"] = path_corr_df.groupby("tissue")["p"].transform(
        lambda s: bh_fdr(s.values)
    )
else:
    path_corr_df["q_bh"] = []

# Provenance summary BEFORE filter.
if len(path_corr_df):
    summary_rows = []
    for tissue, g in path_corr_df.groupby("tissue"):
        q = g["q_bh"].values
        rho_abs = g["rho"].abs().values
        summary_rows.append({
            "tissue": tissue,
            "n_tested": len(g),
            "n_q01": int((q < 0.01).sum()),
            "n_q05": int((q < 0.05).sum()),
            "n_q10": int((q < 0.10).sum()),
            "n_q25": int((q < 0.25).sum()),
            "n_significant_written": int(((q < PATHWAY_Q_CUTOFF) & (rho_abs > PATHWAY_RHO_CUTOFF)).sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
else:
    summary_df = pd.DataFrame(columns=[
        "tissue", "n_tested", "n_q01", "n_q05", "n_q10", "n_q25", "n_significant_written",
    ])
summary_df.to_csv(OUTPUT_DIR / "pathway_correlations_summary.csv", index=False)
print(f"\nWrote pathway_correlations_summary.csv ({len(summary_df)} tissue rows)")
if len(summary_df):
    print(summary_df.to_string(index=False))

# Filter to publishable significant edges.
n_before = len(path_corr_df)
path_corr_df = path_corr_df[
    (path_corr_df["q_bh"] < PATHWAY_Q_CUTOFF)
    & (path_corr_df["rho"].abs() > PATHWAY_RHO_CUTOFF)
].reset_index(drop=True)
if len(path_corr_df):
    path_corr_df = path_corr_df.sort_values(
        ["tissue", "q_bh", "p"], kind="mergesort"
    ).reset_index(drop=True)
path_corr_df.to_csv(OUTPUT_DIR / "pathway_correlations.csv", index=False)
print(f"Pathway corrs: {n_before} -> {len(path_corr_df)} after "
      f"q<{PATHWAY_Q_CUTOFF} and |rho|>{PATHWAY_RHO_CUTOFF}")
if len(path_corr_df):
    print(path_corr_df.groupby("tissue").size().rename("n_significant").to_string())

# %%
# ---- Heatmap: PBMC + Tumor, q>=cutoff dimmed, sig dotted ----
print("\nBuilding phenotype correlation heatmap...")

present = [t for t in HEATMAP_TISSUES if t in pheno_df["tissue"].unique()]
if not present or not len(pheno_df):
    print("  No PBMC/Tumor data — skipping heatmap.")
else:
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 6),
                             squeeze=False)
    axes = axes[0]
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = plt.get_cmap("RdBu_r")
    im = None

    for ax, tissue in zip(axes, present):
        sub = pheno_df[pheno_df["tissue"] == tissue]
        if not len(sub):
            ax.set_visible(False)
            continue

        rho_mat = sub.pivot_table(
            index="t_phenotype", columns="m_phenotype",
            values="rho", aggfunc="mean",
        )
        q_mat = sub.pivot_table(
            index="t_phenotype", columns="m_phenotype",
            values="q_bh", aggfunc="min",
        ).reindex_like(rho_mat)

        rho_vals = rho_mat.values
        q_vals = q_mat.values
        sig_mask = (q_vals < PHENOTYPE_Q_CUTOFF) & (np.abs(rho_vals) > PHENOTYPE_RHO_CUTOFF)

        ax.imshow(rho_vals, cmap=cmap, norm=norm, alpha=0.25,
                  aspect="auto", interpolation="nearest")
        masked = np.where(sig_mask, rho_vals, np.nan)
        im = ax.imshow(masked, cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")
        ys, xs = np.where(sig_mask)
        ax.scatter(xs, ys, s=18, c="black", marker="o", zorder=3)

        ax.set_xticks(np.arange(rho_mat.shape[1]))
        ax.set_xticklabels(rho_mat.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(rho_mat.shape[0]))
        ax.set_yticklabels(rho_mat.index)
        ax.set_xlabel("Myeloid group")
        ax.set_ylabel("T phenotype")
        ax.set_title(f"{tissue}  (q<{PHENOTYPE_Q_CUTOFF}, |rho|>{PHENOTYPE_RHO_CUTOFF})")

    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="weighted Spearman rho")
    fig.suptitle("Cross-lineage phenotype-frequency correlations (myeloid regrouped)")
    fig_path = OUTPUT_DIR / "phenotype_corr_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_path}")
