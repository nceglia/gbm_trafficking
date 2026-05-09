# %%
"""Cross-lineage (T vs myeloid) correlations of phenotype frequency
and pathway scores across patients and timepoints, per tissue.

Inputs come from script 10's temporal aggregations. For each tissue we
correlate:
  (1) phenotype-frequency T vs M, paired on (patient, timepoint)
  (2) pathway-score (T-phenotype, T-pathway) vs (M-phenotype, M-pathway),
      paired on (patient, timepoint)

Correlations use weighted Spearman (ranks weighted by min cell count
across the two lineages in each pair) and BH-FDR per tissue. Candidates
with q_bh < 0.25 are what script 12's LIANA filter consumes.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as tdist
from itertools import product
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

# %%
# ---- Config ----
INPUT_DIR = REPO_ROOT / "results" / "10_temporal_scores"
OUTPUT_DIR = REPO_ROOT / "results" / "11_cross_lineage_correlations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_OBS = 6                # paired (patient, timepoint) samples
MIN_MEAN_FREQ = 0.01       # drop phenotypes that are <1% on average
MIN_CELLS_PATHWAY = 3      # pathway-score row must have >=3 cells

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
    wsum = w.sum()
    if wsum <= 0:
        return np.nan, np.nan
    mx = np.average(rx, weights=w)
    my = np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    vx = np.average((rx - mx) ** 2, weights=w)
    vy = np.average((ry - my) ** 2, weights=w)
    if vx <= 0 or vy <= 0:
        return np.nan, np.nan
    rho = cov / np.sqrt(vx * vy)
    rho = float(np.clip(rho, -1.0, 1.0))
    n = len(x)
    if n <= 2 or abs(rho) >= 1.0:
        p = 0.0 if abs(rho) >= 1.0 else np.nan
    else:
        t_stat = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
        p = 2.0 * (1.0 - tdist.cdf(abs(t_stat), n - 2))
    return rho, float(p)


def vectorized_weighted_spearman(X, Y, w):
    """All pairwise weighted Spearman correlations between columns of X and Y.

    X: (n_obs, n_t)  Y: (n_obs, n_m)  w: (n_obs,)
    Returns rho (n_t, n_m), p (n_t, n_m).
    """
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

print(f"T composition: {len(comp_t)} rows; M composition: {len(comp_m)} rows")
print(f"T pathway:     {len(path_t)} rows; M pathway:     {len(path_m)} rows")

# Tag any missing lineage column for safety.
if "lineage" not in comp_t.columns:
    comp_t["lineage"] = "T"
if "lineage" not in comp_m.columns:
    comp_m["lineage"] = "Myeloid"

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
with tqdm(total=len(patient_obs), desc="patient observations", leave=False, unit="row") as pb:
    for i, start in enumerate(range(0, len(patient_obs), _chunk)):
        end = start + _chunk
        patient_obs.iloc[start:end].to_csv(
            _obs_path,
            mode="w" if i == 0 else "a",
            header=(i == 0),
            index=False,
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
            "tissue": tissue,
            "t_phenotype": tp,
            "m_phenotype": mp,
            "n_obs": n_obs,
            "rho": rho,
            "p": p,
            "mean_t_freq": mt,
            "mean_m_freq": mm,
        })

pheno_df = pd.DataFrame(pheno_rows)
if len(pheno_df):
    pheno_df["q_bh"] = pheno_df.groupby("tissue")["p"].transform(lambda s: bh_fdr(s.values))
    pheno_df = pheno_df.sort_values(["tissue", "q_bh", "p"], kind="mergesort").reset_index(drop=True)
else:
    pheno_df["q_bh"] = []

pheno_df.to_csv(OUTPUT_DIR / "phenotype_correlations.csv", index=False)
print(f"Wrote {len(pheno_df)} phenotype-correlation rows")
if len(pheno_df):
    n_sig = (pheno_df["q_bh"] < 0.25).sum()
    print(f"  q_bh < 0.25: {n_sig}")
    print(pheno_df.groupby("tissue").size().rename("n_tests").to_string())

# %%
# ---- (2) Pathway-score correlations ----
print("\n" + "=" * 60)
print("Pathway-score correlations")
print("=" * 60)

# Filter low-cell pathway rows up front to keep things stable.
path_t = path_t[path_t["n_cells"] >= MIN_CELLS_PATHWAY].copy()
path_m = path_m[path_m["n_cells"] >= MIN_CELLS_PATHWAY].copy()

# Candidate (t_phenotype, m_phenotype, tissue) triples come from step 1's
# phenotype-correlation table — we only score pathway pairs for phenotype
# pairs that survived the frequency filter.
if len(pheno_df):
    candidates = pheno_df[["tissue", "t_phenotype", "m_phenotype"]].drop_duplicates().reset_index(drop=True)
else:
    candidates = pd.DataFrame(columns=["tissue", "t_phenotype", "m_phenotype"])

n_t_pathways = path_t["pathway"].nunique()
n_m_pathways = path_m["pathway"].nunique()
n_candidates = len(candidates)
n_pathway_pairs_per_candidate = n_t_pathways * n_m_pathways
est_total_correlations = n_candidates * n_pathway_pairs_per_candidate
print(f"n_candidates = {n_candidates}")
print(f"n_pathway_pairs_per_candidate = {n_t_pathways} * {n_m_pathways} = {n_pathway_pairs_per_candidate}")
print(f"est_total_correlations = {est_total_correlations}")

import time

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

    # Pivot to (patient, timepoint) x pathway. Cells per (patient, timepoint)
    # are constant across pathways for a fixed (phenotype, tissue), so we
    # also pivot n_cells to recover the per-row weight after the join.
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
    # Drop pathway columns that are all-NaN or constant in this slice.
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
    tqdm.write(f"  {tp_pheno} | {mp_pheno} | {tissue}: "
               f"{n_t}x{n_m} pathways, n_obs={n_obs}, {elapsed:.2f}s")

if path_frames:
    path_corr_df = pd.concat(path_frames, ignore_index=True)
else:
    path_corr_df = pd.DataFrame(columns=[
        "tissue", "t_phenotype", "m_phenotype", "t_pathway", "m_pathway",
        "n_obs", "rho", "p",
    ])

# Global BH within tissue (overrides any per-candidate q_bh).
if len(path_corr_df):
    path_corr_df["q_bh"] = path_corr_df.groupby("tissue")["p"].transform(
        lambda s: bh_fdr(s.values)
    )
else:
    path_corr_df["q_bh"] = []

# Provenance summary BEFORE filtering: per-tissue test counts at several
# q thresholds plus what we end up writing.
PATHWAY_Q_CUTOFF = 0.10
PATHWAY_RHO_CUTOFF = 0.4

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
print(f"Pathway corrs: {n_before} -> {len(path_corr_df)} after q<{PATHWAY_Q_CUTOFF} and |rho|>{PATHWAY_RHO_CUTOFF}")

if len(path_corr_df):
    path_corr_df = path_corr_df.sort_values(
        ["tissue", "q_bh", "p"], kind="mergesort"
    ).reset_index(drop=True)

path_corr_df.to_csv(OUTPUT_DIR / "pathway_correlations.csv", index=False)
print(f"Wrote {len(path_corr_df)} pathway-correlation rows to pathway_correlations.csv")
if len(path_corr_df):
    print(path_corr_df.groupby("tissue").size().rename("n_significant").to_string())
