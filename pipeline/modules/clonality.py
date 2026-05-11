"""Clonality metrics with bootstrap CIs + per-stratum temporal trend fits.

Stratum = one combination of `group_cols`. Clone = unique TRB. All
metrics are computed on the empirical clone-size distribution within
each stratum.
"""
import warnings

import numpy as np
import pandas as pd

MIN_CELLS_FOR_BOOT = 10


def _stats_from_counts(counts, n_total):
    p = counts / n_total
    n_clones = int((counts > 0).sum())
    log_p = np.where(p > 0, np.log(p), 0.0)
    shannon = float(-(p * log_p).sum())
    if n_clones >= 2:
        pielou = shannon / np.log(n_clones)
        clonality = 1.0 - pielou
    else:
        pielou = np.nan
        clonality = np.nan
    sum_p2 = float((p ** 2).sum())
    inv_simpson = 1.0 / sum_p2 if sum_p2 > 0 else np.nan
    return n_clones, shannon, pielou, clonality, inv_simpson


def compute_clonality(df, group_cols, n_boot=200, seed=0):
    """Shannon / Pielou / clonality / inv-Simpson per stratum, with
    2.5/97.5% bootstrap CIs on `clonality` and `inv_simpson` (skipped
    when n_cells < MIN_CELLS_FOR_BOOT)."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    rng = np.random.default_rng(seed)
    rows = []

    for keys, sub in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trbs = sub["trb"].to_numpy()
        n_cells = int(len(trbs))
        if n_cells == 0:
            continue
        codes, _ = pd.factorize(trbs)
        counts = np.bincount(codes)
        n_codes = len(counts)
        n_clones, shannon, pielou, clonality, inv_simpson = _stats_from_counts(
            counts, n_cells)

        if n_cells >= MIN_CELLS_FOR_BOOT:
            boot_codes = codes[rng.integers(0, n_cells, size=(n_boot, n_cells))]
            cb = np.zeros((n_boot, n_codes), dtype=np.int32)
            np.add.at(cb, (np.arange(n_boot)[:, None], boot_codes), 1)
            p = cb / n_cells
            log_p = np.where(p > 0, np.log(p), 0.0)
            shannon_b = -(p * log_p).sum(axis=1)
            n_clones_b = (cb > 0).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                pielou_b = np.where(
                    n_clones_b >= 2,
                    shannon_b / np.log(np.maximum(n_clones_b, 2)),
                    np.nan,
                )
            clonality_b = 1.0 - pielou_b
            sum_p2 = (p ** 2).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_simpson_b = np.where(sum_p2 > 0, 1.0 / sum_p2, np.nan)
            cl_lo = float(np.nanquantile(clonality_b, 0.025))
            cl_hi = float(np.nanquantile(clonality_b, 0.975))
            iv_lo = float(np.nanquantile(inv_simpson_b, 0.025))
            iv_hi = float(np.nanquantile(inv_simpson_b, 0.975))
        else:
            cl_lo = cl_hi = iv_lo = iv_hi = np.nan

        row = dict(zip(group_cols, keys))
        row.update(dict(
            n_cells=n_cells,
            n_clones=n_clones,
            shannon=shannon,
            pielou=pielou,
            clonality=clonality,
            inv_simpson=inv_simpson,
            clonality_lo=cl_lo,
            clonality_hi=cl_hi,
            inv_simpson_lo=iv_lo,
            inv_simpson_hi=iv_hi,
        ))
        rows.append(row)

    return pd.DataFrame(rows)


def fit_temporal_trends(df, n_min=10):
    """Per (phenotype, tissue) MixedLM trend of clonality vs timepoint.

    Parameters
    ----------
    df : DataFrame
        One row per (patient, tissue, timepoint, phenotype) stratum
        with `clonality` populated.
    n_min : int
        Minimum (patient, timepoint) observations required to fit.

    Returns
    -------
    DataFrame with columns:
        phenotype, tissue, n_obs, n_patients, slope, slope_se, slope_p,
        slope_q (BH-FDR across returned rows), spearman_median,
        spearman_iqr, n_pos, n_neg, n_total, direction_agree.
    """
    import statsmodels.formula.api as smf
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    work = df.copy()
    work["_t"] = work["timepoint"].astype(int)
    work = work.dropna(subset=["clonality"])

    rows = []
    for (pheno, tissue), sub in work.groupby(["phenotype", "tissue"], observed=True):
        n_obs = len(sub)
        n_patients = sub["patient"].nunique()
        if n_obs < n_min or n_patients < 2:
            continue

        slope = slope_se = slope_p = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm("clonality ~ _t", sub, groups=sub["patient"])
                fit = model.fit(reml=True, method="lbfgs")
            slope = float(fit.params.get("_t", np.nan))
            slope_se = float(fit.bse.get("_t", np.nan))
            slope_p = float(fit.pvalues.get("_t", np.nan))
        except Exception:
            pass

        rhos = []
        for _, ps in sub.groupby("patient"):
            if len(ps) >= 3:
                rho, _ = spearmanr(ps["_t"], ps["clonality"])
                if not np.isnan(rho):
                    rhos.append(float(rho))
        n_total = len(rhos)
        n_pos = int(sum(1 for r in rhos if r > 0))
        n_neg = int(sum(1 for r in rhos if r < 0))
        n_agree = max(n_pos, n_neg) if n_total > 0 else 0
        direction_agree = (n_agree / n_total) if n_total > 0 else np.nan
        spearman_median = float(np.median(rhos)) if rhos else np.nan
        spearman_iqr = (float(np.percentile(rhos, 75) - np.percentile(rhos, 25))
                         if len(rhos) >= 2 else np.nan)

        rows.append(dict(
            phenotype=pheno, tissue=tissue,
            n_obs=int(n_obs), n_patients=int(n_patients),
            slope=slope, slope_se=slope_se, slope_p=slope_p,
            spearman_median=spearman_median, spearman_iqr=spearman_iqr,
            n_pos=n_pos, n_neg=n_neg, n_total=int(n_total),
            n_agree=int(n_agree), direction_agree=direction_agree,
        ))

    out = pd.DataFrame(rows)
    if len(out):
        out["slope_q"] = np.nan
        valid = out["slope_p"].notna()
        if valid.sum() > 0:
            _, q, _, _ = multipletests(out.loc[valid, "slope_p"].to_numpy(),
                                        method="fdr_bh")
            out.loc[valid, "slope_q"] = q
    return out
