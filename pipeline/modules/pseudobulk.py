"""Pseudobulk construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def _extract_matrix(adata_slice, layer):
    """Return dense numpy array from adata slice (layer or .X)."""
    x = adata_slice.layers[layer] if layer in adata_slice.layers else adata_slice.X
    if sparse.issparse(x):
        x = x.toarray()
    return x


def pseudobulk_mean_expression(adata, groupby_keys, min_cells=10):
    """Return per-group metadata and mean-expression matrix."""
    records = []
    for group, idx in adata.obs.groupby(groupby_keys).groups.items():
        if len(idx) < min_cells:
            continue
        x = adata[idx].X
        if sparse.issparse(x):
            x = x.toarray()
        mean_expr = x.mean(axis=0)
        rec = (
            dict(zip(groupby_keys, group))
            if isinstance(group, tuple)
            else {groupby_keys[0]: group}
        )
        rec["n_cells"] = len(idx)
        rec["expr"] = mean_expr
        records.append(rec)

    pb_df = pd.DataFrame([{k: v for k, v in r.items() if k != "expr"} for r in records])
    expr_mat = np.vstack([r["expr"] for r in records]) if records else np.empty((0, adata.n_vars))
    return pb_df, expr_mat


def pseudobulk_counts_by_group(
    adata,
    group_key="tissue",
    patient_key="patient",
    layer="counts",
):
    """Aggregate integer counts by group and patient."""
    pb_records = []
    obs_records = []
    for (group, patient), idx in adata.obs.groupby([group_key, patient_key]).groups.items():
        x = _extract_matrix(adata[idx], layer)
        pb_records.append(x.sum(axis=0))
        obs_records.append({group_key: group, patient_key: patient, "n_cells": len(idx)})

    counts = pd.DataFrame(np.vstack(pb_records), columns=adata.var_names)
    meta = pd.DataFrame(obs_records)
    meta.index = [f"{r[group_key]}_{r[patient_key]}" for _, r in meta.iterrows()]
    counts.index = meta.index
    counts = counts.loc[:, counts.sum() > 0].astype(int)
    return counts, meta


def pseudobulk_counts_by_clone_tissue(
    adata,
    clone_key="trb",
    tissues=("PBMC", "TP"),
    lineage="CD8",
    min_cells=3,
    layer="counts",
    patient_key="patient",
    phenotype_key="phenotype",
):
    """Pseudobulk raw counts per (clone x tissue) for shared clones.

    Parameters
    ----------
    adata : AnnData with obs columns [clone_key, "tissue", patient_key, phenotype_key].
    tissues : 2-tuple of tissue labels; only cells in these tissues are used.
    lineage : "CD8" or "CD4"; filters on phenotype_key containing this string.
    min_cells : minimum cells per (clone, tissue) to keep.

    Returns
    -------
    counts : DataFrame (n_pseudobulks x n_genes), index = "{clone}__{tissue}".
    meta   : DataFrame with columns [clone, tissue, patient, n_cells], same index.
    """
    from .clone_helpers import infer_lineage_from_phenotype

    sub = adata[
        adata.obs["tissue"].isin(tissues)
        & adata.obs[phenotype_key].map(infer_lineage_from_phenotype).eq(lineage)
    ].copy()

    t1, t2 = tissues
    pb_records, obs_records = [], []

    for clone, clone_idx in sub.obs.groupby(clone_key).groups.items():
        clone_obs = sub.obs.loc[clone_idx]
        cells_t1 = clone_obs[clone_obs["tissue"] == t1].index
        cells_t2 = clone_obs[clone_obs["tissue"] == t2].index
        if len(cells_t1) < min_cells or len(cells_t2) < min_cells:
            continue
        patient = clone_obs[patient_key].iloc[0]
        for tissue, cell_idx in [(t1, cells_t1), (t2, cells_t2)]:
            x = _extract_matrix(sub[cell_idx], layer)
            pb_records.append(x.sum(axis=0))
            obs_records.append({
                "clone": clone, "tissue": tissue,
                "patient": patient, "n_cells": len(cell_idx),
            })

    if not pb_records:
        empty_counts = pd.DataFrame(columns=adata.var_names)
        empty_meta = pd.DataFrame(columns=["clone", "tissue", "patient", "n_cells"])
        return empty_counts, empty_meta

    counts = pd.DataFrame(np.vstack(pb_records), columns=sub.var_names)
    meta = pd.DataFrame(obs_records)
    meta.index = [f"{r['clone']}__{r['tissue']}" for _, r in meta.iterrows()]
    counts.index = meta.index
    counts = counts.loc[:, counts.sum() > 0].astype(int)
    return counts, meta

