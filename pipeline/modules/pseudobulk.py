"""Pseudobulk construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


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
        x = adata[idx].layers[layer] if layer in adata.layers else adata[idx].X
        if sparse.issparse(x):
            x = x.toarray()
        pb_records.append(x.sum(axis=0))
        obs_records.append({group_key: group, patient_key: patient, "n_cells": len(idx)})

    counts = pd.DataFrame(np.vstack(pb_records), columns=adata.var_names)
    meta = pd.DataFrame(obs_records)
    meta.index = [f"{r[group_key]}_{r[patient_key]}" for _, r in meta.iterrows()]
    counts.index = meta.index
    counts = counts.loc[:, counts.sum() > 0].astype(int)
    return counts, meta

