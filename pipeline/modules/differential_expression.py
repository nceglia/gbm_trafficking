"""Differential-expression helpers for tissue-level analyses."""

from __future__ import annotations

from .constants import TISSUES
from .pseudobulk import pseudobulk_counts_by_group
from itertools import combinations


def run_deseq2(counts, meta, design, contrast):
    """Run DESeq2 on pre-built count matrix and sample metadata.

    Parameters
    ----------
    counts : DataFrame  (samples × genes, integer counts)
    meta   : DataFrame  (samples × covariates, index matches counts)
    design : str        e.g. "~ tissue" or "~ patient + tissue"
    contrast : list     e.g. ["tissue", "TP", "PBMC"]

    Returns
    -------
    DataFrame with log2FoldChange, pvalue, padj, stat columns.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, refit_cooks=True)
    dds.deseq2()
    stat = DeseqStats(dds, contrast=contrast)
    stat.summary()
    return stat.results_df.dropna().sort_values("stat", ascending=False)


def pseudobulk_deseq2(adata, group_key="tissue", patient_key="patient", ref="PBMC"):
    """Run pairwise DESeq2 contrasts from pseudobulked counts."""
    counts, meta = pseudobulk_counts_by_group(
        adata,
        group_key=group_key,
        patient_key=patient_key,
        layer="counts",
    )

    results = {}
    for t1, t2 in combinations(TISSUES, 2):
        sub_meta = meta[meta[group_key].isin([t1, t2])].copy()
        sub_counts = counts.loc[sub_meta.index]
        results[f"{t1}_vs_{t2}"] = run_deseq2(
            sub_counts, sub_meta, f"~{group_key}", [group_key, t1, t2],
        )

    return results

