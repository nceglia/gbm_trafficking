"""Differential-expression helpers for tissue-level analyses."""

from __future__ import annotations

from .constants import TISSUES
from .pseudobulk import pseudobulk_counts_by_group
from itertools import combinations

def pseudobulk_deseq2(adata, group_key="tissue", patient_key="patient", ref="PBMC"):
    """Run pairwise DESeq2 contrasts from pseudobulked counts."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

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

            dds = DeseqDataSet(
                counts=sub_counts,
                metadata=sub_meta,
                design=f"~{group_key}",
                refit_cooks=True,
            )
            dds.deseq2()
            stat = DeseqStats(dds, contrast=[group_key, t1, t2])
            stat.summary()
            res = stat.results_df.dropna().sort_values("stat", ascending=False)
            results[f"{t1}_vs_{t2}"] = res

    return results

