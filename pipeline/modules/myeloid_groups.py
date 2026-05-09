"""Coarse myeloid phenotype grouping shared across pipeline scripts.

Maps fine-grained myeloid annotations onto a smaller set of biologically
coherent groups for downstream pooling (composition, pathway scores,
LIANA inputs, etc.). Phenotypes not in the map (mast cells, immature
myeloid, REMOVE labels) are intentionally absent and the helpers drop
those rows.
"""
from __future__ import annotations

import pandas as pd

MYELOID_GROUPS: dict[str, str] = {
    # Microglia
    "Mg_Complement_Immunosupp": "Microglia",
    "Mg_Inflammatory": "Microglia",
    # Macrophage subgroups
    "Mac_Complement_Immunosupp": "Mac_Immunosupp",
    "Mac_Scavenger_Immunosupp": "Mac_Immunosupp",
    "Mac_IFN": "Mac_Inflammatory",
    "Mac_Systemic_Inflammatory": "Mac_Inflammatory",
    "Mac_Hypoxia": "Mac_Hypoxia",
    "Mac_Proliferating": "Mac_Proliferating",
    # Monocyte subgroups
    "Classical monocyte": "Mono_CD14",
    "Mono_Transitional": "Mono_CD14",
    "Mono_IFN": "Mono_CD14",
    "Intermediate monocyte (CD14+CD16+)": "Mono_CD16",
    "Non-classical monocyte (CD16++)": "Mono_CD16",
    # cDC subgroups
    "cDC1": "cDC1",
    "cDC2": "cDC2",
    "DC3": "cDC2",
    "AS-DCs / transitional DCs": "cDC2",
    # Other DC populations
    "mregDC": "mregDC",
    "pre-mregDC": "mregDC",
    "moDC_TME": "moDC",
    "pDC": "pDC",
    "pre-DC": "pDC",
    # Dropped: Mast cell, immature myeloid, REMOVE Classical monocytes-not DCs
}


def regroup_composition(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a temporal-composition table onto coarse myeloid groups.

    Expects columns: patient, tissue, timepoint, phenotype, lineage,
    n_cells_phenotype, n_cells_total. Returns the same schema with
    phenotype replaced by the group label and frac recomputed.
    """
    out = df.copy()
    out["phenotype"] = out["phenotype"].map(MYELOID_GROUPS)
    out = out.dropna(subset=["phenotype"])
    out = out.groupby(
        ["patient", "tissue", "timepoint", "phenotype", "lineage"],
        as_index=False,
    ).agg(
        n_cells_phenotype=("n_cells_phenotype", "sum"),
        n_cells_total=("n_cells_total", "first"),
    )
    out["frac"] = out["n_cells_phenotype"] / out["n_cells_total"]
    return out


def regroup_pathway_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a temporal-pathway-score table onto coarse myeloid groups.

    Expects columns: patient, tissue, timepoint, phenotype, lineage,
    pathway, mean_score, n_cells. Computes a proper weighted mean of
    mean_score across the pooled sub-phenotypes, weighted by n_cells.
    """
    out = df.copy()
    out["phenotype"] = out["phenotype"].map(MYELOID_GROUPS)
    out = out.dropna(subset=["phenotype"])
    out["_weighted"] = out["mean_score"] * out["n_cells"]
    out = out.groupby(
        ["patient", "tissue", "timepoint", "phenotype", "lineage", "pathway"],
        as_index=False,
    ).agg(_weighted=("_weighted", "sum"), n_cells=("n_cells", "sum"))
    out["mean_score"] = out["_weighted"] / out["n_cells"]
    return out.drop(columns="_weighted")


def regroup_obs(obs: pd.DataFrame, col: str = "phenotype") -> pd.Series:
    """Map an AnnData.obs column to coarse groups; unmapped entries -> NaN."""
    return obs[col].map(MYELOID_GROUPS)
