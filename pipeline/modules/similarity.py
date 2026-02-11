"""Expression-similarity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def tissue_distances_per_phenotype(pb_df, expr_mat, tissue_pairs):
    """Compute matched and aggregate cosine distances for each phenotype/tissue pair."""
    records = []
    for pheno in pb_df["phenotype"].unique():
        pheno_mask = pb_df["phenotype"] == pheno
        for t1, t2 in tissue_pairs:
            idx1 = pb_df.index[(pheno_mask) & (pb_df["tissue"] == t1)]
            idx2 = pb_df.index[(pheno_mask) & (pb_df["tissue"] == t2)]
            if len(idx1) == 0 or len(idx2) == 0:
                continue

            patients1 = set(pb_df.loc[idx1, "patient"])
            patients2 = set(pb_df.loc[idx2, "patient"])
            shared_patients = patients1 & patients2
            for patient in shared_patients:
                i1 = pb_df.index[
                    (pheno_mask) & (pb_df["tissue"] == t1) & (pb_df["patient"] == patient)
                ]
                i2 = pb_df.index[
                    (pheno_mask) & (pb_df["tissue"] == t2) & (pb_df["patient"] == patient)
                ]
                if len(i1) == 1 and len(i2) == 1:
                    dist = cosine(expr_mat[i1[0]], expr_mat[i2[0]])
                    records.append(
                        {
                            "phenotype": pheno,
                            "tissue_pair": f"{t1}_vs_{t2}",
                            "patient": patient,
                            "cosine_dist": dist,
                            "type": "matched",
                        }
                    )

            dists = [cosine(expr_mat[i], expr_mat[j]) for i in idx1 for j in idx2]
            records.append(
                {
                    "phenotype": pheno,
                    "tissue_pair": f"{t1}_vs_{t2}",
                    "patient": "aggregate",
                    "cosine_dist": np.mean(dists),
                    "type": "aggregate",
                }
            )
    return pd.DataFrame(records)

