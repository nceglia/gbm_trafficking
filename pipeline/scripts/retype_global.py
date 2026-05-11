"""Global re-typing of T cells via leaf scoring with clone-level constraints."""

# %% setup
import argparse
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltyping import (
    build_phenotype_marker_dict,
    load_markers,
    retype_global,
)
from modules.clone_helpers import infer_lineage_from_phenotype


# %% args
def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Global re-typing for T cells.")
    parser.add_argument(
        "--input-h5ad",
        default=str(repo_root / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"),
    )
    parser.add_argument(
        "--output-h5ad",
        default=str(repo_root / "data" / "objects" / "GBM_TCR_POS_TCELLS_retyped.h5ad"),
    )
    parser.add_argument(
        "--table-sig",
        default="/Users/ceglian/Downloads/41586_2025_9989_MOESM10_ESM.xlsx",
    )
    parser.add_argument("--clone-key", default="trb")
    return parser.parse_args()


# %% diagnostics
def _switching(pheno, obs, clone_key):
    sub = obs[obs[clone_key].notna()]
    lin = pheno.loc[sub.index].apply(infer_lineage_from_phenotype)
    return (lin.groupby(sub[clone_key], observed=True).nunique() > 1).sum()


def main(args):
    print(f"Reading {args.input_h5ad}")
    adata = sc.read(args.input_h5ad)

    markers = load_markers(args.table_sig)
    marker_dict = build_phenotype_marker_dict(markers)
    print({k: len(v) for k, v in marker_dict.items()})

    adata.obs["phenotype_v1"] = adata.obs["phenotype"].astype(str)
    new_labels = retype_global(
        adata, marker_dict, clone_key=args.clone_key, phenotype_key="phenotype"
    )
    adata.obs["phenotype"] = new_labels.values

    obs = adata.obs
    v1 = obs["phenotype_v1"]
    v2 = obs["phenotype"].astype(str)

    print("\n=== Per-tissue fraction of cells changed ===")
    print((v1 != v2).groupby(obs["tissue"], observed=True).mean().round(3))

    print("\n=== Leaf fractions by tissue (v1 vs retyped) ===")
    for leaf in ["CD4_Treg", "CD4_Th", "CD4_Exhausted"]:
        before = (v1 == leaf).groupby(obs["tissue"], observed=True).mean()
        after = (v2 == leaf).groupby(obs["tissue"], observed=True).mean()
        print(f"\n{leaf}:")
        print(pd.DataFrame({"v1": before, "retyped": after}).round(3))

    print("\n=== Top 20 v1 → retyped flips (off-diagonal) ===")
    flips = pd.crosstab(v1, v2).stack()
    off_diag = flips[flips.index.get_level_values(0) != flips.index.get_level_values(1)]
    print(off_diag.sort_values(ascending=False).head(20))

    print("\n=== Lineage-switching clones ===")
    sw_v1 = _switching(v1, obs, args.clone_key)
    sw_v2 = _switching(v2, obs, args.clone_key)
    print(f"v1: {sw_v1}, retyped: {sw_v2}")

    print("\n=== Final phenotype value counts ===")
    print(v2.value_counts())

    print(f"\nWriting {args.output_h5ad}")
    adata.write(args.output_h5ad)


if __name__ == "__main__":
    main(parse_args())
