# %% setup
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import matplotlib.pyplot as plt
import sys
from pathlib import Path

repo_root = Path("/Users/ceglian/Codebase/GitHub/gbm_trafficking")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from pipeline.celltyping import load_markers
from pipeline.modules.clone_helpers import infer_lineage_from_phenotype

adata = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")
adata
treg = adata[adata.obs["phenotype"] == "CD4_Treg"].copy()
print(treg.shape, treg.obs["tissue"].value_counts().to_dict())
len(treg.var_names)
# %% reembed on Tregs only
# sc.pp.highly_variable_genes(treg, n_top_genes=2000, flavor="seurat")
# sc.tl.pca(treg, n_comps=30)
# sce.pp.harmony_integrate(treg, "patient")
# sc.pp.neighbors(treg, use_rep="X_pca_harmony")
# sc.tl.umap(treg)
# sc.tl.leiden(treg, resolution=0.5)
# %%
beltra_path = "/Users/ceglian/Downloads/41586_2025_9989_MOESM10_ESM.xlsx"
markers = load_markers(beltra_path)

def _build_phenotype_marker_dict(marker_tree):
    out = {}
    for lineage, lineage_info in marker_tree.items():
        lineage_genes = lineage_info.get("markers", [])
        for subtype, subtype_info in lineage_info.get("subtypes", {}).items():
            if isinstance(subtype_info, list):
                out[f"{lineage}_{subtype}"] = list(dict.fromkeys(lineage_genes + subtype_info))
                continue
            subtype_genes = subtype_info.get("markers", [])
            for leaf, leaf_genes in subtype_info.get("subtypes", {}).items():
                out[f"{lineage}_{subtype}_{leaf}"] = list(
                    dict.fromkeys(lineage_genes + subtype_genes + leaf_genes)
                )
    return out

phenotype_marker_dict = _build_phenotype_marker_dict(markers)
phenotype_marker_dict["CD8_Activated_TEMRA"] = [
    "CD8A", "CD8B", "CX3CR1", "S1PR5", "KLF2", "FGFBP2", "FCGR3A", "KLRD1"
]
print({k: len(v) for k, v in phenotype_marker_dict.items()})

# %%
for phenotype, gene_list in phenotype_marker_dict.items():
    present_genes = [g for g in gene_list if g in treg.var_names]
    if not present_genes:
        print(f"Skipping {phenotype}: no marker genes found in treg.var_names")
        continue

    sc.tl.score_genes(
        treg,
        gene_list=present_genes,
        score_name=f"{phenotype}_score",
    )

score_cols = [f"{p}_score" for p in phenotype_marker_dict if f"{p}_score" in treg.obs.columns]
treg.obs[score_cols].head()

# %%
# sc.pl.umap(treg, color=score_cols)
# %%
score_cols = [f"{p}_score" for p in phenotype_marker_dict if f"{p}_score" in treg.obs.columns]

# %% reassign by argmax over phenotype scores
treg.obs["phenotype_v2"] = treg.obs[score_cols].idxmax(axis=1).str.replace("_score", "")
print(treg.obs["phenotype_v2"].value_counts())
print(pd.crosstab(treg.obs["phenotype_v2"], treg.obs["tissue"]))
sc.pl.umap(treg, color="phenotype_v2", frameon=False, s=8)

# %% DEG validation
sc.tl.rank_genes_groups(treg, "phenotype_v2", method="wilcoxon")
sc.pl.rank_genes_groups(treg, n_genes=15, sharey=False)
sc.pl.rank_genes_groups_dotplot(treg, n_genes=5, standard_scale="var")

# %% merge back
adata.obs["phenotype_v2"] = adata.obs["phenotype"].astype(str)
adata.obs.loc[treg.obs.index, "phenotype_v2"] = treg.obs["phenotype_v2"].values
lin = lambda x: "CD8" if "CD8" in str(x) else "CD4"
adata.obs["lineage"] = adata.obs["phenotype"].apply(lin)
adata.obs["lineage_v2"] = adata.obs["phenotype_v2"].apply(lin)

# %% lineage flips in reassigned set
flipped = adata.obs.loc[treg.obs.index]
flipped = flipped[flipped["lineage"] != flipped["lineage_v2"]]
print(f"CD4_Treg → CD8 flips: {len(flipped)} / {len(treg)}")
print(flipped["phenotype_v2"].value_counts())

# %% clone-level audit on flipped cells
def clone_audit(adata, idx, clone_key="trb"):
    flipped_set = set(idx)
    rows = []
    for clone, grp in adata.obs.loc[idx].groupby(clone_key, observed=True):
        rest = adata.obs[(adata.obs[clone_key] == clone) & (~adata.obs.index.isin(flipped_set))]
        rows.append({
            "clone": clone,
            "n_flipped": len(grp),
            "new_lineage": grp["lineage_v2"].iloc[0],
            "rest_CD4": int((rest["lineage"] == "CD4").sum()),
            "rest_CD8": int((rest["lineage"] == "CD8").sum()),
        })
    df = pd.DataFrame(rows)
    df["disagrees"] = (
        ((df["new_lineage"] == "CD8") & (df["rest_CD4"] > df["rest_CD8"])) |
        ((df["new_lineage"] == "CD4") & (df["rest_CD8"] > df["rest_CD4"]))
    )
    return df

audit = clone_audit(adata, flipped.index)
print(audit["disagrees"].value_counts())
print(audit.sort_values(["rest_CD4", "rest_CD8"], ascending=False).head(20))

# %% global lineage-switching audit
def switching(obs, col, clone_key="trb"):
    sub = obs[obs[clone_key].notna()]
    return sub.groupby(clone_key)[col].nunique()

sw_v1 = switching(adata.obs, "lineage")
sw_v2 = switching(adata.obs, "lineage_v2")
new_sw = sw_v2[(sw_v2 > 1) & (sw_v1 == 1)].index
print(f"Switching clones — v1: {(sw_v1 > 1).sum()}, v2: {(sw_v2 > 1).sum()}, newly introduced: {len(new_sw)}")

# %% biggest newly-switching clones — evidence for/against real CD4/CD8 switch
sizes = adata.obs[adata.obs["trb"].isin(new_sw)].groupby("trb").size().sort_values(ascending=False)
print(sizes.head(10))

for clone in sizes.head(3).index:
    cells = adata[adata.obs["trb"] == clone]
    print(f"\n=== Clone {clone} (n={cells.n_obs}) ===")
    print(pd.crosstab(cells.obs["phenotype"], cells.obs["phenotype_v2"], margins=True))
    print(pd.crosstab(cells.obs["phenotype_v2"], cells.obs["tissue"], margins=True))
    sc.pl.violin(cells, ["CD4", "CD8A", "CD8B", "FOXP3", "GZMB"],
                 groupby="phenotype_v2", rotation=45)
# %%
# %% v3: clone-constrained reassignment
treg_set = set(treg.obs.index)
non_treg = adata.obs[(~adata.obs.index.isin(treg_set)) & adata.obs["trb"].notna()]
clone_lin = non_treg.groupby("trb", observed=True)["lineage"].agg(
    lambda x: x.mode().iloc[0] if len(x) else None
)

cd4_cols = [c for c in score_cols if c.startswith("CD4_")]
cd8_cols = [c for c in score_cols if c.startswith("CD8_")]
cd4_best = treg.obs[cd4_cols].idxmax(axis=1).str.replace("_score", "")
cd8_best = treg.obs[cd8_cols].idxmax(axis=1).str.replace("_score", "")

cl = treg.obs["trb"].map(clone_lin)
prop = treg.obs["phenotype_v2"].apply(lambda x: "CD8" if "CD8" in x else "CD4")
treg.obs["phenotype_v3"] = treg.obs["phenotype_v2"]
fix_cd4 = cl.notna() & (cl == "CD4") & (prop == "CD8")
fix_cd8 = cl.notna() & (cl == "CD8") & (prop == "CD4")
treg.obs.loc[fix_cd4, "phenotype_v3"] = cd4_best[fix_cd4]
treg.obs.loc[fix_cd8, "phenotype_v3"] = cd8_best[fix_cd8]
print(f"Constrained back to CD4: {fix_cd4.sum()}, to CD8: {fix_cd8.sum()}")
print(treg.obs["phenotype_v3"].value_counts())

# %% merge v3 and re-audit switching
adata.obs["phenotype_v3"] = adata.obs["phenotype"].astype(str)
adata.obs.loc[treg.obs.index, "phenotype_v3"] = treg.obs["phenotype_v3"].values
adata.obs["lineage_v3"] = adata.obs["phenotype_v3"].apply(lin)
sw_v3 = switching(adata.obs, "lineage_v3")
print(f"Switching clones — v2: {(sw_v2 > 1).sum()}, v3: {(sw_v3 > 1).sum()}")

# %% tumor Treg fractions
for col in ["phenotype", "phenotype_v2", "phenotype_v3"]:
    f = adata.obs.groupby("tissue")[col].apply(lambda s: (s == "CD4_Treg").mean())
    print(f"{col}:\n{f.round(3)}\n")
# %%
# %% double-positive audit
def expr(ad, g):
    if g not in ad.var_names: return np.zeros(ad.n_obs)
    x = ad[:, g].X
    return x.toarray().flatten() if hasattr(x, "toarray") else x.flatten()

THR = 0.5
cd4 = expr(adata, "CD4")
cd8 = np.maximum(expr(adata, "CD8A"), expr(adata, "CD8B"))
adata.obs["dp"] = (cd4 > THR) & (cd8 > THR)
adata.obs["cd4_only"] = (cd4 > THR) & (cd8 <= THR)
adata.obs["cd8_only"] = (cd4 <= THR) & (cd8 > THR)

print("Overall:")
print(adata.obs[["dp","cd4_only","cd8_only"]].mean().round(3))
print("\nBy tissue:")
print(adata.obs.groupby("tissue")[["dp","cd4_only","cd8_only"]].mean().round(3))
print("\nBy original phenotype:")
print(adata.obs.groupby("phenotype")[["dp","cd4_only","cd8_only"]].mean().round(3))

# %% DP rate among flipped cells, split by clone-level disagreement
flipped_idx = flipped.index
disagree_clones = set(audit[audit["disagrees"]]["clone"])
consistent_clones = set(audit[~audit["disagrees"]]["clone"])
sub = adata.obs.loc[flipped_idx]
print(f"\nDP rate — disagreeing flips: {sub[sub['trb'].isin(disagree_clones)]['dp'].mean():.3f}")
print(f"DP rate — consistent flips:  {sub[sub['trb'].isin(consistent_clones)]['dp'].mean():.3f}")
print(f"DP rate — non-flipped Tregs: {adata.obs.loc[treg.obs.index].pipe(lambda d: d[~d.index.isin(flipped_idx)])['dp'].mean():.3f}")

# %% doublet signature check — DP cells should have elevated UMIs if they're doublets
qc_col = next((c for c in ["n_genes_by_counts","n_genes","total_counts"] if c in adata.obs.columns), None)
if qc_col:
    print(f"\nMedian {qc_col} by DP status:")
    print(adata.obs.groupby("dp")[qc_col].median())
    print(f"\nMedian {qc_col} by DP × tissue:")
    print(adata.obs.groupby(["tissue","dp"])[qc_col].median())

# %% on the treg UMAP
treg.obs["dp"] = adata.obs.loc[treg.obs.index, "dp"].astype(int).values
sc.pl.umap(treg, color=["dp","CD4","CD8A","CD8B"], frameon=False, s=8, ncols=4)