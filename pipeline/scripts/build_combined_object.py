#%%
import scanpy as sc
import anndata as ad

t = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")
m = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/MYELOID_GBM.h5ad")
# %%
# Set X to raw counts on both
t.X = t.layers["counts"]
m.X = m.layers["counts"]

# %%
# Tag lineage
t.obs["major_lineage"] = "T"
m.obs["major_lineage"] = "Myeloid"
# %%
# Tag sublineage
print("T tissue cols:", [c for c in t.obs.columns if c == "tissue"], "count:", (t.obs.columns == "tissue").sum())
print("M tissue cols:", [c for c in m.obs.columns if c == "tissue"], "count:", (m.obs.columns == "tissue").sum())
print("M all columns:", list(m.obs.columns))
# %%
# Concat (intersect genes)
combined = ad.concat([t, m], join="inner", label="src", keys=["T", "Myeloid"])
# %%
# Normalize + log
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)
# %%
combined.write("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS_MYELOID_combined.h5ad")


# %%

# %% Th1 deep dive
import scanpy as sc
import anndata as ad
import scanpy.external as sce
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
adata.X = adata.layers["counts"]
# sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", subset=True)
th1 = adata[adata.obs["phenotype"] == "CD4_Th"].copy()
sc.pp.normalize_total(th1, target_sum=1e4)
sc.pp.log1p(th1)
print(th1.shape, th1.obs["tissue"].value_counts().to_dict())
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

for name, genes in {p: g for p, g in phenotype_marker_dict.items()}.items():
    sc.tl.score_genes(th1, [g for g in genes if g in th1.var_names], score_name=f"{name}_score")

sc.pl.umap(th1, color=["tissue","patient","TBX21","STAT1","IFNG","CXCR3",
                       "CD4","CD8A","CD8B","FOXP3","CXCL13","PDCD1","TOX","GZMB","MKI67"],
           ncols=4, frameon=False, s=8, vmin=0, vmax="p99")

score_cols_th1 = [f"{p}_score" for p in phenotype_marker_dict]
sc.pl.umap(th1, color=score_cols_th1, ncols=4, frameon=False, s=8, cmap="RdBu_r", vcenter=0)

# %% argmax check (no commit)
proposed = th1.obs[score_cols_th1].idxmax(axis=1).str.replace("_score", "")
print(proposed.value_counts())
print(pd.crosstab(proposed, th1.obs["tissue"]))
# %%
# %% v2 reassignment via clone-constrained argmax + clone-majority cleanup
def _lin(p): return "CD8" if "CD8" in str(p) else "CD4"

th1.obs["phenotype_v2"] = proposed.values

# %% merge into main object
adata.obs["phenotype_v2"] = adata.obs["phenotype"].astype(str)
adata.obs.loc[th1.obs.index, "phenotype_v2"] = th1.obs["phenotype_v2"].values
adata.obs["lineage"]    = adata.obs["phenotype"].apply(_lin)
adata.obs["lineage_v2"] = adata.obs["phenotype_v2"].apply(_lin)

# %% clone-constraint: flipped cells whose clone is unambiguously the other lineage get redirected
cd4_cols = [c for c in score_cols_th1 if c.startswith("CD4_")]
cd8_cols = [c for c in score_cols_th1 if c.startswith("CD8_")]
cd4_best = th1.obs[cd4_cols].idxmax(axis=1).str.replace("_score", "")
cd8_best = th1.obs[cd8_cols].idxmax(axis=1).str.replace("_score", "")

non_th1 = adata.obs[(~adata.obs.index.isin(th1.obs.index)) & adata.obs["trb"].notna()]
clone_lin = non_th1.groupby("trb", observed=True)["lineage"].agg(
    lambda x: x.mode().iloc[0] if len(x) else None)
cl   = th1.obs["trb"].map(clone_lin)
prop = th1.obs["phenotype_v2"].apply(_lin)

th1.obs["phenotype_v3"] = th1.obs["phenotype_v2"]
fix_cd4 = cl.notna() & (cl == "CD4") & (prop == "CD8")
fix_cd8 = cl.notna() & (cl == "CD8") & (prop == "CD4")
th1.obs.loc[fix_cd4, "phenotype_v3"] = cd4_best[fix_cd4]
th1.obs.loc[fix_cd8, "phenotype_v3"] = cd8_best[fix_cd8]
print(f"Constrained back to CD4: {fix_cd4.sum()}, to CD8: {fix_cd8.sum()}")

adata.obs["phenotype_v3"] = adata.obs["phenotype"].astype(str)
adata.obs.loc[th1.obs.index, "phenotype_v3"] = th1.obs["phenotype_v3"].values
adata.obs["lineage_v3"] = adata.obs["phenotype_v3"].apply(_lin)

# %% switching audit and Th1 fraction by tissue
def switching(obs, col, clone_key="trb"):
    sub = obs[obs[clone_key].notna()]
    return sub.groupby(clone_key, observed=True)[col].nunique()

sw_v1 = switching(adata.obs, "lineage")
sw_v2 = switching(adata.obs, "lineage_v2")
sw_v3 = switching(adata.obs, "lineage_v3")
print(f"Switching — v1: {(sw_v1 > 1).sum()}, v2: {(sw_v2 > 1).sum()}, v3: {(sw_v3 > 1).sum()}")

for col in ["phenotype", "phenotype_v2", "phenotype_v3"]:
    f = adata.obs.groupby("tissue", observed=True)[col].apply(lambda s: (s == "CD4_Th").mean())
    print(f"{col}:\n{f.round(3)}\n")
# %%
# %% v4 dry run
clone_lin_counts = adata.obs[adata.obs["trb"].notna()].groupby(
    "trb", observed=True
)["lineage_v3"].value_counts().unstack(fill_value=0)
clone_maj = clone_lin_counts.idxmax(axis=1)
adata.obs["clone_lin"] = adata.obs["trb"].map(clone_maj)

minority = (
    (adata.obs["lineage_v3"] != adata.obs["clone_lin"])
    & adata.obs["clone_lin"].notna()
    & adata.obs.index.isin(th1.obs.index)
)
n = minority.sum()
print(f"Cells changed: {n} ({n / adata.n_obs:.3%} of total)")
print(adata.obs.loc[minority].groupby("clone_lin", observed=True).size())

# %% commit v4
adata.obs["phenotype_v4"] = adata.obs["phenotype_v3"].astype(str)
to_cd4 = adata.obs.index[minority & (adata.obs["clone_lin"] == "CD4")]
to_cd8 = adata.obs.index[minority & (adata.obs["clone_lin"] == "CD8")]
adata.obs.loc[to_cd4, "phenotype_v4"] = cd4_best.loc[to_cd4].values
adata.obs.loc[to_cd8, "phenotype_v4"] = cd8_best.loc[to_cd8].values

adata.obs["lineage_v4"] = adata.obs["phenotype_v4"].apply(_lin)
print(f"Switching — v3: {(sw_v3 > 1).sum()}, v4: {(switching(adata.obs, 'lineage_v4') > 1).sum()}")
print(adata.obs.groupby("tissue", observed=True)["phenotype_v4"].apply(
    lambda s: (s == "CD4_Th").mean()).round(3))
# %%



import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

repo_root = Path("/Users/ceglian/Codebase/GitHub/gbm_trafficking")
sys.path.insert(0, str(repo_root))
from pipeline.modules.style import (
    TCELL_PHENOTYPE_COLORS, TCELL_PHENOTYPE_LABELS, TCELL_PHENOTYPE_ORDER,
    TISSUE_ORDER, TISSUE_LABELS, PATIENT_ORDER,
)

df = adata.obs[["phenotype_v4", "tissue", "timepoint", "patient"]].astype(str).rename(
    columns={"phenotype_v4": "phenotype"})

PHENOS = [p for p in TCELL_PHENOTYPE_ORDER if p in df["phenotype"].unique()]
COLORS = [TCELL_PHENOTYPE_COLORS[p] for p in PHENOS]
LABELS = [TCELL_PHENOTYPE_LABELS[p] for p in PHENOS]
TISSUES = [t for t in TISSUE_ORDER if t in df["tissue"].unique()]

def comp(keys):
    g = df.groupby(keys + ["phenotype"], observed=True).size().unstack(fill_value=0)
    g = g.reindex(columns=PHENOS, fill_value=0)
    return g.div(g.sum(axis=1), axis=0)

def bar(ax, frac, title, xlabel=""):
    x = np.arange(len(frac)); bottom = np.zeros(len(frac))
    for p, c in zip(PHENOS, COLORS):
        ax.bar(x, frac[p].values, bottom=bottom, color=c, edgecolor="white", linewidth=0.5)
        bottom += frac[p].values
    ax.set_xticks(x); ax.set_xticklabels(frac.index, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("Fraction"); ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.4, 1.4], hspace=0.45, wspace=0.3)

overall = df["phenotype"].value_counts(normalize=True).reindex(PHENOS, fill_value=0)
bar(fig.add_subplot(gs[0, 0]), overall.to_frame().T.rename(index={0: "All"}), "Overall")

t_frac = comp(["tissue"]).reindex(TISSUES)
t_frac.index = [TISSUE_LABELS[t] for t in t_frac.index]
bar(fig.add_subplot(gs[0, 1]), t_frac, "By tissue")

bar(fig.add_subplot(gs[0, 2]), comp(["timepoint"]).sort_index(), "By timepoint", "Timepoint")

for i, tis in enumerate(TISSUES):
    sub = df[df["tissue"] == tis]
    g = sub.groupby(["timepoint", "phenotype"], observed=True).size().unstack(fill_value=0)
    g = g.reindex(columns=PHENOS, fill_value=0).sort_index()
    bar(fig.add_subplot(gs[1, i]), g.div(g.sum(axis=1), axis=0),
        f"{TISSUE_LABELS[tis]} × timepoint", "Timepoint")

for i, tis in enumerate(TISSUES):
    sub = df[df["tissue"] == tis]
    g = sub.groupby(["patient", "phenotype"], observed=True).size().unstack(fill_value=0)
    g = g.reindex(columns=PHENOS, fill_value=0).reindex([p for p in PATIENT_ORDER if p in g.index])
    bar(fig.add_subplot(gs[2, i]), g.div(g.sum(axis=1), axis=0),
        f"{TISSUE_LABELS[tis]} × patient", "Patient")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
fig.legend(handles, LABELS, loc="center right", bbox_to_anchor=(1.0, 0.5),
           fontsize=9, frameon=False, title="Phenotype")
fig.suptitle("phenotype_v4 composition audit", fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 0.92, 0.99])
plt.show()



#%%
adata.obs["phenotype_v4"].unique()







# %% finalize — preserve current as v_pre_th1, overwrite phenotype, write
adata.obs["phenotype_pre_th1"] = adata.obs["phenotype"]
adata.obs["phenotype"] = adata.obs["phenotype_v4"]

for c in ["phenotype_v2","phenotype_v3","phenotype_v4",
          "lineage","lineage_v2","lineage_v3","lineage_v4","clone_lin"]:
    if c in adata.obs.columns: del adata.obs[c]

print(adata.obs["phenotype"].value_counts())

# %%
adata.write("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")
# %%
lin = adata.obs["phenotype"].apply(lambda p: "CD8" if "CD8" in str(p) else "CD4").astype("category")
adata.obs["lineage"] = lin
adata.obs["phenotype_level1"] = lin
adata.write("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")
# %%
