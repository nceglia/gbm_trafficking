#%%
import scanpy as sc
adata = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/adata_CSF_Tumor_Plasma_ensembl_rebuilt_full_scANVI_myeloidSubset.h5ad")

# %%
print(adata.obs["timepoint"].value_counts())
# %%

adata.X.max()
# %%
adata.X.min()
# %%
adata = adata.copy()
adata.obs["timepoint"] = adata.obs["timepoint"].astype(int).astype(str)
adata.obs["timepoint"].unique()
# %%
adata.X = adata.layers["log_norm"]
adata = adata.copy()
# %%
adata = adata.copy()
adata.obs["patient"] = adata.obs["subject_trial_id"]
adata.obs["tissue"] = adata.obs["sample_type"]
# %%
ct = []
for x in adata.obs['May_8_annot'].tolist():
    if x == "NEEDS ANNOT -Classical monocyte":
        ct.append("Classical monocyte")
    elif x == 'REMOVE Classical monocyte-not DCs':
        ct.append("Classical monocyte")
    else:
        ct.append(x)
adata = adata.copy()
adata.obs["phenotype"] = ct
# %%
#adata.obs["celltype_fine_Myeloid_v2"].unique()
adata.write("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/MYELOID_GBM.h5ad")
# %%