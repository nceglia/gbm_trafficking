# %%
import palantir
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# %% Subset to CD8 only
adata = sc.read("GBM_TCR_POS_TCELLS.h5ad")
cd8 = adata[adata.obs["phenotype_level1"] == "CD8"].copy()

# %% Palantir needs its own diffusion map components
# Use existing PCA or recompute
sc.pp.pca(cd8, n_comps=30)
dm_res = palantir.utils.run_diffusion_maps(cd8, n_components=10)
ms_data = palantir.utils.determine_multiscale_space(cd8)

# %% Pick root cell — highest Naive score
naive_mask = cd8.obs["phenotype"].str.contains("Naive")
naive_cells = cd8[naive_mask]
# Pick the cell with highest naive marker expression
naive_score = naive_cells[:, ["CCR7", "SELL", "LEF1"]].X.toarray().mean(axis=1)
root_cell = naive_cells.obs_names[naive_score.argmax()]
print(f"Root cell: {root_cell}")

# %% Pick terminal cells — one per end state
terminal_states = {}
for state, markers in [
    ("TEXterm", ["ENTPD1", "HAVCR2", "TOX"]),
    ("TRM", ["CD69", "ITGAE", "JUN"]),
]:
    mask = cd8.obs["phenotype"].str.contains(state)
    subset = cd8[mask]
    available = [g for g in markers if g in cd8.var_names]
    score = subset[:, available].X.toarray().mean(axis=1)
    terminal_states[state] = subset.obs_names[score.argmax()]
    print(f"Terminal {state}: {terminal_states[state]}")

# %% Run Palantir
pr_res = palantir.core.run_palantir(
    cd8, root_cell,
    terminal_states=list(terminal_states.values()),
    num_waypoints=1500,
)

# Rename columns to state names
pr_res.branch_probs.columns = list(terminal_states.keys())

# Store in adata
cd8.obs["pseudotime"] = pr_res.pseudotime
cd8.obs["entropy"] = pr_res.entropy
for state in terminal_states:
    cd8.obs[f"fate_{state}"] = pr_res.branch_probs[state]

# %% Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sc.pl.umap(cd8, color="pseudotime", ax=axes[0, 0], show=False,
           title="Pseudotime", cmap="viridis")
sc.pl.umap(cd8, color="entropy", ax=axes[0, 1], show=False,
           title="Differentiation entropy")
sc.pl.umap(cd8, color="phenotype", ax=axes[0, 2], show=False,
           title="Phenotype")

sc.pl.umap(cd8, color="fate_TEXterm", ax=axes[1, 0], show=False,
           title="P(TEXterm)", cmap="Reds")
sc.pl.umap(cd8, color="fate_TRM", ax=axes[1, 1], show=False,
           title="P(TRM)", cmap="Greens")

# Fate probability by phenotype
fate_by_pheno = cd8.obs.groupby("phenotype")[["fate_TEXterm", "fate_TRM"]].mean()
fate_by_pheno = fate_by_pheno.loc[[x for x in fate_by_pheno.index if "CD8" in x]]
fate_by_pheno.plot.barh(ax=axes[1, 2], color=["#E8A435", "#6EAA5E"])
axes[1, 2].set_title("Mean fate probability by phenotype")
axes[1, 2].set_xlabel("Probability")

plt.tight_layout()
plt.show()

# %% Pseudotime by phenotype — should increase along trajectory
print("\nMean pseudotime by phenotype:")
print(cd8.obs.groupby("phenotype")["pseudotime"].mean().sort_values())

# %% Fate probs by phenotype × tissue
print("\nFate probability by state × tissue:")
cd8.obs["state_tissue"] = [f"{p}_{t}" for p, t
    in zip(cd8.obs["phenotype"].astype(str).values,
           cd8.obs["tissue"].astype(str).values)]
print(cd8.obs.groupby("state_tissue")[["fate_TEXterm", "fate_TRM"]].mean()
      .sort_values("fate_TEXterm", ascending=False).round(3))
# %%
sc.pp.neighbors(adata)
sc.tl.paga(adata, groups='phenotype')
sc.pl.paga(adata)
# %%
tumor_cd8 = cd8[cd8.obs["tissue"] == "TP"].copy()

sc.pp.pca(tumor_cd8, n_comps=30)
dm_res = palantir.utils.run_diffusion_maps(tumor_cd8, n_components=10)
ms_data = palantir.utils.determine_multiscale_space(tumor_cd8)

# Root: TEXprog cell (earliest in tumor)
prog_mask = tumor_cd8.obs["phenotype"].str.contains("TEXprog")
prog_cells = tumor_cd8[prog_mask]
prog_score = prog_cells[:, ["TCF7", "IL7R", "SELL"]].X.toarray().mean(axis=1)
root_cell = prog_cells.obs_names[prog_score.argmax()]

# Terminals: TEXterm and TRM
terminal_states = {}
for state, markers in [
    ("TEXterm", ["ENTPD1", "HAVCR2", "TOX"]),
    ("TRM", ["CD69", "ITGAE", "JUN"]),
]:
    mask = tumor_cd8.obs["phenotype"].str.contains(state)
    subset = tumor_cd8[mask]
    available = [g for g in markers if g in tumor_cd8.var_names]
    score = subset[:, available].X.toarray().mean(axis=1)
    terminal_states[state] = subset.obs_names[score.argmax()]

pr_res = palantir.core.run_palantir(
    tumor_cd8, root_cell,
    terminal_states=list(terminal_states.values()),
    num_waypoints=1500,
)

pr_res.branch_probs.columns = list(terminal_states.keys())
tumor_cd8.obs["pseudotime"] = pr_res.pseudotime
tumor_cd8.obs["entropy"] = pr_res.entropy
for state in terminal_states:
    tumor_cd8.obs[f"fate_{state}"] = pr_res.branch_probs[state]

# %% Visualize
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sc.pl.umap(tumor_cd8, color="pseudotime", ax=axes[0, 0], show=False, cmap="viridis")
sc.pl.umap(tumor_cd8, color="entropy", ax=axes[0, 1], show=False)
sc.pl.umap(tumor_cd8, color="phenotype", ax=axes[0, 2], show=False)
sc.pl.umap(tumor_cd8, color="fate_TEXterm", ax=axes[1, 0], show=False, cmap="Reds")
sc.pl.umap(tumor_cd8, color="fate_TRM", ax=axes[1, 1], show=False, cmap="Greens")

fate_by_pheno = tumor_cd8.obs.groupby("phenotype")[["fate_TEXterm", "fate_TRM"]].mean()
fate_by_pheno.plot.barh(ax=axes[1, 2], color=["#E8A435", "#6EAA5E"])
axes[1, 2].set_title("Mean fate probability (Tumor only)")
plt.tight_layout()
plt.show()

print("\nMean pseudotime by phenotype (Tumor):")
print(tumor_cd8.obs.groupby("phenotype")["pseudotime"].mean().sort_values())
# %%
