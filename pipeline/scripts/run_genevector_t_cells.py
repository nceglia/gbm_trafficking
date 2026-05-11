#%%
import scanpy as sc
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=150,facecolor='white')

from genevector.data import GeneVectorDataset
from genevector.model import GeneVector
from genevector.embedding import GeneEmbedding, CellEmbedding

adata = sc.read("/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS.h5ad")
adata.X = adata.layers["counts"]

# %%
adata = adata.copy()
adata = GeneVectorDataset.quality_control(adata, entropy_threshold=1.0)
adata


# %%
dataset = GeneVectorDataset(adata.copy())
# %%
from genevector.model import GeneVector

model = GeneVector(dataset, output_file="genes.vec", emb_dimension=100)
model.train(4000, threshold=1e-6)
model.plot()  # visualize convergence
# %%
embed = GeneEmbedding("genes.vec", dataset, vector="average")
# %%
embed.compute_similarities("CD69").head(20)
# %%
embed.plot_similarities("CX3CR1", n_genes=20)
# %%
#cembed = CellEmbedding(dataset, embed)
adata = cembed.get_adata()

# %%