# %%
"""Figure 1B — clone-sharing network with T cell + myeloid composition pies.

Renders a 6-timepoint × 3-tissue grid of nodes; each node carries a small
T-cell and a small myeloid composition pie. Edges connect nodes that share
TCR clones and are weighted by the number of shared clones.

Loads the T-cell and myeloid h5ads via GBMDataset.from_paths, regroups the
fine myeloid phenotypes onto the coarse groups defined in
pipeline.modules.myeloid_groups (so colors line up with the centralized
MYELOID_PHENOTYPE_COLORS palette), strips any stale phenotype_colors from
.uns so init_colors_from_dataset can pull from pipeline.modules.style, and
calls trafficking.figures.panel_clone_network on a single axes.

Outputs:
  results/figure1/clone_network.png  (dpi=200)
  results/figure1/clone_network.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from trafficking.dataset import GBMDataset
from trafficking.figures import panel_clone_network
from trafficking.style import init_colors_from_dataset, COLORS

from pipeline.modules.myeloid_groups import MYELOID_GROUPS

# %%
# ---- Config ----
DATA_DIR = REPO_ROOT / "data" / "objects"
TCELL_PATH = DATA_DIR / "GBM_TCR_POS_TCELLS.h5ad"
MYELOID_PATH = DATA_DIR / "MYELOID_GBM.h5ad"

OUTPUT_DIR = REPO_ROOT / "results" / "figure1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ---- Load ----
print("Loading T cell + myeloid AnnDatas via GBMDataset...")
ds = GBMDataset.from_paths(tdata_path=str(TCELL_PATH), mdata_path=str(MYELOID_PATH))

# Strip any stale phenotype_colors from .uns so init_colors_from_dataset
# regenerates from the centralized style modules.
for ad_ in (ds.tdata, ds.mdata):
    if ad_ is not None and "phenotype_colors" in ad_.uns:
        del ad_.uns["phenotype_colors"]

# Regroup myeloid fine -> coarse using pipeline.modules.myeloid_groups so
# the 25 fine phenotypes collapse to the 12 coarse groups defined in
# pipeline.modules.style.MYELOID_PHENOTYPE_COLORS. Phenotypes not in the
# map (REMOVE labels, mast cells, immature myeloid) are intentionally
# dropped per the myeloid_groups docstring.
n_before = len(ds.mdata)
ds.mdata.obs["phenotype"] = ds.mdata.obs["phenotype"].map(MYELOID_GROUPS)
ds.mdata = ds.mdata[ds.mdata.obs["phenotype"].notna()].copy()
ds.mdata.obs["phenotype"] = ds.mdata.obs["phenotype"].astype(str)
n_after = len(ds.mdata)
print(f"Myeloid regrouping: {n_before:,} -> {n_after:,} cells "
      f"({n_before - n_after:,} dropped as unmapped)")

# Pull colors from the centralized palette in pipeline.modules.style.
init_colors_from_dataset(ds)

t_phenos = ds.phenotypes("tcell")
m_phenos = ds.phenotypes("myeloid")
print(f"T cell phenotypes  : {len(t_phenos)} -> {t_phenos}")
print(f"Myeloid phenotypes : {len(m_phenos)} -> {m_phenos}")

# Sanity-check that every phenotype now has a non-fallback color.
missing_t = [p for p in t_phenos if p not in COLORS["tcell_phenotype"]]
missing_m = [p for p in m_phenos if p not in COLORS["myeloid_phenotype"]]
if missing_t or missing_m:
    raise RuntimeError(
        f"Phenotypes without color: T={missing_t} / M={missing_m}"
    )

# %%
# ---- Render ----
# pie_r_t=0.13 / pie_r_m=0.08 are the panel_clone_network defaults; bumped
# slightly here because we have fewer phenotypes per pie now (T=11, M=12)
# so the wedges read more clearly at a slightly larger radius.
fig, ax = plt.subplots(figsize=(16, 7.7), dpi=200, facecolor="white")
panel_clone_network(
    ax, ds,
    pie_r_t=0.16,        # default 0.13
    pie_r_m=0.10,        # default 0.08
    x_spacing=2.4,       # wider columns so legends don't crowd the grid
    edge_alpha=0.35,     # default 0.4
)

png = OUTPUT_DIR / "clone_network.png"
pdf = OUTPUT_DIR / "clone_network.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
plt.close(fig)

# %%
# ---- Summary ----
# panel_clone_network builds its graph internally and doesn't return it;
# rebuild the lightweight version here just to print stats.
G = nx.Graph()
clone_sets = {}
for tis in ds.tissues:
    for tp in ds.timepoints:
        node = (tis, tp)
        G.add_node(node)
        clone_sets[node] = ds.get_clone_set(tissue=tis, timepoint=tp)
nodes = list(G.nodes())
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        sh = len(clone_sets[nodes[i]] & clone_sets[nodes[j]])
        if sh > 0:
            G.add_edge(nodes[i], nodes[j], shared=sh)

n_nonempty = sum(1 for s in clone_sets.values() if len(s) > 0)
n_edges = G.number_of_edges()
max_shared = max((G[u][v]["shared"] for u, v in G.edges()), default=0)

print()
print("=" * 60)
print("DONE: Figure 1B clone network")
print("=" * 60)
print(f"  Nodes drawn         : {len(nodes)} "
      f"({len(ds.tissues)} tissues x {len(ds.timepoints)} timepoints)")
print(f"  Nodes with clones   : {n_nonempty}")
print(f"  Edges               : {n_edges}")
print(f"  Max shared clones   : {max_shared}")
print(f"  T cell phenotypes   : {len(t_phenos)}")
print(f"  Myeloid phenotypes  : {len(m_phenos)}")
print(f"  PNG: {png}")
print(f"  PDF: {pdf}")
