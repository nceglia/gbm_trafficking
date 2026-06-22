# %%
"""Pathway co-enrichment network from Hallmark GSEA results.

Builds a Jaccard-similarity graph of pathways based on leading-edge gene
overlap, colored by tissue affinity and grouped into modules.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import TISSUE_COLORS
from modules.constants import DIRECTED_TISSUE_PAIRS

# ---- Constants ----
JACCARD_THRESHOLD = 0.20
FDR_THRESHOLD = 0.25
LIBRARY_TAG = "hallmark"
RANDOM_SEED = 42

LINEAGES = ("CD8", "CD4")
TISSUES = ("PBMC", "CSF", "TP")
INPUT_DIR = REPO_ROOT / "results" / "pathway_de_gsea"
OUTPUT_DIR = INPUT_DIR

# Module hull colors (categorical palette)
MODULE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def compute_tissue_affinity(gsea_long, lineages=LINEAGES, tissues=TISSUES):
    """Compute z-scored tissue affinity and return argmax tissue per (lineage, pathway).

    Reuses the same logic as 07_pathway_ternary.py Cell 2.
    TODO: deduplicate with 07 by extracting into a shared module.
    """
    records = []
    for lineage in lineages:
        lin_df = gsea_long[gsea_long["lineage"] == lineage]
        nes_pivot = lin_df.pivot_table(
            index="pathway", columns="contrast", values="NES", aggfunc="first",
        )
        for pathway in lin_df["pathway"].unique():
            if pathway not in nes_pivot.index:
                continue
            row = nes_pivot.loc[pathway]
            affinities = {}
            for tissue in tissues:
                up_vals, down_vals = [], []
                for t1, t2 in DIRECTED_TISSUE_PAIRS:
                    label = f"{lineage}_{t1}_vs_{t2}"
                    if label not in row.index or pd.isna(row[label]):
                        continue
                    nes = row[label]
                    if tissue == t2:
                        up_vals.append(nes)
                    elif tissue == t1:
                        down_vals.append(nes)
                up_mean = np.mean(up_vals) if up_vals else 0.0
                down_mean = np.mean(down_vals) if down_vals else 0.0
                affinities[tissue] = up_mean - down_mean
            records.append({"lineage": lineage, "pathway": pathway, **affinities})

    aff = pd.DataFrame(records)
    tissue_cols = list(tissues)

    # Z-score within lineage, then argmax
    for lineage in aff["lineage"].unique():
        mask = aff["lineage"] == lineage
        for col in tissue_cols:
            vals = aff.loc[mask, col]
            mu, sd = vals.mean(), vals.std()
            if sd > 0:
                aff.loc[mask, col] = (vals - mu) / sd
            else:
                aff.loc[mask, col] = 0.0

    aff["dominant_tissue"] = aff[tissue_cols].idxmax(axis=1)
    return aff


# %%
# ---- Cell 1: Load all GSEA tables into one long DataFrame ----
records = []
for lineage in LINEAGES:
    for t1, t2 in DIRECTED_TISSUE_PAIRS:
        label = f"{lineage}_{t1}_vs_{t2}"
        # Try hallmark-slug filename first, fall back to non-slug
        path = INPUT_DIR / f"gsea_{LIBRARY_TAG}_{label}.csv"
        if not path.exists():
            path = INPUT_DIR / f"gsea_{label}.csv"
        if not path.exists():
            print(f"  SKIP (not found): gsea_[{LIBRARY_TAG}_]{label}.csv")
            continue

        gsea = pd.read_csv(path)
        gsea["lineage"] = lineage
        gsea["t1"] = t1
        gsea["t2"] = t2
        gsea["contrast"] = label
        # Parse Lead_genes: semicolon-separated string → set
        gsea["Lead_genes_set"] = gsea["Lead_genes"].apply(
            lambda x: set(str(x).split(";")) if pd.notna(x) else set()
        )
        records.append(gsea)
        print(f"  Loaded {path.name}: {len(gsea)} pathways")

gsea_long = pd.concat(records, ignore_index=True)
gsea_long["pathway"] = gsea_long["Term"]
print(f"\nCombined long DataFrame: {gsea_long.shape}")

# %%
# ---- Cell 2: Filter pathways + build per-(lineage, pathway) leading-edge gene sets ----
node_records = []

for lineage in LINEAGES:
    lin = gsea_long[gsea_long["lineage"] == lineage]
    # min(FDR) < FDR_THRESHOLD within lineage
    min_fdr = lin.groupby("pathway")["FDR q-val"].min()
    passing = min_fdr[min_fdr < FDR_THRESHOLD].index

    for pathway in passing:
        pw_rows = lin[lin["pathway"] == pathway]
        sig_rows = pw_rows[pw_rows["FDR q-val"] < FDR_THRESHOLD]
        # Union of leading-edge genes across significant contrasts
        gene_union = set()
        for _, r in sig_rows.iterrows():
            gene_union |= r["Lead_genes_set"]

        # max |NES| across all contrasts
        max_abs_nes = pw_rows["NES"].abs().max()

        node_records.append({
            "lineage": lineage,
            "pathway": pathway,
            "lead_genes": gene_union,
            "max_abs_nes": max_abs_nes,
        })

node_df = pd.DataFrame(node_records)
print(f"\nUnique (lineage, pathway) nodes surviving FDR < {FDR_THRESHOLD}: {len(node_df)}")
for lin in LINEAGES:
    n = (node_df["lineage"] == lin).sum()
    print(f"  {lin}: {n}")

# %%
# ---- Cell 3: Jaccard similarity graph per lineage ----
graphs = {}

for lineage in LINEAGES:
    sub = node_df[node_df["lineage"] == lineage].reset_index(drop=True)
    G = nx.Graph()
    for _, row in sub.iterrows():
        G.add_node(row["pathway"], lead_genes=row["lead_genes"],
                   max_abs_nes=row["max_abs_nes"])

    pathways = list(sub["pathway"])
    genes = {r["pathway"]: r["lead_genes"] for _, r in sub.iterrows()}

    for i in range(len(pathways)):
        for j in range(i + 1, len(pathways)):
            a, b = genes[pathways[i]], genes[pathways[j]]
            if not a or not b:
                continue
            jacc = len(a & b) / len(a | b)
            if jacc >= JACCARD_THRESHOLD:
                G.add_edge(pathways[i], pathways[j], weight=jacc)

    graphs[lineage] = G

    # Largest connected component stats
    if len(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        pct = len(largest_cc) / len(G) * 100
    else:
        pct = 0.0
    print(f"\n{lineage}: {len(G.nodes)} nodes, {len(G.edges)} edges, "
          f"largest CC = {pct:.0f}% of nodes")

# %%
# ---- Cell 4: Node attributes (color, size, module) ----
# Tissue affinity for color
affinity = compute_tissue_affinity(gsea_long)

for lineage in LINEAGES:
    G = graphs[lineage]
    aff_lin = affinity[affinity["lineage"] == lineage].set_index("pathway")

    for node in G.nodes:
        # Tissue color
        if node in aff_lin.index:
            dom = aff_lin.loc[node, "dominant_tissue"]
        else:
            dom = "PBMC"  # fallback
        G.nodes[node]["tissue_color"] = TISSUE_COLORS.get(dom, "#999")
        G.nodes[node]["dominant_tissue"] = dom

        # Size: scale max|NES| to 100-800 range
        raw = G.nodes[node].get("max_abs_nes", 1.0)
        G.nodes[node]["raw_nes"] = raw

    # Normalise sizes across this lineage
    all_nes = [G.nodes[n]["raw_nes"] for n in G.nodes]
    if all_nes:
        mn, mx = min(all_nes), max(all_nes)
        rng = mx - mn if mx > mn else 1.0
        for n in G.nodes:
            G.nodes[n]["size"] = 100 + 700 * (G.nodes[n]["raw_nes"] - mn) / rng
    else:
        for n in G.nodes:
            G.nodes[n]["size"] = 300

    # Module detection
    if len(G.edges) > 0:
        communities = list(greedy_modularity_communities(G, weight="weight"))
    else:
        communities = [{n} for n in G.nodes]

    for mod_id, members in enumerate(communities):
        for node in members:
            G.nodes[node]["module"] = mod_id

    G.graph["communities"] = communities

    # Print top 3 modules
    sorted_mods = sorted(enumerate(communities), key=lambda x: -len(x[1]))
    print(f"\n{lineage} — {len(communities)} modules detected")
    for mod_id, members in sorted_mods[:3]:
        print(f"  Module {mod_id} ({len(members)} pathways):")
        for pw in sorted(members):
            print(f"    - {pw}")

# %%
# ---- Cell 5: Plot ----
try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for ax, lineage in zip(axes, LINEAGES):
    G = graphs[lineage]
    if len(G) == 0:
        ax.set_title(f"{lineage} (no data)")
        continue

    pos = nx.kamada_kawai_layout(G, weight="weight")

    # Draw module hulls
    communities = G.graph.get("communities", [])
    for mod_id, members in enumerate(communities):
        if len(members) < 3:
            continue
        pts = np.array([pos[n] for n in members])
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            color = MODULE_COLORS[mod_id % len(MODULE_COLORS)]
            polygon = MplPolygon(hull_pts, closed=True, alpha=0.1,
                                 facecolor=color, edgecolor=color,
                                 linewidth=1, linestyle="--", zorder=0)
            ax.add_patch(polygon)
        except Exception:
            pass  # degenerate hull (collinear points)

    # Draw edges
    edge_weights = [G[u][v]["weight"] for u, v in G.edges]
    if edge_weights:
        max_w = max(edge_weights)
        alphas = [0.15 + 0.6 * (w / max_w) for w in edge_weights]
    else:
        alphas = []
    for (u, v), alpha in zip(G.edges, alphas):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="gray", alpha=alpha, linewidth=0.8, zorder=1)

    # Draw nodes
    node_list = list(G.nodes)
    xs = [pos[n][0] for n in node_list]
    ys = [pos[n][1] for n in node_list]
    sizes = [G.nodes[n]["size"] for n in node_list]
    colors = [G.nodes[n]["tissue_color"] for n in node_list]

    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="black",
               linewidths=1, zorder=3, alpha=0.85)

    # Labels: top 40% by size
    size_cutoff = np.percentile(sizes, 60)
    texts = []
    for n, x, y, sz in zip(node_list, xs, ys, sizes):
        if sz >= size_cutoff:
            t = ax.annotate(
                n, (x, y), fontsize=6, ha="center", va="bottom",
                xytext=(0, 4), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7),
                zorder=4,
            )
            texts.append(t)

    if HAS_ADJUST and texts:
        adjust_text(texts, ax=ax)

    ax.set_title(f"{lineage}", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

# Legend for tissue colors
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=TISSUE_COLORS[t], markersize=10,
           markeredgecolor="k", markeredgewidth=0.5, label=t)
    for t in TISSUES
]
axes[1].legend(handles=legend_handles, loc="upper right", fontsize=9,
               title="Dominant tissue", title_fontsize=10, framealpha=0.9)

plt.suptitle(
    f"Pathway Co-enrichment Network — {LIBRARY_TAG}\n"
    f"(Jaccard ≥ {JACCARD_THRESHOLD} on leading-edge genes, "
    f"FDR < {FDR_THRESHOLD})",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
out_path = OUTPUT_DIR / f"pathway_coenrichment_{LIBRARY_TAG}.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_path}")

print("\nDone.")

# %%
