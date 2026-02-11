# %%

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.clone_helpers import (
    categorize_edge,
    infer_lineage_from_phenotype,
    parse_node_id,
    shorten_phenotype_label,
)

adata = sc.read("GBM_TCR_POS_TCELLS.h5ad")
tcr = adata.obs[adata.obs["trb"].notna()].copy()
tcr["tissue"] = tcr["tissue"].astype(str)
tcr["phenotype"] = tcr["phenotype"].astype(str)
tcr["level1"] = tcr["phenotype"].apply(infer_lineage_from_phenotype)

# Clean phenotype names for readability
tcr["pheno_short"] = tcr["phenotype"].apply(shorten_phenotype_label)


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

tcr = adata.obs[adata.obs["trb"].notna()].copy()
tcr["level1"] = tcr["phenotype"].apply(infer_lineage_from_phenotype)
tcr["pheno_short"] = tcr["phenotype"].apply(shorten_phenotype_label)

# Remove remaining switching clones
clone_lin = tcr.groupby("trb")["level1"].nunique()
switching = clone_lin[clone_lin > 1].index
tcr = tcr[~tcr["trb"].isin(switching)]

# For each clone × tissue: fractional phenotype distribution
clone_tissue_dist = (tcr.groupby(["trb", "tissue", "pheno_short"]).size()
    .groupby(level=[0,1]).transform(lambda x: x / x.sum())
    .reset_index(name="frac"))

# Clone sizes per tissue (for weighting)
clone_tissue_size = tcr.groupby(["trb", "tissue"]).size().reset_index(name="n_cells")

# Merge
clone_tissue_dist = clone_tissue_dist.merge(clone_tissue_size, on=["trb", "tissue"])

# Add lineage
clone_lineage = tcr.groupby("trb")["level1"].first()
clone_tissue_dist["lineage"] = clone_tissue_dist["trb"].map(clone_lineage)

# Find clones shared across tissue pairs
clone_tissues = tcr.groupby("trb")["tissue"].apply(set)

compartment_pairs = [("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP")]
pair_labels = ["PBMC → Tumor", "PBMC → CSF", "CSF → Tumor"]

for lineage in ["CD8", "CD4"]:
    # Get phenotype order
    lin_phenos = sorted(tcr[tcr["level1"] == lineage]["pheno_short"].unique())
    
    fig, axes = plt.subplots(len(compartment_pairs), 1, figsize=(14, 5 * len(compartment_pairs)))
    if len(compartment_pairs) == 1:
        axes = [axes]
    
    for ax_idx, ((t1, t2), pair_label) in enumerate(zip(compartment_pairs, pair_labels)):
        # Clones present in both tissues
        shared = clone_tissues[clone_tissues.apply(lambda s: t1 in s and t2 in s)].index
        shared_lin = shared[shared.isin(clone_lineage[clone_lineage == lineage].index)]
        
        if len(shared_lin) == 0:
            axes[ax_idx].text(0.5, 0.5, f"No shared {lineage} clones", 
                             ha="center", va="center", transform=axes[ax_idx].transAxes)
            axes[ax_idx].set_title(f"{pair_label}")
            continue
        
        # For each clone: get distribution in t1 and t2
        sub = clone_tissue_dist[
            (clone_tissue_dist["trb"].isin(shared_lin)) & 
            (clone_tissue_dist["lineage"] == lineage) &
            (clone_tissue_dist["tissue"].isin([t1, t2]))
        ]
        
        # Clone total size (sum across both tissues) for weighting
        clone_total = sub.groupby("trb")["n_cells"].sum()
        sub["clone_weight"] = sub["trb"].map(clone_total)
        
        # Weighted average distribution per tissue
        results = {}
        for tissue in [t1, t2]:
            ts = sub[sub["tissue"] == tissue]
            # Weight each clone's distribution by its total size
            weighted = ts.copy()
            weighted["weighted_frac"] = weighted["frac"] * weighted["clone_weight"]
            agg = weighted.groupby("pheno_short")["weighted_frac"].sum()
            agg = agg / agg.sum()  # renormalize
            results[tissue] = agg.reindex(lin_phenos, fill_value=0)
        
        # Also compute unweighted (equal per clone)
        results_uw = {}
        for tissue in [t1, t2]:
            ts = sub[sub["tissue"] == tissue]
            agg = ts.groupby("pheno_short")["frac"].mean()
            agg = agg / agg.sum()
            results_uw[tissue] = agg.reindex(lin_phenos, fill_value=0)
        
        # Plot: paired bars (weighted)
        x = np.arange(len(lin_phenos))
        w = 0.35
        bars1 = axes[ax_idx].bar(x - w/2, results[t1], w, label=t1, color="#4A90D9", alpha=0.85)
        bars2 = axes[ax_idx].bar(x + w/2, results[t2], w, label=t2, color="#D94A4A", alpha=0.85)
        
        # Add shift arrows
        for i, ph in enumerate(lin_phenos):
            v1, v2 = results[t1].iloc[i], results[t2].iloc[i]
            if abs(v2 - v1) > 0.02:
                color = "#2E7D32" if v2 > v1 else "#C62828"
                axes[ax_idx].annotate(f"{v2-v1:+.0%}", xy=(i, max(v1, v2) + 0.01),
                                     fontsize=8, ha="center", color=color, fontweight="bold")
        
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(lin_phenos, rotation=45, ha="right")
        axes[ax_idx].set_ylabel("Fraction of cells")
        axes[ax_idx].set_title(f"{pair_label} (n={len(shared_lin)} shared clones, weighted by clone size)")
        axes[ax_idx].legend()
        axes[ax_idx].set_ylim(0, axes[ax_idx].get_ylim()[1] * 1.15)
        
        # Print summary
        print(f"\n{lineage} {pair_label}: {len(shared_lin)} shared clones, "
              f"{sub['n_cells'].sum():.0f} total cells")
        summary = pd.DataFrame({"from": results[t1], "to": results[t2], 
                                "shift": results[t2] - results[t1]})
        print(summary.round(3))
    
    plt.suptitle(f"{lineage} Phenotype Redistribution Across Compartments", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
# %%
tcr["level1"] = tcr["phenotype"].apply(infer_lineage_from_phenotype)
tcr["pheno_short"] = tcr["phenotype"].apply(shorten_phenotype_label)

# Verify: switching clones should be resolved
clone_lineage = tcr.groupby("trb")["level1"].nunique()
remaining_switching = (clone_lineage > 1).sum()
print(f"Remaining switching clones: {remaining_switching}")

# Shared clones
clone_tissues = tcr.groupby("trb")["tissue"].nunique()
shared_clones = clone_tissues[clone_tissues >= 2].index
shared = tcr[tcr["trb"].isin(shared_clones)].copy()

# Remove any remaining switching clones
if remaining_switching > 0:
    still_switching = clone_lineage[clone_lineage > 1].index
    shared = shared[~shared["trb"].isin(still_switching)]

print(f"Shared clones: {shared['trb'].nunique()}")
print(f"Cells from shared clones: {len(shared)}")

# Build transition matrices
import matplotlib.pyplot as plt
import seaborn as sns

for lineage in ["CD8", "CD4"]:
    lin = shared[shared["level1"] == lineage].copy()
    lin_tc = lin.groupby("trb")["tissue"].nunique()
    lin_shared = lin_tc[lin_tc >= 2].index
    lin = lin[lin["trb"].isin(lin_shared)]
    
    cs = (lin.groupby(["trb", "tissue"])["pheno_short"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
        .unstack(fill_value=None))
    
    if "PBMC" not in cs.columns or "TP" not in cs.columns:
        continue
    pt = cs[cs["PBMC"].notna() & cs["TP"].notna()].copy()
    if len(pt) == 0:
        continue
    
    trans_norm = pd.crosstab(pt["PBMC"], pt["TP"], normalize="index")
    trans_counts = pd.crosstab(pt["PBMC"], pt["TP"])
    
    patterns = pt.apply(lambda r: f"{r['PBMC']} → {r['TP']}", axis=1)
    top = patterns.value_counts().head(15)
    same = (pt["PBMC"] == pt["TP"]).sum()
    diff = (pt["PBMC"] != pt["TP"]).sum()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.heatmap(trans_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[0], linewidths=0.5)
    axes[0].set_title(f"{lineage}: PBMC → Tumor (n={len(pt)} shared clones)")
    axes[0].set_ylabel("State in PBMC"); axes[0].set_xlabel("State in Tumor")
    
    top.plot.barh(ax=axes[1], color="#E8A435")
    axes[1].set_title(f"{lineage}: Top transitions")
    axes[1].set_xlabel("Number of clones"); axes[1].invert_yaxis()
    
    axes[2].pie([same, diff],
                labels=[f"Same state\n(n={same})", f"Different state\n(n={diff})"],
                colors=["#4CAF50", "#FF5722"], autopct="%1.0f%%", startangle=90)
    axes[2].set_title(f"{lineage}: State preservation")
    
    plt.suptitle(f"{lineage} Clone Trafficking (lineage-corrected)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    
    print(f"\n{lineage}: {len(pt)} shared clones")
    print("Counts:")
    print(trans_counts)
    print("\nRow-normalized:")
    print(trans_norm.round(2))
    
    # Three compartment
    if "CSF" in cs.columns:
        all3 = cs[cs.notna().all(axis=1)]
        if len(all3) > 0:
            all3["trajectory"] = all3["PBMC"] + " → " + all3["CSF"] + " → " + all3["TP"]
            print(f"\n3-compartment trajectories ({len(all3)} clones):")
            print(all3["trajectory"].value_counts().head(15))

# Directionality
print("\n=== Directionality ===")
for lineage in ["CD8", "CD4"]:
    lin = shared[shared["level1"] == lineage]
    cc = lin.groupby(["trb", "tissue"]).size().unstack(fill_value=0)
    if "PBMC" in cc.columns and "TP" in cc.columns:
        both = cc[(cc["PBMC"] > 0) & (cc["TP"] > 0)]
        pbmc_dom = (both["PBMC"] > both["TP"]).sum()
        tp_dom = (both["TP"] > both["PBMC"]).sum()
        eq = (both["PBMC"] == both["TP"]).sum()
        print(f"\n{lineage} ({len(both)} clones):")
        print(f"  PBMC-dominant: {pbmc_dom} ({pbmc_dom/len(both):.0%})")
        print(f"  Tumor-dominant: {tp_dom} ({tp_dom/len(both):.0%})")
        print(f"  Equal: {eq} ({eq/len(both):.0%})")
# %%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from itertools import combinations


def compute_clone_network(
    adata,
    phenotype_key="phenotype",
    clone_key="trb",
    tissue_key="tissue",
    time_key="timepoint",
    lineage=None,
    remove_switching=True,
    min_clone_overlap=1,
):
    tcr = adata.obs[adata.obs[clone_key].notna()].copy()
    for col in [tissue_key, phenotype_key, clone_key, time_key]:
        tcr[col] = tcr[col].astype(str)

    if lineage is not None:
        tcr["_lineage"] = tcr[phenotype_key].apply(infer_lineage_from_phenotype)
        if remove_switching:
            cl = tcr.groupby(clone_key)["_lineage"].nunique()
            tcr = tcr[~tcr[clone_key].isin(cl[cl > 1].index)]
        tcr = tcr[tcr["_lineage"] == lineage]

    nodes = (tcr.groupby([tissue_key, time_key])
        .agg(n_cells=(clone_key, "size"), n_clones=(clone_key, "nunique"))
        .reset_index())
    nodes["node_id"] = nodes[tissue_key] + "_" + nodes[time_key]

    clone_sets = (tcr.groupby([tissue_key, time_key])[clone_key]
        .apply(set).reset_index())
    clone_sets["node_id"] = clone_sets[tissue_key] + "_" + clone_sets[time_key]
    clone_dict = dict(zip(clone_sets["node_id"], clone_sets[clone_key]))

    edges = []
    node_ids = list(clone_dict.keys())
    for n1, n2 in combinations(node_ids, 2):
        overlap = clone_dict[n1] & clone_dict[n2]
        if len(overlap) >= min_clone_overlap:
            edges.append({
                "source": n1, "target": n2,
                "n_shared_clones": len(overlap),
                "shared_clones": overlap,
            })

    edges = pd.DataFrame(edges) if edges else pd.DataFrame(
        columns=["source", "target", "n_shared_clones", "shared_clones"])

    return nodes, edges, clone_dict


def _parse_node(node_id, tissues, timepoints):
    return parse_node_id(node_id, tissues, timepoints)


def _edge_category(source, target, tissues, timepoints):
    return categorize_edge(source, target, tissues, timepoints)


def plot_clone_network(
    nodes,
    edges,
    tissue_key="tissue",
    time_key="timepoint",
    tissues=("PBMC", "CSF", "TP"),
    tissue_labels=("PBMC", "CSF", "Tumor"),
    timepoints=("1", "2", "3", "4", "5", "6"),
    tissue_colors=None,
    node_size_key="n_cells",
    node_scale=0.3,
    edge_scale=0.5,
    edge_alpha=0.3,
    edge_color="#555555",
    edge_colors_by_type=None,
    min_edge_width=0.5,
    max_edge_width=12,
    figsize=(12, 14),
    title=None,
    node_label_key="n_clones",
    show_empty=True,
    edge_filter=None,
    min_edge_clones=1,
    curved_edges=True,
    ax=None,
    savepath=None,
):
    if tissue_colors is None:
        tissue_colors = dict(zip(tissues, ["#4A90D9", "#50C878", "#D94A4A"]))
    if title is None:
        title = "Clone Sharing Network"

    col_positions = {t: i for i, t in enumerate(tissues)}
    row_positions = {t: i for i, t in enumerate(timepoints)}
    col_spacing = 2.5
    row_spacing = 1.8

    node_pos = {}
    for _, row in nodes.iterrows():
        t, tp = str(row[tissue_key]), str(row[time_key])
        if t in col_positions and tp in row_positions:
            x = col_positions[t] * col_spacing
            y = (len(timepoints) - 1 - row_positions[tp]) * row_spacing
            node_pos[row["node_id"]] = (x, y)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    # Filter edges
    plot_edges = edges[edges["n_shared_clones"] >= min_edge_clones].copy()
    if edge_filter is not None:
        plot_edges["_cat"] = plot_edges.apply(
            lambda r: _edge_category(r["source"], r["target"], tissues, timepoints), axis=1)
        if isinstance(edge_filter, str):
            edge_filter = [edge_filter]
        plot_edges = plot_edges[plot_edges["_cat"].isin(edge_filter)]

    # Edge colors by type
    if edge_colors_by_type is None:
        edge_colors_by_type = {
            "within_tissue": "#888888",
            "cross_tissue_same_time": "#E67E22",
            "cross_tissue_diff_time": "#8E44AD",
        }

    # Draw edges
    if len(plot_edges) > 0:
        max_shared = plot_edges["n_shared_clones"].max()
        for _, row in plot_edges.iterrows():
            if row["source"] in node_pos and row["target"] in node_pos:
                x1, y1 = node_pos[row["source"]]
                x2, y2 = node_pos[row["target"]]
                w = min_edge_width + (row["n_shared_clones"] / max_shared) * (max_edge_width - min_edge_width)
                w *= edge_scale

                cat = _edge_category(row["source"], row["target"], tissues, timepoints)
                ec = edge_colors_by_type.get(cat, edge_color)

                if curved_edges and cat != "within_tissue":
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    dx = x2 - x1
                    dy = y2 - y1
                    perp_x, perp_y = -dy * 0.15, dx * 0.15
                    from matplotlib.patches import FancyArrowPatch
                    from matplotlib.path import Path
                    verts = [(x1, y1), (mid_x + perp_x, mid_y + perp_y), (x2, y2)]
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                    path = Path(verts, codes)
                    patch = mpatches.FancyArrowPatch(
                        path=path, arrowstyle="-",
                        linewidth=w, color=ec, alpha=edge_alpha, zorder=1)
                    ax.add_patch(patch)
                else:
                    ax.plot([x1, x2], [y1, y2], '-', color=ec,
                            linewidth=w, alpha=edge_alpha, zorder=1, solid_capstyle='round')

    # Draw nodes
    max_node_val = nodes[node_size_key].max()
    for _, row in nodes.iterrows():
        nid = row["node_id"]
        if nid not in node_pos:
            continue
        x, y = node_pos[nid]
        radius = 0.08 + np.sqrt(row[node_size_key] / max_node_val) * node_scale
        color = tissue_colors.get(str(row[tissue_key]), "#999999")
        circle = plt.Circle((x, y), radius, color=color, ec='black',
                             linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        label = f"{int(row[node_label_key])}"
        ax.text(x, y, label, ha='center', va='center', fontsize=7,
                fontweight='bold', zorder=4)
        ax.text(x, y - radius - 0.08, f"{int(row['n_cells'])}c",
                ha='center', va='top', fontsize=5.5, color='#666666', zorder=4)

    # Empty nodes
    if show_empty:
        for t in tissues:
            for tp in timepoints:
                nid = f"{t}_{tp}"
                if nid not in node_pos:
                    x = col_positions[t] * col_spacing
                    y = (len(timepoints) - 1 - row_positions[tp]) * row_spacing
                    circle = plt.Circle((x, y), 0.08, color='#F5F5F5', ec='#CCCCCC',
                                        linewidth=1, zorder=2, linestyle='--')
                    ax.add_patch(circle)

    # Column headers
    for t, tl in zip(tissues, tissue_labels):
        x = col_positions[t] * col_spacing
        y = (len(timepoints)) * row_spacing - row_spacing * 0.3
        ax.text(x, y, tl, ha='center', va='center', fontsize=13,
                fontweight='bold', color=tissue_colors.get(t, "black"))

    # Row labels
    for tp in timepoints:
        y = (len(timepoints) - 1 - row_positions[tp]) * row_spacing
        ax.text(-0.8, y, f"T{tp}", ha='center', va='center', fontsize=11, fontweight='bold')

    # Edge legend
    legend_elements = [
        mpatches.Patch(color=edge_colors_by_type["within_tissue"], alpha=0.6, label="Within tissue"),
        mpatches.Patch(color=edge_colors_by_type["cross_tissue_same_time"], alpha=0.6, label="Cross-tissue (same time)"),
        mpatches.Patch(color=edge_colors_by_type["cross_tissue_diff_time"], alpha=0.6, label="Cross-tissue (diff time)"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    margin = 1.2
    ax.set_xlim(-margin, (len(tissues) - 1) * col_spacing + margin)
    ax.set_ylim(-margin, (len(timepoints) - 1) * row_spacing + margin + row_spacing)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    if standalone:
        plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {savepath}")

    if standalone:
        plt.show()

    return node_pos


# =====================================================================
# RUN
# =====================================================================

# --- CD8 ---
nodes_cd8, edges_cd8, clones_cd8 = compute_clone_network(
    adata, lineage="CD8", min_clone_overlap=1)

print(f"CD8 — Nodes: {len(nodes_cd8)}, Edges: {len(edges_cd8)}")
print(nodes_cd8.sort_values(["tissue", "timepoint"]).to_string(index=False))
print(f"\nTop 15 edges:")
print(edges_cd8.nlargest(15, "n_shared_clones")[["source", "target", "n_shared_clones"]].to_string(index=False))

plot_clone_network(nodes_cd8, edges_cd8, title="CD8 Clone Sharing Network",
                   min_edge_clones=2, savepath="cd8_clone_network.png")

# --- CD4 ---
nodes_cd4, edges_cd4, clones_cd4 = compute_clone_network(
    adata, lineage="CD4", min_clone_overlap=1)

print(f"\nCD4 — Nodes: {len(nodes_cd4)}, Edges: {len(edges_cd4)}")
print(nodes_cd4.sort_values(["tissue", "timepoint"]).to_string(index=False))
print(f"\nTop 15 edges:")
print(edges_cd4.nlargest(15, "n_shared_clones")[["source", "target", "n_shared_clones"]].to_string(index=False))

plot_clone_network(nodes_cd4, edges_cd4, title="CD4 Clone Sharing Network",
                   min_edge_clones=2, savepath="cd4_clone_network.png")

# --- Edge breakdown ---
for label, edges_df in [("CD8", edges_cd8), ("CD4", edges_cd4)]:
    print(f"\n{label} edge breakdown:")
    tissues = ("PBMC", "CSF", "TP")
    timepoints = ("1", "2", "3", "4", "5", "6")
    edges_df["_cat"] = edges_df.apply(
        lambda r: _edge_category(r["source"], r["target"], tissues, timepoints), axis=1)
    summary = edges_df.groupby("_cat")["n_shared_clones"].agg(["count", "sum", "mean", "max"])
    print(summary.to_string())

# --- Variations ---
# Cross-tissue only
plot_clone_network(nodes_cd8, edges_cd8, title="CD8 Cross-Tissue Clone Sharing",
                   edge_filter=["cross_tissue_same_time", "cross_tissue_diff_time"],
                   min_edge_clones=1, savepath="cd8_cross_tissue_network.png")

# Within-tissue temporal only
plot_clone_network(nodes_cd8, edges_cd8, title="CD8 Temporal Clone Persistence",
                   edge_filter="within_tissue",
                   min_edge_clones=5, savepath="cd8_temporal_network.png")
# %%
