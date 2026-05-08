import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .style import (COLORS, TISSUE_ORDER, tissue_color, tissue_label,
                    init_colors_from_dataset, get_phenotype_colormap, apply_style)


# ==================================================================
# Utilities
# ==================================================================

def _get_compositions(dataset, modality, tissue, timepoint):
    try:
        return dataset.get_composition(modality, tissue=tissue, timepoint=timepoint)
    except Exception:
        return pd.Series(dtype=float)


def _draw_pie(ax, x, y, fracs, colors, radius=0.12):
    if not fracs or sum(fracs) == 0:
        return
    startangle = 90
    for frac, col in zip(fracs, colors):
        if frac <= 0:
            continue
        theta = frac * 360
        ax.add_patch(plt.matplotlib.patches.Wedge(
            (x, y), radius, startangle, startangle + theta,
            facecolor=col, edgecolor="white", linewidth=0.3, zorder=3))
        startangle += theta


# ==================================================================
# Panel A: Sample Overview Grid
# ==================================================================

def panel_sample_overview(ax, dataset, max_bubble=300, x_gap=0.22):
    timepoints = dataset.timepoints
    tc = COLORS["lineage"]["T cell"]
    mc = COLORS["lineage"]["Myeloid"]

    all_counts = []
    for tis in TISSUE_ORDER:
        for tp in timepoints:
            nt = len(dataset.get_cells("tcell", tissue=tis, timepoint=tp)) if dataset.tdata is not None else 0
            nm = len(dataset.get_cells("myeloid", tissue=tis, timepoint=tp)) if dataset.mdata is not None else 0
            pats = set()
            if dataset.tdata is not None:
                sub = dataset.get_cells("tcell", tissue=tis, timepoint=tp)
                if len(sub) > 0:
                    pats.update(sub.obs["patient"].unique())
            if dataset.mdata is not None:
                sub = dataset.get_cells("myeloid", tissue=tis, timepoint=tp)
                if len(sub) > 0:
                    pats.update(sub.obs["patient"].unique())
            all_counts.append((tis, tp, nt, nm, len(pats)))

    max_cells = max(max(r[2], r[3]) for r in all_counts) or 1

    for j, tp in enumerate(timepoints):
        for i, tis in enumerate(TISSUE_ORDER):
            row = [r for r in all_counts if r[0] == tis and r[1] == tp][0]
            _, _, nt, nm, np_ = row
            y = len(TISSUE_ORDER) - 1 - i

            sz_t = max((nt / max_cells) * max_bubble, 8) if nt > 0 else 0
            ax.scatter(j - x_gap, y, s=sz_t, color=tc, alpha=0.5,
                       edgecolors=tc, linewidth=0.5, zorder=3)
            if nt > 0:
                ax.text(j - x_gap, y, f"{nt:,}", ha="center", va="center",
                        fontsize=4, color="#2C3E50", fontweight="bold", zorder=4)

            sz_m = max((nm / max_cells) * max_bubble, 8) if nm > 0 else 0
            ax.scatter(j + x_gap, y, s=sz_m, color=mc, alpha=0.5,
                       edgecolors=mc, linewidth=0.5, zorder=3)
            if nm > 0:
                ax.text(j + x_gap, y, f"{nm:,}", ha="center", va="center",
                        fontsize=4, color="#2C3E50", fontweight="bold", zorder=4)

            ax.text(j, y - 0.4, f"n={np_}", ha="center", va="top",
                    fontsize=5, color="#555")

    ax.set_xlim(-0.7, len(timepoints) - 0.3)
    ax.set_ylim(-0.8, len(TISSUE_ORDER) - 0.4)
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints, fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(TISSUE_ORDER)))
    ax.set_yticklabels([tissue_label(t) for t in reversed(TISSUE_ORDER)],
                        fontsize=8, fontweight="bold")
    for i, tis in enumerate(TISSUE_ORDER):
        ax.get_yticklabels()[i].set_color(tissue_color(tis))
    apply_style(ax, grid_alpha=0)

    for i in range(len(TISSUE_ORDER)):
        ax.axhline(i, color="#eee", lw=0.5, zorder=0)
    for j in range(len(timepoints)):
        ax.axvline(j, color="#eee", lw=0.5, zorder=0)

    handles = [plt.scatter([], [], s=50, color=tc, alpha=0.5, label="T cells"),
               plt.scatter([], [], s=50, color=mc, alpha=0.5, label="Myeloid")]
    ax.legend(handles=handles, loc="upper right", fontsize=6, frameon=False,
              handletextpad=0.3, borderpad=0.3)

    n_t = len(dataset.tdata) if dataset.tdata is not None else 0
    n_m = len(dataset.mdata) if dataset.mdata is not None else 0
    n_pat = len(dataset.patients)
    ax.set_title(f"{n_pat} patients · {len(timepoints)} timepoints · {n_t + n_m:,} cells",
                 fontsize=8, color="#555", loc="left", pad=6)
    return ax


# ==================================================================
# Panel B: Clone Tracking Network
# ==================================================================

def panel_clone_network(ax, dataset, pie_r_t=0.13, pie_r_m=0.08, x_spacing=1.8,
                        max_lw=5, min_lw=0.3, edge_alpha=0.4, n_legend_sizes=4):
    import networkx as nx

    timepoints = dataset.timepoints
    t_phenos = dataset.phenotypes("tcell")
    m_phenos = dataset.phenotypes("myeloid")

    tpc = get_phenotype_colormap("tcell")
    if not tpc:
        tpc = {p: plt.cm.tab20(i / max(len(t_phenos) - 1, 1)) for i, p in enumerate(t_phenos)}
    mpc = get_phenotype_colormap("myeloid")
    if not mpc:
        mpc = {p: plt.cm.tab20c(0.3 + 0.6 * i / max(len(m_phenos) - 1, 1)) for i, p in enumerate(m_phenos)}

    tissue_y = {"TP": 2.0, "Plasma": 1.0, "CSF": 0.0}
    pie_offset = 0.18

    G = nx.Graph()
    pos = {}
    clone_sets = {}
    for tis in TISSUE_ORDER:
        for i, tp in enumerate(timepoints):
            node = (tis, tp)
            G.add_node(node)
            pos[node] = (i * x_spacing, tissue_y[tis])
            clone_sets[node] = dataset.get_clone_set(tissue=tis, timepoint=tp)

    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            n1, n2 = nodes[i], nodes[j]
            shared = len(clone_sets[n1] & clone_sets[n2])
            if shared > 0:
                G.add_edge(n1, n2, shared=shared)

    max_shared = 1
    if G.number_of_edges() > 0:
        max_shared = max(G[u][v]["shared"] for u, v in G.edges())
        for u, v, d in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            lw = min_lw + (d["shared"] / max_shared) * (max_lw - min_lw)
            ax.plot([x1, x2], [y1, y2], color="black", lw=lw,
                    alpha=edge_alpha, zorder=1, solid_capstyle="round")

    for tis in TISSUE_ORDER:
        y = tissue_y[tis]
        for i, tp in enumerate(timepoints):
            x = i * x_spacing
            t_comp = _get_compositions(dataset, "tcell", tis, tp)
            if len(t_comp) > 0:
                _draw_pie(ax, x - pie_offset / 2, y,
                          [t_comp.get(p, 0) for p in t_phenos],
                          [tpc.get(p, "#999") for p in t_phenos], pie_r_t)
            m_comp = _get_compositions(dataset, "myeloid", tis, tp)
            if len(m_comp) > 0:
                _draw_pie(ax, x + pie_offset, y,
                          [m_comp.get(p, 0) for p in m_phenos],
                          [mpc.get(p, "#999") for p in m_phenos], pie_r_m)

    for i, tp in enumerate(timepoints):
        ax.text(i * x_spacing, 2.50, tp, ha="center", fontsize=8,
                fontweight="bold", color="#333")
    for tis in TISSUE_ORDER:
        ax.text(-0.45, tissue_y[tis], tissue_label(tis), ha="right", va="center",
                fontsize=9, fontweight="bold", color=tissue_color(tis))

    legend_vals = np.linspace(1, max_shared, n_legend_sizes).astype(int)
    legend_vals = sorted(set(legend_vals))
    edge_handles = [plt.Line2D([0], [0], color="black",
                    lw=min_lw + (v / max_shared) * (max_lw - min_lw),
                    alpha=edge_alpha, label=f"{v:,}") for v in legend_vals]
    leg_edge = ax.legend(handles=edge_handles, loc="upper right", fontsize=5,
                          title="Shared clones", title_fontsize=6,
                          frameon=True, framealpha=0.9, bbox_to_anchor=(1.02, 1.0),
                          handlelength=2)
    ax.add_artist(leg_edge)

    t_handles = [plt.Line2D([0], [0], marker="o", color="w",
                 markerfacecolor=tpc.get(p, "#999"), markersize=4.5, label=p)
                 for p in t_phenos]
    leg1 = ax.legend(handles=t_handles, loc="lower left", fontsize=4.5,
                     title="T cell", title_fontsize=5.5,
                     frameon=True, framealpha=0.9, ncol=2, bbox_to_anchor=(0, -0.12))
    ax.add_artist(leg1)
    m_handles = [plt.Line2D([0], [0], marker="o", color="w",
                 markerfacecolor=mpc.get(p, "#999"), markersize=4.5, label=p)
                 for p in m_phenos]
    ax.legend(handles=m_handles, loc="lower right", fontsize=4.5,
              title="Myeloid", title_fontsize=5.5,
              frameon=True, framealpha=0.9, ncol=2, bbox_to_anchor=(1, -0.12))

    ax.set_xlim(-0.6, (len(timepoints) - 1) * x_spacing + 0.8)
    ax.set_ylim(-0.5, 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ==================================================================
# Panel C: T Cell UMAP
# ==================================================================

def panel_tcell_umap(ax, dataset, **umap_kwargs):
    import scanpy as sc
    tdata = dataset.tdata
    defaults = dict(color="phenotype", show=False, legend_loc="right margin",
                    frameon=False, title="", size=1.5, legend_fontsize=6,
                    legend_fontoutline=1)
    defaults.update(umap_kwargs)
    sc.pl.umap(tdata, ax=ax, **defaults)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_title(f'T cells ({len(tdata):,} cells · {tdata.obs["phenotype"].nunique()} states)',
                 fontsize=9, loc="left", pad=4)
    ax.set_aspect("equal")
    return ax


# ==================================================================
# Panel D: Myeloid UMAP
# ==================================================================

def panel_myeloid_umap(ax, dataset, **umap_kwargs):
    import scanpy as sc
    mdata = dataset.mdata
    defaults = dict(color="phenotype", show=False, legend_loc="right margin",
                    frameon=False, title="", size=1.5, legend_fontsize=6,
                    legend_fontoutline=1)
    defaults.update(umap_kwargs)
    sc.pl.umap(mdata, ax=ax, **defaults)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_title(f'Myeloid ({len(mdata):,} cells · {mdata.obs["phenotype"].nunique()} states)',
                 fontsize=9, loc="left", pad=4)
    ax.set_aspect("equal")
    return ax


# ==================================================================
# Panel E: TCR Clone Sharing Triangle
# ==================================================================

def panel_tcr_sharing(ax, dataset):
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    positions = {"Plasma": (-0.7, 0.9), "TP": (0.7, 0.9), "CSF": (0, -0.4)}
    for tis, (x, y) in positions.items():
        ax.add_patch(plt.Circle((x, y), 0.45, color=tissue_color(tis), alpha=1.0,
                                ec=tissue_color(tis), linewidth=1.2))
        clones = dataset.get_clone_set(tissue=tis)
        ax.text(x, y + 0.08, tissue_label(tis), ha="center", va="center",
                fontsize=8, fontweight="bold", color="black")
        ax.text(x, y - 0.12, f"{len(clones):,} clones", ha="center", va="center",
                fontsize=6, color="black")

    for t1, t2 in [("Plasma", "TP"), ("Plasma", "CSF"), ("CSF", "TP")]:
        shared = dataset.get_shared_clones(t1, t2)
        x1, y1 = positions[t1]
        x2, y2 = positions[t2]
        dx, dy = x2 - x1, y2 - y1
        norm = np.sqrt(dx**2 + dy**2)
        sx, sy = x1 + dx / norm * 0.5, y1 + dy / norm * 0.5
        ex, ey = x2 - dx / norm * 0.5, y2 - dy / norm * 0.5
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-", color="#333", lw=1.2))
        ax.text(mx + dy / norm * 0.18, my - dx / norm * 0.18,
                f"{len(shared):,}", ha="center", fontsize=7, fontweight="bold",
                color="black")
    return ax


# ==================================================================
# Composite Figures
# ==================================================================

def figure1(dataset, figsize=(18, 16)):
    """Full Figure 1: overview, clone network, UMAPs, TCR sharing."""
    init_colors_from_dataset(dataset)

    fig = plt.figure(figsize=figsize, facecolor="white", dpi=150)
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[0.7, 1.3, 1.4],
                           hspace=0.4, wspace=0.5)

    ax_a = fig.add_subplot(gs[0, :2])
    panel_sample_overview(ax_a, dataset)
    ax_a.text(-0.08, 1.12, "A", transform=ax_a.transAxes, fontsize=16,
              fontweight="bold", va="top")

    ax_b = fig.add_subplot(gs[0:2, 2:])
    panel_clone_network(ax_b, dataset)
    ax_b.text(-0.02, 1.05, "B", transform=ax_b.transAxes, fontsize=16,
              fontweight="bold", va="top")

    if dataset.tdata is not None and "X_umap" in dataset.tdata.obsm:
        ax_c = fig.add_subplot(gs[2, :3])
        panel_tcell_umap(ax_c, dataset)
        ax_c.text(-0.05, 1.08, "C", transform=ax_c.transAxes, fontsize=16,
                  fontweight="bold", va="top")

    if dataset.mdata is not None and "X_umap" in dataset.mdata.obsm:
        ax_d = fig.add_subplot(gs[2, 3:5])
        panel_myeloid_umap(ax_d, dataset)
        ax_d.text(-0.05, 1.08, "D", transform=ax_d.transAxes, fontsize=16,
                  fontweight="bold", va="top")

    ax_e = fig.add_subplot(gs[2, 5])
    panel_tcr_sharing(ax_e, dataset)
    ax_e.text(-0.05, 1.08, "E", transform=ax_e.transAxes, fontsize=16,
              fontweight="bold", va="top")

    return fig