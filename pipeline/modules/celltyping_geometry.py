"""Simplex geometry and clone transition plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.clone_helpers import infer_lineage_from_phenotype, shorten_phenotype_label

def freq_to_size_scaling(freq, a, b):
    return a * (freq ** (1 / b))

def bary_to_cart(a, b, c):
    total = a + b + c
    a, b, c = a / total, b / total, c / total
    x = 0.5 * (2 * b + c)
    y = (np.sqrt(3) / 2) * c
    return x, y

def draw_simplex(ax, title, group_names):
    verts = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]])
    ax.plot(verts[:, 0], verts[:, 1], 'k-', linewidth=2.5)
    for f in [0.25, 0.5, 0.75]:
        for i in range(3):
            arr = np.array([0, 0, 0], dtype=float)
            arr[i] = f
            points = []
            for j in range(3):
                if j != i:
                    b_ = arr.copy()
                    b_[j] = 1 - f
                    points.append(bary_to_cart(b_[0], b_[1], b_[2]))
            ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]],
                    'k-', alpha=0.15, linewidth=1.0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.text(0, -0.06, group_names[0], ha='center', fontsize=9, fontweight='bold')
    ax.text(1, -0.06, group_names[1], ha='center', fontsize=9, fontweight='bold')
    ax.text(0.5, np.sqrt(3) / 2 + 0.04, group_names[2], ha='center', fontsize=9, fontweight='bold')

def plot_clone_simplex(
    adata,
    phenotype_groups,
    lineage="CD8",
    phenotype_key="phenotype",
    clone_key="trb",
    tissue_key="tissue",
    tissues=("PBMC", "CSF", "TP"),
    tissue_labels=("PBMC", "CSF", "Tumor"),
    tissue_colors=None,
    a=3, b=2,
    alpha=0.4,
    jitter=0.012,
    remove_switching=True,
    title=None,
    seed=42,
):
    if tissue_colors is None:
        tissue_colors = dict(zip(tissues, ["#4A90D9", "#50C878", "#D94A4A"]))
    if title is None:
        title = lineage

    tcr = adata.obs[adata.obs[clone_key].notna()].copy()
    tcr[tissue_key] = tcr[tissue_key].astype(str)
    tcr[phenotype_key] = tcr[phenotype_key].astype(str)
    tcr[clone_key] = tcr[clone_key].astype(str)

    tcr["_lineage"] = tcr[phenotype_key].apply(infer_lineage_from_phenotype)
    tcr["_pheno_short"] = tcr[phenotype_key].apply(shorten_phenotype_label)

    if remove_switching:
        cl = tcr.groupby(clone_key)["_lineage"].nunique()
        tcr = tcr[~tcr[clone_key].isin(cl[cl > 1].index)]

    tcr = tcr[tcr["_lineage"] == lineage]

    cts = (tcr.groupby([clone_key, tissue_key, "_pheno_short"])
        .size().reset_index(name="n"))
    totals = cts.groupby([clone_key, tissue_key])["n"].transform("sum")
    cts["frac"] = cts["n"] / totals

    group_names = list(phenotype_groups.keys())
    pheno_to_vertex = {}
    for i, phenos in enumerate(phenotype_groups.values()):
        for p in phenos:
            pheno_to_vertex[p] = i

    records = []
    for (clone, tissue), grp in cts.groupby([clone_key, tissue_key]):
        coords = [0.0, 0.0, 0.0]
        for _, row in grp.iterrows():
            vi = pheno_to_vertex.get(row["_pheno_short"])
            if vi is not None:
                coords[vi] += row["frac"]
        s = sum(coords)
        if s > 0:
            coords = [c / s for c in coords]
        records.append({clone_key: clone, tissue_key: tissue,
                        "v0": coords[0], "v1": coords[1], "v2": coords[2],
                        "n_cells": grp["n"].sum()})

    df = pd.DataFrame(records)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(1, len(tissues), figsize=(6 * len(tissues), 6))
    if len(tissues) == 1:
        axes = [axes]

    for ax, tissue, tlabel in zip(axes, tissues, tissue_labels):
        sub = df[df[tissue_key] == tissue]
        n_clones = len(sub)
        n_cells = int(sub["n_cells"].sum())
        draw_simplex(ax, f"{tlabel}\n({n_clones:,} clones, {n_cells:,} cells)", group_names)

        if len(sub) == 0:
            continue

        x, y = bary_to_cart(sub["v0"].values, sub["v1"].values, sub["v2"].values)
        sizes = freq_to_size_scaling(sub["n_cells"].values.astype(float), a, b)

        x_jit = x + rng.normal(0, jitter, len(x))
        y_jit = y + rng.normal(0, jitter, len(y))

        ax.scatter(x_jit, y_jit, s=sizes, c=tissue_colors[tissue], alpha=alpha, edgecolors='none')

    plt.suptitle(f"{title} Clone Distributions on Phenotype Simplex",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return df

def plot_clone_transitions(
    adata,
    phenotype_groups,
    tissue_from,
    tissue_to,
    lineage="CD8",
    phenotype_key="phenotype",
    clone_key="trb",
    tissue_key="tissue",
    tissue_labels=None,
    clone_subset=None,
    a=3, b=2,
    alpha=0.3,
    jitter=0.012,
    arrow_color="black",
    arrow_alpha=0.15,
    arrow_width=0.003,
    arrow_head_width=0.012,
    arrow_head_length=0.008,
    point_color_from="#4A90D9",
    point_color_to="#D94A4A",
    remove_switching=True,
    title=None,
    seed=42,
    figsize=(8, 7),
):
    if tissue_labels is None:
        tissue_labels = (tissue_from, tissue_to)
    if title is None:
        title = f"{lineage}: {tissue_labels[0]} → {tissue_labels[1]}"

    tcr = adata.obs[adata.obs[clone_key].notna()].copy()
    tcr[tissue_key] = tcr[tissue_key].astype(str)
    tcr[phenotype_key] = tcr[phenotype_key].astype(str)
    tcr[clone_key] = tcr[clone_key].astype(str)

    tcr["_lineage"] = tcr[phenotype_key].apply(infer_lineage_from_phenotype)
    tcr["_pheno_short"] = tcr[phenotype_key].apply(shorten_phenotype_label)

    if remove_switching:
        cl = tcr.groupby(clone_key)["_lineage"].nunique()
        tcr = tcr[~tcr[clone_key].isin(cl[cl > 1].index)]

    tcr = tcr[tcr["_lineage"] == lineage]
    tcr = tcr[tcr[tissue_key].isin([tissue_from, tissue_to])]

    clone_tissues = tcr.groupby(clone_key)[tissue_key].apply(set)
    shared = clone_tissues[clone_tissues.apply(lambda s: tissue_from in s and tissue_to in s)].index

    if clone_subset is not None:
        shared = shared.intersection(pd.Index([str(c) for c in clone_subset]))

    tcr = tcr[tcr[clone_key].isin(shared)]

    cts = (tcr.groupby([clone_key, tissue_key, "_pheno_short"])
        .size().reset_index(name="n"))
    totals = cts.groupby([clone_key, tissue_key])["n"].transform("sum")
    cts["frac"] = cts["n"] / totals

    group_names = list(phenotype_groups.keys())
    pheno_to_vertex = {}
    for i, phenos in enumerate(phenotype_groups.values()):
        for p in phenos:
            pheno_to_vertex[p] = i

    records = []
    for (clone, tissue), grp in cts.groupby([clone_key, tissue_key]):
        coords = [0.0, 0.0, 0.0]
        for _, row in grp.iterrows():
            vi = pheno_to_vertex.get(row["_pheno_short"])
            if vi is not None:
                coords[vi] += row["frac"]
        s = sum(coords)
        if s > 0:
            coords = [c / s for c in coords]
        records.append({clone_key: clone, tissue_key: tissue,
                        "v0": coords[0], "v1": coords[1], "v2": coords[2],
                        "n_cells": grp["n"].sum()})

    df = pd.DataFrame(records)
    df_from = df[df[tissue_key] == tissue_from].set_index(clone_key)
    df_to = df[df[tissue_key] == tissue_to].set_index(clone_key)
    common = df_from.index.intersection(df_to.index)

    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    draw_simplex(ax, title, group_names)

    x1, y1 = bary_to_cart(df_from.loc[common, "v0"].values,
                           df_from.loc[common, "v1"].values,
                           df_from.loc[common, "v2"].values)
    x2, y2 = bary_to_cart(df_to.loc[common, "v0"].values,
                           df_to.loc[common, "v1"].values,
                           df_to.loc[common, "v2"].values)

    x1_jit = x1 + rng.normal(0, jitter, len(x1))
    y1_jit = y1 + rng.normal(0, jitter, len(y1))
    x2_jit = x2 + rng.normal(0, jitter, len(x2))
    y2_jit = y2 + rng.normal(0, jitter, len(y2))

    sizes_from = freq_to_size_scaling(df_from.loc[common, "n_cells"].values.astype(float), a, b)
    sizes_to = freq_to_size_scaling(df_to.loc[common, "n_cells"].values.astype(float), a, b)

    for i in range(len(common)):
        dx = x2_jit[i] - x1_jit[i]
        dy = y2_jit[i] - y1_jit[i]
        ax.arrow(x1_jit[i], y1_jit[i], dx, dy,
                 width=arrow_width, head_width=arrow_head_width,
                 head_length=arrow_head_length,
                 fc=arrow_color, ec=arrow_color, alpha=arrow_alpha,
                 length_includes_head=True, zorder=1)

    ax.scatter(x1_jit, y1_jit, s=sizes_from, c=point_color_from, alpha=alpha,
               edgecolors='none', zorder=2, label=f"{tissue_labels[0]} ({len(common)} clones)")
    ax.scatter(x2_jit, y2_jit, s=sizes_to, c=point_color_to, alpha=alpha,
               edgecolors='none', zorder=2, label=f"{tissue_labels[1]} ({len(common)} clones)")

    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.show()

    return df

