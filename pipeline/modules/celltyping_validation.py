"""Validation and expression plotting utilities for celltyping."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import sparse

from modules.clone_helpers import abbreviate_phenotype_label

def extract_expression(adata, genes, key="phenotype"):
    genes = [g for g in genes if g in adata.var_names]
    X = adata[:, genes].X
    if sparse.issparse(X):
        X = X.toarray()
    expr = pd.DataFrame(X, columns=genes, index=adata.obs.index)
    expr["phenotype"] = adata.obs[key].astype(str).values
    expr["tissue"] = adata.obs["tissue"].astype(str).values
    expr["timepoint"] = adata.obs["timepoint"].astype(str).values
    if "patient" in adata.obs.columns:
        expr["patient"] = adata.obs["patient"].astype(str).values
    return expr, genes

def plot_phenotype_heatmap(adata, genes=None, key="phenotype", figsize=(18, 6)):
    if genes is None:
        genes = [
            "CD8A","CD8B","CD4",
            "NKG7","GZMB","PRF1","GNLY","GZMA","CST7",
            "CX3CR1","S1PR5","KLF2","FGFBP2","FCGR3A",
            "GZMK","EOMES",
            "CD69","ZNF683","ITGAE",
            "TOX","LAG3","PDCD1","HAVCR2","TIGIT","ENTPD1","CXCL13",
            "TCF7","LEF1","CCR7","SELL","MAL","BACH2",
            "IL7R",
            "FOXP3","IL2RA","IKZF2","ICOS","CTLA4",
            "TBX21","STAT1","CXCR3","IFNG",
            "GATA3","STAT6","IL4R",
        ]
    expr, genes = extract_expression(adata, genes, key)

    pheno_order = [
        "CD8_Activated_TEMRA","CD8_Activated_TEXeff","CD8_Activated_TEXterm",
        "CD8_Activated_TRM","CD8_Quiescent_Memory","CD8_Quiescent_Naive",
        "CD8_Quiescent_TEXprog",
        "CD4_Naive_Memory","CD4_Exhausted","CD4_Treg",
        "CD4_Th1_polarized","CD4_Th2_polarized",
    ]
    pheno_order = [p for p in pheno_order if p in expr["phenotype"].unique()]

    means = expr.groupby("phenotype")[genes].mean().loc[pheno_order]
    zscored = (means - means.mean()) / means.std()

    fig, ax = plt.subplots(figsize=figsize)
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=3)
    im = ax.imshow(zscored.values, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(pheno_order)))
    short = [abbreviate_phenotype_label(p) for p in pheno_order]
    ax.set_yticklabels(short, fontsize=9)

    # Add cell counts
    counts = expr["phenotype"].value_counts()
    for i, p in enumerate(pheno_order):
        ax.text(len(genes) + 0.3, i, f"n={counts[p]:,}", va="center", fontsize=7, color="gray")

    plt.colorbar(im, ax=ax, label="z-score", shrink=0.8, pad=0.12)
    ax.set_title("Phenotype marker expression (z-scored across phenotypes)", fontsize=11, pad=10)
    plt.tight_layout()
    return fig

def plot_tissue_stability(adata, genes=None, key="phenotype", figsize=(20, 14)):
    if genes is None:
        genes = [
            "CD8A","CX3CR1","S1PR5","KLF2","GZMB","NKG7","GZMK",
            "CD69","ZNF683","TOX","LAG3","PDCD1","CXCL13",
            "TCF7","LEF1","SELL","IL7R",
            "FOXP3","IKZF2","IL2RA","ICOS","CTLA4",
            "TBX21","IFNG","GATA3",
        ]
    expr, genes = extract_expression(adata, genes, key)

    pheno_order = [
        "CD8_Activated_TEMRA","CD8_Activated_TEXeff","CD8_Activated_TEXterm",
        "CD8_Activated_TRM","CD8_Quiescent_Naive","CD8_Quiescent_TEXprog",
        "CD4_Naive_Memory","CD4_Exhausted","CD4_Treg",
        "CD4_Th1_polarized",
    ]
    pheno_order = [p for p in pheno_order if p in expr["phenotype"].unique()]
    tissues = ["PBMC", "CSF", "TP"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=3)

    # Compute global z-score params from overall means
    overall = expr.groupby("phenotype")[genes].mean().loc[pheno_order]
    mu, sd = overall.mean(), overall.std()

    for j, tissue in enumerate(tissues):
        sub = expr[expr["tissue"] == tissue]
        tissue_means = sub.groupby("phenotype")[genes].mean()

        # Fill missing phenotypes with NaN
        mat = pd.DataFrame(index=pheno_order, columns=genes, dtype=float)
        for p in pheno_order:
            if p in tissue_means.index:
                mat.loc[p] = tissue_means.loc[p]

        zscored = (mat - mu) / sd

        ax = axes[j]
        data = zscored.values.astype(float)
        masked = np.ma.masked_invalid(data)
        ax.imshow(masked, aspect="auto", cmap="RdBu_r", norm=norm)

        # Gray out cells with <10 observations
        tissue_counts = sub["phenotype"].value_counts()
        for i, p in enumerate(pheno_order):
            n = tissue_counts.get(p, 0)
            if n < 10:
                for k in range(len(genes)):
                    ax.add_patch(plt.Rectangle((k-0.5, i-0.5), 1, 1,
                                 fill=True, facecolor="lightgray", edgecolor="none"))
            ax.text(len(genes) + 0.2, i, f"{n:,}", va="center", fontsize=6, color="gray")

        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{tissue}", fontsize=12, fontweight="bold")

        if j == 0:
            ax.set_yticks(range(len(pheno_order)))
            short = [p.split("_",1)[1] for p in pheno_order]
            ax.set_yticklabels(short, fontsize=8)

    fig.suptitle("Phenotype stability across tissues (z-scored, global reference)", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig

def plot_timepoint_stability(adata, key="phenotype", figsize=(16, 12)):
    marker_sets = {
        "TEMRA identity": ("CD8_Activated_TEMRA", ["CX3CR1", "S1PR5", "KLF2", "FGFBP2"]),
        "Exhaustion axis": ("CD8_Activated_TEXterm", ["TOX", "LAG3", "PDCD1", "HAVCR2"]),
        "TRM residency": ("CD8_Activated_TRM", ["CD69", "ZNF683", "ITGAE", "IFNG"]),
        "Naive identity": ("CD8_Quiescent_Naive", ["TCF7", "LEF1", "CCR7", "SELL"]),
        "Treg identity": ("CD4_Treg", ["FOXP3", "IKZF2", "IL2RA", "ICOS"]),
        "CD4 exhaustion": ("CD4_Exhausted", ["CXCL13", "PDCD1", "TOX", "CTLA4"]),
    }

    all_genes = list(set(g for _, (_, gs) in marker_sets.items() for g in gs))
    expr, valid_genes = extract_expression(adata, all_genes, key)
    timepoints = sorted(expr["timepoint"].unique())

    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for idx, (title, (pheno, markers)) in enumerate(marker_sets.items()):
        ax = axes.flat[idx]
        sub = expr[expr["phenotype"] == pheno]

        for i, g in enumerate(markers):
            if g not in valid_genes:
                continue
            means = sub.groupby("timepoint")[g].mean()
            sems = sub.groupby("timepoint")[g].sem()
            x = [timepoints.index(t) for t in means.index]
            ax.plot(x, means.values, "o-", color=colors[i], label=g, linewidth=2, markersize=5)
            ax.fill_between(x, means.values - sems.values, means.values + sems.values,
                          alpha=0.15, color=colors[i])

        ax.set_title(f"{title}\n({pheno.split('_',1)[1]})", fontsize=9)
        ax.set_xticks(range(len(timepoints)))
        ax.set_xticklabels([f"T{t}" for t in timepoints], fontsize=8)
        ax.legend(fontsize=7, loc="best", framealpha=0.8)
        ax.set_ylabel("Mean expression", fontsize=8)
        ax.grid(alpha=0.2)

    fig.suptitle("Marker expression stability across timepoints (mean ± SEM)", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig

def plot_patient_cv(adata, genes=None, key="phenotype", figsize=(16, 6)):
    if genes is None:
        genes = [
            "CD8A","CD4","CX3CR1","S1PR5","KLF2","GZMB","NKG7","GZMK",
            "CD69","ZNF683","TOX","LAG3","PDCD1","CXCL13",
            "TCF7","LEF1","IL7R",
            "FOXP3","IKZF2","CTLA4","TBX21","IFNG","GATA3",
        ]
    expr, genes = extract_expression(adata, genes, key)
    if "patient" not in expr.columns:
        print("No patient column found")
        return None

    pheno_order = [
        "CD8_Activated_TEMRA","CD8_Activated_TEXeff","CD8_Activated_TEXterm",
        "CD8_Activated_TRM","CD8_Quiescent_Naive","CD8_Quiescent_TEXprog",
        "CD4_Naive_Memory","CD4_Exhausted","CD4_Treg","CD4_Th1_polarized",
    ]
    pheno_order = [p for p in pheno_order if p in expr["phenotype"].unique()]

    pat_means = expr.groupby(["phenotype", "patient"])[genes].mean()
    cv = pat_means.groupby("phenotype").agg(lambda x: x.std() / x.mean() if x.mean() > 0.05 else np.nan)
    cv = cv.loc[[p for p in pheno_order if p in cv.index]]

    fig, ax = plt.subplots(figsize=figsize)
    data = cv.values.astype(float)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1.0)

    # Annotate values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cv.index)))
    short = [p.split("_",1)[1] for p in cv.index]
    ax.set_yticklabels(short, fontsize=8)
    plt.colorbar(im, ax=ax, label="CV (std/mean)", shrink=0.8)
    ax.set_title("Cross-patient consistency (CV < 0.3 = highly stable)", fontsize=11, pad=10)
    plt.tight_layout()
    return fig

def plot_foxp3_deepdive(adata, key="phenotype", figsize=(14, 5)):
    genes_compare = ["FOXP3","CXCL13","PDCD1","TOX","IKZF2","IL2RA","CTLA4","ICOS"]
    expr, genes_compare = extract_expression(adata, genes_compare, key)

    phenos = ["CD4_Exhausted", "CD4_Treg"]
    sub = expr[expr["phenotype"].isin(phenos)]

    fig, axes = plt.subplots(1, len(genes_compare), figsize=figsize, sharey=False)
    colors = {"CD4_Exhausted": "#d62728", "CD4_Treg": "#2ca02c"}

    for i, g in enumerate(genes_compare):
        ax = axes[i]
        positions = []
        for j, p in enumerate(phenos):
            data = sub[sub["phenotype"] == p][g].values
            parts = ax.violinplot([data], positions=[j], showmeans=True, showmedians=False, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[p])
                pc.set_alpha(0.6)
            parts["cmeans"].set_color("black")
            parts["cmins"].set_color("gray")
            parts["cmaxes"].set_color("gray")
            parts["cbars"].set_color("gray")

            pct_pos = (data > 0).mean() * 100
            ax.text(j, ax.get_ylim()[1] * 0.05 if i == 0 else data.max() * 1.05,
                   f"{pct_pos:.0f}%+", ha="center", fontsize=6, color="gray")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Exh", "Treg"], fontsize=7)
        ax.set_title(g, fontsize=9)
        if i == 0:
            ax.set_ylabel("Expression", fontsize=8)
        ax.grid(alpha=0.15, axis="y")

    fig.suptitle("CD4_Exhausted vs CD4_Treg: discriminating markers", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig

def plot_dotplot(adata, genes=None, key="phenotype", figsize=(18, 7)):
    if genes is None:
        genes = [
            "CD8A","CD8B","CD4",
            "CX3CR1","S1PR5","KLF2",
            "NKG7","GZMB","PRF1","GZMK",
            "CD69","ZNF683",
            "TOX","LAG3","PDCD1","HAVCR2","CXCL13",
            "TCF7","LEF1","CCR7","SELL","IL7R",
            "FOXP3","IKZF2","IL2RA",
            "TBX21","IFNG","GATA3",
        ]
    expr, genes = extract_expression(adata, genes, key)

    pheno_order = [
        "CD8_Activated_TEMRA","CD8_Activated_TEXeff","CD8_Activated_TEXterm",
        "CD8_Activated_TRM","CD8_Quiescent_Memory","CD8_Quiescent_Naive",
        "CD8_Quiescent_TEXprog",
        "CD4_Naive_Memory","CD4_Exhausted","CD4_Treg",
        "CD4_Th1_polarized","CD4_Th2_polarized",
    ]
    pheno_order = [p for p in pheno_order if p in expr["phenotype"].unique()]

    frac_pos = expr.groupby("phenotype")[genes].apply(lambda x: (x > 0).mean()).loc[pheno_order]
    mean_expr = expr.groupby("phenotype")[genes].mean().loc[pheno_order]

    fig, ax = plt.subplots(figsize=figsize)

    for i, pheno in enumerate(pheno_order):
        for j, gene in enumerate(genes):
            frac = frac_pos.loc[pheno, gene]
            mean_val = mean_expr.loc[pheno, gene]
            size = frac * 200
            if size > 1:
                ax.scatter(j, i, s=size, c=mean_val, cmap="Reds", vmin=0, vmax=2.5,
                          edgecolors="gray", linewidths=0.3)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(pheno_order)))
    short = [abbreviate_phenotype_label(p) for p in pheno_order]
    ax.set_yticklabels(short, fontsize=9)

    # Add horizontal line separating CD8/CD4
    cd4_start = next(i for i, p in enumerate(pheno_order) if "CD4" in p)
    ax.axhline(cd4_start - 0.5, color="black", linewidth=0.8, linestyle="--")

    # Legend for dot size
    for frac_val in [0.25, 0.5, 0.75, 1.0]:
        ax.scatter([], [], s=frac_val * 200, c="gray", alpha=0.5, label=f"{frac_val:.0%}")
    leg = ax.legend(title="% expressing", loc="upper right", fontsize=7, title_fontsize=8,
                   bbox_to_anchor=(1.12, 1))

    sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, 2.5))
    plt.colorbar(sm, ax=ax, label="Mean expression", shrink=0.6, pad=0.15)

    ax.set_title("Dotplot: fraction expressing × mean expression", fontsize=11, pad=10)
    ax.set_xlim(-0.5, len(genes) - 0.5)
    ax.set_ylim(-0.5, len(pheno_order) - 0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def run_validation_plots(adata, key="phenotype", save_prefix="phenotype_validation"):
    figs = {}

    print("Plot 1: Phenotype heatmap...")
    figs["heatmap"] = plot_phenotype_heatmap(adata, key=key)
    figs["heatmap"].savefig(f"{save_prefix}_heatmap.png", dpi=200, bbox_inches="tight")

    print("Plot 2: Tissue stability...")
    figs["tissue"] = plot_tissue_stability(adata, key=key)
    figs["tissue"].savefig(f"{save_prefix}_tissue.png", dpi=200, bbox_inches="tight")

    print("Plot 3: Timepoint stability...")
    figs["timepoint"] = plot_timepoint_stability(adata, key=key)
    figs["timepoint"].savefig(f"{save_prefix}_timepoint.png", dpi=200, bbox_inches="tight")

    print("Plot 4: Patient CV...")
    figs["cv"] = plot_patient_cv(adata, key=key)
    if figs["cv"]:
        figs["cv"].savefig(f"{save_prefix}_patient_cv.png", dpi=200, bbox_inches="tight")

    print("Plot 5: FOXP3 deep dive...")
    figs["foxp3"] = plot_foxp3_deepdive(adata, key=key)
    figs["foxp3"].savefig(f"{save_prefix}_foxp3.png", dpi=200, bbox_inches="tight")

    print("Plot 6: Dotplot...")
    figs["dotplot"] = plot_dotplot(adata, key=key)
    figs["dotplot"].savefig(f"{save_prefix}_dotplot.png", dpi=200, bbox_inches="tight")

    print(f"Done. {len(figs)} figures saved.")
    plt.close("all")
    return figs

