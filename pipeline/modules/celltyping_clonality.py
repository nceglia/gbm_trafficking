"""Clonality analyses and visualizations for celltyping."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stcr
from scipy import stats

from modules.clone_helpers import (
    abbreviate_phenotype_label,
    infer_lineage_from_phenotype,
    shorten_phenotype_label,
)

TISSUES = ("PBMC", "CSF", "TP")
TISSUE_LABELS = ("PBMC", "CSF", "Tumor")
TISSUE_COLORS = {"PBMC": "#f0bd00", "CSF": "#cd442a", "TP": "#7e9437"}

def compute_clonality_patient(
    adata,
    patient_key="patient",
    tissue_key="tissue",
    time_key="timepoint",
    phenotype_key="phenotype",
    clone_key="trb",
    min_cells=20,
):
    tcr = adata[adata.obs[clone_key].notna()].copy()
    records = []
    for (pat, tis, tp), idx in tcr.obs.groupby([patient_key, tissue_key, time_key]).groups.items():
        if len(idx) < min_cells:
            continue
        sub = tcr[idx]
        clon = stcr.tl.clonality(sub)
        for pheno, val in clon.items():
            n = int((sub.obs[phenotype_key] == pheno).sum())
            if n >= 5:
                records.append({
                    "patient": pat, "tissue": str(tis), "timepoint": str(tp),
                    "phenotype": pheno, "clonality": val, "n_cells": n,
                })
    df = pd.DataFrame(records)
    df["lineage"] = df["phenotype"].apply(infer_lineage_from_phenotype)
    df["pheno_tissue"] = df["phenotype"] + "_" + df["tissue"]
    return df

def plot_clonality_heatmap(df, columns_col, title, figsize=(14, 6),
                           vmin=0, vmax=0.6, savepath=None):
    pivot_mean = df.pivot_table(index="phenotype", columns=columns_col,
                                values="clonality", aggfunc="mean")
    pivot_sem = df.pivot_table(index="phenotype", columns=columns_col,
                               values="clonality", aggfunc="sem")
    pivot_n = df.pivot_table(index="phenotype", columns=columns_col,
                             values="clonality", aggfunc="count")

    cd8 = sorted([p for p in pivot_mean.index if "CD8" in p])
    cd4 = sorted([p for p in pivot_mean.index if "CD4" in p])
    order = cd8 + cd4
    pivot_mean = pivot_mean.loc[order]
    pivot_sem = pivot_sem.loc[order]
    pivot_n = pivot_n.loc[order]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot_mean.values, aspect='auto', cmap='YlOrRd',
                   vmin=vmin, vmax=vmax)

    ax.set_xticks(range(pivot_mean.shape[1]))
    ax.set_xticklabels(pivot_mean.columns, rotation=45, ha='right')
    ax.set_yticks(range(pivot_mean.shape[0]))
    ax.set_yticklabels(pivot_mean.index)

    for i in range(pivot_mean.shape[0]):
        for j in range(pivot_mean.shape[1]):
            m = pivot_mean.iloc[i, j]
            s = pivot_sem.iloc[i, j]
            n = pivot_n.iloc[i, j]
            if pd.notna(m):
                color = "white" if m > vmax * 0.6 else "black"
                if pd.notna(s) and n > 1:
                    txt = f"{m:.2f}±{s:.2f}\n(n={int(n)})"
                else:
                    txt = f"{m:.2f}"
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=6.5, color=color, fontweight='bold')

    if cd8 and cd4:
        ax.axhline(len(cd8) - 0.5, color='black', linewidth=2, linestyle='--')

    plt.colorbar(im, ax=ax, label="Clonality", shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_clonality_lines(df, figsize=(18, 10), savepath=None):
    cd8_phenos = sorted([p for p in df["phenotype"].unique() if "CD8" in p])
    cd4_phenos = sorted([p for p in df["phenotype"].unique() if "CD4" in p])
    all_phenos = cd8_phenos + cd4_phenos
    n = len(all_phenos)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, pheno in enumerate(all_phenos):
        ax = axes[idx]
        sub = df[df["phenotype"] == pheno]
        for tissue, tlabel in zip(TISSUES, TISSUE_LABELS):
            tsub = sub[sub["tissue"] == tissue]
            if len(tsub) == 0:
                continue
            agg = tsub.groupby("timepoint")["clonality"].agg(["mean", "sem", "count"]).reset_index()
            agg = agg.sort_values("timepoint")
            agg["sem"] = agg["sem"].fillna(0)
            ax.errorbar(agg["timepoint"], agg["mean"], yerr=agg["sem"],
                        fmt='o-', color=TISSUE_COLORS[tissue], label=tlabel,
                        linewidth=2, markersize=5, capsize=3, capthick=1.5)
        ax.set_title(pheno.replace("CD8_", "").replace("CD4_", ""),
                     fontsize=9, fontweight='bold')
        ax.set_ylim(-0.02, 0.75)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Clonality")

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    fig.suptitle("Clonality by phenotype × tissue × timepoint (mean ± SEM across patients)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_clonality_boxplots(df, group_col="tissue", figsize=(12, 5), savepath=None):
    cd8 = df[df["lineage"] == "CD8"]
    cd4 = df[df["lineage"] == "CD4"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, sub, label in zip(axes, [cd8, cd4], ["CD8", "CD4"]):
        groups = sorted(sub[group_col].unique(), key=lambda x: list(TISSUES).index(x) if x in TISSUES else 99)
        data = [sub[sub[group_col] == g]["clonality"].values for g in groups]
        labels = [TISSUE_LABELS[list(TISSUES).index(g)] if g in TISSUES else g for g in groups]
        colors = [TISSUE_COLORS.get(g, "#999") for g in groups]

        bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.6,
                        showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.4))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        for i, (g, d) in enumerate(zip(groups, data)):
            jitter = np.random.default_rng(42).normal(0, 0.05, len(d))
            ax.scatter(np.full(len(d), i + 1) + jitter, d, s=12, alpha=0.4,
                       color='black', zorder=3, edgecolors='none')

        ax.set_title(f"{label} Clonality", fontsize=12, fontweight='bold')
        ax.set_ylabel("Clonality" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle("Clonality by tissue (patient × timepoint × phenotype replicates)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_clonality_boxplots_by_timepoint(df, figsize=(14, 5), savepath=None):
    cd8 = df[df["lineage"] == "CD8"]
    cd4 = df[df["lineage"] == "CD4"]
    timepoints = sorted(df["timepoint"].unique())

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, sub, label in zip(axes, [cd8, cd4], ["CD8", "CD4"]):
        positions = []
        box_data = []
        box_colors = []
        tick_positions = []
        tick_labels = []

        for ti, tp in enumerate(timepoints):
            for ji, tissue in enumerate(TISSUES):
                tsub = sub[(sub["timepoint"] == tp) & (sub["tissue"] == tissue)]
                pos = ti * 4 + ji
                box_data.append(tsub["clonality"].values if len(tsub) > 0 else [])
                box_colors.append(TISSUE_COLORS[tissue])
                positions.append(pos)
            tick_positions.append(ti * 4 + 1)
            tick_labels.append(f"T{tp}")

        bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                        widths=0.7, showfliers=False)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_title(f"{label} Clonality by Timepoint", fontsize=12, fontweight='bold')
        ax.set_ylabel("Clonality" if ax == axes[0] else "")
        ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=TISSUE_COLORS[t], alpha=0.7, label=tl)
                       for t, tl in zip(TISSUES, TISSUE_LABELS)]
    axes[1].legend(handles=legend_elements, fontsize=8, loc='upper right')

    plt.suptitle("Clonality by timepoint × tissue (boxplots over patients)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()

def compute_temporal_correlations(df, min_obs=4):
    """
    For each pair of phenotype×tissue combinations, compute Spearman
    correlation of clonality over time using patient as replicate.

    Each observation = one patient×timepoint with both phenotype×tissue
    combos present. Tests whether two populations expand/contract together.
    """
    combos = df["pheno_tissue"].unique()
    records = []

    for c1, c2 in combinations(combos, 2):
        d1 = df[df["pheno_tissue"] == c1][["patient", "timepoint", "clonality"]].rename(
            columns={"clonality": "clon1"})
        d2 = df[df["pheno_tissue"] == c2][["patient", "timepoint", "clonality"]].rename(
            columns={"clonality": "clon2"})
        merged = d1.merge(d2, on=["patient", "timepoint"])

        if len(merged) < min_obs:
            continue

        rho, pval = stats.spearmanr(merged["clon1"], merged["clon2"])
        records.append({
            "combo1": c1, "combo2": c2,
            "rho": rho, "pval": pval, "n_obs": len(merged),
            "lineage1": infer_lineage_from_phenotype(c1),
            "lineage2": infer_lineage_from_phenotype(c2),
            "tissue1": c1.split("_")[-1],
            "tissue2": c2.split("_")[-1],
        })

    corr_df = pd.DataFrame(records)
    corr_df["pval_adj"] = np.minimum(corr_df["pval"] * len(corr_df), 1.0)  # Bonferroni
    corr_df["sig"] = corr_df["pval_adj"] < 0.05
    return corr_df.sort_values("rho", ascending=False)

def plot_correlation_heatmap(corr_df, min_obs=6, figsize=(16, 14), savepath=None):
    sig = corr_df[(corr_df["n_obs"] >= min_obs)].copy()

    all_combos = sorted(set(sig["combo1"]) | set(sig["combo2"]),
                        key=lambda x: (0 if "CD8" in x else 1, x))

    mat = pd.DataFrame(np.nan, index=all_combos, columns=all_combos)
    pmat = pd.DataFrame(np.nan, index=all_combos, columns=all_combos)
    np.fill_diagonal(mat.values, 1.0)

    for _, row in sig.iterrows():
        mat.loc[row["combo1"], row["combo2"]] = row["rho"]
        mat.loc[row["combo2"], row["combo1"]] = row["rho"]
        pmat.loc[row["combo1"], row["combo2"]] = row["pval_adj"]
        pmat.loc[row["combo2"], row["combo1"]] = row["pval_adj"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(all_combos)))
    ax.set_xticklabels([abbreviate_phenotype_label(c) for c in all_combos],
                       rotation=90, fontsize=7)
    ax.set_yticks(range(len(all_combos)))
    ax.set_yticklabels([abbreviate_phenotype_label(c) for c in all_combos], fontsize=7)

    for i in range(len(all_combos)):
        for j in range(len(all_combos)):
            v = mat.iloc[i, j]
            p = pmat.iloc[i, j]
            if pd.notna(v) and i != j:
                stars = ""
                if pd.notna(p):
                    if p < 0.001:
                        stars = "***"
                    elif p < 0.01:
                        stars = "**"
                    elif p < 0.05:
                        stars = "*"
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(j, i, f"{v:.2f}{stars}", ha='center', va='center',
                        fontsize=5, color=color)

    cd8_count = sum(1 for c in all_combos if "CD8" in c)
    if cd8_count < len(all_combos):
        ax.axhline(cd8_count - 0.5, color='black', linewidth=2)
        ax.axvline(cd8_count - 0.5, color='black', linewidth=2)

    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.7)
    ax.set_title("Temporal correlations of clonality\n(patient×timepoint replicates, Bonferroni-corrected)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()

def plot_top_correlations(corr_df, df, n_top=12, figsize=(18, 12), savepath=None):
    sig_pos = corr_df[corr_df["sig"] & (corr_df["rho"] > 0)].head(n_top // 2)
    sig_neg = corr_df[corr_df["sig"] & (corr_df["rho"] < 0)].tail(n_top // 2)
    top = pd.concat([sig_pos, sig_neg])

    if len(top) == 0:
        print("No significant correlations to plot")
        return

    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    patient_markers = {'DFCI1': 'o', 'DFCI2': 's', 'DFCI3': '^',
                       'DFCI4': 'D', 'DFCI5': 'v', 'MSK1': 'P'}

    for idx, (_, row) in enumerate(top.iterrows()):
        ax = axes[idx]
        d1 = df[df["pheno_tissue"] == row["combo1"]][["patient", "timepoint", "clonality"]].rename(
            columns={"clonality": "x"})
        d2 = df[df["pheno_tissue"] == row["combo2"]][["patient", "timepoint", "clonality"]].rename(
            columns={"clonality": "y"})
        merged = d1.merge(d2, on=["patient", "timepoint"])

        for pat in merged["patient"].unique():
            psub = merged[merged["patient"] == pat]
            marker = patient_markers.get(pat, 'o')
            ax.scatter(psub["x"], psub["y"], marker=marker, s=30, alpha=0.7,
                       label=pat, edgecolors='black', linewidth=0.5)

        c1_short = shorten_phenotype_label(row["combo1"])
        c2_short = shorten_phenotype_label(row["combo2"])
        stars = "***" if row["pval_adj"] < 0.001 else "**" if row["pval_adj"] < 0.01 else "*"
        ax.set_title(f"ρ={row['rho']:.2f}{stars} (n={row['n_obs']})", fontsize=8, fontweight='bold')
        ax.set_xlabel(c1_short, fontsize=7)
        ax.set_ylabel(c2_short, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.legend(fontsize=5, loc='upper left')

    for idx in range(len(top), len(axes)):
        axes[idx].axis('off')

    fig.suptitle("Top correlated phenotype×tissue pairs (clonality over time)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
