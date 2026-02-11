"""Entrypoint for notebook-style celltyping workflow execution."""

import argparse
import os
from pathlib import Path
import sys
import warnings

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import stcr
import tcri
from matplotlib_venn import venn2, venn3
from scipy import sparse

# Ensure `pipeline/` is importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from celltyping import (
    annotate,
    gene_entropy,
    load_trb,
    phenotype_tcells,
    remove_meaningless_genes,
    run_harmony_workflow,
)
from modules.celltyping_clonality import (
    compute_clonality_patient,
    compute_temporal_correlations,
    plot_clonality_boxplots,
    plot_clonality_boxplots_by_timepoint,
    plot_clonality_heatmap,
    plot_clonality_lines,
    plot_correlation_heatmap,
    plot_top_correlations,
)
from modules.celltyping_geometry import plot_clone_simplex, plot_clone_transitions
from modules.celltyping_io import load_directory_manual, load_directory_scirpy
from modules.celltyping_validation import run_validation_plots
from modules.clone_helpers import infer_lineage_from_phenotype, shorten_phenotype_label


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run celltyping workflow pipeline.")
    parser.add_argument(
        "--data-dir",
        default=str(repo_root / "data" / "btc_gbm_gex_vdj"),
        help="Directory containing paired 10x .h5 and .csv/.csv_clonotypes files.",
    )
    parser.add_argument(
        "--table-sig",
        default="/Users/ceglian/Downloads/41586_2025_9989_MOESM10_ESM.xlsx",
        help="Path to signature table used by phenotype_tcells (Beltra Table S10).",
    )
    parser.add_argument(
        "--table-tf",
        default="/Users/ceglian/Downloads/41586_2025_9989_MOESM6_ESM.xlsx",
        help="Path to TF table (currently retained for parity with notebook workflow).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "pipeline" / "outputs" / "celltyping"),
        help="Directory for all generated figures and outputs.",
    )
    parser.add_argument(
        "--output-h5ad",
        default="GBM_TCR_POS_TCELLS.h5ad",
        help="Final annotated .h5ad filename written under --output-dir.",
    )
    return parser.parse_args()


def main(args):
    warnings.filterwarnings('ignore')
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    directory = args.data_dir
    TABLE_TF = args.table_tf
    TABLE_SIG = args.table_sig

    print(f"Data dir: {directory}")
    print(f"Signature table: {TABLE_SIG}")
    print(f"TF table: {TABLE_TF}")
    print(f"Output dir: {output_dir}")

    adata_scirpy = load_directory_scirpy(directory)
    adata_scirpy
    adata_manual = load_directory_manual(directory, load_trb)
    adata_manual
    adata_manual = annotate(adata_manual)
    adata_scirpy = annotate(adata_scirpy)
    adata_scirpy.obs['trb'] = [str(x) for x in adata_scirpy.obs["IR_VDJ_1_junction_aa"]]
    adata_scirpy = adata_scirpy[adata_scirpy.obs["has_ir"] == "True"].copy()
    adata_scirpy = adata_scirpy[adata_scirpy.obs["trb"]!="nan"].copy()
    adata_scirpy
    adata_manual = adata_manual[adata_manual.obs["trb"]!="None"].copy()
    adata_manual
    adata_manual = adata_manual[adata_scirpy.obs.index].copy()
    adata_scirpy.obs["mait"] = adata_manual.obs["mait"]
    adata_scirpy.obs["inkt"] = adata_manual.obs["inkt"]
    adata_scirpy.obs["trb_all"] = adata_manual.obs["trb"]
    adata_scirpy.obs["tra_all"] = adata_manual.obs["tra"]
    adata = adata_scirpy[adata_scirpy.obs["trb"]!="nan"].copy()
    adata.layers["counts"] = adata.X
    df = adata.obs[["patient", "timepoint", "tissue", "trb"]]
    tissue_trb = {
        tissue: set(df.loc[df["tissue"] == tissue, "trb"].dropna())
        for tissue in df["tissue"].unique()
    }
    tissues = list(tissue_trb.keys())
    if len(tissues) < 2:
        print("Not enough tissues for Venn diagram.")
    elif len(tissues) == 2:
        from matplotlib_venn import venn2
        plt.figure(figsize=(6,6))
        venn2([tissue_trb[tissues[0]], tissue_trb[tissues[1]]], set_labels=(tissues[0], tissues[1]))
        plt.title("Overlap of trb between tissues")
        plt.show()
    else:
        # Pick 3 tissues to visualize, e.g., the first 3 by default
        plt.figure(figsize=(7,7))
        venn3([tissue_trb[tissues[0]], tissue_trb[tissues[1]], tissue_trb[tissues[2]]], set_labels=(tissues[0], tissues[1], tissues[2]))
        plt.title("Overlap of trb between tissues")
        plt.show()
    stcr.pp.register_clonotype_key(adata,"trb")
    stcr.pp.clone_size(adata)
    sns.histplot(data=adata.obs, x="clone_size", hue="tissue",bins=30)
    gene_entropy(adata)
    sns.histplot(adata.var["entropy"])
    adata = remove_meaningless_genes(adata)
    adata_trim=adata[:,adata.var["entropy"]>1.5].copy()
    adata_trim = remove_meaningless_genes(adata_trim,include_tcr=True)
    sc.pp.normalize_total(adata_trim)   
    sc.pp.log1p(adata_trim)
    adata_trim  = run_harmony_workflow(adata_trim,"sample")
    adata.obsm["X_umap"] = adata_trim.obsm["X_umap"]
    sc.pl.umap(adata,color=["patient","timepoint","tissue","clone_size"],ncols=2,frameon=False,add_outline=True,s=10)
    sc.pl.umap(adata,color=["GZMB","LEF1","GZMK","TOX","LAG3","GZMA","GNLY","PRF1","HAVCR2","TIGIT","ENTPD1","SELL","IL7R","NFKB1","EOMES","TCF7","KLRD1","CXCL13","PDCD1","CTLA4"],ncols=3,frameon=False,add_outline=True,s=5)
    sns.countplot(data=adata.obs,x="tissue")
    adata_res=phenotype_tcells(adata, beltra_path=TABLE_SIG)
    sc.pl.matrixplot(adata,["CD8A","CD8B","CD4","GZMB","LEF1","GZMK","TOX","LAG3","GZMA","GNLY","PRF1","HAVCR2","TIGIT","ENTPD1","SELL","IL7R","NFKB1","EOMES","TCF7","KLRD1","CXCL13","PDCD1","CTLA4","FOXP3","ICOS","TBX21","STAT1","CXCR3","IFNG","IL12RB2","GATA3","STAT6","IL4","IL4R","CX3CR1","S1PR5","KLF2"],groupby="phenotype",standard_scale="var")
    genes = [
        "CD8A","CD8B","CD4","GZMB","LEF1","GZMK","TOX","LAG3","GZMA","GNLY",
        "PRF1","HAVCR2","TIGIT","ENTPD1","SELL","IL7R","NFKB1","EOMES","TCF7",
        "KLRD1","CXCL13","PDCD1","CTLA4","FOXP3","ICOS","TBX21","STAT1","CXCR3",
        "IFNG","IL12RB2","GATA3","STAT6","IL4","IL4R","CX3CR1","S1PR5","KLF2",
        # Additional: activation, exhaustion, residency, cycling, trafficking
        "CD69","NKG7","CST7","FGFBP2","FCGR3A","KLRF1","TYROBP",  # effector/NK-like
        "CXCR6","ITGAE","ZNF683","HOBIT",  # tissue residency
        "MKI67","TOP2A",  # proliferation
        "CCR7","S1PR1",  # naive/circulation
        "IL2RA","IKZF2","TNFRSF1B",  # Treg
        "NKG7","GZMH","FCER1G",  # cytotoxicity
        "ISG15","IFIT1",  # IFN-stimulated
        "MAL","BACH2",  # naive
    ]
    genes = list(dict.fromkeys([g for g in genes if g in adata.var_names]))
    if sparse.issparse(adata.X):
        expr = pd.DataFrame(adata[:, genes].X.toarray(), columns=genes, index=adata.obs.index)
    else:
        expr = pd.DataFrame(adata[:, genes].X, columns=genes, index=adata.obs.index)
    expr["phenotype"] = adata.obs["phenotype"].astype(str)
    expr["tissue"] = adata.obs["tissue"].astype(str)
    expr["timepoint"] = adata.obs["timepoint"].astype(str)
    if "patient" in adata.obs.columns:
        expr["patient"] = adata.obs["patient"].astype(str)
    elif "sample" in adata.obs.columns:
        expr["patient"] = adata.obs["sample"].astype(str)
    print("=" * 80)
    print("MEAN EXPRESSION BY PHENOTYPE")
    print("=" * 80)
    pheno_mean = expr.groupby("phenotype")[genes].mean()
    print(pheno_mean.round(2).to_string())
    print("\n" + "=" * 80)
    print("MEAN EXPRESSION BY PHENOTYPE × TISSUE")
    print("=" * 80)
    pt_mean = expr.groupby(["phenotype", "tissue"])[genes].mean()
    print(pt_mean.round(2).to_string())
    print("\n" + "=" * 80)
    print("MEAN EXPRESSION BY PHENOTYPE × TIMEPOINT")
    print("=" * 80)
    ptp_mean = expr.groupby(["phenotype", "timepoint"])[genes].mean()
    print(ptp_mean.round(2).to_string())
    if "patient" in expr.columns:
        print("\n" + "=" * 80)
        print("PHENOTYPE EXPRESSION VARIANCE ACROSS PATIENTS (CV = std/mean)")
        print("=" * 80)
        pat_means = expr.groupby(["phenotype", "patient"])[genes].mean()
        pat_cv = pat_means.groupby("phenotype").agg(lambda x: x.std() / x.mean() if x.mean() > 0.01 else np.nan)
        print(pat_cv.round(2).to_string())
    print("\n" + "=" * 80)
    print("FOXP3 IN CD4_EXHAUSTED vs CD4_TREG")
    print("=" * 80)
    for pheno in ["CD4_Exhausted", "CD4_Treg"]:
        sub = expr[expr["phenotype"] == pheno]
        foxp3 = sub["FOXP3"]
        print(f"\n{pheno} (n={len(sub)}):")
        print(f"  mean={foxp3.mean():.3f}, median={foxp3.median():.3f}, "
              f"% positive (>0): {(foxp3 > 0).mean():.1%}, "
              f"% high (>0.5): {(foxp3 > 0.5).mean():.1%}")
        print(f"  By tissue:")
        for tissue in ["PBMC", "CSF", "TP"]:
            ts = sub[sub["tissue"] == tissue]["FOXP3"]
            if len(ts) > 0:
                print(f"    {tissue} (n={len(ts)}): mean={ts.mean():.3f}, "
                      f"% positive={( ts > 0).mean():.1%}")

        # Also check CTLA4, TIGIT, IL2RA overlap
        print(f"  Co-expression with Treg markers:")
        for g in ["CTLA4", "TIGIT", "IL2RA", "IKZF2", "ICOS"]:
            if g in genes:
                print(f"    {g}: mean={sub[g].mean():.3f}, %pos={(sub[g] > 0).mean():.1%}")
    print("\n" + "=" * 80)
    print("CELL COUNTS: PHENOTYPE × TISSUE × TIMEPOINT")
    print("=" * 80)
    counts = pd.crosstab([expr["phenotype"], expr["tissue"]], expr["timepoint"], margins=True)
    print(counts.to_string())
    warnings.filterwarnings('ignore')
    run_validation_plots(adata)
    print(pd.crosstab(adata.obs["tissue"], adata.obs["timepoint"]))
    print()
    print(pd.crosstab(adata.obs["phenotype"], adata.obs["tissue"]))
    print()
    treg = adata.obs[adata.obs["phenotype"] == "CD4_Treg"]
    print("Treg by tissue × timepoint:")
    print(pd.crosstab(treg["tissue"], treg["timepoint"]))
    sns.countplot(data=adata.obs,x="phenotype")
    plt.xticks(rotation=90)
    sc.tl.rank_genes_groups(adata, "phenotype")
    sc.pl.rank_genes_groups_matrixplot(adata, standard_scale="var", n_genes=3)
    markers = dict()
    for x in adata.obs["phenotype"].unique():
        df = sc.get.rank_genes_groups_df(adata,x)
        markers[x] = df["names"].tolist()[:10]
        print(x, markers[x])
    stcr.pp.register_clonotype_key(adata,"trb")
    stcr.pp.register_phenotype_key(adata,"phenotype")
    stcr.pp.clone_size(adata)
    sns.histplot(data=adata.obs, x="clone_size", hue="tissue")
    clonality = stcr.tl.clonality(adata)
    clonality
    cd8 = adata[adata.obs["phenotype_level1"] == "CD8"].copy()
    stcr.pp.register_clonotype_key(cd8,"trb")
    stcr.pp.register_phenotype_key(cd8,"phenotype")
    stcr.pp.clone_size(cd8)
    cd4 = adata[adata.obs["phenotype_level1"] == "CD4"].copy()
    stcr.pp.register_clonotype_key(cd4,"trb")
    stcr.pp.register_phenotype_key(cd4,"phenotype")
    stcr.pp.clone_size(cd4)
    cd4.uns["stcr_unique_phenotypes"] = [x for x in cd4.uns["stcr_unique_phenotypes"] if "CD4" in x]
    cd8.uns["stcr_unique_phenotypes"] = [x for x in cd8.uns["stcr_unique_phenotypes"] if "CD8" in x]
    stcr.pl.clonality(cd4, groupby = "patient", s=10, figsize=(12,5), rotation=90)
    stcr.pl.clonality(cd8, groupby = "patient", s=10, figsize=(12,5), rotation=90)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    TISSUE_COLORS = {'CSF': '#cd442a', 'PBMC': '#f0bd00', 'TP': '#7e9437'}
    TISSUES = ("PBMC", "CSF", "TP")
    TISSUE_LABELS = ("PBMC", "CSF", "Tumor")
    df = compute_clonality_patient(adata)
    print(f"Total records: {len(df)}, patients: {df['patient'].nunique()}")
    print(f"Phenotype×tissue combos: {df['pheno_tissue'].nunique()}")
    print(f"\nRecords per patient:")
    print(df.groupby("patient").size())
    plot_clonality_heatmap(df, "tissue", "Clonality by Phenotype × Tissue (mean ± SEM)",
                           savepath="clonality_tissue_v2.png")
    plot_clonality_heatmap(df, "timepoint", "Clonality by Phenotype × Timepoint (mean ± SEM)",
                           savepath="clonality_timepoint_v2.png")
    plot_clonality_lines(df, savepath="clonality_lines_v2.png")
    plot_clonality_boxplots(df, group_col="tissue", savepath="clonality_boxplots_tissue.png")
    plot_clonality_boxplots_by_timepoint(df, savepath="clonality_boxplots_timepoint.png")
    corr_df = compute_temporal_correlations(df, min_obs=4)
    print(f"\nCorrelation pairs tested: {len(corr_df)}")
    print(f"Significant (Bonferroni p<0.05): {corr_df['sig'].sum()}")
    print("\nTop 20 positive correlations:")
    print(corr_df[corr_df["sig"]].head(20)[
        ["combo1", "combo2", "rho", "pval_adj", "n_obs"]].to_string(index=False))
    print("\nTop 10 negative correlations:")
    print(corr_df[corr_df["sig"] & (corr_df["rho"] < 0)].tail(10)[
        ["combo1", "combo2", "rho", "pval_adj", "n_obs"]].to_string(index=False))
    cross = corr_df[corr_df["tissue1"] != corr_df["tissue2"]]
    print(f"\nCross-tissue significant correlations: {cross['sig'].sum()}")
    print(cross[cross["sig"]].head(20)[
        ["combo1", "combo2", "rho", "pval_adj", "n_obs"]].to_string(index=False))
    plot_correlation_heatmap(corr_df, savepath="clonality_correlation_heatmap.png")
    plot_top_correlations(corr_df, df, savepath="clonality_top_correlations.png")
    tissue_colors = dict(zip(tissues, tcri.pl.tcri_colors))
    cd8_groups = {
        "Circulating\n(TEMRA/Naive)": ["TEMRA", "Naive"],
        "Exhaustion\n(TEXprog/eff/term)": ["TEXprog", "TEXeff", "TEXterm"],
        "Resident\n(TRM/Memory)": ["TRM", "Memory"],
    }
    cd4_groups = {
        "Naive/Memory": ["Naive_Memory"],
        "Exhausted/Effector\n(Exh/Th1/Th2)": ["Exhausted", "Th1_polarized", "Th2_polarized"],
        "Treg": ["Treg"],
    }
    tissue_colors = {'CSF': '#cd442a', 'PBMC': '#f0bd00', 'TP': '#7e9437'}
    cd8_df = plot_clone_simplex(adata, cd8_groups, tissue_colors=tissue_colors, lineage="CD8", a=50, b=1.8)
    cd4_df = plot_clone_simplex(adata, cd4_groups, tissue_colors=tissue_colors, lineage="CD4", a=50, b=1.8)
    plot_clone_transitions(adata, cd8_groups, "TP", "CSF", lineage="CD8",
                           tissue_labels=("TP", "CSF"),
                            arrow_color="black",
                            arrow_alpha=.7,
                            arrow_width=0.005,
                            arrow_head_width=0.03,
                            arrow_head_length=0.008,
                            point_color_from=tissue_colors["TP"],
                            point_color_to=tissue_colors["CSF"],
                            a=100, b=3)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    tcr = adata[adata.obs["trb"].notna()].copy()
    tcr.obs["level1"] = tcr.obs["phenotype"].apply(infer_lineage_from_phenotype)
    tcr.obs["pheno_short"] = tcr.obs["phenotype"].apply(shorten_phenotype_label)
    tcr.obs["tissue"] = tcr.obs["tissue"].astype(str)
    clone_lineage = tcr.obs.groupby("trb")["level1"].nunique()
    remaining_switching = (clone_lineage > 1).sum()
    print(f"Remaining switching clones: {remaining_switching}")
    clone_tissues = tcr.obs.groupby("trb")["tissue"].nunique()
    shared_clones = clone_tissues[clone_tissues >= 2].index
    shared = tcr[tcr.obs["trb"].isin(shared_clones)].copy()
    if remaining_switching > 0:
        still_switching = clone_lineage[clone_lineage > 1].index
        shared = shared[~shared.obs["trb"].isin(still_switching)]
    print(f"Shared clones: {shared.obs['trb'].nunique()}")
    print(f"Cells from shared clones: {len(shared)}")
    TISSUE_PAIRS = [("PBMC", "TP"), ("PBMC", "CSF"), ("CSF", "TP")]
    PAIR_LABELS = {"PBMC_TP": "PBMC → Tumor", "PBMC_CSF": "PBMC → CSF", "CSF_TP": "CSF → Tumor"}
    for lineage in ["CD8", "CD4"]:
        lin = shared[shared.obs["level1"] == lineage].copy()
        lin_tc = lin.obs.groupby("trb")["tissue"].nunique()
        lin_shared = lin_tc[lin_tc >= 2].index
        lin = lin[lin.obs["trb"].isin(lin_shared)]

        cs = (lin.obs.groupby(["trb", "tissue"])["pheno_short"]
            .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else None)
            .unstack(fill_value=None))

        # --- Pairwise transition matrices ---
        fig, axes = plt.subplots(len(TISSUE_PAIRS), 3, figsize=(22, 6 * len(TISSUE_PAIRS)))

        for row, (t1, t2) in enumerate(TISSUE_PAIRS):
            if t1 not in cs.columns or t2 not in cs.columns:
                for col in range(3):
                    axes[row, col].text(0.5, 0.5, f"No {t1} or {t2} data",
                                         ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].axis('off')
                continue

            pair = cs[cs[t1].notna() & cs[t2].notna()].copy()
            if len(pair) == 0:
                for col in range(3):
                    axes[row, col].axis('off')
                continue

            pair_key = f"{t1}_{t2}"
            label = PAIR_LABELS.get(pair_key, f"{t1} → {t2}")

            trans_norm = pd.crosstab(pair[t1], pair[t2], normalize="index")
            trans_counts = pd.crosstab(pair[t1], pair[t2])

            # Annotate with both percentage and count
            annot = trans_norm.copy().astype(str)
            for i in range(trans_norm.shape[0]):
                for j in range(trans_norm.shape[1]):
                    annot.iloc[i, j] = f"{trans_norm.iloc[i, j]:.2f}\n({trans_counts.iloc[i, j]})"

            sns.heatmap(trans_norm, annot=annot, fmt="", cmap="YlOrRd",
                        ax=axes[row, 0], linewidths=0.5, vmin=0, vmax=1)
            axes[row, 0].set_title(f"{label} (n={len(pair)} clones)", fontsize=11, fontweight='bold')
            axes[row, 0].set_ylabel(f"State in {t1}")
            axes[row, 0].set_xlabel(f"State in {t2}")

            patterns = pair.apply(lambda r: f"{r[t1]} → {r[t2]}", axis=1)
            top = patterns.value_counts().head(15)
            top.plot.barh(ax=axes[row, 1], color="#E8A435")
            axes[row, 1].set_title(f"Top transitions", fontsize=11, fontweight='bold')
            axes[row, 1].set_xlabel("Number of clones")
            axes[row, 1].invert_yaxis()

            same = (pair[t1] == pair[t2]).sum()
            diff = (pair[t1] != pair[t2]).sum()
            axes[row, 2].pie([same, diff],
                             labels=[f"Same state\n(n={same})", f"Different state\n(n={diff})"],
                             colors=["#4CAF50", "#FF5722"], autopct="%1.0f%%", startangle=90,
                             textprops={'fontsize': 10})
            axes[row, 2].set_title(f"State preservation", fontsize=11, fontweight='bold')

            print(f"\n{lineage} {label}: {len(pair)} shared clones")
            print("Counts:")
            print(trans_counts)
            print("\nRow-normalized:")
            print(trans_norm.round(2))

        fig.suptitle(f"{lineage} Clone Trafficking — All Tissue Pairs",
                     fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(f"{lineage.lower()}_transitions_all_pairs.png", dpi=200,
                    bbox_inches='tight', facecolor='white')
        plt.show()

        # --- 3-compartment trajectories ---
        if all(t in cs.columns for t in ["PBMC", "CSF", "TP"]):
            all3 = cs[cs[["PBMC", "CSF", "TP"]].notna().all(axis=1)].copy()
            if len(all3) > 0:
                all3["trajectory"] = all3["PBMC"] + " → " + all3["CSF"] + " → " + all3["TP"]

                # Summarize preservation
                same_all = ((all3["PBMC"] == all3["CSF"]) & (all3["CSF"] == all3["TP"])).sum()
                pbmc_csf_same = (all3["PBMC"] == all3["CSF"]).sum()
                csf_tp_same = (all3["CSF"] == all3["TP"]).sum()
                pbmc_tp_same = (all3["PBMC"] == all3["TP"]).sum()

                print(f"\n{lineage} 3-compartment trajectories ({len(all3)} clones):")
                print(all3["trajectory"].value_counts().head(20))
                print(f"\nState preservation:")
                print(f"  All 3 same: {same_all} ({same_all/len(all3):.0%})")
                print(f"  PBMC=CSF:   {pbmc_csf_same} ({pbmc_csf_same/len(all3):.0%})")
                print(f"  CSF=TP:     {csf_tp_same} ({csf_tp_same/len(all3):.0%})")
                print(f"  PBMC=TP:    {pbmc_tp_same} ({pbmc_tp_same/len(all3):.0%})")

                # Sankey-style summary: group by unique trajectory, show top N
                fig, ax = plt.subplots(figsize=(10, 6))
                top_traj = all3["trajectory"].value_counts().head(15)
                top_traj.plot.barh(ax=ax, color="#7e9437")
                ax.set_title(f"{lineage}: Top 3-Compartment Trajectories (PBMC → CSF → Tumor)\n"
                             f"n={len(all3)} clones in all 3 tissues",
                             fontsize=12, fontweight='bold')
                ax.set_xlabel("Number of clones")
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig(f"{lineage.lower()}_3compartment_trajectories.png", dpi=200,
                            bbox_inches='tight', facecolor='white')
                plt.show()
    print("\n" + "=" * 60)
    print("=== Directionality (cell count dominance) ===")
    print("=" * 60)
    for lineage in ["CD8", "CD4"]:
        lin = shared[shared.obs["level1"] == lineage]
        cc = lin.obs.groupby(["trb", "tissue"]).size().unstack(fill_value=0)

        for t1, t2 in TISSUE_PAIRS:
            if t1 not in cc.columns or t2 not in cc.columns:
                continue
            both = cc[(cc[t1] > 0) & (cc[t2] > 0)]
            t1_dom = (both[t1] > both[t2]).sum()
            t2_dom = (both[t2] > both[t1]).sum()
            eq = (both[t1] == both[t2]).sum()
            label = PAIR_LABELS.get(f"{t1}_{t2}", f"{t1} vs {t2}")
            print(f"\n{lineage} {label} ({len(both)} clones):")
            print(f"  {t1}-dominant: {t1_dom} ({t1_dom/len(both):.0%})")
            print(f"  {t2}-dominant:  {t2_dom} ({t2_dom/len(both):.0%})")
            print(f"  Equal:         {eq} ({eq/len(both):.0%})")
    adata.write(args.output_h5ad)

if __name__ == "__main__":
    main(parse_args())
