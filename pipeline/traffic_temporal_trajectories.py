# %%
"""Temporal trajectories of persistent T cell clones per (tissue × lineage).

Identifies genes (DESeq2) and pathways (decoupler ULM + OLS) whose
expression / activity scales with timepoint within persistent clones,
then computes the cross-tissue contrast representing CSF emptying
(CSF down + TP up).

Reads:
  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad  (Scrublet-filtered)
Writes to results/09_temporal/:
  pseudobulk.h5ad
  genes_<tissue>_<lineage>.csv
  pathways_<tissue>_<lineage>.csv
  pathway_scores_per_pseudobulk.csv
  genes_wide.csv, pathways_wide.csv
  csf_emptying_signature.csv
  gene_volcanoes.png, pathway_volcanoes.png
  csf_tp_scatter_genes.png, csf_tp_scatter_pathways.png
  gene_trajectories.png, pathway_trajectories.png
"""
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import decoupler as dc
import gseapy as gp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.style import (  # noqa: E402
    LINEAGE_COLORS,
    PATHWAY_FAMILY_COLORS,
    TISSUE_COLORS,
    TISSUE_ORDER,
)

DATA_PATH = paths.H5AD_TCELLS_SINGLETS
OUT_DIR = paths.TEMPORAL_TRAJECTORIES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODULES_DIR = REPO_ROOT / "pipeline" / "modules"

TISSUES = list(TISSUE_ORDER)
LINEAGES = ["CD8", "CD4"]
GROUPS = [(t, l) for t in TISSUES for l in LINEAGES]
MIN_CELLS_PB = 5
MIN_PATIENTS = 3
FDR_CSF = 0.10
DPI = 200

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False


# %%
# =========================================================
# Load AnnData + persistent clones
# =========================================================
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

obs = adata.obs[["trb", "tissue", "timepoint", "phenotype", "patient"]].copy()
obs = obs[obs["trb"].notna() & (obs["trb"].astype(str) != "")]
obs["patient"] = obs["patient"].astype(str)
obs["trb"] = obs["trb"].astype(str)
obs["tissue"] = obs["tissue"].astype(str)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["clone_id"] = obs["patient"] + "|" + obs["trb"]
obs["lineage"] = np.where(obs["phenotype"].str.contains("CD8"), "CD8", "CD4")

clone_tissue_n_tp = (obs.groupby(["clone_id", "tissue"], observed=True)
                        ["timepoint"].nunique())
persistent_set = set(
    (cid, tis) for (cid, tis), n in clone_tissue_n_tp.items() if n >= 2
)
obs["persistent"] = [
    (cid, tis) in persistent_set
    for cid, tis in zip(obs["clone_id"], obs["tissue"])
]
obs_p = obs[obs["persistent"]].copy()

pers_counts = (obs_p.drop_duplicates(["clone_id", "tissue", "lineage"])
                    .groupby(["tissue", "lineage"], observed=True)
                    .size().unstack("lineage", fill_value=0)
                    .reindex(TISSUES))
print("\nPersistent clones per (tissue × lineage):")
print(pers_counts.to_string())

for tis in TISSUES:
    tab = (obs_p[obs_p["tissue"] == tis]
           .groupby(["patient", "timepoint"], observed=True).size()
           .unstack("timepoint", fill_value=0))
    print(f"\nPatient × Timepoint cell counts — {tis}:")
    print(tab.to_string())


# %%
# =========================================================
# Pseudobulk per (patient × tissue × timepoint × lineage)
# =========================================================
PB_PATH = OUT_DIR / "pseudobulk.h5ad"
if PB_PATH.exists():
    print(f"\nLoading cached pseudobulk: {PB_PATH.name}")
    pb_adata = sc.read_h5ad(str(PB_PATH))
else:
    print("\nBuilding pseudobulk...")
    obs_pos = pd.Series(np.arange(adata.n_obs), index=adata.obs.index)
    obs_p["pos"] = obs_pos.loc[obs_p.index].values
    counts_layer = adata.layers["counts"]
    groups = obs_p.groupby(["patient", "tissue", "timepoint", "lineage"],
                            observed=True)
    rows, meta_rows = [], []
    for (pat, tis, t, lin), grp in tqdm(
            groups, total=groups.ngroups, desc="  pseudobulk"):
        if len(grp) < MIN_CELLS_PB:
            continue
        row_sum = np.asarray(
            counts_layer[grp["pos"].values, :].sum(axis=0)).ravel()
        rows.append(row_sum)
        meta_rows.append({"patient": pat, "tissue": tis,
                          "timepoint": str(t), "lineage": lin,
                          "n_cells": int(len(grp))})
    meta = pd.DataFrame(meta_rows)
    n_pat_combo = (meta.groupby(["tissue", "timepoint", "lineage"],
                                 observed=True)["patient"].nunique())
    keep_combo = set(n_pat_combo[n_pat_combo >= MIN_PATIENTS].index)
    keep_mask = np.array([
        (r["tissue"], r["timepoint"], r["lineage"]) in keep_combo
        for _, r in meta.iterrows()
    ])
    rows = [r for r, k in zip(rows, keep_mask) if k]
    meta = meta[keep_mask].reset_index(drop=True)
    X = np.vstack(rows).astype(np.float32)
    sample_idx = (meta["patient"] + "|" + meta["tissue"]
                  + "|t" + meta["timepoint"] + "|" + meta["lineage"]).values
    pb_adata = ad.AnnData(
        X=X,
        obs=meta.set_index(pd.Index(sample_idx)),
        var=pd.DataFrame(index=adata.var_names.copy()),
    )
    pb_adata.layers["counts"] = X.copy()
    pb_adata.write_h5ad(PB_PATH)

print(f"  pseudobulk: {pb_adata.n_obs} samples × {pb_adata.n_vars} genes")
print(f"  cells/pseudobulk: median={pb_adata.obs['n_cells'].median():.0f}, "
      f"min={pb_adata.obs['n_cells'].min()}, "
      f"max={pb_adata.obs['n_cells'].max()}")
combos = (pb_adata.obs.groupby(["tissue", "timepoint", "lineage"],
                                observed=True).size())
print("  surviving (tissue × timepoint × lineage):")
print(combos.to_string())


# %%
# =========================================================
# Gene-set library + family filter
# =========================================================
print("\nLoading Hallmark + KEGG gene sets...")
hm_fam = pd.read_csv(MODULES_DIR / "pathway_families_hallmark.tsv", sep="\t")
kegg_fam = pd.read_csv(MODULES_DIR / "pathway_families_kegg.tsv", sep="\t")
fam_map = dict(zip(hm_fam["term"], hm_fam["family"]))
for term, fam in zip(kegg_fam["term"], kegg_fam["family"]):
    if fam == "Disease / infection (excluded)":
        continue
    fam_map[term] = fam
hm_lib = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")
kegg_lib = gp.get_library(name="KEGG_2021_Human", organism="Human")
lib_all = {**hm_lib, **kegg_lib}
net_rows = []
for pw, genes in lib_all.items():
    if pw not in fam_map:
        continue
    for g in genes:
        net_rows.append((pw, g, 1.0))
net = pd.DataFrame(net_rows, columns=["source", "target", "weight"])
N_PATHWAYS = net["source"].nunique()
print(f"  pathways: {N_PATHWAYS}; gene-pathway edges: {len(net)}")


# %%
# =========================================================
# Pre-flight summary + calibration (pydeseq2)
# =========================================================
print(f"\nGenes to fit: {pb_adata.n_vars} × {len(GROUPS)} groups "
      f"= {pb_adata.n_vars * len(GROUPS):,} gene-fits")
print(f"Pathways to fit: {N_PATHWAYS} × {len(GROUPS)} groups "
      f"= {N_PATHWAYS * len(GROUPS):,} pathway-fits")

print("\nCalibration: pydeseq2 on first 500 genes (TP, CD8)...")
_cal_mask = ((pb_adata.obs["tissue"] == "TP")
             & (pb_adata.obs["lineage"] == "CD8")).values
_cal_counts = pd.DataFrame(
    pb_adata.layers["counts"][_cal_mask, :500].astype(int),
    index=pb_adata.obs.index[_cal_mask],
    columns=pb_adata.var_names[:500],
)
_cal_meta = pb_adata.obs.loc[_cal_mask, ["patient", "timepoint"]].copy()
_cal_meta["timepoint"] = _cal_meta["timepoint"].astype(float)
_cal_meta["patient"] = _cal_meta["patient"].astype(str)
_t0 = time.time()
_dds_cal = DeseqDataSet(
    counts=_cal_counts, metadata=_cal_meta,
    design="~ patient + timepoint",
    inference=DefaultInference(n_cpus=1), quiet=True,
)
_dds_cal.deseq2()
_calib_dt = time.time() - _t0
_eta = _calib_dt * (pb_adata.n_vars / 500.0) * len(GROUPS)
print(f"  500g × 1 group: {_calib_dt:.1f}s → "
      f"ETA full ({pb_adata.n_vars} g × {len(GROUPS)} groups): {_eta:.0f}s")


# %%
# =========================================================
# Gene DE per (tissue × lineage) — pydeseq2 + manual Wald
# =========================================================
print("\nFitting pydeseq2 per (tissue × lineage)...")
gene_results = {}
for tis, lin in tqdm(GROUPS, desc="  DESeq2 groups"):
    out_path = OUT_DIR / f"genes_{tis}_{lin}.csv"
    if out_path.exists():
        gene_results[(tis, lin)] = pd.read_csv(out_path)
        continue
    mask = ((pb_adata.obs["tissue"] == tis)
            & (pb_adata.obs["lineage"] == lin)).values
    if mask.sum() < MIN_PATIENTS:
        gene_results[(tis, lin)] = pd.DataFrame()
        continue
    sub_counts = pd.DataFrame(
        pb_adata.layers["counts"][mask].astype(int),
        index=pb_adata.obs.index[mask],
        columns=pb_adata.var_names,
    )
    sub_meta = pb_adata.obs.loc[mask, ["patient", "timepoint"]].copy()
    sub_meta["timepoint"] = sub_meta["timepoint"].astype(float)
    sub_meta["patient"] = sub_meta["patient"].astype(str)
    if sub_meta["patient"].nunique() < 2:
        gene_results[(tis, lin)] = pd.DataFrame()
        continue
    nz = sub_counts.sum(axis=0) > 0
    sub_counts = sub_counts.loc[:, nz]

    dds = DeseqDataSet(
        counts=sub_counts, metadata=sub_meta,
        design="~ patient + timepoint",
        inference=DefaultInference(n_cpus=1), quiet=True,
    )
    dds.deseq2()
    LFC = dds.varm["LFC"]
    cols = list(LFC.columns)
    contrast = np.zeros(len(cols))
    contrast[cols.index("timepoint")] = 1.0
    design_mat = dds.obsm["design_matrix"].values
    size_factors = dds.obs["size_factors"].values
    mu = np.exp(design_mat @ LFC.values.T) * size_factors[:, None]
    ridge = np.diag(np.repeat(1e-6, len(cols)))
    pvals, stats_arr, se_arr = dds.inference.wald_test(
        design_matrix=design_mat,
        disp=dds.var["dispersions"].values,
        lfc=LFC.values, mu=mu, ridge_factor=ridge,
        contrast=contrast, lfc_null=0.0, alt_hypothesis=None,
    )
    log2fc = LFC["timepoint"].values / np.log(2)
    mean_expr = dds.layers["normed_counts"].mean(axis=0)
    valid = np.isfinite(pvals)
    padj = np.full_like(pvals, np.nan)
    if valid.any():
        _, q, _, _ = multipletests(pvals[valid], method="fdr_bh")
        padj[valid] = q
    df = pd.DataFrame({
        "gene": list(dds.var_names),
        "log2FC_per_timepoint": log2fc,
        "stat": stats_arr,
        "se": se_arr,
        "pval": pvals,
        "padj": padj,
        "mean_expression": mean_expr,
    })
    df.to_csv(out_path, index=False)
    gene_results[(tis, lin)] = df


# %%
# =========================================================
# Pathway scoring (decoupler ULM)
# =========================================================
print("\nScoring pseudobulks with decoupler.mt.ulm...")
pb_log = pb_adata.copy()
sc.pp.normalize_total(pb_log, target_sum=1e4)
sc.pp.log1p(pb_log)
_t0 = time.time()
dc.mt.ulm(pb_log, net, verbose=False)
print(f"  decoupler ULM: {time.time() - _t0:.1f}s")
scores = pb_log.obsm["score_ulm"]
scores.to_csv(OUT_DIR / "pathway_scores_per_pseudobulk.csv")
print(f"  scores: {scores.shape[0]} samples × {scores.shape[1]} pathways")


# %%
# =========================================================
# Pathway OLS per (tissue × lineage)
# =========================================================
print("\nCalibration: OLS on 10 pathways (TP, CD8)...")
_cal_pw_mask = ((pb_adata.obs["tissue"] == "TP")
                & (pb_adata.obs["lineage"] == "CD8")).values
_cal_meta_pw = pb_adata.obs.loc[_cal_pw_mask, ["patient", "timepoint"]].copy()
_cal_X = pd.get_dummies(_cal_meta_pw["patient"].astype(str),
                         drop_first=True, dtype=float)
_cal_X.insert(0, "Intercept", 1.0)
_cal_X["timepoint"] = _cal_meta_pw["timepoint"].astype(float).values
_t0 = time.time()
for _pw in scores.columns[:10]:
    _y = scores.loc[pb_adata.obs.index[_cal_pw_mask], _pw].values
    sm.OLS(_y, _cal_X).fit()
_dt_pw = time.time() - _t0
print(f"  10 pw × 1 group: {_dt_pw:.2f}s → "
      f"ETA ({N_PATHWAYS} pw × {len(GROUPS)} groups): "
      f"{_dt_pw * (N_PATHWAYS / 10) * len(GROUPS):.0f}s")

print("\nFitting per-pathway OLS per (tissue × lineage)...")
pw_results = {}
for tis, lin in tqdm(GROUPS, desc="  OLS groups"):
    out_path = OUT_DIR / f"pathways_{tis}_{lin}.csv"
    if out_path.exists():
        pw_results[(tis, lin)] = pd.read_csv(out_path)
        continue
    mask = ((pb_adata.obs["tissue"] == tis)
            & (pb_adata.obs["lineage"] == lin)).values
    if mask.sum() < MIN_PATIENTS:
        pw_results[(tis, lin)] = pd.DataFrame()
        continue
    sub_meta = pb_adata.obs.loc[mask, ["patient", "timepoint"]].copy()
    if sub_meta["patient"].astype(str).nunique() < 2:
        pw_results[(tis, lin)] = pd.DataFrame()
        continue
    X_design = pd.get_dummies(sub_meta["patient"].astype(str),
                               drop_first=True, dtype=float)
    X_design.insert(0, "Intercept", 1.0)
    X_design["timepoint"] = sub_meta["timepoint"].astype(float).values
    sub_scores = scores.loc[pb_adata.obs.index[mask]]
    pw_rows = []
    for pw in sub_scores.columns:
        y = sub_scores[pw].values.astype(float)
        if np.std(y) == 0 or np.isnan(y).any():
            pw_rows.append({"pathway": pw, "beta_timepoint": np.nan,
                            "se": np.nan, "stat": np.nan, "pval": np.nan})
            continue
        try:
            model = sm.OLS(y, X_design).fit()
            pw_rows.append({
                "pathway": pw,
                "beta_timepoint": float(model.params["timepoint"]),
                "se": float(model.bse["timepoint"]),
                "stat": float(model.tvalues["timepoint"]),
                "pval": float(model.pvalues["timepoint"]),
            })
        except Exception:
            pw_rows.append({"pathway": pw, "beta_timepoint": np.nan,
                            "se": np.nan, "stat": np.nan, "pval": np.nan})
    df = pd.DataFrame(pw_rows)
    valid = df["pval"].notna()
    padj = np.full(len(df), np.nan)
    if valid.any():
        _, q, _, _ = multipletests(df.loc[valid, "pval"].values,
                                    method="fdr_bh")
        padj[valid.values] = q
    df["padj"] = padj
    df["family"] = df["pathway"].map(fam_map).fillna("")
    df.to_csv(out_path, index=False)
    pw_results[(tis, lin)] = df


# %%
# =========================================================
# Cross-tissue contrast (wide tables) + emptying signature
# =========================================================
print("\nBuilding cross-tissue contrast tables...")


def _wide(results_dict, feature_col, beta_col):
    out = []
    for lin in LINEAGES:
        parts = []
        for tis in TISSUES:
            df = results_dict.get((tis, lin))
            if df is None or df.empty:
                continue
            tmp = (df[[feature_col, beta_col, "padj"]]
                   .rename(columns={beta_col: f"beta_{tis}",
                                    "padj": f"padj_{tis}"}))
            parts.append(tmp.set_index(feature_col))
        if not parts:
            continue
        m = pd.concat(parts, axis=1, join="outer").reset_index()
        m["lineage"] = lin
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


genes_wide = _wide(gene_results, "gene", "log2FC_per_timepoint")
pw_wide = _wide(pw_results, "pathway", "beta_timepoint")
genes_wide.to_csv(OUT_DIR / "genes_wide.csv", index=False)
pw_wide.to_csv(OUT_DIR / "pathways_wide.csv", index=False)

sig_rows = []
for source_name, df, feature_col in [("gene", genes_wide, "gene"),
                                      ("pathway", pw_wide, "pathway")]:
    if df.empty:
        continue
    needed = {"padj_CSF", "beta_CSF", "beta_TP"}
    if not needed.issubset(df.columns):
        continue
    mask = (
        (df["padj_CSF"] < FDR_CSF)
        & (df["beta_CSF"] < 0)
        & (df["beta_TP"] > 0)
    )
    sig = df[mask].copy()
    sig["type"] = source_name
    sig["feature"] = sig[feature_col]
    sig_rows.append(sig)
signature = (pd.concat(sig_rows, ignore_index=True)
             if sig_rows else pd.DataFrame())
signature.to_csv(OUT_DIR / "csf_emptying_signature.csv", index=False)

print(f"  signature features: {len(signature)}")
if not signature.empty:
    print(signature.groupby(["type", "lineage"]).size().to_string())
    print("\n  Top 20 by |beta_CSF|:")
    cols_show = ["type", "lineage", "feature",
                 "beta_CSF", "beta_TP", "beta_PBMC",
                 "padj_CSF", "padj_TP", "padj_PBMC"]
    cols_show = [c for c in cols_show if c in signature.columns]
    print(signature.assign(_abs=lambda d: d["beta_CSF"].abs())
                   .sort_values("_abs", ascending=False)
                   .head(20)[cols_show].to_string(index=False))


# %%
# =========================================================
# Plot 1: gene volcanoes — 2×3 (rows=lineage, cols=tissue)
# =========================================================
print("\nPlotting gene volcanoes...")
fig, axes = plt.subplots(len(LINEAGES), len(TISSUES),
                         figsize=(13, 8), sharex=False, sharey=False)
for i, lin in enumerate(LINEAGES):
    for j, tis in enumerate(TISSUES):
        ax = axes[i, j]
        df = gene_results.get((tis, lin), pd.DataFrame())
        if df.empty:
            ax.set_title(f"{tis} {lin} (no data)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); continue
        x = df["log2FC_per_timepoint"].values
        y = -np.log10(df["padj"].clip(lower=1e-300).values)
        sig_mask = (df["padj"] < FDR_CSF).values
        ax.scatter(x[~sig_mask], y[~sig_mask], s=3, c="#cccccc",
                   alpha=0.5, edgecolors="none", rasterized=True)
        ax.scatter(x[sig_mask], y[sig_mask], s=4,
                   c=TISSUE_COLORS[tis], alpha=0.8,
                   edgecolors="none", rasterized=True)
        top = df.dropna(subset=["padj"]).nsmallest(15, "padj")
        texts = []
        for _, r in top.iterrows():
            texts.append(ax.text(
                r["log2FC_per_timepoint"],
                -np.log10(max(r["padj"], 1e-300)),
                r["gene"], fontsize=6))
        if HAS_ADJUST and texts:
            try:
                adjust_text(texts, ax=ax, arrowprops=dict(
                    arrowstyle="-", color="#888", lw=0.4))
            except Exception:
                pass
        ax.axvline(0, color="#666", lw=0.5)
        ax.axhline(-np.log10(FDR_CSF), color="#aa3333", lw=0.5, ls="--")
        ax.set_title(f"{tis} · {lin}", fontsize=10, fontweight="bold",
                     color=TISSUE_COLORS[tis])
        ax.set_xlabel("β (log2FC / timepoint)", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{lin}\n−log10(padj)", fontsize=8,
                          color=LINEAGE_COLORS[lin], fontweight="bold")
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
fig.suptitle("Gene-level temporal trajectories", fontsize=12,
             fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(OUT_DIR / "gene_volcanoes.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 2: pathway volcanoes — 2×3
# =========================================================
print("Plotting pathway volcanoes...")
fig, axes = plt.subplots(len(LINEAGES), len(TISSUES),
                         figsize=(13, 8), sharex=False, sharey=False)
for i, lin in enumerate(LINEAGES):
    for j, tis in enumerate(TISSUES):
        ax = axes[i, j]
        df = pw_results.get((tis, lin), pd.DataFrame())
        if df.empty:
            ax.set_title(f"{tis} {lin} (no data)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); continue
        x = df["beta_timepoint"].values
        y = -np.log10(df["padj"].clip(lower=1e-300).values)
        sig_mask = (df["padj"] < FDR_CSF).values
        ax.scatter(x[~sig_mask], y[~sig_mask], s=10, c="#cccccc",
                   alpha=0.6, edgecolors="none")
        ax.scatter(x[sig_mask], y[sig_mask], s=18,
                   c=TISSUE_COLORS[tis], alpha=0.9,
                   edgecolors="black", linewidths=0.3)
        top = df.dropna(subset=["padj"]).nsmallest(15, "padj")
        texts = []
        for _, r in top.iterrows():
            texts.append(ax.text(
                r["beta_timepoint"],
                -np.log10(max(r["padj"], 1e-300)),
                r["pathway"][:30], fontsize=6))
        if HAS_ADJUST and texts:
            try:
                adjust_text(texts, ax=ax, arrowprops=dict(
                    arrowstyle="-", color="#888", lw=0.4))
            except Exception:
                pass
        ax.axvline(0, color="#666", lw=0.5)
        ax.axhline(-np.log10(FDR_CSF), color="#aa3333", lw=0.5, ls="--")
        ax.set_title(f"{tis} · {lin}", fontsize=10, fontweight="bold",
                     color=TISSUE_COLORS[tis])
        ax.set_xlabel("β / timepoint", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{lin}\n−log10(padj)", fontsize=8,
                          color=LINEAGE_COLORS[lin], fontweight="bold")
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
fig.suptitle("Pathway-level temporal trajectories",
             fontsize=12, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(OUT_DIR / "pathway_volcanoes.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 3: CSF vs TP β scatter (genes + pathways), 1×2 per lineage
# =========================================================
print("Plotting CSF vs TP β scatter...")
for source_name, wide_df, feature_col, lim_q in [
    ("genes", genes_wide, "gene", 0.999),
    ("pathways", pw_wide, "pathway", 1.0),
]:
    if wide_df.empty or not {"beta_CSF", "beta_TP"}.issubset(wide_df.columns):
        continue
    fig, axes = plt.subplots(1, len(LINEAGES), figsize=(10, 5),
                             sharex=True, sharey=True)
    for i, lin in enumerate(LINEAGES):
        ax = axes[i]
        sub = wide_df[wide_df["lineage"] == lin].dropna(
            subset=["beta_CSF", "beta_TP"])
        if sub.empty:
            ax.set_title(f"{lin} (no data)"); continue
        sig_mask = ((sub.get("padj_CSF", 1.0) < FDR_CSF)
                    & (sub["beta_CSF"] < 0)
                    & (sub["beta_TP"] > 0))
        x_all = sub["beta_CSF"].values
        y_all = sub["beta_TP"].values
        lim = float(np.quantile(np.abs(np.r_[x_all, y_all]), lim_q))
        ax.axhspan(0, lim, xmin=0, xmax=0.5, color="#fff3d6",
                   alpha=0.6, zorder=0)
        ax.scatter(sub.loc[~sig_mask, "beta_CSF"],
                   sub.loc[~sig_mask, "beta_TP"],
                   s=8, c="#bbbbbb", alpha=0.55,
                   edgecolors="none", rasterized=True)
        ax.scatter(sub.loc[sig_mask, "beta_CSF"],
                   sub.loc[sig_mask, "beta_TP"],
                   s=22, c=LINEAGE_COLORS[lin], alpha=0.95,
                   edgecolors="black", linewidths=0.4)
        if source_name == "pathways":
            top_label = sub[sig_mask].assign(
                _abs=lambda d: d["beta_CSF"].abs()
            ).sort_values("_abs", ascending=False).head(10)
            texts = []
            for _, r in top_label.iterrows():
                texts.append(ax.text(
                    r["beta_CSF"], r["beta_TP"],
                    r[feature_col][:28], fontsize=6))
            if HAS_ADJUST and texts:
                try:
                    adjust_text(texts, ax=ax, arrowprops=dict(
                        arrowstyle="-", color="#888", lw=0.4))
                except Exception:
                    pass
        ax.axvline(0, color="#444", lw=0.6)
        ax.axhline(0, color="#444", lw=0.6)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("β CSF / timepoint", fontsize=9)
        ax.set_ylabel("β TP / timepoint", fontsize=9)
        ax.set_title(f"{lin}  (n_signature={int(sig_mask.sum())})",
                     fontsize=10, fontweight="bold",
                     color=LINEAGE_COLORS[lin])
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"CSF↓ + TP↑ contrast ({source_name})",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"csf_tp_scatter_{source_name}.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# %%
# =========================================================
# Plot 4: gene trajectories (top 6 CSF↓ + top 6 CSF↑ per lineage)
# =========================================================
print("Plotting gene trajectories...")
pb_norm = pb_adata.copy()
sc.pp.normalize_total(pb_norm, target_sum=1e4)
sc.pp.log1p(pb_norm)
expr_df = pd.DataFrame(
    pb_norm.X if not hasattr(pb_norm.X, "toarray") else pb_norm.X.toarray(),
    index=pb_norm.obs.index, columns=pb_norm.var_names,
)
expr_df = expr_df.join(
    pb_norm.obs[["tissue", "timepoint", "lineage", "patient"]])

fig = plt.figure(figsize=(15, 16))
outer = gridspec.GridSpec(len(LINEAGES), 1, figure=fig, hspace=0.45)
for li, lin in enumerate(LINEAGES):
    df_csf = gene_results.get(("CSF", lin), pd.DataFrame())
    if df_csf.empty:
        continue
    sub_sig = df_csf[df_csf["padj"] < FDR_CSF].copy()
    if sub_sig.empty:
        sub_sig = df_csf.copy()
    top_down = (sub_sig[sub_sig["log2FC_per_timepoint"] < 0]
                .nsmallest(6, "log2FC_per_timepoint")["gene"].tolist())
    top_up = (sub_sig[sub_sig["log2FC_per_timepoint"] > 0]
              .nlargest(6, "log2FC_per_timepoint")["gene"].tolist())
    pick = top_down + top_up
    if not pick:
        continue
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 6, subplot_spec=outer[li], hspace=0.45, wspace=0.30,
    )
    for gi, gene in enumerate(pick):
        ax = fig.add_subplot(inner[gi // 6, gi % 6])
        if gene not in expr_df.columns:
            ax.set_title(f"{gene} (missing)", fontsize=7)
            continue
        sub = expr_df[expr_df["lineage"] == lin].copy()
        for tis in TISSUES:
            ssub = sub[sub["tissue"] == tis]
            if ssub.empty:
                continue
            agg = (ssub.groupby("timepoint", observed=True)[gene]
                       .agg(["mean", "sem", "count"]).reset_index())
            agg["timepoint"] = agg["timepoint"].astype(int)
            agg = agg.sort_values("timepoint")
            ax.plot(agg["timepoint"], agg["mean"],
                    color=TISSUE_COLORS[tis], lw=1.5,
                    marker="o", markersize=4)
            ax.fill_between(
                agg["timepoint"],
                agg["mean"] - agg["sem"].fillna(0),
                agg["mean"] + agg["sem"].fillna(0),
                color=TISSUE_COLORS[tis], alpha=0.18, linewidth=0,
            )
        direction = "↓" if gi < 6 else "↑"
        ax.set_title(f"{gene} {direction}", fontsize=8,
                     fontweight="bold")
        ax.set_xlabel("T", fontsize=7)
        ax.set_ylabel("log-norm", fontsize=7)
        ax.tick_params(labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.005, 0.78 - 0.49 * li, lin, fontsize=12,
             fontweight="bold", color=LINEAGE_COLORS[lin],
             rotation=90, va="center")
fig.suptitle("Top CSF-down (left 6) + CSF-up (right 6) gene trajectories",
             fontsize=11, fontweight="bold", y=0.995)
fig.savefig(OUT_DIR / "gene_trajectories.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 5: pathway trajectories — emptying-signature pathways
# =========================================================
print("Plotting pathway trajectories...")
score_df = scores.copy().join(
    pb_adata.obs[["tissue", "timepoint", "lineage", "patient"]])
fig = plt.figure(figsize=(15, 14))
outer = gridspec.GridSpec(len(LINEAGES), 1, figure=fig, hspace=0.45)
for li, lin in enumerate(LINEAGES):
    sig_lin = signature[(signature["type"] == "pathway")
                        & (signature["lineage"] == lin)] if not signature.empty \
                                                          else pd.DataFrame()
    if sig_lin.empty:
        # fall back to top 12 CSF-decreasing pathways
        df_csf = pw_results.get(("CSF", lin), pd.DataFrame())
        if df_csf.empty:
            continue
        pick = (df_csf.sort_values("padj")
                .head(12)["pathway"].tolist())
    else:
        pick = (sig_lin.assign(_abs=lambda d: d["beta_CSF"].abs())
                       .sort_values("_abs", ascending=False)
                       .head(12)["feature"].tolist())
    if not pick:
        continue
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 4, subplot_spec=outer[li], hspace=0.55, wspace=0.30,
    )
    for pi, pw in enumerate(pick):
        ax = fig.add_subplot(inner[pi // 4, pi % 4])
        if pw not in score_df.columns:
            ax.set_title(f"{pw[:30]} (missing)", fontsize=7); continue
        sub = score_df[score_df["lineage"] == lin]
        for tis in TISSUES:
            ssub = sub[sub["tissue"] == tis]
            if ssub.empty:
                continue
            agg = (ssub.groupby("timepoint", observed=True)[pw]
                       .agg(["mean", "sem", "count"]).reset_index())
            agg["timepoint"] = agg["timepoint"].astype(int)
            agg = agg.sort_values("timepoint")
            ax.plot(agg["timepoint"], agg["mean"],
                    color=TISSUE_COLORS[tis], lw=1.6,
                    marker="o", markersize=4, label=tis)
            ax.fill_between(
                agg["timepoint"],
                agg["mean"] - agg["sem"].fillna(0),
                agg["mean"] + agg["sem"].fillna(0),
                color=TISSUE_COLORS[tis], alpha=0.18, linewidth=0,
            )
        fam = fam_map.get(pw, "")
        fam_color = PATHWAY_FAMILY_COLORS.get(fam, "#444")
        ax.set_title(pw[:38], fontsize=7, color=fam_color,
                     fontweight="bold")
        ax.set_xlabel("Timepoint", fontsize=7)
        ax.set_ylabel("ULM score", fontsize=7)
        ax.tick_params(labelsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.text(0.005, 0.78 - 0.49 * li, lin, fontsize=12,
             fontweight="bold", color=LINEAGE_COLORS[lin],
             rotation=90, va="center")
fig.suptitle("Emptying-signature pathway trajectories "
             "(CSF↓, TP↑; top 12 by |β_CSF| per lineage)",
             fontsize=11, fontweight="bold", y=0.995)
fig.savefig(OUT_DIR / "pathway_trajectories.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Summary
# =========================================================
print("\n========== SUMMARY ==========")
for tis, lin in GROUPS:
    g_df = gene_results.get((tis, lin), pd.DataFrame())
    p_df = pw_results.get((tis, lin), pd.DataFrame())
    n_g_sig = int((g_df["padj"] < FDR_CSF).sum()) if not g_df.empty else 0
    n_p_sig = int((p_df["padj"] < FDR_CSF).sum()) if not p_df.empty else 0
    print(f"  {tis} · {lin}: "
          f"{n_g_sig} sig genes, {n_p_sig} sig pathways (padj<{FDR_CSF})")

if signature.empty:
    print("\nEmptying signature: 0 features.")
else:
    print(f"\nEmptying signature totals: "
          f"{signature.groupby(['type','lineage']).size().to_dict()}")
    print("\nTop 5 emptying pathways per lineage by |β_CSF|:")
    sig_p = signature[signature["type"] == "pathway"].copy()
    if not sig_p.empty:
        sig_p["_abs"] = sig_p["beta_CSF"].abs()
        for lin in LINEAGES:
            top = (sig_p[sig_p["lineage"] == lin]
                   .sort_values("_abs", ascending=False)
                   .head(5))
            if top.empty:
                print(f"  [{lin}] (none)")
            else:
                print(f"  [{lin}]")
                for _, r in top.iterrows():
                    print(f"    {r['feature'][:55]:<55}  "
                          f"β_CSF={r['beta_CSF']:+.3f}  "
                          f"β_TP={r['beta_TP']:+.3f}  "
                          f"padj_CSF={r['padj_CSF']:.3g}")

print(f"\nDone. All outputs in {OUT_DIR}")
