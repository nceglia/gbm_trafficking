# %%
"""Exploratory KLF6 expression analysis.

Slices KLF6 expression across five views:

  1. per tissue (PBMC vs CSF vs Tumor)
  2. per tissue × timepoint
  3. per phenotype × tissue
  4. per phenotype × tissue × timepoint
  5. CSF+Tumor shared clones vs all other clones

For every slice we write a tidy summary CSV (mean log1p, frac
expressing, n cells, n patients) plus a small figure. A per-cell long
table of KLF6 values with grouping keys is also dumped so downstream
notebooks can build any extra view without rereading the AnnData.

Usage:
    python pipeline/pathway_klf6_analysis.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse
from scipy import stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.clone_helpers import (  # noqa: E402
    infer_lineage_from_phenotype,
    shorten_phenotype_label,
)
from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TIMEPOINT_COLORS,
    TIMEPOINT_ORDER,
    TISSUE_COLORS,
    TISSUE_LABELS,
    TISSUE_ORDER,
)

# %% ---- Config ----
GENE = "KLF6"
LAYER = "log1p"
OUT_DIR = REPO_ROOT / "results" / "pathway_klf6_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS = 5            # require >=5 cells in a group to compute summary stats
MIN_PATIENTS = 2         # for shared-clone test, require >=2 patients to keep a clone


# %% ---- Load AnnData and extract per-cell KLF6 ----
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
if GENE not in adata.var_names:
    raise SystemExit(f"{GENE} missing from adata.var_names")

gene_idx = adata.var_names.get_loc(GENE)
expr = adata.layers[LAYER][:, gene_idx]
if sparse.issparse(expr):
    expr = expr.toarray().ravel()
expr = np.asarray(expr, dtype=np.float32).ravel()

obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb"]].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = obs["timepoint"].astype(str)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)
obs["pheno_short"] = obs["phenotype"].map(shorten_phenotype_label)
obs[f"{GENE}_log1p"] = expr
obs[f"{GENE}_expressing"] = (expr > 0).astype(np.int8)

per_cell_path = OUT_DIR / f"{GENE.lower()}_per_cell.csv"
obs.to_csv(per_cell_path, index=False)
print(f"Wrote per-cell KLF6 table: {per_cell_path} ({len(obs)} rows)")


# %% ---- Helpers ----
def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean log1p, frac expressing, n cells, n patients per group."""
    g = df.groupby(group_cols, observed=True)
    out = g[f"{GENE}_log1p"].agg(["mean", "median", "std", "size"]).reset_index()
    out = out.rename(columns={"mean": "mean_log1p", "median": "median_log1p",
                              "std": "std_log1p", "size": "n_cells"})
    out["frac_expressing"] = g[f"{GENE}_expressing"].mean().values
    out["n_patients"] = g["patient"].nunique().values
    return out[out["n_cells"] >= MIN_CELLS]


def patient_means(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Mean log1p per (patient × group) — used for paired/strip plots."""
    g = df.groupby(["patient"] + group_cols, observed=True)
    out = g[f"{GENE}_log1p"].agg(["mean", "size"]).reset_index()
    out = out.rename(columns={"mean": "mean_log1p", "size": "n_cells"})
    return out[out["n_cells"] >= MIN_CELLS]


def mannwhitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return np.nan, np.nan
    return float(u), float(p)


# %% ---- View 1: KLF6 per tissue ----
print("\n[1/5] KLF6 per tissue")
view1 = summarize(obs, ["tissue"])
view1["tissue"] = pd.Categorical(view1["tissue"], categories=list(TISSUE_ORDER), ordered=True)
view1 = view1.sort_values("tissue")
view1.to_csv(OUT_DIR / "klf6_per_tissue.csv", index=False)

pm1 = patient_means(obs, ["tissue"])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
ax = axes[0]
order = list(TISSUE_ORDER)
sns.boxplot(data=obs, x="tissue", y=f"{GENE}_log1p", order=order,
            palette=TISSUE_COLORS, showfliers=False, ax=ax, width=0.5)
ax.set_ylabel("KLF6 (log1p)"); ax.set_xlabel("")
ax.set_xticklabels([TISSUE_LABELS[t] for t in order])
ax.set_title("Per-cell KLF6 by tissue")

ax = axes[1]
sns.stripplot(data=pm1, x="tissue", y="mean_log1p", order=order,
              hue="patient", dodge=False, jitter=True, ax=ax, palette="Dark2",
              size=6, edgecolor="black", linewidth=0.5)
sns.pointplot(data=pm1, x="tissue", y="mean_log1p", order=order,
              ax=ax, color="black", errorbar="se",
              markers="_", linestyles="none", scale=1.5)
ax.set_ylabel("Mean KLF6 per patient (log1p)"); ax.set_xlabel("")
ax.set_xticklabels([TISSUE_LABELS[t] for t in order])
ax.set_title("Patient-level means by tissue")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_per_tissue.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "klf6_per_tissue.pdf", bbox_inches="tight")
plt.close(fig)

# Pairwise tissue tests on patient means (paired by patient)
pair_rows = []
wide = pm1.pivot_table(index="patient", columns="tissue", values="mean_log1p")
for t1, t2 in [("PBMC", "CSF"), ("PBMC", "TP"), ("CSF", "TP")]:
    if t1 in wide.columns and t2 in wide.columns:
        paired = wide[[t1, t2]].dropna()
        if len(paired) >= 3:
            stat, p = stats.wilcoxon(paired[t1], paired[t2])
            pair_rows.append({"tissue_a": t1, "tissue_b": t2,
                              "n_patients": int(len(paired)),
                              "mean_diff": float((paired[t1] - paired[t2]).mean()),
                              "wilcoxon_stat": float(stat),
                              "wilcoxon_p": float(p)})
pd.DataFrame(pair_rows).to_csv(OUT_DIR / "klf6_per_tissue_pairwise.csv", index=False)


# %% ---- View 2: KLF6 per tissue × timepoint ----
print("[2/5] KLF6 per tissue × timepoint")
view2 = summarize(obs, ["tissue", "timepoint"])
view2["tissue"] = pd.Categorical(view2["tissue"], categories=list(TISSUE_ORDER), ordered=True)
view2["timepoint"] = pd.Categorical(view2["timepoint"], categories=list(TIMEPOINT_ORDER), ordered=True)
view2 = view2.sort_values(["tissue", "timepoint"])
view2.to_csv(OUT_DIR / "klf6_per_tissue_timepoint.csv", index=False)

pm2 = patient_means(obs, ["tissue", "timepoint"])

fig, ax = plt.subplots(figsize=(9, 5))
for tissue in TISSUE_ORDER:
    sub = view2[view2["tissue"] == tissue]
    if sub.empty:
        continue
    ax.plot(sub["timepoint"].astype(str), sub["mean_log1p"],
            "o-", color=TISSUE_COLORS[tissue], label=TISSUE_LABELS[tissue],
            linewidth=2, markersize=8)
    # patient-level scatter behind
    p_sub = pm2[pm2["tissue"] == tissue]
    if not p_sub.empty:
        ax.scatter(p_sub["timepoint"].astype(str), p_sub["mean_log1p"],
                   color=TISSUE_COLORS[tissue], alpha=0.25, s=20, zorder=1)
ax.set_xlabel("Timepoint")
ax.set_ylabel("KLF6 mean log1p")
ax.set_title("KLF6 across tissues over time")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_per_tissue_timepoint.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "klf6_per_tissue_timepoint.pdf", bbox_inches="tight")
plt.close(fig)


# %% ---- View 3: KLF6 per phenotype × tissue ----
print("[3/5] KLF6 per phenotype × tissue")
view3 = summarize(obs, ["phenotype", "tissue"])
view3["tissue"] = pd.Categorical(view3["tissue"], categories=list(TISSUE_ORDER), ordered=True)
view3["phenotype"] = pd.Categorical(view3["phenotype"], categories=list(TCELL_PHENOTYPE_ORDER), ordered=True)
view3 = view3.sort_values(["phenotype", "tissue"])
view3.to_csv(OUT_DIR / "klf6_per_phenotype_tissue.csv", index=False)

# Heatmap: phenotype × tissue, mean log1p
mat = view3.pivot_table(index="phenotype", columns="tissue", values="mean_log1p",
                        observed=False).reindex(list(TCELL_PHENOTYPE_ORDER))
mat = mat[list(TISSUE_ORDER)]
fig, ax = plt.subplots(figsize=(4.5, 5.5))
sns.heatmap(mat, cmap="magma", annot=True, fmt=".2f", cbar_kws={"label": "mean log1p KLF6"},
            ax=ax, linewidths=0.4, linecolor="white")
ax.set_xticklabels([TISSUE_LABELS[t] for t in mat.columns])
ax.set_yticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in mat.index], rotation=0)
ax.set_xlabel("")
ax.set_ylabel("Phenotype")
ax.set_title("KLF6 by phenotype × tissue")
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_per_phenotype_tissue.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "klf6_per_phenotype_tissue.pdf", bbox_inches="tight")
plt.close(fig)

# Companion: frac expressing
mat_frac = view3.pivot_table(index="phenotype", columns="tissue", values="frac_expressing",
                             observed=False).reindex(list(TCELL_PHENOTYPE_ORDER))
mat_frac = mat_frac[list(TISSUE_ORDER)]
fig, ax = plt.subplots(figsize=(4.5, 5.5))
sns.heatmap(mat_frac, cmap="viridis", annot=True, fmt=".2f", cbar_kws={"label": "frac KLF6+"},
            ax=ax, linewidths=0.4, linecolor="white", vmin=0, vmax=1)
ax.set_xticklabels([TISSUE_LABELS[t] for t in mat_frac.columns])
ax.set_yticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in mat_frac.index], rotation=0)
ax.set_xlabel("")
ax.set_ylabel("Phenotype")
ax.set_title("KLF6+ fraction by phenotype × tissue")
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_per_phenotype_tissue_fracexpr.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# %% ---- View 4: KLF6 per phenotype × tissue × timepoint ----
print("[4/5] KLF6 per phenotype × tissue × timepoint")
view4 = summarize(obs, ["phenotype", "tissue", "timepoint"])
view4["tissue"] = pd.Categorical(view4["tissue"], categories=list(TISSUE_ORDER), ordered=True)
view4["timepoint"] = pd.Categorical(view4["timepoint"], categories=list(TIMEPOINT_ORDER), ordered=True)
view4["phenotype"] = pd.Categorical(view4["phenotype"], categories=list(TCELL_PHENOTYPE_ORDER), ordered=True)
view4 = view4.sort_values(["phenotype", "tissue", "timepoint"])
view4.to_csv(OUT_DIR / "klf6_per_phenotype_tissue_timepoint.csv", index=False)

# One panel per phenotype, lines by tissue across timepoints
phenos = [p for p in TCELL_PHENOTYPE_ORDER if p in obs["phenotype"].unique()]
ncols = 4
nrows = int(np.ceil(len(phenos) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.6 * nrows),
                         sharey=True, sharex=True)
axes = np.atleast_2d(axes)
for ax, pheno in zip(axes.ravel(), phenos):
    sub = view4[view4["phenotype"] == pheno]
    for tissue in TISSUE_ORDER:
        s = sub[sub["tissue"] == tissue]
        if s.empty:
            continue
        ax.plot(s["timepoint"].astype(str), s["mean_log1p"], "o-",
                color=TISSUE_COLORS[tissue], label=TISSUE_LABELS[tissue],
                linewidth=1.8, markersize=5)
    ax.set_title(TCELL_PHENOTYPE_LABELS.get(pheno, pheno), fontsize=9)
    ax.tick_params(labelsize=8)
# hide unused
for ax in axes.ravel()[len(phenos):]:
    ax.axis("off")
# shared labels and legend
fig.supxlabel("Timepoint")
fig.supylabel("KLF6 mean log1p")
handles = [plt.Line2D([0], [0], color=TISSUE_COLORS[t], marker="o", lw=2,
                      label=TISSUE_LABELS[t]) for t in TISSUE_ORDER]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("KLF6 over time, per phenotype × tissue", y=1.0)
fig.tight_layout(rect=(0, 0.03, 1, 0.98))
fig.savefig(OUT_DIR / "klf6_per_phenotype_tissue_timepoint.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "klf6_per_phenotype_tissue_timepoint.pdf", bbox_inches="tight")
plt.close(fig)


# %% ---- View 5: CSF+TP shared clones vs other clones ----
print("[5/5] KLF6 in CSF+Tumor shared clones vs others")
clones = obs[obs["trb"].notna()].copy()
clone_tissues = clones.groupby("trb")["tissue"].agg(lambda s: set(s.astype(str)))
shared_trbs = set(clone_tissues[clone_tissues.apply(lambda s: "CSF" in s and "TP" in s)].index)
print(f"  shared CSF+TP clones: {len(shared_trbs)}")

clones["clone_group"] = np.where(clones["trb"].isin(shared_trbs),
                                 "CSF+Tumor_shared", "Other")
group_counts = clones["clone_group"].value_counts().to_dict()
print(f"  cells per clone_group: {group_counts}")

# Stats: per-cell Mann–Whitney, and patient-pooled paired Wilcoxon
overall = clones.groupby("clone_group", observed=True)[f"{GENE}_log1p"]
mw_u, mw_p = mannwhitney(
    clones.loc[clones["clone_group"] == "CSF+Tumor_shared", f"{GENE}_log1p"].values,
    clones.loc[clones["clone_group"] == "Other", f"{GENE}_log1p"].values,
)
overall_stats = pd.DataFrame([{
    "n_cells_shared": int(clones["clone_group"].eq("CSF+Tumor_shared").sum()),
    "n_cells_other":  int(clones["clone_group"].eq("Other").sum()),
    "n_clones_shared": len(shared_trbs),
    "mean_log1p_shared": float(clones.loc[clones["clone_group"] == "CSF+Tumor_shared", f"{GENE}_log1p"].mean()),
    "mean_log1p_other":  float(clones.loc[clones["clone_group"] == "Other", f"{GENE}_log1p"].mean()),
    "frac_expr_shared": float(clones.loc[clones["clone_group"] == "CSF+Tumor_shared", f"{GENE}_expressing"].mean()),
    "frac_expr_other":  float(clones.loc[clones["clone_group"] == "Other", f"{GENE}_expressing"].mean()),
    "mannwhitney_u": mw_u,
    "mannwhitney_p": mw_p,
}])
overall_stats.to_csv(OUT_DIR / "klf6_csf_tumor_shared_overall.csv", index=False)

# Per (clone_group × tissue × phenotype) summary
view5 = summarize(clones, ["clone_group", "tissue", "phenotype"])
view5["tissue"] = pd.Categorical(view5["tissue"], categories=list(TISSUE_ORDER), ordered=True)
view5["phenotype"] = pd.Categorical(view5["phenotype"], categories=list(TCELL_PHENOTYPE_ORDER), ordered=True)
view5 = view5.sort_values(["clone_group", "tissue", "phenotype"])
view5.to_csv(OUT_DIR / "klf6_csf_tumor_shared_by_tissue_phenotype.csv", index=False)

# Per-patient comparison (paired)
patient_grp = clones.groupby(["patient", "clone_group"], observed=True)[f"{GENE}_log1p"].agg(
    ["mean", "size"]).reset_index().rename(columns={"mean": "mean_log1p", "size": "n_cells"})
patient_grp = patient_grp[patient_grp["n_cells"] >= MIN_CELLS]
patient_grp.to_csv(OUT_DIR / "klf6_csf_tumor_shared_per_patient.csv", index=False)

wide_pat = patient_grp.pivot(index="patient", columns="clone_group", values="mean_log1p")
paired_rows = []
if {"CSF+Tumor_shared", "Other"}.issubset(wide_pat.columns):
    paired = wide_pat[["CSF+Tumor_shared", "Other"]].dropna()
    if len(paired) >= 3:
        stat, p = stats.wilcoxon(paired["CSF+Tumor_shared"], paired["Other"])
        paired_rows.append({"n_patients": int(len(paired)),
                            "mean_shared": float(paired["CSF+Tumor_shared"].mean()),
                            "mean_other":  float(paired["Other"].mean()),
                            "mean_diff":   float((paired["CSF+Tumor_shared"] - paired["Other"]).mean()),
                            "wilcoxon_stat": float(stat),
                            "wilcoxon_p":   float(p)})
pd.DataFrame(paired_rows).to_csv(OUT_DIR / "klf6_csf_tumor_shared_paired.csv", index=False)

# Figure: 2-panel — overall split, and per-tissue split
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
sns.violinplot(data=clones, x="clone_group", y=f"{GENE}_log1p",
               order=["Other", "CSF+Tumor_shared"], palette={"Other": "#bdbdbd",
                                                              "CSF+Tumor_shared": "#9467bd"},
               cut=0, inner="quartile", ax=ax)
ax.set_xlabel(""); ax.set_ylabel("KLF6 (log1p)")
sub_lbl = f"n_shared_clones={len(shared_trbs)}; p_MW={mw_p:.2e}" if not np.isnan(mw_p) else f"n_shared_clones={len(shared_trbs)}"
ax.set_title(f"Per-cell KLF6: shared vs other clones\n{sub_lbl}", fontsize=10)

ax = axes[1]
plot_df = clones[clones["tissue"].isin(list(TISSUE_ORDER))]
sns.boxplot(data=plot_df, x="tissue", y=f"{GENE}_log1p",
            hue="clone_group", order=list(TISSUE_ORDER),
            hue_order=["Other", "CSF+Tumor_shared"],
            palette={"Other": "#bdbdbd", "CSF+Tumor_shared": "#9467bd"},
            showfliers=False, ax=ax, width=0.6)
ax.set_xlabel(""); ax.set_ylabel("KLF6 (log1p)")
ax.set_xticklabels([TISSUE_LABELS[t] for t in TISSUE_ORDER])
ax.set_title("KLF6 by tissue, split by clone group", fontsize=10)
ax.legend(frameon=False, fontsize=8, title=None, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_csf_tumor_shared.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "klf6_csf_tumor_shared.pdf", bbox_inches="tight")
plt.close(fig)

# Per-phenotype split (additional view)
plot_df2 = clones[clones["phenotype"].isin(list(TCELL_PHENOTYPE_ORDER))].copy()
plot_df2["phenotype"] = pd.Categorical(plot_df2["phenotype"],
                                       categories=list(TCELL_PHENOTYPE_ORDER), ordered=True)
fig, ax = plt.subplots(figsize=(12, 4.5))
sns.boxplot(data=plot_df2, x="phenotype", y=f"{GENE}_log1p",
            hue="clone_group", order=list(TCELL_PHENOTYPE_ORDER),
            hue_order=["Other", "CSF+Tumor_shared"],
            palette={"Other": "#bdbdbd", "CSF+Tumor_shared": "#9467bd"},
            showfliers=False, ax=ax, width=0.7)
ax.set_xticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in TCELL_PHENOTYPE_ORDER],
                   rotation=30, ha="right")
ax.set_xlabel(""); ax.set_ylabel("KLF6 (log1p)")
ax.set_title("KLF6 by phenotype, split by clone group", fontsize=10)
ax.legend(frameon=False, fontsize=8, title=None, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_csf_tumor_shared_by_phenotype.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# %% ---- Final summary ----
print("\n" + "=" * 60)
print("DONE — outputs in", OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
