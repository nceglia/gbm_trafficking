# %%
"""KLF6 specificity test for CSF→Tumor migrators.

The previous pathway_klf6_analysis showed CSF+Tumor shared clones have
higher KLF6 than other clones, but that result conflates two things:
shared clones are enriched in tumor (which is the tissue with the
highest KLF6 baseline), and the comparison group ("other") is a mix of
PBMC-only, CSF-only, and TP-only clones.

This script asks a tighter question: is KLF6 *specifically* elevated in
clones that traffic between CSF and tumor, beyond what's explained by
the tissues they occupy and the phenotypes they adopt?

Strategy:
  1. Classify every TCR clone by the tissue *set* it occupies:
     PBMC_only, CSF_only, TP_only, PBMC+CSF, PBMC+TP, CSF+TP, all_three.
  2. Within-tissue comparison — for each tissue (PBMC / CSF / TP),
     compare KLF6 across clone classes that include that tissue. The
     key contrast is in TP cells: CSF+TP vs TP_only / PBMC+TP. If KLF6
     marks the migration event rather than tumor residency in general,
     CSF+TP cells in tumor should be higher.
  3. Phenotype-stratified — repeat (2) within each phenotype so the
     comparison isn't driven by a phenotype-composition shift.
  4. Specificity benchmark — repeat (2) for a panel of related genes
     (AP-1 family, activation markers, KLF family relatives) to show
     whether the CSF+TP signal is unique to KLF6 or part of a broader
     stress/activation program.
  5. "Primed in CSF" — within CSF cells, compare CSF+TP migrators to
     CSF_only / PBMC+CSF clones. If KLF6 is induced *before* tumor
     entry, migrators should already be elevated in CSF.

Usage:
    python pipeline/pathway_klf6_specificity.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.clone_helpers import (  # noqa: E402
    infer_lineage_from_phenotype,
    shorten_phenotype_label,
)
from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_LABELS,
    TISSUE_ORDER,
)

# %% ---- Config ----
GENE = "KLF6"
# Benchmark set: AP-1 / IEG / stress-response genes co-discovered in the
# CSF→TP rewiring step, plus a few canonical activation markers, plus
# other KLF family members to test whether the effect is family-wide.
BENCHMARK_GENES = ["JUN", "JUNB", "FOS", "FOSB", "DUSP1", "NR4A1",
                   "EGR1", "ZFP36", "BTG1", "BTG2",
                   "GZMB", "PRF1", "IFNG", "CD69", "TNF",
                   "KLF2", "KLF4", "KLF10", "KLF13"]

LAYER = "log1p"
OUT_DIR = REPO_ROOT / "results" / "pathway_klf6_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS_GROUP = 20      # minimum cells per (tissue × class) group
MIN_CELLS_PHENO = 10      # phenotype-stratified relaxed
MIN_PATIENTS = 2

TRAFFIC_ORDER = ["PBMC_only", "CSF_only", "TP_only",
                 "PBMC+CSF", "PBMC+TP", "CSF+TP", "all_three"]
TRAFFIC_COLORS = {
    "PBMC_only": "#fcae91",
    "CSF_only":  "#9ecae1",
    "TP_only":   "#cccccc",
    "PBMC+CSF":  "#6baed6",
    "PBMC+TP":   "#fb6a4a",
    "CSF+TP":    "#9467bd",
    "all_three": "#54278f",
}


# %% ---- Load ----
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")
genes_used = [g for g in [GENE] + BENCHMARK_GENES if g in adata.var_names]
missing = [g for g in [GENE] + BENCHMARK_GENES if g not in adata.var_names]
if missing:
    print(f"  benchmark genes missing from var: {missing}")
gene_idx = adata.var_names.get_indexer(genes_used)
X = adata.layers[LAYER][:, gene_idx]
if sparse.issparse(X):
    X = X.toarray()
X = np.asarray(X, dtype=np.float32)

obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb"]].copy()
obs = obs[obs["trb"].notna()].copy()
X = X[obs.index.get_indexer(obs.index)]  # noop reindex (obs has been subset on adata indexing)
# Re-extract per the new obs subset (cleaner)
mask = adata.obs["trb"].notna().values
X = adata.layers[LAYER][mask, :][:, gene_idx]
if sparse.issparse(X):
    X = X.toarray()
X = np.asarray(X, dtype=np.float32)

obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = obs["timepoint"].astype(str)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)
obs["pheno_short"] = obs["phenotype"].map(shorten_phenotype_label)
for j, g in enumerate(genes_used):
    obs[g] = X[:, j]
print(f"Per-clone cells with TCR: {len(obs)}")


# %% ---- Assign trafficking class per clone ----
clone_tissues = obs.groupby("trb")["tissue"].agg(lambda s: frozenset(s.astype(str)))


def classify(tset: frozenset) -> str:
    has_p = "PBMC" in tset
    has_c = "CSF" in tset
    has_t = "TP" in tset
    if has_p and has_c and has_t:
        return "all_three"
    if has_c and has_t and not has_p:
        return "CSF+TP"
    if has_p and has_t and not has_c:
        return "PBMC+TP"
    if has_p and has_c and not has_t:
        return "PBMC+CSF"
    if has_t and not has_c and not has_p:
        return "TP_only"
    if has_c and not has_t and not has_p:
        return "CSF_only"
    if has_p and not has_c and not has_t:
        return "PBMC_only"
    return "other"


trb_to_class = clone_tissues.map(classify)
obs["traffic_class"] = obs["trb"].map(trb_to_class).astype(str)

# Clone-level summary (per-clone, not per-cell)
clone_summary = (
    obs.groupby("trb", observed=True)
       .agg(traffic_class=("traffic_class", "first"),
            lineage=("lineage", "first"),
            n_cells=("trb", "size"),
            n_patients=("patient", "nunique"))
       .reset_index()
)
clone_class_counts = (
    clone_summary["traffic_class"].value_counts()
    .reindex(TRAFFIC_ORDER, fill_value=0)
    .rename("n_clones").to_frame()
)
clone_class_counts["n_cells"] = (
    obs.groupby("traffic_class", observed=True).size()
    .reindex(TRAFFIC_ORDER, fill_value=0)
)
clone_class_counts.to_csv(OUT_DIR / "traffic_class_counts.csv")
print("\nClones per trafficking class:")
print(clone_class_counts)


# %% ---- Helper: stats per group ----
def per_group_stats(df: pd.DataFrame, gene: str, group_cols: list[str],
                    min_cells: int = MIN_CELLS_GROUP) -> pd.DataFrame:
    g = df.groupby(group_cols, observed=True)[gene]
    out = g.agg(["mean", "median", "std", "size"]).reset_index()
    out = out.rename(columns={"mean": "mean_log1p", "median": "median_log1p",
                              "std": "std_log1p", "size": "n_cells"})
    out["frac_expressing"] = df.groupby(group_cols, observed=True)[gene].apply(
        lambda s: (s > 0).mean()).values
    out["n_patients"] = df.groupby(group_cols, observed=True)["patient"].nunique().values
    out["gene"] = gene
    return out[out["n_cells"] >= min_cells]


def mw_test(df: pd.DataFrame, gene: str, group_col: str, a: str, b: str) -> dict:
    xa = df.loc[df[group_col] == a, gene].values
    xb = df.loc[df[group_col] == b, gene].values
    if len(xa) < 5 or len(xb) < 5:
        return {"n_a": len(xa), "n_b": len(xb), "mean_a": np.nan, "mean_b": np.nan,
                "mean_diff": np.nan, "mw_u": np.nan, "mw_p": np.nan}
    try:
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
    except ValueError:
        u, p = np.nan, np.nan
    return {"n_a": int(len(xa)), "n_b": int(len(xb)),
            "mean_a": float(np.mean(xa)), "mean_b": float(np.mean(xb)),
            "mean_diff": float(np.mean(xa) - np.mean(xb)),
            "mw_u": float(u) if u is not None else np.nan,
            "mw_p": float(p) if p is not None else np.nan}


# %% ---- (1) Within-tissue summary, all classes ----
print("\n[1] Within-tissue KLF6 by trafficking class")
tissue_class_klf6 = per_group_stats(obs, GENE, ["tissue", "traffic_class"])
tissue_class_klf6["tissue"] = pd.Categorical(tissue_class_klf6["tissue"],
                                             categories=list(TISSUE_ORDER), ordered=True)
tissue_class_klf6["traffic_class"] = pd.Categorical(tissue_class_klf6["traffic_class"],
                                                    categories=TRAFFIC_ORDER, ordered=True)
tissue_class_klf6 = tissue_class_klf6.sort_values(["tissue", "traffic_class"])
tissue_class_klf6.to_csv(OUT_DIR / "klf6_tissue_x_traffic_class.csv", index=False)

# Boxplot per tissue × class
plot_df = obs[obs["traffic_class"].isin(TRAFFIC_ORDER)].copy()
plot_df["traffic_class"] = pd.Categorical(plot_df["traffic_class"],
                                          categories=TRAFFIC_ORDER, ordered=True)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, tissue in zip(axes, TISSUE_ORDER):
    sub = plot_df[plot_df["tissue"] == tissue]
    classes_here = [c for c in TRAFFIC_ORDER if c in sub["traffic_class"].unique()]
    if not classes_here:
        ax.set_title(f"{TISSUE_LABELS[tissue]}: no data")
        continue
    sns.boxplot(data=sub, x="traffic_class", y=GENE,
                order=classes_here, palette=TRAFFIC_COLORS,
                showfliers=False, ax=ax, width=0.65)
    n_per = sub.groupby("traffic_class", observed=True).size()
    ax.set_xticklabels(
        [f"{c}\nn={int(n_per.get(c, 0))}" for c in classes_here],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"{GENE} (log1p)" if tissue == TISSUE_ORDER[0] else "")
    ax.set_title(TISSUE_LABELS[tissue])
fig.suptitle(f"{GENE} within tissue, split by trafficking class", y=1.02)
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_tissue_x_traffic_class.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "klf6_tissue_x_traffic_class.pdf", bbox_inches="tight")
plt.close(fig)


# %% ---- (2) Targeted within-tissue tests ----
print("\n[2] Targeted within-tissue MW tests")
TESTS = [
    # within TP: does CSF entry mark the cells?
    ("TP", "CSF+TP", "TP_only"),
    ("TP", "CSF+TP", "PBMC+TP"),
    ("TP", "all_three", "TP_only"),
    ("TP", "all_three", "PBMC+TP"),
    # within CSF: do future tumor migrators look different from non-migrants?
    ("CSF", "CSF+TP", "CSF_only"),
    ("CSF", "CSF+TP", "PBMC+CSF"),
    ("CSF", "all_three", "CSF_only"),
    ("CSF", "all_three", "PBMC+CSF"),
    # within PBMC: do any traffickers look different in blood?
    ("PBMC", "all_three", "PBMC_only"),
    ("PBMC", "PBMC+TP", "PBMC_only"),
    ("PBMC", "PBMC+CSF", "PBMC_only"),
]
rows = []
for tissue, a, b in TESTS:
    sub = obs[obs["tissue"] == tissue]
    r = mw_test(sub, GENE, "traffic_class", a, b)
    r.update({"tissue": tissue, "class_a": a, "class_b": b})
    rows.append(r)
test_df = pd.DataFrame(rows)[["tissue", "class_a", "class_b", "n_a", "n_b",
                              "mean_a", "mean_b", "mean_diff", "mw_u", "mw_p"]]
test_df.to_csv(OUT_DIR / "klf6_specificity_tests.csv", index=False)
print(test_df.to_string(index=False))


# %% ---- (3) Phenotype-stratified within TP ----
print("\n[3] Phenotype-stratified within TP")
tp = obs[obs["tissue"] == "TP"]
pheno_rows = []
for pheno in TCELL_PHENOTYPE_ORDER:
    sub = tp[tp["phenotype"] == pheno]
    if len(sub) < 50:
        continue
    for contrast in [("CSF+TP", "TP_only"), ("CSF+TP", "PBMC+TP"),
                     ("all_three", "TP_only"), ("all_three", "PBMC+TP")]:
        a, b = contrast
        if a in sub["traffic_class"].values and b in sub["traffic_class"].values:
            r = mw_test(sub, GENE, "traffic_class", a, b)
            r.update({"phenotype": pheno, "class_a": a, "class_b": b})
            pheno_rows.append(r)
pheno_tests = pd.DataFrame(pheno_rows)[
    ["phenotype", "class_a", "class_b", "n_a", "n_b",
     "mean_a", "mean_b", "mean_diff", "mw_u", "mw_p"]
]
pheno_tests.to_csv(OUT_DIR / "klf6_specificity_tp_by_phenotype.csv", index=False)

# Heatmap: phenotype × class within TP, mean log1p
tp_grid = per_group_stats(tp, GENE, ["phenotype", "traffic_class"],
                          min_cells=MIN_CELLS_PHENO)
mat = tp_grid.pivot_table(index="phenotype", columns="traffic_class",
                          values="mean_log1p", observed=False)
mat = mat.reindex([p for p in TCELL_PHENOTYPE_ORDER if p in mat.index])
class_cols = [c for c in TRAFFIC_ORDER if c in mat.columns]
mat = mat[class_cols]

fig, ax = plt.subplots(figsize=(0.9 * len(class_cols) + 2, 5))
sns.heatmap(mat, cmap="magma", annot=True, fmt=".2f",
            cbar_kws={"label": f"mean log1p {GENE}"},
            ax=ax, linewidths=0.4, linecolor="white")
ax.set_xticklabels(class_cols, rotation=30, ha="right")
ax.set_yticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in mat.index], rotation=0)
ax.set_xlabel("Trafficking class")
ax.set_ylabel("")
ax.set_title(f"{GENE} in TP — phenotype × trafficking class")
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_tp_phenotype_x_class.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# %% ---- (4) Specificity benchmark across genes ----
print("\n[4] Specificity benchmark across genes")
bench_rows = []
for gene in genes_used:
    for tissue, a, b in TESTS:
        sub = obs[obs["tissue"] == tissue]
        r = mw_test(sub, gene, "traffic_class", a, b)
        r.update({"gene": gene, "tissue": tissue, "class_a": a, "class_b": b})
        bench_rows.append(r)
bench_df = pd.DataFrame(bench_rows)[
    ["gene", "tissue", "class_a", "class_b", "n_a", "n_b",
     "mean_a", "mean_b", "mean_diff", "mw_u", "mw_p"]
]
bench_df.to_csv(OUT_DIR / "klf6_specificity_benchmark.csv", index=False)

# Visual: heatmap of mean_diff (a − b) for each gene × contrast
bench_df["contrast"] = bench_df["tissue"] + ": " + bench_df["class_a"] + " vs " + bench_df["class_b"]
contrast_order = bench_df["contrast"].drop_duplicates().tolist()
gene_order = [GENE] + [g for g in genes_used if g != GENE]
mat = (bench_df.pivot_table(index="gene", columns="contrast", values="mean_diff")
       .reindex(gene_order)[contrast_order])
# Significance asterisks
sig = (bench_df.pivot_table(index="gene", columns="contrast", values="mw_p")
       .reindex(gene_order)[contrast_order])

annot = sig.copy().astype(object)
for i in annot.index:
    for c in annot.columns:
        p = sig.loc[i, c]
        d = mat.loc[i, c]
        if pd.isna(p) or pd.isna(d):
            annot.loc[i, c] = ""
        else:
            star = ""
            if p < 1e-10: star = "***"
            elif p < 1e-5: star = "**"
            elif p < 1e-2: star = "*"
            annot.loc[i, c] = f"{d:+.2f}{star}"

vmax = float(np.nanmax(np.abs(mat.values))) if np.isfinite(np.nanmax(np.abs(mat.values))) else 1.0
fig, ax = plt.subplots(figsize=(0.85 * len(contrast_order) + 3, 0.4 * len(gene_order) + 2))
sns.heatmap(mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
            annot=annot.values, fmt="", linewidths=0.4, linecolor="white",
            cbar_kws={"label": "mean log1p (class_a − class_b)"}, ax=ax,
            annot_kws={"fontsize": 7})
ax.set_xticklabels(contrast_order, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(gene_order, rotation=0, fontsize=9)
for tick in ax.get_yticklabels():
    if tick.get_text() == GENE:
        tick.set_fontweight("bold")
ax.set_title(f"Specificity of {GENE} vs benchmark genes\n(* p<0.01, ** p<1e-5, *** p<1e-10)")
ax.set_xlabel(""); ax.set_ylabel("")
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_specificity_benchmark.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "klf6_specificity_benchmark.pdf", bbox_inches="tight")
plt.close(fig)


# %% ---- (5) Patient-level paired test ----
print("\n[5] Patient-level paired test in TP: CSF+TP vs TP_only")
tp = obs[obs["tissue"] == "TP"]
patient_means = (
    tp[tp["traffic_class"].isin(["CSF+TP", "TP_only", "PBMC+TP", "all_three"])]
       .groupby(["patient", "traffic_class"], observed=True)[GENE]
       .agg(["mean", "size"]).reset_index()
       .rename(columns={"mean": "mean_log1p", "size": "n_cells"})
)
patient_means = patient_means[patient_means["n_cells"] >= MIN_CELLS_GROUP]
patient_means.to_csv(OUT_DIR / "klf6_tp_patient_means_by_class.csv", index=False)

paired_rows = []
for contrast in [("CSF+TP", "TP_only"), ("CSF+TP", "PBMC+TP"),
                 ("all_three", "TP_only"), ("all_three", "PBMC+TP")]:
    a, b = contrast
    wide = patient_means.pivot(index="patient", columns="traffic_class",
                               values="mean_log1p")
    if a in wide.columns and b in wide.columns:
        paired = wide[[a, b]].dropna()
        if len(paired) >= 3:
            stat, p = stats.wilcoxon(paired[a], paired[b])
            paired_rows.append({"class_a": a, "class_b": b,
                                "n_patients": int(len(paired)),
                                "mean_a": float(paired[a].mean()),
                                "mean_b": float(paired[b].mean()),
                                "mean_diff": float((paired[a] - paired[b]).mean()),
                                "wilcoxon_stat": float(stat),
                                "wilcoxon_p": float(p)})
paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(OUT_DIR / "klf6_tp_paired_patient_tests.csv", index=False)
print(paired_df.to_string(index=False) if len(paired_df) else "  (no testable contrasts)")

# Patient-level plot
fig, ax = plt.subplots(figsize=(7, 4))
order_present = [c for c in TRAFFIC_ORDER
                 if c in patient_means["traffic_class"].unique()]
sns.boxplot(data=patient_means, x="traffic_class", y="mean_log1p",
            order=order_present, palette=TRAFFIC_COLORS, ax=ax,
            showfliers=False, width=0.5)
sns.stripplot(data=patient_means, x="traffic_class", y="mean_log1p",
              order=order_present, hue="patient", dodge=False,
              palette="Dark2", size=6, edgecolor="black", linewidth=0.5, ax=ax)
ax.set_xlabel(""); ax.set_ylabel(f"Patient mean {GENE} (log1p) in TP")
ax.set_title("Per-patient KLF6 in tumor, by trafficking class")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
ax.tick_params(axis="x", labelrotation=20)
fig.tight_layout()
fig.savefig(OUT_DIR / "klf6_tp_patient_means_by_class.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# %% ---- Final ----
print("\n" + "=" * 60)
print("DONE — additional KLF6-specificity outputs in",
      OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.glob("klf6_specificity*")) + \
         sorted(OUT_DIR.glob("klf6_tissue_x_traffic_class*")) + \
         sorted(OUT_DIR.glob("klf6_tp_*")) + \
         [OUT_DIR / "traffic_class_counts.csv"]:
    if f.exists():
        print(f"  {f.name}")
