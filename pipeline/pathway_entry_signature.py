# %%
"""Tumor-entry transcriptional signature.

A hard question: can we distinguish TP cells that *recently entered* the
tumor from TP cells that have *persisted* there? We don't have a direct
timestamp, but we do have longitudinal clone histories (TCR β chains
sampled across timepoints 1–6 in PBMC, CSF, TP). The trick is to use
each clone's tissue-occupancy trajectory to assign each TP cell a
status:

  - entry_w_prior_circ   : clone's *first* TP appearance, after being
                           seen in PBMC and/or CSF at a prior timepoint.
                           Cleanest "recent entry" group.
  - first_seen_TP_no_prior : clone's first TP appearance, *no* prior
                           PBMC/CSF observation. Ambiguous — could be
                           entry, or a clone we just hadn't sampled yet.
                           Held out of the primary DE.
  - persister_short      : in TP now, and was in TP at 1 prior timepoint.
  - persister_long       : in TP now, and was in TP at ≥2 prior
                           timepoints. The "established resident" group.

The primary DE is entry_w_prior_circ vs persister_long. We pseudobulk
by (patient × tp_class) for the headline contrast, and additionally by
(patient × tp_class × phenotype) for the phenotype-controlled contrast
— because entry cells may differ from persisters partly in *which
phenotypes they are*, not just in what those phenotypes express.

Validation steps:
  (1) Signature score on TP cells should track tp_class monotonically:
      entry > persister_short > persister_long.
  (2) Same clones in circulation (PBMC/CSF) at the entry timepoint
      should already carry an intermediate score (priming).
  (3) Decay across timepoints within persister clones.
  (4) Compare to dissociation/IEG genes (van den Brink-style stress
      panel) so we know the signature isn't just dissociation artifact.

Caveats called out in the README at the bottom of this script.

Usage:
    python pipeline/pathway_entry_signature.py
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
from modules.clone_helpers import infer_lineage_from_phenotype  # noqa: E402
from modules.differential_expression import run_deseq2  # noqa: E402
from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_LABELS,
    TISSUE_ORDER,
)

# %% ---- Config ----
OUT_DIR = REPO_ROOT / "results" / "pathway_entry_signature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CELLS_PB = 10              # minimum cells per pseudobulk sample
MIN_PATIENTS_PER_ARM = 3       # need ≥3 patients with both arms for DESeq2 to be honest
SIG_PADJ = 0.05
SIG_LFC_UP = 0.5               # entry-up genes
SIG_LFC_DN = -0.5              # entry-down genes
TOP_K_SIGNATURE = 50

TP_CLASS_ORDER = ["entry_w_prior_circ", "first_seen_TP_no_prior",
                  "persister_short", "persister_long"]
TP_CLASS_COLORS = {
    "entry_w_prior_circ":      "#d62728",
    "first_seen_TP_no_prior":  "#fdae6b",
    "persister_short":         "#9ecae1",
    "persister_long":          "#08519c",
}

# Dissociation / IEG / stress panel — van den Brink 2017-style, minimum
# we want to verify is not the *entire* signal we're picking up.
DISSOCIATION_GENES = ["FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2",
                      "EGR3", "ATF3", "DUSP1", "DUSP2", "ZFP36", "HSPA1A",
                      "HSPA1B", "HSP90AA1", "HSPB1", "DNAJB1", "BTG1",
                      "BTG2", "NR4A1", "NR4A2", "NR4A3", "GADD45B", "IER2",
                      "IER3", "PPP1R15A", "KLF6"]


# %% ---- Load ----
adata = sc.read(str(paths.H5AD_TCELLS))
print(f"Loaded {adata.n_obs} cells x {adata.n_vars} genes")

obs = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb"]].copy()
obs = obs[obs["trb"].notna() & obs["tissue"].notna() & obs["timepoint"].notna()].copy()
obs["tissue"] = obs["tissue"].astype(str)
obs["timepoint"] = pd.to_numeric(obs["timepoint"], errors="coerce")
obs = obs.dropna(subset=["timepoint"])
obs["timepoint"] = obs["timepoint"].astype(int)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["lineage"] = obs["phenotype"].map(infer_lineage_from_phenotype)


# %% ---- Label tp_class per (patient, clone, timepoint) ----
ptt = (obs.groupby(["patient", "trb", "timepoint"], observed=True)["tissue"]
        .agg(set).reset_index())


def label_per_clone(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("timepoint")
    out: dict[int, str] = {}
    n_prior_tp = 0
    prior_circ_seen = False
    for _, row in g.iterrows():
        t = row["timepoint"]; ts = row["tissue"]
        has_TP = "TP" in ts
        has_circ = ("PBMC" in ts) or ("CSF" in ts)
        if has_TP:
            if n_prior_tp == 0:
                out[t] = "entry_w_prior_circ" if prior_circ_seen else "first_seen_TP_no_prior"
            elif n_prior_tp == 1:
                out[t] = "persister_short"
            else:
                out[t] = "persister_long"
            n_prior_tp += 1
        if has_circ:
            prior_circ_seen = True
    return pd.Series(out, name="tp_class")


print("Labeling per-(patient,clone,timepoint)...")
labs = (ptt.groupby(["patient", "trb"], observed=True)
            .apply(label_per_clone, include_groups=False)
            .reset_index()
            .rename(columns={"level_2": "timepoint", 0: "tp_class"}))
labs["timepoint"] = labs["timepoint"].astype(int)
labs.to_csv(OUT_DIR / "tp_class_labels.csv", index=False)

# Merge labels onto every TP cell
tp_mask = obs["tissue"] == "TP"
tp_obs = obs[tp_mask].merge(labs, on=["patient", "trb", "timepoint"], how="left")
tp_obs["tp_class"] = tp_obs["tp_class"].fillna("unlabeled")

print("\nTP cells per tp_class:")
print(tp_obs["tp_class"].value_counts())
print("\nBy patient:")
pat_tab = tp_obs.groupby(["tp_class", "patient"], observed=True).size().unstack(fill_value=0)
print(pat_tab)
pat_tab.to_csv(OUT_DIR / "tp_class_cells_by_patient.csv")


# %% ---- Pseudobulk + DE: entry vs persister_long, phenotype-aware ----
# Two designs:
#   (a) headline: pseudobulk per (patient × tp_class). Design: ~patient + tp_class.
#                 Top genes here are the "entry signature" including any
#                 phenotype-composition shift between entry and persister.
#   (b) phenotype-controlled: pseudobulk per (patient × tp_class × phenotype).
#                 Design: ~patient + phenotype + tp_class. Top genes are
#                 *within-phenotype* entry differences.
print("\nBuilding pseudobulks...")

# Restrict to TP cells with a clean entry label.
tp_obs_clean = tp_obs[tp_obs["tp_class"].isin(TP_CLASS_ORDER)].copy()
# Index back into adata by .obs_names
adata_tp = adata[tp_obs_clean.index]
adata_tp.obs = adata_tp.obs.copy()
adata_tp.obs["tp_class"] = tp_obs_clean["tp_class"].values
adata_tp.obs["lineage"] = tp_obs_clean["lineage"].values

counts_layer = "counts"


def build_pseudobulks(adata_use, group_cols, min_cells=MIN_CELLS_PB):
    """Sum 'counts' layer per (group_cols) → DataFrame of integer counts + meta."""
    pb_records, meta_records = [], []
    for keys, idx in adata_use.obs.groupby(group_cols, observed=True).groups.items():
        if len(idx) < min_cells:
            continue
        sub = adata_use[idx]
        x = sub.layers[counts_layer]
        if sparse.issparse(x):
            x = x.toarray()
        pb_records.append(np.asarray(x).sum(axis=0))
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys_tuple))
        rec["n_cells"] = int(len(idx))
        meta_records.append(rec)
    counts = pd.DataFrame(np.vstack(pb_records), columns=adata_use.var_names).round().astype(int)
    meta = pd.DataFrame(meta_records)
    meta.index = ["__".join(str(meta.loc[i, k]) for k in group_cols) for i in meta.index]
    counts.index = meta.index
    # Drop all-zero genes
    counts = counts.loc[:, counts.sum() > 0]
    return counts, meta


# --- (a) headline design ---
print("\n[DE-a] headline: ~patient + tp_class, entry_w_prior_circ vs persister_long")
counts_a, meta_a = build_pseudobulks(adata_tp, ["patient", "tp_class"])
meta_a_use = meta_a[meta_a["tp_class"].isin(["entry_w_prior_circ", "persister_long"])].copy()
counts_a_use = counts_a.loc[meta_a_use.index]
# Require ≥3 patients with both arms
pivot = meta_a_use.pivot_table(index="patient", columns="tp_class", values="n_cells", fill_value=0)
patients_both = pivot[(pivot.get("entry_w_prior_circ", 0) > 0) &
                      (pivot.get("persister_long", 0) > 0)].index.tolist()
print(f"  patients with both arms: {patients_both} (n={len(patients_both)})")
meta_a_use = meta_a_use[meta_a_use["patient"].isin(patients_both)]
counts_a_use = counts_a_use.loc[meta_a_use.index]

if len(patients_both) >= MIN_PATIENTS_PER_ARM:
    res_a = run_deseq2(counts_a_use, meta_a_use, "~ patient + tp_class",
                       ["tp_class", "entry_w_prior_circ", "persister_long"])
    res_a = res_a.sort_values("stat", ascending=False)
    res_a.to_csv(OUT_DIR / "deseq2_entry_vs_persister_long_headline.csv")
    print(f"  rows: {len(res_a)}; top 10 up: {res_a.head(10).index.tolist()}")
    print(f"  top 10 down: {res_a.tail(10).index.tolist()}")
else:
    res_a = None
    print(f"  SKIPPED — only {len(patients_both)} patients with both arms")

# --- (b) phenotype-controlled design ---
print("\n[DE-b] phenotype-controlled: ~patient + phenotype + tp_class")
counts_b, meta_b = build_pseudobulks(adata_tp, ["patient", "tp_class", "phenotype"])
meta_b_use = meta_b[meta_b["tp_class"].isin(["entry_w_prior_circ", "persister_long"])].copy()
counts_b_use = counts_b.loc[meta_b_use.index]
# Need ≥2 phenotype levels in design and ≥2 patients
n_phenos = meta_b_use["phenotype"].nunique()
n_patients = meta_b_use["patient"].nunique()
print(f"  pseudobulks: {len(meta_b_use)}; patients: {n_patients}; phenotypes: {n_phenos}")
if len(meta_b_use) >= 8 and n_patients >= MIN_PATIENTS_PER_ARM and n_phenos >= 2:
    res_b = run_deseq2(counts_b_use, meta_b_use,
                       "~ patient + phenotype + tp_class",
                       ["tp_class", "entry_w_prior_circ", "persister_long"])
    res_b = res_b.sort_values("stat", ascending=False)
    res_b.to_csv(OUT_DIR / "deseq2_entry_vs_persister_long_pheno_controlled.csv")
    print(f"  top 10 up (within-phenotype entry): {res_b.head(10).index.tolist()}")
    print(f"  top 10 down: {res_b.tail(10).index.tolist()}")
else:
    res_b = None
    print("  SKIPPED — insufficient pseudobulks/patients/phenotypes")


# %% ---- Build the entry signature (top-K by stat, with phenotype-control annotation) ----
# Use the headline DE (DE-a) to RANK genes — it has the most power. Then
# annotate each gene with whether it survives the phenotype-controlled
# design (DE-b), so the reader can see how much is composition vs
# within-phenotype.
if res_a is None:
    raise SystemExit("No headline DE result — cannot define entry signature")

# Drop TRAV/TRBV/TRAJ/TRBJ TCR gene segments — clonotype-specific by
# construction, not biology.
def _is_tcr_seg(g: str) -> bool:
    return g.startswith(("TRAV", "TRBV", "TRAJ", "TRBJ", "TRAC", "TRBC",
                         "TRDV", "TRGV"))


res_filt = res_a[~res_a.index.to_series().apply(_is_tcr_seg)].copy()
# Drop ribosomal / mitochondrial that dominate composition shifts
res_filt = res_filt[~res_filt.index.str.startswith(("RPS", "RPL", "MT-", "MRPS", "MRPL"))]

# Strict-significant counts (informational)
n_sig_up = ((res_filt["padj"] < SIG_PADJ) & (res_filt["log2FoldChange"] >= SIG_LFC_UP)).sum()
n_sig_dn = ((res_filt["padj"] < SIG_PADJ) & (res_filt["log2FoldChange"] <= SIG_LFC_DN)).sum()
print(f"\n[strict] sig genes (padj<{SIG_PADJ} & |LFC|>0.5) in headline DE: up={n_sig_up}, down={n_sig_dn}")

# Rank-based signature: top K by stat. Filter out NaN stat.
res_filt = res_filt.dropna(subset=["stat"])
entry_up_genes = res_filt.sort_values("stat", ascending=False).head(TOP_K_SIGNATURE).index.tolist()
entry_dn_genes = res_filt.sort_values("stat", ascending=True).head(TOP_K_SIGNATURE).index.tolist()

# Cross-annotate with phenotype-controlled DE
def annotate(genes, direction):
    rows = []
    for g in genes:
        row = res_a.loc[g].rename(lambda c: f"head_{c}").to_dict() if g in res_a.index else {}
        if res_b is not None and g in res_b.index:
            row.update(res_b.loc[g].rename(lambda c: f"phenoctrl_{c}").to_dict())
        row["gene"] = g
        row["direction"] = direction
        rows.append(row)
    return pd.DataFrame(rows)


sig_table = pd.concat([annotate(entry_up_genes, "entry_up"),
                       annotate(entry_dn_genes, "entry_down")], ignore_index=True)
# Reorder columns: gene + direction first, then headline cols, then pheno-ctrl cols
col_order = ["gene", "direction"]
col_order += [c for c in sig_table.columns if c.startswith("head_")]
col_order += [c for c in sig_table.columns if c.startswith("phenoctrl_")]
sig_table = sig_table[[c for c in col_order if c in sig_table.columns]]
sig_table.to_csv(OUT_DIR / "entry_signature_genes.csv", index=False)
print(f"  saved top {TOP_K_SIGNATURE} up + {TOP_K_SIGNATURE} down to entry_signature_genes.csv")

# Survives-phenotype-control: same direction, padj<0.10 in DE-b
if res_b is not None:
    surviving_up = [g for g in entry_up_genes
                    if g in res_b.index and res_b.loc[g, "log2FoldChange"] > 0
                    and res_b.loc[g, "padj"] < 0.10]
    surviving_dn = [g for g in entry_dn_genes
                    if g in res_b.index and res_b.loc[g, "log2FoldChange"] < 0
                    and res_b.loc[g, "padj"] < 0.10]
    print(f"  of those: {len(surviving_up)}/{TOP_K_SIGNATURE} entry-up and "
          f"{len(surviving_dn)}/{TOP_K_SIGNATURE} entry-down survive phenotype control "
          "(same direction, padj<0.10 in DE-b)")
    pd.Series(surviving_up, name="gene").to_csv(
        OUT_DIR / "entry_up_pheno_robust.csv", index=False)
    pd.Series(surviving_dn, name="gene").to_csv(
        OUT_DIR / "entry_down_pheno_robust.csv", index=False)


# %% ---- Score the signature on all cells ----
print("\nScoring signature on all cells (sc.tl.score_genes)...")
# Score against log1p (set as X temporarily for sc.tl.score_genes)
orig_X = adata.X
adata.X = adata.layers["log1p"]

up_present = [g for g in entry_up_genes if g in adata.var_names]
dn_present = [g for g in entry_dn_genes if g in adata.var_names]
if up_present:
    sc.tl.score_genes(adata, gene_list=up_present,
                      score_name="entry_up_score", random_state=42)
else:
    adata.obs["entry_up_score"] = 0.0
if dn_present:
    sc.tl.score_genes(adata, gene_list=dn_present,
                      score_name="entry_down_score", random_state=42)
else:
    adata.obs["entry_down_score"] = 0.0
adata.obs["entry_net_score"] = adata.obs["entry_up_score"] - adata.obs["entry_down_score"]

iegs_present = [g for g in DISSOCIATION_GENES if g in adata.var_names]
sc.tl.score_genes(adata, gene_list=iegs_present,
                  score_name="ieg_dissociation_score", random_state=42)
print(f"  entry_up genes scored: {len(up_present)}, entry_down: {len(dn_present)}, IEG: {len(iegs_present)}")
adata.X = orig_X

# Merge scores back to per-cell df
scored = adata.obs[["patient", "tissue", "timepoint", "phenotype", "trb",
                    "entry_up_score", "entry_down_score", "entry_net_score",
                    "ieg_dissociation_score"]].copy()
scored["cell_id"] = scored.index  # preserve adata cell name for downstream join
scored = scored[scored["trb"].notna() & scored["tissue"].notna() & scored["timepoint"].notna()]
scored["tissue"] = scored["tissue"].astype(str)
scored["timepoint"] = pd.to_numeric(scored["timepoint"], errors="coerce")
scored = scored.dropna(subset=["timepoint"])
scored["timepoint"] = scored["timepoint"].astype(int)
scored["phenotype"] = scored["phenotype"].astype(str)
scored["lineage"] = scored["phenotype"].map(infer_lineage_from_phenotype)

# Attach tp_class labels to TP cells; circulating cells get tissue label.
scored = scored.merge(labs, on=["patient", "trb", "timepoint"], how="left")
scored["status"] = np.where(scored["tissue"] == "TP",
                            scored["tp_class"].fillna("unlabeled_TP"),
                            scored["tissue"])
scored = scored.set_index("cell_id")
scored.to_csv(OUT_DIR / "entry_scores_per_cell.csv")

# Per-(status × patient) summary
status_pat = (scored.groupby(["status", "patient"], observed=True)
                .agg(n_cells=("entry_net_score", "size"),
                     mean_entry=("entry_net_score", "mean"),
                     mean_ieg=("ieg_dissociation_score", "mean"))
                .reset_index())
status_pat.to_csv(OUT_DIR / "entry_score_summary_by_status_patient.csv", index=False)


# %% ---- Validation plots ----
print("\nBuilding validation plots...")

# (1) Score by tp_class within TP — should be monotone
tp_scored = scored[scored["tissue"] == "TP"].copy()
order_tp = [c for c in TP_CLASS_ORDER if c in tp_scored["tp_class"].unique()]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axes[0]
sns.boxplot(data=tp_scored, x="tp_class", y="entry_net_score",
            order=order_tp, palette=TP_CLASS_COLORS, showfliers=False, ax=ax, width=0.55)
ax.set_xlabel("")
ax.set_ylabel("Entry score (up − down)")
ax.tick_params(axis="x", labelrotation=20)
ax.set_title("Entry signature score within TP, by class")

# (2) Patient-level paired
ax = axes[1]
ppt = (tp_scored.groupby(["patient", "tp_class"], observed=True)["entry_net_score"]
        .mean().reset_index())
sns.stripplot(data=ppt, x="tp_class", y="entry_net_score", order=order_tp,
              hue="patient", palette="Dark2", size=8, ax=ax,
              edgecolor="black", linewidth=0.5)
for pid, gdf in ppt.groupby("patient"):
    gdf = gdf.set_index("tp_class").reindex(order_tp).reset_index()
    ax.plot(range(len(gdf)), gdf["entry_net_score"], color="gray", alpha=0.4, linewidth=1)
ax.set_xlabel("")
ax.set_ylabel("Patient mean entry score")
ax.tick_params(axis="x", labelrotation=20)
ax.set_title("Per-patient pairing across classes")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "validation_score_by_tp_class.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# (3) Score across all statuses (PBMC / CSF / TP-classes)
status_order = ["PBMC", "CSF"] + order_tp
fig, ax = plt.subplots(figsize=(10, 4.5))
plot_df = scored[scored["status"].isin(status_order)].copy()
plot_df["status"] = pd.Categorical(plot_df["status"], categories=status_order, ordered=True)
palette = {"PBMC": TISSUE_COLORS["PBMC"], "CSF": TISSUE_COLORS["CSF"]}
palette.update(TP_CLASS_COLORS)
sns.boxplot(data=plot_df, x="status", y="entry_net_score",
            order=status_order, palette=palette, showfliers=False, ax=ax, width=0.6)
ax.set_xlabel("")
ax.set_ylabel("Entry score")
ax.tick_params(axis="x", labelrotation=20)
ax.set_title("Entry signature: circulation → tumor entry → tumor residency")
fig.tight_layout()
fig.savefig(OUT_DIR / "validation_score_circ_to_persister.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# (4) IEG/dissociation correlation — is the entry signature just stress?
fig, ax = plt.subplots(figsize=(5.5, 5))
sub = tp_scored.sample(min(20000, len(tp_scored)), random_state=42)
ax.scatter(sub["ieg_dissociation_score"], sub["entry_net_score"],
           s=2, alpha=0.15, color="#444")
ax.set_xlabel("IEG / dissociation score")
ax.set_ylabel("Entry score")
r = np.corrcoef(tp_scored["ieg_dissociation_score"].values,
                tp_scored["entry_net_score"].values)[0, 1]
ax.set_title(f"Entry vs IEG/dissociation score in TP\nPearson r = {r:.2f}")
fig.tight_layout()
fig.savefig(OUT_DIR / "validation_ieg_vs_entry_score.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  Pearson r (entry, IEG) in TP = {r:.3f}")

# (5) Entry score across phenotypes within TP
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.boxplot(data=tp_scored, x="phenotype", y="entry_net_score",
            order=list(TCELL_PHENOTYPE_ORDER), hue="tp_class",
            hue_order=order_tp, palette=TP_CLASS_COLORS,
            showfliers=False, ax=ax, width=0.7)
ax.set_xticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in TCELL_PHENOTYPE_ORDER],
                   rotation=30, ha="right")
ax.set_xlabel("")
ax.set_ylabel("Entry score")
ax.set_title("Entry score per phenotype × class")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_DIR / "validation_score_by_phenotype.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# %% ---- Signature gene heatmap ----
# Top 25 up + 25 down, mean log1p across statuses
print("\nBuilding signature heatmap...")
top_genes_for_heat = entry_up_genes[:25] + entry_dn_genes[:25]
present = [g for g in top_genes_for_heat if g in adata.var_names]
expr = adata[:, present].layers["log1p"]
if sparse.issparse(expr):
    expr = expr.toarray()
expr = pd.DataFrame(expr, columns=present, index=adata.obs_names)
# scored is indexed by cell_id (= adata.obs_names entries)
cells = [c for c in scored.index if c in expr.index]
expr_meta = scored.loc[cells, ["status"]].join(expr.loc[cells])
expr_meta = expr_meta[expr_meta["status"].isin(status_order)]
heat = expr_meta.groupby("status", observed=True)[present].mean().reindex(status_order)
# Z-score each gene across statuses
heat_z = (heat - heat.mean(axis=0)) / (heat.std(axis=0) + 1e-9)

fig, ax = plt.subplots(figsize=(max(8, 0.25 * len(present)), 4.5))
sns.heatmap(heat_z.T, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
            cbar_kws={"label": "z-score (mean log1p across statuses)"},
            ax=ax, linewidths=0.2, linecolor="white", xticklabels=True, yticklabels=True)
ax.set_xticklabels(status_order, rotation=30, ha="right")
ax.set_ylabel("")
ax.set_title("Top entry-up / entry-down genes across statuses")
fig.tight_layout()
fig.savefig(OUT_DIR / "entry_signature_gene_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)


# %% ---- Caveats summary ----
caveats = [
    "Entry vs persister labels rely on longitudinal sampling: gaps in sampling",
    "(missed timepoints) will mis-label some entries as 'first_seen_TP_no_prior'.",
    "We exclude that ambiguous class from the primary DE.",
    "",
    "Entry cells skew toward certain phenotypes (e.g. CD4_Exhausted in TP) and",
    "patients (DFCI3 dominates), so a per-patient + per-phenotype controlled",
    "design (DE-b) is the rigorous one — DE-a includes composition effects.",
    "",
    "AP-1 / IEG genes (FOS, JUN, KLF6, DUSP1, NR4A1) are notoriously induced by",
    "tissue dissociation. We report the Pearson correlation between the entry",
    "score and a dissociation-IEG control score so the reader can judge how",
    "much of the entry signal is dissociation artifact.",
    "",
    "n=4 patients with both entry and persister_long arms; per-patient mean",
    "tests are underpowered. Treat per-cell p-values as suggestive only.",
]
(OUT_DIR / "CAVEATS.txt").write_text("\n".join(caveats) + "\n")

print("\n" + "=" * 60)
print("DONE — outputs in", OUT_DIR.relative_to(REPO_ROOT))
print("=" * 60)
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
