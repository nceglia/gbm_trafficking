# %%
"""Clonal trafficking archetypes from tissue-occupancy trajectories.

Builds a per-clone feature vector (occupancy + presence + scalars over
the (tissue, timepoint) grid), runs Leiden + hierarchical clustering
sweeps, picks the best silhouette, then compares against a null where
clone observation patterns are bootstrapped from real clones but
tissue assignments are re-rolled by the empirical Q transition matrix.

Reads:
  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad   (paths.H5AD_TCELLS)
  results/traffic_migration_rates/migration_rates.csv
  results/traffic_migration_rates/P_empirical.csv

Writes to results/traffic_archetypes/:
  clone_archetypes.csv, archetype_prototypes.npz, archetype_summary.csv,
  null_comparison.csv, silhouette_sweep.csv,
  silhouette_sweep.png, archetype_prototypes.png, archetype_umap.png,
  null_comparison.png, archetype_phenotypes.png
"""
import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.cluster.hierarchy as sch
from scipy.stats import entropy as shannon_entropy
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules import paths  # noqa: E402
from modules.style import (  # noqa: E402
    LINEAGE_COLORS,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_ORDER,
)

OUT_DIR = paths.TRAFFIC_ARCHETYPES_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = list(TISSUE_ORDER)
TISSUE_IDX = {t: i for i, t in enumerate(TISSUES)}
SIL_SUBSAMPLE = 5000
N_NULL = 5000
LEIDEN_RESOLUTIONS = [0.3, 0.5, 0.8, 1.0, 1.5]
HIER_KS = [4, 5, 6, 7, 8]
DPI = 200

# Feature weights. Cell-count-weighted occupancy drowns the CSF signal
# (median 1 cell per CSF observation), so we upweight the binary presence
# mask and add explicit transition indicators that capture trafficking
# direction independent of cell count.
W_OCCUPANCY = 1.0
W_PRESENCE = 2.0
W_TRANSITION = 3.0


# %%
# =========================================================
# Load adata + eligible clones
# =========================================================
print(f"Loading {paths.H5AD_TCELLS.name}...")
adata = sc.read(str(paths.H5AD_TCELLS))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")

obs = adata.obs[["trb", "tissue", "timepoint", "phenotype",
                  "patient", "sample"]].copy()
obs = obs[obs["trb"].notna() & (obs["trb"].astype(str) != "")]
for c in ("trb", "tissue", "timepoint", "phenotype", "patient", "sample"):
    obs[c] = obs[c].astype(str)
obs["clone_id"] = obs["patient"] + "|" + obs["trb"]
obs["lineage"] = np.where(obs["phenotype"].str.contains("CD8"), "CD8", "CD4")
obs = obs[obs["tissue"].isin(TISSUES)]

TPS = sorted(obs["timepoint"].unique(), key=lambda s: int(s))
T = len(TPS)
TP_IDX = {t: i for i, t in enumerate(TPS)}
print(f"  timepoints: {TPS}  (T={T})")

bin_obs = (obs.groupby(["clone_id", "tissue", "timepoint"], observed=True)
              .size().reset_index(name="n_cells"))
n_bins_per_clone = (bin_obs.groupby("clone_id")["n_cells"].size())
eligible_clones = n_bins_per_clone[n_bins_per_clone >= 2].index.tolist()
obs_e = obs[obs["clone_id"].isin(eligible_clones)].copy()
bin_obs_e = bin_obs[bin_obs["clone_id"].isin(eligible_clones)].copy()
print(f"\nEligible clones (≥2 (tissue, timepoint) observations): "
      f"{len(eligible_clones):,}")

clone_meta = (obs_e.groupby("clone_id", observed=True)
              .agg(patient=("patient", "first"),
                   trb=("trb", "first"),
                   lineage=("lineage", lambda s: s.mode().iloc[0]),
                   clone_size=("clone_id", "size"),
                   n_observations=("tissue", "size"))
              .reset_index())
print("\nLineage breakdown:")
print(clone_meta["lineage"].value_counts().to_string())
print("\nClone-size distribution:")
print(clone_meta["clone_size"].describe(
    percentiles=[.25, .5, .75, .9, .99]).to_string())

print("\nN observed clones per (tissue × timepoint):")
ttp = (bin_obs_e.groupby(["tissue", "timepoint"])["clone_id"]
       .nunique().unstack("timepoint", fill_value=0).reindex(TISSUES))
print(ttp.to_string())


# %%
# =========================================================
# Feature extraction (with calibration)
# =========================================================
print("\nExtracting per-clone features...")

# Per-(clone, tissue, timepoint) cell counts as a long-form lookup
# (clone_id → DataFrame slice). Sorting once for efficient .loc.
bin_obs_e_idx = bin_obs_e.set_index("clone_id").sort_index()
bin_obs_groups = bin_obs_e_idx.groupby(level=0)


def _clone_features(grp):
    """Compute weighted feature vector for one clone.

    Layout (length 3T + 3T + 3 + 3 + 3 + 3 = 6T + 12, T=6 → 48):
      [0       : 3T          ) occupancy (cell-count-weighted, * W_OCCUPANCY)
      [3T      : 6T          ) presence (binary, * W_PRESENCE)
      [6T      : 6T+3        ) first-tissue one-hot
      [6T+3    : 6T+6        ) last-tissue one-hot
      [6T+6    : 6T+9        ) scalars (n_unique, n_trans, entropy)
      [6T+9    : 6T+12       ) transition indicators * W_TRANSITION:
          csf_before_tp, pbmc_before_tp, csf_before_pbmc
    """
    occ = np.zeros((3, T), dtype=float)
    pres = np.zeros((3, T), dtype=int)
    total_per_tp = np.zeros(T, dtype=float)
    first_tp_by_tissue = [None, None, None]
    for _, r in grp.iterrows():
        i = TISSUE_IDX[r["tissue"]]; j = TP_IDX[r["timepoint"]]
        occ[i, j] += float(r["n_cells"])
        pres[i, j] = 1
        total_per_tp[j] += float(r["n_cells"])
        if first_tp_by_tissue[i] is None or j < first_tp_by_tissue[i]:
            first_tp_by_tissue[i] = j
    obs_tp = total_per_tp > 0
    occ[:, obs_tp] = occ[:, obs_tp] / total_per_tp[obs_tp]
    obs_js = np.where(obs_tp)[0]
    seq_tissues = [int(np.argmax(occ[:, j])) for j in obs_js]
    first_t = seq_tissues[0]; last_t = seq_tissues[-1]
    n_unique = len(set(seq_tissues))
    n_trans = int(sum(1 for a, b in zip(seq_tissues, seq_tissues[1:])
                       if a != b))
    tissue_totals = occ.sum(axis=1) + 1e-12
    ent = float(shannon_entropy(tissue_totals / tissue_totals.sum(),
                                 base=2))
    first_oh = np.zeros(3); first_oh[first_t] = 1
    last_oh = np.zeros(3); last_oh[last_t] = 1
    # Transition indicators: tissue X observed at strictly earlier
    # timepoint than tissue Y.
    pbmc_i, csf_i, tp_i = TISSUE_IDX["PBMC"], TISSUE_IDX["CSF"], TISSUE_IDX["TP"]
    csf_before_tp = float(
        first_tp_by_tissue[csf_i] is not None
        and first_tp_by_tissue[tp_i] is not None
        and first_tp_by_tissue[csf_i] < first_tp_by_tissue[tp_i])
    pbmc_before_tp = float(
        first_tp_by_tissue[pbmc_i] is not None
        and first_tp_by_tissue[tp_i] is not None
        and first_tp_by_tissue[pbmc_i] < first_tp_by_tissue[tp_i])
    csf_before_pbmc = float(
        first_tp_by_tissue[csf_i] is not None
        and first_tp_by_tissue[pbmc_i] is not None
        and first_tp_by_tissue[csf_i] < first_tp_by_tissue[pbmc_i])
    return np.concatenate([
        W_OCCUPANCY * occ.ravel(),
        W_PRESENCE * pres.ravel().astype(float),
        first_oh, last_oh,
        np.array([n_unique, n_trans, ent], dtype=float),
        W_TRANSITION * np.array(
            [csf_before_tp, pbmc_before_tp, csf_before_pbmc],
            dtype=float),
    ])


# Calibration on first 500 clones.
calib_n = min(500, len(eligible_clones))
calib_ids = eligible_clones[:calib_n]
_t0 = time.time()
calib_feats = []
for cid in calib_ids:
    g = bin_obs_groups.get_group(cid)
    calib_feats.append(_clone_features(g))
calib_dt = time.time() - _t0
eta_full = calib_dt * (len(eligible_clones) / calib_n)
print(f"  calibration: {calib_n} clones in {calib_dt:.1f}s "
      f"→ ETA full ({len(eligible_clones)} clones): {eta_full:.0f}s")

# Full extraction.
features = np.zeros((len(eligible_clones),
                      len(calib_feats[0])), dtype=float)
features[:calib_n] = np.vstack(calib_feats)
for k, cid in enumerate(
        tqdm(eligible_clones[calib_n:], desc="  features")):
    g = bin_obs_groups.get_group(cid)
    features[calib_n + k] = _clone_features(g)

# Z-score only the 3 scalar features (n_unique, n_trans, entropy).
# The transition indicators at the tail are binary by design and stay
# at scale W_TRANSITION.
sc_start = 3 * T + 3 * T + 3 + 3
sc_end = sc_start + 3
sc_mean = features[:, sc_start:sc_end].mean(axis=0)
sc_std = features[:, sc_start:sc_end].std(axis=0) + 1e-9
features[:, sc_start:sc_end] = (features[:, sc_start:sc_end]
                                  - sc_mean) / sc_std

print(f"  feature matrix: {features.shape}")


# %%
# =========================================================
# Clustering sweep (Leiden + Ward-on-L2-normalized)
# =========================================================
print("\nClustering sweep...")
sweep_rows = []
labels_by_setting = {}

# Build the AnnData wrapper once for Leiden neighbor reuse.
feat_adata = ad.AnnData(X=features.astype(np.float32))
sc.pp.neighbors(feat_adata, n_neighbors=15, metric="cosine",
                use_rep="X", knn=True)

for res in LEIDEN_RESOLUTIONS:
    sc.tl.leiden(feat_adata, resolution=res, key_added=f"leiden_r{res}")
    labels = feat_adata.obs[f"leiden_r{res}"].astype(int).values
    k_eff = int(labels.max()) + 1
    if k_eff < 2 or k_eff > 30:
        sweep_rows.append({"method": "leiden", "setting": f"r={res}",
                            "K": k_eff, "silhouette": np.nan,
                            "calinski": np.nan})
        continue
    idx = (np.random.default_rng(0)
           .choice(features.shape[0],
                   size=min(SIL_SUBSAMPLE, features.shape[0]),
                   replace=False))
    sil = float(silhouette_score(features[idx], labels[idx],
                                  metric="cosine"))
    ch = float(calinski_harabasz_score(features, labels))
    sweep_rows.append({"method": "leiden", "setting": f"r={res}",
                        "K": k_eff, "silhouette": sil, "calinski": ch})
    labels_by_setting[("leiden", f"r={res}")] = labels
    print(f"  leiden r={res}: K={k_eff}, sil={sil:.3f}, CH={ch:.0f}")

# Ward on L2-normalized features (Euclidean here ≈ cosine).
features_l2 = normalize(features, norm="l2", axis=1)
linkage = sch.linkage(features_l2, method="ward", metric="euclidean")
for K in HIER_KS:
    labels = sch.fcluster(linkage, t=K, criterion="maxclust") - 1
    k_eff = int(labels.max()) + 1
    idx = (np.random.default_rng(0)
           .choice(features.shape[0],
                   size=min(SIL_SUBSAMPLE, features.shape[0]),
                   replace=False))
    sil = float(silhouette_score(features[idx], labels[idx],
                                  metric="cosine"))
    ch = float(calinski_harabasz_score(features, labels))
    sweep_rows.append({"method": "ward", "setting": f"K={K}",
                        "K": k_eff, "silhouette": sil, "calinski": ch})
    labels_by_setting[("ward", f"K={K}")] = labels
    print(f"  ward K={K}: K={k_eff}, sil={sil:.3f}, CH={ch:.0f}")

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(OUT_DIR / "silhouette_sweep.csv", index=False)
best_row = (sweep_df.dropna(subset=["silhouette"])
            .sort_values("silhouette", ascending=False).iloc[0])
best_method = best_row["method"]; best_setting = best_row["setting"]
best_labels = labels_by_setting[(best_method, best_setting)]
best_K = int(best_labels.max()) + 1
print(f"\nBest: {best_method} {best_setting} → K={best_K}, "
      f"sil={best_row['silhouette']:.3f}, CH={best_row['calinski']:.0f}")


# %%
# =========================================================
# Empirical-Q null simulation
# =========================================================
print("\nEmpirical-Q null simulation...")
P_emp_df = pd.read_csv(paths.EMPIRICAL_Q_DIR / "P_empirical.csv",
                        index_col=0)
# Collapse the (tissue × phenotype) transition matrix to a (tissue ×
# tissue) marginal by summing destination columns and averaging across
# source rows within each tissue block.
P_tissue = np.zeros((3, 3), dtype=float)
for si, src in enumerate(TISSUES):
    src_rows = [r for r in P_emp_df.index if r.startswith(f"{src}__")]
    if not src_rows:
        continue
    for di, dst in enumerate(TISSUES):
        dst_cols = [c for c in P_emp_df.columns
                    if c.startswith(f"{dst}__")]
        P_tissue[si, di] = P_emp_df.loc[src_rows, dst_cols].values.sum()
# Row-normalize.
P_tissue = P_tissue / P_tissue.sum(axis=1, keepdims=True)
print("  tissue-marginal P (rows=src, cols=dst):")
print(pd.DataFrame(P_tissue, index=TISSUES, columns=TISSUES).round(3).to_string())

# Population tissue marginal (for initial state).
init_tissue = obs_e.groupby("tissue").size().reindex(TISSUES, fill_value=0).values
init_tissue = init_tissue / max(init_tissue.sum(), 1.0)

# Bootstrap observation patterns from real clones.
clone_obs_patterns = {}
for cid, g in bin_obs_groups:
    tps = sorted(g["timepoint"].unique(), key=lambda s: int(s))
    counts_by_tp = g.groupby("timepoint")["n_cells"].sum().to_dict()
    clone_obs_patterns[cid] = [(t, int(counts_by_tp[t])) for t in tps]
pattern_list = list(clone_obs_patterns.values())


def _simulate_clone_features(rng):
    pattern = pattern_list[rng.integers(0, len(pattern_list))]
    cur_tissue = int(rng.choice(3, p=init_tissue))
    rows = []
    for k_obs, (tp, n_cells) in enumerate(pattern):
        if k_obs > 0:
            cur_tissue = int(rng.choice(3, p=P_tissue[cur_tissue]))
        rows.append({"tissue": TISSUES[cur_tissue],
                     "timepoint": tp, "n_cells": n_cells})
    return _clone_features(pd.DataFrame(rows))


rng = np.random.default_rng(0)
# Calibration: N=200.
_t0 = time.time()
for _ in range(200):
    _simulate_clone_features(rng)
sim_dt = time.time() - _t0
sim_eta = sim_dt * (N_NULL / 200.0)
print(f"  sim calibration: 200 clones in {sim_dt:.1f}s "
      f"→ ETA N={N_NULL}: {sim_eta:.0f}s")

null_features = np.zeros((N_NULL, features.shape[1]), dtype=float)
for k in tqdm(range(N_NULL), desc="  sim"):
    null_features[k] = _simulate_clone_features(rng)
# Apply the same scalar-tail z-score using observed stats.
null_features[:, sc_start:sc_end] = (
    null_features[:, sc_start:sc_end] - sc_mean) / sc_std

# Re-cluster using the chosen method+setting.
if best_method == "leiden":
    res = float(best_setting.split("=")[1])
    null_ad = ad.AnnData(X=null_features.astype(np.float32))
    sc.pp.neighbors(null_ad, n_neighbors=15, metric="cosine",
                    use_rep="X", knn=True)
    sc.tl.leiden(null_ad, resolution=res, key_added="leiden")
    null_labels = null_ad.obs["leiden"].astype(int).values
else:
    K = int(best_setting.split("=")[1])
    null_l2 = normalize(null_features, norm="l2", axis=1)
    null_linkage = sch.linkage(null_l2, method="ward", metric="euclidean")
    null_labels = sch.fcluster(null_linkage, t=K, criterion="maxclust") - 1

null_K = int(null_labels.max()) + 1
if null_K >= 2:
    idx = (np.random.default_rng(0)
           .choice(null_features.shape[0],
                   size=min(SIL_SUBSAMPLE, null_features.shape[0]),
                   replace=False))
    null_sil = float(silhouette_score(null_features[idx],
                                       null_labels[idx], metric="cosine"))
    null_ch = float(calinski_harabasz_score(null_features, null_labels))
else:
    null_sil = float("nan"); null_ch = float("nan")
print(f"  null K={null_K}, sil={null_sil:.3f}, CH={null_ch:.0f}")

null_df = pd.DataFrame([{
    "method": best_method, "setting": best_setting,
    "obs_K": best_K, "obs_silhouette": float(best_row["silhouette"]),
    "obs_calinski": float(best_row["calinski"]),
    "null_K": null_K, "null_silhouette": null_sil,
    "null_calinski": null_ch,
    "n_null_clones": N_NULL,
}])
null_df.to_csv(OUT_DIR / "null_comparison.csv", index=False)


# %%
# =========================================================
# Per-archetype summary
# =========================================================
print("\nPer-archetype summaries...")
clone_meta["archetype_id"] = best_labels
clone_meta["n_tissues_visited"] = [
    int((features[i, :3*T].reshape(3, T).sum(axis=1) > 0).sum())
    for i in range(len(clone_meta))
]
clone_meta[["trb", "patient", "clone_id", "archetype_id",
            "clone_size", "lineage", "n_observations",
            "n_tissues_visited"]].to_csv(
    OUT_DIR / "clone_archetypes.csv", index=False)

# Prototype occupancy (3 × T) per archetype.
prototypes = np.zeros((best_K, 3, T), dtype=float)
for a in range(best_K):
    idx = np.where(best_labels == a)[0]
    prototypes[a] = features[idx, :3*T].reshape(-1, 3, T).mean(axis=0)
np.savez(OUT_DIR / "archetype_prototypes.npz",
          prototypes=prototypes, tissues=np.array(TISSUES),
          timepoints=np.array(TPS))

# Dominant phenotype per (archetype × tissue × timepoint).
clone_to_arch = dict(zip(clone_meta["clone_id"], best_labels))
obs_e["archetype_id"] = obs_e["clone_id"].map(clone_to_arch)
phenotypes_present = [p for p in TCELL_PHENOTYPE_ORDER
                      if p in obs_e["phenotype"].unique()]
dom_phen = np.full((best_K, 3, T), "", dtype=object)
phen_frac = np.zeros((best_K, 3, T, len(phenotypes_present)),
                      dtype=float)
for a in range(best_K):
    sub_a = obs_e[obs_e["archetype_id"] == a]
    for ti, tis in enumerate(TISSUES):
        for tj, tp in enumerate(TPS):
            sub = sub_a[(sub_a["tissue"] == tis)
                          & (sub_a["timepoint"] == tp)]
            if sub.empty:
                continue
            cnt = sub["phenotype"].value_counts()
            dom_phen[a, ti, tj] = cnt.idxmax()
            for pi, p in enumerate(phenotypes_present):
                phen_frac[a, ti, tj, pi] = (
                    cnt.get(p, 0) / max(cnt.sum(), 1))

# Archetype summary rows.
summary_rows = []
for a in range(best_K):
    sub_c = clone_meta[clone_meta["archetype_id"] == a]
    sub_o = obs_e[obs_e["archetype_id"] == a]
    proto = prototypes[a]
    first_t_idx = int(np.argmax(proto.sum(axis=1)))
    # First and last observed timepoint dominant tissue.
    first_obs_t = next((j for j in range(T)
                        if proto[:, j].sum() > 0), 0)
    last_obs_t = next((j for j in range(T - 1, -1, -1)
                       if proto[:, j].sum() > 0), 0)
    first_tissue = TISSUES[int(np.argmax(proto[:, first_obs_t]))]
    last_tissue = TISSUES[int(np.argmax(proto[:, last_obs_t]))]
    top_phen = (sub_o["phenotype"].value_counts().head(1)
                .index[0] if not sub_o.empty else "")
    summary_rows.append({
        "archetype_id": a,
        "n_clones": int(len(sub_c)),
        "clone_size_mean": float(sub_c["clone_size"].mean()),
        "clone_size_std": float(sub_c["clone_size"].std()),
        "fraction_CD8": float((sub_c["lineage"] == "CD8").mean()),
        "fraction_CD4": float((sub_c["lineage"] == "CD4").mean()),
        "top_phenotype": top_phen,
        "first_tissue": first_tissue,
        "last_tissue": last_tissue,
        "n_unique_patients": int(sub_c["patient"].nunique()),
        "patients_top": ",".join(
            sub_c["patient"].value_counts().head(3).index.tolist()),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "archetype_summary.csv", index=False)
print(summary_df.to_string(index=False))


# %%
# =========================================================
# Plot 1: silhouette sweep
# =========================================================
print("\nPlotting silhouette sweep...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
leiden = sweep_df[sweep_df["method"] == "leiden"].copy()
leiden["res"] = leiden["setting"].str.replace("r=", "").astype(float)
axes[0].plot(leiden["res"], leiden["silhouette"], "o-",
              color="#2a5db0", lw=1.5, markersize=6,
              label="silhouette")
axes[0].set_xlabel("Leiden resolution", fontsize=9)
axes[0].set_ylabel("Silhouette (cosine)", fontsize=9)
axes[0].set_title("Leiden resolution sweep", fontsize=10,
                   fontweight="bold")
for s in ("top", "right"):
    axes[0].spines[s].set_visible(False)

ward = sweep_df[sweep_df["method"] == "ward"].copy()
ward["K"] = ward["setting"].str.replace("K=", "").astype(int)
axes[1].plot(ward["K"], ward["silhouette"], "o-",
              color="#a87521", lw=1.5, markersize=6,
              label="silhouette")
axes[1].set_xlabel("Hierarchical K", fontsize=9)
axes[1].set_ylabel("Silhouette (cosine)", fontsize=9)
axes[1].set_title("Ward (L2-norm) sweep", fontsize=10,
                   fontweight="bold")
for s in ("top", "right"):
    axes[1].spines[s].set_visible(False)
fig.suptitle(f"Best: {best_method} {best_setting} "
             f"(K={best_K}, sil={best_row['silhouette']:.3f})",
             fontsize=11, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(OUT_DIR / "silhouette_sweep.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 2: archetype prototypes (stacked-area per archetype)
# =========================================================
print("Plotting archetype prototypes...")
n_cols = min(4, best_K)
n_rows = int(np.ceil(best_K / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols,
                                                    2.6 * n_rows),
                         squeeze=False)
xs = np.arange(T)
for a in range(best_K):
    ax = axes[a // n_cols, a % n_cols]
    bottoms = np.zeros(T)
    for ti, tis in enumerate(TISSUES):
        ax.fill_between(xs, bottoms, bottoms + prototypes[a, ti],
                         color=TISSUE_COLORS.get(tis, "#888"),
                         alpha=0.85, edgecolor="white", linewidth=0.5,
                         label=tis if a == 0 else None)
        bottoms += prototypes[a, ti]
    sub_c = clone_meta[clone_meta["archetype_id"] == a]
    ax.set_title(f"Arch {a}  n={len(sub_c)}  "
                  f"sizē={sub_c['clone_size'].mean():.0f}",
                  fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"T{t}" for t in TPS], fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Timepoint", fontsize=8)
    ax.set_ylabel("Mean tissue fraction", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for k in range(best_K, n_rows * n_cols):
    axes[k // n_cols, k % n_cols].axis("off")
handles = [plt.Rectangle((0, 0), 1, 1,
                          facecolor=TISSUE_COLORS[t], alpha=0.85)
           for t in TISSUES]
fig.legend(handles, TISSUES, loc="lower center", ncol=3, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"Archetype prototype trajectories "
             f"({best_method} {best_setting})",
             fontsize=11, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_prototypes.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 3: UMAP of feature matrix, colored by archetype
# =========================================================
print("Plotting archetype UMAP...")
try:
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.30,
                         metric="cosine", random_state=0)
    emb = reducer.fit_transform(features)
except Exception as e:
    print(f"  umap-learn failed ({e}); falling back to PCA")
    from sklearn.decomposition import PCA
    emb = PCA(n_components=2, random_state=0).fit_transform(features)
fig, ax = plt.subplots(figsize=(7, 6))
cmap = plt.get_cmap("tab10" if best_K <= 10 else "tab20")
for a in range(best_K):
    m = best_labels == a
    ax.scatter(emb[m, 0], emb[m, 1], s=4, color=cmap(a),
               alpha=0.7, edgecolors="none", rasterized=True,
               label=f"A{a} (n={int(m.sum())})")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
ax.set_title(f"Clones in feature space — {best_method} {best_setting}",
              fontsize=11, fontweight="bold")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
          fontsize=8, frameon=False)
for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_umap.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 4: null vs observed silhouette + CH
# =========================================================
print("Plotting null comparison...")
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].bar([0, 1],
             [float(best_row["silhouette"]), null_sil],
             color=["#2a5db0", "#bbbbbb"], edgecolor="black",
             linewidth=0.4)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["observed", "null"], fontsize=9)
axes[0].set_ylabel("Silhouette (cosine)", fontsize=9)
axes[0].set_title("Cluster compactness", fontsize=10,
                   fontweight="bold")
axes[1].bar([0, 1],
             [float(best_row["calinski"]), null_ch],
             color=["#2a5db0", "#bbbbbb"], edgecolor="black",
             linewidth=0.4)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["observed", "null"], fontsize=9)
axes[1].set_ylabel("Calinski-Harabasz", fontsize=9)
axes[1].set_title("Cluster separation", fontsize=10,
                   fontweight="bold")
for ax in axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle(f"Observed vs null (N={N_NULL}, "
             f"{best_method} {best_setting})",
             fontsize=11, fontweight="bold", y=0.99)
fig.tight_layout()
fig.savefig(OUT_DIR / "null_comparison.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Plot 5: archetype phenotype composition heatmaps
# =========================================================
print("Plotting archetype phenotypes...")
fig, axes = plt.subplots(best_K, 3, figsize=(11, 1.8 * best_K),
                         squeeze=False, sharex=False, sharey=True)
for a in range(best_K):
    for ti, tis in enumerate(TISSUES):
        ax = axes[a, ti]
        mat = phen_frac[a, ti].T  # (n_phen, T)
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0, vmax=1)
        ax.set_xticks(range(T))
        ax.set_xticklabels([f"T{t}" for t in TPS], fontsize=6)
        if ti == 0:
            ax.set_yticks(range(len(phenotypes_present)))
            ax.set_yticklabels(phenotypes_present, fontsize=6)
            for tick, p in zip(ax.get_yticklabels(), phenotypes_present):
                tick.set_color(TCELL_PHENOTYPE_COLORS.get(p, "#444"))
        else:
            ax.set_yticks([])
        if a == 0:
            ax.set_title(tis, fontsize=9, fontweight="bold",
                          color=TISSUE_COLORS.get(tis, "#444"))
        if ti == 0:
            ax.set_ylabel(f"A{a}\n(n={int((best_labels == a).sum())})",
                          fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=20)
        ax.tick_params(length=0)
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
fig.suptitle("Phenotype composition per (archetype × tissue × timepoint)",
             fontsize=11, fontweight="bold", y=0.995)
fig.tight_layout()
fig.savefig(OUT_DIR / "archetype_phenotypes.png",
            dpi=DPI, bbox_inches="tight")
plt.close(fig)


# %%
# =========================================================
# Final summary
# =========================================================
print("\n========== SUMMARY ==========")
print(f"Best clustering: {best_method} {best_setting}  K={best_K}")
print(f"Observed silhouette: {best_row['silhouette']:.3f}  "
      f"CH: {best_row['calinski']:.0f}")
print(f"Null silhouette:     {null_sil:.3f}  CH: {null_ch:.0f}")
print(f"Eligible clones: {len(eligible_clones):,}")
print()
for _, r in summary_df.iterrows():
    a = int(r["archetype_id"])
    proto = prototypes[a]
    first_obs_t = next((j for j in range(T) if proto[:, j].sum() > 0), 0)
    last_obs_t = next((j for j in range(T - 1, -1, -1)
                        if proto[:, j].sum() > 0), 0)
    cd_label = "CD8" if r["fraction_CD8"] >= 0.5 else "CD4"
    desc = (f"A{a}: n={int(r['n_clones']):4d} "
            f"sizē={r['clone_size_mean']:.0f}  "
            f"{r['first_tissue']}-early → {r['last_tissue']}-late  "
            f"dom={r['top_phenotype']} ({cd_label})")
    print(desc)

print(f"\nDone. All outputs in {OUT_DIR}")
