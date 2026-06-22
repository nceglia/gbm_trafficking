# %%
"""Empirical 3K × 3K transition matrix P from clone-level tracking data,
plus structured Q decomposition.

For each clone × timepoint with >= MIN_CELLS cells across tissues, build a
fractional state vector over the joint (tissue × phenotype) space. Pair
consecutive timepoints, solve a weighted least-squares for P, then
decompose P into migration / transition / expansion via:
  - matrix logarithm (logm)
  - structured-CTMC Frobenius fit (optimizer)

Outputs go to results/traffic_migration_rates/.
"""
import io
import json
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))           # for `trafficking`
sys.path.insert(0, str(REPO_ROOT / "pipeline"))  # for `modules.style`

from modules import paths  # noqa: E402
from modules.style import TCELL_PHENOTYPE_ORDER  # noqa: E402

from trafficking.ctmc import StateSpace  # noqa: E402
from trafficking.empirical import (  # noqa: E402
    build_clone_state_vectors,
    build_empirical_P,
    decompose_P_logm,
    decompose_P_optimizer,
    rate_comparison_table,
    summary_report,
    build_clone_state_vectors_augmented,
    build_empirical_P_augmented,
    decompose_P_augmented,
    UNOBS_LABEL,
)
from scipy.stats import pearsonr, spearmanr  # noqa: E402

# %%
# ---- Config ----
DATA_PATH = paths.H5AD_TCELLS
OUT_DIR = paths.EMPIRICAL_Q_DIR
paths.ensure(OUT_DIR)

TISSUES = ("PBMC", "CSF", "TP")
PHENOTYPES = tuple(TCELL_PHENOTYPE_ORDER)
TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
MIN_CELLS = 2
DT = 1.0
PSEUDOCOUNT = 0.01

# Optimizer
OPT_SHARED_MIGRATION = False
OPT_SHARED_TRANSITION = False
OPT_FIT_EXPANSION = False     # empirical P is row-stochastic
OPT_L2_REG = 1e-3
OPT_MAXITER = 5000
OPT_PRINT_EVERY = 200

# %%
# ---- Load adata ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

ss = StateSpace(TISSUES, PHENOTYPES)
print(f"  {ss}")

# %%
# ---- Clone-level state vectors and observation pairs ----
print("\nBuilding clone state vectors...")
observations = build_clone_state_vectors(
    adata, ss, TRANSITIONS,
    clone_key="trb", phenotype_key="phenotype",
    tissue_key="tissue", time_key="timepoint",
    patient_key="patient", min_cells=MIN_CELLS,
)

# Save observation index (drop the per-row state vectors — too wide).
obs_df = pd.DataFrame([
    {"trb": o["trb"], "patient": o["patient"],
     "t_src": o["t_src"], "t_dst": o["t_dst"],
     "n_cells_src": o["n_cells_src"], "n_cells_dst": o["n_cells_dst"]}
    for o in observations
])
obs_df.to_csv(OUT_DIR / "observations.csv", index=False)
print(f"  wrote {OUT_DIR / 'observations.csv'} ({len(obs_df)} rows)")

# Save the stacked SRC / DST state matrices as compressed npz so the
# expensive vectorization is replayable without re-loading adata.
SRC = np.stack([o["src_vec"] for o in observations])
DST = np.stack([o["dst_vec"] for o in observations])
np.savez_compressed(
    OUT_DIR / "state_vectors.npz",
    SRC=SRC, DST=DST,
    n_cells_src=obs_df["n_cells_src"].to_numpy(),
    n_cells_dst=obs_df["n_cells_dst"].to_numpy(),
    sites=np.array(ss.sites), phenotypes=np.array(ss.phenotypes),
)
print(f"  wrote {OUT_DIR / 'state_vectors.npz'}")

# %%
# ---- Empirical P ----
print("\nEstimating empirical P (weighted_lstsq)...")
P, P_diag = build_empirical_P(
    observations, ss, method="weighted_lstsq", pseudocount=PSEUDOCOUNT,
)

state_labels = [f"{site}__{ph}" for site, ph in ss.labels]
P_df = pd.DataFrame(P, index=state_labels, columns=state_labels)
P_df.index.name = "from"
P_df.columns.name = "to"
P_df.to_csv(OUT_DIR / "P_empirical.csv")
np.save(OUT_DIR / "P_empirical.npy", P)
print(f"  wrote {OUT_DIR / 'P_empirical.csv'} and .npy")

# %%
# ---- Decompose via matrix logarithm ----
print("\nDecomposing P via matrix logarithm...")
res_logm = decompose_P_logm(P, ss, dt=DT)

# ---- Decompose via structured optimizer ----
print("\nDecomposing P via structured CTMC optimizer (Frobenius)...")
res_opt = decompose_P_optimizer(
    P, ss, dt=DT,
    shared_migration=OPT_SHARED_MIGRATION,
    shared_transition=OPT_SHARED_TRANSITION,
    fit_expansion=OPT_FIT_EXPANSION,
    l2_reg=OPT_L2_REG, maxiter=OPT_MAXITER,
    verbose=True, print_every=OPT_PRINT_EVERY,
)


# %%
# ---- Save Q matrices and decomposed rate tables ----
def _save_Q(Q, tag):
    df = pd.DataFrame(Q, index=state_labels, columns=state_labels)
    df.index.name = "from"
    df.columns.name = "to"
    df.to_csv(OUT_DIR / f"Q_{tag}.csv")
    np.save(OUT_DIR / f"Q_{tag}.npy", Q)


def _migration_rows(result, tag):
    rows = []
    mig = result["migration"]
    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            for k in range(ss.K):
                rows.append({
                    "src": ss.sites[s_from], "dst": ss.sites[s_to],
                    "phenotype": ss.phenotypes[k],
                    "rate": float(mig[s_from, s_to, k]),
                    "method": tag,
                })
    return rows


def _transition_rows(result, tag):
    rows = []
    trans = result["transition"]
    for s in range(ss.S):
        for k_from in range(ss.K):
            for k_to in range(ss.K):
                if k_from == k_to:
                    continue
                rows.append({
                    "site": ss.sites[s],
                    "src": ss.phenotypes[k_from],
                    "dst": ss.phenotypes[k_to],
                    "rate": float(trans[s, k_from, k_to]),
                    "method": tag,
                })
    return rows


def _expansion_rows(result, tag):
    rows = []
    exp = result["expansion"]
    for s in range(ss.S):
        for k in range(ss.K):
            rows.append({
                "site": ss.sites[s],
                "phenotype": ss.phenotypes[k],
                "rate": float(exp[s, k]),
                "method": tag,
            })
    return rows


_save_Q(res_logm["Q"], "logm")
_save_Q(res_opt["Q"], "optimizer")
_save_Q(res_opt["P_fitted"], "optimizer_P_fitted")  # for completeness

mig_df = pd.DataFrame(
    _migration_rows(res_logm, "logm") + _migration_rows(res_opt, "optimizer")
)
trans_df = pd.DataFrame(
    _transition_rows(res_logm, "logm") + _transition_rows(res_opt, "optimizer")
)
exp_df = pd.DataFrame(
    _expansion_rows(res_logm, "logm") + _expansion_rows(res_opt, "optimizer")
)
mig_df.to_csv(OUT_DIR / "migration_rates.csv", index=False)
trans_df.to_csv(OUT_DIR / "transition_rates.csv", index=False)
exp_df.to_csv(OUT_DIR / "expansion_rates.csv", index=False)
print(f"  wrote migration_rates.csv, transition_rates.csv, "
      f"expansion_rates.csv")

# %%
# ---- Rate comparison (logm vs optimizer) ----
print("\nBuilding rate comparison table...")
cmp_df = rate_comparison_table(res_logm, res_opt)
cmp_df.to_csv(OUT_DIR / "rate_comparison.csv", index=False)
print(f"  wrote {OUT_DIR / 'rate_comparison.csv'}")

# %%
# ---- Diagnostics + summary report ----
diagnostics = {
    "n_observations": int(P_diag["n_obs"]),
    "n_states": int(ss.N),
    "n_sites": int(ss.S),
    "n_phenotypes": int(ss.K),
    "sites": list(ss.sites),
    "phenotypes": list(ss.phenotypes),
    "transitions": [list(t) for t in TRANSITIONS],
    "min_cells": MIN_CELLS,
    "pseudocount": PSEUDOCOUNT,
    "dt": DT,
    "empirical_P": {
        "condition_number": float(P_diag["condition_number"]),
        "row_coverage_min": int(P_diag["row_coverage"].min()),
        "row_coverage_max": int(P_diag["row_coverage"].max()),
        "row_coverage_mean": float(P_diag["row_coverage"].mean()),
        "n_states_with_lt_5_obs": int((P_diag["row_coverage"] < 5).sum()),
    },
    "logm": {
        "valid_generator": bool(res_logm["valid"]),
        "n_violations": int(res_logm["n_violations"]),
    },
    "optimizer": {
        "loss": float(res_opt["loss"]),
        "frobenius_error": float(res_opt["frobenius_error"]),
        "converged": bool(res_opt["scipy_result"].success),
        "config": res_opt["config"],
    },
}
with open(OUT_DIR / "diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)
print(f"  wrote {OUT_DIR / 'diagnostics.json'}")

# Save row-coverage as a CSV for easy inspection.
cov_df = pd.DataFrame({
    "state": state_labels,
    "n_obs_with_src_gt_0.05": P_diag["row_coverage"],
})
cov_df.to_csv(OUT_DIR / "row_coverage.csv", index=False)

# Capture the summary_report (which prints) to a text file.
buf = io.StringIO()
with redirect_stdout(buf):
    summary_report(P, res_logm, ss, observations)
report_text = buf.getvalue()
(OUT_DIR / "summary_report.txt").write_text(report_text)
print(report_text)
print(f"  wrote {OUT_DIR / 'summary_report.txt'}")


# %%
# ==================================================================
# Augmented model with born/die state
# ==================================================================
print("\n" + "=" * 60)
print("Augmented analysis with born/die unobserved state")
print("=" * 60)

print("\nBuilding augmented clone state vectors (3K+1 dim, "
      "unobserved at index ss.N)...")
aug_observations = build_clone_state_vectors_augmented(
    adata, ss, TRANSITIONS,
    clone_key="trb", phenotype_key="phenotype",
    tissue_key="tissue", time_key="timepoint",
    patient_key="patient", min_cells=MIN_CELLS,
)

# Save augmented observation index (drop state vectors).
aug_obs_df = pd.DataFrame([
    {"trb": o["trb"], "patient": o["patient"],
     "t_src": o["t_src"], "t_dst": o["t_dst"],
     "n_cells_src": o["n_cells_src"], "n_cells_dst": o["n_cells_dst"],
     "kind": o["kind"]}
    for o in aug_observations
])
aug_obs_df.to_csv(OUT_DIR / "observations_augmented.csv", index=False)
print(f"  wrote {OUT_DIR / 'observations_augmented.csv'} "
      f"({len(aug_obs_df)} rows)")

# %%
print("\nEstimating augmented P (3K+1 x 3K+1)...")
P_aug, P_aug_diag = build_empirical_P_augmented(
    aug_observations, ss,
    method="weighted_lstsq", pseudocount=PSEUDOCOUNT,
)
aug_labels = state_labels + [UNOBS_LABEL]
P_aug_df = pd.DataFrame(P_aug, index=aug_labels, columns=aug_labels)
P_aug_df.index.name = "from"
P_aug_df.columns.name = "to"
P_aug_df.to_csv(OUT_DIR / "P_augmented.csv")
np.save(OUT_DIR / "P_augmented.npy", P_aug)
print(f"  wrote {OUT_DIR / 'P_augmented.csv'} and .npy")

# %%
print("\nDecomposing augmented P (interior block, fit_expansion=True)...")
res_aug = decompose_P_augmented(
    P_aug, ss, dt=DT,
    shared_migration=OPT_SHARED_MIGRATION,
    shared_transition=OPT_SHARED_TRANSITION,
    l2_reg=OPT_L2_REG, maxiter=OPT_MAXITER,
    verbose=True, print_every=OPT_PRINT_EVERY,
)

# Persist augmented matrices.
_save_Q(res_aug["Q"], "augmented_optimizer")
_save_Q(res_aug["P_fitted"], "augmented_optimizer_P_fitted_interior")
P_interior_df = pd.DataFrame(res_aug["P_empirical"],
                              index=state_labels, columns=state_labels)
P_interior_df.index.name = "from"
P_interior_df.columns.name = "to"
P_interior_df.to_csv(OUT_DIR / "P_interior_augmented.csv")

# Exit / entry rates.
res_aug["exit_rates"].to_csv(OUT_DIR / "exit_rates.csv", index=False)
res_aug["entry_rates"].to_csv(OUT_DIR / "entry_rates.csv", index=False)
print(f"  wrote exit_rates.csv, entry_rates.csv")

# Decomposed rate tables with _augmented suffix.
mig_aug = pd.DataFrame(_migration_rows(res_aug, "optimizer_augmented"))
trans_aug = pd.DataFrame(_transition_rows(res_aug, "optimizer_augmented"))
exp_aug = pd.DataFrame(_expansion_rows(res_aug, "optimizer_augmented"))
mig_aug.to_csv(OUT_DIR / "migration_rates_augmented.csv", index=False)
trans_aug.to_csv(OUT_DIR / "transition_rates_augmented.csv", index=False)
exp_aug.to_csv(OUT_DIR / "expansion_rates_augmented.csv", index=False)
print("  wrote migration_rates_augmented.csv, transition_rates_augmented.csv, "
      "expansion_rates_augmented.csv")


# %%
# ---- Migration rate consistency: original (closed) vs augmented (open) ----
def _flatten_mig(result):
    rows = []
    mig = result["migration"]
    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            for k in range(ss.K):
                rows.append({
                    "src": ss.sites[s_from], "dst": ss.sites[s_to],
                    "phenotype": ss.phenotypes[k],
                    "rate": float(mig[s_from, s_to, k]),
                })
    return pd.DataFrame(rows)


orig_flat = _flatten_mig(res_opt).rename(columns={"rate": "rate_original"})
aug_flat = _flatten_mig(res_aug).rename(columns={"rate": "rate_augmented"})
mig_compare = orig_flat.merge(
    aug_flat, on=["src", "dst", "phenotype"], how="inner")
mig_compare["abs_diff"] = (
    mig_compare["rate_original"] - mig_compare["rate_augmented"]).abs()
mig_compare.to_csv(OUT_DIR / "migration_rates_compare_original_augmented.csv",
                    index=False)

orig_vec = mig_compare["rate_original"].to_numpy()
aug_vec = mig_compare["rate_augmented"].to_numpy()
r_p, p_p = pearsonr(orig_vec, aug_vec)
r_s, p_s = spearmanr(orig_vec, aug_vec)
print("\n=== Migration rate consistency: original vs augmented ===")
print(f"  Pearson r  = {r_p:.4f} (p = {p_p:.3g})")
print(f"  Spearman rho = {r_s:.4f} (p = {p_s:.3g})")
print(f"  Mean |diff|  = {mig_compare['abs_diff'].mean():.4f}")
print(f"  Max  |diff|  = {mig_compare['abs_diff'].max():.4f}")
print("\nTop 5 rates that shifted most after opening the system:")
print(mig_compare.nlargest(5, "abs_diff").to_string(index=False))

# Diagnostics for augmented run.
augmented_diag = {
    "n_observations": int(len(aug_observations)),
    "n_normal": int((aug_obs_df["kind"] == "normal").sum()),
    "n_exit": int((aug_obs_df["kind"] == "exit").sum()),
    "n_entry": int((aug_obs_df["kind"] == "entry").sum()),
    "P_augmented": {
        "condition_number": float(P_aug_diag["condition_number"]),
        "row_coverage_min": int(P_aug_diag["row_coverage"].min()),
        "row_coverage_max": int(P_aug_diag["row_coverage"].max()),
        "row_coverage_mean": float(P_aug_diag["row_coverage"].mean()),
    },
    "optimizer_augmented": {
        "loss": float(res_aug["loss"]),
        "frobenius_error": float(res_aug["frobenius_error"]),
        "converged": bool(res_aug["scipy_result"].success),
        "config": res_aug["config"],
    },
    "migration_consistency": {
        "pearson_r": float(r_p), "pearson_p": float(p_p),
        "spearman_rho": float(r_s), "spearman_p": float(p_s),
        "mean_abs_diff": float(mig_compare["abs_diff"].mean()),
        "max_abs_diff": float(mig_compare["abs_diff"].max()),
    },
}
with open(OUT_DIR / "diagnostics_augmented.json", "w") as f:
    json.dump(augmented_diag, f, indent=2)
print(f"  wrote {OUT_DIR / 'diagnostics_augmented.json'}")

print("\nDone.")
print(f"All outputs in: {OUT_DIR}")
