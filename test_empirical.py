"""Synthetic test for trafficking.empirical.

Generates clone observations from a known ground-truth Q matrix, runs the
empirical-P estimation and both decompositions, and checks that recovered
migration and phenotype-transition rates correlate with ground truth at
Pearson r > 0.8.

Run: python test_empirical.py
"""
import sys
import numpy as np
from scipy.linalg import expm
from scipy.stats import pearsonr

from trafficking.ctmc import StateSpace, build_generator, decompose_generator
from trafficking.empirical import (
    build_empirical_P,
    decompose_P_logm,
    decompose_P_optimizer,
    rate_comparison_table,
    summary_report,
)


RNG = np.random.default_rng(7)


def make_ground_truth(ss, dt=1.0, scale_mig=0.18, scale_trans=0.12):
    """Build a synthetic, valid row-generator Q with positive off-diagonals."""
    S, K = ss.S, ss.K

    # Migration: phenotype-dependent, but with some heterogeneity across edges
    migration = np.zeros((S, S, K))
    for s_from in range(S):
        for s_to in range(S):
            if s_from == s_to:
                continue
            for k in range(K):
                base = scale_mig * (0.5 + RNG.random())
                migration[s_from, s_to, k] = base

    # Phenotype transitions: per-site, asymmetric
    transition = np.zeros((S, K, K))
    for s in range(S):
        for k_from in range(K):
            for k_to in range(K):
                if k_from == k_to:
                    continue
                base = scale_trans * (0.3 + RNG.random())
                transition[s, k_from, k_to] = base

    # Expansion: zero (keeps P exactly row-stochastic — simplest to test)
    expansion = np.zeros((S, K))

    Q = build_generator(migration, transition, expansion, ss)
    P = expm(Q * dt)
    return Q, P, migration, transition, expansion


def make_observations(P_true, ss, n_clones=100, n_transitions=2,
                      n_cells_mean=80, sample=True, sparse=True):
    """Sample clone observations from P_true.

    For each clone × transition pair, draw a (sparse) source distribution
    over the joint state space, propagate through P_true, and sample
    n_cells from a multinomial to add realistic counting noise.

    Real clones live in 1–2 states most of the time, so ``sparse=True``
    samples each clone's "home" as a small subset (1–3 states) and draws
    a Dirichlet only over that subset.
    """
    observations = []
    for c in range(n_clones):
        for t in range(n_transitions):
            n_src = max(3, int(RNG.poisson(n_cells_mean)))
            n_dst = max(3, int(RNG.poisson(n_cells_mean)))

            if sparse:
                k_home = int(RNG.integers(1, 4))  # 1, 2 or 3 home states
                home = RNG.choice(ss.N, size=k_home, replace=False)
                src_vec_true = np.zeros(ss.N)
                src_vec_true[home] = RNG.dirichlet(np.ones(k_home))
            else:
                alpha = RNG.uniform(0.3, 1.5, size=ss.N) * 0.6
                src_vec_true = RNG.dirichlet(alpha)

            if sample:
                src_counts = RNG.multinomial(n_src, src_vec_true)
                src_vec = src_counts / max(src_counts.sum(), 1)
            else:
                src_vec = src_vec_true

            # P[i, j] is prob(state i -> state j); rows sum to 1.
            # So dst_vec (row) = src_vec (row) @ P
            dst_mean = src_vec_true @ P_true
            if sample:
                dst_counts = RNG.multinomial(n_dst, dst_mean)
                dst_vec = dst_counts / max(dst_counts.sum(), 1)
            else:
                dst_vec = dst_mean

            observations.append({
                "trb": f"clone_{c:03d}",
                "patient": f"P{c % 5}",
                "t_src": str(t + 1),
                "t_dst": str(t + 2),
                "src_vec": src_vec,
                "dst_vec": dst_vec,
                "n_cells_src": int(n_src),
                "n_cells_dst": int(n_dst),
            })
    return observations


def flatten_offdiag_migration(mig):
    """Flatten (S, S, K) migration array, dropping s_from == s_to."""
    S, _, K = mig.shape
    out = []
    for s_from in range(S):
        for s_to in range(S):
            if s_from == s_to:
                continue
            for k in range(K):
                out.append(mig[s_from, s_to, k])
    return np.array(out)


def flatten_offdiag_transition(trans):
    """Flatten (S, K, K) transition array, dropping k_from == k_to."""
    S, K, _ = trans.shape
    out = []
    for s in range(S):
        for k_from in range(K):
            for k_to in range(K):
                if k_from == k_to:
                    continue
                out.append(trans[s, k_from, k_to])
    return np.array(out)


def check_pearson(a, b, label, threshold=0.8):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        print(f"  [SKIP] {label}: zero variance.")
        return True
    r, p = pearsonr(a, b)
    ok = r > threshold
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: r = {r:.3f} "
          f"(p = {p:.2e}, threshold > {threshold})")
    return ok


def main():
    print("=" * 70)
    print("Synthetic test: trafficking.empirical")
    print("=" * 70)

    sites = ["PBMC", "CSF", "TP"]
    phenotypes = ["CD8_Activated_TEXeff", "CD8_Quiescent_Memory",
                  "CD4_Treg", "CD4_Th"]
    ss = StateSpace(sites, phenotypes)
    print(f"  StateSpace: {ss}")

    dt = 1.0
    Q_true, P_true, mig_true, trans_true, exp_true = make_ground_truth(ss, dt=dt)
    print(f"  Ground-truth Q: shape {Q_true.shape}, "
          f"max off-diag = {Q_true[~np.eye(ss.N, dtype=bool)].max():.4f}, "
          f"row-sum max |.| = {np.abs(Q_true.sum(axis=1)).max():.2e}")
    print(f"  Ground-truth P: row-sum max |.−1| = "
          f"{np.abs(P_true.sum(axis=1) - 1).max():.2e}, "
          f"min entry = {P_true.min():.4f}")

    print("\n--- Build observations ---")
    observations = make_observations(
        P_true, ss, n_clones=400, n_transitions=2,
        n_cells_mean=80, sample=True, sparse=True)
    print(f"  Generated {len(observations)} observations from "
          f"{len({o['trb'] for o in observations})} clones, "
          f"{len({o['patient'] for o in observations})} patients.")

    print("\n--- Build empirical P (weighted_lstsq) ---")
    P_hat, diag = build_empirical_P(observations, ss,
                                    method="weighted_lstsq",
                                    pseudocount=0.005)
    print(f"  ||P_hat - P_true||_F = "
          f"{np.linalg.norm(P_hat - P_true):.4f}")

    print("\n--- Decompose via matrix logarithm ---")
    res_logm = decompose_P_logm(P_hat, ss, dt=dt)

    print("\n--- Decompose via structured optimizer ---")
    res_opt = decompose_P_optimizer(
        P_hat, ss, dt=dt,
        shared_migration=False, shared_transition=False,
        fit_expansion=False, l2_reg=0.0,
        maxiter=3000, verbose=True, print_every=100,
    )

    print("\n--- Rate comparison (logm vs optimizer) ---")
    _ = rate_comparison_table(res_logm, res_opt)

    print("\n--- Recovery checks (vs ground truth) ---")
    mig_true_flat = flatten_offdiag_migration(mig_true)
    trans_true_flat = flatten_offdiag_transition(trans_true)
    mig_logm_flat = flatten_offdiag_migration(res_logm["migration"])
    trans_logm_flat = flatten_offdiag_transition(res_logm["transition"])
    mig_opt_flat = flatten_offdiag_migration(res_opt["migration"])
    trans_opt_flat = flatten_offdiag_transition(res_opt["transition"])

    ok_all = True
    print("\n  Pearson correlation with ground truth:")
    ok_all &= check_pearson(mig_true_flat, mig_logm_flat,
                            "migration  (logm vs truth)")
    ok_all &= check_pearson(trans_true_flat, trans_logm_flat,
                            "transition (logm vs truth)")
    ok_all &= check_pearson(mig_true_flat, mig_opt_flat,
                            "migration  (optimizer vs truth)")
    ok_all &= check_pearson(trans_true_flat, trans_opt_flat,
                            "transition (optimizer vs truth)")

    print("\n--- summary_report (smoke test) ---")
    summary_report(P_hat, res_logm, ss, observations)

    print("\n" + "=" * 70)
    print(f"OVERALL: {'PASS' if ok_all else 'FAIL'}")
    print("=" * 70)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
