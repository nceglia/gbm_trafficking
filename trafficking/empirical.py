"""Empirical Q matrix estimation from clone-level tracking data.

The structured CTMC fit in ``trafficking.ctmc`` is fit against marginal
state vectors and is identifiability-limited: many Q matrices produce the
same marginal changes. This module uses clonotypes as natural lineage
tracers — per-clone state vectors (fractional distribution over
site × phenotype at each timepoint) are paired across consecutive
timepoints, the empirical 3K × 3K transition matrix P is estimated, and
P is decomposed into migration, transition, and expansion components.

Convention: P[i, j] is the empirical probability of state i → state j,
so rows of P sum to 1 and ``dst_vec = src_vec @ P`` in row-vector form.
This matches ``expm(Q * dt)`` for the row generator Q built by
``ctmc.build_generator`` and lets the optimizer compare entry-by-entry.
"""
import numpy as np
import pandas as pd
from scipy.linalg import expm, logm
from scipy.optimize import minimize

try:
    from .ctmc import (
        StateSpace, build_generator, decompose_generator,
        _count_params, _unpack_params,
    )
except ImportError:
    from ctmc import (
        StateSpace, build_generator, decompose_generator,
        _count_params, _unpack_params,
    )


# ------------------------------------------------------------------
# 1. Clone-level state vectors and observation pairs
# ------------------------------------------------------------------

def build_clone_state_vectors(adata, ss, transitions,
                              clone_key="trb", phenotype_key="phenotype",
                              tissue_key="tissue", time_key="timepoint",
                              patient_key="patient", min_cells=2):
    """Build per-clone fractional state vectors and pair consecutive timepoints.

    For each (clone, timepoint) where the clone has >= ``min_cells`` cells
    summed across tissues, builds a 3K-dimensional vector summing to 1.0:
        vec[ss.idx(tissue, phenotype)] = n_cells_{clone,tissue,pheno}
                                          / n_cells_{clone,total at t}

    Consecutive pairs (t_src, t_dst) listed in ``transitions`` are kept only
    if the clone has >= ``min_cells`` at BOTH timepoints. Clones that
    "disappear" (have 0 cells at t+1) are skipped.

    Returns
    -------
    observations : list of dict
        keys: ``trb``, ``patient``, ``t_src``, ``t_dst``, ``src_vec`` (3K
        float array summing to 1), ``dst_vec`` (3K float array summing to 1),
        ``n_cells_src`` (int), ``n_cells_dst`` (int).
    """
    sites_set = set(ss.sites)
    phenos_set = set(ss.phenotypes)

    cols = [clone_key, tissue_key, phenotype_key, time_key, patient_key]
    obs = adata.obs[cols].copy()
    obs[time_key] = obs[time_key].astype(str)
    obs = obs.dropna(subset=[clone_key])

    n_total_clones = int(obs[clone_key].nunique())

    # per (clone, t, tissue, phenotype) counts
    grp = (obs.groupby([clone_key, time_key, tissue_key, phenotype_key],
                       observed=True)
              .size()
              .rename("n")
              .reset_index())

    # patient lookup (one patient per clone — pick first)
    pat_by_clone = (obs[[clone_key, patient_key]]
                    .drop_duplicates(subset=[clone_key])
                    .set_index(clone_key)[patient_key]
                    .to_dict())

    # build vectors
    vec_by = {}  # (trb, t) -> (vec, n_total)
    for (trb, t), sub in grp.groupby([clone_key, time_key], observed=True):
        total = int(sub["n"].sum())
        if total < min_cells:
            continue
        vec = np.zeros(ss.N)
        for _, row in sub.iterrows():
            tis = row[tissue_key]
            ph = row[phenotype_key]
            if tis not in sites_set or ph not in phenos_set:
                continue
            vec[ss.idx(tis, ph)] += row["n"]
        s = vec.sum()
        if s <= 0:
            continue
        vec /= s
        vec_by[(trb, t)] = (vec, total)

    n_clones_passing = len({trb for trb, _ in vec_by})

    # pair across transitions
    observations = []
    per_pair_counts = {pair: 0 for pair in transitions}
    for (trb, t_src), (src_vec, n_src) in vec_by.items():
        for ts, td in transitions:
            if t_src != ts:
                continue
            entry = vec_by.get((trb, td))
            if entry is None:
                continue
            dst_vec, n_dst = entry
            observations.append({
                "trb": trb,
                "patient": pat_by_clone.get(trb, "unknown"),
                "t_src": ts,
                "t_dst": td,
                "src_vec": src_vec,
                "dst_vec": dst_vec,
                "n_cells_src": n_src,
                "n_cells_dst": n_dst,
            })
            per_pair_counts[(ts, td)] += 1

    print(f"  Total unique clones: {n_total_clones}")
    print(f"  Clones with >= {min_cells} cells at >=1 timepoint: {n_clones_passing}")
    print(f"  Clone × timepoint vectors built: {len(vec_by)}")
    print(f"  Observation pairs generated: {len(observations)}")
    for (ts, td), n in per_pair_counts.items():
        print(f"    {ts} -> {td}: {n} obs")

    return observations


# ------------------------------------------------------------------
# 2. Empirical P estimation
# ------------------------------------------------------------------

def build_empirical_P(observations, ss, method="weighted_lstsq",
                      pseudocount=0.01):
    """Estimate the 3K × 3K transition matrix P from clone observations.

    Two methods:

    - ``weighted_lstsq``: stack SRC, DST (N_obs × 3K), weight by
      n_cells_src + n_cells_dst, and solve the weighted normal equations
      ``(SRC_w^T SRC_w) P = SRC_w^T DST_w``. Negatives are clipped, then
      a pseudocount is added and rows are normalized to sum to 1.

    - ``direct_count``: for each source state i, take observations whose
      src_vec[i] > 0.05; P[i, :] is the weighted average of their dst_vec
      using weights ``src_vec[i] * (n_cells_src + n_cells_dst)``.

    Returns
    -------
    P : (3K, 3K) row-stochastic transition matrix.
    diagnostics : dict with keys n_obs, row_coverage (per-source-state obs
        count, threshold 0.05), condition_number.
    """
    if len(observations) == 0:
        raise ValueError("No observations provided.")

    SRC = np.stack([o["src_vec"] for o in observations])
    DST = np.stack([o["dst_vec"] for o in observations])
    w = np.array([o["n_cells_src"] + o["n_cells_dst"] for o in observations],
                 dtype=float)
    N_obs, N_states = SRC.shape
    print(f"  Input: N_obs={N_obs}, N_states={N_states}")

    if method == "weighted_lstsq":
        sw = np.sqrt(w)
        SRC_w = SRC * sw[:, None]
        DST_w = DST * sw[:, None]
        gram = SRC_w.T @ SRC_w
        rhs = SRC_w.T @ DST_w
        try:
            P = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(gram) @ rhs
    elif method == "direct_count":
        threshold = 0.05
        P = np.zeros((N_states, N_states))
        for i in range(N_states):
            mask = SRC[:, i] > threshold
            if mask.sum() == 0:
                continue
            wts = SRC[mask, i] * w[mask]
            wsum = wts.sum()
            if wsum <= 0:
                continue
            P[i] = (DST[mask] * wts[:, None]).sum(axis=0) / wsum
    else:
        raise ValueError(f"Unknown method: {method}")

    # clip negatives → pseudocount → row-normalize
    P = np.clip(P, 0.0, None)
    P = P + pseudocount
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1.0)

    # diagnostics
    row_coverage = np.array([int((SRC[:, i] > 0.05).sum())
                             for i in range(N_states)])
    try:
        cond = float(np.linalg.cond(SRC.T @ SRC))
    except Exception:
        cond = float("nan")

    n_sparse = int((row_coverage < 5).sum())
    diag_vals = np.diag(P)
    print(f"  P shape: {P.shape}")
    print(f"  Condition number of SRC^T SRC: {cond:.3e}")
    print(f"  Source states with < 5 observations (potentially noisy): "
          f"{n_sparse} / {N_states}")
    print(f"  Diagonal min/max/mean: {diag_vals.min():.4f} / "
          f"{diag_vals.max():.4f} / {diag_vals.mean():.4f}")

    return P, {
        "n_obs": N_obs,
        "row_coverage": row_coverage,
        "condition_number": cond,
    }


# ------------------------------------------------------------------
# 3. Decompose via matrix logarithm
# ------------------------------------------------------------------

def decompose_P_logm(P, ss, dt=1.0):
    """Decompose empirical P via the matrix logarithm: ``Q = logm(P) / dt``.

    Logm-derived Q is not guaranteed to be a valid generator (positive
    off-diagonals, non-positive row sums). Violations are counted and
    reported but the decomposition proceeds regardless.
    """
    Q_complex = logm(P) / dt
    imag_max = float(np.abs(Q_complex.imag).max())
    if imag_max > 1e-8:
        print(f"  WARNING: logm(P) has non-trivial imaginary parts "
              f"(max |Im| = {imag_max:.3e}); taking real part.")
    Q = np.asarray(Q_complex.real)

    tol = 1e-6
    N = Q.shape[0]
    off_neg = 0
    off_neg_max = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if Q[i, j] < -tol:
                off_neg += 1
                off_neg_max = max(off_neg_max, -float(Q[i, j]))

    row_sums = Q.sum(axis=1)
    row_pos = int((row_sums > tol).sum())
    row_pos_max = float(np.maximum(row_sums, 0.0).max())

    n_violations = off_neg + row_pos
    valid = (n_violations == 0)
    if not valid:
        print(f"  WARNING: Q from logm is not a valid generator. "
              f"Off-diag negatives: {off_neg} (max magnitude {off_neg_max:.3e}); "
              f"row-sums > 0: {row_pos} (max {row_pos_max:.3e}).")
    else:
        print("  Q is a valid generator (off-diag >= 0, row-sums <= 0).")

    migration, transition, expansion = decompose_generator(Q, ss)

    return {
        "Q": Q,
        "migration": migration,
        "transition": transition,
        "expansion": expansion,
        "valid": valid,
        "n_violations": int(n_violations),
        "ss": ss,
        "dt": dt,
    }


# ------------------------------------------------------------------
# 4. Decompose via structured CTMC optimizer
# ------------------------------------------------------------------

def _frobenius_loss(x, P_target, ss, dt, shared_migration, shared_transition,
                    fit_expansion, l2_reg, progress):
    migration, transition, expansion = _unpack_params(
        x, ss, shared_migration, shared_transition, fit_expansion)
    Q = build_generator(migration, transition, expansion, ss)
    try:
        P_pred = expm(Q * dt)
    except Exception:
        return 1e12
    diff = P_pred - P_target
    loss = float(np.sum(diff * diff))
    if l2_reg > 0:
        n_mig, n_trans, _ = _count_params(
            ss, shared_migration, shared_transition, fit_expansion)
        rates = np.exp(x[:n_mig + n_trans])
        loss += l2_reg * float(np.sum(rates * rates))

    progress["count"] += 1
    if loss < progress["best"]:
        progress["best"] = loss
    if (progress["verbose"]
            and progress["interval"] > 0
            and progress["count"] % progress["interval"] == 0):
        print(f"    eval {progress['count']:>5d}: "
              f"loss={loss:.6e} (best={progress['best']:.6e})")
    return loss


def decompose_P_optimizer(P, ss, dt=1.0, shared_migration=False,
                          shared_transition=False, fit_expansion=False,
                          l2_reg=0.01, init=None, method="L-BFGS-B",
                          maxiter=5000, verbose=True, print_every=50):
    """Decompose empirical P into structured CTMC components via Frobenius fit.

    Minimizes ``||expm(Q*dt) - P||_F^2`` over the structured
    migration/transition/expansion parameterization used in
    :func:`trafficking.ctmc.fit_ctmc`.
    """
    n_mig, n_trans, n_exp = _count_params(
        ss, shared_migration, shared_transition, fit_expansion)
    n_params = n_mig + n_trans + n_exp

    if init is None:
        x0 = np.zeros(n_params)
        x0[:n_mig + n_trans] = np.log(0.1)
    else:
        x0 = np.asarray(init, dtype=float)

    progress = {"count": 0, "best": np.inf, "verbose": verbose,
                "interval": print_every}

    if verbose:
        print(f"  Fitting structured Q to empirical P (Frobenius norm): "
              f"{n_params} parameters "
              f"({n_mig} migration, {n_trans} transition, {n_exp} expansion)")

    res = minimize(
        _frobenius_loss, x0,
        args=(P, ss, dt, shared_migration, shared_transition,
              fit_expansion, l2_reg, progress),
        method=method,
        options={"maxiter": maxiter, "disp": verbose},
    )

    migration, transition, expansion = _unpack_params(
        res.x, ss, shared_migration, shared_transition, fit_expansion)
    Q = build_generator(migration, transition, expansion, ss)
    P_fitted = expm(Q * dt)
    frob_err = float(np.sqrt(np.sum((P_fitted - P) ** 2)))

    if verbose:
        print(f"  Final loss: {res.fun:.6e}, converged: {res.success}")
        print(f"  Frobenius error ||P_fitted - P||_F = {frob_err:.4e}")

    return {
        "Q": Q,
        "migration": migration,
        "transition": transition,
        "expansion": expansion,
        "loss": res.fun,
        "params": res.x,
        "scipy_result": res,
        "ss": ss,
        "dt": dt,
        "config": {
            "shared_migration": shared_migration,
            "shared_transition": shared_transition,
            "fit_expansion": fit_expansion,
            "l2_reg": l2_reg,
        },
        "P_empirical": P,
        "P_fitted": P_fitted,
        "frobenius_error": frob_err,
    }


# ------------------------------------------------------------------
# 5. Rate comparison table
# ------------------------------------------------------------------

def rate_comparison_table(result_logm, result_optimizer):
    """Compare rates from the logm vs optimizer decompositions."""
    ss = result_logm["ss"]
    records = []

    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            for k in range(ss.K):
                a = float(result_logm["migration"][s_from, s_to, k])
                b = float(result_optimizer["migration"][s_from, s_to, k])
                records.append({
                    "component": "migration",
                    "from": ss.sites[s_from],
                    "to": ss.sites[s_to],
                    "site_or_phenotype": ss.phenotypes[k],
                    "rate_logm": a,
                    "rate_optimizer": b,
                    "abs_diff": abs(a - b),
                    "sign_agree": bool((a >= 0) == (b >= 0)),
                })

    for s in range(ss.S):
        for k_from in range(ss.K):
            for k_to in range(ss.K):
                if k_from == k_to:
                    continue
                a = float(result_logm["transition"][s, k_from, k_to])
                b = float(result_optimizer["transition"][s, k_from, k_to])
                records.append({
                    "component": "transition",
                    "from": ss.phenotypes[k_from],
                    "to": ss.phenotypes[k_to],
                    "site_or_phenotype": ss.sites[s],
                    "rate_logm": a,
                    "rate_optimizer": b,
                    "abs_diff": abs(a - b),
                    "sign_agree": bool((a >= 0) == (b >= 0)),
                })

    for s in range(ss.S):
        for k in range(ss.K):
            a = float(result_logm["expansion"][s, k])
            b = float(result_optimizer["expansion"][s, k])
            records.append({
                "component": "expansion",
                "from": ss.sites[s],
                "to": ss.sites[s],
                "site_or_phenotype": ss.phenotypes[k],
                "rate_logm": a,
                "rate_optimizer": b,
                "abs_diff": abs(a - b),
                "sign_agree": bool((a >= 0) == (b >= 0)),
            })

    df = (pd.DataFrame(records)
            .sort_values("abs_diff", ascending=False)
            .reset_index(drop=True))

    print("\nTop 10 largest disagreements (logm vs optimizer):")
    print(df.head(10).to_string(index=False))

    return df


# ------------------------------------------------------------------
# 6. Summary report
# ------------------------------------------------------------------

def summary_report(P, result, ss, observations):
    """Print a structured summary of empirical P and decomposed Q."""
    print("=" * 60)
    print("Empirical P / decomposed Q summary")
    print("=" * 60)

    n_obs = len(observations)
    n_pat = len({o["patient"] for o in observations}) if n_obs else 0
    n_clones = len({o["trb"] for o in observations}) if n_obs else 0
    print(f"\nObservations: {n_obs}")
    print(f"Patients: {n_pat}")
    print(f"Clones: {n_clones}")

    print("\nPer-tissue diagonal-block dominance "
          "(probability mass kept in the same tissue):")
    for site in ss.sites:
        idx = ss.site_indices(site)
        block_total = float(P[np.ix_(idx, idx)].sum())
        row_total = float(P[idx, :].sum())
        frac = block_total / row_total if row_total > 0 else 0.0
        print(f"  {site}: {frac:.3f}  "
              f"(block sum {block_total:.3f} / row total {row_total:.3f})")

    mig = result["migration"]
    mig_rows = []
    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            for k in range(ss.K):
                mig_rows.append((ss.sites[s_from], ss.sites[s_to],
                                 ss.phenotypes[k],
                                 float(mig[s_from, s_to, k])))
    mig_rows.sort(key=lambda r: r[3], reverse=True)
    print("\nTop 5 migration rates (cross-tissue, same phenotype):")
    for src, dst, ph, r in mig_rows[:5]:
        print(f"  {src} -> {dst} [{ph}]: {r:+.4f}")

    trans = result["transition"]
    trans_rows = []
    for s in range(ss.S):
        for k_from in range(ss.K):
            for k_to in range(ss.K):
                if k_from == k_to:
                    continue
                trans_rows.append((ss.sites[s], ss.phenotypes[k_from],
                                   ss.phenotypes[k_to],
                                   float(trans[s, k_from, k_to])))
    trans_rows.sort(key=lambda r: r[3], reverse=True)
    print("\nTop 5 phenotype-transition rates (same tissue):")
    for site, fp, tp, r in trans_rows[:5]:
        print(f"  {fp} -> {tp} @ {site}: {r:+.4f}")

    exp = result["expansion"]
    print("\nExpansion rates per (tissue, phenotype):")
    for s in range(ss.S):
        for k in range(ss.K):
            r = float(exp[s, k])
            print(f"  {ss.sites[s]} [{ss.phenotypes[k]}]: {r:+.4f}")

    print("\nMigration ordering by mean rate over phenotypes "
          "(compare against trafficking rates from "
          "results/traffic_branch_empirics/edge_metrics_table.csv):")
    edge_means = []
    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            edge_means.append((ss.sites[s_from], ss.sites[s_to],
                               float(mig[s_from, s_to].mean())))
    edge_means.sort(key=lambda r: r[2], reverse=True)
    for src, dst, m in edge_means:
        print(f"  {src} -> {dst}: {m:+.4f}")


# ==================================================================
# Augmented (born/die) variant
# ==================================================================
#
# The augmented state space has one extra "unobserved" state at index
# ``ss.N``. Clones can enter the system (transition unobserved -> state j)
# or exit (state i -> unobserved). The augmented P is (3K+1) x (3K+1),
# row-stochastic. The interior 3K x 3K block is NOT row-stochastic — its
# rows sum to (1 - exit_prob_i), and that deficit IS the expansion / death
# signal that the optimizer absorbs when ``fit_expansion=True``.


UNOBS_LABEL = "__UNOBSERVED__"


def build_clone_state_vectors_augmented(
        adata, ss, transitions,
        clone_key="trb", phenotype_key="phenotype",
        tissue_key="tissue", time_key="timepoint",
        patient_key="patient", min_cells=2):
    """Build (3K+1)-dimensional clone state vectors with an unobserved state.

    Index ``ss.N`` is the "unobserved" pseudo-state. For each clone and each
    transition pair ``(t_src, t_dst)``:

    - both sides have >= ``min_cells`` → normal observation, both vectors
      are 3K distributions padded with 0 at index ss.N.
    - src side >= min_cells, dst side absent / too few → exit observation,
      ``dst_vec`` is one-hot at ``ss.N``.
    - src side absent / too few, dst side >= min_cells → entry observation,
      ``src_vec`` is one-hot at ``ss.N``.
    - neither side passes → skipped.

    Returns the same observation-dict format as
    :func:`build_clone_state_vectors` plus a ``"kind"`` field
    (``"normal" | "exit" | "entry"``).
    """
    sites_set = set(ss.sites)
    phenos_set = set(ss.phenotypes)
    N_aug = ss.N + 1
    UNOBS = ss.N

    cols = [clone_key, tissue_key, phenotype_key, time_key, patient_key]
    obs = adata.obs[cols].copy()
    obs[time_key] = obs[time_key].astype(str)
    obs = obs.dropna(subset=[clone_key])

    n_total_clones = int(obs[clone_key].nunique())

    grp = (obs.groupby([clone_key, time_key, tissue_key, phenotype_key],
                       observed=True)
              .size()
              .rename("n")
              .reset_index())

    pat_by_clone = (obs[[clone_key, patient_key]]
                    .drop_duplicates(subset=[clone_key])
                    .set_index(clone_key)[patient_key]
                    .to_dict())

    # Build interior (3K-dim) vec for every (trb, t) with >= min_cells.
    vec_by = {}
    for (trb, t), sub in grp.groupby([clone_key, time_key], observed=True):
        total = int(sub["n"].sum())
        if total < min_cells:
            continue
        vec = np.zeros(ss.N)
        for _, row in sub.iterrows():
            tis = row[tissue_key]
            ph = row[phenotype_key]
            if tis not in sites_set or ph not in phenos_set:
                continue
            vec[ss.idx(tis, ph)] += row["n"]
        s = vec.sum()
        if s <= 0:
            continue
        vec /= s
        vec_by[(trb, t)] = (vec, total)

    n_clones_passing = len({trb for trb, _ in vec_by})

    observations = []
    per_pair_counts = {pair: {"normal": 0, "exit": 0, "entry": 0}
                       for pair in transitions}

    for ts, td in transitions:
        # Clones present at either end of this transition.
        clones_here = {trb for (trb, t) in vec_by if t in (ts, td)}
        for trb in clones_here:
            src_entry = vec_by.get((trb, ts))
            dst_entry = vec_by.get((trb, td))

            if src_entry is not None and dst_entry is not None:
                src_vec = np.zeros(N_aug)
                src_vec[:ss.N] = src_entry[0]
                dst_vec = np.zeros(N_aug)
                dst_vec[:ss.N] = dst_entry[0]
                n_src = src_entry[1]
                n_dst = dst_entry[1]
                kind = "normal"
            elif src_entry is not None:
                src_vec = np.zeros(N_aug)
                src_vec[:ss.N] = src_entry[0]
                dst_vec = np.zeros(N_aug)
                dst_vec[UNOBS] = 1.0
                n_src = src_entry[1]
                n_dst = 0
                kind = "exit"
            elif dst_entry is not None:
                src_vec = np.zeros(N_aug)
                src_vec[UNOBS] = 1.0
                dst_vec = np.zeros(N_aug)
                dst_vec[:ss.N] = dst_entry[0]
                n_src = 0
                n_dst = dst_entry[1]
                kind = "entry"
            else:
                continue

            observations.append({
                "trb": trb,
                "patient": pat_by_clone.get(trb, "unknown"),
                "t_src": ts, "t_dst": td,
                "src_vec": src_vec, "dst_vec": dst_vec,
                "n_cells_src": int(n_src),
                "n_cells_dst": int(n_dst),
                "kind": kind,
            })
            per_pair_counts[(ts, td)][kind] += 1

    n_normal = sum(c["normal"] for c in per_pair_counts.values())
    n_exit = sum(c["exit"] for c in per_pair_counts.values())
    n_entry = sum(c["entry"] for c in per_pair_counts.values())
    print(f"  Total unique clones: {n_total_clones}")
    print(f"  Clones with >= {min_cells} cells at >=1 timepoint: "
          f"{n_clones_passing}")
    print(f"  Clone x timepoint vectors built: {len(vec_by)}")
    print(f"  Augmented observation pairs: {len(observations)} "
          f"(normal={n_normal}, exit={n_exit}, entry={n_entry})")
    for (ts, td), c in per_pair_counts.items():
        print(f"    {ts} -> {td}: total={c['normal'] + c['exit'] + c['entry']}, "
              f"normal={c['normal']}, exit={c['exit']}, entry={c['entry']}")

    return observations


def build_empirical_P_augmented(observations, ss,
                                method="weighted_lstsq", pseudocount=0.01):
    """Estimate the (3K+1) x (3K+1) augmented transition matrix.

    Thin wrapper around :func:`build_empirical_P`. Reuses the same weighted
    least-squares / row-normalization logic on (3K+1)-dimensional vectors so
    that the returned ``P_aug`` is row-stochastic. The interior 3K block
    will *not* be row-stochastic, since each row's mass deficit equals the
    probability that the clone exits.
    """
    if len(observations) == 0:
        raise ValueError("No augmented observations provided.")
    N_states = observations[0]["src_vec"].shape[0]
    if N_states != ss.N + 1:
        raise ValueError(
            f"Augmented observations have dimension {N_states}, "
            f"expected ss.N + 1 = {ss.N + 1}.")
    print(f"  Augmented state-space dim: {N_states} (ss.N + 1 = {ss.N + 1})")
    return build_empirical_P(observations, ss, method=method,
                             pseudocount=pseudocount)


def decompose_P_augmented(P_aug, ss, dt=1.0, **optimizer_kwargs):
    """Decompose an augmented P into structured rates + exit/entry tables.

    Extracts:
      - ``P_interior`` = ``P_aug[:ss.N, :ss.N]`` (rows sum to < 1)
      - ``p_exit`` = ``P_aug[:ss.N, ss.N]`` — exit prob per source state
      - ``p_entry`` = ``P_aug[ss.N, :ss.N]`` — entry prob per dest state

    Runs :func:`decompose_P_optimizer` on the interior block with
    ``fit_expansion=True`` so the row-sum deficit becomes the expansion
    (death/birth) signal absorbed into Q's diagonals.

    Returns a dict with the same keys as :func:`decompose_P_optimizer` plus
    ``exit_rates`` (DataFrame), ``entry_rates`` (DataFrame), ``p_exit``,
    ``p_entry`` and ``P_augmented`` (original input).
    """
    N_aug = ss.N + 1
    if P_aug.shape != (N_aug, N_aug):
        raise ValueError(
            f"P_aug shape {P_aug.shape} != ({N_aug}, {N_aug}).")

    P_interior = P_aug[:ss.N, :ss.N]
    p_exit = P_aug[:ss.N, ss.N]
    p_entry = P_aug[ss.N, :ss.N]

    optimizer_kwargs = dict(optimizer_kwargs)
    optimizer_kwargs.setdefault("fit_expansion", True)
    optimizer_kwargs.setdefault("verbose", True)

    print(f"  Interior P row sums: "
          f"min={P_interior.sum(axis=1).min():.4f}, "
          f"max={P_interior.sum(axis=1).max():.4f} "
          f"(deficit -> exit / death signal).")
    print(f"  Exit prob range: [{p_exit.min():.4f}, {p_exit.max():.4f}]")
    print(f"  Entry prob range: [{p_entry.min():.4f}, {p_entry.max():.4f}]")

    result = decompose_P_optimizer(P_interior, ss, dt=dt, **optimizer_kwargs)

    exit_rows = []
    entry_rows = []
    for s in range(ss.S):
        for k in range(ss.K):
            idx = ss.idx(ss.sites[s], ss.phenotypes[k])
            exit_rows.append({
                "tissue": ss.sites[s],
                "phenotype": ss.phenotypes[k],
                "exit_prob": float(p_exit[idx]),
            })
            entry_rows.append({
                "tissue": ss.sites[s],
                "phenotype": ss.phenotypes[k],
                "entry_prob": float(p_entry[idx]),
            })

    result["exit_rates"] = pd.DataFrame(exit_rows)
    result["entry_rates"] = pd.DataFrame(entry_rows)
    result["p_exit"] = p_exit
    result["p_entry"] = p_entry
    result["P_augmented"] = P_aug
    return result
