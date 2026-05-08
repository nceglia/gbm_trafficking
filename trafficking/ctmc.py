import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import pandas as pd
from itertools import product


class StateSpace:
    """Joint (site × phenotype) state space with index mapping."""

    def __init__(self, sites, phenotypes):
        self.sites = list(sites)
        self.phenotypes = list(phenotypes)
        self.S = len(self.sites)
        self.K = len(self.phenotypes)
        self.N = self.S * self.K
        self.labels = [(s, p) for s in self.sites for p in self.phenotypes]
        self._idx = {label: i for i, label in enumerate(self.labels)}

    def idx(self, site, phenotype):
        return self._idx[(site, phenotype)]

    def label(self, i):
        return self.labels[i]

    def site_slice(self, site):
        s = self.sites.index(site)
        return slice(s * self.K, (s + 1) * self.K)

    def site_indices(self, site):
        s = self.sites.index(site)
        return list(range(s * self.K, (s + 1) * self.K))

    def phenotype_indices(self, phenotype):
        k = self.phenotypes.index(phenotype)
        return [s * self.K + k for s in range(self.S)]

    def __repr__(self):
        return f"StateSpace({self.S} sites × {self.K} phenotypes = {self.N} states)"


# ------------------------------------------------------------------
# Generator construction
# ------------------------------------------------------------------

def build_generator(migration, transition, expansion, ss):
    """Assemble generator matrix Q from decomposed rate components.

    Parameters
    ----------
    migration : (S, S, K) array — migration rate from site i to site j for phenotype k.
                Diagonal is ignored (set automatically).
    transition : (S, K, K) array — transition rate from phenotype j to k at site s.
                 Diagonal is ignored.
    expansion : (S, K) array — growth (>0) or death (<0) rate per site/phenotype.
    ss : StateSpace

    Returns
    -------
    Q : (N, N) generator matrix
    """
    N = ss.N
    Q = np.zeros((N, N))

    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            for k in range(ss.K):
                i = s_from * ss.K + k
                j = s_to * ss.K + k
                Q[i, j] = migration[s_from, s_to, k]

    for s in range(ss.S):
        for k_from in range(ss.K):
            for k_to in range(ss.K):
                if k_from == k_to:
                    continue
                i = s * ss.K + k_from
                j = s * ss.K + k_to
                Q[i, j] = transition[s, k_from, k_to]

    for i in range(N):
        Q[i, i] = -Q[i].sum()

    exp = expansion.ravel()
    for i in range(N):
        Q[i, i] += exp[i]

    return Q


def decompose_generator(Q, ss):
    """Extract migration, transition, expansion components from a fitted Q."""
    migration = np.zeros((ss.S, ss.S, ss.K))
    transition = np.zeros((ss.S, ss.K, ss.K))

    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            for k in range(ss.K):
                i = s_from * ss.K + k
                j = s_to * ss.K + k
                migration[s_from, s_to, k] = Q[i, j]

    for s in range(ss.S):
        for k_from in range(ss.K):
            for k_to in range(ss.K):
                if k_from == k_to:
                    continue
                i = s * ss.K + k_from
                j = s * ss.K + k_to
                transition[s, k_from, k_to] = Q[i, j]

    offdiag_sum = np.zeros(ss.N)
    for i in range(ss.N):
        offdiag_sum[i] = Q[i].sum() - Q[i, i]
    expansion = (Q.diagonal() + offdiag_sum).reshape(ss.S, ss.K)

    return migration, transition, expansion


# ------------------------------------------------------------------
# Parameterization (unconstrained <-> constrained)
# ------------------------------------------------------------------

def _count_params(ss, shared_migration=False, shared_transition=False, fit_expansion=False):
    S, K = ss.S, ss.K
    n_mig = S * (S - 1) * (1 if shared_migration else K)
    n_trans = K * (K - 1) * (1 if shared_transition else S)
    n_exp = S * K if fit_expansion else 0
    return n_mig, n_trans, n_exp


def _unpack_params(x, ss, shared_migration=False, shared_transition=False, fit_expansion=False):
    S, K = ss.S, ss.K
    n_mig, n_trans, n_exp = _count_params(ss, shared_migration, shared_transition, fit_expansion)
    pos = 0

    raw_mig = np.exp(x[pos:pos + n_mig])
    pos += n_mig
    migration = np.zeros((S, S, K))
    idx = 0
    for s_from in range(S):
        for s_to in range(S):
            if s_from == s_to:
                continue
            if shared_migration:
                migration[s_from, s_to, :] = raw_mig[idx]
                idx += 1
            else:
                migration[s_from, s_to, :] = raw_mig[idx:idx + K]
                idx += K

    raw_trans = np.exp(x[pos:pos + n_trans])
    pos += n_trans
    transition = np.zeros((S, K, K))
    idx = 0
    for k_from in range(K):
        for k_to in range(K):
            if k_from == k_to:
                continue
            if shared_transition:
                transition[:, k_from, k_to] = raw_trans[idx]
                idx += 1
            else:
                transition[:, k_from, k_to] = raw_trans[idx:idx + S]
                idx += S

    expansion = np.zeros((S, K))
    if fit_expansion:
        expansion = x[pos:pos + n_exp].reshape(S, K)

    return migration, transition, expansion


def _pack_params(migration, transition, expansion, ss,
                 shared_migration=False, shared_transition=False, fit_expansion=False):
    S, K = ss.S, ss.K
    parts = []

    for s_from in range(S):
        for s_to in range(S):
            if s_from == s_to:
                continue
            if shared_migration:
                parts.append(np.log(migration[s_from, s_to, 0] + 1e-10))
            else:
                parts.extend(np.log(migration[s_from, s_to, :] + 1e-10))

    for k_from in range(K):
        for k_to in range(K):
            if k_from == k_to:
                continue
            if shared_transition:
                parts.append(np.log(transition[0, k_from, k_to] + 1e-10))
            else:
                parts.extend(np.log(transition[:, k_from, k_to] + 1e-10))

    if fit_expansion:
        parts.extend(expansion.ravel())

    return np.array(parts)


# ------------------------------------------------------------------
# Fitting
# ------------------------------------------------------------------

def _loss_fn(x, observations, ss, dt, shared_migration, shared_transition,
             fit_expansion, l2_reg):
    migration, transition, expansion = _unpack_params(
        x, ss, shared_migration, shared_transition, fit_expansion)
    Q = build_generator(migration, transition, expansion, ss)

    try:
        P = expm(Q * dt)
    except Exception:
        return 1e12

    loss = 0.0
    for n_obs, n_next in observations:
        n_pred = P @ n_obs
        if n_pred.sum() > 0:
            n_pred_norm = n_pred / n_pred.sum()
        else:
            n_pred_norm = n_pred
        if n_next.sum() > 0:
            n_next_norm = n_next / n_next.sum()
        else:
            n_next_norm = n_next
        loss += np.sum((n_pred_norm - n_next_norm) ** 2)

    if l2_reg > 0:
        loss += l2_reg * np.sum(np.exp(x[:sum(_count_params(
            ss, shared_migration, shared_transition, fit_expansion)[:2])]) ** 2)

    return loss


def fit_ctmc(observations, ss, dt=1.0, shared_migration=False, shared_transition=False,
             fit_expansion=False, l2_reg=0.01, init=None, method="L-BFGS-B", maxiter=5000,
             verbose=True):
    """Fit a CTMC generator matrix to observed state transitions.

    Parameters
    ----------
    observations : list of (n_t, n_{t+1}) pairs
        Each element is a tuple of two arrays of shape (N,) representing the
        state vector at consecutive timepoints. Can include observations from
        multiple patients/intervals.
    ss : StateSpace
    dt : float — time between observations (assumed constant).
    shared_migration : bool — if True, migration rates are phenotype-independent.
    shared_transition : bool — if True, transition rates are site-independent.
    fit_expansion : bool — if True, fit growth/death rates (rows of Q don't sum to 0).
    l2_reg : float — L2 penalty on rate magnitudes.
    init : array or None — initial parameter vector.
    method : str — scipy optimizer method.
    maxiter : int
    verbose : bool

    Returns
    -------
    result : dict with keys Q, migration, transition, expansion, loss, scipy_result
    """
    n_mig, n_trans, n_exp = _count_params(ss, shared_migration, shared_transition, fit_expansion)
    n_params = n_mig + n_trans + n_exp

    if init is None:
        x0 = np.zeros(n_params)
        x0[:n_mig + n_trans] = np.log(0.1)
    else:
        x0 = init

    if verbose:
        print(f"Fitting CTMC: {n_params} parameters "
              f"({n_mig} migration, {n_trans} transition, {n_exp} expansion)")
        print(f"  {len(observations)} observation pairs, dt={dt}")

    res = minimize(
        _loss_fn, x0,
        args=(observations, ss, dt, shared_migration, shared_transition,
              fit_expansion, l2_reg),
        method=method,
        options={"maxiter": maxiter, "disp": verbose},
    )

    migration, transition, expansion = _unpack_params(
        res.x, ss, shared_migration, shared_transition, fit_expansion)
    Q = build_generator(migration, transition, expansion, ss)

    if verbose:
        print(f"  Final loss: {res.fun:.6f}, converged: {res.success}")

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
    }


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict(state, Q, dt):
    """Predict the next state vector: n(t+dt) = expm(Q*dt) @ n(t)."""
    P = expm(Q * dt)
    return P @ state


def predict_trajectory(initial_state, Q, n_steps, dt=1.0):
    """Chain forward predictions across multiple intervals."""
    P = expm(Q * dt)
    trajectory = [initial_state.copy()]
    for _ in range(n_steps):
        trajectory.append(P @ trajectory[-1])
    return np.array(trajectory)


def transition_matrix(Q, dt):
    """Compute the transition matrix P = expm(Q * dt)."""
    return expm(Q * dt)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def holdout_score(observations, ss, holdout_idx=-1, **fit_kwargs):
    """Fit on all but one interval, predict the held-out, return MSE."""
    train = [obs for i, obs in enumerate(observations) if i != holdout_idx % len(observations)]
    held = observations[holdout_idx % len(observations)]

    result = fit_ctmc(train, ss, verbose=False, **fit_kwargs)
    n_pred = predict(held[0], result["Q"], result["dt"])

    pred_norm = n_pred / n_pred.sum() if n_pred.sum() > 0 else n_pred
    obs_norm = held[1] / held[1].sum() if held[1].sum() > 0 else held[1]
    mse = np.mean((pred_norm - obs_norm) ** 2)

    return {"mse": mse, "predicted": n_pred, "observed": held[1], "result": result}


def leave_one_interval_out(observations, ss, **fit_kwargs):
    """Leave-one-out cross-validation across intervals."""
    scores = []
    for i in range(len(observations)):
        sc = holdout_score(observations, ss, holdout_idx=i, **fit_kwargs)
        scores.append(sc["mse"])
        print(f"  Interval {i}: MSE = {sc['mse']:.6f}")
    print(f"  Mean MSE: {np.mean(scores):.6f} ± {np.std(scores):.6f}")
    return scores


# ------------------------------------------------------------------
# Dataset integration
# ------------------------------------------------------------------

def observations_from_dataset(dataset, modality="tcell", patient=None, normalize=True):
    """Extract consecutive (state_t, state_{t+1}) pairs from a GBMDataset."""
    tps = dataset.timepoints
    sites = dataset.tissues
    phenotypes = dataset.phenotypes(modality)
    ss = StateSpace(sites, phenotypes)

    states = {}
    for tp in tps:
        sv = dataset.get_state_vector(modality, timepoint=tp, patient=patient,
                                       normalize=normalize)
        vec = np.zeros(ss.N)
        for (site, pheno), val in sv.items():
            if (site, pheno) in ss._idx:
                vec[ss.idx(site, pheno)] = val
        states[tp] = vec

    observations = []
    for i in range(len(tps) - 1):
        n_t = states[tps[i]]
        n_next = states[tps[i + 1]]
        if n_t.sum() > 0 and n_next.sum() > 0:
            observations.append((n_t, n_next))

    return observations, ss


def observations_from_dataset_by_patient(dataset, modality="tcell", normalize=True):
    """Extract observations pooled across all patients."""
    all_obs = []
    ss = None
    for pat in dataset.patients:
        obs, ss_ = observations_from_dataset(dataset, modality, patient=pat, normalize=normalize)
        all_obs.extend(obs)
        if ss is None:
            ss = ss_
    return all_obs, ss


# ------------------------------------------------------------------
# Rate summary tables
# ------------------------------------------------------------------

def migration_table(result):
    """Readable migration rate table: (source_site, dest_site, phenotype, rate)."""
    ss = result["ss"]
    mig = result["migration"]
    records = []
    for s_from in range(ss.S):
        for s_to in range(ss.S):
            if s_from == s_to:
                continue
            for k in range(ss.K):
                records.append({
                    "from": ss.sites[s_from], "to": ss.sites[s_to],
                    "phenotype": ss.phenotypes[k], "rate": mig[s_from, s_to, k],
                })
    return pd.DataFrame(records).sort_values("rate", ascending=False)


def transition_table(result):
    """Readable phenotype transition rate table: (site, source_pheno, dest_pheno, rate)."""
    ss = result["ss"]
    trans = result["transition"]
    records = []
    for s in range(ss.S):
        for k_from in range(ss.K):
            for k_to in range(ss.K):
                if k_from == k_to:
                    continue
                records.append({
                    "site": ss.sites[s],
                    "from": ss.phenotypes[k_from], "to": ss.phenotypes[k_to],
                    "rate": trans[s, k_from, k_to],
                })
    return pd.DataFrame(records).sort_values("rate", ascending=False)


def expansion_table(result):
    """Readable expansion rate table: (site, phenotype, rate)."""
    ss = result["ss"]
    exp = result["expansion"]
    records = []
    for s in range(ss.S):
        for k in range(ss.K):
            records.append({
                "site": ss.sites[s], "phenotype": ss.phenotypes[k],
                "rate": exp[s, k],
            })
    return pd.DataFrame(records).sort_values("rate", ascending=False)


def rate_summary(result):
    """Print a concise summary of all fitted rates."""
    ss = result["ss"]
    print(f"CTMC Rate Summary ({ss})")
    print(f"  Loss: {result['loss']:.6f}")
    print(f"\nTop migration rates:")
    mt = migration_table(result)
    for _, r in mt.head(6).iterrows():
        print(f"  {r['from']} → {r['to']} [{r['phenotype']}]: {r['rate']:.4f}")
    print(f"\nTop transition rates:")
    tt = transition_table(result)
    for _, r in tt.head(6).iterrows():
        print(f"  {r['from']} → {r['to']} @ {r['site']}: {r['rate']:.4f}")
    if result["config"]["fit_expansion"]:
        print(f"\nExpansion rates:")
        et = expansion_table(result)
        for _, r in et.iterrows():
            print(f"  {r['site']} [{r['phenotype']}]: {r['rate']:+.4f}")