import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import numpy as np
import pandas as pd

from .data import extract_transitions, extract_temporal_transitions, prepare_tensors, summary
from .model import transition_model, transition_guide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_svi(data, phenotypes, n_steps=3000, lr=0.01, hierarchical=True, verbose=True):
    """Run stochastic variational inference on extracted transition data."""
    theta, dst, n_dst, pat_ids, pat_names = prepare_tensors(data)
    K = len(phenotypes)
    P = len(pat_names)

    pyro.clear_param_store()
    optimizer = ClippedAdam({"lr": lr, "betas": (0.9, 0.999)})
    svi = SVI(transition_model, transition_guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(n_steps):
        loss = svi.step(theta, dst, n_dst, pat_ids, K, P, hierarchical)
        losses.append(loss)
        if verbose and step % 500 == 0:
            print(f"  Step {step}: ELBO = {loss:.1f}")

    if verbose:
        print(f"  Final ELBO: {losses[-1]:.1f}")

    return {
        "losses": losses,
        "theta": theta, "dst": dst, "n_dst": n_dst,
        "pat_ids": pat_ids, "pat_names": pat_names,
        "K": K, "P": P, "phenotypes": phenotypes,
        "hierarchical": hierarchical,
    }


def sample_posterior(svi_result, n_samples=2000):
    """Draw posterior samples from the learned guide."""
    args = (svi_result["theta"], svi_result["dst"], svi_result["n_dst"],
            svi_result["pat_ids"], svi_result["K"], svi_result["P"],
            svi_result["hierarchical"])

    T_samples, kappa_samples, T_patient_samples = [], [], []

    for _ in range(n_samples):
        trace = pyro.poutine.trace(transition_guide).get_trace(*args)
        if svi_result["hierarchical"] and svi_result["P"] > 1:
            T_samples.append(trace.nodes["T_global"]["value"].detach().cpu().numpy())
            kappa_samples.append(trace.nodes["kappa"]["value"].item())
            T_patient_samples.append(trace.nodes["T_patient"]["value"].detach().cpu().numpy())
        else:
            T_samples.append(trace.nodes["T"]["value"].detach().cpu().numpy())

    result = {**svi_result}
    T_arr = np.array(T_samples)
    result["T_mean"] = T_arr.mean(axis=0)
    result["T_std"] = T_arr.std(axis=0)
    result["T_samples"] = T_arr

    if kappa_samples:
        result["kappa_samples"] = np.array(kappa_samples)
        result["kappa_mean"] = np.mean(kappa_samples)
        T_pat = np.array(T_patient_samples)
        result["T_patient_mean"] = T_pat.mean(axis=0)
        result["T_patient_std"] = T_pat.std(axis=0)

    return result


def run_inference(adata, t1, t2, lineage="CD8", temporal=True,
                  n_steps=3000, lr=0.01, hierarchical=True, n_samples=2000,
                  verbose=True, **kwargs):
    """Full pipeline: extract data → run SVI → sample posterior."""
    if verbose:
        print(f"=== {lineage} {t1} → {t2} ===")

    if temporal:
        data, phenotypes = extract_temporal_transitions(adata, t1, t2, lineage=lineage, **kwargs)
    else:
        data, phenotypes = extract_transitions(adata, t1, t2, lineage=lineage, **kwargs)

    if len(data) == 0:
        print("  No shared clones found!")
        return None, data, phenotypes

    if verbose:
        summary(data, phenotypes)

    svi_result = run_svi(data, phenotypes, n_steps=n_steps, lr=lr,
                         hierarchical=hierarchical, verbose=verbose)
    result = sample_posterior(svi_result, n_samples=n_samples)
    return result, data, phenotypes


def credible_transitions(result, threshold=0.1, ci=0.9):
    """Extract transitions where the lower bound of the CI exceeds threshold."""
    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q
    phenotypes = result["phenotypes"]
    T = result["T_samples"]
    K = len(phenotypes)

    records = []
    for i in range(K):
        for j in range(K):
            samples = T[:, i, j]
            lo, hi = np.quantile(samples, [lower_q, upper_q])
            records.append({
                "source": phenotypes[i], "destination": phenotypes[j],
                "mean": samples.mean(), "std": samples.std(),
                f"ci_{ci:.0%}_lo": lo, f"ci_{ci:.0%}_hi": hi,
                "credible": lo > threshold,
            })

    df = pd.DataFrame(records)
    return df.sort_values("mean", ascending=False)


def compare_directions(adata, t1, t2, lineage="CD8", **kwargs):
    """Run inference in both directions, compute asymmetry matrix."""
    res_fwd, _, _ = run_inference(adata, t1, t2, lineage=lineage, **kwargs)
    res_rev, _, _ = run_inference(adata, t2, t1, lineage=lineage, **kwargs)

    if res_fwd is None or res_rev is None:
        print("Cannot compare — missing data in one direction")
        return None

    asym = res_fwd["T_mean"] - res_rev["T_mean"].T
    return {"forward": res_fwd, "reverse": res_rev, "asymmetry": asym}
