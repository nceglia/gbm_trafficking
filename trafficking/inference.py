import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import numpy as np
import pandas as pd

from .data import (extract_transitions, extract_temporal_transitions,
                   prepare_tensors, summary, compute_dest_phenotype_fracs)
from .model import (transition_model, transition_guide,
                    ALPHA_FLOOR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reclamp_params(K, P, hierarchical, dest_fracs):
    """Repair NaN / non-positive entries in the variational concentrations.

    A single bad SVI step can put NaNs or values ≤ 0 into the Dirichlet
    concentration parameters. Resetting them to a safe positive value
    (the prior centre) lets the optimizer recover without throwing away
    the rest of the fit.
    """
    if hierarchical and P > 1:
        if "T_global_conc" in pyro.get_param_store():
            tgc = pyro.param("T_global_conc").detach()
            bad = ~torch.isfinite(tgc) | (tgc <= 0)
            if bad.any():
                if dest_fracs is not None:
                    floor = dest_fracs.to(device).clamp(min=0.01)
                    floor = floor / floor.sum()
                    row = (floor * 5.0).clamp(min=ALPHA_FLOOR)
                    repl = row.unsqueeze(0).repeat(K, 1).contiguous()
                else:
                    repl = torch.full((K, K), 5.0, device=device)
                tgc[bad] = repl[bad].to(tgc.dtype)
                pyro.param("T_global_conc").data.copy_(tgc.clamp(min=ALPHA_FLOOR))
        if "T_patient_conc" in pyro.get_param_store():
            tpc = pyro.param("T_patient_conc").detach()
            bad = ~torch.isfinite(tpc) | (tpc <= 0)
            if bad.any():
                tpc[bad] = 5.0
                pyro.param("T_patient_conc").data.copy_(tpc.clamp(min=ALPHA_FLOOR))
        if "kappa_loc" in pyro.get_param_store():
            kl = pyro.param("kappa_loc").detach()
            if not torch.isfinite(kl) or kl <= 0:
                pyro.param("kappa_loc").data.fill_(5.0)
    else:
        if "T_conc" in pyro.get_param_store():
            tc = pyro.param("T_conc").detach()
            bad = ~torch.isfinite(tc) | (tc <= 0)
            if bad.any():
                tc[bad] = 5.0
                pyro.param("T_conc").data.copy_(tc.clamp(min=ALPHA_FLOOR))


def run_svi(data, phenotypes, n_steps=3000, lr=0.01, hierarchical=True,
            verbose=True, dest_fracs=None, clip_norm=1.0):
    """Run stochastic variational inference on extracted transition data.

    ``dest_fracs`` (optional, torch tensor of shape (K,)) is threaded into
    the model/guide as the prior centre. ``None`` reproduces the flat
    Dirichlet(1) behaviour. ``clip_norm`` is the per-step gradient L2-norm
    clip; the default (1.0) is much tighter than ClippedAdam's default of
    10.0 and is what keeps small Dirichlet concentrations from flipping
    sign in a single optimizer step.
    """
    theta, dst, n_dst, pat_ids, pat_names = prepare_tensors(data)
    K = len(phenotypes)
    P = len(pat_names)

    pyro.clear_param_store()
    optimizer = ClippedAdam(
        {"lr": lr, "betas": (0.9, 0.999), "clip_norm": clip_norm})
    svi = SVI(transition_model, transition_guide, optimizer, loss=Trace_ELBO())

    losses = []
    nan_count = 0
    for step in range(n_steps):
        try:
            loss = svi.step(theta, dst, n_dst, pat_ids, K, P, hierarchical,
                            dest_fracs)
        except (ValueError, RuntimeError) as e:
            # Most common: a tiny Dirichlet concentration flipped to NaN under
            # gradient pressure. Drop the bad step and keep going — if it
            # repeats we abort below.
            nan_count += 1
            if verbose and nan_count <= 3:
                print(f"  Step {step}: SVI step failed ({type(e).__name__}); "
                      f"re-clamping params and continuing")
            _reclamp_params(K, P, hierarchical, dest_fracs)
            if nan_count > 10:
                raise RuntimeError(
                    f"SVI diverged: {nan_count} consecutive bad steps. "
                    f"Last error: {e}")
            continue
        if not np.isfinite(loss):
            nan_count += 1
            if verbose and nan_count <= 3:
                print(f"  Step {step}: non-finite ELBO ({loss}); re-clamping")
            _reclamp_params(K, P, hierarchical, dest_fracs)
            if nan_count > 10:
                raise RuntimeError("SVI diverged: non-finite ELBO persists.")
            continue
        nan_count = 0
        losses.append(loss)
        if verbose and step % 500 == 0:
            print(f"  Step {step}: ELBO = {loss:.1f}")

    if not losses:
        raise RuntimeError("SVI produced no finite ELBO values.")

    if verbose:
        print(f"  Final ELBO: {losses[-1]:.1f}")

    return {
        "losses": losses,
        "theta": theta, "dst": dst, "n_dst": n_dst,
        "pat_ids": pat_ids, "pat_names": pat_names,
        "K": K, "P": P, "phenotypes": phenotypes,
        "hierarchical": hierarchical,
        "dest_fracs": dest_fracs,
    }


def sample_posterior(svi_result, n_samples=2000):
    """Draw posterior samples from the learned guide."""
    args = (svi_result["theta"], svi_result["dst"], svi_result["n_dst"],
            svi_result["pat_ids"], svi_result["K"], svi_result["P"],
            svi_result["hierarchical"], svi_result.get("dest_fracs"))

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
                  verbose=True, dest_fracs=None, **kwargs):
    """Full pipeline: extract data → run SVI → sample posterior.

    By default, the Dirichlet prior on ``T_global`` is centred on the
    empirical phenotype composition at the destination tissue ``t2``
    (filtered by ``lineage``). Pass ``dest_fracs`` explicitly to override,
    or pass an all-ones tensor to recover the flat prior.
    """
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

    if dest_fracs is None:
        dest_fracs_series = compute_dest_phenotype_fracs(adata, t2, lineage)
        vec = np.array([float(dest_fracs_series.get(p, 1e-6))
                        for p in phenotypes], dtype=np.float32)
        if vec.sum() <= 0:
            vec = np.full_like(vec, 1.0 / len(vec))
        else:
            vec = vec / vec.sum()
        dest_fracs_tensor = torch.tensor(vec, dtype=torch.float, device=device)
        if verbose:
            print(f"  Dest phenotype fracs @ {t2} ({lineage}) — prior centre:")
            for p, f in zip(phenotypes, vec):
                print(f"    {p:>14s}: {f:.3f}")
    else:
        if not isinstance(dest_fracs, torch.Tensor):
            dest_fracs_tensor = torch.tensor(
                np.asarray(dest_fracs, dtype=np.float32),
                dtype=torch.float, device=device)
        else:
            dest_fracs_tensor = dest_fracs.to(device=device, dtype=torch.float)
        dest_fracs_tensor = dest_fracs_tensor / dest_fracs_tensor.sum()

    svi_result = run_svi(data, phenotypes, n_steps=n_steps, lr=lr,
                         hierarchical=hierarchical, verbose=verbose,
                         dest_fracs=dest_fracs_tensor)
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
