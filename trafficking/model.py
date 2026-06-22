import torch
import pyro
import pyro.distributions as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Numerical-stability floors for Dirichlet concentrations. The reparameterised
# Dirichlet gradient blows up when any component's concentration drops below
# ~0.3, which used to happen routinely on edges where one destination phenotype
# was essentially absent (e.g., TRM in CSF ≈ 0.4% → α ≈ 0.07).
ALPHA_FLOOR = 0.5         # min entry of the global Dirichlet prior on T_global
PATIENT_ALPHA_FLOOR = 0.1  # min entry of κ·T_global passed to the patient plate


def transition_model(theta, dst, n_dst, pat_ids, K, P, hierarchical=True,
                     dest_fracs=None):
    """Bayesian transition model.

    Generative process:
      T_global[k,:] ~ Dirichlet(α)           — global transition row per source state
      T_patient[p,k,:] ~ Dirichlet(κ·T_global[k,:])  — patient-specific (if hierarchical)
      For each clone c:
        π_c = θ_c @ T[patient_c]             — expected destination distribution
        dst_c ~ Multinomial(n_c, π_c)         — observed destination counts

    When ``dest_fracs`` is supplied (a tensor of shape (K,) summing to 1),
    the Dirichlet prior is recentered on the empirical marginal composition
    at the destination tissue: ``α = dest_fracs * K``. To keep the
    reparameterised gradient numerically stable, every entry of α is
    floored at ``ALPHA_FLOOR`` — for K=7 and a rare phenotype (dest frac
    < 1%) the unfloored α would be ~0.07, which routinely produced NaN
    in the variational concentrations.
    """
    if dest_fracs is not None:
        dest_floor = dest_fracs.to(device).clamp(min=0.01)
        dest_floor = dest_floor / dest_floor.sum()
        alpha = (dest_floor * float(K)).clamp(min=ALPHA_FLOOR)
    else:
        alpha = torch.ones(K, device=device)

    if hierarchical and P > 1:
        kappa = pyro.sample("kappa", dist.Gamma(5.0, 1.0))
        with pyro.plate("src_state", K, dim=-1):
            T_global = pyro.sample("T_global", dist.Dirichlet(alpha))
        # Floor κ·T_global so the patient Dirichlet alphas never approach 0
        # when T_global samples put near-zero mass on a rare phenotype.
        patient_alpha = (kappa * T_global).clamp(min=PATIENT_ALPHA_FLOOR)
        with pyro.plate("patient", P, dim=-2):
            with pyro.plate("src_state_patient", K, dim=-1):
                T_patient = pyro.sample("T_patient", dist.Dirichlet(patient_alpha))
        T_clone = T_patient[pat_ids]
        expected = torch.einsum("nk,nkj->nj", theta, T_clone)
    else:
        with pyro.plate("src_state", K, dim=-1):
            T = pyro.sample("T", dist.Dirichlet(alpha))
        expected = theta @ T

    expected = expected / expected.sum(dim=1, keepdim=True).clamp(min=1e-8)
    # Manual log-likelihood: sum of dst_counts * log(probs) per clone
    # Equivalent to inhomogeneous Multinomial but avoids torch limitation
    log_probs = torch.log(expected.clamp(min=1e-8))
    log_lik = (dst * log_probs).sum(dim=1)
    with pyro.plate("clones", theta.shape[0]):
        pyro.factor("obs", log_lik)


def transition_guide(theta, dst, n_dst, pat_ids, K, P, hierarchical=True,
                     dest_fracs=None):
    """Variational guide — learns Dirichlet concentrations for all latent variables.

    When ``dest_fracs`` is provided, the ``T_global_conc`` parameter is
    initialised so every row is centred on the empirical destination
    composition (× 5), matching the model prior, with each entry floored
    at ``ALPHA_FLOOR`` for stability.
    """
    if hierarchical and P > 1:
        kappa_loc = pyro.param("kappa_loc", torch.tensor(5.0, device=device),
                               constraint=torch.distributions.constraints.positive)
        pyro.sample("kappa", dist.Delta(kappa_loc))

        if dest_fracs is not None:
            dest_floor = dest_fracs.to(device).clamp(min=0.01)
            dest_floor = dest_floor / dest_floor.sum()
            row = (dest_floor * 5.0).clamp(min=ALPHA_FLOOR)
            T_global_conc_init = row.unsqueeze(0).repeat(K, 1).contiguous()
        else:
            T_global_conc_init = torch.ones(K, K, device=device) * 5.0
        T_global_conc = pyro.param("T_global_conc", T_global_conc_init,
                                    constraint=torch.distributions.constraints.positive)
        with pyro.plate("src_state", K, dim=-1):
            pyro.sample("T_global", dist.Dirichlet(T_global_conc))

        T_patient_conc = pyro.param("T_patient_conc",
                                     torch.ones(P, K, K, device=device) * 5.0,
                                     constraint=torch.distributions.constraints.positive)
        with pyro.plate("patient", P, dim=-2):
            with pyro.plate("src_state_patient", K, dim=-1):
                pyro.sample("T_patient", dist.Dirichlet(T_patient_conc))
    else:
        T_conc = pyro.param("T_conc", torch.ones(K, K, device=device) * 5.0,
                            constraint=torch.distributions.constraints.positive)
        with pyro.plate("src_state", K, dim=-1):
            pyro.sample("T", dist.Dirichlet(T_conc))