import torch
import pyro
import pyro.distributions as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transition_model(theta, dst, n_dst, pat_ids, K, P, hierarchical=True):
    """Bayesian transition model.

    Generative process:
      T_global[k,:] ~ Dirichlet(α)           — global transition row per source state
      T_patient[p,k,:] ~ Dirichlet(κ·T_global[k,:])  — patient-specific (if hierarchical)
      For each clone c:
        π_c = θ_c @ T[patient_c]             — expected destination distribution
        dst_c ~ Multinomial(n_c, π_c)         — observed destination counts
    """
    alpha = torch.ones(K, device=device)

    if hierarchical and P > 1:
        kappa = pyro.sample("kappa", dist.Gamma(5.0, 1.0))
        with pyro.plate("src_state", K, dim=-1):
            T_global = pyro.sample("T_global", dist.Dirichlet(alpha))
        with pyro.plate("patient", P, dim=-2):
            with pyro.plate("src_state", K, dim=-1):
                T_patient = pyro.sample("T_patient", dist.Dirichlet(kappa * T_global))
        T_clone = T_patient[pat_ids]
        expected = torch.einsum("nk,nkj->nj", theta, T_clone)
    else:
        with pyro.plate("src_state", K, dim=-1):
            T = pyro.sample("T", dist.Dirichlet(alpha))
        expected = theta @ T

    expected = expected / expected.sum(dim=1, keepdim=True).clamp(min=1e-8)
    with pyro.plate("clones", theta.shape[0]):
        pyro.sample("obs", dist.Multinomial(total_count=n_dst, probs=expected), obs=dst)


def transition_guide(theta, dst, n_dst, pat_ids, K, P, hierarchical=True):
    """Variational guide — learns Dirichlet concentrations for all latent variables."""
    if hierarchical and P > 1:
        kappa_loc = pyro.param("kappa_loc", torch.tensor(5.0, device=device),
                               constraint=torch.distributions.constraints.positive)
        pyro.sample("kappa", dist.Delta(kappa_loc))

        T_global_conc = pyro.param("T_global_conc",
                                    torch.ones(K, K, device=device) * 5.0,
                                    constraint=torch.distributions.constraints.positive)
        with pyro.plate("src_state", K, dim=-1):
            pyro.sample("T_global", dist.Dirichlet(T_global_conc))

        T_patient_conc = pyro.param("T_patient_conc",
                                     torch.ones(P, K, K, device=device) * 5.0,
                                     constraint=torch.distributions.constraints.positive)
        with pyro.plate("patient", P, dim=-2):
            with pyro.plate("src_state", K, dim=-1):
                pyro.sample("T_patient", dist.Dirichlet(T_patient_conc))
    else:
        T_conc = pyro.param("T_conc", torch.ones(K, K, device=device) * 5.0,
                            constraint=torch.distributions.constraints.positive)
        with pyro.plate("src_state", K, dim=-1):
            pyro.sample("T", dist.Dirichlet(T_conc))
