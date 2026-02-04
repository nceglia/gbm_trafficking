# Save as: src/trafficking/models.py
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def model_composition_simple(tcell_counts, myeloid_counts):
    n_p, n_t, n_c, n_kt = tcell_counts.shape
    n_km = myeloid_counts.shape[-1]
    
    mu_t = numpyro.sample("mu_t", dist.Normal(0, 2).expand([n_c, n_kt - 1]).to_event(2))
    mu_m = numpyro.sample("mu_m", dist.Normal(0, 2).expand([n_c, n_km - 1]).to_event(2))
    sigma_t = numpyro.sample("sigma_t", dist.HalfNormal(1).expand([n_c]).to_event(1))
    sigma_m = numpyro.sample("sigma_m", dist.HalfNormal(1).expand([n_c]).to_event(1))
    phi_t = numpyro.sample("phi_t", dist.LogNormal(3, 1).expand([n_c]).to_event(1))
    phi_m = numpyro.sample("phi_m", dist.LogNormal(3, 1).expand([n_c]).to_event(1))
    
    with numpyro.plate("timepoints", n_t, dim=-1):
        with numpyro.plate("patients", n_p, dim=-2):
            for c in range(n_c):
                eta_t = numpyro.sample(f"eta_t_c{c}", 
                    dist.Normal(mu_t[c], sigma_t[c]).expand([n_kt - 1]).to_event(1))
                eta_t_full = jnp.concatenate([eta_t, jnp.zeros((*eta_t.shape[:-1], 1))], axis=-1)
                pi_t = jax.nn.softmax(eta_t_full, axis=-1)
                numpyro.sample(f"tcell_c{c}",
                    dist.DirichletMultinomial(phi_t[c] * pi_t + 1e-6, total_count=tcell_counts[:, :, c, :].sum(-1)),
                    obs=tcell_counts[:, :, c, :])
                
                eta_m = numpyro.sample(f"eta_m_c{c}",
                    dist.Normal(mu_m[c], sigma_m[c]).expand([n_km - 1]).to_event(1))
                eta_m_full = jnp.concatenate([eta_m, jnp.zeros((*eta_m.shape[:-1], 1))], axis=-1)
                pi_m = jax.nn.softmax(eta_m_full, axis=-1)
                numpyro.sample(f"myeloid_c{c}",
                    dist.DirichletMultinomial(phi_m[c] * pi_m + 1e-6, total_count=myeloid_counts[:, :, c, :].sum(-1)),
                    obs=myeloid_counts[:, :, c, :])


def model_with_tcr(tcell_counts, myeloid_counts, clone_sharing):
    n_p, n_t, n_c, n_kt = tcell_counts.shape
    n_km = myeloid_counts.shape[-1]
    
    mu_t = numpyro.sample("mu_t", dist.Normal(0, 2).expand([n_c, n_kt - 1]).to_event(2))
    mu_m = numpyro.sample("mu_m", dist.Normal(0, 2).expand([n_c, n_km - 1]).to_event(2))
    sigma_t = numpyro.sample("sigma_t", dist.HalfNormal(1).expand([n_c]).to_event(1))
    sigma_m = numpyro.sample("sigma_m", dist.HalfNormal(1).expand([n_c]).to_event(1))
    phi_t = numpyro.sample("phi_t", dist.LogNormal(3, 1).expand([n_c]).to_event(1))
    phi_m = numpyro.sample("phi_m", dist.LogNormal(3, 1).expand([n_c]).to_event(1))
    
    tau = numpyro.sample("tau", dist.Beta(1, 5).expand([n_kt, n_c, n_c]).to_event(3))
    kappa = numpyro.sample("kappa", dist.LogNormal(2, 1))
    
    with numpyro.plate("timepoints", n_t, dim=-1):
        with numpyro.plate("patients", n_p, dim=-2):
            for c in range(n_c):
                eta_t = numpyro.sample(f"eta_t_c{c}", 
                    dist.Normal(mu_t[c], sigma_t[c]).expand([n_kt - 1]).to_event(1))
                eta_t_full = jnp.concatenate([eta_t, jnp.zeros((*eta_t.shape[:-1], 1))], axis=-1)
                pi_t = jax.nn.softmax(eta_t_full, axis=-1)
                numpyro.sample(f"tcell_c{c}",
                    dist.DirichletMultinomial(phi_t[c] * pi_t + 1e-6, total_count=tcell_counts[:, :, c, :].sum(-1)),
                    obs=tcell_counts[:, :, c, :])
                
                eta_m = numpyro.sample(f"eta_m_c{c}",
                    dist.Normal(mu_m[c], sigma_m[c]).expand([n_km - 1]).to_event(1))
                eta_m_full = jnp.concatenate([eta_m, jnp.zeros((*eta_m.shape[:-1], 1))], axis=-1)
                pi_m = jax.nn.softmax(eta_m_full, axis=-1)
                numpyro.sample(f"myeloid_c{c}",
                    dist.DirichletMultinomial(phi_m[c] * pi_m + 1e-6, total_count=myeloid_counts[:, :, c, :].sum(-1)),
                    obs=myeloid_counts[:, :, c, :])
    
    for c_to in range(n_c):
        for c_from in range(n_c):
            if c_from != c_to:
                obs_sharing = clone_sharing[:, :, c_to, :, c_from].mean(axis=(0, 1))
                obs_sharing = jnp.clip(obs_sharing, 0.001, 0.999)
                numpyro.sample(f"sharing_{c_from}_to_{c_to}",
                    dist.Beta(tau[:, c_from, c_to] * kappa + 0.5, 
                             (1 - tau[:, c_from, c_to]) * kappa + 0.5).to_event(1),
                    obs=obs_sharing)


def model_full_relaxed(tcell_counts, myeloid_counts, clone_sharing):
    n_p, n_t, n_c, n_kt = tcell_counts.shape
    n_km = myeloid_counts.shape[-1]
    
    mu_t = numpyro.sample("mu_t", dist.Normal(0, 2).expand([n_c, n_kt - 1]).to_event(2))
    mu_m = numpyro.sample("mu_m", dist.Normal(0, 2).expand([n_c, n_km - 1]).to_event(2))
    sigma_t = numpyro.sample("sigma_t", dist.HalfNormal(1).expand([n_c]).to_event(1))
    sigma_m = numpyro.sample("sigma_m", dist.HalfNormal(1).expand([n_c]).to_event(1))
    phi_t = numpyro.sample("phi_t", dist.LogNormal(3, 1).expand([n_c]).to_event(1))
    phi_m = numpyro.sample("phi_m", dist.LogNormal(3, 1).expand([n_c]).to_event(1))
    
    gamma = numpyro.sample("gamma", dist.Normal(0, 1).expand([n_c, n_kt, n_km - 1]).to_event(3))
    
    tau = numpyro.sample("tau", dist.Beta(1, 5).expand([n_kt, n_c, n_c]).to_event(3))
    kappa = numpyro.sample("kappa", dist.LogNormal(2, 1))
    
    with numpyro.plate("timepoints", n_t, dim=-1):
        with numpyro.plate("patients", n_p, dim=-2):
            for c in range(n_c):
                eta_t = numpyro.sample(f"eta_t_c{c}", 
                    dist.Normal(mu_t[c], sigma_t[c]).expand([n_kt - 1]).to_event(1))
                eta_t_full = jnp.concatenate([eta_t, jnp.zeros((*eta_t.shape[:-1], 1))], axis=-1)
                pi_t = jax.nn.softmax(eta_t_full, axis=-1)
                
                numpyro.sample(f"tcell_c{c}",
                    dist.DirichletMultinomial(phi_t[c] * pi_t + 1e-6, total_count=tcell_counts[:, :, c, :].sum(-1)),
                    obs=tcell_counts[:, :, c, :])
                
                t_effect = pi_t @ gamma[c]
                eta_m = numpyro.sample(f"eta_m_c{c}",
                    dist.Normal(mu_m[c] + t_effect, sigma_m[c]).to_event(1))
                eta_m_full = jnp.concatenate([eta_m, jnp.zeros((*eta_m.shape[:-1], 1))], axis=-1)
                pi_m = jax.nn.softmax(eta_m_full, axis=-1)
                
                numpyro.sample(f"myeloid_c{c}",
                    dist.DirichletMultinomial(phi_m[c] * pi_m + 1e-6, total_count=myeloid_counts[:, :, c, :].sum(-1)),
                    obs=myeloid_counts[:, :, c, :])
    
    for c_to in range(n_c):
        for c_from in range(n_c):
            if c_from != c_to:
                obs_sharing = clone_sharing[:, :, c_to, :, c_from].mean(axis=(0, 1))
                obs_sharing = jnp.clip(obs_sharing, 0.001, 0.999)
                numpyro.sample(f"sharing_{c_from}_to_{c_to}",
                    dist.Beta(tau[:, c_from, c_to] * kappa + 0.5, 
                             (1 - tau[:, c_from, c_to]) * kappa + 0.5).to_event(1),
                    obs=obs_sharing)