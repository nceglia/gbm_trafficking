# Save as: src/trafficking/inference.py
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, MCMC, NUTS, Predictive


def fit_svi(model, data_kwargs, n_steps=5000, lr=0.01, seed=0):
    """Fit model using SVI with AutoNormal guide."""
    guide = autoguide.AutoNormal(model)
    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    rng_key = jax.random.PRNGKey(seed)
    svi_result = svi.run(rng_key, n_steps, **data_kwargs, progress_bar=True)
    
    return svi_result, guide


def fit_mcmc(model, data_kwargs, n_warmup=500, n_samples=1000, n_chains=1, seed=0):
    """Fit model using MCMC with NUTS sampler."""
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
    mcmc.run(jax.random.PRNGKey(seed), **data_kwargs)
    return mcmc.get_samples()


def get_posterior_samples(model, guide, params, data_kwargs, n_samples=1000, seed=1):
    """Draw samples from the fitted guide."""
    predictive = Predictive(guide, params=params, num_samples=n_samples)
    rng_key = jax.random.PRNGKey(seed)
    return predictive(rng_key, **data_kwargs)