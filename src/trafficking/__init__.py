# Save as: src/trafficking/__init__.py
"""
Bayesian Compartmental Immune Trafficking Model

A hierarchical Bayesian framework for modeling immune phenotype distributions 
and trafficking across blood, CSF, and tumor in glioblastoma.
"""

from .data import TraffickingData, load_trafficking_data, compute_clone_sharing
from .models import model_composition_simple, model_with_tcr, model_full_relaxed
from .inference import fit_svi, get_posterior_samples
from .analysis import (
    compute_compartment_covariance,
    compute_posterior_predictive,
    summarize_correlations
)
from .viz import (
    plot_correlation_vs_clonesharing,
    plot_coupling_heatmap,
    plot_posterior_predictive,
    create_manuscript_figure
)