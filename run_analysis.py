# Save as: run_analysis.py
"""
Bayesian Compartmental Immune Trafficking Analysis

Usage:
    python run_analysis.py --tcell tcells.h5ad --myeloid myeloid.h5ad --output results/
"""

import argparse
import os
import json
import pickle
import jax.numpy as jnp

from src.trafficking.data import load_trafficking_data, TraffickingData
from src.trafficking.models import model_composition_simple, model_with_tcr, model_full_relaxed
from src.trafficking.inference import fit_svi, get_posterior_samples
from src.trafficking.analysis import (
    compute_compartment_covariance,
    compute_posterior_predictive,
    summarize_correlations,
    extract_trafficking_rates,
    extract_coupling_effects
)
from src.trafficking.viz import (
    plot_correlation_vs_clonesharing,
    plot_coupling_heatmap,
    plot_posterior_predictive,
    create_manuscript_figure
)


def summarize_data(data: TraffickingData):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Patients: {len(data.patients)}")
    print(f"Timepoints: {list(data.timepoints)}")
    print(f"Compartments: {list(data.compartments)}")
    print(f"T cell phenotypes: {list(data.tcell_phenotypes)}")
    print(f"Myeloid phenotypes: {list(data.myeloid_phenotypes)}")
    print(f"\nObservation coverage: {data.obs_mask.sum()} / {data.obs_mask.size} "
          f"({100*data.obs_mask.mean():.1f}%)")
    print(f"Total T cells: {data.tcell_counts.sum():,}")
    print(f"Total myeloid: {data.myeloid_counts.sum():,}")
    
    for i, c in enumerate(data.compartments):
        t_n = data.tcell_counts[:, :, i, :].sum()
        m_n = data.myeloid_counts[:, :, i, :].sum()
        obs = data.obs_mask[:, :, i].sum()
        print(f"  {c}: {obs} samples, {t_n:,} T cells, {m_n:,} myeloid")


def run_pipeline(
    tcell_path: str,
    myeloid_path: str,
    output_dir: str,
    n_steps: int = 5000,
    n_latent: int = 3
):
    """Run complete analysis pipeline."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    data = load_trafficking_data(tcell_path, myeloid_path)
    summarize_data(data)
    
    # Prepare data kwargs
    data_kwargs_simple = {
        'tcell_counts': jnp.array(data.tcell_counts),
        'myeloid_counts': jnp.array(data.myeloid_counts),
    }
    
    data_kwargs_tcr = {
        'tcell_counts': jnp.array(data.tcell_counts),
        'myeloid_counts': jnp.array(data.myeloid_counts),
        'clone_sharing': jnp.array(data.clone_sharing),
    }
    
    # Phase 1: Simple composition model
    print("\n" + "="*60)
    print("PHASE 1: Composition model")
    print("="*60)
    svi_simple, guide_simple = fit_svi(model_composition_simple, data_kwargs_simple, n_steps=n_steps)
    samples_simple = get_posterior_samples(model_composition_simple, guide_simple, 
                                           svi_simple.params, data_kwargs_simple)
    
    # Analyze cross-compartment correlations
    t_corrs, m_corrs = summarize_correlations(data, samples_simple)
    
    print("\nT cell compartment correlations:")
    for pheno, corrs in t_corrs.items():
        print(f"  {pheno}: Blood-Tumor = {corrs['Blood-Tumor']:.2f}")
    
    # Phase 2: TCR model
    print("\n" + "="*60)
    print("PHASE 2: TCR trafficking model")
    print("="*60)
    svi_tcr, guide_tcr = fit_svi(model_with_tcr, data_kwargs_tcr, n_steps=n_steps)
    samples_tcr = get_posterior_samples(model_with_tcr, guide_tcr, 
                                        svi_tcr.params, data_kwargs_tcr)
    
    trafficking_rates = extract_trafficking_rates(samples_tcr, data)
    
    print("\nTrafficking rates (Blood→Tumor):")
    for pheno, rates in trafficking_rates.items():
        print(f"  {pheno}: {rates['Blood→Tumor']['mean']:.1%} ± {rates['Blood→Tumor']['std']:.1%}")
    
    # Phase 3: Full model with T-myeloid coupling
    print("\n" + "="*60)
    print("PHASE 3: Full model with T-myeloid coupling")
    print("="*60)
    svi_full, guide_full = fit_svi(model_full_relaxed, data_kwargs_tcr, n_steps=n_steps)
    samples_full = get_posterior_samples(model_full_relaxed, guide_full,
                                         svi_full.params, data_kwargs_tcr)
    
    coupling_effects = extract_coupling_effects(samples_full, data)
    
    print("\nSignificant T→Myeloid associations (|z| > 1.5):")
    for e in coupling_effects[:10]:
        if e['significant']:
            print(f"  {e['compartment']}: {e['t_pheno']} → {e['m_pheno'][:25]}: γ={e['mean']:.2f}, z={e['z']:.2f}")
    
    # Phase 4: Posterior predictive
    print("\n" + "="*60)
    print("PHASE 4: Posterior predictive analysis")
    print("="*60)
    t_pred_corrs, m_pred_corrs, props = compute_posterior_predictive(samples_full, data, 0, 2)
    
    print("\nBlood→Tumor predictive correlations (T cells):")
    for tc in sorted(t_pred_corrs, key=lambda x: -abs(x['mean']))[:5]:
        print(f"  {tc['pheno']}: r = {tc['mean']:.2f} ({tc['ci'][0]:.2f}, {tc['ci'][1]:.2f})")
    
    # Generate figures
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    
    fig1 = plot_correlation_vs_clonesharing(data, samples_full, 
                                            save_path=os.path.join(output_dir, 'correlation_vs_clonesharing.png'))
    print("  Saved: correlation_vs_clonesharing.png")
    
    fig2 = plot_coupling_heatmap(samples_full, data,
                                 save_path=os.path.join(output_dir, 'tcell_myeloid_coupling.png'))
    print("  Saved: tcell_myeloid_coupling.png")
    
    fig3 = plot_posterior_predictive(data, samples_full,
                                     save_path=os.path.join(output_dir, 'posterior_predictive_scatter.png'))
    print("  Saved: posterior_predictive_scatter.png")
    
    fig4 = create_manuscript_figure(data, samples_full, samples_tcr,
                                    save_path=os.path.join(output_dir, 'manuscript_figure.png'))
    print("  Saved: manuscript_figure.png")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    results = {
        't_correlations': t_corrs,
        'm_correlations': m_corrs,
        'trafficking_rates': trafficking_rates,
        'coupling_effects': [e for e in coupling_effects if e['significant']],
        't_predictive_correlations': [{'pheno': t['pheno'], 'mean': t['mean'], 
                                       'ci_low': t['ci'][0], 'ci_high': t['ci'][1]} 
                                      for t in t_pred_corrs],
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Saved: results.json")
    
    # Save samples for downstream analysis
    with open(os.path.join(output_dir, 'samples_full.pkl'), 'wb') as f:
        pickle.dump({k: np.array(v) for k, v in samples_full.items()}, f)
    print("  Saved: samples_full.pkl")
    
    with open(os.path.join(output_dir, 'samples_tcr.pkl'), 'wb') as f:
        pickle.dump({k: np.array(v) for k, v in samples_tcr.items()}, f)
    print("  Saved: samples_tcr.pkl")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    return data, samples_full, samples_tcr, results


def main():
    parser = argparse.ArgumentParser(description='Bayesian Compartmental Immune Trafficking Analysis')
    parser.add_argument('--tcell', required=True, help='Path to T cell h5ad file')
    parser.add_argument('--myeloid', required=True, help='Path to myeloid h5ad file')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--n-steps', type=int, default=5000, help='Number of SVI steps')
    
    args = parser.parse_args()
    
    run_pipeline(
        tcell_path=args.tcell,
        myeloid_path=args.myeloid,
        output_dir=args.output,
        n_steps=args.n_steps
    )


if __name__ == '__main__':
    main()