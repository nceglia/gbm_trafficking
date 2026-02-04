# Save as: src/trafficking/viz.py
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from typing import Dict, List
from .data import TraffickingData
from .analysis import compute_compartment_covariance


def plot_correlation_vs_clonesharing(
    data: TraffickingData, 
    samples: Dict,
    save_path: str = None
) -> plt.Figure:
    """Compare compositional correlation to TCR clone sharing."""
    n_p, n_t, n_c, n_kt = data.tcell_counts.shape
    cov_t = compute_compartment_covariance(samples, n_p, n_t, n_c, n_kt, 't')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    bt_corr = [float(cov_t[:, k, 0, 2].mean()) for k in range(n_kt - 1)]
    bt_share = [float(data.clone_sharing[:, :, 0, k, 2].mean()) for k in range(n_kt - 1)]
    
    ax = axes[0]
    ax.scatter(bt_share, bt_corr, s=100)
    for k, pheno in enumerate(data.tcell_phenotypes[:-1]):
        ax.annotate(pheno, (bt_share[k], bt_corr[k]), fontsize=8, ha='left')
    ax.set_xlabel('Clone sharing (Blood→Tumor)')
    ax.set_ylabel('Compositional correlation')
    ax.set_title('Blood-Tumor: Clone sharing vs Composition')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax = axes[1]
    corr_matrix = np.array(cov_t.mean(axis=0).mean(axis=0))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(data.compartments)
    ax.set_yticklabels(data.compartments)
    ax.set_title('Average T cell correlation')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_coupling_heatmap(
    samples: Dict, 
    data: TraffickingData, 
    threshold_z: float = 1.5,
    save_path: str = None
) -> plt.Figure:
    """Visualize T→Myeloid coupling matrix."""
    gamma = samples['gamma']
    gamma_mean = gamma.mean(axis=0)
    gamma_std = gamma.std(axis=0)
    z_scores = gamma_mean / (gamma_std + 1e-6)
    
    compartments = list(data.compartments)
    t_phenos = list(data.tcell_phenotypes)
    m_phenos = [p[:20] for p in data.myeloid_phenotypes[:-1]]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for c, (ax, comp) in enumerate(zip(axes, compartments)):
        im = ax.imshow(z_scores[c], cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
        
        for i in range(len(t_phenos)):
            for j in range(len(m_phenos)):
                if abs(z_scores[c, i, j]) > threshold_z:
                    ax.scatter(j, i, marker='*', c='black', s=100)
        
        ax.set_xticks(range(len(m_phenos)))
        ax.set_xticklabels(m_phenos, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(t_phenos)))
        ax.set_yticklabels(t_phenos, fontsize=9)
        ax.set_title(f'{comp}\n(* = |z| > {threshold_z})', fontsize=11)
        ax.set_xlabel('Myeloid phenotype')
        if c == 0:
            ax.set_ylabel('T cell phenotype')
    
    plt.colorbar(im, ax=axes, label='z-score', shrink=0.8)
    plt.suptitle('T Cell → Myeloid Coupling by Compartment', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_posterior_predictive(
    data: TraffickingData, 
    samples: Dict, 
    source_comp: int = 0, 
    target_comp: int = 2,
    phenotype_indices: List[int] = None,
    save_path: str = None
) -> plt.Figure:
    """Scatter plots of source vs target compositions with uncertainty."""
    compartments = list(data.compartments)
    src_name = compartments[source_comp]
    tgt_name = compartments[target_comp]
    
    def get_mean_props(samples, comp, lineage='t'):
        eta = samples[f'eta_{lineage}_c{comp}']
        eta_full = jnp.concatenate([eta, jnp.zeros((*eta.shape[:-1], 1))], axis=-1)
        pi = jax.nn.softmax(eta_full, axis=-1)
        return np.array(pi.mean(axis=0)), np.array(pi.std(axis=0))
    
    pi_src_mean, pi_src_std = get_mean_props(samples, source_comp, 't')
    pi_tgt_mean, pi_tgt_std = get_mean_props(samples, target_comp, 't')
    
    if phenotype_indices is None:
        phenotype_indices = [0, 3, 4, 6]  # Cytotoxic, Memory, Naive, Teff
    
    n_phenos = len(phenotype_indices)
    n_cols = min(n_phenos, 2)
    n_rows = (n_phenos + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_phenos == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for ax, k in zip(axes, phenotype_indices):
        pheno = data.tcell_phenotypes[k]
        
        src_vals = pi_src_mean[:, :, k].flatten()
        tgt_vals = pi_tgt_mean[:, :, k].flatten()
        src_err = pi_src_std[:, :, k].flatten()
        tgt_err = pi_tgt_std[:, :, k].flatten()
        
        ax.errorbar(src_vals, tgt_vals, xerr=src_err, yerr=tgt_err, 
                   fmt='o', alpha=0.6, capsize=2, markersize=6)
        
        z = np.polyfit(src_vals, tgt_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(src_vals.min(), src_vals.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7)
        
        r = np.corrcoef(src_vals, tgt_vals)[0, 1]
        ax.set_xlabel(f'{src_name} proportion')
        ax.set_ylabel(f'{tgt_name} proportion')
        ax.set_title(f'{pheno}\nr = {r:.2f}')
        
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k:', alpha=0.3)
    
    plt.suptitle(f'T Cell Proportions: {src_name} vs {tgt_name}', fontsize=13)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_manuscript_figure(
    data: TraffickingData, 
    samples: Dict, 
    samples_tcr: Dict,
    save_path: str = None
) -> plt.Figure:
    """Create multi-panel figure for manuscript."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel A: Compartment correlation heatmap
    ax1 = fig.add_subplot(2, 3, 1)
    n_p, n_t, n_c, n_kt = data.tcell_counts.shape
    cov_t = compute_compartment_covariance(samples, n_p, n_t, n_c, n_kt, 't')
    avg_corr = np.array(cov_t.mean(axis=(0, 1)))
    im1 = ax1.imshow(avg_corr, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.set_xticklabels(data.compartments)
    ax1.set_yticklabels(data.compartments)
    ax1.set_title('A. T cell compartment\ncorrelation', fontsize=11)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Panel B: Trafficking rates
    ax2 = fig.add_subplot(2, 3, 2)
    tau = samples_tcr['tau']
    tau_mean = np.array(tau.mean(axis=0))
    phenos = [p[:8] for p in data.tcell_phenotypes]
    x = np.arange(len(phenos))
    width = 0.35
    ax2.bar(x - width/2, tau_mean[:, 0, 2], width, label='Blood→Tumor', color='#E63946')
    ax2.bar(x + width/2, tau_mean[:, 2, 0], width, label='Tumor→Blood', color='#457B9D')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phenos, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Trafficking rate (τ)')
    ax2.set_title('B. TCR-informed\ntrafficking rates', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 0.25)
    
    # Panel C: Blood-Tumor scatter for Cytotoxic
    ax3 = fig.add_subplot(2, 3, 3)
    eta_src = samples[f'eta_t_c0']
    eta_tgt = samples[f'eta_t_c2']
    
    def to_props(eta):
        eta_full = jnp.concatenate([eta, jnp.zeros((*eta.shape[:-1], 1))], axis=-1)
        return jax.nn.softmax(eta_full, axis=-1)
    
    pi_src = np.array(to_props(eta_src).mean(axis=0))[:, :, 0].flatten()
    pi_tgt = np.array(to_props(eta_tgt).mean(axis=0))[:, :, 0].flatten()
    ax3.scatter(pi_src, pi_tgt, alpha=0.6, s=50)
    z = np.polyfit(pi_src, pi_tgt, 1)
    p = np.poly1d(z)
    x_line = np.linspace(pi_src.min(), pi_src.max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', alpha=0.7)
    ax3.plot([0, 0.8], [0, 0.8], 'k:', alpha=0.3)
    r = np.corrcoef(pi_src, pi_tgt)[0, 1]
    ax3.set_xlabel('Blood proportion')
    ax3.set_ylabel('Tumor proportion')
    ax3.set_title(f'C. Cytotoxic T cells\nr = {r:.2f}', fontsize=11)
    
    # Panel D: T-Myeloid coupling heatmap (Tumor)
    ax4 = fig.add_subplot(2, 3, 4)
    gamma = samples['gamma']
    gamma_mean = np.array(gamma.mean(axis=0))
    gamma_std = np.array(gamma.std(axis=0))
    z_scores = gamma_mean / (gamma_std + 1e-6)
    im4 = ax4.imshow(z_scores[2], cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
    ax4.set_xticks(range(len(data.myeloid_phenotypes) - 1))
    ax4.set_xticklabels([p[:12] for p in data.myeloid_phenotypes[:-1]], rotation=45, ha='right', fontsize=7)
    ax4.set_yticks(range(len(data.tcell_phenotypes)))
    ax4.set_yticklabels(data.tcell_phenotypes, fontsize=8)
    ax4.set_title('D. T→Myeloid coupling\n(Tumor)', fontsize=11)
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='z-score')
    
    # Panel E: Clone sharing vs correlation
    ax5 = fig.add_subplot(2, 3, 5)
    bt_corr = [float(cov_t[:, k, 0, 2].mean()) for k in range(n_kt - 1)]
    bt_share = [float(data.clone_sharing[:, :, 0, k, 2].mean()) for k in range(n_kt - 1)]
    ax5.scatter(bt_share, bt_corr, s=80)
    for k, pheno in enumerate(data.tcell_phenotypes[:-1]):
        ax5.annotate(pheno[:8], (bt_share[k], bt_corr[k]), fontsize=8)
    ax5.set_xlabel('Clone sharing (Blood→Tumor)')
    ax5.set_ylabel('Compositional correlation')
    ax5.set_title('E. Clone sharing vs\ncomposition correlation', fontsize=11)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel F: Summary
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = """
    Key Findings:
    
    • Cytotoxic T cells show unique
      blood-tumor coupling (r=0.31)
    
    • Trafficking rates: 9-15%
      (highest for Cytotoxic)
    
    • Myeloid are tissue-resident
      (no cross-compartment signal)
    
    • T-myeloid coupling is
      compartment-specific
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', transform=ax6.transAxes)
    ax6.set_title('F. Summary', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig