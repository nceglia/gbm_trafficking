import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


def plot_elbo(result, figsize=(8, 3)):
    """ELBO convergence curve."""
    fig, ax = plt.subplots(figsize=figsize)
    losses = result["losses"]
    ax.plot(losses, linewidth=0.8, color="#2c3e50")
    ax.set_xlabel("Step")
    ax.set_ylabel("ELBO Loss")
    ax.set_title("SVI Convergence")
    # show last 20% for scale
    cutoff = int(len(losses) * 0.8)
    ax2 = ax.inset_axes([0.5, 0.4, 0.45, 0.5])
    ax2.plot(range(cutoff, len(losses)), losses[cutoff:], linewidth=0.8, color="#e74c3c")
    ax2.set_title("Last 20%", fontsize=8)
    ax2.tick_params(labelsize=6)
    plt.tight_layout()
    return fig


def plot_posterior(result, t1=None, t2=None, figsize=(14, 5)):
    """Heatmap of T_mean ± T_std and departure from identity."""
    T_mean = result["T_mean"]
    T_std = result["T_std"]
    phenotypes = result["phenotypes"]
    K = len(phenotypes)
    title = f"{t1} → {t2}" if t1 and t2 else "Transition Matrix"

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # T_mean
    im0 = axes[0].imshow(T_mean, cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
    for i in range(K):
        for j in range(K):
            color = "white" if T_mean[i, j] > 0.35 else "black"
            axes[0].text(j, i, f"{T_mean[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")
    axes[0].set_title("T_global (posterior mean)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # T_std
    im1 = axes[1].imshow(T_std, cmap="Purples", vmin=0, vmax=0.15, aspect="auto")
    for i in range(K):
        for j in range(K):
            axes[1].text(j, i, f"{T_std[i,j]:.3f}", ha="center", va="center",
                        fontsize=7, color="black")
    axes[1].set_title("Posterior uncertainty (std)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Departure from identity
    identity = np.eye(K)
    departure = T_mean - identity
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    im2 = axes[2].imshow(departure, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(K):
        for j in range(K):
            color = "white" if abs(departure[i, j]) > 0.3 else "black"
            axes[2].text(j, i, f"{departure[i,j]:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)
    axes[2].set_title("Departure from identity")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xticks(range(K))
        ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels(phenotypes, fontsize=8)
        ax.set_xlabel("Destination")
    axes[0].set_ylabel("Source")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_patient_heterogeneity(result, t1=None, t2=None, figsize=None):
    """Per-patient transition matrices alongside global."""
    if "T_patient_mean" not in result:
        print("No patient-level data (non-hierarchical model)")
        return None

    phenotypes = result["phenotypes"]
    pat_names = result["pat_names"]
    T_global = result["T_mean"]
    T_patient = result["T_patient_mean"]
    P, K = T_patient.shape[0], T_patient.shape[1]

    ncols = min(P + 1, 4)
    nrows = int(np.ceil((P + 1) / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    all_matrices = [("GLOBAL", T_global)] + [(pat_names[p], T_patient[p]) for p in range(P)]

    for idx, (name, T) in enumerate(all_matrices):
        ax = axes[idx]
        ax.imshow(T, cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
        for i in range(K):
            for j in range(K):
                color = "white" if T[i, j] > 0.35 else "black"
                ax.text(j, i, f"{T[i,j]:.2f}", ha="center", va="center",
                       fontsize=6, color=color)
        ax.set_title(name, fontsize=10, fontweight="bold" if idx == 0 else "normal")
        ax.set_xticks(range(K))
        ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(K))
        ax.set_yticklabels(phenotypes, fontsize=6)

    for idx in range(len(all_matrices), len(axes)):
        axes[idx].axis("off")

    title = f"Patient heterogeneity: {t1} → {t2}" if t1 and t2 else "Patient heterogeneity"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_posterior_intervals(result, figsize=(16, 10)):
    """Violin plots showing full posterior distribution for each transition."""
    phenotypes = result["phenotypes"]
    T_samples = result["T_samples"]
    K = len(phenotypes)

    fig, axes = plt.subplots(K, 1, figsize=figsize, sharex=True)
    positions = range(K)

    for i in range(K):
        ax = axes[i]
        data = [T_samples[:, i, j] for j in range(K)]
        parts = ax.violinplot(data, positions=positions, showmeans=True, widths=0.7)
        for j, pc in enumerate(parts["bodies"]):
            pc.set_facecolor("#3498db" if i != j else "#e74c3c")
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")
        ax.set_ylabel(phenotypes[i], fontsize=9, rotation=0, ha="right", va="center")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.grid(alpha=0.15, axis="y")

    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=9)
    axes[-1].set_xlabel("Destination phenotype")
    fig.suptitle("Posterior distributions of transition probabilities", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_kappa(result, figsize=(6, 4)):
    """Distribution of κ (patient concentration parameter)."""
    if "kappa_samples" not in result:
        print("No kappa (non-hierarchical model)")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    kappa = result["kappa_samples"]
    ax.hist(kappa, bins=50, color="#8e44ad", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(kappa), color="black", linewidth=2, label=f"mean={np.mean(kappa):.1f}")
    ax.set_xlabel("κ")
    ax.set_ylabel("Count")
    ax.set_title("Patient concentration parameter κ\n(high = patients similar, low = heterogeneous)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_credible(result, threshold=0.1, ci=0.9, figsize=(10, 6)):
    """Bar chart of credible off-diagonal transitions."""
    from .inference import credible_transitions
    df = credible_transitions(result, threshold=threshold, ci=ci)
    off_diag = df[df["source"] != df["destination"]].copy()
    cred = off_diag[off_diag["credible"]].sort_values("mean", ascending=True)

    if len(cred) == 0:
        print(f"No transitions with {ci:.0%} CI lower bound > {threshold}")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    labels = [f"{r['source']} → {r['destination']}" for _, r in cred.iterrows()]
    lo_col = f"ci_{ci:.0%}_lo"
    hi_col = f"ci_{ci:.0%}_hi"
    xerr_lo = cred["mean"].values - cred[lo_col].values
    xerr_hi = cred[hi_col].values - cred["mean"].values

    ax.barh(range(len(cred)), cred["mean"], xerr=[xerr_lo, xerr_hi],
            color="#2ecc71", alpha=0.8, capsize=3, edgecolor="white")
    ax.set_yticks(range(len(cred)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(threshold, color="red", linestyle="--", alpha=0.5, label=f"threshold={threshold}")
    ax.set_xlabel("Transition probability")
    ax.set_title(f"Credible transitions ({ci:.0%} CI lower bound > {threshold})")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_asymmetry(comparison, phenotypes=None, figsize=(8, 7)):
    """Heatmap of asymmetry matrix from compare_directions."""
    if comparison is None:
        return None

    asym = comparison["asymmetry"]
    if phenotypes is None:
        phenotypes = comparison["forward"]["phenotypes"]
    K = len(phenotypes)

    fig, ax = plt.subplots(figsize=figsize)
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
    im = ax.imshow(asym, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(K):
        for j in range(K):
            color = "white" if abs(asym[i, j]) > 0.25 else "black"
            ax.text(j, i, f"{asym[i,j]:+.2f}", ha="center", va="center",
                   fontsize=7, color=color)

    ax.set_xticks(range(K))
    ax.set_xticklabels(phenotypes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(phenotypes, fontsize=8)
    plt.colorbar(im, ax=ax, label="Asymmetry (fwd − rev')", shrink=0.8)
    ax.set_title("Directional asymmetry\n(positive = preferential forward transition)")
    plt.tight_layout()
    return fig
