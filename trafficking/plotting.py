import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.linalg import expm

from .style import COLORS, tissue_color, tissue_label, apply_style


# ==================================================================
# Bayesian transition model plots (existing)
# ==================================================================

def plot_elbo(result, figsize=(8, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    losses = result["losses"]
    ax.plot(losses, linewidth=0.8, color="#2c3e50")
    ax.set_xlabel("Step")
    ax.set_ylabel("ELBO Loss")
    ax.set_title("SVI Convergence")
    cutoff = int(len(losses) * 0.8)
    ax2 = ax.inset_axes([0.5, 0.4, 0.45, 0.5])
    ax2.plot(range(cutoff, len(losses)), losses[cutoff:], linewidth=0.8, color="#e74c3c")
    ax2.set_title("Last 20%", fontsize=8)
    ax2.tick_params(labelsize=6)
    plt.tight_layout()
    return fig


def plot_posterior(result, t1=None, t2=None, figsize=(14, 5)):
    T_mean = result["T_mean"]
    T_std = result["T_std"]
    phenotypes = result["phenotypes"]
    K = len(phenotypes)
    title = f"{t1} → {t2}" if t1 and t2 else "Transition Matrix"

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(T_mean, cmap="YlOrRd", vmin=0, vmax=0.6, aspect="auto")
    for i in range(K):
        for j in range(K):
            color = "white" if T_mean[i, j] > 0.35 else "black"
            axes[0].text(j, i, f"{T_mean[i,j]:.2f}", ha="center", va="center",
                         fontsize=7, color=color, fontweight="bold")
    axes[0].set_title("T_global (posterior mean)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(T_std, cmap="Purples", vmin=0, vmax=0.15, aspect="auto")
    for i in range(K):
        for j in range(K):
            axes[1].text(j, i, f"{T_std[i,j]:.3f}", ha="center", va="center",
                         fontsize=7, color="black")
    axes[1].set_title("Posterior uncertainty (std)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

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
    fig.suptitle("Posterior distributions of transition probabilities",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_kappa(result, figsize=(6, 4)):
    if "kappa_samples" not in result:
        print("No kappa (non-hierarchical model)")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    kappa = result["kappa_samples"]
    ax.hist(kappa, bins=50, color="#8e44ad", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(kappa), color="black", linewidth=2, label=f"mean={np.mean(kappa):.1f}")
    ax.set_xlabel("κ")
    ax.set_ylabel("Count")
    ax.set_title("Patient concentration parameter κ\n"
                 "(high = patients similar, low = heterogeneous)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_credible(result, threshold=0.1, ci=0.9, figsize=(10, 6)):
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
    ax.axvline(threshold, color="red", linestyle="--", alpha=0.5,
               label=f"threshold={threshold}")
    ax.set_xlabel("Transition probability")
    ax.set_title(f"Credible transitions ({ci:.0%} CI lower bound > {threshold})")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_asymmetry(comparison, phenotypes=None, figsize=(8, 7)):
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


# ==================================================================
# CTMC plots
# ==================================================================

def _annotate_matrix(ax, mat, fmt=".3f", threshold=None):
    N = mat.shape[0]
    for i in range(N):
        for j in range(N):
            val = mat[i, j]
            if threshold is not None:
                color = "white" if abs(val) > threshold else "black"
            else:
                color = "white" if abs(val) > (mat.max() - mat.min()) * 0.6 + mat.min() else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=5, color=color)


def plot_generator(result, figsize=(8, 7)):
    """Heatmap of the full generator matrix Q."""
    Q = result["Q"]
    ss = result["ss"]

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.abs(Q).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(Q, cmap="RdBu_r", norm=norm, aspect="auto")
    _annotate_matrix(ax, Q, fmt=".3f", threshold=vmax * 0.4)

    labels = [f"{s[:3]}:{p[:4]}" for s, p in ss.labels]
    ax.set_xticks(range(ss.N))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(ss.N))
    ax.set_yticklabels(labels, fontsize=6)

    for s in range(1, ss.S):
        pos = s * ss.K - 0.5
        ax.axhline(pos, color="black", linewidth=0.8)
        ax.axvline(pos, color="black", linewidth=0.8)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Rate")
    ax.set_title("Generator matrix Q", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_transition_matrix(result, dt=None, figsize=(8, 7)):
    """Heatmap of the transition probability matrix P = exp(Q·dt)."""
    Q = result["Q"]
    ss = result["ss"]
    if dt is None:
        dt = result.get("dt", 1.0)
    P = expm(Q * dt)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=P.max(), aspect="auto")
    _annotate_matrix(ax, P, fmt=".3f", threshold=P.max() * 0.5)

    labels = [f"{s[:3]}:{p[:4]}" for s, p in ss.labels]
    ax.set_xticks(range(ss.N))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(ss.N))
    ax.set_yticklabels(labels, fontsize=6)

    for s in range(1, ss.S):
        pos = s * ss.K - 0.5
        ax.axhline(pos, color="black", linewidth=0.8)
        ax.axvline(pos, color="black", linewidth=0.8)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Probability")
    ax.set_title(f"Transition matrix P = exp(Q·{dt})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_rate_bars(result, component="migration", top_n=15, figsize=(10, 6)):
    """Horizontal bar chart of top rates from a CTMC fit."""
    from .ctmc import migration_table, transition_table, expansion_table

    if component == "migration":
        df = migration_table(result)
        df["label"] = df["from"] + " → " + df["to"] + " [" + df["phenotype"] + "]"
        title = "Migration rates"
        color = "#3498db"
    elif component == "transition":
        df = transition_table(result)
        df["label"] = df["from"] + " → " + df["to"] + " @ " + df["site"]
        title = "Phenotype transition rates"
        color = "#e74c3c"
    elif component == "expansion":
        df = expansion_table(result)
        df["label"] = df["site"] + " [" + df["phenotype"] + "]"
        title = "Expansion / death rates"
        color = None
    else:
        raise ValueError(f"Unknown component: {component}")

    df = df.head(top_n).sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    if component == "expansion":
        colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in df["rate"]]
        ax.barh(range(len(df)), df["rate"], color=colors, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.5)
    else:
        ax.barh(range(len(df)), df["rate"], color=color, alpha=0.8, edgecolor="white")

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_rates_decomposed(result, figsize=(18, 5)):
    """Three-panel view: migration, transition, expansion rate heatmaps."""
    ss = result["ss"]
    mig = result["migration"]
    trans = result["transition"]
    exp = result["expansion"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Migration: average across phenotypes for a S×S summary
    mig_avg = mig.mean(axis=2)
    im0 = axes[0].imshow(mig_avg, cmap="Blues", vmin=0, aspect="auto")
    for i in range(ss.S):
        for j in range(ss.S):
            axes[0].text(j, i, f"{mig_avg[i,j]:.3f}", ha="center", va="center",
                         fontsize=9, color="white" if mig_avg[i, j] > mig_avg.max() * 0.5 else "black")
    axes[0].set_xticks(range(ss.S))
    axes[0].set_xticklabels(ss.sites, fontsize=9)
    axes[0].set_yticks(range(ss.S))
    axes[0].set_yticklabels(ss.sites, fontsize=9)
    axes[0].set_title("Migration rates\n(avg over phenotypes)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("From")
    axes[0].set_xlabel("To")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # Transition: average across sites for a K×K summary
    trans_avg = trans.mean(axis=0)
    im1 = axes[1].imshow(trans_avg, cmap="Oranges", vmin=0, aspect="auto")
    for i in range(ss.K):
        for j in range(ss.K):
            axes[1].text(j, i, f"{trans_avg[i,j]:.3f}", ha="center", va="center",
                         fontsize=9, color="white" if trans_avg[i, j] > trans_avg.max() * 0.5 else "black")
    axes[1].set_xticks(range(ss.K))
    axes[1].set_xticklabels(ss.phenotypes, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticks(range(ss.K))
    axes[1].set_yticklabels(ss.phenotypes, fontsize=8)
    axes[1].set_title("Transition rates\n(avg over sites)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("From")
    axes[1].set_xlabel("To")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Expansion: S×K heatmap
    vmax = max(abs(exp.min()), abs(exp.max())) or 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = axes[2].imshow(exp, cmap="RdBu_r", norm=norm, aspect="auto")
    for i in range(ss.S):
        for j in range(ss.K):
            axes[2].text(j, i, f"{exp[i,j]:+.3f}", ha="center", va="center",
                         fontsize=9, color="white" if abs(exp[i, j]) > vmax * 0.5 else "black")
    axes[2].set_xticks(range(ss.K))
    axes[2].set_xticklabels(ss.phenotypes, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticks(range(ss.S))
    axes[2].set_yticklabels(ss.sites, fontsize=9)
    axes[2].set_title("Expansion / death rates", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Site")
    axes[2].set_xlabel("Phenotype")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    return fig


def plot_trajectory(result, initial_state, n_steps=5, dt=None,
                    observed_states=None, figsize=(14, 5)):
    """Plot predicted vs observed state trajectories per compartment."""
    from .ctmc import predict_trajectory

    ss = result["ss"]
    if dt is None:
        dt = result.get("dt", 1.0)
    traj = predict_trajectory(initial_state, result["Q"], n_steps, dt)

    fig, axes = plt.subplots(1, ss.S, figsize=figsize, sharey=True)
    times = np.arange(n_steps + 1) * dt

    for s_idx, site in enumerate(ss.sites):
        ax = axes[s_idx]
        sl = ss.site_slice(site)
        for k_idx, pheno in enumerate(ss.phenotypes):
            i = ss.idx(site, pheno)
            ax.plot(times, traj[:, i], "-o", markersize=4, label=pheno, linewidth=1.5)
            if observed_states is not None and len(observed_states) > 0:
                obs_vals = [obs[i] if t < len(observed_states) else np.nan
                            for t, obs in enumerate(observed_states)]
                ax.plot(times[:len(obs_vals)], obs_vals, "x", markersize=8, color="black",
                        alpha=0.6)

        ax.set_title(tissue_label(site) if site in COLORS["tissue"] else site,
                     fontsize=10, fontweight="bold",
                     color=tissue_color(site) if site in COLORS["tissue"] else "black")
        ax.set_xlabel("Time")
        if s_idx == 0:
            ax.set_ylabel("Fraction")
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.15)

    fig.suptitle("CTMC predicted trajectory" +
                 (" (× = observed)" if observed_states else ""),
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_holdout(observations, ss, figsize=(12, 4), **fit_kwargs):
    """Bar chart of leave-one-interval-out MSE scores."""
    from .ctmc import leave_one_interval_out

    scores = leave_one_interval_out(observations, ss, **fit_kwargs)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(scores)), scores, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axhline(np.mean(scores), color="red", linestyle="--", linewidth=1,
               label=f"mean={np.mean(scores):.6f}")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels([f"Interval {i}" for i in range(len(scores))], fontsize=8)
    ax.set_ylabel("MSE")
    ax.set_title("Leave-one-interval-out cross-validation", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig