# %%
"""Publication-quality figures for the empirical-Q analysis.

Reads CSV artifacts produced by ``pipeline/06c_empirical_Q.py`` and emits
four figures in ``results/06c_empirical_Q/figures/``:

  1. ``migration_flow_network.{png,pdf}`` — triangle flow graph between
     PBMC / CSF / TP, mean migration rates as arrow widths/labels.
  2. ``migration_heatmap.{png,pdf}`` — phenotype × directed-edge migration
     heatmap (optimizer rates).
  3. ``transition_heatmaps.{png,pdf}`` — three within-tissue phenotype →
     phenotype heatmaps (shared vmax).
  4. ``exit_entry_rates.{png,pdf}`` — bar charts of clone exit / entry
     probabilities by (tissue, phenotype). Only emitted if the augmented
     pipeline (06c augmented section) has produced ``exit_rates.csv`` and
     ``entry_rates.csv``.

No adata loading — pure CSV → figure.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import to_rgba

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (  # noqa: E402
    TCELL_PHENOTYPE_ORDER,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TISSUE_COLORS,
)

# %%
# ---- Config ----
SRC_DIR = REPO_ROOT / "results" / "06c_empirical_Q"
OUT_DIR = SRC_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = ("PBMC", "CSF", "TP")
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)
EDGES = [(a, b) for a in TISSUES for b in TISSUES if a != b]
EDGE_LABELS = [f"{a}→{b}" for a, b in EDGES]

DPI = 200

# Tissue node positions: PBMC top-left, TP top-right, CSF bottom (triangle).
NODE_POS = {"PBMC": (-0.8, 0.7), "TP": (0.8, 0.7), "CSF": (0.0, -0.7)}
NODE_RADIUS = 0.30


def _save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"  wrote {OUT_DIR / (name + '.png')}")


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)


# %%
# ---- Load source CSVs ----
print(f"Reading from {SRC_DIR}")
mig_df = pd.read_csv(SRC_DIR / "migration_rates.csv")
mig_opt = mig_df[mig_df["method"] == "optimizer"].copy()
print(f"  migration_rates.csv: {len(mig_df)} rows, "
      f"optimizer subset = {len(mig_opt)}")

trans_df = pd.read_csv(SRC_DIR / "transition_rates.csv")
trans_opt = trans_df[trans_df["method"] == "optimizer"].copy()
print(f"  transition_rates.csv: {len(trans_df)} rows, "
      f"optimizer subset = {len(trans_opt)}")

P_emp = pd.read_csv(SRC_DIR / "P_empirical.csv", index_col=0)
print(f"  P_empirical.csv: shape {P_emp.shape}")

exit_path = SRC_DIR / "exit_rates.csv"
entry_path = SRC_DIR / "entry_rates.csv"
HAS_AUGMENTED = exit_path.exists() and entry_path.exists()
if HAS_AUGMENTED:
    exit_df = pd.read_csv(exit_path)
    entry_df = pd.read_csv(entry_path)
    print(f"  exit_rates.csv: {len(exit_df)} rows")
    print(f"  entry_rates.csv: {len(entry_df)} rows")
else:
    print("  exit_rates.csv / entry_rates.csv missing — "
          "Figure 4 will be skipped.")


# Block retention from P_empirical: per tissue, fraction of mass kept
# within the same tissue (block diagonal). State label format is
# "{tissue}__{phenotype}".
def _block_retention(P, tissues, phenotypes):
    out = {}
    for tis in tissues:
        idx = [f"{tis}__{p}" for p in phenotypes if f"{tis}__{p}" in P.index]
        block_sum = float(P.loc[idx, idx].values.sum())
        row_total = float(P.loc[idx, :].values.sum())
        out[tis] = block_sum / row_total if row_total > 0 else float("nan")
    return out


block_ret = _block_retention(P_emp, TISSUES, PHENOTYPES)
print(f"  Block retention per tissue: {block_ret}")


# %%
# ==============================================================
# Figure 1: Migration flow network
# ==============================================================
print("\nFigure 1: migration flow network")
mean_rate = (mig_opt.groupby(["src", "dst"], observed=True)["rate"]
                    .mean()
                    .to_dict())
mean_rate_arr = np.array(list(mean_rate.values()))
rate_max = float(np.abs(mean_rate_arr).max()) if mean_rate_arr.size else 1.0
LW_MIN, LW_MAX = 1.5, 6.0

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.5, 1.3)
ax.set_aspect("equal")
ax.axis("off")

for tis, (x, y) in NODE_POS.items():
    ax.add_patch(Circle((x, y), NODE_RADIUS,
                         facecolor=TISSUE_COLORS[tis],
                         edgecolor="black", linewidth=1.4, zorder=3))
    ax.text(x, y + 0.06, tis, ha="center", va="center",
            fontsize=12, fontweight="bold", color="black", zorder=4)
    ret = block_ret.get(tis, float("nan"))
    if not np.isnan(ret):
        ax.text(x, y - 0.10, f"{ret*100:.0f}%",
                 ha="center", va="center",
                 fontsize=10, color="black",
                 fontweight="normal", zorder=4)

# Draw migration arrows with curvature so both directions are visible.
for (a, b), rate in mean_rate.items():
    x1, y1 = NODE_POS[a]
    x2, y2 = NODE_POS[b]
    dx, dy = x2 - x1, y2 - y1
    L = float(np.hypot(dx, dy))
    if L == 0:
        continue
    ux, uy = dx / L, dy / L
    sx, sy = x1 + ux * NODE_RADIUS, y1 + uy * NODE_RADIUS
    ex, ey = x2 - ux * NODE_RADIUS, y2 - uy * NODE_RADIUS
    color = TISSUE_COLORS[b]
    lw = LW_MIN + (LW_MAX - LW_MIN) * (abs(rate) / max(rate_max, 1e-9))
    arr = FancyArrowPatch(
        posA=(sx, sy), posB=(ex, ey),
        connectionstyle="arc3,rad=0.18",
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=float(lw),
        zorder=2,
    )
    ax.add_patch(arr)

    # Offset label perpendicular to the arrow toward the curved side.
    nx_, ny_ = uy, -ux
    mx, my = (sx + ex) / 2 + nx_ * 0.13, (sy + ey) / 2 + ny_ * 0.13
    ax.text(mx, my, f"{rate:.2f}", ha="center", va="center",
            fontsize=9, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.25",
                       facecolor="white", edgecolor="none", alpha=0.85),
            zorder=5)

ax.set_title("Mean migration rates between compartments",
              fontsize=13, fontweight="bold", pad=8)
ax.text(0.0, -1.40,
         "Arrow width ∝ mean optimizer migration rate (phenotype-averaged); "
         "tissue node label = block retention",
         ha="center", va="center", fontsize=8.5, color="dimgray",
         style="italic")
fig.tight_layout()
_save(fig, "migration_flow_network")
plt.close(fig)


# %%
# ==============================================================
# Figure 2: Per-phenotype migration heatmap
# ==============================================================
print("\nFigure 2: migration heatmap (phenotype × edge)")
mig_opt = mig_opt.copy()
mig_opt["edge"] = mig_opt["src"] + "→" + mig_opt["dst"]
mig_mat = (mig_opt
           .pivot_table(index="phenotype", columns="edge",
                        values="rate", aggfunc="first")
           .reindex(index=PHENOTYPES, columns=EDGE_LABELS))

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(mig_mat.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(EDGE_LABELS)))
ax.set_xticklabels(EDGE_LABELS, rotation=35, ha="right", fontsize=10)
for tick, edge in zip(ax.get_xticklabels(), EDGES):
    tick.set_color(TISSUE_COLORS[edge[1]])
    tick.set_fontweight("bold")
ax.set_yticks(range(len(PHENOTYPES)))
ax.set_yticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in PHENOTYPES],
                    fontsize=10)
for tick, ph in zip(ax.get_yticklabels(), PHENOTYPES):
    tick.set_color(TCELL_PHENOTYPE_COLORS.get(ph, "black"))
    tick.set_fontweight("bold")

for i in range(mig_mat.shape[0]):
    for j in range(mig_mat.shape[1]):
        v = mig_mat.values[i, j]
        if np.isnan(v):
            continue
        # Pick annotation color for contrast.
        rgba = im.cmap(im.norm(v))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "white" if lum < 0.5 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=7, color=text_color)

ax.set_title("Migration rates by phenotype × edge",
              fontsize=13, fontweight="bold", pad=8)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("optimizer migration rate", fontsize=10)
for s in ("top", "right", "bottom", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
_save(fig, "migration_heatmap")
plt.close(fig)


# %%
# ==============================================================
# Figure 3: Within-tissue phenotype transition heatmaps
# ==============================================================
print("\nFigure 3: within-tissue transition heatmaps")
mats = {}
for tis in TISSUES:
    sub = trans_opt[trans_opt["site"] == tis]
    mat = (sub.pivot_table(index="src", columns="dst",
                            values="rate", aggfunc="first")
              .reindex(index=PHENOTYPES, columns=PHENOTYPES)
              .fillna(0.0))
    np.fill_diagonal(mat.values, 0.0)
    mats[tis] = mat

vmax = float(np.max([m.values.max() for m in mats.values()]))
vmax = max(vmax, 1e-9)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
for i, tis in enumerate(TISSUES):
    ax = axes[i]
    mat = mats[tis]
    im = ax.imshow(mat.values, aspect="auto", cmap="Oranges",
                    vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(PHENOTYPES)))
    ax.set_xticklabels(
        [TCELL_PHENOTYPE_LABELS.get(p, p) for p in PHENOTYPES],
        rotation=45, ha="right", fontsize=8)
    for tick, ph in zip(ax.get_xticklabels(), PHENOTYPES):
        tick.set_color(TCELL_PHENOTYPE_COLORS.get(ph, "black"))
    ax.set_yticks(range(len(PHENOTYPES)))
    if i == 0:
        ax.set_yticklabels(
            [TCELL_PHENOTYPE_LABELS.get(p, p) for p in PHENOTYPES],
            fontsize=9)
        for tick, ph in zip(ax.get_yticklabels(), PHENOTYPES):
            tick.set_color(TCELL_PHENOTYPE_COLORS.get(ph, "black"))
            tick.set_fontweight("bold")
    else:
        ax.set_yticklabels([])
    ax.set_title(tis, fontsize=13, fontweight="bold",
                  color=TISSUE_COLORS[tis], pad=6)

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat.values[r, c]
            if v > 0.1:
                rgba = im.cmap(im.norm(v))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if lum < 0.5 else "black"
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color=text_color)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if i == 1:
        ax.set_xlabel("destination phenotype", fontsize=10)
    if i == 0:
        ax.set_ylabel("source phenotype", fontsize=10)

cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
cb.set_label("transition rate (optimizer)", fontsize=10)
fig.suptitle("Within-tissue phenotype transition rates",
              fontsize=13, fontweight="bold")
_save(fig, "transition_heatmaps")
plt.close(fig)


# %%
# ==============================================================
# Figure 4: Exit / entry rates (augmented model)
# ==============================================================
if HAS_AUGMENTED:
    print("\nFigure 4: exit and entry rates")
    exit_sorted = exit_df.sort_values("exit_prob", ascending=False).copy()
    entry_sorted = entry_df.sort_values("entry_prob", ascending=False).copy()
    exit_sorted["label"] = (
        exit_sorted["tissue"] + " · "
        + exit_sorted["phenotype"].map(
            lambda p: TCELL_PHENOTYPE_LABELS.get(p, p)))
    entry_sorted["label"] = (
        entry_sorted["tissue"] + " · "
        + entry_sorted["phenotype"].map(
            lambda p: TCELL_PHENOTYPE_LABELS.get(p, p)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 8))

    for ax, df, col, title in [
        (axes[0], exit_sorted, "exit_prob", "Exit (death / emigration)"),
        (axes[1], entry_sorted, "entry_prob", "Entry (birth / immigration)"),
    ]:
        colors = [TISSUE_COLORS[t] for t in df["tissue"]]
        y = np.arange(len(df))
        ax.barh(y, df[col].values, color=colors, edgecolor="black",
                 linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(df["label"].values, fontsize=8)
        for tick, tis in zip(ax.get_yticklabels(), df["tissue"]):
            tick.set_color(TISSUE_COLORS.get(tis, "black"))
        ax.invert_yaxis()
        ax.set_xlabel("probability", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="x", alpha=0.15, linewidth=0.5)

    fig.suptitle(
        "Clone exit (death/emigration) and entry (birth/immigration) rates",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "exit_entry_rates")
    plt.close(fig)
else:
    print("\nFigure 4: SKIPPED (no augmented exit/entry CSVs found).")

print(f"\nDone. Figures in: {OUT_DIR}")
