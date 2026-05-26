# %%
"""Figure 2 — clone trafficking, dispersion, and within-tissue stability.

Nine panels (4-row layout):
  A  3 density UMAPs (one per tissue)
  B  Tissue similarity triangle (mean cosine distance per tissue pair)
  C  Per-phenotype CSF/PBMC divergence to TP
  D  Resident vs Migratory phenotype enrichment (3 sub-panels)
  E  Ternary persistence plot (CD8 + CD4 side by side)
  F  Within-tissue phenotypic flux (L1 distance violins)
  G  Migration flow network triangle
  H  Tissue retention over time
  I  Reserved placeholder

New category definitions (Panel D, F):
  Resident  : clone is ONLY in source tissue at t+1 (old STAYER_ONLY)
  Migratory : clone has cells in ANY other tissue at t+1
              (old MOVER_ONLY + BOTH merged)

Reads:
  data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad  (paths.H5AD_TCELLS)
  data/embeddings/X_umap.pkl
  results/transcriptome_similarity/
      cosine_distance_summary.csv                      (Panels B, C)
  results/06c_empirical_Q/migration_rates.csv          (Panel G)
  results/06c_empirical_Q/P_empirical.csv              (Panel G)
  results/06d_empirical_Q_per_timepoint/
      block_retention_per_timepoint.csv                (Panel H)

Writes to results/07_figure2/.
"""
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.stats import gaussian_kde, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (  # noqa: E402
    LINEAGE_COLORS,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_ORDER,
)


# %%
# ---- Config ----
from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
UMAP_PATH = REPO_ROOT / "data" / "embeddings" / "X_umap.pkl"
MIG_CSV = (REPO_ROOT / "results" / "06c_empirical_Q"
           / "migration_rates.csv")
P_EMP_CSV = REPO_ROOT / "results" / "06c_empirical_Q" / "P_empirical.csv"
RETENTION_TS_CSV = (REPO_ROOT / "results" / "06d_empirical_Q_per_timepoint"
                    / "block_retention_per_timepoint.csv")
COSINE_CSV = (REPO_ROOT / "results" / "transcriptome_similarity"
              / "cosine_distance_summary.csv")
OUT_DIR = REPO_ROOT / "results" / "07_figure2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = list(TISSUE_ORDER)
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)
K_PHEN = len(PHENOTYPES)
TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
NEXT_TP = {t1: t2 for t1, t2 in TRANSITIONS}
LINEAGES = ["CD8", "CD4"]
FDR_THRESHOLD = 0.10
MIN_N_SRC = 3
MIN_CELLS_BOTH = 3
N_PERM = 1000

# Resident / Migratory palette (replaces the 3-segment Stayer/Both/Mover).
COL_RESIDENT  = "#6699CC"
COL_MIGRATORY = "#E8913A"

DPI_PANEL = 200
DPI_FIG = 300
MIN_FONTSIZE = 7
PANEL_LETTER_XY = (-0.15, 1.05)
RENDER_FULL_FIGURE = True
TOP_N_PATHWAYS_I = 25
MIN_CELLS_PSEUDOBULK_I = 5

# Per-tissue sequential colormaps for the density panels.
DENSITY_CMAPS = {"PBMC": "YlOrBr", "CSF": "OrRd", "TP": "YlGn"}

try:
    from adjustText import adjust_text as _adjust_text_fn
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    _adjust_text_fn = None


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


def _panel_letter(ax, letter, x=None, y=None):
    if x is None: x = PANEL_LETTER_XY[0]
    if y is None: y = PANEL_LETTER_XY[1]
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left")


def _lineage_of(p):
    if isinstance(p, str) and "CD8" in p:
        return "CD8"
    return "CD4"


def check_min_fontsize(ax, label="", min_size=MIN_FONTSIZE):
    offenders = []
    for t in ax.texts:
        if not t.get_visible():
            continue
        if not (t.get_text() or "").strip():
            continue
        fs = t.get_fontsize()
        if fs < min_size:
            offenders.append((t.get_text()[:24], fs))
    if offenders:
        print(f"  WARNING [{label}]: {len(offenders)} text(s) below "
              f"{min_size}pt:")
        for txt, fs in offenders[:5]:
            print(f"    '{txt}'  @{fs}pt")
    return offenders


def check_text_overlaps(fig, ax, label=""):
    fig.canvas.draw()
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        return []
    texts = [t for t in ax.texts if t.get_visible()
             and (t.get_text() or "").strip()]
    overlaps = []
    for i, t1 in enumerate(texts):
        try:
            bb1 = t1.get_window_extent(renderer)
        except Exception:
            continue
        for t2 in texts[i + 1:]:
            try:
                bb2 = t2.get_window_extent(renderer)
            except Exception:
                continue
            if bb1.overlaps(bb2):
                overlaps.append((t1.get_text(), t2.get_text()))
    if overlaps:
        print(f"  WARNING [{label}]: {len(overlaps)} text overlap(s):")
        for a, b in overlaps[:5]:
            a_s = (a[:20] + "…") if len(a) > 20 else a
            b_s = (b[:20] + "…") if len(b) > 20 else b
            print(f"    '{a_s}' ↔ '{b_s}'")
    return overlaps


# %%
# ---- Load adata ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
print(f"  {adata.n_obs:,} cells")

obs = adata.obs[["trb", "tissue", "timepoint", "phenotype",
                  "patient"]].copy()
obs = obs[obs["trb"].notna() & (obs["trb"].astype(str) != "")]
obs["patient"] = obs["patient"].astype(str)
obs["trb"] = obs["trb"].astype(str)
obs["tissue"] = obs["tissue"].astype(str)
obs["phenotype"] = obs["phenotype"].astype(str)
obs["clone_id"] = obs["patient"] + "|" + obs["trb"]
obs["lineage"] = obs["phenotype"].map(_lineage_of)
print(f"  {len(obs):,} TCR+ cells; {obs['clone_id'].nunique():,} clones")

PATIENTS = sorted(obs["patient"].unique())

# Per-(patient, trb, tissue, timepoint, phenotype) cell counts (wide).
ph = (obs.groupby(["patient", "trb", "tissue", "timepoint",
                    "phenotype"], observed=True)
        .size().unstack("phenotype", fill_value=0))
for p in PHENOTYPES:
    if p not in ph.columns:
        ph[p] = 0
ph = ph[PHENOTYPES]
ph["_n"] = ph.sum(axis=1)
ph = ph.reset_index()

present_set = set(
    obs.groupby(["patient", "trb", "tissue", "timepoint"],
                 observed=True).indices.keys()
)

PHENO_IDX = {p: i for i, p in enumerate(PHENOTYPES)}


# %%
# =========================================================
# Panel A — Embedding density UMAPs (unchanged)
# =========================================================
print("\nPanel A: UMAP embedding density by tissue...")
with open(UMAP_PATH, "rb") as _fh:
    umap = pickle.load(_fh)
umap = np.asarray(umap)
if umap.shape[0] != adata.n_obs:
    raise RuntimeError(
        f"UMAP pkl has {umap.shape[0]} rows, adata has {adata.n_obs}; "
        "alignment is not guaranteed.")
adata.obsm["X_umap"] = umap.astype(float)
adata.obs["tissue"] = adata.obs["tissue"].astype("category")

TISSUE_COUNTS = {t: int((adata.obs["tissue"].astype(str) == t).sum())
                 for t in TISSUES}

if "umap_density_tissue" not in adata.obs:
    print("  computing embedding density per tissue...")
    sc.tl.embedding_density(adata, basis="umap", groupby="tissue")


def _draw_panel_A(axes):
    dens_all = adata.obs["umap_density_tissue"].to_numpy(dtype=float)
    x_lo, x_hi = np.percentile(umap[:, 0], [1, 99])
    y_lo, y_hi = np.percentile(umap[:, 1], [1, 99])
    for ax, tis in zip(axes, TISSUES):
        mask = (adata.obs["tissue"].astype(str) == tis).to_numpy()
        ax.scatter(umap[~mask, 0], umap[~mask, 1],
                    s=1.0, c="#e8e8e8", alpha=0.40,
                    edgecolors="none", rasterized=True)
        d_vals = dens_all[mask]
        if d_vals.size:
            vmax = float(np.nanpercentile(d_vals, 99))
            vmax = max(vmax, 1e-6)
        else:
            vmax = 1.0
        cmap = plt.get_cmap(DENSITY_CMAPS.get(tis, "Greys"))
        ax.scatter(umap[mask, 0], umap[mask, 1],
                    c=d_vals, cmap=cmap, vmin=0.0, vmax=vmax,
                    s=3.0, edgecolors="none", alpha=0.95,
                    rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "bottom", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_aspect("equal")
        # Inline title inside axes top-left to avoid wasting vertical
        # whitespace above the UMAP cloud.
        ax.text(0.5, 0.98,
                 f"{tis} (n={TISSUE_COUNTS[tis]:,})",
                 transform=ax.transAxes,
                 ha="center", va="top",
                 fontsize=10, fontweight="bold",
                 color=TISSUE_COLORS.get(tis, "black"))
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.margins(0)


# %%
# =========================================================
# Panel B / C — Tissue similarity (triangle + divergence bars)
# =========================================================
print("\nPanels B/C: tissue similarity from cosine distances...")
cosine_df = pd.read_csv(COSINE_CSV)


def panel_tissue_triangle(ax, cosine_df):
    from matplotlib.patches import Patch

    matched = cosine_df[cosine_df["type"] == "matched"].copy()
    matched["lineage"] = np.where(
        matched["phenotype"].astype(str).str.contains("CD8"), "CD8", "CD4")
    pair_lin_mean = (matched.groupby(["tissue_pair", "lineage"])
                            ["cosine_dist"].mean())
    pairs_avail = set(matched["tissue_pair"].unique())

    def _pair_key(a, b):
        for cand in (f"{a}_vs_{b}", f"{b}_vs_{a}"):
            if cand in pairs_avail:
                return cand
        raise KeyError(f"No tissue_pair for {a}/{b} in cosine_df")

    nodes = {
        "CSF":  (0.5, np.sqrt(3) / 2),
        "PBMC": (0.0, 0.0),
        "TP":   (1.0, 0.0),
    }
    edges = [("PBMC", "CSF"), ("CSF", "TP"), ("PBMC", "TP")]
    edge_keys = [_pair_key(a, b) for a, b in edges]
    all_vals = np.array([float(pair_lin_mean[(k, lin)])
                         for k in edge_keys for lin in ("CD8", "CD4")])
    LW_MIN_, LW_MAX_ = 1.5, 6.0
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

    def _lw(v):
        if vmax > vmin:
            return LW_MIN_ + (LW_MAX_ - LW_MIN_) * (v - vmin) / (vmax - vmin)
        return (LW_MIN_ + LW_MAX_) / 2

    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.15, np.sqrt(3) / 2 + 0.20)
    ax.axis("off")

    R = 0.11
    LINEAGE_RAD = {"CD8": 0.22, "CD4": -0.22}
    for (a, b), key in zip(edges, edge_keys):
        x1, y1 = nodes[a]; x2, y2 = nodes[b]
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        ux, uy = dx / L, dy / L
        sx, sy = x1 + ux * R, y1 + uy * R
        ex, ey = x2 - ux * R, y2 - uy * R
        chord_L = float(np.hypot(ex - sx, ey - sy))
        mx_mid, my_mid = (sx + ex) / 2, (sy + ey) / 2
        for lin in ("CD8", "CD4"):
            v = float(pair_lin_mean[(key, lin)])
            rad = LINEAGE_RAD[lin]
            arr = FancyArrowPatch(
                posA=(sx, sy), posB=(ex, ey),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-",
                color=LINEAGE_COLORS[lin],
                linewidth=_lw(v), zorder=1,
            )
            ax.add_patch(arr)
            sgn = np.sign(rad)
            nx_, ny_ = uy * sgn, -ux * sgn
            apex_d = abs(rad) * 0.5 * chord_L
            tx = mx_mid + nx_ * apex_d
            ty = my_mid + ny_ * apex_d
            ax.text(tx, ty, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color=LINEAGE_COLORS[lin],
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none",
                              alpha=0.92),
                    zorder=4)

    for tis, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), R,
                            facecolor=TISSUE_COLORS[tis],
                            edgecolor="black", linewidth=1.2, zorder=2))
        if tis == "CSF":
            ax.text(x, y + R + 0.04, tis, ha="center", va="bottom",
                    fontsize=9, fontweight="bold",
                    color=TISSUE_COLORS[tis])
        elif tis == "PBMC":
            ax.text(x - R - 0.03, y - 0.02, tis, ha="right", va="top",
                    fontsize=9, fontweight="bold",
                    color=TISSUE_COLORS[tis])
        else:
            ax.text(x + R + 0.03, y - 0.02, tis, ha="left", va="top",
                    fontsize=9, fontweight="bold",
                    color=TISSUE_COLORS[tis])

    handles = [
        Patch(facecolor=LINEAGE_COLORS["CD8"], edgecolor="none", label="CD8"),
        Patch(facecolor=LINEAGE_COLORS["CD4"], edgecolor="none", label="CD4"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.0, 1.0),
                    fontsize=7, frameon=False,
                    handlelength=1.0, handleheight=1.0,
                    handletextpad=0.4)
    leg.set_zorder(6)


def panel_phenotype_divergence(ax, cosine_df):
    from matplotlib.patches import Patch

    matched = cosine_df[cosine_df["type"] == "matched"]
    pairs_avail = set(matched["tissue_pair"].unique())

    def _key(a, b):
        for cand in (f"{a}_vs_{b}", f"{b}_vs_{a}"):
            if cand in pairs_avail:
                return cand
        raise KeyError(f"No tissue_pair for {a}/{b} in cosine_df")

    csf_tp_key = _key("CSF", "TP")
    pbmc_tp_key = _key("PBMC", "TP")
    csf_sub = matched[matched["tissue_pair"] == csf_tp_key]
    pbmc_sub = matched[matched["tissue_pair"] == pbmc_tp_key]
    med_csf = csf_sub.groupby("phenotype")["cosine_dist"].median()
    sem_csf = csf_sub.groupby("phenotype")["cosine_dist"].sem()
    med_pbmc = pbmc_sub.groupby("phenotype")["cosine_dist"].median()
    sem_pbmc = pbmc_sub.groupby("phenotype")["cosine_dist"].sem()

    phens = [p for p in TCELL_PHENOTYPE_ORDER
             if p in med_csf.index and p in med_pbmc.index]
    diff = pd.Series({p: med_pbmc[p] - med_csf[p] for p in phens})
    phens = list(diff.sort_values(ascending=False).index)

    xs = np.arange(len(phens))
    width = 0.72
    csf_vals = np.array([float(med_csf[p]) for p in phens])
    csf_errs = np.array([float(sem_csf.get(p, 0.0) or 0.0) for p in phens])
    pbmc_vals = np.array([float(med_pbmc[p]) for p in phens])
    pbmc_errs = np.array([float(sem_pbmc.get(p, 0.0) or 0.0) for p in phens])

    ax.bar(xs, csf_vals, width=width,
           color=TISSUE_COLORS["CSF"], edgecolor="black", linewidth=0.4,
           zorder=2)
    ax.errorbar(xs, csf_vals, yerr=csf_errs, fmt="none",
                ecolor="black", elinewidth=0.6, capsize=2.0, zorder=3)
    ax.bar(xs, -pbmc_vals, width=width,
           color=TISSUE_COLORS["PBMC"], edgecolor="black", linewidth=0.4,
           zorder=2)
    ax.errorbar(xs, -pbmc_vals, yerr=pbmc_errs, fmt="none",
                ecolor="black", elinewidth=0.6, capsize=2.0, zorder=3)

    ax.axhline(0, color="black", lw=0.8, zorder=4)
    ymax = float(np.nanmax([
        np.nanmax(csf_vals + csf_errs),
        np.nanmax(pbmc_vals + pbmc_errs),
    ]))
    ax.set_ylim(-ymax * 1.18, ymax * 1.18)
    ax.set_xticks(xs)
    ax.set_xticklabels([TCELL_PHENOTYPE_LABELS.get(p, p) for p in phens],
                       rotation=45, ha="right", fontsize=8)
    for tick, p in zip(ax.get_xticklabels(), phens):
        lin = _lineage_of(p)
        tick.set_color(LINEAGE_COLORS.get(lin, "black"))
        tick.set_fontweight("bold")

    yticks = [t for t in ax.get_yticks()
              if -ymax * 1.18 <= t <= ymax * 1.18]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(t):.2f}" for t in yticks], fontsize=8)
    ax.set_ylabel("Cosine distance to TP", fontsize=9)

    ax.text(0.01, 0.98, "CSF", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold",
            color=TISSUE_COLORS["CSF"])
    ax.text(0.01, 0.02, "PBMC", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, fontweight="bold",
            color=TISSUE_COLORS["PBMC"])

    handles = [
        Patch(facecolor=LINEAGE_COLORS["CD8"], edgecolor="none",
              label="CD8"),
        Patch(facecolor=LINEAGE_COLORS["CD4"], edgecolor="none",
              label="CD4"),
    ]
    leg = ax.legend(handles=handles, loc="upper right",
                    fontsize=7, frameon=False,
                    handlelength=1.0, handleheight=1.0,
                    handletextpad=0.4)
    leg.set_zorder(6)
    _style_axis(ax)


# %%
# =========================================================
# Panel B (legacy) — Cross-tissue clone-sharing heatmap
#   (moved to Figure 1; function retained, no longer wired)
# =========================================================
print("\n[legacy] cross-tissue clone-sharing matrices...")


def _shared_clone_matrix(obs_df, lineage):
    sub = obs_df[obs_df["lineage"] == lineage]
    by_clone_tissue = sub.groupby(["clone_id", "tissue"],
                                    observed=True).size()
    by_clone_tissue = by_clone_tissue[by_clone_tissue > 0]
    clone_to_tissues = (by_clone_tissue.reset_index()
                                       .groupby("clone_id")["tissue"]
                                       .apply(set))
    mat = np.zeros((len(TISSUES), len(TISSUES)), dtype=int)
    for ti, t1 in enumerate(TISSUES):
        for tj, t2 in enumerate(TISSUES):
            mat[ti, tj] = int(
                clone_to_tissues.apply(lambda s, a=t1, b=t2:
                                        a in s and b in s).sum())
    return mat


mat_by_lin = {lin: _shared_clone_matrix(obs, lin) for lin in LINEAGES}

traffic_rows = []
for lin in LINEAGES:
    M = mat_by_lin[lin]
    for ti, t1 in enumerate(TISSUES):
        for tj, t2 in enumerate(TISSUES):
            traffic_rows.append({
                "lineage": lin, "t1": t1, "t2": t2,
                "n_shared_clones": int(M[ti, tj]),
            })
pd.DataFrame(traffic_rows).to_csv(
    OUT_DIR / "traffic_volume.csv", index=False)


def _draw_panel_B_heatmap(fig, ax, cax=None):
    """3×3 clone-sharing heatmap: CD8 upper triangle, CD4 lower triangle.
    Diagonal: CD8 and CD4 counts stacked vertically (bold), light-grey
    background. Log-scale colorbar.

    If ``cax`` is provided the colorbar is drawn there (no shrink of
    ``ax``); otherwise it is attached to ``ax`` via fig.colorbar."""
    n = len(TISSUES)
    M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i < j:
                M[i, j] = mat_by_lin["CD8"][i, j]
            else:
                M[i, j] = mat_by_lin["CD4"][i, j]
    off_max = float(np.nanmax(M)) if np.isfinite(np.nanmax(M)) else 1.0
    norm = plt.Normalize(vmin=0.0, vmax=np.log1p(off_max))
    Mcol = np.log1p(np.nan_to_num(M, nan=0.0))
    im = ax.imshow(Mcol, cmap="YlOrRd", norm=norm, aspect="equal")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor="#f0f0f0",
                                        edgecolor="#bbb",
                                        linewidth=0.5, zorder=2))
                cd8 = mat_by_lin["CD8"][i, j]
                cd4 = mat_by_lin["CD4"][i, j]
                ax.text(j, i,
                         f"CD8 {cd8:,}\nCD4 {cd4:,}",
                         ha="center", va="center",
                         fontsize=7, color="#222",
                         fontweight="bold", linespacing=1.10,
                         zorder=3)
                continue
            v = M[i, j]
            rgba = im.cmap(im.norm(np.log1p(v)))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.5 else "black"
            ax.text(j, i, f"{int(v):,}",
                     ha="center", va="center",
                     fontsize=9, color=tc)

    ax.set_xticks(range(n))
    ax.set_xticklabels(TISSUES, fontsize=10)
    for tick, t in zip(ax.get_xticklabels(), TISSUES):
        tick.set_color(TISSUE_COLORS.get(t, "black"))
        tick.set_fontweight("bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(TISSUES, fontsize=10)
    for tick, t in zip(ax.get_yticklabels(), TISSUES):
        tick.set_color(TISSUE_COLORS.get(t, "black"))
        tick.set_fontweight("bold")
    ax.set_title("Cross-tissue clone sharing\n"
                  "(upper Δ = CD8, lower Δ = CD4)",
                  fontsize=11, pad=6, linespacing=1.2)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    if cax is not None:
        cb = fig.colorbar(im, cax=cax)
    else:
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("# shared clones (log scale)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    raw_ticks = sorted({0, 100, 1000, int(off_max)})
    cb.set_ticks([np.log1p(v) for v in raw_ticks])
    cb.set_ticklabels([f"{v:,}" for v in raw_ticks])


# %%
# =========================================================
# Panel C — Resident vs Migratory phenotype enrichment
# =========================================================
print("\nPanel C: Resident vs Migratory phenotype enrichment...")
# Resident  : in_src AND NOT in_other  (old STAYER_ONLY)
# Migratory : in_other                 (old MOVER_ONLY + BOTH merged)

cat_records = []
for _, row in ph.iterrows():
    n_src = int(row["_n"])
    if n_src < MIN_N_SRC:
        continue
    t_src = str(row["timepoint"])
    next_tp = NEXT_TP.get(t_src)
    if next_tp is None:
        continue
    pat = row["patient"]; trb = row["trb"]; src = row["tissue"]
    in_src = (pat, trb, src, next_tp) in present_set
    in_other = any((pat, trb, j, next_tp) in present_set
                   for j in TISSUES if j != src)
    if not (in_src or in_other):
        continue
    cat = "MIGRATORY" if in_other else "RESIDENT"
    frac_vec = row[PHENOTYPES].to_numpy(dtype=float)
    fs = frac_vec.sum()
    if fs > 0:
        frac_vec = frac_vec / fs
    rec = {
        "patient": pat, "trb": trb,
        "src_tissue": src, "t_src": t_src, "t_dst": next_tp,
        "category": cat, "n_src": n_src,
    }
    for k, p in enumerate(PHENOTYPES):
        rec[p] = float(frac_vec[k])
    cat_records.append(rec)
cat_df = pd.DataFrame(cat_records)
print(f"  classified {len(cat_df)} (clone, t_src) entries; "
      f"counts: {cat_df['category'].value_counts().to_dict()}")

# Per-(tissue, category) cell-weighted mean phenotype fractions.
def _cell_weighted_mean(df_sub, phens):
    if df_sub.empty:
        return np.zeros(len(phens))
    w = df_sub["n_src"].to_numpy(dtype=float)
    f = df_sub[phens].to_numpy(dtype=float)
    return (f * w[:, None]).sum(axis=0) / max(w.sum(), 1.0)


# Two-segment summary (replaces old 3-segment stack).
restack_rows = []
for tissue in TISSUES:
    sub = cat_df[cat_df["src_tissue"] == tissue]
    total = max(int(len(sub)), 1)
    n_res = int((sub["category"] == "RESIDENT").sum())
    n_mig = int((sub["category"] == "MIGRATORY").sum())
    restack_rows.append({
        "tissue": tissue,
        "n_resident": n_res, "n_migratory": n_mig,
        "n_total": int(len(sub)),
        "frac_resident": n_res / total,
        "frac_migratory": n_mig / total,
    })
restack_df = pd.DataFrame(restack_rows)
restack_df.to_csv(OUT_DIR / "resident_migratory_counts.csv", index=False)
print(f"  wrote resident_migratory_counts.csv\n"
      f"{restack_df.to_string(index=False)}")


# Per-(tissue, phenotype) cell-weighted Δ = Migratory − Resident, plus
# mixed-effects logistic regression p-values (BBMGLM with patient as
# random intercept; cluster-robust Logit fallback). BH-FDR across the
# 33 tests.
print("  fitting per-(tissue, phenotype) Migratory-vs-Resident models...")
delta_rows = []
stats_rows = []
for tissue in TISSUES:
    sub_t = cat_df[cat_df["src_tissue"] == tissue]
    mig = sub_t[sub_t["category"] == "MIGRATORY"]
    res = sub_t[sub_t["category"] == "RESIDENT"]
    cw_mig = _cell_weighted_mean(mig, PHENOTYPES)
    cw_res = _cell_weighted_mean(res, PHENOTYPES)
    for k, p in enumerate(PHENOTYPES):
        delta_rows.append({
            "src_tissue": tissue, "phenotype": p,
            "mean_frac_resident": float(cw_res[k]),
            "mean_frac_migratory": float(cw_mig[k]),
            "delta_mig_minus_res": float(cw_mig[k] - cw_res[k]),
            "n_resident": int(len(res)),
            "n_migratory": int(len(mig)),
        })

    if sub_t.empty:
        for p in PHENOTYPES:
            stats_rows.append({
                "src_tissue": tissue, "phenotype": p,
                "beta": np.nan, "beta_se": np.nan, "beta_p": np.nan,
                "n_migratory": 0, "n_resident": 0,
                "convergence_status": "no_data",
            })
        continue
    sub_t = sub_t.copy()
    sub_t["is_mig"] = (sub_t["category"] == "MIGRATORY").astype(int)
    n_mig_t = int((sub_t["is_mig"] == 1).sum())
    n_res_t = int((sub_t["is_mig"] == 0).sum())
    patient_codes, _ = pd.factorize(sub_t["patient"].astype(str))
    n_pat = int(patient_codes.max()) + 1 if len(patient_codes) else 0
    exog_vc = np.zeros((len(sub_t), n_pat))
    for ix, pc in enumerate(patient_codes):
        exog_vc[ix, pc] = 1.0
    ident = np.zeros(n_pat, dtype=int)
    for p in PHENOTYPES:
        x = sub_t[p].to_numpy(dtype=float)
        if x.std() == 0 or n_mig_t < 3 or n_res_t < 3:
            stats_rows.append({
                "src_tissue": tissue, "phenotype": p,
                "beta": np.nan, "beta_se": np.nan, "beta_p": np.nan,
                "n_migratory": n_mig_t, "n_resident": n_res_t,
                "convergence_status": "insufficient_variance_or_n",
            })
            continue
        X = np.column_stack([np.ones_like(x), x])
        y = sub_t["is_mig"].to_numpy(dtype=float)
        beta = beta_se = pval = np.nan
        conv = "failed"
        try:
            md = sm.BinomialBayesMixedGLM(
                y, X, exog_vc=exog_vc, ident=ident,
            )
            res_fit = md.fit_vb()
            beta = float(res_fit.fe_mean[1])
            beta_se = float(res_fit.fe_sd[1])
            if beta_se > 0:
                z = beta / beta_se
                pval = float(2.0 * (1.0 - sm.distributions.norm.cdf(
                    abs(z))))
            conv = "bbmglm_ok"
        except Exception:
            try:
                fit = sm.Logit(y, X).fit(
                    disp=0, cov_type="cluster",
                    cov_kwds={"groups": sub_t["patient"].astype(str).values},
                )
                beta = float(fit.params[1])
                beta_se = float(fit.bse[1])
                pval = float(fit.pvalues[1])
                conv = "cluster_robust_ok"
            except Exception as e:
                conv = f"failed: {type(e).__name__}"
        stats_rows.append({
            "src_tissue": tissue, "phenotype": p,
            "beta": beta, "beta_se": beta_se, "beta_p": pval,
            "n_migratory": n_mig_t, "n_resident": n_res_t,
            "convergence_status": conv,
        })

delta_df = pd.DataFrame(delta_rows)
stats_df = pd.DataFrame(stats_rows)
valid = stats_df["beta_p"].notna()
qvals = np.full(len(stats_df), np.nan)
if int(valid.sum()) > 0:
    _, q_flat, _, _ = multipletests(
        stats_df.loc[valid, "beta_p"].values, method="fdr_bh")
    qvals[valid.to_numpy()] = q_flat
stats_df["beta_q"] = qvals
delta_df.to_csv(OUT_DIR / "resident_migratory_delta.csv", index=False)
stats_df.to_csv(OUT_DIR / "resident_migratory_stats.csv", index=False)
print("  wrote resident_migratory_delta.csv, resident_migratory_stats.csv")

# Shared x-limit across all three tissue panels for comparability.
ENRICH_XLIM = max(
    0.10,
    float(np.nanmax(np.abs(delta_df["delta_mig_minus_res"]))) * 1.10
)


def _draw_resmig_bars(axb):
    xs = np.arange(len(TISSUES))
    width = 0.55
    xtick_labels = []
    for ti, tissue in enumerate(TISSUES):
        row = restack_df[restack_df["tissue"] == tissue].iloc[0]
        r, m = (row["frac_resident"], row["frac_migratory"])
        nr, nm = (int(row["n_resident"]), int(row["n_migratory"]))
        axb.bar(ti, r, width, color=COL_RESIDENT,
                 edgecolor=TISSUE_COLORS.get(tissue, "black"),
                 linewidth=1.0)
        axb.bar(ti, m, width, bottom=r, color=COL_MIGRATORY,
                 edgecolor=TISSUE_COLORS.get(tissue, "black"),
                 linewidth=1.0)
        for frac, n, base in [(r, nr, 0), (m, nm, r)]:
            if frac >= 0.05:
                axb.text(ti, base + frac / 2,
                          f"{n}\n{frac*100:.0f}%",
                          ha="center", va="center", fontsize=8,
                          color="white", fontweight="bold")
        xtick_labels.append(f"{tissue}\n(n={int(row['n_total'])})")
    axb.set_xticks(xs)
    axb.set_xticklabels(xtick_labels, fontsize=8, linespacing=1.1)
    for tick, t in zip(axb.get_xticklabels(), TISSUES):
        tick.set_color(TISSUE_COLORS.get(t, "black"))
        tick.set_fontweight("bold")
    axb.set_ylabel("Fraction (clone, t_src)", fontsize=9)
    axb.set_ylim(0, 1.05)
    axb.set_title("Category", fontsize=10, pad=6)
    _style_axis(axb)
    from matplotlib.patches import Patch
    axb.legend(handles=[
        Patch(facecolor=COL_RESIDENT,  label="Resident"),
        Patch(facecolor=COL_MIGRATORY, label="Migratory"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.30),
       ncol=1, fontsize=8, frameon=False)


def _draw_resmig_enrichment(axp, tissue, show_yticklabels, xlim=None):
    if xlim is None:
        xlim = ENRICH_XLIM
    sub_d = delta_df[delta_df["src_tissue"] == tissue]
    sub_s = stats_df[stats_df["src_tissue"] == tissue].set_index(
        "phenotype")
    tip_gap = xlim * 0.025
    for yi, p in enumerate(PHENOTYPES):
        d_row = sub_d[sub_d["phenotype"] == p]
        if d_row.empty:
            continue
        d = float(d_row.iloc[0]["delta_mig_minus_res"])
        c = TCELL_PHENOTYPE_COLORS.get(p, "#888")
        axp.barh(yi, d, color=c, edgecolor="black",
                  linewidth=0.4, alpha=0.95, height=0.78)
        q = sub_s.loc[p, "beta_q"] if p in sub_s.index else np.nan
        if not np.isnan(q) and q < FDR_THRESHOLD:
            if d >= 0:
                ax_x = min(d + tip_gap, xlim - tip_gap * 0.5)
                axp.text(ax_x, yi, "*", fontsize=12, fontweight="bold",
                          color="black", ha="left", va="center")
            else:
                ax_x = max(d - tip_gap, -xlim + tip_gap * 0.5)
                axp.text(ax_x, yi, "*", fontsize=12, fontweight="bold",
                          color="black", ha="right", va="center")
    axp.axvline(0, color="#444", lw=1.0, alpha=0.9)
    axp.set_yticks(range(K_PHEN))
    if show_yticklabels:
        axp.set_yticklabels(
            [TCELL_PHENOTYPE_LABELS.get(p, p) for p in PHENOTYPES],
            fontsize=9)
        for tick, p in zip(axp.get_yticklabels(), PHENOTYPES):
            tick.set_color(TCELL_PHENOTYPE_COLORS.get(p, "black"))
            tick.set_fontweight("bold")
    else:
        axp.set_yticklabels([])
    axp.invert_yaxis()
    axp.set_xlim(-xlim, xlim)
    axp.set_xlabel("Δ migratory − resident", fontsize=9)
    axp.set_title(tissue, fontsize=11, fontweight="bold",
                   color=TISSUE_COLORS.get(tissue, "black"), pad=4)
    _style_axis(axp)
    axp.tick_params(axis="x", labelsize=8)


# %%
# =========================================================
# Panel D — Ternary persistence plot
# =========================================================
print("\nPanel D: ternary persistence plot...")
# For each clone observed at ≥2 (tissue, timepoint) bins, persistence
# vector = (fraction of bins in PBMC, in CSF, in TP).

bins_per_clone = (
    obs.groupby(["clone_id", "tissue", "timepoint"], observed=True)
       .size()
       .reset_index(name="n")
)
clone_bin_counts = (
    bins_per_clone.groupby(["clone_id", "tissue"], observed=True)
                  .size().unstack("tissue", fill_value=0)
)
for t in TISSUES:
    if t not in clone_bin_counts.columns:
        clone_bin_counts[t] = 0
clone_bin_counts = clone_bin_counts[TISSUES]
clone_bin_counts["_total"] = clone_bin_counts.sum(axis=1)
multi_bin = clone_bin_counts[clone_bin_counts["_total"] >= 2].copy()
for t in TISSUES:
    multi_bin[f"frac_{t}"] = multi_bin[t] / multi_bin["_total"]
print(f"  {len(multi_bin):,} clones with ≥2 (tissue, timepoint) bins")

# Dominant phenotype + dominant lineage per clone (for color).
clone_phen = (
    obs.groupby(["clone_id", "phenotype"], observed=True).size()
       .unstack("phenotype", fill_value=0)
)
for p in PHENOTYPES:
    if p not in clone_phen.columns:
        clone_phen[p] = 0
clone_phen = clone_phen[PHENOTYPES]
dominant_phen = clone_phen.idxmax(axis=1)
dominant_lineage = dominant_phen.map(_lineage_of)
multi_bin = multi_bin.join(
    dominant_phen.rename("dominant_phenotype"))
multi_bin = multi_bin.join(
    dominant_lineage.rename("dominant_lineage"))
persistence_csv = multi_bin.reset_index()[
    ["clone_id"] + TISSUES + ["_total",
                              "frac_PBMC", "frac_CSF", "frac_TP",
                              "dominant_phenotype", "dominant_lineage"]
].rename(columns={"_total": "n_bins"})
persistence_csv.to_csv(OUT_DIR / "persistence_vectors.csv", index=False)
print("  wrote persistence_vectors.csv")

# Median persistence per tissue, per lineage (for corner annotations).
median_persistence_per_lineage = {}
for lin in LINEAGES:
    sub = multi_bin[multi_bin["dominant_lineage"] == lin]
    median_persistence_per_lineage[lin] = {
        t: float(sub[f"frac_{t}"].median()) if not sub.empty else float("nan")
        for t in TISSUES
    }


# Barycentric→cartesian (PBMC=top, CSF=left, TP=right).
SQRT3_2 = np.sqrt(3) / 2
TRI_TOP   = np.array([0.5, SQRT3_2])  # PBMC
TRI_LEFT  = np.array([0.0, 0.0])      # CSF
TRI_RIGHT = np.array([1.0, 0.0])      # TP


def _bary_to_xy(a, b, c):
    """a=PBMC, b=CSF, c=TP. Returns (x, y) arrays."""
    a = np.asarray(a); b = np.asarray(b); c = np.asarray(c)
    x = a * TRI_TOP[0] + b * TRI_LEFT[0] + c * TRI_RIGHT[0]
    y = a * TRI_TOP[1] + b * TRI_LEFT[1] + c * TRI_RIGHT[1]
    return x, y


def _draw_ternary_single(ax):
    """Single ternary axes containing all clones with ≥2 bins. Points
    colored by lineage (CD8 / CD4). Background = filled gray KDE with
    4 progressively-opaque levels."""
    sub = multi_bin
    # Tight bounds — corners + a thin label rim.
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.08, SQRT3_2 + 0.06)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)

    tri = plt.Polygon([TRI_TOP, TRI_LEFT, TRI_RIGHT],
                       fill=False, edgecolor="#444", linewidth=1.0,
                       zorder=3)
    ax.add_patch(tri)
    for frac in (0.25, 0.5, 0.75):
        for axis in range(3):
            p1 = np.zeros(3); p2 = np.zeros(3)
            p1[axis] = frac; p2[axis] = frac
            others = [k for k in range(3) if k != axis]
            p1[others[0]] = 1 - frac
            p2[others[1]] = 1 - frac
            x1, y1 = _bary_to_xy(*p1)
            x2, y2 = _bary_to_xy(*p2)
            ax.plot([x1, x2], [y1, y2], color="#e8e8e8",
                    lw=0.5, zorder=1)
    if sub.empty:
        return

    a = sub["frac_PBMC"].to_numpy().astype(float).copy()
    b = sub["frac_CSF"].to_numpy().astype(float).copy()
    c = sub["frac_TP"].to_numpy().astype(float).copy()
    # Visualization-only jitter: clones with CSF persistence == 0 sit
    # exactly on the PBMC↔TP edge and stack as a solid line. Nudge a
    # small random amount into the CSF direction, then renormalize.
    zero_csf = (b == 0)
    n_zero = int(zero_csf.sum())
    if n_zero > 0:
        rng_jit = np.random.default_rng(42)
        b[zero_csf] = rng_jit.uniform(0.0, 0.03, n_zero)
        total = a + b + c
        a = a / total
        b = b / total
        c = c / total
    xs, ys = _bary_to_xy(a, b, c)

    # Filled gray KDE — 4 levels with progressively higher alpha so
    # the densest core reads darkest (stacked layers compound).
    if len(xs) >= 8 and (xs.std() > 1e-6 or ys.std() > 1e-6):
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
            gx = np.linspace(0.0, 1.0, 160)
            gy = np.linspace(0.0, SQRT3_2, 160)
            GX, GY = np.meshgrid(gx, gy)
            A_grid = GY / SQRT3_2
            C_grid = GX - 0.5 * A_grid
            B_grid = 1 - A_grid - C_grid
            mask = (A_grid >= 0) & (B_grid >= 0) & (C_grid >= 0)
            Z = kde(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            Z = np.where(mask, Z, np.nan)
            zmax = float(np.nanmax(Z))
            # Inner-most quartile is the densest core.
            level_qs = [0.50, 0.70, 0.85, 0.95]
            alphas = [0.05, 0.15, 0.30, 0.50]
            for q, a_ in zip(level_qs, alphas):
                thresh = float(np.nanquantile(Z, q))
                ax.contourf(GX, GY, Z,
                            levels=[thresh, zmax + 1e-9],
                            colors=["#666"], alpha=a_, zorder=2)
        except Exception:
            pass

    # Lineage-colored scatter (use dominant_lineage per clone).
    colors = sub["dominant_lineage"].map(
        lambda l: LINEAGE_COLORS.get(l, "#888")
    ).to_numpy()
    ax.scatter(xs, ys, s=18, c=colors, alpha=1.0,
                edgecolors="none", rasterized=True, zorder=4)

    # Corner tissue labels.
    ax.text(TRI_TOP[0], TRI_TOP[1] + 0.020, "PBMC",
             ha="center", va="bottom", fontsize=9, fontweight="bold",
             color=TISSUE_COLORS["PBMC"])
    ax.text(TRI_LEFT[0] - 0.005, TRI_LEFT[1] - 0.015, "CSF",
             ha="right", va="top", fontsize=9, fontweight="bold",
             color=TISSUE_COLORS["CSF"])
    ax.text(TRI_RIGHT[0] + 0.005, TRI_RIGHT[1] - 0.015, "TP",
             ha="left", va="top", fontsize=9, fontweight="bold",
             color=TISSUE_COLORS["TP"])

    # Lineage legend overlaid in upper-left (the empty wedge of the
    # bounding box, outside the triangle).
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=LINEAGE_COLORS["CD8"],
                      edgecolor="none", label="CD8"),
               Patch(facecolor=LINEAGE_COLORS["CD4"],
                      edgecolor="none", label="CD4")]
    leg = ax.legend(handles=handles, title="Lineage",
                     loc="upper left",
                     bbox_to_anchor=(0.0, 0.95),
                     fontsize=8, title_fontsize=9,
                     frameon=False, handlelength=1.0,
                     handleheight=1.0, handletextpad=0.4)
    leg.set_zorder(7)


# %%
# =========================================================
# Panel E — Within-tissue phenotypic flux (L1 distance)
# =========================================================
print("\nPanel E: within-tissue phenotypic flux (L1)...")

# Per-(patient, trb, tissue, t) phenotype vector + count.
key_to_vec = {}
for _, row in ph.iterrows():
    n = int(row["_n"])
    if n <= 0:
        continue
    vec = row[PHENOTYPES].to_numpy(dtype=float)
    s = vec.sum()
    if s <= 0:
        continue
    vec = vec / s
    key_to_vec[(row["patient"], row["trb"], row["tissue"],
                str(row["timepoint"]))] = (vec, n)

# Resident clones for flux: same tissue at both endpoints, ≥3 cells at
# each, absent from every other tissue at t+1.
flux_rows = []
for (pat, trb, tis, t), (vec_t, n_t) in key_to_vec.items():
    t_next = NEXT_TP.get(t)
    if t_next is None:
        continue
    if n_t < MIN_CELLS_BOTH:
        continue
    key_next = (pat, trb, tis, t_next)
    if key_next not in key_to_vec:
        continue
    vec_t1, n_t1 = key_to_vec[key_next]
    if n_t1 < MIN_CELLS_BOTH:
        continue
    if any((pat, trb, j, t_next) in present_set
            for j in TISSUES if j != tis):
        continue
    l1 = float(np.abs(vec_t - vec_t1).sum())
    flux_rows.append({
        "tissue": tis, "clone_id": f"{pat}|{trb}",
        "patient": pat, "trb": trb,
        "t_src": t, "t_dst": t_next,
        "l1_distance": l1,
        "n_cells_t": int(n_t), "n_cells_t1": int(n_t1),
    })
flux_df = pd.DataFrame(flux_rows)
flux_df.to_csv(OUT_DIR / "phenotypic_flux.csv", index=False)
print(f"  {len(flux_df)} resident clone-transitions; "
      "wrote phenotypic_flux.csv")

# Pairwise Mann-Whitney U on raw branch-level L1 values, BH-corrected.
flux_pair_rows = []
flux_pairs = [("PBMC", "CSF"), ("PBMC", "TP"), ("CSF", "TP")]
for t1, t2 in flux_pairs:
    a_vals = flux_df.loc[flux_df["tissue"] == t1, "l1_distance"].to_numpy()
    b_vals = flux_df.loc[flux_df["tissue"] == t2, "l1_distance"].to_numpy()
    if len(a_vals) < 3 or len(b_vals) < 3:
        flux_pair_rows.append({
            "tissue_1": t1, "tissue_2": t2,
            "U_stat": np.nan, "p_raw": np.nan,
            "n_1": int(len(a_vals)), "n_2": int(len(b_vals)),
        })
        continue
    u_stat, p_raw = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    flux_pair_rows.append({
        "tissue_1": t1, "tissue_2": t2,
        "U_stat": float(u_stat), "p_raw": float(p_raw),
        "n_1": int(len(a_vals)), "n_2": int(len(b_vals)),
    })
flux_pair_df = pd.DataFrame(flux_pair_rows)
valid = flux_pair_df["p_raw"].notna()
if int(valid.sum()) > 0:
    _, qvals, _, _ = multipletests(
        flux_pair_df.loc[valid, "p_raw"].values, method="fdr_bh")
    bh = np.full(len(flux_pair_df), np.nan)
    bh[valid.to_numpy()] = qvals
    flux_pair_df["p_bh"] = bh
else:
    flux_pair_df["p_bh"] = np.nan
flux_pair_df = flux_pair_df[
    ["tissue_1", "tissue_2", "U_stat", "p_raw", "p_bh", "n_1", "n_2"]
]
flux_pair_df.to_csv(OUT_DIR / "phenotypic_flux_stats.csv", index=False)
print(f"  wrote phenotypic_flux_stats.csv\n"
      f"{flux_pair_df.to_string(index=False)}")


def _sig_marker(p):
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


FLUX_SIG = {(r["tissue_1"], r["tissue_2"]): _sig_marker(r["p_bh"])
            for _, r in flux_pair_df.iterrows()}

# Null L1: permute phenotype labels within (tissue, timepoint) by
# sampling each clone's count vector from Multinomial(n, p_marg). Take
# the mean over N_PERM permutations.
pheno_marg = {}
for tis in TISSUES:
    for t in sorted(obs["timepoint"].unique()):
        sub = obs[(obs["tissue"] == tis) & (obs["timepoint"] == t)]
        if sub.empty:
            continue
        cnts = sub["phenotype"].value_counts()
        v = np.zeros(K_PHEN)
        for p, c in cnts.items():
            if p in PHENO_IDX:
                v[PHENO_IDX[p]] = float(c)
        if v.sum() > 0:
            pheno_marg[(tis, t)] = v / v.sum()

print(f"  computing L1 null (N_PERM={N_PERM})...")
rng_null = np.random.default_rng(0)
null_l1_by_tissue = {}
for tis in TISSUES:
    sub = flux_df[flux_df["tissue"] == tis]
    if sub.empty:
        null_l1_by_tissue[tis] = float("nan")
        continue
    n_t_arr = sub["n_cells_t"].to_numpy()
    n_t1_arr = sub["n_cells_t1"].to_numpy()
    p_t_list = [pheno_marg.get((tis, ts)) for ts in sub["t_src"]]
    p_t1_list = [pheno_marg.get((tis, td)) for td in sub["t_dst"]]
    valid_mask = np.array([p0 is not None and p1 is not None
                           for p0, p1 in zip(p_t_list, p_t1_list)])
    if not valid_mask.any():
        null_l1_by_tissue[tis] = float("nan")
        continue
    n_t_arr = n_t_arr[valid_mask]
    n_t1_arr = n_t1_arr[valid_mask]
    p_t_arr = np.stack([p for p in p_t_list if p is not None])
    p_t1_arr = np.stack([p for p in p_t1_list if p is not None])

    perm_means = []
    for _ in range(N_PERM):
        v0 = np.array([rng_null.multinomial(int(n), p) / float(n)
                       for n, p in zip(n_t_arr, p_t_arr)])
        v1 = np.array([rng_null.multinomial(int(n), p) / float(n)
                       for n, p in zip(n_t1_arr, p_t1_arr)])
        l1_vals = np.abs(v0 - v1).sum(axis=1)
        perm_means.append(float(np.mean(l1_vals)))
    null_l1_by_tissue[tis] = float(np.nanmean(perm_means))
print(f"  null L1 per tissue: "
      f"{ {k: round(v, 3) for k, v in null_l1_by_tissue.items()} }")

pat_l1_means = {tis: {} for tis in TISSUES}
for (tis, pat), grp in flux_df.groupby(["tissue", "patient"],
                                        observed=True):
    if len(grp) >= 5:
        pat_l1_means[tis][pat] = float(grp["l1_distance"].mean())


def _draw_panel_E(axE):
    """Horizontal boxplot per tissue (filled with tissue color) with a
    swarm-style scatter overlaid on top. Pairwise Mann-Whitney
    brackets to the right of the boxes."""
    from matplotlib.colors import to_rgba

    data_by_tissue = []
    for tis in TISSUES:
        vals = flux_df.loc[flux_df["tissue"] == tis,
                            "l1_distance"].dropna().to_numpy()
        data_by_tissue.append(np.clip(vals, 0.0, 2.0))

    positions = np.arange(len(TISSUES))

    # Boxplot first — soft tissue-colored fill, solid tissue edges.
    bp = axE.boxplot(
        data_by_tissue, positions=positions, vert=False,
        widths=0.55, patch_artist=True, showfliers=False,
        zorder=2,
    )
    for i, tis in enumerate(TISSUES):
        col = TISSUE_COLORS.get(tis, "black")
        bp["boxes"][i].set_facecolor(to_rgba(col, 0.35))
        bp["boxes"][i].set_edgecolor(col)
        bp["boxes"][i].set_linewidth(1.0)
        for w in bp["whiskers"][2 * i:2 * i + 2]:
            w.set_color(col); w.set_linewidth(0.9)
        for cap in bp["caps"][2 * i:2 * i + 2]:
            cap.set_color(col); cap.set_linewidth(0.9)
        bp["medians"][i].set_color(col)
        bp["medians"][i].set_linewidth(1.4)

    # Swarm overlaid on top of the boxes.
    rng_swarm = np.random.default_rng(0)
    for ti, (tis, vals) in enumerate(zip(TISSUES, data_by_tissue)):
        if len(vals) == 0:
            continue
        ys = ti + rng_swarm.uniform(-0.22, 0.22, size=len(vals))
        axE.scatter(vals, ys, s=4,
                     color=TISSUE_COLORS.get(tis, "gray"),
                     alpha=0.55, edgecolors="none",
                     rasterized=True, zorder=4)

    # Pairwise significance brackets — to the right of the boxes.
    bracket_specs = [
        (0, 1, 2.10, FLUX_SIG.get(("PBMC", "CSF"), "ns")),
        (1, 2, 2.10, FLUX_SIG.get(("CSF",  "TP"),  "ns")),
        (0, 2, 2.42, FLUX_SIG.get(("PBMC", "TP"),  "ns")),
    ]
    tick_dx = 0.05
    for y1, y2, x, sig in bracket_specs:
        axE.plot([x - tick_dx, x, x, x - tick_dx],
                  [y1, y1, y2, y2],
                  color="black", lw=0.8, clip_on=False, zorder=5)
        fs = 10 if sig != "ns" else 8
        axE.text(x + 0.04, (y1 + y2) / 2, sig,
                  ha="left", va="center",
                  fontsize=fs, fontweight="bold" if sig != "ns" else "normal",
                  clip_on=False, zorder=5)

    axE.set_xlim(0, 2.65)
    axE.set_ylim(-0.6, len(TISSUES) - 0.4)
    axE.set_yticks(positions)
    axE.set_yticklabels(TISSUES, fontsize=10)
    for tick, t in zip(axE.get_yticklabels(), TISSUES):
        tick.set_color(TISSUE_COLORS.get(t, "black"))
        tick.set_fontweight("bold")
    axE.set_xlabel("L1 distance", fontsize=9)
    axE.set_title("Within-tissue\nPhenotypic Flux",
                   fontsize=10, pad=4, linespacing=1.15)
    axE.invert_yaxis()
    _style_axis(axE)
    axE.tick_params(axis="x", labelsize=8)
    # Keep the visible x-axis spine to the data range; brackets sit
    # outside it.
    axE.spines["bottom"].set_bounds(0, 2.0)
    axE.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])


# %%
# =========================================================
# Panel F — Migration flow network triangle
# =========================================================
print("\nPanel F: migration flow network...")
mig_df_all = pd.read_csv(MIG_CSV)
mig_opt = mig_df_all[mig_df_all["method"] == "optimizer"]
mean_rate = (mig_opt.groupby(["src", "dst"], observed=True)["rate"]
                   .mean().to_dict())
P_emp = pd.read_csv(P_EMP_CSV, index_col=0)


def _block_retention(P, tissues, phenotypes):
    out = {}
    for tis in tissues:
        idx = [f"{tis}__{p}" for p in phenotypes
                if f"{tis}__{p}" in P.index]
        if not idx:
            out[tis] = float("nan"); continue
        block_sum = float(P.loc[idx, idx].values.sum())
        row_total = float(P.loc[idx, :].values.sum())
        out[tis] = block_sum / row_total if row_total > 0 else float("nan")
    return out


block_ret = _block_retention(P_emp, TISSUES, PHENOTYPES)

NODE_POS = {"PBMC": (-0.8, 0.7), "TP": (0.8, 0.7), "CSF": (0.0, -0.7)}
NODE_RADIUS = 0.30
LW_MIN, LW_MAX = 1.5, 6.0


def _draw_flow_network(ax):
    mean_rate_arr = np.array(list(mean_rate.values()))
    rate_max = (float(np.abs(mean_rate_arr).max())
                if mean_rate_arr.size else 1.0)
    # Tight bounds — just enough to fit nodes (r=0.30) plus a thin
    # rim for the bowed arrows + rate labels.
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.15, 1.10)
    ax.set_aspect("equal"); ax.axis("off")

    node_text_objs = []
    for tis, (x, y) in NODE_POS.items():
        ax.add_patch(Circle((x, y), NODE_RADIUS,
                             facecolor=TISSUE_COLORS[tis],
                             edgecolor="black", linewidth=1.4, zorder=3))
        ret = block_ret.get(tis, float("nan"))
        if not np.isnan(ret):
            label = (f"$\\bf{{{tis}}}$\n"
                     f"$\\mathit{{{ret * 100:.0f}\\%}}$")
        else:
            label = f"$\\bf{{{tis}}}$"
        t = ax.text(x, y, label, ha="center", va="center",
                     fontsize=11, color="black",
                     multialignment="center", linespacing=1.25,
                     zorder=4)
        node_text_objs.append(t)

    rate_texts = []
    rate_anchors_x, rate_anchors_y = [], []
    for (a, b), rate in mean_rate.items():
        x1, y1 = NODE_POS[a]; x2, y2 = NODE_POS[b]
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        if L == 0: continue
        ux, uy = dx / L, dy / L
        sx, sy = x1 + ux * NODE_RADIUS, y1 + uy * NODE_RADIUS
        ex, ey = x2 - ux * NODE_RADIUS, y2 - uy * NODE_RADIUS
        color = TISSUE_COLORS[b]
        lw = LW_MIN + (LW_MAX - LW_MIN) * (
            abs(rate) / max(rate_max, 1e-9))
        arr = FancyArrowPatch(
            posA=(sx, sy), posB=(ex, ey),
            connectionstyle="arc3,rad=0.18",
            arrowstyle="-|>", mutation_scale=14,
            color=color, linewidth=float(lw), zorder=2,
        )
        ax.add_patch(arr)
        # Perpendicular offset along (uy, -ux) — the correct
        # right-hand normal for the arc3,rad=0.18 bow direction.
        nx_, ny_ = uy, -ux
        mx = (sx + ex) / 2 + nx_ * 0.20
        my = (sy + ey) / 2 + ny_ * 0.20
        t = ax.text(mx, my, f"{rate:.2f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color="black",
                     bbox=dict(boxstyle="round,pad=0.25",
                                facecolor="white", edgecolor="none",
                                alpha=0.88),
                     zorder=5)
        rate_texts.append(t)
        rate_anchors_x.append(mx)
        rate_anchors_y.append(my)

    if HAS_ADJUST_TEXT and rate_texts:
        try:
            _adjust_text_fn(
                rate_texts, ax=ax,
                x=rate_anchors_x, y=rate_anchors_y,
                objects=node_text_objs,
                only_move={"text": "xy"},
                expand_text=(1.10, 1.25),
                force_text=(0.30, 0.40), force_static=(0.40, 0.40),
                arrowprops=None,
            )
        except TypeError:
            try:
                _adjust_text_fn(
                    rate_texts, ax=ax,
                    x=rate_anchors_x, y=rate_anchors_y,
                    only_move={"text": "xy"},
                    expand_text=(1.10, 1.25),
                    force_text=(0.30, 0.40),
                    arrowprops=None,
                )
            except Exception as e:
                print(f"  WARNING: adjustText nudge failed: {e}")
        except Exception as e:
            print(f"  WARNING: adjustText nudge failed: {e}")


# %%
# =========================================================
# Panel G — Tissue retention over time
# =========================================================
print("\nPanel G: tissue retention over time...")
ret_ts_df = pd.read_csv(RETENTION_TS_CSV)
# Order transitions naturally: 1_to_2, 2_to_3, ...
trans_order = sorted(ret_ts_df["transition"].unique(),
                      key=lambda s: int(s.split("_")[0]))
trans_pretty = {s: s.replace("_to_", "→") for s in trans_order}


def _draw_panel_G(axG):
    xs = np.arange(len(trans_order))
    for tis in TISSUES:
        sub = ret_ts_df[ret_ts_df["tissue"] == tis]
        ys = []
        for tr in trans_order:
            row = sub[sub["transition"] == tr]
            if row.empty:
                ys.append(np.nan)
            else:
                ys.append(float(row.iloc[0]["fraction_retained"]))
        axG.plot(xs, ys, color=TISSUE_COLORS.get(tis, "black"),
                  lw=2.0, marker="o", markersize=7,
                  markeredgecolor="black", markeredgewidth=0.6,
                  label=tis, zorder=3)
    axG.set_xticks(xs)
    axG.set_xticklabels([trans_pretty[t] for t in trans_order],
                         fontsize=9)
    axG.set_ylim(0, 1.0)
    axG.set_ylabel("Clonal retention rate", fontsize=10)
    axG.set_xlabel("Timepoint transition", fontsize=10)
    axG.set_title("Tissue retention over time",
                   fontsize=11, pad=6)
    axG.grid(axis="y", alpha=0.25, linewidth=0.5)
    leg = axG.legend(loc="lower left", fontsize=9, frameon=False,
                      ncol=3, columnspacing=1.2, handlelength=1.4)
    for txt, tis in zip(leg.get_texts(), TISSUES):
        txt.set_color(TISSUE_COLORS.get(tis, "black"))
        txt.set_fontweight("bold")
    _style_axis(axG)


# %%
# =========================================================
# Panel I — Pathway temporal stability across timepoints
# =========================================================
print("\nPanel I: pathway temporal stability across timepoints...")

import time as _time_I  # noqa: E402
import anndata as _ad_I  # noqa: E402
import decoupler as _dc_I  # noqa: E402
import gseapy as _gp_I  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from tqdm import tqdm as _tqdm_I  # noqa: E402

from modules.style import PATHWAY_FAMILY_COLORS  # noqa: E402

PANEL_I_OUT = OUT_DIR / "panel_I"
PANEL_I_OUT.mkdir(exist_ok=True)
MODULES_DIR = REPO_ROOT / "pipeline" / "modules"

_hm_fam = pd.read_csv(MODULES_DIR / "pathway_families_hallmark.tsv", sep="\t")
_kegg_fam = pd.read_csv(MODULES_DIR / "pathway_families_kegg.tsv", sep="\t")
family_map_I = dict(zip(_hm_fam["term"], _hm_fam["family"]))
for _term, _fam in zip(_kegg_fam["term"], _kegg_fam["family"]):
    if _fam == "Disease / infection (excluded)":
        continue
    family_map_I[_term] = _fam

print("  fetching MSigDB_Hallmark_2020 + KEGG_2021_Human gene sets...")
_hm_lib = _gp_I.get_library(name="MSigDB_Hallmark_2020", organism="Human")
_kegg_lib = _gp_I.get_library(name="KEGG_2021_Human", organism="Human")
gene_lib_I = {**_hm_lib, **_kegg_lib}
net_rows_I = []
for _pw, _genes in gene_lib_I.items():
    if _pw not in family_map_I:
        continue
    for _g in _genes:
        net_rows_I.append((_pw, _g, 1.0))
net_I = pd.DataFrame(net_rows_I, columns=["source", "target", "weight"])
N_PATHWAYS_I = net_I["source"].nunique()

clone_tissue_tp_I = (obs.groupby(["clone_id", "tissue"], observed=True)
                        ["timepoint"].nunique())
persistent_I = clone_tissue_tp_I[clone_tissue_tp_I >= 2].reset_index()
persistent_set_I = set(zip(persistent_I["clone_id"], persistent_I["tissue"]))
_pers_counts = (persistent_I.groupby("tissue").size()
                .reindex(TISSUES, fill_value=0).to_dict())

obs_pos_I = pd.Series(np.arange(adata.n_obs), index=adata.obs.index)
sub_I = obs[obs.apply(
    lambda r: (r["clone_id"], r["tissue"]) in persistent_set_I, axis=1)
].copy()
sub_I["pos"] = obs_pos_I.loc[sub_I.index].values

groups_I = sub_I.groupby(["clone_id", "tissue", "timepoint"], observed=True)
counts_layer_I = adata.layers["counts"]
pb_rows_I, pb_meta_I = [], []
for (_cid, _tis, _t), _grp in _tqdm_I(
        groups_I, total=groups_I.ngroups, desc="  pseudobulk"):
    if len(_grp) < MIN_CELLS_PSEUDOBULK_I:
        continue
    _row_sum = np.asarray(
        counts_layer_I[_grp["pos"].values, :].sum(axis=0)).ravel()
    pb_rows_I.append(_row_sum)
    pb_meta_I.append({"clone_id": _cid, "tissue": _tis,
                      "timepoint": str(_t), "n_cells": int(len(_grp))})
pb_meta_df_I = pd.DataFrame(pb_meta_I)
_pb_counts = (pb_meta_df_I.groupby("tissue").size()
              .reindex(TISSUES, fill_value=0).to_dict())

print(f"  persistent clones per tissue: {_pers_counts}")
print(f"  pseudobulks per tissue (>={MIN_CELLS_PSEUDOBULK_I} cells): "
      f"{_pb_counts}")
print(f"  pathways being scored: {N_PATHWAYS_I}")

pb_X_I = np.vstack(pb_rows_I).astype(np.float32)
pb_idx_I = (pb_meta_df_I["clone_id"] + "|" + pb_meta_df_I["tissue"]
            + "|t" + pb_meta_df_I["timepoint"]).values
pb_adata_I = _ad_I.AnnData(
    X=pb_X_I,
    obs=pb_meta_df_I.set_index(pb_idx_I),
    var=pd.DataFrame(index=adata.var_names.copy()),
)
sc.pp.normalize_total(pb_adata_I, target_sum=1e4)
sc.pp.log1p(pb_adata_I)

print("  calibration run (50 pathways × PBMC only)...")
_calib_pw = net_I["source"].drop_duplicates().head(50).tolist()
_calib_net = net_I[net_I["source"].isin(_calib_pw)]
_calib_pb = pb_adata_I[pb_adata_I.obs["tissue"] == "PBMC"].copy()
_t0 = _time_I.time()
_dc_I.mt.ulm(_calib_pb, _calib_net, verbose=False)
_calib_dt = _time_I.time() - _t0
_eta = (_calib_dt * (N_PATHWAYS_I / 50.0)
        * (pb_adata_I.n_obs / max(_calib_pb.n_obs, 1)))
print(f"    50 pw × {_calib_pb.n_obs} pbmc samples: {_calib_dt:.2f}s "
      f"→ ETA full ({N_PATHWAYS_I} pw × {pb_adata_I.n_obs} samples): "
      f"{_eta:.1f}s")

print(f"  scoring {N_PATHWAYS_I} pathways via decoupler.mt.ulm...")
_t0 = _time_I.time()
_dc_I.mt.ulm(pb_adata_I, net_I, verbose=False)
print(f"    full run: {_time_I.time() - _t0:.1f}s")
scores_I = pb_adata_I.obsm["score_ulm"]
scores_I.to_csv(PANEL_I_OUT / "pathway_clone_scores.csv")

TP_ORDER_I = sorted(pb_adata_I.obs["timepoint"].unique(), key=lambda s: int(s))
N_T_I = len(TP_ORDER_I)
pathway_names_I = scores_I.columns.tolist()
agg_I = np.full((len(pathway_names_I), len(TISSUES), N_T_I), np.nan)
for _ti, _tis in enumerate(TISSUES):
    for _tj, _t in enumerate(TP_ORDER_I):
        _mask = ((pb_adata_I.obs["tissue"].values == _tis)
                 & (pb_adata_I.obs["timepoint"].values == _t))
        if not _mask.any():
            continue
        agg_I[:, _ti, _tj] = scores_I.values[_mask, :].mean(axis=0)

mean_abs_I = np.nanmean(
    np.abs(agg_I.reshape(len(pathway_names_I), -1)), axis=1)
mean_abs_ser_I = pd.Series(mean_abs_I, index=pathway_names_I)
cv_per_tissue_I = np.full((len(pathway_names_I), len(TISSUES)), np.nan)
for _ti in range(len(TISSUES)):
    _M = agg_I[:, _ti, :]
    _mu = np.nanmean(_M, axis=1)
    _sd = np.nanstd(_M, axis=1)
    cv_per_tissue_I[:, _ti] = _sd / (np.abs(_mu) + 1e-9)
max_cv_I = np.nanmax(cv_per_tissue_I, axis=1)
max_cv_ser_I = pd.Series(max_cv_I, index=pathway_names_I)

top_pw_I = mean_abs_ser_I.nlargest(TOP_N_PATHWAYS_I).index.tolist()
top_pw_sorted_I = (max_cv_ser_I.loc[top_pw_I]
                   .sort_values(ascending=True).index.tolist())

pidx_I = {p: i for i, p in enumerate(pathway_names_I)}
disp_I = np.full((len(top_pw_sorted_I), len(TISSUES), N_T_I), np.nan)
for _ri, _p in enumerate(top_pw_sorted_I):
    _row = agg_I[pidx_I[_p]]
    _flat = _row.ravel()
    _mu = np.nanmean(_flat); _sd = np.nanstd(_flat)
    if _sd > 0:
        disp_I[_ri] = (_row - _mu) / _sd
    else:
        disp_I[_ri] = 0.0
VMAX_I = float(np.nanpercentile(np.abs(disp_I), 99))

stability_table_I = pd.DataFrame({
    "pathway": pathway_names_I,
    "family": [family_map_I.get(p, "") for p in pathway_names_I],
    "mean_abs_enrichment": mean_abs_I,
    "max_cv_across_tissues": max_cv_I,
    "cv_PBMC": cv_per_tissue_I[:, TISSUES.index("PBMC")],
    "cv_CSF":  cv_per_tissue_I[:, TISSUES.index("CSF")],
    "cv_TP":   cv_per_tissue_I[:, TISSUES.index("TP")],
})
stability_table_I.to_csv(
    PANEL_I_OUT / "pathway_stability_table.csv", index=False)

print("\n  --- Stability summary ---")
for _ti, _tis in enumerate(TISSUES):
    _cv_vec = cv_per_tissue_I[:, _ti]
    _ser = pd.Series(_cv_vec, index=pathway_names_I).dropna()
    _stable = _ser.nsmallest(5)
    _fluct = _ser.nlargest(5)
    print(f"  [{_tis}] top 5 stable (lowest CV):")
    for _p, _v in _stable.items():
        print(f"      {_p:<55} cv={_v:.3f}")
    print(f"  [{_tis}] top 5 fluctuating (highest CV):")
    for _p, _v in _fluct.items():
        print(f"      {_p:<55} cv={_v:.3f}")


def _draw_panel_I(fig, strip_ax, axes_h, cax):
    n_paths, n_tis, n_tp = disp_I.shape
    fam_colors = [PATHWAY_FAMILY_COLORS.get(
        family_map_I.get(p, ""), "#bbbbbb")
        for p in top_pw_sorted_I]
    strip_mat = np.arange(n_paths).reshape(-1, 1)
    strip_ax.imshow(strip_mat, aspect="auto",
                    cmap=ListedColormap(fam_colors),
                    interpolation="nearest")
    strip_ax.set_xticks([])
    strip_ax.set_yticks(range(n_paths))
    strip_ax.set_yticklabels(top_pw_sorted_I, fontsize=7)
    for tick, p in zip(strip_ax.get_yticklabels(), top_pw_sorted_I):
        tick.set_color(PATHWAY_FAMILY_COLORS.get(
            family_map_I.get(p, ""), "#444444"))
        tick.set_fontweight("bold")
    for s in ("top", "right", "bottom", "left"):
        strip_ax.spines[s].set_visible(False)
    strip_ax.tick_params(length=0)

    last_im = None
    cmap_div = plt.get_cmap("RdBu_r").copy()
    cmap_div.set_bad("#eeeeee")
    for ti, (ax_h, tis) in enumerate(zip(axes_h, TISSUES)):
        M = np.ma.masked_invalid(disp_I[:, ti, :])
        im = ax_h.imshow(M, aspect="auto", cmap=cmap_div,
                         vmin=-VMAX_I, vmax=VMAX_I,
                         interpolation="nearest")
        last_im = im
        ax_h.set_xticks(range(n_tp))
        ax_h.set_xticklabels([f"T{t}" for t in TP_ORDER_I], fontsize=8)
        ax_h.set_yticks([])
        ax_h.set_title(tis, fontsize=11, fontweight="bold",
                       color=TISSUE_COLORS[tis], pad=4)
        for s in ("top", "right", "bottom", "left"):
            ax_h.spines[s].set_visible(False)
        ax_h.tick_params(length=0)
        ax_h.set_xlabel("Timepoint", fontsize=9)

    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Row z-score", fontsize=8)
    cb.ax.tick_params(labelsize=7)


# %%
# =========================================================
# Composite Figure 2
# =========================================================
if RENDER_FULL_FIGURE:
    print("\nAssembling composite figure2.{png,pdf}...")
    fig = plt.figure(figsize=(14, 18))
    # Tight outer margins for minimal whitespace.
    gs_outer = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[3.0, 5.0, 3.5, 5.0],
        hspace=0.35,
        left=0.05, right=0.97, top=0.97, bottom=0.04,
    )

    # Shared column ratios for Row 2 (enrichment panels above ternary).
    COL_RATIOS = [1.5, 1.0]
    COL_WSPACE = 0.18

    # ---- Row 1: A (3 UMAPs) | B (tissue triangle) | C (divergence bars) ----
    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[0],
        width_ratios=[3.0, 1.2, 3.5], wspace=0.28,
    )
    gs_A = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_row1[0, 0], wspace=0.06,
    )
    ax_A1 = fig.add_subplot(gs_A[0, 0])
    ax_A2 = fig.add_subplot(gs_A[0, 1])
    ax_A3 = fig.add_subplot(gs_A[0, 2])
    ax_B = fig.add_subplot(gs_row1[0, 1])
    ax_C = fig.add_subplot(gs_row1[0, 2])
    _draw_panel_A([ax_A1, ax_A2, ax_A3])
    panel_tissue_triangle(ax_B, cosine_df)
    panel_phenotype_divergence(ax_C, cosine_df)
    _panel_letter(ax_A1, "A")
    _panel_letter(ax_B, "B")
    _panel_letter(ax_C, "C")

    # ---- Row 2: D (3 enrichment panels) | E (single ternary) ----
    gs_row2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[1],
        width_ratios=COL_RATIOS, wspace=COL_WSPACE,
    )
    gs_C = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_row2[0, 0],
        wspace=0.10,
    )
    ax_C1 = fig.add_subplot(gs_C[0, 0])
    ax_C2 = fig.add_subplot(gs_C[0, 1])
    ax_C3 = fig.add_subplot(gs_C[0, 2])
    _draw_resmig_enrichment(ax_C1, TISSUES[0], show_yticklabels=True)
    _draw_resmig_enrichment(ax_C2, TISSUES[1], show_yticklabels=False)
    _draw_resmig_enrichment(ax_C3, TISSUES[2], show_yticklabels=False)
    _panel_letter(ax_C1, "D")

    ax_D = fig.add_subplot(gs_row2[0, 1])
    _draw_ternary_single(ax_D)
    _panel_letter(ax_D, "E")

    # ---- Row 3: F (20%) | G (30%) | H (50%) ----
    gs_row3 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[2],
        width_ratios=[2.0, 3.0, 5.0], wspace=0.30,
    )
    ax_E = fig.add_subplot(gs_row3[0, 0])
    ax_F = fig.add_subplot(gs_row3[0, 1])
    ax_G = fig.add_subplot(gs_row3[0, 2])
    _draw_panel_E(ax_E)
    _draw_flow_network(ax_F)
    ax_F.set_title("Migration rates", fontsize=10, pad=4)
    _draw_panel_G(ax_G)
    _panel_letter(ax_E, "F")
    _panel_letter(ax_F, "G", x=-0.05, y=1.05)
    _panel_letter(ax_G, "H", x=-0.08, y=1.05)

    # ---- Row 4: I — pathway temporal stability (strip | 3 heatmaps | cbar) ----
    gs_row4 = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=gs_outer[3],
        width_ratios=[0.10, 1.0, 1.0, 1.0, 0.05], wspace=0.05,
    )
    ax_I_strip = fig.add_subplot(gs_row4[0, 0])
    ax_I_pbmc = fig.add_subplot(gs_row4[0, 1])
    ax_I_csf = fig.add_subplot(gs_row4[0, 2])
    ax_I_tp = fig.add_subplot(gs_row4[0, 3])
    cax_I = fig.add_subplot(gs_row4[0, 4])
    _draw_panel_I(fig, ax_I_strip,
                   [ax_I_pbmc, ax_I_csf, ax_I_tp], cax_I)
    _panel_letter(ax_I_strip, "I", x=-0.5, y=1.02)

    # ---- Final sweep ----
    for ax, lab in [
        (ax_A1, "A1"), (ax_A2, "A2"), (ax_A3, "A3"),
        (ax_B,  "B"),  (ax_C,  "C"),
        (ax_C1, "D1"), (ax_C2, "D2"), (ax_C3, "D3"),
        (ax_D,  "E"),
        (ax_E,  "F"),  (ax_F,  "G"),
        (ax_G,  "H"),
        (ax_I_strip, "I-strip"),
        (ax_I_pbmc, "I-PBMC"),
        (ax_I_csf,  "I-CSF"),
        (ax_I_tp,   "I-TP"),
    ]:
        check_text_overlaps(fig, ax, label=lab)
        check_min_fontsize(ax, label=lab)

    fig.savefig(OUT_DIR / "figure2.png", dpi=DPI_FIG, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure2.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure2.{{png,pdf}}")


# %%
# ---- Panel I standalone preview ----
print("\nPanel I preview (standalone)...")
fig_p = plt.figure(figsize=(11, 7))
gs_p = gridspec.GridSpec(
    1, 5, figure=fig_p,
    width_ratios=[0.10, 1.0, 1.0, 1.0, 0.05], wspace=0.06,
    left=0.32, right=0.95, top=0.92, bottom=0.10,
)
ax_strip_p = fig_p.add_subplot(gs_p[0, 0])
ax_p1 = fig_p.add_subplot(gs_p[0, 1])
ax_p2 = fig_p.add_subplot(gs_p[0, 2])
ax_p3 = fig_p.add_subplot(gs_p[0, 3])
cax_p = fig_p.add_subplot(gs_p[0, 4])
_draw_panel_I(fig_p, ax_strip_p, [ax_p1, ax_p2, ax_p3], cax_p)
fig_p.suptitle(
    f"Panel I — pathway temporal stability (top {TOP_N_PATHWAYS_I}, "
    "stable top → fluctuating bottom)",
    fontsize=10, y=0.98,
)
PANEL_I_PREVIEW = OUT_DIR / "panel_I_preview.png"
fig_p.savefig(PANEL_I_PREVIEW, dpi=DPI_FIG, bbox_inches="tight")
plt.close(fig_p)
print(f"  wrote {PANEL_I_PREVIEW}")

print(f"\nDone. All outputs in {OUT_DIR}")
