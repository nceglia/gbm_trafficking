# %%
"""Within-edge phenotypic dispersion via Jensen-Shannon divergence.

For each of the 9 (i→j) edges, plots the distribution of per-branch
JSD between that branch's per-side phenotype distribution and the
edge-mean phenotype distribution. Tight low-JSD violins → the edge
mean (and therefore the 06_branch_empirics paired-bar) is
representative. Wide / high-JSD violins → the paired-bar is averaging
over heterogeneous branches.

Outputs (results/branch_dispersion_jsd/):
  dispersion_jsd.{png,pdf}
  dispersion_summary.csv
  render_check.txt
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.spatial.distance import jensenshannon
from scipy.stats import kruskal, spearmanr

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
)

# %%
# ---- Global config (single source of truth) ----
EXPANSION_CMAP = "PiYG"
EXPANSION_VLIM_AUTO = True
EXPANSION_VLIM_FALLBACK = 1.5
PSEUDOCOUNT = 0.5
MIN_N_SRC = 3
N_BOOT = 100                       # multinomial samples per branch for null
NULL_FILL_COLOR = "#cccccc"        # noise-floor violin fill
NULL_FILL_ALPHA = 0.55
NULL_SEED = 42
NULL_LARGE_N_THRESHOLD = 500       # warn if median null > 0.1 for n_src > this
NULL_LARGE_N_MAX_MEDIAN = 0.10
NEUTRAL_GREY = "#9e9e9e"
ANNOT_FS = 8
TICK_FS = 9
LABEL_FS = 11
TITLE_FS = 13
DPI = 200

from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
SRC_06_DIR = REPO_ROOT / "results" / "06_branch_empirics"
OUT_DIR = REPO_ROOT / "results" / "branch_dispersion_jsd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = ("PBMC", "CSF", "TP")
# Source-grouped order: PBMC src, then CSF src, then TP src.
EDGES = [
    ("PBMC", "PBMC"), ("PBMC", "CSF"), ("PBMC", "TP"),
    ("CSF",  "PBMC"), ("CSF",  "CSF"), ("CSF",  "TP"),
    ("TP",   "PBMC"), ("TP",   "CSF"), ("TP",   "TP"),
]
EDGE_LABELS = [f"{a}→{b}" for a, b in EDGES]
TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)
    ax.tick_params(labelsize=TICK_FS)


# %%
# ---- Load ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
obs = adata.obs[["trb", "tissue", "timepoint", "phenotype",
                 "lineage", "patient"]].copy()
print(f"  {adata.n_obs} cells")

# ---- Clone-tissue-time aggregations (mirrors 06_branch_empirics) ----
ct = (obs.groupby(["trb", "tissue", "timepoint"], observed=True)
        .size().rename("n").reset_index())
ph = (obs.groupby(["trb", "tissue", "timepoint", "phenotype"], observed=True)
        .size().unstack("phenotype", fill_value=0))
for p in PHENOTYPES:
    if p not in ph.columns:
        ph[p] = 0
ph = ph[PHENOTYPES]
ph_norm = ph.div(ph.sum(axis=1), axis=0)

clone_meta = (obs.groupby("trb", observed=True)
                .agg(patient=("patient", lambda s: s.mode().iat[0])))
N_sample = (obs.groupby(["patient", "tissue", "timepoint"], observed=True)
              .size().to_dict())

# %%
# ---- Build branches (mirrors 06_branch_empirics: n_src ≥ 2 AND n_dst ≥ 2) ----
print("Building branches...")
records = []
for t1, t2 in TRANSITIONS:
    src = ct[(ct["timepoint"] == t1) & (ct["n"] >= 2)]
    dst = ct[(ct["timepoint"] == t2) & (ct["n"] >= 2)]
    m = src.merge(dst, on="trb", suffixes=("_src", "_dst"))
    if m.empty:
        continue
    for _, r in m.iterrows():
        patient = clone_meta.loc[r["trb"], "patient"]
        N_src = N_sample.get((patient, r["tissue_src"], t1), 0)
        N_dst = N_sample.get((patient, r["tissue_dst"], t2), 0)
        n_s, n_d = int(r["n_src"]), int(r["n_dst"])
        p_src = (n_s + PSEUDOCOUNT) / (N_src + PSEUDOCOUNT)
        p_dst = (n_d + PSEUDOCOUNT) / (N_dst + PSEUDOCOUNT)
        records.append({
            "trb": r["trb"], "patient": patient,
            "src": r["tissue_src"], "dst": r["tissue_dst"],
            "t_src": t1, "t_dst": t2,
            "n_src": n_s, "n_dst": n_d,
            "log2fc_norm": float(np.log2(p_dst / p_src)),
        })
branches = pd.DataFrame(records)
print(f"  n_branches: {len(branches)}")

branches_used = branches[branches["n_src"] >= MIN_N_SRC].copy()
print(f"  filter n_src >= {MIN_N_SRC}: "
      f"{len(branches_used)}/{len(branches)} kept")

# %%
# ---- Per-edge mean phenotype distributions (mirrors 06's paired-bar) ----
# Uses ALL branches (no MIN_N_SRC filter) so that sanity #1 matches.
edge_src_mean, edge_dst_mean = {}, {}
for i, j in EDGES:
    bsub = branches[(branches["src"] == i) & (branches["dst"] == j)]
    if bsub.empty:
        edge_src_mean[(i, j)] = pd.Series(0.0, index=PHENOTYPES)
        edge_dst_mean[(i, j)] = pd.Series(0.0, index=PHENOTYPES)
        continue
    s_keys = list(zip(bsub["trb"], bsub["src"], bsub["t_src"]))
    d_keys = list(zip(bsub["trb"], bsub["dst"], bsub["t_dst"]))
    sm = ph_norm.loc[s_keys].mean(axis=0).reindex(PHENOTYPES).fillna(0.0)
    dm = ph_norm.loc[d_keys].mean(axis=0).reindex(PHENOTYPES).fillna(0.0)
    edge_src_mean[(i, j)] = sm
    edge_dst_mean[(i, j)] = dm

# %%
# ---- Per-branch JSD (branches_used only) ----
print("Computing per-branch JSD...")


def _jsd2(p, q):
    """JSD with base-2 logarithm. scipy returns sqrt(JSD); squaring undoes it."""
    return float(jensenshannon(p, q, base=2) ** 2)


def _shannon_bits(P):
    """Shannon entropy in bits along the last axis. Zeros contribute 0."""
    P = np.asarray(P)
    return -np.where(P > 0, P * np.log2(P), 0.0).sum(axis=-1)


def _jsd2_vec(P, q):
    """Batched JSD (base 2) between each row of P (N, K) and q (K,).

    Matches scipy.spatial.distance.jensenshannon(p, q, base=2)**2 for a
    single row. Returns array of shape (N,), clipped to [0, 1] to absorb
    floating-point dust.
    """
    P = np.atleast_2d(P)
    M = 0.5 * (P + q[None, :])
    H_M = _shannon_bits(M)
    H_P = _shannon_bits(P)
    H_q = float(_shannon_bits(q))
    return np.clip(H_M - 0.5 * (H_P + H_q), 0.0, 1.0)


jsd_src_by_edge = {e: [] for e in EDGES}
jsd_dst_by_edge = {e: [] for e in EDGES}

for _, b in branches_used.iterrows():
    edge = (b["src"], b["dst"])
    pheno_s = ph_norm.loc[(b["trb"], b["src"], b["t_src"])].to_numpy()
    pheno_d = ph_norm.loc[(b["trb"], b["dst"], b["t_dst"])].to_numpy()
    # Sanity #2: incoming per-clone distributions must already sum to 1.
    if not (abs(pheno_s.sum() - 1.0) < 1e-6
            and abs(pheno_d.sum() - 1.0) < 1e-6):
        raise AssertionError(
            f"per-branch phenotype distribution does not sum to 1: "
            f"trb={b['trb']}, edge={edge}, "
            f"sum_src={pheno_s.sum():.8f}, sum_dst={pheno_d.sum():.8f}"
        )
    mean_s = edge_src_mean[edge].to_numpy()
    mean_d = edge_dst_mean[edge].to_numpy()
    jsd_src_by_edge[edge].append(_jsd2(pheno_s, mean_s))
    jsd_dst_by_edge[edge].append(_jsd2(pheno_d, mean_d))

# Sanity #3: JSD ∈ [0, 1] within numerical tolerance.
for edge in EDGES:
    for side, arr in (("src", jsd_src_by_edge[edge]),
                      ("dst", jsd_dst_by_edge[edge])):
        for v in arr:
            if not (-1e-6 <= v <= 1.0 + 1e-6):
                raise AssertionError(
                    f"JSD out of [0, 1]: edge={edge}, side={side}, v={v}"
                )

# %%
# ---- Null JSD via per-branch multinomial resampling at the edge mean ----
# For each branch, draw N_BOOT counts from Multinomial(n_src, edge_mean_e),
# convert to a frequency vector, compute JSD against edge_mean_e, then
# take the median over the N_BOOT replicates. This is the sampling noise
# floor that a branch with that n would produce even if it were drawn
# directly from the edge mean. Same construction on the destination side.
print(f"Computing null JSD (N_BOOT={N_BOOT}) per branch...")
RNG_NULL = np.random.default_rng(NULL_SEED)

null_jsd_src_by_edge = {e: [] for e in EDGES}
null_jsd_dst_by_edge = {e: [] for e in EDGES}
# Tracking for the high-n / non-zero-null warning (sanity check below).
large_n_high_null = []

for _, b in branches_used.iterrows():
    edge = (b["src"], b["dst"])
    n_s, n_d = int(b["n_src"]), int(b["n_dst"])
    mean_s = edge_src_mean[edge].to_numpy()
    mean_d = edge_dst_mean[edge].to_numpy()

    # Renormalize defensively — multinomial requires p to sum to ≤ 1.
    s_sum, d_sum = float(mean_s.sum()), float(mean_d.sum())
    if s_sum > 0:
        p_s = mean_s / s_sum
        counts_s = RNG_NULL.multinomial(n_s, p_s, size=N_BOOT)
        freqs_s = counts_s / float(n_s)
        if not np.allclose(freqs_s.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError(
                f"null source frequencies do not sum to 1: trb={b['trb']}, "
                f"edge={edge}"
            )
        jsd_s = _jsd2_vec(freqs_s, mean_s)
        med_null_s = float(np.median(jsd_s))
    else:
        med_null_s = float("nan")
    null_jsd_src_by_edge[edge].append(med_null_s)

    if d_sum > 0:
        p_d = mean_d / d_sum
        counts_d = RNG_NULL.multinomial(n_d, p_d, size=N_BOOT)
        freqs_d = counts_d / float(n_d)
        if not np.allclose(freqs_d.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError(
                f"null destination frequencies do not sum to 1: "
                f"trb={b['trb']}, edge={edge}"
            )
        jsd_d = _jsd2_vec(freqs_d, mean_d)
        med_null_d = float(np.median(jsd_d))
    else:
        med_null_d = float("nan")
    null_jsd_dst_by_edge[edge].append(med_null_d)

    # Large-n branches whose noise floor is too high → bug indicator.
    if (n_s > NULL_LARGE_N_THRESHOLD
            and not np.isnan(med_null_s)
            and med_null_s > NULL_LARGE_N_MAX_MEDIAN):
        large_n_high_null.append(
            ("src", b["trb"], edge, n_s, med_null_s)
        )
    if (n_d > NULL_LARGE_N_THRESHOLD
            and not np.isnan(med_null_d)
            and med_null_d > NULL_LARGE_N_MAX_MEDIAN):
        large_n_high_null.append(
            ("dst", b["trb"], edge, n_d, med_null_d)
        )

# Sanity: every null value (the per-branch median) is in [0, 1].
for edge in EDGES:
    for side, arr in (("src", null_jsd_src_by_edge[edge]),
                      ("dst", null_jsd_dst_by_edge[edge])):
        for v in arr:
            if np.isnan(v):
                continue
            if not (-1e-6 <= v <= 1.0 + 1e-6):
                raise AssertionError(
                    f"null JSD out of [0, 1]: edge={edge}, side={side}, v={v}"
                )

# %%
# ---- Load median_log2fc_norm from 06_branch_empirics (for color key) ----
edge_metrics_path = SRC_06_DIR / "edge_metrics_table.csv"
median_lfc = {}
if edge_metrics_path.exists():
    em = pd.read_csv(edge_metrics_path)
    em_map = dict(zip(em["edge"], em["median_log2fc_norm"]))
    for i, j in EDGES:
        median_lfc[(i, j)] = float(em_map.get(f"{i}→{j}", 0.0))
    print(f"Loaded median_log2fc_norm from {edge_metrics_path.name}")
    metrics_loaded_from_06 = True
else:
    print(f"WARNING: {edge_metrics_path} missing; recomputing locally.")
    for i, j in EDGES:
        sub = branches_used[(branches_used["src"] == i)
                              & (branches_used["dst"] == j)]
        median_lfc[(i, j)] = (float(np.median(sub["log2fc_norm"]))
                              if len(sub) else 0.0)
    metrics_loaded_from_06 = False

# %%
# ---- Color norm (identical recipe to 06_branch_empirics expansion graph) ----
_medians = np.array([median_lfc[e] for e in EDGES])
_max_abs = float(np.max(np.abs(_medians))) if len(_medians) else 0.0
if EXPANSION_VLIM_AUTO:
    expansion_vmax = max(_max_abs, EXPANSION_VLIM_FALLBACK)
else:
    expansion_vmax = EXPANSION_VLIM_FALLBACK
expansion_vmin = -expansion_vmax
expansion_norm = TwoSlopeNorm(vmin=expansion_vmin, vcenter=0.0,
                              vmax=expansion_vmax)
expansion_cmap = plt.get_cmap(EXPANSION_CMAP)


def _edge_color(edge):
    return expansion_cmap(expansion_norm(median_lfc[edge]))


# %%
# ---- Summary table ----
summary_rows = []
for i, j in EDGES:
    s_arr = np.array(jsd_src_by_edge[(i, j)])
    d_arr = np.array(jsd_dst_by_edge[(i, j)])
    ns_arr = np.array(null_jsd_src_by_edge[(i, j)], dtype=float)
    nd_arr = np.array(null_jsd_dst_by_edge[(i, j)], dtype=float)
    n = len(s_arr)
    if n == 0:
        summary_rows.append({
            "edge": f"{i}→{j}", "n_branches_used": 0,
            "median_jsd_src": np.nan, "q25_jsd_src": np.nan,
            "q75_jsd_src": np.nan,
            "median_jsd_dst": np.nan, "q25_jsd_dst": np.nan,
            "q75_jsd_dst": np.nan,
            "median_null_jsd_src": np.nan,
            "median_null_jsd_dst": np.nan,
            "excess_dispersion_src": np.nan,
            "excess_dispersion_dst": np.nan,
            "median_log2fc_norm": median_lfc[(i, j)],
        })
        continue
    med_s = float(np.median(s_arr))
    med_d = float(np.median(d_arr))
    med_ns = float(np.nanmedian(ns_arr)) if np.any(~np.isnan(ns_arr)) \
        else float("nan")
    med_nd = float(np.nanmedian(nd_arr)) if np.any(~np.isnan(nd_arr)) \
        else float("nan")
    summary_rows.append({
        "edge": f"{i}→{j}", "n_branches_used": n,
        "median_jsd_src": med_s,
        "q25_jsd_src": float(np.quantile(s_arr, 0.25)),
        "q75_jsd_src": float(np.quantile(s_arr, 0.75)),
        "median_jsd_dst": med_d,
        "q25_jsd_dst": float(np.quantile(d_arr, 0.25)),
        "q75_jsd_dst": float(np.quantile(d_arr, 0.75)),
        "median_null_jsd_src": med_ns,
        "median_null_jsd_dst": med_nd,
        "excess_dispersion_src": med_s - med_ns,
        "excess_dispersion_dst": med_d - med_nd,
        "median_log2fc_norm": median_lfc[(i, j)],
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "dispersion_summary.csv", index=False)

# %%
# ---- Figure ----
print("Plotting dispersion_jsd...")
fig = plt.figure(figsize=(14, 8))
# ax_bot bumped up (y=0.18) to make room for the caption; ax_top keeps
# its position so the two panels share the y=0.54 boundary. ax_cbar
# spans from the new ax_bot bottom (0.18) to the ax_top top (0.90).
ax_top  = fig.add_axes([0.10, 0.54, 0.78, 0.36])
ax_bot  = fig.add_axes([0.10, 0.18, 0.78, 0.36])
ax_cbar = fig.add_axes([0.90, 0.18, 0.018, 0.72])


def _draw_null_violins(ax, null_data_per_edge):
    """Light-grey noise-floor violins behind the real ones."""
    safe_data, valid_positions = [], []
    for pos, edge in enumerate(EDGES):
        arr = [v for v in null_data_per_edge[edge] if not np.isnan(v)]
        if len(arr) == 0:
            continue
        safe_data.append(arr)
        valid_positions.append(pos)
    if not safe_data:
        return
    vp = ax.violinplot(safe_data, positions=valid_positions,
                       widths=0.78, showmedians=False, showextrema=False)
    for pc in vp["bodies"]:
        pc.set_facecolor(NULL_FILL_COLOR)
        pc.set_edgecolor("none")
        pc.set_linewidth(0)
        pc.set_alpha(NULL_FILL_ALPHA)
        pc.set_zorder(1)


def _draw_violins(ax, data_per_edge):
    safe_data, valid_positions, valid_edges = [], [], []
    for pos, edge in enumerate(EDGES):
        arr = data_per_edge[edge]
        if len(arr) == 0:
            continue
        safe_data.append(arr)
        valid_positions.append(pos)
        valid_edges.append(edge)
    if not safe_data:
        return
    vp = ax.violinplot(safe_data, positions=valid_positions,
                       widths=0.78, showmedians=True, showextrema=False)
    for pc, edge in zip(vp["bodies"], valid_edges):
        pc.set_facecolor(_edge_color(edge))
        pc.set_edgecolor(NEUTRAL_GREY)
        pc.set_linewidth(0.6)
        pc.set_alpha(1.0)
        pc.set_zorder(2)
    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.2)
        vp["cmedians"].set_zorder(3)


# Null behind, real in front — call order matters for unset-zorder fallback.
_draw_null_violins(ax_top, null_jsd_src_by_edge)
_draw_null_violins(ax_bot, null_jsd_dst_by_edge)
_draw_violins(ax_top, jsd_src_by_edge)
_draw_violins(ax_bot, jsd_dst_by_edge)

# Legend on ax_top (anchored in axes coords; verified clear of violins
# in sanity check below).
_observed_swatch_color = expansion_cmap(expansion_norm(-expansion_vmax * 0.6))
_legend_handles = [
    Patch(facecolor=NULL_FILL_COLOR, alpha=NULL_FILL_ALPHA,
          edgecolor="none",
          label="expected from sampling at this n"),
    Patch(facecolor=_observed_swatch_color,
          edgecolor=NEUTRAL_GREY, linewidth=0.6,
          label="observed"),
]
_legend = ax_top.legend(
    handles=_legend_handles, loc="upper right",
    bbox_to_anchor=(1.0, 1.0), fontsize=ANNOT_FS,
    frameon=False, handlelength=1.0, handletextpad=0.4,
    labelspacing=0.2, borderaxespad=0.0,
)
_legend.set_zorder(10)

# n_branches_used annotation above each violin (ax_top only). Sits in
# the upper-margin band created by ylim=(0, 1.10); the legend lives in
# the same band at the right edge, so these annotations are placed
# below the legend band but above the JSD≤1 ceiling.
for pos, edge in enumerate(EDGES):
    n = len(jsd_src_by_edge[edge])
    ax_top.text(pos, 0.92, f"n={n}",
                transform=ax_top.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=ANNOT_FS - 1,
                color="dimgray")

for ax in (ax_top, ax_bot):
    # 10% top margin opens space for the noise-floor legend on ax_top
    # without pushing the violins down on either panel; both panels
    # keep the same y-scale and 0–1 JSD framing.
    ax.set_ylim(0.0, 1.10)
    ax.set_xlim(-0.6, len(EDGES) - 0.4)
    ax.set_ylabel("JSD (branch vs edge mean)", fontsize=LABEL_FS)
    for x_sep in (2.5, 5.5):
        ax.axvline(x_sep, color="gray", lw=0.6,
                   linestyle="--", alpha=0.3)
    _style_axis(ax)

# Row titles inside each panel (top-left).
ax_top.text(0.01, 0.95, "source distribution",
            transform=ax_top.transAxes, ha="left", va="top",
            fontsize=LABEL_FS, fontweight="bold")
ax_bot.text(0.01, 0.95, "destination distribution",
            transform=ax_bot.transAxes, ha="left", va="top",
            fontsize=LABEL_FS, fontweight="bold")

# Shared x-axis: ax_top hides labels, ax_bot shows them rotated.
ax_top.tick_params(axis="x", labelbottom=False, length=0)
ax_bot.set_xticks(range(len(EDGES)))
ax_bot.set_xticklabels(EDGE_LABELS, rotation=30, ha="right",
                       fontsize=TICK_FS)
# Color tick labels by SOURCE tissue (the eye groups by source).
for tick, (i, _j) in zip(ax_bot.get_xticklabels(), EDGES):
    tick.set_color(TISSUE_COLORS[i])

# Shared colorbar.
sm = ScalarMappable(norm=expansion_norm, cmap=expansion_cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=ax_cbar)
cb.set_label("median log₂ fold-change (depth-corrected)",
             rotation=270, labelpad=18, fontsize=LABEL_FS)
cb.set_ticks([expansion_vmin, 0.0, expansion_vmax])
cb.ax.tick_params(labelsize=TICK_FS)
cb.ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
for tlbl in cb.ax.get_yticklabels():
    if tlbl.get_text().strip() in ("0", "0.0", "0.00"):
        tlbl.set_fontweight("bold")

fig.suptitle("Within-edge phenotypic dispersion",
             fontsize=TITLE_FS + 1, fontweight="bold", y=0.96)
fig.text(
    0.5, 0.02,
    "Each violin = distribution of branches' JSD from their edge's mean "
    "phenotype distribution. Tight low-JSD violin → edge mean is "
    "representative. Wide / high-JSD violin → edge mean averages over "
    "heterogeneous behaviours, and the 06_branch_empirics paired-bar "
    "hides structure. Violin fill = median depth-corrected log₂ "
    "fold-change for that edge (same scale as 06_branch_empirics "
    "expansion graph).",
    ha="center", va="bottom", fontsize=ANNOT_FS,
    color="dimgray", style="italic", wrap=True,
)

png_path = OUT_DIR / "dispersion_jsd.png"
pdf_path = OUT_DIR / "dispersion_jsd.pdf"
fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")

# %%
# ---- Sanity checks & cross-edge statistics ----
print("Running sanity checks...")
checks = []
ok = True

# Sanity #1: per-edge mean distributions match 06_branch_empirics paired-bar.
pheno_dist_path = SRC_06_DIR / "phenotype_dist.csv"
checks.append("Sanity #1: edge-mean distributions vs 06_branch_empirics:")
if pheno_dist_path.exists():
    pd06 = pd.read_csv(pheno_dist_path)
    for i, j in EDGES:
        label = f"{i}→{j}"
        sub_s = pd06[(pd06["edge"] == label) & (pd06["side"] == "src")]
        sub_d = pd06[(pd06["edge"] == label) & (pd06["side"] == "dst")]
        s06 = dict(zip(sub_s["phenotype"], sub_s["frac"]))
        d06 = dict(zip(sub_d["phenotype"], sub_d["frac"]))
        max_ds = max((abs(edge_src_mean[(i, j)][p] - s06.get(p, 0.0))
                      for p in PHENOTYPES), default=0.0)
        max_dd = max((abs(edge_dst_mean[(i, j)][p] - d06.get(p, 0.0))
                      for p in PHENOTYPES), default=0.0)
        match = max(max_ds, max_dd) < 1e-6
        if not match:
            ok = False
            raise AssertionError(
                f"edge {label}: edge-mean disagrees with 06 paired-bar "
                f"(max|Δsrc|={max_ds:.2e}, max|Δdst|={max_dd:.2e})"
            )
        checks.append(f"  [OK] {label}: max|Δsrc|={max_ds:.2e}, "
                      f"max|Δdst|={max_dd:.2e}")
else:
    checks.append(f"  [WARN] {pheno_dist_path} not found; cross-check skipped.")

# Sanity #2, #3 already raise inline; record outcome.
checks.append("\nSanity #2: per-branch pheno_src/dst sum to 1.0 — passed "
              "(else raised above).")
checks.append("Sanity #3: all JSD in [0, 1] — passed (else raised above).")

# ---- Cross-edge tests (Kruskal-Wallis + Spearman), surfaced first ----
checks.append("\nCross-edge tests:")


def _stdout_and_record(line):
    print(line)
    checks.append(f"  {line}")


src_groups = [jsd_src_by_edge[e] for e in EDGES if jsd_src_by_edge[e]]
dst_groups = [jsd_dst_by_edge[e] for e in EDGES if jsd_dst_by_edge[e]]

try:
    H_s, p_s = kruskal(*src_groups)
    _stdout_and_record(
        f"Kruskal-Wallis (src JSD across 9 edges): "
        f"H={H_s:.2f}, p={p_s:.2e}"
    )
except Exception as e:
    _stdout_and_record(f"Kruskal-Wallis (src JSD): SKIPPED ({e})")

try:
    H_d, p_d = kruskal(*dst_groups)
    _stdout_and_record(
        f"Kruskal-Wallis (dst JSD across 9 edges): "
        f"H={H_d:.2f}, p={p_d:.2e}"
    )
except Exception as e:
    _stdout_and_record(f"Kruskal-Wallis (dst JSD): SKIPPED ({e})")

med_jsd_s_vec = np.array(
    [float(np.median(jsd_src_by_edge[e])) if jsd_src_by_edge[e]
     else np.nan for e in EDGES])
med_jsd_d_vec = np.array(
    [float(np.median(jsd_dst_by_edge[e])) if jsd_dst_by_edge[e]
     else np.nan for e in EDGES])
abs_lfc_vec = np.array([abs(median_lfc[e]) for e in EDGES])

for side_name, vec in (("src", med_jsd_s_vec), ("dst", med_jsd_d_vec)):
    valid = ~np.isnan(vec)
    try:
        if int(valid.sum()) < 3:
            raise ValueError(
                f"need >=3 edges with data, got {int(valid.sum())}"
            )
        rho, p_rho = spearmanr(vec[valid], abs_lfc_vec[valid])
        _stdout_and_record(
            f"Spearman (median_jsd_{side_name} vs "
            f"|median_log2fc_norm|): rho={rho:.3f}, p={p_rho:.2e}"
        )
    except Exception as e:
        _stdout_and_record(
            f"Spearman (median_jsd_{side_name} vs "
            f"|median_log2fc_norm|): SKIPPED ({e})"
        )

# Sanity #4: per-edge summary lines.
checks.append("\nSanity #4: per-edge summary:")
checks.append(f"  {'edge':<14}{'n_used':>8}{'med_jsd_s':>12}"
              f"{'med_jsd_d':>12}{'med_lfc':>10}")
for i, j in EDGES:
    s_arr = jsd_src_by_edge[(i, j)]
    d_arr = jsd_dst_by_edge[(i, j)]
    ms = float(np.median(s_arr)) if s_arr else float("nan")
    md = float(np.median(d_arr)) if d_arr else float("nan")
    checks.append(f"  {i+'→'+j:<14}{len(s_arr):>8}"
                  f"{ms:>12.4f}{md:>12.4f}"
                  f"{median_lfc[(i, j)]:>10.4f}")

# Sanity #5: top-3 dispersion edges per side + any edge with median JSD
# above EXTREME_THRESHOLD. 0.50 was chosen because TP→CSF dst (≈0.533)
# is the only edge currently over it on the dst side, and nothing
# exceeds it on src — so "extreme" actually means something.
EXTREME_THRESHOLD = 0.50
checks.append(
    "\nSanity #5: top-3 dispersion edges per side, plus any edge with "
    f"median JSD > {EXTREME_THRESHOLD:.2f} flagged as extreme:"
)
for side_name, jsd_by_edge in (("src", jsd_src_by_edge),
                                ("dst", jsd_dst_by_edge)):
    medians = [(e, float(np.median(jsd_by_edge[e])))
               for e in EDGES if jsd_by_edge[e]]
    medians.sort(key=lambda x: x[1], reverse=True)
    for edge, med in medians[:3]:
        msg = (f"  TOP DISPERSION ({side_name}): "
               f"{edge[0]}→{edge[1]} (median={med:.3f})")
        print(msg.strip())
        checks.append(msg)
    extreme = [(e, m) for e, m in medians if m > EXTREME_THRESHOLD]
    for edge, med in extreme:
        msg = (f"  EXTREME DISPERSION ({side_name}): "
               f"{edge[0]}→{edge[1]} (median={med:.3f})")
        print(msg.strip())
        checks.append(msg)
    if not extreme:
        checks.append(f"  (no extreme dispersion edges on {side_name})")

if not metrics_loaded_from_06:
    checks.append("\n[WARN] median_log2fc_norm recomputed locally because "
                  "results/06_branch_empirics/edge_metrics_table.csv was "
                  "missing.")

# ---- Noise-floor diagnostic: real dispersion above sampling noise. ----
checks.append("\nNoise-floor diagnostic "
              "(excess_dispersion = median(real) - median(null)):")
checks.append(f"  {'edge':<14}{'n_used':>8}"
              f"{'med_real_s':>12}{'med_null_s':>12}{'excess_s':>10}"
              f"{'med_real_d':>12}{'med_null_d':>12}{'excess_d':>10}")
excess_rows = []
for i, j in EDGES:
    s_arr = jsd_src_by_edge[(i, j)]
    d_arr = jsd_dst_by_edge[(i, j)]
    ns_arr = [v for v in null_jsd_src_by_edge[(i, j)] if not np.isnan(v)]
    nd_arr = [v for v in null_jsd_dst_by_edge[(i, j)] if not np.isnan(v)]
    if not s_arr:
        continue
    med_s = float(np.median(s_arr))
    med_d = float(np.median(d_arr))
    med_ns = float(np.median(ns_arr)) if ns_arr else float("nan")
    med_nd = float(np.median(nd_arr)) if nd_arr else float("nan")
    excess_s = med_s - med_ns
    excess_d = med_d - med_nd
    excess_rows.append({
        "edge": (i, j),
        "med_real_s": med_s, "med_null_s": med_ns, "excess_s": excess_s,
        "med_real_d": med_d, "med_null_d": med_nd, "excess_d": excess_d,
    })
    checks.append(
        f"  {i+'→'+j:<14}{len(s_arr):>8}"
        f"{med_s:>12.4f}{med_ns:>12.4f}{excess_s:>10.4f}"
        f"{med_d:>12.4f}{med_nd:>12.4f}{excess_d:>10.4f}"
    )

# Biology rank: edges sorted by excess_dispersion descending, per side.
checks.append("\nBiology rank — edges sorted by excess_dispersion (src):")
for r in sorted(excess_rows, key=lambda x: x["excess_s"], reverse=True):
    msg = (f"  {r['edge'][0]}→{r['edge'][1]:<6}  "
           f"excess_src={r['excess_s']:+.4f}  "
           f"(real={r['med_real_s']:.3f}, null={r['med_null_s']:.3f})")
    print(msg.strip())
    checks.append(msg)
checks.append("\nBiology rank — edges sorted by excess_dispersion (dst):")
for r in sorted(excess_rows, key=lambda x: x["excess_d"], reverse=True):
    msg = (f"  {r['edge'][0]}→{r['edge'][1]:<6}  "
           f"excess_dst={r['excess_d']:+.4f}  "
           f"(real={r['med_real_d']:.3f}, null={r['med_null_d']:.3f})")
    print(msg.strip())
    checks.append(msg)

# Large-n noise-floor sanity warning.
if large_n_high_null:
    checks.append(
        f"\n[WARN] {len(large_n_high_null)} branches with "
        f"n > {NULL_LARGE_N_THRESHOLD} have median null JSD > "
        f"{NULL_LARGE_N_MAX_MEDIAN}:"
    )
    for side, trb, edge, n, med in large_n_high_null[:10]:
        checks.append(
            f"  [WARN] {side}: trb={trb}, edge={edge[0]}→{edge[1]}, "
            f"n={n}, median_null={med:.3f}"
        )
    if len(large_n_high_null) > 10:
        checks.append(f"  ... and {len(large_n_high_null) - 10} more")
else:
    checks.append(
        f"\n[OK] No branches with n > {NULL_LARGE_N_THRESHOLD} exceed "
        f"median null JSD {NULL_LARGE_N_MAX_MEDIAN}."
    )

# Sanity #6: bbox intersection checks among violins, tick labels, colorbar.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()


def _ovl(b1, b2):
    return not (b1.x1 <= b2.x0 or b2.x1 <= b1.x0
                or b1.y1 <= b2.y0 or b2.y1 <= b1.y0)


checks.append("\nSanity #6: no bbox overlap between violins, edge tick "
              "labels, and colorbar axis:")
cbar_bbox = ax_cbar.get_window_extent(renderer=renderer)

for name, ax in (("ax_top", ax_top), ("ax_bot", ax_bot)):
    ax_bbox = ax.get_window_extent(renderer=renderer)
    clash = _ovl(ax_bbox, cbar_bbox)
    if clash:
        ok = False
        raise AssertionError(f"{name} axes overlaps colorbar axes")
    checks.append(f"  [OK] {name} vs ax_cbar")

for tick in ax_bot.get_xticklabels():
    if not tick.get_text():
        continue
    tb = tick.get_window_extent(renderer=renderer)
    if _ovl(tb, cbar_bbox):
        ok = False
        raise AssertionError(
            f"x tick label '{tick.get_text()}' overlaps colorbar axes"
        )
checks.append("  [OK] x tick labels vs ax_cbar")

# Legend on ax_top must clear every violin body on that axis. The
# rightmost real violin (TP→TP at x=8) is the only plausible offender.
legend_bbox = _legend.get_window_extent(renderer=renderer)
for child in ax_top.collections:
    # violinplot bodies are PolyCollection objects with non-empty paths
    cb = child.get_window_extent(renderer=renderer)
    if _ovl(legend_bbox, cb):
        ok = False
        raise AssertionError(
            "legend on ax_top overlaps a violin body — increase "
            "bbox_to_anchor offset or reduce legend size"
        )
checks.append("  [OK] ax_top legend vs all violin bodies")

# Caption vs ax_bot tick labels — guards the Fix 1 layout change.
checks.append(
    "\nCaption vs ax_bot x tick labels: no bbox overlap:"
)
caption_artists = [t for t in fig.texts
                   if t.get_text().startswith("Each violin")]
if not caption_artists:
    checks.append("  [WARN] caption text artist not found; skipped")
else:
    caption_bbox = caption_artists[0].get_window_extent(renderer=renderer)
    for tick in ax_bot.get_xticklabels():
        if not tick.get_text():
            continue
        tb = tick.get_window_extent(renderer=renderer)
        if _ovl(tb, caption_bbox):
            ok = False
            raise AssertionError(
                f"caption overlaps ax_bot x tick label "
                f"'{tick.get_text()}'"
            )
    checks.append("  [OK] caption clear of all ax_bot tick labels")

checks.insert(0, f"OVERALL: {'PASS' if ok else 'FAIL'}\n")
(OUT_DIR / "render_check.txt").write_text("\n".join(checks))
print("\n".join(checks[:30]))
print(f"... full report: {OUT_DIR / 'render_check.txt'}")

# %%
print("Done.")
