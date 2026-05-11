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
HIGH_DISPERSION_FLAG = 0.3
NEUTRAL_GREY = "#9e9e9e"
ANNOT_FS = 8
TICK_FS = 9
LABEL_FS = 11
TITLE_FS = 13
DPI = 200

DATA_PATH = REPO_ROOT / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"
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
    n = len(s_arr)
    if n == 0:
        summary_rows.append({
            "edge": f"{i}→{j}", "n_branches_used": 0,
            "median_jsd_src": np.nan, "q25_jsd_src": np.nan,
            "q75_jsd_src": np.nan,
            "median_jsd_dst": np.nan, "q25_jsd_dst": np.nan,
            "q75_jsd_dst": np.nan,
            "median_log2fc_norm": median_lfc[(i, j)],
        })
        continue
    summary_rows.append({
        "edge": f"{i}→{j}", "n_branches_used": n,
        "median_jsd_src": float(np.median(s_arr)),
        "q25_jsd_src": float(np.quantile(s_arr, 0.25)),
        "q75_jsd_src": float(np.quantile(s_arr, 0.75)),
        "median_jsd_dst": float(np.median(d_arr)),
        "q25_jsd_dst": float(np.quantile(d_arr, 0.25)),
        "q75_jsd_dst": float(np.quantile(d_arr, 0.75)),
        "median_log2fc_norm": median_lfc[(i, j)],
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "dispersion_summary.csv", index=False)

# %%
# ---- Figure ----
print("Plotting dispersion_jsd...")
fig = plt.figure(figsize=(14, 8))
ax_top  = fig.add_axes([0.10, 0.54, 0.78, 0.36])
ax_bot  = fig.add_axes([0.10, 0.12, 0.78, 0.36])
ax_cbar = fig.add_axes([0.90, 0.12, 0.018, 0.78])


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
    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.2)


_draw_violins(ax_top, jsd_src_by_edge)
_draw_violins(ax_bot, jsd_dst_by_edge)

# n_branches_used annotation above each violin (ax_top only).
for pos, edge in enumerate(EDGES):
    n = len(jsd_src_by_edge[edge])
    ax_top.text(pos, 0.98, f"n={n}",
                transform=ax_top.get_xaxis_transform(),
                ha="center", va="top", fontsize=ANNOT_FS - 1,
                color="dimgray")

for ax in (ax_top, ax_bot):
    ax.set_ylim(0.0, 1.02)
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
    0.5, 0.04,
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

# Sanity #5: high-dispersion flags.
checks.append("\nSanity #5: high-dispersion flags "
              f"(median JSD > {HIGH_DISPERSION_FLAG}):")
flagged_any = False
for i, j in EDGES:
    s_arr = jsd_src_by_edge[(i, j)]
    d_arr = jsd_dst_by_edge[(i, j)]
    if s_arr and np.median(s_arr) > HIGH_DISPERSION_FLAG:
        msg = (f"  HIGH DISPERSION (src): {i}→{j} "
               f"(median={np.median(s_arr):.3f})")
        print(msg.strip())
        checks.append(msg)
        flagged_any = True
    if d_arr and np.median(d_arr) > HIGH_DISPERSION_FLAG:
        msg = (f"  HIGH DISPERSION (dst): {i}→{j} "
               f"(median={np.median(d_arr):.3f})")
        print(msg.strip())
        checks.append(msg)
        flagged_any = True
if not flagged_any:
    checks.append("  (none)")

# Cross-edge statistics.
checks.append("\nCross-edge tests:")
src_groups = [jsd_src_by_edge[e] for e in EDGES if jsd_src_by_edge[e]]
dst_groups = [jsd_dst_by_edge[e] for e in EDGES if jsd_dst_by_edge[e]]
if len(src_groups) >= 2:
    H_s, p_s = kruskal(*src_groups)
    checks.append(f"  Kruskal-Wallis (jsd_src across edges): "
                  f"H={H_s:.3f}, p={p_s:.3g}")
else:
    checks.append("  Kruskal-Wallis (src): skipped (need ≥2 non-empty edges)")
if len(dst_groups) >= 2:
    H_d, p_d = kruskal(*dst_groups)
    checks.append(f"  Kruskal-Wallis (jsd_dst across edges): "
                  f"H={H_d:.3f}, p={p_d:.3g}")
else:
    checks.append("  Kruskal-Wallis (dst): skipped (need ≥2 non-empty edges)")

med_jsd_s_vec = np.array(
    [float(np.median(jsd_src_by_edge[e])) if jsd_src_by_edge[e]
     else np.nan for e in EDGES])
abs_lfc_vec = np.array([abs(median_lfc[e]) for e in EDGES])
valid = ~np.isnan(med_jsd_s_vec)
if int(valid.sum()) >= 3:
    rho, p_rho = spearmanr(med_jsd_s_vec[valid], abs_lfc_vec[valid])
    checks.append(f"  Spearman(median_jsd_src, |median_log2fc_norm|) "
                  f"across edges: rho={rho:.3f}, p={p_rho:.3g}")
else:
    checks.append("  Spearman: skipped (need ≥3 edges with data)")

if not metrics_loaded_from_06:
    checks.append("\n[WARN] median_log2fc_norm recomputed locally because "
                  "results/06_branch_empirics/edge_metrics_table.csv was "
                  "missing.")

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

checks.insert(0, f"OVERALL: {'PASS' if ok else 'FAIL'}\n")
(OUT_DIR / "render_check.txt").write_text("\n".join(checks))
print("\n".join(checks[:30]))
print(f"... full report: {OUT_DIR / 'render_check.txt'}")

# %%
print("Done.")
