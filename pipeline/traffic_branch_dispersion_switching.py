# %%
"""Per-edge dominant-phenotype switching heatmaps.

For each of the 9 (i→j) edges, plots the 12×12 matrix of branch counts
indexed by (dominant src phenotype, dominant dst phenotype). The
question is: when clones change dominant phenotype across the
transition, do they all change in the same way (concentrated
off-diagonal mass = stereotyped switching) or in many different ways
(diffuse off-diagonal mass = diverse switching)?

Outputs (results/branch_dispersion_switching/):
  switching_heatmaps.{png,pdf}
  switching_matrices.csv
  switching_summary.csv
  render_check.txt
"""
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns  # noqa: F401  (registers "rocket_r" with matplotlib)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (
    LINEAGE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
)

# %%
# ---- Global config (single source of truth) ----
MIN_N_SRC = 3
STEREOTYPED_FLAG_NONZERO = 12
STEREOTYPED_FLAG_MIN_BRANCHES = 30
DIFFUSE_FLAG_ENTROPY = 0.6
DIFFUSE_FLAG_FRAC_DIAG = 0.3
TIE_BREAK_WARN_FRAC = 0.10
ANNOT_FS = 8
TICK_FS = 9
LABEL_FS = 11
TITLE_FS = 13
DPI = 200

EMPTY_CELL_COLOR = "#e8e8e8"

from modules import paths  # noqa: E402

DATA_PATH = paths.H5AD_TCELLS
SRC_06_DIR = REPO_ROOT / "results" / "06_branch_empirics"
OUT_DIR = REPO_ROOT / "results" / "branch_dispersion_switching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = ("PBMC", "CSF", "TP")
# Row = source tissue, col = destination tissue.
TISSUE_ROWS = ("PBMC", "CSF", "TP")
TISSUE_COLS = ("PBMC", "CSF", "TP")
EDGES = [(i, j) for i in TISSUE_ROWS for j in TISSUE_COLS]

TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)
N_PHEN = len(PHENOTYPES)


def _lineage_of(p):
    return "CD8" if p.startswith("CD8") else "CD4"


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


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
ph = ph[PHENOTYPES]  # column order = TCELL_PHENOTYPE_ORDER

# %%
# ---- Build branches (same definition as 06_branch_empirics) ----
print("Building branches...")
records = []
for t1, t2 in TRANSITIONS:
    src = ct[(ct["timepoint"] == t1) & (ct["n"] >= 2)]
    dst = ct[(ct["timepoint"] == t2) & (ct["n"] >= 2)]
    m = src.merge(dst, on="trb", suffixes=("_src", "_dst"))
    if m.empty:
        continue
    for _, r in m.iterrows():
        n_s, n_d = int(r["n_src"]), int(r["n_dst"])
        records.append({
            "trb": r["trb"],
            "src": r["tissue_src"], "dst": r["tissue_dst"],
            "t_src": t1, "t_dst": t2,
            "n_src": n_s, "n_dst": n_d,
        })
branches = pd.DataFrame(records)
branches_used = branches[branches["n_src"] >= MIN_N_SRC].copy()
print(f"  n_branches: {len(branches)} (used: {len(branches_used)})")

# %%
# ---- Per-branch dominant phenotype (with tie detection) ----
print("Assigning dominant phenotypes...")
ph_arr_by_key = {key: row.to_numpy() for key, row in ph.iterrows()}


def _dominant(key):
    """Returns (phenotype, is_tied_at_max). Tie-break = earliest in TCELL_PHENOTYPE_ORDER."""
    counts = ph_arr_by_key[key]
    mx = counts.max()
    is_tied = bool((counts == mx).sum() > 1)
    # np.argmax returns the first occurrence; columns are already in
    # TCELL_PHENOTYPE_ORDER, so this is the tie-break rule.
    idx = int(np.argmax(counts))
    return PHENOTYPES[idx], is_tied


src_doms, dst_doms, src_ties, dst_ties = [], [], [], []
for _, b in branches_used.iterrows():
    s_dom, s_tied = _dominant((b["trb"], b["src"], b["t_src"]))
    d_dom, d_tied = _dominant((b["trb"], b["dst"], b["t_dst"]))
    src_doms.append(s_dom); src_ties.append(s_tied)
    dst_doms.append(d_dom); dst_ties.append(d_tied)
branches_used["src_dom"] = src_doms
branches_used["dst_dom"] = dst_doms
branches_used["src_tied"] = src_ties
branches_used["dst_tied"] = dst_ties

# Sanity #2: all dom values are in TCELL_PHENOTYPE_ORDER.
bad_src = set(branches_used["src_dom"]) - set(PHENOTYPES)
bad_dst = set(branches_used["dst_dom"]) - set(PHENOTYPES)
if bad_src or bad_dst:
    raise AssertionError(
        f"unknown dominant phenotype labels: src={bad_src}, dst={bad_dst}"
    )

# %%
# ---- Build switching matrices ----
print("Building switching matrices...")
ph_to_idx = {p: i for i, p in enumerate(PHENOTYPES)}
switching = {edge: np.zeros((N_PHEN, N_PHEN), dtype=int) for edge in EDGES}
n_ties_src = {edge: 0 for edge in EDGES}
n_ties_dst = {edge: 0 for edge in EDGES}

for _, b in branches_used.iterrows():
    edge = (b["src"], b["dst"])
    M = switching[edge]
    M[ph_to_idx[b["src_dom"]], ph_to_idx[b["dst_dom"]]] += 1
    if b["src_tied"]:
        n_ties_src[edge] += 1
    if b["dst_tied"]:
        n_ties_dst[edge] += 1

# Sanity #1: matrix sum == n_branches_used per edge.
for edge in EDGES:
    n_used = int(((branches_used["src"] == edge[0])
                   & (branches_used["dst"] == edge[1])).sum())
    s = int(switching[edge].sum())
    if s != n_used:
        raise AssertionError(
            f"edge {edge}: matrix sum {s} != n_branches_used {n_used}"
        )

# Global vmax for the shared LogNorm.
global_vmax = max(int(switching[e].max()) for e in EDGES)
if global_vmax < 1:
    global_vmax = 1
print(f"  global max cell count: {global_vmax}")

# %%
# ---- Per-edge stats ----
def _matrix_stats(M):
    total = int(M.sum())
    if total == 0:
        return dict(n=0, frac_diag=np.nan, H_norm=np.nan,
                    top_src=None, top_dst=None,
                    top_n=0, top_frac=np.nan,
                    nonzero=0)
    diag = int(np.trace(M))
    p_flat = M.flatten().astype(float) / total
    H = float(scipy_entropy(p_flat, base=2) / np.log2(N_PHEN * N_PHEN))
    flat_idx = int(np.argmax(M))
    top_si, top_di = np.unravel_index(flat_idx, M.shape)
    top_n = int(M[top_si, top_di])
    return dict(
        n=total,
        frac_diag=diag / total,
        H_norm=H,
        top_src=PHENOTYPES[top_si],
        top_dst=PHENOTYPES[top_di],
        top_n=top_n,
        top_frac=top_n / total,
        nonzero=int((M > 0).sum()),
    )


stats_by_edge = {edge: _matrix_stats(switching[edge]) for edge in EDGES}

# %%
# ---- Load median_log2fc_norm from 06 for the summary CSV ----
edge_metrics_path = SRC_06_DIR / "edge_metrics_table.csv"
median_lfc = {}
if edge_metrics_path.exists():
    em = pd.read_csv(edge_metrics_path)
    em_map = dict(zip(em["edge"], em["median_log2fc_norm"]))
    for i, j in EDGES:
        median_lfc[(i, j)] = float(em_map.get(f"{i}→{j}", float("nan")))
    metrics_loaded_from_06 = True
else:
    print(f"WARNING: {edge_metrics_path} missing; median_log2fc_norm = NaN.")
    for edge in EDGES:
        median_lfc[edge] = float("nan")
    metrics_loaded_from_06 = False

# %%
# ---- Long-format matrices CSV ----
long_rows = []
for edge in EDGES:
    M = switching[edge]
    label = f"{edge[0]}→{edge[1]}"
    for si, sp in enumerate(PHENOTYPES):
        for di, dp in enumerate(PHENOTYPES):
            long_rows.append({
                "edge": label,
                "src_phenotype": sp,
                "dst_phenotype": dp,
                "n_branches": int(M[si, di]),
            })
pd.DataFrame(long_rows).to_csv(
    OUT_DIR / "switching_matrices.csv", index=False)

# ---- Summary CSV ----
summary_rows = []
for edge in EDGES:
    s = stats_by_edge[edge]
    summary_rows.append({
        "edge": f"{edge[0]}→{edge[1]}",
        "n_branches_used": s["n"],
        "frac_diagonal": s["frac_diag"],
        "normalized_entropy": s["H_norm"],
        "top_transition_src": s["top_src"],
        "top_transition_dst": s["top_dst"],
        "top_transition_n": s["top_n"],
        "top_transition_frac": s["top_frac"],
        "n_ties_src": n_ties_src[edge],
        "n_ties_dst": n_ties_dst[edge],
        "median_log2fc_norm": median_lfc[edge],
    })
pd.DataFrame(summary_rows).to_csv(
    OUT_DIR / "switching_summary.csv", index=False)

# %%
# ---- Figure ----
print("Plotting switching_heatmaps...")

cmap = mpl.colormaps["rocket_r"].copy()
cmap.set_bad(EMPTY_CELL_COLOR)
norm = LogNorm(vmin=1, vmax=global_vmax)


def _build_figure(wspace, hspace):
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(
        3, 4,
        width_ratios=[1, 1, 1, 0.05],
        wspace=wspace, hspace=hspace,
        left=0.07, right=0.93, top=0.90, bottom=0.08,
    )
    axes = {}
    last_im = None
    for ri, src_t in enumerate(TISSUE_ROWS):
        for ci, dst_t in enumerate(TISSUE_COLS):
            ax = fig.add_subplot(gs[ri, ci])
            edge = (src_t, dst_t)
            M = switching[edge].astype(float)
            M_masked = np.ma.masked_where(M == 0, M)

            im = ax.imshow(M_masked, cmap=cmap, norm=norm,
                           aspect="equal", interpolation="nearest")
            last_im = im

            ax.set_xticks(range(N_PHEN))
            ax.set_yticks(range(N_PHEN))
            ax.set_xticklabels(
                [TCELL_PHENOTYPE_LABELS[p] for p in PHENOTYPES],
                rotation=90, fontsize=ANNOT_FS - 1,
            )
            ax.set_yticklabels(
                [TCELL_PHENOTYPE_LABELS[p] for p in PHENOTYPES],
                rotation=0, fontsize=ANNOT_FS - 1,
            )
            for tick, p in zip(ax.get_xticklabels(), PHENOTYPES):
                tick.set_color(LINEAGE_COLORS[_lineage_of(p)])
            for tick, p in zip(ax.get_yticklabels(), PHENOTYPES):
                tick.set_color(LINEAGE_COLORS[_lineage_of(p)])

            s = stats_by_edge[edge]
            ax.set_title(f"{src_t} → {dst_t}\nn={s['n']}",
                         fontsize=TITLE_FS - 1, fontweight="bold")
            ax.plot([-0.5, N_PHEN - 0.5], [-0.5, N_PHEN - 0.5],
                    linestyle="--", linewidth=0.7, alpha=0.3,
                    color="#555555", zorder=4)

            if s["n"] > 0:
                ann = (f"diag: {s['frac_diag']:.0%}\n"
                       f"H: {s['H_norm']:.2f}")
            else:
                ann = "n=0"
            ax.text(0.98, 0.98, ann,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=ANNOT_FS - 1,
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=2),
                    zorder=5)

            _style_axis(ax)
            axes[edge] = ax

    cax = fig.add_subplot(gs[:, 3])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("n branches (log scale)",
                 rotation=270, labelpad=18, fontsize=LABEL_FS)
    cb.ax.tick_params(labelsize=TICK_FS)

    fig.suptitle("Per-edge phenotype switching modes",
                 fontsize=TITLE_FS + 1, fontweight="bold", y=0.94)
    fig.text(
        0.5, 0.04,
        "Each panel: 12×12 matrix where (row p, col q) = count of "
        "branches on that edge with dominant src phenotype p and "
        "dominant dst phenotype q. Mass on the diagonal = clones "
        "that did not switch dominant phenotype. Concentrated "
        "off-diagonal mass = stereotyped switching; diffuse "
        "off-diagonal mass = diverse switching. frac_diag and "
        "normalized entropy H reported per edge. "
        "Tick label colour = lineage (CD8 / CD4).",
        ha="center", va="bottom", fontsize=ANNOT_FS,
        color="dimgray", style="italic", wrap=True,
    )
    return fig, axes, last_im


fig, axes, _im = _build_figure(wspace=0.40, hspace=0.40)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()


def _ovl(b1, b2):
    return not (b1.x1 <= b2.x0 or b2.x1 <= b1.x0
                or b1.y1 <= b2.y0 or b2.y1 <= b1.y0)


def _tick_title_overlap(fig, axes):
    """True if any tick label in any axis overlaps any title in another axis."""
    title_bboxes = []
    for ax in axes.values():
        title = ax.title
        if title.get_text():
            title_bboxes.append(title.get_window_extent(renderer=renderer))
    for ax in axes.values():
        for tick in (list(ax.get_xticklabels())
                     + list(ax.get_yticklabels())):
            if not tick.get_text():
                continue
            tb = tick.get_window_extent(renderer=renderer)
            for tbox in title_bboxes:
                # Skip same-axis (title's own axis); we don't have an
                # easy axis-back-pointer from the title bbox, but
                # within-axis overlap is impossible by construction
                # because titles sit above the data area.
                if _ovl(tb, tbox):
                    return True
    return False


bumped = False
if _tick_title_overlap(fig, axes):
    print("  bumping wspace/hspace by 0.05 due to tick/title overlap...")
    plt.close(fig)
    fig, axes, _im = _build_figure(wspace=0.45, hspace=0.45)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bumped = True
    if _tick_title_overlap(fig, axes):
        raise AssertionError(
            "tick labels still overlap titles after wspace/hspace bump"
        )

png_path = OUT_DIR / "switching_heatmaps.png"
pdf_path = OUT_DIR / "switching_heatmaps.pdf"
fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")

# %%
# ---- Sanity checks & flags ----
print("Running sanity checks...")
checks = []
ok = True

# Sanity #1 already enforced at construction time.
checks.append("Sanity #1: switching_matrix.sum() == n_branches_used per edge "
              "— passed (else raised above).")
checks.append("Sanity #2: all src_dom / dst_dom in TCELL_PHENOTYPE_ORDER "
              "— passed (else raised above).")

# Sanity #3: audit table.
checks.append("\nSanity #3: per-edge audit table:")
checks.append(f"  {'edge':<14}{'n':>6}{'frac_diag':>11}{'H_norm':>9}"
              f"   top_transition (src→dst, n, frac)")
for edge in EDGES:
    s = stats_by_edge[edge]
    if s["n"] == 0:
        checks.append(f"  {edge[0]+'→'+edge[1]:<14}{0:>6}"
                      f"{'—':>11}{'—':>9}   (no branches)")
        continue
    top = f"{s['top_src']}→{s['top_dst']}, n={s['top_n']}, " \
          f"frac={s['top_frac']:.2f}"
    checks.append(f"  {edge[0]+'→'+edge[1]:<14}{s['n']:>6}"
                  f"{s['frac_diag']:>11.3f}{s['H_norm']:>9.3f}   {top}")

# Sanity #4: STEREOTYPED flags.
checks.append("\nSanity #4: STEREOTYPED flags "
              f"(nonzero < {STEREOTYPED_FLAG_NONZERO} AND "
              f"n > {STEREOTYPED_FLAG_MIN_BRANCHES}):")
flagged_any = False
for edge in EDGES:
    s = stats_by_edge[edge]
    if (s["n"] > STEREOTYPED_FLAG_MIN_BRANCHES
            and s["nonzero"] < STEREOTYPED_FLAG_NONZERO):
        msg = (f"  STEREOTYPED: {edge[0]}→{edge[1]} "
               f"(nonzero={s['nonzero']}, n={s['n']})")
        print(msg.strip())
        checks.append(msg)
        flagged_any = True
if not flagged_any:
    checks.append("  (none)")

# Sanity #5: DIFFUSE flags.
checks.append("\nSanity #5: DIFFUSE flags "
              f"(H_norm > {DIFFUSE_FLAG_ENTROPY} AND "
              f"frac_diag < {DIFFUSE_FLAG_FRAC_DIAG}):")
flagged_any = False
for edge in EDGES:
    s = stats_by_edge[edge]
    if s["n"] == 0:
        continue
    if (s["H_norm"] > DIFFUSE_FLAG_ENTROPY
            and s["frac_diag"] < DIFFUSE_FLAG_FRAC_DIAG):
        msg = (f"  DIFFUSE: {edge[0]}→{edge[1]} "
               f"(H={s['H_norm']:.3f}, frac_diag={s['frac_diag']:.3f})")
        print(msg.strip())
        checks.append(msg)
        flagged_any = True
if not flagged_any:
    checks.append("  (none)")

# Sanity #6: TIE-BREAK FRAGILE flags.
checks.append(f"\nSanity #6: TIE-BREAK FRAGILE flags "
              f"((n_ties_src + n_ties_dst) / (2·n) > {TIE_BREAK_WARN_FRAC}):")
flagged_any = False
for edge in EDGES:
    n = stats_by_edge[edge]["n"]
    if n == 0:
        continue
    tie_frac = (n_ties_src[edge] + n_ties_dst[edge]) / (2.0 * n)
    if tie_frac > TIE_BREAK_WARN_FRAC:
        msg = (f"  TIE-BREAK FRAGILE: {edge[0]}→{edge[1]} "
               f"(tie_frac={tie_frac:.3f}, "
               f"n_ties_src={n_ties_src[edge]}, "
               f"n_ties_dst={n_ties_dst[edge]}) "
               f"— prefer JSD figure as primary read.")
        print(msg.strip())
        checks.append(msg)
        flagged_any = True
if not flagged_any:
    checks.append("  (none)")

# Sanity #7: tick/title overlap outcome.
if bumped:
    checks.append("\nSanity #7: tick/title overlap detected at "
                  "wspace=hspace=0.40; bumped to 0.45 and re-rendered. OK.")
else:
    checks.append("\nSanity #7: no tick/title overlap at "
                  "wspace=hspace=0.40. OK.")

if not metrics_loaded_from_06:
    checks.append("\n[WARN] median_log2fc_norm column = NaN because "
                  "results/06_branch_empirics/edge_metrics_table.csv "
                  "was missing.")

checks.insert(0, f"OVERALL: {'PASS' if ok else 'FAIL'}\n")
(OUT_DIR / "render_check.txt").write_text("\n".join(checks))
print("\n".join(checks[:30]))
print(f"... full report: {OUT_DIR / 'render_check.txt'}")

# %%
print("Done.")
