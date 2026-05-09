# %%
"""Per-clone branch empirics across tissues and adjacent timepoints.

A "branch" = (trb, src tissue i, dst tissue j, t -> t+1) with >=2 cells
on each side. Same-tissue branches included so that "stayed vs left"
contributions of the same clone show up side by side.
"""
import sys
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
)

# %%
# ---- Config ----
DATA_PATH = REPO_ROOT / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"
OUT_DIR = REPO_ROOT / "results" / "06_branch_empirics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_NES = OUT_DIR / "_cache_nes.csv"

TISSUES = ("PBMC", "CSF", "TP")
EDGES = [
    ("PBMC", "PBMC"), ("CSF", "CSF"), ("TP", "TP"),
    ("PBMC", "CSF"), ("PBMC", "TP"), ("CSF", "PBMC"),
    ("CSF", "TP"),  ("TP", "PBMC"), ("TP", "CSF"),
]
EDGE_LABELS = [f"{a}→{b}" for a, b in EDGES]
BASELINE_EDGES = {(t, t) for t in TISSUES}
TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)

HALLMARK_KEEP_DISPLAY = [
    "E2F Targets", "G2-M Checkpoint",
    "Interferon Alpha Response", "Interferon Gamma Response",
    "TNF-alpha Signaling via NF-kB", "Hypoxia",
    "Oxidative Phosphorylation", "Glycolysis",
    "IL-2/STAT5 Signaling", "TGF-beta Signaling", "mTORC1 Signaling",
    "Fatty Acid Metabolism", "Cholesterol Homeostasis",
    "Myc Targets V1", "Apoptosis",
]
HALLMARK_DISPLAY_TO_NORM = {
    "E2F Targets": "E2F_TARGETS",
    "G2-M Checkpoint": "G2M_CHECKPOINT",
    "Interferon Alpha Response": "INTERFERON_ALPHA_RESPONSE",
    "Interferon Gamma Response": "INTERFERON_GAMMA_RESPONSE",
    "TNF-alpha Signaling via NF-kB": "TNFA_SIGNALING_VIA_NFKB",
    "Hypoxia": "HYPOXIA",
    "Oxidative Phosphorylation": "OXIDATIVE_PHOSPHORYLATION",
    "Glycolysis": "GLYCOLYSIS",
    "IL-2/STAT5 Signaling": "IL2_STAT5_SIGNALING",
    "TGF-beta Signaling": "TGF_BETA_SIGNALING",
    "mTORC1 Signaling": "MTORC1_SIGNALING",
    "Fatty Acid Metabolism": "FATTY_ACID_METABOLISM",
    "Cholesterol Homeostasis": "CHOLESTEROL_HOMEOSTASIS",
    "Myc Targets V1": "MYC_TARGETS_V1",
    "Apoptosis": "APOPTOSIS",
}

MAX_CELLS = 5000
RNG = np.random.default_rng(0)
GENE_SET = "MSigDB_Hallmark_2020"

# %%
# ---- Load ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
obs = adata.obs[["trb", "tissue", "timepoint", "phenotype",
                 "lineage", "patient"]].copy()
print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

# %%
# ---- Clone-tissue-time aggregations ----
ct = (obs.groupby(["trb", "tissue", "timepoint"], observed=True)
        .size().rename("n").reset_index())

ph = (obs.groupby(["trb", "tissue", "timepoint", "phenotype"], observed=True)
        .size().unstack("phenotype", fill_value=0))
for p in PHENOTYPES:
    if p not in ph.columns:
        ph[p] = 0
ph = ph[PHENOTYPES]
ph_norm = ph.div(ph.sum(axis=1), axis=0)
ph_dom = ph_norm.idxmax(axis=1)

clone_meta = (obs.groupby("trb", observed=True)
                .agg(lineage=("lineage", lambda s: s.mode().iat[0]),
                     patient=("patient", lambda s: s.mode().iat[0])))

# %%
# ---- Build branches ----
print("Building branches...")
records = []
for t1, t2 in TRANSITIONS:
    src = ct[(ct["timepoint"] == t1) & (ct["n"] >= 2)]
    dst = ct[(ct["timepoint"] == t2) & (ct["n"] >= 2)]
    m = src.merge(dst, on="trb", suffixes=("_src", "_dst"))
    if m.empty:
        continue
    for _, r in m.iterrows():
        s_key = (r["trb"], r["tissue_src"], t1)
        d_key = (r["trb"], r["tissue_dst"], t2)
        ps = ph_norm.loc[s_key].values + 1e-6
        pd_ = ph_norm.loc[d_key].values + 1e-6
        ps /= ps.sum(); pd_ /= pd_.sum()
        ds, dd = ph_dom.loc[s_key], ph_dom.loc[d_key]
        n_s, n_d = int(r["n_src"]), int(r["n_dst"])
        records.append({
            "trb": r["trb"],
            "src": r["tissue_src"], "dst": r["tissue_dst"],
            "t_src": t1, "t_dst": t2,
            "n_src": n_s, "n_dst": n_d,
            "lineage": clone_meta.loc[r["trb"], "lineage"],
            "patient": clone_meta.loc[r["trb"], "patient"],
            "log2fc": float(np.log2((n_d + 1) / (n_s + 1))),
            "jsd": float(jensenshannon(ps, pd_)),
            "dom_src": ds, "dom_dst": dd,
            "dom_switch": bool(ds != dd),
        })
branches = pd.DataFrame(records)
print(f"  n_branches: {len(branches)}")
branches.to_csv(OUT_DIR / "branches.csv", index=False)

n_per_edge = {(i, j): int(((branches["src"] == i) & (branches["dst"] == j)).sum())
              for i, j in EDGES}
print("  per edge: " + ", ".join(f"{i}->{j}:{n_per_edge[(i, j)]}"
                                  for i, j in EDGES))

# %%
# ---- Per-edge branch-weighted phenotype distributions ----
edge_src_dist, edge_dst_dist = {}, {}
pdist_rows = []
for i, j in EDGES:
    bsub = branches[(branches["src"] == i) & (branches["dst"] == j)]
    if bsub.empty:
        edge_src_dist[(i, j)] = pd.Series(0.0, index=PHENOTYPES)
        edge_dst_dist[(i, j)] = pd.Series(0.0, index=PHENOTYPES)
        continue
    s_keys = list(zip(bsub["trb"], bsub["src"], bsub["t_src"]))
    d_keys = list(zip(bsub["trb"], bsub["dst"], bsub["t_dst"]))
    sm = ph_norm.loc[s_keys].mean(axis=0).reindex(PHENOTYPES).fillna(0.0)
    dm = ph_norm.loc[d_keys].mean(axis=0).reindex(PHENOTYPES).fillna(0.0)
    edge_src_dist[(i, j)] = sm
    edge_dst_dist[(i, j)] = dm
    for p in PHENOTYPES:
        pdist_rows.append({"edge": f"{i}→{j}", "side": "src",
                           "phenotype": p, "frac": float(sm[p])})
        pdist_rows.append({"edge": f"{i}→{j}", "side": "dst",
                           "phenotype": p, "frac": float(dm[p])})
phenotype_dist = pd.DataFrame(pdist_rows)
phenotype_dist.to_csv(OUT_DIR / "phenotype_dist.csv", index=False)

# %%
# ---- Pathway track (cached) ----
groups = obs.groupby(["trb", "tissue", "timepoint"], observed=True).indices

if CACHE_NES.exists():
    nes_table = pd.read_csv(CACHE_NES)
    print(f"Loaded NES cache: {len(nes_table)} rows")
else:
    print("Running gseapy.prerank per edge...")
    X = adata.layers["log1p"]
    var_names = adata.var_names.values
    rows = []
    for i, j in EDGES:
        bsub = branches[(branches["src"] == i) & (branches["dst"] == j)]
        if bsub.empty:
            print(f"  {i}->{j}: 0 branches, skip")
            continue
        s_pos, d_pos = [], []
        for _, r in bsub.iterrows():
            s_pos.append(groups[(r["trb"], i, r["t_src"])])
            d_pos.append(groups[(r["trb"], j, r["t_dst"])])
        s = np.unique(np.concatenate(s_pos))
        d = np.unique(np.concatenate(d_pos))
        if len(s) > MAX_CELLS:
            s = RNG.choice(s, MAX_CELLS, replace=False)
        if len(d) > MAX_CELLS:
            d = RNG.choice(d, MAX_CELLS, replace=False)
        m_src = np.asarray(X[s].mean(axis=0)).ravel()
        m_dst = np.asarray(X[d].mean(axis=0)).ravel()
        rnk = pd.Series(m_dst - m_src, index=var_names)
        rnk = rnk[~rnk.index.duplicated()].sort_values(ascending=False)
        try:
            pre = gp.prerank(
                rnk=rnk, gene_sets=GENE_SET, outdir=None,
                seed=42, min_size=10, max_size=500,
                permutation_num=1000, no_plot=True,
            )
        except Exception as e:
            print(f"  {i}->{j}: prerank failed ({e}), skip")
            continue
        res = pre.res2d.copy()
        res["display"] = res["Term"].str.split("__").str[-1]
        keep = res[res["display"].isin(HALLMARK_KEEP_DISPLAY)]
        for _, row in keep.iterrows():
            rows.append({
                "edge": f"{i}→{j}",
                "pathway": HALLMARK_DISPLAY_TO_NORM[row["display"]],
                "NES": float(row["NES"]),
                "FDR": float(row["FDR q-val"]),
            })
        print(f"  {i}->{j}: kept {len(keep)} pathways "
              f"(src={len(s)}, dst={len(d)})")
    nes_table = pd.DataFrame(rows)
    nes_table.to_csv(CACHE_NES, index=False)

nes_table.to_csv(OUT_DIR / "nes_table.csv", index=False)

# %%
# ---- Main figure ----
pivot_nes = (nes_table.pivot(index="pathway", columns="edge", values="NES")
             .reindex(columns=EDGE_LABELS))
pivot_fdr = (nes_table.pivot(index="pathway", columns="edge", values="FDR")
             .reindex(columns=EDGE_LABELS))

if len(pivot_nes) >= 2:
    Z = linkage(pivot_nes.fillna(0).values, method="average")
    order = leaves_list(Z)
    pivot_nes = pivot_nes.iloc[order]
    pivot_fdr = pivot_fdr.iloc[order]

n_paths = max(len(pivot_nes), 1)

fig = plt.figure(figsize=(16.5, 8 + 0.35 * n_paths))
gs = GridSpec(
    4, 3,
    height_ratios=[4.4, 3.0, 1.6, max(2.0, 0.35 * n_paths)],
    width_ratios=[1.0, 0.025, 0.18],
    hspace=0.20, wspace=0.04,
)
ax_flow = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)
ax3 = fig.add_subplot(gs[3, 0], sharex=ax1)
cax = fig.add_subplot(gs[3, 1])
ax_legend = fig.add_subplot(gs[:, 2])
ax_legend.axis("off")

# ---- Flow panel: tissue circles + branch-count arrows ----
NODE_Y = {"PBMC": 0.82, "CSF": 0.50, "TP": 0.18}
SRC_X, DST_X = 0.30, 2.20
NODE_R = 0.10
ax_flow.set_xlim(0, 2.5)
ax_flow.set_ylim(0, 1)
ax_flow.set_aspect("equal", adjustable="box")
ax_flow.axis("off")

for t in TISSUES:
    for x in (SRC_X, DST_X):
        ax_flow.add_patch(Circle(
            (x, NODE_Y[t]), NODE_R,
            facecolor=TISSUE_COLORS[t], edgecolor="black", linewidth=0.8,
            zorder=3,
        ))
        fs = 12 if len(t) <= 3 else 10
        ax_flow.text(x, NODE_Y[t], t, ha="center", va="center",
                     fontsize=fs, fontweight="bold",
                     color="black", clip_on=False, zorder=4)
ax_flow.text(SRC_X, 0.98, "source (t)", ha="center", va="top",
             fontsize=9, color="dimgray", clip_on=False)
ax_flow.text(DST_X, 0.98, "destination (t+1)", ha="center", va="top",
             fontsize=9, color="dimgray", clip_on=False)

# Realize transforms so we can convert NODE_R (data) → points for shrink.
fig.canvas.draw()
_p0 = ax_flow.transData.transform((0.0, 0.0))
_p1 = ax_flow.transData.transform((0.0, NODE_R))
_r_pt = abs(_p1[1] - _p0[1]) * 72.0 / fig.dpi
_SHRINK_PT = _r_pt * 0.85

# Linewidth scale: thinnest arrow (min n) → 1pt, thickest → 5pt via
# sqrt(n). Cap at 2× the second-largest's width as a safety so a
# future outlier can't swamp the layout.
_ns_pos = [v for v in n_per_edge.values() if v > 0]
_ns_sorted = sorted(_ns_pos, reverse=True)
_max_n = _ns_sorted[0] if _ns_sorted else 1
_min_n = min(_ns_pos) if _ns_pos else 1
_scale_n = (_ns_sorted[1] if len(_ns_sorted) >= 2 else _max_n)
_LW_LO, _LW_HI = 1.0, 5.0
_sqrt_min, _sqrt_max = float(np.sqrt(_min_n)), float(np.sqrt(_max_n))
_sqrt_span = max(_sqrt_max - _sqrt_min, 1e-9)

def _lw_raw(n):
    if n <= 0:
        return _LW_LO * 0.6
    f = (float(np.sqrt(n)) - _sqrt_min) / _sqrt_span
    return _LW_LO + f * (_LW_HI - _LW_LO)

_LW_AT_SCALE = _lw_raw(_scale_n)
_LW_CAP = 2.0 * _LW_AT_SCALE

def _arrow_lw(n):
    return float(min(_lw_raw(n), _LW_CAP))

# Label position fractions along each arrow (chosen to spread the
# 6 cross labels through the central crossing region).
LABEL_FRAC = {
    ("PBMC", "PBMC"): 0.50,
    ("CSF",  "CSF"):  0.50,
    ("TP",   "TP"):   0.50,
    ("PBMC", "CSF"):  0.28,
    ("PBMC", "TP"):   0.38,
    ("CSF",  "PBMC"): 0.62,
    ("CSF",  "TP"):   0.72,
    ("TP",   "PBMC"): 0.45,
    ("TP",   "CSF"):  0.55,
}

for i, j in EDGES:
    n = n_per_edge[(i, j)]
    lw = _arrow_lw(n)
    label = f"{i} → {j}"
    posA = (SRC_X, NODE_Y[i])
    posB = (DST_X, NODE_Y[j])
    arr = FancyArrowPatch(
        posA=posA, posB=posB,
        connectionstyle="arc3,rad=0",
        arrowstyle="-|>", mutation_scale=10,
        color="dimgray", linewidth=lw,
        shrinkA=_SHRINK_PT, shrinkB=_SHRINK_PT,
        zorder=2,
    )
    ax_flow.add_patch(arr)

    dx, dy = posB[0] - posA[0], posB[1] - posA[1]
    L = float(np.hypot(dx, dy))
    ux, uy = (dx / L, dy / L) if L > 0 else (1.0, 0.0)
    # Visible line endpoints at the circle boundaries (data coords).
    vstart = (posA[0] + ux * NODE_R, posA[1] + uy * NODE_R)
    vend   = (posB[0] - ux * NODE_R, posB[1] - uy * NODE_R)
    f = LABEL_FRAC[(i, j)]
    lx = vstart[0] + (vend[0] - vstart[0]) * f
    ly = vstart[1] + (vend[1] - vstart[1]) * f
    angle_deg = float(np.degrees(np.arctan2(uy, ux)))
    ax_flow.text(
        lx, ly, label,
        ha="center", va="center",
        fontsize=7, color="black",
        rotation=angle_deg, transform_rotates_text=True,
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="none", alpha=0.85),
        clip_on=False, zorder=5,
    )

# ---- Top: clonality violin ----
data = []
for i, j in EDGES:
    sub = branches[(branches["src"] == i) & (branches["dst"] == j)]
    data.append(sub["log2fc"].values if len(sub) else np.array([0.0]))

vp = ax1.violinplot(data, positions=range(9), widths=0.78,
                    showmedians=True, showextrema=False)
for pc, (i, j) in zip(vp["bodies"], EDGES):
    pc.set_facecolor(TISSUE_COLORS[j])
    pc.set_edgecolor("black")
    pc.set_alpha(0.35)
if "cmedians" in vp:
    vp["cmedians"].set_color("black")

_all_raw = np.log(branches["n_src"].values + branches["n_dst"].values + 1) \
    if len(branches) else np.array([0.0])
_raw_lo, _raw_hi = float(_all_raw.min()), float(_all_raw.max())

for pos, (i, j) in enumerate(EDGES):
    sub = branches[(branches["src"] == i) & (branches["dst"] == j)]
    if len(sub) == 0:
        continue
    jit = (RNG.random(len(sub)) - 0.5) * 0.5
    raw = np.log(sub["n_src"].values + sub["n_dst"].values + 1)
    if _raw_hi > _raw_lo:
        sizes = 2 + (raw - _raw_lo) / (_raw_hi - _raw_lo) * (15 - 2)
    else:
        sizes = np.full_like(raw, 6.0)
    ax1.scatter(pos + jit, sub["log2fc"].values, s=sizes, alpha=0.35,
                color=TISSUE_COLORS[j], edgecolors="none", zorder=3)

ax1.axhline(0, color="gray", lw=0.6, linestyle="--")
ax1.set_ylabel("log2((n_dst+1)/(n_src+1))")
for pos, (i, j) in enumerate(EDGES):
    ax1.text(pos, 0.01, f"n={n_per_edge[(i, j)]}",
             transform=ax1.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=7, color="dimgray")

# ---- Middle: paired stacked phenotype bars ----
BAR_W = 0.35
for pos, (i, j) in enumerate(EDGES):
    sm = edge_src_dist[(i, j)]
    dm = edge_dst_dist[(i, j)]
    sb = db = 0.0
    for p in PHENOTYPES:
        c = TCELL_PHENOTYPE_COLORS[p]
        ax2.bar(pos - 0.2, sm[p], bottom=sb, width=BAR_W,
                color=c, edgecolor="white", linewidth=0.3)
        ax2.bar(pos + 0.2, dm[p], bottom=db, width=BAR_W,
                color=c, edgecolor="white", linewidth=0.3)
        sb += sm[p]; db += dm[p]
    rot = 0 if max(len(i), len(j)) <= 4 else 90
    if sm.sum() > 0:
        ax2.text(pos - 0.2, -0.03, i,
                 transform=ax2.get_xaxis_transform(),
                 ha="center", va="top", fontsize=6.5,
                 color=TISSUE_COLORS[i], rotation=rot)
    if dm.sum() > 0:
        ax2.text(pos + 0.2, -0.03, j,
                 transform=ax2.get_xaxis_transform(),
                 ha="center", va="top", fontsize=6.5,
                 color=TISSUE_COLORS[j], rotation=rot)
ax2.axhline(0.5, color="gray", lw=0.5, linestyle="--", alpha=0.6)
ax2.set_ylim(0, 1)
ax2.set_yticks([])
ax2.set_ylabel("phenotype frac\n(branch-weighted)", fontsize=9)

# ---- Bottom: pathway dot heatmap ----
xs, ys, ss, cs = [], [], [], []
if len(pivot_nes) and pivot_nes.notna().values.any():
    vmax = float(np.nanmax(np.abs(pivot_nes.values)))
    for yi, pname in enumerate(pivot_nes.index):
        for xi, e in enumerate(EDGE_LABELS):
            nes = pivot_nes.iat[yi, xi]
            fdr = pivot_fdr.iat[yi, xi]
            if pd.isna(nes) or pd.isna(fdr) or fdr >= 0.1:
                continue
            xs.append(xi); ys.append(yi)
            ss.append(-np.log10(max(fdr, 1e-6)) * 35)
            cs.append(nes)
else:
    vmax = 1.0

if xs:
    sc_h = ax3.scatter(xs, ys, s=ss, c=cs, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       edgecolors="k", linewidths=0.4)
    cbar = fig.colorbar(sc_h, cax=cax)
    cbar.set_label("NES", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
else:
    cax.axis("off")

ax3.set_yticks(range(len(pivot_nes.index)))
ax3.set_yticklabels(pivot_nes.index, fontsize=8)
ax3.set_ylim(-0.5, max(len(pivot_nes.index), 1) - 0.5)
ax3.set_xlabel("edge")

# ---- Shared x-axis cosmetics ----
for ax in (ax1, ax2):
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", length=0)

ax3.set_xticks(range(9))
ax3.set_xticklabels(EDGE_LABELS, rotation=35, ha="right")
for tick, edge in zip(ax3.get_xticklabels(), EDGES):
    tick.set_color("gray" if edge in BASELINE_EDGES else "black")
    if edge not in BASELINE_EDGES:
        tick.set_fontweight("bold")
ax3.set_xlim(-0.6, 8.6)

# ---- Shared phenotype legend (dedicated right column) ----
legend_handles = [
    Patch(facecolor=TCELL_PHENOTYPE_COLORS[p], edgecolor="white",
          label=TCELL_PHENOTYPE_LABELS.get(p, p))
    for p in PHENOTYPES
]
ax_legend.legend(
    handles=legend_handles, loc="center left",
    bbox_to_anchor=(0.0, 0.5), fontsize=8,
    frameon=False, title="Phenotype", title_fontsize=9,
    borderaxespad=0.0,
)

fig.suptitle("Per-clone branch empirics", fontsize=13, fontweight="bold",
             x=0.5, y=0.995)
fig.text(0.5, 0.972, "violins colored by destination tissue",
         ha="center", va="top", fontsize=9, color="dimgray", style="italic")

png_path = OUT_DIR / "branch_empirics_main.png"
pdf_path = OUT_DIR / "branch_empirics_main.pdf"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# %%
print("Done.")
