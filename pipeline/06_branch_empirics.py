# %%
"""Per-clone branch empirics across tissues and adjacent timepoints.

A "branch" = (trb, src tissue i, dst tissue j, t -> t+1) with >=2 cells
on each side. The flow panel and violin are driven by depth-corrected
log2 fold-changes computed against the (patient, tissue, timepoint)
clonotyped totals, with a Haldane PSEUDOCOUNT and an n_src filter.
"""
import sys
import warnings
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.clonality import compute_clonality
from modules.myeloid_groups import regroup_composition
from modules.style import (
    MYELOID_PHENOTYPE_COLORS,
    MYELOID_PHENOTYPE_LABELS,
    MYELOID_PHENOTYPE_ORDER,
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
)

# %%
# ---- Global config (single source of truth) ----
EXPANSION_CMAP = "PiYG"            # divergent: magenta→yellow→green
TRAFFIC_CMAP   = "plasma"          # sequential: purple→pink→yellow
EXPANSION_VLIM_AUTO = True         # symmetric vlim from data
EXPANSION_VLIM_FALLBACK = 1.5      # if all medians small, use ±1.5
LW_MIN, LW_MAX = 1.0, 5.0          # shared by both flow graphs
NEUTRAL_GREY = "#9e9e9e"           # arrows / outlines where the value is undefined
PSEUDOCOUNT = 0.5                  # Haldane correction
MIN_N_SRC = 3                      # filter low-confidence source clones
EXPAND_THRESHOLD = 0.5             # |log2fc_norm| > 0.5 → expanding/contracting
BAR_W = 0.38                       # paired-bar width (myeloid row)
ROW_GAP = 0.06                     # gridspec hspace base
ANNOT_FS = 8
LABEL_FS = 11
TITLE_FS = 13
DPI = 200

DATA_PATH = REPO_ROOT / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"
MYELOID_COMP_CSV = (REPO_ROOT / "results" / "10_temporal_scores"
                    / "temporal_composition_myeloid.csv")
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
MYELOID_PHS = list(MYELOID_PHENOTYPE_ORDER)

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


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.15, linewidth=0.6)


# %%
# ---- Load ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
obs = adata.obs[["trb", "tissue", "timepoint", "phenotype",
                 "lineage", "patient"]].copy()
print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

# Cache (patient, tissue, timepoint) → total clonotyped cells. Reused
# everywhere depth-correction is applied.
N_sample = (obs.groupby(["patient", "tissue", "timepoint"], observed=True)
              .size().to_dict())

# %%
# ---- Clone-tissue-time aggregations ----
ct = (obs.groupby(["trb", "tissue", "timepoint"], observed=True)
        .size().rename("n").reset_index())

# Patient-aware aggregation drives the trafficking source pool: a clone
# in (patient, src_tissue, t) with n_src >= MIN_N_SRC contributes to the
# pool whether or not it ever appears in branches_used.
ct_patient = (obs.groupby(["patient", "trb", "tissue", "timepoint"],
                           observed=True)
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
# ---- Build branches (incl. depth-corrected log2fc_norm) ----
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
        patient = clone_meta.loc[r["trb"], "patient"]
        N_src = N_sample.get((patient, r["tissue_src"], t1), 0)
        N_dst = N_sample.get((patient, r["tissue_dst"], t2), 0)
        p_src = (n_s + PSEUDOCOUNT) / (N_src + PSEUDOCOUNT)
        p_dst = (n_d + PSEUDOCOUNT) / (N_dst + PSEUDOCOUNT)
        log2fc_norm = float(np.log2(p_dst / p_src))
        records.append({
            "trb": r["trb"],
            "src": r["tissue_src"], "dst": r["tissue_dst"],
            "t_src": t1, "t_dst": t2,
            "n_src": n_s, "n_dst": n_d,
            "N_src": int(N_src), "N_dst": int(N_dst),
            "lineage": clone_meta.loc[r["trb"], "lineage"],
            "patient": patient,
            "log2fc": float(np.log2((n_d + 1) / (n_s + 1))),
            "log2fc_norm": log2fc_norm,
            "jsd": float(jensenshannon(ps, pd_)),
            "dom_src": ds, "dom_dst": dd,
            "dom_switch": bool(ds != dd),
        })
branches = pd.DataFrame(records)
print(f"  n_branches: {len(branches)}")
branches.to_csv(OUT_DIR / "branches.csv", index=False)

branches_used = branches[branches["n_src"] >= MIN_N_SRC].copy()
print(f"  filter n_src >= {MIN_N_SRC}: "
      f"{len(branches_used)}/{len(branches)} kept")

n_per_edge_total = {(i, j): int(((branches["src"] == i)
                                  & (branches["dst"] == j)).sum())
                    for i, j in EDGES}
n_per_edge_used  = {(i, j): int(((branches_used["src"] == i)
                                  & (branches_used["dst"] == j)).sum())
                    for i, j in EDGES}

# %%
# ---- Per-edge T cell phenotype distributions (UNCHANGED — uses ALL branches) ----
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
# ---- Myeloid composition (regrouped) joined per USED branch ----
print("Loading myeloid composition...")
mraw = pd.read_csv(MYELOID_COMP_CSV)
mcomp = regroup_composition(mraw)
mcomp["timepoint"] = mcomp["timepoint"].astype(str)

unmapped = set(mcomp["phenotype"].unique()) - set(MYELOID_PHENOTYPE_COLORS)
if unmapped:
    raise KeyError(
        "MYELOID_PHENOTYPE_COLORS missing palette entries for groups "
        f"emitted by myeloid_groups.MYELOID_GROUPS: {sorted(unmapped)}"
    )

sample_keys = ["patient", "tissue", "timepoint"]
totals = mcomp.groupby(sample_keys, observed=True)["frac"].transform("sum")
mcomp["frac_norm"] = mcomp["frac"] / totals.replace(0, np.nan)

mcomp_pivot = mcomp.pivot_table(
    index=sample_keys, columns="phenotype",
    values="frac_norm", aggfunc="first", fill_value=0.0,
)
for p in MYELOID_PHS:
    if p not in mcomp_pivot.columns:
        mcomp_pivot[p] = 0.0
mcomp_pivot = mcomp_pivot[MYELOID_PHS]

print(f"  myeloid samples: {len(mcomp_pivot)}, "
      f"phenotypes: {len(MYELOID_PHS)}")

edge_src_myeloid = {edge: np.zeros(len(MYELOID_PHS)) for edge in EDGES}
edge_dst_myeloid = {edge: np.zeros(len(MYELOID_PHS)) for edge in EDGES}
n_with_myeloid = {edge: 0 for edge in EDGES}
mp_index = mcomp_pivot.index

for _, b in branches_used.iterrows():
    src_key = (b["patient"], b["src"], b["t_src"])
    dst_key = (b["patient"], b["dst"], b["t_dst"])
    if src_key not in mp_index or dst_key not in mp_index:
        continue
    edge = (b["src"], b["dst"])
    edge_src_myeloid[edge] += mcomp_pivot.loc[src_key].to_numpy()
    edge_dst_myeloid[edge] += mcomp_pivot.loc[dst_key].to_numpy()
    n_with_myeloid[edge] += 1

for edge in EDGES:
    n = n_with_myeloid[edge]
    if n > 0:
        edge_src_myeloid[edge] = edge_src_myeloid[edge] / n
        edge_dst_myeloid[edge] = edge_dst_myeloid[edge] / n

# %%
# ---- Pathway track (cached, UNCHANGED — uses ALL branches) ----
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
# ---- Per-edge expansion stats (depth-corrected) ----
edge_stats = {}
for i, j in EDGES:
    bsub = branches_used[(branches_used["src"] == i)
                          & (branches_used["dst"] == j)]
    if bsub.empty:
        edge_stats[(i, j)] = dict(
            n_branches_used=0, median_log2fc_norm=0.0, mean_log2fc_norm=0.0,
            frac_expanding=np.nan, frac_contracting=np.nan,
        )
        continue
    lf = bsub["log2fc_norm"].to_numpy()
    edge_stats[(i, j)] = dict(
        n_branches_used=int(len(bsub)),
        median_log2fc_norm=float(np.median(lf)),
        mean_log2fc_norm=float(np.mean(lf)),
        frac_expanding=float((lf > EXPAND_THRESHOLD).mean()),
        frac_contracting=float((lf < -EXPAND_THRESHOLD).mean()),
    )

# %%
# ---- Per-edge trafficking rate ----
# rate(patient, t, edge i->j) = (unique clones from the (patient, i, t)
# pool with n_src >= MIN_N_SRC that take a USED branch on that edge) /
# (unique clones in that pool). traffic_rate_e is the mean over
# (patient, t) pairs that contribute (C_src > 0).
print("Computing per-edge trafficking rates...")
src_pool = ct_patient[ct_patient["n"] >= MIN_N_SRC].copy()
pool_by_pti = (src_pool.groupby(["patient", "tissue", "timepoint"],
                                 observed=True)["trb"]
                       .apply(lambda s: set(s)).to_dict())

bu_by_key = {}
for _, b in branches_used.iterrows():
    key = (b["patient"], b["src"], b["dst"], b["t_src"])
    bu_by_key.setdefault(key, set()).add(b["trb"])

T_SRC_VALID = {t1 for t1, _ in TRANSITIONS}

edge_traffic_rate = {}
edge_n_clones_on_edge = {}
edge_n_clones_in_source_pools_summed = {}

for i, j in EDGES:
    rates = []
    pool_sum = 0
    on_edge_clones = set()
    for (patient, tissue, tp), pool in pool_by_pti.items():
        if tissue != i or tp not in T_SRC_VALID:
            continue
        C_src = len(pool)
        if C_src == 0:
            continue
        edge_clones = bu_by_key.get((patient, i, j, tp), set()) & pool
        rates.append(len(edge_clones) / C_src)
        pool_sum += C_src
        on_edge_clones |= edge_clones
    edge_traffic_rate[(i, j)] = float(np.mean(rates)) if rates else 0.0
    edge_n_clones_on_edge[(i, j)] = int(len(on_edge_clones))
    edge_n_clones_in_source_pools_summed[(i, j)] = int(pool_sum)

# %%
# ---- Sanity overlay: per-(patient, tissue, t) sample clonality ----
print("Computing sample clonalities for sanity overlay...")
sample_clon = compute_clonality(obs, ["patient", "tissue", "timepoint"], n_boot=1)
sample_clon["timepoint"] = sample_clon["timepoint"].astype(str)
sc_lookup = (sample_clon.set_index(["patient", "tissue", "timepoint"])
             ["clonality"].to_dict())

edge_delta = {edge: [] for edge in EDGES}
for _, b in branches_used.iterrows():
    s = sc_lookup.get((b["patient"], b["src"], b["t_src"]), np.nan)
    d = sc_lookup.get((b["patient"], b["dst"], b["t_dst"]), np.nan)
    if not (np.isnan(s) or np.isnan(d)):
        edge_delta[(b["src"], b["dst"])].append(d - s)
mean_delta_clon = {e: (float(np.mean(v)) if v else np.nan)
                   for e, v in edge_delta.items()}

# %%
# ---- Audit table ----
edge_rows = []
for i, j in EDGES:
    s = edge_stats[(i, j)]
    edge_rows.append(dict(
        edge=f"{i}→{j}",
        n_branches_total=n_per_edge_total[(i, j)],
        n_branches_used=n_per_edge_used[(i, j)],
        n_branches_with_myeloid=n_with_myeloid[(i, j)],
        traffic_rate=edge_traffic_rate[(i, j)],
        n_unique_clones_on_edge=edge_n_clones_on_edge[(i, j)],
        n_unique_clones_in_source_pools_summed=(
            edge_n_clones_in_source_pools_summed[(i, j)]),
        median_log2fc_norm=s["median_log2fc_norm"],
        mean_log2fc_norm=s["mean_log2fc_norm"],
        frac_expanding=s["frac_expanding"],
        frac_contracting=s["frac_contracting"],
        mean_delta_sample_clonality=mean_delta_clon[(i, j)],
    ))
pd.DataFrame(edge_rows).to_csv(
    OUT_DIR / "edge_metrics_table.csv", index=False)

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

# ---- Color norms shared between flow panels and violins ----
expansion_cmap = plt.get_cmap(EXPANSION_CMAP)
traffic_cmap   = plt.get_cmap(TRAFFIC_CMAP)

_medians = np.array([edge_stats[edge]["median_log2fc_norm"] for edge in EDGES])
_max_abs = float(np.max(np.abs(_medians))) if len(_medians) else 0.0
if EXPANSION_VLIM_AUTO:
    expansion_vmax = max(_max_abs, EXPANSION_VLIM_FALLBACK)
else:
    expansion_vmax = EXPANSION_VLIM_FALLBACK
expansion_vmin = -expansion_vmax
expansion_norm = TwoSlopeNorm(
    vmin=expansion_vmin, vcenter=0.0, vmax=expansion_vmax)

_rates = np.array([edge_traffic_rate[edge] for edge in EDGES])
traffic_max = float(np.max(_rates)) if _rates.size else 0.0
traffic_norm = Normalize(vmin=0.0, vmax=max(traffic_max, 1e-9))


def _expansion_color(edge):
    if edge_stats[edge]["n_branches_used"] == 0:
        return to_rgba(NEUTRAL_GREY)
    return expansion_cmap(expansion_norm(
        edge_stats[edge]["median_log2fc_norm"]))


def _expansion_lw(edge):
    if edge_stats[edge]["n_branches_used"] == 0:
        return LW_MIN
    val = abs(edge_stats[edge]["median_log2fc_norm"]) / expansion_vmax
    return float(LW_MIN + (LW_MAX - LW_MIN) * min(val, 1.0))


def _expansion_annot(edge):
    s = edge_stats[edge]
    if s["n_branches_used"] == 0:
        return "n=0"
    return (f"{s['median_log2fc_norm']:+.2f}\n"
            f"↑{s['frac_expanding']:.2f} ↓{s['frac_contracting']:.2f}")


def _traffic_color(edge):
    if n_per_edge_used[edge] == 0:
        return to_rgba(NEUTRAL_GREY)
    return traffic_cmap(traffic_norm(edge_traffic_rate[edge]))


def _traffic_lw(edge):
    if n_per_edge_used[edge] == 0 or traffic_max <= 0:
        return LW_MIN
    return float(LW_MIN + (LW_MAX - LW_MIN)
                 * (edge_traffic_rate[edge] / traffic_max))


def _traffic_annot(edge):
    return f"{edge_traffic_rate[edge]:.2f}\nn={n_per_edge_used[edge]}"


# ---- Figure layout ----
FIG_WIDTH = 16.5 * 1.30
FIG_HEIGHT = 11.5 + 0.35 * n_paths
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

GS_LEFT, GS_RIGHT = 0.06, 0.78
LEGEND_LEFT, LEGEND_W = 0.81, 0.17
GAP_X = 0.025
GRAPH_W = (GS_RIGHT - GS_LEFT - GAP_X) / 2.0
LEFT_GRAPH_LEFT  = GS_LEFT
RIGHT_GRAPH_LEFT = GS_LEFT + GRAPH_W + GAP_X

TOP_GRAPH_TOP    = 0.96
TOP_GRAPH_BOTTOM = 0.72
CBAR_HEIGHT      = 0.013
CBAR_BOTTOM      = TOP_GRAPH_BOTTOM - 0.04 - CBAR_HEIGHT
GS_TOP           = CBAR_BOTTOM - 0.06

ax_traffic = fig.add_axes([LEFT_GRAPH_LEFT,  TOP_GRAPH_BOTTOM,
                           GRAPH_W, TOP_GRAPH_TOP - TOP_GRAPH_BOTTOM])
ax_expand  = fig.add_axes([RIGHT_GRAPH_LEFT, TOP_GRAPH_BOTTOM,
                           GRAPH_W, TOP_GRAPH_TOP - TOP_GRAPH_BOTTOM])
cax_traffic = fig.add_axes([LEFT_GRAPH_LEFT,  CBAR_BOTTOM,
                            GRAPH_W, CBAR_HEIGHT])
cax_expand  = fig.add_axes([RIGHT_GRAPH_LEFT, CBAR_BOTTOM,
                            GRAPH_W, CBAR_HEIGHT])

gs = GridSpec(
    4, 2,
    height_ratios=[3.0, 1.6, 1.6, max(2.0, 0.35 * n_paths)],
    width_ratios=[1.0, 0.025],
    left=GS_LEFT + 0.01, right=GS_RIGHT, top=GS_TOP, bottom=0.05,
    hspace=ROW_GAP * 4.0, wspace=0.04,
)
ax1   = fig.add_subplot(gs[0, 0])                 # violin
ax_my = fig.add_subplot(gs[1, 0], sharex=ax1)     # myeloid bars
ax2   = fig.add_subplot(gs[2, 0], sharex=ax1)     # T cell bars
ax3   = fig.add_subplot(gs[3, 0], sharex=ax1)     # pathway dots
cax   = fig.add_subplot(gs[3, 1])

ax_my_legend    = fig.add_axes([LEGEND_LEFT, 0.34, LEGEND_W, 0.28])
ax_tcell_legend = fig.add_axes([LEGEND_LEFT, 0.06, LEGEND_W, 0.26])
ax_my_legend.axis("off")
ax_tcell_legend.axis("off")

# ---- Flow graphs (traffic + expansion) ----
NODE_Y = {"PBMC": 0.82, "CSF": 0.50, "TP": 0.18}
SRC_X, DST_X = 0.30, 2.20
NODE_R = 0.10
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


def _draw_flow_graph(ax, color_fn, lw_fn, annot_fn, title):
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=LABEL_FS, fontweight="bold", pad=6)

    for t in TISSUES:
        for x in (SRC_X, DST_X):
            ax.add_patch(Circle(
                (x, NODE_Y[t]), NODE_R,
                facecolor=TISSUE_COLORS[t], edgecolor="black",
                linewidth=0.8, zorder=3,
            ))
            fs = 12 if len(t) <= 3 else 10
            ax.text(x, NODE_Y[t], t, ha="center", va="center",
                    fontsize=fs, fontweight="bold",
                    color="black", clip_on=False, zorder=4)
    ax.text(SRC_X, 0.98, "source (t)", ha="center", va="top",
            fontsize=9, color="dimgray", clip_on=False)
    ax.text(DST_X, 0.98, "destination (t+1)", ha="center", va="top",
            fontsize=9, color="dimgray", clip_on=False)

    fig.canvas.draw()
    p0 = ax.transData.transform((0.0, 0.0))
    p1 = ax.transData.transform((0.0, NODE_R))
    r_pt = abs(p1[1] - p0[1]) * 72.0 / fig.dpi
    shrink_pt = r_pt * 0.85

    cols_out, lws_out = {}, {}
    for i, j in EDGES:
        col = color_fn((i, j))
        lw = lw_fn((i, j))
        cols_out[(i, j)] = to_rgba(col)
        lws_out[(i, j)] = lw

        posA = (SRC_X, NODE_Y[i])
        posB = (DST_X, NODE_Y[j])
        arr = FancyArrowPatch(
            posA=posA, posB=posB,
            connectionstyle="arc3,rad=0",
            arrowstyle="-|>", mutation_scale=11,
            color=col, linewidth=lw,
            shrinkA=shrink_pt, shrinkB=shrink_pt,
            zorder=2,
        )
        ax.add_patch(arr)

        dx, dy = posB[0] - posA[0], posB[1] - posA[1]
        L = float(np.hypot(dx, dy))
        ux, uy = (dx / L, dy / L) if L > 0 else (1.0, 0.0)
        vstart = (posA[0] + ux * NODE_R, posA[1] + uy * NODE_R)
        vend   = (posB[0] - ux * NODE_R, posB[1] - uy * NODE_R)
        f = LABEL_FRAC[(i, j)]
        lx = vstart[0] + (vend[0] - vstart[0]) * f
        ly = vstart[1] + (vend[1] - vstart[1]) * f
        angle_deg = float(np.degrees(np.arctan2(uy, ux)))
        ax.text(
            lx, ly, annot_fn((i, j)),
            ha="center", va="center",
            fontsize=ANNOT_FS - 1, color="black",
            rotation=angle_deg, transform_rotates_text=True,
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="white", edgecolor="none", alpha=0.85),
            clip_on=False, zorder=5,
        )
    return cols_out, lws_out


traffic_arrow_colors, traffic_arrow_lws = _draw_flow_graph(
    ax_traffic, _traffic_color, _traffic_lw, _traffic_annot,
    "trafficking volume",
)
expand_arrow_colors, expand_arrow_lws = _draw_flow_graph(
    ax_expand, _expansion_color, _expansion_lw, _expansion_annot,
    "expansion",
)

# ---- Colorbars below each top graph ----
sm_traffic = ScalarMappable(norm=traffic_norm, cmap=traffic_cmap)
sm_traffic.set_array([])
cb_traffic = fig.colorbar(sm_traffic, cax=cax_traffic, orientation="horizontal")
cb_traffic.set_label("trafficking rate (clones / source pool)",
                     fontsize=LABEL_FS)
cb_traffic.ax.tick_params(labelsize=ANNOT_FS)

sm_expand = ScalarMappable(norm=expansion_norm, cmap=expansion_cmap)
sm_expand.set_array([])
cb_expand = fig.colorbar(sm_expand, cax=cax_expand, orientation="horizontal")
cb_expand.set_label("median log₂ fold-change (depth-corrected)",
                    fontsize=LABEL_FS)
cb_expand.ax.tick_params(labelsize=ANNOT_FS)
cb_expand.set_ticks([expansion_vmin, 0.0, expansion_vmax])
cb_expand.ax.axvline(0.0, color="black", linewidth=0.9, alpha=0.7)
for tlbl in cb_expand.ax.get_xticklabels():
    if tlbl.get_text().strip() in ("0", "0.0", "0.00"):
        tlbl.set_fontweight("bold")

# ---- Top: clonality violin (now: depth-corrected log2fc, USED branches) ----
data = []
for i, j in EDGES:
    sub = branches_used[(branches_used["src"] == i)
                         & (branches_used["dst"] == j)]
    data.append(sub["log2fc_norm"].values if len(sub) else np.array([0.0]))

vp = ax1.violinplot(data, positions=range(9), widths=0.78,
                    showmedians=True, showextrema=False)
violin_facecolors = {}
for pc, edge in zip(vp["bodies"], EDGES):
    fc = _expansion_color(edge)
    pc.set_facecolor(fc)
    pc.set_edgecolor(NEUTRAL_GREY)
    pc.set_linewidth(0.6)
    pc.set_alpha(1.0)
    violin_facecolors[edge] = to_rgba(fc)
if "cmedians" in vp:
    vp["cmedians"].set_color("black")

if len(branches_used):
    _all_raw = np.log(branches_used["n_src"].values
                       + branches_used["n_dst"].values + 1)
else:
    _all_raw = np.array([0.0])
_raw_lo, _raw_hi = float(_all_raw.min()), float(_all_raw.max())

for pos, (i, j) in enumerate(EDGES):
    sub = branches_used[(branches_used["src"] == i)
                         & (branches_used["dst"] == j)]
    if len(sub) == 0:
        continue
    jit = (RNG.random(len(sub)) - 0.5) * 0.5
    raw = np.log(sub["n_src"].values + sub["n_dst"].values + 1)
    if _raw_hi > _raw_lo:
        sizes = 2 + (raw - _raw_lo) / (_raw_hi - _raw_lo) * (15 - 2)
    else:
        sizes = np.full_like(raw, 6.0)
    ax1.scatter(pos + jit, sub["log2fc_norm"].values, s=sizes, alpha=0.55,
                color="#333333", edgecolors="none", zorder=3)

ax1.axhline(0, color="gray", lw=0.6, linestyle="--", alpha=0.4)
ax1.set_ylabel("log₂ fold-change (depth-corrected)", fontsize=LABEL_FS)
for pos, (i, j) in enumerate(EDGES):
    ax1.text(pos, 0.01, f"n_used={n_per_edge_used[(i, j)]}",
             transform=ax1.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=7, color="dimgray")
_style_axis(ax1)

# ---- Myeloid paired stacked bars ----
for pos, (i, j) in enumerate(EDGES):
    sm = edge_src_myeloid[(i, j)]
    dm = edge_dst_myeloid[(i, j)]
    sb = db = 0.0
    for k, p in enumerate(MYELOID_PHS):
        c = MYELOID_PHENOTYPE_COLORS[p]
        ax_my.bar(pos - BAR_W / 2, sm[k], bottom=sb, width=BAR_W,
                  color=c, edgecolor="white", linewidth=0.3)
        ax_my.bar(pos + BAR_W / 2, dm[k], bottom=db, width=BAR_W,
                  color=c, edgecolor="white", linewidth=0.3)
        sb += sm[k]
        db += dm[k]
ax_my.axhline(0.5, color="gray", lw=0.5, linestyle="--", alpha=0.6)
ax_my.set_ylim(0, 1)
ax_my.set_yticks([])
ax_my.set_ylabel("myeloid src | dst", fontsize=LABEL_FS)
_style_axis(ax_my)

# ---- T cell paired stacked phenotype bars (UNCHANGED) ----
T_BAR_W = 0.35
for pos, (i, j) in enumerate(EDGES):
    sm = edge_src_dist[(i, j)]
    dm = edge_dst_dist[(i, j)]
    sb = db = 0.0
    for p in PHENOTYPES:
        c = TCELL_PHENOTYPE_COLORS[p]
        ax2.bar(pos - 0.2, sm[p], bottom=sb, width=T_BAR_W,
                color=c, edgecolor="white", linewidth=0.3)
        ax2.bar(pos + 0.2, dm[p], bottom=db, width=T_BAR_W,
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
ax2.set_ylabel("T cell src | dst", fontsize=LABEL_FS)
_style_axis(ax2)

# ---- Bottom: pathway dot heatmap (UNCHANGED) ----
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
    cbar.set_label("NES", fontsize=ANNOT_FS)
    cbar.ax.tick_params(labelsize=7)
else:
    cax.axis("off")

ax3.set_yticks(range(len(pivot_nes.index)))
ax3.set_yticklabels(pivot_nes.index, fontsize=8)
ax3.set_ylim(-0.5, max(len(pivot_nes.index), 1) - 0.5)
ax3.set_xlabel("edge")
_style_axis(ax3)

for ax in (ax1, ax_my, ax2):
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", length=0)

ax3.set_xticks(range(9))
ax3.set_xticklabels(EDGE_LABELS, rotation=35, ha="right")
for tick, edge in zip(ax3.get_xticklabels(), EDGES):
    tick.set_color("gray" if edge in BASELINE_EDGES else "black")
    if edge not in BASELINE_EDGES:
        tick.set_fontweight("bold")
ax3.set_xlim(-0.6, 8.6)

# ---- Legends in dedicated fig.add_axes regions ----
my_handles = [
    Patch(facecolor=MYELOID_PHENOTYPE_COLORS[p], edgecolor="white",
          label=MYELOID_PHENOTYPE_LABELS[p])
    for p in MYELOID_PHS
]
ncol_my = 2 if len(my_handles) > 7 else 1
ax_my_legend.legend(
    handles=my_handles, loc="upper left",
    bbox_to_anchor=(0.0, 1.0), fontsize=ANNOT_FS,
    frameon=False, title="Myeloid", title_fontsize=ANNOT_FS + 1,
    borderaxespad=0.0, ncol=ncol_my,
    handlelength=1.2, handletextpad=0.4, columnspacing=0.8,
    labelspacing=0.4,
)

tcell_handles = [
    Patch(facecolor=TCELL_PHENOTYPE_COLORS[p], edgecolor="white",
          label=TCELL_PHENOTYPE_LABELS.get(p, p))
    for p in PHENOTYPES
]
ax_tcell_legend.legend(
    handles=tcell_handles, loc="upper left",
    bbox_to_anchor=(0.0, 1.0), fontsize=ANNOT_FS,
    frameon=False, title="T cell", title_fontsize=ANNOT_FS + 1,
    borderaxespad=0.0,
    handlelength=1.2, handletextpad=0.4, labelspacing=0.4,
)

fig.suptitle("Per-clone branch empirics",
             fontsize=TITLE_FS, fontweight="bold", x=0.5, y=0.99)
fig.text(
    0.5, 0.974,
    "left: trafficking rate per edge (plasma); right: depth-corrected "
    "expansion (PiYG, symmetric). violin fill = expansion arrow color; "
    f"n_src ≥ {MIN_N_SRC} for both metrics.",
    ha="center", va="top", fontsize=9, color="dimgray", style="italic",
)

png_path = OUT_DIR / "branch_empirics_main.png"
pdf_path = OUT_DIR / "branch_empirics_main.pdf"
fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# %%
# ---- Sanity checks ----
print("Running sanity checks...")
checks = []
ok = True

# 1. myeloid src/dst sums (each USED-branch-supplied edge sums to 1.0)
checks.append("Myeloid src/dst sums:")
for i, j in EDGES:
    if n_with_myeloid[(i, j)] == 0:
        checks.append(f"  [skip] {i}->{j}: 0 used branches with myeloid")
        continue
    s_sum = float(edge_src_myeloid[(i, j)].sum())
    d_sum = float(edge_dst_myeloid[(i, j)].sum())
    s_ok = abs(s_sum - 1.0) < 1e-6
    d_ok = abs(d_sum - 1.0) < 1e-6
    if not (s_ok and d_ok):
        ok = False
    checks.append(f"  [{'OK' if s_ok and d_ok else 'FAIL'}] "
                  f"{i}->{j}: src_sum={s_sum:.6f}, dst_sum={d_sum:.6f}")

# 2. per-edge counts
checks.append("\nPer-edge branch counts:")
checks.append(f"  {'edge':<14}{'total':>8}{'used':>8}{'myeloid':>10}"
              f"{'used/tot':>10}{'my/used':>10}")
for i, j in EDGES:
    n_t = n_per_edge_total[(i, j)]
    n_u = n_per_edge_used[(i, j)]
    n_m = n_with_myeloid[(i, j)]
    ut = (n_u / n_t) if n_t > 0 else 0.0
    mu = (n_m / n_u) if n_u > 0 else 0.0
    checks.append(f"  {i+'->'+j:<14}{n_t:>8}{n_u:>8}{n_m:>10}"
                  f"{ut:>10.3f}{mu:>10.3f}")

# 3. arrow linewidths (both panels)
checks.append("\nArrow linewidths in [LW_MIN, LW_MAX]:")
for label, lws in [("traffic", traffic_arrow_lws),
                   ("expand",  expand_arrow_lws)]:
    for edge, lw in lws.items():
        in_range = LW_MIN - 1e-9 <= lw <= LW_MAX + 1e-9
        if not in_range:
            ok = False
        checks.append(f"  [{'OK' if in_range else 'FAIL'}] "
                      f"{label} {edge[0]}->{edge[1]}: lw={lw:.3f}")

# 4. arrow colors are valid RGBA tuples (no fixed palette anymore;
# colors come from the plasma / PiYG colormaps with NEUTRAL_GREY for
# undefined edges).
checks.append("\nArrow colors valid RGBA:")
for label, cols in [("traffic", traffic_arrow_colors),
                    ("expand",  expand_arrow_colors)]:
    for edge, col in cols.items():
        ok_rgba = (isinstance(col, tuple) and len(col) == 4
                   and all(-1e-9 <= c <= 1.0 + 1e-9 for c in col))
        if not ok_rgba:
            ok = False
        checks.append(f"  [{'OK' if ok_rgba else 'FAIL'}] "
                      f"{label} {edge[0]}->{edge[1]}: rgba={col}")

# 5. per-edge metric table
checks.append("\nPer-edge metrics:")
checks.append(f"  {'edge':<14}{'median_lfc':>14}{'frac_exp':>12}"
              f"{'frac_con':>12}")
for i, j in EDGES:
    s = edge_stats[(i, j)]
    fe = s["frac_expanding"]
    fc = s["frac_contracting"]
    fe_str = f"{fe:.3f}" if not np.isnan(fe) else "—"
    fc_str = f"{fc:.3f}" if not np.isnan(fc) else "—"
    checks.append(f"  {i+'->'+j:<14}"
                  f"{s['median_log2fc_norm']:>14.4f}"
                  f"{fe_str:>12}{fc_str:>12}")

# 6. sample-clonality sanity overlay + sign mismatch
checks.append("\nSample-clonality sanity overlay (NOT used to color):")
checks.append(f"  {'edge':<14}{'median_lfc':>14}{'mean_Δclon':>14}"
              f"{'mismatch':>16}")
for i, j in EDGES:
    m = edge_stats[(i, j)]["median_log2fc_norm"]
    d = mean_delta_clon[(i, j)]
    mismatch = ""
    if not np.isnan(m) and not np.isnan(d):
        if (m > 0 and d < 0) or (m < 0 and d > 0):
            mismatch = "SIGN MISMATCH"
    d_str = f"{d:.4f}" if not np.isnan(d) else "—"
    checks.append(f"  {i+'->'+j:<14}{m:>14.4f}{d_str:>14}{mismatch:>16}")

# 7. myeloid legend coverage
n_legend = len(my_handles)
n_phen   = len(MYELOID_PHS)
legend_ok = (n_legend == n_phen)
if not legend_ok:
    ok = False
checks.append(f"\n[{'OK' if legend_ok else 'FAIL'}] "
              f"myeloid legend entries: {n_legend} == phenotypes: {n_phen}")

# 8. legend axes vs data axes bbox intersection
fig.canvas.draw()
data_axes = [
    ("traffic_graph", ax_traffic),
    ("expand_graph",  ax_expand),
    ("cbar_traffic",  cax_traffic),
    ("cbar_expand",   cax_expand),
    ("violin",        ax1),
    ("myeloid",       ax_my),
    ("tcell",         ax2),
    ("pathway",       ax3),
    ("cax_pathway",   cax),
]
legend_axes = [
    ("myeloid_legend", ax_my_legend),
    ("tcell_legend",   ax_tcell_legend),
]


def _bbox(ax):
    return ax.get_position()


def _intersects(b1, b2):
    return not (b1.x1 <= b2.x0 or b2.x1 <= b1.x0
                or b1.y1 <= b2.y0 or b2.y1 <= b1.y0)


checks.append("\nLegend ↔ data axis bbox checks:")
for la_name, la in legend_axes:
    lb = _bbox(la)
    for da_name, da in data_axes:
        db = _bbox(da)
        clash = _intersects(lb, db)
        if clash:
            ok = False
        checks.append(f"  [{'OK' if not clash else 'FAIL'}] "
                      f"{la_name} vs {da_name}: clash={clash}")

# 9. per-edge traffic audit
checks.append("\nPer-edge traffic audit:")
checks.append(f"  {'edge':<14}{'traffic':>10}{'n_used':>10}"
              f"{'uniq_on':>10}{'pool_sum':>12}")
for i, j in EDGES:
    checks.append(f"  {i+'->'+j:<14}"
                  f"{edge_traffic_rate[(i, j)]:>10.4f}"
                  f"{n_per_edge_used[(i, j)]:>10}"
                  f"{edge_n_clones_on_edge[(i, j)]:>10}"
                  f"{edge_n_clones_in_source_pools_summed[(i, j)]:>12}")

# 10. Spearman across edges: n_branches_used vs traffic_rate
n_used_vec = np.array([n_per_edge_used[edge] for edge in EDGES], dtype=float)
rate_vec   = np.array([edge_traffic_rate[edge] for edge in EDGES])
if np.std(n_used_vec) > 0 and np.std(rate_vec) > 0:
    rho, p_rho = spearmanr(n_used_vec, rate_vec)
    checks.append(
        f"\nSpearman(n_branches_used, traffic_rate) "
        f"= rho={rho:.3f} (p={p_rho:.3g})"
    )
    if rho < 0.5:
        checks.append(
            "  WARNING: rho < 0.5 — depth bias on the n_branches view "
            "is non-trivial; flag this in the figure caption."
        )
else:
    checks.append("\nSpearman skipped (zero variance in one vector).")

# 11. Violin fill colors must equal right-graph arrow colors edge-wise.
checks.append("\nViolin fill ⟷ expansion arrow color match (RGBA, tol 1e-6):")
for edge in EDGES:
    v = violin_facecolors[edge]
    a = expand_arrow_colors[edge]
    diff = max(abs(vv - aa) for vv, aa in zip(v, a))
    match = diff < 1e-6
    if not match:
        ok = False
    checks.append(f"  [{'OK' if match else 'FAIL'}] "
                  f"{edge[0]}->{edge[1]}: max|Δrgba|={diff:.2e}")

checks.insert(0, f"OVERALL: {'PASS' if ok else 'FAIL'}\n")

(OUT_DIR / "render_check.txt").write_text("\n".join(checks))
print("\n".join(checks[:20]))
print(f"... full report: {OUT_DIR / 'render_check.txt'}")

# %%
print("Done.")
