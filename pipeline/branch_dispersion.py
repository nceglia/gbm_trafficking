# %%
"""Stayer-vs-mover phenotype enrichment per source tissue (ask 2c).

For each (clone c, src_tissue i, t) with at least MIN_N_SRC cells, we
classify the clone's continuation at t+1 as STAYER_ONLY (still in i,
absent from any j≠i), MOVER_ONLY (absent from i, present in some j≠i),
or BOTH. The figure plots, per source tissue, the cell-weighted Δ in
phenotype fraction between MOVER_ONLY and STAYER_ONLY — bars right of
zero = enriched in movers, left = enriched in stayers — with a
clone-level mixed-effects logistic regression backing each phenotype
(BH-FDR across the (tissue × phenotype) grid).

Outputs (results/branch_dispersion/):
  stayer_vs_mover.{png,pdf}
  stayer_vs_mover_table.csv     long form, all three categories
  stayer_vs_mover_stats.csv     per (tissue × phenotype) test row
  render_check.txt
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import (
    TCELL_PHENOTYPE_COLORS,
    TCELL_PHENOTYPE_LABELS,
    TCELL_PHENOTYPE_ORDER,
    TISSUE_COLORS,
    TISSUE_LABELS,
    TISSUE_ORDER,
)

# %%
# ---- Global config (single source of truth) ----
MIN_N_SRC = 3
TRANSITIONS = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("5", "6")]
NEXT_TP = {t1: t2 for t1, t2 in TRANSITIONS}
CATEGORIES = ("STAYER_ONLY", "MOVER_ONLY", "BOTH")
FDR_THRESHOLD = 0.10
MIN_DELTA_FOR_STAR = 0.05
MIN_PER_CATEGORY_WARN = 30
ROW_GAP = 0.06
ANNOT_FS = 8
LABEL_FS = 11
TICK_FS = 9
TITLE_FS = 13
DPI = 200

DATA_PATH = REPO_ROOT / "data" / "objects" / "GBM_TCR_POS_TCELLS.h5ad"
OUT_DIR = REPO_ROOT / "results" / "branch_dispersion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUES = list(TISSUE_ORDER)
PHENOTYPES = list(TCELL_PHENOTYPE_ORDER)
N_PHEN = len(PHENOTYPES)


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.15, linewidth=0.6)
    ax.tick_params(labelsize=TICK_FS)


def _remap_pheno(p):
    if p in TCELL_PHENOTYPE_COLORS:
        return p
    if p in ("CD4_Th1_polarized", "CD4_Th2_polarized"):
        return "CD4_Th"
    return p


# %%
# ---- Load ----
print("Loading adata...")
adata = sc.read(str(DATA_PATH))
adata.obs["timepoint"] = adata.obs["timepoint"].astype(str)
obs = adata.obs[["trb", "tissue", "timepoint", "phenotype", "patient"]].copy()
obs["phenotype"] = obs["phenotype"].astype(str).map(_remap_pheno)
print(f"  {len(obs)} cells, {obs['phenotype'].nunique()} phenotypes, "
      f"{obs['patient'].nunique()} patients")

# %%
# ---- Per-(patient, clone, tissue, timepoint) phenotype counts ----
ph = (obs.groupby(["patient", "trb", "tissue", "timepoint", "phenotype"],
                   observed=True)
        .size().unstack("phenotype", fill_value=0))
for p in PHENOTYPES:
    if p not in ph.columns:
        ph[p] = 0
ph = ph[PHENOTYPES]
ph["_n"] = ph.sum(axis=1)
ph = ph.reset_index()

# Clone presence index: (patient, trb, tissue, timepoint) keys with ≥1 cell.
present_set = set(
    obs.groupby(["patient", "trb", "tissue", "timepoint"], observed=True)
       .indices.keys()
)

# %%
# ---- Build per-(clone, src_tissue, t) entries with category labels ----
print("Categorizing (clone, src_tissue, t) entries...")
records = []
pheno_cols = PHENOTYPES
for _, row in ph.iterrows():
    n_src = int(row["_n"])
    if n_src < MIN_N_SRC:
        continue
    t_src = str(row["timepoint"])
    next_tp = NEXT_TP.get(t_src)
    if next_tp is None:
        continue
    patient = row["patient"]
    trb = row["trb"]
    src_tissue = row["tissue"]

    in_src_next = (patient, trb, src_tissue, next_tp) in present_set
    in_other_next = any(
        (patient, trb, j, next_tp) in present_set
        for j in TISSUES if j != src_tissue
    )
    if not (in_src_next or in_other_next):
        continue  # NEITHER — clone disappears entirely

    if in_src_next and not in_other_next:
        cat = "STAYER_ONLY"
    elif in_other_next and not in_src_next:
        cat = "MOVER_ONLY"
    else:
        cat = "BOTH"

    frac = row[pheno_cols].to_numpy(dtype=float) / n_src
    rec = {
        "patient": patient, "trb": trb,
        "src_tissue": src_tissue, "t_src": t_src,
        "n_src": n_src, "category": cat,
    }
    for k, p in enumerate(pheno_cols):
        rec[p] = float(frac[k])
    records.append(rec)
entries = pd.DataFrame(records)
print(f"  {len(entries)} entries")
print(f"  by category: {entries['category'].value_counts().to_dict()}")

# %%
# ---- Per-(src_tissue, category) cell- and clone-weighted means ----
table_rows = []
for tissue in TISSUES:
    for cat in CATEGORIES:
        sub = entries[(entries["src_tissue"] == tissue)
                       & (entries["category"] == cat)]
        if sub.empty:
            cell_w = np.zeros(N_PHEN)
            clone_w = np.zeros(N_PHEN)
            n_cells_total = 0
        else:
            w = sub["n_src"].to_numpy(dtype=float)
            frac_mat = sub[PHENOTYPES].to_numpy(dtype=float)
            cell_w = (frac_mat * w[:, None]).sum(axis=0) / w.sum()
            clone_w = frac_mat.mean(axis=0)
            n_cells_total = int(w.sum())
        for k, p in enumerate(PHENOTYPES):
            table_rows.append({
                "src_tissue": tissue,
                "phenotype": p,
                "category": cat,
                "mean_frac_cell_weighted": float(cell_w[k]),
                "mean_frac_clone_weighted": float(clone_w[k]),
                "n_clones": int(len(sub)),
                "n_cells_total": n_cells_total,
            })
table_df = pd.DataFrame(table_rows)
table_df.to_csv(OUT_DIR / "stayer_vs_mover_table.csv", index=False)

# Per-tissue category counts for figure annotations.
n_mov = {t: int(((entries["src_tissue"] == t)
                  & (entries["category"] == "MOVER_ONLY")).sum())
         for t in TISSUES}
n_sta = {t: int(((entries["src_tissue"] == t)
                  & (entries["category"] == "STAYER_ONLY")).sum())
         for t in TISSUES}
n_both = {t: int(((entries["src_tissue"] == t)
                   & (entries["category"] == "BOTH")).sum())
          for t in TISSUES}

# %%
# ---- Per-(src_tissue, phenotype) test ----
# Primary: BinomialBayesMixedGLM with patient as a random intercept.
# Fallback: statsmodels.Logit with cluster-robust SE clustered by patient.
print("Fitting per-(src_tissue, phenotype) models...")
stats_rows = []
for tissue in TISSUES:
    sub_t = entries[(entries["src_tissue"] == tissue)
                     & (entries["category"].isin(
                         ["STAYER_ONLY", "MOVER_ONLY"]))].copy()
    if sub_t.empty:
        for p in PHENOTYPES:
            stats_rows.append({
                "src_tissue": tissue, "phenotype": p,
                "beta": np.nan, "beta_se": np.nan, "beta_p": np.nan,
                "n_movers": 0, "n_stayers": 0,
                "convergence_status": "no_data",
            })
        continue
    sub_t["is_mover"] = (sub_t["category"] == "MOVER_ONLY").astype(int)
    n_mov_t = int((sub_t["is_mover"] == 1).sum())
    n_sta_t = int((sub_t["is_mover"] == 0).sum())
    patient_codes, _ = pd.factorize(sub_t["patient"].astype(str))
    n_pat = int(patient_codes.max()) + 1 if len(patient_codes) else 0
    exog_vc = np.zeros((len(sub_t), n_pat))
    for ix, pc in enumerate(patient_codes):
        exog_vc[ix, pc] = 1.0
    ident = np.zeros(n_pat, dtype=int)

    for p in PHENOTYPES:
        x = sub_t[p].to_numpy(dtype=float)
        if x.std() == 0 or n_mov_t < 3 or n_sta_t < 3:
            stats_rows.append({
                "src_tissue": tissue, "phenotype": p,
                "beta": np.nan, "beta_se": np.nan, "beta_p": np.nan,
                "n_movers": n_mov_t, "n_stayers": n_sta_t,
                "convergence_status": "insufficient_variance_or_n",
            })
            continue
        X = np.column_stack([np.ones_like(x), x])
        y = sub_t["is_mover"].to_numpy(dtype=float)
        beta = beta_se = pval = np.nan
        conv = "failed"
        try:
            md = sm.BinomialBayesMixedGLM(
                y, X, exog_vc=exog_vc, ident=ident,
            )
            res = md.fit_vb()
            beta = float(res.fe_mean[1])
            beta_se = float(res.fe_sd[1])
            if beta_se > 0:
                z = beta / beta_se
                pval = float(2.0 * (1.0 - sm.distributions.norm.cdf(abs(z))))
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
            "n_movers": n_mov_t, "n_stayers": n_sta_t,
            "convergence_status": conv,
        })

stats_df = pd.DataFrame(stats_rows)
valid = stats_df["beta_p"].notna()
qvals = np.full(len(stats_df), np.nan)
if int(valid.sum()) > 0:
    _, q_flat, _, _ = multipletests(stats_df.loc[valid, "beta_p"].values,
                                    method="fdr_bh")
    qvals[valid.to_numpy()] = q_flat
stats_df["beta_q"] = qvals
stats_df = stats_df[[
    "src_tissue", "phenotype", "beta", "beta_se", "beta_p", "beta_q",
    "n_movers", "n_stayers", "convergence_status",
]]
stats_df.to_csv(OUT_DIR / "stayer_vs_mover_stats.csv", index=False)

q_map = stats_df.set_index(["src_tissue", "phenotype"])["beta_q"].to_dict()

# %%
# ---- Per-(tissue, phenotype) cell- and clone-weighted Δ ----
def _delta(tissue, kind):
    m = table_df[(table_df["src_tissue"] == tissue)
                  & (table_df["category"] == "MOVER_ONLY")].set_index(
        "phenotype")[kind]
    s = table_df[(table_df["src_tissue"] == tissue)
                  & (table_df["category"] == "STAYER_ONLY")].set_index(
        "phenotype")[kind]
    return np.array([float(m.loc[p] - s.loc[p]) for p in PHENOTYPES])


delta_cell = {t: _delta(t, "mean_frac_cell_weighted") for t in TISSUES}
delta_clone = {t: _delta(t, "mean_frac_clone_weighted") for t in TISSUES}

# Shared symmetric xlim, rounded up to nearest 0.05.
_all_abs = np.concatenate([np.abs(delta_cell[t]) for t in TISSUES])
_max_abs = float(np.max(_all_abs)) if _all_abs.size else 0.05
xlim_val = max(np.ceil(_max_abs / 0.05) * 0.05, 0.05)

# %%
# ---- Figure ----
print("Plotting stayer_vs_mover...")
fig = plt.figure(figsize=(15, 6), dpi=DPI)
GS_LEFT, GS_RIGHT = 0.10, 0.97
GS_TOP, GS_BOTTOM = 0.86, 0.20

PANEL_W = (GS_RIGHT - GS_LEFT - 0.02 * 2) / 3.0
axes = []
for k, tissue in enumerate(TISSUES):
    left_k = GS_LEFT + k * (PANEL_W + 0.02)
    ax = fig.add_axes([left_k, GS_BOTTOM, PANEL_W, GS_TOP - GS_BOTTOM])
    axes.append(ax)

y_pos = np.arange(N_PHEN)
for k, tissue in enumerate(TISSUES):
    ax = axes[k]
    deltas = delta_cell[tissue]
    bars = ax.barh(
        y_pos, deltas,
        color=[TCELL_PHENOTYPE_COLORS[p] for p in PHENOTYPES],
        edgecolor="black", linewidth=0.6,
    )
    ax.axvline(0, color="gray", lw=0.6, linestyle="--", alpha=0.5)
    ax.set_xlim(-xlim_val, xlim_val)
    ax.set_ylim(N_PHEN - 0.5, -0.5)  # invert
    ax.set_title(TISSUE_LABELS[tissue], fontsize=TITLE_FS,
                 fontweight="bold", color=TISSUE_COLORS[tissue], pad=8)
    ax.set_xlabel("Δ phenotype fraction (mover − stayer)",
                  fontsize=LABEL_FS)
    if k == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([TCELL_PHENOTYPE_LABELS[p] for p in PHENOTYPES],
                           fontsize=ANNOT_FS)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
    _style_axis(ax)

    # Bar-tip annotations (per-tissue n_M / n_S — same for every bar in
    # the panel since category counts are per source tissue).
    nm = n_mov[tissue]
    ns = n_sta[tissue]
    annot_pad = xlim_val * 0.02
    for yi, p in enumerate(PHENOTYPES):
        delta = float(deltas[yi])
        tip_x = delta + (annot_pad if delta >= 0 else -annot_pad)
        ha = "left" if delta >= 0 else "right"
        ax.text(tip_x, yi, f"(n_M={nm}, n_S={ns})",
                ha=ha, va="center",
                fontsize=ANNOT_FS - 1, color="dimgray")
        q = q_map.get((tissue, p), np.nan)
        if (not np.isnan(q)) and q < FDR_THRESHOLD \
           and abs(delta) > MIN_DELTA_FOR_STAR:
            star_x = tip_x + (xlim_val * 0.18 if delta >= 0
                              else -xlim_val * 0.18)
            ax.text(star_x, yi, "**",
                    ha=ha, va="center",
                    fontsize=ANNOT_FS + 1, fontweight="bold",
                    color="black")

# Bold y-tick labels (on the leftmost panel) when ANY tissue has q < 0.10
# for that phenotype. The per-tissue "**" already records per-panel
# significance; the bold label flags "worth a second look anywhere."
for tick, p in zip(axes[0].get_yticklabels(), PHENOTYPES):
    any_sig = any(
        (not np.isnan(q_map.get((t, p), np.nan)))
        and q_map.get((t, p)) < FDR_THRESHOLD
        for t in TISSUES
    )
    if any_sig:
        tick.set_fontweight("bold")

# ---- Render-time overflow check (sanity #6): if any annotation text
# extends past the panel's xlim, extend xlim by 10% and re-render once.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()


def _any_text_overflows():
    for ax in axes:
        x0_data, x1_data = ax.get_xlim()
        for t in ax.texts:
            bb = t.get_window_extent(renderer=renderer)
            data_bb = ax.transData.inverted().transform_bbox(bb)
            if data_bb.x0 < x0_data or data_bb.x1 > x1_data:
                return True
    return False


extended = False
if _any_text_overflows():
    xlim_val *= 1.10
    for ax in axes:
        ax.set_xlim(-xlim_val, xlim_val)
    fig.canvas.draw()
    extended = True

# ---- Footer caption ----
fig.suptitle("Phenotype enrichment in movers vs stayers",
             fontsize=TITLE_FS + 1, fontweight="bold", y=0.97)
fig.text(
    0.5, 0.04,
    "Bars: cell-weighted mean phenotype fraction (MOVER_ONLY) minus "
    "mean phenotype fraction (STAYER_ONLY). Right = enriched in "
    "movers; left = enriched in stayers. Bold labels: clone-level "
    "mixed-effects logistic regression q < 0.10 (BH-FDR). n_M, n_S = "
    "(clone, timepoint) entries per category. Clones present in both "
    "i and j≠i at t+1 (BOTH) excluded; see stayer_vs_mover_table.csv.",
    ha="center", va="bottom", fontsize=ANNOT_FS,
    color="dimgray", style="italic", wrap=True,
)

png_path = OUT_DIR / "stayer_vs_mover.png"
pdf_path = OUT_DIR / "stayer_vs_mover.pdf"
fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

# %%
# ---- Sanity checks ----
print("Running sanity checks...")
checks = []
ok = True

# 1. mean_dist sums to 1.0 per (src_tissue, category) for non-empty groups
checks.append("Cell-weighted mean phenotype distributions sum to 1.0:")
for tissue in TISSUES:
    for cat in CATEGORIES:
        sub = table_df[(table_df["src_tissue"] == tissue)
                        & (table_df["category"] == cat)]
        s = float(sub["mean_frac_cell_weighted"].sum())
        n_clones = int(sub["n_clones"].iat[0]) if len(sub) else 0
        if n_clones == 0:
            checks.append(f"  [skip] {tissue}/{cat}: 0 clones")
            continue
        match = abs(s - 1.0) < 1e-6
        if not match:
            ok = False
        checks.append(f"  [{'OK' if match else 'FAIL'}] "
                      f"{tissue}/{cat}: sum={s:.6f}")

# 2. Per (src_tissue, category): n_clones, n_cells_total, median clone size
checks.append("\nPer (src_tissue, category) summaries:")
checks.append(f"  {'tissue':<6}{'category':<14}{'n_clones':>10}"
              f"{'n_cells':>10}{'med_size':>10}")
for tissue in TISSUES:
    for cat in CATEGORIES:
        sub = entries[(entries["src_tissue"] == tissue)
                       & (entries["category"] == cat)]
        nc = int(len(sub))
        ncells = int(sub["n_src"].sum()) if nc else 0
        med = float(sub["n_src"].median()) if nc else 0.0
        checks.append(f"  {tissue:<6}{cat:<14}{nc:>10}{ncells:>10}"
                      f"{med:>10.1f}")

# 3. Each (clone, src_tissue, t_src) entry has exactly one category.
key_counts = entries.groupby(["patient", "trb", "src_tissue", "t_src"]).size()
dup = key_counts[key_counts > 1]
unique_key_ok = dup.empty
if not unique_key_ok:
    ok = False
checks.append(f"\n[{'OK' if unique_key_ok else 'FAIL'}] "
              f"unique (patient, trb, src_tissue, t_src): "
              f"n_dups={len(dup)}")

# 4. Cell-weighted vs clone-weighted Δ correlation per src_tissue
checks.append("\nCell-weighted vs clone-weighted Δ correlation:")
for tissue in TISSUES:
    a = delta_cell[tissue]
    b = delta_clone[tissue]
    if np.std(a) == 0 or np.std(b) == 0:
        checks.append(f"  [skip] {tissue}: zero variance")
        continue
    r, _ = pearsonr(a, b)
    flag = "OK" if r > 0.8 else "WARN"
    if r <= 0.8:
        checks.append(f"  [{flag}] {tissue}: r={r:.3f} "
                      f"(big clones may be driving the cell-weighted view)")
    else:
        checks.append(f"  [{flag}] {tissue}: r={r:.3f}")

# 5. Category counts per src_tissue; warn if < MIN_PER_CATEGORY_WARN
checks.append("\nPer-source-tissue category counts:")
checks.append(f"  {'tissue':<6}{'STAYER_ONLY':>14}{'MOVER_ONLY':>14}"
              f"{'BOTH':>10}")
for tissue in TISSUES:
    s_o = n_sta[tissue]
    m_o = n_mov[tissue]
    b_o = n_both[tissue]
    checks.append(f"  {tissue:<6}{s_o:>14}{m_o:>14}{b_o:>10}")
    for label, n in [("STAYER_ONLY", s_o), ("MOVER_ONLY", m_o),
                     ("BOTH", b_o)]:
        if n < MIN_PER_CATEGORY_WARN:
            checks.append(f"    WARN: {tissue}/{label} n={n} < "
                          f"{MIN_PER_CATEGORY_WARN} — test underpowered")

# 6. Bar-tip overflow handled at render-time; report outcome here.
if extended:
    checks.append("\n[INFO] xlim extended by 10% after detecting bar-tip "
                  f"text overflow. Final xlim ±{xlim_val:.3f}.")
else:
    checks.append(f"\n[OK] bar-tip text fits within xlim ±{xlim_val:.3f}.")

# Stats: convergence summary
checks.append("\nConvergence status counts:")
for status, n in stats_df["convergence_status"].value_counts().items():
    checks.append(f"  {status:<32}{int(n):>6}")

checks.insert(0, f"OVERALL: {'PASS' if ok else 'FAIL'}\n")

(OUT_DIR / "render_check.txt").write_text("\n".join(checks))
print("\n".join(checks[:20]))
print(f"... full report: {OUT_DIR / 'render_check.txt'}")

# %%
print("Done.")
