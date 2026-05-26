# %%
"""Build deploy/bundle/signaling.html — L-R signaling explorer.

Gzip+base64 CSVs inlined, Plotly + pako + papaparse from CDN, no server.
Intermediate CSVs remain under results/signaling_explorer/ on the build machine.
"""
import base64
import gzip
import io
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# %%
# ---- Config ----
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.style import TISSUE_COLORS
from viewers.build import landing
from viewers.paths import (
    CROSS_LINEAGE_CORR_DIR,
    LIANA_SIGNALING_DIR,
    SIGNALING_EXPLORER_DIR,
    SIGNALING_HTML,
    ensure_bundle,
)

LIANA_PATH = LIANA_SIGNALING_DIR / "liana_per_timepoint.csv"
PATHWAY_CORRS_PATH = CROSS_LINEAGE_CORR_DIR / "pathway_correlations.csv"
DATA_OUT_DIR = SIGNALING_EXPLORER_DIR
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_HTML = SIGNALING_HTML
ensure_bundle()

LIANA_PVAL = 0.05
TIMEPOINTS_SRC = [1, 2, 3, 4, 5]
TIMEPOINTS_DST = [2, 3, 4, 5, 6]
TISSUES = ["PBMC", "CSF", "TP"]
SAME_TISSUE_BR = [(t, t) for t in TISSUES]
CROSS_TISSUE_BR = [(i, j) for i in TISSUES for j in TISSUES if i != j]
BRANCHES = SAME_TISSUE_BR + CROSS_TISSUE_BR    # 9 total
BRANCH_STRS = [f"{i}->{j}" for i, j in BRANCHES]

ROW_ABORT_THRESHOLD = 5_000_000
HTML_SIZE_TARGET_MB = 25.0

t_start = time.time()

# %%
# ---- Load + sanity ----
print(f"Loading {LIANA_PATH}...")
_t0 = time.time()
liana = pd.read_csv(LIANA_PATH)
print(f"  {len(liana):,} rows, cols: {list(liana.columns)}")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

REQUIRED = ["ligand_complex", "receptor_complex", "lr_means",
            "cellphone_pvals", "tissue", "timepoint", "source", "target",
            "ligand_pathways", "receptor_pathways"]
for col in REQUIRED:
    if col not in liana.columns:
        raise KeyError(f"Missing required column: {col!r}")

liana["timepoint"] = pd.to_numeric(liana["timepoint"], errors="coerce")
liana = liana[liana["timepoint"].notna()].copy()
liana["timepoint"] = liana["timepoint"].astype(int)

# %%
# ---- Step 1: filter to significant L-R pairs ----
print("\nStep 1: filter to L-R pairs with min p < {:.2f} across all "
      "conditions...".format(LIANA_PVAL))
_t0 = time.time()
lr_min_p = liana.groupby(
    ["ligand_complex", "receptor_complex"])["cellphone_pvals"].min()
sig_lr_mi = lr_min_p[lr_min_p < LIANA_PVAL].index
keep_df = pd.DataFrame(list(sig_lr_mi),
                       columns=["ligand_complex", "receptor_complex"])
liana_sig = liana.merge(keep_df, on=["ligand_complex", "receptor_complex"])
n_lr_sig = len(sig_lr_mi)
print(f"  Kept {n_lr_sig} L-R pairs with min p<{LIANA_PVAL} across all "
      f"conditions ({len(liana_sig):,} L-R rows remain)")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

# %%
# ---- Pre-flight estimates ----
print("\nPre-flight estimates:")
print(f"  Filtered rows after step 1: {len(liana_sig):,}")
print(f"  Unique L-R pairs: {n_lr_sig:,}")
print(f"  Branches expected: {len(BRANCHES)}")
est_entry = n_lr_sig * len(BRANCHES)
est_temporal = n_lr_sig * len(TISSUES) * 6
# Phenotype detail: at most one row per (L-R, tissue, source, target) per
# side, then 9 branch fan-out (same tissue serves multiple branches).
n_phen_pairs = liana_sig[["source", "target"]].drop_duplicates().shape[0]
est_detail_upper = int(len(liana_sig) * 2)    # very loose upper bound
print(f"  Unique (source, target) pheno pairs: {n_phen_pairs}")
print(f"  Entry matrix cells expected: {est_entry:,}")
print(f"  Temporal trajectory rows expected: {est_temporal:,}")
print(f"  Phenotype detail rows (upper bound): {est_detail_upper:,}")

aborts = []
if est_entry > ROW_ABORT_THRESHOLD:
    aborts.append(f"entry matrix ({est_entry:,} rows)")
if est_temporal > ROW_ABORT_THRESHOLD:
    aborts.append(f"temporal trajectory ({est_temporal:,} rows)")
if est_detail_upper > ROW_ABORT_THRESHOLD:
    aborts.append(f"phenotype detail upper-bound ({est_detail_upper:,} rows)")
if aborts:
    print("\nABORTING: " + "; ".join(aborts))
    print("Tighten the upstream filter (lower LIANA_PVAL or pre-filter "
          "the liana_per_timepoint.csv input).")
    sys.exit(1)

# %%
# ---- Step 2: entry matrix ----
print("\nStep 2: entry matrix (L-R x branch)...")
_t0 = time.time()

pos_means = liana_sig.loc[liana_sig["lr_means"] > 0, "lr_means"]
pseudo = float(0.5 * pos_means.min()) if len(pos_means) else 0.01
print(f"  pseudocount = {pseudo:.6g}")

print("  pre-aggregating per (tissue, timepoint, L-R)...")
per_ttlr = (liana_sig
            .groupby(["tissue", "timepoint",
                       "ligand_complex", "receptor_complex"],
                      observed=True)["lr_means"]
            .mean().reset_index())

entry_blocks = []
for i, j in tqdm(BRANCHES, desc="entry-matrix"):
    src = per_ttlr[(per_ttlr["tissue"] == i)
                    & (per_ttlr["timepoint"].isin(TIMEPOINTS_SRC))]
    dst = per_ttlr[(per_ttlr["tissue"] == j)
                    & (per_ttlr["timepoint"].isin(TIMEPOINTS_DST))]
    src_mean = src.groupby(
        ["ligand_complex", "receptor_complex"])["lr_means"].mean()
    dst_mean = dst.groupby(
        ["ligand_complex", "receptor_complex"])["lr_means"].mean()
    block = pd.DataFrame({"src_mean": src_mean, "dst_mean": dst_mean})
    block = block.reindex(sig_lr_mi).reset_index()
    block["branch"] = f"{i}->{j}"
    entry_blocks.append(block)

entry = pd.concat(entry_blocks, ignore_index=True)
entry["log2fc"] = np.log2(
    (entry["dst_mean"].fillna(0) + pseudo)
    / (entry["src_mean"].fillna(0) + pseudo)
)

# Drop L-R pairs whose 9 cells are all empty (NaN or zero on both sides).
src_any = entry["src_mean"].fillna(0).gt(0)
dst_any = entry["dst_mean"].fillna(0).gt(0)
entry["_any"] = src_any | dst_any
keep_mask = (entry.groupby(["ligand_complex", "receptor_complex"])["_any"]
                  .transform("any"))
entry = entry[keep_mask].drop(columns="_any")

entry = entry[["ligand_complex", "receptor_complex", "branch",
                "src_mean", "dst_mean", "log2fc"]]
# Round for compact CSV
for c in ("src_mean", "dst_mean", "log2fc"):
    entry[c] = entry[c].astype(float).round(4)
entry.to_csv(DATA_OUT_DIR / "entry_matrix.csv", index=False)

n_lr_final = (entry[["ligand_complex", "receptor_complex"]]
              .drop_duplicates().shape[0])
print(f"  entry_matrix: {len(entry):,} rows, {n_lr_final} L-R pairs "
      f"(dropped {n_lr_sig - n_lr_final} fully-empty)")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

keep_final = (entry[["ligand_complex", "receptor_complex"]]
              .drop_duplicates())
liana_final = liana_sig.merge(keep_final,
                                on=["ligand_complex", "receptor_complex"])

# %%
# ---- Step 3: phenotype detail ----
print("\nStep 3: phenotype detail (L-R x branch x side x source x target)...")
_t0 = time.time()

src_view = liana_final[liana_final["timepoint"].isin(TIMEPOINTS_SRC)]
dst_view = liana_final[liana_final["timepoint"].isin(TIMEPOINTS_DST)]
src_agg = (src_view.groupby(
    ["tissue", "ligand_complex", "receptor_complex", "source", "target"],
    observed=True).agg(
        lr_means=("lr_means", "mean"),
        cellphone_pvals=("cellphone_pvals", "min"),
    ).reset_index())
dst_agg = (dst_view.groupby(
    ["tissue", "ligand_complex", "receptor_complex", "source", "target"],
    observed=True).agg(
        lr_means=("lr_means", "mean"),
        cellphone_pvals=("cellphone_pvals", "min"),
    ).reset_index())

detail_blocks = []
for i, j in tqdm(BRANCHES, desc="detail"):
    branch_str = f"{i}->{j}"
    s = src_agg[src_agg["tissue"] == i].copy()
    s["branch"] = branch_str
    s["side"] = "src"
    detail_blocks.append(s[["ligand_complex", "receptor_complex",
                             "branch", "side", "source", "target",
                             "lr_means", "cellphone_pvals"]])
    d = dst_agg[dst_agg["tissue"] == j].copy()
    d["branch"] = branch_str
    d["side"] = "dst"
    detail_blocks.append(d[["ligand_complex", "receptor_complex",
                             "branch", "side", "source", "target",
                             "lr_means", "cellphone_pvals"]])

detail = pd.concat(detail_blocks, ignore_index=True)
if len(detail) > ROW_ABORT_THRESHOLD:
    print(f"\nABORTING: phenotype detail has {len(detail):,} rows > "
          f"{ROW_ABORT_THRESHOLD:,}")
    sys.exit(1)
for c in ("lr_means", "cellphone_pvals"):
    detail[c] = detail[c].astype(float).round(4)
detail.to_csv(DATA_OUT_DIR / "phenotype_detail.csv", index=False)
print(f"  phenotype_detail: {len(detail):,} rows")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

# %%
# ---- Step 4: temporal trajectory ----
print("\nStep 4: temporal trajectory (L-R x tissue x timepoint)...")
_t0 = time.time()
temporal = (liana_final.groupby(
    ["ligand_complex", "receptor_complex", "tissue", "timepoint"],
    observed=True).agg(
        lr_means=("lr_means", "mean"),
        cellphone_pvals=("cellphone_pvals", "min"),
    ).reset_index())
for c in ("lr_means", "cellphone_pvals"):
    temporal[c] = temporal[c].astype(float).round(4)
temporal.to_csv(DATA_OUT_DIR / "temporal_trajectory.csv", index=False)
print(f"  temporal_trajectory: {len(temporal):,} rows")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

# %%
# ---- Step 5: pathway cross-reference ----
print("\nStep 5: pathway cross-reference...")
_t0 = time.time()

if PATHWAY_CORRS_PATH.exists():
    pcorrs = pd.read_csv(PATHWAY_CORRS_PATH)
    print(f"  pathway_correlations loaded: {len(pcorrs):,} rows")
    candidates = pcorrs[(pcorrs["q_bh"] < 0.10)
                         & (pcorrs["rho"].abs() > 0.4)].copy()
    cand_pairs = sorted(set(zip(candidates["t_pathway"].astype(str),
                                  candidates["m_pathway"].astype(str))))
    print(f"  Candidate (t_pathway, m_pathway) pairs: {len(cand_pairs)}")
else:
    print(f"  WARNING: {PATHWAY_CORRS_PATH} missing; xref empty")
    cand_pairs = []

# One row per L-R with ligand/receptor pathway strings (semicolon joined).
pw_lookup = (liana_final
             .groupby(["ligand_complex", "receptor_complex"])
             .agg(ligand_pathways=("ligand_pathways", "first"),
                  receptor_pathways=("receptor_pathways", "first"))
             .reset_index())
pw_lookup["ligand_pathways"] = pw_lookup["ligand_pathways"].fillna("")
pw_lookup["receptor_pathways"] = pw_lookup["receptor_pathways"].fillna("")
pw_lookup.to_csv(DATA_OUT_DIR / "pathway_lookup.csv", index=False)
print(f"  pathway_lookup: {len(pw_lookup):,} rows")

xref_rows = []
if cand_pairs:
    for _, row in tqdm(pw_lookup.iterrows(),
                        total=len(pw_lookup), desc="xref"):
        lp = {p.strip() for p in str(row["ligand_pathways"]).split(";")
              if p.strip()}
        rp = {p.strip() for p in str(row["receptor_pathways"]).split(";")
              if p.strip()}
        if not lp and not rp:
            continue
        for t_pw, m_pw in cand_pairs:
            if t_pw in lp and m_pw in rp:
                xref_rows.append({
                    "ligand_complex": row["ligand_complex"],
                    "receptor_complex": row["receptor_complex"],
                    "t_pathway": t_pw, "m_pathway": m_pw,
                    "direction": "ligand_in_t_pathway",
                })
            if t_pw in rp and m_pw in lp:
                xref_rows.append({
                    "ligand_complex": row["ligand_complex"],
                    "receptor_complex": row["receptor_complex"],
                    "t_pathway": t_pw, "m_pathway": m_pw,
                    "direction": "ligand_in_m_pathway",
                })

xref = pd.DataFrame(xref_rows) if xref_rows else pd.DataFrame(
    columns=["ligand_complex", "receptor_complex",
              "t_pathway", "m_pathway", "direction"])
xref.to_csv(DATA_OUT_DIR / "pathway_xref.csv", index=False)
print(f"  pathway_xref: {len(xref):,} rows")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

# %%
# ---- Encode all CSVs as gzip+base64 ----
print("\nEncoding tables (gzip + base64)...")
_t0 = time.time()


def _encode(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    gz = gzip.compress(buf.getvalue(), compresslevel=9)
    return base64.b64encode(gz).decode("ascii")


entry_b64 = _encode(entry)
detail_b64 = _encode(detail)
temporal_b64 = _encode(temporal)
xref_b64 = _encode(xref)
pw_lookup_b64 = _encode(pw_lookup)

for name, b in [("entry", entry_b64), ("detail", detail_b64),
                 ("temporal", temporal_b64), ("xref", xref_b64),
                 ("pw_lookup", pw_lookup_b64)]:
    print(f"  {name}: {len(b):>12,} chars (~{len(b)/1024/1024:.2f} MB)")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

# %%
# ---- HTML template ----
# JS/CSS literal braces are fine inside a raw string; we substitute via
# .replace() to avoid format/f-string escaping headaches.
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GBM T cell - Myeloid signaling explorer</title>
<style>
:root {
  --bg: #ffffff;
  --section-bg: #f9fafb;
  --section-border: #e5e7eb;
  --text: #111827;
  --muted: #6b7280;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter,
               Roboto, sans-serif;
  background: var(--bg); color: var(--text);
}
body { padding: 18px 24px 32px; }

header { text-align: center; padding: 4px 0 18px; }
header h1 {
  font-size: 24px; font-weight: 700; margin: 0 0 4px; color: var(--text);
}
header .subtitle {
  font-size: 14px; color: var(--muted); position: relative;
}
header .info { font-size: 12px; color: var(--muted); margin-top: 8px; }

.help-icon {
  display: inline-block; width: 18px; height: 18px; line-height: 18px;
  border-radius: 50%; background: var(--muted); color: white;
  font-size: 11px; font-weight: 700; text-align: center;
  margin-left: 6px; cursor: help; position: relative;
}
.help-icon:hover .help-tooltip { display: block; }
.help-tooltip {
  display: none; position: absolute; top: 24px; left: 50%;
  transform: translateX(-50%); width: 520px; max-width: 90vw;
  background: #1f2937; color: white; padding: 12px 14px;
  border-radius: 6px; font-size: 12px; line-height: 1.5;
  font-weight: 400; text-align: left; z-index: 1000;
  box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}

.section {
  background: var(--section-bg); border: 1px solid var(--section-border);
  border-radius: 8px; padding: 16px; margin-bottom: 16px;
}
.section-title {
  text-align: center; font-size: 16px; font-weight: 600;
  margin: 0 0 12px; color: var(--text);
}

.controls {
  display: flex; gap: 12px; padding: 4px 0; font-size: 12px;
  align-items: center; flex-wrap: wrap;
}
.controls input, .controls select {
  padding: 5px 8px; font-size: 12px;
  border: 1px solid var(--section-border); border-radius: 4px;
  background: white;
}
.controls label { font-weight: 500; }
.controls input { min-width: 280px; }

.segmented {
  display: inline-flex; border: 1px solid var(--section-border);
  border-radius: 4px; overflow: hidden; background: white;
}
.segmented button {
  padding: 5px 10px; font-size: 12px; font-weight: 500;
  background: white; color: var(--text);
  border: none; border-right: 1px solid var(--section-border);
  cursor: pointer; line-height: 1.2;
}
.segmented button:last-child { border-right: none; }
.segmented button.active { background: #1e3a8a; color: white; }
.segmented button:hover:not(.active) { background: #f3f4f6; }

.pager {
  display: flex; gap: 12px; padding: 6px 0 10px; font-size: 12px;
  align-items: center; flex-wrap: wrap;
}
.pager button {
  padding: 5px 12px; font-size: 12px; cursor: pointer;
  background: white; border: 1px solid var(--section-border);
  border-radius: 4px;
}
.pager button:disabled { opacity: 0.4; cursor: not-allowed; }
.pager .info { color: var(--muted); margin-left: auto; }

.plot { width: 100%; }
/* Floor for min cell hit-target: 320 left + 90 right + 70 * 9 + gaps. */
.plot-a { min-height: 540px; min-width: 1050px; }
.plot-b { height: 500px; }
.plot-c { height: 340px; }

.bcd-title {
  text-align: center; font-size: 16px; font-weight: 600;
  margin: 0 0 2px;
}
.bcd-subtitle {
  text-align: center; font-size: 13px; color: var(--muted);
  margin-bottom: 8px;
}
.tissue-name { font-weight: 600; }

.c-title {
  text-align: center; font-size: 16px; font-weight: 600;
  margin: 0 0 8px;
}

.pwbox { margin-top: 12px; padding: 4px 4px; font-size: 12px; }
.pwbox-section-header {
  font-size: 15px; font-weight: 600; margin: 12px 0 6px;
  color: var(--text);
}
.pwbox-explanation {
  font-size: 12px; color: #4b5563; font-style: italic;
  margin-bottom: 8px; line-height: 1.5;
}
.pwbox-row { margin: 6px 0; line-height: 1.8; }
.pwbox-row .label {
  font-weight: 500; color: #374151; margin-right: 6px;
}
.pill {
  display: inline-block; padding: 4px 8px; margin: 2px 4px 2px 0;
  border-radius: 12px; font-size: 12px; font-weight: 500;
}
.pill-ligand { background: #e0e7ff; color: #1e3a8a; }
.pill-receptor { background: #fee2e2; color: #7f1d1d; }
.card {
  border: 1px solid #d1d5db; border-radius: 6px;
  padding: 10px 12px; margin: 6px 0; background: #ffffff;
}
.card-row { font-size: 13px; margin: 2px 0; color: var(--text); }
.card-explain {
  font-size: 11px; color: var(--muted); margin-top: 4px;
  line-height: 1.45;
}

.placeholder {
  text-align: center; padding: 32px; color: var(--muted);
  font-size: 13px; font-style: italic;
}
</style>
</head>
<body>

<header>
  <h1>GBM T cell - Myeloid signaling explorer</h1>
  <div class="subtitle">
    Cross-tissue ligand-receptor differential analysis<span class="help-icon">?<span class="help-tooltip">Each cell of the matrix below is one ligand-receptor pair at one tissue transition. The transition compares signaling at timepoint t in the source tissue against timepoint t+1 in the destination tissue, pooled across patients and timepoints. Color shows the log2 fold-change: red means stronger at destination, blue means stronger at source. Click any cell for phenotype-level breakdown; click any row label for the temporal trajectory.</span></span>
  </div>
  <div class="info">
    <span id="hdr-stats">loading...</span> · 9 tissue transitions ·
    click any cell to explore · built __BUILD_DATE__
  </div>
</header>

<section class="section">
  <div class="section-title" id="section-a-title">
    Ligand-receptor differential across tissue transitions
  </div>
  <div class="controls">
    <div class="segmented mode-toggle" id="mode-toggle">
      <button data-mode="diff" class="active">Differential (log2 dst/src)</button>
      <button data-mode="src">Source intensity</button>
      <button data-mode="dst">Destination intensity</button>
    </div>
    <label>search:
      <input id="search" type="text"
        placeholder="ligand or receptor (substring, case-insensitive)"/>
    </label>
    <label>sort: <select id="sort"></select></label>
  </div>
  <div class="pager">
    <label>page size: <select id="pagesize">
      <option value="20" selected>20</option>
      <option value="30">30</option>
      <option value="50">50</option>
      <option value="100">100</option>
      <option value="200">200</option>
      <option value="all">all</option>
    </select></label>
    <button id="prev-page">‹ Prev</button>
    <button id="next-page">Next ›</button>
    <span class="info" id="page-info"></span>
  </div>
  <div id="plot-a" class="plot plot-a"></div>
</section>

<section class="section">
  <div class="section-title">Phenotype-level detail</div>
  <div id="b-title-block">
    <div class="placeholder">
      Click a cell in the matrix above to view phenotype-level detail.
    </div>
  </div>
  <div id="plot-b" class="plot plot-b" style="display: none;"></div>
</section>

<section class="section">
  <div class="section-title">Temporal trajectory and pathway context</div>
  <div id="c-title-block">
    <div class="placeholder">
      Click a cell in the matrix above to view temporal trajectory and
      pathway context.
    </div>
  </div>
  <div id="plot-c" class="plot plot-c" style="display: none;"></div>
  <div id="pwbox" class="pwbox" style="display: none;"></div>
</section>

<script type="application/json" id="entry-matrix-b64">__ENTRY_B64__</script>
<script type="application/json" id="phenotype-detail-b64">__DETAIL_B64__</script>
<script type="application/json" id="temporal-trajectory-b64">__TEMPORAL_B64__</script>
<script type="application/json" id="pathway-xref-b64">__XREF_B64__</script>
<script type="application/json" id="pathway-lookup-b64">__PW_LOOKUP_B64__</script>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js"></script>

<script>
const TISSUE_COLORS = {
  "PBMC": "__PBMC_COLOR__",
  "CSF":  "__CSF_COLOR__",
  "TP":   "__TP_COLOR__"
};
const BRANCHES = __BRANCHES_JSON__;
const BRANCH_STRS = BRANCHES.map(b => b[0] + "->" + b[1]);

const COLORSCALE_DIVERGENT = [
  [0,   "#1e3a8a"], [0.2, "#3b82f6"], [0.4, "#bfdbfe"],
  [0.5, "#f8fafc"], [0.6, "#fecaca"], [0.8, "#ef4444"],
  [1.0, "#7f1d1d"]
];
const COLORSCALE_SEQUENTIAL = [
  [0,    "#f8fafc"], [0.25, "#cbd5e1"], [0.5,  "#64748b"],
  [0.75, "#1e3a8a"], [1.0,  "#0c1e4d"]
];

const MIN_ROW_HEIGHT = 22;
const MIN_COL_WIDTH = 70;

function decode(id) {
  const b64 = document.getElementById(id).textContent.trim();
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const csv = pako.ungzip(bytes, { to: "string" });
  const parsed = Papa.parse(csv, {
    header: true, dynamicTyping: true, skipEmptyLines: true
  });
  return parsed.data;
}

console.time("decode");
const entryData    = decode("entry-matrix-b64");
const detailData   = decode("phenotype-detail-b64");
const temporalData = decode("temporal-trajectory-b64");
const xrefData     = decode("pathway-xref-b64");
const pwLookupData = decode("pathway-lookup-b64");
console.timeEnd("decode");

function lrKey(l, r) { return l + "|" + r; }

const entryByLR = new Map();
entryData.forEach(r => {
  if (!r.ligand_complex) return;
  const k = lrKey(r.ligand_complex, r.receptor_complex);
  let obj = entryByLR.get(k);
  if (!obj) { obj = {}; entryByLR.set(k, obj); }
  obj[r.branch] = r;
});

const detailByKey = new Map();
detailData.forEach(r => {
  if (!r.ligand_complex) return;
  const k = lrKey(r.ligand_complex, r.receptor_complex) + "|" + r.branch;
  let arr = detailByKey.get(k);
  if (!arr) { arr = []; detailByKey.set(k, arr); }
  arr.push(r);
});

const temporalByLR = new Map();
temporalData.forEach(r => {
  if (!r.ligand_complex) return;
  const k = lrKey(r.ligand_complex, r.receptor_complex);
  let arr = temporalByLR.get(k);
  if (!arr) { arr = []; temporalByLR.set(k, arr); }
  arr.push(r);
});

const xrefByLR = new Map();
xrefData.forEach(r => {
  if (!r.ligand_complex) return;
  const k = lrKey(r.ligand_complex, r.receptor_complex);
  let arr = xrefByLR.get(k);
  if (!arr) { arr = []; xrefByLR.set(k, arr); }
  arr.push(r);
});

const pwLookupByLR = new Map();
pwLookupData.forEach(r => {
  if (!r.ligand_complex) return;
  pwLookupByLR.set(lrKey(r.ligand_complex, r.receptor_complex), r);
});

const allLRs = Array.from(entryByLR.keys()).map(k => {
  const i = k.indexOf("|");
  return { key: k, ligand: k.slice(0, i), receptor: k.slice(i + 1) };
});

document.getElementById("hdr-stats").textContent =
  allLRs.length.toLocaleString() + " L-R pairs";

// Mode toggle state. "diff" = log2fc, "src" = src_mean, "dst" = dst_mean.
let MODE = "diff";

// Cached 95th-percentile vmax for intensity modes (stable color scale).
const CACHED_VMAX = { src: null, dst: null };
function computeP95(valueKey) {
  const vals = [];
  entryData.forEach(r => {
    const v = r[valueKey];
    if (v != null && !isNaN(v)) vals.push(v);
  });
  if (!vals.length) return 1;
  vals.sort((a, b) => a - b);
  const idx = Math.min(vals.length - 1,
                        Math.floor(0.95 * (vals.length - 1)));
  return vals[idx] || 1;
}
function getVmax(mode) {
  if (mode === "src") {
    if (CACHED_VMAX.src == null) CACHED_VMAX.src = computeP95("src_mean");
    return CACHED_VMAX.src;
  }
  if (mode === "dst") {
    if (CACHED_VMAX.dst == null) CACHED_VMAX.dst = computeP95("dst_mean");
    return CACHED_VMAX.dst;
  }
  return 5;
}
function modeValueKey(mode) {
  if (mode === "src") return "src_mean";
  if (mode === "dst") return "dst_mean";
  return "log2fc";
}

// Sort options per mode.
const DIFF_SORTS = [
  ["absmax",   "max |log2fc|"],
  ["posmax",   "max log2fc"],
  ["negmin",   "min log2fc"],
  ["ligand",   "ligand A-Z"],
  ["receptor", "receptor A-Z"],
];
const INTENSITY_SORTS = [
  ["intensitymax", "max intensity"],
  ["ligand",       "ligand A-Z"],
  ["receptor",     "receptor A-Z"],
];

function populateSortOptions() {
  const sortSel = document.getElementById("sort");
  const prev = sortSel.value;
  const opts = (MODE === "diff") ? DIFF_SORTS : INTENSITY_SORTS;
  sortSel.innerHTML = "";
  opts.forEach(([v, l]) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = l;
    sortSel.appendChild(o);
  });
  BRANCH_STRS.forEach(bs => {
    const o = document.createElement("option");
    o.value = "br:" + bs; o.textContent = "by " + bs;
    sortSel.appendChild(o);
  });
  const allValues = Array.from(sortSel.options).map(o => o.value);
  if (allValues.includes(prev)) {
    sortSel.value = prev;
  } else {
    sortSel.value = opts[0][0];
  }
}
populateSortOptions();

function updateSectionTitle() {
  const titles = {
    diff: "Ligand-receptor differential across tissue transitions",
    src:  "Ligand-receptor source intensity across tissue transitions",
    dst:  "Ligand-receptor destination intensity across tissue transitions"
  };
  document.getElementById("section-a-title").textContent = titles[MODE];
}

function lrAbsMax(key) {
  const obj = entryByLR.get(key);
  if (!obj) return 0;
  let m = 0;
  for (const bs of BRANCH_STRS) {
    const c = obj[bs];
    if (c && c.log2fc != null && Math.abs(c.log2fc) > m) {
      m = Math.abs(c.log2fc);
    }
  }
  return m;
}
function lrMax(key) {
  const obj = entryByLR.get(key);
  if (!obj) return -Infinity;
  let m = -Infinity;
  for (const bs of BRANCH_STRS) {
    const c = obj[bs];
    if (c && c.log2fc != null && c.log2fc > m) m = c.log2fc;
  }
  return m;
}
function lrMin(key) {
  const obj = entryByLR.get(key);
  if (!obj) return Infinity;
  let m = Infinity;
  for (const bs of BRANCH_STRS) {
    const c = obj[bs];
    if (c && c.log2fc != null && c.log2fc < m) m = c.log2fc;
  }
  return m;
}
function lrBranchAbs(key, bs) {
  const obj = entryByLR.get(key);
  if (!obj || !obj[bs]) return 0;
  const v = obj[bs].log2fc;
  return v == null ? 0 : Math.abs(v);
}
function lrIntensityMax(key, valueKey) {
  const obj = entryByLR.get(key);
  if (!obj) return 0;
  let m = 0;
  for (const bs of BRANCH_STRS) {
    const c = obj[bs];
    if (c && c[valueKey] != null && !isNaN(c[valueKey])
        && c[valueKey] > m) {
      m = c[valueKey];
    }
  }
  return m;
}
function lrBranchValue(key, bs, valueKey) {
  const obj = entryByLR.get(key);
  if (!obj || !obj[bs]) return 0;
  const v = obj[bs][valueKey];
  return v == null || isNaN(v) ? 0 : v;
}

// State
let filteredLRs = [];
let pagedLRs = [];
let currentPage = 1;
let selectedLR = null;       // {key, ligand, receptor}
let selectedBranch = null;   // [src, dst]
let panelABound = false;

function applyFilterSort() {
  const q = document.getElementById("search").value.trim().toLowerCase();
  const sortMode = document.getElementById("sort").value;
  let lrs = allLRs.slice();
  if (q) {
    lrs = lrs.filter(lr =>
      lr.ligand.toLowerCase().includes(q)
      || lr.receptor.toLowerCase().includes(q)
    );
  }
  const valueKey = modeValueKey(MODE);
  if (sortMode === "absmax") {
    lrs.sort((a, b) => lrAbsMax(b.key) - lrAbsMax(a.key));
  } else if (sortMode === "posmax") {
    lrs.sort((a, b) => lrMax(b.key) - lrMax(a.key));
  } else if (sortMode === "negmin") {
    lrs.sort((a, b) => lrMin(a.key) - lrMin(b.key));
  } else if (sortMode === "intensitymax") {
    lrs.sort((a, b) =>
      lrIntensityMax(b.key, valueKey) - lrIntensityMax(a.key, valueKey));
  } else if (sortMode === "ligand") {
    lrs.sort((a, b) => a.ligand.localeCompare(b.ligand));
  } else if (sortMode === "receptor") {
    lrs.sort((a, b) => a.receptor.localeCompare(b.receptor));
  } else if (sortMode.startsWith("br:")) {
    const bs = sortMode.slice(3);
    if (MODE === "diff") {
      lrs.sort((a, b) => lrBranchAbs(b.key, bs) - lrBranchAbs(a.key, bs));
    } else {
      lrs.sort((a, b) =>
        lrBranchValue(b.key, bs, valueKey)
        - lrBranchValue(a.key, bs, valueKey));
    }
  }
  filteredLRs = lrs;
}

function pageSlice() {
  const pageSize = document.getElementById("pagesize").value;
  if (pageSize === "all") {
    return { lrs: filteredLRs, totalPages: 1, ps: filteredLRs.length };
  }
  const ps = parseInt(pageSize, 10);
  const totalPages = Math.max(1, Math.ceil(filteredLRs.length / ps));
  if (currentPage > totalPages) currentPage = totalPages;
  if (currentPage < 1) currentPage = 1;
  const start = (currentPage - 1) * ps;
  return { lrs: filteredLRs.slice(start, start + ps), totalPages, ps };
}

function renderPanelA() {
  const sliced = pageSlice();
  pagedLRs = sliced.lrs;
  const totalPages = sliced.totalPages;

  const valueKey = modeValueKey(MODE);
  const z = pagedLRs.map(lr => BRANCH_STRS.map(bs => {
    const obj = entryByLR.get(lr.key);
    const c = obj && obj[bs];
    if (!c) return null;
    const v = c[valueKey];
    return v == null || isNaN(v) ? null : v;
  }));
  const customdata = pagedLRs.map(lr => BRANCH_STRS.map(bs => {
    const obj = entryByLR.get(lr.key);
    const c = obj && obj[bs];
    return [
      lr.ligand, lr.receptor, bs,
      c && c.src_mean != null ? c.src_mean.toFixed(3) : "—",
      c && c.dst_mean != null ? c.dst_mean.toFixed(3) : "—",
      c && c.log2fc != null   ? c.log2fc.toFixed(3)   : "—"
    ];
  }));
  // HTML-bolded row labels (Plotly renders <b> in tick text).
  const ylabels = pagedLRs.map(lr =>
    "<b>" + lr.ligand + "</b> → <b>" + lr.receptor + "</b>");

  // Mode-dependent palette / range / colorbar title; hover template
  // is mode-independent (always shows src_mean, dst_mean, log2fc).
  let colorscale, zmin, zmax, zmid, colorbarOpts;
  if (MODE === "diff") {
    colorscale = COLORSCALE_DIVERGENT;
    zmin = -5; zmid = 0; zmax = 5;
    colorbarOpts = {
      title: { text: "log2(dst / src) interaction strength",
                font: { size: 13 } },
      thickness: 14, len: 0.85,
      tickvals: [-5, -2.5, 0, 2.5, 5],
      ticktext: ["−5", "−2.5", "0", "+2.5", "+5"],
      tickfont: { size: 12 }
    };
  } else {
    const vmax = getVmax(MODE);
    colorscale = COLORSCALE_SEQUENTIAL;
    zmin = 0; zmid = null; zmax = vmax;
    colorbarOpts = {
      title: { text: MODE === "src"
                      ? "Source interaction strength"
                      : "Destination interaction strength",
                font: { size: 13 } },
      thickness: 14, len: 0.85,
      tickfont: { size: 12 }
    };
  }

  const trace = {
    z: z, x: BRANCH_STRS, y: ylabels,
    customdata: customdata,
    xgap: 2, ygap: 2,
    type: "heatmap",
    colorscale: colorscale,
    zmin: zmin, zmax: zmax,
    hovertemplate:
      "<b>%{customdata[0]} → %{customdata[1]}</b><br>"
      + "branch: %{customdata[2]}<br>"
      + "src_mean: %{customdata[3]}<br>"
      + "dst_mean: %{customdata[4]}<br>"
      + "log2fc: %{customdata[5]}<extra></extra>",
    colorbar: colorbarOpts
  };
  if (zmid !== null) trace.zmid = zmid;

  // Color-coded two-line column headers (Plotly tick text doesn't
  // support per-tick coloring; use annotations).
  const colAnnotations = BRANCHES.map((b, i) => ({
    text: "<b>" + b[0] + "</b><br>→ <b>" + b[1] + "</b>",
    xref: "x", yref: "paper",
    x: BRANCH_STRS[i], y: 1.005,
    yanchor: "bottom", xanchor: "center",
    showarrow: false,
    font: { size: 14, color: TISSUE_COLORS[b[0]] },
    align: "center"
  }));

  // Vertical divider after column 3 (between same-tissue and
  // cross-tissue groups).
  const shapes = [{
    type: "line",
    xref: "x", yref: "paper",
    x0: 2.5, x1: 2.5, y0: 0, y1: 1,
    line: { color: "#374151", width: 1.5, dash: "dot" }
  }];

  // Selection highlight: thick border around clicked cell and row.
  if (selectedLR && selectedBranch) {
    const rowIdx = pagedLRs.findIndex(lr => lr.key === selectedLR.key);
    const colIdx = BRANCHES.findIndex(b =>
      b[0] === selectedBranch[0] && b[1] === selectedBranch[1]);
    if (rowIdx >= 0 && colIdx >= 0) {
      shapes.push({
        type: "rect", xref: "x", yref: "y",
        x0: colIdx - 0.5, x1: colIdx + 0.5,
        y0: rowIdx - 0.5, y1: rowIdx + 0.5,
        line: { color: "#000000", width: 3 },
        fillcolor: "rgba(0,0,0,0)"
      });
      // Row-label rectangle in the left margin (paper coords).
      shapes.push({
        type: "rect", xref: "paper", yref: "y",
        x0: -0.30, x1: -0.005,
        y0: rowIdx - 0.5, y1: rowIdx + 0.5,
        line: { color: "#000000", width: 3 },
        fillcolor: "rgba(250, 204, 21, 0.10)"
      });
    }
  }

  // Sizing: enforce min row height. Width is bounded by container,
  // but we keep a min plot width by guarding the section.
  const rowCount = Math.max(pagedLRs.length, 1);
  const plotHeight = Math.max(540, rowCount * MIN_ROW_HEIGHT + 200);
  document.getElementById("plot-a").style.height = plotHeight + "px";

  const layout = {
    margin: { l: 320, r: 90, t: 90, b: 20 },
    xaxis: {
      side: "top",
      showticklabels: false,
      automargin: false,
      fixedrange: true
    },
    yaxis: {
      autorange: "reversed",
      tickfont: { size: 13, color: "#111827" },
      automargin: false,
      fixedrange: true
    },
    annotations: colAnnotations,
    shapes: shapes
  };

  Plotly.react("plot-a", [trace], layout,
    { responsive: true, displayModeBar: false });

  // Pager info + button state
  const pageSizeVal = document.getElementById("pagesize").value;
  const totalRows = filteredLRs.length;
  let info;
  if (pageSizeVal === "all") {
    info = "showing all " + totalRows.toLocaleString() + " rows";
  } else {
    const ps = parseInt(pageSizeVal, 10);
    const startIdx = totalRows === 0 ? 0 : (currentPage - 1) * ps + 1;
    const endIdx = Math.min(currentPage * ps, totalRows);
    info = "Page " + currentPage + " of " + totalPages
      + " (showing rows " + startIdx + "-" + endIdx
      + " of " + totalRows.toLocaleString() + ")";
  }
  document.getElementById("page-info").textContent = info;
  document.getElementById("prev-page").disabled = currentPage <= 1;
  document.getElementById("next-page").disabled =
    pageSizeVal === "all" || currentPage >= totalPages;

  if (!panelABound) {
    panelABound = true;
    document.getElementById("plot-a").on("plotly_click", (ev) => {
      const pt = ev.points && ev.points[0];
      if (!pt) return;
      const yIdx = pt.pointIndex[0], xIdx = pt.pointIndex[1];
      const lr = pagedLRs[yIdx];
      const branch = BRANCHES[xIdx];
      if (!lr || !branch) return;
      selectedLR = lr;
      selectedBranch = branch;
      renderPanelA();
      renderPanelB(lr, branch);
      renderPanelC(lr);
    });
  }
}

function renderPanelB(lr, branch) {
  const branchStr = branch[0] + "->" + branch[1];

  const titleBlock = document.getElementById("b-title-block");
  titleBlock.innerHTML =
    '<div class="bcd-title">Phenotype-level signaling: '
    + lr.ligand + ' → ' + lr.receptor + '</div>'
    + '<div class="bcd-subtitle">'
    + '<span class="tissue-name" style="color:'
    + TISSUE_COLORS[branch[0]] + '">' + branch[0] + '</span>'
    + ' (source) vs '
    + '<span class="tissue-name" style="color:'
    + TISSUE_COLORS[branch[1]] + '">' + branch[1] + '</span>'
    + ' (destination)'
    + '</div>';
  document.getElementById("plot-b").style.display = "block";

  const rows = detailByKey.get(lr.key + "|" + branchStr) || [];
  if (!rows.length) {
    Plotly.purge("plot-b");
    document.getElementById("plot-b").innerHTML =
      '<div class="placeholder">'
      + 'No phenotype-level data for this branch.</div>';
    return;
  }

  const srcRows = rows.filter(r => r.side === "src");
  const dstRows = rows.filter(r => r.side === "dst");

  const sources = Array.from(new Set([
    ...srcRows.map(r => r.source),
    ...dstRows.map(r => r.source)
  ])).filter(x => x != null).sort();
  const targets = Array.from(new Set([
    ...srcRows.map(r => r.target),
    ...dstRows.map(r => r.target)
  ])).filter(x => x != null).sort();

  function build(rows) {
    const idx = new Map();
    rows.forEach(r => idx.set(r.source + "|" + r.target, r));
    const z = sources.map(s => targets.map(t => {
      const r = idx.get(s + "|" + t);
      return r && r.lr_means != null ? r.lr_means : null;
    }));
    const sigX = [], sigY = [];
    rows.forEach(r => {
      if (r.cellphone_pvals != null && r.cellphone_pvals < 0.05) {
        sigX.push(r.target); sigY.push(r.source);
      }
    });
    return { z, sigX, sigY };
  }

  const src = build(srcRows);
  const dst = build(dstRows);

  const flat = [];
  src.z.forEach(row => row.forEach(v => { if (v != null) flat.push(v); }));
  dst.z.forEach(row => row.forEach(v => { if (v != null) flat.push(v); }));
  const zmax = flat.length ? Math.max.apply(null, flat) : 1;

  const traces = [
    {
      z: src.z, x: targets, y: sources,
      type: "heatmap", colorscale: COLORSCALE_SEQUENTIAL,
      zmin: 0, zmax: zmax,
      xaxis: "x", yaxis: "y", showscale: false,
      xgap: 1, ygap: 1,
      hovertemplate:
        "SRC (" + branch[0] + ")<br>source: %{y}<br>"
        + "target: %{x}<br>lr_means: %{z}<extra></extra>"
    },
    {
      z: dst.z, x: targets, y: sources,
      type: "heatmap", colorscale: COLORSCALE_SEQUENTIAL,
      zmin: 0, zmax: zmax,
      xaxis: "x2", yaxis: "y2",
      xgap: 1, ygap: 1,
      colorbar: {
        title: { text: "Interaction strength (lr_means)",
                 font: { size: 12 }, side: "right" },
        thickness: 14, len: 0.85, x: 1.02,
        tickfont: { size: 11 }
      },
      hovertemplate:
        "DST (" + branch[1] + ")<br>source: %{y}<br>"
        + "target: %{x}<br>lr_means: %{z}<extra></extra>"
    },
    {
      x: src.sigX, y: src.sigY,
      xaxis: "x", yaxis: "y",
      type: "scatter", mode: "markers",
      marker: { size: 6, color: "white",
                line: { color: "#222", width: 0.6 } },
      hoverinfo: "skip", showlegend: false
    },
    {
      x: dst.sigX, y: dst.sigY,
      xaxis: "x2", yaxis: "y2",
      type: "scatter", mode: "markers",
      marker: { size: 6, color: "white",
                line: { color: "#222", width: 0.6 } },
      hoverinfo: "skip", showlegend: false
    }
  ];

  const layout = {
    margin: { l: 140, r: 90, t: 50, b: 110 },
    xaxis: {
      title: { text: "target", font: { size: 12, weight: 600 } },
      tickangle: -45,
      tickfont: { size: 12 },
      domain: [0, 0.46]
    },
    yaxis: {
      title: { text: "source", font: { size: 12, weight: 600 } },
      autorange: "reversed",
      tickfont: { size: 12 },
      anchor: "x"
    },
    xaxis2: {
      title: { text: "target", font: { size: 12, weight: 600 } },
      tickangle: -45,
      tickfont: { size: 12 },
      domain: [0.54, 1.0]
    },
    yaxis2: {
      autorange: "reversed",
      tickfont: { size: 12 },
      anchor: "x2",
      showticklabels: false
    },
    annotations: [
      { text: "SRC (" + branch[0] + ")",
        x: 0.23, y: 1.06, xref: "paper", yref: "paper",
        showarrow: false,
        font: { size: 13, color: TISSUE_COLORS[branch[0]] } },
      { text: "DST (" + branch[1] + ")",
        x: 0.77, y: 1.06, xref: "paper", yref: "paper",
        showarrow: false,
        font: { size: 13, color: TISSUE_COLORS[branch[1]] } }
    ]
  };

  Plotly.react("plot-b", traces, layout,
    { responsive: true, displayModeBar: false });
}

function renderPanelC(lr) {
  const titleBlock = document.getElementById("c-title-block");
  titleBlock.innerHTML =
    '<div class="c-title">Signaling intensity over time: '
    + lr.ligand + ' → ' + lr.receptor + '</div>';
  document.getElementById("plot-c").style.display = "block";
  document.getElementById("pwbox").style.display = "block";

  const rows = temporalByLR.get(lr.key) || [];
  const tps = [1, 2, 3, 4, 5, 6];
  const tissues = ["PBMC", "CSF", "TP"];

  const traces = tissues.map(tissue => {
    const r = rows.filter(row => row.tissue === tissue)
                  .sort((a, b) => a.timepoint - b.timepoint);
    const sizes = r.map(row => {
      const p = row.cellphone_pvals;
      if (p == null || isNaN(p)) return 10;
      const np = -Math.log10(Math.max(p, 1e-6));
      return 10 + 4 * np;
    });
    const labels = r.map(row =>
      "p=" + (row.cellphone_pvals != null
              ? row.cellphone_pvals.toExponential(2) : "NA"));
    return {
      x: r.map(row => row.timepoint),
      y: r.map(row => row.lr_means),
      mode: "lines+markers",
      name: tissue,
      line: { color: TISSUE_COLORS[tissue], width: 3 },
      marker: { color: TISSUE_COLORS[tissue], size: sizes,
                line: { color: "#333", width: 0.5 } },
      text: labels,
      hovertemplate: tissue + " t%{x}<br>lr_means: %{y}<br>"
                     + "%{text}<extra></extra>"
    };
  });

  const layout = {
    margin: { l: 80, r: 30, t: 24, b: 60 },
    xaxis: {
      title: { text: "Timepoint",
                font: { size: 14, weight: 600 } },
      range: [0.5, 6.5], tickvals: tps,
      tickfont: { size: 13 }
    },
    yaxis: {
      title: { text: "Interaction intensity",
                font: { size: 14, weight: 600 } },
      tickfont: { size: 13 }
    },
    legend: {
      x: 1, y: 1, xanchor: "right", yanchor: "top",
      font: { size: 13 }
    },
    showlegend: true
  };

  Plotly.react("plot-c", traces, layout,
    { responsive: true, displayModeBar: false });

  // Pathway panel
  const pw = pwLookupByLR.get(lr.key) || {};
  function asPwList(s) {
    if (s == null) return [];
    return String(s).split(";").map(x => x.trim()).filter(Boolean);
  }
  const ligPws = asPwList(pw.ligand_pathways);
  const recPws = asPwList(pw.receptor_pathways);
  const xref = xrefByLR.get(lr.key) || [];

  function pillRow(label, pws, pillClass) {
    if (!pws.length) {
      return '<div class="pwbox-row"><span class="label">'
        + label + '</span><em>(no pathway annotations)</em></div>';
    }
    const tags = pws.map(p =>
      '<span class="pill ' + pillClass + '">' + p + '</span>'
    ).join("");
    return '<div class="pwbox-row"><span class="label">'
      + label + '</span>' + tags + '</div>';
  }

  let html = "";
  html += '<div class="pwbox-section-header">'
       + 'Biological pathways involving this interaction</div>';
  html += pillRow("Ligand (" + lr.ligand + ") is part of:",
                  ligPws, "pill-ligand");
  html += pillRow("Receptor (" + lr.receptor + ") is part of:",
                  recPws, "pill-receptor");

  if (xref.length) {
    html += '<div class="pwbox-section-header">'
         + 'Cross-lineage pathway contexts</div>';
    html += '<div class="pwbox-explanation">'
         + 'This ligand-receptor pair connects T-cell and myeloid '
         + 'pathways whose activity is correlated across patients and '
         + 'timepoints (from cross-lineage correlation analysis). Each '
         + 'row below shows one such connection.</div>';
    xref.forEach(x => {
      const explain = (x.direction === "ligand_in_t_pathway")
        ? "Ligand is expressed in the T-cell pathway; receptor is "
          + "expressed in the myeloid pathway."
        : "Ligand is expressed in the myeloid pathway; receptor is "
          + "expressed in the T-cell pathway.";
      html += '<div class="card">';
      html += '<div class="card-row">T cell pathway: <b>'
           + x.t_pathway + '</b></div>';
      html += '<div class="card-row">Myeloid pathway: <b>'
           + x.m_pathway + '</b></div>';
      html += '<div class="card-explain">' + explain + '</div>';
      html += '</div>';
    });
  }

  document.getElementById("pwbox").innerHTML = html;
}

// Mode toggle. Selection (selectedLR/selectedBranch) is intentionally
// not cleared — Panel 2 and Panel 3 contents are independent of mode.
function setMode(newMode) {
  if (newMode === MODE) return;
  MODE = newMode;
  document.querySelectorAll("#mode-toggle button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === newMode);
  });
  populateSortOptions();
  updateSectionTitle();
  applyFilterSort();
  currentPage = 1;
  renderPanelA();
}
document.querySelectorAll("#mode-toggle button").forEach(btn => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

// Event wiring
document.getElementById("search").addEventListener("input", () => {
  applyFilterSort();
  currentPage = 1;
  renderPanelA();
});
document.getElementById("sort").addEventListener("change", () => {
  applyFilterSort();
  currentPage = 1;
  renderPanelA();
});
document.getElementById("pagesize").addEventListener("change", () => {
  currentPage = 1;
  renderPanelA();
});
document.getElementById("prev-page").addEventListener("click", () => {
  if (currentPage > 1) { currentPage--; renderPanelA(); }
});
document.getElementById("next-page").addEventListener("click", () => {
  currentPage++;
  renderPanelA();
});

// Initial render
updateSectionTitle();
applyFilterSort();
renderPanelA();
</script>
</body>
</html>
"""

# %%
# ---- Substitute placeholders + write HTML ----
print("\nAssembling HTML...")
_t0 = time.time()
build_date = datetime.now().strftime("%Y-%m-%d")
html = HTML_TEMPLATE
html = html.replace("__BUILD_DATE__", build_date)
html = html.replace("__PBMC_COLOR__", TISSUE_COLORS["PBMC"])
html = html.replace("__CSF_COLOR__", TISSUE_COLORS["CSF"])
html = html.replace("__TP_COLOR__", TISSUE_COLORS["TP"])
html = html.replace("__BRANCHES_JSON__",
                     json.dumps([list(b) for b in BRANCHES]))
# B64 payloads last so they don't get masked by other substitutions.
html = html.replace("__ENTRY_B64__", entry_b64)
html = html.replace("__DETAIL_B64__", detail_b64)
html = html.replace("__TEMPORAL_B64__", temporal_b64)
html = html.replace("__XREF_B64__", xref_b64)
html = html.replace("__PW_LOOKUP_B64__", pw_lookup_b64)

OUTPUT_HTML.write_text(html, encoding="utf-8")
print(f"  Elapsed: {time.time() - _t0:.1f}s")

# %%
# ---- Final summary ----
size_bytes = OUTPUT_HTML.stat().st_size
size_mb = size_bytes / 1024 / 1024

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"  L-R pairs in entry matrix: {n_lr_final:,}")
print(f"  entry_matrix.csv      : {len(entry):>10,} rows")
print(f"  phenotype_detail.csv  : {len(detail):>10,} rows")
print(f"  temporal_trajectory.csv: {len(temporal):>10,} rows")
print(f"  pathway_xref.csv      : {len(xref):>10,} rows")
print(f"  pathway_lookup.csv    : {len(pw_lookup):>10,} rows")

try:
    du_out = subprocess.run(
        ["du", "-sh", str(OUTPUT_HTML)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"\n  {du_out}")
except Exception as e:
    print(f"\n  du failed: {e}")

print(f"  HTML size: {size_mb:.2f} MB")
if size_mb > HTML_SIZE_TARGET_MB:
    print(f"  WARNING: HTML > {HTML_SIZE_TARGET_MB:.0f} MB target. "
          "Consider tightening LIANA_PVAL or pre-filtering input.")

elapsed = time.time() - t_start
print(f"  Elapsed total: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
print(f"\nOpen with: open {OUTPUT_HTML}")
landing.write_landing()
print("Done.")
