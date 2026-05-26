"""Build deploy/bundle/temporal.html — self-contained temporal explorer.

Embeds the per-(patient, tissue, timepoint, phenotype) tables produced by
pathway_temporal_scores_*.py as gzip+base64 inline payloads, and lays down
a Plotly + Papaparse + pako single-file explorer with selectable
phenotypes/pathways/genes across 9 small multiples (3 features x 3 tissues).
"""
import base64
import gzip
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viewers.paths import TEMPORAL_HTML, TEMPORAL_SCORES_DIR, ensure_bundle
from viewers.build import landing

INPUT_DIR = TEMPORAL_SCORES_DIR
OUTPUT_PATH = TEMPORAL_HTML

INPUTS = [
    ("comp_tcell",   "data-comp-tcell",   "temporal_composition_tcell.csv"),
    ("comp_myeloid", "data-comp-myeloid", "temporal_composition_myeloid.csv"),
    ("path_tcell",   "data-path-tcell",   "temporal_pathway_scores_tcell.csv"),
    ("path_myeloid", "data-path-myeloid", "temporal_pathway_scores_myeloid.csv"),
    ("gene_tcell",   "data-gene-tcell",   "temporal_gene_expression_tcell.csv"),
    ("gene_myeloid", "data-gene-myeloid", "temporal_gene_expression_myeloid.csv"),
    ("pathdef",      "data-pathdef",      "pathway_definitions.csv"),
]


def encode_table(df: pd.DataFrame) -> tuple[str, list[str]]:
    df = df.copy()
    # n_cells is kept now: the explorer uses it for log-scaled dot sizing
    # and the dot-hover "n=<n_cells>" suffix.
    if "lineage" in df.columns and df["lineage"].nunique(dropna=True) <= 1:
        df = df.drop(columns=["lineage"])
    for c in df.select_dtypes(include="float").columns:
        df[c] = df[c].round(3)
    csv = df.to_csv(index=False).encode("utf-8")
    gz = gzip.compress(csv, compresslevel=6)
    b64 = base64.b64encode(gz).decode("ascii")
    return b64, list(df.columns)


def main():
    ensure_bundle()
    payloads: dict[str, str] = {}
    for key, _tag, fname in INPUTS:
        path = INPUT_DIR / fname
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        b64, cols = encode_table(df)
        payloads[key] = b64
        print(
            f"  {fname}: rows={len(df):>9,}  "
            f"b64={len(b64):>11,} chars (~{len(b64)/1024/1024:5.1f} MB)  "
            f"cols={cols}"
        )

    blocks = "\n".join(
        f'<script id="{tag}" type="application/octet-stream">{payloads[key]}</script>'
        for key, tag, _ in INPUTS
    )
    html = TEMPLATE.replace("__DATA_BLOCKS__", blocks)
    OUTPUT_PATH.write_text(html, encoding="utf-8")

    size_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nWrote {OUTPUT_PATH} ({size_mb:.1f} MB)")
    landing.write_landing()
    print(f"Landing: {OUTPUT_PATH.parent / 'index.html'}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GBM Temporal Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
  <script src="https://unpkg.com/papaparse@5/papaparse.min.js"></script>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; height: 100%;
                 font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                 color: #222; background: #f7f7f8; }
    #loading { padding: 2rem; font-size: 1rem; color: #666; }
    #loading .sub { color: #999; font-size: .85rem; margin-top: .35rem; }
    #controls {
      position: sticky; top: 0; z-index: 50;
      background: #fff;
      border-bottom: 1px solid #ddd;
      padding: .5rem 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,.05);
    }
    .control-row {
      display: flex; align-items: center; gap: .4rem;
      flex-wrap: wrap;
      padding: .2rem 0;
      font-size: .85rem;
    }
    .control-row > label { font-weight: 600; min-width: 90px; }
    .chip {
      display: inline-flex; align-items: center; gap: .25rem;
      padding: .15rem .45rem;
      background: #eef2ff; color: #2952cc;
      border-radius: 12px;
      font-size: .8rem;
    }
    .chip .x { cursor: pointer; opacity: .55; padding: 0 .15rem; }
    .chip .x:hover { opacity: 1; }
    .chip-pheno { background: #e9f5ec; color: #1f6f3a; }
    .chip-path  { background: #fef2e0; color: #874f10; }
    .chip-gene  { background: #fce4ec; color: #8b1d4a; }
    .add-btn {
      background: #fff; border: 1px solid #ccc; border-radius: 4px;
      padding: .2rem .55rem; cursor: pointer; font-size: .8rem;
    }
    .add-btn:hover { background: #f0f0f4; }
    .check-group label { margin-right: .65rem; cursor: pointer; }
    .picker {
      position: absolute; z-index: 100;
      background: #fff; border: 1px solid #bbb; border-radius: 6px;
      box-shadow: 0 4px 18px rgba(0,0,0,.12);
      padding: .35rem;
      width: 340px;
      max-height: 340px;
      flex-direction: column;  /* takes effect once JS sets display:flex */
    }
    .picker input {
      width: 100%; padding: .35rem .5rem;
      border: 1px solid #d0d0d6; border-radius: 4px;
      font-size: .85rem;
      margin-bottom: .35rem;
      font-family: inherit;
    }
    .picker ul {
      list-style: none; padding: 0; margin: 0;
      overflow-y: auto;
      flex: 1;
      font-size: .82rem;
    }
    .picker li {
      padding: .3rem .5rem;
      cursor: pointer;
      border-radius: 3px;
    }
    .picker li:hover, .picker li.active { background: #eef2ff; }
    .picker li.sep {
      color: #888; font-size: .72rem; pointer-events: none;
      border-top: 1px solid #e3e3e6; margin-top: .3rem; padding-top: .35rem;
      text-transform: uppercase; letter-spacing: .04em;
    }
    .picker li .meta { color: #888; font-size: .75rem; margin-left: .35rem; }
    #plot { width: 100%; height: calc(100vh - 220px); min-height: 540px; }
    .hint { padding: .5rem 1rem; color: #666; font-size: .78rem; }
  </style>
</head>
<body>
  <div id="loading">
    Decoding data...
    <div class="sub">First load decompresses ~150 MB of inline data; expect 5-15 seconds.</div>
  </div>
  <div id="app" style="display:none">
    <div id="controls">
      <div class="control-row">
        <label>Phenotypes:</label>
        <span class="chips" data-kind="phenotypes"></span>
        <button class="add-btn" data-add="phenotypes">+ add</button>
      </div>
      <div class="control-row">
        <label>Pathways:</label>
        <span class="chips" data-kind="pathways"></span>
        <button class="add-btn" data-add="pathways">+ add</button>
      </div>
      <div class="control-row">
        <label>Genes:</label>
        <span class="chips" data-kind="genes"></span>
        <button class="add-btn" data-add="genes">+ add</button>
      </div>
      <div class="control-row check-group">
        <label><input type="checkbox" data-toggle="showSD" checked> Mean &plusmn; SD</label>
        <label><input type="checkbox" data-toggle="showDots" checked> Patient dots</label>
        <span style="margin-left:.75rem"></span>
        <label>Y-axis:</label>
        <label><input type="radio" name="ymode" value="raw" checked> Raw</label>
        <label title="(x - mean) / std across the 18 (tissue, timepoint) points within each path/gene trace. Activates only when the row has >=2 series. Y range fixed at [-3, 3]."><input type="radio" name="ymode" value="zscore"> Z-score</label>
        <label title="(x - min) / (max - min) within each path/gene trace. Y range fixed at [0, 1]."><input type="radio" name="ymode" value="minmax"> Min-max</label>
      </div>
    </div>
    <div id="plot"></div>
    <p class="hint">Click a legend entry to toggle that series across all 9 subplots. Pickers support case-insensitive substring search; gene tags <code>(T)</code>/<code>(M)</code> indicate which lineage panel a gene came from.</p>
  </div>

  <div class="picker" id="picker" style="display: none">
    <input class="picker-search" placeholder="Search...">
    <ul class="picker-list"></ul>
  </div>

__DATA_BLOCKS__

  <script>
    /* =====================================================================
     *  GBM Temporal Explorer  --  vanilla JS app
     * ===================================================================== */

    /* ---------- Constants ----------
     * Line-friendly qualitative palettes. The previous Set2/Dark2/Set3 mix
     * was tuned for categorical fills (low saturation, near-white pastels)
     * and made line traces hard to tell apart on a white background. The
     * three palettes below are line-mode equivalents (D3 tab10, ColorBrewer
     * Set1, ColorBrewer Dark2 + extras), all saturated enough to differentiate
     * and with no near-white yellow.
     * Each palette cycles modulo length when more series than colors exist.
     */
    const COMP_PALETTE = [
      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ];  // tab10
    const PATH_PALETTE = [
      '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
      '#a65628', '#f781bf', '#999999', '#dede00', '#00ced1'
    ];  // ColorBrewer Set1 (saturated, distinct)
    const GENE_PALETTE = [
      '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e',
      '#e6ab02', '#a6761d', '#666666', '#1f78b4', '#b2df8a'
    ];  // ColorBrewer Dark2 + extras
    const PALETTES = {comp: COMP_PALETTE, path: PATH_PALETTE, gene: GENE_PALETTE};
    const TISSUES   = ['PBMC','CSF','TP'];
    const ROWS      = ['comp','path','gene'];
    // Y-axis titles (leftmost plot of each row); switch to "Z-score" when normalize is on.
    const Y_AXIS_TITLE = {comp: 'Fraction of cells', path: 'Pathway score', gene: 'Mean expression'};
    // Vertical row labels on the left of the figure.
    const ROW_LEFT_LABEL = {comp: 'Composition', path: 'Pathway score', gene: 'Gene expression'};
    const VALUE_KEY = {comp:'frac', path:'mean_score', gene:'mean_expr'};
    const CELLS_KEY = {comp:'n_cells_phenotype', path:'n_cells', gene:'n_cells'};
    const LINEAGE_TAG = {tcell:'(T)', myeloid:'(M)'};

    // Patient-dot radius scales log10 with cell count (clamped 3..14 px).
    function dotSize(n) {
      return Math.min(14, Math.max(3, Math.log10((n || 0) + 1) * 3 + 3));
    }

    /* ---------- State ---------- */
    const state = {
      phenotypes: [],   // [{name, lineage:'tcell'|'myeloid'}]
      pathways:   [],   // [{name, source:'hallmark'|'kegg'}]
      genes:      [],   // [{name, in:'T'|'M'|'both'}]
      showSD:     true,
      showDots:   true,
      // Y-axis transform applied to path/gene rows when the row has >=2
      // series. Composition is always 'raw'. Single-series rows are
      // always 'raw' regardless of this setting.
      yMode:      'raw',  // 'raw' | 'zscore' | 'minmax'
    };

    /* ---------- Tables + indices ---------- */
    let TABLES = {};
    let IDX = {};        // pre-computed lookup indices
    let UNIVERSE = {phenotypes:[], pathways:[], genes:[]};

    /* ---------- Decoding ---------- */
    async function decodeAll() {
      const ids = [
        ['data-comp-tcell',   'comp_tcell'],
        ['data-comp-myeloid', 'comp_myeloid'],
        ['data-path-tcell',   'path_tcell'],
        ['data-path-myeloid', 'path_myeloid'],
        ['data-gene-tcell',   'gene_tcell'],
        ['data-gene-myeloid', 'gene_myeloid'],
        ['data-pathdef',      'pathdef'],
      ];
      const out = {};
      for (const [tagId, key] of ids) {
        document.getElementById('loading').firstChild.nodeValue = 'Decoding data: ' + key + '...';
        await new Promise(r => setTimeout(r, 0));
        const b64 = document.getElementById(tagId).textContent.trim();
        const bin = atob(b64);
        const u8 = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
        const decompressed = pako.ungzip(u8);
        const csv = new TextDecoder('utf-8').decode(decompressed);
        const result = Papa.parse(csv, {header: true, dynamicTyping: true, skipEmptyLines: true});
        out[key] = result.data;
      }
      return out;
    }

    /* ---------- Index builders for fast filter ---------- */
    function indexBy(rows, keyFn) {
      const m = new Map();
      for (const r of rows) {
        const k = keyFn(r);
        if (k == null) continue;
        const arr = m.get(k);
        if (arr) arr.push(r);
        else m.set(k, [r]);
      }
      return m;
    }
    function buildIndices() {
      IDX.comp_tcell   = indexBy(TABLES.comp_tcell,   r => r.phenotype + '|' + r.tissue);
      IDX.comp_myeloid = indexBy(TABLES.comp_myeloid, r => r.phenotype + '|' + r.tissue);
      IDX.path_tcell   = indexBy(TABLES.path_tcell,   r => r.pathway + '|' + r.phenotype + '|' + r.tissue);
      IDX.path_myeloid = indexBy(TABLES.path_myeloid, r => r.pathway + '|' + r.phenotype + '|' + r.tissue);
      IDX.gene_tcell   = indexBy(TABLES.gene_tcell,   r => r.gene + '|' + r.phenotype + '|' + r.tissue);
      IDX.gene_myeloid = indexBy(TABLES.gene_myeloid, r => r.gene + '|' + r.phenotype + '|' + r.tissue);
    }

    /* ---------- Universe (picker contents) ---------- */
    function buildUniverse() {
      const tcellPhenos = new Set();
      for (const r of TABLES.comp_tcell)   if (r.phenotype != null) tcellPhenos.add(String(r.phenotype));
      const myPhenos = new Set();
      for (const r of TABLES.comp_myeloid) if (r.phenotype != null) myPhenos.add(String(r.phenotype));
      const phenos = [];
      [...tcellPhenos].sort().forEach(p => phenos.push({name: p, lineage: 'tcell'}));
      [...myPhenos].sort().forEach(p   => phenos.push({name: p, lineage: 'myeloid'}));
      UNIVERSE.phenotypes = phenos;

      UNIVERSE.pathways = TABLES.pathdef
        .filter(r => r.pathway != null)
        .map(r => ({name: String(r.pathway), source: String(r.source || '')}))
        .sort((a, b) => a.name.localeCompare(b.name));

      const inT = new Set();
      const inM = new Set();
      for (const r of TABLES.gene_tcell)   if (r.gene != null) inT.add(String(r.gene));
      for (const r of TABLES.gene_myeloid) if (r.gene != null) inM.add(String(r.gene));
      const all = new Set([...inT, ...inM]);
      UNIVERSE.genes = [...all].sort().map(g => ({
        name: g,
        in: inT.has(g) && inM.has(g) ? 'both' : (inT.has(g) ? 'T' : 'M'),
      }));
    }

    /* ---------- Color management (one palette per row) ---------- */
    const COLOR_MAPS = {comp: new Map(), path: new Map(), gene: new Map()};
    function phenoKey(p) { return p.lineage + '|' + p.name; }
    function phenoLabel(p) { return p.name + ' ' + LINEAGE_TAG[p.lineage]; }
    function colorKey(rowKind, pheno, featKey) {
      // Composition keyed by phenotype only; path/gene keyed by (pheno, feature).
      return rowKind === 'comp'
        ? phenoKey(pheno)
        : phenoKey(pheno) + '|' + featKey;
    }
    function colorFor(rowKind, pheno, featKey) {
      const m = COLOR_MAPS[rowKind];
      const k = colorKey(rowKind, pheno, featKey);
      if (!m.has(k)) {
        const pal = PALETTES[rowKind];
        m.set(k, pal[m.size % pal.length]);
      }
      return m.get(k);
    }
    function hexToRgb(hex) {
      const h = hex.replace('#','');
      return {r:parseInt(h.slice(0,2),16), g:parseInt(h.slice(2,4),16), b:parseInt(h.slice(4,6),16)};
    }
    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    /* ---------- Lookup helpers ---------- */
    function lookup(rowKind, pheno, feature, tissue) {
      // Returns array of rows for the (pheno, feature, tissue) combo
      const idxKey = (rowKind === 'comp' ? 'comp_' : rowKind === 'path' ? 'path_' : 'gene_') +
                     pheno.lineage;
      const idx = IDX[idxKey];
      if (!idx) return [];
      const k = (rowKind === 'comp')
        ? (pheno.name + '|' + tissue)
        : (feature + '|' + pheno.name + '|' + tissue);
      return idx.get(k) || [];
    }

    /* ---------- Aggregation ---------- */
    function aggregate(rows, valueKey, cellsKey) {
      const byT = new Map();
      for (const r of rows) {
        const t = r.timepoint;
        if (t == null) continue;
        const v = r[valueKey];
        if (v == null || isNaN(v)) continue;
        const n = cellsKey ? (+r[cellsKey] || 0) : 0;
        const entry = {patient: r.patient, value: +v, n_cells: n};
        const arr = byT.get(t);
        if (arr) arr.push(entry);
        else byT.set(t, [entry]);
      }
      const ts = [...byT.keys()].sort((a, b) => a - b);
      const tArr = [], meanArr = [], sdArr = [], pointArr = [];
      for (const t of ts) {
        const grp = byT.get(t);
        const vs = grp.map(d => d.value);
        const m = vs.reduce((a, b) => a + b, 0) / vs.length;
        const sd = vs.length > 1
          ? Math.sqrt(vs.reduce((a, b) => a + (b - m) * (b - m), 0) / (vs.length - 1))
          : 0;
        tArr.push(t); meanArr.push(m); sdArr.push(sd);
        for (const d of grp) pointArr.push({t, value: d.value, patient: d.patient, n_cells: d.n_cells});
      }
      return {tArr, meanArr, sdArr, pointArr};
    }

    /* ---------- Trace + layout builders ---------- */
    function subplotIdx(r, c) { return r * 3 + c; }
    function axRef(idx) {
      const s = idx === 0 ? '' : (idx + 1).toString();
      return {x: 'x' + s, y: 'y' + s};
    }

    function makeTraces() {
      const traces = [];
      const cellHasData = new Array(9).fill(false);

      // ---- Aggregation cache (avoids double work when y-mode transforms run) ----
      const aggCache = new Map();
      function cachedAgg(rowKind, pheno, featKey, tissue) {
        const k = rowKind + '|' + phenoKey(pheno) + '|' + featKey + '|' + tissue;
        if (!aggCache.has(k)) {
          const rows = lookup(rowKind, pheno, featKey, tissue);
          aggCache.set(k, rows.length ? aggregate(rows, VALUE_KEY[rowKind], CELLS_KEY[rowKind]) : null);
        }
        return aggCache.get(k);
      }

      // ---- Series counts per row (used to gate normalization) ----
      // A "series" = one (phenotype, feature) line drawn across all 3 tissues.
      const seriesCount = {
        comp: state.phenotypes.length,
        path: state.phenotypes.length * state.pathways.length,
        gene: state.phenotypes.length * state.genes.length,
      };

      // ---- Y-axis transform stats per (rowKind, pheno, feat) ----
      // Computed across all 3 tissues * timepoints within each trace.
      // Composition is always 'raw'; single-series rows are also 'raw'.
      // Z-score:  v -> (v - mu) / sigma
      // Min-max:  v -> (v - min) / (max - min) ; skipped if max == min.
      const yTransform = new Map();  // key -> {kind, mu, sigma} or {kind, min, max}
      if (state.yMode !== 'raw') {
        for (const rowKind of ['path', 'gene']) {
          if (seriesCount[rowKind] < 2) continue;  // single-series rows: raw
          const featList = (rowKind === 'path')
            ? state.pathways.map(p => p.name)
            : state.genes.map(g => g.name);
          for (const pheno of state.phenotypes) {
            for (const featKey of featList) {
              const all = [];
              for (const tissue of TISSUES) {
                const a = cachedAgg(rowKind, pheno, featKey, tissue);
                if (a) for (const v of a.meanArr) all.push(v);
              }
              if (all.length < 2) continue;
              const key = rowKind + '|' + phenoKey(pheno) + '|' + featKey;
              if (state.yMode === 'zscore') {
                const mu = all.reduce((s, x) => s + x, 0) / all.length;
                const var_ = all.reduce((s, x) => s + (x - mu) * (x - mu), 0) / (all.length - 1);
                const sigma = Math.sqrt(var_);
                if (sigma > 0) yTransform.set(key, {kind: 'zscore', mu, sigma});
              } else if (state.yMode === 'minmax') {
                let lo = Infinity, hi = -Infinity;
                for (const v of all) { if (v < lo) lo = v; if (v > hi) hi = v; }
                if (hi > lo) yTransform.set(key, {kind: 'minmax', min: lo, max: hi});
                // else: max == min, leave raw (per spec)
              }
            }
          }
        }
      }

      // Phenotype-major iteration so the legend lays out as:
      //   <Pheno A title>   composition / pathway1 / pathway2 / gene1
      //   <Pheno B title>   composition / pathway1 / ...
      // legendgroup = phenoKey only (one block per phenotype). legendgrouptitle
      // is set on the FIRST trace of each phenotype's block only. Toggle of all
      // 3 tissue cells is handled by a custom plotly_legendclick handler in
      // attachLegendHandler() that flips visible on every trace sharing
      // meta.seriesId; legend.groupclick='toggleitem' is the safe default.
      const seenSeries  = new Set();   // seriesId -> first tissue trace already emitted
      const seenPhenoTitle = new Set();  // phenoKey -> legendgrouptitle already attached

      const sortedPhenos = [...state.phenotypes].sort((a, b) => a.name.localeCompare(b.name));

      for (const pheno of sortedPhenos) {
        const phenoK = phenoKey(pheno);

        for (const rowKind of ROWS) {
          const r = ROWS.indexOf(rowKind);

          let features;
          if (rowKind === 'comp') {
            features = [{key: '__comp__', label: 'composition'}];
          } else if (rowKind === 'path') {
            features = state.pathways.map(p => ({key: p.name, label: p.name + ' (' + p.source + ')'}));
          } else {
            features = state.genes.map(g => ({
              key: g.name,
              label: g.name + ' (gene)' + (g.in === 'both' ? '' : ' ' + LINEAGE_TAG[g.in === 'T' ? 'tcell' : 'myeloid']),
            }));
          }

          for (const feat of features) {
            const color    = colorFor(rowKind, pheno, feat.key);
            const seriesId = phenoK + '|' + rowKind + '|' + feat.key;

            for (const tissue of TISSUES) {
              const c = TISSUES.indexOf(tissue);
              const ax = axRef(subplotIdx(r, c));

              let agg = cachedAgg(rowKind, pheno, feat.key, tissue);
              if (!agg || !agg.tArr.length) continue;

              // Apply Y-axis transform (path/gene rows when row has >=2 series).
              if (rowKind !== 'comp') {
                const t = yTransform.get(rowKind + '|' + phenoK + '|' + feat.key);
                if (t) {
                  let xform, sxform;
                  if (t.kind === 'zscore') {
                    xform  = (v) => (v - t.mu) / t.sigma;
                    sxform = (s) => s / t.sigma;
                  } else {
                    const span = t.max - t.min;
                    xform  = (v) => (v - t.min) / span;
                    sxform = (s) => s / span;
                  }
                  agg = {
                    tArr: agg.tArr,
                    meanArr:  agg.meanArr.map(xform),
                    sdArr:    agg.sdArr.map(sxform),
                    pointArr: agg.pointArr.map(p => ({
                      t: p.t, patient: p.patient,
                      n_cells: p.n_cells,
                      value: xform(p.value),
                    })),
                  };
                }
              }

              cellHasData[subplotIdx(r, c)] = true;

              const isFirstForSeries = !seenSeries.has(seriesId);
              if (isFirstForSeries) seenSeries.add(seriesId);

              const setTitle = isFirstForSeries && !seenPhenoTitle.has(phenoK);
              if (setTitle) seenPhenoTitle.add(phenoK);

              const traceName = feat.label;

              // SD ribbon (two zero-line traces; tonexty fills between them)
              if (state.showSD && agg.sdArr.some(s => s > 0)) {
                const lower = agg.meanArr.map((m, i) => m - agg.sdArr[i]);
                const upper = agg.meanArr.map((m, i) => m + agg.sdArr[i]);
                const rgb = hexToRgb(color);
                const fill = 'rgba(' + rgb.r + ',' + rgb.g + ',' + rgb.b + ',0.15)';
                traces.push({
                  x: agg.tArr, y: lower,
                  mode: 'lines', line: {width: 0, color: 'rgba(0,0,0,0)'},
                  legendgroup: phenoK, showlegend: false, hoverinfo: 'skip',
                  meta: {seriesId: seriesId},
                  xaxis: ax.x, yaxis: ax.y,
                });
                traces.push({
                  x: agg.tArr, y: upper,
                  mode: 'lines', line: {width: 0, color: 'rgba(0,0,0,0)'},
                  fill: 'tonexty', fillcolor: fill,
                  legendgroup: phenoK, showlegend: false, hoverinfo: 'skip',
                  meta: {seriesId: seriesId},
                  xaxis: ax.x, yaxis: ax.y,
                });
              }

              // Mean line (single legend entry per series, on first tissue only)
              const meanTrace = {
                x: agg.tArr, y: agg.meanArr,
                mode: 'lines+markers',
                line:   {color: color, width: 2},
                marker: {color: color, size: 5},
                name: traceName,
                legendgroup: phenoK,
                showlegend: isFirstForSeries,
                meta: {seriesId: seriesId},
                xaxis: ax.x, yaxis: ax.y,
                hovertemplate:
                  '<b>' + escapeHtml(phenoLabel(pheno)) + '</b><br>' +
                  escapeHtml(traceName) + '<br>' +
                  'tissue: ' + tissue + '<br>' +
                  'timepoint: %{x}<br>' +
                  'mean: %{y:.3f}<extra></extra>',
              };
              if (setTitle) {
                meanTrace.legendgrouptitle = {text: phenoLabel(pheno)};
              }
              traces.push(meanTrace);

              // Patient dots — size scales log10 with cell count
              if (state.showDots && agg.pointArr.length) {
                const xs    = agg.pointArr.map(p => p.t + (Math.random() - 0.5) * 0.2);
                const ys    = agg.pointArr.map(p => p.value);
                const sizes = agg.pointArr.map(p => dotSize(p.n_cells));
                traces.push({
                  x: xs, y: ys,
                  mode: 'markers',
                  marker: {
                    color: color,
                    size: sizes,
                    opacity: 0.55,
                    line: {width: 0.5, color: 'rgba(0,0,0,0.3)'},
                  },
                  legendgroup: phenoK, showlegend: false, hoverinfo: 'text',
                  meta: {seriesId: seriesId},
                  text: agg.pointArr.map(p =>
                    phenoLabel(pheno) + ' | ' + traceName +
                    '<br>patient: ' + p.patient +
                    '<br>tissue: ' + tissue +
                    '<br>timepoint: ' + p.t +
                    '<br>value: ' + p.value.toFixed(3) +
                    '<br>n=' + p.n_cells
                  ),
                  xaxis: ax.x, yaxis: ax.y,
                });
              }
            }
          }
        }
      }

      return {traces, cellHasData};
    }

    function makeLayout(cellHasData) {
      const layout = {
        grid: {rows: 3, columns: 3, pattern: 'independent', roworder: 'top to bottom'},
        margin: {t: 60, l: 110, r: 30, b: 150},
        legend: {
          orientation: 'h',
          x: 0, y: -0.16,
          xanchor: 'left', yanchor: 'top',
          // Custom plotly_legendclick handler does the cross-tissue toggle;
          // 'toggleitem' is the safe fallback if the handler is absent.
          groupclick: 'toggleitem',
          tracegroupgap: 15,
          itemsizing: 'constant',
          font: {size: 10},
        },
        hovermode: 'closest',
        annotations: [],
      };

      // Returns 'raw' | 'zscore' | 'minmax' for a row given current state.
      // Composition is always 'raw'. Single-series rows are always 'raw'.
      const rowYMode = (rowKind) => {
        if (rowKind === 'comp' || state.yMode === 'raw') return 'raw';
        const n = state.phenotypes.length *
                  (rowKind === 'path' ? state.pathways.length : state.genes.length);
        return n >= 2 ? state.yMode : 'raw';
      };
      const yTitle = (rowKind) => {
        const m = rowYMode(rowKind);
        if (m === 'zscore') return 'Z-score';
        if (m === 'minmax') return 'Min-max';
        return Y_AXIS_TITLE[rowKind];
      };

      for (let i = 0; i < 9; i++) {
        const r = Math.floor(i / 3);
        const c = i % 3;
        const xkey = 'xaxis' + (i === 0 ? '' : (i + 1));
        const ykey = 'yaxis' + (i === 0 ? '' : (i + 1));
        layout[xkey] = {
          range: [0.5, 6.5], dtick: 1,
          showgrid: true,
          title: '',
          ticks: 'outside',
        };
        if (i > 0) layout[xkey].matches = 'x';

        // Y axes: each cell autoranges independently. No matches between
        // cells, no shared range. Composition uses rangemode:'tozero' so the
        // bottom is anchored at 0; pathway/gene let the auto range drift to
        // negatives as needed. Normalize-per-row mode (path/gene with >=2
        // series) overrides with a hard [-3, 3] range.
        const rowKind = ROWS[r];
        layout[ykey] = {
          showgrid: true,
          title: c === 0 ? {text: yTitle(rowKind), font: {size: 11}} : '',
          ticks: 'outside',
        };
        const m = rowYMode(rowKind);
        if (m === 'zscore') {
          layout[ykey].range = [-3, 3];
          layout[ykey].autorange = false;
        } else if (m === 'minmax') {
          layout[ykey].range = [0, 1];
          layout[ykey].autorange = false;
        } else {
          layout[ykey].autorange = true;
          if (rowKind === 'comp') layout[ykey].rangemode = 'tozero';
        }
      }

      // Single x-axis label at bottom, on the bottom-row middle plot's title.
      layout.xaxis8.title = {text: 'Timepoint', font: {size: 12, color: '#333'}};

      // Column headers above row 0.
      const colXrefs = ['x', 'x2', 'x3'];
      for (let c = 0; c < 3; c++) {
        layout.annotations.push({
          xref: colXrefs[c] + ' domain', yref: 'paper',
          x: 0.5, y: 1.04,
          xanchor: 'center', yanchor: 'bottom',
          text: '<b>' + TISSUES[c] + '</b>',
          showarrow: false, font: {size: 13, color: '#333'},
        });
      }

      // Row labels left of each row, rotated 90 degrees.
      const rowYrefs = ['y', 'y4', 'y7'];
      for (let r = 0; r < 3; r++) {
        layout.annotations.push({
          xref: 'paper', yref: rowYrefs[r] + ' domain',
          x: -0.06, y: 0.5,
          xanchor: 'center', yanchor: 'middle',
          text: '<b>' + ROW_LEFT_LABEL[ROWS[r]] + '</b>',
          showarrow: false, font: {size: 12, color: '#333'},
          textangle: -90,
        });
      }

      // Always render the 3x3 grid. Annotate empty visible cells with "no data".
      // When zero phenotypes are selected, place a single "Pick a phenotype above"
      // hint in the top-left cell only.
      if (state.phenotypes.length === 0) {
        layout.annotations.push({
          xref: 'x domain', yref: 'y domain',
          x: 0.05, y: 0.95,
          xanchor: 'left', yanchor: 'top',
          text: 'Pick a phenotype above',
          showarrow: false, font: {color: '#999', size: 11},
        });
      } else {
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            const idx = subplotIdx(r, c);
            if (cellHasData[idx]) continue;
            const s = idx === 0 ? '' : (idx + 1);
            layout.annotations.push({
              xref: 'x' + s + ' domain',
              yref: 'y' + s + ' domain',
              x: 0.5, y: 0.5,
              text: 'no data',
              showarrow: false, font: {color: '#bbb', size: 11},
            });
          }
        }
      }
      return layout;
    }

    /* ---------- Render ----------
     * First paint is deferred one animation frame so the #app container -- which
     * was just flipped from display:none to display:block -- has a chance to
     * lay out. Otherwise Plotly measures the still-collapsed container and the
     * 3x3 grid in `pattern: independent` collapses into a single visible cell.
     * After every paint we also call Plotly.Plots.resize() to force a relayout
     * against the actual rendered container size. Subsequent renders use
     * Plotly.react for in-place updates.
     */
    let _plotInitialized = false;
    function render() {
      const {traces, cellHasData} = makeTraces();
      const layout = makeLayout(cellHasData);
      const plotDiv = document.getElementById('plot');
      const config = {responsive: true, displaylogo: false};

      const draw = () => {
        if (!_plotInitialized) {
          Plotly.newPlot(plotDiv, traces, layout, config);
          _plotInitialized = true;
        } else {
          Plotly.react(plotDiv, traces, layout, config);
        }
        Plotly.Plots.resize(plotDiv);
        attachLegendHandler();
      };

      if (!_plotInitialized) {
        requestAnimationFrame(draw);
      } else {
        draw();
      }
    }

    /* ---------- Legend click handler ----------
     * legendgroup is per-phenotype, so all of a phenotype's entries share a
     * single clustered title in the legend. Plotly's built-in togglegroup
     * would hide every entry of the phenotype on a single click; toggleitem
     * would only flip the visible-legend trace and leave the other 2 tissue
     * cells visible. Neither is what we want. Override: on click, find every
     * trace with the same meta.seriesId (across all 3 tissues plus SD ribbon
     * + patient-dot helpers) and flip them in lockstep.
     */
    function attachLegendHandler() {
      const plot = document.getElementById('plot');
      if (plot._gbmLegendHandlerAttached) return;
      plot._gbmLegendHandlerAttached = true;
      plot.on('plotly_legendclick', function(ev) {
        const ci = ev.curveNumber;
        const fullData = ev.fullData;
        const clicked = fullData[ci];
        if (!clicked || !clicked.meta || !clicked.meta.seriesId) return true;
        const sid = clicked.meta.seriesId;
        const wasVisible = clicked.visible !== 'legendonly' && clicked.visible !== false;
        const newVisible = wasVisible ? 'legendonly' : true;
        const indices = [];
        for (let i = 0; i < fullData.length; i++) {
          if (fullData[i].meta && fullData[i].meta.seriesId === sid) indices.push(i);
        }
        if (indices.length) Plotly.restyle(plot, {visible: newVisible}, indices);
        return false;  // suppress Plotly's default single-trace toggle
      });
    }

    /* ---------- Chips ---------- */
    function renderChips() {
      for (const kind of ['phenotypes','pathways','genes']) {
        const container = document.querySelector('.chips[data-kind="' + kind + '"]');
        container.innerHTML = '';
        const cls = kind === 'phenotypes' ? 'chip-pheno'
                  : kind === 'pathways'   ? 'chip-path'
                                          : 'chip-gene';
        for (let i = 0; i < state[kind].length; i++) {
          const item = state[kind][i];
          const label = (
            kind === 'phenotypes' ? phenoLabel(item)
            : kind === 'pathways' ? item.name + ' (' + item.source + ')'
            : item.name + (item.in === 'both' ? '' : ' (' + item.in + ')')
          );
          const chip = document.createElement('span');
          chip.className = 'chip ' + cls;
          chip.innerHTML = escapeHtml(label) + ' <span class="x" title="remove">&times;</span>';
          chip.querySelector('.x').addEventListener('click', () => {
            state[kind].splice(i, 1);
            renderChips();
            render();
          });
          container.appendChild(chip);
        }
      }
    }

    /* ---------- Picker / dropdown management ----------
     * Single shared picker DOM element (#picker) reused for the three kinds.
     * `activeDropdown` tracks which trigger is currently associated with it.
     * Visibility is driven via inline `style.display` so the author CSS rule
     * `.picker { ... }` (no `display`) doesn't out-specificity-override us.
     */
    let activeDropdown = null;  // {name, panel, trigger} | null

    function closeActiveDropdown() {
      if (activeDropdown === null) return;
      activeDropdown.panel.style.display = 'none';
      activeDropdown = null;
    }
    // Back-compat alias used elsewhere in the file.
    const closePicker = closeActiveDropdown;

    function openPicker(kind, anchor) {
      const picker = document.getElementById('picker');
      // If the same trigger is already active, treat the click as a toggle-close.
      if (activeDropdown !== null && activeDropdown.trigger === anchor) {
        closeActiveDropdown();
        return;
      }
      if (activeDropdown !== null) closeActiveDropdown();

      picker.dataset.kind = kind;
      const rect = anchor.getBoundingClientRect();
      picker.style.top  = (rect.bottom + window.scrollY + 4) + 'px';
      picker.style.left = (rect.left   + window.scrollX) + 'px';
      picker.style.display = 'flex';

      const search = picker.querySelector('input');
      search.value = '';
      populatePicker('');
      search.focus();

      activeDropdown = {name: kind, panel: picker, trigger: anchor};
    }
    function populatePicker(query) {
      const picker = document.getElementById('picker');
      const kind = picker.dataset.kind;
      const list = picker.querySelector('ul');
      list.innerHTML = '';
      const q = query.trim().toLowerCase();
      let items = [];
      const matchName = obj => !q || obj.name.toLowerCase().includes(q);

      if (kind === 'phenotypes') {
        const tcell = UNIVERSE.phenotypes.filter(p => p.lineage === 'tcell').filter(matchName);
        const my    = UNIVERSE.phenotypes.filter(p => p.lineage === 'myeloid').filter(matchName);
        if (tcell.length) { items.push({sep: 'T cell'}); items.push(...tcell); }
        if (my.length)    { items.push({sep: 'Myeloid'}); items.push(...my); }
      } else if (kind === 'pathways') {
        items = UNIVERSE.pathways.filter(p => !q || p.name.toLowerCase().includes(q) || p.source.toLowerCase().includes(q));
      } else {
        items = UNIVERSE.genes.filter(matchName);
      }

      const cap = items.slice(0, 300);
      for (const item of cap) {
        const li = document.createElement('li');
        if (item.sep) {
          li.className = 'sep'; li.textContent = item.sep;
        } else if (kind === 'phenotypes') {
          li.innerHTML = escapeHtml(item.name) + ' <span class="meta">' + LINEAGE_TAG[item.lineage] + '</span>';
          li.addEventListener('click', () => addItem(kind, item));
        } else if (kind === 'pathways') {
          li.innerHTML = escapeHtml(item.name) + ' <span class="meta">(' + escapeHtml(item.source) + ')</span>';
          li.addEventListener('click', () => addItem(kind, item));
        } else {
          const tag = item.in === 'both' ? '' : '(' + item.in + ')';
          li.innerHTML = escapeHtml(item.name) + (tag ? ' <span class="meta">' + tag + '</span>' : '');
          li.addEventListener('click', () => addItem(kind, item));
        }
        list.appendChild(li);
      }
      if (items.length > cap.length) {
        const li = document.createElement('li');
        li.className = 'sep';
        li.textContent = '... ' + (items.length - cap.length) + ' more (refine search)';
        list.appendChild(li);
      }
      if (!cap.length) {
        const li = document.createElement('li');
        li.className = 'sep'; li.textContent = 'no matches';
        list.appendChild(li);
      }
    }
    function addItem(kind, item) {
      const exists = state[kind].some(x =>
        kind === 'phenotypes' ? (x.name === item.name && x.lineage === item.lineage)
                              : (x.name === item.name)
      );
      if (!exists) {
        state[kind].push(item);
        renderChips();
        render();
      }
      // Keep dropdown open so the user can add more entries; clear the search
      // and re-filter the visible list. No stopPropagation -- the document
      // mousedown handler will see the click is inside the panel and not close.
      const picker = document.getElementById('picker');
      const search = picker.querySelector('input');
      search.value = '';
      populatePicker('');
      search.focus();
    }

    /* ---------- Wire up ---------- */
    function wireControls() {
      document.querySelectorAll('button.add-btn').forEach(btn => {
        btn.addEventListener('click', e => {
          // Keep the click out of the document mousedown's bubble so the
          // dropdown we're about to open isn't immediately re-closed.
          e.stopPropagation();
          openPicker(btn.dataset.add, btn);
        });
      });
      const picker = document.getElementById('picker');
      picker.querySelector('input').addEventListener('input', e => populatePicker(e.target.value));

      // Single document-level outside-click listener, attached ONCE.
      // capture: true so we see the event before any bubble-stage handler can
      // intercept. We deliberately don't stopPropagation on item clicks inside
      // the panel; this listener sees panel.contains(target) and short-circuits.
      document.addEventListener('mousedown', (e) => {
        if (activeDropdown === null) return;
        const panel = activeDropdown.panel;
        const trigger = activeDropdown.trigger;
        if (panel.contains(e.target) || trigger.contains(e.target)) return;
        closeActiveDropdown();
      }, true);

      // ESC closes any open dropdown.
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && activeDropdown !== null) {
          closeActiveDropdown();
        }
      });
      document.querySelectorAll('input[data-toggle]').forEach(cb => {
        cb.addEventListener('change', () => {
          state[cb.dataset.toggle] = cb.checked;
          render();
        });
      });
      // Y-axis mode radio (raw / zscore / minmax)
      document.querySelectorAll('input[name="ymode"]').forEach(rb => {
        rb.addEventListener('change', () => {
          if (rb.checked) {
            state.yMode = rb.value;
            render();
          }
        });
      });

      // Keep the 3x3 grid laid out correctly when the window changes size.
      // (Plotly's responsive:true does most of this; an explicit Plots.resize
      // call also reasserts axis domains in pattern: independent mode.)
      window.addEventListener('resize', () => {
        const plotDiv = document.getElementById('plot');
        if (_plotInitialized && plotDiv) Plotly.Plots.resize(plotDiv);
      });
    }

    /* ---------- Boot ---------- */
    (async function boot() {
      try {
        TABLES = await decodeAll();
        document.getElementById('loading').firstChild.nodeValue = 'Indexing...';
        await new Promise(r => setTimeout(r, 0));
        buildIndices();
        buildUniverse();
      } catch (e) {
        document.getElementById('loading').textContent = 'Decode failed: ' + e;
        throw e;
      }
      document.getElementById('loading').style.display = 'none';
      document.getElementById('app').style.display = 'block';
      wireControls();
      render();
    })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
