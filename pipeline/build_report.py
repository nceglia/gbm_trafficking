"""Consolidate results/ into a static gbm_report/ site.

Reads existing CSVs/PNGs under results/, copies/concatenates them into
gbm_report/, writes a manifest.json, and lays down a vanilla-JS frontend
(Tabulator + Plotly via CDN). No analysis is re-run.

Open via a local server (fetch() is blocked over file:// in Chrome):
    python -m http.server -d gbm_report
"""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
REPORT_DIR = REPO_ROOT / "gbm_report"

DATA_DIR = REPORT_DIR / "data"
FIG_DIR = REPORT_DIR / "figures"
ASSETS_DIR = REPORT_DIR / "assets"


# ---------- helpers ----------------------------------------------------------

def caption_from_filename(name: str) -> str:
    return Path(name).stem.replace("_", " ")


def reset_report():
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    for d in (DATA_DIR, FIG_DIR, ASSETS_DIR):
        d.mkdir(parents=True)


def copy_pngs(src_files, dest_subdir):
    src_files = list(src_files)
    if not src_files:
        return []
    out_dir = FIG_DIR / dest_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for src in src_files:
        shutil.copy2(src, out_dir / src.name)
        entries.append({
            "file": f"figures/{dest_subdir}/{src.name}",
            "caption": caption_from_filename(src.name),
        })
    return entries


SEARCH_HINTS = {
    "phenotype", "contrast", "tissue", "tissue_pair", "term", "pathway",
    "patient", "lineage", "gene", "name", "family", "source", "library",
    "pathway_1", "pathway_2",
}


def detect_search_cols(csv_path: Path) -> list:
    cols = list(pd.read_csv(csv_path, nrows=0).columns)
    likely = [c for c in cols if c.lower() in SEARCH_HINTS]
    return likely or cols[:1]


# ---------- (a) tissue separability -----------------------------------------

def section_tissue_separability():
    src_dir = RESULTS_DIR / "03_tissue_separability"
    if not src_dir.is_dir():
        return None

    figures = copy_pngs(sorted(src_dir.glob("*.png")), "tissue_separability")
    csv_renames = [
        ("augur_auc_summary.csv", "augur_results.csv", "Augur AUC"),
        ("augur_per_patient.csv", "augur_per_patient.csv", "Augur per patient"),
        ("cosine_distance_summary.csv", "cosine_distance_summary.csv", "Cosine distance summary"),
    ]
    tables = []
    for src_name, dst_name, title in csv_renames:
        src = src_dir / src_name
        if not src.exists():
            continue
        dst = DATA_DIR / dst_name
        shutil.copy2(src, dst)
        tables.append({
            "id": dst_name.replace(".csv", ""),
            "file": f"data/{dst_name}",
            "title": title,
            "search": detect_search_cols(dst),
        })

    if not figures and not tables:
        return None
    return {
        "id": "tissue_separability",
        "title": "Tissue separability",
        "type": "section",
        "tables": tables,
        "figures": figures,
    }


# ---------- (b) pseudobulk DE + GSEA ----------------------------------------

DE_RE = re.compile(r"^de_(?P<lineage>[A-Z0-9]+)_(?P<contrast>.+)\.csv$")
GSEA_RE = re.compile(
    r"^gsea_(?P<source>hallmark|kegg|gobp)_(?P<lineage>[A-Z0-9]+)_(?P<contrast>.+)\.csv$"
)


def _read_de(p: Path, lineage: str, contrast: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "gene" not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "gene"})
    df["lineage"] = lineage
    df["contrast"] = contrast
    return df


def section_pseudobulk_de_gsea():
    src_dir = RESULTS_DIR / "04_pseudobulk_de_gsea"
    if not src_dir.is_dir():
        return None

    tables = []

    de_frames = []
    for p in sorted(src_dir.glob("de_*.csv")):
        m = DE_RE.match(p.name)
        if not m:
            continue
        de_frames.append(_read_de(p, m.group("lineage"), m.group("contrast")))
    if de_frames:
        de_path = DATA_DIR / "de_results.csv"
        pd.concat(de_frames, ignore_index=True).to_csv(de_path, index=False)
        tables.append({
            "id": "de_results",
            "file": "data/de_results.csv",
            "title": "Differential expression (clone-pseudobulk DESeq2)",
            "search": detect_search_cols(de_path),
        })

    gsea_frames = []
    for p in sorted(src_dir.glob("gsea_*.csv")):
        m = GSEA_RE.match(p.name)
        if not m:
            continue
        df = pd.read_csv(p)
        df["source"] = m.group("source")
        df["lineage"] = m.group("lineage")
        df["contrast"] = m.group("contrast")
        gsea_frames.append(df)
    if gsea_frames:
        gsea_path = DATA_DIR / "gsea_results.csv"
        pd.concat(gsea_frames, ignore_index=True).to_csv(gsea_path, index=False)
        tables.append({
            "id": "gsea_results",
            "file": "data/gsea_results.csv",
            "title": "GSEA results",
            "search": detect_search_cols(gsea_path),
        })

    # Top-level PNGs that are NOT claimed by ternary/coenrichment sections.
    enrich_pngs = [
        p for p in sorted(src_dir.glob("*.png"))
        if not p.name.startswith("pathway_ternary")
        and not p.name.startswith("pathway_coenrichment")
    ]
    figures = copy_pngs(enrich_pngs, "pathway_enrichment")

    if not tables and not figures:
        return None
    return {
        "id": "pathway_enrichment",
        "title": "Pseudobulk DE + pathway enrichment",
        "type": "section",
        "tables": tables,
        "figures": figures,
    }


# ---------- (c) pathway visualizations --------------------------------------

def section_pathway_ternary():
    src_dir = RESULTS_DIR / "04_pseudobulk_de_gsea"
    if not src_dir.is_dir():
        return None
    pngs = sorted(src_dir.glob("pathway_ternary*.png"))
    if not pngs:
        return None
    return {
        "id": "pathway_ternary",
        "title": "Pathway ternary",
        "type": "section",
        "tables": [],
        "figures": copy_pngs(pngs, "pathway_ternary"),
    }


def section_pathway_coenrichment():
    src_dir = RESULTS_DIR / "04_pseudobulk_de_gsea"
    if not src_dir.is_dir():
        return None
    pngs = sorted(src_dir.glob("pathway_coenrichment*.png"))
    if not pngs:
        return None
    return {
        "id": "pathway_coenrichment",
        "title": "Pathway co-enrichment",
        "type": "section",
        "tables": [],
        "figures": copy_pngs(pngs, "pathway_coenrichment"),
    }


EDGES_RE = re.compile(r"^edges_(?P<library>[a-z]+)_(?P<lineage>[A-Z0-9]+)\.csv$")
AFFINITY_RE = re.compile(r"^gene_tissue_affinity_(?P<library>[a-z]+)_(?P<lineage>[A-Z0-9]+)\.csv$")


def section_pathway_networks():
    src_dir = RESULTS_DIR / "04_pseudobulk_de_gsea" / "networks"
    if not src_dir.is_dir():
        return None

    tables = []

    edges_frames = []
    for p in sorted(src_dir.glob("edges_*.csv")):
        m = EDGES_RE.match(p.name)
        if not m:
            continue
        df = pd.read_csv(p)
        df["library"] = m.group("library")
        df["lineage"] = m.group("lineage")
        edges_frames.append(df)
    if edges_frames:
        edges_path = DATA_DIR / "pathway_edges.csv"
        pd.concat(edges_frames, ignore_index=True).to_csv(edges_path, index=False)
        tables.append({
            "id": "pathway_edges",
            "file": "data/pathway_edges.csv",
            "title": "Pathway co-membership edges",
            "search": detect_search_cols(edges_path),
        })

    affinity_frames = []
    for p in sorted(src_dir.glob("gene_tissue_affinity_*.csv")):
        m = AFFINITY_RE.match(p.name)
        if not m:
            continue
        df = pd.read_csv(p)
        df["library"] = m.group("library")
        df["lineage"] = m.group("lineage")
        affinity_frames.append(df)
    if affinity_frames:
        affinity_path = DATA_DIR / "gene_tissue_affinity.csv"
        pd.concat(affinity_frames, ignore_index=True).to_csv(affinity_path, index=False)
        tables.append({
            "id": "gene_tissue_affinity",
            "file": "data/gene_tissue_affinity.csv",
            "title": "Gene tissue affinity",
            "search": detect_search_cols(affinity_path),
        })

    figures = copy_pngs(sorted(src_dir.glob("*.png")), "pathway_networks")

    if not tables and not figures:
        return None
    return {
        "id": "pathway_networks",
        "title": "Pathway proximity networks",
        "type": "section",
        "tables": tables,
        "figures": figures,
    }


# ---------- frontend templates ----------------------------------------------

INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GBM Trafficking Report</title>
  <link rel="stylesheet" href="https://unpkg.com/tabulator-tables@6/dist/css/tabulator.min.css">
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
  <aside id="sidebar">
    <h1>GBM Trafficking</h1>
    <nav id="nav"></nav>
  </aside>
  <main id="content"></main>
  <script src="https://unpkg.com/tabulator-tables@6/dist/js/tabulator.min.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="assets/app.js"></script>
</body>
</html>
"""

STYLES_CSS = """\
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f7f7f8;
  color: #222;
  display: flex;
  min-height: 100vh;
}
#sidebar {
  width: 240px;
  background: #fff;
  border-right: 1px solid #e2e2e6;
  position: sticky;
  top: 0;
  align-self: flex-start;
  height: 100vh;
  overflow-y: auto;
  padding: 1.25rem 1rem;
}
#sidebar h1 { font-size: 1.05rem; margin: 0 0 1rem 0; letter-spacing: .01em; }
#nav a {
  display: block;
  padding: .4rem .6rem;
  color: #333;
  text-decoration: none;
  border-radius: 4px;
  font-size: .92rem;
  margin-bottom: 2px;
}
#nav a.active { background: #eef2ff; color: #2952cc; font-weight: 600; }
#nav a:hover { background: #f0f0f4; }
#content {
  flex: 1;
  padding: 1.5rem 2rem 4rem;
  max-width: 1200px;
}
section { margin-bottom: 2.5rem; }
h2 { margin-top: 0; font-size: 1.4rem; }
h3 { font-size: 1.05rem; margin: 1.5rem 0 .5rem; }
.search-box {
  width: 100%;
  padding: .5rem .75rem;
  margin-bottom: .5rem;
  border: 1px solid #d0d0d6;
  border-radius: 4px;
  font-size: .92rem;
  font-family: inherit;
}
.tabulator { font-size: .82rem; }
.figures { display: grid; grid-template-columns: 1fr; gap: 1.5rem; }
@media (min-width: 1100px) {
  .figures { grid-template-columns: 1fr 1fr; }
}
figure { margin: 0; background: #fff; padding: .75rem; border: 1px solid #e2e2e6; border-radius: 6px; }
figure img { max-width: 100%; height: auto; display: block; }
figcaption { font-size: .82rem; color: #555; margin-top: .35rem; font-family: ui-monospace, monospace; }
.empty { color: #888; font-style: italic; }
"""

APP_JS = r"""// vanilla-JS report renderer; manifest-driven sidebar + tabulator tables.

async function init() {
  const manifest = await fetch("data/manifest.json").then(r => r.json());
  const nav = document.getElementById("nav");
  manifest.sections.forEach(sec => {
    const a = document.createElement("a");
    a.textContent = sec.title;
    a.href = "#" + sec.id;
    a.dataset.id = sec.id;
    a.addEventListener("click", e => {
      e.preventDefault();
      activate(sec.id, manifest);
    });
    nav.appendChild(a);
  });
  const initial = window.location.hash.slice(1) || manifest.sections[0].id;
  activate(initial, manifest);
}

function activate(id, manifest) {
  const sec = manifest.sections.find(s => s.id === id);
  if (!sec) return;
  document.querySelectorAll("#nav a").forEach(a => {
    a.classList.toggle("active", a.dataset.id === id);
  });
  history.replaceState(null, "", "#" + id);
  renderSection(sec, manifest);
}

function renderSection(sec, manifest) {
  const main = document.getElementById("content");
  main.innerHTML = "";
  const heading = document.createElement("h2");
  heading.textContent = sec.title;
  main.appendChild(heading);

  if (sec.type === "static") {
    const intro = document.createElement("p");
    intro.textContent = "GBM trafficking analysis report. Use the sidebar to navigate.";
    main.appendChild(intro);

    const summary = document.createElement("div");
    const ts = manifest.generated_at || "";
    summary.innerHTML =
      "<p><strong>Generated:</strong> " + ts + "</p>" +
      "<p><strong>Sections:</strong> " +
      manifest.sections.filter(s => s.type !== "static").map(s => s.title).join(", ") +
      "</p>";
    main.appendChild(summary);
    return;
  }

  (sec.tables || []).forEach(t => renderTable(main, t));

  const figs = sec.figures || [];
  if (figs.length) {
    const grid = document.createElement("div");
    grid.className = "figures";
    figs.forEach(f => {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = f.file;
      img.loading = "lazy";
      img.alt = f.caption || "";
      const cap = document.createElement("figcaption");
      cap.textContent = f.caption || "";
      fig.appendChild(img);
      fig.appendChild(cap);
      grid.appendChild(fig);
    });
    main.appendChild(grid);
  } else if (!(sec.tables || []).length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No outputs on disk yet.";
    main.appendChild(empty);
  }
}

function renderTable(main, spec) {
  const wrap = document.createElement("section");
  const h = document.createElement("h3");
  h.textContent = spec.title;
  wrap.appendChild(h);

  const search = document.createElement("input");
  search.className = "search-box";
  search.placeholder = "Search across all columns...";
  wrap.appendChild(search);

  const div = document.createElement("div");
  wrap.appendChild(div);
  main.appendChild(wrap);

  fetch(spec.file).then(r => r.text()).then(csv => {
    const rows = parseCSV(csv);
    if (!rows.length) {
      div.textContent = "(empty)";
      return;
    }
    const cols = Object.keys(rows[0]).map(k => ({
      title: k, field: k, headerFilter: "input", resizable: true,
      formatter: (cell) => {
        const v = cell.getValue();
        if (typeof v === "string" && v.length > 80) {
          const span = document.createElement("span");
          span.title = v;
          span.textContent = v.slice(0, 80) + "...";
          return span;
        }
        return v == null ? "" : v;
      },
    }));
    const tab = new Tabulator(div, {
      data: rows,
      columns: cols,
      pagination: "local",
      paginationSize: 50,
      layout: "fitDataStretch",
      height: "560px",
    });

    let timer = null;
    search.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const q = search.value.toLowerCase();
        if (!q) { tab.clearFilter(true); return; }
        tab.setFilter((data) => {
          for (const v of Object.values(data)) {
            if (v != null && String(v).toLowerCase().includes(q)) return true;
          }
          return false;
        });
      }, 200);
    });
  }).catch(err => {
    div.textContent = "Failed to load " + spec.file + ": " + err;
  });
}

// Minimal RFC-ish CSV parser (handles quoted fields w/ commas, newlines, "" escapes).
function parseCSV(text) {
  const rows = [];
  let i = 0, field = "", row = [], inQuote = false;
  while (i < text.length) {
    const c = text[i];
    if (inQuote) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQuote = false; i++; continue;
      }
      field += c; i++;
    } else {
      if (c === '"') { inQuote = true; i++; }
      else if (c === ",") { row.push(field); field = ""; i++; }
      else if (c === "\r") { i++; }
      else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; i++; }
      else { field += c; i++; }
    }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  if (!rows.length) return [];
  const header = rows.shift();
  return rows.filter(r => r.length === header.length).map(r => {
    const obj = {};
    header.forEach((h, j) => { obj[h] = r[j]; });
    return obj;
  });
}

init();
"""


# ---------- driver ----------------------------------------------------------

def write_frontend():
    (REPORT_DIR / "index.html").write_text(INDEX_HTML)
    (ASSETS_DIR / "styles.css").write_text(STYLES_CSS)
    (ASSETS_DIR / "app.js").write_text(APP_JS)


def main():
    reset_report()

    sections = [{"id": "overview", "title": "Overview", "type": "static"}]
    for builder in (
        section_tissue_separability,
        section_pseudobulk_de_gsea,
        section_pathway_ternary,
        section_pathway_coenrichment,
        section_pathway_networks,
    ):
        s = builder()
        if s is not None:
            sections.append(s)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": sections,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    write_frontend()

    # Summary
    print("=== gbm_report/data/ ===")
    for p in sorted(DATA_DIR.iterdir()):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")
    fig_count = sum(1 for p in FIG_DIR.rglob("*") if p.is_file())
    print(f"=== gbm_report/figures/: {fig_count} files ===")
    for sub in sorted(p for p in FIG_DIR.iterdir() if p.is_dir()):
        n = sum(1 for f in sub.rglob("*") if f.is_file())
        print(f"  {sub.name}/: {n}")
    print(f"=== open: {(REPORT_DIR / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
