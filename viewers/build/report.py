"""Build the static narrative report under deploy/bundle/report/.

The report copies curated, publishable artifacts from results/ into the deploy
bundle. It never copies raw AnnData objects, large caches, or compute-only
intermediates unless they are explicitly added to viewers/report_catalog.yaml.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viewers.build import landing
from viewers.paths import BUNDLE_DIR, REPORT_DIR, ensure_bundle

CATALOG_PATH = REPO_ROOT / "viewers" / "report_catalog.yaml"
PIPELINE_MANIFEST = REPO_ROOT / "pipeline" / "manifest.yaml"

DATA_DIR = REPORT_DIR / "data"
FIG_DIR = REPORT_DIR / "figures"
TEXT_DIR = REPORT_DIR / "text"
FILES_DIR = REPORT_DIR / "files"
ASSETS_DIR = REPORT_DIR / "assets"

GLOB_CHARS = set("*?[]")
TABLE_SUFFIXES = {".csv", ".tsv"}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg"}
TEXT_SUFFIXES = {".txt", ".md", ".log"}
FILE_SUFFIXES = {".pdf"}

SEARCH_HINTS = {
    "phenotype", "contrast", "tissue", "tissue_pair", "term", "pathway",
    "patient", "lineage", "gene", "name", "family", "source", "library",
    "pathway_1", "pathway_2", "edge", "timepoint", "clone_id", "trb",
}


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _size_label(n_bytes: int) -> str:
    if n_bytes >= 1024 * 1024 * 1024:
        return f"{n_bytes / 1024 / 1024 / 1024:.1f} GB"
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / 1024 / 1024:.1f} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.0f} KB"
    return f"{n_bytes} B"


def _caption_from_filename(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"[_-]+", " ", stem)
    return stem.strip().capitalize()


def _safe_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", rel)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return bool(out)
    except Exception:
        return True


def _reset_report() -> None:
    ensure_bundle()
    if REPORT_DIR.exists():
        shutil.rmtree(REPORT_DIR)
    for path in (DATA_DIR, FIG_DIR, TEXT_DIR, FILES_DIR, ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _glob_matches(pattern: str) -> list[Path]:
    if any(ch in pattern for ch in GLOB_CHARS):
        matches = sorted(REPO_ROOT.glob(pattern))
    else:
        p = REPO_ROOT / pattern
        matches = [p] if p.exists() else []
    return [p for p in matches if p.is_file()]


def _detect_search_cols(csv_path: Path) -> list[str]:
    try:
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception:
        return []
    likely = [c for c in cols if c.lower() in SEARCH_HINTS]
    return likely or cols[: min(3, len(cols))]


def _copy_artifact(
    src: Path,
    section_id: str,
    kind: str,
    title: str,
    description: str | None,
    artifacts: list[dict],
) -> dict:
    root = {
        "table": DATA_DIR,
        "figure": FIG_DIR,
        "text": TEXT_DIR,
        "file": FILES_DIR,
    }[kind]
    out_dir = root / section_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / _safe_name(src)
    shutil.copy2(src, dest)
    rel_dest = dest.relative_to(REPORT_DIR).as_posix()
    rel_src = src.relative_to(REPO_ROOT).as_posix()
    size = dest.stat().st_size
    entry = {
        "type": kind,
        "section": section_id,
        "source": rel_src,
        "file": rel_dest,
        "title": title,
        "description": description or "",
        "bytes": size,
        "size": _size_label(size),
        "sha256": _sha256(dest),
    }
    artifacts.append(entry)
    return entry


def _resolve_items(
    section: dict,
    item_key: str,
    kind: str,
    allowed_suffixes: set[str],
    max_bytes: int,
    excluded_suffixes: set[str],
    artifacts: list[dict],
    skipped: list[dict],
) -> list[dict]:
    entries = []
    for item in section.get(item_key, []) or []:
        matches = _glob_matches(item["path"])
        if not matches:
            skipped.append({
                "section": section["id"],
                "source": item["path"],
                "type": kind,
                "reason": "missing",
            })
            continue
        for src in matches:
            rel_src = src.relative_to(REPO_ROOT).as_posix()
            suffix = src.suffix.lower()
            size = src.stat().st_size
            limit = int(item.get("max_bytes", max_bytes))
            if suffix in excluded_suffixes:
                skipped.append({
                    "section": section["id"],
                    "source": rel_src,
                    "type": kind,
                    "reason": f"excluded suffix {suffix}",
                })
                continue
            if allowed_suffixes and suffix not in allowed_suffixes:
                skipped.append({
                    "section": section["id"],
                    "source": rel_src,
                    "type": kind,
                    "reason": f"unsupported suffix {suffix or '(none)'}",
                })
                continue
            if size > limit:
                skipped.append({
                    "section": section["id"],
                    "source": rel_src,
                    "type": kind,
                    "reason": f"{_size_label(size)} exceeds {_size_label(limit)}",
                })
                continue
            base_title = item.get("title") or _caption_from_filename(src)
            title = base_title
            if len(matches) > 1:
                title = f"{base_title}: {_caption_from_filename(src)}"
            entry = _copy_artifact(
                src=src,
                section_id=section["id"],
                kind=kind,
                title=title,
                description=item.get("description"),
                artifacts=artifacts,
            )
            if kind == "table":
                entry["search"] = _detect_search_cols(src)
                entry["delimiter"] = "\t" if suffix == ".tsv" else ","
            entries.append(entry)
    return entries


def _pipeline_payload(manifest: dict) -> dict:
    steps = []
    for step_id, spec in sorted(manifest.get("steps", {}).items()):
        steps.append({
            "id": step_id,
            "tier": spec.get("tier", ""),
            "lineage": spec.get("lineage", ""),
            "script": spec.get("file", ""),
            "tcr_required": bool(spec.get("tcr_required", False)),
            "depends_on": spec.get("depends_on", []),
            "writes": spec.get("writes", []),
            "resources": spec.get("resources", {}),
        })
    tiers = []
    for tier in sorted({s["tier"] for s in steps if s["tier"]}):
        members = [s for s in steps if s["tier"] == tier]
        tiers.append({
            "tier": tier,
            "count": len(members),
            "tcr_required": sum(1 for s in members if s["tcr_required"]),
            "lineages": sorted({s["lineage"] for s in members if s["lineage"]}),
        })
    return {
        "version": manifest.get("version"),
        "tier_summary": tiers,
        "steps": steps,
        "data_prep": manifest.get("data_prep", []),
    }


def _build_sections(catalog: dict, artifacts: list[dict], skipped: list[dict]) -> list[dict]:
    limits = catalog.get("publish_limits", {})
    excluded_suffixes = set(catalog.get("excluded_suffixes", []))
    table_limit = int(limits.get("table_max_bytes", 30_000_000))
    figure_limit = int(limits.get("figure_max_bytes", 25_000_000))
    text_limit = int(limits.get("text_max_bytes", 1_000_000))

    sections = []
    for section in catalog.get("sections", []):
        tables = _resolve_items(
            section, "tables", "table", TABLE_SUFFIXES, table_limit,
            excluded_suffixes, artifacts, skipped,
        )
        figures = _resolve_items(
            section, "figures", "figure", FIGURE_SUFFIXES, figure_limit,
            excluded_suffixes, artifacts, skipped,
        )
        text_blocks = _resolve_items(
            section, "text", "text", TEXT_SUFFIXES, text_limit,
            excluded_suffixes, artifacts, skipped,
        )
        files = _resolve_items(
            section, "files", "file", FILE_SUFFIXES, figure_limit,
            excluded_suffixes, artifacts, skipped,
        )
        sections.append({
            "id": section["id"],
            "title": section["title"],
            "type": "result",
            "lead": section.get("lead", ""),
            "methods": section.get("methods", []),
            "notes": section.get("notes", []),
            "related": section.get("related", []),
            "tables": tables,
            "figures": figures,
            "text": text_blocks,
            "files": files,
        })
    return sections


INDEX_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GBM Trafficking Report</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
  <aside id="sidebar">
    <a class="home" href="../index.html">GBM Trafficking</a>
    <nav id="nav"></nav>
  </aside>
  <main id="content"></main>
  <script src="assets/app.js"></script>
</body>
</html>
"""


STYLES_CSS = """\
* { box-sizing: border-box; }
html, body { margin: 0; min-height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f5f6f8;
  color: #20242a;
  display: flex;
}
#sidebar {
  width: 265px;
  background: #fff;
  border-right: 1px solid #dfe3e8;
  height: 100vh;
  position: sticky;
  top: 0;
  overflow-y: auto;
  padding: 1rem .85rem;
}
.home {
  display: block;
  color: #171a1f;
  font-weight: 700;
  text-decoration: none;
  margin: .25rem .35rem 1rem;
}
#nav a {
  display: block;
  padding: .42rem .55rem;
  border-radius: 6px;
  color: #343941;
  text-decoration: none;
  font-size: .9rem;
  line-height: 1.25;
}
#nav a:hover { background: #edf3f7; }
#nav a.active { background: #dfeef4; color: #0f5168; font-weight: 650; }
#content {
  flex: 1;
  max-width: 1280px;
  padding: 1.6rem 2rem 4rem;
}
h1 { margin: 0 0 .5rem; font-size: 1.7rem; }
h2 { margin: 0 0 .45rem; font-size: 1.35rem; }
h3 { margin: 1.25rem 0 .6rem; font-size: 1.05rem; }
p { line-height: 1.5; }
.lead { color: #404852; max-width: 900px; }
.meta-row, .cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: .75rem;
  margin: 1rem 0 1.25rem;
}
.metric, .panel {
  background: #fff;
  border: 1px solid #dfe3e8;
  border-radius: 8px;
  padding: .85rem;
}
.metric .k { font-size: .76rem; color: #66707d; text-transform: uppercase; }
.metric .v { font-size: 1.1rem; font-weight: 700; margin-top: .2rem; }
.panel { margin: .85rem 0; }
.method-list, .notes { margin: .75rem 0 1rem; padding-left: 1.1rem; }
.notes { color: #52606d; }
.related a, .download {
  display: inline-block;
  color: #0b5d78;
  text-decoration: none;
  font-weight: 600;
  margin-right: .8rem;
}
.related a:hover, .download:hover { text-decoration: underline; }
.toolbar {
  display: flex;
  gap: .6rem;
  align-items: center;
  flex-wrap: wrap;
  margin: .6rem 0;
}
button {
  border: 1px solid #b8c4ce;
  background: #fff;
  color: #1f2a33;
  border-radius: 6px;
  padding: .35rem .65rem;
  font: inherit;
  cursor: pointer;
}
button:hover { background: #eef4f6; }
input[type="search"] {
  min-width: 260px;
  padding: .4rem .55rem;
  border: 1px solid #c5ced6;
  border-radius: 6px;
  font: inherit;
}
.table-wrap { overflow: auto; max-height: 580px; border: 1px solid #dfe3e8; }
table { border-collapse: collapse; width: 100%; background: #fff; font-size: .82rem; }
th, td {
  border-bottom: 1px solid #e9edf1;
  padding: .38rem .5rem;
  text-align: left;
  vertical-align: top;
  white-space: nowrap;
}
th { position: sticky; top: 0; background: #f2f5f7; z-index: 1; }
td.long { max-width: 420px; overflow: hidden; text-overflow: ellipsis; }
.figures {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  gap: 1rem;
}
figure { margin: 0; background: #fff; border: 1px solid #dfe3e8; border-radius: 8px; padding: .75rem; }
figure img { width: 100%; height: auto; display: block; }
figcaption { color: #53606c; font-size: .82rem; margin-top: .45rem; }
pre {
  white-space: pre-wrap;
  background: #17212b;
  color: #edf4f8;
  border-radius: 8px;
  padding: .85rem;
  overflow: auto;
}
.pill {
  display: inline-block;
  border: 1px solid #cad4dc;
  border-radius: 999px;
  padding: .1rem .45rem;
  margin: .08rem;
  color: #41505d;
  background: #fff;
  font-size: .75rem;
}
.empty, .muted { color: #66707d; }
@media (max-width: 760px) {
  body { display: block; }
  #sidebar { position: relative; width: auto; height: auto; }
  #content { padding: 1rem; }
  .figures { grid-template-columns: 1fr; }
}
"""


APP_JS = r"""// Static narrative report renderer. No external JS dependencies.

let MANIFEST = null;

async function init() {
  MANIFEST = await fetch("data/manifest.json").then(r => r.json());
  const nav = document.getElementById("nav");
  MANIFEST.sections.forEach(sec => {
    const a = document.createElement("a");
    a.href = "#" + sec.id;
    a.textContent = sec.title;
    a.dataset.id = sec.id;
    a.addEventListener("click", ev => {
      ev.preventDefault();
      activate(sec.id);
    });
    nav.appendChild(a);
  });
  activate(location.hash.slice(1) || "overview");
}

function activate(id) {
  const sec = MANIFEST.sections.find(s => s.id === id) || MANIFEST.sections[0];
  document.querySelectorAll("#nav a").forEach(a => {
    a.classList.toggle("active", a.dataset.id === sec.id);
  });
  history.replaceState(null, "", "#" + sec.id);
  renderSection(sec);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === "className") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  });
  children.forEach(child => node.appendChild(child));
  return node;
}

function renderSection(sec) {
  const main = document.getElementById("content");
  main.innerHTML = "";
  if (sec.type === "overview") renderOverview(main, sec);
  else if (sec.type === "pipeline") renderPipeline(main, sec);
  else if (sec.type === "artifacts") renderArtifacts(main, sec);
  else renderResult(main, sec);
}

function renderHeader(main, sec) {
  main.appendChild(el("h1", {text: sec.title}));
  if (sec.lead) main.appendChild(el("p", {className: "lead", text: sec.lead}));
}

function renderOverview(main, sec) {
  renderHeader(main, sec);
  const r = MANIFEST.release;
  const metrics = [
    ["Generated", r.generated_at],
    ["Git SHA", r.git_sha + (r.git_dirty ? " (dirty)" : "")],
    ["Copied artifacts", String(r.artifact_count)],
    ["Bundle size", r.bundle_size],
    ["Skipped artifacts", String(MANIFEST.skipped.length)],
  ];
  const row = el("div", {className: "meta-row"});
  metrics.forEach(([k, v]) => {
    row.appendChild(el("div", {className: "metric"}, [
      el("div", {className: "k", text: k}),
      el("div", {className: "v", text: v || "unknown"}),
    ]));
  });
  main.appendChild(row);

  const links = el("div", {className: "panel related"});
  links.appendChild(el("h3", {text: "Open first"}));
  [
    ["Narrative report", "#quickstart"],
    ["Temporal explorer", "../temporal.html"],
    ["Signaling explorer", "../signaling.html"],
    ["Clone network", "../clone_network.html"],
    ["Artifact appendix", "#artifacts"],
  ].forEach(([label, href]) => {
    links.appendChild(el("a", {href, text: label}));
  });
  main.appendChild(links);

  if (sec.notes && sec.notes.length) renderList(main, "Deployment notes", sec.notes, "notes");
}

function renderPipeline(main, sec) {
  renderHeader(main, sec);
  const p = MANIFEST.pipeline;
  const cards = el("div", {className: "cards"});
  p.tier_summary.forEach(t => {
    cards.appendChild(el("div", {className: "metric"}, [
      el("div", {className: "k", text: t.tier}),
      el("div", {className: "v", text: `${t.count} steps`}),
      el("div", {className: "muted", text: `${t.tcr_required} TCR-required; ${t.lineages.join(", ")}`}),
    ]));
  });
  main.appendChild(cards);
  const rows = p.steps.map(s => ({
    step: s.id,
    tier: s.tier,
    lineage: s.lineage,
    script: s.script,
    tcr_required: s.tcr_required ? "yes" : "no",
    writes: (s.writes || []).join("; "),
  }));
  renderObjectTable(main, "Pipeline steps", rows);
}

function renderResult(main, sec) {
  renderHeader(main, sec);
  if (sec.methods && sec.methods.length) renderList(main, "How this was made", sec.methods, "method-list");
  if (sec.notes && sec.notes.length) renderList(main, "Notes", sec.notes, "notes");
  if (sec.related && sec.related.length) {
    const box = el("div", {className: "panel related"});
    sec.related.forEach(link => box.appendChild(el("a", {href: link.href, text: link.label})));
    main.appendChild(box);
  }
  if (sec.tables.length) {
    main.appendChild(el("h2", {text: "Tables"}));
    sec.tables.forEach(t => renderTablePanel(main, t));
  }
  if (sec.figures.length) {
    main.appendChild(el("h2", {text: "Figures"}));
    const grid = el("div", {className: "figures"});
    sec.figures.forEach(f => {
      grid.appendChild(el("figure", {}, [
        el("img", {src: f.file, alt: f.title || f.source, loading: "lazy"}),
        el("figcaption", {text: `${f.title} (${f.size})`}),
      ]));
    });
    main.appendChild(grid);
  }
  if (sec.text.length) {
    main.appendChild(el("h2", {text: "Text outputs"}));
    sec.text.forEach(t => renderTextPanel(main, t));
  }
  if (sec.files.length) {
    main.appendChild(el("h2", {text: "Files"}));
    sec.files.forEach(f => {
      main.appendChild(el("p", {}, [el("a", {className: "download", href: f.file, text: `${f.title} (${f.size})`})]));
    });
  }
  if (!sec.tables.length && !sec.figures.length && !sec.text.length && !sec.files.length) {
    main.appendChild(el("p", {className: "empty", text: "No deployable artifacts were found for this section."}));
  }
}

function renderList(main, title, items, cls) {
  main.appendChild(el("h3", {text: title}));
  const ul = el("ul", {className: cls});
  items.forEach(item => ul.appendChild(el("li", {text: item})));
  main.appendChild(ul);
}

function renderTablePanel(main, spec) {
  const panel = el("section", {className: "panel"});
  panel.appendChild(el("h3", {text: spec.title}));
  panel.appendChild(el("p", {className: "muted", text: `${spec.source} - ${spec.size}`}));
  const toolbar = el("div", {className: "toolbar"});
  const load = el("button", {type: "button", text: "Load table"});
  const search = el("input", {type: "search", placeholder: "Search loaded rows..."});
  const download = el("a", {className: "download", href: spec.file, text: "Download"});
  toolbar.appendChild(load);
  toolbar.appendChild(search);
  toolbar.appendChild(download);
  panel.appendChild(toolbar);
  const host = el("div");
  panel.appendChild(host);
  main.appendChild(panel);
  let rows = null;
  load.addEventListener("click", async () => {
    load.disabled = true;
    load.textContent = "Loading...";
    const text = await fetch(spec.file).then(r => r.text());
    rows = parseCSV(text, spec.delimiter || ",");
    renderObjectTable(host, "", rows, search.value);
    load.textContent = `${rows.length} rows loaded`;
  });
  search.addEventListener("input", () => {
    if (rows) renderObjectTable(host, "", rows, search.value);
  });
}

function renderTextPanel(main, spec) {
  const panel = el("section", {className: "panel"});
  panel.appendChild(el("h3", {text: spec.title}));
  panel.appendChild(el("p", {className: "muted", text: `${spec.source} - ${spec.size}`}));
  const pre = el("pre", {text: "Loading..."});
  panel.appendChild(pre);
  main.appendChild(panel);
  fetch(spec.file).then(r => r.text()).then(text => { pre.textContent = text; });
}

function renderArtifacts(main, sec) {
  renderHeader(main, sec);
  renderObjectTable(main, "Copied artifacts", MANIFEST.artifacts.map(a => ({
    section: a.section,
    type: a.type,
    title: a.title,
    size: a.size,
    source: a.source,
    file: a.file,
    sha256: a.sha256.slice(0, 12),
  })));
  renderObjectTable(main, "Skipped or intentionally excluded", MANIFEST.skipped);
}

function renderObjectTable(main, title, rows, filter = "") {
  const host = title ? el("section", {className: "panel"}) : main;
  if (title) host.appendChild(el("h3", {text: title}));
  if (!rows || !rows.length) {
    host.appendChild(el("p", {className: "empty", text: "No rows."}));
    if (title) main.appendChild(host);
    return;
  }
  const q = (filter || "").toLowerCase();
  const filtered = q ? rows.filter(r => Object.values(r).some(v => String(v ?? "").toLowerCase().includes(q))) : rows;
  const columns = Object.keys(filtered[0] || rows[0]);
  const table = el("table");
  const thead = el("thead");
  thead.appendChild(el("tr", {}, columns.map(c => el("th", {text: c}))));
  table.appendChild(thead);
  const tbody = el("tbody");
  filtered.slice(0, 250).forEach(row => {
    tbody.appendChild(el("tr", {}, columns.map(c => {
      const val = row[c] == null ? "" : String(row[c]);
      return el("td", {className: val.length > 80 ? "long" : "", title: val, text: val});
    })));
  });
  table.appendChild(tbody);
  host.querySelectorAll(".table-wrap, .muted.count").forEach(n => n.remove());
  host.appendChild(el("p", {className: "muted count", text: `Showing ${Math.min(filtered.length, 250)} of ${filtered.length} rows`}));
  host.appendChild(el("div", {className: "table-wrap"}, [table]));
  if (title) main.appendChild(host);
}

function parseCSV(text, delimiter) {
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
      else if (c === delimiter) { row.push(field); field = ""; i++; }
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

init().catch(err => {
  document.getElementById("content").textContent = "Report failed to load: " + err;
});
"""


def _write_frontend() -> None:
    (REPORT_DIR / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (ASSETS_DIR / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (ASSETS_DIR / "app.js").write_text(APP_JS, encoding="utf-8")


def _bundle_size() -> int:
    if not BUNDLE_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in BUNDLE_DIR.rglob("*") if p.is_file())


def main() -> None:
    _reset_report()
    catalog = _read_yaml(CATALOG_PATH)
    pipeline_manifest = _read_yaml(PIPELINE_MANIFEST)
    artifacts: list[dict] = []
    skipped: list[dict] = []

    result_sections = _build_sections(catalog, artifacts, skipped)
    generated_at = datetime.now(timezone.utc).isoformat()

    release = {
        "generated_at": generated_at,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "artifact_count": len(artifacts),
        "skipped_count": len(skipped),
        "bundle_size_bytes": 0,
        "bundle_size": "0 B",
    }

    overview = {
        "id": "overview",
        "title": catalog.get("title", "GBM Trafficking Report"),
        "type": "overview",
        "lead": catalog.get("subtitle", ""),
        "notes": [
            "Generated on the analysis machine and deployed as static files.",
            "Raw AnnData objects, compute caches, and full results trees are not copied to the VM.",
            "The report is curated by viewers/report_catalog.yaml.",
        ],
    }
    pipeline = {
        "id": "pipeline",
        "title": "Pipeline methods",
        "type": "pipeline",
        "lead": "DAG, lineage, dependencies, and resource notes from pipeline/manifest.yaml.",
    }
    appendix = {
        "id": "artifacts",
        "title": "Artifact appendix",
        "type": "artifacts",
        "lead": "Copied files, skipped files, checksums, and source paths.",
    }

    manifest = {
        "generated_at": generated_at,
        "catalog": CATALOG_PATH.relative_to(REPO_ROOT).as_posix(),
        "release": release,
        "pipeline": _pipeline_payload(pipeline_manifest),
        "sections": [overview, pipeline] + result_sections + [appendix],
        "artifacts": artifacts,
        "skipped": skipped,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_frontend()

    release["bundle_size_bytes"] = _bundle_size()
    release["bundle_size"] = _size_label(release["bundle_size_bytes"])
    manifest["release"] = release
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (BUNDLE_DIR / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")

    print("=== gbm_report ===")
    print(f"  sections: {len(result_sections)} result sections")
    print(f"  artifacts copied: {len(artifacts)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  report: {REPORT_DIR / 'index.html'}")
    landing.write_landing()


if __name__ == "__main__":
    main()
