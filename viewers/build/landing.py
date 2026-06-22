"""Write deploy/bundle/index.html for the static viewer bundle."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viewers.paths import (
    BUNDLE_DIR,
    CLONE_NETWORK_HTML,
    LANDING_HTML,
    REPORT_DIR,
    SIGNALING_HTML,
    TEMPORAL_HTML,
    ensure_bundle,
)

_VIEWS = (
    ("report/", REPORT_DIR / "index.html", "Narrative report",
     "Start here for methods, provenance, curated results, and the artifact appendix."),
    ("temporal.html", TEMPORAL_HTML, "Temporal explorer",
     "Interactive phenotype, pathway, and gene trajectories across tissues and timepoints."),
    ("signaling.html", SIGNALING_HTML, "Signaling explorer",
     "Ligand-receptor matrix, temporal trajectories, and pathway context."),
    ("clone_network.html", CLONE_NETWORK_HTML, "Clone network",
     "Interactive clone-sharing graph with T-cell and myeloid composition overlays."),
)


def _size_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if path.is_file():
        return path.stat().st_size
    return 0


def _size_label(path_or_bytes: Path | int) -> str:
    n = _size_bytes(path_or_bytes) if isinstance(path_or_bytes, Path) else path_or_bytes
    if n >= 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024 / 1024:.1f} GB"
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _bundle_size() -> int:
    if not BUNDLE_DIR.exists():
        return 0
    return sum(p.stat().st_size for p in BUNDLE_DIR.rglob("*") if p.is_file())


def write_landing() -> Path:
    ensure_bundle()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    release = _read_json(BUNDLE_DIR / "release.json")
    report_manifest = _read_json(REPORT_DIR / "data" / "manifest.json")
    artifact_count = len(report_manifest.get("artifacts", []))
    skipped_count = len(report_manifest.get("skipped", []))
    section_count = max(0, len(report_manifest.get("sections", [])) - 3)
    git_sha = release.get("git_sha", "unknown")
    git_dirty = " dirty" if release.get("git_dirty") else ""

    cards = []
    for href, path, title, blurb in _VIEWS:
        built = path.exists()
        status = "ready" if built else "missing"
        classes = "card disabled" if not built else "card"
        cards.append(f"""
      <a class="{classes}" href="{href}">
        <h2>{title}</h2>
        <p>{blurb}</p>
        <span>{status} · {_size_label(path)}</span>
      </a>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GBM Trafficking Viewers</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f4f6f7;
      color: #1f252b;
    }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 2rem 1.5rem 3rem; }}
    header {{ margin-bottom: 1.5rem; }}
    h1 {{ margin: 0 0 .4rem; font-size: 1.8rem; }}
    .sub {{ margin: 0; color: #56616c; max-width: 780px; line-height: 1.45; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: .75rem;
      margin: 1.2rem 0;
    }}
    .metric, .card, .notice {{
      background: #fff;
      border: 1px solid #dce2e7;
      border-radius: 8px;
      padding: .9rem;
    }}
    .metric b {{ display: block; font-size: .78rem; color: #66727d; text-transform: uppercase; }}
    .metric span {{ display: block; margin-top: .25rem; font-size: 1.05rem; font-weight: 700; }}
    .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); }}
    .card {{ display: block; text-decoration: none; color: inherit; }}
    .card:hover:not(.disabled) {{ border-color: #1b718c; box-shadow: 0 2px 8px rgba(27, 113, 140, .12); }}
    .card.disabled {{ opacity: .45; pointer-events: none; }}
    .card h2 {{ margin: 0 0 .45rem; font-size: 1.05rem; }}
    .card p {{ margin: 0 0 .65rem; color: #48545e; line-height: 1.4; font-size: .9rem; }}
    .card span {{ color: #66727d; font-size: .78rem; text-transform: uppercase; }}
    .notice {{ margin-top: 1rem; color: #48545e; line-height: 1.45; }}
    code {{ background: #eef2f4; padding: .08rem .25rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>GBM Trafficking Pipeline Viewers</h1>
      <p class="sub">Static outputs built on the analysis machine for internal review. Start with the narrative report, then use the explorers for deeper interaction.</p>
    </header>
    <section class="metrics">
      <div class="metric"><b>Landing generated</b><span>{generated}</span></div>
      <div class="metric"><b>Git SHA</b><span>{git_sha}{git_dirty}</span></div>
      <div class="metric"><b>Bundle size</b><span>{_size_label(_bundle_size())}</span></div>
      <div class="metric"><b>Report sections</b><span>{section_count}</span></div>
      <div class="metric"><b>Copied artifacts</b><span>{artifact_count}</span></div>
      <div class="metric"><b>Skipped artifacts</b><span>{skipped_count}</span></div>
    </section>
    <section class="grid">
{"".join(cards)}
    </section>
    <section class="notice">
      This bundle is designed for deployment to <code>slvicosspecdat1</code> as static files only.
      Raw AnnData objects, compute caches, and full local <code>data/</code> or <code>results/</code> trees are not copied.
    </section>
  </main>
</body>
</html>
"""
    LANDING_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {LANDING_HTML}")
    return LANDING_HTML


def main() -> None:
    write_landing()


if __name__ == "__main__":
    main()
