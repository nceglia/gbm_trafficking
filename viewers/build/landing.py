"""Write deploy/bundle/index.html — landing page for all built viewers."""

from __future__ import annotations

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
    ("temporal.html", TEMPORAL_HTML, "Temporal scores",
     "Phenotype, pathway, and gene trajectories across tissues and timepoints."),
    ("signaling.html", SIGNALING_HTML, "L-R signaling",
     "Ligand–receptor signaling across tissue branches and time."),
    ("clone_network.html", CLONE_NETWORK_HTML, "Clone network",
     "Interactive clone sharing graph (T cell + myeloid)."),
    ("report/", REPORT_DIR / "index.html", "Results report",
     "Searchable tables and figures from pipeline CSV/PNG outputs."),
)


def _size_label(path: Path) -> str:
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if total >= 1024 * 1024:
            return f"{total / 1024 / 1024:.1f} MB"
        return f"{total / 1024:.0f} KB"
    if path.is_file():
        n = path.stat().st_size
        if n >= 1024 * 1024:
            return f"{n / 1024 / 1024:.1f} MB"
        return f"{n / 1024:.0f} KB"
    return "not built"


def write_landing() -> Path:
    ensure_bundle()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    cards = []
    for href, path, title, blurb in _VIEWS:
        built = path.exists()
        status = "ready" if built else "missing"
        cards.append(f"""
      <a class="card {'disabled' if not built else ''}" href="{href}">
        <h2>{title}</h2>
        <p>{blurb}</p>
        <span class="meta">{status} · {_size_label(path)}</span>
      </a>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GBM Trafficking — Viewers</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 2rem 1.5rem;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f4f5f7; color: #1a1a1a;
      max-width: 920px; margin-inline: auto;
    }}
    h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
    .sub {{ color: #666; font-size: .9rem; margin-bottom: 1.5rem; }}
    .grid {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }}
    .card {{
      display: block; padding: 1rem 1.1rem;
      background: #fff; border: 1px solid #ddd; border-radius: 8px;
      text-decoration: none; color: inherit;
      box-shadow: 0 1px 2px rgba(0,0,0,.04);
      transition: border-color .15s, box-shadow .15s;
    }}
    .card:hover:not(.disabled) {{
      border-color: #2952cc;
      box-shadow: 0 2px 8px rgba(41,82,204,.12);
    }}
    .card.disabled {{ opacity: .45; pointer-events: none; }}
    .card h2 {{ margin: 0 0 .4rem; font-size: 1.05rem; }}
    .card p {{ margin: 0 0 .6rem; font-size: .85rem; color: #444; line-height: 1.4; }}
    .meta {{ font-size: .75rem; color: #888; text-transform: uppercase; letter-spacing: .03em; }}
  </style>
</head>
<body>
  <h1>GBM Trafficking — Pipeline Viewers</h1>
  <p class="sub">Generated {generated}. Serve this folder over HTTP (see deploy/README.md).</p>
  <div class="grid">
{"".join(cards)}
  </div>
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
