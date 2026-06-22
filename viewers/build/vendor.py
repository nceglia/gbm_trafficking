"""Small vendor-asset cache for self-contained static viewers."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from viewers.paths import REPO_ROOT

VENDOR_DIR = REPO_ROOT / "build" / "vendor"

VENDOR = {
    "plotly": {
        "file": "plotly-2.35.2.min.js",
        "url": "https://cdn.plot.ly/plotly-2.35.2.min.js",
        "needle": "Plotly",
        "min_bytes": 1_000_000,
    },
    "pako": {
        "file": "pako-2.1.0.min.js",
        "url": "https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js",
        "needle": "pako",
        "min_bytes": 20_000,
    },
    "papaparse": {
        "file": "papaparse-5.4.1.min.js",
        "url": "https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.4.1/papaparse.min.js",
        "needle": "Papa",
        "min_bytes": 10_000,
    },
}


def vendor_path(name: str) -> Path:
    try:
        meta = VENDOR[name]
    except KeyError as exc:
        raise KeyError(f"Unknown vendor asset: {name}") from exc
    return VENDOR_DIR / meta["file"]


def fetch_vendor(name: str) -> str:
    """Return cached vendor JavaScript, fetching it if needed."""
    meta = VENDOR[name]
    path = vendor_path(name)
    if path.exists():
        body = path.read_text(encoding="utf-8")
        if len(body) >= meta["min_bytes"] and meta["needle"] in body:
            return body

    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        meta["url"],
        headers={"User-Agent": "gbm-trafficking-viewer-build/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")

    if len(body) < meta["min_bytes"] or meta["needle"] not in body:
        raise RuntimeError(f"Downloaded vendor asset failed validation: {name}")

    path.write_text(body, encoding="utf-8")
    return body


def inline_script_tags(names: list[str]) -> str:
    blocks = []
    for name in names:
        blocks.append(f"<script>\n{fetch_vendor(name)}\n</script>")
    return "\n".join(blocks)
