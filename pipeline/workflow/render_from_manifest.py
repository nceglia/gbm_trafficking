#!/usr/bin/env python3
"""Regenerate pipeline/dag.md from pipeline/manifest.yaml.

This is documentation-only: it does not affect execution.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "pipeline" / "manifest.yaml"
DAG_MD_PATH = REPO_ROOT / "pipeline" / "dag.md"

TIER_ORDER = [
    "qc",
    "pathway_tcell",
    "pathway_myeloid",
    "pathway_cross",
    "traffic_tcr",
    "signaling",
    "figures",
    "explorers",
]

TIER_CLASS = {
    "qc": "qc",
    "pathway_tcell": "pathway",
    "pathway_myeloid": "pathway",
    "pathway_cross": "pathway",
    "traffic_tcr": "traffic",
    "signaling": "signaling",
    "figures": "figure",
    "explorers": "explorer",
}


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def topological_waves(steps: dict) -> list[list[str]]:
    deps = {k: list(v.get("depends_on") or []) for k, v in steps.items()}
    in_deg = {k: 0 for k in steps}
    for k, ds in deps.items():
        for d in ds:
            if d in in_deg:
                in_deg[k] += 1

    q = deque(sorted([k for k, v in in_deg.items() if v == 0]))
    waves: list[list[str]] = []
    while q:
        wave = list(q)
        waves.append(wave)
        q = deque()
        for node in wave:
            for k, ds in deps.items():
                if node in ds:
                    in_deg[k] -= 1
                    if in_deg[k] == 0:
                        q.append(k)
        q = deque(sorted(q))

    remaining = sorted([k for k, v in in_deg.items() if v > 0])
    if remaining:
        waves.append(remaining)
    return waves


def mermaid_graph(steps: dict) -> str:
    lines = ["```mermaid", "flowchart TD"]
    for sid, spec in sorted(steps.items()):
        nid = sid.replace("-", "_")
        label = sid.replace("_", " ")
        tier = spec.get("tier", "traffic_tcr")
        cls = TIER_CLASS.get(tier, "traffic")
        lines.append(f'    {nid}["{label}"]:::{cls}')

    for sid, spec in sorted(steps.items()):
        nid = sid.replace("-", "_")
        for dep in spec.get("depends_on") or []:
            if dep not in steps:
                continue
            did = dep.replace("-", "_")
            lines.append(f"    {did} --> {nid}")

    lines.append("```")
    return "\n".join(lines)


def render_dag_md(manifest: dict) -> str:
    steps = manifest["steps"]
    waves = topological_waves(steps)

    by_tier: dict[str, list[str]] = defaultdict(list)
    for sid, spec in steps.items():
        by_tier[spec.get("tier", "?")].append(sid)

    out: list[str] = []
    out += [
        "# Pipeline DAG",
        "",
        "Auto-generated from [`manifest.yaml`](manifest.yaml). Regenerate with:",
        "",
        "```bash",
        "python pipeline/workflow/render_from_manifest.py",
        "```",
        "",
        "See [`AUDIT.md`](AUDIT.md) for lineage and TCR constraints.",
        "",
        "## Tiers",
        "",
    ]
    for tier in TIER_ORDER:
        ids = sorted(by_tier.get(tier, []))
        if ids:
            out.append(f"- **{tier}**: " + ", ".join(f"`{i}`" for i in ids))

    out += ["", "## Graph", "", mermaid_graph(steps), "", "## Topological waves", ""]
    for i, wave in enumerate(waves):
        out.append(f"**Wave {i}**")
        for sid in wave:
            spec = steps[sid]
            lin = spec.get("lineage", "?")
            tcr = " (TCR)" if spec.get("tcr_required") else ""
            out.append(f"- `{sid}` — lineage: {lin}{tcr}")
        out.append("")

    out += [
        "## Step reference",
        "",
        "| Step | Script | Tier | Lineage | Sentinel output |",
        "|------|--------|------|---------|-----------------|",
    ]
    for sid, spec in sorted(steps.items()):
        writes = spec.get("writes") or []
        sentinel = writes[0] if writes else "—"
        out.append(
            f"| `{sid}` | `{spec.get('file','')}` | {spec.get('tier','')} | "
            f"{spec.get('lineage','')} | `{sentinel}` |"
        )

    if manifest.get("data_prep"):
        out += ["", "## Data prep (manual, not in Snakemake `all`)", ""]
        for item in manifest["data_prep"]:
            out.append(f"- `{item.get('id','')}` → `{item.get('file','')}`")

    return "\n".join(out) + "\n"


def main() -> None:
    manifest = load_manifest()
    DAG_MD_PATH.write_text(render_dag_md(manifest))
    print(f"Wrote {DAG_MD_PATH}")


if __name__ == "__main__":
    main()

