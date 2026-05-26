#!/usr/bin/env python3
"""Emit workflow/rules/generated.smk from pipeline/manifest.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "pipeline" / "manifest.yaml"
OUT = REPO_ROOT / "workflow" / "rules" / "generated.smk"


def q(s: str) -> str:
    return s.replace("\\", "/")


def main() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text())
    steps = manifest["steps"]
    lines = [
        "# Auto-generated — do not edit. Regenerate:",
        "#   python pipeline/workflow/render_snakemake.py",
        "",
    ]

    for sid, spec in sorted(steps.items()):
        writes = spec.get("writes") or []
        if not writes:
            continue
        ins = [q(p) for p in (spec.get("reads") or [])]
        outs = [q(p) for p in writes]  # all listed outputs are Snakemake targets
        script = q(spec["file"]) if "/" in spec["file"] else q(f"pipeline/{spec['file']}")
        cli = spec.get("cli_extra") or []
        cli_str = " ".join(cli)
        res = spec.get("resources") or {}
        mem = res.get("mem_mb", 16000)
        runtime = res.get("runtime", 120)
        gres = res.get("gres", "")

        lines.append(f"rule {sid}:")
        if ins:
            lines.append("    input:")
            for i in ins:
                lines.append(f'        "{i}",')
        lines.append("    output:")
        for o in outs:
            lines.append(f'        "{o}",')
        lines.append("    resources:")
        lines.append(f"        mem_mb={mem},")
        lines.append(f"        runtime={runtime},")
        if gres:
            lines.append(f'        slurm_extra="#SBATCH --gres={gres}",')
        lines.append("    params:")
        lines.append(f'        script="{script}",')
        lines.append(f'        cli="{cli_str}",')
        shell = "python pipeline/workflow/run_step.py {params.script}"
        if cli:
            shell += " -- {params.cli}"
        lines.append("    shell:")
        lines.append(f'        "{shell}"')
        lines.append("")

    targets = manifest.get("targets", {})
    for tname, members in targets.items():
        if tname == "all":
            continue
        if isinstance(members, list):
            outs = []
            for m in members:
                for w in steps.get(m, {}).get("writes") or []:
                    outs.append(f'"{q(w)}"')
            lines.append(f"rule {tname}:")
            lines.append("    input:")
            if outs:
                for o in outs:
                    lines.append(f"        {o},")
            else:
                lines.append("        []")
            lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
