#!/usr/bin/env python3
"""Run a pipeline driver script with batch-safe environment.

Usage:
    python pipeline/workflow/run_step.py pipeline/qc_scrublet_doublets.py
    python pipeline/workflow/run_step.py pipeline/traffic_pseudotime_phenotypes.py -- --lineage CD8
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from modules.batch import configure_batch_env  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    configure_batch_env()

    script = Path(argv[0])
    if not script.is_absolute():
        script = REPO_ROOT / script
    if not script.is_file():
        print(f"Script not found: {script}", file=sys.stderr)
        sys.exit(1)

    extra = []
    if "--" in argv:
        sep = argv.index("--")
        extra = argv[sep + 1 :]
        argv = argv[:sep]

    old_argv = sys.argv
    sys.argv = [str(script), *extra]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
