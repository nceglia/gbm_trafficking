"""Build all static viewers into deploy/bundle/."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUILDERS = {
    "temporal": REPO_ROOT / "viewers/build/temporal.py",
    "signaling": REPO_ROOT / "viewers/build/signaling.py",
    "clone_network": REPO_ROOT / "viewers/build/clone_network.py",
    "report": REPO_ROOT / "viewers/build/report.py",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build static pipeline viewers into deploy/bundle/"
    )
    parser.add_argument(
        "--only",
        choices=sorted(BUILDERS),
        action="append",
        help="Build only selected viewer(s); default: all",
    )
    args = parser.parse_args(argv)

    selected = args.only or sorted(BUILDERS)
    for name in selected:
        script = BUILDERS[name]
        print(f"\n=== {name} ===")
        runpy.run_path(str(script), run_name="__main__")

    runpy.run_path(str(REPO_ROOT / "viewers/build/landing.py"), run_name="__main__")


if __name__ == "__main__":
    main()
