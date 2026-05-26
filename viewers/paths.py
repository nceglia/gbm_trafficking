"""Path constants for viewer builds and deploy bundle outputs."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

_PIPELINE = REPO_ROOT / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from modules.paths import (  # noqa: E402
    BRANCH_EMPIRICS_DIR,
    CLONE_NETWORK_EXPLORER_DIR,
    CROSS_LINEAGE_CORR_DIR,
    H5AD_MYELOID,
    H5AD_TCELLS,
    LIANA_SIGNALING_DIR,
    RESULTS_DIR,
    SIGNALING_EXPLORER_DIR,
    TEMPORAL_SCORES_DIR,
)

BUNDLE_DIR = REPO_ROOT / "deploy" / "bundle"
D3_CACHE = REPO_ROOT / "build" / "d3.v7.min.js"

TEMPORAL_HTML = BUNDLE_DIR / "temporal.html"
SIGNALING_HTML = BUNDLE_DIR / "signaling.html"
CLONE_NETWORK_HTML = BUNDLE_DIR / "clone_network.html"
REPORT_DIR = BUNDLE_DIR / "report"
LANDING_HTML = BUNDLE_DIR / "index.html"


def ensure_bundle() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
