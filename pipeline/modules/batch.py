"""Batch execution helpers for Snakemake and cluster runs."""
from __future__ import annotations

import os


def configure_batch_env() -> None:
    """Non-interactive matplotlib and sensible defaults for workflow runs."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
