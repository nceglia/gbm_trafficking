"""Centralized filesystem path config for the GBM trafficking pipeline.

Every pipeline driver should import its inputs and output dirs from here
instead of redefining `REPO_ROOT / "data" / ...` locally. The constants
below resolve from this module's location, so they work regardless of
the cwd the script is invoked from.

Existing scripts may still define local `DATA_PATH`/`OUT_DIR` constants;
those are being migrated incrementally. New scripts should import from
here from day one.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Top-level data tree ───────────────────────────────────────────────
DATA_DIR        = REPO_ROOT / "data"
OBJECTS_DIR     = DATA_DIR / "objects"
EMBEDDINGS_DIR  = DATA_DIR / "embeddings"

# ── AnnData objects (canonical T cell file + variants) ────────────────
# Original full T cell object — everything to date reads from this.
H5AD_TCELLS              = OBJECTS_DIR / "GBM_TCR_POS_TCELLS.h5ad"
# Doublet-filtered version produced by scrublet_doublet_qc.py.
# Scripts adopt this opt-in, not all at once.
H5AD_TCELLS_SINGLETS     = OBJECTS_DIR / "GBM_TCR_POS_TCELLS_singlets.h5ad"
# Combined T + myeloid object used by signaling scripts.
H5AD_TCELLS_MYELOID      = OBJECTS_DIR / "GBM_TCR_POS_TCELLS_MYELOID_combined.h5ad"
# Retyped / all-genes / GeneVector variants.
H5AD_TCELLS_RETYPED      = OBJECTS_DIR / "GBM_TCR_POS_TCELLS_retyped.h5ad"
H5AD_TCELLS_ALLGENES     = OBJECTS_DIR / "GBM_TCR_POS_TCELLS_allgenes.h5ad"
H5AD_TCELLS_GV           = OBJECTS_DIR / "GBM_TCR_POS_TCELLS_GV.h5ad"
H5AD_TCR_DOUBLETS_LEGACY = OBJECTS_DIR / "GBM_TCR_DOUBLETS.h5ad"
# Standalone myeloid AnnData (parallel to T cell file).
H5AD_MYELOID             = OBJECTS_DIR / "MYELOID_GBM.h5ad"

# ── Other embeddings / artifacts ──────────────────────────────────────
UMAP_PKL = EMBEDDINGS_DIR / "X_umap.pkl"

# ── Results root + per-step output dirs ───────────────────────────────
# Scripts should write to RESULTS_DIR / "<step_dir>" and lookups should
# go through these constants so a future rename touches one file, not 20.
RESULTS_DIR = REPO_ROOT / "results"

# QC / preprocessing
QC_DOUBLETS_DIR              = RESULTS_DIR / "qc_doublets"

# Analysis steps (existing numeric dirs preserved for now; will be
# renamed alongside the pipeline-driver rename in a later refactor).
TISSUE_SEPARABILITY_DIR      = RESULTS_DIR / "03_tissue_separability"
PSEUDOBULK_DE_GSEA_DIR       = RESULTS_DIR / "04_pseudobulk_de_gsea"
BRANCH_EMPIRICS_DIR          = RESULTS_DIR / "06_branch_empirics"
TRANSCRIPTOME_SIMILARITY_DIR = RESULTS_DIR / "transcriptome_similarity"
BRANCH_SIGNALING_DIR         = RESULTS_DIR / "06b_branch_signaling"
EMPIRICAL_Q_DIR              = RESULTS_DIR / "06c_empirical_Q"
EMPIRICAL_Q_PER_TIMEPOINT_DIR = RESULTS_DIR / "06d_empirical_Q_per_timepoint"
BAYESIAN_COMPARISON_DIR      = RESULTS_DIR / "06f_bayesian_comparison"
BAYESIAN_SANKEY_DIR          = RESULTS_DIR / "06g_bayesian_sankey"
FIGURE2_DIR                  = RESULTS_DIR / "07_figure2"
TEMPORAL_TRAJECTORIES_DIR    = RESULTS_DIR / "09_temporal"
TEMPORAL_SCORES_DIR          = RESULTS_DIR / "10_temporal_scores"
CROSS_LINEAGE_CORR_DIR       = RESULTS_DIR / "11_cross_lineage_correlations"
CROSS_LINEAGE_CORR_GROUPED_DIR = RESULTS_DIR / "11_cross_lineage_correlations_grouped"
LIANA_SIGNALING_DIR          = RESULTS_DIR / "12_liana_signaling"
PSEUDOTIME_PHENOTYPES_DIR    = RESULTS_DIR / "13_pseudotime_phenotypes"
PHENOTYPE_DEGS_DIR           = RESULTS_DIR / "14_phenotype_degs"

# Standalone analyses (non-numbered)
BRANCH_DISPERSION_DIR           = RESULTS_DIR / "branch_dispersion"
BRANCH_DISPERSION_JSD_DIR       = RESULTS_DIR / "branch_dispersion_jsd"
BRANCH_DISPERSION_SWITCHING_DIR = RESULTS_DIR / "branch_dispersion_switching"
BRANCH_DISPERSION_TEMPORAL_DIR  = RESULTS_DIR / "branch_dispersion_temporal"
CLONALITY_DIR                   = RESULTS_DIR / "clonality"
CLONE_NETWORK_EXPLORER_DIR      = RESULTS_DIR / "clone_network_explorer"
SIGNALING_EXPLORER_DIR          = RESULTS_DIR / "signaling_explorer"
FIGURE1_DIR                     = RESULTS_DIR / "figure1"


def ensure(*paths: Path) -> None:
    """Create each directory if absent. Convenience for script headers."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
