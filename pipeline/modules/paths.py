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

# ── AnnData objects (T cell) ──────────────────────────────────────────
# Raw object from celltyping (input to Scrublet only).
H5AD_TCELLS_RAW          = OBJECTS_DIR / "GBM_TCR_POS_TCELLS.h5ad"
# Doublet-filtered object from qc_scrublet_doublets.py.
H5AD_TCELLS_SINGLETS     = OBJECTS_DIR / "GBM_TCR_POS_TCELLS_singlets.h5ad"
# Default T-cell AnnData for all analysis drivers (post-Scrublet).
H5AD_TCELLS              = H5AD_TCELLS_SINGLETS
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

# ── Cellranger per-sample outputs (78 samples) ────────────────────────
# These are CellRanger *filtered* outputs (filtered_feature_bc_matrix.h5
# format — barcodes × features, cell-called by CellRanger). The companion
# CSVs are the per-barcode TCR contigs (`<sample>.csv`) and per-clonotype
# table (`<sample>_clonotypes.csv`).
#
# Raw (unfiltered) matrices are NOT in the repo — anything that needs
# the empty-droplet ambient profile (e.g. CellBender) would have to
# source those upstream.
CELLRANGER_DIR = DATA_DIR / "cellranger_output"


def cellranger_h5(sample: str) -> Path:
    """Filtered feature×barcode matrix for one sample."""
    return CELLRANGER_DIR / f"{sample}.h5"


def cellranger_contigs(sample: str) -> Path:
    """Per-barcode VDJ contigs CSV for one sample."""
    return CELLRANGER_DIR / f"{sample}.csv"


def cellranger_clonotypes(sample: str) -> Path:
    """Per-clonotype consensus CSV for one sample."""
    return CELLRANGER_DIR / f"{sample}_clonotypes.csv"


def cellranger_samples() -> list[str]:
    """Enumerate every CellRanger sample present on disk."""
    return sorted(p.stem for p in CELLRANGER_DIR.glob("*.h5"))


# ── Results root + per-step output dirs ───────────────────────────────
# Scripts should write to RESULTS_DIR / "<step_dir>" and lookups should
# go through these constants so a future rename touches one file, not 20.
RESULTS_DIR = REPO_ROOT / "results"

# Every result dir follows the driver-name convention (see pipeline/NAMING.md).
# Where a constant's spelling diverges from the driver name (e.g.,
# EMPIRICAL_Q_DIR ← traffic_migration_rates), the constant is kept for
# backward compatibility but points at the new dir.

# QC / preprocessing
QC_DOUBLETS_DIR              = RESULTS_DIR / "qc_scrublet_doublets"

# Pathway / GSEA
PSEUDOBULK_DE_GSEA_DIR       = RESULTS_DIR / "pathway_de_gsea"
PSEUDOBULK_DE_GSEA_MYELOID_DIR  = RESULTS_DIR / "pathway_de_gsea_myeloid"
TEMPORAL_SCORES_DIR          = RESULTS_DIR / "pathway_temporal_scores"
CROSS_LINEAGE_CORR_DIR       = RESULTS_DIR / "pathway_cross_lineage_corr"
CROSS_LINEAGE_CORR_GROUPED_DIR = RESULTS_DIR / "11_cross_lineage_correlations_grouped"  # orphan, no producer

# Trafficking
TISSUE_SEPARABILITY_DIR      = RESULTS_DIR / "traffic_tissue_separability"
TISSUE_SEPARABILITY_MYELOID_DIR = RESULTS_DIR / "traffic_tissue_separability_myeloid"
TRANSCRIPTOME_SIMILARITY_DIR = RESULTS_DIR / "traffic_transcriptome_cosine"
TRANSCRIPTOME_SIMILARITY_MYELOID_DIR = RESULTS_DIR / "traffic_transcriptome_cosine_myeloid"
BRANCH_EMPIRICS_DIR          = RESULTS_DIR / "traffic_branch_empirics"
EMPIRICAL_Q_DIR              = RESULTS_DIR / "traffic_migration_rates"
EMPIRICAL_Q_PER_TIMEPOINT_DIR = RESULTS_DIR / "traffic_migration_rates_per_tp"
EMPIRICAL_Q_FIGURES_DIR      = EMPIRICAL_Q_DIR / "figures"
BAYESIAN_COMPARISON_DIR      = RESULTS_DIR / "traffic_bayesian_comparison"
BAYESIAN_SANKEY_DIR          = RESULTS_DIR / "traffic_bayesian_sankey"
EMPIRICAL_SANKEY_DIR         = RESULTS_DIR / "traffic_empirical_sankey"
TEMPORAL_TRAJECTORIES_DIR    = RESULTS_DIR / "traffic_temporal_trajectories"
PSEUDOTIME_PHENOTYPES_DIR    = RESULTS_DIR / "traffic_pseudotime_phenotypes"
PHENOTYPE_DEGS_DIR           = RESULTS_DIR / "traffic_phenotype_degs"
CLONAL_TRAFFICKING_DIR       = RESULTS_DIR / "traffic_clonal_persistence"
CLONALITY_DIR                = RESULTS_DIR / "traffic_clonality"
BRANCH_DISPERSION_DIR        = RESULTS_DIR / "traffic_branch_dispersion"
BRANCH_DISPERSION_JSD_DIR    = RESULTS_DIR / "traffic_branch_dispersion_jsd"
BRANCH_DISPERSION_SWITCHING_DIR = RESULTS_DIR / "traffic_branch_dispersion_switching"
BRANCH_DISPERSION_TEMPORAL_DIR  = RESULTS_DIR / "traffic_branch_dispersion_temporal"
TRAFFIC_ARCHETYPES_DIR       = RESULTS_DIR / "traffic_archetypes"
TRAFFIC_ARCHETYPE_GRAPHS_DIR = RESULTS_DIR / "traffic_archetype_graphs"

# Signaling
BRANCH_SIGNALING_DIR         = RESULTS_DIR / "signaling_branch_intersection"
LIANA_SIGNALING_DIR          = RESULTS_DIR / "signaling_liana_pathways"

# Figures / explorers
FIGURE1_DIR                  = RESULTS_DIR / "figure1"  # orphan, written by scripts/figure1_clone_network.py
FIGURE2_DIR                  = RESULTS_DIR / "figure_main2_trafficking"
CLONE_NETWORK_EXPLORER_DIR   = RESULTS_DIR / "explorer_clone_network"
SIGNALING_EXPLORER_DIR       = RESULTS_DIR / "explorer_signaling"

# Repo-root HTML / report artifacts (legacy — prefer deploy/bundle/)
VIEWER_BUNDLE_DIR           = REPO_ROOT / "deploy" / "bundle"
VIEWER_TEMPORAL_HTML        = VIEWER_BUNDLE_DIR / "temporal.html"
VIEWER_SIGNALING_HTML       = VIEWER_BUNDLE_DIR / "signaling.html"
VIEWER_CLONE_NETWORK_HTML   = VIEWER_BUNDLE_DIR / "clone_network.html"
VIEWER_REPORT_DIR           = VIEWER_BUNDLE_DIR / "report"
VIEWER_LANDING_HTML         = VIEWER_BUNDLE_DIR / "index.html"

GBM_TEMPORAL_EXPLORER_HTML  = VIEWER_TEMPORAL_HTML
GBM_SIGNALING_EXPLORER_HTML = VIEWER_SIGNALING_HTML
GBM_REPORT_DIR              = VIEWER_REPORT_DIR


def ensure(*paths: Path) -> None:
    """Create each directory if absent. Convenience for script headers."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
