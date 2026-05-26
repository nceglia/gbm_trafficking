# Pipeline audit summary

This document summarizes the May 2026 audit of `pipeline/` and `trafficking/`.
The machine-readable DAG lives in [`manifest.yaml`](manifest.yaml); regenerate
[`dag.md`](dag.md) with:

```bash
python pipeline/workflow/render_from_manifest.py
```

## Bayesian and empirical models

The installable package [`trafficking/`](../trafficking/) implements clone-based
trafficking inference:

| Module | Role |
|--------|------|
| `trafficking/data.py` | Extract per-clone phenotype count vectors from `trb`; CD8/CD4 filtering |
| `trafficking/model.py` | Hierarchical Dirichlet transition model (Pyro) |
| `trafficking/inference.py` | `run_inference(adata, t1, t2, lineage)` — **t1/t2 are tissues**; time is handled inside `extract_temporal_transitions` |
| `trafficking/empirical.py` | Clone state vectors, empirical P, Q decomposition (CTMC) |
| `trafficking/dataset.py` | `GBMDataset` dual T/myeloid helper (explorers / Figure 1 only) |

**Drivers:**

- [`traffic_migration_rates.py`](traffic_migration_rates.py) — empirical P/Q → `results/06c_empirical_Q/`
- [`traffic_bayesian_comparison.py`](traffic_bayesian_comparison.py) — compares posterior `T_global` to `P_empirical.csv` (**requires empirical Q first**)
- [`traffic_bayesian_sankey.py`](traffic_bayesian_sankey.py) — all directed tissue edges + Sankey heatmaps

These paths are **not portable to myeloid** without TCR clonotypes. A different
unit of analysis (e.g. patient-level composition dynamics) would be a new model,
not a parameter change.

## Critical dependency: branch empirics

[`traffic_branch_empirics.py`](traffic_branch_empirics.py) uses T-cell `trb`
branches but **must run after**
[`pathway_temporal_scores_myeloid.py`](pathway_temporal_scores_myeloid.py) because
it overlays myeloid composition from
`results/10_temporal_scores/temporal_composition_myeloid.csv`.

## Lineage eligibility (short)

| Category | Steps |
|----------|--------|
| **T cell only** | QC scrublet, T temporal scores, T clone DE/GSEA, most `traffic_*` with `trb`, Figure 2 |
| **Myeloid only** | `pathway_temporal_scores_myeloid`, `pathway_de_gsea_myeloid`, parameterized separability/cosine (`--lineage myeloid`) |
| **Cross-lineage** | `pathway_cross_lineage_corr*`, LIANA + intersect (combined h5ad) |
| **TCR required (N/A for myeloid)** | empirical Q, bayesian, branch dispersion*, clonality, clonal persistence, temporal trajectories |
| **Both lineages (visualization)** | `explorer_clone_network`, `explorer_signaling`, `explorer_full_report` |

Steps marked `tcr_required: true` in `manifest.yaml` must not be scheduled for
myeloid-only runs.

## Excluded one-off scripts (`pipeline/scripts/`)

Not part of Snakemake `all` or tier targets:

| Script | Purpose |
|--------|---------|
| `run_celltyping_workflow.py` | Build `GBM_TCR_POS_TCELLS.h5ad` from 10x |
| `standardize_myeloid_object.py` | Build `MYELOID_GBM.h5ad` |
| `build_combined_object.py` | Build combined T+myeloid h5ad for signaling |
| `build_*_family_map.py` | Pathway family TSV maps under `modules/` |
| `retype_global.py`, `retype_v2.py`, `posthoc_treg_adjustment.py`, `run_genevector_t_cells.py` | Object variants / ad hoc |
| `figure1_clone_network.py` | Figure 1 (separate from main DAG) |

## Prerequisites before `snakemake all`

1. `data/objects/GBM_TCR_POS_TCELLS.h5ad`
2. `data/objects/MYELOID_GBM.h5ad`
3. `data/objects/GBM_TCR_POS_TCELLS_MYELOID_combined.h5ad` (signaling tier)
4. `data/embeddings/X_umap.pkl` (Figure 2)

## Workflow

Optional orchestration: [`../workflow/`](../workflow/) (Snakemake 8, local or SLURM
profiles). Batch drivers should be run with `MPLBACKEND=Agg` (set in profiles).
