# Pipeline naming convention

Each driver script in `pipeline/` is named `<domain>_<descriptive_name>.py`. No numeric prefixes — execution order lives in `pipeline/manifest.yaml`, not in filenames.

## Domain prefixes (exactly 7)

| prefix       | meaning                                                       | example                                            |
|--------------|---------------------------------------------------------------|----------------------------------------------------|
| `qc_`        | quality control on input data                                 | `qc_scrublet_doublets.py`, `qc_ambient_audit.py`   |
| `pathway_`   | gene-set / pathway enrichment & scoring                       | `pathway_de_gsea_prerank.py`, `pathway_temporal_scores_tcell.py` |
| `traffic_`   | T-cell trafficking, persistence, temporal trajectories        | `traffic_temporal_trajectories.py`, `traffic_branch_empirics.py` |
| `signaling_` | LIANA / ligand–receptor signaling analyses                    | `signaling_liana_pathways.py`, `signaling_branch_intersection.py` |
| `figure_`    | figure-assembly scripts                                       | `figure_main2_trafficking.py`                      |
| `explorer_`  | interactive HTML / static report builders                     | `explorer_temporal.py`, `explorer_full_report.py`  |
| `_` (leading)| internal helpers used by other drivers (not entry points)     | `_signaling_edges_from_csv.py`                     |

Plus one non-conforming file kept on purpose: `celltyping.py` — function library, no top-level I/O, imported by other scripts.

## Authoritative DAG

- **`pipeline/manifest.yaml`** — machine-readable single source of truth. Each step has `file`, `tier`, `lineage`, `reads`, `writes`, `depends_on`, `resources`. Plus a `data_prep:` block for upstream object/family-map builders.
- **`pipeline/dag.md`** — human-readable Mermaid diagram + topological ordering. Regenerated from manifest via `pipeline/workflow/render_from_manifest.py`.

## Tier vs domain

The filename prefix is the **domain** (loose topic grouping). The manifest adds a **`tier:`** field — the execution wave: `qc → pathway_tcell → pathway_myeloid → traffic → signaling → figure → explorer`. Same script can be domain `traffic_` but tier `figure` if it's a figure dep. Most of the time they match.

## Not on the main DAG

- `pipeline/modules/*.py` — libraries, never drivers
- `pipeline/scripts/*.py` — one-off data prep (object builders, family-map builders), referenced from manifest's `data_prep:` block
- `celltyping.py` at top of `pipeline/` — function library

## Result directories

The `results/<step>/` directory for a driver matches the driver name without the `.py` (e.g., `traffic_bayesian_sankey.py` → `results/traffic_bayesian_sankey/`). All numerically-prefixed dirs (`06c_empirical_Q/`, `07_clonal_trafficking/`, …) have been migrated to this convention as of May 2026.

A handful of `results/<step>/` dirs are still shared by multiple drivers:

| dir | producing drivers |
|-----|-------------------|
| `pathway_de_gsea/` | `pathway_de_gsea_prerank` (primary) + `pathway_coenrichment_graph`, `pathway_proximity_network`, `pathway_tissue_ternary` (extensions) |
| `pathway_temporal_scores/` | `pathway_temporal_scores_tcell` + `pathway_temporal_scores_myeloid` |
| `pathway_cross_lineage_corr/` | `pathway_cross_lineage_corr` + `pathway_cross_lineage_corr_fine` |
| `signaling_liana_pathways/` | `signaling_liana_pathways` + `signaling_intersect_pathways` |
| `traffic_pseudotime_phenotypes/` | `traffic_pseudotime_phenotypes_cd4` + `traffic_pseudotime_phenotypes_cd8` (each writes a lineage subdir) |
| `traffic_migration_rates/` | `traffic_migration_rates` + `traffic_migration_rates_figures` (figures writes to a `figures/` subdir) |

When adding a new step, prefer giving it its own `results/<step>/` dir matching the driver name. Use a shared dir only when downstream consumers explicitly need the outputs co-located.

Two `results/` dirs are kept for historical reasons and aren't part of the main DAG: `figure1/` (from `pipeline/scripts/figure1_clone_network.py`) and `11_cross_lineage_correlations_grouped/` (legacy variant with no current producer).

## Quick reference

| Question                              | Look at                                                   |
|---------------------------------------|-----------------------------------------------------------|
| What scripts run, in what order?      | `pipeline/manifest.yaml`                                  |
| What's the visual graph?              | `pipeline/dag.md`                                         |
| Where does this artifact come from?   | grep `manifest.yaml` for the path under `writes:`         |
| What depends on script X?             | grep `manifest.yaml` for X under any `depends_on:`        |
| Adding a new step                     | edit `manifest.yaml`, add `pipeline/<domain>_<name>.py`, regenerate `dag.md` via the render script |
