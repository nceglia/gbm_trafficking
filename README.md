# gbm_trafficking

## Celltyping CLI

`pipeline/celltyping.py` is now a reusable module (not a standalone CLI script).  
Use the workflow CLI entrypoint instead:

```bash
python /Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/scripts/run_celltyping_workflow.py \
  --data-dir /Users/ceglian/Codebase/GitHub/gbm_trafficking/data/btc_gbm_gex_vdj \
  --table-sig /Users/ceglian/Downloads/41586_2025_9989_MOESM10_ESM.xlsx \
  --table-tf /Users/ceglian/Downloads/41586_2025_9989_MOESM6_ESM.xlsx \
  --output-dir /Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/outputs/celltyping \
  --output-h5ad GBM_TCR_POS_TCELLS.h5ad
```

### CLI arguments

- `--data-dir`: directory containing paired `*.h5`, `*.csv`, and `*_clonotypes.csv` files
- `--table-sig`: marker/signature spreadsheet used by `phenotype_tcells`
- `--table-tf`: TF table path kept for workflow parity
- `--output-dir`: output directory for figures and artifacts
- `--output-h5ad`: final annotated h5ad filename (written inside `--output-dir`)

## Interactive Notebook

For interactive, cell-by-cell execution use:

- `/Users/ceglian/Codebase/GitHub/gbm_trafficking/notebooks/celltyping_interactive_workflow.ipynb`

## Current Module Structure

### Core celltyping module

- `/Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/celltyping.py`
  - data loading helper: `load_trb`
  - hierarchical phenotyping functions: `classify_recursive`, `phenotype_tcells`, gating/correction helpers
  - preprocessing helpers: `annotate`, `gene_entropy`, `remove_meaningless_genes`, `run_harmony_workflow`

### Shared modules

- `/Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/modules/celltyping_io.py`
  - directory-level IO loaders (`load_directory_scirpy`, `load_directory_manual`)
- `/Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/modules/celltyping_validation.py`
  - expression extraction + validation plotting (`run_validation_plots`, etc.)
- `/Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/modules/celltyping_clonality.py`
  - clonality summaries, correlation analysis, and clonality plotting
- `/Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/modules/celltyping_geometry.py`
  - simplex geometry utilities and clone transition plotting
- `/Users/ceglian/Codebase/GitHub/gbm_trafficking/pipeline/modules/clone_helpers.py`
  - lineage/label parsing helpers used across modules
