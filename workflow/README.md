# Snakemake workflow

Orchestrates the GBM trafficking pipeline from [`pipeline/manifest.yaml`](../pipeline/manifest.yaml).

## Setup

```bash
micromamba env create -f environment.yml -n gbm-trafficking
micromamba activate gbm-trafficking
```

Install workflow tools (included in `environment.yml`):

```bash
pip install snakemake pyyaml
```

Regenerate rule stubs after editing the manifest:

```bash
python pipeline/workflow/render_snakemake.py
python pipeline/workflow/render_from_manifest.py
```

## Run

From the repository root:

```bash
# Dry run
snakemake -n -s workflow/Snakefile --profile workflow/profiles/local all_pathway

# Local execution
snakemake -j 4 -s workflow/Snakefile --profile workflow/profiles/local all_traffic_tcr

# SLURM (set slurm_partition in workflow/profiles/slurm/config.yaml)
snakemake -j 100 -s workflow/Snakefile --profile workflow/profiles/slurm all
```

## Targets

| Target | Description |
|--------|-------------|
| `all_pathway` | QC + pathway (T, myeloid, cross) |
| `all_traffic_tcr` | TCR-based trafficking + Bayesian |
| `all_signaling` | LIANA signaling branch |
| `all` | All core steps (excludes `data_prep` scripts) |

Prerequisites: see [`pipeline/AUDIT.md`](../pipeline/AUDIT.md).

**T cells:** `qc_scrublet_doublets` runs first and writes
`GBM_TCR_POS_TCELLS_singlets.h5ad`. All downstream T-cell rules read that file
(`paths.H5AD_TCELLS` in code). Re-run Scrublet after updating the raw celltyping
object.
