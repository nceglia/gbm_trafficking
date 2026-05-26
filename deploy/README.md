# Deploying pipeline viewers on a VM

The analysis pipeline runs on a compute machine with h5ad objects and full
`results/`. Collaborators browse **pre-built static files** in `deploy/bundle/`
on a low-power VM — no AnnData, no Snakemake, no GPU.

## 1. Build on the analysis machine

After the relevant pipeline steps finish:

```bash
# All viewers + landing page
python -m viewers.build.all

# Or individual builds
python viewers/build/temporal.py
python viewers/build/signaling.py
python viewers/build/clone_network.py
python viewers/build/report.py
python viewers/build/landing.py
```

Via Snakemake (includes `viewer_landing`):

```bash
snakemake deploy/bundle/index.html
```

Outputs land in `deploy/bundle/`:

| File | Description |
|------|-------------|
| `index.html` | Landing page linking all views |
| `temporal.html` | Temporal phenotype/pathway/gene explorer |
| `signaling.html` | L-R signaling explorer |
| `clone_network.html` | Clone sharing network (D3) |
| `report/` | Manifest-driven tables + figures |

Typical bundle size: **~65 MB** (three HTML explorers) plus whatever the
report section copies from `results/`.

## 2. Sync to the VM

```bash
chmod +x deploy/sync.sh
./deploy/sync.sh user@vm.example.org:/var/www/gbm-viewer
```

Only `deploy/bundle/` is transferred — not `data/`, not `results/`.

## 3. Serve on the VM

**Quick test:**

```bash
python3 -m http.server 8080 -d /var/www/gbm-viewer
```

**Production (nginx):** see `deploy/nginx.conf`. The report viewer uses
`fetch()` and **must** be served over HTTP (not `file://`).

## Layout

```
viewers/build/     # build scripts (run on compute machine)
deploy/bundle/     # static artifacts (sync to VM)
deploy/sync.sh     # rsync helper
deploy/nginx.conf  # example nginx site
```

Legacy shims at `pipeline/explorer_*.py` delegate to `viewers/build/` for
backward compatibility.
