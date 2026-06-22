# Deploying pipeline viewers on `slvicosspecdat1`

The analysis pipeline runs on the analysis machine with AnnData objects and
full `results/`. Collaborators browse pre-built static files on the VM. The VM
does not generate outputs and should not receive raw `data/`, full `results/`,
`.h5ad`, `.pkl`, `.npy`, or `.npz` files.

## Build locally

After the relevant pipeline steps finish:

```bash
python -m viewers.build.all
```

Outputs land in `deploy/bundle/`:

| Path | Description |
|------|-------------|
| `index.html` | Landing page with release metadata |
| `report/` | Narrative report, methods, artifacts, and checksums |
| `temporal.html` | Temporal phenotype/pathway/gene explorer |
| `signaling.html` | Ligand-receptor signaling explorer |
| `clone_network.html` | Clone-sharing network explorer |
| `release.json` | Build metadata for deployment |

## Deploy atomically

`slvicosspecdat1` currently has `rsync` and `python3`, but no nginx/Apache,
no `/var/www`, and no passwordless sudo. Releases therefore go under the
user-writable home directory by default.

```bash
# Preview what would change
./deploy/release.sh --dry-run

# Upload and activate a timestamped release
./deploy/release.sh
```

Default remote layout:

```text
/home/ceglian/gbm-viewer/
  current -> releases/<timestamp>
  releases/
    <timestamp>/
      index.html
      report/
      temporal.html
      signaling.html
      clone_network.html
```

Rollback is a symlink flip:

```bash
ssh slvicosspecdat1 'cd /home/ceglian/gbm-viewer && ln -sfn releases/<old-release> current'
```

## Preview without a web server

```bash
ssh slvicosspecdat1 'cd /home/ceglian/gbm-viewer/current && python3 -m http.server 8080 --bind 127.0.0.1'
ssh -L 8080:127.0.0.1:8080 slvicosspecdat1
open http://127.0.0.1:8080/
```

## Production service

Ask the VM/admin owner to install nginx or Apache and serve:

```text
/home/ceglian/gbm-viewer/current
```

The site is internal/VPN-only, but light basic auth is useful for collaborator
comfort. On RHEL, `htpasswd` usually comes from `httpd-tools`; if it is not
installed, generate the password file elsewhere and copy it to the VM outside
the repo.

An nginx example lives in `deploy/nginx.conf`.
