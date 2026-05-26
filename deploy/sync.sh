#!/usr/bin/env bash
# Sync the static viewer bundle to a remote VM (no pipeline data).
#
# Usage:
#   ./deploy/sync.sh user@host:/var/www/gbm-viewer
#   DEPLOY_DEST=user@host:/path ./deploy/sync.sh
#
# Only deploy/bundle/ is copied — typically ~65–200 MB depending on which
# explorers and report sections were built on the analysis machine.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/deploy/bundle/"
DEST="${1:-${DEPLOY_DEST:-}}"

if [[ -z "${DEST}" ]]; then
  echo "Usage: $0 user@host:/remote/path" >&2
  echo "   or: DEPLOY_DEST=user@host:/path $0" >&2
  exit 1
fi

if [[ ! -d "${SRC}" ]]; then
  echo "Bundle not found: ${SRC}" >&2
  echo "Build first: python -m viewers.build.all" >&2
  exit 1
fi

echo "Syncing ${SRC} -> ${DEST}"
rsync -avz --delete --progress "${SRC}" "${DEST}"
echo "Done. Serve ${DEST} with nginx (see deploy/nginx.conf) or:"
echo "  ssh ... 'python3 -m http.server 8080 -d ${DEST}'"
