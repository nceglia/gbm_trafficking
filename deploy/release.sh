#!/usr/bin/env bash
# Atomic static-bundle deploy for slvicosspecdat1.
#
# Builds are created on the analysis machine. This script copies only
# deploy/bundle/ to a timestamped release on the VM, then flips current.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/deploy/bundle/"
HOST="${DEPLOY_HOST:-slvicosspecdat1}"
BASE="${DEPLOY_BASE:-/home/ceglian/gbm-viewer}"
RELEASE="${DEPLOY_RELEASE:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN=0
FLIP=1

usage() {
  cat >&2 <<EOF
Usage: $0 [--dry-run] [--no-flip]

Environment:
  DEPLOY_HOST      SSH host (default: slvicosspecdat1)
  DEPLOY_BASE      Remote base directory (default: /home/ceglian/gbm-viewer)
  DEPLOY_RELEASE   Release name (default: UTC timestamp)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --no-flip) FLIP=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -d "${SRC}" ]]; then
  echo "Bundle not found: ${SRC}" >&2
  echo "Build first: python -m viewers.build.all" >&2
  exit 1
fi

if [[ ! -f "${SRC}/index.html" ]]; then
  echo "Bundle index missing: ${SRC}/index.html" >&2
  exit 1
fi

REMOTE_RELEASE="${BASE}/releases/${RELEASE}"
RSYNC_FLAGS=(-az --delete --human-readable --stats --progress)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  RSYNC_FLAGS+=(--dry-run --itemize-changes)
fi

echo "Preflight: ${HOST}"
ssh "${HOST}" "set -e; command -v rsync >/dev/null; command -v python3 >/dev/null; mkdir -p '${BASE}/releases'; df -h '${BASE}' 2>/dev/null || df -h '${HOME}'"

echo "Syncing ${SRC} -> ${HOST}:${REMOTE_RELEASE}/"
rsync "${RSYNC_FLAGS[@]}" "${SRC}" "${HOST}:${REMOTE_RELEASE}/"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Dry run complete. No remote release was activated."
  exit 0
fi

echo "Verifying remote release..."
ssh "${HOST}" "set -e; test -f '${REMOTE_RELEASE}/index.html'; test -f '${REMOTE_RELEASE}/report/index.html'; find '${REMOTE_RELEASE}' -type f | wc -l"

if [[ "${FLIP}" -eq 1 ]]; then
  echo "Activating ${RELEASE}"
  ssh "${HOST}" "set -e; ln -sfn 'releases/${RELEASE}' '${BASE}/current'; ls -ld '${BASE}/current'"
else
  echo "Uploaded ${RELEASE}; current symlink was not changed."
fi

cat <<EOF
Done.

Preview over SSH:
  ssh ${HOST} 'cd ${BASE}/current && python3 -m http.server 8080 --bind 127.0.0.1'
  ssh -L 8080:127.0.0.1:8080 ${HOST}
  open http://127.0.0.1:8080/

Production web-root target:
  ${BASE}/current
EOF
