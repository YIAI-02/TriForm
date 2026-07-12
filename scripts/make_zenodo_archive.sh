#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/dist}"
RELEASE_NAME="${RELEASE_NAME:-dops-micro2026-ae}"

cd "${PROJECT_ROOT}"

if [[ ! -f LICENSE ]]; then
  echo "ERROR: root LICENSE is missing; choose an author-approved license before release." >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean; commit the reviewed release state first." >&2
  git status --short >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
COMMIT_SHA="$(git rev-parse HEAD)"
ARCHIVE="${OUT_DIR}/${RELEASE_NAME}-${COMMIT_SHA:0:12}.tar.gz"

git archive --format=tar --prefix="${RELEASE_NAME}/" HEAD | gzip -n > "${ARCHIVE}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
tar -xzf "${ARCHIVE}" -C "${TMP_DIR}"
"${PYTHON_BIN}" "${TMP_DIR}/${RELEASE_NAME}/scripts/check_release.py" \
  "${TMP_DIR}/${RELEASE_NAME}"

(
  cd "${OUT_DIR}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$(basename "${ARCHIVE}")" > SHA256SUMS
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$(basename "${ARCHIVE}")" > SHA256SUMS
  else
    echo "ERROR: neither sha256sum nor shasum is available" >&2
    exit 2
  fi
)

echo "Archive: ${ARCHIVE}"
echo "Commit:  ${COMMIT_SHA}"
echo "Checksums: ${OUT_DIR}/SHA256SUMS"
