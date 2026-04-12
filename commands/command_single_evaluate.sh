#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${SRC_DIR}/main.py" evaluate \
  --config "${SRC_DIR}/examples/evaluate_test_config.json" \
  --npu_backend fast_mode \
  --pim_fast_mode \
  --debug \
  "$@"

echo "Evaluate mode done."
