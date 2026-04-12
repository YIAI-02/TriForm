#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${SRC_DIR}/main.py" weight-suggest \
  --config "${SRC_DIR}/examples/weight_suggest_test_config.json" \
  --npu_backend fast_mode \
  --pim_fast_mode \
  --debug \
  "$@"

echo "Weight suggest mode done."
