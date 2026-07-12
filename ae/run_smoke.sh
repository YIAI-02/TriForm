#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/output/ae_smoke}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "ERROR: output path already exists: ${OUTPUT_ROOT}" >&2
  echo "Set OUTPUT_ROOT to a new path or remove the old smoke-test output." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
echo "[AE] Python: $(${PYTHON_BIN} --version 2>&1)"
echo "[AE] Output: ${OUTPUT_ROOT}"

"${PYTHON_BIN}" src/main.py evaluate \
  --config ae/smoke_config.json \
  --result_dir "${OUTPUT_ROOT}" \
  --npu_backend fast \
  --pim_fast_mode

"${PYTHON_BIN}" ae/verify_smoke.py "${OUTPUT_ROOT}"
