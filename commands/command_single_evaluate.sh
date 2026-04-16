#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${SRC_DIR}/examples/evaluate_test_config.json}"
NPU_BACKEND="${NPU_BACKEND:-fast_mode}"
PIM_FAST_MODE="${PIM_FAST_MODE:-0}"

# Examples:
#   CONFIG=./src/examples/evaluate_quant_sparse_config.json bash commands/command_single_evaluate.sh
#   CONFIG=./src/examples/evaluate_test_config.json NPU_BACKEND=llmcompass bash commands/command_single_evaluate.sh

cd "${PROJECT_ROOT}"

PIM_ARGS=()
if [[ "${PIM_FAST_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  PIM_ARGS+=(--pim_fast_mode)
fi

"${PYTHON_BIN}" "${SRC_DIR}/main.py" evaluate \
  --config "${CONFIG}" \
  --npu_backend "${NPU_BACKEND}" \
  "${PIM_ARGS[@]}" \
  --debug \
  "$@"

echo "Evaluate mode done."
