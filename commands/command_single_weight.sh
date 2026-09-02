#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-${SRC_DIR}/examples/weight_suggest_test_config.json}"
NPU_BACKEND="${NPU_BACKEND:-}"
PIM_FAST_MODE="${PIM_FAST_MODE:-0}"

# Examples:
#   bash commands/command_single_weight.sh
#   CONFIG=./src/examples/weight_suggest_test_config.json NPU_BACKEND=lut bash commands/command_single_weight.sh
# Add quantization/sparsity settings inside the CONFIG JSON under the `optimizations` block.

cd "${PROJECT_ROOT}"

EXTRA_ARGS=()
if [[ -n "${NPU_BACKEND}" ]]; then
  EXTRA_ARGS+=(--npu_backend "${NPU_BACKEND}")
fi
if [[ "${PIM_FAST_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "Error: weight-suggest does not support PIM fast mode. Set PIM_FAST_MODE=0." >&2
  exit 2
fi
EXTRA_ARGS+=(--no-pim_fast_mode)

"${PYTHON_BIN}" "${SRC_DIR}/main.py" weight-suggest \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}" \
  --debug \
  "$@"

echo "Weight suggest mode done."
