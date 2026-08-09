#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DOPS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${DOPS_PYTHON:-python3}"
MODE="${1:-${DOPS_PRIOR_MODE:-fast}}"
CASE_INDEX="${2:-${SLURM_ARRAY_TASK_ID:-0}}"
OUTPUT_ROOT="${DOPS_PRIOR_OUTPUT_ROOT:-${PROJECT_ROOT}/output/hetinfer_gpu_proxy_prior_grid}"

FAST_CONFIG="${DOPS_PRIOR_FAST_CONFIG:-${PROJECT_ROOT}/configs/hetinfer_gpu_proxy/evaluate_qwen1p8b_gpu_proxy_pim_fast.json}"
TRACE_CONFIG="${DOPS_PRIOR_TRACE_CONFIG:-${PROJECT_ROOT}/configs/hetinfer_gpu_proxy/evaluate_qwen1p8b_gpu_proxy_pim_trace.json}"

if [[ ! -f "${PROJECT_ROOT}/src/main.py" ]]; then
  echo "[FATAL] DOPS_ROOT does not point to a DOPS checkout: ${PROJECT_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[FATAL] DOPS_PYTHON is neither an executable path nor a command: ${PYTHON_BIN}" >&2
  exit 2
fi
cd "${PROJECT_ROOT}"

case "${MODE}" in
  fast)
    # Full interpolation grid: batch={1,4,8}, prefill={128,512,2048}, decode={64,256}.
    CASES=(
      "1 128 64" "1 128 256" "1 512 64" "1 512 256" "1 2048 64" "1 2048 256"
      "4 128 64" "4 128 256" "4 512 64" "4 512 256" "4 2048 64" "4 2048 256"
      "8 128 64" "8 128 256" "8 512 64" "8 512 256" "8 2048 64" "8 2048 256"
    )
    CONFIG="${FAST_CONFIG}"
    NPU_BACKEND="${DOPS_PRIOR_FAST_NPU_BACKEND:-llmcompass}"
    PIM_MODE_ARG="--pim_fast_mode"
    COST_MODEL_LABEL="gpu_${NPU_BACKEND}+pim_analytical_fast"
    ;;
  ramulator-scaled)
    # Three anchors use a real unit trace and Ramulator2, then scale repeated
    # batch/prefill work. This is stronger than analytical fast-mode but is
    # still an approximation, not the strict unrolled validation below.
    CASES=("1 128 16" "4 512 16" "8 2048 16")
    CONFIG="${TRACE_CONFIG}"
    NPU_BACKEND="${DOPS_PRIOR_TRACE_NPU_BACKEND:-llmcompass}"
    PIM_MODE_ARG="--no-pim_fast_mode"
    COST_MODEL_LABEL="gpu_${NPU_BACKEND}+pim_cent_ramulator_scaled"
    export PIM_TRACE_STRICT=1
    export PIM_TRACE_SCALE_REPEATS=1
    ;;
  ramulator-strict)
    # Deliberately small: full prefill/batch trace expansion can be expensive.
    CASES=("1 64 4")
    CONFIG="${TRACE_CONFIG}"
    NPU_BACKEND="${DOPS_PRIOR_TRACE_NPU_BACKEND:-llmcompass}"
    PIM_MODE_ARG="--no-pim_fast_mode"
    COST_MODEL_LABEL="gpu_${NPU_BACKEND}+pim_cent_ramulator_unrolled"
    export PIM_TRACE_STRICT=1
    export PIM_TRACE_SCALE_REPEATS=0
    ;;
  *)
    echo "[FATAL] unknown mode '${MODE}'; expected fast, ramulator-scaled, or ramulator-strict" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CONFIG}" ]]; then
  echo "[FATAL] DOPS prior config does not exist: ${CONFIG}" >&2
  exit 2
fi
if [[ "${NPU_BACKEND}" == "llmcompass" ]] && [[ ! -d "${PROJECT_ROOT}/submodules/LLMCompass" ]]; then
  echo "[FATAL] LLMCompass is missing at ${PROJECT_ROOT}/submodules/LLMCompass" >&2
  exit 2
fi

if ! [[ "${CASE_INDEX}" =~ ^[0-9]+$ ]] || (( CASE_INDEX >= ${#CASES[@]} )); then
  echo "[FATAL] case index '${CASE_INDEX}' is outside [0,$((${#CASES[@]} - 1))] for mode=${MODE}" >&2
  exit 2
fi

if [[ "${MODE}" != "fast" ]]; then
  if [[ ! -d "${PROJECT_ROOT}/submodules/CENT/cent_simulation" ]]; then
    echo "[FATAL] CENT trace generator is missing under ${PROJECT_ROOT}/submodules/CENT" >&2
    exit 2
  fi
  if [[ -z "${RAMULATOR2_BIN:-}" ]]; then
    echo "[FATAL] RAMULATOR2_BIN must name the compiled Ramulator2 executable for ${MODE}" >&2
    exit 2
  fi
  if [[ "${RAMULATOR2_BIN}" == */* ]]; then
    if [[ ! -x "${RAMULATOR2_BIN}" ]]; then
      echo "[FATAL] RAMULATOR2_BIN is not executable: ${RAMULATOR2_BIN}" >&2
      exit 2
    fi
  elif ! command -v "${RAMULATOR2_BIN}" >/dev/null 2>&1; then
    echo "[FATAL] RAMULATOR2_BIN is not on PATH: ${RAMULATOR2_BIN}" >&2
    exit 2
  fi
fi

read -r BATCH PREFILL_LEN DECODE_LEN <<< "${CASES[CASE_INDEX]}"
CASE_TAG="b${BATCH}_p${PREFILL_LEN}_d${DECODE_LEN}"
CASE_ROOT="${OUTPUT_ROOT}/${MODE}/${CASE_TAG}"
RESULT_DIR="${CASE_ROOT}/results"
PRIOR_PATH="${CASE_ROOT}/prior.json"
mkdir -p "${CASE_ROOT}"

# The pickle cache has no inter-process file lock, so each array task gets a
# private file. This avoids corrupting a shared cache during concurrent jobs.
export PIM_LATENCY_CACHE_FILE="${CASE_ROOT}/pim_latency_cache.pkl"

echo "[DOPS-PRIOR] mode=${MODE} case_index=${CASE_INDEX} workload=${CASE_TAG}"
echo "[DOPS-PRIOR] cost_model=${COST_MODEL_LABEL}"
echo "[DOPS-PRIOR] output=${PRIOR_PATH}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/src/main.py" evaluate \
  --config "${CONFIG}" \
  --batch "${BATCH}" \
  --prefill_len "${PREFILL_LEN}" \
  --decode_len "${DECODE_LEN}" \
  --result_dir "${RESULT_DIR}" \
  --algo Bifocal \
  --npu_backend "${NPU_BACKEND}" \
  "${PIM_MODE_ARG}" \
  --hetinfer-prior-out "${PRIOR_PATH}"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_prior_artifact.py" \
  "${PRIOR_PATH}" \
  --expected-batch "${BATCH}" \
  --expected-prefill "${PREFILL_LEN}" \
  --expected-decode "${DECODE_LEN}"

echo "[DOPS-PRIOR] completed ${PRIOR_PATH}"
