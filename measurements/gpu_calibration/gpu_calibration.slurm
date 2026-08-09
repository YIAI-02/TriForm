#!/usr/bin/env bash
#SBATCH -J dops_gpu_calibration
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o dops_gpu_calibration.%j.out

set -euo pipefail

# No partition is embedded here. Submit with:
#   sbatch --partition=<actual GPU partition> --export=ALL .../gpu_calibration.slurm
PROJECT_ROOT="${DOPS_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
# Slurm executes a spool copy of this script, so BASH_SOURCE points under
# /var/spool rather than at the checkout.  Derive every helper from DOPS_ROOT.
SCRIPT_DIR="${PROJECT_ROOT}/measurements/gpu_calibration"
PYTHON_BIN="${DOPS_PYTHON:-python3}"
BENCHMARK_CONFIG="${DOPS_GPU_BENCHMARK_CONFIG:-${SCRIPT_DIR}/benchmark_qwen1p8b_fp16.json}"
BASE_HARDWARE="${DOPS_GPU_BASE_HARDWARE:-${PROJECT_ROOT}/configs/hetinfer_gpu_proxy/hardware_gpu0_pim0_a100_llmcompass_proxy.json}"
BASE_CONFIG="${DOPS_GPU_BASE_CONFIG:-${PROJECT_ROOT}/configs/hetinfer_gpu_proxy/evaluate_qwen1p8b_gpu_proxy_pim_fast.json}"
OUTPUT_ROOT="${DOPS_GPU_CALIBRATION_OUTPUT:-${PROJECT_ROOT}/output/gpu_calibration/${SLURM_JOB_ID:-manual}}"

if [[ ! -f "${PROJECT_ROOT}/src/main.py" ]]; then
  echo "[FATAL] DOPS_ROOT does not point to a DOPS checkout: ${PROJECT_ROOT}" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[FATAL] CUDA_VISIBLE_DEVICES is empty; submit this script to a GPU partition with a GPU allocation" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
RAW_JSON="${OUTPUT_ROOT}/gpu_microbench_raw.json"
FIT_JSON="${OUTPUT_ROOT}/gpu_calibration_fit.json"
RUNTIME_JSON="${OUTPUT_ROOT}/gpu0_runtime_model.json"
HARDWARE_JSON="${OUTPUT_ROOT}/hardware_gpu0_pim0_calibrated.json"
EVALUATE_JSON="${OUTPUT_ROOT}/evaluate_qwen1p8b_gpu_calibrated.json"

EXPECT_ARGS=()
if [[ -n "${DOPS_EXPECT_GPU_REGEX:-}" ]]; then
  EXPECT_ARGS+=(--expect-device-regex "${DOPS_EXPECT_GPU_REGEX}")
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/gpu_microbench.py" \
  --config "${BENCHMARK_CONFIG}" \
  --output "${RAW_JSON}" \
  "${EXPECT_ARGS[@]}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/fit_gpu_calibration.py" \
  --input "${RAW_JSON}" \
  --output "${FIT_JSON}" \
  --device-name-prefix GPU0

"${PYTHON_BIN}" "${SCRIPT_DIR}/export_dops_calibration.py" \
  --fit "${FIT_JSON}" \
  --base-hardware "${BASE_HARDWARE}" \
  --base-config "${BASE_CONFIG}" \
  --hardware-out "${HARDWARE_JSON}" \
  --config-out "${EVALUATE_JSON}" \
  --runtime-model-out "${RUNTIME_JSON}" \
  "${EXPECT_ARGS[@]}"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate_calibrated_export.py" \
  --config "${EVALUATE_JSON}"

echo "[GPU-CALIBRATION] completed output_root=${OUTPUT_ROOT}"
echo "[GPU-CALIBRATION] raw=${RAW_JSON}"
echo "[GPU-CALIBRATION] fit=${FIT_JSON}"
echo "[GPU-CALIBRATION] hardware=${HARDWARE_JSON}"
echo "[GPU-CALIBRATION] config=${EVALUATE_JSON}"
echo "[GPU-CALIBRATION] runtime_model=${RUNTIME_JSON}"
