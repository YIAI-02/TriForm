#!/bin/bash
#SBATCH -J eval_a100_single_op_fit
#SBATCH -p GPU40G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail

echo "[slurm] jobid=${SLURM_JOB_ID} nodelist=${SLURM_NODELIST}"
echo "[slurm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

# ------------------------------------------------------------
# Conda / env
# ------------------------------------------------------------
set +u
source ~/.bashrc
set -u
CONDA_ENV="${CONDA_ENV:-triform310}"
conda activate "${CONDA_ENV}"

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# ------------------------------------------------------------
# Repo root + PYTHONPATH
# ------------------------------------------------------------
TRIFORM_ROOT="${TRIFORM_ROOT:-$(cd ../.. && pwd)}"
cd "${TRIFORM_ROOT}"

export PYTHONPATH="${TRIFORM_ROOT}/algorithms:${TRIFORM_ROOT}:${PYTHONPATH:-}"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
EVAL_SCRIPT="${EVAL_SCRIPT:-${TRIFORM_ROOT}/experiment/paper_fig/eval_gpu_single_op_fit.py}"
OUT_DIR="${OUT_DIR:-${TRIFORM_ROOT}/experiment/paper_fig/eval_gpu_single_op_fit_output}"
PREFIX="${PREFIX:-a100_single_op}"

mkdir -p "${OUT_DIR}"

# schedule_deploy_verify script discovery (override with VERIFY_SCRIPT)
VERIFY_SCRIPT="${VERIFY_SCRIPT:-}"
if [[ -z "${VERIFY_SCRIPT}" ]]; then
  if [[ -f "${TRIFORM_ROOT}/verify/schedule_deploy_verify.py" ]]; then
    VERIFY_SCRIPT="${TRIFORM_ROOT}/verify/schedule_deploy_verify.py"
  else
    VERIFY_SCRIPT="${TRIFORM_ROOT}/schedule_deploy_verify.py"
  fi
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[slurm] ERROR: EVAL_SCRIPT not found: ${EVAL_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${VERIFY_SCRIPT}" ]]; then
  echo "[slurm] ERROR: VERIFY_SCRIPT not found: ${VERIFY_SCRIPT}" >&2
  exit 3
fi

# ------------------------------------------------------------
# Parameters (pass via sbatch --export=ALL,VAR=... )
# ------------------------------------------------------------
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-5}"

GPU_DEVICE="${GPU_DEVICE:-cuda}"
GPU_DTYPE="${GPU_DTYPE:-fp16}"
DTYPE="${DTYPE:-fp16}"

# Sweep parameters
BATCHES="${BATCHES:-1,4,8,16,32}"
PREFILL_LENS="${PREFILL_LENS:-512,1024,2048}"

# Model shape json (llama_7b_shape.json etc)
LLAMA_SHAPE_JSON="${LLAMA_SHAPE_JSON:-}"
DECODE_CONTEXT_LENS="${DECODE_CONTEXT_LENS:-128,256,512,1024}"
DECODE_STRIDE="${DECODE_STRIDE:-64}"
PHASES="${PHASES:-decode}"
OPS="${OPS:-}"

DEVICE_NAME="${DEVICE_NAME:-A100_GPU0}"
A100_TFLOPS="${A100_TFLOPS:-312.0}"
A100_MEM_BW_GBS="${A100_MEM_BW_GBS:-1555.0}"

DEBUG="${DEBUG:-0}"
DEBUG_PRINT_N="${DEBUG_PRINT_N:-8}"

echo "[slurm] TRIFORM_ROOT=${TRIFORM_ROOT}"
echo "[slurm] EVAL_SCRIPT=${EVAL_SCRIPT}"
echo "[slurm] VERIFY_SCRIPT=${VERIFY_SCRIPT}"
echo "[slurm] OUT_DIR=${OUT_DIR} PREFIX=${PREFIX}"
echo "[slurm] warmup=${WARMUP} iters=${ITERS} gpu_dtype=${GPU_DTYPE} costmodel_dtype=${DTYPE}"
echo "[slurm] batches=${BATCHES} prefill_lens=${PREFILL_LENS}"
echo "[slurm] llama_shape_json=${LLAMA_SHAPE_JSON:-<auto>}"
echo "[slurm] decode_context_lens=${DECODE_CONTEXT_LENS} decode_stride=${DECODE_STRIDE}"
echo "[slurm] phases=${PHASES} ops=${OPS:-<default>}"
echo "[slurm] device_name=${DEVICE_NAME} a100_tflops=${A100_TFLOPS} a100_mem_bw_gbs=${A100_MEM_BW_GBS}"
echo "[slurm] debug=${DEBUG} debug_print_n=${DEBUG_PRINT_N}"

cmd=(python "${EVAL_SCRIPT}"
  --run-gpu
  --verify-script "${VERIFY_SCRIPT}"
  --out-dir "${OUT_DIR}"
  --prefix "${PREFIX}"
  --gpu-device "${GPU_DEVICE}"
  --gpu-dtype "${GPU_DTYPE}"
  --dtype "${DTYPE}"
  --batches "${BATCHES}"
  --prefill-lens "${PREFILL_LENS}"
  --decode-context-lens "${DECODE_CONTEXT_LENS}"
  --decode-stride "${DECODE_STRIDE}"
  --phases "${PHASES}"
  --device-name "${DEVICE_NAME}"
  --a100-tflops "${A100_TFLOPS}"
  --a100-mem-bw-gbs "${A100_MEM_BW_GBS}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --debug-print-n "${DEBUG_PRINT_N}"
)

if [[ -n "${LLAMA_SHAPE_JSON}" ]]; then
  cmd+=(--llama-shape-json "${LLAMA_SHAPE_JSON}")
fi

if [[ -n "${OPS}" ]]; then
  cmd+=(--ops "${OPS}")
fi

if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
  cmd+=(--debug)
fi

echo "[slurm] Running: ${cmd[*]}"
"${cmd[@]}"

echo "[slurm] Done."
echo "[slurm] Outputs:"
echo "  - ${OUT_DIR%/}/${PREFIX}.sweep_summary.csv"
echo "  - ${OUT_DIR%/}/${PREFIX}.sweep_detail.csv"
echo "  - (per-case) ${OUT_DIR%/}/${PREFIX}_B*_P*_*.*.csv/.json"
if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
  echo "  - ${OUT_DIR%/}/${PREFIX}.debug_log.txt"
  echo "  - (per-case) ${OUT_DIR%/}/${PREFIX}_B*_P*_* .gpu_debug.txt"
fi
