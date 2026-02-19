#!/bin/bash
#SBATCH -J verify_run_gpu
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
LOG_DIR="${LOG_DIR:-}"
if [[ -n "$LOG_DIR" ]]; then
  mkdir -p "$LOG_DIR"
fi

echo "[slurm] jobid=${SLURM_JOB_ID} nodelist=${SLURM_NODELIST}"
echo "[slurm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi
set +u
source ~/.bashrc
set -u
conda activate triform310

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

# ----------------------------------------------------------------------
# Parameters (recommended to pass via `sbatch --export=...`)
# ----------------------------------------------------------------------
PY_SCRIPT="${PY_SCRIPT:-schedule_deploy_verify.py}"

OUT_DIR="${OUT_DIR:-}"
PREFIX="${PREFIX:-}"

TASKS_JSON="${TASKS_JSON:-}"
OUT_JSON="${OUT_JSON:-}"

if [[ -z "$TASKS_JSON" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  TASKS_JSON="${OUT_DIR%/}/${PREFIX}.gpu_tasks.json"
fi
if [[ -z "$OUT_JSON" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  OUT_JSON="${OUT_DIR%/}/${PREFIX}.gpu_results.json"
fi

# Backward-compatible defaults
TASKS_JSON="${TASKS_JSON:-./out/hefthint_seg.gpu_tasks.json}"
OUT_JSON="${OUT_JSON:-./out/hefthint_seg.gpu_results.json}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-10}"
DEVICE="${DEVICE:-}"
GPU_DTYPE="${GPU_DTYPE:-}"
OK_FILE="${OK_FILE:-}"

echo "[slurm] PY_SCRIPT=${PY_SCRIPT}"
echo "[slurm] TASKS_JSON=${TASKS_JSON}"
echo "[slurm] OUT_JSON=${OUT_JSON}"
echo "[slurm] WARMUP=${WARMUP} ITERS=${ITERS} DEVICE=${DEVICE:-<default>} GPU_DTYPE=${GPU_DTYPE:-<default>}"
echo "[slurm] OK_FILE=${OK_FILE:-<none>}"

if [[ ! -f "$TASKS_JSON" ]]; then
  echo "[slurm] ERROR: TASKS_JSON not found: $TASKS_JSON" >&2
  exit 2
fi

cmd=(python "$PY_SCRIPT" run-gpu \
    --tasks "$TASKS_JSON" \
    --out "$OUT_JSON" \
    --warmup "$WARMUP" --iters "$ITERS")

if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "$DEVICE")
fi
if [[ -n "${GPU_DTYPE}" ]]; then
  cmd+=(--gpu-dtype "$GPU_DTYPE")
fi

echo "[slurm] Running: ${cmd[*]}"
"${cmd[@]}"

if [[ ! -f "$OUT_JSON" ]]; then
  echo "[slurm] ERROR: OUT_JSON was not generated: $OUT_JSON" >&2
  exit 3
fi

if [[ -n "$OK_FILE" ]]; then
  mkdir -p "$(dirname "$OK_FILE")"
  echo "OK jobid=${SLURM_JOB_ID} at $(date)" > "$OK_FILE"
  echo "[slurm] wrote OK_FILE=$OK_FILE"
fi
