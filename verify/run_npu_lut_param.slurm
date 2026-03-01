#!/bin/bash
#SBATCH -J verify_npu_lut
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail

LOG_DIR="${LOG_DIR:-}"
if [[ -n "$LOG_DIR" ]]; then
  mkdir -p "$LOG_DIR"
fi

echo "[slurm] jobid=${SLURM_JOB_ID} nodelist=${SLURM_NODELIST}"
echo "[slurm] host=$(hostname)"
echo "[slurm] pwd=$(pwd)"
echo "[slurm] starting at $(date)"

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
set +u
source ~/.bashrc
set -u
conda activate triform310

# ----------------------------------------------------------------------
# Parameters (recommended to pass via `sbatch --export=...`)
# ----------------------------------------------------------------------
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"

# Ensure repo modules (e.g., config.py) are importable
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/algorithm:$REPO_ROOT/algorithms:${PYTHONPATH:-}"

PY_SCRIPT="${PY_SCRIPT:-$REPO_ROOT/verify/schedule_deploy_verify.py}"

TASKS_JSON="${TASKS_JSON:-}"
OUT_JSON="${OUT_JSON:-}"
OK_FILE="${OK_FILE:-}"

BATCH="${BATCH:-}"

MMAD_LUT="${MMAD_LUT:-}"
SOFTMAX_LUT="${SOFTMAX_LUT:-}"
GELU_LUT="${GELU_LUT:-}"
NORM_LUT="${NORM_LUT:-}"

NPU_DTYPE="${NPU_DTYPE:-}"
NPU_MEM_BW_GBS="${NPU_MEM_BW_GBS:-0}"
NPU_OP_OVERHEAD_US="${NPU_OP_OVERHEAD_US:-0}"
NPU_NO_MEM_BOUND="${NPU_NO_MEM_BOUND:-0}"

NPU_DEBUG="${NPU_DEBUG:-0}"
NPU_DEBUG_TXT="${NPU_DEBUG_TXT:-}"

if [[ -z "$TASKS_JSON" || -z "$OUT_JSON" ]]; then
  echo "[slurm] ERROR: TASKS_JSON and OUT_JSON must be set" >&2
  exit 2
fi
if [[ -z "$OK_FILE" ]]; then
  OK_FILE="${OUT_JSON}.ok"
fi

# If the LUT paths are provided, pass them via CLI (the python code will also set env vars).
cmd=(python "$PY_SCRIPT" run-npu --tasks "$TASKS_JSON" --out "$OUT_JSON")

if [[ -n "$MMAD_LUT" ]]; then
  cmd+=(--mmad-lut "$MMAD_LUT")
fi
if [[ -n "$SOFTMAX_LUT" ]]; then
  cmd+=(--softmax-lut "$SOFTMAX_LUT")
fi
if [[ -n "$GELU_LUT" ]]; then
  cmd+=(--gelu-lut "$GELU_LUT")
fi
if [[ -n "$NORM_LUT" ]]; then
  cmd+=(--norm-lut "$NORM_LUT")
fi

if [[ -n "$NPU_DTYPE" ]]; then
  cmd+=(--npu-dtype "$NPU_DTYPE")
fi
# Always pass numeric knobs (defaults are fine)
cmd+=(--npu-mem-bw-gbs "$NPU_MEM_BW_GBS")
cmd+=(--npu-op-overhead-us "$NPU_OP_OVERHEAD_US")
if [[ "$NPU_NO_MEM_BOUND" == "1" || "$NPU_NO_MEM_BOUND" == "true" ]]; then
  cmd+=(--npu-no-mem-bound)
fi

if [[ -n "$BATCH" ]]; then
  cmd+=(--batch "$BATCH")
fi

if [[ "$NPU_DEBUG" == "1" || "$NPU_DEBUG" == "true" ]]; then
  cmd+=(--debug)
fi
if [[ -n "$NPU_DEBUG_TXT" ]]; then
  cmd+=(--debug-txt "$NPU_DEBUG_TXT")
fi

echo "[slurm] PY_SCRIPT=${PY_SCRIPT}"
echo "[slurm] TASKS_JSON=${TASKS_JSON}"
echo "[slurm] OUT_JSON=${OUT_JSON}"
echo "[slurm] OK_FILE=${OK_FILE}"
echo "[slurm] LUT: MMAD_LUT=${MMAD_LUT:-<default>} SOFTMAX_LUT=${SOFTMAX_LUT:-<default>} GELU_LUT=${GELU_LUT:-<default>} NORM_LUT=${NORM_LUT:-<default>}"
echo "[slurm] NPU_MEM_BW_GBS=${NPU_MEM_BW_GBS} NPU_OP_OVERHEAD_US=${NPU_OP_OVERHEAD_US} NPU_NO_MEM_BOUND=${NPU_NO_MEM_BOUND}"
echo "[slurm] cmd: ${cmd[*]}"

rm -f "$OK_FILE"

set +e
"${cmd[@]}"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "[slurm] ERROR: run-npu failed with exit code $rc" >&2
  exit "$rc"
fi

touch "$OK_FILE"
echo "[slurm] OK_FILE touched: $OK_FILE"
echo "[slurm] finished at $(date)"
