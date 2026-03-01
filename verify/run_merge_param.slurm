#!/bin/bash
#SBATCH -J verify_merge
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
# Environment (keep consistent with your other jobs)
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

PY_SCRIPT="${PY_SCRIPT:-$REPO_ROOT/verify/schedule_deploy_verify_gpu.py}"

# ----------------------------------------------------------------------
# Optional: collect mode (concatenate multiple merge.csv into one)
# ----------------------------------------------------------------------
COLLECT_MERGES="${COLLECT_MERGES:-0}"
MERGE_LIST_FILE="${MERGE_LIST_FILE:-}"
COLLECT_OUT_CSV="${COLLECT_OUT_CSV:-}"
COLLECT_SKIP_MISSING="${COLLECT_SKIP_MISSING:-1}"
SCHEDULE_CSV="${SCHEDULE_CSV:-}" 
OUT_DIR="${OUT_DIR:-}"
PREFIX="${PREFIX:-}"

GPU_RESULTS_JSON="${GPU_RESULTS_JSON:-}" 
PIM_RESULTS_JSON="${PIM_RESULTS_JSON:-}" 
GPU_OK_FILE="${GPU_OK_FILE:-}"
PIM_OK_FILE="${PIM_OK_FILE:-}"
GPU_LOG_DIR="${GPU_LOG_DIR:-}"
PIM_LOG_DIR="${PIM_LOG_DIR:-}"

COMM_MODEL="${COMM_MODEL:-schedule}"
PCIE_LANES="${PCIE_LANES:-16}"
DECODE_STRIDE="${DECODE_STRIDE:-1}"
NON_OVERLAP="${NON_OVERLAP:-0.2}"
COMMS_CSV="${COMMS_CSV:-}"
BATCH="${BATCH:-}"
OUT_CSV="${OUT_CSV:-}"
OUT_STEPS_CSV="${OUT_STEPS_CSV:-}"

if [[ -z "$GPU_RESULTS_JSON" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  GPU_RESULTS_JSON="${OUT_DIR%/}/${PREFIX}.gpu_results.json"
fi
if [[ -z "$PIM_RESULTS_JSON" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  PIM_RESULTS_JSON="${OUT_DIR%/}/${PREFIX}.pim_results.json"
fi
if [[ -z "$OUT_CSV" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  OUT_CSV="${OUT_DIR%/}/${PREFIX}.merge.csv"
fi

# Backward-compatible default for manual runs
OUT_CSV="${OUT_CSV:-merge_report.csv}"

ALLOW_MISSING="${ALLOW_MISSING:-1}"

DEBUG_MERGE="${DEBUG_MERGE:-0}"
DEBUG_TXT="${DEBUG_TXT:-}"

# If collect mode is enabled, run and exit early.
if [[ "${COLLECT_MERGES}" == "1" || "${COLLECT_MERGES}" == "true" ]]; then
  if [[ -z "${MERGE_LIST_FILE}" ]]; then
    echo "[slurm] ERROR: COLLECT_MERGES=1 but MERGE_LIST_FILE is empty" >&2
    exit 20
  fi
  if [[ -z "${COLLECT_OUT_CSV}" ]]; then
    echo "[slurm] ERROR: COLLECT_MERGES=1 but COLLECT_OUT_CSV is empty" >&2
    exit 21
  fi

  echo "[slurm] collect-merge mode"
  echo "[slurm] PY_SCRIPT=${PY_SCRIPT}"
  echo "[slurm] MERGE_LIST_FILE=${MERGE_LIST_FILE}"
  echo "[slurm] COLLECT_OUT_CSV=${COLLECT_OUT_CSV}"
  echo "[slurm] COLLECT_SKIP_MISSING=${COLLECT_SKIP_MISSING}"

  cmd=(python "$PY_SCRIPT" collect-merge \
    --list "$MERGE_LIST_FILE" \
    --out-csv "$COLLECT_OUT_CSV")

  if [[ "${COLLECT_SKIP_MISSING}" == "1" || "${COLLECT_SKIP_MISSING}" == "true" ]]; then
    cmd+=(--skip-missing)
  fi

  echo "[slurm] Running: ${cmd[*]}"
  "${cmd[@]}"
  echo "[slurm] done at $(date)"
  exit 0
fi

if [[ -z "${SCHEDULE_CSV}" ]]; then
  echo "[slurm] ERROR: SCHEDULE_CSV is empty" >&2
  exit 2
fi

echo "[slurm] PY_SCRIPT=${PY_SCRIPT}"
echo "[slurm] SCHEDULE_CSV=${SCHEDULE_CSV}"
echo "[slurm] GPU_RESULTS_JSON=${GPU_RESULTS_JSON:-<none>}"
echo "[slurm] PIM_RESULTS_JSON=${PIM_RESULTS_JSON:-<none>}"
echo "[slurm] GPU_OK_FILE=${GPU_OK_FILE:-<none>}"
echo "[slurm] PIM_OK_FILE=${PIM_OK_FILE:-<none>}"
echo "[slurm] COMM_MODEL=${COMM_MODEL} PCIE_LANES=${PCIE_LANES} DECODE_STRIDE=${DECODE_STRIDE} NON_OVERLAP=${NON_OVERLAP}"
echo "[slurm] COMMS_CSV=${COMMS_CSV:-<none>}"
echo "[slurm] OUT_CSV=${OUT_CSV}"
echo "[slurm] OUT_STEPS_CSV=${OUT_STEPS_CSV:-<none>}"

if [[ -n "$GPU_OK_FILE" && ! -f "$GPU_OK_FILE" ]]; then
  echo "[slurm] ERROR: GPU stage failed or did not finish cleanly (missing GPU_OK_FILE=$GPU_OK_FILE)" >&2
  if [[ -n "$GPU_LOG_DIR" ]]; then
    echo "[slurm] Hint: check GPU logs under: $GPU_LOG_DIR" >&2
  fi
  exit 10
fi
if [[ -n "$PIM_OK_FILE" && ! -f "$PIM_OK_FILE" ]]; then
  echo "[slurm] ERROR: PIM stage failed or did not finish cleanly (missing PIM_OK_FILE=$PIM_OK_FILE)" >&2
  if [[ -n "$PIM_LOG_DIR" ]]; then
    echo "[slurm] Hint: check PIM logs under: $PIM_LOG_DIR" >&2
  fi
  exit 11
fi

if [[ -n "$GPU_RESULTS_JSON" && "$GPU_RESULTS_JSON" != "-" && ! -f "$GPU_RESULTS_JSON" ]]; then
  echo "[slurm] ERROR: GPU_RESULTS_JSON not found: $GPU_RESULTS_JSON" >&2
  exit 12
fi
if [[ -n "$PIM_RESULTS_JSON" && "$PIM_RESULTS_JSON" != "-" && ! -f "$PIM_RESULTS_JSON" ]]; then
  echo "[slurm] ERROR: PIM_RESULTS_JSON not found: $PIM_RESULTS_JSON" >&2
  exit 13
fi

cmd=(python "$PY_SCRIPT" merge \
  --schedule "$SCHEDULE_CSV" \
  --comm-model "$COMM_MODEL" \
  --pcie-lanes "$PCIE_LANES" \
  --decode-stride "$DECODE_STRIDE" \
  --non-overlap "$NON_OVERLAP" \
  --out-csv "$OUT_CSV")

if [[ -n "${BATCH}" ]]; then
  cmd+=(--batch "$BATCH")
fi

if [[ -n "${GPU_RESULTS_JSON}" && "${GPU_RESULTS_JSON}" != "-" ]]; then
  cmd+=(--gpu-results "$GPU_RESULTS_JSON")
fi
if [[ -n "${PIM_RESULTS_JSON}" && "${PIM_RESULTS_JSON}" != "-" ]]; then
  cmd+=(--pim-results "$PIM_RESULTS_JSON")
fi
if [[ -n "${COMMS_CSV}" ]]; then
  cmd+=(--comms "$COMMS_CSV")
fi
if [[ -n "${OUT_STEPS_CSV}" ]]; then
  cmd+=(--out-steps-csv "$OUT_STEPS_CSV")
fi
if [[ "${ALLOW_MISSING}" == "1" || "${ALLOW_MISSING}" == "true" ]]; then
  cmd+=(--allow-missing)
fi

if [[ "${DEBUG_MERGE}" == "1" || "${DEBUG_MERGE}" == "true" ]]; then
  cmd+=(--debug)
  if [[ -n "${DEBUG_TXT}" ]]; then
    cmd+=(--debug-txt "$DEBUG_TXT")
  fi
fi

echo "[slurm] Running: ${cmd[*]}"
"${cmd[@]}"

echo "[slurm] done at $(date)"
