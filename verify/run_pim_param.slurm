#!/bin/bash
#SBATCH -J verify_run_pim
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 04:00:00
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
echo "[slurm] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

echo "[slurm] starting at $(date)"

# ----------------------------------------------------------------------
# Environment (keep consistent with your GPU job env)
# ----------------------------------------------------------------------
set +u
source ~/.bashrc
set -u
conda activate triform310
export LD_LIBRARY_PATH="/lustre/home/2501111916/workspace/TriForm/submodules/CENT/aim_simulator${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ----------------------------------------------------------------------
# Parameters (recommended to pass via `sbatch --export=...`)
# ----------------------------------------------------------------------
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"

PY_SCRIPT="${PY_SCRIPT:-$REPO_ROOT/verify/schedule_deploy_verify.py}"

OUT_DIR="${OUT_DIR:-}"
PREFIX="${PREFIX:-}"

TASKS_JSON="${TASKS_JSON:-}"
OUT_JSON="${OUT_JSON:-}"
PIM_DEBUG="${PIM_DEBUG:-0}"
PIM_DEBUG_TXT="${PIM_DEBUG_TXT:-}"
PIM_NO_CACHE="${PIM_NO_CACHE:-0}"

if [[ -z "$TASKS_JSON" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  TASKS_JSON="${OUT_DIR%/}/${PREFIX}.pim_tasks.json"
fi
if [[ -z "$OUT_JSON" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  OUT_JSON="${OUT_DIR%/}/${PREFIX}.pim_results.json"
fi

if [[ -z "$PIM_DEBUG_TXT" && -n "$OUT_DIR" && -n "$PREFIX" ]]; then
  PIM_DEBUG_TXT="${OUT_DIR%/}/${PREFIX}.pim_debug.log"
fi

# Backward-compatible defaults
TASKS_JSON="${TASKS_JSON:-./out/hefthint_seg.pim_tasks.json}"
OUT_JSON="${OUT_JSON:-./out/hefthint_seg.pim_results.json}"

CENT_SIM_ROOT="${CENT_SIM_ROOT:-}"

PIM_RAMULATOR_CONFIG="${PIM_RAMULATOR_CONFIG:-$REPO_ROOT/algorithms/aim_simulator/example.yaml}"
PIM_HW_JSON="${PIM_HW_JSON:-$REPO_ROOT/algorithms/aim_simulator/PIM_AiM.json}"
PIM_RAMULATOR_BIN="${PIM_RAMULATOR_BIN:-$REPO_ROOT/algorithms/ramulator2}"
PIM_NUM_DEVICES="${PIM_NUM_DEVICES:-4}"

# Optional: success sentinel
OK_FILE="${OK_FILE:-}"

echo "[slurm] PY_SCRIPT=${PY_SCRIPT}"
echo "[slurm] TASKS_JSON=${TASKS_JSON}"
echo "[slurm] OUT_JSON=${OUT_JSON}"
echo "[slurm] CENT_SIM_ROOT=${CENT_SIM_ROOT:-<unset>}"
echo "[slurm] PIM_RAMULATOR_CONFIG=${PIM_RAMULATOR_CONFIG}"
echo "[slurm] PIM_HW_JSON=${PIM_HW_JSON}"
echo "[slurm] PIM_RAMULATOR_BIN=${PIM_RAMULATOR_BIN}"
echo "[slurm] PIM_NUM_DEVICES=${PIM_NUM_DEVICES}"
echo "[slurm] PIM_DEBUG=${PIM_DEBUG}"
echo "[slurm] PIM_DEBUG_TXT=${PIM_DEBUG_TXT:-<none>}"
echo "[slurm] PIM_NO_CACHE=${PIM_NO_CACHE}"
echo "[slurm] OK_FILE=${OK_FILE:-<none>}"
if [[ ! -f "$TASKS_JSON" ]]; then
  echo "[slurm] ERROR: TASKS_JSON not found: $TASKS_JSON" >&2
  exit 2
fi
if [[ ! -f "$PIM_RAMULATOR_CONFIG" ]]; then
  echo "[slurm] ERROR: Ramulator config not found: $PIM_RAMULATOR_CONFIG" >&2
  echo "[slurm] Hint: pass an absolute path via sbatch --export=...PIM_RAMULATOR_CONFIG=/abs/path/example.yaml" >&2
  exit 3
fi
if [[ ! -f "$PIM_HW_JSON" ]]; then
  echo "[slurm] ERROR: PIM_HW_JSON not found: $PIM_HW_JSON" >&2
  exit 4
fi

if [[ "$PIM_RAMULATOR_BIN" == */* || "$PIM_RAMULATOR_BIN" == .* || "$PIM_RAMULATOR_BIN" == ~* ]]; then
  if [[ ! -f "$PIM_RAMULATOR_BIN" ]]; then
    echo "[slurm] ERROR: PIM_RAMULATOR_BIN not found: $PIM_RAMULATOR_BIN" >&2
    exit 5
  fi
fi

cmd=(python "$PY_SCRIPT" run-pim \
  --tasks "$TASKS_JSON" \
  --out "$OUT_JSON" \
  --pim-ramulator-config "$PIM_RAMULATOR_CONFIG" \
  --pim-hw-json "$PIM_HW_JSON" \
  --pim-ramulator-bin "$PIM_RAMULATOR_BIN" \
  --pim-num-devices "$PIM_NUM_DEVICES")

if [[ -n "${CENT_SIM_ROOT}" ]]; then
  cmd+=(--cent-sim-root "$CENT_SIM_ROOT")
fi

if [[ "${PIM_DEBUG}" != "0" ]]; then
  cmd+=(--debug)
  if [[ -n "${PIM_DEBUG_TXT}" ]]; then
    cmd+=(--debug-txt "$PIM_DEBUG_TXT")
  fi
fi

if [[ "${PIM_NO_CACHE}" != "0" ]]; then
  cmd+=(--no-cache)
fi

echo "[slurm] Running: ${cmd[*]}"
"${cmd[@]}"

if [[ ! -f "$OUT_JSON" ]]; then
  echo "[slurm] ERROR: OUT_JSON was not generated: $OUT_JSON" >&2
  exit 6
fi

if [[ -n "$OK_FILE" ]]; then
  mkdir -p "$(dirname "$OK_FILE")"
  echo "OK jobid=${SLURM_JOB_ID} at $(date)" > "$OK_FILE"
  echo "[slurm] wrote OK_FILE=$OK_FILE"
fi

echo "[slurm] done at $(date)"