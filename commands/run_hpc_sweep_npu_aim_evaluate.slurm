#!/bin/bash
#SBATCH -J sweep_models_npu_aim_evaluate
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -eo pipefail

source ~/.bashrc
conda activate triform310

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -f "${SUBMIT_DIR}/commands/sweep_models_npu_aim_evaluate.sh" ]]; then
  cd "${SUBMIT_DIR}"
  bash "${SUBMIT_DIR}/commands/sweep_models_npu_aim_evaluate.sh" --jobs 24

elif [[ -f "${SUBMIT_DIR}/sweep_models_npu_aim_evaluate.sh" ]]; then
  cd "$(cd "${SUBMIT_DIR}/.." && pwd)"
  bash "${SUBMIT_DIR}/sweep_models_npu_aim_evaluate.sh" --jobs 24

else
  echo "[FATAL] sweep_models_npu_aim_evaluate.sh not found." >&2
  echo "SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 2
fi