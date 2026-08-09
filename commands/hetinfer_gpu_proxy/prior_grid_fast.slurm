#!/usr/bin/env bash
#SBATCH -J dops_gpu_pim_fast
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-17%6
#SBATCH -t 12:00:00
#SBATCH -o dops_gpu_pim_fast.%A_%a.out

set -euo pipefail

PROJECT_ROOT="${DOPS_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"

# This is a simulator job. Requesting a physical GPU would not make the
# LLMCompass proxy a real GPU measurement.
bash commands/hetinfer_gpu_proxy/run_prior_profile.sh fast "${SLURM_ARRAY_TASK_ID}"
