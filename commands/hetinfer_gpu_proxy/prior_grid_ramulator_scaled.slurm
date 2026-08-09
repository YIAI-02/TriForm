#!/usr/bin/env bash
#SBATCH -J dops_gpu_pim_ramscale
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-2%1
#SBATCH -t 48:00:00
#SBATCH -o dops_gpu_pim_ramscale.%A_%a.out

set -euo pipefail

PROJECT_ROOT="${DOPS_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"

bash commands/hetinfer_gpu_proxy/run_prior_profile.sh ramulator-scaled "${SLURM_ARRAY_TASK_ID}"
