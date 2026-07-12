#!/bin/bash
#SBATCH -J single_eval
#SBATCH -p C064M1024G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -euo pipefail

echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "PWD=$PWD"

PROJECT_ROOT="${DOPS_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${PROJECT_ROOT}"
bash commands/command_single_evaluate.sh
