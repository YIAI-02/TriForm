#!/bin/bash
#SBATCH -J single_eval
#SBATCH --chdir=/lustre/home/2501111916/workspace/DOPS_0407_final/TriForm
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

bash /lustre/home/2501111916/workspace/DOPS_0407_final/TriForm/commands/command_single_evaluate.sh