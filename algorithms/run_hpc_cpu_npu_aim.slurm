#!/bin/bash
#SBATCH -J run_hpc_cpu_npu_aim
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH --array=0-15%4
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 96:00:00
#SBATCH -o job.%A_%a.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate triform310

bash sweep_models_npu_aim_evaluate.sh --jobs 16

