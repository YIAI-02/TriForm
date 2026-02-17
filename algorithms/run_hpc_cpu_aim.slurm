#!/bin/bash
#SBATCH -J 0105_npu
#SBATCH -p C064M0256G
#SBATCH --qos=low
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate triform310

bash sweep_models_gpu_aim_evaluate.sh --jobs 32
