#!/bin/bash
#SBATCH -J sweep_models_npu_aim_evaluate_4shards_batch16_4
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate triform310

bash sweep_models_npu_aim_evaluate_4shards_batch16_4.sh --jobs 16

