#!/bin/bash
#SBATCH -J run_hpc_cpu_npu_aim_8_batch16_3
#SBATCH -p C064M1024G
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

bash sweep_models_npu_aim_evaluate_8shards_batch16_3.sh --jobs 8

