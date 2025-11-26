#!/bin/bash
#SBATCH -J cpu_test
#SBATCH -p C064M0256G
#SBATCH --qos=low
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o job.%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate cent

bash sweep_models_lens_evaluate.sh
