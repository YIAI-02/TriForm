#!/bin/bash
#SBATCH -J 1212_hybrid_true
#SBATCH -p C064M0256G
#SBATCH --qos=low
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc
conda activate triform

bash sweep_models_lens_evaluate.sh --jobs 16
