#!/bin/bash
#SBATCH -J llama_layer_16
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
conda activate cent

bash sweep_models_lens_evaluate.sh --jobs 8
