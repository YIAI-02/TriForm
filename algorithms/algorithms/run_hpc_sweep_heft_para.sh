#!/bin/bash
#SBATCH -J 0224_test
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
python3 sweep_hefthint.py   --mode grid   --gamma 0 0.2 0.4 0.6 --lambda_ 0 4 8 --eta 50 100 200 --objective total   --outdir ./output/sweep_hefthint_llama7b  --resume

echo "Sweep done."