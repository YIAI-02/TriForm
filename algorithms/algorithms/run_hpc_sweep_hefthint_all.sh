#!/bin/bash
#SBATCH -J sweep_hefthint_all
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

# ----------------------------
# User-tunable knobs
# ----------------------------
MODE=${MODE:-grid}                # random | grid
TRIALS=${TRIALS:-256}               # used when MODE=random
# SEED=${SEED:-42}
OUTDIR=${OUTDIR:-./output/sweep_hefthint_all}
RUNNER_SCRIPT=${RUNNER_SCRIPT:-./command_single_evaluate.sh}
WORKDIR=${WORKDIR:-.}
OBJECTIVE=${OBJECTIVE:-total}
REPEAT=${REPEAT:-1}

# Practical default candidate lists for all current HEFTHINT knobs.
# You can shrink / enlarge them directly here.
python3 ./sweep_hefthint_all_params.py \
  --mode "${MODE}" \
  --trials "${TRIALS}" \
  --script "${RUNNER_SCRIPT}" \
  --workdir "${WORKDIR}" \
  --objective "${OBJECTIVE}" \
  --repeat "${REPEAT}" \
  --h 2 3 4 \
  --gamma 0.6\
  --lambda_ 0\
  --plan-hint-max 3 \
  --eta 1 5 10 \
  --amort-enable true \
  --amort-alpha 2 4 6 8\
  --amort-rmin 1.0\
  --amort-reuse-prob 0.5 1.0 \
  --outdir "${OUTDIR}" \
  --resume \
  "$@"

# If you really want a full grid instead of random sampling, submit like this:
#   sbatch --export=MODE=grid,OUTDIR=./output/sweep_hefthint_all_grid run_hpc_sweep_hefthint_all.sh

echo "HEFTHINT all-parameter sweep done."
