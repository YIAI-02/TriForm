#!/bin/bash
#SBATCH -J sweep_bifocal_all
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
COMMANDS_DIR="${PROJECT_ROOT}/commands"

cd "${PROJECT_ROOT}"

source ~/.bashrc
conda activate triform310

MODE=${MODE:-grid}
TRIALS=${TRIALS:-256}
OUTDIR=${OUTDIR:-${PROJECT_ROOT}/output/sweep_bifocal_all}
RUNNER_SCRIPT=${RUNNER_SCRIPT:-${COMMANDS_DIR}/command_single_evaluate.sh}
WORKDIR=${WORKDIR:-${SRC_DIR}}
OBJECTIVE=${OBJECTIVE:-total}
REPEAT=${REPEAT:-1}

python3 "${COMMANDS_DIR}/sweep_bifocal_all_params.py" \
  --mode "${MODE}" \
  --trials "${TRIALS}" \
  --script "${RUNNER_SCRIPT}" \
  --workdir "${WORKDIR}" \
  --objective "${OBJECTIVE}" \
  --repeat "${REPEAT}" \
  --h 3 \
  --gamma 0 0.2 0.4 0.6 \
  --lambda_ 0 5 10\
  --plan-hint-max 3 \
  --eta 1 5 10 \
  --amort-enable true \
  --amort-alpha 2 4 6 8 \
  --amort-rmin 1.0 \
  --amort-reuse-prob 0.5 1.0 \
  --outdir "${OUTDIR}" \
  --resume \
  "$@"

echo "Bifocal all-parameter sweep done."
