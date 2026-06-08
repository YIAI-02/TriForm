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
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -f "${SUBMIT_DIR}/commands/sweep_bifocal_all_params.py" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
  else
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CANDIDATE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    if [[ -f "${CANDIDATE_ROOT}/commands/sweep_bifocal_all_params.py" ]]; then
      PROJECT_ROOT="${CANDIDATE_ROOT}"
    else
      echo "ERROR: Cannot locate TriForm project root." >&2
      echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
      echo "  pwd=$(pwd)" >&2
      echo "  BASH_SOURCE[0]=${BASH_SOURCE[0]}" >&2
      echo "Submit from the repo root or pass PROJECT_ROOT=/path/to/TriForm." >&2
      exit 2
    fi
  fi
fi

SRC_DIR="${PROJECT_ROOT}/src"
COMMANDS_DIR="${PROJECT_ROOT}/commands"

cd "${PROJECT_ROOT}"
echo "Using PROJECT_ROOT=${PROJECT_ROOT}"

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
  --h 1 2 3 4 \
  --gamma 0 0.1 0.2 0.4 0.6 \
  --lambda_ 0 0.5 1 2 4 \
  --plan-hint-max 0 1 3 5 \
  --eta 0.0 0.05 0.1 0.2 0.5 1.0 \
  --amort-enable true \
  --amort-alpha 0 0.5 1 2 \
  --amort-rmin 1 4 8 \
  --amort-reuse-prob 0.25 0.5 0.75 1.0 \
  --outdir "${OUTDIR}" \
  --resume \
  "$@"

echo "Bifocal all-parameter sweep done."
