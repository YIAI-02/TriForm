#!/bin/bash
#SBATCH -J sweep_bifocal
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

python3 "${SRC_DIR}/sweep_bifocal.py" \
  --mode grid \
  --gamma 0 0.2 0.4 0.6 \
  --lambda_ 0 4 8 \
  --eta 50 100 200 \
  --objective total \
  --script "${COMMANDS_DIR}/command_single_evaluate.sh" \
  --workdir "${SRC_DIR}" \
  --outdir "${PROJECT_ROOT}/output/sweep_bifocal_llama7b" \
  --resume \
  "$@"

echo "Bifocal sweep done."
