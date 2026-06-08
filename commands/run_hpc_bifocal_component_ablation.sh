#!/bin/bash
#SBATCH -J bifocal_ablation
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -eo pipefail

# Slurm often executes a copied batch script from /var/spool/slurmd.
# In that case BASH_SOURCE[0] no longer points to this repository, so
# resolving PROJECT_ROOT from the script path is unsafe.  Prefer an
# explicit PROJECT_ROOT, then SLURM_SUBMIT_DIR, and only fall back to
# BASH_SOURCE for non-Slurm/local execution.
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/commands/run_bifocal_component_ablation.py" ]]; then
  PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ ! -f "${PROJECT_ROOT}/commands/run_bifocal_component_ablation.py" ]]; then
  echo "ERROR: cannot find commands/run_bifocal_component_ablation.py" >&2
  echo "  PROJECT_ROOT=${PROJECT_ROOT}" >&2
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
  echo "  BASH_SOURCE=${BASH_SOURCE[0]}" >&2
  echo "Submit from the repository root or set PROJECT_ROOT=/path/to/TriForm." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
echo "[bifocal-ablation] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[bifocal-ablation] OUTDIR=${OUTDIR:-${PROJECT_ROOT}/output/bifocal_component_ablation}"
echo "[bifocal-ablation] SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"

source ~/.bashrc
conda activate triform310

OUTDIR=${OUTDIR:-${PROJECT_ROOT}/output/bifocal_component_ablation}
CONFIG=${CONFIG:-${PROJECT_ROOT}/src/examples/evaluate_len_sweep_config_npu.json}
HARDWARE_JSON=${HARDWARE_JSON:-${PROJECT_ROOT}/src/examples/hardware_1npu_2aim.json}
NPU_BACKEND=${NPU_BACKEND:-fast_mode}
VARIANT_SUITE=${VARIANT_SUITE:-minimal}
HORIZON_SUITE=${HORIZON_SUITE:-oracle}
FIXED_HORIZON=${FIXED_HORIZON:-256}
BEST_JSON=${BEST_JSON:-}
# Number of independent evaluate subprocesses to run concurrently.
# 0 means the Python driver will choose a conservative value from SLURM_CPUS_PER_TASK
# and cap it at 8. Increase PARALLEL_JOBS only after checking memory pressure.
PARALLEL_JOBS=${PARALLEL_JOBS:-0}
THREADS_PER_RUN=${THREADS_PER_RUN:-1}

BEST_ARGS=()
if [[ -n "${BEST_JSON}" ]]; then
  BEST_ARGS+=(--best-json "${BEST_JSON}")
fi

python3 "${PROJECT_ROOT}/commands/run_bifocal_component_ablation.py" \
  --config "${CONFIG}" \
  --hardware-json "${HARDWARE_JSON}" \
  --npu-backend "${NPU_BACKEND}" \
  --variant-suite "${VARIANT_SUITE}" \
  --horizon-suite "${HORIZON_SUITE}" \
  --fixed-horizon "${FIXED_HORIZON}" \
  --jobs "${PARALLEL_JOBS}" \
  --threads-per-run "${THREADS_PER_RUN}" \
  --outdir "${OUTDIR}" \
  --resume \
  "${BEST_ARGS[@]}" \
  "$@"

echo "Bifocal component ablation done. Results: ${OUTDIR}"
