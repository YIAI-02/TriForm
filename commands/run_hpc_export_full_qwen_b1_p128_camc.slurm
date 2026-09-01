#!/bin/bash
#SBATCH -J qwen_b1p128_camc_export
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o qwen_b1p128_camc_export.%j.out
#SBATCH -e qwen_b1p128_camc_export.%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[FATAL] submit this script with sbatch; do not run it on a login node" >&2
  exit 2
fi

set +u
source ~/.bashrc
set -u
conda activate triform310

python_bin="${DOPS_PYTHON:-/lustre/home/2501111916/anaconda3/bin/python}"
test -x "${python_bin}"

repo_root="${DOPS_REPO:-${SLURM_SUBMIT_DIR}}"
if [[ ! -f "${repo_root}/commands/export_hetinfer_dense_prefill_camc.py" ]]; then
  echo "[FATAL] DOPS_REPO is not the DOPS CAMC checkout: ${repo_root}" >&2
  exit 2
fi

workspace_root="$(cd "${repo_root}/.." && pwd)"
het_infer_source="${HET_INFER_SOURCE_ROOT:-${workspace_root}/current}"
sidecar_root="${het_infer_source}/artifacts/full_qwen_1p8b/sidecars/b1-p128-h128"
output_dir="${CAMC_OUTPUT_DIR:-${repo_root}/output/full_qwen_1p8b/camc/b1-p128-h128/prefill}"

cd "${repo_root}"
PYTHONPATH="${repo_root}/src" "${python_bin}" -m pytest -q \
  tests/test_hetinfer_camc_profile_export.py \
  tests/test_hetinfer_dense_prefill_camc_export.py

"${python_bin}" commands/export_hetinfer_dense_prefill_camc.py \
  --prior "${sidecar_root}/prior.json" \
  --network "${sidecar_root}/network.json" \
  --tensor-bindings "${sidecar_root}/tensor_bindings.json" \
  --hardware "${repo_root}/src/examples/hardware_1npu_2aim.json" \
  --output-dir "${output_dir}"

printf 'source_prior=%s\n' "${sidecar_root}/prior.json"
printf 'output_dir=%s\n' "${output_dir}"
