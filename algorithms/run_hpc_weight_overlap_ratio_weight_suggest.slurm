#!/bin/bash
#SBATCH -J ws_overlap_ratio
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
CONDA_ENV_NAME="${CONDA_ENV_NAME:-triform310}"
conda activate "${CONDA_ENV_NAME}"

# ------------------------------------------------------------------
# Direct-submit wrapper for studying the impact of
# PIM_WEIGHT_LOAD_OVERLAP_RATIO and WEIGHT_LOAD_COMPUTE_OVERLAP_RATIO on weight-suggest(hefthint).
#
# Example:
# export CONFIG=./examples/weight_suggest_overlap_ratio_base.json
# export OUTDIR=./output/ws_overlap_ratio
# export MODE=grid
# export OBJECTIVE=total
# export ALGOS=hefthint
# export COMBO_WORKERS=4
# export MODELS="qwen:7b"
# export PREFILLS="512"
# export DECODES="128"
# export BATCHES="1"
# export DEBUG=1

# sbatch --export=ALL run_hpc_weight_overlap_ratio_weight_suggest.slurm
# ------------------------------------------------------------------

CONFIG="${CONFIG:-}"
if [[ -z "${CONFIG}" ]]; then
  echo "[FATAL] Please set CONFIG=/path/to/base_weight_suggest_or_evaluate_config.json" >&2
  exit 2
fi

OUTDIR="${OUTDIR:-./output/ws_overlap_ratio}"
MODE="${MODE:-grid}"
OBJECTIVE="${OBJECTIVE:-total}"
ALGOS="${ALGOS:-hefthint}"
PIM_WEIGHT_LOAD_OVERLAP_RATIOS="${PIM_WEIGHT_LOAD_OVERLAP_RATIOS:-1.0}"
WEIGHT_LOAD_COMPUTE_OVERLAP_RATIOS="${WEIGHT_LOAD_COMPUTE_OVERLAP_RATIOS:-1.0}"
PARALLEL_GROUP_KEYS="${PARALLEL_GROUP_KEYS:-pim_weight_load_overlap_ratio,weight_load_compute_overlap_ratio,model,prefill_len,decode_len,batch}"
COMBO_WORKERS="${COMBO_WORKERS:-11}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-0}"
MERGE_RESULTS="${MERGE_RESULTS:-1}"
SUMMARY_AFTER_RUN="${SUMMARY_AFTER_RUN:-1}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-./summarize_weight_overlap_sweep.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CONFIG OUTDIR MODE OBJECTIVE ALGOS PIM_WEIGHT_LOAD_OVERLAP_RATIOS WEIGHT_LOAD_COMPUTE_OVERLAP_RATIOS
export PARALLEL_GROUP_KEYS COMBO_WORKERS THREADS_PER_WORKER MERGE_RESULTS PYTHON_BIN

# Forward any optional sweep dimensions already passed via --export, such as:
# MODELS, PREFILLS, DECODES, BATCHES, HARDWARES, TP_QKV, TP_FFN,
# NPU_BACKENDS, FORMAT_OUTER_MAX_ITERS, FORMAT_ND_MARGIN_INIT, etc.

bash ./run_hpc_sweep_weight_suggest.sh "$@"

RESULTS_CSV="${OUTDIR}/results.csv"
if [[ "${SUMMARY_AFTER_RUN}" == "1" && -f "${RESULTS_CSV}" ]]; then
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" \
    --results-csv "${RESULTS_CSV}" \
    --outdir "${OUTDIR}/ratio_summary"
fi

echo "Weight overlap sweep for PIM load and load-vs-compute overlap done."
