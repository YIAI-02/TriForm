#!/bin/bash
#SBATCH -J ds_flash_bifocal_pdffn
#SBATCH -p C064M0256G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set -eo pipefail

source ~/.bashrc
conda activate triform310

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

JOBS="${JOBS:-12}"
CONFIG_FILE="${CONFIG_FILE:-./src/examples/evaluate_deepseek_v4_flash_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/deepseek_pdffn_huawei_4pim}"
MODELS="${MODELS:-deepseek_v4:flash}"
PREFILLS="${PREFILLS:-1024 2048}"
DECODES="${DECODES:-128 512 1024}"
BATCHES="${BATCHES:-4 8 16}"
HARDWARES="${HARDWARES:-./src/examples/hardware_2npu_2pim_huawei.json}"
ALGO="${ALGO:-Bifocal}"
BASELINES="${BASELINES:-PD+FFN}"
TP_QKV="${TP_QKV:-1}"
TP_FFN="${TP_FFN:-1}"
TP_MOE="${TP_MOE:-6}"
KV_PLACE="${KV_PLACE:-pim}"
PD_FFN_KV_PLACE="${PD_FFN_KV_PLACE:-npu}"
KV_PARTITION_DIM="${KV_PARTITION_DIM:-seq}"
KV_SEQ_SHARDS="${KV_SEQ_SHARDS:-2}"
DECODE_SAMPLE_STRIDE="${DECODE_SAMPLE_STRIDE:-${SAMPLE_STRIDE:-2}}"
DECODE_PLAN_REFRESH_STRIDE="${DECODE_PLAN_REFRESH_STRIDE:-${PLAN_REFRESH_STRIDE:-2}}"
DTYPE="${DTYPE:-fp8}"

run_sweep() {
  local sweep_script="$1"
  bash "${sweep_script}" \
    --config "${CONFIG_FILE}" \
    --output_root "${OUTPUT_ROOT}" \
    --models "${MODELS}" \
    --prefills "${PREFILLS}" \
    --decodes "${DECODES}" \
    --batches "${BATCHES}" \
    --hardwares "${HARDWARES}" \
    --algo "${ALGO}" \
    --baselines "${BASELINES}" \
    --tp_qkv "${TP_QKV}" \
    --tp_ffn "${TP_FFN}" \
    --tp_moe "${TP_MOE}" \
    --kv_place "${KV_PLACE}" \
    --pd_ffn_kv_place "${PD_FFN_KV_PLACE}" \
    --kv_partition_dim "${KV_PARTITION_DIM}" \
    --kv_seq_shards "${KV_SEQ_SHARDS}" \
    --decode_sample_stride "${DECODE_SAMPLE_STRIDE}" \
    --decode_plan_refresh_stride "${DECODE_PLAN_REFRESH_STRIDE}" \
    --dtype "${DTYPE}" \
    --jobs "${JOBS}"
}

echo "Sweep parallel jobs : ${JOBS}"
echo "Config              : ${CONFIG_FILE}"
echo "Output root         : ${OUTPUT_ROOT}"
echo "Models              : ${MODELS}"
echo "Batches             : ${BATCHES}"
echo "Hardware            : ${HARDWARES}"
echo "Compare             : ${ALGO} vs ${BASELINES}"
echo "KV place            : ${KV_PLACE}"
echo "PD+FFN KV place     : ${PD_FFN_KV_PLACE}"

if [[ -f "${SUBMIT_DIR}/commands/sweep_models_npu_aim_evaluate.sh" ]]; then
  cd "${SUBMIT_DIR}"
  run_sweep "${SUBMIT_DIR}/commands/sweep_models_npu_aim_evaluate.sh"

elif [[ -f "${SUBMIT_DIR}/sweep_models_npu_aim_evaluate.sh" ]]; then
  cd "$(cd "${SUBMIT_DIR}/.." && pwd)"
  run_sweep "${SUBMIT_DIR}/sweep_models_npu_aim_evaluate.sh"

else
  echo "[FATAL] sweep_models_npu_aim_evaluate.sh not found." >&2
  echo "SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 2
fi
