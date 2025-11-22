#!/bin/bash
# 扫描 weight-suggest（算法格式建议）在不同 S/T/stride 与模型变体下的结果。
# 只跑 weight-suggest，不做 baseline 对比；每次运行会写出：
#   - weight_storage_suggestion_{SxT[_stK][_tag]}.json
#   - all_passes_{SxT[_stK][_tag]}.json
#   - best_summary_{SxT[_stK][_tag]}.json
# 并将模拟日志 pim_sim_{SxT[_stK][_tag]}.txt 放在同一目录中，避免覆盖。

set -e

MODEL_FAMILY_VARIANTS=(
  # "palm:8b"
  "llama:7b"
  # "mpt:7b"
)

PREFILLS=(128 1024)
DECODES=(128 1024)
STRIDE=16

DTYPE="INT8"
BATCH=1
CONFIG_FILE="./examples/weight_suggest_config.json"
BASE_OUTPUT_DIR="./output/weight_sweep_kv_cache_v2_pcie_64gb"

# 可选：给一组实验附加一个短 tag，方便与其它 sweep 区分（会拼到文件名后缀）
TAG_SUFFIX=""  # 比如 TAG_SUFFIX="g128_w8"

echo "Starting WEIGHT-SUGGEST sweep..."
echo "===================================="

for ENTRY in "${MODEL_FAMILY_VARIANTS[@]}"; do
  FAMILY="${ENTRY%%:*}"
  VARIANTS="${ENTRY#*:}"

  for VARIANT in ${VARIANTS}; do
    for S in "${PREFILLS[@]}"; do
      for T in "${DECODES[@]}"; do

        echo ""
        echo "--- Running: Family=${FAMILY}, Variant=${VARIANT}, S=${S}, T=${T}, stride=${STRIDE} ---"

        # 结果目录遵循 main.py 的 _build_result_dir：
        # <BASE_OUTPUT_DIR>/<family>_<variant>_<dtype>_b<batch>
        python main.py weight-suggest --config "${CONFIG_FILE}" --result_dir "${BASE_OUTPUT_DIR}" --model_family "${FAMILY}" --model_variant "${VARIANT}" --dtype "${DTYPE}" --batch "${BATCH}"           --prefill_len "${S}"           --decode_len "${T}"           --decode_sample_stride "${STRIDE}"           --tag "${TAG_SUFFIX}"           --debug

        if [ $? -ne 0 ]; then
          echo "!!!!!! ERROR: Failed on ${FAMILY}-${VARIANT} with S=${S}, T=${T}. Exiting. !!!!!!"
          exit 1
        fi

      done
    done
  done
done

echo "===================================="
echo "All WEIGHT-SUGGEST sweeps completed successfully."
