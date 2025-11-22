#!/bin/bash

# 直接用一个列表表示 "family-variant 组合"
# MODEL_FAMILY_VARIANTS=(
#   "llama:7b 13b 70b"
#   "mpt:7b 30b"
#   "palm:8b 62b 540b"
# )

# MODEL_FAMILY_VARIANTS=(
#   "palm:8b"
# )

MODEL_FAMILY_VARIANTS=(
  "llama:7b"
)

PREFILLS=(128 1024)
DECODES=(128 1024)

ALGOS=("heft")
BASELINES=("ianus" "neupims" "attacc" "facil" "pd" "weights_on_pim" "attn_on_pim")

STRIDE=16
DTYPE="INT8"
BATCH=1
CONFIG_FILE="./examples/evaluate_len_sweep_config.json"
BASE_OUTPUT_DIR="./output/baseline_sweep_kv_cache_v2_pcie_64gb"

echo "Starting model evaluation sweep..."
echo "===================================="

for ENTRY in "${MODEL_FAMILY_VARIANTS[@]}"; do
  FAMILY="${ENTRY%%:*}"        # 冒号前面
  VARIANTS="${ENTRY#*:}"       # 冒号后面的一整串

  for VARIANT in ${VARIANTS}; do
    for S in "${PREFILLS[@]}"; do
      for T in "${DECODES[@]}"; do

        echo ""
        echo "--- Running: Family=${FAMILY}, Variant=${VARIANT}, Prefill=${S}, Decode=${T} ---"

        python main.py evaluate \
          --config "${CONFIG_FILE}" \
          --result_dir "${BASE_OUTPUT_DIR}" \
          --algo "$(IFS=,; echo "${ALGOS[*]}")" \
          --baselines "$(IFS=,; echo "${BASELINES[*]}")" \
          --model_family "${FAMILY}" \
          --model_variant "${VARIANT}" \
          --dtype "${DTYPE}" \
          --batch "${BATCH}" \
          --prefill_len "${S}" \
          --decode_len "${T}" \
          --decode_sample_stride "${STRIDE}" \
          --debug

        if [ $? -ne 0 ]; then
          echo "!!!!!! ERROR: Failed on ${FAMILY}-${VARIANT} with S=${S}, T=${T}. Exiting. !!!!!!"
          exit 1
        fi

      done
    done
  done
done

echo "===================================="
echo "All model evaluations completed successfully."