#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Configurable arguments (with sensible defaults)
# ------------------------------------------------------------
# Models to run: one or more from {mistral, mpt, qwen2}
# - Accepts comma-separated or space-separated list
# - "all" expands to all three models
MODELS="${MODELS:-mpt}"         # default single model "mpt"
DRAM="${DRAM:-DDR3}"            # default DDR3
RAMULATOR_BIN="${RAMULATOR_BIN:-./ramulator}"  # relative to measurements/dram/
OUT_ND_DIR="${OUT_ND_DIR:-./out_nd}"
RUN_OUT_DIR="${RUN_OUT_DIR:-./out_runs}"

# Allow -m / -d / -j / -b flags to override env vars
usage() {
  cat <<EOF
Usage: $(basename "$0") [-m "mistral,mpt" | --models "..."] [-d DRAM | --dram DRAM] [-b BIN | --bin BIN]
  DRAM in {DDR3, DDR4, GDDR5, LPDDR3, LPDDR4, HBM}, default DDR3.
  Models in {mistral, mpt, qwen2}. You can pass "all" to run all three.
  Examples:
    $(basename "$0") -m mpt -d DDR3
    MODELS="mistral,qwen2" DRAM=DDR4
EOF
}

# Parse simple flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--models) MODELS="$2"; shift 2;;
    -d|--dram)   DRAM="$2"; shift 2;;
    -b|--bin)    RAMULATOR_BIN="$2"; shift 2;;
    -h|--help)   usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# Normalize model list
if [ "$(echo "$MODELS" | tr '[:upper:]' '[:lower:]')" = "all" ]; then
  MODELS="mistral mpt qwen2"
fi
# Split comma/space
IFS=', ' read -r -a MODEL_ARR <<< "$MODELS"

# Validate DRAM option
case "$DRAM" in
  DDR3|DDR4|GDDR5|LPDDR3|LPDDR4|HBM) ;;
  *) echo "ERROR: Unsupported DRAM '$DRAM'"; exit 1;;
esac

echo "[INFO] MODELS     = ${MODEL_ARR[*]}"
echo "[INFO] DRAM       = ${DRAM}"
echo "[INFO] BIN        = ${RAMULATOR_BIN}"
echo "[INFO] OUT_ND_DIR = ${OUT_ND_DIR}"
echo "[INFO] RUN_OUT_DIR= ${RUN_OUT_DIR}"

# Resolve config path for ramulator v1
CFG_PATH="../../submodules/ramulator/configs/${DRAM}-config.cfg"
if [[ ! -f "$CFG_PATH" ]]; then
  echo "ERROR: Ramulator config not found: $CFG_PATH"
  exit 1
fi

# Step 0: ensure we are in measurements/dram/
if [[ ! -f "./01_generate_traces.py" ]] || [[ ! -d "../../configs" ]]; then
  echo "ERROR: Please run this script from measurements/dram/ directory."
  exit 1
fi

# ------------------------------------------------------------
# Step 1: Generate ND (row-major) traces for each model
# ------------------------------------------------------------
TRACE_LISTS=()
AMAP_LIST=()

for MODEL in "${MODEL_ARR[@]}"; do
  SHAPE_JSON="../../configs/${MODEL}_shape.json"
  OUTDIR_MODEL="${OUT_ND_DIR}/${MODEL}_shape"
  mkdir -p "$OUTDIR_MODEL"

  if [[ ! -f "$SHAPE_JSON" ]]; then
    echo "ERROR: Shape JSON not found: $SHAPE_JSON"
    exit 1
  fi

  echo "[STEP1] Generating traces for model=${MODEL} -> ${OUTDIR_MODEL}"
  python3 ./01_generate_traces.py \
    --shapes "$SHAPE_JSON" \
    --outdir "$OUT_ND_DIR" \
    --dtype-bytes 2 \
    --cacheline-bytes 64

  TL="${OUTDIR_MODEL}/trace_list.txt"
  AM="${OUTDIR_MODEL}/address_map.csv"
  if [[ ! -f "$TL" ]]; then
    echo "ERROR: trace_list not found after step1: $TL"
    exit 1
  fi
  if [[ ! -f "$AM" ]]; then
    echo "ERROR: address_map not found after step1: $AM"
    exit 1
  fi
  TRACE_LISTS+=("$TL")
  AMAP_LIST+=("$AM")
done

# ------------------------------------------------------------
# Step 2: Feed traces to Ramulator (v1 CLI)
# ------------------------------------------------------------
echo "[STEP2] Running Ramulator with DRAM=${DRAM}"
python3 ./02_run_ramulator.py \
  --ramulator-bin "$RAMULATOR_BIN" \
  --config "$CFG_PATH" \
  --trace-lists "${TRACE_LISTS[@]}" \
  --outdir "$RUN_OUT_DIR"

# Determine stats directory for step 3
if   [[ -d "${RUN_OUT_DIR}/stats" ]]; then
  STATS_DIR="${RUN_OUT_DIR}/stats"
elif [[ -d "${RUN_OUT_DIR}/outlog/stats" ]]; then
  STATS_DIR="${RUN_OUT_DIR}/outlog/stats"
elif [[ -d "${RUN_OUT_DIR}/logs" ]]; then
  STATS_DIR="${RUN_OUT_DIR}/logs"
else
  # fallback to run_out root (not typical)
  STATS_DIR="${RUN_OUT_DIR}"
fi
echo "[INFO] stats-dir = ${STATS_DIR}"

# ------------------------------------------------------------
# Step 3: Parse and fit per-model (read only)
# ------------------------------------------------------------
echo "[STEP3] Parsing stats and fitting per model"
python3 ./03_parse_and_fit.py \
  --run-outdir "$RUN_OUT_DIR" \
  --stats-dir "$STATS_DIR" \
  --address-map "${AMAP_LIST[@]}"

echo "[DONE] Results:"
echo "  - Per-model summaries & fits: ${RUN_OUT_DIR}/fit_results/<model>/"
