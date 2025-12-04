#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# =========================
# Single source of truth
# =========================
CONFIG_FILE="${CONFIG_FILE:-./examples/evaluate_len_sweep_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/lens_eval_sweep}"

# Sweep dims
MODEL_FAMILY_VARIANTS=(
  "mixtral:8x7b"
  "palm:8b"
  "qwen:1.8b"
  # llama:7b
)

PREFILLS=(128 1024)
DECODES=(128 1024)

# Hardware sweep (edit here, or use --hardware_glob)
HARDWARE_CONFIGS=(
  ./examples/hardware_config_scale_down_pima_double.json
  ./examples/hardware_config_scale_down_pima.json
  # ./examples/hardware_config_pimd.json
  # ./examples/hardware_config_pima.json
)

# Run knobs
STRIDE="${STRIDE:-64}"
DTYPE="${DTYPE:-int8}"
BATCH="${BATCH:-1}"

FAST=1
DEBUG=1
HARDWARE_GLOB=""

usage() {
  cat <<EOF
Usage:
  bash sweep_models_lens_evaluate.sh [options]

Options:
  --config <path>         JSON config path (default: ${CONFIG_FILE})
  --output_root <dir>     Output root (default: ${OUTPUT_ROOT})
  --stride <int>          decode_sample_stride (default: ${STRIDE})
  --dtype <str>           dtype (default: ${DTYPE})
  --batch <int>           batch (default: ${BATCH})
  --hardware_glob <glob>  Override HARDWARE_CONFIGS by glob, e.g. "./examples/hardware_*.json"
  --fast                  Enable fast mode (adds --fast_mode)
  --debug                 Enable --debug
  -h, --help              Show help

Notes:
  - FAST MODE is controlled ONLY by presence of --fast (store_true).
EOF
}

# =========================
# Parse args
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)       CONFIG_FILE="$2"; shift 2 ;;
    --output_root)  OUTPUT_ROOT="$2"; shift 2 ;;
    --stride)       STRIDE="$2"; shift 2 ;;
    --dtype)        DTYPE="$2"; shift 2 ;;
    --batch)        BATCH="$2"; shift 2 ;;
    --hardware_glob) HARDWARE_GLOB="$2"; shift 2 ;;
    --fast)         FAST=1; shift ;;
    --debug)        DEBUG=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# normalize dtype to lower (safer with internal maps)
DTYPE="$(printf "%s" "$DTYPE" | tr '[:upper:]' '[:lower:]')"

if [[ -n "$HARDWARE_GLOB" ]]; then
  HARDWARE_CONFIGS=( $HARDWARE_GLOB )
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[FATAL] config not found: $CONFIG_FILE" >&2
  exit 2
fi

if (( ${#HARDWARE_CONFIGS[@]} == 0 )); then
  echo "[FATAL] no hardware_json found. Set HARDWARE_CONFIGS or use --hardware_glob" >&2
  exit 2
fi

# =========================
# Pretty banners
# =========================
BOLD=$'\033[1m'
RESET=$'\033[0m'
RED=$'\033[1;31m'
GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[1;36m'

if (( FAST )); then
  printf "%s\n" "${YELLOW}${BOLD}███████  FAST MODE: ON  ███████${RESET}"
  printf "%s\n" "${YELLOW}${BOLD}Trace simulations disabled (FLOPs/bw estimates only)${RESET}"
else
  printf "%s\n" "${GREEN}${BOLD}███████  FAST MODE: OFF ███████${RESET}"
  printf "%s\n" "${GREEN}${BOLD}Full simulations enabled${RESET}"
fi
echo "Config      : ${CONFIG_FILE}"
echo "Output root : ${OUTPUT_ROOT}"
echo "Stride      : ${STRIDE}"
echo "DType/Batch : ${DTYPE} / ${BATCH}"
echo "Hardwares   : ${#HARDWARE_CONFIGS[@]} file(s)"
echo "Models      : ${#MODEL_FAMILY_VARIANTS[@]} family entry(s)"
echo "===================================="

run_one() {
  local hw_json="$1"
  local family="$2"
  local variant="$3"
  local S="$4"
  local T="$5"

  local hw_stem
  hw_stem="$(basename "$hw_json" .json)"
  hw_stem="${hw_stem#hardware_config_}"


  # IMPORTANT: keep output separated by hardware + stride to avoid overwrites
  local base_out="${OUTPUT_ROOT}/hw_${hw_stem}/st${STRIDE}"
  local expected_dir="${base_out}/${family}_${variant}_${DTYPE}_b${BATCH}"

  printf "\n%s\n" "${CYAN}${BOLD}--- HW=${hw_stem} | ${family}:${variant} | S=${S} T=${T} | stride=${STRIDE} | dtype=${DTYPE} b=${BATCH} ---${RESET}"
  if (( FAST )); then
    printf "%s\n" "${YELLOW}${BOLD}[FAST MODE ACTIVE]${RESET}"
  else
    printf "%s\n" "${GREEN}${BOLD}[FAST MODE INACTIVE]${RESET}"
  fi
  echo "Expected result_dir: ${expected_dir}"

  cmd=(
    python main.py evaluate
    --config "${CONFIG_FILE}"
    --result_dir "${base_out}"
    --hardware_json "${hw_json}"
    --model_family "${family}"
    --model_variant "${variant}"
    --dtype "${DTYPE}"
    --batch "${BATCH}"
    --prefill_len "${S}"
    --decode_len "${T}"
    --decode_sample_stride "${STRIDE}"
  )

  if (( DEBUG )); then cmd+=(--debug); fi
  if (( FAST )); then cmd+=(--fast_mode); fi

  if ! "${cmd[@]}"; then
    printf "%s\n" "${RED}${BOLD}!!!!!! ERROR: Failed on HW=${hw_stem} ${family}-${variant} S=${S} T=${T} !!!!!!${RESET}"
    exit 1
  fi
}

for hw_json in "${HARDWARE_CONFIGS[@]}"; do
  if [[ ! -f "$hw_json" ]]; then
    echo "[WARN] hardware_json not found, skip: $hw_json"
    continue
  fi

  for entry in "${MODEL_FAMILY_VARIANTS[@]}"; do
    family="${entry%%:*}"
    variants="${entry#*:}"

    for variant in ${variants}; do
      for S in "${PREFILLS[@]}"; do
        for T in "${DECODES[@]}"; do
          run_one "$hw_json" "$family" "$variant" "$S" "$T"
        done
      done
    done
  done
done

echo "===================================="
printf "%s\n" "${GREEN}${BOLD}All sweeps completed successfully.${RESET}"
