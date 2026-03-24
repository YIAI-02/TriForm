#!/usr/bin/env bash
set -uo pipefail
shopt -s nullglob

# =========================
# Single source of truth
# =========================
CONFIG_FILE="${CONFIG_FILE:-./examples/evaluate_len_sweep_config_npu_8.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/exp2/8shards}"

# Sweep dims
MODEL_FAMILY_VARIANTS=(
  # "mixtral:8x7b"
  # "palm:62b"
  # "qwen:7b"
  # "qwen:14b"
  "llama:7b"
  "llama:13b"
  "llama:70b"
)

PREFILLS=(512)
DECODES=(128 256 512 1024)

# Hardware sweep (edit here, or use --hardware_glob)
HARDWARE_CONFIGS=(
  # ./examples/hardware_1npu_2aim.json
  # ./examples/hardware_1npu_2aim_star.json
  # ./examples/hardware_1npu_4aim.json
  ./examples/hardware_1npu_8aim.json
  # ./examples/hardware_1npu_4aim_star.json
)

# Run knobs
DECODE_SAMPLE_STRIDE="${DECODE_SAMPLE_STRIDE:-${SAMPLE_STRIDE:-${STRIDE:-8}}}"
DECODE_PLAN_REFRESH_STRIDE="${DECODE_PLAN_REFRESH_STRIDE:-${PLAN_REFRESH_STRIDE:-${STRIDE:-8}}}"
DTYPE="${DTYPE:-fp16}"
BATCHES_STR="${BATCHES:-${BATCH:-"1 4"}}"

declare -a BATCHES

PIM_FAST=0
DEBUG=0
HARDWARE_GLOB=""
JOBS="${JOBS:-}"

declare -a FAILURES=()
declare -a SUCCESSES=()
declare -a RUN_PIDS=()
declare -a RUN_LABELS=()

detect_cpu_count() {
  local count
  if command -v sysctl >/dev/null 2>&1; then
    count="$(sysctl -n hw.ncpu 2>/dev/null || printf '1')"
  elif command -v nproc >/dev/null 2>&1; then
    count="$(nproc 2>/dev/null || printf '1')"
  else
    count="1"
  fi
  printf '%s' "${count:-1}"
}

reap_pid() {
  local idx="$1"
  local pid="${RUN_PIDS[idx]}"
  local label="${RUN_LABELS[idx]}"

  if wait "$pid"; then
    SUCCESSES+=("$label")
  else
    FAILURES+=("$label")
    printf "%s\n" "${RED}${BOLD}!!!!!! ERROR: Failed on ${label} !!!!!!${RESET}"
  fi

  unset 'RUN_PIDS[idx]'
  unset 'RUN_LABELS[idx]'
  RUN_PIDS=("${RUN_PIDS[@]}")
  RUN_LABELS=("${RUN_LABELS[@]}")
}

wait_for_slot() {
  while (( ${#RUN_PIDS[@]} >= JOBS )); do
    reap_pid 0
  done
}

wait_for_all() {
  while (( ${#RUN_PIDS[@]} )); do
    reap_pid 0
  done
}

usage() {
  cat <<EOF2
Usage:
  bash $(basename "$0") [options]

Options:
  --config <path>                     JSON config path (default: ${CONFIG_FILE})
  --output_root <dir>                 Output root (default: ${OUTPUT_ROOT})
  --sample_stride <int>               decode_sample_stride (default: ${DECODE_SAMPLE_STRIDE})
  --decode_sample_stride <int>        Same as --sample_stride
  --plan_refresh_stride <int>         decode_plan_refresh_stride (default: ${DECODE_PLAN_REFRESH_STRIDE})
  --decode_plan_refresh_stride <int>  Same as --plan_refresh_stride
  --stride <int>                      Backward-compatible alias: set BOTH strides to the same value
  --dtype <str>                       dtype (default: ${DTYPE})
  --batch <int>                       single batch, or quoted list if you want to reuse old behavior
  --batches "a b c"                   override batch list (space separated)
  --jobs <int>                        Parallel runs (default: auto-detect cores)
  --hardware_glob <glob>              Override HARDWARE_CONFIGS by glob, e.g. "./examples/hardware_*.json"
  --pim_fast                          Enable PIM fast mode (default: ${PIM_FAST})
  --debug                             Enable --debug
  -h, --help                          Show help
EOF2
}

# =========================
# Parse args
# =========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)                        CONFIG_FILE="$2"; shift 2 ;;
    --output_root)                   OUTPUT_ROOT="$2"; shift 2 ;;
    --sample_stride|--decode_sample_stride)
                                     DECODE_SAMPLE_STRIDE="$2"; shift 2 ;;
    --plan_refresh_stride|--decode_plan_refresh_stride)
                                     DECODE_PLAN_REFRESH_STRIDE="$2"; shift 2 ;;
    --stride)
                                     DECODE_SAMPLE_STRIDE="$2"; DECODE_PLAN_REFRESH_STRIDE="$2"; shift 2 ;;
    --dtype)                         DTYPE="$2"; shift 2 ;;
    --batch)                         BATCHES_STR="$2"; shift 2 ;;
    --batches)                       BATCHES_STR="$2"; shift 2 ;;
    --jobs)                          JOBS="$2"; shift 2 ;;
    --hardware_glob)                 HARDWARE_GLOB="$2"; shift 2 ;;
    --pim_fast)                      PIM_FAST=1; shift ;;
    --debug)                         DEBUG=1; shift ;;
    -h|--help)                       usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# normalize dtype to lower (safer with internal maps)
DTYPE="$(printf "%s" "$DTYPE" | tr '[:upper:]' '[:lower:]')"
# Parse batches string into an array (handles quoted space-separated lists)
read -r -a BATCHES <<< "$BATCHES_STR"

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

if ! [[ "$DECODE_SAMPLE_STRIDE" =~ ^[0-9]+$ ]] || (( DECODE_SAMPLE_STRIDE < 1 )); then
  echo "[FATAL] invalid decode_sample_stride: $DECODE_SAMPLE_STRIDE" >&2
  exit 2
fi

if ! [[ "$DECODE_PLAN_REFRESH_STRIDE" =~ ^[0-9]+$ ]] || (( DECODE_PLAN_REFRESH_STRIDE < 0 )); then
  echo "[FATAL] invalid decode_plan_refresh_stride: $DECODE_PLAN_REFRESH_STRIDE" >&2
  exit 2
fi

if [[ -z "$JOBS" ]]; then
  JOBS="$(detect_cpu_count)"
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS < 1 )); then
  echo "[FATAL] invalid jobs value: $JOBS" >&2
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


if (( PIM_FAST )); then
  printf "%s\n" "${YELLOW}${BOLD}███████  PIM FAST MODE: ON  ███████${RESET}"
else
  printf "%s\n" "${GREEN}${BOLD}███████  PIM FAST MODE: OFF ███████${RESET}"
fi
echo "Config                : ${CONFIG_FILE}"
echo "Output root           : ${OUTPUT_ROOT}"
echo "Sample stride         : ${DECODE_SAMPLE_STRIDE}"
echo "Plan refresh stride   : ${DECODE_PLAN_REFRESH_STRIDE}"
echo "DType/Batch           : ${DTYPE} / ${BATCHES[*]}"
echo "Parallel jobs         : ${JOBS}"
echo "Hardwares             : ${#HARDWARE_CONFIGS[@]} file(s)"
echo "Models                : ${#MODEL_FAMILY_VARIANTS[@]} family entry(s)"
echo "===================================="

run_one() {
  wait_for_slot

  local hw_json="$1"
  local family="$2"
  local variant="$3"
  local S="$4"
  local T="$5"
  local batch="$6"

  local hw_stem
  hw_stem="$(basename "$hw_json" .json)"
  hw_stem="${hw_stem#hardware_config_}"

  local base_out="${OUTPUT_ROOT}/hw_${hw_stem}/sst${DECODE_SAMPLE_STRIDE}_rst${DECODE_PLAN_REFRESH_STRIDE}"
  local expected_dir="${base_out}/${family}_${variant}_${DTYPE}_b${batch}_s${DECODE_SAMPLE_STRIDE}"

  printf "\n%s\n" "${CYAN}${BOLD}--- HW=${hw_stem} | ${family}:${variant} | S=${S} T=${T} | sample_stride=${DECODE_SAMPLE_STRIDE} | refresh_stride=${DECODE_PLAN_REFRESH_STRIDE} | dtype=${DTYPE} b=${batch} ---${RESET}"

  if (( PIM_FAST )); then printf "%s\n" "${YELLOW}${BOLD}[PIM FAST MODE ACTIVE]${RESET}"; else printf "%s\n" "${GREEN}${BOLD}[PIM FAST MODE INACTIVE]${RESET}"; fi
  echo "Expected result_dir   : ${expected_dir}"

  cmd=(
    python main.py evaluate
    --config "${CONFIG_FILE}"
    --result_dir "${base_out}"
    --hardware_json "${hw_json}"
    --model_family "${family}"
    --model_variant "${variant}"
    --dtype "${DTYPE}"
    --batch "${batch}"
    --prefill_len "${S}"
    --decode_len "${T}"
    --decode_sample_stride "${DECODE_SAMPLE_STRIDE}"
    --decode_plan_refresh_stride "${DECODE_PLAN_REFRESH_STRIDE}"
  )

  if (( DEBUG )); then cmd+=(--debug); fi
  if (( PIM_FAST )); then cmd+=(--pim_fast_mode); fi

  (
    "${cmd[@]}"
  ) &

  local pid=$!
  RUN_PIDS+=("$pid")
  RUN_LABELS+=("HW=${hw_stem} ${family}:${variant} S=${S} T=${T} b=${batch} sst=${DECODE_SAMPLE_STRIDE} rst=${DECODE_PLAN_REFRESH_STRIDE}")
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
      for batch in "${BATCHES[@]}"; do
        for S in "${PREFILLS[@]}"; do
          for T in "${DECODES[@]}"; do
            run_one "$hw_json" "$family" "$variant" "$S" "$T" "$batch"
          done
        done
      done
    done
  done
done

wait_for_all

echo "===================================="
total_runs=$(( ${#SUCCESSES[@]} + ${#FAILURES[@]} ))
echo "Runs attempted : ${total_runs}"
echo "Runs succeeded : ${#SUCCESSES[@]}"
echo "Runs failed    : ${#FAILURES[@]}"

if (( ${#FAILURES[@]} )); then
  printf "%s\n" "${RED}${BOLD}Failures detected during sweep:${RESET}"
  for item in "${FAILURES[@]}"; do
    printf "  - %s\n" "$item"
  done
  exit 1
fi

printf "%s\n" "${GREEN}${BOLD}All sweeps completed successfully.${RESET}"
