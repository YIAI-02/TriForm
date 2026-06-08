#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
set -uo pipefail
shopt -s nullglob

CONFIG_FILE="${CONFIG_FILE:-./src/examples/evaluate_deepseek_v4_flash_config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/deepseek_pdffn_huawei_4pim}"

MODELS_STR="${MODELS:-${MODEL_FAMILY_VARIANTS:-deepseek_v4:flash}}"
PREFILLS_STR="${PREFILLS:-1024 2048 4096}"
DECODES_STR="${DECODES:-128 512 1024}"
BATCHES_STR="${BATCHES:-${BATCH:-1 4 8}}"
HARDWARES_STR="${HARDWARES:-./src/examples/hardware_1npu_4pim_huawei.json}"

# DeepSeek-V4: attention/KV has shared KV heads; do not use legacy head TP.
# Use tp_moe to expose selected routed experts / per-expert FFN shards as parallel work.
TP_QKV="${TP_QKV:-1}"
TP_FFN="${TP_FFN:-1}"
TP_MOES_STR="${TP_MOES:-${TP_MOE:-6}}"   # DeepSeek-V4 top_k=6; try "6 12" on 8-PIM if you want extra FFN splitting.
KV_PLACE="${KV_PLACE:-pim}"              # force KV to PIM when capacity is feasible; falls back safely if not.
PD_FFN_KV_PLACE="${PD_FFN_KV_PLACE:-npu}" # PD+FFN baseline only: keep non-FFN attention/KV on NPU by default.
KV_PARTITION_DIM="${KV_PARTITION_DIM:-seq}" # DeepSeek-V4: seq/context sharding gives same-layer PIM parallelism.
KV_SEQ_SHARDS="${KV_SEQ_SHARDS:-4}"        # empty => auto, normally number of PIM devices.
ALGO="${ALGO:-}"                          # optional override, e.g. HEFT or Bifocal
BASELINES="${BASELINES:-PD+FFN}"          # optional override, e.g. "PD,AF" or empty string

# Run knobs
DECODE_SAMPLE_STRIDE="${DECODE_SAMPLE_STRIDE:-${SAMPLE_STRIDE:-${STRIDE:-2}}}"
DECODE_PLAN_REFRESH_STRIDE="${DECODE_PLAN_REFRESH_STRIDE:-${PLAN_REFRESH_STRIDE:-${STRIDE:-2}}}"
DTYPE="${DTYPE:-fp8}"
PIM_FAST=1
DEBUG=0

# Arrays populated after argument parsing.
declare -a MODEL_FAMILY_VARIANTS=()
declare -a PREFILLS=()
declare -a DECODES=()
declare -a BATCH_LIST=()
declare -a HARDWARE_CONFIGS=()
declare -a TP_MOE_LIST=()

declare -a FAILURES=()
declare -a SUCCESSES=()
declare -a RUN_PIDS=()
declare -a RUN_LABELS=()

split_words() {
  local input="$1"
  local -n out_ref="$2"
  # shellcheck disable=SC2206
  out_ref=( ${input} )
}

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
  --models "family:variant ..."       Model sweep list (default: ${MODELS_STR})
  --prefills "a b c"                  Prefill length sweep (default: ${PREFILLS_STR})
  --decodes "a b c"                   Decode length sweep (default: ${DECODES_STR})
  --batches "a b c"                   Batch sweep (default: ${BATCHES_STR})
  --hardwares "a.json b.json"         Hardware sweep list (default: ${HARDWARES_STR})
  --hardware_glob <glob>              Override hardware list by glob
  --tp_qkv <int>                      QKV/head TP; keep 1 for DeepSeek-V4 (default: ${TP_QKV})
  --tp_ffn <int>                      Dense FFN TP; normally 1 for DeepSeek-V4 (default: ${TP_FFN})
  --tp_moe <int>                      Single MoE shard count (default: ${TP_MOES_STR})
  --tp_moes "a b c"                   Sweep MoE shard counts
  --kv_place <host|pim|npu>           Force KV placement if feasible (default: ${KV_PLACE})
  --pd_ffn_kv_place <host|pim|npu|follow> KV placement used only by PD+FFN baseline (default: ${PD_FFN_KV_PLACE})
  --kv_partition_dim <seq|layer|kv_head> KV/PIM partition axis (default: ${KV_PARTITION_DIM})
  --kv_seq_shards <int>                 DeepSeek-V4 sequence shards; empty means auto by PIM count
  --algo <name-or-list>                 Override config algo list
  --baselines <name-or-list>            Override config baseline list; use "" to disable
  --sample_stride <int>               decode_sample_stride (default: ${DECODE_SAMPLE_STRIDE})
  --decode_sample_stride <int>        Same as --sample_stride
  --plan_refresh_stride <int>         decode_plan_refresh_stride (default: ${DECODE_PLAN_REFRESH_STRIDE})
  --decode_plan_refresh_stride <int>  Same as --plan_refresh_stride
  --stride <int>                      Set BOTH strides to the same value
  --dtype <str>                       dtype (default: ${DTYPE})
  --batch <int-or-list>               Backward-compatible alias for --batches
  --jobs <int>                        Parallel runs (default: auto-detect cores)
  --pim_fast                          Keep PIM fast mode enabled (current runtime coerces fast mode)
  --debug                             Enable --debug
  -h, --help                          Show help
EOF2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)                        CONFIG_FILE="$2"; shift 2 ;;
    --output_root)                   OUTPUT_ROOT="$2"; shift 2 ;;
    --models)                        MODELS_STR="$2"; shift 2 ;;
    --prefills)                      PREFILLS_STR="$2"; shift 2 ;;
    --decodes)                       DECODES_STR="$2"; shift 2 ;;
    --batch|--batches)               BATCHES_STR="$2"; shift 2 ;;
    --hardwares)                     HARDWARES_STR="$2"; shift 2 ;;
    --hardware_glob)                 HARDWARES_STR="$2"; shift 2 ;;
    --tp_qkv)                        TP_QKV="$2"; shift 2 ;;
    --tp_ffn)                        TP_FFN="$2"; shift 2 ;;
    --tp_moe)                        TP_MOES_STR="$2"; shift 2 ;;
    --tp_moes)                       TP_MOES_STR="$2"; shift 2 ;;
    --kv_place)                      KV_PLACE="$2"; shift 2 ;;
    --pd_ffn_kv_place)               PD_FFN_KV_PLACE="$2"; shift 2 ;;
    --kv_partition_dim)              KV_PARTITION_DIM="$2"; shift 2 ;;
    --kv_seq_shards)                 KV_SEQ_SHARDS="$2"; shift 2 ;;
    --algo)                          ALGO="$2"; shift 2 ;;
    --baselines)                     BASELINES="$2"; shift 2 ;;
    --sample_stride|--decode_sample_stride)
                                     DECODE_SAMPLE_STRIDE="$2"; shift 2 ;;
    --plan_refresh_stride|--decode_plan_refresh_stride)
                                     DECODE_PLAN_REFRESH_STRIDE="$2"; shift 2 ;;
    --stride)                        DECODE_SAMPLE_STRIDE="$2"; DECODE_PLAN_REFRESH_STRIDE="$2"; shift 2 ;;
    --dtype)                         DTYPE="$2"; shift 2 ;;
    --jobs)                          JOBS="$2"; shift 2 ;;
    --pim_fast)                      PIM_FAST=1; shift ;;
    --debug)                         DEBUG=1; shift ;;
    -h|--help)                       usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

DTYPE="$(printf "%s" "$DTYPE" | tr '[:upper:]' '[:lower:]')"
KV_PLACE="$(printf "%s" "$KV_PLACE" | tr '[:upper:]' '[:lower:]')"
PD_FFN_KV_PLACE="$(printf "%s" "$PD_FFN_KV_PLACE" | tr '[:upper:]' '[:lower:]')"
KV_PARTITION_DIM="$(printf "%s" "$KV_PARTITION_DIM" | tr '[:upper:]' '[:lower:]')"
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT#./}"
fi

split_words "$MODELS_STR" MODEL_FAMILY_VARIANTS
split_words "$PREFILLS_STR" PREFILLS
split_words "$DECODES_STR" DECODES
split_words "$BATCHES_STR" BATCH_LIST
split_words "$HARDWARES_STR" HARDWARE_CONFIGS
split_words "$TP_MOES_STR" TP_MOE_LIST

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[FATAL] config not found: $CONFIG_FILE" >&2
  exit 2
fi
if (( ${#MODEL_FAMILY_VARIANTS[@]} == 0 )); then
  echo "[FATAL] empty model sweep" >&2
  exit 2
fi
if (( ${#PREFILLS[@]} == 0 || ${#DECODES[@]} == 0 || ${#BATCH_LIST[@]} == 0 || ${#HARDWARE_CONFIGS[@]} == 0 || ${#TP_MOE_LIST[@]} == 0 )); then
  echo "[FATAL] one or more sweep dimensions are empty" >&2
  exit 2
fi
if [[ "$KV_PLACE" != "host" && "$KV_PLACE" != "pim" && "$KV_PLACE" != "npu" ]]; then
  echo "[FATAL] invalid kv_place: $KV_PLACE (expected host|pim|npu)" >&2
  exit 2
fi
if [[ "$PD_FFN_KV_PLACE" != "host" && "$PD_FFN_KV_PLACE" != "pim" && "$PD_FFN_KV_PLACE" != "npu" && "$PD_FFN_KV_PLACE" != "follow" ]]; then
  echo "[FATAL] invalid pd_ffn_kv_place: $PD_FFN_KV_PLACE (expected host|pim|npu|follow)" >&2
  exit 2
fi
case "$KV_PARTITION_DIM" in
  seq|sequence|context|layer|kv_head|head) ;;
  *) echo "[FATAL] invalid kv_partition_dim: $KV_PARTITION_DIM (expected seq|layer|kv_head)" >&2; exit 2 ;;
esac
if [[ -n "$KV_SEQ_SHARDS" ]] && { ! [[ "$KV_SEQ_SHARDS" =~ ^[0-9]+$ ]] || (( KV_SEQ_SHARDS < 1 )); }; then
  echo "[FATAL] invalid kv_seq_shards: $KV_SEQ_SHARDS" >&2
  exit 2
fi
if ! [[ "$DECODE_SAMPLE_STRIDE" =~ ^[0-9]+$ ]] || (( DECODE_SAMPLE_STRIDE < 1 )); then
  echo "[FATAL] invalid decode_sample_stride: $DECODE_SAMPLE_STRIDE" >&2
  exit 2
fi
if ! [[ "$DECODE_PLAN_REFRESH_STRIDE" =~ ^[0-9]+$ ]]; then
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

BOLD=$'\033[1m'
RESET=$'\033[0m'
RED=$'\033[1;31m'
GREEN=$'\033[1;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[1;36m'

printf "%s\n" "${YELLOW}${BOLD}███████  PIM FAST MODE: ON  ███████${RESET}"
echo "Config                : ${CONFIG_FILE}"
echo "Output root           : ${OUTPUT_ROOT}"
echo "Models                : ${MODEL_FAMILY_VARIANTS[*]}"
echo "Prefills              : ${PREFILLS[*]}"
echo "Decodes               : ${DECODES[*]}"
echo "Batches               : ${BATCH_LIST[*]}"
echo "Hardwares             : ${HARDWARE_CONFIGS[*]}"
echo "TP_QKV / TP_FFN       : ${TP_QKV} / ${TP_FFN}"
echo "TP_MOE sweep          : ${TP_MOE_LIST[*]}"
echo "KV place              : ${KV_PLACE}"
echo "PD+FFN KV place       : ${PD_FFN_KV_PLACE}"
echo "KV partition dim      : ${KV_PARTITION_DIM}"
[[ -n "$KV_SEQ_SHARDS" ]] && echo "KV seq shards         : ${KV_SEQ_SHARDS}"
[[ -n "$ALGO" ]] && echo "Algo override         : ${ALGO}"
[[ "$BASELINES" != "__unset__" ]] && echo "Baselines override    : ${BASELINES}"
echo "Sample stride         : ${DECODE_SAMPLE_STRIDE}"
echo "Plan refresh stride   : ${DECODE_PLAN_REFRESH_STRIDE}"
echo "DType                 : ${DTYPE}"
echo "Parallel jobs         : ${JOBS}"
echo "===================================="

run_one() {
  wait_for_slot

  local hw_json="$1"
  local family="$2"
  local variant="$3"
  local S="$4"
  local T="$5"
  local batch="$6"
  local tp_moe="$7"

  local hw_stem
  hw_stem="$(basename "$hw_json" .json)"
  hw_stem="${hw_stem#hardware_config_}"

  local kv_tag="kv${KV_PARTITION_DIM}"
  if [[ -n "$KV_SEQ_SHARDS" ]]; then kv_tag="${kv_tag}_seq${KV_SEQ_SHARDS}"; fi
  local base_out="${OUTPUT_ROOT}/hw_${hw_stem}/${kv_tag}/tpmoe${tp_moe}_sst${DECODE_SAMPLE_STRIDE}_rst${DECODE_PLAN_REFRESH_STRIDE}"
  local expected_dir="${base_out}/${family}_${variant}_${DTYPE}_b${batch}_s${DECODE_SAMPLE_STRIDE}"

  printf "\n%s\n" "${CYAN}${BOLD}--- HW=${hw_stem} | ${family}:${variant} | prefill=${S} decode=${T} | batch=${batch} | tp_moe=${tp_moe} | dtype=${DTYPE} ---${RESET}"
  echo "Expected result_dir   : ${expected_dir}"

  cmd=(
    "${PYTHON_BIN}" "${SRC_DIR}/main.py" evaluate
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
    --tp_qkv "${TP_QKV}"
    --tp_ffn "${TP_FFN}"
    --tp_moe "${tp_moe}"
    --kv_place "${KV_PLACE}"
    --pd_ffn_kv_place "${PD_FFN_KV_PLACE}"
    --kv_partition_dim "${KV_PARTITION_DIM}"
  )

  if [[ -n "$KV_SEQ_SHARDS" ]]; then cmd+=(--kv_seq_shards "${KV_SEQ_SHARDS}"); fi

  if [[ -n "$ALGO" ]]; then cmd+=(--algo "${ALGO}"); fi
  if [[ "$BASELINES" != "__unset__" ]]; then cmd+=(--baselines "${BASELINES}"); fi
  if (( DEBUG )); then cmd+=(--debug); fi
  if (( PIM_FAST )); then cmd+=(--pim_fast_mode); fi

  (
    "${cmd[@]}"
  ) &

  local pid=$!
  RUN_PIDS+=("$pid")
  RUN_LABELS+=("HW=${hw_stem} ${family}:${variant} S=${S} T=${T} b=${batch} tp_moe=${tp_moe} sst=${DECODE_SAMPLE_STRIDE} rst=${DECODE_PLAN_REFRESH_STRIDE}")
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
      for tp_moe in "${TP_MOE_LIST[@]}"; do
        for batch in "${BATCH_LIST[@]}"; do
          for S in "${PREFILLS[@]}"; do
            for T in "${DECODES[@]}"; do
              run_one "$hw_json" "$family" "$variant" "$S" "$T" "$batch" "$tp_moe"
            done
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
echo "Result root           : ${OUTPUT_ROOT}"
echo "Combined result JSONs : find ${OUTPUT_ROOT} -name 'baseline_compare_*.json' | sort"
echo "Per-policy summaries  : find ${OUTPUT_ROOT} -name 'best_summary_*.json' | sort"
