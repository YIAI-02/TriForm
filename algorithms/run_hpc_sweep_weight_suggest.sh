#!/bin/bash
#SBATCH -J sweep_weight_suggest
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
# Required inputs

: <<'COMMENT'
export CONFIG=./examples/weight_suggest_overlap_ratio_base.json
export OUTDIR=./output/ws_overlap_ratio_0329
export MODE=grid
export OBJECTIVE=total
export ALGOS="hefthint"
export COMBO_WORKERS=8
export PARALLEL_GROUP_KEYS="weight_local_load_overlap_ratio,model,prefill_len,decode_len,batch"
export MODELS="llama:7b"
export PREFILLS="128 1024"
export DECODES="128 512 1024"
export BATCHES="1 4"
export WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS="1.0"
export DEBUG=1

rm -rf "$OUTDIR"
sbatch --export=ALL run_hpc_weight_overlap_ratio_weight_suggest.slurm
COMMENT
# ------------------------------------------------------------------
CONFIG="${CONFIG:-}"
if [[ -z "${CONFIG}" ]]; then
  echo "[FATAL] Please set CONFIG=/path/to/weight_suggest_config.json" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-./sweep_weight_suggest_params.py}"
MAIN="${MAIN:-./main.py}"
OUTDIR="${OUTDIR:-./output/ws_hpc}"
MODE="${MODE:-random}"
TRIALS="${TRIALS:-128}"
SEED="${SEED:-0}"
OBJECTIVE="${OBJECTIVE:-total}"
REPEAT="${REPEAT:-1}"
MERGE_RESULTS="${MERGE_RESULTS:-1}"
COMBO_WORKERS="${COMBO_WORKERS:-1}"
PARALLEL_GROUP_KEYS="${PARALLEL_GROUP_KEYS:-model,prefill_len,decode_len,batch}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-0}"

MODELS_STR="${MODELS:-}"
DTYPES_STR="${DTYPES:-}"
BATCHES_STR="${BATCHES:-}"
PREFILLS_STR="${PREFILLS:-}"
DECODES_STR="${DECODES:-}"
DECODE_SAMPLE_STRIDES_STR="${DECODE_SAMPLE_STRIDES:-}"
DECODE_PLAN_REFRESH_STRIDES_STR="${DECODE_PLAN_REFRESH_STRIDES:-}"
HARDWARES_STR="${HARDWARES:-}"
TP_QKV_STR="${TP_QKV:-}"
TP_FFN_STR="${TP_FFN:-}"
ALGOS_STR="${ALGOS:-}"
NPU_BACKENDS_STR="${NPU_BACKENDS:-}"
WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS_STR="${WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS:-}"
FORMAT_OUTER_MAX_ITERS_STR="${FORMAT_OUTER_MAX_ITERS:-}"
FORMAT_INNER_MAX_BLOCKS_STR="${FORMAT_INNER_MAX_BLOCKS:-}"
FORMAT_ND_MARGIN_INIT_STR="${FORMAT_ND_MARGIN_INIT:-}"
FORMAT_ND_MARGIN_DECAY_STR="${FORMAT_ND_MARGIN_DECAY:-}"
FORMAT_ND_MARGIN_MIN_STR="${FORMAT_ND_MARGIN_MIN:-}"
FORMAT_INNER_IMPROVE_EPS_STR="${FORMAT_INNER_IMPROVE_EPS:-}"
FORMAT_OUTER_STOP_EPS_STR="${FORMAT_OUTER_STOP_EPS:-}"
FORMAT_BLOCK_LAYER_SPAN_STR="${FORMAT_BLOCK_LAYER_SPAN:-}"
FORMAT_RELOAD_COUNT_MODE_STR="${FORMAT_RELOAD_COUNT_MODE:-}"

declare -a MODELS_ARR=() DTYPES_ARR=() BATCHES_ARR=() PREFILLS_ARR=() DECODES_ARR=()
declare -a DECODE_SAMPLE_STRIDES_ARR=() DECODE_PLAN_REFRESH_STRIDES_ARR=() HARDWARES_ARR=()
declare -a TP_QKV_ARR=() TP_FFN_ARR=() ALGOS_ARR=() NPU_BACKENDS_ARR=() WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS_ARR=()
declare -a FORMAT_OUTER_MAX_ITERS_ARR=() FORMAT_INNER_MAX_BLOCKS_ARR=() FORMAT_ND_MARGIN_INIT_ARR=()
declare -a FORMAT_ND_MARGIN_DECAY_ARR=() FORMAT_ND_MARGIN_MIN_ARR=() FORMAT_INNER_IMPROVE_EPS_ARR=()
declare -a FORMAT_OUTER_STOP_EPS_ARR=() FORMAT_BLOCK_LAYER_SPAN_ARR=() FORMAT_RELOAD_COUNT_MODE_ARR=()

read -r -a MODELS_ARR <<< "${MODELS_STR}"
read -r -a DTYPES_ARR <<< "${DTYPES_STR}"
read -r -a BATCHES_ARR <<< "${BATCHES_STR}"
read -r -a PREFILLS_ARR <<< "${PREFILLS_STR}"
read -r -a DECODES_ARR <<< "${DECODES_STR}"
read -r -a DECODE_SAMPLE_STRIDES_ARR <<< "${DECODE_SAMPLE_STRIDES_STR}"
read -r -a DECODE_PLAN_REFRESH_STRIDES_ARR <<< "${DECODE_PLAN_REFRESH_STRIDES_STR}"
read -r -a HARDWARES_ARR <<< "${HARDWARES_STR}"
read -r -a TP_QKV_ARR <<< "${TP_QKV_STR}"
read -r -a TP_FFN_ARR <<< "${TP_FFN_STR}"
read -r -a ALGOS_ARR <<< "${ALGOS_STR}"
read -r -a NPU_BACKENDS_ARR <<< "${NPU_BACKENDS_STR}"
read -r -a WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS_ARR <<< "${WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS_STR}"
read -r -a FORMAT_OUTER_MAX_ITERS_ARR <<< "${FORMAT_OUTER_MAX_ITERS_STR}"
read -r -a FORMAT_INNER_MAX_BLOCKS_ARR <<< "${FORMAT_INNER_MAX_BLOCKS_STR}"
read -r -a FORMAT_ND_MARGIN_INIT_ARR <<< "${FORMAT_ND_MARGIN_INIT_STR}"
read -r -a FORMAT_ND_MARGIN_DECAY_ARR <<< "${FORMAT_ND_MARGIN_DECAY_STR}"
read -r -a FORMAT_ND_MARGIN_MIN_ARR <<< "${FORMAT_ND_MARGIN_MIN_STR}"
read -r -a FORMAT_INNER_IMPROVE_EPS_ARR <<< "${FORMAT_INNER_IMPROVE_EPS_STR}"
read -r -a FORMAT_OUTER_STOP_EPS_ARR <<< "${FORMAT_OUTER_STOP_EPS_STR}"
read -r -a FORMAT_BLOCK_LAYER_SPAN_ARR <<< "${FORMAT_BLOCK_LAYER_SPAN_STR}"
read -r -a FORMAT_RELOAD_COUNT_MODE_ARR <<< "${FORMAT_RELOAD_COUNT_MODE_STR}"

DEBUG_ARGS=()
declare -a CMD_COMMON=("${PYTHON_BIN}" "${SWEEP_SCRIPT}" --config "${CONFIG}" --main "${MAIN}" --python "${PYTHON_BIN}" --workdir "." --mode "${MODE}" --trials "${TRIALS}" --seed "${SEED}" --objective "${OBJECTIVE}" --repeat "${REPEAT}")
if [[ "${DEBUG}" == "1" ]]; then
  DEBUG_ARGS+=(--debug)
fi
CMD_COMMON+=("${DEBUG_ARGS[@]}")

if (( ${#MODELS_ARR[@]} )); then
  CMD_COMMON+=(--model "${MODELS_ARR[@]}")
fi
if (( ${#DTYPES_ARR[@]} )); then
  CMD_COMMON+=(--dtype "${DTYPES_ARR[@]}")
fi
if (( ${#BATCHES_ARR[@]} )); then
  CMD_COMMON+=(--batch "${BATCHES_ARR[@]}")
fi
if (( ${#PREFILLS_ARR[@]} )); then
  CMD_COMMON+=(--prefill-len "${PREFILLS_ARR[@]}")
fi
if (( ${#DECODES_ARR[@]} )); then
  CMD_COMMON+=(--decode-len "${DECODES_ARR[@]}")
fi
if (( ${#DECODE_SAMPLE_STRIDES_ARR[@]} )); then
  CMD_COMMON+=(--decode-sample-stride "${DECODE_SAMPLE_STRIDES_ARR[@]}")
fi
if (( ${#DECODE_PLAN_REFRESH_STRIDES_ARR[@]} )); then
  CMD_COMMON+=(--decode-plan-refresh-stride "${DECODE_PLAN_REFRESH_STRIDES_ARR[@]}")
fi
if (( ${#HARDWARES_ARR[@]} )); then
  CMD_COMMON+=(--hardware-json "${HARDWARES_ARR[@]}")
fi
if (( ${#TP_QKV_ARR[@]} )); then
  CMD_COMMON+=(--tp-qkv "${TP_QKV_ARR[@]}")
fi
if (( ${#TP_FFN_ARR[@]} )); then
  CMD_COMMON+=(--tp-ffn "${TP_FFN_ARR[@]}")
fi
if (( ${#ALGOS_ARR[@]} )); then
  CMD_COMMON+=(--algo "${ALGOS_ARR[@]}")
fi
if (( ${#NPU_BACKENDS_ARR[@]} )); then
  CMD_COMMON+=(--npu-backend "${NPU_BACKENDS_ARR[@]}")
fi
if (( ${#WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS_ARR[@]} )); then
  CMD_COMMON+=(--weight-local-load-overlap-ratio "${WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS_ARR[@]}")
fi
if (( ${#FORMAT_OUTER_MAX_ITERS_ARR[@]} )); then
  CMD_COMMON+=(--format-outer-max-iters "${FORMAT_OUTER_MAX_ITERS_ARR[@]}")
fi
if (( ${#FORMAT_INNER_MAX_BLOCKS_ARR[@]} )); then
  CMD_COMMON+=(--format-inner-max-blocks "${FORMAT_INNER_MAX_BLOCKS_ARR[@]}")
fi
if (( ${#FORMAT_ND_MARGIN_INIT_ARR[@]} )); then
  CMD_COMMON+=(--format-nd-margin-init "${FORMAT_ND_MARGIN_INIT_ARR[@]}")
fi
if (( ${#FORMAT_ND_MARGIN_DECAY_ARR[@]} )); then
  CMD_COMMON+=(--format-nd-margin-decay "${FORMAT_ND_MARGIN_DECAY_ARR[@]}")
fi
if (( ${#FORMAT_ND_MARGIN_MIN_ARR[@]} )); then
  CMD_COMMON+=(--format-nd-margin-min "${FORMAT_ND_MARGIN_MIN_ARR[@]}")
fi
if (( ${#FORMAT_INNER_IMPROVE_EPS_ARR[@]} )); then
  CMD_COMMON+=(--format-inner-improve-eps "${FORMAT_INNER_IMPROVE_EPS_ARR[@]}")
fi
if (( ${#FORMAT_OUTER_STOP_EPS_ARR[@]} )); then
  CMD_COMMON+=(--format-outer-stop-eps "${FORMAT_OUTER_STOP_EPS_ARR[@]}")
fi
if (( ${#FORMAT_BLOCK_LAYER_SPAN_ARR[@]} )); then
  CMD_COMMON+=(--format-block-layer-span "${FORMAT_BLOCK_LAYER_SPAN_ARR[@]}")
fi
if (( ${#FORMAT_RELOAD_COUNT_MODE_ARR[@]} )); then
  CMD_COMMON+=(--format-reload-count-mode "${FORMAT_RELOAD_COUNT_MODE_ARR[@]}")
fi

merge_parallel_results() {
  local root_outdir="$1"
  local shard_root="$2"
  "${PYTHON_BIN}" - "${root_outdir}" "${shard_root}" "${OBJECTIVE}" "${COMBO_WORKERS}" "${PARALLEL_GROUP_KEYS}" <<'PY'
from pathlib import Path
import csv
import json
import math
import sys

root_outdir = Path(sys.argv[1]).resolve()
shard_root = Path(sys.argv[2]).resolve()
objective_name = sys.argv[3]
combo_workers = int(sys.argv[4])
parallel_group_keys = [x.strip() for x in str(sys.argv[5]).split(',') if x.strip()]

shard_dirs = sorted([p for p in shard_root.glob('worker_*') if p.is_dir()])
result_paths = [p / 'results.csv' for p in shard_dirs if (p / 'results.csv').exists()]
best_paths = [p / 'best_result.json' for p in shard_dirs if (p / 'best_result.json').exists()]
meta_paths = [p / 'dispatch_meta.json' for p in shard_dirs if (p / 'dispatch_meta.json').exists()]

rows = []
fieldnames = None
seen = set()
for rp in result_paths:
    with rp.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and fieldnames is None:
            fieldnames = list(reader.fieldnames)
        for row in reader:
            key = (
                str(row.get('config_sha256', '') or ''),
                str(row.get('repeat_idx', '') or ''),
                str(row.get('params_json', '') or ''),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row))

def _int_key(v: str) -> int:
    try:
        return int(str(v).strip())
    except Exception:
        return 10**18

rows.sort(
    key=lambda row: (
        _int_key(row.get('group_index', '')),
        str(row.get('model', '') or ''),
        _int_key(row.get('prefill_len', '')),
        _int_key(row.get('decode_len', '')),
        _int_key(row.get('batch', '')),
        _int_key(row.get('repeat_idx', '')),
        str(row.get('params_json', '') or ''),
    )
)

if fieldnames and rows:
    with (root_outdir / 'results.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

best_payload = None
best_objective = float('inf')
for bp in best_paths:
    try:
        data = json.loads(bp.read_text(encoding='utf-8'))
    except Exception:
        continue
    try:
        obj = float(data.get('objective'))
    except Exception:
        continue
    if not math.isfinite(obj):
        continue
    if best_payload is None or obj < best_objective:
        best_objective = obj
        best_payload = dict(data)
        best_payload['merged_from_best_json'] = str(bp)

if best_payload is not None:
    with (root_outdir / 'best_result.json').open('w', encoding='utf-8') as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

meta_payload = {
    'objective_name': objective_name,
    'combo_workers': combo_workers,
    'parallel_group_keys': parallel_group_keys,
    'shard_root': str(shard_root),
    'worker_dirs': [str(p) for p in shard_dirs],
    'result_files': [str(p) for p in result_paths],
    'best_files': [str(p) for p in best_paths],
    'dispatch_meta_files': [str(p) for p in meta_paths],
    'merged_row_count': len(rows),
    'has_best_result': best_payload is not None,
}
with (root_outdir / 'merged_summary.json').open('w', encoding='utf-8') as f:
    json.dump(meta_payload, f, ensure_ascii=False, indent=2)

print(f"[merge] shard_root={shard_root}")
print(f"[merge] merged_rows={len(rows)}")
print(f"[merge] results_csv={root_outdir / 'results.csv'}")
if best_payload is not None:
    print(f"[merge] best_result_json={root_outdir / 'best_result.json'} objective={best_objective:.6g}")
else:
    print("[merge] no valid shard best_result.json found")
PY
}

echo "[info] CONFIG=${CONFIG}"
echo "[info] SWEEP_SCRIPT=${SWEEP_SCRIPT}"
echo "[info] MAIN=${MAIN}"
echo "[info] OUTDIR=${OUTDIR}"
echo "[info] MODE=${MODE} TRIALS=${TRIALS} SEED=${SEED} OBJECTIVE=${OBJECTIVE} REPEAT=${REPEAT}"
echo "[info] MODELS=${MODELS_STR}"
echo "[info] PREFILLS=${PREFILLS_STR}"
echo "[info] DECODES=${DECODES_STR}"
echo "[info] BATCHES=${BATCHES_STR}"
echo "[info] HARDWARES=${HARDWARES_STR}"
echo "[info] TP_QKV=${TP_QKV_STR} TP_FFN=${TP_FFN_STR}"
echo "[info] WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS=${WEIGHT_LOCAL_LOAD_OVERLAP_RATIOS_STR}"
echo "[info] FORMAT_OUTER_MAX_ITERS=${FORMAT_OUTER_MAX_ITERS_STR}"
echo "[info] FORMAT_INNER_MAX_BLOCKS=${FORMAT_INNER_MAX_BLOCKS_STR}"
echo "[info] COMBO_WORKERS=${COMBO_WORKERS} PARALLEL_GROUP_KEYS=${PARALLEL_GROUP_KEYS} THREADS_PER_WORKER=${THREADS_PER_WORKER}"
echo "[info] common launch: ${CMD_COMMON[*]} $*"

if (( COMBO_WORKERS <= 1 )); then
  CMD=("${CMD_COMMON[@]}" --outdir "${OUTDIR}")
  echo "[info] single-worker launch: ${CMD[*]} $*"
  "${CMD[@]}" "$@"
  echo "Weight-suggest JSON-parameter sweep done."
  exit 0
fi

TOTAL_CPUS="${SLURM_CPUS_PER_TASK:-1}"
if (( THREADS_PER_WORKER <= 0 )); then
  THREADS_PER_WORKER=$(( TOTAL_CPUS / COMBO_WORKERS ))
  if (( THREADS_PER_WORKER < 1 )); then
    THREADS_PER_WORKER=1
  fi
fi

SHARD_ROOT="${OUTDIR}/shards/${COMBO_WORKERS}w"
mkdir -p "${SHARD_ROOT}"

declare -a PIDS=()
declare -a WORKER_LOGS=()
OVERALL_RC=0

echo "[info] launching ${COMBO_WORKERS} parallel workers under ${SHARD_ROOT}"
for (( wid=0; wid<COMBO_WORKERS; ++wid )); do
  worker_outdir="${SHARD_ROOT}/worker_${wid}"
  worker_log="${SHARD_ROOT}/worker_${wid}.launcher.log"
  mkdir -p "${worker_outdir}"
  WORKER_LOGS+=("${worker_log}")

  (
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
    export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
    export NUMEXPR_NUM_THREADS="${THREADS_PER_WORKER}"

    CMD=(
      "${CMD_COMMON[@]}"
      --outdir "${worker_outdir}"
      --parallel-group-keys "${PARALLEL_GROUP_KEYS}"
      --group-shard-index "${wid}"
      --group-shard-count "${COMBO_WORKERS}"
    )

    echo "[worker ${wid}] launch: ${CMD[*]} $*"
    echo "[worker ${wid}] threads_per_worker=${THREADS_PER_WORKER}"
    "${CMD[@]}" "$@"
  ) > "${worker_log}" 2>&1 &

  PIDS+=("$!")
done

for (( wid=0; wid<COMBO_WORKERS; ++wid )); do
  pid="${PIDS[wid]}"
  if ! wait "${pid}"; then
    echo "[err] worker ${wid} failed. see ${WORKER_LOGS[wid]}" >&2
    OVERALL_RC=1
  else
    echo "[ok] worker ${wid} finished. log=${WORKER_LOGS[wid]}"
  fi
done

if [[ "${MERGE_RESULTS}" == "1" ]]; then
  merge_parallel_results "${OUTDIR}" "${SHARD_ROOT}"
fi

if (( OVERALL_RC != 0 )); then
  echo "[err] one or more workers failed" >&2
  exit "${OVERALL_RC}"
fi

echo "Weight-suggest JSON-parameter sweep done."
