#!/usr/bin/env bash

: <<'USAGE'
Submit verify pipeline: export -> run-gpu + run-pim -> merge

You can optionally override log locations:
GPU_LOG_DIR=/path/logs_gpu \
PIM_LOG_DIR=/path/logs_pim \
MERGE_LOG_DIR=/path/logs_merge \

export PYTHONPATH="$PWD:$PWD/algorithm:$PWD/algorithms:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/lustre/home/2501111916/workspace/XPUPIM_0226_gpupim_parameter/TriForm/submodules/CENT/aim_simulator${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  ./verify/run_verify.sh \
    ./verify/jobs_sweep.tsv \
    ./verify/run_gpu_wm2_param.slurm \
    ./verify/run_pim_param.slurm \
    ./verify/run_merge_param.slurm

USAGE

set -euo pipefail

# Resolve repo root (run_verify.sh lives in <repo>/verify/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# export -> run_gpu and run_pim -> merge
JOB_LIST=${1:-jobs.tsv}
GPU_SLURM=${2:-"$SCRIPT_DIR/run_gpu_wm2_param.slurm"}
PIM_SLURM=${3:-"$SCRIPT_DIR/run_pim_param.slurm"}
MERGE_SLURM=${4:-"$SCRIPT_DIR/run_merge_param.slurm"}
NPU_SLURM=${NPU_SLURM:-"$SCRIPT_DIR/run_npu_lut_param.slurm"}
COMPUTE_BACKEND=${COMPUTE_BACKEND:-auto}

# -------- user-tunable defaults --------
PY_SCRIPT=${PY_SCRIPT:-"$REPO_ROOT/verify/schedule_deploy_verify.py"}
OUT_ROOT=${OUT_ROOT:-"$REPO_ROOT/verify/out"}
SEGMENT_SCOPE=${SEGMENT_SCOPE:-layer}
SHARD_POLICY=${SHARD_POLICY:-fine}  # fine | coarse_majority

LOG_ROOT=${LOG_ROOT:-""}
GPU_LOG_DIR=${GPU_LOG_DIR:-""}
NPU_LOG_DIR=${NPU_LOG_DIR:-""}
PIM_LOG_DIR=${PIM_LOG_DIR:-""}
MERGE_LOG_DIR=${MERGE_LOG_DIR:-""}
JOB_CHDIR=${JOB_CHDIR:-""}

# model shape json for `export`
DEFAULT_CFG=${DEFAULT_CFG:-""}

# Fallback defaults when neither TSV nor filename inference provides them
DEFAULT_PREFILL_LEN=${DEFAULT_PREFILL_LEN:-4096}
DEFAULT_DECODE_STRIDE=${DEFAULT_DECODE_STRIDE:-128}

# Multi-value expansion (used only when TSV column is empty)
PREFILL_LENS=${PREFILL_LENS:-"4096"}
DECODE_STRIDES=${DECODE_STRIDES:-"128"}      
NON_OVERLAP=${NON_OVERLAP:-0.2}      

KV_LOAD_BW_GBS=${KV_LOAD_BW_GBS:-"16384"}
KV_DTYPE_BYTES=${KV_DTYPE_BYTES:-2}         # fp16/bf16=2, int8=1
KV_LOAD_OVERHEAD_US=${KV_LOAD_OVERHEAD_US:-0}
N_KV_HEADS=${N_KV_HEADS:-""}                # GQA/MQA 用；空表示用 n_heads
BATCH=${BATCH:-1}
BATCHES_USER_SET=${BATCHES+x}
BATCHES=${BATCHES:-"$BATCH"}

# run-gpu benchmark args
WARMUP=${WARMUP:-3}
ITERS=${ITERS:-10}
DEVICE=${DEVICE:-cuda}
GPU_DTYPE=${GPU_DTYPE:-bf16}

MMAD_LUT=${MMAD_LUT:-""}
SOFTMAX_LUT=${SOFTMAX_LUT:-""}
GELU_LUT=${GELU_LUT:-""}
NORM_LUT=${NORM_LUT:-""}
NPU_DTYPE=${NPU_DTYPE:-""}              # fp16|bf16|fp32|int8 (optional, for mem bound)
NPU_MEM_BW_GBS=${NPU_MEM_BW_GBS:-"819.2"}   # must be >0 when backend=npu (Activation RW is mandatory)
NPU_OP_OVERHEAD_US=${NPU_OP_OVERHEAD_US:-"0"}
NPU_DEBUG=${NPU_DEBUG:-0}

# run-pim args (global)
CENT_SIM_ROOT=${CENT_SIM_ROOT:-""}
PIM_RAMULATOR_CONFIG=${PIM_RAMULATOR_CONFIG:-"$REPO_ROOT/algorithms/aim_simulator/example.yaml"}
PIM_HW_JSON=${PIM_HW_JSON:-"$REPO_ROOT/algorithms/aim_simulator/PIM_AiM.json"}
PIM_RAMULATOR_BIN=${PIM_RAMULATOR_BIN:-"$REPO_ROOT/algorithms/ramulator2"}
PIM_NUM_DEVICES=${PIM_NUM_DEVICES:-4}
# run-pim debug (optional)
PIM_DEBUG=${PIM_DEBUG:-1}       # 1 to enable verbose per-op debug logging
PIM_NO_CACHE=${PIM_NO_CACHE:-1} # 1 to disable CostModel PIM cache (forces ramulator per-op)

# merge args
COMM_MODEL=${COMM_MODEL:-schedule}
PCIE_LANES=${PCIE_LANES:-16}
ALLOW_MISSING=${ALLOW_MISSING:-1}
WRITE_STEPS_CSV=${WRITE_STEPS_CSV:-0}  # 1 to also write per-step csv

MERGE_DEBUG=${MERGE_DEBUG:-1}

COLLECT_MERGE_CSV=${COLLECT_MERGE_CSV:-1}
COLLECT_SKIP_MISSING=${COLLECT_SKIP_MISSING:-1}

# Optional throttle: set MAX_IN_FLIGHT>0 to limit how many jobs you submit at once.
MAX_IN_FLIGHT=${MAX_IN_FLIGHT:-0}

# Optional extra sbatch arguments
GPU_SBATCH_ARGS=${GPU_SBATCH_ARGS:-""}
NPU_SBATCH_ARGS=${NPU_SBATCH_ARGS:-""}
PIM_SBATCH_ARGS=${PIM_SBATCH_ARGS:-""}
MERGE_SBATCH_ARGS=${MERGE_SBATCH_ARGS:-""}

abspath() {
  # usage: abspath <path>
  python - <<'PY' "$1"
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

abspath_under() {
  # usage: abspath_under <path> <base>
  python - <<'PY' "$1" "$2"
import os, sys
p = os.path.expanduser(sys.argv[1])
base = os.path.expanduser(sys.argv[2])
if not os.path.isabs(p):
    p = os.path.join(base, p)
print(os.path.abspath(p))
PY
}

split_list() {
  # split comma/space separated list -> lines
  # usage: split_list "128,1024 8192"  => prints each token on new line
  local s="$1"
  s=${s//,/ }
  for x in $s; do
    if [[ -n "$x" ]]; then
      echo "$x"
    fi
  done
}

infer_prefill_and_stride() {
  # prints: "<prefill_len> <decode_stride>" or "" if no inference
  local fname="$1"
  local pre=""
  local dec=""

  # pattern A: prefill-8192xdecode_128
  if [[ "$fname" =~ prefill-([0-9]+)xdecode_([0-9]+) ]]; then
    pre="${BASH_REMATCH[1]}"
    dec="${BASH_REMATCH[2]}"
  # pattern B: _8192x128_ops_trace
  elif [[ "$fname" =~ ([0-9]+)x([0-9]+)_ops_trace ]]; then
    pre="${BASH_REMATCH[1]}"
    dec="${BASH_REMATCH[2]}"
  # pattern C: _8192x128
  elif [[ "$fname" =~ ([0-9]+)x([0-9]+) ]]; then
    pre="${BASH_REMATCH[1]}"
    dec="${BASH_REMATCH[2]}"
  fi

  if [[ -z "$pre" || -z "$dec" ]]; then
    echo ""
  else
    echo "$pre $dec"
  fi
}

infer_batch_from_path() {
  local p="$1"
  local b=""
  if [[ "$p" =~ (^|/|_)(b|bs|batch)([0-9]+)($|/|_) ]]; then
    b="${BASH_REMATCH[3]}"
  fi
  echo "$b"
}

infer_compute_backend() {
  # usage: infer_compute_backend <schedule_csv>
  # returns: gpu | npu | mixed | unknown
  python - <<'PY' "$1"
import sys, csv
p = sys.argv[1]
has_gpu = False
has_npu = False

try:
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames or []
        if "device" not in fns:
            print("unknown")
            raise SystemExit(0)

        # Scan a limited number of rows for speed
        for i, row in enumerate(reader):
            if i >= 2000:
                break
            dev = str(row.get("device", "")).lower()
            if "gpu" in dev:
                has_gpu = True
            if "npu" in dev:
                has_npu = True
            if has_gpu and has_npu:
                break
except SystemExit:
    raise
except Exception:
    # Fallback: keep unknown
    pass

if has_gpu and has_npu:
    print("mixed")
elif has_gpu:
    print("gpu")
elif has_npu:
    print("npu")
else:
    print("unknown")
PY
}

maybe_throttle() {
  if [[ "$MAX_IN_FLIGHT" -le 0 ]]; then
    return 0
  fi
  while true; do
    local n
    n=$(squeue -u "$USER" -h | wc -l | tr -d ' ')
    if [[ "$n" -lt "$MAX_IN_FLIGHT" ]]; then
      break
    fi
    echo "[submit] throttling: currently $n jobs in queue (MAX_IN_FLIGHT=$MAX_IN_FLIGHT)" >&2
    sleep 5
  done
}

# ----------------------
# Basic validation
# ----------------------
if [[ ! -f "$JOB_LIST" ]]; then
  echo "[submit] ERROR: job list not found: $JOB_LIST" >&2
  exit 1
fi

for f in "$GPU_SLURM" "$PIM_SLURM" "$MERGE_SLURM"; do
  if [[ ! -f "$f" ]]; then
    echo "[submit] ERROR: slurm script not found: $f" >&2
    exit 1
  fi
done

# Catch the most common mistake: pass run_pim_param.slurm as MERGE_SLURM
if [[ "$(abspath "$MERGE_SLURM")" == "$(abspath "$PIM_SLURM")" ]]; then
  echo "[submit] ERROR: MERGE_SLURM == PIM_SLURM ($MERGE_SLURM)." >&2
  echo "  Did you accidentally pass run_pim_param.slurm as the 4th argument?" >&2
  echo "  Expected: run_merge_param.slurm" >&2
  exit 1
fi
if grep -qE "\brun-pim\b" "$MERGE_SLURM"; then
  echo "[submit] ERROR: MERGE_SLURM seems to run the 'run-pim' subcommand: $MERGE_SLURM" >&2
  echo "  Please pass run_merge_param.slurm as the 4th argument." >&2
  exit 1
fi

# ----------------------
# Resolve tool/config paths
# ----------------------
PY_ABS=$(abspath_under "$PY_SCRIPT" "$REPO_ROOT")
GPU_SLURM_ABS=$(abspath "$GPU_SLURM")
PIM_SLURM_ABS=$(abspath "$PIM_SLURM")
MERGE_SLURM_ABS=$(abspath "$MERGE_SLURM")
NPU_SLURM_ABS=""
if [[ -n "${NPU_SLURM:-}" && -f "$NPU_SLURM" ]]; then
  NPU_SLURM_ABS=$(abspath "$NPU_SLURM")
fi

# Resolve PIM config paths relative to repo root if they are relative
PIM_RAMULATOR_CONFIG_ABS=$(abspath_under "$PIM_RAMULATOR_CONFIG" "$REPO_ROOT")
PIM_HW_JSON_ABS=$(abspath_under "$PIM_HW_JSON" "$REPO_ROOT")

# Fail early with a clear message (instead of letting a later job crash)
if [[ ! -f "$PIM_RAMULATOR_CONFIG_ABS" ]]; then
  echo "[submit] ERROR: Ramulator config not found: $PIM_RAMULATOR_CONFIG_ABS" >&2
  echo "  Set PIM_RAMULATOR_CONFIG to an existing example.yaml (absolute path recommended)." >&2
  exit 1
fi
if [[ ! -f "$PIM_HW_JSON_ABS" ]]; then
  echo "[submit] ERROR: PIM HW json not found: $PIM_HW_JSON_ABS" >&2
  echo "  Set PIM_HW_JSON to an existing PIM_AiM.json (absolute path recommended)." >&2
  exit 1
fi

PIM_RAMULATOR_BIN_ABS="$PIM_RAMULATOR_BIN"
if [[ -n "$PIM_RAMULATOR_BIN" ]]; then
  if [[ "$PIM_RAMULATOR_BIN" == */* || "$PIM_RAMULATOR_BIN" == .* || "$PIM_RAMULATOR_BIN" == ~* ]]; then
    PIM_RAMULATOR_BIN_ABS=$(abspath_under "$PIM_RAMULATOR_BIN" "$REPO_ROOT")
  elif [[ -e "$PIM_RAMULATOR_BIN" ]]; then
    # if user gave a relative executable that exists in CWD
    PIM_RAMULATOR_BIN_ABS=$(abspath "$PIM_RAMULATOR_BIN")
  fi
fi

CENT_SIM_ROOT_ABS="$CENT_SIM_ROOT"
if [[ -n "$CENT_SIM_ROOT" ]]; then
  if [[ "$CENT_SIM_ROOT" == */* || "$CENT_SIM_ROOT" == .* || "$CENT_SIM_ROOT" == ~* ]]; then
    CENT_SIM_ROOT_ABS=$(abspath_under "$CENT_SIM_ROOT" "$REPO_ROOT")
  elif [[ -e "$CENT_SIM_ROOT" ]]; then
    CENT_SIM_ROOT_ABS=$(abspath "$CENT_SIM_ROOT")
  fi
fi

echo "[submit] REPO_ROOT=$REPO_ROOT" >&2
echo "[submit] PY_SCRIPT=$PY_ABS" >&2
echo "[submit] GPU_SLURM=$GPU_SLURM_ABS" >&2
echo "[submit] NPU_SLURM=${NPU_SLURM_ABS:-<none>}" >&2
echo "[submit] PIM_SLURM=$PIM_SLURM_ABS" >&2
echo "[submit] MERGE_SLURM=$MERGE_SLURM_ABS" >&2
echo "[submit] OUT_ROOT=$(abspath_under "$OUT_ROOT" "$REPO_ROOT")" >&2

echo "[submit] PIM_RAMULATOR_CONFIG=$PIM_RAMULATOR_CONFIG_ABS" >&2
echo "[submit] PIM_HW_JSON=$PIM_HW_JSON_ABS" >&2
echo "[submit] PIM_RAMULATOR_BIN=$PIM_RAMULATOR_BIN_ABS" >&2

echo "[submit] reading job list: $JOB_LIST" >&2

# TSV parsing (tab-separated).
while IFS=$'\t,' read -r schedule_csv comms_csv prefix out_dir prefill_len decode_stride cfg batch || [[ -n "${schedule_csv:-}" ]]; do
  # skip empty / comment lines
  if [[ -z "${schedule_csv:-}" ]]; then
    continue
  fi
  if [[ "${schedule_csv:0:1}" == "#" ]]; then
    continue
  fi

  schedule_csv=${schedule_csv//$'\r'/}
  comms_csv=${comms_csv//$'\r'/}
  prefix=${prefix//$'\r'/}
  out_dir=${out_dir//$'\r'/}
  prefill_len=${prefill_len//$'\r'/}
  decode_stride=${decode_stride//$'\r'/}
  cfg=${cfg//$'\r'/}
  batch=${batch//$'\r'/}

  schedule_csv=$(abspath "$schedule_csv")

  # comms is optional
  if [[ -n "${comms_csv:-}" && "${comms_csv}" != "-" ]]; then
    comms_csv=$(abspath "$comms_csv")
  else
    comms_csv=""
  fi

  base_name=$(basename "$schedule_csv")

  if [[ -z "${prefix:-}" ]]; then
    prefix="${base_name%.csv}"
    prefix="${prefix%_ops_trace}"
  fi

  if [[ -z "${out_dir:-}" ]]; then
    out_dir="$OUT_ROOT/$prefix"
  fi
  out_dir=$(abspath_under "$out_dir" "$REPO_ROOT")
  mkdir -p "$out_dir"

  if [[ -n "${LOG_ROOT:-}" ]]; then
    base_log_root=$(abspath_under "$LOG_ROOT" "$REPO_ROOT")
  else
    base_log_root="$out_dir/logs"
  fi

  gpu_log_dir="${GPU_LOG_DIR:-}"
  npu_log_dir="${NPU_LOG_DIR:-}"
  pim_log_dir="${PIM_LOG_DIR:-}"
  merge_log_dir="${MERGE_LOG_DIR:-}"

  if [[ -z "$gpu_log_dir" ]]; then gpu_log_dir="$base_log_root/gpu"; fi
  if [[ -z "$npu_log_dir" ]]; then npu_log_dir="$base_log_root/npu"; fi
  if [[ -z "$pim_log_dir" ]]; then pim_log_dir="$base_log_root/pim"; fi
  if [[ -z "$merge_log_dir" ]]; then merge_log_dir="$base_log_root/merge"; fi

  gpu_log_dir=$(abspath_under "$gpu_log_dir" "$REPO_ROOT")
  npu_log_dir=$(abspath_under "$npu_log_dir" "$REPO_ROOT")
  pim_log_dir=$(abspath_under "$pim_log_dir" "$REPO_ROOT")
  merge_log_dir=$(abspath_under "$merge_log_dir" "$REPO_ROOT")
  mkdir -p "$gpu_log_dir" "$npu_log_dir" "$pim_log_dir" "$merge_log_dir"

  if [[ -n "${JOB_CHDIR:-}" ]]; then
    job_chdir=$(abspath_under "$JOB_CHDIR" "$REPO_ROOT")
  else
    job_chdir="$out_dir"
  fi
  mkdir -p "$job_chdir"

  # model cfg json (required for export unless schedule_deploy_verify.py falls back to defaults)
  if [[ -n "${cfg:-}" && "${cfg}" != "-" ]]; then
    cfg=$(abspath_under "$cfg" "$REPO_ROOT")
  elif [[ -n "$DEFAULT_CFG" ]]; then
    cfg=$(abspath_under "$DEFAULT_CFG" "$REPO_ROOT")
  else
    echo "[submit] ERROR: missing cfg column in TSV and DEFAULT_CFG is empty.\n  Please add a 'cfg' column (e.g. ./configs/llama_7b_shape.json) or export DEFAULT_CFG." >&2
    exit 1
  fi

  if [[ ! -f "$cfg" ]]; then
    echo "[submit] ERROR: model cfg json not found: $cfg" >&2
    exit 1
  fi

  # determine candidate prefill/decode lists
  pre_list=()
  dec_list=()
  batch_list=()

  # 1) If TSV provides explicit values, take them.
  if [[ -n "${prefill_len:-}" ]]; then
    pre_list=("$prefill_len")
  fi
  if [[ -n "${decode_stride:-}" ]]; then
    dec_list=("$decode_stride")
  fi
  if [[ -n "${batch:-}" ]]; then
    batch_list=("$batch")
  fi

  # 2) If TSV did not provide, and the user explicitly sets multi-value lists, expand them.
  #    This takes precedence over filename inference.
  if [[ ${#pre_list[@]} -eq 0 && -n "$PREFILL_LENS" ]]; then
    while IFS= read -r x; do pre_list+=("$x"); done < <(split_list "$PREFILL_LENS")
  fi
  if [[ ${#dec_list[@]} -eq 0 && -n "$DECODE_STRIDES" ]]; then
    while IFS= read -r x; do dec_list+=("$x"); done < <(split_list "$DECODE_STRIDES")
  fi
  if [[ ${#batch_list[@]} -eq 0 && -n "${BATCHES_USER_SET:-}" && -n "$BATCHES" ]]; then
    while IFS= read -r x; do batch_list+=("$x"); done < <(split_list "$BATCHES")
  fi

  # 3) If still missing, try infer from filename (prefill-XXXXxdecode_YYY / 8192x128 etc.)
  if [[ ${#pre_list[@]} -eq 0 || ${#dec_list[@]} -eq 0 ]]; then
    inf=$(infer_prefill_and_stride "$base_name")
    if [[ -n "$inf" ]]; then
      read -r inf_pre inf_dec <<< "$inf"
      if [[ ${#pre_list[@]} -eq 0 ]]; then pre_list=("$inf_pre"); fi
      if [[ ${#dec_list[@]} -eq 0 ]]; then dec_list=("$inf_dec"); fi
    fi
  fi

  # 4) Final fallback defaults
  if [[ ${#pre_list[@]} -eq 0 ]]; then
    pre_list=("$DEFAULT_PREFILL_LEN")
  fi
  if [[ ${#dec_list[@]} -eq 0 ]]; then
    dec_list=("$DEFAULT_DECODE_STRIDE")
  fi
  if [[ ${#batch_list[@]} -eq 0 ]]; then
    batch_list=("$BATCH")
  fi

  # We add suffixes only when at least one dimension expands.
  multi_suffix=0
  if [[ ${#pre_list[@]} -gt 1 || ${#dec_list[@]} -gt 1 || ${#batch_list[@]} -gt 1 ]]; then
    multi_suffix=1
  fi

  echo -e "\n[case] base_prefix=$prefix" >&2
  echo "  schedule=$schedule_csv" >&2
  echo "  comms=${comms_csv:-<none>}" >&2
  echo "  out_dir=$out_dir" >&2
  echo "  cfg=$cfg" >&2
  echo "  prefill_list=(${pre_list[*]}) decode_stride_list=(${dec_list[*]}) batch_list=(${batch_list[*]})" >&2
  merge_list_tsv="$out_dir/${prefix}.merge_list.tsv"
  echo -e "prefix_combo\tprefill_len\tdecode_stride\tbatch\tschedule_csv\tmerge_csv\tdebug_txt" > "$merge_list_tsv"
  case_merge_jids=()

  for pre in "${pre_list[@]}"; do
    for dec in "${dec_list[@]}"; do
      for b in "${batch_list[@]}"; do

      if [[ "$multi_suffix" -eq 1 ]]; then
        suffix_parts=()
        if [[ ${#pre_list[@]} -gt 1 ]]; then suffix_parts+=("p${pre}"); fi
        if [[ ${#dec_list[@]} -gt 1 ]]; then suffix_parts+=("d${dec}"); fi
        if [[ ${#batch_list[@]} -gt 1 ]]; then suffix_parts+=("b${b}"); fi
        if [[ ${#suffix_parts[@]} -gt 0 ]]; then
          suffix_joined=$(IFS=_; echo "${suffix_parts[*]}")
          prefix_combo="${prefix}_${suffix_joined}"
        else
          prefix_combo="$prefix"
        fi
      else
        prefix_combo="$prefix"
      fi

      echo "\n  [combo] prefix=$prefix_combo prefill_len=$pre decode_stride=$dec batch=$b" >&2

      # 1) export tasks
      export_cmd=(python "$PY_ABS" export \
        --schedule "$schedule_csv" \
        --out-dir "$out_dir" \
        --prefix "$prefix_combo" \
        --segment-scope "$SEGMENT_SCOPE" \
        --prefill-len "$pre" \
        --decode-stride "$dec" \
        --cfg "$cfg" \
        --shard-policy "$SHARD_POLICY")

      if [[ -n "$comms_csv" ]]; then
        export_cmd+=(--comms "$comms_csv")
      fi
      if [[ -n "${KV_LOAD_BW_GBS}" ]]; then
        export_cmd+=(--kv-load-bw-gbs "$KV_LOAD_BW_GBS")
      fi
      export_cmd+=(--kv-dtype-bytes "$KV_DTYPE_BYTES")
      export_cmd+=(--kv-load-overhead-us "$KV_LOAD_OVERHEAD_US")
      if [[ -n "${N_KV_HEADS}" ]]; then
        export_cmd+=(--n-kv-heads "$N_KV_HEADS")
      fi
      export_cmd+=(--batch "$b")

      echo "  [combo] export: ${export_cmd[*]}" >&2
      "${export_cmd[@]}"

      gpu_tasks="$out_dir/$prefix_combo.gpu_tasks.json"
      pim_tasks="$out_dir/$prefix_combo.pim_tasks.json"

      gpu_res="$out_dir/$prefix_combo.gpu_results.json"
      npu_res="$out_dir/$prefix_combo.npu_results.json"
      pim_res="$out_dir/$prefix_combo.pim_results.json"

      merge_csv="$out_dir/$prefix_combo.merge.csv"
      steps_csv="$out_dir/$prefix_combo.merge_steps.csv"

      debug_txt="${merge_csv%.csv}.debug.txt"

      gpu_ok="$out_dir/$prefix_combo.gpu.ok"
      npu_ok="$out_dir/$prefix_combo.npu.ok"
      pim_ok="$out_dir/$prefix_combo.pim.ok"

      gpu_debug_txt="$out_dir/$prefix_combo.gpu_debug.log"
      npu_debug_txt="$out_dir/$prefix_combo.npu_debug.log"
      pim_debug_txt="$out_dir/$prefix_combo.pim_debug.log"
      
      rm -f "$gpu_ok" "$npu_ok" "$pim_ok" "$gpu_res" "$npu_res" "$pim_res" "$merge_csv" "$debug_txt" "$steps_csv" "$gpu_debug_txt" "$npu_debug_txt" "$pim_debug_txt"

      if [[ ! -f "$gpu_tasks" ]]; then
        echo "  [combo] ERROR: missing $gpu_tasks" >&2
        exit 2
      fi
      if [[ ! -f "$pim_tasks" ]]; then
        echo "  [combo] ERROR: missing $pim_tasks" >&2
        exit 2
      fi

      maybe_throttle

      # 2) submit compute stage (GPU benchmark on HPC OR NPU LUT lookup on CPU)
      backend="${COMPUTE_BACKEND,,}"
      if [[ -z "$backend" || "$backend" == "auto" ]]; then
        backend="$(infer_compute_backend "$schedule_csv")"
        if [[ "$backend" == "unknown" ]]; then
          backend="gpu"
        fi
      fi

      if [[ "$backend" == "mixed" ]]; then
        echo "  [combo] ERROR: schedule contains both 'gpu' and 'npu' in device names; please set COMPUTE_BACKEND=gpu or COMPUTE_BACKEND=npu explicitly." >&2
        exit 11
      fi

      compute_res="$gpu_res"
      compute_ok="$gpu_ok"
      compute_log_dir="$gpu_log_dir"
      compute_debug_txt="$gpu_debug_txt"
      compute_job="gpu_${prefix_combo}"
      compute_sbatch_args="$GPU_SBATCH_ARGS"
      compute_slurm="$GPU_SLURM_ABS"

      if [[ "$backend" == "npu" ]]; then
        # Activation RW is mandatory for the Ascend LUT backend.
        if [[ -z "${NPU_MEM_BW_GBS:-}" ]]; then
          echo "  [combo] ERROR: backend=npu requires NPU_MEM_BW_GBS to be set and >0 (Activation RW is mandatory)." >&2
          exit 12
        fi
        if ! awk -v v="$NPU_MEM_BW_GBS" 'BEGIN{exit (v>0)?0:1}'; then
          echo "  [combo] ERROR: backend=npu requires NPU_MEM_BW_GBS >0 (got '$NPU_MEM_BW_GBS')." >&2
          exit 12
        fi
        if [[ -z "${NPU_SLURM_ABS:-}" ]]; then
          echo "  [combo] ERROR: backend=npu but NPU_SLURM_ABS is empty. Set NPU_SLURM or place run_npu_lut_param.slurm under $SCRIPT_DIR." >&2
          exit 12
        fi
        compute_res="$npu_res"
        compute_ok="$npu_ok"
        compute_log_dir="$npu_log_dir"
        compute_debug_txt="$npu_debug_txt"
        compute_job="npu_${prefix_combo}"
        compute_sbatch_args="$NPU_SBATCH_ARGS"
        compute_slurm="$NPU_SLURM_ABS"
      fi

      if [[ "$backend" == "gpu" ]]; then
        jid_compute=$(sbatch --parsable $compute_sbatch_args           -J "$compute_job"           --chdir="$job_chdir"           -o "$compute_log_dir/%x.%j.out" -e "$compute_log_dir/%x.%j.err"           --export=ALL,LOG_DIR="$compute_log_dir",OUT_DIR="$out_dir",PREFIX="$prefix_combo",PY_SCRIPT="$PY_ABS",TASKS_JSON="$gpu_tasks",OUT_JSON="$compute_res",WARMUP="$WARMUP",ITERS="$ITERS",DEVICE="$DEVICE",GPU_DTYPE="$GPU_DTYPE",BATCH="$b",OK_FILE="$compute_ok",GPU_DEBUG_TXT="$compute_debug_txt"          "$compute_slurm")
        echo "  [combo] submitted run-gpu: jobid=$jid_compute" >&2
      else
        # NPU (LUT) stage: CPU job that calls `python <PY_SCRIPT> run-npu`
        jid_compute=$(sbatch --parsable $compute_sbatch_args           -J "$compute_job"           --chdir="$job_chdir"           -o "$compute_log_dir/%x.%j.out" -e "$compute_log_dir/%x.%j.err"           --export=ALL,LOG_DIR="$compute_log_dir",OUT_DIR="$out_dir",PREFIX="$prefix_combo",PY_SCRIPT="$PY_ABS",TASKS_JSON="$gpu_tasks",OUT_JSON="$compute_res",BATCH="$b",OK_FILE="$compute_ok",MMAD_LUT="$MMAD_LUT",SOFTMAX_LUT="$SOFTMAX_LUT",GELU_LUT="$GELU_LUT",NORM_LUT="$NORM_LUT",NPU_DTYPE="$NPU_DTYPE",NPU_MEM_BW_GBS="$NPU_MEM_BW_GBS",NPU_OP_OVERHEAD_US="$NPU_OP_OVERHEAD_US",NPU_DEBUG="$NPU_DEBUG",NPU_DEBUG_TXT="$compute_debug_txt"           "$compute_slurm")
        echo "  [combo] submitted run-npu: jobid=$jid_compute" >&2
      fi

    # 3) submit run-pim (CPU job)
      jid_pim=$(sbatch --parsable $PIM_SBATCH_ARGS \
        -J "pim_${prefix_combo}" \
        --chdir="$job_chdir" \
        -o "$pim_log_dir/%x.%j.out" -e "$pim_log_dir/%x.%j.err" \
        --export=ALL,LOG_DIR="$pim_log_dir",OUT_DIR="$out_dir",PREFIX="$prefix_combo",PY_SCRIPT="$PY_ABS",TASKS_JSON="$pim_tasks",OUT_JSON="$pim_res",CENT_SIM_ROOT="$CENT_SIM_ROOT_ABS",PIM_RAMULATOR_CONFIG="$PIM_RAMULATOR_CONFIG_ABS",PIM_HW_JSON="$PIM_HW_JSON_ABS",PIM_RAMULATOR_BIN="$PIM_RAMULATOR_BIN_ABS",PIM_NUM_DEVICES="$PIM_NUM_DEVICES",BATCH="$b",OK_FILE="$pim_ok",PIM_DEBUG="$PIM_DEBUG",PIM_DEBUG_TXT="$pim_debug_txt",PIM_NO_CACHE="$PIM_NO_CACHE" \
        "$PIM_SLURM_ABS")

      echo "  [combo] submitted run-pim: jobid=$jid_pim" >&2

      # 4) submit merge job (always runs after both finish; exits error if any failed)
      dep="afterany:${jid_compute}:${jid_pim}"

      if [[ "$WRITE_STEPS_CSV" == "1" || "$WRITE_STEPS_CSV" == "true" ]]; then
        out_steps="$steps_csv"
      else
        out_steps=""
      fi

      jid_merge=$(sbatch --parsable $MERGE_SBATCH_ARGS \
        -J "merge_${prefix_combo}" \
        --dependency="$dep" \
        --chdir="$job_chdir" \
        -o "$merge_log_dir/%x.%j.out" -e "$merge_log_dir/%x.%j.err" \
        --export=ALL,LOG_DIR="$merge_log_dir",GPU_LOG_DIR="$compute_log_dir",PIM_LOG_DIR="$pim_log_dir",OUT_DIR="$out_dir",PREFIX="$prefix_combo",PY_SCRIPT="$PY_ABS",SCHEDULE_CSV="$schedule_csv",GPU_RESULTS_JSON="$compute_res",PIM_RESULTS_JSON="$pim_res",GPU_OK_FILE="$compute_ok",PIM_OK_FILE="$pim_ok",COMM_MODEL="$COMM_MODEL",PCIE_LANES="$PCIE_LANES",COMMS_CSV="$comms_csv",NON_OVERLAP="$NON_OVERLAP",DECODE_STRIDE="$dec",BATCH="$b",OUT_CSV="$merge_csv",OUT_STEPS_CSV="$out_steps",ALLOW_MISSING="$ALLOW_MISSING",DEBUG_MERGE="$MERGE_DEBUG",DEBUG_TXT="$debug_txt" \
        "$MERGE_SLURM_ABS")

      echo "  [combo] submitted merge: jobid=$jid_merge  (will write: $merge_csv)" >&2

      # record ids
      echo -e "${prefix_combo}\t${jid_compute}\t${jid_pim}\t${jid_merge}\t${merge_csv}" >> "$out_dir/jobids.tsv"

      # record for final collection
      echo -e "${prefix_combo}\t${pre}\t${dec}\t${b}\t${schedule_csv}\t${merge_csv}\t${debug_txt}" >> "$merge_list_tsv"
      case_merge_jids+=("$jid_merge")

      done
    done
  done

  # 5) optional collect: concatenate multiple merge.csv into one merged csv
  if [[ "${COLLECT_MERGE_CSV}" == "1" || "${COLLECT_MERGE_CSV}" == "true" ]]; then
    if [[ ${#case_merge_jids[@]} -gt 0 ]]; then
      dep_collect="afterany:$(IFS=:; echo "${case_merge_jids[*]}")"
      merged_all_csv="$out_dir/${prefix}.merge_all.csv"
      jid_collect=$(sbatch --parsable $MERGE_SBATCH_ARGS \
        -J "collect_${prefix}" \
        --dependency="$dep_collect" \
        --chdir="$job_chdir" \
        -o "$merge_log_dir/%x.%j.out" -e "$merge_log_dir/%x.%j.err" \
        --export=ALL,LOG_DIR="$merge_log_dir",PY_SCRIPT="$PY_ABS",COLLECT_MERGES=1,MERGE_LIST_FILE="$merge_list_tsv",COLLECT_OUT_CSV="$merged_all_csv",COLLECT_SKIP_MISSING="$COLLECT_SKIP_MISSING" \
        "$MERGE_SLURM_ABS")

      echo "  [case] submitted collect-merge: jobid=$jid_collect  (will write: $merged_all_csv)" >&2
      echo -e "@collect_${prefix}\t-\t-\t${jid_collect}\t${merged_all_csv}" >> "$out_dir/jobids.tsv"
    fi
  fi

done < "$JOB_LIST"
echo -e "\n[submit] Done. Job IDs recorded in <out_dir>/jobids.tsv for each case." >&2
echo "[submit] Use: squeue -u $USER" >&2