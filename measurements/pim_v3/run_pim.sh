#!/usr/bin/env bash
set -euo pipefail

# --------------- defaults ---------------
TRACE_DIR="traces"
RESULTS_CSV="ram_results_ext.csv"
MODEL_JSON="model.json"

DIM=256
N_HEADS=8
SEQLENS="64,128,256,512,1024,2048"

# Ramulator 相关
EXTRA_ARGS=""                           # 例如 "--aim"
METRIC_REGEX='Total cycles:\s*([0-9]+)' # 若你的 fork 文案不同，可改
CMD_TEMPLATE='{bin} --config {config} --trace {trace} {extra}'  # 需要时可改

# 必填
PIM_CONFIG=""
RAMULATOR_BIN=""
RAM_CONFIG=""

# --------------- helpers ---------------
usage() {
  cat <<EOF
Usage: $0 -c <pim_config.json> -b <ramulator_bin> -r <ramulator_config.yaml> [options]

Required:
  -c  PIM config JSON 路径（供 gentrace.py 使用）
  -b  Ramulator 可执行文件路径
  -r  Ramulator 配置文件路径（例如 pim-config.yaml）

Options:
  -o  traces 输出目录（默认: ${TRACE_DIR}）
  -f  结果 CSV 文件名（默认: ${RESULTS_CSV}）
  -m  拟合模型输出 JSON（默认: ${MODEL_JSON}）
  -d  模型 dim（默认: ${DIM}）
  -H  注意力头数 n_heads（默认: ${N_HEADS}）
  -S  seqlens 列表（默认: ${SEQLENS}）
  -E  额外 Ramulator 参数（默认: 空；示例: --aim）
  -R  cycles 提取用正则（默认: ${METRIC_REGEX})
  -T  Ramulator 命令模板（默认: ${CMD_TEMPLATE})

示例：
  $0 -c pim_config.json -b ./ramulator -r ./pim-config.yaml \\
     -E "--aim" -S "64,128,256,512,1024,2048"

说明：
- 本脚本依次执行：
  1) gentrace.py 生成 score/output/weight（含一组 weight+AF）的 trace 到 traces/；
  2) run_ramulator.py 批量仿真并输出增强版 CSV（含事件计数与 op_size 特征）；
  3) fit_latency_model.py fit 基于 CSV 拟合线性模型，输出 model.json。
- 若你的 Ramulator CLI 不是 --config/--trace，可用 -T 自定义模板，
  例如：-T '{bin} -c {config} -t {trace} {extra}'
EOF
}

choose_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  else
    echo "python"
  fi
}

# --------------- parse args ---------------
while getopts ":c:b:r:o:f:m:d:H:S:E:R:T:h" opt; do
  case $opt in
    c) PIM_CONFIG="$OPTARG" ;;
    b) RAMULATOR_BIN="$OPTARG" ;;
    r) RAM_CONFIG="$OPTARG" ;;
    o) TRACE_DIR="$OPTARG" ;;
    f) RESULTS_CSV="$OPTARG" ;;
    m) MODEL_JSON="$OPTARG" ;;
    d) DIM="$OPTARG" ;;
    H) N_HEADS="$OPTARG" ;;
    S) SEQLENS="$OPTARG" ;;
    E) EXTRA_ARGS="$OPTARG" ;;
    R) METRIC_REGEX="$OPTARG" ;;
    T) CMD_TEMPLATE="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done

# 校验必填
[[ -z "$PIM_CONFIG" ]] && echo "ERROR: -c pim_config.json 必填" >&2 && usage && exit 1
[[ -z "$RAMULATOR_BIN" ]] && echo "ERROR: -b ramulator_bin 必填" >&2 && usage && exit 1
[[ -z "$RAM_CONFIG" ]] && echo "ERROR: -r ramulator_config.yaml 必填" >&2 && usage && exit 1

# --------------- resolve paths ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$(choose_python)

GENTRACE="${SCRIPT_DIR}/gentrace.py"
RUNRAM="${SCRIPT_DIR}/run_ramulator.py"
FIT="${SCRIPT_DIR}/fit_latency_model.py"

# 存在性检查
[[ ! -f "$GENTRACE" ]] && echo "ERROR: 找不到 $GENTRACE" >&2 && exit 1
[[ ! -f "$RUNRAM"   ]] && echo "ERROR: 找不到 $RUNRAM"   >&2 && exit 1
[[ ! -f "$FIT"      ]] && echo "ERROR: 找不到 $FIT"      >&2 && exit 1
[[ ! -f "$PIM_CONFIG" ]] && echo "ERROR: 找不到 PIM config: $PIM_CONFIG" >&2 && exit 1
[[ ! -x "$RAMULATOR_BIN" ]] && echo "WARN: $RAMULATOR_BIN 不可执行，尝试继续..." >&2

mkdir -p "$TRACE_DIR"

echo "==> Python: $PY"
echo "==> Traces out: $TRACE_DIR"
echo "==> Results CSV: $RESULTS_CSV"
echo "==> Model JSON: $MODEL_JSON"

# --------------- 01 生成 trace ----------------
# score + output（多组 seqlen）
echo "[1/3] 生成 score/output traces ..."
$PY "$GENTRACE" \
  --pim-config "$PIM_CONFIG" \
  --ops score,output \
  --seqlens "$SEQLENS" \
  --out-dir "$TRACE_DIR" \
  --dim "$DIM" --n-heads "$N_HEADS"

# weight（attention）：vec=dim, col=dim
echo "[1/3] 生成 weight(attn) traces ..."
$PY "$GENTRACE" \
  --pim-config "$PIM_CONFIG" \
  --ops weight \
  --vector-dims "$DIM" \
  --matrix-cols "$DIM" \
  --out-dir "$TRACE_DIR" \
  --dim "$DIM" --n-heads "$N_HEADS"

# weight + AF（FFN 第一层）：vec=dim, col=4*dim
FFN_COL=$(( 4 * DIM ))
echo "[1/3] 生成 weight+AF(FFN) traces ..."
$PY "$GENTRACE" \
  --pim-config "$PIM_CONFIG" \
  --ops weight \
  --vector-dims "$DIM" \
  --matrix-cols "$FFN_COL" \
  --with-af \
  --out-dir "$TRACE_DIR" \
  --dim "$DIM" --n-heads "$N_HEADS"

# --------------- 02 跑 Ramulator 并汇总 CSV ---------------
echo "[2/3] 运行 Ramulator 并写 CSV ..."
$PY "$RUNRAM" \
  --traces-dir "$TRACE_DIR" \
  --ramulator-bin "$RAMULATOR_BIN" \
  --config "$RAM_CONFIG" \
  --out-csv "$RESULTS_CSV" \
  --metric-regex "$METRIC_REGEX" \
  --cmd-template "$CMD_TEMPLATE" \
  --extra-args "$EXTRA_ARGS"

# --------------- 03 拟合线性模型 ---------------
echo "[3/3] 基于 CSV 拟合线性模型 ..."
# 如果 run_ramulator 生成的是“增强版 CSV”（含 *_calls/_opsize 列），fit 会直接用；
# 若不是增强版，fit 会回退解析 .aim —— 为稳妥，这里同时传 traces 目录。
$PY "$FIT" fit \
  --results-csv "$RESULTS_CSV" \
  --traces-dir "$TRACE_DIR" \
  --out "$MODEL_JSON"

echo "==> 全流程完成："
echo "  Traces 目录: $TRACE_DIR"
echo "  仿真结果 CSV: $RESULTS_CSV"
echo "  拟合模型 JSON: $MODEL_JSON"
