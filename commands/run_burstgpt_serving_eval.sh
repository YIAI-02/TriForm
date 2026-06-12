#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; echo "[FATAL] rc=$rc line=$LINENO cmd=$BASH_COMMAND"; exit $rc' ERR

DEFAULT_CSV="/lustre/home/2501111916/workspace/DOPS_0407_final/TriForm/data/realworld/BurstGPT_without_fails_1.csv"
CSV_PATH="${1:-$DEFAULT_CSV}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v realpath >/dev/null 2>&1; then
  CSV_PATH="$(realpath "$CSV_PATH")"
else
  CSV_PATH="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$CSV_PATH")"
fi

LOG_DIR="$ROOT_DIR/output/burstgpt_debug_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[ERROR] BurstGPT CSV not found: $CSV_PATH"
  exit 1
fi

echo "[INFO] Debug log: $LOG_FILE"
echo "[INFO] Project root: $ROOT_DIR"
echo "[INFO] BurstGPT CSV: $CSV_PATH"
echo "[INFO] Checking CSV header..."
head -1 "$CSV_PATH"

cd "$ROOT_DIR/src"

export TRIFORM_SKIP_PYCACHE_PURGE=1
export PYTHONUNBUFFERED=1
export DOPS_BURSTGPT_DEBUG="${DOPS_BURSTGPT_DEBUG:-1}"

echo "[INFO] CLI help smoke check:"
python -u main.py --help | sed -n '1,50p'

echo "[INFO] BurstGPT loader smoke check:"
python -u - "$CSV_PATH" <<'PYCODE_LOADER'
import sys
from mainlib.burstgpt_serving_eval import load_burstgpt_csv, _trace_summary
csv_path = sys.argv[1]
reqs = load_burstgpt_csv(csv_path, max_requests=5, skip_zero_output=True, arrival_time_scale=0.1, max_input_len=4096, max_output_len=1024)
print("[loader] loaded", len(reqs), "requests", flush=True)
print("[loader] first", reqs[0].as_dict(), flush=True)
print("[loader] summary", _trace_summary(reqs), flush=True)
PYCODE_LOADER

CMD=(
  python -u -X faulthandler main.py burstgpt-evaluate
  --config examples/burstgpt_eval_config.json
  --burstgpt_csv "$CSV_PATH"
)

echo "[INFO] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"