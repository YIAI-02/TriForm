#!/bin/bash
#SBATCH -J burstgpt_par
#SBATCH -p C064M1024G
#SBATCH --qos=high
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 96:00:00
#SBATCH -o job.%j.out

set +u
set -eo pipefail
trap 'rc=$?; echo "[FATAL] rc=${rc} line=${LINENO} cmd=${BASH_COMMAND}" >&2; exit ${rc}' ERR

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ -d "${SUBMIT_DIR}/src" && -d "${SUBMIT_DIR}/commands" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../src" && -d "${SUBMIT_DIR}/../commands" ]]; then
  PROJECT_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  PROJECT_ROOT="/lustre/home/2501111916/workspace/DOPS_0407_final/TriForm"
fi

CONFIG_PATH="${PROJECT_ROOT}/src/examples/burstgpt_eval_config.json"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="${PROJECT_ROOT}/output/slurm_logs/burstgpt_parallel"
LOG_FILE="${LOG_DIR}/burstgpt_parallel_${JOB_ID}.log"
MARKER_FILE="${LOG_DIR}/burstgpt_parallel_${JOB_ID}.marker"

mkdir -p "${LOG_DIR}"
touch "${MARKER_FILE}"

exec > >(tee -a "${LOG_FILE}") 2>&1

printf '\n========== BurstGPT policy-parallel SLURM job ==========\n'
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NAME=${SLURM_JOB_NAME:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "HOSTNAME=$(hostname)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "LOG_FILE=${LOG_FILE}"
printf '======================================================\n\n'

if [[ ! -d "${PROJECT_ROOT}/src" ]]; then
  echo "[FATAL] PROJECT_ROOT/src does not exist: ${PROJECT_ROOT}/src" >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] Config JSON not found: ${CONFIG_PATH}" >&2
  exit 3
fi

# Activate conda safely with nounset disabled.
set +u
if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi
conda activate triform310
set +u
set -eo pipefail

cd "${PROJECT_ROOT}/src"

echo "[INFO] Current directory: $(pwd)"
echo "[INFO] Python executable: $(which python)"
python --version

python -u main.py --help | grep -n "burstgpt-evaluate" || {
  echo "[FATAL] main.py --help does not show burstgpt-evaluate." >&2
  exit 4
}

echo "[INFO] Checking app.py dispatch..."
python -u - <<'PY_DISPATCH'
import inspect
import mainlib.app as app

src = inspect.getsource(app.main)

if "args.mode == 'burstgpt-evaluate'" not in src and 'args.mode == "burstgpt-evaluate"' not in src:
    raise SystemExit("[FATAL] mainlib/app.py has no burstgpt-evaluate branch.")

bad_outer = "in ('evaluate', 'weight-suggest')" in src or 'in ("evaluate", "weight-suggest")' in src
good_outer = "in ('evaluate', 'weight-suggest', 'burstgpt-evaluate')" in src or 'in ("evaluate", "weight-suggest", "burstgpt-evaluate")' in src

if bad_outer and not good_outer:
    raise SystemExit("[FATAL] mainlib/app.py still has the old outer mode filter; add 'burstgpt-evaluate'.")

print("[INFO] app.py dispatch check passed.", flush=True)
PY_DISPATCH

python -u - "${CONFIG_PATH}" <<'PY_CHECK'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1]).resolve()

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

required = [
    "burstgpt_csv",
    "result_dir",
    "algo",
    "baselines",
    "num_requests",
    "arrival_time_scale",
    "serving_batch_size",
    "prompt_bucket_size",
    "output_bucket_size",
    "output_horizon",
]

missing = [k for k in required if k not in cfg or cfg.get(k) in (None, "")]
if missing:
    raise SystemExit("[FATAL] Missing JSON keys: " + ", ".join(missing))

csv_path = Path(str(cfg["burstgpt_csv"])).expanduser()
if not csv_path.is_absolute():
    csv_path = (cfg_path.parent / csv_path).resolve()

if not csv_path.is_file():
    raise SystemExit(f"[FATAL] burstgpt_csv does not exist: {csv_path}")

workers = int(cfg.get("policy_parallel_workers", cfg.get("parallel_workers", 1)) or 1)
policies = list(cfg.get("baselines") or []) + list(cfg.get("algo") or [])

print("[config] burstgpt_csv            =", csv_path, flush=True)
print("[config] result_dir              =", cfg.get("result_dir"), flush=True)
print("[config] baselines               =", cfg.get("baselines"), flush=True)
print("[config] algo                    =", cfg.get("algo"), flush=True)
print("[config] policy_parallel_workers =", workers, flush=True)
print("[config] total policies          =", len(policies), flush=True)
print("[config] num_requests            =", cfg.get("num_requests"), flush=True)

with csv_path.open("r", encoding="utf-8", errors="replace") as f:
    print("[config] CSV header              =", f.readline().strip(), flush=True)

if workers <= 1 and len(policies) > 1:
    print("[WARN] policy_parallel_workers <= 1; policies will run serially.", flush=True)
PY_CHECK

echo "[INFO] Running JSON-only command:"
echo "       python -u -X faulthandler main.py burstgpt-evaluate --config ${CONFIG_PATH}"

python -u -X faulthandler main.py burstgpt-evaluate --config "${CONFIG_PATH}"

echo "[INFO] Python evaluator finished. Searching for new summaries..."
find "${PROJECT_ROOT}/output" -name 'burstgpt_serving_summary.json' -newer "${MARKER_FILE}" -print || true

SUMMARY_PATH="$(find "${PROJECT_ROOT}/output" -name 'burstgpt_serving_summary.json' -newer "${MARKER_FILE}" -print -quit || true)"

if [[ -z "${SUMMARY_PATH}" ]]; then
  echo "[FATAL] No new burstgpt_serving_summary.json was generated under ${PROJECT_ROOT}/output" >&2
  exit 44
fi

echo "[INFO] Summary generated at: ${SUMMARY_PATH}"

python -u - "${SUMMARY_PATH}" <<'PY_SUMMARY'
import json
import sys

p = sys.argv[1]

with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)

print("[summary] trace_summary =", data.get("trace_summary"), flush=True)

for s in data.get("summaries", []):
    ttft = s.get("ttft") or {}
    tbt = s.get("tbt_token") or s.get("tbt") or {}
    e2e = s.get("e2e_latency") or {}

    print(
        "[summary]",
        s.get("policy"),
        "req=", s.get("request_count"),
        "thr=", s.get("throughput_req_s"),
        "TTFT_p50=", ttft.get("p50_s"),
        "TTFT_p90=", ttft.get("p90_s"),
        "TBT_p50=", tbt.get("p50_s"),
        "TBT_p90=", tbt.get("p90_s"),
        "E2E_p50=", e2e.get("p50_s"),
        "E2E_p90=", e2e.get("p90_s"),
        flush=True,
    )
PY_SUMMARY

echo "[INFO] Job finished successfully."