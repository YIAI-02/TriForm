#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME=$(basename "$0")
CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_DEBUG=${SCRIPT_DEBUG:-0}
if [[ "$SCRIPT_DEBUG" == "1" ]]; then
  set -x
fi

err_report() {
  local exit_code=$?
  local line_no=${1:-unknown}
  echo "[ERROR] ${SCRIPT_NAME} failed at line ${line_no} (exit=${exit_code})." >&2
  echo "[ERROR] PWD=$(pwd) CURRENT_DIR=${CURRENT_DIR}" >&2
  echo "[ERROR] RUN_MODE=${RUN_MODE:-unset} SOC_VERSION=${SOC_VERSION:-unset} ASCEND_INSTALL_PATH=${_ASCEND_INSTALL_PATH:-unset}" >&2
  exit "$exit_code"
}
trap 'err_report ${LINENO}' ERR

info() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
fatal() { echo "[ERROR] $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || fatal "missing command: $1"; }
require_file() { [[ -f "$1" ]] || fatal "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || fatal "missing directory: $1"; }

RUN_MODE=${RUN_MODE:-sim}
SOC_VERSION=${SOC_VERSION:-Ascend910B1}
BUILD_TYPE=${BUILD_TYPE:-Debug}
INSTALL_PREFIX=${INSTALL_PREFIX:-${CURRENT_DIR}/out}
MODES=${MODES:-nd2nz_a,nz2zz_a,nz2zn_b,nd2zz_a,nd2zn_b}
CASES=${CASES:-32x32x32,64x64x64,64x128x64,128x128x128,127x129x255}
REPEAT=${REPEAT:-10}
INNER_LOOPS=${INNER_LOOPS:-64}
USE_MSPROF=${USE_MSPROF:-1}
PARSER_SCOPE=${PARSER_SCOPE:-bench_body}
CMAKE_BIN=${CMAKE_BIN:-cmake}
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$CURRENT_DIR"

require_cmd "$CMAKE_BIN"
require_cmd "$PYTHON_BIN"
require_file "$CURRENT_DIR/CMakeLists.txt"
require_file "$CURRENT_DIR/main.cpp"
require_file "$CURRENT_DIR/format_conv_bench.cpp"
require_file "$CURRENT_DIR/format_conv_bench_kernel.h"
require_file "$CURRENT_DIR/parse_msprof_summary.py"
require_file "$CURRENT_DIR/fit_to_0315_config.py"

if [[ -n "${ASCEND_INSTALL_PATH:-}" ]]; then
  _ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}
elif [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  _ASCEND_INSTALL_PATH=${ASCEND_HOME_PATH}
elif [[ -d "$HOME/Ascend/ascend-toolkit/latest" ]]; then
  _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
else
  _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

require_dir "${_ASCEND_INSTALL_PATH}"
require_file "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

_saved_err_trap=$(trap -p ERR || true)
trap - ERR
set +e
set +u
set +o pipefail
# shellcheck disable=SC1090
source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"
_setenv_rc=$?
set -o pipefail
set -u
set -e
if [[ -n "${_saved_err_trap:-}" ]]; then
  eval "${_saved_err_trap}"
else
  trap 'err_report ${LINENO}' ERR
fi
if [[ ${_setenv_rc} -ne 0 ]]; then
  warn "${_ASCEND_INSTALL_PATH}/bin/setenv.bash returned ${_setenv_rc}; continuing because some Ascend toolkit versions return non-zero during internal probing under strict shells."
fi
unset _setenv_rc _saved_err_trap

if [[ "${RUN_MODE}" == "sim" ]]; then
  require_dir "${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib"
  export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
elif [[ "${RUN_MODE}" == "cpu" ]]; then
  require_dir "${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib"
  export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
elif [[ "${RUN_MODE}" == "npu" ]]; then
  :
else
  fatal "invalid RUN_MODE=${RUN_MODE}; expected sim/cpu/npu"
fi

if [[ "${USE_MSPROF}" == "1" ]]; then
  require_cmd msprof
fi

BUILD_DIR="${CURRENT_DIR}/build"
OUT_DIR="${CURRENT_DIR}/out"
PROFILE_DIR="${CURRENT_DIR}/profile"
BIN="${OUT_DIR}/bin/format_conv_bench_app"

info "CURRENT_DIR=${CURRENT_DIR}"
info "RUN_MODE=${RUN_MODE} SOC_VERSION=${SOC_VERSION} BUILD_TYPE=${BUILD_TYPE}"
info "ASCEND_INSTALL_PATH=${_ASCEND_INSTALL_PATH}"
info "MODES=${MODES}"
info "CASES=${CASES}"
info "REPEAT=${REPEAT} INNER_LOOPS=${INNER_LOOPS} USE_MSPROF=${USE_MSPROF} PARSER_SCOPE=${PARSER_SCOPE}"

rm -rf "$BUILD_DIR" "$OUT_DIR" "$PROFILE_DIR"
mkdir -p "$BUILD_DIR" "$PROFILE_DIR"

"$CMAKE_BIN" -S "$CURRENT_DIR" -B "$BUILD_DIR" \
  -DRUN_MODE="${RUN_MODE}" \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DASCEND_CANN_PACKAGE_PATH="${_ASCEND_INSTALL_PATH}"
"$CMAKE_BIN" --build "$BUILD_DIR" -j
"$CMAKE_BIN" --install "$BUILD_DIR"

[[ -x "${BIN}" ]] || fatal "missing executable: ${BIN}"

IFS=',' read -ra MODES_ARR <<< "${MODES}"
IFS=',' read -ra CASES_ARR <<< "${CASES}"
[[ ${#MODES_ARR[@]} -gt 0 ]] || fatal "empty MODES"
[[ ${#CASES_ARR[@]} -gt 0 ]] || fatal "empty CASES"

for mode in "${MODES_ARR[@]}"; do
  for shape in "${CASES_ARR[@]}"; do
    M=$(echo "$shape" | cut -dx -f1)
    N=$(echo "$shape" | cut -dx -f2)
    K=$(echo "$shape" | cut -dx -f3)
    [[ -n "$M" && -n "$N" && -n "$K" ]] || fatal "bad shape entry: ${shape}; expected MxNxK"
    OUTDIR="${PROFILE_DIR}/${mode}/${M}x${N}x${K}"
    mkdir -p "${OUTDIR}"
    info "mode=${mode} shape=${M}x${N}x${K} repeat=${REPEAT} inner_loops=${INNER_LOOPS}"
    (
      export LD_LIBRARY_PATH="${OUT_DIR}/lib:${OUT_DIR}/lib64:${_ASCEND_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/stub:/usr/local/Ascend/ascend-toolkit/latest/hccl/lib64/stub"
      if [[ "${USE_MSPROF}" == "1" ]]; then
        if [[ "${RUN_MODE}" == "sim" ]]; then
          msprof op simulator \
            --soc-version="${SOC_VERSION}" \
            --output="${OUTDIR}" \
            "${BIN}" \
            --mode "${mode}" \
            --m "${M}" \
            --n "${N}" \
            --k "${K}" \
            --repeat "${REPEAT}" \
            --inner_loops "${INNER_LOOPS}"
        elif [[ "${RUN_MODE}" == "npu" ]]; then
          msprof op \
            --output="${OUTDIR}" \
            "${BIN}" \
            --mode "${mode}" \
            --m "${M}" \
            --n "${N}" \
            --k "${K}" \
            --repeat "${REPEAT}" \
            --inner_loops "${INNER_LOOPS}"
        else
          "${BIN}" --mode "${mode}" --m "${M}" --n "${N}" --k "${K}" --repeat "${REPEAT}" --inner_loops "${INNER_LOOPS}"
        fi
      else
        "${BIN}" --mode "${mode}" --m "${M}" --n "${N}" --k "${K}" --repeat "${REPEAT}" --inner_loops "${INNER_LOOPS}"
      fi
    )
  done
done

"${PYTHON_BIN}" "$CURRENT_DIR/parse_msprof_summary.py" \
  --root "$PROFILE_DIR" \
  --out "$PROFILE_DIR/format_conv_results.csv" \
  --repeat "$REPEAT" \
  --inner-loops "$INNER_LOOPS" \
  --scope "$PARSER_SCOPE" || warn "parse_msprof_summary.py failed"
"${PYTHON_BIN}" "$CURRENT_DIR/fit_to_0315_config.py" --csv "$PROFILE_DIR/format_conv_results.csv" --out "$PROFILE_DIR/format_conv_fit.json" || warn "fit_to_0315_config.py failed"

info "done"
info "results: ${PROFILE_DIR}/format_conv_results.csv"
info "fits:    ${PROFILE_DIR}/format_conv_fit.json"
