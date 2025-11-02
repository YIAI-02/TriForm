#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
INSTALL_PREFIX="${CURRENT_DIR}/out"
BUILD_TYPE="${BUILD_TYPE:-Release}"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :; do
  case "$1" in
    -r|--run-mode) RUN_MODE="$2"; shift 2;;
    -v|--soc-version) SOC_VERSION="$2"; shift 2;;
    -i|--install-path) ASCEND_INSTALL_PATH="$2"; shift 2;;
    -b|--build-type) BUILD_TYPE="$2"; shift 2;;
    -p|--install-prefix) INSTALL_PREFIX="$2"; shift 2;;
    --) shift; break;;
    *) echo "[ERROR] Unexpected option: $1"; exit 3;;
  esac
done

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" ${RUN_MODE:-npu} "* ]]; then
    echo "[ERROR]: RUN_MODE should be one of: cpu sim npu"
    exit 1
fi

# Resolve Ascend Toolkit path
if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then
  _ASCEND_INSTALL_PATH="$ASCEND_INSTALL_PATH"
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
  _ASCEND_INSTALL_PATH="$ASCEND_HOME_PATH"
elif [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
  _ASCEND_INSTALL_PATH="$ASCEND_TOOLKIT_HOME"
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
  _ASCEND_INSTALL_PATH="$HOME/Ascend/ascend-toolkit/latest"
else
  _ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
fi
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"

# Build
rm -rf "${CURRENT_DIR}/build" "${INSTALL_PREFIX}"
cmake -S "${CURRENT_DIR}" -B "${CURRENT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DASCEND_CANN_PACKAGE_PATH="${_ASCEND_INSTALL_PATH}"
cmake --build "${CURRENT_DIR}/build" -j
cmake --install "${CURRENT_DIR}/build"

BIN="${INSTALL_PREFIX}/bin/op_runner"
rm -f "${CURRENT_DIR}/op_runner" && cp "${BIN}" "${CURRENT_DIR}/"

export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${INSTALL_PREFIX}/lib64:${_ASCEND_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH:-}"

REPEAT="${REPEAT:-5}"
WARMUP="${WARMUP:-1}"
DTYPE="${DTYPE:-fp16}"
EPS="${EPS:-1e-6}"

# ------------------ Cases ------------------
# 1) GELU: DIMS like "B x S x H"; default uses flatten semantics (elementwise)
CASES_GELU="${CASES_GELU:-"1x2048x2048,1x4096x4096,32x4096"}"

# 2) RMSNorm (elementwise normalization across last dim)
CASES_RMS="${CASES_RMS:-"1x2048x4096,4x2048x8192"}"

# 3) RMSNorm + GEMM: DIMS as "B x S x H x O" (O = output dim). If 3D given, O=H.
CASES_RMSGEMM="${CASES_RMSGEMM:-"1x2048x4096x4096,1x2048x4096x12288,4x1024x3072x4096"}"

# ------------------------------------------

mkdir -p "${CURRENT_DIR}/profile"
run_one() {
  local OP="$1"; shift
  local CASES="$1"; shift
  IFS=',' read -ra SHAPES <<< "$CASES"
  for shape in "${SHAPES[@]}"; do
    local dims=$(echo "$shape" | tr 'Xx*' 'xxx' | tr -d ' ' )
    local tokens=(${dims//x/ })
    local out=""
    if [ "${OP}" = "rmsnorm_gemm" ]; then
        local B=${tokens[0]:-1}
        local S=${tokens[1]:-1}
        local H=${tokens[2]:-1}
        local O=${tokens[3]:-$H}
        local OUTDIR="${CURRENT_DIR}/profile/${OP}/${B}x${S}x${H}x${O}"
        mkdir -p "${OUTDIR}"
        echo "[INFO] Profile ${OP} ${B}x${S}x${H}x${O} (repeat=${REPEAT})"
        if [ "${RUN_MODE}" = "npu" ]; then
            OP=${OP} DIMS="${B}x${S}x${H}" OUT=${O} REPEAT=${REPEAT} WARMUP=${WARMUP} DTYPE=${DTYPE} EPS=${EPS} \
              msprof op --application="${CURRENT_DIR}/op_runner" --output "${OUTDIR}" | tee "${OUTDIR}/run.log"
        elif [ "${RUN_MODE}" = "sim" ]; then
            OP=${OP} DIMS="${B}x${S}x${H}" OUT=${O} REPEAT=${REPEAT} WARMUP=${WARMUP} DTYPE=${DTYPE} EPS=${EPS} \
              msprof op simulator --application="${CURRENT_DIR}/op_runner" | tee "${OUTDIR}/run.log"
        else
            OP=${OP} DIMS="${B}x${S}x${H}" OUT=${O} REPEAT=${REPEAT} WARMUP=${WARMUP} DTYPE=${DTYPE} EPS=${EPS} \
              "${CURRENT_DIR}/op_runner" | tee "${OUTDIR}/run.log"
        fi
    else
        local OUTDIR="${CURRENT_DIR}/profile/${OP}/${dims}"
        mkdir -p "${OUTDIR}"
        echo "[INFO] Profile ${OP} ${dims} (repeat=${REPEAT})"
        if [ "${RUN_MODE}" = "npu" ]; then
            OP=${OP} DIMS="${dims}" REPEAT=${REPEAT} WARMUP=${WARMUP} DTYPE=${DTYPE} EPS=${EPS} \
              msprof op --application="${CURRENT_DIR}/op_runner" --output "${OUTDIR}" | tee "${OUTDIR}/run.log"
        elif [ "${RUN_MODE}" = "sim" ]; then
            OP=${OP} DIMS="${dims}" REPEAT=${REPEAT} WARMUP=${WARMUP} DTYPE=${DTYPE} EPS=${EPS} \
              msprof op simulator --application="${CURRENT_DIR}/op_runner" | tee "${OUTDIR}/run.log"
        else
            OP=${OP} DIMS="${dims}" REPEAT=${REPEAT} WARMUP=${WARMUP} DTYPE=${DTYPE} EPS=${EPS} \
              "${CURRENT_DIR}/op_runner" | tee "${OUTDIR}/run.log"
        fi
    fi
  done
}

# Run the three sets
run_one gelu "${CASES_GELU}"
run_one rmsnorm "${CASES_RMS}"
run_one rmsnorm_gemm "${CASES_RMSGEMM}"

echo "[INFO] Done. Profiles are under ${CURRENT_DIR}/profile/"
