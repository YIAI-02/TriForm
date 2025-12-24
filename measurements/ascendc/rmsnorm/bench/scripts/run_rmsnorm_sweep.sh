#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   RUN_MODE=sim SOC_VERSION=Ascend910B3 DTYPE=fp16 REPEAT=10 CASES="1x2048x2048,1x1024x1024" \
#   bash scripts/run_rmsnorm_sweep.sh [-r cpu|sim|npu] [-v SOC] [-i ASCEND_INSTALL_PATH] [-b Debug|Release] [-p INSTALL_PREFIX]
#
# If CASES is empty, falls back to B,S,H env vars.

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})/..
    pwd
)

BUILD_TYPE="${BUILD_TYPE:-Release}"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT="r:v:i:b:p:"
LONG="run-mode:,soc-version:,install-path:,build-type:,install-prefix:"
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :; do
    case "$1" in
    -r | --run-mode) RUN_MODE="$2"; shift 2;;
    -v | --soc-version) SOC_VERSION="$2"; shift 2;;
    -i | --install-path) ASCEND_INSTALL_PATH="$2"; shift 2;;
    -b | --build-type) BUILD_TYPE="$2"; shift 2;;
    -p | --install-prefix) INSTALL_PREFIX="$2"; shift 2;;
    --) shift; break;;
    *) echo "[ERROR]: Unexpected option: $1"; break;;
    esac
done

RUN_MODE="${RUN_MODE:-${RUN_MODE:-sim}}"
SOC_VERSION="${SOC_VERSION:-${SOC_VERSION:-Ascend910B3}}"

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "[ERROR]: RUN_MODE error, only support cpu|sim|npu"
    exit 1
fi

VERSION_LIST="Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST]"
    exit 1
fi

if [ -n "${ASCEND_INSTALL_PATH:-}" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
# source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

# Build
rm -rf "${CURRENT_DIR}/build" "${CURRENT_DIR}/out"
cmake -S "${CURRENT_DIR}" -B "${CURRENT_DIR}/build" \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build "${CURRENT_DIR}/build" -j
cmake --install "${CURRENT_DIR}/build"

BIN="${CURRENT_DIR}/out/bin/rmsnorm_bench"
cp -f "${BIN}" "${CURRENT_DIR}/"

CASES=${CASES:-""}
REPEAT=${REPEAT:-5}
DTYPE=${DTYPE:-fp16}
EPS=${EPS:-1e-5}
NO_IO=${NO_IO:-1}

if [ -z "${CASES}" ]; then
    # fallback to B,S,H
    B=${B:-1}; S=${S:-1}; H=${H:-2048}
    CASES="${B}x${S}x${H}"
fi

IFS=',' read -ra SHAPES <<< "$CASES"
for shape in "${SHAPES[@]}"; do
    B=$(echo $shape | tr 'Xx*' 'xxx' | cut -dx -f1)
    S=$(echo $shape | tr 'Xx*' 'xxx' | cut -dx -f2)
    H=$(echo $shape | tr 'Xx*' 'xxx' | cut -dx -f3)
    OUTDIR="${CURRENT_DIR}/profile/${B}x${S}x${H}"
    mkdir -p "${OUTDIR}"
    echo "[INFO] Profile ${B}x${S}x${H} (repeat=$REPEAT, dtype=${DTYPE}) ..."

    export LD_LIBRARY_PATH=${CURRENT_DIR}/out/lib:${CURRENT_DIR}/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

    if [ "${RUN_MODE}" = "npu" ]; then
        B=${B} S=${S} H=${H} REPEAT=${REPEAT} DTYPE=${DTYPE} EPS=${EPS} NO_IO=${NO_IO} \
        msprof op --application="${CURRENT_DIR}/rmsnorm_bench" --output "${OUTDIR}"
    elif [ "${RUN_MODE}" = "sim" ]; then
        # 部分 msprof 版本在 simulator 子命令下不支持 --output，落到当前目录
        B=${B} S=${S} H=${H} REPEAT=${REPEAT} DTYPE=${DTYPE} EPS=${EPS} NO_IO=${NO_IO} \
        msprof op simulator --application="${CURRENT_DIR}/rmsnorm_bench" || true

        # 尝试把当前目录下的结果搬运到 OUTDIR（不同版本的 msprof 路径可能不同）
        for f in ./*.csv ./*.info ./*.json ./output/*; do
            [ -e "$f" ] || continue
            mv -f "$f" "${OUTDIR}/" || true
        done
        rm -f *.log *.dump *.vcd *.toml *_log || true
    else
        # CPU 路径：不走 msprof，仅做功能验证
        B=${B} S=${S} H=${H} REPEAT=${REPEAT} DTYPE=${DTYPE} EPS=${EPS} NO_IO=${NO_IO} \
        "${CURRENT_DIR}/rmsnorm_bench" | tee "${OUTDIR}/host_log.txt"
    fi
done

echo "[INFO] Done. Profiles are under ./profile/<BxSxH>/"
