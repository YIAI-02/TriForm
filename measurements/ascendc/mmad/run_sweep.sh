#!/bin/bash
set -e
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

BUILD_TYPE="${BUILD_TYPE:-Debug}"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
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

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "[ERROR]: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

VERSION_LIST="Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
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
source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

# Build
rm -rf build out
mkdir -p build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j
cmake --install build

# Binary path
BIN=./out/bin/ascendc_kernels_bbit
rm -f ascendc_kernels_bbit && cp "$BIN" ./

# Default test set (can be overridden via env CASES)
# CASES=${CASES:-"32x32x32,64x64x64,64x128x64,127x113x91,128x128x128"}
CASES=${CASES:-"1x2048x2048,1x1024x1024"}
REPEAT=${REPEAT:-5}

# Run and profile. IMPORTANT: we do NOT pass any executable args to satisfy `msprof op` limitation.
IFS=',' read -ra SHAPES <<< "$CASES"
for shape in "${SHAPES[@]}"; do
    M=$(echo $shape | tr 'Xx*' 'xxx' | cut -dx -f1)
    N=$(echo $shape | tr 'Xx*' 'xxx' | cut -dx -f2)
    K=$(echo $shape | tr 'Xx*' 'xxx' | cut -dx -f3)
    OUTDIR=profile/${M}x${N}x${K}
    mkdir -p ${OUTDIR}
    echo "[INFO] Profile ${M}x${N}x${K} (repeat=$REPEAT) ..."

    export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

    # Use env to pass shapes to the app; NO args are used.
    if [ "${RUN_MODE}" = "npu" ]; then
        M=${M} N=${N} K=${K} REPEAT=${REPEAT} NO_IO=1 msprof op --application=./ascendc_kernels_bbit --output ${OUTDIR} 
    elif [ "${RUN_MODE}" = "sim" ]; then
        # M=${M} N=${N} K=${K} REPEAT=${REPEAT} NO_IO=1 msprof op simulator --application=./ascendc_kernels_bbit --output ${OUTDIR} 
        M=${M} N=${N} K=${K} REPEAT=${REPEAT} NO_IO=1 msprof op simulator --application=./ascendc_kernels_bbit
    else
        # cpu debug path does not use msprof
        M=${M} N=${N} K=${K} REPEAT=${REPEAT} NO_IO=1 ./ascendc_kernels_bbit
    fi
done

# Clean simulator temp logs
if [ "${RUN_MODE}" = "sim" ]; then
    rm -f *.log *.dump *.vcd *.toml *_log
fi

echo "[INFO] Done. Profiles are under ./profile/<MxNxK>/"
