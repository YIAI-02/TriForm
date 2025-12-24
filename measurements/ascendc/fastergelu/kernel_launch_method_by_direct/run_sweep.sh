#!/bin/bash
SHORT=r:,v:,s:
LONG=run-mode:,soc-version:,sizes:
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode ) RUN_MODE="$2"; shift 2;;
        (-v | --soc-version ) SOC_VERSION="$2"; shift 2;;
        (-s | --sizes ) SIZES="$2"; shift 2;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; break;;
    esac
done

rm -rf build
mkdir -p build
cd build

# sim 模式下使用 stub so
if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's/\/.*\/runtime\/lib64://g')
    export LD_LIBRARY_PATH=$ASCEND_HOME_DIR/runtime/lib64/stub:$LD_LIBRARY_PATH
fi

source $ASCEND_HOME_DIR/bin/setenv.bash
export LD_LIBRARY_PATH=${ASCEND_HOME_DIR}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_DIR} ..
make -j16

if [ "${RUN_MODE}" = "npu" ] || [ "${RUN_MODE}" = "cpu" ]; then
    # 应用内部自己循环 sizes；这里可以传 --sizes（npu/cpu 允许带参）
    if [ -n "${SIZES:-}" ]; then
        ./fastergelu_direct_kernel_op --sizes "${SIZES}"
    else
        ./fastergelu_direct_kernel_op
    fi
elif [ "${RUN_MODE}" = "sim" ]; then
    mkdir -p msprof_out
    if [ -z "${SIZES:-}" ]; then
        # 未指定 sizes：按默认长度跑一次
        msprof op simulator \
            --application=./fastergelu_direct_kernel_op \
            --output=msprof_out/len_default
    else
        # 为每个长度单独跑一次 msprof，并通过环境变量传参给应用（不在 --application 后拼接）
        for LEN in ${SIZES}; do
            OUT_DIR="msprof_out/len_${LEN}"
            rm -rf "${OUT_DIR}"
            echo "[SIM] profiling len=${LEN} -> ${OUT_DIR}"
            export SIZES="${LEN}"
            msprof op simulator \
                --application=./fastergelu_direct_kernel_op \
                --output="${OUT_DIR}"
        done
    fi
else
    echo "invalid RUN_MODE: ${RUN_MODE}"
fi
