#!/bin/bash

SHORT=r:,v:,
LONG=run-mode:,soc-version:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

rm -rf build
mkdir build
cd build

# in case of running op in simulator, use stub so instead
if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's/\/.*\/runtime\/lib64://g')
    export LD_LIBRARY_PATH=$ASCEND_HOME_DIR/runtime/lib64/stub:$LD_LIBRARY_PATH
fi

source $ASCEND_HOME_DIR/bin/setenv.bash
export LD_LIBRARY_PATH=${ASCEND_HOME_DIR}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

cmake  -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION}  -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_DIR} ..
make -j16

if [ "${RUN_MODE}" = "npu" ]; then
    ./fastergelu_direct_kernel_op
elif [ "${RUN_MODE}" = "sim" ]; then
    export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_DIR}
    export ASCEND_HOME_PATH=${ASCEND_HOME_DIR}
    msprof op simulator --application=./fastergelu_direct_kernel_op
elif [ "${RUN_MODE}" = "cpu" ]; then
    ./fastergelu_direct_kernel_op
fi
