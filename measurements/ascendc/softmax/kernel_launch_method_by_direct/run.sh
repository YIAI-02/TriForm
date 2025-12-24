#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

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

# export SOFTMAX_CASE=1024x2048
export SOFTMAX_CASES="128x128,128x512,256x1024,256x2048,512x1024,512x2048,512x3072,512x4096,1024x4096"

rm -rf build
mkdir build
cd build

export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_DIR

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
    ./softmax_direct_kernel_op
# elif [ "${RUN_MODE}" = "sim" ]; then
#     export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_DIR}
#     export ASCEND_HOME_PATH=${ASCEND_HOME_DIR}
#     msprof op simulator --application=./softmax_direct_kernel_op
elif [ "${RUN_MODE}" = "sim" ]; then
    export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_DIR}
    export ASCEND_HOME_PATH=${ASCEND_HOME_DIR}
    if [ -n "${SOFTMAX_CASES:-}" ]; then
        IFS=',' read -ra CASE_ARR <<< "${SOFTMAX_CASES}"
        idx=0
        for C in "${CASE_ARR[@]}"; do
            export SOFTMAX_CASE="${C}"
            unset SOFTMAX_CASES
            echo "[msprof] Simulating case ${C}"
            msprof op simulator --application=./softmax_direct_kernel_op
            idx=$((idx+1))
        done
        unset SOFTMAX_CASE
    else
        msprof op simulator --application=./softmax_direct_kernel_op
    fi
elif [ "${RUN_MODE}" = "cpu" ]; then
    ./softmax_direct_kernel_op
fi
