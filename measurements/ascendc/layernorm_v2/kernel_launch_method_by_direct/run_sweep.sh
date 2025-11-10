#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

SHORT=r:,v:,A:,R:,o:,
LONG=run-mode:,soc-version:,a-set:,r-set:,output-root:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
A_SET_CSV=""
R_SET_CSV=""
OUT_ROOT=""

while :
do
    case "$1" in
        (-r | --run-mode )     RUN_MODE="$2"; shift 2;;
        (-v | --soc-version )  SOC_VERSION="$2"; shift 2;;
        (-A | --a-set )        A_SET_CSV="$2"; shift 2;;
        (-R | --r-set )        R_SET_CSV="$2"; shift 2;;
        (-o | --output-root )  OUT_ROOT="$2"; shift 2;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done
: "${A_SET_CSV:=1,4,16,32,64}"
: "${R_SET_CSV:=256,384,512,768,1024,2048,4096}"
IFS=',' read -r -a A_SET <<< "$A_SET_CSV"
IFS=',' read -r -a R_SET <<< "$R_SET_CSV"

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

# if [ "${RUN_MODE}" = "npu" ]; then
#     ./layernorm_direct_kernel_op
# elif [ "${RUN_MODE}" = "sim" ]; then
#     export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_DIR}
#     export ASCEND_HOME_PATH=${ASCEND_HOME_DIR}
#     msprof op simulator --application=./layernorm_direct_kernel_op
# elif [ "${RUN_MODE}" = "cpu" ]; then
#     ./layernorm_direct_kernel_op
# fi

if [ "${RUN_MODE}" = "npu" ] || [ "${RUN_MODE}" = "cpu" ]; then
    ./layernorm_direct_kernel_op
elif [ "${RUN_MODE}" = "sim" ]; then
    export ASCEND_TOOLKIT_HOME=${ASCEND_HOME_DIR}
    export ASCEND_HOME_PATH=${ASCEND_HOME_DIR}

    mkdir -p msprof_out
    for a in "${A_SET[@]}"; do
      for r in "${R_SET[@]}"; do
        export LN_A="$a"
        export LN_R="$r"
        export LN_NO_IO=1
        tag="a_${a}_r_${r}"
        echo "[SWEEP] running ${tag}"
        msprof op simulator --application=./layernorm_direct_kernel_op
        last_dir=$(ls -dt msprof* profiler* 2>/dev/null | head -n 1 || true)
        if [ -n "${last_dir}" ]; then
          mv -f "${last_dir}" "msprof_out/${tag}" 2>/dev/null || true
        fi
      done
    done
fi
