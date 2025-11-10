#!/usr/bin/python3
# coding=utf-8

# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import numpy as np
import os

def gen_golden_data_simple():
    shape = [32, 32]
    inputX = np.arange(shape[0]*shape[1]).reshape(shape).astype(np.float32)
    gamma = np.arange(shape[1]).reshape(shape[1]).astype(np.float32)
    beta = np.arange(shape[1]).reshape(shape[1]).astype(np.float32)

    reduce_axis = 1
    mean  = np.mean(inputX, reduce_axis, keepdims=True)
    variance = np.mean(np.power((inputX - mean),2), reduce_axis, keepdims=True)
    result = gamma*((inputX - mean) / np.sqrt(variance + 0.0001)) + beta
    rstd = 1 / np.sqrt(variance + 0.0001)

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    inputX.tofile("./input/input_X.bin")
    gamma.tofile("./input/input_gamma.bin")
    beta.tofile("./input/input_beta.bin")
    result.tofile("./output/golden_output_result.bin")
    mean.tofile("./output/golden_output_mean.bin")
    rstd.tofile("./output/golden_output_rstd.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
