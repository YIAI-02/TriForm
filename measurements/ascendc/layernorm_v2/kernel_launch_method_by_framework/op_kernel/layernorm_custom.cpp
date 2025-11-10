/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../../../kernel_impl/layernorm_custom.h"

extern "C" __global__ __aicore__ void layernorm_custom(GM_ADDR inputXGm, GM_ADDR gammaGm, GM_ADDR betaGm,
    GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputRstdGm, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    LayerNormCustomKernel::VecTiling vecTiling = *reinterpret_cast<LayerNormCustomKernel::VecTiling*>(&tilingData);
    if (TILING_KEY_IS(1)) {
        LayerNormCustomKernel::KernelLayernorm<false> op;
        op.Init(inputXGm, gammaGm, betaGm, outputGm, outputMeanGm, outputRstdGm, vecTiling);
        op.Process();
    }
}