/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "../kernel_impl/layernorm_custom.h"

__aicore__ inline void CopyTiling(LayerNormCustomKernel::VecTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (int i = 0; i < sizeof(LayerNormCustomKernel::VecTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

extern "C" __global__ __aicore__ void layernorm_custom(GM_ADDR inputXGm, GM_ADDR gammaGm, GM_ADDR betaGm,
    GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputRstdGm, GM_ADDR workspace, GM_ADDR tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    LayerNormCustomKernel::VecTiling tilingData;
    CopyTiling(&tilingData, tiling);
    LayerNormCustomKernel::KernelLayernorm<false> op;
    op.Init(inputXGm, gammaGm, betaGm, outputGm, outputMeanGm, outputRstdGm, tilingData);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void layernorm_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *inputXGm, uint8_t *gammaGm,
    uint8_t *betaGm, uint8_t *outputGm, uint8_t *outputMeanGm, uint8_t *outputRstdGm, uint8_t *workspace,
    uint8_t *tiling)
{
    layernorm_custom<<<blockDim, l2ctrl, stream>>>(inputXGm, gammaGm, betaGm, outputGm, outputMeanGm, 
        outputRstdGm, workspace, tiling);
}
#endif