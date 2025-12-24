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
#include "../kernel_impl/faster_gelu_custom.h"

__aicore__ inline void CopyTiling(MyCustomKernel::VecTiling* tiling, GM_ADDR tilingGM)
{
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

    for (uint32_t i = 0; i < sizeof(MyCustomKernel::VecTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

extern "C" __global__ __aicore__ void faster_gelu_custom(GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    MyCustomKernel::KernelFasterGelu<float> op;
    MyCustomKernel::VecTiling tilingData;
    CopyTiling(&tilingData, tiling);
    op.Init(srcGm, dstGm, tilingData.dataLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void faster_gelu_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* srcGm, uint8_t* dstGm,
                           uint8_t* workspace, uint8_t* tiling)
{
    // Calling a kernel function with the kernel caller <<<>>>
    faster_gelu_custom<<<blockDim, l2ctrl, stream>>>(srcGm, dstGm, workspace, tiling);
}
#endif