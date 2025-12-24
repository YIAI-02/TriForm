/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_NORMALIZATION_RMSNORM_RMSNORM_V200_IMPL_H
#define IMPL_NORMALIZATION_RMSNORM_RMSNORM_V200_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"

namespace AscendC {
namespace RmsNormAPI {
// use TransData + adds to do broadcast
__aicore__ inline void RmsNormBasicBlockBrc(const LocalTensor<float>& dst,
    const LocalTensor<float>& inputAddr, const LocalTensor<float>& reduceAddr, const uint32_t hLength,
    const uint32_t bsLength)
{
    constexpr uint32_t BASIC_BLK_HLENGTH = 64;
    constexpr uint32_t BASIC_BLK_BSLENGTH = 8;
    constexpr uint32_t FLOAT_PER_BLOCK = 8;
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    constexpr uint32_t step = 2;
    constexpr uint32_t range = NCHW_CONV_ADDR_LIST_SIZE / step;
    for (uint32_t i = 0; i < range; ++i) {
        dstList[i * step] = (uint64_t)dst[i * hLength].GetPhyAddr();
        dstList[i * step + 1] = (uint64_t)dst[i * hLength + FLOAT_PER_BLOCK].GetPhyAddr();
        srcList[i * step] = (uint64_t)reduceAddr.GetPhyAddr();
        srcList[i * step + 1] = (uint64_t)reduceAddr.GetPhyAddr();
    }
    TransDataTo5HDParams params(false, false, bsLength / FLOAT_PER_BLOCK,
                                hLength * BASIC_BLK_BSLENGTH / FLOAT_PER_BLOCK, 1);
    // for v200, there is no brc instr, using transdataTo5HD to broadcast element to block
    TransDataTo5HD<float>(dstList, srcList, params);
    PipeBarrier<PIPE_V>();

    SetVectorMask<float>(0, bsLength * BASIC_BLK_HLENGTH);
    const uint8_t repStride = hLength / FLOAT_PER_BLOCK;
    UnaryRepeatParams unaryParams(1, 0, repStride, repStride);
    for (uint32_t i = 0; i < hLength / BASIC_BLK_HLENGTH; i++) {
        const uint32_t offset = i * BASIC_BLK_HLENGTH;
        Adds<float, false>(dst[offset], dst, 0, MASK_PLACEHOLDER, 1, unaryParams);
    }
    PipeBarrier<PIPE_V>();
}
} // namespace RmsNormAPI
} // namespace AscendC
#endif // IMPL_NORMALIZATION_RMSNORM_RMSNORM_V200_IMPL_H