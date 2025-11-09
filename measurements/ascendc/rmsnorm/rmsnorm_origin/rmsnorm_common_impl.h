/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IMPL_NORMALIZATION_RMSNORM_RMSNORM_COMMON_IMPL_H
#define IMPL_NORMALIZATION_RMSNORM_RMSNORM_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#if __CCE_AICORE__ == 220
#include "rmsnorm_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "rmsnorm_v200_impl.h"
#endif
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
namespace AscendC {
namespace RmsNormAPI {
constexpr uint32_t BASIC_BLK_HLENGTH = 64;
constexpr uint32_t BASIC_BLK_BSLENGTH = 8;
constexpr uint32_t FLOAT_PER_BLOCK = 8;
constexpr float RSQRT_EXPONENT = -0.5;

struct RmsNormParams {
    __aicore__ RmsNormParams() {};
    uint32_t curBsLength = 0;
    uint32_t curBshLength = 0;
    LocalTensor<float> tmpAddr;
    LocalTensor<float> reducedAddr;
    LocalTensor<float> srcFp32Addr;
};

template <typename T>
__aicore__ inline void GetRmsNormInfo(
    const LocalTensor<float>& tmpLocal, const RmsNormTiling& tiling, RmsNormParams& params)
{
    params.reducedAddr = tmpLocal;
    params.tmpAddr = tmpLocal[tiling.mainBsLengthAlign];
    if constexpr (sizeof(T) == sizeof(half)) {
        params.srcFp32Addr = tmpLocal[tiling.mainBshLength + tiling.mainBsLengthAlign];
    }
    params.curBsLength = tiling.mainBsLength;
    params.curBshLength = tiling.mainBshLength;
}

// support dst reuse src
__aicore__ inline void RmsNormGenericReduceSum(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t bsLength, const uint32_t hLength, const uint32_t originalHLength)
{
    for (uint32_t i = 0; i < bsLength; i++) {
        uint32_t totalNum = originalHLength;
        LocalTensor<float> srcTmp = src[i * hLength];
        LocalTensor<float> dstTmp = srcTmp;

        while (totalNum > 1) {
            if (totalNum <= ONE_REPEAT_FLOAT_SIZE) {
                dstTmp = dst[i];
            }
            SetVectorMask<float>(0, totalNum);
            RepeatReduceSum<float, false>(dstTmp, srcTmp, 1, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE,
                DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
            PipeBarrier<PIPE_V>();
            totalNum = DivCeil(totalNum, ONE_REPEAT_FLOAT_SIZE);
        }
    }
}

template <bool isBasicBlock = false>
__aicore__ inline void RmsNormReduceSum(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t bsLength, const uint32_t hLength, const uint32_t originalHLength)
{
    if constexpr (isBasicBlock) {
        // split hLength to n*64 parts, all n-parts add to block 0, and use n as for-loop
        //          64*n                       64                       bs
        //     -------------------            -----                   ------
        //  bs | 0 | 1 | 2 |     |  vadd   bs | 0 |  RepeatReduceSum
        //     | 0 | 1 | 2 |     |  --->      | 0 |      ----->
        //     -------------------            -----
        // 1. use add to reduce bs*64*n block to bs*64
        SetVectorMask<float>(0, bsLength * BASIC_BLK_HLENGTH);
        const uint32_t basicBlockNum = hLength / BASIC_BLK_HLENGTH;
        const uint8_t repStride = hLength / FLOAT_PER_BLOCK;
        BinaryRepeatParams binaryParams(1, 1, 1, repStride, repStride, repStride);
        for (uint32_t i = 1; i < basicBlockNum; i++) {
            const uint32_t offset = i * BASIC_BLK_HLENGTH;
            Add<float, false>(src, src, src[offset], MASK_PLACEHOLDER, 1, binaryParams);
            PipeBarrier<PIPE_V>();
        }

        // 2. use vcadd to get sum result
        RepeatReduceSum<float, false>(dst, src, 1, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, repStride);
        PipeBarrier<PIPE_V>();
    } else {
        RmsNormGenericReduceSum(dst, src, bsLength, hLength, originalHLength);
    }
}

// src0 is input: [b,s,h], src1 is reduce resulst: [h,], repeatTime is b*s
// use bsLength as for loop, and muls to do broadcast multiply
__aicore__ inline void RmsNormGeneralFirstAxisBrcMul(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t bshLength, const uint32_t bsLength, const uint32_t hLength)
{
    SetVectorMask<float>(0, hLength);
    UnaryRepeatParams unaryParams;
    auto eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    for (uint32_t i = 0; i < bsLength; i++) {
        const uint32_t offset = i * hLength;
        Muls<float, false>(dst[offset], src0[offset], src1.GetValue(i), MASK_PLACEHOLDER, 1, unaryParams);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, bshLength);
}

// src0's shape is (b*s, h), src1's shape is (b*s)
template <bool isBasicBlock = false>
__aicore__ inline void RmsNormFirstAxisBrcMul(const LocalTensor<float>& dst, const LocalTensor<float>& inputAddr,
    const LocalTensor<float>& reduceAddr, const uint32_t bshLength, const uint32_t bsLength, const uint32_t hLength)
{
    if constexpr (isBasicBlock) {
        if (bsLength > BASIC_BLK_BSLENGTH && bsLength > hLength / BASIC_BLK_HLENGTH) {
            RmsNormBasicBlockBrc(dst, inputAddr, reduceAddr, hLength, bsLength);
            SetVectorMask<float>(0, bshLength);
            BinaryRepeatParams binaryParams;
            Mul<float, false>(dst, dst, inputAddr, MASK_PLACEHOLDER, 1, binaryParams);
            PipeBarrier<PIPE_V>();
        } else {
            RmsNormGeneralFirstAxisBrcMul(dst, inputAddr, reduceAddr, bshLength, bsLength, hLength);
        }
    } else {
        RmsNormGeneralFirstAxisBrcMul(dst, inputAddr, reduceAddr, bshLength, bsLength, hLength);
    }
}

__aicore__ inline void RmsNormLastAxisBrcMulImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t bsLength, const uint32_t hLength)
{
    const uint32_t loop = hLength / BASIC_BLK_HLENGTH;
    if (loop >= bsLength) {
        // for loop size >= bsLength, using bsLength as loop is more efficiency
        BinaryRepeatParams binaryParams;
        SetVectorMask<float>(0, hLength);
        for (uint32_t i = 0; i < bsLength; ++i) {
            uint32_t offset = i * hLength;
            Mul<float, false>(dst[offset], src0[offset], src1, MASK_PLACEHOLDER, 1, binaryParams);
        }
    } else {
        // when loop < bsLength, it means hLength is less than 2048
        SetVectorMask<float>(0, bsLength * BASIC_BLK_HLENGTH);
        const uint16_t repStride = hLength / FLOAT_PER_BLOCK;
        BinaryRepeatParams binaryParams(1, 1, 1, repStride, repStride, 0);
        for (uint32_t i = 0; i < loop; ++i) {
            uint32_t offset = i * BASIC_BLK_HLENGTH;
            Mul<float, false>(dst[offset], src0[offset], src1[offset], MASK_PLACEHOLDER, 1, binaryParams);
        }
        if (hLength % BASIC_BLK_HLENGTH != 0) {
            uint32_t offset = loop * BASIC_BLK_HLENGTH;
            uint32_t tail = hLength - offset;
            SetMaskNorm();
            SetVectorMask<float>(0, (1ull << tail) - 1);
            // bsLength as repeat time, cannot be >= 255
            Mul<float, false>(dst[offset], src0[offset], src1[offset], MASK_PLACEHOLDER, bsLength, binaryParams);
            SetMaskCount();
        }
    }
}

// src0 is [b,s,h], src1 is [h,], repeatTime is b*s
template <bool isBasicBlock = false>
__aicore__ inline void RmsNormLastAxisBrcMul(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t bsLength, const uint32_t hLength)
{
    if constexpr (isBasicBlock) {
        RmsNormLastAxisBrcMulImpl(dst, src0, src1, bsLength, hLength);
    } else {
        if (hLength == BASIC_BLK_HLENGTH) {
            BinaryRepeatParams binaryParams;
            binaryParams.src1RepStride = 0;
            SetVectorMask<float>(0, bsLength * hLength);
            Mul<float, false>(dst, src0, src1, MASK_PLACEHOLDER, 1, binaryParams);
        } else if (hLength < BASIC_BLK_HLENGTH) {
            SetMaskNorm();
            SetVectorMask<float>(0, (1ull << hLength) - 1);
            uint32_t repStride = hLength / FLOAT_PER_BLOCK;
            BinaryRepeatParams binaryParams(1, 1, 1, repStride, repStride, 0);
            Mul<float, false>(dst, src0, src1, MASK_PLACEHOLDER, bsLength, binaryParams);
            SetMaskCount();
        } else {
            RmsNormLastAxisBrcMulImpl(dst, src0, src1, bsLength, hLength);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void RmsNormCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& gamma, const T epsilon, const RmsNormTiling& tiling, RmsNormParams& params)
{
    UnaryRepeatParams unaryParams;
    // for fp16 need to cast to fp32
    SetVectorMask<T>(0, params.curBshLength);
    if constexpr (sizeof(T) == sizeof(half)) {
        unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
        Cast<float, half, false>(params.srcFp32Addr, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
        unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE;
    } else {
        params.srcFp32Addr = src;
    }
    // step 1: x²
    BinaryRepeatParams binaryParams;
    Mul<float, false>(params.tmpAddr, params.srcFp32Addr, params.srcFp32Addr, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    // step 2: ∑x²
    RmsNormReduceSum<isBasicBlock>(
        params.reducedAddr, params.tmpAddr, params.curBsLength, tiling.hLength, tiling.originalHLength);
    // step 3: rms = 1/n*∑
    SetVectorMask<T>(0, params.curBsLength);
    Muls<float, false>(
        params.reducedAddr, params.reducedAddr, tiling.reciprocalOfHLength, MASK_PLACEHOLDER, 1, unaryParams);
    // step 4: rms + e
    PipeBarrier<PIPE_V>();
    Adds<float, false>(params.reducedAddr, params.reducedAddr, epsilon, MASK_PLACEHOLDER, 1, unaryParams);
    // step 5: 1 / (rms + e)
    // high precision impl of rsqrt
    PipeBarrier<PIPE_V>();
    Ln<float, false>(params.reducedAddr, params.reducedAddr, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.reducedAddr, params.reducedAddr, RSQRT_EXPONENT, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Exp<float, false>(params.reducedAddr, params.reducedAddr, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // step 6: broadcast reudce result [b,s] to [b,s,h]
    // step 7: x / (rms + e)
    RmsNormFirstAxisBrcMul<isBasicBlock>(params.tmpAddr, params.srcFp32Addr, params.reducedAddr, params.curBshLength,
        params.curBsLength, tiling.hLength);
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(T) == sizeof(half)) {
        unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
        SetVectorMask<T>(0, tiling.hLength);
        Cast<float, half, false>(
            params.srcFp32Addr, gamma, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
        // step 8: x/(rms) * g
        PipeBarrier<PIPE_V>();
        RmsNormLastAxisBrcMul<isBasicBlock>(
            params.tmpAddr, params.tmpAddr, params.srcFp32Addr, params.curBsLength, tiling.hLength);
        unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE;
        unaryParams.dstRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
        SetVectorMask<T>(0, params.curBshLength);
        Cast<half, float, false>(dst, params.tmpAddr, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    } else {
        // step 8: x/(rms+e) * g
        RmsNormLastAxisBrcMul<isBasicBlock>(dst, params.tmpAddr, gamma, params.curBsLength, tiling.hLength);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void RmsNormImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon,
    const RmsNormTiling& tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_ASSERT((IsSameType<T, half>::value || IsSameType<T, float>::value),
        { KERNEL_LOG(KERNEL_ERROR, "RmsNorm only support data type: float/half"); });

    LocalTensor<float> tmpLocal = sharedTmpBuffer.ReinterpretCast<float>();
    RmsNormParams params;
    GetRmsNormInfo<T>(tmpLocal, tiling, params);
    SetMaskCount();
    for (uint32_t i = 0; i < tiling.loopRound; ++i) {
        uint32_t offset = i * tiling.mainBshLength;
        RmsNormCompute<T, isBasicBlock>(
            dstLocal[offset], srcLocal[offset], gammaLocal, epsilon, tiling, params);
    }
    if (tiling.tailBsLength != 0) {
        params.curBshLength = tiling.tailBshLength;
        params.curBsLength = tiling.tailBsLength;
        RmsNormCompute<T, isBasicBlock>(
            dstLocal[tiling.inputTailPos], srcLocal[tiling.inputTailPos], gammaLocal, epsilon, tiling, params);
    }
    SetMaskNorm();
    ResetMask();
}
}  // namespace RmsNormAPI
} // namespace AscendC
#endif
#endif // IMPL_NORMALIZATION_RMSNORM_RMSNORM_COMMON_IMPL_H