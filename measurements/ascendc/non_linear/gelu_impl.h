/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_GELU_GELU_IMPL_H
#define IMPL_ACTIVATION_GELU_GELU_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"

namespace AscendC {
template <typename T>
__aicore__ inline void GeluCalcTanhParams(const LocalTensor<T>& tempTensorA, const LocalTensor<T>& tempTensorB,
    const LocalTensor<T>& srcLocal, const GeluParams<T>& params)
{
    const T coefficientsA = 0.044715;
    const T coefficientsB = 1.5957691216057308;

    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    // 1.5957691216057308 *( x + 0.044715*x^3)
    Mul<T, false>(tempTensorA, srcLocal, srcLocal, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<T, false>(tempTensorB, srcLocal, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<T, false>(tempTensorA, tempTensorB, coefficientsA, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Add<T, false>(tempTensorB, srcLocal, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<T, false>(tempTensorA, tempTensorB, coefficientsB, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GeluCalcYGreaterThanZero(const LocalTensor<T>& tempTensorA, const LocalTensor<T>& tempTensorB,
    const GeluParams<T>& params)
{
    const UnaryRepeatParams unaryParams;

    // exp(min(y, 0)) to avoid overflow, keep exp negative
    Mins<T, false>(tempTensorB, tempTensorA, 0, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<T, false>(tempTensorB, tempTensorB, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool highPerformance = false>
__aicore__ inline void GeluCalcYLessThanZero(const LocalTensor<T>& tempTensorA, const LocalTensor<T>& tempTensorB,
    const LocalTensor<T>& srcLocal, const GeluParams<T>& params)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    // x / (exp^(-abs(y)) + 1)
    Abs<T, false>(tempTensorA, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Muls<T, false>(tempTensorA, tempTensorA, -1, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<T, false>(tempTensorA, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<T, false>(tempTensorA, tempTensorA, 1, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    if constexpr (highPerformance) {
        Reciprocal<T, false>(tempTensorA, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
        PipeBarrier<PIPE_V>();

        Mul<T, false>(tempTensorA, srcLocal, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        PipeBarrier<PIPE_V>();
    } else {
        Div<T, false>(tempTensorA, srcLocal, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool highPerformance = false>
__aicore__ inline void GeluCalcSimplifiedAvoid(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const GeluParams<T>& params)
{
    const BinaryRepeatParams binaryParams;
    const LocalTensor<T>& tempTensorA = params.tempTensorA;
    const LocalTensor<T>& tempTensorB = params.tempTensorB;

    // y = (input_x + 0.044715 * input_x ^ 3) * 1.5957691
    // x / (exp^(-abs(y)) + 1) * exp(min(y, 0))
    GeluCalcTanhParams(tempTensorA, tempTensorB, srcLocal, params);

    // exp(min(y, 0)) to avoid overflow, keep exp negative
    GeluCalcYGreaterThanZero(tempTensorA, tempTensorB, params);

    // x / (exp^(-abs(y)) + 1)
    GeluCalcYLessThanZero<T, highPerformance>(tempTensorA, tempTensorB, srcLocal, params);

    // x / (exp^(-abs(y)) + 1) * exp(min(y, 0))
    Mul<T, false>(dstLocal, tempTensorA, tempTensorB, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool highPerformance = false>
__aicore__ inline void FastGeluCalcSimplified(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const GeluParams<T>& params)
{
    const LocalTensor<T>& stackBuffer = params.tempTensorA;

    // fast_gelu(x) = x / (1 + e^(-1.702x))
    const T coefficients = -1.702;

    //  x1 = 1 + e^(-1.702x)
    const UnaryRepeatParams unaryParams;
    Muls<T, false>(stackBuffer, srcLocal, coefficients, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<T, false>(stackBuffer, stackBuffer, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<T, false>(stackBuffer, stackBuffer, 1.0, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    // fast_gelu(x) = x / x1
    const BinaryRepeatParams binaryParams;
    if constexpr (highPerformance) {
        Reciprocal<T, false>(stackBuffer, stackBuffer, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
        PipeBarrier<PIPE_V>();

        Mul<T, false>(dstLocal, srcLocal, stackBuffer, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        PipeBarrier<PIPE_V>();
    } else {
        Div<T, false>(dstLocal, srcLocal, stackBuffer, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void FastGeluV2ClipParams(const LocalTensor<T>& tempTensorA, const LocalTensor<T>& srcLocal,
    const GeluParams<T>& params)
{
    const T coefficientsA = -0.1444;
    const T coefficientsB = -1.769;
    const T coefficientsBInv = 1.769;
    const T coefficientsC = 0.7071;
    const T coefficientsD = 0.5;

    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    // x1 = (-0.1444) * (clip(|0.7071 * x|, max = 1.769) - 1.769) ^ 2 + 0.5
    Muls<T, false>(tempTensorA, srcLocal, coefficientsC, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Abs<T, false>(tempTensorA, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Mins<T, false>(tempTensorA, tempTensorA, coefficientsBInv, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<T, false>(tempTensorA, tempTensorA, coefficientsB, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<T, false>(tempTensorA, tempTensorA, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<T, false>(tempTensorA, tempTensorA, coefficientsA, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<T, false>(tempTensorA, tempTensorA, coefficientsD, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool highPerformance = false>
__aicore__ inline void FastGeluV2CalcSimplified(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const GeluParams<T>& params)
{
    const T coefficients = 0.000000000001;
    const T coefficientsHalf = 0.5;
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    const LocalTensor<T>& tempTensorA = params.tempTensorA;
    const LocalTensor<T>& tempTensorB = params.tempTensorB;
    const LocalTensor<T>& tempTensorC = params.tempTensorC;

    // x1 = (-0.1444) * (clip(|0.7071 * x|, max = 1.769) - 1.769) ^ 2 + 0.5
    FastGeluV2ClipParams(tempTensorA, srcLocal, params);

    // x2 = (x + 0.000000000001) / |(x + 0.000000000001)|
    Adds<T, false>(tempTensorB, srcLocal, coefficients, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Abs<T, false>(tempTensorC, tempTensorB, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    if constexpr (highPerformance) {
        Reciprocal<T, false>(tempTensorC, tempTensorC, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
        PipeBarrier<PIPE_V>();

        Mul<T, false>(tempTensorB, tempTensorB, tempTensorC, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        PipeBarrier<PIPE_V>();
    } else {
        Div<T, false>(tempTensorB, tempTensorB, tempTensorC, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
        PipeBarrier<PIPE_V>();
    }

    // fast_gelu_v2(x) = x * (x2 * x1 + 0.5)
    Mul<T, false>(tempTensorA, tempTensorA, tempTensorB, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<T, false>(tempTensorA, tempTensorA, coefficientsHalf, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<T, false>(dstLocal, srcLocal, tempTensorA, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool highPrecision = false, uint32_t bufferNumber = 1>
__aicore__ inline void GeluFormulasTmpCalc(GeluParams<T>& params)
{
    uint32_t needConvBuffer = bufferNumber;
    if constexpr (highPrecision) {
        needConvBuffer += 1;
    }

    params.tempTensorA = params.sharedTmpBuffer;
    params.stackSize = params.tmpBufferSize / needConvBuffer / ONE_BLK_SIZE * ONE_BLK_SIZE;
    ASCENDC_ASSERT((params.stackSize > 0), { KERNEL_LOG(KERNEL_ERROR, "params.stackSize must > 0!"); });

    uint32_t nextTmpPos = params.stackSize;
    if constexpr (bufferNumber == TWO_OF_STACK_BUFFER) {
        params.tempTensorB = params.sharedTmpBuffer[nextTmpPos];
        nextTmpPos += params.stackSize;
    }

    if constexpr (bufferNumber >= THREE_OF_STACK_BUFFER) {
        params.tempTensorB = params.sharedTmpBuffer[nextTmpPos];
        nextTmpPos += params.stackSize;
        params.tempTensorC = params.sharedTmpBuffer[nextTmpPos];
        nextTmpPos += params.stackSize;
    }

    if constexpr (highPrecision) {
        params.tempTensorConv = params.sharedTmpBuffer[nextTmpPos];
    }
}

__aicore__ inline void GeluCastIntrinsicsImpl(const LocalTensor<float>& dstLocal, const LocalTensor<half>& srcLocal)
{
    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
    Cast<float, half, false>(dstLocal, srcLocal, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GeluCastIntrinsicsImpl(const LocalTensor<half>& dstLocal, const LocalTensor<float>& srcLocal)
{
    UnaryRepeatParams unaryParams;
    unaryParams.dstRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
    Cast<half, float, false>(dstLocal, srcLocal, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <uint32_t bufferNumber = 1>
__aicore__ inline void GeluFormulasHighPrecision(const LocalTensor<half>& dstLocal, const LocalTensor<half>& srcLocal,
    GeluParams<float>& params,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const GeluParams<float>&))
{
    GeluFormulasTmpCalc<float, true, bufferNumber>(params);

    const LocalTensor<float>& stackBufferConv = params.tempTensorConv;

    const uint32_t round = params.dataSize / params.stackSize;
    const uint32_t tail = params.dataSize % params.stackSize;

    SetMaskCount();
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, params.stackSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        GeluCastIntrinsicsImpl(stackBufferConv, srcLocal[offset]);

        func(stackBufferConv, stackBufferConv, params);

        GeluCastIntrinsicsImpl(dstLocal[offset], stackBufferConv);
        offset = offset + params.stackSize;
    }

    if (tail != 0) {
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tail);

        GeluCastIntrinsicsImpl(stackBufferConv, srcLocal[offset]);

        func(stackBufferConv, stackBufferConv, params);

        GeluCastIntrinsicsImpl(dstLocal[offset], stackBufferConv);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, uint32_t bufferNumber = 1>
__aicore__ inline void GeluFormulas(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    GeluParams<T>& params, void (*func)(const LocalTensor<T>&, const LocalTensor<T>&, const GeluParams<T>&))
{
    GeluFormulasTmpCalc<T, false, bufferNumber>(params);

    const uint32_t round = params.dataSize / params.stackSize;
    const uint32_t tail = params.dataSize % params.stackSize;

    SetMaskCount();
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, params.stackSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        func(dstLocal[offset], srcLocal[offset], params);
        offset = offset + params.stackSize;
    }

    if (tail != 0) {
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tail);
        func(dstLocal[offset], srcLocal[offset], params);
    }

    SetMaskNorm();
    ResetMask();
}

template <uint32_t bufferNumber = 1>
__aicore__ inline void GeluClass(const LocalTensor<half>& dstLocal, const LocalTensor<half>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const GeluParams<float>&))
{
    GeluParams<float> params;
    params.dataSize = dataSize;
    params.sharedTmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    params.tmpBufferSize = sharedTmpBuffer.GetSize() / sizeof(float);

    ASCENDC_ASSERT((params.tmpBufferSize > 0), { KERNEL_LOG(KERNEL_ERROR, "params.tmpBufferSize must > 0!"); });
    GeluFormulasHighPrecision<bufferNumber>(dstLocal, srcLocal, params, func);
}

template <typename T, uint32_t bufferNumber = 1>
__aicore__ inline void GeluClass(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize,
    void (*func)(const LocalTensor<T>&, const LocalTensor<T>&, const GeluParams<T>&))
{
    GeluParams<T> params;
    params.dataSize = dataSize;
    params.sharedTmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
    params.tmpBufferSize = sharedTmpBuffer.GetSize() / sizeof(T);

    ASCENDC_ASSERT((params.tmpBufferSize > 0), { KERNEL_LOG(KERNEL_ERROR, "params.tmpBufferSize must > 0!"); });
    GeluFormulas<T, bufferNumber>(dstLocal, srcLocal, params, func);
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void GeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    if constexpr (highPrecision && (sizeof(T) == sizeof(half))) {
        GeluClass<TWO_OF_STACK_BUFFER>(dstLocal, srcLocal, sharedTmpBuffer, dataSize,
            GeluCalcSimplifiedAvoid<float, highPerformance>);
    } else {
        GeluClass<T, TWO_OF_STACK_BUFFER>(dstLocal, srcLocal, sharedTmpBuffer, dataSize,
            GeluCalcSimplifiedAvoid<T, highPerformance>);
    }
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void GeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dataSize)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    GeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    if constexpr (highPrecision && (sizeof(T) == sizeof(half))) {
        GeluClass(dstLocal, srcLocal, sharedTmpBuffer, dataSize, FastGeluCalcSimplified<float, highPerformance>);
    } else {
        GeluClass(dstLocal, srcLocal, sharedTmpBuffer, dataSize, FastGeluCalcSimplified<T, highPerformance>);
    }
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    FasterGeluImpl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluV2Impl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t dataSize)
{
    if constexpr (highPrecision && (IsSameType<T, half>::value)) {
        GeluClass<THREE_OF_STACK_BUFFER>(dstLocal, srcLocal, sharedTmpBuffer, dataSize,
            FastGeluV2CalcSimplified<float, highPerformance>);
    } else {
        GeluClass<T, THREE_OF_STACK_BUFFER>(dstLocal, srcLocal, sharedTmpBuffer, dataSize,
            FastGeluV2CalcSimplified<T, highPerformance>);
    }
}

template <typename T, bool highPrecision = false, bool highPerformance = false>
__aicore__ inline void FasterGeluV2Impl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint32_t dataSize)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    FasterGeluV2Impl<T, highPrecision, highPerformance>(dstLocal, srcLocal, sharedTmpBuffer, dataSize);
}
#pragma end_pipe
} // namespace AscendC
#endif // IMPL_ACTIVATION_GELU_GELU_IMPL_H