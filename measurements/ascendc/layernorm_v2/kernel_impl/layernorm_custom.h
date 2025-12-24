/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_NORMALIZATION_LAYERNORM_CUSTOM_H
#define EXAMPLES_NORMALIZATION_LAYERNORM_CUSTOM_H
#include "kernel_operator.h"

namespace LayerNormCustomKernel {
struct VecTiling {
    LayerNormSeparateTiling layernormTilingData;
    uint32_t aLength = 0;
    uint32_t rLength = 0;
    uint32_t rLengthWithPadding = 0;
    float epsilon = 0;
};
// __aicore__ constexpr AscendC::LayerNormConfig GetConfig() {
//     return {.isNoBeta = false, .isNoGamma = false, .isOnlyOutput = false};
// }
template <bool isReuseSource = false>
class KernelLayernorm {
public:
    __aicore__ inline KernelLayernorm() {}
    __aicore__ inline void Init(GM_ADDR inputXGm, GM_ADDR gammGm, GM_ADDR betaGm, GM_ADDR outputGm,
        GM_ADDR outputMeanGm, GM_ADDR outputRstdGm, VecTiling tilingData)
    {
        this->epsilon = tilingData.epsilon;
        tiling_ = tilingData.layernormTilingData;
        this->aLength = tilingData.aLength;
        this->rLength = tilingData.rLength;
        this->rLengthWithPadding = tilingData.rLengthWithPadding;

        arLength = aLength * rLengthWithPadding;

        inputXGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(inputXGm), arLength);
        gammGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gammGm), rLengthWithPadding);
        betaGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(betaGm), rLengthWithPadding);

        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputGm), arLength);
        outputMeanGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputMeanGm), aLength);
        outputRstdGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(outputRstdGm), aLength);

        pipe.InitBuffer(inQueueX, 1, sizeof(float) * arLength);
        pipe.InitBuffer(inQueueGamma, 1, sizeof(float) * rLengthWithPadding);
        pipe.InitBuffer(inQueueBeta, 1, sizeof(float) * rLengthWithPadding);
        pipe.InitBuffer(outQueue, 1, sizeof(float) * arLength);
        pipe.InitBuffer(outQueueMean, 1, sizeof(float) * aLength);
        pipe.InitBuffer(outQueueRstd, 1, sizeof(float) * aLength);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> inputXLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
        AscendC::LocalTensor<float> betaLocal = inQueueBeta.AllocTensor<float>();

        AscendC::DataCopy(inputXLocal, inputXGlobal, arLength);
        AscendC::DataCopy(gammaLocal, gammGlobal, rLengthWithPadding);
        AscendC::DataCopy(betaLocal, betaGlobal, rLengthWithPadding);

        inQueueX.EnQue(inputXLocal);
        inQueueGamma.EnQue(gammaLocal);
        inQueueBeta.EnQue(betaLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> inputXLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> gammaLocal = inQueueGamma.DeQue<float>();
        AscendC::LocalTensor<float> betaLocal = inQueueBeta.DeQue<float>();

        AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> meanLocal = outQueueMean.AllocTensor<float>();
        AscendC::LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();

        // const AscendC::LayerNormPara para = {aLength, rLength, rLengthWithPadding};
        // static constexpr AscendC::LayerNormConfig config = GetConfig();
        // AscendC::LayerNorm<float, float, isReuseSource, config>(outputLocal, meanLocal, rstdLocal, inputXLocal,
        //     gammaLocal, betaLocal, (float)epsilon, para, tiling_);
        const AscendC::LayerNormPara para = {aLength, rLength, rLengthWithPadding};
        AscendC::LayerNorm<float, float, isReuseSource>(outputLocal, meanLocal, rstdLocal,inputXLocal, gammaLocal, betaLocal, (float)epsilon, para, tiling_);

        outQueue.EnQue<float>(outputLocal);
        outQueueMean.EnQue<float>(meanLocal);
        outQueueRstd.EnQue<float>(rstdLocal);

        inQueueX.FreeTensor(inputXLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        inQueueBeta.FreeTensor(betaLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> outputLocal = outQueue.DeQue<float>();
        AscendC::LocalTensor<float> meanLocal = outQueueMean.DeQue<float>();
        AscendC::LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();

        AscendC::DataCopy(outputGlobal, outputLocal, arLength);
        AscendC::DataCopy(outputMeanGlobal, meanLocal, aLength);
        AscendC::DataCopy(outputRstdGlobal, rstdLocal, aLength);

        outQueue.FreeTensor(outputLocal);
        outQueueMean.FreeTensor(meanLocal);
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    AscendC::GlobalTensor<float> inputXGlobal;
    AscendC::GlobalTensor<float> gammGlobal;
    AscendC::GlobalTensor<float> betaGlobal;
    AscendC::GlobalTensor<float> outputGlobal;
    AscendC::GlobalTensor<float> outputMeanGlobal;
    AscendC::GlobalTensor<float> outputRstdGlobal;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBeta;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueMean;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueRstd;

    uint32_t aLength;
    uint32_t rLength;
    uint32_t rLengthWithPadding;
    float epsilon;
    LayerNormSeparateTiling tiling_;

    uint32_t arLength;
};
}
#endif // EXAMPLES_NORMALIZATION_LAYERNORM_CUSTOM_H