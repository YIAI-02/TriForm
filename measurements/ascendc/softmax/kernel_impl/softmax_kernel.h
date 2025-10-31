/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_ACTIVATION_SOFTMAX_KERNEL_H
#define EXAMPLES_ACTIVATION_SOFTMAX_KERNEL_H
#include "kernel_operator.h"

namespace MyCustomKernel {
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t FLOAT_NUM_OF_SINGEL_BLOCK = 8;
constexpr uint32_t BASIC_BLOCK_ROW_FACTOR = 8;
constexpr uint32_t BASIC_BLOCK_COLUMN_FACTOR = 64;
constexpr uint32_t BASIC_BLOCK_MAX_COLUMN_LENGTH = 2048;

struct VecTiling {
    uint32_t columnLength = 0;
    uint32_t rowLength = 0;
    uint32_t sharedTmpBufferSize = 0;
    uint32_t usedBlockDim = 0;
    uint32_t coreRowNum = 0;
    uint32_t tailCoreRowNum = 0;
    uint32_t singleLoopCoreRowNum = 0;
    uint32_t singleCoreLoopCount = 0;
    uint32_t singleCoreLoopTail = 0;
    uint32_t tailCoreSingleLoopCoreRowNum = 0;
    uint32_t tailCoreSingleCoreLoopCount = 0;
    uint32_t tailCoreSingleCoreLoopTail = 0;
    SoftMaxTiling softmaxTilingData;
};

class KernelSoftmax {
public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void InitTiling(const VecTiling& tilingData)
    {
        rowLength = tilingData.rowLength;
        sharedTmpBufferSize = tilingData.sharedTmpBufferSize;
        columnLength = tilingData.columnLength;
        usedBlockDim = tilingData.usedBlockDim;
        coreRowNum = tilingData.coreRowNum;
        softmaxTiling = tilingData.softmaxTilingData;
        singleLoopCoreRowNum = tilingData.singleLoopCoreRowNum;
        singleCoreLoopCount = tilingData.singleCoreLoopCount;
        leftRow = tilingData.singleCoreLoopTail;
        tailCoreSingleLoopCoreRowNum = tilingData.tailCoreSingleLoopCoreRowNum;
        tailCoreSingleCoreLoopCount = tilingData.tailCoreSingleCoreLoopCount;
        tailCoreSingleCoreLoopTail = tilingData.tailCoreSingleCoreLoopTail;
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR max, GM_ADDR sum, GM_ADDR z, const VecTiling& tilingData)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        InitTiling(tilingData);

        if (AscendC::GetBlockIdx() == this->usedBlockDim) { // tail core
            this->singleLoopCoreRowNum = this->tailCoreSingleLoopCoreRowNum;
            this->singleCoreLoopCount = this->tailCoreSingleCoreLoopCount;
            this->leftRow = this->tailCoreSingleCoreLoopTail;
        }

        this->blockLength = this->coreRowNum * this->columnLength;
        this->msLength = this->coreRowNum * FLOAT_NUM_OF_SINGEL_BLOCK; // max sum length per block process

        uint32_t offset1 = this->blockLength * AscendC::GetBlockIdx();
        uint32_t offset2 = this->msLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ float*)x + offset1, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float*)z + offset1, this->blockLength);

        maxGm.SetGlobalBuffer((__gm__ float*)max + offset2, this->msLength);
        sumGm.SetGlobalBuffer((__gm__ float*)sum + offset2, this->msLength);


        this->tileLength = this->singleLoopCoreRowNum * this->columnLength;
        pipe.InitBuffer(queueX, BUFFER_NUM, this->tileLength * sizeof(float));

        this->msTileLength = this->singleLoopCoreRowNum * FLOAT_NUM_OF_SINGEL_BLOCK;
        pipe.InitBuffer(queueMax, 1, this->msTileLength * sizeof(float));
        pipe.InitBuffer(queueSum, 1, this->msTileLength * sizeof(float));

        pipe.InitBuffer(sharedTmpBuffer, sharedTmpBufferSize); // 60K tmpbuffer
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetBlockIdx() > this->usedBlockDim) {
            return;
        }

        for (int32_t i = 0; i < this->singleCoreLoopCount; i++) {
            CopyIn(i, this->singleLoopCoreRowNum);
            Compute(i, this->singleLoopCoreRowNum);
            CopyOut(i, this->singleLoopCoreRowNum);
        }
        if (this->leftRow > 0) {
            CopyIn(this->singleCoreLoopCount, this->leftRow);
            Compute(this->singleCoreLoopCount, this->leftRow);
            CopyOut(this->singleCoreLoopCount, this->leftRow);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t rowNum)
    {
        AscendC::LocalTensor<float> xLocal = queueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], rowNum * this->columnLength);
        queueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progressm, uint32_t rowNum)
    {
        AscendC::LocalTensor<float> xLocal = queueX.DeQue<float>();
        AscendC::LocalTensor<float> maxLocal = queueMax.AllocTensor<float>();
        AscendC::LocalTensor<float> sumLocal = queueSum.AllocTensor<float>();
        AscendC::LocalTensor<uint8_t> tmpBuffer = sharedTmpBuffer.Get<uint8_t>();

        AscendC::SoftMaxShapeInfo srcShape = { rowNum, this->columnLength, rowNum, this->columnLength };
        if (rowNum % BASIC_BLOCK_ROW_FACTOR == 0 &&
            this->columnLength % BASIC_BLOCK_COLUMN_FACTOR == 0 &&
            this->columnLength < BASIC_BLOCK_MAX_COLUMN_LENGTH) {
            AscendC::SoftMax<float, true, true>(xLocal, sumLocal, maxLocal, xLocal, tmpBuffer, softmaxTiling, srcShape);
        } else {
            AscendC::SoftMax<float, true>(xLocal, sumLocal, maxLocal, xLocal, tmpBuffer, softmaxTiling, srcShape);
        }
        queueX.EnQue<float>(xLocal);
        queueMax.EnQue<float>(maxLocal);
        queueSum.EnQue<float>(sumLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t rowNum)
    {
        AscendC::LocalTensor<float> zLocal = queueX.DeQue<float>();
        AscendC::LocalTensor<float> maxLocal = queueMax.DeQue<float>();
        AscendC::LocalTensor<float> sumLocal = queueSum.DeQue<float>();

        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, rowNum * this->columnLength);
        AscendC::DataCopy(maxGm[progress * this->msTileLength], maxLocal, rowNum * FLOAT_NUM_OF_SINGEL_BLOCK);
        AscendC::DataCopy(sumGm[progress * this->msTileLength], sumLocal, rowNum * FLOAT_NUM_OF_SINGEL_BLOCK);

        queueX.FreeTensor(zLocal);
        queueMax.FreeTensor(maxLocal);
        queueSum.FreeTensor(sumLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sharedTmpBuffer;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> queueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queueMax, queueSum;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> maxGm;
    AscendC::GlobalTensor<float> sumGm;
    AscendC::GlobalTensor<float> zGm;

    uint32_t blockLength = 0;
    uint32_t usedBlockDim = 0;
    uint32_t msLength = 0;
    uint32_t rowLength = 0;
    uint32_t columnLength = 0;
    uint32_t coreRowNum = 0;
    uint32_t tileLength = 0;
    uint32_t msTileLength = 0;
    uint32_t loopCount = 0;
    uint32_t sharedTmpBufferSize = 0;
    uint32_t singleLoopCoreRowNum = 0;
    uint32_t singleCoreLoopCount = 0;
    uint32_t leftRow = 0;
    uint32_t tailCoreSingleLoopCoreRowNum = 0;
    uint32_t tailCoreSingleCoreLoopCount = 0;
    uint32_t tailCoreSingleCoreLoopTail = 0;
    SoftMaxTiling softmaxTiling;
};
}
#endif // EXAMPLES_ACTIVATION_SOFTMAX_KERNEL_H