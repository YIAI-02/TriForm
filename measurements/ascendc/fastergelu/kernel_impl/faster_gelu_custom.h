/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_ACTIVATION_FASTERGELU_CUSTOM_H
#define EXAMPLES_ACTIVATION_FASTERGELU_CUSTOM_H
#include "kernel_operator.h"

namespace MyCustomKernel {
struct VecTiling {
    uint32_t dataLength;
    uint32_t sharedTmpBufferSize;
};

template <typename srcType> class KernelFasterGelu {
public:
    __aicore__ inline KernelFasterGelu() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm, uint32_t inputSize)
    {
        dataSize = inputSize;

        srcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(src_gm), dataSize);
        dstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ srcType*>(dst_gm), dataSize);

        pipe.InitBuffer(inQueueX, 1, dataSize * sizeof(srcType));
        pipe.InitBuffer(outQueue, 1, dataSize * sizeof(srcType));
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
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.AllocTensor<srcType>();
        AscendC::DataCopy(srcLocal, srcGlobal, dataSize);
        inQueueX.EnQue(srcLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        AscendC::FasterGelu<srcType>(dstLocal, srcLocal, dataSize);
        // AscendC::FasterGelu<srcType, true, false>(dstLocal, srcLocal, dataSize); //开启高精度模式
        // AscendC::FasterGelu<srcType,false, true>(dstLocal, srcLocal, dataSize); //开启高性能模式
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.DeQue<srcType>();
        AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
        outQueue.FreeTensor(dstLocal);
    }

private:
    AscendC::GlobalTensor<srcType> srcGlobal;
    AscendC::GlobalTensor<srcType> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    uint32_t dataSize = 0;
};
}  // namespace MyCustomKernel
#endif  // EXAMPLES_ACTIVATION_FASTERGELU_CUSTOM_H
