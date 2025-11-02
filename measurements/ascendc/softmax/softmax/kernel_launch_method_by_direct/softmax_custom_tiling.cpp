/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling/tiling_api.h"
#include "../host_tiling/softmax_custom_tiling.h"

void GenerateTiling(const uint32_t rowNum, const uint32_t colNum, const uint32_t coreNum, const uint32_t tilingSize,
                    uint8_t* tilingBuffer)
{
    optiling::SoftmaxCustomTilingData tiling;
    SoftmaxCustomTiling::ComputeTiling(rowNum, colNum, coreNum, tiling);

    const uint32_t coreRowNum = (rowNum + coreNum - 1) / coreNum;
    const uint32_t fullBlocks = (coreRowNum == 0) ? 0 : (rowNum / coreRowNum);
    const bool     hasTail    = (coreRowNum == 0) ? false : ((rowNum % coreRowNum) != 0);
    const uint32_t usedIdx    = hasTail ? fullBlocks : (fullBlocks == 0 ? 0u : fullBlocks - 1);
    tiling.set_usedBlockDim(usedIdx);
    
    // Copy tiling to tilingBuffer
    tiling.SaveToBuffer(tilingBuffer, tilingSize);
}

extern "C" size_t GetSoftmaxTilingBufBytes()
{
    optiling::SoftmaxCustomTilingData t;
    return t.GetDataSize();
}
