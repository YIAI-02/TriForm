/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_ACTIVATION_SOFTMAX_CUSTOM_TILING_H
#define EXAMPLES_ACTIVATION_SOFTMAX_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, columnLength);
  TILING_DATA_FIELD_DEF(uint32_t, rowLength);
  TILING_DATA_FIELD_DEF(uint32_t, sharedTmpBufferSize);
  TILING_DATA_FIELD_DEF(uint32_t, usedBlockDim);
  TILING_DATA_FIELD_DEF(uint32_t, coreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, singleLoopCoreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreLoopCount);
  TILING_DATA_FIELD_DEF(uint32_t, singleCoreLoopTail);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleLoopCoreRowNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleCoreLoopCount);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreSingleCoreLoopTail);
  TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)
}
namespace SoftmaxCustomTiling {
constexpr uint32_t SHARED_TMP_BUFFER_SIZE = 61440; // reserved tmpbuffer 60K for softmax compute
struct SingleCoreLoopParam {
    uint32_t singleLoopCoreRowNum{ 0 };  // row num processed in single loop
    uint32_t singleCoreLoopCount{ 0 };   // loop count in single loop
    uint32_t singleCoreLoopTail{ 0 };    // row num of last loop in single core
};

const std::vector<std::pair<uint32_t, uint32_t>> SLICE_TABLE = {
    // {reduce axis length, slice factor}
    { 8192, 1 }, { 4096, 2 }, { 2048, 4 }, { 1024, 8 }, { 512, 16 }, { 256, 32 }, { 0, 64 }
};

SingleCoreLoopParam GetSingleCoreLoopParam(const uint32_t colNum, const uint32_t coreRowNum)
{
    //  Determine the params of single core based on the reduce axis length
    for (auto param : SLICE_TABLE) {
        if (colNum >= param.first) {
            SingleCoreLoopParam singleCoreLoopParam;
            singleCoreLoopParam.singleLoopCoreRowNum = param.second;
            singleCoreLoopParam.singleCoreLoopCount = coreRowNum / param.second;
            singleCoreLoopParam.singleCoreLoopTail = coreRowNum % param.second;
            return singleCoreLoopParam;
        }
    }
}

void ComputeTiling(const uint32_t rowNum, const uint32_t colNum, const uint32_t coreNum,
                   optiling::SoftmaxCustomTilingData& tiling)
{
    uint32_t localworkspaceSize = SHARED_TMP_BUFFER_SIZE;
    auto alignedRowNum = (rowNum + coreNum - 1) / coreNum * coreNum;
    auto coreRowNum = alignedRowNum / coreNum;  // each core equal distribution
    auto tailCoreRowNum = rowNum % coreRowNum;  // last core process the tail rownum
    auto usedBlockDim = rowNum / coreRowNum;    // the core num used actually

    SingleCoreLoopParam mainCoreLoopParam = GetSingleCoreLoopParam(colNum, coreRowNum);
    SingleCoreLoopParam tailCoreLoopParam;
    if (usedBlockDim == coreNum && tailCoreRowNum == 0) {
        tailCoreLoopParam = GetSingleCoreLoopParam(colNum, coreRowNum);
    } else {
        tailCoreLoopParam = GetSingleCoreLoopParam(colNum, tailCoreRowNum);
    }

    tiling.set_columnLength(colNum);
    tiling.set_rowLength(rowNum);
    tiling.set_sharedTmpBufferSize(localworkspaceSize);
    tiling.set_usedBlockDim(usedBlockDim);
    tiling.set_coreRowNum(coreRowNum);
    tiling.set_tailCoreRowNum(tailCoreRowNum);

    tiling.set_singleLoopCoreRowNum(mainCoreLoopParam.singleLoopCoreRowNum);
    tiling.set_singleCoreLoopCount(mainCoreLoopParam.singleCoreLoopCount);
    tiling.set_singleCoreLoopTail(mainCoreLoopParam.singleCoreLoopTail);
    tiling.set_tailCoreSingleLoopCoreRowNum(tailCoreLoopParam.singleLoopCoreRowNum);
    tiling.set_tailCoreSingleCoreLoopCount(tailCoreLoopParam.singleCoreLoopCount);
    tiling.set_tailCoreSingleCoreLoopTail(tailCoreLoopParam.singleCoreLoopTail);

    // get SoftMax Tiling
    ge::Shape softmaxComputeShape({ mainCoreLoopParam.singleLoopCoreRowNum, colNum });
    AscendC::SoftMaxTilingFunc(softmaxComputeShape, sizeof(float), localworkspaceSize, tiling.softmaxTilingData);
}
}  // namespace SoftmaxCustomTiling
#endif // EXAMPLES_ACTIVATION_SOFTMAX_CUSTOM_TILING_H