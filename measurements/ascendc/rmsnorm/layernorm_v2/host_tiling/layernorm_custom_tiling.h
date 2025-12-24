/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_NORMALIZATION_LAYERNORM_CUSTOM_TILING_H
#define EXAMPLES_NORMALIZATION_LAYERNORM_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LayernormCustomTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(LayerNormSeparateTiling, layernormTilingData);
  TILING_DATA_FIELD_DEF(uint32_t, aLength);
  TILING_DATA_FIELD_DEF(uint32_t, rLength);
  TILING_DATA_FIELD_DEF(uint32_t, rLengthWithPadding);
  TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LayernormCustom, LayernormCustomTilingData)
}


void ComputeTiling(const uint32_t aLength, const uint32_t rLength, const uint32_t rLengthWithPadding,
    optiling::LayernormCustomTilingData& tiling)
{
    ge::Shape geShape({ aLength, rLength });

    uint32_t maxTmpSize = 0;
    uint32_t minTmpSize = 0;
    bool isReuseSource = false;
    AscendC::GetLayerNormMaxMinTmpSize(geShape, sizeof(float), isReuseSource, true, false, maxTmpSize, minTmpSize);

    uint32_t localWorkspaceSize = minTmpSize;
    // get layernorm Tiling
    AscendC::GetLayerNormNDTilingInfo(geShape, localWorkspaceSize, sizeof(float), isReuseSource, true, 
        tiling.layernormTilingData);
    tiling.set_aLength(aLength);
    tiling.set_rLength(rLength);
    tiling.set_rLengthWithPadding(rLengthWithPadding);
    tiling.set_epsilon(0.0001);
}
#endif // EXAMPLES_NORMALIZATION_LAYERNORM_CUSTOM_TILING_H