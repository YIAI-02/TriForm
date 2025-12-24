/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include "tiling/tiling_api.h"
#include "../host_tiling/layernorm_custom_tiling.h"
using namespace std;

uint8_t *GetTilingBuf(optiling::LayernormCustomTilingData *tilingData) {
  uint32_t tilingSize = sizeof(optiling::LayernormCustomTilingData);
  uint8_t *buf = (uint8_t *)malloc(tilingSize);
  tilingData->SaveToBuffer(buf, tilingSize);
  return buf;
}

uint8_t* GenerateTiling(uint32_t aLength, uint32_t rLength, uint32_t rLengthWithPadding)
{
    optiling::LayernormCustomTilingData tiling;
    ComputeTiling(aLength, rLength, rLengthWithPadding, tiling);
    return GetTilingBuf(&tiling);
}