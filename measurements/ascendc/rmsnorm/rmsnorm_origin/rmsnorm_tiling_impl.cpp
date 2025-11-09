/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lib/normalization/rmsnorm_tiling.h"

#include "impl/host_log.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(RmsNormTilingOpApi, RmsNormTiling);
}  // namespace optiling

namespace AscendC {
namespace {
constexpr uint32_t ONE_BLK_FLOAT_NUM = 8;
constexpr uint32_t FLOAT_SIZE_IN_BYTE = 4;
constexpr uint32_t HALF_SIZE_IN_BYTE = 2;
constexpr uint32_t ONE_BLK_SIZE = 32;
constexpr uint32_t BASIC_BLK_HLENGTH = 64;
constexpr uint32_t BASIC_BLK_BSLENGTH = 8;
constexpr uint32_t B_INDEX = 0;
constexpr uint32_t S_INDEX = 1;
constexpr uint32_t H_INDEX = 2;
constexpr uint32_t SHAPE_DIM = 3;
constexpr uint32_t MAX_REPEAT = 255;

/*!
 * \brief for given inputValue, return value that aligned to input typeSize
 *
 * \param [in] inputValue: in unit of element
 * \param [in] typeSize: data type size, value is sizeof(TYPE)
 * \return aligned value of input value (in unit of element)
 */
inline uint32_t AlignToBlock(const uint32_t inputValue, const uint32_t typeSize)
{
    ASCENDC_HOST_ASSERT(typeSize > 0, return 0, "typeSize must be greater than 0.");
    const uint32_t alignUnit = ONE_BLK_SIZE / typeSize;
    return (inputValue + alignUnit - 1) / alignUnit * alignUnit;
}

uint32_t GetRmsNormMaxTmpSize(const ge::Shape& srcShape, const uint32_t typeSize)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[B_INDEX]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);

    const uint32_t inputLength = bLength * sLength * hLength;
    const uint32_t bsLength = AlignToBlock(bLength * sLength, typeSize);

    uint32_t maxSize = inputLength;
    if (typeSize == HALF_SIZE_IN_BYTE) {
        // for half, three temp buffers are needed, one for tmp buffer(b*s*h),
        // one for casted src buffer(b*s*h), one for reduced buffer(b*s)
        constexpr uint32_t halfCoeff = 2;
        maxSize = maxSize * halfCoeff + bsLength;
    } else {
        // float32 case only need two temp local buffers:
        // one is tmp buffer whose size is (b*s*h) and b*s can equal to 1
        // one is reducedBuffer whose size is b*s
        maxSize = maxSize + bsLength;
    }

    return maxSize * FLOAT_SIZE_IN_BYTE;
}

// for RmsNorm, if use dst as tmp buffer then min-tmp-size is size of reduceHlength, which is bLenght*sLgngth
uint32_t GetRmsNormMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isBasicBlock = false)
{
    (void)isBasicBlock;
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);

    // bsLength should align to block size
    const uint32_t bsLength = ONE_BLK_FLOAT_NUM;
    uint32_t minSize = hLength;
    if (typeSize == HALF_SIZE_IN_BYTE) {
        // for half type, need three temp local buffers,
        // one for tmp buffer(b*s*h), one for casted src buffer(b*s*h), one for reduced buffer(b*s)
        constexpr uint32_t bufferCoeff = 2;
        minSize = minSize * bufferCoeff + bsLength;
    } else {
        // float32 case only need two temp local buffers:
        // one is tmp buffer whose size is (b*s*h) and b*s can equal to 1
        // one is reducedBuffer whose size is b*s
        minSize = minSize + bsLength;
    }
    return minSize * FLOAT_SIZE_IN_BYTE;
}

inline bool RmsNormCheckBasicBlockShape(const uint32_t bLength, const uint32_t sLength, const uint32_t hLength)
{
    constexpr uint32_t maxBasicBlockH = 2048;
    return hLength % BASIC_BLK_HLENGTH != 0 ||
           bLength * sLength % BASIC_BLK_BSLENGTH != 0 ||
           hLength >= maxBasicBlockH;
}

bool RmsNormCheckShape(const ge::Shape& srcShape, const ge::Shape& originSrcShape, const uint32_t typeSize,
    const bool isBasicBlock = false)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    if (shapeDims.size() != SHAPE_DIM) {
        return false;
    }
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[B_INDEX]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);

    std::vector<int64_t> originDims = originSrcShape.GetDims();
    if (originDims.size() != SHAPE_DIM) {
        return false;
    }
    const uint32_t oriBLength = static_cast<uint32_t>(originDims[B_INDEX]);
    const uint32_t oriSLength = static_cast<uint32_t>(originDims[S_INDEX]);
    const uint32_t oriHLength = static_cast<uint32_t>(originDims[H_INDEX]);

    // original shape and tensor shape shuold match
    if (bLength != oriBLength || sLength != oriSLength || hLength < oriHLength || oriHLength == 0U) {
        return false;
    }
    // hLength should align to block-size
    if (hLength * typeSize % ONE_BLK_SIZE != 0) {
        return false;
    }
    // for basic block hlength should be multiples of 64, and bslength should be multiples of 8
    if (isBasicBlock && RmsNormCheckBasicBlockShape(bLength, sLength, hLength)) {
        return false;
    }
    return true;
}
} // namespace

bool GetRmsNormMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, uint32_t& maxValue,
    uint32_t& minValue, const bool isBasicBlock)
{
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[B_INDEX]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);
    // shape for basic block: bsLength be multiples of 8, and hLength should be multiples of 64.
    if (isBasicBlock && RmsNormCheckBasicBlockShape(bLength, sLength, hLength)) {
        return false;
    }
    maxValue = GetRmsNormMaxTmpSize(srcShape, typeSize);
    minValue = GetRmsNormMinTmpSize(srcShape, typeSize, isBasicBlock);
    return true;
}

bool GetRmsNormTilingInfo(const ge::Shape& srcShape, const ge::Shape& originSrcShape,
    const uint32_t stackBufferByteSize, const uint32_t typeSize, optiling::RmsNormTiling& tiling,
    const bool isBasicBlock)
{
    if (!RmsNormCheckShape(srcShape, originSrcShape, typeSize, isBasicBlock)) {
        return false;
    }
    uint32_t minSize = GetRmsNormMinTmpSize(srcShape, typeSize, isBasicBlock);
    if (stackBufferByteSize < minSize) {
        return false;
    }
    std::vector<int64_t> shapeDims = srcShape.GetDims();
    const uint32_t bLength = static_cast<uint32_t>(shapeDims[B_INDEX]);
    const uint32_t sLength = static_cast<uint32_t>(shapeDims[S_INDEX]);
    const uint32_t hLength = static_cast<uint32_t>(shapeDims[H_INDEX]);
    tiling.set_bLength(bLength);
    tiling.set_sLength(sLength);
    tiling.set_hLength(hLength);
    // 3rd elem is original h-length
    std::vector<int64_t> oriShapeDims = originSrcShape.GetDims();
    const uint32_t originalHLength = static_cast<uint32_t>(oriShapeDims[H_INDEX]);
    tiling.set_originalHLength(originalHLength);
    tiling.set_reciprocalOfHLength(1.0 / static_cast<float>(originalHLength));

    uint32_t totalSize = stackBufferByteSize / FLOAT_SIZE_IN_BYTE;
    uint32_t bsLength = 1;
    // for half type, tmp buffer is doubled
    auto alignToBlock = [](const uint32_t inValue, const uint32_t alignUnit) {
        return (inValue + alignUnit - 1) / alignUnit * alignUnit;
    };
    const uint32_t coeff = (typeSize == sizeof(float) ? 1u: 2u);
    while (totalSize >= (bsLength + 1) * hLength * coeff  + alignToBlock(bsLength + 1,  ONE_BLK_FLOAT_NUM)) {
        bsLength++;
    }
    uint32_t oneTmpSize = bsLength * hLength;
    const uint32_t inputXSize = bLength * sLength * hLength;
    if (oneTmpSize > inputXSize) {
        oneTmpSize = inputXSize;
    }
    // bsLength should align to hlength
    bsLength = oneTmpSize / hLength;
    if (isBasicBlock) {
        // for basic block bsLength should be multiples of BASIC_BLK_BSLENGTH(8)
        bsLength = bsLength < BASIC_BLK_BSLENGTH ? bsLength : bsLength / BASIC_BLK_BSLENGTH * BASIC_BLK_BSLENGTH;
    } else if (bsLength > MAX_REPEAT) {
        bsLength = MAX_REPEAT;
    }
    oneTmpSize = bsLength * hLength;

    tiling.set_mainBshLength(oneTmpSize);
    tiling.set_mainBsLength(bsLength);
    // reduced bsLength should algin to 32B
    tiling.set_mainBsLengthAlign(AlignToBlock(bsLength, FLOAT_SIZE_IN_BYTE));
    tiling.set_loopRound(inputXSize / oneTmpSize);
    // calculate tail tiling info
    const uint32_t inputTailSize = inputXSize % oneTmpSize;
    tiling.set_tailBshLength(inputTailSize);
    tiling.set_inputTailPos(inputXSize - inputTailSize);
    tiling.set_tailBsLength(inputTailSize / hLength);
    return true;
}
}  // namespace AscendC