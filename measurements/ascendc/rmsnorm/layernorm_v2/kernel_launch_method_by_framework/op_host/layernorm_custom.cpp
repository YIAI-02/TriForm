/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "layernorm_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 48;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    LayernormCustomTilingData tiling;
    const gert::StorageShape *x1_shape = context->GetInputShape(0);
    const gert::Shape shape = x1_shape->GetStorageShape();
    // aLength , rLength, rLengthWithPadding
    ComputeTiling(shape[0], shape[1], shape[1], tiling);

    context->SetBlockDim(BLOCK_DIM);
    context->SetTilingKey(1);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class LayernormCustom : public OpDef {
public:
    explicit LayernormCustom(const char *name) : OpDef(name)
    {
        this->Input("inputXGm")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("gammaGm")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("betaGm")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("outputGm")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("outputMeanGm")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("outputRstdGm")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });

        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(LayernormCustom);
}
