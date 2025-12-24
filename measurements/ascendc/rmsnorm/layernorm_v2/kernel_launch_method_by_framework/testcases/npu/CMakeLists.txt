# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_npu_target(OP LayernormV2Custom SRC layernorm_custom_main.cpp)

add_custom_target(run_npu_test
		COMMAND echo "==============================================================================="
		COMMAND echo " Run NPU test at ${CMAKE_CURRENT_BINARY_DIR}"
		COMMAND echo "==============================================================================="
		COMMAND $<TARGET_FILE:layernorm_v2_custom_npu>
		COMMAND echo "==============================================================================="
		)
add_dependencies(run_npu_test layernorm_v2_custom_npu)