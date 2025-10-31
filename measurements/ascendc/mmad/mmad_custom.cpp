/**
 * @file mmad_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
// #ifdef CUSTOM_ASCEND310P
// #include "mmad_custom.h"
// #else
#include "mmad_custom_cube_only.h"
// #endif

extern "C" __global__ __aicore__ void mmad_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
    uint32_t m, uint32_t n, uint32_t k)
{
    KernelMmad op;
    op.SetShape((uint16_t)m, (uint16_t)n, (uint16_t)k);
    op.Init(a, b, bias, c);
    op.Process();
}
