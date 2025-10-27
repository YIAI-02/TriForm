/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_mmad_custom.h"
#else
#include "tikicpulib.h"
extern "C" void mmad_custom(uint8_t *a, uint8_t *b, uint8_t *bias, uint8_t *c);
#endif

int32_t main(int32_t argc, char *argv[])
{
    uint32_t M = 32;
    uint32_t N = 32;
    uint32_t K = 32;
    size_t aFileSize = M * K * sizeof(int16_t); // uint16_t represent half
    size_t bFileSize = K * N * sizeof(int16_t); // uint16_t represent half
    size_t biasFileSize = N * sizeof(int16_t); // uint16_t represent half
    size_t cFileSize = M * N * sizeof(float);
    uint32_t blockDim = 1;

#ifdef ASCENDC_CPU_DEBUG
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
    uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);
    uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSize);

    ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
    ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
    ReadFile("./input/bias_gm.bin", biasFileSize, bias, biasFileSize);

    ICPU_RUN_KF(mmad_custom, blockDim, a, b, bias, c);

    WriteFile("./output/output.bin", c, cFileSize);

    AscendC::GmFree((void *)a);
    AscendC::GmFree((void *)b);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)c);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *aHost;
    uint8_t *aDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, aHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bHost;
    uint8_t *bDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, bHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *biasHost;
    uint8_t *biasDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&biasHost), biasFileSize));
    CHECK_ACL(aclrtMalloc((void **)&biasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/bias_gm.bin", biasFileSize, biasHost, biasFileSize);
    CHECK_ACL(aclrtMemcpy(biasDevice, biasFileSize, biasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cHost;
    uint8_t *cDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&cHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&cDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(mmad_custom)(blockDim, stream, aDevice, bDevice, biasDevice, cDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(cHost, cFileSize, cDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cFileSize);

    CHECK_ACL(aclrtFree(aDevice));
    CHECK_ACL(aclrtFreeHost(aHost));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFree(biasDevice));
    CHECK_ACL(aclrtFreeHost(biasHost));
    CHECK_ACL(aclrtFree(cDevice));
    CHECK_ACL(aclrtFreeHost(cHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
