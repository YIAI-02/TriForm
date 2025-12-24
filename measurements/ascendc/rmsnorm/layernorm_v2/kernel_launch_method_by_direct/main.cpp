/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../common/data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void layernorm_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *inputXGm, uint8_t *gammaGm,
    uint8_t *betaGm, uint8_t *outputGm, uint8_t *outputMeanGm, uint8_t *outputRstdGm, uint8_t *workspace,
    uint8_t *tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void layernorm_custom(GM_ADDR inputXGm, GM_ADDR gammaGm, GM_ADDR betaGm,
    GM_ADDR outputGm, GM_ADDR outputMeanGm, GM_ADDR outputRstdGm, GM_ADDR workspace, GM_ADDR tiling);
#endif

constexpr uint32_t ALENGTH = 32;
constexpr uint32_t RLENGTH = 32;
constexpr uint32_t RLENGTH_WITH_PAD = 32;
constexpr uint32_t BLOCK_DIM = 40;
constexpr uint32_t TILINGDATA_SIZE = 30;
constexpr uint32_t WORKSPACE_SIZE = 1024;

extern uint8_t *GenerateTiling(uint32_t aLength, uint32_t rLength, uint32_t rLengthWithPadding);

static bool CompareResult(const void *outputData, int64_t outSize, std::string goldenName)
{
    void *goldenData;
#ifdef ASCENDC_CPU_DEBUG
    goldenData = (uint8_t *)AscendC::GmAlloc(outSize);
#else
    CHECK_ACL(aclrtMallocHost((void **)(&goldenData), outSize));
#endif
    size_t goldenSize = outSize;
    bool ret = ReadFile("../output/golden_output_" + goldenName + ".bin", goldenSize, goldenData, goldenSize);
    if (ret) {
        printf("ReadFile golden_output_%s.bin success!\n", goldenName.c_str());
    } else {
        printf("test failed!\n");
        return false;
    }
    constexpr float EPS = 1e-5;
    int64_t wrongNum = 0;

    for (int i = 0; i < outSize / sizeof(float); i++) {
        float a = (reinterpret_cast<const float *>(outputData))[i];
        float b = (reinterpret_cast<const float *>(goldenData))[i];
        float ae = std::abs(a - b);
        float re = ae / abs(b);
        if (ae > EPS && re > EPS) {
            printf("CompareResult golden_output_%s.bin failed output is %lf, golden is %lf\n", goldenName.c_str(), a,
                b);
            wrongNum++;
        }
    }
#ifdef ASCENDC_CPU_DEBUG
    AscendC::GmFree((void *)goldenData);
#else
    CHECK_ACL(aclrtFreeHost(goldenData));
#endif
    if (wrongNum != 0) {
        return false;
    } else {
        printf("CompareResult golden_output_%s.bin success!\n", goldenName.c_str());
        return true;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = BLOCK_DIM;
    size_t workspaceSize = WORKSPACE_SIZE * sizeof(float);
    size_t xSize = ALENGTH * RLENGTH_WITH_PAD * sizeof(float);
    size_t gammaSize = RLENGTH_WITH_PAD * sizeof(float);
    size_t betaSize = RLENGTH_WITH_PAD * sizeof(float);
    size_t outputSize = ALENGTH * RLENGTH_WITH_PAD * sizeof(float);
    size_t meanSize = ALENGTH * sizeof(float);
    size_t rstdSize = ALENGTH * sizeof(float);
    size_t tilingFileSize = TILINGDATA_SIZE * sizeof(uint32_t);
    bool goldenResult = true;
    uint8_t *tilingBuf = GenerateTiling(ALENGTH, RLENGTH, RLENGTH_WITH_PAD);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inputX = (uint8_t *)AscendC::GmAlloc(xSize);
    uint8_t *gamma = (uint8_t *)AscendC::GmAlloc(gammaSize);
    uint8_t *beta = (uint8_t *)AscendC::GmAlloc(betaSize);
    uint8_t *result = (uint8_t *)AscendC::GmAlloc(outputSize);
    uint8_t *mean = (uint8_t *)AscendC::GmAlloc(meanSize);
    uint8_t *rstd = (uint8_t *)AscendC::GmAlloc(rstdSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);

    ReadFile("../input/input_inputX.bin", xSize, inputX, xSize);
    ReadFile("../input/input_gamma.bin", gammaSize, gamma, gammaSize);
    ReadFile("../input/input_beta.bin", betaSize, beta, betaSize);
    memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // use this macro for cpu debug
    ICPU_RUN_KF(layernorm_custom, blockDim, inputX, gamma, beta, result, mean, rstd, workspace, tiling);
    WriteFile("../output/output_result.bin", result, outputSize);
    WriteFile("../output/output_mean.bin", mean, meanSize);
    WriteFile("../output/output_rstd.bin", rstd, rstdSize);

    goldenResult &= CompareResult(result, outputSize, "result");
    goldenResult &= CompareResult(mean, meanSize, "mean");
    goldenResult &= CompareResult(rstd, rstdSize, "rstd");

    AscendC::GmFree((void *)inputX);
    AscendC::GmFree((void *)gamma);
    AscendC::GmFree((void *)beta);
    AscendC::GmFree((void *)result);
    AscendC::GmFree((void *)mean);
    AscendC::GmFree((void *)rstd);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputXHost, *gammaHost, *betaHost, *resultHost, *meanHost, *rstdHost, *workspaceHost, *tilingHost;
    uint8_t *inputXDevice, *gammaDevice, *betaDevice, *resultDevice, *meanDevice, *rstdDevice, *workspaceDevice,
        *tilingDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&inputXHost), xSize));
    CHECK_ACL(aclrtMallocHost((void **)(&gammaHost), gammaSize));
    CHECK_ACL(aclrtMallocHost((void **)(&betaHost), betaSize));
    CHECK_ACL(aclrtMallocHost((void **)(&resultHost), outputSize));
    CHECK_ACL(aclrtMallocHost((void **)(&meanHost), meanSize));
    CHECK_ACL(aclrtMallocHost((void **)(&rstdHost), rstdSize));
    CHECK_ACL(aclrtMallocHost((void **)(&workspaceHost), workspaceSize));
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputXDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&gammaDevice, gammaSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&betaDevice, betaSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&resultDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&meanDevice, meanSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&rstdDevice, rstdSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("../input/input_inputX.bin", xSize, inputXHost, xSize);
    ReadFile("../input/input_gamma.bin", gammaSize, gammaHost, gammaSize);
    ReadFile("../input/input_beta.bin", betaSize, betaHost, betaSize);

    CHECK_ACL(aclrtMemcpy(workspaceDevice, workspaceSize, workspaceHost, workspaceSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingBuf,
        tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMemcpy(inputXDevice, xSize, inputXHost, xSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(gammaDevice, gammaSize, gammaHost, gammaSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(betaDevice, betaSize, betaHost, betaSize, ACL_MEMCPY_HOST_TO_DEVICE));

    layernorm_custom_do(blockDim, nullptr, stream, inputXDevice, gammaDevice, betaDevice, resultDevice, meanDevice,
        rstdDevice, workspaceDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(resultHost, outputSize, resultDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(meanHost, meanSize, meanDevice, meanSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(
        aclrtMemcpy(rstdHost, rstdSize, rstdDevice, rstdSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("../output/output_result.bin", resultHost, outputSize);
    WriteFile("../output/output_mean.bin", meanHost, meanSize);
    WriteFile("../output/output_rstd.bin", rstdHost, rstdSize);

    goldenResult &= CompareResult(resultHost, outputSize, "result");
    goldenResult &= CompareResult(meanHost, meanSize, "mean");
    goldenResult &= CompareResult(rstdHost, rstdSize, "rstd");

    CHECK_ACL(aclrtFree(inputXDevice));
    CHECK_ACL(aclrtFree(gammaDevice));
    CHECK_ACL(aclrtFree(betaDevice));
    CHECK_ACL(aclrtFree(resultDevice));
    CHECK_ACL(aclrtFree(meanDevice));
    CHECK_ACL(aclrtFree(rstdDevice));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(inputXHost));
    CHECK_ACL(aclrtFreeHost(gammaHost));
    CHECK_ACL(aclrtFreeHost(betaHost));
    CHECK_ACL(aclrtFreeHost(resultHost));
    CHECK_ACL(aclrtFreeHost(meanHost));
    CHECK_ACL(aclrtFreeHost(rstdHost));
    CHECK_ACL(aclrtFreeHost(workspaceHost));
    CHECK_ACL(aclrtFreeHost(tilingHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    if (goldenResult) {
        printf("test pass!\n");
    } else {
        printf("test failed!\n");
    }
    return 0;
}
