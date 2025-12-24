/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_layernorm_custom.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
#include <stdio.h>
#include <stdlib.h>
#include "../../../../../common/data_utils.h"

aclrtStream CreateStream(int32_t device)
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        printf("acl init failed\n");
        return nullptr;
    }
    if (aclrtSetDevice(device) != ACL_SUCCESS) {
        printf("Set device failed\n");
        (void)aclFinalize();
        return nullptr;
    }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        printf("Create stream failed\n");
        return nullptr;
    }
    return stream;
}

void DestroyStream(aclrtStream stream, int32_t device)
{
    (void)aclrtDestroyStream(stream);
    if (aclrtResetDevice(device) != ACL_SUCCESS) {
        printf("Reset device failed\n");
    }
    if (aclFinalize() != ACL_SUCCESS) {
        printf("Finalize acl failed\n");
    }
}

void DestroyTensor(aclTensor *tensors[], void *devMem[], int32_t tensorCount)
{
    for (auto i = 0; i < tensorCount; i++) {
        if (!tensors[i]) {
            continue;
        }
        if (devMem[i]) {
            CHECK_ACL(aclrtFree(devMem[i]));
        }
        aclDestroyTensor(tensors[i]);
    }
}

struct tensorInfo {
    int64_t *dims;
    int64_t dimCnt;
    aclDataType dtype;
    aclFormat fmt;
};

int64_t GetDataSize(struct tensorInfo *desc)
{
    if (!desc->dims) {
        return 0;
    }
    int64_t size = 1;
    for (auto i = 0; i < desc->dimCnt; i++) {
        size *= desc->dims[i];
    }
    return size * sizeof(float);
}

static bool CompareResult(const void *outputData, int64_t outSize, std::string goldenName)
{
    void *goldenData;
    CHECK_ACL(aclrtMallocHost((void **)(&goldenData), outSize));
    size_t goldenSize = outSize;
    bool ret = ReadFile("../output/golden_output_" + goldenName + ".bin", goldenSize, goldenData, goldenSize);
    if (ret) {
        printf("ReadFile golden_output_%s.bin success!\n", goldenName.c_str());
    } else {
        printf("test failed!\n");
        return false;
    }
    constexpr float EPS = 1e-6;
    int64_t wrongNum = 0;

    for (auto i = 0; i < outSize / sizeof(float); i++) {
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
    CHECK_ACL(aclrtFreeHost(goldenData));

    if (wrongNum != 0) {
        return false;
    } else {
        printf("CompareResult golden_output_%s.bin success\n", goldenName.c_str());
        return true;
    }
}

int32_t main(void)
{
    aclrtStream stream;
    int64_t inputX[] = {32, 32};
    int64_t inputGamma[] = {32};
    int64_t inputBeta[] = {32};
    int64_t outputResult[] = {32, 32};
    int64_t outputMean[] = {32};
    int64_t outputRstd[] = {32};
    struct tensorInfo tensorDesc[] = {{inputX, 2, ACL_FLOAT, ACL_FORMAT_ND},
                                      {inputGamma, 1, ACL_FLOAT, ACL_FORMAT_ND},
                                      {inputBeta, 1, ACL_FLOAT, ACL_FORMAT_ND},
                                      {outputResult, 2, ACL_FLOAT, ACL_FORMAT_ND},
                                      {outputMean, 1, ACL_FLOAT, ACL_FORMAT_ND},
                                      {outputRstd, 1, ACL_FLOAT, ACL_FORMAT_ND},
                                     };
    stream = CreateStream(0);
    int32_t tensorCount = sizeof(tensorDesc) / sizeof(struct tensorInfo);
    aclTensor *tensors[tensorCount];
    void *devMem[tensorCount];
    for (auto i = 0; i < tensorCount; i++) {
        void *data;
        struct tensorInfo *info = &(tensorDesc[i]);
        int64_t size = GetDataSize(info);
        if (size == 0) {
            tensors[i] = nullptr;
            devMem[i] = nullptr;
            continue;
        }
        CHECK_ACL(aclrtMalloc(&data, size, ACL_MEM_MALLOC_HUGE_FIRST));
        // read input
        if (i == 0) {
            size_t inputSize = size;
            void *dataHost;
            CHECK_ACL(aclrtMallocHost((void **)(&dataHost), inputSize));
            ReadFile("../input/input_X.bin", inputSize, dataHost, inputSize);
            CHECK_ACL(aclrtMemcpy(data, size, dataHost, size, ACL_MEMCPY_HOST_TO_DEVICE));
            CHECK_ACL(aclrtFreeHost(dataHost));
        }
        if (i == 1) {
            size_t inputSize = size;
            void *dataHost;
            CHECK_ACL(aclrtMallocHost((void **)(&dataHost), inputSize));
            ReadFile("../input/input_gamma.bin", inputSize, dataHost, inputSize);
            CHECK_ACL(aclrtMemcpy(data, size, dataHost, size, ACL_MEMCPY_HOST_TO_DEVICE));
            CHECK_ACL(aclrtFreeHost(dataHost));
        }
        if (i == 2) {
            size_t inputSize = size;
            void *dataHost;
            CHECK_ACL(aclrtMallocHost((void **)(&dataHost), inputSize));
            ReadFile("../input/input_beta.bin", inputSize, dataHost, inputSize);
            CHECK_ACL(aclrtMemcpy(data, size, dataHost, size, ACL_MEMCPY_HOST_TO_DEVICE));
            CHECK_ACL(aclrtFreeHost(dataHost));
        }
        devMem[i] = data;
        tensors[i] =
            aclCreateTensor(info->dims, info->dimCnt, info->dtype, nullptr, 0, info->fmt, info->dims, info->dimCnt, data);
    }

    size_t workspaceSize = 0;
    aclOpExecutor *handle;
    int32_t ret;
    ret = aclnnLayernormCustomGetWorkspaceSize(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5],
        &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        printf("aclnnLayernormCustomGetWorkspaceSize failed. error code is %u\n", ret);
        DestroyTensor(tensors, devMem, tensorCount);
        DestroyStream(stream, 0);
        return ret;
    }
    printf("aclnnLayernormCustomGetWorkspaceSize ret %u workspace size %lu\n", ret, workspaceSize);
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    ret = aclnnLayernormCustom(workspace, workspaceSize, handle, stream);
    printf("aclnnLayernormCustom ret %u\n", ret);
    if (aclrtSynchronizeStreamWithTimeout(stream, 5000) != ACL_SUCCESS) {
        printf("Synchronize stream failed\n");
    }

    uint8_t *resultHost, *meanHost, *rstdHost;
    int64_t resultHostSize = GetDataSize(&(tensorDesc[3]));
    int64_t meanHostSize = GetDataSize(&(tensorDesc[4]));
    int64_t rstdHostSize = GetDataSize(&(tensorDesc[5]));

    CHECK_ACL(aclrtMallocHost((void **)(&resultHost), resultHostSize));
    CHECK_ACL(aclrtMallocHost((void **)(&meanHost), meanHostSize));
    CHECK_ACL(aclrtMallocHost((void **)(&rstdHost), rstdHostSize));

    CHECK_ACL(aclrtMemcpy(resultHost, resultHostSize, devMem[3], resultHostSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(meanHost, meanHostSize, devMem[4], meanHostSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(rstdHost, rstdHostSize, devMem[5], rstdHostSize, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile("../output/output_result.bin", resultHost, resultHostSize);
    WriteFile("../output/output_mean.bin", meanHost, meanHostSize);
    WriteFile("../output/output_rstd.bin", rstdHost, rstdHostSize);

    bool goldenResult = true;
    goldenResult &= CompareResult(resultHost, resultHostSize, "result");
    goldenResult &= CompareResult(meanHost, meanHostSize, "mean");
    goldenResult &= CompareResult(rstdHost, rstdHostSize, "rstd");
    if (goldenResult) {
        printf("test pass!\n");
    } else {
        printf("test failed!\n");
    }

    CHECK_ACL(aclrtFreeHost(resultHost));
    CHECK_ACL(aclrtFreeHost(meanHost));
    CHECK_ACL(aclrtFreeHost(rstdHost));

    DestroyTensor(tensors, devMem, tensorCount);
    DestroyStream(stream, 0);
    return 0;
}
