/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// #include "/root/TriForm/measurements/ascendc/common/data_util.h"
// #ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// extern void faster_gelu_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* srcGm, uint8_t* dstGm,
//                                   uint8_t* workspace, uint8_t* tiling);
// #else
// #include "tikicpulib.h"
// extern "C" __global__ __aicore__ void faster_gelu_custom(GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR workspace,
//                                                          GM_ADDR tiling);
// #endif

// constexpr uint32_t BLOCK_DIM = 1;
// constexpr uint32_t DATALENGTH = 1024;
// constexpr uint32_t TILINGDATA_SIZE = 2;
// constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

// extern void GenerateTiling(const uint32_t dataLength, const uint32_t tilingSize, uint8_t* tilingBuffer);

// static bool CompareResult(const void* outputData, const int64_t outSize)
// {
//     void* goldenData;
// #ifdef ASCENDC_CPU_DEBUG
//     goldenData = (uint8_t*)AscendC::GmAlloc(outSize);
// #else
//     CHECK_ACL(aclrtMallocHost((void**)(&goldenData), outSize));
// #endif
//     size_t goldenSize = outSize;
//     bool ret = ReadFile("../output/golden.bin", goldenSize, goldenData, goldenSize);
//     if (ret) {
//         printf("ReadFile golden.bin success!\n");
//     } else {
// #ifdef ASCENDC_CPU_DEBUG
//         AscendC::GmFree((void*)goldenData);
// #else
//         CHECK_ACL(aclrtFreeHost(goldenData));
// #endif
//         return false;
//     }
//     constexpr float EPS = 1e-5;
//     int64_t wrongNum = 0;

//     for (int i = 0; i < outSize / sizeof(float); i++) {
//         float a = (reinterpret_cast<const float*>(outputData))[i];
//         float b = (reinterpret_cast<const float*>(goldenData))[i];
//         float ae = std::abs(a - b);
//         float re = ae / abs(b);
//         if (ae > EPS && re > EPS) {
//             printf("CompareResult golden.bin failed. Output[%d] is %lf, golden is %lf\n", i, a, b);
//             wrongNum++;
//         }
//     }
// #ifdef ASCENDC_CPU_DEBUG
//     AscendC::GmFree((void*)goldenData);
// #else
//     CHECK_ACL(aclrtFreeHost(goldenData));
// #endif
//     if (wrongNum != 0) {
//         return false;
//     } else {
//         printf("CompareResult golden.bin success!\n");
//         return true;
//     }
// }

// int32_t main(int32_t argc, char* argv[])
// {
//     uint32_t blockDim = BLOCK_DIM;
//     size_t dataLength = DATALENGTH;
//     size_t dataSize = DATALENGTH * sizeof(float);
//     size_t tilingFileSize = TILINGDATA_SIZE * sizeof(uint32_t);
//     size_t workspaceSize = WORKSPACE_SIZE;

// #ifdef ASCENDC_CPU_DEBUG
//     uint8_t* input = (uint8_t*)AscendC::GmAlloc(dataSize);
//     uint8_t* output = (uint8_t*)AscendC::GmAlloc(dataSize);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingFileSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);

//     ReadFile("../input/input.bin", dataSize, input, dataSize);

//     GenerateTiling(dataLength, tilingFileSize, tiling);

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);                                 // run in aiv mode
//     ICPU_RUN_KF(faster_gelu_custom, blockDim, input, output, workspace, tiling);  // use this macro for cpu debug

//     WriteFile("../output/output.bin", output, dataSize);
//     bool goldenResult = CompareResult(output, dataSize);
//     if (goldenResult) {
//         printf("test pass!\n");
//     } else {
//         printf("test failed!\n");
//     }

//     AscendC::GmFree((void*)input);
//     AscendC::GmFree((void*)output);
//     AscendC::GmFree((void*)tiling);
//     AscendC::GmFree((void*)workspace);
// #else
//     // Initialize resources
//     CHECK_ACL(aclInit(nullptr));
//     aclrtContext context;
//     int32_t deviceId = 0;
//     CHECK_ACL(aclrtSetDevice(deviceId));
//     CHECK_ACL(aclrtCreateContext(&context, deviceId));
//     aclrtStream stream = nullptr;
//     CHECK_ACL(aclrtCreateStream(&stream));

//     uint8_t *inputHost, *outputHost, *workspaceHost, *tilingHost;
//     uint8_t *inputDevice, *outputDevice, *workspaceDevice, *tilingDevice;

//     // Allocate host memory and device memory
//     CHECK_ACL(aclrtMallocHost((void**)(&inputHost), dataSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&outputHost), dataSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&workspaceHost), workspaceSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingFileSize));
//     CHECK_ACL(aclrtMalloc((void**)&inputDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&outputDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

//     // Read input data to host memory
//     ReadFile("../input/input.bin", dataSize, inputHost, dataSize);

//     GenerateTiling(dataLength, tilingFileSize, tilingHost);
//     // Copy host memory to device memory
//     CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
//     CHECK_ACL(aclrtMemcpy(inputDevice, dataSize, inputHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE));

//     // Execute the kernel
//     faster_gelu_custom_do(blockDim, nullptr, stream, inputDevice, outputDevice, workspaceDevice, tilingDevice);

//     // Wait for the stop event to complete
//     CHECK_ACL(aclrtSynchronizeStream(stream));

//     // Copy result to host memory and write to output file
//     CHECK_ACL(aclrtMemcpy(outputHost, dataSize, outputDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST));
//     WriteFile("../output/output.bin", outputHost, dataSize);

//     // Compare the result with the golden result
//     bool goldenResult = CompareResult(outputHost, dataSize);
//     if (goldenResult) {
//         printf("test pass!\n");
//     } else {
//         printf("test failed!\n");
//     }

//     // Clean up memory
//     CHECK_ACL(aclrtFree(inputDevice));
//     CHECK_ACL(aclrtFree(outputDevice));
//     CHECK_ACL(aclrtFree(workspaceDevice));
//     CHECK_ACL(aclrtFree(tilingDevice));
//     CHECK_ACL(aclrtFreeHost(inputHost));
//     CHECK_ACL(aclrtFreeHost(outputHost));
//     CHECK_ACL(aclrtFreeHost(workspaceHost));
//     CHECK_ACL(aclrtFreeHost(tilingHost));

//     CHECK_ACL(aclrtDestroyStream(stream));
//     CHECK_ACL(aclrtDestroyContext(context));
//     CHECK_ACL(aclrtResetDevice(deviceId));
//     CHECK_ACL(aclFinalize());
// #endif
//     return 0;
// }

// main.cpp —— 多维度扫描；无输入/真值对比；每个维度只执行一次
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>

#include "/root/TriForm/measurements/ascendc/common/data_util.h"  // 用于 CHECK_ACL 等
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void faster_gelu_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                                  uint8_t* srcGm, uint8_t* dstGm, uint8_t* workspace, uint8_t* tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void faster_gelu_custom(GM_ADDR srcGm, GM_ADDR dstGm, GM_ADDR workspace,
                                                         GM_ADDR tiling);
#endif

// 保持与工程一致
constexpr uint32_t BLOCK_DIM        = 1;
constexpr uint32_t TILINGDATA_SIZE  = 2;               // dataLength + sharedTmpBufferSize
constexpr uint32_t WORKSPACE_SIZE   = 16 * 1024 * 1024;

extern void GenerateTiling(const uint32_t dataLength, const uint32_t tilingSize, uint8_t* tilingBuffer);

// 读取一组维度（--sizes / 环境变量 SIZES / --len）
static std::vector<size_t> parse_sizes(int argc, char* argv[]) {
    std::vector<size_t> sizes;
    auto push_list = [&](const std::string& s) {
        std::stringstream ss(s); std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            std::stringstream ss2(tok); std::string t2;
            while (ss2 >> t2) sizes.push_back(static_cast<size_t>(std::stoul(t2)));
        }
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--sizes" || a == "-l") && i + 1 < argc) push_list(argv[++i]);
        else if ((a == "--len" || a == "-n") && i + 1 < argc) sizes.push_back(static_cast<size_t>(std::stoul(argv[++i])));
    }
    if (sizes.empty()) {
        const char* env = std::getenv("SIZES");
        if (env && *env) push_list(env);
    }
    if (sizes.empty()) sizes.push_back(1024); // 与示例一致的默认值
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

static void fill_random(uint8_t* buf, size_t bytes) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    float* p = reinterpret_cast<float*>(buf);
    for (size_t i = 0; i < bytes / sizeof(float); ++i) p[i] = dist(rng);
}

int32_t main(int32_t argc, char* argv[])
{
    const auto sizes         = parse_sizes(argc, argv);
    const uint32_t blockDim  = BLOCK_DIM;
    const size_t tilingBytes = TILINGDATA_SIZE * sizeof(uint32_t);
    const size_t wsBytes     = WORKSPACE_SIZE;

#ifdef ASCENDC_CPU_DEBUG
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    for (size_t N : sizes) {
        const size_t dataBytes = N * sizeof(float);
        printf("[INFO] Run faster_gelu_custom len=%zu\n", N);

        uint8_t* in  = (uint8_t*)AscendC::GmAlloc(dataBytes);
        uint8_t* out = (uint8_t*)AscendC::GmAlloc(dataBytes);
        uint8_t* tl  = (uint8_t*)AscendC::GmAlloc(tilingBytes);
        uint8_t* ws  = (uint8_t*)AscendC::GmAlloc(wsBytes);

        fill_random(in, dataBytes);
        GenerateTiling((uint32_t)N, (uint32_t)tilingBytes, tl);

        ICPU_RUN_KF(faster_gelu_custom, blockDim, in, out, ws, tl);

        AscendC::GmFree((void*)in);
        AscendC::GmFree((void*)out);
        AscendC::GmFree((void*)tl);
        AscendC::GmFree((void*)ws);
    }
#else
    // NPU / SIM
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtContext ctx;
    CHECK_ACL(aclrtCreateContext(&ctx, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    for (size_t N : sizes) {
        const size_t dataBytes = N * sizeof(float);
        printf("[INFO] Run faster_gelu_custom len=%zu\n", N);

        uint8_t *hIn, *hOut, *hWs, *hTl;
        uint8_t *dIn, *dOut, *dWs, *dTl;

        CHECK_ACL(aclrtMallocHost((void**)(&hIn),  dataBytes));
        CHECK_ACL(aclrtMallocHost((void**)(&hOut), dataBytes));
        CHECK_ACL(aclrtMallocHost((void**)(&hWs),  wsBytes));
        CHECK_ACL(aclrtMallocHost((void**)(&hTl),  tilingBytes));

        CHECK_ACL(aclrtMalloc((void**)&dIn, dataBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc((void**)&dOut, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc((void**)&dWs, wsBytes,    ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc((void**)&dTl, tilingBytes,ACL_MEM_MALLOC_HUGE_FIRST));

        fill_random(hIn, dataBytes);
        GenerateTiling((uint32_t)N, (uint32_t)tilingBytes, hTl);

        CHECK_ACL(aclrtMemcpy(dTl, tilingBytes, hTl, tilingBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dIn, dataBytes,   hIn, dataBytes,   ACL_MEMCPY_HOST_TO_DEVICE));

        faster_gelu_custom_do(blockDim, nullptr, stream, dIn, dOut, dWs, dTl);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        CHECK_ACL(aclrtFree(dIn));
        CHECK_ACL(aclrtFree(dOut));
        CHECK_ACL(aclrtFree(dWs));
        CHECK_ACL(aclrtFree(dTl));
        CHECK_ACL(aclrtFreeHost(hIn));
        CHECK_ACL(aclrtFreeHost(hOut));
        CHECK_ACL(aclrtFreeHost(hWs));
        CHECK_ACL(aclrtFreeHost(hTl));
    }

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(ctx));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
