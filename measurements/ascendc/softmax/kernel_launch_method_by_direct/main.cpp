/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// #include "../../../common/data_utils.h"
// #ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// extern void softmax_custom_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* max, uint8_t* sum,
//     uint8_t* z, uint8_t* workspace, uint8_t* tiling);
// #else
// #include "tikicpulib.h"
// extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR max, GM_ADDR sum, GM_ADDR z, GM_ADDR workspace,
//     GM_ADDR tiling);
// #endif

// constexpr uint32_t ROW_NUM = 960;
// constexpr uint32_t COLUMN_NUM = 960;
// constexpr uint32_t USED_CORE_NUM = 40;
// constexpr uint32_t WORKSPACE_SIZE = 1024;
// constexpr uint32_t TILINGDATA_SIZE = 28;  // Element count of struct SoftmaxCustomTilingData
// constexpr uint32_t FLOAT_NUM_PER_BLOCK = 8;

// extern void GenerateTiling(const uint32_t m, const uint32_t k, const uint32_t coreNum, const uint32_t tilingSize,
//                            uint8_t* tilingData);

// static int64_t CompareResult(void* outputData, const int64_t outSize)
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
//         printf("ReadFile golden success!\n");
//     } else {
// #ifdef ASCENDC_CPU_DEBUG
//         AscendC::GmFree((void*)goldenData);
// #else
//         CHECK_ACL(aclrtFreeHost(goldenData));
// #endif
//         return -1;
//     }
//     constexpr float EPS = 1e-5;
//     int64_t wrongNum = 0;

//     for (int i = 0; i < outSize / sizeof(float); i++) {
//         float a = ((float*)outputData)[i];
//         float b = ((float*)goldenData)[i];
//         float ae = std::abs(a - b);
//         float re = ae / abs(b);
//         if (ae > EPS && re > EPS) {
//             printf("CompareResult failed output is %lf, golden is %lf\n", a, b);
//             wrongNum++;
//         }
//     }
// #ifdef ASCENDC_CPU_DEBUG
//     AscendC::GmFree((void*)goldenData);
// #else
//     CHECK_ACL(aclrtFreeHost(goldenData));
// #endif
//     return wrongNum;
// }

// int32_t main(int32_t argc, char* argv[])
// {
//     size_t inputSize = ROW_NUM * ROW_NUM * sizeof(float);
//     size_t workspaceSize = WORKSPACE_SIZE * sizeof(float);
//     size_t tilingSize = TILINGDATA_SIZE * sizeof(uint32_t);
//     size_t outputSize = ROW_NUM * ROW_NUM * sizeof(float);
//     size_t outputMaxSize = ROW_NUM * FLOAT_NUM_PER_BLOCK * sizeof(float);
//     size_t outputSumSize = ROW_NUM * FLOAT_NUM_PER_BLOCK * sizeof(float);
//     int64_t wrongNum = -1;
// #ifdef ASCENDC_CPU_DEBUG
//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
//     uint8_t* max = (uint8_t*)AscendC::GmAlloc(outputMaxSize);
//     uint8_t* sum = (uint8_t*)AscendC::GmAlloc(outputSumSize);
//     uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

//     ReadFile("../input/input_x.bin", inputSize, x, inputSize);
//     ReadFile("../input/workspace.bin", workspaceSize, workspace, workspaceSize);

//     GenerateTiling(ROW_NUM, COLUMN_NUM, USED_CORE_NUM, tilingSize, tiling);

//     AscendC::SetKernelMode(KernelMode::AIV_MODE);  // run in aiv mode
//     ICPU_RUN_KF(softmax_custom, USED_CORE_NUM, x, max, sum, z, workspace, tiling); // use this macro for cpu debug

//     WriteFile("../output/output_z.bin", z, outputSize);
//     WriteFile("../output/output_max.bin", max, outputMaxSize);
//     WriteFile("../output/output_sum.bin", sum, outputSumSize);

//     wrongNum = CompareResult(z, outputSize);

//     AscendC::GmFree((void*)x);
//     AscendC::GmFree((void*)max);
//     AscendC::GmFree((void*)sum);
//     AscendC::GmFree((void*)z);
//     AscendC::GmFree((void*)workspace);
//     AscendC::GmFree((void*)tiling);
// #else
//     // Initialize resources
//     CHECK_ACL(aclInit(nullptr));
//     aclrtContext context;
//     int32_t deviceId = 0;
//     CHECK_ACL(aclrtSetDevice(deviceId));
//     CHECK_ACL(aclrtCreateContext(&context, deviceId));
//     aclrtStream stream = nullptr;
//     CHECK_ACL(aclrtCreateStream(&stream));

//     uint8_t *xHost, *zHost, *maxHost, *sumHost, *workspaceHost, *tilingHost;
//     uint8_t *xDevice, *zDevice, *maxDevice, *sumDevice, *workspaceDevice, *tilingDevice;

//     // Allocate host memory and device memory
//     CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&maxHost), outputMaxSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&sumHost), outputSumSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&workspaceHost), workspaceSize));
//     CHECK_ACL(aclrtMallocHost((void**)(&tilingHost), tilingSize));
//     CHECK_ACL(aclrtMalloc((void**)&xDevice, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&maxDevice, outputMaxSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&sumDevice, outputSumSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&zDevice, outputSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     CHECK_ACL(aclrtMalloc((void**)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

//     ReadFile("../input/input_x.bin", inputSize, xHost, inputSize);
//     ReadFile("../input/workspace.bin", workspaceSize, workspaceHost, workspaceSize);

//     GenerateTiling(ROW_NUM, COLUMN_NUM, USED_CORE_NUM, tilingSize, tilingHost);

//     // Copy host memory to device memory
//     CHECK_ACL(aclrtMemcpy(xDevice, inputSize, xHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
//     CHECK_ACL(aclrtMemcpy(workspaceDevice, workspaceSize, workspaceHost, workspaceSize, ACL_MEMCPY_HOST_TO_DEVICE));
//     CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

//     // Execute the kernel
//     softmax_custom_do(USED_CORE_NUM, nullptr, stream, xDevice, maxDevice, sumDevice, zDevice, workspaceDevice,
//                       tilingDevice);

//     // Wait for the stop event to complete
//     CHECK_ACL(aclrtSynchronizeStream(stream));

//     // Copy result to host memory and write to output file
//     CHECK_ACL(aclrtMemcpy(zHost, outputSize, zDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
//     CHECK_ACL(aclrtMemcpy(maxHost, outputMaxSize, maxDevice, outputMaxSize, ACL_MEMCPY_DEVICE_TO_HOST));
//     CHECK_ACL(aclrtMemcpy(sumHost, outputSumSize, sumDevice, outputSumSize, ACL_MEMCPY_DEVICE_TO_HOST));
//     WriteFile("../output/output_z.bin", zHost, outputSize);
//     WriteFile("../output/output_max.bin", maxHost, outputMaxSize);
//     WriteFile("../output/output_sum.bin", sumHost, outputSumSize);

//     // Compare the result with the golden result
//     wrongNum = CompareResult(zHost, outputSize);

//     // Clean up memory
//     CHECK_ACL(aclrtFree(xDevice));
//     CHECK_ACL(aclrtFree(zDevice));
//     CHECK_ACL(aclrtFree(maxDevice));
//     CHECK_ACL(aclrtFree(sumDevice));
//     CHECK_ACL(aclrtFree(workspaceDevice));
//     CHECK_ACL(aclrtFree(tilingDevice));
//     CHECK_ACL(aclrtFreeHost(xHost));
//     CHECK_ACL(aclrtFreeHost(zHost));
//     CHECK_ACL(aclrtFreeHost(maxHost));
//     CHECK_ACL(aclrtFreeHost(sumHost));
//     CHECK_ACL(aclrtFreeHost(workspaceHost));
//     CHECK_ACL(aclrtFreeHost(tilingHost));

//     CHECK_ACL(aclrtDestroyStream(stream));
//     CHECK_ACL(aclrtDestroyContext(context));
//     CHECK_ACL(aclrtResetDevice(deviceId));
//     CHECK_ACL(aclFinalize());
// #endif
//     if (wrongNum != 0) {
//         printf("test failed!\n");
//     } else {
//         printf("test pass!\n");
//     }
//     return 0;
// }

// main.cpp  —— sweep 版本：在同一进程内循环测试多组 (row, col)
#include "data_utils.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void softmax_custom_do(uint32_t coreDim, void *l2ctrl, void *stream,
                              uint8_t *x, uint8_t *max, uint8_t *sum,
                              uint8_t *z, uint8_t *workspace, uint8_t *tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR max, GM_ADDR sum,
                                                     GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);
#endif
extern "C" size_t GetSoftmaxTilingBufBytes();
// === 与原工程中保持一致的常量 ===
constexpr uint32_t USED_CORE_NUM = 24;    // 910B1
constexpr uint32_t WORKSPACE_SIZE = 1024; // float 元素个数（workspace 只是占位）
constexpr uint32_t TILINGDATA_SIZE = 28;  // SoftmaxCustomTilingData 的 u32 数量
constexpr uint32_t FLOATS_PER_BLOCK8 = 8; // 每行 max/sum 写回长度系数

extern void GenerateTiling(const uint32_t m, const uint32_t k,
                           const uint32_t coreNum, const uint32_t tilingSize,
                           uint8_t *tilingData); // 来自 softmax_custom_tiling.cpp

// ========== 简单的工具函数 ==========
static std::vector<std::pair<uint32_t, uint32_t>> ParseCasesFromEnv()
{
    std::vector<std::pair<uint32_t, uint32_t>> cases;
    const char *one = std::getenv("SOFTMAX_CASE");   // 单个：如 "1024x1024" 获取指针
    const char *many = std::getenv("SOFTMAX_CASES"); // 多个：如 "128x128,256x512,1024x2048"
    // 解析字符串s，得到m,k的lambda 函数
    auto parse_one = [](const std::string &s, uint32_t &m, uint32_t &k) -> bool
    {
        auto pos = s.find_first_of("xX*"); // 查找分隔符，三个中任意一个
        if (pos == std::string::npos)
            return false;
        try
        {
            m = static_cast<uint32_t>(std::stoul(s.substr(0, pos))); // 分隔符分出两部分，分别转成int32，赋值
            k = static_cast<uint32_t>(std::stoul(s.substr(pos + 1)));
            return (m > 0 && k > 0);
        }
        catch (...)
        {
            return false;
        }
    };

    if (one && *one)
    {
        uint32_t m, k;
        if (parse_one(one, m, k))
            cases.emplace_back(m, k); // 用上面定义的函数和vector自带的cases 将mk 添加进来
    }
    if (many && *many)
    {
        std::string s(many);
        size_t start = 0;
        while (start < s.size())
        {
            size_t comma = s.find(',', start);
            std::string token = s.substr(start, (comma == std::string::npos ? s.size() : comma) - start);
            uint32_t m, k;
            if (!token.empty() && parse_one(token, m, k))
                cases.emplace_back(m, k);
            if (comma == std::string::npos)
                break;
            start = comma + 1;
        }
    }
    return cases;
}

// 从环境变量中读取一个unit32
static uint32_t GetEnvU32(const char *name, uint32_t defv)
{
    const char *p = std::getenv(name);
    if (!p || !*p)
        return defv;
    try
    {
        auto v = static_cast<uint64_t>(std::stoull(p));
        return (v > 0 && v < (1ull << 32)) ? static_cast<uint32_t>(v) : defv;
    }
    catch (...)
    {
        return defv;
    }
}

// 默认尺寸表：没有指定softmaxcase的时候
static std::vector<std::pair<uint32_t, uint32_t>> DefaultCases()
{
    return {
        {16, 128}, {32, 128}, {64, 128}, {32, 256}, {64, 256}, {128, 256}, {64, 384}, {128, 384}, {64, 512}, {128, 512}, {256, 512}, {128, 768}, {256, 768}, {128, 1024}, {256, 1024}, {512, 1024}, {256, 1536}, {512, 1536}, {256, 2048}, {512, 2048}, {1024, 2048}, {512, 3072}, {1024, 3072}, {512, 4096}, {1024, 4096}};
}

static void FillRandomFloats(float *dst, size_t n)
{
    // 固定种子便于复现
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i)
        dst[i] = dist(rng);
}

// ========== 单个 Case 的执行（CPU/NPU 分别实现） ==========
#ifdef ASCENDC_CPU_DEBUG
static void RunOneCaseCPU(uint32_t M, uint32_t K, uint32_t iters, uint32_t warmup)
{
    size_t inBytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t outBytes = inBytes;
    // size_t msBytes = static_cast<size_t>(M) * FLOATS_PER_BLOCK8 * sizeof(float);
    uint32_t coreNum    = USED_CORE_NUM;
    uint32_t coreRowNum = (M + coreNum - 1) / coreNum;                 // ceil(M / coreNum)
    uint32_t msFloatTot = ((M + coreRowNum - 1) / coreRowNum)          // ceil(M / coreRowNum)
                        * coreRowNum * FLOATS_PER_BLOCK8;            // (= round_up(M, coreRowNum) * 8)
    size_t   msBytes    = (size_t)msFloatTot * sizeof(float);
    size_t wkBytes = static_cast<size_t>(WORKSPACE_SIZE) * sizeof(float);
    // size_t tilBytes = static_cast<size_t>(TILINGDATA_SIZE) * sizeof(uint32_t);
    size_t tilBytes = GetSoftmaxTilingBufBytes();
    std::cout << "[CPU] case M=" << M << " K=" << K
              << " warmup=" << warmup << " iters=" << iters << std::endl;

    for (uint32_t phase = 0; phase < warmup + iters; ++phase)
    {
        uint8_t *x = (uint8_t *)AscendC::GmAlloc(inBytes);
        uint8_t *max = (uint8_t *)AscendC::GmAlloc(msBytes);
        uint8_t *sum = (uint8_t *)AscendC::GmAlloc(msBytes);
        uint8_t *z = (uint8_t *)AscendC::GmAlloc(outBytes);
        uint8_t *wk = (uint8_t *)AscendC::GmAlloc(wkBytes);
        uint8_t *til = (uint8_t *)AscendC::GmAlloc(tilBytes);

        FillRandomFloats(reinterpret_cast<float *>(x), (size_t)M * K);
        std::memset(wk, 0, wkBytes);
        GenerateTiling(M, K, USED_CORE_NUM, tilBytes, til);

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(softmax_custom, USED_CORE_NUM, x, max, sum, z, wk, til);

        AscendC::GmFree(x);
        AscendC::GmFree(max);
        AscendC::GmFree(sum);
        AscendC::GmFree(z);
        AscendC::GmFree(wk);
        AscendC::GmFree(til);
    }
}
#else
static void RunOneCaseNPU(uint32_t M, uint32_t K, uint32_t iters, uint32_t warmup,
                          aclrtStream stream)
{
    size_t inBytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t outBytes = inBytes;
    // size_t msBytes = static_cast<size_t>(M) * FLOATS_PER_BLOCK8 * sizeof(float);
    // 假设 coreNum = USED_CORE_NUM（可用环境变量覆盖）
    uint32_t coreNum    = USED_CORE_NUM;
    uint32_t coreRowNum = (M + coreNum - 1) / coreNum;                 // ceil(M / coreNum)
    uint32_t totalBlocks = (coreRowNum == 0)? 1 : (M + coreNum - 1) / coreNum;
    if (totalBlocks == 0) totalBlocks = 1;
    uint32_t msFloatTot  = totalBlocks * coreRowNum * FLOATS_PER_BLOCK8;
    size_t   msBytes     = (size_t)msFloatTot * sizeof(float);
    size_t wkBytes = static_cast<size_t>(WORKSPACE_SIZE) * sizeof(float);
    // size_t tilBytes = static_cast<size_t>(TILINGDATA_SIZE) * sizeof(uint32_t);
    size_t tilBytes = GetSoftmaxTilingBufBytes();
    
    std::cout << "[NPU] case M=" << M << " K=" << K
              << " warmup=" << warmup << " iters=" << iters << std::endl;

    // host 缓冲区（每次复用即可，避免反复 malloc/free）名称后面H 代表host侧
    uint8_t *xH, *zH, *maxH, *sumH, *wkH, *tilH;
    CHECK_ACL(aclrtMallocHost((void **)(&xH), inBytes));
    CHECK_ACL(aclrtMallocHost((void **)(&zH), outBytes));
    CHECK_ACL(aclrtMallocHost((void **)(&maxH), msBytes));
    CHECK_ACL(aclrtMallocHost((void **)(&sumH), msBytes));
    CHECK_ACL(aclrtMallocHost((void **)(&wkH), wkBytes));
    CHECK_ACL(aclrtMallocHost((void **)(&tilH), tilBytes));

    // device 缓冲区，名称后面D 代表device侧
    uint8_t *xD, *zD, *maxD, *sumD, *wkD, *tilD;
    CHECK_ACL(aclrtMalloc((void **)&xD, inBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zD, outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&maxD, msBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&sumD, msBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&wkD, wkBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilD, tilBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    FillRandomFloats(reinterpret_cast<float *>(xH), (size_t)M * K); // 用随机数模拟输入，xH 是输入的缓冲区
    std::memset(wkH, 0, wkBytes);                                   // 给workspace 分配空间
    GenerateTiling(M, K, USED_CORE_NUM, tilBytes, tilH);

    // 预传 tiling 不变部分
    CHECK_ACL(aclrtMemcpy(tilD, tilBytes, tilH, tilBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_once = [&]()
    {
        // 拷贝输入数据xH 和 workspace 到device 侧
        CHECK_ACL(aclrtMemcpy(xD, inBytes, xH, inBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(wkD, wkBytes, wkH, wkBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        // 调用 kernel函数
        softmax_custom_do(totalBlocks, nullptr, stream, xD, maxD, sumD, zD, wkD, tilD);

        // 等待NPU计算完成，保证写回
        CHECK_ACL(aclrtSynchronizeStream(stream));

        // 结果拷回host
        CHECK_ACL(aclrtMemcpy(zH, outBytes, zD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    };

    // warmup + iters
    for (uint32_t i = 0; i < warmup + iters; ++i)
        run_once();

    // 释放
    CHECK_ACL(aclrtFree(xD));
    CHECK_ACL(aclrtFree(zD));
    CHECK_ACL(aclrtFree(maxD));
    CHECK_ACL(aclrtFree(sumD));
    CHECK_ACL(aclrtFree(wkD));
    CHECK_ACL(aclrtFree(tilD));

    CHECK_ACL(aclrtFreeHost(xH));
    CHECK_ACL(aclrtFreeHost(zH));
    CHECK_ACL(aclrtFreeHost(maxH));
    CHECK_ACL(aclrtFreeHost(sumH));
    CHECK_ACL(aclrtFreeHost(wkH));
    CHECK_ACL(aclrtFreeHost(tilH));
}
#endif

int32_t main(int32_t, char **)
{
    // 读取环境变量（均为可选）
    uint32_t warmup = GetEnvU32("SOFTMAX_WARMUP", 1); // 每 case 预热次数
    uint32_t iters = GetEnvU32("SOFTMAX_ITERS", 1);  // 每 case 计次
    auto cases = ParseCasesFromEnv();
    if (cases.empty())
        cases = DefaultCases();

#ifndef ASCENDC_CPU_DEBUG
    // NPU 域初始化（延用原工程写法）
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
#endif

    std::cout << "[INFO] total cases = " << cases.size()
              << ", warmup=" << warmup << ", iters=" << iters << std::endl;

    for (auto [M, K] : cases)
    {
#ifdef ASCENDC_CPU_DEBUG
        RunOneCaseCPU(M, K, iters, warmup);
#else
        RunOneCaseNPU(M, K, iters, warmup, stream);
#endif
    }

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif

    std::cout << "[INFO] sweep finished." << std::endl;
    return 0;
}
