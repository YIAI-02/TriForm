#include "../../../common/data_utils.h"
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
extern void softmax_custom_do(uint32_t coreDim, void* l2ctrl, void* stream,
                              uint8_t* x, uint8_t* max, uint8_t* sum,
                              uint8_t* z, uint8_t* workspace, uint8_t* tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR max, GM_ADDR sum,
                                                     GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);
#endif

// === 与原工程中保持一致的常量 ===
constexpr uint32_t USED_CORE_NUM     = 40;          // 建议保持不变（可按芯片实际核数调优）
constexpr uint32_t WORKSPACE_SIZE    = 1024;        // float 元素个数（workspace 只是占位）
constexpr uint32_t TILINGDATA_SIZE   = 28;          // SoftmaxCustomTilingData 的 u32 数量
constexpr uint32_t FLOATS_PER_BLOCK8 = 8;           // 每行 max/sum 写回长度系数

extern void GenerateTiling(const uint32_t m, const uint32_t k,
                           const uint32_t coreNum, const uint32_t tilingSize,
                           uint8_t* tilingData);    // 来自 softmax_custom_tiling.cpp

// ========== 简单的工具函数 ==========
static std::vector<std::pair<uint32_t,uint32_t>> ParseCasesFromEnv()
{
    std::vector<std::pair<uint32_t,uint32_t>> cases;
    const char* one = std::getenv("SOFTMAX_CASE");   // 单个：如 "1024x1024", 返回这个环境变量的指针
    const char* many = std::getenv("SOFTMAX_CASES"); // 多个：如 "128x128,256x512,1024x2048"
    auto parse_one = [](const std::string& s, uint32_t& m, uint32_t& k)->bool{
        auto pos = s.find_first_of("xX*");
        if (pos == std::string::npos) return false;
        try {
            m = static_cast<uint32_t>(std::stoul(s.substr(0, pos)));
            k = static_cast<uint32_t>(std::stoul(s.substr(pos + 1)));
            return (m > 0 && k > 0);
        } catch (...) { return false; }
    };

    if (one && *one) {
        uint32_t m,k;
        if (parse_one(one, m, k)) cases.emplace_back(m,k);
    }
    if (many && *many) {
        std::string s(many);
        size_t start = 0;
        while (start < s.size()) {
            size_t comma = s.find(',', start);
            std::string token = s.substr(start, (comma == std::string::npos ? s.size() : comma) - start);
            uint32_t m,k;
            if (!token.empty() && parse_one(token, m, k)) cases.emplace_back(m,k);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    return cases;
}

static uint32_t GetEnvU32(const char* name, uint32_t defv)
{
    const char* p = std::getenv(name);
    if (!p || !*p) return defv;
    try {
        auto v = static_cast<uint64_t>(std::stoull(p));
        return (v > 0 && v < (1ull<<32)) ? static_cast<uint32_t>(v) : defv;
    } catch (...) { return defv; }
}

// 默认尺寸表：覆盖典型 Transformer/推荐场景（注意单次内存占用）
static std::vector<std::pair<uint32_t,uint32_t>> DefaultCases()
{
    // 控制单次分配 < ~512MB：m*k*sizeof(float) ≲ 512MB
    // 例： (4096,32768) 就过大了，这里不放
    return {
        {16,128}, {32,128}, {64,128},
        {32,256}, {64,256}, {128,256},
        {64,384}, {128,384},
        {64,512}, {128,512}, {256,512},
        {128,768}, {256,768},
        {128,1024}, {256,1024}, {512,1024},
        {256,1536}, {512,1536},
        {256,2048}, {512,2048}, {1024,2048},
        {512,3072}, {1024,3072},
        {512,4096}, {1024,4096}
    };
}

static void FillRandomFloats(float* dst, size_t n)
{
    // 固定种子便于复现
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) dst[i] = dist(rng);
}

// ========== 单个 Case 的执行（CPU/NPU 分别实现） ==========
#ifdef ASCENDC_CPU_DEBUG
static void RunOneCaseCPU(uint32_t M, uint32_t K, uint32_t iters, uint32_t warmup)
{
    size_t inBytes  = static_cast<size_t>(M) * K * sizeof(float);
    size_t outBytes = inBytes;
    size_t msBytes  = static_cast<size_t>(M) * FLOATS_PER_BLOCK8 * sizeof(float);
    size_t wkBytes  = static_cast<size_t>(WORKSPACE_SIZE) * sizeof(float);
    size_t tilBytes = static_cast<size_t>(TILINGDATA_SIZE) * sizeof(uint32_t);

    std::cout << "[CPU] case M=" << M << " K=" << K
              << " warmup=" << warmup << " iters=" << iters << std::endl;

    for (uint32_t phase = 0; phase < warmup + iters; ++phase) {
        uint8_t* x   = (uint8_t*)AscendC::GmAlloc(inBytes);
        uint8_t* max = (uint8_t*)AscendC::GmAlloc(msBytes);
        uint8_t* sum = (uint8_t*)AscendC::GmAlloc(msBytes);
        uint8_t* z   = (uint8_t*)AscendC::GmAlloc(outBytes);
        uint8_t* wk  = (uint8_t*)AscendC::GmAlloc(wkBytes);
        uint8_t* til = (uint8_t*)AscendC::GmAlloc(tilBytes);

        FillRandomFloats(reinterpret_cast<float*>(x), (size_t)M*K);
        std::memset(wk, 0, wkBytes);
        GenerateTiling(M, K, USED_CORE_NUM, tilBytes, til);

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(softmax_custom, USED_CORE_NUM, x, max, sum, z, wk, til);

        AscendC::GmFree(x); AscendC::GmFree(max); AscendC::GmFree(sum);
        AscendC::GmFree(z); AscendC::GmFree(wk);  AscendC::GmFree(til);
    }
}
#else
static void RunOneCaseNPU(uint32_t M, uint32_t K, uint32_t iters, uint32_t warmup,
                          aclrtStream stream)
{
    size_t inBytes  = static_cast<size_t>(M) * K * sizeof(float);
    size_t outBytes = inBytes;
    size_t msBytes  = static_cast<size_t>(M) * FLOATS_PER_BLOCK8 * sizeof(float);
    size_t wkBytes  = static_cast<size_t>(WORKSPACE_SIZE) * sizeof(float);
    size_t tilBytes = static_cast<size_t>(TILINGDATA_SIZE) * sizeof(uint32_t);

    std::cout << "[NPU] case M=" << M << " K=" << K
              << " warmup=" << warmup << " iters=" << iters << std::endl;

    // host 缓冲区（每次复用即可，避免反复 malloc/free）
    uint8_t *xH, *zH, *maxH, *sumH, *wkH, *tilH;
    CHECK_ACL(aclrtMallocHost((void**)(&xH), inBytes));
    CHECK_ACL(aclrtMallocHost((void**)(&zH), outBytes));
    CHECK_ACL(aclrtMallocHost((void**)(&maxH), msBytes));
    CHECK_ACL(aclrtMallocHost((void**)(&sumH), msBytes));
    CHECK_ACL(aclrtMallocHost((void**)(&wkH),  wkBytes));
    CHECK_ACL(aclrtMallocHost((void**)(&tilH), tilBytes));

    // device 缓冲区
    uint8_t *xD, *zD, *maxD, *sumD, *wkD, *tilD;
    CHECK_ACL(aclrtMalloc((void**)&xD,   inBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zD,   outBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&maxD, msBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&sumD, msBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&wkD,  wkBytes,  ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&tilD, tilBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    FillRandomFloats(reinterpret_cast<float*>(xH), (size_t)M*K);
    std::memset(wkH, 0, wkBytes);
    GenerateTiling(M, K, USED_CORE_NUM, tilBytes, tilH);

    // 预传 tiling 不变部分
    CHECK_ACL(aclrtMemcpy(tilD, tilBytes, tilH, tilBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto run_once = [&](){
        // 每次迭代给 x 拷贝不同数据以避免某些编译器/硬件路径优化
        FillRandomFloats(reinterpret_cast<float*>(xH), (size_t)M*K);

        CHECK_ACL(aclrtMemcpy(xD,  inBytes,  xH,  inBytes,  ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(wkD, wkBytes,  wkH, wkBytes,  ACL_MEMCPY_HOST_TO_DEVICE));

        softmax_custom_do(USED_CORE_NUM, nullptr, stream, xD, maxD, sumD, zD, wkD, tilD);
        CHECK_ACL(aclrtSynchronizeStream(stream));

        // 回传 z 以确保写回发生（不落盘）
        CHECK_ACL(aclrtMemcpy(zH, outBytes, zD, outBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    };

    // warmup + iters
    for (uint32_t i = 0; i < warmup + iters; ++i) run_once();

    // 释放
    CHECK_ACL(aclrtFree(xD));  CHECK_ACL(aclrtFree(zD));
    CHECK_ACL(aclrtFree(maxD));CHECK_ACL(aclrtFree(sumD));
    CHECK_ACL(aclrtFree(wkD)); CHECK_ACL(aclrtFree(tilD));

    CHECK_ACL(aclrtFreeHost(xH));  CHECK_ACL(aclrtFreeHost(zH));
    CHECK_ACL(aclrtFreeHost(maxH));CHECK_ACL(aclrtFreeHost(sumH));
    CHECK_ACL(aclrtFreeHost(wkH)); CHECK_ACL(aclrtFreeHost(tilH));
}
#endif

int32_t main(int32_t, char**)
{
    // 读取环境变量（均为可选）
    uint32_t warmup = GetEnvU32("SOFTMAX_WARMUP", 2);   // 每 case 预热次数
    uint32_t iters  = GetEnvU32("SOFTMAX_ITERS",  10);  // 每 case 计次
    auto cases = ParseCasesFromEnv();
    if (cases.empty()) cases = DefaultCases();

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

    for (auto [M, K] : cases) {
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
