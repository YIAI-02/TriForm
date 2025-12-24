// #include "data_utils.h"
// #ifndef ASCENDC_CPU_DEBUG
// #include "acl/acl.h"
// #include "aclrtlaunch_mmad_custom.h"
// #else
// #include "tikicpulib.h"
// extern "C" void mmad_custom(uint8_t *a, uint8_t *b, uint8_t *bias, uint8_t *c);
// #endif

// static inline uint32_t RoundUp16(uint32_t x) { return (x + 15) / 16 * 16; }
// static void ParseArgs(int argc, char** argv, uint32_t& M, uint32_t& N, uint32_t& K, uint32_t& repeat) {
//     M = 32; N = 32; K = 32; repeat = 1;
//     for (int i = 1; i < argc; ++i) {
//         std::string a = argv[i];
//         if (a == "--m" && i + 1 < argc) M = std::stoul(argv[++i]);
//         else if (a == "--n" && i + 1 < argc) N = std::stoul(argv[++i]);
//         else if (a == "--k" && i + 1 < argc) K = std::stoul(argv[++i]);
//         else if (a == "--repeat" && i + 1 < argc) repeat = std::stoul(argv[++i]);
//     }
// }

// int32_t main(int32_t argc, char *argv[])
// {
//     uint32_t M, N, K, REPEAT;
//     ParseArgs(argc, argv, M, N, K, REPEAT);
//     uint32_t Mp = RoundUp16(M), Np = RoundUp16(N), Kp = RoundUp16(K);
//     size_t aFileSize = size_t(Mp) * size_t(Kp) * sizeof(int16_t); // uint16_t represent half
//     size_t bFileSize = size_t(Kp) * size_t(Np) * sizeof(int16_t);
//     size_t biasFileSize = size_t(Np) * sizeof(int16_t);
//     size_t cFileSize = size_t(Mp) * size_t(Np) * sizeof(float);
//     uint32_t blockDim = 1;

// #ifdef ASCENDC_CPU_DEBUG
//     AscendC::SetKernelMode(KernelMode::AIC_MODE);
//     uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
//     uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
//     uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);
//     uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSize);

//     ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
//     ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
//     ReadFile("./input/bias_gm.bin", biasFileSize, bias, biasFileSize);

//     ICPU_RUN_KF(mmad_custom, blockDim, a, b, bias, c);

//     WriteFile("./output/output.bin", c, cFileSize);

//     AscendC::GmFree((void *)a); AscendC::GmFree((void *)b);
//     AscendC::GmFree((void *)bias); AscendC::GmFree((void *)c);
// #else
//     CHECK_ACL(aclInit(nullptr));
//     int32_t deviceId = 0;
//     CHECK_ACL(aclrtSetDevice(deviceId));
//     aclrtStream stream = nullptr;
//     CHECK_ACL(aclrtCreateStream(&stream));

//     uint8_t *aHost;  uint8_t *aDevice;
//     CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
//     CHECK_ACL(aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     ReadFile("./input/x1_gm.bin", aFileSize, aHost, aFileSize);
//     CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

//     uint8_t *bHost;  uint8_t *bDevice;
//     CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
//     CHECK_ACL(aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     ReadFile("./input/x2_gm.bin", bFileSize, bHost, bFileSize);
//     CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

//     uint8_t *biasHost;  uint8_t *biasDevice;
//     CHECK_ACL(aclrtMallocHost((void **)(&biasHost), biasFileSize));
//     CHECK_ACL(aclrtMalloc((void **)&biasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
//     ReadFile("./input/bias_gm.bin", biasFileSize, biasHost, biasFileSize);
//     CHECK_ACL(aclrtMemcpy(biasDevice, biasFileSize, biasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

//     uint8_t *cHost;  uint8_t *cDevice;
//     CHECK_ACL(aclrtMallocHost((void **)(&cHost), cFileSize));
//     CHECK_ACL(aclrtMalloc((void **)&cDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

//     for (uint32_t t = 0; t < REPEAT; ++t) {
//         ACLRT_LAUNCH_KERNEL(mmad_custom)(
//             blockDim, stream, aDevice, bDevice, biasDevice, cDevice,
//             Mp, Np, Kp);
//         CHECK_ACL(aclrtSynchronizeStream(stream));
//     }

//     CHECK_ACL(aclrtMemcpy(cHost, cFileSize, cDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
//     WriteFile("./output/output.bin", cHost, cFileSize);

//     CHECK_ACL(aclrtFree(aDevice));  CHECK_ACL(aclrtFreeHost(aHost));
//     CHECK_ACL(aclrtFree(bDevice));  CHECK_ACL(aclrtFreeHost(bHost));
//     CHECK_ACL(aclrtFree(biasDevice)); CHECK_ACL(aclrtFreeHost(biasHost));
//     CHECK_ACL(aclrtFree(cDevice));  CHECK_ACL(aclrtFreeHost(cHost));

//     CHECK_ACL(aclrtDestroyStream(stream));
//     CHECK_ACL(aclrtResetDevice(deviceId));
//     CHECK_ACL(aclFinalize());
// #endif
//     return 0;
// }


#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_mmad_custom.h"
#else
#include "tikicpulib.h"
extern "C" void mmad_custom(uint8_t *a, uint8_t *b, uint8_t *bias, uint8_t *c);
#endif

#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <tuple>
#include <sstream>
#include <iostream>

// ----------------- helpers -----------------
static inline uint32_t RoundUp16(uint32_t x) { return (x + 15) / 16 * 16; }

static inline bool GetEnvU32(const char* key, uint32_t& out) {
    const char* v = std::getenv(key);
    if (!v) return false;
    out = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
    return true;
}

static inline bool GetEnvBool(const char* key, bool defv=false) {
    const char* v = std::getenv(key);
    if (!v) return defv;
    std::string s(v);
    for (auto& c : s) c = std::tolower(c);
    return (s=="1"||s=="on"||s=="true"||s=="yes");
}

// "32x64x128,96x80x72" -> vector of (M,N,K)
static std::vector<std::tuple<uint32_t,uint32_t,uint32_t>> ParseCasesEnv() {
    std::vector<std::tuple<uint32_t,uint32_t,uint32_t>> cases;
    const char* env = std::getenv("CASES");
    if (!env) return cases;
    std::string s(env);
    auto norm = [](char ch)->char { return (ch=='X'||ch=='*'||ch=='\xD7') ? 'x' : ch; };
    for (auto& ch : s) ch = norm(ch);
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        for (auto& ch : item) ch = norm(ch);
        std::stringstream is(item);
        std::string a,b,c;
        if (std::getline(is, a, 'x') && std::getline(is, b, 'x') && std::getline(is, c, 'x')) {
            uint32_t M = std::strtoul(a.c_str(), nullptr, 10);
            uint32_t N = std::strtoul(b.c_str(), nullptr, 10);
            uint32_t K = std::strtoul(c.c_str(), nullptr, 10);
            if (M>0 && N>0 && K>0) cases.emplace_back(M,N,K);
        }
    }
    return cases;
}

static void ParseArgs(int argc, char** argv, uint32_t& M, uint32_t& N, uint32_t& K, uint32_t& repeat) {
    // default
    M = 32; N = 32; K = 32; repeat = 1;
    // env fallback (so app can run under msprof without args)
    (void)GetEnvU32("M", M);
    (void)GetEnvU32("N", N);
    (void)GetEnvU32("K", K);
    (void)GetEnvU32("REPEAT", repeat);

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--m" && i + 1 < argc) M = std::stoul(argv[++i]);
        else if (a == "--n" && i + 1 < argc) N = std::stoul(argv[++i]);
        else if (a == "--k" && i + 1 < argc) K = std::stoul(argv[++i]);
        else if (a == "--repeat" && i + 1 < argc) repeat = std::stoul(argv[++i]);
    }
}

static inline void FillZeros(aclFloat16* p, size_t count) {
    // aclFloat16 is 2 bytes POD; zeroing memory sets 0.0h
    std::memset(p, 0, count * sizeof(aclFloat16));
}

static inline void FillZerosF32(float* p, size_t count) {
    std::memset(p, 0, count * sizeof(float));
}

static inline uint32_t NextLCG(uint32_t& s) {
    s = 1664525u * s + 1013904223u;
    return s;
}

static inline float RandUniformNegPos(uint32_t& s) {
    // range ~[-0.5, 0.5)
    float v = (NextLCG(s) >> 8) * (1.0f / 16777216.0f) - 0.5f;
    return v;
}

// Fill a (rows x cols) into a (rowsPad x colsPad) fp16 buffer with row major leading dim = colsPad
static void FillHalfMatrixPadded(aclFloat16* buf, uint32_t rows, uint32_t cols, uint32_t rowsPad, uint32_t colsPad, uint32_t seed) {
    // zero the whole pad first
    FillZeros(buf, size_t(rowsPad) * size_t(colsPad));
    uint32_t s = seed ? seed : 1234u;
    for (uint32_t i=0;i<rows;i++) {
        aclFloat16* rowPtr = buf + size_t(i) * colsPad;
        for (uint32_t j=0;j<cols;j++) {
            float val = RandUniformNegPos(s);
            rowPtr[j] = aclFloatToFloat16(val);
        }
    }
}

static void FillBiasHalfPadded(aclFloat16* buf, uint32_t n, uint32_t nPad) {
    // zero then (optionally) set small values
    FillZeros(buf, nPad);
    for (uint32_t i=0;i<n;i++) {
        buf[i] = aclFloatToFloat16(0.0f);
    }
}

// ----------------- main -----------------
int32_t main(int32_t argc, char *argv[])
{
    // If CASES is set, we iterate inside a single process (useful when running w/o shell loop).
    auto cases = ParseCasesEnv();

    bool noIO = GetEnvBool("NO_IO", true);      // default true for profiling
    bool dumpOutput = GetEnvBool("DUMP_OUTPUT", false);

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
#endif

    auto run_one = [&](uint32_t M, uint32_t N, uint32_t K, uint32_t REPEAT) -> int {
        // Use original dims for kernel, but allocate padded buffers for storage
        uint32_t Mp = RoundUp16(M), Np = RoundUp16(N), Kp = RoundUp16(K);
        size_t aFileSize = size_t(Mp) * size_t(Kp) * sizeof(aclFloat16);
        size_t bFileSize = size_t(Kp) * size_t(Np) * sizeof(aclFloat16);
        size_t biasFileSize = size_t(Np) * sizeof(aclFloat16);
        size_t cFileSize = size_t(M) * size_t(N) * sizeof(float);     // store only m*n by default
        size_t cFileSizePad = size_t(Mp) * size_t(Np) * sizeof(float);// internal/padded buffer

        uint32_t blockDim = 1;

#ifdef ASCENDC_CPU_DEBUG
        AscendC::SetKernelMode(KernelMode::AIC_MODE);
        uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
        uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
        uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);
        uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSizePad);

        bool filled = false;
        if (!noIO) {
            // try reading files; if missing, fall back to generated data
            size_t sz = 0;
            filled = ReadFile("./input/x1_gm.bin", sz, a, aFileSize)
                  && ReadFile("./input/x2_gm.bin", sz, b, bFileSize)
                  && ReadFile("./input/bias_gm.bin", sz, bias, biasFileSize);
        }
        if (!filled) {
            FillHalfMatrixPadded(reinterpret_cast<aclFloat16*>(a), M, K, Mp, Kp, 1234u);
            FillHalfMatrixPadded(reinterpret_cast<aclFloat16*>(b), K, N, Kp, Np, 5678u);
            FillBiasHalfPadded(reinterpret_cast<aclFloat16*>(bias), N, Np);
        }

        // run kernel REPEAT times
        for (uint32_t t=0; t<REPEAT; ++t) {
            ICPU_RUN_KF(mmad_custom, blockDim,
                        a, b, bias, c);
        }

        if (dumpOutput) {
            // write only m*n valid area
            std::vector<float> c_mn(size_t(M)*size_t(N));
            // copy out and slice valid region row by row
            float* cFull = reinterpret_cast<float*>(c);
            for (uint32_t i=0;i<M;i++) {
                std::memcpy(&c_mn[size_t(i)*N], cFull + size_t(i)*Np, N*sizeof(float));
            }
            WriteFile("./output/output.bin", c_mn.data(), cFileSize);
        }

        AscendC::GmFree((void *)a); AscendC::GmFree((void *)b);
        AscendC::GmFree((void *)bias); AscendC::GmFree((void *)c);
#else
        uint8_t *aHost;  uint8_t *aDevice;
        CHECK_ACL(aclrtMallocHost((void **)(&aHost), aFileSize));
        CHECK_ACL(aclrtMalloc((void **)&aDevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t *bHost;  uint8_t *bDevice;
        CHECK_ACL(aclrtMallocHost((void **)(&bHost), bFileSize));
        CHECK_ACL(aclrtMalloc((void **)&bDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t *biasHost;  uint8_t *biasDevice;
        CHECK_ACL(aclrtMallocHost((void **)(&biasHost), biasFileSize));
        CHECK_ACL(aclrtMalloc((void **)&biasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

        uint8_t *cHostPad;  uint8_t *cDevice;
        CHECK_ACL(aclrtMallocHost((void **)(&cHostPad), cFileSizePad));
        CHECK_ACL(aclrtMalloc((void **)&cDevice, cFileSizePad, ACL_MEM_MALLOC_HUGE_FIRST));

        bool filled = false;
        if (!noIO) {
            // Try reading pre-generated inputs
            size_t szA=0, szB=0, szBias=0;
            bool okA = ReadFile("./input/x1_gm.bin", szA, aHost, aFileSize);
            bool okB = ReadFile("./input/x2_gm.bin", szB, bHost, bFileSize);
            bool okBias = ReadFile("./input/bias_gm.bin", szBias, biasHost, biasFileSize);
            filled = (okA && okB && okBias);
        }
        if (!filled) {
            FillHalfMatrixPadded(reinterpret_cast<aclFloat16*>(aHost), M, K, Mp, Kp, 1234u);
            FillHalfMatrixPadded(reinterpret_cast<aclFloat16*>(bHost), K, N, Kp, Np, 5678u);
            FillBiasHalfPadded(reinterpret_cast<aclFloat16*>(biasHost), N, Np);
        }

        CHECK_ACL(aclrtMemcpy(aDevice, aFileSize, aHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(bDevice, bFileSize, bHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(biasDevice, biasFileSize, biasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

        for (uint32_t t = 0; t < REPEAT; ++t) {
            // IMPORTANT: pass original (unpadded) M/N/K to the kernel so it can handle tail correctly
            ACLRT_LAUNCH_KERNEL(mmad_custom)(
                blockDim, stream, aDevice, bDevice, biasDevice, cDevice,
                M, N, K);
            CHECK_ACL(aclrtSynchronizeStream(stream));
        }

        if (dumpOutput) {
            // Copy back and slice valid region to ./output/output.bin
            CHECK_ACL(aclrtMemcpy(cHostPad, cFileSizePad, cDevice, cFileSizePad, ACL_MEMCPY_DEVICE_TO_HOST));
            std::vector<float> c_mn(size_t(M)*size_t(N));
            float* cFull = reinterpret_cast<float*>(cHostPad);
            for (uint32_t i=0;i<M;i++) {
                std::memcpy(&c_mn[size_t(i)*N], cFull + size_t(i)*Np, N*sizeof(float));
            }
            (void)WriteFile("./output/output.bin", c_mn.data(), cFileSize);
        }

        CHECK_ACL(aclrtFree(aDevice));  CHECK_ACL(aclrtFreeHost(aHost));
        CHECK_ACL(aclrtFree(bDevice));  CHECK_ACL(aclrtFreeHost(bHost));
        CHECK_ACL(aclrtFree(biasDevice)); CHECK_ACL(aclrtFreeHost(biasHost));
        CHECK_ACL(aclrtFree(cDevice));  CHECK_ACL(aclrtFreeHost(cHostPad));
#endif
        return 0;
    };

    int rc = 0;
    if (!cases.empty()) {
        // If REPEAT is set in env, reuse; else default 5 in batch mode
        uint32_t repeat = 5;
        (void)GetEnvU32("REPEAT", repeat);
        for (auto tup : cases) {
            uint32_t M,N,K; std::tie(M,N,K) = tup;
            INFO_LOG("Run case %ux%ux%u (repeat=%u)", M, N, K, repeat);
            rc |= run_one(M,N,K,repeat);
        }
    } else {
        uint32_t M,N,K,REPEAT;
        ParseArgs(argc, argv, M, N, K, REPEAT);
        INFO_LOG("Run case %ux%ux%u (repeat=%u)", M, N, K, REPEAT);
        rc = run_one(M,N,K,REPEAT);
    }

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
#endif
    return rc;
}
