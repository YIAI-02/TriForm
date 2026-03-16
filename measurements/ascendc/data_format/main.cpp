#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_format_conv_bench.h"
#else
#include "tikicpulib.h"
extern "C" void format_conv_bench(uint8_t *src, uint8_t *scratch, uint32_t rows, uint32_t cols, uint32_t mode, uint32_t innerLoops);
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr uint32_t BLOCK = 16;
constexpr uint32_t TILE_ELEMS = BLOCK * BLOCK;

enum FormatConvMode : uint32_t {
    MODE_ND2NZ_A  = 0,
    MODE_ND2NZ_B  = 1,
    MODE_NZ2ZZ_A  = 2,
    MODE_NZ2ZN_B  = 3,
    MODE_ND2ZZ_A  = 4,
    MODE_ND2ZN_B  = 5,
};

static inline uint32_t CeilDiv(uint32_t x, uint32_t y) { return (x + y - 1) / y; }
static inline uint32_t RoundUp16(uint32_t x) { return CeilDiv(x, BLOCK) * BLOCK; }

bool GetEnvU32(const char* key, uint32_t& out) {
    const char* v = std::getenv(key);
    if (!v) return false;
    out = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
    return true;
}

void ParseArgs(int argc, char** argv,
               std::string& modeStr,
               uint32_t& M, uint32_t& N, uint32_t& K,
               uint32_t& repeat, uint32_t& innerLoops)
{
    modeStr = "nd2zn_b";
    M = 128; N = 128; K = 128; repeat = 10; innerLoops = 64;
    (void)GetEnvU32("M", M);
    (void)GetEnvU32("N", N);
    (void)GetEnvU32("K", K);
    (void)GetEnvU32("REPEAT", repeat);
    (void)GetEnvU32("INNER_LOOPS", innerLoops);
    if (const char* env = std::getenv("MODE")) modeStr = env;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--mode" || a == "-m0") && i + 1 < argc) modeStr = argv[++i];
        else if (a == "--m" && i + 1 < argc) M = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--n" && i + 1 < argc) N = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--k" && i + 1 < argc) K = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--repeat" && i + 1 < argc) repeat = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--inner_loops" && i + 1 < argc) innerLoops = static_cast<uint32_t>(std::stoul(argv[++i]));
    }
}

FormatConvMode ParseMode(const std::string& modeStr) {
    std::string s = modeStr;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "nd2nz_a") return MODE_ND2NZ_A;
    if (s == "nd2nz_b") return MODE_ND2NZ_B;
    if (s == "nz2zz_a") return MODE_NZ2ZZ_A;
    if (s == "nz2zn_b") return MODE_NZ2ZN_B;
    if (s == "nd2zz_a") return MODE_ND2ZZ_A;
    return MODE_ND2ZN_B;
}

void ResolveMatrixDims(FormatConvMode mode, uint32_t M, uint32_t N, uint32_t K, uint32_t& rows, uint32_t& cols) {
    switch (mode) {
        case MODE_ND2NZ_A:
        case MODE_NZ2ZZ_A:
        case MODE_ND2ZZ_A:
            rows = M; cols = K; break;
        case MODE_ND2NZ_B:
        case MODE_NZ2ZN_B:
        case MODE_ND2ZN_B:
        default:
            rows = K; cols = N; break;
    }
}

inline uint32_t NextLCG(uint32_t& s) {
    s = 1664525u * s + 1013904223u;
    return s;
}

inline float RandUniform(uint32_t& s) {
    return (NextLCG(s) >> 8) * (1.0f / 16777216.0f) - 0.5f;
}

void FillNdMatrix(std::vector<aclFloat16>& out, uint32_t rows, uint32_t cols, uint32_t seed) {
    out.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    uint32_t st = seed ? seed : 1234u;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            out[static_cast<size_t>(r) * cols + c] = aclFloatToFloat16(RandUniform(st));
        }
    }
}

void PackNdToNz(const std::vector<aclFloat16>& nd, uint32_t rows, uint32_t cols, std::vector<aclFloat16>& nz) {
    const uint32_t RB = CeilDiv(rows, BLOCK); //tile 个数
    const uint32_t CB = CeilDiv(cols, BLOCK); //
    nz.assign(static_cast<size_t>(RB) * static_cast<size_t>(CB) * TILE_ELEMS, aclFloatToFloat16(0.0f));

    for (uint32_t cb = 0; cb < CB; ++cb) {
        for (uint32_t rb = 0; rb < RB; ++rb) {
            const size_t tileBase = (static_cast<size_t>(cb) * RB + rb) * TILE_ELEMS;
            for (uint32_t i = 0; i < BLOCK; ++i) {
                const uint32_t r = rb * BLOCK + i;
                for (uint32_t j = 0; j < BLOCK; ++j) {
                    const uint32_t c = cb * BLOCK + j;
                    aclFloat16 v = aclFloatToFloat16(0.0f);
                    if (r < rows && c < cols) {
                        v = nd[static_cast<size_t>(r) * cols + c];
                    }
                    nz[tileBase + static_cast<size_t>(i) * BLOCK + j] = v;
                }
            }
        }
    }
}

std::vector<aclFloat16> BuildSourceBuffer(FormatConvMode mode, uint32_t rows, uint32_t cols) {
    std::vector<aclFloat16> nd;
    FillNdMatrix(nd, rows, cols, 12345u);
    if (mode == MODE_NZ2ZZ_A || mode == MODE_NZ2ZN_B) {
        std::vector<aclFloat16> nz;
        PackNdToNz(nd, rows, cols, nz);
        return nz;
    }
    return nd;
}

size_t SourceBytesForMode(FormatConvMode mode, uint32_t rows, uint32_t cols) {
    const size_t elem = sizeof(aclFloat16);
    if (mode == MODE_NZ2ZZ_A || mode == MODE_NZ2ZN_B) {
        return static_cast<size_t>(CeilDiv(rows, BLOCK)) * static_cast<size_t>(CeilDiv(cols, BLOCK)) * TILE_ELEMS * elem;
    }
    return static_cast<size_t>(rows) * static_cast<size_t>(cols) * elem;
}

}  // namespace

int main(int argc, char* argv[])
{
    std::string modeStr;
    uint32_t M, N, K, repeat, innerLoops;
    ParseArgs(argc, argv, modeStr, M, N, K, repeat, innerLoops);
    const FormatConvMode mode = ParseMode(modeStr);

    uint32_t rows = 0, cols = 0;
    ResolveMatrixDims(mode, M, N, K, rows, cols);

    const auto srcHostVec = BuildSourceBuffer(mode, rows, cols);
    const size_t srcBytes = srcHostVec.size() * sizeof(aclFloat16);
    const size_t scratchBytes = TILE_ELEMS * sizeof(aclFloat16);
    const uint32_t blockDim = 1;

    INFO_LOG("mode=%s M=%u N=%u K=%u rows=%u cols=%u repeat=%u inner_loops=%u src_bytes=%zu",
             modeStr.c_str(), M, N, K, rows, cols, repeat, innerLoops, srcBytes);

#ifdef ASCENDC_CPU_DEBUG
    AscendC::SetKernelMode(KernelMode::AIC_MODE);
    uint8_t* src = (uint8_t*)AscendC::GmAlloc(srcBytes);
    uint8_t* scratch = (uint8_t*)AscendC::GmAlloc(scratchBytes);
    std::memcpy(src, srcHostVec.data(), srcBytes);
    for (uint32_t t = 0; t < repeat; ++t) {
        ICPU_RUN_KF(format_conv_bench, blockDim, src, scratch, rows, cols, static_cast<uint32_t>(mode), innerLoops);
    }
    AscendC::GmFree((void*)src);
    AscendC::GmFree((void*)scratch);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t* srcHost = nullptr;
    uint8_t* srcDevice = nullptr;
    CHECK_ACL(aclrtMallocHost((void**)(&srcHost), srcBytes));
    CHECK_ACL(aclrtMalloc((void**)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memcpy(srcHost, srcHostVec.data(), srcBytes);
    CHECK_ACL(aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t* scratchHost = nullptr;
    uint8_t* scratchDevice = nullptr;
    CHECK_ACL(aclrtMallocHost((void**)(&scratchHost), scratchBytes));
    CHECK_ACL(aclrtMalloc((void**)&scratchDevice, scratchBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    std::memset(scratchHost, 0, scratchBytes);
    CHECK_ACL(aclrtMemcpy(scratchDevice, scratchBytes, scratchHost, scratchBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t t = 0; t < repeat; ++t) {
        ACLRT_LAUNCH_KERNEL(format_conv_bench)(
            blockDim, stream, srcDevice, scratchDevice,
            rows, cols, static_cast<uint32_t>(mode), innerLoops);
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double hostElapsedUs = std::chrono::duration<double, std::micro>(t1 - t0).count();

    INFO_LOG("HOST_ELAPSED_US total=%.3f avg=%.3f", hostElapsedUs, hostElapsedUs / std::max<uint32_t>(1, repeat));
    INFO_LOG("CSV,mode=%s,M=%u,N=%u,K=%u,rows=%u,cols=%u,repeat=%u,inner_loops=%u,src_bytes=%zu",
             modeStr.c_str(), M, N, K, rows, cols, repeat, innerLoops, srcBytes);

    CHECK_ACL(aclrtFree(srcDevice));
    CHECK_ACL(aclrtFreeHost(srcHost));
    CHECK_ACL(aclrtFree(scratchDevice));
    CHECK_ACL(aclrtFreeHost(scratchHost));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
