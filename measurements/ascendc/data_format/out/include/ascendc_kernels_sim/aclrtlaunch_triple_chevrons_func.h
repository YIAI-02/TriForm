
#ifndef HEADER_ACLRTLAUNCH_FORMAT_CONV_BENCH_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_FORMAT_CONV_BENCH_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_format_conv_bench(uint32_t blockDim, void* stream, void* src, void* scratch, uint32_t rows, uint32_t cols, uint32_t mode, uint32_t innerLoops);

inline uint32_t format_conv_bench(uint32_t blockDim, void* hold, void* stream, void* src, void* scratch, uint32_t rows, uint32_t cols, uint32_t mode, uint32_t innerLoops)
{
    (void)hold;
    return aclrtlaunch_format_conv_bench(blockDim, stream, src, scratch, rows, cols, mode, innerLoops);
}

#endif
