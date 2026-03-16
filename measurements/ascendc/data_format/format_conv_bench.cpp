#include "format_conv_bench_kernel.h"

extern "C" __global__ __aicore__ void format_conv_bench(
    GM_ADDR src,
    GM_ADDR scratch,
    uint32_t rows,
    uint32_t cols,
    uint32_t mode,
    uint32_t innerLoops)
{
    KernelFormatConvBench op;
    op.SetShape(rows, cols, mode, innerLoops);
    op.Init(src, scratch);
    op.Process();
}
