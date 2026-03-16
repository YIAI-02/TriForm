#ifndef HEADER_ACLRTLAUNCH_FORMAT_CONV_BENCH_H
#define HEADER_ACLRTLAUNCH_FORMAT_CONV_BENCH_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_format_conv_bench(uint32_t blockDim, aclrtStream stream, void* src, void* scratch, uint32_t rows, uint32_t cols, uint32_t mode, uint32_t innerLoops);
#endif
