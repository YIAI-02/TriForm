#ifndef HEADER_ACLRTLAUNCH_MMAD_CUSTOM_H
#define HEADER_ACLRTLAUNCH_MMAD_CUSTOM_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_mmad_custom(uint32_t blockDim, aclrtStream stream, void* a, void* b, void* bias, void* c, uint32_t m, uint32_t n, uint32_t k);
#endif
