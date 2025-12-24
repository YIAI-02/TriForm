
#ifndef HEADER_ACLRTLAUNCH_MMAD_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_MMAD_CUSTOM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_mmad_custom(uint32_t blockDim, void* stream, void* a, void* b, void* bias, void* c, uint32_t m, uint32_t n, uint32_t k);

inline uint32_t mmad_custom(uint32_t blockDim, void* hold, void* stream, void* a, void* b, void* bias, void* c, uint32_t m, uint32_t n, uint32_t k)
{
    (void)hold;
    return aclrtlaunch_mmad_custom(blockDim, stream, a, b, bias, c, m, n, k);
}

#endif
