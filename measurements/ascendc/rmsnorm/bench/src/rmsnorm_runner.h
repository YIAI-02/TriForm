#pragma once
#include <cstdint>
#include <vector>
#include <string>

struct RmsnormConfig {
    enum DType { FP16=0, FP32=1 };
    int B{1}, S{1}, H{2048};
    DType dtype{FP16};
    float eps{1e-5f};
    bool verbose{false};
};

class RmsnormRunner {
public:
    bool Init(const RmsnormConfig& cfg);
    bool RunOnce();      // launch one kernel execution
    void SyncDevice();   // device synchronize (for timing correctness)
    ~RmsnormRunner();

private:
    RmsnormConfig cfg_;
    int H_pad_{0};
    size_t elem_size_{2};
    size_t bytes_x_{0};
    size_t bytes_y_{0};
    size_t bytes_gamma_{0};

    // Device buffers (opaque pointers; replace with your ACL/AscendC device ptr types)
    void* d_x_{nullptr};
    void* d_y_{nullptr};
    void* d_gamma_{nullptr};

    bool Allocate();
    void Release();
};
