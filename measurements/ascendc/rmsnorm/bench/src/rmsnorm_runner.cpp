#include "rmsnorm_runner.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

// === Placeholders for your project's device runtime ====================================
// Replace these 5 functions with your actual ACL/AscendC device runtime calls.
static void* DeviceAlloc(size_t nbytes) {
    // TODO: replace with aclrtMalloc or your project's allocator
    return std::malloc(nbytes);  // CPU fallback for bring-up
}
static void DeviceFree(void* p) {
    // TODO: replace with aclrtFree or your project's deallocator
    std::free(p);
}
static bool MemcpyH2D(void* dst_device, const void* src_host, size_t nbytes) {
    // TODO: replace with aclrtMemcpy with ACL_MEMCPY_HOST_TO_DEVICE
    std::memcpy(dst_device, src_host, nbytes);
    return true;
}
static bool MemcpyD2H(void* dst_host, const void* src_device, size_t nbytes) {
    // TODO: replace with aclrtMemcpy with ACL_MEMCPY_DEVICE_TO_HOST
    std::memcpy(dst_host, src_device, nbytes);
    return true;
}
static void DeviceSync() {
    // TODO: replace with aclrtSynchronizeStream or similar
}

// This is the ONLY place you need to actually launch your kernel.
// Hook this to your AscendC RMSNorm kernel entry (like your MMAD example).
static bool LaunchRmsNormKernel(void* x, void* gamma, void* y,
                                int B, int S, int H_pad, int H_orig,
                                int dtype /*0=fp16,1=fp32*/, float eps) {
    // ------------------------------------------------------------------------------------
    // TODO:
    //   - Call into your actual kernel launch function here.
    //   - Pass [B,S,H_pad] as tensor shape with originHLength=H_orig to respect tail.
    //   - Ensure the launch path uses the same tiling as in your uploaded impl
    //     (GetRmsNormTilingInfo / GetRmsNormMaxMinTmpSize).
    //   - The device-side compute follows the steps in the common impl.
    //
    // For bring-up on CPU, we provide a naive fallback so the harness runs end-to-end
    // without device libs; REMOVE this once you've wired in the real kernel.
    // ------------------------------------------------------------------------------------
    (void)eps;
    if (dtype == 1 /*fp32*/) {
        const float* X = reinterpret_cast<const float*>(x);
        const float* G = reinterpret_cast<const float*>(gamma);
        float* Y = reinterpret_cast<float*>(y);
        for (size_t bs=0; bs<(size_t)B*S; ++bs) {
            double acc=0.0;
            for (int i=0;i<H_orig;i++) {
                double v = X[bs*H_pad + i];
                acc += v*v;
            }
            double rms = std::sqrt(acc / H_orig + eps);
            for (int i=0;i<H_orig;i++) {
                Y[bs*H_pad + i] = static_cast<float>( (X[bs*H_pad + i] / rms) * G[i] );
            }
            for (int i=H_orig;i<H_pad;i++) Y[bs*H_pad + i] = 0.0f;
        }
        return true;
    } else {
        const uint16_t* Xh = reinterpret_cast<const uint16_t*>(x); // fp16 storage (IEEE 754 binary16)
        const uint16_t* Gh = reinterpret_cast<const uint16_t*>(gamma);
        uint16_t* Yh = reinterpret_cast<uint16_t*>(y);
        auto h2f=[&](uint16_t h)->float{
            uint16_t s = (h>>15)&1u;
            uint16_t e = (h>>10)&0x1Fu;
            uint16_t m = h & 0x3FFu;
            float f;
            if (e==0) f = (m ? std::ldexp((float)m, -24) : 0.0f);
            else if (e==31) f = m ? NAN : INFINITY;
            else f = std::ldexp((float)(m | 0x400), (int)e-25);
            return s? -f : f;
        };
        auto f2h=[&](float f)->uint16_t{
            int s = std::signbit(f); if (s) f = -f;
            if (!std::isfinite(f)) return (s<<15) | (31<<10);
            int e; float m = std::frexp(f, &e); e += 14;
            if (e<=0) return (uint16_t)(s<<15);
            if (e>=31) return (uint16_t)((s<<15) | (31<<10));
            m = (m*2.0f) - 1.0f; uint16_t mant = (uint16_t)std::lrint(m * 1024.0f);
            return (uint16_t)((s<<15) | ((e&31)<<10) | (mant&0x3FF));
        };
        for (size_t bs=0; bs<(size_t)B*S; ++bs) {
            double acc=0.0;
            for (int i=0;i<H_orig;i++) {
                double v = h2f(Xh[bs*H_pad + i]);
                acc += v*v;
            }
            double rms = std::sqrt(acc / H_orig + eps);
            for (int i=0;i<H_orig;i++) {
                float xv = h2f(Xh[bs*H_pad + i]);
                float gv = h2f(Gh[i]);
                Yh[bs*H_pad + i] = f2h( (float)(xv / rms * gv) );
            }
            for (int i=H_orig;i<H_pad;i++) Yh[bs*H_pad + i] = 0;
        }
        return true;
    }
}

static int align_up(int x, int a) { return (x + a - 1) / a * a; }

bool RmsnormRunner::Init(const RmsnormConfig& cfg) {
    cfg_ = cfg;
    elem_size_ = (cfg.dtype==RmsnormConfig::FP16 ? 2 : 4);
    const int align_elems = 32 / (int)elem_size_; // H must be 32B aligned on device
    H_pad_ = align_up(cfg.H, align_elems);

    bytes_x_ = (size_t)cfg.B * cfg.S * H_pad_ * elem_size_;
    bytes_y_ = bytes_x_;
    bytes_gamma_ = (size_t)H_pad_ * elem_size_;

    bool ok = Allocate();
    if (!ok) {
        std::cerr << "[ERROR] Device alloc failed\n";
        return false;
    }

    // init host buffers with deterministic values
    std::vector<uint8_t> hx(bytes_x_, 0), hy(bytes_y_, 0), hgamma(bytes_gamma_, 0);
    if (cfg.dtype == RmsnormConfig::FP32) {
        float* px = reinterpret_cast<float*>(hx.data());
        float* pg = reinterpret_cast<float*>(hgamma.data());
        for (size_t i=0;i<((size_t)cfg.B*cfg.S*H_pad_);++i) px[i] = float((i%113)+1)/113.0f;
        for (int i=0;i<cfg.H;i++) pg[i] = 1.0f;                 // gamma=1 for simplicity
        for (int i=cfg.H;i<H_pad_;i++) pg[i] = 0.0f;            // pad tail gamma=0
    } else {
        uint16_t* px = reinterpret_cast<uint16_t*>(hx.data());
        uint16_t* pg = reinterpret_cast<uint16_t*>(hgamma.data());
        auto f2h=[&](float f)->uint16_t{
            int s = std::signbit(f); if (s) f = -f;
            if (!std::isfinite(f)) return (s<<15) | (31<<10);
            int e; float m = std::frexp(f, &e); e += 14;
            if (e<=0) return (uint16_t)(s<<15);
            if (e>=31) return (uint16_t)((s<<15) | (31<<10));
            m = (m*2.0f) - 1.0f; uint16_t mant = (uint16_t)std::lrint(m * 1024.0f);
            return (uint16_t)((s<<15) | ((e&31)<<10) | (mant&0x3FF));
        };
        for (size_t i=0;i<((size_t)cfg.B*cfg.S*H_pad_);++i) px[i] = f2h(float((i%113)+1)/113.0f);
        for (int i=0;i<cfg.H;i++) pg[i] = f2h(1.0f);
        for (int i=cfg.H;i<H_pad_;i++) pg[i] = f2h(0.0f);
    }

    // H2D
    if (!MemcpyH2D(d_x_, hx.data(), bytes_x_)) return false;
    if (!MemcpyH2D(d_gamma_, hgamma.data(), bytes_gamma_)) return false;
    return true;
}

bool RmsnormRunner::Allocate() {
    d_x_ = DeviceAlloc(bytes_x_);
    d_y_ = DeviceAlloc(bytes_y_);
    d_gamma_ = DeviceAlloc(bytes_gamma_);
    return d_x_ && d_y_ && d_gamma_;
}
void RmsnormRunner::Release() {
    if (d_x_) DeviceFree(d_x_), d_x_=nullptr;
    if (d_y_) DeviceFree(d_y_), d_y_=nullptr;
    if (d_gamma_) DeviceFree(d_gamma_), d_gamma_=nullptr;
}
RmsnormRunner::~RmsnormRunner() { Release(); }

bool RmsnormRunner::RunOnce() {
    return LaunchRmsNormKernel(
        d_x_, d_gamma_, d_y_,
        cfg_.B, cfg_.S, H_pad_, cfg_.H,
        (cfg_.dtype==RmsnormConfig::FP16?0:1), cfg_.eps
    );
}
void RmsnormRunner::SyncDevice() { DeviceSync(); }
