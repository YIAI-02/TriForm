
#include <acl/acl.h>
#include <acl/ops/acl_op.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>

#define ACL_THROW_IF(expr) do { aclError _ret = (expr); if (_ret != ACL_SUCCESS) { \
    std::cerr << "[ACL] Call failed: " << #expr << ", ret=" << _ret << std::endl; std::exit(1);} } while(0)

static std::string getenv_str(const char* k, const std::string& def="") {
    const char* v = std::getenv(k);
    return v ? std::string(v) : def;
}
static long long getenv_ll(const char* k, long long def) {
    const char* v = std::getenv(k);
    if (!v) return def;
    try { return std::stoll(v); } catch (...) { return def; }
}
static double getenv_double(const char* k, double def) {
    const char* v = std::getenv(k);
    if (!v) return def;
    try { return std::stod(v); } catch (...) { return def; }
}
static bool getenv_bool(const char* k, bool def=false) {
    const char* v = std::getenv(k);
    if (!v) return def;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s=="1" || s=="true" || s=="on" || s=="yes");
}

static std::vector<int64_t> parse_shape(const std::string& s) {
    // Accept separators: 'x', 'X', '*', ','
    std::vector<int64_t> dims;
    if (s.empty()) return dims;
    std::string cur; cur.reserve(s.size());
    for (char c: s) {
        if (c=='x' || c=='X' || c=='*' || c==',' || c==' ' ) {
            if (!cur.empty()) { dims.push_back(std::stoll(cur)); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) dims.push_back(std::stoll(cur));
    return dims;
}

struct DeviceTensor {
    std::vector<int64_t> dims;
    aclDataType dtype;
    size_t bytes;
    void* dev_ptr = nullptr;
    aclTensorDesc* desc = nullptr;
    aclDataBuffer* buf = nullptr;
    DeviceTensor() = default;
};

static size_t dtype_size(aclDataType t) {
    switch (t) {
        case ACL_FLOAT16: return 2;
        case ACL_FLOAT:   return 4;
        case ACL_INT32:   return 4;
        case ACL_INT64:   return 8;
        default: return 0;
    }
}

static DeviceTensor make_tensor(const std::vector<int64_t>& dims, aclDataType dtype) {
    DeviceTensor t;
    t.dims = dims;
    t.dtype = dtype;
    t.bytes = dtype_size(dtype);
    for (auto d : dims) t.bytes *= static_cast<size_t>(d);
    t.desc = aclCreateTensorDesc(dtype, dims.size(), dims.data(), ACL_FORMAT_ND);
    if (!t.desc) { std::cerr << "aclCreateTensorDesc failed\n"; std::exit(1); }
    ACL_THROW_IF(aclrtMalloc(&t.dev_ptr, t.bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    t.buf = aclCreateDataBuffer(t.dev_ptr, t.bytes);
    if (!t.buf) { std::cerr << "aclCreateDataBuffer failed\n"; std::exit(1); }
    return t;
}

static void destroy_tensor(DeviceTensor& t) {
    if (t.buf)  { (void)aclDestroyDataBuffer(t.buf); t.buf=nullptr; }
    if (t.desc) { (void)aclDestroyTensorDesc(t.desc); t.desc=nullptr; }
    if (t.dev_ptr) { (void)aclrtFree(t.dev_ptr); t.dev_ptr=nullptr; }
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static const char* dtype_from_env() {
    std::string dt = getenv_str("DTYPE", "fp16");
    std::transform(dt.begin(), dt.end(), dt.begin(), ::tolower);
    if (dt=="fp32" || dt=="float" || dt=="f32") return "fp32";
    return "fp16";
}

static aclDataType acl_dtype_from_str(const std::string& s) {
    if (s=="fp32") return ACL_FLOAT;
    return ACL_FLOAT16;
}

static void check_stream_sync(const char* tag, aclrtStream stream) {
    (void)tag;
    ACL_THROW_IF(aclrtSynchronizeStream(stream));
}

// Execute a single primitive op using ACL "aclopCompileAndExecute"
static void exec_single(const char* opType,
                        const std::vector<DeviceTensor*>& inputs,
                        const std::vector<DeviceTensor*>& outputs,
                        aclopAttr* attr,
                        aclrtStream stream)
{
    std::vector<const aclTensorDesc*> in_desc(inputs.size());
    std::vector<const aclDataBuffer*> in_buf(inputs.size());
    std::vector<const aclTensorDesc*> out_desc(outputs.size());
    std::vector<const aclDataBuffer*> out_buf(outputs.size());
    for (size_t i=0;i<inputs.size();++i){ in_desc[i]=inputs[i]->desc; in_buf[i]=inputs[i]->buf; }
    for (size_t i=0;i<outputs.size();++i){ out_desc[i]=outputs[i]->desc; out_buf[i]=outputs[i]->buf; }
    ACL_THROW_IF(aclopCompileAndExecute(opType,
        inputs.size(), in_desc.data(), in_buf.data(),
        outputs.size(), out_desc.data(), out_buf.data(),
        attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr, stream));
}

// Build a constant tensor filled with scalar on device (shape [1])
static DeviceTensor make_scalar(double v, aclDataType dtype) {
    DeviceTensor t = make_tensor({1}, dtype);
    // host tmp
    size_t sz = dtype_size(dtype);
    std::vector<uint8_t> host(sz);
    if (dtype==ACL_FLOAT) {
        float fv = static_cast<float>(v);
        std::memcpy(host.data(), &fv, sizeof(float));
    } else if (dtype==ACL_FLOAT16) {
        // naive fp16 conversion (round to nearest) – not perfect but sufficient for eps
        float fv = static_cast<float>(v);
        // fp32->fp16 conversion
        // simple implementation
        uint32_t x; std::memcpy(&x, &fv, 4);
        uint32_t sign = (x >> 31) & 0x1;
        int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (x >> 13) & 0x3FF;
        uint16_t h;
        if (exp <= 0) {
            h = (uint16_t)(sign<<15); // underflow -> 0
        } else if (exp >= 31) {
            h = (uint16_t)((sign<<15) | (31<<10)); // inf
        } else {
            h = (uint16_t)((sign<<15) | ((exp&0x1F)<<10) | mant);
        }
        std::memcpy(host.data(), &h, 2);
    } else {
        // default write zeros
        std::fill(host.begin(), host.end(), 0);
    }
    ACL_THROW_IF(aclrtMemcpy(t.dev_ptr, t.bytes, host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    return t;
}

// GELU single-op runner
static void run_gelu(const std::vector<int64_t>& dims, int repeat, int warmup,
                     aclDataType dtype, aclrtStream stream)
{
    DeviceTensor x = make_tensor(dims, dtype);
    DeviceTensor y = make_tensor(dims, dtype);
    // Warmup
    for (int i=0;i<warmup;++i) {
        exec_single("Gelu", {&x}, {&y}, nullptr, stream);
    }
    check_stream_sync("gelu-warmup", stream);

    std::vector<double> times_ms;
    for (int i=0;i<repeat;++i) {
        double t0 = now_ms();
        exec_single("Gelu", {&x}, {&y}, nullptr, stream);
        check_stream_sync("gelu", stream);
        double t1 = now_ms();
        times_ms.push_back(t1 - t0);
    }

    auto avg = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    std::nth_element(times_ms.begin(), times_ms.begin()+times_ms.size()/2, times_ms.end());
    double p50 = times_ms[times_ms.size()/2];
    std::cout << "[RESULT] op=GELU dtype=" << (dtype==ACL_FLOAT?"fp32":"fp16")
              << " shape=";
    for (size_t i=0;i<dims.size();++i) std::cout << (i? "x":"") << dims[i];
    std::cout << " repeat=" << repeat << " avg_ms=" << avg << " p50_ms=" << p50 << std::endl;

    destroy_tensor(y);
    destroy_tensor(x);
}

// RMSNorm implemented as primitives: y = x / sqrt(mean(x^2, axis) + eps) * gamma(optional)
static void run_rmsnorm_chain(const std::vector<int64_t>& dims, int repeat, int warmup,
                              aclDataType dtype, double eps, bool with_gamma,
                              aclrtStream stream)
{
    if (dims.empty()) { std::cerr << "RMSNorm requires at least 1D\n"; std::exit(1); }
    int64_t axis = (int64_t)dims.size() - 1; // normalize over last dim
    int64_t H = dims.back();

    DeviceTensor x = make_tensor(dims, dtype);
    DeviceTensor x2 = make_tensor(dims, dtype);
    std::vector<int64_t> mean_dims = dims; mean_dims.back() = 1;
    DeviceTensor mean = make_tensor(mean_dims, dtype);
    DeviceTensor eps_c = make_scalar(eps, dtype);
    DeviceTensor den = make_tensor(mean_dims, dtype);   // sqrt(mean+eps)
    // den has shape [..., 1]; elementwise ops will broadcast automatically to full shape
    DeviceTensor y = make_tensor(dims, dtype);
    DeviceTensor gamma;
    if (with_gamma) {
        gamma = make_tensor({H}, dtype);
    }

    // attrs
    aclopAttr* reduce_attr = aclopCreateAttr();
    int64_t reduce_axes[1] = {axis};
    ACL_THROW_IF(aclopSetAttrListInt(reduce_attr, "axes", reduce_axes, 1));
    ACL_THROW_IF(aclopSetAttrBool(reduce_attr, "keep_dims", true));

    for (int i=0;i<warmup;++i) {
        exec_single("Mul", {&x, &x}, {&x2}, nullptr, stream);            // x^2
        exec_single("ReduceMean", {&x2}, {&mean}, reduce_attr, stream);  // mean(x^2), keep dim
        exec_single("Add", {&mean, &eps_c}, {&den}, nullptr, stream);    // + eps
        exec_single("Sqrt", {&den}, {&den}, nullptr, stream);            // sqrt(mean+eps) (in-place ok)
        // Broadcast den to full shape by multiplying with 1
        // Use Div(x, den_broadcast)
        exec_single("Div", {&x, &den}, {&y}, nullptr, stream);
        if (with_gamma) {
            // Broadcast gamma to full shape then mul
            exec_single("Mul", {&y, &gamma}, {&y}, nullptr, stream);
        }
    }
    check_stream_sync("rmsnorm-warmup", stream);

    std::vector<double> times_ms;
    for (int i=0;i<repeat;++i) {
        double t0 = now_ms();
        exec_single("Mul", {&x, &x}, {&x2}, nullptr, stream);            // x^2
        exec_single("ReduceMean", {&x2}, {&mean}, reduce_attr, stream);  // mean(x^2), keep dim
        exec_single("Add", {&mean, &eps_c}, {&den}, nullptr, stream);    // + eps
        exec_single("Sqrt", {&den}, {&den}, nullptr, stream);            // sqrt(mean+eps)
        exec_single("Div", {&x, &den}, {&y}, nullptr, stream);
        if (with_gamma) {
            exec_single("Mul", {&y, &gamma}, {&y}, nullptr, stream);
        }
        check_stream_sync("rmsnorm", stream);
        double t1 = now_ms();
        times_ms.push_back(t1 - t0);
    }
    auto avg = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    std::nth_element(times_ms.begin(), times_ms.begin()+times_ms.size()/2, times_ms.end());
    double p50 = times_ms[times_ms.size()/2];
    std::cout << "[RESULT] op=RMSNorm(dtype=" << (dtype==ACL_FLOAT?"fp32":"fp16")
              << ", eps=" << eps << ", gamma=" << (with_gamma?1:0) << ") shape=";
    for (size_t i=0;i<dims.size();++i) std::cout << (i? "x":"") << dims[i];
    std::cout << " repeat=" << repeat << " avg_ms=" << avg << " p50_ms=" << p50 << std::endl;

    aclopDestroyAttr(reduce_attr);
    if (with_gamma) destroy_tensor(gamma);
    destroy_tensor(y);
    destroy_tensor(den);
    destroy_tensor(eps_c);
    destroy_tensor(mean);
    destroy_tensor(x2);
    destroy_tensor(x);
}

// RMSNorm + BatchMatMul: input [B,S,H] -> norm -> [B,S,H] times weight [H,O] => [B,S,O].
static void run_rmsnorm_gemm(const std::vector<int64_t>& bsh, int64_t out_dim, int repeat, int warmup,
                             aclDataType dtype, double eps, bool with_gamma, bool transpose_w,
                             aclrtStream stream)
{
    if (bsh.size() < 2) { std::cerr << "RMSNorm+GEMM expects at least 2D [*, H]\n"; std::exit(1); }
    int64_t H = bsh.back();
    std::vector<int64_t> yshape = bsh; yshape.back() = out_dim;

    // Tensors
    DeviceTensor x  = make_tensor(bsh, dtype);
    DeviceTensor x2 = make_tensor(bsh, dtype);
    std::vector<int64_t> mean_dims = bsh; mean_dims.back() = 1;
    DeviceTensor mean = make_tensor(mean_dims, dtype);
    DeviceTensor eps_c = make_scalar(eps, dtype);
    DeviceTensor den = make_tensor(mean_dims, dtype);
    DeviceTensor den_b = make_tensor(bsh, dtype);
    DeviceTensor y_norm = make_tensor(bsh, dtype);
    DeviceTensor gamma;
    if (with_gamma) gamma = make_tensor({H}, dtype);
    // Weight: [H,O] (or [O,H] if transpose), broadcast to batch dims for BatchMatMul
    std::vector<int64_t> w_dims = transpose_w ? std::vector<int64_t>{out_dim, H} : std::vector<int64_t>{H, out_dim};
    DeviceTensor W = make_tensor(w_dims, dtype);
    // For BatchMatMul, shape must be [..., M, K] x [..., K, N]; we'll expand dims to [B,S,M,K] and [1,1,K,N]
    std::vector<int64_t> a_dims = bsh; a_dims.push_back(1); // incorrect; we need explicit 2D at end
    // Actually a: [..., M, K] where M=S and K=H if we flatten B,S? Let’s use [..., S, H] with B as batch
    // y_norm already has dims [B,S,H] where M=S and K=H
    // W_b: [1,1,H,O]  so it broadcasts across [B,S].
    DeviceTensor W_b = make_tensor({1,1,H,out_dim}, dtype);
    // We need to copy W into the last two dims of W_b; For benchmarking only, content doesn't matter, so skip copy.

    // attrs
    aclopAttr* reduce_attr = aclopCreateAttr();
    int64_t axis = (int64_t)bsh.size()-1;
    int64_t reduce_axes[1] = {axis};
    ACL_THROW_IF(aclopSetAttrListInt(reduce_attr, "axes", reduce_axes, 1));
    ACL_THROW_IF(aclopSetAttrBool(reduce_attr, "keep_dims", true));

    aclopAttr* bmm_attr = aclopCreateAttr();
    // adj_x1/adj_x2 default false; we won't set unless needed.

    // Warmup: RMSNorm then BatchMatMul
    for (int i=0;i<warmup;++i) {
        exec_single("Mul", {&x, &x}, {&x2}, nullptr, stream);
        exec_single("ReduceMean", {&x2}, {&mean}, reduce_attr, stream);
        exec_single("Add", {&mean, &eps_c}, {&den}, nullptr, stream);
        exec_single("Sqrt", {&den}, {&den}, nullptr, stream);
        exec_single("Div", {&x, &den}, {&y_norm}, nullptr, stream);
        if (with_gamma) {
            exec_single("Mul", {&y_norm, &gamma}, {&y_norm}, nullptr, stream);
        }
        // BatchMatMul: [B,S,H] x [1,1,H,O] -> [B,S,O]
        DeviceTensor y_out = make_tensor(yshape, dtype);
        exec_single("BatchMatMul", {&y_norm, &W_b}, {&y_out}, bmm_attr, stream);
        destroy_tensor(y_out);
    }
    check_stream_sync("rmsnorm_gemm-warmup", stream);

    std::vector<double> times_ms;
    for (int i=0;i<repeat;++i) {
        double t0 = now_ms();
        exec_single("Mul", {&x, &x}, {&x2}, nullptr, stream);
        exec_single("ReduceMean", {&x2}, {&mean}, reduce_attr, stream);
        exec_single("Add", {&mean, &eps_c}, {&den}, nullptr, stream);
        exec_single("Sqrt", {&den}, {&den}, nullptr, stream);
        exec_single("Div", {&x, &den}, {&y_norm}, nullptr, stream);
        if (with_gamma) {
            exec_single("Mul", {&y_norm, &gamma}, {&y_norm}, nullptr, stream);
        }
        DeviceTensor y_out = make_tensor(yshape, dtype);
        exec_single("BatchMatMul", {&y_norm, &W_b}, {&y_out}, bmm_attr, stream);
        check_stream_sync("rmsnorm_gemm", stream);
        double t1 = now_ms();
        times_ms.push_back(t1 - t0);
        destroy_tensor(y_out);
    }

    auto avg = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    std::nth_element(times_ms.begin(), times_ms.begin()+times_ms.size()/2, times_ms.end());
    double p50 = times_ms[times_ms.size()/2];
    std::cout << "[RESULT] op=RMSNorm+BatchMatMul dtype=" << (dtype==ACL_FLOAT?"fp32":"fp16")
              << " in_shape=";
    for (size_t i=0;i<bsh.size();++i) std::cout << (i? "x":"") << bsh[i];
    std::cout << " out_dim=" << out_dim
              << " repeat=" << repeat << " avg_ms=" << avg << " p50_ms=" << p50 << std::endl;

    // cleanup
    aclopDestroyAttr(bmm_attr);
    aclopDestroyAttr(reduce_attr);
    destroy_tensor(W_b);
    destroy_tensor(W);
    if (with_gamma) destroy_tensor(gamma);
    destroy_tensor(y_norm);
    destroy_tensor(den);
    destroy_tensor(eps_c);
    destroy_tensor(mean);
    destroy_tensor(x2);
    destroy_tensor(x);
}

int main() {
    // Basic init
    ACL_THROW_IF(aclInit(nullptr));
    int32_t deviceId = (int32_t)getenv_ll("DEVICE_ID", 0);
    ACL_THROW_IF(aclrtSetDevice(deviceId));
    aclrtContext ctx;
    ACL_THROW_IF(aclrtCreateContext(&ctx, deviceId));
    aclrtStream stream;
    ACL_THROW_IF(aclrtCreateStream(&stream));

    std::string op = getenv_str("OP", "gelu");
    std::transform(op.begin(), op.end(), op.begin(), ::tolower);
    std::string dims_str = getenv_str("DIMS", "");
    int repeat = (int)getenv_ll("REPEAT", 5);
    int warmup = (int)getenv_ll("WARMUP", 1);
    double eps = getenv_double("EPS", 1e-6);
    bool with_gamma = getenv_bool("GAMMA", false);
    bool transpose_w = getenv_bool("TRANSPOSE_W", false);
    std::vector<int64_t> dims = parse_shape(dims_str);
    aclDataType dtype = acl_dtype_from_str(dtype_from_env());

    if (op=="gelu") {
        if (dims.empty()) dims = {1, 2048, 2048};
        run_gelu(dims, repeat, warmup, dtype, stream);
    } else if (op=="rmsnorm") {
        if (dims.empty()) dims = {1, 2048, 4096};
        run_rmsnorm_chain(dims, repeat, warmup, dtype, eps, with_gamma, stream);
    } else if (op=="rmsnorm_gemm") {
        if (dims.empty()) dims = {1, 2048, 4096};
        int64_t out_dim = getenv_ll("OUT", dims.back()); // default same as H
        run_rmsnorm_gemm(dims, out_dim, repeat, warmup, dtype, eps, with_gamma, transpose_w, stream);
    } else {
        std::cerr << "Unknown OP=" << op << ", expected: gelu | rmsnorm | rmsnorm_gemm\n";
        return 2;
    }

    ACL_THROW_IF(aclrtDestroyStream(stream));
    ACL_THROW_IF(aclrtDestroyContext(ctx));
    ACL_THROW_IF(aclrtResetDevice(deviceId));
    ACL_THROW_IF(aclFinalize());
    return 0;
}
