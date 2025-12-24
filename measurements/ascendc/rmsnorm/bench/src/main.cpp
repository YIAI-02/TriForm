#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <tuple>
#include <iostream>
#include <sstream>
#include <chrono>
#include "rmsnorm_runner.h"

// Simple helpers
static inline std::string getenv_str(const char* k, const std::string& defv="") {
    const char* v = std::getenv(k);
    return v ? std::string(v) : defv;
}
static inline int getenv_int(const char* k, int defv) {
    const char* v = std::getenv(k);
    return v ? std::atoi(v) : defv;
}
static inline float getenv_float(const char* k, float defv) {
    const char* v = std::getenv(k);
    return v ? std::atof(v) : defv;
}
static inline bool getenv_bool(const char* k, bool defv=false) {
    const char* v = std::getenv(k);
    if (!v) return defv;
    return std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE";
}

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out; std::stringstream ss(s); std::string item;
    while (std::getline(ss, item, delim)) if (!item.empty()) out.push_back(item);
    return out;
}
static void parse_shape(const std::string& shape, int& B, int& S, int& H) {
    // accept delimiters x, X, *
    std::string t = shape; 
    for (char& c : t) c = (c=='X' ? 'x' : c);
    for (char& c : t) if (c=='*') c='x';
    auto part = split(t, 'x');
    if (part.size() != 3) {
        throw std::runtime_error("Bad shape: " + shape + ", expecting BxSxH");
    }
    B = std::stoi(part[0]); S = std::stoi(part[1]); H = std::stoi(part[2]);
}

int main() {
    const std::string cases = getenv_str("CASES");  // "B×S×H,B×S×H,..."
    int B = getenv_int("B", 1);
    int S = getenv_int("S", 1);
    int H = getenv_int("H", 2048);
    const int repeat = getenv_int("REPEAT", 5);
    const std::string dtype = getenv_str("DTYPE", "fp16"); // fp16|fp32
    const float eps = getenv_float("EPS", 1e-5f);
    const bool no_io = getenv_bool("NO_IO", true);

    std::vector<std::tuple<int,int,int>> shapes;
    if (!cases.empty()) {
        for (auto& s : split(cases, ',')) {
            int b,sq,h; parse_shape(s, b, sq, h);
            shapes.emplace_back(b, sq, h);
        }
    } else {
        shapes.emplace_back(B, S, H);
    }

    // Warmup hint for msprof: some users do 1 warmup before profiling.
    int warmup = getenv_int("WARMUP", 1);

    for (auto [b,sq,h] : shapes) {
        if (!no_io) {
            std::cout << "[INFO] Shape " << b << "x" << sq << "x" << h 
                      << " dtype=" << dtype << " repeat=" << repeat << " eps=" << eps << std::endl;
        }
        RmsnormConfig cfg;
        cfg.B = b; cfg.S = sq; cfg.H = h;
        cfg.dtype = (dtype=="fp32" ? RmsnormConfig::FP32 : RmsnormConfig::FP16);
        cfg.eps = eps;
        cfg.verbose = !no_io;

        RmsnormRunner runner;
        if (!runner.Init(cfg)) {
            std::cerr << "[ERROR] Init failed for shape " << b << "x" << sq << "x" << h << std::endl;
            return 2;
        }

        // Warmup (not profiled meaningfully, but msprof may still see it; keep it small)
        for (int i=0;i<warmup;i++) {
            if (!runner.RunOnce()) {
                std::cerr << "[ERROR] Warmup run failed.\n";
                return 3;
            }
        }

        // Timed runs (host timing for visibility; use msprof for kernel timing)
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i=0;i<repeat;i++) {
            if (!runner.RunOnce()) {
                std::cerr << "[ERROR] Run " << i << " failed.\n";
                return 4;
            }
        }
        runner.SyncDevice();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1-t0).count() / repeat;

        // Emit a single-line summary that's easy to grep.
        std::cout << "RMSNORM_SHAPE=" << b << "x" << sq << "x" << h
                  << " DTYPE=" << dtype
                  << " REPEAT=" << repeat
                  << " HOST_AVG_MS=" << ms << std::endl;
    }
    return 0;
}
