# config.py

# =========================
# Default model/run config
# =========================
DEFAULT_CONFIG = {
    "model_family": "llama",       # or "mpt", "palm" ...
    "model_variant": "7b",
    "dtype": "fp16",               # "fp16" | "bf16" | "fp32" | "int8" | "fp8"
    "batch": 1,
    "prefill_len": 128,
    "decode_len": 32,
    "decode_mode": "mul",          # kept for backward compat; we now simulate progressive decode
}

# =========================
# Hybrid scheduling params
# =========================
ALLOW_HYBRID: bool = True
HYBRID_GATE_BY_DIFF: bool = True
HYBRID_RELATIVE_DIFF: float = 0.05    # <=5% diff
HYBRID_ABSOLUTE_MARGIN: float = 1e-3  # <=1 ms

HYSTERESIS_ENABLE: bool = True
HYST_REL_ENTER: float = 0.03          # <3%
HYST_ABS_ENTER: float = 5e-4          # <0.5 ms
HYST_REL_EXIT: float  = 0.10          # >10%
HYST_ABS_EXIT: float  = 2e-3          # >2 ms

# =========================
# Rank-U weight load option
# =========================
RANKU_INCLUDE_AVG_WEIGHT_LOAD: bool = True

# =========================
# Weight storage & formats
# =========================
ENABLE_TWO_PASS_FORMAT_TUNING: bool = True
WEIGHT_FORMAT_JSON_PATH: str = "./format_tuning/weight_storage_suggestion.json"

# Progressive multi-pass tuning (iterate until converged or reach max passes)
FORMAT_TUNING_MAX_PASSES: int = 1
# stop when |Δtime| <= TIME_EPS AND mapping_change_ratio <= MAP_EPS
FORMAT_TUNING_TIME_EPS: float = 1e-4      # 0.1 ms
FORMAT_TUNING_MAP_EPS: float  = 0.01      # <=1% of weights changed

# Host (weights live in "main memory")
HOST_NAME: str = "CPU0"

# Preferred on-device formats
DEVICE_PREFERRED_FORMAT = {
    "cpu": "ND",
    "npu": "NPU_OPT",
    "pim": "PIM_OPT",
}

# Format size multipliers (alignment/packing overhead modeling)
FORMAT_SIZE_MULTIPLIER = {
    "ND": 1.0,
    "NPU_OPT": 1.0,
    "PIM_OPT": 1.0,
}

# Format conversion bandwidth (GB/s) per device type
FORMAT_CONV_BW_GBs = {
    "cpu": 50.0,
    "npu": 200.0,
    "pim": 100.0,
    "default": 50.0,
}

# =========================
# PIM formula-based latency
# =========================
# 尝试依次从这些路径读取 PIM 公式 JSON（任意一个存在即可）
PIM_FORMULA_PATHS = [
    "../measurements/pim/out_run/results/model_fit_summary.json",
    "../measurements/pim/out_run/results/model_formula.json",
]
# PIM 频率（GHz）：cycles / (PIM_FREQ_GHZ * 1e9) = seconds
PIM_FREQ_GHZ: float = 1.0