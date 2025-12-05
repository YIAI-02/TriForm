# config.py
import logging
# =========================
# Default model/run config
# =========================
DEFAULT_CONFIG = {
    "model_family": "llama",
    "model_variant": "7b",
    "dtype": "fp16",
    "batch": 1,
    "prefill_len": 128,
    "decode_len": 32,
    "decode_sample_stride": 32,   # decode 采样步长
    "pim_config_path":"./aim_simulator/pim.json",
    "gb_config_path":"./aim_simulator/gb.json",
    "ramulator_config_path":"./aim_simulator/example.yaml"
}

# =========================
# Fast mode (disable trace simulations)
# =========================
# When FAST_MODE is True, all trace simulations are disabled and only estimated costs are used
FAST_MODE: bool = True

# =========================
# Hybrid scheduling params
# =========================
ALLOW_HYBRID: bool = False  # allow hybrid scheduling (some ops on NPU, some on PIM)
# =========================
# Rank-U weight load option
# =========================
RANKU_INCLUDE_AVG_WEIGHT_LOAD: bool = True

# =========================
# Weight storage & formats
# =========================

# Progressive multi-pass tuning (iterate until converged or reach max passes)
ENABLE_TWO_PASS_FORMAT_TUNING: bool = True
FORMAT_TUNING_MAX_PASSES: int = 30
FORMAT_TUNING_TIME_EPS: float = 1e-4      # 0.1 ms
FORMAT_TUNING_MAP_EPS: float  = 0.02      # <=1% of weights changed

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

#TODO
# Format conversion bandwidth (GB/s) per device type
FORMAT_CONV_BW_GBs = {
    "cpu": 50.0,
    "npu": 200.0,
    "pim": 100.0,
    "default": 50.0,
}
NONOVERLAP_TIME: float = 0.3  # 0.0 means fully overlapped, 1.0 means non-overlapped

# PIM 频率（GHz）：cycles / (PIM_FREQ_GHZ * 1e9) = seconds
PIM_FREQ_GHZ: float = 1.0
GB_FREQ_GHZ: float = 1.0

# PIM 容量分配系数
PIM_STATIC_ALLOC_RATIO: float = 0.5
PIM_RUNTIME_LRU_THRESHOLD: float = 0.9
# =========================
# Operator device constraints
# =========================
# 定义哪些算子类型可以在哪些设备上运行
# True 表示允许，False 表示禁止
OPERATOR_DEVICE_ALLOWED = {
    # 1) 线性 / GEMM：PIMA / PIMD 都支持
    "Q":       {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "K":       {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "V":       {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "O":       {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "FFN_W1":  {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "FFN_W2":  {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "FFN_W3":  {"cpu": True, "npu": True, "pima": True, "pimd": True},

    # 2) Attention 里的 matmul：
    "QK":      {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "SV":      {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "Softmax": {"cpu": True, "npu": True, "pima": True, "pimd": True},

    # 3) 逐元素：
    "Add":     {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "LN":      {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "SwiGLU":  {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "GELU":    {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "Act":     {"cpu": True, "npu": True, "pima": True, "pimd": True},

    # 4) 其它
    "Identity": {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "KV_read":  {"cpu": True, "npu": True, "pima": True, "pimd": True},
    "KV_write": {"cpu": True, "npu": True, "pima": True, "pimd": True},
}


# 默认策略：如果算子不在上面的列表中，使用这个
DEFAULT_OPERATOR_ALLOWED = {"cpu": True, "npu": True, "pim": False}


# =========================
# Simulated Annealing (optional)
# =========================
# SA_ENABLE = False greedy search only
SA_ENABLE: bool = True          # turn off to disable SA
SA_T0: float = 1.0              # initial temperature
SA_ALPHA: float = 0.85          # cooling rate per pass
SA_FLIP_PROB: float = 0.15      # per-weight flip prob when proposing neighbor

# Global debug switch reflecting the --debug CLI flag
DEBUG_GLOBAL: bool = False

# Logging filter that enforces: only emit DEBUG records when both
# (1) global debug is enabled, and (2) the module's local flag (if provided) is True.
class LocalDebugFilter(logging.Filter):
    def __init__(self, get_local_flag=None):
        super().__init__()
        self.get_local_flag = get_local_flag
    def filter(self, record):
        # Always allow INFO and above
        if record.levelno >= logging.INFO:
            return True
        try:
            from config import DEBUG_GLOBAL
        except Exception:
            DEBUG_GLOBAL = False
        local_ok = True
        if self.get_local_flag is not None:
            try:
                local_ok = bool(self.get_local_flag())
            except Exception:
                local_ok = False
        return bool(DEBUG_GLOBAL) and bool(local_ok)

def attach_local_debug_filter(logger: "logging.Logger", get_local_flag=None):
    """Attach the LocalDebugFilter to a module logger."""
    logger.addFilter(LocalDebugFilter(get_local_flag))

# --- debug logging setup (auto-added) ---
def setup_logging(debug: bool, log_file: str = "./output/debug_log.txt"):
    """
    Configure root logging. When debug is True, logs go to console and a fixed text file.
    Otherwise, silence debug logs by raising level to CRITICAL.
    Also updates config.DEBUG_GLOBAL to reflect CLI flag.
    """
    import logging as _logging
    global DEBUG_GLOBAL
    DEBUG_GLOBAL = bool(debug)

    logger = _logging.getLogger()
    # Remove any pre-existing handlers to avoid duplication
    for h in list(logger.handlers):
        logger.removeHandler(h)
    if debug:
        logger.setLevel(_logging.DEBUG)
        formatter = _logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        # File handler
        fh = _logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(_logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # Console handler
        ch = _logging.StreamHandler()
        ch.setLevel(_logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        # Silence logs by raising the threshold
        logger.setLevel(_logging.CRITICAL)
# --- end debug logging setup ---



# =========================
# Scheduler algorithm controls
# =========================
SCHED_DEFAULT: str = "heft"   # one of: heft | sa | ga | rl | astar
# SA for scheduler (not the separate weight-format SA in this project)
SCHED_SA_ITERS: int = 120
SCHED_SA_T0: float = 1.0
SCHED_SA_ALPHA: float = 0.90
SCHED_SA_FLIP_PROB: float = 0.12

# GA
SCHED_GA_POP: int = 24
SCHED_GA_GENS: int = 40
SCHED_GA_ELITE: int = 2
SCHED_GA_MUT_PROB: float = 0.08
SCHED_GA_CROSS_PROB: float = 0.50

# RL
SCHED_RL_EPISODES: int = 30
SCHED_RL_EPS0: float = 0.30
SCHED_RL_EPSE: float = 0.05
SCHED_RL_ALPHA: float = 0.30
SCHED_RL_GAMMA: float = 0.90

# A* (beam)
SCHED_ASTAR_BEAM: int = 6
SCHED_ASTAR_MAX_EXPANSIONS: int = 200
SCHED_HEFT_LK_DEPTH: int = 3
