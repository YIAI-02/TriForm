# config.py
import logging

# =========================
# Hybrid scheduling params
# =========================
ALLOW_HYBRID: bool = False    # allow hybrid scheduling (some ops on NPU, some on PIM)
# =========================
# Rank-U weight load option
# =========================
RANKU_INCLUDE_AVG_WEIGHT_LOAD: bool = True

# =========================
# Weight storage & formats
# =========================

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
    "cpu": 25.0,
    "npu": 120.0,
    "pim": 60.0,
    "default": 25.0,
}

# data format cold start
FORMAT_CONV_OVERHEAD_US = {
    "cpu": 1,
    "npu": 0.5,
    "pim": 0.8,
    "default": 1,
}

# “latency = transfer + convert” serial or overlap (0~1)
NONOVERLAP_TIME = 1.0

# PIM 频率（GHz）：cycles / (PIM_FREQ_GHZ * 1e9) = seconds
PIM_FREQ_GHZ: float = 1.0
GB_FREQ_GHZ: float = 1.0

# PIM 容量分配系数
PIM_STATIC_ALLOC_RATIO: float = 0.9
PIM_RUNTIME_LRU_THRESHOLD: float = 0.95

# =========================
# PIM weight preload control
# =========================
ENABLE_PIM_WEIGHT_PRELOAD: bool = False
# =========================
# Operator device constraints
# =========================
OPERATOR_DEVICE_ALLOWED = {
    "Q":       {"cpu": True, "npu": True, "pim": True},
    "K":       {"cpu": True, "npu": True, "pim": True},
    "V":       {"cpu": True, "npu": True, "pim": True},
    "O":       {"cpu": True, "npu": True, "pim": True},
    "FFN_W1":  {"cpu": True, "npu": True, "pim": True},
    "FFN_W2":  {"cpu": True, "npu": True, "pim": True},
    "FFN_W3":  {"cpu": True, "npu": True, "pim": True},

    "QK":      {"cpu": True, "npu": True, "pim": True},
    "SV":      {"cpu": True, "npu": True, "pim": True},
    "SOFTMAX": {"cpu": True, "npu": True, "pim": True},

    "ADD":     {"cpu": True, "npu": True, "pim": True},
    "LN":      {"cpu": True, "npu": True, "pim": True},
    "SWIGLU":  {"cpu": True, "npu": True, "pim": True},
    "GELU":    {"cpu": True, "npu": True, "pim": True},
    "ACT":     {"cpu": True, "npu": True, "pim": True},

    "IDENTITY": {"cpu": True, "npu": True, "pim": True},
    "K_WRITE": {"cpu": True, "npu": True, "pim": True},
    "V_WRITE": {"cpu": True, "npu": True, "pim": True},
    "ALLREDUCE": {"cpu": True, "npu": True, "pim": True},
}

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
SCHED_DEFAULT: str = "heft"  

# JointGraphScheduler rolling-window lookahead (receding horizon)
# Used by JointGraphScheduler.schedule_joint() to reduce multi-hop communication thrashing
# in large unrolled decode graphs.
SCHED_JOINT_LK_ENABLE: bool = True
SCHED_JOINT_LK_H: int = 3
SCHED_JOINT_LK_GAMMA: float = 0.2
SCHED_JOINT_LK_CONSIST_LAMBDA: float = 0
SCHED_JOINT_LK_PLAN_HINT_MAX: int =  3
# Weight-reuse bias gain multiplier (eta in bias formula)
SCHED_WEIGHT_BIAS_ETA: float = 100


# -------------------------------------------------------------------------------------------------
# Peak compute utilization model 
# -------------------------------------------------------------------------------------------------
CPU_FALLBACK_TFLOPS = 1e-3  # 1 GFLOP/s
COMPUTE_UTILIZATION = {
    'default': 0.7,
    'npu': {
        'enabled': True,
        'curve': 'sigmoid',
        'min_util': 0.05,
        'max_util': 0.2,
        'flops_low': 5e7,     # <= 0.5 GFLOPs -> near min_util
        'flops_high': 5e12,   # >= 5TFLOPs -> near max_util
        'knee_flops': 1.5e11,  #defualt sqrt(low*high) ≈ 1.58e10
        'slope': 3.0,
    },

}
# -------------------------------------------------------------------------------------------------
# NPU/GPU kernel launch (software/runtime) overhead model
# -------------------------------------------------------------------------------------------------
KERNEL_LAUNCH_OVERHEAD = {
    # Global on/off
    'enabled': True,
    'apply_backends': ['fast'],

    'phase_scale': {
        'prefill': 0.1,
        'decode': 1.0,
    },
    'scale_by_time_scale': False,
    'default_us': 0.0,

    'by_category_us': {
        'norm': 70.5,
        'softmax': 18.4,
        'activation': 14.2,
        'elem': 13.7,
        'gemm': 24.0,
    },

    'by_op_us': {
        'score': 42.0,   # QK
        'output': 28.1,  # SV
        'ffn_up': 21.1,     # ffn_w3
        'ffn_gate': 24.1,   # ffn_w1
        'ffn_down': 22.5,   # ffn_w2
    },
}
