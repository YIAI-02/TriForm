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
    "npu": 409.6,
    "pim": 8192
}

# data format cold start
FORMAT_CONV_OVERHEAD_US = {
    "cpu": 1,
    "npu": 0.0,
    "pim": 0.0
}

# “latency = transfer + convert” serial or overlap (0~1)
NONOVERLAP_TIME = 0.5

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
SCHED_JOINT_LK_CONSIST_LAMBDA: float = 8
SCHED_JOINT_LK_PLAN_HINT_MAX: int =  3
# Weight-reuse bias gain multiplier (eta in bias formula)
SCHED_WEIGHT_BIAS_ETA: float = 50


# -------------------------------------------------------------------------------------------------
# Peak compute utilization model 
# -------------------------------------------------------------------------------------------------
CPU_FALLBACK_TFLOPS = 1e-3  # 1 GFLOP/s
COMPUTE_UTILIZATION = {
    # Device-name based overrides (match prefix in hardware.json "name")
    #   - name="Ascend_910B_NPU0" -> key "Ascend_910B"
    #   - name="A100_GPU0"        -> key "A100"
    'by_device_name': {
        # TODO: tune these for your CUDA stack / kernels.
        'Ascend_910B': {
            'enabled': True,
            'curve': 'sigmoid',
            'min_util': 0.3,
            'max_util': 1.0,
            'flops_low': 5e7,
            'flops_high': 5e12,
            'knee_flops': 1.0e10,
            'slope': 5.0,
        },

        # NVIDIA A100
        'A100': {
            'enabled': True,
            'curve': 'sigmoid',
            'min_util': 0.4,
            'max_util': 0.9,
            'flops_low': 5e7,
            'flops_high': 5e12,
            'knee_flops': 1.5e8,
            'slope': 6.0,
        },

        'pim': {
            'enabled': True,
            'curve': 'sigmoid',
            'min_util': 0.3,
            'max_util': 0.4,
            'flops_low': 5e7,
            'flops_high': 5e12,
            'knee_flops': 1.5e8,
            'slope': 6.0,
        },
    },
}



# -------------------------------------------------------------------------------------------------
# NPU/GPU kernel launch (software/runtime) overhead model
# -------------------------------------------------------------------------------------------------
KERNEL_LAUNCH_OVERHEAD = {
    # Device-name based overrides (match prefix in hardware.json "name")
    #   - name="Ascend_910B_NPU0" -> key "Ascend_910B"
    #   - name="A100_GPU0"        -> key "A100"
    'by_device_name': {
        'Ascend_910B': {
            'enabled': True,
            'apply_backends': ['fast'],
            'phase_scale': {
                'prefill': 0.5,
                'decode': 1.0,
            },
            'scale_by_time_scale': False,
            'default_us': 0.0,
            'by_category_us': {
                'norm': 3.0,
                'softmax': 0.25,
                'activation': 3.0,
                'elem': 3.0,
                'gemm': 4.0,
            },
            'by_op_us': {
                'ln': 3.0,
                'gelu': 3.0,
                'softmax': 0.25,
                'q_proj': 4.0,
            },
        },

        'A100': {
            'enabled': True,
            'apply_backends': ['fast'],
            'phase_scale': {
                'prefill': 1.0,
                'decode': 1.0,
            },

            'scale_by_time_scale': False,
            'default_us': 0.0,
            'by_category_us': {
                'norm': 54.0,        
                'softmax': 11.6,
                'activation': 19.5,
                'elem': 29.0,
                'gemm': 22.0,
            },

            'by_op_us': {
                'score': 37.8,
                'output': 22.0, 
                'ffn_up': 27.3,
                'ffn_gate': 29.5,
                'ffn_down': 26.6,
                'add': 15.5,
                'swiglu': 18.7,
            },
        },
        
        'Aim PIM': {
            'enabled': True,
            'apply_backends': ['fast'],
            'phase_scale': {
                'prefill': 1.0,
                'decode': 1.0,
            },
            'scale_by_time_scale': False,

            'default_us': 0.0,
            'by_category_us': {
                'norm': 0.0,
                'softmax': 0.0,
                'activation': 0.0,
                'elem': 0.0,
                'gemm': 0.0,
            },
            'by_op_us': {
                'ln': 0.0,
                'gelu': 0.0,
                'softmax': 0.0,
                'q_proj': 0.0,
            },
        },
    },
}

