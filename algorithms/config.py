# config.py (AttAcc-calibrated interconnect + memory hierarchy)

# ---------- Hardware pool ----------
NPU_COUNT: int = 1
PIM_COUNT: int = 1

# ---------- Datatype sizes ----------
ACTIVATION_BYTES: int = 2
WEIGHT_BYTES: int = 2

# ---------- Global buffer / HOST DRAM ----------
GLOBAL_BUFFER_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GiB
ALLOW_EVICTION: bool = True
ENFORCE_AT_LEAST_ONE_COPY: bool = True  # 至少保留一种格式，但不强制 ND

# ---------- Formats ----------
FORMATS = ["ND", "NPU_OPT", "PIM_OPT"]
FORMAT_SIZE_MULTIPLIER = {"ND": 1.0, "NPU_OPT": 1.0, "PIM_OPT": 1.0}
FORMAT_CONVERSION_PER_BYTE = {
    ("ND","ND"):0.0, ("ND","NPU_OPT"):1e-10, ("ND","PIM_OPT"):1.5e-10,
    ("NPU_OPT","ND"):1e-10, ("NPU_OPT","NPU_OPT"):0.0, ("NPU_OPT","PIM_OPT"):2e-10,
    ("PIM_OPT","ND"):1.5e-10, ("PIM_OPT","NPU_OPT"):2e-10, ("PIM_OPT","PIM_OPT"):0.0
}
DEVICE_PREFERRED_FORMAT = {"NPU":"NPU_OPT", "PIM":"PIM_OPT"}

# ---------- Interconnect & memory hierarchy ----------
INTRA_DEVICE_PASS_BW = {"NPU": 900e9, "PIM": 670.4e9}  # 同设备本地交接有效带宽

GB_TO_DEVICE_BW = {("GB","NPU"): 512e9, ("GB","PIM"): 512e9}
DEVICE_TO_GB_BW = {("NPU","GB"): 512e9, ("PIM","GB"): 512e9}

INTERCONNECT_MODE = "NVLINK3"  # "NVLINK3" | "PCIE5x16" | "PCIE4x16" | "NONE"
DIRECT_INTERCONNECTS = {
    "NVLINK3":  {("NPU","PIM"):600e9, ("PIM","NPU"):600e9},
    "PCIE5x16": {("NPU","PIM"): 64e9,  ("PIM","NPU"): 64e9},
    "PCIE4x16": {("NPU","PIM"): 32e9,  ("PIM","NPU"): 32e9},
    "NONE": {}
}
# 可自定义额外直连（如 NPU↔NPU）
DEVICE_TO_DEVICE_BW = {}

# ---------- Compute ----------
DEVICE_PEAK_THROUGHPUT = {"NPU": 200e12, "PIM": 50e12}
DEVICE_LAUNCH_OVERHEAD = {"NPU": 8e-6, "PIM": 5e-6}

# ---------- HEFT ----------
RANK_COST_POLICY = "avg"
ALLOW_HYBRID = True
HYBRID_SPEEDUP = 1.6

# ---------- Objective ----------
OBJECTIVE = "balanced"     # "latency" | "memory" | "balanced"
BALANCED_ALPHA = 0.8

# —— weight_format scheduler ——（greedy / dp / beam）
FORMAT_PLANNER_ALGO = "beam"   # "greedy" | "dp" | "beam"
FORMAT_PLANNER_BEAM_WIDTH = 24   # algo="beam" 
FORMAT_PLANNER_DP_MAX_UNITS = 4096  # algo="dp" 时的最大状态数（内存限制）
# 把字节折算成“等效秒”的惩罚（balanced模式下使用；latency=0；memory=直接返回基线）
FORMAT_PLANNER_MEMORY_BYTE_PENALTY = 1e-9  # 默认 1e-9 s/B：≈ 1 GB ⇔ 1 秒

# ---------- Logging ----------
LOG_LEVEL = "INFO"
ENABLE_WEIGHT_PREFETCH = False


# ---------- Placement constraints ----------
OP_PLACEMENT_CONSTRAINTS = {
    # examples
    # "attention": {"allow_npu": True, "allow_pim": True, "allow_hybrid": True},
    # "qk_matmul": {"allow_npu": True, "allow_pim": False, "allow_hybrid": False},
}
NODE_PLACEMENT_CONSTRAINTS = {
    # examples
    # "L3:FFN": {"allow_npu": True, "allow_pim": False, "allow_hybrid": False},
}

# ---------- Hybrid gating & hysteresis ----------
# 当 NPU-only 与 PIM-only 完成时间差距“悬殊”时，禁止 Hybrid；否则才考虑 Hybrid 候选
HYBRID_GATE_BY_DIFF = True
HYBRID_RELATIVE_DIFF = 0.15       # 相对差阈值（如 15%）
HYBRID_ABSOLUTE_MARGIN = 5e-4     # 绝对差阈值（秒），用于过滤极小算子的抖动

# 滞回窗开关与阈值（进入更严格，退出更宽松）
HYSTERESIS_ENABLE = True
HYST_REL_ENTER = 0.10
HYST_ABS_ENTER = 2e-4
HYST_REL_EXIT  = 0.20
HYST_ABS_EXIT  = 6e-4
