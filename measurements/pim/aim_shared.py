#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re, sys, json
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List

# ----------------------- 特征表（统一来源） -----------------------
FEATURE_SPECS = [
    # name, has_opsize, regex patterns（兼容多种书写）
    ("MAC_ABK",     True,  [r"^AiM\s+MAC_ABK\s+(\d+)"]),
    ("MAC_BK_BK",   True,  [r"^AiM\s+MAC_BK_BK\s+(\d+)"]),
    ("MAC_BK_GB",   True,  [r"^AiM\s+MAC_BK_GB\s+(\d+)"]),
    ("WR_GB",       True,  [r"^AiM\s+WR_GB\s+(\d+)"]),
    ("RD_AB",       True,  [r"^AiM\s+RD_AB\s+(\d+)"]),
    ("RD_GB",       True,  [r"^AiM\s+RD_GB\s+(\d+)"]),
    ("WR_AB",       True,  [r"^AiM\s+WR_AB\s+(\d+)"]),
    ("RD_AF",       True,  [r"^AiM\s+RD_AF\s+(\d+)"]),
    ("AF",          True,  [r"^AiM\s+AF\s+(\d+)"]),
    # 如需扩展支持，在此追加并保证 02 与 03 都能自动复用
]
FEATURE_NAMES = [n for n, _, _ in FEATURE_SPECS]
FEATURE_HAS_SIZE = {n: has for n, has, _ in FEATURE_SPECS}
FEATURE_PATTERNS = {n: [re.compile(p) for p in pats] for n, _, pats in FEATURE_SPECS}

def parse_features_from_trace(trace_path: Path) -> Dict[str, Tuple[int, int]]:
    """读取 .aim，返回 {name: (calls, opsize)}"""
    counts = {name: [0, 0] for name in FEATURE_NAMES}
    with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for name in FEATURE_NAMES:
                for pat in FEATURE_PATTERNS[name]:
                    m = pat.search(line)
                    if not m:
                        continue
                    counts[name][0] += 1  # calls
                    if FEATURE_HAS_SIZE[name] and m.lastindex:
                        try:
                            counts[name][1] += int(m.group(1))  # opsize
                        except Exception:
                            pass
                    break
    return {k: (v[0], v[1]) for k, v in counts.items()}

# ----------------------- ramulator cycles 解析 -----------------------
CYCLE_PATTERNS = [r"memory_system_cycles:\s*([0-9]+)"]

def parse_metric(text: str, pattern: Optional[str]) -> Optional[int]:
    """解析 ramulator 输出中的 cycles。pattern 优先；否则使用默认 CYCLE_PATTERNS。"""
    pats = [pattern] if pattern else CYCLE_PATTERNS
    for pat in pats:
        m = re.search(pat, text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                continue
    return None

# ----------------------- trace 文件名/旁车 JSON 的元数据 -----------------------
def parse_meta_from_trace(trace_path: Path) -> Dict[str, str | int | None]:
    """综合 trace 同名 .json 与文件名，提取 op/size/硬件配置等元数据"""
    meta: Dict[str, str | int | None] = {
        "op": None, "with_af": None, "seqlen": None, "vector_dim": None, "matrix_col": None,
        "dim": None, "n_heads": None, "n_kv_heads": None,
        "DRAM_column": None, "DRAM_row": None, "burst_length": None, "num_banks": None, "num_channels": None,
        "threads": None, "reuse_size": None, "channels_per_block": None, "max_seq_len": None,
    }
    name = trace_path.name
    j = trace_path.with_suffix(".json")
    if j.exists():
        try:
            js = json.loads(j.read_text(encoding="utf-8"))
            for k in list(meta.keys()):
                if k in js:
                    meta[k] = js[k]
        except Exception:
            pass
    # 猜测字段（来自文件名）
    if meta["op"] is None:
        if name.startswith("score"):
            meta["op"] = "score"
        elif name.startswith("output"):
            meta["op"] = "output"
        elif name.startswith("weight"):
            meta["op"] = "weight"
    if meta["with_af"] is None:
        meta["with_af"] = 1 if "_withaf" in name or "_with_af" in name else 0
    m = re.search(r"_seq(\d+)_", name)
    if m and meta["seqlen"] is None:
        meta["seqlen"] = int(m.group(1))
    m = re.search(r"_vec(\d+)_", name)
    if m and meta["vector_dim"] is None:
        meta["vector_dim"] = int(m.group(1))
    m = re.search(r"_col(\d+)_", name)
    if m and meta["matrix_col"] is None:
        meta["matrix_col"] = int(m.group(1))
    m = re.search(r"_dim(\d+)_h(\d+)", name)
    if m:
        if meta["dim"] is None:
            meta["dim"] = int(m.group(1))
        if meta["n_heads"] is None:
            meta["n_heads"] = int(m.group(2))
    return meta

# ----------------------- 与 CENT 交互：公共工具 -----------------------
def ensure_cent_on_path(start: Optional[Path] = None) -> tuple[Path, Path]:
    """
    把 <repo root>/submodules/CENT/cent_simulation 与 <repo root> 放入 sys.path，返回 (PROJECT_ROOT, CENT_SIM_DIR)。
    """
    here = (start or Path(__file__)).resolve()
    for p in [here.parent] + list(here.parents):
        cand = p / "submodules" / "CENT" / "cent_simulation"
        if cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p, cand
    raise RuntimeError(f"Cannot find 'submodules/CENT/cent_simulation' above {here}")

# 不同算子的 timing 归属（用于 *_only_trace 的第三个参数）
TIMING = {
    "score":  "breakdown_sa_score",
    "output": "breakdown_sa_output",
    "weight": "breakdown_sa_weight",
}

def load_pim_config(path: Path) -> Dict[str, Any]:
    """从 JSON 读取并统一键名 + 默认值"""
    cfg = json.loads(path.read_text(encoding="utf-8"))
    std: Dict[str, Any] = {}
    alias = {
        "DRAM_column": ["dram_column", "DRAMCol", "dramCol", "dram_col"],
        "DRAM_row": ["dram_row", "DRAMRow", "dramRow", "dram_row"],
        "burst_length": ["burst", "burstLength", "BL"],
        "num_banks": ["banks", "numBanks"],
        "num_channels": ["channels", "numChannels"],
        "threads": ["thread", "nThreads"],
        "reuse_size": ["reuseSize", "reuse", "RS"],
        "channels_per_block": ["channelsPerBlock", "cpb"],
        "max_seq_len": ["maxSeqLen", "max_seq_length"],
    }
    for k, v in cfg.items():
        matched = False
        for stdk, alist in alias.items():
            if k == stdk or k in alist:
                std[stdk] = v
                matched = True
                break
        if not matched:
            std[k] = v
    std.setdefault("DRAM_column", 256)
    std.setdefault("DRAM_row", 64)
    std.setdefault("burst_length", 16)
    std.setdefault("num_banks", 8)
    std.setdefault("num_channels", 4)
    std.setdefault("threads", 1)
    std.setdefault("reuse_size", 32)
    std.setdefault("channels_per_block", None)  # None -> 用 num_channels
    std.setdefault("max_seq_len", 4096)
    return std

def make_tb_args_from_pim(cfg: Dict[str, Any], trace_file: str):
    """把 PIM 配置组合成 TransformerBlock 所需的 SimpleNamespace"""
    from types import SimpleNamespace
    return SimpleNamespace(
        DRAM_column=cfg["DRAM_column"],
        DRAM_row=cfg["DRAM_row"],
        burst_length=cfg["burst_length"],
        num_banks=cfg["num_banks"],
        num_channels=cfg["num_channels"],
        threads=cfg["threads"],
        reuse_size=cfg["reuse_size"],
        channels_per_block=cfg["channels_per_block"],
        max_seq_len=cfg["max_seq_len"],
        # tracing toggles
        only_trace=True, op_trace=False, trace_file=trace_file,
        # TB-specific
        pim_compute=True, model="llama_like", embedding="rope",
        seqlen=16, model_parallel=False, FC_devices=1,
        pipeline_parallel=False, inter_device_attention=False, only_FC=False,
        trace_prepare=False, trace_norm=False, trace_fc_kqvo=False, trace_attention=False,
        trace_softmax=False, trace_fc_ffn=False, trace_activation=False,
        GEMV="reuse-GB",
    )

def make_dic_model(dim: int, n_heads: int, n_kv_heads: Optional[int], seqlen: int) -> Dict[str, Any]:
    """构造 TransformerBlock 所需的 dic_model"""
    if n_kv_heads is None:
        n_kv_heads = n_heads
    import torch  # 延迟导入
    head_dim = dim // n_heads
    assert head_dim > 0, "dim must be divisible by n_heads"
    return {
        "TP_param": torch.tensor(1),
        "dim": torch.tensor(dim),
        "n_heads": torch.tensor(n_heads),
        "n_kv_heads": torch.tensor(n_kv_heads),
        "x": torch.randn(1, seqlen, dim),
        "SANorm": torch.ones(dim),
        "FFNNorm": torch.ones(dim),
        "start_pos": 0,
        "freqs_cis": torch.zeros(seqlen, head_dim, dtype=torch.complex64),
        "sa": torch.zeros(1, n_heads, seqlen, seqlen),
        "h": torch.zeros(1, n_heads, seqlen, head_dim),
        "out": torch.zeros(1, seqlen, dim),
        "ffn": torch.zeros(1, seqlen, dim),
        # 权重/缓存形状（无需真实值，只要形状对齐即可）
        "wq": torch.randn(dim, dim),
        "wk": torch.randn(dim, dim),
        "wv": torch.randn(dim, dim),
        "wo": torch.randn(dim, dim),
        "w1": torch.randn(4*dim, dim),
        "w2": torch.randn(dim, 4*dim),
        "w3": torch.randn(4*dim, dim),
        "xq": torch.randn(1, n_heads, 1, dim // n_heads),
        "xk": torch.randn(1, n_kv_heads, 1, dim // n_heads),
        "xv": torch.randn(1, n_kv_heads, 1, dim // n_heads),
        "cache_k": torch.randn(1, seqlen, n_kv_heads, dim // n_heads),
        "cache_v": torch.randn(1, seqlen, n_kv_heads, dim // n_heads),
        "scores": torch.randn(1, n_heads, 1, seqlen),
        "output": torch.zeros(1, seqlen, dim),
    }

def emit_single_op_trace(block, op: str, row_index_matrix: int,
                         seqlen: Optional[int] = None,
                         vector_dim: Optional[int] = None,
                         matrix_col: Optional[int] = None,
                         with_af: bool = False) -> None:
    """统一的单算子 only_trace 发射器，供 01 与 03 复用"""
    if op == "score":
        if seqlen is None:
            raise ValueError("score 需要 seqlen")
        block.Vector_Matrix_Mul_score_pim_only_trace(row_index_matrix, seqlen, TIMING["score"])
    elif op == "output":
        if seqlen is None:
            raise ValueError("output 需要 seqlen")
        block.Vector_Matrix_Mul_output_pim_only_trace(row_index_matrix, seqlen, TIMING["output"])
    elif op == "weight":
        if vector_dim is None or matrix_col is None:
            raise ValueError("weight 需要 vector_dim 与 matrix_col")
        channel_lst = [i for i in range(block.num_channels)]
        total_banks = block.FC_total_banks
        block.Vector_Matrix_Mul_weight_pim_only_trace(
            channel_lst, row_index_matrix, vector_dim, matrix_col, total_banks, TIMING["weight"]
        )
        if with_af:
            block.AF_only_trace(channel_lst)
            block.RD_AF_only_trace(channel_lst)
    else:
        raise ValueError(f"Unsupported op: {op}")

# ----------------------- 小工具 -----------------------
def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]

# ----------------------- 模型形状（mpt/qwen 等） -----------------------
def load_model_shape(shape_path: Path) -> Dict[str, Any]:
    """
    读取 ../configs/*_shape.json，提取 dim/n_heads/n_kv_heads/seq_length。
    兼容多种命名：
      - dim: hidden_dim, hidden_size, d_model, model_dim, dim
      - n_heads: q_head_num, num_attention_heads, n_head, head_num
      - n_kv_heads: kv_head_num, num_key_value_heads, n_kv_head
      - seq_length: seq_length, max_seq_len, context_length, max_position_embeddings
    """
    j = json.loads(Path(shape_path).read_text(encoding="utf-8"))
    def pick(obj, keys, default=None):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return default
    dim = pick(j, ["hidden_dim", "hidden_size", "d_model", "model_dim", "dim"])
    n_heads = pick(j, ["q_head_num", "num_attention_heads", "n_head", "head_num"])
    n_kv_heads = pick(j, ["kv_head_num", "num_key_value_heads", "n_kv_head"], default=n_heads)
    seq_length = pick(j, ["seq_length", "seq_len", "context_length", "max_seq_len", "max_position_embeddings"])
    if dim is None or n_heads is None:
        raise ValueError(f"模型形状文件缺少必要字段 dim/n_heads: {shape_path}")
    return {
        "dim": int(dim),
        "n_heads": int(n_heads),
        "n_kv_heads": int(n_kv_heads) if n_kv_heads is not None else int(n_heads),
        "seq_length": int(seq_length) if seq_length is not None else None,
        "raw": j,
    }
