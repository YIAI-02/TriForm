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

# 不同算子的 timing 归属（用于 *_only_trace 的第三个参数）
TIMING = {
    "score":  "breakdown_sa_score",
    "output": "breakdown_sa_output",
    "weight": "breakdown_sa_weight",
}







# ----------------------- 小工具 -----------------------
def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]

# ----------------------- 模型形状（mpt/qwen 等） -----------------------
# def load_model_shape(shape_path: Path) -> Dict[str, Any]:
#     """
#     读取 ../configs/*_shape.json，提取 dim/n_heads/n_kv_heads/seq_length。
#     兼容多种命名：
#       - dim: hidden_dim, hidden_size, d_model, model_dim, dim
#       - n_heads: q_head_num, num_attention_heads, n_head, head_num
#       - n_kv_heads: kv_head_num, num_key_value_heads, n_kv_head
#       - seq_length: seq_length, max_seq_len, context_length, max_position_embeddings
#     """
#     j = json.loads(Path(shape_path).read_text(encoding="utf-8"))
#     def pick(obj, keys, default=None):
#         for k in keys:
#             if k in obj and obj[k] is not None:
#                 return obj[k]
#         return default
#     dim = pick(j, ["hidden_dim", "hidden_size", "d_model", "model_dim", "dim"])
#     n_heads = pick(j, ["q_head_num", "num_attention_heads", "n_head", "head_num"])
#     n_kv_heads = pick(j, ["kv_head_num", "num_key_value_heads", "n_kv_head"], default=n_heads)
#     seq_length = pick(j, ["seq_length", "seq_len", "context_length", "max_seq_len", "max_position_embeddings"])
#     if dim is None or n_heads is None:
#         raise ValueError(f"模型形状文件缺少必要字段 dim/n_heads: {shape_path}")
#     return {
#         "dim": int(dim),
#         "n_heads": int(n_heads),
#         "n_kv_heads": int(n_kv_heads) if n_kv_heads is not None else int(n_heads),
#         "seq_length": int(seq_length) if seq_length is not None else None,
#         "raw": j,
#     }
