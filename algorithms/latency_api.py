#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
latency_api.py
为算法提供统一接口：
- 读取 measurements/pim/out_run/results/model_fit_summary.json（或显式传入路径）
- 基于显式公式估算 PIM 延迟（cycles 或秒）
- 提供与算法算子命名一致的映射：
  q*k^t -> score； s*v -> output； x*wq,x*wk,x*wv,ffn -> weight（若有激活 -> weight_af）

NPU 延迟：保留为占位/透传，建议继续使用你现有的 CostModel 实现（如按 flops/吞吐量计算）。
"""

from __future__ import annotations
from typing import Optional
from pathlib import Path

from pim_latency_provider import PIMFormulaLatency

# --- 加载全局 PIM 公式模型 ---
_DEFAULT: Optional[PIMFormulaLatency] = None

def load_pim_model(path: Optional[Path] = None) -> PIMFormulaLatency:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PIMFormulaLatency.load(path) if path else PIMFormulaLatency.load_default()
    return _DEFAULT

def estimate_pim_cycles(op_name: str,
                        *, seqlen: int = 0, vector_dim: int = 0, matrix_col: int = 0, n_heads: int = 0,
                        has_activation_after: bool = False,
                        summary_path: Optional[Path] = None) -> float:
    model = load_pim_model(summary_path)
    return model.estimate_cycles(op_name, seqlen=seqlen, vector_dim=vector_dim,
                                 matrix_col=matrix_col, n_heads=n_heads,
                                 has_activation_after=has_activation_after)

def estimate_pim_seconds(*args, tCK_ns: Optional[float] = None, **kwargs) -> float:
    model = load_pim_model(kwargs.pop("summary_path", None))
    return model.estimate_seconds(*args, tCK_ns=tCK_ns, **kwargs)

# NPU 时间建议仍使用你的 CostModel；这里给个简单透传接口：
def estimate_npu_seconds_by_throughput(work: float, peak_tps: float, launch_overhead: float = 0.0) -> float:
    """按吞吐量与启动开销粗略估计 NPU 延迟（秒）。work 与 peak_tps 单位需要调用方统一。"""
    return launch_overhead + work / max(peak_tps, 1e-9)
