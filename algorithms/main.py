# main.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Set
from hardware import demo_cluster, Cluster
from cost_model import CostModel, DTYPE_BYTES
from buffer_manager import BufferManager
from task_graph import TaskGraph
from config import (
    DEFAULT_CONFIG,
    ENABLE_TWO_PASS_FORMAT_TUNING,
    WEIGHT_FORMAT_JSON_PATH,
    FORMAT_TUNING_MAX_PASSES,
    FORMAT_TUNING_TIME_EPS,
    FORMAT_TUNING_MAP_EPS,
)

DEBUG_MAIN = False
# ------------------------------
# Plan label (PIM memory planning)
# ------------------------------
@dataclass
class PlanLabel:
    pim_mode: str                 # "small" | "medium" | "large"
    kv_in_pim: bool
    pim_weight_capacity_bytes: int = 0
    pinned_fc_on_pim: Set[str] = field(default_factory=set)
    
    def print_debug(self) -> None:
        """Print all PlanLabel settings for debugging."""
        print("=" * 50)
        print("PIM MEMORY PLAN DEBUG INFO")
        print("=" * 50)
        print(f"PIM Mode: {self.pim_mode}")
        print(f"KV Cache in PIM: {self.kv_in_pim}")
        print(f"Number of Pinned FC Weights: {len(self.pinned_fc_on_pim)}")
        if self.pinned_fc_on_pim:
            print("Pinned FC Weights:")
            for weight_id in sorted(self.pinned_fc_on_pim):
                print(f"  - {weight_id}")
        else:
            print("Pinned FC Weights: None")
        print("=" * 50)


def plan_memory_and_label(cfg: Dict, cluster: Cluster) -> PlanLabel:
    """
    Decide PIM mode and effective LRU budget for weights on PIM.
    small:   PIM capacity < KV_total -> kv_in_pim=False, no persistent cache
    medium:  KV_total <= cap < (KV_total + FC_total) -> kv_in_pim=True, weight cache budget = cap - KV_total
    large:   cap >= KV_total + FC_total -> kv_in_pim=True, budget large enough to avoid eviction
    """
    # Build unified graph once to collect weight sizes
    g, shape = build_graph(cfg)

    dtype_bytes = int(DTYPE_BYTES.get(cfg.get("dtype", "fp16"), 2))
    # KV total bytes over prefill+decode
    S = int(cfg.get("prefill_len", 128))
    T = int(cfg.get("decode_len", 32))
    batch = int(cfg.get("batch", 1))
    layers = int(getattr(shape, "layer_num", 1))
    n_kv_heads = int(getattr(shape, "n_kv_heads", 1))
    head_dim = int(getattr(shape, "head_dim", max(1, getattr(shape, "dim", 1) // max(1, getattr(shape, "n_heads", 1)))))

    kv_elems = 2 * (S + T) * n_kv_heads * head_dim * batch * layers  # K+V
    KV_total_bytes = kv_elems * dtype_bytes

    # Sum FC (W1/W2/W3) and attention (Wq/Wk/Wv/Wo) weights sizes
    FC_total_bytes = 0
    per_weight_size: Dict[str, int] = {}
    for n in g.nodes.values():
        if getattr(n, "weight_id", None) and isinstance(n.weight_id, str):
            if (n.weight_id.endswith("W1") or n.weight_id.endswith("W2") or n.weight_id.endswith("W3") or
                n.weight_id.endswith("WQ") or n.weight_id.endswith("WK") or n.weight_id.endswith("WV") or n.weight_id.endswith("WO")):
                FC_total_bytes += int(getattr(n, "weight_size", 0))
                per_weight_size[n.weight_id] = int(getattr(n, "weight_size", 0))

    # PIM capacity (sum)
    pim_bytes = 0
    for d in cluster.devices_by_type("pim"):
        pim_bytes += int(d.mem_capacity_GB * 1e9)
    
    if DEBUG_MAIN:
        print(f"[DEBUG] KV total bytes: {KV_total_bytes:,} ({KV_total_bytes/1e9:.2f} GB)")
        print(f"[DEBUG] FC total bytes: {FC_total_bytes:,} ({FC_total_bytes/1e9:.2f} GB)")
        print(f"[DEBUG] PIM capacity: {pim_bytes:,} ({pim_bytes/1e9:.2f} GB)")
        print(f"[DEBUG] Total weights found: {len(per_weight_size)}")
    
    if pim_bytes < KV_total_bytes:
        label = PlanLabel(pim_mode="small", kv_in_pim=False, pim_weight_capacity_bytes=0)
    elif pim_bytes >= (KV_total_bytes + FC_total_bytes):
        label = PlanLabel(pim_mode="large", kv_in_pim=True, pim_weight_capacity_bytes=pim_bytes-KV_total_bytes)
    else:
        label = PlanLabel(pim_mode="medium",kv_in_pim=True, pim_weight_capacity_bytes=pim_bytes-KV_total_bytes)
    
    # Print debug info
    if DEBUG_MAIN:
        label.print_debug()
    return label

# ------------------------------
# Progressive simulation helpers
# ------------------------------
def simulate_prefill(sched: HEFTScheduler, cfg: Dict, graph: TaskGraph) -> float:
    """
    Simulate prefill phase: process entire prefix at once.
    current_length = prefill_len
    """
    prefill_len = int(cfg.get("prefill_len", 128))
    sched.set_seq_len(prefill_len)  # current_length for prefill
    prefill_sched = sched.schedule(graph, phase="prefill")
    prefill_time = sched.makespan(prefill_sched)
    return prefill_time


def simulate_decode_progressive(sched: HEFTScheduler, cfg: Dict, graph: TaskGraph, prefill_end: float) -> float:
    """
    Simulate decode phase progressively: one token at a time.
    current_length increases from (prefill_len) to (prefill_len + decode_len - 1)
    """
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))
    global_end = prefill_end
    
    for t in range(decode_len):
        current_length = prefill_len + t  # Total sequence length so far
        sched.set_seq_len(current_length)
        dec_sched_t = sched.schedule(graph, phase="decode")
        token_end = sched.makespan(dec_sched_t)
        if token_end > global_end:
            global_end = token_end

    return max(0.0, global_end - prefill_end)


def mapping_diff_ratio(a: Dict[str, str], b: Dict[str, str]) -> float:
    """两个权重-格式映射的差异比例（Hamming ratio）。"""
    if not a and not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    diff = sum(1 for k in keys if a.get(k) != b.get(k))
    return diff / float(len(keys))


# ------------------------------
# CLI & run
# ------------------------------
def run(cfg: Dict):
    # Setup
    cluster = demo_cluster()
    cost = CostModel(cluster, dtype=cfg.get("dtype", "fp16"))
    label = plan_memory_and_label(cfg, cluster)

    prefill_len = int(cfg.get("prefill_len", 128))
    batch = int(cfg.get("batch", 1))
    graph, shape = build_graph(cfg)

    # 多次迭代直至收敛
    fmt_map: Dict[str, str] = {}
    prev_total: float = None  # type: ignore
    prev_map: Dict[str, str] = {}
    best_total: float = None  # type: ignore

    # shared buffer manager across passes to accumulate stats
    buffer_mgr = BufferManager()
    for p in range(1, FORMAT_TUNING_MAX_PASSES + 1):
        sched = HEFTScheduler(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
        sched.set_storage_format_map(fmt_map)#加载之前的format

        # Prefill
        prefill_time = simulate_prefill(sched, cfg, graph)
        # Progressive Decode（current_length 按 token 增长）
        decode_time = simulate_decode_progressive(sched, cfg, graph, prefill_end=prefill_time)

        total_time = prefill_time + decode_time
        if best_total is None or total_time < best_total:
            best_total = total_time

        print(f"[PASS{p}] prefill={prefill_time:.6f}s decode={decode_time:.6f}s total={total_time:.6f}s")

        # 基于“实际加载/流式”统计给出新的主存格式建议
        fmt_suggestion = sched.suggest_weight_storage_formats()

        # 覆盖写 JSON：确保目录存在（你要求的这段）
        os.makedirs(os.path.dirname(WEIGHT_FORMAT_JSON_PATH), exist_ok=True)
        with open(WEIGHT_FORMAT_JSON_PATH, "w") as f:
            json.dump(fmt_suggestion, f, indent=2, sort_keys=True)
        print(f"[INFO] weight storage suggestion saved: {WEIGHT_FORMAT_JSON_PATH}")

        # 收敛判定：时间与映射差异同时达标则停止
        if prev_total is not None:
            time_improve = prev_total - total_time
            map_delta = mapping_diff_ratio(prev_map, fmt_suggestion)
            print(f"[DELTA] Δtime={time_improve:+.6f}s, map_change_ratio={map_delta:.4f}")
            if abs(time_improve) <= FORMAT_TUNING_TIME_EPS and map_delta <= FORMAT_TUNING_MAP_EPS:
                print(f"[STOP] Converged at pass {p}.")
                break

        prev_total = total_time
        prev_map = fmt_suggestion
        fmt_map = fmt_suggestion  # 下一轮采用新的主存权重格式

    print(f"[BEST] best_total={best_total:.6f}s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_family", type=str, default=DEFAULT_CONFIG["model_family"])
    p.add_argument("--model_variant", type=str, default=DEFAULT_CONFIG["model_variant"])
    p.add_argument("--dtype", type=str, default=DEFAULT_CONFIG["dtype"])
    p.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch"])
    p.add_argument("--prefill_len", type=int, default=DEFAULT_CONFIG["prefill_len"])
    p.add_argument("--decode_len", type=int, default=DEFAULT_CONFIG["decode_len"])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        "model_family": args.model_family,
        "model_variant": args.model_variant,
        "dtype": args.dtype,
        "batch": args.batch,
        "prefill_len": args.prefill_len,
        "decode_len": args.decode_len,
    }
    run(cfg)


if __name__ == "__main__":
    main()
