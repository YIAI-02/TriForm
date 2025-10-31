# main.py
from __future__ import annotations

import argparse
import json
import os
import time
import math
import random
from typing import Dict, List
from hardware import demo_cluster, Cluster
from cost_model import CostModel, DTYPE_BYTES,_make_shared_model_dict
from buffer_manager import GlobalMemoryManager
from task_graph import TaskGraph
from model_parser import build_graph
from config import (
    DEFAULT_CONFIG,
    WEIGHT_FORMAT_JSON_PATH,
    FORMAT_TUNING_MAX_PASSES,
    FORMAT_TUNING_TIME_EPS,
    FORMAT_TUNING_MAP_EPS,
    SA_ENABLE, SA_T0, SA_ALPHA, SA_FLIP_PROB,
    ALL_PASSES_RESULT_PATH, BEST_PASS_SUMMARY_PATH
)
from plan_label import PlanLabel
from scheduler import HEFTScheduler, ScheduledTask
from pathlib import Path

DEBUG_MAIN = False

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
def _serialize_schedule(schedule: List[ScheduledTask], *, phase: str, token_idx: int | None = None) -> List[Dict]:
    """Convert ScheduledTask list to JSON-friendly dicts."""
    out: List[Dict] = []
    for t in schedule:
        out.append({
            "node_id": t.node_id,
            "device": t.device,
            "start": float(t.start),
            "finish": float(t.finish),
            "duration": float(max(0.0, t.finish - t.start)),
            "phase": phase,
            "token_idx": token_idx,
        })
    return out

def simulate_prefill(sched: HEFTScheduler, cfg: Dict, graph: TaskGraph) -> tuple[float, List[Dict]]:
    """
    Simulate prefill phase: process entire prefix at once.
    current_length = prefill_len
    """
    prefill_len = int(cfg.get("prefill_len", 128))
    sched.set_seq_len(prefill_len)  # current_length for prefill
    prefill_sched = sched.schedule(graph, phase="prefill")
    prefill_time = sched.makespan(prefill_sched)
    return prefill_time, _serialize_schedule(prefill_sched, phase="prefill", token_idx=None)


def simulate_decode_progressive(sched: HEFTScheduler, cfg: Dict, graph: TaskGraph, prefill_end: float) -> tuple[float, List[Dict]]:
    """
    Simulate decode phase progressively: one token at a time.
    current_length increases from (prefill_len) to (prefill_len + decode_len - 1)
    """
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))
    global_end = prefill_end
    steps_serialized: List[Dict] = []
    
    for t in range(decode_len):
        current_length = prefill_len + t  # Total sequence length so far
        sched.set_seq_len(current_length)
        dec_sched_t = sched.schedule(graph, phase="decode")
        token_end = sched.makespan(dec_sched_t)
        step_time = max(0.0, token_end - global_end)
        steps_serialized.append({
            "t": t,
            "seq_len": current_length,
            "end_time": float(token_end),
            "delta_time": float(step_time),
            "schedule": _serialize_schedule(dec_sched_t, phase="decode", token_idx=t),
        })
        if token_end > global_end:
            global_end = token_end

    return max(0.0, global_end - prefill_end), steps_serialized


def mapping_diff_ratio(a: Dict[str, str], b: Dict[str, str]) -> float:
    """两个权重-格式映射的差异比例（Hamming ratio）。"""
    if not a and not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    diff = sum(1 for k in keys if a.get(k) != b.get(k))
    return diff / float(len(keys))

def _sa_make_neighbor_map(base_map: Dict[str, str], weight_ids: List[str], flip_prob: float = 0.15) -> Dict[str, str]:
    """
    基于 base_map 生成邻域解：对若干权重随机翻转其存储格式（在 ND / NPU_OPT / PIM_OPT 中切换）。
    至少翻转 1 个权重，避免与 base_map 完全相同。
    """
    CAND = ("ND", "NPU_OPT", "PIM_OPT")
    if not weight_ids:
        return dict(base_map)
    out = dict(base_map)
    flips = 0
    for wid in weight_ids:
        if random.random() < max(0.0, min(1.0, flip_prob)):
            old = out.get(wid, base_map.get(wid, "ND"))
            choices = [x for x in CAND if x != old] or ["ND"]
            out[wid] = random.choice(choices)
            flips += 1
    if flips == 0:
        wid = random.choice(weight_ids)
        old = out.get(wid, base_map.get(wid, "ND"))
        choices = [x for x in CAND if x != old] or ["ND"]
        out[wid] = random.choice(choices)
    return out
# ------------------------------
# CLI & run
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_family", type=str, default=DEFAULT_CONFIG["model_family"])
    p.add_argument("--model_variant", type=str, default=DEFAULT_CONFIG["model_variant"])
    p.add_argument("--dtype", type=str, default=DEFAULT_CONFIG["dtype"])
    p.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch"])
    p.add_argument("--prefill_len", type=int, default=DEFAULT_CONFIG["prefill_len"])
    p.add_argument("--decode_len", type=int, default=DEFAULT_CONFIG["decode_len"])
    p.add_argument("--pim_config_path", type=str, default=DEFAULT_CONFIG["pim_config_path"])
    p.add_argument("--gb_config_path", type=str, default=DEFAULT_CONFIG["gb_config_path"], help="Path to the Global Buffer configuration file")  
    p.add_argument("--ramulator_config_path", type=str, default=DEFAULT_CONFIG["ramulator_config_path"])
    p.add_argument("--simulation_log_file", type=str, default="./output/pim_simulation.txt", help="Output log file")
    p.add_argument("--all_passes_json", type=str, default=ALL_PASSES_RESULT_PATH, help="Write all-pass records (schedules/times/weights) to this JSON")
    p.add_argument("--best_summary_json", type=str, default=BEST_PASS_SUMMARY_PATH, help="Write best-pass detail and improvements to this JSON")
    
    return p.parse_args()


def run(cfg: Dict):
    # Setup
    cluster = demo_cluster()
    pim_config_path = Path(cfg["pim_config_path"])
    gb_config_path = Path(cfg["gb_config_path"])
    ramulator_config_path = Path(cfg["ramulator_config_path"])
    
    #build graph
    prefill_len = int(cfg.get("prefill_len", 128))
    batch = int(cfg.get("batch", 1))
    graph, shape = build_graph(cfg)

    print("[Main] Creating shared model dictionary...")
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, "dim", 128)),
        n_heads=int(getattr(shape, "n_heads", 1)),
        n_kv_heads=int(getattr(shape, "n_kv_heads", 1)),
        ffn_dim=int(getattr(shape, "ffn_dim", 512)),
        seqlen=prefill_len
    )
    print(f"[Main] Model dict created: dim={getattr(shape, 'dim')}, "
          f"n_heads={getattr(shape, 'n_heads')}, "
          f"n_kv_heads={getattr(shape, 'n_kv_heads')}, "
          f"ffn_dim={getattr(shape, 'ffn_dim')}")
    
    #cost model
    log_file = Path(cfg.get("simulation_log_file", "pim_simulation.txt"))
    cost = CostModel(
        cluster, 
        dtype=cfg.get("dtype", "fp16"), 
        pim_config_path=pim_config_path,
        gb_config_path=gb_config_path,
        ramulator_config_path=ramulator_config_path,
        simulation_log_file=log_file,
        debug_traces=False,
        model_dict=model_dict
    )
        
    #start simulation
    cost.logger.start_simulation()
    
    try:
        label = plan_memory_and_label(cfg, cluster)

        # 多次迭代直至收敛
        fmt_map: Dict[str, str] = {}
        prev_total: float = None           # 上一次“被接受”的方案的 total_time
        prev_map: Dict[str, str] = {}      # 上一次“被接受”的映射
        best_total: float = None           # 历史最优总代价
        best_map: Dict[str, str] = {}      # 历史最优映射
        best_pass: int = -1
        last_prefill = 0.0
        last_decode = 0.0
        all_pass_records: List[Dict] = []
        # SA 参数
        sa_enable = bool(SA_ENABLE)
        T = float(SA_T0)
        alpha = float(SA_ALPHA)
        flip_prob = float(SA_FLIP_PROB)


        # shared buffer manager across passes to accumulate stats
        buffer_mgr = GlobalMemoryManager()
        for p in range(1, FORMAT_TUNING_MAX_PASSES + 1):
            print(f"\n{'='*80}")
            print(f"Starting optimization pass {p}/{FORMAT_TUNING_MAX_PASSES}")
            print(f"{'='*80}\n")
            
            sched = HEFTScheduler(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
            sched.set_storage_format_map(fmt_map)
            sched.reset_state()
            t_wall0 = time.time()
            print(f"[PASS{p}] Running prefill phase (current map)...")
            prefill_time, prefill_sched_ser = simulate_prefill(sched, cfg, graph)
            print(f"[PASS{p}] Running decode phase (current map)...")
            decode_time, decode_steps_ser = simulate_decode_progressive(sched, cfg, graph, prefill_end=prefill_time)
            total_time = prefill_time + decode_time
            wall_time_current = time.time() - t_wall0
            cost.logger._log(f"[PASS{p}] CurrentMap  WallTime: {wall_time_current:.3f}s | Prefill(sim): {prefill_time:.6f}s, Decode(sim): {decode_time:.6f}s, Total(sim): {total_time:.6f}s")
            fmt_suggestion = sched.suggest_weight_storage_formats()
            wids = sorted(set(list(fmt_suggestion.keys()) + list(fmt_map.keys())))
            if sa_enable:
                neighbor_map = _sa_make_neighbor_map(fmt_suggestion or fmt_map, wids, flip_prob)
                sched2 = HEFTScheduler(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
                sched2.set_storage_format_map(neighbor_map)
                sched2.reset_state()
                t_wall1 = time.time()
                print(f"[PASS{p}] Running prefill phase (neighbor map)...")
                prefill_time_nb, prefill_sched_ser_nb = simulate_prefill(sched, cfg, graph)
                print(f"[PASS{p}] Running decode phase (neighbor map)...")
                decode_time_nb, decode_steps_ser_nb = simulate_decode_progressive(sched, cfg, graph, prefill_end=prefill_time)
                total_time_nb = prefill_time_nb + decode_time_nb
                wall_time_neighbor = time.time() - t_wall1
                cost.logger._log(f"[PASS{p}] NeighborMap WallTime: {wall_time_neighbor:.3f}s | Prefill(sim): {prefill_time_nb:.6f}s, Decode(sim): {decode_time_nb:.6f}s, Total(sim): {total_time_nb:.6f}s")

                # SA 接受准则
                delta = total_time_nb - total_time
                accept = (delta < 0.0) or (random.random() < math.exp(-max(0.0, delta) / max(1e-9, T)))
                fmt_next = neighbor_map if accept else fmt_map
                chosen_total = total_time_nb if accept else total_time
                chosen_prefill = prefill_time_nb if accept else prefill_time
                chosen_decode = decode_time_nb if accept else decode_time
                prefill_sched_ser = prefill_sched_ser_nb if accept else prefill_sched_ser
                decode_steps_ser = decode_steps_ser_nb if accept else decode_steps_ser
                cost.logger._log(f"[PASS{p}] SA decision: {'ACCEPT' if accept else 'REJECT'} (Δ={delta:+.6f}, T={T:.4f})")
            else:
                fmt_next = fmt_suggestion
                chosen_total = total_time
                chosen_prefill = prefill_time
                chosen_decode = decode_time

            # 记录历史最优
            if best_total is None or chosen_total < best_total:
                best_total = chosen_total
                best_map = dict(fmt_next)
                best_pass = p

            # 覆盖写 JSON 为“本轮被接受的映射”（不是建议）
            os.makedirs(os.path.dirname(WEIGHT_FORMAT_JSON_PATH), exist_ok=True)
            with open(WEIGHT_FORMAT_JSON_PATH, "w") as f:
                json.dump(fmt_next, f, indent=2, sort_keys=True)
            print(f"[INFO] Accepted weight storage map saved: {WEIGHT_FORMAT_JSON_PATH}")


            # 记录本轮结果
            weight_stats = sched.export_weight_stats()

            # NEW: append this pass record (full schedules & times)
            all_pass_records.append({
                "pass": p,
                "times": {
                    "prefill": float(prefill_time),
                    "decode": float(decode_time),
                    "total": float(total_time),
                },
                "schedules": {
                    "prefill": prefill_sched_ser,
                    "decode_steps": decode_steps_ser,
                },
                "formats": {
                    "used_storage_map": dict(fmt_map or {}),
                    "suggested_storage_map": dict(fmt_suggestion or {}),
                    "suggestion_json_path": WEIGHT_FORMAT_JSON_PATH,
                },
                "weights": weight_stats,
            })
           # 收敛判定：与“上一次被接受”的方案比较
            if prev_total is not None:
                time_improve = prev_total - chosen_total
                map_delta = mapping_diff_ratio(prev_map, fmt_next)
                print(f"\n[DELTA] Time improvement (accepted vs prev_accepted): {time_improve:+.6f}s")
                print(f"[DELTA] Format map change ratio (accepted vs prev_accepted): {map_delta:.4f}")
                if abs(time_improve) <= FORMAT_TUNING_TIME_EPS and map_delta <= FORMAT_TUNING_MAP_EPS:
                    print(f"\n[CONVERGENCE] Optimization converged at pass {p}")
                    print(f"  Time delta: {abs(time_improve):.6e}s <= {FORMAT_TUNING_TIME_EPS:.6e}s")
                    print(f"  Map delta:  {map_delta:.4f} <= {FORMAT_TUNING_MAP_EPS:.4f}")
                    # 提前退出
                    last_prefill, last_decode = chosen_prefill, chosen_decode
                    prev_total, prev_map, fmt_map = chosen_total, dict(fmt_next), dict(fmt_next)
                    break

            last_prefill, last_decode = chosen_prefill, chosen_decode
            prev_total = chosen_total
            prev_map = dict(fmt_next)
            fmt_map = dict(fmt_next)

            # SA 冷却
            if sa_enable:
                T *= alpha    

        print(f"\n{'='*80}")
        print(f"Optimization Complete")
        print(f"{'='*80}")

        if best_map:
            with open(WEIGHT_FORMAT_JSON_PATH, "w") as f:
                json.dump(best_map, f, indent=2, sort_keys=True)
            print(f"[INFO] Best weight storage map (found at pass {best_pass}) saved to: {WEIGHT_FORMAT_JSON_PATH}")
        print(f"Best total time: {best_total:.6f}s (at pass {best_pass})")
        print(f"Last accepted prefill(sim): {last_prefill:.6f}s")
        print(f"Last accepted decode(sim):  {last_decode:.6f}s")

        out_dir = Path("./output"); out_dir.mkdir(parents=True, exist_ok=True)
        all_path = Path(cfg.get("all_passes_json", ALL_PASSES_RESULT_PATH))
        best_path = Path(cfg.get("best_summary_json", BEST_PASS_SUMMARY_PATH))

        # All passes
        with open(all_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "model_family": cfg.get("model_family"),
                    "model_variant": cfg.get("model_variant"),
                    "dtype": cfg.get("dtype"),
                    "batch": cfg.get("batch"),
                    "prefill_len": cfg.get("prefill_len"),
                    "decode_len": cfg.get("decode_len"),
                },
                "passes": all_pass_records,
            }, f, ensure_ascii=False, indent=2)
        print(f"[REPORT] All passes saved to: {all_path}")

        # Best pass summary
        if all_pass_records:
            # find best by 'times.total'
            best_idx = min(range(len(all_pass_records)), key=lambda i: all_pass_records[i]["times"]["total"])
            best_rec = all_pass_records[best_idx]
            best_total = best_rec["times"]["total"]
            improvements = []
            for rec in all_pass_records:
                tot = rec["times"]["total"]
                delta = float(tot - best_total)
                pct = (delta / tot * 100.0) if tot > 0 else 0.0
                improvements.append({
                    "pass": rec["pass"],
                    "total_time": float(tot),
                    "delta_seconds_vs_best": float(delta),
                    "delta_percent_vs_that_pass": float(pct),
                })

            with open(best_path, "w", encoding="utf-8") as f:
                json.dump({
                    "best_pass": best_rec["pass"],
                    "best_times": best_rec["times"],
                    "best_formats": best_rec["formats"],
                    "best_weights": best_rec["weights"],
                    "prefill_schedule": best_rec["schedules"]["prefill"],
                    "decode_steps": best_rec["schedules"]["decode_steps"],
                    "improvements_vs_each_pass": improvements,
                }, f, ensure_ascii=False, indent=2)
            print(f"[REPORT] Best pass summary saved to: {best_path}")
    
    finally:
        # 结束仿真并打印统计
        cost.logger.end_simulation()
        cost.logger.close()


def main():
    args = parse_args()
    cfg = {
        "model_family": args.model_family,
        "model_variant": args.model_variant,
        "dtype": args.dtype,
        "batch": args.batch,
        "prefill_len": args.prefill_len,
        "decode_len": args.decode_len,
        "pim_config_path": args.pim_config_path,
        "gb_config_path": args.gb_config_path,  # 新增
        "ramulator_config_path": args.ramulator_config_path,
        "simulation_log_file": args.simulation_log_file,
        "all_passes_json": args.all_passes_json,
        "best_summary_json": args.best_summary_json,
    }
    run(cfg)

if __name__ == "__main__":
    main()
