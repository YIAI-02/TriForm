from __future__ import annotations
from config import attach_local_debug_filter
import argparse
import json
import os
import time
import math
import random
from typing import Dict, List
from hardware import demo_cluster, Cluster
from cost_model import CostModel, DTYPE_BYTES, _make_shared_model_dict, reset_simulation_logger
from buffer_manager import GlobalMemoryManager
from model_parser import build_graph
from config import DEFAULT_CONFIG, WEIGHT_FORMAT_JSON_PATH, FORMAT_TUNING_MAX_PASSES, FORMAT_TUNING_TIME_EPS, FORMAT_TUNING_MAP_EPS, SA_ENABLE, SA_T0, SA_ALPHA, SA_FLIP_PROB, ALL_PASSES_RESULT_PATH, BEST_PASS_SUMMARY_PATH, PIM_WEIGHT_CAPACITY_FACTOR
from plan_label import PlanLabel
from scheduler import HEFTScheduler, ScheduledTask
from pathlib import Path
import logging
from config import setup_logging
from task_graph import TaskGraph, TaskNode
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: DEBUG_MAIN)
DEBUG_MAIN = False

def plan_memory_and_label(cfg: Dict, cluster: Cluster) -> PlanLabel:
    """
    Decide PIM mode and effective LRU budget for weights on PIM.
    small:   PIM capacity < KV_total -> kv_in_pim=False, no persistent cache
    medium:  KV_total <= cap < (KV_total + FC_total) -> kv_in_pim=True, weight cache budget = cap - KV_total
    large:   cap >= KV_total + FC_total -> kv_in_pim=True, budget large enough to avoid eviction
    """
    g, shape = build_graph(cfg)
    dtype_bytes = int(DTYPE_BYTES.get(cfg.get('dtype', 'fp16'), 2))
    S = int(cfg.get('prefill_len', 128))
    T = int(cfg.get('decode_len', 32))
    batch = int(cfg.get('batch', 1))
    layers = int(getattr(shape, 'layer_num', 1))
    n_kv_heads = int(getattr(shape, 'n_kv_heads', 1))
    head_dim = int(getattr(shape, 'head_dim', max(1, getattr(shape, 'dim', 1) // max(1, getattr(shape, 'n_heads', 1)))))
    kv_elems = 2 * (S + T) * n_kv_heads * head_dim * batch * layers
    KV_total_bytes = kv_elems * dtype_bytes
    FC_total_bytes = 0
    per_weight_size: Dict[str, int] = {}
    for n in g.nodes.values():
        if getattr(n, 'weight_id', None) and isinstance(n.weight_id, str):
            if n.weight_id.endswith('W1') or n.weight_id.endswith('W2') or n.weight_id.endswith('W3') or n.weight_id.endswith('WQ') or n.weight_id.endswith('WK') or n.weight_id.endswith('WV') or n.weight_id.endswith('WO'):
                FC_total_bytes += int(getattr(n, 'weight_size', 0))
                per_weight_size[n.weight_id] = int(getattr(n, 'weight_size', 0))
    pim_bytes = 0
    for d in cluster.devices_by_type('pim'):
        pim_bytes += int(d.mem_capacity_GB * 1000000000.0)
    if DEBUG_MAIN:
        logger.debug(str(f'[DEBUG] KV total bytes: {KV_total_bytes:,} ({KV_total_bytes / 1000000000.0:.2f} GB)'))
        logger.debug(str(f'[DEBUG] FC total bytes: {FC_total_bytes:,} ({FC_total_bytes / 1000000000.0:.2f} GB)'))
        logger.debug(str(f'[DEBUG] PIM capacity: {pim_bytes:,} ({pim_bytes / 1000000000.0:.2f} GB)'))
        logger.debug(str(f'[DEBUG] Total weights found: {len(per_weight_size)}'))
    if pim_bytes < KV_total_bytes:
        label = PlanLabel(
            pim_mode='small', kv_in_pim=False,
            kv_total_bytes=0,
            pim_weight_capacity_bytes=0
        )
    elif pim_bytes >= KV_total_bytes + FC_total_bytes:
        label = PlanLabel(
            pim_mode='large', kv_in_pim=True,
            kv_total_bytes=int(KV_total_bytes),
            pim_weight_capacity_bytes=int(PIM_WEIGHT_CAPACITY_FACTOR * max(0, pim_bytes - KV_total_bytes))
        )
    else:
        label = PlanLabel(
            pim_mode='medium', kv_in_pim=True,
            kv_total_bytes=int(KV_total_bytes),
            pim_weight_capacity_bytes=int(PIM_WEIGHT_CAPACITY_FACTOR * max(0, pim_bytes - KV_total_bytes))
        )
    if DEBUG_MAIN:
        label.print_debug()
    return label

def _serialize_schedule(schedule: List[ScheduledTask], *, phase: str, token_idx: int | None=None) -> List[Dict]:
    """Convert ScheduledTask list to JSON-friendly dicts."""
    out: List[Dict] = []
    for t in schedule:
        out.append({'node_id': t.node_id, 'device': t.device, 'start': float(t.start), 'finish': float(t.finish), 'duration': float(max(0.0, t.finish - t.start)), 'phase': phase, 'token_idx': token_idx})
    return out

def simulate_prefill(sched: HEFTScheduler, cfg: Dict, graph: TaskGraph) -> tuple[float, List[Dict]]:
    """
    Simulate prefill phase: process entire prefix at once.
    current_length = prefill_len
    """
    prefill_len = int(cfg.get('prefill_len', 128))
    sched.set_seq_len(prefill_len)
    prefill_sched = sched.schedule(graph, phase='prefill')
    prefill_time = sched.makespan(prefill_sched)
    return (prefill_time, _serialize_schedule(prefill_sched, phase='prefill', token_idx=None))

def simulate_decode_progressive(sched: HEFTScheduler, cfg: Dict, graph: TaskGraph, prefill_end: float) -> tuple[float, List[Dict]]:
    """
    Simulate decode phase progressively: one token at a time.
    current_length increases from (prefill_len) to (prefill_len + decode_len - 1)
    """
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len = int(cfg.get('decode_len', 32))
    global_end = prefill_end
    steps_serialized: List[Dict] = []
    for t in range(decode_len):
        logger.debug(str(f'[DEBUG] Simulating decode token {t+1}/{decode_len} (seq_len={prefill_len + t})'))
        current_length = prefill_len + t
        sched.set_seq_len(current_length)
        dec_sched_t = sched.schedule(graph, phase='decode')
        token_end = sched.makespan(dec_sched_t)
        step_time = max(0.0, token_end - global_end)
        steps_serialized.append({'t': t, 'seq_len': current_length, 'end_time': float(token_end), 'delta_time': float(step_time), 'schedule': _serialize_schedule(dec_sched_t, phase='decode', token_idx=t)})
        if token_end > global_end:
            global_end = token_end
    return (max(0.0, global_end - prefill_end), steps_serialized)

def mapping_diff_ratio(a: Dict[str, str], b: Dict[str, str]) -> float:
    """两个权重-格式映射的差异比例（Hamming ratio）。"""
    if not a and (not b):
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    diff = sum((1 for k in keys if a.get(k) != b.get(k)))
    return diff / float(len(keys))

def _sa_make_neighbor_map(base_map: Dict[str, str], weight_ids: List[str], flip_prob: float=0.15) -> Dict[str, str]:
    """
    基于 base_map 生成邻域解：对若干权重随机翻转其存储格式（在 ND / NPU_OPT / PIM_OPT 中切换）。
    至少翻转 1 个权重，避免与 base_map 完全相同。
    """
    CAND = ('ND', 'NPU_OPT', 'PIM_OPT')
    if not weight_ids:
        return dict(base_map)
    out = dict(base_map)
    flips = 0
    for wid in weight_ids:
        if random.random() < max(0.0, min(1.0, flip_prob)):
            old = out.get(wid, base_map.get(wid, 'ND'))
            choices = [x for x in CAND if x != old] or ['ND']
            out[wid] = random.choice(choices)
            flips += 1
    if flips == 0:
        wid = random.choice(weight_ids)
        old = out.get(wid, base_map.get(wid, 'ND'))
        choices = [x for x in CAND if x != old] or ['ND']
        out[wid] = random.choice(choices)
    return out

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_family', type=str, default=DEFAULT_CONFIG['model_family'])
    p.add_argument('--model_variant', type=str, default=DEFAULT_CONFIG['model_variant'])
    p.add_argument('--dtype', type=str, default=DEFAULT_CONFIG['dtype'])
    p.add_argument('--batch', type=int, default=DEFAULT_CONFIG['batch'])
    p.add_argument('--prefill_len', type=int, default=DEFAULT_CONFIG['prefill_len'])
    p.add_argument('--decode_len', type=int, default=DEFAULT_CONFIG['decode_len'])
    p.add_argument('--pim_config_path', type=str, default=DEFAULT_CONFIG['pim_config_path'])
    p.add_argument('--gb_config_path', type=str, default=DEFAULT_CONFIG['gb_config_path'], help='Path to the Global Buffer configuration file')
    p.add_argument('--ramulator_config_path', type=str, default=DEFAULT_CONFIG['ramulator_config_path'])
    p.add_argument('--simulation_log_file', type=str, default='./output/pim_simulation.txt', help='Output log file')
    p.add_argument('--all_passes_json', type=str, default=ALL_PASSES_RESULT_PATH, help='Write all-pass records (schedules/times/weights) to this JSON')
    p.add_argument('--best_summary_json', type=str, default=BEST_PASS_SUMMARY_PATH, help='Write best-pass detail and improvements to this JSON')
    p.add_argument('--debug', action='store_true', help='Enable debug logging')
    p.add_argument('--run_baselines_after', action='store_true',
                   help='Run 3 baselines after finishing your algorithm and print a combined comparison table.')
    p.add_argument('--baseline_out', type=str, default='./output/baseline_compare.json',
            help='Where to save the combined evaluation JSON when --run_baselines_after is set.')
    return p.parse_args()

def run(cfg: Dict):
    cluster = demo_cluster()
    pim_config_path = Path(cfg['pim_config_path'])
    gb_config_path = Path(cfg['gb_config_path'])
    ramulator_config_path = Path(cfg['ramulator_config_path'])
    prefill_len = int(cfg.get('prefill_len', 128))
    batch = int(cfg.get('batch', 1))
    graph, shape = build_graph(cfg)
    logger.debug(str('[Main] Creating shared model dictionary...'))
    model_dict = _make_shared_model_dict(dim=int(getattr(shape, 'dim', 128)), n_heads=int(getattr(shape, 'n_heads', 1)), n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)), ffn_dim=int(getattr(shape, 'ffn_dim', 512)), seqlen=prefill_len)
    logger.debug(str(f"[Main] Model dict created: dim={getattr(shape, 'dim')}, n_heads={getattr(shape, 'n_heads')}, n_kv_heads={getattr(shape, 'n_kv_heads')}, ffn_dim={getattr(shape, 'ffn_dim')}"))
    log_file = Path(cfg.get('simulation_log_file', 'pim_simulation.txt'))
    cost = CostModel(cluster, dtype=cfg.get('dtype', 'fp16'), pim_config_path=pim_config_path, gb_config_path=gb_config_path, ramulator_config_path=ramulator_config_path, simulation_log_file=log_file, debug_traces=False, model_dict=model_dict)
    cost.logger.start_simulation()
    try:
        label = plan_memory_and_label(cfg, cluster)
        fmt_map: Dict[str, str] = {}
        prev_total: float = None
        prev_map: Dict[str, str] = {}
        best_total: float = None
        best_map: Dict[str, str] = {}
        best_pass: int = -1
        last_prefill = 0.0
        last_decode = 0.0
        all_pass_records: List[Dict] = []
        sa_enable = bool(SA_ENABLE)
        T = float(SA_T0)
        alpha = float(SA_ALPHA)
        flip_prob = float(SA_FLIP_PROB)
        buffer_mgr = GlobalMemoryManager()
        for p in range(1, FORMAT_TUNING_MAX_PASSES + 1):
            logger.debug(str(f"\n{'=' * 80}"))
            logger.debug(str(f'Starting optimization pass {p}/{FORMAT_TUNING_MAX_PASSES}'))
            logger.debug(str(f"{'=' * 80}\n"))
            sched = HEFTScheduler(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
            sched.set_storage_format_map(fmt_map)
            sched.reset_state()
            t_wall0 = time.time()
            logger.debug(str(f'[PASS{p}] Running prefill phase (current map)...'))
            prefill_time, prefill_sched_ser = simulate_prefill(sched, cfg, graph)
            logger.debug(str(f'[PASS{p}] Running decode phase (current map)...'))
            decode_time, decode_steps_ser = simulate_decode_progressive(sched, cfg, graph, prefill_end=prefill_time)
            total_time = prefill_time + decode_time
            wall_time_current = time.time() - t_wall0
            cost.logger._log(f'[PASS{p}] CurrentMap  WallTime: {wall_time_current:.3f}s | Prefill(sim): {prefill_time:.6f}s, Decode(sim): {decode_time:.6f}s, Total(sim): {total_time:.6f}s')
            fmt_suggestion = sched.suggest_weight_storage_formats()
            wids = sorted(set(list(fmt_suggestion.keys()) + list(fmt_map.keys())))
            if sa_enable:
                neighbor_map = _sa_make_neighbor_map(fmt_suggestion or fmt_map, wids, flip_prob)
                sched2 = HEFTScheduler(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
                sched2.set_storage_format_map(neighbor_map)
                sched2.reset_state()
                t_wall1 = time.time()
                logger.debug(str(f'[PASS{p}] Running prefill phase (neighbor map)...'))
                prefill_time_nb, prefill_sched_ser_nb = simulate_prefill(sched, cfg, graph)
                logger.debug(str(f'[PASS{p}] Running decode phase (neighbor map)...'))
                decode_time_nb, decode_steps_ser_nb = simulate_decode_progressive(sched, cfg, graph, prefill_end=prefill_time)
                total_time_nb = prefill_time_nb + decode_time_nb
                wall_time_neighbor = time.time() - t_wall1
                cost.logger._log(f'[PASS{p}] NeighborMap WallTime: {wall_time_neighbor:.3f}s | Prefill(sim): {prefill_time_nb:.6f}s, Decode(sim): {decode_time_nb:.6f}s, Total(sim): {total_time_nb:.6f}s')
                delta = total_time_nb - total_time
                accept = delta < 0.0 or random.random() < math.exp(-max(0.0, delta) / max(1e-09, T))
                fmt_next = neighbor_map if accept else fmt_map
                chosen_total = total_time_nb if accept else total_time
                chosen_prefill = prefill_time_nb if accept else prefill_time
                chosen_decode = decode_time_nb if accept else decode_time
                prefill_sched_ser = prefill_sched_ser_nb if accept else prefill_sched_ser
                decode_steps_ser = decode_steps_ser_nb if accept else decode_steps_ser
                cost.logger._log(f"[PASS{p}] SA decision: {('ACCEPT' if accept else 'REJECT')} (Δ={delta:+.6f}, T={T:.4f})")
            else:
                fmt_next = fmt_suggestion
                chosen_total = total_time
                chosen_prefill = prefill_time
                chosen_decode = decode_time
            if best_total is None or chosen_total < best_total:
                best_total = chosen_total
                best_map = dict(fmt_next)
                best_pass = p
            os.makedirs(os.path.dirname(WEIGHT_FORMAT_JSON_PATH), exist_ok=True)
            with open(WEIGHT_FORMAT_JSON_PATH, 'w') as f:
                json.dump(fmt_next, f, indent=2, sort_keys=True)
            logger.debug(str(f'[INFO] Accepted weight storage map saved: {WEIGHT_FORMAT_JSON_PATH}'))
            weight_stats = sched.export_weight_stats()
            all_pass_records.append({'pass': p, 'times': {'prefill': float(prefill_time), 'decode': float(decode_time), 'total': float(total_time)}, 'schedules': {'prefill': prefill_sched_ser, 'decode_steps': decode_steps_ser}, 'formats': {'used_storage_map': dict(fmt_map or {}), 'suggested_storage_map': dict(fmt_suggestion or {}), 'suggestion_json_path': WEIGHT_FORMAT_JSON_PATH}, 'weights': weight_stats})
            if prev_total is not None:
                time_improve = prev_total - chosen_total
                map_delta = mapping_diff_ratio(prev_map, fmt_next)
                logger.debug(str(f'\n[DELTA] Time improvement (accepted vs prev_accepted): {time_improve:+.6f}s'))
                logger.debug(str(f'[DELTA] Format map change ratio (accepted vs prev_accepted): {map_delta:.4f}'))
                if abs(time_improve) <= FORMAT_TUNING_TIME_EPS and map_delta <= FORMAT_TUNING_MAP_EPS:
                    logger.debug(str(f'\n[CONVERGENCE] Optimization converged at pass {p}'))
                    logger.debug(str(f'  Time delta: {abs(time_improve):.6e}s <= {FORMAT_TUNING_TIME_EPS:.6e}s'))
                    logger.debug(str(f'  Map delta:  {map_delta:.4f} <= {FORMAT_TUNING_MAP_EPS:.4f}'))
                    last_prefill, last_decode = (chosen_prefill, chosen_decode)
                    prev_total, prev_map, fmt_map = (chosen_total, dict(fmt_next), dict(fmt_next))
                    break
            last_prefill, last_decode = (chosen_prefill, chosen_decode)
            prev_total = chosen_total
            prev_map = dict(fmt_next)
            fmt_map = dict(fmt_next)
            if sa_enable:
                T *= alpha
        logger.debug(str(f"\n{'=' * 80}"))
        logger.debug(str(f'Optimization Complete'))
        logger.debug(str(f"{'=' * 80}"))
        if best_map:
            with open(WEIGHT_FORMAT_JSON_PATH, 'w') as f:
                json.dump(best_map, f, indent=2, sort_keys=True)
            logger.debug(str(f'[INFO] Best weight storage map (found at pass {best_pass}) saved to: {WEIGHT_FORMAT_JSON_PATH}'))
        logger.debug(str(f'Best total time: {best_total:.6f}s (at pass {best_pass})'))
        logger.debug(str(f'Last accepted prefill(sim): {last_prefill:.6f}s'))
        logger.debug(str(f'Last accepted decode(sim):  {last_decode:.6f}s'))
        pkl_dir = Path('./pkl')
        pkl_dir.mkdir(parents=True, exist_ok=True)
        out_dir = Path('./output')
        out_dir.mkdir(parents=True, exist_ok=True)
        all_path = Path(cfg.get('all_passes_json', ALL_PASSES_RESULT_PATH))
        best_path = Path(cfg.get('best_summary_json', BEST_PASS_SUMMARY_PATH))
        with open(all_path, 'w', encoding='utf-8') as f:
            json.dump({'config': {'model_family': cfg.get('model_family'), 'model_variant': cfg.get('model_variant'), 'dtype': cfg.get('dtype'), 'batch': cfg.get('batch'), 'prefill_len': cfg.get('prefill_len'), 'decode_len': cfg.get('decode_len')}, 'passes': all_pass_records}, f, ensure_ascii=False, indent=2)
        logger.debug(str(f'[REPORT] All passes saved to: {all_path}'))
        if all_pass_records:
            best_idx = min(range(len(all_pass_records)), key=lambda i: all_pass_records[i]['times']['total'])
            best_rec = all_pass_records[best_idx]
            best_total = best_rec['times']['total']
            improvements = []
            for rec in all_pass_records:
                tot = rec['times']['total']
                delta = float(tot - best_total)
                pct = delta / tot * 100.0 if tot > 0 else 0.0
                improvements.append({'pass': rec['pass'], 'total_time': float(tot), 'delta_seconds_vs_best': float(delta), 'delta_percent_vs_that_pass': float(pct)})
            with open(best_path, 'w', encoding='utf-8') as f:
                json.dump({'best_pass': best_rec['pass'], 'best_times': best_rec['times'], 'best_formats': best_rec['formats'], 'best_weights': best_rec['weights'], 'prefill_schedule': best_rec['schedules']['prefill'], 'decode_steps': best_rec['schedules']['decode_steps'], 'improvements_vs_each_pass': improvements}, f, ensure_ascii=False, indent=2)
            logger.debug(str(f'[REPORT] Best pass summary saved to: {best_path}'))
    finally:
        cost.logger.end_simulation()
        cost.logger.close()

# ===== Baseline helpers (inlined) =====
_ATTENTION_KEYS = {
    'q','k','v','o','qk','sv','softmax','attn','attention','attn_softmax','q_proj','k_proj','v_proj','wo_proj'
}

def _is_attention_node(n: TaskNode) -> bool:
    name = (n.name or '').lower()
    if name in _ATTENTION_KEYS:
        return True
    return any(k in name for k in _ATTENTION_KEYS)

def _clone_graph(g: TaskGraph) -> TaskGraph:
    """Deep copy TaskGraph nodes + edges, to safely override `allowed`."""
    new_g = TaskGraph()
    for _, n in g.nodes.items():
        new_n = TaskNode(
            id=n.id, name=n.name,
            flops=n.flops, bytes_read=n.bytes_read, bytes_write=n.bytes_write,
            weight_id=n.weight_id, weight_size=n.weight_size,
            allowed=dict(n.allowed) if isinstance(n.allowed, dict) else {},
            attrs=dict(n.attrs) if isinstance(n.attrs, dict) else {},
        )
        new_g.add_node(new_n)
    succ = getattr(g, 'succ', None)
    if isinstance(succ, dict) and succ:
        for u, nbrs in succ.items():
            for v in nbrs:
                new_g.add_edge(u, v)
    else:
        try:
            _ = g.topological()  # 若实现存在，触发拓扑构建；没有也不影响
        except Exception:
            pass
    return new_g

def _apply_policy_on_graph(g: TaskGraph, policy: str, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if policy == 'pd':
        use_npu = (phase == 'prefill')
        for _, n in g2.nodes.items():
            n.allowed['npu'] = bool(use_npu)
            n.allowed['pim'] = not use_npu
            n.allowed['cpu'] = bool(n.allowed.get('cpu', True))
        return g2

    if policy == 'weights_on_pim':
        for _, n in g2.nodes.items():
            has_w = n.weight_id is not None and (n.weight_size or 0) > 0
            n.allowed['pim'] = bool(has_w)
            n.allowed['npu'] = not has_w
            n.allowed['cpu'] = bool(n.allowed.get('cpu', True))
        return g2

    if policy == 'attn_on_pim':
        for _, n in g2.nodes.items():
            is_attn = _is_attention_node(n)
            n.allowed['pim'] = bool(is_attn)
            n.allowed['npu'] = not is_attn
            n.allowed['cpu'] = bool(n.allowed.get('cpu', True))
        return g2

    raise ValueError(f'Unknown policy: {policy}')

def _run_phase_once(sched, graph: TaskGraph, *, phase: str, seq_len: int) -> float:
    sched.set_seq_len(seq_len)
    schedule = sched.schedule(graph, phase=phase)
    return sched.makespan(schedule)

def _decode_progressive(sched, graph: TaskGraph, *, prefill_len: int, decode_len: int) -> float:
    """逐 token 递增 seq_len 的 decode 累加时延。"""
    total = 0.0
    for t in range(decode_len):
        cur_len = prefill_len + t
        sched.reset_state()
        total += _run_phase_once(sched, graph, phase='decode', seq_len=cur_len)
    return total

def _parse_algo_best_times(cfg: Dict) -> Dict:
    """从你的主流程产出的 all_passes_json 里，读取 best pass 的 prefill/decode/total。"""
    all_path = cfg.get('all_passes_json', None) or './output/all_passes_results.json'
    with open(all_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    passes = payload.get('passes', [])
    if not passes:
        raise RuntimeError('No passes found in all_passes_results.json; make sure your main pipeline produced it.')
    best_idx = min(range(len(passes)), key=lambda i: passes[i].get('times', {}).get('total', float('inf')))
    best = passes[best_idx]
    t_prefill = float(best.get('times', {}).get('prefill', 0.0))
    t_decode  = float(best.get('times', {}).get('decode', 0.0))
    return {
        'policy': 'algo',
        'prefill_time_s': t_prefill,
        'decode_time_s':  t_decode,
        'total_time_s':   t_prefill + t_decode,
    }

def _eval_one_baseline(cfg: Dict, policy: str) -> Dict:
    """
    构建 cluster/cost/graph/label，然后按policy约束allowed，
    分别评估 prefill 与 progressive decode。
    """
    # Reset the global simulation logger before each baseline run
    reset_simulation_logger()
    
    # --- Build environment（与主流程口径一致） ---
    cluster = demo_cluster()
    # 构图 + 形状
    graph, shape = build_graph(cfg)
    batch = int(cfg.get('batch', getattr(shape, 'batch', 1)))
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len = int(cfg.get('decode_len', 32))

    # PIM trace 需要的 model_dict（若你的 CostModel 用不到也无妨）
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, 'dim', 128)),
        n_heads=int(getattr(shape, 'n_heads', 1)),
        n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)),
        ffn_dim=int(getattr(shape, 'ffn_dim', 512)),
        seqlen=prefill_len
    )

    cost = CostModel(
        cluster,
        dtype=cfg.get('dtype', 'fp16'),
        pim_config_path=Path(cfg.get('pim_config_path')),
        gb_config_path=Path(cfg.get('gb_config_path')),
        ramulator_config_path=Path(cfg.get('ramulator_config_path')),
        simulation_log_file=Path(cfg.get('simulation_log_file', './output/pim_simulation.txt')),
        model_dict=model_dict
    )
    
    # Start simulation logger for this baseline
    cost.logger.start_simulation()
    
    try:
        label = plan_memory_and_label(cfg, cluster)

        buffer_mgr = GlobalMemoryManager()
        sched = HEFTScheduler(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)

        # --- Prefill：根据 policy 约束 allowed 后评估 ---
        g_prefill = _apply_policy_on_graph(graph, policy, phase='prefill')
        sched.reset_state()
        t_prefill = _run_phase_once(sched, g_prefill, phase='prefill', seq_len=prefill_len)

        # --- Decode：根据 policy 约束 allowed 后 progressive 累加 ---
        g_decode = _apply_policy_on_graph(graph, policy, phase='decode')
        t_decode = _decode_progressive(sched, g_decode, prefill_len=prefill_len, decode_len=decode_len)

        return {
            'policy': policy,
            'prefill_time_s': float(t_prefill),
            'decode_time_s': float(t_decode),
            'total_time_s': float(t_prefill + t_decode),
            'batch': batch,
            'prefill_len': prefill_len,
            'decode_len': decode_len,
        }
    finally:
        # Clean up logger for this baseline
        cost.logger.end_simulation()
        cost.logger.close()

def _print_combined_table(results: List[Dict], cfg: Dict, wall: float) -> None:
    print("\n=== Combined Evaluation (Your Algo + Baselines) ===")
    print(f"Config: family={cfg['model_family']} variant={cfg['model_variant']} batch={cfg['batch']} dtype={cfg['dtype']}")
    print(f"prefill_len={cfg['prefill_len']} decode_len={cfg['decode_len']}   (sim wall={wall:.2f}s)\n")
    header = f"{'Policy':<18} {'Prefill(s)':>12} {'Decode(s)':>12} {'Total(s)':>12} {'vs Algo':>10} {'vs PD':>10}"
    print(header)
    print('-' * len(header))
    for r in results:
        s_algo = (f"{r.get('speedup_vs_algo', None):.3f}x" if r.get('speedup_vs_algo') is not None else '-')
        s_pd   = (f"{r.get('speedup_vs_pd', None):.3f}x"   if r.get('speedup_vs_pd')   is not None else '-')
        print(f"{r['policy']:<18} {r['prefill_time_s']:>12.4f} {r['decode_time_s']:>12.4f} {r['total_time_s']:>12.4f} {s_algo:>10} {s_pd:>10}")


def main():
    args = parse_args()
    setup_logging(args.debug)
    cfg = {'model_family': args.model_family, 'model_variant': args.model_variant, 'dtype': args.dtype, 'batch': args.batch, 'prefill_len': args.prefill_len, 'decode_len': args.decode_len, 'pim_config_path': args.pim_config_path, 'gb_config_path': args.gb_config_path, 'ramulator_config_path': args.ramulator_config_path, 'simulation_log_file': args.simulation_log_file, 'all_passes_json': args.all_passes_json, 'best_summary_json': args.best_summary_json}
    run(cfg)

    if args.run_baselines_after:
        t0 = time.time()
        algo_res = _parse_algo_best_times(cfg)
        algo_res.update({
            'batch': cfg['batch'],
            'prefill_len': cfg['prefill_len'],
            'decode_len': cfg['decode_len'],
        })

        baseline_names = ['pd', 'weights_on_pim', 'attn_on_pim']
        results = [algo_res]
        for name in baseline_names:
            results.append(_eval_one_baseline(cfg, name))
        wall = time.time() - t0

        idx_algo = 0
        idx_pd = next((i for i, r in enumerate(results) if r['policy'] == 'pd'), None)
        for r in results:
            base_algo = results[idx_algo]['total_time_s']
            r['speedup_vs_algo'] = (base_algo / r['total_time_s']) if r['total_time_s'] > 0 else None
            if idx_pd is not None:
                base_pd = results[idx_pd]['total_time_s']
                r['speedup_vs_pd'] = (base_pd / r['total_time_s']) if r['total_time_s'] > 0 else None

        try:
            os.makedirs(os.path.dirname(args.baseline_out), exist_ok=True)
        except Exception:
            pass
        with open(args.baseline_out, 'w', encoding='utf-8') as f:
            json.dump({'config': cfg, 'results': results, 'wall_time_s': wall}, f, ensure_ascii=False, indent=2)

        _print_combined_table(results, cfg, wall)
        print(f"Saved baseline comparison to: {args.baseline_out}")

if __name__ == '__main__':
    main()