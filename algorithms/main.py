from __future__ import annotations
from config import attach_local_debug_filter
import argparse
import json
import os
import time
import math
import random
from typing import Dict, List, Callable
from hardware import demo_cluster, Cluster
from cost_model import CostModel, DTYPE_BYTES, _make_shared_model_dict, reset_simulation_logger
from buffer_manager import GlobalMemoryManager
from model_parser import build_graph
from config import DEFAULT_CONFIG, WEIGHT_FORMAT_JSON_PATH, FORMAT_TUNING_MAX_PASSES, FORMAT_TUNING_TIME_EPS, FORMAT_TUNING_MAP_EPS, SA_ENABLE, SA_T0, SA_ALPHA, SA_FLIP_PROB, ALL_PASSES_RESULT_PATH, BEST_PASS_SUMMARY_PATH, PIM_WEIGHT_CAPACITY_FACTOR,setup_logging
from plan_label import PlanLabel
from scheduler import HEFTScheduler, ScheduledTask, SimulatedAnnealingScheduler, GeneticScheduler, RLScheduler, AStarBeamScheduler
from pathlib import Path
import logging
from task_graph import TaskGraph, TaskNode
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: True)

# ---- path helper (unify result_dir naming incl. batch) ----
def _build_result_dir(cfg: Dict, default_root: str = './output') -> Path:
    """
    Compose a result directory path that always includes batch:
      <base>/<family>_<variant>_<dtype>_b<batch>
    """
    base   = cfg.get('result_dir') or default_root
    family = cfg.get('model_family', 'unnamed')
    variant= cfg.get('model_variant', '')
    dtype  = cfg.get('dtype', 'fp16')
    batch  = int(cfg.get('batch', 1))
    return Path(base) / f"{family}_{variant}_{dtype}_b{batch}"

def _build_tag(cfg: dict) -> str:
    """Build a safe tag for output files: SxT + optional stride if provided."""
    try:
        S = int(cfg.get('prefill_len', 0) or 0)
    except Exception:
        S = 0
    try:
        T = int(cfg.get('decode_len', 0) or 0)
    except Exception:
        T = 0
    parts = [f"{S}x{T}"]
    st = cfg.get('decode_sample_stride', None)
    try:
        if st is not None:
            stv = int(st)
            if stv > 0:
                parts.append(f"st{stv}")
    except Exception:
        pass
    return "_".join(parts)

def plan_memory_and_label(cfg: Dict, cluster: Cluster) -> PlanLabel:
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
    for n in g.nodes.values():
        if getattr(n, 'weight_id', None) and isinstance(n.weight_id, str):
            if n.weight_id.endswith(('W1','W2','W3','WQ','WK','WV','WO')):
                FC_total_bytes += int(getattr(n, 'weight_size', 0))
    pim_bytes = sum(int(d.mem_capacity_GB * (1024**3)) for d in cluster.devices_by_type('pim'))
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
  
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len  = int(cfg.get('decode_len', 32))
    global_end  = float(prefill_end)
    steps_serialized: List[Dict] = []

    # 取有效 stride（cfg 优先，其次 DEFAULT_CONFIG，最后 16）
    try:
        from config import DEFAULT_CONFIG
        default_stride = int(DEFAULT_CONFIG.get('decode_sample_stride', 16))
    except Exception:
        default_stride = 32
    stride = default_stride
    if isinstance(cfg, dict):
        try:
            stride = int(cfg.get('decode_sample_stride', default_stride))
        except Exception:
            stride = default_stride
    # 精确仿真
    if stride <= 1:
        for t in range(decode_len):
            cur_len = prefill_len + t
            sched.set_seq_len(cur_len)
            dec_sched = sched.schedule(graph, phase='decode')
            token_end = float(sched.makespan(dec_sched))
            step_time = max(0.0, token_end - global_end)
            global_end = token_end
            steps_serialized.append({
                't': t, 'seq_len': cur_len, 'step_time': float(step_time),
                'estimated': False, 'schedule': _serialize_schedule(dec_sched, phase='decode', token_idx=t)
            })
        return (float(global_end - prefill_end), steps_serialized)

    # 采样
    last_sample_time: float | None = None
    last_sample_t: int = -1
    def _advance_to(t_target: float):
        try:
            for name in list(getattr(sched, 'avail', {}).keys()):
                sched.avail[name] = max(float(sched.avail.get(name, 0.0)), float(t_target))
        except Exception:
            pass
        try:
            tl = getattr(getattr(sched, 'comm', None), 'timeline_end', None)
            if isinstance(tl, dict):
                for k in list(tl.keys()):
                    tl[k] = max(float(tl.get(k, 0.0)), float(t_target))
        except Exception:
            pass
    for t in range(decode_len):
        cur_len = prefill_len + t
        is_last = (t == decode_len - 1)
        do_sample = (t % stride == 0) or is_last
        if do_sample:
            if last_sample_time is not None and last_sample_t >= 0 and t > last_sample_t + 1:
                gap = t - last_sample_t - 1
                if gap > 0:
                    global_end += float(last_sample_time) * float(gap)
                    for u in range(last_sample_t + 1, t):
                        steps_serialized.append({'t': u, 'seq_len': prefill_len + u, 'step_time': float(last_sample_time), 'estimated': True, 'schedule': None})
            _advance_to(global_end)
            sched.set_seq_len(cur_len)
            dec_sched = sched.schedule(graph, phase='decode')
            token_end = float(sched.makespan(dec_sched))
            step_time = max(0.0, token_end - global_end)
            global_end = token_end
            last_sample_time = float(step_time)
            last_sample_t = t
            steps_serialized.append({'t': t, 'seq_len': cur_len, 'step_time': float(step_time), 'estimated': False, 'schedule': _serialize_schedule(dec_sched, phase='decode', token_idx=t)})
    return (float(global_end - prefill_end), steps_serialized)

def _parse_algos(raw: str) -> list[str]:
    """
    Parse comma/space separated algo names into a unique, ordered list:
    e.g. "heft,sa, rl ,ga,astar" -> ["heft","sa","rl","ga","astar"]
    """
    raw = (raw or 'heft')
    parts = []
    for token in raw.replace(',', ' ').split():
        t = token.strip().lower()
        if t:
            parts.append(t)
    # de-duplicate but keep order
    seen = set()
    uniq = []
    for t in parts:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def _make_scheduler(name: str, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
    from config import (SCHED_SA_ITERS, SCHED_SA_T0, SCHED_SA_ALPHA, SCHED_SA_FLIP_PROB,
                        SCHED_GA_POP, SCHED_GA_GENS, SCHED_GA_ELITE, SCHED_GA_MUT_PROB, SCHED_GA_CROSS_PROB,
                        SCHED_RL_EPISODES, SCHED_RL_EPS0, SCHED_RL_EPSE, SCHED_RL_ALPHA, SCHED_RL_GAMMA,
                        SCHED_ASTAR_BEAM, SCHED_ASTAR_MAX_EXPANSIONS)
    name = (name or 'heft').strip().lower()
    if name in ('heft','heft+greedy','greedy'):
        return HEFTScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)
    if name in ('sa','anneal','simulated_annealing'):
        return SimulatedAnnealingScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer,
                                           sa_iters=SCHED_SA_ITERS, T0=SCHED_SA_T0, alpha=SCHED_SA_ALPHA, flip_prob=SCHED_SA_FLIP_PROB)
    if name in ('ga','genetic'):
        return GeneticScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer,
                                pop=SCHED_GA_POP, gens=SCHED_GA_GENS, elite=SCHED_GA_ELITE, mut_prob=SCHED_GA_MUT_PROB, cross_prob=SCHED_GA_CROSS_PROB)
    if name in ('rl','bandit'):
        return RLScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer,
                           episodes=SCHED_RL_EPISODES, epsilon_start=SCHED_RL_EPS0, epsilon_end=SCHED_RL_EPSE, alpha=SCHED_RL_ALPHA, gamma=SCHED_RL_GAMMA)
    if name in ('astar','a*','a_star'):
        return AStarBeamScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer,
                                  beam=SCHED_ASTAR_BEAM, max_expansions=SCHED_ASTAR_MAX_EXPANSIONS)
    raise ValueError(f"Unknown scheduler strategy: {name}")

def mapping_diff_ratio(a: Dict[str, str], b: Dict[str, str]) -> float:
    if not a and (not b):
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    diff = sum((1 for k in keys if a.get(k) != b.get(k)))
    return diff / float(len(keys))

def _sa_make_neighbor_map(base_map: Dict[str, str], weight_ids: List[str], flip_prob: float=0.15) -> Dict[str, str]:

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

def run(cfg: Dict):
    result_dir = Path(cfg.get('result_dir') or _build_result_dir(cfg, './output/weight_suggestions'))
    result_dir.mkdir(parents=True, exist_ok=True)
    weight_format_path = Path(cfg.get('weight_format_json') or (result_dir / 'weight_storage_suggestion.json'))

    cluster = demo_cluster()
    pim_config_path = Path(cfg['pim_config_path'])
    gb_config_path = Path(cfg['gb_config_path'])
    ramulator_config_path = Path(cfg['ramulator_config_path'])
    prefill_len = int(cfg.get('prefill_len', 128))
    batch = int(cfg.get('batch', 1))
    graph, shape = build_graph(cfg)
    model_dict = _make_shared_model_dict(dim=int(getattr(shape, 'dim', 128)), n_heads=int(getattr(shape, 'n_heads', 1)), n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)), ffn_dim=int(getattr(shape, 'ffn_dim', 512)), seqlen=prefill_len)
    sim_log_file = cfg.get('simulation_log_file', str(result_dir / 'pim_simulation.txt'))
    cost = CostModel(cluster, dtype=cfg.get('dtype', 'fp16'), pim_config_path=pim_config_path, gb_config_path=gb_config_path, ramulator_config_path=ramulator_config_path,  simulation_log_file=sim_log_file, debug_traces=False, model_dict=model_dict)
    cost.logger.start_simulation()
    try:
        label = plan_memory_and_label(cfg, cluster)
        fmt_map: Dict[str, str] = {}
        prev_total: float|None = None
        prev_map: Dict[str, str] = {}
        best_total: float|None = None
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
                prefill_time_nb, prefill_sched_ser_nb = simulate_prefill(sched2, cfg, graph)
                logger.debug(str(f'[PASS{p}] Running decode phase (neighbor map)...'))
                decode_time_nb, decode_steps_ser_nb = simulate_decode_progressive(sched2, cfg, graph, prefill_end=prefill_time)
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

            # >>> dump per-op and per-link CSV stats for this pass
            try:
                ops_csv = Path(result_dir) / f"pass_{p:02d}_ops.csv"
                comms_csv = Path(result_dir) / f"pass_{p:02d}_comms.csv"
                getattr(sched, 'stats', None) and sched.stats.dump_csv(ops_csv, comms_csv)
            except Exception:
                pass
            # <<< end stats dump

            if best_total is None or chosen_total < best_total:
                best_total = chosen_total
                best_map = dict(fmt_next)
                best_pass = p
            os.makedirs(os.path.dirname(weight_format_path), exist_ok=True)
            with open(weight_format_path, 'w') as f:
                json.dump(fmt_next, f, indent=2, sort_keys=True)
            logger.debug(str(f'[INFO] Accepted weight storage map saved: {weight_format_path}'))
            weight_stats = sched.export_weight_stats()
            all_pass_records.append({'pass': p, 'times': {'prefill': float(prefill_time), 'decode': float(decode_time), 'total': float(total_time)}, 'schedules': {'prefill': prefill_sched_ser, 'decode_steps': decode_steps_ser}, 'formats': {'used_storage_map': dict(fmt_map or {}), 'suggested_storage_map': dict(fmt_suggestion or {}), 'suggestion_json_path': str(weight_format_path)}, 'weights': weight_stats})
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
            with open(weight_format_path, 'w') as f:
                json.dump(best_map, f, indent=2, sort_keys=True)
            logger.debug(str(f'[INFO] Best weight storage map (found at pass {best_pass}) saved to: {weight_format_path}'))
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


# ===== Baseline registry and paper baselines =====
from typing import Callable

_BASELINE_REGISTRY: Dict[str, Callable[[TaskGraph], TaskGraph]] = {}

def register_baseline(name: str):
    name = (name or "").strip().lower()
    def _deco(fn):
        _BASELINE_REGISTRY[name] = fn
        return fn
    return _deco

def _is_op(n: TaskNode, *tags: str) -> bool:
    op = str(getattr(n, 'attrs', {}).get('op') or n.name or '').upper()
    return any(tag.upper() in op for tag in tags)

def _arith_intensity(n: TaskNode) -> float:
    bytes_total = float(getattr(n, 'bytes_read', 0.0) + getattr(n, 'bytes_write', 0.0)) + 1e-9
    return float(getattr(n, 'flops', 0.0)) / bytes_total

def _is_kv_rw(n: TaskNode) -> bool:
    nm = (n.name or '').lower()
    op = str(getattr(n, 'attrs', {}).get('op') or '').lower()
    return any(k in nm or k in op for k in ('kv_read','kv_write','k_cache','v_cache'))

def _is_gemv_like(n: TaskNode, *, phase: str) -> bool:
    op = str(getattr(n, 'attrs', {}).get('op') or n.name or '').upper()
    if phase == 'decode' and any(t in op for t in ['Q','K','V','O','FFN_W1','FFN_W2','GELU']):
        return True
    return str(getattr(n, 'attrs', {}).get('arith_op') or '').lower() == 'gemv'


@register_baseline('ianus')
def _baseline_ianus(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        # 全 NPU
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2

    # decode：按规则划分
    for _, n in g2.nodes.items():
        if _is_op(n, 'QK', 'SV', 'FFN_W1', 'GELU', 'FFN_W2'):
            on_pim = True
        elif _is_op(n, 'Q', 'K', 'SOFTMAX', 'NORM', 'ADD'):
            on_pim = False
        else:
            on_pim = (_arith_intensity(n) < 4.0)  # 兜底：AI<4 走 PIM
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2

@register_baseline('neupims')
def _baseline_neupims(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    for _, n in g2.nodes.items():
        if _is_op(n, 'SOFTMAX', 'NORM', 'ADD'):
            on_pim = False
        elif _is_kv_rw(n):
            on_pim = True
        else:
            inten = _arith_intensity(n)
            on_pim = (inten < 4.0) or _is_gemv_like(n, phase=phase)
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2

@register_baseline('attacc')
def _baseline_attacc(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2
    for _, n in g2.nodes.items():
        on_pim = _is_op(n, 'QK', 'SV') or _is_kv_rw(n)
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2

@register_baseline('facil')
def _baseline_facil(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2
    for _, n in g2.nodes.items():
        if _is_op(n, 'SOFTMAX', 'NORM', 'ADD'):
            on_pim = False
        else:
            on_pim = _is_gemv_like(n, phase='decode') or _is_op(n, 'QK', 'SV')
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2


def _run_phase_once(sched, graph: TaskGraph, *, phase: str, seq_len: int) -> float:
    sched.set_seq_len(seq_len)
    schedule = sched.schedule(graph, phase=phase)
    return sched.makespan(schedule)

def _decode_progressive(sched, graph: TaskGraph, *, prefill_len: int, decode_len: int, cfg=None, policy_name: str = 'baseline') -> float:
    logger = logging.getLogger(__name__)
   
    try:
        from config import DEFAULT_CONFIG
        default_stride = int(DEFAULT_CONFIG.get('decode_sample_stride', 16))
    except Exception:
        default_stride = 32
    stride = default_stride

    if stride <= 1:
        total = 0.0
        for t in range(decode_len):
            cur_len = prefill_len + t
            sched.reset_state()
            step = _run_phase_once(sched, graph, phase='decode', seq_len=cur_len)
            total += float(step)
        return float(total)

    total = 0.0
    D = int(decode_len); P = int(prefill_len)
    for t0 in range(0, D, stride):
        cur_len = P + t0
        blk = min(stride, D - t0)
        sched.reset_state()
        step = _run_phase_once(sched, graph, phase='decode', seq_len=cur_len)
        add  = float(step) * float(blk)
        total += add
    logger.debug(f"[baseline:{policy_name}] decode done (sampled stride={stride}). total={float(total):.6f}s")
    return float(total)


def _eval_one_baseline(cfg: Dict, policy: str) -> Dict:
    """
    Run one baseline policy and return both timing and full schedules
    (prefill schedule + sampled decode steps) so downstream visualization
    matches the weight-suggestion style best_summary.json.
    """
    reset_simulation_logger()
    cluster = demo_cluster()
    graph, shape = build_graph(cfg)
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len  = int(cfg.get('decode_len', 32))
    batch = int(cfg.get('batch', 1))

    base_dir = Path(cfg['result_dir'])
    algo_dir = base_dir / f"algo_{policy}"
    algo_dir.mkdir(parents=True, exist_ok=True)

    try:
        setup_logging(bool(cfg.get('debug', False)), log_file=str(algo_dir / f"debug.txt"))
    except Exception:
        pass

    logger = logging.getLogger(__name__)
    try:
        _stride_for_log = int(cfg.get('decode_sample_stride'))
    except Exception:
        try:
            from config import DEFAULT_CONFIG
            _stride_for_log = int(DEFAULT_CONFIG.get('decode_sample_stride', 16))
        except Exception:
            _stride_for_log = None
    

    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, 'dim', 128)),
        n_heads=int(getattr(shape, 'n_heads', 1)),
        n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)),
        ffn_dim=int(getattr(shape, 'ffn_dim', 512)),
        seqlen=prefill_len,
    )
    cost = CostModel(
        cluster=cluster,
        dtype=cfg.get('dtype', 'fp16'),
        pim_config_path=Path(cfg.get('pim_config_path')),
        gb_config_path=Path(cfg.get('gb_config_path')),
        ramulator_config_path=Path(cfg.get('ramulator_config_path')),
        simulation_log_file=Path(cfg.get('simulation_log_file', './output/pim_simulation.txt')),
        model_dict=model_dict,
    )
    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    # Apply policy to graphs
    g_prefill = (_BASELINE_REGISTRY[policy](graph, phase='prefill') if policy in _BASELINE_REGISTRY else _apply_policy_on_graph(graph, policy, phase='prefill'))
    g_decode  = (_BASELINE_REGISTRY[policy](graph, phase='decode')  if policy in _BASELINE_REGISTRY else _apply_policy_on_graph(graph, policy, phase='decode'))

    # Make scheduler
    label = plan_memory_and_label(cfg, cluster)
    buffer_mgr = GlobalMemoryManager()
    sched = _make_scheduler('heft', cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
    try:
        sched.set_storage_format_map({})
    except Exception:
        pass

    # Prefill (full schedule)
    t_prefill, prefill_sched_ser = simulate_prefill(sched, cfg, g_prefill)

    # Decode (sampled / full schedule depending on stride)
    t_decode, decode_steps_ser = simulate_decode_progressive(sched, cfg, g_decode, prefill_end=t_prefill)
    try:
        ops_csv   = algo_dir / "ops.csv"
        comms_csv = algo_dir / "comms.csv"
        if getattr(sched, 'stats', None):
            sched.stats.dump_csv(ops_csv, comms_csv)
    except Exception as e:
        logger.debug(f"[stats] CSV dump skipped: {e}")

    logger.debug(f"[Baseline] Done policy='{policy}': Prefill={float(t_prefill):.6f}s Decode={float(t_decode):.6f}s Total={float(t_prefill + t_decode):.6f}s")

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    # Return a rich result; the caller (evaluate_suite) will persist via _save_best_json
    return {
        'policy': policy,
        'prefill_time_s': float(t_prefill),
        'decode_time_s': float(t_decode),
        'total_time_s': float(t_prefill + t_decode),
        'batch': batch,
        'prefill_len': prefill_len,
        'decode_len': decode_len,
        'prefill_schedule': prefill_sched_ser,
        'decode_steps': decode_steps_ser,
    }

def _run_strategy_once(strategy: str, cfg: Dict, *, shared_graph=None, shared_shape=None) -> Dict:
    """Build env, run one scheduling strategy (prefill + progressive decode) and return timing."""
    logger.debug(f"\n{'='*60}\n[Strategy] Running for: '{strategy}'\n{'='*60}")
    cluster = demo_cluster()
    # graph/shape
    graph, shape = (shared_graph, shared_shape)
    if graph is None or shape is None:
        graph, shape = build_graph(cfg)

    batch = int(cfg.get('batch', 1))
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len = int(cfg.get('decode_len', 32))

    # make model_dict for PIM cost paths
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, 'dim', 1)),
        n_heads=int(getattr(shape, 'n_heads', 1)),
        n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)),
        ffn_dim=int(getattr(shape, 'ffn_dim', 1)),
        seqlen=prefill_len,
    )

    # fresh CostModel each run to avoid state bleed
    reset_simulation_logger()
    cost = CostModel(
        cluster=cluster,
        dtype=cfg.get('dtype', 'fp16'),
        pim_config_path=Path(cfg.get('pim_config_path')),
        gb_config_path=Path(cfg.get('gb_config_path')),
        ramulator_config_path=Path(cfg.get('ramulator_config_path')),
        simulation_log_file=Path(cfg.get('simulation_log_file', './output/pim_simulation.txt')),
        model_dict=model_dict,
    )
    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    label = plan_memory_and_label(cfg, cluster)
    buffer_mgr = GlobalMemoryManager()
    sched = _make_scheduler(strategy, cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
    try:
        sched.set_storage_format_map({})
    except Exception:
        pass
    sched.reset_state()

    # simulate
    prefill_time, _prefill = simulate_prefill(sched, cfg, graph)
    decode_time, _decode = simulate_decode_progressive(sched, cfg, graph, prefill_end=prefill_time)
    total_time = float(prefill_time + decode_time)
    # ---- dump CSV stats for this algo run ----
    try:
        result_dir = Path(cfg.get('result_dir', './output/strategy_results'))
        result_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{int(cfg.get('prefill_len', 0))}x{int(cfg.get('decode_len', 0))}"
        ops_csv   = result_dir / f"{tag}_ops.csv"
        comms_csv = result_dir / f"{tag}_comms.csv"
        if getattr(sched, 'stats', None):
            sched.stats.dump_csv(ops_csv, comms_csv)
    except Exception as e:
        logger.debug(f"[stats] CSV dump skipped: {e}")

    logger.debug(f"[Strategy] Finished '{strategy}': Prefill={prefill_time:.6f}s, Decode={decode_time:.6f}s, Total={total_time:.6f}s")

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    return {
        'policy': f'algo:{strategy}',
        'prefill_time_s': float(prefill_time),
        'decode_time_s': float(decode_time),
        'total_time_s': total_time,
        'batch': batch,
        'prefill_len': prefill_len,
        'decode_len': decode_len,
    }

def _ensure_dir(p:Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

def _save_best_json(algo_dir: Path, tag: str, policy: str, *, times: Dict, prefill_schedule=None, decode_steps=None, cfg: Dict|None=None):
    payload = {
        'policy': policy,
        'config': {'batch': int((cfg or {}).get('batch', 1)), 'prefill_len': int((cfg or {}).get('prefill_len', 0)), 'decode_len': int((cfg or {}).get('decode_len', 0)), 'dtype': (cfg or {}).get('dtype')},
        'best_times': {'prefill': float(times.get('prefill_time_s', 0.0)), 'decode': float(times.get('decode_time_s', 0.0)), 'total': float(times.get('total_time_s', 0.0))},
    }

    if prefill_schedule is not None:
        payload['prefill_schedule'] = prefill_schedule
    if decode_steps is not None:
        payload['decode_steps'] = decode_steps
    path = algo_dir / f"best_summary_{tag}.json"
    with open(path,'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.debug(f"[Strategy] Saved best summary to: {path}")
    return path

def evaluate_suite(cfg: Dict, *, algos: List[str], baselines: List[str], result_dir: str | None, debug: bool, combined_out: str):
    base_dir = _ensure_dir(Path(result_dir or './output/len_sweep'))
    tag = f"{int(cfg.get('prefill_len', 0))}x{int(cfg.get('decode_len', 0))}"
    results: List[Dict] = []
    # --- baselines ONCE ---
    blist = []
    for b in baselines:
        b = b.strip().lower()
        if not b: continue
        if b not in blist: blist.append(b)
    for b in blist:
        algo_dir = _ensure_dir(base_dir / f"algo_{b}")
        try: 
            setup_logging(bool(cfg.get('debug', False)), log_file=str(algo_dir / f"debug_{tag}.txt"))
        except Exception: 
            logger.error(f"Failed to setup logging for baseline '{b}'")
            pass
        cfg_b = dict(cfg)
        cfg_b['simulation_log_file'] = str(algo_dir / f"pim_sim_{tag}.txt")
        r = _eval_one_baseline(cfg_b, b)
        _save_best_json(algo_dir, tag, policy=f"algo:{b}", times=r, cfg=cfg_b, prefill_schedule=r.get('prefill_schedule'), decode_steps=r.get('decode_steps'))
        results.append({'policy': f"algo:{b}", **{k: r[k] for k in ('prefill_time_s','decode_time_s','total_time_s')}})
    # --- algorithms ---
    alist = []
    for a in algos:
        a = a.strip().lower()
        if not a: continue
        if a not in alist: alist.append(a)
    # Build once to share across algos
    shared_graph, shared_shape = build_graph(cfg)
    for a in alist:
        algo_dir = _ensure_dir(base_dir / f"algo_{a}")
        try: 
            setup_logging(bool(cfg.get('debug', False)), log_file=str(algo_dir / f"debug_{tag}.txt"))
        except Exception: 
            logger.error(f"Failed to setup logging for algo '{a}'")
            pass
            
        cfg_a = dict(cfg)
        cfg_a['simulation_log_file'] = str(algo_dir / f"pim_sim_{tag}.txt")
        cfg_a['result_dir'] = str(algo_dir)
        res = _run_strategy_once(a, cfg_a, shared_graph=shared_graph, shared_shape=shared_shape)
        _save_best_json(algo_dir, tag, policy=res.get('policy', f"algo:{a}"), times=res, prefill_schedule=res.get('prefill_schedule'), decode_steps=res.get('decode_steps'), cfg=cfg_a)
        results.append({'policy': res['policy'], **{k: res[k] for k in ('prefill_time_s','decode_time_s','total_time_s')}})
    # --- combined ---
    if results:
        os.makedirs(os.path.dirname(combined_out), exist_ok=True)
        with open(combined_out, 'w', encoding='utf-8') as f:
            json.dump({'config': cfg, 'results': results}, f, ensure_ascii=False, indent=2)
        print(f"[REPORT] Combined comparison saved to: {combined_out}")
    # Pretty print
    print("\n=== Strategy/Baseline Comparison ===")
    header = f"{'Policy':<22} {'Prefill(s)':>12} {'Decode(s)':>12} {'Total(s)':>12}"
    print(header); print('-'*len(header))
    for r in results:
        print(f"{r['policy']:<22} {r['prefill_time_s']:>12.4f} {r['decode_time_s']:>12.4f} {r['total_time_s']:>12.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode')

    # evaluate mode: run all algos + baselines
    # evaluate mode: run all algos + baselines
    sp_eval = sub.add_parser('evaluate', help='Run selected algos and baselines; outputs go under result_dir.')
    sp_eval.add_argument('--config', required=True, type=str, help='Path to a JSON config with run parameters.')
    sp_eval.add_argument('--debug', action='store_true', help='Enable verbose logging.')
    sp_eval.add_argument('--model_family', type=str)
    sp_eval.add_argument('--model_variant', type=str)
    sp_eval.add_argument('--dtype', type=str)
    sp_eval.add_argument('--batch', type=int)
    sp_eval.add_argument('--prefill_len', type=int)
    sp_eval.add_argument('--decode_len', type=int)
    sp_eval.add_argument('--decode_sample_stride', type=int)
    sp_eval.add_argument('--result_dir', type=str)
    sp_eval.add_argument('--algo', type=str,
                         help='Algo list, e.g. "heft,sa,ga" or single name')
    sp_eval.add_argument('--baselines', type=str,
                         help='Baseline list, e.g. "pd,weights_on_pim,attn_on_pim"')

    # weight-suggest mode: multi-pass SA to suggest weight formats
    sp_ws = sub.add_parser('weight-suggest', help='Run multi-pass SA to suggest weight formats; no baselines.')
    sp_ws.add_argument('--config', required=True, type=str, help='Path to a JSON config with run parameters.')
    sp_ws.add_argument('--debug', action='store_true', help='Enable verbose logging.')
    sp_ws.add_argument('--model_family', type=str)
    sp_ws.add_argument('--model_variant', type=str)
    sp_ws.add_argument('--dtype', type=str)
    sp_ws.add_argument('--batch', type=int)
    sp_ws.add_argument('--prefill_len', type=int)
    sp_ws.add_argument('--decode_len', type=int)
    sp_ws.add_argument('--decode_sample_stride', type=int)
    sp_ws.add_argument('--result_dir', type=str)
    sp_ws.add_argument('--algo', type=str,help='Algo list, e.g. "heft,sa,ga"')
    sp_ws.add_argument('--all_passes_json', type=str, help='Override path for all passes JSON.')
    sp_ws.add_argument('--best_summary_json', type=str, help='Override path for best pass summary JSON.')
    sp_ws.add_argument('--weight_format_json', type=str, help='Override path for accepted weight format JSON.')

    args, unknown = parser.parse_known_args()
    if args.mode is None:
        parser.error("Please specify a mode: 'eval' or 'weight-suggest'.")
    
    return args


def _normalize_list_field(val) -> list[str]:
    if isinstance(val, list):
        return [str(t).strip() for t in val if str(t).strip()]
    if isinstance(val, str):
        return [t for t in val.replace(',', ' ').split() if t]
    return []

def _load_cfg_from_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    cfg = dict(DEFAULT_CONFIG)
    if isinstance(raw, dict):
        cfg.update(raw)
    return cfg


def main():
    args = parse_args()

    if getattr(args, 'mode', None) in ('evaluate', 'weight-suggest'):
        cfg = _load_cfg_from_json(getattr(args, 'config'))
        cfg['debug'] = bool(getattr(args, 'debug', False)) or cfg.get('debug', False)
        
        override_fields = [
            'model_family',
            'model_variant',
            'dtype',
            'batch',
            'prefill_len',
            'decode_len',
            'decode_sample_stride',
            'result_dir',
            'algo',
            'baselines',
            'all_passes_json',
            'best_summary_json',
            'weight_format_json',
        ]
        for key in override_fields:
            val = getattr(args, key, None)
            if val is not None:
                cfg[key] = val

        # result_dir always encodes batch: <base>/<family>_<variant>_<dtype>_b<batch>
        result_dir = str(_build_result_dir(cfg, cfg.get('result_dir') or './output'))
        cfg['result_dir'] = result_dir
        
        tag = f"{int(cfg.get('prefill_len', 0))}x{int(cfg.get('decode_len', 0))}"
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        # Top-level driver logger
        setup_logging(cfg['debug'], log_file=str(Path(result_dir) / "driver_debug.txt"))

        # Normalize stride if provided
        if cfg.get('decode_sample_stride', None) is not None:
            try:
                cfg['decode_sample_stride'] = int(cfg['decode_sample_stride'])
            except Exception:
                pass

        if args.mode == 'weight-suggest':
            # Choose a single algo label for bookkeeping (run() itself performs SA-based tuning).
            algo_field = cfg.get('algo', 'heft')
            if isinstance(algo_field, list):
                algo_chosen = str(algo_field[0]) if algo_field else 'heft'
            else:
                parts = [t for t in str(algo_field).replace(',', ' ').split() if t]
                algo_chosen = parts[0] if parts else 'heft'

            # Output files are derived from result_dir + tag
            tag = _build_tag(cfg)
            if isinstance(cfg.get('tag'), str) and cfg['tag'].strip():
                tag = f"{tag}_{cfg['tag'].strip()}"
            
            # Derive per-run file paths (can be overridden by CLI)
            if not cfg.get('all_passes_json'):
                cfg['all_passes_json'] = str(Path(result_dir) / f"all_passes_{tag}.json")
            if not cfg.get('best_summary_json'):
                cfg['best_summary_json'] = str(Path(result_dir) / f"best_summary_{tag}.json")
            if not cfg.get('weight_format_json'):
                cfg['weight_format_json'] = str(Path(result_dir) / f"weight_storage_suggestion_{tag}.json")
            
            # Put simulation log inside the same result_dir and tag to avoid overwrite
            cfg['simulation_log_file'] = str(Path(result_dir) / f"pim_sim_{tag}.txt")
            
            print(f"[weight-suggest] result_dir={result_dir} tag={tag}")
            run(cfg)
            return

        if args.mode == 'evaluate':
            # Build lists from JSON (support comma-separated string or list)
            algos = _normalize_list_field(cfg.get('algo', 'heft'))
            baselines = _normalize_list_field(cfg.get('baselines', 'pd,weights_on_pim,attn_on_pim'))

            baseline_out = cfg.get('baseline_out') or str(Path(result_dir) / f"baseline_compare_{tag}.json")
            print(f"[evaluate] algos={algos} baselines={baselines} result_dir={result_dir} tag={tag}")
            evaluate_suite(cfg, algos=algos, baselines=baselines, result_dir=result_dir, debug=cfg['debug'], combined_out=baseline_out)
            return

if __name__ == '__main__':
    main()
