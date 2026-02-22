from __future__ import annotations
from config import attach_local_debug_filter
import argparse
import json
import os
import time
import math
import random
import re
from typing import Dict, List, Callable, Any, Tuple
from hardware import demo_cluster, Cluster
from cost_model import CostModel, DTYPE_BYTES
from cost_model_pim_backend import _make_shared_model_dict
from buffer_manager import GlobalMemoryManager
from model_parser import build_graph
from config import (
    PIM_STATIC_ALLOC_RATIO,
    ENABLE_PIM_WEIGHT_PRELOAD,
    setup_logging,
)
from plan_label import PlanLabel
from scheduler import (
    HEFTScheduler,
    NaiveTopoScheduler,
)

# Optional: communication-aware HEFT (may not be present in older versions)
try:
    from scheduler import HEFTCOMMAWAREScheduler
except Exception:  # pragma: no cover
    HEFTCOMMAWAREScheduler = None
from pathlib import Path
import logging
from task_graph import TaskGraph, TaskNode
from stats_recorder import reset_simulation_logger
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: True)

# Default report paths (used when config/CLI doesn't override)
ALL_PASSES_RESULT_PATH = "./output/all_passes.json"
BEST_PASS_SUMMARY_PATH = "./output/best_summary.json"

# ---- path helper (unify result_dir naming incl. batch) ----
def _build_result_dir(cfg: Dict, default_root: str = './output') -> Path:
    """
    Compose a result directory path that always includes batch:
            <base>/<family>_<variant>_<dtype>_b<batch>[_s<stride>]
    """
    base   = cfg.get('result_dir') or default_root
    family = cfg.get('model_family', 'unnamed')
    variant= cfg.get('model_variant', '')
    dtype  = cfg.get('dtype', 'fp16')
    batch  = int(cfg.get('batch', 1))
    stride = cfg.get('decode_sample_stride', None)
    stride_suffix = f"_s{int(stride)}" if stride not in (None, '') else ""
    return Path(base) / f"{family}_{variant}_{dtype}_b{batch}{stride_suffix}"

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

# KV placement helpers
def _normalize_kv_place(kv_place: str) -> str:
    """Normalize KV placement tags to one of: 'host' | 'pim' | 'npu'."""
    s = str(kv_place or '').strip().lower()
    if s in ('cpu', 'host', 'dram'):
        return 'host'
    if s in ('pim', 'aim'):
        return 'pim'
    if s in ('npu', 'gpu', 'device'):
        return 'npu'
    return 'host'

def _infer_kv_dtype_bytes_from_graph(cfg: Dict, graph: TaskGraph) -> float:
    """Infer KV-cache storage element size (bytes)."""
    default_b = float(DTYPE_BYTES.get(cfg.get('dtype', 'fp16'), 2))
    try:
        for n in graph.nodes.values():
            attrs = getattr(n, 'attrs', None) or {}
            if not isinstance(attrs, dict):
                continue
            opt = attrs.get('opt', None)
            if isinstance(opt, dict) and ('kv_dtype_bytes' in opt):
                kb = opt.get('kv_dtype_bytes', None)
                if kb is None:
                    continue
                try:
                    kb_f = float(kb)
                    if kb_f > 0:
                        return float(kb_f)
                except Exception:
                    continue
    except Exception:
        pass
    return float(default_b)


def _effective_tp_qkv(cfg: Dict) -> int:
    """Validated effective TP factor used for KV-head sharding."""
    tp_eff = cfg.get('tp_qkv_effective', cfg.get('_tp_qkv_effective', None))
    if tp_eff is not None:
        return max(1, int(tp_eff))
    return max(1, int(cfg.get('tp_qkv', 1) or 1))


def _compute_kv_plan_info(
    *,
    cfg: Dict,
    cluster: Cluster,
    graph: TaskGraph,
    shape: Any,
) -> Dict[str, Any]:
    """Compute KV/weight sizes and (if PIM exists) a deterministic KV-head->PIM mapping."""
    pim_devs = list(cluster.devices_by_type('pim') or [])
    npu_devs = list(cluster.devices_by_type('npu') or [])

    kv_dtype_bytes = float(_infer_kv_dtype_bytes_from_graph(cfg, graph))
    S = int(cfg.get('prefill_len', 128))
    T = int(cfg.get('decode_len', 32))
    batch = int(cfg.get('batch', 1))

    layers = int(getattr(shape, 'layer_num', 1) or 1)
    n_kv_heads = int(getattr(shape, 'n_kv_heads', 1) or 1)
    head_dim = int(
        getattr(
            shape,
            'head_dim',
            max(1, int(getattr(shape, 'dim', 1) or 1) // max(1, int(getattr(shape, 'n_heads', 1) or 1))),
        )
        or 1
    )

    KV_total_bytes = int(math.ceil(2 * (S + T) * n_kv_heads * head_dim * batch * layers * kv_dtype_bytes))

    # Sum FC weight bytes from graph.
    FC_total_bytes = 0
    for n in graph.nodes.values():
        FC_total_bytes += int(getattr(n, 'weight_size', 0) or 0)

    pim_rr = sorted(pim_devs, key=lambda d: str(d.name))
    pim_bytes_by_name = {d.name: int(d.mem_capacity_GB * (1024**3)) for d in pim_rr}
    pim_bytes_total = int(sum(pim_bytes_by_name.values()))

    # NPU: choose the single best device for KV (largest capacity).
    best_npu = None
    best_npu_cap = 0
    for d in npu_devs:
        cap = int(float(getattr(d, 'mem_capacity_GB', 0.0) or 0.0) * (1024**3))
        if cap > best_npu_cap:
            best_npu_cap = cap
            best_npu = d
    best_npu_name = str(getattr(best_npu, 'name', '')) if best_npu is not None else None

    # Build KV-head shards (only meaningful when PIM exists).
    kv_head_to_pim: Dict[int, str] = {}
    kv_heads_by_pim: Dict[str, List[int]] = {d.name: [] for d in pim_rr}
    kv_bytes_by_pim: Dict[str, int] = {d.name: 0 for d in pim_rr}

    tp_qkv_eff = int(_effective_tp_qkv(cfg))
    kv_heads_total = int(n_kv_heads)
    tp_qkv_eff = max(1, min(int(tp_qkv_eff), kv_heads_total))
    if kv_heads_total % tp_qkv_eff != 0:
        # Should have been validated earlier; fallback to per-head sharding.
        tp_qkv_eff = kv_heads_total
    kv_heads_per_shard = max(1, kv_heads_total // tp_qkv_eff)

    # Build head shards.
    head_shards: List[List[int]] = []
    for si in range(tp_qkv_eff):
        s0 = si * kv_heads_per_shard
        s1 = min(kv_heads_total, (si + 1) * kv_heads_per_shard)
        head_shards.append(list(range(s0, s1)))

    # Assign shards to PIMs (balanced).
    if pim_rr:
        pn = len(pim_rr)
        base = len(head_shards) // pn
        rem = len(head_shards) % pn
        sh_idx = 0
        for pi, dev in enumerate(pim_rr):
            take = base + (1 if pi < rem else 0)
            for _ in range(take):
                if sh_idx >= len(head_shards):
                    break
                shard_heads = head_shards[sh_idx]
                sh_idx += 1
                for hid in shard_heads:
                    kv_head_to_pim[int(hid)] = str(dev.name)
                kv_heads_by_pim[str(dev.name)].extend(int(h) for h in shard_heads)

        # Compute per-PIM KV bytes.
        bytes_per_head_all_layers = float(2 * (S + T) * head_dim * batch * layers) * kv_dtype_bytes
        for dev in pim_rr:
            hcnt = len(kv_heads_by_pim.get(str(dev.name), []) or [])
            kv_bytes_by_pim[str(dev.name)] = int(math.ceil(float(hcnt) * bytes_per_head_all_layers))

    # Feasibility summaries (used when building specific labels).
    feasible_pim = False
    if pim_bytes_total > 0 and KV_total_bytes <= pim_bytes_total:
        feasible_pim = True
        for d in pim_rr:
            need = int(kv_bytes_by_pim.get(d.name, 0))
            cap = int(pim_bytes_by_name.get(d.name, 0))
            if need > cap:
                feasible_pim = False
                break

    feasible_npu = bool(best_npu is not None and int(best_npu_cap) > 0 and int(KV_total_bytes) <= int(best_npu_cap))

    return {
        'kv_total_bytes_all': int(KV_total_bytes),
        'kv_dtype_bytes': float(kv_dtype_bytes),
        'fc_total_bytes': int(FC_total_bytes),
        'tp_qkv_effective': int(tp_qkv_eff),
        'pim_total_capacity_bytes': int(pim_bytes_total),
        'pim_bytes_by_name': dict(pim_bytes_by_name),
        'kv_head_to_pim': dict(kv_head_to_pim),
        'kv_heads_by_pim': dict(kv_heads_by_pim),
        'kv_bytes_by_pim': dict(kv_bytes_by_pim),
        'feasible_pim': bool(feasible_pim),
        'best_npu_name': best_npu_name,
        'best_npu_cap_bytes': int(best_npu_cap),
        'feasible_npu': bool(feasible_npu),
    }


def _make_label_from_kv_plan(
    *,
    cfg: Dict,
    kv_plan: Dict[str, Any],
    kv_place: str,
) -> Tuple[PlanLabel, bool]:
    """Build a PlanLabel for a specific KV placement using precomputed kv_plan."""

    kv_place_req = _normalize_kv_place(kv_place)

    KV_total_bytes = int(kv_plan.get('kv_total_bytes_all', 0) or 0)
    FC_total_bytes = int(kv_plan.get('fc_total_bytes', 0) or 0)
    pim_bytes_total = int(kv_plan.get('pim_total_capacity_bytes', 0) or 0)
    feasible_pim = bool(kv_plan.get('feasible_pim', False))
    feasible_npu = bool(kv_plan.get('feasible_npu', False))
    best_npu_name = kv_plan.get('best_npu_name', None)

    # PIM preload/weight budget depends on whether KV is placed on PIM.
    if kv_place_req == 'pim':
        preload_ok = (int(FC_total_bytes) + int(KV_total_bytes)) <= int(pim_bytes_total)
    else:
        preload_ok = int(FC_total_bytes) <= int(pim_bytes_total)

    weights_preloaded_on_pim = bool(
        bool(ENABLE_PIM_WEIGHT_PRELOAD)
        and pim_bytes_total > 0
        and bool(preload_ok)
    )

    kv_bytes_in_pim = int(KV_total_bytes) if (kv_place_req == 'pim' and feasible_pim) else 0
    leftover_bytes = max(0, int(pim_bytes_total) - int(kv_bytes_in_pim))
    weight_budget = int(min(int(FC_total_bytes), int(leftover_bytes * PIM_STATIC_ALLOC_RATIO)))
    if bool(weights_preloaded_on_pim):
        weight_budget = int(FC_total_bytes)

    if kv_place_req == 'pim' and feasible_pim:
        kv_place_out = 'pim'
        kv_in_pim_out = True
        pim_mode = 'kv_pim_by_head'
    elif kv_place_req == 'npu' and feasible_npu:
        kv_place_out = 'npu'
        kv_in_pim_out = False
        pim_mode = 'kv_npu'
    else:
        kv_place_out = 'host'
        kv_in_pim_out = False
        pim_mode = 'kv_host' if pim_bytes_total > 0 else 'none'

    selected = bool(kv_place_req == kv_place_out)
    if kv_place_req == 'pim':
        feasible = bool(feasible_pim)
    elif kv_place_req == 'npu':
        feasible = bool(feasible_npu)
    else:
        feasible = True

    label = PlanLabel(
        pim_mode=str(pim_mode),
        kv_in_pim=bool(kv_in_pim_out),
        kv_total_bytes=int(kv_bytes_in_pim),
        kv_place=str(kv_place_out),
        kv_in_npu=bool(kv_place_out == 'npu' and feasible_npu),
        kv_npu_device=str(best_npu_name) if (kv_place_out == 'npu' and best_npu_name) else None,
        kv_total_bytes_all=int(KV_total_bytes),
        kv_total_bytes_on_pim=int(kv_bytes_in_pim),
        kv_total_bytes_on_npu=int(KV_total_bytes) if kv_place_out == 'npu' else 0,
        kv_total_bytes_on_host=int(KV_total_bytes) if kv_place_out == 'host' else 0,
        kv_bytes_by_pim=(dict(kv_plan.get('kv_bytes_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_head_to_pim=(dict(kv_plan.get('kv_head_to_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_heads_by_pim=(dict(kv_plan.get('kv_heads_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_partition_dim='kv_head',
        pim_weight_capacity_bytes=int(weight_budget),
    )

    # Extra metadata used by reporting / debugging (kept as attributes for flexibility).
    setattr(label, 'total_weight_bytes', int(FC_total_bytes))
    setattr(label, 'fc_total_bytes', int(FC_total_bytes))
    setattr(label, 'kv_total_bytes_raw', int(KV_total_bytes))
    setattr(label, 'kv_dtype_bytes', float(kv_plan.get('kv_dtype_bytes', 0.0) or 0.0))
    setattr(label, 'tp_qkv_effective', int(kv_plan.get('tp_qkv_effective', 1) or 1))
    setattr(label, 'pim_total_capacity_bytes', int(pim_bytes_total))
    setattr(label, 'weights_preloaded_on_pim', bool(weights_preloaded_on_pim))

    return label, bool(selected and bool(feasible))


def _make_label_given_kv_place(
    *,
    cfg: Dict,
    cluster: Cluster,
    graph: TaskGraph,
    shape: Any,
    kv_place: str,
) -> tuple[PlanLabel, bool]:
    """Build a PlanLabel with KV placement forced. """
    kv_plan = _compute_kv_plan_info(cfg=cfg, cluster=cluster, graph=graph, shape=shape)
    return _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place=kv_place)




def _fmt_kv_policy_scores(scores: Any) -> str:
    """Pretty string for kv-policy score dict."""
    if not isinstance(scores, dict) or not scores:
        return ""

    def _one(tag: str) -> str:
        v = scores.get(tag)
        if v is None:
            return f"{tag}=N/A"
        if isinstance(v, (int, float)):
            return f"{tag}.total={float(v):.6f}s"
        if isinstance(v, dict):
            tp = v.get("prefill_s")
            td = v.get("decode_s")
            tt = v.get("total_s")
            if all(isinstance(x, (int, float)) for x in (tp, td, tt)):
                return f"{tag}: prefill={float(tp):.6f}s decode={float(td):.6f}s total={float(tt):.6f}s"
        return f"{tag}=?"

    # Stable order: host -> npu -> pim.
    parts = []
    if "host" in scores:
        parts.append(_one("host"))
    if "npu" in scores:
        parts.append(_one("npu"))
    if "pim" in scores:
        parts.append(_one("pim"))
    # include other keys if present
    for k in sorted(set(scores.keys()) - {"host", "npu", "pim"}):
        parts.append(_one(str(k)))
    return " | ".join(parts)


def _infer_kv_place_from_label(label: Any) -> str:
    """Best-effort KV placement string from a PlanLabel."""
    try:
        if bool(getattr(label, 'kv_in_pim', False)):
            return 'pim'
    except Exception:
        pass
    try:
        if bool(getattr(label, 'kv_in_npu', False)):
            return 'npu'
    except Exception:
        pass
    try:
        kp = getattr(label, 'kv_place', None)
        if kp is not None:
            return _normalize_kv_place(kp)
    except Exception:
        pass
    return 'host'


def _apply_kv_place_constraints(g: TaskGraph, kv_place: str) -> TaskGraph:
    """Force KV read/write operators to execute on the KV storage device."""

    kv_place = _normalize_kv_place(kv_place)

    # Local KV op detector (avoid dependency on baseline helper ordering).
    def _is_kv_rw_node(n: TaskNode) -> bool:
        try:
            nm = (getattr(n, 'name', '') or '').lower()
            op = str((getattr(n, 'attrs', {}) or {}).get('op') or '').lower()
        except Exception:
            nm, op = '', ''
        for k in (
            'kv_read', 'kv_write',
            'k_read', 'v_read', 'k_write', 'v_write',
            'k_cache', 'v_cache',
        ):
            if k in nm or k in op:
                return True
        return False

    g2 = _clone_graph(g)
    for _, n in g2.nodes.items():
        if not _is_kv_rw_node(n):
            continue
        if not isinstance(getattr(n, 'allowed', None), dict):
            n.allowed = {}
        # Force only one device type for KV R/W.
        n.allowed['cpu'] = bool(kv_place == 'host')
        n.allowed['npu'] = bool(kv_place == 'npu')
        n.allowed['pim'] = bool(kv_place == 'pim')
    return g2

def _estimate_total_time_for_label(
    *,
    strategy: str,
    cfg: Dict,
    cluster: Cluster,
    cost: CostModel,
    graph_prefill: TaskGraph,
    graph_decode: TaskGraph | None,
    label: PlanLabel,
) -> tuple[float, float, float]:
    """Return (prefill_s, decode_s, total_s) for one label under one scheduler."""
    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))

    buffer_mgr = GlobalMemoryManager()
    try:
        sched = _make_scheduler(strategy, cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
    except Exception:
        # Fallback: baseline HEFT if an unknown strategy name is supplied.
        sched = _make_scheduler("heft", cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)

    try:
        sched.reset_state()
    except Exception:
        pass
    if hasattr(sched, "set_storage_format_map"):
        try:
            sched.set_storage_format_map({})
        except Exception:
            pass
    g_prefill = graph_prefill
    g_decode = graph_decode if graph_decode is not None else graph_prefill
    t_prefill, _ = simulate_prefill(sched, cfg, g_prefill)
    t_decode, _ = simulate_decode_progressive(sched, cfg, g_decode, prefill_end=t_prefill)
    return float(t_prefill), float(t_decode), float(t_prefill + t_decode)

 
def _normalize_npu_backend(backend):
    """Normalize npu_backend strings to canonical: fast / ascend_310b_json / llmcompass."""
    if backend is None:
        return None
    b = str(backend).strip().lower().replace('-', '_')
    b = b.replace(' ', '_')
    if b in ('fast', 'fastmode', 'fast_mode'):
        return 'fast'
    if b in ('ascend_310b_json', 'ascend310b_json', 'ascend_json', 'json', 'runtime_json', 'ascend_310b'):
        return 'ascend_310b_json'
    if b in ('llmcompass', 'llm_compass'):
        return 'llmcompass'
    raise ValueError(f"Unknown npu_backend='{backend}'. Expected one of: fast, ascend_310b_json, llmcompass")


def auto_select_kv_policy(
    *,
    strategy: str,
    cfg: Dict,
    cluster: Cluster,
    cost: CostModel,
    graph: TaskGraph,
    graph_decode: TaskGraph | None = None,
    shape: Any,
    capture_best_schedule: bool = False,
) -> PlanLabel:
    """Choose KV placement among: PIM -> NPU -> Host."""
    kv_plan = _compute_kv_plan_info(cfg=cfg, cluster=cluster, graph=graph, shape=shape)
    # Candidate H: host (always)
    label_host, _ = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place='host')
    cand: List[tuple[str, PlanLabel]] = [("host", label_host)]

    # Candidate P: pim (only if feasible)
    label_pim, ok_pim = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place='pim')
    if ok_pim and _infer_kv_place_from_label(label_pim) == 'pim':
        cand.append(("pim", label_pim))

    # Candidate N: npu (only if feasible)
    label_npu, ok_npu = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place='npu')
    if ok_npu and _infer_kv_place_from_label(label_npu) == 'npu':
        cand.append(("npu", label_npu))

    strat = (str(strategy or "").strip().lower())
    if strat in ("naive", "naivetopo", "naivetoposcheduler", "topo"):
        # Capacity-only priority: PIM -> NPU -> Host
        if ok_pim and _infer_kv_place_from_label(label_pim) == 'pim':
            setattr(label_pim, "kv_policy_selected", "pim_by_capacity")
            setattr(label_pim, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
            return label_pim
        if ok_npu and _infer_kv_place_from_label(label_npu) == 'npu':
            setattr(label_npu, "kv_policy_selected", "npu_by_capacity")
            setattr(label_npu, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
            return label_npu
        setattr(label_host, "kv_policy_selected", "host_by_capacity")
        setattr(label_host, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
        return label_host

    def _simulate_candidate(lb: PlanLabel) -> Dict[str, Any]:
        """Run prefill+decode once for a label, and return times + serialized schedules + scheduler."""
        batch = int(cfg.get("batch", 1))
        prefill_len = int(cfg.get("prefill_len", 128))
        buffer_mgr = GlobalMemoryManager()
        sched = _make_scheduler(strategy, cluster, cost, lb, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
        sched.reset_state()

        if hasattr(sched, "set_storage_format_map"):
            try:
                sched.set_storage_format_map({})
            except Exception:
                pass

        kv_place = _infer_kv_place_from_label(lb)
        g_prefill = _apply_kv_place_constraints(graph, kv_place)
        base_decode = graph_decode if graph_decode is not None else graph
        g_decode = _apply_kv_place_constraints(base_decode, kv_place)
        t_prefill, prefill_ser = simulate_prefill(sched, cfg, g_prefill)
        t_decode, decode_ser = simulate_decode_progressive(sched, cfg, g_decode, prefill_end=t_prefill)

        return {
            "prefill_s": float(t_prefill),
            "decode_s": float(t_decode),
            "total_s": float(t_prefill + t_decode),
            "prefill_schedule": prefill_ser,
            "decode_steps": decode_ser,
            "sched": sched,
        }

    best_tag = "host"
    best_label = label_host
    best_total = float("inf")
    best_sim: Dict[str, Any] | None = None

    scores: Dict[str, Dict[str, float]] = {}
    for tag, lb in cand:
        sim = _simulate_candidate(lb)
        tp = float(sim.get("prefill_s", 0.0))
        td = float(sim.get("decode_s", 0.0))
        tt = float(sim.get("total_s", tp + td))
        scores[str(tag)] = {"prefill_s": tp, "decode_s": td, "total_s": tt}

        if float(tt) < float(best_total):
            best_total = float(tt)
            best_tag = str(tag)
            best_label = lb
            best_sim = sim

    setattr(best_label, "kv_policy_selected", str(best_tag))
    setattr(best_label, "kv_policy_scores", dict(scores))

    if bool(capture_best_schedule) and isinstance(best_sim, dict):
        setattr(best_label, "_kv_policy_best_sim", dict(best_sim))

    return best_label

def _serialize_schedule(schedule: List[ScheduledTask], *, phase: str, token_idx: int | None=None) -> List[Dict]:
    """Convert ScheduledTask list to JSON-friendly dicts."""
    out: List[Dict] = []
    for t in schedule:
        out.append({'node_id': t.node_id, 'device': t.device, 'start': float(t.start), 'finish': float(t.finish), 'duration': float(max(0.0, t.finish - t.start)), 'phase': phase, 'token_idx': token_idx})
    return out


def simulate_prefill(sched: SchedulerBase, cfg: Dict, graph: TaskGraph) -> tuple[float, List[Dict]]:
    """
    Simulate prefill phase: process entire prefix at once.
    current_length = prefill_len
    """
    prefill_len = int(cfg.get('prefill_len', 128))
    sched.set_seq_len(prefill_len)
    prefill_sched = sched.schedule(graph, phase='prefill')
    prefill_time = sched.makespan(prefill_sched)
    return (prefill_time, _serialize_schedule(prefill_sched, phase='prefill', token_idx=None))

def simulate_decode_progressive(sched: SchedulerBase, cfg: Dict, graph: TaskGraph, prefill_end: float) -> tuple[float, List[Dict]]:
  
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len  = int(cfg.get('decode_len', 32))
    global_end  = float(prefill_end)
    steps_serialized: List[Dict] = []

    if isinstance(cfg, dict):
        stride = int(cfg.get('decode_sample_stride', 64))
    # token by token
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

    stride = int(max(1, int(stride)))
    # Be defensive even if the user guarantees divisibility.
    n_blocks = int(decode_len // stride) if (decode_len % stride == 0) else int(math.ceil(decode_len / stride))

    for b in range(n_blocks):
        block_start = int(b * stride)
        t = int(block_start)  # sample at block end
        cur_len = int(prefill_len + t)
        block_tokens = int(min(stride, max(0, decode_len - block_start)))

        # Advance the device/comm timelines to the current global time.
        _advance_to(global_end)
        block_begin = float(global_end)

        # Simulate only one token for this block.
        sched.set_seq_len(cur_len)
        dec_sched = sched.schedule(graph, phase='decode')
        token_end = float(sched.makespan(dec_sched))
        step_time = max(0.0, float(token_end - block_begin))

        steps_serialized.append({
            't': int(t),
            'seq_len': int(cur_len),
            'step_time': float(step_time),
            'estimated': False,
            'schedule': _serialize_schedule(dec_sched, phase='decode', token_idx=t),
        })

        block_end_excl = int(min(decode_len, block_start + block_tokens))
        for u in range(int(t) + 1, int(block_end_excl)):
            steps_serialized.append({
                't': int(u),
                'seq_len': int(prefill_len + u),
                'step_time': float(step_time),
                'estimated': True,
                'schedule': None,
            })
        global_end = float(block_begin + float(step_time) * float(block_tokens))
        _advance_to(global_end)
    return (float(global_end - prefill_end), steps_serialized)

def _make_scheduler(name: str, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
    """Factory for scheduler strategies used by evaluate-suite."""

    name = (name or 'heft').strip().lower()

    # Communication-aware HEFT (COMMAWARE-HEFT)
    if name == 'hefthint' :
        if HEFTCOMMAWAREScheduler is None:
            raise ImportError(
                "HEFTCOMMAWAREScheduler is not available. "
                "Please export it from scheduler.py (e.g., `from scheduler_heft import HEFTCOMMAWAREScheduler`)."
            )
        return HEFTCOMMAWAREScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    # Baseline HEFT
    if name in ('heft'):
        return HEFTScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    # Simple topo-order baseline
    if name in ('naive', 'topo', 'fifo', 'ready'):
        return NaiveTopoScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    raise ValueError(f"Unknown scheduler strategy: {name}")

def mapping_diff_ratio(a: Dict[str, str], b: Dict[str, str]) -> float:
    if not a and (not b):
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    diff = sum((1 for k in keys if a.get(k) != b.get(k)))
    return diff / float(len(keys))

def _strip_layer_prefix(wid: str) -> str:
    """Map a per-layer weight_id to a cross-layer block key.

    Examples:
        L12_W1_S0    -> W1_S0
        L3_E2_W1     -> E2_W1
        WTE          -> WTE
    """
    if not wid:
        return ""
    s = str(wid)
    m = re.match(r"^L\d+_(.*)$", s)
    s2 = m.group(1) if m else s
    # Strip head shard suffix: ..._S<digits>
    s2 = re.sub(r"_S\d+$", "", s2)
    return s2

def _build_weight_blocks(weight_ids: List[str]) -> Dict[str, List[str]]:
    """Return {block_key: [wid,...]} using layer-prefix stripping."""
    blocks: Dict[str, List[str]] = {}
    for wid in weight_ids:
        key = _strip_layer_prefix(wid)
        blocks.setdefault(key, []).append(wid)
    return blocks

def _dominant_block_fmt(npu_cnt: int, pim_cnt: int, nd_margin: float) -> str:
    """Decide block format based on reload/miss counts.

    nd_margin:
        A *relative* tolerance in [0,1]. If the NPU/PIM difference is within
        this band, we keep ND.
    """
    npu_cnt = int(npu_cnt or 0)
    pim_cnt = int(pim_cnt or 0)
    total = npu_cnt + pim_cnt
    if total <= 0:
        return "ND"
    # "差不多" band: keep ND.
    if abs(npu_cnt - pim_cnt) <= float(max(0.0, nd_margin)) * float(total):
        return "ND"
    return "NPU_OPT" if npu_cnt > pim_cnt else "PIM_OPT"


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
    #--------------------------------------------
    # 0: init all hardware settings
    #--------------------------------------------
    result_dir = Path(cfg.get('result_dir') or _build_result_dir(cfg, './output/weight_suggestions'))
    result_dir.mkdir(parents=True, exist_ok=True)
    weight_format_path = Path(cfg.get('weight_format_json') or (result_dir / 'weight_storage_suggestion.json'))
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    pim_config_path = Path(cfg['pim_config_path'])
    gb_config_path = Path(cfg['gb_config_path'])
    ramulator_config_path = Path(cfg['ramulator_config_path'])
    prefill_len = int(cfg.get('prefill_len', 128))
    batch = int(cfg.get('batch', 1))
    graph, shape = build_graph(cfg)
    model_dict = _make_shared_model_dict(dim=int(getattr(shape, 'dim', 128)), n_heads=int(getattr(shape, 'n_heads', 1)), n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)), ffn_dim=int(getattr(shape, 'ffn_dim', 512)), seqlen=prefill_len)
    sim_log_file = cfg.get('simulation_log_file', str(result_dir / 'pim_simulation.txt'))
    npu_backend = _normalize_npu_backend(cfg.get('npu_backend', None))
    if _cluster_type_count(cluster, 'npu') <= 0:
        npu_backend = None
    pim_fast_mode = bool(cfg.get('pim_fast_mode', False))
    cost = CostModel(cluster, dtype=cfg.get('dtype', 'fp16'), pim_config_path=pim_config_path, gb_config_path=gb_config_path, ramulator_config_path=ramulator_config_path,  simulation_log_file=sim_log_file, debug_traces=False, model_dict=model_dict, npu_backend=npu_backend, pim_fast_mode=pim_fast_mode)
    cost.logger.start_simulation()
    fmt_map: Dict[str, str] = {}
    prev_total: float|None = None
    prev_map: Dict[str, str] = {}
    best_total: float|None = None
    best_map: Dict[str, str] = {}
    best_pass: int = -1
    last_prefill = 0.0
    last_decode = 0.0
    all_pass_records: List[Dict] = []
    buffer_mgr = GlobalMemoryManager()
    # Choose scheduler class for the tuning run (default: HEFT).
    algo_raw = cfg.get('algo', 'heft')
    if isinstance(algo_raw, list):
        algo_name = str(algo_raw[0]) if algo_raw else 'heft'
    else:
        algo_name = str(algo_raw)
    algo_name = (algo_name.replace(',', ' ').split()[:1] or ['heft'])[0].strip().lower()
    SchedCls = HEFTScheduler
    if algo_name == 'hefthint':
        if HEFTCOMMAWAREScheduler is None:
            raise ImportError(
                "HEFTCOMMAWAREScheduler is not available. "
                "Please export it from scheduler.py.",
            )
        SchedCls = HEFTCOMMAWAREScheduler
    elif algo_name not in ('heft', 'heft+greedy', 'greedy', ''):
        logger.debug(f"[weight-suggest] Unknown algo '{algo_name}', fallback to HEFTScheduler")

    label = auto_select_kv_policy(
        strategy=algo_name,
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=graph,
        shape=shape,
    )

    sel = getattr(label, 'kv_policy_selected', 'unknown')
    sc = getattr(label, 'kv_policy_scores', {})
    msg = _fmt_kv_policy_scores(sc)
    if msg:
        logger.debug(str(f"[KV-SELECT] selected={sel} | {msg}"))
    else:
        logger.debug(str(f"[KV-SELECT] selected={sel}"))

    kv_place = _infer_kv_place_from_label(label)
    graph_kv = _apply_kv_place_constraints(graph, kv_place)
    
    # ------------------------------------------------------------
    # 1: block-CD + 2-layer BFS/beam search
    # ------------------------------------------------------------
    outer_max = int(cfg.get('format_outer_max_iters', cfg.get('outer_max_iters', 3)) or 3)
    inner_max_blocks = int(cfg.get('format_inner_max_blocks', 0) or 0)  # 0 => no cap
    inner_improve_eps = float(cfg.get('format_inner_improve_eps', 1e-6) or 0.0)
    outer_stop_eps = float(cfg.get('format_outer_stop_eps', 0.0) or 0.0)
    nd_margin_init = float(cfg.get('format_nd_margin_init', 0.60) or 0.0)
    nd_margin_decay = float(cfg.get('format_nd_margin_decay', 0.85) or 0.0)
    nd_margin_min = float(cfg.get('format_nd_margin_min', 0.05) or 0.0)

    # Stable blocks built from model graph weight ids.
    all_wids = sorted({str(n.weight_id) for n in graph.nodes.values() if getattr(n, 'weight_id', None)})
    blocks = _build_weight_blocks(all_wids)

    logger.debug(
        f"[AL] init: weights={len(all_wids)} blocks={len(blocks)} "
        f"outer_max={outer_max} inner_max_blocks={('inf' if not inner_max_blocks else int(inner_max_blocks))} "
        f"nd_margin_init={nd_margin_init:.3f} decay={nd_margin_decay:.3f} min={nd_margin_min:.3f} "
        f"inner_eps={inner_improve_eps:g} outer_stop_eps={outer_stop_eps:g}"
    )

    def _normalize_wlc(wstats: Dict) -> Dict[str, Dict[str, int]]:
        raw = (wstats or {}).get('weight_load_counts', {}) or {}
        out: Dict[str, Dict[str, int]] = {}
        for wid, cnts in raw.items():
            try:
                out[str(wid)] = {str(k): int(v) for k, v in (cnts or {}).items()}
            except Exception:
                out[str(wid)] = {}
        return out

    def _block_reload_counts(wlc: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[int, int]]:
        """Return {block_key: (npu_reload, pim_reload)} aggregated across layers."""
        out: Dict[str, Tuple[int, int]] = {}
        for bkey, wids in blocks.items():
            npu = 0
            pim = 0
            for w in wids:
                c = wlc.get(str(w), {}) or {}
                npu += int(c.get('npu', 0) or 0)
                pim += int(c.get('pim', 0) or 0)
            out[str(bkey)] = (int(npu), int(pim))
        return out

    def _apply_block_fmt(map_in: Dict[str, str], bkey: str, fmt: str) -> Dict[str, str]:
        """
        Set the storage format of a given “block” uniformly to fmt, and return a new format map (without modifying the original map)
        """
        out = dict(map_in or {})
        wids = blocks.get(str(bkey), [])
        for w in wids:
            if fmt == 'ND':
                out.pop(str(w), None)
            else:
                out[str(w)] = str(fmt)
        return out

    def _current_block_fmt(map_in: Dict[str, str], bkey: str) -> str:
        wids = blocks.get(str(bkey), [])
        if not wids:
            return 'ND'
        return str((map_in or {}).get(str(wids[0]), 'ND'))

    def _assign_blocks(
        map_in: Dict[str, str],
        wlc: Dict[str, Dict[str, int]],
        *,
        nd_margin: float,
        only_if_current_nd: bool,
    ) -> Dict[str, str]:
        """Outer-step: assign formats for blocks based on reload counts."""
        out = dict(map_in or {})
        blk_cnt = _block_reload_counts(wlc)
        for bkey, (npu, pim) in blk_cnt.items():
            if only_if_current_nd and _current_block_fmt(out, bkey) != 'ND':
                continue
            fmt = _dominant_block_fmt(npu, pim, float(nd_margin))
            if fmt != 'ND':
                out = _apply_block_fmt(out, bkey, fmt)
        return out

    def _evaluate_map(fmt_map_eval: Dict[str, str], *, tag: str) -> Tuple[float, float, float, Any, Any, Dict]:
        """Run prefill+decode simulation under a given host format map."""
        sched = SchedCls(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
        sched.reset_state()
        sched.set_storage_format_map(fmt_map_eval)
        prefill_time, prefill_ser = simulate_prefill(sched, cfg, graph)
        decode_time, decode_ser = simulate_decode_progressive(sched, cfg, graph, prefill_end=prefill_time)
        total_time = float(prefill_time + decode_time)
        wstats = sched.export_weight_stats()
        return (total_time, float(prefill_time), float(decode_time), prefill_ser, decode_ser, wstats)

    def _record(pass_id: int, total: float, prefill_t: float, decode_t: float, prefill_ser: Any, decode_ser: Any, fm: Dict[str, str], wstats: Dict, *, note: str):
        all_pass_records.append({
            'pass': int(pass_id),
            'note': str(note),
            'times': {'prefill': float(prefill_t), 'decode': float(decode_t), 'total': float(total)},
            'schedules': {'prefill': prefill_ser, 'decode_steps': decode_ser},
            'formats': dict(fm or {}),
            'weights': dict(wstats or {}),
            'pim_trace': list(getattr(getattr(cost, 'logger', None), 'pim_trace', []) or []),
        })

    def _inner_sweep(
        map_in: Dict[str, str],
        base_total: float,
        base_prefill: float,
        base_decode: float,
        base_prefill_ser: Any,
        base_decode_ser: Any,
        base_wstats: Dict,
        *,
        sweep_id: int,
    ) -> Tuple[Dict[str, str], float, float, float, Any, Any, Dict]:
        """Inner sweep: try per-block format flips (NPU_OPT->ND->PIM_OPT)."""
        cur_map = dict(map_in or {})
        cur_total = float(base_total)
        cur_prefill = float(base_prefill)
        cur_decode = float(base_decode)
        cur_prefill_ser = base_prefill_ser
        cur_decode_ser = base_decode_ser
        cur_wstats = dict(base_wstats or {})

        wlc = _normalize_wlc(cur_wstats)
        blk_cnt = _block_reload_counts(wlc)

        # Candidate blocks: stored as NPU_OPT but used mostly on PIM, or vice versa.
        candidates: List[Tuple[int, int, str]] = []  # (severity, total_cnt, bkey)
        for bkey, (npu, pim) in blk_cnt.items():
            fmt = _current_block_fmt(cur_map, bkey)
            if fmt == 'NPU_OPT' and pim > npu:
                candidates.append((int(pim - npu), int(npu + pim), str(bkey)))
            elif fmt == 'PIM_OPT' and npu > pim:
                candidates.append((int(npu - pim), int(npu + pim), str(bkey)))

        # Try the most "wrong" blocks first.
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

        logger.debug(
            f"[AL] inner{sweep_id}: start cur_total={cur_total:.6f}s "
            f"candidates={len(candidates)} max_blocks={('inf' if not inner_max_blocks else int(inner_max_blocks))} "
            f"eps={inner_improve_eps:g}"
        )

        if not candidates:
            logger.debug(f"[AL] inner{sweep_id}: no candidates; skip.")
            return (cur_map, cur_total, cur_prefill, cur_decode, cur_prefill_ser, cur_decode_ser, cur_wstats)

        tried = 0
        accepted_cnt = 0
        for _, _, bkey in candidates:
            if inner_max_blocks and tried >= inner_max_blocks:
                break
            tried += 1

            fmt0 = _current_block_fmt(cur_map, bkey)
            # Two-layer BFS along the line: NPU_OPT -> ND -> PIM_OPT (or reverse)
            if fmt0 == 'NPU_OPT':
                fmt_chain = ['ND', 'PIM_OPT']
            elif fmt0 == 'PIM_OPT':
                fmt_chain = ['ND', 'NPU_OPT']
            else:
                continue

            try:
                _npu_r, _pim_r = blk_cnt.get(str(bkey), (0, 0))
            except Exception:
                _npu_r, _pim_r = 0, 0
            logger.debug(
                f"[AL] inner{sweep_id}: try#{tried}/{len(candidates)} "
                f"block={bkey} fmt0={fmt0} reload(npu={int(_npu_r)}, pim={int(_pim_r)}) "
                f"chain={fmt_chain} cur_total={cur_total:.6f}s"
            )

            accepted = False
            best_trial: float | None = None
            for fmt1 in fmt_chain:
                cand_map = _apply_block_fmt(cur_map, bkey, fmt1)
                total_time, prefill_time, decode_time, prefill_ser, decode_ser, wstats = _evaluate_map(cand_map, tag=f"inner{sweep_id}_blk_{bkey}_{fmt0}_to_{fmt1}")
                try:
                    best_trial = float(total_time) if best_trial is None else min(float(best_trial), float(total_time))
                except Exception:
                    pass
                if float(total_time) + float(inner_improve_eps) < float(cur_total):
                    old_total = float(cur_total)
                    cur_map = cand_map
                    cur_total = float(total_time)
                    cur_prefill = float(prefill_time)
                    cur_decode = float(decode_time)
                    cur_prefill_ser = prefill_ser
                    cur_decode_ser = decode_ser
                    cur_wstats = dict(wstats or {})
                    accepted = True
                    accepted_cnt += 1
                    logger.debug(
                        f"[AL] inner{sweep_id}: ACCEPT block={bkey} {fmt0}->{fmt1} "
                        f"total {old_total:.6f}s -> {cur_total:.6f}s (delta={cur_total - old_total:+.6f}s)"
                    )
                    break
                else:
                    # Keep the logs light: only show a few early rejects.
                    if tried <= 3:
                        logger.debug(
                            f"[AL] inner{sweep_id}: reject block={bkey} {fmt0}->{fmt1} "
                            f"trial_total={float(total_time):.6f}s (cur={cur_total:.6f}s)"
                        )
            if accepted:
                # Refresh counts after an accepted change (keeps subsequent tests meaningful).
                wlc = _normalize_wlc(cur_wstats)
                blk_cnt = _block_reload_counts(wlc)
            else:
                if best_trial is not None and tried <= 3:
                    logger.debug(
                        f"[AL] inner{sweep_id}: best_trial_for_block={float(best_trial):.6f}s (no accept; cur={cur_total:.6f}s)"
                    )

        logger.debug(
            f"[AL] inner{sweep_id}: done tried={tried} accepted={accepted_cnt} final_total={cur_total:.6f}s"
        )
        return (cur_map, cur_total, cur_prefill, cur_decode, cur_prefill_ser, cur_decode_ser, cur_wstats)

    # -------------------------------
    # outer iteration 0: all weights ND
    # -------------------------------
    fmt_map = {}
    logger.debug(f"[AL] outer0: start (all weights ND)")
    total_time0, prefill_time0, decode_time0, prefill_time0_ser, decode_time0_ser, wst0 = _evaluate_map(fmt_map, tag='outer0_all_nd')
    logger.debug(
        f"[AL] outer0: done total={float(total_time0):.6f}s prefill={float(prefill_time0):.6f}s decode={float(decode_time0):.6f}s"
    )
    _record(0, total_time0, prefill_time0, decode_time0, prefill_time0_ser, decode_time0_ser, fmt_map, wst0, note='outer0_all_nd')

    # Initial block assignment from outer0 reload statistics (wide ND band).
    wlc0 = _normalize_wlc(wst0)
    fmt_map = _assign_blocks(fmt_map, wlc0, nd_margin=nd_margin_init, only_if_current_nd=False)
    n_npu0 = sum((1 for v in (fmt_map or {}).values() if str(v) == 'NPU_OPT'))
    n_pim0 = sum((1 for v in (fmt_map or {}).values() if str(v) == 'PIM_OPT'))
    logger.debug(
        f"[AL] outer0->outer1: initial assign explicit_weights={len(fmt_map)} "
        f"(NPU_OPT={n_npu0}, PIM_OPT={n_pim0}, ND_default={max(0, len(all_wids) - len(fmt_map))})"
    )

    # -------------------------------
    # outer iteration 1: baseline + inner sweep
    # -------------------------------
    logger.debug(
        f"[AL] outer1: start baseline (explicit_weights={len(fmt_map)} / total_weights={len(all_wids)})"
    )
    total_time1, prefill_time1, deocde_time1, prefill_time1_ser, decode_time1_ser, wst1 = _evaluate_map(fmt_map, tag='outer1_baseline')
    logger.debug(
        f"[AL] outer1: baseline total={float(total_time1):.6f}s prefill={float(prefill_time1):.6f}s decode={float(deocde_time1):.6f}s"
    )
    outer1_base_total = float(total_time1)
    outer1_base_map = dict(fmt_map)
    (fmt_map, total_time1, prefill_time1, deocde_time1, prefill_time1_ser, decode_time1_ser, wst1) = _inner_sweep(
        fmt_map, total_time1, prefill_time1, deocde_time1, prefill_time1_ser, decode_time1_ser, wst1, sweep_id=1,
    )
    logger.debug(
        f"[AL] outer1: after inner total={float(total_time1):.6f}s "
        f"(delta={float(total_time1) - outer1_base_total:+.6f}s, map_diff={mapping_diff_ratio(outer1_base_map, fmt_map):.3f})"
    )
    _record(1, total_time1, prefill_time1, deocde_time1, prefill_time1_ser, decode_time1_ser, fmt_map, wst1, note='outer1_after_inner')

    best_total = float(total_time1)
    best_map = dict(fmt_map)
    best_pass = 1
    last_prefill = float(prefill_time1)
    last_decode = float(deocde_time1)

    prev_outer_total = float(total_time1)
    prev_outer_map = dict(fmt_map)
    prev_wstats = dict(wst1)

    # -------------------------------
    # outer iterations >= 2
    # -------------------------------
    for outer_it in range(2, max(2, outer_max) + 1):
        # Gradually tighten "keep ND" band.
        nd_margin = max(float(nd_margin_min), float(nd_margin_init) * (float(nd_margin_decay) ** float(max(0, outer_it - 1))))
        wlc_prev = _normalize_wlc(prev_wstats)
        cand_map = _assign_blocks(prev_outer_map, wlc_prev, nd_margin=nd_margin, only_if_current_nd=True)

        diff_ratio = mapping_diff_ratio(prev_outer_map, cand_map)
        if diff_ratio != 0.0:
            try:
                keys = set(prev_outer_map.keys()) | set(cand_map.keys())
                changed = sum((1 for k in keys if prev_outer_map.get(k) != cand_map.get(k)))
            except Exception:
                changed = -1
            logger.debug(
                f"[AL] outer{outer_it}: start nd_margin={nd_margin:.3f} outer_step_changed_weights={changed} "
                f"diff_ratio={diff_ratio:.3f} prev_total={float(prev_outer_total):.6f}s"
            )

        # If nothing changes in the outer step, we can stop.
        if diff_ratio == 0.0:
            logger.debug(f"[AL] outer{outer_it}: no ND blocks to split (nd_margin={nd_margin:.3f}); stop.")
            break

        total_time_k, prefill_time_k, decode_time_k, prefill_time_k_ser, decode_time_k_ser, wst_k = _evaluate_map(cand_map, tag=f'outer{outer_it}_baseline')
        logger.debug(
            f"[AL] outer{outer_it}: baseline total={float(total_time_k):.6f}s prefill={float(prefill_time_k):.6f}s decode={float(decode_time_k):.6f}s"
        )
        outer_k_base_total = float(total_time_k)
        outer_k_base_map = dict(cand_map)
        (cand_map, total_k, prefill_time_k, decode_time_k, prefill_time_k_ser, decode_time_k_ser, wst_k) = _inner_sweep(
            cand_map, total_time_k, prefill_time_k, decode_time_k, prefill_time_k_ser, decode_time_k_ser, wst_k, sweep_id=outer_it,
        )

        # IMPORTANT: use the post-inner-sweep total for decisions/records.
        total_time_k = float(total_k)
        logger.debug(
            f"[AL] outer{outer_it}: after inner total={float(total_time_k):.6f}s "
            f"(delta={float(total_time_k) - outer_k_base_total:+.6f}s, map_diff={mapping_diff_ratio(outer_k_base_map, cand_map):.3f})"
        )

        # Stop if the new outer baseline is worse than the previous outer baseline.
        if float(total_time_k) > float(prev_outer_total) + float(outer_stop_eps):
            logger.debug(
                f"[AL] outer{outer_it}: total {total_time_k:.6f}s is worse than prev {prev_outer_total:.6f}s; stop."
            )
            break

        # Accept this outer iteration.
        prev_outer_total = float(total_time_k)
        prev_outer_map = dict(cand_map)
        prev_wstats = dict(wst_k)
        last_prefill = float(prefill_time_k)
        last_decode = float(decode_time_k)
        _record(outer_it, total_time_k, prefill_time_k, decode_time_k, prefill_time_k_ser, decode_time_k_ser, cand_map, wst_k, note=f'outer{outer_it}_after_inner')

        if best_total is None or float(total_time_k) < float(best_total):
            best_total = float(total_time_k)
            best_map = dict(cand_map)
            best_pass = int(outer_it)

    # Persist best map and reports (same artifacts as legacy mode).
    os.makedirs(os.path.dirname(weight_format_path), exist_ok=True)
    with open(weight_format_path, 'w', encoding='utf-8') as f:
        json.dump(best_map or {}, f, indent=2, sort_keys=True)
    logger.debug(str(f'[INFO] Best weight storage map (AL) saved to: {weight_format_path}'))

    # Also emit an explicit *full* map (including ND) for convenience.
    try:
        full_map = {str(w): str((best_map or {}).get(str(w), 'ND')) for w in all_wids}
        full_path = weight_format_path.with_name(weight_format_path.stem + "_full" + weight_format_path.suffix)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(full_map, f, indent=2, sort_keys=True)
        logger.debug(str(f'[INFO] Full weight storage map (AL) saved to: {full_path}'))
    except Exception:
        pass

    all_path = Path(cfg.get('all_passes_json', ALL_PASSES_RESULT_PATH))
    best_path = Path(cfg.get('best_summary_json', BEST_PASS_SUMMARY_PATH))
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'config': {
                    'model_family': cfg.get('model_family'),
                    'model_variant': cfg.get('model_variant'),
                    'dtype': cfg.get('dtype'),
                    'batch': cfg.get('batch'),
                    'prefill_len': cfg.get('prefill_len'),
                    'decode_len': cfg.get('decode_len'),
                },
                'passes': all_pass_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.debug(str(f'[REPORT] All passes (AL) saved to: {all_path}'))

    if all_pass_records:
        best_idx = min(range(len(all_pass_records)), key=lambda i: float(all_pass_records[i]['times']['total']))
        best_rec = all_pass_records[best_idx]
        best_total_rec = float(best_rec['times']['total'])
        improvements = []
        for rec in all_pass_records:
            total_time = float(rec['times']['total'])
            delta = float(total_time - best_total_rec)
            pct = delta / total_time * 100.0 if total_time > 0 else 0.0
            improvements.append({'pass': rec.get('pass', -1), 'total_time': float(total_time), 'delta_seconds_vs_best': float(delta), 'delta_percent_vs_that_pass': float(pct)})
        with open(best_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'best_pass': int(best_rec.get('pass', best_pass)),
                    'best_times': best_rec.get('times', {}),
                    'best_formats': best_rec.get('formats', {}),
                    'best_weights': best_rec.get('weights', {}),
                    'prefill_schedule': best_rec.get('schedules', {}).get('prefill'),
                    'decode_steps': best_rec.get('schedules', {}).get('decode_steps'),
                    'improvements_vs_each_pass': improvements,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.debug(str(f'[REPORT] Best pass summary (AL) saved to: {best_path}'))

    # AL mode terminates here (skip legacy multi-pass loop below).
    cost.logger.end_simulation()
    cost.logger.close()
    
    return

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
            _ = g.topological()
        except Exception:
            pass
    return new_g

# ---- Hardware capability helpers ----
def _cluster_type_count(cluster: Cluster, dev_type: str) -> int:
    """Safe device-type counter (returns 0 on any error)."""
    try:
        return int(len(cluster.devices_by_type(dev_type)) or 0)
    except Exception:
        return 0

def _fallback_npu_to_cpu_if_needed(g: TaskGraph, cluster: Cluster, *, verbose: bool=False) -> TaskGraph:
    npu_cnt = _cluster_type_count(cluster, 'npu')
    cpu_cnt = _cluster_type_count(cluster, 'cpu')
    if npu_cnt > 0 or cpu_cnt <= 0:
        return g

    touched = 0
    total = 0
    for _, n in getattr(g, 'nodes', {}).items():
        total += 1
        try:
            allowed = getattr(n, 'allowed', None)
            if not isinstance(allowed, dict):
                allowed = {}
                setattr(n, 'allowed', allowed)
        except Exception:
            continue

        # If no NPU exists, make it explicit that NPU is not available.
        npu_allowed = bool(allowed.get('npu', False))
        allowed['npu'] = False

        # If this op was intended for NPU, move it to CPU.
        if npu_allowed:
            allowed['cpu'] = True
            touched += 1

        # Final safety: ensure at least one present device type is allowed.
        try:
            cpu_ok = bool(allowed.get('cpu', True))
            pim_ok = bool(allowed.get('pim', True))
        except Exception:
            cpu_ok, pim_ok = True, True
        if not (cpu_ok or pim_ok):
            allowed['cpu'] = True
            touched += 1

    if verbose or touched > 0:
        logger.warning('[HW] No NPU detected; falling back NPU ops to CPU (touched %d/%d nodes).', touched, total)
    return g

def _fallback_pim_to_cpu_if_needed(g: TaskGraph, cluster: Cluster, *, verbose: bool=False) -> TaskGraph:
    pim_cnt = _cluster_type_count(cluster, 'pim')
    cpu_cnt = _cluster_type_count(cluster, 'cpu')
    npu_cnt = _cluster_type_count(cluster, 'npu')
    if pim_cnt > 0 or cpu_cnt <= 0:
        return g

    touched = 0
    total = 0
    for _, n in getattr(g, 'nodes', {}).items():
        total += 1
        try:
            allowed = getattr(n, 'allowed', None)
            if not isinstance(allowed, dict):
                allowed = {}
                setattr(n, 'allowed', allowed)
        except Exception:
            continue

        # If this op is pinned to PIM-only (CPU/NPU both disabled), allow CPU so the
        # graph remains schedulable on a non-PIM topology.
        try:
            pim_only = bool(allowed.get('pim', False)) and (not bool(allowed.get('cpu', False)))
            if npu_cnt > 0:
                pim_only = pim_only and (not bool(allowed.get('npu', False)))
        except Exception:
            pim_only = False
        if pim_only:
            allowed['cpu'] = True
            touched += 1
            
    if verbose or touched > 0:
        logger.warning('[HW] No PIM detected; falling back PIM-only ops to CPU (touched %d/%d nodes).', touched, total)
    return g

def _apply_policy_on_graph(g: TaskGraph, policy: str, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if policy == 'pd':
        for _, n in g2.nodes.items():
            if phase == 'decode':
                n.allowed['npu'] = False
                n.allowed['pim'] = True
                n.allowed['cpu'] = False
            else:
                n.allowed['npu'] = True
                n.allowed['pim'] = False
                n.allowed['cpu'] = True
        return g2

    if policy == 'weights_on_pim':
        for _, n in g2.nodes.items():
            has_w = n.weight_id is not None and (n.weight_size or 0) > 0
            n.allowed['pim'] = bool(has_w)
            n.allowed['npu'] = not has_w
            n.allowed['cpu'] = not has_w
        return g2

    if policy == 'attn_on_pim':
        for _, n in g2.nodes.items():
            is_attn = _is_attention_node(n)
            n.allowed['pim'] = bool(is_attn)
            n.allowed['npu'] = not is_attn
            n.allowed['cpu'] = not is_attn
        return g2

    raise ValueError(f'Unknown policy: {policy}')


# ===== Baseline registry and paper baselines =====
from typing import Callable

_BASELINE_REGISTRY: Dict[str, Callable[[TaskGraph], TaskGraph]] = {}
PD_BASELINES = {'pd','ianus','neupims','attacc','facil',}

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
    return any(k in nm or k in op for k in (
        'kv_read', 'kv_write',
        'k_read', 'v_read', 'k_write', 'v_write',
        'k_cache', 'v_cache',
    ))

def _is_gemv_like(n: TaskNode, *, phase: str) -> bool:
    op = str(getattr(n, 'attrs', {}).get('op') or n.name or '').upper()
    if phase == 'decode' and any(t in op for t in ['Q','K','V','O','FFN_W1','FFN_W2','GELU']):
        return True
    return str(getattr(n, 'attrs', {}).get('arith_op') or '').lower() == 'gemv'


@register_baseline('ianus')
def _baseline_ianus(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2

    for _, n in g2.nodes.items():
        on_pim = True
        if _is_op(n, 'Q', 'K', 'SOFTMAX', 'NORM', 'ADD'):
            on_pim = False
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = n.allowed.get('npu', True)
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2 

@register_baseline('neupims')
def _baseline_neupims(g: TaskGraph, *, phase: str) -> TaskGraph:
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

@register_baseline('attacc')
def _baseline_attacc(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2
    for _, n in g2.nodes.items():
        on_pim = _is_op(n, 'QK', 'SV','SOFTMAX') or _is_kv_rw(n)
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
        on_pim = _is_gemv_like(n, phase='decode') or _is_op(n, 'QK', 'SV')
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2


def _eval_one_baseline(cfg: Dict, policy: str) -> Dict:
    reset_simulation_logger()

    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    graph, shape = build_graph(cfg)

    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))

    base_dir = Path(cfg["result_dir"])
    algo_dir = base_dir / f"algo_{policy}"
    algo_dir.mkdir(parents=True, exist_ok=True)

    # PIM trace 共享的模型形状信息
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, "dim", 128)),
        n_heads=int(getattr(shape, "n_heads", 1)),
        n_kv_heads=int(getattr(shape, "n_kv_heads", 1)),
        ffn_dim=int(getattr(shape, "ffn_dim", 512)),
        seqlen=prefill_len,
    )

    sim_log_path = Path(cfg.get(
        "simulation_log_file",
        algo_dir / "pim_simulation.txt",
    ))

    npu_backend = _normalize_npu_backend(cfg.get('npu_backend', None))
    if _cluster_type_count(cluster, 'npu') <= 0:
        npu_backend = None
    pim_fast_mode = bool(cfg.get('pim_fast_mode', False))
    cost = CostModel(
        cluster=cluster,
        dtype=cfg.get("dtype", "fp16"),
        pim_config_path=Path(cfg.get("pim_config_path")),
        gb_config_path=Path(cfg.get("gb_config_path")),
        ramulator_config_path=Path(cfg.get("ramulator_config_path")),
        simulation_log_file=sim_log_path,
        model_dict=model_dict,
        pim_fast_mode=pim_fast_mode,
        npu_backend=npu_backend,
    )

    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    # 按 baseline policy 生成 prefill / decode 两个 graph
    pol = (policy or "").lower()
    if pol in _BASELINE_REGISTRY:
        g_prefill = _BASELINE_REGISTRY[pol](graph, phase="prefill")
        g_decode = _BASELINE_REGISTRY[pol](graph, phase="decode")
    else:
        g_prefill = _apply_policy_on_graph(graph, policy, phase="prefill")
        g_decode = _apply_policy_on_graph(graph, policy, phase="decode")

    _fallback_npu_to_cpu_if_needed(g_prefill, cluster)
    _fallback_npu_to_cpu_if_needed(g_decode, cluster)
    _fallback_pim_to_cpu_if_needed(g_prefill, cluster)
    _fallback_pim_to_cpu_if_needed(g_decode, cluster)

    is_pd = pol in PD_BASELINES

    best: Dict[str, Any] | None = None
    best_label = None
    best_prefill_ser = None
    best_decode_ser = None
    best_sched = None

    label = auto_select_kv_policy(
        strategy="naive",
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=g_prefill,
        graph_decode=g_decode,
        shape=shape,
    )

    sel = getattr(label, 'kv_policy_selected', 'unknown')
    sc = getattr(label, 'kv_policy_scores', {})
    msg = _fmt_kv_policy_scores(sc)
    if msg:
        logger.debug(str(f"[KV-SELECT] selected={sel} | {msg}"))
    else:
        logger.debug(str(f"[KV-SELECT] selected={sel}"))

    kv_place = _infer_kv_place_from_label(label)
    g_prefill = _apply_kv_place_constraints(g_prefill, kv_place)
    g_decode = _apply_kv_place_constraints(g_decode, kv_place)

    sched = _make_scheduler("naive", cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=GlobalMemoryManager())
    try:
        sched.set_storage_format_map({})
    except Exception:
        pass
    t_prefill, prefill_ser = simulate_prefill(sched, cfg, g_prefill)

    # PD baseline 需要把 KV 从 host 搬到 PIM 的一次性开销算进去
    t_kv_move = 0.0
    if is_pd and label.kv_in_pim and label.kv_total_bytes > 0:
        host = cluster.devices_by_type("cpu")[0]
        pim_list = cluster.devices_by_type("pim")
        if pim_list:
            per = label.kv_total_bytes // max(1, len(pim_list))
            for d in pim_list:
                t_kv_move = max(t_kv_move, cost.comm_cost(host, d, per))

    t_decode, decode_ser = simulate_decode_progressive(
        sched, cfg, g_decode, prefill_end=t_prefill
    )

    # decode_time_effective = float(t_decode + (t_kv_move if is_pd else 0.0))
    decode_time_effective = float(t_decode)
    total_time = float(t_prefill + decode_time_effective)

    best = {
        "prefill_time_s": float(t_prefill),
        "decode_time_s": decode_time_effective,
        "total_time_s": total_time,
        "kv_in_pim": bool(getattr(label, "kv_in_pim", False)),
        "kv_total_bytes": int(getattr(label, "kv_total_bytes", 0) or 0),
        "pim_weight_capacity_bytes": int(getattr(label, "pim_weight_capacity_bytes", 0) or 0),
    }
    best_label = label
    best_prefill_ser = prefill_ser
    best_decode_ser = decode_ser
    best_sched = sched

    try:
        if best_sched is not None and hasattr(best_sched, "stats"):
            prefix = f"{policy}_prefill-{prefill_len}xdecode_{decode_len}"
            decode_stride = int(cfg.get("decode_sample_stride", 1) or 16)
            trace_ops = algo_dir / f"{prefix}_ops_trace.csv"
            trace_comms = algo_dir / f"{prefix}_comms_trace.csv"
            trace_ops.parent.mkdir(parents=True, exist_ok=True)
            best_sched.stats.dump_trace_csv(
                trace_ops,
                trace_comms,
            )
            # Record trace paths on the plan label for downstream scripts.
            try:
                setattr(best_label, 'trace_ops_csv', str(trace_ops))
                setattr(best_label, 'trace_comms_csv', str(trace_comms))
            except Exception:
                pass
    except Exception:
        pass

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    pim_trace = None
    try:
        if best_sched is not None and hasattr(best_sched, "pim_trace"):
            pim_trace = list(getattr(best_sched, "pim_trace") or [])
    except Exception:
        pim_trace = None

    return {
        "policy": policy,
        "pim_strategy": getattr(best_label, 'kv_policy_selected', None),
        "pim_strategy_scores": getattr(best_label, 'kv_policy_scores', None),
        "prefill_time_s": best["prefill_time_s"],
        "decode_time_s": best["decode_time_s"],
        "total_time_s": best["total_time_s"],
        "batch": batch,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "prefill_schedule": best_prefill_ser,
        "decode_steps": best_decode_ser,
        "pim_trace": pim_trace,
        "kv_in_pim": best.get("kv_in_pim", False),
        "kv_total_bytes": best.get("kv_total_bytes", 0),
        "pim_weight_capacity_bytes": best.get("pim_weight_capacity_bytes", 0),
        "label": best_label,
    }


def _run_strategy_once(strategy: str, cfg: Dict, *, shared_graph=None, shared_shape=None) -> Dict:
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    if shared_graph is not None and shared_shape is not None:
        graph, shape = shared_graph, shared_shape
    else:
        graph, shape = build_graph(cfg)

    # If there is no NPU in the hardware topology, fall back NPU ops to CPU.
    _fallback_npu_to_cpu_if_needed(graph, cluster)

    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))

    # PIM trace 需要的模型形状信息
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, "dim", 128)),
        n_heads=int(getattr(shape, "n_heads", 1)),
        n_kv_heads=int(getattr(shape, "n_kv_heads", 1)),
        ffn_dim=int(getattr(shape, "ffn_dim", 512)),
        seqlen=prefill_len,
    )

    reset_simulation_logger()

    sim_log_path = Path(cfg.get(
        "simulation_log_file",
        "./output/pim_simulation.txt",
    ))

    npu_backend = _normalize_npu_backend(cfg.get('npu_backend', None))
    if _cluster_type_count(cluster, 'npu') <= 0:
        npu_backend = None
    pim_fast_mode = bool(cfg.get('pim_fast_mode', False))
    cost = CostModel(
        cluster=cluster,
        dtype=cfg.get("dtype", "fp16"),
        pim_config_path=Path(cfg.get("pim_config_path")),
        gb_config_path=Path(cfg.get("gb_config_path")),
        ramulator_config_path=Path(cfg.get("ramulator_config_path")),
        simulation_log_file=sim_log_path,
        model_dict=model_dict,
        npu_backend=npu_backend,
        pim_fast_mode=pim_fast_mode,
    )

    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    best: Dict[str, Any] | None = None
    best_sched = None
    best_prefill_ser = None
    best_decode_ser = None
    best_label = None

    label = auto_select_kv_policy(
        strategy=strategy,
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=graph,
        shape=shape,
        capture_best_schedule=True,
    )

    sel = getattr(label, 'kv_policy_selected', 'unknown')
    sc = getattr(label, 'kv_policy_scores', {})
    msg = _fmt_kv_policy_scores(sc)
    if msg:
        logger.debug(str(f"[KV-SELECT] selected={sel} | {msg}"))
    else:
        logger.debug(str(f"[KV-SELECT] selected={sel}"))

    kv_place = _infer_kv_place_from_label(label)
    graph_kv = _apply_kv_place_constraints(graph, kv_place)

    kv_in_pim = bool(getattr(label, "kv_in_pim", False))
    kv_total_bytes = int(getattr(label, "kv_total_bytes", 0) or 0)
    kv_weight_cap = int(getattr(label, "pim_weight_capacity_bytes", 0) or 0)
    sim_best = getattr(label, "_kv_policy_best_sim", None)
    if isinstance(sim_best, dict) and sim_best.get("sched") is not None:
        sched = sim_best.get("sched")
        t_prefill = float(sim_best.get("prefill_s", 0.0) or 0.0)
        t_decode = float(sim_best.get("decode_s", 0.0) or 0.0)
        prefill_ser = sim_best.get("prefill_schedule")
        decode_ser = sim_best.get("decode_steps")
        total_time = float(sim_best.get("total_s", t_prefill + t_decode) or (t_prefill + t_decode))
    else:
        buffer_mgr = GlobalMemoryManager()
        sched = _make_scheduler(strategy, cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
        try:
            sched.set_storage_format_map({})
        except Exception:
            pass
        sched.reset_state()

        t_prefill, prefill_ser = simulate_prefill(sched, cfg, graph_kv)
        t_decode, decode_ser = simulate_decode_progressive(
            sched, cfg, graph_kv, prefill_end=t_prefill
        )
        total_time = float(t_prefill + t_decode)

    best = {
        "prefill_time_s": float(t_prefill),
        "decode_time_s": float(t_decode),
        "total_time_s": total_time,
        "kv_in_pim": kv_in_pim,
        "kv_total_bytes": kv_total_bytes,
        "pim_weight_capacity_bytes": kv_weight_cap,
    }
    best_sched = sched
    best_prefill_ser = prefill_ser
    best_decode_ser = decode_ser
    best_label = label

    try:
        if best_sched is not None and hasattr(best_sched, "stats"):
            prefix = f"{strategy}_prefill-{prefill_len}xdecode_{decode_len}"
            result_dir = Path(cfg.get("result_dir", "./output/strategy_results"))
            result_dir.mkdir(parents=True, exist_ok=True)
            decode_stride = int(cfg.get("decode_sample_stride", 1) or 16)
            trace_ops = result_dir / f"{prefix}_ops_trace.csv"
            trace_comms = result_dir / f"{prefix}_comms_trace.csv"
            best_sched.stats.dump_trace_csv(
                trace_ops,
                trace_comms,
            )
            # Record trace paths on the plan label for downstream scripts.
            try:
                setattr(best_label, 'trace_ops_csv', str(trace_ops))
                setattr(best_label, 'trace_comms_csv', str(trace_comms))
            except Exception:
                pass

    except Exception:
        pass

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    pim_trace = None
    try:
        if best_sched is not None and hasattr(best_sched, "pim_trace"):
            pim_trace = list(getattr(best_sched, "pim_trace") or [])
    except Exception:
        pim_trace = None

    return {
        "strategy": strategy,
        "pim_strategy": getattr(best_label, 'kv_policy_selected', None),
        "pim_strategy_scores": getattr(best_label, 'kv_policy_scores', None),
        "prefill_time_s": best["prefill_time_s"],
        "decode_time_s": best["decode_time_s"],
        "total_time_s": best["total_time_s"],
        "batch": batch,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "prefill_schedule": best_prefill_ser,
        "decode_steps": best_decode_ser,
        "pim_trace": pim_trace,
        "kv_in_pim": best.get("kv_in_pim", False),
        "kv_total_bytes": best.get("kv_total_bytes", 0),
        "pim_weight_capacity_bytes": best.get("pim_weight_capacity_bytes", 0),
        "label": best_label,
    }


def _ensure_dir(p:Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

def _label_summary(label: PlanLabel | None) -> Dict[str, Any]:
    if label is None:
        return {}
    out = {
        'kv_place': str(getattr(label, 'kv_place', 'pim' if bool(getattr(label, 'kv_in_pim', False)) else 'host')),
        'kv_in_npu': bool(getattr(label, 'kv_in_npu', False)),
        'kv_in_pim': bool(getattr(label, 'kv_in_pim', False)),
        'kv_total_bytes': int(getattr(label, 'kv_total_bytes', 0) or 0),
        'kv_total_bytes_all': int(getattr(label, 'kv_total_bytes_all', getattr(label, 'kv_total_bytes_raw', 0)) or 0),
        'pim_weight_capacity_bytes': int(getattr(label, 'pim_weight_capacity_bytes', 0) or 0),
        'pinned_fc_on_pim': sorted(list(getattr(label, 'pinned_fc_on_pim', set()) or [])),
    }

    # Optional: record trace file locations if the caller populated them.
    try:
        ops_p = getattr(label, 'trace_ops_csv', None)
        comms_p = getattr(label, 'trace_comms_csv', None)
        if ops_p:
            out['trace_ops_csv'] = str(ops_p)
        if comms_p:
            out['trace_comms_csv'] = str(comms_p)
    except Exception:
        pass

    return out

def _save_best_json(algo_dir: Path, tag: str, policy: str, *, times: Dict, prefill_schedule=None, decode_steps=None, cfg: Dict|None=None, label: PlanLabel | None = None):
    payload = {
        'policy': policy,
        'pim_strategy': times.get('pim_strategy', 'unknown'),
        'config': {'batch': int((cfg or {}).get('batch', 1)), 'prefill_len': int((cfg or {}).get('prefill_len', 0)), 'decode_len': int((cfg or {}).get('decode_len', 0)), 'dtype': (cfg or {}).get('dtype')},
        'best_times': {'prefill': float(times.get('prefill_time_s', 0.0)), 'decode': float(times.get('decode_time_s', 0.0)), 'total': float(times.get('total_time_s', 0.0))},
    }

    label_dict = _label_summary(label) or _label_summary(times.get('label'))
    if label_dict:
        payload['plan_label'] = label_dict

    if prefill_schedule is not None:
        payload['prefill_schedule'] = prefill_schedule
    if decode_steps is not None:
        payload['decode_steps'] = decode_steps
    pim_trace = times.get('pim_trace')
    if pim_trace is not None:
        payload['pim_trace'] = pim_trace

    # Also record the KV policy comparison numbers if present.
    if 'pim_strategy_scores' in times:
        payload['pim_strategy_scores'] = times.get('pim_strategy_scores')
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
        _save_best_json(algo_dir, tag, policy=f"algo:{b}", times=r, cfg=cfg_b, prefill_schedule=r.get('prefill_schedule'), decode_steps=r.get('decode_steps'), label=r.get('label'))
        results.append({
            'policy': f"algo:{b}",
            'pim_strategy': r.get('pim_strategy'),
            'kv_in_pim': bool(r.get('kv_in_pim', False)),
            'kv_total_bytes': int(r.get('kv_total_bytes', 0) or 0),
            'pim_weight_capacity_bytes': int(r.get('pim_weight_capacity_bytes', 0) or 0),
            **{k: r[k] for k in ('prefill_time_s','decode_time_s','total_time_s')},
        })
    # --- algorithms ---
    alist = []
    for a in algos:
        a = a.strip().lower()
        if not a: continue
        if a not in alist: alist.append(a)
    # Build once to share across algos
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
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
        _save_best_json(algo_dir, tag, policy=res.get('policy', f"algo:{a}"), times=res, prefill_schedule=res.get('prefill_schedule'), decode_steps=res.get('decode_steps'), cfg=cfg_a, label=res.get('label'))
        results.append({
            'policy': f"algo:{a}",
            'pim_strategy': res.get('pim_strategy'),
            'pim_strategy_scores': res.get('pim_strategy_scores'),
            'kv_in_pim': bool(res.get('kv_in_pim', False)),
            'kv_total_bytes': int(res.get('kv_total_bytes', 0) or 0),
            'pim_weight_capacity_bytes': int(res.get('pim_weight_capacity_bytes', 0) or 0),
            **{k: res[k] for k in ('prefill_time_s','decode_time_s','total_time_s')},
        })
    # --- combined ---
    if results:
        os.makedirs(os.path.dirname(combined_out), exist_ok=True)
        with open(combined_out, 'w', encoding='utf-8') as f:
            json.dump({'config': cfg, 'results': results}, f, ensure_ascii=False, indent=2)
        print(f"[REPORT] Combined comparison saved to: {combined_out}")
    # Pretty print
    print("\n=== Strategy/Baseline Comparison ===")
    header = f"{'Policy':<22} {'PIM':<8} {'Prefill(s)':>12} {'Decode(s)':>12} {'Total(s)':>12}"
    print(header); print('-'*len(header))
    for r in results:
        pim_s = str(r.get('pim_strategy') or '')
        print(f"{r['policy']:<22} {pim_s:<8} {r['prefill_time_s']:>12.4f} {r['decode_time_s']:>12.4f} {r['total_time_s']:>12.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode')

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
    sp_eval.add_argument('--hardware_json', type=str,
                         help='Path to a JSON file with hardware topology (devices + links).')
    sp_eval.add_argument('--algo', type=str,
                         help='Algo list, e.g. "heft,sa,ga" or single name')
    sp_eval.add_argument('--baselines', type=str,
                         help='Baseline list, e.g. "pd,weights_on_pim,attn_on_pim"')
    sp_eval.add_argument('--npu_backend', type=str, default=None,
                         choices=['fast_mode', 'ascend_310b_json', 'llmcompass'],
                         help='NPU operator-latency backend: fast_mode/ascend_310b_json/llmcompass. Must be explicitly specified (in config JSON or CLI).')
    sp_eval.add_argument('--pim_fast_mode', action='store_true',default=None)
    # Tensor-parallel shard controls (graph splitting)
    sp_eval.add_argument('--tp_qkv', type=int,
                         help='Tensor-parallel shard size for Q/K/V generation and attention head sharding (column split).')
    sp_eval.add_argument('--tp_ffn', type=int,
                         help='Tensor-parallel shard size for FFN intermediate dimension (ffn_dim split).')
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
    sp_ws.add_argument('--hardware_json', type=str,
                         help='Path to a JSON file with hardware topology (devices + links).')
    sp_ws.add_argument('--algo', type=str,help='Algo list, e.g. "heft,sa,ga"')
    sp_ws.add_argument('--all_passes_json', type=str, help='Override path for all passes JSON.')
    sp_ws.add_argument('--best_summary_json', type=str, help='Override path for best pass summary JSON.')
    sp_ws.add_argument('--weight_format_json', type=str, help='Override path for accepted weight format JSON.')
    sp_ws.add_argument('--npu_backend', type=str, default=None,
                        choices=['fast_mode', 'ascend_310b_json', 'llmcompass'],
                        help='NPU operator-latency backend: fast/ascend_310b_json/llmcompass. Must be explicitly specified (in config JSON or CLI).')
    sp_ws.add_argument('--pim_fast_mode', action='store_true')   
    # Tensor-parallel shard controls (graph splitting)
    sp_ws.add_argument('--tp_qkv', type=int,
                        help='Tensor-parallel shard size for Q/K/V generation and attention head sharding (column split).')
    sp_ws.add_argument('--tp_ffn', type=int,
                        help='Tensor-parallel shard size for FFN intermediate dimension (ffn_dim split).')
    # Graph/tensor-parallel controls
    # Weight-format optimization controls
    sp_ws.add_argument('--format_opt_method', type=str,
                       help='Weight-format optimizer: al_bcd_beam (default) | bcd (legacy).')
    sp_ws.add_argument('--format_outer_max_iters', type=int,
                       help='AL outer iterations (default: 8).')
    sp_ws.add_argument('--format_inner_max_blocks', type=int,
                       help='AL inner sweep cap (0 means no cap).')
    sp_ws.add_argument('--format_nd_margin_init', type=float,
                       help='AL initial ND band (wide) in [0,1].')
    sp_ws.add_argument('--format_nd_margin_decay', type=float,
                       help='AL ND band decay per outer iteration.')
    sp_ws.add_argument('--format_nd_margin_min', type=float,
                       help='AL minimum ND band.')
    sp_ws.add_argument('--format_inner_improve_eps', type=float,
                       help='AL accept change if new_total + eps < old_total.')
    sp_ws.add_argument('--format_outer_stop_eps', type=float,
                       help='AL stop when outer_n is worse than outer_{n-1} by eps.')

    args, unknown = parser.parse_known_args()
    if args.mode is None:
        parser.error("Please specify a mode: 'eval' or 'weight-suggest'.")
    
    return args


def _normalize_list_field(val) -> list[str]:
    items: list[str] = []
    if isinstance(val, list):
        seq = val
    else:
        seq = [val]
    for item in seq:
        if item is None:
            continue
        # 先用逗号拆，再按空白拆
        for tok in str(item).replace(',', ' ').split():
            tok = tok.strip()
            if tok:
                items.append(tok)
    return items

def _load_cfg_from_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config JSON must be an object/dict, got: {type(raw).__name__}")
    # cfg 完全由 JSON 决定
    return dict(raw)

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
            'tp_qkv',
            'tp_ffn',
            'result_dir',
            'hardware_json',
            'algo',
            'baselines',
            'all_passes_json',
            'best_summary_json',
            'weight_format_json',
            'npu_backend',
            'pim_fast_mode',
            # weight-format optimizer knobs (optional CLI overrides)
            'format_opt_method',
            'format_outer_max_iters',
            'format_inner_max_blocks',
            'format_nd_margin_init',
            'format_nd_margin_decay',
            'format_nd_margin_min',
            'format_inner_improve_eps',
            'format_outer_stop_eps',
        ]
        for key in override_fields:
            val = getattr(args, key, None)
            if val is not None:
                cfg[key] = val

        # npu_backend is mandatory: must be explicitly specified in config or CLI
        if cfg.get('npu_backend', None) is None:
            raise ValueError("Missing required config key: 'npu_backend'. Choose from: fast, ascend_310b_json, llmcompass")
        cfg['npu_backend'] = _normalize_npu_backend(cfg.get('npu_backend'))

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
