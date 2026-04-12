"""KV-placement planning and label construction helpers."""

from __future__ import annotations

from .shared import *
from .graph_utils import _clone_graph, _cluster_type_count
from .simulator import _make_scheduler, simulate_decode_progressive, simulate_prefill


def _optional_path(value: Any) -> Path | None:
    if value in (None, ''):
        return None
    try:
        return Path(str(value)).expanduser()
    except Exception:
        return None


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
    default_b = float(dtype_bytes(cfg.get('dtype', 'fp16'), default='fp16'))
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

def _build_cost_model_for_run(
    cfg: Dict,
    cluster: Cluster,
    shape: Any,
    *,
    simulation_log_file: Path | str,
    debug_traces: bool = False,
) -> CostModel:
    """Build CostModel with the same runtime knobs across search / compare / evaluate."""
    prefill_len = int(cfg.get('prefill_len', 128))
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, 'dim', 128)),
        n_heads=int(getattr(shape, 'n_heads', 1)),
        n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)),
        ffn_dim=int(getattr(shape, 'ffn_dim', 512)),
        seqlen=prefill_len,
    )

    npu_backend = _normalize_npu_backend(cfg.get('npu_backend', None))
    if _cluster_type_count(cluster, 'npu') <= 0:
        npu_backend = None

    pim_fast_mode = bool(cfg.get('pim_fast_mode', False))
    return CostModel(
        cluster=cluster,
        dtype=cfg.get('dtype', 'fp16'),
        pim_config_path=_optional_path(cfg.get('pim_config_path')),
        ramulator_config_path=_optional_path(cfg.get('ramulator_config_path')),
        simulation_log_file=Path(simulation_log_file),
        debug_traces=bool(debug_traces),
        model_dict=model_dict,
        npu_backend=npu_backend,
        pim_fast_mode=pim_fast_mode,
        tp_qkv=int(cfg.get('tp_qkv', 1) or 1),
        tp_ffn=int(cfg.get('tp_ffn', 1) or 1),
        tp_moe=int(cfg.get('tp_moe', 1) or 1),
    )

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
    # Tensor-parallel shard knobs (used by cost model/NPU backends).
    setattr(label, 'tp_qkv', int(cfg.get('tp_qkv', 1) or 1))
    setattr(label, 'tp_ffn', int(cfg.get('tp_ffn', 1) or 1))
    setattr(label, 'tp_ffn_effective', int(cfg.get('tp_ffn_effective', cfg.get('tp_ffn', 1)) or 1))
    setattr(label, 'tp_moe', int(cfg.get('tp_moe', cfg.get('tp_ffn', 1)) or 1))
    setattr(label, 'tp_moe_effective', int(cfg.get('tp_moe_effective', cfg.get('tp_moe', cfg.get('tp_ffn', 1))) or 1))
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
        sched = _make_scheduler("HEFT", cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)

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
    """Normalize npu_backend strings to canonical: fast / ascend_310b_lut / llmcompass."""
    if backend is None:
        return None
    b = str(backend).strip().lower().replace('-', '_')
    b = b.replace(' ', '_')
    if b in ('fast', 'fastmode', 'fast_mode'):
        return 'fast'
    if b in ('ascend_310b_lut', 'ascend310b_lut', 'ascend_lut', 'lut', 'runtime_lut'):
        return 'ascend_310b_lut'
    if b in ('ascend_310b_json', 'ascend310b_json', 'ascend_json', 'json', 'runtime_json', 'ascend_310b'):
        return 'ascend_310b_json'
    if b in ('llmcompass', 'llm_compass'):
        return 'llmcompass'
    raise ValueError(
        f"Unknown npu_backend='{backend}'. Expected one of: fast, ascend_310b_lut, ascend_310b_json, llmcompass"
    )

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
    """Choose KV placement by capacity only: prefer PIM, otherwise fall back to Host.
    """
    kv_plan = _compute_kv_plan_info(cfg=cfg, cluster=cluster, graph=graph, shape=shape)
    label_host, _ = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place='host')
    label_pim, ok_pim = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place='pim')

    if ok_pim and _infer_kv_place_from_label(label_pim) == 'pim':
        setattr(label_pim, "kv_policy_selected", "pim_by_capacity")
        setattr(label_pim, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
        if bool(capture_best_schedule):
            setattr(label_pim, "_kv_policy_best_sim", None)
        return label_pim

    setattr(label_host, "kv_policy_selected", "host_by_capacity")
    setattr(label_host, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
    if bool(capture_best_schedule):
        setattr(label_host, "_kv_policy_best_sim", None)
    return label_host

