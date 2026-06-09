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

def _normalize_kv_partition_dim(partition_dim: Any, *, default: str = 'kv_head') -> str:
    s = str(partition_dim or default or 'kv_head').strip().lower().replace('-', '_')
    if s in ('seq', 'sequence', 'context', 'ctx', 'kv_seq', 'kv_block', 'context_block'):
        return 'seq'
    if s in ('layer', 'layers', 'layer_rr', 'layer_round_robin'):
        return 'layer'
    if s in ('head', 'heads', 'kvhead', 'kv_head', 'kv_heads'):
        return 'kv_head'
    return str(default or 'kv_head')


def _requested_kv_partition_dim(cfg: Dict, *, is_dsv4: bool, pim_count: int) -> str:
    raw = cfg.get('kv_partition_dim', cfg.get('deepseek_kv_partition_dim', None))
    if raw is None:
        return 'seq' if bool(is_dsv4 and int(pim_count or 0) > 1) else 'kv_head'
    return _normalize_kv_partition_dim(raw, default=('seq' if bool(is_dsv4 and int(pim_count or 0) > 1) else 'kv_head'))


def _requested_kv_seq_shards(cfg: Dict, *, pim_count: int) -> int:
    for key in ('kv_seq_shards', 'attention_seq_shards', 'deepseek_kv_seq_shards'):
        if cfg.get(key) is not None:
            try:
                return max(1, int(cfg.get(key) or 1))
            except Exception:
                pass
    return max(1, int(pim_count or 1))

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


def _is_deepseek_v4_shape(cfg: Dict, shape: Any) -> bool:
    fam = str(cfg.get('model_family', cfg.get('model_type', '')) or '').lower().replace('-', '_')
    return fam in ('deepseek_v4', 'deepseekv4', 'deepseek_v4_pro', 'deepseek_v4_flash') or hasattr(shape, 'deepseek_v4_variant')


def _dsv4_layer_compression_rate(layer_idx: int, shape: Any) -> int:
    ratios = getattr(shape, 'compress_ratios', None)
    if isinstance(ratios, (list, tuple)) and layer_idx < len(ratios):
        try:
            r = int(ratios[layer_idx])
            return r
        except Exception:
            pass

    # Fallback to the compact schedule stored in DOPS shape JSONs.
    first_sliding = int(getattr(shape, 'first_sliding_attention_layers', 0) or 0)
    first_hca = int(getattr(shape, 'first_hca_attention_layers', 0) or 0)
    if layer_idx < first_sliding:
        return 0
    if layer_idx < first_sliding + first_hca:
        return int(getattr(shape, 'hca_compression_rate', 128) or 128)
    rel = int(layer_idx - first_sliding - first_hca)
    if rel % 2 == 0:
        return int(getattr(shape, 'csa_compression_rate', 4) or 4)
    return int(getattr(shape, 'hca_compression_rate', 128) or 128)


def _estimate_deepseek_v4_kv_total_bytes(
    *,
    cfg: Dict,
    shape: Any,
    kv_dtype_bytes: float,
) -> tuple[int, int]:
    """Return (KV bytes, effective_entries_sum) for the compressed DSV4 cache.

    The estimate follows DOPS' graph-level abstraction: each layer stores a
    shared KV stream with one KV head and `head_dim` channels, plus a local
    sliding window for compressed layers and the CSA indexer KV cache.
    Top-k metadata and allocator fragmentation are ignored.
    """
    S = int(cfg.get('prefill_len', 128) or 0)
    T = int(cfg.get('decode_len', 32) or 0)
    L = max(0, int(S + T))
    batch = int(cfg.get('batch', 1) or 1)
    layers = int(getattr(shape, 'layer_num', 1) or 1)
    n_kv_heads = int(getattr(shape, 'n_kv_heads', 1) or 1)
    head_dim = int(getattr(shape, 'head_dim', max(1, int(getattr(shape, 'dim', 1) or 1) // max(1, int(getattr(shape, 'n_heads', 1) or 1)))) or 1)
    window = int(getattr(shape, 'sliding_window', 128) or 128)

    entries_total = 0
    index_entries_total = 0
    csa_rate = int(getattr(shape, 'csa_compression_rate', 4) or 4)
    index_head_dim = int(getattr(shape, 'indexer_head_dim', getattr(shape, 'index_head_dim', 128)) or 128)
    for l in range(layers):
        r = int(_dsv4_layer_compression_rate(l, shape))
        if r <= 1:
            entries = int(min(int(L), max(0, window)))
        else:
            entries = int(math.ceil(float(L) / float(max(1, r))) + min(int(L), max(0, window)))
        entries_total += int(entries)
        # CSA (c4a) uses an auxiliary indexer KV cache.  The main attention KV
        # is a single shared K/V vector, not independent K and V streams.
        if r == csa_rate and csa_rate > 1:
            index_entries_total += int(math.ceil(float(L) / float(csa_rate)))

    main_elems = int(n_kv_heads) * int(head_dim) * int(batch) * int(entries_total)
    index_elems = int(batch) * int(index_head_dim) * int(index_entries_total)
    total = int(math.ceil((float(main_elems) + float(index_elems)) * float(kv_dtype_bytes)))
    return int(total), int(entries_total + index_entries_total)


def _estimate_deepseek_v4_kv_bytes_by_layer(
    *,
    cfg: Dict,
    shape: Any,
    kv_dtype_bytes: float,
) -> List[int]:
    """Approximate DeepSeek-V4 KV bytes per layer.

    DeepSeek-V4 stores a shared compressed KV stream instead of many
    independent KV heads.  For multi-PIM placement we therefore balance by
    layer, not by head.  This helper mirrors
    _estimate_deepseek_v4_kv_total_bytes() but keeps the per-layer sizes.
    """
    S = int(cfg.get('prefill_len', 128) or 0)
    T = int(cfg.get('decode_len', 32) or 0)
    L = max(0, int(S + T))
    batch = int(cfg.get('batch', 1) or 1)
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
    window = int(getattr(shape, 'sliding_window', 128) or 128)
    csa_rate = int(getattr(shape, 'csa_compression_rate', 4) or 4)
    index_head_dim = int(getattr(shape, 'indexer_head_dim', getattr(shape, 'index_head_dim', 128)) or 128)

    out: List[int] = []
    for l in range(max(0, layers)):
        r = int(_dsv4_layer_compression_rate(l, shape))
        if r <= 1:
            entries = int(min(int(L), max(0, window)))
        else:
            entries = int(math.ceil(float(L) / float(max(1, r))) + min(int(L), max(0, window)))
        index_entries = 0
        if r == csa_rate and csa_rate > 1:
            index_entries = int(math.ceil(float(L) / float(csa_rate)))
        main_elems = int(n_kv_heads) * int(head_dim) * int(batch) * int(entries)
        index_elems = int(batch) * int(index_head_dim) * int(index_entries)
        out.append(int(math.ceil((float(main_elems) + float(index_elems)) * float(kv_dtype_bytes))))
    return out

def _build_cost_model_for_run(
    cfg: Dict,
    cluster: Cluster,
    shape: Any,
    *,
    simulation_log_file: Path | str,
    debug_traces: bool = False,
) -> CostModel:
    """Build CostModel for the supported runtime: NPU fast + PIM fast.

    The current target workflow only runs analytical fast mode.  Older LUT/trace
    knobs are accepted in config for compatibility, but are normalized here so
    the runtime never builds the trace-backed torch model dictionary.
    """
    npu_backend = 'fast' if _cluster_type_count(cluster, 'npu') > 0 else None
    pim_fast_mode = bool(_cluster_type_count(cluster, 'pim') > 0)
    cfg['npu_backend'] = 'fast' if npu_backend else None
    cfg['pim_fast_mode'] = bool(pim_fast_mode)

    return CostModel(
        cluster=cluster,
        dtype=cfg.get('dtype', 'fp16'),
        pim_config_path=_optional_path(cfg.get('pim_config_path')),
        ramulator_config_path=_optional_path(cfg.get('ramulator_config_path')),
        simulation_log_file=Path(simulation_log_file),
        debug_traces=bool(debug_traces),
        model_dict=None,
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

    is_dsv4 = bool(_is_deepseek_v4_shape(cfg, shape))
    if is_dsv4:
        KV_total_bytes, _kv_effective_entries = _estimate_deepseek_v4_kv_total_bytes(
            cfg=cfg, shape=shape, kv_dtype_bytes=kv_dtype_bytes
        )
    else:
        _kv_effective_entries = int((S + T) * layers)
        KV_total_bytes = int(math.ceil(2 * (S + T) * n_kv_heads * head_dim * batch * layers * kv_dtype_bytes))

    # Sum graph-resident weight bytes.  For DeepSeek-V4 the graph expands only
    # active top-k routed experts plus the always-on shared expert for latency;
    # capacity/placement baselines must still reserve the full checkpoint weights.
    graph_weight_bytes = 0
    for n in graph.nodes.values():
        graph_weight_bytes += int(getattr(n, 'weight_size', 0) or 0)
    FC_total_bytes = int(graph_weight_bytes)
    if _is_deepseek_v4_shape(cfg, shape):
        reported_b = getattr(shape, 'reported_total_params_b', None)
        if reported_b is not None:
            try:
                wt_dtype_b = float(dtype_bytes(cfg.get('dtype', 'fp16'), default='fp16'))
                FC_total_bytes = int(math.ceil(float(reported_b) * 1e9 * wt_dtype_b))
            except Exception:
                FC_total_bytes = int(graph_weight_bytes)

    pim_rr = sorted(pim_devs, key=lambda d: str(d.name))
    pim_bytes_by_name = {d.name: int(d.mem_capacity_GB * (1024**3)) for d in pim_rr}
    pim_bytes_total = int(sum(pim_bytes_by_name.values()))

    npu_rr = sorted(npu_devs, key=lambda d: str(d.name))
    npu_bytes_by_name = {d.name: int(float(getattr(d, 'mem_capacity_GB', 0.0) or 0.0) * (1024**3)) for d in npu_rr}
    npu_bytes_total = int(sum(npu_bytes_by_name.values()))
    best_npu = None
    best_npu_cap = 0
    for d in npu_rr:
        cap = int(npu_bytes_by_name.get(d.name, 0) or 0)
        if cap > best_npu_cap:
            best_npu_cap = cap
            best_npu = d
    best_npu_name = str(getattr(best_npu, 'name', '')) if best_npu is not None else None

    # Build KV-head shards (only meaningful when PIM exists).
    kv_head_to_pim: Dict[int, str] = {}
    kv_heads_by_pim: Dict[str, List[int]] = {d.name: [] for d in pim_rr}
    kv_layer_to_pim: Dict[int, str] = {}
    kv_layers_by_pim: Dict[str, List[int]] = {d.name: [] for d in pim_rr}
    kv_seq_shard_to_pim: Dict[int, str] = {}
    kv_seq_shards_by_pim: Dict[str, List[int]] = {d.name: [] for d in pim_rr}
    kv_bytes_by_pim: Dict[str, int] = {d.name: 0 for d in pim_rr}
    kv_partition_dim = 'kv_head'

    kv_head_to_npu: Dict[int, str] = {}
    kv_heads_by_npu: Dict[str, List[int]] = {d.name: [] for d in npu_rr}
    kv_layer_to_npu: Dict[int, str] = {}
    kv_layers_by_npu: Dict[str, List[int]] = {d.name: [] for d in npu_rr}
    kv_seq_shard_to_npu: Dict[int, str] = {}
    kv_seq_shards_by_npu: Dict[str, List[int]] = {d.name: [] for d in npu_rr}
    kv_bytes_by_npu: Dict[str, int] = {d.name: 0 for d in npu_rr}
    kv_npu_partition_dim = 'kv_head'

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

    # Assign shards to PIMs (balanced).  DeepSeek-V4 has one shared KV stream
    # (n_kv_heads=1 in the bundled shapes), so head-based placement would pin
    # all QK/SV/KV_WRITE work to PIM0.  The default for DeepSeek-V4 is therefore
    # sequence/context-block sharding, which creates same-layer PIM concurrency.
    if pim_rr and is_dsv4:
        requested_dim = _requested_kv_partition_dim(cfg, is_dsv4=True, pim_count=len(pim_rr))
        if requested_dim == 'seq':
            kv_partition_dim = 'seq'
            seq_shards = max(1, int(_requested_kv_seq_shards(cfg, pim_count=len(pim_rr))))
            per_shard_base = int(KV_total_bytes) // int(seq_shards)
            per_shard_rem = int(KV_total_bytes) % int(seq_shards)
            for si in range(int(seq_shards)):
                dev = pim_rr[int(si) % len(pim_rr)]
                kv_seq_shard_to_pim[int(si)] = str(dev.name)
                kv_seq_shards_by_pim[str(dev.name)].append(int(si))
                shard_bytes = int(per_shard_base + (1 if si < per_shard_rem else 0))
                kv_bytes_by_pim[str(dev.name)] = int(kv_bytes_by_pim.get(str(dev.name), 0) or 0) + int(shard_bytes)
            if kv_heads_total > 0:
                # Metadata only: shared-KV has one logical KV head, but runtime
                # locality is driven by kv_seq_shard_to_pim.
                kv_head_to_pim[0] = str(pim_rr[0].name)
                kv_heads_by_pim[str(pim_rr[0].name)].append(0)
        else:
            kv_partition_dim = 'layer'
            per_layer_bytes = _estimate_deepseek_v4_kv_bytes_by_layer(
                cfg=cfg,
                shape=shape,
                kv_dtype_bytes=kv_dtype_bytes,
            )
            for l, layer_bytes in enumerate(per_layer_bytes):
                dev = pim_rr[int(l) % len(pim_rr)]
                kv_layer_to_pim[int(l)] = str(dev.name)
                kv_layers_by_pim[str(dev.name)].append(int(l))
                kv_bytes_by_pim[str(dev.name)] = int(kv_bytes_by_pim.get(str(dev.name), 0) or 0) + int(layer_bytes)

            # Keep a single-head map only as metadata/fallback.  The scheduler will
            # prefer kv_layer_to_pim when kv_partition_dim == 'layer'.
            if kv_heads_total > 0:
                kv_head_to_pim[0] = str(pim_rr[0].name)
                kv_heads_by_pim[str(pim_rr[0].name)].append(0)

    elif pim_rr:
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
        bytes_per_head_all_layers = float(KV_total_bytes) / float(max(1, kv_heads_total))
        for dev in pim_rr:
            hcnt = len(kv_heads_by_pim.get(str(dev.name), []) or [])
            kv_bytes_by_pim[str(dev.name)] = int(math.ceil(float(hcnt) * bytes_per_head_all_layers))

    if npu_rr and is_dsv4:
        requested_dim_npu = _requested_kv_partition_dim(cfg, is_dsv4=True, pim_count=max(1, len(npu_rr)))
        if requested_dim_npu == 'seq':
            kv_npu_partition_dim = 'seq'
            seq_shards = max(1, int(_requested_kv_seq_shards(cfg, pim_count=max(1, len(npu_rr)))))
            per_shard_base = int(KV_total_bytes) // int(seq_shards)
            per_shard_rem = int(KV_total_bytes) % int(seq_shards)
            for si in range(int(seq_shards)):
                dev = npu_rr[int(si) % len(npu_rr)]
                kv_seq_shard_to_npu[int(si)] = str(dev.name)
                kv_seq_shards_by_npu[str(dev.name)].append(int(si))
                shard_bytes = int(per_shard_base + (1 if si < per_shard_rem else 0))
                kv_bytes_by_npu[str(dev.name)] = int(kv_bytes_by_npu.get(str(dev.name), 0) or 0) + int(shard_bytes)
            if kv_heads_total > 0:
                # Metadata only for shared-KV DSV4; runtime locality is by seq shard.
                kv_head_to_npu[0] = str(npu_rr[0].name)
                kv_heads_by_npu[str(npu_rr[0].name)].append(0)
        else:
            kv_npu_partition_dim = 'layer'
            per_layer_bytes = _estimate_deepseek_v4_kv_bytes_by_layer(
                cfg=cfg,
                shape=shape,
                kv_dtype_bytes=kv_dtype_bytes,
            )
            for l, layer_bytes in enumerate(per_layer_bytes):
                dev = npu_rr[int(l) % len(npu_rr)]
                kv_layer_to_npu[int(l)] = str(dev.name)
                kv_layers_by_npu[str(dev.name)].append(int(l))
                kv_bytes_by_npu[str(dev.name)] = int(kv_bytes_by_npu.get(str(dev.name), 0) or 0) + int(layer_bytes)
            if kv_heads_total > 0:
                kv_head_to_npu[0] = str(npu_rr[0].name)
                kv_heads_by_npu[str(npu_rr[0].name)].append(0)

    elif npu_rr:
        nn = len(npu_rr)
        base = len(head_shards) // nn
        rem = len(head_shards) % nn
        sh_idx = 0
        for ni, dev in enumerate(npu_rr):
            take = base + (1 if ni < rem else 0)
            for _ in range(take):
                if sh_idx >= len(head_shards):
                    break
                shard_heads = head_shards[sh_idx]
                sh_idx += 1
                for hid in shard_heads:
                    kv_head_to_npu[int(hid)] = str(dev.name)
                kv_heads_by_npu[str(dev.name)].extend(int(h) for h in shard_heads)

        bytes_per_head_all_layers = float(KV_total_bytes) / float(max(1, kv_heads_total))
        for dev in npu_rr:
            hcnt = len(kv_heads_by_npu.get(str(dev.name), []) or [])
            kv_bytes_by_npu[str(dev.name)] = int(math.ceil(float(hcnt) * bytes_per_head_all_layers))

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

    feasible_npu = False
    if npu_bytes_total > 0 and KV_total_bytes <= npu_bytes_total:
        feasible_npu = True
        for d in npu_rr:
            need = int(kv_bytes_by_npu.get(d.name, 0))
            cap = int(npu_bytes_by_name.get(d.name, 0))
            if need > cap:
                feasible_npu = False
                break

    return {
        'kv_total_bytes_all': int(KV_total_bytes),
        'kv_dtype_bytes': float(kv_dtype_bytes),
        'kv_effective_entries_all_layers': int(_kv_effective_entries),
        'fc_total_bytes': int(FC_total_bytes),
        'graph_weight_bytes': int(graph_weight_bytes),
        'tp_qkv_effective': int(tp_qkv_eff),
        'pim_total_capacity_bytes': int(pim_bytes_total),
        'pim_bytes_by_name': dict(pim_bytes_by_name),
        'npu_total_capacity_bytes': int(npu_bytes_total),
        'npu_bytes_by_name': dict(npu_bytes_by_name),
        'kv_head_to_pim': dict(kv_head_to_pim),
        'kv_heads_by_pim': dict(kv_heads_by_pim),
        'kv_layer_to_pim': dict(kv_layer_to_pim),
        'kv_layers_by_pim': dict(kv_layers_by_pim),
        'kv_seq_shard_to_pim': dict(kv_seq_shard_to_pim),
        'kv_seq_shards_by_pim': dict(kv_seq_shards_by_pim),
        'kv_partition_dim': str(kv_partition_dim),
        'kv_bytes_by_pim': dict(kv_bytes_by_pim),
        'kv_head_to_npu': dict(kv_head_to_npu),
        'kv_heads_by_npu': dict(kv_heads_by_npu),
        'kv_layer_to_npu': dict(kv_layer_to_npu),
        'kv_layers_by_npu': dict(kv_layers_by_npu),
        'kv_seq_shard_to_npu': dict(kv_seq_shard_to_npu),
        'kv_seq_shards_by_npu': dict(kv_seq_shards_by_npu),
        'kv_npu_partition_dim': str(kv_npu_partition_dim),
        'kv_bytes_by_npu': dict(kv_bytes_by_npu),
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
    kv_bytes_by_npu_plan = dict(kv_plan.get('kv_bytes_by_npu', {}) or {})
    active_npu_names = sorted([str(k) for k, v in kv_bytes_by_npu_plan.items() if int(v or 0) > 0])
    legacy_single_npu_name = None
    if len(active_npu_names) == 1:
        legacy_single_npu_name = active_npu_names[0]
    elif not active_npu_names and best_npu_name:
        legacy_single_npu_name = str(best_npu_name)

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
        pim_mode = 'kv_pim_by_' + str(kv_plan.get('kv_partition_dim', 'kv_head'))
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
        kv_npu_device=(str(legacy_single_npu_name) if (kv_place_out == 'npu' and legacy_single_npu_name) else None),
        kv_npu_devices=(list(active_npu_names) if kv_place_out == 'npu' else []),
        kv_bytes_by_npu=(dict(kv_plan.get('kv_bytes_by_npu', {})) if (kv_place_out == 'npu' and feasible_npu) else {}),
        kv_head_to_npu=(dict(kv_plan.get('kv_head_to_npu', {})) if (kv_place_out == 'npu' and feasible_npu) else {}),
        kv_heads_by_npu=(dict(kv_plan.get('kv_heads_by_npu', {})) if (kv_place_out == 'npu' and feasible_npu) else {}),
        kv_layer_to_npu=(dict(kv_plan.get('kv_layer_to_npu', {})) if (kv_place_out == 'npu' and feasible_npu) else {}),
        kv_layers_by_npu=(dict(kv_plan.get('kv_layers_by_npu', {})) if (kv_place_out == 'npu' and feasible_npu) else {}),
        kv_seq_shard_to_npu=(dict(kv_plan.get('kv_seq_shard_to_npu', {})) if (kv_place_out == 'npu' and feasible_npu) else {}),
        kv_seq_shards_by_npu=(dict(kv_plan.get('kv_seq_shards_by_npu', {})) if (kv_place_out == 'npu' and feasible_npu) else {}),
        kv_npu_partition_dim=(str(kv_plan.get('kv_npu_partition_dim', 'kv_head')) if (kv_place_out == 'npu' and feasible_npu) else 'kv_head'),
        kv_total_bytes_all=int(KV_total_bytes),
        kv_total_bytes_on_pim=int(kv_bytes_in_pim),
        kv_total_bytes_on_npu=int(KV_total_bytes) if kv_place_out == 'npu' else 0,
        kv_total_bytes_on_host=int(KV_total_bytes) if kv_place_out == 'host' else 0,
        kv_bytes_by_pim=(dict(kv_plan.get('kv_bytes_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_head_to_pim=(dict(kv_plan.get('kv_head_to_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_heads_by_pim=(dict(kv_plan.get('kv_heads_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_layer_to_pim=(dict(kv_plan.get('kv_layer_to_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_layers_by_pim=(dict(kv_plan.get('kv_layers_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_seq_shard_to_pim=(dict(kv_plan.get('kv_seq_shard_to_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_seq_shards_by_pim=(dict(kv_plan.get('kv_seq_shards_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_partition_dim=(str(kv_plan.get('kv_partition_dim', 'kv_head')) if (kv_place_out == 'pim' and feasible_pim) else 'kv_head'),
        pim_weight_capacity_bytes=int(weight_budget),
    )

    # Extra metadata used by reporting / debugging (kept as attributes for flexibility).
    setattr(label, 'total_weight_bytes', int(FC_total_bytes))
    setattr(label, 'fc_total_bytes', int(FC_total_bytes))
    setattr(label, 'graph_weight_bytes', int(kv_plan.get('graph_weight_bytes', FC_total_bytes) or FC_total_bytes))
    setattr(label, 'kv_total_bytes_raw', int(KV_total_bytes))
    setattr(label, 'kv_dtype_bytes', float(kv_plan.get('kv_dtype_bytes', 0.0) or 0.0))
    setattr(label, 'tp_qkv_effective', int(kv_plan.get('tp_qkv_effective', 1) or 1))
    # Tensor-parallel shard knobs (used by cost model/NPU backends).
    setattr(label, 'tp_qkv', int(cfg.get('tp_qkv', 1) or 1))
    setattr(label, 'tp_ffn', int(cfg.get('tp_ffn', 1) or 1))
    setattr(label, 'tp_ffn_effective', int(cfg.get('tp_ffn_effective', cfg.get('tp_ffn', 1)) or 1))
    setattr(label, 'tp_moe', int(cfg.get('tp_moe', cfg.get('tp', cfg.get('tp_ffn', 1))) or 1))
    setattr(label, 'tp_moe_effective', int(cfg.get('tp_moe_effective', cfg.get('tp_moe_total_effective', cfg.get('tp_moe', cfg.get('tp', cfg.get('tp_ffn', 1))))) or 1))
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
    """Normalize npu_backend strings.

    Only NPU fast mode is supported by the current workflow.  Legacy backend
    names are accepted and coerced to 'fast' so older configs still run.
    """
    if backend is None:
        return 'fast'
    b = str(backend).strip().lower().replace('-', '_').replace(' ', '_')
    if b in (
        'fast', 'fastmode', 'fast_mode',
        'ascend_310b_lut', 'ascend310b_lut', 'ascend_lut', 'lut', 'runtime_lut',
        'ascend_310b_json', 'ascend310b_json', 'ascend_json', 'json', 'runtime_json', 'ascend_310b',
        'llmcompass', 'llm_compass',
    ):
        return 'fast'
    raise ValueError(f"Unknown npu_backend='{backend}'. Current supported runtime is fast mode only.")


def _cfg_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'no', 'n', 'off', '', 'none', 'null'):
        return False
    return bool(default)


def _weight_suggest_fast_mode_reasons(cfg: Dict) -> List[str]:
    reasons: List[str] = []

    if _normalize_npu_backend(cfg.get('npu_backend', None)) == 'fast':
        reasons.append('npu_backend=fast')

    if _cfg_bool(cfg.get('pim_fast_mode', None), default=False):
        reasons.append('pim_fast_mode=true')

    return reasons


def _ensure_weight_suggest_supported(cfg: Dict) -> None:
    """Fast mode is now the supported path for both evaluate and weight-suggest."""
    cfg['npu_backend'] = _normalize_npu_backend(cfg.get('npu_backend', 'fast'))
    cfg['pim_fast_mode'] = True
    return None

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

    forced = cfg.get('kv_place', cfg.get('force_kv_place', None))
    if forced is not None:
        forced_place = _normalize_kv_place(str(forced))
        label_forced, ok_forced = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place=forced_place)
        if ok_forced and _infer_kv_place_from_label(label_forced) == forced_place:
            setattr(label_forced, "kv_policy_selected", f"forced_{forced_place}")
            setattr(label_forced, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
            if bool(capture_best_schedule):
                setattr(label_forced, "_kv_policy_best_sim", None)
            return label_forced
        # If a forced PIM/NPU placement is infeasible, fall through to the
        # existing capacity-safe policy instead of crashing long sweeps.

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

