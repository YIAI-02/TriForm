"""Baseline graph policies and registry helpers."""

from __future__ import annotations

from .shared import *

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
    policy = _normalize_baseline_name(policy)
    g2 = _clone_graph(g)
    if policy == 'PD':
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

    if policy == 'AF':
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
PD_BASELINES = {'PD', 'PD+FFN', 'NeuPIMs', 'PD+Attn', 'PD+Linear', 'PAPI-inspired', 'PAISE-inspired'}

def register_baseline(name: str):
    name = _normalize_baseline_name(name)
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


# ---- PAPI/PAISE-inspired baseline helpers ----
_FC_OPS = {'Q', 'K', 'V', 'O', 'FFN_W1', 'FFN_W2', 'FFN_W3'}
_PAPI_ATTN_OPS = {'QK', 'SV', 'SOFTMAX', 'ATTN_SOFTMAX'}
_PAISE_ATTN_OPS = {'QK', 'SV'}
_NPU_ONLY_NONLINEAR_OPS = {
    'LN', 'NORM', 'ADD', 'GELU', 'SWIGLU', 'SILU', 'ROUTER',
    'MOE_ROUTER', 'SOFTMAX', 'ATTN_SOFTMAX',
}


def _op_name(n: TaskNode) -> str:
    return str(getattr(n, 'attrs', {}).get('op') or getattr(n, 'name', '') or '').upper()


def _pin_node_to(n: TaskNode, dev_type: str) -> None:
    dev_type = str(dev_type or '').lower()
    if not isinstance(getattr(n, 'allowed', None), dict):
        n.allowed = {}
    n.allowed['cpu'] = (dev_type == 'cpu')
    n.allowed['npu'] = (dev_type == 'npu')
    n.allowed['pim'] = (dev_type == 'pim')



def _cfg_float(cfg: Dict | None, key: str, default: float) -> float:
    try:
        return float((cfg or {}).get(key, default))
    except Exception:
        return float(default)


def _cfg_int(cfg: Dict | None, key: str, default: int) -> int:
    try:
        return int((cfg or {}).get(key, default))
    except Exception:
        return int(default)


def _first_device_by_type(cluster: Cluster | None, dev_type: str):
    if cluster is None:
        return None
    try:
        devs = list(cluster.devices_by_type(dev_type) or [])
    except Exception:
        devs = []
    return devs[0] if devs else None


def _best_cost_on_type(
    n: TaskNode,
    *,
    dev_type: str,
    cost: CostModel | None,
    cluster: Cluster | None,
    cfg: Dict | None,
    phase: str,
    seq_len: int | None = None,
    batch: int | None = None,
) -> float:

    if cost is None or cluster is None:
        return float('inf')
    try:
        devs = list(cluster.devices_by_type(dev_type) or [])
    except Exception:
        devs = []
    if not devs:
        return float('inf')

    b = int(batch if batch is not None else _cfg_int(cfg, 'batch', 1))
    if seq_len is None:
        seq_len = _cfg_int(cfg, 'prefill_len', 128)
    label = PlanLabel(kv_in_pim=True, pim_mode='baseline_probe', kv_place='pim')

    storage_fmt = 'ND'
    try:
        if isinstance(cfg, dict):
            storage_fmt = str(
                cfg.get('_runtime_probe_weight_storage_fmt')
                or cfg.get('inspired_probe_weight_storage_fmt')
                or 'ND'
            )
    except Exception:
        storage_fmt = 'ND'

    vals: List[float] = []
    has_weight = bool(getattr(n, 'weight_id', None)) and int(getattr(n, 'weight_size', 0) or 0) > 0
    for dev in devs:
        try:
            if has_weight:
                host_src_fmt = str(cost.weight_host_source_format(str(storage_fmt), dev))
                resident_fmt = str(cost.weight_resident_format(str(host_src_fmt), dev))
                stage = cost.weighted_compute_stage(
                    n,
                    dev,
                    label,
                    b,
                    int(seq_len),
                    str(phase),
                    resident_weight_fmt=str(resident_fmt),
                )
                vals.append(float(stage.total_s))
            else:
                vals.append(float(cost.node_device_cost(n, dev, label, b, int(seq_len), str(phase))))
        except Exception:
            try:
                vals.append(float(cost.node_device_cost(n, dev, label, b, int(seq_len), str(phase))))
            except Exception:
                continue
    return min(vals) if vals else float('inf')

def _pim_faster_than_npu(
    n: TaskNode,
    *,
    cost: CostModel | None,
    cluster: Cluster | None,
    cfg: Dict | None,
    phase: str,
    min_speedup: float = 0.0,
    seq_len: int | None = None,
    batch: int | None = None,
) -> bool:
    t_npu = _best_cost_on_type(n, dev_type='npu', cost=cost, cluster=cluster, cfg=cfg, phase=phase, seq_len=seq_len, batch=batch)
    t_pim = _best_cost_on_type(n, dev_type='pim', cost=cost, cluster=cluster, cfg=cfg, phase=phase, seq_len=seq_len, batch=batch)
    if not math.isfinite(t_pim):
        return False
    if not math.isfinite(t_npu):
        return True
    if t_npu <= 0.0:
        return False
    return (t_pim <= t_npu * (1.0 - float(min_speedup)))


def _pin_all_prefill_to_npu(g2: TaskGraph) -> TaskGraph:
    for _, n in g2.nodes.items():
        _pin_node_to(n, 'npu')
    return g2


@register_baseline('PD+FFN')
def _baseline_partitioned_ffn(g: TaskGraph, *, phase: str) -> TaskGraph:
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

@register_baseline('NeuPIMs')
def _baseline_neu_pims(g: TaskGraph, *, phase: str) -> TaskGraph:
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

@register_baseline('PD+Attn')
def _baseline_partitioned_attn(g: TaskGraph, *, phase: str) -> TaskGraph:
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

@register_baseline('PD+Linear')
def _baseline_partitioned_linear(g: TaskGraph, *, phase: str) -> TaskGraph:
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


@register_baseline('PAPI-inspired')
def _baseline_papi_inspired(
    g: TaskGraph,
    *,
    phase: str,
    cfg: Dict | None = None,
    cost: CostModel | None = None,
    cluster: Cluster | None = None,
    shape: Any = None,
) -> TaskGraph:
    """PAPI-inspired operator placement.

    The original PAPI scheduler uses runtime RLP*TLP as a lightweight FC
    bottleneck predictor.  This implementation keeps the paper's rule, while
    matching this repository's non-speculative setting by defaulting TLP to 1 and
    RLP to the configured batch size.
    """
    del cost, cluster, shape  # The PAPI rule only needs configured parallelism.
    cfg = cfg or {}
    g2 = _clone_graph(g)
    if phase == 'prefill':
        return _pin_all_prefill_to_npu(g2)

    batch = _cfg_int(cfg, 'batch', 1)
    rlp = _cfg_int(cfg, 'papi_rlp', batch)
    tlp = _cfg_int(cfg, 'papi_tlp', _cfg_int(cfg, 'tlp', 1))
    alpha = _cfg_float(cfg, 'papi_fc_threshold_alpha', _cfg_float(cfg, 'papi_alpha', 0.0))
    fc_on_pim = (float(rlp) * float(tlp)) <= float(alpha)

    for _, n in g2.nodes.items():
        op = _op_name(n)
        if op in _FC_OPS:
            _pin_node_to(n, 'pim' if fc_on_pim else 'npu')
            n.attrs['papi_rlp'] = int(rlp)
            n.attrs['papi_tlp'] = int(tlp)
            n.attrs['papi_alpha'] = float(alpha)
        elif op in _PAPI_ATTN_OPS or _is_kv_rw(n):
            _pin_node_to(n, 'pim')
        else:
            _pin_node_to(n, 'npu')
    return g2


@register_baseline('PAISE-inspired')
def _baseline_paise_inspired(
    g: TaskGraph,
    *,
    phase: str,
    cfg: Dict | None = None,
    cost: CostModel | None = None,
    cluster: Cluster | None = None,
    shape: Any = None,
) -> TaskGraph:
    """PAISE-inspired offloading policy for the common DOPS simulator.

    PAISE's essential decision is kept: use PIM for memory-bound/thin-GEMV
    attention-score/value work, and use the cost model to decide whether FC
    decode kernels are beneficial on PIM.  This baseline does not add PAISE's
    gamma DLA-overhead term or beta idle-bank penalty to the placement rule;
    all existing DOPS scheduler/cost-model weight-load, relayout, communication,
    cache, and compute costs remain unchanged for final timing.
    """
    del shape
    cfg = cfg or {}
    g2 = _clone_graph(g)
    if phase == 'prefill':
        return _pin_all_prefill_to_npu(g2)

    seq_probe = _cfg_int(cfg, 'paise_policy_seq_len', _cfg_int(cfg, 'prefill_len', 128))
    batch = _cfg_int(cfg, 'batch', 1)
    min_speedup = _cfg_float(cfg, 'paise_min_speedup', 0.0)
    attn_policy = str(cfg.get('paise_attention_policy', 'always')).strip().lower()
    fc_policy = str(cfg.get('paise_fc_policy', 'auto')).strip().lower()

    for _, n in g2.nodes.items():
        op = _op_name(n)

        if _is_kv_rw(n):
            _pin_node_to(n, 'pim')
            continue

        if op in _PAISE_ATTN_OPS:
            if attn_policy in {'never', 'npu'}:
                on_pim = False
            elif attn_policy in {'auto', 'cost'}:
                on_pim = _pim_faster_than_npu(
                    n,
                    cost=cost,
                    cluster=cluster,
                    cfg=cfg,
                    phase=phase,
                    min_speedup=min_speedup,
                    seq_len=seq_probe,
                    batch=batch,
                )
            else:
                on_pim = True
            _pin_node_to(n, 'pim' if on_pim else 'npu')
            continue

        if op in _NPU_ONLY_NONLINEAR_OPS:
            _pin_node_to(n, 'npu')
            continue

        if op in _FC_OPS:
            if fc_policy in {'never', 'npu'}:
                on_pim = False
            elif fc_policy in {'always', 'pim'}:
                on_pim = True
            else:
                on_pim = _pim_faster_than_npu(
                    n,
                    cost=cost,
                    cluster=cluster,
                    cfg=cfg,
                    phase=phase,
                    min_speedup=min_speedup,
                    seq_len=seq_probe,
                    batch=batch,
                )
            _pin_node_to(n, 'pim' if on_pim else 'npu')
            continue

        _pin_node_to(n, 'npu')
    return g2


