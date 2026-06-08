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

_FFN_OP_KEYS = {
    'ffn_w1', 'ffn_w2', 'ffn_w3',
    'ffn_up', 'ffn_down', 'ffn_gate',
    'mlp_up', 'mlp_down', 'mlp_gate',
    'swiglu', 'silu_glu', 'silu', 'gelu',
    'act', 'activation',
}

_FFN_OP_EXACT = {k.upper() for k in _FFN_OP_KEYS}

def _node_text(n: TaskNode) -> str:
    attrs = getattr(n, 'attrs', {}) or {}
    op = str(attrs.get('op') or '') if isinstance(attrs, dict) else ''
    return ' '.join([str(getattr(n, 'id', '') or ''), str(getattr(n, 'name', '') or ''), op]).lower()

def _node_op(n: TaskNode) -> str:
    attrs = getattr(n, 'attrs', {}) or {}
    return str((attrs.get('op') if isinstance(attrs, dict) else None) or getattr(n, 'name', '') or '').lower()

def _is_ffn_compute_node(n: TaskNode) -> bool:
    op = _node_op(n)
    op_up = op.upper()
    if op_up in _FFN_OP_EXACT:
        return True
    text = _node_text(n)
    return any(k in op or k in text for k in _FFN_OP_KEYS)

def _is_shared_expert_node(n: TaskNode) -> bool:
    attrs = getattr(n, 'attrs', {}) or {}
    if isinstance(attrs, dict):
        for key in ('expert_shared', 'shared_expert', 'is_shared_expert'):
            if bool(attrs.get(key, False)):
                return True
        # DeepSeek-style shared expert ids are sometimes recorded without a boolean.
        if attrs.get('shared_expert_id', None) is not None:
            return True
    text = _node_text(n)
    return ('shared' in text and ('expert' in text or 'ffn' in text))

def _is_moe_expert_node(n: TaskNode) -> bool:
    attrs = getattr(n, 'attrs', {}) or {}
    if isinstance(attrs, dict):
        for key in ('expert', 'expert_id', 'expert_rank', 'expert_active', 'expert_shard', 'moe_shard'):
            if key in attrs:
                return True
    text = _node_text(n)
    # Legacy Mixtral/DeepSeek graph nodes use names like L3_FFN_W1_E0 or L3_Act_E0.
    return ('_e' in text or 'moe_e' in text) and ('ffn' in text or 'act' in text or 'swiglu' in text)

def _is_cold_moe_ffn_node(n: TaskNode) -> bool:
    """True for routed/cold MoE FFN compute nodes, excluding shared experts.

    ColdMoE baseline semantics:
      - prefill: everything runs on NPU;
      - decode: only non-shared expert FFN compute runs on PIM;
        router, shared experts, attention, norms, residuals, and KV ops stay on NPU.
    """
    if _is_shared_expert_node(n):
        return False
    return bool(_is_moe_expert_node(n) and _is_ffn_compute_node(n))

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
PD_BASELINES = {'PD', 'PD+FFN', 'NeuPIMs', 'PD+Attn', 'PD+Linear'}

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


@register_baseline('PD+FFN')
def _baseline_partitioned_ffn(g: TaskGraph, *, phase: str) -> TaskGraph:

    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True
            n.allowed['pim'] = False
            n.allowed['cpu'] = False
        return g2

    for _, n in g2.nodes.items():
        on_pim = bool(_is_ffn_compute_node(n))
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = False
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




@register_baseline('ColdMoE')
def _baseline_cold_moe(g: TaskGraph, *, phase: str) -> TaskGraph:
    """Hot/cold MoE baseline.

    Semantics requested for DeepSeek/MoE models:
      * prefill: all operators run on NPU;
      * decode: routed (cold) expert FFN compute runs on PIM;
        shared expert FFN, router/combine, attention, KV, norm, residual and
        all other operators run on NPU.
    """
    g2 = _clone_graph(g)
    for _, n in g2.nodes.items():
        on_pim = bool(phase == 'decode' and _is_cold_moe_ffn_node(n))
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = False
    return g2


__all__ = [name for name in globals() if not name.startswith('__')]
