"""Shared constants, helpers, and dataclasses for CostModel."""

from __future__ import annotations
from cProfile import label
from config import attach_local_debug_filter
import config as _config
import json, os, time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from functools import lru_cache
import subprocess
import tempfile
import re
import sys
import shutil
import hashlib
import pickle
from threading import Lock
import time
from datetime import datetime
from collections import defaultdict
from types import SimpleNamespace
from task_graph import TaskGraph, TaskNode
from hardware import Cluster, DeviceSpec
from plan_label import PlanLabel
from config import HOST_NAME, DEVICE_PREFERRED_FORMAT, FORMAT_SIZE_MULTIPLIER, OPERATOR_DEVICE_ALLOWED, NONOVERLAP_TIME, PIM_RUNTIME_LRU_THRESHOLD
import logging
from stats_recorder import SimulationLogger, get_simulation_logger, reset_simulation_logger
from abc import ABC, abstractmethod
from .cost_model_pim_backend import (
    _get_pim_latency_via_trace,
    _normalize_pim_op,
    PIM_TRACE_SUPPORTED_OPS,
    _make_shared_model_dict,
)
from .cost_model_npu_llmcompass_backend import (
    _normalize_npu_backend,
    _llmcompass_guess_device_key,
    _llmcompass_simulate_matmul_s,
    _llmcompass_simulate_softmax_s,
    _llmcompass_simulate_layernorm_s,
    _llmcompass_simulate_gelu_s,
)
from .cost_model_npu_ascend_backend import (
    _predict_mmad_latency_us_from_lut,
    _predict_softmax_latency_us_from_lut,
    _predict_gelu_latency_us_from_lut,
    _predict_layernorm_latency_us_from_lut,
)


def clamp_overlap_ratio(value: float | int | None, *, default: float = 0.0) -> float:
    try:
        x = float(default if value is None else value)
    except Exception:
        x = float(default)
    if x != x:
        x = float(default)
    return float(min(1.0, max(0.0, x)))


@dataclass(frozen=True)
class OverlapBreakdown:
    total_s: float
    saved_s: float
    overlap_ratio: float
    first_s: float
    second_s: float


@dataclass(frozen=True)
class WeightLoadStageBreakdown:
    total_s: float
    host_src_fmt: str
    resident_fmt: str
    l1_comm_s: float = 0.0
    l2_local_s: float = 0.0
    l1_l2_overlap_ratio: float = 0.0
    combine_rule: str = "serial"
    bytes_nd: int = 0
    bytes_src: int = 0


@dataclass(frozen=True)
class WeightComputeStageBreakdown:
    total_s: float
    compute_fmt: str
    backend: str
    combine_rule: str
    b1_s: float = 0.0
    b2_s: float = 0.0
    launch_overhead_s: float = 0.0


@dataclass(frozen=True)
class WeightOpTimingBreakdown:
    total_s: float
    load: WeightLoadStageBreakdown
    compute: WeightComputeStageBreakdown
    load_compute_overlap_ratio: float
    queue_wait_s: float = 0.0
    overlap_saved_s: float = 0.0
    cache_state: str = ""
    weight_id: str = ""
    weight_size_nd: int = 0
    host_storage_fmt: str = "ND"


def overlap_time(first_s: float, second_s: float, overlap_ratio: float) -> OverlapBreakdown:
    a = max(0.0, float(first_s or 0.0))
    b = max(0.0, float(second_s or 0.0))
    r = clamp_overlap_ratio(overlap_ratio, default=0.0)
    saved = float(r * min(a, b))
    total = float(a + b - saved)
    return OverlapBreakdown(
        total_s=float(total),
        saved_s=float(saved),
        overlap_ratio=float(r),
        first_s=float(a),
        second_s=float(b),
    )

from dtype_utils import DTYPE_BYTES, dtype_bytes, normalize_dtype_token

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)

_CONTEXT_FALLBACK_WARNED: set[str] = set()


def _instantiate_context(ctx_type, /, **kwargs):
    """Instantiate a context object, falling back to SimpleNamespace for stale classes."""
    try:
        return ctx_type(**kwargs)
    except TypeError as exc:
        ctx_name = str(getattr(ctx_type, "__name__", str(ctx_type)))
        if ctx_name not in _CONTEXT_FALLBACK_WARNED:
            logger.warning(
                "Falling back to SimpleNamespace for %s because keyword construction failed: %s",
                ctx_name,
                exc,
            )
            _CONTEXT_FALLBACK_WARNED.add(ctx_name)
        return SimpleNamespace(**kwargs)


class NpuFastModeConfigError(ValueError):
    """Raised when per-device config entries required by NPU fast backend are missing."""

class PimFastModeConfigError(ValueError):
    """Raised when per-device config entries required by PIM fast backend are missing."""

@dataclass(frozen=True)
class NpuWeightRuntimeModel:
    path: Path
    bw_gbs: Dict[str, float]
    overhead_us: Dict[str, float]


@dataclass(frozen=True)
class PimWeightRuntimeModel:
    source: str
    bw_gbs: Dict[str, float]
    overhead_us: Dict[str, float]


@dataclass(frozen=True)
class LocalWeightLoadCost:
    serial_s: float
    overlap_s: float
    total_s: float
    overhead_s: float = 0.0
    stage_sum_s: float = 0.0
    stage_max_s: float = 0.0


_WEIGHT_STORAGE_FORMATS = frozenset(str(x) for x in getattr(_config, 'WEIGHT_STORAGE_FORMATS', ('ND', 'NZ', 'PIM-OPT'))) | frozenset({'DUAL'})
_NPU_WEIGHT_TARGET_FORMATS = frozenset({'ZN', 'ZZ'})
_PIM_WEIGHT_LOAD_DEFAULT_PATHS: Dict[str, Dict[str, float]] = {
    'ND->PIM-OPT': {'bw_gbs': 640.0, 'overhead_us': 2.0},
    'PIM-OPT->PIM-OPT': {'bw_gbs': 1920.0, 'overhead_us': 1.0},
    'NZ->ND': {'bw_gbs': 480.0, 'overhead_us': 4.0},
}


def _normalize_weight_format_token(fmt: str, *, allow_compute: bool = False) -> str:
    raw = str(fmt or '').strip()
    if not raw:
        raise ValueError('Weight format cannot be empty.')
    up = raw.upper().replace('_', '-')
    alias_map = {
        'NPU-OPT': 'NZ',
        'PIM-OPT': 'PIM-OPT',
        'PIMOPT': 'PIM-OPT',
        'NZ': 'NZ',
        'ND': 'ND',
        'ZN': 'ZN',
        'ZZ': 'ZZ',
        'DUAL': 'DUAL',
        'DUAL-COPY': 'DUAL',
        'DUALCOPY': 'DUAL',
        'TWO-COPY': 'DUAL',
        'TWOCOPY': 'DUAL',
        'NZ+PIM-OPT': 'DUAL',
        'NZ+PIMOPT': 'DUAL',
        'PIM-OPT+NZ': 'DUAL',
        'PIMOPT+NZ': 'DUAL',
        'PIM_OPT': 'PIM-OPT',
        'NPU_OPT': 'NZ',
    }
    if up in alias_map:
        up = alias_map[up]
    allowed = set(_WEIGHT_STORAGE_FORMATS)
    if allow_compute:
        allowed |= set(_NPU_WEIGHT_TARGET_FORMATS)
    if up not in allowed:
        raise ValueError(f"Unsupported weight format '{raw}'. Allowed={sorted(allowed)}")
    return up


def _normalize_npu_weight_op_name(node: TaskNode) -> str:
    return str(getattr(node, 'name', '') or '').strip().upper()


def _resolve_npu_weight_conversion_steps(src_fmt: str, dst_fmt: str) -> List[Tuple[str, str]]:
    src = _normalize_weight_format_token(src_fmt, allow_compute=True)
    dst = _normalize_weight_format_token(dst_fmt, allow_compute=True)
    if src == 'DUAL':
        src = 'NZ'
    if src == dst:
        return []

    if src == 'ND' and dst == 'NZ':
        return [('ND', 'NZ')]
    if src == 'NZ' and dst in ('ZZ', 'ZN', 'ND'):
        return [('NZ', dst)]
    if src in ('ZZ', 'ZN') and dst == 'ND':
        return [(src, 'ND')]
    if src == 'ND' and dst in ('ZZ', 'ZN'):
        return [('ND', 'NZ'), ('NZ', dst)]
    if src == 'PIM-OPT' and dst == 'ND':
        return [('PIM-OPT', 'ND')]
    if src == 'PIM-OPT' and dst == 'NZ':
        return [('PIM-OPT', 'ND'), ('ND', 'NZ')]
    if src == 'PIM-OPT' and dst in ('ZZ', 'ZN'):
        return [('PIM-OPT', 'ND'), ('ND', 'NZ'), ('NZ', dst)]

    raise ValueError(
        f'Unsupported NPU weight conversion chain: {src}->{dst}. '
        'Please add an explicit rule instead of relying on fallback.'
    )



def _resolve_pim_weight_load_steps(src_fmt: str) -> List[Tuple[str, str]]:
    src = _normalize_weight_format_token(src_fmt, allow_compute=False)
    if src == 'DUAL':
        src = 'PIM-OPT'
    if src == 'ND':
        return [('ND', 'PIM-OPT')]
    if src == 'PIM-OPT':
        return [('PIM-OPT', 'PIM-OPT')]
    if src == 'NZ':
        return [('NZ', 'ND'), ('ND', 'PIM-OPT')]
    raise ValueError(
        f'Unsupported PIM weight load chain: {src}->PIM-OPT. '
        'Please add an explicit rule instead of relying on fallback.'
    )

# Canonical op-key sets for NPU backends
NPU_ACT_KEYS = {
    'gelu','relu','silu','swish','mish','tanh','sigmoid','relu6','leaky_relu','elu','hardtanh','selu','prelu',
    'geglu','swiglu','swi_glu','silu_glu','glu_act','activation','act'
}
NPU_NORM_KEYS = {
    'layernorm','layer_norm','ln','rmsnorm','rms_norm','norm',
    'groupnorm','group_norm','instancenorm','instance_norm','batchnorm','batch_norm'
}
NPU_GEMM_KEYS = {
    'q_proj','k_proj','v_proj','wo_proj','ffn_up','ffn_gate','ffn_down','score','output',
}
# Fast-mode analytical compute-pipe classification.  CUBE is used for
# GEMM/BMM/MMAD/linear-projection-like work; VEC is used for nonlinear,
# normalization, elementwise, and selection/combine work.  Unknown operators
# deliberately fall back to the device's default ``tflops``.
FAST_CUBE_OP_KEYS = {
    *NPU_GEMM_KEYS,
    'q', 'k', 'v', 'o', 'wo',
    'ffn_w1', 'ffn_w2', 'ffn_w3', 'ffn_up', 'ffn_down', 'ffn_gate',
    'dsv4_q_down', 'dsv4_q_up',
    'dsv4_kv_compress', 'dsv4_index_kv_compress', 'dsv4_window_kv',
    'dsv4_indexer_q', 'dsv4_index_score',
    'dsv4_o_g1', 'dsv4_o_g2',
    'mhc_mix', 'moe_router', 'router',
}
FAST_VEC_OP_KEYS = {
    'softmax', 'add', 'identity', 'residual', 'dropout',
    'kv_write', 'k_write', 'v_write', 'kv_read', 'k_read', 'v_read',
    'allreduce', 'reduce', 'scatter',
    'dsv4_topk', 'moe_combine', 'moe_shared_combine',
    *NPU_ACT_KEYS,
    *NPU_NORM_KEYS,
}
NPU_ROUTER_KEYS = {
    'router', 'moe_router',
    # DeepSeek-V4 custom fused/proxy operators use the analytic fallback in
    # LLMCompass/LUT backends because they combine GEMM, compression, routing,
    # top-k, and token-wise mixing semantics that are not single primitive ops.
    'dsv4_q_down', 'dsv4_q_up', 'dsv4_kv_compress', 'dsv4_index_kv_compress', 'dsv4_window_kv',
    'dsv4_indexer_q', 'dsv4_index_score', 'dsv4_topk',
    'dsv4_o_g1', 'dsv4_o_g2', 'mhc_mix', 'moe_combine', 'moe_shared_combine',
    'reduce', 'scatter',
}


def _normalize_npu_backend_safe(backend: Optional[str]) -> str:
    raw = (backend or '').strip().lower()
    if not raw:
        raw = 'fast'
    try:
        b = _normalize_npu_backend(backend)
        if b:
            return str(b)
    except Exception:
        pass
    return raw

# --------------------------------------------------------------------------------------
# Device-name matching helpers (hardware.json -> device-specific params)
# --------------------------------------------------------------------------------------
_DEVICE_NAME_FAMILY_PATTERNS = (
    (re.compile(r'(?i)^ascend_?950dt(?:_|$)'), 'Ascend_950DT'),
    (re.compile(r'(?i)^ascend_?910b(?:_|$)'), 'Ascend_910B'),
    (re.compile(r'(?i)^ascend_?310b(?:_|$)'), 'Ascend_310B'),
    (re.compile(r'(?i)^(?:nvidia_)?a100(?:_|$)'), 'A100'),
    (re.compile(r'(?i)^(?:aim[ _-]?)?pim(?:\d+|[ _-]|$)'), 'pim'),
)

def _device_family_key_from_name(name: str) -> str:
    """Map hardware.json device "name" to a stable family key (e.g., Ascend_910B / A100)."""
    s = str(name or '').strip()
    if not s:
        return ''
    for rx, key in _DEVICE_NAME_FAMILY_PATTERNS:
        try:
            if rx.match(s):
                return str(key)
        except Exception:
            continue
    return ''

def _lookup_cfg_by_device_name(cfg: Any, dev: Optional[Any]) -> Any:
    """Return cfg override from cfg['by_device_name'] (or cfg['by_name']) based on dev.name prefix."""
    if dev is None or not isinstance(cfg, dict) or not cfg:
        return None
    name_map = cfg.get('by_device_name', cfg.get('by_name', None))
    if not isinstance(name_map, dict) or not name_map:
        return None

    dev_name = str(getattr(dev, 'name', '') or '').strip()
    fam = _device_family_key_from_name(dev_name)

    # (1) Prefer canonical family key lookup (Ascend_910B / A100)
    if fam:
        if fam in name_map:
            return name_map.get(fam)
        if fam.lower() in name_map:
            return name_map.get(fam.lower())
        if fam.upper() in name_map:
            return name_map.get(fam.upper())
        if str(fam).lower() == 'pim':
            for alias in (
                'Aim PIM', 'AIM PIM', 'aim pim',
                'Aim_PIM', 'AIM_PIM', 'aim_pim',
                'PIM',
            ):
                if alias in name_map:
                    return name_map.get(alias)
                if alias.lower() in name_map:
                    return name_map.get(alias.lower())
                if alias.upper() in name_map:
                    return name_map.get(alias.upper())

    # (2) Fallback: treat keys as raw prefixes
    if dev_name:
        dn = dev_name.lower()
        for k, v in name_map.items():
            ks = str(k or '').strip()
            if not ks:
                continue
            if dn.startswith(ks.lower()):
                return v

    return None

# NPU op-name aliases (graph node labels -> canonical backend keys)
_NPU_OP_ALIASES: Dict[str, str] = {
    # Attention projections
    'q': 'q_proj',
    'k': 'k_proj',
    'v': 'v_proj',
    'o': 'wo_proj',
    'wo': 'wo_proj',

    # Attention core
    'qk': 'score',
    'score': 'score',
    'softmax': 'softmax',
    'sv': 'output',
    'output': 'output',

    # FFN / MLP (LLaMA SwiGLU naming)
    'ffn_w1': 'ffn_gate',
    'ffn_w3': 'ffn_up',
    'ffn_w2': 'ffn_down',
    'w1': 'ffn_gate',
    'w3': 'ffn_up',
    'w2': 'ffn_down',
    'mlp_w1': 'ffn_gate',
    'mlp_w3': 'ffn_up',
    'mlp_w2': 'ffn_down',

    # Residual / elementwise
    'residual': 'add',

    # MoE router / gate
    'router': 'moe_router',
    'moe_router': 'moe_router',

    # DeepSeek-V4 custom ops (kept as custom keys; backends fall back to
    # analytic estimates via NPU_ROUTER_KEYS).
    'dsv4_q_down': 'dsv4_q_down',
    'dsv4_q_up': 'dsv4_q_up',
    'dsv4_kv_compress': 'dsv4_kv_compress',
    'dsv4_index_kv_compress': 'dsv4_index_kv_compress',
    'dsv4_window_kv': 'dsv4_window_kv',
    'dsv4_indexer_q': 'dsv4_indexer_q',
    'dsv4_index_score': 'dsv4_index_score',
    'dsv4_topk': 'dsv4_topk',
    'dsv4_o_g1': 'dsv4_o_g1',
    'dsv4_o_g2': 'dsv4_o_g2',
    'mhc_mix': 'mhc_mix',
    'moe_combine': 'moe_combine',
    'moe_shared_combine': 'moe_shared_combine',
}


# Token-level matcher for embedded op names.
_NPU_OP_TOKEN_RE = re.compile(
    r'(^|_)(qk|sv|softmax|ffn_w1|ffn_w2|ffn_w3|q_proj|k_proj|v_proj|wo_proj|moe_router|router|dsv4_q_down|dsv4_q_up|dsv4_kv_compress|dsv4_index_kv_compress|dsv4_window_kv|dsv4_indexer_q|dsv4_index_score|dsv4_topk|dsv4_o_g1|dsv4_o_g2|mhc_mix|moe_combine|moe_shared_combine)($|_)'
)


def _normalize_npu_op_key(op_key: str) -> str:
    """Normalize a graph op label to a canonical NPU backend op-key."""
    s = (op_key or '').strip().lower()
    if not s:
        return s
    s = s.replace('-', '_').replace(' ', '_')
    s = re.sub(r'__+', '_', s)
    s2 = re.sub(r'^(l|layer|block)\d+_', '', s)
    s2 = re.sub(r'_(s|shard)\d+$', '', s2)
    s2 = re.sub(r'_e\d+$', '', s2)          # expert id
    s2 = re.sub(r'_expert\d+$', '', s2)
    s2 = re.sub(r'_tp\d+$', '', s2)
    s2 = s2.strip('_')
    if not s2:
        s2 = s

    # Direct alias hit.
    if s2 in _NPU_OP_ALIASES:
        return _NPU_OP_ALIASES[s2]

    # Try last token (handles cases like "attn_q", "proj_q", etc.).
    tail = s2.split('_')[-1] if '_' in s2 else s2
    if tail in _NPU_OP_ALIASES:
        return _NPU_OP_ALIASES[tail]

    # Tokenized / embedded match.
    m = _NPU_OP_TOKEN_RE.search(s2)
    if m:
        tok = (m.group(2) or '').lower()
        if tok in _NPU_OP_ALIASES:
            return _NPU_OP_ALIASES[tok]
        return tok

    return s2

def _is_norm_like(op_key: str) -> bool:
    s = (op_key or '').strip().lower()
    if not s:
        return False
    if s in NPU_NORM_KEYS:
        return True
    s = s.replace('-', '_')
    # Common fused / variant spellings
    if ('rmsnorm' in s) or ('layernorm' in s):
        return True
    if s.startswith('ln') and (len(s) == 2 or s[2:].isdigit()):
        return True
    if s.endswith('norm'):
        return True
    if 'norm' in s:
        # tokenized match: _norm / _layernorm / _rmsnorm ...
        if re.search(r'(^|_)(ln|layernorm|rmsnorm|groupnorm|instancenorm|batchnorm|norm)($|_)', s):
            return True
    return False

__all__ = [name for name in globals() if not name.startswith("__")]
