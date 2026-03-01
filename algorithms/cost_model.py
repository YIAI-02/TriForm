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
from task_graph import TaskGraph, TaskNode
from hardware import Cluster, DeviceSpec
from plan_label import PlanLabel
from config import HOST_NAME, DEVICE_PREFERRED_FORMAT, FORMAT_SIZE_MULTIPLIER, FORMAT_CONV_BW_GBs, FORMAT_CONV_OVERHEAD_US, PIM_FREQ_GHZ, GB_FREQ_GHZ, OPERATOR_DEVICE_ALLOWED, NONOVERLAP_TIME, PIM_RUNTIME_LRU_THRESHOLD
import logging
from stats_recorder import SimulationLogger, get_simulation_logger, reset_simulation_logger
from abc import ABC, abstractmethod
from cost_model_pim_backend import (
    _get_pim_latency_via_trace,
    _simulate_weight_loading_latency,
    _normalize_pim_op,
    PIM_TRACE_SUPPORTED_OPS,
)
from cost_model_npu_llmcompass_backend import (
    _normalize_npu_backend,
    _llmcompass_guess_device_key,
    _llmcompass_simulate_matmul_s,
    _llmcompass_simulate_softmax_s,
    _llmcompass_simulate_layernorm_s,
    _llmcompass_simulate_gelu_s,
)
from cost_model_npu_ascend_backend import (
    _predict_mmad_latency_us_from_lut,
    _predict_softmax_latency_us_from_lut,
    _predict_gelu_latency_us_from_lut,
    _predict_layernorm_latency_us_from_lut,
)

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: True)
DTYPE_BYTES: Dict[str, float] = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}

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
    (re.compile(r'(?i)^ascend_?910b(?:_|$)'), 'Ascend_910B'),
    (re.compile(r'(?i)^(?:nvidia_)?a100(?:_|$)'), 'A100'),
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
}


# Token-level matcher for embedded op names.
_NPU_OP_TOKEN_RE = re.compile(
    r'(^|_)(qk|sv|softmax|ffn_w1|ffn_w2|ffn_w3|q_proj|k_proj|v_proj|wo_proj)($|_)'
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

@dataclass(frozen=True)
class NpuOpContext:
    op_key: str
    attrs: Dict[str, Any]
    batch: int
    seq_len: int
    phase: str
    q_len: int
    kv_len: int
    dim: int
    ffn_dim: int
    q_heads: int
    kv_heads: int
    head_dim: int
    q_dim: int
    kv_dim: int
    o_dim: int
    causal: bool
    attn_pattern: str
    mem_s: float

class NpuBackendBase(ABC):
    name: str = 'base'
    @abstractmethod
    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, ctx: NpuOpContext) -> float:
        """Return estimated end-to-end op latency on NPU (seconds), including memory lower bound."""
        raise NotImplementedError

    def _fallback_fast_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, ctx: NpuOpContext) -> float:
        flops = float(cm.estimate_flops(node, ctx.batch, ctx.seq_len, ctx.phase))
        compute_s = cm.flop_time(flops, dev)
        mem_s = ctx.mem_s
        return max(compute_s, mem_s)



class NpuFastBackend(NpuBackendBase):
    name = 'fast'

    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, ctx: NpuOpContext) -> float:
        logger.debug(str(f"[NPU][FAST] {getattr(node,'name','?')}"))
        return float(self._fallback_fast_s(cm, node, dev, ctx))


class NpuLlmCompassBackend(NpuBackendBase):
    name = 'llmcompass'

    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, ctx: NpuOpContext) -> float:
        """LLMCompass-backed latency estimate."""
        device_key = _llmcompass_guess_device_key(dev)
        op = (ctx.op_key or '').strip().lower()

        if op in ('add', 'identity', 'allreduce', 'k_write', 'v_write', 'kv_write'):
            logger.debug(str(f'[NPU-ELEM][LLMCompass] op={op} device={device_key} mem_s={ctx.mem_s}'))
            return float(ctx.mem_s)

        # (a) Softmax
        if op == 'softmax':
            M_rows = max(1, int(ctx.batch) * max(1, int(ctx.q_heads)) * max(1, int(ctx.q_len)))
            is_dense = str(ctx.attn_pattern).lower() in ('dense', 'none', 'off', 'disabled')
            if str(ctx.phase) == 'prefill':
                K_cols = max(1, int(ctx.seq_len if is_dense else ctx.kv_len))
            else:
                K_cols = max(1, int(ctx.kv_len))

            lat_s = _llmcompass_simulate_softmax_s(device_key, cm.dtype, int(M_rows), int(K_cols))
            logger.debug(str(
                f'[NPU-SOFTMAX][LLMCompass] device={device_key} M={M_rows} K={K_cols} '
                f'phase={ctx.phase} causal={ctx.causal} s={lat_s}'
            ))
            return float(lat_s + float(ctx.mem_s))

        # (b) Activation (use GELU as proxy)
        if op in NPU_ACT_KEYS:
            data_len = max(1, int(ctx.batch)) * max(1, int(ctx.q_len)) * max(1, int(ctx.ffn_dim if ctx.ffn_dim > 0 else ctx.dim))
            lat_s = _llmcompass_simulate_gelu_s(device_key, cm.dtype, int(data_len))
            logger.debug(str(f'[NPU-ACT][LLMCompass] device={device_key} op={op} data_len={data_len} s={lat_s}'))
            return float(lat_s + float(ctx.mem_s))

        # (c) Norm
        if _is_norm_like(op):
            rows = max(1, int(ctx.batch)) * max(1, int(ctx.q_len))
            lat_s = _llmcompass_simulate_layernorm_s(device_key, cm.dtype, int(rows), int(ctx.dim))
            logger.debug(str(f'[NPU-NORM][LLMCompass] device={device_key} op={op} rows={rows} dim={ctx.dim} s={lat_s}'))
            return float(lat_s + float(ctx.mem_s))

        # (d) Matmul-like (GEMM / BatchedMatmul)
        if op in NPU_GEMM_KEYS:
            # ---- Attention score / output: use BatchedMatmul ----
            if op in ('score', 'output'):
                bmm_batch = max(1, int(ctx.batch) * max(1, int(ctx.q_heads)))
                M_mm = max(1, int(ctx.q_len))
                if op == 'score':
                    # [B*H, Tq, Dh] x [B*H, Dh, Tk] => [B*H, Tq, Tk]
                    N_mm = max(1, int(ctx.kv_len))
                    K_mm = max(1, int(ctx.head_dim))
                else:
                    # [B*H, Tq, Tk] x [B*H, Tk, Dh] => [B*H, Tq, Dh]
                    N_mm = max(1, int(ctx.head_dim))
                    K_mm = max(1, int(ctx.kv_len))

                lat_s = _llmcompass_simulate_matmul_s(
                    device_key, cm.dtype, int(M_mm), int(N_mm), int(K_mm),
                    batch=int(bmm_batch), batched=True,
                )
                logger.debug(str(
                    f'[NPU-MMAD][LLMCompass][BMM] device={device_key} '
                    f'batch={bmm_batch} M={M_mm} N={N_mm} K={K_mm} '
                    f'phase={ctx.phase} causal={ctx.causal} s={lat_s}'
                ))
                return float(lat_s + float(ctx.mem_s))

            # ---- Projections / FFN: use GEMM (fold batch*tokens into M) ----
            M_mm = max(1, int(ctx.batch) * max(1, int(ctx.q_len)))
            if op == 'q_proj':
                # [B*T, D] x [D, q_dim]
                K_mm = max(1, int(ctx.dim))
                N_mm = max(1, int(ctx.q_dim) if int(ctx.q_dim) > 0 else int(ctx.dim))
            elif op in ('k_proj', 'v_proj'):
                # [B*T, D] x [D, kv_dim]
                K_mm = max(1, int(ctx.dim))
                N_mm = max(1, int(ctx.kv_dim) if int(ctx.kv_dim) > 0 else int(ctx.dim))
            elif op == 'wo_proj':
                # [B*T, o_dim] x [o_dim, D]
                K_mm = max(1, int(ctx.o_dim) if int(ctx.o_dim) > 0 else int(ctx.dim))
                N_mm = max(1, int(ctx.dim))
            elif op in ('ffn_up', 'ffn_gate'):
                # [B*T, D] x [D, ffn_dim]
                K_mm = max(1, int(ctx.dim))
                N_mm = max(1, int(ctx.ffn_dim) if int(ctx.ffn_dim) > 0 else int(4 * int(ctx.dim)))
            elif op == 'ffn_down':
                # [B*T, ffn_dim] x [ffn_dim, D]
                K_mm = max(1, int(ctx.ffn_dim) if int(ctx.ffn_dim) > 0 else int(4 * int(ctx.dim)))
                N_mm = max(1, int(ctx.dim))
            else:
                # Should be unreachable due to NPU_GEMM_KEYS guard.
                raise RuntimeError(f"[LLMCompass] Internal: unhandled GEMM op='{op}'")

            lat_s = _llmcompass_simulate_matmul_s(device_key, cm.dtype, int(M_mm), int(N_mm), int(K_mm))
            logger.debug(str(
                f'[NPU-MMAD][LLMCompass][GEMM] device={device_key} op={op} '
                f'M={M_mm} N={N_mm} K={K_mm} phase={ctx.phase} s={lat_s}'
            ))
            return float(lat_s + float(ctx.mem_s))

        # (e) Unknown op -> HARD ERROR
        raise RuntimeError(
            f"[LLMCompass] Unrecognized NPU op_key='{op}'. "
            f"Supported categories: softmax, norm-like, activation-like, "
            f"gemm-like ({sorted(NPU_GEMM_KEYS)}), elem-like(add/identity/...). "
            f"Node={getattr(node, 'name', '?')}"
        )



class NpuAscend310BLutBackend(NpuBackendBase):
    """Ascend 310B LUT backend.

    Uses lookup-table (CSV/JSON/XLSX) for:
      - MMAD/GEMM/BMM (via mmad_lut.*)
      - Softmax (via softmax_lut.*)
      - Norm (RMSNorm/LN/etc., via rmsnorm_lut.* or layernorm_lut.*)
      - Activation (GELU as proxy for any activation, via gelu_lut.*)

    If a key is missing, the LUT module performs interpolation.
    """

    name = 'ascend_310b_lut'

    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, ctx: NpuOpContext) -> float:
        op = (ctx.op_key or '').strip().lower()

        # (a) Elementwise / bookkeeping
        if op in ('add', 'identity', 'allreduce', 'k_write', 'v_write', 'kv_write'):
            logger.debug(str(f'[NPU-ELEM][ASCEND-LUT] op={op} mem_s={ctx.mem_s}'))
            return float(ctx.mem_s)

        # (b) Softmax
        if op == 'softmax':
            M_rows = max(1, int(ctx.batch) * max(1, int(ctx.q_heads)) * max(1, int(ctx.q_len)))
            is_dense = str(ctx.attn_pattern).lower() in ('dense', 'none', 'off', 'disabled')
            if str(ctx.phase) == 'prefill':
                K_cols = max(1, int(ctx.seq_len if is_dense else ctx.kv_len))
            else:
                K_cols = max(1, int(ctx.kv_len))

            us = _predict_softmax_latency_us_from_lut(int(M_rows), int(K_cols), phase=str(ctx.phase), causal=bool(ctx.causal))
            logger.debug(str(f'[NPU-SOFTMAX][ASCEND-LUT] M={M_rows} K={K_cols} phase={ctx.phase} causal={ctx.causal} us={us}'))
            if us is not None:
                return float(max(float(us) * 1e-06, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (c) Activation (GELU proxy -> treat as any activation function)
        if op in NPU_ACT_KEYS:
            width = int(ctx.ffn_dim if int(ctx.ffn_dim) > 0 else int(ctx.dim))
            data_len = max(1, int(ctx.batch)) * max(1, int(ctx.q_len)) * max(1, width)
            us = _predict_gelu_latency_us_from_lut(int(data_len))
            logger.debug(str(f'[NPU-ACT][ASCEND-LUT] op={op} data_len={data_len} us={us}'))
            if us is not None:
                return float(max(float(us) * 1e-06, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (d) Norm (RMSNorm LUT is treated as generic norm LUT)
        if _is_norm_like(op):
            rows = max(1, int(ctx.batch) * max(1, int(ctx.q_len)))
            width = max(1, int(ctx.dim))
            us = _predict_layernorm_latency_us_from_lut(int(rows), int(width))
            logger.debug(str(f'[NPU-NORM][ASCEND-LUT] op={op} rows={rows} width={width} us={us}'))
            if us is not None:
                return float(max(float(us) * 1e-06, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (e) MMAD / GEMM / BMM
        if op in NPU_GEMM_KEYS:
            # ---- Attention core: BMM folded into GEMM ----
            if op in ('score', 'output'):
                bmm_batch = max(1, int(ctx.batch) * max(1, int(ctx.q_heads)))
                M_mm = max(1, int(bmm_batch) * max(1, int(ctx.q_len)))

                is_dense = str(ctx.attn_pattern).lower() in ('dense', 'none', 'off', 'disabled')
                if str(ctx.phase) == 'prefill':
                    Tk = max(1, int(ctx.seq_len if is_dense else ctx.kv_len))
                else:
                    Tk = max(1, int(ctx.kv_len))

                Dh = int(ctx.head_dim) if int(ctx.head_dim) > 0 else 0
                if Dh <= 0:
                    try:
                        Dh = max(1, int(int(ctx.dim) // max(1, int(ctx.q_heads))))
                    except Exception:
                        Dh = 1

                if op == 'score':
                    N_mm = int(Tk)
                    K_mm = int(Dh)
                else:
                    N_mm = int(Dh)
                    K_mm = int(Tk)

            # ---- Projections / FFN: GEMM (fold batch*tokens into M) ----
            else:
                M_mm = max(1, int(ctx.batch) * max(1, int(ctx.q_len)))

                if op == 'q_proj':
                    K_mm = max(1, int(ctx.dim))
                    N_mm = max(1, int(ctx.q_dim) if int(ctx.q_dim) > 0 else int(ctx.dim))
                elif op in ('k_proj', 'v_proj'):
                    K_mm = max(1, int(ctx.dim))
                    N_mm = max(1, int(ctx.kv_dim) if int(ctx.kv_dim) > 0 else int(ctx.dim))
                elif op == 'wo_proj':
                    # [B*T, o_dim] x [o_dim, D]
                    K_mm = max(1, int(ctx.o_dim) if int(ctx.o_dim) > 0 else int(ctx.dim))
                    N_mm = max(1, int(ctx.dim))
                elif op in ('ffn_up', 'ffn_gate'):
                    K_mm = max(1, int(ctx.dim))
                    N_mm = max(1, int(ctx.ffn_dim) if int(ctx.ffn_dim) > 0 else int(4 * int(ctx.dim)))
                elif op == 'ffn_down':
                    K_mm = max(1, int(ctx.ffn_dim) if int(ctx.ffn_dim) > 0 else int(4 * int(ctx.dim)))
                    N_mm = max(1, int(ctx.dim))
                else:
                    return float(self._fallback_fast_s(cm, node, dev, ctx))

            us = _predict_mmad_latency_us_from_lut(int(M_mm), int(N_mm), int(K_mm))
            logger.debug(str(f'[NPU-MMAD][ASCEND-LUT] op={op} M={M_mm} N={N_mm} K={K_mm} us={us}'))
            if us is not None:
                return float(max(float(us) * 1e-06, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (f) Unknown -> fallback
        return float(self._fallback_fast_s(cm, node, dev, ctx))


def build_npu_backend(backend: Optional[str]) -> NpuBackendBase:
    raw = (backend or '').strip().lower()
    try:
        b = _normalize_npu_backend(backend)
    except Exception:
        b = raw or 'fast'

    if not b:
        b = 'fast'

    if b == 'fast':
        return NpuFastBackend()
    if b == 'llmcompass':
        return NpuLlmCompassBackend()

    # Ascend 310B LUT (keep old name as alias)
    if b in ('ascend_310b_lut', 'ascend_310b', 'ascend310b', 'ascend'):
        return NpuAscend310BLutBackend()

    raise ValueError(
        f"Unsupported npu_backend='{backend}'. "
        f"Expected one of: fast, llmcompass, ascend_310b_lut (alias ascend_310b_json)"
    )

# ---------------------------------------------------------------------------
# PIM backends
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PimOpContext:
    op_key: str
    attrs: Dict[str, Any]
    batch: int
    seq_len: int
    phase: str
    dim: int
    n_heads: int
    n_kv_heads: int
    ffn_dim: int
    kv_in_pim: bool

class PimBackendBase(ABC):
    name: str = 'base'

    @abstractmethod
    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, label: PlanLabel, ctx: PimOpContext) -> float:
        """Return estimated end-to-end op latency on PIM (seconds)."""
        raise NotImplementedError

    @abstractmethod
    def weight_load_s(self, cm: "CostModel", weight_bytes: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def activation_read_s(self, cm: "CostModel", activation_bytes_nd: int) -> float:
        raise NotImplementedError


class PimFastBackend(PimBackendBase):
    name = 'fast'

    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, label: PlanLabel, ctx: PimOpContext) -> float:
        logger.debug(str(f"[PIM][FAST] {getattr(node,'name','?')}"))
        rd, wr = cm.estimate_activation_bytes(node, ctx.batch, ctx.seq_len, ctx.phase)
        mem_t = float(cm.pim_mem_time(int(rd), int(wr), dev))
        flops = float(cm.estimate_flops(node, ctx.batch, ctx.seq_len, ctx.phase))
        compute_s = cm.flop_time(flops, dev)
        return max(compute_s, mem_t)

    def weight_load_s(self, cm: "CostModel", weight_bytes: int) -> float:
        logger.debug(str(f"[PIM][FAST] weight_load bytes={weight_bytes}"))
        pim_devs = cm.cluster.devices_by_type('pim')
        if pim_devs:
            return float(cm.pim_mem_time(int(weight_bytes), 0, pim_devs[0]))
        return 0.0

    def activation_read_s(self, cm: "CostModel", activation_bytes_nd: int) -> float:
        logger.debug(str(f"[PIM][FAST] activation_read bytes={activation_bytes_nd}"))
        pim_devs = getattr(cm.cluster, 'devices_by_type', lambda *_: [])('pim')
        if pim_devs:
            return float(cm.pim_mem_time(int(activation_bytes_nd), 0, pim_devs[0]))
        return 0.0


class PimTraceBackend(PimBackendBase):
    name = 'trace'

    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, label: PlanLabel, ctx: PimOpContext) -> float:
        # Trace mode: requires configs.
        if not cm.pim_config_path or not cm.ramulator_config_path:
            logger.debug(str(f'[PIM] Warning: PIM configs not set, returning 0 for {getattr(node, "name", "?")}'))
            return 0.0

        op_in = str(ctx.op_key) if ctx.op_key is not None else ''
        op_norm = _normalize_pim_op(op_in) if op_in else ''
        traceable = bool(op_norm) and (op_norm in PIM_TRACE_SUPPORTED_OPS)

        # Basic parameter guardrails
        if (not op_in) or int(ctx.dim) <= 0 or int(ctx.n_heads) <= 0:
            logger.debug(str(
                f'[PIM] Warning: Insufficient parameters for {getattr(node,"name","?")} '
                f'(op={ctx.op_key}, dim={ctx.dim}, heads={ctx.n_heads})'
            ))

            return 0.0

        # Prefer exact trace-based latency when supported.
        if traceable:
            try:
                model_dict = cm.get_model_dict()

                # --- Shard-aware dims (TP) ---
                attrs = (ctx.attrs or {}) if isinstance(ctx.attrs, dict) else {}
                hd = int(attrs.get('head_dim', 0) or 0)
                q_dim = int(attrs.get('q_dim', 0) or 0)
                kv_dim = int(attrs.get('kv_dim', 0) or 0)
                o_dim = int(attrs.get('o_dim', 0) or 0)

                if hd <= 0:
                    try:
                        hd = int(int(ctx.dim) // max(1, int(ctx.n_heads)))
                    except Exception:
                        hd = 0
                if q_dim <= 0 and hd > 0:
                    q_dim = int(max(1, int(ctx.n_heads)) * int(hd))
                if kv_dim <= 0 and hd > 0:
                    kv_dim = int(max(1, int(ctx.n_kv_heads)) * int(hd))
                if o_dim <= 0:
                    # Attention output (pre-WO) is typically q_dim.
                    o_dim = int(q_dim) if q_dim > 0 else 0

                compute_time = float(
                    _get_pim_latency_via_trace(
                        op=str(op_norm),
                        pim_config=cm.pim_config_path,
                        ramulator_config=cm.ramulator_config_path,
                        dim=int(ctx.dim),
                        n_heads=int(ctx.n_heads),
                        n_kv_heads=int(ctx.n_kv_heads),
                        ffn_dim=int(ctx.ffn_dim),
                        seqlen=int(ctx.seq_len) if int(ctx.seq_len) > 0 else None,
                        batch=int(ctx.batch) if int(ctx.batch) > 0 else 1,
                        phase=str(ctx.phase),
                        model_dict=model_dict,
                        use_cache=bool(cm.pim_cache_enabled),
                        head_dim=int(hd) if int(hd) > 0 else None,
                        q_dim=int(q_dim) if int(q_dim) > 0 else None,
                        kv_dim=int(kv_dim) if int(kv_dim) > 0 else None,
                        o_dim=int(o_dim) if int(o_dim) > 0 else None,
                    )
                )
                return float(compute_time)
            except Exception as e:
                # Do not fail scheduling when trace simulation cannot run.
                # Fall back to compute-only estimate (NO mem_time for CENT backend).
                logger.debug(str(
                    f"[PIM] Trace backend failed for {getattr(node,'name','?')}: "
                    f"op='{op_in}' normalized='{op_norm}' err={e}. "
                    f"Falling back to compute-only estimate."
                ))
        else:
            logger.debug(str(
                f"[PIM] Trace backend: skip unsupported op for {getattr(node,'name','?')}: "
                f"op='{op_in}' normalized='{op_norm}'. Using compute-only estimate."
            ))

        # Fallback path: compute-only estimate (seconds).
        try:
            flops = float(cm.estimate_flops(node, ctx.batch, ctx.seq_len, ctx.phase))
            t = float(cm.flop_time(flops, dev))
            return float(t)
        except Exception as e:
            logger.debug(str(f"[PIM] compute-only fallback failed for {getattr(node,'name','?')}: {e}"))
            return 0.0


        return float(compute_time + mem_time)
    def weight_load_s(self, cm: "CostModel", weight_bytes: int) -> float:
        # Keep the original behavior: fast-mode bypass
        if bool(cm.pim_fast_mode):
            return PimFastBackend().weight_load_s(cm, weight_bytes)

        pim_devs = getattr(cm.cluster, 'devices_by_type', lambda *_: [])('pim')
        if pim_devs:
            return float(cm.pim_mem_time(int(weight_bytes), 0, pim_devs[0]))
        return 0.0

    def activation_read_s(self, cm: "CostModel", activation_bytes_nd: int) -> float:
        # Keep the original behavior: fast-mode bypass
        if bool(cm.pim_fast_mode):
            return PimFastBackend().activation_read_s(cm, activation_bytes_nd)

        pim_devs = getattr(cm.cluster, 'devices_by_type', lambda *_: [])('pim')
        if pim_devs:
            return float(cm.pim_mem_time(int(activation_bytes_nd), 0, pim_devs[0]))
        return 0.0


def build_pim_backend(pim_fast_mode: bool) -> PimBackendBase:
    return PimFastBackend() if bool(pim_fast_mode) else PimTraceBackend()


class CostModel:
    def __init__(self, cluster: Cluster, dtype: str='fp16', pim_config_path: Optional[Path]=None, gb_config_path: Optional[Path]=None, ramulator_config_path: Optional[Path]=None, simulation_log_file: Optional[Path]=None, debug_traces: bool=False, model_dict: Optional[Dict]=None, pim_fast_mode: bool=False, npu_backend: Optional[str]=None, tp_qkv: int=1, tp_ffn: int=1):
        self.cluster = cluster
        self.dtype = dtype
        self.pim_config_path = pim_config_path
        self.gb_config_path = gb_config_path
        self.ramulator_config_path = ramulator_config_path
        self.debug_traces = debug_traces
        self.pim_fast_mode = pim_fast_mode  # When True, skip all trace simulations
        self.npu_backend = _normalize_npu_backend_safe(npu_backend)
        try:
            self.tp_qkv = max(1, int(tp_qkv or 1))
        except Exception:
            self.tp_qkv = 1
        try:
            self.tp_ffn = max(1, int(tp_ffn or 1))
        except Exception:
            self.tp_ffn = 1
        self.logger = get_simulation_logger(simulation_log_file)
        self.pim_cache_enabled = True
        self._shared_model_dict: Optional[Dict] = model_dict
        self.kv_pd_separation: bool = False
        if pim_config_path:
            if not pim_config_path.exists():
                raise ValueError(f'PIM config not found: {pim_config_path}')
            if model_dict is None:
                logger.debug(str('[WARNING] PIM config provided but model_dict is None. Call set_model_dict() before using PIM operations.'))
        if gb_config_path:
            if not gb_config_path.exists():
                raise ValueError(f'Global Buffer config not found: {gb_config_path}')
        if ramulator_config_path:
            if not ramulator_config_path.exists():
                raise ValueError(f'Ramulator config not found: {ramulator_config_path}')

        self._npu_backend_impl_name: Optional[str] = None
        self._npu_backend_impl: NpuBackendBase = NpuFastBackend()
        self._pim_backend_fast_mode: Optional[bool] = None
        self._pim_backend_impl: PimBackendBase = PimFastBackend()
        self._ensure_backend_impls()

    def _ensure_backend_impls(self) -> None:
        """(Re)build backend objects if user changes npu_backend / pim_fast_mode after __init__."""
        npu_name = _normalize_npu_backend_safe(self.npu_backend)
        if npu_name is None:
            npu_name = 'fast'
        if npu_name != getattr(self, '_npu_backend_impl_name', None):
            self._npu_backend_impl = build_npu_backend(npu_name)
            self._npu_backend_impl_name = npu_name

        pim_fast = bool(getattr(self, 'pim_fast_mode', False))
        if pim_fast != getattr(self, '_pim_backend_fast_mode', None):
            self._pim_backend_impl = build_pim_backend(pim_fast)
            self._pim_backend_fast_mode = pim_fast

    def set_model_dict(self, model_dict: Dict):
        if model_dict is None:
            raise ValueError('model_dict cannot be None')
        self._shared_model_dict = model_dict
        logger.debug(str(f'[CostModel] Model dictionary set with keys: {list(model_dict.keys())[:5]}...'))

    def get_model_dict(self) -> Dict:
        if self._shared_model_dict is None:
            raise RuntimeError('Model dictionary not set. You must call set_model_dict() or provide model_dict during initialization before using PIM operations.')
        return self._shared_model_dict

    def has_model_dict(self) -> bool:
        return self._shared_model_dict is not None

    def get_host_device(self) -> DeviceSpec:    
        if HOST_NAME in self.cluster.devices:
            return self.cluster.devices[HOST_NAME]
        cpus = self.cluster.devices_by_type('cpu')
        return cpus[0] if cpus else next(iter(self.cluster.devices.values()))

    def device_preferred_fmt(self, dev: DeviceSpec) -> str:
        return DEVICE_PREFERRED_FORMAT.get(dev.type, 'ND')

    def format_size(self, size_bytes: int, fmt: str) -> int:
        m = float(FORMAT_SIZE_MULTIPLIER.get(fmt, 1.0))
        return int(size_bytes * m)

    def _compute_utilization(self, flops: float, dev: DeviceSpec) -> float:
        """Heuristic utilization of peak compute throughput for small workloads."""
        cfg = getattr(_config, 'COMPUTE_UTILIZATION', None)
        if not isinstance(cfg, dict) or not cfg:
            return 1.0
        dev_type = str(getattr(dev, 'type', '') or '').lower()
        params = cfg.get(dev_type, cfg.get('default', cfg))

        self._ensure_backend_impls()
        if dev_type == 'npu' and str(getattr(self, '_npu_backend_impl_name', '') or '').lower() == 'fast':
            per = _lookup_cfg_by_device_name(cfg, dev)
            if not isinstance(per, dict) or not per:
                keys = list((cfg.get('by_device_name', cfg.get('by_name', {})) or {}).keys())
                raise NpuFastModeConfigError(
                    f"[NPU][FAST] COMPUTE_UTILIZATION missing by_device_name for dev_name='{getattr(dev,'name','')}'. "
                    f"available_keys={keys}"
                )
            params = per

        # Allow simple scalar override (constant utilization).
        if isinstance(params, (int, float)):
            u = float(params)
            return 1.0 if u <= 0 else min(1.0, u)
        if not isinstance(params, dict):
            return 1.0

        if not bool(params.get('enabled', True)):
            return 1.0

        min_u = float(params.get('min_util', params.get('min', 1.0)) or 0.0)
        max_u = float(params.get('max_util', params.get('max', 1.0)) or 1.0)

        # Clamp to [0, 1]
        min_u = max(0.0, min(1.0, min_u))
        max_u = max(0.0, min(1.0, max_u))
        if max_u <= 0.0:
            return 1.0
        if min_u > max_u:
            min_u, max_u = max_u, min_u

        flops_low = float(params.get('flops_low', params.get('low_flops', 0.0)) or 0.0)
        flops_high = float(params.get('flops_high', params.get('high_flops', 0.0)) or 0.0)
        curve = str(params.get('curve', params.get('mode', 'log_linear')) or 'log_linear').strip().lower()
        power = float(params.get('power', 1.0) or 1.0)
        power = max(1e-3, power)

        f = float(flops or 0.0)
        if f <= 0.0:
            return max(min_u, 1e-6)

        # If thresholds are not configured, treat as constant max utilization.
        if flops_low <= 0.0 or flops_high <= 0.0 or flops_high <= flops_low:
            return max(min(max_u, 1.0), 1e-6)

        if f <= flops_low:
            return max(min_u, 1e-6)
        if f >= flops_high:
            return max(max_u, 1e-6)

        if curve in ('linear',):
            x = (f - flops_low) / (flops_high - flops_low)
        elif curve in ('sigmoid', 'logistic'):
            # knee defaults to geometric mean; slope controls steepness.
            knee = float(params.get('knee_flops', math.sqrt(flops_low * flops_high)) or math.sqrt(flops_low * flops_high))
            slope = float(params.get('slope', 1.0) or 1.0)
            knee = max(1.0, knee)
            x = (math.log10(f) - math.log10(knee)) * slope
            s = 1.0 / (1.0 + math.exp(-x))
            u = min_u + (max_u - min_u) * s
            return max(min(1.0, u), 1e-6)
        else:
            x = (math.log10(f) - math.log10(flops_low)) / (math.log10(flops_high) - math.log10(flops_low))

        x = max(0.0, min(1.0, x))
        x = x ** power
        u = min_u + (max_u - min_u) * x
        return max(min(1.0, u), 1e-6)


    def effective_tflops(self, flops: float, dev: DeviceSpec) -> float:
        """Peak TFLOPS scaled by utilization."""
        t = float(getattr(dev, "tflops", 0.0) or 0.0)
        if t <= 0.0:
            # For CPU, keep the previous behavior: do not treat missing tflops as "free".
            if str(getattr(dev, 'type', '') or '').lower() == 'cpu':
                t = float(getattr(_config, 'CPU_FALLBACK_TFLOPS', 1e-3) or 1e-3)
            else:
                return 0.0
        util = float(self._compute_utilization(float(flops or 0.0), dev))
        return float(t * max(util, 1e-6))

    def flop_time(self, flops: float, dev: DeviceSpec) -> float:
        """Compute-bound time lower-bound (seconds) using peak*util throughput."""
        eff = float(self.effective_tflops(float(flops or 0.0), dev))
        if eff <= 0.0:
            return float("inf")
        return float(flops) / (eff * 1e12)

    def mem_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        bw = dev.mem_bw_GBs  * 1024 * 1024 * 1024.0
        return 0.0 if bw <= 0 else bytes_amount / bw


    # ---------------------------------------------------------------------
    # Kernel-launch / runtime overhead (NPU/GPU)
    # ---------------------------------------------------------------------
    def _kernel_launch_cfg(self, dev: Optional[DeviceSpec] = None) -> Dict[str, Any]:
        cfg = getattr(_config, 'KERNEL_LAUNCH_OVERHEAD', None)
        if not isinstance(cfg, dict) or not cfg:
            return {}
        merged = dict(cfg)
        if dev is None:
            return merged
        dev_type = str(getattr(dev, 'type', '') or '').lower()

        self._ensure_backend_impls()
        if dev_type == 'npu' and str(getattr(self, '_npu_backend_impl_name', '') or '').lower() == 'fast':
            per = _lookup_cfg_by_device_name(cfg, dev)
            if not isinstance(per, dict) or not per:
                keys = list((cfg.get('by_device_name', cfg.get('by_name', {})) or {}).keys())
                raise NpuFastModeConfigError(
                    f"[NPU][FAST] KERNEL_LAUNCH_OVERHEAD missing by_device_name for dev_name='{getattr(dev,'name','')}'. "
                    f"available_keys={keys}"
                )
            return dict(per)

    def kernel_launch_overhead_s(self, op_key: str, dev: DeviceSpec, *, phase: str = 'prefill') -> float:
        cfg = self._kernel_launch_cfg(dev)
        if not cfg or not bool(cfg.get('enabled', False)):
            return 0.0

        # Backend gating to avoid double-counting with empirical backends.
        apply_backends = cfg.get('apply_backends', None)
        if apply_backends not in (None, True, 'all', 'ALL'):
            name = str(getattr(self, '_npu_backend_impl_name', None) or getattr(self, 'npu_backend', '') or '')
            if isinstance(apply_backends, str):
                apply_list = [apply_backends]
            elif isinstance(apply_backends, (list, tuple, set)):
                apply_list = list(apply_backends)
            else:
                apply_list = []
            if apply_list and (name not in apply_list):
                return 0.0

        op = str(op_key or '').strip().lower()
        if not op:
            return 0.0

        # Phase scaling
        ph = str(phase or '').strip().lower()
        scale = 1.0
        ph_scale = cfg.get('phase_scale', None)
        if isinstance(ph_scale, dict) and ph_scale:
            try:
                scale = float(ph_scale.get(ph, 1.0) or 1.0)
            except Exception:
                scale = 1.0

        # Exact op override
        us = None
        by_op = cfg.get('by_op_us', cfg.get('by_op', None))
        if isinstance(by_op, dict) and by_op:
            if op in by_op:
                try:
                    us = float(by_op[op] or 0.0)
                except Exception:
                    us = 0.0

        # Category fallback
        if us is None:
            # Infer category
            if op == 'softmax':
                cat = 'softmax'
            elif _is_norm_like(op):
                cat = 'norm'
            elif op in NPU_ACT_KEYS:
                cat = 'activation'
            elif op in ('add', 'identity', 'residual', 'dropout'):
                cat = 'elem'
            elif op in NPU_GEMM_KEYS:
                cat = 'gemm'
            else:
                cat = 'default'

            by_cat = cfg.get('by_category_us', cfg.get('by_category', None))
            if isinstance(by_cat, dict) and by_cat and (cat in by_cat):
                try:
                    us = float(by_cat[cat] or 0.0)
                except Exception:
                    us = 0.0
            else:
                try:
                    us = float(cfg.get('default_us', cfg.get('default', 0.0)) or 0.0)
                except Exception:
                    us = 0.0

        us = max(0.0, float(us or 0.0)) * max(0.0, float(scale))
        return float(us) * 1e-6

    def _pim_parallel_access_bytes(self, dev: Optional[DeviceSpec] = None) -> int:
        cfg = {}
        try:
            if dev is not None:
                cfg = getattr(dev, 'pim_memory', None) or {}
        except Exception:
            cfg = {}
        if not cfg:
            cfg = getattr(self.cluster, 'pim_memory', None) or {}
        if not isinstance(cfg, dict):
            return 0
        addr = cfg.get('addr_map') or cfg.get('address_map') or cfg.get('addrmap') or {}
        if not isinstance(addr, dict):
            addr = {}

        unit = str(cfg.get('addr_map_unit', cfg.get('addr_map_units', 'bits')) or 'bits').strip().lower()

        # Line bytes (L_B)
        line_bytes = cfg.get('line_bytes') or cfg.get('line_bytes_B') or cfg.get('line_size_B')
        if line_bytes is None:
            off = addr.get('offset', 6)
            try:
                if unit in ('bits', 'bit'):
                    line_bytes = 1 << int(off)
                else:
                    # Treat offset as bytes when unit != bits
                    line_bytes = int(off)
            except Exception:
                line_bytes = 64
        try:
            line_bytes = int(line_bytes)
        except Exception:
            line_bytes = 64
        line_bytes = max(1, int(line_bytes))

        # Channel/bank parallelism
        ch = addr.get('channel', addr.get('channels', 0))
        bk = addr.get('bank', addr.get('banks', 0))
        try:
            if unit in ('bits', 'bit'):
                num_ch = 1 << int(ch) if ch is not None else 1
                num_bk = 1 << int(bk) if bk is not None else 1
            else:
                num_ch = int(ch) if ch is not None else 1
                num_bk = int(bk) if bk is not None else 1
        except Exception:
            num_ch, num_bk = 1, 1

        num_ch = max(1, int(num_ch))
        num_bk = max(1, int(num_bk))
        return int(line_bytes) * int(num_ch) * int(num_bk)

    def pim_read_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        """PIM read latency (seconds) using line-latency model when available."""
        return float(self.pim_mem_time(int(bytes_amount or 0), 0, dev))

    def pim_write_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        """PIM write latency (seconds) using line-latency model when available."""
        return float(self.pim_mem_time(0, int(bytes_amount or 0), dev))

    def pim_mem_time(self, read_bytes: int, write_bytes: int, dev: DeviceSpec) -> float:
        """
        PIM memory time estimation.
            n_rd = ceil(read_bytes  / bytes_per_access)
            n_wr = ceil(write_bytes / bytes_per_access)
         """
        read_bytes = int(read_bytes or 0)
        write_bytes = int(write_bytes or 0)
        if read_bytes <= 0 and write_bytes <= 0:
            return 0.0

        # Only meaningful for PIM; other devices use bandwidth-only model.
        if str(getattr(dev, 'type', '')).lower() != 'pim':
            return float(self.mem_time(int(read_bytes + write_bytes), dev))

        bytes_per_access = int(self._pim_parallel_access_bytes(dev) or 0)
        rd_lat_ns = float(getattr(dev, 'pim_read_latency_ns', getattr(dev, 'read_latency_ns', 0.0)) or 0.0)
        wr_lat_ns = float(getattr(dev, 'pim_write_latency_ns', getattr(dev, 'write_latency_ns', 0.0)) or 0.0)

        if bytes_per_access > 0 and (rd_lat_ns > 0.0 or wr_lat_ns > 0.0):
            import math
            n_rd = int(math.ceil(float(read_bytes) / float(bytes_per_access))) if read_bytes > 0 else 0
            n_wr = int(math.ceil(float(write_bytes) / float(bytes_per_access))) if write_bytes > 0 else 0
            return float(n_rd) * float(rd_lat_ns) * 1e-9 + float(n_wr) * float(wr_lat_ns) * 1e-9

        # Fallback: bandwidth model (kept for backward compatibility).
        bw = float(getattr(dev, 'mem_bw_GBs', 0.0) or 0.0) * (1024**3)
        if bw <= 0.0:
            return float('inf')
        t_rd = float(read_bytes) / bw if read_bytes > 0 else 0.0
        t_wr = float(write_bytes) / bw if write_bytes > 0 else 0.0
        return float(t_rd + t_wr)

    def cpu_read_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        """CPU read latency (seconds) using access-latency model when available."""
        return float(self.cpu_mem_time(int(bytes_amount or 0), 0, dev))

    def cpu_write_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        """CPU write latency (seconds) using access-latency model when available."""
        return float(self.cpu_mem_time(0, int(bytes_amount or 0), dev))

    def cpu_mem_time(self, read_bytes: int, write_bytes: int, dev: DeviceSpec) -> float:
        """CPU host memory time estimation.

        If ``dev`` is a CPU and (cpu_read_latency_ns/cpu_write_latency_ns) are provided,
        use a cacheline/access-count model:

            n_rd = ceil(read_bytes  / bytes_per_access)
            n_wr = ceil(write_bytes / bytes_per_access)
            T = n_rd * rd_lat + n_wr * wr_lat

        Otherwise, fall back to bandwidth-only ``mem_time``.
        """
        read_bytes = int(read_bytes or 0)
        write_bytes = int(write_bytes or 0)
        if read_bytes <= 0 and write_bytes <= 0:
            return 0.0

        if str(getattr(dev, 'type', '')).lower() != 'cpu':
            return float(self.mem_time(int(read_bytes + write_bytes), dev))

        rd_lat_ns = float(getattr(dev, 'cpu_read_latency_ns', 0.0) or 0.0)
        wr_lat_ns = float(getattr(dev, 'cpu_write_latency_ns', 0.0) or 0.0)
        bytes_per_access = int(getattr(dev, 'cpu_access_bytes_B', 64) or 64)

        if bytes_per_access > 0 and (rd_lat_ns > 0.0 or wr_lat_ns > 0.0):
            n_rd = int(math.ceil(float(read_bytes) / float(bytes_per_access))) if read_bytes > 0 else 0
            n_wr = int(math.ceil(float(write_bytes) / float(bytes_per_access))) if write_bytes > 0 else 0
            return float(n_rd) * float(rd_lat_ns) * 1e-9 + float(n_wr) * float(wr_lat_ns) * 1e-9

        return float(self.mem_time(int(read_bytes + write_bytes), dev))
    def comm_cost(self, src: DeviceSpec, dst: DeviceSpec, bytes_amount: int) -> float:
        """
            T = L + O + n_hat / B
            n_hat = n + ceil(n / MaxPayload) * FlitSize
        """
        if src.name == dst.name:
            return 0.0
        bytes_amount = int(bytes_amount or 0)
        if bytes_amount <= 0 or src.name == dst.name:
            return 0.0

        spec = self.cluster.get_link_spec(src.name, dst.name)
        bw_Bps = float(getattr(spec, "bw_GBs", 0.0) or 0.0) * (1024**3)
        if bw_Bps <= 0.0:
            return float("inf")

        flit = int(getattr(spec, "flit_size_B", 0) or 0)
        maxp = int(getattr(spec, "max_payload_B", 0) or 0)

        n_hat = int(bytes_amount)
        if flit > 0 and maxp > 0:
            packets = (int(bytes_amount) + int(maxp) - 1) // int(maxp)
            n_hat = int(bytes_amount) + int(packets) * int(flit)

        L = float(getattr(spec, "latency_s", 0.0) or 0.0)
        O = float(getattr(spec, "overhead_s", 0.0) or 0.0)
        return float(L + O + float(n_hat) / bw_Bps)

    def format_conversion_time(self, size_src_bytes: int, src_fmt: str, dst_fmt: str, dev: DeviceSpec) -> float:
        if src_fmt == dst_fmt:
            return 0.0
        if size_src_bytes <= 0:
            return 0.0
        bw_gbs = float(FORMAT_CONV_BW_GBs.get(dev.type, FORMAT_CONV_BW_GBs.get('default', 50.0)))
        bw = bw_gbs * 1e9
        if bw <= 0:
            return float('inf')
        t0_us = float(FORMAT_CONV_OVERHEAD_US.get(dev.type, FORMAT_CONV_OVERHEAD_US.get('default', 0.0)))
        t0 = t0_us * 1e-6
        return float(t0 + float(size_src_bytes) / bw)     

    def combine_transfer_and_convert(
        self,
        src: DeviceSpec,
        dst: DeviceSpec,
        bytes_amount: int,
        src_fmt: str,
        dst_fmt: str,
        *,
        nonoverlap: Optional[float] = None,
    ) -> float:
        """Return the combined latency of:
        1) direct transfer ``src -> dst`` for ``bytes_amount`` bytes
        2) destination-side format conversion ``src_fmt -> dst_fmt`` on ``dst``
        The overlap model is controlled by ``NONOVERLAP_TIME`` in config:
            total = transfer + nonoverlap * convert
        """
        bytes_amount = int(bytes_amount or 0)
        if bytes_amount <= 0:
            return 0.0
        t_convert = float(self.format_conversion_time(int(bytes_amount), str(src_fmt), str(dst_fmt), dst))
        t_transfer = float(self.comm_cost(src, dst, int(bytes_amount)))
        if src.name == dst.name:
            return float(t_convert)

        k = float(NONOVERLAP_TIME if nonoverlap is None else nonoverlap)
        return float(t_transfer + k * t_convert)

    # ---------------------------------------------------------------------
    # Software-Optimization helpers
    # ---------------------------------------------------------------------

    def _node_opt(self, node) -> Dict[str, Any]:
        """Return the optional optimization dict attached to the node."""
        try:
            attrs = getattr(node, 'attrs', {}) or {}
            opt = attrs.get('opt', {})
            return opt if isinstance(opt, dict) else {}
        except Exception:
            return {}

    def _act_dtype_bytes(self, node, phase: str) -> float:
        """Activation element byte-width."""
        opt = self._node_opt(node)
        aq = opt.get('activation_quant')
        if isinstance(aq, dict):
            b = aq.get('act_dtype_bytes')
            return float(b)
        return float(DTYPE_BYTES.get(self.dtype, 2))

    def _kv_dtype_bytes(self, node, phase: str) -> float:
        """KV-cache element byte-width.
        """
        opt = self._node_opt(node)
        b = opt.get('kv_dtype_bytes')
        if b is not None:
            return float(b)

        return float(DTYPE_BYTES.get(self.dtype, 2))

    def _activation_density(self, node, phase: str) -> float:
        opt = self._node_opt(node)
        aspec = opt.get('activation_sparsity')
        if not isinstance(aspec, dict):
            return 1.0
        # density_by_phase takes precedence
        dph = aspec.get('density_by_phase')
        if isinstance(dph, dict):
            ph = str(phase or '').lower()
            if ph in dph:
                return float(max(0.0, min(1.0, float(dph[ph]))))
        try:
            d = aspec.get('density', 1.0)
            return float(max(0.0, min(1.0, float(d))))
        except Exception:
            return 1.0

    def _activation_storage_compressed(self, node) -> bool:
        opt = self._node_opt(node)
        aspec = opt.get('activation_sparsity')
        if isinstance(aspec, dict):
            return str(aspec.get('storage', 'dense')).lower() == 'compressed'
        return False

    def _weight_density_for_compute(self, node) -> float:
        opt = self._node_opt(node)
        ws = opt.get('weight_sparsity')
        if not isinstance(ws, dict):
            return 1.0
        if not bool(ws.get('assume_sparse_compute', False)):
            return 1.0
        try:
            d = ws.get('density', 1.0)
            return float(max(0.0, min(1.0, float(d))))
        except Exception:
            return 1.0

    def _activation_density_for_compute(self, node, phase: str) -> float:
        opt = self._node_opt(node)
        aspec = opt.get('activation_sparsity')
        if not isinstance(aspec, dict):
            return 1.0
        if not bool(aspec.get('assume_sparse_compute', False)):
            return 1.0
        return self._activation_density(node, phase)

    def _attention_pairs(self, node, seq_len: int, phase: str, *, causal: bool) -> int:
        """Effective attention pairs for QK / Softmax / SV."""
        T = int(seq_len or 0)
        if T <= 0:
            return 0

        attrs = getattr(node, 'attrs', {}) or {}
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', T)) or T)
        q_len = T if str(phase) == 'prefill' else 1

        def tri(n: int) -> int:
            return n * (n + 1) // 2

        # Dense baseline
        if str(phase) == 'prefill':
            dense_pairs = tri(q_len) if causal else q_len * q_len
        else:
            dense_pairs = kv_len

        opt = self._node_opt(node)
        aspec = opt.get('attention_sparsity')
        if not isinstance(aspec, dict):
            return int(dense_pairs)

        pat = str(aspec.get('pattern', 'dense')).lower()
        if pat in ('dense', 'none', 'off', 'disabled'):
            return int(dense_pairs)

        # Local/sliding window (FlashAttention style window_size=(left,right))
        if pat in ('local', 'sliding', 'sliding_window', 'window'):
            wl = int(aspec.get('window_left', -1) or -1)
            wr = int(aspec.get('window_right', -1) or -1)
            if causal:
                wr = 0
            if wl < 0 and wr < 0:
                return int(dense_pairs)
            # number of keys per query in the steady state
            per_q = int(max(1, (wl if wl >= 0 else kv_len) + (wr if wr >= 0 else 0) + 1))
            if str(phase) == 'prefill':
                # causal local: sum_{i=1..T} min(i, per_q)
                if causal:
                    if T <= per_q:
                        return int(tri(T))
                    return int(tri(per_q) + (T - per_q) * per_q)
                # non-causal: each token attends to up to per_q (clipped by sequence boundaries)
                return int(T * min(T, per_q))
            # decode (single query at end)
            return int(min(kv_len, per_q))

        # Block-sparse: approximate as an effective local window over blocks
        if pat in ('block', 'block_sparse', 'blocksparse'):
            bs = int(aspec.get('block_size', 128) or 128)
            bl = int(aspec.get('blocks_left', 1) or 1)
            br = int(aspec.get('blocks_right', 0) or 0)
            if causal:
                br = 0
            per_q = int(max(1, (bl + br + 1) * bs))
            if str(phase) == 'prefill':
                if causal:
                    if T <= per_q:
                        return int(tri(T))
                    return int(tri(per_q) + (T - per_q) * per_q)
                return int(T * min(T, per_q))
            return int(min(kv_len, per_q))

        # Generic sparse attention matrix density
        if pat in ('matrix', 'sparse_matrix', 'sparse'):
            try:
                dens = float(aspec.get('density', 1.0))
                dens = max(0.0, min(1.0, dens))
            except Exception:
                dens = 1.0
            if str(phase) == 'prefill':
                return int(max(0, math.ceil(dense_pairs * dens)))
            return int(max(0, math.ceil(kv_len * dens)))

        return int(dense_pairs)

    def _effective_kv_len_for_decode(self, node, seq_len: int, phase: str) -> int:
        attrs = getattr(node, 'attrs', {}) or {}
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', seq_len)) or seq_len)
        if kv_len <= 0:
            return 0
        # For decode, pairs == kv_len for dense; for sparse patterns we approximate kv_len_eff via pairs.
        pairs = self._attention_pairs(node, int(seq_len or kv_len), str(phase), causal=bool(attrs.get('causal', True)))
        # pairs in decode is "keys per query".
        return int(max(0, min(kv_len, pairs)))

    def _time_scale_hint(self, node, dev_type: str) -> float:
        """Optional heuristic speedup scale from node.opt['speedup'].
        """
        opt = self._node_opt(node)
        sp = opt.get('speedup')
        if not isinstance(sp, dict):
            return 1.0
        try:
            v = float(sp.get(str(dev_type).lower(), 1.0))
            if v <= 0:
                return 1.0
            return 1.0 / v
        except Exception:
            return 1.0

# ---------------------------------------------------------------------
    def estimate_kv_cache_read_bytes(self, node, batch: int, seq_len: int, phase: str) -> int:
        """
        Estimate bytes of historical KV cache (per K or per V) that must be read during decode.

        This matches the previous explicit K_READ/V_READ operator volume. It is used when KV cache
        reads are modeled implicitly on the K->QK and V->SV edges (i.e., K_read/V_read nodes are removed).
        """
        attrs = getattr(node, 'attrs', {}) or {}
        dtype_bytes = float(self._kv_dtype_bytes(node, phase))
        if phase == 'prefill' or seq_len <= 0:
            return 0

        kv_len = int(self._effective_kv_len_for_decode(node, seq_len, phase))
        qh = int(attrs.get('q_heads', attrs.get('n_heads', 0)) or 0)
        kvh = int(attrs.get('n_kv_heads', attrs.get('kv_heads', qh)) or 0)
        hd = int(attrs.get('head_dim', 0) or 0)
        if kvh <= 0 or hd <= 0:
            return 0
        elems = batch * kvh * hd * kv_len
        return int(math.ceil(float(elems) * float(dtype_bytes)))
        
    def estimate_flops(self, node, batch: int, seq_len: int, phase: str) -> float:
        attrs = getattr(node, 'attrs', {}) or {}
        default = float(getattr(node, 'flops', 0.0) or 0.0)
        b = int(batch or attrs.get('batch', 0) or 0)
        if b <= 0:
            return default

        D = int(attrs.get('dim', 0) or 0)
        Hf = int(attrs.get('ffn_dim', attrs.get('hidden_dim', 0)) or 0)
        qh = int(attrs.get('q_heads', attrs.get('n_head', attrs.get('kv_heads', 0))) or 0)
        kvh = int(attrs.get('kv_heads', attrs.get('n_kv_heads', qh)) or 0)
        hd = int(attrs.get('head_dim', D // max(qh, 1)) or 0)

        q_dim = int(attrs.get('q_dim', qh * hd) or 0)
        kv_dim = int(attrs.get('kv_dim', kvh * hd) or 0)
        o_dim = int(attrs.get('o_dim', qh * hd) or 0)

        q_len = seq_len if phase == 'prefill' else 1
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', seq_len)) or seq_len)
        causal = bool(attrs.get('causal', True))

        def tri(n: int) -> int:
            return n * (n + 1) // 2

        C_MATMUL = 2.0
        C_LN = 5.0
        C_SOFTMAX = 5.0
        C_GELU = 6.0
        C_SILU = 5.0

        name = (getattr(node, 'name', '') or '').upper()

        moe_experts = int(attrs.get('experts', attrs.get('experts_per_layer', 0)) or 0)
        moe_active = int(attrs.get('active_experts',
                                attrs.get('active_experts_per_layer', moe_experts)) or moe_experts)
        moe_active = max(1, min(moe_experts if moe_experts > 0 else moe_active, moe_active))
        moe_top_k = int(attrs.get('top_k', attrs.get('experts_top_k', 0)) or 0)

        def moe_token_fraction() -> float:
            if 'expert' not in attrs or moe_experts <= 0 or moe_top_k <= 0:
                return 1.0
            imbalance = float(attrs.get('moe_imbalance',
                                        attrs.get('moe_imbalance_factor', 1.0)) or 1.0)
            active = max(1.0, float(moe_active))
            base = moe_top_k / active
            return min(1.0, base * imbalance)

        # Common sparsity multipliers (algorithmic FLOPs reduction)
        w_den = float(self._weight_density_for_compute(node))
        a_den = float(self._activation_density_for_compute(node, phase))

        # 1) LayerNorm
        if name == 'LN' and D > 0:
            return float(b * q_len * D * C_LN) * a_den

        # 2) Q / K / V 
        if name in ('Q', 'K', 'V') and D > 0:
            out_dim = q_dim if name == 'Q' else kv_dim
            if out_dim <= 0:
                return default
            return float(C_MATMUL * D * out_dim * b * q_len) * w_den * a_den

        # 3) QK^T
        if name == 'QK' and qh > 0 and (hd > 0):
            # Attention sparsity-aware pairs
            pairs = int(self._attention_pairs(node, seq_len, phase, causal=causal))
            return float(C_MATMUL * b * qh * hd * pairs)

        # 4) Softmax
        if name == 'SOFTMAX' and qh > 0:
            elems = int(self._attention_pairs(node, seq_len, phase, causal=causal))
            return float(b * qh * elems * C_SOFTMAX)

        # 5) S = softmax(QK^T)
        if name == 'SV' and qh > 0 and (hd > 0):
            pairs = int(self._attention_pairs(node, seq_len, phase, causal=causal))
            return float(C_MATMUL * b * qh * hd * pairs)

        # 6) O 
        if name == 'O' and D > 0 and (o_dim > 0):
            return float(C_MATMUL * o_dim * D * b * q_len) * w_den * a_den

        # 7) FFN W1 / W3 / UP / GATE
        if name in ('FFN_W1', 'FFN_W3', 'FFN_UP', 'FFN_GATE') and D > 0 and (Hf > 0):
            frac = moe_token_fraction()
            eff_b = b * frac
            return float(C_MATMUL * D * Hf * eff_b * q_len) * w_den * a_den

        # 8) FFN W2 / DOWN
        if name in ('FFN_W2', 'FFN_DOWN') and D > 0 and (Hf > 0):
            frac = moe_token_fraction()
            eff_b = b * frac
            return float(C_MATMUL * Hf * D * eff_b * q_len) * w_den * a_den

        # 9) SwiGLU / SiLU-GLU
        if name in ('SWIGLU', 'SILU_GLU') and Hf > 0:
            return float(b * q_len * Hf * (C_SILU + 1.0)) * a_den

        # 10) GELU
        if name == 'GELU' and Hf > 0:
            return float(b * q_len * Hf * C_GELU) * a_den

        # 11) Add / Residual / Dropout / Identity
        if name == 'ADD' and D > 0:
            return float(b * q_len * D) * a_den

        if name in ('IDENTITY', 'RESIDUAL', 'DROPOUT') and D > 0:
            return float(b * q_len * D) * a_den

        # 12) MoE Router
        if 'ROUTER' in name and D > 0 and moe_experts > 0:
            # 1) gating linear: [B*T, D] x [D, E]
            gate_linear = C_MATMUL * D * moe_experts * b * q_len

            # 2) softmax over experts
            gate_softmax = b * q_len * moe_experts * C_SOFTMAX

            # 3) top-k selection
            C_TOPK = 2.0
            gate_topk = b * q_len * moe_experts * C_TOPK

            # 4) combine K expert outputs: sum_{i=1..K} p_i * y_i
            combine = C_MATMUL * D * max(1, moe_top_k) * b * q_len

            return float(gate_linear + gate_softmax + gate_topk + combine)

        # 13)
        if name in ('K_WRITE', 'V_WRITE', 'KV_READ', 'KV_WRITE', 'ROPE', 'ALIBI', 'ALLREDUCE'):
            return 0.0

        return default

    def estimate_activation_bytes(self, node, batch: int, seq_len: int, phase: str):
        attrs = getattr(node, 'attrs', {}) or {}
        # Activation dtype may be overridden by quantization annotations.
        dtype_bytes = float(self._act_dtype_bytes(node, phase))
        # If activations are assumed *stored* in compressed sparse form, scale bytes.
        dens_store = float(self._activation_density(node, phase)) if self._activation_storage_compressed(node) else 1.0

        def to_bytes(elems: float) -> int:
            # elems may be fractional after density scaling; ceil to avoid undercount.
            return int(math.ceil(max(0.0, float(elems)) * float(dtype_bytes)))
        b = int(batch or attrs.get('batch', 0) or 1)
        T = int(seq_len or 0)
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', T)) or T)
        q_len = T if phase == 'prefill' else 1
        causal = bool(attrs.get('causal', True))

        def tri(n: int) -> int:
            return n * (n + 1) // 2
        D = int(attrs.get('dim', attrs.get('hidden_size', 0)) or 0)
        Hf = int(attrs.get('ffn_dim', attrs.get('mlp_dim', 0)) or 0)
        hd = int(attrs.get('head_dim', 0) or 0)
        qh = int(attrs.get('q_heads', attrs.get('n_heads', 0)) or 0)
        kvh = int(attrs.get('n_kv_heads', attrs.get('kv_heads', qh)) or 0)
        q_dim = int(attrs.get('q_dim', qh * hd) or 0)
        kv_dim = int(attrs.get('kv_dim', kvh * hd) or 0)
        o_dim = int(attrs.get('o_dim', qh * hd) or 0)
        name = (getattr(node, 'name', attrs.get('op', '')) or '').upper()
        # Attention sparsity-aware pairs.
        attn_pairs = int(self._attention_pairs(node, T, phase, causal=causal))
        moe_experts = int(attrs.get('experts', attrs.get('experts_per_layer', 0)) or 0)
        moe_active = int(attrs.get('active_experts', attrs.get('active_experts_per_layer', moe_experts)) or moe_experts)
        moe_active = max(1, min(moe_experts if moe_experts > 0 else moe_active, moe_active))
        moe_top_k = int(attrs.get('top_k', attrs.get('experts_top_k', 0)) or 0)
        def moe_token_fraction() -> float:
            if 'expert' not in attrs or moe_experts <= 0 or moe_top_k <= 0:
                return 1.0
            imbalance = float(attrs.get('moe_imbalance',
                                        attrs.get('moe_imbalance_factor', 1.0)) or 1.0)
            active = max(1.0, float(moe_active))
            base = moe_top_k / active
            return min(1.0, base * imbalance)
        if name == 'LN' and D > 0:
            elems = dens_store * (b * q_len * D)
            return (to_bytes(elems), to_bytes(elems))
        if name == 'Q' and D > 0:
            out_dim = q_dim if q_dim > 0 else D
            return (
                to_bytes(dens_store * (b * q_len * D)),
                to_bytes(dens_store * (b * q_len * out_dim)),
            )
        if name in ('K', 'V') and D > 0:
            out_dim = kv_dim if kv_dim > 0 else D
            write_tokens = q_len
            return (
                to_bytes(dens_store * (b * q_len * D)),
                to_bytes(dens_store * (b * write_tokens * out_dim)),
            )
        if name == 'O' and D > 0:
            inp_dim = o_dim if o_dim > 0 else D
            return (
                to_bytes(dens_store * (b * q_len * inp_dim)),
                to_bytes(dens_store * (b * q_len * D)),
            )
        if name in ('FFN_W1', 'FFN_W3') and D > 0 and (Hf > 0):
            frac = moe_token_fraction()
            eff_tokens = b * q_len * frac
            return (to_bytes(dens_store * (eff_tokens * D)), to_bytes(dens_store * (eff_tokens * Hf)))
        if name in ('FFN_W2',) and D > 0 and (Hf > 0):
            frac = moe_token_fraction()
            eff_tokens = b * q_len * frac
            return (to_bytes(dens_store * (eff_tokens * Hf)), to_bytes(dens_store * (eff_tokens * D)))
        if name in ('SWIGLU', 'SILU_GLU') and Hf > 0:
            return (
                to_bytes(dens_store * (b * q_len * (2 * Hf))),
                to_bytes(dens_store * (b * q_len * Hf)),
            )
        if name in ('GELU', 'RELU'):
            width = Hf if Hf > 0 else D
            return (to_bytes(dens_store * (b * q_len * width)), to_bytes(dens_store * (b * q_len * width)))
        if name == 'ADD' and D > 0:
            read_elems = dens_store * (b * q_len * D * 2)
            write_elems = dens_store * (b * q_len * D)
            return (to_bytes(read_elems), to_bytes(write_elems))
        if name in ('IDENTITY',):
            elems = dens_store * (b * q_len * D)
            return (to_bytes(elems), to_bytes(elems))
        if name in ('ALLREDUCE',) and D > 0:
            elems = dens_store * (b * q_len * D)
            return (to_bytes(elems), to_bytes(elems))
        if name == 'QK' and qh > 0 and (hd > 0):
            q_read = dens_store * (b * q_len * q_dim)
            write_elems = dens_store * (b * qh * attn_pairs)
            return (to_bytes(q_read), to_bytes(write_elems))
        if name in ('SOFTMAX', 'ATTN_SOFTMAX') and qh > 0:
            elems = dens_store * (b * qh * attn_pairs)
            return (to_bytes(elems), to_bytes(elems))
        if name == 'SV' and qh > 0 and (hd > 0):
            attn_read = dens_store * (b * qh * attn_pairs)
            out_elems = dens_store * (b * qh * q_len * hd)
            return (to_bytes(attn_read), to_bytes(out_elems))
        if name in ('K_WRITE', 'V_WRITE'):
            write_tokens = q_len
            elems = float(b * kvh * hd * write_tokens)
            kv_dtype_bytes = float(self._kv_dtype_bytes(node, phase))
            return (0, int(math.ceil(max(0.0, elems) * kv_dtype_bytes)))
        if 'ROUTER' in name and D > 0:
            tokens = float(b * q_len)
            read_elems = tokens * D
            if moe_top_k > 0:
                read_elems += tokens * moe_top_k * D
            write_elems = tokens * D
            return (to_bytes(read_elems), to_bytes(write_elems))
        if D > 0:
            elems = b * q_len * D
            return (to_bytes(elems), to_bytes(elems))
        return (0, 0)
    
    def node_device_cost(self, node: TaskNode, dev: DeviceSpec, label: PlanLabel, batch: int, seq_len: int, phase: str) -> float:
        attrs = getattr(node, 'attrs', {}) or {}
        time_scale = float(self._time_scale_hint(node, getattr(dev, 'type', '')))
        kv_in_pim = getattr(label, 'kv_in_pim', False)

        b = int(batch or attrs.get('batch', 1) or 1)
        D = int(attrs.get('dim', 0) or 0)
        Hf = int(attrs.get('ffn_dim', attrs.get('hidden_dim', 0)) or 0)
        qh = int(attrs.get('q_heads', attrs.get('n_head', attrs.get('kv_heads', 0))) or 0)
        kvh = int(attrs.get('kv_heads', attrs.get('n_kv_heads', qh)) or 0)
        hd = int(attrs.get('head_dim', D // max(qh, 1)) or 0)

        q_dim = int(attrs.get('q_dim', qh * hd) or 0)
        kv_dim = int(attrs.get('kv_dim', kvh * hd) or 0)
        o_dim = int(attrs.get('o_dim', qh * hd) or 0)

        q_len = seq_len if str(phase) == 'prefill' else 1
        causal = bool(attrs.get('causal', True))

        # NOTE: kv_len is an "effective keys per query" approximation (avg over sparsity / masking).
        pairs = int(self._attention_pairs(node, seq_len, phase, causal=causal))
        if str(phase) == 'prefill':
            kv_len = max(1, int(math.ceil(pairs / max(1, int(q_len)))))
        else:
            kv_len = max(1, int(pairs))

        aspec = self._node_opt(node).get('attention_sparsity')
        pat = str(aspec.get('pattern', 'dense')).lower() if isinstance(aspec, dict) else 'dense'

        # ------------------------------------------------------------------
        # CPU
        # ------------------------------------------------------------------
        if dev.type == 'cpu':
            rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
            mem_t = float(self.mem_time(int(rd + wr), dev))

            flops = float(self.estimate_flops(node, batch, seq_len, phase))
            compute_t = float(self.flop_time(flops, dev))
            return max(compute_t, mem_t) * time_scale

        # ------------------------------------------------------------------
        # NPU 
        # ------------------------------------------------------------------
        if dev.type == 'npu':
            self._ensure_backend_impls()

            # Common memory lower bound
            rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
            mem_t = float(self.mem_time(int(rd + wr), dev))

            raw_key = str(getattr(node, 'name', '') or getattr(node, 'id', '') or '')
            op_key = _normalize_npu_op_key(raw_key)
            ctx = NpuOpContext(
                op_key=op_key,
                attrs=attrs,
                batch=int(b),
                seq_len=int(seq_len),
                phase=str(phase),
                q_len=int(q_len),
                kv_len=int(kv_len),
                dim=int(D),
                ffn_dim=int(Hf),
                q_heads=int(qh),
                kv_heads=int(kvh),
                head_dim=int(hd),
                q_dim=int(q_dim),
                kv_dim=int(kv_dim),
                o_dim=int(o_dim),
                causal=bool(causal),
                attn_pattern=str(pat),
                mem_s=float(mem_t),
            )
            t = float(self._npu_backend_impl.estimate_s(self, node, dev, ctx))
            overhead = float(self.kernel_launch_overhead_s(op_key, dev, phase=str(phase)))
            if bool(self._kernel_launch_cfg(dev).get('scale_by_time_scale', False)):
                overhead *= float(time_scale)
            return float(t) * time_scale + float(overhead)

        # ------------------------------------------------------------------
        # PIM
        # ------------------------------------------------------------------
        if dev.type == 'pim':
            self._ensure_backend_impls()

            op_key = (str(getattr(node, 'name', '') or '')).strip().lower()
            ctx = PimOpContext(
                op_key=op_key,
                attrs=attrs,
                batch=int(b),
                seq_len=int(seq_len),
                phase=str(phase),
                dim=int(D),
                n_heads=int(qh),
                n_kv_heads=int(kvh),
                ffn_dim=int(Hf),
                kv_in_pim=bool(kv_in_pim),
            )
            t = float(self._pim_backend_impl.estimate_s(self, node, dev, label, ctx))
            return float(t) * time_scale

        return 0.0

    def weight_load_time_pim(self, weight_bytes: int) -> float:

        self._ensure_backend_impls()
        return float(self._pim_backend_impl.weight_load_s(self, int(weight_bytes or 0)))

    def activation_read_time_pim(self, activation_bytes_nd: int) -> float:
        self._ensure_backend_impls()
        return float(self._pim_backend_impl.activation_read_s(self, int(activation_bytes_nd or 0)))
