from __future__ import annotations
from cProfile import label
from config import attach_local_debug_filter
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
from cost_model_npu_json_backend import (
    _map_op_to_mmad_dims,
    _predict_mmad_latency_us_from_json,
    _predict_softmax_latency_us_from_json,
    _predict_gelu_latency_us_from_json,
    _predict_layernorm_latency_us_from_json,
)

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)
DTYPE_BYTES: Dict[str, float] = {'fp32': 4, 'fp16': 2, 'bf16': 2, 'int8': 1, 'int4': 0.5}

# Canonical op-key sets for NPU backends
NPU_ACT_KEYS = {
    'gelu','relu','silu','swish','mish','tanh','sigmoid','relu6','leaky_relu','elu','hardtanh','selu','prelu',
    'geglu','swiglu','glu_act','activation'
}
NPU_NORM_KEYS = {
    'layernorm','layer_norm','ln','rmsnorm','rms_norm','norm',
    'groupnorm','group_norm','instancenorm','instance_norm','batchnorm','batch_norm'
}
NPU_GEMM_KEYS = {
    'q_proj','k_proj','v_proj','wo_proj','ffn_up','ffn_gate','ffn_down','score','output',
}

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
        device_key = _llmcompass_guess_device_key(dev)

        # (a) Softmax
        if ctx.op_key == 'softmax':
            M_rows = max(1, int(ctx.batch) * max(1, int(ctx.q_heads)) * max(1, int(ctx.q_len)))
            # Keep the original semantics:
            # - prefill+dense: K_cols = seq_len
            # - prefill+sparse: use kv_len (avg keys per query)
            # - decode: use kv_len
            if str(ctx.phase) == 'prefill':
                K_cols = max(1, int(ctx.seq_len if ctx.attn_pattern in ('dense', 'none', 'off', 'disabled') else ctx.kv_len))
            else:
                K_cols = max(1, int(ctx.kv_len))

            lat_s = _llmcompass_simulate_softmax_s(device_key, cm.dtype, int(M_rows), int(K_cols))
            logger.debug(str(
                f'[NPU-SOFTMAX][LLMCompass] device={device_key} M={M_rows} K={K_cols} '
                f'phase={ctx.phase} causal={ctx.causal} s={lat_s}'
            ))
            return float(lat_s + float(ctx.mem_s))

        # (b) Activation (use GELU as proxy)
        if ctx.op_key in NPU_ACT_KEYS:
            data_len = max(1, int(ctx.batch)) * max(1, int(ctx.q_len)) * max(1, int(ctx.ffn_dim if ctx.ffn_dim > 0 else ctx.dim))
            lat_s = _llmcompass_simulate_gelu_s(device_key, cm.dtype, int(data_len))
            logger.debug(str(f'[NPU-ACT][LLMCompass] device={device_key} data_len={data_len} s={lat_s}'))
            return float(lat_s + float(ctx.mem_s))

        # (c) Norm
        if _is_norm_like(ctx.op_key):
            rows = max(1, int(ctx.batch)) * max(1, int(ctx.q_len))
            try:
                lat_s = _llmcompass_simulate_layernorm_s(device_key, cm.dtype, int(rows), int(ctx.dim))
            except Exception as e:
                logger.debug(str(f'[NPU-NORM][LLMCompass] layernorm simulation failed for op={ctx.op_key}: {e}'))
                return float(self._fallback_fast_s(cm, node, dev, ctx))
            logger.debug(str(f'[NPU-NORM][LLMCompass] device={device_key} rows={rows} dim={ctx.dim} s={lat_s}'))
            return float(lat_s + float(ctx.mem_s))

        # (d) Matmul-like (GEMM / BatchedMatmul)
        if ctx.op_key in NPU_GEMM_KEYS:
            # ---- Attention score / output: use BatchedMatmul ----
            if ctx.op_key in ('score', 'output'):
                bmm_batch = max(1, int(ctx.batch) * max(1, int(ctx.q_heads)))
                M_mm = max(1, int(ctx.q_len))
                if ctx.op_key == 'score':
                    # [B*H, Tq, Dh] x [B*H, Dh, Tk] => [B*H, Tq, Tk]
                    N_mm = max(1, int(ctx.kv_len))
                    K_mm = max(1, int(ctx.head_dim))
                else:
                    # [B*H, Tq, Tk] x [B*H, Tk, Dh] => [B*H, Tq, Dh]
                    N_mm = max(1, int(ctx.head_dim))
                    K_mm = max(1, int(ctx.kv_len))

                lat_s = _llmcompass_simulate_matmul_s(device_key, cm.dtype, int(M_mm), int(N_mm), int(K_mm), batch=int(bmm_batch), batched=True,)
                logger.debug(str(
                    f'[NPU-MMAD][LLMCompass][BMM] device={device_key} '
                    f'batch={bmm_batch} M={M_mm} N={N_mm} K={K_mm} '
                    f'phase={ctx.phase} causal={ctx.causal} s={lat_s}'
                ))
                return float(lat_s + float(ctx.mem_s))

            # ---- Projections / FFN: use GEMM (fold batch*tokens into M) ----
            M_mm = max(1, int(ctx.batch) * max(1, int(ctx.q_len)))

            if ctx.op_key == 'q_proj':
                K_mm = max(1, int(ctx.head_dim))
                N_mm = max(1, int(ctx.q_dim))
            elif ctx.op_key in ('k_proj', 'v_proj'):
                K_mm = max(1, int(ctx.head_dim))
                N_mm = max(1, int(ctx.kv_dim))
            elif ctx.op_key == 'wo_proj':
                K_mm = max(1, int(ctx.o_dim))
                N_mm = max(1, int(ctx.head_dim))
            elif ctx.op_key in ('ffn_up', 'ffn_gate'):
                K_mm = max(1, int(ctx.head_dim))
                N_mm = max(1, int(ctx.ffn_dim))
            elif ctx.op_key == 'ffn_down':
                K_mm = max(1, int(ctx.ffn_dim))
                N_mm = max(1, int(ctx.head_dim))
            else:
                return float(self._fallback_fast_s(cm, node, dev, ctx))

            lat_s = _llmcompass_simulate_matmul_s(device_key, cm.dtype, int(M_mm), int(N_mm), int(K_mm))
            logger.debug(str(
                f'[NPU-MMAD][LLMCompass][GEMM] device={device_key} '
                f'M={M_mm} N={N_mm} K={K_mm} phase={ctx.phase} s={lat_s}'
            ))
            return float(lat_s + float(ctx.mem_s))

        # Fallback
        return float(self._fallback_fast_s(cm, node, dev, ctx))


class NpuAscend310BJsonBackend(NpuBackendBase):
    name = 'ascend_310b_json'

    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, ctx: NpuOpContext) -> float:
        # (a) Softmax
        if ctx.op_key == 'softmax':
            M_rows = max(1, int(ctx.batch) * max(1, int(ctx.q_heads)) * max(1, int(ctx.q_len)))
            if str(ctx.phase) == 'prefill':
                K_cols = max(1, int(ctx.seq_len if ctx.attn_pattern in ('dense', 'none', 'off', 'disabled') else ctx.kv_len))
            else:
                K_cols = max(1, int(ctx.kv_len))

            us = _predict_softmax_latency_us_from_json(int(M_rows), int(K_cols), phase=ctx.phase, causal=ctx.causal)
            logger.debug(str(f'[NPU-SOFTMAX][JSON] M={M_rows} K={K_cols} phase={ctx.phase} causal={ctx.causal} us={us}'))
            if us is not None:
                return float(max(float(us) * 1e-06, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (b) Activation (GELU proxy)
        if ctx.op_key in NPU_ACT_KEYS:
            data_len = max(1, int(ctx.batch)) * max(1, int(ctx.q_len)) * max(1, int(ctx.ffn_dim))
            us = _predict_gelu_latency_us_from_json(int(data_len))
            logger.debug(str(f'[NPU-ACT][JSON] data_len={data_len} us={us}'))
            if us is not None:
                return float(max(float(us) * 1e-06, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (c) Norm
        if _is_norm_like(ctx.op_key):
            rows = max(1, int(ctx.batch)) * (int(ctx.seq_len) if str(ctx.phase) == 'prefill' else 1)
            us = _predict_layernorm_latency_us_from_json(int(rows), int(ctx.dim))
            logger.debug(str(f'[NPU-NORM][JSON] op={ctx.op_key} rows={rows} width={ctx.dim} us={us}'))
            if us is not None:
                return float(max(float(us) * 1e-06, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (d) MMAD
        dims = _map_op_to_mmad_dims(
            ctx.op_key,
            int(ctx.dim),
            int(ctx.q_heads),
            int(ctx.kv_heads),
            int(ctx.ffn_dim if ctx.ffn_dim > 0 else ctx.dim),
            int(ctx.seq_len),
        ) if ctx.op_key else None

        if dims is not None:
            M, N, K, reps = dims
            try:
                qh_eff = max(1, int(ctx.q_heads))
                hd_eff = int(ctx.head_dim if ctx.head_dim > 0 else (ctx.dim // qh_eff))
                kv_len_eff = int(ctx.seq_len) if str(ctx.phase) == 'prefill' else int(ctx.kv_len)

                if ctx.op_key == 'score':
                    M, N, K = 1, max(1, kv_len_eff), max(1, hd_eff)
                    reps = max(1, qh_eff * max(1, int(ctx.q_len)))
                elif ctx.op_key == 'output':
                    M, N, K = 1, max(1, hd_eff), max(1, kv_len_eff)
                    reps = max(1, qh_eff * max(1, int(ctx.q_len)))
                elif ctx.op_key in ('q_proj','k_proj','v_proj','wo_proj','ffn_up','ffn_gate','ffn_down'):
                    reps = max(1, int(ctx.q_len))
            except Exception as _e:
                logger.debug(str(f'[NPU-MMAD][JSON] phase-aware adjust skipped: {_e}'))

            us = _predict_mmad_latency_us_from_json(int(M), int(N), int(K))
            logger.debug(str(f'[NPU-MMAD][JSON] M={M} N={N} K={K} reps={reps} us={us}'))
            if us is not None:
                total_s = float(us) * 1e-6 * max(1, int(reps)) * max(1, int(ctx.batch))
                return float(max(total_s, float(ctx.mem_s)))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # Fallback
        return float(self._fallback_fast_s(cm, node, dev, ctx))


def build_npu_backend(backend: Optional[str]) -> NpuBackendBase:
    b = _normalize_npu_backend(backend)
    if b == 'fast':
        return NpuFastBackend()
    if b == 'llmcompass':
        return NpuLlmCompassBackend()
    if b == 'ascend_310b_json':
        return NpuAscend310BJsonBackend()
    raise ValueError(f"Unsupported npu_backend='{backend}'. Expected one of: fast, llmcompass, ascend_310b_json")

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
            return float(cm.pim_mem_time(0, int(weight_bytes), pim_devs[0]))
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

        compute_time = 0.0
        if op_in and int(ctx.dim) > 0 and (int(ctx.n_heads) > 0):
            if traceable:
                try:
                    model_dict = cm.get_model_dict()
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
                            model_dict=model_dict,
                            use_cache=bool(cm.pim_cache_enabled),
                        )
                    )
                except Exception as e:
                    # Do not fail scheduling when trace simulation cannot run.
                    # Fall back to mem-only cost.
                    logger.debug(str(
                        f"[PIM] Trace backend failed for {getattr(node,'name','?')}: "
                        f"op='{op_in}' normalized='{op_norm}' err={e}. "
                        f"Falling back to mem-only cost."
                    ))
                    compute_time = 0.0
            else:
                logger.debug(str(
                    f"[PIM] Trace backend: skip unsupported op for {getattr(node,'name','?')}: "
                    f"op='{op_in}' normalized='{op_norm}'. Using mem-only cost."
                ))
        else:
            logger.debug(str(
                f'[PIM] Warning: Insufficient parameters for {getattr(node,"name","?")} '
                f'(op={ctx.op_key}, dim={ctx.dim}, heads={ctx.n_heads})'
            ))

        ATTENTION_DATAFLOW = {'QK', 'SV', 'SOFTMAX', 'K_READ', 'V_READ', 'K_WRITE', 'V_WRITE', 'KV_READ', 'KV_WRITE'}
        node_name_upper = ((getattr(node, 'name', '') or '').upper())
        # When KV is assumed to live in PIM, attention dataflow ops are already modeled inside trace.
        # If the op is not traceable, fall back to the bandwidth model to avoid silently charging 0.
        if bool(ctx.kv_in_pim) and (node_name_upper in ATTENTION_DATAFLOW) and traceable:
            mem_time = 0.0
        else:
            rd, wr = cm.estimate_activation_bytes(node, ctx.batch, ctx.seq_len, ctx.phase)
            mem_time = float(cm.pim_mem_time(int(rd), int(wr), dev))

        return float(compute_time + mem_time)
    def weight_load_s(self, cm: "CostModel", weight_bytes: int) -> float:
        """Trace-based PIM weight loading latency (read+write)."""
        # Keep the original behavior: fast-mode bypass
        if bool(cm.pim_fast_mode):
            return PimFastBackend().weight_load_s(cm, weight_bytes)

        if not cm.pim_config_path or not cm.gb_config_path or (not cm.ramulator_config_path):
            raise ValueError('PIM config, GB config, and Ramulator config must be set for weight loading simulation')

        dtype_bytes = DTYPE_BYTES.get(cm.dtype, 2)
        try:
            model_dict = cm.get_model_dict()
            read_lat, write_lat = _simulate_weight_loading_latency(
                int(weight_bytes),
                cm.pim_config_path,
                cm.gb_config_path,
                cm.ramulator_config_path,
                dtype_bytes,
                use_cache=bool(cm.pim_cache_enabled),
                keep_traces=bool(cm.debug_traces),
                model_dict=model_dict,
            )
            return float(read_lat + write_lat)
        except Exception as e:
            logger.debug(str(f'[Weight Load] Falling back to bandwidth estimation due to: {e}'))
            pim_devs = cm.cluster.devices_by_type('pim')
            if pim_devs:
                return float(cm.pim_mem_time(0, int(weight_bytes), pim_devs[0]))
            return 0.0

    def activation_read_s(self, cm: "CostModel", activation_bytes_nd: int) -> float:
        # Keep the original behavior: fast-mode bypass
        if bool(cm.pim_fast_mode):
            return PimFastBackend().activation_read_s(cm, activation_bytes_nd)

        try:
            read_lat, _ = _simulate_weight_loading_latency(
                int(activation_bytes_nd),
                cm.gb_config_path,         # pim_config_path (WRITE side not used)
                cm.pim_config_path,        # gb_config_path (READ side uses PIM cfg)
                cm.ramulator_config_path,
                dtype_bytes=DTYPE_BYTES.get(cm.dtype, 2),
                use_cache=bool(cm.pim_cache_enabled),
                keep_traces=bool(cm.debug_traces),
                model_dict=cm.get_model_dict(),
            )
            return float(read_lat)
        except Exception as e:
            logger.debug(f"[activation_read_time_pim] fallback to mem_time due to: {e}")
            pim_devs = getattr(cm.cluster, 'devices_by_type', lambda *_: [])('pim')
            if pim_devs:
                return float(cm.pim_mem_time(int(activation_bytes_nd), 0, pim_devs[0]))
            return 0.0


def build_pim_backend(pim_fast_mode: bool) -> PimBackendBase:
    return PimFastBackend() if bool(pim_fast_mode) else PimTraceBackend()


class CostModel:
    def __init__(self, cluster: Cluster, dtype: str='fp16', pim_config_path: Optional[Path]=None, gb_config_path: Optional[Path]=None, ramulator_config_path: Optional[Path]=None, simulation_log_file: Optional[Path]=None, debug_traces: bool=False, model_dict: Optional[Dict]=None, pim_fast_mode: bool=False,npu_backend: Optional[str]=None):
        self.cluster = cluster
        self.dtype = dtype
        self.pim_config_path = pim_config_path
        self.gb_config_path = gb_config_path
        self.ramulator_config_path = ramulator_config_path
        self.debug_traces = debug_traces
        self.pim_fast_mode = pim_fast_mode  # When True, skip all trace simulations
        self.npu_backend = _normalize_npu_backend(npu_backend) if npu_backend is not None else (_normalize_npu_backend('fast'))
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
        try:
            npu_name = _normalize_npu_backend(self.npu_backend)
        except Exception:
            npu_name = 'fast'
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

    def flop_time(self, flops: float, dev: DeviceSpec) -> float:
        if dev.tflops <= 0:
            return 0.0
        return flops / (dev.tflops * 1024 * 1024 * 1024.0)

    def mem_time(self, bytes_amount: int, dev: DeviceSpec) -> float:
        bw = dev.mem_bw_GBs  * 1024 * 1024 * 1024.0
        return 0.0 if bw <= 0 else bytes_amount / bw

    def pim_mem_time(self, read_bytes: int, write_bytes: int, dev: DeviceSpec) -> float:
        """
        #TODO only have the fast mode
        PIM memory time estimation for fast-mode (no trace).

        Add a minimal read/write access latency constraint on top of the
        bandwidth model:
          - read:  dev.pim_read_latency_ns  (default: 0)
          - write: dev.pim_write_latency_ns (default: 0)

        For each direction: t = max(bytes / bw, latency).
        """
        bw = dev.mem_bw_GBs * 1024 * 1024 * 1024.0

        t_rd = 0.0 if read_bytes <= 0 or bw <= 0 else float(read_bytes) / bw
        t_wr = 0.0 if write_bytes <= 0 or bw <= 0 else float(write_bytes) / bw

        rd_lat_ns = getattr(dev, 'pim_read_latency_ns', getattr(dev, 'read_latency_ns', 0.0))
        wr_lat_ns = getattr(dev, 'pim_write_latency_ns', getattr(dev, 'write_latency_ns', 0.0))

        return (t_rd + float(rd_lat_ns) * 1e-9) + (t_wr + float(wr_lat_ns) * 1e-9)

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
        if name in ('K_WRITE', 'V_WRITE', 'KV_READ', 'KV_WRITE', 'ROPE', 'ALIBI'):
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
            # New K/V for current tokens written into KV cache
            write_tokens = q_len
            elems = b * kvh * hd * write_tokens
            return (0, to_bytes(elems))
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
        is_shard = bool(attrs.get('is_shard', False))
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
            tflops = float(getattr(dev, 'tflops', 0.0) or 0.0)
            if tflops <= 0.0:
                # Fallback: treat as very slow (1 GFLOP/s equivalent) instead of free.
                tflops = 1e-3
            compute_t = float(flops) / (tflops * 1e12)
            return max(compute_t, mem_t) * time_scale

        # ------------------------------------------------------------------
        # NPU 
        # ------------------------------------------------------------------
        if dev.type == 'npu':
            self._ensure_backend_impls()

            # Common memory lower bound
            rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
            mem_t = float(self.mem_time(int(rd + wr), dev))

            op_key = (str(getattr(node, 'name', '') or '')).strip().lower()
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
            return float(t) * time_scale

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
