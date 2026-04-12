"""Kernel timing and memory-transfer helpers for CostModel."""

from __future__ import annotations

from .shared import *
from .npu_backends import *
from .pim_backends import *

class CostModelComputeMixin:
    def _make_npu_op_context(self, node: TaskNode, dev: DeviceSpec, batch: int, seq_len: int, phase: str, *, mem_s: float = 0.0) -> NpuOpContext:
        attrs = getattr(node, 'attrs', {}) or {}
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
        pairs = int(self._attention_pairs(node, seq_len, phase, causal=causal))
        if str(phase) == 'prefill':
            kv_len = max(1, int(math.ceil(pairs / max(1, int(q_len)))))
        else:
            kv_len = max(1, int(pairs))

        aspec = self._node_opt(node).get('attention_sparsity')
        pat = str(aspec.get('pattern', 'dense')).lower() if isinstance(aspec, dict) else 'dense'
        raw_key = str(getattr(node, 'name', '') or getattr(node, 'id', '') or '')
        op_key = _normalize_npu_op_key(raw_key)
        return _instantiate_context(NpuOpContext,
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
            mem_s=float(mem_s),
        )

    def _make_pim_op_context(self, node: TaskNode, label: PlanLabel, batch: int, seq_len: int, phase: str) -> PimOpContext:
        attrs = getattr(node, 'attrs', {}) or {}
        b = int(batch or attrs.get('batch', 1) or 1)
        D = int(attrs.get('dim', 0) or 0)
        Hf = int(attrs.get('ffn_dim', attrs.get('hidden_dim', 0)) or 0)
        qh = int(attrs.get('q_heads', attrs.get('n_head', attrs.get('kv_heads', 0))) or 0)
        kvh = int(attrs.get('kv_heads', attrs.get('n_kv_heads', qh)) or 0)
        kv_in_pim = getattr(label, 'kv_in_pim', False)
        op_key = (str(getattr(node, 'name', '') or '')).strip().lower()
        return _instantiate_context(PimOpContext,
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

    def _weighted_internal_load_bytes(self, node: TaskNode, batch: int, seq_len: int, phase: str, *, resident_weight_fmt: str) -> int:
        rd, wr = self.estimate_activation_bytes(node, batch, seq_len, phase)
        weight_bytes = int(self.weight_storage_bytes(int(getattr(node, 'weight_size', 0) or 0), str(resident_weight_fmt)))
        return int(max(0, weight_bytes) + max(0, int(rd or 0)) + max(0, int(wr or 0)))

    def npu_weight_b2_time(self, node: TaskNode, dev: DeviceSpec, batch: int, seq_len: int, phase: str) -> float:
        self._ensure_backend_impls()
        ctx = self._make_npu_op_context(node, dev, batch, seq_len, phase, mem_s=0.0)
        return float(self._npu_backend_impl.estimate_s(self, node, dev, ctx))

    def weighted_compute_stage(self, node: TaskNode, dev: DeviceSpec, label: PlanLabel, batch: int, seq_len: int, phase: str, *, resident_weight_fmt: str) -> WeightComputeStageBreakdown:
        dev_type = str(getattr(dev, 'type', '') or '').lower()
        weight_size_nd = int(getattr(node, 'weight_size', 0) or 0)
        time_scale = float(self._time_scale_hint(node, getattr(dev, 'type', '')))

        if weight_size_nd <= 0:
            total = float(self.node_device_cost(node, dev, label, batch, seq_len, phase))
            return WeightComputeStageBreakdown(
                total_s=float(total),
                compute_fmt=str(self.device_preferred_fmt(dev)),
                backend=f'{dev_type}_default',
                combine_rule='direct',
            )

        if dev_type == 'cpu':
            total = float(self.node_device_cost(node, dev, label, batch, seq_len, phase))
            return WeightComputeStageBreakdown(
                total_s=float(total),
                compute_fmt='ND',
                backend='cpu_default',
                combine_rule='direct',
            )

        if dev_type == 'npu':
            self._ensure_backend_impls()
            compute_fmt = str(self.npu_weight_compute_format(node))
            b1 = float(self.npu_weight_conversion_time(int(weight_size_nd), str(resident_weight_fmt), str(compute_fmt)))
            b2 = float(self.npu_weight_b2_time(node, dev, batch, seq_len, phase))
            backend_name = str(getattr(self, '_npu_backend_impl_name', None) or getattr(self, 'npu_backend', '') or 'fast').lower()
            if backend_name == 'fast':
                core = max(float(b1), float(b2))
                rule = 'max'
                backend_tag = 'npu_fast'
            elif backend_name in ('ascend_310b_lut', 'ascend_310b', 'ascend310b', 'ascend'):
                core = float(b1) + float(b2)
                rule = 'sum'
                backend_tag = 'npu_lut'
            elif backend_name == 'llmcompass':
                core = max(float(b1), float(b2))
                rule = 'max'
                backend_tag = 'npu_llmcompass'
            else:
                core = max(float(b1), float(b2))
                rule = 'max'
                backend_tag = f'npu_{backend_name}'
            op_key = _normalize_npu_op_key(str(getattr(node, 'name', '') or getattr(node, 'id', '') or ''))
            overhead = float(self.kernel_launch_overhead_s(op_key, dev, phase=str(phase)))
            return WeightComputeStageBreakdown(
                total_s=float(core + overhead),
                compute_fmt=str(compute_fmt),
                backend=str(backend_tag),
                combine_rule=str(rule),
                b1_s=float(b1),
                b2_s=float(b2),
                launch_overhead_s=float(overhead),
            )

        if dev_type == 'pim':
            self._ensure_backend_impls()
            compute_fmt = 'PIM-OPT'
            op_key_ovh = _normalize_npu_op_key(str(getattr(node, 'name', '') or getattr(node, 'id', '') or ''))
            if bool(getattr(self, 'pim_fast_mode', False)):
                internal_bytes = int(self._weighted_internal_load_bytes(node, batch, seq_len, phase, resident_weight_fmt=str(resident_weight_fmt or 'PIM-OPT')))
                b1 = float(self.mem_time(int(internal_bytes), dev))
                flops = float(self.estimate_flops(node, batch, seq_len, phase))
                b2 = float(self.flop_time(flops, dev))
                b1 *= float(time_scale)
                b2 *= float(time_scale)
                core = max(float(b1), float(b2))
                rule = 'max'
                backend_tag = 'pim_fast'
            else:
                ctx = self._make_pim_op_context(node, label, batch, seq_len, phase)
                core = float(self._pim_backend_impl.estimate_s(self, node, dev, label, ctx)) * float(time_scale)
                b1 = float(core)
                b2 = 0.0
                rule = 'trace'
                backend_tag = 'pim_trace'
            overhead = float(self.kernel_launch_overhead_s(op_key_ovh, dev, phase=str(phase), time_scale=float(time_scale)))
            return WeightComputeStageBreakdown(
                total_s=float(core + overhead),
                compute_fmt=str(compute_fmt),
                backend=str(backend_tag),
                combine_rule=str(rule),
                b1_s=float(b1),
                b2_s=float(b2),
                launch_overhead_s=float(overhead),
            )

        total = float(self.node_device_cost(node, dev, label, batch, seq_len, phase))
        return WeightComputeStageBreakdown(
            total_s=float(total),
            compute_fmt=str(self.device_preferred_fmt(dev)),
            backend=f'{dev_type}_default',
            combine_rule='direct',
        )

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

        if dev_type == 'pim' and bool(getattr(self, 'pim_fast_mode', False)):
            per = _lookup_cfg_by_device_name(cfg, dev)
            if isinstance(per, dict) and per:
                params = per
            else:
                try:
                    name_map = cfg.get('by_device_name', cfg.get('by_name', {})) or {}
                except Exception:
                    name_map = {}
                fallback = None
                for k in ('pim', 'PIM'):
                    if k in name_map:
                        fallback = name_map.get(k)
                        break
                    if str(k).lower() in name_map:
                        fallback = name_map.get(str(k).lower())
                        break
                    if str(k).upper() in name_map:
                        fallback = name_map.get(str(k).upper())
                        break
                if isinstance(fallback, dict) and fallback:
                    params = fallback
                else:
                    keys = list((cfg.get('by_device_name', cfg.get('by_name', {})) or {}).keys())
                    logger.debug(
                        "[PIM][FAST] COMPUTE_UTILIZATION missing by_device_name for dev_name='%s'. available_keys=%s. Fallback utilization=1.0",
                        str(getattr(dev, 'name', '') or ''),
                        keys,
                    )

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
        if dev is None:
            return dict(cfg)
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

        if dev_type == 'pim' and bool(getattr(self, 'pim_fast_mode', False)):
            per = _lookup_cfg_by_device_name(cfg, dev)
            if isinstance(per, dict) and per:
                return dict(per)
            try:
                name_map = cfg.get('by_device_name', cfg.get('by_name', {})) or {}
            except Exception:
                name_map = {}
            for k in ('pim', 'Aim PIM', 'AIM PIM', 'Aim_PIM', 'AIM_PIM', 'PIM'):
                if k in name_map and isinstance(name_map.get(k), dict):
                    return dict(name_map.get(k) or {})
                kl = str(k).lower()
                ku = str(k).upper()
                if kl in name_map and isinstance(name_map.get(kl), dict):
                    return dict(name_map.get(kl) or {})
                if ku in name_map and isinstance(name_map.get(ku), dict):
                    return dict(name_map.get(ku) or {})

            keys = list((cfg.get('by_device_name', cfg.get('by_name', {})) or {}).keys())
            logger.debug(
                "[PIM][FAST] KERNEL_LAUNCH_OVERHEAD missing by_device_name for dev_name='%s'. available_keys=%s. "
                "Fallback overhead=0",
                str(getattr(dev, 'name', '') or ''),
                keys,
            )
            return {}

        return {}

    def kernel_launch_overhead_s(
        self,
        op_key: str,
        dev: DeviceSpec,
        *,
        phase: str = 'prefill',
        time_scale: Optional[float] = None,
    ) -> float:
        cfg = self._kernel_launch_cfg(dev)
        if not cfg or not bool(cfg.get('enabled', False)):
            return 0.0

        # Backend gating to avoid double-counting with empirical backends.
        apply_backends = cfg.get('apply_backends', None)
        if apply_backends not in (None, True, 'all', 'ALL'):
            dev_type = str(getattr(dev, 'type', '') or '').lower()
            if dev_type == 'pim':
                try:
                    name = str(getattr(self._pim_backend_impl, 'name', '') or '')
                except Exception:
                    name = 'fast' if bool(getattr(self, 'pim_fast_mode', False)) else 'trace'
            else:
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

        if bool(cfg.get('scale_by_time_scale', False)) and time_scale is not None:
            try:
                ts = float(time_scale)
                if ts > 0:
                    us *= ts
            except Exception:
                pass

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

        path, bw_root, ovh_root = self._runtime_model_device_sections(str(getattr(dev, 'type', '') or '').lower())
        bw_paths = bw_root.get('paths') if isinstance(bw_root.get('paths'), dict) else {}
        ovh_paths = ovh_root.get('paths') if isinstance(ovh_root.get('paths'), dict) else {}
        path_key = f'{src_fmt}->{dst_fmt}'
        bw_gbs = bw_paths.get(path_key, bw_root.get('default', None))
        if bw_gbs is None:
            raise ValueError(
                f"Missing format-conversion bandwidth for device_type='{getattr(dev,'type','')}' path='{path_key}' in {path}"
            )
        bw = float(bw_gbs) * 1e9
        if bw <= 0:
            return float('inf')
        t0_us = float(ovh_paths.get(path_key, ovh_root.get('default', 0.0)) or 0.0)
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
