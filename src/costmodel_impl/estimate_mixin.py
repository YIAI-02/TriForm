"""Operator-level FLOP and activation estimators for CostModel."""

from __future__ import annotations

from .shared import *
from .npu_backends import *
from .pim_backends import *

class CostModelEstimateMixin:
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
        return float(dtype_bytes(self.dtype, default='fp16'))

    def _kv_dtype_bytes(self, node, phase: str) -> float:
        """KV-cache element byte-width.
        """
        opt = self._node_opt(node)
        b = opt.get('kv_dtype_bytes')
        if b is not None:
            return float(b)

        return float(dtype_bytes(self.dtype, default='fp16'))

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
            frac_explicit = attrs.get('moe_token_fraction', attrs.get('expert_token_fraction', None))
            if frac_explicit is not None:
                try:
                    return float(min(1.0, max(0.0, float(frac_explicit))))
                except Exception:
                    pass
            if 'expert' not in attrs or moe_experts <= 0 or moe_top_k <= 0:
                return 1.0
            imbalance = float(attrs.get('moe_imbalance',
                                        attrs.get('moe_imbalance_factor', 1.0)) or 1.0)
            active = max(1.0, float(moe_active))
            base = moe_top_k / active
            return min(1.0, base * max(1.0, imbalance))

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
            router_experts = int(attrs.get('router_experts', attrs.get('num_local_experts', moe_experts)) or moe_experts)
            router_experts = max(1, router_experts)
            router_local_top_k = attrs.get('local_top_k', attrs.get('router_local_top_k', attrs.get('num_experts_per_tok', moe_top_k)))
            try:
                local_top_k = float(router_local_top_k if router_local_top_k is not None else moe_top_k)
            except Exception:
                local_top_k = float(moe_top_k or 1)
            local_top_k = max(0.0, local_top_k)

            # 1) gating linear: [B*T, D] x [D, E]
            gate_linear = C_MATMUL * D * router_experts * b * q_len

            # 2) softmax over experts
            gate_softmax = b * q_len * router_experts * C_SOFTMAX

            # 3) top-k selection
            C_TOPK = 2.0
            gate_topk = b * q_len * router_experts * C_TOPK

            # 4) combine local expert outputs: sum_i p_i * y_i
            combine = C_MATMUL * D * local_top_k * b * q_len

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
            frac_explicit = attrs.get('moe_token_fraction', attrs.get('expert_token_fraction', None))
            if frac_explicit is not None:
                try:
                    return float(min(1.0, max(0.0, float(frac_explicit))))
                except Exception:
                    pass
            if 'expert' not in attrs or moe_experts <= 0 or moe_top_k <= 0:
                return 1.0
            imbalance = float(attrs.get('moe_imbalance',
                                        attrs.get('moe_imbalance_factor', 1.0)) or 1.0)
            active = max(1.0, float(moe_active))
            base = moe_top_k / active
            return min(1.0, base * max(1.0, imbalance))
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
            router_local_top_k = attrs.get('local_top_k', attrs.get('router_local_top_k', attrs.get('num_experts_per_tok', moe_top_k)))
            try:
                local_top_k = float(router_local_top_k if router_local_top_k is not None else moe_top_k)
            except Exception:
                local_top_k = float(moe_top_k or 1)
            local_top_k = max(0.0, local_top_k)
            read_elems = tokens * D
            if local_top_k > 0.0:
                read_elems += tokens * local_top_k * D
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
            ctx = _instantiate_context(NpuOpContext,
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
            return float(t) + float(overhead)

        # ------------------------------------------------------------------
        # PIM
        # ------------------------------------------------------------------
        if dev.type == 'pim':
            self._ensure_backend_impls()

            op_key = (str(getattr(node, 'name', '') or '')).strip().lower()
            raw_key = str(getattr(node, 'name', '') or getattr(node, 'id', '') or '')
            op_key_ovh = _normalize_npu_op_key(raw_key)
            ctx = _instantiate_context(PimOpContext,
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
            overhead = float(self.kernel_launch_overhead_s(op_key_ovh, dev, phase=str(phase), time_scale=float(time_scale)))
            return float(t) * float(time_scale) + float(overhead)

        return 0.0


    def activation_read_time_pim(self, activation_bytes_nd: int) -> float:
        self._ensure_backend_impls()
        return float(self._pim_backend_impl.activation_read_s(self, int(activation_bytes_nd or 0)))
