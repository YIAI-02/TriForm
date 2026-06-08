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

    def _attention_partition_fraction(self, node) -> float:
        """Fraction of the KV/context axis represented by an attention shard."""
        try:
            attrs = getattr(node, 'attrs', {}) or {}
            for key in ('attention_partition_fraction', 'kv_seq_fraction', 'kv_block_fraction'):
                if key in attrs and attrs.get(key) is not None:
                    v = float(attrs.get(key))
                    if math.isfinite(v) and v > 0.0:
                        return max(1e-12, min(1.0, v))
            shards = int(attrs.get('kv_seq_shards', attrs.get('attention_seq_shards', 1)) or 1)
            if shards > 1:
                return 1.0 / float(shards)
        except Exception:
            pass
        return 1.0

    def _scale_partitioned_count(self, value: int | float, frac: float) -> int:
        v = int(max(0, int(value or 0)))
        f = float(frac or 1.0)
        if v <= 0 or f >= 0.999999999:
            return int(v)
        return int(max(1, math.ceil(float(v) * max(0.0, min(1.0, f)))))

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
        part_frac = float(self._attention_partition_fraction(node))

        def _scale(v: int | float) -> int:
            return self._scale_partitioned_count(v, part_frac)

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
            aspec = attrs.get('attention_sparsity')
        if not isinstance(aspec, dict):
            pat0 = str(attrs.get('attention_pattern', 'dense')).lower()
            if pat0 in ('dense', 'none', 'off', 'disabled'):
                return _scale(dense_pairs)
            aspec = {'pattern': pat0}

        pat = str(aspec.get('pattern', attrs.get('attention_pattern', 'dense'))).lower()
        if pat in ('dense', 'none', 'off', 'disabled'):
            return _scale(dense_pairs)

        # Local/sliding window (FlashAttention style window_size=(left,right))
        if pat in ('local', 'sliding', 'sliding_window', 'window'):
            wl = int(aspec.get('window_left', -1) or -1)
            wr = int(aspec.get('window_right', -1) or -1)
            if causal:
                wr = 0
            if wl < 0 and wr < 0:
                return _scale(dense_pairs)
            # number of keys per query in the steady state
            per_q = int(max(1, (wl if wl >= 0 else kv_len) + (wr if wr >= 0 else 0) + 1))
            if str(phase) == 'prefill':
                # causal local: sum_{i=1..T} min(i, per_q)
                if causal:
                    if T <= per_q:
                        return _scale(tri(T))
                    return _scale(tri(per_q) + (T - per_q) * per_q)
                # non-causal: each token attends to up to per_q (clipped by sequence boundaries)
                return _scale(T * min(T, per_q))
            # decode (single query at end)
            return _scale(min(kv_len, per_q))

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
                        return _scale(tri(T))
                    return _scale(tri(per_q) + (T - per_q) * per_q)
                return _scale(T * min(T, per_q))
            return _scale(min(kv_len, per_q))

        # DeepSeek-V4 CSA / HCA: attention is performed over compressed KV
        # entries plus an explicit short sliding-window branch.  CSA adds a
        # sparse top-k selection over compressed blocks; HCA is dense over a
        # much more aggressively compressed hierarchy.
        if pat in ('deepseek_csa', 'csa', 'compressed_sparse_attention'):
            m = max(1, int(aspec.get('compression_rate', attrs.get('csa_compression_rate', 4)) or 4))
            top_k = max(1, int(aspec.get('top_k', attrs.get('csa_top_k', 512)) or 512))
            window = max(0, int(aspec.get('sliding_window', attrs.get('sliding_window', 0)) or 0))

            def sum_min_ceil(n: int, rate: int, cap: int) -> int:
                limit = min(int(n), int(rate) * int(cap))
                q = int(limit // rate)
                r = int(limit % rate)
                total = int(rate * q * (q + 1) // 2 + r * (q + 1))
                if n > limit:
                    total += int((n - limit) * cap)
                return int(total)

            def sum_window(n: int, w: int) -> int:
                if w <= 0:
                    return 0
                if n <= w:
                    return int(tri(n))
                return int(tri(w) + (n - w) * w)

            if str(phase) == 'prefill':
                return _scale(sum_min_ceil(T, m, top_k) + sum_window(T, window))
            return _scale(min(top_k, int(math.ceil(kv_len / float(m)))) + min(kv_len, window))

        if pat in ('deepseek_hca', 'hca', 'hierarchical_compressed_attention'):
            m = max(1, int(aspec.get('compression_rate', attrs.get('hca_compression_rate', 128)) or 128))
            window = max(0, int(aspec.get('sliding_window', attrs.get('sliding_window', 0)) or 0))

            def sum_ceil(n: int, rate: int) -> int:
                q = int(n // rate)
                r = int(n % rate)
                return int(rate * q * (q + 1) // 2 + r * (q + 1))

            def sum_window(n: int, w: int) -> int:
                if w <= 0:
                    return 0
                if n <= w:
                    return int(tri(n))
                return int(tri(w) + (n - w) * w)

            if str(phase) == 'prefill':
                return _scale(sum_ceil(T, m) + sum_window(T, window))
            return _scale(int(math.ceil(kv_len / float(m))) + min(kv_len, window))

        # Generic sparse attention matrix density
        if pat in ('matrix', 'sparse_matrix', 'sparse'):
            try:
                dens = float(aspec.get('density', 1.0))
                dens = max(0.0, min(1.0, dens))
            except Exception:
                dens = 1.0
            if str(phase) == 'prefill':
                return _scale(max(0, math.ceil(dense_pairs * dens)))
            return _scale(max(0, math.ceil(kv_len * dens)))

        return _scale(dense_pairs)

    def _effective_kv_len_for_decode(self, node, seq_len: int, phase: str) -> int:
        attrs = getattr(node, 'attrs', {}) or {}
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', seq_len)) or seq_len)
        if kv_len <= 0:
            return 0
        # For decode, pairs == kv_len for dense; for sparse patterns we approximate kv_len_eff via pairs.
        pairs = self._attention_pairs(node, int(seq_len or kv_len), str(phase), causal=bool(attrs.get('causal', True)))
        # pairs in decode is "keys per query".
        return int(max(0, min(kv_len, pairs)))

    def _attention_unique_kv_entries(self, node, seq_len: int, phase: str) -> int:
        """Approximate unique shared-KV vectors read by a DeepSeek-V4 attention op.

        `_attention_pairs` counts query-key dot products.  For memory traffic we
        need a separate estimate because DeepSeek-V4 uses one shared K/V vector
        per cache entry and reuses it across all Q heads.
        """
        T = int(seq_len or 0)
        if T <= 0:
            return 0
        attrs = getattr(node, 'attrs', {}) or {}
        kv_len = int(attrs.get('kv_len', attrs.get('past_kv_len', T)) or T)
        part_frac = float(self._attention_partition_fraction(node))

        def _scale(v: int | float) -> int:
            return self._scale_partitioned_count(v, part_frac)

        opt = self._node_opt(node)
        aspec = opt.get('attention_sparsity')
        if not isinstance(aspec, dict):
            aspec = attrs.get('attention_sparsity')
        pat = str(aspec.get('pattern', attrs.get('attention_pattern', 'dense'))).lower() if isinstance(aspec, dict) else str(attrs.get('attention_pattern', 'dense')).lower()
        window = max(0, int((aspec or {}).get('sliding_window', attrs.get('sliding_window', 0)) or 0)) if isinstance(aspec, dict) else max(0, int(attrs.get('sliding_window', 0) or 0))

        if pat in ('local', 'sliding', 'sliding_window', 'window'):
            return _scale(min(kv_len if str(phase) != 'prefill' else T, window if window > 0 else kv_len))

        if pat in ('deepseek_csa', 'csa', 'compressed_sparse_attention'):
            m = max(1, int((aspec or {}).get('compression_rate', attrs.get('csa_compression_rate', 4)) or 4)) if isinstance(aspec, dict) else max(1, int(attrs.get('csa_compression_rate', 4) or 4))
            top_k = max(1, int((aspec or {}).get('top_k', attrs.get('csa_top_k', 512)) or 512)) if isinstance(aspec, dict) else max(1, int(attrs.get('csa_top_k', 512) or 512))
            n = T if str(phase) == 'prefill' else kv_len
            compressed = int(math.ceil(float(n) / float(m)))
            if str(phase) != 'prefill':
                compressed = min(int(top_k), int(compressed))
            return _scale(compressed + min(int(n), int(window)))

        if pat in ('deepseek_hca', 'hca', 'hierarchical_compressed_attention'):
            m = max(1, int((aspec or {}).get('compression_rate', attrs.get('hca_compression_rate', 128)) or 128)) if isinstance(aspec, dict) else max(1, int(attrs.get('hca_compression_rate', 128) or 128))
            n = T if str(phase) == 'prefill' else kv_len
            return _scale(math.ceil(float(n) / float(m)) + min(int(n), int(window)))

        return _scale(T if str(phase) == 'prefill' else kv_len)

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
        if bool(attrs.get('kv_cache_shared', attrs.get('shared_kv', False))):
            cache_width = int(attrs.get('kv_cache_dim', attrs.get('shared_kv_dim', hd)) or hd)
        else:
            cache_width = int(attrs.get('kv_cache_dim', max(1, kvh * hd)) or max(1, kvh * hd))
        elems = batch * max(1, int(cache_width)) * kv_len
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

        # DeepSeek-V4 specific operators
        if name in ('DSV4_Q_DOWN', 'DSV4_Q_UP'):
            inp = int(attrs.get('input_dim', D) or D)
            out = int(attrs.get('output_dim', attrs.get('q_dim', 0)) or 0)
            if inp > 0 and out > 0:
                return float(C_MATMUL * inp * out * b * q_len) * w_den * a_den

        if name == 'DSV4_INDEXER_Q':
            inp = int(attrs.get('input_dim', D) or D)
            out = int(attrs.get('output_dim', attrs.get('q_dim', 0)) or 0)
            aux_in = int(attrs.get('aux_input_dim', attrs.get('model_dim', attrs.get('dim', D))) or D)
            aux_out = int(attrs.get('aux_output_dim', attrs.get('indexer_heads', 0)) or 0)
            main = float(C_MATMUL * inp * out * b * q_len) if inp > 0 and out > 0 else 0.0
            aux = float(C_MATMUL * aux_in * aux_out * b * q_len) if aux_in > 0 and aux_out > 0 else 0.0
            return (main + aux) * w_den * a_den

        if name in ('DSV4_KV_COMPRESS', 'DSV4_INDEX_KV_COMPRESS', 'DSV4_WINDOW_KV'):
            inp = int(attrs.get('input_dim', D) or D)
            out = int(attrs.get('output_dim', max(1, hd)) or max(1, hd))
            comp = max(1, int(attrs.get('compression_rate', 1) or 1))
            if name == 'DSV4_WINDOW_KV':
                return float(C_MATMUL * inp * out * b * q_len + b * q_len * out * C_LN) * w_den * a_den
            proj_dim = int(attrs.get('projected_dim', attrs.get('compressor_projected_dim', out)) or out)
            compressed_tokens = max(1, int(math.ceil(float(q_len) / float(comp))))
            # wkv and wgate are both dim -> projected_dim.  The softmax/gated
            # pooling cost is small but included so c4a overlap is visible.
            proj = float(2.0 * C_MATMUL * inp * proj_dim * b * q_len)
            mix = float(b * compressed_tokens * max(1, proj_dim) * comp * 3.0)
            return (proj + mix) * w_den * a_den

        if name == 'DSV4_INDEX_SCORE':
            idx_h = int(attrs.get('indexer_heads', 1) or 1)
            idx_d = int(attrs.get('indexer_head_dim', hd) or hd or 1)
            m = max(1, int(attrs.get('csa_compression_rate', attrs.get('compression_rate', 4)) or 4))
            if str(phase) == 'prefill':
                q = int(T := q_len) // m
                r = int(T) % m
                pairs_idx = int(m * q * (q + 1) // 2 + r * (q + 1))
            else:
                pairs_idx = int(math.ceil(float(kv_len) / float(m)))
            return float(C_MATMUL * b * idx_h * idx_d * pairs_idx)

        if name == 'DSV4_TOPK':
            m = max(1, int(attrs.get('csa_compression_rate', attrs.get('compression_rate', 4)) or 4))
            topk = max(1, int(attrs.get('top_k', attrs.get('csa_top_k', 512)) or 512))
            if str(phase) == 'prefill':
                candidates = int(math.ceil(float(seq_len) / float(m)))
                elems = b * q_len * candidates
            else:
                candidates = int(math.ceil(float(kv_len) / float(m)))
                elems = b * candidates
            return float(elems * max(1.0, math.log2(float(min(topk, max(2, candidates))))))

        if name == 'DSV4_O_G1':
            if bool(attrs.get('grouped_linear', False)):
                groups = max(1, int(attrs.get('groups', attrs.get('output_projection_groups', 1)) or 1))
                gin = int(attrs.get('group_input_dim', 0) or 0)
                gout = int(attrs.get('group_output_dim', 0) or 0)
                if gin > 0 and gout > 0:
                    return float(C_MATMUL * groups * gin * gout * b * q_len) * w_den * a_den
            inp = int(attrs.get('input_dim', attrs.get('o_dim', 0)) or 0)
            out = int(attrs.get('output_dim', attrs.get('dim', D)) or D)
            if inp > 0 and out > 0:
                return float(C_MATMUL * inp * out * b * q_len) * w_den * a_den

        if name == 'DSV4_O_G2':
            inp = int(attrs.get('input_dim', attrs.get('o_dim', 0)) or 0)
            out = int(attrs.get('output_dim', attrs.get('dim', D)) or D)
            if inp > 0 and out > 0:
                return float(C_MATMUL * inp * out * b * q_len) * w_den * a_den

        if name == 'MHC_MIX' and D > 0:
            nhc = max(1, int(attrs.get('mhc_expansion_factor', 1) or 1))
            sinkhorn_iters = max(0, int(attrs.get('sinkhorn_iters', 0) or 0))
            mix = float(b * q_len * D * nhc * nhc)
            normalize = float(b * q_len * nhc * nhc * sinkhorn_iters)
            return (mix + normalize) * a_den

        if name == 'MOE_COMBINE' and D > 0:
            inputs = max(1, int(attrs.get('combine_inputs', attrs.get('active_routed_experts', attrs.get('top_k', 1))) or 1))
            return float(C_MATMUL * b * q_len * D * inputs) * a_den

        if name == 'MOE_SHARED_COMBINE' and D > 0:
            shared = max(0, int(attrs.get('shared_experts', 0) or 0))
            active = max(0, int(attrs.get('active_routed_experts', attrs.get('active_experts', 0)) or 0))
            inputs = attrs.get('combine_inputs', None)
            if inputs is None:
                inputs = shared + min(1, active)
            return float(b * q_len * D * max(1, int(inputs or 1))) * a_den

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
            qk_dim = int(attrs.get('qk_dim', hd) or hd)
            return float(C_MATMUL * b * qh * qk_dim * pairs)

        # 4) Softmax
        if name == 'SOFTMAX' and qh > 0:
            elems = int(self._attention_pairs(node, seq_len, phase, causal=causal))
            return float(b * qh * elems * C_SOFTMAX)

        # 5) S = softmax(QK^T)
        if name == 'SV' and qh > 0 and (hd > 0):
            pairs = int(self._attention_pairs(node, seq_len, phase, causal=causal))
            value_dim = int(attrs.get('value_dim', hd) or hd)
            return float(C_MATMUL * b * qh * value_dim * pairs)

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

            return float(gate_linear + gate_softmax + gate_topk)

        # 13)
        if name in ('K_WRITE', 'V_WRITE', 'KV_READ', 'KV_WRITE', 'ROPE', 'ALIBI', 'ALLREDUCE'):
            return 0.0

        return default

    def estimate_activation_bytes(self, node, batch: int, seq_len: int, phase: str):
        attrs = getattr(node, 'attrs', {}) or {}
        kv_seq_frac = float(self._attention_partition_fraction(node))
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
        if name in ('DSV4_Q_DOWN', 'DSV4_Q_UP'):
            inp = int(attrs.get('input_dim', D) or D)
            out = int(attrs.get('output_dim', attrs.get('q_dim', 0)) or 0)
            return (
                to_bytes(dens_store * (b * q_len * inp)),
                to_bytes(dens_store * (b * q_len * out)),
            )
        if name == 'DSV4_INDEXER_Q':
            inp = int(attrs.get('input_dim', D) or D)
            out = int(attrs.get('output_dim', attrs.get('q_dim', 0)) or 0)
            aux_out = int(attrs.get('aux_output_dim', attrs.get('indexer_heads', 0)) or 0)
            return (
                to_bytes(dens_store * (b * q_len * inp)),
                to_bytes(dens_store * (b * q_len * (out + max(0, aux_out)))),
            )
        if name in ('DSV4_KV_COMPRESS', 'DSV4_INDEX_KV_COMPRESS', 'DSV4_WINDOW_KV'):
            inp = int(attrs.get('input_dim', D) or D)
            out = int(attrs.get('output_dim', max(1, hd)) or max(1, hd))
            comp = max(1, int(attrs.get('compression_rate', 1) or 1))
            # DeepSeek-V4 stores one shared K/V vector per cache entry.  Window
            # K/V is written at token granularity; compressor outputs are written
            # at block granularity.  The indexer compressor has its own 128-wide
            # cache used only by CSA top-k selection.
            if name == 'DSV4_WINDOW_KV':
                write_tokens = float(q_len)
            elif str(phase) == 'prefill':
                write_tokens = float(max(1, int(math.ceil(float(q_len) / float(comp)))))
            else:
                write_tokens = float(1.0 / float(comp))
            return (
                to_bytes(dens_store * (b * q_len * inp)),
                to_bytes(dens_store * (b * write_tokens * out)),
            )
        if name == 'DSV4_INDEX_SCORE':
            idx_h = int(attrs.get('indexer_heads', 1) or 1)
            idx_d = int(attrs.get('indexer_head_dim', hd) or hd or 1)
            m = max(1, int(attrs.get('csa_compression_rate', attrs.get('compression_rate', 4)) or 4))
            candidates = int(math.ceil(float(T if str(phase) == 'prefill' else kv_len) / float(m)))
            # The implementation first forms per-index-head scores, then applies
            # learned head weights and reduces to one scalar score per candidate.
            # Read traffic must include the shared indexer KV cache, but the node
            # output is the reduced scalar score matrix used by TOPK.
            q_tokens = q_len if str(phase) == 'prefill' else 1
            score_elems = b * q_tokens * candidates
            read_elems = b * q_tokens * idx_h * idx_d + b * candidates * idx_d
            return (to_bytes(dens_store * read_elems), to_bytes(dens_store * score_elems))
        if name == 'DSV4_TOPK':
            topk = max(1, int(attrs.get('top_k', attrs.get('csa_top_k', 512)) or 512))
            m = max(1, int(attrs.get('csa_compression_rate', attrs.get('compression_rate', 4)) or 4))
            candidates = int(math.ceil(float(T if str(phase) == 'prefill' else kv_len) / float(m)))
            q_tokens = q_len if str(phase) == 'prefill' else 1
            read_scores = b * q_tokens * candidates
            write_idx = b * q_tokens * min(topk, max(1, candidates))
            return (to_bytes(dens_store * read_scores), to_bytes(dens_store * write_idx))
        if name in ('DSV4_O_G1', 'DSV4_O_G2'):
            inp = int(attrs.get('input_dim', attrs.get('o_dim', 0)) or 0)
            out = int(attrs.get('output_dim', attrs.get('dim', D)) or D)
            return (
                to_bytes(dens_store * (b * q_len * inp)),
                to_bytes(dens_store * (b * q_len * out)),
            )
        if name == 'MHC_MIX' and D > 0:
            elems = dens_store * (b * q_len * D)
            return (to_bytes(elems), to_bytes(elems))
        if name == 'MOE_COMBINE' and D > 0:
            inputs = max(1, int(attrs.get('combine_inputs', attrs.get('active_routed_experts', attrs.get('top_k', 1))) or 1))
            read_elems = dens_store * (b * q_len * D * inputs)
            write_elems = dens_store * (b * q_len * D)
            return (to_bytes(read_elems), to_bytes(write_elems))
        if name == 'MOE_SHARED_COMBINE' and D > 0:
            shared = max(0, int(attrs.get('shared_experts', 0) or 0))
            inputs = attrs.get('combine_inputs', None)
            if inputs is None:
                inputs = max(2, shared + 1)
            read_elems = dens_store * (b * q_len * D * max(1, int(inputs or 1)))
            write_elems = dens_store * (b * q_len * D)
            return (to_bytes(read_elems), to_bytes(write_elems))
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
            qk_dim = int(attrs.get('qk_dim', hd) or hd)
            q_read = dens_store * (b * q_len * qh * qk_dim)
            unique_kv = self._attention_unique_kv_entries(node, T, phase)
            vectors = 1 if bool(attrs.get('kv_cache_shared', attrs.get('shared_kv', False))) else max(1, kvh)
            k_read = dens_store * (b * unique_kv * vectors * qk_dim)
            write_elems = dens_store * (b * qh * attn_pairs)
            return (to_bytes(q_read + k_read), to_bytes(write_elems))
        if name in ('SOFTMAX', 'ATTN_SOFTMAX') and qh > 0:
            elems = dens_store * (b * qh * attn_pairs)
            return (to_bytes(elems), to_bytes(elems))
        if name == 'SV' and qh > 0 and (hd > 0):
            value_dim = int(attrs.get('value_dim', hd) or hd)
            unique_kv = self._attention_unique_kv_entries(node, T, phase)
            vectors = 1 if bool(attrs.get('kv_cache_shared', attrs.get('shared_kv', False))) else max(1, kvh)
            attn_read = dens_store * (b * qh * attn_pairs)
            v_read = dens_store * (b * unique_kv * vectors * value_dim)
            out_elems = dens_store * (b * qh * q_len * value_dim)
            return (to_bytes(attn_read + v_read), to_bytes(out_elems))
        if name in ('K_WRITE', 'V_WRITE', 'KV_WRITE'):
            comp = max(1, int(attrs.get('compression_rate', 1) or 1))
            kv_dtype_bytes = float(self._kv_dtype_bytes(node, phase))
            if name == 'KV_WRITE' and str(attrs.get('model_family', '')).lower() == 'deepseek_v4':
                window = max(0, int(attrs.get('sliding_window', 0) or 0))
                shared = bool(attrs.get('kv_cache_shared', attrs.get('shared_kv', False)))
                if shared:
                    cache_width = int(attrs.get('kv_cache_dim', attrs.get('shared_kv_dim', hd)) or hd)
                else:
                    cache_width = int(attrs.get('kv_cache_dim', max(1, kvh * hd)) or max(1, kvh * hd))
                if comp <= 1:
                    entries = float(q_len)
                elif str(phase) == 'prefill':
                    # Prefill writes the full local-window stream plus the block-compressed stream.
                    entries = float(q_len + max(1, int(math.ceil(float(q_len) / float(comp)))))
                else:
                    # Decode writes one local token plus an amortized compressed block entry.
                    entries = float(1.0 + 1.0 / float(comp))
                elems = float(b * max(1, cache_width) * entries) * float(kv_seq_frac)
                return (0, int(math.ceil(max(0.0, elems) * kv_dtype_bytes)))
            write_tokens = q_len if name != 'KV_WRITE' else max(1, int(math.ceil(float(q_len) / float(comp))))
            if bool(attrs.get('kv_cache_shared', attrs.get('shared_kv', False))):
                cache_width = int(attrs.get('kv_cache_dim', attrs.get('shared_kv_dim', hd)) or hd)
            else:
                cache_width = int(attrs.get('kv_cache_dim', max(1, kvh * hd)) or max(1, kvh * hd))
            elems = float(b * cache_width * write_tokens) * float(kv_seq_frac)
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
        if not isinstance(aspec, dict):
            aspec = attrs.get('attention_sparsity')
        pat = str(aspec.get('pattern', attrs.get('attention_pattern', 'dense'))).lower() if isinstance(aspec, dict) else str(attrs.get('attention_pattern', 'dense')).lower()

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
