"""NPU backend adapters used by CostModel."""

from __future__ import annotations

from dataclasses import dataclass

from .shared import *

@dataclass(frozen=True, init=False)
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

    def __init__(
        self,
        *,
        op_key: str,
        attrs: Dict[str, Any],
        batch: int,
        seq_len: int,
        phase: str,
        q_len: int,
        kv_len: int,
        dim: int,
        ffn_dim: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        q_dim: int,
        kv_dim: int,
        o_dim: int,
        causal: bool,
        attn_pattern: str,
        mem_s: float,
    ) -> None:
        object.__setattr__(self, 'op_key', str(op_key))
        object.__setattr__(self, 'attrs', attrs)
        object.__setattr__(self, 'batch', int(batch))
        object.__setattr__(self, 'seq_len', int(seq_len))
        object.__setattr__(self, 'phase', str(phase))
        object.__setattr__(self, 'q_len', int(q_len))
        object.__setattr__(self, 'kv_len', int(kv_len))
        object.__setattr__(self, 'dim', int(dim))
        object.__setattr__(self, 'ffn_dim', int(ffn_dim))
        object.__setattr__(self, 'q_heads', int(q_heads))
        object.__setattr__(self, 'kv_heads', int(kv_heads))
        object.__setattr__(self, 'head_dim', int(head_dim))
        object.__setattr__(self, 'q_dim', int(q_dim))
        object.__setattr__(self, 'kv_dim', int(kv_dim))
        object.__setattr__(self, 'o_dim', int(o_dim))
        object.__setattr__(self, 'causal', bool(causal))
        object.__setattr__(self, 'attn_pattern', str(attn_pattern))
        object.__setattr__(self, 'mem_s', float(mem_s))


def make_npu_op_context(**kwargs: Any):
    """Construct NpuOpContext with a runtime fallback for stale imports."""
    return _instantiate_context(NpuOpContext, **kwargs)


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

        # (e) MoE router: use the analytic fallback because it mixes gate GEMM, softmax, top-k,
        # and weighted combine in one graph node.
        if op in NPU_ROUTER_KEYS:
            logger.debug(str(f'[NPU-ROUTER][LLMCompass] fallback-fast op={op} node={getattr(node, "name", "?")}'))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (f) Unknown op -> HARD ERROR
        raise RuntimeError(
            f"[LLMCompass] Unrecognized NPU op_key='{op}'. "
            f"Supported categories: softmax, norm-like, activation-like, router-like, "
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

        # (f) MoE router: use analytic fallback (combined gate GEMM + softmax + top-k + combine).
        if op in NPU_ROUTER_KEYS:
            logger.debug(str(f'[NPU-ROUTER][ASCEND-LUT] fallback-fast op={op} node={getattr(node, "name", "?")}'))
            return float(self._fallback_fast_s(cm, node, dev, ctx))

        # (g) Unknown -> fallback
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

__all__ = [name for name in globals() if not name.startswith("__")]
