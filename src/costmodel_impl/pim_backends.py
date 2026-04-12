"""PIM backend adapters used by CostModel."""

from __future__ import annotations

from dataclasses import dataclass

from .shared import *

@dataclass(frozen=True, init=False)
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

    def __init__(
        self,
        *,
        op_key: str,
        attrs: Dict[str, Any],
        batch: int,
        seq_len: int,
        phase: str,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        kv_in_pim: bool,
    ) -> None:
        object.__setattr__(self, 'op_key', str(op_key))
        object.__setattr__(self, 'attrs', attrs)
        object.__setattr__(self, 'batch', int(batch))
        object.__setattr__(self, 'seq_len', int(seq_len))
        object.__setattr__(self, 'phase', str(phase))
        object.__setattr__(self, 'dim', int(dim))
        object.__setattr__(self, 'n_heads', int(n_heads))
        object.__setattr__(self, 'n_kv_heads', int(n_kv_heads))
        object.__setattr__(self, 'ffn_dim', int(ffn_dim))
        object.__setattr__(self, 'kv_in_pim', bool(kv_in_pim))


def make_pim_op_context(**kwargs: Any):
    """Construct PimOpContext with a runtime fallback for stale imports."""
    return _instantiate_context(PimOpContext, **kwargs)


class PimBackendBase(ABC):
    name: str = 'base'

    @abstractmethod
    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, label: PlanLabel, ctx: PimOpContext) -> float:
        """Return estimated end-to-end op latency on PIM (seconds)."""
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

    def activation_read_s(self, cm: "CostModel", activation_bytes_nd: int) -> float:
        logger.debug(str(f"[PIM][FAST] activation_read bytes={activation_bytes_nd}"))
        pim_devs = getattr(cm.cluster, 'devices_by_type', lambda *_: [])('pim')
        if pim_devs:
            return float(cm.pim_mem_time(int(activation_bytes_nd), 0, pim_devs[0]))
        return 0.0


class PimTraceBackend(PimBackendBase):
    name = 'trace'

    def estimate_s(self, cm: "CostModel", node: TaskNode, dev: DeviceSpec, label: PlanLabel, ctx: PimOpContext) -> float:
        op_in = str(ctx.op_key) if ctx.op_key is not None else ''
        op_norm = _normalize_pim_op(op_in) if op_in else ''
        traceable = bool(op_norm) and (op_norm in PIM_TRACE_SUPPORTED_OPS)

        # Strict mode can be enabled either via CostModel(..., pim_trace_strict=True)
        # or via environment variable PIM_TRACE_STRICT=1.
        strict = bool(getattr(cm, 'pim_trace_strict', False))
        env_strict = str(os.environ.get('PIM_TRACE_STRICT', '') or '').strip().lower()
        if env_strict in ('1', 'true', 'yes', 'on'):
            strict = True

        # Trace mode requires configs.
        if traceable and (not cm.pim_config_path or not cm.ramulator_config_path):
            msg = (
                "[PIM] Trace backend requires pim_config_path and ramulator_config_path, but got "
                f"pim_config_path={cm.pim_config_path!r}, ramulator_config_path={cm.ramulator_config_path!r}. "
                f"op='{op_in}' normalized='{op_norm}' node='{getattr(node, 'name', getattr(node, 'id', '?'))}'"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg + " Falling back to compute-only estimate.")
            traceable = False

        # Basic parameter guardrails for trace path only.
        if traceable and (int(ctx.dim) <= 0 or int(ctx.n_heads) <= 0):
            msg = (
                f"[PIM] Trace backend missing/invalid parameters for node='{getattr(node,'name','?')}' "
                f"(op='{op_in}' normalized='{op_norm}', dim={ctx.dim}, heads={ctx.n_heads})."
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg + " Falling back to compute-only estimate.")
            traceable = False

        # Prefer exact trace-based latency when supported.
        if traceable:
            try:
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

                seqlen_i = int(ctx.seq_len) if int(ctx.seq_len) > 0 else 1
                model_dict = cm.get_or_make_pim_model_dict(
                    dim=int(ctx.dim),
                    n_heads=int(ctx.n_heads),
                    n_kv_heads=int(ctx.n_kv_heads),
                    ffn_dim=int(ctx.ffn_dim),
                    seqlen=int(seqlen_i),
                )

                compute_time = float(
                    _get_pim_latency_via_trace(
                        op=str(op_norm),
                        pim_config=cm.pim_config_path,
                        ramulator_config=cm.ramulator_config_path,
                        dim=int(ctx.dim),
                        n_heads=int(ctx.n_heads),
                        n_kv_heads=int(ctx.n_kv_heads),
                        ffn_dim=int(ctx.ffn_dim),
                        seqlen=int(seqlen_i) if int(seqlen_i) > 0 else None,
                        batch=int(ctx.batch) if int(ctx.batch) > 0 else 1,
                        phase=str(ctx.phase),
                        model_dict=model_dict,
                        use_cache=bool(cm.pim_cache_enabled),
                        head_dim=int(hd) if int(hd) > 0 else None,
                        q_dim=int(q_dim) if int(q_dim) > 0 else None,
                        kv_dim=int(kv_dim) if int(kv_dim) > 0 else None,
                        o_dim=int(o_dim) if int(o_dim) > 0 else None,
                        ramulator_timeout_s=int(getattr(cm, 'pim_ramulator_timeout_s', 3000) or 3000),
                        keep_traces=bool(getattr(cm, 'pim_trace_keep_traces', False)),
                        trace_dir=getattr(cm, 'pim_trace_dir', None),
                        pim_freq_ghz=float(getattr(dev, 'freq_ghz', 0.0) or 0.0),
                    )
                )

                return float(compute_time)

            except Exception as e:
                msg = (
                    f"[PIM] Trace backend failed for node='{getattr(node,'name','?')}' "
                    f"op='{op_in}' normalized='{op_norm}' err={e}."
                )
                if strict:
                    raise
                logger.warning(msg + " Falling back to compute-only estimate.")

        else:
            # Only log unsupported ops at debug level to avoid noise.
            if op_in:
                logger.debug(
                    str(
                f"[PIM] Trace backend: skip unsupported op for {getattr(node,'name','?')}: "
                f"op='{op_in}' normalized='{op_norm}'. Using compute-only estimate."
                    )
                )

        return PimFastBackend().estimate_s(cm, node, dev, label, ctx)

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

__all__ = [name for name in globals() if not name.startswith("__")]
