from __future__ import annotations

"""Optimization interface layer for the operator-partitioning simulator.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from task_graph import TaskGraph, TaskNode


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def _lower(x: Any, default: str = "") -> str:
    try:
        s = str(x).strip().lower()
        return s if s else default
    except Exception:
        return default


def _upper(x: Any, default: str = "") -> str:
    try:
        s = str(x).strip().upper()
        return s if s else default
    except Exception:
        return default


def _clamp01(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return float(min(1.0, max(0.0, v)))
    except Exception:
        return float(default)


def _ceil_div(a: int, b: int) -> int:
    a = int(a)
    b = int(b)
    return (a + b - 1) // b if b > 0 else 0


def _dtype_bytes_from_bits(bits: int) -> float:
    # Keep it local to avoid importing cost_model (which imports many heavy deps).
    bits = int(bits)
    if bits <= 0:
        return 0.0
    return float(bits) / 8.0


def _match_any(name_up: str, patterns: Iterable[str]) -> bool:
    """Substring match (case-insensitive) against a list of patterns."""
    for p in patterns:
        if not p:
            continue
        if str(p).strip().upper() in name_up:
            return True
    return False


@dataclass
class QuantizationSpec:
    enable: bool = False

    mode: str = "none"  # 'none' | 'weight_only' | 'w8a8' | 'w4a16' | 'w4a8'
    weight_bits: int = 16
    activation_bits: Optional[int] = None
    activation_io: str = "fp16"  # 'fp16' | 'int8' | 'int4'

    # Group-wise quantization parameters (GPTQ/AWQ/torchao/ORT int4 commonly use group_size=128).
    group_size: int = 128
    per_channel: bool = True

    scale_dtype_bits: int = 16  # fp16 by default

    # Optional performance hint: speedup per device type.
    # Example: {"npu": 1.5, "pim": 1.0}
    speedup: Dict[str, float] = field(default_factory=dict)

    # Filters
    apply_to: List[str] = field(default_factory=list)  # op-name substrings; empty => default linear ops
    exclude: List[str] = field(default_factory=list)

    def enabled(self) -> bool:
        return bool(self.enable) and self.mode not in ("none", "off", "disable", "disabled")


@dataclass
class WeightSparsitySpec:
    enable: bool = False
    method: str = "global"  # 'global' | 'layerwise' | ... (informational)

    # pattern:
    #   - 'unstructured' (random/magnitude/global)
    #   - 'n:m' (semi-structured, e.g., 2:4)
    #   - 'block' (block sparse)
    pattern: str = "unstructured"
    sparsity: float = 0.0  # fraction of zeros

    # n:m structured
    n: Optional[int] = None
    m: Optional[int] = None

    # storage:
    #  - 'dense'      : keep dense tensor with zeros (size unchanged)
    #  - 'compressed' : assume sparse encoding (size scales with nnz + metadata)
    storage: str = "dense"

    # Whether we assume compute uses sparse kernels (reduces effective FLOPs).
    assume_sparse_compute: bool = False

    # Metadata overhead approximation for compressed formats.
    # Interpreted as "extra bytes per non-zero" (e.g., CSR indices etc.).
    metadata_bytes_per_nnz: float = 0.0

    speedup: Dict[str, float] = field(default_factory=dict)
    apply_to: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    def enabled(self) -> bool:
        if not bool(self.enable):
            return False

        pat = _lower(self.pattern, "unstructured")
        if pat in ("n:m", "nm"):
            try:
                n = int(self.n) if self.n is not None else 0
                m = int(self.m) if self.m is not None else 0
                if m > 0 and 0 < n < m:
                    return True
            except Exception:
                pass

        return self.sparsity > 0.0

    def density(self) -> float:
        if self.pattern in ("n:m", "nm", "2:4", "semi", "semi_structured", "semistructured") and self.n and self.m and self.m > 0:
            return float(self.n) / float(self.m)
        return float(1.0 - _clamp01(self.sparsity, 0.0))


@dataclass
class ActivationSparsitySpec:
    enable: bool = False
    mode: str = "threshold"  # 'threshold' | 'topk' | 'dynamic' (informational)

    # You can specify either sparsity or density.
    sparsity: Optional[float] = None
    density: Optional[float] = None

    # Optional per-phase expectation.
    density_by_phase: Dict[str, float] = field(default_factory=dict)  # {'prefill':0.8,'decode':0.6}

    # storage: keep dense activations or assume compressed.
    storage: str = "dense"

    assume_sparse_compute: bool = False
    speedup: Dict[str, float] = field(default_factory=dict)

    apply_to: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    def enabled(self) -> bool:
        return bool(self.enable) and (self.sparsity is not None or self.density is not None or bool(self.density_by_phase))

    def density_default(self) -> float:
        if self.density is not None:
            return _clamp01(self.density, 1.0)
        if self.sparsity is not None:
            return 1.0 - _clamp01(self.sparsity, 0.0)
        # default: no sparsity
        return 1.0


@dataclass
class AttentionSparsitySpec:
    enable: bool = False
    pattern: str = "dense"  # 'dense' | 'local' | 'block' | 'matrix'

    # Local/sliding window attention.
    # Aligns with FlashAttention interface window_size=(left,right).
    window_left: int = -1
    window_right: int = -1

    # Block-sparse attention.
    block_size: int = 128
    blocks_left: int = 1
    blocks_right: int = 0

    # Generic sparse attention matrix density.
    density: float = 1.0

    apply_to: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)

    def enabled(self) -> bool:
        if not self.enable:
            return False
        pat = _lower(self.pattern, "dense")
        return pat not in ("dense", "none", "off", "disable", "disabled")


@dataclass
class OptimizationConfig:
    quant: QuantizationSpec = field(default_factory=QuantizationSpec)
    w_sparsity: WeightSparsitySpec = field(default_factory=WeightSparsitySpec)
    a_sparsity: ActivationSparsitySpec = field(default_factory=ActivationSparsitySpec)
    attn_sparsity: AttentionSparsitySpec = field(default_factory=AttentionSparsitySpec)

    # Optional per-layer overrides (stringified layer index -> dict of sections).
    per_layer: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def parse_optimization_config(cfg: Dict[str, Any]) -> OptimizationConfig:

    opt_root = (
        _as_dict(cfg.get("optimizations"))
        or _as_dict(cfg.get("optimization"))
        or _as_dict(cfg.get("optim"))
    )

    # Allow both styles simultaneously: top-level keys override opt_root.
    quant_d = _as_dict(cfg.get("quantization")) or _as_dict(opt_root.get("quantization"))
    spars_d = _as_dict(cfg.get("sparsity")) or _as_dict(opt_root.get("sparsity"))
    attn_d = _as_dict(cfg.get("attention_sparsity")) or _as_dict(opt_root.get("attention_sparsity"))

    # ---- Quantization ----
    q = QuantizationSpec(
        enable=bool(quant_d.get("enable", quant_d.get("enabled", False))),
        mode=_lower(quant_d.get("mode", quant_d.get("type", "none")), "none"),
        weight_bits=int(quant_d.get("weight_bits", quant_d.get("w_bits", quant_d.get("bits", 16))) or 16),
        activation_bits=(
            None
            if quant_d.get("activation_bits", quant_d.get("a_bits")) is None
            else int(quant_d.get("activation_bits", quant_d.get("a_bits")) or 0)
        ),
        activation_io=_lower(quant_d.get("activation_io", quant_d.get("act_io", "fp16")), "fp16"),
        group_size=int(quant_d.get("group_size", 128) or 128),
        per_channel=bool(quant_d.get("per_channel", quant_d.get("per_row", True))),
        scale_dtype_bits=int(quant_d.get("scale_dtype_bits", quant_d.get("scale_bits", 16)) or 16),
        speedup={str(k).lower(): float(v) for k, v in _as_dict(quant_d.get("speedup")).items()},
        apply_to=[str(x) for x in _as_list(quant_d.get("apply_to"))],
        exclude=[str(x) for x in _as_list(quant_d.get("exclude"))],
    )

    # ---- Weight sparsity ----
    w_d = _as_dict(spars_d.get("weight")) or _as_dict(spars_d.get("weights")) or _as_dict(cfg.get("weight_sparsity"))
    ws = WeightSparsitySpec(
        enable=bool(w_d.get("enable", w_d.get("enabled", False))),
        method=_lower(w_d.get("method", "global"), "global"),
        pattern=_lower(w_d.get("pattern", w_d.get("type", "unstructured")), "unstructured"),
        sparsity=_clamp01(w_d.get("sparsity", w_d.get("ratio", 0.0)), 0.0),
        n=(None if w_d.get("n") is None else int(w_d.get("n") or 0)),
        m=(None if w_d.get("m") is None else int(w_d.get("m") or 0)),
        storage=_lower(w_d.get("storage", "dense"), "dense"),
        assume_sparse_compute=bool(w_d.get("assume_sparse_compute", w_d.get("sparse_compute", False))),
        metadata_bytes_per_nnz=float(w_d.get("metadata_bytes_per_nnz", w_d.get("index_bytes", 0.0)) or 0.0),
        speedup={str(k).lower(): float(v) for k, v in _as_dict(w_d.get("speedup")).items()},
        apply_to=[str(x) for x in _as_list(w_d.get("apply_to"))],
        exclude=[str(x) for x in _as_list(w_d.get("exclude"))],
    )

    # Convenience: allow "2:4" string.
    if ws.pattern in ("2:4", "2_4", "2-4"):
        ws.pattern = "n:m"
        ws.n, ws.m = 2, 4
        # semi-structured typically implies sparse compute + compressed weights
        if "assume_sparse_compute" not in w_d:
            ws.assume_sparse_compute = True
        if "storage" not in w_d:
            ws.storage = "compressed"

    # ---- Activation sparsity ----
    a_d = _as_dict(spars_d.get("activation")) or _as_dict(spars_d.get("activations")) or _as_dict(cfg.get("activation_sparsity"))
    aspec = ActivationSparsitySpec(
        enable=bool(a_d.get("enable", a_d.get("enabled", False))),
        mode=_lower(a_d.get("mode", a_d.get("type", "threshold")), "threshold"),
        sparsity=(None if a_d.get("sparsity") is None else _clamp01(a_d.get("sparsity"), 0.0)),
        density=(None if a_d.get("density") is None else _clamp01(a_d.get("density"), 1.0)),
        density_by_phase={str(k).lower(): _clamp01(v, 1.0) for k, v in _as_dict(a_d.get("density_by_phase", a_d.get("density_by_pass", {}))).items()},
        storage=_lower(a_d.get("storage", "dense"), "dense"),
        assume_sparse_compute=bool(a_d.get("assume_sparse_compute", a_d.get("sparse_compute", False))),
        speedup={str(k).lower(): float(v) for k, v in _as_dict(a_d.get("speedup")).items()},
        apply_to=[str(x) for x in _as_list(a_d.get("apply_to"))],
        exclude=[str(x) for x in _as_list(a_d.get("exclude"))],
    )

    # ---- Attention sparsity ----
    att = AttentionSparsitySpec(
        enable=bool(attn_d.get("enable", attn_d.get("enabled", False))),
        pattern=_lower(attn_d.get("pattern", attn_d.get("type", "dense")), "dense"),
        window_left=int(attn_d.get("window_left", attn_d.get("left", -1)) or -1),
        window_right=int(attn_d.get("window_right", attn_d.get("right", -1)) or -1),
        block_size=int(attn_d.get("block_size", 128) or 128),
        blocks_left=int(attn_d.get("blocks_left", attn_d.get("block_left", 1)) or 1),
        blocks_right=int(attn_d.get("blocks_right", attn_d.get("block_right", 0)) or 0),
        density=_clamp01(attn_d.get("density", 1.0), 1.0),
        apply_to=[str(x) for x in _as_list(attn_d.get("apply_to"))],
        exclude=[str(x) for x in _as_list(attn_d.get("exclude"))],
    )

    per_layer = _as_dict(cfg.get("per_layer")) or _as_dict(opt_root.get("per_layer"))
    # Also accept per_layer overrides inside each section.
    for sec_name, sec in (("quantization", quant_d), ("weight_sparsity", w_d), ("activation_sparsity", a_d), ("attention_sparsity", attn_d)):
        pl = _as_dict(sec.get("per_layer"))
        if pl:
            for k, v in pl.items():
                per_layer.setdefault(str(k), {})
                per_layer[str(k)].setdefault(sec_name, {})
                if isinstance(v, dict):
                    per_layer[str(k)][sec_name].update(v)

    return OptimizationConfig(quant=q, w_sparsity=ws, a_sparsity=aspec, attn_sparsity=att, per_layer=per_layer)


def _default_linear_ops() -> Tuple[str, ...]:
    # Most transformer weights are in these projections.
    return (
        "Q", "K", "V", "O",
        "FFN_W1", "FFN_W2", "FFN_W3",
        "WQ", "WK", "WV", "WO", "W1", "W2", "W3",
        "EXPERT", "MOE", "MLP",
        "EMBED", "LM_HEAD",
    )


def _should_apply(node_name_up: str, *, apply_to: List[str], exclude: List[str], default_apply: Iterable[str]) -> bool:
    if exclude and _match_any(node_name_up, exclude):
        return False
    if apply_to:
        return _match_any(node_name_up, apply_to)
    return _match_any(node_name_up, default_apply)


def _merge_overrides(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = v
    return out


def _effective_phase_density(spec: ActivationSparsitySpec, phase: str) -> float:
    p = _lower(phase, "")
    if spec.density_by_phase and p in spec.density_by_phase:
        return _clamp01(spec.density_by_phase[p], spec.density_default())
    return spec.density_default()


def apply_optimizations_to_graph(
    g: TaskGraph,
    cfg: Dict[str, Any],
    *,
    base_weight_dtype_bytes: int,
    shape: Any = None,
) -> OptimizationConfig:
    """Annotate the graph nodes in-place and adjust weight_size for storage effects."""
    opt = parse_optimization_config(cfg)
    if not (opt.quant.enabled() or opt.w_sparsity.enabled() or opt.a_sparsity.enabled() or opt.attn_sparsity.enabled()):
        return opt

    # Create a stable top-level opt dict on each node.
    for node in g.nodes.values():
        if not isinstance(getattr(node, "attrs", None), dict):
            node.attrs = {}
        node.attrs.setdefault("opt", {})
        node.attrs["opt"].setdefault("base_weight_dtype_bytes", int(base_weight_dtype_bytes))

    # Helper: per-layer override lookup.
    def layer_override(layer: Optional[int]) -> Dict[str, Any]:
        if layer is None:
            return {}
        return _as_dict(opt.per_layer.get(str(int(layer))))

    for node in g.nodes.values():
        attrs = node.attrs or {}
        optd = attrs.get("opt") or {}
        layer = attrs.get("layer")
        try:
            layer_i = None if layer is None else int(layer)
        except Exception:
            layer_i = None

        name_up = _upper(getattr(node, "name", ""), "")

        # -----------------------------
        # Resolve per-layer overrides
        # -----------------------------
        lo = layer_override(layer_i)
        q_over = _as_dict(lo.get("quantization"))
        ws_over = _as_dict(lo.get("weight_sparsity"))
        as_over = _as_dict(lo.get("activation_sparsity"))
        att_over = _as_dict(lo.get("attention_sparsity"))
        if q_over:
            optd.setdefault("quantization", {})
            optd["quantization"] = _merge_overrides(_as_dict(optd.get("quantization")), q_over)
        if ws_over:
            optd.setdefault("weight_sparsity", {})
            optd["weight_sparsity"] = _merge_overrides(_as_dict(optd.get("weight_sparsity")), ws_over)
        if as_over:
            optd.setdefault("activation_sparsity", {})
            optd["activation_sparsity"] = _merge_overrides(_as_dict(optd.get("activation_sparsity")), as_over)
        if att_over:
            optd.setdefault("attention_sparsity", {})
            optd["attention_sparsity"] = _merge_overrides(_as_dict(optd.get("attention_sparsity")), att_over)

        # -----------------------------
        # Apply attention sparsity tags
        # -----------------------------
        if opt.attn_sparsity.enabled():
            # By default apply to attention core ops (QK / Softmax / SV) only.
            default_apply = ("QK", "SV", "SOFTMAX", "ATTN")
            if _should_apply(name_up, apply_to=opt.attn_sparsity.apply_to, exclude=opt.attn_sparsity.exclude, default_apply=default_apply):
                optd["attention_sparsity"] = {
                    "pattern": _lower(opt.attn_sparsity.pattern, "dense"),
                    "window_left": int(opt.attn_sparsity.window_left),
                    "window_right": int(opt.attn_sparsity.window_right),
                    "block_size": int(opt.attn_sparsity.block_size),
                    "blocks_left": int(opt.attn_sparsity.blocks_left),
                    "blocks_right": int(opt.attn_sparsity.blocks_right),
                    "density": float(opt.attn_sparsity.density),
                }

        # -----------------------------
        # Apply activation sparsity tags
        # -----------------------------
        if opt.a_sparsity.enabled():
            default_apply = ("FFN", "SWIGLU", "SILU", "GELU", "RELU", "QK", "SV", "SOFTMAX", "ADD", "LN", "NORM")
            if _should_apply(name_up, apply_to=opt.a_sparsity.apply_to, exclude=opt.a_sparsity.exclude, default_apply=default_apply):
                optd["activation_sparsity"] = {
                    "mode": _lower(opt.a_sparsity.mode, "threshold"),
                    "density": float(opt.a_sparsity.density_default()),
                    "density_by_phase": dict(opt.a_sparsity.density_by_phase),
                    "storage": _lower(opt.a_sparsity.storage, "dense"),
                    "assume_sparse_compute": bool(opt.a_sparsity.assume_sparse_compute),
                    "speedup": dict(opt.a_sparsity.speedup),
                }

        # -----------------------------
        # Apply weight sparsity + quantization to weight-bearing nodes
        # -----------------------------
        has_weight = bool(getattr(node, "weight_id", None)) and int(getattr(node, "weight_size", 0) or 0) > 0
        if has_weight:
            orig_w_bytes = int(getattr(node, "weight_size", 0) or 0)
            optd.setdefault("orig_weight_size", orig_w_bytes)

            # Weight sparsity parameters
            ws_apply = opt.w_sparsity.enabled() and _should_apply(
                name_up,
                apply_to=opt.w_sparsity.apply_to,
                exclude=opt.w_sparsity.exclude,
                default_apply=_default_linear_ops(),
            )
            if ws_apply:
                w_density = float(opt.w_sparsity.density())
                optd["weight_sparsity"] = {
                    "method": _lower(opt.w_sparsity.method, "global"),
                    "pattern": _lower(opt.w_sparsity.pattern, "unstructured"),
                    "sparsity": float(opt.w_sparsity.sparsity),
                    "density": float(w_density),
                    "n": None if opt.w_sparsity.n is None else int(opt.w_sparsity.n),
                    "m": None if opt.w_sparsity.m is None else int(opt.w_sparsity.m),
                    "storage": _lower(opt.w_sparsity.storage, "dense"),
                    "assume_sparse_compute": bool(opt.w_sparsity.assume_sparse_compute),
                    "metadata_bytes_per_nnz": float(opt.w_sparsity.metadata_bytes_per_nnz),
                    "speedup": dict(opt.w_sparsity.speedup),
                }
            else:
                w_density = 1.0

            # Quantization parameters
            q_apply = opt.quant.enabled() and _should_apply(
                name_up,
                apply_to=opt.quant.apply_to,
                exclude=opt.quant.exclude,
                default_apply=_default_linear_ops(),
            )
            if q_apply:
                act_bits = int(opt.quant.activation_bits or 16)
                optd["quantization"] = {
                    "mode": _lower(opt.quant.mode, "none"),
                    "weight_bits": int(opt.quant.weight_bits),
                    "activation_bits": int(act_bits),
                    "activation_io": _lower(opt.quant.activation_io, "fp16"),
                    "group_size": int(opt.quant.group_size),
                    "per_channel": bool(opt.quant.per_channel),
                    "scale_dtype_bits": int(opt.quant.scale_dtype_bits),
                    "speedup": dict(opt.quant.speedup),
                }
            # -----------------------------
            # Weight storage size update
            # -----------------------------
            new_bytes = orig_w_bytes

            # 1) Quantization changes element width (weights)
            if q_apply:
                # Estimate number of (dense) weight elements based on base dtype.
                base_b = max(1, int(base_weight_dtype_bytes))
                elems = int(max(0, orig_w_bytes // base_b))
                w_bpe = _dtype_bytes_from_bits(int(opt.quant.weight_bits))
                q_bytes = float(elems) * float(w_bpe)

                # Group-wise scales (+ optional zero-point).
                gsz = int(opt.quant.group_size or 0)
                if gsz > 0 and elems > 0 and int(opt.quant.weight_bits) < 16:
                    groups = _ceil_div(elems, gsz)
                    scale_b = _dtype_bytes_from_bits(int(opt.quant.scale_dtype_bits))
                    q_bytes += float(groups) * float(scale_b)
                new_bytes = int(math.ceil(q_bytes))

            # 2) Sparsity changes stored nnz volume (optional)
            if ws_apply and _lower(opt.w_sparsity.storage, "dense") == "compressed":
                # Apply density to the (possibly quantized) payload bytes.
                payload = float(new_bytes) * float(w_density)

                # Add metadata per nnz if requested.
                if opt.w_sparsity.metadata_bytes_per_nnz and opt.w_sparsity.metadata_bytes_per_nnz > 0:
                    # Estimate nnz based on element count in base dtype.
                    base_b = max(1, int(base_weight_dtype_bytes))
                    elems = int(max(0, orig_w_bytes // base_b))
                    nnz = float(elems) * float(w_density)
                    payload += float(nnz) * float(opt.w_sparsity.metadata_bytes_per_nnz)
                new_bytes = int(math.ceil(payload))

            # Finalize
            node.weight_size = max(0, int(new_bytes))

        # -----------------------------
        # Apply activation quantization (optional inter-op activation dtype)
        # -----------------------------
        if opt.quant.enabled() and opt.quant.activation_bits is not None:
            # Only tag nodes if user explicitly wants inter-op activations quantized.
            if _lower(opt.quant.activation_io, "fp16") in ("int8", "int4"):
                # Tag everything by default; CostModel can decide which ops to use it.
                optd.setdefault("activation_quant", {})
                optd["activation_quant"] = {
                    "act_bits": int(opt.quant.activation_bits),
                    "act_dtype_bytes": float(_dtype_bytes_from_bits(int(opt.quant.activation_bits))),
                    "io": _lower(opt.quant.activation_io, "fp16"),
                }

                # KV-cache storage dtype can be smaller than model compute dtype.
                kv_ops = {"K", "V", "QK", "SV", "K_WRITE", "V_WRITE", "KV_READ", "KV_WRITE"}
                if (getattr(node, "name", "") or "").upper() in kv_ops:
                    optd.setdefault("kv_dtype_bytes", float(optd["activation_quant"].get("act_dtype_bytes", 0.0)))

        # Provide a unified speedup hint table (device_type -> scale)
        # This is purely optional; CostModel will ignore if absent.
        speedup: Dict[str, float] = {}
        if opt.quant.enabled() and opt.quant.speedup:
            for k, v in opt.quant.speedup.items():
                try:
                    speedup[str(k).lower()] = max(speedup.get(str(k).lower(), 1.0), float(v))
                except Exception:
                    pass
        if opt.w_sparsity.enabled() and opt.w_sparsity.speedup:
            for k, v in opt.w_sparsity.speedup.items():
                try:
                    speedup[str(k).lower()] = max(speedup.get(str(k).lower(), 1.0), float(v))
                except Exception:
                    pass
        if opt.a_sparsity.enabled() and opt.a_sparsity.speedup:
            for k, v in opt.a_sparsity.speedup.items():
                try:
                    speedup[str(k).lower()] = max(speedup.get(str(k).lower(), 1.0), float(v))
                except Exception:
                    pass
        if speedup:
            optd["speedup"] = speedup

        # Finally store opt dict back.
        attrs["opt"] = optd
        node.attrs = attrs

    return opt
