
from __future__ import annotations

"""
ONNX -> TaskGraph adapter.

Goal:
- Keep the rest of the simulator (scheduler/cost_model/eval) unchanged.
- Replace the hand-written model_definition graph with a TaskGraph derived from an ONNX model.

Design choices:
- One ONNX NodeProto -> one TaskNode.
- Dependencies are inferred from tensor producer/consumer relationships.
- FLOPs + activation element counts are estimated from inferred tensor shapes (best-effort).
- Weight bytes are estimated from initializer tensor sizes.

Notes:
- This module requires the `onnx` Python package at runtime.
- If shape inference fails or some dims are symbolic/unknown, we fall back to conservative defaults.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set
import math
import os
import re
import logging

from task_graph import TaskGraph, TaskNode
from config import OPERATOR_DEVICE_ALLOWED

logger = logging.getLogger(__name__)


def _upper(x: Any, default: str = "") -> str:
    try:
        s = str(x).strip().upper()
        return s if s else default
    except Exception:
        return default


def _prod(dims: Iterable[int]) -> int:
    p = 1
    for d in dims:
        d = int(d)
        if d <= 0:
            # unknown dim: treat as 1 (caller may have substituted symbolic dims already)
            d = 1
        p *= d
    return int(p)


def _guess_layer_id_from_names(*names: str) -> Optional[int]:
    """
    Try to extract a transformer-like layer id from common naming patterns:
      - "layers.12." / "layer.12." / "layer_12_" / "L12_" etc.
    """
    patts = [
        r"(?:^|[^\w])layers[._/](\d+)(?:[^\w]|$)",
        r"(?:^|[^\w])layer[._/](\d+)(?:[^\w]|$)",
        r"(?:^|[^\w])layer_(\d+)(?:[^\w]|$)",
        r"(?:^|[^\w])l(\d+)(?:[^\w]|$)",
        r"^L(\d+)[_\-/\.]",
    ]
    for nm in names:
        if not nm:
            continue
        s = str(nm)
        for p in patts:
            m = re.search(p, s, flags=re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
    return None


def _op_allowed_from_config(op_name: str) -> Dict[str, bool]:
    key = _upper(op_name, "")
    # NOTE: current scheduler checks `allowed[dev.type]` (cpu/npu/pim).
    # config.py uses keys like pima/pimd in some configs; we keep what is provided.
    return (OPERATOR_DEVICE_ALLOWED.get(key) or {}).copy()


# -----------------------------
# Shape extraction helpers
# -----------------------------

@dataclass(frozen=True)
class TensorInfo:
    shape: Tuple[int, ...]
    # ONNX dtype is optional; we mainly use dtype_bytes from cfg.
    onnx_dtype: Optional[int] = None


def _build_symbol_map(cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Map common symbolic dimension names to concrete integers.
    Users can override/extend via cfg['onnx_symbolic_dims'].
    """
    batch = int(cfg.get("batch", 1) or 1)
    max_seq_len = int(cfg.get("prefill_len", 128) + cfg.get("decode_len", 32))
    sym = {
        "batch": batch,
        "b": batch,
        "bs": batch,
        "n": batch,
        "seq": max_seq_len,
        "seqlen": max_seq_len,
        "seq_len": max_seq_len,
        "sequence": max_seq_len,
        "s": max_seq_len,
        "t": max_seq_len,
        "tokens": max_seq_len,
        "token": max_seq_len,
        "time": max_seq_len,
        "len": max_seq_len,
    }
    user = cfg.get("onnx_symbolic_dims") or cfg.get("onnx_dim_map") or {}
    if isinstance(user, dict):
        for k, v in user.items():
            try:
                sym[str(k).strip().lower()] = int(v)
            except Exception:
                continue
    return sym


def _resolve_dim_param(dim_param: str, sym_map: Dict[str, int]) -> Optional[int]:
    if not dim_param:
        return None
    key = str(dim_param).strip().lower()
    # direct hit
    if key in sym_map:
        return int(sym_map[key])
    # substring hit (common in exporters: "batch_size", "seq_length")
    for k, v in sym_map.items():
        if k and k in key:
            return int(v)
    return None


def _tensor_shape_from_valueinfo(vi, sym_map: Dict[str, int]) -> Optional[Tuple[int, ...]]:
    try:
        t = vi.type.tensor_type
        shp = []
        for d in t.shape.dim:
            # Prefer concrete dim_value.
            if getattr(d, "dim_value", 0):
                shp.append(int(d.dim_value))
                continue
            # Else try symbolic dim_param.
            dp = getattr(d, "dim_param", "") or ""
            resolved = _resolve_dim_param(dp, sym_map)
            if resolved is not None:
                shp.append(int(resolved))
            else:
                shp.append(1)  # unknown => 1
        return tuple(int(x) for x in shp)
    except Exception:
        return None


def _collect_tensor_infos_from_model(model, cfg: Dict[str, Any]) -> Tuple[Dict[str, TensorInfo], Set[str], Dict[str, int]]:
    """
    Returns:
      - tensor_infos: value_name -> TensorInfo(shape, dtype)
      - initializer_names: set of initializer tensor names (weights/constants)
      - initializer_numel: initializer_name -> number of elements
    """
    sym_map = _build_symbol_map(cfg)
    tensor_infos: Dict[str, TensorInfo] = {}
    initializer_names: Set[str] = set()
    initializer_numel: Dict[str, int] = {}

    g = model.graph

    # Initializers carry exact shapes (and dtypes)
    try:
        for init in g.initializer:
            initializer_names.add(str(init.name))
            dims = tuple(int(x) for x in getattr(init, "dims", []) or [])
            if dims:
                tensor_infos[str(init.name)] = TensorInfo(shape=dims, onnx_dtype=int(getattr(init, "data_type", 0) or 0))
                initializer_numel[str(init.name)] = int(_prod(dims))
            else:
                initializer_numel[str(init.name)] = 0
    except Exception:
        pass

    # User can explicitly override input shapes (helps if the exporter left them symbolic).
    input_over = cfg.get("onnx_input_shapes") or {}
    if not isinstance(input_over, dict):
        input_over = {}

    def add_vi_list(vlist):
        for vi in vlist:
            try:
                name = str(vi.name)
            except Exception:
                continue
            # Override if user provided explicit shape for this value.
            if name in input_over:
                try:
                    shp = tuple(int(x) for x in input_over[name])
                    tensor_infos[name] = TensorInfo(shape=shp, onnx_dtype=None)
                    continue
                except Exception:
                    pass
            shp = _tensor_shape_from_valueinfo(vi, sym_map)
            if shp is None:
                continue
            # Keep existing initializer info if present.
            if name not in tensor_infos:
                tensor_infos[name] = TensorInfo(shape=shp, onnx_dtype=int(getattr(vi.type.tensor_type, "elem_type", 0) or 0))

    try:
        add_vi_list(g.input)
        add_vi_list(g.output)
        add_vi_list(g.value_info)
    except Exception:
        pass

    return tensor_infos, initializer_names, initializer_numel


# -----------------------------
# FLOPs estimation (best-effort)
# -----------------------------

def _matmul_flops(a: Tuple[int, ...], b: Tuple[int, ...], y: Tuple[int, ...]) -> int:
    # a: [..., M, K], b: [..., K, N], y: [..., M, N]
    if len(a) < 2 or len(b) < 2 or len(y) < 2:
        return 0
    M = int(a[-2])
    K = int(a[-1])
    N = int(b[-1])
    batch = _prod(y[:-2]) if len(y) > 2 else 1
    if M <= 0 or K <= 0 or N <= 0:
        return 0
    return int(2 * batch * M * N * K)


def _conv_flops(x: Tuple[int, ...], w: Tuple[int, ...], y: Tuple[int, ...], groups: int = 1) -> int:
    # Common NCHW: x=[N,Cin,H,W], w=[Cout,Cin/group,kH,kW], y=[N,Cout,Ho,Wo]
    if len(y) < 3:
        return 0
    N = int(y[0]) if len(y) >= 1 else 1
    Cout = int(y[1]) if len(y) >= 2 else 1
    out_spatial = _prod(y[2:])
    Cin = int(x[1]) if len(x) >= 2 else (int(w[1]) * groups if len(w) >= 2 else 0)
    if len(w) >= 3:
        k = _prod(w[2:])
    else:
        k = 1
    if Cin <= 0 or Cout <= 0:
        return 0
    cin_g = max(1, int(Cin // max(1, groups)))
    return int(2 * max(1, N) * max(1, out_spatial) * max(1, Cout) * max(1, cin_g) * max(1, k))


def estimate_node_flops(op_type: str, in_shapes: List[Tuple[int, ...]], out_shapes: List[Tuple[int, ...]], attrs: Dict[str, Any]) -> int:
    op = _upper(op_type, "")
    if not out_shapes:
        return 0

    y = out_shapes[0]
    out_elems = _prod(y)

    if op in ("MATMUL", "GEMM"):
        if len(in_shapes) >= 2:
            return _matmul_flops(in_shapes[0], in_shapes[1], y)
        return 0

    if op in ("CONV", "CONVTRANSPOSE"):
        groups = int(attrs.get("group", attrs.get("groups", 1)) or 1)
        if len(in_shapes) >= 2:
            return _conv_flops(in_shapes[0], in_shapes[1], y, groups=groups)
        return 0

    if op in ("ADD", "SUB", "MUL", "DIV", "POW", "MAX", "MIN"):
        return int(out_elems)

    if op in ("RELU", "GELU", "SILU", "SWISH", "TANH", "SIGMOID", "CLIP", "LEAKYRELU"):
        return int(out_elems)

    if op == "SOFTMAX":
        # rough: exp+sum+div etc
        return int(5 * out_elems)

    if "NORM" in op:
        # rough layer/rms norm
        return int(5 * out_elems)

    if op in ("TRANSPOSE", "RESHAPE", "SQUEEZE", "UNSQUEEZE", "IDENTITY", "CAST", "CONCAT", "SPLIT"):
        return 0

    # fallback: assume 1 flop per output element
    return int(out_elems)


# -----------------------------
# Main builder
# -----------------------------

def build_task_graph_from_onnx(
    onnx_path: str,
    cfg: Dict[str, Any],
    *,
    dtype_bytes: int,
) -> TaskGraph:
    """
    Build TaskGraph from an ONNX model file.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model.
    cfg:
        The same config dict used by the rest of the simulator. We use it for
        batch/seq symbolic dim substitution and optional overrides:
          - onnx_input_shapes: {input_name: [dims...]}
          - onnx_symbolic_dims: {symbol: value}
    dtype_bytes:
        Base dtype bytes used to estimate weight/activation sizes when the ONNX dtype is absent.
    """
    try:
        import onnx  # type: ignore
        from onnx import shape_inference  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "ONNX graph support requires the `onnx` package. "
            "Please install it (e.g., `pip install onnx`) and retry."
        ) from e

    if not isinstance(onnx_path, str) or not onnx_path.strip():
        raise ValueError("onnx_path is empty")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = onnx.load(onnx_path)

    # Best-effort shape inference.
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("[ONNX] shape_inference failed (%s). Proceeding with partial shapes.", e)

    tensor_infos, initializer_names, initializer_numel = _collect_tensor_infos_from_model(model, cfg)

    g = TaskGraph()

    # Producer map: tensor_name -> node_id
    produced_by: Dict[str, str] = {}
    used_node_ids: Set[str] = set()

    def unique_id(base: str) -> str:
        b = re.sub(r"[^0-9a-zA-Z_]+", "_", base).strip("_") or "node"
        nid = b
        k = 1
        while nid in used_node_ids:
            k += 1
            nid = f"{b}_{k}"
        used_node_ids.add(nid)
        return nid

    # Map some ONNX op_types to internal coarse names where it helps existing cost-model heuristics.
    # (We still keep the raw op_type in attrs['onnx_op_type']).
    OP_NAME_MAP = {
        "LAYERNORMALIZATION": "LN",
        "LAYERNORM": "LN",
        "RMSNORMALIZATION": "LN",
        "RMSNORM": "LN",
    }

    nodes = list(getattr(model.graph, "node", []) or [])
    for idx, node in enumerate(nodes):
        op_type = str(getattr(node, "op_type", "") or "")
        op_up = _upper(op_type, "")
        raw_name = str(getattr(node, "name", "") or "")
        node_id = unique_id(raw_name or f"n{idx}_{op_type or 'Op'}")

        # Determine a display/operator name (used by cost_model heuristics)
        name_for_cost = OP_NAME_MAP.get(op_up, op_type or "Op")

        # Gather shapes for FLOPs/bytes estimation
        in_shapes: List[Tuple[int, ...]] = []
        out_shapes: List[Tuple[int, ...]] = []

        # Node attributes from ONNX
        onnx_attrs: Dict[str, Any] = {}
        try:
            for a in getattr(node, "attribute", []) or []:
                try:
                    onnx_attrs[str(a.name)] = onnx.helper.get_attribute_value(a)  # type: ignore
                except Exception:
                    # Fallback: keep raw attr object.
                    onnx_attrs[str(a.name)] = a
        except Exception:
            pass

        # Weight detection: initializer inputs
        weight_inits: List[str] = []
        act_inputs: List[str] = []
        for inp in list(getattr(node, "input", []) or []):
            inp = str(inp)
            if not inp:
                continue
            if inp in initializer_names:
                weight_inits.append(inp)
            else:
                act_inputs.append(inp)

            ti = tensor_infos.get(inp)
            if ti and ti.shape:
                in_shapes.append(tuple(int(x) for x in ti.shape))

        for out in list(getattr(node, "output", []) or []):
            out = str(out)
            if not out:
                continue
            ti = tensor_infos.get(out)
            if ti and ti.shape:
                out_shapes.append(tuple(int(x) for x in ti.shape))

        # Element counts for generic activation byte fallback in cost_model
        in_elems = 0
        for tname in act_inputs:
            ti = tensor_infos.get(tname)
            if ti and ti.shape:
                in_elems += _prod(ti.shape)
        out_elems = 0
        for tname in list(getattr(node, "output", []) or []):
            ti = tensor_infos.get(str(tname))
            if ti and ti.shape:
                out_elems += _prod(ti.shape)

        # Weights
        w_elems = 0
        for wname in weight_inits:
            w_elems += int(initializer_numel.get(wname, 0) or 0)
        weight_size = int(w_elems * int(dtype_bytes))

        # FLOPs
        flops = int(estimate_node_flops(op_type, in_shapes, out_shapes, onnx_attrs))

        # Try to recover a "layer id" for KV RR mapping & per-layer opt override.
        layer_id = _guess_layer_id_from_names(raw_name, *weight_inits)

        attrs: Dict[str, Any] = {}
        attrs.update(onnx_attrs)
        attrs.update({
            "onnx_op_type": op_type,
            "onnx_name": raw_name,
            "in_elems": int(in_elems),
            "out_elems": int(out_elems),
            "generic_flops": True,   # allow cost_model to apply density scaling to default flops if desired
        })
        if layer_id is not None:
            attrs["layer"] = int(layer_id)

        # NOTE: TaskNode supports only one weight_id. We aggregate multiple initializers into one logical "weight group".
        weight_id = None
        if weight_inits and weight_size > 0:
            # Keep it deterministic and reasonably short.
            if len(weight_inits) == 1:
                weight_id = str(weight_inits[0])
            elif len(weight_inits) <= 4:
                weight_id = "+".join(str(x) for x in weight_inits)
            else:
                weight_id = f"{weight_inits[0]}+{len(weight_inits)-1}more"

        allowed = _op_allowed_from_config(name_for_cost)

        g.add_node(TaskNode(
            id=str(node_id),
            name=str(name_for_cost),
            flops=float(flops),
            weight_id=weight_id,
            weight_size=int(weight_size),
            allowed=allowed,
            attrs=attrs,
        ))

        # Add edges based on data dependencies (skip initializer inputs).
        for inp in act_inputs:
            src = produced_by.get(inp)
            if src and src != node_id:
                try:
                    g.add_edge(src, node_id)
                except Exception:
                    pass

        # Record outputs as produced by this node.
        for out in list(getattr(node, "output", []) or []):
            out = str(out)
            if out:
                produced_by[out] = node_id

    return g
