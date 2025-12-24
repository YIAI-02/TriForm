from __future__ import annotations
from dataclasses import dataclass, field
import math
import os
from typing import List, Dict, Tuple, Optional, Iterable, Set
from task_graph import TaskGraph, TaskNode
from config import OPERATOR_DEVICE_ALLOWED


def _env_str(name: str, default: str) -> str:
    try:
        v = os.environ.get(name, default)
        return str(v) if v is not None else str(default)
    except Exception:
        return str(default)


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, None)
        if v is None:
            return int(default)
        return int(str(v).strip())
    except Exception:
        return int(default)


def _normalize_split_by(v: str) -> str:
    """
    Control switch:
      - "layer"    : keep original per-layer graph (no intra-layer parallelism)
      - "head_num" : split projections/attention/ffn by head shards (tensor-parallel style)
    """
    s = (v or "").strip().lower()
    if s in ("head", "heads", "head_num", "headnum", "q_heads", "qheads", "n_heads", "nheads", "tp", "tensor", "tensor_parallel"):
        return "head"
    if s in ("layer", "layers", "layer_num", "layernum", "pp", "pipeline"):
        return "layer"
    if "head" in s:
        return "head"
    if "layer" in s:
        return "layer"
    return "layer"


def _default_split_by() -> str:
    # Can be overridden without touching code:
    #   GRAPH_SPLIT_BY=head_num   (or: head)
    #   GRAPH_SPLIT_BY=layer
    return _normalize_split_by(_env_str("GRAPH_SPLIT_BY", _env_str("MODEL_SPLIT_BY", "layer")))


def _default_split_shards() -> int:
    # Optional: limit shard count when split_by=head to avoid huge graphs.
    #   GRAPH_SPLIT_SHARDS=8
    # 0 means "auto" (use n_heads / n_kv_heads).
    return max(0, _env_int("GRAPH_SPLIT_SHARDS", _env_int("MODEL_SPLIT_SHARDS", 0)))


def _partition_ranges(total: int, parts: int) -> List[Tuple[int, int]]:
    """Return contiguous (start, count) ranges that sum to `total`."""
    total = int(max(0, total))
    if total <= 0:
        return []
    parts = int(max(1, min(parts, total)))
    base = total // parts
    rem = total % parts
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(parts):
        cnt = base + (1 if i < rem else 0)
        out.append((start, cnt))
        start += cnt
    return out


def _partition_sizes(total: int, parts: int) -> List[int]:
    return [cnt for _, cnt in _partition_ranges(total, parts)]


def _kv_head_for_q_head(q_idx: int, q_heads: int, kv_heads: int) -> int:
    """Map q-head index -> kv-head index for GQA/MQA."""
    q_heads = max(1, int(q_heads))
    kv_heads = max(1, int(kv_heads))
    q_idx = int(min(max(0, q_idx), q_heads - 1))
    return int((q_idx * kv_heads) // q_heads)


def _kv_heads_for_q_range(q_start: int, q_cnt: int, q_heads: int, kv_heads: int) -> Set[int]:
    s: Set[int] = set()
    for h in range(int(q_start), int(q_start) + int(q_cnt)):
        s.add(_kv_head_for_q_head(h, q_heads=q_heads, kv_heads=kv_heads))
    return s


def _make_head_to_shard_map(ranges: List[Tuple[int, int]]) -> Dict[int, int]:
    m: Dict[int, int] = {}
    for sid, (st, cnt) in enumerate(ranges):
        for h in range(st, st + cnt):
            m[int(h)] = int(sid)
    return m


@dataclass
class ModelShape:
    layer_num: int
    dim: int
    ffn_dim: int
    n_heads: int         # Q heads
    n_kv_heads: int      # shared KV heads (GQA/MQA)
    batch: int
    max_seq_len: int     # Maximum sequence length (for graph structure)

    # Graph partition knobs (read from env by default; can also be set via setattr(shape, ...)).
    # - split_by: "layer" (default, original graph) or "head_num"/"head" (split projections/FFN by head).
    # - split_shards: optional cap for shard count when split_by=head (0 means auto).
    split_by: str = field(default_factory=_default_split_by)
    split_shards: int = field(default_factory=_default_split_shards)

    @property
    def head_dim(self) -> int:
        # Default per-head dimension: dim // q_heads
        return self.dim // max(1, self.n_heads)

# ---- Common helpers ----

def split_even(total: int, parts: int) -> List[int]:
    """Split `total` into `parts` integers whose sum equals total.
    Example: total=10, parts=3 -> [4,3,3]
    """
    total = int(total or 0)
    parts = max(1, int(parts or 1))
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def get_op_allowed(op_name: str) -> Dict[str, bool]:
    key = str(op_name).strip().upper()
    return OPERATOR_DEVICE_ALLOWED.get(key,{}).copy()


def _effective_shards(shape: ModelShape, *, for_q: bool = False, for_kv: bool = False, for_ffn: bool = False) -> int:
    """
    Pick a shard count based on shape + optional cap.
    - for_q  : shard count targets n_heads
    - for_kv : shard count targets n_kv_heads
    - for_ffn: shard count targets n_heads (so it scales with the same parallel degree)
    """
    cap = int(getattr(shape, "split_shards", 0) or 0)
    if for_q:
        total = int(getattr(shape, "n_heads", 1) or 1)
    elif for_kv:
        total = int(getattr(shape, "n_kv_heads", getattr(shape, "n_heads", 1)) or 1)
    elif for_ffn:
        total = int(getattr(shape, "n_heads", 1) or 1)
    else:
        total = int(getattr(shape, "n_heads", 1) or 1)

    if cap > 0:
        return max(1, min(total, cap))
    return max(1, total)


def add_llama_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    b = shape.batch
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim
    q_dim, kv_dim, o_in_dim = qh * hd, kvh * hd, qh * hd
    base_attr = {
        "layer": l,
        "q_heads": qh,
        "kv_heads": kvh,
        "head_dim": hd,
        "dim": dim,
        "ffn_dim": ffn,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "o_dim": o_in_dim,
        "batch": b,
    }

    # LN1
    nid_LN1 = f"L{l}_LN1"
    g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    # Carry K/V cache writes to next layer input
    if l > 0:
        prev_kw = f"L{l-1}_K_write"
        prev_vw = f"L{l-1}_V_write"
        g.add_edge(prev_kw, nid_LN1)
        g.add_edge(prev_vw, nid_LN1)
    # Q/K/V
    nid_Q = f"L{l}_Q"
    nid_K = f"L{l}_K"
    nid_V = f"L{l}_V"
    g.add_node(TaskNode(nid_Q, "Q", flops=0.0,
                        weight_id=f"L{l}_WQ", weight_size=dim * q_dim * dtype_bytes,
                        attrs=dict(base_attr), allowed=get_op_allowed("Q")))
    g.add_node(TaskNode(nid_K, "K", flops=0.0,
                        weight_id=f"L{l}_WK", weight_size=dim * kv_dim * dtype_bytes,
                        attrs=dict(base_attr), allowed=get_op_allowed("K")))
    g.add_node(TaskNode(nid_V, "V", flops=0.0,
                        weight_id=f"L{l}_WV", weight_size=dim * kv_dim * dtype_bytes,
                        attrs=dict(base_attr), allowed=get_op_allowed("V")))

    # Attention core
    nid_QK = f"L{l}_QK"; nid_SO = f"L{l}_Softmax"; nid_SV = f"L{l}_SV"
    g.add_node(TaskNode(nid_QK, "QK", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("QK")))
    g.add_node(TaskNode(nid_SO, "Softmax", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Softmax")))
    g.add_node(TaskNode(nid_SV, "SV", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("SV")))

    # O
    nid_O = f"L{l}_O"
    g.add_node(TaskNode(nid_O, "O", flops=0.0,
                        weight_id=f"L{l}_WO", weight_size=o_in_dim * dim * dtype_bytes,
                        attrs=dict(base_attr), allowed=get_op_allowed("O")))

    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    # MLP: SwiGLU (W1,W3)->Act->W2
    nid_LN2 = f"L{l}_LN2"; nid_W1=f"L{l}_FFN_W1"; nid_W3=f"L{l}_FFN_W3"; nid_ACT=f"L{l}_Act"; nid_W2=f"L{l}_FFN_W2"; nid_Add2=f"L{l}_Add2"
    g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    g.add_node(TaskNode(nid_W1, "FFN_W1", flops=0.0,
                        weight_id=f"L{l}_W1", weight_size=dim * ffn * dtype_bytes,
                        attrs=dict(base_attr), allowed=get_op_allowed("FFN_W1")))
    g.add_node(TaskNode(nid_W3, "FFN_W3", flops=0.0,
                        weight_id=f"L{l}_W3", weight_size=dim * ffn * dtype_bytes,
                        attrs=dict(base_attr), allowed=get_op_allowed("FFN_W3")))
    g.add_node(TaskNode(nid_ACT, "SwiGLU", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("SwiGLU")))
    g.add_node(TaskNode(nid_W2, "FFN_W2", flops=0.0,
                        weight_id=f"L{l}_W2", weight_size=ffn * dim * dtype_bytes,
                        attrs=dict(base_attr), allowed=get_op_allowed("FFN_W2")))
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    # KV explicit ops (used during decode, no cost in prefill)
    nid_KW = f"L{l}_K_write"; nid_VW = f"L{l}_V_write"
    g.add_node(TaskNode(nid_KW, "K_write", attrs=dict(base_attr), allowed=get_op_allowed("K_write")))
    g.add_node(TaskNode(nid_VW, "V_write", attrs=dict(base_attr), allowed=get_op_allowed("V_write")))

    # Wire connections (pre-norm, sequential)
    x_in = f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X = f"L{l}_X"; g.add_node(TaskNode(nid_X, "Identity", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Identity")))
        g.add_edge(x_in, nid_X); g.add_edge(nid_X, nid_LN1)

    g.add_edge(nid_LN1, nid_Q); g.add_edge(nid_LN1, nid_K); g.add_edge(nid_LN1, nid_V)
    g.add_edge(nid_Q, nid_QK); g.add_edge(nid_K, nid_QK)
    g.add_edge(nid_QK, nid_SO); g.add_edge(nid_SO, nid_SV)
    g.add_edge(nid_V, nid_SV);
    g.add_edge(nid_SV, nid_O)
    g.add_edge(nid_K, nid_KW); g.add_edge(nid_V, nid_VW)
    g.add_edge(nid_O, nid_Add1);
    if x_in: g.add_edge(x_in, nid_Add1)

    g.add_edge(nid_Add1, nid_LN2); g.add_edge(nid_LN2, nid_W1); g.add_edge(nid_LN2, nid_W3)
    g.add_edge(nid_W1, nid_ACT); g.add_edge(nid_W3, nid_ACT)
    g.add_edge(nid_ACT, nid_W2); g.add_edge(nid_W2, nid_Add2); g.add_edge(nid_Add1, nid_Add2)


def add_mpt_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    # Similar to LLaMA but MLP uses GELU (no W3 gate by default)
    b = shape.batch
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim
    q_dim, kv_dim, o_in_dim = qh * hd, kvh * hd, qh * hd
    base_attr = {
        "layer": l,
        "q_heads": qh,
        "kv_heads": kvh,
        "head_dim": hd,
        "dim": dim,
        "ffn_dim": ffn,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "o_dim": o_in_dim,
        "batch": b,
    }

    nid_LN1=f"L{l}_LN1"; g.add_node(TaskNode(nid_LN1,"LN",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("LN")))
    if l>0:
        prev_kw=f"L{l-1}_K_write"; prev_vw=f"L{l-1}_V_write"
        g.add_edge(prev_kw, nid_LN1)
        g.add_edge(prev_vw, nid_LN1)
    nid_Q=f"L{l}_Q"; nid_K=f"L{l}_K"; nid_V=f"L{l}_V"
    g.add_node(TaskNode(nid_Q,"Q",flops=0.0,weight_id=f"L{l}_WQ",weight_size=dim*q_dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("Q")))
    g.add_node(TaskNode(nid_K,"K",flops=0.0,weight_id=f"L{l}_WK",weight_size=dim*kv_dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("K")))
    g.add_node(TaskNode(nid_V,"V",flops=0.0,weight_id=f"L{l}_WV",weight_size=dim*kv_dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("V")))

    nid_QK=f"L{l}_QK"; nid_SO=f"L{l}_Softmax"; nid_SV=f"L{l}_SV"
    g.add_node(TaskNode(nid_QK,"QK",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("QK")))
    g.add_node(TaskNode(nid_SO,"Softmax",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Softmax")))
    g.add_node(TaskNode(nid_SV,"SV",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("SV")))

    nid_O=f"L{l}_O"; g.add_node(TaskNode(nid_O,"O",flops=0.0,weight_id=f"L{l}_WO",weight_size=o_in_dim*dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("O")))
    nid_Add1=f"L{l}_Add1"; g.add_node(TaskNode(nid_Add1,"Add",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Add")))

    nid_LN2=f"L{l}_LN2"; nid_W1=f"L{l}_FFN_W1"; nid_G=f"L{l}_GELU"; nid_W2=f"L{l}_FFN_W2"; nid_Add2=f"L{l}_Add2"
    g.add_node(TaskNode(nid_LN2,"LN",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("LN")))
    g.add_node(TaskNode(nid_W1,"FFN_W1",flops=0.0,weight_id=f"L{l}_W1",weight_size=dim*ffn*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("FFN_W1")))
    g.add_node(TaskNode(nid_G,"GELU",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("GELU")))
    g.add_node(TaskNode(nid_W2,"FFN_W2",flops=0.0,weight_id=f"L{l}_W2",weight_size=ffn*dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("FFN_W2")))
    g.add_node(TaskNode(nid_Add2,"Add",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Add")))

    nid_KW=f"L{l}_K_write"; nid_VW=f"L{l}_V_write"
    g.add_node(TaskNode(nid_KW,"K_write",attrs=dict(base_attr),allowed=get_op_allowed("K_write")))
    g.add_node(TaskNode(nid_VW,"V_write",attrs=dict(base_attr),allowed=get_op_allowed("V_write")))

    x_in=f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"; g.add_node(TaskNode(nid_X,"Identity",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN1)

    g.add_edge(nid_LN1,nid_Q); g.add_edge(nid_LN1,nid_K); g.add_edge(nid_LN1,nid_V)
    g.add_edge(nid_Q,nid_QK); g.add_edge(nid_K,nid_QK)
    g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV)
    g.add_edge(nid_V, nid_SV);
    g.add_edge(nid_SV,nid_O)
    g.add_edge(nid_K,nid_KW); g.add_edge(nid_V,nid_VW)
    g.add_edge(nid_O,nid_Add1);
    if x_in: g.add_edge(x_in,nid_Add1)

    g.add_edge(nid_Add1,nid_LN2); g.add_edge(nid_LN2,nid_W1); g.add_edge(nid_LN2,nid_G); g.add_edge(nid_G,nid_W2); g.add_edge(nid_W2,nid_Add2); g.add_edge(nid_Add1,nid_Add2)


def add_palm_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    # PaLM uses pre-LN and PARALLEL residual: x + Attn(LN(x)) + MLP(LN(x))
    b = shape.batch
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim
    q_dim, kv_dim, o_in_dim = qh * hd, kvh * hd, qh * hd
    base_attr = {
        "layer": l,
        "q_heads": qh,
        "kv_heads": kvh,
        "head_dim": hd,
        "dim": dim,
        "ffn_dim": ffn,
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "o_dim": o_in_dim,
        "batch": b,
    }

    nid_LN = f"L{l}_LN"  # one LN feeding both branches
    g.add_node(TaskNode(nid_LN,"LN",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("LN")))
    if l>0:
        prev_kw=f"L{l-1}_K_write"; prev_vw=f"L{l-1}_V_write"
        g.add_edge(prev_kw, nid_LN)
        g.add_edge(prev_vw, nid_LN)

    # Attn branch
    nid_Q=f"L{l}_Q"; nid_K=f"L{l}_K"; nid_V=f"L{l}_V"
    g.add_node(TaskNode(nid_Q,"Q",flops=0.0,weight_id=f"L{l}_WQ",weight_size=dim*q_dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("Q")))
    g.add_node(TaskNode(nid_K,"K",flops=0.0,weight_id=f"L{l}_WK",weight_size=dim*kv_dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("K")))
    g.add_node(TaskNode(nid_V,"V",flops=0.0,weight_id=f"L{l}_WV",weight_size=dim*kv_dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("V")))

    nid_QK=f"L{l}_QK"; nid_SO=f"L{l}_Softmax"; nid_SV=f"L{l}_SV"
    g.add_node(TaskNode(nid_QK,"QK",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("QK")))
    g.add_node(TaskNode(nid_SO,"Softmax",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Softmax")))
    g.add_node(TaskNode(nid_SV,"SV",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("SV")))

    nid_O=f"L{l}_O"
    g.add_node(TaskNode(nid_O,"O",flops=0.0,weight_id=f"L{l}_WO",weight_size=o_in_dim*dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("O")))

    # MLP branch (GELU)
    nid_W1=f"L{l}_FFN_W1"; nid_G=f"L{l}_GELU"; nid_W2=f"L{l}_FFN_W2"
    g.add_node(TaskNode(nid_W1,"FFN_W1",flops=0.0,weight_id=f"L{l}_W1",weight_size=dim*ffn*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("FFN_W1")))
    g.add_node(TaskNode(nid_G,"GELU",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("GELU")))
    g.add_node(TaskNode(nid_W2,"FFN_W2",flops=0.0,weight_id=f"L{l}_W2",weight_size=ffn*dim*dtype_bytes,attrs=dict(base_attr),allowed=get_op_allowed("FFN_W2")))

    # Merge: X + Attn + MLP
    nid_Add2=f"L{l}_Add2"; g.add_node(TaskNode(nid_Add2,"Add",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Add")))

    # KV ops on decode
    nid_KW=f"L{l}_K_write"; nid_VW=f"L{l}_V_write"
    g.add_node(TaskNode(nid_KW,"K_write",attrs=dict(base_attr),allowed=get_op_allowed("K_write")))
    g.add_node(TaskNode(nid_VW,"V_write",attrs=dict(base_attr),allowed=get_op_allowed("V_write")))

    # Wire
    x_in=f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"; g.add_node(TaskNode(nid_X,"Identity",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN)

    g.add_edge(nid_LN,nid_Q); g.add_edge(nid_LN,nid_K); g.add_edge(nid_LN,nid_V)
    g.add_edge(nid_Q,nid_QK); g.add_edge(nid_K,nid_QK)
    g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV)
    g.add_edge(nid_SV,nid_O);g.add_edge(nid_V,nid_SV)
    g.add_edge(nid_K,nid_KW); g.add_edge(nid_V,nid_VW)
    # MLP branch
    g.add_edge(nid_LN,nid_W1); g.add_edge(nid_W1,nid_G); g.add_edge(nid_G,nid_W2)
    # Merge both outputs plus residual X
    if x_in: g.add_edge(x_in,nid_Add2)
    g.add_edge(nid_O,nid_Add2); g.add_edge(nid_W2,nid_Add2)



# =========================
# Head-parallel blocks (NEW)
# =========================

def add_llama_block_split_by_heads(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """
    Split Q/K/V/O + attention + FFN inside a layer by head_num shards.

    Notes:
    - QK/Softmax/SV/O are shard-parallel (each shard is a group of heads).
    - O and FFN_W2 produce partial contributions to a full-dim output; we model a merge via an Identity node.
    - Keep original per-layer K_write/V_write (single node per layer) for KV-cache side effects.
    """
    b = shape.batch
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim

    q_shards = _effective_shards(shape, for_q=True)
    kv_shards = _effective_shards(shape, for_kv=True)
    ffn_shards = _effective_shards(shape, for_ffn=True)

    q_ranges = _partition_ranges(qh, q_shards)
    kv_ranges = _partition_ranges(kvh, kv_shards)
    kv_head_to_shard = _make_head_to_shard_map(kv_ranges)

    base_attr_full = {
        "layer": l,
        "q_heads": qh,
        "kv_heads": kvh,
        "head_dim": hd,
        "dim": dim,
        "ffn_dim": ffn,
        "q_dim": qh * hd,
        "kv_dim": kvh * hd,
        "o_dim": qh * hd,
        "batch": b,
    }

    # LN1
    nid_LN1 = f"L{l}_LN1"
    g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr_full), allowed=get_op_allowed("LN")))

    # Carry K/V cache writes to next layer input (keep original behavior)
    if l > 0:
        prev_kw = f"L{l-1}_K_write"
        prev_vw = f"L{l-1}_V_write"
        g.add_edge(prev_kw, nid_LN1)
        g.add_edge(prev_vw, nid_LN1)

    # Residual carry-in
    x_in = f"L{l-1}_Add2" if l > 0 else None
    if x_in:
        nid_X = f"L{l}_X"
        g.add_node(TaskNode(nid_X, "Identity", flops=0.0, attrs=dict(base_attr_full), allowed=get_op_allowed("Identity")))
        g.add_edge(x_in, nid_X)
        g.add_edge(nid_X, nid_LN1)

    # ---- K/V shards (by kv-head groups) ----
    k_nodes: Dict[int, str] = {}
    v_nodes: Dict[int, str] = {}
    for sid, (kv_st, kv_cnt) in enumerate(kv_ranges):
        kv_dim_s = kv_cnt * hd
        attr_kv = dict(base_attr_full)
        attr_kv.update({"kv_heads": kv_cnt, "kv_dim": kv_dim_s})

        nid_K = f"L{l}_K_S{sid}"
        nid_V = f"L{l}_V_S{sid}"
        g.add_node(TaskNode(
            nid_K, "K", flops=0.0,
            weight_id=f"L{l}_WK_S{sid}",
            weight_size=dim * kv_dim_s * dtype_bytes,
            attrs=dict(attr_kv),
            allowed=get_op_allowed("K"),
        ))
        g.add_node(TaskNode(
            nid_V, "V", flops=0.0,
            weight_id=f"L{l}_WV_S{sid}",
            weight_size=dim * kv_dim_s * dtype_bytes,
            attrs=dict(attr_kv),
            allowed=get_op_allowed("V"),
        ))

        g.add_edge(nid_LN1, nid_K)
        g.add_edge(nid_LN1, nid_V)
        k_nodes[sid] = nid_K
        v_nodes[sid] = nid_V

    # KV writes (single per layer)
    nid_KW = f"L{l}_K_write"; nid_VW = f"L{l}_V_write"
    g.add_node(TaskNode(nid_KW, "K_write", attrs=dict(base_attr_full), allowed=get_op_allowed("K_write")))
    g.add_node(TaskNode(nid_VW, "V_write", attrs=dict(base_attr_full), allowed=get_op_allowed("V_write")))
    for sid in k_nodes:
        g.add_edge(k_nodes[sid], nid_KW)
    for sid in v_nodes:
        g.add_edge(v_nodes[sid], nid_VW)

    # ---- Q + attention shards (by q-head groups) ----
    o_nodes: List[str] = []
    for sid, (q_st, q_cnt) in enumerate(q_ranges):
        q_dim_s = q_cnt * hd
        attr_q = dict(base_attr_full)
        attr_q.update({"q_heads": q_cnt, "q_dim": q_dim_s, "o_dim": q_dim_s})

        nid_Q = f"L{l}_Q_S{sid}"
        g.add_node(TaskNode(
            nid_Q, "Q", flops=0.0,
            weight_id=f"L{l}_WQ_S{sid}",
            weight_size=dim * q_dim_s * dtype_bytes,
            attrs=dict(attr_q),
            allowed=get_op_allowed("Q"),
        ))
        g.add_edge(nid_LN1, nid_Q)

        nid_QK = f"L{l}_QK_S{sid}"; nid_SO = f"L{l}_Softmax_S{sid}"; nid_SV = f"L{l}_SV_S{sid}"
        g.add_node(TaskNode(nid_QK, "QK", flops=0.0, attrs=dict(attr_q), allowed=get_op_allowed("QK")))
        g.add_node(TaskNode(nid_SO, "Softmax", flops=0.0, attrs=dict(attr_q), allowed=get_op_allowed("Softmax")))
        g.add_node(TaskNode(nid_SV, "SV", flops=0.0, attrs=dict(attr_q), allowed=get_op_allowed("SV")))

        g.add_edge(nid_Q, nid_QK)

        # Connect required K/V shards for this q-range (GQA/MQA)
        kv_need = _kv_heads_for_q_range(q_st, q_cnt, q_heads=qh, kv_heads=kvh)
        kv_shard_need: Set[int] = set()
        for kh in kv_need:
            kv_shard_need.add(int(kv_head_to_shard.get(int(kh), 0)))
        for kv_sid in sorted(kv_shard_need):
            g.add_edge(k_nodes[kv_sid], nid_QK)
            g.add_edge(v_nodes[kv_sid], nid_SV)

        g.add_edge(nid_QK, nid_SO)
        g.add_edge(nid_SO, nid_SV)

        nid_O = f"L{l}_O_S{sid}"
        g.add_node(TaskNode(
            nid_O, "O", flops=0.0,
            weight_id=f"L{l}_WO_S{sid}",
            weight_size=q_dim_s * dim * dtype_bytes,
            attrs=dict(attr_q),
            allowed=get_op_allowed("O"),
        ))
        g.add_edge(nid_SV, nid_O)
        o_nodes.append(nid_O)

    # Merge O shards -> full attention output
    nid_Omerge = f"L{l}_O_merge"
    g.add_node(TaskNode(nid_Omerge, "Identity", flops=0.0, attrs=dict(base_attr_full), allowed=get_op_allowed("Identity")))
    for nid_O in o_nodes:
        g.add_edge(nid_O, nid_Omerge)

    # Residual Add1 after attention
    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr_full), allowed=get_op_allowed("Add")))
    g.add_edge(nid_Omerge, nid_Add1)
    if x_in:
        g.add_edge(x_in, nid_Add1)

    # LN2
    nid_LN2 = f"L{l}_LN2"
    g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr_full), allowed=get_op_allowed("LN")))
    g.add_edge(nid_Add1, nid_LN2)

    # FFN shards: SwiGLU (W1,W3)->Act->W2
    ffn_sizes = _partition_sizes(ffn, ffn_shards)
    w2_nodes: List[str] = []
    for sid, ffn_cnt in enumerate(ffn_sizes):
        if int(ffn_cnt) <= 0:
            continue
        attr_ffn = dict(base_attr_full)
        attr_ffn.update({"ffn_dim": int(ffn_cnt)})

        nid_W1 = f"L{l}_FFN_W1_S{sid}"
        nid_W3 = f"L{l}_FFN_W3_S{sid}"
        nid_ACT = f"L{l}_Act_S{sid}"
        nid_W2 = f"L{l}_FFN_W2_S{sid}"

        g.add_node(TaskNode(nid_W1, "FFN_W1", flops=0.0,
                            weight_id=f"L{l}_W1_S{sid}", weight_size=dim * int(ffn_cnt) * dtype_bytes,
                            attrs=dict(attr_ffn), allowed=get_op_allowed("FFN_W1")))
        g.add_node(TaskNode(nid_W3, "FFN_W3", flops=0.0,
                            weight_id=f"L{l}_W3_S{sid}", weight_size=dim * int(ffn_cnt) * dtype_bytes,
                            attrs=dict(attr_ffn), allowed=get_op_allowed("FFN_W3")))
        g.add_node(TaskNode(nid_ACT, "SwiGLU", flops=0.0, attrs=dict(attr_ffn), allowed=get_op_allowed("SwiGLU")))
        g.add_node(TaskNode(nid_W2, "FFN_W2", flops=0.0,
                            weight_id=f"L{l}_W2_S{sid}", weight_size=int(ffn_cnt) * dim * dtype_bytes,
                            attrs=dict(attr_ffn), allowed=get_op_allowed("FFN_W2")))

        g.add_edge(nid_LN2, nid_W1)
        g.add_edge(nid_LN2, nid_W3)
        g.add_edge(nid_W1, nid_ACT)
        g.add_edge(nid_W3, nid_ACT)
        g.add_edge(nid_ACT, nid_W2)
        w2_nodes.append(nid_W2)

    nid_FFNmerge = f"L{l}_FFN_merge"
    g.add_node(TaskNode(nid_FFNmerge, "Identity", flops=0.0, attrs=dict(base_attr_full), allowed=get_op_allowed("Identity")))
    for nid_W2 in w2_nodes:
        g.add_edge(nid_W2, nid_FFNmerge)

    # Final residual Add2
    nid_Add2 = f"L{l}_Add2"
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr_full), allowed=get_op_allowed("Add")))
    g.add_edge(nid_Add1, nid_Add2)
    g.add_edge(nid_FFNmerge, nid_Add2)


def add_mpt_block_split_by_heads(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """
    MPT: split Q/K/V/O + attention by head shards, and split FFN by ffn_dim shards.
    """
    b = shape.batch
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim

    q_shards = _effective_shards(shape, for_q=True)
    kv_shards = _effective_shards(shape, for_kv=True)
    ffn_shards = _effective_shards(shape, for_ffn=True)

    q_ranges = _partition_ranges(qh, q_shards)
    kv_ranges = _partition_ranges(kvh, kv_shards)
    kv_head_to_shard = _make_head_to_shard_map(kv_ranges)

    base_attr_full = {
        "layer": l,
        "q_heads": qh,
        "kv_heads": kvh,
        "head_dim": hd,
        "dim": dim,
        "ffn_dim": ffn,
        "q_dim": qh * hd,
        "kv_dim": kvh * hd,
        "o_dim": qh * hd,
        "batch": b,
    }

    # LN1
    nid_LN1 = f"L{l}_LN1"
    g.add_node(TaskNode(nid_LN1,"LN",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("LN")))
    if l>0:
        g.add_edge(f"L{l-1}_K_write", nid_LN1)
        g.add_edge(f"L{l-1}_V_write", nid_LN1)

    x_in = f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"
        g.add_node(TaskNode(nid_X,"Identity",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN1)

    # K/V shards
    k_nodes: Dict[int, str] = {}
    v_nodes: Dict[int, str] = {}
    for sid, (kv_st, kv_cnt) in enumerate(kv_ranges):
        kv_dim_s = kv_cnt * hd
        attr_kv = dict(base_attr_full); attr_kv.update({"kv_heads": kv_cnt, "kv_dim": kv_dim_s})
        nid_K = f"L{l}_K_S{sid}"
        nid_V = f"L{l}_V_S{sid}"
        g.add_node(TaskNode(nid_K,"K",flops=0.0,weight_id=f"L{l}_WK_S{sid}",weight_size=dim*kv_dim_s*dtype_bytes,attrs=dict(attr_kv),allowed=get_op_allowed("K")))
        g.add_node(TaskNode(nid_V,"V",flops=0.0,weight_id=f"L{l}_WV_S{sid}",weight_size=dim*kv_dim_s*dtype_bytes,attrs=dict(attr_kv),allowed=get_op_allowed("V")))
        g.add_edge(nid_LN1, nid_K); g.add_edge(nid_LN1, nid_V)
        k_nodes[sid]=nid_K; v_nodes[sid]=nid_V

    nid_KW=f"L{l}_K_write"; nid_VW=f"L{l}_V_write"
    g.add_node(TaskNode(nid_KW,"K_write",attrs=dict(base_attr_full),allowed=get_op_allowed("K_write")))
    g.add_node(TaskNode(nid_VW,"V_write",attrs=dict(base_attr_full),allowed=get_op_allowed("V_write")))
    for sid in k_nodes: g.add_edge(k_nodes[sid], nid_KW)
    for sid in v_nodes: g.add_edge(v_nodes[sid], nid_VW)

    # Attention shards
    o_nodes: List[str] = []
    for sid, (q_st, q_cnt) in enumerate(q_ranges):
        q_dim_s = q_cnt * hd
        attr_q = dict(base_attr_full); attr_q.update({"q_heads": q_cnt, "q_dim": q_dim_s, "o_dim": q_dim_s})

        nid_Q = f"L{l}_Q_S{sid}"
        g.add_node(TaskNode(nid_Q,"Q",flops=0.0,weight_id=f"L{l}_WQ_S{sid}",weight_size=dim*q_dim_s*dtype_bytes,attrs=dict(attr_q),allowed=get_op_allowed("Q")))
        g.add_edge(nid_LN1, nid_Q)

        nid_QK=f"L{l}_QK_S{sid}"; nid_SO=f"L{l}_Softmax_S{sid}"; nid_SV=f"L{l}_SV_S{sid}"
        g.add_node(TaskNode(nid_QK,"QK",flops=0.0,attrs=dict(attr_q),allowed=get_op_allowed("QK")))
        g.add_node(TaskNode(nid_SO,"Softmax",flops=0.0,attrs=dict(attr_q),allowed=get_op_allowed("Softmax")))
        g.add_node(TaskNode(nid_SV,"SV",flops=0.0,attrs=dict(attr_q),allowed=get_op_allowed("SV")))
        g.add_edge(nid_Q, nid_QK)

        kv_need = _kv_heads_for_q_range(q_st, q_cnt, q_heads=qh, kv_heads=kvh)
        kv_shard_need: Set[int] = set(int(kv_head_to_shard.get(int(kh), 0)) for kh in kv_need)
        for kv_sid in sorted(kv_shard_need):
            g.add_edge(k_nodes[kv_sid], nid_QK)
            g.add_edge(v_nodes[kv_sid], nid_SV)

        g.add_edge(nid_QK, nid_SO); g.add_edge(nid_SO, nid_SV)

        nid_O=f"L{l}_O_S{sid}"
        g.add_node(TaskNode(nid_O,"O",flops=0.0,weight_id=f"L{l}_WO_S{sid}",weight_size=q_dim_s*dim*dtype_bytes,attrs=dict(attr_q),allowed=get_op_allowed("O")))
        g.add_edge(nid_SV, nid_O)
        o_nodes.append(nid_O)

    nid_Omerge=f"L{l}_O_merge"
    g.add_node(TaskNode(nid_Omerge,"Identity",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Identity")))
    for n in o_nodes: g.add_edge(n, nid_Omerge)

    nid_Add1=f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1,"Add",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Add")))
    g.add_edge(nid_Omerge, nid_Add1)
    if x_in: g.add_edge(x_in, nid_Add1)

    # LN2
    nid_LN2=f"L{l}_LN2"
    g.add_node(TaskNode(nid_LN2,"LN",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("LN")))
    g.add_edge(nid_Add1, nid_LN2)

    # FFN shards: W1 -> GELU -> W2
    ffn_sizes = _partition_sizes(ffn, ffn_shards)
    w2_nodes: List[str] = []
    for sid, ffn_cnt in enumerate(ffn_sizes):
        if int(ffn_cnt) <= 0:
            continue
        attr_ffn = dict(base_attr_full); attr_ffn.update({"ffn_dim": int(ffn_cnt)})
        nid_W1=f"L{l}_FFN_W1_S{sid}"; nid_G=f"L{l}_GELU_S{sid}"; nid_W2=f"L{l}_FFN_W2_S{sid}"
        g.add_node(TaskNode(nid_W1,"FFN_W1",flops=0.0,weight_id=f"L{l}_W1_S{sid}",weight_size=dim*int(ffn_cnt)*dtype_bytes,attrs=dict(attr_ffn),allowed=get_op_allowed("FFN_W1")))
        g.add_node(TaskNode(nid_G,"GELU",flops=0.0,attrs=dict(attr_ffn),allowed=get_op_allowed("GELU")))
        g.add_node(TaskNode(nid_W2,"FFN_W2",flops=0.0,weight_id=f"L{l}_W2_S{sid}",weight_size=int(ffn_cnt)*dim*dtype_bytes,attrs=dict(attr_ffn),allowed=get_op_allowed("FFN_W2")))
        g.add_edge(nid_LN2, nid_W1); g.add_edge(nid_W1, nid_G); g.add_edge(nid_G, nid_W2)
        w2_nodes.append(nid_W2)

    nid_FFNmerge=f"L{l}_FFN_merge"
    g.add_node(TaskNode(nid_FFNmerge,"Identity",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Identity")))
    for n in w2_nodes: g.add_edge(n, nid_FFNmerge)

    nid_Add2=f"L{l}_Add2"
    g.add_node(TaskNode(nid_Add2,"Add",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Add")))
    g.add_edge(nid_Add1, nid_Add2); g.add_edge(nid_FFNmerge, nid_Add2)


def add_palm_block_split_by_heads(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """
    PaLM: pre-LN, then parallel residual: x + Attn(LN(x)) + FFN(LN(x)).
    We shard Attn by q-heads and shard FFN by ffn_dim, then merge into Add2.
    """
    b = shape.batch
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim

    q_shards = _effective_shards(shape, for_q=True)
    kv_shards = _effective_shards(shape, for_kv=True)
    ffn_shards = _effective_shards(shape, for_ffn=True)

    q_ranges = _partition_ranges(qh, q_shards)
    kv_ranges = _partition_ranges(kvh, kv_shards)
    kv_head_to_shard = _make_head_to_shard_map(kv_ranges)

    base_attr_full = {
        "layer": l,
        "q_heads": qh,
        "kv_heads": kvh,
        "head_dim": hd,
        "dim": dim,
        "ffn_dim": ffn,
        "q_dim": qh * hd,
        "kv_dim": kvh * hd,
        "o_dim": qh * hd,
        "batch": b,
    }

    nid_LN = f"L{l}_LN"
    g.add_node(TaskNode(nid_LN,"LN",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("LN")))
    if l>0:
        g.add_edge(f"L{l-1}_K_write", nid_LN)
        g.add_edge(f"L{l-1}_V_write", nid_LN)

    x_in = f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"
        g.add_node(TaskNode(nid_X,"Identity",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN)

    # K/V shards
    k_nodes: Dict[int, str] = {}
    v_nodes: Dict[int, str] = {}
    for sid, (kv_st, kv_cnt) in enumerate(kv_ranges):
        kv_dim_s = kv_cnt * hd
        attr_kv = dict(base_attr_full); attr_kv.update({"kv_heads": kv_cnt, "kv_dim": kv_dim_s})
        nid_K=f"L{l}_K_S{sid}"; nid_V=f"L{l}_V_S{sid}"
        g.add_node(TaskNode(nid_K,"K",flops=0.0,weight_id=f"L{l}_WK_S{sid}",weight_size=dim*kv_dim_s*dtype_bytes,attrs=dict(attr_kv),allowed=get_op_allowed("K")))
        g.add_node(TaskNode(nid_V,"V",flops=0.0,weight_id=f"L{l}_WV_S{sid}",weight_size=dim*kv_dim_s*dtype_bytes,attrs=dict(attr_kv),allowed=get_op_allowed("V")))
        g.add_edge(nid_LN, nid_K); g.add_edge(nid_LN, nid_V)
        k_nodes[sid]=nid_K; v_nodes[sid]=nid_V

    nid_KW=f"L{l}_K_write"; nid_VW=f"L{l}_V_write"
    g.add_node(TaskNode(nid_KW,"K_write",attrs=dict(base_attr_full),allowed=get_op_allowed("K_write")))
    g.add_node(TaskNode(nid_VW,"V_write",attrs=dict(base_attr_full),allowed=get_op_allowed("V_write")))
    for sid in k_nodes: g.add_edge(k_nodes[sid], nid_KW)
    for sid in v_nodes: g.add_edge(v_nodes[sid], nid_VW)

    # Attention shards
    o_nodes: List[str] = []
    for sid, (q_st, q_cnt) in enumerate(q_ranges):
        q_dim_s = q_cnt * hd
        attr_q = dict(base_attr_full); attr_q.update({"q_heads": q_cnt, "q_dim": q_dim_s, "o_dim": q_dim_s})

        nid_Q=f"L{l}_Q_S{sid}"
        g.add_node(TaskNode(nid_Q,"Q",flops=0.0,weight_id=f"L{l}_WQ_S{sid}",weight_size=dim*q_dim_s*dtype_bytes,attrs=dict(attr_q),allowed=get_op_allowed("Q")))
        g.add_edge(nid_LN, nid_Q)

        nid_QK=f"L{l}_QK_S{sid}"; nid_SO=f"L{l}_Softmax_S{sid}"; nid_SV=f"L{l}_SV_S{sid}"
        g.add_node(TaskNode(nid_QK,"QK",flops=0.0,attrs=dict(attr_q),allowed=get_op_allowed("QK")))
        g.add_node(TaskNode(nid_SO,"Softmax",flops=0.0,attrs=dict(attr_q),allowed=get_op_allowed("Softmax")))
        g.add_node(TaskNode(nid_SV,"SV",flops=0.0,attrs=dict(attr_q),allowed=get_op_allowed("SV")))
        g.add_edge(nid_Q, nid_QK)

        kv_need = _kv_heads_for_q_range(q_st, q_cnt, q_heads=qh, kv_heads=kvh)
        kv_shard_need: Set[int] = set(int(kv_head_to_shard.get(int(kh), 0)) for kh in kv_need)
        for kv_sid in sorted(kv_shard_need):
            g.add_edge(k_nodes[kv_sid], nid_QK)
            g.add_edge(v_nodes[kv_sid], nid_SV)

        g.add_edge(nid_QK, nid_SO); g.add_edge(nid_SO, nid_SV)

        nid_O=f"L{l}_O_S{sid}"
        g.add_node(TaskNode(nid_O,"O",flops=0.0,weight_id=f"L{l}_WO_S{sid}",weight_size=q_dim_s*dim*dtype_bytes,attrs=dict(attr_q),allowed=get_op_allowed("O")))
        g.add_edge(nid_SV, nid_O)
        o_nodes.append(nid_O)

    nid_Omerge=f"L{l}_O_merge"
    g.add_node(TaskNode(nid_Omerge,"Identity",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Identity")))
    for n in o_nodes: g.add_edge(n, nid_Omerge)

    # FFN shards: W1 -> GELU -> W2
    ffn_sizes = _partition_sizes(ffn, ffn_shards)
    w2_nodes: List[str] = []
    for sid, ffn_cnt in enumerate(ffn_sizes):
        if int(ffn_cnt) <= 0:
            continue
        attr_ffn = dict(base_attr_full); attr_ffn.update({"ffn_dim": int(ffn_cnt)})
        nid_W1=f"L{l}_FFN_W1_S{sid}"; nid_G=f"L{l}_GELU_S{sid}"; nid_W2=f"L{l}_FFN_W2_S{sid}"
        g.add_node(TaskNode(nid_W1,"FFN_W1",flops=0.0,weight_id=f"L{l}_W1_S{sid}",weight_size=dim*int(ffn_cnt)*dtype_bytes,attrs=dict(attr_ffn),allowed=get_op_allowed("FFN_W1")))
        g.add_node(TaskNode(nid_G,"GELU",flops=0.0,attrs=dict(attr_ffn),allowed=get_op_allowed("GELU")))
        g.add_node(TaskNode(nid_W2,"FFN_W2",flops=0.0,weight_id=f"L{l}_W2_S{sid}",weight_size=int(ffn_cnt)*dim*dtype_bytes,attrs=dict(attr_ffn),allowed=get_op_allowed("FFN_W2")))
        g.add_edge(nid_LN, nid_W1); g.add_edge(nid_W1, nid_G); g.add_edge(nid_G, nid_W2)
        w2_nodes.append(nid_W2)

    nid_FFNmerge=f"L{l}_FFN_merge"
    g.add_node(TaskNode(nid_FFNmerge,"Identity",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Identity")))
    for n in w2_nodes: g.add_edge(n, nid_FFNmerge)

    # Merge: X + Attn + MLP
    nid_Add2=f"L{l}_Add2"
    g.add_node(TaskNode(nid_Add2,"Add",flops=0.0,attrs=dict(base_attr_full),allowed=get_op_allowed("Add")))
    if x_in: g.add_edge(x_in, nid_Add2)
    g.add_edge(nid_Omerge, nid_Add2)
    g.add_edge(nid_FFNmerge, nid_Add2)


class LLaMADef:
    name = "llama"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        split_by = _normalize_split_by(getattr(shape, 'split_by', 'layer'))
        for l in range(shape.layer_num):
            if split_by == 'head':
                add_llama_block_split_by_heads(g, l, shape, dtype_bytes)
            else:
                add_llama_block(g, l, shape, dtype_bytes)
        return g

class MPTDef:
    name = "mpt"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        split_by = _normalize_split_by(getattr(shape, 'split_by', 'layer'))
        for l in range(shape.layer_num):
            if split_by == 'head':
                add_mpt_block_split_by_heads(g, l, shape, dtype_bytes)
            else:
                add_mpt_block(g, l, shape, dtype_bytes)
        return g

class PaLMDef:
    name = "palm"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        split_by = _normalize_split_by(getattr(shape, 'split_by', 'layer'))
        for l in range(shape.layer_num):
            if split_by == 'head':
                add_palm_block_split_by_heads(g, l, shape, dtype_bytes)
            else:
                add_palm_block(g, l, shape, dtype_bytes)
        return g

class QwenDef:
    name = "qwen"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        split_by = _normalize_split_by(getattr(shape, 'split_by', 'layer'))
        for l in range(shape.layer_num):
            if split_by == 'head':
                add_llama_block_split_by_heads(g, l, shape, dtype_bytes)
            else:
                add_llama_block(g, l, shape, dtype_bytes)
        return g

class MixtralDef:
    name = "mixtral"

    @staticmethod
    def _active_expert_count(total: int, top_k: int, imbalance: float) -> int:
        if total <= 0:
            return 0
        guard = max(1.0, float(imbalance or 1.0))
        baseline = max(1, top_k)
        return max(1, min(total, int(math.ceil(baseline * guard))))

    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        experts = int(getattr(shape, "experts_per_layer", 1))
        top_k = int(getattr(shape, "experts_top_k", 1))
        moe_imbalance = float(getattr(shape, "moe_imbalance_factor", 1.0))
        active_experts = int(getattr(shape, "active_experts_per_layer", 0) or 0)
        if active_experts <= 0 or active_experts > experts:
            active_experts = self._active_expert_count(experts, top_k, moe_imbalance)
        setattr(shape, "active_experts_per_layer", active_experts)
        setattr(shape, "moe_pruned_experts_per_layer", max(0, experts - active_experts))

        b = shape.batch
        dim, ffn = shape.dim, shape.ffn_dim
        qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim
        q_dim, kv_dim, o_in_dim = qh * hd, kvh * hd, qh * hd

        for l in range(shape.layer_num):
            base_attr = {
                "layer": l,
                "q_heads": qh,
                "kv_heads": kvh,
                "head_dim": hd,
                "dim": dim,
                "ffn_dim": ffn,
                "q_dim": q_dim,
                "kv_dim": kv_dim,
                "o_dim": o_in_dim,
                "batch": b,
                "experts": experts,
                "top_k": top_k,
                "moe_imbalance_factor": moe_imbalance,
            }

            # Attention (same as llama block)
            nid_LN1 = f"L{l}_LN1"
            g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            if l > 0:
                prev_kw = f"L{l-1}_K_write"
                prev_vw = f"L{l-1}_V_write"
                g.add_edge(prev_kw, nid_LN1)
                g.add_edge(prev_vw, nid_LN1)

            nid_Q = f"L{l}_Q"
            nid_K = f"L{l}_K"
            nid_V = f"L{l}_V"
            g.add_node(TaskNode(nid_Q, "Q", flops=0.0,
                                weight_id=f"L{l}_WQ", weight_size=dim * q_dim * dtype_bytes,
                                attrs=dict(base_attr), allowed=get_op_allowed("Q")))
            g.add_node(TaskNode(nid_K, "K", flops=0.0,
                                weight_id=f"L{l}_WK", weight_size=dim * kv_dim * dtype_bytes,
                                attrs=dict(base_attr), allowed=get_op_allowed("K")))
            g.add_node(TaskNode(nid_V, "V", flops=0.0,
                                weight_id=f"L{l}_WV", weight_size=dim * kv_dim * dtype_bytes,
                                attrs=dict(base_attr), allowed=get_op_allowed("V")))

            nid_QK = f"L{l}_QK"
            nid_SO = f"L{l}_Softmax"
            nid_SV = f"L{l}_SV"
            g.add_node(TaskNode(nid_QK, "QK", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("QK")))
            g.add_node(TaskNode(nid_SO, "Softmax", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Softmax")))
            g.add_node(TaskNode(nid_SV, "SV", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("SV")))

            nid_O = f"L{l}_O"
            g.add_node(TaskNode(nid_O, "O", flops=0.0,
                                weight_id=f"L{l}_WO", weight_size=o_in_dim * dim * dtype_bytes,
                                attrs=dict(base_attr), allowed=get_op_allowed("O")))

            nid_Add1 = f"L{l}_Add1"
            g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

            nid_KW = f"L{l}_K_write"
            nid_VW = f"L{l}_V_write"
            g.add_node(TaskNode(nid_KW, "K_write", attrs=dict(base_attr), allowed=get_op_allowed("K_write")))
            g.add_node(TaskNode(nid_VW, "V_write", attrs=dict(base_attr), allowed=get_op_allowed("V_write")))

            x_in = f"L{l-1}_Add2" if l > 0 else None
            if x_in:
                nid_X = f"L{l}_X"
                g.add_node(TaskNode(nid_X, "Identity", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Identity")))
                g.add_edge(x_in, nid_X)
                g.add_edge(nid_X, nid_LN1)

            g.add_edge(nid_LN1, nid_Q); g.add_edge(nid_LN1, nid_K); g.add_edge(nid_LN1, nid_V)
            g.add_edge(nid_Q, nid_QK);  g.add_edge(nid_K, nid_QK)
            g.add_edge(nid_QK, nid_SO); g.add_edge(nid_SO, nid_SV)
            g.add_edge(nid_V, nid_SV)
            g.add_edge(nid_SV, nid_O)
            g.add_edge(nid_K, nid_KW)
            g.add_edge(nid_V, nid_VW)
            g.add_edge(nid_O, nid_Add1)
            if x_in:
                g.add_edge(x_in, nid_Add1)

            # LN2
            nid_LN2 = f"L{l}_LN2"
            g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            g.add_edge(nid_Add1, nid_LN2)

            # Experts
            expert_outputs: List[str] = []
            for e in range(active_experts):
                expert_attr = {**base_attr, "expert": e, "experts": experts, "active_experts": active_experts,
                               "top_k": top_k, "moe_imbalance": moe_imbalance}
                nid_W1 = f"L{l}_FFN_W1_E{e}"
                nid_W3 = f"L{l}_FFN_W3_E{e}"
                nid_ACT = f"L{l}_Act_E{e}"
                nid_W2 = f"L{l}_FFN_W2_E{e}"
                g.add_node(TaskNode(nid_W1, "FFN_W1", flops=0.0,
                                    weight_id=f"L{l}_E{e}_W1",
                                    weight_size=dim * ffn * dtype_bytes,
                                    attrs=dict(expert_attr), allowed=get_op_allowed("FFN_W1")))
                g.add_node(TaskNode(nid_W3, "FFN_W3", flops=0.0,
                                    weight_id=f"L{l}_E{e}_W3",
                                    weight_size=dim * ffn * dtype_bytes,
                                    attrs=dict(expert_attr), allowed=get_op_allowed("FFN_W3")))
                g.add_node(TaskNode(nid_ACT, "SwiGLU", flops=0.0,
                                    attrs=dict(expert_attr), allowed=get_op_allowed("SwiGLU")))
                g.add_node(TaskNode(nid_W2, "FFN_W2", flops=0.0,
                                    weight_id=f"L{l}_E{e}_W2",
                                    weight_size=ffn * dim * dtype_bytes,
                                    attrs=dict(expert_attr), allowed=get_op_allowed("FFN_W2")))
                g.add_edge(nid_LN2, nid_W1)
                g.add_edge(nid_LN2, nid_W3)
                g.add_edge(nid_W1, nid_ACT)
                g.add_edge(nid_W3, nid_ACT)
                g.add_edge(nid_ACT, nid_W2)
                expert_outputs.append(nid_W2)

            # Router
            nid_router = f"L{l}_Router"
            g.add_node(TaskNode(
                nid_router, "MoE_Router", flops=0.0,
                attrs={**base_attr, "experts": experts, "active_experts": active_experts,
                       "top_k": top_k, "moe_imbalance": moe_imbalance}, allowed=get_op_allowed("MoE_Router"),
            ))
            g.add_edge(nid_LN2, nid_router)
            for out in expert_outputs:
                g.add_edge(out, nid_router)

            # Residual Add2
            nid_Add2 = f"L{l}_Add2"
            g.add_node(TaskNode(nid_Add2, "Add", flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Add")))
            g.add_edge(nid_router, nid_Add2)
            g.add_edge(nid_Add1, nid_Add2)

        return g
def make_model_def(family: str):
    f = family.lower()
    if f == "llama": return LLaMADef()
    if f == "mpt":   return MPTDef()
    if f == "palm":  return PaLMDef()
    if f == "mixtral": return MixtralDef()
    if f == "qwen": return QwenDef()
    raise ValueError(f"Unknown model family: {family}")
