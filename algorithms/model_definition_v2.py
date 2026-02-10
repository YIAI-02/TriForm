from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Set, Tuple

from task_graph import TaskGraph, TaskNode
from config import OPERATOR_DEVICE_ALLOWED


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


@dataclass
class ModelShape:
    layer_num: int
    dim: int
    ffn_dim: int
    n_heads: int         # Q heads
    n_kv_heads: int      # shared KV heads (GQA/MQA)
    batch: int
    max_seq_len: int     # Maximum sequence length (for graph structure)

    @property
    def head_dim(self) -> int:
        return int(self.dim // max(1, int(self.n_heads)))


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


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
    return OPERATOR_DEVICE_ALLOWED.get(key, {}).copy()


def _kv_head_for_q_head(q_idx: int, q_heads: int, kv_heads: int) -> int:
    """Map q-head index -> kv-head index for GQA/MQA."""
    q_heads = max(1, int(q_heads))
    kv_heads = max(1, int(kv_heads))
    q_idx = int(min(max(0, int(q_idx)), q_heads - 1))
    return int((q_idx * kv_heads) // q_heads)


# ---------------------------------------------------------------------------
# Attention head-splitting + KV-cache write sharding
# ---------------------------------------------------------------------------


def _add_attention_head_split_llama_style(
    g: TaskGraph,
    *,
    l: int,
    shape: ModelShape,
    dtype_bytes: int,
    base_attr: Dict,
    ln_nid: str,
    x_in: Optional[str],
    add1_nid: str,
    add_x_identity: bool,
):
    """Add LLaMA/Mixtral/Qwen style attention:

        LN -> (Q,K,V) -> per-head (QK->Softmax->SV) -> O -> Add1 (+res)
        Also emits per-KV-head (K_write, V_write) nodes.

    Q/K/V/O weights are NOT sharded.
    Attention compute is split by *query head* (one QK/Softmax/SV chain per head).
    KV writes are sharded by *KV head* to support PIM KV-cache chunk placement.
    """

    b = int(shape.batch)
    dim = int(shape.dim)
    qh = int(shape.n_heads)
    kvh = int(shape.n_kv_heads)
    hd = int(shape.head_dim)

    q_dim = int(qh * hd)
    kv_dim = int(kvh * hd)
    o_in_dim = int(qh * hd)

    # --- Q/K/V (unsplit) ---
    nid_Q = f"L{l}_Q"
    nid_K = f"L{l}_K"
    nid_V = f"L{l}_V"
    g.add_node(
        TaskNode(
            nid_Q,
            "Q",
            flops=0.0,
            weight_id=f"L{l}_WQ",
            weight_size=int(dim * q_dim * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("Q"),
        )
    )
    g.add_node(
        TaskNode(
            nid_K,
            "K",
            flops=0.0,
            weight_id=f"L{l}_WK",
            weight_size=int(dim * kv_dim * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("K"),
        )
    )
    g.add_node(
        TaskNode(
            nid_V,
            "V",
            flops=0.0,
            weight_id=f"L{l}_WV",
            weight_size=int(dim * kv_dim * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("V"),
        )
    )

    g.add_edge(ln_nid, nid_Q)
    g.add_edge(ln_nid, nid_K)
    g.add_edge(ln_nid, nid_V)

    # Optional: explicit residual identity node
    if add_x_identity and x_in is not None:
        nid_X = f"L{l}_X"
        g.add_node(TaskNode(nid_X, "Identity", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Identity")))
        g.add_edge(x_in, nid_X)
        g.add_edge(nid_X, ln_nid)

    # --- KV-head slice + KV-write sharding (for KV-cache placement) ---
    # We slice current-token K/V by KV head and write each shard independently.
    # Past KV is modeled via implicit KV-cache reads in QK/SV (scheduler/cost_model).
    k_slice: Dict[int, str] = {}
    v_slice: Dict[int, str] = {}
    k_write: Dict[int, str] = {}
    v_write: Dict[int, str] = {}
    for kh in range(kvh):
        # K slice
        nid_KS = f"L{l}_K_slice_H{kh}"
        ks_attr = dict(base_attr)
        ks_attr.update(
            {
                "op": "K_slice",
                "dim": int(hd),
                "kv_heads": 1,
                "n_kv_heads": 1,
                "kv_dim": int(hd),
                "head_dim": int(hd),
                "kv_head_id": int(kh),
                "kv_head_ids": [int(kh)],
            }
        )
        g.add_node(TaskNode(nid_KS, "Identity", flops=0.0, attrs=ks_attr, allowed=get_op_allowed("Identity")))
        g.add_edge(nid_K, nid_KS)
        k_slice[int(kh)] = nid_KS

        # V slice
        nid_VS = f"L{l}_V_slice_H{kh}"
        vs_attr = dict(base_attr)
        vs_attr.update(
            {
                "op": "V_slice",
                "dim": int(hd),
                "kv_heads": 1,
                "n_kv_heads": 1,
                "kv_dim": int(hd),
                "head_dim": int(hd),
                "kv_head_id": int(kh),
                "kv_head_ids": [int(kh)],
            }
        )
        g.add_node(TaskNode(nid_VS, "Identity", flops=0.0, attrs=vs_attr, allowed=get_op_allowed("Identity")))
        g.add_edge(nid_V, nid_VS)
        v_slice[int(kh)] = nid_VS

        # K write
        nid_KW = f"L{l}_K_write_H{kh}"
        kw_attr = dict(base_attr)
        kw_attr.update(
            {
                "kv_heads": 1,
                "n_kv_heads": 1,
                "kv_dim": int(hd),
                "head_dim": int(hd),
                "kv_head_id": int(kh),
                "kv_head_ids": [int(kh)],
            }
        )
        g.add_node(TaskNode(nid_KW, "K_write", attrs=kw_attr, allowed=get_op_allowed("K_write")))
        g.add_edge(nid_KS, nid_KW)
        k_write[int(kh)] = nid_KW

        # V write
        nid_VW = f"L{l}_V_write_H{kh}"
        vw_attr = dict(base_attr)
        vw_attr.update(
            {
                "kv_heads": 1,
                "n_kv_heads": 1,
                "kv_dim": int(hd),
                "head_dim": int(hd),
                "kv_head_id": int(kh),
                "kv_head_ids": [int(kh)],
            }
        )
        g.add_node(TaskNode(nid_VW, "V_write", attrs=vw_attr, allowed=get_op_allowed("V_write")))
        g.add_edge(nid_VS, nid_VW)
        v_write[int(kh)] = nid_VW

    # --- Query-head slice + per-head attention compute ---
    sv_heads: List[str] = []
    for qh_id in range(qh):
        kv_id = _kv_head_for_q_head(qh_id, q_heads=qh, kv_heads=kvh)

        # Q slice (one query head)
        nid_QS = f"L{l}_Q_slice_H{qh_id}"
        qs_attr = dict(base_attr)
        qs_attr.update(
            {
                "op": "Q_slice",
                "dim": int(hd),
                "q_heads": 1,
                "q_dim": int(hd),
                "head_dim": int(hd),
                "q_head_id": int(qh_id),
                "q_head_ids": [int(qh_id)],
                # Keep totals for GQA bookkeeping
                "n_heads": int(qh),
                "n_kv_heads": int(kvh),
            }
        )
        g.add_node(TaskNode(nid_QS, "Identity", flops=0.0, attrs=qs_attr, allowed=get_op_allowed("Identity")))
        g.add_edge(nid_Q, nid_QS)

        # QK
        nid_QK = f"L{l}_QK_H{qh_id}"
        qk_attr = dict(base_attr)
        qk_attr.update(
            {
                "head_shard": int(qh_id),
                "q_heads": 1,
                "kv_heads": 1,
                "n_heads": int(qh),
                "n_kv_heads": int(kvh),
                "q_dim": int(hd),
                "kv_dim": int(hd),
                "head_dim": int(hd),
                "q_head_id": int(qh_id),
                "q_head_ids": [int(qh_id)],
                "kv_head_id": int(kv_id),
                "kv_head_ids": [int(kv_id)],
            }
        )
        g.add_node(TaskNode(nid_QK, "QK", flops=0.0, attrs=qk_attr, allowed=get_op_allowed("QK")))
        g.add_edge(nid_QS, nid_QK)
        g.add_edge(k_slice[int(kv_id)], nid_QK)

        # Softmax
        nid_SO = f"L{l}_Softmax_H{qh_id}"
        so_attr = dict(qk_attr)
        so_attr.update({"op": "Softmax"})
        g.add_node(TaskNode(nid_SO, "Softmax", flops=0.0, attrs=so_attr, allowed=get_op_allowed("Softmax")))
        g.add_edge(nid_QK, nid_SO)

        # SV
        nid_SV = f"L{l}_SV_H{qh_id}"
        sv_attr = dict(qk_attr)
        sv_attr.update({"op": "SV"})
        g.add_node(TaskNode(nid_SV, "SV", flops=0.0, attrs=sv_attr, allowed=get_op_allowed("SV")))
        g.add_edge(nid_SO, nid_SV)
        g.add_edge(v_slice[int(kv_id)], nid_SV)
        sv_heads.append(nid_SV)

    # --- Output projection O (unsplit) ---
    nid_O = f"L{l}_O"
    g.add_node(
        TaskNode(
            nid_O,
            "O",
            flops=0.0,
            weight_id=f"L{l}_WO",
            weight_size=int(o_in_dim * dim * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("O"),
        )
    )
    for nid_SV in sv_heads:
        g.add_edge(nid_SV, nid_O)

    # Add1 (residual)
    g.add_edge(nid_O, add1_nid)
    if x_in is not None:
        g.add_edge(x_in, add1_nid)

    # Return KV write node ids so caller can connect to next layer
    return {
        "k_write": [k_write[k] for k in sorted(k_write.keys())],
        "v_write": [v_write[k] for k in sorted(v_write.keys())],
    }


# ---------------------------------------------------------------------------
# Model blocks
# ---------------------------------------------------------------------------


def add_llama_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """LLaMA/Qwen style block with attention head split + KV write sharding."""

    b = int(shape.batch)
    dim, ffn = int(shape.dim), int(shape.ffn_dim)
    qh, kvh, hd = int(shape.n_heads), int(shape.n_kv_heads), int(shape.head_dim)
    q_dim, kv_dim, o_in_dim = int(qh * hd), int(kvh * hd), int(qh * hd)

    base_attr = {
        "layer": int(l),
        "batch": int(b),
        "dim": int(dim),
        "ffn_dim": int(ffn),
        "q_heads": int(qh),
        "kv_heads": int(kvh),
        "n_heads": int(qh),
        "n_kv_heads": int(kvh),
        "head_dim": int(hd),
        "q_dim": int(q_dim),
        "kv_dim": int(kv_dim),
        "o_dim": int(o_in_dim),
    }

    # LN
    nid_LN = f"L{l}_LN"
    g.add_node(TaskNode(nid_LN, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))

    # Depend on previous layer KV writes (if any)
    if l > 0:
        for kh in range(kvh):
            g.add_edge(f"L{l-1}_K_write_H{kh}", nid_LN)
            g.add_edge(f"L{l-1}_V_write_H{kh}", nid_LN)

    # Add1 placeholder (created early so attention helper can wire it)
    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    x_in = f"L{l-1}_Add2" if l > 0 else None

    # Attention
    kv_nodes = _add_attention_head_split_llama_style(
        g,
        l=l,
        shape=shape,
        dtype_bytes=dtype_bytes,
        base_attr=base_attr,
        ln_nid=nid_LN,
        x_in=x_in,
        add1_nid=nid_Add1,
        add_x_identity=(x_in is not None),
    )

    # LN2
    nid_LN2 = f"L{l}_LN2"
    g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    g.add_edge(nid_Add1, nid_LN2)

    # FFN (unsplit)
    nid_W1 = f"L{l}_FFN_W1"
    nid_W3 = f"L{l}_FFN_W3"
    nid_ACT = f"L{l}_SwiGLU"
    nid_W2 = f"L{l}_FFN_W2"
    nid_Add2 = f"L{l}_Add2"

    g.add_node(
        TaskNode(
            nid_W1,
            "FFN_W1",
            flops=0.0,
            weight_id=f"L{l}_W1",
            weight_size=int(dim * ffn * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("FFN_W1"),
        )
    )
    g.add_node(
        TaskNode(
            nid_W3,
            "FFN_W3",
            flops=0.0,
            weight_id=f"L{l}_W3",
            weight_size=int(dim * ffn * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("FFN_W3"),
        )
    )
    g.add_node(TaskNode(nid_ACT, "SwiGLU", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("SwiGLU")))
    g.add_node(
        TaskNode(
            nid_W2,
            "FFN_W2",
            flops=0.0,
            weight_id=f"L{l}_W2",
            weight_size=int(ffn * dim * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("FFN_W2"),
        )
    )
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    g.add_edge(nid_LN2, nid_W1)
    g.add_edge(nid_LN2, nid_W3)
    g.add_edge(nid_W1, nid_ACT)
    g.add_edge(nid_W3, nid_ACT)
    g.add_edge(nid_ACT, nid_W2)
    g.add_edge(nid_W2, nid_Add2)
    g.add_edge(nid_Add1, nid_Add2)

    # NOTE: kv_nodes returned by attention helper are intentionally not merged.
    # They are only used for cross-layer ordering above.
    return kv_nodes


def add_mpt_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """MPT style: attention + GELU MLP. Keep attention head split + KV write sharding."""

    b = int(shape.batch)
    dim, ffn = int(shape.dim), int(shape.ffn_dim)
    qh, kvh, hd = int(shape.n_heads), int(shape.n_kv_heads), int(shape.head_dim)
    q_dim, kv_dim, o_in_dim = int(qh * hd), int(kvh * hd), int(qh * hd)

    base_attr = {
        "layer": int(l),
        "batch": int(b),
        "dim": int(dim),
        "ffn_dim": int(ffn),
        "q_heads": int(qh),
        "kv_heads": int(kvh),
        "n_heads": int(qh),
        "n_kv_heads": int(kvh),
        "head_dim": int(hd),
        "q_dim": int(q_dim),
        "kv_dim": int(kv_dim),
        "o_dim": int(o_in_dim),
    }

    nid_LN1 = f"L{l}_LN1"
    g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    if l > 0:
        for kh in range(kvh):
            g.add_edge(f"L{l-1}_K_write_H{kh}", nid_LN1)
            g.add_edge(f"L{l-1}_V_write_H{kh}", nid_LN1)

    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
    x_in = f"L{l-1}_Add2" if l > 0 else None

    _add_attention_head_split_llama_style(
        g,
        l=l,
        shape=shape,
        dtype_bytes=dtype_bytes,
        base_attr=base_attr,
        ln_nid=nid_LN1,
        x_in=x_in,
        add1_nid=nid_Add1,
        add_x_identity=(x_in is not None),
    )

    # LN2 + GELU MLP
    nid_LN2 = f"L{l}_LN2"
    nid_W1 = f"L{l}_FFN_W1"
    nid_G = f"L{l}_GELU"
    nid_W2 = f"L{l}_FFN_W2"
    nid_Add2 = f"L{l}_Add2"

    g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    g.add_node(
        TaskNode(
            nid_W1,
            "FFN_W1",
            flops=0.0,
            weight_id=f"L{l}_W1",
            weight_size=int(dim * ffn * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("FFN_W1"),
        )
    )
    g.add_node(TaskNode(nid_G, "GELU", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("GELU")))
    g.add_node(
        TaskNode(
            nid_W2,
            "FFN_W2",
            flops=0.0,
            weight_id=f"L{l}_W2",
            weight_size=int(ffn * dim * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("FFN_W2"),
        )
    )
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    g.add_edge(nid_Add1, nid_LN2)
    g.add_edge(nid_LN2, nid_W1)
    g.add_edge(nid_W1, nid_G)
    g.add_edge(nid_G, nid_W2)
    g.add_edge(nid_W2, nid_Add2)
    g.add_edge(nid_Add1, nid_Add2)


def add_palm_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """PaLM uses pre-LN and PARALLEL residual: x + Attn(LN(x)) + MLP(LN(x)).

    We keep attention split by head and KV write sharding.
    MLP is unsplit.
    """

    b = int(shape.batch)
    dim, ffn = int(shape.dim), int(shape.ffn_dim)
    qh, kvh, hd = int(shape.n_heads), int(shape.n_kv_heads), int(shape.head_dim)
    q_dim, kv_dim, o_in_dim = int(qh * hd), int(kvh * hd), int(qh * hd)

    base_attr = {
        "layer": int(l),
        "batch": int(b),
        "dim": int(dim),
        "ffn_dim": int(ffn),
        "q_heads": int(qh),
        "kv_heads": int(kvh),
        "n_heads": int(qh),
        "n_kv_heads": int(kvh),
        "head_dim": int(hd),
        "q_dim": int(q_dim),
        "kv_dim": int(kv_dim),
        "o_dim": int(o_in_dim),
    }

    nid_LN = f"L{l}_LN"
    g.add_node(TaskNode(nid_LN, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    if l > 0:
        for kh in range(kvh):
            g.add_edge(f"L{l-1}_K_write_H{kh}", nid_LN)
            g.add_edge(f"L{l-1}_V_write_H{kh}", nid_LN)

    x_in = f"L{l-1}_Add2" if l > 0 else None
    if x_in is not None:
        nid_X = f"L{l}_X"
        g.add_node(TaskNode(nid_X, "Identity", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Identity")))
        g.add_edge(x_in, nid_X)
        g.add_edge(nid_X, nid_LN)

    # Attention branch output projection ends at L{l}_O (created inside helper).
    # We'll then add it into Add2 together with MLP branch and residual.
    nid_Add_Attn = f"L{l}_Add_attn"
    g.add_node(TaskNode(nid_Add_Attn, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    # Build attention and wire its output to nid_Add_Attn.
    # We need an Add node that sums: residual + attn_out + mlp_out.
    # We'll do it as: attn_out -> Add_attn (with residual), then Add2 adds mlp.
    _add_attention_head_split_llama_style(
        g,
        l=l,
        shape=shape,
        dtype_bytes=dtype_bytes,
        base_attr=base_attr,
        ln_nid=nid_LN,
        x_in=x_in,
        add1_nid=nid_Add_Attn,
        add_x_identity=False,  # already created X above
    )

    # MLP branch (unsplit) reading from the same LN
    nid_W1 = f"L{l}_FFN_W1"
    nid_ACT = f"L{l}_GELU"
    nid_W2 = f"L{l}_FFN_W2"
    g.add_node(
        TaskNode(
            nid_W1,
            "FFN_W1",
            flops=0.0,
            weight_id=f"L{l}_W1",
            weight_size=int(dim * ffn * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("FFN_W1"),
        )
    )
    g.add_node(TaskNode(nid_ACT, "GELU", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("GELU")))
    g.add_node(
        TaskNode(
            nid_W2,
            "FFN_W2",
            flops=0.0,
            weight_id=f"L{l}_W2",
            weight_size=int(ffn * dim * dtype_bytes),
            attrs=dict(base_attr),
            allowed=get_op_allowed("FFN_W2"),
        )
    )
    g.add_edge(nid_LN, nid_W1)
    g.add_edge(nid_W1, nid_ACT)
    g.add_edge(nid_ACT, nid_W2)

    nid_Add2 = f"L{l}_Add2"
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
    g.add_edge(nid_Add_Attn, nid_Add2)
    g.add_edge(nid_W2, nid_Add2)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class LLaMADef:
    name = "llama"

    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_llama_block(g, l, shape, int(dtype_bytes))
        return g


class MPTDef:
    name = "mpt"

    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_mpt_block(g, l, shape, int(dtype_bytes))
        return g


class PaLMDef:
    name = "palm"

    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_palm_block(g, l, shape, int(dtype_bytes))
        return g


class QwenDef:
    name = "qwen"

    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        # Qwen is LLaMA-style for our graph purposes.
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_llama_block(g, l, shape, int(dtype_bytes))
        return g


class MixtralDef:
    name = "mixtral"

    @staticmethod
    def _active_expert_count(total: int, top_k: int, imbalance: float) -> int:
        if total <= 0:
            return 0
        guard = max(1.0, float(imbalance or 1.0))
        baseline = max(1, int(top_k))
        return max(1, min(int(total), int(math.ceil(float(baseline) * guard))))

    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()

        experts = int(getattr(shape, "experts_per_layer", 1) or 1)
        top_k = int(getattr(shape, "experts_top_k", 1) or 1)
        moe_imbalance = float(getattr(shape, "moe_imbalance_factor", 1.0) or 1.0)
        active_experts = int(getattr(shape, "active_experts_per_layer", 0) or 0)
        if active_experts <= 0 or active_experts > experts:
            active_experts = self._active_expert_count(experts, top_k, moe_imbalance)
        setattr(shape, "active_experts_per_layer", int(active_experts))
        setattr(shape, "moe_pruned_experts_per_layer", max(0, int(experts - active_experts)))

        b = int(shape.batch)
        dim, ffn = int(shape.dim), int(shape.ffn_dim)
        qh, kvh, hd = int(shape.n_heads), int(shape.n_kv_heads), int(shape.head_dim)
        q_dim, kv_dim, o_in_dim = int(qh * hd), int(kvh * hd), int(qh * hd)

        for l in range(int(shape.layer_num)):
            base_attr = {
                "layer": int(l),
                "batch": int(b),
                "dim": int(dim),
                "ffn_dim": int(ffn),
                "q_heads": int(qh),
                "kv_heads": int(kvh),
                "n_heads": int(qh),
                "n_kv_heads": int(kvh),
                "head_dim": int(hd),
                "q_dim": int(q_dim),
                "kv_dim": int(kv_dim),
                "o_dim": int(o_in_dim),
                "experts": int(experts),
                "top_k": int(top_k),
                "moe_imbalance_factor": float(moe_imbalance),
            }

            # Attention part (LLaMA-style, but LN is LN1 here)
            nid_LN1 = f"L{l}_LN1"
            g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            if l > 0:
                for kh in range(kvh):
                    g.add_edge(f"L{l-1}_K_write_H{kh}", nid_LN1)
                    g.add_edge(f"L{l-1}_V_write_H{kh}", nid_LN1)

            nid_Add1 = f"L{l}_Add1"
            g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            x_in = f"L{l-1}_Add2" if l > 0 else None

            _add_attention_head_split_llama_style(
                g,
                l=l,
                shape=shape,
                dtype_bytes=int(dtype_bytes),
                base_attr=base_attr,
                ln_nid=nid_LN1,
                x_in=x_in,
                add1_nid=nid_Add1,
                add_x_identity=(x_in is not None),
            )

            # LN2
            nid_LN2 = f"L{l}_LN2"
            g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            g.add_edge(nid_Add1, nid_LN2)

            # Experts (unsplit weights)
            expert_outputs: List[str] = []
            for e in range(int(active_experts)):
                expert_attr = {
                    **base_attr,
                    "expert": int(e),
                    "experts": int(experts),
                    "active_experts": int(active_experts),
                    "top_k": int(top_k),
                    "moe_imbalance": float(moe_imbalance),
                }
                nid_W1 = f"L{l}_FFN_W1_E{e}"
                nid_W3 = f"L{l}_FFN_W3_E{e}"
                nid_ACT = f"L{l}_Act_E{e}"
                nid_W2 = f"L{l}_FFN_W2_E{e}"

                g.add_node(
                    TaskNode(
                        nid_W1,
                        "FFN_W1",
                        flops=0.0,
                        weight_id=f"L{l}_E{e}_W1",
                        weight_size=int(dim * ffn * dtype_bytes),
                        attrs=dict(expert_attr),
                        allowed=get_op_allowed("FFN_W1"),
                    )
                )
                g.add_node(
                    TaskNode(
                        nid_W3,
                        "FFN_W3",
                        flops=0.0,
                        weight_id=f"L{l}_E{e}_W3",
                        weight_size=int(dim * ffn * dtype_bytes),
                        attrs=dict(expert_attr),
                        allowed=get_op_allowed("FFN_W3"),
                    )
                )
                g.add_node(TaskNode(nid_ACT, "SwiGLU", flops=0.0, attrs=dict(expert_attr), allowed=get_op_allowed("SwiGLU")))
                g.add_node(
                    TaskNode(
                        nid_W2,
                        "FFN_W2",
                        flops=0.0,
                        weight_id=f"L{l}_E{e}_W2",
                        weight_size=int(ffn * dim * dtype_bytes),
                        attrs=dict(expert_attr),
                        allowed=get_op_allowed("FFN_W2"),
                    )
                )

                g.add_edge(nid_LN2, nid_W1)
                g.add_edge(nid_LN2, nid_W3)
                g.add_edge(nid_W1, nid_ACT)
                g.add_edge(nid_W3, nid_ACT)
                g.add_edge(nid_ACT, nid_W2)
                expert_outputs.append(nid_W2)

            # Router
            nid_router = f"L{l}_Router"
            g.add_node(
                TaskNode(
                    nid_router,
                    "MoE_Router",
                    flops=0.0,
                    attrs={
                        **base_attr,
                        "experts": int(experts),
                        "active_experts": int(active_experts),
                        "top_k": int(top_k),
                        "moe_imbalance": float(moe_imbalance),
                    },
                    allowed=get_op_allowed("MoE_Router"),
                )
            )
            g.add_edge(nid_LN2, nid_router)
            for out in expert_outputs:
                g.add_edge(out, nid_router)

            nid_Add2 = f"L{l}_Add2"
            g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            g.add_edge(nid_router, nid_Add2)
            g.add_edge(nid_Add1, nid_Add2)

        return g


def make_model_def(family: str):
    f = (family or "").lower()
    if f == "llama":
        return LLaMADef()
    if f == "mpt":
        return MPTDef()
    if f == "palm":
        return PaLMDef()
    if f == "mixtral":
        return MixtralDef()
    if f == "qwen":
        return QwenDef()
    raise ValueError(f"Unknown model family: {family}")
