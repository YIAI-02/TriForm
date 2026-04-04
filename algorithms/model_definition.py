from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, List, Optional
from task_graph import TaskGraph, TaskNode
from config import OPERATOR_DEVICE_ALLOWED

# =============================================================================
# Model shape
# =============================================================================

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


# =============================================================================
# Helpers
# =============================================================================

def get_op_allowed(op_name: str) -> Dict[str, bool]:
    """Device constraints per operator."""
    key = str(op_name).strip().upper()
    return OPERATOR_DEVICE_ALLOWED.get(key, {}).copy()

def _weight_bytes(num_elems: int | float, dtype_bytes: float) -> int:
    elems = max(0.0, float(num_elems))
    return int(math.ceil(elems * float(dtype_bytes)))


def _weight_attrs(base_attr: Dict, num_elems: int | float, dtype_bytes: float) -> Dict:
    attrs = dict(base_attr)
    attrs['weight_elements'] = int(max(0.0, float(num_elems)))
    attrs['weight_dtype_bytes'] = float(dtype_bytes)
    return attrs

def _normalize_topology(topology: Optional[str]) -> str:
    """Normalize topology string to one of {'fc','star'} (default 'fc')."""
    if not topology:
        return 'fc'
    t = str(topology).strip().lower()
    if t in ('fully_connected', 'full', 'fc', 'mesh'):
        return 'fc'
    if t in ('star', 'host', 'host_star', 'host-centric', 'host_centric'):
        return 'star'
    return t

def _insert_row_parallel_collective(
    g: TaskGraph,
    *,
    l: int,
    tag: str,
    base_attr: Dict,
    inputs: List[str],
    cfg: Optional[Dict] = None,
) -> str:
    topo = _normalize_topology((cfg or {}).get('topology'))
    tag = str(tag)
    if topo == 'star':
        nid_red = f"L{l}_Reduce_{tag}"
        nid_scat = f"L{l}_Scatter_{tag}"
        red_attr = dict(base_attr)
        red_attr.update({'topology': 'star', 'primitive': 'reduce'})
        scat_attr = dict(base_attr)
        # In our graph abstraction, scatter acts as host distribution (broadcast).
        scat_attr.update({'topology': 'star', 'primitive': 'scatter', 'scatter_mode': 'broadcast', 'target_type': 'pim'})
        g.add_node(TaskNode(nid_red, 'REDUCE', flops=0.0, attrs=red_attr, allowed=get_op_allowed('REDUCE')))
        g.add_node(TaskNode(nid_scat, 'SCATTER', flops=0.0, attrs=scat_attr, allowed=get_op_allowed('SCATTER')))
        for u in inputs:
            g.add_edge(u, nid_red)
        g.add_edge(nid_red, nid_scat)
        return nid_scat
    else:
        nid_ar = f"L{l}_AllReduce_{tag}"
        g.add_node(TaskNode(nid_ar, 'ALLREDUCE', flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed('ALLREDUCE')))
        for u in inputs:
            g.add_edge(u, nid_ar)
        return nid_ar

def _add_attention_llama_style_unsplit(
    g: TaskGraph,
    *,
    l: int,
    shape: ModelShape,
    dtype_bytes: float,
    base_attr: Dict,
    ln_nid: str,
    x_in: Optional[str],
    add1_nid: str,
) -> Dict[str, str]:
    """Add a *non-splittable* LLaMA-style attention subgraph."""
    dim = int(shape.dim)
    qh = int(shape.n_heads)
    kvh = int(shape.n_kv_heads)
    hd = int(shape.head_dim)

    q_dim = int(qh * hd)
    kv_dim = int(kvh * hd)
    o_in_dim = int(qh * hd)

    # --- Q/K/V projections ---
    nid_Q = f"L{l}_Q"
    nid_K = f"L{l}_K"
    nid_V = f"L{l}_V"
    g.add_node(
        TaskNode(
            nid_Q,
            "Q",
            flops=0.0,
            weight_id=f"L{l}_WQ",
            weight_size=_weight_bytes(dim * q_dim, dtype_bytes),
            attrs=_weight_attrs(base_attr, dim * q_dim, dtype_bytes),
            allowed=get_op_allowed("Q"),
        )
    )
    g.add_node(
        TaskNode(
            nid_K,
            "K",
            flops=0.0,
            weight_id=f"L{l}_WK",
            weight_size=_weight_bytes(dim * kv_dim, dtype_bytes),
            attrs=_weight_attrs(base_attr, dim * kv_dim, dtype_bytes),
            allowed=get_op_allowed("K"),
        )
    )
    g.add_node(
        TaskNode(
            nid_V,
            "V",
            flops=0.0,
            weight_id=f"L{l}_WV",
            weight_size=_weight_bytes(dim * kv_dim, dtype_bytes),
            attrs=_weight_attrs(base_attr, dim * kv_dim, dtype_bytes),
            allowed=get_op_allowed("V"),
        )
    )

    g.add_edge(ln_nid, nid_Q)
    g.add_edge(ln_nid, nid_K)
    g.add_edge(ln_nid, nid_V)

    # --- KV cache writes (single operator per layer; not sharded) ---
    nid_KW = f"L{l}_K_write"
    nid_VW = f"L{l}_V_write"
    g.add_node(TaskNode(nid_KW, "K_write", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("K_write")))
    g.add_node(TaskNode(nid_VW, "V_write", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("V_write")))
    g.add_edge(nid_K, nid_KW)
    g.add_edge(nid_V, nid_VW)

    # --- QK / Softmax / SV / O ---
    nid_QK = f"L{l}_QK"
    nid_SM = f"L{l}_Softmax"
    nid_SV = f"L{l}_SV"
    nid_O = f"L{l}_O"

    g.add_node(TaskNode(nid_QK, "QK", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("QK")))
    g.add_node(TaskNode(nid_SM, "Softmax", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Softmax")))
    g.add_node(TaskNode(nid_SV, "SV", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("SV")))
    g.add_node(
        TaskNode(
            nid_O,
            "O",
            flops=0.0,
            weight_id=f"L{l}_WO",
            weight_size=_weight_bytes(o_in_dim * dim, dtype_bytes),
            attrs=_weight_attrs(base_attr, o_in_dim * dim, dtype_bytes),
            allowed=get_op_allowed("O"),
        )
    )

    g.add_edge(nid_Q, nid_QK)
    g.add_edge(nid_K, nid_QK)

    g.add_edge(nid_QK, nid_SM)

    g.add_edge(nid_SM, nid_SV)
    g.add_edge(nid_V, nid_SV)

    g.add_edge(nid_SV, nid_O)

    # Residual add
    if x_in is not None:
        g.add_edge(x_in, add1_nid)
    g.add_edge(nid_O, add1_nid)

    return {"k_write": nid_KW, "v_write": nid_VW}


def _partition_head_shards_llama_style(
    *,
    q_heads: int,
    kv_heads: int,
    tp_qkv: int,
) -> List[Dict[str, int]]:
    """Return head-parallel shards for GQA/MQA."""
    qh = int(q_heads)
    kvh = int(kv_heads)
    tp = max(1, int(tp_qkv))
    if tp <= 1:
        return [
            {
                "shard": 0,
                "q_head_start": 0,
                "q_head_end": int(qh),
                "kv_head_start": 0,
                "kv_head_end": int(kvh),
                "q_heads": int(qh),
                "kv_heads": int(kvh),
            }
        ]

    if qh <= 0 or kvh <= 0:
        return []
    if (qh % tp) != 0 or (kvh % tp) != 0:
        # Should be validated upstream; fallback to per-kv-head shards.
        tp = int(kvh)

    q_per = max(1, qh // tp)
    kv_per = max(1, kvh // tp)
    shards: List[Dict[str, int]] = []
    for i in range(int(tp)):
        q0 = int(i * q_per)
        q1 = int((i + 1) * q_per)
        kv0 = int(i * kv_per)
        kv1 = int((i + 1) * kv_per)
        shards.append(
            {
                "shard": int(i),
                "q_head_start": int(q0),
                "q_head_end": int(q1),
                "kv_head_start": int(kv0),
                "kv_head_end": int(kv1),
                "q_heads": int(q_per),
                "kv_heads": int(kv_per),
            }
        )
    return shards


def _add_attention_llama_style_tp(
    g: TaskGraph,
    *,
    l: int,
    shape: ModelShape,
    dtype_bytes: float,
    base_attr: Dict,
    ln_nid: str,
    x_in: Optional[str],
    add1_nid: str,
    tp_qkv: int,
    cfg: Optional[Dict] = None,
) -> Dict[str, List[str]]:
    """Add a sharded (TP) LLaMA-style attention subgraph.

    - Q/K/V generation are column-parallel (by head groups).
    - Attention dataflow (QK/Softmax/SV) runs per shard.
    - O (WO) is row-parallel => an ALLREDUCE node is inserted.
    - KV writes are sharded by the same KV-head slices.
    """
    dim = int(shape.dim)
    qh = int(shape.n_heads)
    kvh = int(shape.n_kv_heads)
    hd = int(shape.head_dim)

    shards = _partition_head_shards_llama_style(q_heads=qh, kv_heads=kvh, tp_qkv=int(tp_qkv))
    o_shards: List[str] = []
    k_writes: List[str] = []
    v_writes: List[str] = []

    for s in shards:
        si = int(s["shard"])
        qhs = int(s["q_heads"])
        kvhs = int(s["kv_heads"])
        q_dim = int(qhs * hd)
        kv_dim = int(kvhs * hd)

        sh_attr = dict(base_attr)
        sh_attr.update(
            {
                "q_heads": int(qhs),
                "kv_heads": int(kvhs),
                "n_heads": int(qhs),
                "n_kv_heads": int(kvhs),
                "q_dim": int(q_dim),
                "kv_dim": int(kv_dim),
                "o_dim": int(q_dim),
                "head_shard": int(si),
                "q_head_start": int(s["q_head_start"]),
                "q_head_end": int(s["q_head_end"]),
                "kv_head_start": int(s["kv_head_start"]),
                "kv_head_end": int(s["kv_head_end"]),
                "tp_qkv": int(tp_qkv),
            }
        )

        # --- Q/K/V projections (column-parallel) ---
        nid_Q = f"L{l}_Q_s{si}"
        nid_K = f"L{l}_K_s{si}"
        nid_V = f"L{l}_V_s{si}"
        g.add_node(
            TaskNode(
                nid_Q,
                "Q",
                flops=0.0,
                weight_id=f"L{l}_WQ_s{si}",
                weight_size=_weight_bytes(dim * q_dim, dtype_bytes),
                attrs=_weight_attrs(sh_attr, dim * q_dim, dtype_bytes),
                allowed=get_op_allowed("Q"),
            )
        )
        g.add_node(
            TaskNode(
                nid_K,
                "K",
                flops=0.0,
                weight_id=f"L{l}_WK_s{si}",
                weight_size=_weight_bytes(dim * kv_dim, dtype_bytes),
                attrs=_weight_attrs(sh_attr, dim * kv_dim, dtype_bytes),
                allowed=get_op_allowed("K"),
            )
        )
        g.add_node(
            TaskNode(
                nid_V,
                "V",
                flops=0.0,
                weight_id=f"L{l}_WV_s{si}",
                weight_size=_weight_bytes(dim * kv_dim, dtype_bytes),
                attrs=_weight_attrs(sh_attr, dim * kv_dim, dtype_bytes),
                allowed=get_op_allowed("V"),
            )
        )
        g.add_edge(ln_nid, nid_Q)
        g.add_edge(ln_nid, nid_K)
        g.add_edge(ln_nid, nid_V)

        # --- KV cache writes (sharded) ---
        nid_KW = f"L{l}_K_write_s{si}"
        nid_VW = f"L{l}_V_write_s{si}"
        g.add_node(TaskNode(nid_KW, "K_write", flops=0.0, attrs=dict(sh_attr), allowed=get_op_allowed("K_write")))
        g.add_node(TaskNode(nid_VW, "V_write", flops=0.0, attrs=dict(sh_attr), allowed=get_op_allowed("V_write")))
        g.add_edge(nid_K, nid_KW)
        g.add_edge(nid_V, nid_VW)
        k_writes.append(nid_KW)
        v_writes.append(nid_VW)

        # --- QK / Softmax / SV ---
        nid_QK = f"L{l}_QK_s{si}"
        nid_SM = f"L{l}_Softmax_s{si}"
        nid_SV = f"L{l}_SV_s{si}"
        nid_O = f"L{l}_O_s{si}"

        g.add_node(TaskNode(nid_QK, "QK", flops=0.0, attrs=dict(sh_attr), allowed=get_op_allowed("QK")))
        g.add_node(TaskNode(nid_SM, "Softmax", flops=0.0, attrs=dict(sh_attr), allowed=get_op_allowed("Softmax")))
        g.add_node(TaskNode(nid_SV, "SV", flops=0.0, attrs=dict(sh_attr), allowed=get_op_allowed("SV")))
        g.add_node(
            TaskNode(
                nid_O,
                "O",
                flops=0.0,
                weight_id=f"L{l}_WO_s{si}",
                # WO is row-parallel: shard rows of the input dim (q_dim).
                weight_size=_weight_bytes(q_dim * dim, dtype_bytes),
                attrs=_weight_attrs(sh_attr, q_dim * dim, dtype_bytes),
                allowed=get_op_allowed("O"),
            )
        )

        g.add_edge(nid_Q, nid_QK)
        g.add_edge(nid_K, nid_QK)
        g.add_edge(nid_QK, nid_SM)
        g.add_edge(nid_SM, nid_SV)
        g.add_edge(nid_V, nid_SV)
        g.add_edge(nid_SV, nid_O)

        o_shards.append(nid_O)


    # Row-parallel WO requires a topology-dependent collective to form the full hidden state.
    nid_COL = _insert_row_parallel_collective(
        g,
        l=l,
        tag='O',
        base_attr=base_attr,
        inputs=o_shards,
        cfg=cfg,
    )

    # Residual add consumes the reduced output.
    if x_in is not None:
        g.add_edge(x_in, add1_nid)
    g.add_edge(nid_COL, add1_nid)

    return {"k_writes": k_writes, "v_writes": v_writes, "o_shards": o_shards, "o_collective": [nid_COL]}


# =============================================================================
# Block builders
# =============================================================================

def add_llama_block(
    g: TaskGraph,
    l: int,
    shape: ModelShape,
    dtype_bytes: float,
    cfg: Optional[Dict] = None,
):
    """LLaMA/Qwen style block."""

    b = int(shape.batch)
    dim, ffn = int(shape.dim), int(shape.ffn_dim)
    qh, kvh, hd = int(shape.n_heads), int(shape.n_kv_heads), int(shape.head_dim)
    q_dim, kv_dim, o_in_dim = int(qh * hd), int(kvh * hd), int(qh * hd)

    tp_qkv = int((cfg or {}).get('tp_qkv_effective', 1) or 1)
    tp_ffn = int((cfg or {}).get('tp_ffn_effective', 1) or 1)

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

    if l > 0:
        g.add_edge(f"L{l-1}_Add2", nid_LN)

    # Add1 placeholder
    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
    x_in = f"L{l-1}_Add2" if l > 0 else None

    if tp_qkv > 1:
        _add_attention_llama_style_tp(
            g,
            l=l,
            shape=shape,
            dtype_bytes=float(dtype_bytes),
            base_attr=base_attr,
            ln_nid=nid_LN,
            x_in=x_in,
            add1_nid=nid_Add1,
            tp_qkv=int(tp_qkv),
            cfg=cfg,
        )
    else:
        _add_attention_llama_style_unsplit(
            g,
            l=l,
            shape=shape,
            dtype_bytes=float(dtype_bytes),
            base_attr=base_attr,
            ln_nid=nid_LN,
            x_in=x_in,
            add1_nid=nid_Add1,
        )

    # LN2
    nid_LN2 = f"L{l}_LN2"
    g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    g.add_edge(nid_Add1, nid_LN2)

    # FFN (SwiGLU)
    nid_Add2 = f"L{l}_Add2"
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    if tp_ffn > 1:
        ffn_sh = int(ffn // tp_ffn)
        w2_shards: List[str] = []

        for si in range(int(tp_ffn)):
            sh_attr = dict(base_attr)
            sh_attr.update({"ffn_dim": int(ffn_sh), "tp_ffn": int(tp_ffn), "ffn_shard": int(si)})

            nid_W1 = f"L{l}_FFN_W1_s{si}"
            nid_W3 = f"L{l}_FFN_W3_s{si}"
            nid_ACT = f"L{l}_SwiGLU_s{si}"
            nid_W2 = f"L{l}_FFN_W2_s{si}"

            g.add_node(
                TaskNode(
                    nid_W1,
                    "FFN_W1",
                    flops=0.0,
                    weight_id=f"L{l}_W1_s{si}",
                    weight_size=_weight_bytes(dim * ffn_sh, dtype_bytes),
                    attrs=_weight_attrs(sh_attr, dim * ffn_sh, dtype_bytes),
                    allowed=get_op_allowed("FFN_W1"),
                )
            )
            g.add_node(
                TaskNode(
                    nid_W3,
                    "FFN_W3",
                    flops=0.0,
                    weight_id=f"L{l}_W3_s{si}",
                    weight_size=_weight_bytes(dim * ffn_sh, dtype_bytes),
                    attrs=_weight_attrs(sh_attr, dim * ffn_sh, dtype_bytes),
                    allowed=get_op_allowed("FFN_W3"),
                )
            )
            g.add_node(
                TaskNode(
                    nid_ACT,
                    "SwiGLU",
                    flops=0.0,
                    attrs=dict(sh_attr),
                    allowed=get_op_allowed("SwiGLU"),
                )
            )
            g.add_node(
                TaskNode(
                    nid_W2,
                    "FFN_W2",
                    flops=0.0,
                    weight_id=f"L{l}_W2_s{si}",
                    # FFN_W2 is row-parallel: shard rows of the ffn_dim.
                    weight_size=_weight_bytes(ffn_sh * dim, dtype_bytes),
                    attrs=_weight_attrs(sh_attr, ffn_sh * dim, dtype_bytes),
                    allowed=get_op_allowed("FFN_W2"),
                )
            )

            g.add_edge(nid_LN2, nid_W1)
            g.add_edge(nid_LN2, nid_W3)
            g.add_edge(nid_W1, nid_ACT)
            g.add_edge(nid_W3, nid_ACT)
            g.add_edge(nid_ACT, nid_W2)
            w2_shards.append(nid_W2)

        # Row-parallel FFN down-proj requires a topology-dependent collective.
        nid_COL = _insert_row_parallel_collective(
            g,
            l=l,
            tag='FFN',
            base_attr=base_attr,
            inputs=w2_shards,
            cfg=cfg,
        )
        g.add_edge(nid_COL, nid_Add2)
    else:
        nid_W1 = f"L{l}_FFN_W1"
        nid_W3 = f"L{l}_FFN_W3"
        nid_ACT = f"L{l}_SwiGLU"
        nid_W2 = f"L{l}_FFN_W2"

        g.add_node(
            TaskNode(
                nid_W1,
                "FFN_W1",
                flops=0.0,
                weight_id=f"L{l}_W1",
                weight_size=_weight_bytes(dim * ffn, dtype_bytes),
                attrs=_weight_attrs(base_attr, dim * ffn, dtype_bytes),
                allowed=get_op_allowed("FFN_W1"),
            )
        )
        g.add_node(
            TaskNode(
                nid_W3,
                "FFN_W3",
                flops=0.0,
                weight_id=f"L{l}_W3",
                weight_size=_weight_bytes(dim * ffn, dtype_bytes),
                attrs=_weight_attrs(base_attr, dim * ffn, dtype_bytes),
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
                weight_size=_weight_bytes(ffn * dim, dtype_bytes),
                attrs=_weight_attrs(base_attr, ffn * dim, dtype_bytes),
                allowed=get_op_allowed("FFN_W2"),
            )
        )

        g.add_edge(nid_LN2, nid_W1)
        g.add_edge(nid_LN2, nid_W3)
        g.add_edge(nid_W1, nid_ACT)
        g.add_edge(nid_W3, nid_ACT)
        g.add_edge(nid_ACT, nid_W2)
        g.add_edge(nid_W2, nid_Add2)

    g.add_edge(nid_Add1, nid_Add2)


def add_mpt_block(
    g: TaskGraph,
    l: int,
    shape: ModelShape,
    dtype_bytes: float,
    cfg: Optional[Dict] = None,
):
    """MPT style: attention + GELU MLP.

    Sharding rules:
    - Attention:
        * Q/K/V are column-parallel (head-group shards) when tp_qkv_effective > 1.
        * WO is row-parallel => topology-dependent collective (ALLREDUCE vs REDUCE+SCATTER).
    - MLP:
        * W1 is column-parallel on ffn_dim when tp_ffn_effective > 1.
        * W2 is row-parallel => topology-dependent collective.
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

    tp_qkv = int((cfg or {}).get('tp_qkv_effective', 1) or 1)
    tp_ffn = int((cfg or {}).get('tp_ffn_effective', 1) or 1)

    nid_LN1 = f"L{l}_LN1"
    g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    if l > 0:
        g.add_edge(f"L{l-1}_Add2", nid_LN1)

    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
    x_in = f"L{l-1}_Add2" if l > 0 else None

    if tp_qkv > 1:
        _add_attention_llama_style_tp(
            g,
            l=l,
            shape=shape,
            dtype_bytes=float(dtype_bytes),
            base_attr=base_attr,
            ln_nid=nid_LN1,
            x_in=x_in,
            add1_nid=nid_Add1,
            tp_qkv=int(tp_qkv),
            cfg=cfg,
        )
    else:
        _add_attention_llama_style_unsplit(
            g,
            l=l,
            shape=shape,
            dtype_bytes=float(dtype_bytes),
            base_attr=base_attr,
            ln_nid=nid_LN1,
            x_in=x_in,
            add1_nid=nid_Add1,
        )

    # LN2 + GELU MLP
    nid_LN2 = f"L{l}_LN2"
    nid_W1 = f"L{l}_FFN_W1"
    nid_G = f"L{l}_GELU"
    nid_W2 = f"L{l}_FFN_W2"
    nid_Add2 = f"L{l}_Add2"

    g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    g.add_edge(nid_Add1, nid_LN2)

    if tp_ffn > 1:
        ffn_sh = int(ffn // tp_ffn)
        w2_shards: List[str] = []

        for si in range(int(tp_ffn)):
            sh_attr = dict(base_attr)
            sh_attr.update({"ffn_dim": int(ffn_sh), "tp_ffn": int(tp_ffn), "ffn_shard": int(si)})

            nid_W1_s = f"L{l}_FFN_W1_s{si}"
            nid_G_s = f"L{l}_GELU_s{si}"
            nid_W2_s = f"L{l}_FFN_W2_s{si}"

            g.add_node(
                TaskNode(
                    nid_W1_s,
                    "FFN_W1",
                    flops=0.0,
                    weight_id=f"L{l}_W1_s{si}",
                    weight_size=_weight_bytes(dim * ffn_sh, dtype_bytes),
                    attrs=_weight_attrs(sh_attr, dim * ffn_sh, dtype_bytes),
                    allowed=get_op_allowed("FFN_W1"),
                )
            )
            g.add_node(TaskNode(nid_G_s, "GELU", flops=0.0, attrs=dict(sh_attr), allowed=get_op_allowed("GELU")))
            g.add_node(
                TaskNode(
                    nid_W2_s,
                    "FFN_W2",
                    flops=0.0,
                    weight_id=f"L{l}_W2_s{si}",
                    # FFN_W2 is row-parallel: shard rows of the ffn_dim.
                    weight_size=_weight_bytes(ffn_sh * dim, dtype_bytes),
                    attrs=_weight_attrs(sh_attr, ffn_sh * dim, dtype_bytes),
                    allowed=get_op_allowed("FFN_W2"),
                )
            )

            g.add_edge(nid_LN2, nid_W1_s)
            g.add_edge(nid_W1_s, nid_G_s)
            g.add_edge(nid_G_s, nid_W2_s)
            w2_shards.append(nid_W2_s)

        nid_COL = _insert_row_parallel_collective(
            g,
            l=l,
            tag='FFN',
            base_attr=base_attr,
            inputs=w2_shards,
            cfg=cfg,
        )
        g.add_edge(nid_COL, nid_Add2)
    else:
        g.add_node(
            TaskNode(
                nid_W1,
                "FFN_W1",
                flops=0.0,
                weight_id=f"L{l}_W1",
                weight_size=_weight_bytes(dim * ffn, dtype_bytes),
                attrs=_weight_attrs(base_attr, dim * ffn, dtype_bytes),
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
                weight_size=_weight_bytes(ffn * dim, dtype_bytes),
                attrs=_weight_attrs(base_attr, ffn * dim, dtype_bytes),
                allowed=get_op_allowed("FFN_W2"),
            )
        )

        g.add_edge(nid_LN2, nid_W1)
        g.add_edge(nid_W1, nid_G)
        g.add_edge(nid_G, nid_W2)
        g.add_edge(nid_W2, nid_Add2)

    g.add_edge(nid_Add1, nid_Add2)


def add_palm_block(
    g: TaskGraph,
    l: int,
    shape: ModelShape,
    dtype_bytes: float,
    cfg: Optional[Dict] = None,
):
    """PaLM uses pre-LN and PARALLEL residual: x + Attn(LN(x)) + MLP(LN(x)).

    Sharding rules:
    - Attention: tp_qkv_effective (head-parallel) with topology-dependent WO collective.
    - MLP: tp_ffn_effective (W1 column-parallel, W2 row-parallel) with topology-dependent collective.
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

    tp_qkv = int((cfg or {}).get('tp_qkv_effective', 1) or 1)
    tp_ffn = int((cfg or {}).get('tp_ffn_effective', 1) or 1)

    nid_LN = f"L{l}_LN"
    g.add_node(TaskNode(nid_LN, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
    if l > 0:
        g.add_edge(f"L{l-1}_Add2", nid_LN)

    x_in = f"L{l-1}_Add2" if l > 0 else None

    nid_Add_Attn = f"L{l}_Add_attn"
    g.add_node(TaskNode(nid_Add_Attn, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    if tp_qkv > 1:
        _add_attention_llama_style_tp(
            g,
            l=l,
            shape=shape,
            dtype_bytes=float(dtype_bytes),
            base_attr=base_attr,
            ln_nid=nid_LN,
            x_in=x_in,
            add1_nid=nid_Add_Attn,
            tp_qkv=int(tp_qkv),
            cfg=cfg,
        )
    else:
        _add_attention_llama_style_unsplit(
            g,
            l=l,
            shape=shape,
            dtype_bytes=float(dtype_bytes),
            base_attr=base_attr,
            ln_nid=nid_LN,
            x_in=x_in,
            add1_nid=nid_Add_Attn,
        )

    # MLP branch reading from the same LN
    nid_W1 = f"L{l}_FFN_W1"
    nid_ACT = f"L{l}_GELU"
    nid_W2 = f"L{l}_FFN_W2"

    if tp_ffn > 1:
        ffn_sh = int(ffn // tp_ffn)
        w2_shards: List[str] = []
        for si in range(int(tp_ffn)):
            sh_attr = dict(base_attr)
            sh_attr.update({"ffn_dim": int(ffn_sh), "tp_ffn": int(tp_ffn), "ffn_shard": int(si)})

            nid_W1_s = f"L{l}_FFN_W1_s{si}"
            nid_ACT_s = f"L{l}_GELU_s{si}"
            nid_W2_s = f"L{l}_FFN_W2_s{si}"

            g.add_node(
                TaskNode(
                    nid_W1_s,
                    "FFN_W1",
                    flops=0.0,
                    weight_id=f"L{l}_W1_s{si}",
                    weight_size=_weight_bytes(dim * ffn_sh, dtype_bytes),
                    attrs=_weight_attrs(sh_attr, dim * ffn_sh, dtype_bytes),
                    allowed=get_op_allowed("FFN_W1"),
                )
            )
            g.add_node(TaskNode(nid_ACT_s, "GELU", flops=0.0, attrs=dict(sh_attr), allowed=get_op_allowed("GELU")))
            g.add_node(
                TaskNode(
                    nid_W2_s,
                    "FFN_W2",
                    flops=0.0,
                    weight_id=f"L{l}_W2_s{si}",
                    weight_size=_weight_bytes(ffn_sh * dim, dtype_bytes),
                    attrs=_weight_attrs(sh_attr, ffn_sh * dim, dtype_bytes),
                    allowed=get_op_allowed("FFN_W2"),
                )
            )

            g.add_edge(nid_LN, nid_W1_s)
            g.add_edge(nid_W1_s, nid_ACT_s)
            g.add_edge(nid_ACT_s, nid_W2_s)
            w2_shards.append(nid_W2_s)

        nid_COL = _insert_row_parallel_collective(
            g,
            l=l,
            tag='FFN',
            base_attr=base_attr,
            inputs=w2_shards,
            cfg=cfg,
        )
        nid_MLP_OUT = nid_COL
    else:
        g.add_node(
            TaskNode(
                nid_W1,
                "FFN_W1",
                flops=0.0,
                weight_id=f"L{l}_W1",
                weight_size=_weight_bytes(dim * ffn, dtype_bytes),
                attrs=_weight_attrs(base_attr, dim * ffn, dtype_bytes),
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
                weight_size=_weight_bytes(ffn * dim, dtype_bytes),
                attrs=_weight_attrs(base_attr, ffn * dim, dtype_bytes),
                allowed=get_op_allowed("FFN_W2"),
            )
        )
        g.add_edge(nid_LN, nid_W1)
        g.add_edge(nid_W1, nid_ACT)
        g.add_edge(nid_ACT, nid_W2)
        nid_MLP_OUT = nid_W2

    nid_Add2 = f"L{l}_Add2"
    g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
    g.add_edge(nid_Add_Attn, nid_Add2)
    g.add_edge(nid_MLP_OUT, nid_Add2)


# =============================================================================
# Model definitions
# =============================================================================

class LLaMADef:
    name = "llama"

    def build(self, shape: ModelShape, dtype_bytes: float, cfg: Optional[Dict] = None) -> TaskGraph:
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_llama_block(g, l, shape, float(dtype_bytes), cfg=cfg)
        return g


class MPTDef:
    name = "mpt"

    def build(self, shape: ModelShape, dtype_bytes: float, cfg: Optional[Dict] = None) -> TaskGraph:
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_mpt_block(g, l, shape, float(dtype_bytes), cfg=cfg)
        return g


class PaLMDef:
    name = "palm"

    def build(self, shape: ModelShape, dtype_bytes: float, cfg: Optional[Dict] = None) -> TaskGraph:
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_palm_block(g, l, shape, float(dtype_bytes), cfg=cfg)
        return g


class QwenDef:
    name = "qwen"

    def build(self, shape: ModelShape, dtype_bytes: float, cfg: Optional[Dict] = None) -> TaskGraph:
        # Qwen is LLaMA-style for our graph purposes.
        g = TaskGraph()
        for l in range(int(shape.layer_num)):
            add_llama_block(g, l, shape, float(dtype_bytes), cfg=cfg)
        return g



class MixtralDef:
    name = "mixtral"

    @staticmethod
    def _resolve_top_k(total: int, top_k: int) -> int:
        total_i = max(1, int(total or 1))
        return max(1, min(total_i, int(top_k or 1)))

    @staticmethod
    def _select_first_k_experts(total: int, top_k: int) -> List[int]:
        total_i = max(1, int(total or 1))
        top_k_i = MixtralDef._resolve_top_k(total_i, top_k)
        return [int(e) for e in range(top_k_i)]

    @staticmethod
    def _plan_selected_expert_shards(selected_experts: List[int], tp_total: int) -> Dict[str, object]:
        selected = [int(e) for e in (selected_experts or [])]
        if not selected:
            return {
                "tp_total": 1,
                "tp_expert_ffn": 1,
                "experts_by_shard": [[]],
                "shards_by_expert": {},
            }

        top_k = len(selected)
        tp_total_i = max(1, int(tp_total or 1))

        if tp_total_i <= top_k:
            experts_by_shard: List[List[int]] = [[] for _ in range(int(tp_total_i))]
            shards_by_expert: Dict[int, List[int]] = {}
            for rank, e in enumerate(selected):
                # When tp <= top-k, we only distribute selected experts across total shards.
                # Each expert FFN itself stays unsplit.
                shard = int((int(rank) * int(tp_total_i)) // int(top_k))
                experts_by_shard[shard].append(int(e))
                shards_by_expert[int(e)] = [int(shard)]
            tp_expert_ffn = 1
        else:
            # When tp > top-k, each selected expert is split evenly into tp / top-k shards.
            if (tp_total_i % top_k) != 0:
                raise ValueError(
                    f"Invalid Mixtral tp={tp_total_i}: when tp > top_k, require tp%top_k==0 (top_k={top_k})."
                )
            tp_expert_ffn = int(tp_total_i // top_k)
            experts_by_shard = [[] for _ in range(int(tp_total_i))]
            shards_by_expert = {}
            gid = 0
            for e in selected:
                shard_ids: List[int] = []
                for _ in range(int(tp_expert_ffn)):
                    experts_by_shard[gid].append(int(e))
                    shard_ids.append(int(gid))
                    gid += 1
                shards_by_expert[int(e)] = shard_ids

        return {
            "tp_total": int(tp_total_i),
            "tp_expert_ffn": int(tp_expert_ffn),
            "experts_by_shard": [[int(x) for x in xs] for xs in experts_by_shard],
            "shards_by_expert": {int(k): [int(x) for x in v] for k, v in shards_by_expert.items()},
        }

    @staticmethod
    def _add_selected_expert_ffn(
        g: TaskGraph,
        *,
        l: int,
        shape: ModelShape,
        dtype_bytes: float,
        base_attr: Dict,
        ln_nid: str,
        expert_id: int,
        expert_rank: int,
        shard_ids: List[int],
        cfg: Optional[Dict] = None,
    ) -> str:
        dim = int(shape.dim)
        ffn = int(shape.ffn_dim)
        e = int(expert_id)
        shards = [int(s) for s in (shard_ids or [0])]
        shard_count = max(1, len(shards))

        expert_base_attr = {
            **base_attr,
            "expert": int(e),
            "expert_rank": int(expert_rank),
            "expert_active": True,
            # Deterministic static simulation: selected experts are treated as fully used.
            "moe_token_fraction": 1.0,
            "tp_ffn": int(shard_count),
            "tp_expert_ffn": int(shard_count),
            "expert_shard_ids": [int(s) for s in shards],
        }

        if shard_count > 1:
            if (ffn % shard_count) != 0:
                raise ValueError(
                    f"Invalid Mixtral expert split: require ffn_dim%tp_expert_ffn==0 "
                    f"(ffn_dim={ffn}, tp_expert_ffn={shard_count})."
                )
            ffn_sh = int(ffn // shard_count)
            w2_shards: List[str] = []

            for local_si, global_si in enumerate(shards):
                sh_attr = dict(expert_base_attr)
                sh_attr.update(
                    {
                        "ffn_dim": int(ffn_sh),
                        "ffn_shard": int(local_si),
                        "expert_shard_local": int(local_si),
                        "expert_shard": int(global_si),
                        "moe_shard": int(global_si),
                    }
                )

                nid_W1 = f"L{l}_FFN_W1_E{e}_s{local_si}"
                nid_W3 = f"L{l}_FFN_W3_E{e}_s{local_si}"
                nid_ACT = f"L{l}_Act_E{e}_s{local_si}"
                nid_W2 = f"L{l}_FFN_W2_E{e}_s{local_si}"

                g.add_node(
                    TaskNode(
                        nid_W1,
                        "FFN_W1",
                        flops=0.0,
                        weight_id=f"L{l}_E{e}_W1_s{local_si}",
                        weight_size=_weight_bytes(dim * ffn_sh, dtype_bytes),
                        attrs=_weight_attrs(sh_attr, dim * ffn_sh, dtype_bytes),
                        allowed=get_op_allowed("FFN_W1"),
                    )
                )
                g.add_node(
                    TaskNode(
                        nid_W3,
                        "FFN_W3",
                        flops=0.0,
                        weight_id=f"L{l}_E{e}_W3_s{local_si}",
                        weight_size=_weight_bytes(dim * ffn_sh, dtype_bytes),
                        attrs=_weight_attrs(sh_attr, dim * ffn_sh, dtype_bytes),
                        allowed=get_op_allowed("FFN_W3"),
                    )
                )
                g.add_node(
                    TaskNode(
                        nid_ACT,
                        "SwiGLU",
                        flops=0.0,
                        attrs=dict(sh_attr),
                        allowed=get_op_allowed("SwiGLU"),
                    )
                )
                g.add_node(
                    TaskNode(
                        nid_W2,
                        "FFN_W2",
                        flops=0.0,
                        weight_id=f"L{l}_E{e}_W2_s{local_si}",
                        weight_size=_weight_bytes(ffn_sh * dim, dtype_bytes),
                        attrs=_weight_attrs(sh_attr, ffn_sh * dim, dtype_bytes),
                        allowed=get_op_allowed("FFN_W2"),
                    )
                )

                g.add_edge(ln_nid, nid_W1)
                g.add_edge(ln_nid, nid_W3)
                g.add_edge(nid_W1, nid_ACT)
                g.add_edge(nid_W3, nid_ACT)
                g.add_edge(nid_ACT, nid_W2)
                w2_shards.append(nid_W2)

            collective_attr = dict(expert_base_attr)
            collective_attr.update(
                {
                    "ffn_dim": int(ffn),
                    "tp_ffn": int(shard_count),
                    "tp_expert_ffn": int(shard_count),
                }
            )
            return _insert_row_parallel_collective(
                g,
                l=l,
                tag=f"MoE_E{e}",
                base_attr=collective_attr,
                inputs=w2_shards,
                cfg=cfg,
            )

        shard_id = int(shards[0]) if shards else 0
        expert_attr = dict(expert_base_attr)
        expert_attr.update(
            {
                "ffn_dim": int(ffn),
                "ffn_shard": 0,
                "expert_shard_local": 0,
                "expert_shard": int(shard_id),
                "moe_shard": int(shard_id),
            }
        )

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
                weight_size=_weight_bytes(dim * ffn, dtype_bytes),
                attrs=_weight_attrs(expert_attr, dim * ffn, dtype_bytes),
                allowed=get_op_allowed("FFN_W1"),
            )
        )
        g.add_node(
            TaskNode(
                nid_W3,
                "FFN_W3",
                flops=0.0,
                weight_id=f"L{l}_E{e}_W3",
                weight_size=_weight_bytes(dim * ffn, dtype_bytes),
                attrs=_weight_attrs(expert_attr, dim * ffn, dtype_bytes),
                allowed=get_op_allowed("FFN_W3"),
            )
        )
        g.add_node(
            TaskNode(
                nid_ACT,
                "SwiGLU",
                flops=0.0,
                attrs=dict(expert_attr),
                allowed=get_op_allowed("SwiGLU"),
            )
        )
        g.add_node(
            TaskNode(
                nid_W2,
                "FFN_W2",
                flops=0.0,
                weight_id=f"L{l}_E{e}_W2",
                weight_size=_weight_bytes(ffn * dim, dtype_bytes),
                attrs=_weight_attrs(expert_attr, ffn * dim, dtype_bytes),
                allowed=get_op_allowed("FFN_W2"),
            )
        )

        g.add_edge(ln_nid, nid_W1)
        g.add_edge(ln_nid, nid_W3)
        g.add_edge(nid_W1, nid_ACT)
        g.add_edge(nid_W3, nid_ACT)
        g.add_edge(nid_ACT, nid_W2)
        return nid_W2

    def build(self, shape: ModelShape, dtype_bytes: float, cfg: Optional[Dict] = None) -> TaskGraph:
        g = TaskGraph()

        experts = int(getattr(shape, "experts_per_layer", 1) or 1)
        top_k = self._resolve_top_k(experts, int(getattr(shape, "experts_top_k", 2) or 2))
        selected_experts = self._select_first_k_experts(experts, top_k)
        active_experts = int(len(selected_experts))
        setattr(shape, "active_experts_per_layer", int(active_experts))
        setattr(shape, "moe_pruned_experts_per_layer", max(0, int(experts - active_experts)))

        b = int(shape.batch)
        dim, ffn = int(shape.dim), int(shape.ffn_dim)
        qh, kvh, hd = int(shape.n_heads), int(shape.n_kv_heads), int(shape.head_dim)
        q_dim, kv_dim, o_in_dim = int(qh * hd), int(kvh * hd), int(qh * hd)
        router_weight_elems = int(dim * experts)
        router_weight_size = _weight_bytes(router_weight_elems, dtype_bytes)

        tp_qkv = int((cfg or {}).get('tp_qkv_effective', 1) or 1)
        tp_total = int((cfg or {}).get('tp_moe_total_effective', (cfg or {}).get('tp_moe_effective', 1) or 1) or 1)
        shard_plan = self._plan_selected_expert_shards(selected_experts, tp_total)
        tp_expert_ffn = int(shard_plan["tp_expert_ffn"])
        experts_by_shard = list(shard_plan["experts_by_shard"])
        shards_by_expert = dict(shard_plan["shards_by_expert"])

        moe_imbalance = float(getattr(shape, "moe_imbalance_factor", 1.0) or 1.0)
        router_aux_loss_coef = getattr(shape, "router_aux_loss_coef", None)
        router_jitter_noise = getattr(shape, "router_jitter_noise", None)

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
                "experts_total": int(experts),
                "active_experts": int(active_experts),
                "top_k": int(top_k),
                "num_local_experts": int(experts),
                "num_experts_per_tok": int(top_k),
                "selected_experts": [int(e) for e in selected_experts],
                "selected_experts_by_shard": [[int(e) for e in xs] for xs in experts_by_shard],
                "router_kind": "topk_static_first_k",
                "moe_selection_policy": "first_k",
                "moe_imbalance_factor": float(moe_imbalance),
                "router_weight_size": int(router_weight_size),
                "tp_qkv": int(tp_qkv),
                "tp": int(tp_total),
                "tp_moe": int(tp_total),
                "tp_moe_total": int(tp_total),
                "tp_expert_ffn": int(tp_expert_ffn),
            }
            if router_aux_loss_coef is not None:
                base_attr["router_aux_loss_coef"] = float(router_aux_loss_coef)
            if router_jitter_noise is not None:
                base_attr["router_jitter_noise"] = float(router_jitter_noise)

            # Attention part (Mistral/LLaMA-style, LN1 here)
            nid_LN1 = f"L{l}_LN1"
            g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            if l > 0:
                g.add_edge(f"L{l-1}_Add2", nid_LN1)

            nid_Add1 = f"L{l}_Add1"
            g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            x_in = f"L{l-1}_Add2" if l > 0 else None

            if tp_qkv > 1:
                _add_attention_llama_style_tp(
                    g,
                    l=l,
                    shape=shape,
                    dtype_bytes=float(dtype_bytes),
                    base_attr=base_attr,
                    ln_nid=nid_LN1,
                    x_in=x_in,
                    add1_nid=nid_Add1,
                    tp_qkv=int(tp_qkv),
                    cfg=cfg,
                )
            else:
                _add_attention_llama_style_unsplit(
                    g,
                    l=l,
                    shape=shape,
                    dtype_bytes=float(dtype_bytes),
                    base_attr=base_attr,
                    ln_nid=nid_LN1,
                    x_in=x_in,
                    add1_nid=nid_Add1,
                )

            # LN2
            nid_LN2 = f"L{l}_LN2"
            g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            g.add_edge(nid_Add1, nid_LN2)

            expert_outputs: Dict[int, str] = {}
            for rank, e in enumerate(selected_experts):
                expert_outputs[int(e)] = self._add_selected_expert_ffn(
                    g,
                    l=l,
                    shape=shape,
                    dtype_bytes=float(dtype_bytes),
                    base_attr=base_attr,
                    ln_nid=nid_LN2,
                    expert_id=int(e),
                    expert_rank=int(rank),
                    shard_ids=[int(s) for s in shards_by_expert.get(int(e), [0])],
                    cfg=cfg,
                )

            nid_router = f"L{l}_Router"
            router_attr = {
                **base_attr,
                "experts": int(experts),
                "active_experts": int(active_experts),
                "top_k": int(top_k),
                "router_experts": int(experts),
                "local_experts": int(experts),
                "local_active_experts": int(active_experts),
                "local_top_k": float(top_k),
                "tp": int(tp_total),
                "tp_moe": int(tp_total),
                "tp_moe_total": int(tp_total),
                "tp_expert_ffn": int(tp_expert_ffn),
                "router_replicated": False,
            }
            g.add_node(
                TaskNode(
                    nid_router,
                    "MoE_Router",
                    flops=0.0,
                    weight_id=f"L{l}_ROUTER_W",
                    weight_size=int(router_weight_size),
                    attrs=_weight_attrs(router_attr, router_weight_elems, dtype_bytes),
                    allowed=get_op_allowed("MoE_Router"),
                )
            )
            g.add_edge(nid_LN2, nid_router)
            for eid in selected_experts:
                g.add_edge(expert_outputs[int(eid)], nid_router)

            # Residual Add2
            nid_Add2 = f"L{l}_Add2"
            g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            g.add_edge(nid_Add1, nid_Add2)
            g.add_edge(nid_router, nid_Add2)

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
