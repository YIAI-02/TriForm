"""Model definitions and graph builders for supported transformer families."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
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
    # Some newer architectures decouple attention head width from
    # hidden_size / num_attention_heads. DeepSeek-V4 uses very wide
    # sparse-attention heads, so the parser may set this override.
    head_dim_override: Optional[int] = None

    @property
    def head_dim(self) -> int:
        if self.head_dim_override is not None:
            return int(self.head_dim_override)
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

def _normalize_kv_partition_dim_token(v: Optional[str], *, default: str = "kv_head") -> str:
    """Normalize the KV/PIM partitioning mode used by graph construction.

    `seq` means context/KV-length block sharding: every attention layer emits one
    QK/Softmax/SV lane per sequence shard so multiple PIMs can work on the same
    layer concurrently.
    """
    s = str(v or default or "kv_head").strip().lower().replace("-", "_")
    if s in ("seq", "sequence", "context", "ctx", "kv_seq", "kv_block", "context_block"):
        return "seq"
    if s in ("layer", "layers", "layer_rr", "layer_round_robin"):
        return "layer"
    if s in ("head", "heads", "kvhead", "kv_head", "kv_heads"):
        return "kv_head"
    return str(default or "kv_head")


def _cfg_list_of_strings(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v if str(x)]
    return [x for x in str(v).replace(",", " ").split() if x]


def _infer_pim_names_from_cfg(cfg: Optional[Dict]) -> List[str]:
    """Best-effort PIM-name inference for graph expansion.

    The scheduler has the authoritative Cluster object later, but DeepSeek-V4
    graph construction needs to know whether to materialize context-sharded
    attention lanes.  Evaluation fills cfg['hardware_json']; for standalone use we
    also honor explicit pim_names/_pim_device_names.
    """
    cfg = cfg or {}
    for key in ("_pim_device_names", "pim_device_names", "pim_names"):
        xs = _cfg_list_of_strings(cfg.get(key))
        if xs:
            return xs

    hw_path = cfg.get("hardware_json") or cfg.get("hardware_config") or cfg.get("hardware")
    if not hw_path:
        return []
    try:
        path = Path(str(hw_path)).expanduser()
        if not path.is_file():
            alt = Path.cwd() / str(hw_path)
            if alt.is_file():
                path = alt
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        hw = data.get("hardware", data) if isinstance(data, dict) else {}
        devices = hw.get("devices", []) if isinstance(hw, dict) else []
        out = []
        for d in devices:
            if isinstance(d, dict) and str(d.get("type", "")).lower() == "pim":
                out.append(str(d.get("name", f"PIM{len(out)}")))
        return out
    except Exception:
        return []


def _deepseek_v4_seq_shards_from_cfg(cfg: Optional[Dict]) -> int:
    cfg = cfg or {}
    for key in ("kv_seq_shards", "attention_seq_shards", "deepseek_kv_seq_shards"):
        if cfg.get(key) is not None:
            try:
                return max(1, int(cfg.get(key) or 1))
            except Exception:
                pass
    return max(1, len(_infer_pim_names_from_cfg(cfg)))


def _deepseek_v4_partition_dim_from_cfg(cfg: Optional[Dict], *, n_pim: int) -> str:
    cfg = cfg or {}
    raw = cfg.get("kv_partition_dim", cfg.get("deepseek_kv_partition_dim", None))
    if raw is None:
        # For DeepSeek-V4/shared-KV, sequence/context sharding is the only mode
        # here that creates true same-layer PIM concurrency.
        return "seq" if int(n_pim or 0) > 1 else "kv_head"
    return _normalize_kv_partition_dim_token(raw, default=("seq" if int(n_pim or 0) > 1 else "kv_head"))


def _partition_fraction_attrs(shard: int, shards: int) -> Dict:
    shards = max(1, int(shards or 1))
    shard = max(0, min(int(shard), shards - 1))
    frac = 1.0 / float(shards)
    return {
        "kv_partition_dim": "seq",
        "kv_seq_shard": int(shard),
        "kv_seq_shards": int(shards),
        "kv_seq_fraction": float(frac),
        "attention_partition_fraction": float(frac),
        "kv_seq_start_frac": float(shard) / float(shards),
        "kv_seq_end_frac": float(shard + 1) / float(shards),
    }

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

__all__ = [name for name in globals() if not name.startswith("__")]

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

__all__ = [name for name in globals() if not name.startswith("__")]

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

    @staticmethod
    def _add_routed_moe_combine(
        g: TaskGraph,
        *,
        l: int,
        router_nid: str,
        expert_outputs: Dict[int, str],
        base_attr: Dict,
        active_routed: int,
        top_k: int,
    ) -> str:
        """Combine routed expert outputs using router scores.

        The router itself must execute before routed experts because it produces
        the top-k selection/gating scores.  The combine node is the true sink
        that waits for both router scores and expert FFN outputs.
        """

        nid = f"L{l}_RoutedMoE_Combine"
        combine_attr = dict(base_attr)
        combine_attr.update(
            {
                "shared_experts": 0,
                "active_routed_experts": int(active_routed),
                "active_experts": int(active_routed),
                "top_k": int(top_k),
                "combine_inputs": int(max(1, top_k)),
            }
        )
        g.add_node(
            TaskNode(
                nid,
                "MOE_COMBINE",
                flops=0.0,
                attrs=combine_attr,
                allowed=get_op_allowed("MOE_COMBINE"),
            )
        )
        g.add_edge(router_nid, nid)
        for out_nid in expert_outputs.values():
            g.add_edge(out_nid, nid)
        return nid

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

            expert_outputs: Dict[int, str] = {}
            for rank, e in enumerate(selected_experts):
                expert_outputs[int(e)] = self._add_selected_expert_ffn(
                    g,
                    l=l,
                    shape=shape,
                    dtype_bytes=float(dtype_bytes),
                    base_attr=base_attr,
                    ln_nid=nid_router,
                    expert_id=int(e),
                    expert_rank=int(rank),
                    shard_ids=[int(s) for s in shards_by_expert.get(int(e), [0])],
                    cfg=cfg,
                )

            moe_out = self._add_routed_moe_combine(
                g,
                l=l,
                router_nid=nid_router,
                expert_outputs=expert_outputs,
                base_attr=base_attr,
                active_routed=int(active_experts),
                top_k=int(top_k),
            )

            # Residual Add2
            nid_Add2 = f"L{l}_Add2"
            g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            g.add_edge(nid_Add1, nid_Add2)
            g.add_edge(moe_out, nid_Add2)

        return g


class DeepSeekV4Def(MixtralDef):
    """DeepSeek-V4 graph builder.

    This builder extends the existing DOPS transformer abstraction with the
    operators that make DeepSeek-V4 distinct from Mixtral/LLaMA-style models:
    compressed sparse attention (CSA), hierarchical compressed attention (HCA),
    a sliding-window branch, mHC residual mixing, and DeepSeekMoE with always-on
    shared experts plus top-k routed experts.
    """

    name = "deepseek_v4"

    @staticmethod
    def _shape_int(shape: ModelShape, key: str, default: int) -> int:
        return int(getattr(shape, key, default) or default)

    @staticmethod
    def _attention_kind(layer: int, shape: ModelShape) -> str:
        ratios = getattr(shape, "compress_ratios", None)
        if isinstance(ratios, (list, tuple)) and int(layer) < len(ratios):
            try:
                r = int(ratios[int(layer)])
                if r <= 1:
                    return "sliding"
                if r == int(getattr(shape, "csa_compression_rate", 4) or 4):
                    return "csa"
                if r == int(getattr(shape, "hca_compression_rate", 128) or 128):
                    return "hca"
                return "csa" if r < 64 else "hca"
            except Exception:
                pass

        schedule = getattr(shape, "attention_schedule", None)
        if isinstance(schedule, (list, tuple)) and int(layer) < len(schedule):
            item = str(schedule[int(layer)]).strip().lower()
            if item in ("csa", "hca", "sliding", "window", "local"):
                return "sliding" if item in ("window", "local") else item

        variant = str(getattr(shape, "deepseek_v4_variant", getattr(shape, "model_variant", "pro"))).lower()
        first_sliding = DeepSeekV4Def._shape_int(
            shape,
            "first_sliding_attention_layers",
            2 if variant == "flash" else 0,
        )
        first_hca = DeepSeekV4Def._shape_int(
            shape,
            "first_hca_attention_layers",
            2 if variant == "pro" else 0,
        )
        l = int(layer)
        if l < first_sliding:
            return "sliding"
        if l < first_hca:
            return "hca"
        offset = max(int(first_sliding), int(first_hca))
        return "csa" if ((l - offset) % 2 == 0) else "hca"

    @staticmethod
    def _add_mhc_mix(
        g: TaskGraph,
        *,
        l: int,
        tag: str,
        input_nid: Optional[str],
        shape: ModelShape,
        dtype_bytes: float,
        base_attr: Dict,
    ) -> str:
        nhc = max(1, int(getattr(shape, "mhc_expansion_factor", getattr(shape, "m_hc_expansion_factor", 1)) or 1))
        if nhc <= 1:
            if input_nid is None:
                # Degenerate mHC-disabled first-layer input: keep an explicit source-like node.
                nid = f"L{l}_mHC_{tag}_Identity"
                g.add_node(TaskNode(nid, "MHC_MIX", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("MHC_MIX")))
                return nid
            return input_nid
        dim = int(shape.dim)
        sinkhorn_iters = int(getattr(shape, "sinkhorn_iters", 20) or 20)
        # DOPS keeps a single logical residual stream.  The node below models the
        # additional token-wise mixing/projection work introduced by mHC without
        # expanding every downstream tensor by n_hc.
        weight_elems = int(dim * nhc * nhc + 3 * dim * nhc)
        nid = f"L{l}_mHC_{tag}"
        attr = dict(base_attr)
        attr.update(
            {
                "mhc_expansion_factor": int(nhc),
                "sinkhorn_iters": int(sinkhorn_iters),
                "op_detail": str(tag),
                "input_dim": int(dim),
                "output_dim": int(dim),
            }
        )
        g.add_node(
            TaskNode(
                nid,
                "MHC_MIX",
                flops=0.0,
                weight_id=f"L{l}_MHC_{tag}_W",
                weight_size=_weight_bytes(weight_elems, dtype_bytes),
                attrs=_weight_attrs(attr, weight_elems, dtype_bytes),
                allowed=get_op_allowed("MHC_MIX"),
            )
        )
        if input_nid is not None:
            g.add_edge(input_nid, nid)
        return nid

    @staticmethod
    def _add_shared_expert_ffn(
        g: TaskGraph,
        *,
        l: int,
        shape: ModelShape,
        dtype_bytes: float,
        base_attr: Dict,
        ln_nid: str,
        shared_id: int,
    ) -> str:
        dim = int(shape.dim)
        ffn = int(shape.ffn_dim)
        sid = int(shared_id)
        expert_attr = dict(base_attr)
        expert_attr.update(
            {
                "expert": int(-(sid + 1)),
                "shared_expert_id": int(sid),
                "expert_shared": True,
                "expert_active": True,
                "moe_token_fraction": 1.0,
                "tp_ffn": 1,
                "tp_expert_ffn": 1,
                "ffn_dim": int(ffn),
            }
        )

        nid_W1 = f"L{l}_Shared{sid}_FFN_W1"
        nid_W3 = f"L{l}_Shared{sid}_FFN_W3"
        nid_ACT = f"L{l}_Shared{sid}_SwiGLU"
        nid_W2 = f"L{l}_Shared{sid}_FFN_W2"

        g.add_node(
            TaskNode(
                nid_W1,
                "FFN_W1",
                flops=0.0,
                weight_id=f"L{l}_SHARED{sid}_W1",
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
                weight_id=f"L{l}_SHARED{sid}_W3",
                weight_size=_weight_bytes(dim * ffn, dtype_bytes),
                attrs=_weight_attrs(expert_attr, dim * ffn, dtype_bytes),
                allowed=get_op_allowed("FFN_W3"),
            )
        )
        g.add_node(TaskNode(nid_ACT, "SwiGLU", flops=0.0, attrs=dict(expert_attr), allowed=get_op_allowed("SwiGLU")))
        g.add_node(
            TaskNode(
                nid_W2,
                "FFN_W2",
                flops=0.0,
                weight_id=f"L{l}_SHARED{sid}_W2",
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

    @staticmethod
    def _add_deepseek_v4_attention(
        g: TaskGraph,
        *,
        l: int,
        shape: ModelShape,
        dtype_bytes: float,
        base_attr: Dict,
        ln_nid: str,
        x_in: Optional[str],
        add1_nid: str,
        kind: str,
        cfg: Optional[Dict] = None,
    ) -> Dict[str, str]:
        dim = int(shape.dim)
        qh = int(shape.n_heads)
        hd = int(shape.head_dim)
        dc = int(getattr(shape, "query_compression_dim", 1024) or 1024)
        rope_hd = int(getattr(shape, "qk_rope_head_dim", getattr(shape, "rope_head_dim", 64)) or 64)
        rope_hd = max(0, min(int(rope_hd), int(hd)))
        nope_hd = max(0, int(hd) - int(rope_hd))
        csa_m = max(1, int(getattr(shape, "csa_compression_rate", 4) or 4))
        hca_m = max(1, int(getattr(shape, "hca_compression_rate", 128) or 128))
        window = max(0, int(getattr(shape, "sliding_window", 128) or 128))
        csa_top_k = max(1, int(getattr(shape, "csa_top_k", getattr(shape, "index_topk", 512)) or 512))
        index_heads = int(getattr(shape, "indexer_heads", 64) or 64)
        index_head_dim = int(getattr(shape, "indexer_head_dim", 128) or 128)
        groups = max(1, int(getattr(shape, "output_projection_groups", 8) or 8))
        group_dim = int(getattr(shape, "group_output_dim", 1024) or 1024)

        q_out_dim = int(qh * hd)
        attn_kind = str(kind).lower()
        pattern = "deepseek_csa" if attn_kind == "csa" else ("deepseek_hca" if attn_kind == "hca" else "local")
        compression_rate = csa_m if attn_kind == "csa" else (hca_m if attn_kind == "hca" else 1)
        compressor_coff = 2 if int(compression_rate) == 4 and attn_kind != "sliding" else 1

        attn_attr = dict(base_attr)
        attn_attr.update(
            {
                "attention_kind": attn_kind,
                "attention_pattern": pattern,
                "attention_sparsity": {
                    "pattern": pattern,
                    "compression_rate": int(compression_rate),
                    "top_k": int(csa_top_k),
                    "sliding_window": int(window),
                    "window_left": int(max(0, window - 1)),
                    "window_right": 0,
                },
                "q_heads": int(qh),
                "kv_heads": 1,
                "n_heads": int(qh),
                "n_kv_heads": 1,
                "head_dim": int(hd),
                "qk_dim": int(hd),
                "value_dim": int(hd),
                "kv_cache_dim": int(hd),
                "q_lora_rank": int(dc),
                "qk_rope_head_dim": int(rope_hd),
                "qk_nope_head_dim": int(nope_hd),
                "rope_head_dim": int(rope_hd),
                "q_dim": int(q_out_dim),
                "kv_dim": int(hd),
                "o_dim": int(q_out_dim),
                "shared_kv": True,
                "kv_cache_shared": True,
                "kv_cache_vectors_per_entry": 1,
                "query_compression_dim": int(dc),
                "csa_compression_rate": int(csa_m),
                "hca_compression_rate": int(hca_m),
                "sliding_window": int(window),
                "csa_top_k": int(csa_top_k),
                "indexer_heads": int(index_heads),
                "indexer_head_dim": int(index_head_dim),
                "output_projection_groups": int(groups),
                "group_output_dim": int(group_dim),
            }
        )

        # Down/up query projection.  Q-down is shared by the CSA indexer path.
        nid_QD = f"L{l}_DSV4_Q_Down"
        qd_attr = dict(attn_attr)
        qd_attr.update({"dim": int(dim), "input_dim": int(dim), "output_dim": int(dc), "q_dim": int(dc)})
        g.add_node(
            TaskNode(
                nid_QD,
                "DSV4_Q_DOWN",
                flops=0.0,
                weight_id=f"L{l}_DSV4_Q_DOWN_W",
                weight_size=_weight_bytes(dim * dc, dtype_bytes),
                attrs=_weight_attrs(qd_attr, dim * dc, dtype_bytes),
                allowed=get_op_allowed("DSV4_Q_DOWN"),
            )
        )

        nid_QU = f"L{l}_DSV4_Q_Up"
        qu_attr = dict(attn_attr)
        qu_attr.update({"dim": int(dc), "input_dim": int(dc), "output_dim": int(q_out_dim), "q_dim": int(q_out_dim)})
        g.add_node(
            TaskNode(
                nid_QU,
                "DSV4_Q_UP",
                flops=0.0,
                weight_id=f"L{l}_DSV4_Q_UP_W",
                weight_size=_weight_bytes(dc * q_out_dim, dtype_bytes),
                attrs=_weight_attrs(qu_attr, dc * q_out_dim, dtype_bytes),
                allowed=get_op_allowed("DSV4_Q_UP"),
            )
        )
        g.add_edge(ln_nid, nid_QD)
        g.add_edge(nid_QD, nid_QU)

        # DeepSeek-V4 attention uses a single shared K/V vector.  Every layer
        # produces an uncompressed short-window KV stream; compressed layers add
        # a learned compressor stream on top of it.  This is different from a
        # LLaMA/Mixtral K+V projection and must not be modeled as 2*head_dim.
        nid_WIN_KV = f"L{l}_DSV4_Window_KV"
        win_kv_weight = int(dim * hd)
        win_kv_attr = dict(attn_attr)
        win_kv_attr.update(
            {
                "input_dim": int(dim),
                "output_dim": int(hd),
                "kv_dim": int(hd),
                "kv_cache_dim": int(hd),
                "compression_rate": 1,
                "compressor_mode": "window",
                "shared_kv": True,
                "kv_cache_vectors_per_entry": 1,
            }
        )
        g.add_node(
            TaskNode(
                nid_WIN_KV,
                "DSV4_WINDOW_KV",
                flops=0.0,
                weight_id=f"L{l}_DSV4_WINDOW_KV_W",
                weight_size=_weight_bytes(win_kv_weight, dtype_bytes),
                attrs=_weight_attrs(win_kv_attr, win_kv_weight, dtype_bytes),
                allowed=get_op_allowed("DSV4_WINDOW_KV"),
            )
        )
        g.add_edge(ln_nid, nid_WIN_KV)

        kv_sources = [nid_WIN_KV]
        if attn_kind != "sliding":
            nid_KV = f"L{l}_DSV4_{attn_kind.upper()}_KV_Compress"
            proj_dim = int(compressor_coff * hd)
            # Compressor has wkv and wgate, both dim -> coff*head_dim, plus
            # learned APE/gating parameters for each token within the block.
            kv_weight = int(2 * dim * proj_dim + compression_rate * proj_dim)
            kv_attr = dict(attn_attr)
            kv_attr.update(
                {
                    "input_dim": int(dim),
                    "output_dim": int(hd),
                    "projected_dim": int(proj_dim),
                    "kv_dim": int(hd),
                    "kv_cache_dim": int(hd),
                    "compression_rate": int(compression_rate),
                    "compressor_coff": int(compressor_coff),
                    "compressor_mode": attn_kind,
                    "shared_kv": True,
                    "kv_cache_vectors_per_entry": 1,
                }
            )
            g.add_node(
                TaskNode(
                    nid_KV,
                    "DSV4_KV_COMPRESS",
                    flops=0.0,
                    weight_id=f"L{l}_DSV4_{attn_kind.upper()}_KV_COMPRESS_W",
                    weight_size=_weight_bytes(kv_weight, dtype_bytes),
                    attrs=_weight_attrs(kv_attr, kv_weight, dtype_bytes),
                    allowed=get_op_allowed("DSV4_KV_COMPRESS"),
                )
            )
            g.add_edge(ln_nid, nid_KV)
            kv_sources.append(nid_KV)

        kvw_attr = dict(attn_attr)
        kvw_attr.update(
            {
                "compression_rate": int(compression_rate),
                "kv_cache_dim": int(hd),
                "kv_cache_shared": True,
                "shared_kv": True,
                "kv_cache_vectors_per_entry": 1,
                "write_window_cache": True,
                "write_compressed_cache": bool(attn_kind != "sliding"),
            }
        )

        pim_names = _infer_pim_names_from_cfg(cfg)
        partition_dim = _deepseek_v4_partition_dim_from_cfg(cfg, n_pim=len(pim_names))
        seq_shards = _deepseek_v4_seq_shards_from_cfg(cfg) if partition_dim == "seq" else 1
        # Keep the graph size bounded by the available PIM lanes unless the user
        # explicitly asks for more sequence shards.
        if partition_dim == "seq" and len(pim_names) > 0:
            seq_shards = max(1, int(seq_shards))
        else:
            seq_shards = 1
        use_seq_sharded_attention = bool(partition_dim == "seq" and seq_shards > 1)

        kv_write_ids: List[str] = []
        if use_seq_sharded_attention:
            for si in range(int(seq_shards)):
                nid = f"L{l}_KV_write_seq{si}"
                shard_attr = dict(kvw_attr)
                shard_attr.update(_partition_fraction_attrs(si, seq_shards))
                g.add_node(TaskNode(nid, "KV_WRITE", flops=0.0, attrs=shard_attr, allowed=get_op_allowed("KV_WRITE")))
                for kv_source in kv_sources:
                    g.add_edge(kv_source, nid)
                kv_write_ids.append(nid)
        else:
            nid = f"L{l}_KV_write"
            g.add_node(TaskNode(nid, "KV_WRITE", flops=0.0, attrs=dict(kvw_attr), allowed=get_op_allowed("KV_WRITE")))
            for kv_source in kv_sources:
                g.add_edge(kv_source, nid)
            kv_write_ids.append(nid)
        nid_KVW = kv_write_ids[0]

        topk_source: Optional[str] = None
        if attn_kind == "csa":
            nid_IQ = f"L{l}_DSV4_CSA_Indexer_Q"
            iq_weight = int(dc * index_heads * index_head_dim + dim * index_heads)
            iq_attr = dict(attn_attr)
            iq_attr.update(
                {
                    "input_dim": int(dc),
                    "output_dim": int(index_heads * index_head_dim),
                    "q_dim": int(index_heads * index_head_dim),
                    "aux_input_dim": int(dim),
                    "aux_output_dim": int(index_heads),
                    "indexer_projection": True,
                }
            )
            g.add_node(
                TaskNode(
                    nid_IQ,
                    "DSV4_INDEXER_Q",
                    flops=0.0,
                    weight_id=f"L{l}_DSV4_CSA_INDEXER_Q_W",
                    weight_size=_weight_bytes(iq_weight, dtype_bytes),
                    attrs=_weight_attrs(iq_attr, iq_weight, dtype_bytes),
                    allowed=get_op_allowed("DSV4_INDEXER_Q"),
                )
            )
            g.add_edge(nid_QD, nid_IQ)

            nid_IKV = f"L{l}_DSV4_CSA_Indexer_KV_Compress"
            idx_coff = 2 if int(csa_m) == 4 else 1
            idx_proj_dim = int(idx_coff * index_head_dim)
            idx_kv_weight = int(2 * dim * idx_proj_dim + csa_m * idx_proj_dim)
            ikv_attr = dict(attn_attr)
            ikv_attr.update(
                {
                    "input_dim": int(dim),
                    "output_dim": int(index_head_dim),
                    "projected_dim": int(idx_proj_dim),
                    "kv_dim": int(index_head_dim),
                    "kv_cache_dim": int(index_head_dim),
                    "head_dim": int(index_head_dim),
                    "qk_dim": int(index_head_dim),
                    "compression_rate": int(csa_m),
                    "compressor_coff": int(idx_coff),
                    "compressor_mode": "csa_indexer",
                    "indexer_projection": True,
                    "indexer_kv_cache": True,
                    "shared_kv": True,
                    "kv_cache_vectors_per_entry": 1,
                }
            )
            g.add_node(
                TaskNode(
                    nid_IKV,
                    "DSV4_INDEX_KV_COMPRESS",
                    flops=0.0,
                    weight_id=f"L{l}_DSV4_CSA_INDEXER_KV_COMPRESS_W",
                    weight_size=_weight_bytes(idx_kv_weight, dtype_bytes),
                    attrs=_weight_attrs(ikv_attr, idx_kv_weight, dtype_bytes),
                    allowed=get_op_allowed("DSV4_INDEX_KV_COMPRESS"),
                )
            )
            g.add_edge(ln_nid, nid_IKV)

            nid_IS = f"L{l}_DSV4_CSA_Index_Score"
            is_attr = dict(attn_attr)
            is_attr.update({"indexer_heads": int(index_heads), "indexer_head_dim": int(index_head_dim)})
            g.add_node(TaskNode(nid_IS, "DSV4_INDEX_SCORE", flops=0.0, attrs=dict(is_attr), allowed=get_op_allowed("DSV4_INDEX_SCORE")))
            g.add_edge(nid_IQ, nid_IS)
            g.add_edge(nid_IKV, nid_IS)

            nid_TK = f"L{l}_DSV4_CSA_TopK"
            tk_attr = dict(attn_attr)
            tk_attr.update({"top_k": int(csa_top_k), "compression_rate": int(csa_m)})
            g.add_node(TaskNode(nid_TK, "DSV4_TOPK", flops=0.0, attrs=dict(tk_attr), allowed=get_op_allowed("DSV4_TOPK")))
            g.add_edge(nid_IS, nid_TK)
            topk_source = nid_TK

        # Sparse/dense-on-compressed attention core.  For DeepSeek-V4/shared-KV,
        # head sharding is ineffective because n_kv_heads == 1.  Sequence/context
        # sharding materializes one QK/Softmax/SV lane per KV block, so all PIMs
        # can work on the same layer concurrently.  The two ALLREDUCE nodes model
        # the global softmax statistics and the final partial-value accumulation.
        if use_seq_sharded_attention:
            qk_ids: List[str] = []
            sm_ids: List[str] = []
            sv_ids: List[str] = []
            for si, kvw_id in enumerate(kv_write_ids):
                shard_attr = dict(attn_attr)
                shard_attr.update(_partition_fraction_attrs(si, seq_shards))
                nid_QK_i = f"L{l}_QK_seq{si}"
                nid_SM_i = f"L{l}_Softmax_seq{si}"
                nid_SV_i = f"L{l}_SV_seq{si}"
                g.add_node(TaskNode(nid_QK_i, "QK", flops=0.0, attrs=dict(shard_attr), allowed=get_op_allowed("QK")))
                g.add_node(TaskNode(nid_SM_i, "Softmax", flops=0.0, attrs=dict(shard_attr), allowed=get_op_allowed("Softmax")))
                g.add_node(TaskNode(nid_SV_i, "SV", flops=0.0, attrs=dict(shard_attr), allowed=get_op_allowed("SV")))
                g.add_edge(nid_QU, nid_QK_i)
                g.add_edge(kvw_id, nid_QK_i)
                if topk_source is not None:
                    g.add_edge(topk_source, nid_QK_i)
                qk_ids.append(nid_QK_i)
                sm_ids.append(nid_SM_i)
                sv_ids.append(nid_SV_i)

            stats_attr = dict(attn_attr)
            stats_attr.update({
                "primitive": "allreduce",
                "kv_partition_dim": "seq",
                "collective_role": "deepseek_v4_softmax_stats",
                "dim": int(2 * qh),
                "q_dim": int(2 * qh),
                "o_dim": int(2 * qh),
            })
            nid_SM_STATS = f"L{l}_AllReduce_DSV4_SoftmaxStats"
            g.add_node(TaskNode(nid_SM_STATS, "ALLREDUCE", flops=0.0, attrs=stats_attr, allowed=get_op_allowed("ALLREDUCE")))
            for nid_QK_i in qk_ids:
                g.add_edge(nid_QK_i, nid_SM_STATS)

            for si, (nid_QK_i, nid_SM_i, nid_SV_i, kvw_id) in enumerate(zip(qk_ids, sm_ids, sv_ids, kv_write_ids)):
                g.add_edge(nid_QK_i, nid_SM_i)
                g.add_edge(nid_SM_STATS, nid_SM_i)
                g.add_edge(nid_SM_i, nid_SV_i)
                g.add_edge(kvw_id, nid_SV_i)
                if topk_source is not None:
                    g.add_edge(topk_source, nid_SV_i)

            out_attr = dict(attn_attr)
            out_attr.update({
                "primitive": "allreduce",
                "kv_partition_dim": "seq",
                "collective_role": "deepseek_v4_attention_output",
                "dim": int(q_out_dim),
                "q_dim": int(q_out_dim),
                "o_dim": int(q_out_dim),
            })
            nid_ATTN_OUT = f"L{l}_AllReduce_DSV4_AttnOut"
            g.add_node(TaskNode(nid_ATTN_OUT, "ALLREDUCE", flops=0.0, attrs=out_attr, allowed=get_op_allowed("ALLREDUCE")))
            for nid_SV_i in sv_ids:
                g.add_edge(nid_SV_i, nid_ATTN_OUT)

            nid_QK = qk_ids[0]
            nid_SM = nid_SM_STATS
            nid_SV = nid_ATTN_OUT
            attn_out_nid = nid_ATTN_OUT
        else:
            nid_QK = f"L{l}_QK"
            nid_SM = f"L{l}_Softmax"
            nid_SV = f"L{l}_SV"
            g.add_node(TaskNode(nid_QK, "QK", flops=0.0, attrs=dict(attn_attr), allowed=get_op_allowed("QK")))
            g.add_node(TaskNode(nid_SM, "Softmax", flops=0.0, attrs=dict(attn_attr), allowed=get_op_allowed("Softmax")))
            g.add_node(TaskNode(nid_SV, "SV", flops=0.0, attrs=dict(attn_attr), allowed=get_op_allowed("SV")))
            g.add_edge(nid_QU, nid_QK)
            g.add_edge(nid_KVW, nid_QK)
            if topk_source is not None:
                g.add_edge(topk_source, nid_QK)
            g.add_edge(nid_QK, nid_SM)
            g.add_edge(nid_SM, nid_SV)
            g.add_edge(nid_KVW, nid_SV)
            if topk_source is not None:
                g.add_edge(topk_source, nid_SV)
            attn_out_nid = nid_SV

        nid_OG1 = f"L{l}_DSV4_O_Group"
        group_input_dim = int(max(1, (qh // groups)) * hd)
        og1_weight = int(groups * group_input_dim * group_dim)
        og1_attr = dict(attn_attr)
        og1_attr.update(
            {
                "input_dim": int(q_out_dim),
                "output_dim": int(groups * group_dim),
                "group_input_dim": int(group_input_dim),
                "group_output_dim": int(group_dim),
                "groups": int(groups),
                "grouped_linear": True,
                "o_dim": int(q_out_dim),
                "dim": int(groups * group_dim),
            }
        )
        g.add_node(
            TaskNode(
                nid_OG1,
                "DSV4_O_G1",
                flops=0.0,
                weight_id=f"L{l}_DSV4_O_G1_W",
                weight_size=_weight_bytes(og1_weight, dtype_bytes),
                attrs=_weight_attrs(og1_attr, og1_weight, dtype_bytes),
                allowed=get_op_allowed("DSV4_O_G1"),
            )
        )

        nid_OG2 = f"L{l}_DSV4_O_Final"
        og2_in = int(groups * group_dim)
        og2_weight = int(og2_in * dim)
        og2_attr = dict(attn_attr)
        og2_attr.update({"input_dim": int(og2_in), "output_dim": int(dim), "o_dim": int(og2_in), "dim": int(dim)})
        g.add_node(
            TaskNode(
                nid_OG2,
                "DSV4_O_G2",
                flops=0.0,
                weight_id=f"L{l}_DSV4_O_G2_W",
                weight_size=_weight_bytes(og2_weight, dtype_bytes),
                attrs=_weight_attrs(og2_attr, og2_weight, dtype_bytes),
                allowed=get_op_allowed("DSV4_O_G2"),
            )
        )
        g.add_edge(attn_out_nid, nid_OG1)
        g.add_edge(nid_OG1, nid_OG2)

        # Attention produces a standard sub-layer output.  mHC write-back is modeled
        # after the residual add in build(); placing it here would make the mix consume
        # only the attention branch and would incorrectly execute before Add1.
        g.add_edge(nid_OG2, add1_nid)
        if x_in is not None:
            g.add_edge(x_in, add1_nid)

        return {
            "q_down": nid_QD,
            "q_up": nid_QU,
            "kv": kv_sources[-1] if kv_sources else None,
            "kv_write": nid_KVW,
            "kv_writes": list(kv_write_ids),
            "qk": nid_QK,
            "softmax": nid_SM,
            "sv": nid_SV,
            "o_group": nid_OG1,
            "o_final": nid_OG2,
        }

    def build(self, shape: ModelShape, dtype_bytes: float, cfg: Optional[Dict] = None) -> TaskGraph:
        g = TaskGraph()

        routed_experts = int(
            getattr(shape, "routed_experts_per_layer", getattr(shape, "experts_per_layer", 1)) or 1
        )
        shared_experts = int(getattr(shape, "shared_experts_per_layer", 1) or 0)
        top_k = self._resolve_top_k(routed_experts, int(getattr(shape, "experts_top_k", 6) or 6))
        selected_experts = self._select_first_k_experts(routed_experts, top_k)
        active_routed = int(len(selected_experts))
        active_total = int(active_routed + max(0, shared_experts))
        setattr(shape, "active_routed_experts_per_layer", int(active_routed))
        setattr(shape, "active_experts_per_layer", int(active_total))
        setattr(shape, "moe_pruned_experts_per_layer", max(0, int(routed_experts - active_routed)))

        b = int(shape.batch)
        dim, ffn = int(shape.dim), int(shape.ffn_dim)
        qh, hd = int(shape.n_heads), int(shape.head_dim)
        q_dim, kv_dim, o_in_dim = int(qh * hd), int(hd), int(qh * hd)
        router_weight_elems = int(dim * routed_experts)
        router_weight_size = _weight_bytes(router_weight_elems, dtype_bytes)
        total_moe_weight_elems = int((routed_experts + shared_experts) * (2 * dim * ffn + ffn * dim))

        tp_qkv = int((cfg or {}).get("tp_qkv_effective", 1) or 1)
        tp_total = int((cfg or {}).get("tp_moe_total_effective", (cfg or {}).get("tp_moe_effective", 1) or 1) or 1)
        shard_plan = self._plan_selected_expert_shards(selected_experts, tp_total)
        tp_expert_ffn = int(shard_plan["tp_expert_ffn"])
        experts_by_shard = list(shard_plan["experts_by_shard"])
        shards_by_expert = dict(shard_plan["shards_by_expert"])

        moe_imbalance = float(getattr(shape, "moe_imbalance_factor", 1.0) or 1.0)
        router_aux_loss_coef = getattr(shape, "router_aux_loss_coef", None)
        router_jitter_noise = getattr(shape, "router_jitter_noise", None)
        hash_layers = int(getattr(shape, "hash_routing_layers", 3) or 0)

        for l in range(int(shape.layer_num)):
            attn_kind = self._attention_kind(l, shape)
            base_attr = {
                "layer": int(l),
                "batch": int(b),
                "dim": int(dim),
                "ffn_dim": int(ffn),
                "q_heads": int(qh),
                "kv_heads": 1,
                "n_heads": int(qh),
                "n_kv_heads": 1,
                "head_dim": int(hd),
                "q_dim": int(q_dim),
                "kv_dim": int(kv_dim),
                "o_dim": int(o_in_dim),
                "model_family": "deepseek_v4",
                "deepseek_v4_variant": str(getattr(shape, "deepseek_v4_variant", "pro")),
                "attention_kind": str(attn_kind),
                "experts": int(routed_experts),
                "experts_total": int(routed_experts),
                "routed_experts": int(routed_experts),
                "shared_experts": int(shared_experts),
                "active_experts": int(active_total),
                "active_routed_experts": int(active_routed),
                "top_k": int(top_k),
                "num_local_experts": int(routed_experts),
                "num_experts_per_tok": int(top_k),
                "selected_experts": [int(e) for e in selected_experts],
                "selected_experts_by_shard": [[int(e) for e in xs] for xs in experts_by_shard],
                "router_kind": "deepseek_v4_sqrt_softplus_topk",
                "router_affinity": "sqrt_softplus",
                "hash_routing_active": bool(l < hash_layers),
                "hash_routing_layers": int(hash_layers),
                "moe_selection_policy": "first_k_runtime_proxy",
                "moe_imbalance_factor": float(moe_imbalance),
                "router_weight_size": int(router_weight_size),
                "moe_total_weight_elements_per_layer": int(total_moe_weight_elems),
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

            # mHC is a residual-stream read/write mix.  Keep LayerNorm inside each
            # sub-layer: residual_in -> mHC_Pre* -> LN -> branch -> Add -> mHC_Post*.
            layer_input_nid = f"L{l-1}_mHC_PostMoE" if l > 0 else None
            nid_ATT_IN = self._add_mhc_mix(
                g,
                l=l,
                tag="PreAttn",
                input_nid=layer_input_nid,
                shape=shape,
                dtype_bytes=float(dtype_bytes),
                base_attr=base_attr,
            )

            nid_LN1 = f"L{l}_LN1"
            g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            g.add_edge(nid_ATT_IN, nid_LN1)

            nid_Add1 = f"L{l}_Add1"
            g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            self._add_deepseek_v4_attention(
                g,
                l=l,
                shape=shape,
                dtype_bytes=float(dtype_bytes),
                base_attr=base_attr,
                ln_nid=nid_LN1,
                x_in=nid_ATT_IN,
                add1_nid=nid_Add1,
                kind=attn_kind,
                cfg=cfg,
            )

            nid_POST_ATTN = self._add_mhc_mix(
                g,
                l=l,
                tag="PostAttn",
                input_nid=nid_Add1,
                shape=shape,
                dtype_bytes=float(dtype_bytes),
                base_attr=base_attr,
            )

            # DeepSeekMoE block: top-k routed experts plus shared experts.
            nid_MOE_IN = self._add_mhc_mix(
                g,
                l=l,
                tag="PreMoE",
                input_nid=nid_POST_ATTN,
                shape=shape,
                dtype_bytes=float(dtype_bytes),
                base_attr=base_attr,
            )

            nid_LN2 = f"L{l}_LN2"
            g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            g.add_edge(nid_MOE_IN, nid_LN2)

            shared_outputs: List[str] = []
            for sid in range(max(0, int(shared_experts))):
                shared_outputs.append(
                    self._add_shared_expert_ffn(
                        g,
                        l=l,
                        shape=shape,
                        dtype_bytes=float(dtype_bytes),
                        base_attr=base_attr,
                        ln_nid=nid_LN2,
                        shared_id=int(sid),
                    )
                )

            nid_router = f"L{l}_Router"
            router_attr = {
                **base_attr,
                "experts": int(routed_experts),
                "active_experts": int(active_routed),
                "active_routed_experts": int(active_routed),
                "shared_experts": int(shared_experts),
                "top_k": int(top_k),
                "router_experts": int(routed_experts),
                "local_experts": int(routed_experts),
                "local_active_experts": int(active_routed),
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

            expert_outputs: Dict[int, str] = {}
            for rank, e in enumerate(selected_experts):
                expert_outputs[int(e)] = self._add_selected_expert_ffn(
                    g,
                    l=l,
                    shape=shape,
                    dtype_bytes=float(dtype_bytes),
                    base_attr=base_attr,
                    ln_nid=nid_router,
                    expert_id=int(e),
                    expert_rank=int(rank),
                    shard_ids=[int(s) for s in shards_by_expert.get(int(e), [0])],
                    cfg=cfg,
                )

            moe_out = self._add_routed_moe_combine(
                g,
                l=l,
                router_nid=nid_router,
                expert_outputs=expert_outputs,
                base_attr=base_attr,
                active_routed=int(active_routed),
                top_k=int(top_k),
            )
            if shared_outputs:
                nid_combine = f"L{l}_SharedMoE_Combine"
                combine_attr = dict(base_attr)
                combine_attr.update({"shared_experts": int(shared_experts), "active_routed_experts": int(active_routed), "combine_inputs": int(shared_experts + 1)})
                g.add_node(TaskNode(nid_combine, "MOE_SHARED_COMBINE", flops=0.0, attrs=combine_attr, allowed=get_op_allowed("MOE_SHARED_COMBINE")))
                g.add_edge(moe_out, nid_combine)
                for so in shared_outputs:
                    g.add_edge(so, nid_combine)
                moe_out = nid_combine

            nid_Add2 = f"L{l}_Add2"
            g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            g.add_edge(nid_MOE_IN, nid_Add2)
            g.add_edge(moe_out, nid_Add2)

            self._add_mhc_mix(
                g,
                l=l,
                tag="PostMoE",
                input_nid=nid_Add2,
                shape=shape,
                dtype_bytes=float(dtype_bytes),
                base_attr=base_attr,
            )

        return g

def make_model_def(family: str):
    f = (family or "").lower().replace("-", "_")
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
    if f in ("deepseek_v4", "deepseekv4", "deepseek_v4_pro", "deepseek_v4_flash"):
        return DeepSeekV4Def()
    raise ValueError(f"Unknown model family: {family}")

__all__ = [
    "LLaMADef",
    "MPTDef",
    "MixtralDef",
    "DeepSeekV4Def",
    "ModelShape",
    "PaLMDef",
    "QwenDef",
    "add_llama_block",
    "add_mpt_block",
    "add_palm_block",
    "get_op_allowed",
    "make_model_def",
]
