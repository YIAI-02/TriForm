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


def _add_attention_llama_style_unsplit(
    g: TaskGraph,
    *,
    l: int,
    shape: ModelShape,
    dtype_bytes: int,
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
            weight_size=int(o_in_dim * dim * dtype_bytes),
            attrs=dict(base_attr),
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


# =============================================================================
# Block builders (no operator splitting)
# =============================================================================

def add_llama_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """LLaMA/Qwen style block."""

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
        g.add_edge(f"L{l-1}_K_write", nid_LN)
        g.add_edge(f"L{l-1}_V_write", nid_LN)

    # Add1 placeholder
    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
    x_in = f"L{l-1}_Add2" if l > 0 else None

    _add_attention_llama_style_unsplit(
        g,
        l=l,
        shape=shape,
        dtype_bytes=int(dtype_bytes),
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


def add_mpt_block(g: TaskGraph, l: int, shape: ModelShape, dtype_bytes: int):
    """MPT style: attention + GELU MLP (NO head-splitting)."""

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
        g.add_edge(f"L{l-1}_K_write", nid_LN1)
        g.add_edge(f"L{l-1}_V_write", nid_LN1)

    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
    x_in = f"L{l-1}_Add2" if l > 0 else None

    _add_attention_llama_style_unsplit(
        g,
        l=l,
        shape=shape,
        dtype_bytes=int(dtype_bytes),
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

    NO tensor-parallel / NO head-splitting.
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
        g.add_edge(f"L{l-1}_K_write", nid_LN)
        g.add_edge(f"L{l-1}_V_write", nid_LN)

    x_in = f"L{l-1}_Add2" if l > 0 else None

    nid_Add_Attn = f"L{l}_Add_attn"
    g.add_node(TaskNode(nid_Add_Attn, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))

    _add_attention_llama_style_unsplit(
        g,
        l=l,
        shape=shape,
        dtype_bytes=int(dtype_bytes),
        base_attr=base_attr,
        ln_nid=nid_LN,
        x_in=x_in,
        add1_nid=nid_Add_Attn,
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


# =============================================================================
# Model definitions
# =============================================================================

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
        """Mixtral MoE (NO head-splitting)."""
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

            # Attention part (LLaMA-style, LN1 here)
            nid_LN1 = f"L{l}_LN1"
            g.add_node(TaskNode(nid_LN1, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            if l > 0:
                g.add_edge(f"L{l-1}_K_write", nid_LN1)
                g.add_edge(f"L{l-1}_V_write", nid_LN1)

            nid_Add1 = f"L{l}_Add1"
            g.add_node(TaskNode(nid_Add1, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            x_in = f"L{l-1}_Add2" if l > 0 else None

            _add_attention_llama_style_unsplit(
                g,
                l=l,
                shape=shape,
                dtype_bytes=int(dtype_bytes),
                base_attr=base_attr,
                ln_nid=nid_LN1,
                x_in=x_in,
                add1_nid=nid_Add1,
            )

            # LN2
            nid_LN2 = f"L{l}_LN2"
            g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            g.add_edge(nid_Add1, nid_LN2)

            # Experts (unsplit weights). Router is not explicitly modeled here; we just add active experts.
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

            # Router (combine expert outputs; modeled as a single op)
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

            nid_MLP_OUT = nid_router

            # Residual Add2
            nid_Add2 = f"L{l}_Add2"
            g.add_node(TaskNode(nid_Add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            g.add_edge(nid_Add1, nid_Add2)
            g.add_edge(nid_MLP_OUT, nid_Add2)

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
