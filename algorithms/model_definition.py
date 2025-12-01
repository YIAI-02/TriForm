from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
from task_graph import TaskGraph, TaskNode
from config import OPERATOR_DEVICE_ALLOWED, DEFAULT_OPERATOR_ALLOWED


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
        # Default per-head dimension: dim // q_heads
        return self.dim // self.n_heads

# ---- Common helpers ----
def get_op_allowed(op_name: str) -> Dict[str, bool]:
    return OPERATOR_DEVICE_ALLOWED.get(op_name, DEFAULT_OPERATOR_ALLOWED).copy()

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
    nid_KR = f"L{l}_K_read"; nid_VR = f"L{l}_V_read"
    nid_KW = f"L{l}_K_write"; nid_VW = f"L{l}_V_write"
    g.add_node(TaskNode(nid_KR, "K_read", attrs=dict(base_attr), allowed=get_op_allowed("KV_read")))
    g.add_node(TaskNode(nid_VR, "V_read", attrs=dict(base_attr), allowed=get_op_allowed("KV_read")))
    g.add_node(TaskNode(nid_KW, "K_write", attrs=dict(base_attr), allowed=get_op_allowed("KV_write")))
    g.add_node(TaskNode(nid_VW, "V_write", attrs=dict(base_attr), allowed=get_op_allowed("KV_write")))

    # Wire connections (pre-norm, sequential)
    x_in = f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X = f"L{l}_X"; g.add_node(TaskNode(nid_X, "Identity", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Identity")))
        g.add_edge(x_in, nid_X); g.add_edge(nid_X, nid_LN1)

    g.add_edge(nid_LN1, nid_Q); g.add_edge(nid_LN1, nid_K); g.add_edge(nid_LN1, nid_V)
    g.add_edge(nid_Q, nid_QK); g.add_edge(nid_K, nid_QK)
    g.add_edge(nid_KR, nid_QK)
    g.add_edge(nid_QK, nid_SO); g.add_edge(nid_SO, nid_SV)
    g.add_edge(nid_V, nid_SV); g.add_edge(nid_VR, nid_SV)
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

    nid_KR=f"L{l}_K_read"; nid_VR=f"L{l}_V_read"; nid_KW=f"L{l}_K_write"; nid_VW=f"L{l}_V_write"
    g.add_node(TaskNode(nid_KR,"K_read",attrs=dict(base_attr),allowed=get_op_allowed("KV_read")))
    g.add_node(TaskNode(nid_VR,"V_read",attrs=dict(base_attr),allowed=get_op_allowed("KV_read")))
    g.add_node(TaskNode(nid_KW,"K_write",attrs=dict(base_attr),allowed=get_op_allowed("KV_write")))
    g.add_node(TaskNode(nid_VW,"V_write",attrs=dict(base_attr),allowed=get_op_allowed("KV_write")))

    x_in=f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"; g.add_node(TaskNode(nid_X,"Identity",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN1)

    g.add_edge(nid_LN1,nid_Q); g.add_edge(nid_LN1,nid_K); g.add_edge(nid_LN1,nid_V)
    g.add_edge(nid_Q,nid_QK); g.add_edge(nid_K,nid_QK)
    g.add_edge(nid_KR,nid_QK)
    g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV)
    g.add_edge(nid_V,nid_SV); g.add_edge(nid_VR,nid_SV)
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
    nid_KR=f"L{l}_K_read"; nid_VR=f"L{l}_V_read"; nid_KW=f"L{l}_K_write"; nid_VW=f"L{l}_V_write"
    g.add_node(TaskNode(nid_KR,"K_read",attrs=dict(base_attr),allowed=get_op_allowed("KV_read")))
    g.add_node(TaskNode(nid_VR,"V_read",attrs=dict(base_attr),allowed=get_op_allowed("KV_read")))
    g.add_node(TaskNode(nid_KW,"K_write",attrs=dict(base_attr),allowed=get_op_allowed("KV_write")))
    g.add_node(TaskNode(nid_VW,"V_write",attrs=dict(base_attr),allowed=get_op_allowed("KV_write")))

    # Wire
    x_in=f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"; g.add_node(TaskNode(nid_X,"Identity",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN)

    g.add_edge(nid_LN,nid_Q); g.add_edge(nid_LN,nid_K); g.add_edge(nid_LN,nid_V)
    g.add_edge(nid_Q,nid_QK); g.add_edge(nid_K,nid_QK)
    g.add_edge(nid_KR,nid_QK)
    g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV)
    g.add_edge(nid_V,nid_SV); g.add_edge(nid_VR,nid_SV)
    g.add_edge(nid_SV,nid_O)
    g.add_edge(nid_K,nid_KW); g.add_edge(nid_V,nid_VW)
    # MLP branch
    g.add_edge(nid_LN,nid_W1); g.add_edge(nid_W1,nid_G); g.add_edge(nid_G,nid_W2)
    # Merge both outputs plus residual X
    if x_in: g.add_edge(x_in,nid_Add2)
    g.add_edge(nid_O,nid_Add2); g.add_edge(nid_W2,nid_Add2)

class LLaMADef:
    name = "llama"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(shape.layer_num):
            add_llama_block(g, l, shape, dtype_bytes)
        return g

class MPTDef:
    name = "mpt"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(shape.layer_num):
            add_mpt_block(g, l, shape, dtype_bytes)
        return g

class PaLMDef:
    name = "palm"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(shape.layer_num):
            add_palm_block(g, l, shape, dtype_bytes)
        return g
    name = "palm"

class QwenDef:
    name = "qwen"
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(shape.layer_num):
            # 利用与LLaMA类似的层结构（RMSNorm -> Self-Attn -> MLP）
            add_llama_block(g, l, shape, dtype_bytes)
        return g

class MixtralDef:
    name = "mixtral"     
    def build(self, shape: ModelShape, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        E = getattr(shape, "experts_per_layer", 8)    # 每层专家数
        K = getattr(shape, "experts_top_k", 2)        # 每token选用的专家数
        dim = shape.hidden_dim
        q_dim = shape.q_head_num * shape.head_dim
        kv_dim = shape.kv_head_num * shape.head_dim
        o_in_dim = q_dim
        ffn = shape.intermediate_dim  # 用于 MoE 是所有专家之和

        for l in range(shape.layer_num):
            base_attr = {"layer": l, "q_heads": shape.q_head_num, "kv_heads": shape.kv_head_num,
                        "head_dim": shape.head_dim, "dim": dim}

            # LN1, Attention: Q, K, V, QK, Softmax, SV, O
            nid_LN1=f"L{l}_LN1"; g.add_node(TaskNode(nid_LN1,"LN",flops=0.0,attrs=dict(base_attr),allowed=get_op_allowed("LN")))
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
            # LN2 for FFN
            nid_LN2 = f"L{l}_LN2"
            g.add_node(TaskNode(nid_LN2, "LN", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("LN")))
            g.add_edge(f"L{l}_Add1", nid_LN2)

            # MoE FFN experts: FFN_W1/W3/W2
            expert_outputs = []
            for e in range(E):
                w1 = f"L{l}_FFN_W1_{e}"
                w3 = f"L{l}_FFN_W3_{e}"
                act = f"L{l}_Act_{e}"
                w2 = f"L{l}_FFN_W2_{e}"

                g.add_node(TaskNode(w1, "FFN_W1", flops=0.0,
                                    weight_id=f"{w1}_w", weight_size=dim * (ffn // 2) * dtype_bytes,
                                    attrs={**base_attr, "expert": e}, allowed=get_op_allowed("FFN_W1")))
                g.add_node(TaskNode(w3, "FFN_W3", flops=0.0,
                                    weight_id=f"{w3}_w", weight_size=dim * (ffn // 2) * dtype_bytes,
                                    attrs={**base_attr, "expert": e}, allowed=get_op_allowed("FFN_W3")))
                g.add_node(TaskNode(act, "SwiGLU", flops=0.0,
                                    attrs={**base_attr, "expert": e}, allowed=get_op_allowed("SwiGLU")))
                g.add_node(TaskNode(w2, "FFN_W2", flops=0.0,
                                    weight_id=f"{w2}_w", weight_size=(ffn // 2) * dim * dtype_bytes,
                                    attrs={**base_attr, "expert": e}, allowed=get_op_allowed("FFN_W2")))

                # connect FFN
                g.add_edge(nid_LN2, w1)
                g.add_edge(nid_LN2, w3)
                g.add_edge(w1, act)
                g.add_edge(w3, act)
                g.add_edge(act, w2)
                expert_outputs.append(w2)

            # Router
            nid_router = f"L{l}_Router"
            g.add_node(TaskNode(nid_router, "MoE_Router",
                                flops=0.0,
                                attrs={**base_attr, "experts": E, "top_k": K},
                                allowed=get_op_allowed("MoE_Router")))
            g.add_edge(nid_LN2, nid_router)
            for out in expert_outputs:
                g.add_edge(out, nid_router)

            # Residual Add2
            nid_add2 = f"L{l}_Add2"
            g.add_node(TaskNode(nid_add2, "Add", flops=0.0, attrs=dict(base_attr), allowed=get_op_allowed("Add")))
            g.add_edge(nid_router, nid_add2)
            g.add_edge(f"L{l}_Add1", nid_add2)

        return g
def make_model_def(family: str):
    f = family.lower()
    if f == "llama": return LLaMADef()
    if f == "mpt":   return MPTDef()
    if f == "palm":  return PaLMDef()
    if f == "mixtral": return MixtralDef()
    if f == "qwen": return QwenDef()
    raise ValueError(f"Unknown model family: {family}")
