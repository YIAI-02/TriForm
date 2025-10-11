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
    """根据算子名称获取设备约束"""
    return OPERATOR_DEVICE_ALLOWED.get(op_name, DEFAULT_OPERATOR_ALLOWED).copy()

def linear_flops(inp, out):
    return 2.0 * inp * out

def add_llama_block(g: TaskGraph, l: int, shape: ModelShape, phase: str, dtype_bytes: int):
    b, s = shape.batch, shape.seq_len
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim
    q_dim, kv_dim, o_in_dim = qh*hd, kvh*hd, qh*hd
    attr = {"layer": l, "q_heads": qh, "kv_heads": kvh, "head_dim": hd}

    # LN1
    nid_LN1 = f"L{l}_LN1"
    g.add_node(TaskNode(nid_LN1, "LN", flops=b*s*dim, attrs=attr, allowed=get_op_allowed("LN")))
    # Q/K/V
    nid_Q = f"L{l}_Q"
    nid_K = f"L{l}_K"
    nid_V = f"L{l}_V"
    g.add_node(TaskNode(nid_Q, "Q", flops=linear_flops(dim, q_dim)*b*(s if phase=='prefill' else 1),
                        weight_id=f"L{l}_WQ", weight_size=dim*q_dim*dtype_bytes, attrs=attr, allowed=get_op_allowed("Q")))
    g.add_node(TaskNode(nid_K, "K", flops=linear_flops(dim, kv_dim)*b*(s if phase=='prefill' else 1),
                        weight_id=f"L{l}_WK", weight_size=dim*kv_dim*dtype_bytes, attrs=attr, allowed=get_op_allowed("K")))
    g.add_node(TaskNode(nid_V, "V", flops=linear_flops(dim, kv_dim)*b*(s if phase=='prefill' else 1),
                        weight_id=f"L{l}_WV", weight_size=dim*kv_dim*dtype_bytes, attrs=attr, allowed=get_op_allowed("V")))

    # Attention core
    nid_QK = f"L{l}_QK"; nid_SO = f"L{l}_Softmax"; nid_SV = f"L{l}_SV"
    if phase == "prefill":
        qk_flops = 2.0 * b * qh * (s*s) * hd
        sv_flops = 2.0 * b * qh * (s*s) * hd
    else:
        qk_flops = 2.0 * b * qh * (s*hd)
        sv_flops = 2.0 * b * qh * (s*hd)
    g.add_node(TaskNode(nid_QK, "QK", flops=qk_flops, attrs=attr, allowed=get_op_allowed("QK")))
    g.add_node(TaskNode(nid_SO, "Softmax", flops=b*qh*(s if phase=='decode' else s*s), attrs=attr, allowed=get_op_allowed("Softmax")))
    g.add_node(TaskNode(nid_SV, "SV", flops=sv_flops, attrs=attr, allowed=get_op_allowed("SV")))

    # O
    nid_O = f"L{l}_O"
    g.add_node(TaskNode(nid_O, "O", flops=linear_flops(o_in_dim, dim)*b*(s if phase=='prefill' else 1),
                        weight_id=f"L{l}_WO", weight_size=o_in_dim*dim*dtype_bytes, attrs=attr, allowed=get_op_allowed("O")))

    nid_Add1 = f"L{l}_Add1"
    g.add_node(TaskNode(nid_Add1, "Add", flops=b*dim*(s if phase=='prefill' else 1), allowed=get_op_allowed("Add")))

    # MLP: SwiGLU (W1,W3)->Act->W2
    nid_LN2 = f"L{l}_LN2"; nid_W1=f"L{l}_FFN_W1"; nid_W3=f"L{l}_FFN_W3"; nid_ACT=f"L{l}_Act"; nid_W2=f"L{l}_FFN_W2"; nid_Add2=f"L{l}_Add2"
    g.add_node(TaskNode(nid_LN2, "LN", flops=b*s*dim, attrs=attr, allowed=get_op_allowed("LN")))
    g.add_node(TaskNode(nid_W1, "FFN_W1", flops=linear_flops(dim, ffn)*b*(s if phase=='prefill' else 1), weight_id=f"L{l}_W1", weight_size=dim*ffn*dtype_bytes, attrs=attr, allowed=get_op_allowed("FFN_W1")))
    g.add_node(TaskNode(nid_W3, "FFN_W3", flops=linear_flops(dim, ffn)*b*(s if phase=='prefill' else 1), weight_id=f"L{l}_W3", weight_size=dim*ffn*dtype_bytes, attrs=attr, allowed=get_op_allowed("FFN_W3")))
    g.add_node(TaskNode(nid_ACT, "SwiGLU", flops=b*ffn*(s if phase=='prefill' else 1), attrs=attr, allowed=get_op_allowed("SwiGLU")))
    g.add_node(TaskNode(nid_W2, "FFN_W2", flops=linear_flops(ffn, dim)*b*(s if phase=='prefill' else 1), weight_id=f"L{l}_W2", weight_size=ffn*dim*dtype_bytes, attrs=attr, allowed=get_op_allowed("FFN_W2")))
    g.add_node(TaskNode(nid_Add2, "Add", flops=b*dim*(s if phase=='prefill' else 1), allowed=get_op_allowed("Add")))

    # KV explicit ops in decode
    if phase == "decode":
        nid_KVr=f"L{l}_KV_read"; nid_KVw=f"L{l}_KV_write"
        g.add_node(TaskNode(nid_KVr, "KV_read", attrs=attr, allowed=get_op_allowed("KV_read")))
        g.add_node(TaskNode(nid_KVw, "KV_write", attrs=attr, allowed=get_op_allowed("KV_write")))

    # Wire connections (pre-norm, sequential)
    x_in = f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X = f"L{l}_X"; g.add_node(TaskNode(nid_X, "Identity", allowed=get_op_allowed("Identity")))
        g.add_edge(x_in, nid_X); g.add_edge(nid_X, nid_LN1)

    g.add_edge(nid_LN1, nid_Q); g.add_edge(nid_LN1, nid_K); g.add_edge(nid_LN1, nid_V)
    if phase == "decode":
        g.add_edge(nid_Q, nid_QK); g.add_edge(nid_QK, nid_SO); g.add_edge(nid_SO, nid_SV); g.add_edge(nid_SV, nid_O)
        g.add_edge(nid_KVr, nid_QK); g.add_edge(nid_KVr, nid_SV)
        g.add_edge(nid_K, nid_KVw); g.add_edge(nid_V, nid_KVw)
    else:
        g.add_edge(nid_Q, nid_QK); g.add_edge(nid_K, nid_QK); g.add_edge(nid_QK, nid_SO); g.add_edge(nid_SO, nid_SV); g.add_edge(nid_V, nid_SV); g.add_edge(nid_SV, nid_O)
    g.add_edge(nid_O, nid_Add1); 
    if x_in: g.add_edge(x_in, nid_Add1)

    g.add_edge(nid_Add1, nid_LN2); g.add_edge(nid_LN2, nid_W1); g.add_edge(nid_LN2, nid_W3)
    g.add_edge(nid_W1, nid_ACT); g.add_edge(nid_W3, nid_ACT)
    g.add_edge(nid_ACT, nid_W2); g.add_edge(nid_W2, nid_Add2); g.add_edge(nid_Add1, nid_Add2)

def add_mpt_block(g: TaskGraph, l: int, shape: ModelShape, phase: str, dtype_bytes: int):
    # Similar to LLaMA but MLP uses GELU (no W3 gate by default)
    b, s = shape.batch, shape.seq_len
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim
    q_dim, kv_dim, o_in_dim = qh*hd, kvh*hd, qh*hd
    attr = {"layer": l, "q_heads": qh, "kv_heads": kvh, "head_dim": hd}

    nid_LN1=f"L{l}_LN1"; g.add_node(TaskNode(nid_LN1,"LN",flops=b*s*dim,attrs=attr,allowed=get_op_allowed("LN")))
    nid_Q=f"L{l}_Q"; nid_K=f"L{l}_K"; nid_V=f"L{l}_V"
    g.add_node(TaskNode(nid_Q,"Q",flops=linear_flops(dim,q_dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WQ",weight_size=dim*q_dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("Q")))
    g.add_node(TaskNode(nid_K,"K",flops=linear_flops(dim,kv_dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WK",weight_size=dim*kv_dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("K")))
    g.add_node(TaskNode(nid_V,"V",flops=linear_flops(dim,kv_dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WV",weight_size=dim*kv_dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("V")))

    nid_QK=f"L{l}_QK"; nid_SO=f"L{l}_Softmax"; nid_SV=f"L{l}_SV"
    if phase=='prefill':
        qk=2.0*b*qh*(s*s)*hd; sv=2.0*b*qh*(s*s)*hd
    else:
        qk=2.0*b*qh*(s*hd);   sv=2.0*b*qh*(s*hd)
    g.add_node(TaskNode(nid_QK,"QK",flops=qk,attrs=attr,allowed=get_op_allowed("QK")))
    g.add_node(TaskNode(nid_SO,"Softmax",flops=b*qh*(s if phase=='decode' else s*s),attrs=attr,allowed=get_op_allowed("Softmax")))
    g.add_node(TaskNode(nid_SV,"SV",flops=sv,attrs=attr,allowed=get_op_allowed("SV")))

    nid_O=f"L{l}_O"; g.add_node(TaskNode(nid_O,"O",flops=linear_flops(o_in_dim,dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WO",weight_size=o_in_dim*dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("O")))
    nid_Add1=f"L{l}_Add1"; g.add_node(TaskNode(nid_Add1,"Add",flops=b*dim*(s if phase=='prefill' else 1),allowed=get_op_allowed("Add")))

    nid_LN2=f"L{l}_LN2"; nid_W1=f"L{l}_FFN_W1"; nid_G=f"L{l}_GELU"; nid_W2=f"L{l}_FFN_W2"; nid_Add2=f"L{l}_Add2"
    g.add_node(TaskNode(nid_LN2,"LN",flops=b*s*dim,attrs=attr,allowed=get_op_allowed("LN")))
    g.add_node(TaskNode(nid_W1,"FFN_W1",flops=linear_flops(dim,ffn)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_W1",weight_size=dim*ffn*dtype_bytes,attrs=attr,allowed=get_op_allowed("FFN_W1")))
    g.add_node(TaskNode(nid_G,"GELU",flops=b*ffn*(s if phase=='prefill' else 1),attrs=attr,allowed=get_op_allowed("GELU")))
    g.add_node(TaskNode(nid_W2,"FFN_W2",flops=linear_flops(ffn,dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_W2",weight_size=ffn*dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("FFN_W2")))
    g.add_node(TaskNode(nid_Add2,"Add",flops=b*dim*(s if phase=='prefill' else 1),allowed=get_op_allowed("Add")))

    if phase=='decode':
        nid_KVr=f"L{l}_KV_read"; nid_KVw=f"L{l}_KV_write"
        g.add_node(TaskNode(nid_KVr,"KV_read",attrs=attr,allowed=get_op_allowed("KV_read")))
        g.add_node(TaskNode(nid_KVw,"KV_write",attrs=attr,allowed=get_op_allowed("KV_write")))

    x_in=f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"; g.add_node(TaskNode(nid_X,"Identity",allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN1)

    g.add_edge(nid_LN1,nid_Q); g.add_edge(nid_LN1,nid_K); g.add_edge(nid_LN1,nid_V)
    if phase=='decode':
        g.add_edge(nid_Q,nid_QK); g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV); g.add_edge(nid_SV,nid_O)
        g.add_edge(nid_KVr,nid_QK); g.add_edge(nid_KVr,nid_SV)
        g.add_edge(nid_K,nid_KVw); g.add_edge(nid_V,nid_KVw)
    else:
        g.add_edge(nid_Q,nid_QK); g.add_edge(nid_K,nid_QK); g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV); g.add_edge(nid_V,nid_SV); g.add_edge(nid_SV,nid_O)
    g.add_edge(nid_O,nid_Add1); 
    if x_in: g.add_edge(x_in,nid_Add1)

    g.add_edge(nid_Add1,nid_LN2); g.add_edge(nid_LN2,nid_W1); g.add_edge(nid_W1,nid_G); g.add_edge(nid_G,nid_W2); g.add_edge(nid_W2,nid_Add2); g.add_edge(nid_Add1,nid_Add2)

def add_palm_block(g: TaskGraph, l: int, shape: ModelShape, phase: str, dtype_bytes: int):
    # PaLM uses pre-LN and PARALLEL residual: x + Attn(LN(x)) + MLP(LN(x))
    b, s = shape.batch, shape.seq_len
    dim, ffn = shape.dim, shape.ffn_dim
    qh, kvh, hd = shape.n_heads, shape.n_kv_heads, shape.head_dim
    # Many PaLM variants use MQA (kv_heads=1). head_dim uses dim//q_heads by default.
    q_dim, kv_dim, o_in_dim = qh*hd, kvh*hd, qh*hd
    attr = {"layer": l, "q_heads": qh, "kv_heads": kvh, "head_dim": hd}

    nid_LN = f"L{l}_LN"  # one LN feeding both branches
    g.add_node(TaskNode(nid_LN,"LN",flops=b*s*dim,attrs=attr,allowed=get_op_allowed("LN")))

    # Attn branch
    nid_Q=f"L{l}_Q"; nid_K=f"L{l}_K"; nid_V=f"L{l}_V"
    g.add_node(TaskNode(nid_Q,"Q",flops=linear_flops(dim,q_dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WQ",weight_size=dim*q_dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("Q")))
    g.add_node(TaskNode(nid_K,"K",flops=linear_flops(dim,kv_dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WK",weight_size=dim*kv_dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("K")))
    g.add_node(TaskNode(nid_V,"V",flops=linear_flops(dim,kv_dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WV",weight_size=dim*kv_dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("V")))

    nid_QK=f"L{l}_QK"; nid_SO=f"L{l}_Softmax"; nid_SV=f"L{l}_SV"
    if phase=='prefill': qk=2.0*b*qh*(s*s)*hd; sv=2.0*b*qh*(s*s)*hd
    else:               qk=2.0*b*qh*(s*hd);   sv=2.0*b*qh*(s*hd)
    g.add_node(TaskNode(nid_QK,"QK",flops=qk,attrs=attr,allowed=get_op_allowed("QK")))
    g.add_node(TaskNode(nid_SO,"Softmax",flops=b*qh*(s if phase=='decode' else s*s),attrs=attr,allowed=get_op_allowed("Softmax")))
    g.add_node(TaskNode(nid_SV,"SV",flops=sv,attrs=attr,allowed=get_op_allowed("SV")))

    nid_O=f"L{l}_O"
    g.add_node(TaskNode(nid_O,"O",flops=linear_flops(o_in_dim,dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_WO",weight_size=o_in_dim*dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("O")))

    # MLP branch (GELU)
    nid_W1=f"L{l}_FFN_W1"; nid_G=f"L{l}_GELU"; nid_W2=f"L{l}_FFN_W2"
    g.add_node(TaskNode(nid_W1,"FFN_W1",flops=linear_flops(dim,ffn)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_W1",weight_size=dim*ffn*dtype_bytes,attrs=attr,allowed=get_op_allowed("FFN_W1")))
    g.add_node(TaskNode(nid_G,"GELU",flops=b*ffn*(s if phase=='prefill' else 1),attrs=attr,allowed=get_op_allowed("GELU")))
    g.add_node(TaskNode(nid_W2,"FFN_W2",flops=linear_flops(ffn,dim)*b*(s if phase=='prefill' else 1),weight_id=f"L{l}_W2",weight_size=ffn*dim*dtype_bytes,attrs=attr,allowed=get_op_allowed("FFN_W2")))

    # Merge: X + Attn + MLP
    nid_Add2=f"L{l}_Add2"; g.add_node(TaskNode(nid_Add2,"Add",flops=b*dim*(s if phase=='prefill' else 1),allowed=get_op_allowed("Add")))

    # KV ops on decode
    if phase=='decode':
        nid_KVr=f"L{l}_KV_read"; nid_KVw=f"L{l}_KV_write"
        g.add_node(TaskNode(nid_KVr,"KV_read",attrs=attr,allowed=get_op_allowed("KV_read")))
        g.add_node(TaskNode(nid_KVw,"KV_write",attrs=attr,allowed=get_op_allowed("KV_write")))

    # Wire
    x_in=f"L{l-1}_Add2" if l>0 else None
    if x_in:
        nid_X=f"L{l}_X"; g.add_node(TaskNode(nid_X,"Identity",allowed=get_op_allowed("Identity")))
        g.add_edge(x_in,nid_X); g.add_edge(nid_X,nid_LN)

    # Both branches from same LN
    g.add_edge(nid_LN,nid_Q); g.add_edge(nid_LN,nid_K); g.add_edge(nid_LN,nid_V)
    if phase=='decode':
        g.add_edge(nid_Q,nid_QK); g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV); g.add_edge(nid_SV,nid_O)
        g.add_edge(nid_KVr,nid_QK); g.add_edge(nid_KVr,nid_SV)
        g.add_edge(nid_K,nid_KVw); g.add_edge(nid_V,nid_KVw)
    else:
        g.add_edge(nid_Q,nid_QK); g.add_edge(nid_K,nid_QK); g.add_edge(nid_QK,nid_SO); g.add_edge(nid_SO,nid_SV); g.add_edge(nid_V,nid_SV); g.add_edge(nid_SV,nid_O)
    # MLP branch
    g.add_edge(nid_LN,nid_W1); g.add_edge(nid_W1,nid_G); g.add_edge(nid_G,nid_W2)
    # Merge both outputs plus residual X
    if x_in: g.add_edge(x_in,nid_Add2)
    g.add_edge(nid_O,nid_Add2); g.add_edge(nid_W2,nid_Add2)

class LLaMADef:
    name = "llama"
    def build(self, shape: ModelShape, phase: str, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(shape.layer_num):
            add_llama_block(g, l, shape, phase, dtype_bytes)
        return g

class MPTDef:
    name = "mpt"
    def build(self, shape: ModelShape, phase: str, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(shape.layer_num):
            add_mpt_block(g, l, shape, phase, dtype_bytes)
        return g

class PaLMDef:
    name = "palm"
    def build(self, shape: ModelShape, phase: str, dtype_bytes: int) -> TaskGraph:
        g = TaskGraph()
        for l in range(shape.layer_num):
            add_palm_block(g, l, shape, phase, dtype_bytes)
        return g

def make_model_def(family: str):
    f = family.lower()
    if f == "llama": return LLaMADef()
    if f == "mpt":   return MPTDef()
    if f == "palm":  return PaLMDef()
    raise ValueError(f"Unknown model family: {family}")
