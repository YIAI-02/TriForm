from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
from task_graph import TaskGraph, TaskNode
from model_definition import ModelShape, get_model_def
from config import ACTIVATION_BYTES, WEIGHT_BYTES
import json

@dataclass
class ParserConfig:
    batch: int = 1
    seq_len: int = 128

# FLOPs Estimation
def flops_linear(b,s,in_dim,out_dim): return 2.0*b*s*in_dim*out_dim
def flops_qk(b,s,dim,h): hd=dim//max(1,h); return 2.0*b*h*s*s*hd
def flops_sv(b,s,dim,h): hd=dim//max(1,h); return 2.0*b*h*s*s*hd
def flops_softmax(b,s,h): return 5.0*b*h*s*s
def flops_add(b,s,dim):   return 1.0*b*s*dim
def flops_ln(b,s,dim):    return 5.0*b*s*dim
def flops_gelu(b,s,dim):  return 6.0*b*s*dim
def flops_swiglu(b,s,dim):return 6.0*b*s*dim

def act_bytes(shape): 
    prod=1
    for x in shape: prod*=int(x)
    return prod*ACTIVATION_BYTES

def weight_bytes(shape):
    prod=1
    for x in shape: prod*=int(x)
    return prod*WEIGHT_BYTES

def parse_shape_json(path: str) -> Tuple[str, ModelShape]:
    with open(path,"r") as f: obj=json.load(f)
    model_type = obj.get("type","llama")
    shape = ModelShape(
        layer_num=int(obj["layer_num"]),
        dim=int(obj["hidden_dim"]),
        ffn_dim=int(obj["intermediate_dim"]),
        n_heads=int(obj["q_head_num"]),
        n_kv_heads=int(obj.get("kv_head_num", obj["q_head_num"]))
    )
    return model_type, shape

def build_model_graph(model_type: str, shape: ModelShape, cfg: ParserConfig) -> TaskGraph:
    b, s, dim, ffn, h = cfg.batch, cfg.seq_len, shape.dim, shape.ffn_dim, shape.n_heads
    ops, edges = get_model_def(model_type).layer_blueprint()
    g = TaskGraph()
    prev_out = None

    for l in range(shape.layer_num):
        nid = {}
        for op in ops:
            tid = f"L{l}:{op}"
            nid[op] = tid
            work=0.0; out_sz=0; wid=None; wsz=0

            if op == "X":
                out_sz = act_bytes((b,s,dim))
            elif op in ("LN1","LN2"):
                work = flops_ln(b,s,dim); out_sz = act_bytes((b,s,dim))
                wid = f"W:{tid}:LN"; wsz = weight_bytes((2,dim))
            elif op in ("Q","K","V","O"):
                work = flops_linear(b,s,dim,dim); out_sz = act_bytes((b,s,dim))
                wid = f"W:{tid}"; wsz = weight_bytes((dim,dim))
            elif op == "QK":
                work = flops_qk(b,s,dim,h); out_sz = act_bytes((b,h,s,s))
            elif op == "Softmax":
                work = flops_softmax(b,s,h); out_sz = act_bytes((b,h,s,s))
            elif op == "SV":
                work = flops_sv(b,s,dim,h); out_sz = act_bytes((b,s,dim))
            elif op in ("Add1","Add2"):
                work = flops_add(b,s,dim); out_sz = act_bytes((b,s,dim))
            elif op == "FFN_W1":
                work = flops_linear(b,s,dim,ffn); out_sz = act_bytes((b,s,ffn))
                wid = f"W:{tid}"; wsz = weight_bytes((dim,ffn))
            elif op == "FFN_W3":
                work = flops_linear(b,s,dim,ffn); out_sz = act_bytes((b,s,ffn))
                wid = f"W:{tid}"; wsz = weight_bytes((dim,ffn))
            elif op == "SwiGLU":
                work = flops_swiglu(b,s,ffn); out_sz = act_bytes((b,s,ffn))
            elif op == "GELU":
                work = flops_gelu(b,s,ffn); out_sz = act_bytes((b,s,ffn))
            elif op == "FFN_W2":
                work = flops_linear(b,s,ffn,dim); out_sz = act_bytes((b,s,dim))
                wid = f"W:{tid}"; wsz = weight_bytes((ffn,dim))
            else:
                out_sz = act_bytes((b,s,dim))

            g.add_node(TaskNode(
                id=tid, name=tid, work=work, out_size=out_sz,
                weight_id=wid, weight_size=wsz,
                allowed=(True,True),
                attrs={"op": op, "layer": l, "model": model_type}
            ))

        for u,v in edges: g.add_edge(nid[u], nid[v])
        if prev_out is not None: g.add_edge(prev_out, nid["X"])
        prev_out = nid["Add2"]

    return g
