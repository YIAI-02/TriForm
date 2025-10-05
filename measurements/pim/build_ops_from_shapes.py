#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List
from dialect import load_trace_dialect, TraceDialect, bytes_per_elem_from_pim
from dialect import ARG_DEFAULTS, CFR_EWMUL_BG
from weight_layout import plan_weight_layout_for_linear, emit_weight_write_trace
from mvm import emit_mvm, emit_activation_op

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser("Per-operator AiM trace emitter (one-layer example)")
    p.add_argument("--shapes", nargs="+", default=["/configs/llama_shape.json"], help="List of model shape JSON paths.")
    p.add_argument("--trace", default="submodules/aim_simulator/test/all_isr.trace", help="Reference trace path to learn allowed opcodes.")
    p.add_argument("--pim", default="/configs/pim.json", help="PIM config JSON path.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1)
    p.add_argument("--outdir", default="out_per_op")
    return p.parse_args()

# -----------------------
# Model layer (one-layer example)
# -----------------------
def num_elems(shape: List[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n

def head_dim(hidden_dim: int, q_heads: int) -> int:
    return hidden_dim // max(1, q_heads)

def make_linear(name, in_shape, out_dim, weight_shape, bpe, notes=""):
    return {
        "name": name, "category":"matmul", "op":"linear",
        "inputs":[{"tensor":"x","shape":in_shape}],
        "outputs":[{"tensor":"y","shape":in_shape[:-1]+[out_dim]}],
        "weights":[{"name":"W","shape":weight_shape,"size_bytes":num_elems(weight_shape)*bpe}],
        "notes": notes,
    }

def make_qkt(name, b,s,qh,kvh,hd):
    return {"name":name,"category":"matmul","op":"qk^T",
            "inputs":[{"tensor":"Q","shape":[b,s,qh,hd]},{"tensor":"K","shape":[b,s,kvh,hd]}],
            "outputs":[{"tensor":"scores","shape":[b,qh,s,s]}], "weights":[], "notes":""}

def make_av(name, b,s,qh,hd,kvh):
    return {"name":name,"category":"matmul","op":"A@V",
            "inputs":[{"tensor":"A","shape":[b,qh,s,s]},{"tensor":"V","shape":[b,s,kvh,hd]}],
            "outputs":[{"tensor":"ctx","shape":[b,s,qh,hd]}], "weights":[], "notes":""}

def make_vecmul(name, shapes, notes=""):
    return {"name":name,"category":"vecmul","op":"elemwise_mul",
            "inputs":[{"tensor":f"x{i}","shape":shp} for i,shp in enumerate(shapes)],
            "outputs":[{"tensor":"y","shape":shapes[0]}], "weights":[], "notes":notes}

def make_activation(name, act, shape):
    return {"name":name,"category":"activation","op":act.lower(),
            "inputs":[{"tensor":"x","shape":shape}],
            "outputs":[{"tensor":"y","shape":shape}], "weights":[], "notes":""}

def build_one_layer(model_type: str, shape: Dict, defaults: Dict, b: int, s: int, bpe: int) -> Dict:
    dim = int(shape["hidden_dim"]); inter = int(shape["intermediate_dim"])
    qh  = int(shape["q_head_num"]);  kvh  = int(shape["kv_head_num"])
    hd  = head_dim(dim, qh)

    L=0; ops=[]; x=[b,s,dim]
    # Norm1
    if defaults["norm"]=="rmsnorm":
        ops.append(make_vecmul(f"L{L}.RMSNorm1.scale",[x,[dim]],"RMSNorm gamma"))
    else:
        ops.append(make_vecmul(f"L{L}.LayerNorm1.scale",[x,[dim]],"LayerNorm gamma"))
    # Q/K/V
    ops.append(make_linear(f"L{L}.attn.q_proj",x,qh*hd,[dim,qh*hd],bpe,"no bias"))
    ops.append(make_linear(f"L{L}.attn.k_proj",x,kvh*hd,[dim,kvh*hd],bpe,"no bias (GQA)"))
    ops.append(make_linear(f"L{L}.attn.v_proj",x,kvh*hd,[dim,kvh*hd],bpe,"no bias (GQA)"))
    # RoPE
    ops.append(make_vecmul(f"L{L}.attn.rope",[[b,s,qh,hd]],"RoPE rotation"))
    # Attn core
    ops.append(make_qkt(f"L{L}.attn.qkT",b,s,qh,kvh,hd))
    ops.append(make_activation(f"L{L}.attn.softmax","softmax",[b,qh,s,s]))
    ops.append(make_av(f"L{L}.attn.attn_v",b,s,qh,hd,kvh))
    # out proj
    ops.append(make_linear(f"L{L}.attn.out_proj",[b,s,qh*hd],dim,[qh*hd,dim],bpe,"no bias"))
    # Norm2
    if defaults["norm"]=="rmsnorm":
        ops.append(make_vecmul(f"L{L}.RMSNorm2.scale",[x,[dim]],"RMSNorm gamma"))
    else:
        ops.append(make_vecmul(f"L{L}.LayerNorm2.scale",[x,[dim]],"LayerNorm gamma"))
    # MLP
    if defaults["mlp"]=="swiglu":
        ops.append(make_linear(f"L{L}.mlp.gate_proj",x,inter,[dim,inter],bpe,"no bias"))
        ops.append(make_linear(f"L{L}.mlp.up_proj",  x,inter,[dim,inter],bpe,"no bias"))
        ops.append(make_activation(f"L{L}.mlp.act","silu",[b,s,inter]))
        ops.append(make_vecmul(f"L{L}.mlp.gated_mul",[[b,s,inter],[b,s,inter]],"SwiGLU gate"))
        ops.append(make_linear(f"L{L}.mlp.down_proj",[b,s,inter],dim,[inter,dim],bpe,"no bias"))
    else:
        ops.append(make_linear(f"L{L}.mlp.up_proj",x,inter,[dim,inter],bpe,"bias may be present"))
        ops.append(make_activation(f"L{L}.mlp.act","gelu",[b,s,inter]))
        ops.append(make_linear(f"L{L}.mlp.down_proj",[b,s,inter],dim,[inter,dim],bpe,"bias may be present"))
    return {
        "model_type": model_type,
        "shape":{"hidden_dim":dim,"intermediate_dim":inter,"q_head_num":qh,"kv_head_num":kvh,"head_dim":hd},
        "assumptions":{"norm":defaults["norm"],"mlp":defaults["mlp"],"bytes_per_elem":bpe,
                       "notes":"one-layer example; only matmul/vecmul/activation"},
        "layers":[{"layer_index":0,"ops":ops}]
    }

# -----------------------
# Defaults per model
# -----------------------
MODEL_DEFAULTS = {
    "llama":  {"norm":"rmsnorm","mlp":"swiglu"},
    "mistral":{"norm":"rmsnorm","mlp":"swiglu"},
    "qwen2":  {"norm":"rmsnorm","mlp":"swiglu"},
    "mpt":    {"norm":"layernorm","mlp":"gelu"},
}

# -----------------------
# Utils
# -----------------------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._\-]+', '_', name)

def next_op_is_activation(layer_ops: list[dict], idx: int) -> bool:
    if idx + 1 < len(layer_ops):
        return layer_ops[idx+1]["category"] == "activation"
    return False

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    dialect = load_trace_dialect(args.trace)
    pim     = json.loads(Path(args.pim).read_text(encoding="utf-8"))
    bpe     = bytes_per_elem_from_pim(pim, fallback=2)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for sp in args.shapes:
        p = Path(sp)
        if not p.exists():
            print(f"[WARN] shape file not found: {p}")
            continue
        shape = json.loads(p.read_text(encoding="utf-8"))
        mtype = shape["type"].lower()
        defaults = MODEL_DEFAULTS.get(mtype, {"norm":"rmsnorm","mlp":"swiglu"})
        model_ops = build_one_layer(mtype, shape, defaults, args.batch, args.seq_len, bpe)

        model_dir = outdir / mtype
        ops_dir   = model_dir / "ops"
        model_dir.mkdir(parents=True, exist_ok=True)
        ops_dir.mkdir(parents=True, exist_ok=True)

        # 保存 ops JSON
        (model_dir / f"{mtype}_ops.json").write_text(json.dumps(model_ops, indent=2), encoding="utf-8")

        # 逐ops 生成 trace
        index = []
        ops = model_ops["layers"][0]["ops"]
        for i, op in enumerate(ops):
            name = op["name"]
            cat  = op["category"]
            fname = f"{i:03d}_" + sanitize_filename(name) + ".aim.trace"
            lines: list[str] = [f"# --- op trace ---", f"# {name} ({cat}/{op['op']})"]

            if cat == "matmul":
                # 判断mvm的下一个op是不是act，如果是，直接从mvm把中间结果给act
                defer_final = next_op_is_activation(ops, i)

                if op["op"] == "linear" and op.get("weights"):
                    # 分开生成两个trace文件：权重写入和计算
                    
                    # 1) 权重写入
                    weight_fname = f"{i:03d}_" + sanitize_filename(name) + "_weights.trace"
                    weight_lines = [f"# --- weight initialization trace ---", 
                                    f"# {name} ({cat}/{op['op']}) - Weights"]
                    wshape = op["weights"][0]["shape"]
                    layout = plan_weight_layout_for_linear(wshape, pim, bpe)
                    weight_lines += emit_weight_write_trace(layout, dialect)
                    (ops_dir / weight_fname).write_text("\n".join(weight_lines) + "\n", encoding="utf-8")
                    
                    # 添加权重文件到index
                    index.append({
                        "op_index": i,
                        "name": f"{name} (weights)",
                        "category": "weight_init",
                        "op": "weight_write",
                        "trace_file": weight_fname,
                        "inputs": [],
                        "outputs": op.get("weights", []),
                        "notes": f"Weight initialization for {name}"
                    })
                    
                    # 2) 计算
                    compute_fname = f"{i:03d}_" + sanitize_filename(name) + "_compute.aim.trace"
                    compute_lines = [f"# --- computation trace ---", 
                                     f"# {name} ({cat}/{op['op']}) - Computation"]
                    compute_lines += emit_mvm(op, pim, dialect, bpe, weight_layout=layout,
                                      defer_final_rdmac_to_activation=defer_final)
                    (ops_dir / compute_fname).write_text("\n".join(compute_lines) + "\n", encoding="utf-8")
                    
                    # 更新文件名以添加到index
                    fname = compute_fname
                    lines = compute_lines
                else:
                    # 非 linear（qk^T / A@V 等）：不写权重，仅生成计算trace
                    compute_fname = f"{i:03d}_" + sanitize_filename(name) + "_compute.aim.trace"
                    lines = [f"# --- computation trace ---", f"# {name} ({cat}/{op['op']})"]
                    lines += emit_mvm(op, pim, dialect, bpe, weight_layout=None,
                                  defer_final_rdmac_to_activation=defer_final)
                    fname = compute_fname

            elif cat == "activation":
                # RD_MAC -> W CFR 2 1 -> AF -> RD_AF
                lines += emit_activation_op(op, dialect)
                
            # 写计算文件（对于activation类型直接写入）
            (ops_dir / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")
            index.append({
                "op_index": i,
                "name": name,
                "category": cat,
                "op": op["op"],
                "trace_file": fname,
                "inputs": op["inputs"],
                "outputs": op["outputs"],
                "weights": op.get("weights", []) if cat != "weight_init" else [],
                "notes": op.get("notes", "")
            })

        # index.json
        (ops_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
