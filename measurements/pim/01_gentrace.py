#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Optional, List

# 共享工具
from aim_shared import (
    ensure_cent_on_path, load_pim_config, make_tb_args_from_pim, make_dic_model,
    emit_single_op_trace, parse_int_list, load_model_shape
)

# 引入 CENT 的模块（确保搜索路径）
ensure_cent_on_path()
from TransformerBlock import TransformerBlock  # type: ignore

def main():
    ap = argparse.ArgumentParser(description="Simplified single-op trace generator (AiM).")
    ap.add_argument("--pim-config", type=Path, required=True, help="JSON with PIM-related params")
    ap.add_argument("--ops", type=str, required=True, help="Comma-separated: score,output,weight")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seqlens", type=str, default=None, help="for score/output")
    ap.add_argument("--vector-dims", type=str, default=None, help="for weight")
    ap.add_argument("--matrix-cols", type=str, default=None, help="for weight")
    ap.add_argument("--with-af", action="store_true", help="Append AF+RD_AF after weight GEMV")
    ap.add_argument("--dim", type=int, default=None, help="若未显式给出，优先从 --model-shape 中读取")
    ap.add_argument("--n-heads", type=int, default=None, help="若未显式给出，优先从 --model-shape 中读取")
    ap.add_argument("--n-kv-heads", type=int, default=None, help="若未显式给出，尝试从 --model-shape 读取，否则等于 n-heads")
    ap.add_argument("--model-shape", type=Path, default=None, help="读取 ../configs/*_shape.json，提取 dim/n_heads/n_kv_heads/seq_length")
    args = ap.parse_args()

    shape = None
    if args.model_shape:
        shape = load_model_shape(args.model_shape)

    # 解析/覆盖 模型形状
    dim = args.dim if args.dim is not None else (shape["dim"] if shape else 256)
    n_heads = args.n_heads if args.n_heads is not None else (shape["n_heads"] if shape else 8)
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else (shape["n_kv_heads"] if shape else n_heads)

    pim_cfg = load_pim_config(args.pim_config)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    seqlens = parse_int_list(args.seqlens)
    v_dims = parse_int_list(args.vector_dims)
    m_cols = parse_int_list(args.matrix_cols)

    # 默认 seqlen：若 shape 指明 seq_length，则默认取[16..seq_length]的一个对数间隔采样；否则用一组常用点
    if any(o in ("score","output") for o in ops) and not seqlens:
        if shape and shape.get("seq_length"):
            Lmax = int(shape["seq_length"])
            cand = [16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096]
            seqlens = [x for x in cand if x <= Lmax] or [min(512, Lmax)]
        else:
            seqlens = [64, 128, 256, 512, 1024, 2048]
    if ("weight" in ops) and (not v_dims or not m_cols):
        v_dims = [dim]
        m_cols = [dim, 4*dim]

    # 构造模型字典（用于 TransformerBlock）
    dic_model = make_dic_model(dim, n_heads, n_kv_heads, seqlens[0] if seqlens else 16)
    row_index_matrix = 0

    for op in ops:
        if op in ("score","output"):
            for L in seqlens:
                trace_path = out_dir / f"{op}_seq{L}_dim{dim}_h{n_heads}.aim"
                tb_args = make_tb_args_from_pim(pim_cfg, str(trace_path))
                tb_args.seqlen = L
                block = TransformerBlock(dic_model, tb_args)
                emit_single_op_trace(block, op, row_index_matrix, seqlen=L)
                block.finish()
                (trace_path.with_suffix(".json")).write_text(
                    json.dumps({"op":op,"seqlen":L, "dim":dim,"n_heads":n_heads,"n_kv_heads":n_kv_heads, **pim_cfg}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                print(f"[ok] wrote {trace_path}")
        elif op == "weight":
            for V in v_dims:
                for N in m_cols:
                    suffix = "_with_af" if args.with_af else ""
                    trace_path = out_dir / f"weight_vec{V}_col{N}_dim{dim}_h{n_heads}{suffix}.aim"
                    tb_args = make_tb_args_from_pim(pim_cfg, str(trace_path))
                    block = TransformerBlock(dic_model, tb_args)
                    emit_single_op_trace(block, "weight", row_index_matrix, vector_dim=V, matrix_col=N, with_af=args.with_af)
                    block.finish()
                    (trace_path.with_suffix(".json")).write_text(
                        json.dumps({"op":"weight","vector_dim":V,"matrix_col":N,"with_af":bool(args.with_af),
                                    "dim":dim,"n_heads":n_heads,"n_kv_heads":n_kv_heads, **pim_cfg}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                    print(f"[ok] wrote {trace_path}")
        else:
            raise ValueError(f"Unsupported op: {op}")

if __name__ == "__main__":
    main()
