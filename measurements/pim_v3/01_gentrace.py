#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ------------------ Bootstrap: locate CENT/cent_simulation ------------------
_HERE = Path(__file__).resolve()
PROJECT_ROOT = None
for p in [_HERE.parent] + list(_HERE.parents):
    cand = p / "submodules" / "CENT" / "cent_simulation"
    if cand.exists():
        PROJECT_ROOT = p
        CENT_SIM_DIR = cand
        break
if PROJECT_ROOT is None:
    raise RuntimeError("Cannot find 'submodules/CENT/cent_simulation' above {}".format(_HERE))

if str(CENT_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(CENT_SIM_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import TransformerBlock and torch
from TransformerBlock import TransformerBlock  # type: ignore
import torch  # type: ignore

#区分mac时间属于哪个阶段
TIMING = {
    "score": "breakdown_sa_score",
    "output": "breakdown_sa_output",
    "weight": "breakdown_sa_weight",
}

def load_pim_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    std = {}
    alias = {
        "DRAM_column": ["dram_column", "DRAMCol", "dramCol", "dram_col"],
        "DRAM_row": ["dram_row", "DRAMRow", "dramRow", "dram_row"],
        "burst_length": ["burst", "burstLength", "BL"],
        "num_banks": ["banks", "numBanks"],
        "num_channels": ["channels", "numChannels"],
        "threads": ["thread", "nthreads"],
        "reuse_size": ["reuse", "reuseSize"],
        "channels_per_block": ["channelsPerBlock", "cpb"],
        "max_seq_len": ["maxSeqLen", "max_seq_length"],
    }
    for k in list(cfg.keys()):
        v = cfg[k]
        for stdk, alist in alias.items():
            if k == stdk or k in alist:
                std[stdk] = v
                break
        else:
            std[k] = v
    
    std.setdefault("DRAM_column", 256)
    std.setdefault("DRAM_row", 64)
    std.setdefault("burst_length", 16)
    std.setdefault("num_banks", 8)
    std.setdefault("num_channels", 4)
    std.setdefault("threads", 1)
    std.setdefault("reuse_size", 32)
    std.setdefault("channels_per_block", None)  # None -> 用 num_channels
    std.setdefault("max_seq_len", 4096)
    return std

def make_tb_args_from_pim(cfg: Dict[str, Any], trace_file: str) -> Any:
    from types import SimpleNamespace
    return SimpleNamespace(
        DRAM_column=cfg["DRAM_column"],
        DRAM_row=cfg["DRAM_row"],
        burst_length=cfg["burst_length"],
        num_banks=cfg["num_banks"],
        num_channels=cfg["num_channels"],
        threads=cfg["threads"],
        reuse_size=cfg["reuse_size"],
        channels_per_block=cfg["channels_per_block"],
        max_seq_len=cfg["max_seq_len"],
        # tracing toggles
        only_trace=True,
        op_trace=False,
        trace_file=trace_file,
        # TB-specific
        pim_compute=True, model="llama_like", embedding="rope",
        seqlen=16,
        model_parallel=False, FC_devices=1,
        pipeline_parallel=False, inter_device_attention=False, only_FC=False,
        trace_prepare=False, trace_norm=False, trace_fc_kqvo=False, trace_attention=False,
        trace_softmax=False, trace_fc_ffn=False, trace_activation=False,
        GEMV="reuse-GB",
    )

def make_dic_model(dim: int, n_heads: int, n_kv_heads: Optional[int], seqlen: int) -> Dict[str, Any]:
    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = dim // n_heads
    assert head_dim > 0, "dim must be divisible by n_heads"
    import torch
    return {
        "TP_param": torch.tensor(1),
        "dim": torch.tensor(dim),
        "n_heads": torch.tensor(n_heads),
        "n_kv_heads": torch.tensor(n_kv_heads),
        "x": torch.randn(1, seqlen, dim),
        "SANorm": torch.ones(dim),
        "FFNNorm": torch.ones(dim),
        "start_pos": 0,
        "freqs_cis": torch.zeros(seqlen, head_dim, dtype=torch.complex64),
        "sa": torch.zeros(1, n_heads, seqlen, seqlen),
        "h": torch.zeros(1, n_heads, seqlen, head_dim),
        "out": torch.zeros(1, seqlen, dim),
        "wq": torch.randn(dim, dim),
        "wk": torch.randn(dim, dim),
        "wv": torch.randn(dim, dim),
        "wo": torch.randn(dim, dim),
        "w1": torch.randn(4*dim, dim),
        "w2": torch.randn(dim, 4*dim),
        "w3": torch.randn(4*dim, dim),
        "xq": torch.randn(1, n_heads, 1, head_dim),
        "xk": torch.randn(1, n_kv_heads, 1, head_dim),
        "xv": torch.randn(1, n_kv_heads, 1, head_dim),
        "cache_k": torch.randn(1, seqlen, n_kv_heads, head_dim),
        "cache_v": torch.randn(1, seqlen, n_kv_heads, head_dim),
        "scores": torch.randn(1, n_heads, 1, seqlen),
        "output": torch.zeros(1, seqlen, dim),
        "ffn": torch.zeros(1, seqlen, dim),
    }

def call_score_or_output(block: TransformerBlock, op: str, row_index_matrix: int, seqlen: int):
    timing = TIMING[op]
    if op == "score":
        block.Vector_Matrix_Mul_score_pim_only_trace(row_index_matrix, seqlen, timing)
    else:
        block.Vector_Matrix_Mul_output_pim_only_trace(row_index_matrix, seqlen, timing)

def call_weight(block: TransformerBlock, row_index_matrix: int, vector_dim: int, matrix_col: int,
                with_af: bool):
    timing = TIMING["weight"]
    channel_lst = [i for i in range(block.num_channels)]   # 足够触发 hex_channel_mask
    total_banks = block.FC_total_banks

    block.Vector_Matrix_Mul_weight_pim_only_trace(
        channel_lst, row_index_matrix, vector_dim, matrix_col, total_banks, timing
    )
    if with_af:
        block.AF_only_trace(channel_lst)
        block.RD_AF_only_trace(channel_lst)

def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s: return None
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(description="Simplified single-op trace generator (AiM).")
    ap.add_argument("--pim-config", type=Path, required=True, help="JSON with PIM-related params")
    ap.add_argument("--ops", type=str, required=True, help="Comma-separated: score,output,weight")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seqlens", type=str, default=None, help="for score/output")
    ap.add_argument("--vector-dims", type=str, default=None, help="for weight")
    ap.add_argument("--matrix-cols", type=str, default=None, help="for weight")
    ap.add_argument("--with-af", action="store_true", help="Append AF+RD_AF after weight GEMV")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-kv-heads", type=int, default=None)
    args = ap.parse_args()

    pim_cfg = load_pim_config(args.pim_config)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    seqlens = parse_int_list(args.seqlens)
    v_dims = parse_int_list(args.vector_dims)
    m_cols = parse_int_list(args.matrix_cols)

    # default ops
    if any(o in ("score","output") for o in ops) and not seqlens:
        seqlens = [64, 128, 256, 512, 1024, 2048]
    if ("weight" in ops) and (not v_dims or not m_cols):
        v_dims = [args.dim]
        m_cols = [args.dim] + [4*args.dim]

    dic_model = make_dic_model(args.dim, args.n_heads, args.n_kv_heads, seqlens[0] if seqlens else 16)
    row_index_matrix = 0

    for op in ops:
        if op in ("score","output"):
            for L in seqlens:
                trace_path = out_dir / f"{op}_seq{L}_dim{args.dim}_h{args.n_heads}.aim"
                tb_args = make_tb_args_from_pim(pim_cfg, str(trace_path))
                tb_args.seqlen = L
                dic_model["x"] = torch.randn(1, L, args.dim)
                block = TransformerBlock(dic_model, tb_args)
                call_score_or_output(block, op, row_index_matrix, L)
                (trace_path.with_suffix(".json")).write_text(
                    json.dumps({"op":op,"seqlen":L, "dim":args.dim, "n_heads":args.n_heads, **pim_cfg}, indent=2, ensure_ascii=False),
                    encoding="utf-8") #配置文件写json，写aim在函数内部处理
                print(f"[ok] wrote {trace_path}")
        elif op == "weight":
            for V in v_dims:
                for N in m_cols:
                    suffix = "_with_af" if args.with_af else ""
                    trace_path = out_dir / f"weight_vec{V}_col{N}_dim{args.dim}_h{args.n_heads}{suffix}.aim"
                    tb_args = make_tb_args_from_pim(pim_cfg, str(trace_path))
                    block = TransformerBlock(dic_model, tb_args)
                    call_weight(block, row_index_matrix, V, N, args.with_af)
                    (trace_path.with_suffix(".json")).write_text(
                        json.dumps({"op":"weight","vector_dim":V,"matrix_col":N,"with_af":bool(args.with_af),
                                    "dim":args.dim,"n_heads":args.n_heads, **pim_cfg}, indent=2, ensure_ascii=False),
                        encoding="utf-8")
                    print(f"[ok] wrote {trace_path}")
        else:
            raise ValueError(f"Unsupported op: {op}")

if __name__ == "__main__":
    main()
