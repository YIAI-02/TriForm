#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch  
# from TransformerBlock import TransformerBlock  # type: ignore

#原cent中，class transformerblock继承了class pim，如果想使用两个里面的生成trace的函数，实例化一个transformer block即可
#其中的self attention和feed foward整块写完，没有封装成单个算子，所以写单个算子的时候直接copy 代码，没法直接例化函数

# ------------------------------ sweep helpers ------------------------------
def _parse_int_list(s: Optional[str]) -> Optional[list[int]]:
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out or None

def _common_model_presets() -> dict[str, dict[str, list[int]]]:
    """
    Return a compact catalog of common LLM shapes. These are *not* exact specs,
    just representative values to sweep over. You can edit freely.
    """
    return {
        "tiny":   {"dim": [512, 768],    "n_heads": [8, 12],     "seqlen": [128, 256, 512]},
        "bert":   {"dim": [768, 1024],   "n_heads": [12, 16],    "seqlen": [128, 384, 512]},
        "gpt2":   {"dim": [768, 1024],   "n_heads": [12, 16],    "seqlen": [128, 256, 512, 1024]},
        "llama":  {"dim": [4096, 5120],  "n_heads": [32, 40],    "seqlen": [2048, 4096]},
        "mistral":{"dim": [4096],        "n_heads": [32],        "seqlen": [4096, 8192]},
        "common": {"dim": [512, 768, 1024, 2048, 4096],
                   "n_heads": [8, 12, 16, 32],
                   "seqlen": [128, 256, 512, 1024, 2048, 4096]}
    }

def _build_param_grid(
    presets: list[str] | None,
    dims: list[int] | None,
    heads: list[int] | None,
    n_kv_policy: list[str] | list[int] | None,
    max_seqlen: int,
    seq_candidates: list[int] | None,
    ffn_mult: float
) -> list[dict]:
    """
    Compose a sweep over (dim, n_heads, n_kv_heads, ffn_dim, seqlens). 
    filter out invalid combos
    where dim % n_heads != 0. n_kv_heads is derived using policies:
      - "same"    -> n_kv_heads = n_heads    (MHA)
      - "quarter" -> n_kv_heads = max(1, n_heads // 4) (typical GQA)

    """
    cat = _common_model_presets()
    dims_set: set[int] = set(dims or [])
    heads_set: set[int] = set(heads or [])
    seqs_set: set[int] = set()

    # Merge from presets
    for p in (presets or ["common"]):
        if p not in cat:
            continue
        dims_set.update(cat[p]["dim"])
        heads_set.update(cat[p]["n_heads"])
        seqs_set.update(cat[p]["seqlen"])

    # Merge explicit candidates
    if seq_candidates:
        seqs_set.update(seq_candidates)

    # Clip seqlens by max
    seqlens = sorted([s for s in seqs_set if s <= max_seqlen]) or [min(512, max_seqlen)]
    dims = sorted(dims_set) or cat["common"]["dim"]
    heads = sorted(heads_set) or cat["common"]["n_heads"]

    # Build n_kv list
    if not n_kv_policy:
        n_kv_policy = ["same", "quarter"]
    n_kv_list: list[int] = []
    if all(isinstance(x, int) for x in n_kv_policy):  # explicit values
        n_kv_list = list({int(x) for x in n_kv_policy})  # unique
    else:
        # keep the policy tokens; we'll expand per (dim, n_heads)
        pass

    grid: list[dict] = []
    for d in dims:
        for h in heads:
            if d % h != 0:
                continue  # invalid combo
            # decide n_kv candidates
            kv_candidates: list[int]
            if n_kv_list:
                kv_candidates = n_kv_list
            else:
                kv_candidates = []
                if "same" in n_kv_policy:
                    kv_candidates.append(h)
                if "quarter" in n_kv_policy:
                    kv_candidates.append(max(1, h // 4))
                # unique & sorted
                kv_candidates = sorted(set(kv_candidates))
            for hk in kv_candidates:
                grid.append({
                    "dim": d,
                    "n_heads": h,
                    "n_kv_heads": hk,
                    "ffn_dim": int(round(ffn_mult * d)),
                    "seqlens": seqlens
                })
    return grid

# ------------------------------ helper ------------------------------
def _mk_trace_name(op: str, dim: int, n_heads: int, n_kv_heads: int,
                   seqlen: Optional[int], V: Optional[int], N: Optional[int],
                   with_af: bool) -> str:
    parts = [op, f"dim{dim}_h{n_heads}_hk{n_kv_heads}"]
    if seqlen is not None: parts.append(f"seq{seqlen}")
    if V is not None:      parts.append(f"vec{V}")
    if N is not None:      parts.append(f"col{N}")
    if with_af: parts.append("withaf")
    return "_".join(parts) + ".trace"

def _calc_channels(block):
    if getattr(block, "model_parallel", False):
        FC_total_banks   = int(block.total_banks) * int(block.FC_devices)
        channels_required = int(block.num_channels)
    else:
        FC_total_banks   = int(block.total_banks)
        channels_required = int(block.channels_per_block)

    num_channels = int(block.num_channels)
    channels_required = int(channels_required)

    channel_multi_tb_required = int((num_channels // channels_required) * channels_required)
    channel_lst = [channel for channel in range(channel_multi_tb_required)]
    return channel_lst, FC_total_banks, channels_required

def _make_dic_model(dim: int, n_heads: int, n_kv_heads: Optional[int], seqlen: int, ffn_dim: int) -> Dict[str, Any]:
    if n_kv_heads is None:
        n_kv_heads = n_heads
    assert dim % n_heads == 0
    head_dim = dim // n_heads
    TP_param = 1
    assert head_dim > 0, "dim must be divisible by n_heads"
    return {            
        "TP_param": torch.tensor(TP_param),
        "dim": torch.tensor(dim),
        "n_heads": torch.tensor(n_heads),
        "n_kv_heads": torch.tensor(n_kv_heads),
        "x": torch.zeros((1, 1, dim)),
        "SANorm": torch.zeros(dim),
        "FFNNorm": torch.zeros(dim),
        "sa": torch.zeros((1, 1, dim)),
        "h": torch.zeros((1, 1, dim)),
        "out": torch.zeros((1, 1, dim)),
        "wq": torch.zeros((dim // TP_param, dim)),
        "wk": torch.zeros((head_dim * n_kv_heads), dim),
        "wv": torch.zeros((head_dim * n_kv_heads), dim),
        "xq": torch.zeros((1, 1, dim)),
        "xk": torch.zeros((1, 1, head_dim * n_heads)),
        "xv": torch.zeros((1, 1, head_dim * n_heads)),
        "start_pos": torch.tensor(seqlen - 1),
        "cache_k": torch.zeros((1, seqlen, n_kv_heads, head_dim)),
        "cache_v": torch.zeros((1, seqlen, n_kv_heads, head_dim)),
        "scores": torch.zeros((1, n_heads, 1, seqlen)),
        "output": torch.zeros((1, 1, dim)),
        "wo": torch.zeros((dim // TP_param, dim)),
        "w1": torch.zeros((ffn_dim // TP_param, dim)),
        "w3": torch.zeros((ffn_dim // TP_param, dim)),
        "w2": torch.zeros((dim // TP_param, ffn_dim)),
        "ffn": torch.zeros((1, 1, dim))
    }

def _load_pim_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    std: Dict[str, Any] = {}
    alias = {
        "DRAM_column": ["dram_column", "DRAMCol", "dramCol", "dram_col"],
        "DRAM_row": ["dram_row", "DRAMRow", "dramRow", "dram_row"],
        "burst_length": ["burst", "burstLength", "BL"],
        "num_banks": ["banks", "numBanks"],
        "num_channels": ["channels", "numChannels"],
        "threads": ["thread", "nThreads"],
        "reuse_size": ["reuseSize", "reuse", "RS"],
        "channels_per_block": ["channelsPerBlock", "cpb"],
        "max_seq_len": ["maxSeqLen", "max_seq_length"],
    }
    for k, v in cfg.items():
        matched = False
        for stdk, alist in alias.items():
            if k == stdk or k in alist:
                std[stdk] = v
                matched = True
                break
        if not matched:
            std[k] = v
    std.setdefault("DRAM_column", 256)
    std.setdefault("DRAM_row", 64)
    std.setdefault("burst_length", 16)
    std.setdefault("num_banks", 8)
    std.setdefault("num_channels", 4)
    std.setdefault("threads", 1)
    std.setdefault("reuse_size", 32)
    std.setdefault("channels_per_block", None)
    std.setdefault("max_seq_len", 4096)
    return std

def _make_tb_args_from_pim(cfg: Dict[str, Any], trace_file: str):
    from types import SimpleNamespace
    return SimpleNamespace(
        DRAM_column        = int(cfg["DRAM_column"]),
        DRAM_row           = int(cfg["DRAM_row"]),
        burst_length       = int(cfg["burst_length"]),
        num_banks          = int(cfg["num_banks"]),
        num_channels       = int(cfg["num_channels"]),
        threads            = int(cfg["threads"]),
        reuse_size         = int(cfg["reuse_size"]),
        channels_per_block = int(cfg["channels_per_block"]),
        max_seq_len        = int(cfg["max_seq_len"]),
        only_trace         = True, 
        op_trace           = False, 
        trace_file         = trace_file,
        pim_compute        = True, 
        model              = "llama_like", 
        embedding          = "rope",
        seqlen             = 16, 
        model_parallel     = False, 
        FC_devices         = 1,
        pipeline_parallel  = False, 
        inter_device_attention=False, 
        only_FC            = False,
        trace_prepare      = False, 
        trace_norm         = False, 
        trace_fc_kqvo      = False, 
        trace_attention    = False,
        trace_softmax      = False, 
        trace_fc_ffn       = False, 
        trace_activation   = False,
        GEMV               = "reuse-GB",
    )

def _ensure_cent_on_path(start: Optional[Path] = None) -> tuple[Path, Path]:
    here = (start or Path(__file__)).resolve()
    for p in [here.parent] + list(here.parents):
        cand = p / "submodules" / "CENT" / "cent_simulation"
        if cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return p, cand
    raise RuntimeError(f"Cannot find 'submodules/CENT/cent_simulation' above {here}")

def _emit_single_op_trace(block, op:str, dim:int, n_heads:int, n_kv_heads:int, ffn_dim:int, seqlens:int):
    channel_lst, FC_total_banks, channels_required = _calc_channels(block)
    if op in ("q_proj", "k_proj", "v_proj", "wo_proj", "ffn_up", "ffn_gate", "ffn_down"):
        V = N = None
        if  op == "q_proj":
            row_tag, V, N = "wq_row_index", dim, dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "k_proj":
            row_tag, V, N = "wk_row_index", dim, n_kv_heads * dim/n_heads
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "v_proj":
            row_tag, V, N = "wv_row_index", dim, n_kv_heads * dim/n_heads
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "wo_proj":
            row_tag, V, N = "wo_row_index", dim, dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_sa_weight")
        elif op == "ffn_up":
            row_tag, V, N = "w1_row_index", dim, ffn_dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_ffn_weight")
        elif op == "ffn_gate":
            row_tag, V, N = "w3_row_index", dim, ffn_dim
            block.Vector_Matrix_Mul_weight_af_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_ffn_weight")
        elif op == "ffn_down":
            row_tag, V, N = "w2_row_index", ffn_dim, dim
            block.Vector_Matrix_Mul_weight_pim_only_trace(channel_lst, getattr(block, row_tag), V, N, FC_total_banks, "breakdown_ffn_weight")

    elif op in ("score","softmax","output"):
        for S in seqlens or [1]:
            if op == "score":
                block.Vector_Matrix_Mul_score_pim_only_trace(block.cache_k_row_index, S, "breakdown_sa_score")
            elif op == "output":
                block.Vector_Matrix_Mul_output_pim_only_trace(block.cache_v_row_index, S, "breakdown_sa_output")
            elif op == "softmax":
                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = block.DRAM_column // block.burst_length if r < rows_per_score-1 else (S - block.DRAM_column*r - 1)//block.burst_length + 1
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)
                block.time["RD_SBK"] += block.timing_constant["RD_SBK"] + (S * block.n_heads) // block.burst_length
                block.load_from_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 2, S)
                block.time["WR_SBK"] += block.timing_constant["WR_SBK"] + (S * block.n_heads) // block.burst_length
                block.store_for_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 0, S)
                rows_per_score = (S - 1) // block.DRAM_column + 1
                for r in range(rows_per_score):
                    op_size = block.DRAM_column // block.burst_length if r < rows_per_score-1 else (S - block.DRAM_column*r - 1)//block.burst_length + 1
                    block.EWMUL_only_trace(channel_lst, block.scores_row_index + r, op_size)
                block.load_from_EWMUL_score_only_trace(channels_required, block.scores_row_index, block.total_banks, 2, S)

    elif op == "rmsnorm":
        input_len = (dim - 1)//(block.total_banks//2) + 1
        block.WR_BIAS_only_trace(channel_lst)
        block.MAC_ABK_only_trace(channel_lst, block.x_row_index, (input_len - 1)//block.burst_length + 1, "breakdown_sa_pow")
        block.RD_MAC_only_trace(channel_lst)
        ew_len = (dim - 1)//(block.total_banks//4) + 1
        ew_banks = (dim - 1)//ew_len + 1
        block.time["WR_SBK"] += block.timing_constant["WR_SBK"] + dim // block.burst_length
        block.store_for_EWMUL_input_only_trace(channels_required, ew_banks, 1, block.x_copy_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.x_copy_row_index, (ew_len - 1)//block.burst_length + 1)
        for bank in range(block.num_banks):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.x_copy_row_index, (ew_len - 1)//block.burst_length + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank-1, block.SANorm_row_index, (ew_len - 1)//block.burst_length + 1)
        block.EWMUL_only_trace(channel_lst, block.SANorm_row_index, (ew_len - 1)//block.burst_length + 1)
        block.time["RD_SBK"] += block.timing_constant["RD_SBK"] + block.dim // block.burst_length
        block.load_from_EWMUL_input_only_trace(channels_required, ew_banks, 2, block.SANorm_row_index, ew_len)
        block.SYNC_only_trace()
    
    elif op == "rope":
        ew_len = (dim/n_heads - 1)//(block.total_banks//4) + 1
        ew_size = (ew_len - 1)//block.burst_length + 1
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, (dim - 1)//ew_len + 1, 1, block.xq_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.xq_row_index, ew_size)
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, (dim - 1)//ew_len + 1, 1, block.xk_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.xk_row_index, ew_size)
        
    elif op in ("silu", "gelu"):
        ew_len = (ffn_dim - 1)//(block.total_banks//4) + 1
        ew_banks = (ffn_dim - 1)//ew_len + 1
        block.time["WR_SBK"] += block.timing_constant["WR_SBK"] + ffn_dim // block.burst_length
        block.store_for_EWMUL_input_only_trace(block.channels_per_block, ew_banks, 1, block.ffn_row_index, ew_len)
        block.EWMUL_only_trace(channel_lst, block.ffn_row_index, (ew_len - 1)//block.burst_length + 1)
        for bank in range(block.num_banks):
            if bank % 4 == 2:
                block.COPY_BK_GB_only_trace(channel_lst, bank, block.ffn_row_index, (ew_len - 1)//block.burst_length + 1)
                block.COPY_GB_BK_only_trace(channel_lst, bank-1, block.ffn_row_index, (ew_len - 1)//block.burst_length + 1)
        block.EWMUL_only_trace(channel_lst, block.ffn_row_index, (ew_len - 1)//block.burst_length + 1)
        block.time["RD_SBK"] += block.timing_constant["RD_SBK"] + ffn_dim// block.burst_length
        block.SYNC_only_trace()
        
    elif op == "residual":
        op_size = block.dim // block.burst_length
        block.EWADD_only_trace(op_size)
    else:
        raise ValueError(f"Unsupported op: {op}")
    
    if hasattr(block, "file") and block.file:
        block.file.write("AiM EOC\n")
        block.file.flush()

def main():
    ap = argparse.ArgumentParser(description="CENT op trace generator")
    ap.add_argument("--pim-config", type=Path, required=True)
    ap.add_argument("--ops", type=str, default="q_proj, k_proj, v_proj, wo_proj, ffn_up, ffn_gate, ffn_down,score,softmax,output,rmsnorm,rope,silu,gelu,residual")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["single","sweep"], default="single")
    ap.add_argument("--with-af", action="store_true")

    # single mode
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--n-kv-heads", type=int, default=None)
    ap.add_argument("--seqlens", type=str, default=None)
    ap.add_argument("--ffn-mult", type=float, default=4.0)
    ap.add_argument("--model-shape", type=Path, default=None)

    # sweep mode
    ap.add_argument("--presets", type=str, default="common")
    ap.add_argument("--dims", type=str, default=None)
    ap.add_argument("--heads", type=str, default=None)
    ap.add_argument("--kv-head-policy", type=str, default="same,quarter")
    ap.add_argument("--seq-candidates", type=str, default=None)
    ap.add_argument("--max-seqlen", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()
    _ensure_cent_on_path()
    try:
        from Llama import TransformerBlockLlama as TransformerBlock
    except Exception:
        from TransformerBlock import TransformerBlock

    pim_cfg = _load_pim_config(args.pim_config)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ops = [o.strip() for o in (args.ops or "").split(",") if o.strip()]
    if not raw_ops:
        raw_ops = ["q_proj","k_proj","v_proj","wo_proj","score","softmax","output",
                   "ffn_up","ffn_gate","ffn_down","rmsnorm","rope","silu","gelu","residual"]
    ops: list[str] = raw_ops

    jobs: list[dict] = []

    if args.mode == "single":
        dim = args.dim
        n_heads = args.n_heads
        n_kv_heads = args.n_kv_heads
        if args.model_shape:
            try:
                shape = json.loads(Path(args.model_shape).read_text(encoding="utf-8"))
                dim = dim or shape.get("dim") or shape.get("hidden_size")
                n_heads = n_heads or shape.get("n_heads") or shape.get("num_attention_heads")
                n_kv_heads = n_kv_heads or shape.get("n_kv_heads") or shape.get("num_key_value_heads")
            except Exception as e:
                raise SystemExit(f"Failed to read --model-shape: {e}")
        if dim is None or n_heads is None:
            raise SystemExit("single mode requires --dim and --n-heads")
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if dim % n_heads != 0:
            raise SystemExit(f"dim {dim} must be divisible by n_heads {n_heads}")
        ffn_dim = int(round(args.ffn_mult * dim))
        seqlens = _parse_int_list(args.seqlens) or [min(2048, args.max_seqlen)]
        jobs = [{
            "dim": dim,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "ffn_dim": ffn_dim,
            "seqlens": seqlens,
        }]
    else:
        presets = [p.strip() for p in (args.presets or "common").split(",") if p.strip()]
        dims = _parse_int_list(args.dims)
        heads = _parse_int_list(args.heads)
        # kv policy could be ints or tokens
        kv_tokens = [t.strip() for t in (args.kv_head_policy or "").split(",") if t.strip()]
        kv_policy: list[str] | list[int]
        # If all tokens are ints, treat as explicit kv heads
        try:
            kv_policy = [int(t) for t in kv_tokens]
        except ValueError:
            kv_policy = kv_tokens or ["same","quarter"]
        seq_candidates = _parse_int_list(args.seq_candidates)
        grid = _build_param_grid(
            presets=presets,
            dims=dims, heads=heads,
            n_kv_policy=kv_policy,
            max_seqlen=args.max_seqlen,
            seq_candidates=seq_candidates,
            ffn_mult=args.ffn_mult,
        )
        if args.limit is not None and args.limit > 0:
            grid = grid[:args.limit]
        jobs = grid

    # 生成trace并保存metadata
    total_written = 0
    metadata_list = []
    
    for job in jobs:
        dim = job["dim"]
        n_heads = job["n_heads"]
        n_kv_heads = job["n_kv_heads"]
        ffn_dim = job["ffn_dim"]
        seqlens = job["seqlens"]
        max_seq_for_alloc = max(seqlens) if seqlens else 1

        for op in ops:
            trace_name = _mk_trace_name(op, dim, n_heads, n_kv_heads, max_seq_for_alloc, None, None, args.with_af)
            op_dir = out_dir / op
            op_dir.mkdir(parents=True, exist_ok=True)
            trace_path = op_dir / trace_name

            block_args = _make_tb_args_from_pim(pim_cfg, str(trace_path))
            block_args.op_trace = True
            block_args.seqlen = max_seq_for_alloc

            dic_model = _make_dic_model(dim, n_heads, n_kv_heads, block_args.seqlen, ffn_dim)
            block = TransformerBlock(dic_model, block_args)
            
            if hasattr(block, "memory_mapping"):
                block.memory_mapping()

            try:
                _emit_single_op_trace(block, op, dim, n_heads, n_kv_heads, ffn_dim, seqlens)
                
                # 保存metadata到JSON
                meta = {
                    "op": op,
                    "dim": dim,
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "ffn_dim": ffn_dim,
                    "seqlen": max_seq_for_alloc,
                    "vector_dim": dim,
                    "matrix_col": dim if op in ["q_proj", "wo_proj"] else (
                        int(n_kv_heads * dim / n_heads) if op in ["k_proj", "v_proj"] else
                        ffn_dim if op in ["ffn_up", "ffn_gate"] else
                        dim if op == "ffn_down" else None
                    ),
                    "with_af": 1 if args.with_af else 0,
                    "trace_file": str(trace_path),
                    **pim_cfg 
                }
                
                json_path = trace_path.with_suffix(".json")
                json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                
                metadata_list.append(meta)
                total_written += 1
                
            finally:
                f = getattr(block, "file", None)
                try:
                    if f:
                        f.close()
                except Exception:
                    pass

    # 保存所有metadata到一个汇总文件
    summary_path = out_dir / "trace_metadata.json"
    summary_path.write_text(json.dumps(metadata_list, indent=2), encoding="utf-8")

    print(f"[ok] wrote {total_written} traces under: {out_dir}")
    print(f"[ok] metadata summary: {summary_path}")

if __name__ == "__main__":
    main()