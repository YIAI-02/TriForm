#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, re, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# ------------------------- 与 run_ramulator 对齐的特征表 -------------------------
FEATURE_SPECS = [
    ("MAC_ABK",     True,  [r"^AiM\s+MAC_ABK\s+(\d+)"]),
    ("MAC_BK_BK",   True,  [r"^AiM\s+MAC_BK_BK\s+(\d+)"]),
    ("MAC_BK_GB",   True,  [r"^AiM\s+MAC_BK_GB\s+(\d+)"]),
    ("WR_GB",       True,  [r"^AiM\s+WR_GB\s+(\d+)"]),
    ("COPY_BK_GB",  True,  [r"^AiM\s+COPY_BK_GB\s+(\d+)", r"^AiM\s+COPY_BKGB\s+(\d+)"]),
    ("COPY_GB_BK",  True,  [r"^AiM\s+COPY_GB_BK\s+(\d+)", r"^AiM\s+COPY_GBBK\s+(\d+)"]),
    ("EWMUL",       True,  [r"^AiM\s+EWMUL\s+(\d+)"]),
    ("EWADD",       True,  [r"^AiM\s+EWADD\s+(\d+)"]),
    ("AF",          False, [r"^AiM\s+AF\b"]),
    ("RD_MAC",      False, [r"^AiM\s+RD_MAC\b"]),
    ("RD_AF",       False, [r"^AiM\s+RD_AF\b"]),
    ("WR_BIAS",     False, [r"^AiM\s+WR_BIAS\b"]),
    ("RD_SBK",      False, [r"^AiM\s+RD_SBK\b"]),
    ("WR_SBK",      False, [r"^AiM\s+WR_SBK\b"]),
]
FEATURE_NAMES = [n for n, _, _ in FEATURE_SPECS]
FEATURE_HAS_SIZE = {n: has for n, has, _ in FEATURE_SPECS}
AIM_PATTERNS = {n: [re.compile(p) for p in pats] for n, _, pats in FEATURE_SPECS}

def parse_features_from_trace(trace_path: Path) -> Dict[str, Tuple[int, int]]:
    feats = {name: [0, 0] for name in FEATURE_NAMES}
    with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for name in FEATURE_NAMES:
                for pat in AIM_PATTERNS[name]:
                    m = pat.search(line)
                    if not m:
                        continue
                    feats[name][0] += 1
                    if FEATURE_HAS_SIZE[name] and m.lastindex:
                        try:
                            feats[name][1] += int(m.group(1))
                        except Exception:
                            pass
                    break
    return {k: (v[0], v[1]) for k, v in feats.items()}

# ------------------------------ 拟合：fit --------------------------------
def fit_model_from_csv(results_csv: Path, traces_dir: Optional[Path], out_model: Path) -> None:
    """
    - 若 CSV 包含 <NAME>_calls 与 <NAME>_opsize 列：直接用它们拟合；
    - 否则，需要 --traces-dir，脚本会解析 .aim。
    """
    with results_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        headers = rd.fieldnames or []

    has_feature_cols = all((f"{name}_calls" in headers and f"{name}_opsize" in headers) for name in FEATURE_NAMES)

    X_rows, y = [], []
    for row in rows:
        cyc = row.get("cycles", "")
        if cyc == "" or cyc is None:
            continue
        try:
            cycles = int(cyc)
        except Exception:
            continue

        if has_feature_cols:
            calls = []
            opsizes = []
            for name in FEATURE_NAMES:
                c = int(row.get(f"{name}_calls", 0) or 0)
                s = int(row.get(f"{name}_opsize", 0) or 0)
                if not FEATURE_HAS_SIZE[name]:
                    s = 0
                calls.append(c)
                opsizes.append(s)
        else:
            if not traces_dir:
                print("CSV 无特征列，且未提供 --traces-dir 以解析 .aim。", file=sys.stderr)
                sys.exit(1)
            tpath = Path(row["trace"])
            if not tpath.is_file():
                tpath = traces_dir / Path(row["trace"]).name
            feats = parse_features_from_trace(tpath)
            calls = [feats[name][0] for name in FEATURE_NAMES]
            opsizes = [feats[name][1] if FEATURE_HAS_SIZE[name] else 0 for name in FEATURE_NAMES]

        X_rows.append(calls + opsizes)
        y.append(cycles)

    if not X_rows:
        print("没有可用数据行（可能 cycles 为空或 CSV 为空）。", file=sys.stderr)
        sys.exit(1)

    X = np.array(X_rows, dtype=float)
    yv = np.array(y, dtype=float).reshape(-1, 1)

    # 最小二乘求解
    w, *_ = np.linalg.lstsq(X, yv, rcond=None) #cycles = w1*calls + w2*opsizes w=[w1,w2]
    w = w.flatten()
    n = len(FEATURE_NAMES)
    model = {
        "feature_order_calls": FEATURE_NAMES,
        "feature_order_opsize": FEATURE_NAMES,
        "weights_calls": w[:n].tolist(),
        "weights_opsize": w[n:].tolist(),
    }
    out_model.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"[ok] wrote model to {out_model}")

# ------------------------------ 预测：predict ------------------------------
def predict_cycles_with_model(model_json: Path,
                              op: str,
                              vector_dim: Optional[int],
                              matrix_col: Optional[int],
                              seqlen: Optional[int],
                              with_af: bool,
                              dim: int, n_heads: int, n_kv_heads: Optional[int],
                              DRAM_column: int, DRAM_row: int, burst_length: int,
                              num_banks: int, num_channels: int,
                              max_seq_len: int) -> float:
    """
    生成一个极小 only_trace，抽取特征（calls/opsize），按线性模型估计 cycles。
    - 如果 op=weight 且 with_af=True，会在末尾追加 AF 与 RD_AF 两条 trace。
    """
    # --- 引导导入 CENT ---
    _HERE = Path(__file__).resolve()
    PROJECT_ROOT = None
    for p in [_HERE.parent] + list(_HERE.parents):
        cand = p / "submodules" / "CENT" / "cent_simulation"
        if cand.exists():
            PROJECT_ROOT = p
            CENT_SIM_DIR = cand
            break
    if PROJECT_ROOT is None:
        raise RuntimeError("Cannot find submodules/CENT/cent_simulation from {}".format(_HERE))
    if str(CENT_SIM_DIR) not in sys.path:
        sys.path.insert(0, str(CENT_SIM_DIR))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from TransformerBlock import TransformerBlock  # type: ignore
    import torch  # type: ignore
    from types import SimpleNamespace

    if n_kv_heads is None:
        n_kv_heads = n_heads
    head_dim = dim // n_heads

    # 构造 args 与 dic_model
    tmp_trace = _HERE.parent / "_tmp_predict.aim"
    tb_args = SimpleNamespace(
        DRAM_column=DRAM_column, DRAM_row=DRAM_row, burst_length=burst_length,
        num_banks=num_banks, num_channels=num_channels, threads=1, reuse_size=32,
        channels_per_block=None, max_seq_len=max_seq_len,
        only_trace=True, op_trace=False, trace_file=str(tmp_trace),
        pim_compute=True, model="llama_like", embedding="rope",
        seqlen=seqlen or 16, model_parallel=False, FC_devices=1,
        pipeline_parallel=False, inter_device_attention=False, only_FC=False,
        trace_prepare=False, trace_norm=False, trace_fc_kqvo=False, trace_attention=False,
        trace_softmax=False, trace_fc_ffn=False, trace_activation=False,
    )
    dic_model = {
        "TP_param": torch.tensor(1),
        "dim": torch.tensor(dim),
        "n_heads": torch.tensor(n_heads),
        "n_kv_heads": torch.tensor(n_kv_heads),
        "x": torch.randn(1, tb_args.seqlen, dim),
        "SANorm": torch.ones(dim),
        "FFNNorm": torch.ones(dim),
        "start_pos": 0,
        "freqs_cis": torch.zeros(tb_args.seqlen, head_dim, dtype=torch.complex64),
        "sa": torch.zeros(1, n_heads, tb_args.seqlen, tb_args.seqlen),
        "h": torch.zeros(1, n_heads, tb_args.seqlen, head_dim),
        "out": torch.zeros(1, tb_args.seqlen, dim),
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
        "cache_k": torch.randn(1, tb_args.seqlen, n_kv_heads, head_dim),
        "cache_v": torch.randn(1, tb_args.seqlen, n_kv_heads, head_dim),
        "scores": torch.randn(1, n_heads, 1, tb_args.seqlen),
        "output": torch.zeros(1, tb_args.seqlen, dim),
        "ffn": torch.zeros(1, tb_args.seqlen, dim),
    }
    block = TransformerBlock(dic_model, tb_args)

    # 调用单算子，生成极小 trace（使用 *_only_trace）
    row_index_matrix = 0
    TIMING = {
        "score":  "breakdown_sa_score",
        "output": "breakdown_sa_output",
        "weight": "breakdown_sa_weight",
    }
    if op == "score":
        if seqlen is None:
            raise ValueError("score 需要 --seqlen")
        block.Vector_Matrix_Mul_score_pim_only_trace(row_index_matrix, seqlen, TIMING["score"])
    elif op == "output":
        if seqlen is None:
            raise ValueError("output 需要 --seqlen")
        block.Vector_Matrix_Mul_output_pim_only_trace(row_index_matrix, seqlen, TIMING["output"])
    elif op == "weight":
        if vector_dim is None or matrix_col is None:
            raise ValueError("weight 需要 --vector-dim 与 --matrix-col")
        channel_lst = [i for i in range(block.num_channels)]
        total_banks = block.FC_total_banks
        block.Vector_Matrix_Mul_weight_pim_only_trace(
            channel_lst, row_index_matrix, vector_dim, matrix_col, total_banks, TIMING["weight"]
        )
        if with_af:
            block.AF_only_trace(channel_lst)
            block.RD_AF_only_trace(channel_lst)
    else:
        raise ValueError(f"Unsupported op: {op}")

    # 解析特征，应用线性模型
    feats = parse_features_from_trace(tmp_trace)
    model = json.loads(Path(model_json).read_text(encoding="utf-8"))
    w_calls = np.array(model["weights_calls"], dtype=float)
    w_opsz  = np.array(model["weights_opsize"], dtype=float)

    x_calls = np.array([feats[name][0] for name in FEATURE_NAMES], dtype=float)
    x_opsz  = np.array([feats[name][1] if FEATURE_HAS_SIZE[name] else 0 for name in FEATURE_NAMES], dtype=float)
    pred = float(x_calls.dot(w_calls) + x_opsz.dot(w_opsz))
    try:
        tmp_trace.unlink(missing_ok=True)
    except Exception:
        pass
    return pred

# ------------------------------ CLI ------------------------------
def cmd_fit(args):
    traces_dir = args.traces_dir if args.traces_dir else None
    fit_model_from_csv(args.results_csv, traces_dir, args.out)

def cmd_predict(args):
    pred = predict_cycles_with_model(
        model_json=args.model,
        op=args.op,
        vector_dim=args.vector_dim,
        matrix_col=args.matrix_col,
        seqlen=args.seqlen,
        with_af=args.with_af,
        dim=args.dim, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
        DRAM_column=args.DRAM_column, DRAM_row=args.DRAM_row, burst_length=args.burst_length,
        num_banks=args.num_banks, num_channels=args.num_channels, max_seq_len=args.max_seq_len
    )
    print(int(round(pred)))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    ap_fit = sub.add_parser("fit", help="Fit linear model from enhanced CSV (or fallback to traces)")
    ap_fit.add_argument("--results-csv", type=Path, required=True)
    ap_fit.add_argument("--traces-dir", type=Path, default=None, help="Optional: only needed if CSV lacks feature columns")
    ap_fit.add_argument("--out", type=Path, required=True)
    ap_fit.set_defaults(func=cmd_fit)

    ap_pred = sub.add_parser("predict", help="Predict latency for a new size (single operator)")
    ap_pred.add_argument("--model", type=Path, required=True)
    ap_pred.add_argument("--op", type=str, choices=["score","output","weight"], required=True)
    ap_pred.add_argument("--vector-dim", type=int, default=None, help="for weight")
    ap_pred.add_argument("--matrix-col", type=int, default=None, help="for weight")
    ap_pred.add_argument("--with-af", action="store_true", help="append AF & RD_AF after weight GEMV")
    ap_pred.add_argument("--seqlen", type=int, default=None, help="for score/output")
    ap_pred.add_argument("--dim", type=int, default=256)
    ap_pred.add_argument("--n-heads", type=int, default=8)
    ap_pred.add_argument("--n-kv-heads", type=int, default=None)
    ap_pred.add_argument("--DRAM-column", type=int, default=256)
    ap_pred.add_argument("--DRAM-row", type=int, default=64)
    ap_pred.add_argument("--burst-length", type=int, default=16)
    ap_pred.add_argument("--num-banks", type=int, default=8)
    ap_pred.add_argument("--num-channels", type=int, default=4)
    ap_pred.add_argument("--max-seq-len", type=int, default=4096)
    ap_pred.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
