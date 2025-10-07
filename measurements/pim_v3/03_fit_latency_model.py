#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, sys, math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# 共享：trace解析（保留兼容），以及模型形状读取（predict 的默认值可来自 shape）
from aim_shared import (
    FEATURE_NAMES, FEATURE_HAS_SIZE, parse_features_from_trace,  # 兼容旧逻辑
    load_model_shape
)

OP_KEYS = ("score", "output", "weight", "weight_af")

# ------------------------------ 工具 ------------------------------
def _normalize_op_label(row: Dict[str, Any]) -> str:
    """从 CSV 行推断标准 op_label（score/output/weight/weight_af）。"""
    op_label = (row.get("op_label") or "").strip()
    if op_label:
        return op_label
    op = (row.get("op") or "").strip()
    waf = str(row.get("with_af", "0")).strip().lower()
    if op == "weight" and waf in ("1", "true", "yes", "y"):
        return "weight_af"
    if op in ("score", "output", "weight"):
        return op
    return "unknown"

def _read_rows(csv_path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        headers = rd.fieldnames or []
    return rows, headers

# ------------------------------ 公式基函数 ------------------------------
# 变量简记：L=seqlen, V=vector_dim, N=matrix_col, H=n_heads
# 我们选择的“相对独立”的自变量：
# - score/output：主要与 L、H 相关（注意 dim 与 H 存在线性关系 dim=H*head_dim，因而不单独作为自变量）
# - weight/weight_af：主要与 V、N 相关，H 作为并行/调度维度可能带来线性项
# 采用低阶多项式 + 交互项；若某列缺失则按 0 处理。
FORMULA_SPEC: Dict[str, List[Tuple[str, str]]] = {
    "score": [
        ("1", "1"),
        ("L", "seqlen"),
        ("L2", "seqlen**2"),
        ("H", "n_heads"),
        ("LxH", "seqlen*n_heads"),
    ],
    "output": [
        ("1", "1"),
        ("L", "seqlen"),
        ("H", "n_heads"),
        ("LxH", "seqlen*n_heads"),
    ],
    "weight": [
        ("1", "1"),
        ("V", "vector_dim"),
        ("N", "matrix_col"),
        ("VxN", "vector_dim*matrix_col"),
        ("H", "n_heads"),
    ],
    "weight_af": [
        ("1", "1"),
        ("V", "vector_dim"),
        ("N", "matrix_col"),
        ("VxN", "vector_dim*matrix_col"),
        ("H", "n_heads"),
    ],
}

def _eval_feature(expr: str, row: Dict[str, Any]) -> float:
    # 提取变量并做安全求值（仅支持 +-* / ** 和变量）
    L = float(row.get("seqlen", 0) or 0)
    V = float(row.get("vector_dim", 0) or 0)
    N = float(row.get("matrix_col", 0) or 0)
    H = float(row.get("n_heads", 0) or 0)
    # 替换变量名为数值
    expr2 = expr.replace("seqlen", str(L)).replace("vector_dim", str(V)).replace("matrix_col", str(N)).replace("n_heads", str(H))
    return float(eval(expr2, {"__builtins__": {}}, {}))

def _build_X_y_for_op(rows: List[Dict[str, Any]], opk: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    spec = FORMULA_SPEC[opk]
    X = []
    y = []
    for row in rows:
        if _normalize_op_label(row) != opk:
            continue
        cycles = row.get("cycles", "")
        if cycles in ("", None):
            continue
        try:
            yy = float(cycles)
        except Exception:
            continue
        feats = [ _eval_feature(expr, row) for (_, expr) in spec ]
        X.append(feats)
        y.append(yy)
    if not X:
        return np.zeros((0, len(spec))), np.zeros((0,)), [name for (name, _) in spec]
    return np.array(X, dtype=float), np.array(y, dtype=float), [name for (name, _) in spec]

def _fit_ls(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """最小二乘拟合，返回权重和指标（rmse、r2）。"""
    if X.shape[0] == 0:
        return np.zeros((X.shape[1],)), {"rmse": float("nan"), "r2": float("nan")}
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ w
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2))) if y.size else float("nan")
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if y.size else float("nan")
    r2 = float(1 - ss_res/ss_tot) if ss_tot not in (0.0, 0.0) else float("nan")
    return w.flatten(), {"rmse": rmse, "r2": r2}

# ------------------------------ 拟合：fit（按算子分别拟合“显式公式”） --------------------------------
def fit_model_formula(results_csv: Path, out_model: Path, out_summary_csv: Optional[Path]) -> None:
    rows, headers = _read_rows(results_csv)
    models: Dict[str, Any] = {}
    summary_rows = []

    for opk in OP_KEYS:
        X, y, basis = _build_X_y_for_op(rows, opk)
        if X.shape[0] == 0:
            continue
        w, metrics = _fit_ls(X, y)
        # 组装可读公式
        terms = [f"{coef:.6g}*{name}" if name != "1" else f"{coef:.6g}" for coef, name in zip(w, basis)]
        expr = " + ".join(terms)
        models[opk] = {
            "basis": basis,
            "coeffs": [float(x) for x in w],
            "expr": f"cycles ≈ {expr}",
            "num_samples": int(X.shape[0]),
            "metrics": metrics,
        }
        summary_rows.append({
            "op_label": opk,
            "expr": models[opk]["expr"],
            "num_samples": int(X.shape[0]),
            "rmse": f"{metrics['rmse']:.6g}",
            "r2": f"{metrics['r2']:.6g}",
        })

    if not models:
        print("没有可用数据行进行拟合（检查 cycles 或 op_label）。", file=sys.stderr)
        sys.exit(1)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps({"per_op_formula": models}, indent=2), encoding="utf-8")
    print(f"[ok] wrote per-op **formula** model JSON -> {out_model}")

    if out_summary_csv:
        with out_summary_csv.open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=["op_label", "expr", "num_samples", "rmse", "r2"])
            wcsv.writeheader()
            wcsv.writerows(summary_rows)
        print(f"[ok] wrote formula fit summary -> {out_summary_csv}")

# ------------------------------ 预测：predict（优先用公式，不再生成 trace） ------------------------------
def predict_with_formula(model_json: Path,
                         op: str,
                         vector_dim: Optional[int],
                         matrix_col: Optional[int],
                         seqlen: Optional[int],
                         n_heads: Optional[int]) -> float:
    obj = json.loads(Path(model_json).read_text(encoding="utf-8"))
    op_key = "weight_af" if (op == "weight" and bool(matrix_col) and bool(vector_dim) and bool(seqlen) is not None and False) else ( "weight_af" if op=="weight" and False else op )
    # 简化：预测接口保持与之前一致：如果指定了 --with-af，则上层会传 op='weight' 且 with_af=True；在 auto_fit_pipeline 中我们已经写了两行分别调用。
    # 这里不再区分，通过调用者传入正确的 op：score/output/weight/weight_af
    if "per_op_formula" not in obj:
        raise RuntimeError("模型文件缺少 per_op_formula。请先用 `fit` 生成。")
    model = obj["per_op_formula"].get(op)
    if model is None:
        raise RuntimeError(f"模型文件中缺少算子 {op} 的公式。")

    basis = model["basis"]
    coeffs = model["coeffs"]
    # 构造行字典
    row = {
        "seqlen": seqlen or 0,
        "vector_dim": vector_dim or 0,
        "matrix_col": matrix_col or 0,
        "n_heads": n_heads or 0,
    }
    # 计算特征
    def feval(name: str) -> float:
        if name == "1": return 1.0
        if name == "L": return float(row["seqlen"])
        if name == "L2": return float(row["seqlen"])**2
        if name == "H": return float(row["n_heads"])
        if name == "LxH": return float(row["seqlen"]) * float(row["n_heads"])
        if name == "V": return float(row["vector_dim"])
        if name == "N": return float(row["matrix_col"])
        if name == "VxN": return float(row["vector_dim"]) * float(row["matrix_col"])
        return 0.0
    feats = np.array([feval(n) for n in basis], dtype=float)
    w = np.array(coeffs, dtype=float)
    return float(feats.dot(w))

# ------------------------------ CLI ------------------------------
def cmd_fit(args):
    out_summary_csv = args.summary_csv if args.summary_csv else (args.out.parent / (args.out.stem + "_fit_summary.csv"))
    fit_model_formula(args.results_csv, args.out, out_summary_csv)

def cmd_predict(args):
    # 支持直接从 --model-shape 读取 n_heads（如果未提供）
    n_heads = args.n_heads
    if n_heads is None and args.model_shape:
        shape = load_model_shape(args.model_shape)
        n_heads = shape["n_heads"]
    pred = predict_with_formula(
        model_json=args.model,
        op=("weight_af" if (args.op == "weight" and args.with_af) else args.op),
        vector_dim=args.vector_dim,
        matrix_col=args.matrix_col,
        seqlen=args.seqlen,
        n_heads=n_heads,
    )
    print(int(round(pred)))

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)

    ap_fit = sub.add_parser("fit", help="Fit *separate* closed-form latency formulas per op from CSV")
    ap_fit.add_argument("--results-csv", type=Path, required=True)
    ap_fit.add_argument("--out", type=Path, required=True, help="Output JSON; also writes <stem>_fit_summary.csv by default or --summary-csv")
    ap_fit.add_argument("--summary-csv", type=Path, default=None)
    ap_fit.set_defaults(func=cmd_fit)

    ap_pred = sub.add_parser("predict", help="Predict latency using per-op formulas (no trace needed)")
    ap_pred.add_argument("--model", type=Path, required=True)
    ap_pred.add_argument("--op", type=str, choices=["score","output","weight","weight_af"], required=True)
    ap_pred.add_argument("--vector-dim", type=int, default=None, help="for weight/weight_af")
    ap_pred.add_argument("--matrix-col", type=int, default=None, help="for weight/weight_af")
    ap_pred.add_argument("--with-af", action="store_true", help="(compat) if --op weight and --with-af, internally uses weight_af")
    ap_pred.add_argument("--seqlen", type=int, default=None, help="for score/output")
    ap_pred.add_argument("--n-heads", type=int, default=None)
    ap_pred.add_argument("--model-shape", type=Path, default=None, help="If provided and --n-heads missing, read from shape")
    ap_pred.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
