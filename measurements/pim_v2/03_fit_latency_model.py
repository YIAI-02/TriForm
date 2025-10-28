# ===== measurements/pim/03_fit_latency_model.py  （替换整个文件）=====
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# ------------------------------ OP & Basis ------------------------------
OP_KEYS = (
    # attention
    "score", "attn_score", "softmax", "attn_out", "output",
    # gemv / fc
    "weight", "weight_af", "q_proj", "k_proj", "v_proj", "wo_proj",
    "ffn_up", "ffn_gate", "ffn_down",
    # vector-ish
    "rmsnorm", "rope", "silu", "gelu", "residual",
)

FORMULA_SPEC: Dict[str, List[str]] = {
    # L=seqlen, H=n_heads, V=vector_dim, N=matrix_col
    "score":      ["1","L","L2","H","LxH"],
    "attn_score": ["1","L","L2","H","LxH"],
    "softmax":    ["1","L","H","LxH"],
    "attn_out":   ["1","L","H","LxH"],
    "output":     ["1","L","H","LxH"],

    "weight":     ["1","V","N","VxN","H"],
    "weight_af":  ["1","V","N","VxN","H"],
    "q_proj":     ["1","V","N","VxN","H"],
    "k_proj":     ["1","V","N","VxN","H"],
    "v_proj":     ["1","V","N","VxN","H"],
    "wo_proj":    ["1","V","N","VxN","H"],
    "ffn_up":     ["1","V","N","VxN","H"],
    "ffn_gate":   ["1","V","N","VxN","H"],
    "ffn_down":   ["1","V","N","VxN","H"],

    "rmsnorm":    ["1","V","H"],
    "rope":       ["1","V","H"],
    "silu":       ["1","V","H"],
    "gelu":       ["1","V","H"],
    "residual":   ["1","V","H"],
}

# ------------------------------ 读取 CSV ------------------------------
def _read_rows(csv_path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        headers = rd.fieldnames or []
    return rows, headers

def _get(row: Dict[str, Any], key: str, default: float=0.0) -> float:
    v = row.get(key, "")
    try: return float(v) if str(v).strip() != "" else default
    except: return default

# ------------------------------ 组装 X/y ------------------------------
def _feval(name: str, row: Dict[str, Any]) -> float:
    L = _get(row, "seqlen")
    H = _get(row, "n_heads")
    V = _get(row, "vector_dim")
    N = _get(row, "matrix_col")
    if   name == "1":   return 1.0
    elif name == "L":   return L
    elif name == "L2":  return L*L
    elif name == "H":   return H
    elif name == "LxH": return L*H
    elif name == "V":   return V
    elif name == "N":   return N
    elif name == "VxN": return V*N
    return 0.0

def _build_X_y_for_op(rows: List[Dict[str, Any]], opk: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    basis = FORMULA_SPEC[opk]
    xs, ys = [], []
    for r in rows:
        if (r.get("op") or "").strip() != opk:  # JSON 中我们直接把逻辑 op 写到 op 字段
            continue
        cyc = r.get("cycles", "")
        if str(cyc).strip() == "":  # 跳过无结果
            continue
        vec = [_feval(b, r) for b in basis]
        xs.append(vec)
        ys.append(float(cyc))
    X = np.array(xs, dtype=float) if xs else np.zeros((0, len(basis)))
    y = np.array(ys, dtype=float) if ys else np.zeros((0,))
    return X, y, basis

def _fit_ls(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    if X.shape[0] == 0:
        return np.zeros((X.shape[1],)), {"rmse": float("nan"), "r2": float("nan")}
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ w
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - ss_res/ss_tot) if ss_tot != 0 else float("nan")
    return w.flatten(), {"rmse": rmse, "r2": r2}

# ------------------------------ 拟合输出 ------------------------------
def fit_model_formula(results_csv: Path, out_model: Path, out_summary_csv: Optional[Path]) -> None:
    rows, _ = _read_rows(results_csv)
    models: Dict[str, Any] = {}
    summary_rows = []

    for opk in OP_KEYS:
        X, y, basis = _build_X_y_for_op(rows, opk)
        if X.shape[0] == 0:
            continue
        w, metrics = _fit_ls(X, y)
        expr = " + ".join([f"{coef:.6g}*{name}" if name!="1" else f"{coef:.6g}" for coef, name in zip(w, basis)])
        models[opk] = {"basis": basis, "coeffs": [float(x) for x in w],
                       "expr": f"cycles ≈ {expr}",
                       "num_samples": int(X.shape[0]), "metrics": metrics}
        summary_rows.append({"op_label": opk, "expr": models[opk]["expr"],
                             "num_samples": int(X.shape[0]),
                             "rmse": f"{metrics['rmse']:.6g}",
                             "r2": f"{metrics['r2']:.6g}"})

    if not models:
        print("没有可用数据行进行拟合。", file=sys.stderr); sys.exit(1)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_model.write_text(json.dumps({"per_op_formula": models}, indent=2), encoding="utf-8")
    print(f"[ok] wrote per-op **formula** model JSON -> {out_model}")

    if out_summary_csv:
        with out_summary_csv.open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=["op_label","expr","num_samples","rmse","r2"])
            wcsv.writeheader(); wcsv.writerows(summary_rows)
        print(f"[ok] wrote formula fit summary -> {out_summary_csv}")

# ------------------------------ 预测接口（用于插值） ------------------------------
def predict_with_formula(model_json: Path,
                         op: str,
                         vector_dim: Optional[int],
                         matrix_col: Optional[int],
                         seqlen: Optional[int],
                         n_heads: Optional[int]) -> float:
    obj = json.loads(Path(model_json).read_text(encoding="utf-8"))
    model = obj["per_op_formula"].get(op)
    if model is None:
        raise RuntimeError(f"模型中缺少算子 {op} 的公式。")
    basis = model["basis"]; coeffs = model["coeffs"]
    row = {"seqlen": seqlen or 0, "vector_dim": vector_dim or 0,
           "matrix_col": matrix_col or 0, "n_heads": n_heads or 0}
    def feval(name: str) -> float:
        if   name=="1": return 1.0
        elif name=="L": return float(row["seqlen"])
        elif name=="L2":return float(row["seqlen"])**2
        elif name=="H": return float(row["n_heads"])
        elif name=="LxH":return float(row["seqlen"])*float(row["n_heads"])
        elif name=="V": return float(row["vector_dim"])
        elif name=="N": return float(row["matrix_col"])
        elif name=="VxN":return float(row["vector_dim"])*float(row["matrix_col"])
        return 0.0
    feats = np.array([feval(n) for n in basis], dtype=float)
    w = np.array(coeffs, dtype=float)
    return float(feats.dot(w))

# ------------------------------ CLI ------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fit = sub.add_parser("fit")
    ap_fit.add_argument("--results-csv", type=Path, required=True)
    ap_fit.add_argument("--out-model", type=Path, required=True)
    ap_fit.add_argument("--out-summary-csv", type=Path, default=None)

    ap_pred = sub.add_parser("predict")
    ap_pred.add_argument("--model-json", type=Path, required=True)
    ap_pred.add_argument("--op", type=str, required=True, choices=list(OP_KEYS))
    ap_pred.add_argument("--vector-dim", type=int, default=None)
    ap_pred.add_argument("--matrix-col", type=int, default=None)
    ap_pred.add_argument("--seqlen", type=int, default=None)
    ap_pred.add_argument("--n-heads", type=int, default=None)

    args = ap.parse_args()
    if args.cmd == "fit":
        fit_model_formula(args.results_csv, args.out_model, args.out_summary_csv)
    else:
        val = predict_with_formula(args.model_json, args.op, args.vector_dim, args.matrix_col, args.seqlen, args.n_heads)
        print(f"{val:.3f}")

if __name__ == "__main__":
    main()
