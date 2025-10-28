#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_fit_latency_model.py
读取 02 的 CSV 结果，按算子（op）分别做 OLS 拟合：
  cycles_total ≈ β0 + ∑ (β_i * <MICRO_OP>_calls) + ∑ (γ_i * <MICRO_OP>_opsize)
将每个算子的模型写入 JSON，并输出 summary CSV。

用法：
  python 03_fit_latency_model.py fit \
    --results-csv pim_v2/aim_results.csv \
    --out-model pim_v2/aim_models.json \
    --out-summary-csv pim_v2/aim_fit_summary.csv

也支持预测（需要提供“特征”而不是高层维度）——仅作为工具函数：
  python 03_fit_latency_model.py predict \
    --model-json pim_v2/aim_models.json \
    --op q_proj \
    --features-json features_example.json
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

MICRO_OPS = [
    "COPY_GB_BK","COPY_BK_GB","WR_GB","MAC_ABK","MAC_BK_BK","MAC_BK_GB",
    "EWMUL","EWADD","AF","RD_MAC","RD_AF","WR_BIAS","RD_SBK","WR_SBK"
]

def _safe_float(x): 
    try: return float(x)
    except: return 0.0

def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]

def build_design_matrix(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # 特征：所有 MICRO_OP 的 calls + opsize
    feats = []
    for k in MICRO_OPS:
        feats.append(f"{k}_calls")
        feats.append(f"{k}_opsize")
    X = np.zeros((len(rows), len(feats) + 1), dtype=np.float64)  # +1 for intercept
    y = np.zeros((len(rows),), dtype=np.float64)
    for i, row in enumerate(rows):
        X[i, 0] = 1.0  # intercept
        for j, name in enumerate(feats):
            X[i, j+1] = _safe_float(row.get(name, 0))
        y[i] = _safe_float(row.get("cycles_total", 0))
    return X, y, ["intercept"] + feats

def ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    # 最小二乘 OLS，带伪逆以稳定
    beta = np.linalg.pinv(X) @ y
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(resid.T @ resid)
    ss_tot = float(((y - y.mean())**2).sum()) if len(y) > 0 else 0.0
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(ss_res / max(1, len(y)-X.shape[1])))
    mae = float(np.mean(np.abs(resid)))
    return beta, {"r2": r2, "rmse": rmse, "mae": mae}

def fit_per_op(csv_path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = load_rows(csv_path)
    # 按算子聚合
    by_op: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        op = row.get("op", "unknown")
        by_op.setdefault(op, []).append(row)
    models: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    for op, subset in sorted(by_op.items()):
        if len(subset) < 3:
            # 数据太少就跳过，避免奇异
            continue
        X, y, names = build_design_matrix(subset)
        beta, metrics = ols_fit(X, y)
        coef_map = {names[i]: float(beta[i]) for i in range(len(names))}
        models[op] = {"features": names, "coef": coef_map, "metrics": metrics, "n": len(subset)}
        summary_rows.append({"op": op, "n": len(subset), **metrics})
    return models, summary_rows

def predict(model_for_op: Dict[str, Any], features: Dict[str, float]) -> float:
    names = model_for_op["features"]
    coef = model_for_op["coef"]
    # 构造向量
    x = [features.get(name, 1.0 if name == "intercept" else 0.0) for name in names]
    s = 0.0
    for name, xv in zip(names, x):
        s += coef.get(name, 0.0) * float(xv)
    return float(s)

def main():
    ap = argparse.ArgumentParser(description="Fit per-operator latency models from AiM trace CSV")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_fit = sub.add_parser("fit")
    ap_fit.add_argument("--results-csv", type=Path, required=True)
    ap_fit.add_argument("--out-model", type=Path, required=True)
    ap_fit.add_argument("--out-summary-csv", type=Path, required=True)

    ap_pred = sub.add_parser("predict")
    ap_pred.add_argument("--model-json", type=Path, required=True)
    ap_pred.add_argument("--op", type=str, required=True)
    ap_pred.add_argument("--features-json", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "fit":
        models, summary_rows = fit_per_op(args.results_csv)
        # 写 JSON
        args.out_model.parent.mkdir(parents=True, exist_ok=True)
        with args.out_model.open("w", encoding="utf-8") as f:
            json.dump(models, f, indent=2, ensure_ascii=False)
        # 写 summary CSV
        with args.out_summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["op","n","r2","rmse","mae"])
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[ok] wrote {args.out_model} and {args.out_summary_csv}")
    else:
        # 预测
        model = json.loads(args.model_json.read_text(encoding="utf-8"))
        if args.op not in model:
            raise SystemExit(f"op {args.op} not found in model")
        feats = json.loads(args.features_json.read_text(encoding="utf-8"))
        val = predict(model[args.op], feats)
        print(f"{val:.3f}")

if __name__ == "__main__":
    main()
