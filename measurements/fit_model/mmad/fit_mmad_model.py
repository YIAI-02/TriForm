#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, json, math, argparse, textwrap, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

def ceil_div(a,b): return (a + b - 1)//b

def parse_mnk_from_name(path: str) -> Tuple[int,int,int]:
    m = re.search(r"(\d+)x(\d+)x(\d+)", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse MxNxK from file name: {path}")
    return tuple(map(int, m.groups()))

def extract_compute_total_us(df: pd.DataFrame) -> Tuple[float, int]:
    d = df[df["code"].astype(str).str.contains("mmad_custom_cube_only.h")]
    if d.empty: return 0.0, 0
    d = d.copy()
    d["line"] = d["code"].str.extract(r"h:(\d+)").astype(int) #用正则表达式提取行号
    x = d[d["line"].isin([70,71,72,73])]
    return float(x["running_time(us)"].sum()), int(x["line"].nunique()) #第二个值是不同取值的个数

def build_feature_row(M,N,K,block,meas_us): #向设定的块大小取整
    MB, NB, KB = ceil_div(M,block), ceil_div(N,block), ceil_div(K,block)
    return {
        "M":M, "N":N, "K":K,
        "MB":MB, "NB":NB, "KB":KB,
        "tiles": MB*NB*KB,
        "mn": MB*NB,
        "sum_b": MB+NB+KB,
        "meas_us": float(meas_us)
    }

def fit_linear(X: np.ndarray, y: np.ndarray):
    Phi = np.hstack([np.ones((X.shape[0],1)), X])
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    yhat = Phi @ coef
    res = y - yhat
    ss_res = float((res**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else 1.0
    mae = float(np.abs(res).mean())
    mape = float((np.abs(res)/np.maximum(1e-9, y)).mean())*100.0
    return coef, yhat, {"r2":r2, "mae":mae, "mape":mape}

def loocv_linear(X: np.ndarray, y: np.ndarray):
    n = len(y)
    errs, perfs = [], []
    for i in range(n):
        tr_idx = [j for j in range(n) if j!=i]
        te_idx = [i]
        coef, yhat_tr, _ = fit_linear(X[tr_idx], y[tr_idx])
        Phi_te = np.hstack([np.ones((1,1)), X[te_idx]])
        y_pred = float((Phi_te @ coef)[0])  # 取第一个元素再转 float
        errs.append(abs(y_pred - float(y[i])))
        perfs.append(100.0*abs(y_pred - float(y[i]))/max(1e-9, float(y[i])))
    return {
        "loocv_mae": float(np.mean(errs)),
        "loocv_mape": float(np.mean(perfs)),
        "n": n
    }

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Fit MMAD latency model from Ascend CSV profilers."
    )
    ap.add_argument("--csv_glob", default="./results/core0.cubecore0_code_exe_*.csv")
    ap.add_argument("--block_size", type=int, default=16, help="Cube block size (default: 16)")
    ap.add_argument("--out_model", default="mmad_latency_model.json", help="Output JSON model path (default: mmad_latency_model.json)")
    args = ap.parse_args()

    csvs = sorted(glob.glob(args.csv_glob))
    if not csvs:
        raise FileNotFoundError(f"No CSV matched: {args.csv_glob}")

    rows = []
    dropped = []
    for p in csvs:
        df = pd.read_csv(p)
        total_us, nlines = extract_compute_total_us(df)
        M,N,K = parse_mnk_from_name(p) #在文件名中解析出M,N,K
        row = build_feature_row(M,N,K,args.block_size,total_us)
        row["file"] = os.path.basename(p)
        row["n_lines_found"] = nlines
        if nlines == 4:
            rows.append(row)
        else:
            dropped.append(row)

    data = pd.DataFrame(rows).sort_values(["M","N","K"]).reset_index(drop=True)
    if data.empty:
        raise RuntimeError("No valid samples (need all of lines 70–73).")
    X = data[["tiles","mn","sum_b"]].to_numpy(dtype=float) #特征矩阵，MB*NB*KB, MB*NB, MB+NB+KB
    y = data["meas_us"].to_numpy(dtype=float)

    coef, yhat, fit = fit_linear(X, y)
    data["pred_us"] = yhat
    data["abs_err_us"] = (data["pred_us"] - data["meas_us"]).abs()
    data["abs_pct_err"] = 100.0 * data["abs_err_us"] / np.maximum(1e-9, data["meas_us"])

    # 留一交叉验证
    loocv = loocv_linear(X, y)

    # 导出 JSON 模型
    model = {
        "version": "v1",
        "block_size": args.block_size,
        "features": ["tiles","mn","sum_b"],
        "coefficients": {
            "b0": float(coef[0]),
            "b_tiles": float(coef[1]),
            "b_mn": float(coef[2]),
            "b_sum": float(coef[3]),
        },
        "fit_metrics": {
            "r2": float(fit["r2"]),
            "mae_us": float(fit["mae"]),
            "mape_pct": float(fit["mape"]),
            "loocv_mae_us": float(loocv["loocv_mae"]),
            "loocv_mape_pct": float(loocv["loocv_mape"]),
            "n_samples": int(len(data)),
        },
        "source_files": data["file"].tolist()
    }
    out_model_dir = os.path.dirname(args.out_model)
    if out_model_dir:  # 只有当目录非空时才创建
        os.makedirs(out_model_dir, exist_ok=True)
    with open(args.out_model, "w") as f:
        json.dump(model, f, indent=2)
    print(f"[OK] Saved model to: {args.out_model}")

    out_dir = os.path.dirname(args.out_model) or "."  # 如果为空则使用当前目录
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "mmad_latency_fit_summary.csv")
    data.to_csv(summary_csv, index=False)

    # 预测对比图
    plt.figure()
    plt.scatter(data["meas_us"], data["pred_us"])
    lo, hi = float(data["meas_us"].min()), float(data["meas_us"].max())
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("Measured (us)")
    plt.ylabel("Predicted (us)")
    plt.title(f"MMAD latency model: R2={fit['r2']:.6f}, MAPE={fit['mape']:.3f}%")
    fig_path = os.path.join(out_dir, "mmad_latency_fit_plot.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)

    print(f"[OK] Wrote: {summary_csv}")
    print(f"[OK] Wrote: {fig_path}")

if __name__ == "__main__":
    main()
