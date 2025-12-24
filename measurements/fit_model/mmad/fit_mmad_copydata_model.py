
import os
import re
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_shape_from_filename(p: str):
    m = re.search(r'_(\d+)x(\d+)x(\d+)\.csv$', p)
    if not m:
        return None
    return tuple(map(int, m.groups()))

def bytes_in_out(m, k, n, bytes_half=2, bytes_float=4):
    # CopyIn: A(m*k, half) + B(k*n, half) + bias(n, half)
    # CopyOut: C(m*n, float)
    bytes_in = bytes_half*(m*k + k*n + n)
    bytes_out = bytes_float*(m*n)
    return bytes_in, bytes_out

def read_code_exec_csv(p: str, time_col: str = "running_time(us)") -> pd.DataFrame:
    df = pd.read_csv(p)
    if "code" not in df.columns or time_col not in df.columns or "call_count" not in df.columns:
        raise ValueError(f"{p}: need column contains code, call_count, {time_col}")
    m = df["code"].astype(str).str.extract(r'(?P<file>[^:]+):(?P<line>\\d+)$')
    df["file"] = m["file"]
    df["line"] = pd.to_numeric(m["line"], errors="coerce")
    df["agg_time_us"] = df[time_col] * df["call_count"]
    return df

def corr_and_slope(subdf: pd.DataFrame, x_col: str):
    x = subdf[x_col].values.astype(float)
    y = subdf["agg_time_us"].values.astype(float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan
    corr = np.corrcoef(x, y)[0,1]
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]  # y = a*x + b
    return corr, a, b

def fit_linear(x: np.ndarray, y: np.ndarray):
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]  # y = a*x + b
    y_pred = a*x + b
    sse = float(np.sum((y - y_pred)**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = float(1 - sse/sst) if sst > 0 else float("nan")
    mae = float(np.mean(np.abs(y - y_pred)))
    return a, b, r2, mae, y_pred

def auto_discover_copy_lines(long_df: pd.DataFrame, corr_thresh_in=0.9, corr_thresh_out=0.9, min_shapes=3):
    grouped = []
    for code, sub in long_df.groupby("code"):
        corr_in, a_in, b_in = corr_and_slope(sub, "bytes_in")
        corr_out, a_out, b_out = corr_and_slope(sub, "bytes_out")
        grouped.append({
            "code": code,
            "file": sub["file"].iloc[0],
            "line": sub["line"].iloc[0],
            "n_shapes": sub["src_csv"].nunique(),
            "corr_vs_bytes_in": corr_in,
            "slope_in": a_in, "intercept_in": b_in,
            "corr_vs_bytes_out": corr_out,
            "slope_out": a_out, "intercept_out": b_out,
            "total_time_us": sub["agg_time_us"].sum(),
        })
    corr_df = pd.DataFrame(grouped)

    copy_in = corr_df[(corr_df["n_shapes"] >= min_shapes) &
                      (corr_df["corr_vs_bytes_in"] > corr_thresh_in) &
                      (corr_df["corr_vs_bytes_in"] >= corr_df["corr_vs_bytes_out"])].copy()
    copy_out = corr_df[(corr_df["n_shapes"] >= min_shapes) &
                       (corr_df["corr_vs_bytes_out"] > corr_thresh_out) &
                       (corr_df["corr_vs_bytes_out"] > corr_df["corr_vs_bytes_in"])].copy()
    return copy_in.sort_values("total_time_us", ascending=False), copy_out.sort_values("total_time_us", ascending=False)

def build_long_table(csv_list):
    rows = []
    for p in csv_list:
        shape = parse_shape_from_filename(p)
        if shape is None:
            continue
        m,k,n = shape
        bin_, bout_ = bytes_in_out(m,k,n)
        df = read_code_exec_csv(p)
        for _, r in df.iterrows():
            rows.append({
                "src_csv": p,
                "m": m, "k": k, "n": n,
                "bytes_in": bin_,
                "bytes_out": bout_,
                "code": r["code"],
                "file": r["file"],
                "line": r["line"],
                "call_count": r["call_count"],
                "running_time_us": r["running_time(us)"],
                "agg_time_us": r["agg_time_us"],
            })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/core0.cubecore0_code_exe_*.csv")
    ap.add_argument("--out_dir", default="./copydata_fit_outputs")
    ap.add_argument("--corr_in", type=float, default=0.9)
    ap.add_argument("--corr_out", type=float, default=0.9)
    ap.add_argument("--min_shapes", type=int, default=3)
    args = ap.parse_args()

    csv_list = sorted(glob.glob(args.glob))
    if not csv_list:
        raise SystemExit(f"Cannot find CSV files: {args.glob}")

    long_df = build_long_table(csv_list)
    copy_in_lines, copy_out_lines = auto_discover_copy_lines(long_df, args.corr_in, args.corr_out, args.min_shapes)

    # Summarize CopyIn/CopyOut time for each shape
    copy_in_codes = set(copy_in_lines["code"].tolist())
    copy_out_codes = set(copy_out_lines["code"].tolist())

    per_shape = []
    for src_csv, sub in long_df.groupby("src_csv"):
        m = int(sub["m"].iloc[0]); k = int(sub["k"].iloc[0]); n = int(sub["n"].iloc[0])
        bin_, bout_ = int(sub["bytes_in"].iloc[0]), int(sub["bytes_out"].iloc[0])
        t_in = float(sub[sub["code"].isin(copy_in_codes)]["agg_time_us"].sum())
        t_out = float(sub[sub["code"].isin(copy_out_codes)]["agg_time_us"].sum())
        per_shape.append({
            "src_csv": src_csv,
            "m": m, "k": k, "n": n,
            "bytes_in": bin_, "bytes_out": bout_,
            "copy_in_time_us": t_in, "copy_out_time_us": t_out,
        })
    per_df = pd.DataFrame(per_shape)

    # 只用非零数据拟合
    df_in = per_df[per_df["copy_in_time_us"] > 0].copy()
    df_out = per_df[per_df["copy_out_time_us"] > 0].copy()

    a_in, b_in, r2_in, mae_in, yhat_in = fit_linear(df_in["bytes_in"].values.astype(float), df_in["copy_in_time_us"].values.astype(float))
    a_out, b_out, r2_out, mae_out, yhat_out = fit_linear(df_out["bytes_out"].values.astype(float), df_out["copy_out_time_us"].values.astype(float))

    # 输出与可视化
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 保存候选行
    copy_in_lines.to_csv(out_dir / "copyin_candidate_lines.csv", index=False)
    copy_out_lines.to_csv(out_dir / "copyout_candidate_lines.csv", index=False)

    # 保存拟合摘要
    summary = pd.DataFrame({
        "direction": ["copy_in", "copy_out"],
        "alpha_us_per_byte": [a_in, a_out],
        "beta_us": [b_in, b_out],
        "R2": [r2_in, r2_out],
        "MAE_us": [mae_in, mae_out],
        "effective_bandwidth_GBps": [1.0/(a_in*1e-6)/1e9, 1.0/(a_out*1e-6)/1e9],
    })
    summary.to_csv(out_dir / "copy_fit_summary.csv", index=False)

    # 保存模型 JSON
    model = {
        "copy_in": {"alpha_us_per_byte": float(a_in), "beta_us": float(b_in), "R2": float(r2_in)},
        "copy_out": {"alpha_us_per_byte": float(a_out), "beta_us": float(b_out), "R2": float(r2_out)},
        "notes": "Fitted with y = alpha*bytes + beta using auto-identified lines by correlation thresholds."
    }
    with open(out_dir / "copy_fit_model.json", "w") as f:
        json.dump(model, f, indent=2)

    # 画图（每张图一个figure，不设置颜色）
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(df_in["bytes_in"], df_in["copy_in_time_us"], label="Measured")
    xgrid = np.linspace(df_in["bytes_in"].min(), df_in["bytes_in"].max(), 256)
    plt.plot(xgrid, a_in*xgrid + b_in, label="Fit")
    plt.xlabel("Bytes In"); plt.ylabel("CopyIn Time (us)"); plt.title("CopyIn: time vs bytes"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "copyin_fit.png"); plt.close()

    plt.figure()
    plt.scatter(df_out["bytes_out"], df_out["copy_out_time_us"], label="Measured")
    xgrid = np.linspace(df_out["bytes_out"].min(), df_out["bytes_out"].max(), 256)
    plt.plot(xgrid, a_out*xgrid + b_out, label="Fit")
    plt.xlabel("Bytes Out"); plt.ylabel("CopyOut Time (us)"); plt.title("CopyOut: time vs bytes"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir / "copyout_fit.png"); plt.close()

    print("[OK] CopyIn: time_us = alpha_in * bytes_in + beta_in")
    print(f"     alpha_in = {a_in:.6e} us/B  (effective_bw ≈ {1.0/(a_in*1e-6)/1e9:.3f} GB/s), beta_in = {b_in:.3f} us, R2={r2_in:.4f}")
    print("[OK] CopyOut: time_us = alpha_out * bytes_out + beta_out")
    print(f"     alpha_out = {a_out:.6e} us/B (effective_bw ≈ {1.0/(a_out*1e-6)/1e9:.3f} GB/s), beta_out = {b_out:.3f} us, R2={r2_out:.4f}")
    print(f"[OK] 输出目录: {out_dir}")
    print("     - copyin_candidate_lines.csv / copyout_candidate_lines.csv")
    print("     - copy_fit_summary.csv, copy_fit_model.json")
    print("     - copyin_fit.png, copyout_fit.png")

if __name__ == "__main__":
    main()
