import os, re, glob, json, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

TARGET_FILE = "softmax_kernel.h"
TARGET_LINES = (98, 103)

def parse_mk_from_path(path: str) -> Tuple[int,int]:
    m = re.search(r"(\d+)\s*[xX\*]\s*(\d+)", path)
    if not m: return (-1, -1)
    return tuple(map(int, m.groups()))

def read_code_exe(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    if "code" not in df.columns or "running_time(us)" not in df.columns:
        raise ValueError(f"{p}: missing required columns 'code' and 'running_time(us)'")
    m = df["code"].astype(str).str.extract(r'(?P<file>[^:]+):(?P<line>\d+)$')
    df["file"] = m["file"]
    df["line"] = m["line"].astype("Int64")
    return df

def sum_softmax_time_us(df: pd.DataFrame) -> float:
    mask = df["file"].astype(str).str.endswith(TARGET_FILE, na=False) & df["line"].isin(TARGET_LINES)
    return float(df.loc[mask, "running_time(us)"].sum())

def summarize_one(p: str) -> dict:
    df = read_code_exe(p)
    M, K = parse_mk_from_path(p)
    meas_us = sum_softmax_time_us(df)
    soft_file_total_us = float(df[df["file"].astype(str).str.endswith(TARGET_FILE, na=False)]["running_time(us)"].sum())
    grand_total_us = float(df["running_time(us)"].sum())
    present_lines = sorted(df[df["file"].astype(str).str.endswith(TARGET_FILE, na=False)]["line"].dropna().unique().tolist())
    found_lines = sorted(df[(df["file"].astype(str).str.endswith(TARGET_FILE, na=False)) & (df["line"].isin(TARGET_LINES))]["line"].unique().tolist())
    note = ""
    if not found_lines:
        note = f"WARNING: none of {TARGET_LINES} appeared; present lines head={present_lines[:10]}"
    elif set(found_lines) != set(TARGET_LINES):
        note = f"INFO: only found lines {found_lines}, target={TARGET_LINES}"
    return dict(path=p, M=M, K=K, meas_us=meas_us,
                softmax_file_total_us=soft_file_total_us,
                grand_total_us=grand_total_us,
                ratio_target_over_file=(meas_us/soft_file_total_us if soft_file_total_us>0 else math.nan),
                ratio_target_over_total=(meas_us/grand_total_us if grand_total_us>0 else math.nan),
                found_lines=found_lines, present_lines=present_lines, note=note)

def build_design(df_cases: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "MK": df_cases["M"]*df_cases["K"],
        "M":  df_cases["M"],
        "bias": 1.0
    })

def fit_linear(X: pd.DataFrame, y: pd.Series):
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX.values, Xty.values)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX.values) @ Xty.values
    yhat = X.values @ beta
    resid = y.values - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y.values - y.values.mean())**2)) if len(y)>1 else 0.0
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else float("nan")
    mape = float(np.mean(np.abs(resid / np.maximum(1e-9, y.values))) * 100.0)
    return beta, yhat, r2, mape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", default="./results/core0.veccore0_code_exe_*.csv",
                    help="Glob pattern of msprof code_exe CSV files (default: ./results/core0.veccore0_code_exe_*.csv)")
    ap.add_argument("--out_dir", default="./softmax_fit_out")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.csv_glob))
    if not paths:
        raise SystemExit(f"No CSV files match: {args.csv_glob}")

    rows = [summarize_one(p) for p in paths]
    data = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "softmax_latency_summary.csv")
    data.to_csv(summary_csv, index=False)

    bad = data[(data["ratio_target_over_file"]>1.0) | (data["ratio_target_over_file"]<0)]
    if len(bad):
        print("[WARN] target/file ratio out of [0,1] range for some rows:")
        print(bad[["path","ratio_target_over_file"]].to_string(index=False))

    fit_df = data[(data["M"]>0) & (data["K"]>0)].copy()
    model_json = os.path.join(args.out_dir, "softmax_latency_model.json")
    if len(fit_df) >= 3:
        X = build_design(fit_df)
        y = fit_df["meas_us"]
        beta, yhat, r2, mape = fit_linear(X, y)
        coefs = dict(a_MK=float(beta[0]), b_M=float(beta[1]), c_bias=float(beta[2]))
        json.dump(dict(model="T_us = a*MK + b*M + c",
                       target_file=TARGET_FILE, target_lines=TARGET_LINES,
                       coefficients=coefs, metrics=dict(r2=r2, mape=mape),
                       samples=len(fit_df)),
                  open(model_json, "w"), indent=2)
        plt.figure()
        plt.scatter(y, yhat, s=16)
        lo, hi = float(y.min()), float(y.max())
        plt.plot([lo, hi], [lo, hi], linewidth=1)
        plt.xlabel("Measured (us)")
        plt.ylabel("Predicted (us)")
        plt.title(f"Softmax latency model: R2={r2:.6f}, MAPE={mape:.3f}%")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "softmax_latency_fit_plot.png"), dpi=160)
        print("[OK] Fitted model: T_us = a*MK + b*M + c")
        print("     a_MK={:.6e}, b_M={:.6e}, c={:.6e}".format(coefs["a_MK"], coefs["b_M"], coefs["c_bias"]))
        print("     R2={:.6f}, MAPE={:.3f}%".format(r2, mape))
    else:
        json.dump(dict(model="T_us = a*MK + b*M + c",
                       target_file=TARGET_FILE, target_lines=TARGET_LINES,
                       error="Not enough samples to fit; need >=3 with distinct (M,K).",
                       samples=len(fit_df)),
                  open(model_json, "w"), indent=2)
        print("[WARN] Not enough samples to fit model (need >=3 distinct shapes); wrote JSON with explanation.")

    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Model JSON : {model_json}")

if __name__ == "__main__":
    main()
