#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_mk_from_any(s: str):
    m = re.search(r"(\d+)\s*[xX\*]\s*(\d+)", os.path.basename(s))
    if not m: 
        raise ValueError("Cannot parse MxK")
    return tuple(map(int, m.groups()))

def read_code_exe(p):
    df = pd.read_csv(p)
    m = df["code"].astype(str).str.extract(r'(?P<file>[^:]+):(?P<line>\d+)$')
    df["file"] = m["file"]
    df["line"] = m["line"].astype("Int64")
    return df

def sum_softmax_us(df, target_file, target_lines):
    mask = df["file"].astype(str).str.endswith(target_file, na=False) & df["line"].isin(target_lines)
    return float(df.loc[mask, "running_time(us)"].sum())

def summarize_one(p, target_file, target_lines, case_hint=None):
    df = read_code_exe(p)
    # shape
    M = K = -1
    for probe in [case_hint, p, os.path.dirname(p), os.getenv("SOFTMAX_CASE","")]:
        if not probe: 
            continue
        try:
            M,K = parse_mk_from_any(probe); break
        except: pass
    # measurements
    meas = sum_softmax_us(df, target_file, target_lines)
    soft_file_total = float(df[df["file"].astype(str).str.endswith(target_file, na=False)]["running_time(us)"].sum())
    grand_total = float(df["running_time(us)"].sum())
    present_lines = sorted(df[df["file"].astype(str).str.endswith(target_file, na=False)]["line"].dropna().unique().tolist())
    found_lines = sorted(df[(df["file"].astype(str).str.endswith(target_file, na=False)) & (df["line"].isin(target_lines))]["line"].unique().tolist())
    return dict(path=p, M=M, K=K, meas_us=meas, softmax_file_total_us=soft_file_total, grand_total_us=grand_total,
                found_lines=found_lines, present_lines=present_lines,
                ratio_target_over_file=(meas/soft_file_total if soft_file_total>0 else np.nan),
                ratio_target_over_total=(meas/grand_total if grand_total>0 else np.nan))

def build_X(df_cases):
    return pd.DataFrame({
        "MK": df_cases["M"]*df_cases["K"],
        "M":  df_cases["M"],
        "bias": 1.0
    })

def fit_linear(X, y):
    XtX = X.T.dot(X).values.astype(float)
    Xty = X.T.dot(y).values.astype(float)
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX).dot(Xty)
    yhat = X.values.dot(beta)
    resid = y.values - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y.values - y.values.mean())**2)) if len(y)>1 else 0.0
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else float("nan")
    mape = float(np.mean(np.abs(resid / np.maximum(1e-9, y.values))) * 100.0)
    return beta, yhat, r2, mape

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv_glob", default="./results/core0.veccore0_code_exe_*.csv")
    ap.add_argument("--case", default="", help="Fallback MxK if not parseable from path, e.g. 1024x2048")
    ap.add_argument("--lines", default="98,103", help="Comma-separated softmax_kernel.h line numbers to sum")
    ap.add_argument("--target-file", default="softmax_kernel.h")
    ap.add_argument("--out-dir", default="softmax_fit_out")
    args = ap.parse_args()

    target_lines = tuple(int(x) for x in args.lines.split(","))
    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise SystemExit(f"No files match: {args.inputs}")
    rows = [summarize_one(p, args.target_file, target_lines, case_hint=args.case) for p in paths]
    data = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    data.to_csv(os.path.join(args.out_dir, "softmax_latency_summary.csv"), index=False)

    fit_df = data[(data["M"]>0)&(data["K"]>0)].copy()
    if len(fit_df) >= 3:
        X = build_X(fit_df)
        y = fit_df["meas_us"]
        beta, yhat, r2, mape = fit_linear(X, y)
        coefs = dict(a_MK=float(beta[0]), b_M=float(beta[1]), c_bias=float(beta[2]))
        json.dump(dict(model="T_us = a*MK + b*M + c",
                       coefficients=coefs, metrics=dict(r2=r2, mape=mape)),
                  open(os.path.join(args.out_dir, "softmax_latency_model.json"), "w"), indent=2)
        # parity plot
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
        print("     a_MK={a:.6e}, b_M={b:.6e}, c={c:.6e}".format(a=coefs["a_MK"], b=coefs["b_M"], c=coefs["c_bias"]))
        print(f"     R2={r2:.6f}, MAPE={mape:.3f}%")
    else:
        json.dump(dict(model="T_us = a*MK + b*M + c",
                       error="Not enough samples to fit; need >=3 with distinct (M,K).",
                       have=len(fit_df)),
                  open(os.path.join(args.out_dir, "softmax_latency_model.json"), "w"), indent=2)
        print("[WARN] Not enough samples to fit coefficients. Collected rows:", len(fit_df))
        print("       Provide at least 3 CSVs with distinct (M,K).")

    print(f"[OK] Summary CSV: {os.path.join(args.out_dir, 'softmax_latency_summary.csv')}")
    print(f"[OK] Model JSON : {os.path.join(args.out_dir, 'softmax_latency_model.json')}")

if __name__ == "__main__":
    main()
