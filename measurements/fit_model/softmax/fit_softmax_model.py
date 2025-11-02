import os, re, glob, json, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

# ---------------- fixed targets & knobs ----------------
TARGET_FILE  = "softmax_kernel.h"
TARGET_LINES = (98, 103)
CORES   = 24      # for Ascend 910B1
K_ALIGN = 128     # coarse alignment feature; adjust if you know the kernel's true tile-K

def parse_mk_from_path(path: str) -> Tuple[int,int]:
    # Parse 'M x K' from any component of the path; delimiters: x/X/*
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

def summarize_one(csv_path: str) -> Dict:
    df = read_code_exe(csv_path)
    M, K = parse_mk_from_path(csv_path)
    meas_us = sum_softmax_time_us(df)
    soft_file_total_us = float(df[df["file"].astype(str).str.endswith(TARGET_FILE, na=False)]["running_time(us)"].sum())
    grand_total_us     = float(df["running_time(us)"].sum())
    present_lines = sorted(df[df["file"].astype(str).str.endswith(TARGET_FILE, na=False)]["line"].dropna().unique().tolist())
    found_lines = sorted(df[(df["file"].astype(str).str.endswith(TARGET_FILE, na=False)) & (df["line"].isin(TARGET_LINES))]["line"].unique().tolist())
    note = ""
    if not found_lines:
        note = f"WARNING: none of {TARGET_LINES} appeared; present head={present_lines[:10]}"
    elif set(found_lines) != set(TARGET_LINES):
        note = f"INFO: only found lines {found_lines}, target={TARGET_LINES}"

    # light features
    if M > 0:
        core_row = (M + CORES - 1) // CORES        # ceil(M / CORES)
        blocks   = (M + core_row - 1) // core_row  # ceil(M / core_row)
    else:
        core_row = 0; blocks = 0
    k_tail = 1 if (K > 0 and (K % K_ALIGN) != 0) else 0

    return dict(path=csv_path, M=M, K=K, meas_us=meas_us,
                softmax_file_total_us=soft_file_total_us,
                grand_total_us=grand_total_us,
                ratio_target_over_file=(meas_us/soft_file_total_us if soft_file_total_us>0 else math.nan),
                ratio_target_over_total=(meas_us/grand_total_us if grand_total_us>0 else math.nan),
                found_lines=found_lines, present_lines=present_lines, note=note,
                blocks=blocks, k_tail=k_tail)

def build_X(df_cases: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "MK":   df_cases["M"]*df_cases["K"],
        "M":    df_cases["M"],
        "blk":  df_cases["blocks"],
        "ktl":  df_cases["k_tail"],
        "bias": 1.0
    })

def fit_wls_relative(X: pd.DataFrame, y: pd.Series):
    # Weighted least squares with weights ~ 1 / y^2 to target relative error (MAPE-like).
    yv = y.values.astype(float)
    Xv = X.values.astype(float)
    w  = 1.0 / np.maximum(1e-9, yv)**2              # emphasize relative error
    WX = Xv * w[:, None]
    Wy = yv * w
    XtWX = Xv.T @ WX
    XtWy = Xv.T @ Wy
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtWX) @ XtWy
    yhat = Xv @ beta

    # metrics
    resid = yv - yhat
    ape   = np.abs(resid / np.maximum(1e-9, yv)) * 100.0
    mape  = float(np.mean(ape))
    p50   = float(np.percentile(ape, 50))
    p90   = float(np.percentile(ape, 90))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((yv - yv.mean())**2)) if len(yv)>1 else 0.0
    r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else float("nan")
    return beta, yhat, dict(MAPE=mape, P50=p50, P90=p90, R2=r2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", default="./results/core0.veccore0_code_exe_*.csv",
                    help="Glob of msprof code_exe CSVs (default: ./results/core0.veccore0_code_exe_*.csv)")
    ap.add_argument("--out_dir",  default="./softmax_fit_out")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.csv_glob))
    if not paths:
        raise SystemExit(f"No CSV files match: {args.csv_glob}")

    rows = [summarize_one(p) for p in paths]
    data = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "softmax_latency_summary.csv")
    data.to_csv(summary_csv, index=False)

    # sanity prints
    bad = data[(data["ratio_target_over_file"]>1.0) | (data["ratio_target_over_file"]<0)]
    if len(bad):
        print("[WARN] target/file ratio out of [0,1] for rows:")
        print(bad[["path","ratio_target_over_file"]].to_string(index=False))

    # fit (need >=3 with valid shapes)
    fit_df = data[(data["M"]>0)&(data["K"]>0)].copy()
    if len(fit_df) < 3:
        model_json = os.path.join(args.out_dir, "softmax_latency_model.json")
        json.dump(dict(
            model="T_us = a*MK + b*M + d*blocks + e*k_tail + c",
            error="Not enough samples to fit; need >=3 with distinct (M,K).",
            samples=len(fit_df),
            targets=dict(file=TARGET_FILE, lines=TARGET_LINES),
        ), open(model_json, "w"), indent=2)
        print("[WARN] Not enough samples to fit. Wrote JSON with explanation.")
        print(f"[OK] Summary CSV: {summary_csv}")
        print(f"[OK] Model JSON : {model_json}")
        return

    X = build_X(fit_df)
    y = fit_df["meas_us"]
    beta, yhat, metrics = fit_wls_relative(X, y)

    coefs = dict(a_MK=float(beta[0]), b_M=float(beta[1]), d_blk=float(beta[2]),
                 e_ktl=float(beta[3]), c_bias=float(beta[4]))

    # save model
    model_json = os.path.join(args.out_dir, "softmax_latency_model.json")
    json.dump(dict(
        model="T_us = a*MK + b*M + d*blocks + e*k_tail + c",
        coefficients=coefs, metrics=metrics, samples=len(fit_df),
        targets=dict(file=TARGET_FILE, lines=TARGET_LINES),
        knobs=dict(CORES=CORES, K_ALIGN=K_ALIGN),
    ), open(model_json, "w"), indent=2)

    # parity plot
    plt.figure()
    plt.scatter(y, yhat, s=16)
    lo, hi = float(y.min()), float(y.max())
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("Measured (us)")
    plt.ylabel("Predicted (us)")
    plt.title(f"Softmax latency model (WLS): R2={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "softmax_latency_fit_plot.png"), dpi=160)

    print("[OK] Fitted model (WLS, relative):")
    print("     a_MK={a:.6e}, b_M={b:.6e}, d_blk={d:.6e}, e_ktl={e:.6e}, c={c:.6e}".format(
        a=coefs["a_MK"], b=coefs["b_M"], d=coefs["d_blk"], e=coefs["e_ktl"], c=coefs["c_bias"]))
    print("     Metrics:", metrics)
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Model JSON : {model_json}")

if __name__ == "__main__":
    main()