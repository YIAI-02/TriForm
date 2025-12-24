#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, glob, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_len_from_filename(p: str):
    m = re.search(r'len(\d+)\.csv$', p)
    return int(m.group(1)) if m else None

def read_code_exec_csv(p: str, time_col: str = "running_time(us)") -> pd.DataFrame:
    df = pd.read_csv(p)
    need = {"code","call_count",time_col}
    if not need.issubset(df.columns):
        raise ValueError(f"{p}: need column {need}")
    m = df["code"].astype(str).str.extract(r'(?P<file>[^:]+):(?P<line>\d+)$')
    df["file"] = m["file"]
    df["line"] = pd.to_numeric(m["line"], errors="coerce")
    df["agg_time_us"] = df[time_col] * df["call_count"]
    return df

def build_long_table(csv_list):
    rows = []
    for p in csv_list:
        L = parse_len_from_filename(p)
        if L is None: 
            continue
        df = read_code_exec_csv(p)
        for _, r in df.iterrows():
            rows.append({
                "src_csv": p, "length": L,
                "code": r["code"], "file": r["file"], "line": r["line"],
                "call_count": r["call_count"], "running_time_us": r["running_time(us)"],
                "agg_time_us": r["agg_time_us"],
            })
    return pd.DataFrame(rows)

def corr_and_slope(subdf, x_col):
    x = subdf[x_col].values.astype(float)
    y = subdf["agg_time_us"].values.astype(float)
    if len(x) < 3 or np.std(x)==0 or np.std(y)==0:
        return np.nan, np.nan, np.nan
    corr = np.corrcoef(x, y)[0,1]
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]  # y = a*x + b
    return float(corr), float(a), float(b)

def fit_linear(x, y):
    X = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = a*x + b
    sse = np.sum((y - y_pred)**2); sst = np.sum((y - np.mean(y))**2)
    r2 = 1 - sse/sst if sst>0 else np.nan
    mae = np.mean(np.abs(y - y_pred))
    return float(a), float(b), float(r2), float(mae), y_pred

def auto_pick_bytes_per_elem(long_df, file_filter=None, corr_thresh=0.9):
    best = None
    for bpe in (2,4):
        df = long_df.copy()
        df["bytes_in"] = df["length"] * bpe
        df["bytes_out"]= df["length"] * bpe
        # 仅考虑目标文件（如 faster_gelu_custom.h）
        sub = df[df["file"].str.contains(file_filter)] if file_filter else df
        group = []
        for code, s in sub.groupby("code"):
            ci, ai, bi = corr_and_slope(s, "bytes_in")
            co, ao, bo = corr_and_slope(s, "bytes_out")
            group.append((code, ci, co))
        gdf = pd.DataFrame(group, columns=["code","corr_in","corr_out"])
        score = gdf["corr_in"].clip(lower=0).sum() + gdf["corr_out"].clip(lower=0).sum()
        if (best is None) or (score > best[0]):
            best = (score, bpe)
    return best[1] if best else 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/mnt/data/core0.veccore0_code_exe_len*.csv",
                    help="通配符读取 veccore0 CSVs")
    ap.add_argument("--out_dir", default="/mnt/data/faster_gelu_copy_fit_outputs",
                    help="输出目录")
    ap.add_argument("--bytes_per_elem", default="auto", choices=["auto","2","4"],
                    help="元素字节数（auto/2/4），auto 会在 2 与 4 中择优")
    ap.add_argument("--file_filter", default="fasterg elu|faster_gelu|fastergelu_custom|faster_gelu_custom".replace(" ",""),
                    help="正则过滤文件路径，缩小到 FasterGELU 内核所在头文件/源文件")
    ap.add_argument("--corr_thresh", type=float, default=0.90, help="相关系数阈值")
    ap.add_argument("--min_lengths", type=int, default=3, help="至少出现多少种 length")
    ap.add_argument("--assume_symmetry", action="store_true",
                    help="将总拷贝时间视作 CopyIn 与 CopyOut 对称分摊（默认不对称）")
    args = ap.parse_args()

    csv_list = sorted(glob.glob(args.glob))
    if not csv_list:
        raise SystemExit(f"找不到CSV: {args.glob}")
    long_df = build_long_table(csv_list)

    # bytes/elem
    if args.bytes_per_elem == "auto":
        bpe = auto_pick_bytes_per_elem(long_df, args.file_filter, args.corr_thresh)
    else:
        bpe = int(args.bytes_per_elem)

    df = long_df.copy()
    df["bytes_in"]  = df["length"] * bpe
    df["bytes_out"] = df["length"] * bpe

    # 只在目标文件中挑选随字节线性增长的候选行
    sub = df[df["file"].str.contains(args.file_filter, regex=True, na=False)].copy()
    grouped = []
    for code, s in sub.groupby("code"):
        ci, ai, bi = corr_and_slope(s, "bytes_in")
        co, ao, bo = corr_and_slope(s, "bytes_out")
        grouped.append({
            "code": code, "file": s["file"].iloc[0], "line": s["line"].iloc[0],
            "n_lengths": s["src_csv"].nunique(),
            "corr_vs_bytes_in": ci, "slope_in": ai, "intercept_in": bi,
            "corr_vs_bytes_out": co, "slope_out": ao, "intercept_out": bo,
            "total_time_us": float(s["agg_time_us"].sum())
        })
    corr_df = pd.DataFrame(grouped).sort_values("total_time_us", ascending=False)

    cand = corr_df[(corr_df["n_lengths"]>=args.min_lengths)&
                   (corr_df[["corr_vs_bytes_in","corr_vs_bytes_out"]].max(axis=1)>args.corr_thresh)].copy()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out_dir/"copy_candidate_lines.csv", index=False)

    # 对于 FasterGELU，bytes_in == bytes_out，因此难以仅凭相关性区分 in/out：
    # 1) assume_symmetry：把候选行的总时间视作总拷贝，再均分到 in/out；
    # 2) 否则：启发式分配（仅供参考）。
    copy_codes = set(cand["code"])

    per = []
    for src_csv, s in sub.groupby("src_csv"):
        L = int(s["length"].iloc[0])
        bin_ = bout_ = L * bpe

        # 总拷贝候选时间
        t_total = float(s[s["code"].isin(copy_codes)]["agg_time_us"].sum())

        if args.assume_symmetry:
            t_in = t_total / 2.0
            t_out = t_total / 2.0
        else:
            # 简单启发：把 slope_in >= slope_out 的行计入 in，其余计入 out。
            lines_in = set(cand[cand["slope_in"] >= cand["slope_out"]]["code"])
            lines_out= copy_codes - lines_in
            t_in = float(s[s["code"].isin(lines_in)]["agg_time_us"].sum())
            t_out= float(s[s["code"].isin(lines_out)]["agg_time_us"].sum())

        per.append({"src_csv":src_csv, "length":L, "bytes_in":bin_, "bytes_out":bout_,
                    "copy_in_time_us":t_in, "copy_out_time_us":t_out})

    per_df = pd.DataFrame(per).sort_values("length")
    per_df.to_csv(out_dir/"per_length_summary.csv", index=False)

    # 线性拟合
    dfi = per_df[per_df["copy_in_time_us"]>0]
    dfo = per_df[per_df["copy_out_time_us"]>0]
    if dfi.empty or dfo.empty:
        # 构造对称的 per_df：把总拷贝时间一分为二
        sym = per_df.copy()
        total = (sym["copy_in_time_us"].fillna(0) + sym["copy_out_time_us"].fillna(0)).values
        sym["copy_in_time_us"] = total / 2.0
        sym["copy_out_time_us"] = total / 2.0
        dfi = sym.copy(); dfo = sym.copy()

    ai, bi, r2i, maei, _ = fit_linear(dfi["bytes_in"].values.astype(float), dfi["copy_in_time_us"].values.astype(float))
    ao, bo, r2o, maeo, _ = fit_linear(dfo["bytes_out"].values.astype(float), dfo["copy_out_time_us"].values.astype(float))

    # 输出与可视化
    summary = pd.DataFrame({
        "direction":["copy_in","copy_out"],
        "bytes_per_elem":[bpe,bpe],
        "alpha_us_per_byte":[ai,ao],
        "beta_us":[bi,bo],
        "R2":[r2i,r2o],
        "MAE_us":[maei,maeo],
        "effective_bandwidth_GBps":[ 1.0/(ai*1e-6)/1e9 if ai>0 else np.nan,
                                     1.0/(ao*1e-6)/1e9 if ao>0 else np.nan],
        "assume_symmetry":[args.assume_symmetry, args.assume_symmetry]
    })
    summary.to_csv(out_dir/"faster_gelu_copy_fit_summary.csv", index=False)

    model = {
        "bytes_per_elem": int(bpe),
        "assume_symmetry": bool(args.assume_symmetry),
        "copy_in": {"alpha_us_per_byte": float(ai), "beta_us": float(bi), "R2": float(r2i)},
        "copy_out":{"alpha_us_per_byte": float(ao), "beta_us": float(bo), "R2": float(r2o)},
        "notes": "Fitted from veccore0 CSVs with candidate copy lines filtered by file name and correlation."
    }
    with open(out_dir/"faster_gelu_copy_fit_model.json","w") as f:
        json.dump(model, f, indent=2)

    # 画图
    plt.figure()
    plt.scatter(dfi["bytes_in"], dfi["copy_in_time_us"], label="CopyIn")
    xg = np.linspace(dfi["bytes_in"].min(), dfi["bytes_in"].max(), 256)
    plt.plot(xg, ai*xg + bi, label="CopyIn Fit")
    plt.xlabel("Bytes"); plt.ylabel("Time (us)"); plt.title("FasterGELU CopyIn: time vs bytes"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir/"copyin_fit.png"); plt.close()

    plt.figure()
    plt.scatter(dfo["bytes_out"], dfo["copy_out_time_us"], label="CopyOut")
    xg = np.linspace(dfo["bytes_out"].min(), dfo["bytes_out"].max(), 256)
    plt.plot(xg, ao*xg + bo, label="CopyOut Fit")
    plt.xlabel("Bytes"); plt.ylabel("Time (us)"); plt.title("FasterGELU CopyOut: time vs bytes"); plt.legend()
    plt.tight_layout(); plt.savefig(out_dir/"copyout_fit.png"); plt.close()

    print("[OK] FasterGELU Copy 模型: time_us = alpha * bytes + beta")
    print(f"     bytes/elem = {bpe} (auto 可由 2/4 之间选择)")
    print(f"     CopyIn : alpha = {ai:.6e} us/B, beta = {bi:.3f} us, R2 = {r2i:.4f}")
    print(f"     CopyOut: alpha = {ao:.6e} us/B, beta = {bo:.3f} us, R2 = {r2o:.4f}")
    print(f"[OK] 输出目录: {out_dir}")
    print("     - copy_candidate_lines.csv, per_length_summary.csv")
    print("     - faster_gelu_copy_fit_summary.csv, faster_gelu_copy_fit_model.json")
    print("     - copyin_fit.png, copyout_fit.png")

if __name__ == "__main__":
    main()
