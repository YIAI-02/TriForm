#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import argparse
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
DEFAULTS = {
    "GLOB_PATTERN": "./results/*.csv",
    "TARGET_FILE": "faster_gelu_custom.h",
    "TARGET_LINES": "37",
    "TIME_COL": "running_time(us)",
    "OUT_DIR": "./gelu_fit_out"
}


def _norm_lines_to_set(lines_spec) -> List[int]:
    """
    lines_spec: 可为 int / str("1,2,3") / list[int]
    返回: set[int]
    """
    if lines_spec is None:
        return set()
    if isinstance(lines_spec, int):
        return {int(lines_spec)}
    if isinstance(lines_spec, (list, tuple, set)):
        return {int(x) for x in lines_spec}
    if isinstance(lines_spec, str):
        return {int(x.strip()) for x in lines_spec.split(",") if x.strip()}
    raise ValueError(f"Unsupported TARGET_LINES type: {type(lines_spec)}")


def parse_datalength_from_path(path: str) -> int:
    # 优先匹配如 datalength=1024 / datalength_1024 / L1024 / len1024
    patts = [
        r"[^\d]datalength[=_\- ]*(\d+)",
        r"[^\d]len[=_\- ]*(\d+)",
        r"[^\d]L(\d+)",
    ]
    s = os.path.basename(path)
    for p in patts:
        m = re.search(p, s, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))

    # 兜底：提取路径中的第一个整数
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1


def read_code_exec_csv(p: str, time_col: str) -> pd.DataFrame:
    """
    读取单个 code-exec CSV，拆出 file/line 两列。
    需要至少包含列: code, time_col(默认'running_time(us)')
    """
    df = pd.read_csv(p)
    if "code" not in df.columns or time_col not in df.columns:
        raise ValueError(f"{p}: 必须包含列 'code' 与 '{time_col}'")

    # 从 'code' 列解析出 file 与 line
    # 形如 "/path/to/file.cpp:39"
    m = df["code"].astype(str).str.extract(r'(?P<file>[^:]+):(?P<line>\d+)$')
    df["file"] = m["file"]
    df["line"] = pd.to_numeric(m["line"], errors="coerce").astype("Int64")
    return df


def sum_target_time_us(df: pd.DataFrame, target_file: str, target_lines: List[int], time_col: str) -> float:
    """
    在单个 CSV（DataFrame）中过滤目标文件 + 目标行号，返回这些记录的时间和（微秒）
    """
    mask_file = df["file"].astype(str).str.endswith(target_file, na=False)
    if len(target_lines) > 0:
        mask_line = df["line"].isin(list(target_lines))
        mask = mask_file & mask_line
    else:
        # 如果不指定行号，就按整个文件求和
        mask = mask_file
    return float(df.loc[mask, time_col].sum())


def summarize_one_csv(csv_path: str, target_file: str, target_lines: List[int], time_col: str) -> Dict:
    """
    汇总单个 CSV：抽取 dataLength、目标时间（us）、以及辅助信息
    """
    df = read_code_exec_csv(csv_path, time_col)
    L = parse_datalength_from_path(csv_path)

    meas_us = sum_target_time_us(df, target_file, target_lines, time_col)
    file_total_us = float(df[df["file"].astype(str).str.endswith(target_file, na=False)][time_col].sum())
    grand_total_us = float(df[time_col].sum())

    present_lines = sorted(
        df[df["file"].astype(str).str.endswith(target_file, na=False)]["line"].dropna().unique().tolist()
    )
    found_lines = sorted(
        df[(df["file"].astype(str).str.endswith(target_file, na=False)) &
           (df["line"].isin(list(target_lines)))]["line"].dropna().unique().tolist()
    )

    note = ""
    if len(target_lines) > 0 and not found_lines:
        note = f"WARNING: 指定的行号 {sorted(list(target_lines))} 在 {os.path.basename(csv_path)} 中未出现；" \
               f"该点的 meas_us=0"

    return {
        "csv": csv_path,
        "dataLength": int(L),
        "meas_us": float(meas_us),
        "file_total_us": float(file_total_us),
        "grand_total_us": float(grand_total_us),
        "present_lines": present_lines,
        "found_lines": found_lines,
        "note": note
    }


# =========================
# ==   线性模型与评估     ==
# =========================
def fit_linear(L: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    拟合 y = alpha * L + beta
    返回 (alpha, beta)
    """
    # 设计矩阵 [L, 1]
    X = np.column_stack([L, np.ones_like(L)])
    # 最小二乘闭式解
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    return alpha, beta


def predict_linear(L: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return alpha * L + beta


def calc_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    residual = y - yhat
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mape = float(np.mean(np.abs(residual) / np.maximum(np.abs(y), eps)) * 100.0)
    return {"R2": r2, "RMSE": rmse, "MAPE": mape}


def main():
    parser = argparse.ArgumentParser(description="Fit GELU latency ~ dataLength (linear)")
    parser.add_argument("--glob", default=DEFAULTS["GLOB_PATTERN"])
    parser.add_argument("--target-file", default=DEFAULTS["TARGET_FILE"])
    parser.add_argument("--target-lines", default=DEFAULTS["TARGET_LINES"])
    parser.add_argument("--time-col", default=DEFAULTS["TIME_COL"])
    parser.add_argument("--out-dir", default=DEFAULTS["OUT_DIR"])
    args = parser.parse_args()

    target_lines = _norm_lines_to_set(args.target_lines)

    csv_list = sorted(glob.glob(args.glob, recursive=True))
    if not csv_list:
        raise FileNotFoundError(f"未找到任何 CSV：{args.glob}")

    rows = []
    for p in csv_list:
        try:
            rows.append(summarize_one_csv(p, args.target_file, target_lines, args.time_col))
        except Exception as e:
            print(f"[WARN] 跳过 {p}: {e}")

    df_sum = pd.DataFrame(rows)
    # 过滤无效点（dataLength<=0 或 整体时间缺失）
    df_sum = df_sum[(df_sum["dataLength"] > 0) & df_sum["meas_us"].notna()]
    if df_sum.empty:
        raise RuntimeError("no data")

    df_sum = df_sum.sort_values(by="dataLength").reset_index(drop=True)

    L = df_sum["dataLength"].to_numpy(dtype=float)
    y = df_sum["meas_us"].to_numpy(dtype=float)

    # 线性拟合
    alpha, beta = fit_linear(L, y)
    yhat = predict_linear(L, alpha, beta)
    metrics = calc_metrics(y, yhat)

    # 输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 保存汇总 CSV
    df_out = df_sum.copy()
    df_out["pred_us"] = yhat
    df_out["residual_us"] = df_out["meas_us"] - df_out["pred_us"]
    df_out.to_csv(os.path.join(args.out_dir, "gelu_latency_fit_summary.csv"), index=False)

    # 保存模型 JSON
    model = {
        "model": "time_us = alpha * dataLength + beta",
        "alpha": alpha,
        "beta": beta,
        "metrics": metrics,
        "target_file": args.target_file,
        "target_lines": sorted(list(target_lines)) if target_lines else "ALL_LINES_IN_FILE",
        "time_col": args.time_col,
        "num_points": int(len(df_out))
    }
    with open(os.path.join(args.out_dir, "gelu_latency_linear_model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    # 绘图 1：measured vs dataLength + 拟合线
    plt.figure()
    plt.scatter(L, y, label="Measured (us)")
    # 拟合线（按范围画）
    L_line = np.linspace(L.min(), L.max(), 100)
    plt.plot(L_line, predict_linear(L_line, alpha, beta), label="Fitted line (us)")
    plt.xlabel("dataLength")
    plt.ylabel("Latency (us)")
    plt.title(f"GELU latency ~ dataLength (Linear): R2={metrics['R2']:.4f}, MAPE={metrics['MAPE']:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "gelu_latency_fit_line.png"), dpi=160)

    # 绘图 2：measured vs predicted
    plt.figure()
    plt.scatter(y, yhat)
    ymin, ymax = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
    plt.plot([ymin, ymax], [ymin, ymax])
    plt.xlabel("Measured (us)")
    plt.ylabel("Predicted (us)")
    plt.title("GELU latency: measured vs predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "gelu_latency_meas_vs_pred.png"), dpi=160)

    # 控制台输出
    print("[OK] Linear model for GELU:")
    print("     time_us = alpha * dataLength + beta")
    print(f"     alpha = {alpha:.6e}, beta = {beta:.6e}")
    print("     metrics:", metrics)
    print(f"[OK] Summary CSV: {os.path.join(args.out_dir, 'gelu_latency_fit_summary.csv')}")
    print(f"[OK] Model JSON : {os.path.join(args.out_dir, 'gelu_latency_linear_model.json')}")
    print(f"[OK] Plots      : {args.out_dir}/gelu_latency_fit_line.png, "
          f"{args.out_dir}/gelu_latency_meas_vs_pred.png")


if __name__ == "__main__":
    main()
