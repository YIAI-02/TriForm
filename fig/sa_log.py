#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化：模拟退火（SA）权重存储格式优化的调参/搜索过程

支持两种来源：
1) 直接解析调试日志（推荐）：--log /path/to/debug.log
   - 从日志中提取每一轮(pass)的“时间改进量 Δt”和“格式变更比例”，
     并基于最后一次接受方案的预填/解码耗时，反向重建每一轮的“已接受方案总耗时”。
2) 可选读取结果 JSON（如果存在）：--report_dir ./output/weight_suggestions
   - 自动查找 all_passes_*.json / best_summary_*.json 增强信息（字段自适配）。
   - 若 JSON 格式无法识别，不影响日志路径的可视化。

输出：
- 三张图（默认保存到 --output_dir，默认 ./figs）
  1) sa_total_time_per_pass.png    —— 每轮“已接受方案”的总耗时曲线（并高亮最佳轮次）
  2) sa_time_improvement.png       —— 每轮 Δt（柱状图，>0 为加速，<0 为变慢）
  3) sa_format_change_ratio.png    —— 每轮格式映射变化比例（柱状图）
- 一份 CSV：sa_pass_metrics.csv    —— 汇总每轮指标（pass, delta_time, accepted_total_time, best_so_far, format_change_ratio）

使用示例：
    python sa_log.py --log ../algorithms/output/weight_suggestions/driver_debug.txt --report_dir ../algorithms/output/weight_suggestions --output_dir ./sa_figs
python sa_log.py --log ../algorithms/output/weight_suggestion_30/driver_debug.txt --report_dir ../algorithms/output/weight_suggestion_30 --output_dir ./sa_figs  
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import glob
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 兼容无显示环境
import matplotlib.pyplot as plt

# 颜色配置（按你的要求）
COL_DECODE  = "#395aad"  # Δt
COL_E2E     = "#1d2e53"  # Accepted total time

# -----------------------------
# 日志解析
# -----------------------------

PASS_START_RE = re.compile(r"Starting optimization pass (\d+)\s*/\s*(\d+)", re.IGNORECASE)
DELTA_RE      = re.compile(r"\[DELTA\]\s*Time improvement.*?:\s*([+\-]?\d+\.\d+)s", re.IGNORECASE)
FORMAT_RE     = re.compile(r"\[DELTA\]\s*Format map change ratio.*?:\s*(\d+\.\d+)", re.IGNORECASE)
BEST_FOUND_RE = re.compile(r"Best weight storage map \(found at pass (\d+)\)", re.IGNORECASE)
BEST_TIME_RE  = re.compile(r"Best total time:\s*([\d\.]+)s", re.IGNORECASE)
LAST_PREF_RE  = re.compile(r"Last accepted prefill\(sim\):\s*([\d\.]+)s", re.IGNORECASE)
LAST_DEC_RE   = re.compile(r"Last accepted decode\(sim\):\s*([\d\.]+)s", re.IGNORECASE)


def parse_log(log_path: str) -> Dict:
    """解析调试日志，返回结构化信息。"""
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"日志文件不存在：{log_path}")
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    passes: Dict[int, Dict] = {}
    current_pass: Optional[int] = None
    total_passes: Optional[int] = None

    best_pass: Optional[int] = None
    best_total_time: Optional[float] = None
    last_prefill: Optional[float] = None
    last_decode: Optional[float] = None

    for line in lines:
        # 起始
        m = PASS_START_RE.search(line)
        if m:
            current_pass = int(m.group(1))
            total_passes = int(m.group(2))
            passes.setdefault(current_pass, {})
            continue

        # Δt
        m = DELTA_RE.search(line)
        if m and current_pass is not None:
            passes.setdefault(current_pass, {})["delta_time"] = float(m.group(1))
            continue

        # format ratio
        m = FORMAT_RE.search(line)
        if m and current_pass is not None:
            passes.setdefault(current_pass, {})["format_change_ratio"] = float(m.group(1))
            continue

        # 最佳
        m = BEST_FOUND_RE.search(line)
        if m:
            best_pass = int(m.group(1))
            continue

        m = BEST_TIME_RE.search(line)
        if m:
            best_total_time = float(m.group(1))
            continue

        # 最后一次接受时间（用于反推）
        m = LAST_PREF_RE.search(line)
        if m:
            last_prefill = float(m.group(1))
            continue

        m = LAST_DEC_RE.search(line)
        if m:
            last_decode = float(m.group(1))
            continue

    # 对齐每轮
    if total_passes is None:
        total_passes = max(passes.keys()) if passes else 0

    for i in range(1, total_passes + 1):
        passes.setdefault(i, {})

    result = {
        "passes": passes,  # {pass_idx: {delta_time, format_change_ratio}}
        "num_passes": total_passes,
        "best_pass": best_pass,
        "best_total_time": best_total_time,
        "last_prefill": last_prefill,
        "last_decode": last_decode,
        "last_accepted_total": (last_prefill + last_decode) if (last_prefill is not None and last_decode is not None) else None,
    }
    return result


# -----------------------------
# JSON 报告辅助（可选）
# -----------------------------

def pick_latest(path_glob: str) -> Optional[str]:
    files = glob.glob(path_glob)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def try_read_json_reports(report_dir: Optional[str]) -> Dict:
    """尽力读取 JSON 报告，容错式解析字段。"""
    out = {"records": [], "best_pass": None, "best_total": None}
    if not report_dir or not os.path.isdir(report_dir):
        return out

    all_json = pick_latest(os.path.join(report_dir, "all_passes_*.json"))
    best_json = pick_latest(os.path.join(report_dir, "best_summary_*.json"))

    # 读取 all_passes
    if all_json and os.path.isfile(all_json):
        try:
            with open(all_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                if "passes" in data and isinstance(data["passes"], list):
                    records = data["passes"]
                elif "records" in data and isinstance(data["records"], list):
                    records = data["records"]
                else:
                    records = [data]
            else:
                records = []
            out["records"] = records
        except Exception as e:
            print(f"[WARN] 读取 {all_json} 失败：{e}", file=sys.stderr)

    # 读取 best_summary
    if best_json and os.path.isfile(best_json):
        try:
            with open(best_json, "r", encoding="utf-8") as f:
                best = json.load(f)
            for k in ["best_pass", "best_pass_idx", "best_idx", "pass", "pass_idx"]:
                if k in best:
                    out["best_pass"] = int(best[k])
                    break
            for k in ["best_total_time", "best_total", "total_time", "total"]:
                if k in best:
                    out["best_total"] = float(best[k])
                    break
        except Exception as e:
            print(f"[WARN] 读取 {best_json} 失败：{e}", file=sys.stderr)

    return out


# -----------------------------
# 反向重建每轮“已接受方案总耗时”
# -----------------------------

def reconstruct_accepted_times_from_log(num_passes: int,
                                        deltas: Dict[int, Optional[float]],
                                        last_accepted_total: Optional[float]) -> Dict[int, Optional[float]]:
    """
    根据“已接受 vs 上一已接受”的 Δt 以及最后一次已接受总耗时，反向重建每轮的已接受总耗时。
    记：T_i = 第 i 轮已接受总耗时；Δt_i = T_{i-1} - T_i
    则：T_{i-1} = T_i + Δt_i
    """
    accepted = {i: None for i in range(1, num_passes + 1)}
    if last_accepted_total is None or num_passes == 0:
        return accepted

    accepted[num_passes] = last_accepted_total
    for i in range(num_passes, 1, -1):
        d = deltas.get(i, None)
        if d is None or accepted[i] is None:
            break
        accepted[i - 1] = accepted[i] + d
    return accepted


# -----------------------------
# 可视化
# -----------------------------

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def plot_total_and_delta(df: pd.DataFrame, out_dir: str):
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=160)

    # ←—— 让背景透明（Figure 与 Axes）
    fig.patch.set_alpha(0.0)
    ax1.set_facecolor('none')

    x = df["pass"].astype(int)
    y_total = df["accepted_total_time"].astype(float)

    line1 = ax1.plot(
        x, y_total, marker="o", linestyle="-",
        color=COL_E2E, label="Accepted total time (s)"
    )
    ax1.set_xlabel("Pass")
    ax1.set_ylabel("Accepted Total Time (s)", color=COL_E2E)
    ax1.tick_params(axis="y", labelcolor=COL_E2E)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 右轴
    ax2 = ax1.twinx()
    ax2.set_facecolor('none')  # ←—— twin 轴也透明
    y_delta = df["delta_time"].astype(float)
    line2 = ax2.plot(
        x, y_delta, marker="s", linestyle="--",
        color=COL_DECODE, label="Δt vs prev-accepted (s)"
    )
    ax2.set_ylabel("Δt vs prev-accepted (s)", color=COL_DECODE)
    ax2.tick_params(axis="y", labelcolor=COL_DECODE)
    ax2.axhline(0.0, linewidth=1.0, color=COL_DECODE, alpha=0.15)

    # 高亮最佳
    if df["accepted_total_time"].notna().any():
        idx_min = y_total.idxmin()
        x_best = int(df.loc[idx_min, "pass"])
        y_best = float(df.loc[idx_min, "accepted_total_time"])
        ax1.scatter([x_best], [y_best], s=64, marker="*", color=COL_E2E, zorder=5)
        ax1.annotate(f"Best P{x_best}: {y_best:.6f}s",
                     (x_best, y_best), xytext=(6, 10), textcoords="offset points")

    # 合并图例，去掉图例底色
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", frameon=False)  # ←—— legend 透明

    fig.tight_layout()
    out_path = os.path.join(out_dir, "sa_total_and_delta.png")
    fig.savefig(out_path, bbox_inches="tight", transparent=True)  # ←—— 关键：透明导出
    plt.close(fig)
    print(f"[OK] 保存图像：{out_path}")



def plot_format_ratio(df: pd.DataFrame, out_dir: str):
    if "format_change_ratio" not in df.columns or df["format_change_ratio"].isna().all():
        print("[INFO] 日志/JSON未包含 format_change_ratio，跳过该图。")
        return

    fig = plt.figure(figsize=(8, 4.5), dpi=160)
    ax = fig.add_subplot(111)

    # ←—— 背景透明
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    ax.bar(df["pass"], df["format_change_ratio"].fillna(0.0).astype(float))
    ax.set_xlabel("Pass")
    ax.set_ylabel("Format Map Change Ratio")
    ax.set_title("Format-map change ratio per pass")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "sa_format_change_ratio.png")
    fig.savefig(out_path, bbox_inches="tight", transparent=True)  # ←—— 透明导出
    plt.close(fig)
    print(f"[OK] 保存图像：{out_path}")



# -----------------------------
# 主流程
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="可视化模拟退火优化过程（权重存储格式）")
    parser.add_argument("--log", type=str, required=False, help="调试日志路径（推荐）")
    parser.add_argument("--report_dir", type=str, default=None, help="JSON 报告目录（可选）")
    parser.add_argument("--output_dir", type=str, default="./figs", help="图表/CSV 输出目录")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    # 1) 日志解析
    log_info = {"passes": {}, "num_passes": 0, "best_pass": None, "best_total_time": None,
                "last_prefill": None, "last_decode": None, "last_accepted_total": None}
    if args.log:
        log_info = parse_log(args.log)

    passes = log_info["passes"]
    num_passes = log_info["num_passes"]
    last_total = log_info["last_accepted_total"]

    # 2) JSON 报告（可选）
    json_info = try_read_json_reports(args.report_dir)

    # 3) 组装 DataFrame
    rows = []
    deltas = {}
    for i in range(1, num_passes + 1):
        di = passes.get(i, {})
        d = di.get("delta_time", None)
        deltas[i] = d
        rows.append({
            "pass": i,
            "delta_time": (float(d) if d is not None else np.nan),
            "format_change_ratio": (float(di.get("format_change_ratio")) if di.get("format_change_ratio") is not None else np.nan),
            "accepted_total_time": np.nan,  # 先空，后续反推或用 JSON 覆盖
        })

    # 反推每轮已接受总耗时
    accepted = reconstruct_accepted_times_from_log(num_passes, deltas, last_total)
    for r in rows:
        i = r["pass"]
        if accepted.get(i, None) is not None:
            r["accepted_total_time"] = float(accepted[i])

    # 若 JSON 提供 more precise/complete 的耗时，尝试覆盖
    def candidate(d, names, default=None):
        for k in names:
            if k in d:
                return d[k]
        return default

    if json_info["records"]:
        json_records = json_info["records"]
        for idx, r in enumerate(rows):
            # 按 pass 匹配，否则按顺序兜底
            use_rec = None
            for rec in json_records:
                p = candidate(rec, ["pass", "pass_idx", "iteration", "iter"])
                if p is not None and int(p) == int(r["pass"]):
                    use_rec = rec
                    break
            if use_rec is None and idx < len(json_records):
                use_rec = json_records[idx]

            if use_rec:
                tot = candidate(use_rec, ["accepted_total_time", "accepted_total", "total_time", "total"], None)
                if tot is not None:
                    r["accepted_total_time"] = float(tot)

                dt = candidate(use_rec, ["time_improvement", "delta_time", "delta_total", "delta"], None)
                if dt is not None:
                    r["delta_time"] = float(dt)

                fr = candidate(use_rec, ["format_change_ratio", "format_ratio", "fmt_change_ratio"], None)
                if fr is not None:
                    r["format_change_ratio"] = float(fr)

    df = pd.DataFrame(rows).sort_values("pass").reset_index(drop=True)

    # best_so_far（便于分析）
    if df["accepted_total_time"].notna().any():
        df["best_so_far"] = df["accepted_total_time"].cummin()
    else:
        df["best_so_far"] = np.nan

    # 4) 输出 CSV
    csv_path = os.path.join(args.output_dir, "sa_pass_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] 写出汇总 CSV：{csv_path}")

    # 打印关键信息
    if log_info["best_pass"] is not None and log_info["best_total_time"] is not None:
        print(f"[INFO] 日志提示最佳：pass {log_info['best_pass']} | {log_info['best_total_time']:.6f}s")
    elif df["accepted_total_time"].notna().any():
        idx_min = df["accepted_total_time"].astype(float).idxmin()
        print(f"[INFO] 计算得到最佳：pass {int(df.loc[idx_min, 'pass'])} | {float(df.loc[idx_min, 'accepted_total_time']):.6f}s")

    # 5) 绘图
    if df["accepted_total_time"].notna().any():
        plot_total_and_delta(df, args.output_dir)
    else:
        print("[WARN] 缺少每轮“已接受方案总耗时”，无法绘制 sa_total_and_delta.png（请确保日志包含最后一次 prefill/ decode 或提供 JSON）")

    if "format_change_ratio" in df.columns and df["format_change_ratio"].notna().any():
        plot_format_ratio(df, args.output_dir)


if __name__ == "__main__":
    main()
