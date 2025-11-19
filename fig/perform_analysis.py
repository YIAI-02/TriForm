#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python perform_analysis.py \
  --algo_dir ../algorithms/output/len_sweep/llama_7b_fp16_b32/algo_heft

"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===== 色板（来自你给的第 4 张图片）=====
PALETTE_BLUES: List[str] = [
    "#e6eefb", "#ccddf8", "#b3ccf4", "#99bbf0", "#80abed", "#679ae9",
    "#4d89e5", "#3478e1", "#1a67de", "#0156da", "#014dc4", "#0145ae",
    "#013c99", "#013483", "#012b6d", "#002257", "#001a41", "#00112c",
    "#000916",
]

# 00 比例图三色（固定）
COLOR_NPU_TOTAL  = "#1d2e53"
COLOR_PIM_TOTAL  = "#395aad"
COLOR_COMM_TOTAL = "#84b4fc"

# 04（阶段内 NPU vs PIM）分组柱状两色，也取自动画板，保证风格一致
COLOR_GROUP_NPU = "#0156da"
COLOR_GROUP_PIM = "#4d89e5"


# ========== 目录与元信息 ==========
def find_output_anchor_dir(path: Path) -> Tuple[Path, Path]:
    """返回 (output 同级根目录, output 后的尾路径)"""
    parts = path.resolve().parts
    low   = [p.lower() for p in parts]
    if "output" not in low:
        return path.resolve(), Path(".")
    idx = max(i for i, s in enumerate(low) if s == "output")
    root = Path(*parts[:idx]) if idx > 0 else Path("/")
    tail = Path(*parts[idx+1:]) if idx+1 < len(parts) else Path(".")
    return root, tail


def discover_pairs(algo_dir: Path) -> List[str]:
    """发现 <lenpair>（去掉后缀），取 ops 与 comms 交集。"""
    ops = {p.stem[:-4] for p in algo_dir.glob("*_ops.csv") if p.name.endswith("_ops.csv")}
    cms = {p.stem[:-6] for p in algo_dir.glob("*_comms.csv") if p.name.endswith("_comms.csv")}
    pairs = sorted(list(ops & cms), key=lambda s: (
        int(re.match(r"(\d+)", s).group(1)) if re.match(r"(\d+)", s) else 10**9,
        int(re.search(r"x(\d+)", s).group(1)) if re.search(r"x(\d+)", s) else 10**9
    ))
    return pairs


def meta_from_algo_dir(algo_dir: Path) -> Dict[str, str]:
    """解析 sweep / model_pack / algo。"""
    s = str(algo_dir.resolve()).lower()
    sweep = "len_sweep" if "len_sweep" in s else "sweep_unknown"
    m = re.search(r"([\w\-.]+_(?:fp\d+|fp8|fp16|fp32|bf16|int\d+)_b\d+)", s)
    model_pack = m.group(1) if m else "unknown_model_pack"
    m = re.search(r"(algo[_-]?[a-z0-9]+)", s)
    algo = m.group(1) if m else "algo_unknown"
    return {"sweep": sweep, "model_pack": model_pack, "algo": algo}


def fig_dir_for_pair(algo_dir: Path, lenpair: str) -> Path:
    """fig 根与层级复刻（包含 <lenpair> 子目录）。"""
    root, tail = find_output_anchor_dir(algo_dir)
    fig_root   = root / "fig"
    outdir     = fig_root / tail / lenpair
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def fig_root_for_algo(algo_dir: Path) -> Path:
    """该 algo 的 fig 根目录（不含 <lenpair>），用于汇总图。"""
    root, tail = find_output_anchor_dir(algo_dir)
    outdir = root / "fig" / tail
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ========== 数据读取与标准化 ==========
def normalize_phase(val) -> str:
    s = str(val).strip().lower()
    if s in {"prefill", "pre", "pre-fill", "pre_fill", "prompt", "encode", "encoding"}:
        return "prefill"
    if s in {"decode", "decoding", "gen", "generate", "generation"}:
        return "decode"
    return "unknown"


def load_pair(algo_dir: Path, lenpair: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ops   = pd.read_csv(algo_dir / f"{lenpair}_ops.csv")
    comms = pd.read_csv(algo_dir / f"{lenpair}_comms.csv")

    # 关键列
    for df, name in [(ops, "ops"), (comms, "comms")]:
        if "duration" not in df.columns:
            raise ValueError(f"{name}.csv 缺少 'duration' 列")
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    # 设备类型
    if "device_type" not in ops.columns:
        if "mode" in ops.columns:
            ops["device_type"] = ops["mode"].astype(str).str.lower()
        else:
            raise ValueError(f"{lenpair}_ops.csv 缺少 'device_type'（或 'mode'）列")
    else:
        ops["device_type"] = ops["device_type"].astype(str).str.lower()

    # phase
    if "phase" not in ops.columns:
        raise ValueError(f"{lenpair}_ops.csv 缺少 'phase' 列（需要区分 prefill / decode）")
    ops["phase"] = ops["phase"].map(normalize_phase)

    # 其它列
    if "op" not in ops.columns:
        raise ValueError(f"{lenpair}_ops.csv 缺少 'op' 列")
    if "node_id" not in ops.columns:
        ops["node_id"] = "unknown_node"

    return ops, comms


# ========== 绘图基础 ==========
def cap_and_group_others(df: pd.DataFrame, key_col: str, val_col: str, topk: int) -> pd.DataFrame:
    d = df.sort_values(val_col, ascending=False)
    if topk and len(d) > topk:
        head = d.iloc[:topk].copy()
        rest = d.iloc[topk:][val_col].sum()
        head.loc[len(head)] = {key_col: "Others", val_col: rest}
        return head
    return d


def save_totals_ratio(ops: pd.DataFrame, comms: pd.DataFrame, out_csv: Path, out_png: Path):
    npu = ops.loc[ops["device_type"] == "npu", "duration"].sum()
    pim = ops.loc[ops["device_type"] == "pim", "duration"].sum()
    com = comms["duration"].sum()
    df  = pd.DataFrame({"category": ["NPU Compute", "PIM Compute", "Communication"],
                        "seconds":  [npu,             pim,             com]})
    df["percent"] = df["seconds"] / max(df["seconds"].sum(), 1e-9) * 100.0
    df.to_csv(out_csv, index=False)

    vals = df["seconds"].to_numpy(dtype=float)
    total = float(vals.sum()) if float(vals.sum()) > 0 else 1.0
    colors = [COLOR_NPU_TOTAL, COLOR_PIM_TOTAL, COLOR_COMM_TOTAL]
    labels = df["category"].tolist()

    fig = plt.figure(figsize=(6, 5))
    bottom = 0.0
    for i, v in enumerate(vals):
        p = v / total
        plt.bar([0], [p], bottom=bottom, color=colors[i], label=f"{labels[i]} ({p*100:.1f}%)")
        plt.text(0, bottom + p/2, f"{p*100:.1f}%\n{v:.2f}s",
                 ha="center", va="center", fontsize=9, color="white")
        bottom += p
    plt.ylim(0, 1)
    plt.xticks([0], ["Time Ratio"])
    plt.ylabel("Share of total time")
    plt.title("NPU vs PIM vs Communication (ratio)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    return float(npu), float(pim), float(com)


def plot_03_pie(ops: pd.DataFrame, device: str, phase: str, out_png: Path, topk: int = 25):
    """
    03：单设备×单阶段，按 op 类型聚合（先对节点求和），饼图展示占比。
    """
    d = (
        ops[(ops["device_type"] == device) & (ops["phase"] == phase)]
        .groupby(["op"], as_index=False)["duration"].sum()
    )
    if d.empty:
        fig = plt.figure(figsize=(5, 5))
        plt.title(f"{device.upper()} - {phase} (no data)")
        plt.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    d = cap_and_group_others(d, "op", "duration", topk)
    values = d["duration"].to_numpy(dtype=float)
    labels = d["op"].tolist()
    colors = [PALETTE_BLUES[i % len(PALETTE_BLUES)] for i in range(len(labels))]

    fig = plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title(f"{device.upper()} — {phase.capitalize()} — op-type share (sum over nodes)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_04_phase_compare(ops: pd.DataFrame, phase: str, out_png: Path, topk: int = 20):
    """
    04：同一阶段内对比 NPU vs PIM（分组柱状，Top-K 算子）。
    """
    sub = ops[ops["phase"] == phase]
    if sub.empty:
        fig = plt.figure(figsize=(6, 4))
        plt.title(f"{phase} (no data)")
        plt.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    # 先对节点求和，再对 op×device 聚合
    d = sub.groupby(["device_type", "op"], as_index=False)["duration"].sum()
    piv = d.pivot(index="op", columns="device_type", values="duration").fillna(0.0)
    for col in ("npu", "pim"):
        if col not in piv.columns:
            piv[col] = 0.0
    piv = piv[["npu", "pim"]]

    keep = piv.sum(axis=1).sort_values(ascending=False).head(topk).index
    piv  = piv.loc[keep]

    xs = np.arange(len(piv))
    width = 0.44
    fig = plt.figure(figsize=(max(10.0, 0.6 * len(piv)), 6))
    plt.bar(xs - width/2, piv["npu"].to_numpy(), width=width, color=COLOR_GROUP_NPU, label="NPU")
    plt.bar(xs + width/2, piv["pim"].to_numpy(), width=width, color=COLOR_GROUP_PIM, label="PIM")
    plt.xticks(xs, piv.index.tolist(), rotation=55, ha="right")
    plt.ylabel("Total compute time (s)")
    plt.title(f"{phase.capitalize()} — NPU vs PIM by op type (sum over nodes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_00_all_pairs_mix_ratio(summary: pd.DataFrame, out_png: Path):
    """
    汇总所有长度对的 00 比例图：每个长度对一个堆叠条。
    X 轴标签：P<prefill>/D<decode>（例如 P128/D1024）。
    """
    if summary.empty:
        return
    vals = summary[["npu", "pim", "comm"]].to_numpy(dtype=float)
    totals = vals.sum(axis=1)
    totals[totals == 0] = 1.0
    parts = (vals.T / totals).T  # 比例

    x = np.arange(len(summary))
    fig = plt.figure(figsize=(max(10.0, 1.2 * len(summary)), 6))
    bottom = np.zeros(len(summary), dtype=float)
    labels = ["NPU", "PIM", "COMM"]
    colors = [COLOR_NPU_TOTAL, COLOR_PIM_TOTAL, COLOR_COMM_TOTAL]

    for i in range(3):
        plt.bar(x, parts[:, i], bottom=bottom, color=colors[i], label=labels[i])
        bottom += parts[:, i]

    # 标注具体百分比
    for xi in range(len(summary)):
        y0 = 0.0
        for i in range(3):
            h = parts[xi, i]
            if h > 0.05:
                plt.text(xi, y0 + h/2, f"{h*100:.1f}%", ha="center", va="center", fontsize=8, color="white")
            y0 += h

    # 轴与标题
    xticks = [f"P{row['prefill']}/D{row['decode']}" for _, row in summary.iterrows()]
    plt.xticks(x, xticks, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Share of total time")
    plt.title("All length pairs — NPU vs PIM vs Communication (ratio)")
    plt.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0))
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close(fig)


# ========== 单个长度对处理 ==========
def process_one_lenpair(algo_dir: Path, lenpair: str, topk_device_phase: int, topk_phase_compare: int) -> Dict[str, float]:
    """
    处理一个长度对：生成 per-pair 图表及表格；返回供汇总 00 图使用的 npu/pim/comm。
    """
    ops, comms = load_pair(algo_dir, lenpair)
    outdir = fig_dir_for_pair(algo_dir, lenpair)

    # 00：比例图 + totals
    npu, pim, com = save_totals_ratio(
        ops, comms,
        out_csv=outdir / "00_totals.csv",
        out_png=outdir / "00_mix_ratio.png"
    )

    # 03：四张饼图（设备×阶段）
    plot_03_pie(ops, "pim", "prefill", outdir / "03_pim_prefill_pie.png", topk=topk_device_phase)
    plot_03_pie(ops, "pim", "decode",  outdir / "03_pim_decode_pie.png",  topk=topk_device_phase)
    plot_03_pie(ops, "npu", "prefill", outdir / "03_npu_prefill_pie.png", topk=topk_device_phase)
    plot_03_pie(ops, "npu", "decode",  outdir / "03_npu_decode_pie.png",  topk=topk_device_phase)

    # 03：导出聚合表（确认“所有 layer/节点均已累加”）
    agg = (
        ops.groupby(["device_type", "phase", "op", "node_id"], as_index=False)["duration"].sum()
        .groupby(["device_type", "phase", "op"], as_index=False)["duration"].sum()
        .sort_values(["device_type", "phase", "duration"], ascending=[True, True, False])
    )
    agg.to_csv(outdir / "03_ops_by_type_phase_device.csv", index=False)

    # 04：阶段内 NPU vs PIM 对比（两张）
    plot_04_phase_compare(ops, "prefill", outdir / "04_prefill_pim_vs_npu_by_optype.png", topk=topk_phase_compare)
    plot_04_phase_compare(ops, "decode",  outdir / "04_decode_pim_vs_npu_by_optype.png",  topk=topk_phase_compare)

    # 返回汇总用
    try:
        prefill, decode = lenpair.split("x")
        prefill_i = int(prefill)
        decode_i  = int(decode)
    except Exception:
        prefill_i, decode_i = -1, -1

    return {
        "lenpair": lenpair,
        "prefill": prefill_i,
        "decode": decode_i,
        "npu": npu, "pim": pim, "comm": com
    }


# ========== 主入口 ==========
def main():
    parser = argparse.ArgumentParser(description="len_sweep 可视化（含 00 合并图 + 03 饼图）")
    parser.add_argument("--algo_dir", required=True, help="目录：.../output/len_sweep/<model_pack>/<algo_xxx>")
    parser.add_argument("--pair", default=None, help="只处理某个长度对（如 128x1024），缺省处理全部")
    parser.add_argument("--topk-device-phase", type=int, default=25, help="03 饼图：单设备单阶段保留的算子数（其余合并为 Others）")
    parser.add_argument("--topk-phase-compare", type=int, default=20, help="04 图：阶段内 NPU vs PIM 的 Top-K 算子")
    args = parser.parse_args()

    algo_dir = Path(args.algo_dir).resolve()
    if not algo_dir.exists():
        raise FileNotFoundError(f"algo_dir 不存在：{algo_dir}")

    # 发现所有长度对
    pairs = [args.pair] if args.pair else discover_pairs(algo_dir)
    if not pairs:
        raise RuntimeError(f"未在 {algo_dir} 发现 *_ops.csv 与 *_comms.csv 的成对文件。")

    print("[发现长度对] ", ", ".join(pairs))

    # 单对逐一处理，同时收集 00 的汇总数据
    summary_rows: List[Dict[str, float]] = []
    for lp in pairs:
        print(f"\n==> 处理：{lp}")
        row = process_one_lenpair(
            algo_dir, lp,
            topk_device_phase=args.topk_device_phase,
            topk_phase_compare=args.topk_phase_compare
        )
        summary_rows.append(row)

    # 00 合并图（放在该 algo 的 fig 根）
    fig_root = fig_root_for_algo(algo_dir)
    summary_df = pd.DataFrame(summary_rows).sort_values(["prefill", "decode"]).reset_index(drop=True)
    summary_df.to_csv(fig_root / "00_all_pairs_totals.csv", index=False)
    plot_00_all_pairs_mix_ratio(summary_df, fig_root / "00_all_pairs_mix_ratio.png")

    print("\n✅ 完成。单对图表写入各自子目录；合并图：", fig_root / "00_all_pairs_mix_ratio.png")


if __name__ == "__main__":
    main()
