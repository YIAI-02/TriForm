#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from dataclasses import dataclass

"""
python perform_analysis.py \
  --paths ../algorithms/output/baseline_sweep_pima/llama_7b_int8_b1/algo_heft --stride 16 

"""
COLOR_NPU_TOTAL  = "#1d2e53"
COLOR_NPU_PIM    = "#013483"
COLOR_PIM_TOTAL  = "#0145ae"
COLOR_PIM_COMM   = "#80abed"
COLOR_COMM_TOTAL = "#ccddf8"
COLOR_NPU_COMM   = "#1a67de"

LABELS_6 = [
    "NPU",
    "PIM",
    "DATA_Trans",
    "NPU_PIM_overlap",
    "NPU_DATA_Trans_overlap",
    "PIM_DATA_Trans_overlap",
]

COLORS_6 = [
    COLOR_NPU_TOTAL,     # NPU
    COLOR_PIM_TOTAL,     # PIM
    COLOR_COMM_TOTAL,    # DATA_Trans
    COLOR_NPU_PIM,       # NPU_PIM_overlap
    COLOR_NPU_COMM,      # NPU_DATA_Trans_overlap
    COLOR_PIM_COMM,      # PIM_DATA_Trans_overlap
]

TITLE_COLORS = [
    "#0d3b66",
    "#007f5f",
    "#ff7b00",
    "#6a4c93",
]


@dataclass(frozen=True)
class PairFiles:
    key: str
    ops_path: Path
    comms_path: Path
    prefill_len: int
    decode_len: int


def parse_lenpair_from_stem(stem: str) -> Tuple[int, int]:
    """从 stem 中抓 prefill/decode 长度，例如 xxx_128x1024 -> (128, 1024)。抓不到就返回 (0,0)。"""
    m = re.search(r"(\d+)x(\d+)", stem)
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def discover_pairs_in_dir(d: Path) -> Dict[str, Tuple[Path, Path]]:
    """在目录 d 里找 lenpair -> (ops_path, comms_path)。"""
    ops_files = list(d.glob("*_ops.csv"))
    comms_files = list(d.glob("*_comms.csv"))

    ops_map = {p.stem[:-4]: p for p in ops_files if p.name.endswith("_ops.csv")}         # remove "_ops"
    comms_map = {p.stem[:-6]: p for p in comms_files if p.name.endswith("_comms.csv")}  # remove "_comms"

    out: Dict[str, Tuple[Path, Path]] = {}
    for k in sorted(set(ops_map.keys()) & set(comms_map.keys())):
        out[k] = (ops_map[k], comms_map[k])
    return out


def collect_pairs_from_paths(paths: List[Path]) -> List[PairFiles]:
    """
    输入可以是：
    - 目录：扫描 *_ops.csv / *_comms.csv
    - 文件：如果是 *_ops.csv 或 *_comms.csv，就补齐同目录下的另一个

    返回 PairFiles 列表（去重后）。
    """
    pairs: Dict[str, Tuple[Path, Path]] = {}

    for p in paths:
        p = p.expanduser().resolve()
        if p.is_dir():
            pairs.update(discover_pairs_in_dir(p))
            continue

        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {p}")

        name = p.name
        if name.endswith("_ops.csv"):
            key = p.stem[:-4]
            ops_path = p
            comms_path = p.with_name(f"{key}_comms.csv")
            if not comms_path.exists():
                raise FileNotFoundError(f"需要匹配的 comms 文件不存在: {comms_path}")
            pairs[key] = (ops_path, comms_path)
        elif name.endswith("_comms.csv"):
            key = p.stem[:-6]
            comms_path = p
            ops_path = p.with_name(f"{key}_ops.csv")
            if not ops_path.exists():
                raise FileNotFoundError(f"需要匹配的 ops 文件不存在: {ops_path}")
            pairs[key] = (ops_path, comms_path)
        else:
            raise ValueError(f"不支持的文件名（需要 *_ops.csv 或 *_comms.csv）: {p}")

    out: List[PairFiles] = []
    for key, (ops_path, comms_path) in pairs.items():
        prefill_len, decode_len = parse_lenpair_from_stem(key)
        out.append(PairFiles(
            key=key,
            ops_path=ops_path,
            comms_path=comms_path,
            prefill_len=prefill_len,
            decode_len=decode_len,
        ))

    out.sort(key=lambda x: (x.prefill_len, x.decode_len, x.key))
    return out


def load_pair_files(pair: PairFiles) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ops = pd.read_csv(pair.ops_path)
    comms = pd.read_csv(pair.comms_path)

    for df, name in ((ops, "ops"), (comms, "comms")):
        for col in ("start", "end", "duration"):
            if col not in df.columns:
                raise ValueError(f"{name} 文件缺少 '{col}' 列：{pair.key}")
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "device_type" not in ops.columns:
        if "mode" in ops.columns:
            ops["device_type"] = ops["mode"].astype(str).str.lower()
        else:
            raise ValueError(f"{pair.ops_path.name} 缺少 device_type（或 mode）列")
    else:
        ops["device_type"] = ops["device_type"].astype(str).str.lower()

    if "phase" not in ops.columns or "phase" not in comms.columns:
        raise ValueError(f"{pair.key}: ops/comms 缺少 phase 列（需要区分 prefill/decode）")

    return ops, comms


def classify_time_slices(ops_phase: pd.DataFrame, comms_phase: pd.DataFrame) -> Dict[frozenset, float]:
    """
    扫时间线，统计每个“活跃集合”（n,p,c）的持续时间：
      n = NPU 任意 op 活跃
      p = PIM 任意 op 活跃
      c = 任意 comm 活跃
    """
    events: List[Tuple[float, int, str]] = []

    for _, row in ops_phase.iterrows():
        dev = str(row["device_type"]).lower()
        if dev not in ("npu", "pim"):
            continue
        cat = "n" if dev == "npu" else "p"
        s, e = float(row["start"]), float(row["end"])
        events.append((s, 1, cat))
        events.append((e, -1, cat))

    for _, row in comms_phase.iterrows():
        s, e = float(row["start"]), float(row["end"])
        events.append((s, 1, "c"))
        events.append((e, -1, "c"))

    if not events:
        return {}

    events.sort(key=lambda x: (x[0], -x[1]))  # 同一时刻：先处理 end 再处理 start

    active: set[str] = set()
    totals: Dict[frozenset, float] = {}
    prev_t = events[0][0]

    for t, typ, cat in events:
        if t > prev_t and active:
            key = frozenset(active)
            totals[key] = totals.get(key, 0.0) + (t - prev_t)

        if typ == 1:
            active.add(cat)
        else:
            active.discard(cat)

        prev_t = t

    return totals


def compute_overlap_6bins(ops: pd.DataFrame, comms: pd.DataFrame, stride: int) -> Dict[str, float]:
    """
    返回 6 个扇区的“加权时间”，decode 段会乘 stride。
    """
    bins = {k: 0.0 for k in LABELS_6}

    def add_to_bin(active_set: set[str], dur: float):
        if active_set == {"n"}:
            bins["NPU"] += dur
        elif active_set == {"p"}:
            bins["PIM"] += dur
        elif active_set == {"c"}:
            bins["DATA_Trans"] += dur
        elif active_set == {"n", "p"}:
            bins["NPU_PIM_overlap"] += dur
        elif active_set == {"n", "c"}:
            bins["NPU_DATA_Trans_overlap"] += dur
        elif active_set == {"p", "c"}:
            bins["PIM_DATA_Trans_overlap"] += dur
        elif active_set == {"n", "p", "c"}:
            # 三者同时活跃：为了仍然只有 6 块，这里平均拆到 3 个 overlap
            share = dur / 3.0
            bins["NPU_PIM_overlap"] += share
            bins["NPU_DATA_Trans_overlap"] += share
            bins["PIM_DATA_Trans_overlap"] += share

    for phase in ("prefill", "decode"):
        scale = 1.0 if phase == "prefill" else float(stride)
        ops_ph = ops[ops["phase"] == phase]
        comms_ph = comms[comms["phase"] == phase]
        totals = classify_time_slices(ops_ph, comms_ph)

        for k, v in totals.items():
            add_to_bin(set(k), v * scale)

    return bins

def plot_2x2(pairs: List[PairFiles], stride: int, out_png: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    legend_handles = None

    for i, ax in enumerate(axes):
        if i >= len(pairs):
            ax.axis("off")
            continue

        pair = pairs[i]
        ops, comms = load_pair_files(pair)
        bins = compute_overlap_6bins(ops, comms, stride=stride)

        sizes = [bins[k] for k in LABELS_6]
        total = sum(sizes)
        if total <= 0:
            ax.text(0.5, 0.5, "无有效数据", ha="center", va="center")
            ax.axis("off")
            continue

        ratios = [v / total for v in sizes]  # 只展示比例，不展示绝对时间

        wedges, _, _ = ax.pie(
            ratios,
            labels=LABELS_6,
            colors=COLORS_6,
            autopct=lambda x: f"{x:.1f}%",
            startangle=90,
            radius=0.5,
            textprops={"fontsize":16},
        )
        if legend_handles is None:
            legend_handles = wedges
        ax.axis("equal")

        title_color = TITLE_COLORS[i % len(TITLE_COLORS)]
        if pair.prefill_len and pair.decode_len:
            ax.set_title(
                f"{pair.key}\n(prefill={pair.prefill_len}, decode={pair.decode_len}, stride={stride})",
                fontsize=18,
                color=title_color,
            )
        else:
            ax.set_title(
                f"{pair.key}\n(stride={stride})",
                fontsize=18,
                color=title_color,
            )

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            LABELS_6,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            fontsize=14,
            frameon=False,
        )

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="指定路径（文件或目录），自动解析 prefill x decode，并画 2x2 六扇区重叠饼图"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="一个或多个路径：可以是目录（扫描 *_ops/_comms.csv），也可以是 *_ops.csv / *_comms.csv 文件",
    )
    parser.add_argument("--stride", type=int, default=16, help="decode 采样 stride（decode 时间 * stride）")
    parser.add_argument("--out", type=str, default="overlap_6bins_pies.png", help="输出图片文件名")
    parser.add_argument("--max_pairs", type=int, default=4, help="最多画几组（默认 4 -> 2x2）")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    pairs = collect_pairs_from_paths(paths)
    if not pairs:
        raise SystemExit("没有发现任何可用的 *_ops.csv / *_comms.csv 对")

    pairs = pairs[: args.max_pairs]  # 默认取前 4 组
    out_png = Path(args.out).expanduser().resolve()

    plot_2x2(pairs, stride=args.stride, out_png=out_png)
    print(f"✅ 已输出：{out_png}")


if __name__ == "__main__":
    main()
