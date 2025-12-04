#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python perform_analysis.py \
  --root /Users/yangjiaqi/WW/project_1/python/TriForm_bak/TriForm/algorithms/output/lens_eval_sweep/hw_scale_down_pima/st16/mixtral_8x7b_int8_b1
'''
from __future__ import annotations
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 服务器/无GUI环境
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Wedge


# ====== 完全沿用你给的配色/标签/标题风格 ======
COLOR_NPU_TOTAL  = "#1d2e53"
COLOR_PIM_TOTAL  = "#0145ae"
COLOR_COMM_TOTAL = "#ccddf8"

LABELS_6 = [
    "NPU",
    "PIM",
    "DATA_Trans",
    "NPU_PIM_overlap",
    "NPU_DATA_Trans_overlap",
    "PIM_DATA_Trans_overlap",
]

COLORS_6 = [
    COLOR_NPU_TOTAL,
    COLOR_PIM_TOTAL,
    COLOR_COMM_TOTAL,
    COLOR_NPU_TOTAL,
    COLOR_COMM_TOTAL,
    COLOR_PIM_TOTAL,
]

OVERLAP_COLOR_BANDS = {
    "NPU_PIM_overlap": (COLOR_NPU_TOTAL, COLOR_PIM_TOTAL),
    "NPU_DATA_Trans_overlap": (COLOR_NPU_TOTAL, COLOR_COMM_TOTAL),
    "PIM_DATA_Trans_overlap": (COLOR_PIM_TOTAL, COLOR_COMM_TOTAL),
}


# overlap_summary/segments 里用到的 label
COL_NPU_ONLY = "NPU_only"
COL_PIM_ONLY = "PIM_only"
COL_COMM_ONLY = "COMM_only"
COL_NP = "NPU+PIM"
COL_NC = "NPU+COMM"
COL_PC = "PIM+COMM"
COL_NPC = "NPU+PIM+COMM"
COL_IDLE = "IDLE"


@dataclass(frozen=True)
class OverlapCSV:
    tag: str
    prefill: int
    decode: int
    kind: str  # "summary" or "segments"
    path: Path


_RE_FILE = re.compile(
    r"^(?P<tag>.+)_(?P<prefill>\d+)x(?P<decode>\d+)_overlap_(?P<kind>summary|segments)\.csv$"
)


def find_algo_dirs(root: Path, algo_name: str) -> List[Path]:
    root = root.resolve()
    if root.is_dir() and root.name == algo_name:
        return [root]
    dirs = sorted([p for p in root.rglob(algo_name) if p.is_dir() and p.name == algo_name])
    return dirs


def detect_stride_from_path(p: Path) -> Optional[int]:
    # 例如 .../st16/... -> 16（取路径中最后一个 stXX）
    st = None
    for part in p.parts:
        m = re.fullmatch(r"st(\d+)", part)
        if m:
            st = int(m.group(1))
    return st


def collect_overlap_csvs(algo_dir: Path) -> List[OverlapCSV]:
    algo_dir = algo_dir.resolve()

    # 优先 summary；如不存在则用 segments
    summaries = list(algo_dir.glob("*_overlap_summary.csv"))
    segments  = list(algo_dir.glob("*_overlap_segments.csv"))

    chosen = summaries if summaries else segments
    out: List[OverlapCSV] = []
    for f in chosen:
        m = _RE_FILE.match(f.name)
        if not m:
            continue
        out.append(
            OverlapCSV(
                tag=m.group("tag"),
                prefill=int(m.group("prefill")),
                decode=int(m.group("decode")),
                kind=m.group("kind"),
                path=f,
            )
        )
    out.sort(key=lambda x: (x.tag, x.prefill, x.decode))
    return out


def _phase_table_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "phase" not in df.columns:
        raise ValueError(f"{path} 缺少 phase 列")

    # segments：有 label + duration
    if "label" in df.columns and "duration" in df.columns:
        table = (
            df.groupby(["phase", "label"], as_index=False)["duration"]
            .sum()
            .pivot(index="phase", columns="label", values="duration")
            .fillna(0.0)
        )
        return table

    # summary：直接 groupby 相加（通常每 phase 只有一行，但这样更鲁棒）
    numeric_cols = [c for c in df.columns if c != "phase" and pd.api.types.is_numeric_dtype(df[c])]
    table = df.groupby("phase")[numeric_cols].sum()
    return table


def _get(table: pd.DataFrame, phase: str, col: str) -> float:
    if phase not in table.index:
        return 0.0
    v = table.loc[phase].get(col, 0.0)
    try:
        return float(v)
    except Exception:
        return 0.0


def compute_6bins_from_overlap_csv(path: Path) -> Tuple[List[float], Dict[str, float], float]:
    """
    返回：
        - sizes_6: [NPU, PIM, COMM, NP_ol, NC_ol, PC_ol] 的总时长（prefill+decode）
        - debug: 额外信息
        - used_decode_scale: summary 里记录的 decode_multiplier（没有则为1，仅供展示）
    """
    table = _phase_table_from_csv(path)

    def phase_bins(phase: str, scale: float) -> Dict[str, float]:
        npu_only = _get(table, phase, COL_NPU_ONLY) * scale
        pim_only = _get(table, phase, COL_PIM_ONLY) * scale
        comm_only = _get(table, phase, COL_COMM_ONLY) * scale
        np = _get(table, phase, COL_NP) * scale
        nc = _get(table, phase, COL_NC) * scale
        pc = _get(table, phase, COL_PC) * scale
        npc = _get(table, phase, COL_NPC) * scale

        # 三者 overlap 平均拆到三种 overlap
        share = npc / 3.0 if npc > 0 else 0.0
        return {
            "NPU": npu_only,
            "PIM": pim_only,
            "COMM": comm_only,
            "NP": np + share,
            "NC": nc + share,
            "PC": pc + share,
        }

    reported_scale = 1.0
    if "decode_multiplier" in table.columns:
        try:
            reported_scale = float(_get(table, "decode", "decode_multiplier"))
        except Exception:
            reported_scale = 1.0

    pre = phase_bins("prefill", 1.0)
    dec = phase_bins("decode", 1.0)

    total = {k: pre.get(k, 0.0) + dec.get(k, 0.0) for k in ["NPU", "PIM", "COMM", "NP", "NC", "PC"]}
    sizes_6 = [total["NPU"], total["PIM"], total["COMM"], total["NP"], total["NC"], total["PC"]]

    debug = {
        "prefill_active": sum(pre.values()),
        "decode_active_raw_or_weighted": sum(dec.values()),
        "decode_scale_used": reported_scale,
    }
    return sizes_6, debug, reported_scale


def make_default_pdf_name(algo_dir: Path, tag: str) -> str:
    algo_dir = algo_dir.resolve()
    model = algo_dir.parent.name if algo_dir.parent else "model"
    algo = algo_dir.name
    st = detect_stride_from_path(algo_dir) or 1
    # 例如：mixtral_8x7b_int8_b1_st16_algo_heft_heft_kv_first_overlap_pies.pdf
    return f"{model}_st{st}_{algo}_{tag}_overlap_pies.pdf"


def plot_tag_to_pdf(
    algo_dir: Path,
    tag: str,
    csvs: List[OverlapCSV],
    *,
    out_pdf: Path,
) -> None:
    # 每页 2x2
    per_page = 4
    csvs = sorted(csvs, key=lambda x: (x.prefill, x.decode))

    with PdfPages(out_pdf) as pdf:
        for page_start in range(0, len(csvs), per_page):
            page_items = csvs[page_start:page_start + per_page]

            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

            for ax_i, ax in enumerate(axes):
                if ax_i >= len(page_items):
                    ax.axis("off")
                    continue

                item = page_items[ax_i]
                sizes, dbg, _ = compute_6bins_from_overlap_csv(item.path)

                total = sum(sizes)
                if total <= 0:
                    ax.text(0.5, 0.5, "无有效数据", ha="center", va="center")
                    ax.axis("off")
                    continue

                ratios = [v / total for v in sizes]

                wedges, label_texts, autopct_texts = ax.pie(
                    ratios,
                    labels=LABELS_6,
                    colors=COLORS_6,
                    autopct=lambda x: f"{x:.1f}%",
                    startangle=90,
                    radius=0.95,
                    textprops={"fontsize": 12},
                )
                plt.setp(autopct_texts, fontsize=11)

                # overlap 类别直接叠加两层完整的扇形颜色
                for wedge, label in zip(wedges, LABELS_6):
                    if label not in OVERLAP_COLOR_BANDS:
                        continue
                    theta1, theta2 = wedge.theta1, wedge.theta2
                    color_a, color_b = OVERLAP_COLOR_BANDS[label]
                    ax.add_patch(
                        Wedge(
                            center=(0, 0),
                            r=0.95,
                            theta1=theta1,
                            theta2=theta2,
                            facecolor=color_a,
                            alpha=0.55,
                            linewidth=0,
                            zorder=wedge.get_zorder() + 0.1,
                        )
                    )
                    ax.add_patch(
                        Wedge(
                            center=(0, 0),
                            r=0.95,
                            theta1=theta1,
                            theta2=theta2,
                            facecolor=color_b,
                            alpha=0.55,
                            linewidth=0,
                            zorder=wedge.get_zorder() + 0.2,
                        )
                    )

                # 标签放在饼图外侧，通过箭头指向相应扇区
                ax.axis("equal")

                ax.set_title(
                    f"prefill={item.prefill}  decode={item.decode}",
                    fontsize=18,
                    color="black",
                    fontweight="bold",
                )

            # 页眉（可选）：写清楚 algo_dir
            fig.suptitle(str(algo_dir), fontsize=12)
            plt.tight_layout(rect=(0, 0.08, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="自动扫描 algo_heft 下所有 *_overlap_summary.csv 画饼图，并自动命名输出 PDF（多页2x2）"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="根目录：不给就用当前目录。会递归查找 algo_heft。",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="algo_heft",
        help="算法目录名，默认 algo_heft",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 PDF 路径。不填则自动在 algo_dir 内生成：<model>_stXX_<algo>_<tag>_overlap_pies.pdf",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    algo_dirs = find_algo_dirs(root, args.algo)
    if not algo_dirs:
        raise SystemExit(f"❌ 在 {root} 下没找到目录: {args.algo}")

    for algo_dir in algo_dirs:
        all_csvs = collect_overlap_csvs(algo_dir)
        if not all_csvs:
            print(f"⚠️ 目录里没找到 overlap csv: {algo_dir}")
            continue

        # 按 tag 分组（例如 heft_kv_first）
        by_tag: Dict[str, List[OverlapCSV]] = {}
        for c in all_csvs:
            by_tag.setdefault(c.tag, []).append(c)

        for tag, csvs in by_tag.items():
            if args.out:
                out_pdf = Path(args.out).expanduser().resolve()
                # 如果用户给的是目录，就放在目录下自动命名
                if out_pdf.exists() and out_pdf.is_dir():
                    out_pdf = out_pdf / make_default_pdf_name(algo_dir, tag)
            else:
                out_pdf = algo_dir / make_default_pdf_name(algo_dir, tag)

            plot_tag_to_pdf(
                algo_dir, tag, csvs,
                out_pdf=out_pdf,
            )
            print(f"✅ 写出: {out_pdf}")


if __name__ == "__main__":
    main()
