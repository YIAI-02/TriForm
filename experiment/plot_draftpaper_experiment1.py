
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Plot Experiment-1 results (grouped by policy, colored by model).

python plot_draftpaper_experiment1.py \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/qwen_1.8b_int8_b1_s64/baseline_compare_128x1024.json --label "qwen-1.8b b1" \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/qwen_7b_int8_b8_s64/baseline_compare_128x1024.json   --label "qwen-7b b8" \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/palm_62b_int8_b32_s64/baseline_compare_128x1024.json  --label "palm-62b b32" \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/mixtral_8x7b_int8_b8_s64/baseline_compare_128x1024.json   --label "mixtral-8x7b b8" \
  --out ../figs/draftpaper_exp1/exp1_policy_x_models_128x1024.pdf\
  --linear


python plot_draftpaper_experiment1.py \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/qwen_1.8b_int8_b1_s64/baseline_compare_1024x128.json --label "qwen-1.8b b1" \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/qwen_7b_int8_b8_s64/baseline_compare_1024x128.json   --label "qwen-7b b8" \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/palm_62b_int8_b32_s64/baseline_compare_1024x128.json  --label "palm-62b b32" \
  --json ../algorithms/output/experiment_npu/hw_npu_2aim/st64/mixtral_8x7b_int8_b8_s64/baseline_compare_1024x128.json   --label "mixtral-8x7b b8" \
  --out ../figs/draftpaper_exp1/exp1_policy_x_models_1024x128.pdf\
  --linear
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ========== 固定颜色（按 prefill / decode 区分） ==========
COLOR_PREFILL = "#FFBF02"
COLOR_DECODE = "#0092FE"

# ========== policy 规范化（与你之前脚本语义对齐） ==========
ALIASES = {
    "atten_on_pim": "attn_on_pim",
    "attn_on_pim": "attn_on_pim",
    "weight_on_pim": "weights_on_pim",
    "weights_on_pim": "weights_on_pim",
    "hefthint": "this work",  # 你之前脚本把 hefthint 显示为 this work
}
EXCLUDE_ALGOS = {"heft"}  # Experiment-1 固定为 7 个 policy

PREFERRED_ORDER = [
    "pd",
    "attn_on_pim",
    "weights_on_pim",
    "facil",
    "attacc",
    "ianus",
    "this work",
]

DISPLAY_NAME = {
    "pd": "PD",
    "attn_on_pim": "Attn-on-PIM",
    "weights_on_pim": "Weights-on-PIM",
    "facil": "FACIL",
    "attacc": "ATTACC",
    "ianus": "IANUS",
    "this work": "This work",
}

# ========== 4 个 model 的 hatch / speedup 线型 ==========
# [MOD-3] hatch 更“密集”：用更多字符让纹理更密，粗细由 rcParams['hatch.linewidth'] 控制
MODEL_HATCHES = [
    "",            # 纯色
    "....",        # 点状（更密）
    "//////",      # 条纹（更密）
    "xxxxxx",      # 阴影（交叉更密）
]
# 对应线型：实线 / 点线 / 虚线 / 点划线
MODEL_LINESTYLES = [
    "-",
    ":",
    "--",
    "-.",
]


def _canonical_algo(policy: str) -> str:
    name = (policy or "").strip()
    if name.startswith("algo:"):
        name = name.split(":", 1)[1]
    return ALIASES.get(name, name)


def _read_compare(path: Path) -> Dict[str, Tuple[float, float, float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    time_map: Dict[str, Tuple[float, float, float]] = {}
    for r in obj.get("results", []) or []:
        pol = r.get("policy")
        if not pol:
            continue
        time_map[str(pol)] = (
            float(r.get("prefill_time_s", 0.0)),
            float(r.get("decode_time_s", 0.0)),
            float(r.get("total_time_s", 0.0)),
        )
    return time_map


def _extract_policy_vectors(time_map: Dict[str, Tuple[float, float, float]]):
    """Return (prefill[7], decode[7], total[7]) aligned to PREFERRED_ORDER."""
    by_can: Dict[str, Tuple[float, float, float]] = {}
    for pol, (pre, de, tot) in time_map.items():
        can = _canonical_algo(pol)
        if can in EXCLUDE_ALGOS:
            continue
        by_can[can] = (pre, de, tot)

    pre_list: List[float] = []
    de_list: List[float] = []
    tot_list: List[float] = []
    for can in PREFERRED_ORDER:
        if can in by_can:
            p, d, t = by_can[can]
        else:
            p, d, t = (np.nan, np.nan, np.nan)
        pre_list.append(p)
        de_list.append(d)
        tot_list.append(t)

    return (
        np.array(pre_list, dtype=float),
        np.array(de_list, dtype=float),
        np.array(tot_list, dtype=float),
    )


def _default_model_specs():
    """Experiment-1 固定配置（你给的 batch）"""
    return [
        ("qwen-1.8b b1", "qwen_1.8b_int8_b1_s64"),
        ("qwen-7b b8", "qwen_7b_int8_b8_s64"),
        ("palm-62b b32", "palm_62b_int8_b32_s64"),
        ("mixtral-8x7b b8", "mixtral_8x7b_int8_b8_s64"),
    ]


def plot_experiment1(
    models: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    out_path: Path,
    show_speedup: bool = True,
    logy: bool = True,
    figsize: Tuple[float, float] = (10.6, 8),
    group_width: float = 0.7,
    dpi: int = 300,
) -> None:
    """models: list of (label, prefill[7], decode[7], total[7])"""

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 9,   # [MOD-4] legend 略放大
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.5,   # [MOD-3] hatch 线宽 = 柱子边框线宽（你柱子 linewidth=0.5）
    })

    n_models = len(models)
    n_policies = len(PREFERRED_ORDER)
    group_spacing = 0.85
    x = np.arange(n_policies) * group_spacing

    # [MOD-2] 上方 speedup subplot kw调整比例
    fig, (ax_sp, ax_lat) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 3.4], "hspace": 0.1},
    )

    # 每个 policy 内柱子更宽
    bar_w = group_width / max(1, n_models)
    offsets = (-group_width / 2) + (np.arange(n_models) + 0.5) * bar_w

    # ========== latency 柱状图（下图） ==========
    eps = 1e-4  # log 轴下避免 0
    all_positive_tot = []
    for _, _pre, _de, tot in models:
        all_positive_tot.extend([v for v in tot.tolist() if np.isfinite(v) and v > 0])

    for i, (label, pre, de, tot) in enumerate(models):
        hatch = MODEL_HATCHES[i % len(MODEL_HATCHES)]

        pre_v = pre.copy()
        de_v = de.copy()

        if logy:
            pre_v = np.where(np.isfinite(pre_v) & (pre_v > 0), pre_v, eps)
            de_v = np.where(np.isfinite(de_v) & (de_v > 0), de_v, eps)
        else:
            pre_v = np.where(np.isfinite(pre_v), pre_v, np.nan)
            de_v = np.where(np.isfinite(de_v), de_v, np.nan)

        xpos = x + offsets[i]

        ax_lat.bar(
            xpos,
            pre_v,
            width=bar_w * 0.98,
            color=COLOR_PREFILL,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            zorder=2,
        )
        ax_lat.bar(
            xpos,
            de_v,
            width=bar_w * 0.98,
            bottom=pre_v,
            color=COLOR_DECODE,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            zorder=2,
        )

    ax_lat.set_ylabel("Latency (s)", fontweight="bold")
    ax_lat.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.9, zorder=0)

    if logy:
        ax_lat.set_yscale("log")
        if all_positive_tot:
            ymax = max(all_positive_tot)
            ymin = min([v for v in all_positive_tot if v > 0])
            ax_lat.set_ylim(bottom=max(eps, ymin * 0.5), top=ymax * 1.35)

    # ========== speedup 折线（上图） ==========
    if show_speedup:
        sp_all: List[float] = []

        for i, (label, _pre, _de, tot) in enumerate(models):
            ls = MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]
            xpos = x + offsets[i]

            pd_tot = tot[0]  # PREFERRED_ORDER[0] == 'pd'
            sp = np.full_like(tot, np.nan, dtype=float)
            if np.isfinite(pd_tot) and pd_tot > 0:
                sp = pd_tot / tot

            sp_all.extend([v for v in sp.tolist() if np.isfinite(v)])

            ax_sp.plot(
                xpos,
                sp,
                color="black",
                linestyle=ls,
                linewidth=1.0,
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.2,
                zorder=3,
            )

        ax_sp.axhline(1.0, linestyle="--", linewidth=1.0, color="black", alpha=0.55, zorder=1)

        # [MOD-1] speedup y-label 移到右边；左边留给 latency
        ax_sp.set_ylabel("Speedup(×)", fontweight="bold")
        ax_sp.yaxis.set_label_position("right")
        ax_sp.yaxis.tick_right()
        ax_sp.yaxis.set_label_coords(1.035, -0.05) 
        ax_sp.tick_params(axis="y", which="both",
                        left=False, labelleft=False,   # 左侧刻度/label 关闭
                        right=True, labelright=True)   # 右侧刻度/label 打开
        ax_sp.spines["left"].set_visible(True)

        ax_sp.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.9, zorder=0)
        ax_sp.tick_params(axis="x", labelbottom=False)

        # [MOD-2] speedup y 轴范围：给 min/max 加 padding，确保 <1 的点不会被裁掉
        if sp_all:
            lo = float(np.min(sp_all))
            hi = float(np.max(sp_all))
            if hi == lo:
                pad = 0.5 if hi <= 1.0 else 0.1 * hi
            else:
                pad = 0.5 * (hi - lo)
            bottom = -1
            top = hi + pad
            ax_sp.set_ylim(bottom=bottom, top=top)

    # ========== x 轴 policy 标签 ==========
    xticklabels = [DISPLAY_NAME.get(p, p) for p in PREFERRED_ORDER]
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(xticklabels, rotation=10, ha="right", fontsize = 10,fontweight="bold")

    # [MOD-5] 收紧 x 轴两侧空白：精确设置 xlim（sharex=True 上下两幅同时生效）
    # 计算最左/最右柱子的实际边界（考虑 offsets + 真实柱宽 bar_w*0.98）
    left_edge = (x[0] + float(np.min(offsets))) - (bar_w * 0.98) / 2
    right_edge = (x[-1] + float(np.max(offsets))) + (bar_w * 0.98) / 2
    pad = bar_w * 0.05  # 想更贴边就调小，比如 0.05；想更松就调大
    ax_lat.set_xlim(left_edge - 0.5*pad, right_edge + pad)

    # ========== 一排 legend，放在 policy 标签下方 ==========
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    handles: List[object] = []
    labels: List[str] = []

    handles.append(mpatches.Patch(facecolor=COLOR_PREFILL, edgecolor="black", label="Prefill", linewidth=0.8))
    labels.append("Prefill")
    handles.append(mpatches.Patch(facecolor=COLOR_DECODE, edgecolor="black", label="Decode", linewidth=0.8))
    labels.append("Decode")

    # [MOD-4] legend 里的 model 句柄做大：patch 更大 + line marker 更大
    for i, (lab, _pre, _de, _tot) in enumerate(models):
        hatch = MODEL_HATCHES[i % len(MODEL_HATCHES)]
        ls = MODEL_LINESTYLES[i % len(MODEL_LINESTYLES)]

        patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch=hatch, linewidth=0.8)
        line = Line2D(
            [0],
            [0],
            color="black",
            linestyle=ls,
            marker="o",
            markersize=6.2,        # bigger marker in legend
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            linewidth=1.0,         # thicker line in legend
        )
        handles.append((patch, line))
        labels.append(lab)

    ncol = max(1, len(handles))

    # [MOD-4] legend 往上挪：靠近 x label，同时把 handle 做大以便看清 hatch
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),  # 往上挪（原来 0.01 太靠下）
        ncol=ncol,
        frameon=True,
        columnspacing=1.0,
        handletextpad=0.6,
        borderaxespad=0.2,
        handlelength=2.5,            # 让 patch/line 更长，更容易看清纹理
        handleheight=1.7,            # 让 patch 更高
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    # [MOD-4] 同步减少 bottom 留白，让 legend 紧贴 x label
    fig.subplots_adjust(bottom=0.28)
    

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Path to .../output/experiment_npu/hw_npu_2aim/st64 . If set, will auto-load default 4 models.",
    )
    ap.add_argument("--prefill", type=int, default=128)
    ap.add_argument("--decode", type=int, default=1024)
    ap.add_argument("--out", type=str, required=True, help="Output path (png/pdf).")
    ap.add_argument("--no-speedup", action="store_true", help="Disable speedup subplot.")
    ap.add_argument("--linear", action="store_true", help="Use linear y-scale for latency (default: log).")
    ap.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(8.6, 3),
        metavar=("W", "H"),
        help="Figure size in inches, e.g. --figsize 9.6 2.6",
    )
    ap.add_argument(
        "--group-width",
        type=float,
        default=0.8,
        help="Total width for each policy group (larger -> thicker bars).",
    )
    ap.add_argument(
        "--json",
        action="append",
        default=[],
        help="(Optional) Provide a baseline_compare_*.json file directly (can repeat).",
    )
    ap.add_argument(
        "--label",
        action="append",
        default=[],
        help="(Optional) Label for each --json (same count).",
    )
    args = ap.parse_args()

    compare_name = f"baseline_compare_{args.prefill}x{args.decode}.json"
    models: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    if args.json:
        if args.label and len(args.label) != len(args.json):
            raise SystemExit("--label count must match --json count (or omit --label).")

        for i, jp in enumerate(args.json):
            p = Path(jp)
            label = args.label[i] if args.label else p.parent.name
            tm = _read_compare(p)
            pre, de, tot = _extract_policy_vectors(tm)
            models.append((label, pre, de, tot))

    else:
        if not args.output_root:
            raise SystemExit("Either provide --output-root or at least one --json.")

        root = Path(args.output_root)
        for label, model_dir in _default_model_specs():
            p = root / model_dir / compare_name
            if not p.exists():
                continue
            tm = _read_compare(p)
            pre, de, tot = _extract_policy_vectors(tm)
            models.append((label, pre, de, tot))

        if not models:
            raise SystemExit(f"No compare files found. Expected: {root}/<model_dir>/{compare_name}")

    plot_experiment1(
        models,
        out_path=Path(args.out),
        show_speedup=(not args.no_speedup),
        logy=(not args.linear),
        figsize=(float(args.figsize[0]), float(args.figsize[1])),
        group_width=float(args.group_width),
    )


if __name__ == "__main__":
    main()

