#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

1) 批量：
python plot_exp1_simulated.py \
  --output-root ../../algorithms/output/exp1\
  --out-dir ../../figs/speedup/exp1


不要传hw_ 这一层目录

2) 指定单个模型目录（目录里有 baseline_compare_*.json）：
   python analyze_speedup_comparison.py --model-dir ../algorithms/output/evaluate_single_test/hardware_config_scale_down_11pima/llama_7b_int8_b1_s64

3) 只画某个目录，并把 PDF 输出到指定目录：
   python speedup.py --model-dir ... --out-dir ./figs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

COL_PREFILL = "#326568"
COL_DECODE = "#A2D091"
COL_SPEEDUP = "#000000"

# ===== 算法排序（按你的要求）=====
# 顺序：pd, attn_on_pim, weights_on_pim, facil, attacc, ianus, hefthint（heft 与 hefthint 会二选一画成 hefthint）
PREFERRED_ORDER = [
    "pd",
    "attn_on_pim",
    "weights_on_pim",
    "facil",
    "attacc",
    "ianus",
    "this work",
]

# 用户可能写错的别名（做兼容）
ALIASES = {
    "atten_on_pim": "attn_on_pim",
    "attn_on_pim": "attn_on_pim",
    "weight_on_pim": "weights_on_pim",
    "weights_on_pim": "weights_on_pim",
    "hefthint": "this work",  # backward compatibility
}

# 要去掉的算法（注意：heft 不单独画；会与 hefthint 取总时长更小者，统一画成 hefthint）
EXCLUDE_ALGOS = {"heft"}


def _canonical_algo(policy: str) -> str:
    """policy 可能是 'algo:xxx'；这里统一成 'xxx' 并做别名映射。"""
    name = policy or ""
    if name.startswith("algo:"):
        name = name.split(":", 1)[1]
    return ALIASES.get(name, name)


def _order_policies(policies: Sequence[str]) -> List[str]:
    """按指定顺序排序；未在列表内的保持原出现顺序，heft（如有）强制放最后。"""
    # 过滤掉 neupims
    filtered = [p for p in policies if _canonical_algo(p) not in EXCLUDE_ALGOS]

    # 去重（按出现顺序保留）
    seen = set()
    uniq: List[str] = []
    for p in filtered:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    filtered = uniq

    # canonical -> policy（只取第一次出现的）
    by_name: Dict[str, str] = {}
    for p in filtered:
        by_name.setdefault(_canonical_algo(p), p)

    ordered: List[str] = []
    for nm in PREFERRED_ORDER:
        if nm in by_name and by_name[nm] not in ordered:
            ordered.append(by_name[nm])

    remaining = [p for p in filtered if p not in ordered]

    return ordered + remaining


def _parse_case_from_config_or_name(cfg: Dict[str, object], path: Path) -> Tuple[int, int]:
    """返回 (prefill_len, decode_len)；优先 config，其次从文件名 baseline_compare_128x1024.json 解析。"""
    S = cfg.get("prefill_len")
    T = cfg.get("decode_len")
    if S is None or T is None:
        m = re.search(r"(\d+)x(\d+)", path.stem)
        if m:
            S, T = int(m.group(1)), int(m.group(2))
        else:
            S, T = -1, -1
    return int(S), int(T)


def _load_baseline_compare(
    path: Path,
) -> Tuple[Tuple[int, int], Dict[str, Tuple[float, float, float]], Dict[str, object]]:
    """
    读取 baseline_compare_*.json
    返回：
      case = (S, T)
      time_map: policy -> (prefill_s, decode_s, total_s)
      cfg: config dict
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    cfg = obj.get("config", {}) or {}
    case = _parse_case_from_config_or_name(cfg, path)

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
    return case, time_map, cfg




def _strip_algo_prefix(policy: str) -> str:
    """把 'algo:xxx' 统一成 'xxx'。"""
    if policy.startswith("algo:"):
        return policy.split(":", 1)[1]
    return policy


def _effective_total_s(t: Tuple[float, float, float]) -> float:
    """total_time_s 可能为 0；这种情况下用 prefill+decode 作为 total。"""
    pre, de, tot = t
    tot = float(tot)
    if tot > 0:
        return tot
    return float(pre) + float(de)


def _choose_heft_or_hefthint_for_plot(
    time_map: Dict[str, Tuple[float, float, float]],
    *,
    case: Tuple[int, int],
    fp: Optional[Path] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """在 time_map 里，如果同时存在 heft 与 hefthint，则选总时长更小者来画。

    - 图上统一显示成 'hefthint'（不单独画 'heft'）。
    - 在日志里打印本次实际采用的是哪一个（heft / hefthint）。

    返回新的 time_map（不会修改原 dict）。
    """

    # 找出所有 policy 中 base name 为 'heft' / 'hefthint' 的条目
    # 兼容：有些旧数据可能用 'this work' 作为 policy 名
    hint_names = {"hefthint", "this work"}

    heft_keys = [k for k in time_map.keys() if _strip_algo_prefix(k) == "heft"]
    hint_keys = [k for k in time_map.keys() if _strip_algo_prefix(k) in hint_names]

    if not heft_keys and not hint_keys:
        return time_map

    def best_of(keys: List[str]) -> Tuple[Optional[str], Optional[float]]:
        best_k: Optional[str] = None
        best_t: Optional[float] = None
        for k in keys:
            t = _effective_total_s(time_map[k])
            if best_t is None or t < best_t:
                best_t = t
                best_k = k
        return best_k, best_t

    best_heft_k, best_heft_t = best_of(heft_keys) if heft_keys else (None, None)
    best_hint_k, best_hint_t = best_of(hint_keys) if hint_keys else (None, None)

    chosen_k: Optional[str] = None
    chosen_from: str = ""

    if best_heft_k is not None and best_hint_k is not None:
        # 总时长更小者胜；若相等，优先保留 hefthint（更符合命名）
        assert best_heft_t is not None and best_hint_t is not None
        if best_heft_t < best_hint_t:
            chosen_k = best_heft_k
            chosen_from = "heft"
        else:
            chosen_k = best_hint_k
            chosen_from = "hefthint"
    elif best_heft_k is not None:
        chosen_k = best_heft_k
        chosen_from = "heft"
    else:
        chosen_k = best_hint_k
        chosen_from = "hefthint"

    # 新 map：移除所有 heft/hefthint 原始条目，统一塞回 'algo:hefthint'
    new_map: Dict[str, Tuple[float, float, float]] = dict(time_map)
    for k in heft_keys + hint_keys:
        new_map.pop(k, None)

    if chosen_k is not None:
        new_map["algo:hefthint"] = time_map[chosen_k]

    # 日志
    S, T = case
    tag = fp.name if fp is not None else "<unknown file>"

    if best_heft_k is not None and best_hint_k is not None:
        assert best_heft_t is not None and best_hint_t is not None
        print(
            f"[SELECT] {tag} (prefill={S}, decode={T}): plot 'hefthint' using {chosen_from} "
            f"(heft={best_heft_t:.6g}s, hefthint={best_hint_t:.6g}s)"
        )
    else:
        only = "heft" if best_heft_k is not None else "hefthint"
        tot_only = best_heft_t if best_heft_t is not None else best_hint_t
        if tot_only is not None:
            print(
                f"[SELECT] {tag} (prefill={S}, decode={T}): only {only} present; plot as 'hefthint' "
                f"(total={tot_only:.6g}s)"
            )
        else:
            print(
                f"[SELECT] {tag} (prefill={S}, decode={T}): only {only} present; plot as 'hefthint'"
            )

    return new_map


def _safe_filename(s: str, max_len: int = 240) -> str:
    """文件名安全化（尽量保留 '__' 作为路径分隔符风格）。"""
    s = s.replace(os.sep, "__")
    s = re.sub(r"\s+", "_", s)
    # 允许字母数字、下划线、点、等号、加减号
    s = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", s).strip("_")
    if len(s) > max_len:
        keep = max_len - 10
        s = s[: keep // 2] + "__TRUNC__" + s[-keep // 2 :]
    return s


def _plot_one_case(
    ax: plt.Axes,
    time_map: Dict[str, Tuple[float, float, float]],
    *,
    title: str,
):
    """
    单个子图：
      - 左轴：stacked latency bar（prefill + decode）
      - 右轴：total speedup vs pd 折线 + 数值
    """
    ordered = _order_policies(list(time_map.keys()))
    if not ordered:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return None, None, None, None

    labels = [_canonical_algo(p) for p in ordered]
    pre = np.array([time_map[p][0] for p in ordered], dtype=float)
    de = np.array([time_map[p][1] for p in ordered], dtype=float)
    tot = np.array([time_map[p][2] for p in ordered], dtype=float)
    tot = np.where(tot > 0, tot, pre + de)

    x = np.arange(len(ordered))

    # stacked bars（一根柱）
    bars_prefill = ax.bar(
        x,
        pre,
        label="Prefill",
        color=COL_PREFILL,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )
    bars_decode = ax.bar(
        x,
        de,
        bottom=pre,
        label="Decode",
        color=COL_DECODE,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )

    # 轴与样式
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle=":", linewidth=1.0, alpha=0.85, zorder=0)

    # 折线：speedup vs pd（total）
    pd_total = None
    for p, t in zip(ordered, tot):
        if _canonical_algo(p) == "pd":
            pd_total = float(t)
            break

    line = None
    ax2 = None
    if pd_total and pd_total > 0:
        speedup = pd_total / tot

        ax2 = ax.twinx()
        ax2.patch.set_visible(False)
        ax2.set_zorder(ax.get_zorder() + 1)

        (line,) = ax2.plot(
            x,
            speedup,
            color=COL_SPEEDUP,
            marker="o",
            linewidth=2.0,
            zorder=5,
        )

        finite = speedup[np.isfinite(speedup)]
        if finite.size:
            su_min = float(np.min(finite))
            su_max = float(np.max(finite))
            rng = su_max - su_min
            offset = 0.02 * rng if rng > 0 else 0.05 * su_max

            top = su_max * 1.15 if su_max > 0 else 1.0
            ax2.set_ylim(0.0, top)

            for xi, su in zip(x, speedup):
                if not np.isfinite(su):
                    continue
                ax2.text(
                    xi,
                    su + offset,
                    f"{su:.2f}×",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=COL_SPEEDUP,
                )

    return bars_prefill, bars_decode, line, ax2


def plot_compare_grid(files: Sequence[Path], *, outfile: Path, sharey: bool = False) -> None:
    """把一个 model_dir 下的 baseline_compare_*.json 画成多行子图（每行最多 4 个）并保存。"""
    loaded = []
    for fp in files:
        case, tm, _ = _load_baseline_compare(fp)
        tm = _choose_heft_or_hefthint_for_plot(tm, case=case, fp=fp)
        loaded.append((case, fp, tm))

    loaded.sort(key=lambda x: (x[0][0], x[0][1]))
    n = len(loaded)
    if n == 0:
        raise ValueError("No baseline_compare files to plot")

    cols = 4
    rows = math.ceil(n / cols)

    fig_w = max(6.0, 5.2 * min(n, cols))
    fig_h = max(4.2, 4.6 * rows)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_w, fig_h),
        sharey=sharey,
        squeeze=False,
        gridspec_kw={"wspace": 0.2, "hspace": 0.35},
    )
    axes_flat = axes.ravel()

    legend_handles = None
    legend_labels = None

    for i, (case, _fp, tm) in enumerate(loaded):
        ax = axes_flat[i]
        row_idx, col_idx = divmod(i, cols)

        S, T = case
        title = f"prefill={S}, decode={T}"
        bp, bd, line, ax2 = _plot_one_case(ax, tm, title=title)

        # 左轴：每行只在第 1 列保留刻度和标签
        if col_idx == 0:
            ax.set_ylabel("Latency (s)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

        # 右轴：每行只在最右列（或最后一个子图）保留刻度和标签
        if ax2 is not None:
            is_row_last = (col_idx == cols - 1)
            is_global_last = (i == n - 1)
            if is_row_last or is_global_last:
                ax2.set_ylabel("Speedup vs pd")
            else:
                ax2.set_ylabel("")
                ax2.tick_params(axis="y", which="both", right=False, labelright=False)
                ax2.spines["right"].set_visible(False)

        if legend_handles is None and bp is not None and bd is not None:
            handles = [bp, bd]
            labels = ["Prefill", "Decode"]
            if line is not None:
                handles.append(line)
                labels.append("Speedup vs pd")
            legend_handles = handles
            legend_labels = labels

    # 关掉空白子图
    for j in range(n, rows * cols):
        axes_flat[j].axis("off")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.subplots_adjust(right=0.99, left=0.06, bottom=0.18, top=0.88, wspace=0.2, hspace=0.3)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=220, bbox_inches="tight")
    plt.close(fig)


def iter_model_dirs(output_root: Path) -> Iterator[Tuple[str, str, Path]]:
    """模仿 batch_speedup.py：遍历 hw_* / sst* / <model_dir>"""
    output_root = output_root.resolve()
    for hw_dir in sorted([p for p in output_root.glob("hw_*") if p.is_dir()]):
        for stride_dir in sorted([p for p in hw_dir.glob("sst*") if p.is_dir()]):
            for model_dir in sorted([p for p in stride_dir.iterdir() if p.is_dir()]):
                yield hw_dir.name, stride_dir.name, model_dir


def collect_compare_files(model_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        files = sorted(model_dir.rglob(pattern))
    else:
        files = sorted(model_dir.glob(pattern))
        if not files:
            # 兼容：有些结果可能把 compare 放在更深层
            files = sorted(model_dir.rglob(pattern))
    return files


def build_outfile_name(
    hw: str,
    stride: str,
    model_dir_name: str,
    cases: Sequence[Tuple[int, int]],
    *,
    ext: str = "pdf",
) -> str:
    """PDF 文件名里包含：硬件/stride/模型目录名 + 全部 (SxT) case。"""
    case_tag = "__".join([f"{S}x{T}" for (S, T) in cases])
    base = f"{hw}__{stride}__{model_dir_name}__{case_tag}"
    return f"{_safe_filename(base)}.{ext}"


def process_one_model_dir(
    *,
    hw: str,
    stride: str,
    model_dir: Path,
    pattern: str,
    recursive: bool,
    sharey: bool,
    out_dir: Optional[Path],
    dry_run: bool,
) -> Optional[Path]:
    files = collect_compare_files(model_dir, pattern=pattern, recursive=recursive)
    if not files:
        return None

    # cases 用于命名
    cases: List[Tuple[int, int]] = []
    for fp in files:
        try:
            case, _, _ = _load_baseline_compare(fp)
        except Exception:
            case = _parse_case_from_config_or_name({}, fp)
        cases.append(case)
    cases_sorted = sorted(set(cases), key=lambda x: (x[0], x[1]))

    out_name = build_outfile_name(hw, stride, model_dir.name, cases_sorted, ext="pdf")
    out_path = ((out_dir or model_dir) / out_name).resolve()

    if dry_run:
        print(f"[DRY-RUN] {model_dir} -> {out_path.name}")
        return out_path

    plot_compare_grid(files, outfile=out_path, sharey=sharey)
    print(f"[OK] {model_dir} -> {out_path.name}")
    return out_path


def _infer_hw_stride_from_model_dir(model_dir: Path) -> Tuple[str, str]:
    """如果用户只给 model_dir，不给 output_root，就从父目录猜 hw / stride。"""
    stride = model_dir.parent.name if model_dir.parent else "st?"
    hw = model_dir.parent.parent.name if model_dir.parent and model_dir.parent.parent else "hw?"
    return hw, stride


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=str, default=None, help="批量扫描根目录（包含 hw_* / st* / model_dir）")
    ap.add_argument("--model-dir", type=str, default=None, help="只处理单个 model_dir")
    ap.add_argument("--pattern", type=str, default="baseline_compare_*.json", help="compare json 匹配模式")
    ap.add_argument("--recursive", action="store_true", help="在 model_dir 下递归查找 compare json")
    ap.add_argument("--sharey", action="store_true", help="所有子图共享左侧 Y 轴（不建议用于跨度很大的 case）")
    ap.add_argument("--out-dir", type=str, default=None, help="可选：把 PDF 输出到该目录（默认输出到各自 model_dir）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要生成的文件，不实际绘图")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None

    # 默认：如果都不传，就把当前目录当作 model_dir
    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        hw, stride = _infer_hw_stride_from_model_dir(model_dir)
        process_one_model_dir(
            hw=hw,
            stride=stride,
            model_dir=model_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            sharey=args.sharey,
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
        return

    output_root = Path(args.output_root).resolve() if args.output_root else Path.cwd().resolve()

    found_any = False
    for hw, stride, model_dir in iter_model_dirs(output_root):
        out = process_one_model_dir(
            hw=hw,
            stride=stride,
            model_dir=model_dir,
            pattern=args.pattern,
            recursive=args.recursive,
            sharey=args.sharey,
            out_dir=out_dir,
            dry_run=args.dry_run,
        )
        if out is not None:
            found_any = True

    if not found_any:
        print(f"[WARN] No '{args.pattern}' found under: {output_root}")


if __name__ == "__main__":
    main()
