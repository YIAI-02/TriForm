#!/usr/bin/env python3
"""
python plot_exp3_iter_from_txt_json.py \
  ../../algorithms/output/evaluate_single_test/hardware_1npu_2aim_test4/llama_7b_fp16_b1_s8/driver_debug_simple.txt \
  ../../algorithms/output/evaluate_single_test/hardware_1npu_2aim_test4/llama_7b_fp16_b1_s8/all_passes_512x128_st8.json \
  -o ../../figs/exp3/exp3_iter/llama_7b_fp16_b1_s8_512x128.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch


LINE_COLORS = ["#000000"]
PIE_COLORS = {
    "ND": "#e4adb5",
    "NZ": "#bdade4",
    "PIM-OPT": "#aee4ad",
}
GRID_COLOR = "#d9d9e3"
FONT_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


@dataclass
class StepState:
    outer_pass: int
    accepted_step: int
    x_label: str
    total_makespan_s: float
    ratios: Dict[str, float]
    note: str


@dataclass
class ALRun:
    run_index: int
    raw_text: str
    outer0_total: float
    block_layer_span: Optional[int]
    weights: Optional[int]
    blocks: Optional[int]


def configure_fonts() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    selected = next((n for n in FONT_CANDIDATES if n in available), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = selected
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="根据 [AL] txt 日志 + all_passes json 重建 exp3 iter 图"
    )
    p.add_argument("txt", help="driver_debug_simple.txt 或完整 driver_debug.txt")
    p.add_argument("json", help="all_passes_*.json")
    p.add_argument("-o", "--output", default="exp3_iter.pdf", help="输出图文件（pdf/png 都可）")
    p.add_argument("--run-index", type=int, default=None, help="txt 中若有多个 [AL] run，可手动指定使用第几个（从 0 开始）")
    p.add_argument("--lang", choices=["zh", "en"], default="en")
    p.add_argument("--title", default=None)
    p.add_argument("--pie-zoom", type=float, default=0.18)
    return p.parse_args(argv)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_al_runs(txt: str) -> List[ALRun]:
    matches = list(re.finditer(r"\[AL\] init:", txt))
    if not matches:
        raise ValueError("在 txt 中没有找到任何 [AL] init 段。")
    runs: List[ALRun] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(txt)
        seg = txt[start:end]

        init_m = re.search(
            r"\[AL\] init: .*?weights=(\d+).*?blocks=(\d+).*?block_layer_span=(\d+)",
            seg,
        )
        outer0_m = re.search(r"\[AL\] outer0: done total=([0-9.]+)s", seg)
        if not outer0_m:
            continue

        runs.append(
            ALRun(
                run_index=idx,
                raw_text=seg,
                outer0_total=float(outer0_m.group(1)),
                block_layer_span=int(init_m.group(3)) if init_m else None,
                weights=int(init_m.group(1)) if init_m else None,
                blocks=int(init_m.group(2)) if init_m else None,
            )
        )
    if not runs:
        raise ValueError("找到了 [AL] init，但没有找到 outer0: done。")
    return runs


def choose_run(runs: List[ALRun], json_data: Dict, forced_run_index: Optional[int]) -> ALRun:
    if forced_run_index is not None:
        for run in runs:
            if run.run_index == forced_run_index:
                return run
        raise ValueError(f"--run-index={forced_run_index} 超出范围，可选 run_index 为 {[r.run_index for r in runs]}")

    passes = json_data.get("passes", [])
    if not passes:
        raise ValueError("json 里没有 passes。")
    json_outer0 = float(passes[0]["times"]["total"])
    json_pass_count = len(passes)

    # 自动匹配：优先 outer0 total 最接近，同时 outer 层数更接近。
    scored: List[Tuple[float, float, int, ALRun]] = []
    for run in runs:
        max_outer_in_txt = 0
        for m in re.finditer(r"\[AL\] outer(\d+): after inner total=", run.raw_text):
            max_outer_in_txt = max(max_outer_in_txt, int(m.group(1)))
        pass_count_gap = abs(max_outer_in_txt + 1 - json_pass_count)
        total_gap = abs(run.outer0_total - json_outer0)
        scored.append((total_gap, float(pass_count_gap), run.run_index, run))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0][3]


def parse_log_run(run_text: str) -> Dict:
    out: Dict = {}
    out0 = re.search(
        r"\[AL\] outer0: done total=([0-9.]+)s prefill=([0-9.]+)s decode=([0-9.]+)s",
        run_text,
    )
    if not out0:
        raise ValueError("选中的 [AL] run 中没有 outer0: done。")
    out["outer0"] = {
        "total": float(out0.group(1)),
        "prefill": float(out0.group(2)),
        "decode": float(out0.group(3)),
    }

    init_assign = re.search(
        r"\[AL\] outer0->outer1: initial assign explicit_weights=(\d+) "
        r"\(NZ=(\d+), PIM-OPT=(\d+), ND_default=(\d+)\)",
        run_text,
    )
    if init_assign:
        out["initial_assign"] = {
            "explicit_weights": int(init_assign.group(1)),
            "NZ": int(init_assign.group(2)),
            "PIM-OPT": int(init_assign.group(3)),
            "ND": int(init_assign.group(4)),
        }
    else:
        out["initial_assign"] = None

    baselines: Dict[int, Dict[str, float]] = {}
    for outer, total, prefill, decode in re.findall(
        r"\[AL\] outer(\d+): baseline total=([0-9.]+)s prefill=([0-9.]+)s decode=([0-9.]+)s",
        run_text,
    ):
        baselines[int(outer)] = {
            "total": float(total),
            "prefill": float(prefill),
            "decode": float(decode),
        }
    out["baselines"] = baselines

    accepts: Dict[int, List[Dict]] = defaultdict(list)
    for outer, block, src, dst, old_total, new_total in re.findall(
        r"\[AL\] inner(\d+): ACCEPT block=([A-Z0-9\-_.]+) ([A-Z0-9\-]+)->([A-Z0-9\-]+) "
        r"total ([0-9.]+)s -> ([0-9.]+)s",
        run_text,
    ):
        accepts[int(outer)].append(
            {
                "block": block,
                "src": src,
                "dst": dst,
                "old_total": float(old_total),
                "new_total": float(new_total),
            }
        )
    out["accepts"] = dict(accepts)
    return out


def block_to_weight_keys(block: str, valid_weight_keys: set[str]) -> List[str]:
    m = re.fullmatch(r"L(\d{4})-(\d{4})_(W[0-9A-Z]+)", block)
    if not m:
        raise ValueError(f"无法解析 block 名称: {block}")
    lo, hi, kind = m.groups()
    keys: List[str] = []
    for layer in range(int(lo), int(hi) + 1):
        for shard in (0, 1):
            key = f"L{layer}_{kind}_s{shard}"
            if key not in valid_weight_keys:
                raise KeyError(f"block={block} 映射出的 key 不在 json.weight_sizes 中: {key}")
            keys.append(key)
    return keys


def ratios_from_map(fmt_map: Dict[str, str], weight_sizes: Dict[str, int]) -> Dict[str, float]:
    total_bytes = float(sum(weight_sizes.values()))
    bytes_per_fmt: Dict[str, float] = {"ND": 0.0, "NZ": 0.0, "PIM-OPT": 0.0}
    explicit_total = 0.0
    for key, fmt in fmt_map.items():
        size = float(weight_sizes[key])
        explicit_total += size
        if fmt not in bytes_per_fmt:
            bytes_per_fmt[fmt] = 0.0
        bytes_per_fmt[fmt] += size
    bytes_per_fmt["ND"] += max(0.0, total_bytes - explicit_total)
    return {k: v / total_bytes for k, v in bytes_per_fmt.items()}


def reconstruct_states_from_txt_json(run_text: str, json_data: Dict) -> List[StepState]:
    passes = json_data.get("passes", [])
    if not passes:
        raise ValueError("json 里没有 passes。")

    parsed = parse_log_run(run_text)
    weight_sizes: Dict[str, int] = passes[0]["weights"]["weight_sizes"]
    valid_keys = set(weight_sizes.keys())

    # final maps: pass i -> after_inner map
    final_maps: Dict[int, Dict[str, str]] = {}
    final_times: Dict[int, float] = {}
    for p in passes:
        outer = int(p["pass"])
        final_maps[outer] = dict(p.get("formats", {}) or {})
        final_times[outer] = float(p["times"]["total"])

    states: List[StepState] = []

    # outer0 / step0
    states.append(
        StepState(
            outer_pass=0,
            accepted_step=0,
            x_label="outer0/step0",
            total_makespan_s=float(passes[0]["times"]["total"]),
            ratios=ratios_from_map({}, weight_sizes),
            note=str(passes[0].get("note", "outer0_all_nd")),
        )
    )

    accepts_by_outer: Dict[int, List[Dict]] = parsed["accepts"]
    baselines: Dict[int, Dict[str, float]] = parsed["baselines"]

    max_outer = max(final_maps.keys()) if final_maps else 0
    for outer in range(1, max_outer + 1):
        if outer not in final_maps:
            raise ValueError(f"json 里缺少 outer{outer} 的 final map。")
        if outer not in baselines:
            raise ValueError(f"txt 里缺少 outer{outer} 的 baseline total。")

        # reverse accepts: final(after_inner) -> baseline(step0)
        baseline_map = dict(final_maps[outer])
        for rec in reversed(accepts_by_outer.get(outer, [])):
            for key in block_to_weight_keys(rec["block"], valid_keys):
                cur_fmt = baseline_map.get(key, "ND")
                if cur_fmt != rec["dst"]:
                    raise ValueError(
                        f"反推 outer{outer} baseline 时格式不一致: key={key}, "
                        f"当前={cur_fmt}, 预期终态={rec['dst']}"
                    )
                if rec["src"] == "ND":
                    baseline_map.pop(key, None)
                else:
                    baseline_map[key] = rec["src"]

        # baseline step0
        states.append(
            StepState(
                outer_pass=outer,
                accepted_step=0,
                x_label=f"outer{outer}/step0",
                total_makespan_s=float(baselines[outer]["total"]),
                ratios=ratios_from_map(baseline_map, weight_sizes),
                note=f"outer{outer}_baseline",
            )
        )

        # replay accepts: step1..N
        cur_map = dict(baseline_map)
        accepts = accepts_by_outer.get(outer, [])
        for i, rec in enumerate(accepts, start=1):
            for key in block_to_weight_keys(rec["block"], valid_keys):
                if rec["dst"] == "ND":
                    cur_map.pop(key, None)
                else:
                    cur_map[key] = rec["dst"]

            total = float(rec["new_total"])
            # 最后一个 step 用 json 的 after_inner 精确 total 覆盖，避免日志只有 6 位小数。
            if i == len(accepts) and outer in final_times:
                total = float(final_times[outer])

            states.append(
                StepState(
                    outer_pass=outer,
                    accepted_step=i,
                    x_label=f"outer{outer}/step{i}",
                    total_makespan_s=total,
                    ratios=ratios_from_map(cur_map, weight_sizes),
                    note=f"{rec['block']}:{rec['src']}->{rec['dst']}",
                )
            )

        # sanity check: 回放之后应与 json 最终 map 一致
        if cur_map != final_maps[outer]:
            raise ValueError(f"outer{outer} 回放 ACCEPT 后得到的 map 与 json 最终格式图不一致。")

    return states


def create_pie_image(values: Sequence[float], colors: Sequence[str], size: int = 180) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(1, 1), dpi=size)
    safe_values = [float(v) for v in values]
    total = sum(safe_values)
    if total <= 0:
        safe_values = [1.0]
        colors = ["#eeeeee"]
    ax.pie(
        safe_values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(linewidth=0.7, edgecolor="white"),
    )
    ax.set(aspect="equal")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return img


@dataclass
class PieSpec:
    x: float
    y: float
    img: np.ndarray
    zoom: float
    point_idx: int


def _candidate_offsets(point_idx: int) -> List[Tuple[float, float]]:
    base_angles = [30, -30, 70, -70, 110, -110, 150, -150, 0, 180, 90, -90, 45, -45, 135, -135]
    shift = point_idx % len(base_angles)
    angles = base_angles[shift:] + base_angles[:shift]
    radii = [26, 38, 50, 62, 74, 86]
    candidates: List[Tuple[float, float]] = []
    for radius in radii:
        for angle in angles:
            rad = math.radians(angle)
            candidates.append((radius * math.cos(rad), radius * math.sin(rad)))
    return candidates


def _boxes_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], margin: float = 2.0) -> Tuple[bool, float]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0 - margin, bx0 - margin)
    iy0 = max(ay0 - margin, by0 - margin)
    ix1 = min(ax1 + margin, bx1 + margin)
    iy1 = min(ay1 + margin, by1 + margin)
    if ix1 <= ix0 or iy1 <= iy0:
        return False, 0.0
    return True, (ix1 - ix0) * (iy1 - iy0)


def place_pies(ax: plt.Axes, pie_specs: Sequence[PieSpec]) -> None:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)

    occupied: List[Tuple[float, float, float, float]] = []

    for spec in pie_specs:
        point_px = ax.transData.transform((spec.x, spec.y))
        h_px = spec.img.shape[0] * spec.zoom
        w_px = spec.img.shape[1] * spec.zoom
        best_offset: Optional[Tuple[float, float]] = None
        best_score = float("inf")

        for off_x, off_y in _candidate_offsets(spec.point_idx):
            cx = point_px[0] + off_x
            cy = point_px[1] + off_y
            box = (cx - w_px / 2.0, cy - h_px / 2.0, cx + w_px / 2.0, cy + h_px / 2.0)

            overlap_area = 0.0
            overlap_count = 0
            for other in occupied:
                hit, area = _boxes_overlap(box, other, margin=2.0)
                if hit:
                    overlap_area += area
                    overlap_count += 1

            overflow = 0.0
            if box[0] < axes_box.x0:
                overflow += axes_box.x0 - box[0]
            if box[2] > axes_box.x1:
                overflow += box[2] - axes_box.x1
            if box[1] < axes_box.y0:
                overflow += axes_box.y0 - box[1]
            if box[3] > axes_box.y1:
                overflow += box[3] - axes_box.y1

            radius = math.hypot(off_x, off_y)
            score = overlap_count * 1e9 + overlap_area * 1e5 + overflow * 1e3 + radius
            if score < best_score:
                best_score = score
                best_offset = (off_x, off_y)

            if overlap_count == 0 and overflow == 0:
                best_offset = (off_x, off_y)
                break

        if best_offset is None:
            best_offset = (40.0, 20.0)

        cx = point_px[0] + best_offset[0]
        cy = point_px[1] + best_offset[1]
        final_box = (cx - w_px / 2.0, cy - h_px / 2.0, cx + w_px / 2.0, cy + h_px / 2.0)
        occupied.append(final_box)

        artist = AnnotationBbox(
            OffsetImage(spec.img, zoom=spec.zoom),
            (spec.x, spec.y),
            xybox=best_offset,
            xycoords="data",
            boxcoords="offset points",
            pad=0,
            frameon=False,
            zorder=6,
        )
        ax.add_artist(artist)


def choose_figsize(num_x: int) -> Tuple[float, float]:
    width = max(6, min(20.0, 3.0 + 0.6 * num_x))
    height = 3.8
    return width, height


def make_text(lang: str) -> Dict[str, str]:
    if lang == "zh":
        return {
            "title": "实验3：搜索优化轨迹（由 txt + json 重建）",
            "xlabel": "Outer pass / accepted step",
            "ylabel": "Total makespan (s)",
            "run_legend": "轨迹",
            "fmt_legend": "格式占比",
            "footer": "小饼图表示该步 ND / NZ / PIM-OPT 的权重字节占比。",
            "series_label": "reconstructed run",
        }
    return {
        "title": "Experiment 3: reconstructed search trajectory",
        "xlabel": "Outer pass / accepted step",
        "ylabel": "Total makespan (s)",
        "run_legend": "Trajectory",
        "fmt_legend": "Format share",
        "footer": "Small pie charts show the ND / NZ / PIM-OPT byte ratio at each accepted step.",
        "series_label": "reconstructed run",
    }


def plot_states(states: Sequence[StepState], output_path: Path, lang: str = "zh", pie_zoom: float = 0.16, title: Optional[str] = None) -> None:
    text = make_text(lang)
    width, height = choose_figsize(len(states))
    fig, ax = plt.subplots(figsize=(width, height))

    xs = list(range(len(states)))
    ys = [s.total_makespan_s for s in states]
    xlabels = [s.x_label.replace("/", " / ") for s in states]

    line_color = LINE_COLORS[0]
    ax.plot(
        xs,
        ys,
        color=line_color,
        linewidth=2.2,
        marker="o",
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        markeredgecolor=line_color,
        zorder=3,
        label=text["series_label"],
    )

    pie_specs: List[PieSpec] = []
    for i, s in enumerate(states):
        values = [s.ratios.get("ND", 0.0), s.ratios.get("NZ", 0.0), s.ratios.get("PIM-OPT", 0.0)]
        colors = [PIE_COLORS["ND"], PIE_COLORS["NZ"], PIE_COLORS["PIM-OPT"]]
        pie_specs.append(
            PieSpec(
                x=float(i),
                y=float(s.total_makespan_s),
                img=create_pie_image(values, colors, size=180),
                zoom=pie_zoom,
                point_idx=i,
            )
        )

    ymin, ymax = min(ys), max(ys)
    spread = ymax - ymin
    pad = max(0.0008, spread * 0.10)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.35, len(states) - 0.4)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=9.5)
    ax.set_xlabel(text["xlabel"], fontsize=11)
    ax.set_ylabel(text["ylabel"], fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=GRID_COLOR, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    place_pies(ax, pie_specs)

    fig.suptitle(title or text["title"], fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.012, 0.02, text["footer"], fontsize=8.8, color="#555555")

    line_handles = [
        Line2D(
            [0], [0],
            color=line_color,
            linewidth=2.2,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            markeredgecolor=line_color,
            label=text["series_label"],
        )
    ]
    legend_runs = ax.legend(
        handles=line_handles,
        title=text["run_legend"],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        title_fontsize=9,
    )
    ax.add_artist(legend_runs)

    fmt_handles = [
        Patch(facecolor=PIE_COLORS["ND"], edgecolor="none", label="ND"),
        Patch(facecolor=PIE_COLORS["NZ"], edgecolor="none", label="NZ"),
        Patch(facecolor=PIE_COLORS["PIM-OPT"], edgecolor="none", label="PIM-OPT"),
    ]
    ax.legend(
        handles=fmt_handles,
        title=text["fmt_legend"],
        loc="upper left",
        bbox_to_anchor=(1.01, 0.58),
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        title_fontsize=9,
    )

    fig.tight_layout(rect=[0.0, 0.05, 0.82, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_fonts()
    args = parse_args(argv)

    txt_path = Path(args.txt).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not txt_path.is_file():
        raise FileNotFoundError(f"txt 不存在: {txt_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"json 不存在: {json_path}")

    txt = txt_path.read_text(encoding="utf-8", errors="ignore")
    json_data = load_json(json_path)

    runs = extract_al_runs(txt)
    chosen = choose_run(runs, json_data, args.run_index)
    states = reconstruct_states_from_txt_json(chosen.raw_text, json_data)
    plot_states(states, out_path, lang=args.lang, pie_zoom=args.pie_zoom, title=args.title)

    print(f"[INFO] txt 中检测到 {len(runs)} 个 [AL] run，选中了 run_index={chosen.run_index}")
    print(f"[INFO] 该 run: outer0_total={chosen.outer0_total:.6f}s block_layer_span={chosen.block_layer_span}")
    print(f"[INFO] 重建出 {len(states)} 个点：")
    for s in states:
        print(
            f"  outer={s.outer_pass} step={s.accepted_step} total={s.total_makespan_s:.9f}s "
            f"ND={s.ratios['ND']:.4f} NZ={s.ratios['NZ']:.4f} PIM={s.ratios['PIM-OPT']:.4f} "
            f"note={s.note}"
        )
    print(f"[INFO] 输出图文件: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
