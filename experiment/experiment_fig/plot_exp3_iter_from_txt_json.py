#!/usr/bin/env python3
"""Plot Weight Layout Arbiter iteration trajectories from ``driver_debug.txt``
and ``all_passes.json``.

Run this script from ``experiment/experiment_fig``. The input files are emitted
by the current ``python src/main.py weight-suggest`` flow and usually live under
one run directory inside the repository-level ``output/`` tree.

Example
-------
python plot_exp3_iter_from_txt_json.py \
  ../../output/ws_hpc/shards/8w/worker_5/runs/example_run/artifacts/llama_7b_fp16_b1_s8/driver_debug.txt \
  ../../output/ws_hpc/shards/8w/worker_5/runs/example_run/all_passes.json \
  -o ../../figs/exp3/exp3_iter/llama_7b_fp16_b1_s8_128x512.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Patch, Wedge
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator


LINE_COLORS = ["#000000"]
PIE_COLORS = {
    "ND": "#d7aeb3",
    "NZ": "#b5abd8",
    "PIM-OPT": "#b7d9ae",
}
LEGEND_LABELS = {
    "ND": "Linear",
    "NZ": "NPU-OPT",
    "PIM-OPT": "PIM-OPT",
}
GRID_COLOR = "#d7d7dc"
PIE_OUTLINE_COLOR = "#000000"
PIE_OUTLINE_WIDTH = 1.0
PIE_SEPARATOR_WIDTH = 1.7
FONT_CANDIDATES = [
    "DejaVu Sans",
    "Arial",
    "Liberation Sans",
    "Helvetica",
]

# In driver_debug_simple.txt, lines usually start with [AL] outer...
# In the full driver_debug.txt, they may appear as [AL][ND] outer... / [AL][NZ] ...
# Allow an optional mode tag right after [AL].
AL_PREFIX = r"\[AL\](?:\[[^\]]+\])?"


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


@dataclass
class PieSpec:
    x: float
    y: float
    box: DrawingArea
    diameter_pt: float
    point_idx: int


def configure_fonts() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    selected = next((n for n in FONT_CANDIDATES if n in available), "DejaVu Sans")
    matplotlib.rcParams["font.family"] = selected
    matplotlib.rcParams["font.sans-serif"] = FONT_CANDIDATES
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild the Experiment 3 iteration figure from an [AL] txt log and all_passes json"
    )
    p.add_argument("txt", help="driver_debug_simple.txt or the full driver_debug.txt")
    p.add_argument("json", help="all_passes_*.json")
    p.add_argument("-o", "--output", default="exp3_iter.pdf", help="Output figure path (pdf/png)")
    p.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="If the txt contains multiple [AL] runs, select one manually (0-based)",
    )
    p.add_argument("--lang", choices=["en"], default="en")
    p.add_argument(
        "--title",
        default=None,
        help="Optional centered title. Omit it for the compact paper-style layout.",
    )
    p.add_argument(
        "--panel-label",
        default=None,
        help="Optional panel label, for example: e  -> renders as (e)",
    )
    p.add_argument(
        "--pie-zoom",
        type=float,
        default=0.18,
        help="Relative pie size. 0.18 is the default reference size.",
    )
    p.add_argument(
        "--show-x-step-labels",
        action="store_true",
        help="Show the original per-point x tick labels instead of the compact paper-style axis.",
    )
    return p.parse_args(argv)



def load_json(path: Path) -> Dict:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            json_names = [
                n for n in zf.namelist()
                if n.lower().endswith(".json") and not n.startswith("__MACOSX/")
            ]
            if not json_names:
                raise ValueError(f"No json file found inside zip: {path}")
            with zf.open(json_names[0]) as f:
                return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def extract_al_runs(txt: str) -> List[ALRun]:
    matches = list(re.finditer(r"\[AL\] init:", txt))
    if not matches:
        raise ValueError("No [AL] init section was found in the txt file.")
    runs: List[ALRun] = []
    for idx, m in enumerate(matches):
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(txt)
        seg = txt[start:end]

        init_m = re.search(
            r"\[AL\] init: .*?weights=(\d+).*?blocks=(\d+).*?block_layer_span=(\d+)",
            seg,
        )
        outer0_m = re.search(fr"{AL_PREFIX} outer0: done total=([0-9.]+)s", seg)
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
        raise ValueError("Found [AL] init, but did not find outer0: done.")
    return runs



def choose_run(runs: List[ALRun], json_data: Dict, forced_run_index: Optional[int]) -> ALRun:
    if forced_run_index is not None:
        for run in runs:
            if run.run_index == forced_run_index:
                return run
        raise ValueError(
            f"--run-index={forced_run_index} is out of range. Available run_index values: {[r.run_index for r in runs]}"
        )

    passes = json_data.get("passes", [])
    if not passes:
        raise ValueError("No passes were found in the json file.")
    json_outer0 = float(passes[0]["times"]["total"])
    json_pass_count = len(passes)

    # Automatic matching: prefer the closest outer0 total, then the closest outer-pass count.
    scored: List[Tuple[float, float, int, ALRun]] = []
    for run in runs:
        max_outer_in_txt = 0
        for m in re.finditer(fr"{AL_PREFIX} outer(\d+): after inner total=", run.raw_text):
            max_outer_in_txt = max(max_outer_in_txt, int(m.group(1)))
        pass_count_gap = abs(max_outer_in_txt + 1 - json_pass_count)
        total_gap = abs(run.outer0_total - json_outer0)
        scored.append((total_gap, float(pass_count_gap), run.run_index, run))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return scored[0][3]



def parse_log_run(run_text: str) -> Dict:
    out: Dict = {}
    out0 = re.search(
        fr"{AL_PREFIX} outer0: done total=([0-9.]+)s prefill=([0-9.]+)s decode=([0-9.]+)s",
        run_text,
    )
    if not out0:
        raise ValueError("The selected [AL] run does not contain outer0: done.")
    out["outer0"] = {
        "total": float(out0.group(1)),
        "prefill": float(out0.group(2)),
        "decode": float(out0.group(3)),
    }

    init_assign = re.search(
        fr"{AL_PREFIX} outer0->outer1: initial assign explicit_weights=(\d+) "
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
        fr"{AL_PREFIX} outer(\d+): baseline total=([0-9.]+)s prefill=([0-9.]+)s decode=([0-9.]+)s",
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
        fr"{AL_PREFIX} inner(\d+): ACCEPT block=([A-Z0-9\-_.]+) ([A-Z0-9\-]+)->([A-Z0-9\-]+) "
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
        raise ValueError(f"Unable to parse block name: {block}")
    lo, hi, kind = m.groups()
    keys: List[str] = []
    for layer in range(int(lo), int(hi) + 1):
        for shard in (0, 1):
            key = f"L{layer}_{kind}_s{shard}"
            if key not in valid_weight_keys:
                raise KeyError(f"Key mapped from block={block} is not present in json.weight_sizes: {key}")
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
        raise ValueError("No passes were found in the json file.")

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
            raise ValueError(f"Missing final map for outer{outer} in json.")
        if outer not in baselines:
            raise ValueError(f"Missing baseline total for outer{outer} in txt.")

        # reverse accepts: final(after_inner) -> baseline(step0)
        baseline_map = dict(final_maps[outer])
        for rec in reversed(accepts_by_outer.get(outer, [])):
            for key in block_to_weight_keys(rec["block"], valid_keys):
                cur_fmt = baseline_map.get(key, "ND")
                if cur_fmt != rec["dst"]:
                    raise ValueError(
                        f"Format mismatch while reconstructing outer{outer} baseline: key={key}, "
                        f"current={cur_fmt}, expected_final={rec['dst']}"
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
            # For the last accepted step, use the exact after_inner total from json to avoid
            # precision loss from the 6-decimal log value.
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

        # Sanity check: replayed ACCEPT operations should match the final json map.
        if cur_map != final_maps[outer]:
            raise ValueError(
                f"After replaying ACCEPT operations for outer{outer}, the map does not match the final json format map."
            )

    return states



def create_pie_box(values: Sequence[float], colors: Sequence[str], diameter_pt: float = 30.0) -> DrawingArea:
    safe_values = [max(0.0, float(v)) for v in values]
    total = sum(safe_values)
    safe_colors = list(colors)
    if total <= 0:
        safe_values = [1.0]
        safe_colors = ["#eeeeee"]
        total = 1.0

    da = DrawingArea(diameter_pt, diameter_pt, clip=False)
    center = (diameter_pt / 2.0, diameter_pt / 2.0)
    radius = max(0.1, diameter_pt / 2.0 - 0.75)

    end_angle = 90.0
    for value, color in zip(safe_values, safe_colors):
        if value <= 0:
            continue
        sweep = 360.0 * value / total
        theta1 = end_angle - sweep
        theta2 = end_angle
        da.add_artist(
            Wedge(
                center,
                radius,
                theta1,
                theta2,
                facecolor=color,
                edgecolor="white",
                linewidth=PIE_SEPARATOR_WIDTH,
                antialiased=True,
                joinstyle="round",
            )
        )
        end_angle = theta1

    da.add_artist(
        Circle(
            center,
            radius=radius,
            fill=False,
            edgecolor=PIE_OUTLINE_COLOR,
            linewidth=PIE_OUTLINE_WIDTH,
            antialiased=True,
            zorder=5,
        )
    )
    return da



def _candidate_offsets(point_idx: int) -> List[Tuple[float, float]]:
    above = [
        (0.0, 28.0),
        (-18.0, 26.0),
        (18.0, 26.0),
        (-28.0, 20.0),
        (28.0, 20.0),
        (0.0, 38.0),
        (-24.0, 36.0),
        (24.0, 36.0),
        (-36.0, 18.0),
        (36.0, 18.0),
    ]
    below = [
        (0.0, -28.0),
        (-18.0, -26.0),
        (18.0, -26.0),
        (-28.0, -20.0),
        (28.0, -20.0),
        (0.0, -38.0),
        (-24.0, -36.0),
        (24.0, -36.0),
        (-36.0, -18.0),
        (36.0, -18.0),
    ]
    side = [(-40.0, 0.0), (40.0, 0.0), (-48.0, 12.0), (48.0, 12.0), (-48.0, -12.0), (48.0, -12.0)]

    # Start with a mostly alternating top/bottom preference so the figure resembles the compact paper layout.
    # Step 0 prefers below, later steps mostly alternate above and below.
    if point_idx == 0:
        preferred = below + above
    elif point_idx % 2 == 1:
        preferred = above + below
    else:
        preferred = below + above
    return preferred + side



def _boxes_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    margin: float = 3.0,
) -> Tuple[bool, float]:
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
        w_px = renderer.points_to_pixels(spec.diameter_pt)
        h_px = renderer.points_to_pixels(spec.diameter_pt)
        best_offset: Optional[Tuple[float, float]] = None
        best_score = float("inf")

        for off_x_pt, off_y_pt in _candidate_offsets(spec.point_idx):
            off_x_px = renderer.points_to_pixels(off_x_pt)
            off_y_px = renderer.points_to_pixels(off_y_pt)
            cx = point_px[0] + off_x_px
            cy = point_px[1] + off_y_px
            box = (cx - w_px / 2.0, cy - h_px / 2.0, cx + w_px / 2.0, cy + h_px / 2.0)

            overlap_area = 0.0
            overlap_count = 0
            for other in occupied:
                hit, area = _boxes_overlap(box, other, margin=3.0)
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

            radius = math.hypot(off_x_px, off_y_px)
            score = overlap_count * 1e9 + overlap_area * 1e5 + overflow * 1e3 + radius
            if score < best_score:
                best_score = score
                best_offset = (off_x_pt, off_y_pt)

            if overlap_count == 0 and overflow == 0:
                best_offset = (off_x_pt, off_y_pt)
                break

        if best_offset is None:
            best_offset = (0.0, 30.0)

        cx = point_px[0] + renderer.points_to_pixels(best_offset[0])
        cy = point_px[1] + renderer.points_to_pixels(best_offset[1])
        final_box = (cx - w_px / 2.0, cy - h_px / 2.0, cx + w_px / 2.0, cy + h_px / 2.0)
        occupied.append(final_box)

        artist = AnnotationBbox(
            spec.box,
            (spec.x, spec.y),
            xybox=best_offset,
            xycoords="data",
            boxcoords="offset points",
            pad=0,
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=6,
        )
        ax.add_artist(artist)



def choose_figsize(num_x: int) -> Tuple[float, float]:
    width = max(10.0, min(18.0, 2.2 + 0.95 * num_x))
    height = 2.9
    return width, height



def make_text(lang: str) -> Dict[str, str]:
    return {
        "title": "",
        "xlabel": "Steps",
        "ylabel": "Total time (s)",
    }



def _set_bold_ticklabels(ax: plt.Axes) -> None:
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")



def plot_states(
    states: Sequence[StepState],
    output_path: Path,
    lang: str = "en",
    pie_zoom: float = 0.18,
    title: Optional[str] = None,
    panel_label: Optional[str] = None,
    show_x_step_labels: bool = False,
) -> None:
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
        linewidth=2.0,
        marker="o",
        markersize=6.4,
        markerfacecolor=line_color,
        markeredgewidth=0.0,
        markeredgecolor=line_color,
        zorder=4,
    )

    pie_specs: List[PieSpec] = []
    base_diameter_pt = 30.0
    diameter_pt = base_diameter_pt * (pie_zoom / 0.18)
    for i, s in enumerate(states):
        values = [s.ratios.get("ND", 0.0), s.ratios.get("NZ", 0.0), s.ratios.get("PIM-OPT", 0.0)]
        colors = [PIE_COLORS["ND"], PIE_COLORS["NZ"], PIE_COLORS["PIM-OPT"]]
        pie_specs.append(
            PieSpec(
                x=float(i),
                y=float(s.total_makespan_s),
                box=create_pie_box(values, colors, diameter_pt=diameter_pt),
                diameter_pt=diameter_pt,
                point_idx=i,
            )
        )

    ymin, ymax = min(ys), max(ys)
    spread = ymax - ymin
    pad = max(0.010, spread * 0.22)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(-0.15, len(states) - 0.45)

    ax.set_xticks(xs)
    if show_x_step_labels:
        ax.set_xticklabels(xlabels, rotation=26, ha="right", fontsize=11)
        ax.tick_params(axis="x", pad=8, length=4.5, width=1.2)
    else:
        ax.set_xticklabels(["" for _ in xs])
        ax.tick_params(axis="x", labelbottom=False, length=4.5, width=1.2)

    ax.set_ylabel(text["ylabel"], fontsize=18, fontweight="bold")
    ax.text(
        0.995,
        0.02,
        text["xlabel"],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
        fontweight="normal",
    )

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", labelsize=16, width=1.6, length=6.0)
    _set_bold_ticklabels(ax)

    ax.grid(axis="y", which="major", linestyle="--", linewidth=1.15, color=GRID_COLOR, alpha=1.0)
    ax.grid(axis="y", which="minor", linestyle="--", linewidth=1.15, color=GRID_COLOR, alpha=1.0)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")

    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.82)
    place_pies(ax, pie_specs)

    legend_handles = [
        Patch(facecolor=PIE_COLORS["ND"], edgecolor="none", label=LEGEND_LABELS["ND"]),
        Patch(facecolor=PIE_COLORS["NZ"], edgecolor="none", label=LEGEND_LABELS["NZ"]),
        Patch(facecolor=PIE_COLORS["PIM-OPT"], edgecolor="none", label=LEGEND_LABELS["PIM-OPT"]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.93, 0.985),
        ncol=3,
        frameon=False,
        prop={"weight": "bold", "size": 14},
        handlelength=1.1,
        handleheight=1.4,
        handletextpad=0.45,
        columnspacing=0.95,
        borderaxespad=0.0,
    )

    normalized_panel_label = None
    if panel_label is not None and panel_label.strip():
        pl = panel_label.strip()
        normalized_panel_label = pl if pl.startswith("(") else f"({pl})"

    if title:
        fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=16, fontweight="bold")

    if normalized_panel_label:
        fig.text(0.995, 0.965, normalized_panel_label, ha="right", va="top", fontsize=17, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)



def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_fonts()
    args = parse_args(argv)

    txt_path = Path(args.txt).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if not txt_path.is_file():
        raise FileNotFoundError(f"txt file does not exist: {txt_path}")
    if not json_path.is_file():
        raise FileNotFoundError(f"json file does not exist: {json_path}")

    txt = txt_path.read_text(encoding="utf-8", errors="ignore")
    json_data = load_json(json_path)

    runs = extract_al_runs(txt)
    chosen = choose_run(runs, json_data, args.run_index)
    states = reconstruct_states_from_txt_json(chosen.raw_text, json_data)
    plot_states(
        states,
        out_path,
        lang=args.lang,
        pie_zoom=args.pie_zoom,
        title=args.title,
        panel_label=args.panel_label,
        show_x_step_labels=args.show_x_step_labels,
    )

    print(f"[INFO] Detected {len(runs)} [AL] run(s) in txt, selected run_index={chosen.run_index}")
    print(f"[INFO] Selected run: outer0_total={chosen.outer0_total:.6f}s block_layer_span={chosen.block_layer_span}")
    print(f"[INFO] Reconstructed {len(states)} point(s):")
    for s in states:
        print(
            f"  outer={s.outer_pass} step={s.accepted_step} total={s.total_makespan_s:.9f}s "
            f"ND={s.ratios['ND']:.4f} NZ={s.ratios['NZ']:.4f} PIM={s.ratios['PIM-OPT']:.4f} "
            f"note={s.note}"
        )
    print(f"[INFO] Output figure: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
