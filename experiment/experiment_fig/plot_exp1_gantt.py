#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot a stage-local Gantt chart from one ``best_summary_*.json`` file.

Run this script from ``experiment/experiment_fig``. The JSON input is produced
by ``python src/main.py evaluate`` and is usually stored inside an ``algo_*``
directory under ``output/``.

Example
-------
python plot_exp1_gantt.py \
  --json ../../output/evaluate_single_test/hardware_1npu_2aim/llama_7b_fp16_b1_s2/algo_Bifocal/best_summary_128x512.json \
  --stage decode \
  --layer 0 \
  --token 8 \
  --time_unit ms \
  --out ../../figs/exp1/gantt/llama_7b_bifocal_128x512_decode_L8.pdf \
  --fig_w 18 --fig_h 4.8 \
  --label_min_frac 0.04
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# User-requested palette only
PALETTE = {
    "nonlinear": "#bdade4",  # N / Softmax
    "qkv": "#e4adb5",        # Q / K / V
    "attn_core": "#aee4ad",  # QK^T / SV
    "proj": "#e4ddad",       # Proj
    "ffn": "#add9e4",        # FFN1 / FFN2 / FFN3
}

# Raw op -> display label / color bucket
DISPLAY_STYLE: Dict[str, Dict[str, str]] = {
    "N": {"color": PALETTE["nonlinear"], "label": "N"},
    "Q": {"color": PALETTE["qkv"], "label": "Q"},
    "K": {"color": PALETTE["qkv"], "label": "K"},
    "V": {"color": PALETTE["qkv"], "label": "V"},
    "QK^T": {"color": PALETTE["attn_core"], "label": r"QK$^T$"},
    "Softmax": {"color": PALETTE["nonlinear"], "label": "Softmax"},
    "SV": {"color": PALETTE["attn_core"], "label": "SV"},
    "Proj": {"color": PALETTE["proj"], "label": "Proj"},
    "FFN1": {"color": PALETTE["ffn"], "label": "FFN1"},
    "FFN2": {"color": PALETTE["ffn"], "label": "FFN2"},
    "FFN3": {"color": PALETTE["ffn"], "label": "FFN3"},
}
DISPLAY_ORDER = ["N", "Q", "K", "V", "QK^T", "Softmax", "SV", "Proj", "FFN1", "FFN2", "FFN3"]

_NUM_TAIL_RE = re.compile(r"^(.*?)(\d+)$")
LAYER_RE = re.compile(r"^L(\d+)_([^\s]+)$")


def _natural_key(s: str) -> Tuple[str, int]:
    s = str(s)
    m = _NUM_TAIL_RE.match(s)
    if m:
        return m.group(1), int(m.group(2))
    return s, -1


def clamp_font(v: float) -> float:
    return max(7.1, float(v))


def layer_of(node_id: str) -> Optional[int]:
    m = LAYER_RE.match(str(node_id))
    return int(m.group(1)) if m else None


def op_of(node_id: str) -> str:
    m = LAYER_RE.match(str(node_id))
    return m.group(2) if m else str(node_id)


def is_kv_write(op: str) -> bool:
    c = str(op).lower()
    return c.startswith("k_write") or c.startswith("v_write")


def display_name_of(op: str) -> str:
    c = str(op).lower()
    if c in {"ln", "ln2", "add1", "add2", "swiglu_s0", "swiglu_s1", "swiglu"}:
        return "N"
    if c.startswith("q_"):
        return "Q"
    if c.startswith("k_"):
        return "K"
    if c.startswith("v_"):
        return "V"
    if c.startswith("qk"):
        return "QK^T"
    if c.startswith("softmax"):
        return "Softmax"
    if c.startswith("sv"):
        return "SV"
    if c.startswith("o_") or c.startswith("allreduce_o"):
        return "Proj"
    if c.startswith("ffn_w1"):
        return "FFN1"
    if c.startswith("ffn_w2") or c.startswith("allreduce_ffn"):
        return "FFN2"
    if c.startswith("ffn_w3"):
        return "FFN3"
    if c.startswith("swiglu"):
        return "N"
    return "N"


def device_sort_key(name: str) -> Tuple[int, Tuple[str, int]]:
    s = str(name).lower()
    if "npu" in s or "ascend" in s:
        return 0, _natural_key(str(name))
    if s.startswith("pim"):
        return 1, _natural_key(str(name))
    if s == "comm":
        return 99, _natural_key(str(name))
    return 50, _natural_key(str(name))


def time_scale_and_unit(unit: str) -> Tuple[float, str]:
    unit = str(unit).lower()
    if unit == "s":
        return 1.0, "s"
    if unit == "ms":
        return 1e3, "ms"
    if unit == "us":
        return 1e6, "μs"
    if unit == "ns":
        return 1e9, "ns"
    raise ValueError(f"Unsupported time unit: {unit}")


def load_stage_df(json_path: Path, stage: str, token: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, object]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stage = str(stage).lower()
    if stage == "prefill":
        rows = data.get("prefill_schedule", [])
        meta = {
            "policy": data.get("policy"),
            "config": data.get("config", {}),
            "token": None,
            "seq_len": data.get("config", {}).get("prefill_len"),
        }
    elif stage == "decode":
        steps = data.get("decode_steps", [])
        if token is None:
            raise ValueError("--token is required for stage=decode")
        step = None
        for item in steps:
            if int(item.get("t", -1)) == int(token):
                step = item
                break
        if step is None:
            raise ValueError(f"Decode token {token} not found. Available range: 0..{max(len(steps)-1, 0)}")
        rows = step.get("schedule", [])
        meta = {
            "policy": data.get("policy"),
            "config": data.get("config", {}),
            "token": int(step.get("t", token)),
            "seq_len": step.get("seq_len"),
            "step_time": step.get("step_time"),
        }
    else:
        raise ValueError("stage must be one of: prefill, decode")

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No events found for stage={stage}")

    df = df.copy()
    df["node_id"] = df["node_id"].astype(str)
    df["device"] = df["device"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df["duration"] = pd.to_numeric(df.get("duration"), errors="coerce")
    df = df.dropna(subset=["start", "finish"]).copy()
    df["duration"] = df["finish"] - df["start"]
    df["layer"] = df["node_id"].map(layer_of)
    df["op"] = df["node_id"].map(op_of)
    df["display_op"] = df["op"].map(display_name_of)
    return df.reset_index(drop=True), meta


def prepare_df(
    df: pd.DataFrame,
    *,
    layer: Optional[int],
    shift_to_zero: bool,
    hide_comm: bool,
    hide_kv_write: bool,
) -> Tuple[pd.DataFrame, float, float]:
    out = df.copy()
    if layer is not None:
        out = out[out["layer"] == int(layer)].copy()
        if out.empty:
            raise ValueError(f"No rows found for layer={layer}")
    if hide_comm:
        out = out[out["device"].str.upper() != "COMM"].copy()
    if hide_kv_write:
        out = out[~out["op"].map(is_kv_write)].copy()
    if out.empty:
        raise ValueError("No rows left after filtering")
    t0 = float(out["start"].min())
    t1 = float(out["finish"].max())
    if shift_to_zero:
        out["start"] = out["start"] - t0
        out["finish"] = out["finish"] - t0
    out["duration"] = out["finish"] - out["start"]
    return out.reset_index(drop=True), t0, t1


def short_device_label(name: str) -> str:
    s = str(name)
    if s.upper() == "COMM":
        return "COMM"
    if "NPU" in s.upper() or "ASCEND" in s.upper():
        m = re.search(r"NPU(\d+)", s.upper())
        return f"NPU{m.group(1)}" if m else "NPU"
    return s


def auto_title(stage: str, layer: Optional[int], token: Optional[int]) -> str:
    if stage == "prefill":
        return f"Prefill-Layer{layer if layer is not None else 'x'}"
    return f"Decode-Token{token if token is not None else 'x'}-Layer{layer if layer is not None else 'x'}"


def plot_gantt(
    df: pd.DataFrame,
    out_path: Path,
    *,
    stage: str,
    layer: Optional[int],
    meta: Dict[str, object],
    time_unit: str,
    shift_to_zero: bool,
    fig_w: float,
    fig_h: float,
    subplot_left: float,
    subplot_right: float,
    subplot_top: float,
    subplot_bottom: float,
    lane_h: float,
    lane_gap: float,
    dpi: int,
    title_fontsize: float,
    axis_label_fontsize: float,
    tick_fontsize: float,
    lane_label_fontsize: float,
    legend_fontsize: float,
    bar_label_fontsize: float,
    note_fontsize: float,
    edge_linewidth: float,
    tiny_event_threshold_s: float,
    label_min_frac: float,
    label_rotation: float,
    show_title: bool,
    show_legend: bool,
    legend_ncol: int,
    show_note: bool,
    title_override: Optional[str] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale, unit_label = time_scale_and_unit(time_unit)
    span = float(df["finish"].max() - df["start"].min())
    span_scaled = span * scale
    label_threshold = max(0.0, span_scaled * float(label_min_frac))

    devices = sorted(df["device"].drop_duplicates().tolist(), key=device_sort_key)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    yticks: List[float] = []
    ylabels: List[str] = []
    y0 = 0.0
    present_labels: List[str] = []
    tiny_scaled = tiny_event_threshold_s * scale

    for dev in devices:
        sub = df[df["device"] == dev].sort_values(["start", "finish", "node_id"]) 
        yticks.append(y0 + lane_h / 2.0)
        ylabels.append(short_device_label(dev))
        for _, row in sub.iterrows():
            disp = row["display_op"]
            if disp not in present_labels:
                present_labels.append(disp)
            color = DISPLAY_STYLE[disp]["color"]
            start = float(row["start"]) * scale
            dur = float(row["duration"]) * scale
            end = float(row["finish"]) * scale
            if dur <= tiny_scaled:
                x = start if dur == 0 else 0.5 * (start + end)
                ax.vlines(x, y0, y0 + lane_h, color=color, linewidth=max(1.0, edge_linewidth + 0.25), alpha=0.98)
            else:
                ax.broken_barh(
                    [(start, dur)],
                    (y0, lane_h),
                    facecolors=color,
                    edgecolors="white",
                    linewidth=edge_linewidth,
                    alpha=0.98,
                )
            if dur >= label_threshold:
                ax.text(
                    start + max(dur / 2.0, 0.0),
                    y0 + lane_h / 2.0,
                    DISPLAY_STYLE[disp]["label"],
                    ha="center",
                    va="center",
                    rotation=label_rotation,
                    fontsize=bar_label_fontsize,
                    color="black",
                    clip_on=True,
                )
        y0 += lane_h + lane_gap

    max_end = float(df["finish"].max()) * scale
    ax.set_xlim(0.0, max(1e-12, max_end * 1.02))
    ax.set_ylim(-lane_gap * 0.35, y0 - lane_gap + lane_gap * 0.35)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=lane_label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Time ({unit_label})", fontsize=axis_label_fontsize)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.35)

    if show_title:
        title = title_override or auto_title(stage, layer, meta.get("token"))
        ax.set_title(title, fontsize=title_fontsize, pad=8)

    if show_note:
        note = "Shifted to the selected window start" if shift_to_zero else "Absolute timeline"
        ax.text(
            1.0,
            1.01,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=note_fontsize,
        )

    if show_legend:
        handles = [
            Patch(facecolor=DISPLAY_STYLE[name]["color"], edgecolor="none", label=DISPLAY_STYLE[name]["label"])
            for name in DISPLAY_ORDER if name in present_labels
        ]
        if handles:
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=max(1, int(legend_ncol)),
                frameon=False,
                fontsize=legend_fontsize,
                columnspacing=1.0,
                handletextpad=0.4,
            )

    fig.subplots_adjust(left=subplot_left, right=subplot_right, top=subplot_top, bottom=subplot_bottom)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot HEFTHint Gantt chart from a summary JSON (paper-style)")
    ap.add_argument("--json", required=True, help="Path to best_summary_*.json")
    ap.add_argument("--stage", required=True, choices=["prefill", "decode"], help="Which stage to draw")
    ap.add_argument("--token", type=int, default=None, help="Decode token index t (0-based, required when stage=decode)")
    ap.add_argument("--layer", type=int, default=None, help="Optional layer index to isolate, e.g. 13")
    ap.add_argument("--time_unit", default="ms", choices=["s", "ms", "us", "ns"], help="Time unit for the x-axis")
    ap.add_argument("--no_shift_to_zero", action="store_true", help="Keep absolute time instead of shifting to window start")
    ap.add_argument("--out", required=True, help="Output figure path (.png or .pdf)")

    ap.add_argument("--show_comm", action="store_true", help="Show COMM lane (default: hidden)")
    ap.add_argument("--show_kv_write", action="store_true", help="Show K_write/V_write events (default: hidden)")

    ap.add_argument("--fig_w", type=float, default=18.0, help="Figure width in inches")
    ap.add_argument("--fig_h", type=float, default=4.8, help="Figure height in inches")
    ap.add_argument("--subplot_left", type=float, default=0.07, help="matplotlib subplots_adjust(left)")
    ap.add_argument("--subplot_right", type=float, default=0.995, help="matplotlib subplots_adjust(right)")
    ap.add_argument("--subplot_top", type=float, default=0.87, help="matplotlib subplots_adjust(top)")
    ap.add_argument("--subplot_bottom", type=float, default=0.22, help="matplotlib subplots_adjust(bottom)")
    ap.add_argument("--lane_h", type=float, default=0.82, help="Lane height")
    ap.add_argument("--lane_gap", type=float, default=0.28, help="Gap between lanes")
    ap.add_argument("--dpi", type=int, default=240, help="Output DPI")

    ap.add_argument("--title_fontsize", type=float, default=15.0, help="Title fontsize (>7)")
    ap.add_argument("--axis_label_fontsize", type=float, default=12.0, help="Axis label fontsize (>7)")
    ap.add_argument("--tick_fontsize", type=float, default=10.5, help="X tick fontsize (>7)")
    ap.add_argument("--lane_label_fontsize", type=float, default=11.5, help="Y/lane label fontsize (>7)")
    ap.add_argument("--legend_fontsize", type=float, default=10.0, help="Legend fontsize (>7)")
    ap.add_argument("--bar_label_fontsize", type=float, default=8.2, help="Bar label fontsize (>7)")
    ap.add_argument("--note_fontsize", type=float, default=9.2, help="Note fontsize (>7)")

    ap.add_argument("--edge_linewidth", type=float, default=0.9, help="Bar edge linewidth")
    ap.add_argument("--tiny_event_threshold_s", type=float, default=2e-7, help="Events at or below this duration are drawn as vertical lines")
    ap.add_argument("--label_min_frac", type=float, default=0.035, help="Show a bar label only if duration >= total_span * this fraction")
    ap.add_argument("--label_rotation", type=float, default=0.0, help="Rotation angle for bar labels")

    ap.add_argument("--no_title", action="store_true", help="Hide the title")
    ap.add_argument("--no_legend", action="store_true", help="Hide the legend")
    ap.add_argument("--show_note", action="store_true", help="Show the small note above the plot")
    ap.add_argument("--legend_ncol", type=int, default=6, help="Legend column count")
    ap.add_argument("--title", type=str, default=None, help="Optional title override")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    json_path = Path(args.json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    df, meta = load_stage_df(json_path, args.stage, args.token)
    df, _, _ = prepare_df(
        df,
        layer=args.layer,
        shift_to_zero=(not args.no_shift_to_zero),
        hide_comm=(not args.show_comm),
        hide_kv_write=(not args.show_kv_write),
    )

    plot_gantt(
        df,
        out_path,
        stage=args.stage,
        layer=args.layer,
        meta=meta,
        time_unit=args.time_unit,
        shift_to_zero=(not args.no_shift_to_zero),
        fig_w=args.fig_w,
        fig_h=args.fig_h,
        subplot_left=args.subplot_left,
        subplot_right=args.subplot_right,
        subplot_top=args.subplot_top,
        subplot_bottom=args.subplot_bottom,
        lane_h=args.lane_h,
        lane_gap=args.lane_gap,
        dpi=args.dpi,
        title_fontsize=clamp_font(args.title_fontsize),
        axis_label_fontsize=clamp_font(args.axis_label_fontsize),
        tick_fontsize=clamp_font(args.tick_fontsize),
        lane_label_fontsize=clamp_font(args.lane_label_fontsize),
        legend_fontsize=clamp_font(args.legend_fontsize),
        bar_label_fontsize=clamp_font(args.bar_label_fontsize),
        note_fontsize=clamp_font(args.note_fontsize),
        edge_linewidth=float(args.edge_linewidth),
        tiny_event_threshold_s=float(args.tiny_event_threshold_s),
        label_min_frac=float(args.label_min_frac),
        label_rotation=float(args.label_rotation),
        show_title=(not args.no_title),
        show_legend=(not args.no_legend),
        legend_ncol=int(args.legend_ncol),
        show_note=args.show_note,
        title_override=args.title,
    )


if __name__ == "__main__":
    main()
