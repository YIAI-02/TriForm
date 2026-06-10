#!/usr/bin/env python3
"""
python ./experiment/experiment_fig/plot_exp4_serving_burstgpt.py \
    /lustre/home/2501111916/workspace/DOPS_0606_rebuttal/TriForm/output/burstgpt_eval_backlog/llama_7b_fp16_b1_s1/burstgpt_serving/burstgpt_serving_summary.json \
    --out_dir ./figs/supp_exp/exp4_request \
    --bar_width 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POLICY_COLORS: Dict[str, str] = {
    "PD": "#b8add9",
    "AF": "#d8b6bd",
    "PD+FFN": "#bfd8b0",
    "PD+Linear": "#b9dce7",
    "PD+Attn": "#d8d1a1",
    "Bifocal": "#e4c7a1",
}

POLICY_ORDER = ["PD", "AF", "PD+FFN", "PD+Linear", "PD+Attn", "Bifocal"]


def _clean_policy_name(name: object) -> str:
    raw = str(name or "").strip()
    if raw.startswith("algo:"):
        raw = raw[len("algo:"):]
    if raw.upper() == "BIFOCAL":
        return "Bifocal"
    return raw


def load_results(path: Path) -> pd.DataFrame:
    """Load result table from a BurstGPT summary JSON or a prepared CSV."""
    if path.is_dir():
        path = path / "burstgpt_serving_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows: List[dict] = []
        for s in data.get("summaries", []):
            policy = _clean_policy_name(s.get("policy"))
            ttft = s.get("ttft") or {}
            tbt = s.get("tbt_token") or s.get("tbt") or {}
            e2e = s.get("e2e_latency") or {}
            rows.append({
                "policy": policy,
                "ttft_p50_s": ttft.get("p50_s"),
                "ttft_p90_s": ttft.get("p90_s"),
                "tbt_p50_s": tbt.get("p50_s"),
                "tbt_p90_s": tbt.get("p90_s"),
                "e2e_p50_s": e2e.get("p50_s"),
                "e2e_p90_s": e2e.get("p90_s"),
            })
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
        df["policy"] = df["policy"].map(_clean_policy_name)

    required = {
        "policy", "ttft_p50_s", "ttft_p90_s", "tbt_p50_s", "tbt_p90_s", "e2e_p50_s", "e2e_p90_s"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in sorted(required - {"policy"}):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["policy"])

    order_map = {p: i for i, p in enumerate(POLICY_ORDER)}
    df["_order"] = df["policy"].map(lambda x: order_map.get(x, 10_000))
    df = df.sort_values(["_order", "policy"]).drop(columns=["_order"]).reset_index(drop=True)
    return df


def _metric_config(percentile: str):
    if percentile not in {"p50", "p90"}:
        raise ValueError("percentile must be p50 or p90")
    return [
        ("TTFT", f"ttft_{percentile}_s", "Latency (s)", 1.0),
        ("TBT", f"tbt_{percentile}_s", "Latency (ms)", 1000.0),
        ("E2E", f"e2e_{percentile}_s", "Latency (s)", 1.0),
    ]


def _safe_speedup(pd_value: float, value: float) -> float:
    if value is None or not np.isfinite(value) or value <= 0:
        return float("nan")
    return float(pd_value) / float(value)


def _draw_panel(
    ax,
    df: pd.DataFrame,
    metric_name: str,
    col: str,
    ylabel: str,
    scale: float,
    row_label: str,
    show_xticklabels: bool,
    bar_width: float,
    label_fontsize: float,
    speedup_fontsize: float,
) -> None:
    policies = df["policy"].tolist()
    # Center spacing == bar width => adjacent bars touch with no horizontal gap.
    x = np.arange(len(policies), dtype=float) * bar_width
    values = df[col].to_numpy(dtype=float) * scale
    pd_raw = float(df.loc[df["policy"] == "PD", col].iloc[0]) if (df["policy"] == "PD").any() else float(df[col].iloc[0])
    colors = [POLICY_COLORS.get(p, "#cccccc") for p in policies]

    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.45,
    )

    ymax = float(np.nanmax(values)) if len(values) and np.isfinite(np.nanmax(values)) else 1.0
    if ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0, ymax * 1.18)

    # Tight frame on left/right: frame hugs the first and last bars.
    ax.set_xlim(x[0] - bar_width / 2, x[-1] + bar_width / 2)
    ax.margins(x=0)

    for i, bar in enumerate(bars):
        value_raw = float(df.loc[i, col])
        speed = _safe_speedup(pd_raw, value_raw)
        label = f"{speed:.2f}x" if np.isfinite(speed) else "-"
        y = bar.get_height()
        y_text = y + 0.02 * ymax
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            label,
            ha="center",
            va="bottom",
            fontsize=speedup_fontsize,
            rotation=90,
            color="black",
            clip_on=False,
        )

    ax.set_title(metric_name, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(policies, rotation=90, ha="center", fontsize=label_fontsize)
    else:
        ax.set_xticklabels([])

    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.55)
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", pad=3)

    # Keep the full frame visible.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    if row_label:
        ax.text(
            -0.11,
            0.5,
            row_label,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=16,
            fontweight="bold",
        )


def plot_combined(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    bar_width: float = 0.28,
    label_fontsize: float = 16,
    speedup_fontsize: float = 16,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.9), constrained_layout=False)

    for j, (metric_name, col, ylabel, scale) in enumerate(_metric_config("p50")):
        _draw_panel(
            axes[0, j],
            df,
            metric_name,
            col,
            ylabel,
            scale,
            row_label="p50" if j == 0 else "",
            show_xticklabels=False,
            bar_width=bar_width,
            label_fontsize=label_fontsize,
            speedup_fontsize=speedup_fontsize,
        )

    for j, (metric_name, col, ylabel, scale) in enumerate(_metric_config("p90")):
        _draw_panel(
            axes[1, j],
            df,
            metric_name,
            col,
            ylabel,
            scale,
            row_label="p90" if j == 0 else "",
            show_xticklabels=True,
            bar_width=bar_width,
            label_fontsize=label_fontsize,
            speedup_fontsize=speedup_fontsize,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=POLICY_COLORS[p], edgecolor="black")
        for p in df["policy"].tolist()
    ]

    # Reserve explicit vertical space to avoid title/legend overlap.
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.88, wspace=0.22, hspace=0.14)
    fig.legend(
        handles,
        df["policy"].tolist(),
        loc="upper center",
        ncol=min(len(df), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        fontsize=16,
        handlelength=1.6,
        columnspacing=1.6,
    )
    # fig.suptitle(title, fontsize=18, y=0.94)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to burstgpt_serving_summary.json, its directory, or a summary CSV")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument("--title", default="BurstGPT Serving", help="Figure title")
    parser.add_argument("--bar_width", type=float, default=0.28, help="Bar width; center spacing is the same value")
    parser.add_argument("--label_fontsize", type=float, default=16, help="Font size for x tick labels")
    parser.add_argument("--speedup_fontsize", type=float, default=16, help="Font size for speedup annotations")
    args = parser.parse_args()

    df = load_results(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # df.to_csv(out_dir / "burstgpt_result_table_policycolor_2x3.csv", index=False)

    plot_combined(
        df,
        out_dir / "exp4_llama.pdf",
        args.title,
        bar_width=args.bar_width,
        label_fontsize=args.label_fontsize,
        speedup_fontsize=args.speedup_fontsize,
    )


if __name__ == "__main__":
    main()
