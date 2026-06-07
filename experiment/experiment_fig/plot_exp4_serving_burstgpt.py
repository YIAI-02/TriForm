#!/usr/bin/env python3
"""Plot BurstGPT serving results as p50/p90 three-panel bar charts.

python plot_exp4_serving_burstgpt.py \
  /path/to/burstgpt_serving_summary.json \
  --out_dir /path/to/output_dir \
  --title "BurstGPT trace replay on Llama-7B / HP32"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# User-specified colors. The experiment has six policies; the sixth color is a
# matching pastel tone for Bifocal.
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


def plot_one_percentile(df: pd.DataFrame, percentile: str, out_path: Path, title: str, log_scale: bool = True) -> None:
    """Draw one PDF with three panels: TTFT, TBT, E2E."""
    policies = df["policy"].tolist()
    x = np.arange(len(policies))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)

    for ax, (metric_name, col, ylabel, scale) in zip(axes, _metric_config(percentile)):
        values = df[col].to_numpy(dtype=float) * scale
        pd_raw = float(df.loc[df["policy"] == "PD", col].iloc[0]) if (df["policy"] == "PD").any() else float(values[0] / scale)
        colors = [POLICY_COLORS.get(p, "#cccccc") for p in policies]
        bars = ax.bar(x, values, width=0.68, color=colors, edgecolor="black", linewidth=0.7)

        if log_scale:
            positive = values[np.isfinite(values) & (values > 0)]
            if len(positive) > 0:
                ax.set_yscale("linear")
                ax.set_ylim(0, ymax * 1.20)
        else:
            ymax = float(np.nanmax(values)) if len(values) else 1.0
            ax.set_ylim(0, ymax * 1.22)

        for i, (bar, policy) in enumerate(zip(bars, policies)):
            value_raw = float(df.loc[i, col])
            speed = _safe_speedup(pd_raw, value_raw)
            if np.isfinite(speed):
                label = f"{speed:.2f}x"
            else:
                label = "-"
            y = bar.get_height()
            if log_scale:
                y_text = y * 1.12
            else:
                y_text = y + 0.03 * max(values)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                rotation=0,
            )

        ax.set_title(metric_name, fontsize=12)
        ax.set_ylabel(ylabel + (", log scale" if log_scale else ""), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=28, ha="right", fontsize=9)
        ax.grid(axis="y", which="both", linestyle="--", linewidth=0.45, alpha=0.55)
        ax.text(
            0.02, 0.98,
            "Numbers: speedup over PD",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0),
        )

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=POLICY_COLORS[p], edgecolor="black") for p in policies]
    fig.legend(handles, policies, loc="upper center", ncol=min(len(policies), 6), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to burstgpt_serving_summary.json, its directory, or a summary CSV")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument("--title", help="Figure title prefix")
    parser.add_argument("--linear", action="store_true", help="Use linear y-axis instead of log scale")
    args = parser.parse_args()

    df = load_results(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "burstgpt_result_table_policycolor_3panel.csv", index=False)

    plot_one_percentile(
        df,
        "p50",
        out_dir / "fig_burstgpt_serving_p50_3panel.pdf",
        f"{args.title} - p50",
        log_scale=not args.linear,
    )
    plot_one_percentile(
        df,
        "p90",
        out_dir / "fig_burstgpt_serving_p90_3panel.pdf",
        f"{args.title} - p90",
        log_scale=not args.linear,
    )


if __name__ == "__main__":
    main()
