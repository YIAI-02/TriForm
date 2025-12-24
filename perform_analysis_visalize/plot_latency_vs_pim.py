#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot latency vs #PIM(AiM) from baseline_compare_{prefill}x{decode}.json

目录结构只要路径里包含 hw_*aim* 段即可，例如：
output/experiment_1/hw_2aim_2npu/st64/llama_7b_int8_b1_s64/baseline_compare_512x128.json
output/experiment_1/hw_4aim_2npu/st64/llama_7b_int8_b1_s64/baseline_compare_512x128.json
...

用法示例：
python plot_latency_vs_pim.py \
  --root ../algorithms/output/experiment_1 \
  --model llama_7b_int8 \
  --batch 1 \
  --prefill 512 --decode 128 \
  --metric total_time_s \
  --stride 64 \
  --hw-group npu
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 你指定的配色池
COLOR_POOL = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
              "#FFBF02", "#FE8000", "#FE5D00", "#FF3F04"]


def _parse_model_str(model: str) -> Tuple[str, str, str]:
    """
    model like: llama_7b_int8 / mixtral_8x7b_int8 / palm_62b_int8
    returns: (family, variant, dtype)
    """
    parts = model.split("_")
    if len(parts) < 3:
        raise ValueError(f"--model expects llama_7b_int8, got: {model}")
    family = parts[0]
    dtype = parts[-1]
    variant = "_".join(parts[1:-1])
    return family, variant, dtype


def parse_hw_segment_from_path(path_str: str) -> Optional[str]:
    """Extract 'hw_...'(one path segment) from a path string."""
    m = re.search(r"(hw_[^/\\]+)", path_str)
    return m.group(1) if m else None


def parse_aim_from_hw_segment(hw_seg: str) -> Optional[int]:
    """
    hw_2aim_2npu -> 2
    hw_aim_npu   -> 1
    """
    m = re.search(r"hw_(\d*)aim", hw_seg)
    if not m:
        return None
    g = m.group(1)
    return int(g) if g else 1


def parse_hw_group_from_hw_segment(hw_seg: str) -> str:
    """
    hw_2aim_2npu -> 2npu
    hw_aim_npu   -> npu
    hw_2aim_halfnpu -> halfnpu
    """
    seg2 = hw_seg[3:] if hw_seg.startswith("hw_") else hw_seg
    if "aim_" in seg2:
        return seg2.split("aim_", 1)[1]
    m = re.match(r"(\d*)aim(.*)", seg2)
    if m:
        return m.group(2).lstrip("_")
    return ""


def elbow_point(x: np.ndarray, y: np.ndarray) -> Optional[int]:
    """
    Elbow detector for "latency decreases then saturates".
    Returns the x value (aim count) at elbow.

    - normalize x to [0,1]
    - invert y to "improvement" (bigger is better) and normalize to [0,1]
    - pick point with max (y_norm - x_norm)
    """
    if len(x) < 3:
        return None

    order = np.argsort(x)
    x = x[order].astype(float)
    y = y[order].astype(float)

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None

    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    if x_span <= 0 or y_span <= 0:
        return None

    x_norm = (x - x.min()) / x_span

    # smaller latency => larger improvement
    improv = (y.max() - y)
    improv_span = improv.max() - improv.min()
    if improv_span <= 0:
        return None
    y_norm = (improv - improv.min()) / improv_span

    diff = y_norm - x_norm
    idx = int(np.argmax(diff))
    return int(x[idx])


def load_one_json(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_candidate_jsons(root: Path, prefill: int, decode: int) -> Iterable[Path]:
    pattern = f"baseline_compare_{prefill}x{decode}.json"
    return root.rglob(pattern)


def build_dataframe(
    json_paths: List[Path],
    *,
    model_family: str,
    model_variant: str,
    dtype: str,
    batch: int,
    prefill: int,
    decode: int,
    stride: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for p in json_paths:
        try:
            obj = load_one_json(p)
        except Exception:
            continue

        cfg = obj.get("config", {})
        if not cfg:
            continue

        # filter by condition
        if str(cfg.get("model_family")) != str(model_family):
            continue
        if str(cfg.get("model_variant")) != str(model_variant):
            continue
        if str(cfg.get("dtype")) != str(dtype):
            continue
        if int(cfg.get("batch", -1)) != int(batch):
            continue
        if int(cfg.get("prefill_len", -1)) != int(prefill):
            continue
        if int(cfg.get("decode_len", -1)) != int(decode):
            continue
        if stride is not None and int(cfg.get("decode_sample_stride", -1)) != int(stride):
            continue

        # extract hw segment from result_dir OR from path
        hw_seg = parse_hw_segment_from_path(str(cfg.get("result_dir", ""))) or parse_hw_segment_from_path(str(p))
        if hw_seg is None:
            continue
        aim = parse_aim_from_hw_segment(hw_seg)
        if aim is None:
            continue
        hw_group = parse_hw_group_from_hw_segment(hw_seg)

        for r in obj.get("results", []):
            policy_full = str(r.get("policy", ""))
            policy = policy_full.split(":", 1)[1] if ":" in policy_full else policy_full

            rows.append({
                "aim": aim,
                "hw_group": hw_group,
                "policy": policy,
                "prefill_time_s": r.get("prefill_time_s", np.nan),
                "decode_time_s": r.get("decode_time_s", np.nan),
                "total_time_s": r.get("total_time_s", np.nan),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # if duplicates exist, average them
    agg_cols = ["prefill_time_s", "decode_time_s", "total_time_s"]
    df = (df.groupby(["hw_group", "aim", "policy"], as_index=False)[agg_cols]
            .mean())
    return df


def plot_one_group(
    df: pd.DataFrame,
    *,
    hw_group: str,
    metric: str,
    model_family: str,
    model_variant: str,
    dtype: str,
    batch: int,
    prefill: int,
    decode: int,
    stride: Optional[int],
    outdir: Path,
) -> Path:
    g = df[df["hw_group"] == hw_group].copy()
    if g.empty:
        raise ValueError(f"No data for hw_group={hw_group}")

    policies = sorted(g["policy"].unique().tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    elbow_summary = []

    for i, pol in enumerate(policies):
        sub = g[g["policy"] == pol].sort_values("aim")
        x = sub["aim"].to_numpy()
        y = sub[metric].to_numpy()

        color = COLOR_POOL[i % len(COLOR_POOL)]
        ax.plot(x, y, marker="o", linewidth=2, label=pol, color=color)

        elbow = elbow_point(x, y)
        if elbow is not None:
            y_elbow = float(sub.loc[sub["aim"] == elbow, metric].iloc[0])
            ax.scatter([elbow], [y_elbow], marker="*", s=180, color=color, zorder=5)
            elbow_summary.append((pol, elbow, y_elbow))

    ax.set_xlabel("#AiM (PIM count)  [from hw_*aim*]")
    ax.set_ylabel(f"{metric} (s)")
    title = f"{model_family}_{model_variant}_{dtype} | b{batch} | {prefill}x{decode}"
    if stride is not None:
        title += f" | stride={stride}"
    title += f" | hw_group={hw_group}"
    ax.set_title(title)

    aims_sorted = sorted(g["aim"].unique().tolist())
    ax.set_xticks(aims_sorted)

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(ncol=2, fontsize=9)

    outdir.mkdir(parents=True, exist_ok=True)
    stride_tag = f"_s{stride}" if stride is not None else ""
    outpath = outdir / f"latency_vs_aim_{model_family}_{model_variant}_{dtype}_b{batch}_{prefill}x{decode}{stride_tag}_{hw_group}_{metric}.png"
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    if elbow_summary:
        elbow_summary.sort(key=lambda t: (t[1], t[0]))
        print(f"\n[Elbow estimate] hw_group={hw_group}, metric={metric}")
        for pol, elbow, y_elbow in elbow_summary:
            print(f"  - {pol:16s} elbow@aim={elbow:>3d}  {metric}={y_elbow:.6g}s")
    else:
        print(f"\n[Elbow estimate] hw_group={hw_group}: not enough points (need >=3 aims per policy).")

    return outpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder to search (e.g., output/experiment_1)")
    ap.add_argument("--model", type=str, default=None, help="e.g., llama_7b_int8 / mixtral_8x7b_int8")
    ap.add_argument("--model-family", type=str, default=None)
    ap.add_argument("--model-variant", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)

    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--prefill", type=int, required=True)
    ap.add_argument("--decode", type=int, required=True)

    ap.add_argument("--stride", type=int, default=None, help="Optional decode_sample_stride filter, e.g., 64")
    ap.add_argument("--metric", type=str, default="total_time_s",
                    choices=["total_time_s", "decode_time_s", "prefill_time_s"])
    ap.add_argument("--hw-group", type=str, default=None,
                    help="Optional: choose one group (2npu/npu/halfnpu...). If omitted, plot all groups.")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for png files.")
    args = ap.parse_args()

    if args.model:
        model_family, model_variant, dtype = _parse_model_str(args.model)
    else:
        if not (args.model_family and args.model_variant and args.dtype):
            raise SystemExit("Either --model or (--model-family --model-variant --dtype) must be provided.")
        model_family, model_variant, dtype = args.model_family, args.model_variant, args.dtype

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"--root not found: {root}")

    jsons = list(iter_candidate_jsons(root, args.prefill, args.decode))
    if not jsons:
        raise SystemExit(f"No baseline_compare_{args.prefill}x{args.decode}.json found under: {root}")

    df = build_dataframe(
        jsons,
        model_family=model_family,
        model_variant=model_variant,
        dtype=dtype,
        batch=args.batch,
        prefill=args.prefill,
        decode=args.decode,
        stride=args.stride,
    )

    if df.empty:
        raise SystemExit("No matched jsons after filtering (model/batch/prefill/decode/stride).")

    outdir = Path(args.outdir)
    groups = sorted(df["hw_group"].unique().tolist())
    if args.hw_group is not None:
        if args.hw_group not in groups:
            raise SystemExit(f"--hw-group={args.hw_group} not found. Available: {groups}")
        groups = [args.hw_group]

    print(df.sort_values(["hw_group", "policy", "aim"]).to_string(index=False))

    for g in groups:
        outpath = plot_one_group(
            df,
            hw_group=g,
            metric=args.metric,
            model_family=model_family,
            model_variant=model_variant,
            dtype=dtype,
            batch=args.batch,
            prefill=args.prefill,
            decode=args.decode,
            stride=args.stride,
            outdir=outdir,
        )
        print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
