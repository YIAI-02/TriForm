#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""plot_latency_vs_pim.py

Plot latency vs #PIM(AiM) from baseline_compare_{prefill}x{decode}.json.

只要路径里包含一个形如 `hw_*aim*` 的目录段即可，例如：
  output/experiment_1/hw_2aim_2npu/st64/llama_7b_int8_b1_s64/baseline_compare_512x128.json
  output/experiment_1/hw_4aim_2npu/st64/llama_7b_int8_b1_s64/baseline_compare_512x128.json

输出：pdf (默认输出到当前目录，可用 --output-dir 指定)

=== 绘图模式（默认都会生成；可用参数关闭） ===
1) 分立绘图（保留原方式）：按 (model, batch, prefill, decode, stride) 分组，画每个 hw_group 一张图。
   - 关闭：--no-plot-per-combo

2) Model 平均绘图（更“平均”）：同一个 model 内，跨所有 batch + 所有 prefill_len + 所有 decode_len 求平均，
   再画每个 hw_group 一张图。
   - 关闭：--no-plot-model-avg

3) Global 平均绘图（最“平均”）：跨所有 model + 所有 batch + 所有 prefill_len + 所有 decode_len 求平均，
   对每个 hw_group 画一张图（相当于“一个 npu 值出一个 elbow 的 pim”）。
   - 关闭：--no-plot-global-avg


python plot_latency_vs_pim.py \
  --root ../algorithms/output/experiment_4npu \
  --metric total_time_s \
  --output-dir ../figs/experiment_4npu/experiment_2 \
  --no-plot-per-combo \
  --no-plot-model-avg
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 你指定的配色池
COLOR_POOL = [
    "#000000",
    "#0024FF",
    "#006CFE",
    "#0092FE",
    "#FFBF02",
    "#FE8000",
    "#FF3F04",
]


def _parse_model_str(model: str) -> Tuple[str, str, str]:
    """Parse model string.

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
    """Return AiM count from a hw segment.

    Supports variants such as:
      - hw_2aim_2npu
      - hw_aim_npu / hw_npu_aim
      - hw_npu_4aim
    """
    m = re.search(r"(\d*)aim", hw_seg)
    if not m:
        return None
    g = m.group(1)
    return int(g) if g else 1


def parse_hw_group_from_hw_segment(hw_seg: str) -> str:
    """Return the suffix after removing the AiM token.

    Examples:
      - hw_2aim_2npu    -> 2npu
      - hw_aim_npu      -> npu
      - hw_npu_4aim     -> npu
      - hw_2aim_halfnpu -> halfnpu
    """
    seg2 = hw_seg[3:] if hw_seg.startswith("hw_") else hw_seg
    seg2 = re.sub(r"_?\d*aim_?", "_", seg2)
    return seg2.strip("_")


def elbow_point(x: np.ndarray, y: np.ndarray) -> Optional[int]:
    """Elbow detector for "latency decreases then saturates".

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


def iter_candidate_jsons(root: Path, prefill: Optional[int], decode: Optional[int]) -> Iterable[Path]:
    if prefill is None or decode is None:
        return root.rglob("baseline_compare_*.json")
    pattern = f"baseline_compare_{prefill}x{decode}.json"
    return root.rglob(pattern)


def _to_int_or_none(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def build_dataframe(
    json_paths: List[Path],
    *,
    model_family: Optional[str] = None,
    model_variant: Optional[str] = None,
    dtype: Optional[str] = None,
    batch: Optional[int] = None,
    prefill: Optional[int] = None,
    decode: Optional[int] = None,
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
        cfg_model_family = cfg.get("model_family")
        cfg_model_variant = cfg.get("model_variant")
        cfg_dtype = cfg.get("dtype")
        cfg_batch = _to_int_or_none(cfg.get("batch"))
        cfg_prefill = _to_int_or_none(cfg.get("prefill_len"))
        cfg_decode = _to_int_or_none(cfg.get("decode_len"))
        cfg_stride = _to_int_or_none(cfg.get("decode_sample_stride"))

        if model_family is not None and str(cfg_model_family) != str(model_family):
            continue
        if model_variant is not None and str(cfg_model_variant) != str(model_variant):
            continue
        if dtype is not None and str(cfg_dtype) != str(dtype):
            continue
        if batch is not None and cfg_batch != int(batch):
            continue
        if prefill is not None and cfg_prefill != int(prefill):
            continue
        if decode is not None and cfg_decode != int(decode):
            continue
        if stride is not None and cfg_stride != int(stride):
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

            rows.append(
                {
                    "model_family": cfg_model_family,
                    "model_variant": cfg_model_variant,
                    "dtype": cfg_dtype,
                    "batch": cfg_batch,
                    "prefill_len": cfg_prefill,
                    "decode_len": cfg_decode,
                    "stride": cfg_stride,
                    "aim": aim,
                    "hw_group": hw_group,
                    "policy": policy,
                    "prefill_time_s": r.get("prefill_time_s", np.nan),
                    "decode_time_s": r.get("decode_time_s", np.nan),
                    "total_time_s": r.get("total_time_s", np.nan),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # if duplicates exist, average them while keeping full config keys
    agg_cols = ["prefill_time_s", "decode_time_s", "total_time_s"]
    group_keys = [
        "model_family",
        "model_variant",
        "dtype",
        "batch",
        "prefill_len",
        "decode_len",
        "stride",
        "hw_group",
        "aim",
        "policy",
    ]

    # 注意：pandas groupby 默认 dropna=True，会丢掉 stride=None 的组；这里用 dropna=False 保留。
    df = df.groupby(group_keys, as_index=False, dropna=False)[agg_cols].mean()
    return df


def _summarize_int_values(vals: Iterable, *, max_elems: int = 8) -> str:
    """Compact representation for a set of ints (for titles)."""
    xs: List[int] = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        try:
            xs.append(int(v))
        except Exception:
            continue

    xs = sorted(set(xs))
    if not xs:
        return "{}"
    if len(xs) <= max_elems:
        return "{" + ",".join(map(str, xs)) + "}"
    return f"{{{xs[0]}..{xs[-1]}}} (n={len(xs)})"


def _stride_tag(stride_val: Optional[int]) -> str:
    return f"_s{stride_val}" if stride_val is not None else ""


def plot_latency_vs_aim(
    df: pd.DataFrame,
    *,
    hw_group: str,
    metric: str,
    title: str,
    outpath: Path,
) -> Path:
    """Plot one figure for a given hw_group with multiple policy lines."""

    g = df[df["hw_group"] == hw_group].copy()
    if g.empty:
        raise ValueError(f"No data for hw_group={hw_group}")

    # ensure numeric
    g = g.dropna(subset=["aim", metric])
    if g.empty:
        raise ValueError(f"No valid numeric data for hw_group={hw_group}, metric={metric}")

    policies = sorted(g["policy"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(9, 5))
    elbow_summary = []

    for i, pol in enumerate(policies):
        sub = g[g["policy"] == pol].copy()
        if sub.empty:
            continue

        # ensure 1 point per aim
        sub = sub.groupby(["aim"], as_index=False, dropna=False)[[metric]].mean()
        sub = sub.sort_values("aim")

        x = sub["aim"].to_numpy()
        y = sub[metric].to_numpy()
        if len(x) == 0:
            continue

        color = COLOR_POOL[i % len(COLOR_POOL)]
        ax.plot(x, y, marker="o", linewidth=2, label=pol, color=color)

        elbow = elbow_point(x, y)
        if elbow is not None:
            try:
                y_elbow = float(sub.loc[sub["aim"] == elbow, metric].iloc[0])
            except Exception:
                y_elbow = None
            if y_elbow is not None and np.isfinite(y_elbow):
                ax.scatter([elbow], [y_elbow], marker="*", s=180, color=color, zorder=5)
                elbow_summary.append((pol, elbow, y_elbow))

    ax.set_xlabel("#AiM (PIM count)  [from hw_*aim*]")
    ax.set_ylabel(f"{metric} (s)")
    ax.set_title(title)

    aims_sorted = sorted({int(a) for a in g["aim"].dropna().unique().tolist()})
    ax.set_xticks(aims_sorted)

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(ncol=2, fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder to search (e.g., output/experiment_1)")

    # model filter
    ap.add_argument("--model", type=str, default=None, help="Optional: only plot a specific model like llama_7b_int8")
    ap.add_argument("--model-family", type=str, default=None, help="Optional model_family filter")
    ap.add_argument("--model-variant", type=str, default=None, help="Optional model_variant filter")
    ap.add_argument("--dtype", type=str, default=None, help="Optional dtype filter")

    # raw config filter
    ap.add_argument("--batch", type=int, default=None, help="Optional batch filter")
    ap.add_argument("--prefill", type=int, default=None, help="Optional prefill length filter")
    ap.add_argument("--decode", type=int, default=None, help="Optional decode length filter")
    ap.add_argument("--stride", type=int, default=None, help="Optional decode_sample_stride filter, e.g., 64")

    ap.add_argument(
        "--metric",
        type=str,
        default="total_time_s",
        choices=["total_time_s", "decode_time_s", "prefill_time_s"],
    )
    ap.add_argument(
        "--hw-group",
        type=str,
        default=None,
        help="Optional: choose one group (2npu/npu/halfnpu...). If omitted, plot all groups.",
    )

    # output
    ap.add_argument(
        "--output-dir",
        "--outdir",
        type=str,
        default=".",
        dest="outdir",
        help="Output directory for exported pdf files.",
    )

    # plotting mode switches
    ap.add_argument(
        "--no-plot-per-combo",
        action="store_true",
        help="Disable original per-(batch,prefill,decode) separated plots.",
    )
    ap.add_argument(
        "--no-plot-model-avg",
        action="store_true",
        help="Disable per-model average plots (avg over all batch+prefill+decode).",
    )
    ap.add_argument(
        "--no-plot-global-avg",
        action="store_true",
        help="Disable global average plots (avg over all models + all batch+prefill+decode).",
    )

    # aliases for compatibility (do not show in help)
    ap.add_argument(
        "--no-plot-per-batch",
        action="store_true",
        dest="no_plot_per_combo",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--no-plot-batch-avg",
        action="store_true",
        dest="no_plot_model_avg",
        help=argparse.SUPPRESS,
    )

    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"--root not found: {root}")

    jsons = list(iter_candidate_jsons(root, args.prefill, args.decode))
    if not jsons:
        raise SystemExit(f"No baseline_compare_*.json found under: {root}")

    model_family = model_variant = dtype = None
    if args.model:
        model_family, model_variant, dtype = _parse_model_str(args.model)
    else:
        model_family, model_variant, dtype = args.model_family, args.model_variant, args.dtype

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

    # helper columns
    df = df.copy()
    df["model_id"] = (
        df["model_family"].astype(str)
        + "_"
        + df["model_variant"].astype(str)
        + "_"
        + df["dtype"].astype(str)
    )

    # choose hw groups
    all_hw_groups = sorted(df["hw_group"].dropna().unique().tolist())
    if args.hw_group is not None:
        if args.hw_group not in all_hw_groups:
            raise SystemExit(f"--hw-group={args.hw_group} not found. Available: {all_hw_groups}")
        hw_groups = [args.hw_group]
    else:
        hw_groups = all_hw_groups

    agg_cols = ["prefill_time_s", "decode_time_s", "total_time_s"]

    # ------------------------------------------------------------------
    # 1) Per-combo plots (original style)
    # ------------------------------------------------------------------
    if not args.no_plot_per_combo:
        combo_cols = [
            "model_family",
            "model_variant",
            "dtype",
            "batch",
            "prefill_len",
            "decode_len",
            "stride",
        ]

        for combo_vals, combo_df in df.groupby(combo_cols, dropna=False):
            combo = dict(zip(combo_cols, combo_vals))

            mf = combo.get("model_family")
            mv = combo.get("model_variant")
            dt = combo.get("dtype")
            b = combo.get("batch")
            pf = combo.get("prefill_len")
            dc = combo.get("decode_len")
            st_val = combo.get("stride")

            # skip incomplete combos
            if any(pd.isna(v) for v in [mf, mv, dt, b, pf, dc]):
                continue

            stride_val: Optional[int] = None if pd.isna(st_val) else int(st_val)

            # optionally restrict hw_group
            groups = sorted(combo_df["hw_group"].dropna().unique().tolist())
            if args.hw_group is not None:
                if args.hw_group not in groups:
                    continue
                groups = [args.hw_group]

            print("\n=== Combo (per-config) ===")
            print(
                f"model={mf}_{mv}_{dt} | b={int(b)} | prefill={int(pf)} decode={int(dc)}"
                + (f" | stride={stride_val}" if stride_val is not None else "")
            )

            for g in groups:
                title = f"{mf}_{mv}_{dt} | b{int(b)} | {int(pf)}x{int(dc)}"
                if stride_val is not None:
                    title += f" | stride={stride_val}"
                title += f" | hw_group={g}"

                outpath = (
                    outdir
                    / (
                        f"latency_vs_aim_{mf}_{mv}_{dt}_b{int(b)}_{int(pf)}x{int(dc)}"
                        f"{_stride_tag(stride_val)}_{g}_{args.metric}.pdf"
                    )
                )

                saved = plot_latency_vs_aim(combo_df, hw_group=g, metric=args.metric, title=title, outpath=outpath)
                print(f"Saved: {saved}")

    # ------------------------------------------------------------------
    # 2) Per-model average (avg over ALL batch + prefill + decode)
    # ------------------------------------------------------------------
    if not args.no_plot_model_avg:
        model_avg_keys = [
            "model_family",
            "model_variant",
            "dtype",
            "stride",
            "hw_group",
            "aim",
            "policy",
        ]
        df_model_avg = df.groupby(model_avg_keys, as_index=False, dropna=False)[agg_cols].mean()

        base_cols = ["model_family", "model_variant", "dtype", "stride"]
        for base_vals, base_df in df_model_avg.groupby(base_cols, dropna=False):
            base = dict(zip(base_cols, base_vals))
            mf = base.get("model_family")
            mv = base.get("model_variant")
            dt = base.get("dtype")
            st_val = base.get("stride")
            stride_val: Optional[int] = None if pd.isna(st_val) else int(st_val)

            # collect ranges from RAW df (not averaged) so the title reflects what got averaged
            raw_sel = df[
                (df["model_family"] == mf)
                & (df["model_variant"] == mv)
                & (df["dtype"] == dt)
                & (
                    (df["stride"].isna() & pd.isna(st_val))
                    | (df["stride"] == st_val)
                )
            ]

            b_s = _summarize_int_values(raw_sel["batch"].unique().tolist())
            pf_s = _summarize_int_values(raw_sel["prefill_len"].unique().tolist())
            dc_s = _summarize_int_values(raw_sel["decode_len"].unique().tolist())

            # optionally restrict hw_group
            groups = sorted(base_df["hw_group"].dropna().unique().tolist())
            if args.hw_group is not None:
                if args.hw_group not in groups:
                    continue
                groups = [args.hw_group]

            print("\n=== Model AVG (avg over batch+prefill+decode) ===")
            print(
                f"model={mf}_{mv}_{dt}"
                + (f" | stride={stride_val}" if stride_val is not None else "")
                + f" | batch={b_s} prefill={pf_s} decode={dc_s}"
            )

            for g in groups:
                title = (
                    f"{mf}_{mv}_{dt} | AVG(batch={b_s}, prefill={pf_s}, decode={dc_s})"
                    + (f" | stride={stride_val}" if stride_val is not None else "")
                    + f" | hw_group={g}"
                )

                outpath = (
                    outdir
                    / (
                        f"latency_vs_aim_avg_b_pfdc_{mf}_{mv}_{dt}"
                        f"{_stride_tag(stride_val)}_{g}_{args.metric}.pdf"
                    )
                )

                saved = plot_latency_vs_aim(base_df, hw_group=g, metric=args.metric, title=title, outpath=outpath)
                print(f"Saved: {saved}")

    # ------------------------------------------------------------------
    # 3) Global average (avg over ALL models + batch + prefill + decode)
    # ------------------------------------------------------------------
    if not args.no_plot_global_avg:
        global_avg_keys = ["hw_group", "aim", "policy"]
        df_global_avg = df.groupby(global_avg_keys, as_index=False, dropna=False)[agg_cols].mean()

        # for title summary, compute counts/ranges from raw df
        n_models_total = int(df["model_id"].nunique(dropna=True))
        b_s_total = _summarize_int_values(df["batch"].unique().tolist())
        pf_s_total = _summarize_int_values(df["prefill_len"].unique().tolist())
        dc_s_total = _summarize_int_values(df["decode_len"].unique().tolist())

        print("\n=== GLOBAL AVG (avg over ALL models + batch+prefill+decode) ===")
        print(f"models={n_models_total} | batch={b_s_total} prefill={pf_s_total} decode={dc_s_total}")

        for g in hw_groups:
            title = (
                f"GLOBAL AVG | models={n_models_total} | "
                f"AVG(batch={b_s_total}, prefill={pf_s_total}, decode={dc_s_total})"
                f" | hw_group={g}"
            )

            outpath = outdir / f"latency_vs_aim_globalavg_{g}_{args.metric}.pdf"
            try:
                saved = plot_latency_vs_aim(df_global_avg, hw_group=g, metric=args.metric, title=title, outpath=outpath)
            except ValueError as e:
                print(f"Skip hw_group={g}: {e}")
                continue
            print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
