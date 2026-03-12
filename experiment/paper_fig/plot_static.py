#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只画片，不画点：--point-style none
点改成空心、只描边：--point-style hollow
不画片：--surface-mode none
片的透明度：--surface-alpha 0.10
点大小：--point-size 42
点描边粗细：--point-edge-width 0.6
实心点透明度：--point-alpha 0.95

Typical usage:
    python plot_exp1_static.py ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst64_rst64 \
        --include-models llama_7b qwen_1.8b qwen_14b \
        --mode static \
        --metric total_time_s \
        --point-size 42 \
        --point-edge-width 1\
        --surface-mode convex_hull \
        --point-alpha 0.4 \
        --output ../../figs/exp1/fig_static_brittleness_3d.pdf

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from scipy.spatial import ConvexHull
    HAVE_SCIPY = True
except Exception:
    ConvexHull = None
    HAVE_SCIPY = False


# ============================================================
# Policy styling
# ============================================================

POLICY_ORDER = [
    "pd",
    "ianus",
    "facil",
    "attacc",
    "weights_on_pim",
    "attn_on_pim",
    "heft",
    "hefthint",
    "TIE",
]

POLICY_COLORS = {
    "pd": "#5837a8",
    "ianus": "#a83747",
    "facil": "#3791a8",
    "attacc": "#3760a9",
    "weights_on_pim": "#39a937",
    "attn_on_pim": "#a89a37",
    "heft": "#b07aa1",
    "hefthint": "#b07aa1",
    "TIE": "#7f7f7f",
}

POLICY_LABELS = {
    "pd": "PD",
    "ianus": "IANUS",
    "facil": "FACIL",
    "attacc": "ATTACC",
    "weights_on_pim": "Weights-on-PIM",
    "attn_on_pim": "Attn-on-PIM",
    "heft": "HEFT",
    "hefthint": "HEFTHint",
    "TIE": "Tie",
}

STATIC_POLICIES = {
    "pd",
    "ianus",
    "facil",
    "attacc",
    "weights_on_pim",
    "attn_on_pim",
}

DYNAMIC_POLICIES = {
    "heft",
    "hefthint",
}


# ============================================================
# Helpers
# ============================================================

def normalize_policy(policy: str) -> str:
    policy = str(policy).strip()
    if policy.startswith("algo:"):
        policy = policy.split(":", 1)[1]
    return policy


def model_key(model_family: str, model_variant: str) -> str:
    return f"{str(model_family).lower()}_{str(model_variant).lower()}"


def pretty_model_label(key: str) -> str:
    mapping = {
        "llama_7b": "LLaMA-7B",
        "qwen_1.8b": "Qwen-1.8B",
        "qwen_7b": "Qwen-7B",
        "qwen_14b": "Qwen-14B",
    }
    return mapping.get(key.lower(), key.replace("_", "-"))


def safe_log2(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"All axis values must be positive for log2 transform, got {arr}")
    return np.log2(arr)


def numeric_rank(points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    centered = points - points.mean(axis=0, keepdims=True)
    return int(np.linalg.matrix_rank(centered))


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def infer_model_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Tries to infer model_family and model_variant from any directory / file component.
    Expected examples:
        llama_7b_fp16_b16_s64
        qwen_1.8b_fp16_b4_s64
        qwen_14b_fp16_b2_s64
    """
    patterns = [
        re.compile(r"(?P<family>[A-Za-z0-9.]+)_(?P<variant>[0-9.]+b)_fp\d+_b\d+_s\d+", re.IGNORECASE),
        re.compile(r"(?P<family>[A-Za-z0-9.]+)_(?P<variant>[0-9.]+b)", re.IGNORECASE),
    ]

    for part in reversed(path.parts):
        for pat in patterns:
            m = pat.search(part)
            if m:
                return m.group("family"), m.group("variant")

    return None, None


def get_nested(d: dict, keys: Sequence[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ============================================================
# JSON parsing
# ============================================================

def parse_baseline_compare(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data.get("config", {})
    family = cfg.get("model_family")
    variant = cfg.get("model_variant")
    batch = cfg.get("batch")
    prefill = cfg.get("prefill_len")
    decode = cfg.get("decode_len")

    if family is None or variant is None or batch is None or prefill is None or decode is None:
        family2, variant2 = infer_model_from_path(json_path)
        family = family if family is not None else family2
        variant = variant if variant is not None else variant2

    if family is None or variant is None or batch is None or prefill is None or decode is None:
        raise ValueError(f"Missing config fields in {json_path}")

    records = []
    for res in data.get("results", []):
        pol = normalize_policy(res.get("policy", ""))
        records.append(
            {
                "source": str(json_path),
                "model_key": model_key(family, variant),
                "model_family": family,
                "model_variant": variant,
                "batch": int(batch),
                "prefill": int(prefill),
                "decode": int(decode),
                "policy": pol,
                "prefill_time_s": float(res.get("prefill_time_s", np.nan)),
                "decode_time_s": float(res.get("decode_time_s", np.nan)),
                "total_time_s": float(res.get("total_time_s", np.nan)),
            }
        )
    return records


def parse_best_summary(json_path: Path) -> Optional[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    policy = normalize_policy(data.get("policy", ""))
    cfg = data.get("config", {})
    batch = cfg.get("batch")
    prefill = cfg.get("prefill_len")
    decode = cfg.get("decode_len")

    family = cfg.get("model_family")
    variant = cfg.get("model_variant")
    if family is None or variant is None:
        family2, variant2 = infer_model_from_path(json_path)
        family = family if family is not None else family2
        variant = variant if variant is not None else variant2

    if family is None or variant is None or batch is None or prefill is None or decode is None:
        return None

    best_times = data.get("best_times", {})
    return {
        "source": str(json_path),
        "model_key": model_key(family, variant),
        "model_family": family,
        "model_variant": variant,
        "batch": int(batch),
        "prefill": int(prefill),
        "decode": int(decode),
        "policy": policy,
        "prefill_time_s": float(best_times.get("prefill", np.nan)),
        "decode_time_s": float(best_times.get("decode", np.nan)),
        "total_time_s": float(best_times.get("total", np.nan)),
    }


def collect_records(root: Path) -> pd.DataFrame:
    baseline_files = sorted(
        p for p in root.rglob("baseline_compare_*.json")
        if "__MACOSX" not in str(p)
    )

    records: List[dict] = []

    if baseline_files:
        for path in baseline_files:
            try:
                records.extend(parse_baseline_compare(path))
            except Exception as e:
                print(f"[warn] skip {path}: {e}")
    else:
        summary_files = sorted(
            p for p in root.rglob("best_summary_*.json")
            if "__MACOSX" not in str(p)
        )
        for path in summary_files:
            try:
                rec = parse_best_summary(path)
                if rec is not None:
                    records.append(rec)
            except Exception as e:
                print(f"[warn] skip {path}: {e}")

    if not records:
        raise FileNotFoundError(
            f"No usable JSON found under {root}. "
            f"Expected baseline_compare_*.json or best_summary_*.json."
        )

    df = pd.DataFrame.from_records(records)

    needed_cols = [
        "model_key",
        "batch",
        "prefill",
        "decode",
        "policy",
        "prefill_time_s",
        "decode_time_s",
        "total_time_s",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Parsed dataframe missing columns: {missing}")

    return df


# ============================================================
# Winner selection
# ============================================================

def choose_policy_subset(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = mode.lower()
    if mode == "static":
        return df[df["policy"].isin(STATIC_POLICIES)].copy()
    if mode == "dynamic":
        return df[df["policy"].isin(DYNAMIC_POLICIES)].copy()
    if mode == "all":
        return df.copy()
    raise ValueError(f"Unknown mode: {mode}")


def compute_winners(
    df: pd.DataFrame,
    metric: str = "total_time_s",
    mode: str = "static",
    tie_tol: float = 1e-12,
    tie_mode: str = "mixed",
) -> pd.DataFrame:
    df = choose_policy_subset(df, mode=mode)

    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not found in dataframe columns")

    group_cols = ["model_key", "batch", "prefill", "decode"]
    rows = []

    for key, g in df.groupby(group_cols, sort=True):
        vals = g[metric].astype(float).to_numpy()
        if len(vals) == 0 or np.all(np.isnan(vals)):
            continue

        min_val = np.nanmin(vals)
        winners = sorted(
            g.loc[np.isclose(vals, min_val, atol=tie_tol, rtol=0.0), "policy"].tolist()
        )

        if not winners:
            continue

        if len(winners) == 1:
            winner_label = winners[0]
            tie_count = 1
        else:
            tie_count = len(winners)
            if tie_mode == "first":
                winner_label = winners[0]
            elif tie_mode == "join":
                winner_label = "/".join(winners)
            else:
                winner_label = "TIE"

        row = dict(zip(group_cols, key))
        row.update(
            {
                "winner": winner_label,
                "winner_time_s": float(min_val),
                "tied_policies": "/".join(winners),
                "tie_count": tie_count,
            }
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


# ============================================================
# Plotting
# ============================================================

def add_convex_hull(
    ax,
    pts_xyz: np.ndarray,
    color: str,
    alpha: float = 0.10,
) -> None:
    """
    Draw a translucent 3D convex hull around points, if possible.
    Requires scipy.spatial.ConvexHull.
    """
    if not HAVE_SCIPY:
        return
    if pts_xyz is None or len(pts_xyz) < 4:
        return

    pts = np.unique(np.asarray(pts_xyz, dtype=float), axis=0)
    if len(pts) < 4:
        return

    # Need full 3D volume; coplanar / collinear point sets are skipped.
    if numeric_rank(pts) < 3:
        return

    try:
        hull = ConvexHull(pts)
    except Exception:
        return

    faces = [pts[simplex] for simplex in hull.simplices]
    poly = Poly3DCollection(
        faces,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zsort="average",
    )
    ax.add_collection3d(poly)


def apply_ticks(ax, xs: Sequence[int], ys: Sequence[int], zs: Sequence[int]) -> None:
    ux = sorted(set(int(v) for v in xs))
    uy = sorted(set(int(v) for v in ys))
    uz = sorted(set(int(v) for v in zs))

    ax.set_xticks(safe_log2(ux))
    ax.set_xticklabels([str(v) for v in ux], fontsize=9)

    ax.set_yticks(safe_log2(uy))
    ax.set_yticklabels([str(v) for v in uy], fontsize=9)

    ax.set_zticks(safe_log2(uz))
    ax.set_zticklabels([str(v) for v in uz], fontsize=9)


def style_3d_axes(ax, elev: float, azim: float) -> None:
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # slightly transparent white panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis.set_pane_color((1.0, 1.0, 1.0, 0.92))
        except Exception:
            pass


def scatter_policy_points(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    color: str,
    point_style: str = "filled",
    point_size: float = 42.0,
    point_alpha: float = 0.96,
    point_edge_width: float = 0.35,
) -> None:
    if point_style == "none":
        return

    if point_style == "filled":
        ax.scatter(
            x,
            y,
            z,
            s=point_size,
            c=color,
            alpha=point_alpha,
            edgecolors="black",
            linewidths=point_edge_width,
            depthshade=False,
        )
        return

    if point_style == "hollow":
        ax.scatter(
            x,
            y,
            z,
            s=point_size,
            facecolors="none",
            edgecolors=color,
            linewidths=point_edge_width,
            alpha=1.0,
            depthshade=False,
        )
        return

    raise ValueError(f"Unknown point_style: {point_style}")


def make_legend_handles(
    shown_policies: List[str],
    point_style: str,
    surface_mode: str,
) -> List:
    handles = []

    for pol in shown_policies:
        color = POLICY_COLORS.get(pol, "#7f7f7f")
        label = POLICY_LABELS.get(pol, pol)

        if point_style == "none":
            if surface_mode == "convex_hull":
                handles.append(Patch(facecolor=color, edgecolor="none", alpha=0.35, label=label))
            else:
                handles.append(Patch(facecolor=color, edgecolor="none", alpha=0.90, label=label))
        elif point_style == "hollow":
            handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.0,
                    markersize=8,
                    label=label,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markeredgewidth=0.35,
                    markersize=8,
                    label=label,
                )
            )

    return handles


def make_figure(
    winners: pd.DataFrame,
    output: Path,
    include_models: Optional[List[str]] = None,
    surface_mode: str = "convex_hull",
    surface_alpha: float = 0.10,
    point_style: str = "filled",
    point_size: float = 42.0,
    point_alpha: float = 0.96,
    point_edge_width: float = 0.35,
    elev: float = 24.0,
    azim: float = -58.0,
    title: Optional[str] = None,
    dpi: int = 300,
) -> None:
    if include_models:
        model_order = [m for m in include_models if m in set(winners["model_key"])]
    else:
        preferred = ["llama_7b", "qwen_1.8b", "qwen_14b", "qwen_7b"]
        present = unique_keep_order(winners["model_key"].tolist())
        model_order = [m for m in preferred if m in present] + [m for m in present if m not in preferred]

    if not model_order:
        raise ValueError("No models left to plot after filtering.")

    n = len(model_order)
    fig = plt.figure(figsize=(5.6 * n, 5.7), constrained_layout=False)

    winners = winners.copy()
    winners["x"] = safe_log2(winners["batch"])
    winners["y"] = safe_log2(winners["decode"])
    winners["z"] = safe_log2(winners["prefill"])

    shown_policies: List[str] = []

    for i, model in enumerate(model_order, start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        sub = winners[winners["model_key"] == model].copy()
        if sub.empty:
            continue

        ordered_policies = POLICY_ORDER + sorted(set(sub["winner"]) - set(POLICY_ORDER))
        ordered_policies = unique_keep_order(ordered_policies)

        for pol in ordered_policies:
            ps = sub[sub["winner"] == pol]
            if ps.empty:
                continue

            shown_policies.append(pol)
            color = POLICY_COLORS.get(pol, "#7f7f7f")

            x = ps["x"].to_numpy()
            y = ps["y"].to_numpy()
            z = ps["z"].to_numpy()

            if surface_mode == "convex_hull":
                add_convex_hull(
                    ax,
                    np.column_stack([x, y, z]),
                    color=color,
                    alpha=surface_alpha,
                )

            scatter_policy_points(
                ax,
                x, y, z,
                color=color,
                point_style=point_style,
                point_size=point_size,
                point_alpha=point_alpha,
                point_edge_width=point_edge_width,
            )

        apply_ticks(ax, sub["batch"], sub["decode"], sub["prefill"])
        ax.set_xlabel("Batch size", labelpad=8)
        ax.set_ylabel("Decode length", labelpad=10)
        ax.set_zlabel("Prefill length", labelpad=10)
        ax.set_title(pretty_model_label(model), fontsize=13, pad=12)
        style_3d_axes(ax, elev=elev, azim=azim)

    if title:
        fig.suptitle(title, y=0.98, fontsize=15)

    shown_policies = unique_keep_order(shown_policies)
    legend_handles = make_legend_handles(shown_policies, point_style=point_style, surface_mode=surface_mode)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), max(3, len(legend_handles))),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    plt.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.14, wspace=0.05)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")

    # also save sibling PDF/SVG for paper use
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")

    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a 3D static-policy brittleness figure from a directory of JSON files."
    )
    parser.add_argument(
        "json_root",
        type=Path,
        help="Root directory containing JSON files.",
    )
    parser.add_argument(
        "--mode",
        default="static",
        choices=["static", "dynamic", "all"],
        help="Policy set to compare.",
    )
    parser.add_argument(
        "--metric",
        default="total_time_s",
        choices=["prefill_time_s", "decode_time_s", "total_time_s"],
        help="Latency metric used to choose the best policy.",
    )
    parser.add_argument(
        "--tie-mode",
        default="mixed",
        choices=["mixed", "first", "join"],
        help=(
            "'mixed': ties are colored as gray 'Tie'; "
            "'first': pick the lexicographically first winner; "
            "'join': keep joined labels in CSV/plot."
        ),
    )
    parser.add_argument(
        "--include-models",
        nargs="*",
        default=None,
        help="Optional model keys to include, e.g. llama_7b qwen_1.8b qwen_14b",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("static_brittleness_3d.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save the winner table as CSV.",
    )
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument(
        "--surface-mode",
        default="convex_hull",
        choices=["convex_hull", "none"],
        help="How to connect points.",
    )
    parser.add_argument(
        "--surface-alpha",
        type=float,
        default=0.10,
        help="Transparency of the surface patches.",
    )

    parser.add_argument(
        "--point-style",
        default="filled",
        choices=["filled", "hollow", "none"],
        help="Point drawing style.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=42.0,
        help="Point size.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.96,
        help="Alpha for filled points.",
    )
    parser.add_argument(
        "--point-edge-width",
        type=float,
        default=0.35,
        help="Edge width for points.",
    )

    parser.add_argument("--elev", type=float, default=24.0, help="3D camera elevation.")
    parser.add_argument("--azim", type=float, default=-58.0, help="3D camera azimuth.")
    parser.add_argument("--title", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.surface_mode == "convex_hull" and not HAVE_SCIPY:
        print("[warn] scipy not found; convex hull surfaces will be skipped. Install scipy to enable surfaces.")

    df = collect_records(args.json_root)

    winners = compute_winners(
        df,
        metric=args.metric,
        mode=args.mode,
        tie_mode=args.tie_mode,
    )

    if args.include_models:
        winners = winners[winners["model_key"].isin(args.include_models)].copy()
        if winners.empty:
            raise ValueError("No data left after --include-models filtering.")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        winners.to_csv(args.csv, index=False)

    make_figure(
        winners=winners,
        output=args.output,
        include_models=args.include_models,
        surface_mode=args.surface_mode,
        surface_alpha=args.surface_alpha,
        point_style=args.point_style,
        point_size=args.point_size,
        point_alpha=args.point_alpha,
        point_edge_width=args.point_edge_width,
        elev=args.elev,
        azim=args.azim,
        title=args.title,
        dpi=args.dpi,
    )

    print(f"[ok] figure saved to {args.output}")
    print(f"[ok] rows plotted: {len(winners)}")
    print(f"[ok] models: {sorted(winners['model_key'].unique().tolist())}")
    print(f"[ok] policies shown: {sorted(winners['winner'].unique().tolist())}")


if __name__ == "__main__":
    main()