#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  python analyze_heft_speedup.py \
  --root ../algorithms/output/lens_eval_sweep \
  --out ../algorithms/output/analysis_out \
  --baseline ALL
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# Optional ML (for "which variable is most related" after controlling others)
try:
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Optional plotting
try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

PALETTE = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
           "#FF3F04", "#FE5D00", "#FE8000", "#FFBF02"]

DIVERGING_SPEEDUP = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
                     "#FFBF02", "#FE8000", "#FE5D00", "#FF3F04"]


# --------------------------
# Parsing helpers
# --------------------------
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def add_derived_features(row: dict) -> dict:
    """Ensure total_tokens / ratios / deltas exist for EVERY row (including best_baseline)."""
    prefill = _to_float(row.get("prefill_len"))
    decode  = _to_float(row.get("decode_len"))

    if np.isfinite(prefill) and np.isfinite(decode):
        tt = prefill + decode
        row["total_tokens"] = tt
        if tt != 0:
            row["decode_ratio"]  = decode / tt
            row["prefill_ratio"] = prefill / tt
        else:
            row["decode_ratio"] = np.nan
            row["prefill_ratio"] = np.nan
    else:
        row["total_tokens"] = np.nan
        row["decode_ratio"] = np.nan
        row["prefill_ratio"] = np.nan

    bt = _to_float(row.get("baseline_total_time_s"))
    ht = _to_float(row.get("heft_total_time_s"))
    bpf = _to_float(row.get("baseline_prefill_time_s"))
    hpf = _to_float(row.get("heft_prefill_time_s"))
    bdc = _to_float(row.get("baseline_decode_time_s"))
    hdc = _to_float(row.get("heft_decode_time_s"))

    row["delta_total_s"]  = bt - ht if (np.isfinite(bt) and np.isfinite(ht)) else np.nan
    row["delta_prefill_s"]= bpf - hpf if (np.isfinite(bpf) and np.isfinite(hpf)) else np.nan
    row["delta_decode_s"] = bdc - hdc if (np.isfinite(bdc) and np.isfinite(hdc)) else np.nan

    dt = row.get("delta_total_s", np.nan)
    dd = row.get("delta_decode_s", np.nan)
    if np.isfinite(dt) and dt != 0 and np.isfinite(dd):
        row["gain_from_decode_ratio"] = dd / dt
    else:
        row["gain_from_decode_ratio"] = np.nan

    return row

def parse_algo_name(policy: str) -> str:
    """policy like 'algo:heft' -> 'heft'"""
    if not isinstance(policy, str):
        return ""
    return policy.split(":")[-1] if ":" in policy else policy


def parse_model_size_b(model_variant: Optional[str]) -> float:
    """
    Heuristic: '8b' -> 8, '1.8b' -> 1.8, '8x7b' -> 56
    If unparseable -> NaN
    """
    if not model_variant:
        return np.nan
    s = str(model_variant).lower().strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)b", s)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)b", s)
    if m:
        return float(m.group(1))
    return np.nan


def infer_hardware(cfg: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """
    Your rule:
      - hardware_json contains 'large'/'small' -> PIM size
      - contains 'npu' -> NPU compute is doubled
      - none -> normal
    Also try to infer 'stXX' from path segment like /st64/
    """
    hw_path = str(cfg.get("hardware_json", "") or "")
    hw_name = os.path.basename(hw_path).lower()
    fp_low = file_path.lower()

    flag_large = ("large" in hw_name) or ("large" in fp_low)
    flag_small = ("small" in hw_name) or ("small" in fp_low)
    if flag_large and not flag_small:
        pim_size = "large"
    elif flag_small and not flag_large:
        pim_size = "small"
    else:
        pim_size = "normal"

    # Any 'npu' in hw json name/path -> treat as doubled NPU by your definition
    npu_double = ("npu" in hw_name) or ("_npu" in fp_low) or ("/npu" in fp_low) or re.search(r"\bnpu\b", fp_low) is not None

    # Parse stXX if exists (e.g. .../st64/...)
    st = None
    for part in Path(file_path).parts:
        m = re.fullmatch(r"st(\d+)", part.lower())
        if m:
            st = int(m.group(1))
            break

    hardware_tag = f"pim={pim_size}|npu={'2x' if npu_double else '1x'}|st={st if st is not None else 'NA'}"
    return {
        "hardware_json": hw_path,
        "pim_size": pim_size,
        "npu_double": int(npu_double),
        "st": st,
        "hardware_tag": hardware_tag,
    }


def pick_best_per_algo(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    If one file contains multiple entries for same algo (different strategies),
    we keep the one with minimum total_time_s.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in results:
        algo = parse_algo_name(r.get("policy", ""))
        if not algo:
            continue
        cur = best.get(algo)
        if cur is None or float(r.get("total_time_s", np.inf)) < float(cur.get("total_time_s", np.inf)):
            best[algo] = r
    return best


def safe_div(a, b) -> float:
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return np.nan
    if b == 0:
        return np.nan
    return a / b


def load_baseline_compare(file_path: str, add_best_baseline_row: bool = True) -> List[Dict[str, Any]]:
    """
    Returns rows: one per baseline algo vs HEFT
    """
    with open(file_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    cfg = j.get("config", {})
    results = j.get("results", [])
    best = pick_best_per_algo(results)

    if "heft" not in best:
        return []

    heft = best["heft"]

    base_meta = {
        "file_path": file_path,
        "result_dir": cfg.get("result_dir"),

        "model_family": cfg.get("model_family"),
        "model_variant": cfg.get("model_variant"),
        "model_size_b": parse_model_size_b(cfg.get("model_variant")),
        "dtype": cfg.get("dtype"),

        "batch": cfg.get("batch"),
        "prefill_len": cfg.get("prefill_len"),
        "decode_len": cfg.get("decode_len"),
        "decode_sample_stride": cfg.get("decode_sample_stride"),

        "heft_strategy": heft.get("pim_strategy"),
        "heft_prefill_time_s": heft.get("prefill_time_s"),
        "heft_decode_time_s": heft.get("decode_time_s"),
        "heft_total_time_s": heft.get("total_time_s"),
    }
    base_meta.update(infer_hardware(cfg, file_path))

    rows: List[Dict[str, Any]] = []

    # Pairwise: HEFT vs each non-heft algo in file
    for algo, r in best.items():
        if algo == "heft":
            continue
        row = dict(base_meta)
        row.update({
            "baseline_algo": algo,
            "baseline_strategy": r.get("pim_strategy"),
            "baseline_prefill_time_s": r.get("prefill_time_s"),
            "baseline_decode_time_s": r.get("decode_time_s"),
            "baseline_total_time_s": r.get("total_time_s"),
        })

        row["speedup_total"] = safe_div(row["baseline_total_time_s"], row["heft_total_time_s"])
        row["speedup_prefill"] = safe_div(row["baseline_prefill_time_s"], row["heft_prefill_time_s"])
        row["speedup_decode"] = safe_div(row["baseline_decode_time_s"], row["heft_decode_time_s"])

        # extra useful features
        prefill = float(row["prefill_len"]) if row["prefill_len"] is not None else np.nan
        decode = float(row["decode_len"]) if row["decode_len"] is not None else np.nan
        row["total_tokens"] = prefill + decode if np.isfinite(prefill) and np.isfinite(decode) else np.nan
        row["decode_ratio"] = safe_div(decode, row["total_tokens"])
        row["prefill_ratio"] = safe_div(prefill, row["total_tokens"])

        # where the gain comes from (time delta split)
        row["delta_total_s"] = safe_div(float(row["baseline_total_time_s"]) - float(row["heft_total_time_s"]), 1.0)
        row["delta_prefill_s"] = safe_div(float(row["baseline_prefill_time_s"]) - float(row["heft_prefill_time_s"]), 1.0)
        row["delta_decode_s"] = safe_div(float(row["baseline_decode_time_s"]) - float(row["heft_decode_time_s"]), 1.0)
        row["gain_from_decode_ratio"] = safe_div(row["delta_decode_s"], row["delta_total_s"])  # ~1 => mostly decode contributed
        row = add_derived_features(row)
        rows.append(row)

    # Optional: HEFT vs "best baseline" (min total within cfg.baselines)
    if add_best_baseline_row:
        baselines = cfg.get("baselines", [])
        candidate = []
        for b in baselines:
            if b in best:
                candidate.append((b, best[b]))
        if candidate:
            bname, br = min(candidate, key=lambda x: float(x[1].get("total_time_s", np.inf)))
            row = dict(base_meta)
            row.update({
                "baseline_algo": "best_baseline",
                "best_baseline_name": bname,
                "baseline_strategy": br.get("pim_strategy"),
                "baseline_prefill_time_s": br.get("prefill_time_s"),
                "baseline_decode_time_s": br.get("decode_time_s"),
                "baseline_total_time_s": br.get("total_time_s"),
            })
            row["speedup_total"] = safe_div(row["baseline_total_time_s"], row["heft_total_time_s"])
            row["speedup_prefill"] = safe_div(row["baseline_prefill_time_s"], row["heft_prefill_time_s"])
            row["speedup_decode"] = safe_div(row["baseline_decode_time_s"], row["heft_decode_time_s"])
            row = add_derived_features(row)
            rows.append(row)

    return rows


# --------------------------
# Analysis helpers
# --------------------------

def spearman_corrs(df: pd.DataFrame, ycol: str, cols: List[str]) -> pd.DataFrame:
    """
    计算每个数值与speedup_total的相关系数（Spearman correlation）。
    取值 [-1, 1]
    只看“单调关系”（不要求线性）
    abs_spearman 越大，说明越“相关”（但不代表因果）
    """
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        if df[c].nunique(dropna=True) < 2 or df[ycol].nunique(dropna=True) < 2:
            corr = np.nan
            n = int(df[[c, ycol]].dropna().shape[0])
        else:
            corr = df[[c, ycol]].corr(method="spearman").iloc[0, 1]
            n = int(df[[c, ycol]].dropna().shape[0])
        out.append({
            "feature": c,
            "spearman": corr,
            "abs_spearman": abs(corr) if pd.notna(corr) else np.nan,
            "n": n,
        })
    return pd.DataFrame(out).sort_values("abs_spearman", ascending=False)


def eta_squared(df: pd.DataFrame, cat_col: str, ycol: str) -> float:
    """
    How much variance in y is explained by grouping on cat_col.
    0 ~ no effect, closer to 1 ~ strong categorical separation.
    eta² ≈ 0：不同类别之间 speedup 没差（或差很小）
    eta² 越接近 1：不同类别差异越大（category 很重要）
    """
    if cat_col not in df.columns:
        return np.nan
    y = pd.to_numeric(df[ycol], errors="coerce").values
    mask = np.isfinite(y)
    y = y[mask]
    if len(y) < 2:
        return np.nan
    overall = float(np.mean(y))
    ss_total = float(np.sum((y - overall) ** 2))
    if ss_total == 0:
        return 0.0

    ss_within = 0.0
    for _, g in df.loc[mask].groupby(cat_col):
        yg = pd.to_numeric(g[ycol], errors="coerce").values
        yg = yg[np.isfinite(yg)]
        if len(yg) == 0:
            continue
        m = float(np.mean(yg))
        ss_within += float(np.sum((yg - m) ** 2))

    return 1.0 - ss_within / ss_total

def filter_nonempty_cols(df: pd.DataFrame, cols: list) -> list:
    keep = []
    for c in cols:
        if c in df.columns and df[c].notna().any():
            keep.append(c)
    return keep

def cv_r2_score(X: pd.DataFrame, y: np.ndarray,
                numeric_cols: List[str], categorical_cols: List[str],
                n_splits: int = 5, random_state: int = 0) -> float:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    model = Ridge(alpha=1.0, random_state=random_state)

    numeric_cols = filter_nonempty_cols(X, numeric_cols)
    categorical_cols = filter_nonempty_cols(X, categorical_cols)

    if len(numeric_cols) == 0 and len(categorical_cols) == 0:
        return np.nan
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    n = min(n_splits, len(y))
    if n < 2:
        return np.nan
    cv = KFold(n_splits=n, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def drop_column_importance(df: pd.DataFrame, ycol: str,
                           numeric_cols: List[str], categorical_cols: List[str],
                           group_map: Dict[str, List[str]],
                           n_splits: int = 5, random_state: int = 0) -> pd.DataFrame:
    """
    控制其它变量后，哪个变量最关键
    用 Ridge 回归去预测 log(speedup_total)
    先算 base R²（交叉验证平均）
    然后把一组特征（比如只去掉 decode_len 相关）删掉，再算 R²
    drop_in_r2 = base_r2 - r2_without_group
    越大：说明这组特征对解释 speedup 更关键（在控制其它变量后）
    """
    y = pd.to_numeric(df[ycol], errors="coerce").values
    mask = np.isfinite(y) & (y > 0)
    df2 = df.loc[mask].copy()
    if len(df2) < 5:
        return pd.DataFrame([{
            "group": "N/A",
            "base_r2": np.nan,
            "r2_without_group": np.nan,
            "drop_in_r2": np.nan,
            "cols": "",
            "n": len(df2),
            "note": "too_few_samples_for_ml"
        }])

    ylog = np.log(pd.to_numeric(df2[ycol], errors="coerce").values)

    # Keep only existing columns
    num = [c for c in numeric_cols if c in df2.columns]
    cat = [c for c in categorical_cols if c in df2.columns]
    X = df2[num + cat].copy()

    base = cv_r2_score(X, ylog, num, cat, n_splits=n_splits, random_state=random_state)

    rows = []
    for gname, cols_to_drop in group_map.items():
        cols_to_drop = [c for c in cols_to_drop if c in X.columns]
        num2 = [c for c in num if c not in cols_to_drop]
        cat2 = [c for c in cat if c not in cols_to_drop]
        X2 = X[num2 + cat2].copy()
        score = cv_r2_score(X2, ylog, num2, cat2, n_splits=n_splits, random_state=random_state)
        rows.append({
            "group": gname,
            "base_r2": base,
            "r2_without_group": score,
            "drop_in_r2": (base - score) if (np.isfinite(base) and np.isfinite(score)) else np.nan,
            "cols": ",".join(cols_to_drop),
            "n": len(df2),
        })

    return pd.DataFrame(rows).sort_values("drop_in_r2", ascending=False)


def save_basic_plots(df: pd.DataFrame, out_dir: Path, title_prefix: str):
    if not MPL_OK or df.empty:
        return

    # Scatter: speedup_total vs decode_len / prefill_len
    for xcol in ["decode_len", "prefill_len", "batch"]:
        if xcol not in df.columns:
            continue
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df["speedup_total"], errors="coerce")
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            continue

        plt.figure()
        plt.scatter(x[m], y[m])
        plt.xlabel(xcol)
        plt.ylabel("speedup_total (baseline/heft)")
        plt.title(f"{title_prefix} | speedup_total vs {xcol}")
        plt.grid(True, linestyle="--", alpha=0.3)
        # log scale helps when values are 1/32/128/1024 etc
        if x[m].nunique() > 3:
            try:
                plt.xscale("log", base=2)
            except Exception:
                plt.xscale("log")
        plt.tight_layout()
        plt.savefig(out_dir / f"{title_prefix}_scatter_speedup_vs_{xcol}.png", dpi=200)
        plt.close()

    # Heatmap pivot: prefill_len x decode_len
    if "prefill_len" in df.columns and "decode_len" in df.columns:
        pv = df.pivot_table(index="prefill_len", columns="decode_len",
                            values="speedup_total", aggfunc="mean")
        if pv.size > 1:
            arr = pv.values.astype(float)

            # log2 so that 0 == speedup 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                arr = np.log2(arr)

            arr = np.ma.masked_invalid(arr)
            finite = arr[~arr.mask]
            if finite.size == 0:
                return

            # Build symmetric-ish, finite bounds for the colorbar to avoid +/-inf
            vmin, vmax = np.nanpercentile(finite, [5, 95])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                return
            vmax_abs = max(abs(vmin), abs(vmax))
            vmin, vmax = -vmax_abs, vmax_abs

            # Continuous diverging colormap for smoother gradient
            cmap = LinearSegmentedColormap.from_list("speedup_div", DIVERGING_SPEEDUP, N=256)

            plt.figure()
            plt.imshow(arr, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label="log2(mean speedup_total)")
            plt.xticks(range(len(pv.columns)), [str(c) for c in pv.columns], rotation=30, ha="right")
            plt.yticks(range(len(pv.index)), [str(i) for i in pv.index])
            plt.xlabel("decode_len")
            plt.ylabel("prefill_len")
            plt.title(f"{title_prefix} | mean speedup_total heatmap")
            plt.tight_layout()
            plt.savefig(out_dir / f"{title_prefix}_heatmap_speedup_prefill_decode.png", dpi=200)
            plt.close()



# --------------------------
# Main
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root dir containing results (will search baseline_compare_*.json recursively)")
    ap.add_argument("--out", type=str, default="heft_analysis_out", help="Output directory")
    ap.add_argument("--baseline", type=str, default="ALL",
                    help="Which baseline algo to analyze (e.g. 'ianus' or 'ianus,neupims'). Use ALL for all.")
    ap.add_argument("--topk", type=int, default=30, help="Top-K cases to export for each baseline")
    ap.add_argument("--no-ml", action="store_true", help="Disable ML drop-column importance")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("baseline_compare_*.json"))
    if not files:
        raise SystemExit(f"No baseline_compare_*.json found under: {root}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            rows.extend(load_baseline_compare(str(fp), add_best_baseline_row=True))
        except Exception as e:
            print(f"[WARN] failed to load {fp}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Parsed 0 rows. (Maybe files don't contain 'heft' in results?)")

    # Normalize types quickly
    for c in ["batch", "prefill_len", "decode_len", "st", "npu_double", "model_size_b",
              "speedup_total", "speedup_decode", "speedup_prefill",
              "heft_total_time_s", "baseline_total_time_s",
              "heft_decode_time_s", "baseline_decode_time_s",
              "heft_prefill_time_s", "baseline_prefill_time_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Save raw table
    df.to_csv(out_dir / "all_pairs_speedup.csv", index=False)
    print(f"[OK] wrote: {out_dir/'all_pairs_speedup.csv'} (rows={len(df)})")

    # Global summary
    summary = df.groupby("baseline_algo")["speedup_total"].agg(
        n="count",
        mean="mean",
        median="median",
        p90=lambda x: np.nanpercentile(x, 90),
        max="max",
        min="min",
    ).sort_values("mean", ascending=False)
    summary.to_csv(out_dir / "summary_by_baseline.csv")
    print(f"[OK] wrote: {out_dir/'summary_by_baseline.csv'}")

    # Filter baselines
    if args.baseline.strip().upper() == "ALL":
        baselines = sorted(df["baseline_algo"].dropna().unique().tolist())
    else:
        baselines = [b.strip() for b in args.baseline.split(",") if b.strip()]

    # Analysis per baseline
    numeric_cols = [
        "batch", "prefill_len", "decode_len",
        "total_tokens", "decode_ratio",
        "model_size_b",
        "npu_double", "st",
    ]
    categorical_cols = [
        "pim_size", "hardware_tag",
        "model_family", "model_variant", "dtype",
        # strategies sometimes matter
        "heft_strategy", "baseline_strategy",
    ]

    group_map = {
        # exactly what you asked to compare
        "batch": ["batch"],
        "prefill_len": ["prefill_len"],
        "decode_len": ["decode_len"],
        "hardware": ["hardware_tag", "pim_size", "npu_double", "st"],
        "model": ["model_family", "model_variant", "model_size_b", "dtype"],
        # optional extra groups
        "ratios": ["total_tokens", "decode_ratio"],
        "strategy": ["heft_strategy", "baseline_strategy"],
    }

    effects_rows = []

    for b in baselines:
        sub = df[df["baseline_algo"] == b].copy()
        if sub.empty:
            continue

        btag = re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(b))
        bdir = out_dir / f"baseline={btag}"
        bdir.mkdir(parents=True, exist_ok=True)

        # Top-K best cases
        topk = sub.sort_values("speedup_total", ascending=False).head(args.topk)
        topk.to_csv(bdir / "topk_speedup_total.csv", index=False)

        # Spearman correlations (numeric)
        corr = spearman_corrs(sub, "speedup_total", [c for c in numeric_cols if c in sub.columns])
        corr.to_csv(bdir / "spearman_corr_numeric.csv", index=False)

        # Categorical effect sizes (eta^2)
        for cat in ["hardware_tag", "pim_size", "npu_double", "model_family", "model_variant", "dtype"]:
            if cat not in sub.columns:
                continue
            eff = eta_squared(sub, cat, "speedup_total")
            effects_rows.append({
                "baseline_algo": b,
                "cat": cat,
                "eta_squared": eff,
                "n": len(sub),
            })

        # ML: drop-column importance (controls other vars)
        if (not args.no_ml) and SKLEARN_OK:
            imp = drop_column_importance(sub, "speedup_total",
                                        numeric_cols=numeric_cols,
                                        categorical_cols=categorical_cols,
                                        group_map=group_map,
                                        n_splits=5,
                                        random_state=0)
            imp.to_csv(bdir / "drop_column_importance_ridge.csv", index=False)

        # Plots
        if not args.no_plots:
            save_basic_plots(sub, bdir, title_prefix=f"baseline={btag}")

        # Also save group means to help interpret "when best"
        # (1) by hardware
        if "hardware_tag" in sub.columns:
            sub.groupby("hardware_tag")["speedup_total"].mean().sort_values(ascending=False)\
                .to_csv(bdir / "mean_speedup_by_hardware_tag.csv")
        # (2) by model
        if "model_variant" in sub.columns:
            sub.groupby("model_variant")["speedup_total"].mean().sort_values(ascending=False)\
                .to_csv(bdir / "mean_speedup_by_model_variant.csv")

        # (3) by (batch, prefill, decode)
        sub.groupby(["batch", "prefill_len", "decode_len"])["speedup_total"].mean().reset_index()\
            .sort_values("speedup_total", ascending=False)\
            .to_csv(bdir / "mean_speedup_by_batch_prefill_decode.csv", index=False)

        print(f"[OK] baseline={b} -> {bdir}")

    effects = pd.DataFrame(effects_rows)
    if not effects.empty:
        effects.to_csv(out_dir / "eta_squared_categorical_effects.csv", index=False)
        print(f"[OK] wrote: {out_dir/'eta_squared_categorical_effects.csv'}")

    print("\nDONE.")
    print(f"Outputs in: {out_dir.resolve()}")
    print("Key files:")
    print("  - all_pairs_speedup.csv")
    print("  - summary_by_baseline.csv")
    print("  - baseline=<algo>/topk_speedup_total.csv")
    print("  - baseline=<algo>/spearman_corr_numeric.csv")
    if SKLEARN_OK and (not args.no_ml):
        print("  - baseline=<algo>/drop_column_importance_ridge.csv")
    if MPL_OK and (not args.no_plots):
        print("  - baseline=<algo>/*.png (plots)")


if __name__ == "__main__":
    main()
