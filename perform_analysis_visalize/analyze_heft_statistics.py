
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one HEFT speedup + correlation analysis + visualization.

One command, one file:
  python analyze_heft_speedup_all_in_one.py \
    --root /lustre/home/2501111916/workspace/TriForm_1207_hybrid_false/TriForm_1207_hybrid_false/algorithms/output/lens_eval_sweep/hw_scale_down_npu_large \
    --out  /lustre/home/2501111916/workspace/TriForm_1207_hybrid_false/TriForm_1207_hybrid_false/algorithms/output/hw_scale_down_npu_large \
    --baseline ALL

It will:
  1) Recursively parse baseline_compare_*.json
  2) Build a table of (baseline_algo vs HEFT) speedups
  3) Produce richer correlation metrics (Pearson/Spearman/Kendall/DistanceCorr/MI/PartialCorr)
  4) Do "winner (fastest algo)" association analysis with proper binning for numeric features
  5) Generate plots (merged from your visualize script) under *_viz/*.png

MAIN_ALGO = "hefthint"  # default reference algo for speedup comparisons

Outputs (examples):
  out/
    all_pairs_speedup.csv
    summary_by_baseline.csv
    eta_squared_categorical_effects.csv
    best_algo_per_config.csv
    best_algo_total_counts.csv
    best_algo_total_assoc_cramersv_binned.csv
    best_algo_total_assoc_mi_binned.csv
    baseline=<algo>/
      topk_speedup_total.csv
      spearman_corr_numeric.csv
      correlations_numeric.csv
      partial_corr_numeric.csv
      drop_column_importance_ridge.csv
      categorical_effects.csv
      mean_speedup_by_*.csv
      _viz/*.png

Output interpretation:
    all_pairs_speedup.csv : each row is one (baseline vs HEFT) comparison
    summary_by_baseline.csv : aggregate speedup stats per baseline algo heft相对于每一个算法的统计
        mean 平均加速比
        nedian 中位数加速比
        p90 90%的数据，加速比都不超过
        max
        min
    eta_squared_categorical_effects.csv : eta-squared of categorical features vs speedup_total
    best_algo_per_config.csv : which algo is best per JSON config (total/prefill/decode)
        每种配置下best_algo_total,best_algo_prefill,best_algo_decode
    best_algo_total_counts.csv : counts of best_algo_total occurrences (best_algo_{metric_tag}_counts) 
    best_algo_total_assoc_*.csv : association metrics (Cramér's V / MI) between best_algo_total and categorical features
        如果是离散变量用原值，如果是数值变成区间
    baseline=<algo>/ : per-baseline detailed analysis + plots

Notes:
  - Distance correlation is O(n^2). We automatically subsample when n is large.
  - Mutual information here is computed on discretized (binned) variables for stability.

--no-plots
    Skip all plotting steps.
--viz-only
    replot from existing CSV results, skip analysis.



"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------
# Optional plotting (merged from visualize script)
# --------------------------
MPL_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, TwoSlopeNorm
    MPL_OK = True
except Exception:
    plt = None
    ListedColormap = BoundaryNorm = LinearSegmentedColormap = None  # type: ignore

PALETTE = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
           "#FF3F04", "#FE5D00", "#FE8000", "#FFBF02"]

# Diverging for log2(speedup): <1 (blue) | >1 (warm)
DIVERGING_SPEEDUP = ["#0024FF", "#0349FF", "#006CFE", "#0092FE",
                     "#FFBF02", "#FE8000", "#FE5D00", "#FF3F04"]


# --------------------------
# Small utilities
# --------------------------
def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_div(a: Any, b: Any) -> float:
    a = _to_float(a)
    b = _to_float(b)
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return float(a / b)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def color_cycle(n: int) -> List[str]:
    if n <= 0:
        return []
    return [PALETTE[i % len(PALETTE)] for i in range(n)]

def read_csv_robust(p: Path) -> pd.DataFrame:
    """Robust CSV reader: handles weird index columns."""
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        # attempt with python engine / different encodings
        try:
            df = pd.read_csv(p, engine="python")
            return df
        except Exception:
            # last resort: try utf-8-sig
            df = pd.read_csv(p, encoding="utf-8-sig")
            return df

def savefig(out_path: Path):
    if not MPL_OK:
        return
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# --------------------------
# Domain parsing: algo/model/hardware
# --------------------------
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
    Rule (from your script):
      - hardware_json contains 'large'/'small' -> PIM size
      - contains 'npu' -> NPU compute is doubled
      - none -> normal
    Also try to infer 'stXX' from path segment like /st64/
    """
    hw_path = str(cfg.get("hardware_json", "") or "")
    hw_name = os.path.basename(hw_path).lower()
    fp_low = file_path.lower()

    npu_double = ("npu" in hw_name) or ("npu" in fp_low)

    flag_large = ("large" in hw_name) or ("large" in fp_low)
    flag_small = ("small" in hw_name) or ("small" in fp_low)
    if flag_large and not flag_small and not npu_double:
        pim_size = "large"
    elif flag_small and not flag_large and not npu_double:
        pim_size = "small"
    else:
        pim_size = "normal"

    st = None
    segs = re.split(r"[\\/]+", fp_low)
    for seg in segs:
        m = re.search(r"st(\d+)", seg)
        if m:
            try:
                st = int(m.group(1))
                break
            except Exception:
                pass

    hardware_tag = f"pim={pim_size}|npu={'2x' if npu_double else '1x'}|st={st if st is not None else 'NA'}"
    return {
        "hardware_json": hw_path,
        "pim_size": pim_size,
        "npu_double": int(npu_double),
        "st": st,
        "hardware_tag": hardware_tag,
    }


# --------------------------
# Derived features
# --------------------------
def add_derived_features(row: dict) -> dict:
    """Ensure total_tokens / ratios / deltas exist for EVERY row."""
    prefill = _to_float(row.get("prefill_len"))
    decode = _to_float(row.get("decode_len"))

    if np.isfinite(prefill) and np.isfinite(decode):
        tt = prefill + decode
        row["total_tokens"] = tt
        if tt != 0:
            row["decode_ratio"] = decode / tt
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

    row["delta_total_s"] = (bt - ht) if (np.isfinite(bt) and np.isfinite(ht)) else np.nan
    row["delta_prefill_s"] = (bpf - hpf) if (np.isfinite(bpf) and np.isfinite(hpf)) else np.nan
    row["delta_decode_s"] = (bdc - hdc) if (np.isfinite(bdc) and np.isfinite(hdc)) else np.nan

    dt = row.get("delta_total_s", np.nan)
    dd = row.get("delta_decode_s", np.nan)
    if np.isfinite(dt) and dt != 0 and np.isfinite(dd):
        row["gain_from_decode_ratio"] = dd / dt
    else:
        row["gain_from_decode_ratio"] = np.nan

    return row


# --------------------------
# JSON parsing
# --------------------------
def pick_best_per_algo(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    If one file contains multiple entries for same algo (different strategies),
    keep the one with minimum total_time_s.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for r in results:
        algo = parse_algo_name(r.get("policy", ""))
        if not algo:
            continue
        cur = best.get(algo)
        try:
            t = float(r.get("total_time_s", np.inf))
        except Exception:
            t = np.inf
        try:
            cur_t = float(cur.get("total_time_s", np.inf)) if cur else np.inf
        except Exception:
            cur_t = np.inf
        if cur is None or t < cur_t:
            best[algo] = r
    return best

def load_baseline_compare(
    file_path: str,
    add_best_baseline_row: bool = True,
    main_algo: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns rows: one per baseline algo vs MAIN_ALGO.

    MAIN_ALGO / main_algo 表示“参考算法”（默认是 'heft'）；
    speedup_total = baseline_total_time_s / main_algo_total_time_s
    """
    if main_algo is None:
        main_algo = MAIN_ALGO

    with open(file_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    cfg = j.get("config", {})
    results = j.get("results", [])
    best = pick_best_per_algo(results)

    # 这个 JSON 里如果没有 main_algo，就跳过
    if main_algo not in best:
        return []

    anchor = best[main_algo]  # 原来的 heft

    hw = infer_hardware(cfg, file_path)

    # 注意：下面这些列名仍然叫 heft_*，
    # 但语义变成了“main_algo_*”，只是为了兼容后面所有分析代码
    base_meta: Dict[str, Any] = {
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

        "heft_strategy": anchor.get("pim_strategy"),
        "heft_prefill_time_s": anchor.get("prefill_time_s"),
        "heft_decode_time_s": anchor.get("decode_time_s"),
        "heft_total_time_s": anchor.get("total_time_s"),
    }
    base_meta.update(hw)

    rows: List[Dict[str, Any]] = []
    for algo, r in best.items():
        # 跳过参考算法本身
        if algo == main_algo:
            continue
        row = dict(base_meta)
        row.update({
            "baseline_algo": algo,
            "baseline_strategy": r.get("pim_strategy"),
            "baseline_prefill_time_s": r.get("prefill_time_s"),
            "baseline_decode_time_s": r.get("decode_time_s"),
            "baseline_total_time_s": r.get("total_time_s"),
        })

        # speedup = baseline_time / main_algo_time
        row["speedup_total"] = safe_div(row["baseline_total_time_s"], row["heft_total_time_s"])
        row["speedup_prefill"] = safe_div(row["baseline_prefill_time_s"], row["heft_prefill_time_s"])
        row["speedup_decode"] = safe_div(row["baseline_decode_time_s"], row["heft_decode_time_s"])
        row = add_derived_features(row)
        rows.append(row)

    # 额外加一行 “best_baseline” 的情况
    if add_best_baseline_row:
        baselines = cfg.get("baselines", [])
        candidate = [(b, best[b]) for b in baselines if b in best]
        if candidate:
            bname, br = min(candidate, key=lambda x: _to_float(x[1].get("total_time_s")))
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



def _get_metric_time(r: Dict[str, Any], metric: str) -> float:
    if metric == "prefill":
        return _to_float(r.get("prefill_time_s"))
    if metric == "decode":
        return _to_float(r.get("decode_time_s"))
    return _to_float(r.get("total_time_s"))

def pick_best_algo(best_per_algo: Dict[str, Dict[str, Any]], metric: str = "total") -> Dict[str, Any]:
    """
    Given best-per-algo dict, pick global best under metric and also second best etc.
    metric in {total, prefill, decode}
    """
    items = []
    for algo, r in best_per_algo.items():
        t = _get_metric_time(r, metric)
        if np.isfinite(t):
            items.append((algo, t, r))
    if not items:
        return {}

    items.sort(key=lambda x: x[1])
    best_algo, best_time, best_r = items[0]
    second_algo, second_time, second_r = (items[1] if len(items) > 1 else (None, np.nan, {}))

    tie_algos = {best_algo}
    for algo, t, _ in items[1:]:
        if np.isfinite(t) and np.isfinite(best_time) and abs(t - best_time) <= 1e-12:
            tie_algos.add(algo)
        else:
            break

    return {
        "best_algo": best_algo,
        "best_time_s": best_time,
        "best_r": best_r,
        "second_algo": second_algo,
        "second_time_s": second_time,
        "second_r": second_r,
        "is_tie": int(len(tie_algos) > 1),
        "tie_algos": "|".join(sorted(tie_algos)),
        "n_algos": len(items),
    }

def load_best_algo_row(file_path: str) -> Optional[Dict[str, Any]]:
    """
    One row per JSON configuration:
      - which algo is best (min total_time_s)
      - second best & margin
      - workload + model + hardware fields
    """
    with open(file_path, "r", encoding="utf-8") as f:
        j = json.load(f)
    cfg = j.get("config", {})
    results = j.get("results", [])
    if not results:
        return None

    best_per_algo = pick_best_per_algo(results)

    p_total = pick_best_algo(best_per_algo, metric="total")
    if not p_total:
        return None
    p_prefill = pick_best_algo(best_per_algo, metric="prefill") if best_per_algo else None
    p_decode = pick_best_algo(best_per_algo, metric="decode") if best_per_algo else None

    hw = infer_hardware(cfg, file_path)

    row = {
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
    }
    row.update(hw)

    # derived token stats
    prefill = _to_float(row.get("prefill_len"))
    decode = _to_float(row.get("decode_len"))
    if np.isfinite(prefill) and np.isfinite(decode):
        tt = prefill + decode
        row["total_tokens"] = tt
        row["decode_ratio"] = (decode / tt) if tt else np.nan
        row["prefill_ratio"] = (prefill / tt) if tt else np.nan
    else:
        row["total_tokens"] = np.nan
        row["decode_ratio"] = np.nan
        row["prefill_ratio"] = np.nan

    # total winner
    row["best_algo_total"] = p_total["best_algo"]
    row["best_total_time_s"] = p_total["best_time_s"]
    row["second_best_algo_total"] = p_total["second_algo"]
    row["second_best_total_time_s"] = p_total["second_time_s"]
    row["best_is_tie_total"] = p_total["is_tie"]
    row["best_tie_algos_total"] = p_total["tie_algos"]
    if np.isfinite(row["best_total_time_s"]) and np.isfinite(row["second_best_total_time_s"]):
        row["best_margin_s_total"] = row["second_best_total_time_s"] - row["best_total_time_s"]
        row["best_margin_pct_total"] = safe_div(row["best_margin_s_total"], row["second_best_total_time_s"])
    else:
        row["best_margin_s_total"] = np.nan
        row["best_margin_pct_total"] = np.nan

    # prefill winner
    row["best_algo_prefill"] = p_prefill["best_algo"] if p_prefill else None
    row["best_prefill_time_s"] = p_prefill["best_time_s"] if p_prefill else np.nan

    # decode winner
    row["best_algo_decode"] = p_decode["best_algo"] if p_decode else None
    row["best_decode_time_s"] = p_decode["best_time_s"] if p_decode else np.nan

    main = MAIN_ALGO
    row["heft_is_best_total"] = int(row["best_algo_total"] == main)
    row["heft_is_best_decode"] = int(row.get("best_algo_decode") == main)
    row["heft_is_best_prefill"] = int(row.get("best_algo_prefill") == main)
    return row


# --------------------------
# Correlation / association metrics
# --------------------------
def spearman_corrs(df: pd.DataFrame, ycol: str, cols: List[str]) -> pd.DataFrame:
    """Compatibility output: Spearman correlation only."""
    out = []
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        if n < 2 or pd.Series(x[m]).nunique() < 2 or pd.Series(y[m]).nunique() < 2:
            corr = np.nan
        else:
            corr = pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman")
        out.append({
            "feature": c,
            "spearman": corr,
            "abs_spearman": abs(corr) if pd.notna(corr) else np.nan,
            "n": n,
        })
    return pd.DataFrame(out).sort_values("abs_spearman", ascending=False)

def eta_squared(df: pd.DataFrame, cat_col: str, ycol: str) -> float:
    """Variance explained by grouping on cat_col (0~1)."""
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
    for _, g in df.loc[mask].groupby(cat_col, dropna=False):
        yg = pd.to_numeric(g[ycol], errors="coerce").values
        yg = yg[np.isfinite(yg)]
        if len(yg) == 0:
            continue
        m = float(np.mean(yg))
        ss_within += float(np.sum((yg - m) ** 2))

    return 1.0 - ss_within / ss_total

def cramers_v(ct: pd.DataFrame) -> float:
    """Cramér's V association between two categorical variables (0~1)."""
    if ct is None or ct.empty:
        return np.nan
    obs = ct.to_numpy(dtype=float)
    n = obs.sum()
    if n <= 0:
        return np.nan
    r, c = obs.shape
    if r < 2 or c < 2:
        return 0.0
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    mask = expected > 0
    chi2 = ((obs[mask] - expected[mask]) ** 2 / expected[mask]).sum()
    k = min(r - 1, c - 1)
    if k <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))

def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())

def mutual_information_discrete(x: Sequence[Any], y: Sequence[Any]) -> Tuple[float, float]:
    """
    Mutual information between two discrete variables.
    Returns: (mi, nmi) where nmi = mi / min(Hx, Hy) in [0, 1] (when defined).
    """
    x = pd.Series(list(x)).astype("category")
    y = pd.Series(list(y)).astype("category")
    # drop rows with NA categories
    m = (~x.isna()) & (~y.isna())
    x = x[m].cat.codes.to_numpy()
    y = y[m].cat.codes.to_numpy()
    if len(x) < 2:
        return (np.nan, np.nan)

    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    nx = len(x_vals)
    ny = len(y_vals)

    joint = np.zeros((nx, ny), dtype=float)
    for i in range(len(x_inv)):
        joint[x_inv[i], y_inv[i]] += 1.0

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    n = joint.sum()
    if n <= 0:
        return (np.nan, np.nan)

    # MI
    mi = 0.0
    for i in range(nx):
        for j in range(ny):
            pij = joint[i, j]
            if pij <= 0:
                continue
            mi += (pij / n) * math.log((pij * n) / (px[i, 0] * py[0, j]))

    hx = _entropy_from_counts(px[:, 0])
    hy = _entropy_from_counts(py[0, :])
    denom = min(hx, hy)
    nmi = (mi / denom) if denom > 0 else 0.0
    return (float(mi), float(nmi))

def _quantile_bin(s: pd.Series, bins: int) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.nunique(dropna=True) < 2:
        return pd.Series(["(const)"] * len(s2), index=s2.index)
    q = min(bins, int(s2.nunique(dropna=True)))
    q = max(q, 2)
    try:
        b = pd.qcut(s2, q=q, duplicates="drop")
        return b.astype(str)
    except Exception:
        try:
            b = pd.cut(s2, bins=q)
            return b.astype(str)
        except Exception:
            return s2.astype(str)

def _is_mostly_numeric(s: pd.Series, min_ratio: float = 0.8) -> bool:
    x = pd.to_numeric(s, errors="coerce")
    ratio = np.isfinite(x.to_numpy()).mean() if len(x) else 0.0
    return ratio >= min_ratio and x.nunique(dropna=True) >= 3

def distance_correlation(x: np.ndarray, y: np.ndarray, max_n: int = 2000, random_state: int = 0) -> float:
    """
    Distance correlation for 1D arrays. Captures non-linear dependence (0~1).
    O(n^2) memory/time; we subsample if n>max_n.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)
    if n < 3:
        return np.nan

    if n > max_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_n, replace=False)
        x = x[idx]
        y = y[idx]
        n = len(x)

    # distance matrices
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    # double-centering
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = (A * B).sum() / (n * n)
    dvarx2 = (A * A).sum() / (n * n)
    dvary2 = (B * B).sum() / (n * n)
    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0
    dcor = math.sqrt(max(dcov2, 0.0) / math.sqrt(dvarx2 * dvary2))
    return float(min(max(dcor, 0.0), 1.0))

def compute_numeric_correlations(
    df: pd.DataFrame,
    ycol: str,
    numeric_cols: Sequence[str],
    mi_bins: int = 10,
    dcor_max_n: int = 2000,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    A richer "correlation" table for numeric features:
      Pearson(x, y), Pearson(x, log(y)), Spearman, Kendall, DistanceCorr, MI(binned), NMI(binned)
    """
    out = []
    y = pd.to_numeric(df[ycol], errors="coerce")
    ylog = np.log(y.where(y > 0))
    y_bin = _quantile_bin(ylog, bins=mi_bins)

    for c in numeric_cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        m = np.isfinite(x) & np.isfinite(y)
        n = int(m.sum())
        if n < 3 or pd.Series(x[m]).nunique() < 2 or pd.Series(y[m]).nunique() < 2:
            out.append({
                "feature": c,
                "n": n,
                "pearson": np.nan,
                "pearson_logy": np.nan,
                "spearman": np.nan,
                "kendall": np.nan,
                "distance_corr": np.nan,
                "mi": np.nan,
                "nmi": np.nan,
            })
            continue

        xs = x[m]
        ys = y[m]
        pearson = pd.Series(xs).corr(pd.Series(ys), method="pearson")
        spearman = pd.Series(xs).corr(pd.Series(ys), method="spearman")
        kendall = pd.Series(xs).corr(pd.Series(ys), method="kendall")

        # pearson on log(y)
        m2 = np.isfinite(xs) & np.isfinite(ylog[m])
        pearson_logy = pd.Series(xs[m2]).corr(pd.Series(ylog[m].values[m2]), method="pearson") if m2.sum() >= 3 else np.nan

        dcor = distance_correlation(xs.to_numpy(), ys.to_numpy(), max_n=dcor_max_n, random_state=random_state)

        # MI on binned x and binned log(y) for stability
        x_bin = _quantile_bin(xs, bins=mi_bins)
        mi, nmi = mutual_information_discrete(x_bin, y_bin.loc[m])
        out.append({
            "feature": c,
            "n": n,
            "pearson": pearson,
            "pearson_logy": pearson_logy,
            "spearman": spearman,
            "kendall": kendall,
            "distance_corr": dcor,
            "mi": mi,
            "nmi": nmi,
        })

    df_out = pd.DataFrame(out)
    # helpful ranks
    df_out["abs_spearman"] = df_out["spearman"].abs()
    df_out["abs_kendall"] = df_out["kendall"].abs()
    df_out["abs_pearson"] = df_out["pearson"].abs()
    df_out = df_out.sort_values(["nmi", "abs_spearman"], ascending=False)
    return df_out

def compute_categorical_effects(
    df: pd.DataFrame,
    ycol: str,
    categorical_cols: Sequence[str],
    mi_bins: int = 10,
) -> pd.DataFrame:
    """Eta² + MI/NMI (category vs binned log(y))."""
    y = pd.to_numeric(df[ycol], errors="coerce")
    ylog = np.log(y.where(y > 0))
    y_bin = _quantile_bin(ylog, bins=mi_bins)

    out = []
    for cat in categorical_cols:
        if cat not in df.columns:
            continue
        x = df[cat].astype(str)
        # eta² on log(y) tends to be more stable; also provide eta² on raw y
        eff_raw = eta_squared(df, cat, ycol)
        df2 = df.copy()
        df2["_ylog"] = ylog
        eff_log = eta_squared(df2, cat, "_ylog")

        mi, nmi = mutual_information_discrete(x, y_bin)
        out.append({
            "category": cat,
            "n": int(pd.to_numeric(y, errors="coerce").notna().sum()),
            "n_groups": int(x.nunique(dropna=True)),
            "eta_squared": eff_raw,
            "eta_squared_logy": eff_log,
            "mi": mi,
            "nmi": nmi,
        })
    return pd.DataFrame(out).sort_values(["nmi", "eta_squared_logy"], ascending=False)


# --------------------------
# Lightweight Ridge + CV (no sklearn dependency)
# --------------------------
@dataclass
class DesignMatrix:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]

def _standardize_inplace(x: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) and standardize in-place."""
    mu = np.nanmean(x, axis=0)
    sig = np.nanstd(x, axis=0)
    sig = np.where(sig < eps, 1.0, sig)
    x -= mu
    x /= sig
    return mu, sig

def build_design_matrix(
    df: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> DesignMatrix:
    """
    Build a design matrix with:
      - numeric: median-impute + standardize
      - categorical: one-hot (including NA)
      - intercept: NOT included (handled inside ridge)
    """
    parts = []
    names: List[str] = []

    # numeric
    for c in numeric_cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        # median impute
        med = np.nanmedian(v) if np.isfinite(v).any() else 0.0
        v = np.where(np.isfinite(v), v, med)
        parts.append(v.reshape(-1, 1))
        names.append(c)

    # categorical -> dummies
    if categorical_cols:
        cats = {}
        for c in categorical_cols:
            if c not in df.columns:
                continue
            cats[c] = df[c].astype(str).fillna("NA")
        if cats:
            dummies = pd.get_dummies(pd.DataFrame(cats), columns=list(cats.keys()), dummy_na=False)
            parts.append(dummies.to_numpy(dtype=float))
            names.extend(list(dummies.columns))

    if parts:
        X = np.concatenate(parts, axis=1)
    else:
        X = np.zeros((len(df), 0), dtype=float)

    # standardize numeric columns only (they are first len(numeric_present))
    n_num = sum(1 for c in numeric_cols if c in df.columns)
    if n_num > 0 and X.shape[1] >= n_num:
        _standardize_inplace(X[:, :n_num])

    # y already expected finite
    return DesignMatrix(X=X, y=y, feature_names=names)

def ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Ridge regression with intercept, closed-form:
      beta = argmin ||y - (b0 + Xb)||^2 + alpha ||b||^2
    """
    # add intercept
    n_train = X_train.shape[0]
    n_feat = X_train.shape[1]
    X0 = np.concatenate([np.ones((n_train, 1), dtype=float), X_train], axis=1)
    X1 = np.concatenate([np.ones((X_test.shape[0], 1), dtype=float), X_test], axis=1)

    # regularization (do not regularize intercept)
    A = X0.T @ X0
    reg = np.eye(n_feat + 1, dtype=float)
    reg[0, 0] = 0.0
    A = A + alpha * reg
    b = X0.T @ y_train
    try:
        coef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
    return X1 @ coef

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan
    yt = y_true[m]
    yp = y_pred[m]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot

def kfold_indices(n: int, n_splits: int = 5, random_state: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)
    out = []
    for i in range(n_splits):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i]) if n_splits > 1 else idx
        out.append((train, test))
    return out

def cv_r2_ridge(
    df: pd.DataFrame,
    ylog: np.ndarray,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    alpha: float = 1.0,
    n_splits: int = 5,
    random_state: int = 0,
) -> float:
    """
    Cross-validated R² for Ridge on log(y).

    Implementation detail:
      - We build the design matrix ONCE for the full dataframe to keep
        feature columns / scaling consistent across folds.
      - This slightly leaks test statistics into scaling, but avoids the much worse
        "train/test scaled differently" bug and is usually fine for relative importance ranking.
    """
    n = len(df)
    if n < 5:
        return np.nan
    if len(ylog) != n:
        return np.nan

    n_splits = min(n_splits, n)
    splits = kfold_indices(n, n_splits=n_splits, random_state=random_state)

    dm = build_design_matrix(df, ylog, numeric_cols, categorical_cols)
    X = dm.X

    scores = []
    for train_idx, test_idx in splits:
        y_pred = ridge_fit_predict(X[train_idx], ylog[train_idx], X[test_idx], alpha=alpha)
        scores.append(r2_score(ylog[test_idx], y_pred))

    return float(np.nanmean(scores)) if len(scores) else np.nan
def drop_column_importance_ridge(
    df: pd.DataFrame,
    ycol: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    group_map: Dict[str, List[str]],
    alpha: float = 1.0,
    n_splits: int = 5,
    random_state: int = 0,
    max_rows: int = 20000,
) -> pd.DataFrame:
    """
    "Controls other vars" importance via drop-column on Ridge CV-R² using log(y).
    drop_in_r2 = base_r2 - r2_without_group.
    """
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(y) & (y > 0)
    df2 = df.loc[m].copy()
    ylog = np.log(y[m])

    if len(df2) < 10:
        return pd.DataFrame(columns=["group", "base_r2", "r2_without_group", "drop_in_r2", "cols", "n"])

    # sample for speed if too big
    if len(df2) > max_rows:
        df2 = df2.sample(n=max_rows, random_state=random_state).copy()

    # safer: align ylog to df2 index by rebuilding
    ylog = np.log(pd.to_numeric(df2[ycol], errors="coerce").to_numpy(dtype=float))

    base = cv_r2_ridge(df2, ylog, numeric_cols, categorical_cols, alpha=alpha, n_splits=n_splits, random_state=random_state)

    rows = []
    for gname, cols_to_drop in group_map.items():
        num2 = [c for c in numeric_cols if c not in cols_to_drop]
        cat2 = [c for c in categorical_cols if c not in cols_to_drop]
        score = cv_r2_ridge(df2, ylog, num2, cat2, alpha=alpha, n_splits=n_splits, random_state=random_state)
        rows.append({
            "group": gname,
            "base_r2": base,
            "r2_without_group": score,
            "drop_in_r2": (base - score) if (np.isfinite(base) and np.isfinite(score)) else np.nan,
            "cols": ",".join(cols_to_drop),
            "n": len(df2),
        })
    return pd.DataFrame(rows).sort_values("drop_in_r2", ascending=False)

def partial_corr_ridge(
    df: pd.DataFrame,
    ycol: str,
    xcols: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Partial correlation via residualization with Ridge (log(y)):
      - For each x in xcols:
          y_res = residual( ylog ~ controls )
          x_res = residual( x ~ controls )
          partial_pearson = corr(x_res, y_res)
          partial_spearman = corr(rank(x_res), rank(y_res))
    """
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(y) & (y > 0)
    df2 = df.loc[m].copy()
    if len(df2) < 10:
        return pd.DataFrame(columns=["feature", "n", "partial_pearson", "partial_spearman"])

    ylog = np.log(pd.to_numeric(df2[ycol], errors="coerce").to_numpy(dtype=float))

    out = []
    for xname in xcols:
        if xname not in df2.columns:
            continue
        x = pd.to_numeric(df2[xname], errors="coerce").to_numpy(dtype=float)
        mx = np.isfinite(x)
        if mx.sum() < 10:
            out.append({"feature": xname, "n": int(mx.sum()), "partial_pearson": np.nan, "partial_spearman": np.nan})
            continue

        df3 = df2.loc[mx].copy()
        ylog3 = ylog[mx]
        x3 = x[mx]

        # controls exclude this x
        num_ctrl = [c for c in numeric_cols if (c != xname and c in df3.columns)]
        cat_ctrl = [c for c in categorical_cols if c in df3.columns]

        dm = build_design_matrix(df3, ylog3, num_ctrl, cat_ctrl)
        # y residual
        yhat = ridge_fit_predict(dm.X, dm.y, dm.X, alpha=alpha)
        yres = dm.y - yhat

        # x residual
        dm_x = build_design_matrix(df3, x3, num_ctrl, cat_ctrl)
        xhat = ridge_fit_predict(dm_x.X, dm_x.y, dm_x.X, alpha=alpha)
        xres = dm_x.y - xhat

        # Pearson
        if np.nanstd(xres) <= 1e-12 or np.nanstd(yres) <= 1e-12:
            pc = np.nan
        else:
            pc = float(np.corrcoef(xres, yres)[0, 1])

        # Spearman on residuals (rank)
        try:
            ps = pd.Series(xres).corr(pd.Series(yres), method="spearman")
        except Exception:
            ps = np.nan

        out.append({"feature": xname, "n": int(len(xres)), "partial_pearson": pc, "partial_spearman": ps})

    df_out = pd.DataFrame(out)
    df_out["abs_partial_pearson"] = df_out["partial_pearson"].abs()
    df_out = df_out.sort_values("abs_partial_pearson", ascending=False)
    return df_out


# --------------------------
# Best-algo (winner) association with binning
# --------------------------
def best_algo_association_binned(
    best_df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    bins: int = 8,
) -> pd.DataFrame:
    """Rank features by Cramér's V with numeric features binned first."""
    rows = []
    for feat in feature_cols:
        if feat not in best_df.columns:
            continue
        col = best_df[feat]
        if _is_mostly_numeric(col):
            f2 = _quantile_bin(col, bins=bins)
        else:
            f2 = col.astype(str).fillna("NA")

        tmp = pd.DataFrame({label_col: best_df[label_col].astype(str), feat: f2}).dropna()
        if tmp.empty:
            continue
        ct = pd.crosstab(tmp[label_col].astype(str), tmp[feat].astype(str))
        rows.append({
            "label": label_col,
            "feature": feat,
            "cramers_v": cramers_v(ct),
            "n": int(ct.to_numpy().sum()),
            "n_label": int(ct.shape[0]),
            "n_feature": int(ct.shape[1]),
            "binned": int(_is_mostly_numeric(col)),
        })
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False)

def best_algo_mi_binned(
    best_df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    bins: int = 8,
) -> pd.DataFrame:
    rows = []
    label = best_df[label_col].astype(str).fillna("NA")
    for feat in feature_cols:
        if feat not in best_df.columns:
            continue
        col = best_df[feat]
        if _is_mostly_numeric(col):
            f2 = _quantile_bin(col, bins=bins)
        else:
            f2 = col.astype(str).fillna("NA")
        mi, nmi = mutual_information_discrete(f2, label)
        rows.append({
            "label": label_col,
            "feature": feat,
            "mi": mi,
            "nmi": nmi,
            "n": int((~label.isna()).sum()),
            "binned": int(_is_mostly_numeric(col)),
        })
    return pd.DataFrame(rows).sort_values("nmi", ascending=False)

def best_algo_conditional_binned_long(
    best_df: pd.DataFrame,
    label_col: str,
    feature_cols: Sequence[str],
    bins: int = 8,
) -> pd.DataFrame:
    """
    Long-form conditional distribution:
      feature, feature_value, best_algo, count, frac
    where frac is P(best_algo | feature_value).
    Numeric features are binned first.
    """
    all_rows = []
    for feat in feature_cols:
        if feat not in best_df.columns:
            continue
        col = best_df[feat]
        if _is_mostly_numeric(col):
            feat_val = _quantile_bin(col, bins=bins)
        else:
            feat_val = col.astype(str).fillna("NA")

        tmp = pd.DataFrame({label_col: best_df[label_col].astype(str), feat: feat_val}).dropna()
        if tmp.empty:
            continue
        ct = pd.crosstab(tmp[feat].astype(str), tmp[label_col].astype(str))
        frac = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)

        count_long = ct.reset_index().melt(id_vars=[feat], var_name="best_algo", value_name="count")
        frac_long = frac.reset_index().melt(id_vars=[feat], var_name="best_algo", value_name="frac")
        merged = pd.merge(count_long, frac_long, on=[feat, "best_algo"], how="left")
        merged = merged.rename(columns={feat: "feature_value"})
        merged.insert(0, "feature", feat)
        all_rows.append(merged)

    if not all_rows:
        return pd.DataFrame(columns=["feature", "feature_value", "best_algo", "count", "frac"])
    out = pd.concat(all_rows, ignore_index=True)
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["frac"] = pd.to_numeric(out["frac"], errors="coerce")
    return out


# --------------------------
# Visualization (merged + extended)
# --------------------------
def plot_barh_categories(df: pd.DataFrame, label_col: str, val_col: str, title: str, out_path: Path,
                         positive_color: str = PALETTE[1], negative_color: str = PALETTE[6]):
    if not MPL_OK or df.empty or label_col not in df.columns or val_col not in df.columns:
        return
    d = df[[label_col, val_col]].dropna().copy()
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce")
    d = d.dropna(subset=[val_col]).sort_values(val_col, ascending=True)
    y = d[label_col].astype(str).values
    x = d[val_col].values
    colors = [positive_color if v >= 0 else negative_color for v in x]

    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    plt.barh(y[::-1], x[::-1], color=colors[::-1])
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(val_col)
    savefig(out_path)

def plot_summary_by_baseline(p: Path, out_dir: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "baseline_algo" not in df.columns:
        df = df.rename(columns={df.columns[0]: "baseline_algo"})
    for c in ["mean", "median", "p90", "n"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["mean"]).sort_values("mean", ascending=False)

    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    colors = color_cycle(len(df))
    plt.barh(df["baseline_algo"].astype(str)[::-1], df["mean"][::-1], color=colors[::-1])
    if "median" in df.columns:
        plt.scatter(df["median"], df["baseline_algo"].astype(str), color=PALETTE[4], s=18, label="median")
        plt.legend(loc="lower right")
    plt.axvline(1.0, linewidth=1)
    plt.title("Summary: mean speedup_total by baseline")
    plt.xlabel("mean(speedup_total)  (baseline/heft)")
    savefig(out_dir / "_viz" / "summary_by_baseline_mean.png")

def plot_all_pairs_speedup(p: Path, out_dir: Path, max_baselines: int = 16):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "baseline_algo" not in df.columns or "speedup_total" not in df.columns:
        return
    df["speedup_total"] = pd.to_numeric(df["speedup_total"], errors="coerce")
    df = df.dropna(subset=["baseline_algo", "speedup_total"])
    if df.empty:
        return

    # boxplot for top baselines by count
    counts = df["baseline_algo"].value_counts().head(max_baselines)
    baselines = counts.index.tolist()
    data = [df[df["baseline_algo"] == b]["speedup_total"].values for b in baselines]

    plt.figure(figsize=(max(10, 0.6 * len(baselines)), 5))
    bp = plt.boxplot(data, labels=baselines, patch_artist=True, showfliers=False)
    cols = color_cycle(len(baselines))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    plt.axhline(1.0, linewidth=1)
    plt.title(f"Distribution: speedup_total by baseline (top {len(baselines)} by count)")
    plt.ylabel("speedup_total (baseline/heft)")
    plt.xticks(rotation=30, ha="right")
    savefig(out_dir / "_viz" / "all_pairs_speedup_boxplot.png")

def plot_spearman(p: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "feature" not in df.columns or "spearman" not in df.columns:
        return

    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce")
    if "abs_spearman" not in df.columns:
        df["abs_spearman"] = df["spearman"].abs()
    df = df.dropna(subset=["spearman"]).sort_values("abs_spearman", ascending=True)

    out_dir = p.parent / "_viz"
    plt.figure(figsize=(10, max(4, 0.4 * len(df))))
    colors = [PALETTE[1] if v >= 0 else PALETTE[6] for v in df["spearman"].values]
    plt.barh(df["feature"].astype(str), df["spearman"].values, color=colors)
    plt.axvline(0, linewidth=1)
    plt.title("Spearman correlation with speedup_total")
    plt.xlabel("spearman")
    savefig(out_dir / f"{p.stem}_bar.png")

def plot_correlations_numeric(p: Path):
    """Plot the richer correlations table (if present)."""
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "feature" not in df.columns:
        return

    out_dir = p.parent / "_viz"
    # 1) abs_spearman
    if "spearman" in df.columns:
        tmp = df.copy()
        tmp["spearman"] = pd.to_numeric(tmp["spearman"], errors="coerce")
        tmp = tmp.dropna(subset=["spearman"])
        if not tmp.empty:
            tmp["abs_spearman"] = tmp["spearman"].abs()
            tmp = tmp.sort_values("abs_spearman", ascending=True)
            plt.figure(figsize=(10, max(4, 0.4 * len(tmp))))
            colors = [PALETTE[1] if v >= 0 else PALETTE[6] for v in tmp["spearman"].values]
            plt.barh(tmp["feature"].astype(str), tmp["spearman"].values, color=colors)
            plt.axvline(0, linewidth=1)
            plt.title("Correlation (Spearman) with speedup_total")
            plt.xlabel("spearman")
            savefig(out_dir / f"{p.stem}_spearman.png")

    # 2) distance corr
    if "distance_corr" in df.columns:
        tmp = df.copy()
        tmp["distance_corr"] = pd.to_numeric(tmp["distance_corr"], errors="coerce")
        tmp = tmp.dropna(subset=["distance_corr"]).sort_values("distance_corr", ascending=True)
        if not tmp.empty:
            plt.figure(figsize=(10, max(4, 0.4 * len(tmp))))
            plt.barh(tmp["feature"].astype(str), tmp["distance_corr"].values, color=PALETTE[2])
            plt.axvline(0, linewidth=1)
            plt.title("Dependence (Distance correlation) with speedup_total")
            plt.xlabel("distance_corr")
            savefig(out_dir / f"{p.stem}_distance_corr.png")

    # 3) NMI
    if "nmi" in df.columns:
        tmp = df.copy()
        tmp["nmi"] = pd.to_numeric(tmp["nmi"], errors="coerce")
        tmp = tmp.dropna(subset=["nmi"]).sort_values("nmi", ascending=True)
        if not tmp.empty:
            plt.figure(figsize=(10, max(4, 0.4 * len(tmp))))
            plt.barh(tmp["feature"].astype(str), tmp["nmi"].values, color=PALETTE[5])
            plt.axvline(0, linewidth=1)
            plt.title("Association (NMI, binned) with log(speedup_total)")
            plt.xlabel("nmi")
            savefig(out_dir / f"{p.stem}_nmi.png")

def plot_partial_corr(p: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "feature" not in df.columns or "partial_pearson" not in df.columns:
        return
    df["partial_pearson"] = pd.to_numeric(df["partial_pearson"], errors="coerce")
    df = df.dropna(subset=["partial_pearson"])
    if df.empty:
        return
    df["abs_partial_pearson"] = df["partial_pearson"].abs()
    df = df.sort_values("abs_partial_pearson", ascending=True)

    out_dir = p.parent / "_viz"
    plt.figure(figsize=(10, max(4, 0.4 * len(df))))
    colors = [PALETTE[1] if v >= 0 else PALETTE[6] for v in df["partial_pearson"].values]
    plt.barh(df["feature"].astype(str), df["partial_pearson"].values, color=colors)
    plt.axvline(0, linewidth=1)
    plt.title("Partial correlation (Ridge residualized) with log(speedup_total)")
    plt.xlabel("partial_pearson")
    savefig(out_dir / f"{p.stem}_bar.png")

def plot_drop_importance(p: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "group" not in df.columns or "drop_in_r2" not in df.columns:
        return
    df["drop_in_r2"] = pd.to_numeric(df["drop_in_r2"], errors="coerce")
    df = df.dropna(subset=["drop_in_r2"]).sort_values("drop_in_r2", ascending=True)

    out_dir = p.parent / "_viz"
    plt.figure(figsize=(10, max(4, 0.4 * len(df))))
    plt.barh(df["group"].astype(str), df["drop_in_r2"].values, color=PALETTE[3])
    plt.axvline(0, linewidth=1)
    plt.title("Drop-column importance (Ridge CV-R² on log(speedup_total))")
    plt.xlabel("drop_in_r2")
    savefig(out_dir / f"{p.stem}_bar.png")

def plot_topk(p: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "speedup_total" not in df.columns:
        return
    df["speedup_total"] = pd.to_numeric(df["speedup_total"], errors="coerce")
    df = df.dropna(subset=["speedup_total"]).sort_values("speedup_total", ascending=False).head(20)

    out_dir = p.parent / "_viz"
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(range(len(df))[::-1], df["speedup_total"].values[::-1], color=PALETTE[0])
    plt.axvline(1.0, linewidth=1)
    plt.title("Top cases by speedup_total")
    plt.xlabel("speedup_total")
    savefig(out_dir / f"{p.stem}_top20.png")

def plot_series_csv(p: Path, guess_title: str = ""):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if df.shape[1] < 2:
        return
    label_col, val_col = df.columns[:2]
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col]).sort_values(val_col, ascending=True).tail(40)

    out_dir = p.parent / "_viz"
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(df[label_col].astype(str), df[val_col].values, color=PALETTE[2])
    plt.axvline(1.0, linewidth=1)
    title = guess_title or p.stem
    plt.title(title)
    plt.xlabel(val_col)
    savefig(out_dir / f"{p.stem}_bar.png")

def plot_eta_effects(p: Path, out_dir: Path):
    """Heatmap: baseline_algo × category -> eta²."""
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    need = {"baseline_algo", "cat", "eta_squared"}
    if not need.issubset(df.columns):
        return
    df["eta_squared"] = pd.to_numeric(df["eta_squared"], errors="coerce")
    df = df.dropna(subset=["eta_squared"])
    if df.empty:
        return

    pv = df.pivot_table(index="baseline_algo", columns="cat", values="eta_squared", aggfunc="mean")
    if pv.empty:
        return

    out_dir = out_dir / "_viz"
    ensure_dir(out_dir)

    cmap = ListedColormap(DIVERGING_SPEEDUP) if ListedColormap else None
    bounds = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
    norm = BoundaryNorm(bounds, cmap.N) if (BoundaryNorm and cmap is not None) else None

    plt.figure(figsize=(max(8, 1.2 * len(pv.columns)), max(5, 0.5 * len(pv.index))))
    plt.imshow(pv.values, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    plt.colorbar(label="eta_squared (0~1)")
    plt.xticks(range(len(pv.columns)), [str(c) for c in pv.columns], rotation=30, ha="right")
    plt.yticks(range(len(pv.index)), [str(i) for i in pv.index])
    plt.title("Categorical effect size (eta²): baseline_algo × category")
    savefig(out_dir / "eta_squared_heatmap.png")

def plot_mean_speedup_batch_prefill_decode(p: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    need = {"batch", "prefill_len", "decode_len", "speedup_total"}
    if not need.issubset(df.columns):
        return
    for c in ["batch", "prefill_len", "decode_len", "speedup_total"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["batch", "prefill_len", "decode_len", "speedup_total"])
    if df.empty:
        return

    out_dir = p.parent / "_viz"
    cmap = ListedColormap(DIVERGING_SPEEDUP) if ListedColormap else None

    # 从路径推断 baseline 名字: .../baseline=<algo>/
    baseline_tag = None
    for part in p.parts[::-1]:
        if part.startswith("baseline="):
            baseline_tag = part.split("=", 1)[-1]
            break
    baseline_label = f"baseline={baseline_tag}" if baseline_tag else "baseline"

    def _draw_heatmap(pv: pd.DataFrame, title_suffix: str, fname: str):
        if pv.empty or cmap is None:
            return
        arr = pv.values.astype(float)

        # mask 掉 NaN / inf
        mask = ~np.isfinite(arr)
        if np.all(mask):
            return
        valid = arr[~mask]
        if valid.size == 0:
            return

        center = 1.0  # speedup=1 作为中间点

        # 用数据最小值和最大值到 center 的距离构造对称区间
        vmin_raw = float(np.min(valid))
        vmax_raw = float(np.max(valid))
        dist_low = abs(center - vmin_raw)
        dist_high = abs(vmax_raw - center)
        half = max(dist_low, dist_high)
        if not np.isfinite(half) or half == 0:
            half = 1.0  # 所有值都一样（比如全是 1），随便给个范围

        vmin = center - half
        vmax = center + half
        # 这里一定满足 vmin < center < vmax
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

        plt.figure(figsize=(10, 7))
        im = plt.imshow(
            np.ma.array(arr, mask=mask),
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        plt.colorbar(im, label="speedup_total (baseline/heft)")
        plt.xticks(range(len(pv.columns)), [str(c) for c in pv.columns], rotation=30, ha="right")
        plt.yticks(range(len(pv.index)), [str(i) for i in pv.index])
        plt.xlabel("decode_len")
        plt.ylabel("prefill_len")
        plt.title(f"{baseline_label} speedup heatmap{title_suffix}")
        savefig(out_dir / fname)

    # per-batch heatmaps
    for b in sorted(df["batch"].unique()):
        sub = df[df["batch"] == b]
        pv = sub.pivot_table(
            index="prefill_len",
            columns="decode_len",
            values="speedup_total",
            aggfunc="mean",
        )
        _draw_heatmap(pv, f" | batch={int(b)}", f"heatmap_speedup_batch={int(b)}.png")

    # 聚合所有 batch
    pv_all = df.pivot_table(
        index="prefill_len",
        columns="decode_len",
        values="speedup_total",
        aggfunc="mean",
    )
    _draw_heatmap(pv_all, " | all batches", "heatmap_speedup_all_batches.png")



def plot_best_algo_counts(p: Path, out_dir: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "count" not in df.columns:
        return
    algo_cols = [c for c in df.columns if c != "count"]
    if not algo_cols:
        return
    algo_col = algo_cols[0]

    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df.dropna(subset=["count"]).sort_values("count", ascending=True)
    if df.empty:
        return
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(df[algo_col].astype(str), df["count"].values, color=PALETTE[0])
    plt.title(f"Winner counts: {algo_col}")
    plt.xlabel("count")
    savefig(out_dir / "_viz" / f"{p.stem}.png")

def plot_best_algo_assoc_cramersv(p: Path, out_dir: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "feature" not in df.columns or "cramers_v" not in df.columns:
        return
    df["cramers_v"] = pd.to_numeric(df["cramers_v"], errors="coerce")
    df = df.dropna(subset=["cramers_v"]).sort_values("cramers_v", ascending=True)
    if df.empty:
        return
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(df["feature"].astype(str), df["cramers_v"].values, color=PALETTE[3])
    plt.title("Association with winner (Cramér's V)")
    plt.xlabel("cramers_v")
    savefig(out_dir / "_viz" / f"{p.stem}_bar.png")

def plot_best_algo_assoc_mi(p: Path, out_dir: Path):
    if not MPL_OK:
        return
    df = read_csv_robust(p)
    if "feature" not in df.columns or "nmi" not in df.columns:
        return
    df["nmi"] = pd.to_numeric(df["nmi"], errors="coerce")
    df = df.dropna(subset=["nmi"]).sort_values("nmi", ascending=True)
    if df.empty:
        return
    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(df["feature"].astype(str), df["nmi"].values, color=PALETTE[5])
    plt.title("Association with winner (NMI, binned)")
    plt.xlabel("nmi")
    savefig(out_dir / "_viz" / f"{p.stem}_bar.png")


def dispatch(csv_path: Path, root_out: Path):
    name = csv_path.name
    if name == "summary_by_baseline.csv":
        plot_summary_by_baseline(csv_path, root_out)
    elif name == "all_pairs_speedup.csv":
        plot_all_pairs_speedup(csv_path, root_out)
    elif name == "eta_squared_categorical_effects.csv":
        plot_eta_effects(csv_path, root_out)
    elif name == "spearman_corr_numeric.csv":
        plot_spearman(csv_path)
    elif name == "correlations_numeric.csv":
        plot_correlations_numeric(csv_path)
    elif name == "partial_corr_numeric.csv":
        plot_partial_corr(csv_path)
    elif name == "drop_column_importance_ridge.csv":
        plot_drop_importance(csv_path)
    elif name == "topk_speedup_total.csv":
        plot_topk(csv_path)
    elif name == "mean_speedup_by_batch_prefill_decode.csv":
        plot_mean_speedup_batch_prefill_decode(csv_path)
    elif name in ("mean_speedup_by_hardware_tag.csv", "mean_speedup_by_model_variant.csv"):
        plot_series_csv(csv_path, guess_title=name.replace(".csv", ""))
    elif name.startswith("best_algo_") and name.endswith("_counts.csv"):
        plot_best_algo_counts(csv_path, root_out)
    elif name.startswith("best_algo_") and "assoc_cramersv" in name:
        plot_best_algo_assoc_cramersv(csv_path, root_out)
    elif name.startswith("best_algo_") and "assoc_mi" in name:
        plot_best_algo_assoc_mi(csv_path, root_out)
    else:
        # generic fallback: if it looks like key/value -> barh
        try:
            df = read_csv_robust(csv_path)
            if df.shape[1] == 2:
                plot_series_csv(csv_path, guess_title=csv_path.stem)
        except Exception:
            pass

def run_visualization(root_out: Path):
    if not MPL_OK:
        print("[WARN] matplotlib not available -> skip visualization")
        return
    csvs = sorted(root_out.rglob("*.csv"))
    if not csvs:
        print(f"[WARN] no CSV found under {root_out}")
        return
    for p in csvs:
        try:
            dispatch(p, root_out)
        except Exception as e:
            print(f"[WARN] plot failed: {p}: {e}")
    print("VIZ DONE. Check *_viz/*.png under:", root_out)


# --------------------------
# Main analysis pipeline
# --------------------------
def normalize_numeric_columns(df: pd.DataFrame):
    for c in [
        "batch", "prefill_len", "decode_len", "st", "npu_double", "model_size_b",
        "speedup_total", "speedup_decode", "speedup_prefill",
        "heft_total_time_s", "baseline_total_time_s",
        "heft_decode_time_s", "baseline_decode_time_s",
        "heft_prefill_time_s", "baseline_prefill_time_s",
        "total_tokens", "decode_ratio", "prefill_ratio",
        "delta_total_s", "delta_decode_s", "delta_prefill_s",
        "gain_from_decode_ratio",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def analyze(root: Path, out_dir: Path, baseline_arg: str, topk: int,
            no_ml: bool, no_plots: bool,
            mi_bins: int, assoc_bins: int, dcor_max_n: int,
            random_state: int, main_algo: str):

    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(root.rglob("baseline_compare_*.json"))
    if not files:
        raise SystemExit(f"No baseline_compare_*.json found under: {root}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            rows.extend(
                load_baseline_compare(
                    str(fp),
                    add_best_baseline_row=True,
                    main_algo=main_algo,
                )
            )
        except Exception as e:
            print(f"[WARN] failed to load {fp}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Parsed 0 rows. (Maybe files don't contain 'heft' in results?)")
    normalize_numeric_columns(df)

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

    # --------------------------
    # Best-algorithm per config analysis (winner = min total_time_s)
    # --------------------------
    best_rows = []
    for fp in files:
        try:
            r = load_best_algo_row(str(fp))
            if r is not None:
                best_rows.append(r)
        except Exception as e:
            print(f"[WARN] failed best-algo parse {fp}: {e}")

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        normalize_numeric_columns(best_df)
        best_df.to_csv(out_dir / "best_algo_per_config.csv", index=False)
        print(f"[OK] wrote: {out_dir/'best_algo_per_config.csv'} (rows={len(best_df)})")

        assoc_features = [
            "batch", "prefill_len", "decode_len",
            "total_tokens", "decode_ratio",
            "hardware_tag", "pim_size", "npu_double", "st",
            "model_family", "model_variant", "model_size_b", "dtype"
        ]

        # 针对 total / decode / prefill 分别做赢家统计与关联分析
        metric_cols = [
            ("total", "best_algo_total"),
            ("decode", "best_algo_decode"),
            ("prefill", "best_algo_prefill"),
        ]

        for metric_tag, col in metric_cols:
            if col not in best_df.columns:
                continue
            sub = best_df.dropna(subset=[col]).copy()
            if sub.empty:
                continue

            counts = sub[col].astype(str).value_counts().rename_axis(col).reset_index(name="count")
            counts.to_csv(out_dir / f"best_algo_{metric_tag}_counts.csv", index=False)
            print(f"[OK] wrote: {out_dir / f'best_algo_{metric_tag}_counts.csv'}")

            assoc_b = best_algo_association_binned(sub, col, assoc_features, bins=assoc_bins)
            assoc_b.to_csv(out_dir / f"best_algo_{metric_tag}_assoc_cramersv_binned.csv", index=False)
            print(f"[OK] wrote: {out_dir / f'best_algo_{metric_tag}_assoc_cramersv_binned.csv'}")

            mi_b = best_algo_mi_binned(sub, col, assoc_features, bins=assoc_bins)
            mi_b.to_csv(out_dir / f"best_algo_{metric_tag}_assoc_mi_binned.csv", index=False)
            print(f"[OK] wrote: {out_dir / f'best_algo_{metric_tag}_assoc_mi_binned.csv'}")

            cond_b = best_algo_conditional_binned_long(sub, col, assoc_features, bins=assoc_bins)
            cond_b.to_csv(out_dir / f"best_algo_{metric_tag}_conditional_binned_long.csv", index=False)
            print(f"[OK] wrote: {out_dir / f'best_algo_{metric_tag}_conditional_binned_long.csv'}")

    # Filter baselines
    if baseline_arg.strip().upper() == "ALL":
        baselines = sorted(df["baseline_algo"].dropna().unique().tolist())
    else:
        baselines = [b.strip() for b in baseline_arg.split(",") if b.strip()]

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
        "heft_strategy", "baseline_strategy",
    ]
    group_map = {
        "batch": ["batch"],
        "prefill_len": ["prefill_len"],
        "decode_len": ["decode_len"],
        "hardware": ["hardware_tag", "pim_size", "npu_double", "st"],
        "model": ["model_family", "model_variant", "model_size_b", "dtype"],
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

        # Top-K
        topk_df = sub.sort_values("speedup_total", ascending=False).head(topk)
        topk_df.to_csv(bdir / "topk_speedup_total.csv", index=False)

        # Spearman (compat)
        corr_sp = spearman_corrs(sub, "speedup_total", [c for c in numeric_cols if c in sub.columns])
        corr_sp.to_csv(bdir / "spearman_corr_numeric.csv", index=False)

        # Rich correlations
        corr_full = compute_numeric_correlations(
            sub, "speedup_total",
            [c for c in numeric_cols if c in sub.columns],
            mi_bins=mi_bins,
            dcor_max_n=dcor_max_n,
            random_state=random_state,
        )
        corr_full.to_csv(bdir / "correlations_numeric.csv", index=False)

        # Categorical effects
        cat_eff = compute_categorical_effects(sub, "speedup_total", [c for c in categorical_cols if c in sub.columns], mi_bins=mi_bins)
        cat_eff.to_csv(bdir / "categorical_effects.csv", index=False)

        # Global eta² (for heatmap)
        for cat in ["hardware_tag", "pim_size", "npu_double", "model_family", "model_variant", "dtype"]:
            if cat not in sub.columns:
                continue
            eff = eta_squared(sub, cat, "speedup_total")
            effects_rows.append({"baseline_algo": b, "cat": cat, "eta_squared": eff, "n": len(sub)})

        # ML-style: partial corr + drop-column importance
        if not no_ml:
            try:
                pc = partial_corr_ridge(sub, "speedup_total",
                                        xcols=[c for c in numeric_cols if c in sub.columns],
                                        numeric_cols=[c for c in numeric_cols if c in sub.columns],
                                        categorical_cols=[c for c in categorical_cols if c in sub.columns],
                                        alpha=1.0)
                pc.to_csv(bdir / "partial_corr_numeric.csv", index=False)
            except Exception as e:
                print(f"[WARN] partial corr failed baseline={b}: {e}")

            try:
                imp = drop_column_importance_ridge(
                    sub, "speedup_total",
                    numeric_cols=[c for c in numeric_cols if c in sub.columns],
                    categorical_cols=[c for c in categorical_cols if c in sub.columns],
                    group_map=group_map,
                    alpha=1.0,
                    n_splits=5,
                    random_state=random_state,
                )
                imp.to_csv(bdir / "drop_column_importance_ridge.csv", index=False)
            except Exception as e:
                print(f"[WARN] drop-column importance failed baseline={b}: {e}")

        # Aggregations
        if "hardware_tag" in sub.columns:
            sub.groupby("hardware_tag")["speedup_total"].mean().sort_values(ascending=False).reset_index() \
                .to_csv(bdir / "mean_speedup_by_hardware_tag.csv", index=False)
        if "model_variant" in sub.columns:
            sub.groupby("model_variant")["speedup_total"].mean().sort_values(ascending=False).reset_index() \
                .to_csv(bdir / "mean_speedup_by_model_variant.csv", index=False)

        if all(c in sub.columns for c in ["batch", "prefill_len", "decode_len"]):
            sub.groupby(["batch", "prefill_len", "decode_len"])["speedup_total"].mean().reset_index() \
                .sort_values("speedup_total", ascending=False) \
                .to_csv(bdir / "mean_speedup_by_batch_prefill_decode.csv", index=False)

        # Basic plots (optional, quick; visualization pass will add more)
        if MPL_OK and (not no_plots):
            try:
                # scatter of speedup vs some features
                for xcol in ["decode_len", "prefill_len", "batch"]:
                    if xcol not in sub.columns:
                        continue
                    x = pd.to_numeric(sub[xcol], errors="coerce")
                    y = pd.to_numeric(sub["speedup_total"], errors="coerce")
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() < 3:
                        continue
                    plt.figure()
                    plt.scatter(x[m], y[m], s=8)
                    plt.axhline(1.0, linewidth=1)
                    plt.xlabel(xcol)
                    plt.ylabel("speedup_total")
                    plt.title(f"baseline={b} | speedup_total vs {xcol}")
                    savefig(bdir / "_viz" / f"scatter_speedup_vs_{xcol}.png")
            except Exception as e:
                print(f"[WARN] basic plots failed baseline={b}: {e}")

        print(f"[OK] baseline={b} -> {bdir}")

    effects = pd.DataFrame(effects_rows)
    if not effects.empty:
        effects.to_csv(out_dir / "eta_squared_categorical_effects.csv", index=False)
        print(f"[OK] wrote: {out_dir/'eta_squared_categorical_effects.csv'}")

    print("\nANALYSIS DONE. Outputs in:", out_dir.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, required=False,
                    help="Root containing results (will search baseline_compare_*.json recursively)")
    ap.add_argument("--out", type=str, default="heft_analysis_out", help="Output directory")
    ap.add_argument("--baseline", type=str, default="ALL",
                    help="Which baseline algo to analyze (e.g. 'ianus' or 'ianus,neupims'). Use ALL for all.")
    ap.add_argument("--topk", type=int, default=30, help="Top-K cases to export for each baseline")
    ap.add_argument("--no-ml", action="store_true", help="Disable ML-style analyses (partial corr, drop-column importance)")
    ap.add_argument("--no-plots", action="store_true", help="Disable plots (still writes CSV)")
    ap.add_argument("--viz-only", action="store_true", help="Only visualize existing CSV outputs under --out (skip JSON parsing)")
    ap.add_argument("--analysis-only", action="store_true", help="Only analyze and write CSV (skip visualization)")
    ap.add_argument("--mi-bins", type=int, default=10, help="Quantile bins for MI/NMI (default: 10)")
    ap.add_argument("--assoc-bins", type=int, default=8, help="Quantile bins for winner association (default: 8)")
    ap.add_argument("--dcor-max-n", type=int, default=2000, help="Max sample size for distance correlation (default: 2000)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (subsampling, CV splits)")
    ap.add_argument(
        "--main-algo",
        type=str,
        default="hefthint",
        help="reference algo for speedup (default: 'hefthint')",
    )

    args = ap.parse_args()
    global MAIN_ALGO
    MAIN_ALGO = args.main_algo


    if (not args.viz_only) and (not args.root):
        raise SystemExit("--root is required unless you use --viz-only")


    out_dir = Path(args.out)
    if not args.viz_only:
        analyze(
            root=Path(args.root),
            out_dir=out_dir,
            baseline_arg=args.baseline,
            topk=args.topk,
            no_ml=args.no_ml,
            no_plots=args.no_plots,
            mi_bins=max(2, args.mi_bins),
            assoc_bins=max(2, args.assoc_bins),
            dcor_max_n=max(200, args.dcor_max_n),
            random_state=args.seed,
            main_algo=args.main_algo,
        )

    if args.analysis_only or args.no_plots:
        return

    run_visualization(out_dir)


if __name__ == "__main__":
    main()