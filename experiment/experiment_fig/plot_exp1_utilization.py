#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot per-family device utilization from trace CSVs.

Supported file name pattern (recursive scan under --search-dir):
  <algo>_prefill-<P>xdecode_<D>_<comms|ops>_trace.csv

Examples
--------
# Plot selected prefills / decodes and show signed gap = (accel - pim)
python3 plot_exp1_utilization.py \
  --search-dir ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst64_rst64/qwen_1.8b_fp16_b4_s64 \
  --prefills 128,256,512,1024 \
  --decodes 64,128,256,512,1024 \
  --output ../../figs/exp1/util/qwen_1_8b_fp16_b4_s64_utilization.pdf
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_COLORS: List[str] = [
    "#5837a8",
    "#a83747",
    "#3791a8",
    "#3760a9",
    "#39a937",
    "#a89a37",
]

TRACE_RE = re.compile(
    r"^(?P<algo>.+?)_prefill-(?P<prefill>\d+)xdecode_(?P<decode>\d+)_(?P<kind>comms|ops)_trace\.csv$",
    re.IGNORECASE,
)

DISPLAY_NAME_MAP: Dict[str, str] = {
    "hefthint": "this work",
}

HEFT_VARIANTS = {"heft", "hefthint"}

FAMILY_ORDER = ["pim", "accel"]
FAMILY_LABELS: Dict[str, str] = {
    "pim": "PIM",
    "accel": "NPU/GPU",
    "cpu": "CPU",
}
FAMILY_MARKERS: Dict[str, str] = {
    "pim": "o",
    "accel": "s",
}
FAMILY_X_OFFSETS: Dict[str, float] = {
    "pim": -0.06,
    "accel": 0.06,
}
DIFF_MARKER = "^"
DIFF_LINESTYLE = "--"

PREFERRED_ALGO_ORDER: List[str] = ["pd", "ianus", "facil"]


@dataclass(frozen=True)
class TraceKey:
    algo: str
    prefill: int
    decode: int


@dataclass
class TracePair:
    comms_path: Path
    ops_path: Path


@dataclass
class UtilResult:
    source_algo: str
    plotted_algo: str
    prefill: int
    decode: int
    overall_utilization: float
    pim_utilization: Optional[float]
    accel_utilization: Optional[float]
    makespan_s: float
    n_devices: int
    family_counts: Dict[str, int]
    family_devices: Dict[str, List[str]]
    device_utils: Dict[str, float]
    device_families: Dict[str, str]
    comms_path: Path
    ops_path: Path


# -----------------------------
# Parsing / discovery helpers
# -----------------------------

def parse_int_list(s: Optional[str]) -> List[int]:
    """
    Supported forms:
      128,256,512
      64:512:64
      64-70
      all
      None / empty -> []
    """
    if s is None:
        return []
    s = s.strip().lower()
    if not s or s == "all":
        return []

    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            toks = [t.strip() for t in part.split(":")]
            if len(toks) not in {2, 3}:
                raise ValueError(f"invalid range token: {part}")
            start = int(toks[0])
            stop = int(toks[1])
            step = int(toks[2]) if len(toks) == 3 else (1 if stop >= start else -1)
            if step == 0:
                raise ValueError(f"range step cannot be zero: {part}")
            if (stop - start) * step < 0:
                raise ValueError(f"range step direction mismatches bounds: {part}")
            for v in range(start, stop + (1 if step > 0 else -1), step):
                out.append(v)
            continue

        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            a, b = [x.strip() for x in part.split("-", 1)]
            start = int(a)
            stop = int(b)
            step = 1 if stop >= start else -1
            for v in range(start, stop + step, step):
                out.append(v)
            continue

        out.append(int(part))

    return sorted(dict.fromkeys(out))


def parse_str_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def discover_trace_files(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("*_trace.csv"))


def normalize_algo_token(algo: str) -> str:
    raw = (algo or "").strip().lower()
    compact = re.sub(r"[\s_\-]+", "", raw)
    if compact in {"heft", "hefthint", "thiswork"}:
        return "hefthint"
    return raw


def parse_trace_filename(path: Path) -> Optional[Tuple[TraceKey, str]]:
    m = TRACE_RE.match(path.name)
    if not m:
        return None
    key = TraceKey(
        algo=m.group("algo").lower(),
        prefill=int(m.group("prefill")),
        decode=int(m.group("decode")),
    )
    kind = m.group("kind").lower()
    return key, kind


def _pair_from_candidates(comms_paths: List[Path], ops_paths: List[Path]) -> Optional[TracePair]:
    if not comms_paths or not ops_paths:
        return None

    comms_paths = sorted(comms_paths)
    ops_paths = sorted(ops_paths)

    for c in comms_paths:
        for o in ops_paths:
            if c.parent == o.parent:
                return TracePair(comms_path=c, ops_path=o)

    print(
        f"[WARN] multiple trace candidates found but no same-parent pair; "
        f"use first comms={comms_paths[0]} ops={ops_paths[0]}",
        file=sys.stderr,
    )
    return TracePair(comms_path=comms_paths[0], ops_path=ops_paths[0])


def build_trace_index(paths: Sequence[Path]) -> Dict[TraceKey, TracePair]:
    raw: Dict[TraceKey, Dict[str, List[Path]]] = {}
    for path in paths:
        parsed = parse_trace_filename(path)
        if parsed is None:
            continue
        key, kind = parsed
        raw.setdefault(key, {}).setdefault(kind, []).append(path)

    out: Dict[TraceKey, TracePair] = {}
    for key, bucket in raw.items():
        pair = _pair_from_candidates(bucket.get("comms", []), bucket.get("ops", []))
        if pair is not None:
            out[key] = pair
    return out


# -----------------------------
# Device-family helpers
# -----------------------------

def _safe_lower(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    return "" if s in {"", "nan", "none"} else s


def _family_from_type_text(type_text: object, include_cpu: bool = False) -> Optional[str]:
    t = _safe_lower(type_text)
    if not t:
        return None

    if any(tok in t for tok in ["comm", "network"]):
        return None
    if any(tok in t for tok in ["pim", "aim"]):
        return "pim"
    if any(tok in t for tok in ["npu", "gpu"]):
        return "accel"
    if include_cpu and "cpu" in t:
        return "cpu"
    return None


def _family_from_device_name(device: object, include_cpu: bool = False) -> Optional[str]:
    d = _safe_lower(device)
    if not d:
        return None

    if d == "comm" or d.startswith("comm"):
        return None
    if "pim" in d or re.search(r"(^|[^a-z])aim(\d+)?($|[^a-z])", d):
        return "pim"
    if any(tok in d for tok in ["npu", "gpu", "ascend", "cuda"]):
        return "accel"
    if include_cpu and d.startswith("cpu"):
        return "cpu"
    return None


def _resolve_device_family(device: str,
                           type_hints: Sequence[str],
                           include_cpu: bool = False) -> Optional[str]:
    votes: Counter[str] = Counter()

    for hint in type_hints:
        fam = _family_from_type_text(hint, include_cpu=include_cpu)
        if fam is not None:
            votes[fam] += 1

    name_fam = _family_from_device_name(device, include_cpu=include_cpu)
    if name_fam is not None:
        votes[name_fam] += 1

    if not votes:
        return None

    if len(votes) == 1:
        return next(iter(votes))

    if name_fam is not None:
        max_votes = max(votes.values())
        if votes[name_fam] == max_votes:
            return name_fam

    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _collect_device_families(ops_df: pd.DataFrame,
                             comms_df: pd.DataFrame,
                             include_cpu: bool = False) -> Dict[str, str]:
    type_hints: DefaultDict[str, List[str]] = defaultdict(list)

    if not ops_df.empty and "device" in ops_df.columns:
        if "device_type" in ops_df.columns:
            for dev, dtype in zip(ops_df["device"], ops_df["device_type"]):
                dev_s = _safe_lower(dev)
                if dev_s:
                    type_hints[str(dev)].append(str(dtype))
        else:
            for dev in ops_df["device"].dropna().astype(str):
                if _safe_lower(dev):
                    type_hints[dev].append("")

    if not comms_df.empty:
        if "src" in comms_df.columns:
            if "src_type" in comms_df.columns:
                for dev, dtype in zip(comms_df["src"], comms_df["src_type"]):
                    dev_s = _safe_lower(dev)
                    if dev_s:
                        type_hints[str(dev)].append(str(dtype))
            else:
                for dev in comms_df["src"].dropna().astype(str):
                    if _safe_lower(dev):
                        type_hints[dev].append("")

        if "dst" in comms_df.columns:
            if "dst_type" in comms_df.columns:
                for dev, dtype in zip(comms_df["dst"], comms_df["dst_type"]):
                    dev_s = _safe_lower(dev)
                    if dev_s:
                        type_hints[str(dev)].append(str(dtype))
            else:
                for dev in comms_df["dst"].dropna().astype(str):
                    if _safe_lower(dev):
                        type_hints[dev].append("")

    device_families: Dict[str, str] = {}
    for dev, hints in type_hints.items():
        family = _resolve_device_family(dev, hints, include_cpu=include_cpu)
        if family is not None:
            device_families[dev] = family

    return device_families


# -----------------------------
# Utilization computation
# -----------------------------

def _filter_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if phase == "all":
        return df
    if "phase" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["phase"].astype(str).str.lower() == phase.lower()].copy()


def _merge_intervals(intervals: Iterable[Tuple[float, float]]) -> float:
    arr: List[Tuple[float, float]] = []
    for start, end in intervals:
        if pd.isna(start) or pd.isna(end):
            continue
        s = float(start)
        e = float(end)
        if e < s:
            continue
        arr.append((s, e))

    if not arr:
        return 0.0

    arr.sort(key=lambda x: (x[0], x[1]))
    total = 0.0
    cur_s, cur_e = arr[0]
    for s, e in arr[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _read_trace_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["start", "end", "duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _family_mean(device_utils: Dict[str, float],
                 device_families: Dict[str, str],
                 family: str) -> Optional[float]:
    vals = [util for dev, util in device_utils.items() if device_families.get(dev) == family]
    if not vals:
        return None
    return float(np.mean(vals))


def compute_device_utilization(comms_path: Path,
                               ops_path: Path,
                               phase: str = "all",
                               include_cpu: bool = False) -> UtilResult:
    comms_df = _filter_phase(_read_trace_csv(comms_path), phase=phase)
    ops_df = _filter_phase(_read_trace_csv(ops_path), phase=phase)

    if comms_df.empty and ops_df.empty:
        raise ValueError(f"both traces are empty after phase filter: {phase}")

    starts: List[float] = []
    ends: List[float] = []
    for df in (comms_df, ops_df):
        if "start" in df.columns:
            starts.extend(df["start"].dropna().astype(float).tolist())
        if "end" in df.columns:
            ends.extend(df["end"].dropna().astype(float).tolist())

    if not starts or not ends:
        raise ValueError("cannot compute makespan because start/end columns are missing or empty")

    t0 = float(min(starts))
    t1 = float(max(ends))
    makespan = max(0.0, t1 - t0)
    if makespan <= 0:
        raise ValueError(f"non-positive makespan: {makespan}")

    device_families = _collect_device_families(
        ops_df=ops_df,
        comms_df=comms_df,
        include_cpu=include_cpu,
    )
    if not device_families:
        raise ValueError("no physical devices discovered from comms/ops traces")

    device_utils: Dict[str, float] = {}
    for dev in sorted(device_families.keys()):
        intervals: List[Tuple[float, float]] = []

        if not ops_df.empty and "device" in ops_df.columns:
            sub = ops_df[ops_df["device"].astype(str) == dev]
            if {"start", "end"}.issubset(sub.columns):
                intervals.extend(zip(sub["start"], sub["end"]))

        if not comms_df.empty:
            if "src" in comms_df.columns:
                sub = comms_df[comms_df["src"].astype(str) == dev]
                if {"start", "end"}.issubset(sub.columns):
                    intervals.extend(zip(sub["start"], sub["end"]))
            if "dst" in comms_df.columns:
                sub = comms_df[comms_df["dst"].astype(str) == dev]
                if {"start", "end"}.issubset(sub.columns):
                    intervals.extend(zip(sub["start"], sub["end"]))

        busy_time = _merge_intervals(intervals)
        device_utils[dev] = busy_time / makespan

    family_devices: Dict[str, List[str]] = {
        fam: sorted([dev for dev, f in device_families.items() if f == fam])
        for fam in sorted(set(device_families.values()))
    }
    family_counts = {fam: len(devs) for fam, devs in family_devices.items()}

    overall_utilization = float(np.mean(list(device_utils.values())))
    pim_utilization = _family_mean(device_utils, device_families, family="pim")
    accel_utilization = _family_mean(device_utils, device_families, family="accel")

    parsed = parse_trace_filename(comms_path)
    if parsed is None:
        raise ValueError(f"unexpected comms file name: {comms_path.name}")
    key, _ = parsed

    return UtilResult(
        source_algo=key.algo,
        plotted_algo=normalize_algo_token(key.algo),
        prefill=key.prefill,
        decode=key.decode,
        overall_utilization=overall_utilization,
        pim_utilization=pim_utilization,
        accel_utilization=accel_utilization,
        makespan_s=makespan,
        n_devices=len(device_utils),
        family_counts=family_counts,
        family_devices=family_devices,
        device_utils=device_utils,
        device_families=device_families,
        comms_path=comms_path,
        ops_path=ops_path,
    )


# -----------------------------
# Heft / hefthint selection
# -----------------------------

def util_pick_score(res: UtilResult, mode: str = "family_mean") -> float:
    if mode == "overall":
        return float(res.overall_utilization)
    if mode == "pim":
        return float("-inf") if res.pim_utilization is None else float(res.pim_utilization)
    if mode == "accel":
        return float("-inf") if res.accel_utilization is None else float(res.accel_utilization)

    vals: List[float] = []
    if res.pim_utilization is not None:
        vals.append(float(res.pim_utilization))
    if res.accel_utilization is not None:
        vals.append(float(res.accel_utilization))
    if not vals:
        return float("-inf")
    return float(np.mean(vals))


# -----------------------------
# Plot helpers
# -----------------------------

def pretty_algo_name(algo: str) -> str:
    return DISPLAY_NAME_MAP.get(algo, algo)


def order_algorithms(all_algos: Sequence[str],
                     algo_order: Optional[Sequence[str]] = None) -> List[str]:
    present_norm: List[str] = []
    for algo in all_algos:
        norm = normalize_algo_token(algo)
        if norm and norm not in present_norm:
            present_norm.append(norm)

    if not present_norm:
        return []

    if algo_order:
        preferred_norm: List[str] = []
        for algo in algo_order:
            norm = normalize_algo_token(algo)
            if norm in present_norm and norm not in preferred_norm:
                preferred_norm.append(norm)
        rest = [a for a in present_norm if a not in set(preferred_norm)]
        return preferred_norm + sorted(rest)

    ordered: List[str] = [a for a in PREFERRED_ALGO_ORDER if a in present_norm]
    rest = sorted([a for a in present_norm if a not in set(ordered) and a != "hefthint"])
    ordered.extend(rest)
    if "hefthint" in present_norm:
        ordered.append("hefthint")
    return ordered


def _get_or_compute(cache: Dict[TraceKey, UtilResult],
                    index: Dict[TraceKey, TracePair],
                    key: TraceKey,
                    phase: str,
                    include_cpu: bool) -> Optional[UtilResult]:
    if key in cache:
        return cache[key]
    pair = index.get(key)
    if pair is None:
        return None
    res = compute_device_utilization(
        comms_path=pair.comms_path,
        ops_path=pair.ops_path,
        phase=phase,
        include_cpu=include_cpu,
    )
    cache[key] = res
    return res


def build_results(index: Dict[TraceKey, TracePair],
                  prefills: Sequence[int],
                  decodes: Sequence[int],
                  phase: str = "all",
                  include_cpu: bool = False,
                  heft_pick_by: str = "family_mean") -> Dict[Tuple[int, int, str], UtilResult]:
    results: Dict[Tuple[int, int, str], UtilResult] = {}
    cache: Dict[TraceKey, UtilResult] = {}

    actual_algos = sorted({key.algo for key in index})
    normal_algos = [a for a in actual_algos if a not in HEFT_VARIANTS]

    for p in prefills:
        for d in decodes:
            for algo in normal_algos:
                key = TraceKey(algo=algo, prefill=p, decode=d)
                if key not in index:
                    continue
                try:
                    res = _get_or_compute(cache, index, key, phase=phase, include_cpu=include_cpu)
                    if res is not None:
                        results[(p, d, algo)] = res
                except Exception as exc:
                    print(
                        f"[WARN] skip {algo} prefill={p} decode={d}: {exc}",
                        file=sys.stderr,
                    )

            variant_results: List[UtilResult] = []
            for variant in ["heft", "hefthint"]:
                key = TraceKey(algo=variant, prefill=p, decode=d)
                if key not in index:
                    continue
                try:
                    res = _get_or_compute(cache, index, key, phase=phase, include_cpu=include_cpu)
                    if res is not None:
                        variant_results.append(res)
                except Exception as exc:
                    print(
                        f"[WARN] skip {variant} prefill={p} decode={d}: {exc}",
                        file=sys.stderr,
                    )

            if variant_results:
                def _sel_key(r: UtilResult) -> Tuple[float, int, int]:
                    score = util_pick_score(r, mode=heft_pick_by)
                    prefer_hefthint = 1 if r.source_algo == "hefthint" else 0
                    overall = int(round(r.overall_utilization * 1e12))
                    return (score, prefer_hefthint, overall)

                best = max(variant_results, key=_sel_key)
                best_for_plot = UtilResult(
                    source_algo=best.source_algo,
                    plotted_algo="hefthint",
                    prefill=best.prefill,
                    decode=best.decode,
                    overall_utilization=best.overall_utilization,
                    pim_utilization=best.pim_utilization,
                    accel_utilization=best.accel_utilization,
                    makespan_s=best.makespan_s,
                    n_devices=best.n_devices,
                    family_counts=dict(best.family_counts),
                    family_devices={k: list(v) for k, v in best.family_devices.items()},
                    device_utils=dict(best.device_utils),
                    device_families=dict(best.device_families),
                    comms_path=best.comms_path,
                    ops_path=best.ops_path,
                )
                results[(p, d, "hefthint")] = best_for_plot

                score_txt = util_pick_score(best, mode=heft_pick_by)
                print(
                    f"[INFO] prefill={p} decode={d}: choose '{best.source_algo}' for plotted label 'hefthint' "
                    f"(criterion={heft_pick_by}, score={score_txt:.6f}) from {best.comms_path.parent}",
                    file=sys.stderr,
                )

    return results


def _family_util_for_plot(res: Optional[UtilResult], family: str) -> float:
    if res is None:
        return float("nan")
    if family == "pim":
        return float("nan") if res.pim_utilization is None else float(res.pim_utilization)
    if family == "accel":
        return float("nan") if res.accel_utilization is None else float(res.accel_utilization)
    raise ValueError(f"unknown family: {family}")


def _family_diff_for_plot(res: Optional[UtilResult],
                          diff_mode: str,
                          diff_order: str) -> float:
    if diff_mode == "none":
        return float("nan")
    if res is None or res.pim_utilization is None or res.accel_utilization is None:
        return float("nan")

    pim = float(res.pim_utilization)
    accel = float(res.accel_utilization)
    delta = accel - pim if diff_order == "accel-pim" else pim - accel
    if diff_mode == "signed":
        return delta
    if diff_mode == "abs":
        return abs(accel - pim)
    raise ValueError(f"unknown diff_mode: {diff_mode}")


def _diff_expr_text(diff_mode: str, diff_order: str) -> str:
    accel_name = FAMILY_LABELS["accel"]
    pim_name = FAMILY_LABELS["pim"]
    if diff_mode == "abs":
        return f"|{accel_name} - {pim_name}|"
    if diff_order == "accel-pim":
        return f"{accel_name} - {pim_name}"
    return f"{pim_name} - {accel_name}"


def _diff_ylabel(util_scale: str, diff_mode: str, diff_order: str) -> str:
    expr = _diff_expr_text(diff_mode=diff_mode, diff_order=diff_order)
    if util_scale == "percent":
        return f"Gap: {expr} (pp)"
    return f"Gap: {expr}"


def _save_summary_csv(results: Dict[Tuple[int, int, str], UtilResult], output_path: Path) -> None:
    rows: List[Dict[str, object]] = []
    for (prefill, decode, plotted_algo), res in sorted(results.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        accel_minus_pim = float("nan")
        abs_gap = float("nan")
        if res.pim_utilization is not None and res.accel_utilization is not None:
            accel_minus_pim = float(res.accel_utilization - res.pim_utilization)
            abs_gap = float(abs(res.accel_utilization - res.pim_utilization))

        rows.append({
            "prefill": prefill,
            "decode": decode,
            "plotted_algo": plotted_algo,
            "source_algo": res.source_algo,
            "overall_utilization": res.overall_utilization,
            "pim_utilization": res.pim_utilization,
            "accel_utilization": res.accel_utilization,
            "accel_minus_pim": accel_minus_pim,
            "abs_family_gap": abs_gap,
            "makespan_s": res.makespan_s,
            "n_devices": res.n_devices,
            "n_pim_devices": res.family_counts.get("pim", 0),
            "n_accel_devices": res.family_counts.get("accel", 0),
            "comms_path": str(res.comms_path),
            "ops_path": str(res.ops_path),
        })

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] saved summary -> {output_path.resolve()}")


def plot_results(results: Dict[Tuple[int, int, str], UtilResult],
                 prefills: Sequence[int],
                 decodes: Sequence[int],
                 algorithms: Sequence[str],
                 output: Path,
                 colors: Sequence[str],
                 title: Optional[str] = None,
                 util_scale: str = "percent",
                 diff_mode: str = "signed",
                 diff_order: str = "accel-pim",
                 dpi: int = 200,
                 marker_size: float = 5.5,
                 line_width: float = 1.8,
                 max_cols: int = 2) -> None:
    if not prefills:
        raise ValueError("no prefill lengths to plot")
    if not decodes:
        raise ValueError("no decode lengths to plot")
    if not algorithms:
        raise ValueError("no algorithms to plot")
    if not colors:
        raise ValueError("colors list is empty")

    show_diff = diff_mode != "none"
    n_subplots = len(prefills)
    ncols = max(1, min(max_cols, n_subplots))
    nrows_base = int(math.ceil(n_subplots / ncols))
    total_rows = nrows_base * (2 if show_diff else 1)

    per_ax_w = max(5.4, 0.72 * len(algorithms))
    fig_w = min(28.0, per_ax_w * ncols)
    per_row_h = 3.4 if show_diff else 3.9
    fig_h = max(per_row_h * total_rows, 4.2)

    fig, axes = plt.subplots(total_rows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    x = np.arange(len(algorithms), dtype=float)

    if util_scale == "percent":
        y_factor = 100.0
        abs_y_label = "Avg. device utilization (%)"
        abs_y_lim = (0.0, 100.0)
        diff_min_span = 5.0
    else:
        y_factor = 1.0
        abs_y_label = "Avg. device utilization"
        abs_y_lim = (0.0, 1.0)
        diff_min_span = 0.05

    diff_values_scaled: List[float] = []
    if show_diff:
        for p in prefills:
            for d in decodes:
                for algo in algorithms:
                    val = _family_diff_for_plot(results.get((p, d, algo)), diff_mode=diff_mode, diff_order=diff_order)
                    if not np.isnan(val):
                        diff_values_scaled.append(val * y_factor)

        if diff_mode == "signed":
            max_abs = max([abs(v) for v in diff_values_scaled], default=0.0)
            bound = max(diff_min_span, max_abs * 1.15)
            diff_y_lim = (-bound, bound)
        else:
            upper = max(diff_min_span, max(diff_values_scaled, default=0.0) * 1.15)
            diff_y_lim = (0.0, upper)
    else:
        diff_y_lim = (0.0, 1.0)

    used_slots = set()

    for idx, p in enumerate(prefills):
        row = idx // ncols
        col = idx % ncols
        abs_ax = axes[row, col]
        used_slots.add((row, col))

        diff_ax = None
        if show_diff:
            diff_ax = axes[row + nrows_base, col]
            used_slots.add((row + nrows_base, col))

        any_abs_line = False
        any_diff_line = False

        for line_idx, d in enumerate(decodes):
            color = colors[line_idx % len(colors)]

            for family in FAMILY_ORDER:
                y_vals = np.asarray([
                    _family_util_for_plot(results.get((p, d, algo)), family=family) * y_factor
                    for algo in algorithms
                ], dtype=float)

                if np.all(np.isnan(y_vals)):
                    continue

                any_abs_line = True
                x_vals = x + FAMILY_X_OFFSETS[family]
                abs_ax.plot(
                    x_vals,
                    y_vals,
                    marker=FAMILY_MARKERS[family],
                    markersize=marker_size,
                    linewidth=line_width,
                    color=color,
                    label=None,
                )

            if diff_ax is not None:
                diff_vals = np.asarray([
                    _family_diff_for_plot(results.get((p, d, algo)), diff_mode=diff_mode, diff_order=diff_order) * y_factor
                    for algo in algorithms
                ], dtype=float)
                if not np.all(np.isnan(diff_vals)):
                    any_diff_line = True
                    diff_ax.plot(
                        x,
                        diff_vals,
                        marker=DIFF_MARKER,
                        markersize=marker_size,
                        linewidth=line_width,
                        linestyle=DIFF_LINESTYLE,
                        color=color,
                        label=None,
                    )

        abs_title = f"prefill={p} | absolute" if show_diff else f"prefill={p}"
        abs_ax.set_title(abs_title)
        abs_ax.set_ylim(*abs_y_lim)
        abs_ax.set_xlim(-0.5, len(algorithms) - 0.5)
        abs_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        abs_ax.set_xticks(x)
        if show_diff:
            abs_ax.set_xticklabels([])
        else:
            abs_ax.set_xticklabels([pretty_algo_name(a) for a in algorithms], rotation=30, ha="right")

        if col == 0:
            abs_ax.set_ylabel(abs_y_label)

        if not any_abs_line:
            abs_ax.text(0.5, 0.5, "No data", transform=abs_ax.transAxes,
                        ha="center", va="center", fontsize=12)

        if diff_ax is not None:
            diff_ax.set_title(f"prefill={p} | gap")
            diff_ax.set_ylim(*diff_y_lim)
            diff_ax.set_xlim(-0.5, len(algorithms) - 0.5)
            diff_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
            diff_ax.set_xticks(x)
            diff_ax.set_xticklabels([pretty_algo_name(a) for a in algorithms], rotation=30, ha="right")
            if diff_mode == "signed":
                diff_ax.axhline(0.0, linestyle=":", linewidth=0.9, color="black", alpha=0.55)
            if col == 0:
                diff_ax.set_ylabel(_diff_ylabel(util_scale=util_scale, diff_mode=diff_mode, diff_order=diff_order))
            if not any_diff_line:
                diff_ax.text(0.5, 0.5, "No data", transform=diff_ax.transAxes,
                             ha="center", va="center", fontsize=12)

    for r in range(total_rows):
        for c in range(ncols):
            if (r, c) not in used_slots:
                axes[r, c].axis("off")

    if title:
        fig.suptitle(title, y=1.02)

    legend_handles: List[Line2D] = [
        Line2D([0], [0], color="black", lw=0, marker=FAMILY_MARKERS[family],
               markersize=marker_size + 0.5, label=FAMILY_LABELS[family])
        for family in FAMILY_ORDER
    ]

    if show_diff:
        legend_handles.append(
            Line2D([0], [0], color="black", lw=line_width, linestyle=DIFF_LINESTYLE,
                   marker=DIFF_MARKER, markersize=marker_size + 0.5,
                   label=f"gap: {_diff_expr_text(diff_mode=diff_mode, diff_order=diff_order)}")
        )

    legend_handles.extend([
        Line2D([0], [0], color=colors[i % len(colors)], lw=line_width, marker=None,
               label=f"decode={d}")
        for i, d in enumerate(decodes)
    ])

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_handles), max(4, ncols * 3)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    top_margin = 0.90 if title else 0.92
    fig.tight_layout(rect=[0.02, 0.02, 0.98, top_margin])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved figure -> {output.resolve()}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-dir", type=str, required=True,
                    help="Root directory that contains all algo_* subdirectories and trace csv files.")
    ap.add_argument("--prefills", type=str, default=None,
                    help="Prefill lengths: 128,256,512 or 128:512:128 or all. Omit to auto-scan all.")
    ap.add_argument("--decodes", type=str, default=None,
                    help="Decode lengths: 64,128,256 or 64:512:64 or all. Omit to auto-scan all.")
    ap.add_argument("--algo-order", type=str, default=None,
                    help="Optional x-axis algorithm order, e.g. pd,ianus,facil,hefthint,attacc")
    ap.add_argument("--output", type=str, default="device_utilization.png")
    ap.add_argument("--summary-csv", type=str, default=None,
                    help="Optional CSV dump of all computed utilizations / gaps.")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--phase", type=str, choices=["all", "prefill", "decode"], default="all",
                    help="Use all trace events, or only one phase.")
    ap.add_argument("--include-cpu", action="store_true",
                    help="Also include CPU when computing overall-utilization-based statistics. CPU is not plotted.")
    ap.add_argument("--heft-pick-by", type=str,
                    choices=["family_mean", "overall", "pim", "accel"],
                    default="family_mean",
                    help=(
                        "How to choose between heft and hefthint for each (prefill, decode): "
                        "family_mean = mean of plotted PIM/NPU-GPU utilizations; "
                        "overall = all physical devices; pim = PIM only; accel = NPU/GPU only."
                    ))
    ap.add_argument("--colors", type=str, default=",".join(DEFAULT_COLORS),
                    help="Comma-separated line colors for decode lengths.")
    ap.add_argument("--util-scale", type=str, choices=["percent", "fraction"], default="percent",
                    help="Display utilization as 0~100 percent or 0~1 fraction.")
    ap.add_argument("--diff-mode", type=str, choices=["signed", "abs", "none"], default="signed",
                    help=(
                        "How to plot the family gap: signed = keep sign, "
                        "abs = absolute gap, none = disable the extra gap subplot."
                    ))
    ap.add_argument("--diff-order", type=str, choices=["accel-pim", "pim-accel"], default="accel-pim",
                    help="Signed-gap direction. Ignored when --diff-mode abs/none.")
    ap.add_argument("--max-cols", type=int, default=2,
                    help="Maximum number of subplot columns for prefills.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--marker-size", type=float, default=5.5)
    ap.add_argument("--line-width", type=float, default=1.8)

    args = ap.parse_args()

    search_dir = Path(args.search_dir)
    if not search_dir.exists():
        ap.error(f"--search-dir not found: {search_dir}")

    paths = discover_trace_files(search_dir)
    if not paths:
        ap.error(f"No *_trace.csv found under: {search_dir}")

    index = build_trace_index(paths)
    if not index:
        ap.error("No valid comms/ops trace pairs found. Check file names and directory.")

    all_prefills = sorted({key.prefill for key in index})
    all_decodes = sorted({key.decode for key in index})
    all_algos_norm = sorted({normalize_algo_token(key.algo) for key in index})

    try:
        prefills = parse_int_list(args.prefills) or all_prefills
        decodes = parse_int_list(args.decodes) or all_decodes
    except ValueError as exc:
        ap.error(str(exc))

    colors = parse_str_list(args.colors) or DEFAULT_COLORS
    algo_order = parse_str_list(args.algo_order)
    algorithms = order_algorithms(all_algos_norm, algo_order=algo_order)

    missing_prefills = [p for p in prefills if p not in all_prefills]
    missing_decodes = [d for d in decodes if d not in all_decodes]
    if missing_prefills:
        print(f"[WARN] requested prefills not found under search-dir: {missing_prefills}", file=sys.stderr)
    if missing_decodes:
        print(f"[WARN] requested decodes not found under search-dir: {missing_decodes}", file=sys.stderr)

    results = build_results(
        index=index,
        prefills=prefills,
        decodes=decodes,
        phase=args.phase,
        include_cpu=args.include_cpu,
        heft_pick_by=args.heft_pick_by,
    )

    if not results:
        ap.error("No usable results for the specified prefills/decodes.")

    active_algos = []
    for algo in algorithms:
        if any(k[2] == algo for k in results.keys()):
            active_algos.append(algo)

    if args.summary_csv:
        _save_summary_csv(results=results, output_path=Path(args.summary_csv))

    plot_results(
        results=results,
        prefills=prefills,
        decodes=decodes,
        algorithms=active_algos,
        output=Path(args.output),
        colors=colors,
        title=args.title,
        util_scale=args.util_scale,
        diff_mode=args.diff_mode,
        diff_order=args.diff_order,
        dpi=args.dpi,
        marker_size=args.marker_size,
        line_width=args.line_width,
        max_cols=max(1, args.max_cols),
    )


if __name__ == "__main__":
    main()
