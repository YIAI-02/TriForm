#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot speedup + absolute device utilization + family utilization gap from trace CSVs.

Supported file name pattern (recursive scan under --search-dir):
  <algo>_prefill-<P>xdecode_<D>_<comms|ops>_trace.csv

Examples
--------
# Plot selected prefills / decodes and show signed gap = (accel - pim)
python3 plot_exp1_utilization.py \
  --search-dir ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst64_rst64/qwen_1.8b_fp16_b4_s64 \
  --prefills 128,1024\
  --decodes 128,512,1024 \
  --exclude-algos weights_on_pim\
  --algo-label-map 'hefthint=Bifocal (this work)' \
  --highlight-algo 'heft'\
  --output ../../figs/exp1/util/qwen_1_8b_fp16_b4_s64_utilization.pdf

python3 plot_exp1_utilization.py \
  --search-dir ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst64_rst64/llama_7b_fp16_b16_s64 \
  --prefills 128 \
  --decodes 128,512,1024 \
  --output ../../figs/exp1/util/llama_7b_fp16_b16_s64_s64_utilization.pdf

python3 plot_exp1_utilization.py \
  --search-dir ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst64_rst64/qwen_1.8b_fp16_b8_s64 \
  --prefills 128,256,512,1024 \
  --decodes 64,128,256,512,1024 \
  --output ../../figs/exp1/util/qwen_1_8b_fp16_b8_s64_utilization.pdf

prefill panel的高度
per_group_h
fig_h = max(per_group_h * n_group_rows, 5.8)
三个图比例
inner = outer[group, col].subgridspec(3, 1, hspace=0.03)
inner = outer[group, col].subgridspec(
    3, 1,
    hspace=0.03,
    height_ratios=[0.9, 1.2, 0.9]
)
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    "hefthint": "Bifocal（this work）",
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
BASELINE_COMPARE_RE = re.compile(
    r"^baseline_compare_(?P<prefill>\d+)x(?P<decode>\d+)\.json$",
    re.IGNORECASE,
)

SPEEDUP_MARKER = "D"
PREFERRED_ALGO_ORDER: List[str] = ["pd", "ianus", "facil"]

CACHE_VERSION = "v4"


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
    concurrent_utilization: Optional[float]
    makespan_s: float
    n_devices: int
    family_counts: Dict[str, int]
    family_devices: Dict[str, List[str]]
    device_utils: Dict[str, float]
    device_families: Dict[str, str]
    comms_path: Path
    ops_path: Path


@dataclass
class TimeResult:
    source_algo: str
    plotted_algo: str
    prefill: int
    decode: int
    prefill_time_s: Optional[float]
    decode_time_s: Optional[float]
    total_time_s: Optional[float]
    json_path: Path


# -----------------------------
# Cache helpers
# -----------------------------

def util_result_to_dict(res: UtilResult) -> Dict[str, object]:
    return {
        "source_algo": res.source_algo,
        "plotted_algo": res.plotted_algo,
        "prefill": res.prefill,
        "decode": res.decode,
        "overall_utilization": res.overall_utilization,
        "pim_utilization": res.pim_utilization,
        "accel_utilization": res.accel_utilization,
        "concurrent_utilization": res.concurrent_utilization,
        "makespan_s": res.makespan_s,
        "n_devices": res.n_devices,
        "family_counts": res.family_counts,
        "family_devices": res.family_devices,
        "device_utils": res.device_utils,
        "device_families": res.device_families,
        "comms_path": str(res.comms_path),
        "ops_path": str(res.ops_path),
    }


def util_result_from_dict(data: Dict[str, object]) -> UtilResult:
    return UtilResult(
        source_algo=str(data["source_algo"]),
        plotted_algo=str(data["plotted_algo"]),
        prefill=int(data["prefill"]),
        decode=int(data["decode"]),
        overall_utilization=float(data["overall_utilization"]),
        pim_utilization=None if data["pim_utilization"] is None else float(data["pim_utilization"]),
        accel_utilization=None if data["accel_utilization"] is None else float(data["accel_utilization"]),
        concurrent_utilization=None if data["concurrent_utilization"] is None else float(data["concurrent_utilization"]),
        makespan_s=float(data["makespan_s"]),
        n_devices=int(data["n_devices"]),
        family_counts={str(k): int(v) for k, v in dict(data["family_counts"]).items()},
        family_devices={str(k): [str(x) for x in v] for k, v in dict(data["family_devices"]).items()},
        device_utils={str(k): float(v) for k, v in dict(data["device_utils"]).items()},
        device_families={str(k): str(v) for k, v in dict(data["device_families"]).items()},
        comms_path=Path(str(data["comms_path"])),
        ops_path=Path(str(data["ops_path"])),
    )


def _file_fingerprint(path: Path) -> Dict[str, object]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
    }


def _util_time_signature(time_res: Optional[TimeResult]) -> Dict[str, Optional[float]]:
    if time_res is None:
        return {
            "prefill_time_s": None,
            "decode_time_s": None,
            "total_time_s": None,
        }
    return {
        "prefill_time_s": None if time_res.prefill_time_s is None else float(time_res.prefill_time_s),
        "decode_time_s": None if time_res.decode_time_s is None else float(time_res.decode_time_s),
        "total_time_s": None if time_res.total_time_s is None else float(time_res.total_time_s),
    }


def _util_cache_key(comms_path: Path,
                    ops_path: Path,
                    phase: str,
                    include_cpu: bool,
                    time_res: Optional[TimeResult]) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "phase": phase,
        "include_cpu": include_cpu,
        "time_signature": _util_time_signature(time_res),
        "comms": _file_fingerprint(comms_path),
        "ops": _file_fingerprint(ops_path),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_util_cache(cache_dir: Path,
                     comms_path: Path,
                     ops_path: Path,
                     phase: str,
                     include_cpu: bool,
                     time_res: Optional[TimeResult]) -> Optional[UtilResult]:
    key = _util_cache_key(comms_path, ops_path, phase, include_cpu, time_res)
    cache_path = cache_dir / f"{key}.json"
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text())
        return util_result_from_dict(payload["result"])
    except Exception as exc:
        print(f"[WARN] failed to read cache {cache_path}: {exc}", file=sys.stderr)
        return None


def _save_util_cache(cache_dir: Path,
                     res: UtilResult,
                     phase: str,
                     include_cpu: bool,
                     time_res: Optional[TimeResult]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _util_cache_key(res.comms_path, res.ops_path, phase, include_cpu, time_res)
    cache_path = cache_dir / f"{key}.json"

    payload = {
        "cache_version": CACHE_VERSION,
        "phase": phase,
        "include_cpu": include_cpu,
        "time_signature": _util_time_signature(time_res),
        "result": util_result_to_dict(res),
    }

    try:
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"[WARN] failed to write cache {cache_path}: {exc}", file=sys.stderr)


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


def parse_name_map(s: Optional[str]) -> Dict[str, str]:
    if s is None:
        return {}
    raw = s.strip()
    if not raw:
        return {}
    parts = raw.split(";") if ";" in raw else raw.split(",")
    out: Dict[str, str] = {}
    for part in parts:
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"invalid name-map item '{item}'. Use key=value pairs separated by ';'"
            )
        key, value = item.split("=", 1)
        norm_key = normalize_algo_token(key.strip())
        display = value.strip()
        if not norm_key:
            raise ValueError(f"invalid algorithm key in name map: '{item}'")
        if not display:
            raise ValueError(f"empty display name in name map: '{item}'")
        out[norm_key] = display
    return out


def discover_trace_files(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("*_trace.csv"))


def discover_baseline_compare_files(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("baseline_compare_*.json"))


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


def parse_baseline_compare_filename(path: Path) -> Optional[Tuple[int, int]]:
    m = BASELINE_COMPARE_RE.match(path.name)
    if not m:
        return None
    return int(m.group("prefill")), int(m.group("decode"))


def _safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def _policy_to_algo(policy: object) -> str:
    s = str(policy or "").strip().lower()
    if not s:
        return ""
    if s.startswith("algo:"):
        s = s.split(":", 1)[1]
    return s.strip()


def build_time_index(paths: Sequence[Path]) -> Dict[Tuple[int, int, str], TimeResult]:
    out: Dict[Tuple[int, int, str], TimeResult] = {}

    for path in sorted(paths):
        parsed = parse_baseline_compare_filename(path)
        if parsed is None:
            continue
        prefill, decode = parsed

        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"[WARN] skip invalid baseline json {path}: {exc}", file=sys.stderr)
            continue

        rows = payload.get("results", [])
        if not isinstance(rows, list):
            print(f"[WARN] skip baseline json without list 'results': {path}", file=sys.stderr)
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            source_algo = _policy_to_algo(row.get("policy"))
            if not source_algo:
                continue

            key = (prefill, decode, source_algo)
            time_res = TimeResult(
                source_algo=source_algo,
                plotted_algo=normalize_algo_token(source_algo),
                prefill=prefill,
                decode=decode,
                prefill_time_s=_safe_float(row.get("prefill_time_s")),
                decode_time_s=_safe_float(row.get("decode_time_s")),
                total_time_s=_safe_float(row.get("total_time_s")),
                json_path=path,
            )

            if key in out:
                print(
                    f"[WARN] duplicate timing for prefill={prefill} decode={decode} algo={source_algo}; keep first from {out[key].json_path}",
                    file=sys.stderr,
                )
                continue
            out[key] = time_res

    return out


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
                        type_hints[str(dev)].append("")

        if "dst" in comms_df.columns:
            if "dst_type" in comms_df.columns:
                for dev, dtype in zip(comms_df["dst"], comms_df["dst_type"]):
                    dev_s = _safe_lower(dev)
                    if dev_s:
                        type_hints[str(dev)].append(str(dtype))
            else:
                for dev in comms_df["dst"].dropna().astype(str):
                    if _safe_lower(dev):
                        type_hints[str(dev)].append("")

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
        return df.copy()
    if "phase" not in df.columns:
        return df.copy()
    return df[df["phase"].astype(str).str.lower() == phase.lower()].copy()


def _clip_intervals_to_window(intervals: Iterable[Tuple[float, float]],
                              window_start: float,
                              window_end: float) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    for start, end in intervals:
        if pd.isna(start) or pd.isna(end):
            continue
        s = max(float(start), float(window_start))
        e = min(float(end), float(window_end))
        if e <= s:
            continue
        clipped.append((s, e))
    return clipped


def _resolve_phase_window(time_res: Optional[TimeResult],
                          phase: str) -> Tuple[float, float, float]:
    if time_res is None:
        raise ValueError("missing baseline_compare timing for utilization")

    prefill_s = _safe_float(time_res.prefill_time_s)
    decode_s = _safe_float(time_res.decode_time_s)
    total_s = _safe_float(time_res.total_time_s)

    if phase == "all":
        runtime_s = total_s
        if runtime_s is None and prefill_s is not None and decode_s is not None:
            runtime_s = prefill_s + decode_s
        if runtime_s is None or runtime_s <= 0:
            raise ValueError("baseline_compare is missing a valid total_time_s")
        return 0.0, float(runtime_s), float(runtime_s)

    if phase == "prefill":
        if prefill_s is None or prefill_s <= 0:
            raise ValueError("baseline_compare is missing a valid prefill_time_s")
        return 0.0, float(prefill_s), float(prefill_s)

    if phase == "decode":
        if decode_s is None or decode_s <= 0:
            raise ValueError("baseline_compare is missing a valid decode_time_s")

        if prefill_s is not None and prefill_s >= 0:
            start = float(prefill_s)
        elif total_s is not None and total_s >= decode_s:
            start = float(total_s - decode_s)
        else:
            start = 0.0

        end = start + float(decode_s)
        return start, end, float(decode_s)

    raise ValueError(f"unknown phase: {phase}")


def _merge_intervals(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(e - s for s, e in _normalize_intervals(intervals))


def _normalize_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
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
        return []

    arr.sort(key=lambda x: (x[0], x[1]))
    cur_s, cur_e = arr[0]
    merged: List[Tuple[float, float]] = []
    for s, e in arr[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


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


def _family_concurrent_utilization(device_busy_intervals: Dict[str, List[Tuple[float, float]]],
                                   device_families: Dict[str, str],
                                   family_counts: Dict[str, int],
                                   makespan: float,
                                   lhs_family: str = "pim",
                                   rhs_family: str = "accel") -> Optional[float]:
    lhs_n = int(family_counts.get(lhs_family, 0))
    rhs_n = int(family_counts.get(rhs_family, 0))
    if lhs_n <= 0 or rhs_n <= 0 or makespan <= 0:
        return None

    family_event_deltas: DefaultDict[float, Dict[str, int]] = defaultdict(
        lambda: {lhs_family: 0, rhs_family: 0}
    )

    for dev, merged in device_busy_intervals.items():
        family = device_families.get(dev)
        if family not in {lhs_family, rhs_family}:
            continue
        for start, end in merged:
            family_event_deltas[start][family] += 1
            family_event_deltas[end][family] -= 1

    timeline = sorted(set(family_event_deltas.keys()))
    if not timeline:
        return 0.0

    all_times = sorted(set(timeline + [0.0]))
    active = {lhs_family: 0, rhs_family: 0}
    area = 0.0

    prev = all_times[0]
    for family, delta in family_event_deltas.get(prev, {}).items():
        active[family] += delta

    for t in all_times[1:]:
        dt = t - prev
        if dt > 0:
            lhs_util = active[lhs_family] / lhs_n
            rhs_util = active[rhs_family] / rhs_n
            area += dt * min(lhs_util, rhs_util)
        for family, delta in family_event_deltas.get(t, {}).items():
            active[family] += delta
        prev = t

    tail = makespan - prev
    if tail > 0:
        lhs_util = active[lhs_family] / lhs_n
        rhs_util = active[rhs_family] / rhs_n
        area += tail * min(lhs_util, rhs_util)

    return float(area / makespan)


def compute_device_utilization(comms_path: Path,
                               ops_path: Path,
                               phase: str = "all",
                               include_cpu: bool = False,
                               time_res: Optional[TimeResult] = None) -> UtilResult:
    ops_df = _filter_phase(_read_trace_csv(ops_path), phase=phase)
    if ops_df.empty:
        raise ValueError(f"ops trace is empty after phase filter: {phase}")
    if "device" not in ops_df.columns:
        raise ValueError(f"ops trace missing 'device' column: {ops_path}")
    if not {"start", "end"}.issubset(ops_df.columns):
        raise ValueError(f"ops trace missing start/end columns: {ops_path}")

    window_start, window_end, runtime_s = _resolve_phase_window(time_res, phase)

    device_families = _collect_device_families(
        ops_df=ops_df,
        comms_df=pd.DataFrame(),
        include_cpu=include_cpu,
    )
    if not device_families:
        raise ValueError("no physical devices discovered from ops trace")

    device_busy_intervals: Dict[str, List[Tuple[float, float]]] = {}
    device_utils: Dict[str, float] = {}
    for dev in sorted(device_families.keys()):
        sub = ops_df[ops_df["device"].astype(str) == dev]
        intervals = _clip_intervals_to_window(
            zip(sub["start"], sub["end"]),
            window_start=window_start,
            window_end=window_end,
        )
        merged_intervals = [
            (s - window_start, e - window_start)
            for s, e in _normalize_intervals(intervals)
        ]
        device_busy_intervals[dev] = merged_intervals
        busy_time = sum(e - s for s, e in merged_intervals)
        device_utils[dev] = busy_time / runtime_s

    family_devices: Dict[str, List[str]] = {
        fam: sorted([dev for dev, f in device_families.items() if f == fam])
        for fam in sorted(set(device_families.values()))
    }
    family_counts = {fam: len(devs) for fam, devs in family_devices.items()}

    overall_utilization = float(np.mean(list(device_utils.values()))) if device_utils else 0.0
    pim_utilization = _family_mean(device_utils, device_families, family="pim")
    accel_utilization = _family_mean(device_utils, device_families, family="accel")
    concurrent_utilization = _family_concurrent_utilization(
        device_busy_intervals=device_busy_intervals,
        device_families=device_families,
        family_counts=family_counts,
        makespan=runtime_s,
        lhs_family="pim",
        rhs_family="accel",
    )

    parsed = parse_trace_filename(ops_path)
    if parsed is None:
        parsed = parse_trace_filename(comms_path)
    if parsed is None:
        raise ValueError(f"unexpected trace file name: {ops_path.name}")
    key, _ = parsed

    return UtilResult(
        source_algo=key.algo,
        plotted_algo=normalize_algo_token(key.algo),
        prefill=key.prefill,
        decode=key.decode,
        overall_utilization=overall_utilization,
        pim_utilization=pim_utilization,
        accel_utilization=accel_utilization,
        concurrent_utilization=concurrent_utilization,
        makespan_s=runtime_s,
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

def pretty_algo_name(algo: str,
                     display_name_map: Optional[Dict[str, str]] = None) -> str:
    name_map = DISPLAY_NAME_MAP if display_name_map is None else display_name_map
    return name_map.get(algo, algo)


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


def _get_or_compute(mem_cache: Dict[TraceKey, UtilResult],
                    index: Dict[TraceKey, TracePair],
                    key: TraceKey,
                    phase: str,
                    include_cpu: bool,
                    time_res: Optional[TimeResult],
                    disk_cache_dir: Optional[Path] = None) -> Optional[UtilResult]:
    if key in mem_cache:
        return mem_cache[key]

    pair = index.get(key)
    if pair is None:
        return None

    if time_res is None:
        raise ValueError(
            f"missing baseline_compare timing for algo={key.algo} prefill={key.prefill} decode={key.decode}"
        )

    if disk_cache_dir is not None:
        cached = _load_util_cache(
            cache_dir=disk_cache_dir,
            comms_path=pair.comms_path,
            ops_path=pair.ops_path,
            phase=phase,
            include_cpu=include_cpu,
            time_res=time_res,
        )
        if cached is not None:
            print(
                f"[CACHE HIT] {key.algo} prefill={key.prefill} decode={key.decode} phase={phase}",
                file=sys.stderr,
            )
            mem_cache[key] = cached
            return cached

    res = compute_device_utilization(
        comms_path=pair.comms_path,
        ops_path=pair.ops_path,
        phase=phase,
        include_cpu=include_cpu,
        time_res=time_res,
    )
    mem_cache[key] = res

    if disk_cache_dir is not None:
        _save_util_cache(
            cache_dir=disk_cache_dir,
            res=res,
            phase=phase,
            include_cpu=include_cpu,
            time_res=time_res,
        )

    return res


def build_results(index: Dict[TraceKey, TracePair],
                  time_index: Dict[Tuple[int, int, str], TimeResult],
                  prefills: Sequence[int],
                  decodes: Sequence[int],
                  phase: str = "all",
                  include_cpu: bool = False,
                  heft_pick_by: str = "family_mean",
                  disk_cache_dir: Optional[Path] = None) -> Dict[Tuple[int, int, str], UtilResult]:
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
                    res = _get_or_compute(
                        cache,
                        index,
                        key,
                        phase=phase,
                        include_cpu=include_cpu,
                        time_res=time_index.get((p, d, algo)),
                        disk_cache_dir=disk_cache_dir,
                    )
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
                    res = _get_or_compute(
                        cache,
                        index,
                        key,
                        phase=phase,
                        include_cpu=include_cpu,
                        time_res=time_index.get((p, d, variant)),
                        disk_cache_dir=disk_cache_dir,
                    )
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
                    concurrent_utilization=best.concurrent_utilization,
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


def _lookup_time_result(time_index: Dict[Tuple[int, int, str], TimeResult],
                        prefill: int,
                        decode: int,
                        plotted_algo: str,
                        res: Optional[UtilResult]) -> Optional[TimeResult]:
    candidates: List[str] = []
    if res is not None and res.source_algo:
        candidates.append(res.source_algo)

    norm_algo = normalize_algo_token(plotted_algo)
    if norm_algo == "hefthint":
        candidates.extend(["hefthint", "heft"])
    else:
        candidates.append(norm_algo)

    seen = set()
    for cand in candidates:
        cand = (cand or "").strip().lower()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        time_res = time_index.get((prefill, decode, cand))
        if time_res is not None:
            return time_res
    return None


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "nan"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(v):
        return "nan"
    return f"{100.0 * v:.2f}%"


def report_over_100_utilization(results: Dict[Tuple[int, int, str], UtilResult],
                                time_index: Dict[Tuple[int, int, str], TimeResult],
                                threshold: float = 1.0,
                                eps: float = 1e-9) -> int:
    case_count = 0

    for (prefill, decode, plotted_algo), res in sorted(results.items()):
        metric_issues: List[Tuple[str, float]] = []
        for metric_name, metric_value in [
            ("overall", res.overall_utilization),
            ("pim", res.pim_utilization),
            ("accel", res.accel_utilization),
            ("co_util", res.concurrent_utilization),
        ]:
            if metric_value is None:
                continue
            value = float(metric_value)
            if math.isnan(value):
                continue
            if value > threshold + eps:
                metric_issues.append((metric_name, value))

        device_issues: List[Tuple[str, str, float]] = []
        for dev, value in sorted(res.device_utils.items(), key=lambda kv: (-kv[1], kv[0])):
            if value > threshold + eps:
                device_issues.append((dev, res.device_families.get(dev, "unknown"), float(value)))

        if not metric_issues and not device_issues:
            continue

        case_count += 1
        time_res = _lookup_time_result(
            time_index=time_index,
            prefill=prefill,
            decode=decode,
            plotted_algo=plotted_algo,
            res=res,
        )

        print(
            f"[OVER100] prefill={prefill} decode={decode} plotted_algo={plotted_algo} "
            f"source_algo={res.source_algo}",
            file=sys.stderr,
        )

        for metric_name, value in metric_issues:
            print(
                f"  metric={metric_name:<8} value={value:.6f} ({100.0 * value:.2f}%)",
                file=sys.stderr,
            )
            if metric_name in {"pim", "accel"}:
                fam_devs = res.family_devices.get(metric_name, [])
                if fam_devs:
                    fam_desc = ", ".join(
                        f"{dev}={_fmt_pct(res.device_utils.get(dev))}" for dev in fam_devs
                    )
                    print(f"    family_devices[{metric_name}] -> {fam_desc}", file=sys.stderr)

        if device_issues:
            print("  devices_over100:", file=sys.stderr)
            for dev, family, value in device_issues:
                print(
                    f"    {dev} family={family} value={value:.6f} ({100.0 * value:.2f}%)",
                    file=sys.stderr,
                )

        if res.family_devices:
            print("  family_device_utils:", file=sys.stderr)
            for family in sorted(res.family_devices):
                devs = res.family_devices[family]
                desc = ", ".join(
                    f"{dev}={_fmt_pct(res.device_utils.get(dev))}" for dev in devs
                )
                print(f"    {family}: {desc}", file=sys.stderr)

        print(
            f"  makespan_s={res.makespan_s:.9f} n_devices={res.n_devices} "
            f"family_counts={res.family_counts}",
            file=sys.stderr,
        )
        if time_res is not None:
            print(
                f"  baseline={time_res.json_path} "
                f"prefill_time_s={time_res.prefill_time_s} "
                f"decode_time_s={time_res.decode_time_s} "
                f"total_time_s={time_res.total_time_s}",
                file=sys.stderr,
            )
        else:
            print("  baseline=<missing>", file=sys.stderr)

        print(f"  ops_path={res.ops_path}", file=sys.stderr)
        print(f"  comms_path={res.comms_path}", file=sys.stderr)

    return case_count


def _family_util_for_plot(res: Optional[UtilResult], family: str) -> float:
    if res is None:
        return float("nan")
    if family == "pim":
        return float("nan") if res.pim_utilization is None else float(res.pim_utilization)
    if family == "accel":
        return float("nan") if res.accel_utilization is None else float(res.accel_utilization)
    raise ValueError(f"unknown family: {family}")


def _default_time_field_for_phase(phase: str) -> str:
    if phase == "prefill":
        return "prefill_time_s"
    if phase == "decode":
        return "decode_time_s"
    return "total_time_s"


def _time_value(time_res: Optional[TimeResult], field: str) -> Optional[float]:
    if time_res is None:
        return None
    v = getattr(time_res, field, None)
    if v is None:
        return None
    if v <= 0:
        return None
    return float(v)


def lookup_runtime_s(time_index: Dict[Tuple[int, int, str], TimeResult],
                     prefill: int,
                     decode: int,
                     plotted_algo: str,
                     res: Optional[UtilResult],
                     time_field: str) -> Optional[float]:
    candidates: List[str] = []
    if res is not None and res.source_algo:
        candidates.append(res.source_algo)

    norm_algo = normalize_algo_token(plotted_algo)
    if norm_algo == "hefthint":
        candidates.extend(["hefthint", "heft"])
    else:
        candidates.append(norm_algo)

    seen = set()
    for cand in candidates:
        cand = (cand or "").strip().lower()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        value = _time_value(time_index.get((prefill, decode, cand)), time_field)
        if value is not None:
            return value
    return None


def _available_runtimes_for_case(time_index: Dict[Tuple[int, int, str], TimeResult],
                                 results: Dict[Tuple[int, int, str], UtilResult],
                                 prefill: int,
                                 decode: int,
                                 algorithms: Sequence[str],
                                 time_field: str) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for algo in algorithms:
        res = results.get((prefill, decode, algo))
        runtime = lookup_runtime_s(
            time_index=time_index,
            prefill=prefill,
            decode=decode,
            plotted_algo=algo,
            res=res,
            time_field=time_field,
        )
        if runtime is not None:
            out.append((algo, runtime))
    return out


def resolve_speedup_reference(time_index: Dict[Tuple[int, int, str], TimeResult],
                              results: Dict[Tuple[int, int, str], UtilResult],
                              prefill: int,
                              decode: int,
                              algorithms: Sequence[str],
                              time_field: str,
                              speedup_ref: str) -> Tuple[Optional[str], Optional[float]]:
    ref_key = (speedup_ref or "slowest").strip().lower()
    available = _available_runtimes_for_case(
        time_index=time_index,
        results=results,
        prefill=prefill,
        decode=decode,
        algorithms=algorithms,
        time_field=time_field,
    )
    if not available:
        return None, None

    if ref_key in {"slowest", "worst", "max"}:
        algo, runtime = max(available, key=lambda kv: kv[1])
        return algo, runtime

    if ref_key in {"best", "fastest", "min"}:
        algo, runtime = min(available, key=lambda kv: kv[1])
        return algo, runtime

    ref_algo = normalize_algo_token(ref_key)
    runtime = lookup_runtime_s(
        time_index=time_index,
        prefill=prefill,
        decode=decode,
        plotted_algo=ref_algo,
        res=results.get((prefill, decode, ref_algo)),
        time_field=time_field,
    )
    if runtime is None:
        return ref_algo, None
    return ref_algo, runtime


def _family_diff_for_plot(res: Optional[UtilResult],
                          diff_order: str = "accel-pim") -> float:
    if res is None or res.pim_utilization is None or res.accel_utilization is None:
        return float("nan")

    pim = float(res.pim_utilization)
    accel = float(res.accel_utilization)
    if diff_order == "accel-pim":
        return accel - pim
    if diff_order == "pim-accel":
        return pim - accel
    raise ValueError(f"unknown diff_order: {diff_order}")


def _concurrent_util_for_plot(res: Optional[UtilResult]) -> float:
    if res is None or res.concurrent_utilization is None:
        return float("nan")
    return float(res.concurrent_utilization)


def _diff_expr_text(diff_order: str = "accel-pim") -> str:
    accel_name = FAMILY_LABELS["accel"]
    pim_name = FAMILY_LABELS["pim"]
    if diff_order == "accel-pim":
        return f"{accel_name} - {pim_name}"
    if diff_order == "pim-accel":
        return f"{pim_name} - {accel_name}"
    raise ValueError(f"unknown diff_order: {diff_order}")


def _diff_ylabel(util_scale: str, diff_order: str = "accel-pim") -> str:
    expr = _diff_expr_text(diff_order=diff_order)
    if util_scale == "percent":
        return f"Util. gap: {expr} (pp)"
    return f"Util. gap: {expr}"


def _third_metric_value_for_plot(res: Optional[UtilResult],
                                 third_metric: str = "cu",
                                 diff_order: str = "accel-pim") -> float:
    if third_metric == "cu":
        return _concurrent_util_for_plot(res)
    if third_metric == "gap":
        return _family_diff_for_plot(res, diff_order=diff_order)
    raise ValueError(f"unknown third_metric: {third_metric}")


def _third_metric_label(util_scale: str,
                        third_metric: str = "cu",
                        diff_order: str = "accel-pim") -> str:
    if third_metric == "cu":
        if util_scale == "percent":
            return "Co-util. (%)"
        return "Co-util. (%)"
    if third_metric == "gap":
        return _diff_ylabel(util_scale=util_scale, diff_order=diff_order)
    raise ValueError(f"unknown third_metric: {third_metric}")


def _third_metric_legend_label(third_metric: str = "cu",
                               diff_order: str = "accel-pim") -> str:
    if third_metric == "cu":
        return "co-utilization"
    if third_metric == "gap":
        return f"util gap: {_diff_expr_text(diff_order=diff_order)}"
    raise ValueError(f"unknown third_metric: {third_metric}")


def _speedup_for_plot(time_index: Dict[Tuple[int, int, str], TimeResult],
                      results: Dict[Tuple[int, int, str], UtilResult],
                      prefill: int,
                      decode: int,
                      algo: str,
                      algorithms: Sequence[str],
                      time_field: str,
                      speedup_ref: str) -> float:
    runtime = lookup_runtime_s(
        time_index=time_index,
        prefill=prefill,
        decode=decode,
        plotted_algo=algo,
        res=results.get((prefill, decode, algo)),
        time_field=time_field,
    )
    ref_algo, ref_runtime = resolve_speedup_reference(
        time_index=time_index,
        results=results,
        prefill=prefill,
        decode=decode,
        algorithms=algorithms,
        time_field=time_field,
        speedup_ref=speedup_ref,
    )
    if runtime is None or ref_runtime is None or runtime <= 0 or ref_runtime <= 0:
        return float("nan")
    return float(ref_runtime / runtime)
def _nice_ceil(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(x))
    frac = x / exp
    for m in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if frac <= m:
            return m * exp
    return 10.0 * exp


def _positive_ylim(values: Sequence[float],
                   pad_ratio: float = 0.15,
                   fallback_top: float = 1.0,
                   min_top: Optional[float] = None) -> Tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        top = fallback_top
    else:
        vmax = max(finite)
        if vmax <= 0:
            top = fallback_top
        else:
            top = _nice_ceil(vmax * (1.0 + pad_ratio))

    if min_top is not None:
        top = max(top, min_top)
    return (0.0, top)


def _symmetric_ylim(values: Sequence[float],
                    pad_ratio: float = 0.15,
                    fallback_half_span: float = 1.0,
                    min_half_span: Optional[float] = None) -> Tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        half = fallback_half_span
    else:
        vmax = max(abs(v) for v in finite)
        if vmax <= 0:
            half = fallback_half_span
        else:
            half = _nice_ceil(vmax * (1.0 + pad_ratio))

    if min_half_span is not None:
        half = max(half, min_half_span)
    return (-half, half)
    
def _save_summary_csv(results: Dict[Tuple[int, int, str], UtilResult],
                      algorithms: Sequence[str],
                      prefills: Sequence[int],
                      decodes: Sequence[int],
                      time_index: Dict[Tuple[int, int, str], TimeResult],
                      output_path: Path,
                      time_field: str,
                      speedup_ref: str,
                      third_metric: str = "cu",
                      diff_order: str = "accel-pim") -> None:
    rows: List[Dict[str, object]] = []
    for (prefill, decode, plotted_algo), res in sorted(results.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        runtime_s = lookup_runtime_s(
            time_index=time_index,
            prefill=prefill,
            decode=decode,
            plotted_algo=plotted_algo,
            res=res,
            time_field=time_field,
        )
        speedup_ref_algo, speedup_ref_time_s = resolve_speedup_reference(
            time_index=time_index,
            results=results,
            prefill=prefill,
            decode=decode,
            algorithms=algorithms,
            time_field=time_field,
            speedup_ref=speedup_ref,
        )

        speedup = float("nan")
        if runtime_s is not None and speedup_ref_time_s is not None and runtime_s > 0 and speedup_ref_time_s > 0:
            speedup = float(speedup_ref_time_s / runtime_s)

        accel_minus_pim = _family_diff_for_plot(res=res, diff_order="accel-pim")
        pim_minus_accel = _family_diff_for_plot(res=res, diff_order="pim-accel")
        concurrent_utilization = _concurrent_util_for_plot(res=res)
        configured_gap = _family_diff_for_plot(res=res, diff_order=diff_order)
        configured_third_metric_value = _third_metric_value_for_plot(
            res=res,
            third_metric=third_metric,
            diff_order=diff_order,
        )

        rows.append({
            "prefill": prefill,
            "decode": decode,
            "plotted_algo": plotted_algo,
            "source_algo": res.source_algo,
            "speedup": speedup,
            "speedup_ref": speedup_ref,
            "speedup_ref_algo": speedup_ref_algo,
            "speedup_ref_time_s": speedup_ref_time_s,
            "runtime_field": time_field,
            "runtime_s": runtime_s,
            "overall_utilization": res.overall_utilization,
            "pim_utilization": res.pim_utilization,
            "accel_utilization": res.accel_utilization,
            "concurrent_utilization": concurrent_utilization,
            "accel_minus_pim_utilization": accel_minus_pim,
            "pim_minus_accel_utilization": pim_minus_accel,
            "configured_util_gap": configured_gap,
            "configured_util_gap_expr": _diff_expr_text(diff_order=diff_order),
            "configured_third_metric": third_metric,
            "configured_third_metric_value": configured_third_metric_value,
            "configured_third_metric_label": _third_metric_label(
                util_scale="fraction",
                third_metric=third_metric,
                diff_order=diff_order,
            ),
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



def _drop_zero_ytick(ax, atol: float = 1e-12) -> None:
    ticks = ax.get_yticks()
    kept = [t for t in ticks if not np.isclose(t, 0.0, atol=atol)]
    ax.set_yticks(kept)


def _set_left_ylabel(ax, text: str, x: float = -0.14) -> None:
    ax.set_ylabel(text)
    ax.yaxis.set_label_coords(x, 0.5)


def _boxed_legend_slot_width(labels: Sequence[str],
                             min_w: float = 0.16,
                             max_w: float = 0.32,
                             base_w: float = 0.08,
                             char_w: float = 0.018) -> float:
    if not labels:
        return min_w
    max_len = max(len(str(lbl)) for lbl in labels)
    return float(max(min_w, min(max_w, base_w + char_w * max_len)))


def _style_boxed_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    frame.set_edgecolor("0.55")
    frame.set_linewidth(0.8)
    try:
        frame.set_linestyle((0, (2.0, 2.0)))
    except Exception:
        pass


def _add_boxed_item_legends(ax,
                            handles: Sequence[Line2D],
                            labels: Sequence[str],
                            *,
                            y: float = 0.98,
                            right: float = 0.995,
                            left: Optional[float] = None,
                            gap: float = 0.015,
                            slot_width: Optional[float] = None,
                            fontsize: Optional[float] = None,
                            handlelength: float = 0.85,
                            handletextpad: float = 0.35,
                            borderpad: float = 0.28) -> None:
    items = [(h, str(lbl)) for h, lbl in zip(handles, labels)]
    if not items:
        return

    if slot_width is None:
        slot_width = _boxed_legend_slot_width([lbl for _, lbl in items])

    total_w = len(items) * slot_width + max(0, len(items) - 1) * gap
    if left is not None:
        start_x = max(0.01, float(left))
    else:
        start_x = max(0.01, right - total_w)

    for i, (handle, label) in enumerate(items):
        x = start_x + i * (slot_width + gap)
        leg = ax.legend(
            handles=[handle],
            labels=[label],
            loc="upper left",
            bbox_to_anchor=(x, y),
            bbox_transform=ax.transAxes,
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            fontsize=fontsize,
            handlelength=handlelength,
            handletextpad=handletextpad,
            borderpad=borderpad,
            borderaxespad=0.0,
        )
        _style_boxed_legend(leg)
        try:
            leg._legend_box.align = "left"
        except Exception:
            pass
        ax.add_artist(leg)


def plot_results(results: Dict[Tuple[int, int, str], UtilResult],
                 time_index: Dict[Tuple[int, int, str], TimeResult],
                 prefills: Sequence[int],
                 decodes: Sequence[int],
                 algorithms: Sequence[str],
                 output: Path,
                 colors: Sequence[str],
                 title: Optional[str] = None,
                 util_scale: str = "percent",
                 speedup_ref: str = "slowest",
                 time_field: str = "total_time_s",
                 third_metric: str = "cu",
                 diff_order: str = "accel-pim",
                 dpi: int = 200,
                 marker_size: float = 5.5,
                 line_width: float = 1.8,
                 max_cols: int = 2,
                 display_name_map: Optional[Dict[str, str]] = None,
                 highlight_algo: Optional[str] = "hefthint",
                 highlight_color: str = "blue") -> None:
    if not prefills:
        raise ValueError("no prefill lengths to plot")
    if not decodes:
        raise ValueError("no decode lengths to plot")
    if not algorithms:
        raise ValueError("no algorithms to plot")
    if not colors:
        raise ValueError("colors list is empty")

    n_panels = len(prefills)
    ncols = max(1, min(max_cols, n_panels))
    n_group_rows = int(math.ceil(n_panels / ncols))

    # -----------------------------
    # 这几个参数就是你后面最常调的地方
    # -----------------------------
    panel_width = max(5, 0.65 * len(algorithms))   # 单个 prefill panel 的宽度
    figure_max_width = 16.0                          # 整张图最大宽度
    panel_height = 3.2                               # 单个 prefill panel 的高度

    outer_wspace = 0.1                              # 左右两个 prefill panel 的横向间距
    outer_hspace = 0.30                              # 多行 prefill panel 之间的纵向间距
    inner_hspace = 0.00                              # 一个 panel 内三幅子图之间的空隙，设 0 表示无缝

    fig_w = min(figure_max_width, panel_width * ncols)
    fig_h = max(panel_height * n_group_rows, 3.2)

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        n_group_rows,
        ncols,
        hspace=outer_hspace,
        wspace=outer_wspace,
    )
    x = np.arange(len(algorithms), dtype=float)

    if util_scale == "percent":
        y_factor = 100.0
        abs_y_label = "Avg. util. (%)"
        diff_min_span = 5.0
        abs_fallback_top = 100.0
        abs_min_top = 10.0
        cu_fallback_top = 10.0
        cu_min_top = 10.0
    else:
        y_factor = 1.0
        abs_y_label = "Avg. util."
        diff_min_span = 0.05
        abs_fallback_top = 1.0
        abs_min_top = 0.1
        cu_fallback_top = 0.1
        cu_min_top = 0.1

    speed_axes: List[plt.Axes] = []
    abs_axes: List[plt.Axes] = []
    third_axes: List[plt.Axes] = []

    for idx, p in enumerate(prefills):
        group = idx // ncols
        col = idx % ncols

        inner = outer[group, col].subgridspec(
            3, 1,
            hspace=inner_hspace,
            height_ratios=[0.6, 1.0, 0.6],
        )
        speed_ax = fig.add_subplot(inner[0, 0])
        abs_ax = fig.add_subplot(inner[1, 0], sharex=speed_ax)
        third_ax = fig.add_subplot(inner[2, 0], sharex=speed_ax)

        speed_axes.append(speed_ax)
        abs_axes.append(abs_ax)
        third_axes.append(third_ax)

        any_speed_line = False
        any_abs_line = False
        any_third_line = False

        # 每个 panel 自己统计一遍 y 值，用于单独收紧 y 轴范围
        panel_speed_values: List[float] = []
        panel_abs_values: List[float] = []
        panel_third_values: List[float] = []

        for line_idx, d in enumerate(decodes):
            color = colors[line_idx % len(colors)]

            speed_vals = np.asarray([
                _speedup_for_plot(
                    time_index=time_index,
                    results=results,
                    prefill=p,
                    decode=d,
                    algo=algo,
                    algorithms=algorithms,
                    time_field=time_field,
                    speedup_ref=speedup_ref,
                )
                for algo in algorithms
            ], dtype=float)

            finite_speed = speed_vals[np.isfinite(speed_vals)]
            if finite_speed.size > 0:
                any_speed_line = True
                panel_speed_values.extend(finite_speed.tolist())
                speed_ax.plot(
                    x,
                    speed_vals,
                    marker=SPEEDUP_MARKER,
                    markersize=marker_size,
                    linewidth=line_width,
                    color=color,
                    label=None,
                )

            for family in FAMILY_ORDER:
                abs_vals = np.asarray([
                    _family_util_for_plot(results.get((p, d, algo)), family=family) * y_factor
                    for algo in algorithms
                ], dtype=float)

                finite_abs = abs_vals[np.isfinite(abs_vals)]
                if finite_abs.size > 0:
                    any_abs_line = True
                    panel_abs_values.extend(finite_abs.tolist())
                    abs_ax.plot(
                        x + FAMILY_X_OFFSETS[family],
                        abs_vals,
                        marker=FAMILY_MARKERS[family],
                        markersize=marker_size,
                        linewidth=line_width,
                        color=color,
                        label=None,
                    )

            third_vals = np.asarray([
                _third_metric_value_for_plot(
                    results.get((p, d, algo)),
                    third_metric=third_metric,
                    diff_order=diff_order,
                ) * y_factor
                for algo in algorithms
            ], dtype=float)

            finite_third = third_vals[np.isfinite(third_vals)]
            if finite_third.size > 0:
                any_third_line = True
                panel_third_values.extend(finite_third.tolist())
                third_ax.plot(
                    x,
                    third_vals,
                    marker=DIFF_MARKER,
                    markersize=marker_size,
                    linewidth=line_width,
                    linestyle=DIFF_LINESTYLE,
                    color=color,
                    label=None,
                )

        # -----------------------------
        # 每个 panel 单独决定 y 轴范围，去掉没用上的大面积空白
        # -----------------------------
        panel_speed_y_lim = _positive_ylim(
            panel_speed_values,
            pad_ratio=0.08,
            fallback_top=1.0,
            min_top=1.0,
        )

        if util_scale == "percent":
            panel_abs_y_lim = (0.0, 100.0)
        else:
            panel_abs_y_lim = (0.0, 1.0)

        if third_metric == "cu":
            panel_third_y_lim = _positive_ylim(
                panel_third_values,
                pad_ratio=0.08,
                fallback_top=cu_fallback_top,
                min_top=cu_min_top,
            )
        else:
            panel_third_y_lim = _symmetric_ylim(
                panel_third_values,
                pad_ratio=0.08,
                fallback_half_span=diff_min_span,
                min_half_span=diff_min_span,
            )

        # -----------------------------
        # speedup subplot
        # -----------------------------
        speed_ax.set_title(f"prefill={p}")
        speed_ax.set_ylim(*panel_speed_y_lim)
        speed_ax.set_xlim(-0.5, len(algorithms) - 0.5)
        speed_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        speed_ax.set_xticks(x)
        speed_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # 三幅图无缝时，去掉上两幅图的 0 刻度，避免边界重叠
        _drop_zero_ytick(speed_ax)

        # 只保留下一幅图的 top spine 作为分隔线，避免双线重叠
        speed_ax.spines["bottom"].set_visible(False)

        if col == 0:
            _set_left_ylabel(speed_ax, "Speedup (x)")
        if not any_speed_line:
            speed_ax.text(
                0.5, 0.5, "No data",
                transform=speed_ax.transAxes,
                ha="center", va="center", fontsize=12,
            )

        # -----------------------------
        # absolute utilization subplot
        # -----------------------------
        abs_ax.set_ylim(*panel_abs_y_lim)
        abs_ax.set_xlim(-0.5, len(algorithms) - 0.5)
        abs_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
        abs_ax.set_xticks(x)
        abs_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        _drop_zero_ytick(abs_ax)
        abs_ax.spines["bottom"].set_visible(False)

        if col == 0:
            _set_left_ylabel(abs_ax, abs_y_label)
        if not any_abs_line:
            abs_ax.text(
                0.5, 0.5, "No data",
                transform=abs_ax.transAxes,
                ha="center", va="center", fontsize=12,
            )

        # -----------------------------
        # third subplot
        # -----------------------------
        third_ax.set_ylim(*panel_third_y_lim)
        third_ax.set_xlim(-0.5, len(algorithms) - 0.5)
        third_ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

        if third_metric == "gap":
            third_ax.axhline(0.0, linestyle=":", linewidth=0.9, color="black", alpha=0.55)

        third_ax.set_xticks(x)
        third_ax.set_xticklabels(
            [pretty_algo_name(a, display_name_map=display_name_map) for a in algorithms],
            rotation=30,
            ha="right",
        )
        highlight_norm = None if not highlight_algo else normalize_algo_token(highlight_algo)
        if highlight_norm is not None:
            for algo_name, tick_label in zip(algorithms, third_ax.get_xticklabels()):
                if normalize_algo_token(algo_name) == highlight_norm:
                    tick_label.set_color(highlight_color)
                    tick_label.set_fontweight("bold")

        if col == 0:
            _set_left_ylabel(
                third_ax,
                _third_metric_label(
                    util_scale=util_scale,
                    third_metric=third_metric,
                    diff_order=diff_order,
                ),
            )
        if not any_third_line:
            third_ax.text(
                0.5, 0.5, "No data",
                transform=third_ax.transAxes,
                ha="center", va="center", fontsize=12,
            )

    if title:
        fig.suptitle(title, y=0.99)

    family_legend_handles: List[Line2D] = [
        Line2D(
            [0], [0],
            color="black",
            lw=0,
            marker=FAMILY_MARKERS[family],
            markersize=marker_size + 0.5,
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]

    decode_legend_handles: List[Line2D] = [
        Line2D(
            [0], [0],
            color=colors[i % len(colors)],
            lw=line_width,
            marker=None,
            label=f"decode={d}",
        )
        for i, d in enumerate(decodes)
    ]

    # -------------------------------------------------
    # Legend: 右上角，拆成一个个小框，效果接近你给的示意图
    # 对于一行两图，会放到第一行最右边那一列；只有一图时就放到唯一那张图上。
    # -------------------------------------------------
    legend_panel_idx = min(max(0, ncols - 1), max(0, len(speed_axes) - 1))

    decode_labels = [f"decode={d}" for d in decodes]
    family_labels = [FAMILY_LABELS[family] for family in FAMILY_ORDER]

    if speed_axes:
        n_decode_panels = min(len(speed_axes), max(1, len(decode_legend_handles)))
        base = len(decode_legend_handles) // n_decode_panels
        rem = len(decode_legend_handles) % n_decode_panels
        chunk_sizes = [base] * n_decode_panels
        for i in range(rem):
            chunk_sizes[n_decode_panels - rem + i] += 1

        start = 0
        for panel_idx, chunk_size in enumerate(chunk_sizes):
            if chunk_size <= 0:
                continue
            end = start + chunk_size
            panel_handles = decode_legend_handles[start:end]
            panel_labels = decode_labels[start:end]
            decode_target_ax = speed_axes[panel_idx]
            decode_slot_width = _boxed_legend_slot_width(panel_labels, min_w=0.20, max_w=0.31)
            if (panel_idx % ncols) == 0:
                _add_boxed_item_legends(
                    decode_target_ax,
                    panel_handles,
                    panel_labels,
                    y=0.965,
                    right=0.995,
                    gap=0.014,
                    slot_width=decode_slot_width,
                    handlelength=0.70,
                    handletextpad=0.30,
                    borderpad=0.22,
                )
            else:
                _add_boxed_item_legends(
                    decode_target_ax,
                    panel_handles,
                    panel_labels,
                    y=0.965,
                    left=0.015,
                    gap=0.014,
                    slot_width=decode_slot_width,
                    handlelength=0.70,
                    handletextpad=0.30,
                    borderpad=0.22,
                )
            start = end

    if abs_axes:
        family_target_ax = abs_axes[legend_panel_idx]
        _add_boxed_item_legends(
            family_target_ax,
            family_legend_handles,
            family_labels,
            y=0.965,
            right=0.995,
            gap=0.016,
            slot_width=_boxed_legend_slot_width(family_labels, min_w=0.16, max_w=0.24),
            handlelength=0.70,
            handletextpad=0.40,
            borderpad=0.22,
        )

    top_margin = 0.90 if title else 0.94
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.08, top=top_margin)

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
                    help="Root directory that contains algo_* traces and baseline_compare_*.json files.")
    ap.add_argument("--prefills", type=str, default=None,
                    help="Prefill lengths: 128,256,512 or 128:512:128 or all. Omit to auto-scan all.")
    ap.add_argument("--decodes", type=str, default=None,
                    help="Decode lengths: 64,128,256 or 64:512:64 or all. Omit to auto-scan all.")
    ap.add_argument("--algo-order", type=str, default=None,
                    help="Optional x-axis algorithm order, e.g. pd,ianus,facil,hefthint,attacc")
    ap.add_argument("--include-algos", type=str, default=None,
                    help="Only plot these algorithms, in the given order, e.g. pd,ianus,facil,hefthint")
    ap.add_argument("--exclude-algos", type=str, default=None,
                    help="Algorithms to drop from plotting, e.g. attacc,weights_on_pim")
    ap.add_argument("--algo-label-map", type=str, default=None,
                    help="Override x-axis display names with key=value pairs separated by ';', e.g. 'hefthint=Bifocal（this work）;pd=PD'")
    ap.add_argument("--highlight-algo", type=str, default="hefthint",
                    help="Highlight this algorithm's x-axis label. Default: hefthint")
    ap.add_argument("--highlight-color", type=str, default="blue",
                    help="Color for the highlighted algorithm x-axis label.")
    ap.add_argument("--output", type=str, default="device_utilization.png")
    ap.add_argument("--summary-csv", type=str, default=None,
                    help="Optional CSV dump of speedup / utilization / co-utilization values.")
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
    ap.add_argument("--speedup-ref", type=str, default="slowest",
                    help=(
                        "Reference for speedup. Use an algorithm name (e.g. ianus, pd, hefthint), "
                        "or one of: slowest / best."
                    ))
    ap.add_argument("--speedup-time-field", type=str,
                    choices=["auto", "total_time_s", "prefill_time_s", "decode_time_s"],
                    default="auto",
                    help="Which field from baseline_compare_*.json to use for speedup. auto follows --phase.")
    ap.add_argument("--third-metric", type=str, choices=["cu", "gap"], default="cu",
                    help="Metric for the third subplot: cu = co-utilization, gap = NPU/GPU-PIM utilization gap.")
    ap.add_argument("--diff-order", type=str, choices=["accel-pim", "pim-accel"], default="accel-pim",
                    help="Direction of the third-subplot gap when --third-metric=gap. Default is NPU/GPU minus PIM.")
    ap.add_argument("--max-cols", type=int, default=2,
                    help="Maximum number of subplot columns for prefills.")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--marker-size", type=float, default=5.5)
    ap.add_argument("--line-width", type=float, default=1.8)
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="Directory for on-disk utilization cache. Default: <search-dir>/.plot_cache")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable on-disk cache.")

    args = ap.parse_args()

    search_dir = Path(args.search_dir)
    if not search_dir.exists():
        ap.error(f"--search-dir not found: {search_dir}")

    trace_paths = discover_trace_files(search_dir)
    if not trace_paths:
        ap.error(f"No *_trace.csv found under: {search_dir}")

    index = build_trace_index(trace_paths)
    if not index:
        ap.error("No valid comms/ops trace pairs found. Check file names and directory.")

    baseline_paths = discover_baseline_compare_files(search_dir)
    time_index = build_time_index(baseline_paths)
    if not baseline_paths:
        print(f"[WARN] no baseline_compare_*.json found under: {search_dir}", file=sys.stderr)
    elif not time_index:
        print(f"[WARN] baseline_compare_*.json found but no valid timings were parsed under: {search_dir}", file=sys.stderr)

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
    include_algos = [normalize_algo_token(a) for a in parse_str_list(args.include_algos)]
    exclude_algos = [normalize_algo_token(a) for a in parse_str_list(args.exclude_algos)]
    display_name_map = dict(DISPLAY_NAME_MAP)
    try:
        display_name_map.update(parse_name_map(args.algo_label_map))
    except ValueError as exc:
        ap.error(str(exc))

    if include_algos:
        missing_include = [a for a in include_algos if a not in all_algos_norm]
        if missing_include:
            print(f"[WARN] requested include-algos not found under search-dir: {missing_include}", file=sys.stderr)
        algorithms = []
        for algo in include_algos:
            if algo in all_algos_norm and algo not in algorithms:
                algorithms.append(algo)
    else:
        algorithms = order_algorithms(all_algos_norm, algo_order=algo_order)

    if exclude_algos:
        missing_exclude = [a for a in exclude_algos if a not in all_algos_norm]
        if missing_exclude:
            print(f"[WARN] requested exclude-algos not found under search-dir: {missing_exclude}", file=sys.stderr)
        exclude_set = set(exclude_algos)
        algorithms = [a for a in algorithms if a not in exclude_set]

    if not algorithms:
        ap.error("No algorithms left to plot after include/exclude filtering.")

    missing_prefills = [p for p in prefills if p not in all_prefills]
    missing_decodes = [d for d in decodes if d not in all_decodes]
    if missing_prefills:
        print(f"[WARN] requested prefills not found under search-dir: {missing_prefills}", file=sys.stderr)
    if missing_decodes:
        print(f"[WARN] requested decodes not found under search-dir: {missing_decodes}", file=sys.stderr)

    if args.no_cache:
        disk_cache_dir = None
    else:
        disk_cache_dir = Path(args.cache_dir) if args.cache_dir else (search_dir / ".plot_cache")
        print(f"[INFO] using cache dir: {disk_cache_dir}", file=sys.stderr)

    results = build_results(
        index=index,
        time_index=time_index,
        prefills=prefills,
        decodes=decodes,
        phase=args.phase,
        include_cpu=args.include_cpu,
        heft_pick_by=args.heft_pick_by,
        disk_cache_dir=disk_cache_dir,
    )

    if not results:
        ap.error("No usable results for the specified prefills/decodes.")

    active_algos = []
    for algo in algorithms:
        if any(k[2] == algo for k in results.keys()):
            active_algos.append(algo)

    time_field = args.speedup_time_field
    if time_field == "auto":
        time_field = _default_time_field_for_phase(args.phase)

    if args.summary_csv:
        _save_summary_csv(
            results=results,
            algorithms=active_algos,
            prefills=prefills,
            decodes=decodes,
            time_index=time_index,
            output_path=Path(args.summary_csv),
            time_field=time_field,
            speedup_ref=args.speedup_ref,
            third_metric=args.third_metric,
            diff_order=args.diff_order,
        )

    plot_results(
        results=results,
        time_index=time_index,
        prefills=prefills,
        decodes=decodes,
        algorithms=active_algos,
        output=Path(args.output),
        colors=colors,
        title=args.title,
        util_scale=args.util_scale,
        speedup_ref=args.speedup_ref,
        time_field=time_field,
        third_metric=args.third_metric,
        diff_order=args.diff_order,
        dpi=args.dpi,
        marker_size=args.marker_size,
        line_width=args.line_width,
        max_cols=max(1, args.max_cols),
        display_name_map=display_name_map,
        highlight_algo=args.highlight_algo,
        highlight_color=args.highlight_color,
    )


if __name__ == "__main__":
    main()
