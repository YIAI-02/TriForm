#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example for qwen 7b, batch 16:

python plot_exp1_utilization_simple.py \
  --search-dir ../../output/deepseek_v4_flash/deepseek_v4_flash_fp8_b4_s16 \
  --model-prefix deepseek \
  --batch-size 4 \
  --prefill 4096 \
  --decode 1024 \
  --exclude-algos weights_on_pim \
  --speedup-ref pd \
  --phase all \
  --weight-stage-util-mode l1_l2

Input file naming expected by this script:
  <algo>_linear_prefill-<prefill>xdecode_<decode>_ops_trace.csv
  <algo>_linear_prefill-<prefill>xdecode_<decode>_comms_trace.csv
  baseline_compare_<prefill>x<decode>.json

Run directory naming expected by this script:
  <model_name>_b<batch_size>_s<shards>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


TRACE_RE = re.compile(
    r"^(?P<algo>.+?)_linear_prefill-(?P<prefill>\d+)xdecode_(?P<decode>\d+)_(?P<kind>comms|ops)_trace\.csv$",
    re.IGNORECASE,
)
BASELINE_COMPARE_RE = re.compile(
    r"^baseline_compare_(?P<prefill>\d+)x(?P<decode>\d+)\.json$",
    re.IGNORECASE,
)
RUN_DIR_RE = re.compile(
    r"^(?P<model>.+?)_b(?P<batch>\d+)_s(?P<shards>\d+)$",
    re.IGNORECASE,
)

DEFAULT_PHASE = "all"
DEFAULT_SPEEDUP_REF = "pd"
DEFAULT_SPEEDUP_MODE = "ratio"
DEFAULT_WEIGHT_STAGE_UTIL_MODE = "none"

PREFERRED_ALGO_ORDER = ["pd", "attn_on_pim", "ianus", "facil", "attacc", "hefthint"]


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    run_dir: Path
    model_name: str
    batch_size: Optional[int]
    shards: Optional[int]


@dataclass(frozen=True)
class TraceKey:
    run_id: str
    algo: str
    prefill: int
    decode: int


@dataclass(frozen=True)
class TracePair:
    comms_path: Path
    ops_path: Path


@dataclass(frozen=True)
class TimeResult:
    run_id: str
    source_algo: str
    plotted_algo: str
    prefill: int
    decode: int
    prefill_time_s: Optional[float]
    decode_time_s: Optional[float]
    total_time_s: Optional[float]
    json_path: Path


@dataclass
class UtilResult:
    run_id: str
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


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def warn(message: str) -> None:
    eprint(f"[WARN] {message}")


def die(message: str, code: int = 2) -> None:
    eprint(f"[ERROR] {message}")
    raise SystemExit(code)


def compact_name(s: object) -> str:
    return re.sub(r"[\s_\-]+", "", str(s or "").strip().lower())


def normalize_algo_token(algo: str) -> str:
    raw = (algo or "").strip().lower()
    compact = compact_name(raw)
    if compact in {"heft", "hefthint", "thiswork", "bifocal"}:
        return "hefthint"
    return raw


def parse_str_list(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_case_token(case: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not case:
        return None, None
    m = re.match(r"^\s*(\d+)\s*[xX,/]\s*(\d+)\s*$", case)
    if not m:
        raise ValueError("--case must look like 128x128 or 128,128")
    return int(m.group(1)), int(m.group(2))


def safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def safe_lower(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    return "" if s in {"", "nan", "none"} else s


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def format_float(v: Optional[float], digits: int = 6) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{float(v):.{digits}f}"


def format_percent(v: Optional[float], digits: int = 2) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{100.0 * float(v):.{digits}f}"


def policy_to_algo(policy: object) -> str:
    s = str(policy or "").strip().lower()
    if not s:
        return ""
    if s.startswith("algo:"):
        s = s.split(":", 1)[1]
    return s.strip()


# -----------------------------------------------------------------------------
# Discovery / parsing
# -----------------------------------------------------------------------------

def parse_trace_filename(path: Path) -> Optional[Tuple[str, int, int, str]]:
    m = TRACE_RE.match(path.name)
    if not m:
        return None
    return (
        m.group("algo").lower(),
        int(m.group("prefill")),
        int(m.group("decode")),
        m.group("kind").lower(),
    )


def parse_baseline_compare_filename(path: Path) -> Optional[Tuple[int, int]]:
    m = BASELINE_COMPARE_RE.match(path.name)
    if not m:
        return None
    return int(m.group("prefill")), int(m.group("decode"))


def parse_case_from_baseline_payload(payload: Dict[str, object], path: Path) -> Optional[Tuple[int, int]]:
    cfg = payload.get("config", {})
    if isinstance(cfg, dict):
        try:
            prefill = int(cfg.get("prefill_len")) if cfg.get("prefill_len") is not None else None
            decode = int(cfg.get("decode_len")) if cfg.get("decode_len") is not None else None
        except (TypeError, ValueError):
            prefill, decode = None, None
        if prefill is not None and decode is not None:
            return prefill, decode
    return parse_baseline_compare_filename(path)


def infer_run_info(path: Path) -> RunInfo:
    # Search upward for a directory like qwen_1.8b_fp16_b16_s8.
    for parent in [path.parent, *path.parents]:
        m = RUN_DIR_RE.match(parent.name)
        if m:
            return RunInfo(
                run_id=str(parent.resolve()),
                run_dir=parent,
                model_name=m.group("model").lower(),
                batch_size=int(m.group("batch")),
                shards=int(m.group("shards")),
            )

    # Fallback for unusual layouts.
    return RunInfo(
        run_id=str(path.parent.resolve()),
        run_dir=path.parent,
        model_name=path.parent.name.lower(),
        batch_size=None,
        shards=None,
    )


def discover_trace_files(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("*_trace.csv"))


def discover_baseline_compare_files(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("baseline_compare_*.json"))


def filter_run_infos(
    run_infos: Dict[str, RunInfo],
    *,
    model_prefix: Optional[str],
    batch_size: Optional[int],
    shards: Optional[int],
) -> Dict[str, RunInfo]:
    out: Dict[str, RunInfo] = {}
    raw_prefix = (model_prefix or "").strip().lower()
    compact_prefix = compact_name(raw_prefix)

    for run_id, info in run_infos.items():
        if raw_prefix:
            raw_model = info.model_name.lower()
            raw_dir = info.run_dir.name.lower()
            compact_model = compact_name(info.model_name)
            compact_dir = compact_name(info.run_dir.name)
            if not (
                raw_model.startswith(raw_prefix)
                or raw_dir.startswith(raw_prefix)
                or compact_model.startswith(compact_prefix)
                or compact_dir.startswith(compact_prefix)
            ):
                continue
        if batch_size is not None and info.batch_size != batch_size:
            continue
        if shards is not None and info.shards != shards:
            continue
        out[run_id] = info
    return out


def pair_from_candidates(comms_paths: List[Path], ops_paths: List[Path]) -> Optional[TracePair]:
    if not comms_paths or not ops_paths:
        return None
    comms_paths = sorted(comms_paths)
    ops_paths = sorted(ops_paths)
    for c in comms_paths:
        for o in ops_paths:
            if c.parent == o.parent:
                return TracePair(comms_path=c, ops_path=o)
    warn(f"multiple trace candidates but no same-parent pair; using first: {comms_paths[0]} / {ops_paths[0]}")
    return TracePair(comms_path=comms_paths[0], ops_path=ops_paths[0])


def build_trace_index(paths: Sequence[Path], allowed_run_ids: Optional[Sequence[str]] = None) -> Dict[TraceKey, TracePair]:
    allowed = None if allowed_run_ids is None else set(allowed_run_ids)
    raw: Dict[TraceKey, Dict[str, List[Path]]] = {}

    for path in paths:
        parsed = parse_trace_filename(path)
        if parsed is None:
            continue
        algo, prefill, decode, kind = parsed
        run_info = infer_run_info(path)
        if allowed is not None and run_info.run_id not in allowed:
            continue
        key = TraceKey(run_id=run_info.run_id, algo=algo, prefill=prefill, decode=decode)
        raw.setdefault(key, {}).setdefault(kind, []).append(path)

    out: Dict[TraceKey, TracePair] = {}
    for key, bucket in raw.items():
        pair = pair_from_candidates(bucket.get("comms", []), bucket.get("ops", []))
        if pair is not None:
            out[key] = pair
    return out


def build_time_index(
    paths: Sequence[Path],
    allowed_run_ids: Optional[Sequence[str]] = None,
) -> Dict[Tuple[str, int, int, str], TimeResult]:
    allowed = None if allowed_run_ids is None else set(allowed_run_ids)
    out: Dict[Tuple[str, int, int, str], TimeResult] = {}

    for path in sorted(paths):
        run_info = infer_run_info(path)
        if allowed is not None and run_info.run_id not in allowed:
            continue

        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            warn(f"skip invalid baseline json {path}: {exc}")
            continue

        parsed = parse_case_from_baseline_payload(payload, path)
        if parsed is None:
            continue
        prefill, decode = parsed

        rows = payload.get("results", [])
        if not isinstance(rows, list):
            warn(f"skip baseline json without list 'results': {path}")
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            source_algo = policy_to_algo(row.get("policy"))
            if not source_algo:
                continue

            key = (run_info.run_id, prefill, decode, source_algo)
            time_res = TimeResult(
                run_id=run_info.run_id,
                source_algo=source_algo,
                plotted_algo=normalize_algo_token(source_algo),
                prefill=prefill,
                decode=decode,
                prefill_time_s=safe_float(row.get("prefill_time_s")),
                decode_time_s=safe_float(row.get("decode_time_s")),
                total_time_s=safe_float(row.get("total_time_s")),
                json_path=path,
            )
            if key in out:
                warn(
                    f"duplicate timing for run={run_info.run_dir.name} p={prefill} d={decode} "
                    f"algo={source_algo}; keep first from {out[key].json_path}"
                )
                continue
            out[key] = time_res
    return out


# -----------------------------------------------------------------------------
# Device-family helpers
# -----------------------------------------------------------------------------

def family_from_type_text(type_text: object, include_cpu: bool = False) -> Optional[str]:
    t = safe_lower(type_text)
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


def family_from_device_name(device: object, include_cpu: bool = False) -> Optional[str]:
    d = safe_lower(device)
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


def resolve_device_family(device: str, type_hints: Sequence[str], include_cpu: bool = False) -> Optional[str]:
    votes: Counter[str] = Counter()
    for hint in type_hints:
        fam = family_from_type_text(hint, include_cpu=include_cpu)
        if fam is not None:
            votes[fam] += 1
    name_fam = family_from_device_name(device, include_cpu=include_cpu)
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


def collect_device_families(
    ops_df: pd.DataFrame,
    comms_df: pd.DataFrame,
    include_cpu: bool = False,
) -> Dict[str, str]:
    type_hints: DefaultDict[str, List[str]] = defaultdict(list)

    if not ops_df.empty and "device" in ops_df.columns:
        if "device_type" in ops_df.columns:
            for dev, dtype in zip(ops_df["device"], ops_df["device_type"]):
                dev_s = str(dev).strip()
                if safe_lower(dev_s):
                    type_hints[dev_s].append(str(dtype))
        else:
            for dev in ops_df["device"].dropna().astype(str):
                if safe_lower(dev):
                    type_hints[dev].append("")

    if not comms_df.empty:
        if "src" in comms_df.columns:
            if "src_type" in comms_df.columns:
                for dev, dtype in zip(comms_df["src"], comms_df["src_type"]):
                    dev_s = str(dev).strip()
                    if safe_lower(dev_s):
                        type_hints[dev_s].append(str(dtype))
            else:
                for dev in comms_df["src"].dropna().astype(str):
                    if safe_lower(dev):
                        type_hints[dev].append("")
        if "dst" in comms_df.columns:
            if "dst_type" in comms_df.columns:
                for dev, dtype in zip(comms_df["dst"], comms_df["dst_type"]):
                    dev_s = str(dev).strip()
                    if safe_lower(dev_s):
                        type_hints[dev_s].append(str(dtype))
            else:
                for dev in comms_df["dst"].dropna().astype(str):
                    if safe_lower(dev):
                        type_hints[dev].append("")

    device_families: Dict[str, str] = {}
    for dev, hints in type_hints.items():
        fam = resolve_device_family(dev, hints, include_cpu=include_cpu)
        if fam is not None:
            device_families[dev] = fam
    return device_families


# -----------------------------------------------------------------------------
# Utilization and optional weight-stage accounting
# -----------------------------------------------------------------------------

def read_trace_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for col in [
        "start",
        "end",
        "duration",
        "bytes",
        "bytes_full_nd",
        "bytes_nd",
        "cache_capacity_bytes",
        "cached_before_nd",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if phase == "all":
        return df.copy()
    if "phase" not in df.columns:
        return df.copy()
    return df[df["phase"].astype(str).str.lower() == phase.lower()].copy()


def safe_text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    s = df[col].fillna("").astype(str).str.strip()
    return s.mask(s.str.lower().isin({"nan", "none"}), "")


def weight_stage_sidecar_paths(ops_path: Path) -> Dict[str, Path]:
    stem = ops_path.stem
    return {
        "overall": ops_path.with_name(f"{stem}_weight_stage_overall.csv"),
        "by_phase": ops_path.with_name(f"{stem}_weight_stage_by_phase.csv"),
        "by_device_type": ops_path.with_name(f"{stem}_weight_stage_by_device_type.csv"),
        "summary": ops_path.with_name(f"{stem}_weight_stage_summary.json"),
    }


def weight_load_mask(comms_df: pd.DataFrame) -> pd.Series:
    if comms_df.empty:
        return pd.Series(False, index=comms_df.index)
    mask = pd.Series(False, index=comms_df.index)
    if "tag" in comms_df.columns:
        mask |= safe_text_series(comms_df, "tag").str.lower().eq("weight_load")
    if "action" in comms_df.columns and "payload" in comms_df.columns:
        action = safe_text_series(comms_df, "action").str.lower()
        payload = safe_text_series(comms_df, "payload").str.lower()
        mask |= action.eq("load") & payload.eq("weight")
    return mask


def weight_load_needs_l2_mask(comms_df: pd.DataFrame) -> pd.Series:
    if comms_df.empty:
        return pd.Series(False, index=comms_df.index)
    mask = weight_load_mask(comms_df)
    if not mask.any():
        return mask

    if "to_fmt" not in comms_df.columns:
        dst_type = safe_text_series(comms_df, "dst_type").str.lower()
        dst = safe_text_series(comms_df, "dst").str.lower()
        return mask & (dst_type.str.contains("pim", regex=False) | dst.str.contains("pim", regex=False))

    invalid = {"", "NAN", "NONE"}
    to_fmt = safe_text_series(comms_df, "to_fmt").str.upper()
    from_fmt = safe_text_series(comms_df, "from_fmt").str.upper()
    needs_l2 = (~to_fmt.isin(list(invalid | {"ND"}))) & (
        from_fmt.isin(list(invalid)) | to_fmt.ne(from_fmt)
    )
    return mask & needs_l2


def weight_stage_overlap_saved_s(ops_path: Path) -> Optional[float]:
    overall_path = weight_stage_sidecar_paths(ops_path)["overall"]
    if not overall_path.exists():
        return None
    try:
        overall_df = read_trace_csv(overall_path)
    except Exception as exc:
        warn(f"failed to read weight-stage overall CSV {overall_path}: {exc}")
        return None
    if overall_df.empty or "load_l1_l2_saved_s_sum" not in overall_df.columns:
        return None
    return safe_float(overall_df.iloc[0].get("load_l1_l2_saved_s_sum"))


def weight_stage_l2_sec_per_byte_by_phase(
    ops_path: Path,
    pack_df: pd.DataFrame,
) -> Tuple[Dict[str, float], Optional[float]]:
    phase_coeffs: Dict[str, float] = {}
    overall_coeff: Optional[float] = None

    if pack_df.empty or "bytes" not in pack_df.columns:
        return phase_coeffs, overall_coeff

    bytes_series = pd.to_numeric(pack_df["bytes"], errors="coerce").fillna(0.0)
    total_pack_bytes = float(bytes_series.clip(lower=0.0).sum())
    if total_pack_bytes <= 0:
        return phase_coeffs, overall_coeff

    sidecars = weight_stage_sidecar_paths(ops_path)
    overall_path = sidecars["overall"]
    if overall_path.exists():
        try:
            overall_df = read_trace_csv(overall_path)
            if not overall_df.empty and "load_l2_s_sum" in overall_df.columns:
                l2_sum = safe_float(overall_df.iloc[0].get("load_l2_s_sum"))
                if l2_sum is not None and l2_sum >= 0:
                    overall_coeff = float(l2_sum / total_pack_bytes)
        except Exception as exc:
            warn(f"failed to read weight-stage overall CSV {overall_path}: {exc}")

    by_phase_path = sidecars["by_phase"]
    if by_phase_path.exists() and "phase" in pack_df.columns:
        try:
            by_phase_df = read_trace_csv(by_phase_path)
            if {"phase", "load_l2_s_sum"}.issubset(by_phase_df.columns):
                phase_key_series = safe_text_series(pack_df, "phase").str.lower()
                bytes_by_phase = (
                    pd.DataFrame({"phase_key": phase_key_series, "bytes": bytes_series})
                    .groupby("phase_key", dropna=False)["bytes"]
                    .sum()
                    .to_dict()
                )
                for _, row in by_phase_df.iterrows():
                    phase_key = safe_lower(row.get("phase"))
                    l2_sum = safe_float(row.get("load_l2_s_sum"))
                    phase_bytes = float(bytes_by_phase.get(phase_key, 0.0))
                    if phase_key and phase_bytes > 0 and l2_sum is not None and l2_sum >= 0:
                        phase_coeffs[phase_key] = float(l2_sum / phase_bytes)
        except Exception as exc:
            warn(f"failed to read weight-stage by-phase CSV {by_phase_path}: {exc}")

    return phase_coeffs, overall_coeff


def build_weight_stage_busy_intervals(
    *,
    comms_df: pd.DataFrame,
    ops_path: Path,
    mode: str,
) -> Dict[str, List[Tuple[float, float]]]:
    mode = (mode or DEFAULT_WEIGHT_STAGE_UTIL_MODE).strip().lower()
    if mode == DEFAULT_WEIGHT_STAGE_UTIL_MODE or comms_df.empty:
        return {}
    if mode not in {"l1", "l1_l2"}:
        raise ValueError(f"unknown weight-stage util mode: {mode}")

    if not {"dst", "start", "end"}.issubset(comms_df.columns):
        warn(f"weight-stage util mode requested but comms trace misses dst/start/end: {ops_path}")
        return {}

    load_df = comms_df.loc[weight_load_mask(comms_df)].copy()
    if load_df.empty:
        return {}

    intervals: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
    for _, row in load_df.iterrows():
        dev = str(row.get("dst", "")).strip()
        if not dev:
            continue
        start = safe_float(row.get("start"))
        end = safe_float(row.get("end"))
        if start is None or end is None or end <= start:
            continue
        intervals[dev].append((float(start), float(end)))

    if mode != "l1_l2":
        return {dev: list(v) for dev, v in intervals.items()}

    overlap_saved = weight_stage_overlap_saved_s(ops_path)
    if overlap_saved is not None and overlap_saved > 1e-12:
        warn(f"weight-stage L1/L2 overlap is non-zero for {ops_path}; L2 is modeled after L1 end")

    pack_df = load_df.loc[weight_load_needs_l2_mask(load_df)].copy()
    if pack_df.empty:
        return {dev: list(v) for dev, v in intervals.items()}
    if "bytes" not in pack_df.columns:
        warn(f"l1_l2 mode requested but comms trace misses bytes column: {ops_path}; using L1 only")
        return {dev: list(v) for dev, v in intervals.items()}

    phase_coeffs, overall_coeff = weight_stage_l2_sec_per_byte_by_phase(ops_path, pack_df)
    if overall_coeff is None and not phase_coeffs:
        warn(f"l1_l2 mode requested but no usable weight-stage sidecar found: {ops_path}; using L1 only")
        return {dev: list(v) for dev, v in intervals.items()}

    for _, row in pack_df.iterrows():
        dev = str(row.get("dst", "")).strip()
        if not dev:
            continue
        end = safe_float(row.get("end"))
        nbytes = safe_float(row.get("bytes"))
        if end is None or nbytes is None or nbytes <= 0:
            continue
        phase_key = safe_lower(row.get("phase"))
        coeff = phase_coeffs.get(phase_key, overall_coeff)
        if coeff is None or coeff <= 0:
            continue
        l2_dur = float(coeff * nbytes)
        if l2_dur > 0:
            intervals[dev].append((float(end), float(end + l2_dur)))

    return {dev: list(v) for dev, v in intervals.items()}


def resolve_phase_window(time_res: Optional[TimeResult], phase: str) -> Tuple[float, float, float]:
    if time_res is None:
        raise ValueError("missing baseline_compare timing for utilization")

    prefill_s = safe_float(time_res.prefill_time_s)
    decode_s = safe_float(time_res.decode_time_s)
    total_s = safe_float(time_res.total_time_s)

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


def normalize_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    arr: List[Tuple[float, float]] = []
    for start, end in intervals:
        if pd.isna(start) or pd.isna(end):
            continue
        s = float(start)
        e = float(end)
        if e <= s:
            continue
        arr.append((s, e))
    if not arr:
        return []

    arr.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[float, float]] = []
    cur_s, cur_e = arr[0]
    for s, e in arr[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def clip_intervals_to_window(
    intervals: Iterable[Tuple[float, float]],
    window_start: float,
    window_end: float,
) -> List[Tuple[float, float]]:
    clipped: List[Tuple[float, float]] = []
    for start, end in intervals:
        if pd.isna(start) or pd.isna(end):
            continue
        s = max(float(start), float(window_start))
        e = min(float(end), float(window_end))
        if e > s:
            clipped.append((s, e))
    return clipped


def family_mean(device_utils: Dict[str, float], device_families: Dict[str, str], family: str) -> Optional[float]:
    vals = [util for dev, util in device_utils.items() if device_families.get(dev) == family]
    return mean_or_none(vals)


def compute_device_utilization(
    *,
    run_id: str,
    source_algo: str,
    plotted_algo: str,
    prefill: int,
    decode: int,
    comms_path: Path,
    ops_path: Path,
    phase: str,
    include_cpu: bool,
    time_res: TimeResult,
    weight_stage_util_mode: str,
) -> UtilResult:
    ops_df = filter_phase(read_trace_csv(ops_path), phase=phase)
    if ops_df.empty:
        raise ValueError(f"ops trace is empty after phase filter={phase}: {ops_path}")
    if "device" not in ops_df.columns:
        raise ValueError(f"ops trace missing 'device' column: {ops_path}")
    if not {"start", "end"}.issubset(ops_df.columns):
        raise ValueError(f"ops trace missing start/end columns: {ops_path}")

    weight_stage_util_mode = (weight_stage_util_mode or DEFAULT_WEIGHT_STAGE_UTIL_MODE).strip().lower()
    comms_df = pd.DataFrame()
    if weight_stage_util_mode != DEFAULT_WEIGHT_STAGE_UTIL_MODE:
        comms_df = filter_phase(read_trace_csv(comms_path), phase=phase)

    window_start, window_end, runtime_s = resolve_phase_window(time_res, phase)

    device_families = collect_device_families(ops_df=ops_df, comms_df=comms_df, include_cpu=include_cpu)
    if not device_families:
        raise ValueError("no PIM/NPU physical devices discovered from ops trace")

    device_raw_intervals: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
    for dev in sorted(device_families.keys()):
        sub = ops_df[ops_df["device"].astype(str) == dev]
        for start, end in zip(sub["start"], sub["end"]):
            if pd.isna(start) or pd.isna(end):
                continue
            if float(end) > float(start):
                device_raw_intervals[dev].append((float(start), float(end)))

    if weight_stage_util_mode != DEFAULT_WEIGHT_STAGE_UTIL_MODE and not comms_df.empty:
        extra_intervals = build_weight_stage_busy_intervals(
            comms_df=comms_df,
            ops_path=ops_path,
            mode=weight_stage_util_mode,
        )
        for dev, intervals in extra_intervals.items():
            if dev in device_families:
                device_raw_intervals[dev].extend(intervals)

    device_utils: Dict[str, float] = {}
    for dev in sorted(device_families.keys()):
        clipped = clip_intervals_to_window(
            device_raw_intervals.get(dev, []),
            window_start=window_start,
            window_end=window_end,
        )
        merged = [(s - window_start, e - window_start) for s, e in normalize_intervals(clipped)]
        busy_time = sum(e - s for s, e in merged)
        device_utils[dev] = busy_time / runtime_s

    family_devices: Dict[str, List[str]] = {
        fam: sorted([dev for dev, f in device_families.items() if f == fam])
        for fam in sorted(set(device_families.values()))
    }
    family_counts = {fam: len(devs) for fam, devs in family_devices.items()}

    return UtilResult(
        run_id=run_id,
        source_algo=source_algo,
        plotted_algo=plotted_algo,
        prefill=prefill,
        decode=decode,
        overall_utilization=mean_or_none(list(device_utils.values())) or 0.0,
        pim_utilization=family_mean(device_utils, device_families, "pim"),
        accel_utilization=family_mean(device_utils, device_families, "accel"),
        makespan_s=runtime_s,
        n_devices=len(device_utils),
        family_counts=family_counts,
        family_devices=family_devices,
        device_utils=device_utils,
        device_families=device_families,
        comms_path=comms_path,
        ops_path=ops_path,
    )


# -----------------------------------------------------------------------------
# Runtime / speedup
# -----------------------------------------------------------------------------

def default_time_field_for_phase(phase: str) -> str:
    if phase == "prefill":
        return "prefill_time_s"
    if phase == "decode":
        return "decode_time_s"
    return "total_time_s"


def time_value(time_res: Optional[TimeResult], field: str) -> Optional[float]:
    if time_res is None:
        return None
    if field == "total_time_s":
        total = time_res.total_time_s
        if total is not None and total > 0:
            return float(total)
        if time_res.prefill_time_s is not None and time_res.decode_time_s is not None:
            eff_total = float(time_res.prefill_time_s) + float(time_res.decode_time_s)
            return eff_total if eff_total > 0 else None
        return None
    value = getattr(time_res, field, None)
    if value is None or value <= 0:
        return None
    return float(value)


def runtime_candidates(source_algo: Optional[str], plotted_algo: str) -> List[str]:
    out: List[str] = []
    for cand in [source_algo, plotted_algo]:
        if cand and cand not in out:
            out.append(cand)
    if normalize_algo_token(plotted_algo) == "hefthint":
        for cand in ["hefthint", "this work", "heft", "bifocal"]:
            if cand not in out:
                out.append(cand)
    return out


def lookup_time_res(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    run_id: str,
    prefill: int,
    decode: int,
    source_algo: Optional[str],
    plotted_algo: str,
) -> Optional[TimeResult]:
    for cand in runtime_candidates(source_algo, plotted_algo):
        tr = time_index.get((run_id, prefill, decode, cand))
        if tr is not None:
            return tr
    return None


def lookup_runtime_s(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    run_id: str,
    prefill: int,
    decode: int,
    source_algo: Optional[str],
    plotted_algo: str,
    time_field: str,
) -> Optional[float]:
    tr = lookup_time_res(
        time_index=time_index,
        run_id=run_id,
        prefill=prefill,
        decode=decode,
        source_algo=source_algo,
        plotted_algo=plotted_algo,
    )
    return time_value(tr, time_field)


def util_pick_score(res: UtilResult, mode: str) -> float:
    mode = (mode or "family_mean").strip().lower()
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
    return float("-inf") if not vals else sum(vals) / len(vals)


def order_algorithms(algos: Sequence[str], algo_order: Optional[Sequence[str]] = None) -> List[str]:
    present: List[str] = []
    for algo in algos:
        norm = normalize_algo_token(algo)
        if norm and norm not in present:
            present.append(norm)
    if not present:
        return []

    if algo_order:
        preferred: List[str] = []
        for algo in algo_order:
            norm = normalize_algo_token(algo)
            if norm in present and norm not in preferred:
                preferred.append(norm)
        rest = [a for a in present if a not in set(preferred)]
        return preferred + sorted(rest)

    ordered = [a for a in PREFERRED_ALGO_ORDER if a in present]
    rest = sorted([a for a in present if a not in set(ordered)])
    return ordered + rest


def compute_results_for_case(
    *,
    run_id: str,
    prefill: int,
    decode: int,
    index: Dict[TraceKey, TracePair],
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    algorithms: Sequence[str],
    phase: str,
    include_cpu: bool,
    heft_pick_by: str,
    weight_stage_util_mode: str,
    strict: bool,
) -> Dict[str, UtilResult]:
    # Group actual trace algorithm names under the plotted algorithm name.
    actual_by_plotted: DefaultDict[str, List[str]] = defaultdict(list)
    for key in index:
        if key.run_id != run_id or key.prefill != prefill or key.decode != decode:
            continue
        plotted = normalize_algo_token(key.algo)
        if key.algo not in actual_by_plotted[plotted]:
            actual_by_plotted[plotted].append(key.algo)

    results: Dict[str, UtilResult] = {}
    for plotted_algo in algorithms:
        candidates = sorted(actual_by_plotted.get(plotted_algo, []))
        if not candidates:
            warn(f"no trace found for algorithm={plotted_algo}")
            continue

        candidate_results: List[UtilResult] = []
        for source_algo in candidates:
            key = TraceKey(run_id=run_id, algo=source_algo, prefill=prefill, decode=decode)
            pair = index.get(key)
            if pair is None:
                continue
            tr = lookup_time_res(
                time_index=time_index,
                run_id=run_id,
                prefill=prefill,
                decode=decode,
                source_algo=source_algo,
                plotted_algo=plotted_algo,
            )
            if tr is None:
                msg = f"missing baseline_compare timing for algo={source_algo}, p={prefill}, d={decode}"
                if strict:
                    die(msg)
                warn(msg)
                continue
            try:
                res = compute_device_utilization(
                    run_id=run_id,
                    source_algo=source_algo,
                    plotted_algo=plotted_algo,
                    prefill=prefill,
                    decode=decode,
                    comms_path=pair.comms_path,
                    ops_path=pair.ops_path,
                    phase=phase,
                    include_cpu=include_cpu,
                    time_res=tr,
                    weight_stage_util_mode=weight_stage_util_mode,
                )
                candidate_results.append(res)
            except Exception as exc:
                msg = f"skip algo={source_algo}: {exc}"
                if strict:
                    die(msg)
                warn(msg)

        if not candidate_results:
            continue

        # When both heft and hefthint variants exist, keep the better one, same idea as the plot script.
        def sel_key(r: UtilResult) -> Tuple[float, int, float]:
            prefer_hefthint = 1 if r.source_algo == "hefthint" else 0
            return (util_pick_score(r, heft_pick_by), prefer_hefthint, r.overall_utilization)

        best = max(candidate_results, key=sel_key)
        best.plotted_algo = plotted_algo
        results[plotted_algo] = best

    return results


def resolve_speedup_reference(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    results: Dict[str, UtilResult],
    run_id: str,
    prefill: int,
    decode: int,
    algorithms: Sequence[str],
    time_field: str,
    speedup_ref: str,
) -> Tuple[Optional[str], Optional[float]]:
    ref_key = (speedup_ref or "slowest").strip().lower()

    available: List[Tuple[str, float]] = []
    for algo in algorithms:
        res = results.get(algo)
        runtime = lookup_runtime_s(
            time_index=time_index,
            run_id=run_id,
            prefill=prefill,
            decode=decode,
            source_algo=(res.source_algo if res else None),
            plotted_algo=algo,
            time_field=time_field,
        )
        if runtime is not None:
            available.append((algo, runtime))

    if ref_key in {"slowest", "worst", "max"}:
        return max(available, key=lambda kv: kv[1]) if available else (None, None)
    if ref_key in {"best", "fastest", "min"}:
        return min(available, key=lambda kv: kv[1]) if available else (None, None)

    ref_algo = normalize_algo_token(ref_key)
    res = results.get(ref_algo)
    runtime = lookup_runtime_s(
        time_index=time_index,
        run_id=run_id,
        prefill=prefill,
        decode=decode,
        source_algo=(res.source_algo if res else ref_algo),
        plotted_algo=ref_algo,
        time_field=time_field,
    )
    return ref_algo, runtime


# -----------------------------------------------------------------------------
# Selection and output
# -----------------------------------------------------------------------------

def select_single_run(run_infos: Dict[str, RunInfo]) -> RunInfo:
    if not run_infos:
        die("no matching run directory found")
    if len(run_infos) == 1:
        return next(iter(run_infos.values()))

    lines = ["multiple matching run directories; narrow with --batch-size / --shards / --model-prefix:"]
    for info in sorted(run_infos.values(), key=lambda x: x.run_dir.name):
        lines.append(f"  - {info.run_dir}  model={info.model_name} batch={info.batch_size} shards={info.shards}")
    die("\n".join(lines))


def select_case(
    *,
    run_id: str,
    index: Dict[TraceKey, TracePair],
    requested_prefill: Optional[int],
    requested_decode: Optional[int],
) -> Tuple[int, int]:
    cases = sorted({(k.prefill, k.decode) for k in index if k.run_id == run_id})
    if not cases:
        die("no trace cases found for selected run")

    if requested_prefill is not None and requested_decode is not None:
        case = (requested_prefill, requested_decode)
        if case not in cases:
            available = ", ".join(f"{p}x{d}" for p, d in cases)
            die(f"requested case {case[0]}x{case[1]} not found. available: {available}")
        return case

    if requested_prefill is not None or requested_decode is not None:
        die("--prefill and --decode must be provided together, or use --case PXD")

    if len(cases) == 1:
        return cases[0]

    available = ", ".join(f"{p}x{d}" for p, d in cases)
    die(f"multiple cases found; specify --prefill/--decode or --case. available: {available}")


def build_text_report(
    *,
    run_info: RunInfo,
    prefill: int,
    decode: int,
    phase: str,
    time_field: str,
    speedup_ref: str,
    speedup_ref_algo: Optional[str],
    speedup_ref_runtime: Optional[float],
    weight_stage_util_mode: str,
    algorithms: Sequence[str],
    results: Dict[str, UtilResult],
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    speedup_mode: str,
    util_scale: str,
    print_device_details: bool,
) -> str:
    lines: List[str] = []
    lines.append("=== single-case utilization / speedup ===")
    lines.append(f"run_dir: {run_info.run_dir}")
    lines.append(f"model: {run_info.model_name}")
    lines.append(f"batch_size: {run_info.batch_size}")
    lines.append(f"shards: {run_info.shards}")
    lines.append(f"prefill: {prefill}")
    lines.append(f"decode: {decode}")
    lines.append(f"phase: {phase}")
    lines.append(f"runtime_field: {time_field}")
    lines.append(f"weight_stage_util_mode: {weight_stage_util_mode}")
    lines.append(
        f"speedup_ref: {speedup_ref}"
        f" (resolved_algo={speedup_ref_algo or 'NA'}, runtime_s={format_float(speedup_ref_runtime, 9)})"
    )
    lines.append("")

    if util_scale == "fraction":
        pim_header = "PIM_util"
        npu_header = "NPU_util"
    else:
        pim_header = "PIM_util(%)"
        npu_header = "NPU_util(%)"

    header_cols = ["algorithm", "source_algo", "runtime_s", "speedup_x", pim_header, npu_header]
    widths = [18, 14, 14, 12, 13, 13]
    header = "  ".join(col.ljust(width) for col, width in zip(header_cols, widths))
    lines.append(header)
    lines.append("-" * len(header))

    for algo in algorithms:
        res = results.get(algo)
        if res is None:
            continue
        runtime_s = lookup_runtime_s(
            time_index=time_index,
            run_id=run_info.run_id,
            prefill=prefill,
            decode=decode,
            source_algo=res.source_algo,
            plotted_algo=algo,
            time_field=time_field,
        )
        speedup_ratio: Optional[float] = None
        if runtime_s is not None and speedup_ref_runtime is not None and runtime_s > 0 and speedup_ref_runtime > 0:
            ratio = float(speedup_ref_runtime / runtime_s)
            speedup_ratio = ratio if speedup_mode == "ratio" else ratio - 1.0

        if util_scale == "fraction":
            pim_val = format_float(res.pim_utilization, 6)
            npu_val = format_float(res.accel_utilization, 6)
        else:
            pim_val = format_percent(res.pim_utilization, 2)
            npu_val = format_percent(res.accel_utilization, 2)

        row = [
            algo,
            res.source_algo,
            format_float(runtime_s, 9),
            format_float(speedup_ratio, 6),
            pim_val,
            npu_val,
        ]
        lines.append("  ".join(str(val).ljust(width) for val, width in zip(row, widths)))

    if print_device_details:
        lines.append("")
        lines.append("--- device details ---")
        for algo in algorithms:
            res = results.get(algo)
            if res is None:
                continue
            lines.append(f"[{algo}] source_algo={res.source_algo}")
            for dev in sorted(res.device_utils):
                fam = res.device_families.get(dev, "unknown")
                util = res.device_utils[dev]
                util_txt = format_float(util, 6) if util_scale == "fraction" else format_percent(util, 2) + "%"
                lines.append(f"  {dev:<24} family={fam:<6} util={util_txt}")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Text-only speedup + PIM/NPU utilization for one case."
    )
    ap.add_argument("--search-dir", required=True, help="Root directory or direct run directory to scan.")
    ap.add_argument(
        "--model-prefix",
        default=None,
        help="Model prefix, e.g. qwen1.8b or qwen_1.8b_fp16. Compact matching ignores '_'/'-'/spaces.",
    )
    ap.add_argument("--batch-size", type=int, default=None, help="Batch size to select, e.g. 16.")
    ap.add_argument("--shards", type=int, default=None, help="Optional shard count to disambiguate run dirs.")
    ap.add_argument("--prefill", type=int, default=None, help="Prefill length of the case.")
    ap.add_argument("--decode", type=int, default=None, help="Decode length of the case.")
    ap.add_argument("--case", type=str, default=None, help="Shortcut for --prefill/--decode, e.g. 128x128.")
    ap.add_argument("--phase", choices=["all", "prefill", "decode"], default=DEFAULT_PHASE)
    ap.add_argument(
        "--include-algos",
        default=None,
        help="Comma-separated algorithms to print in this order, e.g. pd,ianus,facil,hefthint.",
    )
    ap.add_argument(
        "--exclude-algos",
        default=None,
        help="Comma-separated algorithms to skip, e.g. weights_on_pim.",
    )
    ap.add_argument(
        "--algo-order",
        default=None,
        help="Optional algorithm order when --include-algos is not set.",
    )
    ap.add_argument("--speedup-ref", default=DEFAULT_SPEEDUP_REF, help="Algorithm name, or slowest/best.")
    ap.add_argument(
        "--speedup-time-field",
        choices=["auto", "total_time_s", "prefill_time_s", "decode_time_s"],
        default="auto",
        help="Timing field from baseline_compare JSON. auto follows --phase.",
    )
    ap.add_argument(
        "--speedup-mode",
        choices=["ratio", "improvement"],
        default=DEFAULT_SPEEDUP_MODE,
        help="ratio prints ref/algo. improvement prints ref/algo - 1.",
    )
    ap.add_argument(
        "--util-scale",
        choices=["percent", "fraction"],
        default="percent",
        help="Print utilization as percent or 0~1 fraction.",
    )
    ap.add_argument("--include-cpu", action="store_true", help="Include CPU in device discovery; PIM/NPU columns unchanged.")
    ap.add_argument(
        "--heft-pick-by",
        choices=["family_mean", "overall", "pim", "accel"],
        default="family_mean",
        help="If multiple HEFT-like traces exist, choose the one with this utilization criterion.",
    )
    ap.add_argument(
        "--weight-stage-util-mode",
        choices=["none", "l1", "l1_l2"],
        default=DEFAULT_WEIGHT_STAGE_UTIL_MODE,
        help="none: legacy ops-only; l1: add weight-load transfer; l1_l2: also add sidecar-modeled pack time.",
    )
    ap.add_argument(
        "--include-weight-stage-time",
        dest="weight_stage_util_mode",
        action="store_const",
        const="l1_l2",
        help="Alias for --weight-stage-util-mode l1_l2.",
    )
    ap.add_argument("--output-text", default=None, help="Optional path to also save the text report.")
    ap.add_argument("--print-device-details", action="store_true", help="Also print per-device utilization.")
    ap.add_argument("--strict", action="store_true", help="Fail on any missing/invalid algorithm instead of warning.")
    args = ap.parse_args()

    search_dir = Path(args.search_dir)
    if not search_dir.exists():
        die(f"--search-dir not found: {search_dir}")

    case_prefill, case_decode = parse_case_token(args.case)
    if case_prefill is not None:
        if args.prefill is not None or args.decode is not None:
            die("use either --case or --prefill/--decode, not both")
        args.prefill = case_prefill
        args.decode = case_decode

    trace_paths = discover_trace_files(search_dir)
    if not trace_paths:
        die(f"No *_trace.csv found under: {search_dir}")

    all_run_infos: Dict[str, RunInfo] = {}
    for path in trace_paths:
        info = infer_run_info(path)
        all_run_infos[info.run_id] = info

    filtered_run_infos = filter_run_infos(
        all_run_infos,
        model_prefix=args.model_prefix,
        batch_size=args.batch_size,
        shards=args.shards,
    )
    run_info = select_single_run(filtered_run_infos)

    allowed_run_ids = [run_info.run_id]
    index = build_trace_index(trace_paths, allowed_run_ids=allowed_run_ids)
    if not index:
        die("No valid comms/ops trace pairs found after run filtering.")

    baseline_paths = discover_baseline_compare_files(search_dir)
    time_index = build_time_index(baseline_paths, allowed_run_ids=allowed_run_ids)
    if not baseline_paths:
        die(f"No baseline_compare_*.json found under: {search_dir}")
    if not time_index:
        die("baseline_compare_*.json exists, but no valid timings were parsed.")

    prefill, decode = select_case(
        run_id=run_info.run_id,
        index=index,
        requested_prefill=args.prefill,
        requested_decode=args.decode,
    )

    actual_algos = sorted({
        key.algo
        for key in index
        if key.run_id == run_info.run_id and key.prefill == prefill and key.decode == decode
    })
    plotted_algos_all = sorted({normalize_algo_token(a) for a in actual_algos})

    include_algos = [normalize_algo_token(a) for a in parse_str_list(args.include_algos)]
    exclude_algos = {normalize_algo_token(a) for a in parse_str_list(args.exclude_algos)}
    algo_order = parse_str_list(args.algo_order)

    if include_algos:
        algorithms = [a for a in include_algos if a in plotted_algos_all]
        missing = [a for a in include_algos if a not in plotted_algos_all]
        if missing:
            warn(f"requested include-algos not found: {missing}")
    else:
        algorithms = order_algorithms(plotted_algos_all, algo_order=algo_order)

    if exclude_algos:
        algorithms = [a for a in algorithms if a not in exclude_algos]

    if not algorithms:
        die("No algorithms left after include/exclude filtering.")

    time_field = args.speedup_time_field
    if time_field == "auto":
        time_field = default_time_field_for_phase(args.phase)

    eprint(
        f"[INFO] selected run={run_info.run_dir.name} | case={prefill}x{decode} | "
        f"algorithms={','.join(algorithms)} | phase={args.phase}"
    )

    results = compute_results_for_case(
        run_id=run_info.run_id,
        prefill=prefill,
        decode=decode,
        index=index,
        time_index=time_index,
        algorithms=algorithms,
        phase=args.phase,
        include_cpu=args.include_cpu,
        heft_pick_by=args.heft_pick_by,
        weight_stage_util_mode=args.weight_stage_util_mode,
        strict=args.strict,
    )
    if not results:
        die("No usable algorithm results for the selected case.")

    active_algorithms = [a for a in algorithms if a in results]
    speedup_ref_algo, speedup_ref_runtime = resolve_speedup_reference(
        time_index=time_index,
        results=results,
        run_id=run_info.run_id,
        prefill=prefill,
        decode=decode,
        algorithms=active_algorithms,
        time_field=time_field,
        speedup_ref=args.speedup_ref,
    )
    if speedup_ref_runtime is None:
        die(f"Could not resolve speedup reference runtime for --speedup-ref={args.speedup_ref!r}")

    report = build_text_report(
        run_info=run_info,
        prefill=prefill,
        decode=decode,
        phase=args.phase,
        time_field=time_field,
        speedup_ref=args.speedup_ref,
        speedup_ref_algo=speedup_ref_algo,
        speedup_ref_runtime=speedup_ref_runtime,
        weight_stage_util_mode=args.weight_stage_util_mode,
        algorithms=active_algorithms,
        results=results,
        time_index=time_index,
        speedup_mode=args.speedup_mode,
        util_scale=args.util_scale,
        print_device_details=args.print_device_details,
    )

    print(report)
    if args.output_text:
        out_path = Path(args.output_text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        eprint(f"[OK] wrote text report -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
