
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot three stacked violin plots for:
  1) speedup ratio relative to the chosen reference
  2) PIM + NPU utilization (combined in one axis)
  3) co-utilization

Compared with the original line-plot script, this version aggregates *all*
(batch size, prefill length, decode length) cases into distributions per
algorithm, which is a better fit for violin plots.

Key changes
-----------
- Uses seaborn.violinplot instead of per-(prefill, decode) line plots.
- Scans all batch-size folders under a model root when requested.
- Keeps the original utilization / co-utilization computation logic.
- Keeps on-disk cache support.
- Uses "Arial" as the preferred plotting font.
- Enlarges fonts and spacing to avoid overlap.

Typical usage
-------------
# Scan all llama_7b_fp16 batch-size folders under sst8_rst8
python3 plot_exp1_utilization_violin.py \
  --search-dir ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst8_rst8 \
  --model-prefix llama_7b_fp16 \
  --exclude-algos weights_on_pim \
  --speedup-plot-max 3 \
  --algo-label-map 'hefthint=Bifocal (this work),pd=PD,attn_on_pim=AF,ianus=PD+FFN,facil=PD+Linear,attacc=PD+Attention' \
  --output ../../figs/exp1/util/llama_7b_fp16_all_batches_violin.pdf

# Or point directly to a batch-size folder; the script will still work
python3 plot_exp1_utilization_violin.py \
  --search-dir ../../algorithms/output/exp1/hw_hardware_1npu_2aim/sst8_rst8/llama_7b_fp16_b16_s8 \
  --output ../../figs/exp1/util/llama_7b_fp16_b16_violin.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
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
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------

PLOT_FONT_FAMILY = "Arial"
COMMON_UI_FONT_PT = 22
TICK_FONT_PT = 19
ANNOT_FONT_PT = 16

METRIC_COLORS: Dict[str, str] = {
    "speedup": "#39a937",
    "pim_utilization": "#3760a9",
    "accel_utilization": "#5837a8",
    "concurrent_utilization": "#a83747",
}
DEFAULT_ALPHA = 0.58
DEFAULT_STRIP_ALPHA = 0.95
DEFAULT_STRIP_SIZE = 2.8
DEFAULT_STRIP_JITTER = 0.16
DEFAULT_INNER_STYLE = "quart"
DEFAULT_VIOLIN_CUT = 0.0
DEFAULT_DENSITY_NORM = "count"
DEFAULT_UTIL_GAP = 0.03
DEFAULT_SPEEDUP_REF = "pd"
DEFAULT_SPEEDUP_MODE = "ratio"
DEFAULT_SPEEDUP_PLOT_MAX = 5.0
DEFAULT_UTIL_LAYOUT = "overlay"

CACHE_VERSION = "violin_v9"

TRACE_RE = re.compile(
    r"^(?P<algo>.+?)_prefill-(?P<prefill>\d+)xdecode_(?P<decode>\d+)_(?P<kind>comms|ops)_trace\.csv$",
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

DISPLAY_NAME_MAP: Dict[str, str] = {
    "hefthint": "Bifocal (this work)",
}

HEFT_VARIANTS = {"heft", "hefthint"}

FAMILY_ORDER = ["pim", "accel"]
FAMILY_LABELS: Dict[str, str] = {
    "pim": "PIM",
    "accel": "NPU",
    "cpu": "CPU",
}


def apply_global_plot_style(
    *,
    font_family: str = PLOT_FONT_FAMILY,
    font_size: float = COMMON_UI_FONT_PT,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": [font_family],
        "font.sans-serif": [font_family],
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": TICK_FONT_PT,
        "ytick.labelsize": TICK_FONT_PT,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def enforce_figure_fonts(
    fig: plt.Figure,
    *,
    min_font_pt: float = COMMON_UI_FONT_PT,
    font_family: str = PLOT_FONT_FAMILY,
) -> None:
    for text in fig.findobj(Text):
        try:
            current_size = float(text.get_fontsize())
        except (TypeError, ValueError):
            current_size = min_font_pt
        text.set_fontfamily(font_family)
        text.set_fontsize(max(min_font_pt, current_size))


apply_global_plot_style()


# ---------------------------------------------------------------------
# Small runtime helpers
# ---------------------------------------------------------------------

class ProgressPrinter:
    def __init__(self, total: int, *, label: str = "items", enabled: bool = True) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.enabled = bool(enabled) and self.total > 0
        self.done = 0
        self.stats: Counter[str] = Counter()
        self._step = max(1, self.total // 40)

    def update(self, status: str = "processed") -> None:
        if not self.enabled:
            return
        self.done += 1
        self.stats[status] += 1
        should_print = (
            self.done == 1
            or self.done == self.total
            or (self.done % self._step == 0)
        )
        if should_print:
            pct = 100.0 * self.done / max(self.total, 1)
            parts = [
                f"\r[PROGRESS] {self.label}: {self.done}/{self.total} ({pct:5.1f}%)",
            ]
            for key in ["computed", "disk_cache", "memory_cache", "error"]:
                if self.stats.get(key, 0):
                    parts.append(f"{key}={self.stats[key]}")
            end = "\n" if self.done == self.total else ""
            print(" | ".join(parts), end=end, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

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


@dataclass
class TracePair:
    comms_path: Path
    ops_path: Path


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
    run_id: str
    source_algo: str
    plotted_algo: str
    prefill: int
    decode: int
    prefill_time_s: Optional[float]
    decode_time_s: Optional[float]
    total_time_s: Optional[float]
    json_path: Path


# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------

def util_result_to_dict(res: UtilResult) -> Dict[str, object]:
    return {
        "run_id": res.run_id,
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
        run_id=str(data["run_id"]),
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


def _util_cache_key(
    comms_path: Path,
    ops_path: Path,
    phase: str,
    include_cpu: bool,
    time_res: Optional[TimeResult],
) -> str:
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


def _load_util_cache(
    cache_dir: Path,
    comms_path: Path,
    ops_path: Path,
    phase: str,
    include_cpu: bool,
    time_res: Optional[TimeResult],
) -> Optional[UtilResult]:
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


def _save_util_cache(
    cache_dir: Path,
    res: UtilResult,
    phase: str,
    include_cpu: bool,
    time_res: Optional[TimeResult],
) -> None:
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


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def parse_int_list(s: Optional[str]) -> List[int]:
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


def normalize_algo_token(algo: str) -> str:
    raw = (algo or "").strip().lower()
    compact = re.sub(r"[\s_\-]+", "", raw)
    if compact in {"heft", "hefthint", "thiswork"}:
        return "hefthint"
    return raw


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


def discover_trace_files(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("*_trace.csv"))


def discover_baseline_compare_files(search_dir: Path) -> List[Path]:
    return sorted(search_dir.rglob("baseline_compare_*.json"))


def infer_run_info(path: Path) -> RunInfo:
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

    # Fallback: use file parent when the directory name does not follow *_b*_s*.
    return RunInfo(
        run_id=str(path.parent.resolve()),
        run_dir=path.parent,
        model_name=path.parent.name.lower(),
        batch_size=None,
        shards=None,
    )


def filter_run_infos(
    run_infos: Dict[str, RunInfo],
    *,
    model_prefix: Optional[str] = None,
) -> Dict[str, RunInfo]:
    if not model_prefix:
        return dict(run_infos)

    prefix = model_prefix.strip().lower()
    out = {
        run_id: info
        for run_id, info in run_infos.items()
        if info.model_name.startswith(prefix) or info.run_dir.name.lower().startswith(prefix)
    }
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
            print(f"[WARN] skip invalid baseline json {path}: {exc}", file=sys.stderr)
            continue

        parsed = parse_case_from_baseline_payload(payload, path)
        if parsed is None:
            continue
        prefill, decode = parsed

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

            key = (run_info.run_id, prefill, decode, source_algo)
            time_res = TimeResult(
                run_id=run_info.run_id,
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
                    f"[WARN] duplicate timing for run={run_info.run_dir.name} "
                    f"prefill={prefill} decode={decode} algo={source_algo}; "
                    f"keep first from {out[key].json_path}",
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


def build_trace_index(
    paths: Sequence[Path],
    allowed_run_ids: Optional[Sequence[str]] = None,
) -> Dict[TraceKey, TracePair]:
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

        key = TraceKey(
            run_id=run_info.run_id,
            algo=algo,
            prefill=prefill,
            decode=decode,
        )
        raw.setdefault(key, {}).setdefault(kind, []).append(path)

    out: Dict[TraceKey, TracePair] = {}
    for key, bucket in raw.items():
        pair = _pair_from_candidates(bucket.get("comms", []), bucket.get("ops", []))
        if pair is not None:
            out[key] = pair
    return out


# ---------------------------------------------------------------------
# Device-family helpers
# ---------------------------------------------------------------------

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


def _resolve_device_family(
    device: str,
    type_hints: Sequence[str],
    include_cpu: bool = False,
) -> Optional[str]:
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


def _collect_device_families(
    ops_df: pd.DataFrame,
    comms_df: pd.DataFrame,
    include_cpu: bool = False,
) -> Dict[str, str]:
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


# ---------------------------------------------------------------------
# Utilization computation
# ---------------------------------------------------------------------

def _filter_phase(df: pd.DataFrame, phase: str) -> pd.DataFrame:
    if phase == "all":
        return df.copy()
    if "phase" not in df.columns:
        return df.copy()
    return df[df["phase"].astype(str).str.lower() == phase.lower()].copy()


def _clip_intervals_to_window(
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
        if e <= s:
            continue
        clipped.append((s, e))
    return clipped


def _resolve_phase_window(time_res: Optional[TimeResult], phase: str) -> Tuple[float, float, float]:
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


def _family_mean(device_utils: Dict[str, float], device_families: Dict[str, str], family: str) -> Optional[float]:
    vals = [util for dev, util in device_utils.items() if device_families.get(dev) == family]
    if not vals:
        return None
    return float(np.mean(vals))


def _family_concurrent_utilization(
    device_busy_intervals: Dict[str, List[Tuple[float, float]]],
    device_families: Dict[str, str],
    family_counts: Dict[str, int],
    makespan: float,
    lhs_family: str = "pim",
    rhs_family: str = "accel",
) -> Optional[float]:
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


def compute_device_utilization(
    *,
    run_id: str,
    source_algo: str,
    plotted_algo: str,
    prefill: int,
    decode: int,
    comms_path: Path,
    ops_path: Path,
    phase: str = "all",
    include_cpu: bool = False,
    time_res: Optional[TimeResult] = None,
) -> UtilResult:
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

    return UtilResult(
        run_id=run_id,
        source_algo=source_algo,
        plotted_algo=plotted_algo,
        prefill=prefill,
        decode=decode,
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


# ---------------------------------------------------------------------
# Selection / ordering
# ---------------------------------------------------------------------

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


def pretty_algo_name(algo: str, display_name_map: Optional[Dict[str, str]] = None) -> str:
    name_map = DISPLAY_NAME_MAP if display_name_map is None else display_name_map
    return name_map.get(algo, algo)


PREFERRED_ALGO_ORDER: List[str] = ["pd", "attn_on_pim", "ianus", "facil", "attacc"]


def order_algorithms(
    all_algos: Sequence[str],
    algo_order: Optional[Sequence[str]] = None,
) -> List[str]:
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


# ---------------------------------------------------------------------
# Results building
# ---------------------------------------------------------------------

def _get_or_compute(
    mem_cache: Dict[TraceKey, UtilResult],
    index: Dict[TraceKey, TracePair],
    key: TraceKey,
    *,
    phase: str,
    include_cpu: bool,
    time_res: Optional[TimeResult],
    disk_cache_dir: Optional[Path] = None,
) -> Tuple[Optional[UtilResult], str]:
    if key in mem_cache:
        return mem_cache[key], "memory_cache"

    pair = index.get(key)
    if pair is None:
        return None, "missing"

    if time_res is None:
        raise ValueError(
            f"missing baseline_compare timing for run_id={key.run_id} "
            f"algo={key.algo} prefill={key.prefill} decode={key.decode}"
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
            mem_cache[key] = cached
            return cached, "disk_cache"

    res = compute_device_utilization(
        run_id=key.run_id,
        source_algo=key.algo,
        plotted_algo=normalize_algo_token(key.algo),
        prefill=key.prefill,
        decode=key.decode,
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

    return res, "computed"


def build_results(
    *,
    index: Dict[TraceKey, TracePair],
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    prefills: Sequence[int],
    decodes: Sequence[int],
    phase: str = "all",
    include_cpu: bool = False,
    heft_pick_by: str = "family_mean",
    disk_cache_dir: Optional[Path] = None,
) -> Dict[Tuple[str, int, int, str], UtilResult]:
    results: Dict[Tuple[str, int, int, str], UtilResult] = {}
    cache: Dict[TraceKey, UtilResult] = {}

    actual_algos = sorted({key.algo for key in index})
    normal_algos = [a for a in actual_algos if a not in HEFT_VARIANTS]
    run_ids = sorted({key.run_id for key in index})

    total_jobs = 0
    for run_id in run_ids:
        for p in prefills:
            for d in decodes:
                for algo in normal_algos:
                    if TraceKey(run_id=run_id, algo=algo, prefill=p, decode=d) in index:
                        total_jobs += 1
                for variant in ["heft", "hefthint"]:
                    if TraceKey(run_id=run_id, algo=variant, prefill=p, decode=d) in index:
                        total_jobs += 1

    progress = ProgressPrinter(total_jobs, label="trace pairs")

    for run_id in run_ids:
        for p in prefills:
            for d in decodes:
                for algo in normal_algos:
                    key = TraceKey(run_id=run_id, algo=algo, prefill=p, decode=d)
                    if key not in index:
                        continue
                    try:
                        res, status = _get_or_compute(
                            cache,
                            index,
                            key,
                            phase=phase,
                            include_cpu=include_cpu,
                            time_res=time_index.get((run_id, p, d, algo)),
                            disk_cache_dir=disk_cache_dir,
                        )
                        progress.update(status=status)
                        if res is not None:
                            results[(run_id, p, d, algo)] = res
                    except Exception as exc:
                        progress.update(status="error")
                        print(
                            f"[WARN] skip run={run_id} algo={algo} prefill={p} decode={d}: {exc}",
                            file=sys.stderr,
                        )

                variant_results: List[UtilResult] = []
                for variant in ["heft", "hefthint"]:
                    key = TraceKey(run_id=run_id, algo=variant, prefill=p, decode=d)
                    if key not in index:
                        continue
                    try:
                        res, status = _get_or_compute(
                            cache,
                            index,
                            key,
                            phase=phase,
                            include_cpu=include_cpu,
                            time_res=time_index.get((run_id, p, d, variant)),
                            disk_cache_dir=disk_cache_dir,
                        )
                        progress.update(status=status)
                        if res is not None:
                            variant_results.append(res)
                    except Exception as exc:
                        progress.update(status="error")
                        print(
                            f"[WARN] skip run={run_id} {variant} prefill={p} decode={d}: {exc}",
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
                        run_id=best.run_id,
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
                    results[(run_id, p, d, "hefthint")] = best_for_plot

    return results


# ---------------------------------------------------------------------
# Runtime / speedup helpers
# ---------------------------------------------------------------------

def _default_time_field_for_phase(phase: str) -> str:
    if phase == "prefill":
        return "prefill_time_s"
    if phase == "decode":
        return "decode_time_s"
    return "total_time_s"


def _time_value(time_res: Optional[TimeResult], field: str) -> Optional[float]:
    if time_res is None:
        return None

    if field == "total_time_s":
        total = getattr(time_res, "total_time_s", None)
        if total is not None and total > 0:
            return float(total)
        prefill = getattr(time_res, "prefill_time_s", None)
        decode = getattr(time_res, "decode_time_s", None)
        if prefill is not None and decode is not None and prefill > 0 and decode >= 0:
            eff_total = float(prefill) + float(decode)
            if eff_total > 0:
                return eff_total
        return None

    v = getattr(time_res, field, None)
    if v is None:
        return None
    if v <= 0:
        return None
    return float(v)


def _lookup_runtime_info(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    run_id: str,
    prefill: int,
    decode: int,
    candidates: Sequence[str],
    time_field: str,
) -> Tuple[Optional[str], Optional[float]]:
    seen = set()
    for cand in candidates:
        cand = (cand or "").strip().lower()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        value = _time_value(time_index.get((run_id, prefill, decode, cand)), time_field)
        if value is not None:
            return cand, value
    return None, None


def lookup_runtime_info(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    run_id: str,
    prefill: int,
    decode: int,
    plotted_algo: str,
    res: Optional[UtilResult],
    time_field: str,
    mode: str = "plot",
) -> Tuple[Optional[str], Optional[float]]:
    candidates: List[str] = []
    norm_algo = normalize_algo_token(plotted_algo)
    mode = (mode or "plot").strip().lower()

    if mode == "speedup":
        if norm_algo == "hefthint":
            candidates.extend(["hefthint", "this work"])
        else:
            candidates.append(norm_algo)
    else:
        if res is not None and res.source_algo:
            candidates.append(res.source_algo)
        if norm_algo == "hefthint":
            candidates.extend(["hefthint", "this work", "heft"])
        else:
            candidates.append(norm_algo)

    return _lookup_runtime_info(
        time_index=time_index,
        run_id=run_id,
        prefill=prefill,
        decode=decode,
        candidates=candidates,
        time_field=time_field,
    )


def lookup_runtime_s(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    run_id: str,
    prefill: int,
    decode: int,
    plotted_algo: str,
    res: Optional[UtilResult],
    time_field: str,
    mode: str = "plot",
) -> Optional[float]:
    _algo, value = lookup_runtime_info(
        time_index=time_index,
        run_id=run_id,
        prefill=prefill,
        decode=decode,
        plotted_algo=plotted_algo,
        res=res,
        time_field=time_field,
        mode=mode,
    )
    return value


def _available_runtimes_for_case(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    results: Dict[Tuple[str, int, int, str], UtilResult],
    run_id: str,
    prefill: int,
    decode: int,
    algorithms: Sequence[str],
    time_field: str,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for algo in algorithms:
        res = results.get((run_id, prefill, decode, algo))
        runtime = lookup_runtime_s(
            time_index=time_index,
            run_id=run_id,
            prefill=prefill,
            decode=decode,
            plotted_algo=algo,
            res=res,
            time_field=time_field,
        )
        if runtime is not None:
            out.append((algo, runtime))
    return out


def resolve_speedup_reference(
    *,
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    results: Dict[Tuple[str, int, int, str], UtilResult],
    run_id: str,
    prefill: int,
    decode: int,
    algorithms: Sequence[str],
    time_field: str,
    speedup_ref: str,
) -> Tuple[Optional[str], Optional[float]]:
    ref_key = (speedup_ref or "slowest").strip().lower()
    available = _available_runtimes_for_case(
        time_index=time_index,
        results=results,
        run_id=run_id,
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
        run_id=run_id,
        prefill=prefill,
        decode=decode,
        plotted_algo=ref_algo,
        res=results.get((run_id, prefill, decode, ref_algo)),
        time_field=time_field,
        mode="speedup",
    )
    if runtime is None:
        return ref_algo, None
    return ref_algo, runtime


# ---------------------------------------------------------------------
# Dataframe assembly
# ---------------------------------------------------------------------

def build_plot_dataframe(
    *,
    results: Dict[Tuple[str, int, int, str], UtilResult],
    time_index: Dict[Tuple[str, int, int, str], TimeResult],
    run_infos: Dict[str, RunInfo],
    algorithms: Sequence[str],
    display_name_map: Optional[Dict[str, str]],
    time_field: str,
    speedup_ref: str,
    speedup_mode: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    speedup_mode = (speedup_mode or DEFAULT_SPEEDUP_MODE).strip().lower()

    for (run_id, prefill, decode, plotted_algo), res in sorted(
        results.items(),
        key=lambda x: (
            run_infos.get(x[0][0]).batch_size if run_infos.get(x[0][0]) and run_infos.get(x[0][0]).batch_size is not None else -1,
            x[0][1],
            x[0][2],
            x[0][3],
        ),
    ):
        speedup_runtime_algo, runtime_s = lookup_runtime_info(
            time_index=time_index,
            run_id=run_id,
            prefill=prefill,
            decode=decode,
            plotted_algo=plotted_algo,
            res=res,
            time_field=time_field,
            mode="speedup",
        )
        speedup_ref_algo, speedup_ref_time_s = resolve_speedup_reference(
            time_index=time_index,
            results=results,
            run_id=run_id,
            prefill=prefill,
            decode=decode,
            algorithms=algorithms,
            time_field=time_field,
            speedup_ref=speedup_ref,
        )

        speedup_ratio = float("nan")
        speedup_improvement = float("nan")
        speedup = float("nan")
        if runtime_s is not None and speedup_ref_time_s is not None and runtime_s > 0 and speedup_ref_time_s > 0:
            speedup_ratio = float(speedup_ref_time_s / runtime_s)
            speedup_improvement = float(speedup_ratio - 1.0)
            speedup = speedup_ratio if speedup_mode == "ratio" else speedup_improvement

        run_info = run_infos.get(run_id)
        batch_size = None if run_info is None else run_info.batch_size
        model_name = "" if run_info is None else run_info.model_name
        shards = None if run_info is None else run_info.shards
        case_id = f"{model_name}|b{batch_size}|p{prefill}|d{decode}"

        rows.append({
            "run_id": run_id,
            "model_name": model_name,
            "batch_size": batch_size,
            "shards": shards,
            "prefill": prefill,
            "decode": decode,
            "case_id": case_id,
            "plotted_algo": plotted_algo,
            "source_algo": res.source_algo,
            "algorithm_label": pretty_algo_name(plotted_algo, display_name_map=display_name_map),
            "runtime_field": time_field,
            "runtime_s": runtime_s,
            "speedup_runtime_algo": speedup_runtime_algo,
            "speedup_ref": speedup_ref,
            "speedup_ref_algo": speedup_ref_algo,
            "speedup_ref_time_s": speedup_ref_time_s,
            "speedup_mode": speedup_mode,
            "speedup_ratio": speedup_ratio,
            "speedup_improvement": speedup_improvement,
            "speedup": speedup,
            "overall_utilization_fraction": res.overall_utilization,
            "pim_utilization_fraction": res.pim_utilization,
            "accel_utilization_fraction": res.accel_utilization,
            "concurrent_utilization_fraction": res.concurrent_utilization,
            "overall_utilization_percent": None if res.overall_utilization is None else 100.0 * res.overall_utilization,
            "pim_utilization_percent": None if res.pim_utilization is None else 100.0 * res.pim_utilization,
            "accel_utilization_percent": None if res.accel_utilization is None else 100.0 * res.accel_utilization,
            "concurrent_utilization_percent": None if res.concurrent_utilization is None else 100.0 * res.concurrent_utilization,
            "n_devices": res.n_devices,
            "n_pim_devices": res.family_counts.get("pim", 0),
            "n_accel_devices": res.family_counts.get("accel", 0),
            "comms_path": str(res.comms_path),
            "ops_path": str(res.ops_path),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    key: str
    column: str
    ylabel: str
    color: str
    ylim: Optional[Tuple[float, float]] = None


def rgba_with_alpha(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    r, g, b, _ = to_rgba(hex_color)
    return (r, g, b, alpha)


def _nice_ceil(x: float) -> float:
    if not np.isfinite(x) or x <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(x))
    frac = x / exp
    for mult in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if frac <= mult:
            return mult * exp
    return 10.0 * exp



def _positive_ylim(values: Sequence[float], *, pad_ratio: float = 0.12, fallback_top: float = 1.0) -> Tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return (0.0, fallback_top)
    vmax = max(finite)
    if vmax <= 0:
        return (0.0, fallback_top)
    return (0.0, _nice_ceil(vmax * (1.0 + pad_ratio)))


def _value_ylim(
    values: Sequence[float],
    *,
    include_zero: bool = True,
    pad_ratio: float = 0.12,
    min_pad_abs: float = 0.06,
) -> Tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if include_zero:
        finite.append(0.0)
    if not finite:
        return (-0.5, 0.5)

    vmin = min(finite)
    vmax = max(finite)
    if abs(vmax - vmin) <= 1e-12:
        pad = max(min_pad_abs, abs(vmax) * pad_ratio + min_pad_abs)
        return (vmin - pad, vmax + pad)

    span = vmax - vmin
    pad_low = max(min_pad_abs, abs(vmin) * pad_ratio, span * 0.08)
    pad_high = max(min_pad_abs, abs(vmax) * pad_ratio, span * 0.08)

    lo = vmin - pad_low
    hi = vmax + pad_high

    if lo < 0:
        lo = -_nice_ceil(abs(lo))
    else:
        lo = 0.0

    if hi > 0:
        hi = _nice_ceil(hi)
    else:
        hi = 0.0

    if abs(hi - lo) <= 1e-12:
        hi = lo + 1.0
    return (lo, hi)


def _util_ylim(
    values: Sequence[float],
    *,
    util_scale: str,
    near_full_threshold: float = 92.0,
    pad_ratio: float = 0.10,
) -> Tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return (0.0, 100.0 if util_scale == "percent" else 1.0)

    vmax = max(finite)
    if util_scale == "percent":
        if vmax >= near_full_threshold:
            top = 100.0
        else:
            top = min(100.0, _nice_ceil(max(1.0, vmax * (1.0 + pad_ratio))))
        return (0.0, top)

    threshold = near_full_threshold / 100.0
    if vmax >= threshold:
        top = 1.0
    else:
        top = min(1.0, _nice_ceil(max(0.01, vmax * (1.0 + pad_ratio))))
    return (0.0, top)


def _set_left_ylabel(ax: plt.Axes, text: str, x: float = -0.11, fontsize: float = COMMON_UI_FONT_PT) -> None:
    ax.set_ylabel(text, fontsize=fontsize)
    ax.yaxis.set_label_coords(x, 0.5)


def _style_axis_box(ax: plt.Axes) -> None:
    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1.25)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.25)


def _style_violin_artists(
    ax: plt.Axes,
    *,
    start_collection: int,
    start_line: int,
    alpha: float,
    edgecolor: str = "black",
    linewidth: float = 1.15,
) -> List[object]:
    collections = list(ax.collections[start_collection:])
    for coll in collections:
        try:
            face = np.array(coll.get_facecolor(), copy=True)
            if face.size:
                face[:, -1] = alpha
                coll.set_facecolor(face)
        except Exception:
            pass
        try:
            coll.set_edgecolor(edgecolor)
        except Exception:
            pass
        try:
            coll.set_linewidth(linewidth)
        except Exception:
            pass
    for line in ax.lines[start_line:]:
        try:
            line.set_color("black")
            line.set_linewidth(max(1.0, linewidth))
            line.set_zorder(5)
            if hasattr(line, "set_markerfacecolor"):
                line.set_markerfacecolor("white")
                line.set_markeredgecolor("black")
        except Exception:
            pass
    return collections


def _emphasize_violin(collection: object, *, alpha: float = 0.85, linewidth: float = 2.0) -> None:
    try:
        face = np.array(collection.get_facecolor(), copy=True)
        if face.size:
            face[:, -1] = alpha
            collection.set_facecolor(face)
    except Exception:
        pass
    try:
        collection.set_edgecolor("black")
        collection.set_linewidth(linewidth)
        collection.set_zorder(4)
    except Exception:
        pass


def _compat_violinplot(
    *,
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Sequence[str],
    bw_adjust: float,
    color: Optional[str] = None,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    dodge: bool = True,
    inner: str = DEFAULT_INNER_STYLE,
    cut: float = DEFAULT_VIOLIN_CUT,
    density_norm: str = DEFAULT_DENSITY_NORM,
    split: bool = False,
    gap: Optional[float] = None,
    width: float = 0.92,
) -> None:
    params = inspect.signature(sns.violinplot).parameters
    kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "order": order,
        "ax": ax,
        "cut": cut,
        "inner": inner,
        "linewidth": 1.15,
        "saturation": 1.0,
        "split": split,
    }
    if "bw_adjust" in params:
        kwargs["bw_adjust"] = bw_adjust
    elif "bw" in params:
        kwargs["bw"] = bw_adjust

    if "density_norm" in params:
        kwargs["density_norm"] = density_norm
    elif "scale" in params:
        kwargs["scale"] = density_norm

    if "width" in params:
        kwargs["width"] = width
    if "gap" in params and gap is not None:
        kwargs["gap"] = gap
    if "linecolor" in params:
        kwargs["linecolor"] = "black"

    if hue is not None:
        kwargs["hue"] = hue
        kwargs["hue_order"] = hue_order
        kwargs["palette"] = palette
        kwargs["dodge"] = dodge
    elif color is not None:
        kwargs["color"] = color

    sns.violinplot(**kwargs)


def _compat_stripplot(
    *,
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    order: Sequence[str],
    color: Optional[str] = None,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    palette: Optional[Dict[str, str]] = None,
    dodge: bool = True,
    jitter: float = DEFAULT_STRIP_JITTER,
    size: float = DEFAULT_STRIP_SIZE,
) -> None:
    params = inspect.signature(sns.stripplot).parameters
    kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "order": order,
        "ax": ax,
        "jitter": jitter,
        "size": size,
        "linewidth": 0.0,
        "edgecolor": "black",
        "zorder": 6,
    }
    if hue is not None:
        kwargs["hue"] = hue
        kwargs["hue_order"] = hue_order
        kwargs["palette"] = palette
        kwargs["dodge"] = dodge
        if "legend" in params:
            kwargs["legend"] = False
    elif color is not None:
        kwargs["color"] = color
    sns.stripplot(**kwargs)


def _style_strip_artists(
    ax: plt.Axes,
    *,
    start_collection: int,
    alpha: float,
    edgecolor: str = "black",
    linewidth: float = 0.0,
) -> List[object]:
    collections = list(ax.collections[start_collection:])
    for coll in collections:
        try:
            face = np.array(coll.get_facecolor(), copy=True)
            if face.size:
                face[:, -1] = alpha
                coll.set_facecolor(face)
        except Exception:
            pass
        try:
            coll.set_edgecolor(edgecolor)
        except Exception:
            pass
        try:
            coll.set_linewidth(linewidth)
        except Exception:
            pass
        try:
            coll.set_zorder(6)
        except Exception:
            pass
    return collections



def _draw_constant_group_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    order: Sequence[str],
    color: str = "black",
    linewidth: float = 1.35,
    half_width: float = 0.28,
) -> None:
    if data.empty or x not in data.columns or y not in data.columns:
        return
    for idx, label in enumerate(order):
        vals = pd.to_numeric(data.loc[data[x] == label, y], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        if float(np.nanmax(vals) - np.nanmin(vals)) <= 1e-12:
            yv = float(np.nanmean(vals))
            ax.hlines(yv, idx - half_width, idx + half_width, colors=color, linewidth=linewidth, zorder=7)


def _add_manual_color_legend(
    ax: plt.Axes,
    *,
    entries: Sequence[Tuple[str, str]],
    alpha: float,
    loc: str = "upper right",
    bbox_to_anchor: Optional[Tuple[float, float]] = None,
    borderaxespad: float = 0.35,
    ncol: Optional[int] = None,
) -> None:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    handles = [
        Patch(facecolor=rgba_with_alpha(color, alpha), edgecolor="black", linewidth=1.0, label=label)
        for label, color in entries
    ]
    legend = ax.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        title=None,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        ncol=(min(2, max(1, len(handles))) if ncol is None else int(ncol)),
        borderpad=0.35,
        handletextpad=0.45,
        columnspacing=0.7,
        borderaxespad=borderaxespad,
    )
    if legend is not None:
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_alpha(1.0)
        for txt in legend.get_texts():
            txt.set_fontsize(TICK_FONT_PT)
            txt.set_fontfamily(PLOT_FONT_FAMILY)


def _rebuild_clean_legend(
    ax: plt.Axes,
    *,
    loc: str = "upper right",
    bbox_to_anchor: Optional[Tuple[float, float]] = None,
    borderaxespad: float = 0.35,
    ncol: Optional[int] = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles or not labels:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        return
    seen = set()
    uniq_handles = []
    uniq_labels = []
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    legend = ax.legend(
        uniq_handles,
        uniq_labels,
        title=None,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        ncol=(min(2, max(1, len(uniq_labels))) if ncol is None else int(ncol)),
        borderpad=0.35,
        handletextpad=0.45,
        columnspacing=0.7,
        borderaxespad=borderaxespad,
    )
    if legend is not None:
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_alpha(1.0)
        for txt in legend.get_texts():
            txt.set_fontsize(TICK_FONT_PT)
            txt.set_fontfamily(PLOT_FONT_FAMILY)


def _style_metric_axis(
    ax: plt.Axes,
    *,
    ylabel: str,
    ylim: Optional[Tuple[float, float]],
    show_xlabels: bool,
    algorithm_order_labels: Sequence[str],
    highlight_label: Optional[str],
    xrotation: float = 18.0,
) -> None:
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=TICK_FONT_PT)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    if show_xlabels:
        ax.tick_params(axis="x", labelsize=TICK_FONT_PT, labelrotation=xrotation)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
            if highlight_label and label.get_text() == highlight_label:
                label.set_fontweight("bold")
    else:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlim(-0.5, len(algorithm_order_labels) - 0.5)
    _set_left_ylabel(ax, ylabel)
    _style_axis_box(ax)


def plot_violin_results(
    *,
    df: pd.DataFrame,
    algorithms: Sequence[str],
    display_name_map: Optional[Dict[str, str]],
    output: Path,
    title: Optional[str],
    util_scale: str,
    alpha: float,
    strip_alpha: float,
    strip_size: float,
    strip_jitter: float,
    show_strip: bool,
    inner_style: str,
    violin_cut: float,
    density_norm: str,
    util_layout: str,
    util_gap: float,
    dpi: int,
    highlight_algo: Optional[str],
    violin_bw_adjust: float,
    speedup_mode: str,
    speedup_plot_max: Optional[float],
) -> None:
    if df.empty:
        raise ValueError("plot dataframe is empty")

    speedup_mode = (speedup_mode or DEFAULT_SPEEDUP_MODE).strip().lower()
    util_layout = (util_layout or DEFAULT_UTIL_LAYOUT).strip().lower()

    algorithm_order_labels = [
        pretty_algo_name(algo, display_name_map=display_name_map)
        for algo in algorithms
    ]
    highlight_label = None
    if highlight_algo:
        highlight_label = pretty_algo_name(normalize_algo_token(highlight_algo), display_name_map=display_name_map)
    highlight_idx = None
    if highlight_label and highlight_label in algorithm_order_labels:
        highlight_idx = algorithm_order_labels.index(highlight_label)

    util_suffix = "percent" if util_scale == "percent" else "fraction"
    util_unit = "(%)" if util_scale == "percent" else ""

    ref_name = None
    if "speedup_ref" in df.columns and not df["speedup_ref"].dropna().empty:
        raw_ref = str(df["speedup_ref"].dropna().iloc[0]).strip()
        if raw_ref.lower() not in {"slowest", "worst", "max", "best", "fastest", "min"}:
            ref_name = pretty_algo_name(normalize_algo_token(raw_ref), display_name_map=display_name_map)

    if speedup_mode == "ratio":
        speedup_ylabel = "Speedup (x)"
    else:
        speedup_ylabel = f"Rel. speedup vs {ref_name}" if ref_name else "Rel. speedup"

    fig_w = max(14.8, 1.85 * len(algorithm_order_labels) + 5.0)
    fig_h = 12.4
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.18, 1.0], "hspace": 0.04},
    )
    fig.patch.set_edgecolor("black")
    fig.patch.set_linewidth(1.35)

    speedup_ax, util_ax, coutil_ax = axes

    # 1) Speedup
    speedup_col = "speedup"
    speedup_df = df[["algorithm_label", speedup_col]].dropna().copy()
    speedup_plot_df = speedup_df.copy()
    speedup_removed_n = 0
    if speedup_plot_max is not None and np.isfinite(speedup_plot_max):
        raw_speed = pd.to_numeric(speedup_plot_df[speedup_col], errors="coerce")
        keep_mask = (~raw_speed.isna()) & (raw_speed <= float(speedup_plot_max))
        speedup_removed_n = int((~keep_mask & ~raw_speed.isna()).sum())
        speedup_plot_df = speedup_plot_df.loc[keep_mask].copy()
        if speedup_removed_n > 0:
            print(
                f"[INFO] speedup panel dropped {speedup_removed_n} points with {speedup_col} > {float(speedup_plot_max):g} (plot only)",
                file=sys.stderr,
                flush=True,
            )

    if highlight_idx is not None:
        speedup_ax.axvspan(
            highlight_idx - 0.48,
            highlight_idx + 0.48,
            color=rgba_with_alpha(METRIC_COLORS["speedup"], 0.08),
            zorder=0,
        )

    if speedup_plot_df.empty:
        speedup_ax.text(0.5, 0.5, "No data", transform=speedup_ax.transAxes, ha="center", va="center")
    else:
        c0 = len(speedup_ax.collections)
        l0 = len(speedup_ax.lines)
        _compat_violinplot(
            ax=speedup_ax,
            data=speedup_plot_df,
            x="algorithm_label",
            y=speedup_col,
            order=algorithm_order_labels,
            bw_adjust=violin_bw_adjust,
            color=METRIC_COLORS["speedup"],
            inner=inner_style,
            cut=violin_cut,
            density_norm=density_norm,
            width=0.92,
        )
        _style_violin_artists(speedup_ax, start_collection=c0, start_line=l0, alpha=alpha)
        _draw_constant_group_lines(
            speedup_ax,
            speedup_plot_df,
            x="algorithm_label",
            y=speedup_col,
            order=algorithm_order_labels,
            color="black",
        )
        if show_strip:
            sc0 = len(speedup_ax.collections)
            _compat_stripplot(
                ax=speedup_ax,
                data=speedup_plot_df,
                x="algorithm_label",
                y=speedup_col,
                order=algorithm_order_labels,
                color="black",
                jitter=strip_jitter,
                size=strip_size,
            )
            _style_strip_artists(speedup_ax, start_collection=sc0, alpha=strip_alpha, linewidth=0.0)
        if speedup_mode != "ratio":
            speedup_ax.axhline(0.0, color="black", linewidth=0.95, alpha=0.72, zorder=1)

    speedup_ylim = _value_ylim(
        speedup_plot_df[speedup_col].astype(float).tolist(),
        include_zero=True,
    ) if not speedup_plot_df.empty else (-0.5, 0.5)
    if speedup_plot_max is not None and np.isfinite(speedup_plot_max):
        lo, hi = speedup_ylim
        hi = min(float(speedup_plot_max), hi)
        if hi <= lo:
            hi = lo + max(0.5, abs(lo) * 0.1 + 0.1)
        speedup_ylim = (lo, hi)

    _style_metric_axis(
        speedup_ax,
        ylabel=speedup_ylabel,
        ylim=speedup_ylim,
        show_xlabels=False,
        algorithm_order_labels=algorithm_order_labels,
        highlight_label=highlight_label,
    )

    # 2) Combined PIM + NPU utilization
    pim_col = f"pim_utilization_{util_suffix}"
    accel_col = f"accel_utilization_{util_suffix}"
    pim_df = df[["algorithm_label", pim_col]].rename(columns={pim_col: "util_value"}).dropna().copy()
    accel_df = df[["algorithm_label", accel_col]].rename(columns={accel_col: "util_value"}).dropna().copy()
    util_long = pd.concat([
        pim_df.assign(util_family="PIM"),
        accel_df.assign(util_family="NPU"),
    ], ignore_index=True).dropna()

    if util_long.empty:
        util_ax.text(0.5, 0.5, "No data", transform=util_ax.transAxes, ha="center", va="center")
    else:
        if util_layout == "overlay":
            overlay_alpha = min(alpha, 0.40)
            for fam_label, fam_df, fam_color in [
                ("NPU", accel_df, METRIC_COLORS["accel_utilization"]),
                ("PIM", pim_df, METRIC_COLORS["pim_utilization"]),
            ]:
                if fam_df.empty:
                    continue
                c0 = len(util_ax.collections)
                l0 = len(util_ax.lines)
                _compat_violinplot(
                    ax=util_ax,
                    data=fam_df,
                    x="algorithm_label",
                    y="util_value",
                    order=algorithm_order_labels,
                    bw_adjust=violin_bw_adjust,
                    color=fam_color,
                    inner=inner_style,
                    cut=violin_cut,
                    density_norm=density_norm,
                    width=0.92,
                )
                _style_violin_artists(util_ax, start_collection=c0, start_line=l0, alpha=overlay_alpha)
                _draw_constant_group_lines(
                    util_ax,
                    fam_df,
                    x="algorithm_label",
                    y="util_value",
                    order=algorithm_order_labels,
                    color="black",
                )

            if show_strip:
                for fam_df in [pim_df, accel_df]:
                    if fam_df.empty:
                        continue
                    sc0 = len(util_ax.collections)
                    _compat_stripplot(
                        ax=util_ax,
                        data=fam_df,
                        x="algorithm_label",
                        y="util_value",
                        order=algorithm_order_labels,
                        color="black",
                        jitter=strip_jitter,
                        size=strip_size,
                    )
                    _style_strip_artists(util_ax, start_collection=sc0, alpha=strip_alpha, linewidth=0.0)

            _add_manual_color_legend(
                util_ax,
                entries=[
                    ("PIM", METRIC_COLORS["pim_utilization"]),
                    ("NPU", METRIC_COLORS["accel_utilization"]),
                ],
                alpha=min(alpha, 0.40),
                loc="upper left",
                bbox_to_anchor=(1.005, 1.0),
                borderaxespad=0.0,
                ncol=1,
            )
        else:
            c0 = len(util_ax.collections)
            l0 = len(util_ax.lines)
            is_split = util_layout == "split"
            is_dodge = util_layout == "dodge"
            _compat_violinplot(
                ax=util_ax,
                data=util_long,
                x="algorithm_label",
                y="util_value",
                order=algorithm_order_labels,
                bw_adjust=violin_bw_adjust,
                hue="util_family",
                hue_order=["PIM", "NPU"],
                palette={
                    "PIM": METRIC_COLORS["pim_utilization"],
                    "NPU": METRIC_COLORS["accel_utilization"],
                },
                dodge=is_dodge,
                inner=inner_style,
                cut=violin_cut,
                density_norm=density_norm,
                split=is_split,
                gap=util_gap if is_dodge else None,
                width=0.92,
            )
            _style_violin_artists(util_ax, start_collection=c0, start_line=l0, alpha=alpha)
            if show_strip:
                sc0 = len(util_ax.collections)
                _compat_stripplot(
                    ax=util_ax,
                    data=util_long,
                    x="algorithm_label",
                    y="util_value",
                    order=algorithm_order_labels,
                    hue="util_family",
                    hue_order=["PIM", "NPU"],
                    palette={"PIM": "black", "NPU": "black"},
                    dodge=is_dodge,
                    jitter=strip_jitter,
                    size=strip_size,
                )
                _style_strip_artists(util_ax, start_collection=sc0, alpha=strip_alpha, linewidth=0.0)
            _rebuild_clean_legend(
                util_ax,
                loc="upper left",
                bbox_to_anchor=(1.005, 1.0),
                borderaxespad=0.0,
                ncol=1,
            )

    _style_metric_axis(
        util_ax,
        ylabel=f"Util. {util_unit}".strip(),
        ylim=_util_ylim(util_long["util_value"].astype(float).tolist(), util_scale=util_scale) if not util_long.empty else ((0.0, 100.0) if util_scale == "percent" else (0.0, 1.0)),
        show_xlabels=False,
        algorithm_order_labels=algorithm_order_labels,
        highlight_label=highlight_label,
    )

    # 3) Co-utilization, with extra emphasis for the highlighted method
    coutil_col = f"concurrent_utilization_{util_suffix}"
    coutil_df = df[["algorithm_label", coutil_col]].dropna().copy()
    if highlight_idx is not None:
        coutil_ax.axvspan(
            highlight_idx - 0.48,
            highlight_idx + 0.48,
            color=rgba_with_alpha(METRIC_COLORS["concurrent_utilization"], 0.08),
            zorder=0,
        )

    if coutil_df.empty:
        coutil_ax.text(0.5, 0.5, "No data", transform=coutil_ax.transAxes, ha="center", va="center")
    else:
        c0 = len(coutil_ax.collections)
        l0 = len(coutil_ax.lines)
        _compat_violinplot(
            ax=coutil_ax,
            data=coutil_df,
            x="algorithm_label",
            y=coutil_col,
            order=algorithm_order_labels,
            bw_adjust=violin_bw_adjust,
            color=METRIC_COLORS["concurrent_utilization"],
            inner=inner_style,
            cut=violin_cut,
            density_norm=density_norm,
            width=0.92,
        )
        collections = _style_violin_artists(coutil_ax, start_collection=c0, start_line=l0, alpha=max(0.42, alpha - 0.06))
        _draw_constant_group_lines(
            coutil_ax,
            coutil_df,
            x="algorithm_label",
            y=coutil_col,
            order=algorithm_order_labels,
            color="black",
        )
        if show_strip:
            sc0 = len(coutil_ax.collections)
            _compat_stripplot(
                ax=coutil_ax,
                data=coutil_df,
                x="algorithm_label",
                y=coutil_col,
                order=algorithm_order_labels,
                color="black",
                jitter=strip_jitter,
                size=strip_size,
            )
            _style_strip_artists(coutil_ax, start_collection=sc0, alpha=strip_alpha, linewidth=0.0)
        if highlight_idx is not None and 0 <= highlight_idx < len(collections):
            _emphasize_violin(collections[highlight_idx], alpha=min(0.92, alpha + 0.20), linewidth=2.15)
            median_series = coutil_df.groupby("algorithm_label")[coutil_col].median()
            if highlight_label in median_series.index:
                coutil_ax.scatter(
                    [highlight_idx],
                    [float(median_series.loc[highlight_label])],
                    s=54,
                    facecolors="black",
                    edgecolors="black",
                    linewidths=0.0,
                    zorder=7,
                )

    _style_metric_axis(
        coutil_ax,
        ylabel=f"Co-util. {util_unit}".strip(),
        ylim=_util_ylim(coutil_df[coutil_col].astype(float).tolist(), util_scale=util_scale) if not coutil_df.empty else ((0.0, 100.0) if util_scale == "percent" else (0.0, 1.0)),
        show_xlabels=True,
        algorithm_order_labels=algorithm_order_labels,
        highlight_label=highlight_label,
        xrotation=18.0,
    )

    top_margin = 0.965 if title else 0.985
    fig.subplots_adjust(left=0.115, right=0.87, top=top_margin, bottom=0.12)
    if title:
        fig.suptitle(title, y=0.99, fontsize=COMMON_UI_FONT_PT + 1)

    enforce_figure_fonts(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    print(f"[OK] saved figure -> {output.resolve()}")


# ---------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------

def save_summary_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] saved summary -> {output_path.resolve()}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--search-dir",
        type=str,
        required=True,
        help=(
            "Root directory that contains batch folders (e.g. sst8_rst8) "
            "or a direct batch-size folder (e.g. llama_7b_fp16_b16_s8)."
        ),
    )
    ap.add_argument(
        "--model-prefix",
        type=str,
        default=None,
        help=(
            "Optional model prefix used to collect all batch-size folders under --search-dir, "
            "e.g. llama_7b_fp16 or qwen_1.8b_fp16. "
            "When omitted, everything under --search-dir is considered."
        ),
    )
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
                    help="Override x-axis display names with key=value pairs separated by ';'")
    ap.add_argument("--highlight-algo", type=str, default="hefthint",
                    help="Highlight this algorithm label on the x-axis. Default: hefthint")
    ap.add_argument("--output", type=str, required=True, help="Output figure path (.pdf/.png/.svg ...)")
    ap.add_argument("--summary-csv", type=str, default=None,
                    help="Optional CSV dump of all violin-source rows.")
    ap.add_argument("--title", type=str, default=None)
    ap.add_argument("--phase", type=str, choices=["all", "prefill", "decode"], default="all",
                    help="Use all trace events, or only one phase.")
    ap.add_argument("--include-cpu", action="store_true",
                    help="Also include CPU when computing overall-utilization-based statistics. CPU is not plotted.")
    ap.add_argument("--heft-pick-by", type=str,
                    choices=["family_mean", "overall", "pim", "accel"],
                    default="family_mean",
                    help="How to choose between heft and hefthint for each case.")
    ap.add_argument("--util-scale", type=str, choices=["percent", "fraction"], default="percent",
                    help="Display utilization as 0~100 percent or 0~1 fraction.")
    ap.add_argument("--speedup-ref", type=str, default=DEFAULT_SPEEDUP_REF,
                    help="Speedup reference: an algorithm name, or one of slowest / best. Default: pd")
    ap.add_argument("--speedup-time-field", type=str,
                    choices=["auto", "total_time_s", "prefill_time_s", "decode_time_s"],
                    default="auto",
                    help="Which field from baseline_compare_*.json to use for speedup. auto follows --phase.")
    ap.add_argument("--speedup-mode", type=str, choices=["ratio", "improvement"], default=DEFAULT_SPEEDUP_MODE,
                    help="ratio: ref/algo, so the reference is 1. improvement: ref/algo - 1, so the reference is 0. Default: ratio")
    ap.add_argument("--speedup-plot-max", type=float, default=DEFAULT_SPEEDUP_PLOT_MAX,
                    help="Optional display-only clamp for the speedup panel; points with speedup above this value are dropped from the violin/strip plot. Use a negative value to disable. Default: 5")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Alpha for violin fill color. Default: 0.58")
    ap.add_argument("--show-strip", action="store_true", default=True,
                    help="Overlay jittered raw points with seaborn.stripplot. Default: on")
    ap.add_argument("--no-strip", action="store_false", dest="show_strip",
                    help="Disable stripplot point overlay.")
    ap.add_argument("--strip-alpha", type=float, default=DEFAULT_STRIP_ALPHA,
                    help="Alpha for stripplot points. Default: 0.42")
    ap.add_argument("--strip-size", type=float, default=DEFAULT_STRIP_SIZE,
                    help="Marker size for stripplot points. Default: 2.6")
    ap.add_argument("--strip-jitter", type=float, default=DEFAULT_STRIP_JITTER,
                    help="Jitter amount for stripplot points. Default: 0.18")
    ap.add_argument("--inner-style", type=str, choices=["box", "quart", "point", "stick"], default=DEFAULT_INNER_STYLE,
                    help="Interior representation for violins. Default: quart")
    ap.add_argument("--violin-cut", type=float, default=DEFAULT_VIOLIN_CUT,
                    help="How far the KDE extends beyond observed data. Default: 0")
    ap.add_argument("--density-norm", type=str, choices=["area", "count", "width"], default=DEFAULT_DENSITY_NORM,
                    help="Width normalization mode for violins. Default: count")
    ap.add_argument("--util-layout", type=str, choices=["overlay", "split", "dodge"], default=DEFAULT_UTIL_LAYOUT,
                    help="Layout for the combined PIM/NPU panel. overlay: centered full violins; split: left/right halves; dodge: side-by-side. Default: overlay")
    ap.add_argument("--util-split", dest="util_layout", action="store_const", const="split",
                    help="Compatibility alias: use split violin for the combined PIM/NPU panel.")
    ap.add_argument("--no-util-split", dest="util_layout", action="store_const", const="overlay",
                    help="Compatibility alias: use centered overlaid violins for the combined PIM/NPU panel.")
    ap.add_argument("--util-gap", type=float, default=DEFAULT_UTIL_GAP,
                    help="Gap between dodged PIM/NPU violins when util-layout=dodge. Default: 0.03")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--font-family", type=str, default=PLOT_FONT_FAMILY,
                    help="Preferred plotting font family. Default: Arial")
    ap.add_argument("--font-size", type=float, default=COMMON_UI_FONT_PT,
                    help="Base UI font size.")
    ap.add_argument("--bw-adjust", type=float, default=0.50,
                    help="Seaborn violinplot bw_adjust. Default: 0.50")
    ap.add_argument("--cache-dir", type=str, default=None,
                    help="Directory for on-disk utilization cache. Default: <search-dir>/.plot_cache_violin")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable on-disk cache.")

    args = ap.parse_args()

    apply_global_plot_style(font_family=args.font_family, font_size=args.font_size)

    search_dir = Path(args.search_dir)
    if not search_dir.exists():
        ap.error(f"--search-dir not found: {search_dir}")

    trace_paths = discover_trace_files(search_dir)
    if not trace_paths:
        ap.error(f"No *_trace.csv found under: {search_dir}")

    # First discover all run infos from traces, then filter by model-prefix if requested.
    all_run_infos: Dict[str, RunInfo] = {}
    for path in trace_paths:
        info = infer_run_info(path)
        all_run_infos[info.run_id] = info

    filtered_run_infos = filter_run_infos(all_run_infos, model_prefix=args.model_prefix)
    if not filtered_run_infos:
        ap.error(
            f"No matching run directories found under {search_dir} for model-prefix={args.model_prefix!r}"
        )

    allowed_run_ids = sorted(filtered_run_infos.keys())
    index = build_trace_index(trace_paths, allowed_run_ids=allowed_run_ids)
    if not index:
        ap.error("No valid comms/ops trace pairs found after filtering. Check file names and directory.")

    baseline_paths = discover_baseline_compare_files(search_dir)
    time_index = build_time_index(baseline_paths, allowed_run_ids=allowed_run_ids)
    if not baseline_paths:
        print(f"[WARN] no baseline_compare_*.json found under: {search_dir}", file=sys.stderr)
    elif not time_index:
        print(f"[WARN] baseline_compare_*.json found but no valid timings were parsed under: {search_dir}", file=sys.stderr)

    all_prefills = sorted({key.prefill for key in index})
    all_decodes = sorted({key.decode for key in index})
    all_algos_norm = sorted({normalize_algo_token(key.algo) for key in index})

    print(
        f"[INFO] matched {len(filtered_run_infos)} run dirs | {len(index)} trace pairs | "
        f"prefills={len(all_prefills)} | decodes={len(all_decodes)} | algos={len(all_algos_norm)}",
        file=sys.stderr,
        flush=True,
    )

    try:
        prefills = parse_int_list(args.prefills) or all_prefills
        decodes = parse_int_list(args.decodes) or all_decodes
    except ValueError as exc:
        ap.error(str(exc))

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
        disk_cache_dir = Path(args.cache_dir) if args.cache_dir else (search_dir / ".plot_cache_violin")
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
        if any(k[3] == algo for k in results.keys()):
            active_algos.append(algo)

    if not active_algos:
        ap.error("No active algorithms remain after reading usable results.")

    time_field = args.speedup_time_field
    if time_field == "auto":
        time_field = _default_time_field_for_phase(args.phase)

    plot_df = build_plot_dataframe(
        results=results,
        time_index=time_index,
        run_infos=filtered_run_infos,
        algorithms=active_algos,
        display_name_map=display_name_map,
        time_field=time_field,
        speedup_ref=args.speedup_ref,
        speedup_mode=args.speedup_mode,
    )
    if plot_df.empty:
        ap.error("No rows available for violin plotting.")

    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(args.output).with_suffix("").with_name(Path(args.output).stem + "_violin_source.csv")
    save_summary_csv(plot_df, summary_csv)

    case_n = int(plot_df["case_id"].nunique()) if "case_id" in plot_df.columns else 0
    print(f"[INFO] assembled {len(plot_df)} violin rows from {case_n} unique cases", file=sys.stderr, flush=True)
    title = args.title

    plot_violin_results(
        df=plot_df,
        algorithms=active_algos,
        display_name_map=display_name_map,
        output=Path(args.output),
        title=title,
        util_scale=args.util_scale,
        alpha=args.alpha,
        strip_alpha=args.strip_alpha,
        strip_size=args.strip_size,
        strip_jitter=args.strip_jitter,
        show_strip=args.show_strip,
        inner_style=args.inner_style,
        violin_cut=args.violin_cut,
        density_norm=args.density_norm,
        util_layout=args.util_layout,
        util_gap=args.util_gap,
        dpi=args.dpi,
        highlight_algo=args.highlight_algo,
        violin_bw_adjust=args.bw_adjust,
        speedup_mode=args.speedup_mode,
        speedup_plot_max=(None if args.speedup_plot_max is None or args.speedup_plot_max < 0 else args.speedup_plot_max),
    )


if __name__ == "__main__":
    main()
