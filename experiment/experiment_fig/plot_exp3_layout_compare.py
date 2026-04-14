#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare layout-search variants using summary tables and trace files.

Run this script from ``experiment/experiment_fig``. The CSV input should point
to a table derived from the current weight-suggest workflow, and any trace-file
lookups are resolved relative to the repository-level ``output/`` tree.

Example
-------
python plot_exp3_layout_compare.py \
  --csv ../../output/ws_high_npu_bw/results.csv \
  --outdir ../../figs/exp3/compare \
  --metrics total ttft \
  --fig-format png pdf \
  --share-y
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


# Keep the same font fallback behavior as the current script.
_available_font_names = {f.name for f in font_manager.fontManager.ttflist}
if "Arial" in _available_font_names:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
else:
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


@dataclass(frozen=True)
class BarSpec:
    value_col: str
    label: str
    color: str
    trace_role: str


@dataclass(frozen=True)
class TraceCandidate:
    path: Path
    source: str
    context: Mapping[str, Any]


@dataclass(frozen=True)
class TraceResolution:
    path: Path | None
    source: str
    warning: str | None = None


@dataclass(frozen=True)
class TraceStats:
    path: Path
    raw_total_s: float
    raw_ttft_s: float
    raw_ttft_ratio: float
    first_decode_cycle_rows: int
    inferred_decode_steps: int | None


# Five fixed bars, in the requested order.
# Last column uses the Bifocal dual best result.
BAR_SPECS: List[BarSpec] = [
    BarSpec("PD+Linear_initial", "PD\nLinear", "#bdade4", "pd_linear"),
    BarSpec("PD+Dual_initial", "PD\nDual", "#e4adb5", "pd_dual"),
    BarSpec("Bifocal+Linear_initial", "Bifocal\nLinear\nInit", "#aee4ad", "bifocal_linear_initial"),
    BarSpec("Bifocal+Linear_best", "Bifocal\nLinear\nBest", "#add9e4", "bifocal_linear_best"),
    BarSpec("Bifocal+Dual_best", "Bifocal\nDual\nBest", "#e4ddad", "bifocal_dual"),
]

REQUIRED_COLUMNS = [
    "model",
    "batch",
    "prefill_len",
    "decode_len",
    *(spec.value_col for spec in BAR_SPECS),
]

RUN_PATH_COLUMNS = [
    "generated_config_json",
    "best_summary_json",
    "all_passes_json",
    "weight_format_json",
    "weight_format_compare_json",
    "log_path",
]

JSON_HINT_COLUMNS = [
    "best_summary_json",
    "all_passes_json",
    "generated_config_json",
    "weight_format_json",
    "weight_format_compare_json",
]

# Visually thinner bars, but bars still touch each other.
BAR_WIDTH = 0.46
BAR_STEP = BAR_WIDTH  # no gap between neighboring bars
SIDE_PADDING = 0.92


CACHE_SCHEMA_VERSION = 2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProgressLogger:
    def __init__(self, *, verbose: bool = False) -> None:
        self.verbose = bool(verbose)
        self._t0 = time.perf_counter()

    def _prefix(self) -> str:
        elapsed = time.perf_counter() - self._t0
        return f"[+{elapsed:8.1f}s]"

    def log(self, message: str) -> None:
        print(f"{self._prefix()} {message}", flush=True)

    def verbose_log(self, message: str) -> None:
        if self.verbose:
            self.log(message)


class TTFTPersistentCache:
    def __init__(
        self,
        path: Path | None,
        *,
        logger: ProgressLogger,
        rebuild: bool = False,
    ) -> None:
        self.path = path
        self.logger = logger
        self.rebuild = bool(rebuild)
        self._dirty = False
        self.data: Dict[str, Any] = self._empty_data()
        self._trace_stats_hits = 0
        self._trace_stats_misses = 0
        self._trace_stats_invalid = 0
        self._trace_stats_stores = 0
        self._resolution_hits = 0
        self._resolution_misses = 0
        self._resolution_stores = 0
        self._load()

    def _empty_data(self) -> Dict[str, Any]:
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "updated_at_utc": utc_now_iso(),
            "trace_stats": {},
            "trace_resolutions": {},
        }

    def _load(self) -> None:
        if self.path is None:
            self.logger.log("TTFT persistent cache: disabled")
            return

        if self.rebuild:
            self.logger.log(f"TTFT persistent cache: rebuild requested, ignoring existing file: {self.path}")
            return

        if not self.path.exists():
            self.logger.log(f"TTFT persistent cache: no existing cache file, will create: {self.path}")
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self.logger.log(f"TTFT persistent cache: failed to load {self.path} ({exc}), starting empty")
            return

        if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            self.logger.log(
                f"TTFT persistent cache: schema mismatch in {self.path}, expected version {CACHE_SCHEMA_VERSION}; starting empty"
            )
            return

        self.data = payload
        n_trace_stats = len(self.data.get("trace_stats", {}))
        n_resolutions = len(self.data.get("trace_resolutions", {}))
        self.logger.log(
            f"TTFT persistent cache: loaded {n_trace_stats} trace stats and {n_resolutions} trace resolutions from {self.path}"
        )

    def _trace_fp(self, path: Path) -> Tuple[int, int] | None:
        try:
            st = path.stat()
        except Exception:
            return None
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        return int(st.st_size), mtime_ns

    def get_trace_stats(self, path: Path) -> TraceStats | None:
        if self.path is None:
            self._trace_stats_misses += 1
            return None

        key = os.path.normpath(str(path))
        entry = self.data.get("trace_stats", {}).get(key)
        if not isinstance(entry, dict):
            self._trace_stats_misses += 1
            return None

        fp = self._trace_fp(path)
        if fp is None:
            self._trace_stats_invalid += 1
            return None

        size, mtime_ns = fp
        if int(entry.get("size", -1)) != size or int(entry.get("mtime_ns", -1)) != mtime_ns:
            self._trace_stats_invalid += 1
            return None

        try:
            stats = TraceStats(
                path=Path(entry.get("path", key)),
                raw_total_s=float(entry["raw_total_s"]),
                raw_ttft_s=float(entry["raw_ttft_s"]),
                raw_ttft_ratio=float(entry["raw_ttft_ratio"]),
                first_decode_cycle_rows=int(entry["first_decode_cycle_rows"]),
                inferred_decode_steps=(
                    None if entry.get("inferred_decode_steps") is None else int(entry.get("inferred_decode_steps"))
                ),
            )
        except Exception:
            self._trace_stats_invalid += 1
            return None

        self._trace_stats_hits += 1
        return stats

    def put_trace_stats(self, stats: TraceStats) -> None:
        if self.path is None:
            return
        fp = self._trace_fp(stats.path)
        if fp is None:
            return
        size, mtime_ns = fp
        key = os.path.normpath(str(stats.path))
        self.data.setdefault("trace_stats", {})[key] = {
            "path": str(stats.path),
            "size": size,
            "mtime_ns": mtime_ns,
            "raw_total_s": float(stats.raw_total_s),
            "raw_ttft_s": float(stats.raw_ttft_s),
            "raw_ttft_ratio": float(stats.raw_ttft_ratio),
            "first_decode_cycle_rows": int(stats.first_decode_cycle_rows),
            "inferred_decode_steps": (
                None if stats.inferred_decode_steps is None else int(stats.inferred_decode_steps)
            ),
            "cached_at_utc": utc_now_iso(),
        }
        self._dirty = True
        self._trace_stats_stores += 1

    def get_trace_resolution(self, key: str) -> TraceResolution | None:
        if self.path is None:
            self._resolution_misses += 1
            return None
        entry = self.data.get("trace_resolutions", {}).get(key)
        if not isinstance(entry, dict):
            self._resolution_misses += 1
            return None
        path_str = entry.get("path")
        path_obj = None
        if path_str:
            path_obj = Path(path_str)
            if not path_obj.exists():
                self._resolution_misses += 1
                return None
        self._resolution_hits += 1
        return TraceResolution(
            path=path_obj,
            source=str(entry.get("source", "cache")),
            warning=(None if not entry.get("warning") else str(entry.get("warning"))),
        )

    def put_trace_resolution(self, key: str, resolution: TraceResolution) -> None:
        if self.path is None:
            return
        self.data.setdefault("trace_resolutions", {})[key] = {
            "path": None if resolution.path is None else str(resolution.path),
            "source": str(resolution.source),
            "warning": resolution.warning,
            "cached_at_utc": utc_now_iso(),
        }
        self._dirty = True
        self._resolution_stores += 1

    def save(self) -> None:
        if self.path is None:
            return
        if not self._dirty and self.path.exists():
            self.logger.log(self.summary(prefix="TTFT persistent cache: no changes; "))
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at_utc"] = utc_now_iso()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, self.path)
        self.logger.log(f"TTFT persistent cache: saved to {self.path}")
        self.logger.log(self.summary(prefix="TTFT persistent cache: "))

    def summary(self, prefix: str = "") -> str:
        n_trace_stats = len(self.data.get("trace_stats", {}))
        n_resolutions = len(self.data.get("trace_resolutions", {}))
        return (
            f"{prefix}trace_stats={n_trace_stats} (hits={self._trace_stats_hits}, misses={self._trace_stats_misses}, invalid={self._trace_stats_invalid}, stores={self._trace_stats_stores}); "
            f"trace_resolutions={n_resolutions} (hits={self._resolution_hits}, misses={self._resolution_misses}, stores={self._resolution_stores})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-(model, batch) multi-panel figures from results.csv. "
            "Each panel is one (prefill_len, decode_len) pair, and each panel "
            "contains five touching bars: PD linear, PD dual, ND init, ND best, "
            "and Bifocal dual best. TTFT is inferred from *_ops_trace.csv."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to results.csv")
    parser.add_argument("--outdir", default="plots_layout_compare", help="Output directory")
    parser.add_argument(
        "--fig-format",
        nargs="+",
        default=["png"],
        help="Output format(s), e.g. png pdf",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    parser.add_argument(
        "--ncols",
        default="auto",
        help="Number of columns per figure. Use auto or a positive integer.",
    )
    parser.add_argument(
        "--share-y",
        action="store_true",
        help="Share the y-axis range across all panels in the same figure.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["total", "ttft"],
        choices=["total", "ttft"],
        help="Which latency figures to draw. Default: total ttft",
    )
    parser.add_argument(
        "--extra-search-root",
        nargs="*",
        default=[],
        help=(
            "Extra directories to search recursively for *_ops_trace.csv. "
            "Useful when the absolute paths recorded in results.csv need to be remapped."
        ),
    )
    parser.add_argument(
        "--path-remap",
        nargs="*",
        default=[],
        help=(
            "Prefix remap in SRC=DST form. Example: "
            "/lustre/home/user/project=/data/project_copy"
        ),
    )
    parser.add_argument(
        "--save-augmented-csv",
        default=None,
        help=(
            "Optional path to save an augmented CSV with TTFT columns, ratios, "
            "trace paths, and resolution sources."
        ),
    )
    parser.add_argument(
        "--strict-ttft",
        action="store_true",
        help="Fail immediately if a TTFT trace cannot be resolved or parsed.",
    )
    parser.add_argument(
        "--ttft-cache",
        default=None,
        help=(
            "Persistent TTFT cache JSON path. Default: next to results.csv as "
            "<results_stem>__ttft_cache.json"
        ),
    )
    parser.add_argument(
        "--rebuild-ttft-cache",
        action="store_true",
        help="Ignore any existing TTFT cache file and rebuild it from scratch.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print TTFT row progress every N rows. Default: 1",
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Print extra progress information for cache hits and directory scanning.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def format_length_value(value: object) -> str:
    try:
        value_f = float(value)
        if np.isfinite(value_f) and value_f.is_integer():
            return str(int(value_f))
        if np.isfinite(value_f):
            return f"{value_f:g}"
    except Exception:
        pass
    return str(value)


def format_seconds(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    value = float(value)
    if abs(value) < 0.1:
        return f"{value:.4f}s"
    if abs(value) < 1.0:
        return f"{value:.3f}s"
    return f"{value:.2f}s"


def format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.1f}%"


def describe_row_for_progress(row: pd.Series) -> str:
    parts: List[str] = []
    for key in ["model", "dtype"]:
        if key in row.index:
            value = row.get(key)
            if pd.notna(value) and str(value).lower() != "nan":
                parts.append(f"{key}={value}")
    if "batch" in row.index and pd.notna(row.get("batch")):
        parts.append(f"batch={format_length_value(row.get('batch'))}")
    if "prefill_len" in row.index and pd.notna(row.get("prefill_len")):
        parts.append(f"prefill={format_length_value(row.get('prefill_len'))}")
    if "decode_len" in row.index and pd.notna(row.get("decode_len")):
        parts.append(f"decode={format_length_value(row.get('decode_len'))}")
    return " | ".join(parts) if parts else "<row>"


def normalize_scalar_for_key(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        if not np.isfinite(value_f):
            return None
        if value_f.is_integer():
            return int(value_f)
        return value_f
    return str(value)


def make_resolution_cache_key(
    row: pd.Series,
    *,
    role: str,
    remaps: Sequence[Tuple[str, str]],
    extra_search_roots: Sequence[Path],
) -> str:
    payload: Dict[str, Any] = {
        "role": role,
        "model": normalize_scalar_for_key(row.get("model")),
        "dtype": normalize_scalar_for_key(row.get("dtype")),
        "batch": normalize_scalar_for_key(row.get("batch")),
        "prefill_len": normalize_scalar_for_key(row.get("prefill_len")),
        "decode_len": normalize_scalar_for_key(row.get("decode_len")),
        "best_pass": normalize_scalar_for_key(row.get("best_pass")),
        "run_path_columns": {
            col: normalize_scalar_for_key(row.get(col))
            for col in RUN_PATH_COLUMNS
            if col in row.index
        },
        "path_remaps": [[str(src), str(dst)] for src, dst in remaps],
        "extra_search_roots": [str(Path(p)) for p in extra_search_roots],
    }
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()


def auto_ncols(n_panels: int) -> int:
    if n_panels <= 3:
        return n_panels
    if n_panels <= 6:
        return 3
    if n_panels <= 8:
        return 4
    return 5


def resolve_ncols(arg_value: str, n_panels: int) -> int:
    if str(arg_value).strip().lower() == "auto":
        return auto_ncols(n_panels)
    ncols = int(arg_value)
    if ncols <= 0:
        raise ValueError("--ncols must be auto or a positive integer")
    return ncols


def finite_max(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.max())


def compute_ylim(max_value: float) -> Tuple[float, float]:
    if not np.isfinite(max_value) or max_value <= 0:
        return (0.0, 1.0)
    return (0.0, max_value * 1.24)


def compute_bifocal_linear_best_reduction_pct(initial_value: float, best_value: float) -> float:
    if not np.isfinite(initial_value) or initial_value <= 0 or not np.isfinite(best_value):
        return float("nan")
    return (initial_value - best_value) / initial_value * 100.0


def add_value_labels(ax: plt.Axes, bars, values: Sequence[float], fontsize: float = 8.1) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.018
    for bar, value in zip(bars, values):
        value_f = float(value)
        y = max(0.0, value_f if np.isfinite(value_f) else 0.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + offset,
            format_seconds(value_f),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="0.15",
        )


def parse_path_remaps(raw_items: Sequence[str]) -> List[Tuple[str, str]]:
    remaps: List[Tuple[str, str]] = []
    for item in raw_items:
        if "=" not in str(item):
            raise ValueError(f"Invalid --path-remap entry (expected SRC=DST): {item}")
        src, dst = str(item).split("=", 1)
        src = src.strip().rstrip("/\\")
        dst = dst.strip().rstrip("/\\")
        if not src or not dst:
            raise ValueError(f"Invalid --path-remap entry (empty SRC or DST): {item}")
        remaps.append((src, dst))
    return remaps


def apply_path_remaps(path_str: str, remaps: Sequence[Tuple[str, str]]) -> Path:
    text = str(path_str).strip()
    if not text:
        return Path(text)
    norm_text = text.replace("\\", "/")
    for src, dst in remaps:
        norm_src = src.replace("\\", "/")
        if norm_text == norm_src or norm_text.startswith(norm_src + "/"):
            suffix = norm_text[len(norm_src) :]
            return Path(dst + suffix)
    return Path(text)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = os.path.normpath(str(p))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def materialize_path(
    raw_path: object,
    base_dirs: Sequence[Path] | None,
    remaps: Sequence[Tuple[str, str]],
) -> Path | None:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text or text.lower() == "nan":
        return None

    p = Path(text).expanduser()
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)
        candidates.append(apply_path_remaps(text, remaps))
    else:
        bases = list(base_dirs or [Path.cwd()])
        if not bases:
            bases = [Path.cwd()]
        for base in bases:
            base = Path(base).expanduser()
            candidates.append(base / p)
            candidates.append(apply_path_remaps(str(base / p), remaps))
        candidates.append(apply_path_remaps(text, remaps))

    candidates = unique_paths(candidates)
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else None


def get_run_dir_from_row(row: pd.Series, remaps: Sequence[Tuple[str, str]]) -> Path | None:
    for col in RUN_PATH_COLUMNS:
        if col not in row.index:
            continue
        path = materialize_path(row[col], base_dirs=[Path.cwd()], remaps=remaps)
        if path is None:
            continue
        if path.suffix:
            return path.parent
        return path
    return None


def build_pair_tag(row: pd.Series) -> str:
    return (
        f"prefill-{format_length_value(row['prefill_len'])}"
        f"xdecode_{format_length_value(row['decode_len'])}"
    )


def scan_ops_trace_inventory(
    root: Path,
    *,
    inventory_cache: MutableMapping[str, List[Path]],
    progress: ProgressLogger | None,
) -> List[Path]:
    root_key = os.path.normpath(str(root))
    cached = inventory_cache.get(root_key)
    if cached is not None:
        if progress is not None:
            progress.verbose_log(f"[scan] reuse cached root inventory: {root} ({len(cached)} ops traces)")
        return cached

    if progress is not None:
        progress.log(f"[scan] indexing ops traces under root: {root}")

    paths: List[Path] = []
    dir_count = 0
    t0 = time.perf_counter()

    def _onerror(exc: OSError) -> None:
        if progress is not None:
            progress.verbose_log(f"[scan] warning while walking {root}: {exc}")

    for dirpath, _, filenames in os.walk(root, onerror=_onerror):
        dir_count += 1
        if progress is not None and progress.verbose and (dir_count == 1 or dir_count % 200 == 0):
            elapsed = time.perf_counter() - t0
            progress.verbose_log(
                f"[scan] root={root} dirs={dir_count} matched={len(paths)} elapsed={elapsed:.1f}s"
            )
        for filename in filenames:
            if not str(filename).endswith("_ops_trace.csv"):
                continue
            path = Path(dirpath) / filename
            if path.is_file():
                paths.append(path)

    inventory_cache[root_key] = paths
    if progress is not None:
        elapsed = time.perf_counter() - t0
        progress.log(
            f"[scan] finished root={root} dirs={dir_count} ops_traces={len(paths)} elapsed={elapsed:.1f}s"
        )
    return paths


def discover_fs_trace_candidates(
    row: pd.Series,
    run_dir: Path | None,
    extra_search_roots: Sequence[Path],
    *,
    progress: ProgressLogger | None = None,
    inventory_cache: MutableMapping[str, List[Path]] | None = None,
    match_cache: MutableMapping[Tuple[str, str], List[Path]] | None = None,
) -> List[TraceCandidate]:
    pair_tag = build_pair_tag(row)
    pair_tag_low = pair_tag.lower()

    roots: List[Path] = []
    if run_dir is not None:
        roots.extend([run_dir / "artifacts", run_dir])
    roots.extend(Path(root).expanduser() for root in extra_search_roots)

    inventory_cache = inventory_cache if inventory_cache is not None else {}
    match_cache = match_cache if match_cache is not None else {}

    candidates: List[TraceCandidate] = []
    seen: set[str] = set()
    for root in unique_paths(roots):
        if not root.exists() or not root.is_dir():
            if progress is not None:
                progress.verbose_log(f"[TTFT] skip missing search root: {root}")
            continue

        root_key = os.path.normpath(str(root))
        match_key = (root_key, pair_tag_low)
        matched_paths = match_cache.get(match_key)
        if matched_paths is None:
            all_paths = scan_ops_trace_inventory(root, inventory_cache=inventory_cache, progress=progress)
            matched_paths = [p for p in all_paths if pair_tag_low in str(p).lower().replace("\\", "/")]
            match_cache[match_key] = matched_paths
            if progress is not None:
                progress.verbose_log(
                    f"[TTFT] matched {len(matched_paths)} ops traces for pair_tag={pair_tag} under root={root}"
                )
        else:
            if progress is not None:
                progress.verbose_log(
                    f"[TTFT] reuse cached matches for pair_tag={pair_tag} under root={root}: {len(matched_paths)}"
                )

        for path in matched_paths:
            key = os.path.normpath(str(path))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(TraceCandidate(path=path, source="filesystem", context={}))
    return candidates


def _looks_like_ops_trace_path(text: str) -> bool:
    low = str(text).lower()
    return low.endswith("_ops_trace.csv") or "_ops_trace.csv" in low


def _walk_json_for_trace_candidates(
    obj: Any,
    json_path: Path,
    run_dir: Path | None,
    remaps: Sequence[Tuple[str, str]],
    inherited_context: Mapping[str, Any],
    out: MutableMapping[str, TraceCandidate],
) -> None:
    if isinstance(obj, dict):
        context: Dict[str, Any] = dict(inherited_context)
        for key, value in obj.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                context[str(key)] = value

        for key, value in obj.items():
            if isinstance(value, str) and _looks_like_ops_trace_path(value):
                path = materialize_path(
                    value,
                    base_dirs=[json_path.parent, run_dir or json_path.parent],
                    remaps=remaps,
                )
                if path is not None:
                    out[os.path.normpath(str(path))] = TraceCandidate(
                        path=path,
                        source=f"json:{json_path.name}",
                        context=dict(context),
                    )
            else:
                _walk_json_for_trace_candidates(
                    value,
                    json_path=json_path,
                    run_dir=run_dir,
                    remaps=remaps,
                    inherited_context=context,
                    out=out,
                )
        return

    if isinstance(obj, list):
        for item in obj:
            _walk_json_for_trace_candidates(
                item,
                json_path=json_path,
                run_dir=run_dir,
                remaps=remaps,
                inherited_context=inherited_context,
                out=out,
            )
        return

    if isinstance(obj, str) and _looks_like_ops_trace_path(obj):
        path = materialize_path(
            obj,
            base_dirs=[json_path.parent, run_dir or json_path.parent],
            remaps=remaps,
        )
        if path is not None:
            out[os.path.normpath(str(path))] = TraceCandidate(
                path=path,
                source=f"json:{json_path.name}",
                context=dict(inherited_context),
            )


def discover_json_trace_candidates(
    row: pd.Series,
    run_dir: Path | None,
    remaps: Sequence[Tuple[str, str]],
) -> List[TraceCandidate]:
    out: Dict[str, TraceCandidate] = {}
    for col in JSON_HINT_COLUMNS:
        if col not in row.index:
            continue
        json_path = materialize_path(row[col], base_dirs=[Path.cwd()], remaps=remaps)
        if json_path is None or not json_path.exists() or not json_path.is_file():
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        _walk_json_for_trace_candidates(
            data,
            json_path=json_path,
            run_dir=run_dir,
            remaps=remaps,
            inherited_context={},
            out=out,
        )
    return list(out.values())


def score_path_for_role(path: Path, role: str, pair_tag: str) -> int:
    s = str(path).lower().replace("\\", "/")
    pair_tag_low = pair_tag.lower()
    score = 0

    if pair_tag_low in s:
        score += 100

    if role == "pd_linear":
        if "pd+linear" in s or "pd_linear" in s or "pd-linear" in s:
            score += 70
        if "pd_linear" in s or "pd+linear" in s or "pd-linear" in s:
            score += 60
        if "linear" in s:
            score += 10
        if "bifocal" in s:
            score -= 80
        if "dual" in s:
            score -= 15

    elif role == "pd_dual":
        if "pd+dual" in s or "pd_dual" in s or "pd-dual" in s:
            score += 70
        if "pd_dual_copy" in s or "pd_dual" in s or "pd+dual" in s or "pd-dual" in s:
            score += 60
        if "dual_copy" in s or "dual" in s:
            score += 10
        if "bifocal" in s:
            score -= 80
        if "linear" in s:
            score -= 15

    elif role == "bifocal_linear":
        if "bifocal_linear" in s or "bifocal+linear" in s or "bifocal-linear" in s:
            score += 80
        if "bifocal" in s and "linear" in s:
            score += 60
        if "pd+" in s or "/algo_pd/" in s or "algo:pd" in s:
            score -= 60
        if "dual" in s and "linear" not in s:
            score -= 15

    elif role == "bifocal_dual":
        if "bifocal_dual" in s or "bifocal+dual" in s or "bifocal-dual" in s:
            score += 80
        if "bifocal" in s and ("dual" in s or "dual_copy" in s):
            score += 60
        if "pd+" in s or "/algo_pd/" in s or "algo:pd" in s:
            score -= 60
        if "linear" in s and "dual" not in s:
            score -= 15

    elif role == "bifocal_linear_best":
        if "best" in s:
            score += 30
        if "search" in s:
            score += 10
        if "bifocal_linear" in s or "bifocal+linear" in s or "bifocal-linear" in s:
            score += 10

    if path.is_file():
        score += 5
    return score


def closeness_score(candidate_value: float, target_value: float) -> int:
    if not np.isfinite(candidate_value) or not np.isfinite(target_value):
        return 0
    diff = abs(float(candidate_value) - float(target_value))
    scale = max(abs(float(target_value)), 1.0)
    if diff <= 1e-9 * scale:
        return 50
    if diff <= 1e-7 * scale:
        return 40
    if diff <= 1e-5 * scale:
        return 30
    if diff <= 1e-3 * scale:
        return 20
    if diff <= 1e-2 * scale:
        return 10
    return 0


def score_json_candidate(
    candidate: TraceCandidate,
    *,
    role: str,
    pair_tag: str,
    target_total: float | None,
    pass_hint: int | None,
    preferred_terms: Sequence[str] | None,
) -> int:
    score = score_path_for_role(candidate.path, role, pair_tag)
    path_text = str(candidate.path).lower().replace("\\", "/")
    ctx = {str(k).lower(): v for k, v in candidate.context.items()}
    ctx_text = " ".join(str(v).lower() for v in ctx.values() if isinstance(v, str))

    if preferred_terms:
        for term in preferred_terms:
            term_low = str(term).lower()
            if term_low and (term_low in ctx_text or term_low in path_text):
                score += 12

    if pass_hint is not None:
        for key, value in ctx.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                if int(round(float(value))) == int(pass_hint) and any(
                    token in key for token in ("pass", "iter", "best")
                ):
                    score += 35

    if target_total is not None and np.isfinite(target_total):
        for value in ctx.values():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                score += closeness_score(float(value), float(target_total))

    return score


def choose_best_path_candidate(
    candidates: Sequence[TraceCandidate],
    *,
    role: str,
    pair_tag: str,
) -> Path | None:
    scored: List[Tuple[int, str, Path]] = []
    for candidate in candidates:
        if not candidate.path.exists() or not candidate.path.is_file():
            continue
        score = score_path_for_role(candidate.path, role, pair_tag)
        scored.append((score, str(candidate.path), candidate.path))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_path = scored[0]
    if best_score < 20:
        return None
    return best_path


def choose_best_json_candidate(
    candidates: Sequence[TraceCandidate],
    *,
    role: str,
    pair_tag: str,
    target_total: float | None,
    pass_hint: int | None,
    preferred_terms: Sequence[str] | None,
) -> Path | None:
    scored: List[Tuple[int, str, Path]] = []
    for candidate in candidates:
        if not candidate.path.exists() or not candidate.path.is_file():
            continue
        score = score_json_candidate(
            candidate,
            role=role,
            pair_tag=pair_tag,
            target_total=target_total,
            pass_hint=pass_hint,
            preferred_terms=preferred_terms,
        )
        scored.append((score, str(candidate.path), candidate.path))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_path = scored[0]
    if best_score < 30:
        return None
    return best_path


def _safe_int(value: Any) -> int | None:
    try:
        value_f = float(value)
        if np.isfinite(value_f):
            return int(round(value_f))
    except Exception:
        pass
    return None


def resolve_trace_for_role(
    row: pd.Series,
    *,
    role: str,
    fs_candidates: Sequence[TraceCandidate],
    json_candidates: Sequence[TraceCandidate],
) -> TraceResolution:
    pair_tag = build_pair_tag(row)

    if role == "pd_linear":
        path = choose_best_path_candidate(fs_candidates, role="pd_linear", pair_tag=pair_tag)
        if path is not None:
            return TraceResolution(path=path, source="filesystem:pd_linear")
        return TraceResolution(path=None, source="filesystem:pd_linear", warning="PD linear trace not found")

    if role == "pd_dual":
        path = choose_best_path_candidate(fs_candidates, role="pd_dual", pair_tag=pair_tag)
        if path is not None:
            return TraceResolution(path=path, source="filesystem:pd_dual")
        return TraceResolution(path=None, source="filesystem:pd_dual", warning="PD dual trace not found")

    if role == "bifocal_dual":
        path = choose_best_path_candidate(fs_candidates, role="bifocal_dual", pair_tag=pair_tag)
        if path is not None:
            return TraceResolution(path=path, source="filesystem:bifocal_dual")
        return TraceResolution(path=None, source="filesystem:bifocal_dual", warning="Bifocal dual trace not found")

    if role == "bifocal_linear_initial":
        path = choose_best_json_candidate(
            json_candidates,
            role="bifocal_linear",
            pair_tag=pair_tag,
            target_total=float(row.get("Bifocal+Linear_initial", np.nan)),
            pass_hint=0,
            preferred_terms=["initial", "pass0", "pass_0", "baseline"],
        )
        if path is not None:
            return TraceResolution(path=path, source="json:bifocal_linear_initial")

        path = choose_best_path_candidate(fs_candidates, role="bifocal_linear", pair_tag=pair_tag)
        if path is not None:
            return TraceResolution(
                path=path,
                source="filesystem:bifocal_linear_fallback_for_initial",
            )
        return TraceResolution(path=None, source="bifocal_linear_initial", warning="Bifocal linear initial trace not found")

    if role == "bifocal_linear_best":
        best_pass = _safe_int(row.get("best_pass", None))
        target_total = float(row.get("Bifocal+Linear_best", np.nan))

        # First try a dedicated best-pass trace from sidecar JSONs.
        path = choose_best_json_candidate(
            json_candidates,
            role="bifocal_linear_best",
            pair_tag=pair_tag,
            target_total=target_total,
            pass_hint=best_pass,
            preferred_terms=["best", "best_pass", "search", "bifocal+linear_best", "bifocal_linear_best"],
        )
        if path is not None:
            return TraceResolution(path=path, source="json:bifocal_linear_best")

        # Next try a linear hint trace directly from the artifacts tree.
        path = choose_best_path_candidate(fs_candidates, role="bifocal_linear_best", pair_tag=pair_tag)
        if path is not None:
            return TraceResolution(path=path, source="filesystem:bifocal_linear_best")

        path = choose_best_path_candidate(fs_candidates, role="bifocal_linear", pair_tag=pair_tag)
        if path is not None:
            warn = (
                "Bifocal linear best trace was not found explicitly; falling back to bifocal-linear "
                "ops trace for TTFT ratio."
            )
            return TraceResolution(
                path=path,
                source="filesystem:bifocal_linear_fallback_for_best",
                warning=warn,
            )

        return TraceResolution(path=None, source="bifocal_linear_best", warning="Bifocal linear best trace not found")

    raise ValueError(f"Unsupported trace role: {role}")


def compute_signature_series(df: pd.DataFrame) -> pd.Series:
    sig_cols = [c for c in ["node_id", "op", "device", "device_type", "mode"] if c in df.columns]
    if not sig_cols:
        raise ValueError("Trace CSV must contain at least one of: node_id, op, device, device_type, mode")
    return df[sig_cols].fillna("").astype(str).agg("\u241f".join, axis=1)


def detect_first_decode_cycle_rows(decode_df: pd.DataFrame) -> int:
    if decode_df.empty:
        raise ValueError("Decode trace is empty")
    if len(decode_df) == 1:
        return 1

    sig = compute_signature_series(decode_df)
    prefix_len = max(1, min(8, len(sig) // 4 if len(sig) >= 4 else 1))
    prefix = sig.iloc[:prefix_len].tolist()
    first_sig = prefix[0]

    candidate_indices = np.flatnonzero(sig.iloc[1:].to_numpy() == first_sig) + 1
    for idx in candidate_indices.tolist():
        if idx + prefix_len > len(sig):
            continue
        if sig.iloc[idx : idx + prefix_len].tolist() == prefix:
            return int(idx)

    # Fallback: if the sequence never repeats, treat the whole decode segment as one step.
    return int(len(decode_df))


def read_trace_stats(path: Path) -> TraceStats:
    required = {"phase", "start", "end", "node_id", "op", "device", "device_type", "mode"}
    df = pd.read_csv(path, usecols=lambda c: c in required, low_memory=False)

    missing = [c for c in ["phase", "start", "end"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns for TTFT inference: {missing}")

    df["phase"] = df["phase"].astype(str).str.lower()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df[np.isfinite(df["start"]) & np.isfinite(df["end"])].copy()
    if df.empty:
        raise ValueError(f"{path}: trace is empty after filtering invalid start/end rows")

    decode_df = df[df["phase"] == "decode"].reset_index(drop=True)
    if decode_df.empty:
        raise ValueError(f"{path}: no decode phase found in trace")

    first_cycle_rows = detect_first_decode_cycle_rows(decode_df)
    first_cycle_df = decode_df.iloc[:first_cycle_rows]
    if first_cycle_df.empty:
        raise ValueError(f"{path}: failed to isolate the first decode cycle")

    global_start = float(df["start"].min())
    global_end = float(df["end"].max())
    raw_total_s = global_end - global_start
    if raw_total_s <= 0 or not np.isfinite(raw_total_s):
        raise ValueError(f"{path}: invalid raw total duration: {raw_total_s}")

    raw_ttft_s = float(first_cycle_df["end"].max()) - global_start
    if raw_ttft_s <= 0 or not np.isfinite(raw_ttft_s):
        raise ValueError(f"{path}: invalid raw TTFT duration: {raw_ttft_s}")

    inferred_decode_steps: int | None = None
    if first_cycle_rows > 0 and len(decode_df) % first_cycle_rows == 0:
        inferred_decode_steps = int(len(decode_df) // first_cycle_rows)

    return TraceStats(
        path=path,
        raw_total_s=raw_total_s,
        raw_ttft_s=raw_ttft_s,
        raw_ttft_ratio=float(raw_ttft_s / raw_total_s),
        first_decode_cycle_rows=int(first_cycle_rows),
        inferred_decode_steps=inferred_decode_steps,
    )


def metric_column_name(value_col: str, metric: str) -> str:
    if metric == "total":
        return value_col
    if metric == "ttft":
        return f"{value_col}__ttft_s"
    raise ValueError(f"Unsupported metric: {metric}")


def metric_display_name(metric: str) -> str:
    if metric == "total":
        return "Total latency"
    if metric == "ttft":
        return "TTFT"
    return metric


def enrich_with_ttft(
    df: pd.DataFrame,
    *,
    extra_search_roots: Sequence[Path],
    remaps: Sequence[Tuple[str, str]],
    strict_ttft: bool,
    progress: ProgressLogger | None,
    progress_every: int,
    persistent_cache: TTFTPersistentCache | None,
) -> Tuple[pd.DataFrame, List[str], List[Dict[str, Any]]]:
    out = df.copy()
    warnings_out: List[str] = []
    usage_records: List[Dict[str, Any]] = []
    trace_stats_cache: Dict[str, TraceStats] = {}
    resolution_cache: Dict[str, TraceResolution] = {}
    inventory_cache: Dict[str, List[Path]] = {}
    match_cache: Dict[Tuple[str, str], List[Path]] = {}

    for spec in BAR_SPECS:
        out[f"{spec.value_col}__ttft_s"] = np.nan
        out[f"{spec.value_col}__ttft_ratio"] = np.nan
        out[f"{spec.value_col}__ttft_trace_path"] = ""
        out[f"{spec.value_col}__ttft_trace_source"] = ""

    total_rows = len(out)
    if progress is not None:
        progress.log(f"[TTFT] starting TTFT enrichment for {total_rows} rows")

    progress_every = max(int(progress_every), 1)

    for row_idx, row in out.iterrows():
        should_log_row = (
            row_idx == 0
            or (row_idx + 1) % progress_every == 0
            or (row_idx + 1) == total_rows
        )
        if progress is not None and should_log_row:
            progress.log(f"[TTFT] row {row_idx + 1}/{total_rows}: {describe_row_for_progress(row)}")

        run_dir = get_run_dir_from_row(row, remaps=remaps)
        fs_candidates: List[TraceCandidate] | None = None
        json_candidates: List[TraceCandidate] | None = None

        for spec in BAR_SPECS:
            total_value = float(row[spec.value_col]) if np.isfinite(row[spec.value_col]) else float("nan")
            resolution_key = make_resolution_cache_key(
                row,
                role=spec.trace_role,
                remaps=remaps,
                extra_search_roots=extra_search_roots,
            )
            resolution = resolution_cache.get(resolution_key)
            if resolution is None and persistent_cache is not None:
                resolution = persistent_cache.get_trace_resolution(resolution_key)
                if resolution is not None:
                    resolution_cache[resolution_key] = resolution
                    if progress is not None:
                        progress.verbose_log(
                            f"[TTFT] resolution cache hit for bar={spec.value_col}: {resolution.path}"
                        )

            if resolution is None:
                if fs_candidates is None:
                    fs_candidates = discover_fs_trace_candidates(
                        row=row,
                        run_dir=run_dir,
                        extra_search_roots=extra_search_roots,
                        progress=progress,
                        inventory_cache=inventory_cache,
                        match_cache=match_cache,
                    )
                if json_candidates is None:
                    json_candidates = discover_json_trace_candidates(
                        row=row,
                        run_dir=run_dir,
                        remaps=remaps,
                    )
                resolution = resolve_trace_for_role(
                    row=row,
                    role=spec.trace_role,
                    fs_candidates=fs_candidates,
                    json_candidates=json_candidates,
                )
                resolution_cache[resolution_key] = resolution
                if persistent_cache is not None and resolution.path is not None:
                    persistent_cache.put_trace_resolution(resolution_key, resolution)

            out.at[row_idx, f"{spec.value_col}__ttft_trace_source"] = resolution.source
            if resolution.path is not None:
                out.at[row_idx, f"{spec.value_col}__ttft_trace_path"] = str(resolution.path)

            if resolution.warning:
                msg = (
                    f"[row={row_idx} model={row.get('model')} batch={row.get('batch')} "
                    f"prefill={row.get('prefill_len')} decode={row.get('decode_len')} "
                    f"bar={spec.value_col}] {resolution.warning}"
                )
                warnings_out.append(msg)

            if resolution.path is None:
                if strict_ttft:
                    raise FileNotFoundError(warnings_out[-1] if warnings_out else f"Trace not found for {spec.value_col}")
                continue

            trace_cache_key = os.path.normpath(str(resolution.path))
            stats = trace_stats_cache.get(trace_cache_key)
            if stats is None and persistent_cache is not None:
                stats = persistent_cache.get_trace_stats(resolution.path)
                if stats is not None:
                    trace_stats_cache[trace_cache_key] = stats
                    if progress is not None:
                        progress.verbose_log(f"[TTFT] trace stats cache hit: {resolution.path}")

            if stats is None:
                try:
                    if progress is not None:
                        progress.log(f"[TTFT] parsing ops trace: {resolution.path}")
                    stats = read_trace_stats(resolution.path)
                    trace_stats_cache[trace_cache_key] = stats
                    if persistent_cache is not None:
                        persistent_cache.put_trace_stats(stats)
                except Exception as exc:
                    msg = (
                        f"[row={row_idx} model={row.get('model')} batch={row.get('batch')} "
                        f"prefill={row.get('prefill_len')} decode={row.get('decode_len')} "
                        f"bar={spec.value_col}] Failed to parse TTFT from {resolution.path}: {exc}"
                    )
                    warnings_out.append(msg)
                    if strict_ttft:
                        raise RuntimeError(msg) from exc
                    continue

            ttft_value = total_value * stats.raw_ttft_ratio if np.isfinite(total_value) else float("nan")
            out.at[row_idx, f"{spec.value_col}__ttft_s"] = ttft_value
            out.at[row_idx, f"{spec.value_col}__ttft_ratio"] = stats.raw_ttft_ratio

            usage_records.append(
                {
                    "row_index": int(row_idx),
                    "model": row.get("model"),
                    "dtype": row.get("dtype") if "dtype" in row.index else None,
                    "batch": row.get("batch"),
                    "prefill_len": row.get("prefill_len"),
                    "decode_len": row.get("decode_len"),
                    "bar_value_col": spec.value_col,
                    "bar_label": spec.label.replace("\n", " "),
                    "trace_path": str(resolution.path),
                    "trace_source": resolution.source,
                    "total_s": total_value,
                    "ttft_s": ttft_value,
                    "ttft_ratio": stats.raw_ttft_ratio,
                    "raw_total_s": stats.raw_total_s,
                    "raw_ttft_s": stats.raw_ttft_s,
                    "first_decode_cycle_rows": stats.first_decode_cycle_rows,
                    "inferred_decode_steps": stats.inferred_decode_steps,
                }
            )

    # Deduplicate warnings while keeping order.
    deduped: List[str] = []
    seen: set[str] = set()
    for msg in warnings_out:
        if msg not in seen:
            seen.add(msg)
            deduped.append(msg)
    return out, deduped, usage_records


def build_usage_report_paths(results_csv: Path, outdir: Path) -> Dict[str, Path]:
    stem = sanitize_filename(results_csv.stem)
    return {
        "detail_csv": outdir / f"{stem}__ttft_trace_usage.csv",
        "trace_txt": outdir / f"{stem}__ttft_trace_usage.txt",
        "used_csvs_txt": outdir / f"{stem}__used_csvs.txt",
    }


def write_usage_reports(
    *,
    results_csv: Path,
    outdir: Path,
    usage_records: Sequence[Mapping[str, Any]],
    logger: ProgressLogger | None,
) -> Dict[str, Path]:
    paths = build_usage_report_paths(results_csv, outdir)

    detail_df = pd.DataFrame(list(usage_records))
    if not detail_df.empty:
        sort_cols = [c for c in ["row_index", "bar_value_col"] if c in detail_df.columns]
        if sort_cols:
            detail_df = detail_df.sort_values(sort_cols).reset_index(drop=True)
    detail_df.to_csv(paths["detail_csv"], index=False)

    unique_traces: List[str] = []
    seen_trace_paths: set[str] = set()
    for record in usage_records:
        trace_path = str(record.get("trace_path", "")).strip()
        if not trace_path:
            continue
        key = os.path.normpath(trace_path)
        if key in seen_trace_paths:
            continue
        seen_trace_paths.add(key)
        unique_traces.append(trace_path)

    with open(paths["trace_txt"], "w", encoding="utf-8") as f:
        for trace_path in unique_traces:
            f.write(trace_path + "\n")

    with open(paths["used_csvs_txt"], "w", encoding="utf-8") as f:
        f.write(str(results_csv) + "\n")
        for trace_path in unique_traces:
            f.write(trace_path + "\n")

    if logger is not None:
        logger.log(f"[TTFT] wrote detailed trace usage CSV: {paths['detail_csv']}")
        logger.log(f"[TTFT] wrote unique used ops trace list: {paths['trace_txt']}")
        logger.log(f"[TTFT] wrote all used CSV list (results.csv + ops traces): {paths['used_csvs_txt']}")

    return paths


def make_panel(
    ax: plt.Axes,
    row: pd.Series,
    value_cols: Sequence[str],
    y_lim: Tuple[float, float] | None,
) -> None:
    labels = [spec.label for spec in BAR_SPECS]
    raw_values = [float(row[col]) if col in row.index else float("nan") for col in value_cols]
    values = np.array(raw_values, dtype=float)
    display_values = np.where(np.isfinite(values), values, 0.0)
    colors = [spec.color for spec in BAR_SPECS]

    x = np.arange(len(labels), dtype=float) * BAR_STEP
    bars = ax.bar(
        x,
        display_values,
        width=BAR_WIDTH,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
        align="center",
    )

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            bar.set_facecolor("white")
            bar.set_edgecolor("0.5")
            bar.set_hatch("//")

    max_value = finite_max(values)
    if y_lim is None:
        ax.set_ylim(*compute_ylim(max_value))
    else:
        ax.set_ylim(*y_lim)

    add_value_labels(ax, bars, values)

    bifocal_linear_reduction_pct = compute_bifocal_linear_best_reduction_pct(
        initial_value=float(values[2]) if len(values) >= 3 else float("nan"),
        best_value=float(values[3]) if len(values) >= 4 else float("nan"),
    )
    ax.text(
        0.98,
        0.97,
        f"ND best reduction: {format_percent(bifocal_linear_reduction_pct)}",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=8.4,
        color="0.12",
        bbox={
            "facecolor": "white",
            "edgecolor": "0.85",
            "linewidth": 0.6,
            "alpha": 0.95,
            "pad": 0.28,
        },
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.0)
    ax.tick_params(axis="x", pad=2.0, length=0)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)

    left_edge = x[0] - BAR_WIDTH / 2.0
    right_edge = x[-1] + BAR_WIDTH / 2.0
    ax.set_xlim(left_edge - SIDE_PADDING, right_edge + SIDE_PADDING)

    ax.set_title(
        f"Prefill={format_length_value(row['prefill_len'])}, Decode={format_length_value(row['decode_len'])}",
        fontsize=10.5,
        pad=6,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_figure_title(group_key: Tuple, group_cols: Sequence[str], metric: str) -> str:
    kv = dict(zip(group_cols, group_key))
    parts: List[str] = []
    if "model" in kv:
        parts.append(str(kv["model"]))
    if "dtype" in kv:
        parts.append(f"dtype={kv['dtype']}")
    if "batch" in kv:
        parts.append(f"batch={kv['batch']}")
    parts.append(f"metric={metric_display_name(metric)}")
    return " | ".join(parts)


def make_figure(
    group_df: pd.DataFrame,
    group_key: Tuple,
    group_cols: Sequence[str],
    outdir: Path,
    fig_formats: Sequence[str],
    dpi: int,
    ncols_arg: str,
    share_y: bool,
    metric: str,
) -> List[Path]:
    n_panels = len(group_df)
    ncols = resolve_ncols(ncols_arg, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig_w = max(4.6 * ncols, 9.2)
    fig_h = max(3.8 * nrows + 0.9, 5.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.array(axes).reshape(-1)

    value_cols = [metric_column_name(spec.value_col, metric) for spec in BAR_SPECS]

    shared_y_lim = None
    if share_y:
        max_value = finite_max(group_df[value_cols].to_numpy(dtype=float).reshape(-1))
        shared_y_lim = compute_ylim(max_value)

    for ax, (_, row) in zip(axes, group_df.iterrows()):
        make_panel(ax, row, value_cols=value_cols, y_lim=shared_y_lim)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle(build_figure_title(group_key, group_cols, metric=metric), fontsize=14.2, y=0.98)
    try:
        fig.supylabel(f"{metric_display_name(metric)} (s)", fontsize=11.2, x=0.02)
    except Exception:
        pass
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.14, wspace=0.26, hspace=0.36)

    title_stem = build_figure_title(group_key, group_cols, metric=metric)
    stem = sanitize_filename(title_stem)
    out_paths: List[Path] = []
    for fmt in fig_formats:
        fmt = str(fmt).lower().lstrip(".")
        out_path = outdir / f"{stem}_layout_compare.{fmt}"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06, dpi=dpi)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths


def main() -> None:
    args = parse_args()
    logger = ProgressLogger(verbose=bool(args.verbose_progress))

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    logger.log(f"[startup] results.csv: {csv_path}")
    logger.log(f"[startup] output directory: {outdir}")

    remaps = parse_path_remaps(args.path_remap)
    extra_search_roots = [Path(p).expanduser() for p in args.extra_search_root]
    if remaps:
        logger.log(f"[startup] path remaps: {remaps}")
    if extra_search_roots:
        logger.log(f"[startup] extra search roots: {[str(p) for p in extra_search_roots]}")

    logger.log(f"[startup] loading results.csv from: {csv_path}")
    df = load_table(csv_path)
    logger.log(f"[startup] loaded {len(df)} rows x {len(df.columns)} columns")
    require_columns(df, REQUIRED_COLUMNS)

    numeric_cols = [
        "batch",
        "prefill_len",
        "decode_len",
        *(spec.value_col for spec in BAR_SPECS),
    ]
    logger.log(f"[startup] converting numeric columns: {numeric_cols}")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = [col for col in ["model", "dtype", "batch", "prefill_len", "decode_len"] if col in df.columns]
    logger.log(f"[startup] sorting rows by: {sort_cols}")
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    requested_metrics = list(dict.fromkeys(str(m).lower() for m in args.metrics))
    logger.log(f"[startup] requested metrics: {requested_metrics}")

    warnings_out: List[str] = []
    usage_records: List[Dict[str, Any]] = []
    usage_report_paths: Dict[str, Path] = {}

    ttft_cache: TTFTPersistentCache | None = None
    if "ttft" in requested_metrics:
        ttft_cache_path = None
        if args.ttft_cache:
            ttft_cache_path = Path(args.ttft_cache).expanduser()
            if not ttft_cache_path.is_absolute():
                ttft_cache_path = (Path.cwd() / ttft_cache_path).resolve()
        else:
            ttft_cache_path = csv_path.with_name(f"{csv_path.stem}__ttft_cache.json")

        logger.log(f"[startup] TTFT cache file: {ttft_cache_path}")
        ttft_cache = TTFTPersistentCache(
            ttft_cache_path,
            logger=logger,
            rebuild=bool(args.rebuild_ttft_cache),
        )

        logger.log("[startup] preparing TTFT columns and resolving trace CSVs")
        df, warnings_out, usage_records = enrich_with_ttft(
            df,
            extra_search_roots=extra_search_roots,
            remaps=remaps,
            strict_ttft=bool(args.strict_ttft),
            progress=logger,
            progress_every=int(args.progress_every),
            persistent_cache=ttft_cache,
        )

        if ttft_cache is not None:
            ttft_cache.save()

        usage_report_paths = write_usage_reports(
            results_csv=csv_path,
            outdir=outdir,
            usage_records=usage_records,
            logger=logger,
        )

    group_cols = [col for col in ["model", "dtype", "batch"] if col in df.columns]
    grouped_items: List[Tuple[Tuple[Any, ...], pd.DataFrame]] = []
    for group_key, group_df in df.groupby(group_cols, sort=False):
        group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        grouped_items.append((group_key_tuple, group_df.reset_index(drop=True)))

    saved_paths: List[Path] = []
    total_figures = len(grouped_items) * len(requested_metrics)
    figure_idx = 0
    logger.log(f"[plot] preparing {total_figures} figure(s)")

    for metric in requested_metrics:
        logger.log(f"[plot] metric: {metric_display_name(metric)}")
        for group_key, group_df in grouped_items:
            figure_idx += 1
            title = build_figure_title(group_key, group_cols, metric=metric)
            logger.log(f"[plot] figure {figure_idx}/{total_figures}: {title}")
            saved_paths.extend(
                make_figure(
                    group_df=group_df,
                    group_key=group_key,
                    group_cols=group_cols,
                    outdir=outdir,
                    fig_formats=args.fig_format,
                    dpi=args.dpi,
                    ncols_arg=str(args.ncols),
                    share_y=bool(args.share_y),
                    metric=metric,
                )
            )

    if args.save_augmented_csv:
        augmented_csv_path = Path(args.save_augmented_csv).expanduser()
        if not augmented_csv_path.is_absolute():
            augmented_csv_path = outdir / augmented_csv_path
        augmented_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(augmented_csv_path, index=False)
        logger.log(f"[startup] saved augmented CSV to: {augmented_csv_path}")

    logger.log(f"Saved {len(saved_paths)} figure(s) to: {outdir}")
    for path in saved_paths:
        print(path)

    if usage_report_paths:
        logger.log(f"[TTFT] results.csv used: {csv_path}")
        logger.log(f"[TTFT] unique used ops trace list: {usage_report_paths['trace_txt']}")
        logger.log(f"[TTFT] all used CSV list: {usage_report_paths['used_csvs_txt']}")

    if warnings_out:
        print(f"\nTTFT warnings ({len(warnings_out)}):")
        for msg in warnings_out:
            print(f"WARNING: {msg}")


if __name__ == "__main__":
    main()
