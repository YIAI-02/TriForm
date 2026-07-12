#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python plot_exp3_layout_compare.py \
  --csv ../../algorithms/output/ws_overlap_1p0_low_npu_bw_nz0p95/shards/24w/results.csv  ../../algorithms/output/ws_overlap_1p0_low_npu_bw_nz0p95/shards/27w/results.csv\
  --outdir ../../figs/exp3/compare \
  --metrics total \
  --prefill-len 8 32 128 1024\
  --decode-len 8 128\
  --batch 1 4 \
  --algo-label pd_linear_initial='PD\nLinear' \
  --algo-label pd_dual_copy_initial='PD\nDual' \
  --algo-label nd_initial='Bifocal' \
  --algo-label nd_best='Bifocal\n+\nWeight\nArbiter' \
  --algo-label hefthint_dual_copy_best='Bifocal\nDual' \
  --format-label ND='Linear' \
  --format-label NZ='NPU-OPT' \
  --format-label PIM-OPT='PIM-OPT'\
  --fig-format png pdf \
  --share-y

cd experiment/experiment_fig

python plot_exp3_layout_compare.py \
  --csv ../../algorithms/output/ws_overlap_1p0_low_npu_bw_nz0p95_qwen14b/shards/75w/results.csv \
  --outdir ../../figs/exp3/compare_qwen14b \
  --fig-format png \
  --path-remap /path/to/original/results=/path/to/remapped/results \
  --share-y \
  --max-panels-per-figure 6 \
  --verbose-pie
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# Must be set before importing pyplot on HPC.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Matplotlib defaults
# -----------------------------------------------------------------------------
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
mpl.rcParams["figure.max_open_warning"] = 0
mpl.rcParams["agg.path.chunksize"] = 10000


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
BACKGROUND = "#efefef"
EDGE_COLOR = "#2b1d1a"
BAR_WIDTH = 0.90
BAR_STEP = BAR_WIDTH
PANEL_X_PAD = 0.28
DEFAULT_MAX_PIE_JSON_MB = 32.0


@dataclass(frozen=True)
class BarSpec:
    key: str
    color: str
    legend_label: str


BAR_SPECS: List[BarSpec] = [
    BarSpec("PD+Linear", "#b8add9", "PD+Linear"),
    BarSpec("PD+Dual", "#d8b6bd", "PD+Dual"),
    BarSpec("Bifocal+Linear", "#bfd8b0", "Bifocal+Linear"),
    BarSpec("Bifocal+WLA", "#b9dce7", "Bifocal+WLA"),
    BarSpec("Bifocal+Dual", "#d8d1a1", "Bifocal+Dual"),
]

BAR_LEGEND_ORDER = [
    "PD+Dual",
    "PD+Linear",
    "Bifocal+WLA",
    "Bifocal+Linear",
    "Bifocal+Dual",
]

BAR_COLOR_BY_KEY = {spec.key: spec.color for spec in BAR_SPECS}

PIE_ORDER = ["Linear", "NPU_OPT", "PIM_OPT"]
PIE_COLORS = {
    "Linear": "#d8b6bd",
    "NPU_OPT": "#b8add9",
    "PIM_OPT": "#bfd8b0",
}

GROUP_COLS_CANDIDATES = ["model", "dtype"]
PANEL_SORT_COLS = ["batch", "prefill_len", "decode_len"]

META_ALIASES: Dict[str, List[str]] = {
    "model": ["model", "model_name", "net", "network"],
    "dtype": ["dtype", "data_type", "precision"],
    "batch": ["batch", "batch_size", "bs"],
    "prefill_len": ["prefill_len", "prefill_length", "prefill", "prompt_len", "context_len", "input_len"],
    "decode_len": ["decode_len", "decode_length", "decode", "gen_len", "output_len"],
    "best_pass": ["best_pass", "best_iter", "best_iteration", "best_round"],
    "generated_config_json": ["generated_config_json", "generated_config", "config_json"],
    "best_summary_json": ["best_summary_json", "best_summary", "summary_json"],
    "all_passes_json": ["all_passes_json", "all_passes", "passes_json"],
    "weight_format_json": ["weight_format_json", "weight_format", "layout_json", "format_json"],
    "weight_format_compare_json": [
        "weight_format_compare_json",
        "weight_format_compare",
        "layout_compare_json",
        "format_compare_json",
    ],
    "log_path": ["log_path", "log", "logfile", "run_dir"],
}

# For pie extraction, only search lightweight, layout-related JSONs.
PIE_JSON_HINT_COLUMNS = [
    "weight_format_compare_json",
    "weight_format_json",
    "best_summary_json",
    "generated_config_json",
]

RUN_PATH_COLUMNS = [
    "generated_config_json",
    "best_summary_json",
    "all_passes_json",
    "weight_format_json",
    "weight_format_compare_json",
    "log_path",
]

BAR_ALIASES: Dict[str, List[str]] = {
    "PD+Linear": [
        "PD+Linear",
        "PD+Linear_initial",
        "PD+Linear_best",
        "pd+linear",
        "pd+linear_initial",
        "pd+linear_best",
        "pd_linear",
        "pd_linear_initial",
        "pd_linear_best",
        "algo_pd+linear",
        "algo_pd_linear",
    ],
    "PD+Dual": [
        "PD+Dual",
        "PD+Dual_initial",
        "PD+Dual_best",
        "pd+dual",
        "pd+dual_initial",
        "pd+dual_best",
        "pd_dual",
        "pd_dual_initial",
        "pd_dual_best",
        "pd_dual_copy",
        "pd_dual_copy_initial",
        "pd_dual_copy_best",
        "algo_pd+dual",
        "algo_pd_dual",
    ],
    "Bifocal+Linear": [
        "nd_linear",
    ],
    "Bifocal+WLA": [
        "nd_best",
    ],
    "Bifocal+Dual": [
        "Bifocal+Dual",
        "Bifocal+Dual_best",
        "Bifocal+Dual_initial",
        "bifocal+dual",
        "bifocal+dual_best",
        "bifocal+dual_initial",
        "bifocal_dual",
        "bifocal_dual_best",
        "bifocal_dual_initial",
        "Hefthint+Dual",
        "Hefthint+Dual_best",
        "Hefthint+Dual_initial",
        "hefthint+dual",
        "hefthint+dual_best",
        "hefthint+dual_initial",
        "heft_hint+dual",
        "heft_hint+dual_best",
        "heft_hint+dual_initial",
        "hefthint_dual",
        "hefthint_dual_best",
        "hefthint_dual_initial",
        "hefthint_dual_copy",
        "hefthint_dual_copy_best",
        "hefthint_dual_copy_initial",
    ],
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "figure"


def normalize_text(text: Any) -> str:
    s = str(text)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = s.lower()
    s = s.replace("heft_hint", "hefthint")
    s = s.replace("heft-hint", "hefthint")
    s = s.replace("bifocalhint", "bifocal")
    s = s.replace("npuopt", "npu_opt")
    s = s.replace("pimopt", "pim_opt")
    s = s.replace("dualcopy", "dual_copy")
    s = s.replace("dual-copy", "dual_copy")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def safe_int(value: Any) -> Optional[int]:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return int(round(out))


def format_num(value: Any) -> str:
    v = safe_float(value)
    if np.isfinite(v):
        if float(v).is_integer():
            return str(int(v))
        return f"{v:g}"
    return str(value)


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def has_glob_magic(text: str) -> bool:
    return any(ch in str(text) for ch in "*?[]")


def resolve_input_table_paths(inputs: Sequence[str], *, directory_pattern: str) -> List[Path]:
    resolved: List[Path] = []

    for raw_item in inputs:
        raw = str(raw_item).strip()
        if not raw:
            continue

        expanded = os.path.expanduser(raw)
        if has_glob_magic(expanded):
            matches = [Path(p).expanduser().resolve() for p in glob.glob(expanded, recursive=True)]
            matches = [p for p in matches if p.exists() and p.is_file()]
            if not matches:
                raise FileNotFoundError(f"No input tables matched glob: {raw}")
            resolved.extend(sorted(matches))
            continue

        path = Path(expanded).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            resolved.append(path.resolve())
            continue

        if path.is_dir():
            matches = sorted(p.resolve() for p in path.rglob(directory_pattern) if p.is_file())
            if not matches:
                raise FileNotFoundError(
                    f"No input tables matching pattern '{directory_pattern}' were found under: {path}"
                )
            resolved.extend(matches)
            continue

        raise ValueError(f"Unsupported input path: {path}")

    resolved = unique_paths(resolved)
    if not resolved:
        raise ValueError("No input tables were resolved from --csv")
    return resolved


def build_input_stem(input_paths: Sequence[Path]) -> str:
    paths = [Path(p) for p in input_paths]
    if not paths:
        return "merged_results"
    if len(paths) == 1:
        return sanitize_filename(paths[0].stem)

    try:
        common_parent = Path(os.path.commonpath([str(p.parent) for p in paths]))
        parent_name = sanitize_filename(common_parent.name)
    except Exception:
        parent_name = ""

    name = parent_name or "merged_results"
    return f"{name}_{len(paths)}tables"


def is_missing_like(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return True
    return False


def values_equivalent(a: Any, b: Any) -> bool:
    if is_missing_like(a) and is_missing_like(b):
        return True

    af = safe_float(a)
    bf = safe_float(b)
    if np.isfinite(af) and np.isfinite(bf):
        return bool(np.isclose(af, bf, rtol=1e-9, atol=1e-12))

    return str(a).strip() == str(b).strip()


def unique_non_missing_values(values: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    for value in values:
        if is_missing_like(value):
            continue
        if any(values_equivalent(value, existing) for existing in out):
            continue
        out.append(value)
    return out


def coalesce_duplicate_panel_rows(
    df: pd.DataFrame,
    *,
    key_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    usable_key_cols = [col for col in key_cols if col in df.columns]
    if not usable_key_cols or df.empty:
        return df.reset_index(drop=True), []

    merged_rows: List[pd.Series] = []
    warnings: List[str] = []

    for group_key, group_df in df.groupby(usable_key_cols, dropna=False, sort=False):
        if len(group_df) == 1:
            merged_rows.append(group_df.iloc[0].copy())
            continue

        row = group_df.iloc[0].copy()
        for col in df.columns:
            uniq = unique_non_missing_values(group_df[col].tolist())
            if not uniq:
                row[col] = np.nan
                continue
            if col == "__source_csv":
                row[col] = ";".join(str(x) for x in uniq)
                continue
            row[col] = uniq[0]
            if len(uniq) > 1 and col not in usable_key_cols:
                warnings.append(
                    f"duplicate panel {group_key}: column '{col}' had {len(uniq)} different values; kept the first non-missing one"
                )

        merged_rows.append(row)

    out = pd.DataFrame(merged_rows).reset_index(drop=True)
    return out, warnings


def load_and_canonicalize_inputs(input_paths: Sequence[Path]) -> Tuple[pd.DataFrame, Dict[str, str], List[Tuple[Path, Dict[str, str]]]]:
    canonical_frames: List[pd.DataFrame] = []
    merged_source_map: Dict[str, str] = {}
    per_file_maps: List[Tuple[Path, Dict[str, str]]] = []

    for input_path in input_paths:
        df_raw = load_table(input_path)
        df_one, source_map = canonicalize_columns(df_raw)
        df_one["__source_csv"] = str(input_path)
        canonical_frames.append(df_one)
        per_file_maps.append((input_path, source_map))
        for key, value in source_map.items():
            merged_source_map.setdefault(key, value)

    if not canonical_frames:
        raise ValueError("No input tables were loaded.")

    combined = pd.concat(canonical_frames, ignore_index=True, sort=False)
    return combined, merged_source_map, per_file_maps


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = os.path.normpath(str(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


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


def materialize_path(raw_path: Any, base_dirs: Sequence[Path] | None, remaps: Sequence[Tuple[str, str]]) -> Optional[Path]:
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

    for cand in unique_paths(candidates):
        if cand.exists():
            return cand
    return None


def get_run_dir_from_row(row: pd.Series, remaps: Sequence[Tuple[str, str]]) -> Optional[Path]:
    for col in RUN_PATH_COLUMNS:
        if col not in row.index:
            continue
        path = materialize_path(row.get(col), base_dirs=[Path.cwd()], remaps=remaps)
        if path is None:
            continue
        return path.parent if path.suffix else path
    return None


# -----------------------------------------------------------------------------
# Column canonicalization
# -----------------------------------------------------------------------------
def build_normalized_column_map(columns: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for col in columns:
        out.setdefault(normalize_text(col), col)
    return out


def find_first_existing_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    norm_map = build_normalized_column_map(df.columns)
    for alias in aliases:
        src = norm_map.get(normalize_text(alias))
        if src is not None:
            return src
    return None


def infer_canonical_bar_from_column(name: str) -> Optional[str]:
    n = normalize_text(name)

    algo: Optional[str] = None
    if "bifocal" in n or "hefthint" in n:
        algo = "Bifocal"
    else:
        tokens = set(n.split("_"))
        if "pd" in tokens or n.startswith("pd_") or n.endswith("_pd"):
            algo = "PD"

    if algo is None:
        return None

    if (
        "wla" in n
        or "linear_best" in n
        or "best_linear" in n
        or "weight_layout" in n
        or "weight_format_compare" in n
    ):
        layout = "WLA"
    elif "dual" in n:
        layout = "Dual"
    elif "linear" in n:
        layout = "Linear"
    else:
        return None

    return f"{algo}+{layout}"


def choose_best_bar_source(df: pd.DataFrame, canonical_bar: str) -> Optional[str]:
    # 1) explicit alias priority
    src = find_first_existing_column(df, BAR_ALIASES[canonical_bar])
    if src is not None:
        return src

    # 2) rule-based fallback among numeric columns only.
    scores: List[Tuple[int, str]] = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        inferred = infer_canonical_bar_from_column(col)
        if inferred != canonical_bar:
            continue
        score = 0
        n = normalize_text(col)

        if "role" in n:
            continue

        if canonical_bar == "Bifocal+WLA":
            if "wla" in n:
                score += 120
            if "linear_best" in n or "best_linear" in n:
                score += 100
            if n.endswith("_best"):
                score += 20

        if canonical_bar == "Bifocal+Linear":
            if "linear_initial" in n:
                score += 100
            if n.endswith("_initial"):
                score += 20

        if canonical_bar == "Bifocal+Dual":
            if "dual_copy_best" in n or "dual_best" in n:
                score += 140
            elif "dual_copy_initial" in n or "dual_initial" in n:
                score += 100
            elif "dual_copy" in n:
                score += 60
            elif "dual" in n:
                score += 30

        if canonical_bar == "PD+Dual":
            if "dual_copy_initial" in n or "dual_initial" in n:
                score += 120
            elif "dual_copy_best" in n or "dual_best" in n:
                score += 80
            elif "dual_copy" in n:
                score += 50

        if canonical_bar == "PD+Linear":
            if "linear_initial" in n:
                score += 120
            elif "linear_best" in n:
                score += 40

        if n == normalize_text(canonical_bar):
            score += 200
        score += max(0, 30 - len(n))
        scores.append((score, col))

    if not scores:
        return None
    scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scores[0][1]


def canonicalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    source_map: Dict[str, str] = {}

    for canonical, aliases in META_ALIASES.items():
        if canonical in out.columns:
            source_map[canonical] = canonical
            continue
        src = find_first_existing_column(out, aliases)
        if src is not None:
            out[canonical] = out[src]
            source_map[canonical] = src

    if "model" not in out.columns:
        out["model"] = "model"
        source_map["model"] = "<constant:model>"
    if "dtype" not in out.columns:
        out["dtype"] = ""
        source_map["dtype"] = "<constant:dtype>"

    for spec in BAR_SPECS:
        if spec.key in out.columns:
            source_map[spec.key] = spec.key
            continue
        src = choose_best_bar_source(out, spec.key)
        if src is not None:
            out[spec.key] = out[src]
            source_map[spec.key] = src

    for col in ["batch", "prefill_len", "decode_len", "best_pass", *[spec.key for spec in BAR_SPECS]]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    required = ["model", "batch", "prefill_len", "decode_len", *[spec.key for spec in BAR_SPECS]]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(
            "Missing required canonical columns after alias mapping: "
            f"{missing}\nAvailable input columns: {list(df.columns)}"
        )

    return out, source_map


# -----------------------------------------------------------------------------
# Pie extraction (full-JSON only)
# -----------------------------------------------------------------------------
FULL_JSON_PREFERRED_NAMES = [
    "weight_storage_suggestion_full.json",
]

FULL_JSON_EXPLICIT_COLUMN_ALIASES = [
    "weight_storage_suggestion_full_json",
    "weight_storage_full_json",
    "best_weight_format_full_json",
    "full_weight_json",
    "full_layout_json",
    "full_format_json",
]


def canonical_layout_family(value: Any) -> Optional[str]:
    n = normalize_text(value)
    if not n:
        return None

    if n in {"nd", "linear"} or "linear" in n:
        return "Linear"
    if n in {"nz", "npu", "npu_opt"} or "npu_opt" in n:
        return "NPU_OPT"
    if n in {"pim", "pim_opt"} or "pim_opt" in n:
        return "PIM_OPT"
    return None


def _count_layout_families_in_full_json(obj: Any, counts: MutableMapping[str, float]) -> None:
    if isinstance(obj, Mapping):
        count_dict_hit = False
        for key, value in obj.items():
            family = canonical_layout_family(key)
            numeric = safe_float(value)
            if family is not None and np.isfinite(numeric) and numeric >= 0:
                counts[family] += float(numeric)
                count_dict_hit = True
        if count_dict_hit:
            return

        for value in obj.values():
            _count_layout_families_in_full_json(value, counts)
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            _count_layout_families_in_full_json(item, counts)
        return

    family = canonical_layout_family(obj)
    if family is not None:
        counts[family] += 1.0


def load_pie_counts_from_full_json(
    path: Path,
    *,
    pie_cache: MutableMapping[str, Optional[Dict[str, float]]],
    verbose: bool,
) -> Optional[Dict[str, float]]:
    cache_key = os.path.normpath(str(path))
    if cache_key in pie_cache:
        return pie_cache[cache_key]

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        if verbose:
            log(f"[pie] failed to parse full JSON {path}: {exc}")
        pie_cache[cache_key] = None
        return None

    counts: Dict[str, float] = {family: 0.0 for family in PIE_ORDER}
    _count_layout_families_in_full_json(data, counts)
    total = sum(counts.values())

    resolved = counts if total > 0 else None
    pie_cache[cache_key] = resolved

    if verbose:
        if resolved is None:
            log(f"[pie] no recognized layout tags found in: {path}")
        else:
            log(
                "[pie] loaded from full JSON: "
                f"{path} -> Linear={int(resolved['Linear'])}, "
                f"NPU_OPT={int(resolved['NPU_OPT'])}, "
                f"PIM_OPT={int(resolved['PIM_OPT'])}"
            )

    return resolved


def _full_json_candidate_score(path: Path) -> Tuple[int, str]:
    name = normalize_text(path.name)
    score = 0
    if path.name == "weight_storage_suggestion_full.json":
        score += 1000
    if "weight_storage_suggestion" in name:
        score += 300
    if "storage_suggestion" in name:
        score += 120
    if "weight" in name:
        score += 60
    if name.endswith("full_json") or name.endswith("full"):
        score += 20
    return score, str(path)


def locate_full_json_for_row(
    row: pd.Series,
    *,
    remaps: Sequence[Tuple[str, str]],
    verbose: bool,
) -> Optional[Path]:
    alias_norms = {normalize_text(x) for x in FULL_JSON_EXPLICIT_COLUMN_ALIASES}

    # 1) Explicit full-json column, if present.
    for col in row.index:
        col_norm = normalize_text(col)
        if col_norm not in alias_norms and not ("full" in col_norm and "json" in col_norm and any(tok in col_norm for tok in ["weight", "format", "layout", "storage"])):
            continue
        path = materialize_path(row.get(col), base_dirs=[Path.cwd()], remaps=remaps)
        if path is not None and path.exists() and path.is_file():
            if verbose:
                log(f"[pie] using explicit full JSON column {col}: {path}")
            return path

    run_dir = get_run_dir_from_row(row, remaps=remaps)
    if run_dir is None or not run_dir.exists() or not run_dir.is_dir():
        if verbose:
            log(
                f"[pie] run dir not found for row: batch={format_num(row.get('batch'))}, "
                f"prefill={format_num(row.get('prefill_len'))}, decode={format_num(row.get('decode_len'))}"
            )
        return None

    candidates: List[Path] = []

    # 2) Preferred exact name in the run directory.
    for name in FULL_JSON_PREFERRED_NAMES:
        candidates.append(run_dir / name)

    # 3) Derive sibling *_full.json next to weight_format_json if possible.
    if "weight_format_json" in row.index:
        wf_path = materialize_path(
            row.get("weight_format_json"),
            base_dirs=[Path.cwd(), run_dir],
            remaps=remaps,
        )
        if wf_path is not None:
            candidates.append(wf_path.with_name(f"{wf_path.stem}_full.json"))
            candidates.append(wf_path.with_name("weight_storage_suggestion_full.json"))

    existing = [p for p in unique_paths(candidates) if p.exists() and p.is_file()]
    if existing:
        existing.sort(key=_full_json_candidate_score, reverse=True)
        chosen = existing[0]
        if verbose:
            log(f"[pie] using preferred full JSON: {chosen}")
        return chosen

    # 4) Fallback: only scan for *_full.json under the run directory.
    scanned = [p for p in run_dir.rglob("*_full.json") if p.is_file()]
    if not scanned:
        if verbose:
            log(f"[pie] no *_full.json found under run dir: {run_dir}")
        return None

    scanned = unique_paths(scanned)
    scanned.sort(key=_full_json_candidate_score, reverse=True)
    chosen = scanned[0]
    if verbose:
        log(f"[pie] selected scanned full JSON: {chosen}")
    return chosen


def resolve_pie_counts(
    row: pd.Series,
    *,
    remaps: Sequence[Tuple[str, str]],
    pie_cache: MutableMapping[str, Optional[Dict[str, float]]],
    max_json_bytes: int,
    verbose: bool,
) -> Optional[Dict[str, float]]:
    del max_json_bytes  # kept only for CLI compatibility
    path = locate_full_json_for_row(row, remaps=remaps, verbose=verbose)
    if path is None:
        return None
    return load_pie_counts_from_full_json(path, pie_cache=pie_cache, verbose=verbose)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def auto_ncols(n_panels: int) -> int:
    if n_panels <= 6:
        return max(1, n_panels)
    if n_panels <= 12:
        return 6
    if n_panels <= 16:
        return 4
    return 5


def resolve_ncols(raw: str, n_panels: int) -> int:
    if str(raw).strip().lower() == "auto":
        return auto_ncols(n_panels)
    ncols = int(raw)
    if ncols <= 0:
        raise ValueError("--ncols must be auto or a positive integer")
    return ncols


def finite_max(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else float("nan")


def compute_ylim(max_value: float) -> Tuple[float, float]:
    if not np.isfinite(max_value) or max_value <= 0:
        return (0.0, 1.0)
    return (0.0, max_value * 1.16)


def speedup_vs_baseline(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    baseline = arr[0] if arr.size else float("nan")
    out: List[float] = []
    for value in arr:
        if np.isfinite(baseline) and baseline > 0 and np.isfinite(value) and value > 0:
            out.append(float(baseline / value))
        else:
            out.append(float("nan"))
    return out


def add_speedup_labels(ax: plt.Axes, x: np.ndarray, values: Sequence[float], fontsize: float = 11.5) -> None:
    labels = speedup_vs_baseline(values)
    ymin, ymax = ax.get_ylim()
    base_y = ymin + (ymax - ymin) * 0.02
    for xi, label in zip(x, labels):
        text = f"{label:.2f}x" if np.isfinite(label) else "N/A"
        ax.text(
            xi,
            base_y,
            text,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fontsize,
            color="black",
            zorder=5,
        )


def add_panel_y_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "",
        xy=(0.0, 1.03),
        xytext=(0.0, -0.02),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.4, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )


def draw_pie(ax: plt.Axes, counts: Mapping[str, float]) -> None:
    values = [float(counts.get(name, 0.0)) for name in PIE_ORDER]
    if sum(values) <= 0:
        return

    pie_ax = ax.inset_axes([0.58, 0.67, 0.30, 0.30])
    pie_ax.pie(
        values,
        colors=[PIE_COLORS[name] for name in PIE_ORDER],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(linewidth=1.0, edgecolor=EDGE_COLOR),
    )
    pie_ax.set_aspect("equal")
    pie_ax.set_facecolor("none")
    pie_ax.set_xticks([])
    pie_ax.set_yticks([])


def make_panel(
    ax: plt.Axes,
    row: pd.Series,
    *,
    shared_ylim: Optional[Tuple[float, float]],
    panel_index: int,
    ncols: int,
    show_pie: bool,
    pie_counts: Optional[Mapping[str, float]],
) -> None:
    values = np.array([safe_float(row.get(spec.key)) for spec in BAR_SPECS], dtype=float)
    display_values = np.where(np.isfinite(values), values, 0.0)
    x = np.arange(len(BAR_SPECS), dtype=float) * BAR_STEP

    bars = ax.bar(
        x,
        display_values,
        width=BAR_WIDTH,
        color=[spec.color for spec in BAR_SPECS],
        edgecolor=EDGE_COLOR,
        linewidth=1.45,
        zorder=3,
        align="center",
    )

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            bar.set_facecolor("white")
            bar.set_edgecolor("0.4")
            bar.set_hatch("//")

    if shared_ylim is None:
        ax.set_ylim(*compute_ylim(finite_max(values)))
    else:
        ax.set_ylim(*shared_ylim)

    add_speedup_labels(ax, x, values)

    if show_pie and pie_counts is not None:
        draw_pie(ax, pie_counts)

    panel_label = f"({format_num(row.get('batch'))}, {format_num(row.get('prefill_len'))}, {format_num(row.get('decode_len'))})"
    ax.text(
        0.97,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        color="black",
    )

    ax.set_facecolor(BACKGROUND)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="-", linewidth=0.6, alpha=0.18, zorder=0)

    left_edge = x[0] - BAR_WIDTH / 2.0
    right_edge = x[-1] + BAR_WIDTH / 2.0
    ax.set_xlim(left_edge - PANEL_X_PAD, right_edge + PANEL_X_PAD)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(1.35)
    add_panel_y_arrow(ax)

    if (panel_index % ncols) != 0:
        ax.set_yticklabels([])


def add_figure_legends(fig: plt.Figure, *, panel_letter: Optional[str], include_pie_legend: bool) -> None:
    bar_handles = [
        mpatches.Patch(facecolor=BAR_COLOR_BY_KEY[key], edgecolor="none", label=key)
        for key in BAR_LEGEND_ORDER
    ]
    bar_legend = fig.legend(
        handles=bar_handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.995),
        ncol=3,
        fontsize=11,
        title="Color of bar:",
        title_fontsize=13,
        frameon=True,
        fancybox=False,
        columnspacing=0.8,
        handlelength=0.9,
        handletextpad=0.3,
        borderpad=0.4,
    )
    bar_legend.get_frame().set_facecolor("white")
    bar_legend.get_frame().set_edgecolor("0.45")
    bar_legend.get_frame().set_linewidth(1.2)
    bar_legend.get_frame().set_linestyle("--")

    if include_pie_legend:
        pie_handles = [
            mpatches.Patch(facecolor=PIE_COLORS[key], edgecolor="none", label=key)
            for key in PIE_ORDER
        ]
        pie_legend = fig.legend(
            handles=pie_handles,
            loc="upper center",
            bbox_to_anchor=(0.56, 0.995),
            ncol=3,
            fontsize=11,
            title="Color of pie:",
            title_fontsize=13,
            frameon=True,
            fancybox=False,
            columnspacing=0.9,
            handlelength=0.9,
            handletextpad=0.3,
            borderpad=0.4,
        )
        pie_legend.get_frame().set_facecolor("white")
        pie_legend.get_frame().set_edgecolor("0.45")
        pie_legend.get_frame().set_linewidth(1.2)
        pie_legend.get_frame().set_linestyle("--")

    right_text = "upper-right corner label:\n(batch, prefill length, decode length)"
    if panel_letter:
        right_text = right_text + f"\n({panel_letter})"
    fig.text(
        0.985,
        0.985,
        right_text,
        ha="right",
        va="top",
        fontsize=12.5,
        bbox=dict(facecolor="white", edgecolor="0.45", linewidth=1.2, linestyle="--", pad=6.0),
    )


def build_group_title(group_key: Tuple[Any, ...], group_cols: Sequence[str]) -> str:
    pieces: List[str] = []
    mapping = dict(zip(group_cols, group_key))
    model = str(mapping.get("model", "")).strip()
    dtype = str(mapping.get("dtype", "")).strip()
    if model:
        pieces.append(model)
    if dtype:
        pieces.append(dtype)
    return "_".join(sanitize_filename(x) for x in pieces if x) or "layout_compare"


def chunk_group_df(group_df: pd.DataFrame, max_panels_per_figure: Optional[int]) -> List[pd.DataFrame]:
    if max_panels_per_figure is None or max_panels_per_figure <= 0 or len(group_df) <= max_panels_per_figure:
        return [group_df.reset_index(drop=True)]
    pages: List[pd.DataFrame] = []
    for start in range(0, len(group_df), max_panels_per_figure):
        pages.append(group_df.iloc[start:start + max_panels_per_figure].reset_index(drop=True))
    return pages


def make_figure(
    group_df: pd.DataFrame,
    *,
    group_key: Tuple[Any, ...],
    group_cols: Sequence[str],
    outdir: Path,
    fig_formats: Sequence[str],
    dpi: int,
    ncols_arg: str,
    share_y: bool,
    show_pie: bool,
    remaps: Sequence[Tuple[str, str]],
    panel_letter: Optional[str],
    pie_cache: MutableMapping[str, Optional[Dict[str, float]]],
    page_index: int,
    n_pages: int,
    max_json_bytes: int,
    verbose_pie: bool,
) -> List[Path]:
    n_panels = len(group_df)
    ncols = resolve_ncols(ncols_arg, n_panels)
    nrows = math.ceil(n_panels / ncols)

    panel_w = 2.15
    panel_h = 2.35
    fig_w = max(8.5, panel_w * ncols + 0.65)
    fig_h = max(3.5, panel_h * nrows + 1.10)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    fig.patch.set_facecolor(BACKGROUND)

    shared_ylim: Optional[Tuple[float, float]] = None
    if share_y:
        all_values = group_df[[spec.key for spec in BAR_SPECS]].to_numpy(dtype=float).reshape(-1)
        shared_ylim = compute_ylim(finite_max(all_values))

    pie_cache_by_row: Dict[int, Optional[Dict[str, float]]] = {}
    pie_found = False
    if show_pie:
        for row_idx, row in group_df.iterrows():
            counts = resolve_pie_counts(
                row,
                remaps=remaps,
                pie_cache=pie_cache,
                max_json_bytes=max_json_bytes,
                verbose=verbose_pie,
            )
            pie_cache_by_row[int(row_idx)] = counts
            if counts is not None:
                pie_found = True

    for panel_index, (ax, (_, row)) in enumerate(zip(axes, group_df.iterrows())):
        make_panel(
            ax,
            row,
            shared_ylim=shared_ylim,
            panel_index=panel_index,
            ncols=ncols,
            show_pie=show_pie,
            pie_counts=pie_cache_by_row.get(int(row.name)),
        )

    for ax in axes[n_panels:]:
        ax.axis("off")
        ax.set_facecolor(BACKGROUND)

    add_figure_legends(fig, panel_letter=panel_letter, include_pie_legend=(show_pie and pie_found))
    fig.text(0.017, 0.54, "Latency(s)", rotation=90, ha="center", va="center", fontsize=16)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.77, bottom=0.12, wspace=0.06, hspace=0.28)

    stem = build_group_title(group_key, group_cols)
    if n_pages > 1:
        stem = f"{stem}_p{page_index + 1:02d}"

    saved: List[Path] = []
    for fmt in fig_formats:
        fmt = str(fmt).lower().lstrip(".")
        out_path = outdir / f"{stem}_layout_compare.{fmt}"
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
        saved.append(out_path)

    plt.close(fig)
    gc.collect()
    return saved


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot total-latency layout comparison figures in the sample style. "
            "TTFT is intentionally removed."
        )
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help=(
            "One or more input tables, directories, or glob patterns. "
            "Directories are searched recursively for results tables."
        ),
    )
    parser.add_argument(
        "--csv-pattern",
        default="results*.csv",
        help="When a --csv input is a directory, recursively include files matching this pattern. Default: results*.csv",
    )
    parser.add_argument("--outdir", default="plots_layout_compare", help="Output directory")
    parser.add_argument("--fig-format", nargs="+", default=["png"], help="Output format(s), e.g. png pdf")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI")
    parser.add_argument("--ncols", default="auto", help="Number of columns per figure, or auto")
    parser.add_argument("--share-y", action="store_true", help="Share y-axis range across all panels in the same figure")
    parser.add_argument(
        "--no-pie",
        action="store_true",
        help="Disable the per-panel WLA pie chart even if pie data can be resolved.",
    )
    parser.add_argument(
        "--panel-letter",
        default="a",
        help="Figure panel letter shown in the upper-right legend box. Use empty string to hide.",
    )
    parser.add_argument(
        "--path-remap",
        nargs="*",
        default=[],
        help="Prefix remap in SRC=DST form, useful when JSON paths were generated elsewhere.",
    )
    parser.add_argument(
        "--save-canonical-csv",
        action="store_true",
        help="Also save the merged, canonicalized table alongside the figures.",
    )
    parser.add_argument(
        "--max-panels-per-figure",
        type=int,
        default=6,
        help="Split each logical group into multiple pages. Default: 6 panels per figure.",
    )
    parser.add_argument(
        "--pie-json-max-mb",
        type=float,
        default=DEFAULT_MAX_PIE_JSON_MB,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verbose-pie",
        action="store_true",
        help="Print pie-resolution diagnostics for *_full.json selection and counting.",
    )
    parser.add_argument(
        "--no-merge-duplicate-panels",
        action="store_true",
        help=(
            "Do not coalesce rows that share the same (model, dtype, batch, prefill_len, decode_len). "
            "By default those rows are merged so split results can be plotted together."
        ),
    )
    # Legacy compatibility flags: parsed and ignored so old commands keep working.
    parser.add_argument("--algo-label", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument("--format-label", action="append", default=[], help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = resolve_input_table_paths(args.csv, directory_pattern=str(args.csv_pattern))
    input_stem = build_input_stem(input_paths)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    remaps = parse_path_remaps(args.path_remap)

    log(f"[startup] resolved {len(input_paths)} input table(s)")
    for idx, input_path in enumerate(input_paths, start=1):
        log(f"  [{idx:02d}] {input_path}")
    log(f"[startup] output dir:  {outdir}")
    if remaps:
        log(f"[startup] path remaps: {remaps}")
    if args.algo_label:
        log(f"[startup] legacy --algo-label arguments detected and ignored: {args.algo_label}")
    if args.format_label:
        log(f"[startup] legacy --format-label arguments detected and ignored: {args.format_label}")

    df, source_map, per_file_maps = load_and_canonicalize_inputs(input_paths)
    log(f"[startup] merged canonical table: {len(df)} rows x {len(df.columns)} columns")

    if per_file_maps:
        log("[startup] canonical column mapping:")
        for key in ["model", "dtype", "batch", "prefill_len", "decode_len", *[spec.key for spec in BAR_SPECS]]:
            if key in source_map:
                log(f"  - {key} <- {source_map[key]}")

    if not bool(args.no_merge_duplicate_panels):
        dedupe_key_cols = [col for col in ["model", "dtype", "batch", "prefill_len", "decode_len"] if col in df.columns]
        before = len(df)
        df, duplicate_warnings = coalesce_duplicate_panel_rows(df, key_cols=dedupe_key_cols)
        after = len(df)
        if after != before:
            log(f"[startup] coalesced duplicate panels: {before} -> {after} rows")
        if duplicate_warnings:
            log(f"[startup] duplicate-panel merge notes: {len(duplicate_warnings)}")
            for msg in duplicate_warnings[:20]:
                log(f"  - {msg}")
            if len(duplicate_warnings) > 20:
                log(f"  - ... {len(duplicate_warnings) - 20} more")

    sort_cols = [col for col in PANEL_SORT_COLS if col in df.columns]
    group_cols = [col for col in GROUP_COLS_CANDIDATES if col in df.columns]
    df = df.sort_values(group_cols + sort_cols).reset_index(drop=True)

    if args.save_canonical_csv:
        canonical_csv = outdir / f"{input_stem}__canonicalized.csv"
        df.to_csv(canonical_csv, index=False)
        log(f"[startup] saved canonicalized CSV: {canonical_csv}")

        source_list = outdir / f"{input_stem}__source_tables.txt"
        with open(source_list, "w", encoding="utf-8") as f:
            for input_path in input_paths:
                f.write(str(input_path) + "\n")
        log(f"[startup] saved source table list: {source_list}")

    pie_cache: Dict[str, Optional[Dict[str, float]]] = {}

    saved_paths: List[Path] = []
    grouped_items: List[Tuple[Tuple[Any, ...], pd.DataFrame]] = []
    for group_key, group_df in df.groupby(group_cols, sort=False, dropna=False):
        group_key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        grouped_items.append((group_key_tuple, group_df.reset_index(drop=True)))

    if not grouped_items:
        raise ValueError("No rows available after canonicalization.")

    total_pages = 0
    page_plan: List[Tuple[Tuple[Any, ...], List[pd.DataFrame]]] = []
    for group_key, group_df in grouped_items:
        pages = chunk_group_df(group_df, args.max_panels_per_figure)
        total_pages += len(pages)
        page_plan.append((group_key, pages))

    log(f"[plot] preparing {total_pages} figure(s)")
    for group_key, pages in page_plan:
        title_stub = build_group_title(group_key, group_cols)
        for page_index, page_df in enumerate(pages):
            if len(pages) == 1:
                log(f"[plot] rendering figure for: {title_stub} ({len(page_df)} panels)")
            else:
                log(f"[plot] rendering figure for: {title_stub} page {page_index + 1}/{len(pages)} ({len(page_df)} panels)")
            saved_paths.extend(
                make_figure(
                    page_df,
                    group_key=group_key,
                    group_cols=group_cols,
                    outdir=outdir,
                    fig_formats=args.fig_format,
                    dpi=int(args.dpi),
                    ncols_arg=str(args.ncols),
                    share_y=bool(args.share_y),
                    show_pie=not bool(args.no_pie),
                    remaps=remaps,
                    panel_letter=(str(args.panel_letter).strip() or None),
                    pie_cache=pie_cache,
                    page_index=page_index,
                    n_pages=len(pages),
                    max_json_bytes=int(float(args.pie_json_max_mb) * 1024 * 1024),
                    verbose_pie=bool(args.verbose_pie),
                )
            )

    log(f"[done] saved {len(saved_paths)} figure(s):")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
