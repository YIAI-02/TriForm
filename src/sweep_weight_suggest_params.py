#!/usr/bin/env python3
"""
Sweep weight-suggest over JSON-level run parameters.


Examples
--------

1) Grid search over model / S / T / batch (keep format_* fixed from base JSON):

python3 sweep_weight_suggest_params.py \
  --config ./examples/weight_suggest.json \
  --mode grid \
  --model qwen:1.8b qwen:7b llama:7b \
  --prefill-len 128 256 \
  --decode-len 128 256 512 \
  --batch 1 4 \
  --outdir ./output/ws_model_len_batch \
  --resume

2) Random search over model / batch / hardware plus some format_* knobs:

python3 sweep_weight_suggest_params.py \
  --config ./examples/weight_suggest.json \
  --mode random \
  --trials 64 \
  --model qwen:7b llama:13b \
  --batch 1 4 8 \
  --hardware-json ./examples/hardware_1npu_2aim.json ./examples/hardware_1npu_4aim.json \
  --format-outer-max-iters 3 4 \
  --format-nd-margin-init 0.45 0.60 \
  --outdir ./output/ws_random \
  --resume

3) Sweep arbitrary JSON keys that are not exposed as dedicated CLI flags:

python3 sweep_weight_suggest_params.py \
  --config ./examples/weight_suggest.json \
  --mode grid \
  --sweep some_key=1,2,4 \
  --sweep nested.inner.flag=true,false \
  --set npu_backend=fast_mode \
  --outdir ./output/ws_custom \
  --resume
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from dtype_utils import normalize_dtype_token
# Common JSON-level knobs we want easy first-class support for.
COMMON_SWEEP_FIELDS: Tuple[str, ...] = (
    "model",
    "dtype",
    "batch",
    "prefill_len",
    "decode_len",
    "decode_sample_stride",
    "decode_plan_refresh_stride",
    "hardware_json",
    "tp_qkv",
    "tp_ffn",
    "algo",
    "npu_backend",
    "pim_weight_load_overlap_ratio",
    "weight_load_compute_overlap_ratio",
)

# Optional format_* knobs. These are now optional axes, not the only sweep target.
FORMAT_SWEEP_FIELDS: Tuple[str, ...] = (
    "format_outer_max_iters",
    "format_block_change_percent",
    "format_inner_max_blocks",
    "format_nd_margin_init",
    "format_nd_margin_decay",
    "format_nd_margin_min",
    "format_inner_improve_eps",
    "format_outer_stop_eps",
    "format_block_layer_span",
    "format_reload_count_mode",
)

RESULT_FIELDNAMES: Tuple[str, ...] = (
    "timestamp",
    "config_sha256",
    "run_id",
    "repeat_idx",
    "group_index",
    "group_key_json",
    "shard_index",
    "shard_count",
    "model",
    "model_family",
    "model_variant",
    "dtype",
    "batch",
    "prefill_len",
    "decode_len",
    "decode_sample_stride",
    "decode_plan_refresh_stride",
    "hardware_json",
    "tp_qkv",
    "tp_ffn",
    "algo",
    "npu_backend",
    "pim_weight_load_overlap_ratio",
    "weight_load_compute_overlap_ratio",
    "format_outer_max_iters",
    "format_block_change_percent",
    "format_inner_max_blocks",
    "format_nd_margin_init",
    "format_nd_margin_decay",
    "format_nd_margin_min",
    "format_inner_improve_eps",
    "format_outer_stop_eps",
    "format_block_layer_span",
    "format_reload_count_mode",
    "params_json",
    "objective_name",
    "objective",
    "prefill",
    "decode",
    "total",
    "search_format",
    "best_pass",
    "nd_role",
    "nd_initial",
    "nd_best",
    "nz_role",
    "nz_initial",
    "nz_best",
    "pim_opt_role",
    "pim_opt_initial",
    "pim_opt_best",
    "PD+Linear_role",
    "PD+Linear_initial",
    "PD+Linear_best",
    "PD+Dual_role",
    "PD+Dual_initial",
    "PD+Dual_best",
    "Bifocal+Linear_role",
    "Bifocal+Linear_initial",
    "Bifocal+Linear_best",
    "Bifocal+Dual_role",
    "Bifocal+Dual_initial",
    "Bifocal+Dual_best",
    "iter_gain_s",
    "iter_gain_pct",
    "returncode",
    "generated_config_json",
    "best_summary_json",
    "all_passes_json",
    "weight_format_json",
    "weight_format_compare_json",
    "log_path",
    "weight_suggest_al_log_path",
)

DEFAULT_PARALLEL_GROUP_KEYS = "model,prefill_len,decode_len,batch"


# ---------------------------------------------------------------------------
# basic utils
# ---------------------------------------------------------------------------

def _fmt_num(v: float) -> str:
    if not math.isfinite(v):
        raise ValueError(f"non-finite value: {v}")
    return f"{v:.12g}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sanitize_tag(s: str) -> str:
    s = str(s)
    s = s.replace("=", "_").replace(":", "_").replace("/", "_")
    s = s.replace(" ", "")
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    return s


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?$")


def parse_scalar_token(token: str) -> Any:
    s = str(token).strip()
    if s == "":
        return ""
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    if _INT_RE.fullmatch(s):
        try:
            return int(s)
        except Exception:
            pass
    if _FLOAT_RE.fullmatch(s):
        try:
            return float(s)
        except Exception:
            pass
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            return json.loads(s)
        except Exception:
            pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def parse_model_token(token: str) -> Dict[str, str]:
    raw = str(token).strip()
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            f"invalid --model entry: {token!r}. expected family:variant, e.g. qwen:7b"
        )
    family, variant = raw.split(":", 1)
    family = family.strip()
    variant = variant.strip()
    if not family or not variant:
        raise argparse.ArgumentTypeError(
            f"invalid --model entry: {token!r}. expected family:variant, e.g. qwen:7b"
        )
    return {"model_family": family, "model_variant": variant, "model": f"{family}:{variant}"}


# ---------------------------------------------------------------------------
# dotted-key config helpers
# ---------------------------------------------------------------------------

def set_dotted(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in str(dotted_key).split(".") if p]
    if not parts:
        raise ValueError(f"invalid dotted key: {dotted_key!r}")
    cur: Dict[str, Any] = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def get_dotted(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    parts = [p for p in str(dotted_key).split(".") if p]
    cur: Any = cfg
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def parse_group_keys_csv(text: str) -> List[str]:
    raw = str(text or "").strip()
    if raw == "":
        return []
    out: List[str] = []
    seen: set[str] = set()
    for tok in raw.split(","):
        key = str(tok).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def model_from_cfg(cfg: Dict[str, Any]) -> str:
    family = cfg.get("model_family") if isinstance(cfg, dict) else None
    variant = cfg.get("model_variant") if isinstance(cfg, dict) else None
    if family and variant:
        return f"{family}:{variant}"
    if family not in (None, ""):
        return str(family)
    if isinstance(cfg, dict):
        raw_model = cfg.get("model", "")
        if raw_model not in (None, ""):
            return str(raw_model)
    return ""


def group_value_from_cfg(cfg: Dict[str, Any], key: str) -> Any:
    if key == "model":
        return model_from_cfg(cfg)
    return get_dotted(cfg, key, "")


def build_group_values(cfg: Dict[str, Any], group_keys: Sequence[str]) -> Dict[str, Any]:
    return {str(key): group_value_from_cfg(cfg, str(key)) for key in group_keys}


def assign_group_indices(run_specs: List[Dict[str, Any]], group_keys: Sequence[str]) -> Tuple[int, List[Dict[str, Any]]]:
    if not run_specs:
        return 0, []
    if not group_keys:
        group_key_json = stable_json_dumps({})
        for spec in run_specs:
            spec["group_index"] = 0
            spec["group_key_json"] = group_key_json
            spec["group_values"] = {}
        return 1, [{"group_index": 0, "group_key_json": group_key_json, "group_values": {}}]

    group_index_by_key: Dict[str, int] = {}
    group_rows: List[Dict[str, Any]] = []
    for spec in run_specs:
        group_values = build_group_values(spec["effective_cfg"], group_keys)
        group_key_json = stable_json_dumps(group_values)
        group_index = group_index_by_key.get(group_key_json)
        if group_index is None:
            group_index = len(group_rows)
            group_index_by_key[group_key_json] = group_index
            group_rows.append({
                "group_index": group_index,
                "group_key_json": group_key_json,
                "group_values": group_values,
            })
        spec["group_index"] = int(group_index)
        spec["group_key_json"] = group_key_json
        spec["group_values"] = group_values
    return len(group_rows), group_rows


def shard_accepts_group(group_index: int, shard_index: int, shard_count: int) -> bool:
    if shard_count <= 1:
        return True
    return int(group_index) % int(shard_count) == int(shard_index)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# arg parsing for generic sweeps
# ---------------------------------------------------------------------------

def parse_set_arg(item: str) -> Tuple[str, Any]:
    if "=" not in str(item):
        raise argparse.ArgumentTypeError(f"invalid --set item: {item!r}; expected key=value")
    key, rhs = str(item).split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"invalid --set item: {item!r}; empty key")
    return key, parse_scalar_token(rhs)


def _split_csv_values(rhs: str) -> List[str]:
    # Simple CSV-like split for shell-friendly values. For values that themselves
    # contain commas, users can pass JSON list syntax, e.g. key=["a,b","c"].
    return [tok.strip() for tok in str(rhs).split(",") if tok.strip() != ""]


def parse_sweep_arg(item: str) -> Tuple[str, List[Any]]:
    if "=" not in str(item):
        raise argparse.ArgumentTypeError(f"invalid --sweep item: {item!r}; expected key=v1,v2,...")
    key, rhs = str(item).split("=", 1)
    key = key.strip()
    rhs = rhs.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"invalid --sweep item: {item!r}; empty key")
    if rhs == "":
        raise argparse.ArgumentTypeError(f"invalid --sweep item: {item!r}; empty value list")

    vals: List[Any]
    if rhs.startswith("[") and rhs.endswith("]"):
        try:
            parsed = json.loads(rhs)
        except Exception as exc:  # pragma: no cover - user input errors
            raise argparse.ArgumentTypeError(f"invalid JSON list in --sweep {item!r}: {exc}") from exc
        if not isinstance(parsed, list):
            raise argparse.ArgumentTypeError(f"invalid --sweep item: {item!r}; JSON value must be a list")
        vals = list(parsed)
    else:
        vals = [parse_scalar_token(tok) for tok in _split_csv_values(rhs)]

    if not vals:
        raise argparse.ArgumentTypeError(f"invalid --sweep item: {item!r}; empty value list")
    return key, vals


# ---------------------------------------------------------------------------
# result parsing
# ---------------------------------------------------------------------------

def parse_best_summary(best_summary_path: Path) -> Optional[Dict[str, Any]]:
    if not best_summary_path.exists():
        return None
    try:
        with best_summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        best_times = data.get("best_times", {}) or {}
        search_format = data.get("search_format")
        if search_format is None:
            search_format = data.get("best_start_mode") or data.get("start_mode")
        return {
            "prefill": float(best_times.get("prefill", 0.0) or 0.0),
            "decode": float(best_times.get("decode", 0.0) or 0.0),
            "total": float(best_times.get("total", 0.0) or 0.0),
            "search_format": str(search_format or ""),
            "best_pass": int(data.get("best_pass", -1)),
        }
    except Exception:
        return None


def parse_compare_json(compare_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "nd_role": "",
        "nd_initial": "",
        "nd_best": "",
        "nz_role": "",
        "nz_initial": "",
        "nz_best": "",
        "pim_opt_role": "",
        "pim_opt_initial": "",
        "pim_opt_best": "",
        "PD+Linear_role": "",
        "PD+Linear_initial": "",
        "PD+Linear_best": "",
        "PD+Dual_role": "",
        "PD+Dual_initial": "",
        "PD+Dual_best": "",
        "Bifocal+Linear_role": "",
        "Bifocal+Linear_initial": "",
        "Bifocal+Linear_best": "",
        "Bifocal+Dual_role": "",
        "Bifocal+Dual_initial": "",
        "Bifocal+Dual_best": "",
    }
    if not compare_path.exists():
        return out

    try:
        with compare_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        def _fill(prefix: str, row: Dict[str, Any]) -> None:
            out[f"{prefix}_role"] = str(row.get("role", ""))
            out[f"{prefix}_initial"] = float(row.get("initial_total_s", 0.0) or 0.0)
            out[f"{prefix}_best"] = float(row.get("best_total_s", 0.0) or 0.0)

        rows = data.get("rows")
        if isinstance(rows, list):
            by_exp = {str(row.get("experiment_id", "")): row for row in rows if isinstance(row, dict)}
            by_fmt = {str(row.get("format", "")): row for row in rows if isinstance(row, dict)}

            if "search_nd_tuned" in by_exp:
                _fill("nd", by_exp["search_nd_tuned"])
            elif "ND" in by_fmt:
                _fill("nd", by_fmt["ND"])

            if "NZ" in by_fmt:
                _fill("nz", by_fmt["NZ"])
            if "PIM-OPT" in by_fmt:
                _fill("pim_opt", by_fmt["PIM-OPT"])

            if "PD+Linear" in by_exp:
                _fill("PD+Linear", by_exp["PD+Linear"])
            if "PD+Dual" in by_exp:
                _fill("PD+Dual", by_exp["PD+Dual"])
            if "Bifocal+Linear" in by_exp:
                _fill("Bifocal+Linear", by_exp["Bifocal+Linear"])
            if "Bifocal+Dual" in by_exp:
                _fill("Bifocal+Dual", by_exp["Bifocal+Dual"])
            return out

        # Backward compatibility with older compare payloads.
        old_rows = data.get("start_modes")
        if isinstance(old_rows, list):
            by_key = {str(row.get("start_mode", "")): row for row in old_rows if isinstance(row, dict)}
            if "ND" in by_key:
                row = dict(by_key["ND"])
                row["role"] = "search"
                _fill("nd", row)
            if "NZ" in by_key:
                row = dict(by_key["NZ"])
                row["role"] = "compare"
                _fill("nz", row)
            if "PIM-OPT" in by_key:
                row = dict(by_key["PIM-OPT"])
                row["role"] = "compare"
                _fill("pim_opt", row)
    except Exception:
        return out
    return out


# ---------------------------------------------------------------------------
# csv helpers
# ---------------------------------------------------------------------------

def append_result(results_csv: Path, row: Dict[str, Any]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_FIELDNAMES))
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDNAMES})


def load_done_keys(results_csv: Path, *, config_sha256: str) -> set[Tuple[str, int, str]]:
    done: set[Tuple[str, int, str]] = set()
    if not results_csv.exists():
        return done
    try:
        with results_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("config_sha256", "")) != str(config_sha256):
                    continue
                params_json = str(row.get("params_json", "") or "")
                rep = int(row.get("repeat_idx", "1"))
                done.add((str(config_sha256), rep, params_json))
    except Exception:
        pass
    return done


def save_best_json(best_path: Path, payload: Dict[str, Any]) -> None:
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# launch helpers
# ---------------------------------------------------------------------------

def run_command(cmd: Sequence[str], *, workdir: Path, env: Dict[str, str]) -> Tuple[int, str]:
    p = subprocess.run(
        list(cmd),
        cwd=str(workdir),
        env=env,
        capture_output=True,
        text=True,
    )
    out = ""
    if p.stdout:
        out += p.stdout
    if p.stderr:
        out += ("\n" if out else "") + p.stderr
    return p.returncode, out


# ---------------------------------------------------------------------------
# combo builders
# ---------------------------------------------------------------------------

def _normalize_axis(values: Optional[Iterable[Any]]) -> Optional[List[Any]]:
    if values is None:
        return None
    out = list(values)
    if len(out) == 0:
        return None
    return out


def collect_fixed_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    fixed: Dict[str, Any] = {}
    for item in args.set or []:
        key, value = item
        if key == "model":
            model_info = parse_model_token(str(value))
            fixed["model_family"] = model_info["model_family"]
            fixed["model_variant"] = model_info["model_variant"]
            continue
        fixed[key] = value
    return fixed


def collect_axes(args: argparse.Namespace) -> Dict[str, List[Any]]:
    axes: Dict[str, List[Any]] = {}

    def add_axis(name: str, values: Optional[Iterable[Any]]) -> None:
        norm = _normalize_axis(values)
        if norm is not None:
            axes[name] = norm

    add_axis("model", args.model)
    add_axis("dtype", args.dtype)
    add_axis("batch", args.batch)
    add_axis("prefill_len", args.prefill_len)
    add_axis("decode_len", args.decode_len)
    add_axis("decode_sample_stride", args.decode_sample_stride)
    add_axis("decode_plan_refresh_stride", args.decode_plan_refresh_stride)
    add_axis("hardware_json", args.hardware_json)
    add_axis("tp_qkv", args.tp_qkv)
    add_axis("tp_ffn", args.tp_ffn)
    add_axis("algo", args.algo)
    add_axis("npu_backend", args.npu_backend)
    add_axis("pim_weight_load_overlap_ratio", args.pim_weight_load_overlap_ratio)
    add_axis("weight_load_compute_overlap_ratio", args.weight_load_compute_overlap_ratio)

    add_axis("format_outer_max_iters", args.format_outer_max_iters)
    add_axis("format_block_change_percent", args.format_block_change_percent)
    add_axis("format_inner_max_blocks", args.format_inner_max_blocks)
    add_axis("format_nd_margin_init", args.format_nd_margin_init)
    add_axis("format_nd_margin_decay", args.format_nd_margin_decay)
    add_axis("format_nd_margin_min", args.format_nd_margin_min)
    add_axis("format_inner_improve_eps", args.format_inner_improve_eps)
    add_axis("format_outer_stop_eps", args.format_outer_stop_eps)
    add_axis("format_block_layer_span", args.format_block_layer_span)
    add_axis("format_reload_count_mode", args.format_reload_count_mode)

    for key, vals in args.sweep or []:
        if key in axes:
            raise ValueError(f"duplicate sweep axis for key {key!r}")
        axes[key] = list(vals)

    return axes


def build_combos(axes: Dict[str, List[Any]], *, mode: str, trials: int, seed: int) -> List[Dict[str, Any]]:
    if not axes:
        return [{}]

    names = list(axes.keys())
    value_lists = [list(axes[name]) for name in names]
    combos: List[Dict[str, Any]] = [{name: value for name, value in zip(names, values)} for values in itertools.product(*value_lists)]

    if mode == "grid":
        return combos

    rnd = random.Random(seed)
    rnd.shuffle(combos)
    if trials <= 0 or trials >= len(combos):
        return combos
    return combos[: int(trials)]


# ---------------------------------------------------------------------------
# config assembly
# ---------------------------------------------------------------------------

def apply_overrides(base_cfg: Dict[str, Any], fixed_overrides: Dict[str, Any], combo: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = copy.deepcopy(base_cfg)
    applied: Dict[str, Any] = {}

    def _apply_pair(key: str, value: Any) -> None:
        if key == "model":
            model_info = parse_model_token(str(value))
            cfg["model_family"] = model_info["model_family"]
            cfg["model_variant"] = model_info["model_variant"]
            applied["model"] = model_info["model"]
            applied["model_family"] = model_info["model_family"]
            applied["model_variant"] = model_info["model_variant"]
            return
        if key == "dtype":
            value = normalize_dtype_token(value, default="fp16")
        set_dotted(cfg, key, value)
        applied[key] = value

    for key, value in fixed_overrides.items():
        _apply_pair(key, value)
    for key, value in combo.items():
        _apply_pair(key, value)

    return cfg, applied


def effective_summary_fields(cfg: Dict[str, Any], params_json: str) -> Dict[str, Any]:
    family = cfg.get("model_family")
    variant = cfg.get("model_variant")
    model = model_from_cfg(cfg)

    row: Dict[str, Any] = {
        "model": model,
        "model_family": family if family is not None else "",
        "model_variant": variant if variant is not None else "",
        "dtype": (normalize_dtype_token(cfg.get("dtype"), default="fp16") if cfg.get("dtype") not in (None, "") else ""),
        "batch": cfg.get("batch", ""),
        "prefill_len": cfg.get("prefill_len", ""),
        "decode_len": cfg.get("decode_len", ""),
        "decode_sample_stride": cfg.get("decode_sample_stride", ""),
        "decode_plan_refresh_stride": cfg.get("decode_plan_refresh_stride", ""),
        "hardware_json": cfg.get("hardware_json", ""),
        "tp_qkv": cfg.get("tp_qkv", ""),
        "tp_ffn": cfg.get("tp_ffn", ""),
        "algo": cfg.get("algo", ""),
        "npu_backend": cfg.get("npu_backend", ""),
        "pim_weight_load_overlap_ratio": cfg.get("pim_weight_load_overlap_ratio", ""),
        "weight_load_compute_overlap_ratio": cfg.get("weight_load_compute_overlap_ratio", ""),
        "format_outer_max_iters": cfg.get("format_outer_max_iters", ""),
        "format_block_change_percent": cfg.get("format_block_change_percent", ""),
        "format_inner_max_blocks": cfg.get("format_inner_max_blocks", ""),
        "format_nd_margin_init": cfg.get("format_nd_margin_init", ""),
        "format_nd_margin_decay": cfg.get("format_nd_margin_decay", ""),
        "format_nd_margin_min": cfg.get("format_nd_margin_min", ""),
        "format_inner_improve_eps": cfg.get("format_inner_improve_eps", ""),
        "format_outer_stop_eps": cfg.get("format_outer_stop_eps", ""),
        "format_block_layer_span": cfg.get("format_block_layer_span", ""),
        "format_reload_count_mode": cfg.get("format_reload_count_mode", ""),
        "params_json": params_json,
    }
    return row


def build_run_specs(
    *,
    base_cfg: Dict[str, Any],
    fixed_overrides: Dict[str, Any],
    combos: Sequence[Dict[str, Any]],
    group_keys: Sequence[str],
) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    run_specs: List[Dict[str, Any]] = []
    for combo in combos:
        effective_cfg, applied_params = apply_overrides(base_cfg, fixed_overrides, combo)
        params_json = stable_json_dumps(applied_params)
        run_specs.append(
            {
                "combo": dict(combo),
                "effective_cfg": effective_cfg,
                "applied_params": applied_params,
                "params_json": params_json,
            }
        )

    group_count, group_rows = assign_group_indices(run_specs, group_keys)
    return run_specs, group_count, group_rows


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sweep weight-suggest over JSON config parameters")
    ap.add_argument("--config", required=True, help="path to the base weight-suggest JSON config")
    ap.add_argument("--main", default="./main.py", help="path to main.py")
    ap.add_argument("--python", default=sys.executable, help="python executable used to launch main.py")
    ap.add_argument("--workdir", default=".", help="working directory")

    ap.add_argument("--mode", choices=["grid", "random"], default="random")
    ap.add_argument("--objective", choices=["total", "decode", "prefill"], default="total")

    # Common JSON-level axes.
    ap.add_argument("--model", nargs="*", default=None, help="candidate family:variant pairs, e.g. qwen:7b llama:13b")
    ap.add_argument("--dtype", nargs="*", default=None, help="candidate dtype values")
    ap.add_argument("--batch", type=int, nargs="*", default=None, help="candidate batch values")
    ap.add_argument("--prefill-len", dest="prefill_len", type=int, nargs="*", default=None, help="candidate prefill_len values")
    ap.add_argument("--decode-len", dest="decode_len", type=int, nargs="*", default=None, help="candidate decode_len values")
    ap.add_argument("--decode-sample-stride", dest="decode_sample_stride", type=int, nargs="*", default=None, help="candidate decode_sample_stride values")
    ap.add_argument("--decode-plan-refresh-stride", dest="decode_plan_refresh_stride", type=int, nargs="*", default=None, help="candidate decode_plan_refresh_stride values")
    ap.add_argument("--hardware-json", dest="hardware_json", nargs="*", default=None, help="candidate hardware_json paths")
    ap.add_argument("--tp-qkv", dest="tp_qkv", type=int, nargs="*", default=None, help="candidate tp_qkv values")
    ap.add_argument("--tp-ffn", dest="tp_ffn", type=int, nargs="*", default=None, help="candidate tp_ffn values")
    ap.add_argument("--algo", nargs="*", default=None, help="candidate algo values")
    ap.add_argument("--npu-backend", dest="npu_backend", nargs="*", default=None, help="candidate npu_backend values")
    ap.add_argument("--pim-weight-load-overlap-ratio", dest="pim_weight_load_overlap_ratio", type=float, nargs="*", default=None, help="candidate PIM_WEIGHT_LOAD_OVERLAP_RATIO values in [0,1]")
    ap.add_argument("--weight-load-compute-overlap-ratio", dest="weight_load_compute_overlap_ratio", type=float, nargs="*", default=None, help="candidate WEIGHT_LOAD_COMPUTE_OVERLAP_RATIO values in [0,1]")

    # Optional format_* axes.
    ap.add_argument("--format-outer-max-iters", dest="format_outer_max_iters", type=int, nargs="*", default=None, help="candidate format_outer_max_iters values")
    ap.add_argument("--format-block-change-percent", dest="format_block_change_percent", type=float, nargs="*", default=None, help="candidate format_block_change_percent values")
    ap.add_argument("--format-inner-max-blocks", dest="format_inner_max_blocks", type=int, nargs="*", default=None, help="candidate format_inner_max_blocks values")
    ap.add_argument("--format-nd-margin-init", dest="format_nd_margin_init", type=float, nargs="*", default=None, help="candidate format_nd_margin_init values")
    ap.add_argument("--format-nd-margin-decay", dest="format_nd_margin_decay", type=float, nargs="*", default=None, help="candidate format_nd_margin_decay values")
    ap.add_argument("--format-nd-margin-min", dest="format_nd_margin_min", type=float, nargs="*", default=None, help="candidate format_nd_margin_min values")
    ap.add_argument("--format-inner-improve-eps", dest="format_inner_improve_eps", type=float, nargs="*", default=None, help="candidate format_inner_improve_eps values")
    ap.add_argument("--format-outer-stop-eps", dest="format_outer_stop_eps", type=float, nargs="*", default=None, help="candidate format_outer_stop_eps values")
    ap.add_argument("--format-block-layer-span", dest="format_block_layer_span", type=int, nargs="*", default=None, help="candidate format_block_layer_span values")
    ap.add_argument("--format-reload-count-mode", dest="format_reload_count_mode", nargs="*", default=None, help="candidate format_reload_count_mode values")

    # Generic JSON-level controls.
    ap.add_argument("--set", type=parse_set_arg, action="append", default=None, help="fixed override applied to every run: key=value")
    ap.add_argument("--sweep", type=parse_sweep_arg, action="append", default=None, help="additional sweep axis: key=v1,v2,... ; dotted keys supported")

    ap.add_argument("--trials", type=int, default=128, help="random mode: number of sampled combinations")
    ap.add_argument("--seed", type=int, default=0, help="random mode seed")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each combo N times")
    ap.add_argument("--max-runs", type=int, default=0, help="cap total runs (0=unlimited)")
    ap.add_argument("--outdir", default="./output/sweep_weight_suggest", help="output directory for logs/results")
    ap.add_argument("--parallel-group-keys", default="", help=f"comma-separated keys that stay together on one worker, e.g. {DEFAULT_PARALLEL_GROUP_KEYS}")
    ap.add_argument("--group-shard-index", type=int, default=0, help="process only dispatch groups assigned to this shard index")
    ap.add_argument("--group-shard-count", type=int, default=1, help="number of dispatch-group shards/workers")
    ap.add_argument("--resume", action="store_true", help="skip combos already in results.csv for the same base config hash")
    ap.add_argument("--debug", action="store_true", help="pass --debug to main.py weight-suggest")
    return ap


# ---------------------------------------------------------------------------
# best-result bookkeeping
# ---------------------------------------------------------------------------

def build_best_payload(
    *,
    objective_name: str,
    objective: float,
    applied_params: Dict[str, Any],
    cfg_row: Dict[str, Any],
    summary: Dict[str, Any],
    compare: Dict[str, Any],
    config_sha256: str,
    group_index: int,
    group_key_json: str,
    shard_index: int,
    shard_count: int,
    generated_config_json: Path,
    best_summary_json: Path,
    all_passes_json: Path,
    weight_format_json: Path,
    weight_format_compare_json: Path,
    log_path: Path,
    weight_suggest_al_log_path: Path,
) -> Dict[str, Any]:
    return {
        "config_sha256": config_sha256,
        "group_index": int(group_index),
        "group_key_json": str(group_key_json),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "objective_name": objective_name,
        "objective": objective,
        "applied_params": applied_params,
        "effective_config_fields": cfg_row,
        "search_format": summary.get("search_format", ""),
        "best_pass": int(summary.get("best_pass", -1)),
        "prefill": float(summary.get("prefill", 0.0) or 0.0),
        "decode": float(summary.get("decode", 0.0) or 0.0),
        "total": float(summary.get("total", 0.0) or 0.0),
        **compare,
        "generated_config_json": str(generated_config_json),
        "best_summary_json": str(best_summary_json),
        "all_passes_json": str(all_passes_json),
        "weight_format_json": str(weight_format_json),
        "weight_format_compare_json": str(weight_format_compare_json),
        "log_path": str(log_path),
        "weight_suggest_al_log_path": str(weight_suggest_al_log_path),
    }


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main() -> int:
    ap = make_parser()
    args, extra = ap.parse_known_args()

    workdir = Path(args.workdir).resolve()
    config_path = (workdir / args.config).resolve()
    main_path = (workdir / args.main).resolve()
    outdir = Path(args.outdir).resolve()
    results_csv = outdir / "results.csv"
    best_json = outdir / "best_result.json"

    if not config_path.exists():
        print(f"[err] config not found: {config_path}", file=sys.stderr)
        return 2
    if not main_path.exists():
        print(f"[err] main.py not found: {main_path}", file=sys.stderr)
        return 2
    if int(args.group_shard_count) < 1:
        print(f"[err] --group-shard-count must be >= 1, got {args.group_shard_count}", file=sys.stderr)
        return 2
    if int(args.group_shard_index) < 0 or int(args.group_shard_index) >= int(args.group_shard_count):
        print(
            f"[err] --group-shard-index must be in [0, {int(args.group_shard_count) - 1}], got {args.group_shard_index}",
            file=sys.stderr,
        )
        return 2

    base_config_text = _read_text(config_path)
    config_sha256 = _sha256_text(base_config_text)
    try:
        base_cfg = json.loads(base_config_text)
    except Exception as exc:
        print(f"[err] failed to parse config JSON: {config_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(base_cfg, dict):
        print(f"[err] config JSON must be an object/dict: {config_path}", file=sys.stderr)
        return 2

    fixed_overrides = collect_fixed_overrides(args)
    axes = collect_axes(args)
    combos = build_combos(axes, mode=args.mode, trials=args.trials, seed=args.seed)

    group_keys = parse_group_keys_csv(args.parallel_group_keys)
    if int(args.group_shard_count) > 1 and not group_keys:
        group_keys = parse_group_keys_csv(DEFAULT_PARALLEL_GROUP_KEYS)

    run_specs, total_group_count, group_rows = build_run_specs(
        base_cfg=base_cfg,
        fixed_overrides=fixed_overrides,
        combos=combos,
        group_keys=group_keys,
    )
    total_combo_count = len(run_specs)

    if int(args.group_shard_count) > 1:
        selected_run_specs = [
            spec
            for spec in run_specs
            if shard_accepts_group(int(spec.get("group_index", 0)), int(args.group_shard_index), int(args.group_shard_count))
        ]
    else:
        selected_run_specs = list(run_specs)

    selected_group_indices = sorted({int(spec.get("group_index", 0)) for spec in selected_run_specs})
    selected_group_rows = [group_rows[idx] for idx in selected_group_indices if 0 <= idx < len(group_rows)]
    selected_combo_count = len(selected_run_specs)

    outdir.mkdir(parents=True, exist_ok=True)
    dispatch_meta = {
        "config": str(config_path),
        "config_sha256": config_sha256,
        "main": str(main_path),
        "mode": args.mode,
        "objective": args.objective,
        "repeat": int(args.repeat),
        "total_combo_count": int(total_combo_count),
        "selected_combo_count": int(selected_combo_count),
        "parallel_group_keys": list(group_keys),
        "total_group_count": int(total_group_count),
        "selected_group_count": int(len(selected_group_rows)),
        "shard_index": int(args.group_shard_index),
        "shard_count": int(args.group_shard_count),
        "selected_groups": selected_group_rows,
    }
    write_json(outdir / "dispatch_meta.json", dispatch_meta)

    done_keys = load_done_keys(results_csv, config_sha256=config_sha256) if args.resume else set()

    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")

    best: Dict[str, Any] = {
        "objective_name": args.objective,
        "objective": float("inf"),
        "applied_params": None,
        "search_format": None,
        "best_pass": None,
        "prefill": None,
        "decode": None,
        "total": None,
        "group_index": None,
        "group_key_json": None,
        "shard_index": int(args.group_shard_index),
        "shard_count": int(args.group_shard_count),
        "generated_config_json": None,
        "best_summary_json": None,
        "all_passes_json": None,
        "weight_format_json": None,
        "weight_format_compare_json": None,
        "log_path": None,
        "weight_suggest_al_log_path": None,
    }

    print(f"[info] mode={args.mode} combos={total_combo_count} repeat={args.repeat} objective={args.objective}")
    print(f"[info] config={config_path}")
    print(f"[info] config_sha256={config_sha256}")
    print(f"[info] main={main_path}")
    print(f"[info] workdir={workdir}")
    print(f"[info] outdir={outdir}")
    if fixed_overrides:
        print(f"[info] fixed overrides: {stable_json_dumps(fixed_overrides)}")
    if axes:
        axis_sizes = {k: len(v) for k, v in axes.items()}
        print(f"[info] sweep axes: {stable_json_dumps(axis_sizes)}")
    if group_keys:
        print(
            f"[info] dispatch groups keys={stable_json_dumps(group_keys)} total_groups={total_group_count} "
            f"selected_groups={len(selected_group_rows)} shard={int(args.group_shard_index)}/{int(args.group_shard_count)}"
        )
        if selected_group_rows:
            preview = stable_json_dumps(selected_group_rows[: min(5, len(selected_group_rows))])
            print(f"[info] selected group preview: {preview}")
    if extra:
        print(f"[info] ignored extra args: {' '.join(extra)}")

    if not selected_run_specs:
        print("[info] no combos assigned to this shard.")
        return 0

    run_id = 0
    executed = 0
    stop_requested = False

    for spec in selected_run_specs:
        effective_cfg_template = spec["effective_cfg"]
        applied_params = spec["applied_params"]
        params_json = spec["params_json"]
        group_index = int(spec.get("group_index", 0))
        group_key_json = str(spec.get("group_key_json", "{}"))

        for rep in range(1, int(args.repeat) + 1):
            done_key = (config_sha256, rep, params_json)
            if done_key in done_keys:
                continue

            if args.max_runs and executed >= int(args.max_runs):
                print("[info] reached --max-runs cap, stop.")
                stop_requested = True
                break

            run_id += 1
            executed += 1

            effective_cfg = copy.deepcopy(effective_cfg_template)
            model = model_from_cfg(effective_cfg)
            prefill_len = effective_cfg.get("prefill_len", "")
            decode_len = effective_cfg.get("decode_len", "")
            batch = effective_cfg.get("batch", "")
            tag_core = f"g={group_index}_model={model}_S={prefill_len}_T={decode_len}_b={batch}_r={rep}"
            short_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()[:10]
            run_dir = outdir / "runs" / f"{run_id:06d}_{sanitize_tag(tag_core)}_{short_hash}"
            run_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            generated_config_path = run_dir / "generated_config.json"
            log_path = run_dir / "run.log"
            best_summary_path = run_dir / "best_summary.json"
            all_passes_path = run_dir / "all_passes.json"
            weight_format_path = run_dir / "weight_storage_suggestion.json"
            compare_path = run_dir / "weight_format_compare.json"
            weight_suggest_al_log_path = artifacts_dir / "weight_suggest_al_debug.txt"

            effective_cfg["result_dir"] = str(artifacts_dir)
            effective_cfg["all_passes_json"] = str(all_passes_path)
            effective_cfg["best_summary_json"] = str(best_summary_path)
            effective_cfg["weight_format_json"] = str(weight_format_path)
            effective_cfg["weight_format_compare_json"] = str(compare_path)
            effective_cfg["weight_suggest_al_log_path"] = str(weight_suggest_al_log_path)

            with generated_config_path.open("w", encoding="utf-8") as f:
                json.dump(effective_cfg, f, ensure_ascii=False, indent=2)

            cmd = [
                str(args.python),
                str(main_path),
                "weight-suggest",
                "--config",
                str(generated_config_path),
            ]
            if args.debug:
                cmd.append("--debug")

            ts = dt.datetime.now().isoformat(timespec="seconds")
            print(
                f"\n=== [{run_id}] {ts} group={group_index} model={model} S={prefill_len} T={decode_len} b={batch} rep={rep} ==="
            )
            if applied_params:
                print(f"[cfg] overrides={params_json}")
            if group_key_json not in ("", "{}"):
                print(f"[cfg] dispatch_group={group_key_json}")

            rc, out = run_command(cmd, workdir=workdir, env=env)

            with log_path.open("w", encoding="utf-8") as f:
                f.write("# cmd:\n")
                f.write("#   " + " ".join(cmd) + "\n")
                f.write(f"# time: {ts}\n")
                f.write(f"# config_sha256: {config_sha256}\n")
                f.write(f"# group_index: {group_index}\n")
                f.write(f"# group_key_json: {group_key_json}\n")
                f.write("# applied_params:\n")
                f.write("#   " + params_json + "\n\n")
                f.write(out)

            summary = parse_best_summary(best_summary_path)
            compare = parse_compare_json(compare_path)
            cfg_row = effective_summary_fields(effective_cfg, params_json)

            row: Dict[str, Any] = {
                "timestamp": ts,
                "config_sha256": config_sha256,
                "run_id": run_id,
                "repeat_idx": rep,
                "group_index": group_index,
                "group_key_json": group_key_json,
                "shard_index": int(args.group_shard_index),
                "shard_count": int(args.group_shard_count),
                **cfg_row,
                "objective_name": args.objective,
                "objective": "",
                "prefill": "",
                "decode": "",
                "total": "",
                "search_format": "",
                "best_pass": "",
                **compare,
                "iter_gain_s": "",
                "iter_gain_pct": "",
                "returncode": rc,
                "generated_config_json": str(generated_config_path),
                "best_summary_json": str(best_summary_path),
                "all_passes_json": str(all_passes_path),
                "weight_format_json": str(weight_format_path),
                "weight_format_compare_json": str(compare_path),
                "log_path": str(log_path),
                "weight_suggest_al_log_path": str(weight_suggest_al_log_path),
            }

            if summary is None:
                append_result(results_csv, row)
                print(f"[warn] cannot parse best_summary.json. rc={rc} log={log_path}")
                if rc != 0:
                    print("[warn] command failed (non-zero). continue.")
                continue

            obj = float(summary[args.objective])
            try:
                nd_initial = float(compare.get("nd_initial", 0.0) or 0.0)
                nd_best = float(compare.get("nd_best", 0.0) or 0.0)
                iter_gain_s = float(nd_initial - nd_best)
                iter_gain_pct = float((iter_gain_s / nd_initial) * 100.0) if nd_initial > 0.0 else 0.0
            except Exception:
                iter_gain_s = 0.0
                iter_gain_pct = 0.0
            row.update(
                {
                    "objective": obj,
                    "prefill": float(summary["prefill"]),
                    "decode": float(summary["decode"]),
                    "total": float(summary["total"]),
                    "search_format": str(summary.get("search_format", "")),
                    "best_pass": int(summary.get("best_pass", -1)),
                    "iter_gain_s": float(iter_gain_s),
                    "iter_gain_pct": float(iter_gain_pct),
                }
            )
            append_result(results_csv, row)

            print(
                f"[ok] search={summary.get('search_format', '')} pass={int(summary.get('best_pass', -1))} "
                f"prefill={float(summary['prefill']):.4f} decode={float(summary['decode']):.4f} total={float(summary['total']):.4f} "
                f"-> objective({args.objective})={obj:.4f}"
            )
            print(
                f"[cmp] ND-search({compare.get('nd_role', '')})={compare.get('nd_best', '')} "
                f"PD+Linear({compare.get('PD+Linear_role', '')})={compare.get('PD+Linear_best', '')} "
                f"PD+Dual({compare.get('PD+Dual_role', '')})={compare.get('PD+Dual_best', '')} "
                f"Bifocal+Linear({compare.get('Bifocal+Linear_role', '')})={compare.get('Bifocal+Linear_best', '')} "
                f"Bifocal+Dual({compare.get('Bifocal+Dual_role', '')})={compare.get('Bifocal+Dual_best', '')} "
                f"iter_gain_s={row.get('iter_gain_s', '')} iter_gain_pct={row.get('iter_gain_pct', '')}"
            )

            if obj < float(best["objective"]):
                best_payload = build_best_payload(
                    objective_name=args.objective,
                    objective=obj,
                    applied_params=applied_params,
                    cfg_row=cfg_row,
                    summary=summary,
                    compare=compare,
                    config_sha256=config_sha256,
                    group_index=group_index,
                    group_key_json=group_key_json,
                    shard_index=int(args.group_shard_index),
                    shard_count=int(args.group_shard_count),
                    generated_config_json=generated_config_path,
                    best_summary_json=best_summary_path,
                    all_passes_json=all_passes_path,
                    weight_format_json=weight_format_path,
                    weight_format_compare_json=compare_path,
                    log_path=log_path,
                    weight_suggest_al_log_path=weight_suggest_al_log_path,
                )
                best.update(best_payload)
                save_best_json(best_json, best)
                print(
                    f"[best] objective={obj:.6f} search={best['search_format']} pass={best['best_pass']} "
                    f"group={group_index} params={stable_json_dumps(applied_params)}"
                )
                print(f"[best] log: {best['log_path']}")

        if stop_requested:
            break

    print(f"\n[info] results at {results_csv}")
    print(f"[info] dispatch meta at {outdir / 'dispatch_meta.json'}")
    if best["applied_params"] is not None:
        print(f"[info] best summary at {best_json}")
        print(
            f"[info] best objective={best['objective']:.6g} search={best['search_format']} "
            f"pass={best['best_pass']} group={best['group_index']} params={stable_json_dumps(best['applied_params'])}"
        )
        print(f"[info] best log={best['log_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
