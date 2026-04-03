#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU/PIM-only verification driver aligned with the current simulator.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any
import os
import re
import json
import sys
import time
import hashlib
import subprocess
import tempfile
import shutil
import uuid
import importlib
import importlib.util
from types import SimpleNamespace
import yaml
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

_COMM_OPS = {
    # KV cache writebacks
    "k_write",
    "v_write",
    "kv_write",

    # Collectives / comm
    "allreduce",
    "all_reduce",
    "reduce",
    "scatter",
    "reducescatter",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "broadcast",
    "alltoall",
    "all_to_all",

    # Point-to-point / misc
    "send",
    "recv",
    "identity",
}


# cxl model is optional
try:
    import cxl_latency  # type: ignore
except Exception:
    cxl_latency = None  # type: ignore
# ==========================================================
# Path helpers (workdir-robust)
# ==========================================================

def _script_dir() -> Path:
    return Path(__file__).resolve().parent

def _add_sys_path(p: Path) -> None:
    p = p.resolve()
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

def resolve_existing_path(path_str: str, *, extra_roots: Optional[List[Path]] = None) -> Path:
    p = Path(path_str).expanduser()
    if p.exists():
        return p.resolve()

    cand = (Path.cwd() / p)
    if cand.exists():
        return cand.resolve()

    cand = (_script_dir().parent / p)
    if cand.exists():
        return cand.resolve()

    if extra_roots:
        for r in extra_roots:
            cand = (r / p)
            if cand.exists():
                return cand.resolve()

    raise FileNotFoundError(f"Path not found: {path_str}")


# ==========================================================
# CENT / aim_sim
# ==========================================================

def find_cent_sim_root(start: Optional[Path] = None) -> Optional[Path]:
    here = (start or _script_dir()).resolve()
    for p in [here] + list(here.parents):
        cand = p / "submodules" / "CENT" / "cent_simulation"
        if cand.exists():
            return cand.resolve()
    return None

def import_aim_transformer_block(cent_sim_root: Optional[str] = None):
    """Import TransformerBlockLlama (preferred) from CENT/AiM simulator.
    """
    root = None
    if cent_sim_root:
        root = Path(cent_sim_root).expanduser().resolve()
    elif os.environ.get("CENT_SIM_ROOT"):
        root = Path(os.environ["CENT_SIM_ROOT"]).expanduser().resolve()
    else:
        root = find_cent_sim_root()

    if root is None or not root.exists():
        raise RuntimeError(
            "Cannot locate cent_simulation. Please pass --cent-sim-root "
        )
    _add_sys_path(root)

    import importlib.util
    prev_utils = sys.modules.get("utils", None)
    cent_utils_path = (Path(root) / "utils.py").resolve()
    cent_utils_mod = None

    if cent_utils_path.exists():
        spec = importlib.util.spec_from_file_location("_cent_utils_for_aim", str(cent_utils_path))
        if spec and spec.loader:
            cent_utils_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cent_utils_mod)  # type: ignore[attr-defined]
            sys.modules["utils"] = cent_utils_mod

    try:
        # Prefer Llama wrapper (has memory_mapping()) but accept different casings.
        for mod_name in ("Llama", "llama"):
            try:
                mod = __import__(mod_name, fromlist=["TransformerBlockLlama"])
                if hasattr(mod, "TransformerBlockLlama"):
                    return getattr(mod, "TransformerBlockLlama")
            except Exception:
                pass

        # Fallback to raw TransformerBlock (also accept different casings)
        for mod_name in ("TransformerBlock", "transformerblock"):
            try:
                mod = __import__(mod_name, fromlist=["TransformerBlock"])
                if hasattr(mod, "TransformerBlock"):
                    return getattr(mod, "TransformerBlock")
            except Exception:
                pass

        raise RuntimeError(f"Neither TransformerBlockLlama nor TransformerBlock could be imported from {root}")
    finally:
        if prev_utils is not None:
            sys.modules["utils"] = prev_utils
        else:
            if "utils" in sys.modules and sys.modules["utils"] is cent_utils_mod:
                try:
                    del sys.modules["utils"]
                except Exception:
                    pass


def parse_layer(node_id: str) -> int:
    m = re.match(r"L(\d+)_", str(node_id))
    return int(m.group(1)) if m else -1

def parse_shard(node_id: str) -> int:
    m = re.search(r"_(?:s|shard)_?(\d+)", str(node_id), flags=re.IGNORECASE)
    return int(m.group(1)) if m else -1

def strip_shard_suffix(node_id: str) -> str:
    """Strip trailing shard suffix from node_id.

    Example: 'L0_Q_s3' -> 'L0_Q'.
    """
    s = str(node_id)
    return re.sub(r"_(?:s|shard)_?\d+$", "", s, flags=re.IGNORECASE)


def parse_prefill_decode_len_from_path(path: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        base = os.path.basename(str(path))
    except Exception:
        base = str(path)

    prefill_len: Optional[int] = None
    decode_len: Optional[int] = None

    m = re.search(r"prefill[_-]?(\d+)", base, flags=re.IGNORECASE)
    if m:
        try:
            prefill_len = int(m.group(1))
        except Exception:
            prefill_len = None

    # Most common: decode_1024 / decode-1024
    m = re.search(r"decode[_-]?(\d+)", base, flags=re.IGNORECASE)
    if m:
        try:
            decode_len = int(m.group(1))
        except Exception:
            decode_len = None

    # Fallback: prefill-1024xdecode_1024 style (prefill sometimes not matched above)
    if prefill_len is None:
        m = re.search(r"prefill[_-]?(\d+)\s*[xX]\s*decode", base, flags=re.IGNORECASE)
        if m:
            try:
                prefill_len = int(m.group(1))
            except Exception:
                prefill_len = None

    return prefill_len, decode_len


def decode_token_index_from_sample_step(step_sample: int, stride: int) -> int:
    s = int(step_sample)
    st = int(stride) if stride is not None else 1
    if st <= 1:
        return s
    if s <= 0:
        return 0
    if s == 1:
        return 1
    return int((s - 1) * st - 1)


def decode_token_block_start_from_sample_step(step_sample: int, stride: int) -> int:
    s = int(step_sample)
    st = int(stride) if stride is not None else 1
    if st <= 1:
        return s
    if s <= 0:
        return 0
    if s == 1:
        return 1
    return int((s - 1) * st)

def _is_fixed_mode_value(v: Any) -> bool:
    s = str(v).strip().lower()
    if not s or s == "nan":
        return False
    return s.startswith("fixed")

def _decode_trace_has_expanded_fixed_plan(schedule_df: pd.DataFrame) -> bool:
    """Whether the decode trace already contains per-token FIXED_PLAN/FIXED_COMM rows.

    Legacy sampled traces only keep sampled decode steps; expanded traces materialize
    every decode token and mark reused-plan tokens as FIXED_*.
    """
    if schedule_df is None or schedule_df.empty or "phase" not in schedule_df.columns:
        return False

    dec = schedule_df[schedule_df["phase"].astype(str) == "decode"]
    if dec.empty:
        return False

    try:
        if "sig_step" in dec.columns and "step" in dec.columns:
            n_sig = int(pd.to_numeric(dec["sig_step"], errors="coerce").fillna(-1).astype(int).nunique())
            n_step = int(pd.to_numeric(dec["step"], errors="coerce").fillna(-1).astype(int).nunique())
            if n_sig > 0 and n_step > n_sig:
                return True
    except Exception:
        pass

    if "mode" not in dec.columns:
        return False

    base = dec
    if "op" in dec.columns:
        op_l = dec["op"].astype(str).str.strip().str.lower()
        comp = dec[~op_l.isin(_COMM_OPS)]
        if not comp.empty:
            base = comp

    try:
        modes = base["mode"].astype(str).str.strip().str.lower()
    except Exception:
        return False
    return bool(modes.map(_is_fixed_mode_value).any())


def compute_decode_step_scale_map(
    schedule_df: pd.DataFrame,
    schedule_path: Optional[str],
    decode_stride: int,
) -> Tuple[Dict[int, int], str, Optional[int]]:
    st = int(decode_stride) if decode_stride is not None else 1
    if st < 1:
        st = 1

    if schedule_df is None or schedule_df.empty or "phase" not in schedule_df.columns:
        return ({}, "no_decode", None)

    df = schedule_df
    if "step" not in df.columns:
        try:
            df = add_step_column(df)
        except Exception:
            return ({}, "unknown", None)

    dec = df[df["phase"].astype(str) == "decode"]
    if dec.empty:
        return ({}, "no_decode", None)

    dec_steps = (
        pd.to_numeric(dec["step"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .unique()
        .tolist()
    )
    dec_steps = sorted([int(s) for s in dec_steps if int(s) >= 0])

    # Expanded traces already materialize every decode token step, so each actual step
    # should be counted exactly once and must NOT be rescaled by decode_stride.
    if _decode_trace_has_expanded_fixed_plan(df):
        decode_len_i: Optional[int] = None
        try:
            if "actual_token_index" in dec.columns:
                decode_len_i = int(
                    pd.to_numeric(dec["actual_token_index"], errors="coerce").fillna(-1).astype(int).max() + 1
                )
            else:
                decode_len_i = int(max(dec_steps) + 1) if dec_steps else 0
        except Exception:
            decode_len_i = int(max(dec_steps) + 1) if dec_steps else 0
        return ({int(s): 1 for s in dec_steps}, "expanded_fixed_plan", decode_len_i)

    # stride==1 -> no expansion
    if st == 1:
        return ({int(s): 1 for s in dec_steps}, "stride1", None)
    decode_len: Optional[int] = None
    try:
        _, decode_len = parse_prefill_decode_len_from_path(schedule_path or "")
    except Exception:
        decode_len = None
    if decode_len is None:
        raise ValueError(
            "decode_len must be parsed from schedule filename (e.g., '*decode_1024*'). "
            f"Got schedule_path={schedule_path!r}."
        )
    decode_len_i = int(decode_len)
    if decode_len_i <= 0:
        raise ValueError(f"decode_len must be > 0 (got {decode_len_i}) from {schedule_path!r}")

    # Legacy sampled-only scaling policy:
    #   step 0 -> token 0, scale 1
    #   step 1 -> token 1, scale (stride-1)
    #   step >=2 -> sample at token (k*stride-1), scale stride (last block clipped)
    mode = "token0_token1"
    scale_by_step: Dict[int, int] = {}

    # Optional sanity check: warn if sample count doesn't match the expected pattern.
    try:
        if decode_len_i > 1 and st > 1:
            n_blocks = int((decode_len_i + st - 1) // st)
            expected = int(n_blocks + 1)
            if len(dec_steps) != expected or (dec_steps and dec_steps[0] != 0) or (len(dec_steps) > 1 and dec_steps[1] != 1):
                print(
                    f"[warn] decode sampling steps mismatch for {os.path.basename(str(schedule_path or ''))}: "
                    f"decode_len={decode_len_i} stride={st} expected_steps={expected} got_steps={len(dec_steps)} "
                    f"(first_steps={dec_steps[:4]})",
                    file=sys.stderr,
                )
    except Exception:
        pass

    for s in dec_steps:
        si = int(s)
        if si <= 0:
            scale_by_step[si] = 1
            continue
        if si == 1:
            # token 1 represents the remaining tokens in the first block.
            scale_by_step[si] = max(0, min(int(st - 1), int(decode_len_i - 1)))
            continue
        # step>=2: block index b = step-1, represents tokens [b*stride .. (b+1)*stride-1].
        b = int(si - 1)
        block_start = int(b * st)
        remaining = int(decode_len_i - block_start)
        scale_by_step[si] = max(0, min(int(st), remaining))

    return scale_by_step, mode, decode_len_i


def infer_decode_steps(df: pd.DataFrame) -> int:
    d = df[df["phase"] == "decode"]
    if len(d) == 0:
        return 0
    if "sig_step" in d.columns:
        try:
            return int(pd.to_numeric(d["sig_step"], errors="coerce").fillna(-1).astype(int).max() + 1)
        except Exception:
            pass
    nodes_per_step = d["node_id"].nunique()
    if nodes_per_step == 0:
        return 0
    if len(d) % nodes_per_step != 0:
        return 1
    return len(d) // nodes_per_step


def add_step_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add actual trace step column: prefill=-1, decode=0..(steps-1)."""
    df = df.copy()
    df["step"] = -1
    dmask = df["phase"] == "decode"
    if dmask.any():
        d = df[dmask]
        nodes_per_step = d["node_id"].nunique()
        if nodes_per_step > 0:
            # assume rows grouped by token, one token = nodes_per_step rows
            df.loc[dmask, "step"] = (d.groupby("phase").cumcount() // nodes_per_step).astype(int).values
    return df


def annotate_decode_sampling(df: pd.DataFrame, decode_stride: int) -> pd.DataFrame:
    """Annotate decode rows with sampled-step metadata.

    Added columns:
      - sig_step: benchmark/signature step id used in SegmentSig lookup.
      - actual_token_index: actual token index represented by this row group.
      - sig_token_index: actual token index of the sampled step reused by this row group.
      - is_sampled_step: True for real sampled steps, False for FIXED_PLAN/FIXED_COMM steps.

    Legacy sampled-only traces: sig_step == step and token indices follow the old
    token0/token1/stride sampling rule.

    Expanded FIXED_PLAN traces: step is the *actual* decode token index in the trace,
    while sig_step is a dense sampled-step ordinal that only advances on non-FIXED
    decode steps. FIXED_PLAN rows reuse the most recent sampled sig_step.
    """
    out = df.copy()
    if "step" not in out.columns:
        out = add_step_column(out)

    out["sig_step"] = out["step"]
    out["actual_token_index"] = out["step"]
    out["sig_token_index"] = out["step"]
    out["is_sampled_step"] = out["phase"].astype(str) != "decode"

    dec_mask = out["phase"].astype(str) == "decode"
    if not dec_mask.any():
        return out

    st = int(decode_stride) if decode_stride is not None else 1
    if st < 1:
        st = 1

    dec = out.loc[dec_mask].copy()
    dec_steps = sorted(
        pd.to_numeric(dec["step"], errors="coerce").fillna(-1).astype(int).unique().tolist()
    )
    dec_steps = [int(s) for s in dec_steps if int(s) >= 0]

    expanded = False
    step_is_sampled: Dict[int, bool] = {int(s): True for s in dec_steps}
    if "mode" in dec.columns:
        for step_i, g in dec.groupby("step", sort=False):
            base = g
            if "op" in g.columns:
                op_l = g["op"].astype(str).str.strip().str.lower()
                comp = g[~op_l.isin(_COMM_OPS)]
                if not comp.empty:
                    base = comp
            try:
                modes = base["mode"].astype(str).str.strip().str.lower()
                fixed_flags = modes.map(_is_fixed_mode_value)
                is_sampled = bool((~fixed_flags).any()) or len(modes) == 0
            except Exception:
                is_sampled = True
            step_is_sampled[int(step_i)] = bool(is_sampled)
        expanded = any((not v) for v in step_is_sampled.values()) and any(bool(v) for v in step_is_sampled.values())

    actual_tok_by_step: Dict[int, int] = {}
    sig_step_by_step: Dict[int, int] = {}
    sig_tok_by_step: Dict[int, int] = {}
    sampled_flag_by_step: Dict[int, bool] = {}

    if not expanded:
        for s in dec_steps:
            tok = int(decode_token_index_from_sample_step(int(s), int(st)))
            actual_tok_by_step[int(s)] = tok
            sig_step_by_step[int(s)] = int(s)
            sig_tok_by_step[int(s)] = tok
            sampled_flag_by_step[int(s)] = True
    else:
        cur_sig_step: Optional[int] = None
        cur_sig_tok: Optional[int] = None
        next_sig_step = 0
        for s in dec_steps:
            s_i = int(s)
            sampled = bool(step_is_sampled.get(s_i, True))
            if sampled or cur_sig_step is None:
                cur_sig_step = int(next_sig_step)
                cur_sig_tok = int(s_i)
                next_sig_step += 1
            actual_tok_by_step[s_i] = int(s_i)
            sig_step_by_step[s_i] = int(cur_sig_step)
            sig_tok_by_step[s_i] = int(cur_sig_tok if cur_sig_tok is not None else s_i)
            sampled_flag_by_step[s_i] = sampled

    step_series = pd.to_numeric(out.loc[dec_mask, "step"], errors="coerce").fillna(-1).astype(int)
    out.loc[dec_mask, "actual_token_index"] = step_series.map(actual_tok_by_step).astype(int).values
    out.loc[dec_mask, "sig_step"] = step_series.map(sig_step_by_step).astype(int).values
    out.loc[dec_mask, "sig_token_index"] = step_series.map(sig_tok_by_step).astype(int).values
    out.loc[dec_mask, "is_sampled_step"] = step_series.map(sampled_flag_by_step).astype(bool).values
    return out


def infer_decode_context_lens_from_schedule(
    schedule_df: pd.DataFrame,
    cfg: Any,
    *,
    decode_stride: int,
) -> List[int]:
    if schedule_df is None or schedule_df.empty or "phase" not in schedule_df.columns:
        return []

    dec = schedule_df[schedule_df["phase"].astype(str) == "decode"]
    if dec.empty:
        return []

    toks: List[int] = []
    if "sig_step" in dec.columns and "sig_token_index" in dec.columns:
        try:
            rep = (
                dec.groupby("sig_step", sort=True)["sig_token_index"]
                .first()
                .reset_index(drop=True)
                .tolist()
            )
            toks = [int(x) for x in rep]
        except Exception:
            toks = []

    if not toks:
        steps = sorted(pd.to_numeric(dec["step"], errors="coerce").fillna(-1).astype(int).unique().tolist())
        steps = [int(s) for s in steps if int(s) >= 0]
        st = int(decode_stride) if decode_stride is not None else 1
        if st < 1:
            st = 1
        toks = [int(decode_token_index_from_sample_step(int(s), int(st))) for s in steps]

    prefill_len = int(getattr(cfg, "prefill_len", 0) or 0)
    return [int(prefill_len + 1 + int(tok)) for tok in toks]


def build_actual_step_token_index_map(
    schedule_df: pd.DataFrame,
    *,
    decode_stride: int,
) -> Dict[int, int]:
    if schedule_df is None or schedule_df.empty or "phase" not in schedule_df.columns or "step" not in schedule_df.columns:
        return {}

    dec = schedule_df[schedule_df["phase"].astype(str) == "decode"]
    if dec.empty:
        return {}

    if "actual_token_index" in dec.columns:
        try:
            g = dec.groupby("step", sort=True)["actual_token_index"].first()
            return {int(k): int(v) for k, v in g.items()}
        except Exception:
            pass

    st = int(decode_stride) if decode_stride is not None else 1
    if st < 1:
        st = 1
    steps = sorted(pd.to_numeric(dec["step"], errors="coerce").fillna(-1).astype(int).unique().tolist())
    return {int(s): int(decode_token_index_from_sample_step(int(s), int(st))) for s in steps if int(s) >= 0}


def normalize_device_type(v: Any) -> str:
    s = str(v).strip().lower()
    # Common aliases
    if s in ("npu", "cuda"):
        return "npu"
    if s in ("aim",):
        return "pim"
    if s in ("host", "x86", "arm"):
        return "cpu"
    return s

def load_schedule_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"phase","node_id","op","device","device_type","start","end","duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"schedule csv missing columns: {sorted(missing)}")
    df = df.copy()
    df["_row"] = range(len(df))
    df["device_type"] = df["device_type"].apply(normalize_device_type)
    # best-effort numeric conversion (avoid string dtype surprises later)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = add_step_column(df)
    df["layer"] = df["node_id"].apply(parse_layer)
    df["shard"] = df["node_id"].apply(parse_shard)
    return df


# ==========================================================
# Comms trace (optional) helpers
# ==========================================================
def load_comms_csv(path: str) -> pd.DataFrame:
    """Load a comms trace CSV."""
    df = pd.read_csv(path)
    # best-effort normalization: keep original columns, only ensure duration is numeric
    if "duration" in df.columns:
        df = df.copy()
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    return df



def extract_weight_load_seconds_by_phase(
    comms_paths: List[str],
) -> Dict[str, Dict[str, float]]:
    """Aggregate direct `weight_load` durations from comms trace(s), split by phase.

    Intentional simplification for deploy/verify:
      - use the durations explicitly marked in comms.csv
      - sum rows directly
      - do not match individual loads back to ops
      - do not do subtraction /补足 / max-by-destination bookkeeping
    """

    zero = {"npu_s": 0.0, "pim_s": 0.0, "unknown_s": 0.0, "total_s": 0.0}
    if not comms_paths:
        return {"all": dict(zero)}

    rows: List[pd.DataFrame] = []

    def _norm_str(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").replace("nan", "", regex=False).str.strip()

    def _norm_phase(s: pd.Series) -> pd.Series:
        ph = s.astype(str).fillna("prefill").replace("nan", "prefill", regex=False).str.strip().str.lower()
        ph = ph.where(ph != "", "prefill")
        ph = ph.str.replace(r"^pre.*", "prefill", regex=True)
        ph = ph.str.replace(r"^dec.*", "decode", regex=True)
        return ph

    for cp in comms_paths:
        p = resolve_existing_path(cp)
        df = load_comms_csv(str(p))
        if df is None or df.empty or "tag" not in df.columns or "duration" not in df.columns:
            continue

        tag_l = df["tag"].astype(str).str.strip().str.lower()
        wl = df[tag_l == "weight_load"].copy()
        if wl.empty:
            continue

        wl["duration"] = pd.to_numeric(wl["duration"], errors="coerce").fillna(0.0).astype(float)
        phase = _norm_phase(wl["phase"]) if "phase" in wl.columns else pd.Series(["prefill"] * len(wl), index=wl.index)

        dst = _norm_str(wl["dst"]) if "dst" in wl.columns else pd.Series([""] * len(wl), index=wl.index)
        dst_type = _norm_str(wl["dst_type"]) if "dst_type" in wl.columns else pd.Series([""] * len(wl), index=wl.index)

        dst_norm = dst.astype(str).str.upper().str.replace("_", "", regex=False).str.strip()
        dst_type_l = dst_type.astype(str).str.lower().str.strip()

        pim_mask = dst_norm.str.startswith("PIM") | dst_type_l.str.contains("pim")
        npu_mask = dst_norm.str.contains("NPU") | dst_type_l.str.contains("npu")
        npu_mask = npu_mask & (~pim_mask)

        cls = np.where(pim_mask.to_numpy(), "pim", np.where(npu_mask.to_numpy(), "npu", "unknown"))
        rows.append(
            pd.DataFrame(
                {
                    "phase": phase.to_numpy(),
                    "class": cls,
                    "duration": wl["duration"].to_numpy(dtype=np.float64),
                }
            )
        )

    if not rows:
        return {"all": dict(zero)}

    all_wl = pd.concat(rows, ignore_index=True)
    out: Dict[str, Dict[str, float]] = {}
    for ph, sub in all_wl.groupby("phase", sort=False):
        cls_sum = sub.groupby("class", sort=False)["duration"].sum()
        npu_s = float(cls_sum.get("npu", 0.0) or 0.0)
        pim_s = float(cls_sum.get("pim", 0.0) or 0.0)
        unknown_s = float(cls_sum.get("unknown", 0.0) or 0.0)
        out[str(ph)] = {
            "npu_s": float(npu_s),
            "pim_s": float(pim_s),
            "unknown_s": float(unknown_s),
            "total_s": float(npu_s + pim_s + unknown_s),
        }

    all_npu = float(sum(float(v.get("npu_s", 0.0) or 0.0) for v in out.values()))
    all_pim = float(sum(float(v.get("pim_s", 0.0) or 0.0) for v in out.values()))
    all_unknown = float(sum(float(v.get("unknown_s", 0.0) or 0.0) for v in out.values()))
    out["all"] = {
        "npu_s": float(all_npu),
        "pim_s": float(all_pim),
        "unknown_s": float(all_unknown),
        "total_s": float(all_npu + all_pim + all_unknown),
    }
    return out


def extract_weight_load_seconds_by_step(
    comms_paths: List[str],
    schedule_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Aggregate `weight_load` durations per (phase, step)."""

    cols = ["phase", "step", "npu_s", "pim_s", "unknown_s", "total_s"]
    if not comms_paths:
        return pd.DataFrame(columns=cols)

    def _norm_str(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("").replace("nan", "", regex=False).str.strip()

    def _norm_phase(s: pd.Series) -> pd.Series:
        ph = s.astype(str).fillna("prefill").replace("nan", "prefill", regex=False).str.strip().str.lower()
        ph = ph.where(ph != "", "prefill")
        ph = ph.str.replace(r"^pre.*", "prefill", regex=True)
        ph = ph.str.replace(r"^dec.*", "decode", regex=True)
        return ph

    # Precompute decode step start times from schedule for time->step mapping (when comm trace has no 'step' col)
    step_ids = None
    step_starts = None
    try:
        if schedule_df is not None and (not schedule_df.empty) and ("phase" in schedule_df.columns) and ("step" in schedule_df.columns):
            ddec = schedule_df[schedule_df["phase"].astype(str) == "decode"]
            if (ddec is not None) and (not ddec.empty) and ("start" in ddec.columns):
                bounds = (
                    ddec.groupby("step", as_index=True)
                    .agg(start_min=("start", "min"))
                    .sort_values("start_min")
                )
                step_ids = bounds.index.to_numpy(dtype=int)
                step_starts = bounds["start_min"].to_numpy(dtype=float)
    except Exception:
        step_ids = None
        step_starts = None

    rows: List[pd.DataFrame] = []
    for cp in comms_paths:
        p = resolve_existing_path(cp)
        df = load_comms_csv(str(p))
        if df is None or df.empty:
            continue
        if "tag" not in df.columns or "duration" not in df.columns:
            continue

        tag_l = df["tag"].astype(str).str.strip().str.lower()
        wl = df[tag_l == "weight_load"].copy()
        if wl.empty:
            continue

        wl["duration"] = pd.to_numeric(wl["duration"], errors="coerce").fillna(0.0).astype(float)

        # phase
        if "phase" in wl.columns:
            phase = _norm_phase(wl["phase"])
        else:
            phase = pd.Series(["prefill"] * len(wl), index=wl.index)

        # step (prefer explicit, else map by time)
        if "step" in wl.columns:
            step = pd.to_numeric(wl["step"], errors="coerce").fillna(-1).astype(int)
        else:
            step = pd.Series([-1] * len(wl), index=wl.index, dtype=int)
            if (step_ids is not None) and (step_starts is not None) and (len(step_ids) > 0):
                # use comm start time; fallback to end
                st = pd.to_numeric(wl.get("start"), errors="coerce") if "start" in wl.columns else pd.Series([np.nan] * len(wl), index=wl.index)
                en = pd.to_numeric(wl.get("end"), errors="coerce") if "end" in wl.columns else pd.Series([np.nan] * len(wl), index=wl.index)
                t = st.fillna(en)
                mask_dec = phase.astype(str) == "decode"
                if mask_dec.any():
                    tt = t[mask_dec].to_numpy(dtype=float)
                    idx = np.searchsorted(step_starts, tt, side="right") - 1
                    idx[idx < 0] = 0
                    idx[idx >= len(step_ids)] = len(step_ids) - 1
                    step.loc[mask_dec] = step_ids[idx]

        # prefill (and any non-decode) is treated as step=-1 to align with schedule
        step = step.where(phase.astype(str) == "decode", -1).astype(int)

        # destination-based classification (npu/pim/unknown), same as phase-level function
        dst = _norm_str(wl["dst"]) if "dst" in wl.columns else pd.Series([""] * len(wl), index=wl.index)
        dst_type = _norm_str(wl["dst_type"]) if "dst_type" in wl.columns else pd.Series([""] * len(wl), index=wl.index)

        dst_u = dst.astype(str).str.upper().str.strip()
        dst_norm = dst_u.str.replace("_", "", regex=False)
        dst_norm = dst_norm.str.replace(r"^NPU", "npu", regex=True)
        dst_type_l = dst_type.astype(str).str.strip().str.lower()

        pim_mask = dst_norm.str.startswith("PIM") | dst_type_l.str.contains("pim")
        npu_mask = (
            dst_norm.str.startswith("npu")
            | dst_norm.str.startswith("NPU")
            | dst_type_l.str.contains("npu")
            | dst_type_l.str.contains("npu")
            | dst_type_l.isin(["other"])
        )
        npu_mask = npu_mask & (~pim_mask)
        cls = np.where(pim_mask.to_numpy(), "pim", np.where(npu_mask.to_numpy(), "npu", "unknown"))

        rows.append(
            pd.DataFrame(
                {
                    "phase": phase.to_numpy(),
                    "step": step.to_numpy(dtype=np.int64),
                    "class": cls,
                    "dst": dst_norm.to_numpy(),
                    "duration": wl["duration"].to_numpy(dtype=np.float64),
                }
            )
        )

    if not rows:
        return pd.DataFrame(columns=cols)

    all_wl = pd.concat(rows, ignore_index=True)

    per_dst = (
        all_wl.groupby(["phase", "step", "class", "dst"], dropna=False)["duration"]
        .sum()
        .reset_index()
    )

    recs: List[Dict[str, Any]] = []
    for (ph, st), sub in per_dst.groupby(["phase", "step"], sort=False):
        pim_by = sub[sub["class"] == "pim"].set_index("dst")["duration"].to_dict()
        npu_by = sub[sub["class"] == "npu"].set_index("dst")["duration"].to_dict()
        unk_by = sub[sub["class"] == "unknown"].set_index("dst")["duration"].to_dict()

        pim_part = max(pim_by.values()) if pim_by else 0.0
        npu_part = max(npu_by.values()) if npu_by else 0.0
        unk_part = sum(unk_by.values()) if unk_by else 0.0

        npu_s = float(npu_part + unk_part)
        pim_s = float(pim_part)
        unknown_s = float(unk_part)
        total_s = float(npu_s + pim_s)

        recs.append(
            {
                "phase": str(ph),
                "step": int(st),
                "npu_s": float(npu_s),
                "pim_s": float(pim_s),
                "unknown_s": float(unknown_s),
                "total_s": float(total_s),
            }
        )

    out_df = pd.DataFrame(recs)
    if out_df.empty:
        return pd.DataFrame(columns=cols)
    out_df["phase"] = out_df["phase"].astype(str)
    out_df["step"] = pd.to_numeric(out_df["step"], errors="coerce").fillna(-1).astype(int)
    return out_df[cols]


def extract_weight_load_seconds(
    comms_paths: List[str],
    *,
    decode_stride: int = 1,
) -> Dict[str, float]:
    """Aggregate `weight_load` durations from comms trace(s).

    Backward-compatible wrapper over :func:`extract_weight_load_seconds_by_phase`.
    """
    by_phase = extract_weight_load_seconds_by_phase(comms_paths)
    rec = by_phase.get("all", {}) if isinstance(by_phase, dict) else {}
    return {
        "npu_s": float(rec.get("npu_s", 0.0) or 0.0),
        "pim_s": float(rec.get("pim_s", 0.0) or 0.0),
        "total_s": float(rec.get("total_s", 0.0) or 0.0),
        "unknown_s": float(rec.get("unknown_s", 0.0) or 0.0),
    }

# ==========================================================
# Layer-scope comm accumulation (kv_load)
# ==========================================================

def extract_layer_comm_seconds(
    comms_paths: List[str],
    schedule_df: pd.DataFrame,
    *,
    tags: Tuple[str, ...] = ("kv_load",),
) -> pd.DataFrame:
    """Extract per-(phase, step, layer) comm seconds for selected comm tags.
    Returned columns:
      phase, step, layer, comm_s, weight_load_s, kv_load_s
    """

    if not comms_paths:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    # Load + concat
    parts = []
    for p in comms_paths:
        if not p:
            continue
        p = resolve_existing_path(p)
        if p is None:
            continue
        try:
            c = load_comms_csv(str(p))
        except Exception:
            continue
        if c is None or c.empty:
            continue
        parts.append(c)

    if not parts:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    df = pd.concat(parts, ignore_index=True)

    if "tag" not in df.columns or "duration" not in df.columns:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    df = df.copy()
    df["tag_l"] = (
        df["tag"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    tags_l = [str(t).strip().lower().replace(" ", "_") for t in tags]
    df = df[df["tag_l"].isin(tags_l)].copy()
    if df.empty:
        return pd.DataFrame(columns=["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"])

    # Phase
    if "phase" in df.columns:
        df["phase"] = df["phase"].astype(str).str.strip()
    else:
        df["phase"] = "prefill"

    # Duration
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).astype(float)

    # Layer
    if "layer" in df.columns:
        df["layer"] = pd.to_numeric(df["layer"], errors="coerce").fillna(-1).astype(int)
    else:
        src_col = None
        for c in ("node_id", "prod_node", "cons_node"):
            if c in df.columns:
                src_col = c
                break
        if src_col is None:
            df["layer"] = -1
        else:
            df["layer"] = df[src_col].apply(parse_layer).astype(int)

    # Step
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(-1).astype(int)
    else:
        # Build decode step start times from schedule
        d = schedule_df
        if d is None or d.empty or "phase" not in d.columns or "step" not in d.columns:
            df["step"] = -1
        else:
            dph = d["phase"].astype(str)
            ddec = d[dph == "decode"]
            if ddec.empty:
                df["step"] = -1
            else:
                bounds = (
                    ddec.groupby("step", as_index=True)
                    .agg(start_min=("start", "min"), end_max=("end", "max"))
                    .sort_values("start_min")
                )
                step_ids = bounds.index.to_numpy(dtype=int)
                step_starts = bounds["start_min"].to_numpy(dtype=float)

                # Use comm start time; fall back to end
                st = pd.to_numeric(df.get("start"), errors="coerce")
                en = pd.to_numeric(df.get("end"), errors="coerce")
                t = st.fillna(en)

                # Default -1 for non-decode
                df["step"] = -1
                mask_dec = df["phase"].astype(str) == "decode"
                if mask_dec.any() and step_starts.size > 0:
                    tt = t[mask_dec].to_numpy(dtype=float)
                    import numpy as np
                    idx = np.searchsorted(step_starts, tt, side="right") - 1
                    idx[idx < 0] = 0
                    idx[idx >= len(step_ids)] = len(step_ids) - 1
                    df.loc[mask_dec, "step"] = step_ids[idx]

    # Aggregate
    g = (
        df.groupby(["phase", "step", "layer", "tag_l"], as_index=False)["duration"]
        .sum()
        .rename(columns={"duration": "comm_s"})
    )

    wide = (
        g.pivot_table(index=["phase", "step", "layer"], columns="tag_l", values="comm_s", aggfunc="sum", fill_value=0.0)
        .reset_index()
    )

    # Ensure columns
    if "weight_load" not in wide.columns:
        wide["weight_load"] = 0.0
    if "kv_load" not in wide.columns:
        wide["kv_load"] = 0.0

    wide = wide.rename(columns={"weight_load": "weight_load_s", "kv_load": "kv_load_s"})
    wide["comm_s"] = wide["weight_load_s"].astype(float) + wide["kv_load_s"].astype(float)

    # Order
    cols = ["phase", "step", "layer", "comm_s", "weight_load_s", "kv_load_s"]
    for c in cols:
        if c not in wide.columns:
            wide[c] = 0
    wide = wide[cols]

    return wide


def estimate_kv_load_seconds_from_ops_csv(
    schedule_df: pd.DataFrame,
    cfg: WorkloadConfig,
    *,
    decode_stride: int,
) -> Dict[Tuple[str, int, int], float]:
    """Estimate kv_load time per (phase, step, layer) by scanning QK/SV ops.

    Model (decode only):
        bytes(K or V) = batch * kv_heads_shard * head_dim * kv_len * kv_dtype_bytes
        time_s        = bytes / (kv_load_bw_gbs * 1e9) + kv_load_overhead_us * 1e-6
    """

    if schedule_df is None or schedule_df.empty:
        return {}

    bw = float(getattr(cfg, "kv_load_bw_gbs", 0.0) or 0.0)
    if bw <= 0.0:
        return {}

    kv_dtype_bytes = float(getattr(cfg, "kv_dtype_bytes", 2.0) or 2.0)
    overhead_s = float(getattr(cfg, "kv_load_overhead_us", 0.0) or 0.0) * 1e-6
    batch = int(getattr(cfg, "batch", 1) or 1)
    n_kv_heads = getattr(cfg, "n_kv_heads", None)
    if n_kv_heads is None:
        n_kv_heads = int(getattr(cfg, "n_heads", 0) or 0)
    else:
        n_kv_heads = int(n_kv_heads)
    dim = int(getattr(cfg, "dim", 0) or 0)
    q_heads = int(getattr(cfg, "n_heads", 0) or 0)
    head_dim = max(1, int(dim // max(1, q_heads)))
    shards = int(getattr(cfg, "shards", 1) or 1)
    shards = max(1, shards)

    def _kv_heads_for_shard(shard_id: int) -> int:
        """Distribute KV heads across shards (supports non-divisible cases)."""
        return int(_split_count_across_shards(int(n_kv_heads), int(shards), int(shard_id)))

    # decode context length lookup
    dcl = getattr(cfg, "decode_context_lens", None)
    dcl_list = list(dcl) if isinstance(dcl, (list, tuple)) else None

    prefill_len = int(getattr(cfg, "prefill_len", 0) or 0)
    ds = int(decode_stride or getattr(cfg, "decode_stride", 1) or 1)
    if ds <= 0:
        ds = 1

    actual_tok_by_step: Dict[int, int] = {}
    try:
        if "actual_token_index" in schedule_df.columns and "step" in schedule_df.columns:
            dd = schedule_df[schedule_df["phase"].astype(str) == "decode"]
            if not dd.empty:
                gg = (
                    dd.groupby("step", sort=True)["actual_token_index"]
                    .first()
                    .astype(int)
                )
                actual_tok_by_step = {int(k): int(v) for k, v in gg.items()}
    except Exception:
        actual_tok_by_step = {}

    def _kv_len_for_step(step: int) -> int:
        if int(step) in actual_tok_by_step:
            tok = int(actual_tok_by_step[int(step)])
            return int(prefill_len + 1 + tok)
        if dcl_list is not None and 0 <= int(step) < len(dcl_list):
            try:
                return int(dcl_list[int(step)])
            except Exception:
                pass
        tok = int(decode_token_index_from_sample_step(int(step), int(ds)))
        return int(prefill_len + 1 + tok)

    # Filter QK/SV ops in decode.
    ph = schedule_df.get("phase")
    op = schedule_df.get("op")
    if ph is None or op is None:
        return {}

    ph_s = ph.astype(str).str.strip()
    op_u = op.astype(str).str.strip().str.upper()
    mask = (ph_s == "decode") & (op_u.isin(["QK", "SV"]))
    if not bool(mask.any()):
        return {}

    sub = schedule_df.loc[mask, ["phase", "step", "layer", "device", "shard", "op"]].copy()
    # best-effort numeric
    sub["step"] = pd.to_numeric(sub["step"], errors="coerce").fillna(-1).astype(int)
    sub["layer"] = pd.to_numeric(sub["layer"], errors="coerce").fillna(-1).astype(int)
    sub["shard"] = pd.to_numeric(sub["shard"], errors="coerce").fillna(-1).astype(int)

    per_dev: Dict[Tuple[str, int, int, str], float] = {}
    for r in sub.itertuples(index=False):
        phase = str(r.phase)
        step = int(r.step)
        layer = int(r.layer)
        device = str(r.device)
        shard_id = int(r.shard)

        if step < 0:
            # safety: skip malformed decode rows
            continue

        kv_len = _kv_len_for_step(step)
        kvh = _kv_heads_for_shard(shard_id)
        if kv_len <= 0 or kvh <= 0:
            continue

        elems = float(batch) * float(kvh) * float(head_dim) * float(kv_len)
        bytes_ = float(elems) * float(kv_dtype_bytes)
        t = float(bytes_ / (float(bw) * 1e9) + float(overhead_s))

        k = (phase, step, layer, device)
        per_dev[k] = float(per_dev.get(k, 0.0) + t)

    # collapse device dimension: layer-scope kv_load is the max across devices
    layer_map: Dict[Tuple[str, int, int], float] = {}
    for (phase, step, layer, _dev), t in per_dev.items():
        kk = (phase, int(step), int(layer))
        layer_map[kk] = max(float(layer_map.get(kk, 0.0)), float(t))

    return layer_map



# ==========================================================
# Workload config + shape inference (same as v2)
# ==========================================================

@dataclass
class WorkloadConfig:
    # model
    dim: int = 4096
    ffn_dim: int = 11008
    n_heads: int = 32
    shards: int = 4
    # schedule lengths
    prefill_len: int = 128
    decode_context_lens: Optional[List[int]] = None  # length = decode steps
    decode_stride: int = 1
    device: str = "none"
    npu_dtype: str = "fp16"

    # KV-cache load modeling (verification-time)
    kv_load_bw_gbs: float = 0.0
    kv_dtype_bytes: float = 2.0
    kv_load_overhead_us: float = 0.0
    n_kv_heads: Optional[int] = None
    batch: int = 1

    # pim sim config (defaults roughly match aim_sim)
    pim_dram_column: int = 256
    pim_dram_row: int = 64
    pim_burst_length: int = 16
    pim_num_banks: int = 8
    pim_num_channels: int = 4
    pim_threads: int = 1
    pim_reuse_size: int = 32
    pim_num_devices: int = 1  # DIMM/device count (model-parallel)

    # AiM-Ramulator integration
    pim_ramulator_bin: Optional[str] = None
    pim_ramulator_config: Optional[str] = None  # required to enable ramulator mode
    pim_freq_ghz: float = 1.0
    pim_ramulator_timeout_s: int = 1500
    pim_keep_traces: bool = False
    pim_trace_dir: Optional[str] = None
    pim_hw_json_path: Optional[str] = None
    debug: bool = False

    # segmenting
    segment_scope: str = "layer"

    # shard placement granularity:
    shard_policy: str = "fine"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkloadConfig":
        cfg = WorkloadConfig()
        for k,v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass(frozen=True)
class OpSig:
    """Internal operator signature for shape inference and execution."""
    device_type: str
    phase: str
    step: int
    op: str
    shard: int

@dataclass(frozen=True)
class OpShape:
    dim: int
    shard_dim: int
    ffn_shard_dim: int
    query_len: int
    key_len: int
    heads_per_shard: int
    head_dim: int


def _split_count_across_shards(total: int, shards: int, shard_id: int) -> int:
    total = int(max(0, total))
    shards = int(max(1, shards))
    shard_id = int(shard_id)
    if shard_id < 0 or shards <= 1:
        return int(total)
    base = int(total) // int(shards)
    rem = int(total) % int(shards)
    return int(base + (1 if int(shard_id) < int(rem) else 0))


def infer_op_shape(sig: OpSig, cfg: WorkloadConfig) -> OpShape:
    D = int(cfg.dim)
    Sd = int(D // max(1, cfg.shards))
    Fd = int(cfg.ffn_dim)
    Fsd = int(Fd // max(1, cfg.shards))
    Hd = int(D // max(1, cfg.n_heads))
    H = int(_split_count_across_shards(int(cfg.n_heads), int(max(1, cfg.shards)), int(getattr(sig, "shard", -1))))

    if sig.phase == "prefill":
        T = int(cfg.prefill_len)
        K = int(cfg.prefill_len)
    else:
        if cfg.decode_context_lens is None:
            stride = int(getattr(cfg, "decode_stride", 1) or 1)
            if stride <= 0:
                stride = 1
            tok = int(decode_token_index_from_sample_step(int(sig.step), int(stride)))
            K = int(cfg.prefill_len + 1 + tok)
        else:
            if sig.step < 0 or sig.step >= len(cfg.decode_context_lens):
                if cfg.decode_context_lens:
                    K = int(cfg.decode_context_lens[-1])
                else:
                    stride = int(getattr(cfg, "decode_stride", 1) or 1)
                    if stride <= 0:
                        stride = 1
                    tok = int(decode_token_index_from_sample_step(int(sig.step), int(stride)))
                    K = int(cfg.prefill_len + 1 + tok)
            else:
                K = int(cfg.decode_context_lens[sig.step])
        T = 1

    return OpShape(
        dim=D,
        shard_dim=Sd,
        ffn_shard_dim=Fsd,
        query_len=T,
        key_len=K,
        heads_per_shard=H,
        head_dim=Hd,
    )


# ==========================================================
# Segment signature + extraction
# ==========================================================

@dataclass(frozen=True)
class SegmentSig:
    """
    Segment signature:
    - device_type: 'npu' or 'pim' (measured backends).
      Other device types (e.g. 'cpu') are treated as trace-only: we skip exporting
      them as tasks and keep their schedule durations directly in merge().
    - phase: 'prefill' or 'decode'
    - step: prefill=-1, decode step id
    - ops: ordered list of (op, shard) inside the segment
    """
    device_type: str
    phase: str
    step: int
    ops: Tuple[Tuple[str,int], ...]

    def ops_repr(self) -> str:
        return ",".join([f"{op}:{shard}" for op,shard in self.ops])

    def to_key(self) -> str:
        # Key is stable and short, but collision-resistant
        h = hashlib.md5(self.ops_repr().encode("utf-8")).hexdigest()
        return f"{self.device_type}|{self.phase}|{self.step}|{h}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "phase": self.phase,
            "step": int(self.step),
            "ops": [{"op": op, "shard": int(shard)} for op, shard in self.ops],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SegmentSig":
        ops = tuple((str(x["op"]), int(x.get("shard", -1))) for x in d["ops"])
        return SegmentSig(device_type=str(d["device_type"]), phase=str(d["phase"]), step=int(d["step"]), ops=ops)

def extract_segments(df: pd.DataFrame, segment_scope: str) -> Tuple[Dict[str, SegmentSig], Counter]:
    """Extract unique segments from a schedule.

    We always segment by **layer**: one segment per (phase, actual_step, layer, device).
    For expanded FIXED_PLAN traces, SegmentSig.step uses the sampled-step id from
    ``sig_step`` instead of the raw actual token step.

    Notes:
      - Ops in `_COMM_OPS` are treated as communication/synchronization and are **not** exported
        as compute segments (NPU/PIM benchmarking skips them).
      - Device-scope segmentation (splitting a device timeline by comm ops) has been removed.

    Returns:
      - uniq: dict key -> SegmentSig
      - ctr:  Counter over keys (each occurrence corresponds to one (phase, step, layer, device) group)
    """

    scope = str(segment_scope).lower().strip()
    if scope != "layer":
        raise ValueError(
            f"Only segment_scope='layer' is supported (got {segment_scope!r}). "
            "Device-scope segmentation has been removed."
        )

    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()

    group_cols = ["phase", "step", "layer", "device"]
    for (phase, step, layer, device), g in df.groupby(group_cols, sort=False):
        g = g.sort_values(["start", "_row"], kind="mergesort")
        dev_type = str(g["device_type"].iloc[0])
        try:
            sig_step = int(pd.to_numeric(g.get("sig_step"), errors="coerce").fillna(step).iloc[0]) if "sig_step" in g.columns else int(step)
        except Exception:
            sig_step = int(step)
        ops: List[Tuple[str, int]] = []
        for _, r in g.iterrows():
            op = str(r["op"]).strip()
            if op.lower() in _COMM_OPS:
                continue
            shard = int(r["shard"]) if pd.notna(r["shard"]) else -1
            ops.append((op, shard))

        if not ops:
            continue

        seg = SegmentSig(device_type=dev_type, phase=str(phase), step=int(sig_step), ops=tuple(ops))
        k = seg.to_key()
        uniq[k] = seg
        ctr[k] += 1

    return uniq, ctr


# coarse-majority verification path removed; only fine-grained shard placement is supported now.

class _DebugLogger:
    """Lightweight logger for verbose debug runs.

    When enabled, logs are printed to stdout and optionally duplicated to a
    text file.
    """

    def __init__(self, enabled: bool, out_path: Optional[str] = None):
        self.enabled = bool(enabled)
        self.out_path: Optional[Path] = None
        self._fp = None
        if self.enabled and out_path:
            p = Path(out_path).expanduser().resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            self.out_path = p
            self._fp = p.open('w', encoding='utf-8')

    def log(self, msg: str) -> None:
        if not self.enabled:
            return
        s = str(msg)
        print(s)
        if self._fp is not None:
            try:
                self._fp.write(s + "\n")
                self._fp.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None


def _resolve_ramulator_bin_path(bin_str: str) -> Path:
    """Resolve ramulator2 executable path from a CLI/config string."""
    if not bin_str:
        raise ValueError("pim_ramulator_bin is empty")

    # Best-effort: treat as path first, then fall back to PATH lookup.
    p: Optional[Path] = None
    try:
        # resolve_existing_path handles cwd and script-root relative lookups.
        p = resolve_existing_path(bin_str)
    except Exception:
        which = shutil.which(bin_str)
        if which:
            p = Path(which)

    if p is None:
        raise FileNotFoundError(f"ramulator2 executable not found: {bin_str!r}")

    if p.is_dir():
        cand = p / 'ramulator2'
        if cand.exists():
            p = cand
        else:
            raise FileNotFoundError(f"ramulator2 executable not found in directory: {p}")

    return p.expanduser().resolve()


def _infer_max_seqlen_from_tasks(tasks: List[Dict[str, Any]], cfg: WorkloadConfig) -> int:
    """Infer a safe max_seq_len for trace args from the task list."""
    mx = int(max(1, cfg.prefill_len))
    # Prefer explicit decode_context_lens if present in config.
    if cfg.decode_context_lens:
        try:
            mx = max(mx, int(max(cfg.decode_context_lens)))
        except Exception:
            pass
    # Scan tasks as fallback.
    for t in tasks:
        try:
            ph = str(t.get('sig', {}).get('phase', 'decode'))
            st = int(t.get('sig', {}).get('step', 0))
            ops = t.get('ops', []) or []
            if not ops:
                continue
            op0 = str(ops[0].get('op', ''))
            sh0 = int(ops[0].get('shard', -1))
            shp = infer_op_shape(OpSig(device_type='pim', phase=ph, step=st, op=op0, shard=sh0), cfg)
            mx = max(mx, int(shp.key_len))
        except Exception:
            continue
    return int(max(1, mx))


class PIMBackendViaCostModel:
    """PIM latency backend that delegates to cost_model_pim_backend.

    This unifies op normalization and trace emission so schedule verification
    and CostModel use the exact same implementation (no duplicated emitters).
    """

    def __init__(
        self,
        cfg: WorkloadConfig,
        tasks: List[Dict[str, Any]],
        *,
        debug_logger: _DebugLogger,
        use_cache: bool = True,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for AiM simulator.")
        self.cfg = cfg
        self.debug = bool(getattr(cfg, 'debug', False))
        self._dbg = debug_logger
        self.use_cache = bool(use_cache)

        # Lazy import so accelerator-only flows don't depend on CostModel modules.
        try:
            # Make import robust even when invoked from ./verify.
            try:
                _add_sys_path(_script_dir().parent)
            except Exception:
                pass
            import cost_model_pim_backend as cm_pim_backend  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import cost_model_pim_backend. "
                "Ensure schedule_deploy_verify is run from the repo root (so PYTHONPATH works). "
                f"Import error: {e}"
            ) from e
        self.cm = cm_pim_backend

        # Resolve configs
        if not cfg.pim_ramulator_config:
            raise ValueError("pim_ramulator_config must be provided")
        self.ramulator_config = resolve_existing_path(str(cfg.pim_ramulator_config))

        if not cfg.pim_ramulator_bin:
            raise ValueError("pim_ramulator_bin must be provided")
        self.ramulator_bin = _resolve_ramulator_bin_path(str(cfg.pim_ramulator_bin))

        # cost_model_pim_backend historically uses Path.cwd()/ramulator2;
        # we standardize it via env var.
        os.environ['RAMULATOR2_BIN'] = str(self.ramulator_bin)

        # PIM hw config (pim_config)
        self._tmp_pim_cfg_dir: Optional[Path] = None
        self.pim_config = self._resolve_or_synthesize_pim_config(tasks)

        # Model dict cache (keyed by (dim, heads, kv_heads, ffn_dim, seqlen)).
        self._model_dict_cache: Dict[Tuple[int, int, int, int, int], Dict[str, Any]] = {}
        self.debug_segments: Optional[Dict[str, Any]] = {} if self.debug else None

        self._print_debug_header()

    def _resolve_or_synthesize_pim_config(self, tasks: List[Dict[str, Any]]) -> Path:
        # If user provided a PIM hw JSON path, use it directly.
        if getattr(self.cfg, 'pim_hw_json_path', None):
            return resolve_existing_path(str(self.cfg.pim_hw_json_path))

        # Otherwise synthesize a minimal JSON compatible with cost_model backend.
        max_seq_len = _infer_max_seqlen_from_tasks(tasks, self.cfg)
        pim_cfg = {
            'DRAM_column': int(self.cfg.pim_dram_column),
            'DRAM_row': int(self.cfg.pim_dram_row),
            'burst_length': int(self.cfg.pim_burst_length),
            'num_banks': int(self.cfg.pim_num_banks),
            'num_channels': int(self.cfg.pim_num_channels),
            'threads': int(self.cfg.pim_threads),
            'reuse_size': int(self.cfg.pim_reuse_size),
            'channels_per_block': int(self.cfg.pim_num_channels),
            'max_seq_len': int(max_seq_len),
        }

        self._tmp_pim_cfg_dir = Path(tempfile.mkdtemp(prefix='pim_hw_synth_')).resolve()
        p = self._tmp_pim_cfg_dir / 'pim_hw_synth.json'
        p.write_text(json.dumps(pim_cfg, indent=2), encoding='utf-8')
        if self.debug:
            self._dbg.log(f"[run-pim][debug] synthesized pim_config JSON: {p}")
        return p

    def close(self) -> None:
        if self._tmp_pim_cfg_dir is not None:
            try:
                shutil.rmtree(self._tmp_pim_cfg_dir, ignore_errors=True)
            except Exception:
                pass
            self._tmp_pim_cfg_dir = None

    def _print_debug_header(self) -> None:
        if not self.debug:
            return

        # Try to surface the CostModel constant used for cycles->sec.
        try:
            from config import PIM_FREQ_GHZ as CM_PIM_FREQ_GHZ  # type: ignore
        except Exception:
            CM_PIM_FREQ_GHZ = None  # type: ignore

        self._dbg.log("=" * 80)
        self._dbg.log("[run-pim][debug] Using cost_model_pim_backend._get_pim_latency_via_trace")
        self._dbg.log(f"[run-pim][debug] shard_policy={self.cfg.shard_policy!r} shards={int(self.cfg.shards)}")
        self._dbg.log(f"[run-pim][debug] model: dim={int(self.cfg.dim)} ffn_dim={int(self.cfg.ffn_dim)} n_heads={int(self.cfg.n_heads)} n_kv_heads={(int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else 'auto->n_heads')}")
        self._dbg.log(f"[run-pim][debug] lengths: prefill_len={int(self.cfg.prefill_len)} decode_stride={int(getattr(self.cfg,'decode_stride',1) or 1)} decode_context_lens_len={(len(self.cfg.decode_context_lens) if self.cfg.decode_context_lens else 0)}")
        self._dbg.log(f"[run-pim][debug] pim_config={self.pim_config} (from {'pim_hw_json_path' if self.cfg.pim_hw_json_path else 'synth'})")
        self._dbg.log(f"[run-pim][debug] ramulator_config={self.ramulator_config}")
        self._dbg.log(f"[run-pim][debug] ramulator_bin={self.ramulator_bin} (exported as env RAMULATOR2_BIN)")
        self._dbg.log(f"[run-pim][debug] ramulator_timeout_s={int(getattr(self.cfg,'pim_ramulator_timeout_s',0) or 0)}")
        self._dbg.log(f"[run-pim][debug] cache_enabled={self.use_cache}")
        try:
            self._dbg.log(
                f"[run-pim][debug] PIM_TRACE_SCALE_REPEATS={getattr(self.cm, 'PIM_TRACE_SCALE_REPEATS', None)} "
                f"(env PIM_TRACE_SCALE_REPEATS; if True, simulates a unit decode trace and scales by batch/seqlen)"
            )
        except Exception:
            pass
        self._dbg.log(f"[run-pim][debug] schedule cfg.pim_freq_ghz={float(getattr(self.cfg,'pim_freq_ghz',0.0) or 0.0)}")
        self._dbg.log(f"[run-pim][debug] cost_model config.PIM_FREQ_GHZ={CM_PIM_FREQ_GHZ}")
        self._dbg.log("=" * 80)


    def _pim_params_for_op(
        self,
        *,
        op: str,
        shard: int,
        phase: str,
        step: int,
    ) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        sig = OpSig(device_type='pim', phase=str(phase), step=int(step), op=str(op), shard=int(shard))
        sh = infer_op_shape(sig, self.cfg)

        shards = int(max(1, getattr(self.cfg, 'shards', 1) or 1))
        shard_policy = str(getattr(self.cfg, 'shard_policy', 'fine')).strip().lower()
        fine = shard_policy in ('fine',)

        n_kv_total = int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else int(self.cfg.n_heads)
        dim = int(sh.dim)

        if fine and int(shard) >= 0 and shards > 1:
            n_heads = int(max(1, _split_count_across_shards(int(self.cfg.n_heads), int(shards), int(shard))))
            ffn_dim = int(max(1, sh.ffn_shard_dim))
            n_kv_heads = int(max(1, _split_count_across_shards(int(n_kv_total), int(shards), int(shard))))
        else:
            n_heads = int(max(1, self.cfg.n_heads))
            ffn_dim = int(max(1, self.cfg.ffn_dim))
            n_kv_heads = int(max(1, n_kv_total))

        head_dim = int(max(1, sh.head_dim))
        q_dim = int(max(1, int(n_heads) * int(head_dim)))
        kv_dim = int(max(1, int(n_kv_heads) * int(head_dim)))
        o_dim = int(max(1, int(q_dim)))

        seqlen = int(max(1, sh.key_len))
        qlen = int(max(1, sh.query_len))
        return dim, n_heads, n_kv_heads, ffn_dim, seqlen, qlen, head_dim, q_dim, kv_dim, o_dim

    def _get_model_dict(self, *, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, seqlen: int) -> Dict[str, Any]:
        key = (int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), int(seqlen))
        md = self._model_dict_cache.get(key)
        if md is not None:
            return md

        # Use CostModel's shared model_dict generator to avoid drift.
        md = self.cm._make_shared_model_dict(int(dim), int(n_heads), int(n_kv_heads), int(ffn_dim), int(seqlen))
        self._model_dict_cache[key] = md
        return md

    def benchmark_segment(
        self,
        seg: SegmentSig,
        *,
        seg_key: Optional[str] = None,
        task_meta: Optional[Dict[str, Any]] = None,
    ) -> float:
        # Try to surface the CostModel constant used for cycles->sec.
        try:
            from config import PIM_FREQ_GHZ as CM_PIM_FREQ_GHZ  # type: ignore
        except Exception:
            CM_PIM_FREQ_GHZ = None  # type: ignore

        key_s = seg_key or seg.to_key()
        meta = task_meta or {}
        try:
            cnt = int(meta.get("count_hint", 0))
        except Exception:
            cnt = 0
        ops_repr = meta.get("ops_repr", None)

        # Local helper: format logical tensor shapes (not the internal CENT tensors).
        def _fmt_shape(shape: Tuple[int, ...]) -> str:
            return "x".join(str(int(x)) for x in shape)

        def _op_io_logical(op: str, sh: OpShape) -> Tuple[List[Tuple[str, Tuple[int, ...]]], Tuple[str, Tuple[int, ...]]]:
            """Logical op I/O shapes (aligned with infer_op_shape)."""
            D = int(sh.dim)
            Sd = int(sh.shard_dim)
            Fsd = int(sh.ffn_shard_dim)
            T = int(sh.query_len)
            H = int(sh.heads_per_shard)
            Hd = int(sh.head_dim)
            K = int(sh.key_len)

            if op == "LN":
                return [("x", (T, D)), ("w", (D,))], ("y", (T, D))
            if op in ("Q", "K", "V"):
                return [("x", (T, D)), ("w", (Sd, D))], ("y", (T, Sd))
            if op == "O":
                return [("x", (T, Sd)), ("w", (D, Sd))], ("y", (T, D))
            if op in ("FFN_W1", "FFN_W3"):
                return [("x", (T, D)), ("w", (Fsd, D))], ("y", (T, Fsd))
            if op == "SwiGLU":
                return [("a", (T, Fsd)), ("b", (T, Fsd))], ("y", (T, Fsd))
            if op == "FFN_W2":
                return [("x", (T, Fsd)), ("w", (D, Fsd))], ("y", (T, D))
            if op == "Add":
                return [("a", (T, D)), ("b", (T, D))], ("y", (T, D))
            if op == "QK":
                return [("q", (H, T, Hd)), ("k", (H, Hd, K))], ("y", (H, T, K))
            if op == "Softmax":
                return [("scores", (H, T, K))], ("y", (H, T, K))
            if op == "SV":
                return [("p", (H, T, K)), ("v", (H, K, Hd))], ("y", (H, T, Hd))
            return [("?", tuple())], ("?", tuple())

        # Debug segment header + op list (logical shapes)
        if self.debug:
            self._dbg.log("=" * 88)
            self._dbg.log(f"[pim-debug] segment_key={key_s}")
            self._dbg.log(f"[pim-debug] sig: device_type={seg.device_type} phase={seg.phase} step={seg.step} (count_hint={cnt})")
            if ops_repr:
                self._dbg.log(f"[pim-debug] ops_repr: {ops_repr}")
            self._dbg.log(
                f"[pim-debug] cfg: shard_policy={self.cfg.shard_policy!r} shards={int(self.cfg.shards)} "
                f"dim={int(self.cfg.dim)} ffn_dim={int(self.cfg.ffn_dim)} n_heads={int(self.cfg.n_heads)} "
                f"n_kv_heads={(int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else 'auto->n_heads')}"
            )
            self._dbg.log(f"[pim-debug] pim_config={self.pim_config}")
            self._dbg.log(f"[pim-debug] ramulator_config={self.ramulator_config}")
            self._dbg.log(f"[pim-debug] cost_model config.PIM_FREQ_GHZ={CM_PIM_FREQ_GHZ}")
            self._dbg.log("[pim-debug] op plan (logical shapes inferred from schedule cfg):")
            for j, (op, shard) in enumerate(seg.ops):
                if str(op).strip().lower() in _COMM_OPS:
                    continue
                sig = OpSig(device_type="pim", phase=str(seg.phase), step=int(seg.step), op=str(op), shard=int(shard))
                sh = infer_op_shape(sig, self.cfg)
                ins, out = _op_io_logical(str(op), sh)
                ins_s = " + ".join([f"{n}=[{_fmt_shape(shape)}]" for n, shape in ins if shape])
                out_s = f"{out[0]}=[{_fmt_shape(out[1])}]" if out[1] else out[0]
                self._dbg.log(
                    f"  - op#{j:02d}  op={str(op):<8} shard={int(shard):<2d}  {ins_s}  ->  {out_s}  (T={int(sh.query_len)}, K={int(sh.key_len)})"
                )


        total = 0.0
        per_op_dbg: List[Dict[str, Any]] = []
        base_dim = int(self.cfg.dim)
        base_n_heads = int(self.cfg.n_heads)
        base_n_kv_heads = int(self.cfg.n_kv_heads) if self.cfg.n_kv_heads is not None else int(self.cfg.n_heads)
        base_ffn_dim = int(self.cfg.ffn_dim)

        for j, (op, shard) in enumerate(seg.ops):
            if str(op).strip().lower() in _COMM_OPS:
                continue

            dim, n_heads, n_kv_heads, ffn_dim, seqlen, qlen, head_dim, q_dim, kv_dim, o_dim = self._pim_params_for_op(
                op=str(op), shard=int(shard), phase=str(seg.phase), step=int(seg.step)
            )
            op_norm = self.cm._normalize_pim_op(str(op))

            pim_freq_ghz = float(getattr(self.cfg, 'pim_freq_ghz', 0.0) or 0.0)

            cache_hit = None
            if self.debug and self.use_cache:
                try:
                    cache_phase = str(seg.phase)
                    cache_batch = int(getattr(self.cfg, 'batch', 1) or 1)
                    try:
                        if bool(getattr(self.cm, 'PIM_TRACE_SCALE_REPEATS', False)):
                            cache_phase = 'decode'
                            cache_batch = 1
                    except Exception:
                        pass
                    cache_hit = self.cm._pim_cache.get(
                        op_norm,
                        str(cache_phase),
                        int(dim),
                        int(n_heads),
                        int(n_kv_heads),
                        int(ffn_dim),
                        int(seqlen),
                        int(head_dim),
                        int(q_dim),
                        int(kv_dim),
                        int(o_dim),
                        self.pim_config,
                        self.ramulator_config,
                        batch=int(cache_batch),
                        pim_freq_ghz=float(pim_freq_ghz),
                    )
                except Exception:
                    cache_hit = None

            model_dict = self._get_model_dict(dim=base_dim, n_heads=base_n_heads, n_kv_heads=base_n_kv_heads, ffn_dim=base_ffn_dim, seqlen=seqlen)

            # Provide a stable prefix so traces are easy to correlate (if kept).
            trace_prefix = None
            if self.debug:
                trace_prefix = f"{seg.phase}_step{seg.step}_sh{shard}_{op_norm}"

            sec = float(
                self.cm._get_pim_latency_via_trace(
                    op=op_norm,
                    pim_config=self.pim_config,
                    ramulator_config=self.ramulator_config,
                    dim=int(dim),
                    n_heads=int(n_heads),
                    n_kv_heads=int(n_kv_heads),
                    ffn_dim=int(ffn_dim),
                    seqlen=int(seqlen),
                    batch=int(getattr(self.cfg, 'batch', 1) or 1),
                    phase=str(seg.phase),
                    model_dict=model_dict,
                    use_cache=bool(self.use_cache),
                    head_dim=int(head_dim),
                    q_dim=int(q_dim),
                    kv_dim=int(kv_dim),
                    o_dim=int(o_dim),
                    ramulator_timeout_s=int(getattr(self.cfg, 'pim_ramulator_timeout_s', 300) or 300),
                    keep_traces=bool(getattr(self.cfg, 'pim_keep_traces', False)),
                    trace_dir=(Path(self.cfg.pim_trace_dir).expanduser().resolve() if getattr(self.cfg, 'pim_trace_dir', None) else None),
                    trace_prefix=trace_prefix,
                    pim_freq_ghz=float(pim_freq_ghz),
                )
            )
            total += sec

            per_op_dbg.append(
                {
                    "idx": int(j),
                    "op": str(op),
                    "op_norm": str(op_norm),
                    "shard": int(shard),
                    "phase": str(seg.phase),
                    "step": int(seg.step),
                    "dim": int(dim),
                    "n_heads": int(n_heads),
                    "head_dim": int(head_dim),
                    "q_dim": int(q_dim),
                    "kv_dim": int(kv_dim),
                    "o_dim": int(o_dim),
                    "n_kv_heads": int(n_kv_heads),
                    "ffn_dim": int(ffn_dim),
                    "seqlen": int(seqlen),
                    "qlen": int(qlen),
                    "cache_hit": bool(cache_hit is not None) if self.use_cache else False,
                    "latency_s": float(sec),
                    "trace_prefix": str(trace_prefix) if trace_prefix else None,
                }
            )

            if self.debug:
                self._dbg.log(
                    f"[pim-debug] op#{j:02d} op={str(op):<8} norm={op_norm:<10} sh={int(shard):<2d} "
                    f"dim={int(dim):<5d} heads={int(n_heads):<3d} hd={int(head_dim):<3d} q_dim={int(q_dim):<5d} kv_heads={int(n_kv_heads):<3d} kv_dim={int(kv_dim):<5d} o_dim={int(o_dim):<5d} ffn_dim={int(ffn_dim):<6d} "
                    f"seqlen={int(seqlen):<6d} qlen={int(qlen):<3d} cache={'Y' if (cache_hit is not None and self.use_cache) else 'N'} "
                )

        if self.debug:
            self._dbg.log(f"[pim-debug] segment total latency: {total*1e3:.3f} ms")

        if self.debug_segments is not None:
            self.debug_segments[key_s] = {
                "segment_key": key_s,
                "device_type": str(seg.device_type),
                "phase": str(seg.phase),
                "step": int(seg.step),
                "count_hint": int(cnt),
                "ops_repr": str(ops_repr) if ops_repr is not None else seg.ops_repr(),
                "total_s": float(total),
                "per_op": per_op_dbg,
            }

        return float(total)
# ==========================================================
# JSON helpers
# ==========================================================
def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _load_json(path: str) -> Dict[str, Any]:
    p = resolve_existing_path(path)
    return json.loads(p.read_text(encoding="utf-8"))


# ==========================================================
# Model shape JSON helpers
# ==========================================================
_MODEL_DIM_KEYS = [
    "hidden_dim",
    "hidden_size",
    "dim",
    "d_model",
    "model_dim",
    "n_embd",
]
_MODEL_FFN_KEYS = [
    "intermediate_dim",
    "intermediate_size",
    "ffn_dim",
    "ffn_hidden_dim",
    "mlp_dim",
]
_MODEL_NHEAD_KEYS = [
    "q_head_num",
    "num_attention_heads",
    "n_heads",
    "num_heads",
    "head_num",
    "n_head",
]
_MODEL_KV_NHEAD_KEYS = [
    "kv_head_num",
    "num_key_value_heads",
    "n_kv_heads",
    "num_kv_heads",
    "kv_heads",
]


def load_model_shape_json(path: str) -> Dict[str, int]:
    """Load a model shape/config JSON and return {dim, ffn_dim, n_heads, n_kv_heads}."""
    p = resolve_existing_path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"model cfg must be a JSON object (dict), got {type(obj)}: {p}")

    lower = {str(k).lower(): v for k, v in obj.items()}

    def _pick_int(keys: List[str]) -> Optional[int]:
        for k in keys:
            if k.lower() in lower:
                v = lower[k.lower()]
                try:
                    return int(v)
                except Exception:
                    continue
        return None

    dim = _pick_int(_MODEL_DIM_KEYS)
    ffn = _pick_int(_MODEL_FFN_KEYS)
    n_heads = _pick_int(_MODEL_NHEAD_KEYS)
    n_kv_heads = _pick_int(_MODEL_KV_NHEAD_KEYS)

    missing: List[str] = []
    if dim is None:
        missing.append("dim")
    if ffn is None:
        missing.append("ffn_dim")
    if n_heads is None:
        missing.append("n_heads")
    if missing:
        raise ValueError(
            f"model cfg missing required field(s): {missing}. "
            f"Got keys={sorted(list(lower.keys()))}. File={p}"
        )

    dim = int(dim)
    ffn = int(ffn)
    n_heads = int(n_heads)
    if n_kv_heads is None:
        n_kv_heads = int(n_heads)
    else:
        n_kv_heads = int(n_kv_heads)

    if dim <= 0 or ffn <= 0 or n_heads <= 0 or n_kv_heads <= 0:
        raise ValueError(
            f"Invalid model cfg values in {p}: dim={dim}, ffn_dim={ffn}, n_heads={n_heads}, n_kv_heads={n_kv_heads}"
        )
    if dim % n_heads != 0:
        raise ValueError(f"hidden dim {dim} must be divisible by n_heads {n_heads} (file={p})")
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads {n_heads} must be divisible by n_kv_heads {n_kv_heads} for GQA/MQA (file={p})"
        )

    return {
        "dim": int(dim),
        "ffn_dim": int(ffn),
        "n_heads": int(n_heads),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(dim // n_heads),
    }

def load_pim_hw_json(path: str) -> Dict[str, Any]:
    """Load and normalize a minimal PIM HW spec JSON.
    Required keys (any alias below is accepted):
      - DRAM_column / dram_column / column / columns
      - DRAM_row / dram_row / row / rows
      - burst_length / burst / bl
      - num_banks / banks
      - num_channels / channels
      - pim_threads / threads
      - pim_reuse_size / reuse_size
      - pim_freq_ghz / freq_ghz / frequency_ghz
    """
    p = resolve_existing_path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"PIM HW json must be a JSON object (dict), got {type(obj)}: {p}")

    # case-insensitive lookup table
    lower = {str(k).lower(): v for k, v in obj.items()}

    def _get_any(keys: List[str], *, required: bool = True) -> Any:
        for k in keys:
            lk = str(k).lower()
            if lk in lower and lower[lk] is not None:
                return lower[lk]
        if required:
            raise KeyError(f"Missing required key in {p}: one of {keys}")
        return None

    def _as_pos_int(name: str, v: Any) -> int:
        try:
            iv = int(v)
        except Exception as e:
            raise ValueError(f"{name} must be an integer, got {v!r}") from e
        if iv <= 0:
            raise ValueError(f"{name} must be > 0, got {iv}")
        return iv

    def _as_pos_float(name: str, v: Any) -> float:
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError(f"{name} must be a float, got {v!r}") from e
        if fv <= 0:
            raise ValueError(f"{name} must be > 0, got {fv}")
        return fv

    hw: Dict[str, Any] = {
        "pim_dram_column": _as_pos_int("pim_dram_column", _get_any(["dram_column", "dram_col", "column", "columns", "dram_column".upper()])),
        "pim_dram_row": _as_pos_int("pim_dram_row", _get_any(["dram_row", "row", "rows", "dram_row".upper()])),
        "pim_burst_length": _as_pos_int("pim_burst_length", _get_any(["burst_length", "burst", "bl"])),
        "pim_num_banks": _as_pos_int("pim_num_banks", _get_any(["num_banks", "banks"])),
        "pim_num_channels": _as_pos_int("pim_num_channels", _get_any(["num_channels", "channels"])),
        "pim_threads": None,
        "pim_reuse_size": None,
        "pim_freq_ghz": None,
    }

    v = _get_any(["pim_num_devices", "num_devices", "devices", "fc_devices"], required=False)
    if v is not None:
        raise ValueError(
            f"{p}: device-count keys (pim_num_devices/num_devices/devices/FC_devices) are no longer supported in --pim-hw-json. "
            "Pass device count via CLI: run-pim --pim-num-devices N."
        )

    v = _get_any(["pim_threads", "threads"], required=False)
    if v is not None:
        hw["pim_threads"] = _as_pos_int("pim_threads", v)

    v = _get_any(["pim_reuse_size", "reuse_size"], required=False)
    if v is not None:
        hw["pim_reuse_size"] = _as_pos_int("pim_reuse_size", v)

    v = _get_any(["pim_freq_ghz", "freq_ghz", "frequency_ghz"], required=False)
    if v is not None:
        hw["pim_freq_ghz"] = _as_pos_float("pim_freq_ghz", v)

    return hw

# ==========================================================
# Export / Run / Merge
# ==========================================================
def export_tasks(
    schedule_paths: List[str],
    cfg: WorkloadConfig,
    out_dir: str,
    prefix: str,
    *,
    comms_paths: Optional[List[str]] = None,
) -> Tuple[Path, Path]:
    outp = Path(out_dir).expanduser().resolve()
    outp.mkdir(parents=True, exist_ok=True)

    # load schedules
    dfs: List[pd.DataFrame] = []
    resolved_paths: List[Path] = []
    max_decode_steps = 0
    max_shard_seen = -1
    for sp in schedule_paths:
        p = resolve_existing_path(sp)
        resolved_paths.append(p)
        df = load_schedule_csv(str(p))
        df = annotate_decode_sampling(df, decode_stride=int(getattr(cfg, "decode_stride", 1) or 1))
        dfs.append(df)
        max_decode_steps = max(max_decode_steps, infer_decode_steps(df))

        # infer max shard index across all schedules
        if "shard" in df.columns:
            ms = pd.to_numeric(df["shard"], errors="coerce").fillna(-1).astype(int).max()
            max_shard_seen = max(max_shard_seen, int(ms))

    inferred = int(max_shard_seen + 1) if int(max_shard_seen) >= 0 else 1
    if int(getattr(cfg, "shards", 0) or 0) != inferred:
        print(f"[export] inferred shards={inferred} from schedule (max_shard={max_shard_seen}), override cfg.shards={cfg.shards}")
    cfg.shards = inferred

    # optional comms traces (kept for kv_load / fallback accounting)
    resolved_comms: List[Path] = []
    if comms_paths:
        for cp in comms_paths:
            if not cp:
                continue
            resolved_comms.append(resolve_existing_path(cp))

    wl_phase = extract_weight_load_seconds_by_phase([str(p) for p in resolved_comms])
    wl = wl_phase.get("all", {}) if isinstance(wl_phase, dict) else {}
    weight_load_npu_s = float(wl.get("npu_s", 0.0) or 0.0)
    weight_load_pim_s = float(wl.get("pim_s", 0.0) or 0.0)
    weight_load_unknown_s = float(wl.get("unknown_s", 0.0) or 0.0)

    weight_load_by_schedule: Dict[str, Dict[str, float]] = {}
    weight_load_phase_by_schedule: Dict[str, Dict[str, Dict[str, float]]] = {}
    if comms_paths and len(resolved_comms) == len(resolved_paths):
        for sp, cp in zip(resolved_paths, resolved_comms):
            by_phase = extract_weight_load_seconds_by_phase([str(cp)])
            w = by_phase.get("all", {}) if isinstance(by_phase, dict) else {}
            weight_load_by_schedule[str(sp)] = {
                "npu_s": float(w.get("npu_s", 0.0) or 0.0),
                "pim_s": float(w.get("pim_s", 0.0) or 0.0),
                "unknown_s": float(w.get("unknown_s", 0.0) or 0.0),
                "total_s": float(w.get("total_s", 0.0) or 0.0),
            }
            weight_load_phase_by_schedule[str(sp)] = {
                str(ph): {
                    "npu_s": float((rec or {}).get("npu_s", 0.0) or 0.0),
                    "pim_s": float((rec or {}).get("pim_s", 0.0) or 0.0),
                    "unknown_s": float((rec or {}).get("unknown_s", 0.0) or 0.0),
                    "total_s": float((rec or {}).get("total_s", 0.0) or 0.0),
                }
                for ph, rec in (by_phase or {}).items()
                if str(ph) != "all"
            }

    if cfg.decode_context_lens is None:
        stride = int(getattr(cfg, "decode_stride", 1) or 1)
        if stride <= 0:
            stride = 1
        inferred_dcl: List[int] = []
        for df in dfs:
            dcl_i = infer_decode_context_lens_from_schedule(df, cfg, decode_stride=stride)
            if len(dcl_i) > len(inferred_dcl):
                inferred_dcl = list(dcl_i)
        if inferred_dcl:
            cfg.decode_context_lens = inferred_dcl
        else:
            cfg.decode_context_lens = (
                [
                    int(cfg.prefill_len + 1 + int(decode_token_index_from_sample_step(int(i), int(stride))))
                    for i in range(max_decode_steps)
                ]
                if max_decode_steps > 0
                else []
            )

    uniq: Dict[str, SegmentSig] = {}
    ctr: Counter = Counter()

    for df in dfs:
        u, c = extract_segments(df, cfg.segment_scope)
        uniq.update(u)
        ctr.update(c)

    npu_tasks: List[Dict[str, Any]] = []
    pim_tasks: List[Dict[str, Any]] = []
    skipped_tasks: List[Dict[str, Any]] = []
    for k, seg in uniq.items():
        item = {
            "key": k,
            "sig": {"device_type": seg.device_type, "phase": seg.phase, "step": int(seg.step)},
            "ops": [{"op": op, "shard": int(shard)} for op, shard in seg.ops],
            "ops_repr": seg.ops_repr(),
            "count_hint": int(ctr.get(k, 0)),
        }
        dev_type = str(seg.device_type).strip().lower()
        if dev_type == "npu":
            npu_tasks.append(item)
        elif dev_type == "pim":
            pim_tasks.append(item)
        else:
            skipped_tasks.append(item)

    npu_json = {
        "version": 4,
        "task_type": "segment",
        "backend": "npu",
        "segment_scope": cfg.segment_scope,
        "schedules": [str(p) for p in resolved_paths],
        "weight_load_s": float(weight_load_npu_s),
        "weight_load_meta": {
            "npu_s": float(weight_load_npu_s),
            "pim_s": float(weight_load_pim_s),
            "unknown_s": float(weight_load_unknown_s),
            "total_s": float(weight_load_npu_s + weight_load_pim_s),
        },
        "weight_load_by_schedule": weight_load_by_schedule,
        "weight_load_phase_by_schedule": weight_load_phase_by_schedule,
        "config": cfg.to_dict(),
        "tasks": sorted(npu_tasks, key=lambda x: x["key"]),
    }
    pim_json = {
        "version": 4,
        "task_type": "segment",
        "backend": "pim",
        "segment_scope": cfg.segment_scope,
        "schedules": [str(p) for p in resolved_paths],
        "comms_traces": [str(p) for p in resolved_comms],
        "weight_load_s": float(weight_load_pim_s),
        "weight_load_meta": {
            "npu_s": float(weight_load_npu_s),
            "pim_s": float(weight_load_pim_s),
            "unknown_s": float(weight_load_unknown_s),
            "total_s": float(weight_load_npu_s + weight_load_pim_s),
        },
        "weight_load_by_schedule": weight_load_by_schedule,
        "weight_load_phase_by_schedule": weight_load_phase_by_schedule,
        "config": cfg.to_dict(),
        "tasks": sorted(pim_tasks, key=lambda x: x["key"]),
    }

    npu_path = outp / f"{prefix}.npu_tasks.json"
    pim_path = outp / f"{prefix}.pim_tasks.json"
    _save_json(npu_path, npu_json)
    _save_json(pim_path, pim_json)

    print(f"[export] segment_scope={cfg.segment_scope}")
    if resolved_comms:
        print(
            f"[export] weight_load_s: npu={weight_load_npu_s:.6f}s pim={weight_load_pim_s:.6f}s "
            f"(unknown={weight_load_unknown_s:.6f}s) from {len(resolved_comms)} comms trace(s)"
        )
    print(f"[export] wrote {npu_path} ({len(npu_tasks)} segments)")
    print(f"[export] wrote {pim_path} ({len(pim_tasks)} segments)")
    if skipped_tasks:
        by_type: Counter = Counter([str(t.get("sig", {}).get("device_type", "")).strip().lower() for t in skipped_tasks])
        by_type_s = ", ".join([f"{k or '<empty>'}={v}" for k, v in by_type.items()])
        print(
            f"[export] skipped {len(skipped_tasks)} trace-only segment(s) (not npu/pim). "
            f"They will be kept via schedule durations in merge(). by_device_type: {by_type_s}"
        )
    return npu_path, pim_path


# ==========================================================
# NPU LUT backend (Ascend) + run-npu
# ==========================================================
_NPU_LUT_BACKEND = None

def _get_npu_lut_backend():
    """Lazy import for cost_model_npu_ascend_backend (keep default npu flow unaffected)."""
    global _NPU_LUT_BACKEND
    if _NPU_LUT_BACKEND is not None:
        return _NPU_LUT_BACKEND
    try:
        import cost_model_npu_ascend_backend as m  # type: ignore
        _NPU_LUT_BACKEND = m
        return _NPU_LUT_BACKEND
    except Exception as e1:
        # Fallback: load from file colocated with this script
        try:
            p = Path(__file__).resolve().parent / "cost_model_npu_ascend_backend.py"
        except Exception:
            p = Path("cost_model_npu_ascend_backend.py").resolve()
        if p.exists() and p.is_file():
            try:
                spec = importlib.util.spec_from_file_location("cost_model_npu_ascend_backend", str(p))
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"spec_from_file_location failed for {p}")
                mod = importlib.util.module_from_spec(spec)
                sys.modules["cost_model_npu_ascend_backend"] = mod
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                _NPU_LUT_BACKEND = mod
                return _NPU_LUT_BACKEND
            except Exception as e2:
                raise RuntimeError(
                    "Failed to import cost_model_npu_ascend_backend. "
                    "Make sure cost_model_npu_ascend_backend.py is in the same folder as this script, "
                    "and that your PYTHONPATH contains the repo root (for its dependencies like config.py)."
                ) from e2
        raise RuntimeError(
            "Failed to import cost_model_npu_ascend_backend. "
            "Make sure it is importable (same folder or PYTHONPATH)."
        ) from e1

_NPU_COST_MODEL_MODULES = None

def _get_algorithm_npu_cost_model_modules():
    """Import the shared algorithm-side NPU CostModel modules."""
    global _NPU_COST_MODEL_MODULES
    if _NPU_COST_MODEL_MODULES is not None:
        return _NPU_COST_MODEL_MODULES

    roots = []
    try:
        repo_root = _script_dir().parent
        roots.extend([repo_root, repo_root / 'algorithms', repo_root / 'algorithm'])
    except Exception:
        pass

    for r in roots:
        try:
            rp = Path(r).expanduser().resolve()
            if rp.exists():
                _add_sys_path(rp)
        except Exception:
            continue

    try:
        import cost_model as cm_mod  # type: ignore
        import hardware as hw_mod  # type: ignore
        import task_graph as tg_mod  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import shared algorithm modules (cost_model / hardware / task_graph). "
            "Ensure PYTHONPATH includes the repo root and ./algorithms."
        ) from e

    _NPU_COST_MODEL_MODULES = (cm_mod, hw_mod, tg_mod)
    return _NPU_COST_MODEL_MODULES


def _set_npu_lut_env(
    *,
    mmad_lut: Optional[str] = None,
    softmax_lut: Optional[str] = None,
    gelu_lut: Optional[str] = None,
    norm_lut: Optional[str] = None,
) -> None:
    """Set LUT override env vars so the backend can locate LUT files."""
    if mmad_lut:
        os.environ["TRIFORM_MMAD_LUT"] = str(mmad_lut)
    if softmax_lut:
        os.environ["TRIFORM_SOFTMAX_LUT"] = str(softmax_lut)
    if gelu_lut:
        os.environ["TRIFORM_GELU_LUT"] = str(gelu_lut)
    if norm_lut:
        os.environ["TRIFORM_NORM_LUT"] = str(norm_lut)



def infer_compute_backend_from_schedules(schedule_paths: List[str]) -> str:
    has_npu = False
    for sp in schedule_paths:
        try:
            df = pd.read_csv(sp, usecols=["device"])
            devs = df.get("device")
        except Exception:
            # fallback: full load + column selection
            try:
                df2 = load_schedule_csv(str(sp))
                devs = df2.get("device")
            except Exception:
                devs = None

        if devs is None:
            continue

        try:
            s = devs.astype(str).str.lower()
        except Exception:
            s = pd.Series([str(x).lower() for x in list(devs)])

        try:
            if bool(s.str.contains("npu", regex=False).any()):
                has_npu = True
        except Exception:
            joined = " ".join(list(map(str, list(devs)))).lower()
            if "npu" in joined:
                has_npu = True
    if has_npu:
        return "npu"
    if has_npu:
        return "npu"
    return "unknown"


_NPU_DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "int8": 1,
}


@dataclass
class _NpuOpParams:
    dim: int
    ffn_dim: int
    q_heads: int
    kv_heads: int
    head_dim: int
    q_len: int
    kv_len: int
    seq_len: int
    q_dim: int
    kv_dim: int
    o_dim: int
    batch: int


class NPUBackendViaCostModel:
    """NPU segment latency backend that reuses algorithms/cost_model.py.

    Weighted linear ops are re-evaluated from the shared CostModel after segment
    extraction. Their contribution is the algorithm-side B-stage time:
        max(b1, b2)

    `weight_load` is NOT reconstructed here. It is taken directly from comms.csv
    during merge() and combined with B-stage using the merge-time overlap knob.
    """

    _WEIGHTED_LINEAR_OPS = {"Q", "K", "V", "O", "FFN_W1", "FFN_W2", "FFN_W3"}

    def __init__(
        self,
        cfg: WorkloadConfig,
        *,
        debug_logger: _DebugLogger,
        npu_dtype: str = "fp16",
        npu_mem_bw_gbs: float = 0.0,
        op_overhead_us: float = 0.0,
    ):
        self.cfg = cfg
        self._dbg = debug_logger
        self.debug = bool(getattr(cfg, "debug", False)) or bool(getattr(debug_logger, "enabled", False))

        self.npu_dtype = str(npu_dtype).lower().strip()
        if self.npu_dtype not in _NPU_DTYPE_BYTES:
            raise ValueError(f"Unsupported --npu-dtype: {npu_dtype!r}. Supported: {sorted(_NPU_DTYPE_BYTES.keys())}")

        self.npu_mem_bw_gbs = float(npu_mem_bw_gbs)
        if self.npu_mem_bw_gbs <= 0.0:
            raise ValueError(
                "Memory BW (mem bound) is mandatory for NPU verification. "
                "Please provide a positive --npu-mem-bw-gbs (GB/s)."
            )
        self.op_overhead_us = float(op_overhead_us)

        self.cm_mod, self.hw_mod, self.tg_mod = _get_algorithm_npu_cost_model_modules()

        cluster = self.hw_mod.Cluster()
        self.dev = self.hw_mod.DeviceSpec(
            name="NPU0",
            type="npu",
            tflops=1.0,
            mem_bw_GBs=float(self.npu_mem_bw_gbs),
            mem_capacity_GB=1.0,
            arch="Ascend_310B",
        )
        cluster.add_device(self.dev)

        self.cm = self.cm_mod.CostModel(
            cluster=cluster,
            dtype=str(self.npu_dtype),
            npu_backend="ascend_310b_lut",
        )
        self._label = SimpleNamespace(kv_in_pim=False)
        self.debug_segments: Optional[Dict[str, Any]] = {} if self.debug else None

        shard_policy = str(getattr(self.cfg, "shard_policy", "fine") or "").strip().lower()
        if shard_policy not in ("", "fine"):
            raise ValueError(
                f"Only shard_policy='fine' is supported in verify (got {getattr(self.cfg, 'shard_policy', None)!r})."
            )

        if self.debug:
            self._print_debug_header()

    def _print_debug_header(self) -> None:
        backend_file = None
        try:
            backend_file = getattr(_get_npu_lut_backend(), "__file__", None)
        except Exception:
            backend_file = None

        self._dbg.log("=" * 80)
        self._dbg.log("[run-npu][debug] Using shared algorithms/cost_model.py Ascend LUT backend")
        self._dbg.log("[run-npu][debug] weighted linear ops use B-stage = max(b1, b2); weight_load is merged later from comms.csv")
        if backend_file:
            self._dbg.log(f"[run-npu][debug] lut_backend_file={backend_file}")
        self._dbg.log(
            f"[run-npu][debug] npu_dtype={self.npu_dtype} npu_mem_bw_gbs={self.npu_mem_bw_gbs} op_overhead_us={self.op_overhead_us}"
        )
        self._dbg.log(
            f"[run-npu][debug] model: dim={int(self.cfg.dim)} ffn_dim={int(self.cfg.ffn_dim)} n_heads={int(self.cfg.n_heads)} "
            f"n_kv_heads={(int(self.cfg.n_kv_heads) if getattr(self.cfg,'n_kv_heads',None) is not None else 'auto->n_heads')} "
            f"shards={int(self.cfg.shards)} shard_policy=fine batch={int(getattr(self.cfg,'batch',1) or 1)}"
        )
        self._dbg.log("=" * 80)

    def _infer_params(self, *, shard: int, sh: OpShape) -> _NpuOpParams:
        dim = int(getattr(self.cfg, "dim"))
        ffn_dim_total = int(getattr(self.cfg, "ffn_dim"))
        n_heads_total = int(getattr(self.cfg, "n_heads"))
        n_kv_total = int(getattr(self.cfg, "n_kv_heads", None) or n_heads_total)
        shards = int(max(1, getattr(self.cfg, "shards", 1) or 1))

        q_len = int(max(1, getattr(sh, "query_len")))
        kv_len = int(max(1, getattr(sh, "key_len")))
        seq_len = int(kv_len)
        head_dim = int(max(1, getattr(sh, "head_dim")))
        batch = int(max(1, getattr(self.cfg, "batch", 1) or 1))

        if int(shard) >= 0 and shards > 1:
            q_heads = int(max(1, _split_count_across_shards(int(n_heads_total), int(shards), int(shard))))
            kv_heads = int(max(1, _split_count_across_shards(int(n_kv_total), int(shards), int(shard))))
            ffn_dim = int(max(1, getattr(sh, "ffn_shard_dim")))
        else:
            q_heads = int(max(1, n_heads_total))
            kv_heads = int(max(1, n_kv_total))
            ffn_dim = int(max(1, ffn_dim_total))

        q_dim = int(max(1, q_heads * head_dim))
        kv_dim = int(max(1, kv_heads * head_dim))
        o_dim = int(max(1, q_dim))

        return _NpuOpParams(
            dim=dim,
            ffn_dim=ffn_dim,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            q_len=q_len,
            kv_len=kv_len,
            seq_len=seq_len,
            q_dim=q_dim,
            kv_dim=kv_dim,
            o_dim=o_dim,
            batch=batch,
        )

    def _make_node(self, sig: OpSig, sh: OpShape):
        p = self._infer_params(shard=int(sig.shard), sh=sh)
        attrs = {
            "dim": int(p.dim),
            "ffn_dim": int(p.ffn_dim),
            "q_heads": int(p.q_heads),
            "n_heads": int(p.q_heads),
            "kv_heads": int(p.kv_heads),
            "n_kv_heads": int(p.kv_heads),
            "head_dim": int(p.head_dim),
            "q_dim": int(p.q_dim),
            "kv_dim": int(p.kv_dim),
            "o_dim": int(p.o_dim),
            "causal": True,
            "batch": int(p.batch),
        }
        node = self.tg_mod.TaskNode(
            id=f"{sig.phase}_{int(sig.step)}_{sig.op}_{int(sig.shard)}",
            name=str(sig.op),
            attrs=attrs,
        )
        return node, p

    def resolve_weight_resident_fmt(
        self,
        *,
        resident_fmt: Optional[str] = None,
        host_src_fmt: Optional[str] = None,
        host_storage_fmt: Optional[str] = None,
    ) -> str:
        rf = str(resident_fmt or "").strip()
        if rf and rf.lower() != "nan":
            return rf

        hs = str(host_src_fmt or "").strip()
        if hs and hs.lower() != "nan":
            try:
                rf = str(self.cm.weight_resident_format(str(hs), self.dev))
                if rf:
                    return rf
            except Exception:
                pass

        src = str(host_storage_fmt or "").strip()
        if src and src.lower() != "nan":
            try:
                hs2 = str(self.cm.weight_host_source_format(str(src), self.dev))
                rf2 = str(self.cm.weight_resident_format(str(hs2), self.dev))
                if rf2:
                    return rf2
            except Exception:
                pass

        try:
            hs3 = str(self.cm.weight_host_source_format("ND", self.dev))
            rf3 = str(self.cm.weight_resident_format(str(hs3), self.dev))
            if rf3:
                return rf3
        except Exception:
            pass
        return "ND"

    def _is_weighted_linear_op(self, op: str) -> bool:
        return str(op).strip().upper() in self._WEIGHTED_LINEAR_OPS

    def _infer_weight_size_nd(self, sig: OpSig, sh: OpShape) -> int:
        op_u = str(sig.op).strip().upper()
        dim = int(getattr(self.cfg, "dim"))
        ffn_dim = int(getattr(self.cfg, "ffn_dim"))
        n_heads_total = int(getattr(self.cfg, "n_heads"))
        n_kv_total = int(getattr(self.cfg, "n_kv_heads", None) or n_heads_total)
        head_dim = int(max(1, getattr(sh, "head_dim")))

        q_dim_total = int(n_heads_total * head_dim)
        kv_dim_total = int(n_kv_total * head_dim)
        o_dim_total = int(q_dim_total)

        if op_u == "Q":
            return int(dim * q_dim_total)
        if op_u in ("K", "V"):
            return int(dim * kv_dim_total)
        if op_u == "O":
            return int(o_dim_total * dim)
        if op_u in ("FFN_W1", "FFN_W3"):
            return int(dim * ffn_dim)
        if op_u == "FFN_W2":
            return int(ffn_dim * dim)
        return 0

    def estimate_weighted_b_stage_s(
        self,
        *,
        sig: OpSig,
        sh: OpShape,
    ) -> Tuple[float, Dict[str, Any]]:
        if not self._is_weighted_linear_op(str(sig.op)):
            raise ValueError(f"{sig.op!r} is not a weighted linear op")

        node, p = self._make_node(sig, sh)
        weight_size_nd = int(max(0, self._infer_weight_size_nd(sig, sh)))
        node.weight_size = int(weight_size_nd)
        node.weight_id = f"W::{str(sig.op).upper()}"

        resident_fmt = self.resolve_weight_resident_fmt(host_storage_fmt="ND")
        seq_len = int(max(1, p.seq_len))

        stage = self.cm.weighted_compute_stage(
            node,
            self.dev,
            self._label,
            int(p.batch),
            int(seq_len),
            str(sig.phase),
            resident_weight_fmt=str(resident_fmt),
        )

        b1_s = float(getattr(stage, "b1_s", 0.0) or 0.0)
        b2_s = float(getattr(stage, "b2_s", 0.0) or 0.0)
        b_s = float(max(b1_s, b2_s))

        return float(b_s), {
            "op": str(sig.op),
            "op_key": str(self.cm_mod._normalize_npu_op_key(str(sig.op))),
            "b1_s": float(b1_s),
            "b2_s": float(b2_s),
            "b_stage_s": float(b_s),
            "weight_size_nd": int(weight_size_nd),
            "resident_fmt": str(resident_fmt),
            "compute_rule": "max_b1_b2",
            "params": {
                "dim": int(p.dim),
                "ffn_dim": int(p.ffn_dim),
                "q_heads": int(p.q_heads),
                "kv_heads": int(p.kv_heads),
                "head_dim": int(p.head_dim),
                "q_len": int(p.q_len),
                "kv_len": int(p.kv_len),
                "q_dim": int(p.q_dim),
                "kv_dim": int(p.kv_dim),
                "o_dim": int(p.o_dim),
                "batch": int(p.batch),
            },
        }

    def estimate_op_s(
        self,
        *,
        sig: OpSig,
        sh: OpShape,
    ) -> Tuple[float, Dict[str, Any]]:
        node, p = self._make_node(sig, sh)
        seq_len = int(max(1, p.seq_len))

        rd, wr = self.cm.estimate_activation_bytes(node, int(p.batch), int(seq_len), str(sig.phase))
        mem_s = float(self.cm.mem_time(int(rd + wr), self.dev))
        op_key = str(self.cm_mod._normalize_npu_op_key(str(sig.op)))

        sec = float(self.cm.node_device_cost(node, self.dev, self._label, int(p.batch), int(seq_len), str(sig.phase)))
        sec += float(self.op_overhead_us) * 1e-6

        return float(sec), {
            "op": str(sig.op),
            "op_key": str(op_key),
            "mem_s": float(mem_s),
            "overhead_us": float(self.op_overhead_us),
            "lat_s": float(sec),
            "source": "plain_cost_model",
            "params": {
                "dim": int(p.dim),
                "ffn_dim": int(p.ffn_dim),
                "q_heads": int(p.q_heads),
                "kv_heads": int(p.kv_heads),
                "head_dim": int(p.head_dim),
                "q_len": int(p.q_len),
                "kv_len": int(p.kv_len),
                "q_dim": int(p.q_dim),
                "kv_dim": int(p.kv_dim),
                "o_dim": int(p.o_dim),
                "batch": int(p.batch),
            },
        }

    def benchmark_segment(
        self,
        seg: SegmentSig,
        *,
        seg_key: Optional[str] = None,
        task_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, Optional[Dict[str, Any]]]:
        key_s = seg_key or seg.to_key()
        meta = task_meta or {}

        total_s = 0.0
        total_b_stage_s = 0.0
        per_op: List[Dict[str, Any]] = []

        if self.debug:
            self._dbg.log("=" * 88)
            self._dbg.log(f"[npu-debug] segment_key={key_s}")
            self._dbg.log(f"[npu-debug] sig: device_type={seg.device_type} phase={seg.phase} step={seg.step} (count_hint={meta.get('count_hint',0)})")
            ops_repr = meta.get("ops_repr", None)
            if ops_repr:
                self._dbg.log(f"[npu-debug] ops_repr: {ops_repr}")
            self._dbg.log("[npu-debug] op plan (logical shapes inferred from schedule cfg):")

        plan: List[Tuple[OpSig, OpShape]] = []
        for op, shard in seg.ops:
            sig = OpSig(device_type=seg.device_type, phase=seg.phase, step=seg.step, op=op, shard=shard)
            sh = infer_op_shape(sig, self.cfg)
            plan.append((sig, sh))

            if self.debug and str(op).strip().lower() not in _COMM_OPS:
                D = int(sh.dim)
                Sd = int(sh.shard_dim)
                Fsd = int(sh.ffn_shard_dim)
                T = int(sh.query_len)
                H = int(sh.heads_per_shard)
                Hd = int(sh.head_dim)
                K = int(sh.key_len)

                def _fmt_shape(tup: Tuple[int, ...]) -> str:
                    return "x".join(str(int(x)) for x in tup)

                if op == "LN":
                    ins = [("x", (T, D)), ("w", (D,))]
                    out = ("y", (T, D))
                elif op in ("Q", "K", "V"):
                    ins = [("x", (T, D)), ("w", (Sd, D))]
                    out = ("y", (T, Sd))
                elif op == "O":
                    ins = [("x", (T, Sd)), ("w", (D, Sd))]
                    out = ("y", (T, D))
                elif op in ("FFN_W1", "FFN_W3"):
                    ins = [("x", (T, D)), ("w", (Fsd, D))]
                    out = ("y", (T, Fsd))
                elif op == "SwiGLU":
                    ins = [("a", (T, Fsd)), ("b", (T, Fsd))]
                    out = ("y", (T, Fsd))
                elif op == "FFN_W2":
                    ins = [("x", (T, Fsd)), ("w", (D, Fsd))]
                    out = ("y", (T, D))
                elif op == "Add":
                    ins = [("a", (T, D)), ("b", (T, D))]
                    out = ("y", (T, D))
                elif op == "QK":
                    ins = [("q", (H, T, Hd)), ("k", (H, Hd, K))]
                    out = ("y", (H, T, K))
                elif op == "Softmax":
                    ins = [("scores", (H, T, K))]
                    out = ("y", (H, T, K))
                elif op == "SV":
                    ins = [("p", (H, T, K)), ("v", (H, K, Hd))]
                    out = ("y", (H, T, Hd))
                else:
                    ins = [("?", tuple())]
                    out = ("?", tuple())

                ins_s = " + ".join([f"{n}=[{_fmt_shape(shape)}]" for n, shape in ins if shape])
                out_s = f"{out[0]}=[{_fmt_shape(out[1])}]" if out[1] else out[0]
                self._dbg.log(f"  - op op={str(op):<8} shard={int(shard):<2d}  {ins_s}  ->  {out_s}  (T={int(T)}, K={int(K)})")

        for j, (sig, sh) in enumerate(plan):
            if self._is_weighted_linear_op(str(sig.op)):
                sec, dbg = self.estimate_weighted_b_stage_s(sig=sig, sh=sh)
                dbg["source"] = "weighted_b_stage"
                total_b_stage_s += float(sec)
            else:
                sec, dbg = self.estimate_op_s(sig=sig, sh=sh)

            total_s += float(sec)
            if self.debug:
                dbg["idx"] = int(j)
                dbg["phase"] = str(sig.phase)
                dbg["step"] = int(sig.step)
                dbg["shard"] = int(sig.shard)
                per_op.append(dbg)
                if dbg.get("source") == "weighted_b_stage":
                    self._dbg.log(
                        f"[npu-debug] op#{j:02d} op={dbg.get('op'):<8} B=max({dbg.get('b1_s',0)*1e3:.3f},{dbg.get('b2_s',0)*1e3:.3f})ms -> {dbg.get('b_stage_s',0)*1e3:.3f}ms"
                    )
                else:
                    self._dbg.log(
                        f"[npu-debug] op#{j:02d} op={dbg.get('op'):<8} key={dbg.get('op_key'):<10} "
                        f"mem={dbg.get('mem_s',0)*1e3:.3f}ms lat={dbg.get('lat_s',0)*1e3:.3f}ms"
                    )

        if self.debug:
            self._dbg.log(f"[npu-debug] segment total latency: {total_s*1e3:.3f} ms (weighted_b_total={total_b_stage_s*1e3:.3f} ms)")

        seg_dbg = None
        if self.debug:
            seg_dbg = {
                "segment_key": key_s,
                "device_type": str(seg.device_type),
                "phase": str(seg.phase),
                "step": int(seg.step),
                "count_hint": int(meta.get("count_hint", 0) or 0),
                "ops_repr": str(meta.get("ops_repr", "")),
                "total_s": float(total_s),
                "b_stage_s": float(total_b_stage_s),
                "per_op": per_op,
            }
        return float(total_s), float(total_b_stage_s), seg_dbg


def run_npu(
    tasks_json: str,
    out_json: str,
    *,
    npu_dtype: Optional[str] = None,
    npu_mem_bw_gbs: float = 0.0,
    op_overhead_us: float = 0.0,
    use_mem_bound: bool = True,
    batch: Optional[int] = None,
    debug: bool = False,
    debug_txt: Optional[str] = None,
) -> Path:
    """Compute LUT-based NPU latency for each exported segment."""
    data = _load_json(tasks_json)
    cfg_dict = data.get("config", {}) or {}
    cfg = WorkloadConfig.from_dict(cfg_dict)
    seg_scope = data.get("segment_scope", cfg.segment_scope)
    tasks = data.get("tasks", [])

    try:
        weight_load_s = float(data.get("weight_load_s", 0.0) or 0.0)
    except Exception:
        weight_load_s = 0.0

    weight_load_by_schedule = data.get("weight_load_by_schedule", {}) or {}
    weight_load_phase_by_schedule = data.get("weight_load_phase_by_schedule", {}) or {}

    if batch is not None:
        cfg.batch = int(batch)

    if debug:
        setattr(cfg, "debug", True)

    if npu_dtype is None:
        npu_dtype = str(getattr(cfg, "npu_dtype", "fp16"))

    outp = Path(out_json).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    dbg_logger = _DebugLogger(enabled=bool(debug), out_path=debug_txt)

    backend = NPUBackendViaCostModel(
        cfg,
        debug_logger=dbg_logger,
        npu_dtype=str(npu_dtype),
        npu_mem_bw_gbs=float(npu_mem_bw_gbs),
        op_overhead_us=float(op_overhead_us),
    )

    results: Dict[str, float] = {}
    b_stage_results: Dict[str, float] = {}
    debug_segments: Optional[Dict[str, Any]] = {} if bool(debug) else None

    for t in tasks:
        key = str(t.get("key"))
        sig_d = t.get("sig", {}) or {}
        ops_l = t.get("ops", []) or []
        seg = SegmentSig(
            device_type=str(sig_d.get("device_type", "npu")),
            phase=str(sig_d.get("phase", "decode")),
            step=int(sig_d.get("step", 0)),
            ops=tuple((str(x.get("op")), int(x.get("shard", -1))) for x in ops_l),
        )
        seg_s, seg_b_s, seg_dbg = backend.benchmark_segment(seg, seg_key=key, task_meta=t)
        results[key] = float(seg_s)
        b_stage_results[key] = float(seg_b_s)
        if debug_segments is not None and seg_dbg is not None:
            debug_segments[key] = seg_dbg

    out = {
        "version": 3,
        "task_type": "segment",
        "backend": "npu_ascend_310b_lut",
        "segment_scope": seg_scope,
        "config": cfg.to_dict(),
        "weight_load_s": float(weight_load_s),
        "weight_load_by_schedule": weight_load_by_schedule,
        "weight_load_phase_by_schedule": weight_load_phase_by_schedule,
        "env": {
            "TRIFORM_MMAD_LUT": os.environ.get("TRIFORM_MMAD_LUT"),
            "TRIFORM_SOFTMAX_LUT": os.environ.get("TRIFORM_SOFTMAX_LUT"),
            "TRIFORM_GELU_LUT": os.environ.get("TRIFORM_GELU_LUT"),
            "TRIFORM_NORM_LUT": os.environ.get("TRIFORM_NORM_LUT"),
        },
        "npu_dtype": str(npu_dtype),
        "npu_mem_bw_gbs": float(npu_mem_bw_gbs),
        "op_overhead_us": float(op_overhead_us),
        "use_mem_bound": bool(use_mem_bound),
        "results": results,
        "b_stage_results": b_stage_results,
    }

    if debug_segments is not None:
        out["debug_segments"] = debug_segments

    _save_json(outp, out)
    dbg_logger.close()
    return outp


def run_pim(
    tasks_json: str,
    out_json: str,
    *,
    cent_sim_root: Optional[str] = None,
    ramulator_config: Optional[str] = None,
    pim_hw_json: Optional[str] = None,
    pim_num_devices: Optional[int] = None,
    ramulator_bin: Optional[str] = None,
    ramulator_timeout_s: Optional[int] = None,
    keep_traces: bool = False,
    trace_dir: Optional[str] = None,
    batch: Optional[int] = None,
    debug: bool = False,
    debug_txt: Optional[str] = None,
    no_cache: bool = False,
) -> Path:

    data = _load_json(tasks_json)
    cfg_dict = data.get("config", {}) or {}
    cfg = WorkloadConfig.from_dict(cfg_dict)
    if batch is not None:
        try:
            cfg.batch = int(batch)  # type: ignore[attr-defined]
        except Exception as e:
            raise ValueError(f"batch must be an integer, got {batch!r}") from e
        if int(getattr(cfg, 'batch', 0) or 0) <= 0:
            raise ValueError(f"batch must be > 0, got {getattr(cfg, 'batch', None)!r}")
    seg_scope = data.get("segment_scope", cfg.segment_scope)
    tasks = data.get("tasks", [])

    try:
        weight_load_s = float(data.get("weight_load_s", 0.0) or 0.0)
    except Exception:
        weight_load_s = 0.0

    weight_load_by_schedule = data.get("weight_load_by_schedule", {}) or {}
    weight_load_phase_by_schedule = data.get("weight_load_phase_by_schedule", {}) or {}

    if ramulator_config is not None:
        cfg.pim_ramulator_config = ramulator_config
    if ramulator_bin is not None:
        cfg.pim_ramulator_bin = ramulator_bin
    if ramulator_timeout_s is not None:
        cfg.pim_ramulator_timeout_s = int(ramulator_timeout_s)

    if keep_traces:
        cfg.pim_keep_traces = True
    if trace_dir is not None:
        cfg.pim_trace_dir = trace_dir
    # ------------------------------
    # PIM HW spec: read from a single JSON (PIM_AiM.json)
    # ------------------------------
    if pim_hw_json is not None:
        hw = load_pim_hw_json(pim_hw_json)
        # Required topology for aim_sim
        cfg.pim_dram_column = int(hw["pim_dram_column"])
        cfg.pim_dram_row = int(hw["pim_dram_row"])
        cfg.pim_burst_length = int(hw["pim_burst_length"])
        cfg.pim_num_banks = int(hw["pim_num_banks"])
        cfg.pim_num_channels = int(hw["pim_num_channels"])

        # Optional overrides (if present in JSON)
        if hw.get("pim_threads") is not None:
            cfg.pim_threads = int(hw["pim_threads"])  # type: ignore[arg-type]
        if hw.get("pim_reuse_size") is not None:
            cfg.pim_reuse_size = int(hw["pim_reuse_size"])  # type: ignore[arg-type]
        if hw.get("pim_freq_ghz") is not None:
            cfg.pim_freq_ghz = float(hw["pim_freq_ghz"])  # type: ignore[arg-type]
        # Keep the original JSON path so the CostModel trace backend can consume it.
        cfg.pim_hw_json_path = str(resolve_existing_path(pim_hw_json))
    if pim_num_devices is not None:
        try:
            iv = int(pim_num_devices)
        except Exception as e:
            raise ValueError(f"pim_num_devices must be an integer, got {pim_num_devices!r}") from e
        if iv <= 0:
            raise ValueError(f"pim_num_devices must be > 0, got {iv}")
        cfg.pim_num_devices = iv


    if not cfg.pim_ramulator_bin:
        raise ValueError(
            "pim_ramulator_bin must be provided to run ramulator.\n"
            "Provide it either when exporting tasks (export/all) or at run time: "
            "run-pim --pim-ramulator-bin <path>."
        )
    if not cfg.pim_ramulator_config:
        raise ValueError(
            "pim_ramulator_config must be provided to run ramulator.\n"
            "Provide it either when exporting tasks (export/all) or at run time: "
            "run-pim --pim-ramulator-config <path> --pim-hw-json <path>."
        )

    # Debug output + (optional) tee to file
    cfg.debug = bool(debug or getattr(cfg, 'debug', False))
    dbg = _DebugLogger(enabled=bool(cfg.debug), out_path=debug_txt)
    backend = PIMBackendViaCostModel(cfg, tasks, debug_logger=dbg, use_cache=(not bool(no_cache)))

    results: Dict[str, float] = {}
    print(f"[run-pim] tasks={len(tasks)} segment_scope={seg_scope}")
    if weight_load_s:
        print(f"[run-pim] extra weight_load_s={weight_load_s:.6f}s (will be added in merge)")
    try:
        for i, t in enumerate(tasks):
            key = t["key"]
            ops = tuple((x["op"], int(x.get("shard",-1))) for x in t.get("ops", []))
            sig = SegmentSig(device_type="pim", phase=str(t["sig"]["phase"]), step=int(t["sig"]["step"]), ops=ops)
            sec = backend.benchmark_segment(sig, seg_key=str(key), task_meta=t)
            results[key] = float(sec)
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                print(f"  progress {i+1}/{len(tasks)}")
    finally:
        try:
            backend.close()
        except Exception:
            pass
        try:
            dbg.close()
        except Exception:
            pass

    outp = Path(out_json).expanduser().resolve()
    out = {
        "version": 3,
        "task_type": "segment",
        "backend": "pim",
        "segment_scope": seg_scope,
        "config": cfg.to_dict(),
        "weight_load_s": float(weight_load_s),
        "weight_load_by_schedule": weight_load_by_schedule,
        "weight_load_phase_by_schedule": weight_load_phase_by_schedule,
        "results": results,
    }
    if getattr(cfg, "debug", False) and getattr(backend, "debug_segments", None):
        out["debug_segments"] = backend.debug_segments
    _save_json(outp, out)
    print(f"[run-pim] wrote {outp}")
    return outp


def _comm_latency_from_df(df: pd.DataFrame, comm_model: str, pcie_lanes: int = 16) -> float:
    comm_model = comm_model.lower()
    comm_df = df[df["op"].astype(str).str.strip().str.lower().isin(_COMM_OPS)]
    if comm_model == "none":
        return 0.0
    if comm_model == "schedule":
        return float(comm_df["duration"].sum())
    if comm_model == "cxl":
        if cxl_latency is None:
            raise RuntimeError("cxl_latency.py not importable; cannot use --comm-model cxl")
        # crude: each comm node duration is estimated based on bytes inferred from node_id/op
        total = 0.0
        for _, r in comm_df.iterrows():
            op = str(r["op"])
            # Heuristic: KV write ~ shard_dim * dtype_bytes; Identity ~ dim * dtype_bytes
            # This is rough; adjust if you have a better mapping.
            layer = int(r.get("layer", -1))
            shard = int(r.get("shard", -1))
            # assume fp16 = 2 bytes
            dtype_bytes = 2
            # use cfg? we don't have cfg here; fallback to reading from trace? use 4096/4
            dim = 4096
            shards = 4
            shard_dim = dim // shards
            if op in ("K_write","V_write"):
                bytes_ = shard_dim * dtype_bytes
            else:
                bytes_ = dim * dtype_bytes
            total += float(cxl_latency.cxl_memcpy_latency(bytes_, lanes=pcie_lanes))
        return total
    raise ValueError(f"Unknown comm_model: {comm_model}")


def _try_lookup_segment_latency(seg: SegmentSig, res: Dict[str, float]) -> Optional[float]:
    k = seg.to_key()
    v = res.get(k, None)
    if v is not None:
        return float(v)
    return None


def _build_row_durations_layer_scope(
    df: pd.DataFrame,
    *,
    npu_res: Dict[str, float],
    pim_res: Dict[str, float],
    comm_model: str,
    pcie_lanes: int,
    decode_stride: int,
    dim: int,
    shards: int,
    allow_missing: bool,
) -> Tuple[np.ndarray, int]:
    n = len(df)
    dur_s = np.zeros(n, dtype=np.float64)

    op_l = df["op"].astype(str).str.strip().str.lower()
    is_comm = op_l.isin(_COMM_OPS).to_numpy()

    # 1) COMM rows
    if comm_model.lower() == "schedule":
        dur_s[is_comm] = df.loc[is_comm, "duration"].astype(float).to_numpy()

    # 2) Compute rows (layer scope)
    comp_cols = ["phase", "step", "layer", "device", "device_type", "op", "shard", "start", "_row", "duration"]
    if "sig_step" in df.columns:
        comp_cols.append("sig_step")
    compute_df = df.loc[~is_comm, comp_cols].copy()

    # Sort to match extract_segments() ordering: per group sorted by (start, _row)
    compute_df = compute_df.sort_values(["phase", "step", "layer", "device", "start", "_row"], kind="mergesort")

    missing = 0
    group_cols = ["phase", "step", "layer", "device"]
    for (phase, step, layer, device), g in compute_df.groupby(group_cols, sort=False):
        dev_type = str(g["device_type"].iloc[0])
        dev_type_n = dev_type.strip().lower()

        ops: List[Tuple[str, int]] = []
        for _, rr in g.iterrows():
            op = str(rr["op"]).strip()
            shard = int(rr["shard"]) if pd.notna(rr["shard"]) else -1
            ops.append((op, shard))

        idxs = g.index.to_numpy()
        if idxs.size == 0:
            continue

        w = pd.to_numeric(g["duration"], errors="coerce").fillna(0.0).astype(float).to_numpy()
        if dev_type_n not in ("npu", "pim"):
            dur_s[idxs] = w
            continue

        try:
            sig_step_i = int(pd.to_numeric(g.get("sig_step"), errors="coerce").fillna(step).iloc[0]) if "sig_step" in g.columns else int(step)
        except Exception:
            sig_step_i = int(step)
        seg = SegmentSig(device_type=dev_type, phase=str(phase), step=int(sig_step_i), ops=tuple(ops))
        if dev_type_n == "npu":
            total_lat = _try_lookup_segment_latency(seg, npu_res)
        elif dev_type_n == "pim":
            total_lat = _try_lookup_segment_latency(seg, pim_res)
        else:
            total_lat = None

        extra_local = 0.0

        if total_lat is None:
            if allow_missing:
                total_lat = 0.0
            else:
                missing += 1
                total_lat = 0.0

        total_lat = float(total_lat) + float(extra_local)

        wsum = float(np.nansum(w.astype(np.float64)))
        if wsum > 0:
            dur_s[idxs] = float(total_lat) * (w / wsum)
        else:
            dur_s[idxs] = float(total_lat) / float(len(idxs))

    return dur_s, int(missing)


def _simulate_block_overlap(df_block: pd.DataFrame, dur_s: np.ndarray) -> float:
    """
    Simulate one "block" (either prefill or one decode sampled step) with:
      - layer-by-layer sequential dependency (Transformer layers are sequential)
      - within each layer: sharded ops (S0..Sk) can overlap; shardless ops (S=-1) sync with all shards
      - per-resource serialization: each device is a resource; COMM is a separate resource

    df_block must be in the original trace order (strictly follow ops_trace.csv row order).
    """
    if df_block.empty:
        return 0.0

    # Precompute comm mask for block
    op_l = df_block["op"].astype(str).str.strip().str.lower()
    is_comm = op_l.isin(_COMM_OPS).to_numpy()

    idxs = df_block.index.to_numpy()
    layers = df_block["layer"].astype(int).to_numpy()
    shards = df_block["shard"].astype(int).to_numpy()
    devices = df_block["device"].astype(str).to_numpy()

    # Resource timelines
    res_t: Dict[str, float] = {}

    def get_res_time(res: str) -> float:
        return float(res_t.get(res, 0.0))

    def set_res_time(res: str, t: float) -> None:
        res_t[res] = float(t)

    prev_layer_done = 0.0
    cur_layer: Optional[int] = None
    layer_max = 0.0
    global_chain = 0.0
    shard_chain: Dict[int, float] = {}

    for i in range(len(idxs)):
        idx = int(idxs[i])
        layer = int(layers[i])
        shard = int(shards[i])

        # layer boundary
        if cur_layer is None:
            cur_layer = layer
            global_chain = prev_layer_done
            shard_chain.clear()
            layer_max = prev_layer_done
        elif layer != cur_layer:
            prev_layer_done = layer_max
            cur_layer = layer
            global_chain = prev_layer_done
            shard_chain.clear()
            layer_max = prev_layer_done

        # resource
        res = "COMM" if bool(is_comm[i]) else str(devices[i])

        d = float(dur_s[idx])

        # dependency inside layer
        if shard < 0:
            shard_max = max(shard_chain.values()) if shard_chain else prev_layer_done
            deps = max(global_chain, shard_max)
        else:
            deps = max(global_chain, shard_chain.get(shard, prev_layer_done))

        st = max(get_res_time(res), deps)
        et = st + d
        set_res_time(res, et)

        if shard < 0:
            global_chain = et
        else:
            shard_chain[shard] = et

        if et > layer_max:
            layer_max = et

    # finish last layer
    prev_layer_done = max(prev_layer_done, layer_max)
    return float(prev_layer_done)

def compute_trace_times_from_ops_csv(df: pd.DataFrame, *, decode_stride: int = 1) -> Dict[str, float]:
    """Compute prefill/decode e2e time from the original ops_trace.csv timeline."""

    if df is None or df.empty:
        return {
            "trace_prefill_s": 0.0,
            "trace_decode_s": 0.0,
            "trace_decode_scaled_s": 0.0,
            "trace_total_raw_s": 0.0,
            "trace_total_scaled_s": 0.0,
        }

    # Best-effort numeric conversion
    start = pd.to_numeric(df.get("start"), errors="coerce")
    end = pd.to_numeric(df.get("end"), errors="coerce")

    def _span(mask: np.ndarray) -> float:
        if mask is None or mask.size == 0 or (not mask.any()):
            return 0.0
        st = float(np.nanmin(start[mask].to_numpy(dtype=np.float64)))
        en = float(np.nanmax(end[mask].to_numpy(dtype=np.float64)))
        if not np.isfinite(st) or not np.isfinite(en):
            return 0.0
        return float(max(0.0, en - st))

    ph = df.get("phase")
    if ph is None:
        pre = 0.0
        dec = 0.0
    else:
        ph_s = ph.astype(str)
        pre = _span((ph_s == "prefill").to_numpy())
        dec = _span((ph_s == "decode").to_numpy())

    total_raw = _span(np.ones(len(df), dtype=bool))

    ds = int(decode_stride) if decode_stride is not None else 1
    if ds <= 0:
        ds = 1
    total = float(pre + dec)

    return {
        "trace_prefill_s": float(pre),
        "trace_decode_s": float(dec),
        "trace_decode_scaled_s": float(dec),
        "trace_total_raw_s": float(total_raw),
        "trace_total_scaled_s": float(total),
        "trace_total_s": float(total),
    }

# legacy weight-stage reconstruction helpers removed; merge() now uses direct weight_load from comms.csv.


def merge(
    schedule_paths: List[str],
    npu_results_json: Optional[str],
    pim_results_json: Optional[str],
    *,
    comms_paths: Optional[List[str]] = None,
    overlap: float = 0.0,
    comm_model: str = "schedule",
    pcie_lanes: int = 16,
    decode_stride: int = 1,
    kv_load_bw_gbs_override: Optional[float] = None,
    kv_dtype_bytes_override: Optional[float] = None,
    kv_load_overhead_us_override: Optional[float] = None,
    n_kv_heads_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    out_csv: str,
    out_steps_csv: Optional[str] = None,
    allow_missing: bool = False,
    slack_model: str = "span",
    slack_clamp_negative: bool = True,
    debug: bool = False,
    debug_txt: Optional[str] = None,
) -> None:
    """Merge schedule trace(s) with NPU/PIM segment latencies.

    Simplified verify logic:
      1) Segment NPU/PIM compute is recomputed from CostModel after segment split.
      2) For NPU weighted linear ops, the segment contribution is `max(b1, b2)`.
      3) `weight_load` time is taken directly from comms.csv and added per phase.
      4) B-stage and `weight_load` can overlap via:
            added_weight_load = total_weight_load - overlap * min(npu_b_stage, npu_weight_load)
         PIM / unknown-class weight_load stays serial.
      5) No per-load matching, subtraction, 补足, or coarse placement path.
    """

    slack_model = str(slack_model or "none").strip().lower()
    if slack_model not in ("none", "span"):
        raise ValueError(f"Unknown slack_model={slack_model!r} (expected 'none' or 'span')")

    if decode_stride < 1:
        raise ValueError(f"decode_stride must be >=1 (got {decode_stride})")
    if overlap < 0.0 or overlap > 1.0:
        raise ValueError(f"overlap must be within [0,1] (got {overlap})")

    if debug and (not debug_txt):
        try:
            oc = Path(out_csv).expanduser()
            if str(oc).lower().endswith(".csv"):
                debug_txt = str(oc)[:-4] + ".debug.txt"
            else:
                debug_txt = str(oc) + ".debug.txt"
        except Exception:
            debug_txt = str(out_csv) + ".debug.txt"

    def _load_optional_json(path_str: Optional[str]) -> Dict[str, Any]:
        if not path_str or str(path_str).strip() in ("", "-"):
            return {}
        return _load_json(path_str)

    npu_data = _load_optional_json(npu_results_json)
    pim_data = _load_optional_json(pim_results_json)

    def _coerce_results(d: Dict[str, Any], field: str = "results") -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not isinstance(d, dict):
            return out
        for k, v in (d.get(field) or {}).items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

    npu_res = _coerce_results(npu_data, "results")
    npu_b_stage_res = _coerce_results(npu_data, "b_stage_results")
    pim_res = _coerce_results(pim_data, "results")

    cfg_meta: Dict[str, Any] = {}
    if isinstance(npu_data.get("config"), dict):
        cfg_meta = dict(npu_data["config"])
    elif isinstance(pim_data.get("config"), dict):
        cfg_meta = dict(pim_data["config"])

    dim = int(cfg_meta.get("dim", 4096))
    shards = int(cfg_meta.get("shards", 1))

    if kv_load_bw_gbs_override is not None:
        cfg_meta["kv_load_bw_gbs"] = float(kv_load_bw_gbs_override)
    if kv_dtype_bytes_override is not None:
        cfg_meta["kv_dtype_bytes"] = float(kv_dtype_bytes_override)
    if kv_load_overhead_us_override is not None:
        cfg_meta["kv_load_overhead_us"] = float(kv_load_overhead_us_override)
    if n_kv_heads_override is not None:
        cfg_meta["n_kv_heads"] = int(n_kv_heads_override)
    if batch_override is not None:
        cfg_meta["batch"] = int(batch_override)

    comms_paths = [p for p in (comms_paths or []) if p]

    def _pick_comms_for_schedule(i: int) -> List[str]:
        if not comms_paths:
            return []
        if len(comms_paths) == 1:
            return comms_paths
        if len(comms_paths) == len(schedule_paths):
            return [comms_paths[i]]
        return comms_paths

    def _lookup_weight_load_phase_from_results(schedule_abs: str) -> Dict[str, Dict[str, float]]:
        zero = {"npu_s": 0.0, "pim_s": 0.0, "unknown_s": 0.0, "total_s": 0.0}
        out: Dict[str, Dict[str, float]] = {"prefill": dict(zero), "decode": dict(zero)}
        found = False
        for d in (npu_data, pim_data):
            try:
                m = d.get("weight_load_phase_by_schedule") or {}
            except Exception:
                m = {}
            rec = m.get(schedule_abs)
            if isinstance(rec, dict):
                found = True
                for ph, v in rec.items():
                    if str(ph) == "all" or not isinstance(v, dict):
                        continue
                    out[str(ph)] = {
                        "npu_s": float(v.get("npu_s", 0.0) or 0.0),
                        "pim_s": float(v.get("pim_s", 0.0) or 0.0),
                        "unknown_s": float(v.get("unknown_s", 0.0) or 0.0),
                        "total_s": float(v.get("total_s", 0.0) or 0.0),
                    }
        if found:
            return out

        total = 0.0
        for d in (npu_data, pim_data):
            try:
                m = d.get("weight_load_by_schedule") or {}
                rec = m.get(schedule_abs)
                if isinstance(rec, dict):
                    total = float(rec.get("total_s", 0.0) or 0.0)
                    break
            except Exception:
                pass
        if total > 0.0:
            out["prefill"]["total_s"] = float(total)
            out["prefill"]["npu_s"] = float(total)
        return out

    def _lookup_latency(dev_type: str, phase: str, step_sample: int, ops: Tuple[Tuple[str, int], ...]) -> Tuple[Optional[float], Optional[str]]:
        dev = dev_type.strip().lower()
        if dev == "npu":
            table = npu_res
        elif dev == "pim":
            table = pim_res
        else:
            return None, None

        k = SegmentSig(device_type=dev, phase=phase, step=int(step_sample), ops=ops).to_key()
        if k in table:
            return float(table[k]), str(k)
        return None, None

    def _devicewise_max_sum(
        g: pd.DataFrame,
        value_col: str,
        *,
        group_cols: List[str],
        device_col: str = "device",
    ) -> Dict[Tuple[str, int, int], float]:
        if g.empty:
            return {}
        by_dev = g.groupby(group_cols + [device_col], sort=False)[value_col].sum().reset_index()
        by_key = by_dev.groupby(group_cols, sort=False)[value_col].max()
        return {(ph, int(st), int(ly)): float(v) for (ph, st, ly), v in by_key.items()}

    rows_out: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []

    dbg_f = None
    if debug and debug_txt:
        dbg_f = open(debug_txt, "w", encoding="utf-8")

    try:
        for i, sp in enumerate(schedule_paths):
            p = resolve_existing_path(sp)
            df = load_schedule_csv(str(p))
            df = annotate_decode_sampling(df, decode_stride=int(decode_stride))

            decode_scale_by_step, decode_scale_mode, decode_len_hint = compute_decode_step_scale_map(
                df, str(p), int(decode_stride)
            )
            actual_token_index_by_step = build_actual_step_token_index_map(df, decode_stride=int(decode_stride))

            comms_for_sched = _pick_comms_for_schedule(i)
            if comms_for_sched:
                weight_load_phase = extract_weight_load_seconds_by_phase(comms_for_sched)
            else:
                weight_load_phase = _lookup_weight_load_phase_from_results(str(p))

            def _phase_wl(phase_name: str) -> Dict[str, float]:
                rec = weight_load_phase.get(str(phase_name), {}) if isinstance(weight_load_phase, dict) else {}
                return {
                    "npu_s": float((rec or {}).get("npu_s", 0.0) or 0.0),
                    "pim_s": float((rec or {}).get("pim_s", 0.0) or 0.0),
                    "unknown_s": float((rec or {}).get("unknown_s", 0.0) or 0.0),
                    "total_s": float((rec or {}).get("total_s", 0.0) or 0.0),
                }

            wl_pref = _phase_wl("prefill")
            wl_dec = _phase_wl("decode")

            op_l = df["op"].astype(str).str.strip().str.lower()

            slack_by_group: Dict[Tuple[str, int, int, str, str], float] = {}
            if slack_model == "span":
                try:
                    df_sl = df[df["device_type"].astype(str).str.lower().isin(["npu", "pim"])].copy()
                    if not df_sl.empty:
                        op_l2 = df_sl["op"].astype(str).str.strip().str.lower()
                        df_sl = df_sl[~op_l2.isin(_COMM_OPS)].copy()
                    if not df_sl.empty:
                        df_sl["_st"] = pd.to_numeric(df_sl.get("start"), errors="coerce")
                        df_sl["_en"] = pd.to_numeric(df_sl.get("end"), errors="coerce")
                        df_sl["_du"] = pd.to_numeric(df_sl.get("duration"), errors="coerce").fillna(0.0).astype(float)
                        for (ph0, st0, ly0, dev0, dt0), gg in df_sl.groupby(
                            ["phase", "step", "layer", "device", "device_type"], sort=False
                        ):
                            try:
                                st_i = int(st0)
                            except Exception:
                                st_i = int(pd.to_numeric(pd.Series([st0]), errors="coerce").fillna(-1).iloc[0])
                            try:
                                ly_i = int(ly0)
                            except Exception:
                                ly_i = int(pd.to_numeric(pd.Series([ly0]), errors="coerce").fillna(-1).iloc[0])

                            st_min = float(np.nanmin(gg["_st"].to_numpy(dtype=np.float64)))
                            en_max = float(np.nanmax(gg["_en"].to_numpy(dtype=np.float64)))
                            if (not np.isfinite(st_min)) or (not np.isfinite(en_max)):
                                continue
                            span = float(max(0.0, en_max - st_min))
                            sum_dur = float(np.nansum(gg["_du"].to_numpy(dtype=np.float64)))
                            slack = float(span - sum_dur)
                            if slack_clamp_negative and slack < 0.0:
                                slack = 0.0
                            dt = str(dt0).strip().lower()
                            ph = str(ph0)
                            dev = str(dev0)
                            slack_by_group[(ph, st_i, ly_i, dev, dt)] = float(slack)
                except Exception:
                    slack_by_group = {}

            npu_layer_busy: Dict[Tuple[str, int, int], float] = {}
            pim_layer_busy: Dict[Tuple[str, int, int], float] = {}
            npu_layer_slack: Dict[Tuple[str, int, int], float] = {}
            pim_layer_slack: Dict[Tuple[str, int, int], float] = {}
            npu_layer_eff: Dict[Tuple[str, int, int], float] = {}
            pim_layer_eff: Dict[Tuple[str, int, int], float] = {}
            npu_b_layer_sum: Dict[Tuple[str, int, int], float] = {}
            missing_segments = 0

            comp = df[df["device_type"].astype(str).str.lower().isin(["npu", "pim"])].copy()
            for (phase, step, layer, device), g in comp.groupby(["phase", "step", "layer", "device"], sort=False):
                dev_type = str(g["device_type"].iloc[0]).strip().lower()
                g = g.sort_values(["start", "_row"], kind="mergesort")
                try:
                    sig_step = int(pd.to_numeric(g.get("sig_step"), errors="coerce").fillna(step).iloc[0]) if "sig_step" in g.columns else int(step)
                except Exception:
                    sig_step = int(step)

                ops: List[Tuple[str, int]] = []
                for _, r in g.iterrows():
                    op = str(r["op"]).strip()
                    if op.lower() in _COMM_OPS:
                        continue
                    ops.append((op, int(r["shard"]) if pd.notna(r["shard"]) else -1))
                if not ops:
                    continue

                lat, seg_key = _lookup_latency(dev_type, str(phase), int(sig_step), tuple(ops))
                if lat is None:
                    missing_segments += 1
                    if dbg_f:
                        dbg_f.write(
                            f"[missing] schedule={p.name} phase={phase} step={step} sig_step={sig_step} layer={layer} device={device} device_type={dev_type} n_ops={len(ops)}\n"
                        )
                    lat = 0.0

                lat = float(lat)

                slack = float(
                    slack_by_group.get((str(phase), int(step), int(layer), str(device), str(dev_type)), 0.0)
                ) if slack_model != "none" else 0.0
                eff = float(lat) + float(slack)

                key = (str(phase), int(step), int(layer))
                if dev_type == "npu":
                    if eff > float(npu_layer_eff.get(key, 0.0)):
                        npu_layer_eff[key] = float(eff)
                        npu_layer_busy[key] = float(lat)
                        npu_layer_slack[key] = float(slack)
                    if seg_key is not None:
                        npu_b_layer_sum[key] = float(npu_b_layer_sum.get(key, 0.0)) + float(npu_b_stage_res.get(seg_key, 0.0) or 0.0)
                elif dev_type == "pim":
                    if eff > float(pim_layer_eff.get(key, 0.0)):
                        pim_layer_eff[key] = float(eff)
                        pim_layer_busy[key] = float(lat)
                        pim_layer_slack[key] = float(slack)

            comm_ops_df = df[op_l.isin(_COMM_OPS)].copy()
            if comm_model.lower() == "none" or comm_ops_df.empty:
                comm_ops_df["lat_s"] = 0.0
            elif comm_model.lower() == "schedule":
                comm_ops_df["lat_s"] = pd.to_numeric(comm_ops_df["duration"], errors="coerce").fillna(0.0).astype(float)
            else:
                raise ValueError(f"unknown comm_model: {comm_model}")
            comm_ops_layer = _devicewise_max_sum(comm_ops_df, "lat_s", group_cols=["phase", "step", "layer"])

            other_df = df[(~df["device_type"].astype(str).str.lower().isin(["npu", "pim"])) & (~op_l.isin(_COMM_OPS))].copy()
            other_layer = _devicewise_max_sum(other_df, "duration", group_cols=["phase", "step", "layer"])

            prefill_time_s = 0.0
            decode_time_s = 0.0

            npu_busy_s = 0.0
            pim_busy_s = 0.0
            npu_b_prefill_s = 0.0
            npu_b_decode_s = 0.0
            npu_slack_s = 0.0
            pim_slack_s = 0.0
            npu_slack_prefill_s = 0.0
            pim_slack_prefill_s = 0.0
            npu_slack_decode_s = 0.0
            pim_slack_decode_s = 0.0
            compute_slack_used_s = 0.0
            compute_slack_used_prefill_s = 0.0
            compute_slack_used_decode_s = 0.0
            sched_comm_s = 0.0
            sched_other_s = 0.0

            pre_layers = sorted(df[df["phase"] == "prefill"]["layer"].unique())
            for ly in pre_layers:
                k = ("prefill", -1, int(ly))
                npu_busy = float(npu_layer_busy.get(k, 0.0))
                pim_busy = float(pim_layer_busy.get(k, 0.0))
                npu_sl = float(npu_layer_slack.get(k, 0.0)) if slack_model != "none" else 0.0
                pim_sl = float(pim_layer_slack.get(k, 0.0)) if slack_model != "none" else 0.0
                npu_eff = float(npu_layer_eff.get(k, npu_busy + npu_sl))
                pim_eff = float(pim_layer_eff.get(k, pim_busy + pim_sl))
                compute_max = max(npu_eff, pim_eff)
                comm_s = float(comm_ops_layer.get(k, 0.0))
                oth_s = float(other_layer.get(k, 0.0))
                layer_s = max(compute_max, comm_s, oth_s)
                prefill_time_s += layer_s

                npu_busy_s += npu_busy
                pim_busy_s += pim_busy
                npu_b_prefill_s += float(npu_b_layer_sum.get(k, 0.0))
                npu_slack_s += npu_sl
                pim_slack_s += pim_sl
                npu_slack_prefill_s += npu_sl
                pim_slack_prefill_s += pim_sl
                if npu_eff >= pim_eff:
                    compute_slack_used_s += npu_sl
                    compute_slack_used_prefill_s += npu_sl
                else:
                    compute_slack_used_s += pim_sl
                    compute_slack_used_prefill_s += pim_sl
                sched_comm_s += comm_s
                sched_other_s += oth_s

            dec = df[df["phase"] == "decode"]
            dec_steps = sorted(dec["step"].unique())
            dec_layers = sorted(dec["layer"].unique())

            for st in dec_steps:
                step_s_per_token = 0.0
                npu_step = 0.0
                pim_step = 0.0
                npu_step_b = 0.0
                npu_step_slack = 0.0
                pim_step_slack = 0.0
                step_slack_used = 0.0
                comm_step = 0.0
                oth_step = 0.0

                for ly in dec_layers:
                    k = ("decode", int(st), int(ly))
                    npu_busy = float(npu_layer_busy.get(k, 0.0))
                    pim_busy = float(pim_layer_busy.get(k, 0.0))
                    npu_sl = float(npu_layer_slack.get(k, 0.0)) if slack_model != "none" else 0.0
                    pim_sl = float(pim_layer_slack.get(k, 0.0)) if slack_model != "none" else 0.0
                    npu_eff = float(npu_layer_eff.get(k, npu_busy + npu_sl))
                    pim_eff = float(pim_layer_eff.get(k, pim_busy + pim_sl))
                    compute_max = max(npu_eff, pim_eff)
                    comm_s = float(comm_ops_layer.get(k, 0.0))
                    oth_s = float(other_layer.get(k, 0.0))
                    layer_s = max(compute_max, comm_s, oth_s)

                    step_s_per_token += layer_s
                    npu_step += npu_busy
                    pim_step += pim_busy
                    npu_step_b += float(npu_b_layer_sum.get(k, 0.0))
                    npu_step_slack += npu_sl
                    pim_step_slack += pim_sl
                    if npu_eff >= pim_eff:
                        step_slack_used += npu_sl
                    else:
                        step_slack_used += pim_sl
                    comm_step += comm_s
                    oth_step += oth_s

                scale_tokens = int(decode_scale_by_step.get(int(st), int(decode_stride)))
                step_s = step_s_per_token * float(scale_tokens)
                decode_time_s += step_s

                npu_busy_s += npu_step * float(scale_tokens)
                pim_busy_s += pim_step * float(scale_tokens)
                npu_b_decode_s += npu_step_b * float(scale_tokens)
                npu_slack_s += npu_step_slack * float(scale_tokens)
                pim_slack_s += pim_step_slack * float(scale_tokens)
                npu_slack_decode_s += npu_step_slack * float(scale_tokens)
                pim_slack_decode_s += pim_step_slack * float(scale_tokens)
                compute_slack_used_s += step_slack_used * float(scale_tokens)
                compute_slack_used_decode_s += step_slack_used * float(scale_tokens)
                sched_comm_s += comm_step * float(scale_tokens)
                sched_other_s += oth_step * float(scale_tokens)

                step_rows.append(
                    {
                        "schedule": p.name,
                        "phase": "decode",
                        "sample_step": int(st),
                        "token_index": int(actual_token_index_by_step.get(int(st), decode_token_block_start_from_sample_step(int(st), int(decode_stride)))),
                        "tokens": int(scale_tokens),
                        "step_time_s": float(step_s),
                        "step_time_per_token_s": float(step_s_per_token),
                    }
                )

            prefill_overlap_saved_s = float(overlap) * min(float(wl_pref["npu_s"]), float(npu_b_prefill_s))
            decode_overlap_saved_s = float(overlap) * min(float(wl_dec["npu_s"]), float(npu_b_decode_s))

            prefill_time_s += float(wl_pref["total_s"]) - float(prefill_overlap_saved_s)
            decode_time_s += float(wl_dec["total_s"]) - float(decode_overlap_saved_s)

            total_time_s = prefill_time_s + decode_time_s

            trace_t = compute_trace_times_from_ops_csv(df, decode_stride=decode_stride)
            trace_prefill_s = float(trace_t.get("trace_prefill_s", 0.0))
            trace_decode_s = float(trace_t.get("trace_decode_scaled_s", trace_t.get("trace_decode_s", 0.0)))
            trace_total_s = float(trace_t.get("trace_total_scaled_s", trace_t.get("trace_total_s", 0.0)))

            row = {
                "schedule": p.name,
                "dim": dim,
                "shards": shards,
                "decode_stride": int(decode_stride),
                "overlap": float(overlap),
                "slack_model": str(slack_model),

                "prefill_time_s": float(prefill_time_s),
                "decode_time_s": float(decode_time_s),
                "total_time_s": float(total_time_s),

                "trace_prefill_s": float(trace_prefill_s),
                "trace_decode_s": float(trace_decode_s),
                "trace_total_s": float(trace_total_s),

                "delta_prefill_s": float(prefill_time_s - trace_prefill_s),
                "delta_decode_s": float(decode_time_s - trace_decode_s),
                "delta_total_s": float(total_time_s - trace_total_s),

                "npu_busy_s": float(npu_busy_s),
                "pim_busy_s": float(pim_busy_s),
                "npu_b_prefill_s": float(npu_b_prefill_s),
                "npu_b_decode_s": float(npu_b_decode_s),
                "npu_b_total_s": float(npu_b_prefill_s + npu_b_decode_s),

                "npu_slack_s": float(npu_slack_s),
                "pim_slack_s": float(pim_slack_s),
                "npu_slack_prefill_s": float(npu_slack_prefill_s),
                "pim_slack_prefill_s": float(pim_slack_prefill_s),
                "npu_slack_decode_s": float(npu_slack_decode_s),
                "pim_slack_decode_s": float(pim_slack_decode_s),

                "compute_slack_used_s": float(compute_slack_used_s),
                "compute_slack_used_prefill_s": float(compute_slack_used_prefill_s),
                "compute_slack_used_decode_s": float(compute_slack_used_decode_s),

                "schedule_comm_s": float(sched_comm_s),
                "schedule_other_s": float(sched_other_s),

                "weight_load_prefill_s": float(wl_pref["total_s"]),
                "weight_load_decode_s": float(wl_dec["total_s"]),
                "weight_load_total_s": float(wl_pref["total_s"] + wl_dec["total_s"]),

                "weight_load_prefill_npu_s": float(wl_pref["npu_s"]),
                "weight_load_prefill_pim_s": float(wl_pref["pim_s"]),
                "weight_load_prefill_unknown_s": float(wl_pref["unknown_s"]),
                "weight_load_decode_npu_s": float(wl_dec["npu_s"]),
                "weight_load_decode_pim_s": float(wl_dec["pim_s"]),
                "weight_load_decode_unknown_s": float(wl_dec["unknown_s"]),

                "weight_load_prefill_overlap_saved_s": float(prefill_overlap_saved_s),
                "weight_load_decode_overlap_saved_s": float(decode_overlap_saved_s),
                "weight_load_overlap_saved_total_s": float(prefill_overlap_saved_s + decode_overlap_saved_s),

                "missing_segments": int(missing_segments),
            }
            rows_out.append(row)

            if dbg_f:
                dbg_f.write(
                    f"[schedule] {p.name} prefill={prefill_time_s:.6f}s decode={decode_time_s:.6f}s total={total_time_s:.6f}s "
                    f"wl_prefill={float(wl_pref['total_s']):.6f}s wl_decode={float(wl_dec['total_s']):.6f}s "
                    f"ov_prefill={prefill_overlap_saved_s:.6f}s ov_decode={decode_overlap_saved_s:.6f}s "
                    f"npu_b_prefill={npu_b_prefill_s:.6f}s npu_b_decode={npu_b_decode_s:.6f}s missing_segments={missing_segments}\n"
                )
    finally:
        if dbg_f:
            dbg_f.close()

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)

    if out_steps_csv:
        os.makedirs(os.path.dirname(out_steps_csv) or ".", exist_ok=True)
        pd.DataFrame(step_rows).to_csv(out_steps_csv, index=False)


def collect_merge_csv(list_tsv: str, *, out_csv: str, skip_missing: bool = False) -> None:
    """Concatenate multiple per-run merge.csv files into one combined CSV."""
    lp = resolve_existing_path(list_tsv)
    try:
        lst = pd.read_csv(str(lp), sep="\t")
    except Exception as e:
        raise ValueError(f"Failed to read list TSV: {lp}") from e

    if lst is None or lst.empty:
        raise ValueError(f"merge list is empty: {lp}")

    # Find merge_csv column (recommended name)
    merge_col = None
    if "merge_csv" in lst.columns:
        merge_col = "merge_csv"
    else:
        # best effort: any column containing 'merge' and 'csv'
        for c in lst.columns:
            cl = str(c).lower()
            if "merge" in cl and "csv" in cl:
                merge_col = c
                break
    if merge_col is None:
        raise ValueError(f"{lp}: missing required column 'merge_csv' (columns={list(lst.columns)})")

    frames: List[pd.DataFrame] = []
    missing: List[str] = []

    for _, r in lst.iterrows():
        m = str(r.get(merge_col, "")).strip()
        if not m or m.lower() in ("nan", "none", "-"):
            continue
        mp = Path(m).expanduser()
        if not mp.is_absolute():
            mp = (lp.parent / mp).resolve()

        if not mp.exists():
            if skip_missing:
                missing.append(str(mp))
                continue
            raise FileNotFoundError(f"merge_csv not found: {mp}")

        df = pd.read_csv(str(mp))
        # Always keep the physical path for traceability
        if "merge_csv" not in df.columns:
            df["merge_csv"] = str(mp)

        # Attach metadata columns from list file
        for c in lst.columns:
            if c == merge_col:
                continue
            meta_name = c if c not in df.columns else f"meta_{c}"
            df[meta_name] = r.get(c, "")
        frames.append(df)

    if not frames:
        raise ValueError(f"No merge CSVs found to collect from: {lp}")

    out = pd.concat(frames, ignore_index=True)

    outp = Path(out_csv).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(str(outp), index=False)

    if missing:
        print(f"[collect-merge] skipped {len(missing)} missing merge csv(s) (skip_missing=1)")


# ==========================================================
# CLI
# ==========================================================

def add_common_model_args(p: argparse.ArgumentParser) -> None:

    p.add_argument("--cfg", type=str, default=None, help="model shape json, e.g. ./configs/llama_7b_shape.json")
    p.add_argument("--prefill-len", type=int, default=128, dest="prefill_len")
    p.add_argument("--decode-context-lens", type=str, nargs="?", const="auto", default=None,)

    p.add_argument("--shard-policy", type=str, default="fine", choices=["fine"],
                   help="shard placement policy for verification (only fine is supported)")

    p.add_argument(
        "--kv-load-bw-gbs",
        type=float,
        default=0.0,
        help="Estimated KV-cache load bandwidth in GB/s (decode). Set >0 to enable implicit kv_load before QK/SV.",
    )
    p.add_argument(
        "--kv-dtype-bytes",
        type=float,
        default=2.0,
        dest="kv_dtype_bytes",
        help="Bytes per KV element for kv_load estimate (per K or per V). Default=2 (fp16/bf16).",
    )
    p.add_argument(
        "--kv-load-overhead-us",
        type=float,
        default=0.0,
        dest="kv_load_overhead_us",
        help="Fixed overhead per KV-load event (per shard/op), in microseconds.",
    )
    p.add_argument(
        "--n-kv-heads",
        type=int,
        default=None,
        dest="n_kv_heads",
        help="Optional KV head count for GQA/MQA. If omitted, defaults to n_heads.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        dest="batch",
        help="Optional batch size for KV-load bytes estimate (default=1).",
    )

    # PIM config
    p.add_argument("--pim-dram-column", type=int, default=256)
    p.add_argument("--pim-dram-row", type=int, default=64)
    p.add_argument("--pim-burst-length", type=int, default=16)
    p.add_argument("--pim-num-banks", type=int, default=8)
    p.add_argument("--pim-num-channels", type=int, default=4)
    p.add_argument("--pim-threads", type=int, default=1)
    p.add_argument("--pim-reuse-size", type=int, default=32)

    p.add_argument("--pim-num-devices", type=int, default=1, help="PIM device/DIMM count")
    p.add_argument("--pim-ramulator-bin", type=str, default=None, help="path/name of AiM-enabled ramulator2 executable (required for run-pim)",)
    p.add_argument("--pim-ramulator-config", type=str, default=None, help="ramulator2 config file (required for run-pim)")
    p.add_argument("--pim-freq-ghz", type=float, default=1.0, help="PIM clock frequency in GHz for cycles->seconds")
    p.add_argument("--pim-ramulator-timeout-s", type=int, default=3000, help="ramulator2 timeout per trace (seconds)")
    p.add_argument("--pim-keep-traces", action="store_true", help="keep generated AiM traces (debug)")
    p.add_argument("--pim-trace-dir", type=str, default=None, help="trace output dir when keeping traces")
    p.add_argument("--pim-no-override-ramulator-config", action="store_true",
                   help="do not rewrite channels/banks/devices into ramulator config (NOT recommended)")
    # segmenting
    p.add_argument("--segment-scope", type=str, default="layer", choices=["layer"],
                   help="segment granularity")


def build_cfg_from_args(args: argparse.Namespace) -> WorkloadConfig:
    cfg = WorkloadConfig()

    # ------------------------------
    # Model shape from --cfg
    # ------------------------------
    cfg_path = getattr(args, "cfg", None)
    if cfg_path:
        shape = load_model_shape_json(cfg_path)
        cfg.dim = int(shape["dim"])
        cfg.ffn_dim = int(shape["ffn_dim"])
        cfg.n_heads = int(shape["n_heads"])
        if shape.get("n_kv_heads") is not None:
            cfg.n_kv_heads = int(shape["n_kv_heads"])
    # ------------------------------
    # Schedule lengths / dtype / device / PIM params
    # ------------------------------
    cfg.prefill_len = int(getattr(args, "prefill_len", cfg.prefill_len))
    cfg.decode_context_lens = None
    cfg.device = str(getattr(args, "device", cfg.device))
    cfg.npu_dtype = str(getattr(args, "npu_dtype", cfg.npu_dtype))

    cfg.kv_load_bw_gbs = float(getattr(args, "kv_load_bw_gbs", cfg.kv_load_bw_gbs))
    cfg.kv_dtype_bytes = float(getattr(args, "kv_dtype_bytes", cfg.kv_dtype_bytes))
    cfg.kv_load_overhead_us = float(getattr(args, "kv_load_overhead_us", cfg.kv_load_overhead_us))
    if hasattr(args, "n_kv_heads") and getattr(args, "n_kv_heads") is not None:
        cfg.n_kv_heads = int(getattr(args, "n_kv_heads"))
    if hasattr(args, "batch"):
        cfg.batch = int(getattr(args, "batch"))

    cfg.pim_dram_column = int(getattr(args, "pim_dram_column", cfg.pim_dram_column))
    cfg.pim_dram_row = int(getattr(args, "pim_dram_row", cfg.pim_dram_row))
    cfg.pim_burst_length = int(getattr(args, "pim_burst_length", cfg.pim_burst_length))
    cfg.pim_num_banks = int(getattr(args, "pim_num_banks", cfg.pim_num_banks))
    cfg.pim_num_channels = int(getattr(args, "pim_num_channels", cfg.pim_num_channels))
    cfg.pim_threads = int(getattr(args, "pim_threads", cfg.pim_threads))
    cfg.pim_reuse_size = int(getattr(args, "pim_reuse_size", cfg.pim_reuse_size))
    cfg.pim_num_devices = int(getattr(args, "pim_num_devices", cfg.pim_num_devices))

    cfg.pim_ramulator_bin = getattr(args, "pim_ramulator_bin", cfg.pim_ramulator_bin)
    cfg.pim_ramulator_config = getattr(args, "pim_ramulator_config", cfg.pim_ramulator_config)
    cfg.pim_freq_ghz = float(getattr(args, "pim_freq_ghz", cfg.pim_freq_ghz))
    cfg.pim_ramulator_timeout_s = int(getattr(args, "pim_ramulator_timeout_s", cfg.pim_ramulator_timeout_s))
    cfg.pim_keep_traces = bool(getattr(args, "pim_keep_traces", cfg.pim_keep_traces))
    cfg.pim_trace_dir = getattr(args, "pim_trace_dir", cfg.pim_trace_dir)

    cfg.segment_scope = str(getattr(args, "segment_scope", cfg.segment_scope))

    cfg.shard_policy = str(getattr(args, "shard_policy", cfg.shard_policy))

    # decode stride (used to generate decode-context-lens / shape inference)
    if hasattr(args, "decode_stride") and getattr(args, "decode_stride") is not None:
        cfg.decode_stride = int(getattr(args, "decode_stride"))

    # decode context lengths
    dcl = getattr(args, "decode_context_lens", None)
    if dcl is not None:
        dcl_s = str(dcl).strip()
        if dcl_s and dcl_s.lower() not in ("auto", "none"):
            cfg.decode_context_lens = [int(x) for x in dcl_s.split(",") if str(x).strip()]
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(argv) if argv is not None else sys.argv[1:]
    subcmds = {"export", "run-npu", "run-pim", "merge", "collect-merge", "all"}

    if not argv or argv[0].startswith("-") or argv[0] not in subcmds:
        argv = ["all"] + argv

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # export
    p_exp = sub.add_parser("export", help="export npu_tasks.json and pim_tasks.json (segment-level) from schedule csv(s)")
    p_exp.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_exp.add_argument("--schedules", type=str, nargs="+", default=None, help="schedule csv paths (one or more)")
    p_exp.add_argument("--comms", type=str, default=None, help="(optional) single comms trace csv path (weight_load/kv_load, layer mapping)")
    p_exp.add_argument("--comms-traces", type=str, nargs="+", default=None,
                       help="(optional) comms trace csv paths (one or more); if same count as --schedules, treated as paired")
    p_exp.add_argument("--out-dir", type=str, default=".", help="output directory")
    p_exp.add_argument("--prefix", type=str, default="tasks", help="output file prefix")
    p_exp.add_argument("--decode-stride", type=int, default=1)
    add_common_model_args(p_exp)

    # run-npu
    p_npu = sub.add_parser("run-npu", help="run NPU LUT lookup from npu_tasks.json -> npu_results.json")
    p_npu.add_argument("--tasks", type=str, required=True, help="npu_tasks.json")
    p_npu.add_argument("--out", type=str, required=True, help="npu_results.json")
    p_npu.add_argument("--mmad-lut", type=str, default=None, help="Override MMAD LUT path (sets env TRIFORM_MMAD_LUT)")
    p_npu.add_argument("--softmax-lut", type=str, default=None, help="Override Softmax LUT path (sets env TRIFORM_SOFTMAX_LUT)")
    p_npu.add_argument("--gelu-lut", type=str, default=None, help="Override GELU LUT path (sets env TRIFORM_GELU_LUT)")
    p_npu.add_argument("--norm-lut", type=str, default=None, help="Override Norm/LN LUT path (sets env TRIFORM_NORM_LUT)")
    p_npu.add_argument("--npu-dtype", type=str, default=None, help="dtype for activation RW model (fp16|bf16|fp32|int8)")
    p_npu.add_argument("--npu-mem-bw-gbs", type=float, default=0.0, help="REQUIRED: activation bandwidth in GB/s (must be >0)")
    p_npu.add_argument("--npu-op-overhead-us", type=float, default=0.0, help="Constant overhead per op (us)")
    p_npu.add_argument("--npu-no-mem-bound", action="store_true", help="(DEPRECATED/unsupported) Activation RW is mandatory; this flag will error")
    p_npu.add_argument("--batch", type=int, default=None, help="override batch size in tasks config (affects LUT dims)")
    p_npu.add_argument("--debug", action="store_true", help="Verbose per-op debug for NPU lookup")
    p_npu.add_argument("--debug-txt", type=str, default=None, help="Also write debug log to this file")

    # run-pim
    p_pim = sub.add_parser("run-pim", help="run PIM segment simulation from pim_tasks.json -> pim_results.json")
    p_pim.add_argument("--tasks", type=str, required=True, help="pim_tasks.json")
    p_pim.add_argument("--out", type=str, required=True, help="pim_results.json")
    p_pim.add_argument("--cent-sim-root", type=str, default=None,
                       help="path to .../submodules/CENT/cent_simulation (or set env CENT_SIM_ROOT)")
    p_pim.add_argument("--pim-ramulator-bin", type=str, default=None, help="path/name of AiM-enabled ramulator2 executable (required unless present in pim_tasks.json config)")
    p_pim.add_argument("--pim-ramulator-config", type=str, required=True,
                       help="ramulator2 config file (YAML/JSON), e.g. example.yaml")
    p_pim.add_argument("--pim-hw-json", type=str, required=True,
                       help="PIM HW spec JSON, e.g. PIM_AiM.json")
    p_pim.add_argument("--pim-num-devices", type=int, default=None,
                       help="Override PIM device/DIMM count (used as FC_devices for model-parallel mapping). This overrides the value stored in pim_tasks.json.")
    p_pim.add_argument("--batch", type=int, default=None,
                       help="override batch size in tasks config (affects PIM trace generation; default uses pim_tasks.json config.batch)")
    p_pim.add_argument("--debug", action="store_true",
                       help="Verbose debug: print all resolved configs + per-op parameters/latencies (very noisy)")
    p_pim.add_argument("--debug-txt", type=str, default=None,
                       help="(optional) also write the --debug log to this file")
    p_pim.add_argument("--no-cache", action="store_true",
                       help="Disable CostModel PIM latency cache (force ramulator runs for every op)")

    # merge
    p_m = sub.add_parser("merge", help="merge schedule(s) with npu_results.json + pim_results.json (segment-level)")
    p_m.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_m.add_argument("--schedules", type=str, nargs="+", default=None)
    p_m.add_argument("--npu-results", type=str, default=None)
    p_m.add_argument("--pim-results", type=str, default=None)
    p_m.add_argument("--comm-model", type=str, default="schedule", choices=["schedule", "cxl", "none"])
    p_m.add_argument("--pcie-lanes", type=int, default=16)
    p_m.add_argument("--decode-stride", type=int, required=True,
                     help="decode stride used in simulation; decode_token_index = 1 + step*stride")
    p_m.add_argument("--comms", type=str, default=None,
                     help="(optional) single comms trace csv path; used for weight_load/kv_load accumulation")
    p_m.add_argument("--comms-traces", type=str, nargs="+", default=None,
                     help="(optional) comms trace csv paths; if same count as --schedules, treated as paired")
    p_m.add_argument("--overlap", type=float, default=None,
                     help="Allow overlap between NPU B-stage and comms.csv weight_load. 0=no overlap, 1=full overlap.")
    p_m.add_argument("--non-overlap", type=float, default=None,
                     help=argparse.SUPPRESS)
    p_m.add_argument("--kv-load-bw-gbs", type=float, default=None,
                     help="Override config.kv_load_bw_gbs for implicit kv_load estimate (GB/s).")
    p_m.add_argument("--kv-dtype-bytes", type=float, default=None, dest="kv_dtype_bytes",
                     help="Override config.kv_dtype_bytes for implicit kv_load estimate.")
    p_m.add_argument("--kv-load-overhead-us", type=float, default=None, dest="kv_load_overhead_us",
                     help="Override config.kv_load_overhead_us for implicit kv_load estimate.")
    p_m.add_argument("--n-kv-heads", type=int, default=None, dest="n_kv_heads",
                     help="Override config.n_kv_heads for implicit kv_load estimate.")
    p_m.add_argument("--batch", type=int, default=None,
                     help="Override config.batch for implicit kv_load estimate.")
    p_m.add_argument("--out-csv", type=str, required=True,
                     help="where to save merged latency report (csv)")
    p_m.add_argument("--out-steps-csv", type=str, default=None,
                     help="(optional) save per-decode-step block latency (csv)")
    p_m.add_argument("--allow-missing", action="store_true",
                     help="if a segment key is missing in results, treat its cost as 0 instead of error")
    p_m.add_argument("--slack-model", type=str, default="span", choices=["span", "none"],
                     help="Slack modeling for npu/pim device timelines inside each (phase,step,layer,device): 'span' adds schedule slack; 'none' disables.")
    p_m.add_argument("--slack-no-clamp-negative", action="store_true",
                     help="Do not clamp negative slack to 0 (advanced; default clamps).")
    p_m.add_argument("--debug", action="store_true",
                     help="write per-segment (measured vs schedule-trace) comparison into a txt file")
    p_m.add_argument("--debug-txt", type=str, default=None,
                     help="(optional) output path for --debug; default: <out-csv>.debug.txt")

    # collect-merge
    p_c = sub.add_parser("collect-merge", help="collect multiple merge.csv into a single CSV")
    p_c.add_argument("--list", type=str, required=True,
                     help="TSV file listing merge csv paths. Recommended column name: merge_csv")
    p_c.add_argument("--out-csv", type=str, required=True,
                     help="output combined CSV path")
    p_c.add_argument("--skip-missing", action="store_true",
                     help="skip missing merge csvs instead of failing")

    # all
    p_all = sub.add_parser("all", help="single-machine mode: export -> run-npu -> run-pim -> merge")
    p_all.add_argument("--schedule", type=str, default=None, help="single schedule csv path")
    p_all.add_argument("--schedules", type=str, nargs="+", default=None)
    p_all.add_argument("--comms", type=str, default=None, help="(optional) single comms trace csv path (weight_load/kv_load, layer-scope)")
    p_all.add_argument("--comms-traces", type=str, nargs="+", default=None,
                       help="(optional) comms trace csv paths (one or more); if same count as --schedules, treated as paired")
    p_all.add_argument("--out-dir", type=str, default=".", help="where to place tasks/results")
    p_all.add_argument("--prefix", type=str, default="run", help="prefix for tasks/results files")
    p_all.add_argument("--mmad-lut", type=str, default=None, help="Override MMAD LUT path (sets env TRIFORM_MMAD_LUT)")
    p_all.add_argument("--softmax-lut", type=str, default=None, help="Override Softmax LUT path (sets env TRIFORM_SOFTMAX_LUT)")
    p_all.add_argument("--gelu-lut", type=str, default=None, help="Override GELU LUT path (sets env TRIFORM_GELU_LUT)")
    p_all.add_argument("--norm-lut", type=str, default=None, help="Override Norm/LN LUT path (sets env TRIFORM_NORM_LUT)")
    p_all.add_argument("--npu-dtype", type=str, default=None, help="dtype for activation RW model (fp16|bf16|fp32|int8)")
    p_all.add_argument("--npu-mem-bw-gbs", type=float, default=0.0, help="REQUIRED: activation bandwidth in GB/s (must be >0)")
    p_all.add_argument("--npu-op-overhead-us", type=float, default=0.0, help="Constant overhead per op (us)")
    p_all.add_argument("--npu-no-mem-bound", action="store_true", help="(DEPRECATED/unsupported) Activation RW is mandatory; this flag will error")
    p_all.add_argument("--npu-debug", action="store_true", help="Verbose per-op debug for NPU lookup (separate from merge --debug)")
    p_all.add_argument("--npu-debug-txt", type=str, default=None, help="Also write NPU debug log to this file")
    p_all.add_argument("--cent-sim-root", type=str, default=None)
    p_all.add_argument("--pim-hw-json", type=str, default=None)
    p_all.add_argument("--comm-model", type=str, default="schedule", choices=["schedule", "cxl", "none"])
    p_all.add_argument("--pcie-lanes", type=int, default=16)
    p_all.add_argument("--decode-stride", type=int, required=True,
                       help="decode stride used in simulation; decode_token_index = 1 + step*stride")
    p_all.add_argument("--overlap", type=float, default=None,
                       help="Allow overlap between NPU B-stage and comms.csv weight_load. 0=no overlap, 1=full overlap.")
    p_all.add_argument("--non-overlap", type=float, default=None,
                       help=argparse.SUPPRESS)
    p_all.add_argument("--merge-out-csv", type=str, default=None,
                       help="where to save merged latency report (csv); default: <out-dir>/<prefix>.merge.csv")
    p_all.add_argument("--merge-out-steps-csv", type=str, default=None,
                       help="(optional) save per-decode-step block latency (csv)")
    p_all.add_argument("--allow-missing", action="store_true")
    p_all.add_argument("--debug", action="store_true",
                       help="write per-segment (measured vs schedule-trace) comparison into a txt file during merge")
    p_all.add_argument("--debug-txt", type=str, default=None,
                       help="(optional) output path for --debug; default: alongside merge_out_csv")
    p_all.add_argument("--slack-model", type=str, default="span", choices=["span", "none"],
                       help="Slack modeling for npu/pim device timelines inside each (phase,step,layer,device) during merge.")
    p_all.add_argument("--slack-no-clamp-negative", action="store_true",
                       help="Do not clamp negative slack to 0 (advanced; default clamps).")
    add_common_model_args(p_all)

    args = parser.parse_args(argv)

    if args.cmd == "export":
        cfg = build_cfg_from_args(args)
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        comms_paths = ([args.comms] if getattr(args, "comms", None) else (getattr(args, "comms_traces", None) or []))
        if not schedule_paths:
            raise SystemExit("export requires --schedule or --schedules")
        export_tasks(schedule_paths, cfg, args.out_dir, args.prefix, comms_paths=comms_paths)
        return

    if args.cmd == "run-npu":
        _set_npu_lut_env(
            mmad_lut=getattr(args, "mmad_lut", None),
            softmax_lut=getattr(args, "softmax_lut", None),
            gelu_lut=getattr(args, "gelu_lut", None),
            norm_lut=getattr(args, "norm_lut", None),
        )
        run_npu(
            args.tasks,
            args.out,
            npu_dtype=getattr(args, "npu_dtype", None),
            npu_mem_bw_gbs=float(getattr(args, "npu_mem_bw_gbs", 0.0) or 0.0),
            op_overhead_us=float(getattr(args, "npu_op_overhead_us", 0.0) or 0.0),
            use_mem_bound=not bool(getattr(args, "npu_no_mem_bound", False)),
            batch=getattr(args, "batch", None),
            debug=bool(getattr(args, "debug", False)),
            debug_txt=getattr(args, "debug_txt", None),
        )
        return

    if args.cmd == "run-pim":
        run_pim(
            args.tasks,
            args.out,
            batch=getattr(args, "batch", None),
            cent_sim_root=args.cent_sim_root,
            ramulator_bin=args.pim_ramulator_bin,
            ramulator_config=args.pim_ramulator_config,
            pim_hw_json=args.pim_hw_json,
            pim_num_devices=args.pim_num_devices,
            debug=bool(getattr(args, 'debug', False)),
            debug_txt=getattr(args, 'debug_txt', None),
            no_cache=bool(getattr(args, 'no_cache', False)),
        )
        return

    if args.cmd == "merge":
        if (args.npu_results is None and args.pim_results is None):
            raise SystemExit("merge requires --npu-results and/or --pim-results")
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        if not schedule_paths:
            raise SystemExit("merge requires --schedule or --schedules")
        comms_paths = ([args.comms] if getattr(args, "comms", None) else (getattr(args, "comms_traces", None) or []))
        merge(
            schedule_paths,
            args.npu_results,
            args.pim_results,
            comms_paths=comms_paths,
            overlap=(args.overlap if args.overlap is not None else (args.non_overlap if args.non_overlap is not None else 0.0)),
            comm_model=args.comm_model,
            pcie_lanes=args.pcie_lanes,
            decode_stride=args.decode_stride,
            kv_load_bw_gbs_override=getattr(args, "kv_load_bw_gbs", None),
            kv_dtype_bytes_override=getattr(args, "kv_dtype_bytes", None),
            kv_load_overhead_us_override=getattr(args, "kv_load_overhead_us", None),
            n_kv_heads_override=getattr(args, "n_kv_heads", None),
            batch_override=getattr(args, "batch", None),
            out_csv=args.out_csv,
            out_steps_csv=args.out_steps_csv,
            allow_missing=args.allow_missing,
            slack_model=getattr(args, "slack_model", "span"),
            slack_clamp_negative=(not bool(getattr(args, "slack_no_clamp_negative", False))),
            debug=args.debug,
            debug_txt=args.debug_txt,
        )
        return

    if args.cmd == "collect-merge":
        collect_merge_csv(args.list, out_csv=args.out_csv, skip_missing=args.skip_missing)
        return

    if args.cmd == "all":
        cfg = build_cfg_from_args(args)
        out_dir = args.out_dir
        prefix = args.prefix
        schedule_paths = ([args.schedule] if getattr(args, "schedule", None) else (args.schedules or []))
        comms_paths = ([args.comms] if getattr(args, "comms", None) else (getattr(args, "comms_traces", None) or []))
        if not schedule_paths:
            raise SystemExit("all requires --schedule or --schedules")

        npu_tasks, pim_tasks = export_tasks(schedule_paths, cfg, out_dir, prefix, comms_paths=comms_paths)

        _set_npu_lut_env(
            mmad_lut=getattr(args, "mmad_lut", None),
            softmax_lut=getattr(args, "softmax_lut", None),
            gelu_lut=getattr(args, "gelu_lut", None),
            norm_lut=getattr(args, "norm_lut", None),
        )

        npu_res_path = None
        pim_res_path = None

        npu_data = _load_json(str(npu_tasks))
        if len(npu_data.get("tasks", [])) > 0:
            npu_res_path = run_npu(
                str(npu_tasks),
                str(Path(out_dir) / f"{prefix}.npu_results.json"),
                npu_dtype=getattr(args, "npu_dtype", None),
                npu_mem_bw_gbs=float(getattr(args, "npu_mem_bw_gbs", 0.0) or 0.0),
                op_overhead_us=float(getattr(args, "npu_op_overhead_us", 0.0) or 0.0),
                use_mem_bound=not bool(getattr(args, "npu_no_mem_bound", False)),
                batch=getattr(args, "batch", None),
                debug=bool(getattr(args, "npu_debug", False)),
                debug_txt=getattr(args, "npu_debug_txt", None),
            )

        pim_data = _load_json(str(pim_tasks))
        if len(pim_data.get("tasks", [])) > 0:
            if not getattr(args, "pim_ramulator_config", None):
                raise SystemExit("all + run-pim requires --pim-ramulator-config")
            if not getattr(args, "pim_hw_json", None):
                raise SystemExit("all + run-pim requires --pim-hw-json")
            pim_res_path = run_pim(
                str(pim_tasks),
                str(Path(out_dir) / f"{prefix}.pim_results.json"),
                cent_sim_root=args.cent_sim_root,
                ramulator_bin=getattr(args, "pim_ramulator_bin", None),
                ramulator_config=getattr(args, "pim_ramulator_config", None),
                pim_hw_json=getattr(args, "pim_hw_json", None),
                pim_num_devices=getattr(args, "pim_num_devices", None),
                batch=getattr(args, "batch", None),
            )

        merge(
            schedule_paths,
            str(npu_res_path) if npu_res_path else None,
            str(pim_res_path) if pim_res_path else None,
            comms_paths=comms_paths,
            overlap=(args.overlap if args.overlap is not None else (args.non_overlap if args.non_overlap is not None else 0.0)),
            comm_model=args.comm_model,
            pcie_lanes=args.pcie_lanes,
            decode_stride=args.decode_stride,
            out_csv=(args.merge_out_csv or str(Path(out_dir) / f"{prefix}.merge.csv")),
            out_steps_csv=(args.merge_out_steps_csv),
            allow_missing=args.allow_missing,
            slack_model=getattr(args, "slack_model", "span"),
            slack_clamp_negative=(not bool(getattr(args, "slack_no_clamp_negative", False))),
            debug=args.debug,
            debug_txt=args.debug_txt,
        )
        return

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
