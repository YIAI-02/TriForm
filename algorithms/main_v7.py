from __future__ import annotations
from config import attach_local_debug_filter
import argparse
import json
import os
import time
import math
import random
import re
from typing import Dict, List, Callable, Any, Tuple
from hardware import demo_cluster, Cluster
from cost_model import CostModel, DTYPE_BYTES
from cost_model_pim_backend import _make_shared_model_dict
from buffer_manager import GlobalMemoryManager
from model_parser import build_graph
from config import (
    PIM_STATIC_ALLOC_RATIO,
    ENABLE_PIM_WEIGHT_PRELOAD,
    setup_logging,
)
from plan_label import PlanLabel
from scheduler import (
    HEFTScheduler,
    NaiveTopoScheduler,
)

# Optional: communication-aware HEFT (may not be present in older versions)
try:
    from scheduler import HEFTCOMMAWAREScheduler
except Exception:  # pragma: no cover
    HEFTCOMMAWAREScheduler = None
from pathlib import Path
import logging
from task_graph import TaskGraph, TaskNode
from stats_recorder import reset_simulation_logger
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: True)

_WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY = False
_WEIGHT_SUGGEST_PROGRESS_ENABLED = False
_WEIGHT_SUGGEST_AL_LOGGER: logging.Logger | None = None
_WEIGHT_SUGGEST_AL_LOG_PATH: str | None = None

def _reset_weight_suggest_al_logger() -> None:
    global _WEIGHT_SUGGEST_AL_LOGGER, _WEIGHT_SUGGEST_AL_LOG_PATH
    if _WEIGHT_SUGGEST_AL_LOGGER is not None:
        for h in list(_WEIGHT_SUGGEST_AL_LOGGER.handlers):
            _WEIGHT_SUGGEST_AL_LOGGER.removeHandler(h)
            try:
                h.flush()
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
    _WEIGHT_SUGGEST_AL_LOGGER = None
    _WEIGHT_SUGGEST_AL_LOG_PATH = None

def _setup_weight_suggest_al_logger(log_file: str | None) -> None:
    global _WEIGHT_SUGGEST_AL_LOGGER, _WEIGHT_SUGGEST_AL_LOG_PATH
    _reset_weight_suggest_al_logger()
    if not log_file:
        return

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    al_logger = logging.getLogger(f"{__name__}.weight_suggest_al")
    al_logger.setLevel(logging.DEBUG)
    al_logger.propagate = False

    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] {__name__}: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    al_logger.addHandler(fh)

    _WEIGHT_SUGGEST_AL_LOGGER = al_logger
    _WEIGHT_SUGGEST_AL_LOG_PATH = str(path)


def _emit_weight_suggest_al_log(msg: str, *, level: int = logging.DEBUG) -> None:
    text = str(msg or "")
    if not text or _WEIGHT_SUGGEST_AL_LOGGER is None:
        return
    try:
        _WEIGHT_SUGGEST_AL_LOGGER.log(level, text)
    except Exception:
        pass
 

def _set_weight_suggest_debug_summary_only(enabled: bool, *, emit_progress: bool | None = None) -> None:
    global _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY, _WEIGHT_SUGGEST_PROGRESS_ENABLED
    _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY = bool(enabled)
    if emit_progress is not None:
        _WEIGHT_SUGGEST_PROGRESS_ENABLED = bool(emit_progress)


def _render_log_message(msg: Any, args: tuple[Any, ...]) -> str:
    text = str(msg)
    if not args:
        return text
    try:
        return text % args
    except Exception:
        try:
            return " ".join([text, *(str(a) for a in args)])
        except Exception:
            return text


def _is_key_weight_suggest_al_message(msg: str) -> bool:
    text = str(msg or "")
    if text.startswith('[BASELINE]'):
        return (' start ' in text) or (' done total=' in text)
    if '[AL]' not in text:
        return False
    if text.startswith('[AL] init:'):
        return True
    if 'outer0->outer1: initial assign' in text:
        return True
    if re.search(r"\[AL\] inner\d+: ACCEPT ", text):
        return True
    if re.search(r"\[AL\]\[[^\]]+\] outer\d+: baseline total=", text):
        return True
    if re.search(r"\[AL\]\[[^\]]+\] outer\d+: after inner total=", text):
        return True
    if re.search(r"\[AL\]\[[^\]]+\] outer\d+: total .* stop\.$", text):
        return True
    if 'no ND blocks to split' in text:
        return True
    return False


def _emit_weight_suggest_progress(msg: str) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(msg)
    else:
        print(msg)


def _debug(msg: Any, *args: Any, **kwargs: Any) -> None:
    text: str | None = None
    if _WEIGHT_SUGGEST_AL_LOGGER is not None or _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY:
        text = _render_log_message(msg, tuple(args))
        if '[AL]' in text:
            _emit_weight_suggest_al_log(text, level=logging.DEBUG)
    
    if _WEIGHT_SUGGEST_DEBUG_SUMMARY_ONLY:
        if not _WEIGHT_SUGGEST_PROGRESS_ENABLED:
            return
        text = _render_log_message(msg, tuple(args))
        if text is None:
            text = _render_log_message(msg, tuple(args))
        if _is_key_weight_suggest_al_message(text):
            _emit_weight_suggest_progress(text)
        return
    logger.debug(msg, *args, **kwargs)


def _normalize_weight_storage_fmt(value: Any, *, default: str = 'ND') -> str:
    raw = str(default if value is None else value).strip().lower()
    raw = raw.replace('_', '-').replace(' ', '')
    if raw in ('', 'nd', 'dense', 'normal-dense', 'all-nd'):
        return 'ND'
    if raw in ('nz', 'npu-opt', 'npuopt', 'npu-optimized', 'all-nz'):
        return 'NZ'
    if raw in ('pim-opt', 'pimopt', 'pim-optimized', 'all-pim-opt'):
        return 'PIM-OPT'
    if raw in (
        'dual',
        'dual-copy',
        'dualcopy',
        'twocopy',
        'two-copy',
        'both',
        'nz+pim-opt',
        'nz+pimopt',
        'pim-opt+nz',
        'pimopt+nz',
        'all-dual',
    ):
        return 'DUAL'
    raise ValueError(
        f"Unknown weight storage format '{value}'. Expected one of: nd, nz, pim-opt, dual-copy"
    )



def _weight_map_format_counts(weight_ids: List[str], fmt_map: Dict[str, str]) -> Dict[str, int]:
    counts = {'ND': 0, 'NZ': 0, 'PIM-OPT': 0, 'DUAL': 0}
    fm = dict(fmt_map or {})
    for wid in weight_ids:
        try:
            fmt = _normalize_weight_storage_fmt(fm.get(str(wid), 'ND'))
        except Exception:
            fmt = 'ND'
        counts[fmt] += 1
    return counts


def _weight_map_summary(weight_ids: List[str], fmt_map: Dict[str, str]) -> Dict[str, Any]:
    return {
        'total_weights': int(len(weight_ids)),
        'explicit_weights': int(len(dict(fmt_map or {}))),
        'counts': _weight_map_format_counts(weight_ids, fmt_map),
    }


def _collect_weight_ids_from_graph(g: TaskGraph) -> List[str]:
    return sorted({str(n.weight_id) for n in getattr(g, 'nodes', {}).values() if getattr(n, 'weight_id', None)})


def _build_uniform_weight_storage_map(g: TaskGraph, storage_fmt: str | None) -> Dict[str, str]:
    fmt = _normalize_weight_storage_fmt(storage_fmt or 'ND')
    if fmt == 'ND':
        return {}
    return {wid: fmt for wid in _collect_weight_ids_from_graph(g)}


def _storage_mode_display_name(storage_fmt: str | None) -> str:
    fmt = _normalize_weight_storage_fmt(storage_fmt or 'ND')
    if fmt == 'DUAL':
        return 'DUAL'
    if fmt == 'ND':
        return 'Linear'
    return str(fmt)


def _artifact_tag_token(tag: str | None) -> str:
    raw = str(tag or '').strip().lower()
    if not raw:
        return ''
    raw = re.sub(r'[^0-9a-zA-Z_.-]+', '_', raw)
    raw = raw.strip('_.-')
    return raw

# Default report paths (used when config/CLI doesn't override)
ALL_PASSES_RESULT_PATH = "./output/all_passes.json"
BEST_PASS_SUMMARY_PATH = "./output/best_summary.json"

def _result_stride_for_naming(cfg: Dict) -> Any:
    stride = cfg.get('decode_plan_refresh_stride', None)
    if stride in (None, ''):
        stride = cfg.get('decode_sample_stride', None)
    return stride

# ---- path helper (unify result_dir naming incl. batch) ----
def _build_result_dir(cfg: Dict, default_root: str = './output') -> Path:
    """
    Compose a result directory path that always includes batch:
            <base>/<family>_<variant>_<dtype>_b<batch>[_s<refresh_stride>]
    """
    base   = cfg.get('result_dir') or default_root
    family = cfg.get('model_family', 'unnamed')
    variant= cfg.get('model_variant', '')
    dtype  = cfg.get('dtype', 'fp16')
    batch  = int(cfg.get('batch', 1))
    stride = _result_stride_for_naming(cfg)
    stride_suffix = f"_s{int(stride)}" if stride not in (None, '') else ""
    return Path(base) / f"{family}_{variant}_{dtype}_b{batch}{stride_suffix}"

def _build_tag(cfg: dict) -> str:
    """Build a safe tag for output files: SxT + optional refresh stride if provided."""
    try:
        S = int(cfg.get('prefill_len', 0) or 0)
    except Exception:
        S = 0
    try:
        T = int(cfg.get('decode_len', 0) or 0)
    except Exception:
        T = 0
    parts = [f"{S}x{T}"]
    st = _result_stride_for_naming(cfg)
    try:
        if st is not None:
            stv = int(st)
            if stv > 0:
                parts.append(f"st{stv}")
    except Exception:
        pass
    return "_".join(parts)

# KV placement helpers
def _normalize_kv_place(kv_place: str) -> str:
    """Normalize KV placement tags to one of: 'host' | 'pim' | 'npu'."""
    s = str(kv_place or '').strip().lower()
    if s in ('cpu', 'host', 'dram'):
        return 'host'
    if s in ('pim', 'aim'):
        return 'pim'
    if s in ('npu', 'gpu', 'device'):
        return 'npu'
    return 'host'

def _infer_kv_dtype_bytes_from_graph(cfg: Dict, graph: TaskGraph) -> float:
    """Infer KV-cache storage element size (bytes)."""
    default_b = float(DTYPE_BYTES.get(cfg.get('dtype', 'fp16'), 2))
    try:
        for n in graph.nodes.values():
            attrs = getattr(n, 'attrs', None) or {}
            if not isinstance(attrs, dict):
                continue
            opt = attrs.get('opt', None)
            if isinstance(opt, dict) and ('kv_dtype_bytes' in opt):
                kb = opt.get('kv_dtype_bytes', None)
                if kb is None:
                    continue
                try:
                    kb_f = float(kb)
                    if kb_f > 0:
                        return float(kb_f)
                except Exception:
                    continue
    except Exception:
        pass
    return float(default_b)


def _effective_tp_qkv(cfg: Dict) -> int:
    """Validated effective TP factor used for KV-head sharding."""
    tp_eff = cfg.get('tp_qkv_effective', cfg.get('_tp_qkv_effective', None))
    if tp_eff is not None:
        return max(1, int(tp_eff))
    return max(1, int(cfg.get('tp_qkv', 1) or 1))


def _compute_kv_plan_info(
    *,
    cfg: Dict,
    cluster: Cluster,
    graph: TaskGraph,
    shape: Any,
) -> Dict[str, Any]:
    """Compute KV/weight sizes and (if PIM exists) a deterministic KV-head->PIM mapping."""
    pim_devs = list(cluster.devices_by_type('pim') or [])
    npu_devs = list(cluster.devices_by_type('npu') or [])

    kv_dtype_bytes = float(_infer_kv_dtype_bytes_from_graph(cfg, graph))
    S = int(cfg.get('prefill_len', 128))
    T = int(cfg.get('decode_len', 32))
    batch = int(cfg.get('batch', 1))

    layers = int(getattr(shape, 'layer_num', 1) or 1)
    n_kv_heads = int(getattr(shape, 'n_kv_heads', 1) or 1)
    head_dim = int(
        getattr(
            shape,
            'head_dim',
            max(1, int(getattr(shape, 'dim', 1) or 1) // max(1, int(getattr(shape, 'n_heads', 1) or 1))),
        )
        or 1
    )

    KV_total_bytes = int(math.ceil(2 * (S + T) * n_kv_heads * head_dim * batch * layers * kv_dtype_bytes))

    # Sum FC weight bytes from graph.
    FC_total_bytes = 0
    for n in graph.nodes.values():
        FC_total_bytes += int(getattr(n, 'weight_size', 0) or 0)

    pim_rr = sorted(pim_devs, key=lambda d: str(d.name))
    pim_bytes_by_name = {d.name: int(d.mem_capacity_GB * (1024**3)) for d in pim_rr}
    pim_bytes_total = int(sum(pim_bytes_by_name.values()))

    # NPU: choose the single best device for KV (largest capacity).
    best_npu = None
    best_npu_cap = 0
    for d in npu_devs:
        cap = int(float(getattr(d, 'mem_capacity_GB', 0.0) or 0.0) * (1024**3))
        if cap > best_npu_cap:
            best_npu_cap = cap
            best_npu = d
    best_npu_name = str(getattr(best_npu, 'name', '')) if best_npu is not None else None

    # Build KV-head shards (only meaningful when PIM exists).
    kv_head_to_pim: Dict[int, str] = {}
    kv_heads_by_pim: Dict[str, List[int]] = {d.name: [] for d in pim_rr}
    kv_bytes_by_pim: Dict[str, int] = {d.name: 0 for d in pim_rr}

    tp_qkv_eff = int(_effective_tp_qkv(cfg))
    kv_heads_total = int(n_kv_heads)
    tp_qkv_eff = max(1, min(int(tp_qkv_eff), kv_heads_total))
    if kv_heads_total % tp_qkv_eff != 0:
        # Should have been validated earlier; fallback to per-head sharding.
        tp_qkv_eff = kv_heads_total
    kv_heads_per_shard = max(1, kv_heads_total // tp_qkv_eff)

    # Build head shards.
    head_shards: List[List[int]] = []
    for si in range(tp_qkv_eff):
        s0 = si * kv_heads_per_shard
        s1 = min(kv_heads_total, (si + 1) * kv_heads_per_shard)
        head_shards.append(list(range(s0, s1)))

    # Assign shards to PIMs (balanced).
    if pim_rr:
        pn = len(pim_rr)
        base = len(head_shards) // pn
        rem = len(head_shards) % pn
        sh_idx = 0
        for pi, dev in enumerate(pim_rr):
            take = base + (1 if pi < rem else 0)
            for _ in range(take):
                if sh_idx >= len(head_shards):
                    break
                shard_heads = head_shards[sh_idx]
                sh_idx += 1
                for hid in shard_heads:
                    kv_head_to_pim[int(hid)] = str(dev.name)
                kv_heads_by_pim[str(dev.name)].extend(int(h) for h in shard_heads)

        # Compute per-PIM KV bytes.
        bytes_per_head_all_layers = float(2 * (S + T) * head_dim * batch * layers) * kv_dtype_bytes
        for dev in pim_rr:
            hcnt = len(kv_heads_by_pim.get(str(dev.name), []) or [])
            kv_bytes_by_pim[str(dev.name)] = int(math.ceil(float(hcnt) * bytes_per_head_all_layers))

    # Feasibility summaries (used when building specific labels).
    feasible_pim = False
    if pim_bytes_total > 0 and KV_total_bytes <= pim_bytes_total:
        feasible_pim = True
        for d in pim_rr:
            need = int(kv_bytes_by_pim.get(d.name, 0))
            cap = int(pim_bytes_by_name.get(d.name, 0))
            if need > cap:
                feasible_pim = False
                break

    feasible_npu = bool(best_npu is not None and int(best_npu_cap) > 0 and int(KV_total_bytes) <= int(best_npu_cap))

    return {
        'kv_total_bytes_all': int(KV_total_bytes),
        'kv_dtype_bytes': float(kv_dtype_bytes),
        'fc_total_bytes': int(FC_total_bytes),
        'tp_qkv_effective': int(tp_qkv_eff),
        'pim_total_capacity_bytes': int(pim_bytes_total),
        'pim_bytes_by_name': dict(pim_bytes_by_name),
        'kv_head_to_pim': dict(kv_head_to_pim),
        'kv_heads_by_pim': dict(kv_heads_by_pim),
        'kv_bytes_by_pim': dict(kv_bytes_by_pim),
        'feasible_pim': bool(feasible_pim),
        'best_npu_name': best_npu_name,
        'best_npu_cap_bytes': int(best_npu_cap),
        'feasible_npu': bool(feasible_npu),
    }


def _make_label_from_kv_plan(
    *,
    cfg: Dict,
    kv_plan: Dict[str, Any],
    kv_place: str,
) -> Tuple[PlanLabel, bool]:
    """Build a PlanLabel for a specific KV placement using precomputed kv_plan."""

    kv_place_req = _normalize_kv_place(kv_place)

    KV_total_bytes = int(kv_plan.get('kv_total_bytes_all', 0) or 0)
    FC_total_bytes = int(kv_plan.get('fc_total_bytes', 0) or 0)
    pim_bytes_total = int(kv_plan.get('pim_total_capacity_bytes', 0) or 0)
    feasible_pim = bool(kv_plan.get('feasible_pim', False))
    feasible_npu = bool(kv_plan.get('feasible_npu', False))
    best_npu_name = kv_plan.get('best_npu_name', None)

    # PIM preload/weight budget depends on whether KV is placed on PIM.
    if kv_place_req == 'pim':
        preload_ok = (int(FC_total_bytes) + int(KV_total_bytes)) <= int(pim_bytes_total)
    else:
        preload_ok = int(FC_total_bytes) <= int(pim_bytes_total)

    weights_preloaded_on_pim = bool(
        bool(ENABLE_PIM_WEIGHT_PRELOAD)
        and pim_bytes_total > 0
        and bool(preload_ok)
    )

    kv_bytes_in_pim = int(KV_total_bytes) if (kv_place_req == 'pim' and feasible_pim) else 0
    leftover_bytes = max(0, int(pim_bytes_total) - int(kv_bytes_in_pim))
    weight_budget = int(min(int(FC_total_bytes), int(leftover_bytes * PIM_STATIC_ALLOC_RATIO)))
    if bool(weights_preloaded_on_pim):
        weight_budget = int(FC_total_bytes)

    if kv_place_req == 'pim' and feasible_pim:
        kv_place_out = 'pim'
        kv_in_pim_out = True
        pim_mode = 'kv_pim_by_head'
    elif kv_place_req == 'npu' and feasible_npu:
        kv_place_out = 'npu'
        kv_in_pim_out = False
        pim_mode = 'kv_npu'
    else:
        kv_place_out = 'host'
        kv_in_pim_out = False
        pim_mode = 'kv_host' if pim_bytes_total > 0 else 'none'

    selected = bool(kv_place_req == kv_place_out)
    if kv_place_req == 'pim':
        feasible = bool(feasible_pim)
    elif kv_place_req == 'npu':
        feasible = bool(feasible_npu)
    else:
        feasible = True

    label = PlanLabel(
        pim_mode=str(pim_mode),
        kv_in_pim=bool(kv_in_pim_out),
        kv_total_bytes=int(kv_bytes_in_pim),
        kv_place=str(kv_place_out),
        kv_in_npu=bool(kv_place_out == 'npu' and feasible_npu),
        kv_npu_device=str(best_npu_name) if (kv_place_out == 'npu' and best_npu_name) else None,
        kv_total_bytes_all=int(KV_total_bytes),
        kv_total_bytes_on_pim=int(kv_bytes_in_pim),
        kv_total_bytes_on_npu=int(KV_total_bytes) if kv_place_out == 'npu' else 0,
        kv_total_bytes_on_host=int(KV_total_bytes) if kv_place_out == 'host' else 0,
        kv_bytes_by_pim=(dict(kv_plan.get('kv_bytes_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_head_to_pim=(dict(kv_plan.get('kv_head_to_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_heads_by_pim=(dict(kv_plan.get('kv_heads_by_pim', {})) if (kv_place_out == 'pim' and feasible_pim) else {}),
        kv_partition_dim='kv_head',
        pim_weight_capacity_bytes=int(weight_budget),
    )

    # Extra metadata used by reporting / debugging (kept as attributes for flexibility).
    setattr(label, 'total_weight_bytes', int(FC_total_bytes))
    setattr(label, 'fc_total_bytes', int(FC_total_bytes))
    setattr(label, 'kv_total_bytes_raw', int(KV_total_bytes))
    setattr(label, 'kv_dtype_bytes', float(kv_plan.get('kv_dtype_bytes', 0.0) or 0.0))
    setattr(label, 'tp_qkv_effective', int(kv_plan.get('tp_qkv_effective', 1) or 1))
    # Tensor-parallel shard knobs (used by cost model/NPU backends).
    setattr(label, 'tp_qkv', int(cfg.get('tp_qkv', 1) or 1))
    setattr(label, 'tp_ffn', int(cfg.get('tp_ffn', 1) or 1))
    setattr(label, 'tp_ffn_effective', int(cfg.get('tp_ffn_effective', cfg.get('tp_ffn', 1)) or 1))
    setattr(label, 'tp_moe', int(cfg.get('tp_moe', cfg.get('tp_ffn', 1)) or 1))
    setattr(label, 'tp_moe_effective', int(cfg.get('tp_moe_effective', cfg.get('tp_moe', cfg.get('tp_ffn', 1))) or 1))
    setattr(label, 'pim_total_capacity_bytes', int(pim_bytes_total))
    setattr(label, 'weights_preloaded_on_pim', bool(weights_preloaded_on_pim))

    return label, bool(selected and bool(feasible))


def _make_label_given_kv_place(
    *,
    cfg: Dict,
    cluster: Cluster,
    graph: TaskGraph,
    shape: Any,
    kv_place: str,
) -> tuple[PlanLabel, bool]:
    """Build a PlanLabel with KV placement forced. """
    kv_plan = _compute_kv_plan_info(cfg=cfg, cluster=cluster, graph=graph, shape=shape)
    return _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place=kv_place)




def _fmt_kv_policy_scores(scores: Any) -> str:
    """Pretty string for kv-policy score dict."""
    if not isinstance(scores, dict) or not scores:
        return ""

    def _one(tag: str) -> str:
        v = scores.get(tag)
        if v is None:
            return f"{tag}=N/A"
        if isinstance(v, (int, float)):
            return f"{tag}.total={float(v):.6f}s"
        if isinstance(v, dict):
            tp = v.get("prefill_s")
            td = v.get("decode_s")
            tt = v.get("total_s")
            if all(isinstance(x, (int, float)) for x in (tp, td, tt)):
                return f"{tag}: prefill={float(tp):.6f}s decode={float(td):.6f}s total={float(tt):.6f}s"
        return f"{tag}=?"

    # Stable order: host -> npu -> pim.
    parts = []
    if "host" in scores:
        parts.append(_one("host"))
    if "npu" in scores:
        parts.append(_one("npu"))
    if "pim" in scores:
        parts.append(_one("pim"))
    # include other keys if present
    for k in sorted(set(scores.keys()) - {"host", "npu", "pim"}):
        parts.append(_one(str(k)))
    return " | ".join(parts)


def _infer_kv_place_from_label(label: Any) -> str:
    """Best-effort KV placement string from a PlanLabel."""
    try:
        if bool(getattr(label, 'kv_in_pim', False)):
            return 'pim'
    except Exception:
        pass
    try:
        if bool(getattr(label, 'kv_in_npu', False)):
            return 'npu'
    except Exception:
        pass
    try:
        kp = getattr(label, 'kv_place', None)
        if kp is not None:
            return _normalize_kv_place(kp)
    except Exception:
        pass
    return 'host'


def _apply_kv_place_constraints(g: TaskGraph, kv_place: str) -> TaskGraph:
    """Force KV read/write operators to execute on the KV storage device."""

    kv_place = _normalize_kv_place(kv_place)

    # Local KV op detector (avoid dependency on baseline helper ordering).
    def _is_kv_rw_node(n: TaskNode) -> bool:
        try:
            nm = (getattr(n, 'name', '') or '').lower()
            op = str((getattr(n, 'attrs', {}) or {}).get('op') or '').lower()
        except Exception:
            nm, op = '', ''
        for k in (
            'kv_read', 'kv_write',
            'k_read', 'v_read', 'k_write', 'v_write',
            'k_cache', 'v_cache',
        ):
            if k in nm or k in op:
                return True
        return False

    g2 = _clone_graph(g)
    for _, n in g2.nodes.items():
        if not _is_kv_rw_node(n):
            continue
        if not isinstance(getattr(n, 'allowed', None), dict):
            n.allowed = {}
        # Force only one device type for KV R/W.
        n.allowed['cpu'] = bool(kv_place == 'host')
        n.allowed['npu'] = bool(kv_place == 'npu')
        n.allowed['pim'] = bool(kv_place == 'pim')
    return g2

def _estimate_total_time_for_label(
    *,
    strategy: str,
    cfg: Dict,
    cluster: Cluster,
    cost: CostModel,
    graph_prefill: TaskGraph,
    graph_decode: TaskGraph | None,
    label: PlanLabel,
) -> tuple[float, float, float]:
    """Return (prefill_s, decode_s, total_s) for one label under one scheduler."""
    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))

    buffer_mgr = GlobalMemoryManager()
    try:
        sched = _make_scheduler(strategy, cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
    except Exception:
        # Fallback: baseline HEFT if an unknown strategy name is supplied.
        sched = _make_scheduler("heft", cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)

    try:
        sched.reset_state()
    except Exception:
        pass
    if hasattr(sched, "set_storage_format_map"):
        try:
            sched.set_storage_format_map({})
        except Exception:
            pass
    g_prefill = graph_prefill
    g_decode = graph_decode if graph_decode is not None else graph_prefill
    t_prefill, _ = simulate_prefill(sched, cfg, g_prefill)
    t_decode, _ = simulate_decode_progressive(sched, cfg, g_decode, prefill_end=t_prefill)
    return float(t_prefill), float(t_decode), float(t_prefill + t_decode)

 

def _normalize_npu_backend(backend):
    """Normalize npu_backend strings to canonical: fast / ascend_310b_lut / llmcompass."""
    if backend is None:
        return None
    b = str(backend).strip().lower().replace('-', '_')
    b = b.replace(' ', '_')
    if b in ('fast', 'fastmode', 'fast_mode'):
        return 'fast'
    if b in ('ascend_310b_lut', 'ascend310b_lut', 'ascend_lut', 'lut', 'runtime_lut'):
        return 'ascend_310b_lut'
    if b in ('ascend_310b_json', 'ascend310b_json', 'ascend_json', 'json', 'runtime_json', 'ascend_310b'):
        return 'ascend_310b_json'
    if b in ('llmcompass', 'llm_compass'):
        return 'llmcompass'
    raise ValueError(
        f"Unknown npu_backend='{backend}'. Expected one of: fast, ascend_310b_lut, ascend_310b_json, llmcompass"
    )


def auto_select_kv_policy(
    *,
    strategy: str,
    cfg: Dict,
    cluster: Cluster,
    cost: CostModel,
    graph: TaskGraph,
    graph_decode: TaskGraph | None = None,
    shape: Any,
    capture_best_schedule: bool = False,
) -> PlanLabel:
    """Choose KV placement by capacity only: prefer PIM, otherwise fall back to Host.
    """
    kv_plan = _compute_kv_plan_info(cfg=cfg, cluster=cluster, graph=graph, shape=shape)
    label_host, _ = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place='host')
    label_pim, ok_pim = _make_label_from_kv_plan(cfg=cfg, kv_plan=kv_plan, kv_place='pim')

    if ok_pim and _infer_kv_place_from_label(label_pim) == 'pim':
        setattr(label_pim, "kv_policy_selected", "pim_by_capacity")
        setattr(label_pim, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
        if bool(capture_best_schedule):
            setattr(label_pim, "_kv_policy_best_sim", None)
        return label_pim

    setattr(label_host, "kv_policy_selected", "host_by_capacity")
    setattr(label_host, "kv_policy_scores", {"host": None, "npu": None, "pim": None})
    if bool(capture_best_schedule):
        setattr(label_host, "_kv_policy_best_sim", None)
    return label_host

def _serialize_schedule(schedule: List[ScheduledTask], *, phase: str, token_idx: int | None=None) -> List[Dict]:
    """Convert ScheduledTask list to JSON-friendly dicts."""
    out: List[Dict] = []
    for t in schedule:
        out.append({'node_id': t.node_id, 'device': t.device, 'start': float(t.start), 'finish': float(t.finish), 'duration': float(max(0.0, t.finish - t.start)), 'phase': phase, 'token_idx': token_idx})
    return out


def simulate_prefill(sched: SchedulerBase, cfg: Dict, graph: TaskGraph) -> tuple[float, List[Dict]]:
    """
    Simulate prefill phase: process entire prefix at once.
    current_length = prefill_len
    """
    prefill_len = int(cfg.get('prefill_len', 128))
    sched.set_seq_len(prefill_len)
    prefill_sched = sched.schedule(graph, phase='prefill')
    prefill_time = sched.makespan(prefill_sched)
    return (prefill_time, _serialize_schedule(prefill_sched, phase='prefill', token_idx=None))


def simulate_decode_progressive(sched: SchedulerBase, cfg: Dict, graph: TaskGraph, prefill_end: float) -> tuple[float, List[Dict]]:
    prefill_len = int(cfg.get('prefill_len', 128))
    decode_len = int(cfg.get('decode_len', 32))
    global_end = float(prefill_end)
    steps_serialized: List[Dict] = []

    if isinstance(cfg, dict):
        dump_raw = cfg.get('decode_sample_stride', 1)
        refresh_cfg = cfg.get('decode_plan_refresh_stride', 64)
        dump_stride = int(1 if dump_raw is None else dump_raw)
        refresh_raw = int(64 if refresh_cfg is None else refresh_cfg)
    else:
        dump_stride = 1
        refresh_raw = 64

    dump_stride = max(1, int(dump_stride))
    refresh_stride = max(0, int(refresh_raw))
    current_plan: Dict[str, Any] | None = None

    def _set_decode_ctx(token_idx: int) -> None:
        setter = getattr(sched, 'set_decode_context', None)
        if callable(setter):
            try:
                setter(cur_token_idx=int(token_idx), total_decode_tokens=int(decode_len), cfg=cfg)
            except TypeError:
                try:
                    setter(cur_token_idx=int(token_idx), total_decode_tokens=int(decode_len))
                except Exception:
                    pass
            except Exception:
                pass

    def _clear_decode_ctx() -> None:
        clearer = getattr(sched, 'clear_decode_context', None)
        if callable(clearer):
            try:
                clearer()
            except Exception:
                pass

    def _need_exact_token(token_idx: int) -> bool:
        if token_idx < 2:
            return True
        if current_plan is None:
            return True
        if refresh_stride > 0 and (token_idx % refresh_stride) == 0:
            return True
        return False

    def _should_dump_schedule(token_idx: int) -> bool:
        if dump_stride <= 1:
            return True
        if token_idx in (0, 1, max(0, decode_len - 1)):
            return True
        return (token_idx % dump_stride) == 0

    def _validate_fixed_plan_or_raise(plan_obj: Any) -> Dict[str, Any]:
        try:
            plan_map = dict(plan_obj or {})
        except Exception as e:
            raise RuntimeError('Fixed decode plan is not mapping-like') from e

        raw_order = plan_map.get('order', None)
        if raw_order is None or isinstance(raw_order, (str, bytes)):
            raise RuntimeError("Fixed decode plan missing iterable 'order'")
        try:
            order = tuple(str(x) for x in raw_order)
        except Exception as e:
            raise RuntimeError("Fixed decode plan 'order' is not iterable") from e
        if not order:
            raise RuntimeError("Fixed decode plan has empty 'order'")
        if len(set(order)) != len(order):
            raise RuntimeError("Fixed decode plan 'order' contains duplicate node ids")

        raw_dev_map = plan_map.get('device_by_node', None)
        try:
            device_by_node = {str(k): str(v) for k, v in dict(raw_dev_map or {}).items()}
        except Exception as e:
            raise RuntimeError("Fixed decode plan missing mapping 'device_by_node'") from e
        if not device_by_node:
            raise RuntimeError("Fixed decode plan has empty 'device_by_node'")

        return {
            'order': order,
            'device_by_node': device_by_node,
        }

    def _refresh_plan_from_schedule(dec_sched: List[ScheduledTask]) -> None:
        nonlocal current_plan
        exporter = getattr(sched, 'export_fixed_plan', None)
        if not callable(exporter):
            raise RuntimeError('Scheduler does not implement export_fixed_plan()')
        plan_obj = exporter(dec_sched)
        current_plan = _validate_fixed_plan_or_raise(plan_obj)

    try:
        for t in range(decode_len):
            cur_len = int(prefill_len + t)
            sched.set_seq_len(cur_len)
            _set_decode_ctx(t)

            if _need_exact_token(t):
                dec_sched = sched.schedule(graph, phase='decode')
                _refresh_plan_from_schedule(dec_sched)
                estimated = False
            else:
                plan_runner = getattr(sched, 'schedule_with_plan', None)
                if not callable(plan_runner):
                    raise RuntimeError(
                        'decode fixed-plan replay requested, but scheduler does not implement schedule_with_plan()'
                    )
                if current_plan is None:
                    raise RuntimeError('decode fixed-plan replay requested before any valid fixed plan was prepared')
                dec_sched = plan_runner(graph, phase='decode', plan=current_plan)
                estimated = True

            token_end = float(sched.makespan(dec_sched))
            step_time = max(0.0, float(token_end - global_end))
            global_end = float(token_end)

            steps_serialized.append({
                't': int(t),
                'seq_len': int(cur_len),
                'step_time': float(step_time),
                'estimated': bool(estimated),
                'schedule': (
                    _serialize_schedule(dec_sched, phase='decode', token_idx=t)
                    if _should_dump_schedule(t)
                    else None
                ),
            })
    finally:
        _clear_decode_ctx()

    return (float(global_end - prefill_end), steps_serialized)

def _make_scheduler(name: str, cluster: Cluster, cost: CostModel, label: PlanLabel, batch: int, seq_len: int, buffer: GlobalMemoryManager):
    """Factory for scheduler strategies used by evaluate-suite."""

    name = (name or 'heft').strip().lower()

    # Communication-aware HEFT (COMMAWARE-HEFT)
    if name == 'hefthint' :
        if HEFTCOMMAWAREScheduler is None:
            raise ImportError(
                "HEFTCOMMAWAREScheduler is not available. "
                "Please export it from scheduler.py (e.g., `from scheduler_heft import HEFTCOMMAWAREScheduler`)."
            )
        return HEFTCOMMAWAREScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    # Baseline HEFT
    if name in ('heft'):
        return HEFTScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    # Simple topo-order baseline
    if name in ('naive', 'topo', 'fifo', 'ready'):
        return NaiveTopoScheduler(cluster, cost, label, batch=batch, seq_len=seq_len, buffer=buffer)

    raise ValueError(f"Unknown scheduler strategy: {name}")

def mapping_diff_ratio(a: Dict[str, str], b: Dict[str, str]) -> float:
    if not a and (not b):
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    diff = sum((1 for k in keys if a.get(k) != b.get(k)))
    return diff / float(len(keys))

_LAYER_PREFIX_RE = re.compile(r"^L(?P<layer>\d+)_(?P<rest>.*)$", re.IGNORECASE)

def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'no', 'n', 'off', '', 'none', 'null'):
        return False
    return bool(default)


def _coerce_positive_int(value: Any, *, default: int = 0) -> int:
    try:
        iv = int(value)
    except Exception:
        try:
            iv = int(default)
        except Exception:
            iv = 0
    return iv if iv > 0 else 0


def _coerce_fraction(value: Any, *, default: float = 0.0) -> float:
    try:
        fv = float(value)
    except Exception:
        try:
            fv = float(default)
        except Exception:
            fv = 0.0
    if not math.isfinite(fv):
        try:
            fv = float(default)
        except Exception:
            fv = 0.0
    return float(min(1.0, max(0.0, fv)))


def _normalize_block_span_overrides(raw: Any) -> Dict[str, int]:
    """Normalize layer-span overrides such as {"W1": 8, "W2": 4}."""
    parsed: Any = raw
    if parsed is None:
        return {}
    if isinstance(parsed, str):
        s = str(parsed).strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = {}
            for item in s.split(','):
                token = str(item).strip()
                if not token or ':' not in token:
                    continue
                k, v = token.split(':', 1)
                parsed[str(k).strip()] = v
    if not isinstance(parsed, dict):
        return {}

    out: Dict[str, int] = {}
    for k, v in parsed.items():
        key = str(k or '').strip().upper().replace(' ', '')
        span = _coerce_positive_int(v, default=0)
        if key and span > 0:
            out[key] = int(span)
    return out


def _split_layer_prefixed_weight_id(wid: str) -> Tuple[int | None, str]:
    """Return (layer_idx, rest_name) for layer-scoped weight ids."""
    if not wid:
        return (None, "")
    s = str(wid)
    m = _LAYER_PREFIX_RE.match(s)
    if not m:
        return (None, s)
    try:
        layer_idx = int(m.group('layer'))
    except Exception:
        layer_idx = None
    return (layer_idx, m.group('rest') or "")


def _strip_weight_shard_suffix(name: str) -> str:
    """Strip trailing tensor-parallel shard suffix: ..._s0 / ..._S1."""
    s = str(name or "")
    return re.sub(r"_[sS]\d+$", "", s)


def _base_weight_family(name: str) -> str:
    """Return the coarse weight family name used for override lookup.

    Examples:
        W1_s0   -> W1
        E2_W1   -> W1
        WQ_s1   -> WQ
    """
    base = _strip_weight_shard_suffix(str(name or ""))
    su = str(base).upper()
    m = re.search(r"(WQ|WK|WV|WO|W1|W2|W3)(?:_|$)", su)
    if m:
        return str(m.group(1))
    return su


def _resolve_block_layer_span(
    weight_name: str,
    *,
    default_layer_span: int,
    layer_span_by_weight: Dict[str, int] | None,
) -> int:
    overrides = dict(layer_span_by_weight or {})
    if overrides:
        exact = str(weight_name or '').strip().upper().replace(' ', '')
        no_shard = _strip_weight_shard_suffix(exact).upper()
        family = _base_weight_family(weight_name)
        for key in (exact, no_shard, family):
            span = overrides.get(str(key), None)
            if span is not None:
                return _coerce_positive_int(span, default=0)
    return _coerce_positive_int(default_layer_span, default=0)


def _block_local_weight_name(wid: str, *, preserve_shards: bool = False) -> str:
    _layer_idx, rest = _split_layer_prefixed_weight_id(wid)
    base = rest if rest else str(wid or "")
    if not preserve_shards:
        base = _strip_weight_shard_suffix(base)
    return str(base)


def _weight_block_key(
    wid: str,
    *,
    layer_span: int = 0,
    layer_span_by_weight: Dict[str, int] | None = None,
    preserve_shards: bool = False,
) -> str:
    """Map a per-layer weight_id to a block key. """
    layer_idx, rest = _split_layer_prefixed_weight_id(wid)
    local_name = _block_local_weight_name(wid, preserve_shards=bool(preserve_shards))
    if layer_idx is None:
        return str(local_name)

    span = _resolve_block_layer_span(
        str(rest or local_name),
        default_layer_span=int(layer_span or 0),
        layer_span_by_weight=layer_span_by_weight,
    )
    if span <= 0:
        return str(local_name)

    lo = (int(layer_idx) // int(span)) * int(span)
    hi = int(lo) + int(span) - 1
    if int(span) == 1:
        return f"L{int(layer_idx):04d}_{local_name}"
    return f"L{int(lo):04d}-{int(hi):04d}_{local_name}"


def _build_weight_blocks(
    weight_ids: List[str],
    *,
    layer_span: int = 0,
    layer_span_by_weight: Dict[str, int] | None = None,
    preserve_shards: bool = False,
) -> Dict[str, List[str]]:
    """Return {block_key: [wid,...]} using optional layer-range grouping."""
    blocks: Dict[str, List[str]] = {}
    for wid in weight_ids:
        key = _weight_block_key(
            wid,
            layer_span=int(layer_span or 0),
            layer_span_by_weight=layer_span_by_weight,
            preserve_shards=bool(preserve_shards),
        )
        blocks.setdefault(str(key), []).append(str(wid))
    return blocks


def _normalize_reload_count_mode(value: Any) -> str:
    s = str(value or 'per_device').strip().lower().replace('-', '_').replace(' ', '_')
    if s in ('raw', 'sum', 'total'):
        return 'raw'
    if s in ('per_device', 'avg_per_device', 'normalized', 'normalize', 'fair'):
        return 'per_device'
    if s in ('soft_per_device', 'soft', 'alpha'):
        return 'soft_per_device'
    raise ValueError(
        f"Unknown format_reload_count_mode='{value}'. "
        f"Expected 'raw', 'per_device', or 'soft_per_device'."
    )


def _dominant_block_fmt(npu_cnt: float, pim_cnt: float, nd_margin: float) -> str:
    """Decide block format based on normalized reload pressure.

    nd_margin:
        A *relative* tolerance in [0,1]. If the NPU/PIM difference is within
        this band, we keep ND.
    """
    npu_cnt = float(npu_cnt or 0.0)
    pim_cnt = float(pim_cnt or 0.0)
    total = float(npu_cnt + pim_cnt)
    if total <= 0.0:
        return "ND"
    if abs(float(npu_cnt - pim_cnt)) <= float(max(0.0, nd_margin)) * float(total):
        return "ND"
    return "NZ" if npu_cnt > pim_cnt else "PIM-OPT"

def _sa_make_neighbor_map(base_map: Dict[str, str], weight_ids: List[str], flip_prob: float=0.15) -> Dict[str, str]:

    CAND = ('ND', 'NZ', 'PIM-OPT')
    if not weight_ids:
        return dict(base_map)
    out = dict(base_map)
    flips = 0
    for wid in weight_ids:
        if random.random() < max(0.0, min(1.0, flip_prob)):
            old = out.get(wid, base_map.get(wid, 'ND'))
            choices = [x for x in CAND if x != old] or ['ND']
            out[wid] = random.choice(choices)
            flips += 1
    if flips == 0:
        wid = random.choice(weight_ids)
        old = out.get(wid, base_map.get(wid, 'ND'))
        choices = [x for x in CAND if x != old] or ['ND']
        out[wid] = random.choice(choices)
    return out


def run(cfg: Dict):
    #--------------------------------------------
    # 0: init all hardware settings
    #--------------------------------------------
    result_dir = Path(cfg.get('result_dir') or _build_result_dir(cfg, './output/weight_suggestions'))
    result_dir.mkdir(parents=True, exist_ok=True)
    weight_format_path = Path(cfg.get('weight_format_json') or (result_dir / 'weight_storage_suggestion.json'))
    compare_path = Path(
        cfg.get('weight_format_compare_json')
        or weight_format_path.with_name(weight_format_path.stem + '_compare' + weight_format_path.suffix)
    )
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    pim_config_path = Path(cfg['pim_config_path'])
    gb_config_path = Path(cfg['gb_config_path'])
    ramulator_config_path = Path(cfg['ramulator_config_path'])
    prefill_len = int(cfg.get('prefill_len', 128))
    batch = int(cfg.get('batch', 1))
    graph, shape = build_graph(cfg)
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, 'dim', 128)),
        n_heads=int(getattr(shape, 'n_heads', 1)),
        n_kv_heads=int(getattr(shape, 'n_kv_heads', 1)),
        ffn_dim=int(getattr(shape, 'ffn_dim', 512)),
        seqlen=prefill_len,
    )
    sim_log_file = cfg.get('simulation_log_file', str(result_dir / 'pim_simulation.txt'))
    npu_backend = _normalize_npu_backend(cfg.get('npu_backend', None))
    if _cluster_type_count(cluster, 'npu') <= 0:
        npu_backend = None
    pim_fast_mode = bool(cfg.get('pim_fast_mode', False))
    cost = CostModel(
        cluster,
        dtype=cfg.get('dtype', 'fp16'),
        pim_config_path=pim_config_path,
        gb_config_path=gb_config_path,
        ramulator_config_path=ramulator_config_path,
        simulation_log_file=sim_log_file,
        debug_traces=False,
        model_dict=model_dict,
        npu_backend=npu_backend,
        pim_fast_mode=pim_fast_mode,
        tp_qkv=int(cfg.get('tp_qkv', 1) or 1),
        tp_ffn=int(cfg.get('tp_ffn', 1) or 1),
    )
    cost.logger.start_simulation()

    fmt_map: Dict[str, str] = {}
    best_total: float | None = None
    best_map: Dict[str, str] = {}
    best_pass: int = -1
    all_pass_records: List[Dict] = []
    buffer_mgr = GlobalMemoryManager()
    search_start_mode = 'ND'
    fixed_baseline_experiments = [
        {
            'experiment_id': 'pd_linear',
            'display_name': 'algo:PD + Linear',
            'algo': 'pd',
            'storage_fmt': 'ND',
            'runner': 'baseline',
        },
        {
            'experiment_id': 'pd_dual_copy',
            'display_name': 'algo:PD + DUAL',
            'algo': 'pd',
            'storage_fmt': 'DUAL',
            'runner': 'baseline',
        },
        {
            'experiment_id': 'hefthint_linear',
            'display_name': 'algo:hefthint + Linear',
            'algo': 'hefthint',
            'storage_fmt': 'ND',
            'runner': 'strategy',
        },
        {
            'experiment_id': 'hefthint_dual_copy',
            'display_name': 'algo:hefthint + DUAL',
            'algo': 'hefthint',
            'storage_fmt': 'DUAL',
            'runner': 'strategy',
        },
    ]

    # Choose scheduler class for the tuning run (default: HEFT).
    algo_raw = cfg.get('algo', 'heft')
    if isinstance(algo_raw, list):
        algo_name = str(algo_raw[0]) if algo_raw else 'heft'
    else:
        algo_name = str(algo_raw)
    algo_name = (algo_name.replace(',', ' ').split()[:1] or ['heft'])[0].strip().lower()
    SchedCls = HEFTScheduler
    if algo_name == 'hefthint':
        if HEFTCOMMAWAREScheduler is None:
            raise ImportError(
                "HEFTCOMMAWAREScheduler is not available. "
                "Please export it from scheduler.py.",
            )
        SchedCls = HEFTCOMMAWAREScheduler
    elif algo_name not in ('heft', 'heft+greedy', 'greedy', ''):
        _debug(f"[weight-suggest] Unknown algo '{algo_name}', fallback to HEFTScheduler")

    label = auto_select_kv_policy(
        strategy=algo_name,
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=graph,
        shape=shape,
    )

    sel = getattr(label, 'kv_policy_selected', 'unknown')
    sc = getattr(label, 'kv_policy_scores', {})
    msg = _fmt_kv_policy_scores(sc)
    if msg:
        _debug(str(f"[KV-SELECT] selected={sel} | {msg}"))
    else:
        _debug(str(f"[KV-SELECT] selected={sel}"))

    kv_place = _infer_kv_place_from_label(label)
    graph_kv = _apply_kv_place_constraints(graph, kv_place)

    # ------------------------------------------------------------
    # 1: block-CD + 2-layer BFS/beam search
    # ------------------------------------------------------------
    legacy_outer_max_cfg = cfg.get('format_outer_max_iters', cfg.get('outer_max_iters', None))
    block_change_percent_cfg = cfg.get('format_block_change_percent', None)
    if block_change_percent_cfg is None:
        try:
            legacy_outer_max = int(legacy_outer_max_cfg or 0)
        except Exception:
            legacy_outer_max = 0
        if legacy_outer_max > 0:
            block_change_percent = float(1.0 / float(legacy_outer_max))
        else:
            block_change_percent = 0.20
    else:
        block_change_percent = _coerce_fraction(block_change_percent_cfg, default=0.20)
        if block_change_percent <= 0.0:
            block_change_percent = 0.20

    inner_max_blocks = int(cfg.get('format_inner_max_blocks', 0) or 0)  # 0 => no cap
    inner_improve_eps = float(cfg.get('format_inner_improve_eps', 1e-6) or 0.0)
    outer_stop_eps = float(cfg.get('format_outer_stop_eps', 0.0) or 0.0)
    block_layer_span = int(cfg.get('format_block_layer_span', 8) or 0)
    reload_count_mode_cfg = cfg.get('format_reload_count_mode', None)
    if reload_count_mode_cfg is None and ('format_normalize_reload_by_device_count' in cfg):
        reload_count_mode_cfg = 'per_device' if bool(cfg.get('format_normalize_reload_by_device_count')) else 'raw'
    reload_count_alpha = float(cfg.get('format_reload_device_count_alpha', 1.0) or 1.0)
    reload_count_mode = _normalize_reload_count_mode(reload_count_mode_cfg)
    type_device_counts = {
        'npu': max(1, _cluster_type_count(cluster, 'npu')),
        'pim': max(1, _cluster_type_count(cluster, 'pim')),
    }

    # Stable blocks built from model graph weight ids.
    all_wids = sorted({str(n.weight_id) for n in graph.nodes.values() if getattr(n, 'weight_id', None)})
    blocks = _build_weight_blocks(all_wids, layer_span=block_layer_span)
    outer_max = max(1, int(math.ceil(1.0 / float(block_change_percent))))
    max_outer_block_changes = 0
    if blocks:
        max_outer_block_changes = min(
            len(blocks),
            max(1, int(math.ceil(float(block_change_percent) * float(len(blocks))))),
        )

    _debug(
        f"[AL] init: weights={len(all_wids)} blocks={len(blocks)} "
        f"block_change_percent={block_change_percent:.3f} outer_max={outer_max} outer_topk={max_outer_block_changes} "
        f"inner_max_blocks={('inf' if not inner_max_blocks else int(inner_max_blocks))} "
        f"inner_eps={inner_improve_eps:g} outer_stop_eps={outer_stop_eps:g} "
        f"block_layer_span={block_layer_span} reload_count_mode={reload_count_mode} "
        f"device_counts(npu={type_device_counts['npu']}, pim={type_device_counts['pim']}) "
        f"search_start_mode={search_start_mode} baseline_experiments="
        f"{[spec['experiment_id'] for spec in fixed_baseline_experiments]}"
    )

    def _normalize_wlc(wstats: Dict) -> Dict[str, Dict[str, int]]:
        raw = (wstats or {}).get('weight_load_counts', {}) or {}
        out: Dict[str, Dict[str, int]] = {}
        for wid, cnts in raw.items():
            try:
                out[str(wid)] = {str(k): int(v) for k, v in (cnts or {}).items()}
            except Exception:
                out[str(wid)] = {}
        return out

    def _normalize_reload_count(dev_type: str, raw_cnt: int | float) -> float:
        value = float(raw_cnt or 0.0)

        if reload_count_mode == 'per_device':
            denom = float(max(1, int(type_device_counts.get(str(dev_type), 1) or 1)))
            return value / denom

        if reload_count_mode == 'soft_per_device':
            denom = float(max(1, int(type_device_counts.get(str(dev_type), 1) or 1)))
            alpha = max(0.0, float(reload_count_alpha))
            return value / (denom ** alpha)

        return value

    def _block_reload_counts(wlc: Dict[str, Dict[str, int]]) -> Dict[str, Tuple[float, float]]:
        """Return {block_key: (npu_pressure, pim_pressure)} aggregated across blocks."""
        out: Dict[str, Tuple[float, float]] = {}
        for bkey, wids in blocks.items():
            npu = 0.0
            pim = 0.0
            for w in wids:
                c = wlc.get(str(w), {}) or {}
                npu += _normalize_reload_count('npu', c.get('npu', 0) or 0)
                pim += _normalize_reload_count('pim', c.get('pim', 0) or 0)
            out[str(bkey)] = (float(npu), float(pim))
        return out

    def _apply_block_fmt(map_in: Dict[str, str], bkey: str, fmt: str) -> Dict[str, str]:
        """
        Set the storage format of a given “block” uniformly to fmt, and return a new format map (without modifying the original map)
        """
        out = dict(map_in or {})
        wids = blocks.get(str(bkey), [])
        for w in wids:
            if fmt == 'ND':
                out.pop(str(w), None)
            else:
                out[str(w)] = str(fmt)
        return out

    def _current_block_fmt(map_in: Dict[str, str], bkey: str) -> str:
        wids = blocks.get(str(bkey), [])
        if not wids:
            return 'ND'
        return str((map_in or {}).get(str(wids[0]), 'ND'))

    def _map_stats(map_in: Dict[str, str]) -> Dict[str, Any]:
        return _weight_map_summary(all_wids, map_in)

    def _effective_counts_for_cost(raw_counts: Dict[str, int | float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for dev_type, cnt in dict(raw_counts or {}).items():
            eff = float(_normalize_reload_count(str(dev_type), float(cnt or 0.0)))
            if eff > 0.0:
                out[str(dev_type)] = float(eff)
        return out

    def _rank_outer_block_updates(
        map_in: Dict[str, str],
        wstats: Dict,
        sched_eval: Any,
        *,
        only_if_current_nd: bool,
    ) -> List[Dict[str, Any]]:
        """Rank outer-step block flips by estimated cost reduction."""
        wlc = _normalize_wlc(wstats)
        try:
            chain_hits = {
                str(k): float(v or 0.0)
                for k, v in ((wstats or {}).get('weight_chain_hits', {}) or {}).items()
            }
        except Exception:
            chain_hits = {}
        max_chain_hits = float(max(chain_hits.values(), default=0.0)) if chain_hits else 0.0

        ranked: List[Dict[str, Any]] = []
        eps = 1e-12
        for bkey, wids in blocks.items():
            cur_fmt = _current_block_fmt(map_in, bkey)
            if only_if_current_nd and cur_fmt != 'ND':
                continue

            cost_by_fmt: Dict[str, float] = {'ND': 0.0, 'NZ': 0.0, 'PIM-OPT': 0.0}
            for wid in wids:
                counts_eff = _effective_counts_for_cost(wlc.get(str(wid), {}) or {})
                if not counts_eff:
                    continue
                for fmt in ('ND', 'NZ', 'PIM-OPT'):
                    cost_by_fmt[str(fmt)] += float(
                        sched_eval._estimate_weight_host_to_device_cost(
                            str(wid),
                            counts_eff,
                            str(fmt),
                            lookahead_beta=0.0,
                            max_chain_hits=max_chain_hits,
                            chain_hits=chain_hits,
                        )
                    )

            cur_cost = float(cost_by_fmt.get(cur_fmt, cost_by_fmt.get('ND', 0.0)))
            best_fmt = str(cur_fmt)
            best_cost = float(cur_cost)
            # Keep the current format on ties; for ND blocks this means we only
            # promote to NZ/PIM-OPT when the surrogate cost is strictly smaller.
            for fmt in ('ND', 'NZ', 'PIM-OPT'):
                trial_cost = float(cost_by_fmt.get(fmt, 0.0))
                if trial_cost + eps < best_cost:
                    best_cost = float(trial_cost)
                    best_fmt = str(fmt)

            gain = float(cur_cost - best_cost)
            if best_fmt != cur_fmt and gain > eps:
                ranked.append({
                    'block': str(bkey),
                    'cur_fmt': str(cur_fmt),
                    'next_fmt': str(best_fmt),
                    'gain': float(gain),
                    'cur_cost': float(cur_cost),
                    'next_cost': float(best_cost),
                    'costs': {str(k): float(v) for k, v in cost_by_fmt.items()},
                })

        ranked.sort(
            key=lambda item: (
                -float(item.get('gain', 0.0) or 0.0),
                -float(item.get('cur_cost', 0.0) or 0.0),
                str(item.get('block', '')),
            )
        )
        return ranked

    def _apply_ranked_outer_block_updates(
        map_in: Dict[str, str],
        ranked_updates: List[Dict[str, Any]],
        *,
        max_changes: int,
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        out = dict(map_in or {})
        applied: List[Dict[str, Any]] = []
        limit = max(0, int(max_changes))
        if limit <= 0 or not ranked_updates:
            return out, applied

        for item in ranked_updates[:limit]:
            out = _apply_block_fmt(out, str(item.get('block', '')), str(item.get('next_fmt', 'ND')))
            applied.append(dict(item))
        return out, applied

    def _evaluate_map(fmt_map_eval: Dict[str, str], *, tag: str) -> Tuple[float, float, float, Any, Any, Dict, Any]:
        """Run prefill+decode simulation under a given host format map."""
        sched = SchedCls(cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
        sched.reset_state()
        sched.set_storage_format_map(fmt_map_eval)
        graph_eval = graph_kv
        prefill_time, prefill_ser = simulate_prefill(sched, cfg, graph_eval)
        decode_time, decode_ser = simulate_decode_progressive(sched, cfg, graph_eval, prefill_end=prefill_time)
        total_time = float(prefill_time + decode_time)
        wstats = sched.export_weight_stats()
        return (total_time, float(prefill_time), float(decode_time), prefill_ser, decode_ser, wstats, sched)

    def _record(pass_id: int, total: float, prefill_t: float, decode_t: float, prefill_ser: Any, decode_ser: Any, fm: Dict[str, str], wstats: Dict, *, note: str):
        all_pass_records.append({
            'search_format': str(search_start_mode),
            'role': 'search',
            'pass': int(pass_id),
            'note': str(note),
            'times': {'prefill': float(prefill_t), 'decode': float(decode_t), 'total': float(total)},
            'schedules': {'prefill': prefill_ser, 'decode_steps': decode_ser},
            'formats': dict(fm or {}),
            'format_summary': _map_stats(fm),
            'weights': dict(wstats or {}),
            'pim_trace': list(getattr(getattr(cost, 'logger', None), 'pim_trace', []) or []),
        })

    def _inner_sweep(
        map_in: Dict[str, str],
        base_total: float,
        base_prefill: float,
        base_decode: float,
        base_prefill_ser: Any,
        base_decode_ser: Any,
        base_wstats: Dict,
        base_sched: Any,
        *,
        sweep_id: int,
    ) -> Tuple[Dict[str, str], float, float, float, Any, Any, Dict, Any]:
        """Inner sweep: try per-block format flips (NZ->ND->PIM-OPT)."""
        cur_map = dict(map_in or {})
        cur_total = float(base_total)
        cur_prefill = float(base_prefill)
        cur_decode = float(base_decode)
        cur_prefill_ser = base_prefill_ser
        cur_decode_ser = base_decode_ser
        cur_wstats = dict(base_wstats or {})
        cur_sched = base_sched

        wlc = _normalize_wlc(cur_wstats)
        blk_cnt = _block_reload_counts(wlc)

        # Candidate blocks: stored as NZ but used mostly on PIM, or vice versa.
        candidates: List[Tuple[float, float, str]] = []  # (severity, total_cnt, bkey)
        for bkey, (npu, pim) in blk_cnt.items():
            fmt = _current_block_fmt(cur_map, bkey)
            if fmt == 'NZ' and pim > npu:
                candidates.append((float(pim - npu), float(npu + pim), str(bkey)))
            elif fmt == 'PIM-OPT' and npu > pim:
                candidates.append((float(npu - pim), float(npu + pim), str(bkey)))

        # Try the most "wrong" blocks first.
        candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

        _debug(
            f"[AL] inner{sweep_id}: start cur_total={cur_total:.6f}s "
            f"candidates={len(candidates)} max_blocks={('inf' if not inner_max_blocks else int(inner_max_blocks))} "
            f"eps={inner_improve_eps:g}"
        )

        if not candidates:
            _debug(f"[AL] inner{sweep_id}: no candidates; skip.")
            return (cur_map, cur_total, cur_prefill, cur_decode, cur_prefill_ser, cur_decode_ser, cur_wstats, cur_sched)

        tried = 0
        accepted_cnt = 0
        for _, _, bkey in candidates:
            if inner_max_blocks and tried >= inner_max_blocks:
                break
            tried += 1

            fmt0 = _current_block_fmt(cur_map, bkey)
            # Two-layer BFS along the line: NZ -> ND -> PIM-OPT (or reverse)
            if fmt0 == 'NZ':
                fmt_chain = ['ND', 'PIM-OPT']
            elif fmt0 == 'PIM-OPT':
                fmt_chain = ['ND', 'NZ']
            else:
                continue

            try:
                _npu_r, _pim_r = blk_cnt.get(str(bkey), (0, 0))
            except Exception:
                _npu_r, _pim_r = 0, 0
            _debug(
                f"[AL] inner{sweep_id}: try#{tried}/{len(candidates)} "
                f"block={bkey} fmt0={fmt0} reload(npu={float(_npu_r):.3f}, pim={float(_pim_r):.3f}) "
                f"chain={fmt_chain} cur_total={cur_total:.6f}s"
            )

            accepted = False
            best_trial: float | None = None
            for fmt1 in fmt_chain:
                cand_map = _apply_block_fmt(cur_map, bkey, fmt1)
                total_time, prefill_time, decode_time, prefill_ser, decode_ser, wstats, sched_eval = _evaluate_map(
                    cand_map,
                    tag=f"inner{sweep_id}_blk_{bkey}_{fmt0}_to_{fmt1}",
                )
                try:
                    best_trial = float(total_time) if best_trial is None else min(float(best_trial), float(total_time))
                except Exception:
                    pass
                if float(total_time) + float(inner_improve_eps) < float(cur_total):
                    old_total = float(cur_total)
                    cur_map = cand_map
                    cur_total = float(total_time)
                    cur_prefill = float(prefill_time)
                    cur_decode = float(decode_time)
                    cur_prefill_ser = prefill_ser
                    cur_decode_ser = decode_ser
                    cur_wstats = dict(wstats or {})
                    cur_sched = sched_eval
                    accepted = True
                    accepted_cnt += 1
                    _debug(
                        f"[AL] inner{sweep_id}: ACCEPT block={bkey} {fmt0}->{fmt1} "
                        f"total {old_total:.6f}s -> {cur_total:.6f}s (delta={cur_total - old_total:+.6f}s)"
                    )
                    break
                else:
                    # Keep the logs light: only show a few early rejects.
                    if tried <= 3:
                        _debug(
                            f"[AL] inner{sweep_id}: reject block={bkey} {fmt0}->{fmt1} "
                            f"trial_total={float(total_time):.6f}s (cur={cur_total:.6f}s)"
                        )
            if accepted:
                # Refresh counts after an accepted change (keeps subsequent tests meaningful).
                wlc = _normalize_wlc(cur_wstats)
                blk_cnt = _block_reload_counts(wlc)
            else:
                if best_trial is not None and tried <= 3:
                    _debug(
                        f"[AL] inner{sweep_id}: best_trial_for_block={float(best_trial):.6f}s (no accept; cur={cur_total:.6f}s)"
                    )

        _debug(
            f"[AL] inner{sweep_id}: done tried={tried} accepted={accepted_cnt} final_total={cur_total:.6f}s"
        )
        return (cur_map, cur_total, cur_prefill, cur_decode, cur_prefill_ser, cur_decode_ser, cur_wstats, cur_sched)

    # -------------------------------
    # ND is the only real search start.
    # -------------------------------
    fmt_map = {}
    _debug(f"[AL][{search_start_mode}] outer0: start (all weights ND)")
    total_time0, prefill_time0, decode_time0, prefill_time0_ser, decode_time0_ser, wst0, sched0 = _evaluate_map(
        fmt_map,
        tag='outer0_all_nd',
    )
    _debug(
        f"[AL][{search_start_mode}] outer0: done total={float(total_time0):.6f}s prefill={float(prefill_time0):.6f}s decode={float(decode_time0):.6f}s"
    )
    _record(0, total_time0, prefill_time0, decode_time0, prefill_time0_ser, decode_time0_ser, fmt_map, wst0, note='outer0_all_nd')

    best_total = float(total_time0)
    best_map = dict(fmt_map)
    best_pass = 0

    prev_outer_total = float(total_time0)
    prev_outer_map = dict(fmt_map)
    prev_wstats = dict(wst0)
    prev_sched = sched0

    # -------------------------------
    # outer iterations: cost-ranked top-K block changes + unchanged inner sweep
    # -------------------------------
    for outer_it in range(1, max(1, outer_max) + 1):
        ranked_updates = _rank_outer_block_updates(
            prev_outer_map,
            prev_wstats,
            prev_sched,
            only_if_current_nd=True,
        )

        if not ranked_updates:
            _debug(
                f"[AL][{search_start_mode}] outer{outer_it}: no outer blocks need modification; stop."
            )
            break

        cand_map, applied_updates = _apply_ranked_outer_block_updates(
            prev_outer_map,
            ranked_updates,
            max_changes=max_outer_block_changes,
        )

        diff_ratio = mapping_diff_ratio(prev_outer_map, cand_map)
        if diff_ratio == 0.0 or not applied_updates:
            _debug(
                f"[AL][{search_start_mode}] outer{outer_it}: ranked candidates exist but no block was applied; stop."
            )
            break

        try:
            keys = set(prev_outer_map.keys()) | set(cand_map.keys())
            changed = sum((1 for k in keys if prev_outer_map.get(k) != cand_map.get(k)))
        except Exception:
            changed = -1

        top0 = applied_updates[0]
        _debug(
            f"[AL][{search_start_mode}] outer{outer_it}: apply blocks={len(applied_updates)}/{max_outer_block_changes} "
            f"changed_weights={changed} diff_ratio={diff_ratio:.3f} prev_total={float(prev_outer_total):.6f}s "
            f"top1={top0.get('block')} {top0.get('cur_fmt')}->{top0.get('next_fmt')} gain={float(top0.get('gain', 0.0)):.6e}"
        )

        total_time_k, prefill_time_k, decode_time_k, prefill_time_k_ser, decode_time_k_ser, wst_k, sched_k = _evaluate_map(
            cand_map,
            tag=f'outer{outer_it}_baseline',
        )
        _debug(
            f"[AL][{search_start_mode}] outer{outer_it}: baseline total={float(total_time_k):.6f}s prefill={float(prefill_time_k):.6f}s decode={float(decode_time_k):.6f}s"
        )
        outer_k_base_total = float(total_time_k)
        outer_k_base_map = dict(cand_map)
        (
            cand_map,
            total_k,
            prefill_time_k,
            decode_time_k,
            prefill_time_k_ser,
            decode_time_k_ser,
            wst_k,
            sched_k,
        ) = _inner_sweep(
            cand_map,
            total_time_k,
            prefill_time_k,
            decode_time_k,
            prefill_time_k_ser,
            decode_time_k_ser,
            wst_k,
            sched_k,
            sweep_id=outer_it,
        )

        # IMPORTANT: use the post-inner-sweep total for decisions/records.
        total_time_k = float(total_k)
        _debug(
            f"[AL][{search_start_mode}] outer{outer_it}: after inner total={float(total_time_k):.6f}s "
            f"(delta={float(total_time_k) - outer_k_base_total:+.6f}s, map_diff={mapping_diff_ratio(outer_k_base_map, cand_map):.3f})"
        )

        # Revert + stop on regression. If we only revert without stopping, the
        # next outer pass would simply re-propose the same top-ranked blocks.
        if float(total_time_k) > float(prev_outer_total) + float(outer_stop_eps):
            _debug(
                f"[AL][{search_start_mode}] outer{outer_it}: total {total_time_k:.6f}s is worse than prev {prev_outer_total:.6f}s; revert and stop."
            )
            break

        # Accept this outer iteration.
        prev_outer_total = float(total_time_k)
        prev_outer_map = dict(cand_map)
        prev_wstats = dict(wst_k)
        prev_sched = sched_k
        _record(
            outer_it,
            total_time_k,
            prefill_time_k,
            decode_time_k,
            prefill_time_k_ser,
            decode_time_k_ser,
            cand_map,
            wst_k,
            note=f'outer{outer_it}_after_inner',
        )

        if best_total is None or float(total_time_k) < float(best_total):
            best_total = float(total_time_k)
            best_map = dict(cand_map)
            best_pass = int(outer_it)

    best_rec: Dict[str, Any]
    improvements: List[Dict[str, Any]] = []
    if all_pass_records:
        best_idx = min(range(len(all_pass_records)), key=lambda i: float(all_pass_records[i]['times']['total']))
        best_rec = dict(all_pass_records[best_idx])
        best_total_rec = float(best_rec['times']['total'])
        best_map = dict(best_rec.get('formats') or {})
        best_pass = int(best_rec.get('pass', best_pass))
        for rec in all_pass_records:
            total_time = float(rec['times']['total'])
            delta = float(total_time - best_total_rec)
            pct = delta / total_time * 100.0 if total_time > 0 else 0.0
            improvements.append({
                'pass': rec.get('pass', -1),
                'total_time': float(total_time),
                'delta_seconds_vs_best': float(delta),
                'delta_percent_vs_that_pass': float(pct),
            })
    else:
        best_total_rec = float(best_total or 0.0)
        best_rec = {
            'search_format': str(search_start_mode),
            'role': 'search',
            'pass': int(best_pass),
            'note': 'outer0_all_nd',
            'times': {'prefill': float(prefill_time0), 'decode': float(decode_time0), 'total': float(best_total_rec)},
            'schedules': {'prefill': prefill_time0_ser, 'decode_steps': decode_time0_ser},
            'formats': dict(best_map or {}),
            'format_summary': _map_stats(best_map),
            'weights': dict(wst0 or {}),
            'pim_trace': list(getattr(getattr(cost, 'logger', None), 'pim_trace', []) or []),
        }
        improvements.append({
            'pass': int(best_pass),
            'total_time': float(best_total_rec),
            'delta_seconds_vs_best': 0.0,
            'delta_percent_vs_that_pass': 0.0,
        })

    # Persist best ND-search map and reports.
    weight_format_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weight_format_path, 'w', encoding='utf-8') as f:
        json.dump(best_map or {}, f, indent=2, sort_keys=True)
    _debug(str(f'[INFO] Best weight storage map (ND search) saved to: {weight_format_path}'))

    full_map = {str(w): str((best_map or {}).get(str(w), 'ND')) for w in all_wids}
    full_path = weight_format_path.with_name(weight_format_path.stem + "_full" + weight_format_path.suffix)
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(full_map, f, indent=2, sort_keys=True)
    _debug(str(f'[INFO] Full weight storage map (ND search) saved to: {full_path}'))

    # ------------------------------------------------------------
    # 2: fixed baseline experiments
    # ------------------------------------------------------------
    nd_initial_rec = dict(all_pass_records[0]) if all_pass_records else dict(best_rec)
    nd_initial_total = float((nd_initial_rec.get('times') or {}).get('total', 0.0) or 0.0)
    compare_rows: List[Dict[str, Any]] = [
        {
            'experiment_id': 'search_nd_tuned',
            'display_name': f"algo:{str(algo_name or 'heft')} + search(best map)",
            'format': 'ND',
            'algo': str(algo_name or 'heft'),
            'storage_mode': 'AL-search(best map)',
            'storage_format': 'MIXED',
            'role': 'search',
            'search_executed': True,
            'comparison_only': False,
            'initial_times': dict(nd_initial_rec.get('times') or {}),
            'initial_total_s': float(nd_initial_total),
            'best_total_s': float(best_total_rec),
            'delta_vs_initial_s': float(best_total_rec - nd_initial_total),
            'delta_vs_initial_pct': float(((best_total_rec - nd_initial_total) / nd_initial_total * 100.0) if nd_initial_total > 0 else 0.0),
            'best_pass': int(best_rec.get('pass', best_pass)),
            'best_format_summary': dict(best_rec.get('format_summary') or _map_stats(best_map)),
            'best_weight_format_json': str(weight_format_path),
            'best_weight_format_full_json': str(full_path),
        }
    ]

    baseline_experiment_ids = [str(spec.get('experiment_id', '')) for spec in fixed_baseline_experiments]
    for spec in fixed_baseline_experiments:
        exp_id = str(spec.get('experiment_id', '') or '')
        runner = str(spec.get('runner', 'strategy') or 'strategy').strip().lower()
        algo_for_row = str(spec.get('algo', '') or '')
        storage_fmt = _normalize_weight_storage_fmt(spec.get('storage_fmt', 'ND'))
        storage_mode_name = _storage_mode_display_name(storage_fmt)

        _debug(
            f"[BASELINE][{exp_id}] start algo={algo_for_row} storage={storage_mode_name}"
        )

        if runner == 'baseline':
            result = _eval_one_baseline(
                cfg,
                algo_for_row,
                shared_graph=graph,
                shared_shape=shape,
                uniform_weight_storage_fmt=storage_fmt,
                artifact_tag=exp_id,
            )
        else:
            result = _run_strategy_once(
                algo_for_row,
                cfg,
                shared_graph=graph,
                shared_shape=shape,
                uniform_weight_storage_fmt=storage_fmt,
                artifact_tag=exp_id,
            )

        init_times = {
            'prefill': float(result.get('prefill_time_s', 0.0) or 0.0),
            'decode': float(result.get('decode_time_s', 0.0) or 0.0),
            'total': float(result.get('total_time_s', 0.0) or 0.0),
        }
        _debug(
            f"[BASELINE][{exp_id}] done total={float(init_times['total']):.6f}s "
            f"prefill={float(init_times['prefill']):.6f}s decode={float(init_times['decode']):.6f}s"
        )
        compare_rows.append({
            'experiment_id': exp_id,
            'display_name': str(spec.get('display_name', exp_id) or exp_id),
            'format': str(storage_fmt),
            'algo': str(algo_for_row),
            'storage_mode': str(result.get('weight_storage_mode', storage_mode_name) or storage_mode_name),
            'storage_format': str(result.get('weight_storage_format', storage_fmt) or storage_fmt),
            'role': 'baseline',
            'search_executed': False,
            'comparison_only': False,
            'initial_times': dict(init_times),
            'initial_total_s': float(init_times['total']),
            'best_total_s': float(init_times['total']),
            'delta_vs_initial_s': 0.0,
            'delta_vs_initial_pct': 0.0,
            'best_pass': 0,
            'best_format_summary': dict(result.get('weight_storage_map_summary') or _map_stats(_build_uniform_weight_storage_map(graph, storage_fmt))),
            'pim_strategy': result.get('pim_strategy'),
            'pim_strategy_scores': result.get('pim_strategy_scores'),
        })

    comparison_payload = {
        'config': {
            'model_family': cfg.get('model_family'),
            'model_variant': cfg.get('model_variant'),
            'dtype': cfg.get('dtype'),
            'batch': cfg.get('batch'),
            'prefill_len': cfg.get('prefill_len'),
            'decode_len': cfg.get('decode_len'),
            'search_format': str(search_start_mode),
            'compare_only_formats': [],
            'baseline_experiment_ids': list(baseline_experiment_ids),
            'format_block_change_percent': float(block_change_percent),
            'format_outer_max_iters': int(outer_max),
            'format_inner_max_blocks': cfg.get('format_inner_max_blocks', 0),
            'format_nd_margin_init': cfg.get('format_nd_margin_init', 0.60),
            'format_nd_margin_decay': cfg.get('format_nd_margin_decay', 0.85),
            'format_nd_margin_min': cfg.get('format_nd_margin_min', 0.05),
            'format_inner_improve_eps': cfg.get('format_inner_improve_eps', 1e-6),
            'format_outer_stop_eps': cfg.get('format_outer_stop_eps', 0.0),
            'weight_local_load_overlap_ratio': cfg.get('weight_local_load_overlap_ratio', None),
        },
        'rows': compare_rows,
        'search_format': str(search_start_mode),
        'compare_only_formats': [],
        'baseline_experiment_ids': list(baseline_experiment_ids),
        'best_pass': int(best_rec.get('pass', best_pass)),
        'best_total_s': float(best_total_rec),
        'best_weight_format_json': str(weight_format_path),
        'best_weight_format_full_json': str(full_path),
    }
    compare_path.parent.mkdir(parents=True, exist_ok=True)
    with open(compare_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_payload, f, ensure_ascii=False, indent=2)
    _debug(str(f'[REPORT] Weight-format comparison saved to: {compare_path}'))

    all_path = Path(cfg.get('all_passes_json', ALL_PASSES_RESULT_PATH))
    best_path = Path(cfg.get('best_summary_json', BEST_PASS_SUMMARY_PATH))
    all_path.parent.mkdir(parents=True, exist_ok=True)
    with open(all_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'config': {
                    'model_family': cfg.get('model_family'),
                    'model_variant': cfg.get('model_variant'),
                    'dtype': cfg.get('dtype'),
                    'batch': cfg.get('batch'),
                    'prefill_len': cfg.get('prefill_len'),
                    'decode_len': cfg.get('decode_len'),
                    'search_format': str(search_start_mode),
                    'compare_only_formats': [],
                    'baseline_experiment_ids': list(baseline_experiment_ids),
                    'weight_local_load_overlap_ratio': cfg.get('weight_local_load_overlap_ratio', None),
                },
                'weight_format_comparison': comparison_payload,
                'passes': all_pass_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _debug(str(f'[REPORT] All passes (ND search only) saved to: {all_path}'))

    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'search_format': str(search_start_mode),
                'best_pass': int(best_rec.get('pass', best_pass)),
                'best_times': best_rec.get('times', {}),
                'best_formats': best_rec.get('formats', {}),
                'best_format_summary': best_rec.get('format_summary', {}),
                'best_weights': best_rec.get('weights', {}),
                'prefill_schedule': best_rec.get('schedules', {}).get('prefill'),
                'decode_steps': best_rec.get('schedules', {}).get('decode_steps'),
                'improvements_vs_each_pass': improvements,
                'weight_format_comparison': compare_rows,
                'baseline_experiment_ids': list(baseline_experiment_ids),
                'weight_local_load_overlap_ratio': cfg.get('weight_local_load_overlap_ratio', None),
                'best_weight_format_json': str(weight_format_path),
                'best_weight_format_full_json': str(full_path),
                'weight_format_compare_json': str(compare_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _debug(str(f'[REPORT] Best pass summary (ND search only) saved to: {best_path}'))

    print("\n=== Weight-Suggest Format Comparison ===")
    header = (
        f"{'Experiment':<40} {'Role':<10} {'Outer0(s)':>12} {'Best(s)':>12} {'Delta(s)':>12} "
        f"{'BestPass':>9} {'ND':>8} {'NZ':>8} {'PIM':>8} {'DUAL':>8}"
    )
    print(header)
    print('-' * len(header))
    for row in compare_rows:
        counts = dict((row.get('best_format_summary') or {}).get('counts', {}) or {})
        print(
            f"{str(row.get('display_name', row.get('experiment_id', row.get('format', '')))):<40} "
            f"{str(row.get('role', '')):<10} "
            f"{float(row.get('initial_total_s', 0.0)):>12.4f} "
            f"{float(row.get('best_total_s', 0.0)):>12.4f} "
            f"{float(row.get('delta_vs_initial_s', 0.0)):>12.4f} "
            f"{int(row.get('best_pass', -1)):>9d} "
            f"{int(counts.get('ND', 0)):>8d} "
            f"{int(counts.get('NZ', 0)):>8d} "
            f"{int(counts.get('PIM-OPT', 0)):>8d} "
            f"{int(counts.get('DUAL', 0)):>8d}"
        )
    print(
        f"[weight-suggest] search_format={str(search_start_mode)} "
        f"weight_local_load_overlap_ratio={cfg.get('weight_local_load_overlap_ratio', None)} "
        f"best_total={float(best_total_rec):.6f}s"
    )

    # AL mode terminates here (skip legacy multi-pass loop below).
    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    return

# ===== Baseline helpers (inlined) =====
_ATTENTION_KEYS = {
    'q','k','v','o','qk','sv','softmax','attn','attention','attn_softmax','q_proj','k_proj','v_proj','wo_proj'
}

def _is_attention_node(n: TaskNode) -> bool:
    name = (n.name or '').lower()
    if name in _ATTENTION_KEYS:
        return True
    return any(k in name for k in _ATTENTION_KEYS)

def _clone_graph(g: TaskGraph) -> TaskGraph:
    """Deep copy TaskGraph nodes + edges, to safely override `allowed`."""
    new_g = TaskGraph()
    for _, n in g.nodes.items():
        new_n = TaskNode(
            id=n.id, name=n.name,
            flops=n.flops, bytes_read=n.bytes_read, bytes_write=n.bytes_write,
            weight_id=n.weight_id, weight_size=n.weight_size,
            allowed=dict(n.allowed) if isinstance(n.allowed, dict) else {},
            attrs=dict(n.attrs) if isinstance(n.attrs, dict) else {},
        )
        new_g.add_node(new_n)
    succ = getattr(g, 'succ', None)
    if isinstance(succ, dict) and succ:
        for u, nbrs in succ.items():
            for v in nbrs:
                new_g.add_edge(u, v)
    else:
        try:
            _ = g.topological()
        except Exception:
            pass
    return new_g

# ---- Hardware capability helpers ----
def _cluster_type_count(cluster: Cluster, dev_type: str) -> int:
    """Safe device-type counter (returns 0 on any error)."""
    try:
        return int(len(cluster.devices_by_type(dev_type)) or 0)
    except Exception:
        return 0

def _fallback_npu_to_cpu_if_needed(g: TaskGraph, cluster: Cluster, *, verbose: bool=False) -> TaskGraph:
    npu_cnt = _cluster_type_count(cluster, 'npu')
    cpu_cnt = _cluster_type_count(cluster, 'cpu')
    if npu_cnt > 0 or cpu_cnt <= 0:
        return g

    touched = 0
    total = 0
    for _, n in getattr(g, 'nodes', {}).items():
        total += 1
        try:
            allowed = getattr(n, 'allowed', None)
            if not isinstance(allowed, dict):
                allowed = {}
                setattr(n, 'allowed', allowed)
        except Exception:
            continue

        # If no NPU exists, make it explicit that NPU is not available.
        npu_allowed = bool(allowed.get('npu', False))
        allowed['npu'] = False

        # If this op was intended for NPU, move it to CPU.
        if npu_allowed:
            allowed['cpu'] = True
            touched += 1

        # Final safety: ensure at least one present device type is allowed.
        try:
            cpu_ok = bool(allowed.get('cpu', True))
            pim_ok = bool(allowed.get('pim', True))
        except Exception:
            cpu_ok, pim_ok = True, True
        if not (cpu_ok or pim_ok):
            allowed['cpu'] = True
            touched += 1

    if verbose or touched > 0:
        logger.warning('[HW] No NPU detected; falling back NPU ops to CPU (touched %d/%d nodes).', touched, total)
    return g

def _fallback_pim_to_cpu_if_needed(g: TaskGraph, cluster: Cluster, *, verbose: bool=False) -> TaskGraph:
    pim_cnt = _cluster_type_count(cluster, 'pim')
    cpu_cnt = _cluster_type_count(cluster, 'cpu')
    npu_cnt = _cluster_type_count(cluster, 'npu')
    if pim_cnt > 0 or cpu_cnt <= 0:
        return g

    touched = 0
    total = 0
    for _, n in getattr(g, 'nodes', {}).items():
        total += 1
        try:
            allowed = getattr(n, 'allowed', None)
            if not isinstance(allowed, dict):
                allowed = {}
                setattr(n, 'allowed', allowed)
        except Exception:
            continue

        # If this op is pinned to PIM-only (CPU/NPU both disabled), allow CPU so the
        # graph remains schedulable on a non-PIM topology.
        try:
            pim_only = bool(allowed.get('pim', False)) and (not bool(allowed.get('cpu', False)))
            if npu_cnt > 0:
                pim_only = pim_only and (not bool(allowed.get('npu', False)))
        except Exception:
            pim_only = False
        if pim_only:
            allowed['cpu'] = True
            touched += 1
            
    if verbose or touched > 0:
        logger.warning('[HW] No PIM detected; falling back PIM-only ops to CPU (touched %d/%d nodes).', touched, total)
    return g

def _apply_policy_on_graph(g: TaskGraph, policy: str, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if policy == 'pd':
        for _, n in g2.nodes.items():
            if phase == 'decode':
                n.allowed['npu'] = False
                n.allowed['pim'] = True
                n.allowed['cpu'] = False
            else:
                n.allowed['npu'] = True
                n.allowed['pim'] = False
                n.allowed['cpu'] = True
        return g2

    if policy == 'weights_on_pim':
        for _, n in g2.nodes.items():
            has_w = n.weight_id is not None and (n.weight_size or 0) > 0
            n.allowed['pim'] = bool(has_w)
            n.allowed['npu'] = not has_w
            n.allowed['cpu'] = not has_w
        return g2

    if policy == 'attn_on_pim':
        for _, n in g2.nodes.items():
            is_attn = _is_attention_node(n)
            n.allowed['pim'] = bool(is_attn)
            n.allowed['npu'] = not is_attn
            n.allowed['cpu'] = not is_attn
        return g2

    raise ValueError(f'Unknown policy: {policy}')


# ===== Baseline registry and paper baselines =====
from typing import Callable

_BASELINE_REGISTRY: Dict[str, Callable[[TaskGraph], TaskGraph]] = {}
PD_BASELINES = {'pd','ianus','neupims','attacc','facil',}

def register_baseline(name: str):
    name = (name or "").strip().lower()
    def _deco(fn):
        _BASELINE_REGISTRY[name] = fn
        return fn
    return _deco

def _is_op(n: TaskNode, *tags: str) -> bool:
    op = str(getattr(n, 'attrs', {}).get('op') or n.name or '').upper()
    return any(tag.upper() in op for tag in tags)

def _arith_intensity(n: TaskNode) -> float:
    bytes_total = float(getattr(n, 'bytes_read', 0.0) + getattr(n, 'bytes_write', 0.0)) + 1e-9
    return float(getattr(n, 'flops', 0.0)) / bytes_total

def _is_kv_rw(n: TaskNode) -> bool:
    nm = (n.name or '').lower()
    op = str(getattr(n, 'attrs', {}).get('op') or '').lower()
    return any(k in nm or k in op for k in (
        'kv_read', 'kv_write',
        'k_read', 'v_read', 'k_write', 'v_write',
        'k_cache', 'v_cache',
    ))

def _is_gemv_like(n: TaskNode, *, phase: str) -> bool:
    op = str(getattr(n, 'attrs', {}).get('op') or n.name or '').upper()
    if phase == 'decode' and any(t in op for t in ['Q','K','V','O','FFN_W1','FFN_W2','GELU']):
        return True
    return str(getattr(n, 'attrs', {}).get('arith_op') or '').lower() == 'gemv'


@register_baseline('ianus')
def _baseline_ianus(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2

    for _, n in g2.nodes.items():
        on_pim = True
        if _is_op(n, 'Q', 'K', 'SOFTMAX', 'NORM', 'ADD'):
            on_pim = False
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = n.allowed.get('npu', True)
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2 

@register_baseline('neupims')
def _baseline_neupims(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2
    for _, n in g2.nodes.items():
        on_pim = _is_op(n, 'QK', 'SV') or _is_kv_rw(n)
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2

@register_baseline('attacc')
def _baseline_attacc(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2
    for _, n in g2.nodes.items():
        on_pim = _is_op(n, 'QK', 'SV','SOFTMAX') or _is_kv_rw(n)
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2

@register_baseline('facil')
def _baseline_facil(g: TaskGraph, *, phase: str) -> TaskGraph:
    g2 = _clone_graph(g)
    if phase == 'prefill':
        for _, n in g2.nodes.items():
            n.allowed['npu'] = True; n.allowed['pim'] = False; n.allowed['cpu'] = n.allowed.get('cpu', True)
        return g2
    for _, n in g2.nodes.items():
        on_pim = _is_gemv_like(n, phase='decode') or _is_op(n, 'QK', 'SV')
        n.allowed['pim'] = on_pim
        n.allowed['npu'] = not on_pim
        n.allowed['cpu'] = n.allowed.get('cpu', True)
    return g2


def _eval_one_baseline(
    cfg: Dict,
    policy: str,
    *,
    shared_graph: TaskGraph | None = None,
    shared_shape: Any = None,
    uniform_weight_storage_fmt: str | None = None,
    artifact_tag: str | None = None,
) -> Dict:
    reset_simulation_logger()

    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    if shared_graph is not None and shared_shape is not None:
        graph, shape = shared_graph, shared_shape
    else:
        graph, shape = build_graph(cfg)

    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))

    base_dir = Path(cfg["result_dir"])
    tag_tok = _artifact_tag_token(artifact_tag)
    algo_dir_name = f"algo_{policy}" + (f"__{tag_tok}" if tag_tok else "")
    algo_dir = base_dir / algo_dir_name
    algo_dir.mkdir(parents=True, exist_ok=True)

    # PIM trace 共享的模型形状信息
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, "dim", 128)),
        n_heads=int(getattr(shape, "n_heads", 1)),
        n_kv_heads=int(getattr(shape, "n_kv_heads", 1)),
        ffn_dim=int(getattr(shape, "ffn_dim", 512)),
        seqlen=prefill_len,
    )

    if tag_tok:
        sim_log_path = algo_dir / f"pim_simulation_{tag_tok}.txt"
    else:
        sim_log_path = Path(cfg.get(
            "simulation_log_file",
            algo_dir / "pim_simulation.txt",
        ))

    npu_backend = _normalize_npu_backend(cfg.get('npu_backend', None))
    if _cluster_type_count(cluster, 'npu') <= 0:
        npu_backend = None
    pim_fast_mode = bool(cfg.get('pim_fast_mode', False))
    cost = CostModel(
        cluster=cluster,
        dtype=cfg.get("dtype", "fp16"),
        pim_config_path=Path(cfg.get("pim_config_path")),
        gb_config_path=Path(cfg.get("gb_config_path")),
        ramulator_config_path=Path(cfg.get("ramulator_config_path")),
        simulation_log_file=sim_log_path,
        model_dict=model_dict,
        pim_fast_mode=pim_fast_mode,
        npu_backend=npu_backend
    )

    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    # 按 baseline policy 生成 prefill / decode 两个 graph
    pol = (policy or "").lower()
    if pol in _BASELINE_REGISTRY:
        g_prefill = _BASELINE_REGISTRY[pol](graph, phase="prefill")
        g_decode = _BASELINE_REGISTRY[pol](graph, phase="decode")
    else:
        g_prefill = _apply_policy_on_graph(graph, policy, phase="prefill")
        g_decode = _apply_policy_on_graph(graph, policy, phase="decode")

    _fallback_npu_to_cpu_if_needed(g_prefill, cluster)
    _fallback_npu_to_cpu_if_needed(g_decode, cluster)
    _fallback_pim_to_cpu_if_needed(g_prefill, cluster)
    _fallback_pim_to_cpu_if_needed(g_decode, cluster)

    is_pd = pol in PD_BASELINES

    best: Dict[str, Any] | None = None
    best_label = None
    best_prefill_ser = None
    best_decode_ser = None
    best_sched = None

    label = auto_select_kv_policy(
        strategy="naive",
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=g_prefill,
        graph_decode=g_decode,
        shape=shape,
    )

    sel = getattr(label, 'kv_policy_selected', 'unknown')
    sc = getattr(label, 'kv_policy_scores', {})
    msg = _fmt_kv_policy_scores(sc)
    if msg:
        _debug(str(f"[KV-SELECT] selected={sel} | {msg}"))
    else:
        _debug(str(f"[KV-SELECT] selected={sel}"))

    kv_place = _infer_kv_place_from_label(label)
    g_prefill = _apply_kv_place_constraints(g_prefill, kv_place)
    g_decode = _apply_kv_place_constraints(g_decode, kv_place)

    sched = _make_scheduler("naive", cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=GlobalMemoryManager())
    weight_fmt_map = _build_uniform_weight_storage_map(graph, uniform_weight_storage_fmt)
    sched.set_storage_format_map(weight_fmt_map)
    t_prefill, prefill_ser = simulate_prefill(sched, cfg, g_prefill)

    # PD baseline 需要把 KV 从 host 搬到 PIM 的一次性开销算进去
    t_kv_move = 0.0
    if is_pd and label.kv_in_pim and label.kv_total_bytes > 0:
        host = cluster.devices_by_type("cpu")[0]
        pim_list = cluster.devices_by_type("pim")
        if pim_list:
            per = label.kv_total_bytes // max(1, len(pim_list))
            for d in pim_list:
                t_kv_move = max(t_kv_move, cost.comm_cost(host, d, per))

    t_decode, decode_ser = simulate_decode_progressive(
        sched, cfg, g_decode, prefill_end=t_prefill
    )

    # decode_time_effective = float(t_decode + (t_kv_move if is_pd else 0.0))
    decode_time_effective = float(t_decode)
    total_time = float(t_prefill + decode_time_effective)

    best = {
        "prefill_time_s": float(t_prefill),
        "decode_time_s": decode_time_effective,
        "total_time_s": total_time,
        "kv_in_pim": bool(getattr(label, "kv_in_pim", False)),
        "kv_total_bytes": int(getattr(label, "kv_total_bytes", 0) or 0),
        "pim_weight_capacity_bytes": int(getattr(label, "pim_weight_capacity_bytes", 0) or 0),
    }
    best_label = label
    best_prefill_ser = prefill_ser
    best_decode_ser = decode_ser
    best_sched = sched

    try:
        if best_sched is not None and hasattr(best_sched, "stats"):
            mode_tok = _artifact_tag_token(_storage_mode_display_name(uniform_weight_storage_fmt))
            prefix = f"{policy}" + (f"_{mode_tok}" if mode_tok else "") + f"_prefill-{prefill_len}xdecode_{decode_len}"
            decode_stride = int(cfg.get("decode_sample_stride", 64) or 64)
            trace_ops = algo_dir / f"{prefix}_ops_trace.csv"
            trace_comms = algo_dir / f"{prefix}_comms_trace.csv"
            trace_ops.parent.mkdir(parents=True, exist_ok=True)
            best_sched.stats.dump_trace_csv(
                trace_ops,
                trace_comms,
            )
            # Record trace paths on the plan label for downstream scripts.
            try:
                setattr(best_label, 'trace_ops_csv', str(trace_ops))
                setattr(best_label, 'trace_comms_csv', str(trace_comms))
            except Exception:
                pass
    except Exception:
        pass

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    pim_trace = None
    try:
        if best_sched is not None and hasattr(best_sched, "pim_trace"):
            pim_trace = list(getattr(best_sched, "pim_trace") or [])
    except Exception:
        pim_trace = None

    return {
        "policy": policy,
        "pim_strategy": getattr(best_label, 'kv_policy_selected', None),
        "pim_strategy_scores": getattr(best_label, 'kv_policy_scores', None),
        "prefill_time_s": best["prefill_time_s"],
        "decode_time_s": best["decode_time_s"],
        "total_time_s": best["total_time_s"],
        "batch": batch,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "prefill_schedule": best_prefill_ser,
        "decode_steps": best_decode_ser,
        "pim_trace": pim_trace,
        "kv_in_pim": best.get("kv_in_pim", False),
        "kv_total_bytes": best.get("kv_total_bytes", 0),
        "pim_weight_capacity_bytes": best.get("pim_weight_capacity_bytes", 0),
        "weight_storage_mode": _storage_mode_display_name(uniform_weight_storage_fmt),
        "weight_storage_format": _normalize_weight_storage_fmt(uniform_weight_storage_fmt or 'ND'),
        "weight_storage_map_summary": _weight_map_summary(_collect_weight_ids_from_graph(graph), weight_fmt_map),
        "label": best_label,
    }


def _run_strategy_once(
    strategy: str,
    cfg: Dict,
    *,
    shared_graph: TaskGraph | None = None,
    shared_shape: Any = None,
    uniform_weight_storage_fmt: str | None = None,
    artifact_tag: str | None = None,
) -> Dict:
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    if shared_graph is not None and shared_shape is not None:
        graph, shape = _clone_graph(shared_graph), shared_shape
    else:
        graph, shape = build_graph(cfg)

    # If there is no NPU in the hardware topology, fall back NPU ops to CPU.
    _fallback_npu_to_cpu_if_needed(graph, cluster)

    batch = int(cfg.get("batch", 1))
    prefill_len = int(cfg.get("prefill_len", 128))
    decode_len = int(cfg.get("decode_len", 32))

    # PIM trace 需要的模型形状信息
    model_dict = _make_shared_model_dict(
        dim=int(getattr(shape, "dim", 128)),
        n_heads=int(getattr(shape, "n_heads", 1)),
        n_kv_heads=int(getattr(shape, "n_kv_heads", 1)),
        ffn_dim=int(getattr(shape, "ffn_dim", 512)),
        seqlen=prefill_len,
    )

    reset_simulation_logger()

    tag_tok = _artifact_tag_token(artifact_tag)
    if tag_tok:
        result_dir = Path(cfg.get("result_dir", "./output/strategy_results"))
        result_dir.mkdir(parents=True, exist_ok=True)
        sim_log_path = result_dir / f"pim_simulation_{str(strategy).lower()}_{tag_tok}.txt"
    else:
        sim_log_path = Path(cfg.get(
            "simulation_log_file",
            "./output/pim_simulation.txt",
        ))

    npu_backend = _normalize_npu_backend(cfg.get('npu_backend', None))
    if _cluster_type_count(cluster, 'npu') <= 0:
        npu_backend = None
    pim_fast_mode = bool(cfg.get('pim_fast_mode', False))
    cost = CostModel(
        cluster=cluster,
        dtype=cfg.get("dtype", "fp16"),
        pim_config_path=Path(cfg.get("pim_config_path")),
        gb_config_path=Path(cfg.get("gb_config_path")),
        ramulator_config_path=Path(cfg.get("ramulator_config_path")),
        simulation_log_file=sim_log_path,
        model_dict=model_dict,
        npu_backend=npu_backend,
        pim_fast_mode=pim_fast_mode
    )

    try:
        cost.logger.start_simulation()
    except Exception:
        pass

    best: Dict[str, Any] | None = None
    best_sched = None
    best_prefill_ser = None
    best_decode_ser = None
    best_label = None

    label = auto_select_kv_policy(
        strategy=strategy,
        cfg=cfg,
        cluster=cluster,
        cost=cost,
        graph=graph,
        shape=shape,
        capture_best_schedule=True,
    )

    sel = getattr(label, 'kv_policy_selected', 'unknown')
    sc = getattr(label, 'kv_policy_scores', {})
    msg = _fmt_kv_policy_scores(sc)
    if msg:
        _debug(str(f"[KV-SELECT] selected={sel} | {msg}"))
    else:
        _debug(str(f"[KV-SELECT] selected={sel}"))

    kv_place = _infer_kv_place_from_label(label)
    graph_kv = _apply_kv_place_constraints(graph, kv_place)

    kv_in_pim = bool(getattr(label, "kv_in_pim", False))
    kv_total_bytes = int(getattr(label, "kv_total_bytes", 0) or 0)
    kv_weight_cap = int(getattr(label, "pim_weight_capacity_bytes", 0) or 0)
    sim_best = getattr(label, "_kv_policy_best_sim", None)
    weight_fmt_map = _build_uniform_weight_storage_map(graph, uniform_weight_storage_fmt)
    if not weight_fmt_map and isinstance(sim_best, dict) and sim_best.get("sched") is not None:
        sched = sim_best.get("sched")
        t_prefill = float(sim_best.get("prefill_s", 0.0) or 0.0)
        t_decode = float(sim_best.get("decode_s", 0.0) or 0.0)
        prefill_ser = sim_best.get("prefill_schedule")
        decode_ser = sim_best.get("decode_steps")
        total_time = float(sim_best.get("total_s", t_prefill + t_decode) or (t_prefill + t_decode))
    else:
        buffer_mgr = GlobalMemoryManager()
        sched = _make_scheduler(strategy, cluster, cost, label, batch=batch, seq_len=prefill_len, buffer=buffer_mgr)
        sched.reset_state()
        sched.set_storage_format_map(weight_fmt_map)

        t_prefill, prefill_ser = simulate_prefill(sched, cfg, graph_kv)
        t_decode, decode_ser = simulate_decode_progressive(
            sched, cfg, graph_kv, prefill_end=t_prefill
        )
        total_time = float(t_prefill + t_decode)

    best = {
        "prefill_time_s": float(t_prefill),
        "decode_time_s": float(t_decode),
        "total_time_s": total_time,
        "kv_in_pim": kv_in_pim,
        "kv_total_bytes": kv_total_bytes,
        "pim_weight_capacity_bytes": kv_weight_cap,
    }
    best_sched = sched
    best_prefill_ser = prefill_ser
    best_decode_ser = decode_ser
    best_label = label

    try:
        if best_sched is not None and hasattr(best_sched, "stats"):
            mode_tok = _artifact_tag_token(_storage_mode_display_name(uniform_weight_storage_fmt))
            prefix = f"{strategy}" + (f"_{mode_tok}" if mode_tok else "") + f"_prefill-{prefill_len}xdecode_{decode_len}"
            result_dir = Path(cfg.get("result_dir", "./output/strategy_results"))
            result_dir.mkdir(parents=True, exist_ok=True)
            decode_stride = int(cfg.get("decode_sample_stride", 64) or 64)
            trace_ops = result_dir / f"{prefix}_ops_trace.csv"
            trace_comms = result_dir / f"{prefix}_comms_trace.csv"
            best_sched.stats.dump_trace_csv(
                trace_ops,
                trace_comms,
            )
            # Record trace paths on the plan label for downstream scripts.
            try:
                setattr(best_label, 'trace_ops_csv', str(trace_ops))
                setattr(best_label, 'trace_comms_csv', str(trace_comms))
            except Exception:
                pass

    except Exception:
        pass

    try:
        cost.logger.end_simulation()
        cost.logger.close()
    except Exception:
        pass

    pim_trace = None
    try:
        if best_sched is not None and hasattr(best_sched, "pim_trace"):
            pim_trace = list(getattr(best_sched, "pim_trace") or [])
    except Exception:
        pim_trace = None

    return {
        "strategy": strategy,
        "pim_strategy": getattr(best_label, 'kv_policy_selected', None),
        "pim_strategy_scores": getattr(best_label, 'kv_policy_scores', None),
        "prefill_time_s": best["prefill_time_s"],
        "decode_time_s": best["decode_time_s"],
        "total_time_s": best["total_time_s"],
        "batch": batch,
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "prefill_schedule": best_prefill_ser,
        "decode_steps": best_decode_ser,
        "pim_trace": pim_trace,
        "kv_in_pim": best.get("kv_in_pim", False),
        "kv_total_bytes": best.get("kv_total_bytes", 0),
        "pim_weight_capacity_bytes": best.get("pim_weight_capacity_bytes", 0),
        "weight_storage_mode": _storage_mode_display_name(uniform_weight_storage_fmt),
        "weight_storage_format": _normalize_weight_storage_fmt(uniform_weight_storage_fmt or 'ND'),
        "weight_storage_map_summary": _weight_map_summary(_collect_weight_ids_from_graph(graph), weight_fmt_map),
        "label": best_label,
    }


def _ensure_dir(p:Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p

def _label_summary(label: PlanLabel | None) -> Dict[str, Any]:
    if label is None:
        return {}
    out = {
        'kv_place': str(getattr(label, 'kv_place', 'pim' if bool(getattr(label, 'kv_in_pim', False)) else 'host')),
        'kv_in_npu': bool(getattr(label, 'kv_in_npu', False)),
        'kv_in_pim': bool(getattr(label, 'kv_in_pim', False)),
        'kv_total_bytes': int(getattr(label, 'kv_total_bytes', 0) or 0),
        'kv_total_bytes_all': int(getattr(label, 'kv_total_bytes_all', getattr(label, 'kv_total_bytes_raw', 0)) or 0),
        'pim_weight_capacity_bytes': int(getattr(label, 'pim_weight_capacity_bytes', 0) or 0),
        'pinned_fc_on_pim': sorted(list(getattr(label, 'pinned_fc_on_pim', set()) or [])),
    }

    # Optional: record trace file locations if the caller populated them.
    try:
        ops_p = getattr(label, 'trace_ops_csv', None)
        comms_p = getattr(label, 'trace_comms_csv', None)
        if ops_p:
            out['trace_ops_csv'] = str(ops_p)
        if comms_p:
            out['trace_comms_csv'] = str(comms_p)
    except Exception:
        pass

    return out

def _save_best_json(algo_dir: Path, tag: str, policy: str, *, times: Dict, prefill_schedule=None, decode_steps=None, cfg: Dict|None=None, label: PlanLabel | None = None):
    payload = {
        'policy': policy,
        'pim_strategy': times.get('pim_strategy', 'unknown'),
        'config': {'batch': int((cfg or {}).get('batch', 1)), 'prefill_len': int((cfg or {}).get('prefill_len', 0)), 'decode_len': int((cfg or {}).get('decode_len', 0)), 'dtype': (cfg or {}).get('dtype')},
        'best_times': {'prefill': float(times.get('prefill_time_s', 0.0)), 'decode': float(times.get('decode_time_s', 0.0)), 'total': float(times.get('total_time_s', 0.0))},
    }

    label_dict = _label_summary(label) or _label_summary(times.get('label'))
    if label_dict:
        payload['plan_label'] = label_dict

    if prefill_schedule is not None:
        payload['prefill_schedule'] = prefill_schedule
    if decode_steps is not None:
        payload['decode_steps'] = decode_steps
    pim_trace = times.get('pim_trace')
    if pim_trace is not None:
        payload['pim_trace'] = pim_trace

    # Also record the KV policy comparison numbers if present.
    if 'pim_strategy_scores' in times:
        payload['pim_strategy_scores'] = times.get('pim_strategy_scores')
    path = algo_dir / f"best_summary_{tag}.json"
    with open(path,'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _debug(f"[Strategy] Saved best summary to: {path}")
    return path

def evaluate_suite(cfg: Dict, *, algos: List[str], baselines: List[str], result_dir: str | None, debug: bool, combined_out: str):
    base_dir = _ensure_dir(Path(result_dir or './output/len_sweep'))
    tag = f"{int(cfg.get('prefill_len', 0))}x{int(cfg.get('decode_len', 0))}"
    results: List[Dict] = []
    # --- baselines ONCE ---
    blist = []
    for b in baselines:
        b = b.strip().lower()
        if not b: continue
        if b not in blist: blist.append(b)
    for b in blist:
        algo_dir = _ensure_dir(base_dir / f"algo_{b}")
        try: 
            setup_logging(bool(cfg.get('debug', False)), log_file=str(algo_dir / f"debug_{tag}.txt"))
        except Exception: 
            logger.error(f"Failed to setup logging for baseline '{b}'")
            pass
        cfg_b = dict(cfg)
        cfg_b['simulation_log_file'] = str(algo_dir / f"pim_sim_{tag}.txt")
        r = _eval_one_baseline(cfg_b, b)
        _save_best_json(algo_dir, tag, policy=f"algo:{b}", times=r, cfg=cfg_b, prefill_schedule=r.get('prefill_schedule'), decode_steps=r.get('decode_steps'), label=r.get('label'))
        results.append({
            'policy': f"algo:{b}",
            'pim_strategy': r.get('pim_strategy'),
            'kv_in_pim': bool(r.get('kv_in_pim', False)),
            'kv_total_bytes': int(r.get('kv_total_bytes', 0) or 0),
            'pim_weight_capacity_bytes': int(r.get('pim_weight_capacity_bytes', 0) or 0),
            **{k: r[k] for k in ('prefill_time_s','decode_time_s','total_time_s')},
        })
    # --- algorithms ---
    alist = []
    for a in algos:
        a = a.strip().lower()
        if not a: continue
        if a not in alist: alist.append(a)
    # Build once to share across algos
    cluster = demo_cluster(cfg)
    cfg['topology'] = getattr(cluster, 'topology', cfg.get('topology', None))
    shared_graph, shared_shape = build_graph(cfg)
    for a in alist:
        algo_dir = _ensure_dir(base_dir / f"algo_{a}")
        try: 
            setup_logging(bool(cfg.get('debug', False)), log_file=str(algo_dir / f"debug_{tag}.txt"))
        except Exception: 
            logger.error(f"Failed to setup logging for algo '{a}'")
            pass
            
        cfg_a = dict(cfg)
        cfg_a['simulation_log_file'] = str(algo_dir / f"pim_sim_{tag}.txt")
        cfg_a['result_dir'] = str(algo_dir)
        res = _run_strategy_once(a, cfg_a, shared_graph=shared_graph, shared_shape=shared_shape)
        _save_best_json(algo_dir, tag, policy=res.get('policy', f"algo:{a}"), times=res, prefill_schedule=res.get('prefill_schedule'), decode_steps=res.get('decode_steps'), cfg=cfg_a, label=res.get('label'))
        results.append({
            'policy': f"algo:{a}",
            'pim_strategy': res.get('pim_strategy'),
            'pim_strategy_scores': res.get('pim_strategy_scores'),
            'kv_in_pim': bool(res.get('kv_in_pim', False)),
            'kv_total_bytes': int(res.get('kv_total_bytes', 0) or 0),
            'pim_weight_capacity_bytes': int(res.get('pim_weight_capacity_bytes', 0) or 0),
            **{k: res[k] for k in ('prefill_time_s','decode_time_s','total_time_s')},
        })
    # --- combined ---
    if results:
        os.makedirs(os.path.dirname(combined_out), exist_ok=True)
        with open(combined_out, 'w', encoding='utf-8') as f:
            json.dump({'config': cfg, 'results': results}, f, ensure_ascii=False, indent=2)
        print(f"[REPORT] Combined comparison saved to: {combined_out}")
    # Pretty print
    print("\n=== Strategy/Baseline Comparison ===")
    header = f"{'Policy':<22} {'PIM':<8} {'Prefill(s)':>12} {'Decode(s)':>12} {'Total(s)':>12}"
    print(header); print('-'*len(header))
    for r in results:
        pim_s = str(r.get('pim_strategy') or '')
        print(f"{r['policy']:<22} {pim_s:<8} {r['prefill_time_s']:>12.4f} {r['decode_time_s']:>12.4f} {r['total_time_s']:>12.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode')

    # evaluate mode: run all algos + baselines
    sp_eval = sub.add_parser('evaluate', help='Run selected algos and baselines; outputs go under result_dir.')
    sp_eval.add_argument('--config', required=True, type=str, help='Path to a JSON config with run parameters.')
    sp_eval.add_argument('--debug', action='store_true', help='Enable verbose logging.')
    sp_eval.add_argument('--model_family', type=str)
    sp_eval.add_argument('--model_variant', type=str)
    sp_eval.add_argument('--dtype', type=str)
    sp_eval.add_argument('--batch', type=int)
    sp_eval.add_argument('--prefill_len', type=int)
    sp_eval.add_argument('--decode_len', type=int)
    sp_eval.add_argument('--decode_sample_stride', type=int)
    sp_eval.add_argument('--decode_plan_refresh_stride', type=int,
                         help='Run a full decode search every N tokens; hidden tokens replay a fixed plan. 0 means freeze after warmup.')
    sp_eval.add_argument('--result_dir', type=str)
    sp_eval.add_argument('--hardware_json', type=str,
                         help='Path to a JSON file with hardware topology (devices + links).')
    sp_eval.add_argument('--algo', type=str,
                         help='Algo list, e.g. "heft,sa,ga" or single name')
    sp_eval.add_argument('--baselines', type=str,
                         help='Baseline list, e.g. "pd,weights_on_pim,attn_on_pim"')
    sp_eval.add_argument('--npu_backend', type=str, default=None,
                         choices=['fast_mode', 'ascend_310b_json', 'llmcompass'],
                         help='NPU operator-latency backend: fast_mode/ascend_310b_json/llmcompass. Must be explicitly specified (in config JSON or CLI).')
    sp_eval.add_argument('--pim_fast_mode', action='store_true',default=None)
    sp_eval.add_argument('--weight-local-load-overlap-ratio', dest='weight_local_load_overlap_ratio', type=float,
                         help='Override config.WEIGHT_LOCAL_LOAD_OVERLAP_RATIO in [0,1] for this process only.')
    # Tensor-parallel shard controls (graph splitting)
    sp_eval.add_argument('--tp_qkv', type=int,
                         help='Tensor-parallel shard size for Q/K/V generation and attention head sharding (column split).')
    sp_eval.add_argument('--tp_ffn', type=int,
                         help='Tensor-parallel shard size for FFN intermediate dimension (ffn_dim split).')
    sp_eval.add_argument('--tp_moe', type=int,
                         help='Expert-parallel shard size for MoE experts / Mixtral router-expert partitioning.')
    # weight-suggest mode: multi-pass SA to suggest weight formats
    sp_ws = sub.add_parser('weight-suggest', help='Run multi-pass SA to suggest weight formats and fixed baseline experiments.')
    sp_ws.add_argument('--config', required=True, type=str, help='Path to a JSON config with run parameters.')
    sp_ws.add_argument('--debug', action='store_true', help='Enable verbose logging.')
    sp_ws.add_argument('--model_family', type=str)
    sp_ws.add_argument('--model_variant', type=str)
    sp_ws.add_argument('--dtype', type=str)
    sp_ws.add_argument('--batch', type=int)
    sp_ws.add_argument('--prefill_len', type=int)
    sp_ws.add_argument('--decode_len', type=int)
    sp_ws.add_argument('--decode_sample_stride', type=int)
    sp_ws.add_argument('--decode_plan_refresh_stride', type=int,
                       help='Run a full decode search every N tokens; hidden tokens replay a fixed plan. 0 means freeze after warmup.')
    sp_ws.add_argument('--result_dir', type=str)
    sp_ws.add_argument('--hardware_json', type=str,
                         help='Path to a JSON file with hardware topology (devices + links).')
    sp_ws.add_argument('--algo', type=str,help='Algo list, e.g. "heft,sa,ga"')
    sp_ws.add_argument('--all_passes_json', type=str, help='Override path for all passes JSON.')
    sp_ws.add_argument('--best_summary_json', type=str, help='Override path for best pass summary JSON.')
    sp_ws.add_argument('--weight_format_json', type=str, help='Override path for accepted weight format JSON.')
    sp_ws.add_argument('--npu_backend', type=str, default=None,
                        choices=['fast_mode', 'ascend_310b_json', 'llmcompass'],
                        help='NPU operator-latency backend: fast/ascend_310b_json/llmcompass. Must be explicitly specified (in config JSON or CLI).')
    sp_ws.add_argument('--pim_fast_mode', action='store_true', default=None)   
    sp_ws.add_argument('--weight-local-load-overlap-ratio', dest='weight_local_load_overlap_ratio', type=float,
                       help='Override config.WEIGHT_LOCAL_LOAD_OVERLAP_RATIO in [0,1] for this process only.')
    # Tensor-parallel shard controls (graph splitting)
    sp_ws.add_argument('--tp_qkv', type=int,
                        help='Tensor-parallel shard size for Q/K/V generation and attention head sharding (column split).')
    sp_ws.add_argument('--tp_ffn', type=int,
                        help='Tensor-parallel shard size for FFN intermediate dimension (ffn_dim split).')
    sp_ws.add_argument('--tp_moe', type=int,
                        help='Expert-parallel shard size for MoE experts / Mixtral router-expert partitioning.')
    sp_ws.add_argument('--format_outer_max_iters', type=int,
                       help='Deprecated compatibility knob. If format_block_change_percent is unset, percent is derived as 1/format_outer_max_iters.')
    sp_ws.add_argument('--format_block_change_percent', type=float,
                       help='At most this fraction of total blocks may change per outer iteration. outer_max is auto-derived as ceil(1/percent). Default: 0.2.')
    sp_ws.add_argument('--format_inner_max_blocks', type=int,
                       help='AL inner sweep cap (0 means no cap).')
    sp_ws.add_argument('--format_nd_margin_init', type=float,
                       help='AL initial ND band (wide) in [0,1].')
    sp_ws.add_argument('--format_nd_margin_decay', type=float,
                       help='AL ND band decay per outer iteration.')
    sp_ws.add_argument('--format_nd_margin_min', type=float,
                       help='AL minimum ND band.')
    sp_ws.add_argument('--format_inner_improve_eps', type=float,
                       help='AL accept change if new_total + eps < old_total.')
    sp_ws.add_argument('--format_outer_stop_eps', type=float,
                       help='AL stop when outer_n is worse than outer_{n-1} by eps.')
    sp_ws.add_argument('--format_block_layer_span', type=int,
                       help='Group the same weight across every N layers into one block. 8 or 4 are typical; 0 keeps the legacy all-layer merge.')
    sp_ws.add_argument('--format_reload_count_mode', type=str, choices=['raw', 'per_device', 'soft_per_device'],
                       help='Use raw reload totals or normalize by the number of devices of each type when comparing NPU vs PIM reload pressure.')

    args, unknown = parser.parse_known_args()
    if args.mode is None:
        parser.error("Please specify a mode: 'eval' or 'weight-suggest'.")
    
    return args


def _normalize_list_field(val) -> list[str]:
    items: list[str] = []
    if isinstance(val, list):
        seq = val
    else:
        seq = [val]
    for item in seq:
        if item is None:
            continue
        # 先用逗号拆，再按空白拆
        for tok in str(item).replace(',', ' ').split():
            tok = tok.strip()
            if tok:
                items.append(tok)
    return items

def _load_cfg_from_json(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config JSON must be an object/dict, got: {type(raw).__name__}")
    # cfg 完全由 JSON 决定
    return dict(raw)

def _apply_runtime_config_overrides(cfg: Dict) -> Dict[str, Any]:
    """Apply per-run overrides to the imported config module without editing config.py on disk."""
    import config as _runtime_config

    applied: Dict[str, Any] = {}

    if 'weight_local_load_overlap_ratio' in cfg and cfg.get('weight_local_load_overlap_ratio') is not None:
        try:
            ratio = float(cfg.get('weight_local_load_overlap_ratio'))
        except Exception as exc:
            raise ValueError(
                f"Invalid weight_local_load_overlap_ratio={cfg.get('weight_local_load_overlap_ratio')!r}; expected a float in [0, 1]"
            ) from exc
        if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                f"weight_local_load_overlap_ratio must be within [0, 1], got {ratio!r}"
            )
        setattr(_runtime_config, 'WEIGHT_LOCAL_LOAD_OVERLAP_RATIO', float(ratio))
        applied['WEIGHT_LOCAL_LOAD_OVERLAP_RATIO'] = float(ratio)

    return applied

def main():
    args = parse_args()

    if getattr(args, 'mode', None) in ('evaluate', 'weight-suggest'):
        cfg = _load_cfg_from_json(getattr(args, 'config'))
        requested_debug = bool(getattr(args, 'debug', False)) or cfg.get('debug', False)
        cfg['debug'] = bool(requested_debug)
        cfg['_weight_suggest_debug_summary_only'] = bool(getattr(args, 'mode', None) == 'weight-suggest')
        _set_weight_suggest_debug_summary_only(
            cfg['_weight_suggest_debug_summary_only'],
            emit_progress=bool(requested_debug),
        )
        if cfg['_weight_suggest_debug_summary_only']:
            cfg['_requested_debug'] = bool(requested_debug)
            cfg['debug'] = bool(requested_debug)
        
        override_fields = [
            'model_family',
            'model_variant',
            'dtype',
            'batch',
            'prefill_len',
            'decode_len',
            'decode_sample_stride',
            'decode_plan_refresh_stride',
            'tp_qkv',
            'tp_ffn',
            'tp_moe',
            'result_dir',
            'hardware_json',
            'algo',
            'baselines',
            'all_passes_json',
            'best_summary_json',
            'weight_format_json',
            'npu_backend',
            'pim_fast_mode',
            'weight_local_load_overlap_ratio',
            'format_outer_max_iters',
            'format_block_change_percent',
            'format_inner_max_blocks',
            'format_nd_margin_init',
            'format_nd_margin_decay',
            'format_nd_margin_min',
            'format_inner_improve_eps',
            'format_outer_stop_eps',
            'format_block_layer_span',
            'format_reload_count_mode',
        ]
        for key in override_fields:
            val = getattr(args, key, None)
            if val is not None:
                cfg[key] = val

        runtime_cfg_overrides = _apply_runtime_config_overrides(cfg)
        if runtime_cfg_overrides:
            print(
                f"[runtime-config] applied {json.dumps(runtime_cfg_overrides, ensure_ascii=False, sort_keys=True)}"
            )

        # npu_backend is mandatory: must be explicitly specified in config or CLI
        if cfg.get('npu_backend', None) is None:
            raise ValueError("Missing required config key: 'npu_backend'. Choose from: fast, ascend_310b_lut, ascend_310b_json, llmcompass")
        cfg['npu_backend'] = _normalize_npu_backend(cfg.get('npu_backend'))

        # result_dir always encodes batch: <base>/<family>_<variant>_<dtype>_b<batch>
        result_dir = str(_build_result_dir(cfg, cfg.get('result_dir') or './output'))
        cfg['result_dir'] = result_dir
        if args.mode == 'weight-suggest':
            ws_al_log_path = cfg.get('weight_suggest_al_log_path')
            if not ws_al_log_path:
                ws_al_log_path = str(Path(result_dir) / "weight_suggest_al_debug.txt")
            cfg['weight_suggest_al_log_path'] = str(ws_al_log_path)
        else:
            _setup_weight_suggest_al_logger(None)
        
        tag = f"{int(cfg.get('prefill_len', 0))}x{int(cfg.get('decode_len', 0))}"
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        # Top-level driver logger
        setup_logging(bool(cfg.get('debug', False)), log_file=str(Path(result_dir) / "driver_debug.txt"))
        if args.mode == 'weight-suggest':
            _setup_weight_suggest_al_logger(cfg.get('weight_suggest_al_log_path'))
        
        # Normalize stride if provided
        if cfg.get('decode_sample_stride', None) is not None:
            try:
                cfg['decode_sample_stride'] = int(cfg['decode_sample_stride'])
            except Exception:
                pass

        if args.mode == 'weight-suggest':
            # Choose a single algo label for bookkeeping (run() itself performs SA-based tuning).
            algo_field = cfg.get('algo', 'heft')
            if isinstance(algo_field, list):
                algo_chosen = str(algo_field[0]) if algo_field else 'heft'
            else:
                parts = [t for t in str(algo_field).replace(',', ' ').split() if t]
                algo_chosen = parts[0] if parts else 'heft'

            # Output files are derived from result_dir + tag
            tag = _build_tag(cfg)
            if isinstance(cfg.get('tag'), str) and cfg['tag'].strip():
                tag = f"{tag}_{cfg['tag'].strip()}"
            
            # Derive per-run file paths (can be overridden by CLI)
            if not cfg.get('all_passes_json'):
                cfg['all_passes_json'] = str(Path(result_dir) / f"all_passes_{tag}.json")
            if not cfg.get('best_summary_json'):
                cfg['best_summary_json'] = str(Path(result_dir) / f"best_summary_{tag}.json")
            if not cfg.get('weight_format_json'):
                cfg['weight_format_json'] = str(Path(result_dir) / f"weight_storage_suggestion_{tag}.json")
            
            # Put simulation log inside the same result_dir and tag to avoid overwrite
            cfg['simulation_log_file'] = str(Path(result_dir) / f"pim_sim_{tag}.txt")
            
            print(f"[weight-suggest] result_dir={result_dir} tag={tag}")
            run(cfg)
            return

        if args.mode == 'evaluate':
            # Build lists from JSON (support comma-separated string or list)
            algos = _normalize_list_field(cfg.get('algo', 'heft'))
            baselines = _normalize_list_field(cfg.get('baselines', 'pd,weights_on_pim,attn_on_pim'))

            baseline_out = cfg.get('baseline_out') or str(Path(result_dir) / f"baseline_compare_{tag}.json")
            print(f"[evaluate] algos={algos} baselines={baselines} result_dir={result_dir} tag={tag}")
            evaluate_suite(cfg, algos=algos, baselines=baselines, result_dir=result_dir, debug=cfg['debug'], combined_out=baseline_out)
            return

if __name__ == '__main__':
    main()
