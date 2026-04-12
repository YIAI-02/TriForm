"""Path, tag, and weight-map helpers used by the main workflows."""

from __future__ import annotations

from .shared import *

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

ALL_PASSES_RESULT_PATH = "./output/all_passes.json"
BEST_PASS_SUMMARY_PATH = "./output/best_summary.json"

def _result_stride_for_naming(cfg: Dict) -> Any:
    stride = cfg.get('decode_plan_refresh_stride', None)
    if stride in (None, ''):
        stride = cfg.get('decode_sample_stride', None)
    return stride

def _build_result_dir(cfg: Dict, default_root: str = './output') -> Path:
    """
    Compose a result directory path that always includes batch:
            <base>/<family>_<variant>_<dtype>_b<batch>[_s<refresh_stride>]
    """
    base   = cfg.get('result_dir') or default_root
    family = cfg.get('model_family', 'unnamed')
    variant= cfg.get('model_variant', '')
    dtype  = normalize_dtype_token(cfg.get('dtype', 'fp16'), default='fp16')
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

