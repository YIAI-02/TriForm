"""Weight-format search helpers and normalization utilities."""

from __future__ import annotations

from .shared import *

def mapping_diff_ratio(a: Dict[str, str], b: Dict[str, str]) -> float:
    if not a and (not b):
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    diff = sum((1 for k in keys if a.get(k) != b.get(k)))
    return diff / float(len(keys))

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

