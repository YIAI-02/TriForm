from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Canonical dtype names used across the simulator.
DTYPE_BYTES: Dict[str, float] = {
    'fp32': 4.0,
    'fp16': 2.0,
    'bf16': 2.0,
    'fp8': 1.0,
    'fp4': 0.5,
    'int8': 1.0,
    'int4': 0.5,
}

# Accept a fairly broad set of user-facing aliases so JSON / CLI / scripts can
# use common framework spellings without silently falling back to fp16.
_DTYPE_ALIASES_RAW = {
    'fp32': 'fp32',
    'f32': 'fp32',
    'float32': 'fp32',
    'float': 'fp32',
    'single': 'fp32',
    'fp 32': 'fp32',
    'float 32': 'fp32',
    'fp16': 'fp16',
    'f16': 'fp16',
    'float16': 'fp16',
    'half': 'fp16',
    'fp 16': 'fp16',
    'float 16': 'fp16',
    'bf16': 'bf16',
    'bfloat16': 'bf16',
    'bf 16': 'bf16',
    'bfloat 16': 'bf16',
    'fp8': 'fp8',
    'float8': 'fp8',
    'e4m3': 'fp8',
    'e5m2': 'fp8',
    'fp 8': 'fp8',
    'float 8': 'fp8',
    'fp4': 'fp4',
    'float4': 'fp4',
    'fp 4': 'fp4',
    'float 4': 'fp4',
    'int8': 'int8',
    'i8': 'int8',
    'qint8': 'int8',
    'int 8': 'int8',
    'int4': 'int4',
    'i4': 'int4',
    'qint4': 'int4',
    'int 4': 'int4',
}


def _collapse_dtype_token(dtype: Any) -> str:
    s = str(dtype or '').strip().lower()
    if not s:
        return ''
    for prefix in ('torch.', 'numpy.', 'np.'):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for ch in (' ', '\t', '\r', '\n', '_', '-'):
        s = s.replace(ch, '')
    return s


_DTYPE_ALIASES: Dict[str, str] = {
    _collapse_dtype_token(k): v for k, v in _DTYPE_ALIASES_RAW.items()
}


def supported_dtypes() -> Tuple[str, ...]:
    return tuple(sorted(DTYPE_BYTES.keys()))


def normalize_dtype_token(dtype: Any, *, default: str = 'fp16', allow_none: bool = False) -> Optional[str]:
    if dtype is None:
        if allow_none:
            return None
        dtype = default

    raw = str(dtype).strip()
    if not raw:
        if allow_none:
            return None
        raw = str(default).strip()

    key = _collapse_dtype_token(raw)
    normalized = _DTYPE_ALIASES.get(key)
    if normalized is None:
        supported = ', '.join(sorted(DTYPE_BYTES.keys()))
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported canonical values: {supported}."
        )
    return normalized


def dtype_bytes(dtype: Any, *, default: str = 'fp16') -> float:
    normalized = normalize_dtype_token(dtype, default=default)
    assert normalized is not None
    return float(DTYPE_BYTES[normalized])