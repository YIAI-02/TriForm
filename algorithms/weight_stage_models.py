from __future__ import annotations

from dataclasses import dataclass


def clamp_overlap_ratio(value: float | int | None, *, default: float = 0.0) -> float:
    try:
        x = float(default if value is None else value)
    except Exception:
        x = float(default)
    if x != x:  # NaN
        x = float(default)
    return float(min(1.0, max(0.0, x)))


@dataclass(frozen=True)
class OverlapBreakdown:
    total_s: float
    saved_s: float
    overlap_ratio: float
    first_s: float
    second_s: float


@dataclass(frozen=True)
class WeightLoadStageBreakdown:
    total_s: float
    host_src_fmt: str
    resident_fmt: str
    l1_comm_s: float = 0.0
    l2_local_s: float = 0.0
    l1_l2_overlap_ratio: float = 0.0
    combine_rule: str = "serial"
    bytes_nd: int = 0
    bytes_src: int = 0


@dataclass(frozen=True)
class WeightComputeStageBreakdown:
    total_s: float
    compute_fmt: str
    backend: str
    combine_rule: str
    b1_s: float = 0.0
    b2_s: float = 0.0
    launch_overhead_s: float = 0.0


@dataclass(frozen=True)
class WeightOpTimingBreakdown:
    total_s: float
    load: WeightLoadStageBreakdown
    compute: WeightComputeStageBreakdown
    load_compute_overlap_ratio: float
    queue_wait_s: float = 0.0
    overlap_saved_s: float = 0.0
    cache_state: str = ""
    weight_id: str = ""
    weight_size_nd: int = 0
    host_storage_fmt: str = "ND"



def overlap_time(first_s: float, second_s: float, overlap_ratio: float) -> OverlapBreakdown:
    a = max(0.0, float(first_s or 0.0))
    b = max(0.0, float(second_s or 0.0))
    r = clamp_overlap_ratio(overlap_ratio, default=0.0)
    saved = float(r * min(a, b))
    total = float(a + b - saved)
    return OverlapBreakdown(
        total_s=float(total),
        saved_s=float(saved),
        overlap_ratio=float(r),
        first_s=float(a),
        second_s=float(b),
    )
