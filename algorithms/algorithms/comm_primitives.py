"""Communication primitives & topology-aware collectives.

Primitives supported:
  - allreduce (ring):
      * FC topology: uses the existing ring-step model.
      * STAR topology: models as reduce-to-host + optional scatter-from-host.
  - reduce (to host)
  - gather (to host)
  - scatter (from host)
  - transfer (p2p)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TYPE_CHECKING

from hardware import Cluster

if TYPE_CHECKING:
    from cost_model import CostModel


def normalize_topology(topology: Optional[str]) -> str:
    """Normalize topology string to one of {'fc','star'} (default 'fc')."""
    if not topology:
        return 'fc'
    t = str(topology).strip().lower()
    if t in ('fully_connected', 'full', 'fc', 'mesh'):
        return 'fc'
    if t in ('star', 'host', 'host_star', 'host-centric', 'host_centric'):
        return 'star'
    # Unknown: be conservative and keep contention model (i.e., 'star'-like)
    return t


def _p2p_duration_s(cost: CostModel, cluster: Cluster, src: str, dst: str, bytes_amount: int) -> float:
    try:
        sdev = cluster.devices[str(src)]
        ddev = cluster.devices[str(dst)]
    except Exception:
        return float('inf')
    t = float(cost.comm_cost(sdev, ddev, int(bytes_amount)))
    return t

def _reserve_p2p(
    *,
    comm: Any,
    cost: CostModel,
    cluster: Cluster,
    src: str,
    dst: str,
    bytes_amount: int,
    earliest: float,
    commit: bool,
    tag: str,
    extra: Optional[Dict[str, Any]] = None,
    local_tl: Optional[MutableMapping[Tuple[str, str], float]] = None,
) -> Tuple[float, float]:
    src = str(src)
    dst = str(dst)
    bytes_amount = int(bytes_amount)
    earliest = float(earliest)

    if commit:
        return comm.reserve(src, dst, bytes_amount, earliest=earliest, commit=True, tag=tag, extra=extra)

    if local_tl is None:
        local_tl = {}

    key = (src, dst)
    base_end = 0.0
    try:
        base_end = float(getattr(comm, 'timeline_end', {}).get(key, 0.0))
    except Exception:
        base_end = 0.0
    start = max(float(earliest), float(local_tl.get(key, 0.0)), float(base_end))
    dur = float(_p2p_duration_s(cost, cluster, src, dst, bytes_amount))
    if not math.isfinite(dur) or dur <= 0.0:
        return (float('inf'), float('inf'))
    end = start + dur
    local_tl[key] = float(end)
    return (float(start), float(end))


def ring_allreduce(
    *,
    cost: CostModel,
    cluster: Cluster,
    ring: Sequence[str],
    tensor_bytes: int,
    start: float,
) -> float:
    ring = [str(x) for x in ring]
    p = len(ring)
    if p <= 1:
        return float(start)

    tensor_bytes = int(max(0, tensor_bytes))
    chunk = int(math.ceil(float(tensor_bytes) / float(p)))
    t = float(start)

    # reduce-scatter + all-gather
    steps = 2 * (p - 1)
    for _ in range(steps):
        dt = 0.0
        for i in range(p):
            src = ring[i]
            dst = ring[(i + 1) % p]
            hop = float(_p2p_duration_s(cost, cluster, src, dst, chunk))
            if not math.isfinite(hop) or hop <= 0.0:
                return float('inf')
            dt = max(dt, hop)
        t += dt
    return float(t)

def reduce_to_host(
    *,
    comm: Any,
    cost: CostModel,
    cluster: Cluster,
    participants: Sequence[str],
    tensor_bytes: int,
    start: float,
    commit: bool,
    tag: str = 'reduce',
    extra_base: Optional[Dict[str, Any]] = None,
    host_name: Optional[str] = None,
) -> float:
    """Reduce: all participants send their tensors to host (root).

    Returns completion time (end timestamp) after all sends complete.
    """
    host = str(host_name or cost.get_host_device().name)
    participants = [str(x) for x in participants]
    tensor_bytes = int(max(0, tensor_bytes))

    local_tl: Dict[Tuple[str, str], float] = {}
    ends: List[float] = [float(start)]
    for src in participants:
        if src == host:
            continue
        extra = dict(extra_base or {})
        extra.update({'primitive': 'reduce', 'root': host, 'src': src})
        _, e = _reserve_p2p(
            comm=comm,
            cost=cost,
            cluster=cluster,
            src=src,
            dst=host,
            bytes_amount=tensor_bytes,
            earliest=float(start),
            commit=bool(commit),
            tag=tag,
            extra=extra,
            local_tl=local_tl,
        )
        ends.append(float(e))
    return float(max(ends))


def gather_to_host(
    *,
    comm: Any,
    cost: CostModel,
    cluster: Cluster,
    participants: Sequence[str],
    tensor_bytes: int,
    start: float,
    commit: bool,
    tag: str = 'gather',
    extra_base: Optional[Dict[str, Any]] = None,
    host_name: Optional[str] = None,
) -> float:
    """Gather: all participants send their tensors to host (root) without reduction."""
    # Communication pattern identical to reduce.
    return reduce_to_host(
        comm=comm,
        cost=cost,
        cluster=cluster,
        participants=participants,
        tensor_bytes=tensor_bytes,
        start=start,
        commit=commit,
        tag=tag,
        extra_base=extra_base,
        host_name=host_name,
    )


def scatter_from_host(
    *,
    comm: Any,
    cost: CostModel,
    cluster: Cluster,
    targets: Sequence[str],
    bytes_per_target: int,
    start: float,
    commit: bool,
    tag: str = 'scatter',
    extra_base: Optional[Dict[str, Any]] = None,
    host_name: Optional[str] = None,
) -> float:
    """Scatter (or broadcast): host sends (bytes_per_target) to each target.

    If bytes_per_target is the full tensor size, this behaves like broadcast.
    Returns completion time (end timestamp).
    """
    host = str(host_name or cost.get_host_device().name)
    targets = [str(x) for x in targets]
    bytes_per_target = int(max(0, bytes_per_target))

    local_tl: Dict[Tuple[str, str], float] = {}
    ends: List[float] = [float(start)]
    for dst in targets:
        if dst == host:
            continue
        extra = dict(extra_base or {})
        extra.update({'primitive': 'scatter', 'root': host, 'dst': dst})
        _, e = _reserve_p2p(
            comm=comm,
            cost=cost,
            cluster=cluster,
            src=host,
            dst=dst,
            bytes_amount=bytes_per_target,
            earliest=float(start),
            commit=bool(commit),
            tag=tag,
            extra=extra,
            local_tl=local_tl,
        )
        ends.append(float(e))
    return float(max(ends))


def transfer_p2p(
    *,
    comm: Any,
    cost: CostModel,
    cluster: Cluster,
    src: str,
    dst: str,
    bytes_amount: int,
    start: float,
    commit: bool,
    tag: str = 'transfer',
    extra: Optional[Dict[str, Any]] = None,
) -> float:
    """Ordinary p2p transfer completion time."""
    local_tl: Dict[Tuple[str, str], float] = {}
    _, e = _reserve_p2p(
        comm=comm,
        cost=cost,
        cluster=cluster,
        src=str(src),
        dst=str(dst),
        bytes_amount=int(bytes_amount),
        earliest=float(start),
        commit=bool(commit),
        tag=tag,
        extra=extra,
        local_tl=local_tl,
    )
    return float(e)