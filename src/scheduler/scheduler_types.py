from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class _GraphIndex:
    nodes: tuple
    nodes_set: frozenset
    preds: dict
    succs: dict
    topo: tuple
    rev_topo: tuple
    rank_u_by_phase: dict = field(default_factory=dict)      # phase -> {nid: upward_rank}
    allowed_actions: dict = field(default_factory=dict)       # (phase, label_kv_sig, nid) -> tuple[action...]

@dataclass
class ScheduledTask:
    node_id: str
    device: str
    start: float
    finish: float

