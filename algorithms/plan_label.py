from __future__ import annotations
from config import attach_local_debug_filter
from dataclasses import dataclass, field
from typing import Dict, Set
import logging

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)


@dataclass
class PlanLabel:
    # Whether KV cache is stored on PIM (otherwise on host/CPU memory).
    kv_in_pim: bool

    # Human-readable tag (optional; mainly used for logging).
    pim_mode: str = "auto"

    # Total KV cache bytes reserved on PIM (0 if kv_in_pim=False).
    kv_total_bytes: int = 0

    # ---- Layer-based KV partitioning ----
    kv_bytes_per_layer: int = 0
    kv_layer_to_pim: Dict[int, str] = field(default_factory=dict)   # layer_id -> pim_name
    kv_bytes_by_pim: Dict[str, int] = field(default_factory=dict)   # pim_name -> bytes

    # ---- PIM static weight budget ----
    pim_weight_capacity_bytes: int = 0

    # Optional: pin selected FC weights on PIM (weight_id set).
    pinned_fc_on_pim: Set[str] = field(default_factory=set)
