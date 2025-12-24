from __future__ import annotations
from config import attach_local_debug_filter
from dataclasses import dataclass, field
from typing import Dict, Set
import logging
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)

@dataclass
class PlanLabel:
    """Memory planning decision shared across modules."""
    kv_in_pim: bool
    # Human-readable tag (optional; mainly used for logging).
    pim_mode: str = "auto"
    kv_total_bytes: int = 0

    # KV cache partitioning dimension
    #   - "layer"    : kv_layer_to_pim is used
    #   - "head_num" : kv_head_to_pim is used
    kv_partition_dim: str = "layer"
    kv_bytes_per_layer: int = 0
    kv_layer_to_pim: Dict[int, str] = field(default_factory=dict)
    kv_bytes_by_pim: Dict[str, int] = field(default_factory=dict)
    kv_bytes_per_head: int = 0
    kv_head_to_pim: Dict[int, str] = field(default_factory=dict)

    pim_weight_capacity_bytes: int = 0
    pinned_fc_on_pim: Set[str] = field(default_factory=set)