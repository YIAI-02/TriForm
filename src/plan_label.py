from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Literal

from config import attach_local_debug_filter

logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)
KVPlace = Literal["host", "pim", "npu"]
@dataclass
class PlanLabel:
    # Whether KV cache is stored on PIM (otherwise on host/CPU memory).
    kv_in_pim: bool

    # Human-readable tag (optional; mainly used for logging).
    pim_mode: str = "auto"

    # Total KV cache bytes reserved on PIM (0 if kv_in_pim=False).
    kv_total_bytes: int = 0


    # ---- KV-head-based KV partitioning (preferred for TP) ----
    # kv_head_id -> pim_name
    kv_head_to_pim: Dict[int, str] = field(default_factory=dict)
    # pim_name -> list[kv_head_id]
    kv_heads_by_pim: Dict[str, List[int]] = field(default_factory=dict)
    # DeepSeek-V4 / shared-KV architectures may have only one KV head, so
    # head-based sharding cannot use multiple PIMs.  In that case we can place
    # KV by layer instead: layer_id -> pim_name and pim_name -> layer_ids.
    kv_layer_to_pim: Dict[int, str] = field(default_factory=dict)
    kv_layers_by_pim: Dict[str, List[int]] = field(default_factory=dict)
    # DeepSeek-V4/shared-KV can instead shard the context/KV-length axis.
    # kv_seq_shard_id -> pim_name and inverse map.  This creates true same-layer
    # QK/Softmax/SV concurrency, unlike layer round-robin placement.
    kv_seq_shard_to_pim: Dict[int, str] = field(default_factory=dict)
    kv_seq_shards_by_pim: Dict[str, List[int]] = field(default_factory=dict)
    kv_bytes_by_pim: Dict[str, int] = field(default_factory=dict)
    # Optional: which dimension is used for KV partitioning ('seq', 'layer', or 'kv_head').
    kv_partition_dim: str = "kv_head"

    # ---- PIM static weight budget ----
    pim_weight_capacity_bytes: int = 0

    kv_place: KVPlace = "host"
    kv_in_npu: bool = False

    kv_npu_device: Optional[str] = None

    kv_total_bytes_all: int = 0
    kv_total_bytes_on_pim: int = 0
    kv_total_bytes_on_npu: int = 0
    kv_total_bytes_on_host: int = 0

    trace_ops_csv: Optional[str] = None
    trace_comms_csv: Optional[str] = None