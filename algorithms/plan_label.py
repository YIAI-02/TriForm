from __future__ import annotations
from config import attach_local_debug_filter
from dataclasses import dataclass, field
from typing import Set
import logging
logger = logging.getLogger(__name__)
attach_local_debug_filter(logger, lambda: False)

@dataclass
class PlanLabel:
    """Memory planning decision shared across modules."""
    pim_mode: str
    kv_in_pim: bool
    kv_total_bytes: int = 0
    pim_weight_capacity_bytes: int = 0
    kv_home: str = "host"
    pinned_fc_on_pim: Set[str] = field(default_factory=set)
    pim_strategy: str = "kv_first"

    def __post_init__(self) -> None:
        if self.kv_in_pim and self.kv_home == "host":
            self.kv_home = "pim"
        try:
            self.pim_strategy = (self.pim_strategy or "kv_first").lower()
        except:
            self.pim_strategy = "kv_first"

    def print_debug(self) -> None:
        """Print all PlanLabel settings for debugging."""
        logger.debug(str('=' * 50))
        logger.debug(str('PIM MEMORY PLAN DEBUG INFO'))
        logger.debug(str('=' * 50))
        logger.debug(str(f'PIM Mode: {self.pim_mode}'))
        logger.debug(str(f'PIM Strategy: {self.pim_strategy}'))
        logger.debug(str(f'KV Cache in PIM: {self.kv_in_pim}'))
        logger.debug(str(f'KV Home: {self.kv_home}'))
        logger.debug(str(f'KV Total Bytes: {self.kv_total_bytes:,}'))
        logger.debug(str(f'PIM Weight Capacity Bytes: {self.pim_weight_capacity_bytes:,}'))
        logger.debug(str(f'Number of Pinned FC Weights: {len(self.pinned_fc_on_pim)}'))
        if self.pinned_fc_on_pim:
            logger.debug(str('Pinned FC Weights:'))
            for weight_id in sorted(self.pinned_fc_on_pim):
                logger.debug(str(f'  - {weight_id}'))
        else:
            logger.debug(str('Pinned FC Weights: None'))
        logger.debug(str('=' * 50))